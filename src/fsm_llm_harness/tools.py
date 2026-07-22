"""
Root-confined filesystem and shell actions for the harness's role workers.

The harness drives real work with a 4B model.  Unconstrained, that is arbitrary
filesystem and shell access driven by a small model's output, so every action a
role worker can take goes through :class:`Workspace`, whose :meth:`Workspace.resolve`
is the single confinement chokepoint (invariant I9): a path that does not
resolve *under* the root is rejected **before any I/O**.

There are exactly **two roots**, and a path resolves under exactly one of them:

* the **workspace** -- the code the protocol is changing.  Only the EXECUTE
  role receives its write tools.
* the **plan directory** -- the protocol's own filesystem-as-memory, reached
  through :class:`PlanMemory`.  Reads are confined; writes are confined *and*
  narrowed to the artifacts ``rules.OWNERSHIP`` grants the calling role, so a
  role that must not write an artifact does not hold a tool that can
  (invariant I7).  :class:`PlanMemory` **composes** :class:`Workspace` rather
  than re-deriving containment -- a second confinement implementation is a
  named Complexity-Budget BREACH.

``run_command`` exists but is **disabled by default** (decisions.md D-008).
Enabling it is a deliberate act by the harness's caller -- never by the LLM --
and even then it runs ``shell=False`` against an executable allowlist, in the
workspace root, under a timeout, with a minimal environment.

Composition, not reimplementation:

* :func:`build_workspace_tools` registers the confined operations through the
  existing ``@tool`` decorator and ``ToolRegistry``
  (``fsm_llm_agents/tools.py:46,542``), so every action carries a JSON schema
  for LLM function-calling and every failure is converted to a failed
  ``ToolResult`` by ``ToolRegistry.execute``.  This module writes **no
  tool-dispatch mechanism** -- that is a named Complexity-Budget BREACH.
* The confinement idiom mirrors ``FileSessionStore._path``
  (``fsm_llm/session.py:135-142``): validate, then reject, before touching
  anything.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType

from fsm_llm.logging import logger
from fsm_llm_agents.definitions import ToolResult
from fsm_llm_agents.tools import ToolRegistry, tool

from .constants import ArtifactNames, ContextKeys, Defaults
from .exceptions import (
    HarnessConfinementError,
    HarnessError,
    HarnessOwnershipError,
)
from .rules import OWNERSHIP

__all__ = [
    "COMMAND_ALLOWLIST",
    "DISK_DERIVED_COUNTS",
    "PLAN_READ_TOOLS",
    "PLAN_WRITE_TOOLS",
    "READ_ONLY_TOOLS",
    "SHELL_TOOLS",
    "VERIFICATION_COMMANDS",
    "WRITE_TOOLS",
    "PlanMemory",
    "PlanTools",
    "Workspace",
    "WorkspaceTools",
    "build_plan_tools",
    "build_workspace_tools",
    "count_gate_files",
    "derive_disk_counts",
    "gate_files",
    "has_bytes",
]


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------
#
# Every bound below exists for the same reason: a role worker's tool output is
# fed straight back into a 4B model's context window.  An unbounded read of a
# 500MB file does not fail loudly -- it silently destroys the run.  Exceeding a
# bound TRUNCATES with an explicit marker rather than raising, so the model
# still gets usable information and can narrow its next request.

#: Longest file read returned to a worker, in bytes.
MAX_READ_BYTES = 64_000

#: Longest command stdout/stderr capture returned to a worker, in bytes.
MAX_OUTPUT_BYTES = 8_000

#: Most directory entries returned by ``list_dir``.
MAX_LIST_ENTRIES = 200

#: Most matches returned by ``grep``.
MAX_GREP_HITS = 50

#: Files larger than this are skipped by ``grep`` (they are not source).
MAX_GREP_FILE_BYTES = 1_000_000

#: Most files ``grep`` will open in one call.
MAX_GREP_FILES = 2_000

#: Default wall-clock budget for one ``run_command`` invocation, in seconds.
DEFAULT_COMMAND_TIMEOUT = 30.0

#: Appended wherever a bound above truncated the payload.
TRUNCATION_MARKER = "\n... [truncated: {omitted} more {unit}]"

#: Directory names ``grep`` never descends into.
_GREP_SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
        ".tox",
    }
)

#: Executables ``run_command`` permits when shell access is explicitly enabled.
#:
#: Every entry here READS bytes and executes nothing that lives inside the
#: workspace.  General-purpose interpreters (``python``, ``sh``, ``bash``,
#: ``node``, ``perl``) are excluded for the obvious reason -- one of them on
#: this list runs anything -- and so is every tool that loads a *config or
#: plugin file authored by the EXECUTE role*; those live in
#: :data:`VERIFICATION_COMMANDS` and must be opted into by name.
COMMAND_ALLOWLIST: tuple[str, ...] = (
    "cat",
    "grep",
    "head",
    "ls",
    "tail",
    "wc",
)

# DECISION plan-2026-07-21T125237-191b2eb2/D-050
# These five are NOT in the default allowlist and must not be "restored" to it
# for convenience. Each one executes code that the EXECUTE role can write into
# the workspace, which is exactly the property the allowlist above claims to
# have: `make` runs a Makefile, `pytest` imports `conftest.py`, `git` honours
# `.git/hooks/*` and `[alias]`/`core.pager` in a repo-local config, `mypy`
# imports whatever `plugins =` names in its config, and `ruff` is here only
# because it travels with the other four in a verification profile. Handing
# them to a 4B model by default makes the allowlist decorative (review W5).
# A caller who wants a verifying REFLECT role opts in explicitly:
#   Workspace(root, allow_shell=True,
#             allowed_commands=COMMAND_ALLOWLIST + VERIFICATION_COMMANDS)
# `run_command` stays disabled by default either way.
# See decisions.md D-050.
#: Verification entry points, excluded from the default allowlist because each
#: one executes workspace-authored code.  Opt in by name.
VERIFICATION_COMMANDS: tuple[str, ...] = (
    "git",
    "make",
    "mypy",
    "pytest",
    "ruff",
)

#: Characters that may never appear in a workspace-relative path.
#:
#: NUL and newline are the two a small model actually emits (a NUL through a
#: mangled decode, a newline by pasting a path out of a listing).  Both are
#: rejected together with every other C0 control byte and DEL.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")

#: Leading components a small model prepends when it emits a filesystem-absolute
#: path where a WORKSPACE-relative one was asked for (``/workspace/uploader.py``).
#: The root's own basename is added per instance by :meth:`Workspace.resolve`.
_WORKSPACE_SENTINELS: frozenset[str] = frozenset({"workspace"})

#: The same, for the PLAN directory (``/plan/state.md``).  ``workspace`` is
#: here as well because a model that reaches for the plan tool with a workspace
#: path still lands on an ownership refusal, which is the legible failure; the
#: reverse is NOT true, which is why the two sets are not one (see the D-006
#: block in :meth:`Workspace.resolve`).  :meth:`PlanMemory.locate` adds the
#: plan id and the memory root's basename per instance.
_PLAN_SENTINELS: frozenset[str] = frozenset({"plan", "workspace"})


class WorkspaceTools:
    """Tool names registered by :func:`build_workspace_tools`.

    Read as data, not as an abstraction: these are the string ids a
    ``RoleSpec.tool_scope`` selects from, defined once so roles.py cannot
    misspell one into a silently missing capability.
    """

    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    APPEND_FILE = "append_file"
    DELETE_FILE = "delete_file"
    LIST_DIR = "list_dir"
    PATH_EXISTS = "path_exists"
    GREP_FILES = "grep_files"
    RUN_COMMAND = "run_command"


#: Inspection only.  Every role gets at least these -- a role that can read
#: nothing cannot do its job, and ``ReactAgent`` refuses an empty registry.
READ_ONLY_TOOLS: tuple[str, ...] = (
    WorkspaceTools.READ_FILE,
    WorkspaceTools.LIST_DIR,
    WorkspaceTools.PATH_EXISTS,
    WorkspaceTools.GREP_FILES,
)

#: Mutation.  Invariant I7 in tool form: only the EXECUTE role receives these.
WRITE_TOOLS: tuple[str, ...] = (
    WorkspaceTools.WRITE_FILE,
    WorkspaceTools.APPEND_FILE,
    WorkspaceTools.DELETE_FILE,
)

#: Subprocess execution.  Inert unless the workspace was built with
#: ``allow_shell=True``; the tool is still registered so the refusal is a
#: legible failed ``ToolResult`` rather than an unknown-tool error.
SHELL_TOOLS: tuple[str, ...] = (WorkspaceTools.RUN_COMMAND,)


class PlanTools:
    """Tool names registered by :func:`build_plan_tools`.

    Deliberately distinct from every :class:`WorkspaceTools` name: the plan
    directory and the workspace are two roots, and a role must never be able to
    reach one through the other's tool.
    """

    READ_PLAN_FILE = "read_plan_file"
    WRITE_PLAN_FILE = "write_plan_file"
    APPEND_PLAN_FILE = "append_plan_file"
    LIST_PLAN_DIR = "list_plan_dir"
    PLAN_PATH_EXISTS = "plan_path_exists"


#: Protocol-memory inspection.  Every role gets these: the protocol's first
#: operative rule in almost every state is "read the artifacts before acting".
PLAN_READ_TOOLS: tuple[str, ...] = (
    PlanTools.READ_PLAN_FILE,
    PlanTools.LIST_PLAN_DIR,
    PlanTools.PLAN_PATH_EXISTS,
)

#: Protocol-memory mutation.  Handed only to roles that own at least one
#: artifact in ``rules.OWNERSHIP``; the tools then refuse every artifact the
#: calling role does not own (:class:`PlanMemory`).
PLAN_WRITE_TOOLS: tuple[str, ...] = (
    PlanTools.WRITE_PLAN_FILE,
    PlanTools.APPEND_PLAN_FILE,
)

#: Cross-plan artifacts, which live one level ABOVE the plan directory.
_CROSS_PLAN_FILES: frozenset[str] = frozenset(
    (*ArtifactNames.CROSS_PLAN, ArtifactNames.LESSONS_ARCHIVE)
)

#: Per-plan files a role may be granted; the two subdirectories are matched by
#: their first path component instead.
_PER_PLAN_FILES: frozenset[str] = frozenset(
    (*ArtifactNames.PER_PLAN, ArtifactNames.SUMMARY)
)

#: Per-plan subdirectories owned as a whole (``findings/``, ``checkpoints/``).
_PER_PLAN_DIRS: frozenset[str] = frozenset(
    (ArtifactNames.FINDINGS_DIR, ArtifactNames.CHECKPOINTS_DIR)
)

#: Gate keys the protocol DERIVES from the filesystem instead of believing:
#: context key -> (the plan-directory subdirectory whose non-empty ``.md`` files
#: are counted, the threshold that state's exit gate applies to the count).
#:
#: This table lives HERE, one import below the confinement chokepoint, because
#: two very different callers must read the same fact: ``roles.py``'s worker
#: factory, which replaces the worker's self-reported integer with the derived
#: one (D-015), and :func:`build_plan_tools`, which reports the derived one back
#: to the model in the write tool's own result (D-027).  A second table would be
#: a gate and a progress report that can disagree.
DISK_DERIVED_COUNTS: Mapping[str, tuple[str, int]] = MappingProxyType(
    {
        ContextKeys.FINDINGS_COUNT: (
            ArtifactNames.FINDINGS_DIR,
            Defaults.FINDINGS_THRESHOLD,
        )
    }
)

#: A tool -> the tool performing the SAME operation in the OTHER confined root.
#:
#: Used only to make a FAILED call's message corrective (D-027 part B); nothing
#: is ever re-routed through it.  ``delete_file``, ``grep_files`` and
#: ``run_command`` are absent because they have no counterpart: naming a tool
#: that does not exist is worse than saying nothing.
_COUNTERPART_TOOL: Mapping[str, str] = MappingProxyType(
    {
        WorkspaceTools.READ_FILE: PlanTools.READ_PLAN_FILE,
        WorkspaceTools.WRITE_FILE: PlanTools.WRITE_PLAN_FILE,
        WorkspaceTools.APPEND_FILE: PlanTools.APPEND_PLAN_FILE,
        WorkspaceTools.LIST_DIR: PlanTools.LIST_PLAN_DIR,
        WorkspaceTools.PATH_EXISTS: PlanTools.PLAN_PATH_EXISTS,
        PlanTools.READ_PLAN_FILE: WorkspaceTools.READ_FILE,
        PlanTools.WRITE_PLAN_FILE: WorkspaceTools.WRITE_FILE,
        PlanTools.APPEND_PLAN_FILE: WorkspaceTools.APPEND_FILE,
        PlanTools.LIST_PLAN_DIR: WorkspaceTools.LIST_DIR,
        PlanTools.PLAN_PATH_EXISTS: WorkspaceTools.PATH_EXISTS,
    }
)

#: What a write did to a target that ALREADY existed, per action.
_REPEAT_PHRASE: Mapping[str, str] = MappingProxyType(
    {
        "wrote": "OVERWROTE an existing file",
        "appended to": "extended an existing file",
    }
)

#: A leading path component that is a plan directory's own id.
_PLAN_ID_RE = re.compile(r"^plan[-_]\d{4}-\d{2}-\d{2}")


def _truncate(text: str, limit: int, unit: str = "characters") -> str:
    """Return *text* bounded to *limit*, marking what was dropped."""
    if len(text) <= limit:
        return text
    return text[:limit] + TRUNCATION_MARKER.format(omitted=len(text) - limit, unit=unit)


def _strip_root_sentinel(candidate: str, sentinels: Iterable[str]) -> str | None:
    """Rewrite an absolute *candidate* as root-relative, or refuse it.

    Interface contract (2 call sites: :meth:`Workspace.resolve` and
    :meth:`PlanMemory.locate`):
        - Parameters: a filesystem-absolute path, and the leading component
          names that stand for "the root" in a model's output.
        - Returns the remainder with the anchor **and exactly one** leading
          sentinel component removed.  The rewrite is purely LEXICAL: the
          caller still has to resolve and compare the result.
        - Returns ``None`` -- meaning refuse, never guess -- when the first
          component is not a sentinel or nothing is left after dropping it.
        - Performs no I/O and never raises.
    """
    parts = Path(candidate).parts[1:]
    if not parts or parts[0] not in sentinels:
        return None
    rest = parts[1:]
    if not rest:
        return None
    return str(Path(*rest))


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------


class Workspace:
    """A filesystem root that harness role workers cannot escape.

    Every operation resolves its argument through :meth:`resolve` first, so the
    class has exactly one confinement decision point rather than one per
    method.

    Args:
        root: Directory the workspace is confined to.  Created if absent, then
            resolved once so a symlinked root compares consistently.
        allow_shell: Opt-in for :meth:`run_command`.  Default ``False``: the
            out-of-the-box configuration cannot execute anything at all.
        allowed_commands: Executable basenames :meth:`run_command` permits when
            *allow_shell* is true.
        command_timeout: Wall-clock budget for one command, in seconds.
        max_read_bytes: Cap on :meth:`read_text` output.
        max_output_bytes: Cap on captured stdout/stderr.

    Example::

        ws = Workspace("/tmp/scratch")
        ws.write_text("notes/todo.md", "- ship it\\n")
        ws.read_text("notes/todo.md")
        ws.resolve("../../etc/passwd")   # HarnessConfinementError
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        allow_shell: bool = False,
        allowed_commands: Sequence[str] = COMMAND_ALLOWLIST,
        command_timeout: float = DEFAULT_COMMAND_TIMEOUT,
        max_read_bytes: int = MAX_READ_BYTES,
        max_output_bytes: int = MAX_OUTPUT_BYTES,
    ) -> None:
        path = Path(root).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        # Resolve the root ONCE, after creating it: every containment test
        # below compares two fully-resolved paths, so a symlinked root (macOS
        # /tmp, a bind-mounted checkout) can never make a legitimate path look
        # like an escape.
        self._root = path.resolve()
        self._allow_shell = bool(allow_shell)
        self._allowed_commands = tuple(allowed_commands)
        self._command_timeout = float(command_timeout)
        self._max_read_bytes = int(max_read_bytes)
        self._max_output_bytes = int(max_output_bytes)

    # -- properties -----------------------------------------------------

    @property
    def root(self) -> Path:
        """The resolved directory every path must live under."""
        return self._root

    @property
    def allow_shell(self) -> bool:
        """Whether :meth:`run_command` is permitted to execute anything."""
        return self._allow_shell

    @property
    def allowed_commands(self) -> tuple[str, ...]:
        """Executable basenames :meth:`run_command` accepts."""
        return self._allowed_commands

    def __repr__(self) -> str:
        return f"Workspace(root={str(self._root)!r}, allow_shell={self._allow_shell})"

    # -- confinement ----------------------------------------------------

    def resolve(self, relative_path: str) -> Path:
        """Resolve *relative_path* under the root, or refuse.

        Interface contract (the confinement chokepoint -- every other method
        goes through it):
            - Parameter: a workspace-relative path as a ``str``.
            - Returns the fully resolved absolute :class:`~pathlib.Path`.
            - Raises :class:`~fsm_llm_harness.exceptions.HarnessConfinementError`
              for anything that does not land strictly inside the root, and
              :class:`TypeError` for a non-``str`` argument.
            - Performs no read, write, create or delete of the target.

        Raises:
            HarnessConfinementError: For an absolute path, a ``..`` escape, a
                symlink whose target is outside the root, a path containing a
                control character, or an empty path.
        """
        # DECISION plan-2026-07-21T125237-191b2eb2/D-032
        # The order here is load-bearing and must not be "simplified":
        # RESOLVE FIRST, COMPARE SECOND. Three simplifications reopen a
        # confinement hole, and all three look tidier than this code:
        #   1. Do NOT compare before resolving (`str(root / p).startswith(root)`
        #      or a `".." not in parts` check). A symlink CREATED INSIDE the
        #      workspace pointing at /etc passes both of those tests and then
        #      reads /etc/passwd. Only `Path.resolve()` collapses the link.
        #   2. Do NOT use `os.path.normpath` / `Path.parents` alone. normpath is
        #      purely lexical: it folds `a/../b` without ever asking whether `a`
        #      is a symlink, so `link/../../etc` normalises to something that
        #      looks contained and is not.
        #   3. Do NOT use `str.startswith(str(root))` for the containment test.
        #      `/tmp/ws-evil` startswith `/tmp/ws`. The comparison must be on
        #      path COMPONENTS -- `resolved == root or root in resolved.parents`.
        # The residual hole is TOCTOU: a symlink swapped between this resolve
        # and the caller's open still wins. Closing that needs openat2/O_NOFOLLOW
        # and is not what this guard claims to do -- it stops a small model's
        # path arguments, not a hostile local process.
        # See decisions.md D-032.
        if not isinstance(relative_path, str):
            raise TypeError(
                f"workspace path must be a str, got {type(relative_path).__name__}"
            )

        candidate = relative_path.strip()
        if not candidate:
            raise HarnessConfinementError("", str(self._root))
        if _CONTROL_CHARS_RE.search(candidate):
            raise HarnessConfinementError(relative_path, str(self._root))
        if Path(candidate).is_absolute():
            # DECISION plan-2026-07-21T191807-bf7ffe24/D-006
            # `:4b` emits `/workspace/uploader.py` when it means `uploader.py`,
            # and the flat refusal that used to stand here cost 2 of 3 measured
            # writes. Two properties of this repair must NOT be "simplified":
            #   1. Do NOT just drop the leading `/` and resolve what is left.
            #      That reinterprets `/etc/passwd` as `<root>/etc/passwd` --
            #      confined, but a confinement layer that INVENTS a target is a
            #      path-rewriter. Only a leading SENTINEL component is dropped;
            #      every other absolute path is refused exactly as before, which
            #      is what keeps the escape parametrisation green.
            #   2. Strip the sentinel and THEN fall through to D-032's
            #      resolve-and-compare -- never resolve a repaired path here and
            #      trust it. `/workspace/../../etc/passwd` strips to
            #      `../../etc/passwd`, and only the untouched compare below sees
            #      that it climbs out.
            # An absolute path that ALREADY lands inside the root needs no
            # repair: `root / <absolute>` is that absolute path, so it reaches
            # the same compare unchanged.
            #   3. `plan` is NOT a sentinel HERE, and must not be "restored for
            #      symmetry" with `PlanMemory.locate`. It was, for one commit,
            #      and review W4 measured the consequence: `/plan/findings/x.md`
            #      resolved to `<workspace>/findings/x.md`, so a role emitting a
            #      protocol path into a workspace tool wrote a protocol artifact
            #      into the USER'S SOURCE TREE -- confined, but into the wrong
            #      root and past every ownership check, where before the repair
            #      it simply raised. The two roots are two vocabularies: `plan`
            #      means the plan directory and nothing in this class can
            #      address it, so refusing is the only honest answer.
            # See decisions.md D-006.
            absolute = Path(candidate).resolve()
            if absolute != self._root and self._root not in absolute.parents:
                repaired = _strip_root_sentinel(
                    candidate, (self._root.name, *_WORKSPACE_SENTINELS)
                )
                if repaired is None:
                    raise HarnessConfinementError(relative_path, str(self._root))
                candidate = repaired

        resolved = (self._root / candidate).resolve()
        if resolved != self._root and self._root not in resolved.parents:
            raise HarnessConfinementError(relative_path, str(self._root))
        return resolved

    def relative(self, path: Path) -> str:
        """Render a resolved path back as a workspace-relative string."""
        try:
            return str(path.relative_to(self._root)) or "."
        except ValueError:  # pragma: no cover - unreachable via resolve()
            return str(path)

    # -- reads ----------------------------------------------------------

    def read_text(self, path: str) -> str:
        """Read a UTF-8 text file, bounded by ``max_read_bytes``."""
        target = self.resolve(path)
        data = target.read_bytes()
        text = data.decode("utf-8", errors="replace")
        return _truncate(text, self._max_read_bytes, "bytes")

    def exists(self, path: str) -> bool:
        """Whether a confined path exists."""
        return self.resolve(path).exists()

    def list_dir(self, path: str = ".") -> list[str]:
        """List a directory's entries, bounded by ``MAX_LIST_ENTRIES``.

        Directories are suffixed with ``/`` so a worker can tell them apart
        without a second call.
        """
        target = self.resolve(path)
        entries = sorted(target.iterdir(), key=lambda p: p.name)
        names = [f"{p.name}/" if p.is_dir() else p.name for p in entries]
        if len(names) > MAX_LIST_ENTRIES:
            omitted = len(names) - MAX_LIST_ENTRIES
            names = names[:MAX_LIST_ENTRIES]
            names.append(f"... [truncated: {omitted} more entries]")
        return names

    def grep(
        self,
        pattern: str,
        path: str = ".",
        *,
        max_hits: int = MAX_GREP_HITS,
    ) -> list[str]:
        """Search text files under *path* for a regular expression.

        Returns ``"<relpath>:<lineno>: <line>"`` strings, at most *max_hits* of
        them, followed by a truncation marker when the search was cut short.
        Binary files, oversized files and VCS/build directories are skipped.

        Raises:
            HarnessError: If *pattern* is not a valid regular expression.
        """
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            raise HarnessError(f"Invalid grep pattern {pattern!r}: {exc}") from exc

        target = self.resolve(path)
        hits: list[str] = []
        files_seen = 0
        truncated = False

        for candidate in self._walk_files(target):
            files_seen += 1
            if files_seen > MAX_GREP_FILES:
                truncated = True
                break
            try:
                if candidate.stat().st_size > MAX_GREP_FILE_BYTES:
                    continue
                text = candidate.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue  # unreadable or binary: not a source file
            for lineno, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    rel = self.relative(candidate)
                    hits.append(f"{rel}:{lineno}: {_truncate(line.strip(), 200)}")
                    if len(hits) >= max_hits:
                        truncated = True
                        break
            if truncated:
                break

        if truncated:
            hits.append(f"... [truncated: search stopped at {len(hits)} matches]")
        return hits

    def _walk_files(self, target: Path) -> Iterable[Path]:
        """Yield regular files under *target*, skipping VCS/build directories."""
        if target.is_file():
            yield target
            return
        for dirpath, dirnames, filenames in os.walk(target):
            dirnames[:] = sorted(d for d in dirnames if d not in _GREP_SKIP_DIRS)
            base = Path(dirpath)
            for name in sorted(filenames):
                candidate = base / name
                if candidate.is_symlink():
                    # A symlink met during the walk is NOT resolved-and-checked
                    # here; skipping is cheaper than re-deriving containment and
                    # a workspace's own files are always reachable directly.
                    continue
                yield candidate

    # -- writes ---------------------------------------------------------

    def write_text(self, path: str, content: str) -> str:
        """Write a UTF-8 text file, creating parent directories inside the root."""
        target = self.resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        rel = self.relative(target)
        logger.debug(f"workspace wrote {rel} ({len(content)} chars)")
        return rel

    def append_text(self, path: str, content: str) -> str:
        """Append to a UTF-8 text file, creating it (and its parents) if absent."""
        target = self.resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(content)
        return self.relative(target)

    def delete(self, path: str) -> str:
        """Delete a single confined FILE.

        Directories are refused on purpose: the revert-first protocol needs to
        undo a file it created, not to prune a tree, and ``rmtree`` driven by a
        4B model's path argument is the one confined operation whose blast
        radius is unbounded.

        Raises:
            HarnessError: If the target is a directory.
            FileNotFoundError: If the target does not exist.
        """
        target = self.resolve(path)
        if target.is_dir():
            raise HarnessError(
                f"Refusing to delete directory '{self.relative(target)}': "
                "delete only removes single files"
            )
        target.unlink()
        return self.relative(target)

    # -- subprocess -----------------------------------------------------

    def run_command(
        self,
        argv: Sequence[str],
        *,
        timeout: float | None = None,
    ) -> ToolResult:
        """Run an allowlisted executable inside the workspace root.

        Interface contract:
            - ``argv``: a non-empty argument VECTOR.  ``argv[0]`` must be a bare
              executable name present in :attr:`allowed_commands`.
            - Returns a ``ToolResult``.  A non-zero exit, a timeout or a missing
              executable is ``success=False`` -- **not** an exception -- so an
              agent loop can read the failure and adapt.
            - Raises :class:`~fsm_llm_harness.exceptions.HarnessError` only for
              the two POLICY refusals: shell access disabled, and a
              non-allowlisted executable.  Registered as a tool, those become
              failed ``ToolResult``s too (``ToolRegistry.execute`` converts
              them), so the LLM path never sees a raw exception either.

        Raises:
            HarnessError: If shell access is disabled or the executable is not
                allowlisted.
        """
        if not self._allow_shell:
            raise HarnessError(
                "run_command is disabled. Shell access is off by default; "
                "the harness caller (never the model) enables it with "
                "Workspace(root, allow_shell=True)."
            )

        parts = [str(part) for part in argv]
        if not parts or not parts[0].strip():
            raise HarnessError("run_command requires a non-empty argument vector")

        executable = parts[0]
        if "/" in executable or "\\" in executable:
            raise HarnessError(
                f"run_command takes a bare executable name, got {executable!r}"
            )
        if executable not in self._allowed_commands:
            raise HarnessError(
                f"Executable {executable!r} is not allowlisted. Allowed: "
                f"{', '.join(self._allowed_commands)}"
            )

        env = self._command_env()
        resolved_exe = shutil.which(executable, path=env["PATH"])
        if resolved_exe is None:
            return ToolResult(
                tool_name=WorkspaceTools.RUN_COMMAND,
                success=False,
                error=f"Executable {executable!r} not found on PATH",
            )

        budget = self._command_timeout if timeout is None else float(timeout)
        try:
            completed = subprocess.run(
                [resolved_exe, *parts[1:]],
                shell=False,
                cwd=str(self._root),
                env=env,
                capture_output=True,
                text=True,
                timeout=budget,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                tool_name=WorkspaceTools.RUN_COMMAND,
                success=False,
                error=f"Command timed out after {budget:g}s: {' '.join(parts)}",
            )
        except OSError as exc:
            return ToolResult(
                tool_name=WorkspaceTools.RUN_COMMAND,
                success=False,
                error=f"Command failed to start: {exc}",
            )

        stdout = _truncate(completed.stdout or "", self._max_output_bytes, "bytes")
        stderr = _truncate(completed.stderr or "", self._max_output_bytes, "bytes")
        payload = f"exit={completed.returncode}\nstdout:\n{stdout}\nstderr:\n{stderr}"
        if completed.returncode == 0:
            return ToolResult(
                tool_name=WorkspaceTools.RUN_COMMAND,
                success=True,
                result=payload,
            )
        return ToolResult(
            tool_name=WorkspaceTools.RUN_COMMAND,
            success=False,
            error=payload,
        )

    def _command_env(self) -> dict[str, str]:
        """Build the minimal environment a subprocess inherits.

        Nothing else from ``os.environ`` crosses the boundary: an API key, a
        token or a proxy setting in the parent process is not the workspace's
        to hand to a model-chosen command.
        """
        return {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "HOME": str(self._root),
            "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
        }


# ---------------------------------------------------------------------------
# Plan directory (filesystem-as-memory), ownership-scoped
# ---------------------------------------------------------------------------


class PlanMemory:
    """The protocol's filesystem-as-memory tier, scoped to ONE role.

    Reads are confined; writes are confined **and** checked against
    ``rules.OWNERSHIP``, so a role that does not own an artifact is refused by
    the tool rather than asked not to call it (invariant I7).

    Args:
        plan_dir: This run's plan directory.  Created if absent.
        role: The calling role, a ``Role.WORKERS`` member.  Fixed for the
            lifetime of the object -- one ``PlanMemory`` per dispatch.

    Example::

        memory = PlanMemory("plans/plan-2026-07-21T125237-191b2eb2",
                            role=Role.EXPLORER)
        memory.write_text("findings/tool-scope.md", "...")   # owned
        memory.write_text("plan.md", "...")                  # HarnessOwnershipError
    """

    # DECISION plan-2026-07-21T125237-191b2eb2/D-047
    # The confinement root is the plan directory's PARENT, not the plan
    # directory. Do NOT "tighten" it to the plan dir itself: the protocol's
    # cross-plan tier (LESSONS.md, SYSTEM.md, INDEX.md, FINDINGS.md,
    # DECISIONS.md, LESSONS-archive.md) lives BESIDE the plan directories, and
    # `rules.OWNERSHIP` grants all six to the archivist. Rooting at the plan dir
    # would leave those writes reachable only through `../`, i.e. only by
    # weakening confinement -- which is the named breach this design exists to
    # avoid. The narrowing that replaces it is OWNERSHIP: the wider root buys
    # nothing, because every write is classified to an artifact name first and
    # an unclassifiable path is refused outright.
    # And do NOT write a second confinement implementation here. This class
    # COMPOSES `Workspace`; `Workspace.resolve` stays the single chokepoint, so
    # the resolve-before-compare rule of D-032 protects the plan directory for
    # free. See decisions.md D-047.

    def __init__(self, plan_dir: str | os.PathLike[str], *, role: str) -> None:
        path = Path(plan_dir).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        resolved = path.resolve()
        self._workspace = Workspace(resolved.parent)
        self._plan_id = resolved.name
        self._role = role

    # -- properties -----------------------------------------------------

    @property
    def role(self) -> str:
        """The role every write is authorised against."""
        return self._role

    @property
    def plan_id(self) -> str:
        """The plan directory's name, which is also its id."""
        return self._plan_id

    @property
    def plan_dir(self) -> Path:
        """The resolved plan directory."""
        return self._workspace.root / self._plan_id

    @property
    def root(self) -> Path:
        """The resolved memory root: the plan directory's parent."""
        return self._workspace.root

    def __repr__(self) -> str:
        return f"PlanMemory(plan_dir={str(self.plan_dir)!r}, role={self._role!r})"

    # -- addressing -----------------------------------------------------

    def locate(self, path: str) -> str:
        """Map a caller-supplied path to a memory-root-relative one.

        Interface contract (2+ call sites: every read and write below):
            - A bare cross-plan filename (``LESSONS.md``) addresses the
              cross-plan tier at the memory root.
            - A path already starting with this plan's id is taken as-is.
            - Everything else is relative to the plan directory, which is what
              the operative rules say (``findings/<topic>.md``).
            - Returns a string; performs no I/O and never raises.
        """
        # An ABSOLUTE path is passed through UNCHANGED so that
        # `Workspace.resolve` refuses it. Prefixing it with the plan id would
        # turn "/etc/passwd" into the relative "<plan-id>/etc/passwd" -- still
        # confined, but the caller gets a bewildering FileNotFoundError deep
        # inside the plan directory instead of the confinement refusal the
        # attempt earned. Found by the step-7b probe; do not "tidy" the branch
        # away.
        candidate = path.strip()
        if not candidate:
            return candidate
        if Path(candidate).is_absolute():
            # DECISION plan-2026-07-21T191807-bf7ffe24/D-006
            # The plan-memory half of the same repair. `:4b` emits
            # `/plan/state.md`, and this method -- not `Workspace.resolve` --
            # is where that has to be caught, because the composed workspace is
            # rooted at the plan directory's PARENT: a sentinel stripped down
            # there lands `state.md` at the MEMORY ROOT, where it is not a
            # protocol artifact and dies as an ownership error instead. Strip
            # here and the path re-enters the normal relative branch below,
            # which prefixes the plan id.
            # Do NOT widen this to every absolute path -- when the sentinel does
            # not match we return the candidate UNCHANGED, on purpose, so the
            # comment above still holds and `Workspace.resolve` issues the
            # refusal. See decisions.md D-006.
            repaired = _strip_root_sentinel(
                candidate,
                (self._plan_id, self._workspace.root.name, *_PLAN_SENTINELS),
            )
            if repaired is None:
                return candidate
            candidate = repaired
        parts = Path(candidate).parts
        head = parts[0] if parts else ""
        if head in _CROSS_PLAN_FILES or head == self._plan_id:
            return candidate
        return f"{self._plan_id}/{candidate}"

    def artifact_for(self, path: str) -> str | None:
        """Return the ``ArtifactNames`` id *path* addresses, or ``None``.

        ``None`` means "not a protocol artifact", which is never writable --
        the ownership table is the whole of what may be written.

        Raises:
            HarnessConfinementError: If the path escapes the memory root.
        """
        return self._classify(path)[1]

    def _classify(self, path: str) -> tuple[Path, str | None]:
        """Resolve *path* and name the artifact it addresses."""
        target = self._workspace.resolve(self.locate(path))
        parts = Path(self._workspace.relative(target)).parts
        if not parts:
            return target, None  # the memory root itself
        if len(parts) == 1:
            return target, parts[0] if parts[0] in _CROSS_PLAN_FILES else None
        if parts[0] != self._plan_id:
            # Another plan's directory. Readable (the sliding window needs it);
            # never writable.
            return target, None
        rest = parts[1:]
        if rest[0] in _PER_PLAN_DIRS:
            return target, rest[0]
        if len(rest) == 1 and rest[0] in _PER_PLAN_FILES:
            return target, rest[0]
        return target, None

    def authorise(self, path: str) -> Path:
        """Resolve *path* for writing, or refuse.

        Interface contract (2 call sites: :meth:`write_text`,
        :meth:`append_text`):
            - Returns the resolved absolute path when ``OWNERSHIP`` lists this
              memory's role as a writer of the addressed artifact.
            - Performs no write.

        Raises:
            HarnessConfinementError: If the path escapes the memory root.
            HarnessOwnershipError: If the role does not own the artifact, or
                the path is not a protocol artifact at all.
        """
        target, artifact = self._classify(path)
        owners = OWNERSHIP.get(artifact, ()) if artifact is not None else ()
        if self._role not in owners:
            raise HarnessOwnershipError(
                artifact or self.locate(path),
                self._role,
                ", ".join(owners) or "none",
            )
        return target

    # -- reads ----------------------------------------------------------

    def read_text(self, path: str) -> str:
        """Read a protocol artifact, bounded by the workspace's read cap."""
        return self._workspace.read_text(self.locate(path))

    def exists(self, path: str) -> bool:
        """Whether a protocol path exists."""
        return self._workspace.exists(self.locate(path))

    def list_dir(self, path: str = ".") -> list[str]:
        """List a plan-directory (or memory-root) directory's entries."""
        return self._workspace.list_dir(self.locate(path))

    # -- writes ---------------------------------------------------------

    def write_text(self, path: str, content: str) -> str:
        """Write an OWNED artifact, replacing it if it exists."""
        self.authorise(path)
        return self._workspace.write_text(self.locate(path), content)

    def append_text(self, path: str, content: str) -> str:
        """Append to an OWNED artifact, creating it if absent."""
        self.authorise(path)
        return self._workspace.append_text(self.locate(path), content)


# ---------------------------------------------------------------------------
# Feedback: what a write CHANGED, and what the gate now sees
# ---------------------------------------------------------------------------


def has_bytes(reader: Callable[[str], str], path: str) -> bool:
    """Whether *path* carries non-whitespace content, read through its own root.

    Interface contract (shared predicate, 2 call sites: :func:`gate_files` here
    and ``roles.py``'s ``_verified_writes``):
        - ``reader``: a CONFINED reader -- :meth:`Workspace.read_text` or
          :meth:`PlanMemory.read_text`.  Passing ``Path.read_text`` would bypass
          the chokepoint, which is why the parameter is the bound method and not
          a root.
        - Returns ``False`` for a missing file, a refused path, an unreadable
          file and an empty one.  Every failure means "not verified"; none of
          them means "assume it worked".
        - Never raises.
    """
    try:
        return bool(reader(path).strip())
    except Exception:  # missing / refused / undecodable -- all fail closed
        return False


def gate_files(memory: PlanMemory, directory: str) -> tuple[str, ...]:
    """The non-empty ``.md`` files in one plan-directory subdirectory.

    Interface contract (the ONE derivation, 3 call sites: :func:`count_gate_files`
    for the gate value ``roles.py`` puts into context, and :func:`_gate_clause`
    for the two tool results that report it back):
        - ``memory``: the ROLE-SCOPED plan memory the write tools wrote through,
          so the set is read back over exactly the confined root that produced
          it.
        - Returns the file NAMES, sorted, for a directory that exists; an empty
          tuple for one that does not -- "none", never "unknown".
        - Never raises.

    A caller that wants this set or its size must call this function: a second
    count is a gate and a progress report that can disagree, which is the
    fail-open defect D-015 closed.
    """
    try:
        entries = memory.list_dir(directory)
    except Exception:  # the directory does not exist yet: zero, not unknown
        return ()
    return tuple(
        sorted(
            name
            for name in entries
            if name.endswith(".md")
            and has_bytes(memory.read_text, f"{directory}/{name}")
        )
    )


def count_gate_files(memory: PlanMemory, directory: str) -> int:
    """How many non-empty ``.md`` files one plan-directory subdirectory holds."""
    return len(gate_files(memory, directory))


def derive_disk_counts(memory: PlanMemory, keys: Iterable[str]) -> dict[str, int]:
    """The filesystem's own value for every disk-derived gate count in *keys*.

    Interface contract (2 call sites, and they must not be able to disagree:
    ``roles.py``'s worker factory, which replaces the worker's self-reported
    integer, and ``harness.py``'s driver, which derives the same number itself
    regardless of whether the dispatch succeeded):
        - ``memory``: a role-scoped :class:`PlanMemory`; the count is read back
          over exactly the confined root the write tools wrote through.
        - ``keys``: the caller's writable-key set.  A key not in
          :data:`DISK_DERIVED_COUNTS` is not a disk-derived count and is
          ignored; a disk-derived key the caller does not own is not counted
          for it.
        - Returns ``{key: count}``, ``{}`` when no key qualifies.  A directory
          that does not exist counts 0 -- "none", never "unknown".
        - Never raises.
    """
    owned = frozenset(keys)
    return {
        key: count_gate_files(memory, directory)
        for key, (directory, _threshold) in DISK_DERIVED_COUNTS.items()
        if key in owned
    }


def _gate_clause(memory: PlanMemory, artifact: str | None, *, verb: str) -> str:
    """Render the derived, gate-relevant state of *artifact*, or ``""``.

    ``verb`` is the whole of what differs between the callers: a write says
    "now holds" or "still holds" (the repeat-vs-new signal), a listing just
    says "holds".
    """
    for directory, threshold in DISK_DERIVED_COUNTS.values():
        if artifact != directory:
            continue
        names = gate_files(memory, directory)
        # The NAMES are part of the observation, not decoration: the measured
        # failure is one topic written 3-11 times, and a bare count leaves the
        # model to remember which topics it already covered.
        listed = f": {', '.join(names)}" if names else ""
        return (
            f" {directory}/ {verb} {len(names)} of the {threshold} distinct "
            f"non-empty files the exit gate requires{listed}."
        )
    return ""


def _gate_state(memory: PlanMemory, path: str, *, existed: bool) -> str:
    """Report the derived, gate-relevant count the WRITTEN *path* contributes to."""
    try:
        artifact = memory.artifact_for(path)
    except Exception:  # a refused path reports its refusal, not a count
        return ""
    return _gate_clause(
        memory, artifact, verb="still holds" if existed else "now holds"
    )


def _owned_empty_directory(memory: PlanMemory, path: str) -> bool:
    """Whether *path* names a per-plan directory the protocol owns but that is
    not on disk yet.

    # DECISION plan-2026-07-21T191807-bf7ffe24/D-027
    # `findings/` and `checkpoints/` are artifact names the OWNERSHIP table
    # defines, so "the directory is not there" is a fact about the filesystem
    # and not about the protocol: `gate_files` has always answered "none, not
    # unknown" for exactly this case. Measured, step-22 attempt 1: the routing
    # hint correctly moved the explorer off `list_dir("findings/")` and onto
    # `list_plan_dir("findings/")` -- which then answered ENOENT six times in a
    # row, and the run wrote nothing. That ENOENT is the same false belief
    # D-013 recorded from the other side ("cannot write findings without the
    # directory existing"), and `PlanMemory.write_text` creates parents, so it
    # was never true.
    # Do NOT widen this to every missing plan path: a missing `plan.md` really
    # is missing, and answering "(empty)" for an arbitrary path would make the
    # tool lie. Only a path whose LAST component is a per-plan directory
    # artifact qualifies, and only `list_plan_dir` uses it -- reads and writes
    # are untouched.
    # See decisions.md D-027.
    """
    try:
        if memory.artifact_for(path) not in _PER_PLAN_DIRS:
            return False
        return Path(memory.locate(path)).name == memory.artifact_for(path)
    except Exception:
        return False


def _write_result(
    action: str,
    target: str,
    content: str,
    *,
    existed: bool,
    gate: str = "",
) -> str:
    """Render a write tool's result as an OBSERVATION of what changed.

    Interface contract (shared renderer, 4 call sites: the two workspace write
    tools and the two plan write tools):
        - ``action``: a key of :data:`_REPEAT_PHRASE` -- the verb the result
          opens with.
        - ``target``: whatever the confined writer returned (the path it
          actually wrote, which may differ from the path the model asked for).
        - ``existed``: whether the target was there BEFORE this call.
        - ``gate``: an already-rendered clause from :func:`_gate_state`, or
          ``""``.
        - Returns one short line.  Never raises.
    """
    novelty = _REPEAT_PHRASE[action] if existed else "NEW file"
    return f"{action} {target} ({len(content)} chars); {novelty}.{gate}"


def _addresses_plan_memory(path: str) -> bool:
    """Whether *path* names something that lives in the PLAN directory."""
    candidate = path.strip()
    if not candidate:
        return False
    parts = [part for part in Path(candidate).parts if part not in ("/", "\\")]
    if not parts:
        return False
    head = parts[0]
    return (
        head in _CROSS_PLAN_FILES
        or head in _PER_PLAN_FILES
        or head in _PER_PLAN_DIRS
        or head == "plan"
        or bool(_PLAN_ID_RE.match(head))
    )


def _routing_hint(path: str, *, tool_name: str, memory: PlanMemory | None) -> str:
    """Name the tool that owns *path*'s root, when this call aimed at the other.

    ``memory`` is the plan tier for a plan tool and ``None`` for a workspace
    tool -- which is also what selects the direction of the test.
    """
    counterpart = _COUNTERPART_TOOL.get(tool_name)
    if counterpart is None:
        return ""
    if path.strip() in ("", ".", "./"):
        # "the current directory" is not a wrong-root guess in either root.
        return ""
    if memory is None:
        if not _addresses_plan_memory(path):
            return ""
        return (
            "That path belongs to the plan directory, not the workspace: "
            f"use `{counterpart}`."
        )
    try:
        if memory.artifact_for(path) is not None:
            return ""
    except Exception:  # unclassifiable: it is not a protocol artifact either
        pass
    return (
        "That path is not a protocol artifact, so it belongs to the workspace, "
        f"not the plan directory: use `{counterpart}`."
    )


def _relocated_hint(path: str, memory: PlanMemory) -> str:
    """``problem-scope.md`` -> ``findings/problem-scope.md``, when that exists.

    Interface contract (1 call site, :func:`_missing_target_hint`):
        - Returns a "did you mean" clause only when the BASENAME of *path* is
          really a file inside one of the owned per-plan directories, so the
          suggestion is a fact read off the filesystem, never a guess.
        - Returns ``""`` otherwise.  Never raises.
    """
    name = Path(path.strip()).name
    if not name:
        return ""
    for directory in sorted(_PER_PLAN_DIRS):
        try:
            if name in memory.list_dir(directory):
                return f"Did you mean `{directory}/{name}`?"
        except Exception:  # the directory is not there: no suggestion, no crash
            continue
    return ""


def _missing_target_hint(
    path: str, *, tool_name: str, memory: PlanMemory | None
) -> str:
    """Say what is TRUE about a plan artifact a READ could not find.

    Interface contract (1 call site, :func:`_corrective`):
        - Fires only for a FAILED ``read_plan_file`` holding plan memory.  A
          write is never told "write it", and a workspace call is untouched.
        - Returns ``""`` when the path does exist (the failure was something
          else), when it cannot be classified, or when there is nothing true to
          add.
        - Never raises.
    """
    # DECISION plan-2026-07-21T191807-bf7ffe24/D-036
    # The same seam as D-027's routing hint, for the other half of the same
    # measurement: an ALREADY-FAILED call is told what is true, and nothing is
    # repaired, re-routed or turned into a success. Step 25's n=10 live block is
    # what this is sized from: 124 of 323 failed tool calls were reads of a
    # `findings/*.md` file that did not exist yet, and in the three runs that
    # missed the gate those reads WERE the failure -- run 3 called
    # `read_plan_file('findings/constraints-and-patterns.md')` 15 times, ENOENT
    # every time, and spent nine dispatches' worth of turns without ever writing
    # the file it had been assigned. A bare `[Errno 2] No such file or
    # directory` says nothing about what to do next, and D-013 already recorded
    # the same false belief from the write side ("cannot write findings without
    # the directory existing") -- `PlanMemory.write_text` creates parents, so it
    # was never true.
    # Do NOT "fix" this by answering the read with empty content instead: D-027
    # deliberately confined that treatment to an owned DIRECTORY, because a
    # missing `plan.md` really is missing and a tool that reports absent files
    # as empty ones lies to every caller.
    # Do NOT move this into the prompt. The prompt already names the target
    # path once, at the top of the task; this fires at the moment and the place
    # the model meets the error, every time it does, and prompt wording is the
    # mechanism that has now failed three separate times (decisions.md D-027).
    # See decisions.md D-036.
    if memory is None or tool_name != PlanTools.READ_PLAN_FILE:
        return ""
    try:
        if memory.exists(path):
            return ""  # it is there; this failure is about something else
        artifact = memory.artifact_for(path)
    except Exception:  # unclassifiable / outside the root: routing answers it
        return ""
    if artifact is None:
        return _relocated_hint(path, memory)
    return (
        "That protocol artifact does not exist yet -- nothing has written it. "
        "Reading it is not a prerequisite for writing it: "
        f"`{PlanTools.WRITE_PLAN_FILE}` creates the file, and any missing "
        "folder, in one call."
    )


@contextmanager
def _corrective(
    path: str,
    *,
    tool_name: str,
    memory: PlanMemory | None = None,
) -> Iterator[None]:
    """Add the counterpart tool's name to a FAILED cross-root call's message.

    # DECISION plan-2026-07-21T191807-bf7ffe24/D-027
    # This ANNOTATES a failure; it must never repair one. The step-5 spike
    # measured 54 of 298 tool calls (18%) aiming a workspace tool at a plan
    # artifact or the reverse -- `read_file("state.md")`,
    # `write_plan_file("uploader.py")` -- with ZERO hallucinated tool NAMES, so
    # the model knows the tools and mis-picks the ROOT. Three things here are
    # load-bearing and must not be "simplified":
    #   1. Do NOT re-route the call to the counterpart tool. Confinement and
    #      ownership are the two properties an adversarial review attacked at 43
    #      shapes with zero escapes; a layer that silently retries the other
    #      root turns "the EXPLORER may not write plan.md" into "the EXPLORER
    #      may not write plan.md HERE".
    #   2. Do NOT convert the failure into a successful ToolResult. A harness
    #      refusal keeps its own CLASS and every attribute (`.path`, `.role`,
    #      `.artifact`): its message is enriched IN PLACE, so every existing
    #      assertion about the refusal survives untouched. `ToolRegistry.execute`
    #      renders `str(exc)`, which is the only thing the model ever sees.
    #      An `OSError` cannot be enriched that way -- its `__str__` is built
    #      from errno/strerror/filename and ignores `args` -- and ENOENT from a
    #      workspace read of `state.md` is the single most common shape of this
    #      error, so that one branch re-raises as a `HarnessError` chained to
    #      the original rather than dropping the hint.
    #   3. Only a call that ALREADY failed is annotated. A pre-check would have
    #      to refuse `plan.md` in a workspace that legitimately contains one.
    # See decisions.md D-027.
    """
    try:
        yield
    except Exception as exc:
        # A hint about what IS there outranks one about which root to use: for
        # `read_plan_file("problem-scope.md")` the routing hint sends the model
        # to the workspace, and the file is really at `findings/problem-scope.md`
        # (measured 11 times in one step-25 run).  See `_missing_target_hint`.
        hint = _missing_target_hint(
            path, tool_name=tool_name, memory=memory
        ) or _routing_hint(path, tool_name=tool_name, memory=memory)
        if not hint:
            raise
        if not isinstance(exc, OSError) and exc.args and isinstance(exc.args[0], str):
            exc.args = (f"{exc.args[0]}. {hint}", *exc.args[1:])
            raise
        raise HarnessError(f"{exc}. {hint}") from exc


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


def build_workspace_tools(
    workspace: Workspace,
    *,
    allowed: Iterable[str] | None = None,
) -> ToolRegistry:
    """Register *workspace*'s confined operations as agent tools.

    Interface contract (shared helper, 2+ call sites: ``roles.py``'s default
    worker factory, and the live spike):
        - ``workspace``: the root every registered tool is confined to.
        - ``allowed``: tool names to register, from :class:`WorkspaceTools`.
          ``None`` registers every tool.  A role's ``tool_scope`` is passed
          here, which is how invariant I7 becomes structural: a read-only role
          never receives a write tool, rather than being told not to call one.
        - Returns a fresh :class:`~fsm_llm_agents.tools.ToolRegistry`.  Each
          tool carries an inferred JSON schema (via the ``@tool`` decorator), so
          ``get_json_schemas()`` works for native function calling.
        - Raises ``ValueError`` for an unknown tool name.

    This function registers; it does not dispatch.  Execution stays with
    ``ToolRegistry.execute``, which already converts every raised exception
    into a failed ``ToolResult``.
    """

    @tool
    def read_file(path: str) -> str:
        """Read a text file from the workspace. Path is relative to the root."""
        with _corrective(path, tool_name=WorkspaceTools.READ_FILE):
            return workspace.read_text(path)

    @tool
    def write_file(path: str, content: str) -> str:
        """Write a text file in the workspace, replacing it if it exists."""
        with _corrective(path, tool_name=WorkspaceTools.WRITE_FILE):
            existed = workspace.exists(path)
            target = workspace.write_text(path, content)
        return _write_result("wrote", target, content, existed=existed)

    @tool
    def append_file(path: str, content: str) -> str:
        """Append text to a file in the workspace, creating it if absent."""
        with _corrective(path, tool_name=WorkspaceTools.APPEND_FILE):
            existed = workspace.exists(path)
            target = workspace.append_text(path, content)
        return _write_result("appended to", target, content, existed=existed)

    @tool
    def delete_file(path: str) -> str:
        """Delete one file from the workspace. Directories are refused."""
        return f"deleted {workspace.delete(path)}"

    @tool
    def list_dir(path: str = ".") -> str:
        """List the entries of a workspace directory. Directories end with /."""
        with _corrective(path, tool_name=WorkspaceTools.LIST_DIR):
            return "\n".join(workspace.list_dir(path)) or "(empty)"

    @tool
    def path_exists(path: str) -> str:
        """Report whether a workspace path exists."""
        with _corrective(path, tool_name=WorkspaceTools.PATH_EXISTS):
            return "yes" if workspace.exists(path) else "no"

    @tool
    def grep_files(pattern: str, path: str = ".") -> str:
        """Search workspace files for a regular expression. Returns path:line: text."""
        hits = workspace.grep(pattern, path)
        return "\n".join(hits) or "(no matches)"

    @tool
    def run_command(command: str, args: str = "") -> str:
        """Run one allowlisted command in the workspace. Disabled by default."""
        argv = [command, *shlex.split(args)] if args else [command]
        result = workspace.run_command(argv)
        return result.summary

    candidates: Mapping[str, object] = {
        WorkspaceTools.READ_FILE: read_file,
        WorkspaceTools.WRITE_FILE: write_file,
        WorkspaceTools.APPEND_FILE: append_file,
        WorkspaceTools.DELETE_FILE: delete_file,
        WorkspaceTools.LIST_DIR: list_dir,
        WorkspaceTools.PATH_EXISTS: path_exists,
        WorkspaceTools.GREP_FILES: grep_files,
        WorkspaceTools.RUN_COMMAND: run_command,
    }

    names = tuple(candidates) if allowed is None else tuple(allowed)
    unknown = [name for name in names if name not in candidates]
    if unknown:
        raise ValueError(
            f"Unknown workspace tool(s): {', '.join(unknown)}. "
            f"Available: {', '.join(candidates)}"
        )

    registry = ToolRegistry()
    for name in names:
        fn = candidates[name]
        registry.register(fn._tool_definition)  # type: ignore[attr-defined]
    return registry


def build_plan_tools(
    memory: PlanMemory,
    *,
    allowed: Iterable[str] | None = None,
    registry: ToolRegistry | None = None,
) -> ToolRegistry:
    """Register *memory*'s confined, ownership-scoped operations as agent tools.

    Interface contract (shared helper, 2+ call sites: ``roles.py``'s default
    worker factory, and any caller writing its own factory):
        - ``memory``: the plan-directory tier, already scoped to one role.
        - ``allowed``: tool names from :class:`PlanTools`.  ``None`` registers
          every tool.  A role's ``plan_tool_scope`` is passed here.
        - ``registry``: register into an EXISTING registry (the workspace one)
          instead of a fresh one, so a dispatch holds exactly one registry
          spanning both roots.
        - Returns the registry written to.
        - Raises ``ValueError`` for an unknown tool name.

    An ownership refusal surfaces as a failed ``ToolResult`` through
    ``ToolRegistry.execute``, exactly like a confinement refusal, so the agent
    loop can read it and adapt.
    """

    @tool
    def read_plan_file(path: str) -> str:
        """Read a protocol artifact. Paths are relative to the plan directory."""
        with _corrective(path, tool_name=PlanTools.READ_PLAN_FILE, memory=memory):
            return memory.read_text(path)

    @tool
    def write_plan_file(path: str, content: str) -> str:
        """Write a protocol artifact you own, replacing it if it exists."""
        # DECISION plan-2026-07-21T191807-bf7ffe24/D-027
        # The result reports what the write CHANGED and what the exit gate now
        # sees, because from inside the agent loop a repeat write to the same
        # path was indistinguishable from progress. Measured, step-5 spike: the
        # explorer called this tool with the SAME path 3-11 times in one
        # dispatch (`findings/uploader_state.md` x11), never named a second
        # topic, and produced 3 distinct findings files in 0 of 10 runs -- while
        # the tool answered every call with a bare "wrote <path>".
        # Two things must not be "tidied":
        #   1. The count comes from `count_gate_files`, the SAME derivation
        #      `roles.py` puts into the gate key (D-015). Do NOT compute a
        #      second count here, and do NOT cache one across calls: a progress
        #      report that can disagree with the gate is the fail-open defect
        #      this protocol already fixed once.
        #   2. `existed` is read BEFORE the write and rendered explicitly. The
        #      whole point is that a REPEAT is visibly different from a NEW
        #      file; a result that only says "wrote" carries the same
        #      information for both, which is the signal the model lacked.
        # Do NOT answer this with prompt wording instead. The 3-file
        # requirement is already stated twice in the system message, and
        # wording has now failed three independent times (D-013's
        # anti-fabrication clause, step 21's `writesfix` arm, the step-5 stop-
        # rule ablation, which made writes WORSE).
        # See decisions.md D-027.
        with _corrective(path, tool_name=PlanTools.WRITE_PLAN_FILE, memory=memory):
            existed = memory.exists(path)
            target = memory.write_text(path, content)
            gate = _gate_state(memory, path, existed=existed)
        return _write_result("wrote", target, content, existed=existed, gate=gate)

    @tool
    def append_plan_file(path: str, content: str) -> str:
        """Append to a protocol artifact you own, creating it if absent."""
        with _corrective(path, tool_name=PlanTools.APPEND_PLAN_FILE, memory=memory):
            existed = memory.exists(path)
            target = memory.append_text(path, content)
            gate = _gate_state(memory, path, existed=existed)
        return _write_result("appended to", target, content, existed=existed, gate=gate)

    @tool
    def list_plan_dir(path: str = ".") -> str:
        """List a plan-directory listing. Directories end with /."""
        with _corrective(path, tool_name=PlanTools.LIST_PLAN_DIR, memory=memory):
            try:
                entries = memory.list_dir(path)
            except FileNotFoundError:
                if not _owned_empty_directory(memory, path):
                    raise
                entries = []  # an owned directory nobody has written to yet
            listing = "\n".join(entries) or "(empty)"
            gate = _gate_clause(memory, memory.artifact_for(path), verb="holds")
        return f"{listing}\n{gate.strip()}" if gate else listing

    @tool
    def plan_path_exists(path: str) -> str:
        """Report whether a plan-directory path exists."""
        with _corrective(path, tool_name=PlanTools.PLAN_PATH_EXISTS, memory=memory):
            return "yes" if memory.exists(path) else "no"

    candidates: Mapping[str, object] = {
        PlanTools.READ_PLAN_FILE: read_plan_file,
        PlanTools.WRITE_PLAN_FILE: write_plan_file,
        PlanTools.APPEND_PLAN_FILE: append_plan_file,
        PlanTools.LIST_PLAN_DIR: list_plan_dir,
        PlanTools.PLAN_PATH_EXISTS: plan_path_exists,
    }

    names = tuple(candidates) if allowed is None else tuple(allowed)
    unknown = [name for name in names if name not in candidates]
    if unknown:
        raise ValueError(
            f"Unknown plan tool(s): {', '.join(unknown)}. "
            f"Available: {', '.join(candidates)}"
        )

    target = ToolRegistry() if registry is None else registry
    for name in names:
        fn = candidates[name]
        target.register(fn._tool_definition)  # type: ignore[attr-defined]
    return target
