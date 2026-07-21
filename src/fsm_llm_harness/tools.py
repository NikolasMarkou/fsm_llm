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
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from fsm_llm.logging import logger
from fsm_llm_agents.definitions import ToolResult
from fsm_llm_agents.tools import ToolRegistry, tool

from .constants import ArtifactNames
from .exceptions import (
    HarnessConfinementError,
    HarnessError,
    HarnessOwnershipError,
)
from .rules import OWNERSHIP

__all__ = [
    "COMMAND_ALLOWLIST",
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


def _truncate(text: str, limit: int, unit: str = "characters") -> str:
    """Return *text* bounded to *limit*, marking what was dropped."""
    if len(text) <= limit:
        return text
    return text[:limit] + TRUNCATION_MARKER.format(omitted=len(text) - limit, unit=unit)


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
            raise HarnessConfinementError(relative_path, str(self._root))

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
        if not candidate or Path(candidate).is_absolute():
            return candidate
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
        return workspace.read_text(path)

    @tool
    def write_file(path: str, content: str) -> str:
        """Write a text file in the workspace, replacing it if it exists."""
        return f"wrote {workspace.write_text(path, content)}"

    @tool
    def append_file(path: str, content: str) -> str:
        """Append text to a file in the workspace, creating it if absent."""
        return f"appended to {workspace.append_text(path, content)}"

    @tool
    def delete_file(path: str) -> str:
        """Delete one file from the workspace. Directories are refused."""
        return f"deleted {workspace.delete(path)}"

    @tool
    def list_dir(path: str = ".") -> str:
        """List the entries of a workspace directory. Directories end with /."""
        return "\n".join(workspace.list_dir(path)) or "(empty)"

    @tool
    def path_exists(path: str) -> str:
        """Report whether a workspace path exists."""
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
        return memory.read_text(path)

    @tool
    def write_plan_file(path: str, content: str) -> str:
        """Write a protocol artifact you own, replacing it if it exists."""
        return f"wrote {memory.write_text(path, content)}"

    @tool
    def append_plan_file(path: str, content: str) -> str:
        """Append to a protocol artifact you own, creating it if absent."""
        return f"appended to {memory.append_text(path, content)}"

    @tool
    def list_plan_dir(path: str = ".") -> str:
        """List a plan-directory listing. Directories end with /."""
        return "\n".join(memory.list_dir(path)) or "(empty)"

    @tool
    def plan_path_exists(path: str) -> str:
        """Report whether a plan-directory path exists."""
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
