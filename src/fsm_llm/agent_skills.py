"""
Agent Skills: folders with a ``SKILL.md`` (YAML frontmatter plus markdown body).

The format layer, read-only: a tolerant parser, bounded loading and a
progressive-disclosure catalog. Skill text is untrusted prose: content never raises
(a defect becomes a ``SkillIssue``), nothing in a skill folder is executed or
imported, ``allowed-tools`` is reported, never enforced, the body is never kept.

Spec status: the open standard at agentskills.io, read on 2026-10-23, unversioned
and mutable. Names are ASCII (``[a-z0-9]`` and single hyphens), as Anthropic's docs
say; the standard's prose allows unicode letters, so a unicode name is skipped
(``name-invalid``), never loaded under a lookalike.
"""

from __future__ import annotations

# DECISION plan-2026-10-23T043147-4c2042f3/D-025
# Nothing a skill folder holds is executed. Do NOT import subprocess, runpy,
# importlib, shlex, ctypes, multiprocessing or pty in this module, do NOT call
# exec, eval, compile, os.system, os.popen, os.exec* or os.spawn*, and do NOT add
# a "run this script" helper. `scripts/` files are listed and readable as text;
# `allowed-tools` is reported, never granted. See decisions.md D-025.
import html
import os
import re
import stat
import time
from collections.abc import Iterable, Iterator
from difflib import get_close_matches
from itertools import islice
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .constants import (
    SKILL_BODY_MAX_LINES,
    SKILL_CATALOG_DESCRIPTION_CHARS,
    SKILL_CATALOG_END,
    SKILL_CATALOG_MAX_ITEMS,
    SKILL_COMPATIBILITY_MAX_CHARS,
    SKILL_DESCRIPTION_MAX_CHARS,
    SKILL_ECHO_CHARS,
    SKILL_INCOMPLETE_NOTE,
    SKILL_MAX_DEPTH,
    SKILL_MAX_FILE_BYTES,
    SKILL_MAX_FRONTMATTER_BYTES,
    SKILL_MAX_ISSUES,
    SKILL_MAX_ISSUES_PER_SKILL,
    SKILL_MAX_LOAD_SECONDS,
    SKILL_MAX_RESOURCES,
    SKILL_MAX_SKILLS,
    SKILL_MAX_YAML_DEPTH,
    SKILL_NAME_ECHO_CHARS,
    SKILL_NAME_MAX_CHARS,
    SKILL_READ_PAGE_CHARS,
    SKILL_SCAN_FACTOR,
    SKILL_TEXT_MAX_CHARS,
    SKILL_UNTRUSTED_LABEL,
)
from .definitions import SkillError
from .frontmatter import Note, load_yaml, read_bytes, split_frontmatter
from .okf import capped_issues, capped_line, strip_unsafe_text, text_page
from .security import resolve_within

__all__ = [
    "AgentSkill",
    "SkillCatalog",
    "SkillIssue",
    "SkillLimits",
    "incomplete_note",
    "load_skills",
    "parse_skill_text",
]

# The six keys of the open standard; any other key is reported, never an error.
_KNOWN_KEYS = frozenset(
    {"name", "description", "license", "compatibility", "metadata", "allowed-tools"}
)
_Severity = Literal["error", "warning", "info"]
_NAME_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
# Anthropic's API refuses these inside a name; kept as a portability warning.
_RESERVED_WORDS = ("anthropic", "claude")
# One tool: a run of non-space, non-comma characters, with an optional ``(...)``
# argument that may hold spaces (``Bash(git add:*), Read Grep``).
_TOOL_RE = re.compile(r"[^\s,(]+(?:\([^)]*\))?")


class SkillIssue(BaseModel):
    """One problem found in a skill; untrusted ``message`` and ``path`` are stripped."""

    model_config = ConfigDict(frozen=True)

    severity: _Severity
    code: str
    message: str
    path: str | None = None

    @field_validator("message", "path")
    @classmethod
    def _clean(cls, value: str | None) -> str | None:
        return value and strip_unsafe_text(value)[:SKILL_TEXT_MAX_CHARS]


def _clean(value: Any) -> Any:
    """Control-strip every string in a str, tuple or dict value; others as is."""
    if isinstance(value, str):
        return strip_unsafe_text(value)
    if isinstance(value, tuple):
        return tuple(_clean(item) for item in value)
    if isinstance(value, dict):
        return {_clean(key): _clean(item) for key, item in value.items()}
    return value


# DECISION plan-2026-10-23T043147-4c2042f3/D-022
# Do NOT add a ``body`` (or ``text``) field here, and do NOT keep the body in a
# cache next to the catalog to "save a disk read". Progressive disclosure is the
# point of the format: the catalog holds metadata only, and ``read_body`` re-reads
# the file, bounded and paged. A retained body costs memory per skill and puts
# untrusted prose where no reader decided to read it. See decisions.md D-022.
class AgentSkill(BaseModel):
    """The metadata of one skill: no body, only its length in lines.

    Interface contract: built by ``parse_skill_text`` (the loader sets
    ``directory`` and ``resources``); every string is control-stripped here,
    whoever constructs it. ``allowed_tools``, ``license`` and ``compatibility``
    are claims the skill makes, reported and never enforced."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    directory: Path | None = None
    skill_md: str = "SKILL.md"
    license: str | None = None
    compatibility: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    allowed_tools: tuple[str, ...] = ()
    extra_keys: tuple[str, ...] = ()
    line_count: int = 0
    resources: tuple[str, ...] = ()

    @field_validator("*")
    @classmethod
    def _strip_untrusted(cls, value: Any) -> Any:
        return _clean(value)


def _text(
    data: dict[Any, Any], key: str, code: str, note: Note, *, filled: bool = False
) -> str:
    """The stripped string at *key*; ``""`` when absent, null or not a string
    (a non-string adds a ``code`` warning, so does an empty one when *filled*)."""
    value = data.get(key)
    if isinstance(value, str):
        text = strip_unsafe_text(value).strip()
        if filled and not text:
            note("warning", code, f"{key!r} is empty; ignored")
        return text
    if value is not None:
        note("warning", code, f"{key!r} must be a string; ignored")
    return ""


def parse_skill_text(
    text: str,
    folder_name: str,
    *,
    directory: Path | None = None,
    skill_md: str = "SKILL.md",
    resources: tuple[str, ...] = (),
) -> tuple[AgentSkill | None, list[SkillIssue]]:
    """Parse the text of one ``SKILL.md`` into ``(skill or None, issues)``.

    Interface contract: pure and total (no filesystem access, never raises on
    content). ``None`` comes with an ``error`` issue: bad frontmatter, a missing
    or non-string description, or a name that is not ``[a-z0-9]`` words joined by
    single hyphens in 1 to 64 characters (defaulting to *folder_name*). All else
    loads with a warning (at most ``SKILL_MAX_ISSUES_PER_SKILL``, then one
    ``issues-truncated`` info); the body is counted, discarded."""
    issues: list[SkillIssue] = []
    where = f"{folder_name}/{skill_md}"
    warned = 0

    def note(severity: _Severity, code: str, msg: str) -> None:
        nonlocal warned
        if severity == "warning":  # bound what hostile frontmatter can build
            warned += 1
            if warned > SKILL_MAX_ISSUES_PER_SKILL:
                if warned > SKILL_MAX_ISSUES_PER_SKILL + 1:
                    return
                severity, code = "info", "issues-truncated"
                msg = "further warnings for this skill are not shown"
        issues.append(SkillIssue(severity=severity, code=code, message=msg, path=where))

    raw, body, problem = split_frontmatter(text)
    if raw is None:
        note("error", problem or "frontmatter-missing", "no complete frontmatter block")
        return None, issues
    data = load_yaml(
        raw, note, max_bytes=SKILL_MAX_FRONTMATTER_BYTES, max_depth=SKILL_MAX_YAML_DEPTH
    )
    if any(issue.severity == "error" for issue in issues):
        return None, issues

    given_name = _text(data, "name", "value-invalid", note)
    if not given_name:
        note("warning", "name-missing", "no usable 'name'; using the folder name")
    name = given_name or strip_unsafe_text(folder_name).strip()
    raw = data.get("description")
    description = strip_unsafe_text(raw).strip() if isinstance(raw, str) else ""
    if not description:
        missing = raw is None or isinstance(raw, str)
        note(
            "error",
            "description-missing" if missing else "description-invalid",
            "'description' is required and must be a non-empty string",
        )
    if len(name) > SKILL_NAME_MAX_CHARS or not _NAME_RE.fullmatch(name):
        note(
            "error",
            "name-invalid",
            f"name {name[:SKILL_NAME_ECHO_CHARS]!r} must be lowercase letters and "
            f"digits joined by single hyphens, at most {SKILL_NAME_MAX_CHARS} characters",
        )
    if any(issue.severity == "error" for issue in issues):
        return None, issues

    if given_name and given_name != folder_name:
        note("warning", "name-dir-mismatch", f"name {name!r} differs from its folder")
    if any(word in name for word in _RESERVED_WORDS):
        note("warning", "name-reserved", "name holds a word reserved by Anthropic")
    if "<" in description or ">" in description:  # the name regex excludes them
        note("warning", "xml-chars", "description holds '<' or '>' (XML tags)")
    if len(description) > SKILL_DESCRIPTION_MAX_CHARS:
        note(
            "warning",
            "description-long",
            f"description has {len(description)} characters; "
            f"the standard allows {SKILL_DESCRIPTION_MAX_CHARS}",
        )
    compatibility = _text(
        data, "compatibility", "compatibility-invalid", note, filled=True
    )
    if len(compatibility) > SKILL_COMPATIBILITY_MAX_CHARS:
        note("warning", "compatibility-long", "compatibility is over the limit")

    given_metadata = data.get("metadata")
    metadata = (
        {
            k: v
            for k, v in given_metadata.items()
            if isinstance(k, str) and isinstance(v, str)
        }
        if isinstance(given_metadata, dict)
        else {}
    )
    if given_metadata is not None and (
        not isinstance(given_metadata, dict) or len(metadata) != len(given_metadata)
    ):
        note("warning", "metadata-invalid", "'metadata' must map strings to strings")
    tools = data.get("allowed-tools")
    allowed: tuple[str, ...] = ()
    if isinstance(tools, str):
        allowed = tuple(_TOOL_RE.findall(strip_unsafe_text(tools)))
    elif isinstance(tools, list) and all(isinstance(tool, str) for tool in tools):
        allowed = tuple(tools)
    elif tools is not None:
        note("warning", "allowed-tools-format", "'allowed-tools' ignored: not text")
    extras = tuple(
        capped_line(key, 64) or f"<{type(key).__name__}>"
        for key in data
        if key not in _KNOWN_KEYS
    )
    if extras:
        note("warning", "key-unknown", f"unknown keys: {', '.join(extras)}")

    line_count = body.count("\n") + (1 if body and not body.endswith("\n") else 0)
    if line_count > SKILL_BODY_MAX_LINES:
        note("info", "body-long", f"body has {line_count} lines; guidance is 500")
    skill = AgentSkill(
        name=name,
        description=description,
        directory=directory,
        skill_md=skill_md,
        license=_text(data, "license", "license-invalid", note, filled=True) or None,
        compatibility=compatibility or None,
        metadata=metadata,
        allowed_tools=allowed,
        extra_keys=extras,
        line_count=line_count,
        resources=resources,
    )
    return skill, issues


# --- Loading ---
_SKILL_MD = "SKILL.md"
_SKIPPED_NAMES = ("node_modules", "__pycache__")
INCOMPLETE_CODES = (
    "max-skills",
    "load-deadline",
    "dir-truncated",
    "resources-truncated",
)
# code -> fixed text of a loader issue or of a refused read (``SkillError``,
# the code in ``details``): never skill text, never a path.
_MESSAGES = {
    "symlink-skipped": "symlinks inside a skill are not followed; skipped",
    "not-utf8": "SKILL.md is not valid UTF-8; decoded with replacement",
    "skill-md-too-large": "SKILL.md is over the byte bound; skipped",
    "skill-md-unreadable": "SKILL.md cannot be read; skipped",
    "name-duplicate": "an earlier skill has this name; skipped",
    "resource-name-unsafe": "files with unsafe names are not listed",
    "resources-truncated": "resource list cut at a bound",
    "dir-unreadable": "directory cannot be listed; skipped",
    "dir-truncated": "directory has too many entries; rest ignored",
    "max-skills": "skill limit (max_skills) reached; load stopped",
    "load-deadline": "load time limit (max_seconds) reached; load stopped",
    "no-skills": "no skills found under the given path",
    "path-refused": "resource path is not allowed",
    "path-indirect": "resource is behind a symlink; not read",
    "not-regular": "resource is not a regular file",
    "file-unreadable": "resource cannot be read",
    "file-too-large": "resource is over the size bound",
    "not-text": "resource is binary or not UTF-8 text",
}
# the loader issues that are errors; every other one is a warning
_ERROR_CODES = frozenset(
    {"skill-md-too-large", "skill-md-unreadable", "dir-truncated"}
    | {"max-skills", "load-deadline"}
)


class SkillLimits(BaseModel):
    """Resource bounds for reading skill folders (defaults from ``SKILL_*``).

    ``max_file_bytes`` bounds SKILL.md and every resource read;
    ``max_seconds`` is a wall-clock budget for the whole load."""

    model_config = ConfigDict(frozen=True)

    max_skills: int = Field(SKILL_MAX_SKILLS, gt=0)
    max_file_bytes: int = Field(SKILL_MAX_FILE_BYTES, gt=0)
    max_resources_per_skill: int = Field(SKILL_MAX_RESOURCES, gt=0)
    max_seconds: float = Field(SKILL_MAX_LOAD_SECONDS, gt=0, allow_inf_nan=False)


# DECISION plan-2026-10-23T043147-4c2042f3/D-024
# Do NOT widen this allowlist to dot entries, and do NOT relax a segment rule
# here: `.env`, `.git` and `.ssh` live in dot names, and a hostile skill author
# chooses every name. Listing and reading share this one path predicate (size,
# symlink and content checks are separate, at read time). See decisions.md D-024.
def _path_parts(path: object) -> tuple[str, ...] | None:
    """The segments of a safe relative path (1 to 300 characters, unchanged by
    ``strip_unsafe_text``, no backslash, no empty, ``.``, ``..`` or dot segment),
    else ``None``. Never raises."""
    if not isinstance(path, str) or not 0 < len(path) <= SKILL_TEXT_MAX_CHARS:
        return None
    if "\\" in path or strip_unsafe_text(path) != path:
        return None
    parts = tuple(path.split("/"))
    return None if any(not p or p.startswith(".") for p in parts) else parts


# DECISION plan-2026-10-23T043147-4c2042f3/D-023
# Do NOT follow a symlink inside a skill, for listing or for reading. Do NOT
# relax the lexical-equality test below to the plain `resolve_within`
# containment test (a link to `.env` or to a FIFO inside the same skill passes
# it) and do NOT drop the `O_NOFOLLOW` of `read_bytes`. The skill author places
# the links; skipping them is the defence. Only a skill DIRECTORY directly under
# a root is followed (the caller named the root). See decisions.md D-023.
def _confined(directory: Path, parts: tuple[str, ...]) -> Path | None:
    """*directory* joined with *parts* (already resolved), ``None`` when a symlink
    lies on the way or the path cannot be resolved. Never raises."""
    target = directory.joinpath(*parts)
    try:
        return target if resolve_within(directory, "/".join(parts)) == target else None
    except (OSError, RuntimeError, ValueError):
        return None


def _refuse(code: str) -> SkillError:
    return SkillError(_MESSAGES[code], details={"code": code})


def _read_file(directory: Path, parts: tuple[str, ...], limit: int) -> bytes:
    """The bytes of one confined, regular file of at most *limit* bytes; else a
    ``SkillError`` (fixed message, code in ``details``). Never blocks."""
    target = _confined(directory, parts)
    if target is None:
        raise _refuse("path-indirect")
    try:
        if not stat.S_ISREG(os.lstat(target).st_mode):
            raise _refuse("not-regular")
        data = read_bytes(target, limit)
    except OSError as exc:
        raise _refuse("file-unreadable") from exc
    if len(data) > limit:
        raise _refuse("file-too-large")
    return data


def _note(issues: list[SkillIssue], code: str, path: str, extra: str = "") -> None:
    message = f"{_MESSAGES[code]}: {extra}" if extra else _MESSAGES[code]
    severity: _Severity = "error" if code in _ERROR_CODES else "warning"
    issues.append(SkillIssue(severity=severity, code=code, message=message, path=path))


def _list_resources(
    skill_dir: Path,
    folder: str,
    limits: SkillLimits,
    deadline: float,
    issues: list[SkillIssue],
) -> tuple[str, ...]:
    """Sorted relative paths of the regular files a skill bundles (never raises);
    a bound or *deadline* ends the walk with ``resources-truncated``."""
    found: list[str] = []
    links, first_link, unsafe, unreadable, visited, cut = 0, "", 0, 0, 0, False
    budget = limits.max_resources_per_skill * SKILL_SCAN_FACTOR
    stack: list[tuple[Path, tuple[str, ...], int]] = [(skill_dir, (), 0)]
    while stack and not cut:
        directory, parts, depth = stack.pop()
        try:
            with os.scandir(directory) as scan:
                entries = sorted(islice(scan, budget + 1), key=lambda e: e.name)
        except OSError:
            unreadable += 1
            continue
        subdirs = []
        for entry in entries:
            visited += 1
            if visited > budget or time.monotonic() > deadline:
                cut = True
                break
            name, here = entry.name, (*parts, entry.name)
            if name.startswith(".") or name in _SKIPPED_NAMES:
                continue
            try:
                if entry.is_symlink():
                    links, first_link = links + 1, first_link or "/".join(here)
                    continue
                if _path_parts("/".join(here)) is None:
                    unsafe += 1
                elif entry.is_dir(follow_symlinks=False):
                    if depth < SKILL_MAX_DEPTH:
                        subdirs.append((Path(entry.path), here, depth + 1))
                    else:
                        cut = True
                elif entry.is_file(follow_symlinks=False) and here != (_SKILL_MD,):
                    if len(found) >= limits.max_resources_per_skill:
                        cut = True
                        break
                    found.append("/".join(here))
            except OSError:
                continue
        stack.extend(reversed(subdirs))
    if links:
        _note(issues, "symlink-skipped", f"{folder}/{first_link}", f"{links} skipped")
    if unsafe:
        _note(issues, "resource-name-unsafe", folder, f"{unsafe} not listed")
    if unreadable:
        _note(issues, "dir-unreadable", folder, f"{unreadable} directories")
    if cut:
        _note(issues, "resources-truncated", folder)
    return tuple(sorted(found))


def _open_roots(path_or_paths: Any) -> list[Path]:
    """The resolved roots; ``SkillError`` for caller misuse only."""
    try:
        given = (
            [path_or_paths]
            if isinstance(path_or_paths, (str, os.PathLike))
            else list(path_or_paths)
        )
        roots = [Path(p).resolve() for p in given]
        bad = next((r for r in roots if not r.is_dir()), None)
    except (TypeError, OSError, RuntimeError, ValueError) as exc:
        raise SkillError(f"cannot open skill root: {type(exc).__name__}") from exc
    if not roots:
        raise SkillError("no skill path given")
    if bad is not None:
        raise SkillError(f"skill root is not a directory: {bad}")
    return roots


def _candidates(
    root: Path, limits: SkillLimits, issues: list[SkillIssue]
) -> list[tuple[str, Path]]:
    """``(folder name, resolved dir)``: the root if it holds a ``SKILL.md``, else
    its sorted, non-dot child directories that do (a symlinked one is followed)."""
    if os.path.lexists(root / _SKILL_MD):
        return [(root.name, root)]
    cap = limits.max_skills * SKILL_SCAN_FACTOR
    try:
        with os.scandir(root) as scan:
            entries = sorted(islice(scan, cap + 1), key=lambda e: e.name)
    except OSError:
        _note(issues, "dir-unreadable", root.name)
        return []
    if len(entries) > cap:
        _note(issues, "dir-truncated", root.name)
    out: list[tuple[str, Path]] = []
    for entry in entries:
        try:
            if entry.name.startswith(".") or not entry.is_dir():
                continue
            skill_dir = Path(entry.path).resolve()
            if os.path.lexists(skill_dir / _SKILL_MD):
                out.append((entry.name, skill_dir))
        except (OSError, RuntimeError, ValueError):
            continue
    return out


def load_skills(
    path_or_paths: str | os.PathLike[str] | Iterable[str | os.PathLike[str]],
    *,
    limits: SkillLimits | None = None,
) -> SkillCatalog:
    """Load the skills under the named directories into a ``SkillCatalog``.

    Interface contract: ``SkillError`` only for caller misuse (no path, a root
    that is missing or not a directory); every other problem is a ``SkillIssue``.
    A root holding a ``SKILL.md`` is one skill, else each child directory holding
    one is (depth 1); a symlinked skill directory is followed once, symlinks
    inside a skill are skipped. The earlier root, then sorted order, wins a
    duplicate name. A spent bound makes the catalog ``incomplete``. Residual
    holes: TOCTOU (``O_NOFOLLOW`` guards the last part) and hardlinks."""
    limits = limits or SkillLimits()
    roots = _open_roots(path_or_paths)
    deadline = time.monotonic() + limits.max_seconds
    issues: list[SkillIssue] = []
    skills: dict[str, AgentSkill] = {}
    seen_dirs: set[Path] = set()
    candidates = [c for root in roots for c in _candidates(root, limits, issues)]
    for folder, skill_dir in candidates:
        if skill_dir in seen_dirs:
            continue
        seen_dirs.add(skill_dir)
        if time.monotonic() > deadline:
            _note(issues, "load-deadline", folder)
            break
        if len(skills) >= limits.max_skills:
            _note(issues, "max-skills", folder)
            break
        where = f"{folder}/{_SKILL_MD}"
        try:
            data = _read_file(skill_dir, (_SKILL_MD,), limits.max_file_bytes)
        except SkillError as refused:
            code = {
                "path-indirect": "symlink-skipped",
                "file-too-large": "skill-md-too-large",
            }.get(refused.details["code"], "skill-md-unreadable")
            _note(issues, code, where)
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("utf-8", "replace")
            _note(issues, "not-utf8", where)
        skill, found = parse_skill_text(text, folder, directory=skill_dir)
        issues.extend(found)
        if skill is None:
            continue
        if skill.name in skills:
            _note(issues, "name-duplicate", where)
            continue
        listed = _list_resources(skill_dir, folder, limits, deadline, issues)
        skills[skill.name] = skill.model_copy(update={"resources": listed})
    if not skills:
        _note(issues, "no-skills", ", ".join(r.name for r in roots)[:SKILL_ECHO_CHARS])
    issues = capped_issues(issues, SKILL_MAX_ISSUES, INCOMPLETE_CODES, SkillIssue)
    return SkillCatalog(
        skills.values(), issues=issues, max_file_bytes=limits.max_file_bytes
    )


# --- Catalog ---
class SkillCatalog:
    """The loaded skills (metadata only) and what was found while loading them.

    Interface contract: ``get``, ``in``, ``len``, iteration and ``names`` never
    raise. ``read_body`` and ``read_resource`` re-read from disk, confined,
    bounded and paged; an unknown name or any refusal is a ``SkillError`` with a
    FIXED message (the code is in ``details``). ``render_catalog`` is the
    labelled, escaped block for a caller's task string (``""`` for no skills)."""

    def __init__(
        self,
        skills: Iterable[AgentSkill],
        *,
        issues: Iterable[SkillIssue] = (),
        max_file_bytes: int = SKILL_MAX_FILE_BYTES,
    ) -> None:
        self._skills = {skill.name: skill for skill in skills}
        self.issues = list(issues)
        self._max_file_bytes = max_file_bytes

    def __len__(self) -> int:
        return len(self._skills)

    def __iter__(self) -> Iterator[AgentSkill]:
        return iter(self._skills.values())

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._skills

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._skills)

    @property
    def incomplete(self) -> tuple[str, ...]:
        """The stop codes among the issues; empty for a complete load."""
        seen = {issue.code for issue in self.issues}
        return tuple(code for code in INCOMPLETE_CODES if code in seen)

    def get(self, name: object) -> AgentSkill | None:
        return self._skills.get(name) if isinstance(name, str) else None

    def closest(self, name: object, n: int = 5) -> list[str]:
        """Up to *n* skill names close to *name* (the not-found hint)."""
        return get_close_matches(str(name)[:SKILL_ECHO_CHARS], self._skills, n=n)

    def render_catalog(
        self,
        *,
        max_items: int = SKILL_CATALOG_MAX_ITEMS,
        description_chars: int = SKILL_CATALOG_DESCRIPTION_CHARS,
    ) -> str:
        """The label, then ``<available_skills>`` with one escaped ``<skill>`` per
        skill (name, capped description; no path), a fixed end line, the incomplete
        note. Escaped by ``html.escape(quote=False)``: tags and ``&`` only."""
        if not self._skills:
            return ""
        shown = list(self._skills.values())[: max(max_items, 0)]
        lines = [SKILL_UNTRUSTED_LABEL, "<available_skills>"]
        for skill in shown:
            name = html.escape(skill.name, quote=False)
            text = capped_line(skill.description, description_chars)
            text = html.escape(text, quote=False)
            lines.append(
                f"<skill><name>{name}</name><description>{text}</description></skill>"
            )
        if len(shown) < len(self._skills):
            lines.append(f"({len(self._skills) - len(shown)} more, use skill_list)")
        lines += ["</available_skills>", SKILL_CATALOG_END]
        if note := incomplete_note(self):
            lines.append(note)
        return "\n".join(lines)

    def read_body(
        self, name: str, offset: Any = 0, *, size: int = SKILL_READ_PAGE_CHARS
    ) -> tuple[str, int | None, int]:
        """``(page, next offset or None, length)`` of the stripped body of *name*,
        re-read from SKILL.md; pages add up to the whole body."""
        text = self._text(name, (_SKILL_MD,), strict=False)
        body = strip_unsafe_text(split_frontmatter(text)[1], multiline=True)
        return text_page(body, offset, max(size, 1))

    def read_resource(
        self,
        name: str,
        path: Any,
        offset: Any = 0,
        *,
        size: int = SKILL_READ_PAGE_CHARS,
    ) -> tuple[str, int | None, int]:
        """``(page, next offset or None, length)`` of a bundled text file; a
        ``SkillError`` for an unsafe path, a symlink, a non-regular, oversize or
        unreadable file, and binary (NUL) or non-UTF-8 content."""
        parts = _path_parts(path)
        if parts is None:
            raise _refuse("path-refused")
        text = self._text(name, parts, strict=True)
        return text_page(strip_unsafe_text(text, multiline=True), offset, max(size, 1))

    def _text(self, name: str, parts: tuple[str, ...], *, strict: bool) -> str:
        skill = self.get(name)
        if skill is None or skill.directory is None:
            raise SkillError("unknown skill")
        data = _read_file(skill.directory, parts, self._max_file_bytes)
        if not strict:
            return data.decode("utf-8", "replace")
        try:
            if b"\0" in data:
                raise ValueError("NUL byte")
            return data.decode("utf-8")
        except ValueError:  # includes UnicodeDecodeError
            raise _refuse("not-text") from None


def incomplete_note(catalog: SkillCatalog) -> str:
    """The fixed note for a partial catalog (codes only, no issue text); else ``""``."""
    codes = catalog.incomplete
    return SKILL_INCOMPLETE_NOTE.format(codes=", ".join(codes)) if codes else ""
