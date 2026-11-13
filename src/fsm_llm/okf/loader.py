"""
The bounded OKF bundle loader.

SECURITY: every file read passes ``security.resolve_within`` and the no-follow
read of ``fsm_llm.frontmatter``; symlinks are skipped, never followed; size, file
count, depth and time are bounded and a spent bound becomes an issue.
"""

from __future__ import annotations

import os
import re
import time
from itertools import islice
from pathlib import Path
from typing import Literal

from fsm_llm import frontmatter as _fm
from fsm_llm.core.definitions import OKFError
from fsm_llm.logging import logger
from fsm_llm.security import resolve_within

from .bundle import OKFBundle, _Concepts
from .links import _id_key, _id_problem, _is_reserved
from .models import OKFIssue, OKFLimits, _capped
from .parser import _load_yaml, _read_bytes, _split_frontmatter, parse_concept_text

# code -> (severity, message) for every problem the loader itself reports.
_LOAD_PROBLEMS: dict[str, tuple[Literal["error", "warning", "info"], str]] = {
    "id-unsafe": ("warning", "file name is not a safe concept id; skipped"),
    "id-collision": ("warning", "id looks like an earlier one; both loaded"),
    "parse-failed": ("warning", "file could not be processed; skipped"),
    "path-indirect": ("warning", "path is indirect or outside the root; skipped"),
    "file-too-large": ("warning", "file is over the size limit; skipped"),
    "file-unreadable": ("warning", "file cannot be read; skipped"),
    "dir-unreadable": ("warning", "directory cannot be listed; skipped"),
    "dir-truncated": ("warning", "directory has too many entries; rest ignored"),
    "symlink-skipped": ("warning", "symlinks are not followed; skipped"),
    "not-utf8": ("error", "file is not valid UTF-8; skipped"),
    "max-total-bytes": ("error", "bundle byte limit reached; load stopped"),
    "max-files": ("error", "bundle file limit reached; load stopped"),
    "max-depth": ("error", "directory is nested too deeply; skipped"),
    "load-deadline": ("error", "load time limit (max_seconds) reached; load stopped"),
    "log-heading": ("warning", "log.md heading is not an ISO date (spec 9)"),
    "index-frontmatter": ("warning", "index.md has frontmatter (spec 8)"),
}


def load_bundle(
    path: str | os.PathLike[str], *, limits: OKFLimits | None = None
) -> OKFBundle:
    """Load the bundle under *path*. ``OKFError`` only for a missing root; every
    other problem (oversize, non-UTF-8, symlinked, unsafe-named or unreadable
    files, a spent limit) becomes an ``OKFIssue`` and the rest loads.

    Interface contract:
        - Files are visited in sorted order; only exact ``.md`` names count.
          ``index.md`` and ``log.md`` in any case, at any level, are reserved,
          never concepts (the root ``index.md`` supplies ``okf_version`` and
          ``index_text``). Entries starting with ``.`` are skipped silently.
        - An id (the path without ``.md``) must pass the writer's id rule, else
          ``id-unsafe``. Ids that collide after NFC, case folding and removal of
          soft hyphen and joiners all load, with an ``id-collision`` warning
          naming the first (``write_bundle`` refuses such a bundle).
        - Symlinks are skipped, never followed; files pass ``resolve_within``.
          Stops with an error issue at ``max_files``, ``max_total_bytes`` and
          ``max_seconds`` (``load-deadline``, checked between entries); what was
          loaded is kept (``issues`` are capped, stops are never cut). Directories
          past ``max_depth`` are not entered.
        - Residual holes: TOCTOU (``O_NOFOLLOW`` guards the last part), hardlinks.
    """
    limits = limits or OKFLimits()
    try:
        root = Path(path).resolve()
        is_dir = root.is_dir()
    except (OSError, RuntimeError, ValueError) as exc:
        raise OKFError(f"cannot open bundle root: {type(exc).__name__}") from exc
    if not is_dir:
        raise OKFError(f"bundle root is not a directory: {root}")

    deadline = time.monotonic() + limits.max_seconds
    issues: list[OKFIssue] = []
    concepts: _Concepts = []
    first_id: dict[str, str] = {}  # _id_key -> first id loaded with it
    total, version, index, stop = 0, None, "", False

    def issue(code: str, rel: str, note: str = "") -> None:
        severity, message = _LOAD_PROBLEMS[code]
        message = f"{message}: {note}" if note else message
        issues.append(OKFIssue(severity=severity, code=code, message=message, path=rel))

    def halt(code: str, rel: str) -> None:
        nonlocal stop
        issue(code, rel)
        stop = True

    def read(parts: tuple[str, ...], size: int) -> str | None:
        """Checked read of one file; ``None`` after recording why it was skipped."""
        nonlocal total, stop
        rel, code, data = "/".join(parts), "", b""
        try:
            if resolve_within(root, rel) != root.joinpath(*parts):
                code = "path-indirect"
            elif size > limits.max_file_bytes:
                code = "file-too-large"
            elif total + size > limits.max_total_bytes:
                code, stop = "max-total-bytes", True
            else:
                data = _read_bytes(root.joinpath(*parts), limits.max_file_bytes)
                code = "file-too-large" if len(data) > limits.max_file_bytes else ""
                total += len(data)
        except (OSError, RuntimeError, ValueError):
            code = "file-unreadable"
        if not code:
            try:
                return data.decode("utf-8")
            except UnicodeDecodeError:
                code = "not-utf8"
        issue(code, rel)
        return None

    def reserved(parts: tuple[str, ...], text: str) -> None:
        """Spec 8 and 9: index has no frontmatter (root: only ``okf_version``);
        log headings are ISO dates. Problems are warnings; these never load."""
        nonlocal version, index
        rel, is_root = "/".join(parts), len(parts) == 1
        raw, body, _ = _split_frontmatter(text)
        if parts[-1].casefold() == "log.md":
            headings = re.findall(r"^## (.*)$", body, re.MULTILINE)
            if not all(re.match(r"\d{4}-\d{2}-\d{2}", h) for h in headings):
                issue("log-heading", rel)
            return
        note: _fm.Note = lambda s, c, m: issues.append(  # noqa: E731
            OKFIssue(severity=s, code=c, message=m, path=rel)
        )
        data = _load_yaml(raw, note) if raw is not None else {}
        if is_root:
            found = data.get("okf_version")
            if isinstance(found, (str, int, float)):
                version = str(found).strip()
            index = body
        if set(data) - ({"okf_version"} if is_root else set()):
            issue("index-frontmatter", rel)

    def add_concept(cid: str, text: str) -> None:
        concept, found = parse_concept_text(cid, text)
        concepts.append(concept)
        if (first := first_id.setdefault(_id_key(cid), cid)) != cid:
            issue("id-collision", f"{cid}.md", f"same as {first[:80]!r}")
        issues.extend(found)

    # DECISION plan-2026-10-21T140931-095ea5f8/D-017
    # Do NOT skip files with `st_nlink > 1`: a hardlink looks like any file,
    # `cp -al` and uv create them, and the skip emptied honest bundles.
    def visit_file(entry: os.DirEntry[str], here: tuple[str, ...]) -> None:
        """One regular ``.md`` file: skip it with an issue, or load it."""
        rel, cid = "/".join(here), "/".join(here)[:-3]
        if here[-1].startswith(".") and cid not in (".", ".."):
            return  # hidden, like a dot-directory (``...md`` is id ``..``: unsafe)
        size = entry.stat(follow_symlinks=False).st_size
        is_reserved = _is_reserved(here[-1][:-3])
        if _id_problem(cid, allow_reserved=True):
            issue("id-unsafe", rel)
        elif not is_reserved and len(concepts) >= limits.max_files:
            halt("max-files", rel)
        elif (text := read(here, size)) is not None:
            try:
                if is_reserved:
                    reserved(here, text)
                else:
                    add_concept(cid, text)
            except Exception as exc:  # defence in depth: one file never aborts a load
                logger.debug(f"OKF file {rel!r} failed: {exc!r}")
                issue("parse-failed", rel)

    def visit(
        entry: os.DirEntry[str], parts: tuple[str, ...], depth: int
    ) -> tuple[Path, tuple[str, ...], int] | None:
        """Handle one entry; return a directory to scan later, if it is one."""
        name, here = entry.name, (*parts, entry.name)
        if entry.is_symlink():
            if name.endswith(".md") or entry.is_dir():
                issue("symlink-skipped", "/".join(here))
        elif entry.is_dir(follow_symlinks=False):
            if name.startswith("."):
                return None
            if depth < limits.max_depth:
                return Path(entry.path), here, depth + 1
            issue("max-depth", "/".join(here))
        elif (
            name.endswith(".md")
            and len(name) > 3
            and entry.is_file(follow_symlinks=False)
        ):
            visit_file(entry, here)
        return None

    stack: list[tuple[Path, tuple[str, ...], int]] = [(root, (), 0)]
    while stack and not stop:
        directory, parts, depth = stack.pop()
        try:
            with os.scandir(directory) as scan:
                entries = sorted(
                    islice(scan, limits.max_files * 2), key=lambda e: e.name
                )
        except OSError:
            issue("dir-unreadable", "/".join(parts))
            continue
        if len(entries) >= limits.max_files * 2:
            issue("dir-truncated", "/".join(parts))
        subdirs = []
        for entry in entries:
            if time.monotonic() > deadline:
                halt("load-deadline", "/".join((*parts, entry.name)))
            if stop:
                break
            try:
                subdir = visit(entry, parts, depth)
            except OSError:
                issue("file-unreadable", "/".join((*parts, entry.name)))
                continue
            if subdir is not None:
                subdirs.append(subdir)
        stack.extend(reversed(subdirs))

    return OKFBundle(
        concepts,
        root=root,
        issues=_capped(issues),
        okf_version=version,
        index_text=index,
        max_seconds=limits.max_seconds,
    )
