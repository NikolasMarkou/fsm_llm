"""
The OKF writer: render, validate and write a bundle atomically per file.

SECURITY: every target passes ``security.resolve_within`` and must EQUAL its
lexical path (no symlink); every file is rendered and encoded before the first byte
is written; what is written always loads back within the loader's caps.
"""

from __future__ import annotations

import os
import re
import tempfile
from contextlib import suppress
from datetime import date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import yaml

from fsm_llm import frontmatter as _fm
from fsm_llm.constants import (
    OKF_MAX_FILE_BYTES,
    OKF_MAX_FRONTMATTER_BYTES,
    OKF_MAX_YAML_DEPTH,
    OKF_SUMMARY_FIELD_CHARS,
)
from fsm_llm.core.definitions import OKFError
from fsm_llm.security import resolve_within
from fsm_llm.text import iso_utc, strip_unsafe_text

from .bundle import OKFBundle
from .links import _id_key, _id_problem
from .models import OKFConcept
from .parser import _KNOWN_KEYS

_INDEX_BUDGET = OKF_MAX_FILE_BYTES * 3 // 4  # room for the last line, so it loads back
_MAX_CLASHES_SHOWN = 5
_MAX_PATH_BYTES = 4_000  # Linux PATH_MAX is 4,096; room left for the temp sibling


class _OKFDumper(yaml.SafeDumper):
    """``SafeDumper`` without anchors or aliases (the loader refuses them)."""

    def ignore_aliases(self, data: Any) -> bool:
        return True


def _plain(value: Any) -> Any:
    """Copy *value* for dumping; datetimes and dates become offset-bearing text."""
    if isinstance(value, datetime):
        return iso_utc(value)
    if isinstance(value, date):
        return iso_utc(datetime(value.year, value.month, value.day))
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def has_lone_surrogate(value: Any) -> bool:
    """True when a string, key or item in *value* is a lone surrogate; ``OKFError``
    past ``OKF_MAX_YAML_DEPTH`` (a cycle ends there) or ``OKF_MAX_FRONTMATTER_BYTES``
    nodes, surrogate or not."""
    try:
        return _fm.lone_surrogate(
            value, max_depth=OKF_MAX_YAML_DEPTH, max_nodes=OKF_MAX_FRONTMATTER_BYTES
        )
    except RecursionError as exc:
        raise OKFError(str(exc)) from exc


class _FrontmatterTooLargeError(OKFError):
    """Frontmatter over ``OKF_MAX_FRONTMATTER_BYTES`` (the export cuts metadata)."""


def _render_concept(concept: OKFConcept) -> bytes:
    """The UTF-8 file bytes of *concept*; ``OKFError`` when it cannot be written
    or would not load back (frontmatter or file over its byte cap). Keys: the
    known ones, ``usage_window`` after ``sources``, then ``extra``."""
    extra = dict(concept.extra)
    shadowed = sorted(k for k in extra if k in _KNOWN_KEYS)
    if shadowed:
        raise OKFError(
            f"concept {concept.id!r}: extra keys shadow known keys: {shadowed}"
        )
    front: dict[Any, Any] = {"type": concept.type}
    for key in ("title", "description", "resource"):
        if (value := getattr(concept, key)) is not None:
            front[key] = value
    if concept.tags:
        front["tags"] = list(concept.tags)
    if concept.sources:
        front["sources"] = concept.sources
    if "usage_window" in extra:
        front["usage_window"] = extra.pop("usage_window")
    if concept.generated is not None:
        front["generated"] = concept.generated
    if concept.verified:
        front["verified"] = concept.verified
    if concept.status is not None:
        front["status"] = concept.status
    if concept.stale_after is not None:
        front["stale_after"] = concept.stale_after
    front.update(extra)
    try:
        lone = has_lone_surrogate(front)  # before anything recurses over it
        plain = _plain(front)
        dumped = yaml.dump(
            plain,
            Dumper=_OKFDumper,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )
    except (yaml.YAMLError, RecursionError, OKFError) as exc:
        why = exc if isinstance(exc, OKFError) else type(exc).__name__
        raise OKFError(
            f"concept {concept.id!r}: frontmatter cannot be written ({why})"
        ) from exc
    body = concept.body + ("\n" if concept.body and concept.body[-1] != "\n" else "")
    if lone or _fm.SURROGATE_RE.search(body):  # YAML would escape it
        raise OKFError(f"concept {concept.id!r}: holds a lone surrogate (not UTF-8)")
    # DECISION plan-2026-10-21T140931-095ea5f8/D-018
    # The loader's bounds and measure: do NOT write what `load_bundle` rejects.
    data = f"---\n{dumped}---\n{body}".encode()
    if len(dumped.encode("utf-8", "replace")) > OKF_MAX_FRONTMATTER_BYTES:
        raise _FrontmatterTooLargeError(
            f"concept {concept.id!r}: frontmatter too large"
        )
    if len(data) > OKF_MAX_FILE_BYTES:
        raise OKFError(f"concept {concept.id!r}: file too large")
    return data


def writes_within_caps(concept: OKFConcept) -> bool:
    """False when ``write_bundle`` would refuse *concept* for frontmatter size, else
    True (any other refusal is left for the writer to name). Never raises."""
    try:
        _render_concept(concept)
    except OKFError as exc:  # any other refusal: the writer names it
        return not isinstance(exc, _FrontmatterTooLargeError)
    return True


def _render_index(bundle: OKFBundle) -> str:
    """Root ``index.md``: ``okf_version`` frontmatter, a section per type, cut at
    ``_INDEX_BUDGET`` bytes with a last ``* ... and N more concepts`` line."""
    rows: list[tuple[str, str]] = []  # (type heading, bullet)
    for concept in bundle:
        heading = strip_unsafe_text(concept.type).strip() or "Untyped"
        title = re.sub(r"([\\\[\]])", r"\\\1", concept.display_title)
        desc = strip_unsafe_text(concept.description or "").strip()
        line = f"* [{title}](/{quote(concept.id, safe='/')}.md)"
        rows.append(
            (heading, f"{line} - {desc[:OKF_SUMMARY_FIELD_CHARS]}" if desc else line)
        )
    rows.sort(key=lambda row: row[0])
    head = '---\nokf_version: "0.2"\n---\n'
    lines, used, prev = [], len(head), None
    for kept, (heading, bullet) in enumerate(rows):
        new = [bullet] if heading == prev else [f"# {heading}", "", bullet]
        if (used := used + sum(len(x.encode()) + 1 for x in new)) > _INDEX_BUDGET:
            lines.append(f"* ... and {len(rows) - kept} more concepts")
            break
        lines += new
        prev = heading
    return head + "".join(f"{x}\n" for x in lines)


def _target_problem(root: Path, rel: str) -> str | None:
    """Why ``root/rel`` may not be written, or ``None``. SECURITY:
    ``security.resolve_within`` is the only confinement check; the resolved path
    must EQUAL the lexical one (no symlink) and no ancestor may be a file."""
    lexical = root.joinpath(*rel.split("/"))
    try:
        if len(os.fsencode(lexical)) > _MAX_PATH_BYTES:
            return "is too long for the file system"
        resolved = resolve_within(root, rel)
        if resolved is None:
            return "resolves outside the target directory"
        # DECISION plan-2026-10-21T140931-095ea5f8/D-014
        # Do NOT relax this to the plain `resolve_within` containment test: a
        # symlinked directory INSIDE the root would then be written through.
        if resolved != lexical:
            return "passes through a symbolic link"
        if lexical.is_dir():
            return "is an existing directory"
        for ancestor in lexical.parents:
            if ancestor == root:
                break
            if ancestor.exists() and not ancestor.is_dir():
                return "has a file where a directory is needed"
    except (OSError, RuntimeError, ValueError):
        return "cannot be resolved"
    return None


# DECISION plan-2026-10-21T140931-095ea5f8/D-014
# Do NOT swap this private copy for a shared helper (the
# `agents/semantic_memory.py` anchor and the layering rule forbid it), and do
# NOT widen the cleanup to `except BaseException`.
def _atomic_write(path: Path, data: bytes) -> None:
    """Write *data* to *path* via a temp file and rename; removed on ANY failure."""
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".okf-tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        with suppress(OSError):  # gone after a successful rename
            os.unlink(tmp)


def write_bundle(
    bundle: OKFBundle,
    path: str | os.PathLike[str],
    *,
    overwrite: bool = False,
    write_index: bool = True,
) -> list[Path]:
    """Write *bundle* under *path* (created if missing); return the files written.

    Interface contract:
        - One ``<id>.md`` per concept: frontmatter in a fixed key order (type,
          title, description, resource, tags, sources, usage_window, generated,
          verified, status, stale_after, then ``extra`` in original order), the
          body, a trailing newline. Unknown keys survive and a second write of
          a loaded bundle is byte-identical; ``datetime`` values get an explicit
          UTC offset (naive is UTC), strings read from disk are verbatim.
        - Every id and target is validated and every file rendered AND encoded
          BEFORE the first byte is written: a bad or lookalike id, a lone
          surrogate, a frontmatter value too deep, large or cyclic
          (``OKF_MAX_YAML_DEPTH``, ``OKF_MAX_FRONTMATTER_BYTES`` nodes), a
          frontmatter or file over its byte cap, an over-long path, or a file
          and a directory sharing a path leave the directory untouched
          (``OKFError`` naming them, capped). What is written always loads back;
          the writer refuses what it cannot render within the caps and names
          the concept (a concept that loads clean near a cap can be refused: a
          re-rendered block scalar grows about 6%).
        - Each target passes ``security.resolve_within``; a symlinked target or
          directory component is refused even with ``overwrite=True``. An
          existing file is replaced only with ``overwrite=True``.
        - ``write_index`` adds a root ``index.md`` (``okf_version: "0.2"``, a
          bullet per concept per ``# <type>``, cut at ``_INDEX_BUDGET`` bytes
          with a last ``* ... and N more concepts``); kept if it exists, unless
          ``overwrite``.
        - Atomic PER FILE (temp file, fsync, rename), not per bundle: an I/O
          failure midway leaves the files already written, and a parent
          directory swapped for a symlink between validation and the write
          still wins.
    """
    try:
        root = Path(path).resolve()
        if root.exists() and not root.is_dir():
            raise OKFError(f"bundle root exists and is not a directory: {root}")
    except (OSError, RuntimeError, ValueError) as exc:
        raise OKFError(f"cannot open bundle root: {type(exc).__name__}") from exc

    problems: list[str] = []
    plan: list[tuple[str, Path, bytes]] = []  # (id, target, encoded file)
    seen: set[str] = set()
    for concept in bundle:
        cid = concept.id
        reason = _id_problem(cid)
        if reason is None and _id_key(cid) in seen:
            reason = "collides with another id on a case-insensitive disk"
        if reason is None:
            seen.add(_id_key(cid))
            reason = _target_problem(root, f"{cid}.md")
        if reason is None and not overwrite and os.path.lexists(root / f"{cid}.md"):
            reason = "already exists (pass overwrite=True to replace it)"
        if reason is not None:
            problems.append(f"{cid[:80]!r} {reason}")
            continue
        try:
            plan.append((cid, root / f"{cid}.md", _render_concept(concept)))
        except OKFError as exc:
            problems.append(str(exc))
    index = root / "index.md"
    if write_index and (overwrite or not os.path.lexists(index)):  # else user-owned
        if (reason := _target_problem(root, "index.md")) is not None:
            problems.append(f"'index.md' {reason}")
        else:
            plan.append(("index", index, _render_index(bundle).encode()))
    files = {_id_key(t.relative_to(root).as_posix()) for _, t, _ in plan}
    for cid, target, _ in plan:  # ids "q" and "q.md/z": a file and a directory
        parts = _id_key(target.relative_to(root).as_posix()).split("/")
        if any("/".join(parts[:i]) in files for i in range(1, len(parts))):
            problems.append(f"{cid[:80]!r} needs a directory where a file goes")
    if problems:
        shown = "; ".join(problems[:_MAX_CLASHES_SHOWN])
        more = len(problems) - _MAX_CLASHES_SHOWN
        raise OKFError(
            f"nothing written, {len(problems)} problem(s): {shown}"
            + (f"; and {more} more" if more > 0 else "")
        )

    written: list[Path] = []
    for cid, target, data in plan:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(target, data)
        except OSError as exc:
            raise OKFError(
                f"cannot write {cid[:80]!r}: {type(exc).__name__}; "
                f"{len(written)} file(s) were already written"
            ) from exc
        written.append(target)
    return sorted(written)
