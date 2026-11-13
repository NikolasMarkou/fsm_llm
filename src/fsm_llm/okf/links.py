"""
Concept ids and links, lexical and in id space: the one id rule of the loader and
the writer, the lookalike key, the body link reader and the link-target resolver.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator
from itertools import islice
from urllib.parse import unquote

from fsm_llm.text import _JOINERS, strip_unsafe_text

_Ids = list[str]
_RESERVED_STEMS = frozenset({"index", "log"})
_ID_MAX_SEGMENT_BYTES = 200


def _is_reserved(stem: str) -> bool:
    """``index`` and ``log`` in any case: spec 8 and 9 files, never concepts."""
    return stem.casefold() in _RESERVED_STEMS


def _id_key(concept_id: str) -> str:
    """Lookalike key of an id: NFC + casefold, joiners and soft hyphen removed."""
    plain = "".join(ch for ch in concept_id if ch not in _JOINERS)
    return unicodedata.normalize("NFC", plain).casefold()


def _id_problem(concept_id: object, *, allow_reserved: bool = False) -> str | None:
    """Why *concept_id* may not become a file path, or ``None``: the ONE id rule
    of the loader and the writer (the loader sets ``allow_reserved``)."""
    if not isinstance(concept_id, str) or not concept_id:
        return "empty or not a string"
    if strip_unsafe_text(concept_id) != concept_id:
        return "holds control or format characters"
    if concept_id.startswith("/") or "\\" in concept_id:
        return "absolute or holds a backslash"
    parts = concept_id.split("/")
    if any(p in ("", ".", "..") or p.startswith(".") for p in parts):
        return "has an empty, dot or dot-prefixed segment"
    if any(len(p.encode("utf-8")) > _ID_MAX_SEGMENT_BYTES for p in parts):
        return "has a segment that is too long"
    if not allow_reserved and _is_reserved(parts[-1]):
        return "is reserved (index and log are never concepts)"
    return None


_SCHEME_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*:")
_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")


# Every sub-pattern excludes the characters that open the next match, so a
# hostile body (a long run of ``[](``) is scanned in linear time.
_LINK_RE = re.compile(
    r"\[[^\]\[\n]{0,300}\]\(\s*(?:<([^<>\n]{1,2048})>|([^)\s\[(<]{1,2048}))"
    r"(?:\s+(?:\"[^\"\n]*\"|'[^'\n]*'))?\s*\)"
)


_MAX_LINKS_PER_BODY = 1_000


def _body_links(body: str) -> Iterator[str]:
    """Distinct raw link targets in the prose of *body* (fenced and inline code
    ignored); at most ``_MAX_LINKS_PER_BODY`` links are read."""
    prose: list[str] = []
    fence: str | None = None
    for line in body.split("\n"):
        if fence is None:
            if match := _FENCE_RE.match(line):
                fence = match.group(1)
            else:
                prose.append(_INLINE_CODE_RE.sub("", line))
        else:
            bare = line.strip()
            if bare and set(bare) == {fence[0]} and len(bare) >= len(fence):
                fence = None
    found = islice(_LINK_RE.finditer("\n".join(prose)), _MAX_LINKS_PER_BODY)
    yield from dict.fromkeys(m.group(1) or m.group(2) for m in found)


# DECISION plan-2026-10-21T140931-095ea5f8/D-019
# Do NOT memoize this (a module-level lru_cache): it held 121 MiB after the
# bundle was dropped and was no faster, the per-body dedupe does the work.
def _link_candidates(target: object, from_id: str) -> _Ids:
    """Concept ids a link or path-valued key may name, best first (D-003):
    ``/``-prefixed is root relative; else file-relative (spec 6.2), then root
    relative (6.3). Fragment and query dropped, percent-decoded once; external,
    non-``.md`` and above-root targets give nothing. Lexical only."""
    if not isinstance(target, str) or len(target) > 4096:
        return []
    path = re.split(r"[#?]", target.strip(), maxsplit=1)[0]
    if not path or path.startswith("//") or _SCHEME_RE.match(path):
        return []
    path = unquote(path)
    if "\\" in path or strip_unsafe_text(path) != path or not path.endswith(".md"):
        return []
    here = tuple(from_id.split("/")[:-1])
    found: _Ids = []
    for base in ((),) if path.startswith("/") else (here, ()):
        stack = list(base)
        for part in path.split("/"):
            if part in ("", "."):
                continue
            if part == "..":
                if not stack:
                    break  # climbs above the bundle root
                stack.pop()
            else:
                stack.append(part)
        else:
            if stack and stack[-1] != ".md" and not _is_reserved(stack[-1][:-3]):
                cid = "/".join(stack)[:-3]
                if cid not in found:
                    found.append(cid)
    return found
