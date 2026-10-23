"""
Open Knowledge Format (OKF): a directory of markdown files with YAML frontmatter.

The format layer: models, a tolerant parser, trust tier and staleness rules,
link graph and search, bounded loading, checks, writing and memory export. OKF
text is untrusted third-party prose: content never raises (a problem becomes an
``OKFIssue``), nothing a document names is executed, unknown keys are kept.

Spec status: OKF v0.2 at commit ad30107 of GoogleCloudPlatform/open-knowledge-format,
pre-1.0 and mutable. The trust tier is a CLAIM the bundle makes about itself
(a free ``verified[].by`` string), advisory only, never authorization.
"""

from __future__ import annotations

import difflib
import math
import os
import re
import tempfile
import time
import unicodedata
from collections.abc import Callable, Container, Iterable, Iterator, Mapping
from contextlib import suppress
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from functools import partial
from itertools import islice
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote, unquote

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from . import frontmatter as _fm
from .__version__ import __version__
from .constants import (
    OKF_EXPORT_TEXT_MAX_CHARS,
    OKF_INCOMPLETE_NOTE,
    OKF_INDEX_MAX_CHARS,
    OKF_MAX_DEPTH,
    OKF_MAX_FILE_BYTES,
    OKF_MAX_FILES,
    OKF_MAX_FRONTMATTER_BYTES,
    OKF_MAX_ISSUES,
    OKF_MAX_LOAD_SECONDS,
    OKF_MAX_TOTAL_BYTES,
    OKF_MAX_YAML_DEPTH,
    OKF_SUMMARY_FIELD_CHARS,
    OKF_SUMMARY_MAX_ITEMS,
    RESERVED_CONTEXT_KEYS,
)
from .definitions import OKFError
from .handlers import BaseHandler, HandlerTiming, create_handler
from .logging import logger
from .security import (
    has_internal_prefix,
    is_forbidden_context_entry,
    resolve_within,
)

# --- Untrusted-text helper (shared by every output path) ---
_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)?")
_LINE_BREAKS = "\n\r\t\u2028\u2029\x0b\x0c\x85"
# Cf characters that spell words (Persian and Indic ZWNJ, emoji ZWJ sequences,
# hyphenation hints). Kept only between two visible characters.
_JOINERS = "\u200c\u200d\u00ad"
_VISIBLE_CATEGORIES = "LMNPS"


def strip_unsafe_text(text: str, *, multiline: bool = False) -> str:
    """Remove what could drive a terminal or hide text: ANSI escapes, control
    (Cc) and format (Cf: zero-width, bidi, BOM) characters, except ZWNJ, ZWJ and
    soft hyphen between two visible characters (so ``می‌خواهم`` and emoji
    sequences survive). A line break or tab becomes one space, unless
    ``multiline`` keeps ``\\n`` and ``\\t``."""
    if text.isascii() and text.isprintable():
        return text  # nothing to strip: the common case, fast
    clean = _ANSI_RE.sub("", text)
    out: list[str] = []
    for i, ch in enumerate(clean):
        if multiline and ch in "\n\t":
            out.append(ch)
        elif ch in _LINE_BREAKS:
            out.append(" ")
        elif ch in _JOINERS:
            if 0 < i < len(clean) - 1 and all(
                unicodedata.category(n)[0] in _VISIBLE_CATEGORIES
                for n in (clean[i - 1], clean[i + 1])
            ):
                out.append(ch)
        elif unicodedata.category(ch) not in ("Cc", "Cf", "Cs"):
            out.append(ch)
    return "".join(out)


# --- Models ---
class TrustTier(str, Enum):
    """Advisory trust label derived from ``verified[].by`` (a claim, not auth)."""

    UNVERIFIED = "unverified"
    MACHINE_CONFIRMED = "machine-confirmed"
    HUMAN_REVIEWED = "human-reviewed"


class OKFIssue(BaseModel):
    """One problem found while reading or checking a bundle; ``message``,
    ``path`` and ``id`` are untrusted, so control-stripped and capped here."""

    model_config = ConfigDict(frozen=True)

    severity: Literal["error", "warning", "info"]
    code: str
    message: str
    path: str | None = None
    id: str | None = None

    @field_validator("message", "path", "id")
    @classmethod
    def _clean(cls, value: str | None) -> str | None:
        return None if value is None else strip_unsafe_text(value)[:300]


class OKFLimits(BaseModel):
    """Resource bounds for reading a bundle (defaults from ``constants.OKF_*``);
    ``max_seconds`` is a wall-clock budget (``load-deadline``, ``check-deadline``)."""

    model_config = ConfigDict(frozen=True)

    max_files: int = Field(OKF_MAX_FILES, gt=0)
    max_file_bytes: int = Field(OKF_MAX_FILE_BYTES, gt=0)
    max_total_bytes: int = Field(OKF_MAX_TOTAL_BYTES, gt=0)
    max_depth: int = Field(OKF_MAX_DEPTH, gt=0)
    max_seconds: float = Field(OKF_MAX_LOAD_SECONDS, gt=0, allow_inf_nan=False)


_TS_RE = re.compile(
    r"(\d{4})-(\d{2})-(\d{2})"
    r"(?:[Tt ](\d{2}):(\d{2})(?::(\d{2})(?:[.,](\d+))?)?"
    r"\s*(Z|z|[+-]\d{2}(?::?\d{2})?)?)?"
)


# The strict RFC 3339 / ISO 8601 profile: ``T`` or space, ``.`` fraction, ``Z``
# or ``+HH:MM`` offset, nothing lowercase. A stamp outside it gets a warning.
_STRICT_TS_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}"
    r"(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:\d{2})?)?"
)


# DECISION plan-2026-10-21T140931-095ea5f8/D-014
# Do NOT replace this parser with `datetime.fromisoformat`: on Python 3.10 it
# rejects a `Z` suffix and fractions that are not 3 or 6 digits, and OKF
# stamps use both. See decisions.md D-014.
def parse_okf_datetime(value: object) -> tuple[datetime | None, str | None]:
    """Parse an OKF timestamp: ``(aware datetime | None, warning code)``; the
    code is ``None`` (explicit offset: ``Z``, ``+HH``, ``+HH:MM``, ``+HHMM``),
    ``"date-only"`` (midnight UTC), ``"naive"`` (read as UTC) or ``"invalid"``."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc), "naive"
        return value, None
    if isinstance(value, date):
        midnight = datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
        return midnight, "date-only"
    match = _TS_RE.fullmatch(value.strip()) if isinstance(value, str) else None
    if match is None:
        return None, "invalid"
    year, month, day, hour, minute, second, frac, zone = match.groups()
    try:
        micro = int((frac or "0")[:6].ljust(6, "0"))
        tz: timezone | None = None
        if zone:
            if zone in ("Z", "z"):
                tz = timezone.utc
            else:
                digits = zone[1:].replace(":", "")
                delta = timedelta(hours=int(digits[:2]), minutes=int(digits[2:] or 0))
                tz = timezone(-delta if zone[0] == "-" else delta)
        nums = (year, month, day, hour or 0, minute or 0, second or 0)
        y, mo, d, h, mi, sec = map(int, nums)
        parsed = datetime(y, mo, d, h, mi, sec, micro)
    except ValueError:
        return None, "invalid"
    if hour is None:
        return parsed.replace(tzinfo=timezone.utc), "date-only"
    if tz is None:
        return parsed.replace(tzinfo=timezone.utc), "naive"
    return parsed.replace(tzinfo=tz), None


class OKFConcept(BaseModel):
    """One concept document; ``id`` is the file path minus ``.md``. Values are
    kept as read (timestamps stay strings, unknown keys stay in ``extra``);
    ``type`` is ``""`` when the document gave none (an error issue)."""

    id: str
    type: str = ""
    title: str | None = None
    description: str | None = None
    resource: str | None = None
    tags: list[str] = Field(default_factory=list)
    sources: list[dict[Any, Any]] = Field(default_factory=list)
    generated: dict[Any, Any] | None = None
    verified: list[dict[Any, Any]] = Field(default_factory=list)
    status: str | None = None
    stale_after: str | datetime | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
    body: str = ""

    @property
    def trust(self) -> TrustTier:
        """``verified[].by``: any ``human:`` -> human-reviewed, else any -> machine."""
        verifiers = [
            by.strip()
            for entry in self.verified
            if isinstance(by := entry.get("by"), str) and by.strip()
        ]
        if any(by.startswith("human:") for by in verifiers):
            return TrustTier.HUMAN_REVIEWED
        return TrustTier.MACHINE_CONFIRMED if verifiers else TrustTier.UNVERIFIED

    def is_stale(self, now: datetime | None = None) -> bool:
        """True when ``now >= stale_after``; absent or unparseable is not stale."""
        deadline, _ = parse_okf_datetime(self.stale_after)
        if deadline is None:
            return False
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        return now >= deadline

    @property
    def is_deprecated(self) -> bool:
        return (self.status or "").strip().lower() == "deprecated"

    @property
    def is_draft(self) -> bool:
        return (self.status or "").strip().lower() == "draft"

    def flags(self, now: datetime | None = None) -> list[str]:
        """Advisory flags in a fixed order: ``stale``, ``draft``, ``deprecated``."""
        state = {
            "stale": self.is_stale(now),
            "draft": self.is_draft,
            "deprecated": self.is_deprecated,
        }
        return [name for name, on in state.items() if on]

    @property
    def display_title(self) -> str:
        """Control-stripped title, else the last id segment."""
        title = strip_unsafe_text(self.title or "").strip()
        return title or strip_unsafe_text(self.id.rsplit("/", 1)[-1])


# --- Tolerant parser ---
_KNOWN_KEYS = frozenset(
    {"type", "title", "description", "resource", "tags", "sources"}
    | {"generated", "verified", "status", "stale_after"}
)
_STATUSES = ("draft", "stable", "deprecated")
_MAX_FILE_WARNINGS = 25


# The one frontmatter seam is ``frontmatter.py``. The private names below stay
# module globals of this file (tests patch them); OKF passes its own limits.
_Note = _fm.Note
_SURROGATE_RE = _fm.SURROGATE_RE
_split_frontmatter = _fm.split_frontmatter
_read_bytes = _fm.read_bytes
_load_yaml = partial(
    _fm.load_yaml, max_bytes=OKF_MAX_FRONTMATTER_BYTES, max_depth=OKF_MAX_YAML_DEPTH
)


def _text(data: dict[str, Any], key: str, note: _Note) -> str | None:
    value = data.get(key)
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (bool, int, float, date)):
        note("warning", "value-coerced", f"{key!r} is not a string; read as text")
        return str(value)
    note("warning", "value-invalid", f"{key!r} must be a string; ignored")
    return None


def _tags(value: Any, note: _Note) -> list[str]:
    if isinstance(value, str):
        note("warning", "tags-string", "'tags' is a string; split on commas")
        items: list[Any] = value.split(",")
    elif isinstance(value, list):
        items = value
    else:
        if value is not None:
            note("warning", "tags-invalid", "'tags' must be a list of strings")
        return []
    if any(i is not None and not isinstance(i, str) for i in items):
        note("warning", "value-coerced", "'tags' holds non-strings; read as text")
    return [s for i in items if i is not None and (s := str(i).strip())]


def _mapping_list(data: dict[str, Any], key: str, note: _Note) -> list[dict[Any, Any]]:
    """A list of mappings; a bare mapping is a one-element list (spec MUST for
    ``verified``). Other entries are dropped with a warning."""
    value = data.get(key)
    if value is None:
        return []
    items = [value] if isinstance(value, dict) else value
    if not isinstance(items, list):
        note("warning", f"{key}-invalid", f"{key!r} must be a list of mappings")
        return []
    kept = [item for item in items if isinstance(item, dict)]
    if len(kept) != len(items):
        note("warning", f"{key}-invalid", f"{key!r} entries that are not mappings")
    return kept


def _check_timestamp(label: str, value: Any, note: _Note) -> None:
    code = parse_okf_datetime(value)[1]
    if code == "invalid":
        note("warning", "timestamp-invalid", f"{label} is not a timestamp: {value!r}")
    elif code:
        note("warning", f"timestamp-{code}", f"{label} lacks an explicit UTC offset")
    if (
        code != "invalid"
        and isinstance(value, str)
        and not _STRICT_TS_RE.fullmatch(value.strip())
    ):
        note(
            "warning",
            "timestamp-nonstandard",
            f"{label} is not strict RFC 3339 (read leniently): {value!r}",
        )


def _check_timestamps(
    stale_after: Any, generated: Any, verified: Any, note: _Note
) -> None:
    """Check ``stale_after``, ``generated.at`` and every ``verified[].at``."""
    stamps = [
        ("stale_after", stale_after),
        ("generated.at", (generated or {}).get("at")),
    ]
    stamps += [("verified.at", entry.get("at")) for entry in verified]
    for label, value in stamps:
        if value is not None:
            _check_timestamp(label, value, note)


def parse_concept_text(concept_id: str, text: str) -> tuple[OKFConcept, list[OKFIssue]]:
    """Parse one concept document into ``(concept, issues)``; never raises on
    content. Without usable frontmatter the text is the body."""
    issues: list[OKFIssue] = []
    warnings = 0

    def note(
        severity: Literal["error", "warning", "info"], code: str, msg: str
    ) -> None:
        nonlocal warnings
        if severity == "warning":  # bound the work a hostile document can cause
            warnings += 1
            if warnings > _MAX_FILE_WARNINGS:
                if warnings == _MAX_FILE_WARNINGS + 1:
                    severity, code = "info", "issues-truncated"
                    msg = "further warnings for this concept are not shown"
                else:
                    return
        issues.append(
            OKFIssue(severity=severity, code=code, message=msg, id=concept_id)
        )

    try:
        concept = _build_concept(concept_id, text, note)
    except Exception as exc:  # the parser is total on content
        logger.debug(f"OKF concept {concept_id!r} unreadable: {exc!r}")
        note("error", "frontmatter-invalid", f"{type(exc).__name__}: {exc}"[:160])
        concept = OKFConcept(id=concept_id, body=text)
    return concept, issues


def _build_concept(concept_id: str, text: str, note: _Note) -> OKFConcept:
    raw, body, problem = _split_frontmatter(text)
    if problem:
        note("error", problem, f"no complete frontmatter block ({problem})")
    loaded = _load_yaml(raw, note) if raw is not None else {}
    data: dict[str, Any] = {}
    for key, value in loaded.items():
        if not isinstance(key, str):
            note("warning", "key-not-string", f"non-string key {key!r} read as text")
        data[str(key)] = value

    type_text = _text(data, "type", note)
    if "type" not in data:
        note("error", "type-missing", "'type' is required")
    elif type_text is None or not type_text.strip():
        note("error", "type-empty", "'type' must be a non-empty string")
    status = _text(data, "status", note)
    if status is not None and status.strip().lower() not in _STATUSES:
        note("warning", "status-unknown", f"unknown status {status!r}")
    generated = data.get("generated")
    if generated is not None and not isinstance(generated, dict):
        note("warning", "generated-invalid", "'generated' must be a mapping")
        generated = None
    verified = _mapping_list(data, "verified", note)
    stale_after = _text(data, "stale_after", note)

    _check_timestamps(stale_after, generated, verified, note)
    return OKFConcept(
        id=concept_id,
        type=(type_text or "").strip(),
        title=_text(data, "title", note),
        description=_text(data, "description", note),
        resource=_text(data, "resource", note),
        tags=_tags(data.get("tags"), note),
        sources=_mapping_list(data, "sources", note),
        generated=generated,
        verified=verified,
        status=status,
        stale_after=stale_after,
        extra={k: v for k, v in data.items() if k not in _KNOWN_KEYS},
        body=body,
    )


# --- Links (lexical, in id space) ---
_Ids = list[str]
_Concepts = list[OKFConcept]
_Hits = list[tuple[OKFConcept, float]]

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
_MAX_LINK_ISSUES = 20  # link-broken issues per concept
_HAY_FIELDS = ((3, 0), (2, 1), (1, 2))  # (weight, index into the haystack)
_HAY_WEIGHT = sum(weight for weight, _ in _HAY_FIELDS)  # best score per token


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


# --- Bundle ---
_CHECK_DEADLINE = OKFIssue(  # frozen, so one instance serves every bundle
    severity="error",
    code="check-deadline",
    message="check time limit (max_seconds) reached; the rest was not checked",
)


class OKFBundle:
    """Concepts by id, plus the link graph and lexical search over them.

    Interface contract:
        - Build with ``OKFBundle(concepts, root=, issues=, okf_version=,
          index_text=, max_seconds=)`` or ``load_bundle``; ``root`` is ``None``
          for an in-memory bundle. ``index_text`` is control-stripped and
          capped. ``max_seconds`` (``None``: unbounded) bounds ``check_bundle``
          and the first ``backlinks`` call, each on its own clock.
        - ``get(id)`` -> concept or ``None``; ``len``, ``in`` and iteration
          (id order) work as for a mapping of ids; ``list(type=, prefix=,
          include_deprecated=False)`` -> concepts in id order (``type`` is
          compared case-insensitively).
        - ``search(query, limit=5, include_deprecated=False)`` -> ``(concept,
          score)`` best first, score in (0, 1], ties by id; lexical, tokens of
          3+ characters; ``[]`` for a non-string or token-less query.
        - ``resolve_link(target, from_id)`` -> the id of an EXISTING concept
          or ``None`` (external, escaping, broken and non-concept targets all
          give ``None``); it also serves path-valued keys such as ``resource``.
        - ``outgoing(id)`` / ``backlinks(id)`` -> ids in order, no self-links;
          outgoing covers body links, ``resource`` and ``sources[].resource``.
          A ``backlinks`` build past ``max_seconds`` adds one ``check-deadline``
          issue and answers from the part built; a finished build removes it.
        - ``incomplete`` -> the ``INCOMPLETE_CODES`` in ``issues`` (a load
          stop); ``put(concept)`` adds or replaces and invalidates every cache.
        Nothing here raises on bundle content or touches the filesystem.
    """

    def __init__(
        self,
        concepts: Iterable[OKFConcept] = (),
        *,
        root: Path | None = None,
        issues: Iterable[OKFIssue] = (),
        okf_version: str | None = None,
        index_text: str = "",
        max_seconds: float | None = None,
    ) -> None:
        self.root = root
        self.max_seconds = max_seconds
        self.issues: list[OKFIssue] = list(issues)
        self.okf_version = okf_version
        self.index_text = strip_unsafe_text(index_text, multiline=True)[
            :OKF_INDEX_MAX_CHARS
        ]
        self._concepts: dict[str, OKFConcept] = {}
        self._reset()
        for concept in concepts:
            self.put(concept)

    @property
    def incomplete(self) -> tuple[str, ...]:
        seen = {i.code for i in self.issues}
        return tuple(c for c in INCOMPLETE_CODES if c in seen)

    def _deadline(self) -> float:
        return time.monotonic() + (self.max_seconds or math.inf)

    def _reset(self) -> None:
        self._ids: _Ids | None = None
        self._out: dict[str, _Ids] = {}
        self._back: dict[str, _Ids] | None = None
        self._hay: dict[str, tuple[str, str, str]] = {}

    def put(self, concept: OKFConcept) -> None:
        self._concepts[concept.id] = concept
        self._reset()

    def __len__(self) -> int:
        return len(self._concepts)

    def __contains__(self, concept_id: object) -> bool:
        return concept_id in self._concepts

    def _sorted_ids(self) -> _Ids:
        if self._ids is None:
            self._ids = sorted(self._concepts)
        return self._ids

    def __iter__(self) -> Iterator[OKFConcept]:
        return (self._concepts[cid] for cid in self._sorted_ids())

    def get(self, concept_id: str) -> OKFConcept | None:
        return self._concepts.get(concept_id)

    def list(
        self,
        type: str | None = None,
        prefix: str | None = None,
        include_deprecated: bool = False,
    ) -> _Concepts:
        wanted = type.strip().casefold() if type else None
        return [
            c
            for c in self
            if (include_deprecated or not c.is_deprecated)
            and (wanted is None or c.type.casefold() == wanted)
            and (not prefix or c.id.startswith(prefix))
        ]

    def search(
        self, query: str, limit: int = 5, include_deprecated: bool = False
    ) -> _Hits:
        if not isinstance(query, str) or limit < 1:
            return []
        found = re.findall(r"\w{3,}", query[:1000].casefold())
        tokens = list(dict.fromkeys(found))[:20]
        hits: _Hits = []
        for concept in self:
            if tokens and (include_deprecated or not concept.is_deprecated):
                hay = self._haystack(concept)
                score = sum(
                    weight
                    for token in tokens
                    for weight, i in _HAY_FIELDS
                    if token in hay[i]
                )
                if score:
                    hits.append((concept, score / (_HAY_WEIGHT * len(tokens))))
        hits.sort(key=lambda hit: (-hit[1], hit[0].id))
        return hits[:limit]

    def _haystack(self, c: OKFConcept) -> tuple[str, str, str]:
        if c.id not in self._hay:
            self._hay[c.id] = (
                f"{c.title or ''}\n{' '.join(c.tags)}".casefold(),
                f"{c.description or ''}\n{c.id}".casefold(),
                c.body.casefold(),
            )
        return self._hay[c.id]

    def resolve_link(self, target: object, from_id: str) -> str | None:
        return next(
            (c for c in _link_candidates(target, from_id) if c in self._concepts),
            None,
        )

    def outgoing(self, concept_id: str) -> _Ids:
        concept = self._concepts.get(concept_id)
        if concept is None:
            return []
        if concept_id not in self._out:
            refs = [
                *_body_links(concept.body),
                concept.resource,
                *(s.get("resource") for s in concept.sources),
            ]
            found = dict.fromkeys(
                t for r in refs if (t := self.resolve_link(r, concept_id))
            )
            found.pop(concept_id, None)
            self._out[concept_id] = list(found)
        return list(self._out[concept_id])

    def backlinks(self, concept_id: str) -> _Ids:
        if self._back is None:
            back: dict[str, _Ids] = {}
            deadline = self._deadline()
            for cid in self._sorted_ids():
                if time.monotonic() > deadline:
                    if not any(i.code == "check-deadline" for i in self.issues):
                        self.issues.append(_CHECK_DEADLINE)
                    return list(back.get(concept_id, []))
                for target in self.outgoing(cid):
                    back.setdefault(target, []).append(cid)
            self._back = back
            self.issues[:] = [i for i in self.issues if i.code != "check-deadline"]
        return list(self._back.get(concept_id, []))


# --- Loader ---
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


INCOMPLETE_CODES = ("max-total-bytes", "max-files", "load-deadline", "check-deadline")


def incomplete_note(bundle: OKFBundle) -> str:
    """The fixed note for a partial bundle (codes only, no issue text); else ``""``."""
    codes = bundle.incomplete
    return OKF_INCOMPLETE_NOTE.format(codes=", ".join(codes)) if codes else ""


def capped_issues(
    issues: list[Any], cap: int, stops: Container[str], kind: Callable[..., Any]
) -> list[Any]:
    """At most *cap* issues (*stops* codes first), then one ``issues-truncated``."""
    if len(issues) <= cap:
        return issues
    rank = ("error", "warning", "info")
    order = sorted(issues, key=lambda i: (i.code not in stops, rank.index(i.severity)))
    extra = f"{len(issues) - cap} more issues are not shown"
    return [*order[:cap], kind(severity="info", code="issues-truncated", message=extra)]


def _capped(issues: list[OKFIssue]) -> list[OKFIssue]:
    return capped_issues(issues, OKF_MAX_ISSUES, INCOMPLETE_CODES, OKFIssue)


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
        note: _Note = lambda s, c, m: issues.append(  # noqa: E731
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


# --- Conformance (spec section 11) ---
def check_bundle(bundle: OKFBundle) -> list[OKFIssue]:
    """Spec section 11 conformance as a list of issues. Never raises.

    The load-time issues plus checks that also cover concepts added with
    ``put``: non-empty ``type``, no reserved id, timestamps without an explicit
    UTC offset, unknown ``okf_version`` (info), broken links (info; at most
    ``_MAX_LINK_ISSUES`` per concept and twice ``OKF_MAX_ISSUES`` in all, then
    one ``issues-truncated``). Duplicates dropped, result capped as in
    ``load_bundle``; it stops at ``max_seconds`` with ``check-deadline``.
    """
    out = list(bundle.issues)
    seen = {(i.code, i.path, i.id, i.message) for i in out}
    typed = {i.id for i in out if i.code in ("type-missing", "type-empty")}

    def add(
        severity: Literal["error", "warning", "info"],
        code: str,
        message: str,
        cid: str | None = None,
    ) -> None:
        issue = OKFIssue(severity=severity, code=code, message=message, id=cid)
        if (key := (code, None, issue.id, issue.message)) not in seen:
            seen.add(key)
            out.append(issue)

    deadline, broken, cut = bundle._deadline(), 0, False
    if (version := bundle.okf_version) not in (None, "0.1", "0.2"):
        add("info", "okf-version-unknown", f"okf_version {version!r} unknown")
    for c in bundle:
        if time.monotonic() > deadline:
            out.append(_CHECK_DEADLINE)
            break
        note = partial(add, cid=c.id)

        if _is_reserved(c.id.rsplit("/", 1)[-1]):
            note("error", "reserved-id", "index and log are reserved, never concepts")
        if not c.type.strip() and c.id not in typed:
            note("error", "type-empty", "'type' must be a non-empty string")
        _check_timestamps(c.stale_after, c.generated, c.verified, note)
        first = broken
        for target in _body_links(c.body):
            if (cands := _link_candidates(target, c.id)) and not any(
                x in bundle for x in cands
            ):
                if broken - first >= _MAX_LINK_ISSUES or broken >= 2 * OKF_MAX_ISSUES:
                    cut = True
                    break
                broken += 1
                note("info", "link-broken", f"link target {target!r} not in the bundle")
    if cut:
        add("info", "issues-truncated", "more broken links are not shown")
    return _capped(out)


# --- Writer ---
_INDEX_BUDGET = OKF_MAX_FILE_BYTES * 3 // 4  # room for the last line, so it loads back
_MAX_CLASHES_SHOWN = 5
_MAX_PATH_BYTES = 4_000  # Linux PATH_MAX is 4,096; room left for the temp sibling


class _OKFDumper(yaml.SafeDumper):
    """``SafeDumper`` without anchors or aliases (the loader refuses them)."""

    def ignore_aliases(self, data: Any) -> bool:
        return True


def _iso(value: datetime) -> str:
    """ISO 8601 with an explicit UTC offset; a naive value is taken as UTC."""
    return (value if value.tzinfo else value.replace(tzinfo=timezone.utc)).isoformat()


def _plain(value: Any) -> Any:
    """Copy *value* for dumping; datetimes and dates become offset-bearing text."""
    if isinstance(value, datetime):
        return _iso(value)
    if isinstance(value, date):
        return _iso(datetime(value.year, value.month, value.day))
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _lone_surrogate(value: Any) -> bool:
    """True when a string, key or item in *value* is a lone surrogate; ``OKFError``
    past ``OKF_MAX_YAML_DEPTH`` (a cycle ends there) or ``OKF_MAX_FRONTMATTER_BYTES``
    nodes, surrogate or not."""
    try:
        return _fm.lone_surrogate(
            value, max_depth=OKF_MAX_YAML_DEPTH, max_nodes=OKF_MAX_FRONTMATTER_BYTES
        )
    except RecursionError as exc:
        raise OKFError(str(exc)) from exc


class _FrontmatterTooLarge(OKFError):
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
        lone = _lone_surrogate(front)  # before anything recurses over it
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
    if lone or _SURROGATE_RE.search(body):  # YAML would escape it
        raise OKFError(f"concept {concept.id!r}: holds a lone surrogate (not UTF-8)")
    # DECISION plan-2026-10-21T140931-095ea5f8/D-018
    # The loader's bounds and measure: do NOT write what `load_bundle` rejects.
    data = f"---\n{dumped}---\n{body}".encode()
    if len(dumped.encode("utf-8", "replace")) > OKF_MAX_FRONTMATTER_BYTES:
        raise _FrontmatterTooLarge(f"concept {concept.id!r}: frontmatter too large")
    if len(data) > OKF_MAX_FILE_BYTES:
        raise OKFError(f"concept {concept.id!r}: file too large")
    return data


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
# Do NOT swap this private copy for a shared helper or `harness._atomic` (the
# `agents/semantic_memory.py` anchor and the layering rule forbid both), and do
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


# --- Export of memory entries ---
_DROP = object()
_MAX_META_ITEMS = 100
_MAX_META_DEPTH = 4
_MAX_META_TEXT = 2_000
_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _native(
    value: Any, cuts: set[str], depth: int = 0, left: list[int] | None = None
) -> Any:
    """JSON-native copy of *value* or ``_DROP``; secret-looking and internal
    keys are removed. Each real cut adds "depth", "items", "size" (node budget),
    "text" or "values" to *cuts*; a replaced lone surrogate adds "surrogates"."""
    # DECISION plan-2026-10-21T140931-095ea5f8/D-018
    # A quarter of the writer's node cap (it counts a mapping item twice), so one
    # hostile entry never fails a whole export. Do NOT raise it or drop it.
    left = [OKF_MAX_FRONTMATTER_BYTES // 4] if left is None else left
    left[0] -= 1
    if left[0] < 0:
        cuts.add("size")
        return _DROP
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):  # an IntEnum is not representable as YAML
        return int(value)
    if isinstance(value, float):
        if math.isfinite(value):
            return float(value)
    elif isinstance(value, str):
        kept = value[:_MAX_META_TEXT]
        clean, replaced = _SURROGATE_RE.subn("\ufffd", kept)
        if len(kept) < len(value):
            cuts.add("text")
        if replaced:
            cuts.add("surrogates")
        return clean
    elif isinstance(value, (list, tuple, dict)):
        if depth >= _MAX_META_DEPTH:
            cuts.add("depth")
            return _DROP
        if len(value) > _MAX_META_ITEMS:
            cuts.add("items")
        if isinstance(value, dict):
            out: dict[str, Any] = {}
            for key, item in islice(value.items(), _MAX_META_ITEMS):
                if not isinstance(key, str):
                    cuts.add("values")
                elif (
                    not has_internal_prefix(key)
                    and not is_forbidden_context_entry(key, item)
                    and (clean := _native(item, cuts, depth + 1, left)) is not _DROP
                ):
                    out[_native(key, cuts)] = clean
            return out
        kids = (
            _native(v, cuts, depth + 1, left) for v in islice(value, _MAX_META_ITEMS)
        )
        return [v for v in kids if v is not _DROP]
    cuts.add("values")
    return _DROP


def capped_line(value: object, cap: int = OKF_SUMMARY_FIELD_CHARS) -> str:
    """Control-stripped, trimmed, capped one-line text; ``""`` for a non-string."""
    return strip_unsafe_text(value).strip()[:cap] if isinstance(value, str) else ""


def text_page(text: str, offset: Any, size: int) -> tuple[str, int | None, int]:
    """``(page, next offset or None, length)`` of *text*, which the caller has
    already control-stripped; ``offset`` is clamped into the text (a non-int is
    0) and pages add up to it. Shared by OKF bodies and Agent Skills files."""
    try:
        start = min(max(int(offset), 0), len(text))
    except (TypeError, ValueError, OverflowError):
        start = 0
    end = min(start + size, len(text))
    return text[start:end], (end if end < len(text) else None), len(text)


def body_page(
    concept: OKFConcept, offset: Any, size: int
) -> tuple[str, int | None, int]:
    """``text_page`` of a concept's stripped body."""
    return text_page(strip_unsafe_text(concept.body, multiline=True), offset, size)


def close_ids(
    bundle: OKFBundle, wanted: str, *, include_deprecated: bool = False, n: int = 5
) -> list[str]:
    """Up to *n* ids of *bundle* close to *wanted* (the not-found hint)."""
    ids = [c.id for c in bundle.list(include_deprecated=include_deprecated)]
    return difflib.get_close_matches(wanted[:200], ids, n=n)


def _slug(raw: object) -> str:
    slug = _SLUG_RE.sub("-", str(raw)[:200]).strip("-.")[:80].strip("-.") or "entry"
    return f"entry-{slug}" if slug.casefold() in ("index", "log") else slug


def _loads_back(concept: OKFConcept) -> bool:
    """False when the writer would refuse *concept* for frontmatter size."""
    try:
        _render_concept(concept)
    except OKFError as exc:  # any other refusal: the writer names it
        return not isinstance(exc, _FrontmatterTooLarge)
    return True


def bundle_from_memory_entries(
    entries: Iterable[Any],
    *,
    generated_by: str | None = None,
    now: datetime | str | None = None,
) -> OKFBundle:
    """Turn memory entries (plain mappings) into an in-memory ``OKFBundle``.

    Interface contract:
        - ``entries``: mappings shaped like ``SemanticMemoryStore`` entries;
          only ``id``, ``text`` and ``metadata`` are read, embeddings NEVER. A
          non-mapping entry or a missing, blank or non-string ``text`` is
          skipped with a warning ``OKFIssue``; nothing raises.
        - Concept id ``memory/<slug of the entry id>`` (characters outside
          ``A-Za-z0-9._-`` become ``-``, at most 80; a repeated slug gets
          ``-2``, ``-3``), so a hostile id cannot shape the path.
        - ``type`` is metadata ``type`` else ``Memory``; ``title`` is metadata
          ``title`` else the first non-blank line (capped at
          ``OKF_SUMMARY_FIELD_CHARS``); ``tags`` are the string items of a
          metadata ``tags`` list. The body is the text (cut at
          ``OKF_EXPORT_TEXT_MAX_CHARS`` with a warning). A lone surrogate in the
          text, id or metadata becomes U+FFFD (``text-sanitized``).
        - The other metadata goes under ``extra["metadata"]`` as JSON-native
          data cut at depth 4, 100 items per level, 2,000 characters per string
          and 4,096 nodes: ``int``, ``float`` and ``str`` subclasses
          (``IntEnum``) become their base type, tuples lists; any other value
          (``Decimal``, ``UUID``, ``datetime``, ``set``, ``bytes``) and
          non-string keys are dropped. If the frontmatter would still pass
          ``OKF_MAX_FRONTMATTER_BYTES`` bytes, ``extra["metadata"]`` goes
          whole, then the tags. ONE ``metadata-truncated`` warning per entry
          names the cuts (depth, items, size, text, values). Secret or
          internal-prefixed keys are dropped at any depth, silently.
        - ``generated`` is ``{by: generated_by or "fsm-llm/<version>", at: now
          or the current time}`` (explicit UTC offset); no ``verified``, so
          every export is ``unverified``.
    """
    issues: list[OKFIssue] = []
    concepts: list[OKFConcept] = []
    used: set[str] = set()
    stamp = _iso(now) if isinstance(now, datetime) else capped_line(now, 100)
    generated = {
        "by": capped_line(generated_by, 200) or f"fsm-llm/{__version__}",
        "at": stamp or _iso(datetime.now(timezone.utc)),
    }

    def skip(index: int, why: str, code: str = "entry-skipped") -> None:
        msg = f"memory entry {index}: {why}"
        issues.append(OKFIssue(severity="warning", code=code, message=msg))

    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            skip(index, "skipped, not a mapping")
            continue
        text = entry.get("text")
        if not isinstance(text, str) or not text.strip():
            skip(index, "skipped, text is missing, blank or not a string")
            continue
        if len(text) > OKF_EXPORT_TEXT_MAX_CHARS:
            text = text[:OKF_EXPORT_TEXT_MAX_CHARS]
            skip(index, "text was cut", "text-truncated")
        raw_meta, notes = entry.get("metadata"), set[str]()
        meta = _native(dict(raw_meta), notes) if isinstance(raw_meta, Mapping) else {}
        ident = i if isinstance(i := entry.get("id", ""), (str, int, float)) else ""
        if "surrogates" in notes or _lone_surrogate((text, ident)):
            text = _SURROGATE_RE.sub("\ufffd", text)
            skip(index, "lone surrogates replaced with U+FFFD", "text-sanitized")
        type_, title, tags = (meta.pop(k, None) for k in ("type", "title", "tags"))
        first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
        base = _slug(ident)
        slug, n = base, 1
        while slug.casefold() in used:
            n += 1
            slug = f"{base}-{n}"
        used.add(slug.casefold())
        concept = OKFConcept(
            id=f"memory/{slug}",
            type=capped_line(type_, 100) or "Memory",
            title=capped_line(title if isinstance(title, str) else first_line) or None,
            tags=[t for x in tags if (t := capped_line(x, 100))]
            if isinstance(tags, list)
            else [],
            generated=dict(generated),
            extra={"metadata": meta} if meta else {},
            body=text,
        )
        if not _loads_back(concept):  # drop metadata, then tags (D-018)
            notes.add("size")
            concept.extra.clear()
            if not _loads_back(concept):
                concept.tags = []
        concepts.append(concept)
        if cut := sorted(notes - {"surrogates"}):
            skip(index, f"metadata cut ({', '.join(cut)})", "metadata-truncated")
    return OKFBundle(concepts, issues=issues)


# --- Plain-FSM handler ---
def create_okf_handler(
    bundle: OKFBundle,
    *,
    context_key: str = "okf_concepts",
    timing: HandlerTiming = HandlerTiming.PRE_PROCESSING,
    states: Iterable[str] = (),
    query_key: str | None = None,
    max_items: int = OKF_SUMMARY_MAX_ITEMS,
    include_deprecated: bool = False,
    name: str = "OKFContext",
) -> BaseHandler:
    """Build a handler that writes capped concept summaries into the context.

    Interface contract:
        - Writes ``context_key`` = a list of at most ``max_items`` dicts
          ``{id, type, title, description, trust_claimed, flags}`` (JSON-native;
          strings control-stripped and cut at ``OKF_SUMMARY_FIELD_CHARS``;
          ``flags`` sorted from stale, draft, deprecated). NEVER a body.
        - Concepts: with ``query_key`` set and a non-blank ``str`` there (cut
          to 1,000 chars), the best ``bundle.search`` hits; else the first
          ``max_items`` by id. Deprecated ones only with ``include_deprecated``.
          No ``states`` means all. Not critical: a failure is logged.
        - Raises ``OKFError`` at build time for a blank, internal-prefixed,
          secret-looking (for a string AND the list written) or reserved
          ``context_key``; ``max_items`` outside ``1..OKF_SUMMARY_MAX_ITEMS``
          (never clamped); bad ``states``, ``timing`` or ``query_key`` types.
        - ``PRE_PROCESSING`` (default) runs on every turn, a BLOCKED one too.
    """
    if not isinstance(context_key, str) or not context_key.strip():
        raise OKFError(f"context_key must be a non-blank str, got {context_key!r}")
    if (
        has_internal_prefix(context_key)
        or is_forbidden_context_entry(context_key, "x")
        or is_forbidden_context_entry(context_key, [])
        or context_key in RESERVED_CONTEXT_KEYS
    ):
        raise OKFError(
            f"context_key {context_key!r} is hidden from prompts or reserved; "
            "use a plain public name"
        )
    if not isinstance(max_items, int) or isinstance(max_items, bool):
        raise OKFError(f"max_items must be an int, got {max_items!r}")
    if not 1 <= max_items <= OKF_SUMMARY_MAX_ITEMS:
        raise OKFError(f"max_items must be 1..{OKF_SUMMARY_MAX_ITEMS}, got {max_items}")
    if not isinstance(timing, HandlerTiming):
        raise OKFError(f"timing must be a HandlerTiming, got {timing!r}")
    if isinstance(states, (str, bytes)) or not isinstance(states, Iterable):
        raise OKFError(f"states must be an iterable of str, got {states!r}")
    state_names = tuple(states)
    if not all(isinstance(s, str) for s in state_names):
        raise OKFError(f"states must hold str names, got {state_names!r}")
    if query_key is not None and (not isinstance(query_key, str) or not query_key):
        raise OKFError(f"query_key must be None or a non-blank str, got {query_key!r}")

    def summarize(ctx: dict[str, Any]) -> dict[str, Any]:
        raw = ctx.get(query_key) if query_key is not None else None
        query = strip_unsafe_text(raw[:1000]).strip() if isinstance(raw, str) else ""
        if query:
            found = [c for c, _ in bundle.search(query, max_items, include_deprecated)]
        else:
            found = bundle.list(include_deprecated=include_deprecated)[:max_items]
        now = datetime.now(timezone.utc)
        return {
            context_key: [
                {
                    "id": capped_line(c.id),
                    "type": capped_line(c.type),
                    "title": capped_line(c.display_title),
                    "description": capped_line(c.description),
                    "trust_claimed": c.trust.value,
                    "flags": sorted(c.flags(now)),
                }
                for c in found
            ]
        }

    return create_handler(name).at(timing).on_state(*state_names).do(summarize)
