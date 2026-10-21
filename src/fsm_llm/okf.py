"""
Open Knowledge Format (OKF): a directory of markdown files with YAML frontmatter.

This module is the format layer: models, a tolerant frontmatter parser, the
trust tier and staleness rules. OKF text is untrusted third-party prose. The
parser never raises on content (a problem becomes an ``OKFIssue``), never
executes anything the document names, and keeps every unknown key so a bundle
can be written back unchanged.

Spec status: OKF v0.2 at commit ad30107 of GoogleCloudPlatform/open-knowledge-format,
pre-1.0 and mutable. The trust tier is a CLAIM the bundle makes about itself
(a free ``verified[].by`` string), advisory only, never authorization.

Entry points: ``parse_concept_text(concept_id, text) -> (OKFConcept, issues)``,
``load_bundle(path) -> OKFBundle`` (bounded, confined, never follows symlinks),
``OKFBundle`` (lookup, lexical search, link graph), ``check_bundle(bundle)``
(spec section 11 conformance as a list of issues), ``write_bundle(bundle,
path)`` (atomic per file, confined) and ``bundle_from_memory_entries(entries)``
(plain memory-entry mappings to a bundle), ``create_okf_handler(bundle)`` (capped
concept summaries into a public context key of a plain FSM).
"""

from __future__ import annotations

import math
import os
import re
import tempfile
import time
import unicodedata
from collections.abc import Callable, Iterable, Iterator, Mapping
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from itertools import islice
from pathlib import Path
from typing import Any, ClassVar, Literal, cast
from urllib.parse import quote, unquote

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .__version__ import __version__
from .constants import (
    OKF_EXPORT_TEXT_MAX_CHARS,
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

# --------------------------------------------------------------
# Untrusted-text helper (shared by every output path)
# --------------------------------------------------------------

_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)?")
_LINE_BREAKS = "\n\r\t\u2028\u2029\x0b\x0c\x85"
# Cf characters that spell words (Persian and Indic ZWNJ, emoji ZWJ sequences,
# hyphenation hints). Kept only between two visible characters.
_JOINERS = "\u200c\u200d\u00ad"
_VISIBLE_CATEGORIES = "LMNPS"


def strip_unsafe_text(text: str, *, multiline: bool = False) -> str:
    """Remove what could drive a terminal or hide text from a reader.

    Drops ANSI escape sequences, control characters (Cc) and format characters
    (Cf: zero-width and bidi marks, BOM). Exception: ZWNJ, ZWJ and soft hyphen
    stay when a letter, mark, number, punctuation or symbol sits on both sides
    (so ``می‌خواهم`` and emoji sequences survive); at an edge, beside a space
    or a dropped character they go. Each line break or tab becomes one space,
    unless ``multiline`` keeps ``\\n`` and ``\\t``. Never raises.
    """
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


# --------------------------------------------------------------
# Models
# --------------------------------------------------------------


class TrustTier(str, Enum):
    """Advisory trust label derived from ``verified[].by`` (a claim, not auth)."""

    UNVERIFIED = "unverified"
    MACHINE_CONFIRMED = "machine-confirmed"
    HUMAN_REVIEWED = "human-reviewed"


class OKFIssue(BaseModel):
    """One problem found while reading or checking a bundle.

    ``message``, ``path`` and ``id`` come from untrusted input, so they are
    control-stripped and the message is capped when the issue is built.
    """

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
    """Resource bounds for reading a bundle (defaults from ``constants.OKF_*``).

    ``max_seconds`` is a wall-clock budget checked between files; the load
    stops with a ``load-deadline`` error issue and keeps what it has.
    """

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


# The strict RFC 3339 / ISO 8601 profile: ``T`` or a space between date and
# time, ``.`` fraction, ``Z`` or ``+HH:MM`` offset, nothing lowercase, no space
# before the offset. A stamp the parser accepts outside it gets a warning.
_STRICT_TS_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}"
    r"(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:\d{2})?)?"
)


# DECISION plan-2026-10-21T140931-095ea5f8/D-014
# Do NOT replace this parser with `datetime.fromisoformat`: on Python 3.10 it
# rejects a `Z` suffix and fractions that are not 3 or 6 digits, and OKF
# stamps use both. See decisions.md D-014.
def parse_okf_datetime(value: object) -> tuple[datetime | None, str | None]:
    """Parse an OKF timestamp; return ``(aware datetime | None, warning code)``.

    Does not use ``datetime.fromisoformat``: on Python 3.10 it rejects a ``Z``
    suffix and fractions that are not 3 or 6 digits. Accepts ``Z``, ``+HH``,
    ``+HH:MM``, ``+HHMM`` and any fraction length. The code is ``None`` for the
    conformant form (explicit UTC offset), ``"date-only"`` (midnight UTC),
    ``"naive"`` (read as UTC) or ``"invalid"`` (datetime is ``None``).
    """
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc), "naive"
        return value, None
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc), (
            "date-only"
        )
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
        parsed = datetime(
            int(year),
            int(month),
            int(day),
            int(hour or 0),
            int(minute or 0),
            int(second or 0),
            micro,
        )
    except ValueError:
        return None, "invalid"
    if hour is None:
        return parsed.replace(tzinfo=timezone.utc), "date-only"
    if tz is None:
        return parsed.replace(tzinfo=timezone.utc), "naive"
    return parsed.replace(tzinfo=tz), None


class OKFConcept(BaseModel):
    """One concept document. ``id`` is the file path minus ``.md``.

    Values are kept as read (timestamps stay strings, unknown keys stay in
    ``extra``) so a write-back is faithful. ``type`` is ``""`` when the document
    gave none (an error issue, not a rejection).
    """

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
        return [
            name
            for name, on in (
                ("stale", self.is_stale(now)),
                ("draft", self.is_draft),
                ("deprecated", self.is_deprecated),
            )
            if on
        ]

    @property
    def display_title(self) -> str:
        """Control-stripped title, else the last id segment."""
        title = strip_unsafe_text(self.title or "").strip()
        return title or strip_unsafe_text(self.id.rsplit("/", 1)[-1])


# --------------------------------------------------------------
# Tolerant parser
# --------------------------------------------------------------

_Note = Callable[[Literal["error", "warning", "info"], str, str], None]
_TIMESTAMP_TAG = "tag:yaml.org,2002:timestamp"
_KNOWN_KEYS = frozenset(
    {"type", "title", "description", "resource", "tags", "sources"}
    | {"generated", "verified", "status", "stale_after"}
)
_STATUSES = ("draft", "stable", "deprecated")
_CLOSING_RE = re.compile(r"^---[ \t]*$", re.MULTILINE)
_MAX_FILE_WARNINGS = 25


class _OKFLoader(yaml.SafeLoader):
    """``SafeLoader`` without timestamp auto-resolution, aliases and anchors.

    SECURITY: the resolver table is a COPY. Mutating
    ``yaml.SafeLoader.yaml_implicit_resolvers`` would change ``yaml.safe_load``
    for the whole process. Timestamps stay strings (OKF code parses them).
    Aliases and anchors are refused in ``compose_node``, before any expansion,
    so an alias bomb cannot grow. Duplicate keys are recorded; the last wins.
    """

    # DECISION plan-2026-10-21T140931-095ea5f8/D-014
    # The timestamp resolver is left out of a COPY of the table. Do NOT call
    # `.remove()` or `del` on `yaml.SafeLoader.yaml_implicit_resolvers`: that
    # table is shared, so it would change `yaml.safe_load` for the whole
    # process. See decisions.md D-014.
    yaml_implicit_resolvers: ClassVar[dict[Any, Any]] = {
        first: [(tag, rx) for tag, rx in rules if tag != _TIMESTAMP_TAG]
        for first, rules in yaml.SafeLoader.yaml_implicit_resolvers.items()
    }

    def __init__(self, stream: str) -> None:
        super().__init__(stream)
        self.duplicate_keys: list[str] = []

    def compose_node(self, parent: Any, index: Any) -> Any:
        event = self.peek_event()
        if getattr(event, "anchor", None):  # AliasEvent and anchored nodes
            raise yaml.YAMLError("YAML aliases and anchors are not allowed")
        return super().compose_node(parent, index)

    def construct_mapping(self, node: Any, deep: bool = False) -> Any:
        seen: set[str] = set()
        for key_node, _ in node.value:
            if isinstance(key_node, yaml.ScalarNode):
                if key_node.value in seen:
                    self.duplicate_keys.append(key_node.value)
                seen.add(key_node.value)
        return super().construct_mapping(node, deep)


def _split_frontmatter(text: str) -> tuple[str | None, str, str | None]:
    """Return ``(yaml text | None, body, problem code | None)``.

    The text has its BOM removed and CRLF / CR normalised to ``\\n`` (the body
    is therefore returned with ``\\n`` line ends). Without a complete
    ``---`` ... ``---`` block the whole text is the body.
    """
    text = text.removeprefix("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    first, _, rest = text.partition("\n")
    if first.rstrip() != "---":
        return None, text, "frontmatter-missing"
    closing = _CLOSING_RE.search(rest)
    if closing is None:
        return None, text, "frontmatter-unterminated"
    return rest[: closing.start()], rest[closing.end() :].removeprefix("\n"), None


def _flow_depth_exceeded(raw: str) -> bool:
    """PyYAML's scanner is quadratic on deeply nested ``[``/``{``: bound it first.

    Brackets in quoted scalars count too; a false positive needs 64 unclosed
    brackets in one string and costs that concept one error issue.
    """
    depth = 0
    for ch in raw:
        if ch in "[{":
            depth += 1
            if depth > OKF_MAX_YAML_DEPTH:
                return True
        elif ch in "]}" and depth:
            depth -= 1
    return False


def _load_yaml(raw: str, note: _Note) -> dict[Any, Any]:
    # SECURITY: size bound BEFORE any YAML work (pure-Python parse: up to ~27 ms/KB).
    if len(raw.encode("utf-8", "replace")) > OKF_MAX_FRONTMATTER_BYTES:
        note("error", "frontmatter-too-large", "frontmatter block is too large")
        return {}
    duplicates: list[str] = []
    try:
        loader = _OKFLoader(raw)
        if _flow_depth_exceeded(raw):
            raise RecursionError
        data = cast("object", loader.get_single_data())
        duplicates = loader.duplicate_keys
    except RecursionError:  # first: a RuntimeError, so it must not reach the catch-all
        note("error", "frontmatter-too-deep", "frontmatter is nested too deeply")
        return {}
    except (yaml.YAMLError, ValueError, TypeError, OverflowError) as exc:
        problem = getattr(exc, "problem", None) or exc
        note("error", "frontmatter-invalid", f"{type(exc).__name__}: {problem}"[:160])
        return {}
    except Exception as exc:  # PyYAML constructors raise anything
        # A malformed built-in tag (``!!timestamp x``, an empty ``!!int``) escapes
        # as AttributeError, KeyError or IndexError. The issue names the class.
        logger.debug(f"OKF frontmatter rejected: {type(exc).__name__}: {exc}")
        note("error", "frontmatter-invalid", f"{type(exc).__name__}: {exc}"[:160])
        return {}
    for key in duplicates:
        note("warning", "key-duplicate", f"duplicate key {key!r}; the last value wins")
    if data is None:
        note("warning", "frontmatter-empty", "frontmatter block is empty")
        return {}
    if not isinstance(data, dict):
        note("error", "frontmatter-not-mapping", "frontmatter is not a mapping")
        return {}
    return data


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


def parse_concept_text(concept_id: str, text: str) -> tuple[OKFConcept, list[OKFIssue]]:
    """Parse one concept document. Never raises on content.

    Returns the concept (``type == ""`` when the document gave none) and the
    issues found. A document without usable frontmatter keeps its text as the
    body. Dates are not converted: timestamps stay as written and are checked
    with ``parse_okf_datetime``. An unexpected failure in any step becomes an
    error issue and the text is kept as the body.
    """
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

    for label, value in (
        ("stale_after", stale_after),
        ("generated.at", (generated or {}).get("at")),
        *(("verified.at", entry.get("at")) for entry in verified),
    ):
        if value is not None:
            _check_timestamp(label, value, note)
    concept = OKFConcept(
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
    return concept


# --------------------------------------------------------------
# Links (lexical, in id space)
# --------------------------------------------------------------

_Ids = list[str]
_Concepts = list[OKFConcept]
_Hits = list[tuple[OKFConcept, float]]

_KNOWN_VERSIONS = frozenset({"0.1", "0.2"})
_RESERVED_STEMS = frozenset({"index", "log"})
_ID_MAX_SEGMENT_BYTES = 200


def _is_reserved(stem: str) -> bool:
    """``index`` and ``log`` in any case: spec 8 and 9 files, never concepts."""
    return stem.casefold() in _RESERVED_STEMS


def _id_key(concept_id: str) -> str:
    """Identity of an id on a case-insensitive, normalising disk (loader, writer)."""
    return unicodedata.normalize("NFC", concept_id).casefold()


def _id_problem(concept_id: object, *, allow_reserved: bool = False) -> str | None:
    """Why *concept_id* may not become a file path, or ``None`` when it may.

    The ONE id-safety rule of the loader and the writer. ``allow_reserved`` is
    for the loader, which routes ``index.md`` and ``log.md`` by name first.
    """
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
_HAY_FIELDS = ((3, 0), (2, 1), (1, 2))  # (weight, index into the haystack)
_HAY_WEIGHT = sum(weight for weight, _ in _HAY_FIELDS)  # best score per token


def _body_links(body: str) -> Iterator[str]:
    """Raw link targets of the prose of *body* (fenced and inline code ignored)."""
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
    for found in islice(_LINK_RE.finditer("\n".join(prose)), _MAX_LINKS_PER_BODY):
        yield found.group(1) or found.group(2)


def _link_candidates(target: object, from_id: str) -> _Ids:
    """Concept ids a link or path-valued key may name, best first (D-003).

    A ``/``-prefixed target is bundle-root relative. A relative target is tried
    file-relative first (spec 6.2), then bundle-root relative (spec 6.3 and the
    reference bundles, issue #29). The fragment and query are dropped, the rest
    is percent-decoded once, external schemes and non-``.md`` targets give
    nothing, and a target that climbs above the root is not a candidate.

    SECURITY: this is lexical on purpose. It works in id space and never
    touches the filesystem, so nothing here can read outside the bundle;
    D-032's "no normpath" governs the loader's file access, not this.
    """
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


# --------------------------------------------------------------
# Bundle
# --------------------------------------------------------------


class OKFBundle:
    """Concepts by id, plus the link graph and lexical search over them.

    Interface contract:
        - Build with ``OKFBundle(concepts, root=, issues=, okf_version=,
          index_text=)`` or ``load_bundle``; ``root`` is ``None`` for an
          in-memory bundle. ``index_text`` is control-stripped and capped.
        - ``get(id)`` -> concept or ``None``; ``len``, ``in`` and iteration
          (concepts in id order) work as for a mapping of ids.
        - ``list(type=, prefix=, include_deprecated=False)`` -> concepts in id
          order (``type`` compared case-insensitively).
        - ``search(query, limit=5, include_deprecated=False)`` -> ``(concept,
          score)`` best first, score in (0, 1], ties by id; lexical, tokens of
          3+ characters; ``[]`` for a non-string or token-less query.
        - ``resolve_link(target, from_id)`` -> the id of an EXISTING concept
          or ``None`` (external, escaping, broken and non-concept targets all
          give ``None``); it also serves path-valued keys such as ``resource``.
        - ``outgoing(id)`` / ``backlinks(id)`` -> ids in order, no self-links;
          outgoing covers body links, ``resource`` and ``sources[].resource``.
        - ``put(concept)`` adds or replaces and invalidates every cache.
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
    ) -> None:
        self.root = root
        self.issues: list[OKFIssue] = list(issues)
        self.okf_version = okf_version
        self.index_text = strip_unsafe_text(index_text, multiline=True)[
            :OKF_INDEX_MAX_CHARS
        ]
        self._concepts: dict[str, OKFConcept] = {}
        self._reset()
        for concept in concepts:
            self.put(concept)

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
        tokens = list(dict.fromkeys(re.findall(r"\w{3,}", query[:1000].casefold())))[
            :20
        ]
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
            for cid in self._sorted_ids():
                for target in self.outgoing(cid):
                    back.setdefault(target, []).append(cid)
            self._back = back
        return list(self._back.get(concept_id, []))


# --------------------------------------------------------------
# Loader
# --------------------------------------------------------------

# code -> (severity, message) for every problem the loader itself reports.
_LOAD_PROBLEMS: dict[str, tuple[Literal["error", "warning", "info"], str]] = {
    "id-unsafe": ("warning", "file name is not a safe concept id; skipped"),
    "id-collision": ("warning", "id repeats an earlier one (case or Unicode form)"),
    "hardlink-skipped": ("warning", "file has several hard links; skipped"),
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
    "load-deadline": ("error", "load time limit reached; load stopped"),
    "log-heading": ("warning", "log.md heading is not an ISO date (spec 9)"),
    "index-frontmatter": ("warning", "index.md has frontmatter (spec 8)"),
}


def _capped(issues: list[OKFIssue]) -> list[OKFIssue]:
    """At most ``OKF_MAX_ISSUES`` issues; errors are kept before warnings, info."""
    if len(issues) <= OKF_MAX_ISSUES:
        return issues
    rank = ("error", "warning", "info")
    kept = sorted(issues, key=lambda i: rank.index(i.severity))[:OKF_MAX_ISSUES]
    extra = f"{len(issues) - OKF_MAX_ISSUES} more issues are not shown"
    kept.append(OKFIssue(severity="info", code="issues-truncated", message=extra))
    return kept


def _read_bytes(path: Path, limit: int) -> bytes:
    """Read at most ``limit + 1`` bytes; never follows a final symlink."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    with os.fdopen(os.open(path, flags), "rb") as handle:
        return handle.read(limit + 1)


def load_bundle(
    path: str | os.PathLike[str], *, limits: OKFLimits | None = None
) -> OKFBundle:
    """Load the bundle under *path*. Raises ``OKFError`` only for the root.

    A missing or non-directory root is an ``OKFError`` and is never created.
    Everything else (oversize, non-UTF-8, symlinked, hardlinked, unsafe-named
    or unreadable files, a spent limit, an unexpected failure on one file)
    becomes an ``OKFIssue`` and the rest still loads.

    Interface contract:
        - Files are visited in sorted order; only exact ``.md`` names count.
          ``index.md`` and ``log.md`` in any case, at any level, are reserved,
          never concepts (the root ``index.md`` supplies ``okf_version`` and
          ``index_text``). Directories and files starting with ``.`` are
          skipped silently.
        - An id (the path without ``.md``) must pass the writer's id rule (no
          control characters, no empty, ``.``, ``..`` or over-long segment),
          else ``id-unsafe``. Ids that are equal after NFC normalisation and
          case folding keep the first one visited; the rest are
          ``id-collision`` (what ``write_bundle`` would refuse anyway).
        - Symlinks (files and directories) are skipped with an issue, never
          followed; so are files with more than one hard link
          (``hardlink-skipped``); every file also passes
          ``security.resolve_within``.
        - Stops with an error issue at ``max_files`` and ``max_total_bytes``,
          and at ``max_seconds`` of wall-clock time (``load-deadline``, checked
          between entries); the concepts loaded so far are kept. Directories
          deeper than ``max_depth`` are not entered.
        - An unexpected exception while processing one file becomes a
          ``parse-failed`` issue for that file.
        - Residual hole (TOCTOU): a path swapped for a symlink between the
          scan and the open is refused by ``O_NOFOLLOW`` on its last part only.
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
    claimed: set[str] = set()  # _id_key of every concept loaded
    total, version, index, stop = 0, None, "", False

    def issue(code: str, rel: str) -> None:
        severity, message = _LOAD_PROBLEMS[code]
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
        claimed.add(_id_key(cid))
        issues.extend(found)

    def visit_file(entry: os.DirEntry[str], here: tuple[str, ...]) -> None:
        """One regular ``.md`` file: skip it with an issue, or load it."""
        rel, cid = "/".join(here), "/".join(here)[:-3]
        if here[-1].startswith(".") and cid not in (".", ".."):
            return  # hidden, like a dot-directory (``...md`` is id ``..``: unsafe)
        info = entry.stat(follow_symlinks=False)
        is_reserved = _is_reserved(here[-1][:-3])
        if _id_problem(cid, allow_reserved=True):
            issue("id-unsafe", rel)
        elif info.st_nlink > 1:
            issue("hardlink-skipped", rel)
        elif not is_reserved and _id_key(cid) in claimed:
            issue("id-collision", rel)
        elif not is_reserved and len(concepts) >= limits.max_files:
            halt("max-files", rel)
        elif (text := read(here, info.st_size)) is not None:
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
    )


# --------------------------------------------------------------
# Conformance (spec section 11)
# --------------------------------------------------------------


def check_bundle(bundle: OKFBundle) -> list[OKFIssue]:
    """Spec section 11 conformance as a list of issues. Never raises.

    The load-time issues (parse problems, skipped files, reserved-file checks)
    plus model-level checks that also cover concepts added with ``put``: a
    non-empty ``type``, no reserved concept id, timestamp forms without an
    explicit UTC offset (warnings), broken internal links (info: consumers must
    tolerate them) and an unknown ``okf_version`` (info). Duplicates are
    dropped; the result is capped like ``load_bundle``'s.
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

    if bundle.okf_version not in (None, *_KNOWN_VERSIONS):
        add(
            "info",
            "okf-version-unknown",
            f"okf_version {bundle.okf_version!r} is unknown",
        )
    for c in bundle:

        def note(
            severity: Literal["error", "warning", "info"],
            code: str,
            message: str,
            cid: str = c.id,
        ) -> None:
            add(severity, code, message, cid)

        if _is_reserved(c.id.rsplit("/", 1)[-1]):
            note("error", "reserved-id", "index and log are reserved, never concepts")
        if not c.type.strip() and c.id not in typed:
            note("error", "type-empty", "'type' must be a non-empty string")
        stamps = [
            ("stale_after", c.stale_after),
            ("generated.at", (c.generated or {}).get("at")),
        ]
        stamps += [("verified.at", v.get("at")) for v in c.verified]
        for label, value in stamps:
            if value is not None:
                _check_timestamp(label, value, note)
        for target in _body_links(c.body):
            if (cands := _link_candidates(target, c.id)) and not any(
                x in bundle for x in cands
            ):
                note(
                    "info",
                    "link-broken",
                    f"link target {target!r} is not in the bundle",
                )
    return _capped(out)


# --------------------------------------------------------------
# Writer
# --------------------------------------------------------------

_INDEX_TYPE_FALLBACK = "Untyped"
_MAX_CLASHES_SHOWN = 5


class _OKFDumper(yaml.SafeDumper):
    """``SafeDumper`` that never emits anchors or aliases.

    The loader refuses aliases, so a value shared by two keys (the same dict
    object twice) must be written out twice, not as ``&id001`` / ``*id001``.
    """

    def ignore_aliases(self, data: Any) -> bool:
        return True


def _iso(value: datetime) -> str:
    """ISO 8601 with an explicit UTC offset; a naive value is taken as UTC."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


def _plain(value: Any) -> Any:
    """Copy *value* for dumping: datetimes (and dates, as midnight UTC) become
    offset-bearing strings; strings read from disk pass through verbatim."""
    if isinstance(value, datetime):
        return _iso(value)
    if isinstance(value, date):
        return _iso(datetime(value.year, value.month, value.day))
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _render_concept(concept: OKFConcept) -> str:
    """The file text of *concept*. Raises ``OKFError`` when it cannot be dumped.

    Key order is fixed: the known keys, ``usage_window`` (kept in ``extra``)
    after ``sources``, then the remaining ``extra`` keys in original order.
    """
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
        dumped = yaml.dump(
            _plain(front),
            Dumper=_OKFDumper,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )
    except (yaml.YAMLError, RecursionError) as exc:
        raise OKFError(
            f"concept {concept.id!r}: frontmatter cannot be written "
            f"({type(exc).__name__})"
        ) from exc
    body = concept.body
    if body and not body.endswith("\n"):
        body += "\n"
    return f"---\n{dumped}---\n{body}"


def _render_index(bundle: OKFBundle) -> str:
    """Root ``index.md``: ``okf_version`` frontmatter, a section per type."""
    groups: dict[str, list[str]] = {}
    for concept in bundle:
        heading = strip_unsafe_text(concept.type).strip() or _INDEX_TYPE_FALLBACK
        title = re.sub(r"([\\\[\]])", r"\\\1", concept.display_title)
        desc = strip_unsafe_text(concept.description or "").strip()
        line = f"* [{title}](/{quote(concept.id, safe='/')}.md)"
        groups.setdefault(heading, []).append(
            f"{line} - {desc[:OKF_SUMMARY_FIELD_CHARS]}" if desc else line
        )
    sections = [
        f"# {name}\n\n" + "\n".join(groups[name]) + "\n" for name in sorted(groups)
    ]
    return '---\nokf_version: "0.2"\n---\n' + "\n".join(sections)


def _target_problem(root: Path, rel: str) -> str | None:
    """Why ``root/rel`` may not be written, or ``None``.

    SECURITY: ``security.resolve_within`` is the only confinement check. On top
    of "stays inside" the resolved path must EQUAL the lexical one, so a
    symlinked directory component or target (even one pointing inside the
    root) is refused, and no ancestor may be a regular file.
    """
    lexical = root.joinpath(*rel.split("/"))
    try:
        resolved = resolve_within(root, rel)
        if resolved is None:
            return "resolves outside the target directory"
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


def _atomic_write(path: Path, text: str) -> None:
    """Write *text* to *path* through a same-directory temp file and rename.

    Mirrors ``session.FileSessionStore.save`` on purpose (a private copy; the
    anchor at ``agents/semantic_memory.py`` forbids a shared helper without a
    maintainer decision, and core must not import ``harness``). The suffix is
    not ``.md``, so a crash never leaves a loadable file. ``fsync`` before the
    rename, and an existence-checked ``finally`` cleans up on ANY failure; no
    ``except BaseException``. The file mode is the temp file's (owner-only).
    """
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".okf-tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


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
          body, a trailing newline. Unknown keys survive; a second write of a
          loaded bundle is byte-identical. ``datetime`` values get an explicit
          UTC offset (naive taken as UTC); strings read from disk are verbatim.
        - Every id and target is validated and every file rendered BEFORE the
          first byte is written, so a bad id, a clash or an unwritable value
          leaves the directory untouched (``OKFError`` naming them, capped).
          An id that is empty, absolute, has an empty, ``.``, ``..`` or
          dot-prefixed segment, control characters, or ends in ``index`` /
          ``log`` is refused, as is a case-insensitive duplicate.
        - Each target passes ``security.resolve_within``; a symlinked target or
          directory component is refused even with ``overwrite=True``.
        - An existing file is replaced only with ``overwrite=True``; otherwise
          every clash is listed in the ``OKFError``. A root that exists as a
          file is refused.
        - ``write_index`` adds a root ``index.md`` (``okf_version: "0.2"``, a
          ``* [Title](/id.md) - description`` bullet per concept under a
          ``# <type>`` heading). An existing ``index.md`` is kept unless
          ``overwrite=True``.
        - Atomic PER FILE (temp file, fsync, rename). The whole bundle is not
          atomic: an I/O failure midway leaves the files already written.
    """
    try:
        root = Path(path).resolve()
        if root.exists() and not root.is_dir():
            raise OKFError(f"bundle root exists and is not a directory: {root}")
    except (OSError, RuntimeError, ValueError) as exc:
        raise OKFError(f"cannot open bundle root: {type(exc).__name__}") from exc

    problems: list[str] = []
    plan: list[tuple[str, Path, str]] = []  # (id, target, text)
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
    if write_index:
        index = root / "index.md"
        if os.path.lexists(index) and not overwrite:
            pass  # a user-owned index.md is not ours to replace
        elif (reason := _target_problem(root, "index.md")) is not None:
            problems.append(f"'index.md' {reason}")
        else:
            plan.append(("index", index, _render_index(bundle)))
    if problems:
        shown = "; ".join(problems[:_MAX_CLASHES_SHOWN])
        more = len(problems) - _MAX_CLASHES_SHOWN
        raise OKFError(
            f"nothing written, {len(problems)} problem(s): {shown}"
            + (f"; and {more} more" if more > 0 else "")
        )

    written: list[Path] = []
    for cid, target, text in plan:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(target, text)
        except OSError as exc:
            raise OKFError(
                f"cannot write {cid[:80]!r}: {type(exc).__name__}; "
                f"{len(written)} file(s) were already written"
            ) from exc
        written.append(target)
    return sorted(written)


# --------------------------------------------------------------
# Export of memory entries
# --------------------------------------------------------------

_DROP = object()
_MAX_META_ITEMS = 100
_MAX_META_DEPTH = 4
_MAX_META_TEXT = 2_000
_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _native(value: Any, depth: int = 0) -> Any:
    """JSON-native copy of *value* or ``_DROP``; secret-looking and internal
    keys at any depth are removed (``security`` is the single decision point)."""
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else _DROP
    if isinstance(value, str):
        return value[:_MAX_META_TEXT]
    if depth >= _MAX_META_DEPTH:
        return _DROP
    if isinstance(value, (list, tuple)):
        kept = (_native(v, depth + 1) for v in islice(value, _MAX_META_ITEMS))
        return [v for v in kept if v is not _DROP]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in islice(value.items(), _MAX_META_ITEMS):
            if (
                isinstance(key, str)
                and not has_internal_prefix(key)
                and not is_forbidden_context_entry(key, item)
                and (clean := _native(item, depth + 1)) is not _DROP
            ):
                out[key[:_MAX_META_TEXT]] = clean
        return out
    return _DROP


def _line(value: object, cap: int = OKF_SUMMARY_FIELD_CHARS) -> str:
    """Control-stripped, trimmed, capped one-line text; ``""`` for a non-string."""
    return strip_unsafe_text(value).strip()[:cap] if isinstance(value, str) else ""


def _slug(raw: object) -> str:
    slug = _SLUG_RE.sub("-", str(raw)[:200]).strip("-.")[:80].strip("-.")
    if not slug:
        return "entry"
    return f"entry-{slug}" if slug.casefold() in ("index", "log") else slug


def bundle_from_memory_entries(
    entries: Iterable[Any],
    *,
    generated_by: str | None = None,
    now: datetime | str | None = None,
) -> OKFBundle:
    """Turn memory entries (plain mappings) into an in-memory ``OKFBundle``.

    Interface contract:
        - ``entries``: mappings shaped like ``SemanticMemoryStore`` entries,
          ``{id, text, metadata, embedding}``. Only ``id``, ``text`` and
          ``metadata`` are read; embeddings are NEVER exported. A non-mapping
          entry, a missing or non-string ``text`` or a blank one is skipped
          with a warning ``OKFIssue`` on ``bundle.issues``; nothing raises.
        - Concept id ``memory/<slug of the entry id>`` (characters outside
          ``A-Za-z0-9._-`` become ``-``, at most 80); a repeated slug gets
          ``-2``, ``-3``. A hostile id can therefore not shape the path.
        - ``type`` is metadata ``type`` (a non-blank string) else ``Memory``;
          ``title`` is metadata ``title`` else the first non-blank line, capped
          at ``OKF_SUMMARY_FIELD_CHARS``; ``tags`` are the string items of a
          metadata ``tags`` list. The body is the text (cut at
          ``OKF_EXPORT_TEXT_MAX_CHARS`` with a warning issue).
        - Every other metadata entry whose value is JSON-native goes under
          ``extra["metadata"]``. Entries are dropped when
          ``is_forbidden_context_entry(key, value)`` is true or the key is
          internal-prefixed, at any depth; non-JSON values are dropped.
        - ``generated`` is ``{by: generated_by or "fsm-llm/<version>", at: now
          or the current time}`` with an explicit UTC offset. There is no
          ``verified``, so every export is ``unverified`` by construction.
    """
    issues: list[OKFIssue] = []
    concepts: list[OKFConcept] = []
    used: set[str] = set()
    stamp = _iso(now) if isinstance(now, datetime) else _line(now, 100)
    generated = {
        "by": _line(generated_by, 200) or f"fsm-llm/{__version__}",
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
        raw_meta = entry.get("metadata")
        meta = _native(dict(raw_meta)) if isinstance(raw_meta, Mapping) else {}
        type_, title, tags = (meta.pop(k, None) for k in ("type", "title", "tags"))
        first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
        base = _slug(entry.get("id", ""))
        slug, n = base, 1
        while slug.casefold() in used:
            n += 1
            slug = f"{base}-{n}"
        used.add(slug.casefold())
        concepts.append(
            OKFConcept(
                id=f"memory/{slug}",
                type=_line(type_, 100) or "Memory",
                title=_line(title if isinstance(title, str) else first_line) or None,
                tags=[t for x in tags if (t := _line(x, 100))]
                if isinstance(tags, list)
                else [],
                generated=dict(generated),
                extra={"metadata": meta} if meta else {},
                body=text,
            )
        )
    return OKFBundle(concepts, issues=issues)


# --------------------------------------------------------------
# Plain-FSM handler
# --------------------------------------------------------------


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
          ``{id, type, title, description, trust, flags}`` (JSON-native; every
          string control-stripped and cut at ``OKF_SUMMARY_FIELD_CHARS``;
          ``flags`` a sorted list from stale, draft, deprecated). NEVER a body.
        - Concepts: with ``query_key`` set and a non-blank ``str`` there (cut
          to 1,000 chars, control-stripped), the best ``bundle.search`` hits;
          any other value (missing, None, number, list, dict, blank) means the
          first ``max_items`` by id. Deprecated concepts only with
          ``include_deprecated``. ``states`` empty means every state.
        - Not critical: a failure is logged, the turn goes on.
        - Raises ``OKFError`` at build time for: a ``context_key`` that is not a
          non-blank ``str``, is internal-prefixed, secret-looking (judged for a
          string AND for the list value written, so ``user_token`` is refused)
          or in ``RESERVED_CONTEXT_KEYS`` (the prompt filter or the pipeline would
          drop it); ``max_items`` not an ``int`` in ``1..OKF_SUMMARY_MAX_ITEMS``
          (refused, never clamped); a ``states`` that is a ``str`` or holds a
          non-``str``; a ``timing`` that is not a ``HandlerTiming``; a
          ``query_key`` that is neither ``None`` nor a non-blank ``str``.
        - The default ``PRE_PROCESSING`` timing runs on every turn, a BLOCKED
          one included (``PRE_TRANSITION`` handlers do not).
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
        query = ctx.get(query_key) if query_key is not None else None
        query = (
            strip_unsafe_text(query[:1000]).strip() if isinstance(query, str) else ""
        )
        if query:
            found = [c for c, _ in bundle.search(query, max_items, include_deprecated)]
        else:
            found = bundle.list(include_deprecated=include_deprecated)[:max_items]
        now = datetime.now(timezone.utc)
        return {
            context_key: [
                {
                    "id": _line(c.id),
                    "type": _line(c.type),
                    "title": _line(c.display_title),
                    "description": _line(c.description),
                    "trust": c.trust.value,
                    "flags": sorted(c.flags(now)),
                }
                for c in found
            ]
        }

    return create_handler(name).at(timing).on_state(*state_names).do(summarize)
