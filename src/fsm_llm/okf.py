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

Entry point: ``parse_concept_text(concept_id, text) -> (OKFConcept, issues)``.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from typing import Any, ClassVar, Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .constants import (
    OKF_MAX_DEPTH,
    OKF_MAX_FILE_BYTES,
    OKF_MAX_FILES,
    OKF_MAX_TOTAL_BYTES,
    OKF_MAX_YAML_DEPTH,
)

# --------------------------------------------------------------
# Untrusted-text helper (shared by every output path)
# --------------------------------------------------------------

_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)?")
_LINE_BREAKS = "\n\r\t\u2028\u2029\x0b\x0c\x85"


def strip_unsafe_text(text: str, *, multiline: bool = False) -> str:
    """Remove what could drive a terminal or hide text from a reader.

    Drops ANSI escape sequences, control characters (Cc), and format characters
    (Cf: zero-width and bidi marks, BOM). Each line break or tab becomes one
    space, unless ``multiline`` keeps ``\\n`` and ``\\t``. Never raises.
    """
    out: list[str] = []
    for ch in _ANSI_RE.sub("", text):
        if multiline and ch in "\n\t":
            out.append(ch)
        elif ch in _LINE_BREAKS:
            out.append(" ")
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
    """Resource bounds for reading a bundle (defaults from ``constants.OKF_*``)."""

    model_config = ConfigDict(frozen=True)

    max_files: int = Field(OKF_MAX_FILES, gt=0)
    max_file_bytes: int = Field(OKF_MAX_FILE_BYTES, gt=0)
    max_total_bytes: int = Field(OKF_MAX_TOTAL_BYTES, gt=0)
    max_depth: int = Field(OKF_MAX_DEPTH, gt=0)


_TS_RE = re.compile(
    r"(\d{4})-(\d{2})-(\d{2})"
    r"(?:[Tt ](\d{2}):(\d{2})(?::(\d{2})(?:[.,](\d+))?)?"
    r"\s*(Z|z|[+-]\d{2}(?::?\d{2})?)?)?"
)


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


class _OKFLoader(yaml.SafeLoader):
    """``SafeLoader`` without timestamp auto-resolution, aliases and anchors.

    SECURITY: the resolver table is a COPY. Mutating
    ``yaml.SafeLoader.yaml_implicit_resolvers`` would change ``yaml.safe_load``
    for the whole process. Timestamps stay strings (OKF code parses them).
    Aliases and anchors are refused in ``compose_node``, before any expansion,
    so an alias bomb cannot grow. Duplicate keys are recorded; the last wins.
    """

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
    duplicates: list[str] = []
    try:
        loader = _OKFLoader(raw)
        if _flow_depth_exceeded(raw):
            raise RecursionError
        data = cast("object", loader.get_single_data())
        duplicates = loader.duplicate_keys
    except (yaml.YAMLError, ValueError, TypeError, OverflowError) as exc:
        problem = getattr(exc, "problem", None) or exc
        note("error", "frontmatter-invalid", f"{type(exc).__name__}: {problem}"[:160])
        return {}
    except RecursionError:
        note("error", "frontmatter-too-deep", "frontmatter is nested too deeply")
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


def parse_concept_text(concept_id: str, text: str) -> tuple[OKFConcept, list[OKFIssue]]:
    """Parse one concept document. Never raises on content.

    Returns the concept (``type == ""`` when the document gave none) and the
    issues found. A document without usable frontmatter keeps its text as the
    body. Dates are not converted: timestamps stay as written and are checked
    with ``parse_okf_datetime``.
    """
    issues: list[OKFIssue] = []

    def note(
        severity: Literal["error", "warning", "info"], code: str, msg: str
    ) -> None:
        issues.append(
            OKFIssue(severity=severity, code=code, message=msg, id=concept_id)
        )

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
    return concept, issues
