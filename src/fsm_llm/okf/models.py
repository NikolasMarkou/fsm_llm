"""
OKF models: the trust tier, an issue, the load limits, the timestamp parser and a concept.

``OKFIssue`` fields are untrusted and cleaned here (control-stripped, capped);
``capped_issues`` bounds an issue list. Imports ``fsm_llm.text``, ``constants`` and
pydantic only.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime, timedelta, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from fsm_llm.constants import (
    OKF_MAX_DEPTH,
    OKF_MAX_FILE_BYTES,
    OKF_MAX_FILES,
    OKF_MAX_ISSUES,
    OKF_MAX_LOAD_SECONDS,
    OKF_MAX_TOTAL_BYTES,
)
from fsm_llm.text import capped_issues, strip_unsafe_text


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
# Do NOT replace this parser with `datetime.fromisoformat`: on Python 3.10 (the
# old floor) it rejected a `Z` suffix and fractions that are not 3 or 6 digits,
# and OKF stamps use both. The parser stays: it also reports a warning code per
# input shape, which `fromisoformat` cannot. See decisions.md D-014.
def parse_okf_datetime(value: object) -> tuple[datetime | None, str | None]:
    """Parse an OKF timestamp: ``(aware datetime | None, warning code)``; the
    code is ``None`` (explicit offset: ``Z``, ``+HH``, ``+HH:MM``, ``+HHMM``),
    ``"date-only"`` (midnight UTC), ``"naive"`` (read as UTC) or ``"invalid"``."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC), "naive"
        return value, None
    if isinstance(value, date):
        midnight = datetime(value.year, value.month, value.day, tzinfo=UTC)
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
                tz = UTC
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
        return parsed.replace(tzinfo=UTC), "date-only"
    if tz is None:
        return parsed.replace(tzinfo=UTC), "naive"
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
        now = now or datetime.now(UTC)
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
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


INCOMPLETE_CODES = ("max-total-bytes", "max-files", "load-deadline", "check-deadline")


def _capped(issues: list[OKFIssue]) -> list[OKFIssue]:
    return capped_issues(issues, OKF_MAX_ISSUES, INCOMPLETE_CODES, OKFIssue)
