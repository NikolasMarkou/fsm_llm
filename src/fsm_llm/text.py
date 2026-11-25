"""
Untrusted-text hygiene shared by every output path (a kernel module).

Interface contract (every name is also re-exported by ``fsm_llm.okf``, where the
helpers began):
    - ``strip_unsafe_text(text, *, multiline=False)`` -> ``str``: ANSI escapes and
      control (Cc), format (Cf) and surrogate (Cs) characters removed, except
      ZWNJ, ZWJ and soft hyphen between two visible characters; a line break or
      tab becomes one space unless ``multiline`` keeps ``\\n`` and ``\\t``.
    - ``capped_line(value, cap=OKF_SUMMARY_FIELD_CHARS)`` -> ``str``: stripped,
      trimmed, capped one-line text; ``""`` for a non-string.
    - ``safe_text(text, cap=SAFE_TEXT_CHARS)`` -> ``str``: ``capped_line`` with ``[ ] < >``
      defanged by ``SKILL_DEFANG``, so the text can spell no fence or tag; the one
      defang of model-authored text (``synthesis`` and ``simulation`` both use it).
    - ``capped_issues(issues, cap, stops, kind)`` -> the list capped at *cap*, one
      ``kind`` info ``issues-truncated`` for the rest (``ValueError`` on an
      unknown severity).
    - ``capped_note(emit, cap, what)`` -> a ``note(severity, code, message)``
      callback that passes at most *cap* warnings to *emit*, then one info.
    - ``text_page(text, offset, size)`` -> ``(page, next offset or None, length)``.
    - ``iso_utc(value)`` -> ISO 8601 text with an explicit UTC offset.
Standard library and ``fsm_llm.constants`` only (``frontmatter`` for a type name
under ``TYPE_CHECKING``); no OKF or Agent Skills model, no logging, never raises
on text content. Not imported by ``fsm_llm/__init__.py``.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Container
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from .constants import OKF_SUMMARY_FIELD_CHARS, SAFE_TEXT_CHARS, SKILL_DEFANG

if TYPE_CHECKING:
    from . import frontmatter as _fm

__all__ = [
    "capped_issues",
    "capped_line",
    "capped_note",
    "iso_utc",
    "safe_text",
    "strip_unsafe_text",
    "text_page",
]

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


def capped_line(value: object, cap: int = OKF_SUMMARY_FIELD_CHARS) -> str:
    """Control-stripped, trimmed, capped one-line text; ``""`` for a non-string."""
    return strip_unsafe_text(value).strip()[:cap] if isinstance(value, str) else ""


# DECISION plan-2026-11-25T074601-c720601d/D-007
# ONE defang function (this one) and ONE fence (``synthesis.prompts._fence``) for the text a
# model sees from ``simulation`` and ``synthesis`` (``synthesizer.py`` and ``skills.catalog``
# apply ``SKILL_DEFANG`` to other texts). Do NOT write a second defang function there, and
# do NOT make ``simulation`` import the fence from ``synthesis`` (the layer forbids it): the
# caller hands the writer its text READY-FENCED.
def safe_text(text: str, cap: int = SAFE_TEXT_CHARS) -> str:
    """``text`` as one safe line: control-stripped, cut at ``cap`` characters and
    ``[ ] < >`` defanged. ``""`` for a non-string."""
    return capped_line(text, cap).translate(SKILL_DEFANG)


def capped_issues(
    issues: list[Any], cap: int, stops: Container[str], kind: Callable[..., Any]
) -> list[Any]:
    """At most *cap* issues, *stops* codes first then error, warning, info; the rest
    become one ``kind`` info ``issues-truncated``. An unknown severity: ``ValueError``."""
    if len(issues) <= cap:
        return issues
    rank = ("error", "warning", "info")
    order = sorted(issues, key=lambda i: (i.code not in stops, rank.index(i.severity)))
    extra = f"{len(issues) - cap} more issues are not shown"
    return [*order[:cap], kind(severity="info", code="issues-truncated", message=extra)]


def capped_note(emit: Callable[..., None], cap: int, what: str) -> _fm.Note:
    """A ``note(severity, code, message)`` that calls ``emit(severity=, code=,
    message=)`` for at most *cap* warnings, then once for an info
    ``issues-truncated`` ("further warnings for this *what* are not shown"), then
    never for a warning; errors and infos pass uncounted. State is per call."""
    warned = 0

    def note(
        severity: Literal["error", "warning", "info"], code: str, msg: str
    ) -> None:
        nonlocal warned
        if severity == "warning":  # bound the work a hostile document can cause
            warned += 1
            if warned > cap + 1:
                return
            if warned > cap:
                severity, code = "info", "issues-truncated"
                msg = f"further warnings for this {what} are not shown"
        emit(severity=severity, code=code, message=msg)

    return note


def text_page(text: str, offset: Any, size: int) -> tuple[str, int | None, int]:
    """``(page, next offset or None, length)`` of *text*, which the caller has
    already control-stripped; ``offset`` is clamped into the text (a bool or
    other non-int is 0) and pages add up to it. Shared by OKF bodies and Agent Skills files."""
    try:
        start = 0 if isinstance(offset, bool) else min(max(int(offset), 0), len(text))
    except (TypeError, ValueError, OverflowError):
        start = 0
    end = min(start + size, len(text))
    return text[start:end], (end if end < len(text) else None), len(text)


def iso_utc(value: datetime) -> str:
    """ISO 8601 with an explicit UTC offset; a naive value is taken as UTC."""
    return (value if value.tzinfo else value.replace(tzinfo=UTC)).isoformat()
