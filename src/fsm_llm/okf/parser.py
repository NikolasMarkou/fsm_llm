"""
The tolerant OKF parser: one concept document to ``(OKFConcept, issues)``.

Content never raises (a problem becomes an ``OKFIssue``); the frontmatter work is
the shared seam ``fsm_llm.frontmatter``. ``_split_frontmatter``, ``_read_bytes`` and
``_load_yaml`` stay module globals here (the loader imports them; a test patch aims
at the module that performs the lookup).
"""

from __future__ import annotations

from datetime import date
from functools import partial
from typing import Any

from fsm_llm import frontmatter as _fm
from fsm_llm.constants import OKF_MAX_FRONTMATTER_BYTES, OKF_MAX_YAML_DEPTH
from fsm_llm.logging import logger
from fsm_llm.text import capped_note

from .models import _STRICT_TS_RE, OKFConcept, OKFIssue, parse_okf_datetime

_KNOWN_KEYS = frozenset(
    {"type", "title", "description", "resource", "tags", "sources"}
    | {"generated", "verified", "status", "stale_after"}
)
_STATUSES = ("draft", "stable", "deprecated")
_MAX_FILE_WARNINGS = 25


# The one frontmatter seam is ``frontmatter.py``; these names stay module globals (tests patch them).
_split_frontmatter = _fm.split_frontmatter


_read_bytes = _fm.read_bytes
_load_yaml = partial(
    _fm.load_yaml, max_bytes=OKF_MAX_FRONTMATTER_BYTES, max_depth=OKF_MAX_YAML_DEPTH
)


def _text(data: dict[str, Any], key: str, note: _fm.Note) -> str | None:
    value = data.get(key)
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (bool, int, float, date)):
        note("warning", "value-coerced", f"{key!r} is not a string; read as text")
        return str(value)
    note("warning", "value-invalid", f"{key!r} must be a string; ignored")
    return None


def _tags(value: Any, note: _fm.Note) -> list[str]:
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


def _mapping_list(
    data: dict[str, Any], key: str, note: _fm.Note
) -> list[dict[Any, Any]]:
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


def _check_timestamp(label: str, value: Any, note: _fm.Note) -> None:
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
    stale_after: Any, generated: Any, verified: Any, note: _fm.Note
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
    note = capped_note(
        lambda **kw: issues.append(OKFIssue(id=concept_id, **kw)),
        _MAX_FILE_WARNINGS,
        "concept",
    )

    try:
        concept = _build_concept(concept_id, text, note)
    except Exception as exc:  # the parser is total on content
        logger.debug(f"OKF concept {concept_id!r} unreadable: {exc!r}")
        note("error", "frontmatter-invalid", f"{type(exc).__name__}: {exc}"[:160])
        concept = OKFConcept(id=concept_id, body=text)
    return concept, issues


def _build_concept(concept_id: str, text: str, note: _fm.Note) -> OKFConcept:
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
