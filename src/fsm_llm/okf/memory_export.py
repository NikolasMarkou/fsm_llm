"""
Export of memory entries to Open Knowledge Format: ``bundle_from_memory_entries``.

Turns memory entries (plain mappings, shaped like ``SemanticMemoryStore`` entries) into an
in-memory ``OKFBundle``. It reads the OKF models and the writer's caps of this package and the
kernel text helpers, and nothing from the engine or ``handlers``: it moved here from
``fsm_llm.engine.okf_bridge`` (plan d454fd62, step 95, D-042) so the ``fsm-llm-okf export``
subcommand (``okf.cli``) reaches it without importing an engine module. The old module keeps
the name importable: ``from fsm_llm.engine.okf_bridge import bundle_from_memory_entries``.

Not re-exported by the ``fsm_llm.okf`` facade (import it by this module's name). OKF text is
untrusted third-party prose: summaries are control-stripped and capped.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from itertools import islice
from typing import Any

from fsm_llm.__version__ import __version__
from fsm_llm.constants import OKF_EXPORT_TEXT_MAX_CHARS, OKF_MAX_FRONTMATTER_BYTES
from fsm_llm.frontmatter import SURROGATE_RE
from fsm_llm.security import has_internal_prefix, is_forbidden_context_entry
from fsm_llm.text import capped_line, iso_utc

from .bundle import OKFBundle
from .models import OKFConcept, OKFIssue
from .writer import has_lone_surrogate, writes_within_caps

__all__ = ["bundle_from_memory_entries"]

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
        clean, replaced = SURROGATE_RE.subn("\ufffd", kept)
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


def _slug(raw: object) -> str:
    slug = _SLUG_RE.sub("-", str(raw)[:200]).strip("-.")[:80].strip("-.") or "entry"
    return f"entry-{slug}" if slug.casefold() in ("index", "log") else slug


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
    stamp = iso_utc(now) if isinstance(now, datetime) else capped_line(now, 100)
    generated = {
        "by": capped_line(generated_by, 200) or f"fsm-llm/{__version__}",
        "at": stamp or iso_utc(datetime.now(UTC)),
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
        if "surrogates" in notes or has_lone_surrogate((text, ident)):
            text = SURROGATE_RE.sub("\ufffd", text)
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
        if not writes_within_caps(concept):  # drop metadata, then tags (D-018)
            notes.add("size")
            concept.extra.clear()
            if not writes_within_caps(concept):
                concept.tags = []
        concepts.append(concept)
        if cut := sorted(notes - {"surrogates"}):
            skip(index, f"metadata cut ({', '.join(cut)})", "metadata-truncated")
    return OKFBundle(concepts, issues=issues)
