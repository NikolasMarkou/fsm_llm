"""
Bridges between memory/plain FSMs and Open Knowledge Format bundles.

``bundle_from_memory_entries`` turns memory entries (plain mappings) into an
in-memory ``OKFBundle``; ``create_okf_handler`` puts capped concept summaries of a
bundle into a plain FSM's context. The format itself (models, parser, loader,
writer, ``strip_unsafe_text``) is core ``fsm_llm.okf``, which this module imports
by public names only. OKF text is untrusted third-party prose: summaries are
control-stripped and capped, and nothing a bundle names is executed.

A core module: it imports no subpackage and no provider SDK, and ``fsm_llm/__init__.py``
does not load it. Import it by name: ``from fsm_llm.okf_bridge import ...``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from itertools import islice
from typing import Any

from .__version__ import __version__
from .constants import (
    OKF_EXPORT_TEXT_MAX_CHARS,
    OKF_MAX_FRONTMATTER_BYTES,
    OKF_SUMMARY_MAX_ITEMS,
    RESERVED_CONTEXT_KEYS,
)
from .definitions import OKFError
from .frontmatter import SURROGATE_RE
from .handlers import BaseHandler, HandlerTiming, create_handler
from .okf import (
    OKFBundle,
    OKFConcept,
    OKFIssue,
    capped_line,
    has_lone_surrogate,
    iso_utc,
    strip_unsafe_text,
    writes_within_caps,
)
from .security import has_internal_prefix, is_forbidden_context_entry

__all__ = ["bundle_from_memory_entries", "create_okf_handler"]

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
        "at": stamp or iso_utc(datetime.now(timezone.utc)),
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
