"""
Bridge between plain FSMs and Open Knowledge Format bundles.

``create_okf_handler`` puts capped concept summaries of a bundle into a plain FSM's
context. ``bundle_from_memory_entries`` (memory entries to an in-memory ``OKFBundle``) is
defined in ``fsm_llm.okf.memory_export`` since step 95 of plan d454fd62 (D-042) and is
still importable from here. The format itself (models, parser, loader, writer) is
``fsm_llm.okf``, which this module imports by public names only; the text helpers
(``strip_unsafe_text``, ``capped_line``) are in ``fsm_llm.text``.
OKF text is untrusted third-party prose: summaries are control-stripped and capped,
and nothing a bundle names is executed.

An engine module (the one engine module on OKF): it imports no subpackage, no provider
SDK and no other engine module, and ``fsm_llm/__init__.py`` does not load it. Import it
by name: ``from fsm_llm.engine.okf_bridge import ...``.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from fsm_llm.constants import OKF_SUMMARY_MAX_ITEMS, RESERVED_CONTEXT_KEYS
from fsm_llm.core.definitions import OKFError
from fsm_llm.handlers import BaseHandler, HandlerTiming, create_handler
from fsm_llm.okf import OKFBundle
from fsm_llm.okf.memory_export import bundle_from_memory_entries
from fsm_llm.security import has_internal_prefix, is_forbidden_context_entry
from fsm_llm.text import capped_line, strip_unsafe_text

__all__ = ["bundle_from_memory_entries", "create_okf_handler"]


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
        now = datetime.now(UTC)
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
