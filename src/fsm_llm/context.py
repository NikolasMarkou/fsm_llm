"""
Context management utilities for FSM-LLM.

Provides stateless utility functions for context cleaning and compaction,
kept separate from the orchestration classes.
"""

from __future__ import annotations

from typing import Any

from .constants import (
    MAX_CONTEXT_FILTER_DEPTH,
    has_internal_prefix,
    is_forbidden_context_entry,
)
from .definitions import ResponseGenerationRequest
from .logging import logger
from .utilities import filter_context_tree, redact_non_json_leaf


class ContextCompactor:
    """Configurable context compaction for FSM conversations.

    Removes transient keys each turn and prunes state-specific keys on
    transitions.  Designed to be used as handler callbacks via the fluent
    ``HandlerBuilder`` API::

        compactor = ContextCompactor(
            transient_keys={"action_result", "action_errors"},
            prune_on_entry={"review": {"structure_done", "connections_done"}},
        )

        api.register_handler(
            api.create_handler("compactor")
            .at(HandlerTiming.PRE_PROCESSING)
            .do(compactor.compact)
        )
        api.register_handler(
            api.create_handler("pruner")
            .at(HandlerTiming.POST_TRANSITION)
            .do(compactor.prune)
        )

    Args:
        transient_keys: Keys cleared every turn (PRE_PROCESSING).
        prune_on_entry: Mapping of ``state_name`` → set of keys to clear
            when that state is entered (POST_TRANSITION).
    """

    def __init__(
        self,
        transient_keys: set[str] | None = None,
        prune_on_entry: dict[str, set[str]] | None = None,
        summarize_on_trim: bool = False,
    ) -> None:
        self.transient_keys: set[str] = transient_keys or set()
        self.prune_on_entry: dict[str, set[str]] = prune_on_entry or {}
        self.summarize_on_trim: bool = summarize_on_trim

    def compact(self, context: dict[str, Any]) -> dict[str, Any]:
        """Clear transient keys from the previous turn.

        Intended as a ``PRE_PROCESSING`` handler callback.  Returns a dict
        with ``None`` values for every transient key present in *context*,
        which the handler system interprets as deletion.

        The actual deletion (and, since D-018, the matching provenance
        digest cleanup) happens in ``MessagePipeline.execute_handlers``'s
        ``merge_delta`` (``pipeline.py``) -- this method only returns the
        delta dict, it never touches ``context.data``/``context.metadata``
        directly. See decisions.md D-018.
        """
        removals = {key: None for key in self.transient_keys if key in context}
        if removals:
            logger.debug(f"Context compactor cleared transient keys: {list(removals)}")
        return removals

    def prune(self, context: dict[str, Any]) -> dict[str, Any]:
        """Clear state-specific keys when entering a new state.

        Intended as a ``POST_TRANSITION`` handler callback.  Reads
        ``_current_state`` (set by the pipeline on every transition) to
        determine which keys to prune.

        See ``compact()``'s docstring above: the actual deletion (and, since
        D-018, the matching provenance digest cleanup) happens in
        ``pipeline.py``'s ``merge_delta``, not here.
        """
        target = context.get("_current_state", "")
        keys_to_clear = self.prune_on_entry.get(target, set())
        removals = {key: None for key in keys_to_clear if key in context}
        if removals:
            logger.debug(
                f"Context compactor pruned keys on entry to '{target}': {list(removals)}"
            )
        return removals

    def summarize(
        self,
        conversation: Any,
        llm_interface: Any | None = None,
    ) -> str | None:
        """Summarize older conversation exchanges into a compact representation.

        If *llm_interface* is provided, uses it to generate an LLM-powered
        summary. Otherwise, falls back to a simple text concatenation of
        message previews.

        This method is designed to be called explicitly or registered as a
        handler callback. It updates ``conversation.summary`` in place and
        returns the summary text.

        Args:
            conversation: A ``Conversation`` instance whose exchanges
                should be summarized.
            llm_interface: Optional ``LLMInterface`` for LLM-powered
                summarization. If ``None``, uses simple text extraction.

        Returns:
            The summary text, or ``None`` if no exchanges to summarize.
        """
        if not hasattr(conversation, "exchanges") or not conversation.exchanges:
            return None

        # Build text from all exchanges for summarization
        lines: list[str] = []
        for exchange in conversation.exchanges:
            for role, message in exchange.items():
                preview = message[:200]
                if len(message) > 200:
                    preview += "..."
                lines.append(f"{role}: {preview}")

        if not lines:
            return None

        if llm_interface is not None:
            try:
                prompt = (
                    "Summarize the following conversation exchanges into a concise "
                    "paragraph preserving key facts, decisions, and user preferences. "
                    "Keep the summary under 500 characters.\n\n" + "\n".join(lines)
                )
                request = ResponseGenerationRequest(
                    system_prompt="You are a conversation summarizer.",
                    user_message=prompt,
                    extracted_data={},
                    context={},
                )
                response = llm_interface.generate_response(request)
                summary = response.message
                conversation.summary = summary[:2000]
                logger.debug("Context compactor: LLM-powered summary generated")
                result: str | None = conversation.summary
                return result
            except Exception as e:
                logger.warning(
                    f"Context compactor: LLM summarization failed ({e}), "
                    "falling back to text extraction"
                )

        # Fallback: simple text concatenation
        summary_text = " | ".join(lines)
        if len(summary_text) > 2000:
            summary_text = summary_text[:2000]
        conversation.summary = summary_text
        logger.debug("Context compactor: text-based summary generated")
        fallback_result: str | None = conversation.summary
        return fallback_result


# DECISION plan-2026-07-19T191147-4b664252/D-010 [STALE]
# The depth bound is a SECURITY control, not a performance tweak, and the
# behavior AT the bound is fail-CLOSED on purpose: a container nested deeper
# than MAX_CONTEXT_FILTER_DEPTH is DROPPED, never passed through. Do NOT "fix"
# the data loss by returning the sub-tree unfiltered at the limit -- that hands an
# attacker a one-line bypass (bury the secret 17 levels down). Do NOT remove
# the bound in favour of a cycle-detecting `seen` set either: the bound is
# what makes a self-referential dict (`d["self"] = d`) terminate, and a
# RecursionError here is a crash inside prompt construction on
# provider-influenced data. The bound itself lives in constants.py so this
# filter and prompts.py's `_filter_context_for_security` share ONE value
# (D-011). See decisions.md D-010, D-011.
#
# DECISION plan-2026-09-12T135914-45a654de/D-016
# The depth-bounded dict/list/tuple recursion itself (byte-identical to
# `fsm.py`'s walker) now lives in ONE shared place,
# `utilities.filter_context_tree`. This call site supplies its OWN 5-reason
# predicate and its OWN logging/`removed_keys`/`warned_keys` tracking via the
# `should_drop`/`on_drop` closures below -- every behavior documented in this
# function's docstring (None-stripping, forbidden-pattern check, WARNING
# logging) is reproduced exactly as before, just relocated into the closures.
# `fsm.py`'s call site does NOT get any of this richer policy (see its own
# D-010 comment). See decisions.md D-016.


def clean_context_keys(
    data: dict[str, Any],
    conversation_id: str,
    remove_none_values: bool = True,
    strip_forbidden_keys: bool = False,
) -> dict[str, Any]:
    """
    Clean invalid keys from context data, at every nesting level.

    Only strips None values and keys with internal prefix patterns.
    Empty lists and empty strings are preserved as they can be
    semantically meaningful (e.g., ``{"allergies": []}`` means "no allergies").
    That falsy-survives contract holds at every depth, not just the top level.

    The same key filter is applied recursively to nested dicts and to dicts
    inside lists/tuples, so ``{"user": {"password": "x"}}`` and
    ``{"users": [{"password": "x"}]}`` are filtered like their flat
    equivalents.  Recursion is bounded at ``MAX_CONTEXT_FILTER_DEPTH``; anything
    deeper is dropped rather than passed through unfiltered (see D-010).

    Args:
        data: Dictionary to clean
        conversation_id: For logging context
        remove_none_values: Remove keys with None values
        strip_forbidden_keys: Remove keys matching forbidden security patterns
            (password, secret, token, api_key) instead of just warning

    Returns:
        Cleaned dictionary with invalid keys removed. Nested dicts/lists are
        rebuilt (new objects); scalar values are returned unchanged.
    """
    log = logger.bind(conversation_id=conversation_id)
    removed_keys: list[str] = []
    warned_keys: list[str] = []

    def should_drop(key: Any, value: Any, full_key: str) -> str | None:
        """Return a removal reason, or ``None`` to keep ``key``.

        # DECISION plan-2026-07-19T191147-4b664252/D-017 [STALE]
        # The `isinstance(key, str)` guard MUST stay ABOVE the emptiness
        # check. It used to sit below it, so `if not key` fired first and
        # `0`, `False`, `0.0` and `()` were destroyed as "empty key" --
        # while the sibling filter in `prompts.py` KEPT them, so the two
        # filters disagreed on exactly the falsy non-`str` keys. Do NOT
        # "tidy" the emptiness check back to the top: `not key` is only a
        # meaningful test for `str`, and D-010's recursion into arbitrary
        # nested data is what makes int-keyed dicts reachable here.
        #
        # A non-`str` key is also logged at WARNING (not silently skipped):
        # pre-fix, `b"password".startswith("_")` RAISED, and converting a
        # loud failure into a silent pass-through is a fail-OPEN default
        # inside a fail-CLOSED control. `bytes` keys in particular bypass
        # every name check. Do NOT downgrade this to debug/remove it.
        # See decisions.md D-017.
        """
        if not isinstance(key, str):
            log.warning(
                f"Context key {key!r} ({type(key).__name__}) skipped the "
                "security name checks: only str keys can be matched "
                "against internal prefixes and forbidden patterns"
            )
            if remove_none_values and value is None:
                return "None value"
            return None

        # Check for empty-string keys
        if not key:
            return "empty key"

        # Check for None values
        if remove_none_values and value is None:
            return "None value"

        # Check for internal prefix patterns
        if has_internal_prefix(key):
            return "internal key prefix"

        # Check for forbidden security patterns. `value` feeds layer 2
        # (constants.py D-019), which decides the ambiguous
        # `<qualifier>_key` shape on the VALUE's shape -- the NAME cannot
        # separate `stripe_key` from `order_key`.
        if is_forbidden_context_entry(key, value):
            if strip_forbidden_keys:
                return "forbidden security pattern"
            warned_keys.append(full_key)

        return None

    def on_drop(full_key: str, reason: str) -> None:
        if reason == "too_deep":
            removed_keys.append(f"{full_key} (nested deeper than max depth)")
            log.warning(
                f"Context value '{full_key}' dropped: nested deeper than "
                f"{MAX_CONTEXT_FILTER_DEPTH} levels and cannot be security-filtered"
            )
        elif reason == "cycle":
            removed_keys.append(f"{full_key} (reference cycle)")
            log.warning(f"Context value '{full_key}' dropped: reference cycle")
        else:
            removed_keys.append(f"{full_key} ({reason})")
            log.debug(f"Context key '{full_key}' removed: {reason}")

    # DECISION plan-2026-09-21T203800-8a03483a/D-010
    # DECISION plan-2026-09-21T203800-8a03483a/D-011
    # DECISION plan-2026-09-21T203800-8a03483a/D-045
    # The leaf hook redacts non-JSON-native values (their str() carried object
    # fields past the key filter); do NOT drop it to "keep values as-is". The
    # walker's active-path cycle guard and per-subtree memoisation bound work on
    # aliased input next to the depth bound (D-010 above). Do NOT pass a
    # truncating node budget here: the result is COMMITTED to context (the
    # extracted-data commit), so a cut value would be silently lost data.
    # See decisions.md D-010, D-011, D-045.
    cleaned = filter_context_tree(
        data,
        MAX_CONTEXT_FILTER_DEPTH,
        should_drop,
        on_drop,
        leaf=redact_non_json_leaf,
    )

    if warned_keys:
        log.warning(
            f"Context contains keys matching forbidden security patterns: {warned_keys}. "
            "Storing sensitive data (passwords, secrets, tokens, API keys) in FSM context "
            "is a security risk. Set strip_forbidden_keys=True to auto-remove."
        )

    if removed_keys:
        log.debug(f"Removed context keys: {removed_keys}")

    return cleaned
