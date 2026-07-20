from __future__ import annotations

"""
Context management utilities for FSM-LLM.

Provides stateless utility functions for context cleaning and compaction,
kept separate from the orchestration classes.
"""

from typing import Any

from .constants import (
    COMPILED_FORBIDDEN_CONTEXT_PATTERNS,
    MAX_CONTEXT_FILTER_DEPTH,
    has_internal_prefix,
)
from .definitions import ResponseGenerationRequest
from .logging import logger


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


# DECISION plan-2026-07-19T191147-4b664252/D-010
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

# Sentinel: a container past MAX_CONTEXT_FILTER_DEPTH, which the caller drops.
_TOO_DEEP = object()


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

    def drop_too_deep(path: str) -> None:
        removed_keys.append(f"{path} (nested deeper than max depth)")
        log.warning(
            f"Context value '{path}' dropped: nested deeper than "
            f"{MAX_CONTEXT_FILTER_DEPTH} levels and cannot be security-filtered"
        )

    def clean_value(value: Any, path: str, depth: int) -> Any:
        """Recurse into containers; scalars pass through untouched at ANY depth
        (they carry no keys to filter, so the falsy contract holds everywhere).
        Returns ``_TOO_DEEP`` for a container past the bound."""
        if not isinstance(value, (dict, list, tuple)):
            return value
        if depth > MAX_CONTEXT_FILTER_DEPTH:
            return _TOO_DEEP
        if isinstance(value, dict):
            return clean_mapping(value, path, depth)

        # Lists/tuples are in scope: `{"users": [{"password": "x"}]}` is the
        # same leak as `{"user": {"password": "x"}}` and must not survive it.
        items = []
        for index, item in enumerate(value):
            element_path = f"{path}[{index}]"
            cleaned_item = clean_value(item, element_path, depth + 1)
            if cleaned_item is _TOO_DEEP:
                drop_too_deep(element_path)
                continue
            items.append(cleaned_item)
        return tuple(items) if isinstance(value, tuple) else items

    def clean_mapping(source: dict[str, Any], path: str, depth: int) -> dict[str, Any]:
        """Apply the key filter to one mapping level, then recurse into values."""
        cleaned: dict[str, Any] = {}

        for key, value in source.items():
            full_key = f"{path}.{key}" if path else str(key)
            removal_reason = ""

            # DECISION plan-2026-07-19T191147-4b664252/D-017
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
            if not isinstance(key, str):
                log.warning(
                    f"Context key {key!r} ({type(key).__name__}) skipped the "
                    "security name checks: only str keys can be matched "
                    "against internal prefixes and forbidden patterns"
                )
                if remove_none_values and value is None:
                    removal_reason = "None value"

            # Check for empty-string keys
            elif not key:
                removal_reason = "empty key"

            # Check for None values
            elif remove_none_values and value is None:
                removal_reason = "None value"

            # Check for internal prefix patterns
            elif has_internal_prefix(key):
                removal_reason = "internal key prefix"

            # Check for forbidden security patterns
            elif any(p.match(key) for p in COMPILED_FORBIDDEN_CONTEXT_PATTERNS):
                if strip_forbidden_keys:
                    removal_reason = "forbidden security pattern"
                else:
                    warned_keys.append(full_key)

            if removal_reason:
                removed_keys.append(f"{full_key} ({removal_reason})")
                log.debug(f"Context key '{full_key}' removed: {removal_reason}")
                continue

            cleaned_value = clean_value(value, full_key, depth + 1)
            if cleaned_value is _TOO_DEEP:
                drop_too_deep(full_key)
                continue

            cleaned[key] = cleaned_value

        return cleaned

    cleaned = clean_mapping(data, "", 0)

    if warned_keys:
        log.warning(
            f"Context contains keys matching forbidden security patterns: {warned_keys}. "
            "Storing sensitive data (passwords, secrets, tokens, API keys) in FSM context "
            "is a security risk. Set strip_forbidden_keys=True to auto-remove."
        )

    if removed_keys:
        log.debug(f"Removed context keys: {removed_keys}")

    return cleaned
