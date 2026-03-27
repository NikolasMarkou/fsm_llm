from __future__ import annotations

"""
Context management utilities for FSM-LLM.

Provides stateless utility functions for context cleaning and compaction,
kept separate from the orchestration classes.
"""

from typing import Any

from .constants import COMPILED_FORBIDDEN_CONTEXT_PATTERNS, INTERNAL_KEY_PREFIXES
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
                response = llm_interface.generate_response(
                    system_prompt="You are a conversation summarizer.",
                    user_message=prompt,
                    extracted_data={},
                    context={},
                )
                summary = (
                    response.response_text
                    if hasattr(response, "response_text")
                    else str(response)
                )
                conversation.summary = summary[:2000]
                logger.debug("Context compactor: LLM-powered summary generated")
                return conversation.summary
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
        return conversation.summary


def clean_context_keys(
    data: dict[str, Any],
    conversation_id: str,
    remove_none_values: bool = True,
    strip_forbidden_keys: bool = False,
) -> dict[str, Any]:
    """
    Clean invalid keys from context data.

    Only strips None values and keys with internal prefix patterns.
    Empty lists and empty strings are preserved as they can be
    semantically meaningful (e.g., ``{"allergies": []}`` means "no allergies").

    Args:
        data: Dictionary to clean
        conversation_id: For logging context
        remove_none_values: Remove keys with None values
        strip_forbidden_keys: Remove keys matching forbidden security patterns
            (password, secret, token, api_key) instead of just warning

    Returns:
        Cleaned dictionary with invalid keys removed
    """
    log = logger.bind(conversation_id=conversation_id)
    cleaned = {}
    removed_keys = []
    warned_keys = []

    for key, value in data.items():
        should_remove = False
        removal_reason = ""

        # Check for empty-string keys
        if not key:
            should_remove = True
            removal_reason = "empty key"

        # Check for None values
        elif remove_none_values and value is None:
            should_remove = True
            removal_reason = "None value"

        # Check for internal prefix patterns
        elif any(key.startswith(prefix) for prefix in INTERNAL_KEY_PREFIXES):
            should_remove = True
            removal_reason = "internal key prefix"

        # Check for forbidden security patterns
        elif any(p.match(key) for p in COMPILED_FORBIDDEN_CONTEXT_PATTERNS):
            if strip_forbidden_keys:
                should_remove = True
                removal_reason = "forbidden security pattern"
            else:
                warned_keys.append(key)

        # Keep the key-value pair
        if not should_remove:
            cleaned[key] = value
        else:
            removed_keys.append(f"{key} ({removal_reason})")
            log.debug(f"Context key '{key}' removed: {removal_reason}")

    if warned_keys:
        log.warning(
            f"Context contains keys matching forbidden security patterns: {warned_keys}. "
            "Storing sensitive data (passwords, secrets, tokens, API keys) in FSM context "
            "is a security risk. Set strip_forbidden_keys=True to auto-remove."
        )

    if removed_keys:
        log.debug(f"Removed context keys: {removed_keys}")

    return cleaned
