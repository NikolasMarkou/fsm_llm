from __future__ import annotations

"""
Context management utilities for FSM-LLM.

Extracted from FSMManager to keep stateless utility functions separate
from the orchestration class.
"""

from typing import Any

from .logging import logger
from .constants import INTERNAL_KEY_PREFIXES, COMPILED_FORBIDDEN_CONTEXT_PATTERNS


def clean_context_keys(
        data: dict[str, Any],
        conversation_id: str,
        remove_none_values: bool = True
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

        # Check for None values
        if remove_none_values and value is None:
            should_remove = True
            removal_reason = "None value"

        # Check for internal prefix patterns
        elif any(key.startswith(prefix) for prefix in INTERNAL_KEY_PREFIXES):
            should_remove = True
            removal_reason = "internal key prefix"

        # Keep the key-value pair
        if not should_remove:
            cleaned[key] = value
            # Warn on forbidden context patterns (security)
            if any(p.match(key) for p in COMPILED_FORBIDDEN_CONTEXT_PATTERNS):
                warned_keys.append(key)
        else:
            removed_keys.append(f"{key} ({removal_reason})")

    if warned_keys:
        log.warning(
            f"Context contains keys matching forbidden security patterns: {warned_keys}. "
            "Storing sensitive data (passwords, secrets, tokens, API keys) in FSM context "
            "is a security risk."
        )

    if removed_keys:
        log.debug(f"Removed empty context keys: {removed_keys}")

    return cleaned
