from __future__ import annotations

"""
Ollama-specific helpers for structured output compatibility.

Ollama's Qwen3-family models activate "thinking mode" by default, which
produces ``<think>...</think>`` reasoning traces that corrupt structured
JSON output.  Thinking and structured output cannot coexist in a single
Ollama call (ollama/ollama#10538).

This module centralises detection and parameter configuration so that
both ``LiteLLMInterface`` and the classification ``Classifier`` (which
calls ``litellm.completion()`` directly) apply the same fixes:

- ``reasoning_effort = "none"`` — LiteLLM maps this to ``think: false``
- ``extra_body.think = False`` — belt-and-suspenders direct flag
- ``temperature = 0`` — deterministic output for structured calls
- ``/nothink`` prompt prefix — Qwen3-specific prompt-level override
- ``json_schema`` response format with explicit schema
"""

from .logging import logger

# ------------------------------------------------------------------
# JSON Schema constants for structured output
# ------------------------------------------------------------------

EXTRACTION_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "extracted_data": {
            "type": "object",
        },
        "confidence": {
            "type": "number",
        },
        "reasoning": {
            "type": "string",
        },
    },
    "required": ["extracted_data", "confidence"],
}

TRANSITION_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "selected_transition": {
            "type": "string",
        },
        "reasoning": {
            "type": "string",
        },
    },
    "required": ["selected_transition"],
}

# Map call types to their schemas
_CALL_TYPE_SCHEMAS: dict[str, tuple[dict, str]] = {
    "data_extraction": (EXTRACTION_JSON_SCHEMA, "data_extraction"),
    "transition_decision": (TRANSITION_JSON_SCHEMA, "transition_decision"),
}


# ------------------------------------------------------------------
# Model detection helpers
# ------------------------------------------------------------------


def is_ollama_model(model: str) -> bool:
    """Check if the model string targets an Ollama backend."""
    return "ollama" in model.lower()


def is_qwen3_model(model: str) -> bool:
    """Check if the model is a Qwen3-family model running on Ollama.

    Qwen3 models support the ``/nothink`` prompt prefix to disable
    thinking mode at the prompt level.
    """
    lower = model.lower()
    return is_ollama_model(lower) and "qwen3" in lower


# ------------------------------------------------------------------
# Parameter helpers
# ------------------------------------------------------------------


def apply_ollama_params(
    call_params: dict,
    model: str,
    *,
    structured: bool = True,
) -> None:
    """Apply Ollama-specific parameters to disable thinking mode.

    Mutates *call_params* in place.

    Args:
        call_params: The parameter dict that will be passed to
            ``litellm.completion()``.
        model: The model identifier string.
        structured: When ``True`` (the default), also forces
            ``temperature=0`` for deterministic structured output.
            Set to ``False`` for response-generation calls where the
            user's configured temperature should be preserved.
    """
    if not is_ollama_model(model):
        return

    # Disable thinking mode.  LiteLLM (>=1.82) maps
    # ``reasoning_effort`` to Ollama's top-level ``think`` flag.
    # ``"none"`` evaluates to ``False`` (thinking off).
    call_params["reasoning_effort"] = "none"

    # Deterministic output for structured calls only
    if structured:
        call_params["temperature"] = 0

    logger.debug(
        f"Applied Ollama params: reasoning_effort=none"
        f"{', temperature=0' if structured else ''}"
    )


def prepend_nothink(messages: list[dict[str, str]], model: str) -> None:
    """Prepend ``/nothink`` to the last user message for Qwen3 models.

    Mutates *messages* in place.  This is a belt-and-suspenders approach
    that tells Qwen3 models at the prompt level to skip thinking.

    For non-Qwen3 or non-Ollama models this is a no-op.
    """
    if not is_qwen3_model(model):
        return

    for msg in reversed(messages):
        if msg.get("role") == "user":
            msg["content"] = "/nothink\n" + msg["content"]
            logger.debug("Prepended /nothink to user message for Qwen3 model")
            break


def build_ollama_response_format(call_type: str) -> dict | None:
    """Build a ``json_schema`` response format for the given call type.

    Returns ``None`` if *call_type* has no associated schema (e.g.
    ``response_generation``).
    """
    entry = _CALL_TYPE_SCHEMAS.get(call_type)
    if entry is None:
        return None

    schema, name = entry
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "schema": schema,
        },
    }
