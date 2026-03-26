from __future__ import annotations

"""
Ollama-specific helpers for structured output compatibility.

Ollama models with thinking capabilities (e.g. Qwen3) activate "thinking
mode" by default, producing ``<think>...</think>`` reasoning traces that
corrupt structured JSON output.  Thinking and structured output cannot
coexist in a single Ollama call (ollama/ollama#10538).

This module centralises detection and parameter configuration so that
both ``LiteLLMInterface`` and the classification ``Classifier`` (which
calls ``litellm.completion()`` directly) apply the same fixes:

- ``reasoning_effort = "none"`` — LiteLLM (>=1.82) maps this to
  Ollama's top-level ``think: false`` flag
- ``temperature = 0`` — deterministic output for structured calls
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

FIELD_EXTRACTION_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "field_name": {
            "type": "string",
        },
        "value": {},  # Any type
        "confidence": {
            "type": "number",
        },
        "reasoning": {
            "type": "string",
        },
    },
    "required": ["field_name", "value", "confidence"],
}

# Map call types to their schemas
_CALL_TYPE_SCHEMAS: dict[str, tuple[dict, str]] = {
    "data_extraction": (EXTRACTION_JSON_SCHEMA, "data_extraction"),
    "transition_decision": (TRANSITION_JSON_SCHEMA, "transition_decision"),
    "field_extraction": (FIELD_EXTRACTION_JSON_SCHEMA, "field_extraction"),
}


# ------------------------------------------------------------------
# Model detection
# ------------------------------------------------------------------


def is_ollama_model(model: str) -> bool:
    """Check if the model string targets an Ollama backend."""
    return "ollama" in model.lower()


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
