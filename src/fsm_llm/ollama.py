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

import copy
import json

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

# DECISION plan-2026-09-19T175721-21cd7f8e/D-001
# The `value` schema MUST carry a TYPE; do NOT restore `"value": {}` ("any type").
# The schema is sent as an Ollama grammar AND pasted into the user message
# (`prepare_ollama_messages`), and qwen3.5:9b reads `"value": {}` as "an object goes
# here", returning dict-wrapped junk (`{'blue': 'blue'}`, `{'full_name_missing': True}`)
# that is truthy, passes `is not None` gates and drives transitions (LV-01, measured
# live). A schema `description` is NOT model-visible over `ollama_chat` (LESSONS
# L154-156), so only the TYPE can steer; a global union was rejected because a
# `dict` field must still accept objects, and unwrapping a dict is guesswork on data
# the model invented. See decisions.md D-001.
#
# Per-field_type `value` type unions. Every union includes "null" (ungrounded ->
# unset) and "string" (models quote numbers/booleans); only `dict` allows "object".
_VALUE_TYPES_ANY: list[str] = ["string", "number", "boolean", "array", "null"]
_VALUE_TYPES_BY_FIELD_TYPE: dict[str, list[str]] = {
    "str": ["string", "null"],
    "int": ["number", "string", "null"],
    "float": ["number", "string", "null"],
    "bool": ["boolean", "string", "null"],
    "list": ["array", "string", "null"],
    "dict": ["object", "string", "null"],
    "any": _VALUE_TYPES_ANY,
}

FIELD_EXTRACTION_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "field_name": {
            "type": "string",
        },
        "value": {"type": list(_VALUE_TYPES_ANY)},  # default: `any`, no object
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
    # NOT reachable from llm.py: `_make_llm_call` gates structured output on
    # `call_type in ["data_extraction", "field_extraction"]`, and no caller ever
    # passes "transition_decision" (transitions are decided by rules, not by the
    # LLM). Kept because `build_ollama_response_format` is public and tested;
    # the exclusion is pinned by test_regression_jsonlogic_and_merge.py's VB3.
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


def prepare_ollama_messages(
    messages: list[dict[str, str]],
    model: str,
    response_format: dict | None = None,
) -> list[dict[str, str]]:
    """Prepare messages for Ollama structured output calls.

    Applies two guide recommendations that operate at the message level
    (as opposed to ``apply_ollama_params`` which operates on call params):

    1. Prepends ``/nothink`` to the last user message — belt-and-suspenders
       approach to ensure thinking stays off for Qwen3-family models.
    2. Appends the JSON schema to the last user message when a
       ``json_schema`` response format is provided, so the model sees the
       schema as context (Ollama's grammar constrains tokens but the model
       doesn't otherwise see the schema).

    Returns a shallow copy of *messages* with the last user message
    modified.  The original list is not mutated.

    Args:
        messages: The message list for the LLM call.
        model: The model identifier string.
        response_format: The ``response_format`` dict (if any) that will
            be passed to ``litellm.completion()``.
    """
    if not is_ollama_model(model):
        return messages

    # Find the last user message
    last_user_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user_idx = i
            break

    if last_user_idx == -1:
        return messages

    # Shallow copy to avoid mutating the caller's list
    messages = [msg.copy() for msg in messages]
    content = messages[last_user_idx]["content"]

    # 1. Prepend /nothink
    content = f"/nothink\n{content}"

    # 2. Append schema from response_format if present
    if (
        response_format is not None
        and response_format.get("type") == "json_schema"
        and "json_schema" in response_format
    ):
        schema = response_format["json_schema"].get("schema")
        if schema:
            schema_str = json.dumps(schema)
            content += f"\n\nRespond in JSON matching this schema:\n{schema_str}"

    messages[last_user_idx]["content"] = content

    logger.debug(
        "Applied Ollama message preparation: /nothink prefix + schema in prompt"
    )
    return messages


def build_ollama_response_format(
    call_type: str, field_type: str | None = None
) -> dict | None:
    """Build a ``json_schema`` response format for the given call type.

    Args:
        call_type: ``"data_extraction"``, ``"transition_decision"`` or
            ``"field_extraction"``.
        field_type: Only read for ``"field_extraction"``: the declared
            ``FieldExtractionConfig.field_type`` selects the ``value`` type union
            (see ``_VALUE_TYPES_BY_FIELD_TYPE``). ``None`` or an unknown type
            gives the ``any`` union (no object). Ignored for other call types.

    Returns:
        A ``response_format`` dict, or ``None`` if *call_type* has no
        associated schema (e.g. ``response_generation``). A typed
        ``field_extraction`` schema is a fresh copy, never the shared constant.
    """
    entry = _CALL_TYPE_SCHEMAS.get(call_type)
    if entry is None:
        return None

    schema, name = entry
    if call_type == "field_extraction" and field_type in _VALUE_TYPES_BY_FIELD_TYPE:
        schema = copy.deepcopy(schema)
        schema["properties"]["value"] = {
            "type": list(_VALUE_TYPES_BY_FIELD_TYPE[field_type])
        }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "schema": schema,
        },
    }
