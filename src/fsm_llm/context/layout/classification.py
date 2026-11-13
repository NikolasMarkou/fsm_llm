"""Classification prompts and schema, and the public text sanitizer.

Layer ``context`` (plan d454fd62, step 74): split out of the old ``fsm_llm.prompts``.
"""

import json
from typing import Any

from fsm_llm.core.definitions import ClassificationPromptConfig, ClassificationSchema

from .base import BasePromptBuilder, _cap_history_line, _history_messages


def _reasoning_property() -> dict:
    """The ``reasoning`` schema property both classification shapes carry."""
    return {
        "type": "string",
        "description": "Chain-of-thought explanation before classification",
    }


def _confidence_property() -> dict:
    """The ``confidence`` schema property both classification shapes carry."""
    return {
        "type": "number",
        "minimum": 0.0,
        "maximum": 1.0,
        "description": "Model confidence in this classification",
    }


def build_classification_json_schema(
    schema: ClassificationSchema,
    *,
    multi_intent: bool = False,
    max_intents: int = 3,
    include_reasoning: bool = True,
    include_entities: bool = True,
) -> dict:
    """
    Build a JSON Schema dict from a ClassificationSchema.

    Ordering: reasoning (CoT) precedes intent to mitigate constrained-decoding
    probability distortion.
    """
    intent_enum = schema.intent_names

    if multi_intent:
        intent_score_props: dict = {
            "intent": {
                "type": "string",
                "enum": intent_enum,
                "description": "Classified intent name",
            },
            "confidence": _confidence_property(),
        }
        intent_score_required = ["intent", "confidence"]
        if include_entities:
            intent_score_props["entities"] = {
                "type": "object",
                "additionalProperties": {"type": "string"},
                "description": "Extracted entities relevant to this intent",
            }
            intent_score_required.append("entities")

        properties: dict = {}
        required: list[str] = []

        if include_reasoning:
            properties["reasoning"] = _reasoning_property()
            required.append("reasoning")

        properties["intents"] = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": intent_score_props,
                "required": intent_score_required,
                "additionalProperties": False,
            },
            "minItems": 1,
            "maxItems": max_intents,
            "description": "Ranked list of detected intents, most probable first",
        }
        required.append("intents")

    else:
        properties = {}
        required = []

        if include_reasoning:
            properties["reasoning"] = _reasoning_property()
            required.append("reasoning")

        properties["intent"] = {
            "type": "string",
            "enum": intent_enum,
            "description": "The primary classified intent of the user input",
        }
        required.append("intent")

        properties["confidence"] = _confidence_property()
        required.append("confidence")

        if include_entities:
            properties["entities"] = {
                "type": "object",
                "additionalProperties": {"type": "string"},
                "description": "Extracted entities relevant to the intent",
            }
            required.append("entities")

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def build_classification_system_prompt(
    schema: ClassificationSchema,
    config: ClassificationPromptConfig | None = None,
    *,
    user_message: str | None = "",
) -> str:
    """
    Build the system prompt for an intent classification task.

    Includes intent definitions and behavioral rules. The JSON schema
    is embedded so prompt-only approaches also get structure guidance.

    ``user_message`` is the text being classified; only ``None`` changes the
    prompt. ``None`` means there is no user message: the prompt then asks for
    a classification of the current context and recent conversation. Any
    string, the empty one included, gives the message wording.
    """
    if config is None:
        config = ClassificationPromptConfig()

    intent_lines = []
    for intent in schema.intents:
        intent_lines.append(f"- {intent.name}: {intent.description}")
    intent_block = "\n".join(intent_lines)

    json_schema = build_classification_json_schema(
        schema,
        multi_intent=config.multi_intent,
        max_intents=config.max_intents,
        include_reasoning=config.include_reasoning,
        include_entities=config.include_entities,
    )
    schema_str = json.dumps(json_schema, indent=2)

    rules = [
        "1. Output ONLY valid JSON matching the schema below.",
        "2. Set confidence between 0.0 and 1.0 based on how clear the intent is.",
    ]
    next_rule = 3
    if config.include_entities:
        rules.append(
            f"{next_rule}. Extract any relevant entities (e.g., order IDs, product names, amounts)."
        )
        next_rule += 1
    rules.append(
        f"{next_rule}. "
        f'Use "{schema.fallback_intent}" when intent is ambiguous or does not fit other categories.'
    )
    next_rule += 1
    if config.include_reasoning:
        rules.append(f"{next_rule}. Write your reasoning before selecting the intent.")
        next_rule += 1
    if config.multi_intent:
        rules.append(
            f"{next_rule}. "
            + (
                "If several intents apply, return them ranked by confidence."
                if user_message is None
                else "If the message contains multiple intents, "
                "return them ranked by confidence."
            )
        )

    rules_block = "\n".join(rules)

    if config.multi_intent:
        classify_instruction = (
            "classify it into one or more of the following intents, "
            "ranked by confidence"
        )
    else:
        classify_instruction = "classify it into exactly one of the following intents"

    subject = (
        "the current context and the recent conversation given below "
        "(there is no user message) and"
        if user_message is None
        else "the user's message and"
    )
    return (
        f"You are an intent classification engine. Analyze {subject} "
        f"{classify_instruction}:\n\n"
        f"{intent_block}\n\n"
        f"Rules:\n{rules_block}\n\n"
        f"Output JSON Schema:\n```json\n{schema_str}\n```"
    )


# One default-config builder lends its sanitizer and security filter to the
# classifier context block and to ``sanitize_text_for_prompt``, so every
# caller uses exactly the helpers the Pass-1/Pass-2 builders use (no third
# filter copy).
_CLASSIFICATION_CONTEXT_BUILDER = BasePromptBuilder()


def sanitize_text_for_prompt(text: str | None) -> str:
    """Escape XML-like tags and flatten newlines in *text* for a prompt.

    Public entry point to the one sanitiser every prompt builder uses
    (``BasePromptBuilder._sanitize_text_for_prompt``); output is identical.
    ``None`` becomes ``""``. Never raises for ``str`` or ``None`` input.
    """
    return _CLASSIFICATION_CONTEXT_BUILDER._sanitize_text_for_prompt(text)


def build_classification_context_block(
    context: dict[str, Any] | None, *, user_message: str | None = ""
) -> str:
    """Render per-call classifier context as a ``<classification_context>`` block.

    Contract (shared by ``Classifier`` and ``MessagePipeline``):

    - ``context`` keys, all optional: ``history`` (list of
      ``{"user"|"system": text}`` exchanges, oldest first), ``purpose`` (str),
      ``data`` (dict of context values).
    - Every history message and the purpose pass through
      ``_sanitize_text_for_prompt``; each history line is then capped by
      ``_cap_history_line`` (``_HISTORY_LINE_CHARS``); ``data`` first passes the same
      ``_filter_context_for_security`` and key cap as Pass-2
      ``<current_context>``, then its JSON is sanitized too.
    - Returns ``""`` when ``context`` is None/empty or nothing survives
      filtering, so a context-free call keeps a byte-identical system prompt.
      Unserializable data is dropped with a WARNING; this never raises.
    - ``user_message`` is the text being classified; only ``None`` (no user
      message) changes the block: its lead line then says the context is
      what to classify, not an aid to reading a message.
    """
    if not context:
        return ""
    builder = _CLASSIFICATION_CONTEXT_BUILDER
    sanitize = builder._sanitize_text_for_prompt
    parts: list[str] = []

    lines = [
        f"{'user' if role == 'user' else 'assistant'}: "
        f"{_cap_history_line(sanitize(str(text)))}"
        for role, text in _history_messages(context.get("history") or [])
    ]
    if lines:
        parts += ["<conversation_history>", *lines, "</conversation_history>"]

    purpose = context.get("purpose")
    if purpose:
        parts.append(f"<state_purpose>{sanitize(str(purpose))}</state_purpose>")

    data = builder._limit_context_by_key_count(
        builder._filter_context_for_security(dict(context.get("data") or {}))
    )
    if data:
        data_json = builder._dump_json(data, "classifier context data", sort_keys=True)
        if data_json is not None:
            parts.append(f"<context_data>{sanitize(data_json)}</context_data>")

    if not parts:
        return ""
    return "\n".join(
        [
            "",
            "",
            "<classification_context>",
            (
                "There is no user message. Classify from this context."
                if user_message is None
                else "Use this context only to interpret the user's message; "
                "classify the message itself."
            ),
            *parts,
            "</classification_context>",
        ]
    )
