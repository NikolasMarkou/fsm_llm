"""
Prompt construction for classification tasks.

Builds system prompts from a ClassificationSchema following the fsm_llm
prompt builder patterns (security filtering, deterministic output).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .definitions import ClassificationSchema

# --------------------------------------------------------------
# Configuration
# --------------------------------------------------------------


@dataclass(frozen=True)
class ClassificationPromptConfig:
    """Controls prompt generation behavior."""

    include_reasoning: bool = True
    max_tokens: int = 512
    temperature: float = 0.0
    include_entities: bool = True
    multi_intent: bool = False
    max_intents: int = 3


# --------------------------------------------------------------
# JSON Schema Generation
# --------------------------------------------------------------


def build_json_schema(
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
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Model confidence in this classification",
            },
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
            properties["reasoning"] = {
                "type": "string",
                "description": "Chain-of-thought explanation before classification",
            }
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
            properties["reasoning"] = {
                "type": "string",
                "description": "Chain-of-thought explanation before classification",
            }
            required.append("reasoning")

        properties["intent"] = {
            "type": "string",
            "enum": intent_enum,
            "description": "The primary classified intent of the user input",
        }
        required.append("intent")

        properties["confidence"] = {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Model confidence in this classification",
        }
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


# --------------------------------------------------------------
# System Prompt Builder
# --------------------------------------------------------------


def build_system_prompt(
    schema: ClassificationSchema,
    config: ClassificationPromptConfig | None = None,
) -> str:
    """
    Build the system prompt for an intent classification task.

    Includes intent definitions and behavioral rules. The JSON schema
    is embedded so prompt-only approaches also get structure guidance.
    """
    if config is None:
        config = ClassificationPromptConfig()

    intent_lines = []
    for intent in schema.intents:
        intent_lines.append(f"- {intent.name}: {intent.description}")
    intent_block = "\n".join(intent_lines)

    json_schema = build_json_schema(
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
        rules.append(
            f"{next_rule}. "
            "Write your reasoning before selecting the intent."
        )
        next_rule += 1
    if config.multi_intent:
        rules.append(
            f"{next_rule}. "
            "If the message contains multiple intents, return them ranked by confidence."
        )

    rules_block = "\n".join(rules)

    if config.multi_intent:
        classify_instruction = (
            "classify it into one or more of the following intents, "
            "ranked by confidence"
        )
    else:
        classify_instruction = "classify it into exactly one of the following intents"

    return (
        f"You are an intent classification engine. Analyze the user's message and "
        f"{classify_instruction}:\n\n"
        f"{intent_block}\n\n"
        f"Rules:\n{rules_block}\n\n"
        f"Output JSON Schema:\n```json\n{schema_str}\n```"
    )
