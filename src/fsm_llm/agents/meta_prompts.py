"""
Prompt and request builders for the meta-builder.

Build request: ``build_artifact_prompt`` (the one user turn of the build
call), ``artifact_schema`` (the per-type extraction schema) and
``build_response_format`` (the response format of the meta FSM's ``build``
completion state). Collect turn: ``build_collect_response_instructions``
(the ``collect`` state's Pass-2 instructions). Also the review presentation,
welcome message, follow-up helpers and output formatting.
"""

from __future__ import annotations

import copy
from typing import Any

from .constants import (
    META_AGENT_PATTERN_INTENTS,
    META_BUILD_PROMPT,
    MetaContextKeys,
)
from .definitions import ArtifactType
from .meta_builders import (
    AgentArtifactBuilder,
    ArtifactBuilder,
    WorkflowArtifactBuilder,
)

# ------------------------------------------------------------------
# Build request -- schemas, response format, prompt
# ------------------------------------------------------------------

# Name of the JSON schema in the build state's response format.
_BUILD_SCHEMA_NAME = "artifact_spec"

_FSM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "description": {"type": "string"},
        "persona": {"type": "string"},
        "states": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "state_id": {"type": "string"},
                    "description": {"type": "string"},
                    "purpose": {"type": "string"},
                },
                "required": ["state_id", "description", "purpose"],
            },
        },
        "transitions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "from_state": {"type": "string"},
                    "target_state": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["from_state", "target_state", "description"],
            },
        },
    },
    "required": ["name", "description", "states"],
}

_WORKFLOW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "description": {"type": "string"},
        "workflow_id": {"type": "string"},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "step_id": {"type": "string"},
                    # DECISION plan-2026-09-24T091842-c1d5bfbc/D-010: the
                    # enum is the grammar that steers the model (ollama's
                    # ``format`` is this schema verbatim). Do NOT drop it
                    # back to a free string or hand-copy the list: a free
                    # string lets the model invent a type that only fails
                    # later in ``validate_complete``.
                    "step_type": {
                        "type": "string",
                        "enum": sorted(WorkflowArtifactBuilder.VALID_STEP_TYPES),
                    },
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["step_id", "step_type", "name"],
            },
        },
    },
    "required": ["name", "description", "steps"],
}

_AGENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "description": {"type": "string"},
        "tools": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["name", "description"],
            },
        },
    },
    "required": ["name", "description", "tools"],
}

_ARTIFACT_SCHEMAS: dict[ArtifactType, dict[str, Any]] = {
    ArtifactType.FSM: _FSM_SCHEMA,
    ArtifactType.WORKFLOW: _WORKFLOW_SCHEMA,
    ArtifactType.AGENT: _AGENT_SCHEMA,
}

_TYPE_EXAMPLES: dict[ArtifactType, str] = {
    ArtifactType.FSM: (
        "Create states with unique state_ids, descriptions, and purposes. "
        "Add transitions between states. The first state is the initial state.\n"
        "Example:\n"
        '{"name":"MyBot","description":"A bot","persona":"friendly",'
        '"states":[{"state_id":"start","description":"Welcome","purpose":"Greet user"},'
        '{"state_id":"end","description":"Goodbye","purpose":"Close the conversation"}],'
        '"transitions":[{"from_state":"start","target_state":"end","description":"Done"}]}'
    ),
    ArtifactType.WORKFLOW: (
        "Create steps with unique step_ids, step types (auto_transition, "
        "llm_processing, api_call, condition), names, and descriptions.\n"
        "Example:\n"
        '{"name":"MyFlow","description":"A flow","workflow_id":"wf1",'
        '"steps":[{"step_id":"s1","step_type":"auto_transition","name":"Start","description":"Begin"}]}'
    ),
    ArtifactType.AGENT: (
        "Create tool definitions with clear names and descriptions.\n"
        "Example:\n"
        '{"name":"MyAgent","description":"An agent",'
        '"tools":[{"name":"search","description":"Search the web"}]}'
    ),
}


def artifact_schema(artifact_type: ArtifactType) -> dict[str, Any]:
    """Return a fresh copy of the extraction schema for ``artifact_type``.

    The schema of the artifact spec the build call asks for (FSM states and
    transitions, workflow steps with the ``step_type`` enum, agent tools).
    Never raises for an ``ArtifactType`` member.
    """
    return copy.deepcopy(_ARTIFACT_SCHEMAS[artifact_type])


def build_response_format(artifact_type: ArtifactType) -> dict[str, Any]:
    """Return the response format of the meta FSM's ``build`` completion state.

    ``{"type": "json_schema", "json_schema": {"name", "schema"}}`` around
    :func:`artifact_schema`. For an agent the schema also requires
    ``agent_type``, an enum of ``AgentArtifactBuilder.VALID_AGENT_TYPES`` whose
    description lists each pattern, so the build call picks the agent pattern
    (no second classifier call, D-011). Never raises for an ``ArtifactType``
    member.
    """
    schema = artifact_schema(artifact_type)
    if artifact_type == ArtifactType.AGENT:
        patterns = "; ".join(
            f"{name}: {description}" for name, description in META_AGENT_PATTERN_INTENTS
        )
        schema["properties"]["agent_type"] = {
            "type": "string",
            "enum": sorted(AgentArtifactBuilder.VALID_AGENT_TYPES),
            "description": (
                f"The agent pattern that fits the requirement best. {patterns}"
            ),
        }
        schema["required"] = [*schema["required"], "agent_type"]
    return {
        "type": "json_schema",
        "json_schema": {"name": _BUILD_SCHEMA_NAME, "schema": schema},
    }


def build_artifact_prompt(artifact_type: ArtifactType, requirement: str) -> str:
    """Return the user turn of the build call for ``requirement``.

    Names the artifact type, embeds the requirement text verbatim and gives a
    per-type example; asks for a JSON object with actual values, not a
    schema. Never raises for an ``ArtifactType`` member.
    """
    type_label = artifact_type.value.upper()
    hint = _TYPE_EXAMPLES[artifact_type]
    return (
        f"<task>Design a {type_label} based on the user requirement below.</task>\n"
        f"<requirement>{requirement}</requirement>\n"
        f"<instructions>{hint}\n"
        f"Output ONLY a JSON object with actual values (not a schema). "
        f"Do NOT output type definitions.</instructions>"
    )


# ------------------------------------------------------------------
# Collect turn
# ------------------------------------------------------------------


def build_collect_response_instructions() -> str:
    """Return the Pass-2 instructions of the meta FSM's ``collect`` state.

    The reply reads the artifact type and the requirements gathered so far
    from context (and the last build's validation errors, if any) and ends
    with the build prompt the driver's keyword trigger listens for. The
    instructions end with that exact sentence: live on qwen3.5:4b, an "End
    with" clause followed by a length rule was dropped in 13 of 15 replies,
    and core's transition note made the first reply talk about the
    classify state, hence the no-internals rule.
    """
    return (
        "You are helping the user design an FSM-LLM artifact. The context "
        f"key {MetaContextKeys.ARTIFACT_TYPE} names its type (fsm, workflow "
        f"or agent) and {MetaContextKeys.REQUIREMENTS} lists everything the "
        "user has asked for so far; when "
        f"{MetaContextKeys.VALIDATION_ERRORS} is present, the last build "
        "failed for those reasons. In 2-3 sentences: acknowledge what the "
        "user just said, note what you will include in the artifact, and "
        "ask one short follow-up question about anything still unclear. "
        "Never mention states, transitions, classification or any other "
        "internals of this assistant. End your message with this exact "
        f"sentence: {META_BUILD_PROMPT}"
    )


# ------------------------------------------------------------------
# Review -- presentation helpers
# ------------------------------------------------------------------


def build_review_presentation(
    builder: ArtifactBuilder,
    artifact_type: ArtifactType,
) -> str:
    """Build the review presentation shown to the user."""
    errors = builder.validate_complete()
    warnings = builder.validate_partial()
    summary = builder.get_summary(detail_level="full")

    parts: list[str] = [
        f"Here is the {artifact_type.value.upper()} I built:\n",
        summary,
    ]

    if errors:
        parts.append(f"\nValidation errors ({len(errors)}):")
        for e in errors:
            parts.append(f"  - {e}")
    elif warnings:
        parts.append(f"\nValidation warnings ({len(warnings)}):")
        for w in warnings:
            parts.append(f"  - {w}")
    else:
        parts.append("\nValidation: passed (no errors)")

    parts.append(
        "\nWould you like to approve this artifact, or describe what "
        "changes you'd like me to make?"
    )

    return "\n".join(parts)


def build_welcome_message() -> str:
    """Build the welcome message for when no initial input is provided."""
    return (
        "Welcome! I can help you build:\n"
        "  1. An FSM (Finite State Machine) for stateful conversations\n"
        "  2. A Workflow for multi-step async processes\n"
        "  3. An Agent for tool-using AI agents\n\n"
        "Describe what you'd like to create."
    )


def build_followup_message(
    artifact_type: ArtifactType | None,
    has_name: bool,
    has_description: bool,
) -> str:
    """Build a follow-up question for missing intake fields."""
    if artifact_type is None:
        return (
            "What type of artifact would you like to build?\n"
            "  - FSM: for stateful conversations\n"
            "  - Workflow: for multi-step processes\n"
            "  - Agent: for tool-using AI agents"
        )

    missing: list[str] = []
    if not has_name:
        missing.append("a name")
    if not has_description:
        missing.append("a description of what it should do")

    type_label = artifact_type.value.upper()
    if missing:
        return (
            f"Building a {type_label}. I still need {' and '.join(missing)}. "
            f"You can also describe the components (states, steps, or tools) "
            f"you want included."
        )

    return f"Building a {type_label}. Describe the components you want included."


def build_output_message(artifact_json: str) -> str:
    """Build the final output message presenting the artifact JSON."""
    return (
        f"Your artifact is ready! Here is the JSON definition:\n\n"
        f"{artifact_json}\n\n"
        f"You can save this to a file and use it with fsm-llm."
    )
