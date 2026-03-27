from __future__ import annotations

"""
Prompt builders for the meta-agent's phases: intake, build, and review.
"""

from .definitions import ArtifactType
from .meta_builders import ArtifactBuilder

# ------------------------------------------------------------------
# Phase 1: Intake -- extract requirements from user message
# ------------------------------------------------------------------

INTAKE_SYSTEM_PROMPT = (
    "You are a JSON extraction assistant. Extract fields from the user's message. "
    "Be AGGRESSIVE -- infer everything you can. If the user mentions states, "
    "transitions, chatbot, conversation, or dialogue, artifact_type is 'fsm'. "
    "If they mention pipeline, process, steps, or automation, it is 'workflow'. "
    "If they mention tools, search, react, or actions, it is 'agent'. "
    "Use the ENTIRE user message as artifact_description if they describe what "
    "the artifact should do, even if not labeled as a 'description'. "
    "Respond with ONLY valid JSON."
)

INTAKE_EXTRACTION_PROMPT = (
    "Extract ALL available information. Return JSON with these fields "
    "(use null ONLY if truly absent -- prefer inferring over null):\n"
    '  "artifact_type": "fsm" or "workflow" or "agent"\n'
    '  "artifact_name": name (infer from context if not explicit)\n'
    '  "artifact_description": what it should do (use the full user description)\n'
    '  "artifact_persona": personality/role if mentioned\n'
    '  "components": list of states/steps/tools/transitions mentioned\n\n'
    "Aliases -- infer the type from keywords:\n"
    '  "conversation", "chatbot", "states", "dialogue" -> "fsm"\n'
    '  "pipeline", "process", "steps", "automation" -> "workflow"\n'
    '  "tools", "search", "react", "actions" -> "agent"\n'
    "\nExamples:\n"
    'User: "build an fsm"\n'
    '-> {"artifact_type": "fsm", "artifact_name": null, '
    '"artifact_description": null, "artifact_persona": null, "components": []}\n\n'
    'User: "I want a chatbot with 3 states: greeting, help, goodbye"\n'
    '-> {"artifact_type": "fsm", "artifact_name": "Chatbot", '
    '"artifact_description": "A chatbot with greeting, help, and goodbye states", '
    '"artifact_persona": null, '
    '"components": ["greeting state", "help state", "goodbye state"]}'
)


def build_intake_user_message(conversation_history: list[dict[str, str]]) -> str:
    """Build the user message for intake extraction from conversation history."""
    parts: list[str] = []
    for msg in conversation_history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user" and content:
            parts.append(content)
    combined = "\n".join(parts)
    return f"User message(s):\n{combined}\n\n{INTAKE_EXTRACTION_PROMPT}"


# ------------------------------------------------------------------
# Phase 2: Build -- task prompt for ReactAgent with builder tools
# ------------------------------------------------------------------

BUILD_SPEC_SYSTEM_PROMPT = (
    "You are an artifact builder. Generate a COMPLETE artifact specification "
    "as a single JSON object. Respond with ONLY valid JSON, no other text. "
    "Be creative and fill in reasonable defaults for anything not specified."
)

_FSM_SPEC_SCHEMA = (
    "Generate a complete FSM (Finite State Machine) as JSON with this EXACT format:\n"
    "{\n"
    '  "name": "string",\n'
    '  "description": "string",\n'
    '  "persona": "string or empty",\n'
    '  "initial_state": "state_id of the first state",\n'
    '  "states": [\n'
    "    {\n"
    '      "id": "unique_state_id",\n'
    '      "description": "what this state does (short)",\n'
    '      "purpose": "what should be accomplished here (short)",\n'
    '      "extraction_instructions": "what data to extract from user (short, <200 chars)",\n'
    '      "response_instructions": "how to respond to the user (short, <200 chars)"\n'
    "    }\n"
    "  ],\n"
    '  "transitions": [\n'
    '    {"source": "state_id", "target": "state_id", "description": "when to transition"}\n'
    "  ]\n"
    "}\n\n"
    "Rules:\n"
    "- MUST have at least 2 states\n"
    "- MUST have at least 1 transition\n"
    "- The last state should be terminal (no outgoing transitions)\n"
    "- Keep ALL string fields SHORT (under 200 characters)\n"
    "- If user gave no details, invent a simple greeting chatbot\n"
)

_WORKFLOW_SPEC_SCHEMA = (
    "Generate a complete Workflow as JSON with this EXACT format:\n"
    "{\n"
    '  "workflow_id": "snake_case_id",\n'
    '  "name": "string",\n'
    '  "description": "string",\n'
    '  "initial_step_id": "first_step_id",\n'
    '  "steps": [\n'
    "    {\n"
    '      "id": "unique_step_id",\n'
    '      "name": "Step Name",\n'
    '      "step_type": "auto_transition or llm_processing or condition or wait_for_event",\n'
    '      "description": "what this step does",\n'
    '      "next_step": "next_step_id or null for last step"\n'
    "    }\n"
    "  ]\n"
    "}\n\n"
    "Rules:\n"
    "- MUST have at least 2 steps\n"
    "- Valid step_types: auto_transition, api_call, condition, llm_processing, "
    "wait_for_event, timer, parallel, conversation\n"
    "- If user gave no details, invent a simple 3-step workflow\n"
)

_AGENT_SPEC_SCHEMA = (
    "Generate a complete Agent configuration as JSON with this EXACT format:\n"
    "{\n"
    '  "name": "string",\n'
    '  "description": "string",\n'
    '  "agent_type": "react or plan_execute or reflexion",\n'
    '  "tools": [\n'
    '    {"name": "tool_name", "description": "what this tool does"}\n'
    "  ],\n"
    '  "config": {"max_iterations": 10, "temperature": 0.5}\n'
    "}\n\n"
    "Rules:\n"
    "- Valid agent_types: react, plan_execute, reflexion, rewoo, evaluator_optimizer, "
    "maker_checker, prompt_chain, self_consistency, debate, orchestrator, adapt\n"
    "- MUST have at least 1 tool\n"
    "- If user gave no details, create a react agent with a search tool\n"
)

_SPEC_SCHEMAS: dict[str, str] = {
    "fsm": _FSM_SPEC_SCHEMA,
    "workflow": _WORKFLOW_SPEC_SCHEMA,
    "agent": _AGENT_SPEC_SCHEMA,
}


def build_spec_prompt(
    artifact_type: ArtifactType,
    name: str | None,
    description: str | None,
    persona: str | None,
    components: list[str] | None,
    user_messages: str = "",
) -> str:
    """Build a prompt that asks the LLM to generate the full artifact spec as JSON."""
    parts: list[str] = []

    schema = _SPEC_SCHEMAS.get(artifact_type.value, _FSM_SPEC_SCHEMA)
    parts.append(schema)

    # User context
    context_parts: list[str] = []
    if name:
        context_parts.append(f"Name: {name}")
    if description:
        context_parts.append(f"Description: {description}")
    if persona:
        context_parts.append(f"Persona: {persona}")
    if components:
        context_parts.append("Components: " + ", ".join(components))
    if user_messages:
        context_parts.append(f"User request: {user_messages}")

    if context_parts:
        parts.append("User specifications:\n" + "\n".join(context_parts))
    else:
        parts.append("No specific requirements given -- generate a creative example.")

    parts.append("\nRespond with ONLY the JSON object.")
    return "\n".join(parts)


def build_revision_spec_prompt(
    artifact_type: ArtifactType,
    revision_request: str,
    current_spec: str,
) -> str:
    """Build a prompt to revise an existing artifact spec."""
    schema = _SPEC_SCHEMAS.get(artifact_type.value, _FSM_SPEC_SCHEMA)
    return (
        f"{schema}\n"
        f"Current artifact:\n{current_spec}\n\n"
        f"User wants these changes: {revision_request}\n\n"
        f"Output the COMPLETE updated JSON (not just the changes). "
        f"Respond with ONLY the JSON object."
    )


# ------------------------------------------------------------------
# Phase 3: Review -- presentation helpers
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
        "Tell me what you'd like to create. The more detail you provide "
        "(name, description, states/steps/tools), the faster I can build it."
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
