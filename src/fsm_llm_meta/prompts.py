from __future__ import annotations

"""
Prompt builders for the meta-agent's three phases: intake, build, and review.
"""


from .builders import ArtifactBuilder
from .definitions import ArtifactType

# ------------------------------------------------------------------
# Phase 1: Intake — extract requirements from user message
# ------------------------------------------------------------------

_TYPE_ALIASES = (
    'Aliases — infer the type from keywords:\n'
    '  "conversation", "chatbot", "states", "dialogue" -> "fsm"\n'
    '  "pipeline", "process", "steps", "automation" -> "workflow"\n'
    '  "tools", "search", "react", "actions" -> "agent"\n'
)

INTAKE_SYSTEM_PROMPT = (
    "You are a JSON extraction assistant for an artifact builder. "
    "Extract as many fields as possible from the user's message. "
    "Respond with ONLY valid JSON, no other text."
)

INTAKE_EXTRACTION_PROMPT = (
    "Extract all available information from the user's message.\n\n"
    "Return a JSON object with these fields (use null for missing):\n"
    '  "artifact_type": one of "fsm", "workflow", "agent" (REQUIRED)\n'
    '  "artifact_name": name for the artifact\n'
    '  "artifact_description": what it does\n'
    '  "artifact_persona": personality/role (mainly for FSMs)\n'
    '  "components": list of component descriptions the user mentioned '
    "(states, steps, tools, transitions, etc.)\n\n"
    + _TYPE_ALIASES
    + "\nExample:\n"
    '{"artifact_type": "fsm", "artifact_name": "CustomerBot", '
    '"artifact_description": "Handles customer support inquiries", '
    '"artifact_persona": "A helpful support agent", '
    '"components": ["greeting state", "issue classification state", '
    '"resolution state", "farewell state", '
    '"transition from greeting to classification"]}'
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
# Phase 2: Build — task prompt for the ReactAgent
# ------------------------------------------------------------------

_FSM_INSTRUCTIONS = (
    "Build this FSM step by step:\n"
    "1. Call set_overview with the name, description, and persona\n"
    "2. Add ALL states the user described (use add_state)\n"
    "3. For each state, provide a clear description and purpose\n"
    "4. Add logical transitions between states (use add_transition)\n"
    "5. Ensure there is at least one terminal state (no outgoing transitions)\n"
    "6. Call validate() to check for errors\n"
    "7. Fix any validation errors by adding missing states/transitions\n"
    "8. When the artifact is valid, conclude with a brief summary"
)

_WORKFLOW_INSTRUCTIONS = (
    "Build this workflow step by step:\n"
    "1. Call set_overview with the workflow ID, name, and description\n"
    "2. Add ALL steps the user described (use add_step with appropriate step_type)\n"
    "3. Connect steps with transitions (use set_step_transition)\n"
    "4. Call validate() to check for errors\n"
    "5. Fix any validation errors\n"
    "6. When the artifact is valid, conclude with a brief summary"
)

_AGENT_INSTRUCTIONS = (
    "Build this agent step by step:\n"
    "1. Call set_overview with the name and description\n"
    "2. Set the agent type (use set_agent_type)\n"
    "3. Add ALL tools the user described (use add_tool)\n"
    "4. Optionally adjust config if the user specified preferences\n"
    "5. Call validate() to check for errors\n"
    "6. Fix any validation errors\n"
    "7. When the artifact is valid, conclude with a brief summary"
)


def build_task_prompt(
    artifact_type: ArtifactType,
    name: str | None,
    description: str | None,
    persona: str | None,
    components: list[str] | None,
    user_messages: str = "",
) -> str:
    """Build the task prompt for the ReactAgent's build phase."""
    parts: list[str] = []

    # Header
    type_label = artifact_type.value.upper()
    parts.append(f"Build a {type_label} artifact with the following specifications:")

    # Metadata
    if name:
        parts.append(f"Name: {name}")
    if description:
        parts.append(f"Description: {description}")
    if persona:
        parts.append(f"Persona: {persona}")

    # Component hints
    if components:
        parts.append("\nUser-described components:")
        for c in components:
            parts.append(f"  - {c}")

    # Include raw user messages for context
    if user_messages:
        parts.append(f"\nOriginal user request:\n{user_messages}")

    # Type-specific instructions
    parts.append("")
    if artifact_type == ArtifactType.FSM:
        parts.append(_FSM_INSTRUCTIONS)
    elif artifact_type == ArtifactType.WORKFLOW:
        parts.append(_WORKFLOW_INSTRUCTIONS)
    elif artifact_type == ArtifactType.AGENT:
        parts.append(_AGENT_INSTRUCTIONS)

    parts.append(
        "\nIMPORTANT: Do NOT ask the user questions. Build the artifact "
        "based on what was described. Make reasonable design decisions for "
        "any unspecified details. Always call validate() before concluding."
    )

    return "\n".join(parts)


def build_revision_prompt(
    revision_request: str,
    builder_summary: str,
) -> str:
    """Build the task prompt for revising an existing artifact."""
    return (
        f"Modify the existing artifact based on this feedback:\n"
        f"{revision_request}\n\n"
        f"Current artifact state:\n{builder_summary}\n\n"
        f"Make the requested changes, then call validate() and conclude "
        f"with a summary of what was changed.\n\n"
        f"IMPORTANT: Do NOT rebuild from scratch. Only modify what was "
        f"requested. The existing structure should be preserved unless "
        f"the user explicitly asked to change it."
    )


# ------------------------------------------------------------------
# Phase 3: Review — presentation helpers
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
