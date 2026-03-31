from __future__ import annotations

"""
Prompt builders for the meta-agent — hybrid architecture.

The monolithic spec generation prompts have been removed. Data collection
now happens via the FSM's ``field_extractions`` pipeline. This module
retains: review presentation, welcome message, follow-up helpers, and
output formatting.
"""

from .definitions import ArtifactType
from .meta_builders import ArtifactBuilder

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
        "  3. An Agent for tool-using AI agents\n"
        "  4. A Monitor dashboard for metrics and alerts\n\n"
        "Tell me what you'd like to create. The more detail you provide "
        "(name, description, states/steps/tools/panels), the faster I can build it."
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
            "  - Agent: for tool-using AI agents\n"
            "  - Monitor: for metrics and alerts dashboards"
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
