from __future__ import annotations

"""
Prompt builders for agent tool awareness and observation formatting.
"""

from .tools import ToolRegistry


def build_think_extraction_instructions(
    registry: ToolRegistry,
    include_observations: bool = True,
) -> str:
    """Build extraction instructions for the think state."""
    tool_list = registry.to_prompt_description()

    if include_observations:
        intro = (
            "Analyze the task and all previous observations to decide your next action."
        )
    else:
        intro = "Analyze the task to decide your next action."

    parts = [
        intro,
        "",
        tool_list,
        "",
        "Extract the following as JSON:",
        '- "tool_name": name of the tool to use (must be one of the available tools), or "none" if no tool is needed',
        '- "tool_input": a JSON object with the parameters for the tool',
        '- "reasoning": your step-by-step reasoning for choosing this action',
        '- "should_terminate": true if you have enough information to answer the task, false otherwise',
    ]

    if include_observations:
        parts.extend(
            [
                "",
                "IMPORTANT: Review all previous observations carefully before deciding.",
                "If previous tool calls have provided sufficient information, set should_terminate to true.",
                "Do NOT repeat the same tool call with the same parameters.",
            ]
        )

    return "\n".join(parts)


def build_think_response_instructions() -> str:
    """Build response instructions for the think state."""
    return (
        "Briefly explain your reasoning and what action you are taking next. "
        "If you have decided to terminate, explain why you have enough information."
    )


def build_act_response_instructions() -> str:
    """Build response instructions for the act state."""
    return (
        "Summarize what tool was called and what was observed from the result. "
        "Be concise but include all relevant information from the tool output."
    )


def build_conclude_extraction_instructions() -> str:
    """Build extraction instructions for the conclude state."""
    return (
        "Based on all the observations gathered, formulate your final answer to the task.\n"
        "Extract:\n"
        '- "final_answer": your complete, well-structured answer to the original task\n'
        '- "confidence": your confidence in the answer (0.0 to 1.0)\n'
    )


def build_conclude_response_instructions() -> str:
    """Build response instructions for the conclude state."""
    return (
        "Present your final answer clearly and completely. "
        "Reference the evidence from your tool observations to support your answer."
    )


def build_approval_extraction_instructions() -> str:
    """Build extraction instructions for the approval-waiting state."""
    return (
        "The previous action requires human approval before execution.\n"
        "Wait for the user's response.\n"
        "Extract:\n"
        '- "approval_granted": true if the user approves, false if denied\n'
    )
