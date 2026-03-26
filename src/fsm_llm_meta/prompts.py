from __future__ import annotations

"""
Prompt builders for static states of the meta-agent FSM.

Only the states with fixed instructions (welcome, classify, output)
use prompt builders.  Dynamic states (gather_overview, design_structure,
define_connections, review) receive their context through builder-state
injection via handlers — see ``handlers.py``.
"""


def build_welcome_response_instructions() -> str:
    return (
        "Welcome the user and ask what type of artifact they want to build. "
        "Explain that you can help them build:\n"
        "1. An FSM (Finite State Machine) - for stateful conversations\n"
        "2. A Workflow - for multi-step async processes\n"
        "3. An Agent - for tool-using AI agents\n\n"
        "Ask them which one they'd like to create."
    )


def build_classify_extraction_instructions() -> str:
    return (
        "Extract the type of artifact the user wants to build.\n\n"
        "You MUST respond with ONLY valid JSON, no other text.\n\n"
        "Example responses:\n"
        '{"artifact_type": "fsm"}\n'
        '{"artifact_type": "workflow"}\n'
        '{"artifact_type": "agent"}\n\n'
        "The artifact_type MUST be exactly one of: fsm, workflow, agent\n\n"
        "If the user hasn't clearly specified, infer from context:\n"
        '- "conversation", "chatbot", "states" → "fsm"\n'
        '- "pipeline", "process", "steps", "automation" → "workflow"\n'
        '- "tools", "search", "react", "actions" → "agent"'
    )


def build_classify_response_instructions() -> str:
    return (
        "Confirm what type of artifact you'll help build based on their choice. "
        "Then ask them for a name, description, and (for FSMs) an optional persona. "
        "Be encouraging and conversational."
    )


def build_output_response_instructions() -> str:
    return (
        "The artifact has been built successfully! "
        "Present the final JSON to the user and let them know it's ready. "
        "Mention that they can save it to a file and use it with fsm-llm."
    )
