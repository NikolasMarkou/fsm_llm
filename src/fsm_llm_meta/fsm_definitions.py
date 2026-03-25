from __future__ import annotations

"""
Pre-built FSM definition for the meta-agent's conversational flow.
"""

from typing import Any

from .constants import ContextKeys, MetaStates
from .prompts import (
    build_classify_extraction_instructions,
    build_classify_response_instructions,
    build_output_response_instructions,
    build_welcome_response_instructions,
)


def build_meta_fsm() -> dict[str, Any]:
    """
    Build the FSM definition that drives the meta-agent conversation.

    States:
        welcome -> classify -> gather_overview -> design_structure
        -> define_connections -> review -> output (terminal)

    The design_structure and define_connections states loop until the user
    signals they are done. The review state can loop back to design_structure
    if revisions are needed.

    Note: The extraction/response instructions for dynamic states
    (gather_overview, design_structure, define_connections, review) are
    injected at runtime by handlers based on the current builder state.
    Static states (welcome, classify, output) have fixed instructions.
    """
    persona = (
        "You are a friendly FSM-LLM architect assistant. You help users "
        "design and build FSM definitions, workflow configurations, and "
        "agent setups through interactive conversation. You are knowledgeable "
        "about state machine design patterns, workflow orchestration, and "
        "agentic AI patterns. Guide the user step by step."
    )

    states: dict[str, Any] = {
        MetaStates.WELCOME: {
            "id": MetaStates.WELCOME,
            "description": "Welcome the user and ask what they want to build",
            "purpose": "Introduce the meta-agent and determine user intent",
            "response_instructions": build_welcome_response_instructions(),
            "transitions": [
                {
                    "target_state": MetaStates.CLASSIFY,
                    "description": "Always transition to classify after welcome",
                    "priority": 100,
                }
            ],
        },
        MetaStates.CLASSIFY: {
            "id": MetaStates.CLASSIFY,
            "description": "Determine what type of artifact the user wants to build",
            "purpose": "Extract artifact type (FSM, Workflow, or Agent) from user input",
            "extraction_instructions": build_classify_extraction_instructions(),
            "response_instructions": build_classify_response_instructions(),
            "transitions": [
                {
                    "target_state": MetaStates.GATHER_OVERVIEW,
                    "description": "Artifact type has been determined",
                    "priority": 100,
                    "conditions": [
                        {
                            "description": "Artifact type is set",
                            "logic": {"has_context": ContextKeys.ARTIFACT_TYPE},
                        }
                    ],
                }
            ],
        },
        MetaStates.GATHER_OVERVIEW: {
            "id": MetaStates.GATHER_OVERVIEW,
            "description": "Gather name, description, and persona for the artifact",
            "purpose": "Collect basic metadata before designing structure",
            "extraction_instructions": (
                "Extract overview fields from user input.\n"
                "Respond with JSON: "
                '{"artifact_name": "...", "artifact_description": "...", '
                '"artifact_persona": "..." or null}'
            ),
            "response_instructions": (
                "Ask the user for the artifact name and description. "
                "For FSMs, also ask about a persona."
            ),
            "transitions": [
                {
                    "target_state": MetaStates.DESIGN_STRUCTURE,
                    "description": "Overview is complete, begin structure design",
                    "priority": 100,
                    "conditions": [
                        {
                            "description": "Name and description are provided",
                            "logic": {
                                "and": [
                                    {"has_context": ContextKeys.ARTIFACT_NAME},
                                    {"has_context": ContextKeys.ARTIFACT_DESCRIPTION},
                                ]
                            },
                        }
                    ],
                }
            ],
        },
        MetaStates.DESIGN_STRUCTURE: {
            "id": MetaStates.DESIGN_STRUCTURE,
            "description": "Iteratively design states, steps, or tools",
            "purpose": "Build the artifact's components through conversation",
            "extraction_instructions": (
                "Extract the user's action. Respond with JSON: "
                '{"action": "<action>", "action_params": {...}}'
            ),
            "response_instructions": (
                "Help the user add components to their artifact. "
                "Confirm each addition and ask about the next one."
            ),
            "transitions": [
                {
                    "target_state": MetaStates.DEFINE_CONNECTIONS,
                    "description": "User is done adding components",
                    "priority": 100,
                    "conditions": [
                        {
                            "description": "User signaled structure is done",
                            "logic": {
                                "==": [{"var": ContextKeys.STRUCTURE_DONE}, True]
                            },
                        }
                    ],
                }
            ],
        },
        MetaStates.DEFINE_CONNECTIONS: {
            "id": MetaStates.DEFINE_CONNECTIONS,
            "description": "Define transitions and connections between components",
            "purpose": "Wire up the artifact's components",
            "extraction_instructions": (
                "Extract the user's connection action. Respond with JSON: "
                '{"action": "<action>", "action_params": {...}}'
            ),
            "response_instructions": (
                "Help the user define connections between components. "
                "Suggest logical connections when appropriate."
            ),
            "transitions": [
                {
                    "target_state": MetaStates.REVIEW,
                    "description": "User is done defining connections",
                    "priority": 100,
                    "conditions": [
                        {
                            "description": "User signaled connections are done",
                            "logic": {
                                "==": [{"var": ContextKeys.CONNECTIONS_DONE}, True]
                            },
                        }
                    ],
                }
            ],
        },
        MetaStates.REVIEW: {
            "id": MetaStates.REVIEW,
            "description": "Review the complete artifact and validate",
            "purpose": "Present summary, run validation, get user approval",
            "extraction_instructions": (
                "Extract user decision. Respond with JSON: "
                '{"user_decision": "approve" or "revise"}'
            ),
            "response_instructions": (
                "Present the artifact summary and validation results. "
                "Ask the user to approve or request revisions."
            ),
            "transitions": [
                {
                    "target_state": MetaStates.OUTPUT,
                    "description": "User approves the artifact",
                    "priority": 10,
                    "conditions": [
                        {
                            "description": "User approved",
                            "logic": {
                                "==": [
                                    {"var": ContextKeys.USER_DECISION},
                                    "approve",
                                ]
                            },
                        }
                    ],
                },
                {
                    "target_state": MetaStates.DESIGN_STRUCTURE,
                    "description": "User wants to revise the artifact",
                    "priority": 200,
                    "conditions": [
                        {
                            "description": "User wants revisions",
                            "logic": {
                                "==": [
                                    {"var": ContextKeys.USER_DECISION},
                                    "revise",
                                ]
                            },
                        }
                    ],
                },
            ],
        },
        MetaStates.OUTPUT: {
            "id": MetaStates.OUTPUT,
            "description": "Present the final artifact",
            "purpose": "Output the completed and validated artifact",
            "response_instructions": build_output_response_instructions(),
            "transitions": [],  # Terminal state
        },
    }

    return {
        "name": "meta_agent",
        "description": (
            "A conversational agent that helps users interactively build "
            "FSM definitions, workflow configurations, and agent setups"
        ),
        "initial_state": MetaStates.WELCOME,
        "persona": persona,
        "version": "4.1",
        "states": states,
    }
