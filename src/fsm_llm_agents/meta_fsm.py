from __future__ import annotations

"""
FSM definition for the MetaBuilderAgent.

Builds a 3-state classification-driven FSM:
  INTAKE → REVIEW ↔ (self-loop on revise) → OUTPUT

Classification extractions drive routing at:
  - INTAKE: artifact_type (fsm/workflow/agent/clarify)
  - REVIEW: review_decision (approve/revise)

Build and revision operations happen in POST_TRANSITION handlers
registered by MetaBuilderAgent, not as separate FSM states.
"""

from typing import Any

from .constants import MetaBuilderStates


def build_meta_builder_fsm() -> dict[str, Any]:
    """Build the MetaBuilderAgent's 3-state FSM definition.

    Returns a dict compatible with ``fsm_llm.API.from_definition()``.

    States:
        - **INTAKE**: Extracts requirements and classifies artifact type.
          Transitions to REVIEW when type is determined (POST_TRANSITION
          handler builds the artifact). Self-loops on clarify.
        - **REVIEW**: Presents artifact for approval. Transitions to OUTPUT
          on approve. Self-loops on revise (POST_TRANSITION handler applies
          the revision).
        - **OUTPUT**: Terminal state. Presents final artifact JSON.
    """
    return {
        "name": "MetaBuilder",
        "description": "Interactive artifact builder using classification-driven routing",
        "initial_state": MetaBuilderStates.INTAKE,
        "persona": (
            "You are an artifact builder that helps users create FSM definitions, "
            "workflow definitions, and agent configurations. Be concise and "
            "action-oriented. Infer as much as possible from the user's description. "
            "Never ask more than one clarifying question at a time."
        ),
        "states": {
            # ----------------------------------------------------------
            # INTAKE: Extract requirements, classify artifact type
            # ----------------------------------------------------------
            MetaBuilderStates.INTAKE: {
                "id": MetaBuilderStates.INTAKE,
                "description": "Extract user requirements and classify artifact type",
                "purpose": (
                    "Determine what the user wants to build and gather enough "
                    "detail to proceed"
                ),
                "extraction_instructions": (
                    "Extract ALL of the following from the user's message. "
                    "Be AGGRESSIVE -- infer everything you can:\n"
                    "- artifact_name: name for the artifact (infer from context)\n"
                    "- artifact_description: what it should do (use the full user "
                    "description even if not labeled as 'description')\n"
                    "- artifact_persona: personality or role if mentioned\n"
                    "- components: list of states, steps, or tools mentioned\n"
                    "- user_request: the user's full original message verbatim"
                ),
                "response_instructions": (
                    "If the artifact type could not be determined (classified as "
                    "'clarify'), ask what type they want: FSM (conversations), "
                    "Workflow (multi-step processes), or Agent (tool-using AI). "
                    "Otherwise, briefly confirm what you'll build and that you're "
                    "proceeding. Do NOT ask follow-up questions if you have enough "
                    "to start building -- the builder will fill in defaults."
                ),
                "classification_extractions": [
                    {
                        "field_name": "artifact_type",
                        "schema": {
                            "intents": [
                                {
                                    "name": "fsm",
                                    "description": (
                                        "Stateful conversation: chatbot, dialogue, "
                                        "FAQ, interview, quiz, onboarding, support, "
                                        "survey, assistant, greeting, customer "
                                        "support, states, transitions"
                                    ),
                                },
                                {
                                    "name": "workflow",
                                    "description": (
                                        "Multi-step process: pipeline, ETL, batch, "
                                        "automation, data processing, sequential "
                                        "tasks, process flow, steps, orchestration"
                                    ),
                                },
                                {
                                    "name": "agent",
                                    "description": (
                                        "Tool-using autonomous agent: search, "
                                        "research, ReAct, browsing, task execution "
                                        "with tools, agentic, tool calling, actions"
                                    ),
                                },
                                {
                                    "name": "clarify",
                                    "description": (
                                        "User message is too vague to determine "
                                        "artifact type -- no mention of "
                                        "conversations, processes, or tools"
                                    ),
                                },
                            ],
                            "fallback_intent": "fsm",
                        },
                        "confidence_threshold": 0.4,
                        "required": True,
                    },
                ],
                "transitions": [
                    {
                        "target_state": MetaBuilderStates.INTAKE,
                        "description": "Need clarification on artifact type",
                        "priority": 200,
                        "conditions": [
                            {
                                "description": "Artifact type is unclear",
                                "logic": {"==": [{"var": "artifact_type"}, "clarify"]},
                            }
                        ],
                    },
                    {
                        "target_state": MetaBuilderStates.REVIEW,
                        "description": (
                            "Artifact type determined, proceed to build and review"
                        ),
                        "priority": 100,
                        "conditions": [
                            {
                                "description": "Artifact type is known",
                                "logic": {
                                    "in": [
                                        {"var": "artifact_type"},
                                        ["fsm", "workflow", "agent"],
                                    ]
                                },
                            }
                        ],
                    },
                ],
            },
            # ----------------------------------------------------------
            # REVIEW: Present artifact for approval or revision
            # ----------------------------------------------------------
            MetaBuilderStates.REVIEW: {
                "id": MetaBuilderStates.REVIEW,
                "description": "Present the built artifact for user approval",
                "purpose": "Get user approval or revision requests",
                "extraction_instructions": (
                    "Determine what the user wants:\n"
                    "- If they approve, set review_decision to 'approve'\n"
                    "- If they want changes, extract their full revision "
                    "request as 'revision_request'\n"
                    "Store the user's full message as 'user_request'"
                ),
                "response_instructions": (
                    "If the artifact was just built or revised, present the "
                    "artifact summary from the context key 'builder_summary'. "
                    "Include validation status from 'validation_status'. "
                    "Ask: 'Would you like to approve this artifact, or describe "
                    "what changes you'd like me to make?'\n"
                    "If the user approved, confirm the artifact is finalized."
                ),
                "classification_extractions": [
                    {
                        "field_name": "review_decision",
                        "schema": {
                            "intents": [
                                {
                                    "name": "approve",
                                    "description": (
                                        "User approves: yes, looks good, ship it, "
                                        "perfect, done, accept, ok, lgtm, great, "
                                        "fine, sure, approve, no changes, correct"
                                    ),
                                },
                                {
                                    "name": "revise",
                                    "description": (
                                        "User wants changes: modify, change, update, "
                                        "add, remove, fix, different, rename, wrong, "
                                        "not right, redo, try again, also, include"
                                    ),
                                },
                            ],
                            "fallback_intent": "revise",
                        },
                        "confidence_threshold": 0.5,
                    },
                ],
                "transitions": [
                    {
                        "target_state": MetaBuilderStates.OUTPUT,
                        "description": "User approves the artifact",
                        "priority": 100,
                        "conditions": [
                            {
                                "description": "User approved",
                                "logic": {
                                    "==": [
                                        {"var": "review_decision"},
                                        "approve",
                                    ]
                                },
                            }
                        ],
                    },
                    {
                        "target_state": MetaBuilderStates.REVIEW,
                        "description": "User wants changes, apply revision",
                        "priority": 200,
                        "conditions": [
                            {
                                "description": "User wants revision",
                                "logic": {
                                    "==": [
                                        {"var": "review_decision"},
                                        "revise",
                                    ]
                                },
                            }
                        ],
                    },
                ],
            },
            # ----------------------------------------------------------
            # OUTPUT: Terminal state
            # ----------------------------------------------------------
            MetaBuilderStates.OUTPUT: {
                "id": MetaBuilderStates.OUTPUT,
                "description": "Format and present the final artifact",
                "purpose": "Output the completed artifact as JSON",
                "extraction_instructions": "No extraction needed.",
                "response_instructions": (
                    "Present the final artifact JSON from the context key "
                    "'artifact_json'. Inform the user they can save it to a "
                    "file and use it with fsm-llm."
                ),
                "transitions": [],
            },
        },
    }
