from __future__ import annotations

"""
FSM definition for the MetaBuilderAgent — hybrid architecture.

Builds a 4-state extraction-driven FSM:
  CLASSIFY → COLLECT → CONFIRM ↔ (self-loop on revise) → OUTPUT

The LLM collects data piece by piece using ``field_extractions``.
Software assembles the artifact deterministically on CONFIRM entry
via a POST_TRANSITION handler registered by MetaBuilderAgent.

States:
    - **CLASSIFY**: Extract artifact type (classification), name, description,
      persona (string fields). Self-loops on ``clarify``.
    - **COLLECT**: Extract component names as a list, plus a free-text
      description of connections/flow. Self-loops to accumulate more
      components if the user adds details.
    - **CONFIRM**: POST_TRANSITION handler assembles artifact on first entry
      (from COLLECT) or applies revision on self-loop. Presents artifact
      for approval. Self-loops on ``revise``.
    - **OUTPUT**: Terminal. Presents final artifact JSON.
"""

from typing import Any

from .constants import MetaBuilderStates


def build_meta_builder_fsm() -> dict[str, Any]:
    """Build the MetaBuilderAgent's 4-state hybrid FSM definition.

    Returns a dict compatible with ``fsm_llm.API.from_definition()``.
    """
    return {
        "name": "MetaBuilder",
        "description": (
            "Hybrid artifact builder: LLM collects data, software assembles"
        ),
        "initial_state": MetaBuilderStates.CLASSIFY,
        "persona": (
            "You are an artifact builder that helps users create FSM definitions, "
            "workflow definitions, agent configurations, and monitoring dashboards. "
            "Be concise and action-oriented. Infer as much as possible from the "
            "user's description. Never ask more than one clarifying question."
        ),
        "states": {
            # ----------------------------------------------------------
            # CLASSIFY: Determine artifact type + extract overview
            # ----------------------------------------------------------
            MetaBuilderStates.CLASSIFY: {
                "id": MetaBuilderStates.CLASSIFY,
                "description": "Classify artifact type and extract overview",
                "purpose": (
                    "Determine what the user wants to build and extract "
                    "name, description, and persona"
                ),
                "extraction_instructions": (
                    "Extract ALL of the following from the user's message. "
                    "Be AGGRESSIVE — infer everything you can:\n"
                    "- artifact_name: a short name for the artifact\n"
                    "- artifact_description: what it should do (use the full "
                    "user description)\n"
                    "- artifact_persona: personality or role if mentioned\n"
                    "- user_request: the user's full original message verbatim"
                ),
                "field_extractions": [
                    {
                        "field_name": "artifact_name",
                        "field_type": "str",
                        "extraction_instructions": (
                            "Extract or infer a short name for the artifact "
                            "the user wants to build. If none given, infer "
                            "from the description (e.g. 'Greeting Bot', "
                            "'Order Pipeline', 'Research Agent')."
                        ),
                        "required": False,
                    },
                    {
                        "field_name": "artifact_description",
                        "field_type": "str",
                        "extraction_instructions": (
                            "Extract what the artifact should do. Use the "
                            "user's full description even if not explicitly "
                            "labeled as a 'description'. Include mentions of "
                            "states, steps, tools, or components."
                        ),
                        "required": False,
                    },
                    {
                        "field_name": "artifact_persona",
                        "field_type": "str",
                        "extraction_instructions": (
                            "Extract the personality or role described for "
                            "the artifact, if any. Examples: 'friendly "
                            "receptionist', 'strict reviewer'. Return empty "
                            "string if not mentioned."
                        ),
                        "required": False,
                    },
                ],
                "classification_extractions": [
                    {
                        "field_name": "artifact_type",
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
                                "name": "monitor",
                                "description": (
                                    "Monitoring dashboard: metrics, alerts, "
                                    "observability, telemetry, dashboard panels, "
                                    "performance monitoring, health checks"
                                ),
                            },
                            {
                                "name": "clarify",
                                "description": (
                                    "User message is too vague to determine "
                                    "artifact type — no mention of "
                                    "conversations, processes, or tools"
                                ),
                            },
                        ],
                        "fallback_intent": "fsm",
                        "confidence_threshold": 0.4,
                        "required": True,
                    },
                ],
                "response_instructions": (
                    "If the artifact type could not be determined (classified "
                    "as 'clarify'), ask what type they want: FSM, Workflow, "
                    "Agent, or Monitor. Otherwise, briefly confirm what you'll "
                    "build and ask the user to list the components they want "
                    "(states, steps, tools, or panels)."
                ),
                "transitions": [
                    {
                        "target_state": MetaBuilderStates.CLASSIFY,
                        "description": "Need clarification on artifact type",
                        "priority": 200,
                        "conditions": [
                            {
                                "description": "Artifact type is unclear",
                                "logic": {
                                    "==": [{"var": "artifact_type"}, "clarify"]
                                },
                            }
                        ],
                    },
                    {
                        "target_state": MetaBuilderStates.COLLECT,
                        "description": "Artifact type known, collect components",
                        "priority": 100,
                        "conditions": [
                            {
                                "description": "Artifact type is known",
                                "logic": {
                                    "in": [
                                        {"var": "artifact_type"},
                                        ["fsm", "workflow", "agent", "monitor"],
                                    ]
                                },
                            }
                        ],
                    },
                ],
            },
            # ----------------------------------------------------------
            # COLLECT: Extract component names + descriptions
            # ----------------------------------------------------------
            MetaBuilderStates.COLLECT: {
                "id": MetaBuilderStates.COLLECT,
                "description": "Collect component names and descriptions",
                "purpose": (
                    "Extract the list of components (states, steps, tools, or "
                    "panels) the user wants, plus a description of how they "
                    "connect"
                ),
                "field_extractions": [
                    {
                        "field_name": "component_names",
                        "field_type": "list",
                        "extraction_instructions": (
                            "Extract ALL component names mentioned by the user "
                            "as a JSON array of strings. Components are:\n"
                            "- For FSMs: state names (e.g. greeting, help, bye)\n"
                            "- For workflows: step names (e.g. upload, process)\n"
                            "- For agents: tool names (e.g. search, summarize)\n"
                            "- For monitors: panel names (e.g. cpu, latency)\n"
                            "Include components from both the current message "
                            "AND any mentioned in the artifact_description from "
                            "earlier. Return [] if no components mentioned."
                        ),
                        "required": False,
                    },
                    {
                        "field_name": "component_flow",
                        "field_type": "str",
                        "extraction_instructions": (
                            "Describe how the components connect or flow. "
                            "For FSMs: which state leads to which. "
                            "For workflows: the step order. "
                            "For agents: how tools are used together. "
                            "For monitors: grouping or layout. "
                            "Return empty string if not described."
                        ),
                        "required": False,
                    },
                ],
                "extraction_instructions": (
                    "Extract the list of component names and how they connect. "
                    "Components are states (FSM), steps (workflow), tools "
                    "(agent), or panels (monitor)."
                ),
                "response_instructions": (
                    "If component_names is empty or has fewer than 2 items, "
                    "ask the user to list the components they want. Give "
                    "examples based on the artifact type.\n"
                    "If components were collected, confirm what you found "
                    "and say you're building the artifact now."
                ),
                "transitions": [
                    {
                        "target_state": MetaBuilderStates.COLLECT,
                        "description": "Need more component details",
                        "priority": 200,
                        "conditions": [
                            {
                                "description": "No components extracted yet",
                                "logic": {
                                    "!": [{"var": "component_names"}]
                                },
                            }
                        ],
                    },
                    {
                        "target_state": MetaBuilderStates.CONFIRM,
                        "description": (
                            "Components collected, assemble and present artifact"
                        ),
                        "priority": 100,
                        "conditions": [
                            {
                                "description": "At least one component extracted",
                                "logic": {
                                    "!!": [{"var": "component_names"}]
                                },
                            }
                        ],
                    },
                ],
            },
            # ----------------------------------------------------------
            # CONFIRM: Present for approval or revision
            # ----------------------------------------------------------
            MetaBuilderStates.CONFIRM: {
                "id": MetaBuilderStates.CONFIRM,
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
                    "artifact summary from context key 'builder_summary'. "
                    "Include validation status from 'validation_status'. "
                    "Ask: 'Would you like to approve this, or describe "
                    "what changes you'd like?'\n"
                    "If the user approved, confirm the artifact is finalized."
                ),
                "classification_extractions": [
                    {
                        "field_name": "review_decision",
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
                                    "User wants changes: modify, change, "
                                    "update, add, remove, fix, different, "
                                    "rename, wrong, not right, redo, try again"
                                ),
                            },
                        ],
                        "fallback_intent": "revise",
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
                        "target_state": MetaBuilderStates.CONFIRM,
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
