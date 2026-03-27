from __future__ import annotations

"""
FSM definition for the MetaBuilderAgent.

Builds a classification-driven FSM with 6 states:
  INTAKE → PLAN → BUILD → REVIEW → OUTPUT
                             ↑         |
                             └─ REVISE ←┘

Classification extractions drive routing at:
  - INTAKE: artifact_type + intake_completeness
  - PLAN: agent_pattern (only meaningful for agent artifacts)
  - REVIEW: review_decision
  - REVISE: revision_scope
"""

from typing import Any

from .constants import MetaBuilderStates


def build_meta_builder_fsm() -> dict[str, Any]:
    """Build the MetaBuilderAgent's FSM definition with classification_extractions.

    Returns a dict compatible with fsm_llm.API.from_definition().
    """
    return {
        "name": "MetaBuilder",
        "description": "Interactive artifact builder using classification-driven routing",
        "initial_state": MetaBuilderStates.INTAKE,
        "persona": (
            "You are an artifact builder that helps users create FSM definitions, "
            "workflow definitions, and agent configurations. Be concise and action-oriented. "
            "Infer as much as possible from the user's description."
        ),
        "states": {
            # ----------------------------------------------------------
            # INTAKE: Extract requirements, classify artifact type
            # ----------------------------------------------------------
            MetaBuilderStates.INTAKE: {
                "id": MetaBuilderStates.INTAKE,
                "description": "Extract user requirements and classify artifact type",
                "purpose": "Determine what the user wants to build and gather enough detail to proceed",
                "extraction_instructions": (
                    "Extract: artifact_name, artifact_description, artifact_persona, "
                    "components (states/steps/tools mentioned). "
                    "Be aggressive -- infer everything you can from the user's message."
                ),
                "response_instructions": (
                    "If you have enough information (type + at least a brief description), "
                    "acknowledge what you'll build and confirm you're proceeding. "
                    "If the user's message is too vague to determine anything, "
                    "ask what type of artifact they want (FSM, Workflow, or Agent)."
                ),
                "classification_extractions": [
                    {
                        "field_name": "artifact_type",
                        "intents": [
                            {
                                "name": "fsm",
                                "description": (
                                    "Stateful conversation: chatbot, dialogue, FAQ, "
                                    "interview, quiz, onboarding, support desk, survey, "
                                    "assistant, greeting, customer support, states, transitions"
                                ),
                            },
                            {
                                "name": "workflow",
                                "description": (
                                    "Multi-step process: pipeline, ETL, batch job, "
                                    "automation, data processing, sequential tasks, "
                                    "process flow, steps, orchestration"
                                ),
                            },
                            {
                                "name": "agent",
                                "description": (
                                    "Tool-using autonomous agent: search, research, "
                                    "ReAct, browsing, task execution with tools, "
                                    "agentic, tool calling, actions"
                                ),
                            },
                            {
                                "name": "clarify",
                                "description": (
                                    "User message is too vague to determine artifact type -- "
                                    "no mention of conversations, processes, or tools"
                                ),
                            },
                        ],
                        "fallback_intent": "fsm",
                        "confidence_threshold": 0.4,
                        "required": True,
                    },
                    {
                        "field_name": "intake_completeness",
                        "intents": [
                            {
                                "name": "complete",
                                "description": (
                                    "User provided enough detail: name/purpose/components "
                                    "are clear or easily inferable from the message"
                                ),
                            },
                            {
                                "name": "minimal",
                                "description": (
                                    "User gave minimal info but indicated to proceed: "
                                    "'just build it', 'use defaults', 'whatever', 'don't care'"
                                ),
                            },
                            {
                                "name": "incomplete",
                                "description": (
                                    "Key details missing and user hasn't indicated to skip: "
                                    "no description, no components, unclear purpose"
                                ),
                            },
                        ],
                        "fallback_intent": "complete",
                        "confidence_threshold": 0.3,
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
                                "logic": {
                                    "==": [{"var": "artifact_type"}, "clarify"]
                                },
                            }
                        ],
                    },
                    {
                        "target_state": MetaBuilderStates.INTAKE,
                        "description": "Need more information from user",
                        "priority": 150,
                        "conditions": [
                            {
                                "description": "Intake is incomplete",
                                "logic": {
                                    "==": [
                                        {"var": "intake_completeness"},
                                        "incomplete",
                                    ]
                                },
                            }
                        ],
                    },
                    {
                        "target_state": MetaBuilderStates.PLAN,
                        "description": "Ready to plan the build",
                        "priority": 100,
                        "conditions": [
                            {
                                "description": "Have enough info to proceed",
                                "logic": {
                                    "or": [
                                        {
                                            "==": [
                                                {"var": "intake_completeness"},
                                                "complete",
                                            ]
                                        },
                                        {
                                            "==": [
                                                {"var": "intake_completeness"},
                                                "minimal",
                                            ]
                                        },
                                    ]
                                },
                            }
                        ],
                    },
                ],
            },
            # ----------------------------------------------------------
            # PLAN: Prepare build strategy (classify agent pattern if needed)
            # ----------------------------------------------------------
            MetaBuilderStates.PLAN: {
                "id": MetaBuilderStates.PLAN,
                "description": "Prepare the build strategy based on artifact type",
                "purpose": "Set up the builder and classify agent pattern if building an agent",
                "extraction_instructions": "No additional extraction needed.",
                "response_instructions": (
                    "Briefly confirm what will be built. If building an agent, "
                    "mention the selected pattern."
                ),
                "classification_extractions": [
                    {
                        "field_name": "agent_pattern",
                        "intents": [
                            {
                                "name": "react",
                                "description": "General tool-using agent with think-act-observe loop",
                            },
                            {
                                "name": "plan_execute",
                                "description": "Plan all steps first, then execute sequentially",
                            },
                            {
                                "name": "reflexion",
                                "description": "Self-reflecting agent that learns from mistakes",
                            },
                            {
                                "name": "orchestrator",
                                "description": "Delegates to worker agents, synthesizes results",
                            },
                            {
                                "name": "debate",
                                "description": "Multiple perspectives argue, judge decides",
                            },
                            {
                                "name": "maker_checker",
                                "description": "Draft-review verification loop",
                            },
                            {
                                "name": "rewoo",
                                "description": "Plan all tool calls upfront, execute without LLM between",
                            },
                            {
                                "name": "prompt_chain",
                                "description": "Sequential prompt pipeline with quality gates",
                            },
                            {
                                "name": "evaluator_optimizer",
                                "description": "Iterative evaluation and optimization loop",
                            },
                            {
                                "name": "self_consistency",
                                "description": "Multiple samples with majority voting",
                            },
                            {
                                "name": "adapt",
                                "description": "Adaptive complexity with decomposition on failure",
                            },
                        ],
                        "fallback_intent": "react",
                        "confidence_threshold": 0.4,
                    },
                ],
                "transitions": [
                    {
                        "target_state": MetaBuilderStates.BUILD,
                        "description": "Proceed to autonomous build phase",
                        "priority": 100,
                    },
                ],
            },
            # ----------------------------------------------------------
            # BUILD: Autonomous construction (handler spawns ReactAgent)
            # ----------------------------------------------------------
            MetaBuilderStates.BUILD: {
                "id": MetaBuilderStates.BUILD,
                "description": "Autonomously construct the artifact using builder tools",
                "purpose": "Build the complete artifact specification",
                "extraction_instructions": "No extraction needed -- this state is handler-driven.",
                "response_instructions": "Report build progress and results.",
                "transitions": [
                    {
                        "target_state": MetaBuilderStates.REVIEW,
                        "description": "Build phase complete, present for review",
                        "priority": 100,
                        "conditions": [
                            {
                                "description": "Build is complete",
                                "logic": {
                                    "==": [{"var": "build_complete"}, True]
                                },
                            }
                        ],
                    },
                ],
            },
            # ----------------------------------------------------------
            # REVIEW: Present artifact for approval
            # ----------------------------------------------------------
            MetaBuilderStates.REVIEW: {
                "id": MetaBuilderStates.REVIEW,
                "description": "Present the built artifact for user approval",
                "purpose": "Get user approval or revision requests",
                "extraction_instructions": "Determine if the user approves or wants changes.",
                "response_instructions": (
                    "If the user approves, confirm and proceed to output. "
                    "If they want changes, acknowledge and ask for specifics. "
                    "If they ask a question, answer it and re-present the artifact."
                ),
                "classification_extractions": [
                    {
                        "field_name": "review_decision",
                        "intents": [
                            {
                                "name": "approve",
                                "description": (
                                    "User approves the artifact: yes, looks good, "
                                    "ship it, perfect, done, accept, ok, lgtm, "
                                    "great, fine, sure, approve, no changes"
                                ),
                            },
                            {
                                "name": "revise",
                                "description": (
                                    "User wants changes: modify, change, update, "
                                    "add, remove, fix, different, rename, wrong, "
                                    "not right, redo, try again"
                                ),
                            },
                            {
                                "name": "clarify",
                                "description": (
                                    "User asks a question about the artifact or "
                                    "wants explanation, not requesting changes"
                                ),
                            },
                        ],
                        "fallback_intent": "clarify",
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
                        "target_state": MetaBuilderStates.REVISE,
                        "description": "User wants changes",
                        "priority": 150,
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
                    {
                        "target_state": MetaBuilderStates.REVIEW,
                        "description": "User needs clarification",
                        "priority": 200,
                        "conditions": [
                            {
                                "description": "User asked a question",
                                "logic": {
                                    "==": [
                                        {"var": "review_decision"},
                                        "clarify",
                                    ]
                                },
                            }
                        ],
                    },
                ],
            },
            # ----------------------------------------------------------
            # REVISE: Apply user-requested changes
            # ----------------------------------------------------------
            MetaBuilderStates.REVISE: {
                "id": MetaBuilderStates.REVISE,
                "description": "Apply user-requested changes to the artifact",
                "purpose": "Classify revision scope and apply changes",
                "extraction_instructions": "Extract the specific changes the user wants.",
                "response_instructions": "Acknowledge the changes and apply them.",
                "classification_extractions": [
                    {
                        "field_name": "revision_scope",
                        "intents": [
                            {
                                "name": "targeted",
                                "description": (
                                    "Small change: rename, add one state/tool, "
                                    "change a description, adjust a transition, "
                                    "tweak a field"
                                ),
                            },
                            {
                                "name": "structural",
                                "description": (
                                    "Major change: different pattern, reorganize "
                                    "states, change artifact type, start over, "
                                    "completely different approach"
                                ),
                            },
                            {
                                "name": "config",
                                "description": (
                                    "Configuration change: model, temperature, "
                                    "timeout, thresholds, max iterations"
                                ),
                            },
                        ],
                        "fallback_intent": "targeted",
                        "confidence_threshold": 0.4,
                    },
                ],
                "transitions": [
                    {
                        "target_state": MetaBuilderStates.REVIEW,
                        "description": "Revision applied, back to review",
                        "priority": 100,
                    },
                ],
            },
            # ----------------------------------------------------------
            # OUTPUT: Format and finalize (terminal state)
            # ----------------------------------------------------------
            MetaBuilderStates.OUTPUT: {
                "id": MetaBuilderStates.OUTPUT,
                "description": "Format and present the final artifact",
                "purpose": "Output the completed artifact as JSON",
                "extraction_instructions": "No extraction needed.",
                "response_instructions": (
                    "Present the final artifact JSON and inform the user "
                    "they can save it to a file."
                ),
                "transitions": [],
            },
        },
    }
