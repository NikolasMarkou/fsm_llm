from __future__ import annotations

"""
Pre-built FSM definitions for agent patterns.
"""

from typing import Any

from .constants import ContextKeys
from .definitions import ChainStep
from .tools import ToolRegistry

# ---------------------------------------------------------------------------
# Orchestrator-Workers FSM
# ---------------------------------------------------------------------------


def build_orchestrator_fsm(
    task_description: str = "",
) -> dict[str, Any]:
    """
    Build an Orchestrator-Workers FSM definition.

    The FSM implements task decomposition and delegation:
    orchestrate -> delegate -> collect -> synthesize (all collected)
                                       -> orchestrate (more work needed)
    """
    from .prompts import (
        build_collect_extraction_instructions,
        build_collect_response_instructions,
        build_delegate_response_instructions,
        build_orchestrate_extraction_instructions,
        build_orchestrate_response_instructions,
        build_orchestrator_synthesize_extraction_instructions,
        build_orchestrator_synthesize_response_instructions,
    )

    persona = (
        "You are an orchestrator AI agent that solves tasks by decomposing them "
        "into subtasks and delegating to workers. You analyze results, determine "
        "if more work is needed, and synthesize a final answer from all results."
    )

    states: dict[str, Any] = {
        "orchestrate": {
            "id": "orchestrate",
            "description": "Decompose the task into subtasks for delegation",
            "purpose": "Analyze the task and create a delegation plan",
            "extraction_instructions": build_orchestrate_extraction_instructions(),
            "response_instructions": build_orchestrate_response_instructions(),
            "transitions": [
                {
                    "target_state": "delegate",
                    "description": "Subtasks are ready for delegation",
                    "priority": 100,
                    "conditions": [
                        {
                            "description": "Subtasks have been generated",
                            "logic": {"has_context": "subtasks"},
                        }
                    ],
                },
                {
                    "target_state": "synthesize",
                    "description": "Iteration limit reached without subtask decomposition",
                    "priority": 900,
                    "conditions": [
                        {
                            "description": "Budget exhausted, skip to synthesis",
                            "logic": {"==": [{"var": "should_terminate"}, True]},
                        }
                    ],
                },
            ],
        },
        "delegate": {
            "id": "delegate",
            "description": "Delegate subtasks to workers and collect results",
            "purpose": "Execute worker_factory for each subtask",
            "response_instructions": build_delegate_response_instructions(),
            "transitions": [
                {
                    "target_state": "collect",
                    "description": "Workers have finished, review results",
                    "priority": 100,
                }
            ],
        },
        "collect": {
            "id": "collect",
            "description": "Review worker results and decide if more work is needed",
            "purpose": "Assess completeness of gathered results",
            "extraction_instructions": build_collect_extraction_instructions(),
            "response_instructions": build_collect_response_instructions(),
            "transitions": [
                {
                    "target_state": "synthesize",
                    "description": "All results collected, produce final answer",
                    "priority": 10,
                    "conditions": [
                        {
                            "description": "All needed results are collected",
                            "logic": {"==": [{"var": "all_collected"}, True]},
                        }
                    ],
                },
                {
                    "target_state": "orchestrate",
                    "description": "More work needed, decompose further",
                    "priority": 300,
                    "conditions": [
                        {
                            "description": "More subtasks are needed",
                            "logic": {"==": [{"var": "all_collected"}, False]},
                        }
                    ],
                },
            ],
        },
        "synthesize": {
            "id": "synthesize",
            "description": "Synthesize all worker results into a final answer",
            "purpose": "Produce a comprehensive answer from all worker results",
            "extraction_instructions": build_orchestrator_synthesize_extraction_instructions(),
            "response_instructions": build_orchestrator_synthesize_response_instructions(),
            "transitions": [],
        },
    }

    return {
        "name": "orchestrator_agent",
        "description": task_description or "Orchestrator-Workers agent",
        "initial_state": "orchestrate",
        "persona": persona,
        "states": states,
    }


# ---------------------------------------------------------------------------
# ADaPT FSM
# ---------------------------------------------------------------------------


def build_adapt_fsm(
    registry: ToolRegistry | None = None,
    task_description: str = "",
    max_depth: int = 3,
) -> dict[str, Any]:
    """
    Build an ADaPT (Adaptive Decomposition and Planning for Tasks) FSM definition.

    The FSM implements try-first, decompose-on-failure:
    attempt -> assess -> combine (success)
                      -> decompose (failure) -> combine (depth limit)
                                             -> [triggers recursive run()]
    """
    from .prompts import (
        build_assess_extraction_instructions,
        build_assess_response_instructions,
        build_attempt_extraction_instructions,
        build_attempt_response_instructions,
        build_combine_extraction_instructions,
        build_combine_response_instructions,
        build_decompose_extraction_instructions,
        build_decompose_response_instructions,
    )

    persona = (
        "You are an adaptive AI agent that attempts tasks directly first. "
        "If the direct attempt is insufficient, you decompose the task into "
        "simpler subtasks and solve them recursively. "
        "Always try the direct approach before decomposing."
    )

    states: dict[str, Any] = {
        "attempt": {
            "id": "attempt",
            "description": "Attempt to solve the task directly",
            "purpose": "Give a direct attempt at solving the task",
            "required_context_keys": ["attempt_result"],
            "extraction_instructions": build_attempt_extraction_instructions(registry),
            "response_instructions": build_attempt_response_instructions(),
            "transitions": [
                {
                    "target_state": "combine",
                    "description": "Iteration limit reached, produce best-effort answer",
                    "priority": 1,
                    "conditions": [
                        {
                            "description": "Should terminate due to iteration limit",
                            "logic": {"==": [{"var": "should_terminate"}, True]},
                        }
                    ],
                },
                {
                    "target_state": "assess",
                    "description": "Evaluate the attempt quality",
                    "priority": 100,
                },
            ],
        },
        "assess": {
            "id": "assess",
            "description": "Assess whether the attempt succeeded",
            "purpose": "Determine if the attempt is satisfactory or needs decomposition",
            "required_context_keys": ["attempt_succeeded"],
            "extraction_instructions": build_assess_extraction_instructions(),
            "response_instructions": build_assess_response_instructions(),
            "transitions": [
                {
                    "target_state": "combine",
                    "description": "Iteration limit reached, produce best-effort answer",
                    "priority": 1,
                    "conditions": [
                        {
                            "description": "Should terminate due to iteration limit",
                            "logic": {"==": [{"var": "should_terminate"}, True]},
                        }
                    ],
                },
                {
                    "target_state": "combine",
                    "description": "Attempt succeeded, produce final answer",
                    "priority": 10,
                    "conditions": [
                        {
                            "description": "The attempt was successful",
                            "logic": {"==": [{"var": "attempt_succeeded"}, True]},
                        }
                    ],
                },
                {
                    "target_state": "decompose",
                    "description": "Attempt failed, decompose into subtasks",
                    "priority": 150,
                    "conditions": [
                        {
                            "description": "Attempt failed and depth allows decomposition",
                            "logic": {
                                "and": [
                                    {"==": [{"var": "attempt_succeeded"}, False]},
                                    {
                                        "<": [
                                            {"var": "current_depth"},
                                            max_depth,
                                        ]
                                    },
                                ]
                            },
                        }
                    ],
                },
                {
                    "target_state": "combine",
                    "description": "Attempt failed but depth limit reached, use best effort",
                    "priority": 200,
                    "conditions": [
                        {
                            "description": "Depth limit reached, force best effort",
                            "logic": {"==": [{"var": "attempt_succeeded"}, False]},
                        }
                    ],
                },
            ],
        },
        "decompose": {
            "id": "decompose",
            "description": "Decompose the task into simpler subtasks",
            "purpose": "Break the task down for recursive solving",
            "required_context_keys": ["subtasks"],
            "extraction_instructions": build_decompose_extraction_instructions(),
            "response_instructions": build_decompose_response_instructions(),
            "transitions": [
                {
                    "target_state": "combine",
                    "description": "Iteration limit reached, produce best-effort answer",
                    "priority": 1,
                    "conditions": [
                        {
                            "description": "Should terminate due to iteration limit",
                            "logic": {"==": [{"var": "should_terminate"}, True]},
                        }
                    ],
                },
                {
                    "target_state": "combine",
                    "description": "Subtasks defined, combine after recursive solving",
                    "priority": 100,
                    "conditions": [
                        {
                            "description": "Subtasks have been generated",
                            "logic": {"has_context": "subtasks"},
                        }
                    ],
                },
            ],
        },
        "combine": {
            "id": "combine",
            "description": "Combine all results into the final answer",
            "purpose": "Synthesize attempt results and subtask results",
            "required_context_keys": ["final_answer"],
            "extraction_instructions": build_combine_extraction_instructions(),
            "response_instructions": build_combine_response_instructions(),
            "transitions": [],
        },
    }

    return {
        "name": "adapt_agent",
        "description": task_description or "ADaPT agent with recursive decomposition",
        "initial_state": "attempt",
        "persona": persona,
        "states": states,
    }


# ---------------------------------------------------------------------------
# Reflexion FSM
# ---------------------------------------------------------------------------


def build_reflexion_fsm(
    registry: ToolRegistry,
    task_description: str = "",
) -> dict[str, Any]:
    """
    Build a Reflexion FSM definition from a tool registry.

    Extends the ReAct loop with evaluation and self-reflection:
    think -> act -> evaluate -> reflect (if failed) -> think (loop)
                              -> conclude (if passed)
    """
    from .prompts import (
        build_act_response_instructions,
        build_conclude_extraction_instructions,
        build_conclude_response_instructions,
        build_evaluate_extraction_instructions,
        build_evaluate_response_instructions,
        build_reflect_extraction_instructions,
        build_reflect_response_instructions,
        build_think_extraction_instructions,
        build_think_response_instructions,
    )

    tool_names = registry.tool_names

    persona = (
        "You are a reflective AI agent that solves tasks by using tools, "
        "then evaluating and critiquing your own work. "
        "After each tool action, you evaluate whether you have a good answer. "
        "If not, you reflect on what went wrong and try a different approach. "
        "Use your episodic memory to avoid repeating past mistakes."
    )

    states: dict[str, Any] = {
        "think": {
            "id": "think",
            "description": "Reason about the task and select the next tool to use",
            "purpose": "Analyze the task, episodic memory, and previous observations",
            "required_context_keys": ["tool_name", "tool_input", "should_terminate"],
            "extraction_instructions": build_think_extraction_instructions(registry),
            "response_instructions": build_think_response_instructions(),
            "transitions": [
                {
                    "target_state": "conclude",
                    "description": "Agent decided to terminate",
                    "priority": 10,
                    "conditions": [
                        {
                            "description": "Agent decided to terminate",
                            "logic": {"==": [{"var": "should_terminate"}, True]},
                        }
                    ],
                },
                {
                    "target_state": "act",
                    "description": "Execute the selected tool",
                    "priority": 300,
                    "conditions": [
                        {
                            "description": "A tool has been selected",
                            "logic": {
                                "in": [
                                    {"var": "tool_name"},
                                    [*tool_names, ContextKeys.NO_TOOL],
                                ]
                            },
                        }
                    ],
                },
            ],
        },
        "act": {
            "id": "act",
            "description": "Execute the selected tool and observe the result",
            "purpose": "Run the tool and record the observation",
            "response_instructions": build_act_response_instructions(),
            "transitions": [
                {
                    "target_state": "evaluate",
                    "description": "Evaluate the result quality",
                    "priority": 100,
                }
            ],
        },
        "evaluate": {
            "id": "evaluate",
            "description": "Assess whether gathered information is sufficient",
            "purpose": "Evaluate answer quality and decide whether to reflect or conclude",
            "required_context_keys": ["evaluation_score"],
            "extraction_instructions": build_evaluate_extraction_instructions(),
            "response_instructions": build_evaluate_response_instructions(),
            "transitions": [
                {
                    "target_state": "conclude",
                    "description": "Evaluation passed, produce final answer",
                    "priority": 10,
                    "conditions": [
                        {
                            "description": "Evaluation passed",
                            "logic": {"==": [{"var": "evaluation_passed"}, True]},
                        }
                    ],
                },
                {
                    "target_state": "reflect",
                    "description": "Evaluation failed, reflect on approach",
                    "priority": 300,
                    "conditions": [
                        {
                            "description": "Evaluation did not pass",
                            "logic": {"==": [{"var": "evaluation_passed"}, False]},
                        }
                    ],
                },
            ],
        },
        "reflect": {
            "id": "reflect",
            "description": "Self-critique and plan a revised approach",
            "purpose": "Analyze what went wrong and generate lessons for next attempt",
            "required_context_keys": ["reflection"],
            "extraction_instructions": build_reflect_extraction_instructions(),
            "response_instructions": build_reflect_response_instructions(),
            "transitions": [
                {
                    "target_state": "think",
                    "description": "Return to thinking with updated memory",
                    "priority": 100,
                }
            ],
        },
        "conclude": {
            "id": "conclude",
            "description": "Formulate and present the final answer",
            "purpose": "Synthesize all observations into a complete answer",
            "required_context_keys": ["final_answer"],
            "extraction_instructions": build_conclude_extraction_instructions(),
            "response_instructions": build_conclude_response_instructions(),
            "transitions": [],
        },
    }

    return {
        "name": "reflexion_agent",
        "description": task_description or "Reflexion agent with self-evaluation",
        "initial_state": "think",
        "persona": persona,
        "states": states,
    }


# ---------------------------------------------------------------------------
# Plan-and-Execute FSM
# ---------------------------------------------------------------------------


def build_plan_execute_fsm(
    registry: ToolRegistry | None = None,
    task_description: str = "",
) -> dict[str, Any]:
    """
    Build a Plan-and-Execute FSM definition.

    Separates strategic planning from tactical execution:
    plan -> execute_step -> check_result -> synthesize (all done)
                                          -> replan (step failed) -> execute_step
                                          -> execute_step (next step)
    """
    from .prompts import (
        build_check_result_extraction_instructions,
        build_check_result_response_instructions,
        build_execute_step_extraction_instructions,
        build_plan_extraction_instructions,
        build_replan_extraction_instructions,
        build_synthesize_extraction_instructions,
        build_synthesize_response_instructions,
    )

    persona = (
        "You are a strategic AI agent that solves tasks by first creating a plan, "
        "then executing each step methodically. "
        "If a step fails, you can revise the remaining plan. "
        "When all steps are complete, you synthesize results into a final answer."
    )

    states: dict[str, Any] = {
        "plan": {
            "id": "plan",
            "description": "Decompose the task into a sequence of steps",
            "purpose": "Create an actionable plan to solve the task",
            "extraction_instructions": build_plan_extraction_instructions(registry),
            "response_instructions": (
                "Present your plan as a numbered list of steps. "
                "Explain why this decomposition makes sense."
            ),
            "transitions": [
                {
                    "target_state": "execute_step",
                    "description": "Plan is ready, begin executing steps",
                    "priority": 100,
                    "conditions": [
                        {
                            "description": "Plan steps have been generated",
                            "logic": {"has_context": "plan_steps"},
                        }
                    ],
                }
            ],
        },
        "execute_step": {
            "id": "execute_step",
            "description": "Execute the current plan step",
            "purpose": "Produce a result for the current step using tools or LLM",
            "extraction_instructions": build_execute_step_extraction_instructions(
                registry
            ),
            "response_instructions": (
                "Describe what you did for this step and what result was produced."
            ),
            "transitions": [
                {
                    "target_state": "check_result",
                    "description": "Step executed, check the result",
                    "priority": 100,
                }
            ],
        },
        "check_result": {
            "id": "check_result",
            "description": "Assess the step result and decide next action",
            "purpose": "Determine if step succeeded and whether to continue, replan, or synthesize",
            "extraction_instructions": build_check_result_extraction_instructions(),
            "response_instructions": build_check_result_response_instructions(),
            "transitions": [
                {
                    "target_state": "synthesize",
                    "description": "All steps complete, synthesize final answer",
                    "priority": 10,
                    "conditions": [
                        {
                            "description": "All plan steps are complete",
                            "logic": {"==": [{"var": "all_steps_complete"}, True]},
                        }
                    ],
                },
                {
                    "target_state": "replan",
                    "description": "Step failed, revise the plan",
                    "priority": 150,
                    "conditions": [
                        {
                            "description": "The step did not succeed",
                            "logic": {"==": [{"var": "step_failed"}, True]},
                        }
                    ],
                },
                {
                    "target_state": "execute_step",
                    "description": "Proceed to the next plan step",
                    "priority": 300,
                },
            ],
        },
        "replan": {
            "id": "replan",
            "description": "Revise the remaining plan after a step failure",
            "purpose": "Incorporate lessons from the failure into a revised plan",
            "extraction_instructions": build_replan_extraction_instructions(registry),
            "response_instructions": (
                "Explain what went wrong and present the revised plan."
            ),
            "transitions": [
                {
                    "target_state": "execute_step",
                    "description": "Resume execution with revised plan",
                    "priority": 100,
                }
            ],
        },
        "synthesize": {
            "id": "synthesize",
            "description": "Combine all step results into a final answer",
            "purpose": "Produce a comprehensive answer from all step results",
            "extraction_instructions": build_synthesize_extraction_instructions(),
            "response_instructions": build_synthesize_response_instructions(),
            "transitions": [],
        },
    }

    return {
        "name": "plan_execute_agent",
        "description": task_description or "Plan-and-Execute agent",
        "initial_state": "plan",
        "persona": persona,
        "states": states,
    }


def build_react_fsm(
    registry: ToolRegistry,
    task_description: str = "",
    include_approval_state: bool = False,
    use_classification: bool = False,
    output_schema: type | None = None,
) -> dict[str, Any]:
    """
    Build a ReAct FSM definition from a tool registry.

    The FSM implements the Observe-Think-Act loop:
    - think: LLM reasons about the task and selects a tool
    - act: tool is executed (via handler), observation is recorded
    - conclude: LLM produces final answer when should_terminate is true

    Optionally includes an await_approval state for HITL patterns.

    When *use_classification* is True, tool selection in the think state
    uses a ``classification_extractions`` config (backed by the core
    ``Classifier``) instead of relying solely on extraction instructions.
    This can improve tool selection accuracy for large tool registries.
    """
    from .prompts import (
        build_act_response_instructions,
        build_approval_extraction_instructions,
        build_conclude_extraction_instructions,
        build_conclude_response_instructions,
        build_think_extraction_instructions,
        build_think_response_instructions,
    )

    tool_names = registry.tool_names

    # Build persona with tool awareness
    persona = (
        "You are a methodical AI agent that solves tasks by using tools step by step. "
        "Think carefully before each action. Review previous observations before deciding. "
        "Terminate when you have gathered enough information to answer the task."
    )

    # Think state transitions
    # NOTE: Lower priority number = higher confidence in TransitionEvaluator.
    # Terminal transitions (conclude) get lowest priority numbers.
    think_transitions: list[dict[str, Any]] = [
        {
            "target_state": "conclude",
            "description": "Task can be answered with current observations",
            "priority": 10,
            "conditions": [
                {
                    "description": "Agent decided to terminate",
                    "logic": {"==": [{"var": "should_terminate"}, True]},
                }
            ],
        },
    ]

    if include_approval_state:
        think_transitions.append(
            {
                "target_state": "await_approval",
                "description": "Action requires human approval before execution",
                "priority": 150,
                "conditions": [
                    {
                        "description": "Approval is required for this action",
                        "logic": {"==": [{"var": "approval_required"}, True]},
                    }
                ],
            }
        )

    think_transitions.append(
        {
            "target_state": "act",
            "description": "Execute the selected tool",
            "priority": 300,
            "conditions": [
                {
                    "description": "A tool has been selected",
                    "logic": {
                        "in": [{"var": "tool_name"}, [*tool_names, ContextKeys.NO_TOOL]]
                    },
                }
            ],
        }
    )

    think_state: dict[str, Any] = {
        "id": "think",
        "description": "Reason about the task and select the next tool to use",
        "purpose": "Analyze the task and previous observations to decide the next action",
        "required_context_keys": ["tool_name", "tool_input", "should_terminate"],
        "extraction_instructions": build_think_extraction_instructions(registry),
        "response_instructions": build_think_response_instructions(),
        "transitions": think_transitions,
    }

    if use_classification:
        schema = registry.to_classification_schema()
        think_state["classification_extractions"] = [
            {
                "field_name": "tool_name",
                "intents": schema["intents"],
                "fallback_intent": schema["fallback_intent"],
                "confidence_threshold": schema.get("confidence_threshold", 0.4),
            }
        ]

    states: dict[str, Any] = {
        "think": think_state,
        "act": {
            "id": "act",
            "description": "Execute the selected tool and observe the result",
            "purpose": "Run the tool and record the observation",
            "response_instructions": build_act_response_instructions(),
            "transitions": [
                {
                    "target_state": "think",
                    "description": "Return to thinking with new observation",
                    "priority": 100,
                }
            ],
        },
        "conclude": {
            "id": "conclude",
            "description": "Formulate and present the final answer",
            "purpose": "Synthesize all observations into a complete answer",
            "required_context_keys": (
                ["final_answer"]
                + (
                    list(output_schema.model_fields.keys())
                    if output_schema and hasattr(output_schema, "model_fields")
                    else []
                )
            ),
            "extraction_instructions": build_conclude_extraction_instructions(
                output_schema
            ),
            "response_instructions": build_conclude_response_instructions(),
            "transitions": [],
        },
    }

    if include_approval_state:
        states["await_approval"] = {
            "id": "await_approval",
            "description": "Waiting for human approval before executing action",
            "purpose": "Present the planned action and wait for user approval",
            "extraction_instructions": build_approval_extraction_instructions(),
            "response_instructions": (
                "Explain what action you want to take and why, "
                "then ask the user for approval."
            ),
            "transitions": [
                {
                    "target_state": "act",
                    "description": "Approval granted, proceed with action",
                    "priority": 10,
                    "conditions": [
                        {
                            "description": "User approved the action",
                            "logic": {"==": [{"var": "approval_granted"}, True]},
                        }
                    ],
                },
                {
                    "target_state": "think",
                    "description": "Approval denied, reconsider approach",
                    "priority": 300,
                    "conditions": [
                        {
                            "description": "User denied the action",
                            "logic": {"==": [{"var": "approval_granted"}, False]},
                        }
                    ],
                },
            ],
        }

    return {
        "name": "react_agent",
        "description": task_description or "ReAct agent with tool use",
        "initial_state": "think",
        "persona": persona,
        "states": states,
    }


# ---------------------------------------------------------------------------
# Prompt Chain FSM
# ---------------------------------------------------------------------------


def build_prompt_chain_fsm(
    chain: list[ChainStep],
    task_description: str = "",
) -> dict[str, Any]:
    """
    Build a Prompt Chain FSM definition from a list of ChainStep objects.

    Creates a linear pipeline of states: step_0 -> step_1 -> ... -> output.
    Each step uses the ChainStep's extraction and response instructions.
    """
    from .prompts import (
        build_chain_output_extraction_instructions,
        build_chain_output_response_instructions,
    )

    persona = (
        "You are a methodical AI assistant that processes tasks through a "
        "structured pipeline of steps. Execute each step carefully, building "
        "on the results of previous steps."
    )

    states: dict[str, Any] = {}

    for i, step in enumerate(chain):
        state_id = f"step_{i}"
        is_last = i == len(chain) - 1
        next_state = "output" if is_last else f"step_{i + 1}"

        states[state_id] = {
            "id": state_id,
            "description": f"Step {i + 1}: {step.name}",
            "purpose": step.name,
            "extraction_instructions": step.extraction_instructions,
            "response_instructions": step.response_instructions,
            "transitions": [
                {
                    "target_state": next_state,
                    "description": f"Proceed to {'output' if is_last else step.name}",
                    "priority": 100,
                }
            ],
        }

    # Output (terminal) state
    states["output"] = {
        "id": "output",
        "description": "Produce the final output from the chain",
        "purpose": "Synthesize all step results into a final answer",
        "extraction_instructions": build_chain_output_extraction_instructions(),
        "response_instructions": build_chain_output_response_instructions(),
        "transitions": [],
    }

    return {
        "name": "prompt_chain_agent",
        "description": task_description or "Prompt chain agent",
        "initial_state": "step_0",
        "persona": persona,
        "states": states,
    }


# ---------------------------------------------------------------------------
# Self-Consistency FSM
# ---------------------------------------------------------------------------


def build_self_consistency_fsm(
    task_description: str = "",
) -> dict[str, Any]:
    """
    Build a simple single-state FSM for self-consistency sampling.

    Each invocation generates one answer. The SelfConsistencyAgent runs
    this FSM multiple times with different temperatures and aggregates.
    """
    from .prompts import (
        build_generate_extraction_instructions,
        build_generate_response_instructions,
    )

    persona = (
        "You are a precise AI assistant. Answer the given task directly and "
        "concisely. Provide your best answer and your confidence level."
    )

    states: dict[str, Any] = {
        "generate": {
            "id": "generate",
            "description": "Generate an answer to the task",
            "purpose": "Produce a direct, complete answer to the task",
            "extraction_instructions": build_generate_extraction_instructions(),
            "response_instructions": build_generate_response_instructions(),
            "transitions": [],
        },
    }

    return {
        "name": "self_consistency_sample",
        "description": task_description or "Self-consistency single sample",
        "initial_state": "generate",
        "persona": persona,
        "states": states,
    }


# ---------------------------------------------------------------------------
# Debate FSM
# ---------------------------------------------------------------------------


def build_debate_fsm(
    task_description: str = "",
    proposer_persona: str = "",
    critic_persona: str = "",
    judge_persona: str = "",
    max_rounds: int = 3,
) -> dict[str, Any]:
    """
    Build a Debate FSM definition.

    Implements a multi-round debate loop:
    propose -> critique -> counter -> judge -> propose (loop)
                                             -> conclude (consensus or max rounds)
    """
    from .prompts import (
        build_counter_extraction_instructions,
        build_counter_response_instructions,
        build_critique_extraction_instructions,
        build_critique_response_instructions,
        build_debate_conclude_extraction_instructions,
        build_debate_conclude_response_instructions,
        build_judge_extraction_instructions,
        build_judge_response_instructions,
        build_propose_extraction_instructions,
        build_propose_response_instructions,
    )

    # Use proposer persona as the top-level FSM persona since it starts
    persona = proposer_persona or (
        "You are a thoughtful AI that explores questions through structured debate. "
        "Multiple perspectives are considered to arrive at the best answer."
    )

    states: dict[str, Any] = {
        "propose": {
            "id": "propose",
            "description": "Generate or refine a proposition for the task",
            "purpose": "Present a well-reasoned argument or answer",
            "extraction_instructions": build_propose_extraction_instructions(
                proposer_persona
            ),
            "response_instructions": build_propose_response_instructions(),
            "transitions": [
                {
                    "target_state": "critique",
                    "description": "Proposition ready for critique",
                    "priority": 100,
                }
            ],
        },
        "critique": {
            "id": "critique",
            "description": "Critically analyze the current proposition",
            "purpose": "Identify weaknesses, gaps, and counterpoints",
            "extraction_instructions": build_critique_extraction_instructions(
                critic_persona
            ),
            "response_instructions": build_critique_response_instructions(),
            "transitions": [
                {
                    "target_state": "counter",
                    "description": "Critique complete, allow counter-argument",
                    "priority": 100,
                }
            ],
        },
        "counter": {
            "id": "counter",
            "description": "Address the critique with counter-arguments",
            "purpose": "Strengthen the proposition by addressing criticisms",
            "extraction_instructions": build_counter_extraction_instructions(
                proposer_persona
            ),
            "response_instructions": build_counter_response_instructions(),
            "transitions": [
                {
                    "target_state": "judge",
                    "description": "Counter-argument ready for judgment",
                    "priority": 100,
                }
            ],
        },
        "judge": {
            "id": "judge",
            "description": "Evaluate the debate exchange and decide next action",
            "purpose": "Determine whether consensus has been reached",
            "extraction_instructions": build_judge_extraction_instructions(
                judge_persona, max_rounds
            ),
            "response_instructions": build_judge_response_instructions(),
            "transitions": [
                {
                    "target_state": "conclude",
                    "description": "Consensus reached or max rounds hit",
                    "priority": 10,
                    "conditions": [
                        {
                            "description": "Consensus has been reached",
                            "logic": {"==": [{"var": "consensus_reached"}, True]},
                        }
                    ],
                },
                {
                    "target_state": "propose",
                    "description": "Another round of debate needed",
                    "priority": 300,
                },
            ],
        },
        "conclude": {
            "id": "conclude",
            "description": "Produce the final answer from the debate",
            "purpose": "Synthesize the debate into a definitive answer",
            "extraction_instructions": build_debate_conclude_extraction_instructions(),
            "response_instructions": build_debate_conclude_response_instructions(),
            "transitions": [],
        },
    }

    return {
        "name": "debate_agent",
        "description": task_description or "Debate agent",
        "initial_state": "propose",
        "persona": persona,
        "states": states,
    }


# ---------------------------------------------------------------------------
# REWOO FSM
# ---------------------------------------------------------------------------


def build_rewoo_fsm(
    registry: ToolRegistry,
    task_description: str = "",
) -> dict[str, Any]:
    """
    Build a REWOO FSM definition from a tool registry.

    The FSM implements the REWOO pattern with exactly 2 LLM calls:
    - plan_all: single LLM call generates a complete plan with #E1, #E2 refs
    - execute_plans: handler executes all tool calls sequentially (no LLM)
    - solve: single LLM call synthesizes the final answer from all evidence
    """
    from .prompts import (
        build_rewoo_execute_response_instructions,
        build_rewoo_plan_extraction_instructions,
        build_rewoo_plan_response_instructions,
        build_rewoo_solve_extraction_instructions,
        build_rewoo_solve_response_instructions,
    )

    persona = (
        "You are a methodical AI agent that solves tasks by planning all tool calls "
        "upfront, then executing them, and finally synthesizing an answer from the "
        "collected evidence. You think before you act, and you plan completely."
    )

    states: dict[str, Any] = {
        "plan_all": {
            "id": "plan_all",
            "description": "Create a complete plan of all tool calls needed",
            "purpose": "Generate a full plan with tool calls and variable references",
            "extraction_instructions": build_rewoo_plan_extraction_instructions(
                registry
            ),
            "response_instructions": build_rewoo_plan_response_instructions(),
            "transitions": [
                {
                    "target_state": "execute_plans",
                    "description": "Plan is complete, proceed to execution",
                    "priority": 100,
                }
            ],
        },
        "execute_plans": {
            "id": "execute_plans",
            "description": "Execute all planned tool calls sequentially",
            "purpose": "Run every tool call from the plan, substituting variable references",
            "response_instructions": build_rewoo_execute_response_instructions(),
            "transitions": [
                {
                    "target_state": "solve",
                    "description": "All plans executed, proceed to synthesize answer",
                    "priority": 100,
                }
            ],
        },
        "solve": {
            "id": "solve",
            "description": "Synthesize the final answer from all evidence",
            "purpose": "Combine the task, plan, and all tool results into a final answer",
            "extraction_instructions": build_rewoo_solve_extraction_instructions(),
            "response_instructions": build_rewoo_solve_response_instructions(),
            "transitions": [],
        },
    }

    return {
        "name": "rewoo_agent",
        "description": task_description or "REWOO agent with upfront planning",
        "initial_state": "plan_all",
        "persona": persona,
        "states": states,
    }


# ---------------------------------------------------------------------------
# Evaluator-Optimizer FSM
# ---------------------------------------------------------------------------


def build_evalopt_fsm(
    task_description: str = "",
) -> dict[str, Any]:
    """
    Build an Evaluator-Optimizer FSM definition.

    The FSM implements the generate-evaluate-refine loop:
    - generate: LLM generates output
    - evaluate: handler runs external evaluation function
    - refine: LLM refines based on feedback
    - output: terminal state with final answer
    """
    from .prompts import (
        build_evalopt_evaluate_response_instructions,
        build_evalopt_generate_extraction_instructions,
        build_evalopt_generate_response_instructions,
        build_evalopt_output_extraction_instructions,
        build_evalopt_output_response_instructions,
        build_evalopt_refine_extraction_instructions,
        build_evalopt_refine_response_instructions,
    )

    persona = (
        "You are an AI agent that produces high-quality outputs through iterative "
        "refinement. You generate output, receive evaluation feedback, and improve "
        "your output until it meets the required quality standards."
    )

    states: dict[str, Any] = {
        "generate": {
            "id": "generate",
            "description": "Generate an initial output for the task",
            "purpose": "Produce the best possible first attempt at the task",
            "extraction_instructions": build_evalopt_generate_extraction_instructions(),
            "response_instructions": build_evalopt_generate_response_instructions(),
            "required_context_keys": ["generated_output"],
            "transitions": [
                {
                    "target_state": "evaluate",
                    "description": "Output generated, proceed to evaluation",
                    "priority": 100,
                    "conditions": [
                        {
                            "description": "Output has been generated",
                            "logic": {"has_context": "generated_output"},
                        }
                    ],
                }
            ],
        },
        "evaluate": {
            "id": "evaluate",
            "description": "Evaluate the generated output",
            "purpose": "Run the external evaluation function on the current output",
            "response_instructions": build_evalopt_evaluate_response_instructions(),
            "transitions": [
                {
                    "target_state": "output",
                    "description": "Evaluation passed, produce final output",
                    "priority": 10,
                    "conditions": [
                        {
                            "description": "Output passed evaluation",
                            "logic": {"==": [{"var": "evaluation_passed"}, True]},
                        }
                    ],
                },
                {
                    "target_state": "refine",
                    "description": "Evaluation failed, refine the output",
                    "priority": 300,
                    "conditions": [
                        {
                            "description": "Output did not pass evaluation",
                            "logic": {"==": [{"var": "evaluation_passed"}, False]},
                        }
                    ],
                },
            ],
        },
        "refine": {
            "id": "refine",
            "description": "Refine the output based on evaluation feedback",
            "purpose": "Improve the output by addressing specific feedback points",
            "extraction_instructions": build_evalopt_refine_extraction_instructions(),
            "response_instructions": build_evalopt_refine_response_instructions(),
            "required_context_keys": ["generated_output"],
            "transitions": [
                {
                    "target_state": "evaluate",
                    "description": "Refined output ready for re-evaluation",
                    "priority": 100,
                }
            ],
        },
        "output": {
            "id": "output",
            "description": "Present the final evaluated output",
            "purpose": "Extract and present the final answer",
            "extraction_instructions": build_evalopt_output_extraction_instructions(),
            "response_instructions": build_evalopt_output_response_instructions(),
            "transitions": [],
        },
    }

    return {
        "name": "evalopt_agent",
        "description": task_description or "Evaluator-Optimizer agent",
        "initial_state": "generate",
        "persona": persona,
        "states": states,
    }


# ---------------------------------------------------------------------------
# Maker-Checker FSM
# ---------------------------------------------------------------------------


def build_maker_checker_fsm(
    maker_instructions: str,
    checker_instructions: str,
    task_description: str = "",
) -> dict[str, Any]:
    """
    Build a Maker-Checker FSM definition.

    The FSM implements the make-check-revise loop:
    - make: maker persona generates a draft
    - check: checker persona evaluates the draft
    - revise: maker persona revises based on feedback
    - output: terminal state with final answer
    """
    from .prompts import (
        build_checker_extraction_instructions,
        build_checker_response_instructions,
        build_maker_checker_output_extraction_instructions,
        build_maker_checker_output_response_instructions,
        build_maker_extraction_instructions,
        build_maker_response_instructions,
        build_revise_extraction_instructions,
        build_revise_response_instructions,
    )

    persona = (
        "You are an AI agent that produces high-quality outputs through a "
        "maker-checker process. You alternate between creating content and "
        "critically evaluating it to ensure the highest quality."
    )

    states: dict[str, Any] = {
        "make": {
            "id": "make",
            "description": "Maker generates a draft output",
            "purpose": "Produce a high-quality draft following the maker instructions",
            "extraction_instructions": build_maker_extraction_instructions(
                maker_instructions
            ),
            "response_instructions": build_maker_response_instructions(),
            "transitions": [
                {
                    "target_state": "check",
                    "description": "Draft complete, proceed to checker review",
                    "priority": 100,
                }
            ],
        },
        "check": {
            "id": "check",
            "description": "Checker evaluates the draft",
            "purpose": "Critically evaluate the draft against quality criteria",
            "required_context_keys": [
                "checker_passed",
                "checker_feedback",
                "quality_score",
            ],
            "extraction_instructions": build_checker_extraction_instructions(
                checker_instructions
            ),
            "response_instructions": build_checker_response_instructions(),
            "transitions": [
                {
                    "target_state": "output",
                    "description": "Draft passed review, produce final output",
                    "priority": 10,
                    "conditions": [
                        {
                            "description": "Checker approved the draft",
                            "logic": {"==": [{"var": "checker_passed"}, True]},
                        }
                    ],
                },
                {
                    "target_state": "revise",
                    "description": "Draft needs revision based on feedback",
                    "priority": 300,
                    "conditions": [
                        {
                            "description": "Checker found issues with the draft",
                            "logic": {"==": [{"var": "checker_passed"}, False]},
                        }
                    ],
                },
            ],
        },
        "revise": {
            "id": "revise",
            "description": "Maker revises the draft based on checker feedback",
            "purpose": "Address all checker feedback and produce an improved draft",
            "extraction_instructions": build_revise_extraction_instructions(
                maker_instructions
            ),
            "response_instructions": build_revise_response_instructions(),
            "transitions": [
                {
                    "target_state": "check",
                    "description": "Revised draft ready for re-evaluation",
                    "priority": 100,
                }
            ],
        },
        "output": {
            "id": "output",
            "description": "Present the final reviewed output",
            "purpose": "Extract and present the final answer",
            "extraction_instructions": build_maker_checker_output_extraction_instructions(),
            "response_instructions": build_maker_checker_output_response_instructions(),
            "transitions": [],
        },
    }

    return {
        "name": "maker_checker_agent",
        "description": task_description or "Maker-Checker agent",
        "initial_state": "make",
        "persona": persona,
        "states": states,
    }
