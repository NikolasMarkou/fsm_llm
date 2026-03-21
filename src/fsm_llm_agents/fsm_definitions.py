from __future__ import annotations

"""
Pre-built FSM definitions for agent patterns.
"""

from typing import Any

from .tools import ToolRegistry


def build_react_fsm(
    registry: ToolRegistry,
    task_description: str = "",
    include_approval_state: bool = False,
) -> dict[str, Any]:
    """
    Build a ReAct FSM definition from a tool registry.

    The FSM implements the Observe-Think-Act loop:
    - think: LLM reasons about the task and selects a tool
    - act: tool is executed (via handler), observation is recorded
    - conclude: LLM produces final answer when should_terminate is true

    Optionally includes an await_approval state for HITL patterns.
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
    think_transitions: list[dict[str, Any]] = [
        {
            "target_state": "conclude",
            "description": "Task can be answered with current observations",
            "priority": 200,
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
            "priority": 100,
            "conditions": [
                {
                    "description": "A tool has been selected",
                    "logic": {"in": [{"var": "tool_name"}, [*tool_names, "none"]]},
                }
            ],
        }
    )

    states: dict[str, Any] = {
        "think": {
            "id": "think",
            "description": "Reason about the task and select the next tool to use",
            "purpose": "Analyze the task and previous observations to decide the next action",
            "extraction_instructions": build_think_extraction_instructions(registry),
            "response_instructions": build_think_response_instructions(),
            "transitions": think_transitions,
        },
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
            "extraction_instructions": build_conclude_extraction_instructions(),
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
                    "priority": 200,
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
                    "priority": 100,
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
