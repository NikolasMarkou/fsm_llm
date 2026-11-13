"""Plan-and-Execute FSM (``build_plan_execute_fsm``)."""

from __future__ import annotations

from typing import Any

from fsm_llm.agents.constants import ContextKeys, PlanExecuteStates
from fsm_llm.agents.prompts import (
    build_execute_step_instructions,
    build_plan_steps_instructions,
    build_synthesize_response_instructions,
)
from fsm_llm.tools import ToolRegistry

from ._common import _by_id, _edge, _finalize_fsm, _is, _state, _when
from ._loop import _field, _tool_selection_field_extractions, loop_field_context_keys


def build_plan_execute_fsm(
    registry: ToolRegistry | None = None, task_description: str = ""
) -> dict[str, Any]:
    """
    Build a Plan-and-Execute FSM definition.

    Separates strategic planning from tactical execution:
    plan -> execute_step -> check_result -> synthesize (all done)
                                          -> replan (step failed) -> execute_step
                                          -> execute_step (next step)
    """
    persona = (
        "You are a strategic AI agent that solves tasks by first creating a plan, "
        "then executing each step methodically. "
        "If a step fails, you can revise the remaining plan. "
        "When all steps are complete, you synthesize results into a final answer."
    )

    # Typed per-field values only and no Pass-2 prose on the intermediate
    # states (D-009 of plan 06a5ec0a, PAT-01/02). `plan_steps` is a typed
    # list: an `any` config took a string, whose characters were then
    # counted as steps. check_result extracts nothing: the step checker
    # decides `step_failed` from the tool status on entry.
    step_context = (ContextKeys.CURRENT_STEP_DESCRIPTION, ContextKeys.STEP_RESULTS)
    replan_context = (ContextKeys.STEP_RESULTS, ContextKeys.PREVIOUS_PLAN_STEPS)
    step_instructions = build_execute_step_instructions(
        registry, task_description=task_description
    )
    has_tools = registry is not None and len(registry) > 0
    step_fields: list[dict[str, Any]] = [
        _field(
            ContextKeys.STEP_RESULT,
            "str",
            build_execute_step_instructions(
                registry, task_description=task_description, step_result=True
            ),
            step_context,
            # With tools the step result is the tool observation.
            required=not has_tools,
        )
    ]
    if has_tools:
        step_fields = [
            *_tool_selection_field_extractions(
                step_instructions, context_keys=loop_field_context_keys(step_context)
            ),
            *step_fields,
        ]

    states = _by_id(
        _state(
            PlanExecuteStates.PLAN,
            "Decompose the task into a sequence of steps",
            "Create an actionable plan to solve the task",
            [
                _when(
                    PlanExecuteStates.EXECUTE_STEP,
                    "Plan is ready, begin executing steps",
                    100,
                    "Plan steps have been generated",
                    {"has_context": ContextKeys.PLAN_STEPS},
                )
            ],
            required=[ContextKeys.PLAN_STEPS],
            fields=[
                _field(
                    ContextKeys.PLAN_STEPS,
                    "list",
                    build_plan_steps_instructions(
                        registry, task_description=task_description
                    ),
                )
            ],
        ),
        _state(
            PlanExecuteStates.EXECUTE_STEP,
            "Execute the current plan step",
            "Produce a result for the current step using tools or LLM",
            [
                _edge(
                    PlanExecuteStates.CHECK_RESULT,
                    "Step executed, check the result",
                    100,
                )
            ],
            fields=step_fields,
        ),
        _state(
            PlanExecuteStates.CHECK_RESULT,
            "Assess the step result and decide next action",
            "Determine if step succeeded and whether to continue, replan, or synthesize",
            [
                _when(
                    PlanExecuteStates.SYNTHESIZE,
                    "All steps complete, synthesize final answer",
                    10,
                    "All plan steps are complete",
                    _is(ContextKeys.ALL_STEPS_COMPLETE, True),
                ),
                _when(
                    PlanExecuteStates.REPLAN,
                    "Step failed, revise the plan",
                    150,
                    "The step did not succeed",
                    _is(ContextKeys.STEP_FAILED, True),
                ),
                _edge(
                    PlanExecuteStates.EXECUTE_STEP, "Proceed to the next plan step", 300
                ),
            ],
            extraction_instructions="",
        ),
        _state(
            PlanExecuteStates.REPLAN,
            "Revise the remaining plan after a step failure",
            "Incorporate lessons from the failure into a revised plan",
            [
                _edge(
                    PlanExecuteStates.EXECUTE_STEP,
                    "Resume execution with revised plan",
                    100,
                )
            ],
            fields=[
                _field(
                    ContextKeys.PLAN_STEPS,
                    "list",
                    build_plan_steps_instructions(
                        registry, task_description=task_description, replan=True
                    ),
                    replan_context,
                )
            ],
        ),
        _state(
            PlanExecuteStates.SYNTHESIZE,
            "Combine all step results into a final answer",
            "Produce a comprehensive answer from all step results",
            response_instructions=build_synthesize_response_instructions(),
        ),
    )

    return _finalize_fsm(
        "plan_execute_agent",
        task_description,
        "Plan-and-Execute agent",
        PlanExecuteStates.PLAN,
        persona,
        states,
    )
