"""REWOO FSM (``build_rewoo_fsm``)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from fsm_llm.agents.constants import ContextKeys, REWOOStates
from fsm_llm.agents.prompts import (
    build_rewoo_plan_field_instructions,
    build_rewoo_solve_response_instructions,
)
from fsm_llm.tools import ToolRegistry

from ._common import _by_id, _edge, _finalize_fsm, _state
from ._loop import _field


def build_rewoo_fsm(
    registry: ToolRegistry, task_description: str = "", context_keys: Sequence[str] = ()
) -> dict[str, Any]:
    """
    Build a REWOO FSM definition from a tool registry.

    The FSM implements the REWOO pattern with exactly 2 LLM calls:
    - plan_all: single LLM call generates a complete plan with #E1, #E2 refs
    - execute_plans: handler executes all tool calls sequentially (no LLM)
    - solve: single LLM call synthesizes the final answer from all evidence

    plan_all and execute_plans are silent (empty response instructions): only
    solve speaks. ``plan_blueprint`` is an explicit ``list`` field whose
    prompt shows the task and the caller's ``context_keys``, never
    ``agent_trace`` (DECISION D-053 of plan 06a5ec0a, anchored in
    :func:`build_adapt_fsm`). Raises ``ValueError`` like
    :func:`loop_field_context_keys`.

    ``task_description`` is only the retrieval query of a semantic tool
    registry (the tool section of the prompts); the definition's name and
    description are static, so the task never enters definition text
    (D-002 and D-008 of plan a555dcfe).
    """
    persona = (
        "You are a methodical AI agent that solves tasks by planning all tool calls "
        "upfront, then executing them, and finally synthesizing an answer from the "
        "collected evidence. You think before you act, and you plan completely."
    )

    states = _by_id(
        _state(
            REWOOStates.PLAN_ALL,
            "Create a complete plan of all tool calls needed",
            "Generate a full plan with tool calls and variable references",
            [
                _edge(
                    REWOOStates.EXECUTE_PLANS,
                    "Plan is complete, proceed to execution",
                    100,
                )
            ],
            required=[ContextKeys.PLAN_BLUEPRINT],
            # No state-level bulk call (D-031 of plan 07ad3f8c, anchored in
            # build_orchestrator_fsm): `plan_blueprint` is the one key read.
            fields=[
                _field(
                    ContextKeys.PLAN_BLUEPRINT,
                    "list",
                    build_rewoo_plan_field_instructions(
                        registry, task_description=task_description
                    ),
                    context_keys,
                )
            ],
        ),
        _state(
            REWOOStates.EXECUTE_PLANS,
            "Execute all planned tool calls sequentially",
            "Run every tool call from the plan, substituting variable references",
            [
                _edge(
                    REWOOStates.SOLVE,
                    "All plans executed, proceed to synthesize answer",
                    100,
                )
            ],
        ),
        _state(
            REWOOStates.SOLVE,
            "Synthesize the final answer from all evidence",
            "Combine the task, plan, and all tool results into a final answer",
            response_instructions=build_rewoo_solve_response_instructions(),
        ),
    )

    return _finalize_fsm(
        "rewoo_agent",
        "REWOO agent with upfront planning",
        REWOOStates.PLAN_ALL,
        persona,
        states,
    )
