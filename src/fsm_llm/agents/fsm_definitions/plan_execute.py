"""Plan-and-Execute FSM (``build_plan_execute_fsm``): the shipped file ``patterns/plan_execute.json``, instantiated."""

from __future__ import annotations

from typing import Any

from fsm_llm.tools import ToolRegistry


def build_plan_execute_fsm(
    registry: ToolRegistry | None = None, task_description: str = ""
) -> dict[str, Any]:
    """
    Build a Plan-and-Execute FSM definition.

    plan -> execute_step -> check_result -> synthesize (all done)
                                          -> replan (step failed) -> execute_step
                                          -> execute_step (next step)

    ``task_description`` is only the retrieval query of a semantic tool registry (the tool section
    of the prompts); the definition's name and description are static (D-002 and D-008 of plan
    a555dcfe). With no tool (``None`` or an empty registry) the ``without_tools`` variant drops the
    tool selection fields. The FSM is the file ``fsm_llm/agents/patterns/plan_execute.json`` filled
    by ``pattern_fsm`` (plan a555dcfe step 15): the same dict, in the same key order, as the builder
    this replaced.
    """
    # imported here: pattern_inputs imports the sibling `_loop` (see build_react_fsm)
    from fsm_llm.agents.pattern_files import load_envelope
    from fsm_llm.agents.pattern_inputs import pattern_fsm

    return pattern_fsm(
        load_envelope("plan_execute"),
        tools=registry,
        task_description=task_description,
    )
