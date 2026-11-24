"""REWOO FSM (``build_rewoo_fsm``): the shipped file ``patterns/rewoo.json``, instantiated."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from fsm_llm.tools import ToolRegistry


def build_rewoo_fsm(
    registry: ToolRegistry, task_description: str = "", context_keys: Sequence[str] = ()
) -> dict[str, Any]:
    """
    Build a REWOO FSM definition from a tool registry.

    Two LLM calls: ``plan_all`` plans every tool call with #E1, #E2 refs, ``execute_plans`` runs
    them with no model call and ``solve`` answers from the evidence. ``plan_all`` and
    ``execute_plans`` are silent; ``plan_blueprint`` is an explicit ``list`` field whose prompt shows
    the task and the caller's ``context_keys``, never ``agent_trace``. Raises ``ValueError`` like
    :func:`loop_field_context_keys`. ``task_description`` is only the retrieval query of a semantic
    tool registry; the definition's name and description are static (D-002 and D-008 of plan
    a555dcfe). The FSM is the file ``fsm_llm/agents/patterns/rewoo.json`` filled by
    ``pattern_fsm`` (plan a555dcfe step 14): the same dict, in the same key order, as the builder
    this replaced.
    """
    # imported here: pattern_inputs imports the sibling `_loop` (see build_react_fsm)
    from fsm_llm.agents.pattern_files import load_envelope
    from fsm_llm.agents.pattern_inputs import pattern_fsm

    return pattern_fsm(
        load_envelope("rewoo"),
        tools=registry,
        task_description=task_description,
        context_keys=context_keys,
    )
