"""Reflexion FSM (``build_reflexion_fsm``): the shipped file ``patterns/reflexion.json``, instantiated."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from fsm_llm.tools import ToolRegistry


def build_reflexion_fsm(
    registry: ToolRegistry,
    task_description: str = "",
    include_approval_state: bool = False,
    context_keys: Sequence[str] = (),
) -> dict[str, Any]:
    """
    Build a Reflexion FSM definition from a tool registry.

    Extends the ReAct loop with evaluation and self-reflection:
    think -> act -> evaluate -> reflect (if failed) -> think (loop)
                              -> conclude (if passed)

    With *include_approval_state*, a gated call goes
    think -> await_approval -> act, the same state and priorities as
    :func:`build_react_fsm`.

    ``task_description`` is only the retrieval query of a semantic tool
    registry (the tool section of the prompts); the definition's name and
    description are static, so the task never enters definition text
    (D-002 and D-008 of plan a555dcfe). The FSM is the pattern file
    ``fsm_llm/agents/patterns/reflexion.json`` with the ``approval`` variant
    selected by the flag and the tool text and the caller keys filled by
    ``pattern_inputs.pattern_fsm`` (plan a555dcfe step 10): the same dict, in
    the same key order, as the builder this replaced.
    """
    # imported here: pattern_inputs imports the sibling `_loop` (see build_react_fsm)
    from fsm_llm.agents.pattern_files import load_envelope
    from fsm_llm.agents.pattern_inputs import pattern_fsm

    return pattern_fsm(
        load_envelope("reflexion"),
        tools=registry,
        task_description=task_description,
        approval=bool(include_approval_state),
        context_keys=context_keys,
    )
