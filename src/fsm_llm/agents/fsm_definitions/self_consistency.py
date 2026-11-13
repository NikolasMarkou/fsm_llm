"""Self-Consistency FSM (``build_self_consistency_fsm``)."""

from __future__ import annotations

from typing import Any

from fsm_llm.agents.constants import SelfConsistencyStates
from fsm_llm.agents.prompts import build_generate_response_instructions

from ._common import _by_id, _finalize_fsm, _state


def build_self_consistency_fsm(task_description: str = "") -> dict[str, Any]:
    """
    Build a simple single-state FSM for self-consistency sampling.

    Each invocation generates one answer. The SelfConsistencyAgent runs
    this FSM multiple times with different temperatures and aggregates.
    ``generate`` is a terminal initial state, so core never extracts there:
    it has no extraction instructions, and the sample is the reply, which ends
    with an ``Answer:`` line for the vote.
    """
    persona = (
        "You are a precise AI assistant. Answer the given task directly and concisely."
    )

    states = _by_id(
        _state(
            SelfConsistencyStates.GENERATE,
            "Generate an answer to the task",
            "Produce a direct, complete answer to the task",
            response_instructions=build_generate_response_instructions(),
            extraction_instructions="",
        )
    )

    return _finalize_fsm(
        "self_consistency_sample",
        task_description,
        "Self-consistency single sample",
        SelfConsistencyStates.GENERATE,
        persona,
        states,
    )
