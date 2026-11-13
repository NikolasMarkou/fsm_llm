"""Prompt Chain FSM (``build_prompt_chain_fsm``)."""

from __future__ import annotations

from typing import Any

from fsm_llm.agents.constants import ContextKeys, PromptChainStates, StopReason
from fsm_llm.agents.definitions import ChainStep
from fsm_llm.agents.prompts import (
    build_chain_output_response_instructions,
    build_chain_step_field_instructions,
)

from ._common import _edge, _finalize_fsm, _is, _state, _when
from ._loop import _field


def build_prompt_chain_fsm(
    chain: list[ChainStep], task_description: str = ""
) -> dict[str, Any]:
    """
    Build a Prompt Chain FSM definition from a list of ChainStep objects.

    Creates a linear pipeline of states: step_0 -> step_1 -> ... -> output.
    Each step extracts one typed ``chain_step_result`` (any, D-050) whose prompt
    shows the task and ``chain_results``; the ChainStep's instructions word
    that field, and its ``response_instructions`` stay the step's reply
    (user-owned). A step after a gated step (one with ``validation_fn``) also
    has an edge to ``output`` taken when that gate failed.
    """
    persona = (
        "You are a methodical AI assistant that processes tasks through a "
        "structured pipeline of steps. Execute each step carefully, building "
        "on the results of previous steps."
    )

    states: dict[str, Any] = {}

    for i, step in enumerate(chain):
        state_id = f"{PromptChainStates.STEP_PREFIX}{i}"
        is_last = i == len(chain) - 1
        next_state = (
            PromptChainStates.OUTPUT
            if is_last
            else f"{PromptChainStates.STEP_PREFIX}{i + 1}"
        )

        transitions: list[dict[str, Any]] = []
        if i > 0 and chain[i - 1].validation_fn is not None:
            # The gate of step i-1 runs on entry here (PromptChainAgent).
            transitions.append(
                _when(
                    PromptChainStates.OUTPUT,
                    f"Stop: the gate of {chain[i - 1].name} failed",
                    50,
                    "The previous step's gate failed",
                    _is(ContextKeys.FORCED_STOP_REASON, StopReason.GATE_FAILED),
                )
            )
        transitions.append(
            _edge(next_state, f"Proceed to {'output' if is_last else step.name}", 100)
        )

        states[state_id] = _state(
            state_id,
            f"Step {i + 1}: {step.name}",
            step.name,
            transitions,
            response_instructions=step.response_instructions,
            fields=[
                _field(
                    ContextKeys.CHAIN_STEP_RESULT,
                    "any",
                    build_chain_step_field_instructions(
                        i,
                        step.name,
                        step.response_instructions,
                        step.extraction_instructions,
                    ),
                    (ContextKeys.CHAIN_RESULTS,),
                )
            ],
        )

    # Output (terminal) state
    states[PromptChainStates.OUTPUT] = _state(
        PromptChainStates.OUTPUT,
        "Produce the final output from the chain",
        "Synthesize all step results into a final answer",
        response_instructions=build_chain_output_response_instructions(),
    )

    return _finalize_fsm(
        "prompt_chain_agent",
        task_description,
        "Prompt chain agent",
        f"{PromptChainStates.STEP_PREFIX}0",
        persona,
        states,
    )
