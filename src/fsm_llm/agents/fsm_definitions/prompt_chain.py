"""Prompt Chain FSM: the ``agents.prompt_chain`` generator (plan a555dcfe step 11, D-019).

The FSM of a chain depends on the chain (one state per step, a gate edge after a gated step), so
``patterns/prompt_chain.json`` declares ``generator: agents.prompt_chain`` and this module holds
the function (``prompt_chain_fsm``, registered in the ``generators`` namespace by
``agents.handlers.library``). ``build_prompt_chain_fsm`` is the builder wrapper over the file.

Interface contract: ``prompt_chain_fsm(options) -> dict``: ``options["steps"]`` is a non-empty list
of ``{name, extraction_instructions, response_instructions, gated}`` (the options schema of the
file has checked them); the result is the FSM dict, ``step_0 -> ... -> output``, in the key order
the old builder wrote. It reads nothing else (no task, no variant, no callable); a bad ``steps``
raises (the seam reports the exception type only). ``chain_options(chain) -> dict`` turns a list of
``ChainStep`` into the options of the file (``steps`` and the ``max_iterations`` budget).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from fsm_llm.agents.constants import (
    ContextKeys,
    Defaults,
    PromptChainStates,
    StopReason,
)
from fsm_llm.agents.definitions import ChainStep
from fsm_llm.agents.prompts import (
    build_chain_output_response_instructions,
    build_chain_step_field_instructions,
)

from ._common import _edge, _finalize_fsm, _is, _state, _when
from ._loop import _field


def chain_options(chain: Sequence[ChainStep]) -> dict[str, Any]:
    """The options of ``patterns/prompt_chain.json`` for *chain* (a step is gated by its function)."""
    return {
        "steps": [
            {
                "name": step.name,
                "extraction_instructions": step.extraction_instructions,
                "response_instructions": step.response_instructions,
                "gated": step.validation_fn is not None,
            }
            for step in chain
        ],
        "max_iterations": len(chain) * Defaults.FSM_BUDGET_MULTIPLIER,
    }


def prompt_chain_fsm(options: Mapping[str, Any]) -> dict[str, Any]:
    """The generator: a linear pipeline ``step_0 -> step_1 -> ... -> output`` (see the module text).

    Each step extracts one typed ``chain_step_result`` (any, D-050) whose prompt shows the task
    and ``chain_results``; the step's instructions word that field, and its
    ``response_instructions`` stay the step's reply (user-owned). A step after a gated step also
    has an edge to ``output`` taken when that gate failed.
    """
    steps = options["steps"]
    if not steps:
        raise ValueError("a chain needs a step")
    persona = (
        "You are a methodical AI assistant that processes tasks through a "
        "structured pipeline of steps. Execute each step carefully, building "
        "on the results of previous steps."
    )

    states: dict[str, Any] = {}

    for i, step in enumerate(steps):
        state_id = f"{PromptChainStates.STEP_PREFIX}{i}"
        is_last = i == len(steps) - 1
        next_state = (
            PromptChainStates.OUTPUT
            if is_last
            else f"{PromptChainStates.STEP_PREFIX}{i + 1}"
        )

        transitions: list[dict[str, Any]] = []
        if i > 0 and steps[i - 1]["gated"]:
            # The gate of step i-1 runs on entry here (agents.chain_gate_checker).
            transitions.append(
                _when(
                    PromptChainStates.OUTPUT,
                    f"Stop: the gate of {steps[i - 1]['name']} failed",
                    50,
                    "The previous step's gate failed",
                    _is(ContextKeys.FORCED_STOP_REASON, StopReason.GATE_FAILED.value),
                )
            )
        transitions.append(
            _edge(
                next_state, f"Proceed to {'output' if is_last else step['name']}", 100
            )
        )

        states[state_id] = _state(
            state_id,
            f"Step {i + 1}: {step['name']}",
            step["name"],
            transitions,
            response_instructions=step["response_instructions"],
            fields=[
                _field(
                    ContextKeys.CHAIN_STEP_RESULT,
                    "any",
                    build_chain_step_field_instructions(
                        i,
                        step["name"],
                        step["response_instructions"],
                        step["extraction_instructions"],
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
        "Prompt chain agent",
        f"{PromptChainStates.STEP_PREFIX}0",
        persona,
        states,
    )


def build_prompt_chain_fsm(chain: Sequence[ChainStep]) -> dict[str, Any]:
    """Build a Prompt Chain FSM definition from a list of ``ChainStep`` objects.

    The FSM is the pattern file ``fsm_llm/agents/patterns/prompt_chain.json`` run through its
    generator (plan a555dcfe step 11): the same dict, in the same key order, as the builder this
    replaced. The task never enters definition text (D-002).
    """
    # imported here: pattern_inputs imports the sibling `_loop` (see build_react_fsm)
    from fsm_llm.agents.pattern_files import load_envelope
    from fsm_llm.agents.pattern_inputs import pattern_fsm

    return pattern_fsm(
        load_envelope("prompt_chain"), tools=None, options=chain_options(chain)
    )
