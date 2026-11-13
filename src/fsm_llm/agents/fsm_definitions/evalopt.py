"""Evaluator-Optimizer FSM (``build_evalopt_fsm``)."""

from __future__ import annotations

from typing import Any

from fsm_llm.agents.constants import ContextKeys, EvalOptStates
from fsm_llm.agents.prompts import (
    build_evalopt_field_instructions,
    build_evalopt_output_response_instructions,
)

from ._common import _by_id, _edge, _finalize_fsm, _is, _state, _when
from ._loop import _field


def build_evalopt_fsm(task_description: str = "") -> dict[str, Any]:
    """
    Build an Evaluator-Optimizer FSM definition.

    The FSM implements the generate-evaluate-refine loop:
    - generate: LLM generates output
    - evaluate: handler runs external evaluation function
    - refine: LLM refines based on feedback
    - output: terminal state with final answer

    generate and refine each extract one typed ``generated_output`` (any) and
    are silent (empty response instructions); ``evaluation_passed`` is written
    only by the evaluation handler, never extracted.
    """
    fields = build_evalopt_field_instructions()

    persona = (
        "You are an AI agent that produces high-quality outputs through iterative "
        "refinement. You generate output, receive evaluation feedback, and improve "
        "your output until it meets the required quality standards."
    )

    states = _by_id(
        _state(
            EvalOptStates.GENERATE,
            "Generate an initial output for the task",
            "Produce the best possible first attempt at the task",
            [
                _when(
                    EvalOptStates.EVALUATE,
                    "Output generated, proceed to evaluation",
                    100,
                    "Output has been generated",
                    {"has_context": ContextKeys.GENERATED_OUTPUT},
                ),
                # DECISION plan-2026-09-24T045559-3e4eb3e5/D-002
                # Unconditional lowest-priority fallback. Do NOT remove it or gate it:
                # a missing generated_output would BLOCK `generate` (no PRE_TRANSITION
                # limiter runs on a BLOCKED turn). `evaluate` judges the empty output,
                # then `refine` retries and the refinement cap / limiter end the loop.
                _edge(
                    EvalOptStates.EVALUATE,
                    "Fallback: evaluate even if no output was extracted",
                    900,
                ),
            ],
            fields=[
                _field(
                    ContextKeys.GENERATED_OUTPUT, "any", fields[EvalOptStates.GENERATE]
                )
            ],
        ),
        _state(
            EvalOptStates.EVALUATE,
            "Evaluate the generated output",
            "Run the external evaluation function on the current output",
            [
                _when(
                    EvalOptStates.OUTPUT,
                    "Evaluation passed, produce final output",
                    10,
                    "Output passed evaluation",
                    _is(ContextKeys.EVALUATION_PASSED, True),
                ),
                _when(
                    EvalOptStates.REFINE,
                    "Evaluation failed, refine the output",
                    300,
                    "Output did not pass evaluation",
                    _is(ContextKeys.EVALUATION_PASSED, False),
                ),
                _edge(
                    EvalOptStates.OUTPUT,
                    "Fallback: produce output if evaluation stalls",
                    900,
                    empty=True,
                ),
            ],
        ),
        _state(
            EvalOptStates.REFINE,
            "Refine the output based on evaluation feedback",
            "Improve the output by addressing specific feedback points",
            [
                _edge(
                    EvalOptStates.EVALUATE,
                    "Refined output ready for re-evaluation",
                    100,
                )
            ],
            fields=[
                _field(
                    ContextKeys.GENERATED_OUTPUT,
                    "any",
                    fields[EvalOptStates.REFINE],
                    (ContextKeys.PREVIOUS_OUTPUT, ContextKeys.REFINEMENT_FEEDBACK),
                )
            ],
        ),
        _state(
            EvalOptStates.OUTPUT,
            "Present the final evaluated output",
            "Extract and present the final answer",
            response_instructions=build_evalopt_output_response_instructions(),
        ),
    )

    return _finalize_fsm(
        "evalopt_agent",
        task_description,
        "Evaluator-Optimizer agent",
        EvalOptStates.GENERATE,
        persona,
        states,
    )
