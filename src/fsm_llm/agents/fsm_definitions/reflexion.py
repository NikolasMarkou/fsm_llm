"""Reflexion FSM (``build_reflexion_fsm``)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from fsm_llm.agents.constants import ContextKeys, ReflexionStates
from fsm_llm.agents.prompts import (
    build_conclude_response_instructions,
    build_evaluate_field_instructions,
    build_reflect_field_instructions,
    build_think_extraction_instructions,
)
from fsm_llm.tools import ToolRegistry

from ._common import _by_id, _edge, _finalize_fsm, _is, _state, _when
from ._loop import (
    _act_state,
    _await_approval_state,
    _conclude_state,
    _field,
    _think_field_extractions,
    _think_transitions,
    conclude_on_evidence_logic,
)


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
    """
    persona = (
        "You are a reflective AI agent that solves tasks by using tools, "
        "then evaluating and critiquing your own work. "
        "After each tool action, you evaluate whether you have a good answer. "
        "If not, you reflect on what went wrong and try a different approach. "
        "Use your episodic memory to avoid repeating past mistakes."
    )

    think_instructions = build_think_extraction_instructions(
        registry, task_description=task_description
    )
    evaluate_fields = build_evaluate_field_instructions()
    reflect_fields = build_reflect_field_instructions()
    # REACT-01: reflect reads the verdict it critiques and the earlier
    # episodes, so each episode's reflection is its own.
    reflect_context = (
        ContextKeys.EVALUATION_FEEDBACK,
        ContextKeys.EVALUATION_SCORE,
        ContextKeys.EPISODIC_MEMORY,
    )

    states = _by_id(
        _state(
            ReflexionStates.THINK,
            "Reason about the task and select the next tool to use",
            "Analyze the task, episodic memory, and previous observations",
            _think_transitions(
                "Agent decided to terminate", include_approval=include_approval_state
            ),
            required=[
                ContextKeys.TOOL_NAME,
                ContextKeys.TOOL_INPUT,
                ContextKeys.SHOULD_TERMINATE,
            ],
            # D-009 of plan 06a5ec0a: typed per-field values only, no bulk
            # call; the episodic memory stays in the think prompts.
            fields=_think_field_extractions(
                think_instructions,
                context_keys=(ContextKeys.EPISODIC_MEMORY, *context_keys),
            ),
        ),
        _act_state(
            "Execute the selected tool and observe the result",
            "Run the tool and record the observation",
            ReflexionStates.EVALUATE,
            "Evaluate the result quality",
        ),
        _state(
            ReflexionStates.EVALUATE,
            "Assess whether gathered information is sufficient",
            "Evaluate answer quality and decide whether to reflect or conclude",
            [
                # DECISION plan-2026-09-24T091842-c1d5bfbc/D-008: a pass
                # concludes only with tool evidence or a forced stop. Do
                # NOT gate on evaluation_passed alone: a self-evaluated
                # memory answer then succeeds with zero tool calls. The
                # 900 fallback to reflect keeps evaluate from BLOCKING.
                # A forced stop alone also concludes here (P2-W3): do NOT
                # route it through reflect -> think, or max_iterations=1
                # cycles to the 3-turn ceiling (BudgetExhaustedError).
                _when(
                    ReflexionStates.CONCLUDE,
                    "Evaluation passed, produce final answer",
                    10,
                    "Passed on evidence, or forced stop",
                    {
                        "or": [
                            conclude_on_evidence_logic(ContextKeys.EVALUATION_PASSED),
                            _is(ContextKeys.MAX_ITERATIONS_REACHED, True),
                        ]
                    },
                ),
                _when(
                    ReflexionStates.REFLECT,
                    "Evaluation failed, reflect on approach",
                    300,
                    "Evaluation did not pass",
                    _is(ContextKeys.EVALUATION_PASSED, False),
                ),
                _edge(
                    ReflexionStates.REFLECT,
                    "Fallback: reflect if evaluation result unclear",
                    900,
                    empty=True,
                ),
            ],
            required=[ContextKeys.EVALUATION_SCORE, ContextKeys.EVALUATION_PASSED],
            # Typed per-field values only (D-009 of plan 06a5ec0a); an
            # intermediate state, so no Pass-2 prose (LOOP-09). With an
            # evaluation_fn the verdict is set on evaluate entry and these
            # extractions are skipped (skip-if-set).
            fields=[
                _field(
                    ContextKeys.EVALUATION_PASSED,
                    "bool",
                    evaluate_fields[ContextKeys.EVALUATION_PASSED],
                ),
                _field(
                    ContextKeys.EVALUATION_SCORE,
                    "float",
                    evaluate_fields[ContextKeys.EVALUATION_SCORE],
                    required=False,
                ),
                _field(
                    ContextKeys.EVALUATION_FEEDBACK,
                    "str",
                    evaluate_fields[ContextKeys.EVALUATION_FEEDBACK],
                    required=False,
                ),
            ],
        ),
        _state(
            ReflexionStates.REFLECT,
            "Self-critique and plan a revised approach",
            "Analyze what went wrong and generate lessons for next attempt",
            [
                _edge(
                    ReflexionStates.THINK, "Return to thinking with updated memory", 100
                )
            ],
            required=[ContextKeys.REFLECTION],
            fields=[
                _field(
                    ContextKeys.REFLECTION,
                    "str",
                    reflect_fields[ContextKeys.REFLECTION],
                    reflect_context,
                ),
                _field(
                    ContextKeys.LESSONS,
                    "str",
                    reflect_fields[ContextKeys.LESSONS],
                    reflect_context,
                    required=False,
                ),
            ],
        ),
        _conclude_state(
            build_conclude_response_instructions(refused_actions=include_approval_state)
        ),
    )
    if include_approval_state:
        states[ReflexionStates.AWAIT_APPROVAL] = _await_approval_state()

    return _finalize_fsm(
        "reflexion_agent",
        task_description,
        "Reflexion agent with self-evaluation",
        ReflexionStates.THINK,
        persona,
        states,
    )
