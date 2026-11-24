"""Orchestrator-Workers FSM (``build_orchestrator_fsm``)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from fsm_llm.agents.constants import ContextKeys, OrchestratorStates
from fsm_llm.agents.prompts import (
    build_orchestrator_field_instructions,
    build_orchestrator_synthesize_response_instructions,
)

from ._common import _by_id, _edge, _finalize_fsm, _is, _state, _when
from ._loop import _field


def build_orchestrator_fsm(context_keys: Sequence[str] = ()) -> dict[str, Any]:
    """
    Build an Orchestrator-Workers FSM definition.

    The FSM implements task decomposition and delegation:
    orchestrate -> delegate -> collect -> synthesize (all collected)
                                       -> orchestrate (more work needed)

    ``subtasks`` and ``all_collected`` are explicit typed fields whose prompts
    show the task, ``worker_results`` and the caller's ``context_keys``
    (never ``agent_trace`` or ``skipped_subtasks``). Raises ``ValueError``
    like :func:`loop_field_context_keys` for a disallowed context key.
    """
    persona = (
        "You are an orchestrator AI agent that solves tasks by decomposing them "
        "into subtasks and delegating to workers. You analyze results, determine "
        "if more work is needed, and synthesize a final answer from all results."
    )
    fields = build_orchestrator_field_instructions()
    # DECISION plan-2026-09-29T103145-06a5ec0a/D-053: explicit narrowed
    # configs replace core's auto-minted ones (whole context, agent_trace
    # included). `subtasks` stays `any`: the delegator also takes a lone
    # string as one subtask. Do NOT add `skipped_subtasks` here (D-049).
    # DECISION plan-2026-09-30T062855-07ad3f8c/D-031: `orchestrate` and
    # `collect` make no state-level bulk call. Their typed fields cover every
    # key the run reads (`subtasks`, `all_collected`); the bulk call only
    # added `delegation_plan` and `reasoning`, which nothing read. Do NOT add
    # bulk instructions back as a second chance for a null field: a null
    # takes the priority-900 edge to `synthesize`.
    # DECISION plan-2026-11-06T093029-0068b055/D-013: `orchestrate`,
    # `delegate` and `collect` are silent (no `response_instructions`); only
    # `synthesize` speaks. Do NOT re-add narration to an intermediate state: a
    # silent state makes no model call, and the evidence the judges read
    # (`worker_results`) is in `<current_context>`, not in the replies. The
    # old narration cost 3 calls a round and was written before `all_collected`
    # was judged. Mocked evidence only; live quality is F-AR-05-LIVE.
    judged = (ContextKeys.WORKER_RESULTS, *context_keys)

    states = _by_id(
        _state(
            OrchestratorStates.ORCHESTRATE,
            "Decompose the task into subtasks for delegation",
            "Analyze the task and create a delegation plan",
            [
                _when(
                    OrchestratorStates.DELEGATE,
                    "Subtasks are ready for delegation",
                    100,
                    "Subtasks have been generated",
                    {"has_context": ContextKeys.SUBTASKS},
                ),
                _edge(
                    OrchestratorStates.SYNTHESIZE,
                    "Fallback: skip to synthesis if decomposition stalls",
                    900,
                    empty=True,
                ),
            ],
            required=[ContextKeys.SUBTASKS],
            fields=[
                _field(
                    ContextKeys.SUBTASKS, "any", fields[ContextKeys.SUBTASKS], judged
                )
            ],
        ),
        _state(
            OrchestratorStates.DELEGATE,
            "Delegate subtasks to workers and collect results",
            "Execute worker_factory for each subtask",
            [
                _edge(
                    OrchestratorStates.COLLECT,
                    "Workers have finished, review results",
                    100,
                )
            ],
        ),
        _state(
            OrchestratorStates.COLLECT,
            "Review worker results and decide if more work is needed",
            "Assess completeness of gathered results",
            [
                _when(
                    OrchestratorStates.SYNTHESIZE,
                    "All results collected, produce final answer",
                    10,
                    "All needed results are collected",
                    _is(ContextKeys.ALL_COLLECTED, True),
                ),
                _when(
                    OrchestratorStates.ORCHESTRATE,
                    "More work needed, decompose further",
                    300,
                    "More subtasks are needed",
                    _is(ContextKeys.ALL_COLLECTED, False),
                ),
                _edge(
                    OrchestratorStates.SYNTHESIZE,
                    "Fallback: synthesize with available results if decision stalls",
                    900,
                    empty=True,
                ),
            ],
            fields=[
                _field(
                    ContextKeys.ALL_COLLECTED,
                    "bool",
                    fields[ContextKeys.ALL_COLLECTED],
                    judged,
                )
            ],
        ),
        _state(
            OrchestratorStates.SYNTHESIZE,
            "Synthesize all worker results into a final answer",
            "Produce a comprehensive answer from all worker results",
            response_instructions=build_orchestrator_synthesize_response_instructions(),
        ),
    )

    return _finalize_fsm(
        "orchestrator_agent",
        "Orchestrator-Workers agent",
        OrchestratorStates.ORCHESTRATE,
        persona,
        states,
    )
