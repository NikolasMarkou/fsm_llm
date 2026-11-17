"""The pipeline's side of the event log: the executor and transition events (plan 21e6f00d).

Layer ``engine`` (``turn``). Three module-level functions the Pass-1 seam calls, each one line at
its call site so the anchored pipeline bodies stay as they are. With no turn scope bound (the log
is off) each returns at once and builds nothing. They import ``core`` and ``runtime`` only: the
pipeline cannot import ``engine/fsm/`` (the import contract), where the state blob lives.

Interface contract (callers: ``pass1.py`` and ``transitions.py``):
    - ``record_executor_start(state)`` and ``record_executor_result(state, response)`` append
      ``executor/start`` and ``executor/result``: ``kind`` (the registry name of the state's
      kind: ``conversation``, ``completion``, ``synthesize``) and ``state`` (its id); the result
      adds ``extracted_keys`` (the key NAMES of the extracted data, sorted, never a value) and
      ``extraction_failed``. Written around the one dispatch seam of Pass 1 only, so the
      re-extraction after a transition (``after_transition``) is not an executor event of its own.
    - ``record_transition(state, evaluation, target, *, executed)`` appends ``transition``:
      ``from_state``, ``to_state`` (``None`` when the turn stays), ``outcome`` (the evaluation's
      ``deterministic``, ``ambiguous`` or ``blocked``), ``executed`` and, only when the log can
      state it exactly, ``priority`` (an AMBIGUOUS tie: the tied priority; a DETERMINISTIC win:
      the priority of the one edge to the target, omitted when several edges share the target
      and the evaluation does not say which one passed) and ``candidates`` (the tied targets).
    - Failure mode: a failing append raises ``EventLogError`` (``runtime.emit``); the writer is
      poisoned, so the next checkpoint refuses the turn.
"""

from __future__ import annotations

from typing import Any

from fsm_llm.core.definitions import (
    DataExtractionResponse,
    State,
    TransitionEvaluation,
    TransitionEvaluationResult,
)
from fsm_llm.core.state_kinds import kind_of
from fsm_llm.runtime import EventType, current_scope, emit


def record_executor_start(state: State) -> None:
    """Append ``executor/start`` for the state about to extract (no-op with the log off)."""
    if current_scope() is not None:
        emit(EventType.EXECUTOR_START, kind=kind_of(state).name, state=state.id)


def record_executor_result(state: State, response: DataExtractionResponse) -> None:
    """Append ``executor/result``: key names of what was extracted, never values."""
    if current_scope() is not None:
        emit(
            EventType.EXECUTOR_RESULT,
            kind=kind_of(state).name,
            state=state.id,
            extracted_keys=sorted(str(key) for key in response.extracted_data),
            extraction_failed=bool(response.extraction_failed),
        )


def _priority(
    state: State, evaluation: TransitionEvaluation, target: str | None
) -> dict[str, Any]:
    """The ``priority`` and ``candidates`` fields the evaluation lets us state exactly."""
    if evaluation.result_type == TransitionEvaluationResult.AMBIGUOUS:
        options = evaluation.available_options
        return {
            "priority": options[0].priority,
            "candidates": sorted({option.target_state for option in options}),
        }
    if evaluation.result_type == TransitionEvaluationResult.DETERMINISTIC and target:
        edges = [t for t in state.transitions if t.target_state == target]
        if len(edges) == 1:
            return {"priority": edges[0].priority}
    return {}


def record_transition(
    state: State,
    evaluation: TransitionEvaluation,
    target: str | None,
    *,
    executed: bool,
) -> None:
    """Append ``transition`` for one evaluation of *state* (no-op with the log off)."""
    if current_scope() is None:
        return
    emit(
        EventType.TRANSITION,
        from_state=state.id,
        to_state=target,
        outcome=evaluation.result_type.value,
        executed=executed,
        **_priority(state, evaluation, target),
    )
