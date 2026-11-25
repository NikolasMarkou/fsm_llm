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
    - ``arecord_extraction`` (the blob and the line ONE lane job) and ``record_extraction_miss`` append ``extraction`` (``keys``, and
      ``delta_ref`` at level ``values``) and ``extraction/miss`` (the declared keys still ``None``
      after the Pass-1 commit): best effort, never an internal-prefixed or secret-looking key.
    - ``record_transition(state, evaluation, target, *, executed)`` appends ``transition``:
      ``from_state``, ``to_state`` (``None`` when the turn stays), ``outcome`` (the evaluation's
      ``deterministic``, ``ambiguous`` or ``blocked``), ``executed`` and, only when the log can
      state it exactly, ``priority`` (an AMBIGUOUS tie: the tied priority; a DETERMINISTIC win:
      the priority of the one edge to the target, omitted when several edges share the target
      and the evaluation does not say which one passed) and ``candidates`` (the tied targets).
    - Failure mode: a failing executor or transition append raises ``EventLogError``
      (``runtime.emit``); the writer is poisoned, so the next checkpoint refuses the turn. The
      extraction pair never raises (a WARNING).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from fsm_llm.core.definitions import (
    DataExtractionResponse,
    FSMDefinition,
    FSMInstance,
    State,
    TransitionEvaluation,
    TransitionEvaluationResult,
    handler_only_names,
)
from fsm_llm.core.state_kinds import kind_of
from fsm_llm.runtime import (
    EventType,
    RedactionKind,
    TurnScope,
    current_scope,
    emit,
    filter_context,
    visible_entries,
    warn_on_failure,
)


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


async def arecord_extraction(state: State, data: Mapping[str, Any]) -> None:
    """Append ``extraction`` for the visible entries of *data*, minus the kind's result key.

    The delta is filtered here, on the loop; its blob and the line are ONE lane job."""
    scope = current_scope()
    if scope is None:
        return
    with warn_on_failure(EventType.EXTRACTION):
        own = kind_of(state).result_key(state)
        visible = visible_entries({k: v for k, v in data.items() if k != own})
        if not visible:
            return
        body = filter_context(visible) if scope.log.stores_values else None
        await scope.log.lane_call(
            scope.root_id, _write_extraction, scope, state.id, sorted(visible), body
        )


def _write_extraction(
    scope: TurnScope, state_id: str, keys: list[str], body: dict[str, Any] | None
) -> None:
    fields: dict[str, Any] = {}
    if body is not None:
        fields["delta_ref"] = scope.log.put_json(body, kind=RedactionKind.DELTA)
    scope.log.append(
        scope.root_id,
        EventType.EXTRACTION,
        frame=scope.frame,
        state=state_id,
        keys=keys,
        **fields,
    )


def record_extraction_miss(
    state: State,
    instance: FSMInstance,
    fsm_resolver: Callable[[str], FSMDefinition],
) -> None:
    """Append ``extraction/miss``: declared keys (minted plus classification fields, minus the
    handler-only names, which include every result key) still ``None`` in the context."""
    if current_scope() is None:
        return
    with warn_on_failure(EventType.EXTRACTION_MISS):
        handler_only = handler_only_names(fsm_resolver(instance.fsm_id))
        declared = kind_of(state).produced_keys(state, handler_only) - handler_only
        data = instance.context.data
        missing = visible_entries({k: None for k in declared if data.get(k) is None})
        if missing:
            emit(EventType.EXTRACTION_MISS, state=state.id, keys=sorted(missing))


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
