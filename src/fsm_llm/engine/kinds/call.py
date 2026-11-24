"""``CallExecutor``: the call kind, Pass 1 resolves a target and writes a request, no model call.

Layer ``engine`` (plan baa37f40, step 11, D-022, roadmap E3.2). A state with ``State.call`` runs
one target (an executable id) and merges its declared outputs. Pass 1 of such a state only
DECIDES: it picks the target (the fixed ``target``, or the value of ``choose.key`` when it is in
the closed list, else ``fallback``), resolves it through the pipeline's ``TargetResolver``, reads
the declared ``inputs`` and either

- refuses, with the FINAL record ``{status: failed, reason, target, template_identity,
  instance_identity}`` (so the same turn's transition evaluation takes the failure edge), or
- writes the request, the ``running`` record ``{status, target, template_identity,
  instance_identity, mode, request}``, as handler-only turn data.

Both are the one extracted key ``{config.result_key: record}``. The executor makes no model call,
holds no stack and imports neither ``engine.api``, ``engine.turn`` nor ``engine.fsm``: a record
written in Pass 1 rolls back with the turn like every other write. What runs the request (the
push, the pop, the merge of ``returns``) is ``engine/api/calls.py`` (``settle``), which takes the
target, the inputs and the budgets from the parent's own definition and context, never from the
record: ``request`` is evidence and the durable intent, not an instruction (D-022, D-023).

Delegated mode (plan baa37f40 phase K7, D-051, roadmap E3.2 second half): when the frame resolver
answers ``None`` and the resolver is also a ``RunLauncher``, a code-backed ``pattern``
target (``resolve_run``) is DELEGATED. Pass 1 then reserves a minted handle (``claim``) and writes the
``running`` record with ``mode: delegated``, the ``handle`` and a wall-clock ``deadline``; settle
(K7.4) starts the run after the commit. On every later turn the executor POLLS the launcher instead of
skipping: still running before the deadline writes nothing; finished writes ONE extraction (the final
record and the declared ``returns``); past the deadline it cancels and writes ``aborted/budget_spent``;
a handle the launcher never saw for this conversation is ``failed/run_lost``. The executor still makes
no model call, starts no thread and waits for nothing.

Interface contract:

- ``CallExecutor.extract`` / ``aextract(pipeline, instance, user_message, conversation_id,
  state)``: the one Pass-1 body (``extract`` is ``run_sync`` over ``aextract``). Skip-if-set: a
  non-None value at the result key (any status) returns no data and asks no resolver. A
  pipeline with no resolver raises ``MissingExecutorError``; a state with no ``call`` config
  raises ``FSMError``.
- ``write_delegated_request(result_key, *, state_id, target, delegated, handle, inputs, max_steps,
  max_seconds)``: the delegated twin of ``write_call_request``: the same ``request`` body, ``mode:
  delegated``, the handle it is given and ``deadline = time.time() + delegated_seconds(max_seconds)``.
  ``final_record(state_id, status, reason, target, template_identity, instance_identity)`` is the one
  builder of a final record (the executor and ``settle``); ``is_delegated_running(record, state_id)``
  is the one test of "poll, do not skip".
- ``write_call_request(result_key, *, state_id, target, resolved, inputs, max_steps,
  max_seconds)`` and ``refuse(result_key, reason, *, state_id, target=None, resolved=None)``: the
  shared body (the synthesize executor, plan baa37f40 step 20.2, reuses both, with
  ``record_response`` and ``commit_arrival``). Each returns a ``DataExtractionResponse`` with
  confidence 1.0 and exactly the result key. ``reason`` is a ``core.codes.Reason`` or its wire
  string (``ValueError`` for another). ``request.inputs`` holds the values it is given, each
  through ``runtime.to_jsonable`` and cut to a marker past ``_REQUEST_VALUE_MAX_CHARS``.
  ``record_response(result_key, record)`` wraps any record the same way;
  ``commit_arrival(pipeline, instance, committed, state_id)`` is the post-transition commit of
  a decision made ON ARRIVAL.
"""

from __future__ import annotations

import json
import math
import time
import uuid
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, TypeGuard

from fsm_llm.aio import run_sync
from fsm_llm.context import commit_with_key_rollback
from fsm_llm.core.codes import Reason, Status
from fsm_llm.core.definitions import (
    CallStateConfig,
    DataExtractionResponse,
    FSMError,
    FSMInstance,
    MissingExecutorError,
    State,
)
from fsm_llm.handlers import HandlerTiming
from fsm_llm.logging import logger
from fsm_llm.runtime import to_jsonable

from .base import StateExecutor
from .services import _PipelineServices
from .targets import (
    MAX_DELEGATED_RETURN_CHARS,
    TARGET_MODE_DELEGATED,
    TARGET_MODE_FRAME,
    DelegatedTarget,
    ResolvedTarget,
    RunPoll,
    delegated_seconds,
    is_run_launcher,
)

if TYPE_CHECKING:  # the result envelope loads on request, never with `import fsm_llm`
    from fsm_llm.core.results import RunResult

#: A declared input whose JSON text is longer than this is recorded as a one-line marker (the
#: value itself still reaches the child): the record stays small and is never a second copy of
#: a large input in every prompt.
_REQUEST_VALUE_MAX_CHARS = 2000


def _safe_and_size(value: Any) -> tuple[Any, int]:
    """A JSON-safe copy of *value* (``to_jsonable``: iterative, depth-capped) and the length of
    its compact JSON text."""
    safe = to_jsonable(value)
    return safe, len(json.dumps(safe, ensure_ascii=False, separators=(",", ":")))


def _recorded(value: Any) -> Any:
    """A JSON-safe copy of an input value, or a marker naming its type and size when large."""
    safe, size = _safe_and_size(value)
    if size <= _REQUEST_VALUE_MAX_CHARS:
        return safe
    return f"<{type(value).__name__}, {size} characters of JSON, not recorded>"


def bounded_value(value: Any) -> Any:
    """A JSON-safe copy of a value a finished delegated run returns, capped (D-051).

    Contract (public since D-057: the run host caps a result with it at STORE time, the merge
    caps it again): walked iteratively with a depth cap (never a recursive dump: a small Windows
    stack). A text over ``MAX_DELEGATED_RETURN_CHARS`` is cut there and ends with a marker; any
    other value over it is a one-line marker naming its type and size. Never raises for a value
    ``to_jsonable`` accepts (every value).
    """
    safe = to_jsonable(value)
    if isinstance(safe, str):
        if len(safe) <= MAX_DELEGATED_RETURN_CHARS:
            return safe
        return f"{safe[:MAX_DELEGATED_RETURN_CHARS]} [cut at {MAX_DELEGATED_RETURN_CHARS} characters]"
    size = len(json.dumps(safe, ensure_ascii=False, separators=(",", ":")))
    if size <= MAX_DELEGATED_RETURN_CHARS:
        return safe
    return f"<{type(value).__name__}, {size} characters of JSON, not returned>"


# DECISION plan-2026-11-23T122249-baa37f40/D-025
# A call record belongs to the state that wrote it (`state`, the id of the state whose `call`
# config decided it), and the executor and settle share THIS one rule. Do NOT skip on any
# non-None value of the result key (two call states share the default `call_result`: the second
# would see the first's final `ok`, skip, and take its success edge with no child pushed), do NOT
# read the stamp in one reader only (settle applying a record of another state pushes the wrong
# target), and do NOT clear the key on arrival instead (an initial call state and an agent-managed
# FSM have no arrival). A record with no stamp (a handler, a seed, an older snapshot) keeps the
# skip-if-set of D-022; a revisit of the SAME state keeps its own stale final record (an author
# clears it with `clear_keys_on_entry`).
def record_belongs(record: Any, state_id: str) -> bool:
    """Whether a value at a call state's result key is a record of ``state_id``.

    Contract: ``True`` for a dict whose ``state`` is ``state_id`` or absent; ``False`` for a
    dict stamped with another id (or a stamp that is not text) and for any non-dict. Never
    raises. The executor skips only a value that belongs, and ``settle`` applies only a
    ``running`` record that belongs to the parent's current state.
    """
    if not isinstance(record, dict):
        return False
    stamp = record.get("state")
    return stamp is None or stamp == state_id


def _identities(
    resolved: ResolvedTarget | DelegatedTarget | None,
) -> dict[str, str | None]:
    return {
        "template_identity": None if resolved is None else resolved.template_identity,
        "instance_identity": None if resolved is None else resolved.instance_identity,
    }


def record_response(
    result_key: str, record: Mapping[str, Any]
) -> DataExtractionResponse:
    """A record as the one extracted key ``result_key`` (confidence 1.0).

    Contract: the wrapper every call-shaped record and every synthesize record shares
    (``write_call_request``, ``refuse``, ``SynthesizeExecutor``); it copies nothing and checks
    nothing, so the caller hands it a record it built."""
    return DataExtractionResponse(extracted_data={result_key: record}, confidence=1.0)


def write_call_request(
    result_key: str,
    *,
    state_id: str,
    target: str,
    resolved: ResolvedTarget,
    inputs: Mapping[str, Any],
    max_steps: int | None,
    max_seconds: float | None,
) -> DataExtractionResponse:
    """The ``running`` request record of a call, as the one extracted key ``result_key``.

    ``state_id`` is the id of the call state, stamped as ``state`` (``record_belongs``)."""
    record = {
        "status": Status.RUNNING.value,
        "state": state_id,
        "target": target,
        **_identities(resolved),
        "mode": TARGET_MODE_FRAME,
        "request": _request_body(target, inputs, max_steps, max_seconds),
    }
    return record_response(result_key, record)


def _request_body(
    target: str,
    inputs: Mapping[str, Any],
    max_steps: int | None,
    max_seconds: float | None,
) -> dict[str, Any]:
    """The ``request`` of a running record: the target, the declared inputs as recorded values
    and the budgets of the call (evidence and durable intent, never an instruction to settle)."""
    return {
        "target": target,
        "inputs": {name: _recorded(value) for name, value in inputs.items()},
        "max_steps": max_steps,
        "max_seconds": max_seconds,
    }


def write_delegated_request(
    result_key: str,
    *,
    state_id: str,
    target: str,
    delegated: DelegatedTarget,
    handle: str,
    inputs: Mapping[str, Any],
    max_steps: int | None,
    max_seconds: float | None,
) -> DataExtractionResponse:
    """The ``running`` request record of a DELEGATED call (``mode: delegated``, the minted
    ``handle`` and the wall-clock ``deadline``), as the one extracted key ``result_key``."""
    record = {
        "status": Status.RUNNING.value,
        "state": state_id,
        "target": target,
        **_identities(delegated),
        "mode": TARGET_MODE_DELEGATED,
        "handle": handle,
        "deadline": time.time() + delegated_seconds(max_seconds),
        "request": _request_body(target, inputs, max_steps, max_seconds),
    }
    return record_response(result_key, record)


def final_record(
    state_id: str,
    status: Status,
    reason: Reason | None,
    target: str | None,
    template_identity: str | None,
    instance_identity: str | None,
) -> dict[str, Any]:
    """The FINAL record of a call: the projection the loader's sample results pin
    (``CALL_SAMPLES``) plus the ``state`` stamp (``record_belongs``).

    Contract: the one builder of a final record, shared by the executor (a refusal, a finished
    delegated run) and by ``settle`` (a pop, a refusal); it copies nothing and checks nothing.
    """
    return {
        "status": status.value,
        "state": state_id,
        "reason": None if reason is None else reason.value,
        "target": target,
        "template_identity": template_identity,
        "instance_identity": instance_identity,
    }


def refuse(
    result_key: str,
    reason: Reason | str,
    *,
    state_id: str,
    target: str | None = None,
    resolved: ResolvedTarget | DelegatedTarget | None = None,
) -> DataExtractionResponse:
    """The FINAL ``failed`` record of a refused call, as the one extracted key ``result_key``.

    ``state_id`` is the id of the call state, stamped as ``state`` (``record_belongs``)."""
    return record_response(
        result_key,
        final_record(
            state_id,
            Status.FAILED,
            Reason.from_string(reason),
            target,
            **_identities(resolved),
        ),
    )


def commit_arrival(
    pipeline: _PipelineServices,
    instance: FSMInstance,
    committed: dict[str, Any],
    state_id: str,
) -> None:
    """Commit the record an executor decided ON ARRIVAL at ``state_id`` (D-026 (1)).

    Contract: the post-transition commit shape: ``committed`` (the extracted data of the
    decision, the one result key) is written to the context, then the CONTEXT_UPDATE handlers
    run for those keys; a handler that fails undoes only them (one WARNING naming the state).
    Shared by ``CallExecutor`` and ``SynthesizeExecutor``; the source state's
    ``extraction_response`` is not touched."""
    commit_with_key_rollback(
        instance.context,
        committed,
        lambda: pipeline.execute_handlers(
            instance,
            HandlerTiming.CONTEXT_UPDATE,
            current_state=instance.current_state,
            updated_keys=set(committed),
        ),
        lambda error: logger.warning(
            f"CONTEXT_UPDATE handler failed ({type(error).__name__}: {error}), "
            f"rolling back the call record of state '{state_id}'"
        ),
    )


def chosen_target(config: CallStateConfig, data: Mapping[str, Any]) -> str | None:
    """The target id a call state calls now, or ``None``.

    Contract: ``config`` is ``state.call``; ``data`` the context of the frame. The fixed
    ``target``; or, for ``choose``, the value of ``choose.key`` when it is a ``str`` in the
    closed ``targets`` list, else ``fallback`` (``None`` when there is none). Reads nothing
    else and never calls anything, so the executor and ``settle`` share ONE rule and a value
    outside the list is never an id to resolve.
    """
    choose = config.choose
    if choose is None:
        target: str | None = config.target
        return target
    value = data.get(choose.key)
    if isinstance(value, str) and value in choose.targets:
        return value
    fallback: str | None = choose.fallback
    return fallback


class CallExecutor(StateExecutor):
    """The call kind: Pass 1 resolves the target and writes a request or a refusal."""

    # Why False: the record is the one typed result of the state and is committed as is, like a
    # completion result: ``clean_context_keys`` drops None values at every depth (``target:
    # None`` and ``reason`` of a refusal are part of the shape).
    cleans_extracted = False
    kind = "call"

    def extract(
        self,
        pipeline: _PipelineServices,
        instance: FSMInstance,
        user_message: str | None,
        conversation_id: str,
        state: State,
    ) -> DataExtractionResponse:
        return run_sync(
            self.aextract(pipeline, instance, user_message, conversation_id, state)
        )

    async def aextract(
        self,
        pipeline: _PipelineServices,
        instance: FSMInstance,
        user_message: str | None,
        conversation_id: str,
        state: State,
    ) -> DataExtractionResponse:
        config = state.call
        if config is None:
            raise FSMError(
                f"CallExecutor.extract needs a state with a call config; state "
                f"'{state.id}' has none"
            )
        data = instance.context.data
        record = data.get(config.result_key)
        if not record_belongs(record, state.id):
            response = self._decide(pipeline, data, conversation_id, state, config)
        elif is_delegated_running(record, state.id):
            response = self._poll(
                pipeline, data, conversation_id, state, config, record
            )
        else:
            response = DataExtractionResponse(extracted_data={}, confidence=1.0)
        instance.last_extraction_response = response
        return response

    def after_transition(
        self,
        pipeline: _PipelineServices,
        instance: FSMInstance,
        user_message: str | None,
        conversation_id: str,
        extraction_response: DataExtractionResponse,
        previous_state: str | None,
        new_state: State,
    ) -> None:
        run_sync(
            self.aafter_transition(
                pipeline,
                instance,
                user_message,
                conversation_id,
                extraction_response,
                previous_state,
                new_state,
            )
        )

    async def aafter_transition(
        self,
        pipeline: _PipelineServices,
        instance: FSMInstance,
        user_message: str | None,
        conversation_id: str,
        extraction_response: DataExtractionResponse,
        previous_state: str | None,
        new_state: State,
    ) -> None:
        """Decide ON ARRIVAL (D-026 (1)): the request or the refusal is written in the turn
        that enters the call state, so the end-of-entry settle pushes the child in that turn
        and the user's next message goes to it. The write is the post-transition commit shape
        (the keys, then the CONTEXT_UPDATE handlers for them; a failed handler undoes only
        them); the source state's ``extraction_response`` is not touched. A value at the key
        that already belongs to the state (a revisit) is skipped, as in ``aextract``."""
        config = new_state.call
        if config is None:
            raise FSMError(
                f"CallExecutor.after_transition needs a state with a call config; state "
                f"'{new_state.id}' has none"
            )
        data = instance.context.data
        if record_belongs(data.get(config.result_key), new_state.id):
            return
        commit_arrival(
            pipeline,
            instance,
            self._decide(
                pipeline, data, conversation_id, new_state, config
            ).extracted_data,
            new_state.id,
        )

    @staticmethod
    def _decide(
        pipeline: _PipelineServices,
        data: Mapping[str, Any],
        conversation_id: str,
        state: State,
        config: CallStateConfig,
    ) -> DataExtractionResponse:
        resolver = pipeline.target_resolver
        if resolver is None:
            raise MissingExecutorError(
                f"State '{state.id}' is of kind 'call', which resolves its target through a "
                "TargetResolver and this pipeline has none: pass "
                "targets=<a TargetResolver, for example an ExecutableRegistry> to API, "
                "FSMManager or MessagePipeline",
                state_id=state.id,
                kind="call",
            )
        key = config.result_key
        target = chosen_target(config, data)
        if target is None:
            return refuse(key, Reason.UNKNOWN_TARGET, state_id=state.id)
        launcher = resolver if is_run_launcher(resolver) else None
        resolved = resolver.resolve_target(target, conversation_id=conversation_id)
        delegated: DelegatedTarget | None = None
        if resolved is None and launcher is not None:
            # only an id the frame resolver does not answer may be a delegated one
            delegated = launcher.resolve_run(
                target,
                input_keys=config.inputs,
                max_steps=config.max_steps,
                conversation_id=conversation_id,
            )
        found = resolved or delegated
        if found is None:
            return refuse(key, Reason.UNKNOWN_TARGET, state_id=state.id, target=target)
        inputs: dict[str, Any] = {}
        for name in config.inputs:
            value = data.get(name)
            if value is None:
                return refuse(
                    key,
                    Reason.INPUTS_MISSING,
                    state_id=state.id,
                    target=target,
                    resolved=found,
                )
            inputs[name] = value
        if resolved is not None:
            return write_call_request(
                key,
                state_id=state.id,
                target=target,
                resolved=resolved,
                inputs=inputs,
                max_steps=config.max_steps,
                max_seconds=config.max_seconds,
            )
        if launcher is not None and delegated is not None:
            handle = f"run-{uuid.uuid4().hex}"
            if launcher.claim(handle, conversation_id=conversation_id):
                return write_delegated_request(
                    key,
                    state_id=state.id,
                    target=target,
                    delegated=delegated,
                    handle=handle,
                    inputs=inputs,
                    max_steps=config.max_steps,
                    max_seconds=config.max_seconds,
                )
            reason = Reason.PUSH_FAILED  # the host is full: nothing was reserved
        else:
            reason = Reason.UNKNOWN_TARGET  # unreachable: `found` is one of the two
        return refuse(key, reason, state_id=state.id, target=target, resolved=found)

    # DECISION plan-2026-11-23T122249-baa37f40/D-051
    # A delegated call is POLLED, never skipped and never waited for. Do NOT skip on a `running`
    # record of `mode: delegated` (the run lives on a host thread and its end must land in a later
    # turn), do NOT wait inside the turn (a lock may be held and the loop would stall: the one
    # bounded wait is at the end of `aadvance` and of the step streams), do NOT read the target or
    # the identities of the final record from the record (a forged public `call_result` could name
    # any: the target is re-derived from the parent's own state, the identities are the host's own,
    # taken from the poll, D-058), do NOT trust a `deadline` that is not a finite number (a
    # forged one that never comes is a run that never stops: fail closed), and do NOT write the
    # returns and the final record in two passes (ONE extraction, so no window shows the returns
    # with the record still `running`). A handle the launcher never saw FOR THIS conversation is
    # `failed/run_lost`: nothing is ever re-run.
    @staticmethod
    def _poll(
        pipeline: _PipelineServices,
        data: Mapping[str, Any],
        conversation_id: str,
        state: State,
        config: CallStateConfig,
        record: Mapping[str, Any],
    ) -> DataExtractionResponse:
        launcher = pipeline.target_resolver
        target = chosen_target(config, data)

        def settled(
            status: Status,
            reason: Reason | None,
            merged: Mapping[str, Any] | None = None,
            polled: RunPoll | None = None,
        ) -> DataExtractionResponse:
            template_identity = instance_identity = None
            if polled is not None and polled.template_identity is not None:
                # what the host STARTED the run with: a reload during the run, or a forged
                # public record, never changes the identities of the end (D-058)
                template_identity = polled.template_identity
                instance_identity = polled.instance_identity
            elif is_run_launcher(launcher) and target is not None:
                now = launcher.resolve_run(
                    target,
                    input_keys=config.inputs,
                    max_steps=config.max_steps,
                    conversation_id=conversation_id,
                )
                if now is not None:
                    template_identity = now.template_identity
                    instance_identity = now.instance_identity
            final = final_record(
                state.id, status, reason, target, template_identity, instance_identity
            )
            return DataExtractionResponse(
                extracted_data={config.result_key: final, **(merged or {})},
                confidence=1.0,
            )

        handle = record.get("handle")
        if not is_run_launcher(launcher) or not isinstance(handle, str):
            return settled(Status.FAILED, Reason.RUN_LOST)
        polled = (
            launcher.poll(handle, conversation_id=conversation_id)
            if handle.strip()
            else None
        )
        if polled is None:
            return settled(Status.FAILED, Reason.RUN_LOST)
        if polled.finished:
            if polled.result is None:  # a finished run with no result is a lost one
                return settled(Status.FAILED, Reason.RUN_LOST)
            return _finished(
                polled.result,
                config,
                lambda status, reason, merged=None: settled(
                    status, reason, merged, polled
                ),
            )
        deadline = record.get("deadline")
        if (
            isinstance(deadline, (int, float))
            and not isinstance(deadline, bool)
            and math.isfinite(deadline)
            and time.time() < deadline
        ):
            return DataExtractionResponse(extracted_data={}, confidence=1.0)
        launcher.cancel(
            handle, conversation_id=conversation_id
        )  # a pattern is only detached
        return settled(Status.ABORTED, Reason.BUDGET_SPENT, None, polled)


def is_delegated_running(record: Any, state_id: str) -> TypeGuard[dict[str, Any]]:
    """Whether a value at a call state's result key is a ``running`` DELEGATED record of
    ``state_id`` (the one record the executor polls instead of skipping).

    Contract: ``True`` for a dict that belongs to the state (``record_belongs``), whose
    ``status`` is ``running`` and whose ``mode`` is ``delegated``; ``False`` for anything else.
    Never raises."""
    return (
        record_belongs(record, state_id)
        and record.get("status") == Status.RUNNING.value
        and record.get("mode") == TARGET_MODE_DELEGATED
    )


#: The pseudo output of a delegated target: ``returns`` may name it to take ``RunResult.answer``
#: (a pattern declares no outputs; D-016 (b)). A declared output of that name wins.
ANSWER_OUTPUT = "answer"


def _finished(
    result: RunResult,
    config: CallStateConfig,
    settled: Callable[..., DataExtractionResponse],
) -> DataExtractionResponse:
    """The final record, and the declared ``returns``, of a finished delegated run (D-051 (5))."""
    if result.status is Status.OK:
        merged: dict[str, Any] = {}
        for parent_key, child_key in (config.returns or {}).items():
            value = result.outputs.get(child_key)
            if value is None and child_key == ANSWER_OUTPUT:
                value = result.answer
            if value is None:
                return settled(Status.FAILED, Reason.RETURN_KEYS_MISSING)
            merged[parent_key] = bounded_value(value)
        return settled(Status.OK, None, merged)
    if result.status is Status.ABORTED:
        return settled(Status.ABORTED, result.reason or Reason.BUDGET_SPENT)
    if result.status is Status.FAILED:
        return settled(Status.FAILED, result.reason or Reason.UNEXPECTED_ERROR)
    return settled(Status.FAILED, Reason.UNEXPECTED_ERROR)  # abstained or still running
