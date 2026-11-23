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

Interface contract:

- ``CallExecutor.extract`` / ``aextract(pipeline, instance, user_message, conversation_id,
  state)``: the one Pass-1 body (``extract`` is ``run_sync`` over ``aextract``). Skip-if-set: a
  non-None value at the result key (any status) returns no data and asks no resolver. A
  pipeline with no resolver raises ``MissingExecutorError``; a state with no ``call`` config
  raises ``FSMError``.
- ``write_call_request(result_key, *, target, resolved, inputs, max_steps, max_seconds)`` and
  ``refuse(result_key, reason, *, target=None, resolved=None)``: the shared body (the
  synthesize executor of step 20 reuses both). Each returns a ``DataExtractionResponse`` with
  confidence 1.0 and exactly the result key. ``reason`` is a ``core.codes.Reason`` or its wire
  string (``ValueError`` for another). ``request.inputs`` holds the values it is given, each
  through ``runtime.to_jsonable`` and cut to a marker past ``_REQUEST_VALUE_MAX_CHARS``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from fsm_llm.aio import run_sync
from fsm_llm.core.codes import Reason, Status
from fsm_llm.core.definitions import (
    CallStateConfig,
    DataExtractionResponse,
    FSMError,
    FSMInstance,
    MissingExecutorError,
    State,
)
from fsm_llm.runtime import to_jsonable

from .base import StateExecutor
from .services import _PipelineServices
from .targets import TARGET_MODE_FRAME, ResolvedTarget

#: A declared input whose JSON text is longer than this is recorded as a one-line marker (the
#: value itself still reaches the child): the record stays small and is never a second copy of
#: a large input in every prompt.
_REQUEST_VALUE_MAX_CHARS = 2000


def _recorded(value: Any) -> Any:
    """A JSON-safe copy of an input value, or a marker naming its type and size when large."""
    safe = to_jsonable(value)
    text = json.dumps(safe, ensure_ascii=False, separators=(",", ":"))
    if len(text) <= _REQUEST_VALUE_MAX_CHARS:
        return safe
    return f"<{type(value).__name__}, {len(text)} characters of JSON, not recorded>"


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


def _identities(resolved: ResolvedTarget | None) -> dict[str, str | None]:
    return {
        "template_identity": None if resolved is None else resolved.template_identity,
        "instance_identity": None if resolved is None else resolved.instance_identity,
    }


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
        "request": {
            "target": target,
            "inputs": {name: _recorded(value) for name, value in inputs.items()},
            "max_steps": max_steps,
            "max_seconds": max_seconds,
        },
    }
    return DataExtractionResponse(extracted_data={result_key: record}, confidence=1.0)


def refuse(
    result_key: str,
    reason: Reason | str,
    *,
    state_id: str,
    target: str | None = None,
    resolved: ResolvedTarget | None = None,
) -> DataExtractionResponse:
    """The FINAL ``failed`` record of a refused call, as the one extracted key ``result_key``.

    ``state_id`` is the id of the call state, stamped as ``state`` (``record_belongs``)."""
    record = {
        "status": Status.FAILED.value,
        "state": state_id,
        "reason": Reason.from_string(reason).value,
        "target": target,
        **_identities(resolved),
    }
    return DataExtractionResponse(extracted_data={result_key: record}, confidence=1.0)


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
        response = (
            DataExtractionResponse(extracted_data={}, confidence=1.0)
            if record_belongs(data.get(config.result_key), state.id)
            else self._decide(pipeline, data, conversation_id, state, config)
        )
        instance.last_extraction_response = response
        return response

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
        resolved = resolver.resolve_target(target, conversation_id=conversation_id)
        if resolved is None:
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
                    resolved=resolved,
                )
            inputs[name] = value
        return write_call_request(
            key,
            state_id=state.id,
            target=target,
            resolved=resolved,
            inputs=inputs,
            max_steps=config.max_steps,
            max_seconds=config.max_seconds,
        )
