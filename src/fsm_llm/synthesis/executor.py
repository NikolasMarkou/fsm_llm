"""``SynthesizeExecutor``: the synthesize kind on the engine, no second driver.

Layer ``synthesis`` (plan baa37f40, step 20.2, D-038, roadmap E3.5 and P7.1). The deleted
``ProcedureRunner`` gave the ``synthesize`` field its meaning from outside the engine, with a hook
of its own, its own bookkeeping and its own converse loop. This executor gives it the meaning inside
the engine, the way ``CallExecutor`` does for the ``call`` kind (a bare ``API`` with no such
executor refuses a ``synthesize`` state with ``MissingExecutorError``, plan baa37f40 D-041): Pass 1
of a synthesize state obtains a procedure and writes a REQUEST, and ``engine/api/calls.py``
(``settle``) pushes the child after the turn committed and pops it when it ends. Register it with
``API(executors={"synthesize": SynthesizeExecutor(synth)})`` (or the list form).

Interface contract:

- ``SynthesizeExecutor(synthesizer, *, trace_sink=None)``: ``synthesizer`` obtains the
  procedures; ``trace_sink(event, fields)`` an optional plain callable (``TypeError`` if not
  callable). The events are the six of the old runner, kept as the wire of ``TraceRecorder``
  (``obtained``, ``pushed``, ``stepped``, ``aborted``, ``popped``, ``failed``), scalars only, never a value, user text or reply (D-044), every one with
  ``conv`` (the conversation id of the frame that holds the synthesize state, the root's for a
  run of the root); a sink that raises is swallowed with one WARNING naming the event.
  ``step`` is the engine's count of the child's turns (0 at the push and at a refusal) for a run
  that started on a message-free step, ``None`` for one that started on a user message
  (``RunTrace.mode``); ``stepped`` always carries the count. ``kind`` is ``"synthesize"``.
- Pass 1 (``extract``/``aextract`` for a turn that begins in the state, ``after_transition``/
  ``aafter_transition`` ON ARRIVAL, so the user's next message goes to the child): skip-if-set
  (``engine.kinds.call.record_belongs``); then depth (``policy.max_synthesis_depth`` against the
  depth of the conversation: the executor learns it at each push), inputs present, ``obtain``
  (through ``acall``: a patch of the sync ``obtain`` intercepts in sync mode, ``aobtain`` runs in
  async mode), ``runnable_reason`` and the persona merge. A refusal is the FINAL record
  ``{status: failed, procedure_id, version, reason}`` as extracted data, so the SAME turn takes
  the ``on_failure`` edge. Success is the ``running`` record of the call kind
  (``write_call_request``: ``state``, ``mode``, ``request``, the two identities) plus
  ``procedure_id``, ``version`` and ``reason: None``.
- ``settle`` hooks: ``aplan_push`` re-resolves the procedure from the record's ``procedure_id``
  (a lookup key only: this executor's bounded table, else the library) and judges it AGAIN
  against the state's own ``inputs``, ``return_keys`` and ``min_lifecycle``; ``arefusal`` writes
  the final record of a refusal settle decided (the depth, a push the engine refused, a greeting
  outage); ``on_pushed`` emits ``pushed`` and notes the child; ``on_child_turn`` and
  ``aplan_pop`` emit ``stepped`` (and ``aborted`` for a spent budget); ``aplan_pop`` reads the
  outcome BEFORE the pop and returns the one merge (the return keys only when the run is ok, and
  the final record); ``on_ended`` counts the run in the library (never for an outage on a run
  that is not ok, D-061) and emits ``popped``.
- The host drives the parent: the engine runs no parent turn after a pop (D-026 (2)) and starts
  no Pass 1 for a child whose INITIAL state is a synthesize state, so a host that wants the run
  to continue in the same ``converse`` calls ``advance`` after it, as the old runner's ``_settle``
  did. ``run_until_terminal`` needs nothing.
- Failure mode: no exception reaches the caller from a refusal (a final ``failed`` record); a
  ``CancelledError``, an interrupt or any other ``BaseException`` during ``obtain`` propagates and
  no record is written, nothing is pending and no run is counted (a cancel rolls the turn back
  whole; after an interrupt the next turn in the state obtains again, D-039).
"""

from __future__ import annotations

import copy
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from fsm_llm.aio import acall, run_sync
from fsm_llm.core.codes import Reason
from fsm_llm.core.definitions import (
    DataExtractionResponse,
    FSMContext,
    FSMDefinition,
    FSMError,
    FSMInstance,
    State,
    SynthesizeConfig,
    definition_digest,
    fsm_id_of,
)
from fsm_llm.core.expressions import is_missing
from fsm_llm.core.transition_evaluator import TransitionEvaluator
from fsm_llm.engine.kinds.base import StateExecutor
from fsm_llm.engine.kinds.call import (
    commit_arrival,
    record_belongs,
    record_response,
    write_call_request,
)
from fsm_llm.engine.kinds.requests import (
    ChildView,
    PopContext,
    PushContext,
    PushPlan,
)
from fsm_llm.engine.kinds.services import _PipelineServices
from fsm_llm.engine.kinds.targets import ResolvedTarget
from fsm_llm.logging import logger

from .constants import (
    ABORT_STATE_ID,
    LIFECYCLE_VERIFIED,
    REASON_BUDGET_SPENT,
    REASON_DEPTH_REFUSED,
    REASON_INPUTS_MISSING,
    REASON_PROVIDER_ERROR,
    REASON_PUSH_FAILED,
    REASON_RETURN_KEYS_MISSING,
    REASON_UNEXPECTED_ERROR,
    REASONS,
    STATUS_ABORTED,
    STATUS_FAILED,
    STATUS_OK,
    TRACE_KIND_OTHER,
    TRACE_KINDS,
)
from .exceptions import LibraryError, SynthesisError
from .procedures import (
    _result,
    _round_codes,
    _round_string,
    reason_of,
    runnable_reason,
)
from .synthesizer import FSMSynthesizer, Procedure
from .traces import gate_reads

#: Procedures this executor obtained that are kept to be pushed (the lookup key is the digest of
#: the record); past it the oldest is dropped and a later push finds it in the library, or not.
_MAX_KNOWN = 256

#: Runs, depths and start marks kept in memory (one entry per child, dropped at its pop).
_MAX_TRACKED = 4096


@dataclass
class _Run:
    """One child this executor has pushed: what the pop and the trace need."""

    procedure: Procedure
    config: SynthesizeConfig
    converse: bool
    traced: int = 0
    ok: bool = False
    outage: bool = False


def _bounded(table: dict[Any, Any], key: Any, value: Any, cap: int) -> None:
    table[key] = value
    while len(table) > cap:
        table.pop(next(iter(table)))


# DECISION plan-2026-11-23T122249-baa37f40/D-038
# The synthesize kind runs on the engine through this executor and `settle`; the record is
# evidence and never an instruction. Do NOT read anything of the record but the lookup key
# `procedure_id` (judge what it names AGAIN with `runnable_reason` against the state's own
# inputs, return keys and lifecycle before it is pushed), do NOT hand the child any key but the
# declared inputs (no `inherit_context`, never `_approval_granted`), do NOT push the persona of a
# compile (the persona is the PARENT frame's: G5 forbids a compile persona and the pushed digest
# differs from the library digest, which is why `procedure_id` is the library digest and
# `instance_identity` the pushed one), do NOT write to the library before the run ends
# (`record_run` only in `on_ended`, never for an outage on a run that is not ok), do NOT catch
# `BaseException` in Pass 1 (a cancel or an interrupt during `obtain` writes no record and counts
# no run; a cancel also rolls the turn back whole) and do NOT import `engine.api`, `engine.turn` or the manager.
# Everything the executor needs of the stack arrives as a plain value (`engine/kinds/requests`).
class SynthesizeExecutor(StateExecutor):
    """The synthesize kind: Pass 1 obtains and requests, settle pushes and pops (see the module
    docstring for the contract)."""

    # Why False: the record is the one typed result of the state and is committed as is, like a
    # completion result and a call record: ``clean_context_keys`` drops None values at every
    # depth (``procedure_id: None`` and ``reason: None`` are part of the wire shape).
    cleans_extracted = False
    kind = "synthesize"
    serves_requests = True
    watches_outage = True

    def __init__(
        self,
        synthesizer: FSMSynthesizer,
        *,
        trace_sink: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        if trace_sink is not None and not callable(trace_sink):
            raise TypeError("trace_sink must be callable")
        self.synthesizer = synthesizer
        self._sink = trace_sink
        self._lock = (
            threading.Lock()
        )  # a leaf: never held across an await or a call out
        self._known: OrderedDict[str, Procedure] = OrderedDict()
        self._runs: dict[str, _Run] = {}  # child conversation id -> run
        self._depths: dict[str, int] = {}  # conversation id -> synthesis depth
        self._by_message: dict[
            tuple[str, str], bool
        ] = {}  # (frame, state) -> user message
        self._planned: dict[
            tuple[str, str], Procedure
        ] = {}  # (frame, state) -> to push

    # --- Pass 1 --------------------------------------------------------------

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
        config = self._config(state)
        if record_belongs(instance.context.data.get(config.result_key), state.id):
            response = DataExtractionResponse(extracted_data={}, confidence=1.0)
        else:
            response = await self._start(
                pipeline, instance, user_message, conversation_id, state, config
            )
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
        """Start ON ARRIVAL (as the call kind does, D-026): the request or the refusal is part of
        the arriving turn, so the end-of-entry settle pushes the child in that turn."""
        config = self._config(new_state)
        if record_belongs(instance.context.data.get(config.result_key), new_state.id):
            return
        response = await self._start(
            pipeline, instance, user_message, conversation_id, new_state, config
        )
        commit_arrival(pipeline, instance, response.extracted_data, new_state.id)

    @staticmethod
    def _config(state: State) -> SynthesizeConfig:
        config = state.synthesize
        if config is None:
            raise FSMError(
                f"SynthesizeExecutor needs a state with a synthesize config; state "
                f"'{state.id}' has none"
            )
        return config

    async def _start(
        self,
        pipeline: _PipelineServices,
        instance: FSMInstance,
        user_message: str | None,
        conversation_id: str,
        state: State,
        config: SynthesizeConfig,
    ) -> DataExtractionResponse:
        """Everything refusable BEFORE the turn commits; the request, or a final ``failed``."""
        data = instance.context.data
        policy = self.synthesizer.policy
        procedure: Procedure | None = None
        error: BaseException | None = None
        reason: str | None = None
        converse = user_message is not None
        try:
            if self._depth_of(conversation_id) >= policy.max_synthesis_depth:
                reason = REASON_DEPTH_REFUSED
            elif any(is_missing(data, key) for key in config.inputs):
                reason = REASON_INPUTS_MISSING
            else:
                try:
                    procedure = await acall(
                        self.synthesizer,
                        "obtain",
                        config.goal,
                        inputs=config.inputs,
                        return_keys=config.return_keys,
                        min_lifecycle=config.min_lifecycle,
                    )
                except (SynthesisError, ValueError) as e:
                    reason, error = reason_of(e), e
                else:
                    self._emit_obtained(conversation_id, procedure)
                    reason = self._unrunnable(procedure, config)
                    if reason is None:
                        host = pipeline.fsm_resolver(instance.fsm_id)
                        self._remember(procedure)
                        with self._lock:
                            _bounded(
                                self._by_message,
                                (conversation_id, state.id),
                                converse,
                                _MAX_TRACKED,
                            )
                        return self._request(state, config, procedure, host, data)
        except Exception as e:  # fail closed: the result must never stay `running`
            logger.warning(f"synthesis ended on an unexpected {type(e).__name__}")
            reason, error = REASON_UNEXPECTED_ERROR, e
        assert reason is not None
        return record_response(
            config.result_key,
            self._refusal(
                conversation_id,
                reason,
                procedure,
                error,
                step=None if converse else 0,
            ),
        )

    def _request(
        self,
        state: State,
        config: SynthesizeConfig,
        procedure: Procedure,
        host: FSMDefinition,
        data: dict[str, Any],
    ) -> DataExtractionResponse:
        """The ``running`` record: the call kind's request plus the wire keys (D-038 (2))."""
        pushed = self._with_persona(procedure.definition, host.persona)
        response = write_call_request(
            config.result_key,
            state_id=state.id,
            target=procedure.digest,
            resolved=ResolvedTarget(
                definition=pushed,
                template_identity=procedure.digest,
                instance_identity=definition_digest(pushed),
            ),
            inputs={key: data[key] for key in config.inputs},
            max_steps=self.synthesizer.policy.max_turns,
            max_seconds=None,
        )
        record = response.extracted_data[config.result_key]
        record.update(
            procedure_id=procedure.digest,
            version=procedure.record.version if procedure.record else None,
            reason=None,
        )
        return response

    def _refusal(
        self,
        conversation_id: str,
        reason: str,
        procedure: Procedure | None,
        error: BaseException | None,
        *,
        step: int | None,
    ) -> dict[str, Any]:
        """The final ``failed`` record (the wire shape) and its ``failed`` event."""
        self._emit(
            "failed",
            conversation_id,
            reason=reason,
            step=step,
            digest=procedure.digest if procedure else None,
            error=type(error).__name__ if error else None,
            rounds=_round_codes(error),
        )
        return _result(STATUS_FAILED, reason, procedure)

    # --- the procedure the record names -----------------------------------------

    def _unrunnable(self, procedure: Procedure, config: SynthesizeConfig) -> str | None:
        """The reason ``procedure`` may not run under ``config``, else ``None``."""
        record = procedure.record
        return runnable_reason(
            procedure.definition,
            record.lifecycle if record else LIFECYCLE_VERIFIED,
            min_lifecycle=config.min_lifecycle,
            policy=self.synthesizer.policy,
            inputs=config.inputs,
            return_keys=config.return_keys,
            gates_passed=record.provenance["gates_passed"] if record else None,
        )

    @staticmethod
    def _with_persona(definition: FSMDefinition, persona: str | None) -> FSMDefinition:
        # a compile may not set a persona (G5): the PARENT frame's is injected here (P7, D-038)
        return definition.model_copy(update={"persona": persona})

    def _remember(self, procedure: Procedure) -> None:
        with self._lock:
            self._known[procedure.digest] = procedure
            self._known.move_to_end(procedure.digest)
            while len(self._known) > _MAX_KNOWN:
                self._known.popitem(last=False)

    def _procedure_of(self, digest: object) -> Procedure | None:
        """The procedure named by ``digest``: this executor's table, else the library (a restart)."""
        if not isinstance(digest, str):
            return None
        with self._lock:
            known = self._known.get(digest)
        if known is not None:
            return known
        library = self.synthesizer.library
        if library is None:
            return None
        try:
            return Procedure(
                digest,
                library.get_definition(digest),
                library.get_record(digest),
                "library",
            )
        except LibraryError:
            return None

    def _depth_of(self, conversation_id: str) -> int:
        with self._lock:
            return self._depths.get(conversation_id, 0)

    # --- settle: the push ------------------------------------------------------------

    async def aplan_push(self, ctx: PushContext) -> PushPlan | Mapping[str, Any] | None:
        config = ctx.state.synthesize
        if config is None:
            return None
        procedure = self._procedure_of(ctx.record.get("procedure_id"))
        reason: str | None = (
            REASON_PUSH_FAILED  # a record that names nothing this can push
        )
        error: BaseException | None = None
        try:
            if procedure is not None:
                reason = self._unrunnable(procedure, config)
            if reason is None and procedure is not None:
                missing = any(is_missing(ctx.data, key) for key in config.inputs)
                if missing:
                    reason = REASON_INPUTS_MISSING
                else:
                    with self._lock:
                        _bounded(
                            self._planned,
                            (ctx.parent_id, ctx.state.id),
                            procedure,
                            _MAX_TRACKED,
                        )
                    return PushPlan(
                        self._with_persona(
                            procedure.definition, ctx.definition.persona
                        ),
                        {key: copy.deepcopy(ctx.data[key]) for key in config.inputs},
                        self.synthesizer.policy.max_turns,
                        None,
                    )
        except Exception as e:  # fail closed
            logger.warning(f"synthesis ended on an unexpected {type(e).__name__}")
            reason, error = REASON_UNEXPECTED_ERROR, e
        assert reason is not None
        return self._refusal(ctx.parent_id, reason, procedure, error, step=0)

    async def arefusal(self, ctx: PushContext, reason: Reason) -> Mapping[str, Any]:
        with self._lock:
            procedure = self._planned.pop((ctx.parent_id, ctx.state.id), None)
        code = reason.value if reason.value in REASONS else REASON_PUSH_FAILED
        return self._refusal(ctx.parent_id, code, procedure, None, step=0)

    async def on_pushed(
        self,
        root_id: str,
        parent_id: str,
        child_id: str,
        state: State,
        record: Mapping[str, Any],
    ) -> None:
        config = state.synthesize
        with self._lock:
            procedure = self._planned.pop((parent_id, state.id), None)
            converse = self._by_message.pop((parent_id, state.id), False)
            if procedure is None or config is None:
                return
            _bounded(
                self._runs, child_id, _Run(procedure, config, converse), _MAX_TRACKED
            )
            _bounded(
                self._depths, child_id, self._depths.get(parent_id, 0) + 1, _MAX_TRACKED
            )
        self._emit(
            "pushed",
            parent_id,
            digest=procedure.digest,
            step=None if converse else 0,
        )

    # --- settle: the child's turns and its end -----------------------------------------

    async def on_child_turn(self, ctx: PopContext) -> None:
        run = self._runs.get(ctx.child.conversation_id)
        if run is not None:
            self._trace_steps(ctx.parent_id, run, ctx.child)

    async def aplan_pop(self, ctx: PopContext) -> Mapping[str, Any] | None:
        config = ctx.state.synthesize
        if config is None:
            return None
        child = ctx.child
        run = self._runs.get(child.conversation_id) or self._proved(ctx, config)
        if run is None:
            # a run this executor cannot prove after a restart: nothing merges and no run counts
            return {config.result_key: _result(STATUS_FAILED, REASON_PUSH_FAILED, None)}
        self._trace_steps(ctx.parent_id, run, child)
        # DECISION plan-2026-11-10T194139-2cefeb61/D-044
        # Read the outcome BEFORE the child is popped and the result written: afterwards the
        # top frame is the parent, and its state says nothing of the child's edges. (The rule
        # travelled here from the deleted runner, plan baa37f40 step 22, D-041.)
        if not child.ended:
            self._emit(
                "aborted",
                ctx.parent_id,
                digest=run.procedure.digest,
                step=None if run.converse else child.steps,
                **self._view(run, child, outcome=True),
            )
        aborted = not child.ended or child.state_id == ABORT_STATE_ID
        ok = not aborted and not any(
            is_missing(child.data, key) for key in config.return_keys
        )
        # DECISION plan-2026-11-10T194139-2cefeb61/D-061
        # Core swallows an extraction error, so an outage looks like a stuck child; a run whose
        # interface errored is `provider_error`, which `diagnose` never patches. Do NOT call it
        # `budget_spent`, and do NOT let an ok run (a retried call) become a fault.
        outage = child.errors > 0
        status = STATUS_OK if ok else STATUS_ABORTED if aborted else STATUS_FAILED
        reason = (
            None
            if ok
            else REASON_PROVIDER_ERROR
            if outage
            else REASON_BUDGET_SPENT
            if aborted
            else REASON_RETURN_KEYS_MISSING
        )
        run.ok, run.outage = ok, outage
        merged = {key: child.data[key] for key in config.return_keys} if ok else {}
        # DECISION plan-2026-11-10T194139-2cefeb61/D-029
        # Return keys go back ONLY when every key is present and the child was not aborted, in
        # the one merge of the pop. Do NOT pass `shared_context_keys` at the push (D-017 of that
        # plan): a failed or aborted child would then merge partial data into the parent. Read
        # the child's data BEFORE the pop (after it the top of the stack is the parent).
        return {**merged, config.result_key: _result(status, reason, run.procedure)}

    def _proved(self, ctx: PopContext, config: SynthesizeConfig) -> _Run | None:
        """The run of a child this executor has no memo of (a restart): the procedure the record
        names, ONLY when the child frame runs exactly its persona-merged definition (the D-027
        rule: the frame's own ``fsm_id`` proves what ran)."""
        procedure = self._procedure_of(ctx.record.get("procedure_id"))
        if procedure is None:
            return None
        pushed = self._with_persona(procedure.definition, ctx.definition.persona)
        if fsm_id_of(pushed) != ctx.child.fsm_id:
            return None
        run = _Run(procedure, config, converse=False)
        with self._lock:
            _bounded(self._runs, ctx.child.conversation_id, run, _MAX_TRACKED)
        return run

    async def on_ended(
        self,
        root_id: str,
        parent_id: str,
        child_id: str,
        state: State,
        record: Mapping[str, Any],
    ) -> None:
        with self._lock:
            run = self._runs.pop(child_id, None)
            self._depths.pop(child_id, None)
        if run is None:
            return
        procedure, library = run.procedure, self.synthesizer.library
        # an outage is no fault of the procedure and counts no run (D-061; the tool path counts
        # none either); an ok run that retried a call still counts
        if (
            library is not None
            and procedure.record is not None
            and (run.ok or not run.outage)
        ):
            try:
                library.record_run(
                    procedure.digest,
                    run.ok,
                    trust_after_successes=self.synthesizer.policy.trust_after_successes,
                )
            except SynthesisError:  # the run happened; a library fault must not undo it
                logger.warning(f"could not count the run of {procedure.digest[:12]}")
        self._emit(
            "popped",
            parent_id,
            digest=procedure.digest,
            status=record.get("status"),
            reason=record.get("reason"),
            step=None if run.converse else run.traced,
        )

    # --- the trace -----------------------------------------------------------------------

    def _trace_steps(self, conv: str, run: _Run, child: ChildView) -> None:
        """Trace the turns the engine counted on the child since the last event: one ``stepped``
        event carries the engine's lifetime count (turns run by hand count, and several turns
        between two settle rounds are one event); nothing is counted here."""
        taken = child.steps
        if taken is None or taken <= run.traced:
            return
        run.traced = taken
        self._emit(
            "stepped",
            conv,
            digest=run.procedure.digest,
            step=taken,
            **self._view(run, child),
        )

    def _view(
        self, run: _Run, child: ChildView, *, outcome: bool = False
    ) -> dict[str, Any]:
        """Value-free facts about the child for a trace event: its ``state``, the declared names
        present (``keys``), ``name:type`` for those an edge gate reads (``kinds``, types from
        ``TRACE_KINDS``) and, with ``outcome``, what the evaluator makes of its edges now. ``{}``
        without a sink, and on any failure (one WARNING): a trace never changes a run."""
        if self._sink is None:
            return {}
        try:
            definition, config, data = run.procedure.definition, run.config, child.data
            gated = gate_reads(definition.states.values())
            seen = sorted(
                key
                for key in gated | {*config.inputs, *config.return_keys}
                if not is_missing(data, key)
            )
            kinds = (
                f"{key}:{TRACE_KINDS.get(type(data[key]), TRACE_KIND_OTHER)}"
                for key in seen
                if key in gated
            )
            view = {
                "state": child.state_id,
                "keys": ",".join(seen),
                "kinds": ",".join(kinds),
            }
            if outcome:
                found = TransitionEvaluator().evaluate_transitions(
                    definition.states[child.state_id], FSMContext(data=dict(data))
                )
                view["outcome"] = found.result_type.value
            return view
        except Exception as e:
            logger.warning(f"could not read the child for a trace: {type(e).__name__}")
            return {}

    def _emit_obtained(self, conv: str, procedure: Procedure) -> None:
        self._emit(
            "obtained",
            conv,
            digest=procedure.digest,
            version=procedure.record.version if procedure.record else None,
            source=procedure.source,
            rounds=_round_string(procedure.rounds),
            seconds=procedure.seconds,
            calls=procedure.usage.calls,
            prompt_tokens=procedure.usage.prompt_tokens,
            completion_tokens=procedure.usage.completion_tokens,
        )

    def _emit(self, event: str, conv: str, **fields: Any) -> None:
        if self._sink is None:
            return
        try:
            self._sink(event, {"conv": conv, **fields})
        except Exception as e:
            logger.warning(f"trace_sink raised {type(e).__name__} on {event!r}")
