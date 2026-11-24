"""``FsmBodyExecutable``: one adapter for the ``fsm``, ``strategy`` and ``procedure`` kinds.

All three kinds run an ``FSMDefinition`` the same way (roadmap 6.3, plan ace7a8c8 D-059): the
adapter owns its own ``API`` and conversation tree, drives the body with the message-free
step loop and reports a ``RunResult`` whose identity carries both digests.

Interface contract:

- ``FsmBodyExecutable(envelope, *, setup=None, pins=None)``: ``envelope`` is a parsed
  ``FsmEnvelope``, ``StrategyEnvelope`` or ``ProcedureEnvelope``; ``setup(api)`` is an optional
  seam called once per run on the fresh ``API`` before the conversation starts (the strategy
  FSMs that need handlers register them there); ``pins`` is anything with ``pin(executable)`` and
  ``unpin(executable)`` (the registry passes itself; used alone nothing is pinned). Header
  fields are read through (``id``, ``kind``, ``version``, ``contract``, ``requires``,
  ``capabilities``, ``lifecycle``); ``template_identity`` is computed once.
- ``instantiate(options=None, variables=None) -> FsmInstance``: core ``instantiate_envelope``
  (``InstantiationError`` for a missing, undeclared or non-text variable, any option, or a
  body that does not load); the instance holds the definition and the instantiation record and
  is never kept by the executable. ``rebuild(record, expected_instance_identity)`` re-instantiates
  from a logged record and raises ``InstanceIdentityMismatchError`` when the digest differs.
  ``run(inputs, *, context=None)`` and ``arun`` instantiate with no variables, then run.
- ``FsmInstance.arun(inputs, *, context=None)`` is the one body; ``FsmInstance.run`` is
  ``run_sync`` over it. ``context`` defaults to a bare ``RunContext()``. Before any model call:
  an input key the contract does not declare, or one with an internal prefix, is a
  ``ValueError`` (after ``contract.input_templates`` are rendered from the arguments, D-033); a missing declared input returns ``failed/inputs_missing``. Then the run takes
  the interface of the context (or the API's own model resolution), its event log, the budget
  (``max_steps`` defaults to ``DEFAULT_EXECUTABLE_MAX_STEPS``) and checks the cancel token before
  every step. The run's own ``API`` is given ``targets=context.registry``, so a body holding a
  ``call`` state resolves its target like a bare ``API(targets=)`` does (and, with no registry in
  the context, is refused with ``MissingExecutorError`` when the run starts). Results: ``ok``; ``aborted/budget_spent`` (``RunBudgetExceededError``);
  ``aborted/cancelled`` (the token); ``failed/provider_error`` (``LLMResponseError``);
  ``failed/return_keys_missing`` (a declared output absent at the terminal state). Anything else
  propagates and an ``asyncio.CancelledError`` is never caught: the engine rolls the turn back
  and the ``finally`` ends the conversation once. ``handle`` is the conversation id.
- Known limit: a body that needs a user message runs message-free steps until the step budget is
  spent and ends ``aborted/budget_spent`` (no stall detection).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from fsm_llm.aio import run_sync
from fsm_llm.constants import has_internal_prefix
from fsm_llm.core.codes import Lifecycle, Reason, Status
from fsm_llm.core.definitions import (
    FSMDefinition,
    LLMResponseError,
    RunBudgetExceededError,
)
from fsm_llm.core.definitions.executable import (
    Capability,
    Contract,
    ExecutableEnvelope,
    Requires,
    StrategyEnvelope,
)
from fsm_llm.core.definitions.executable_instance import (
    identity_of,
    instantiate_envelope,
    rebuild_instance,
    render_inputs,
)
from fsm_llm.core.definitions.fsm import definition_digest
from fsm_llm.core.results import Identity, RunResult
from fsm_llm.engine.api import API
from fsm_llm.engine.context import RunCancelledError, RunContext
from fsm_llm.logging import logger

from .protocol import DEFAULT_EXECUTABLE_MAX_STEPS

__all__ = [
    "FsmBodyExecutable",
    "FsmInstance",
    "fsm_adapter",
    "procedure_adapter",
    "strategy_adapter",
]


class PinSink(Protocol):
    """What an executable pins itself on for the length of a run (the registry)."""

    def pin(self, executable: Any) -> None: ...

    def unpin(self, executable: Any) -> None: ...


class FsmBodyExecutable:
    """A template: an envelope with an FSM body, served by id. See the module docstring."""

    def __init__(
        self,
        envelope: ExecutableEnvelope,
        *,
        setup: Callable[[API], None] | None = None,
        pins: PinSink | None = None,
    ) -> None:
        self.envelope = envelope
        self.setup = setup
        self.pins = pins
        self.template_identity: str = identity_of(envelope).template_identity or ""

    @property
    def id(self) -> str:
        return self.envelope.id

    @property
    def kind(self) -> str:
        return self.envelope.kind

    @property
    def version(self) -> str:
        return self.envelope.version

    @property
    def contract(self) -> Contract:
        return self.envelope.contract

    @property
    def requires(self) -> Requires:
        return self.envelope.requires

    @property
    def capabilities(self) -> Sequence[Capability]:
        return self.envelope.capabilities

    @property
    def lifecycle(self) -> Lifecycle:
        return self.envelope.lifecycle

    def instantiate(
        self,
        options: Mapping[str, Any] | None = None,
        variables: Mapping[str, str] | None = None,
    ) -> FsmInstance:
        """A runnable instance with *variables* filled (see the module docstring)."""
        definition, record = instantiate_envelope(
            self.envelope, options=options, variables=variables
        )
        return FsmInstance(self, definition, record, definition_digest(definition))

    def rebuild(
        self, record: Mapping[str, Any], expected_instance_identity: str
    ) -> FsmInstance:
        """The instance of a logged *record*; its digest must equal the logged identity."""
        definition = rebuild_instance(self.envelope, record, expected_instance_identity)
        return FsmInstance(
            self,
            definition,
            {
                "options": dict(record.get("options") or {}),
                "variables": dict(record.get("variables") or {}),
            },
            expected_instance_identity,
        )

    def run(
        self, inputs: Mapping[str, Any], *, context: RunContext | None = None
    ) -> RunResult:
        """The default instance's run, driven with no event loop."""
        return run_sync(self.arun(inputs, context=context))

    async def arun(
        self, inputs: Mapping[str, Any], *, context: RunContext | None = None
    ) -> RunResult:
        """The default instance's run (variables are not filled: see ``instantiate``)."""
        return await self.instantiate().arun(inputs, context=context)


@dataclass(frozen=True, eq=False)
class FsmInstance:
    """One instantiated body: the definition, its record and its digest. Never retained."""

    executable: FsmBodyExecutable
    definition: FSMDefinition
    record: Mapping[str, Any]
    instance_identity: str

    def run(
        self, inputs: Mapping[str, Any], *, context: RunContext | None = None
    ) -> RunResult:
        """The sync name of ``arun``, driven with no event loop."""
        return run_sync(self.arun(inputs, context=context))

    # DECISION plan-2026-11-20T055240-ace7a8c8/D-059
    # The run owns its API and its conversation: the conversation is ended and the API closed
    # exactly once in the `finally` (`_release`), a cancel (`CancelledError`) is never swallowed
    # and rolls back whole, and an error in cleanup never replaces the outcome. Do NOT end the
    # conversation outside `finally`, do NOT catch `BaseException` or `CancelledError` here, and
    # do NOT let `_release` raise.
    async def arun(
        self, inputs: Mapping[str, Any], *, context: RunContext | None = None
    ) -> RunResult:
        """Run the body once (the one body; see the module docstring)."""
        owner = self.executable
        context = RunContext() if context is None else context
        inputs = render_inputs(owner.contract, inputs)
        self._check_input_keys(inputs)
        identity = Identity(
            template_identity=owner.template_identity,
            instance_identity=self.instance_identity,
        )
        missing = [key for key in owner.contract.inputs if key not in inputs]
        if missing:
            return RunResult(
                status=Status.FAILED,
                reason=Reason.INPUTS_MISSING,
                detail={"kind": owner.kind, "missing": missing},
                identity=identity,
            )
        if owner.pins is not None:
            owner.pins.pin(owner)
        api: API | None = None
        conversation_id: str | None = None
        try:
            api = API.from_definition(
                self.definition,
                llm_interface=context.resolve_llm(),
                event_log=context.event_log,
                targets=context.registry,
            )
            if owner.setup is not None:
                owner.setup(api)
            context.cancel.raise_if_cancelled()
            conversation_id, greeting = await api.astart_conversation(dict(inputs))
            results = await api.arun_until_terminal(
                conversation_id,
                max_steps=context.max_steps or DEFAULT_EXECUTABLE_MAX_STEPS,
                max_seconds=context.max_seconds,
                before_step=lambda _step: context.cancel.raise_if_cancelled(),
            )
            data = api.get_data(conversation_id)
            final_state = api.get_current_state(conversation_id)
            spoken = [r.response for r in results if r.response]
            return self._finished(
                data,
                answer=(spoken[-1] if spoken else greeting) or None,
                steps=len(results),
                final_state=final_state,
                handle=conversation_id,
                identity=identity,
            )
        except RunBudgetExceededError as error:
            return self._stopped(
                Status.ABORTED,
                Reason.BUDGET_SPENT,
                conversation_id,
                identity,
                budget=error.budget,
                limit=error.limit,
                steps_done=error.steps_done,
            )
        except RunCancelledError:
            return self._stopped(
                Status.ABORTED, Reason.CANCELLED, conversation_id, identity
            )
        except LLMResponseError as error:
            # the type only: a provider message can carry text the log should not
            return self._stopped(
                Status.FAILED,
                Reason.PROVIDER_ERROR,
                conversation_id,
                identity,
                error=type(error).__name__,
            )
        finally:
            _release(api, conversation_id)
            if owner.pins is not None:
                owner.pins.unpin(owner)

    def _check_input_keys(self, inputs: Mapping[str, Any]) -> None:
        declared = set(self.executable.contract.inputs)
        if undeclared := sorted(str(k) for k in inputs if k not in declared):
            raise ValueError(
                f"executable {self.executable.id!r} declares no input {undeclared}; "
                f"declared: {sorted(declared)}"
            )
        if internal := sorted(str(k) for k in inputs if has_internal_prefix(str(k))):
            raise ValueError(
                f"inputs {internal} use a reserved prefix: a caller input is data, "
                "never an engine key"
            )

    def _stopped(
        self,
        status: Status,
        reason: Reason,
        handle: str | None,
        identity: Identity,
        **detail: Any,
    ) -> RunResult:
        return RunResult(
            status=status,
            reason=reason,
            detail={"kind": self.executable.kind, **detail},
            identity=identity,
            handle=handle,
        )

    def _finished(
        self,
        data: Mapping[str, Any],
        *,
        answer: str | None,
        steps: int,
        final_state: str,
        handle: str,
        identity: Identity,
    ) -> RunResult:
        owner = self.executable
        envelope = owner.envelope
        detail = {"kind": owner.kind, "steps": steps, "final_state": final_state}
        if isinstance(envelope, StrategyEnvelope):
            strategy = envelope.body.strategy
            source = {
                orchestrator: data[key]
                for orchestrator, key in strategy.results.items()
                if data.get(key) is not None
            }
            solution = source.get(strategy.solution_key)
            answer = solution if isinstance(solution, str) and solution else answer
        else:
            source = dict(data)
        declared = owner.contract.outputs
        if absent := [key for key in declared if key not in source]:
            return RunResult(
                status=Status.FAILED,
                reason=Reason.RETURN_KEYS_MISSING,
                answer=answer,
                detail={**detail, "missing": absent},
                identity=identity,
                handle=handle,
            )
        outputs = (
            {key: source[key] for key in declared}
            if declared
            else (source if isinstance(envelope, StrategyEnvelope) else {})
        )
        return RunResult(
            status=Status.OK,
            outputs=outputs,
            answer=answer,
            detail=detail,
            identity=identity,
            handle=handle,
        )


def _release(api: API | None, conversation_id: str | None) -> None:
    """End the run's conversation once and close its API; a failure here is logged, never raised.

    Contract: called from the ``finally`` of ``FsmInstance.arun`` only, so an error in cleanup
    must not replace the outcome (a result, or the exception in flight, a cancel included).
    """
    if api is None:
        return
    try:
        if conversation_id is not None:
            api.end_conversation(conversation_id)
    except Exception as error:
        logger.warning(
            f"executable run: ending conversation {conversation_id} failed: {error!s}"
        )
    try:
        api.close()
    except Exception as error:
        logger.warning(f"executable run: closing the API failed: {error!s}")


def fsm_adapter(envelope: ExecutableEnvelope, **options: Any) -> FsmBodyExecutable:
    """The ``executable_kinds`` factory of the ``fsm`` kind."""
    return FsmBodyExecutable(envelope, **options)


def strategy_adapter(envelope: ExecutableEnvelope, **options: Any) -> FsmBodyExecutable:
    """The ``executable_kinds`` factory of the ``strategy`` kind."""
    return FsmBodyExecutable(envelope, **options)


def procedure_adapter(
    envelope: ExecutableEnvelope, **options: Any
) -> FsmBodyExecutable:
    """The ``executable_kinds`` factory of the ``procedure`` kind."""
    return FsmBodyExecutable(envelope, **options)
