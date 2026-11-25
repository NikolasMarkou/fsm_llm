"""``WorkflowExecutable``: the adapter of the ``workflow`` kind (roadmap E5.1, plan 7d9c0955 D-008).

A ``workflow`` envelope (``core.definitions.executable_workflow``) is data; this module is what runs
it. Importing it registers the kind (``EXECUTABLE_KINDS``): ``import fsm_llm.workflows`` alone
registers nothing and loads none of this (D-004), so a host that wants a registry to load workflow
documents imports ``fsm_llm.workflows.executable`` by name.

Interface contract:

- ``WorkflowExecutable(envelope, *, pins=None)``: header fields are read through from the envelope
  (``id``, ``kind``, ``version``, ``contract``, ``requires``, ``capabilities``, ``lifecycle``);
  ``template_identity`` is the digest of the document. ``instantiate()`` returns a
  ``WorkflowBodyInstance`` (any option or variable is ``InstantiationError``: a workflow takes none);
  its ``instance_identity`` is the digest of the body tree. ``run`` is ``asyncio.run(arun(...))`` and
  raises ``RuntimeError`` (nothing started) when a loop is running in the calling thread: use
  ``arun`` there. It is never ``run_sync``: the engine needs a real loop.
- ``build_definition(callables=None, registry=None) -> WorkflowDefinition``: the document with the
  host objects bound. Every name a step holds resolves in ``callables`` (a ``NameRegistry`` of
  namespace ``callables``); an ``ExecutableStep`` gets ``registry`` (anything with ``get(id)``, none
  means every target is unknown, the step's failure route) and ``callables``. A document never
  carries those two fields (V3 refuses them): the host binds them here. Raises
  ``WorkflowValidationError`` (every failure at once) for a name the registry lacks.
- ``WorkflowBodyInstance.arun(inputs, *, context=None)``: checks, before any step, that every input
  key is declared and none has an internal prefix (``ValueError``), that declared inputs are present
  (``failed/inputs_missing``), that ``requires.callables`` are bound by ``context.callables``
  (``ExecutableError``) and that the cancel token is not set (``aborted/cancelled``). It then runs
  ONE ``WorkflowEngine(max_steps=context.max_steps or DEFAULT_EXECUTABLE_MAX_STEPS)`` per run, on
  the loop of the caller, with ``run_context_scope(context)`` around ``start`` (an ``ExecutableStep``
  derives its child run from it: one model, one event log, one token, one registry). Builds no
  approver and no grant; takes no model of its own.
- Results: COMPLETED is ``ok`` (outputs are the declared ``contract.outputs`` of the public context,
  ``{}`` when none is declared; a declared one absent is ``failed/return_keys_missing``);
  CANCELLED, or any instance that is not COMPLETED while the token is set, is ``aborted/cancelled``;
  a deadline (``max_seconds``) is ``aborted/budget_spent`` with ``budget`` ``max_seconds``; FAILED
  with as many successful steps as the engine limit is ``aborted/budget_spent`` with ``budget``
  ``max_steps`` (a failure right after the last allowed step reads the same: known limit); any other
  FAILED is ``failed`` (no reason, the error text capped in ``detail``); an instance still RUNNING or
  WAITING after ``start`` is ``failed`` with ``detail.workflow_status`` (the factory refuses pausing
  steps; this is the guard). ``handle`` is the instance id.
- Cancel: a ``step_started`` hook asks ``engine.cancel`` once the token is set and the engine honours
  it between steps (the step in flight finishes, later ones do not run). A run of steps that never
  yields to the loop (an ``async`` callable with no await) cannot be interrupted between steps. The
  engine is closed in ``finally``; an ``asyncio.CancelledError`` is never caught here.
- ``workflow_adapter(envelope, **options)``: the ``executable_kinds`` factory. It runs V2 in
  structure-only form (the loader over stub objects plus ``validate_structure()``) and refuses a
  pausing step (``WaitEventStep``, ``TimerStep``, a ``RetryStep`` around one) with
  ``ExecutableDocumentError(problems=)``, so a registry never serves a document the engine could
  park forever.
"""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from fsm_llm.core.codes import Lifecycle, Reason, Status
from fsm_llm.core.definitions.executable import (
    Capability,
    Contract,
    ExecutableDocumentError,
    InstantiationError,
    Requires,
    WorkflowEnvelope,
)
from fsm_llm.core.definitions.executable_instance import identity_of, render_inputs
from fsm_llm.core.definitions.executable_workflow import walk_steps, workflow_names
from fsm_llm.core.names import NameRegistry
from fsm_llm.core.results import Identity, RunResult
from fsm_llm.engine.context import RunContext
from fsm_llm.engine.executables.adapters import PinSink
from fsm_llm.engine.executables.namespaces import EXECUTABLE_KINDS
from fsm_llm.engine.executables.protocol import (
    DEFAULT_EXECUTABLE_MAX_STEPS,
    ExecutableNotFoundError,
    check_input_keys,
    require_callables,
)
from fsm_llm.logging import logger
from fsm_llm.text import capped_line

from .constants import CALLABLES_NAMESPACE
from .definitions import WorkflowDefinition
from .engine import WorkflowEngine
from .exceptions import WorkflowTimeoutError, WorkflowValidationError
from .loader import load_workflow
from .models import WorkflowInstance, WorkflowStatus
from .steps.helpers import run_context_scope

__all__ = ["WorkflowBodyInstance", "WorkflowExecutable", "workflow_adapter"]

# the names the host-bound fields of an ExecutableStep resolve under while a definition loads: a
# fixed pair, registered after the host's own names (a collision is a NameCollisionError)
_REGISTRY_NAME = "$run.registry"
_CALLABLES_NAME = "$run.callables"
_ERROR_CHARS = 300


class _NoRegistry:
    """The registry of a run that supplies none: every target is unknown (the step's failure route)."""

    def get(self, target: str) -> Any:
        raise ExecutableNotFoundError(
            f"no executable registry is bound to this run: {capped_line(target, 60)!r} is unknown"
        )


class _Stub:
    """One object that plays every role a document can name (a callable, an agent, an LLM)."""

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        return None

    run = generate = __call__


class WorkflowExecutable:
    """A template: a ``workflow`` envelope served by id. See the module docstring."""

    def __init__(
        self, envelope: WorkflowEnvelope, *, pins: PinSink | None = None
    ) -> None:
        self.envelope = envelope
        self.pins = pins
        body = {
            "initial_step_id": envelope.body.initial_step_id,
            "steps": envelope.body.steps,
        }
        both = identity_of(envelope, body)
        self.template_identity: str = both.template_identity or ""
        self._instance_identity: str = both.instance_identity or ""

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
    ) -> WorkflowBodyInstance:
        """The runnable instance; a workflow takes no option and no variable."""
        if options or variables:
            raise InstantiationError(
                f"workflow {self.id!r} takes no options or variables"
            )
        return WorkflowBodyInstance(
            self, {"options": {}, "variables": {}}, self._instance_identity
        )

    def run(
        self, inputs: Mapping[str, Any], *, context: RunContext | None = None
    ) -> RunResult:
        """The default instance's run: ``asyncio.run`` of ``arun`` (see the module docstring)."""
        return self.instantiate().run(inputs, context=context)

    async def arun(
        self, inputs: Mapping[str, Any], *, context: RunContext | None = None
    ) -> RunResult:
        """The default instance's run."""
        return await self.instantiate().arun(inputs, context=context)

    def build_definition(
        self,
        callables: NameRegistry[Any] | None = None,
        registry: Any = None,
    ) -> WorkflowDefinition:
        """The document with the host objects bound (see the module docstring)."""
        envelope = self.envelope
        steps = copy.deepcopy(envelope.body.steps)
        objects: NameRegistry[Any] = NameRegistry(CALLABLES_NAMESPACE)
        for name, obj in callables.items() if callables is not None else ():
            objects.register(name, obj)
        if not callable(getattr(registry, "get", None)):
            registry = _NoRegistry()
        objects.register(_REGISTRY_NAME, registry)
        if callables is not None:
            objects.register(_CALLABLES_NAME, callables)
        for step, _, _ in walk_steps(steps):
            if isinstance(step, dict) and step.get("type") == "ExecutableStep":
                step["registry"] = _REGISTRY_NAME
                step.pop("callables", None)
                if callables is not None:
                    step["callables"] = _CALLABLES_NAME
        return load_workflow(
            {
                "workflow_id": envelope.id,
                "name": envelope.name or envelope.id,
                "description": envelope.description,
                "initial_step_id": envelope.body.initial_step_id,
                "steps": steps,
            },
            objects=objects,
        )


@dataclass(frozen=True, eq=False)
class WorkflowBodyInstance:
    """One instantiated workflow: its record and its digest. Never retained."""

    executable: WorkflowExecutable
    record: Mapping[str, Any]
    instance_identity: str

    def run(
        self, inputs: Mapping[str, Any], *, context: RunContext | None = None
    ) -> RunResult:
        """``asyncio.run`` of ``arun``; refused inside a running loop, nothing started."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.arun(inputs, context=context))
        raise RuntimeError(
            f"workflow {self.executable.id!r}: run() cannot be called from a running event "
            "loop (the engine needs its own); await arun() instead"
        )

    # DECISION plan-2026-11-25T195353-7d9c0955/D-008
    # One WorkflowEngine per run on the loop of the caller; sync `run` is `asyncio.run` and refuses
    # inside a running loop (never `run_sync`: the engine needs a real loop). The cancel goes through
    # `engine.cancel` from a hook ONLY (never `task.cancel()` of the running chain, never a second lock,
    # D-017 of plan 5ed71cf2); a CancelledError is never caught and cleanup never replaces the outcome
    # (D-059 of plan ace7a8c8). Outputs are the declared contract, never the whole context. No approver,
    # no grant, no model. Do NOT catch BaseException, accept an internal-prefixed input, load the probe
    # with the real registry, or register the kind from `workflows/__init__.py`.
    async def arun(
        self, inputs: Mapping[str, Any], *, context: RunContext | None = None
    ) -> RunResult:
        """Run the workflow once (the one body; see the module docstring)."""
        owner = self.executable
        context = RunContext() if context is None else context
        inputs = render_inputs(owner.contract, inputs)
        check_input_keys(owner.id, inputs, owner.contract.inputs)
        identity = Identity(
            template_identity=owner.template_identity,
            instance_identity=self.instance_identity,
        )
        if missing := [key for key in owner.contract.inputs if key not in inputs]:
            return self._stopped(
                Status.FAILED, Reason.INPUTS_MISSING, identity, missing=missing
            )
        require_callables(owner.id, owner.requires.callables, context.callables)
        if context.cancel.cancelled:
            return self._stopped(Status.ABORTED, Reason.CANCELLED, identity)
        definition = owner.build_definition(context.callables, context.registry)
        engine = WorkflowEngine(
            max_steps=context.max_steps or DEFAULT_EXECUTABLE_MAX_STEPS
        )
        if owner.pins is not None:
            owner.pins.pin(owner)
        try:
            return await self._drive(engine, definition, inputs, context, identity)
        finally:
            try:
                await engine.close()
            except Exception as error:
                logger.warning(f"workflow run: closing the engine failed: {error!s}")
            if owner.pins is not None:
                owner.pins.unpin(owner)

    async def _drive(
        self,
        engine: WorkflowEngine,
        definition: WorkflowDefinition,
        inputs: Mapping[str, Any],
        context: RunContext,
        identity: Identity,
    ) -> RunResult:
        """Start the one instance and map how it ended."""
        steps_done = 0
        cancelling = False

        def hook(event: str, instance: WorkflowInstance, data: dict[str, Any]) -> Any:
            nonlocal steps_done, cancelling
            if event == "step_completed" and data.get("success"):
                steps_done += 1
            elif (
                event == "step_started" and context.cancel.cancelled and not cancelling
            ):
                cancelling = True
                return engine.cancel(instance.instance_id, "the run's token was set")
            return None

        engine.register_workflow(definition)
        engine.add_hook(hook)
        timed_out: WorkflowTimeoutError | None = None
        with run_context_scope(context):
            try:
                instance_id = await engine.start(
                    definition.workflow_id,
                    initial_context=dict(inputs),
                    max_seconds=context.max_seconds,
                )
            except WorkflowTimeoutError as error:
                timed_out, instance_id = error, error.instance_id or ""
        instance = engine.get_instance(instance_id)
        if instance is None:
            raise timed_out or RuntimeError("the engine lost the instance of the run")
        return self._ended(instance, identity, context, steps_done, engine, timed_out)

    def _ended(
        self,
        instance: WorkflowInstance,
        identity: Identity,
        context: RunContext,
        steps: int,
        engine: WorkflowEngine,
        timed_out: WorkflowTimeoutError | None,
    ) -> RunResult:
        status = instance.status
        detail: dict[str, Any] = {
            "steps": steps,
            "workflow_status": status.value,
            "current_step_id": instance.current_step_id,
        }
        if status is WorkflowStatus.COMPLETED:
            return self._completed(instance, identity, detail)
        if status is WorkflowStatus.CANCELLED or context.cancel.cancelled:
            return self._stopped(
                Status.ABORTED,
                Reason.CANCELLED,
                identity,
                instance.instance_id,
                **detail,
            )
        if timed_out is not None or (
            status is WorkflowStatus.FAILED and steps >= engine.max_steps
        ):
            budget = (
                ("max_seconds", context.max_seconds)
                if timed_out
                else (
                    "max_steps",
                    engine.max_steps,
                )
            )
            return self._stopped(
                Status.ABORTED,
                Reason.BUDGET_SPENT,
                identity,
                instance.instance_id,
                budget=budget[0],
                limit=budget[1],
                **detail,
            )
        if instance.error is not None:
            detail["error"] = capped_line(instance.error, _ERROR_CHARS)
        return self._stopped(
            Status.FAILED, None, identity, instance.instance_id, **detail
        )

    def _completed(
        self, instance: WorkflowInstance, identity: Identity, detail: dict[str, Any]
    ) -> RunResult:
        owner = self.executable
        source = instance.run_result().outputs
        declared = owner.contract.outputs
        if absent := [key for key in declared if key not in source]:
            return self._stopped(
                Status.FAILED,
                Reason.RETURN_KEYS_MISSING,
                identity,
                instance.instance_id,
                missing=absent,
                **detail,
            )
        return RunResult(
            status=Status.OK,
            outputs={key: source[key] for key in declared},
            detail={"kind": owner.kind, **detail},
            identity=identity,
            handle=instance.instance_id,
        )

    def _stopped(
        self,
        status: Status,
        reason: Reason | None,
        identity: Identity,
        handle: str | None = None,
        **detail: Any,
    ) -> RunResult:
        return RunResult(
            status=status,
            reason=reason,
            detail={"kind": self.executable.kind, **detail},
            identity=identity,
            handle=handle,
        )


def _structure_problems(executable: WorkflowExecutable) -> list[str]:
    """V2 for a document: the loader over stubs plus the structure check and the pausing rule.

    Contract: returns capped lines (empty when the structure is sound); never runs a step and never
    reads a host object (a name resolves to a stub, the registry is none).
    """
    probe: NameRegistry[Any] = NameRegistry(CALLABLES_NAMESPACE)
    names = workflow_names(executable.envelope.body.steps).callables
    for name in names:
        probe.register(name, _Stub())
    try:
        definition = executable.build_definition(probe)
    except WorkflowValidationError as error:
        return [
            capped_line(line, _ERROR_CHARS) for line in error.validation_errors[:10]
        ]
    return [
        f"step {capped_line(step_id, 60)!r}: a workflow run of a registry cannot pause, "
        "so a WaitEventStep, a TimerStep or a RetryStep around one is refused"
        for step_id, step in definition.steps.items()
        if definition._is_pausing_step(step)
    ]


def workflow_adapter(envelope: Any, **options: Any) -> WorkflowExecutable:
    """The ``executable_kinds`` factory of the ``workflow`` kind (V2 runs here)."""
    executable = WorkflowExecutable(envelope, **options)
    if problems := _structure_problems(executable):
        raise ExecutableDocumentError(
            "invalid workflow document: " + "; ".join(problems[:10]), problems=problems
        )
    return executable


# `override=True`: re-importing the module (a reload) replaces the factory, never collides.
EXECUTABLE_KINDS.register("workflow", workflow_adapter, override=True)
