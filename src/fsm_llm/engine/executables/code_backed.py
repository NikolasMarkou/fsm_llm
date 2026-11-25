"""``CodeBackedExecutable``: the base of an executable whose body is Python code, not a document.

A built-in agent pattern or ``reasoning.solve`` has no envelope to load: it is a class that builds
and runs an object per run (roadmap E2.4; plan ace7a8c8 D-061). The package that owns the code
subclasses this base and registers its instances through the ``executable_sources`` namespace, so
the engine imports no L6 package. The base holds what every such executable shares and leaves one
hook.

Interface contract:

- ``CodeBackedExecutable(*, id, kind, impl, contract, requires=None, capabilities=(),
  optional_inputs=(), optional_callables=(), version="1")``: ``impl`` is the dotted name of the
  code (a claim of the code revision, never of behaviour); ``lifecycle`` is always ``builtin``.
  ``template_identity`` is ``canonical_digest({kind, id, version, impl, contract, fsm_llm version, requires})``: it
  never depends on a caller's task (Principle 14). ``requires.callables`` names the host objects a
  run REQUIRES, ``optional_callables`` those it reads when present, both looked up in
  ``RunContext.callables``; ``optional_inputs`` are input keys a caller may give beside the
  declared ones.
- ``instantiate(options=None, variables=None) -> CodeBackedInstance``: any option or variable is
  an ``InstantiationError`` (the body has none yet); the instance has ``instance_identity=None``
  (a code-backed body has no static instance, the known limit of D-061) and an empty record.
- ``run(inputs, *, context=None)`` is ``run_sync`` over ``arun``; ``arun`` is the default
  instance's run. The instance checks, in order and before any model call: an input key that is
  neither declared nor optional, or that has an internal prefix, is a ``ValueError``; a missing
  declared input is ``failed/inputs_missing``; an unresolved ``requires.callables`` name raises
  ``ExecutableError`` naming it; a set cancel token is ``aborted/cancelled``. Then it awaits the
  hook ``execute(inputs, objects, context, identity)`` (``objects`` is the resolved required
  names plus the present optional ones) and returns its ``RunResult``. What ``execute`` raises
  propagates unchanged.
- A subclass implements ``execute`` and maps its own errors to results (``stopped`` builds one).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from fsm_llm.__version__ import __version__ as _FSM_LLM_VERSION
from fsm_llm.aio import run_sync
from fsm_llm.core.codes import Lifecycle, Reason, Status
from fsm_llm.core.definitions.executable import (
    Capability,
    Contract,
    InstantiationError,
    Requires,
)
from fsm_llm.core.definitions.executable_instance import (
    canonical_digest,
    render_inputs,
)
from fsm_llm.core.results import Identity, RunResult
from fsm_llm.engine.context import RunCancelledError, RunContext

from .protocol import check_input_keys, require_callables

__all__ = ["CodeBackedExecutable", "CodeBackedInstance"]


class CodeBackedExecutable(ABC):
    """A template whose body is code. See the module docstring."""

    #: The ``RunContext`` budgets (``"max_steps"``, ``"max_seconds"``) this executable's
    #: ``execute`` refuses with a ``ValueError``. A caller that knows the executable before a run
    #: (``ExecutableStep``) refuses them at construction; the run still refuses them itself.
    refused_budgets: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        *,
        id: str,
        kind: str,
        impl: str,
        contract: Contract,
        requires: Requires | None = None,
        capabilities: Sequence[Capability] = (),
        optional_inputs: Sequence[str] = (),
        optional_callables: Sequence[str] = (),
        version: str = "1",
    ) -> None:
        self.id = id
        self.kind = kind
        self.version = version
        self.impl = impl
        self.contract = contract
        self.requires = requires if requires is not None else Requires()
        self.capabilities: list[Capability] = list(capabilities)
        self.lifecycle = Lifecycle.BUILTIN
        self.optional_inputs = tuple(optional_inputs)
        self.optional_callables = tuple(optional_callables)
        self.template_identity: str = canonical_digest(
            {
                "kind": kind,
                "id": id,
                "version": version,
                "impl": impl,
                "contract": contract.model_dump(mode="json"),
                "fsm_llm": _FSM_LLM_VERSION,
                "requires": self.requires.model_dump(mode="json"),
            }
        )

    @abstractmethod
    async def execute(
        self,
        inputs: Mapping[str, Any],
        objects: Mapping[str, Any],
        context: RunContext,
        identity: Identity,
    ) -> RunResult:
        """Run the code once. The checks of ``CodeBackedInstance.arun`` have passed."""

    def instantiate(
        self,
        options: Mapping[str, Any] | None = None,
        variables: Mapping[str, str] | None = None,
    ) -> CodeBackedInstance:
        """The runnable instance; a code-backed body takes no option and no variable."""
        if options or variables:
            raise InstantiationError(
                f"executable {self.id!r} is code-backed and takes no options or "
                "variables (a pattern file declares them)"
            )
        return CodeBackedInstance(self, {"options": {}, "variables": {}}, None)

    def run(
        self, inputs: Mapping[str, Any], *, context: RunContext | None = None
    ) -> RunResult:
        """The default instance's run, driven with no event loop."""
        return run_sync(self.arun(inputs, context=context))

    async def arun(
        self, inputs: Mapping[str, Any], *, context: RunContext | None = None
    ) -> RunResult:
        """The default instance's run."""
        return await self.instantiate().arun(inputs, context=context)

    def stopped(
        self,
        status: Status,
        reason: Reason | None,
        identity: Identity,
        **detail: Any,
    ) -> RunResult:
        """A result for a run that did not finish ``ok`` (``detail`` gains the kind)."""
        return RunResult(
            status=status,
            reason=reason,
            detail={"kind": self.kind, **detail},
            identity=identity,
        )

    def _check_input_keys(self, inputs: Mapping[str, Any]) -> None:
        check_input_keys(
            self.id, inputs, {*self.contract.inputs, *self.optional_inputs}
        )

    def _resolve_objects(self, context: RunContext) -> dict[str, Any]:
        registry = context.callables
        required = list(self.requires.callables)
        require_callables(self.id, required, registry)
        if registry is None:
            return {}
        objects = {name: registry.resolve(name) for name in required}
        for name in self.optional_callables:
            if name not in objects and name in registry:
                objects[name] = registry.resolve(name)
        return objects


@dataclass(frozen=True, eq=False)
class CodeBackedInstance:
    """One run's view of a code-backed executable. Never retained by a registry."""

    executable: CodeBackedExecutable
    record: Mapping[str, Any]
    instance_identity: str | None

    def run(
        self, inputs: Mapping[str, Any], *, context: RunContext | None = None
    ) -> RunResult:
        """The sync name of ``arun``, driven with no event loop."""
        return run_sync(self.arun(inputs, context=context))

    async def arun(
        self, inputs: Mapping[str, Any], *, context: RunContext | None = None
    ) -> RunResult:
        """Check, resolve, then await the hook (see the module docstring)."""
        owner = self.executable
        context = RunContext() if context is None else context
        inputs = render_inputs(owner.contract, inputs, optional=owner.optional_inputs)
        owner._check_input_keys(inputs)
        identity = Identity(
            template_identity=owner.template_identity,
            instance_identity=self.instance_identity,
        )
        missing = [key for key in owner.contract.inputs if key not in inputs]
        if missing:
            return owner.stopped(
                Status.FAILED, Reason.INPUTS_MISSING, identity, missing=missing
            )
        objects = owner._resolve_objects(context)
        try:
            context.cancel.raise_if_cancelled()
        except RunCancelledError:
            return owner.stopped(Status.ABORTED, Reason.CANCELLED, identity)
        return await owner.execute(inputs, objects, context, identity)
