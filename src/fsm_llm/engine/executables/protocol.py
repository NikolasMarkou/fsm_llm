"""The executable protocol and its errors (roadmap E2.1, contract C11; plan ace7a8c8 D-059).

An *executable* is anything that can be run by id under a ``RunContext``: an FSM-body
document (``fsm``, ``strategy``, ``procedure`` kinds, served by ``FsmBodyExecutable``) or, later,
a code-backed built-in. This module holds the two Protocols, the errors, one default and the two
checks every adapter makes before a run (``check_input_keys``, ``require_callables``).

Interface contract:

- ``Executable`` (a runtime-checkable Protocol): ``id``, ``kind``, ``version``, ``contract``,
  ``requires``, ``capabilities``, ``lifecycle``, ``template_identity`` (the digest of the
  canonical envelope, ``None`` is never allowed here: a code-backed built-in states its own);
  ``instantiate(options=None, variables=None) -> Instance``; ``run(inputs, *, context)`` (the
  default instance's run, a sync name) and ``arun(inputs, *, context)`` (the one body).
- ``Instance`` (a Protocol): ``record`` (the instantiation record, ``{"options", "variables"}``),
  ``instance_identity`` (``None`` when the body has no static instance), ``run`` and ``arun``
  as above. An instance is built per run and is never retained by a registry.
- Both ``run`` and ``arun`` return a ``fsm_llm.core.results.RunResult`` and raise
  ``ValueError`` for an input key the contract does not declare (before any model call).
- Errors, all ``ExecutableError`` (``FSMError``, the root of ``core.definitions.executable``):
  ``ExecutableNotFoundError`` (no such id or version), ``ExecutableCollisionError`` (a duplicate
  ``(id, version)`` or alias clash), ``ExecutableKindError`` (a kind with no adapter),
  ``ExecutableEvictedError`` (a template identity is neither current nor retained; names it).
  ``resume`` is carried to phase S.
- ``check_input_keys(executable_id, inputs, allowed)``: ``ValueError`` for an input key not in
  *allowed* and for any key with an internal prefix (a caller input is data, never an engine key);
  ``require_callables(executable_id, required, registry)``: ``ExecutableError`` naming the
  *required* names a ``RunContext.callables`` registry (or ``None``) does not hold. Both run before
  any model call and are the one copy of the rule for the body, code-backed and workflow adapters.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from fsm_llm.constants import has_internal_prefix
from fsm_llm.core.codes import Lifecycle
from fsm_llm.core.definitions.executable import (
    Capability,
    Contract,
    ExecutableError,
    Requires,
)
from fsm_llm.core.names import NameRegistry
from fsm_llm.core.results import RunResult
from fsm_llm.engine.context import RunContext

__all__ = [
    "DEFAULT_EXECUTABLE_MAX_STEPS",
    "Executable",
    "ExecutableCollisionError",
    "ExecutableError",
    "ExecutableEvictedError",
    "ExecutableKindError",
    "ExecutableNotFoundError",
    "Instance",
    "check_input_keys",
    "require_callables",
]

#: The step budget of a run whose ``RunContext`` sets no ``max_steps``: a run is always bounded.
DEFAULT_EXECUTABLE_MAX_STEPS = 100


class ExecutableNotFoundError(ExecutableError):
    """No executable has the id (or the id has no such version)."""


class ExecutableCollisionError(ExecutableError):
    """A second executable claims an ``(id, version)`` that is taken."""


class ExecutableKindError(ExecutableError):
    """The ``kind`` of a document has no adapter in the ``executable_kinds`` namespace."""


class ExecutableEvictedError(ExecutableError):
    """A template identity is neither current nor retained (never loaded, or evicted)."""


@runtime_checkable
class Instance(Protocol):
    """One instantiated executable: a body with its variables filled, runnable."""

    record: Mapping[str, Any]
    instance_identity: str | None

    def run(
        self, inputs: Mapping[str, Any], *, context: RunContext | None = None
    ) -> RunResult: ...

    async def arun(
        self, inputs: Mapping[str, Any], *, context: RunContext | None = None
    ) -> RunResult: ...


@runtime_checkable
class Executable(Protocol):
    """Something a registry can serve and a runner can run. See the module docstring."""

    id: str
    kind: str
    version: str
    contract: Contract
    requires: Requires
    capabilities: Sequence[Capability]
    lifecycle: Lifecycle
    template_identity: str

    def instantiate(
        self,
        options: Mapping[str, Any] | None = None,
        variables: Mapping[str, str] | None = None,
    ) -> Instance: ...

    def run(
        self, inputs: Mapping[str, Any], *, context: RunContext | None = None
    ) -> RunResult: ...

    async def arun(
        self, inputs: Mapping[str, Any], *, context: RunContext | None = None
    ) -> RunResult: ...


def check_input_keys(
    executable_id: str, inputs: Mapping[str, Any], allowed: Collection[str]
) -> None:
    """Refuse an input key that is not in *allowed* or that has an internal prefix (``ValueError``)."""
    if undeclared := sorted(str(k) for k in inputs if k not in allowed):
        raise ValueError(
            f"executable {executable_id!r} declares no input {undeclared}; "
            f"declared: {sorted(allowed)}"
        )
    if internal := sorted(str(k) for k in inputs if has_internal_prefix(str(k))):
        raise ValueError(
            f"inputs {internal} use a reserved prefix: a caller input is data, "
            "never an engine key"
        )


def require_callables(
    executable_id: str, required: Sequence[str], registry: NameRegistry[Any] | None
) -> None:
    """``ExecutableError`` naming the *required* names that *registry* (or ``None``) lacks."""
    absent = [n for n in required if registry is None or n not in registry]
    if absent:
        raise ExecutableError(
            f"executable {executable_id!r} needs the callables {absent}, which the run "
            "context does not supply (RunContext.callables)",
            details={"executable": executable_id, "unresolved": absent},
        )
