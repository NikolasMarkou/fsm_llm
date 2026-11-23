"""The executable protocol and its errors (roadmap E2.1, contract C11; plan ace7a8c8 D-059).

An *executable* is anything that can be run by id under a ``RunContext``: an FSM-body
document (``fsm``, ``strategy``, ``procedure`` kinds, served by ``FsmBodyExecutable``) or, later,
a code-backed built-in. This module holds only the two Protocols, the errors and one default.

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
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from fsm_llm.core.codes import Lifecycle
from fsm_llm.core.definitions.executable import (
    Capability,
    Contract,
    ExecutableError,
    Requires,
)
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
