"""The target resolver of the ``call`` kind: how a state names something to run by id.

Layer ``engine`` (plan baa37f40, step 10, D-021, roadmap E3.2). The engine layers below
``engine.executables`` and ``engine.context`` import neither (contract ``engine peers import no
executables or run context`` of ``.importlinter``), so the call executor and settle reach a
registry only through the ``TargetResolver`` Protocol defined here and injected as
``API(targets=)`` (or, for the nested ``API`` of ``FsmInstance.arun``, as
``RunContext.registry``). This module imports ``fsm_llm.core.definitions`` only.

Interface contract:

- ``TargetResolver`` (a runtime-checkable Protocol): ``resolve_target(target_id, *,
  conversation_id=None) -> ResolvedTarget | None``. ``None`` means "resolves to nothing" (an
  unknown id, a kind the frame mode cannot push, a body that does not instantiate): the one
  answer a caller maps to ``failed/unknown_target``. An implementation never raises for a
  miss, never loads a module by the id, and keeps no instance. ``conversation_id`` is the
  root conversation that asks, for a resolver that routes per conversation (the registry
  does not yet).
- ``ResolvedTarget(definition, template_identity, instance_identity, mode="frame", returns=None)``:
  frozen. ``definition`` is the ``FSMDefinition`` the stack pushes, unchanged (so its ``fsm_id``
  is the recomputed digest a restore checks). ``template_identity`` is the digest of the
  executable (``None`` when the resolver has none) and ``instance_identity`` the digest of this
  instantiation. ``mode`` has ONE value, ``TARGET_MODE_FRAME``; any other raises ``ValueError``.
  ``returns`` is the target's OWN output map (plan baa37f40 D-030): parent key to child key, the
  shape of ``CallStateConfig.returns``; a strategy's ``results``. ``None`` when the target
  declares none. A call that declares no ``returns`` merges every non-None child value of this
  map; a declared ``returns`` is the whole rule. Stored as a read-only copy; a map that is not
  ``str`` to ``str`` raises ``TypeError``.
- ``check_targets(value)``: ``None`` or an object with a callable ``resolve_target``, returned
  as is; anything else raises ``TypeError``. The one check of ``API(targets=)``,
  ``FSMManager(targets=)``, ``MessagePipeline(targets=)`` and ``RunContext(registry=)``.

Delegated mode (plan baa37f40 phase K7, D-049, roadmap E3.2 second half): a code-backed
``pattern`` or ``workflow`` has no FSM body to push, so it is not a ``ResolvedTarget``.

- ``DelegatedTarget(target_id, kind, executable, template_identity, instance_identity, outputs)``:
  frozen, the mode is the constant ``TARGET_MODE_DELEGATED`` (not a parameter). ``executable`` is
  the code-backed executable (typed ``Any``: this layer imports no ``engine.executables``),
  ``instance_identity`` is ``None`` (a code-backed body has no static instance) and ``outputs``
  the output names its contract declares.
- ``RunLauncher`` (a runtime-checkable Protocol, six methods; one object may be a
  ``TargetResolver`` too, behind the one ``API(targets=)``): ``resolve_run(target_id, *,
  input_keys=(), max_steps=None, conversation_id=None) -> DelegatedTarget | None`` (``None``
  is every refusal, mapped to ``failed/unknown_target``); ``claim(handle, *, conversation_id)
  -> bool`` reserves a minted handle for a conversation in a bounded set (``False``: the host is
  full, mapped to ``failed/push_failed``); ``start(handle, target, inputs, *, conversation_id,
  max_seconds=None, announce=None) -> bool`` consumes the reservation, calls ``announce()`` (the
  ``run/start`` write) and only then starts the run, returning ``False`` when the handle holds no
  reservation of that conversation (a forged record starts nothing); ``poll(handle, *,
  conversation_id) -> RunPoll | None`` (``None``: the host never saw the handle for that
  conversation); ``cancel(handle, *, conversation_id=None) -> bool``; ``cancel_for(conversation_id)
  -> int`` (every run and reservation of one conversation). A launcher never raises for an
  unknown handle, never nests a manager lock and never calls back into the ``API`` it serves.
- ``RunPoll(finished, result)``: frozen; ``result`` is the ``RunResult`` of a finished run, else
  ``None``.
- ``DEFAULT_DELEGATED_MAX_SECONDS`` (600.0), ``MAX_DELEGATED_MAX_SECONDS`` (3600.0) and
  ``delegated_seconds(max_seconds)``: the wall-clock bound of a delegated call (the default when the
  call declares none, cut to the cap), the one rule of the executor's ``deadline`` and of the limit
  settle hands the launcher. ``MAX_DELEGATED_RETURN_CHARS`` (8000): the most characters of one value a
  finished run merges into the parent (D-051).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from fsm_llm.core.definitions import FSMDefinition

if TYPE_CHECKING:  # the result envelope loads on request, never with `import fsm_llm`
    from fsm_llm.core.results import RunResult

__all__ = [
    "DEFAULT_DELEGATED_MAX_SECONDS",
    "MAX_DELEGATED_MAX_SECONDS",
    "MAX_DELEGATED_RETURN_CHARS",
    "TARGET_MODE_DELEGATED",
    "TARGET_MODE_FRAME",
    "DelegatedTarget",
    "ResolvedTarget",
    "RunLauncher",
    "RunPoll",
    "TargetResolver",
    "check_targets",
    "delegated_seconds",
]

#: The one mode of a resolved target: a child FSM pushed as a frame of the parent's stack.
TARGET_MODE_FRAME = "frame"
#: The mode of a delegated target: a code-backed run on a host thread, polled by the parent.
TARGET_MODE_DELEGATED = "delegated"

#: The wall-clock bound of a delegated call that declares no ``max_seconds`` (owner-visible; plan
#: baa37f40 D-051): a delegated call is always bounded, because the parent polls a run it cannot
#: interrupt.
DEFAULT_DELEGATED_MAX_SECONDS = 600.0
#: The cap of ANY delegated ``max_seconds``: a longer one is cut to this (owner-visible, D-051).
MAX_DELEGATED_MAX_SECONDS = 3600.0
#: The most characters of one value a finished delegated run merges into the parent: a text is
#: cut at it (and ends with a marker), a container over it becomes a one-line marker (D-051).
MAX_DELEGATED_RETURN_CHARS = 8_000


def delegated_seconds(max_seconds: float | None) -> float:
    """The wall-clock bound of a delegated call: ``max_seconds`` or the default, cut to the cap.

    Contract: ``None`` gives ``DEFAULT_DELEGATED_MAX_SECONDS``; a number above
    ``MAX_DELEGATED_MAX_SECONDS`` gives the cap. The one rule of the executor (the ``deadline``
    of the record) and of settle (the limit handed to the launcher). Never raises for a number.
    """
    seconds = DEFAULT_DELEGATED_MAX_SECONDS if max_seconds is None else max_seconds
    return min(float(seconds), MAX_DELEGATED_MAX_SECONDS)


@dataclass(frozen=True)
class ResolvedTarget:
    """What a resolver answers for a target id: the definition to push and its identities."""

    definition: FSMDefinition
    template_identity: str | None
    instance_identity: str | None
    mode: str = TARGET_MODE_FRAME
    returns: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if self.mode != TARGET_MODE_FRAME:
            raise ValueError(
                f"a resolved target has the one mode {TARGET_MODE_FRAME!r}, got {self.mode!r}"
            )
        if self.returns is not None:
            if not all(
                isinstance(parent, str) and isinstance(child, str)
                for parent, child in self.returns.items()
            ):
                raise TypeError("ResolvedTarget.returns must map str keys to str keys")
            object.__setattr__(self, "returns", MappingProxyType(dict(self.returns)))


@runtime_checkable
class TargetResolver(Protocol):
    """Resolves a target id to the definition a call pushes. See the module docstring."""

    def resolve_target(
        self, target_id: str, *, conversation_id: str | None = None
    ) -> ResolvedTarget | None: ...


# DECISION plan-2026-11-23T122249-baa37f40/D-049
# A delegated target is a SEPARATE value type, not a second mode of `ResolvedTarget`. A code-backed
# pattern or workflow has no `FSMDefinition` to push, and `ResolvedTarget.definition` is read
# without a guard at every frame site (the push plan of settle, the synthesis executor): do NOT make
# it Optional and do NOT put a placeholder definition in it, a frame consumer must never hold a
# target with nothing to push. The one answer of the frame resolver stays `ResolvedTarget | None`;
# `RunLauncher.resolve_run` is the delegated twin, so no existing `TargetResolver` is widened.
@dataclass(frozen=True)
class DelegatedTarget:
    """What a launcher answers for a target id: the executable to run and its identities."""

    target_id: str
    kind: str
    executable: Any
    template_identity: str
    instance_identity: str | None
    outputs: tuple[str, ...] = ()
    mode: str = field(default=TARGET_MODE_DELEGATED, init=False)


@dataclass(frozen=True)
class RunPoll:
    """What a poll of a launched run reports: finished or not, and the result once it is."""

    finished: bool
    result: RunResult | None = None


@runtime_checkable
class RunLauncher(Protocol):
    """Starts and watches delegated runs. See the module docstring."""

    def resolve_run(
        self,
        target_id: str,
        *,
        input_keys: Sequence[str] = (),
        max_steps: int | None = None,
        conversation_id: str | None = None,
    ) -> DelegatedTarget | None: ...

    def claim(self, handle: str, *, conversation_id: str) -> bool: ...

    def start(
        self,
        handle: str,
        target: DelegatedTarget,
        inputs: Mapping[str, Any],
        *,
        conversation_id: str,
        max_seconds: float | None = None,
        announce: Callable[[], None] | None = None,
    ) -> bool: ...

    def poll(self, handle: str, *, conversation_id: str) -> RunPoll | None: ...

    def cancel(self, handle: str, *, conversation_id: str | None = None) -> bool: ...

    def cancel_for(self, conversation_id: str) -> int: ...


def check_targets(targets: Any) -> TargetResolver | None:
    """Return *targets* when it is ``None`` or a resolver, else raise ``TypeError``."""
    if targets is None:
        return None
    if not callable(getattr(targets, "resolve_target", None)):
        raise TypeError(
            "targets must provide resolve_target(target_id, *, conversation_id=None) "
            f"(a TargetResolver, for example an ExecutableRegistry), got {type(targets).__name__}"
        )
    resolver: TargetResolver = targets
    return resolver
