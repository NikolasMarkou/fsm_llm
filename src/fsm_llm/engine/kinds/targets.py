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
- ``ResolvedTarget(definition, template_identity, instance_identity, mode="frame")``: frozen.
  ``definition`` is the ``FSMDefinition`` the stack pushes, unchanged (so its ``fsm_id`` is the
  recomputed digest a restore checks). ``template_identity`` is the digest of the executable
  (``None`` when the resolver has none) and ``instance_identity`` the digest of this
  instantiation. ``mode`` has ONE value, ``TARGET_MODE_FRAME``; any other raises ``ValueError``.
- ``check_targets(value)``: ``None`` or an object with a callable ``resolve_target``, returned
  as is; anything else raises ``TypeError``. The one check of ``API(targets=)``,
  ``FSMManager(targets=)``, ``MessagePipeline(targets=)`` and ``RunContext(registry=)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from fsm_llm.core.definitions import FSMDefinition

__all__ = ["TARGET_MODE_FRAME", "ResolvedTarget", "TargetResolver", "check_targets"]

#: The one mode of a resolved target: a child FSM pushed as a frame of the parent's stack.
TARGET_MODE_FRAME = "frame"


@dataclass(frozen=True)
class ResolvedTarget:
    """What a resolver answers for a target id: the definition to push and its identities."""

    definition: FSMDefinition
    template_identity: str | None
    instance_identity: str | None
    mode: str = TARGET_MODE_FRAME

    def __post_init__(self) -> None:
        if self.mode != TARGET_MODE_FRAME:
            raise ValueError(
                f"a resolved target has the one mode {TARGET_MODE_FRAME!r}, got {self.mode!r}"
            )


@runtime_checkable
class TargetResolver(Protocol):
    """Resolves a target id to the definition a call pushes. See the module docstring."""

    def resolve_target(
        self, target_id: str, *, conversation_id: str | None = None
    ) -> ResolvedTarget | None: ...


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
