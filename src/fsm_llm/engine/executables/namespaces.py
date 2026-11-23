"""The two executable namespaces (roadmap E2.3, contract C10; plan ace7a8c8 D-059).

``EXECUTABLE_KINDS`` (entry-point group ``fsm_llm.executable_kinds``) maps a document ``kind`` to
the adapter factory that turns a parsed envelope into an ``Executable``: ``fsm``, ``strategy`` and
``procedure`` are registered here, each by its own function (a ``NameRegistry`` refuses one object
under two names). ``EXECUTABLE_SOURCES`` (group ``fsm_llm.executable_sources``) maps a source name
to a zero-argument factory returning an iterable of ``Executable`` objects or envelope documents:
how a package such as ``fsm_llm.agents`` declares its built-ins without the engine importing it.

Interface contract:

- ``adapter_for(kind)`` returns the factory ``factory(envelope, **options) -> Executable`` of
  the kind, loading the entry-point group on a miss, or raises ``ExecutableKindError`` naming the
  kind and the known ones.
- Neither namespace has an entry in ``pyproject.toml`` yet (E2.3 proper): a source registers by
  an explicit ``EXECUTABLE_SOURCES.register(name, factory)`` call. Both registries are plain
  module constants, listed by the namespace test of ``tests/test_fsm_llm/test_names.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from fsm_llm.core.names import NameNotFoundError, NameRegistry

from .adapters import fsm_adapter, procedure_adapter, strategy_adapter
from .protocol import ExecutableKindError

__all__ = ["EXECUTABLE_KINDS", "EXECUTABLE_SOURCES", "adapter_for"]

EXECUTABLE_KINDS: NameRegistry[Callable[..., Any]] = NameRegistry(
    "executable_kinds", entry_point_group="fsm_llm.executable_kinds"
)
EXECUTABLE_KINDS.register("fsm", fsm_adapter)
EXECUTABLE_KINDS.register("strategy", strategy_adapter)
EXECUTABLE_KINDS.register("procedure", procedure_adapter)

EXECUTABLE_SOURCES: NameRegistry[Callable[[], Iterable[Any]]] = NameRegistry(
    "executable_sources", entry_point_group="fsm_llm.executable_sources"
)


def adapter_for(
    kind: str, kinds: NameRegistry[Callable[..., Any]] | None = None
) -> Callable[..., Any]:
    """The adapter factory of *kind* (``kinds`` defaults to ``EXECUTABLE_KINDS``)."""
    registry = EXECUTABLE_KINDS if kinds is None else kinds
    try:
        return registry.resolve(kind)
    except NameNotFoundError:
        raise ExecutableKindError(
            f"no adapter for the executable kind {kind!r}; known: {registry.list()}"
        ) from None
