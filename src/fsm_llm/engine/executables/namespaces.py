"""The executable namespaces (roadmap E2.3, contract C10; plan ace7a8c8 D-059; plan a555dcfe D-013).

``EXECUTABLE_KINDS`` (entry-point group ``fsm_llm.executable_kinds``) maps a document ``kind`` to
the adapter factory that turns a parsed envelope into an ``Executable``: ``fsm``, ``strategy`` and
``procedure`` are registered here, each by its own function (a ``NameRegistry`` refuses one object
under two names). ``EXECUTABLE_SOURCES`` (group ``fsm_llm.executable_sources``) maps a source name
to a zero-argument factory returning an iterable of ``Executable`` objects or envelope documents:
how a package such as ``fsm_llm.agents`` declares its built-ins without the engine importing it.
``HANDLERS`` (group ``fsm_llm.handlers``) maps a handler name a pattern file binds to its FACTORY
``factory(ctx, binding) -> list[Handler]`` (called once per run, never a built handler: per-run state is
never in a registry) and ``GENERATORS`` (group ``fsm_llm.generators``) maps a generator name to the
function that writes a pattern FSM; ``fsm_llm.agents.handlers`` registers its library at import.
``PATTERN_AGENTS`` (``pattern_agents``, plan a555dcfe D-030) maps a built-in pattern id to the agent
class ``create_agent`` builds for it: the one class table, filled by ``fsm_llm.agents`` at import.

Interface contract:

- ``adapter_for(kind)`` returns the factory ``factory(envelope, **options) -> Executable`` of
  the kind, loading the entry-point group on a miss, or raises ``ExecutableKindError`` naming the
  kind and the known ones.
- ``namespace_resolver(handlers=None, generators=None)`` is the ``(category, name) -> bool`` that
  ``load_executable(resolver=)`` takes: ``handlers`` and ``generators`` are answered by the two
  namespaces, any other category (tools, callables, executables) is not judged here and answers True.
- No namespace has an entry in ``pyproject.toml`` yet (E2.3 proper): a source registers by
  an explicit ``EXECUTABLE_SOURCES.register(name, factory)`` call. Both registries are plain
  module constants, listed by the namespace test of ``tests/test_fsm_llm/test_names.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from fsm_llm.core.names import NameNotFoundError, NameRegistry

from .adapters import fsm_adapter, procedure_adapter, strategy_adapter
from .protocol import ExecutableKindError

__all__ = [
    "EXECUTABLE_KINDS",
    "EXECUTABLE_SOURCES",
    "GENERATORS",
    "HANDLERS",
    "PATTERN_AGENTS",
    "adapter_for",
    "namespace_resolver",
]

EXECUTABLE_KINDS: NameRegistry[Callable[..., Any]] = NameRegistry(
    "executable_kinds", entry_point_group="fsm_llm.executable_kinds"
)
EXECUTABLE_KINDS.register("fsm", fsm_adapter)
EXECUTABLE_KINDS.register("strategy", strategy_adapter)
EXECUTABLE_KINDS.register("procedure", procedure_adapter)

EXECUTABLE_SOURCES: NameRegistry[Callable[[], Iterable[Any]]] = NameRegistry(
    "executable_sources", entry_point_group="fsm_llm.executable_sources"
)

HANDLERS: NameRegistry[Callable[..., Any]] = NameRegistry(
    "handlers", entry_point_group="fsm_llm.handlers"
)
GENERATORS: NameRegistry[Callable[..., Any]] = NameRegistry(
    "generators", entry_point_group="fsm_llm.generators"
)
PATTERN_AGENTS: NameRegistry[type] = NameRegistry("pattern_agents")


def namespace_resolver(
    handlers: NameRegistry[Callable[..., Any]] | None = None,
    generators: NameRegistry[Callable[..., Any]] | None = None,
) -> Callable[[str, str], bool]:
    """The ``(category, name) -> bool`` resolver of ``load_executable`` over the two namespaces."""
    by_category = {
        "handlers": HANDLERS if handlers is None else handlers,
        "generators": GENERATORS if generators is None else generators,
    }

    def resolve(category: str, name: str) -> bool:
        registry = by_category.get(category)
        return True if registry is None else name in registry

    return resolve


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
