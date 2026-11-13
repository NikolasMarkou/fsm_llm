"""The executor table, the per-instance map check and the lookup.

Layer ``engine`` (plan d454fd62, step 80): split out of the old ``fsm_llm.executors``; the table,
``register_executor``, ``check_executors`` and ``executor_for`` (with the D-026 anchor) are moved
unchanged. Interface contract: the package docstring of ``fsm_llm.engine.kinds``.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from fsm_llm.core.definitions import State
from fsm_llm.core.state_kinds import get_kind, kind_of

from .base import StateExecutor
from .completion import CompletionExecutor
from .conversation import ConversationExecutor

_CONVERSATION = ConversationExecutor()

_CONVERSATION_KIND = "conversation"

# Kind name -> executor. ``synthesize`` is passive and has no entry: it runs as
# conversation through ``executor_for``'s fallback (a bare ``converse`` on it
# stays BLOCKED, D-002 of plan 2cefeb61), which also honours a per-instance
# override of ``conversation``.
_EXECUTORS: dict[str, StateExecutor] = {
    _CONVERSATION_KIND: _CONVERSATION,
    "completion": CompletionExecutor(),
}


def register_executor(kind_name: str, executor: StateExecutor) -> None:
    """Register (or replace) the executor of the kind named ``kind_name``."""
    _EXECUTORS[kind_name] = executor


def check_executors(
    executors: Mapping[str, StateExecutor] | None,
) -> Mapping[str, StateExecutor]:
    """Validate an ``executors=`` argument and return a read-only copy.

    Raises:
        TypeError: ``executors`` is not a mapping, a key is not a ``str``, or a
            value is not a ``StateExecutor``.
        ValueError: a key names no registered state kind.
    """
    if executors is None:
        return MappingProxyType({})
    if not isinstance(executors, Mapping):
        raise TypeError(
            "executors must be a mapping of kind name to StateExecutor, got "
            f"{type(executors).__name__}"
        )
    checked: dict[str, StateExecutor] = {}
    for name, executor in executors.items():
        if not isinstance(name, str):
            raise TypeError(
                f"executors keys must be kind names (str), got {type(name).__name__}"
            )
        if not isinstance(executor, StateExecutor):
            raise TypeError(
                f"executors[{name!r}] must be a StateExecutor, got "
                f"{type(executor).__name__}"
            )
        try:
            get_kind(name)
        except KeyError:
            raise ValueError(
                f"executors names unknown state kind {name!r}: register its "
                "KindSpec first"
            ) from None
        checked[name] = executor
    return MappingProxyType(checked)


def executor_for(
    state: State, executors: Mapping[str, StateExecutor] | None = None
) -> StateExecutor:
    """The executor of ``state``'s kind.

    Interface contract (callers: the turn pipeline, three sites):
        - Resolution order: the pipeline's own ``executors`` map for the
          kind's name, then the module table for it, then the ``conversation``
          entry of the same two tables in the same order (a kind with no
          executor of its own, ``synthesize`` or a runtime kind nobody
          registered, runs as conversation, so an override of ``conversation``
          in the per-instance map applies to it too), then the built-in
          conversation executor.
        - Never raises for a kind name; ``kind_of`` raises nothing for a
          state.
    """
    # DECISION plan-2026-11-12T073150-d454fd62/D-026
    # A passive kind falls back through the per-instance "conversation" entry.
    # Do NOT register ``synthesize`` in the module table as an alias of the
    # built-in conversation executor (it was one): the alias bypassed an
    # ``API(executors={"conversation": X})`` override for a synthesize state
    # (review gate 1, concern 11). A real synthesize executor registers under
    # its own name and then wins.
    name = kind_of(state).name
    for table in (executors, _EXECUTORS):
        if table is not None and name in table:
            return table[name]
    for table in (executors, _EXECUTORS):
        if table is not None and _CONVERSATION_KIND in table:
            return table[_CONVERSATION_KIND]
    return _CONVERSATION
