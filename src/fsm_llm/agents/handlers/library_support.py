"""The helpers every handler factory of the library shares (split from ``library.py``, plan a555dcfe).

``_states``, ``_priority``, ``_params``, ``_keys`` read the ``HandlerBinding`` a pattern file gives a factory (by
attribute, ``None`` allowed) and ``_on_entry`` builds the one entry handler of a state. Interface contract:
``binding`` is a ``HandlerBinding`` or ``None``; ``default`` is the factory's own value; ``_keys`` raises
``AgentError`` for an internal-prefixed or secret-looking key; ``_on_entry`` returns a one-element list.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from fsm_llm.agents.tools import flagged_tool_names
from fsm_llm.constants import has_internal_prefix, is_forbidden_context_entry
from fsm_llm.core.definitions import AgentError
from fsm_llm.handlers import create_handler

if TYPE_CHECKING:
    from fsm_llm.agents.handlers.context import HandlerContext
    from fsm_llm.core.definitions.executable_pattern import HandlerBinding
    from fsm_llm.handlers import BaseHandler


def _refuse_flagged(ctx: HandlerContext, name: str) -> None:
    """Fail closed (D-035): a factory that runs tools with no approval path refuses a flagged tool.

    ``agents.rewoo_executor`` and ``agents.native_fc`` never ask the run's approval predicate, so a
    registry holding a ``requires_approval`` tool is an ``AgentError`` here, whatever the file or its
    agent claims (``agents.tool_executor`` is gated and does not call this).
    """
    if flagged := flagged_tool_names(ctx.tools):
        raise AgentError(
            f"{name} has no human-in-the-loop approval, but tools {flagged} have "
            "requires_approval=True. Use ReactAgent, ReflexionAgent or ReasoningReactAgent with a "
            "HumanInTheLoop, or register these tools without requires_approval."
        )


def _states(binding: HandlerBinding | None, default: Sequence[str]) -> tuple[str, ...]:
    """The states of a handler: the binding's when it lists any, else the factory's own."""
    listed = tuple(getattr(binding, "states", None) or ())
    return listed or tuple(default)


def _priority(binding: HandlerBinding | None, default: int) -> int:
    """The priority of a handler: the binding's when it sets one, else the factory's own."""
    chosen = getattr(binding, "priority", None)
    return default if chosen is None else int(chosen)


def _params(binding: HandlerBinding | None) -> dict[str, Any]:
    return dict(getattr(binding, "params", None) or {})


def _keys(binding: HandlerBinding | None, default: Sequence[str]) -> tuple[str, ...]:
    """The keys of a handler: the binding's (each checked, D2) or the factory's own."""
    listed = tuple(getattr(binding, "keys", None) or ())
    for key in listed:
        # a neutral value: the name decides (patterns), a value-shape layer needs a real secret
        if has_internal_prefix(key) or is_forbidden_context_entry(key, "x"):
            raise AgentError(
                f"handler binding names key {key!r}: internal-prefixed and secret-looking "
                "keys are never named by a pattern file"
            )
    return listed or tuple(default)


def _on_entry(
    name: str,
    binding: HandlerBinding | None,
    state: str,
    priority: int,
    do: Callable[[dict[str, Any]], dict[str, Any]],
) -> list[BaseHandler]:
    """One entry handler on *state* (the binding's ``states`` and ``priority`` win)."""
    return [
        create_handler(name)
        .with_priority(_priority(binding, priority))
        .on_state_entry(*_states(binding, (state,)))
        .do(do)
    ]
