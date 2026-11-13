"""``clear_keys_on_entry``: a ready-made handler built on ``create_handler``.

Package ``fsm_llm.handlers`` (plan d454fd62, step 77: split out of the old module
``fsm_llm/handlers.py``). It uses ``clear_keys_delta`` from ``fsm_llm.context.commit``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from fsm_llm.context.commit import clear_keys_delta
from fsm_llm.core.definitions import checked_key_names

from .base import BaseHandler
from .builder import create_handler


def clear_keys_on_entry(
    keys: Iterable[str],
    *,
    state: str,
    name: str | None = None,
    priority: int = 100,
) -> BaseHandler:
    """Build a handler that clears ``keys`` whenever ``state`` is entered.

    Core extracts a key only while it is unset (skip-if-set), so a value left
    in context from an earlier visit is never re-extracted. Clearing it on
    entry to the state that produces it lets that state's extraction run again.

    Contract:
        - ``keys``: non-empty; duplicates are dropped, order kept.
        - ``state``: the state whose entry (``POST_TRANSITION`` with that
          target state, :meth:`HandlerBuilder.on_state_entry`) fires the
          handler.
        - ``name``: handler name; defaults to ``clear_keys_on_<state>``.
        - ``priority``: handler priority (lower runs first), default 100.
        - Returns a :class:`BaseHandler` to pass to ``API.register_handler``;
          its delta is :func:`clear_keys_delta` of ``keys``.
        - Raises ``ValueError`` at build time for empty ``keys`` or ``keys``
          that is not a collection of ``str`` names (``None`` or a bare
          ``str``, ``definitions.checked_key_names``).
    """
    cleared = checked_key_names(keys, argument="keys")
    if not cleared:
        raise ValueError("clear_keys_on_entry needs at least one key")

    def clear_keys(context: dict[str, Any]) -> dict[str, Any]:
        return clear_keys_delta(cleared, context)

    return (
        create_handler(name or f"clear_keys_on_{state}")
        .on_state_entry(state)
        .with_priority(priority)
        .do(clear_keys)
    )
