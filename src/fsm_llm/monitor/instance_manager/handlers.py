"""The observer handlers the manager registers on an ``API`` (plan d454fd62 step 93; the old ``monitor/instance_manager.py``, same dotted path).

``_MonitorHandler`` (observe-only, hands the core's state arguments to a collector callback through the
``_HANDLER_STATES`` context variable), the builders over a collector's callbacks, and the idempotent
``register_monitor_handlers`` / ``unregister_monitor_handlers`` / ``monitor_handlers_on``. Imports ``managed``
only (a context-capture handler writes into a ``ManagedAgent``).
"""

from __future__ import annotations

import contextvars
from typing import Any

from fsm_llm import API, HandlerTiming
from fsm_llm.handlers import BaseHandler

from ..collector import EventCollector
from ..constants import (
    MONITOR_HANDLER_NAME,
    MONITOR_HANDLER_PRIORITY,
)
from .managed import ManagedAgent

# The (current_state, target_state) the core passed to the last ``should_execute``
# of this context; ``execute`` reads it back (D-003 below). A ContextVar, not a
# ``threading.local``: with ``handler_timeout`` the core runs ``execute`` on a
# worker thread that starts from a copy of the caller's context.
_HANDLER_STATES: contextvars.ContextVar[tuple[str | None, str | None]] = (
    contextvars.ContextVar("monitor_handler_states", default=(None, None))
)


class _MonitorHandler(BaseHandler):
    """Observer handler that hands the core's state arguments to a monitor
    callback.

    # DECISION plan-2026-09-28T090000-3c9e41d2/D-003
    # The core passes ``current_state``/``target_state`` to ``should_execute``
    # only; the handler context has no ``_target_state``. Do NOT go back to a
    # ``create_handler(...).do(callback)`` lambda that reads the states from
    # the context: every transition event then had an empty target. The
    # states are kept in ``_HANDLER_STATES`` between ``should_execute`` and
    # ``execute`` (the core calls them back to back). Do NOT go back to a
    # ``threading.local``: with ``handler_timeout`` set ``execute`` runs on a
    # worker thread and read ``None, None`` (plan-2026-11-05T185903-5980fbb7,
    # F-HV-01); the worker starts from ``contextvars.copy_context()``.
    """

    def __init__(
        self,
        name: str,
        timing: HandlerTiming,
        callback: Any,
        collector: EventCollector,
        priority: int = MONITOR_HANDLER_PRIORITY,
    ) -> None:
        super().__init__(name=name, priority=priority)
        self.timing = timing
        self.callback = callback
        self.collector = collector
        self.active = True

    def should_execute(
        self,
        timing: HandlerTiming,
        current_state: str,
        target_state: str | None,
        context: dict[str, Any],
        updated_keys: set[str] | None = None,
    ) -> bool:
        if not self.active or timing != self.timing:
            return False
        _HANDLER_STATES.set((current_state, target_state))
        return True

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        current, target = _HANDLER_STATES.get()
        self.callback(context, current_state=current, target_state=target)
        # Observe only: never write into the conversation context.
        return {}


def _handlers_for(
    callbacks: dict[str, Any],
    collector: EventCollector,
    name_prefix: str,
    priority: int,
) -> list[_MonitorHandler]:
    return [
        _MonitorHandler(
            name=f"{name_prefix}_{timing_name.lower()}",
            timing=HandlerTiming[timing_name],
            callback=callback,
            collector=collector,
            priority=priority,
        )
        for timing_name, callback in callbacks.items()
    ]


def monitor_handlers_on(api: API, collector: EventCollector) -> list[_MonitorHandler]:
    """The active monitor handlers of ``collector`` registered on ``api``."""
    handlers = getattr(getattr(api, "handler_system", None), "handlers", []) or []
    return [
        h
        for h in handlers
        if isinstance(h, _MonitorHandler) and h.collector is collector and h.active
    ]


def register_monitor_handlers(
    api: API, collector: EventCollector
) -> list[_MonitorHandler]:
    """Register observer handlers on an API instance for event collection.

    Idempotent per (api, collector): calling it again does not stack a second
    set (the core has no way to unregister a handler). Returns the handlers
    now active for this collector; ``unregister_monitor_handlers`` switches
    them off.
    """
    existing = monitor_handlers_on(api, collector)
    if existing:
        return existing
    handlers = _handlers_for(
        collector.create_handler_callbacks(),
        collector,
        MONITOR_HANDLER_NAME,
        MONITOR_HANDLER_PRIORITY,
    )
    for handler in handlers:
        api.register_handler(handler)
    return handlers


def unregister_monitor_handlers(api: API, collector: EventCollector) -> int:
    """Switch off ``collector``'s monitor handlers on ``api``; returns how many."""
    handlers = monitor_handlers_on(api, collector)
    for handler in handlers:
        handler.active = False
    return len(handlers)


def _build_monitor_handlers(collector: EventCollector) -> list[Any]:
    """Build a list of handler objects for injection into agents via api_kwargs."""
    return list(
        _handlers_for(
            collector.create_handler_callbacks(),
            collector,
            MONITOR_HANDLER_NAME,
            MONITOR_HANDLER_PRIORITY,
        )
    )


def _build_context_capture_handlers(
    collector: EventCollector,
    sink: ManagedAgent,
) -> list[Any]:
    """Build handler objects that capture context snapshots into a ManagedAgent.

    Delegates to ``EventCollector.create_context_capture_callbacks`` for the
    actual capture logic, then wraps each callback as a handler object suitable
    for injection into agents via ``api_kwargs["handlers"]``.
    """
    return list(
        _handlers_for(
            collector.create_context_capture_callbacks(sink),
            collector,
            f"{MONITOR_HANDLER_NAME}_capture",
            MONITOR_HANDLER_PRIORITY - 1,
        )
    )
