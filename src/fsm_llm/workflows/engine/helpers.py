"""The small pieces the engine modules share (plan d454fd62 step 91; the old ``workflows/engine.py``, same dotted path).

``LifecycleHook`` (the hook signature), ``Timer`` (a scheduled timer), ``_CancelRequest`` (a ``cancel`` call
waiting for an instance lock), ``_timer_key`` (the one spelling of the three timer keys) and ``_timeout_error``
(the one spelling of a ``WorkflowTimeoutError`` raised by the engine). Imports no other module of the package
besides ``models`` and ``exceptions``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime
from typing import Any

from ..exceptions import WorkflowTimeoutError
from ..models import WorkflowInstance

#: Signature of a lifecycle hook: ``hook(event_name, instance, data)``.
LifecycleHook = Callable[[str, WorkflowInstance, dict[str, Any]], Any]


class _CancelRequest:
    """A ``cancel`` call waiting for an instance's lock.

    Interface contract: ``reason`` is the caller's text; ``honored`` is set
    to ``True`` only by the run loop that cancelled the instance on this
    request. Never raises; one request per instance at a time.
    """

    __slots__ = ("honored", "reason")

    def __init__(self, reason: str) -> None:
        self.reason = reason
        self.honored = False


class Timer:
    """Information about a timer scheduled for a workflow instance."""

    def __init__(
        self,
        instance_id: str,
        next_step_id: str | None,
        expires_at: datetime,
        task: asyncio.Task | None = None,
    ):
        self.instance_id = instance_id
        self.next_step_id = next_step_id
        self.expires_at = expires_at
        self.task = task

    def cancel(self) -> None:
        """Cancel the timer task (never the task that is currently running,
        which would abort the very transition that task is performing)."""
        if self.task and not self.task.done():
            try:
                current = asyncio.current_task()
            except RuntimeError:
                current = None
            if self.task is not current:
                self.task.cancel()


def _timer_key(instance_id: str, kind: str, event_type: str = "") -> str:
    """Key of an instance's timer in ``WorkflowEngine.timers``.

    ``kind`` is ``"timer"`` (``{id}_timer``), ``"deadline"`` (``{id}_deadline``)
    or ``"timeout"`` with the ``event_type`` waited for
    (``{id}_{event_type}_timeout``). The one spelling of the three keys.
    """
    return "_".join(part for part in (instance_id, event_type, kind) if part)


def _timeout_error(
    operation: str, timeout_seconds: float, instance_id: str
) -> WorkflowTimeoutError:
    """The one spelling of a ``WorkflowTimeoutError`` raised by the engine."""
    return WorkflowTimeoutError(
        f"Operation '{operation}' timed out after {timeout_seconds} seconds",
        operation=operation,
        timeout_seconds=timeout_seconds,
        instance_id=instance_id,
    )
