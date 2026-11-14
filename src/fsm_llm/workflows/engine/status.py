"""Hooks, the per-instance lock and the status of an instance (plan d454fd62 step 91; the old ``workflows/engine.py``, same dotted path).

``StatusMixin`` of ``WorkflowEngine``: the lifecycle hooks and their emission, the lock registry, and the
one place a status changes (``_set_status``, which also releases a terminal instance's resources) with
``_fail``. The method bodies are the old ones, moved.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

from fsm_llm.logging import logger

from ..models import WorkflowInstance, WorkflowStatus
from .helpers import LifecycleHook
from .surface import _EngineSurface


class StatusMixin(_EngineSurface):
    """Hooks, the instance locks and every status change of ``WorkflowEngine``."""

    def add_hook(self, hook: LifecycleHook) -> None:
        """Register ``hook(event_name, instance, data)``.

        Events: ``"step_started"`` (``{"step_id"}``), ``"step_completed"``
        (``{"step_id", "success", "next_step_id", "error"}``) and
        ``"status_changed"`` (``{"status", "error"}``). A hook's exception is
        logged and never affects the workflow. A hook returning an awaitable
        is scheduled as a task.
        """
        self._hooks.append(hook)

    def remove_hook(self, hook: LifecycleHook) -> bool:
        """Unregister a hook. Returns False if it was not registered."""
        try:
            self._hooks.remove(hook)
            return True
        except ValueError:
            return False

    def _emit(self, event: str, instance: WorkflowInstance, **data: Any) -> None:
        for hook in list(self._hooks):
            try:
                result = hook(event, instance, data)
                if inspect.isawaitable(result):
                    task = asyncio.ensure_future(result)
                    self._background_tasks.add(task)
                    task.add_done_callback(self._background_tasks.discard)
            except Exception as e:
                logger.warning(f"Workflow hook failed on {event}: {e!s}")

    def _get_instance_lock(self, instance_id: str) -> asyncio.Lock:
        """Get (creating if absent) the per-instance lock for ``instance_id``.

        # DECISION plan-2026-09-12T065608-089d0ec7/D-003
        # F4 fix: the lock is acquired ONLY at the outermost public entry
        # points (`start`, `advance`, `cancel`,
        # `process_event`'s per-instance delivery loop,
        # `_handle_timer_expiration`, `_handle_event_timeout`,
        # `_handle_deadline`) -- never inside
        # `_execute_workflow_step`/`_transition_to_state` themselves.
        # `_transition_to_state` runs `_execute_workflow_step`, so acquiring
        # this same `asyncio.Lock` at those inner levels would deadlock on
        # the very first reentrant call -- `asyncio.Lock` is NOT reentrant.
        # See decisions.md D-003.

        Interface contract: keyed by `instance_id`; returns the same
        `asyncio.Lock` object for repeated calls with the same id until that
        id's entry is removed (see `remove_instance`,
        `_purge_oldest_terminal_instances`). Never raises.
        """
        lock = self._instance_locks.get(instance_id)
        if lock is None:
            lock = asyncio.Lock()
            self._instance_locks[instance_id] = lock
        return lock

    def _set_status(
        self,
        instance: WorkflowInstance,
        status: WorkflowStatus,
        error: Exception | None = None,
    ) -> None:
        """Change an instance's status, notify hooks, and release every
        timer, listener and buffered event of an instance that became
        terminal (completed, failed and cancelled alike)."""
        previous = instance.status
        instance.update_status(status, error)
        if previous != status:
            self._emit(
                "status_changed",
                instance,
                status=status.value,
                error=str(error) if error else None,
            )
        if instance.is_terminal():
            self._release_instance_resources(instance.instance_id)
            self._purge_oldest_terminal_instances()

    def _fail(self, instance: WorkflowInstance, error: Exception) -> None:
        """FAIL an instance unless it is already terminal.

        # DECISION plan-2026-09-12T065608-089d0ec7/D-005
        # Do NOT call update_status(FAILED, ...) unconditionally. If the
        # instance is already terminal (e.g. a concurrent cancel won
        # the race for this instance's lock before a step exception was
        # handled), update_status raises WorkflowStateError (terminal states
        # map to an empty transition set in _VALID_STATUS_TRANSITIONS) from
        # inside an except-block handler, which REPLACES the original
        # exception -- the caller never sees the real error. See
        # decisions.md D-005.
        """
        if instance.is_terminal():
            logger.debug(
                f"Instance {instance.instance_id} already terminal "
                f"({instance.status.value}); not overwriting with FAILED"
            )
            return
        self._set_status(instance, WorkflowStatus.FAILED, error)
