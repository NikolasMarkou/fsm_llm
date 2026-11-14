"""Release of an instance's timers, listeners and buffers, and the purge of terminal instances (plan d454fd62 step 91; the old ``workflows/engine.py``, same dotted path).

``ResourcesMixin`` of ``WorkflowEngine``: ``_drop_wait_resources`` (the wait only, the deadline watchdog
stays), ``_release_instance_resources`` (everything, matched by ``Timer.instance_id``), ``remove_instance``,
``_forget_instance`` and the purge of the oldest terminal instances. The method bodies are the old ones,
moved.
"""

from __future__ import annotations

from fsm_llm.logging import logger

from .helpers import _timer_key
from .surface import _EngineSurface


class ResourcesMixin(_EngineSurface):
    """Release of wait and instance resources, removal and purge of terminal instances."""

    def _drop_wait_resources(self, instance_id: str) -> None:
        """Drop the listeners, event timeouts and timer of ``instance_id``
        (everything tied to its current wait; the deadline watchdog stays)."""
        deadline_key = _timer_key(instance_id, "deadline")
        for timer_key, timer in list(self.timers.items()):
            if timer.instance_id == instance_id and timer_key != deadline_key:
                timer.cancel()
                self.timers.pop(timer_key, None)
        for listeners in self.event_listeners.values():
            listeners.pop(instance_id, None)

    def _release_instance_resources(self, instance_id: str) -> None:
        """Drop every timer, listener and buffered event of ``instance_id``.

        # DECISION plan-2026-09-27T120000-5d1e7a3b/D-005
        # Match timers by `Timer.instance_id`, NOT by the key prefix
        # f"{instance_id}_": with custom ids, cancelling "order" also matched
        # "order_2_timer" and killed another instance's timer.
        """
        for timer_key, timer in list(self.timers.items()):
            if timer.instance_id != instance_id:
                continue
            try:
                timer.cancel()
            except Exception as e:
                logger.warning(f"Failed to cancel timer {timer_key}: {e}")
            finally:
                self.timers.pop(timer_key, None)
        for listeners in self.event_listeners.values():
            listeners.pop(instance_id, None)
        self._event_buffers.pop(instance_id, None)

    def remove_instance(self, instance_id: str) -> bool:
        """Remove a terminal workflow instance from memory.

        Returns True if the instance was found and removed, False otherwise.
        Only terminal instances (COMPLETED, FAILED, CANCELLED) can be removed.
        """
        instance = self.workflow_instances.get(instance_id)
        if instance is None:
            return False
        if not instance.is_terminal():
            logger.warning(
                f"Cannot remove active instance {instance_id} "
                f"(status={instance.status.value})"
            )
            return False
        self._forget_instance(instance_id)
        logger.debug(f"Removed terminal instance {instance_id}")
        return True

    def _forget_instance(self, instance_id: str) -> None:
        self.workflow_instances.pop(instance_id, None)
        # F4: drop the per-instance lock too, otherwise _instance_locks grows
        # unbounded across the engine's lifetime.
        self._instance_locks.pop(instance_id, None)
        self._cancel_requests.pop(instance_id, None)
        self._instance_definitions.pop(instance_id, None)
        self._release_instance_resources(instance_id)

    def _purge_oldest_terminal_instances(self) -> None:
        """Remove oldest terminal instances if max_completed_instances is exceeded."""
        if self.max_completed_instances is None:
            return
        terminal = [
            (iid, inst)
            for iid, inst in self.workflow_instances.items()
            if inst.is_terminal()
        ]
        if len(terminal) <= self.max_completed_instances:
            return
        # Sort by completion time (oldest first)
        terminal.sort(key=lambda x: x[1].completed_at or x[1].updated_at)
        to_remove = len(terminal) - self.max_completed_instances
        for iid, _ in terminal[:to_remove]:
            self._forget_instance(iid)
        logger.debug(f"Purged {to_remove} oldest terminal workflow instances")
