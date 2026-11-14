"""Timers, timer expiry and the deadline watchdog of ``WorkflowEngine`` (plan d454fd62 step 91; the old ``workflows/engine.py``, same dotted path).

``TimersMixin``: ``_arm_timer`` (the one way a timer task is created, for a timer, an event timeout and a
deadline), ``schedule_timer``, the expiry handlers that act only on a still-WAITING instance, and
``_live_instance`` (the guard of the entry points that arm something). The method bodies are the old ones,
moved.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from fsm_llm.logging import logger

from ..constants import CONTEXT_KEY_TIMER_EXPIRED
from ..exceptions import WorkflowInstanceError
from ..models import WorkflowInstance, WorkflowStatus
from .helpers import Timer, _timeout_error, _timer_key
from .surface import _EngineSurface


class TimersMixin(_EngineSurface):
    """Timers, timer expiry and the deadline watchdog of ``WorkflowEngine``."""

    def _live_instance(self, instance_id: str, purpose: str) -> WorkflowInstance:
        """The instance ``instance_id`` if it exists and is not terminal.

        Raises ``WorkflowInstanceError`` for an unknown or finished instance
        (``purpose`` completes "cannot ..."): the one guard of the entry
        points that arm something for an instance.
        """
        instance = self.workflow_instances.get(instance_id)
        if instance is None:
            raise WorkflowInstanceError(
                "Workflow instance not found", instance_id=instance_id
            )
        if instance.is_terminal():
            raise WorkflowInstanceError(
                f"Instance is {instance.status.value}; cannot {purpose}",
                instance_id=instance_id,
            )
        return instance

    def _arm_timer(
        self,
        key: str,
        instance_id: str,
        next_step_id: str | None,
        delay: float,
        handler: Callable[[], Any],
        label: str,
    ) -> None:
        """Run ``await handler()`` after ``delay`` seconds as the timer ``key``.

        Interface contract: ``handler`` is a zero-argument coroutine function;
        a timer already stored under ``key`` is cancelled first (re-arming must
        not orphan a task that could still fire and transition twice). The
        task swallows its own cancellation and logs any other error under
        ``label``. Needs a running loop; never raises.
        """

        async def fire() -> None:
            try:
                await asyncio.sleep(delay)
                await handler()
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Error in {label} task: {e!s}")

        task = asyncio.create_task(fire())
        existing = self.timers.get(key)
        if existing is not None:
            existing.cancel()
        expires_at = datetime.now(UTC) + timedelta(seconds=delay)
        self.timers[key] = Timer(instance_id, next_step_id, expires_at, task)

    def _forget_timer(self, key: str) -> None:
        """Drop the entry ``key`` when it is the task running this call (a timer
        that fired; a timer re-armed meanwhile keeps its entry)."""
        timer = self.timers.get(key)
        if timer is not None and timer.task is asyncio.current_task():
            del self.timers[key]

    async def schedule_timer(
        self, instance_id: str, delay_seconds: float, next_step_id: str
    ) -> None:
        """Schedule a timer that moves a WAITING instance to ``next_step_id``.

        Raises ``WorkflowInstanceError`` for an unknown or finished instance.
        """
        self._live_instance(instance_id, "schedule a timer")
        self._arm_timer(
            _timer_key(instance_id, "timer"),
            instance_id,
            next_step_id,
            delay_seconds,
            lambda: self._handle_timer_expiration(instance_id, next_step_id),
            "timer",
        )
        logger.info(
            f"Scheduled timer for instance {instance_id}: {delay_seconds} seconds"
        )

    async def _handle_timer_expiration(
        self, instance_id: str, next_step_id: str
    ) -> None:
        """Handle a timer expiration."""
        logger.info(f"Timer expired for instance {instance_id}")

        instance = self.workflow_instances.get(instance_id)
        if instance is None:
            return

        # F4: per-instance lock, same reasoning as _handle_event_timeout above.
        async with self._get_instance_lock(instance_id):
            # Symmetric guard (RW3-001): if the instance was advanced out of
            # WAITING by some other path (e.g. an external advance)
            # before this timer fired, do not drive a stale transition.
            if instance.status != WorkflowStatus.WAITING:
                logger.debug(
                    f"Timer for {instance_id} ignored: instance no longer WAITING "
                    f"(status={instance.status.value})"
                )
                return

            instance.context[CONTEXT_KEY_TIMER_EXPIRED] = {
                "expired_at": datetime.now(UTC).isoformat()
            }

            self._forget_timer(_timer_key(instance_id, "timer"))

            # NOTE (W-ISSUE-004): the step budget restarts here (timer-mediated
            # transition); loops through a timer are allowed by design.
            try:
                if next_step_id == "":
                    # DECISION plan-2026-11-05T134105-5ed71cf2/D-002
                    # An empty timer target is the terminal-route marker the
                    # validator accepts (same as wait_event success_step_id="",
                    # see _deliver_event). Do NOT send it to
                    # _transition_to_state: it would raise "Invalid next
                    # state" and FAIL the instance after the full delay. Do
                    # NOT apply this to APIStep(failure_step_id=""): a
                    # failure with an empty route still fails (workflows
                    # D-003; completing would hide the error).
                    self._leave_wait(instance, WorkflowStatus.COMPLETED)
                else:
                    await self._transition_to_state(instance, next_step_id)
            except Exception as e:
                self._fail(instance, e)

    def _arm_deadline(self, instance: WorkflowInstance) -> None:
        """Arm a watchdog that FAILS the instance if it is still WAITING at
        its deadline (a step boundary never comes while it waits)."""
        assert instance.deadline is not None
        instance_id = instance.instance_id
        delay = max(0.0, (instance.deadline - datetime.now(UTC)).total_seconds())
        self._arm_timer(
            _timer_key(instance_id, "deadline"),
            instance_id,
            None,
            delay,
            lambda: self._handle_deadline(instance_id),
            "deadline",
        )

    async def _handle_deadline(self, instance_id: str) -> None:
        instance = self.workflow_instances.get(instance_id)
        if instance is None:
            return
        async with self._get_instance_lock(instance_id):
            self._forget_timer(_timer_key(instance_id, "deadline"))
            if instance.status != WorkflowStatus.WAITING:
                return
            logger.warning(
                f"Workflow instance {instance_id} reached its deadline while WAITING"
            )
            self._fail(
                instance,
                _timeout_error(
                    f"step {instance.current_step_id}",
                    self._timeout_seconds(instance),
                    instance_id,
                ),
            )
