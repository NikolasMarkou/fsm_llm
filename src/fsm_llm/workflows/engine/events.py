"""Event waits, event delivery and buffered events of ``WorkflowEngine`` (plan d454fd62 step 91; the old ``workflows/engine.py``, same dotted path).

``EventsMixin``: ``register_event_listener`` and its timeout, ``process_event`` (collect and consume the
matching listeners with no await, then deliver per instance under its lock), the event mapping and the
buffer of targeted events that arrived early. The method bodies are the old ones, moved.
"""

from __future__ import annotations

import asyncio
import collections
from datetime import UTC, datetime, timedelta
from typing import Any

from fsm_llm.logging import logger

from ..constants import (
    CONTEXT_KEY_EVENT_TIMEOUT,
    CONTEXT_KEY_LAST_EVENT,
    CONTEXT_KEY_WAITING_INFO,
    MAX_BUFFERED_EVENTS_PER_INSTANCE,
)
from ..exceptions import WorkflowEventError
from ..models import (
    EventListener,
    WorkflowEvent,
    WorkflowInstance,
    WorkflowStatus,
)
from .helpers import _timeout_error, _timer_key
from .surface import _EngineSurface


class EventsMixin(_EngineSurface):
    """Event listeners, event delivery and the event buffers of ``WorkflowEngine``."""

    async def register_event_listener(
        self,
        instance_id: str,
        event_type: str,
        success_step_id: str | None = None,
        timeout_seconds: float | None = None,
        timeout_step_id: str | None = None,
        event_mapping: dict[str, str] | None = None,
        correlation_key: str | None = None,
    ) -> None:
        """Register a workflow instance to listen for an event.

        ``success_step_id`` of ``""``/``None`` completes the instance when the
        event arrives. With ``timeout_seconds``, the wait times out to
        ``timeout_step_id``, or FAILS the instance when none is given.
        ``correlation_key`` restricts delivery to events whose payload value
        for that key equals the instance's context value for it now.
        """
        if not event_type:
            raise WorkflowEventError("event_type is required", event_type="")
        instance = self._live_instance(instance_id, "listen for events")

        # Create listener
        listener = EventListener(
            instance_id=instance_id,
            success_step_id=success_step_id or "",
            event_mapping=event_mapping or {},
            correlation_key=correlation_key,
            correlation_value=(
                instance.context.get(correlation_key) if correlation_key else None
            ),
            timeout_step_id=timeout_step_id,
            step_id=instance.current_step_id,
        )

        if timeout_seconds is not None:
            listener.timeout_at = datetime.now(UTC) + timedelta(seconds=timeout_seconds)

        self.event_listeners.setdefault(event_type, {})[instance_id] = listener
        logger.info(
            f"Registered event listener: instance {instance_id} for event {event_type}"
        )

        # DECISION plan-2026-09-27T120000-5d1e7a3b/D-001
        # Schedule the timeout whenever timeout_seconds is set, with or
        # without timeout_step_id. Do NOT go back to "only with timeout_step_id":
        # the listener then expired silently, later events were skipped, and
        # the instance stayed WAITING forever. No timeout_step_id -> FAILED.
        if timeout_seconds is not None:
            self._arm_timer(
                _timer_key(instance_id, "timeout", event_type),
                instance_id,
                timeout_step_id,
                timeout_seconds,
                lambda: self._handle_event_timeout(
                    instance_id, event_type, timeout_step_id
                ),
                "event timeout",
            )
            logger.info(f"Set up event timeout: {timeout_seconds} seconds")

    async def _handle_event_timeout(
        self, instance_id: str, event_type: str, timeout_step_id: str | None
    ) -> None:
        """Handle an event timeout: go to ``timeout_step_id`` or FAIL."""
        logger.info(
            f"Event timeout for instance {instance_id} waiting for {event_type}"
        )

        instance = self.workflow_instances.get(instance_id)
        if instance is None:
            return

        # F4: hold the per-instance lock across the status re-check and the
        # transition, so this timeout cannot interleave with a concurrent
        # advance/cancel/process_event transition on the
        # same instance.
        async with self._get_instance_lock(instance_id):
            # DECISION plan_2026-05-29_5b2fbb09/D-001 [STALE]
            # If the event already fired, process_event has consumed the listener
            # and transitioned the instance out of WAITING. A timeout firing
            # during that (slow) success transition must NOT drive a second,
            # concurrent transition to the timeout state. Only a still-WAITING
            # instance should time out (RW3-001).
            if instance.status != WorkflowStatus.WAITING:
                logger.debug(
                    f"Event timeout for {instance_id} ignored: instance no longer "
                    f"WAITING (status={instance.status.value})"
                )
                return

            self._forget_timer(_timer_key(instance_id, "timeout", event_type))
            self.event_listeners.get(event_type, {}).pop(instance_id, None)

            instance.context[CONTEXT_KEY_EVENT_TIMEOUT] = {
                "event_type": event_type,
                "timeout_at": datetime.now(UTC).isoformat(),
            }

            if not timeout_step_id:
                wait_seconds = None
                waiting_info = instance.context.get(CONTEXT_KEY_WAITING_INFO) or {}
                if isinstance(waiting_info, dict):
                    wait_seconds = waiting_info.get("timeout_seconds")
                self._fail(
                    instance,
                    _timeout_error(
                        f"wait for event '{event_type}'",
                        float(wait_seconds or 0.0),
                        instance_id,
                    ),
                )
                return

            # NOTE (W-ISSUE-004): the step budget restarts here (event-mediated
            # transition); loops through a wait are allowed by design.
            try:
                await self._transition_to_state(instance, timeout_step_id)
            except Exception as e:
                self._fail(instance, e)

    async def process_event(self, event: WorkflowEvent) -> list[str]:
        """Deliver an external event; return the ids of the instances it woke.

        A broadcast event (``event.instance_id is None``) goes to every
        instance waiting for ``event.event_type`` whose correlation matches;
        a targeted event only to that instance, and is buffered for it when
        it is not waiting for the event yet. One instance's failure never
        prevents delivery to the others.
        """
        event_type = event.event_type
        target = event.instance_id

        # Collect matching listeners (no awaits inside), consuming them so a
        # second event cannot claim them.
        # DECISION plan-2026-09-27T120000-5d1e7a3b/D-001
        # Do NOT mutate instance context here: delivery happens below under
        # each instance's own lock, after its status is re-checked.
        candidates: list[tuple[str, EventListener]] = []
        listeners = self.event_listeners.get(event_type, {})
        for instance_id, listener in list(listeners.items()):
            if target is not None and instance_id != target:
                continue
            if instance_id not in self.workflow_instances:
                listeners.pop(instance_id, None)
                continue
            # Skip expired listeners (their timeout task handles them)
            if listener.is_expired():
                logger.warning(
                    f"Skipping expired listener for event '{event_type}' "
                    f"on instance {instance_id}"
                )
                continue
            if not self._correlates(
                listener.correlation_key, listener.correlation_value, event
            ):
                continue
            listeners.pop(instance_id, None)
            candidates.append((instance_id, listener))

        if not candidates:
            if target is not None:
                self._buffer_event(target, event)
            else:
                logger.debug(f"No listeners for event type: {event_type}")
            return []

        affected_instances: list[str] = []
        delivered = 0
        try:
            for instance_id, listener in candidates:
                delivered += 1
                await self._deliver_event(
                    event, instance_id, listener, affected_instances
                )
        except asyncio.CancelledError:
            # Listeners were consumed up front: put back those not yet
            # delivered so the instances can still be woken by a later event.
            self._restore_listeners(event_type, candidates[delivered:])
            raise

        logger.info(
            f"Processed event {event_type}, affected instances: {len(affected_instances)}"
        )
        return affected_instances

    def _restore_listeners(
        self, event_type: str, pending: list[tuple[str, EventListener]]
    ) -> None:
        listeners = self.event_listeners.setdefault(event_type, {})
        for instance_id, listener in pending:
            instance = self.workflow_instances.get(instance_id)
            if (
                instance is not None
                and instance.status == WorkflowStatus.WAITING
                and instance.current_step_id == listener.step_id
            ):
                listeners.setdefault(instance_id, listener)

    async def _deliver_event(
        self,
        event: WorkflowEvent,
        instance_id: str,
        listener: EventListener,
        affected_instances: list[str],
    ) -> None:
        """Deliver ``event`` to one instance under its lock."""
        event_type = event.event_type
        instance = self.workflow_instances.get(instance_id)
        if instance is None:
            return
        # DECISION plan-2026-09-27T120000-5d1e7a3b/D-001
        # Per-instance isolation: an exception while delivering to one
        # instance (deadline passed, instance cancelled meanwhile, bad
        # state) must NOT abort delivery to the rest -- their listeners
        # were already consumed above and they would stay WAITING forever.
        try:
            async with self._get_instance_lock(instance_id):
                # Deliver only to the wait this listener belongs to: the
                # instance may have moved on to a different wait (also
                # WAITING) while the lock was held elsewhere.
                if (
                    instance.status != WorkflowStatus.WAITING
                    or listener.step_id not in (None, instance.current_step_id)
                ):
                    logger.debug(
                        f"Event '{event_type}' not delivered to {instance_id}: "
                        f"status is {instance.status.value}, step "
                        f"{instance.current_step_id}"
                    )
                    return
                # Cancel the wait's timeout BEFORE transitioning: the chain
                # may re-wait on the same event type and arm a new timeout
                # under the same key, which a later cancel would kill.
                self._cancel_event_timeout(instance_id, event_type)
                self._apply_event(instance, listener.event_mapping, event)
                affected_instances.append(instance_id)
                # NOTE (W-ISSUE-004): the step budget restarts here
                # (event-mediated transition); loops through a wait are
                # allowed by design.
                if listener.success_step_id:
                    await self._transition_to_state(instance, listener.success_step_id)
                else:
                    self._leave_wait(instance, WorkflowStatus.COMPLETED)
        except Exception as e:
            logger.error(
                f"Delivering event '{event_type}' to {instance_id} failed: {e!s}"
            )
            self._fail(instance, e)

    @staticmethod
    def _correlates(
        correlation_key: str | None, expected: Any, event: WorkflowEvent
    ) -> bool:
        if not correlation_key:
            return True
        return (
            correlation_key in event.payload
            and event.payload[correlation_key] == expected
        )

    def _apply_event(
        self,
        instance: WorkflowInstance,
        event_mapping: dict[str, str] | None,
        event: WorkflowEvent,
    ) -> None:
        """Map an event's payload into the instance context."""
        for context_key, payload_key in (event_mapping or {}).items():
            if payload_key in event.payload:
                instance.context[context_key] = event.payload[payload_key]
            else:
                logger.warning(
                    f"Event mapping key '{payload_key}' not found in "
                    f"event payload for instance {instance.instance_id}; "
                    f"context key '{context_key}' will not be set"
                )
        instance.context[CONTEXT_KEY_LAST_EVENT] = event.model_dump()

    def _buffer_event(self, instance_id: str, event: WorkflowEvent) -> None:
        instance = self.workflow_instances.get(instance_id)
        if instance is None or instance.is_terminal():
            logger.warning(
                f"Targeted event '{event.event_type}' dropped: instance "
                f"{instance_id} is unknown or finished"
            )
            return
        buffer = self._event_buffers.setdefault(
            instance_id, collections.deque(maxlen=MAX_BUFFERED_EVENTS_PER_INSTANCE)
        )
        buffer.append(event)
        logger.info(
            f"Buffered event '{event.event_type}' for instance {instance_id} "
            f"({len(buffer)} buffered)"
        )

    def _take_buffered_event(
        self,
        instance: WorkflowInstance,
        event_type: str,
        correlation_key: str | None,
    ) -> WorkflowEvent | None:
        buffer = self._event_buffers.get(instance.instance_id)
        if not buffer:
            return None
        expected = instance.context.get(correlation_key) if correlation_key else None
        for event in list(buffer):
            if event.event_type == event_type and self._correlates(
                correlation_key, expected, event
            ):
                buffer.remove(event)
                return event
        return None

    def _cancel_event_timeout(self, instance_id: str, event_type: str) -> None:
        """Cancel an event timeout."""
        timer = self.timers.pop(_timer_key(instance_id, "timeout", event_type), None)
        if timer is not None:
            timer.cancel()
