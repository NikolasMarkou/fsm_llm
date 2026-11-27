"""Registration, ending and teardown of instances of ``InstanceManager`` (plan d454fd62 step 93; the old ``monitor/instance_manager.py``, same dotted path).

``LifecycleMixin``: ``_register_instance`` (make room, add, announce), ``end_instance`` and ``_teardown`` (shared
with ``close``), ``_make_room`` (evict the oldest finished instance) and ``_close_fsm_api`` (close a managed
FSM's API BEFORE detaching the global handlers, D-015); with ``_pending_tasks``, the strong references to the
fire-and-forget engine-close tasks. The method bodies are the old ones, moved.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fsm_llm.logging import logger

from ..collector import EventCollector
from ..constants import (
    ACTIVE_AGENT_STATUSES,
    EVENT_INSTANCE_DESTROYED,
    EVENT_INSTANCE_LAUNCHED,
    TERMINAL_STATUSES,
)
from ..exceptions import (
    MonitorCapacityError,
)
from .handlers import unregister_monitor_handlers
from .managed import (
    ManagedAgent,
    ManagedFSM,
    ManagedInstance,
    ManagedWorkflow,
    add_usage,
    api_usage,
)
from .surface import _InstanceSurface

# Strong references to fire-and-forget tasks (the loop keeps only weak ones).
_pending_tasks: set[Any] = set()


class LifecycleMixin(_InstanceSurface):
    """Adding, ending and tearing down managed instances."""

    def _close_fsm_api(self, inst: ManagedFSM, instance_id: str) -> None:
        """End every live conversation of a manager-owned FSM API, then detach
        the global collector's handlers.

        # DECISION plan-2026-11-05T134105-5ed71cf2/D-015
        # Close BEFORE unregistering: ``api.close()`` fires END_CONVERSATION,
        # and an inactive handler records nothing, so the conversation stayed
        # "active" on the global collector and its OTEL span was never ended.
        # Do NOT swap the two calls back. The unregister stays in a
        # ``finally`` so a failing close cannot leave the handlers on. Live
        # conversations are cached first so the Conversations view keeps
        # them, as for a conversation ended through ``end_conversation``.
        # ``attach_api`` (a caller-owned API) is left alone on purpose:
        # closing a caller's API is a policy choice.
        """
        api = inst.api
        try:
            for conversation_id in api.list_active_conversations():
                self._cache_ended_conversation(api, conversation_id, instance_id)
            api.close()
        except Exception as e:
            logger.debug(f"Failed to close FSM API of instance {instance_id}: {e}")
        finally:
            unregister_monitor_handlers(api, self._global_collector)

    def _register_instance(
        self,
        managed: ManagedInstance,
        collector: EventCollector,
        noun: str,
        **launched_data: Any,
    ) -> None:
        """Make room for ``managed``, add it with its collector and announce
        the launch.

        # DECISION plan-2026-11-05T185903-5980fbb7/D-005
        # Callers build and validate everything that can fail BEFORE this.
        # Do NOT move ``_make_room`` back to the top of a ``launch_*``: a
        # request that then failed validation had already destroyed the oldest
        # finished instance and its history.

        # DECISION plan-2026-11-27T102400-1f0925d2/D-005
        # [EXTENDS the D-005 of plan 5980fbb7 above, which stays]
        # The capacity checks and the insert are ONE ``_lock`` hold: launches run in
        # worker threads, so a check in an earlier scope lets two launches pass and
        # both register. Eviction stays OUTSIDE (``_make_room``: a join and an API
        # close never run under ``_lock``). Do NOT split the scope, take
        # ``_catalog_lock`` or log here.
        """
        self._make_room()
        instance_id = managed.instance_id
        with self._lock:
            self._check_instance_room()
            if isinstance(managed, ManagedAgent):
                self._check_agent_capacity()
            self._instances[instance_id] = managed
            self._collectors[instance_id] = collector
        self._emit_event(
            EVENT_INSTANCE_LAUNCHED,
            message=f"{noun} launched: {managed.label}",
            data={
                "instance_type": managed.instance_type,
                "instance_id": instance_id,
                **launched_data,
            },
        )
        logger.info(
            f"Launched {managed.instance_type} instance {instance_id}: {managed.label}"
        )

    def end_instance(self, instance_id: str) -> None:
        """End a managed instance (the ``end_*`` verb) and clean up resources."""
        # An FSM's usage is read ONCE, before the pop (it is still listed, so a metrics
        # read counts it through its own meter), and folded in the pop's scope: a read
        # between the two would see it twice or not at all. A launched FSM builds its
        # own interface, so the fold never repeats a live one; ``close`` folds nothing.
        with self._lock:
            listed = self._instances.get(instance_id)
        found = api_usage(listed.api) if isinstance(listed, ManagedFSM) else None
        with self._lock:
            inst = self._instances.pop(instance_id, None)
            self._collectors.pop(instance_id, None)
            if inst is not None:
                if found is not None:
                    self._ended_usage = add_usage(self._ended_usage, found[1])
                if isinstance(inst, ManagedAgent):
                    inst.cancel_event.set()
                    if inst.status in ACTIVE_AGENT_STATUSES:
                        inst.status = "cancelled"
                elif inst.status == "running":
                    inst.status = "completed"

        if inst is None:
            raise KeyError(f"Instance not found: {instance_id}")

        self._teardown(inst, join_agent=True)
        self._emit_event(
            EVENT_INSTANCE_DESTROYED,
            message=f"Instance destroyed: {inst.label}",
            data={
                "instance_type": inst.instance_type,
                "instance_id": instance_id,
            },
        )

    def _teardown(self, inst: ManagedInstance, *, join_agent: bool) -> None:
        """Release what an instance holds: end an FSM's conversations, stop a
        workflow engine, signal an agent. ``end_instance`` and
        ``close`` share it; only ``end_instance`` waits for the agent."""
        if isinstance(inst, ManagedFSM):
            # End every conversation while the global handlers are still
            # active, then detach them (see ``_close_fsm_api``).
            self._close_fsm_api(inst, inst.instance_id)
        elif isinstance(inst, ManagedWorkflow):
            # Stop the engine's timers and background runs on the loop that
            # owns them (this may run in a worker thread).
            engine, loop = inst.engine, inst.loop
            if engine is not None and loop is not None and not loop.is_closed():
                try:
                    running = asyncio.get_running_loop()
                except RuntimeError:
                    running = None
                if running is loop:
                    task = loop.create_task(engine.close())
                    _pending_tasks.add(task)
                    task.add_done_callback(_pending_tasks.discard)
                else:
                    asyncio.run_coroutine_threadsafe(engine.close(), loop)
        else:
            inst.cancel_event.set()
            # DECISION plan-2026-09-12T065608-089d0ec7/D-007
            # `agent.run()` has no cancellation hook (see base/agent.py's `run()`
            # abstract signature) so setting cancel_event cannot interrupt
            # in-flight work — do NOT assume the background thread stops
            # here. Bound our OWN wait instead: join with a short timeout so
            # end_instance (called synchronously from an async FastAPI
            # route) never blocks indefinitely, and log a warning if the
            # thread is still alive after the timeout so the leak stays
            # observable. True mid-run cancellation remains OUT OF SCOPE —
            # see decisions.md D-007.
            if join_agent and inst.thread is not None and inst.thread.is_alive():
                inst.thread.join(timeout=1.5)
                if inst.thread.is_alive():
                    logger.warning(
                        f"Agent {inst.instance_id} did not stop within timeout "
                        "after cancellation; background thread still running"
                    )

    def _make_room(self) -> None:
        """Keep at most ``max_instances`` instances: evict the oldest
        finished ones, and refuse a launch when everything is still active."""
        with self._lock:
            excess = len(self._instances) + 1 - self._config.max_instances
            if excess <= 0:
                return
            finished = sorted(
                (
                    inst
                    for inst in self._instances.values()
                    if inst.status in TERMINAL_STATUSES
                ),
                key=lambda i: i.created_at,
            )
            victims = [i.instance_id for i in finished[:excess]]
        for victim in victims:
            try:
                self.end_instance(victim)
            except KeyError:
                pass
        self._check_instance_room()

    def _check_instance_room(self) -> None:
        """Refuse a launch when ``max_instances`` instances are listed."""
        with self._lock:
            if len(self._instances) >= self._config.max_instances:
                raise MonitorCapacityError(
                    f"{len(self._instances)} instances are active "
                    f"(max_instances={self._config.max_instances}); destroy one first"
                )
