"""The declarations the mixins of ``WorkflowEngine`` share (mypy only; plan d454fd62 step 91; the old ``workflows/engine.py``, same dotted path).

What a mixin of the old ``fsm_llm.workflows.engine.WorkflowEngine`` may rely on: the attributes the
constructor of the shell sets and the methods the other mixins define, declared for mypy only (a runtime
no-op, like ``agents/meta_builder/surface.py``). A mixin that defines one of these is checked against the
declaration here; one that calls a method of another mixin is checked against it too.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import asyncio
    import collections
    from collections.abc import Callable
    from concurrent.futures import Executor

    from fsm_llm.engine.context import RunContext

    from ..definitions import WorkflowDefinition
    from ..models import (
        EventListener,
        WorkflowEvent,
        WorkflowInstance,
        WorkflowStatus,
    )
    from .helpers import LifecycleHook, Timer, _CancelRequest


class _EngineSurface:
    """What every mixin of ``WorkflowEngine`` may rely on: the attributes of the shell and the methods of the
    other mixins, declared for mypy only (a runtime no-op).
    """

    if TYPE_CHECKING:
        max_concurrent_workflows: int
        max_completed_instances: int | None
        max_steps: int
        executor: Executor | None
        _context: RunContext | None
        workflow_definitions: dict[str, WorkflowDefinition]
        workflow_instances: dict[str, WorkflowInstance]
        event_listeners: dict[str, dict[str, EventListener]]
        timers: dict[str, Timer]
        _instance_locks: dict[str, asyncio.Lock]
        _cancel_requests: dict[str, _CancelRequest]
        _instance_definitions: dict[str, WorkflowDefinition]
        _event_buffers: dict[str, collections.deque[WorkflowEvent]]
        _hooks: list[LifecycleHook]
        _background_tasks: set[asyncio.Task]

        # StatusMixin (status.py)
        def _emit(
            self, event: str, instance: WorkflowInstance, **data: Any
        ) -> None: ...

        def _get_instance_lock(self, instance_id: str) -> asyncio.Lock: ...

        def _set_status(
            self,
            instance: WorkflowInstance,
            status: WorkflowStatus,
            error: Exception | None = None,
        ) -> None: ...

        def _fail(self, instance: WorkflowInstance, error: Exception) -> None: ...

        # LifecycleMixin (lifecycle.py)
        def _definition_for(self, instance: WorkflowInstance) -> WorkflowDefinition: ...

        # DriverMixin (driver.py)
        async def _execute_workflow_step(self, instance: WorkflowInstance) -> None: ...

        @staticmethod
        def _timeout_seconds(instance: WorkflowInstance) -> float: ...

        def _leave_wait(
            self, instance: WorkflowInstance, status: WorkflowStatus
        ) -> None: ...

        async def _transition_to_state(
            self, instance: WorkflowInstance, next_step_id: str
        ) -> None: ...

        # EventsMixin (events.py)
        async def register_event_listener(
            self,
            instance_id: str,
            event_type: str,
            success_step_id: str | None = None,
            timeout_seconds: float | None = None,
            timeout_step_id: str | None = None,
            event_mapping: dict[str, str] | None = None,
            correlation_key: str | None = None,
        ) -> None: ...

        def _apply_event(
            self,
            instance: WorkflowInstance,
            event_mapping: dict[str, str] | None,
            event: WorkflowEvent,
        ) -> None: ...

        def _take_buffered_event(
            self,
            instance: WorkflowInstance,
            event_type: str,
            correlation_key: str | None,
        ) -> WorkflowEvent | None: ...

        # TimersMixin (timers.py)
        def _live_instance(
            self, instance_id: str, purpose: str
        ) -> WorkflowInstance: ...

        def _arm_timer(
            self,
            key: str,
            instance_id: str,
            next_step_id: str | None,
            delay: float,
            handler: Callable[[], Any],
            label: str,
        ) -> None: ...

        def _forget_timer(self, key: str) -> None: ...

        async def schedule_timer(
            self, instance_id: str, delay_seconds: float, next_step_id: str
        ) -> None: ...

        def _arm_deadline(self, instance: WorkflowInstance) -> None: ...

        # ResourcesMixin (resources.py)
        def _drop_wait_resources(self, instance_id: str) -> None: ...

        def _release_instance_resources(self, instance_id: str) -> None: ...

        def _purge_oldest_terminal_instances(self) -> None: ...

        # QueriesMixin (queries.py)
        def list_active_instances(self) -> list[str]: ...
