"""``WorkflowEngine``, the shell over its mixins (plan d454fd62 step 91; the old ``workflows/engine.py``, same dotted path).

The class holds the constructor and the storage; every method lives in one mixin of this package
(``status``, ``lifecycle``, ``driver``, ``events``, ``timers``, ``resources``, ``queries``).
"""

from __future__ import annotations

import asyncio
import collections
from concurrent.futures import Executor

from fsm_llm.engine.context import RunContext
from fsm_llm.logging import logger

from ..constants import DEFAULT_MAX_COMPLETED_INSTANCES, DEFAULT_MAX_STEPS
from ..definitions import WorkflowDefinition
from ..models import EventListener, WorkflowEvent, WorkflowInstance
from .driver import DriverMixin
from .events import EventsMixin
from .helpers import LifecycleHook, Timer, _CancelRequest
from .lifecycle import LifecycleMixin
from .queries import QueriesMixin
from .resources import ResourcesMixin
from .status import StatusMixin
from .timers import TimersMixin


class WorkflowEngine(
    StatusMixin,
    LifecycleMixin,
    DriverMixin,
    EventsMixin,
    TimersMixin,
    ResourcesMixin,
    QueriesMixin,
):
    """Engine for executing workflow definitions.

    The engine runs its own async state machine: it executes an instance's
    steps back to back until one pauses (event or timer wait), ends, or
    fails. Everything is in memory.

    Every argument is keyword-only: the first positional slot held the
    removed ``handler_system`` (plan 07ad3f8c step 18), so an old positional
    call must fail at construction, not later inside ``start``.

    Args:
        max_concurrent_workflows: Maximum number of active (RUNNING/WAITING)
            instances.
        max_completed_instances: Maximum number of terminal instances kept in
            memory (oldest purged first). ``None`` keeps all of them.
        max_steps: Maximum steps one engine call may execute before
            the instance is FAILED (guards custom steps with dynamic routing;
            static synchronous cycles are rejected at registration).
        executor: Executor for synchronous step callables. ``None`` uses the
            event loop's default executor.
        context: A ``RunContext`` (plan 7efd5cb4 D-012). Its cancel token and shared ledger
            are checked before each step (a set token CANCELS the instance, a refused charge
            FAILS it like ``max_steps``) and it is the ambient parent of every
            ``ExecutableStep`` child, per step (a resumed instance has it too). Nothing else
            is read from it: no ``callables``, no limit, no grant.
    """

    def __init__(
        self,
        *,
        max_concurrent_workflows: int = 100,
        max_completed_instances: int | None = DEFAULT_MAX_COMPLETED_INSTANCES,
        max_steps: int = DEFAULT_MAX_STEPS,
        executor: Executor | None = None,
        context: RunContext | None = None,
    ):
        """Initialize the workflow engine."""
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if context is not None and not isinstance(context, RunContext):
            raise TypeError(
                f"context must be a RunContext, got {type(context).__name__}"
            )
        self._context = context

        # Configuration
        self.max_concurrent_workflows = max_concurrent_workflows
        self.max_completed_instances = max_completed_instances
        self.max_steps = max_steps
        self.executor = executor

        # Storage
        self.workflow_definitions: dict[str, WorkflowDefinition] = {}
        self.workflow_instances: dict[str, WorkflowInstance] = {}
        self.event_listeners: dict[str, dict[str, EventListener]] = {}
        self.timers: dict[str, Timer] = {}
        self._instance_locks: dict[str, asyncio.Lock] = {}
        # Cancels waiting for an instance lock, by instance id. Registered by
        # `cancel` BEFORE it awaits the lock, removed in its
        # `finally` (and with the instance); read once per loop iteration.
        self._cancel_requests: dict[str, _CancelRequest] = {}
        # Definition each instance was started with (re-registering a
        # workflow id does not change instances already running).
        self._instance_definitions: dict[str, WorkflowDefinition] = {}
        # Targeted events that arrived before their instance waited for them.
        self._event_buffers: dict[str, collections.deque[WorkflowEvent]] = {}
        self._hooks: list[LifecycleHook] = []
        self._background_tasks: set[asyncio.Task] = set()
        self._closed = False

        logger.info("Workflow engine initialized")
