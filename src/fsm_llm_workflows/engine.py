"""
Workflow engine for executing workflow definitions.
"""

from __future__ import annotations

import asyncio
import collections
import inspect
import uuid
from collections.abc import Callable
from concurrent.futures import Executor
from datetime import datetime, timedelta, timezone
from typing import Any

from fsm_llm.constants import has_internal_prefix
from fsm_llm.handlers import HandlerSystem
from fsm_llm.logging import logger

# --------------------------------------------------------------
# local imports
# --------------------------------------------------------------
from .constants import (
    DEFAULT_MAX_COMPLETED_INSTANCES,
    KEY_CANCELLATION_REASON,
    KEY_LAST_EVENT,
    KEY_TIMEOUT,
    KEY_TIMER_EXPIRED,
    KEY_TIMER_INFO,
    KEY_USER_INPUT,
    KEY_WAITING_INFO,
    KEY_WORKFLOW_INFO,
    MAX_BUFFERED_EVENTS_PER_INSTANCE,
    MAX_STEP_DEPTH,
    MAX_STEPS_PER_RUN,
    STEP_INTERNAL_WHITELIST,
)
from .definitions import WorkflowDefinition
from .exceptions import (
    WorkflowDefinitionError,
    WorkflowEventError,
    WorkflowInstanceError,
    WorkflowResourceError,
    WorkflowStateError,
    WorkflowStepError,
    WorkflowTimeoutError,
)
from .models import (
    EventListener,
    WorkflowEvent,
    WorkflowInstance,
    WorkflowStatus,
    WorkflowStepResult,
)
from .steps import _STEP_EXECUTOR

# Backwards-compatible module-level names (the constants live in constants.py).
_KEY_WAITING_INFO = KEY_WAITING_INFO
_KEY_TIMER_INFO = KEY_TIMER_INFO
_KEY_WORKFLOW_INFO = KEY_WORKFLOW_INFO
_KEY_TIMEOUT = KEY_TIMEOUT
_KEY_TIMER_EXPIRED = KEY_TIMER_EXPIRED
_KEY_LAST_EVENT = KEY_LAST_EVENT
_KEY_USER_INPUT = KEY_USER_INPUT
_KEY_CANCELLATION_REASON = KEY_CANCELLATION_REASON
_STEP_INTERNAL_WHITELIST = STEP_INTERNAL_WHITELIST

__all__ = ["MAX_STEP_DEPTH", "MAX_STEPS_PER_RUN", "Timer", "WorkflowEngine"]

#: Signature of a lifecycle hook: ``hook(event_name, instance, data)``.
LifecycleHook = Callable[[str, WorkflowInstance, dict[str, Any]], Any]

# --------------------------------------------------------------


class Timer:
    """Information about a timer scheduled for a workflow instance."""

    def __init__(
        self,
        instance_id: str,
        next_state: str | None,
        expires_at: datetime,
        task: asyncio.Task | None = None,
    ):
        self.instance_id = instance_id
        self.next_state = next_state
        self.expires_at = expires_at
        self.task = task

    def is_expired(self) -> bool:
        """Check if the timer has expired."""
        return datetime.now(timezone.utc) > self.expires_at

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


class WorkflowEngine:
    """Engine for executing workflow definitions.

    The engine runs its own async state machine: it executes an instance's
    steps back to back until one pauses (event or timer wait), ends, or
    fails. Everything is in memory.

    Args:
        handler_system: Accepted for backwards compatibility and stored as
            ``self.handler_system``; the engine does not call it. Use
            ``add_hook`` to observe instances.
        max_concurrent_workflows: Maximum number of active (RUNNING/WAITING)
            instances.
        max_completed_instances: Maximum number of terminal instances kept in
            memory (oldest purged first). ``None`` keeps all of them.
        max_steps_per_run: Maximum steps one engine call may execute before
            the instance is FAILED (guards custom steps with dynamic routing;
            static synchronous cycles are rejected at registration).
        executor: Executor for synchronous step callables. ``None`` uses the
            event loop's default executor.
    """

    def __init__(
        self,
        handler_system: HandlerSystem | None = None,
        max_concurrent_workflows: int = 100,
        max_completed_instances: int | None = DEFAULT_MAX_COMPLETED_INSTANCES,
        max_steps_per_run: int = MAX_STEPS_PER_RUN,
        executor: Executor | None = None,
    ):
        """Initialize the workflow engine."""
        if max_steps_per_run < 1:
            raise ValueError("max_steps_per_run must be >= 1")
        self.handler_system = handler_system or HandlerSystem()

        # Configuration
        self.max_concurrent_workflows = max_concurrent_workflows
        self.max_completed_instances = max_completed_instances
        self.max_steps_per_run = max_steps_per_run
        self.executor = executor

        # Storage
        self.workflow_definitions: dict[str, WorkflowDefinition] = {}
        self.workflow_instances: dict[str, WorkflowInstance] = {}
        self.event_listeners: dict[str, dict[str, EventListener]] = {}
        self.timers: dict[str, Timer] = {}
        self._listener_lock = asyncio.Lock()
        self._instance_locks: dict[str, asyncio.Lock] = {}
        # Definition each instance was started with (re-registering a
        # workflow id does not change instances already running).
        self._instance_definitions: dict[str, WorkflowDefinition] = {}
        # Targeted events that arrived before their instance waited for them.
        self._event_buffers: dict[str, collections.deque[WorkflowEvent]] = {}
        self._hooks: list[LifecycleHook] = []
        self._background_tasks: set[asyncio.Task] = set()
        self._closed = False

        logger.info("Workflow engine initialized")

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def add_hook(self, hook: LifecycleHook) -> None:
        """Register ``hook(event_name, instance, data)``.

        Events: ``"step_started"`` (``{"step_id"}``), ``"step_completed"``
        (``{"step_id", "success", "next_state", "error"}``) and
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

    # ------------------------------------------------------------------
    # Locks and status
    # ------------------------------------------------------------------

    def _get_instance_lock(self, instance_id: str) -> asyncio.Lock:
        """Get (creating if absent) the per-instance lock for ``instance_id``.

        # DECISION plan-2026-09-12T065608-089d0ec7/D-003
        # F4 fix: the lock is acquired ONLY at the outermost public entry
        # points (`start_workflow`, `advance_workflow`, `cancel_workflow`,
        # `process_event`'s per-instance delivery loop,
        # `_handle_timer_expiration`, `_handle_event_timeout`,
        # `_handle_deadline`) -- never inside
        # `_execute_workflow_step`/`_transition_to_state` themselves.
        # `_transition_to_state` runs `_execute_workflow_step`, so acquiring
        # this same `asyncio.Lock` at those inner levels would deadlock on
        # the very first reentrant call -- `asyncio.Lock` is NOT reentrant.
        # See decisions.md D-003.
        #
        # Lock-ordering note: several outermost entry points call into a
        # method that acquires `self._listener_lock` internally
        # (`register_event_listener`, `_cleanup_workflow_resources`) while the
        # instance lock is already held. The ordering is consistently
        # instance-lock-then-listener-lock everywhere in this file; never
        # the reverse. Do not add a call path that acquires
        # `self._listener_lock` first and this instance lock second.

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
        """FAIL an instance unless it is already terminal."""
        if instance.is_terminal():
            return
        self._set_status(instance, WorkflowStatus.FAILED, error)

    # ------------------------------------------------------------------
    # Registration and start
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        """Stop the engine: cancel and await every timer and background
        task, drop listeners and buffered events, and refuse new starts.

        Instances keep their current status; WAITING instances can no longer
        be woken by timers.
        """
        self._closed = True
        tasks = [t.task for t in self.timers.values() if t.task is not None]
        tasks.extend(self._background_tasks)
        for timer in self.timers.values():
            timer.cancel()
        for task in self._background_tasks:
            task.cancel()
        self.timers.clear()
        current = asyncio.current_task()
        pending = [t for t in tasks if t is not current and not t.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._background_tasks.clear()
        async with self._listener_lock:
            self.event_listeners.clear()
        self._event_buffers.clear()

    def register_workflow(self, workflow: WorkflowDefinition) -> None:
        """Validate and register a workflow definition.

        The engine stores a copy (with its own ``steps`` dict), so later
        ``with_step`` calls on ``workflow`` do not change the registered
        definition. Re-registering an id replaces the definition for NEW
        instances only; running instances keep the one they started with.
        """
        workflow.validate()
        registered = workflow.model_copy(update={"steps": dict(workflow.steps)})
        if workflow.workflow_id in self.workflow_definitions:
            logger.info(
                f"Re-registering workflow {workflow.workflow_id}: running "
                "instances keep their original definition"
            )
        self.workflow_definitions[workflow.workflow_id] = registered
        logger.info(f"Registered workflow: {workflow.workflow_id}")

    async def start_workflow(
        self,
        workflow_id: str,
        initial_context: dict[str, Any] | None = None,
        instance_id: str | None = None,
        workflow_timeout: float | None = None,
        wait: bool = True,
    ) -> str:
        """Start a new workflow instance.

        Args:
            workflow_id: ID of the registered workflow definition.
            initial_context: Initial context data for the workflow (copied).
            instance_id: Optional custom instance ID (generated if omitted).
                Must not belong to an instance the engine still holds.
            workflow_timeout: Optional total time limit in seconds for the
                entire workflow execution, waits included. ``None`` means no
                limit.
            wait: ``True`` (default) runs the steps before returning.
                ``False`` returns the id immediately and runs them in a
                background task.

        Raises:
            WorkflowResourceError: the engine is shut down or the concurrent
                limit is reached.
            WorkflowInstanceError: ``instance_id`` is already in use.
            WorkflowTimeoutError: the run exceeded ``workflow_timeout``; its
                ``instance_id`` attribute names the (FAILED) instance.
        """
        if self._closed:
            raise WorkflowResourceError(
                resource_type="workflow_engine",
                resource_id="shutdown",
                message="Engine has been shut down",
            )
        # Check concurrent workflow limit
        active_workflows = len(
            [i for i in self.workflow_instances.values() if i.is_active()]
        )
        if active_workflows >= self.max_concurrent_workflows:
            raise WorkflowResourceError(
                resource_type="workflow_engine",
                resource_id="concurrent_limit",
                message=f"Maximum concurrent workflows ({self.max_concurrent_workflows}) exceeded",
            )

        # Get and validate workflow definition
        workflow_def = self._get_workflow_definition(workflow_id)

        # DECISION plan-2026-09-27T120000-5d1e7a3b/D-005
        # Do NOT silently overwrite an existing instance: its timers and
        # listeners are keyed by this id and would then drive the NEW
        # instance (wrong transitions, possibly into a different workflow).
        instance_id = instance_id or str(uuid.uuid4())
        if instance_id in self.workflow_instances:
            raise WorkflowInstanceError(
                instance_id=instance_id,
                message="Instance id already in use (remove_instance it first)",
            )
        instance = self._create_workflow_instance(
            workflow_def, instance_id, initial_context
        )

        # Set deadline for workflow-level timeout
        if workflow_timeout is not None:
            instance.workflow_timeout = workflow_timeout
            instance.deadline = datetime.now(timezone.utc) + timedelta(
                seconds=workflow_timeout
            )

        # Store and execute
        self.workflow_instances[instance_id] = instance
        self._instance_definitions[instance_id] = workflow_def
        log = logger.bind(
            workflow_id=workflow_id,
            instance_id=instance_id,
            package="fsm_llm_workflows",
        )
        log.info(f"Started workflow instance: {instance_id} (workflow: {workflow_id})")

        if instance.deadline is not None:
            self._schedule_deadline(instance)

        if not wait:
            task = asyncio.ensure_future(self._run_in_background(instance))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
            return instance_id

        async with self._get_instance_lock(instance_id):
            await self._execute_workflow_step(instance)
        return instance_id

    async def _run_in_background(self, instance: WorkflowInstance) -> None:
        try:
            async with self._get_instance_lock(instance.instance_id):
                await self._execute_workflow_step(instance)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(
                f"Background run of instance {instance.instance_id} ended with: {e!s}"
            )

    def _get_workflow_definition(self, workflow_id: str) -> WorkflowDefinition:
        """Get a workflow definition, raising an error if not found."""
        if workflow_id not in self.workflow_definitions:
            raise WorkflowDefinitionError(
                workflow_id=workflow_id, message="Workflow definition not found"
            )
        return self.workflow_definitions[workflow_id]

    def _definition_for(self, instance: WorkflowInstance) -> WorkflowDefinition:
        """The definition ``instance`` runs on (pinned at start)."""
        pinned = self._instance_definitions.get(instance.instance_id)
        if pinned is not None:
            return pinned
        return self._get_workflow_definition(instance.workflow_id)

    def _create_workflow_instance(
        self,
        workflow_def: WorkflowDefinition,
        instance_id: str,
        initial_context: dict[str, Any] | None,
    ) -> WorkflowInstance:
        """Create a new workflow instance."""
        if not workflow_def.initial_step_id:
            raise WorkflowDefinitionError(
                workflow_id=workflow_def.workflow_id,
                message="Workflow does not have an initial step defined",
            )

        # Create instance (the caller's dict is copied, never shared)
        instance = WorkflowInstance(
            instance_id=instance_id,
            workflow_id=workflow_def.workflow_id,
            current_step_id=workflow_def.initial_step_id,
            context=dict(initial_context or {}),
        )

        # Add workflow metadata to context
        instance.context[KEY_WORKFLOW_INFO] = {
            "workflow_id": workflow_def.workflow_id,
            "instance_id": instance_id,
        }

        instance.update_status(WorkflowStatus.RUNNING)
        return instance

    # ------------------------------------------------------------------
    # Step driver
    # ------------------------------------------------------------------

    async def _execute_workflow_step(
        self, instance: WorkflowInstance, _depth: int = 0
    ) -> None:
        """Run ``instance`` from its current step until it pauses, ends or fails.

        # DECISION plan-2026-09-27T120000-5d1e7a3b/D-002
        # This is a LOOP with a per-call step budget, not recursion: do NOT
        # go back to `_transition_to_state` -> `_execute_workflow_step`
        # recursion with a depth cap. The old cap (20) failed legitimate
        # acyclic chains longer than 20 steps with an "infinite loop" error,
        # and every exception unwound (and was logged) once per level.

        Step exceptions FAIL the instance and are not raised, except
        ``WorkflowTimeoutError`` (workflow deadline), which is re-raised after
        the instance is FAILED. ``_depth`` is accepted for backwards
        compatibility and counts toward the step budget.
        """
        steps_run = max(0, int(_depth))
        while True:
            if steps_run >= self.max_steps_per_run:
                error = WorkflowStateError(
                    current_state=instance.current_step_id,
                    operation="execute",
                    message=(
                        f"Step budget exceeded: {self.max_steps_per_run} steps in "
                        f"one engine call for instance {instance.instance_id} "
                        "(a custom step may be routing in a loop)"
                    ),
                )
                logger.error(str(error))
                self._fail(instance, error)
                return

            next_state = await self._run_current_step(instance)
            steps_run += 1
            if next_state is None:
                return
            try:
                self._prepare_transition(instance, next_state)
            except Exception as e:
                await self._handle_step_exception(instance, e)
                return

    async def _run_current_step(self, instance: WorkflowInstance) -> str | None:
        """Execute the instance's current step once.

        Returns the state to transition to next, or ``None`` when the run
        stops here (paused, completed, failed).
        """

        # Report the configured workflow_timeout (not deadline-minus-created_at,
        # which is inflated by the construction-to-start gap). Fall back to the
        # deadline span only if the timeout was not recorded.
        # DECISION plan-2026-09-12T065608-089d0ec7/D-011
        # Do NOT re-introduce `int(...)` truncation here: a sub-second
        # `workflow_timeout`/deadline span (e.g. 0.5s) would report as `0` in
        # `WorkflowTimeoutError`'s message, which is actively misleading (the
        # `asyncio.wait_for` call below already receives the correct float
        # timeout — only this reporting helper was truncating). See D-011.
        def _timeout_error() -> WorkflowTimeoutError:
            return WorkflowTimeoutError(
                timeout_seconds=self._timeout_seconds(instance),
                operation=f"step {instance.current_step_id}",
                instance_id=instance.instance_id,
            )

        # Check workflow-level timeout (cheap fast-path at the step boundary)
        if (
            instance.deadline is not None
            and datetime.now(timezone.utc) > instance.deadline
        ):
            error = _timeout_error()
            self._fail(instance, error)
            raise error
        try:
            # Get workflow definition and current step
            workflow_def = self._definition_for(instance)
            current_step = self._get_current_step(
                workflow_def, instance.current_step_id
            )
            step_id = instance.current_step_id

            logger.info(f"Executing step: {step_id} (instance: {instance.instance_id})")
            self._emit("step_started", instance, step_id=step_id)

            # Execute the step. When a workflow-level deadline is set, bound the
            # step itself by the remaining budget so a single long step cannot
            # run unbounded past the deadline.
            token = _STEP_EXECUTOR.set(self.executor)
            try:
                if instance.deadline is not None:
                    remaining = (
                        instance.deadline - datetime.now(timezone.utc)
                    ).total_seconds()
                    if remaining <= 0:
                        raise _timeout_error()
                    try:
                        result = await asyncio.wait_for(
                            current_step.execute(instance.context), timeout=remaining
                        )
                    except asyncio.TimeoutError as e:
                        raise _timeout_error() from e
                else:
                    result = await current_step.execute(instance.context)
            finally:
                _STEP_EXECUTOR.reset(token)

            if not isinstance(result, WorkflowStepResult):
                raise WorkflowStepError(
                    step_id=step_id,
                    message=(
                        f"execute() returned {type(result).__name__}, "
                        "expected a WorkflowStepResult"
                    ),
                )

            # Update context and history (filter internal keys to prevent overwrites).
            # DECISION plan-2026-07-20T040150-876e7164/D-003 [STALE]
            # TWO layers, and both are load-bearing:
            #   1. `has_internal_prefix` is the canonical predicate. Do NOT
            #      re-inline `k.startswith("_")` -- it is case-sensitive and
            #      blind to `system_`/`internal_`/`__`, which is the F-13 leak.
            #   2. `_STEP_INTERNAL_WHITELIST` is a DELIBERATE override on top of
            #      layer 1. Do NOT drop it while "simplifying" the predicate:
            #      _waiting_info and _timer_info must reach the context or
            #      _handle_step_without_transition can no longer detect
            #      waiting/timer steps (invariant I-10). See decisions.md D-003.
            if result.data:
                filtered_data = {
                    k: v
                    for k, v in result.data.items()
                    if not has_internal_prefix(k) or k in _STEP_INTERNAL_WHITELIST
                }
                instance.context.update(filtered_data)

            instance.add_history_entry(
                step_id=step_id,
                message=result.message or "",
                data=result.data,
                error=result.error,
            )
            self._emit(
                "step_completed",
                instance,
                step_id=step_id,
                success=result.success,
                next_state=result.next_state,
                error=result.error,
            )

            # Handle the result
            if result.success:
                return await self._handle_successful_step(instance, result)
            return self._handle_failed_step(instance, result)

        except Exception as e:
            await self._handle_step_exception(instance, e)
            # Propagate workflow-level timeouts to the caller, matching the
            # behavior of the step-boundary deadline check above.
            if isinstance(e, WorkflowTimeoutError):
                raise
            return None

    @staticmethod
    def _timeout_seconds(instance: WorkflowInstance) -> float:
        if instance.workflow_timeout is not None:
            return float(instance.workflow_timeout)
        if instance.deadline is not None:
            return (instance.deadline - instance.created_at).total_seconds()
        return 0.0

    def _get_current_step(self, workflow_def: WorkflowDefinition, step_id: str):
        """Get the current step, raising an error if not found."""
        if step_id not in workflow_def.steps:
            raise WorkflowStateError(
                current_state=step_id,
                operation="get_step",
                message="Step not found in workflow definition",
            )
        return workflow_def.steps[step_id]

    async def _handle_successful_step(
        self, instance: WorkflowInstance, result: WorkflowStepResult, _depth: int = 0
    ) -> str | None:
        """Handle a successful step execution; return the next state or None."""
        if result.next_state:
            return result.next_state
        # DECISION plan-2026-09-12T135914-45a654de/D-013
        # A step (e.g. SwitchStep) can signal "this specific route is
        # terminal" by returning next_state="" -- distinct from a step
        # that has no next_state field at all (also falsy, e.g. a step
        # awaiting an event/timer, whose model default is None). Do NOT
        # re-derive terminality from WorkflowDefinition.get_terminal_states()
        # here: that is a static, WHOLE-STEP analysis (used for
        # reachability warnings) and cannot see that only ONE of a
        # SwitchStep's cases routes to "terminal". This applies to ANY
        # step type whose result carries next_state == "" -- not just
        # SwitchStep -- since ConditionStep, APICallStep,
        # LLMProcessingStep, ParallelStep and RetryStep-wrapped steps all
        # share the same per-route "" == terminal semantics, and
        # definitions.py's own get_terminal_states() already treats ""
        # as a plan-wide terminal-step marker (see
        # `referenced_states.discard("")` in definitions.py).
        #
        # DECISION plan-2026-09-12T135914-45a654de/D-020
        # A prior completion-fix (D-018) narrowed this check to
        # `isinstance(step, SwitchStep)`. A second adversarial review
        # pass found that narrowing REOPENED this exact bug for a
        # RetryStep-wrapped SwitchStep plus 4 other step types. Do NOT
        # reintroduce an isinstance-based narrowing of this predicate --
        # keep the wide `result.next_state == ""` check for ANY
        # successful step result. See decisions.md D-020.
        explicitly_terminal = result.next_state == ""
        return await self._handle_step_without_transition(
            instance, explicitly_terminal=explicitly_terminal
        )

    def _handle_failed_step(
        self, instance: WorkflowInstance, result: WorkflowStepResult, _depth: int = 0
    ) -> str | None:
        """Handle a failed step result: follow its route or FAIL the instance."""
        logger.warning(f"Step failed: {instance.current_step_id} - {result.message}")

        if result.next_state:
            return result.next_state
        message = result.message or "Step failed without error message"
        if result.error and result.error not in message:
            message = f"{message} ({result.error})"
        error = WorkflowStepError(step_id=instance.current_step_id, message=message)
        self._fail(instance, error)
        return None

    async def _handle_step_exception(
        self, instance: WorkflowInstance, exception: Exception
    ) -> None:
        """Handle an exception during step execution."""
        logger.error(f"Error executing step {instance.current_step_id}: {exception!s}")
        # DECISION plan-2026-09-12T065608-089d0ec7/D-005
        # Do NOT call update_status(FAILED, ...) unconditionally here. If the
        # instance is already terminal (e.g. a concurrent cancel_workflow won
        # the race for this instance's lock before this exception path ran),
        # update_status raises WorkflowStateError (terminal states map to an
        # empty transition set in _VALID_STATUS_TRANSITIONS) from inside this
        # except-block handler, which REPLACES the original exception `e` in
        # _execute_workflow_step's except block -- the caller never sees the
        # real error. See decisions.md D-005.
        if instance.is_terminal():
            logger.debug(
                f"Instance {instance.instance_id} already terminal "
                f"({instance.status.value}); not overwriting with FAILED"
            )
            return
        self._set_status(instance, WorkflowStatus.FAILED, exception)

    def _prepare_transition(self, instance: WorkflowInstance, next_state: str) -> None:
        """Validate ``next_state`` and move the instance onto it (RUNNING)."""
        workflow_def = self._definition_for(instance)

        if next_state not in workflow_def.steps:
            raise WorkflowStateError(
                current_state=instance.current_step_id,
                operation="transition",
                message=f"Invalid next state: {next_state}",
            )

        logger.info(f"Transitioning from {instance.current_step_id} to {next_state}")
        instance.context.pop(KEY_WAITING_INFO, None)
        instance.context.pop(KEY_TIMER_INFO, None)
        instance.current_step_id = next_state
        self._set_status(instance, WorkflowStatus.RUNNING)

    async def _transition_to_state(
        self, instance: WorkflowInstance, next_state: str, _depth: int = 0
    ) -> None:
        """Transition to ``next_state`` and run from there.

        Raises ``WorkflowStateError`` for an unknown state (the instance is
        unchanged) and ``WorkflowTimeoutError`` for a deadline breach.
        """
        self._prepare_transition(instance, next_state)
        await self._execute_workflow_step(instance)

    async def _handle_step_without_transition(
        self, instance: WorkflowInstance, explicitly_terminal: bool = False
    ) -> str | None:
        """Handle a step that doesn't specify a next state.

        Returns a state to continue with (a buffered event satisfied a wait)
        or ``None``.

        Args:
            instance: The workflow instance whose current step just ran.
            explicitly_terminal: True when the step's OWN result signalled
                termination for this invocation (``next_state == ""``, e.g. a
                ``SwitchStep`` routed to its terminal branch). When True, the
                instance completes unconditionally, without consulting
                ``WorkflowDefinition.get_terminal_states()``'s static
                whole-step analysis (see D-013 in decisions.md).
        """
        waiting_info = instance.context.get(KEY_WAITING_INFO) or {}
        timer_info = instance.context.get(KEY_TIMER_INFO) or {}

        if explicitly_terminal:
            logger.info(
                f"Workflow instance {instance.instance_id} completed successfully "
                "(step signalled explicit terminal route)"
            )
            self._set_status(instance, WorkflowStatus.COMPLETED)
        elif isinstance(waiting_info, dict) and waiting_info.get("waiting_for_event"):
            event_type = waiting_info.get("event_type") or ""
            success_state = waiting_info.get("success_state") or ""
            correlation_key = waiting_info.get("correlation_key")
            buffered = self._take_buffered_event(instance, event_type, correlation_key)
            if buffered is not None:
                logger.info(
                    f"Workflow instance {instance.instance_id} consumed a buffered "
                    f"'{event_type}' event"
                )
                self._apply_event(instance, waiting_info.get("event_mapping"), buffered)
                if success_state:
                    return success_state
                self._set_status(instance, WorkflowStatus.COMPLETED)
                return None
            logger.info(
                f"Workflow instance {instance.instance_id} is waiting for event"
            )
            self._set_status(instance, WorkflowStatus.WAITING)
            # Auto-register the event listener from step metadata
            await self.register_event_listener(
                instance_id=instance.instance_id,
                event_type=event_type,
                success_state=success_state,
                timeout_seconds=waiting_info.get("timeout_seconds"),
                timeout_state=waiting_info.get("timeout_state"),
                event_mapping=waiting_info.get("event_mapping"),
                correlation_key=correlation_key,
            )
        elif isinstance(timer_info, dict) and timer_info.get("waiting_for_timer"):
            logger.info(
                f"Workflow instance {instance.instance_id} is waiting for timer"
            )
            self._set_status(instance, WorkflowStatus.WAITING)
            # Auto-schedule the timer from step metadata
            await self.schedule_timer(
                instance_id=instance.instance_id,
                delay_seconds=timer_info["delay_seconds"],
                next_state=timer_info["next_state"],
            )
        else:
            # Check if this is a terminal step
            workflow_def = self._definition_for(instance)
            terminal_states = workflow_def.get_terminal_states()

            if instance.current_step_id in terminal_states:
                logger.info(
                    f"Workflow instance {instance.instance_id} completed successfully"
                )
                self._set_status(instance, WorkflowStatus.COMPLETED)
            else:
                logger.warning(
                    f"Step {instance.current_step_id} has no transition and is not "
                    "terminal; the instance stays RUNNING until advance_workflow"
                )
        return None

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    async def register_event_listener(
        self,
        instance_id: str,
        event_type: str,
        success_state: str | None = None,
        timeout_seconds: float | None = None,
        timeout_state: str | None = None,
        event_mapping: dict[str, str] | None = None,
        correlation_key: str | None = None,
    ) -> None:
        """Register a workflow instance to listen for an event.

        ``success_state`` of ``""``/``None`` completes the instance when the
        event arrives. With ``timeout_seconds``, the wait times out to
        ``timeout_state``, or FAILS the instance when none is given.
        ``correlation_key`` restricts delivery to events whose payload value
        for that key equals the instance's context value for it now.
        """
        if not event_type:
            raise WorkflowEventError(event_type="", message="event_type is required")
        instance = self.workflow_instances.get(instance_id)
        if instance is None:
            raise WorkflowInstanceError(
                instance_id=instance_id, message="Workflow instance not found"
            )
        if instance.is_terminal():
            raise WorkflowInstanceError(
                instance_id=instance_id,
                message=f"Instance is {instance.status.value}; cannot listen for events",
            )

        # Create listener
        listener = EventListener(
            instance_id=instance_id,
            success_state=success_state or "",
            event_mapping=event_mapping or {},
            correlation_key=correlation_key,
            correlation_value=(
                instance.context.get(correlation_key) if correlation_key else None
            ),
            timeout_state=timeout_state,
        )

        if timeout_seconds is not None:
            listener.timeout_at = datetime.now(timezone.utc) + timedelta(
                seconds=timeout_seconds
            )

        # Store listener under lock to prevent races with process_event
        async with self._listener_lock:
            if event_type not in self.event_listeners:
                self.event_listeners[event_type] = {}
            self.event_listeners[event_type][instance_id] = listener
        logger.info(
            f"Registered event listener: instance {instance_id} for event {event_type}"
        )

        # DECISION plan-2026-09-27T120000-5d1e7a3b/D-001
        # Schedule the timeout whenever timeout_seconds is set, with or
        # without timeout_state. Do NOT go back to "only with timeout_state":
        # the listener then expired silently, later events were skipped, and
        # the instance stayed WAITING forever. No timeout_state -> FAILED.
        if timeout_seconds is not None:
            await self._schedule_event_timeout(
                instance_id, event_type, timeout_state, timeout_seconds
            )

    async def _schedule_event_timeout(
        self,
        instance_id: str,
        event_type: str,
        timeout_state: str | None,
        timeout_seconds: float,
    ) -> None:
        """Schedule a timeout for an event listener."""
        timeout_task = asyncio.create_task(
            self._event_timeout_task(
                instance_id, event_type, timeout_state, timeout_seconds
            )
        )

        timer_key = f"{instance_id}_{event_type}_timeout"
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=timeout_seconds)
        # Cancel any existing timeout task under this key before replacing it,
        # otherwise re-arming (e.g. re-advancing a WAITING step) orphans the old
        # asyncio.Task, which can still fire and cause a double transition.
        existing = self.timers.get(timer_key)
        if existing is not None:
            existing.cancel()
        self.timers[timer_key] = Timer(
            instance_id, timeout_state, expires_at, timeout_task
        )

        logger.info(f"Set up event timeout: {timeout_seconds} seconds")

    async def _event_timeout_task(
        self,
        instance_id: str,
        event_type: str,
        timeout_state: str | None,
        timeout_seconds: float,
    ) -> None:
        """Task to handle event timeouts."""
        try:
            await asyncio.sleep(timeout_seconds)
            await self._handle_event_timeout(instance_id, event_type, timeout_state)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in event timeout task: {e!s}")

    async def _handle_event_timeout(
        self, instance_id: str, event_type: str, timeout_state: str | None
    ) -> None:
        """Handle an event timeout: go to ``timeout_state`` or FAIL."""
        logger.info(
            f"Event timeout for instance {instance_id} waiting for {event_type}"
        )

        instance = self.workflow_instances.get(instance_id)
        if instance is None:
            return

        # F4: hold the per-instance lock across the status re-check and the
        # transition, so this timeout cannot interleave with a concurrent
        # advance_workflow/cancel_workflow/process_event transition on the
        # same instance (instance-lock, then listener-lock, never the reverse).
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

            # This timer has fired: drop its entry (it is not "active" any more).
            timer_key = f"{instance_id}_{event_type}_timeout"
            timer = self.timers.get(timer_key)
            if timer is not None and timer.task is asyncio.current_task():
                self.timers.pop(timer_key, None)

            async with self._listener_lock:
                self.event_listeners.get(event_type, {}).pop(instance_id, None)

            instance.context[KEY_TIMEOUT] = {
                "event_type": event_type,
                "timeout_at": datetime.now(timezone.utc).isoformat(),
            }

            if not timeout_state:
                wait_seconds = None
                waiting_info = instance.context.get(KEY_WAITING_INFO) or {}
                if isinstance(waiting_info, dict):
                    wait_seconds = waiting_info.get("timeout_seconds")
                self._fail(
                    instance,
                    WorkflowTimeoutError(
                        operation=f"wait for event '{event_type}'",
                        timeout_seconds=float(wait_seconds or 0.0),
                        instance_id=instance_id,
                    ),
                )
                return

            # NOTE (W-ISSUE-004): the step budget restarts here (event-mediated
            # transition); loops through a wait are allowed by design.
            try:
                await self._transition_to_state(instance, timeout_state)
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

        # Collect matching listeners under the listener lock (no awaits
        # inside), consuming them so a second event cannot claim them.
        # DECISION plan-2026-09-27T120000-5d1e7a3b/D-001
        # Do NOT mutate instance context here: delivery happens below under
        # each instance's own lock, after its status is re-checked.
        candidates: list[tuple[str, EventListener]] = []
        async with self._listener_lock:
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
        for instance_id, listener in candidates:
            instance = self.workflow_instances.get(instance_id)
            if instance is None:
                continue
            # DECISION plan-2026-09-27T120000-5d1e7a3b/D-001
            # Per-instance isolation: an exception while delivering to one
            # instance (deadline passed, instance cancelled meanwhile, bad
            # state) must NOT abort delivery to the rest -- their listeners
            # were already consumed above and they would stay WAITING forever.
            try:
                async with self._get_instance_lock(instance_id):
                    if instance.status != WorkflowStatus.WAITING:
                        logger.debug(
                            f"Event '{event_type}' not delivered to {instance_id}: "
                            f"status is {instance.status.value}"
                        )
                        continue
                    # Cancel the wait's timeout BEFORE transitioning: the chain
                    # may re-wait on the same event type and arm a new timeout
                    # under the same key, which a later cancel would kill.
                    self._cancel_event_timeout(instance_id, event_type)
                    self._apply_event(instance, listener.event_mapping, event)
                    affected_instances.append(instance_id)
                    # NOTE (W-ISSUE-004): the step budget restarts here
                    # (event-mediated transition); loops through a wait are
                    # allowed by design.
                    if listener.success_state:
                        await self._transition_to_state(
                            instance, listener.success_state
                        )
                    else:
                        self._set_status(instance, WorkflowStatus.COMPLETED)
            except Exception as e:
                logger.error(
                    f"Delivering event '{event_type}' to {instance_id} failed: {e!s}"
                )
                self._fail(instance, e)

        logger.info(
            f"Processed event {event_type}, affected instances: {len(affected_instances)}"
        )
        return affected_instances

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
        instance.context[KEY_LAST_EVENT] = event.model_dump()

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
        timer_key = f"{instance_id}_{event_type}_timeout"
        timer = self.timers.pop(timer_key, None)
        if timer is not None:
            timer.cancel()

    # ------------------------------------------------------------------
    # Timers and deadlines
    # ------------------------------------------------------------------

    async def schedule_timer(
        self, instance_id: str, delay_seconds: float, next_state: str
    ) -> None:
        """Schedule a timer that moves a WAITING instance to ``next_state``."""
        if instance_id not in self.workflow_instances:
            raise WorkflowInstanceError(
                instance_id=instance_id, message="Workflow instance not found"
            )
        timer_task = asyncio.create_task(
            self._timer_task(instance_id, delay_seconds, next_state)
        )

        expires_at = datetime.now(timezone.utc) + timedelta(seconds=delay_seconds)
        timer_key = f"{instance_id}_timer"
        # Cancel any existing timer under this key before replacing it, otherwise
        # re-arming (e.g. re-advancing a WAITING TimerStep) orphans the old
        # asyncio.Task, which can still fire and cause a double transition.
        existing = self.timers.get(timer_key)
        if existing is not None:
            existing.cancel()
        self.timers[timer_key] = Timer(instance_id, next_state, expires_at, timer_task)

        logger.info(
            f"Scheduled timer for instance {instance_id}: {delay_seconds} seconds"
        )

    async def _timer_task(
        self, instance_id: str, delay_seconds: float, next_state: str
    ) -> None:
        """Task to handle timer expirations."""
        try:
            await asyncio.sleep(delay_seconds)
            await self._handle_timer_expiration(instance_id, next_state)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in timer task: {e!s}")

    async def _handle_timer_expiration(self, instance_id: str, next_state: str) -> None:
        """Handle a timer expiration."""
        logger.info(f"Timer expired for instance {instance_id}")

        instance = self.workflow_instances.get(instance_id)
        if instance is None:
            return

        # F4: per-instance lock, same reasoning as _handle_event_timeout above.
        async with self._get_instance_lock(instance_id):
            # Symmetric guard (RW3-001): if the instance was advanced out of
            # WAITING by some other path (e.g. an external advance_workflow)
            # before this timer fired, do not drive a stale transition.
            if instance.status != WorkflowStatus.WAITING:
                logger.debug(
                    f"Timer for {instance_id} ignored: instance no longer WAITING "
                    f"(status={instance.status.value})"
                )
                return

            instance.context[KEY_TIMER_EXPIRED] = {
                "expired_at": datetime.now(timezone.utc).isoformat()
            }

            # Clean up timer
            timer_key = f"{instance_id}_timer"
            if timer_key in self.timers:
                del self.timers[timer_key]

            # NOTE (W-ISSUE-004): the step budget restarts here (timer-mediated
            # transition); loops through a timer are allowed by design.
            try:
                await self._transition_to_state(instance, next_state)
            except Exception as e:
                self._fail(instance, e)

    def _schedule_deadline(self, instance: WorkflowInstance) -> None:
        """Arm a watchdog that FAILS the instance if it is still WAITING at
        its deadline (a step boundary never comes while it waits)."""
        assert instance.deadline is not None
        delay = max(
            0.0, (instance.deadline - datetime.now(timezone.utc)).total_seconds()
        )
        task = asyncio.ensure_future(self._deadline_task(instance.instance_id, delay))
        self.timers[f"{instance.instance_id}_deadline"] = Timer(
            instance.instance_id, None, instance.deadline, task
        )

    async def _deadline_task(self, instance_id: str, delay: float) -> None:
        try:
            await asyncio.sleep(delay)
            await self._handle_deadline(instance_id)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in deadline task: {e!s}")

    async def _handle_deadline(self, instance_id: str) -> None:
        instance = self.workflow_instances.get(instance_id)
        if instance is None:
            return
        async with self._get_instance_lock(instance_id):
            self.timers.pop(f"{instance_id}_deadline", None)
            if instance.status != WorkflowStatus.WAITING:
                return
            logger.warning(
                f"Workflow instance {instance_id} reached its deadline while WAITING"
            )
            self._fail(
                instance,
                WorkflowTimeoutError(
                    operation=f"step {instance.current_step_id}",
                    timeout_seconds=self._timeout_seconds(instance),
                    instance_id=instance_id,
                ),
            )

    # ------------------------------------------------------------------
    # Advance / cancel / cleanup
    # ------------------------------------------------------------------

    async def advance_workflow(self, instance_id: str, user_input: str = "") -> bool:
        """Re-run an active instance's current step.

        ``user_input`` is placed in the context under ``_user_input`` for the
        duration of this run (``ConversationStep(use_user_input=True)`` sends
        it as a message) and removed afterwards. Re-running a waiting step
        re-arms its wait (and its timeout). Returns False if the instance is
        unknown or not active.
        """
        if instance_id not in self.workflow_instances:
            return False

        instance = self.workflow_instances[instance_id]

        # F4: hold the per-instance lock across the is_active() check AND the
        # step execution, so a concurrent cancel_workflow/advance_workflow on
        # the same instance cannot interleave with this call's mutation of
        # instance.context/status (re-check is_active() under the lock).
        async with self._get_instance_lock(instance_id):
            if not instance.is_active():
                return False

            if user_input:
                instance.context[KEY_USER_INPUT] = user_input
            try:
                await self._execute_workflow_step(instance)
            finally:
                instance.context.pop(KEY_USER_INPUT, None)
        return True

    async def cancel_workflow(
        self, instance_id: str, reason: str = "Cancelled by user"
    ) -> bool:
        """Cancel a workflow instance.

        :returns: ``True`` if this call transitioned the instance to
            ``CANCELLED``. ``False`` if the instance id is unknown, or the
            instance had already reached a terminal status by the time this
            call acquired the per-instance lock (see D-015 -- a benign no-op,
            not an error).
        """
        if instance_id not in self.workflow_instances:
            return False

        instance = self.workflow_instances[instance_id]

        # F4: see advance_workflow's comment -- same per-instance lock,
        # held across the status mutation and resource cleanup.
        async with self._get_instance_lock(instance_id):
            # DECISION plan-2026-09-12T065608-089d0ec7/D-015
            # Do NOT call update_status(CANCELLED, ...) unconditionally here.
            # Since F4/D-003, this lock can be won only AFTER a concurrent
            # in-flight step already finished and committed a terminal status
            # (e.g. COMPLETED) for this same instance. update_status(CANCELLED)
            # on an already-terminal instance raises WorkflowStateError, which
            # would propagate out of this `-> bool` API instead of the plain
            # False it returns for every other "nothing to cancel" case. See
            # decisions.md D-015.
            if instance.is_terminal():
                logger.debug(
                    f"Instance {instance_id} already terminal "
                    f"({instance.status.value}); cancel_workflow is a no-op"
                )
                return False
            instance.context[KEY_CANCELLATION_REASON] = reason
            self._set_status(instance, WorkflowStatus.CANCELLED)

            # Clean up resources (already released by _set_status; kept so the
            # listener lock is observed on this path too)
            await self._cleanup_workflow_resources(instance_id)

        logger.info(f"Workflow instance {instance_id} cancelled: {reason}")
        return True

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

    async def _cleanup_workflow_resources(self, instance_id: str) -> None:
        """Clean up resources for a workflow instance (under the listener lock)."""
        async with self._listener_lock:
            self._release_instance_resources(instance_id)

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

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_workflow_instance(self, instance_id: str) -> WorkflowInstance | None:
        """Get a workflow instance by ID (the live object)."""
        return self.workflow_instances.get(instance_id)

    def get_workflow_definition(self, workflow_id: str) -> WorkflowDefinition | None:
        """Get a registered workflow definition by ID."""
        return self.workflow_definitions.get(workflow_id)

    def get_workflow_status(self, instance_id: str) -> WorkflowStatus | None:
        """Get the status of a workflow instance."""
        instance = self.get_workflow_instance(instance_id)
        return instance.status if instance else None

    def get_workflow_context(self, instance_id: str) -> dict[str, Any] | None:
        """Get a shallow copy of a workflow instance's context."""
        instance = self.get_workflow_instance(instance_id)
        return dict(instance.context) if instance else None

    def get_active_workflows(self) -> list[str]:
        """Get a list of active workflow instance IDs."""
        return [
            instance_id
            for instance_id, instance in self.workflow_instances.items()
            if instance.is_active()
        ]

    def get_statistics(self) -> dict[str, Any]:
        """Get workflow engine statistics."""
        statuses: dict[str, int] = {}
        for instance in self.workflow_instances.values():
            status = instance.status.value
            statuses[status] = statuses.get(status, 0) + 1

        return {
            "total_workflows": len(self.workflow_instances),
            "active_workflows": len(self.get_active_workflows()),
            "registered_definitions": len(self.workflow_definitions),
            "event_listeners": sum(
                len(listeners) for listeners in self.event_listeners.values()
            ),
            "active_timers": sum(
                1 for t in self.timers.values() if t.task is None or not t.task.done()
            ),
            "buffered_events": sum(len(b) for b in self._event_buffers.values()),
            "status_breakdown": statuses,
        }


# --------------------------------------------------------------
