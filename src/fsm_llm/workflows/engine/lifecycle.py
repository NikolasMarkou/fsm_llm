"""Registration, start, advance, cancel and close of ``WorkflowEngine`` (plan d454fd62 step 91; the old ``workflows/engine.py``, same dotted path).

``LifecycleMixin``: the entry points that take the per-instance lock (``start``, ``_run_in_background``,
``advance``, ``cancel``), the registry of definitions and the creation of an instance. The method bodies are
the old ones, moved.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from fsm_llm.logging import logger

from ..constants import (
    CONTEXT_KEY_CANCELLATION_REASON,
    CONTEXT_KEY_USER_INPUT,
    CONTEXT_KEY_WORKFLOW_INFO,
)
from ..definitions import WorkflowDefinition
from ..exceptions import (
    WorkflowDefinitionError,
    WorkflowInstanceError,
    WorkflowResourceError,
)
from ..models import WorkflowInstance, WorkflowStatus
from .helpers import _CancelRequest
from .surface import _EngineSurface


class LifecycleMixin(_EngineSurface):
    """Registration, start, advance, cancel and close of ``WorkflowEngine``."""

    async def close(self) -> None:
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
        self.event_listeners.clear()
        self._event_buffers.clear()

    def register_workflow(self, workflow: WorkflowDefinition) -> None:
        """Validate and register a workflow definition.

        The engine stores a copy (with its own ``steps`` dict), so later
        ``with_step`` calls on ``workflow`` do not change the registered
        definition. Re-registering an id replaces the definition for NEW
        instances only; running instances keep the one they started with.
        """
        workflow.validate_structure()
        registered = workflow.model_copy(update={"steps": dict(workflow.steps)})
        if workflow.workflow_id in self.workflow_definitions:
            logger.info(
                f"Re-registering workflow {workflow.workflow_id}: running "
                "instances keep their original definition"
            )
        self.workflow_definitions[workflow.workflow_id] = registered
        logger.info(f"Registered workflow: {workflow.workflow_id}")

    async def start(
        self,
        workflow_id: str,
        initial_context: dict[str, Any] | None = None,
        instance_id: str | None = None,
        max_seconds: float | None = None,
        wait: bool = True,
    ) -> str:
        """Start a new workflow instance.

        Args:
            workflow_id: ID of the registered workflow definition.
            initial_context: Initial context data for the workflow (copied).
            instance_id: Optional custom instance ID (generated if omitted).
                Must not belong to an instance the engine still holds.
            max_seconds: Optional total time limit in seconds for the
                entire workflow execution, waits included. ``None`` means no
                limit.
            wait: ``True`` (default) runs the steps before returning.
                ``False`` returns the id immediately and runs them in a
                background task.

        Raises:
            WorkflowResourceError: the engine is closed or the concurrent
                limit is reached.
            WorkflowInstanceError: ``instance_id`` is already in use.
            WorkflowTimeoutError: the run exceeded ``max_seconds``; its
                ``instance_id`` attribute names the (FAILED) instance.
        """
        if self._closed:
            raise WorkflowResourceError(
                "Engine has been closed",
                resource_type="workflow_engine",
                resource_id="closed",
            )
        # Check concurrent workflow limit
        if len(self.list_active_instances()) >= self.max_concurrent_workflows:
            raise WorkflowResourceError(
                f"Maximum concurrent workflows ({self.max_concurrent_workflows}) exceeded",
                resource_type="workflow_engine",
                resource_id="concurrent_limit",
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
                "Instance id already in use (remove_instance it first)",
                instance_id=instance_id,
            )
        instance = self._create_workflow_instance(
            workflow_def, instance_id, initial_context
        )

        # Set deadline for workflow-level timeout
        if max_seconds is not None:
            instance.max_seconds = max_seconds
            instance.deadline = datetime.now(UTC) + timedelta(seconds=max_seconds)

        # Store and execute
        self.workflow_instances[instance_id] = instance
        self._instance_definitions[instance_id] = workflow_def
        self._set_status(instance, WorkflowStatus.RUNNING)
        log = logger.bind(
            workflow_id=workflow_id,
            instance_id=instance_id,
            package="fsm_llm.workflows",
        )
        log.info(f"Started workflow instance: {instance_id} (workflow: {workflow_id})")

        if instance.deadline is not None:
            self._arm_deadline(instance)

        if not wait:
            task = asyncio.ensure_future(
                self._run_in_background(instance, len(instance.history))
            )
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
            return instance_id

        async with self._get_instance_lock(instance_id):
            await self._execute_workflow_step(instance)
        return instance_id

    async def _run_in_background(
        self, instance: WorkflowInstance, history_len: int
    ) -> None:
        try:
            async with self._get_instance_lock(instance.instance_id):
                # Skip if another entry point already drove or cancelled the
                # instance before this task got the lock (its initial step
                # must run once, and never after a cancel).
                if (
                    instance.status != WorkflowStatus.RUNNING
                    or len(instance.history) != history_len
                ):
                    return
                await self._execute_workflow_step(instance)
        except Exception as e:
            logger.error(
                f"Background run of instance {instance.instance_id} ended with: {e!s}"
            )

    def _get_workflow_definition(self, workflow_id: str) -> WorkflowDefinition:
        """Get a workflow definition, raising an error if not found."""
        if workflow_id not in self.workflow_definitions:
            raise WorkflowDefinitionError(
                "Workflow definition not found", workflow_id=workflow_id
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
        """Create a new (PENDING) workflow instance; ``start`` moves it
        to RUNNING once the engine holds it, so hooks hear the change."""
        if not workflow_def.initial_step_id:
            raise WorkflowDefinitionError(
                "Workflow does not have an initial step defined",
                workflow_id=workflow_def.workflow_id,
            )

        # Create instance (the caller's dict is copied, never shared)
        instance = WorkflowInstance(
            instance_id=instance_id,
            workflow_id=workflow_def.workflow_id,
            current_step_id=workflow_def.initial_step_id,
            context=dict(initial_context or {}),
        )

        # Add workflow metadata to context
        instance.context[CONTEXT_KEY_WORKFLOW_INFO] = {
            "workflow_id": workflow_def.workflow_id,
            "instance_id": instance_id,
        }

        return instance

    async def advance(self, instance_id: str, user_input: str = "") -> bool:
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
        # step execution, so a concurrent cancel/advance on
        # the same instance cannot interleave with this call's mutation of
        # instance.context/status (re-check is_active() under the lock).
        async with self._get_instance_lock(instance_id):
            if not instance.is_active():
                return False

            if user_input:
                instance.context[CONTEXT_KEY_USER_INPUT] = user_input
            try:
                await self._execute_workflow_step(instance)
            finally:
                instance.context.pop(CONTEXT_KEY_USER_INPUT, None)
        return True

    async def cancel(self, instance_id: str, reason: str = "Cancelled by user") -> bool:
        """Cancel a workflow instance.

        A cancel of an instance whose chain of steps is running takes effect
        between steps (the step in flight runs to completion, later steps do
        not run); the call returns once the run has stopped.

        :returns: ``True`` if this call (or the run loop, on its request)
            transitioned the instance to ``CANCELLED``. ``False`` if the
            instance id is unknown, or the instance reached a terminal status
            some other way (a natural completion, a failure) by the time the
            run released the per-instance lock (see D-015 -- a benign no-op,
            not an error).
        """
        if instance_id not in self.workflow_instances:
            return False

        instance = self.workflow_instances[instance_id]

        # Registered BEFORE the lock wait: the lock is held by a running chain
        # until it ends, so the chain loop has to see the request (D-017).
        request = self._cancel_requests.setdefault(instance_id, _CancelRequest(reason))
        try:
            # F4: see advance's comment -- same per-instance lock,
            # held across the status mutation and resource cleanup.
            async with self._get_instance_lock(instance_id):
                # DECISION plan-2026-09-12T065608-089d0ec7/D-015
                # Do NOT call update_status(CANCELLED, ...) unconditionally
                # here. Since F4/D-003, this lock can be won only AFTER a
                # concurrent in-flight step already finished and committed a
                # terminal status (e.g. COMPLETED) for this same instance.
                # update_status(CANCELLED) on an already-terminal instance
                # raises WorkflowStateError, which would propagate out of this
                # `-> bool` API instead of the plain False it returns for
                # every other "nothing to cancel" case. See decisions.md D-015.
                # A terminal instance returns `request.honored` (D-017): True
                # when the run loop cancelled it on this request, False for a
                # natural completion.
                if instance.is_terminal():
                    logger.debug(
                        f"Instance {instance_id} already terminal "
                        f"({instance.status.value}); cancel is a no-op"
                    )
                    if not request.honored:
                        return False
                else:
                    instance.context[CONTEXT_KEY_CANCELLATION_REASON] = reason
                    self._set_status(instance, WorkflowStatus.CANCELLED)
        finally:
            if self._cancel_requests.get(instance_id) is request:
                del self._cancel_requests[instance_id]

        logger.info(f"Workflow instance {instance_id} cancelled: {reason}")
        return True
