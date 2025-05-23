"""
Workflow engine for executing workflow definitions using LLM-FSM.
"""

import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from .definitions import WorkflowDefinition
from .models import WorkflowInstance, WorkflowStatus, WorkflowEvent, EventListener
from .exceptions import (
    WorkflowInstanceError, WorkflowDefinitionError, WorkflowStepError,
    WorkflowStateError, WorkflowResourceError
)
from llm_fsm.logging import logger
from llm_fsm.fsm import FSMManager
from llm_fsm.handler_system import HandlerSystem


class Timer:
    """Information about a timer scheduled for a workflow instance."""

    def __init__(self, instance_id: str, next_state: str, expires_at: datetime, task: Optional[asyncio.Task] = None):
        self.instance_id = instance_id
        self.next_state = next_state
        self.expires_at = expires_at
        self.task = task

    def is_expired(self) -> bool:
        """Check if the timer has expired."""
        return datetime.now() > self.expires_at

    def cancel(self) -> None:
        """Cancel the timer task."""
        if self.task and not self.task.done():
            self.task.cancel()


class WorkflowEngine:
    """Engine for executing workflow definitions using LLM-FSM."""

    def __init__(self, fsm_manager: Optional[FSMManager] = None, llm_interface=None,
                 handler_system: Optional[HandlerSystem] = None, max_concurrent_workflows: int = 100):
        """Initialize the workflow engine."""
        # Initialize FSM components if not provided
        if not fsm_manager:
            if not llm_interface:
                raise ValueError("Either fsm_manager or llm_interface must be provided")

            self.handler_system = handler_system or HandlerSystem()
            self.fsm_manager = FSMManager(
                llm_interface=llm_interface,
                handler_system=self.handler_system
            )
        else:
            self.fsm_manager = fsm_manager
            self.handler_system = fsm_manager.handler_system

        # Configuration
        self.max_concurrent_workflows = max_concurrent_workflows

        # Storage
        self.workflow_definitions: Dict[str, WorkflowDefinition] = {}
        self.workflow_instances: Dict[str, WorkflowInstance] = {}
        self.conversation_map: Dict[str, str] = {}
        self.event_listeners: Dict[str, Dict[str, EventListener]] = {}
        self.timers: Dict[str, Timer] = {}

        # Register workflow handlers
        self._register_workflow_handlers()

        logger.info("Workflow engine initialized")

    def _register_workflow_handlers(self) -> None:
        """Register handlers for workflow execution."""
        from .handlers import AutoTransitionHandler, EventHandler, TimerHandler

        # Register handlers
        self.handler_system.register_handler(
            AutoTransitionHandler(self, self.fsm_manager)
        )
        self.handler_system.register_handler(
            EventHandler(self, self.fsm_manager)
        )
        self.handler_system.register_handler(
            TimerHandler(self, self.fsm_manager)
        )

    def register_workflow(self, workflow: WorkflowDefinition) -> None:
        """Register a workflow definition."""
        # Validate the workflow
        workflow.validate()

        # Store the workflow
        self.workflow_definitions[workflow.workflow_id] = workflow
        logger.info(f"Registered workflow: {workflow.workflow_id}")

    async def start_workflow(self, workflow_id: str, initial_context: Optional[Dict[str, Any]] = None,
                             instance_id: Optional[str] = None) -> str:
        """Start a new workflow instance."""
        # Check concurrent workflow limit
        active_workflows = len([i for i in self.workflow_instances.values() if i.is_active()])
        if active_workflows >= self.max_concurrent_workflows:
            raise WorkflowResourceError(
                resource_type="workflow_engine",
                resource_id="concurrent_limit",
                message=f"Maximum concurrent workflows ({self.max_concurrent_workflows}) exceeded"
            )

        # Get and validate workflow definition
        workflow_def = self._get_workflow_definition(workflow_id)

        # Create instance
        instance_id = instance_id or str(uuid.uuid4())
        instance = self._create_workflow_instance(workflow_def, instance_id, initial_context)

        # Store and execute
        self.workflow_instances[instance_id] = instance
        logger.info(f"Started workflow instance: {instance_id} (workflow: {workflow_id})")

        await self._execute_workflow_step(instance)
        return instance_id

    def _get_workflow_definition(self, workflow_id: str) -> WorkflowDefinition:
        """Get a workflow definition, raising an error if not found."""
        if workflow_id not in self.workflow_definitions:
            raise WorkflowDefinitionError(
                workflow_id=workflow_id,
                message="Workflow definition not found"
            )
        return self.workflow_definitions[workflow_id]

    def _create_workflow_instance(self, workflow_def: WorkflowDefinition, instance_id: str,
                                  initial_context: Optional[Dict[str, Any]]) -> WorkflowInstance:
        """Create a new workflow instance."""
        if not workflow_def.initial_step_id:
            raise WorkflowDefinitionError(
                workflow_id=workflow_def.workflow_id,
                message="Workflow does not have an initial step defined"
            )

        # Create instance
        instance = WorkflowInstance(
            instance_id=instance_id,
            workflow_id=workflow_def.workflow_id,
            current_step_id=workflow_def.initial_step_id,
            context=initial_context or {}
        )

        # Add workflow metadata to context
        instance.context["_workflow_info"] = {
            "workflow_id": workflow_def.workflow_id,
            "instance_id": instance_id,
            "auto_transition_states": {
                step_id: step.__class__.__name__ == "AutoTransitionStep"
                for step_id, step in workflow_def.steps.items()
            }
        }

        instance.update_status(WorkflowStatus.RUNNING)
        return instance

    async def _execute_workflow_step(self, instance: WorkflowInstance) -> None:
        """Execute the current step in a workflow."""
        try:
            # Get workflow definition and current step
            workflow_def = self._get_workflow_definition(instance.workflow_id)
            current_step = self._get_current_step(workflow_def, instance.current_step_id)

            logger.info(f"Executing step: {instance.current_step_id} (instance: {instance.instance_id})")

            # Execute the step
            result = await current_step.execute(instance.context)

            # Update context and history
            if result.data:
                instance.context.update(result.data)

            instance.add_history_entry(
                step_id=instance.current_step_id,
                message=result.message or "",
                data=result.data
            )

            # Handle the result
            if result.success:
                await self._handle_successful_step(instance, result)
            else:
                await self._handle_failed_step(instance, result)

        except Exception as e:
            await self._handle_step_exception(instance, e)

    def _get_current_step(self, workflow_def: WorkflowDefinition, step_id: str):
        """Get the current step, raising an error if not found."""
        if step_id not in workflow_def.steps:
            raise WorkflowStateError(
                current_state=step_id,
                operation="get_step",
                message="Step not found in workflow definition"
            )
        return workflow_def.steps[step_id]

    async def _handle_successful_step(self, instance: WorkflowInstance, result) -> None:
        """Handle a successful step execution."""
        if result.next_state:
            await self._transition_to_state(instance, result.next_state)
        else:
            await self._handle_step_without_transition(instance)

    async def _handle_failed_step(self, instance: WorkflowInstance, result) -> None:
        """Handle a failed step execution."""
        logger.warning(f"Step failed: {instance.current_step_id} - {result.message}")

        if result.next_state:
            await self._transition_to_state(instance, result.next_state)
        else:
            error = WorkflowStepError(
                step_id=instance.current_step_id,
                message=result.message or "Step failed without error message"
            )
            instance.update_status(WorkflowStatus.FAILED, error)

    async def _handle_step_exception(self, instance: WorkflowInstance, exception: Exception) -> None:
        """Handle an exception during step execution."""
        logger.error(f"Error executing step {instance.current_step_id}: {str(exception)}")
        instance.update_status(WorkflowStatus.FAILED, exception)

    async def _transition_to_state(self, instance: WorkflowInstance, next_state: str) -> None:
        """Transition to a new state."""
        workflow_def = self._get_workflow_definition(instance.workflow_id)

        if next_state not in workflow_def.steps:
            raise WorkflowStateError(
                current_state=instance.current_step_id,
                operation="transition",
                message=f"Invalid next state: {next_state}"
            )

        logger.info(f"Transitioning from {instance.current_step_id} to {next_state}")
        instance.current_step_id = next_state
        instance.update_status(WorkflowStatus.RUNNING)

        await self._execute_workflow_step(instance)

    async def _handle_step_without_transition(self, instance: WorkflowInstance) -> None:
        """Handle a step that doesn't specify a next state."""
        waiting_info = instance.context.get("_waiting_info", {})
        timer_info = instance.context.get("_timer_info", {})

        if waiting_info.get("waiting_for_event") or timer_info.get("waiting_for_timer"):
            logger.info(f"Workflow instance {instance.instance_id} is waiting")
            instance.update_status(WorkflowStatus.WAITING)
        else:
            # Check if this is a terminal step
            workflow_def = self._get_workflow_definition(instance.workflow_id)
            terminal_states = workflow_def.get_terminal_states()

            if instance.current_step_id in terminal_states:
                logger.info(f"Workflow instance {instance.instance_id} completed successfully")
                instance.update_status(WorkflowStatus.COMPLETED)
            else:
                logger.warning(f"Step {instance.current_step_id} has no transition and is not terminal")

    async def register_event_listener(self, instance_id: str, event_type: str, success_state: Optional[str] = None,
                                      timeout_seconds: Optional[int] = None, timeout_state: Optional[str] = None,
                                      event_mapping: Optional[Dict[str, str]] = None) -> None:
        """Register a workflow instance to listen for an event."""
        if instance_id not in self.workflow_instances:
            raise WorkflowInstanceError(
                instance_id=instance_id,
                message="Workflow instance not found"
            )

        # Initialize event listeners for this event type
        if event_type not in self.event_listeners:
            self.event_listeners[event_type] = {}

        # Create listener
        listener = EventListener(
            instance_id=instance_id,
            success_state=success_state or "",
            event_mapping=event_mapping or {}
        )

        if timeout_seconds:
            listener.timeout_at = datetime.now() + timedelta(seconds=timeout_seconds)

        # Store listener
        self.event_listeners[event_type][instance_id] = listener
        logger.info(f"Registered event listener: instance {instance_id} for event {event_type}")

        # Set up timeout if needed
        if timeout_seconds and timeout_state:
            await self._schedule_event_timeout(instance_id, event_type, timeout_state, timeout_seconds)

    async def _schedule_event_timeout(self, instance_id: str, event_type: str, timeout_state: str,
                                      timeout_seconds: int) -> None:
        """Schedule a timeout for an event listener."""
        timeout_task = asyncio.create_task(
            self._event_timeout_task(instance_id, event_type, timeout_state, timeout_seconds)
        )

        timer_key = f"{instance_id}_{event_type}_timeout"
        expires_at = datetime.now() + timedelta(seconds=timeout_seconds)
        self.timers[timer_key] = Timer(instance_id, timeout_state, expires_at, timeout_task)

        logger.info(f"Set up event timeout: {timeout_seconds} seconds")

    async def _event_timeout_task(self, instance_id: str, event_type: str, timeout_state: str,
                                  timeout_seconds: int) -> None:
        """Task to handle event timeouts."""
        try:
            await asyncio.sleep(timeout_seconds)
            await self._handle_event_timeout(instance_id, event_type, timeout_state)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in event timeout task: {str(e)}")

    async def _handle_event_timeout(self, instance_id: str, event_type: str, timeout_state: str) -> None:
        """Handle an event timeout."""
        logger.info(f"Event timeout for instance {instance_id} waiting for {event_type}")

        if instance_id not in self.workflow_instances:
            return

        # Remove listener
        if event_type in self.event_listeners and instance_id in self.event_listeners[event_type]:
            del self.event_listeners[event_type][instance_id]

        # Update instance
        instance = self.workflow_instances[instance_id]
        instance.context["_timeout"] = {
            "event_type": event_type,
            "timeout_at": datetime.now().isoformat()
        }

        await self._transition_to_state(instance, timeout_state)

    async def schedule_timer(self, instance_id: str, delay_seconds: int, next_state: str) -> None:
        """Schedule a timer for a workflow instance."""
        timer_task = asyncio.create_task(
            self._timer_task(instance_id, delay_seconds, next_state)
        )

        expires_at = datetime.now() + timedelta(seconds=delay_seconds)
        timer_key = f"{instance_id}_timer"
        self.timers[timer_key] = Timer(instance_id, next_state, expires_at, timer_task)

        logger.info(f"Scheduled timer for instance {instance_id}: {delay_seconds} seconds")

    async def _timer_task(self, instance_id: str, delay_seconds: int, next_state: str) -> None:
        """Task to handle timer expirations."""
        try:
            await asyncio.sleep(delay_seconds)
            await self._handle_timer_expiration(instance_id, next_state)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in timer task: {str(e)}")

    async def _handle_timer_expiration(self, instance_id: str, next_state: str) -> None:
        """Handle a timer expiration."""
        logger.info(f"Timer expired for instance {instance_id}")

        if instance_id not in self.workflow_instances:
            return

        # Update instance
        instance = self.workflow_instances[instance_id]
        instance.context["_timer_expired"] = {
            "expired_at": datetime.now().isoformat()
        }

        # Clean up timer
        timer_key = f"{instance_id}_timer"
        if timer_key in self.timers:
            del self.timers[timer_key]

        await self._transition_to_state(instance, next_state)

    async def process_event(self, event: WorkflowEvent) -> List[str]:
        """Process an external event."""
        event_type = event.event_type

        if event_type not in self.event_listeners:
            logger.debug(f"No listeners for event type: {event_type}")
            return []

        affected_instances = []

        # Process event for each listener
        for instance_id, listener in list(self.event_listeners[event_type].items()):
            if instance_id not in self.workflow_instances:
                continue

            # Update instance context
            instance = self.workflow_instances[instance_id]

            # Map event payload to context
            for context_key, payload_key in listener.event_mapping.items():
                if payload_key in event.payload:
                    instance.context[context_key] = event.payload[payload_key]

            instance.context["_last_event"] = event.model_dump()

            # Transition to success state
            if listener.success_state:
                await self._transition_to_state(instance, listener.success_state)

                # Clean up listener
                del self.event_listeners[event_type][instance_id]

                # Cancel timeout
                self._cancel_event_timeout(instance_id, event_type)

                affected_instances.append(instance_id)

        logger.info(f"Processed event {event_type}, affected instances: {len(affected_instances)}")
        return affected_instances

    def _cancel_event_timeout(self, instance_id: str, event_type: str) -> None:
        """Cancel an event timeout."""
        timer_key = f"{instance_id}_{event_type}_timeout"
        if timer_key in self.timers:
            self.timers[timer_key].cancel()
            del self.timers[timer_key]

    async def advance_workflow(self, instance_id: str, user_input: str = "") -> bool:
        """Advance a workflow instance."""
        if instance_id not in self.workflow_instances:
            return False

        instance = self.workflow_instances[instance_id]

        if not instance.is_active():
            return False

        if user_input:
            instance.context["_user_input"] = user_input

        await self._execute_workflow_step(instance)
        return True

    async def cancel_workflow(self, instance_id: str, reason: str = "Cancelled by user") -> bool:
        """Cancel a workflow instance."""
        if instance_id not in self.workflow_instances:
            return False

        instance = self.workflow_instances[instance_id]
        instance.update_status(WorkflowStatus.CANCELLED)
        instance.context["_cancellation_reason"] = reason

        # Clean up resources
        self._cleanup_workflow_resources(instance_id)

        logger.info(f"Workflow instance {instance_id} cancelled: {reason}")
        return True

    def _cleanup_workflow_resources(self, instance_id: str) -> None:
        """Clean up resources for a workflow instance."""
        # Cancel timers
        for timer_key in list(self.timers.keys()):
            if timer_key.startswith(f"{instance_id}_"):
                self.timers[timer_key].cancel()
                del self.timers[timer_key]

        # Remove event listeners
        for event_type in list(self.event_listeners.keys()):
            if instance_id in self.event_listeners[event_type]:
                del self.event_listeners[event_type][instance_id]

    # Getter methods
    def get_workflow_instance(self, instance_id: str) -> Optional[WorkflowInstance]:
        """Get a workflow instance by ID."""
        return self.workflow_instances.get(instance_id)

    def get_workflow_definition(self, workflow_id: str) -> Optional[WorkflowDefinition]:
        """Get a workflow definition by ID."""
        return self.workflow_definitions.get(workflow_id)

    def get_workflow_status(self, instance_id: str) -> Optional[WorkflowStatus]:
        """Get the status of a workflow instance."""
        instance = self.get_workflow_instance(instance_id)
        return instance.status if instance else None

    def get_workflow_context(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """Get the context of a workflow instance."""
        instance = self.get_workflow_instance(instance_id)
        return instance.context if instance else None

    def get_active_workflows(self) -> List[str]:
        """Get a list of active workflow instance IDs."""
        return [
            instance_id for instance_id, instance in self.workflow_instances.items()
            if instance.is_active()
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """Get workflow engine statistics."""
        statuses = {}
        for instance in self.workflow_instances.values():
            status = instance.status.value
            statuses[status] = statuses.get(status, 0) + 1

        return {
            "total_workflows": len(self.workflow_instances),
            "active_workflows": len(self.get_active_workflows()),
            "registered_definitions": len(self.workflow_definitions),
            "event_listeners": sum(len(listeners) for listeners in self.event_listeners.values()),
            "active_timers": len(self.timers),
            "status_breakdown": statuses
        }