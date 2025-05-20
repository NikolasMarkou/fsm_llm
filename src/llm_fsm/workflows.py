"""
LLM-FSM Workflow System
=======================

A generic workflow system built on top of LLM-FSM that enables:
- Automated state transitions
- Event-driven workflows
- External system integration
- Parallel workflow execution
- Monitoring and error recovery

This module extends the core LLM-FSM framework to support complex
AI workflows that can operate with or without user interaction.
"""

import uuid
import asyncio
import inspect
import traceback
from enum import Enum
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, TypeVar
from pydantic import BaseModel, Field, validator, root_validator, ConfigDict

# --------------------------------------------------------------
# local imports
# --------------------------------------------------------------


from .logging import logger
from .fsm import FSMManager
from .handler_system import HandlerSystem, BaseHandler, HandlerTiming

# --------------------------------------------------------------

# Type variable for generic models
T = TypeVar('T')


# --------------------------------------------------------------
# Core Models
# --------------------------------------------------------------

class WorkflowStatus(str, Enum):
    """Status of a workflow instance."""
    PENDING = "pending"
    RUNNING = "running"
    WAITING = "waiting"  # Waiting for user input or external event
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowEvent(BaseModel):
    """Represents an event that can trigger workflow transitions."""
    event_type: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat(),
        }
    )


class WorkflowStepResult(BaseModel):
    """Result of a workflow step execution."""
    success: bool
    data: Dict[str, Any] = Field(default_factory=dict)
    next_state: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )

    @root_validator(pre=True)
    def process_error(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Exception objects to error strings."""
        if 'error' in values and isinstance(values['error'], Exception):
            values['error'] = str(values['error'])
        return values


class WorkflowStep(BaseModel):
    """Base class for workflow steps."""
    step_id: str
    name: str
    description: str = ""

    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )

    async def execute(self, context: Dict[str, Any]) -> WorkflowStepResult:
        """
        Execute the step. Override in subclasses.

        Args:
            context: Workflow context data

        Returns:
            Result of the step execution
        """
        raise NotImplementedError("Subclasses must implement execute()")


class AutoTransitionStep(WorkflowStep):
    """A step that automatically transitions to the next state without user input."""
    next_state: str
    action: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None

    async def execute(self, context: Dict[str, Any]) -> WorkflowStepResult:
        """
        Execute the step, perform the action if any, and transition.

        Args:
            context: Workflow context data

        Returns:
            Result with next_state set
        """
        try:
            data = {}
            if self.action:
                # Execute the action
                if inspect.iscoroutinefunction(self.action):
                    data = await self.action(context)
                else:
                    data = self.action(context)

            return WorkflowStepResult(
                success=True,
                data=data,
                next_state=self.next_state,
                message=f"Auto-transitioned to {self.next_state}"
            )
        except Exception as e:
            logger.error(f"Error in auto transition step {self.step_id}: {str(e)}")
            return WorkflowStepResult(
                success=False,
                message=f"Auto-transition failed: {str(e)}",
                error=str(e)
            )


class APICallStep(WorkflowStep):
    """A step that calls an external API and processes the result."""
    api_function: Callable
    success_state: str
    failure_state: str
    input_mapping: Dict[str, str] = Field(default_factory=dict)
    output_mapping: Dict[str, str] = Field(default_factory=dict)

    async def execute(self, context: Dict[str, Any]) -> WorkflowStepResult:
        """
        Execute the API call and process the result.

        Args:
            context: Workflow context data

        Returns:
            Result with next_state set based on API call outcome
        """
        try:
            # Prepare API parameters from context
            params = {}
            for api_param, context_key in self.input_mapping.items():
                if context_key in context:
                    params[api_param] = context[context_key]

            # Call the API
            if inspect.iscoroutinefunction(self.api_function):
                api_result = await self.api_function(**params)
            else:
                api_result = self.api_function(**params)

            # Process the result and update context
            output_data = {}
            if isinstance(api_result, dict):
                for context_key, result_key in self.output_mapping.items():
                    if result_key in api_result:
                        output_data[context_key] = api_result[result_key]

            return WorkflowStepResult(
                success=True,
                data=output_data,
                next_state=self.success_state,
                message=f"API call successful, transitioning to {self.success_state}"
            )
        except Exception as e:
            logger.error(f"Error in API call step {self.step_id}: {str(e)}")
            return WorkflowStepResult(
                success=False,
                next_state=self.failure_state,
                message=f"API call failed: {str(e)}",
                error=str(e)
            )


class ConditionStep(WorkflowStep):
    """A step that evaluates a condition and transitions accordingly."""
    condition: Callable[[Dict[str, Any]], bool]
    true_state: str
    false_state: str

    async def execute(self, context: Dict[str, Any]) -> WorkflowStepResult:
        """
        Evaluate the condition and determine the next state.

        Args:
            context: Workflow context data

        Returns:
            Result with next_state set based on condition evaluation
        """
        try:
            # Evaluate the condition
            if inspect.iscoroutinefunction(self.condition):
                result = await self.condition(context)
            else:
                result = self.condition(context)

            # Determine the next state
            next_state = self.true_state if result else self.false_state

            return WorkflowStepResult(
                success=True,
                next_state=next_state,
                message=f"Condition evaluated to {result}, transitioning to {next_state}"
            )
        except Exception as e:
            logger.error(f"Error in condition step {self.step_id}: {str(e)}")
            return WorkflowStepResult(
                success=False,
                message=f"Condition evaluation failed: {str(e)}",
                error=str(e)
            )


class LLMProcessingStep(WorkflowStep):
    """A step that processes data using an LLM and transitions based on the result."""
    llm_interface: Any  # Cannot type this directly due to circular imports
    prompt_template: str
    context_mapping: Dict[str, str]
    output_mapping: Dict[str, str]
    next_state: str
    error_state: Optional[str] = None

    async def execute(self, context: Dict[str, Any]) -> WorkflowStepResult:
        """
        Process data with the LLM and determine the next state.

        Args:
            context: Workflow context data

        Returns:
            Result with next_state set based on LLM processing
        """
        try:
            # Prepare the prompt
            prompt_vars = {}
            for prompt_var, context_key in self.context_mapping.items():
                if context_key in context:
                    prompt_vars[prompt_var] = context[context_key]

            # Format the prompt template
            prompt = self.prompt_template.format(**prompt_vars)

            # Call the LLM
            llm_response = await self.llm_interface.generate(prompt)

            # Process the result and update context
            output_data = {}
            # Simplified parsing - in practice you'd use more robust methods
            for context_key, output_pattern in self.output_mapping.items():
                # Implement parsing logic based on your needs
                # This is a placeholder
                output_data[context_key] = llm_response

            return WorkflowStepResult(
                success=True,
                data=output_data,
                next_state=self.next_state,
                message=f"LLM processing successful, transitioning to {self.next_state}"
            )
        except Exception as e:
            logger.error(f"Error in LLM processing step {self.step_id}: {str(e)}")
            next_state = self.error_state if self.error_state else self.next_state
            return WorkflowStepResult(
                success=False,
                next_state=next_state,
                message=f"LLM processing failed: {str(e)}",
                error=str(e)
            )


class WaitEventConfig(BaseModel):
    """Configuration for waiting for an event."""
    event_type: str
    success_state: str
    timeout_seconds: Optional[int] = None
    timeout_state: Optional[str] = None
    event_mapping: Dict[str, str] = Field(default_factory=dict)


class WaitForEventStep(WorkflowStep):
    """A step that waits for an external event before transitioning."""
    config: WaitEventConfig

    async def execute(self, context: Dict[str, Any]) -> WorkflowStepResult:
        """
        Set up the workflow to wait for an event.

        This doesn't actually wait - it just registers the required information
        in the context so the workflow engine knows what event to wait for.

        Args:
            context: Workflow context data

        Returns:
            Result indicating the workflow is waiting for an event
        """
        # Set up the waiting state
        waiting_info = {
            "waiting_for_event": True,
            "event_type": self.config.event_type,
            "timeout_seconds": self.config.timeout_seconds,
            "timeout_state": self.config.timeout_state,
            "success_state": self.config.success_state,
            "event_mapping": self.config.event_mapping,
            "waiting_since": datetime.now().isoformat()
        }

        return WorkflowStepResult(
            success=True,
            data={"_waiting_info": waiting_info},
            message=f"Waiting for event of type {self.config.event_type}"
        )


class TimerStep(WorkflowStep):
    """A step that waits for a specified time before transitioning."""
    delay_seconds: int
    next_state: str

    async def execute(self, context: Dict[str, Any]) -> WorkflowStepResult:
        """
        Set up a timer to transition after a delay.

        This doesn't actually wait - it just registers the required information
        in the context so the workflow engine knows when to transition.

        Args:
            context: Workflow context data

        Returns:
            Result indicating the workflow is waiting for a timer
        """
        # Set up the timer
        timer_info = {
            "waiting_for_timer": True,
            "delay_seconds": self.delay_seconds,
            "next_state": self.next_state,
            "timer_start": datetime.now().isoformat(),
            "timer_end": (datetime.now() + timedelta(seconds=self.delay_seconds)).isoformat()
        }

        return WorkflowStepResult(
            success=True,
            data={"_timer_info": timer_info},
            message=f"Timer set for {self.delay_seconds} seconds, will transition to {self.next_state}"
        )


class ParallelStep(WorkflowStep):
    """A step that executes multiple steps in parallel and aggregates results."""
    steps: List[WorkflowStep]
    next_state: str
    error_state: Optional[str] = None
    aggregation_function: Optional[Callable[[List[WorkflowStepResult]], Dict[str, Any]]] = None

    async def execute(self, context: Dict[str, Any]) -> WorkflowStepResult:
        """
        Execute multiple steps in parallel and aggregate results.

        Args:
            context: Workflow context data

        Returns:
            Aggregated result from all parallel steps
        """
        try:
            # Execute all steps in parallel
            tasks = [step.execute(context) for step in self.steps]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Check for exceptions
            errors = [r for r in results if isinstance(r, Exception)]
            if errors:
                error_msgs = [str(e) for e in errors]
                if self.error_state:
                    return WorkflowStepResult(
                        success=False,
                        next_state=self.error_state,
                        message=f"Parallel step errors: {'; '.join(error_msgs)}",
                        error=error_msgs[0]
                    )

            # Aggregate results
            aggregated_data = {}
            if self.aggregation_function:
                # Filter out exceptions before passing to the aggregation function
                valid_results = [r for r in results if isinstance(r, WorkflowStepResult)]
                aggregated_data = self.aggregation_function(valid_results)
            else:
                # Simple default aggregation
                for i, result in enumerate(results):
                    if isinstance(result, WorkflowStepResult) and result.data:
                        prefix = f"step_{i}_"
                        for key, value in result.data.items():
                            aggregated_data[f"{prefix}{key}"] = value

            return WorkflowStepResult(
                success=all(
                    not isinstance(r, Exception) and r.success for r in results if isinstance(r, WorkflowStepResult)),
                data=aggregated_data,
                next_state=self.next_state,
                message=f"Parallel step completed, transitioning to {self.next_state}"
            )
        except Exception as e:
            logger.error(f"Error in parallel step {self.step_id}: {str(e)}")
            next_state = self.error_state if self.error_state else self.next_state
            return WorkflowStepResult(
                success=False,
                next_state=next_state,
                message=f"Parallel step failed: {str(e)}",
                error=str(e)
            )


class WorkflowDefinition(BaseModel):
    """Definition of a workflow."""
    workflow_id: str
    name: str
    description: str = ""
    steps: Dict[str, WorkflowStep] = Field(default_factory=dict)
    initial_step_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def with_step(self, step: WorkflowStep, is_initial: bool = False) -> 'WorkflowDefinition':
        """
        Add a step to the workflow.

        Args:
            step: Step to add
            is_initial: Whether this is the initial step

        Returns:
            Self for chaining
        """
        self.steps[step.step_id] = step
        if is_initial:
            self.initial_step_id = step.step_id
        return self

    def with_initial_step(self, step: WorkflowStep) -> 'WorkflowDefinition':
        """
        Add the initial step to the workflow.

        Args:
            step: Initial step to add

        Returns:
            Self for chaining
        """
        return self.with_step(step, is_initial=True)

    @validator('initial_step_id')
    def validate_initial_step(cls, v, values):
        """Validate that the initial step exists in the steps dictionary."""
        if v is not None and 'steps' in values and v not in values['steps']:
            raise ValueError(f"Initial step '{v}' not found in the steps dictionary")
        return v


class WorkflowHistoryEntry(BaseModel):
    """An entry in the workflow execution history."""
    timestamp: datetime = Field(default_factory=datetime.now)
    step_id: str
    status: str
    message: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class WorkflowInstance(BaseModel):
    """Runtime instance of a workflow."""
    instance_id: str
    workflow_id: str
    current_step_id: str
    context: Dict[str, Any] = Field(default_factory=dict)
    status: WorkflowStatus = WorkflowStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    history: List[WorkflowHistoryEntry] = Field(default_factory=list)

    def update_status(self, status: WorkflowStatus, error: Optional[Exception] = None) -> None:
        """
        Update the workflow status.

        Args:
            status: New status
            error: Optional error if the workflow failed
        """
        self.status = status
        self.updated_at = datetime.now()
        if status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]:
            self.completed_at = datetime.now()
        if error:
            self.error = str(error)

        # Add to history
        self.history.append(WorkflowHistoryEntry(
            timestamp=self.updated_at,
            status=status.value,
            step_id=self.current_step_id,
            error=str(error) if error else None
        ))

    def add_history_entry(self, step_id: str, message: str, data: Dict[str, Any] = None) -> None:
        """
        Add an entry to the workflow history.

        Args:
            step_id: ID of the step being executed
            message: Message describing the event
            data: Optional data associated with the event
        """
        self.history.append(WorkflowHistoryEntry(
            step_id=step_id,
            status=self.status.value,
            message=message,
            data=data or {}
        ))


class EventListener(BaseModel):
    """Information about a workflow instance listening for an event."""
    instance_id: str
    success_state: str
    event_mapping: Dict[str, str] = Field(default_factory=dict)
    registered_at: datetime = Field(default_factory=datetime.now)
    timeout_at: Optional[datetime] = None


class Timer(BaseModel):
    """Information about a timer scheduled for a workflow instance."""
    instance_id: str
    next_state: str
    expires_at: datetime
    task: Optional[asyncio.Task] = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )


# --------------------------------------------------------------
# Handlers
# --------------------------------------------------------------

class AutoTransitionHandler(BaseHandler):
    """Handler that detects and executes automatic transitions in LLM-FSM."""

    def __init__(self, workflow_engine: 'WorkflowEngine', fsm_manager: FSMManager):
        """
        Initialize the auto transition handler.

        Args:
            workflow_engine: WorkflowEngine instance
            fsm_manager: FSMManager instance
        """
        super().__init__(name="AutoTransitionHandler", priority=10)
        self.workflow_engine = workflow_engine
        self.fsm_manager = fsm_manager

    def should_execute(self, timing, current_state, target_state, context, updated_keys=None):
        """
        Determine if this handler should execute.

        Args:
            timing: The timing point in the FSM lifecycle
            current_state: Current FSM state
            target_state: Target state if transitioning
            context: Current context data
            updated_keys: Keys being updated (for CONTEXT_UPDATE timing)

        Returns:
            True if the handler should execute
        """
        # Execute after any state transition
        if timing == HandlerTiming.POST_TRANSITION:
            # Check if this is an auto-transition state
            workflow_info = context.get("_workflow_info", {})
            is_auto_transition = workflow_info.get("auto_transition_states", {}).get(target_state, False)
            return is_auto_transition
        return False

    def execute(self, context):
        """
        Execute the handler logic.

        Args:
            context: Current context data

        Returns:
            Updated context
        """
        # Get the conversation ID from context
        conversation_id = context.get("_conversation_id")
        if not conversation_id:
            logger.error("No conversation ID found in context")
            return {}

        # Get workflow information
        workflow_info = context.get("_workflow_info", {})
        workflow_instance_id = workflow_info.get("instance_id")

        if not workflow_instance_id:
            logger.error("No workflow instance ID found in context")
            return {}

        # Schedule an automatic advance in the workflow
        try:
            asyncio.create_task(
                self.workflow_engine.advance_workflow(workflow_instance_id, "")
            )
            return {"_auto_transition_scheduled": True}
        except Exception as e:
            logger.error(f"Error scheduling auto transition: {str(e)}")
            return {"_auto_transition_error": str(e)}


class EventHandler(BaseHandler):
    """Handler that processes external events in LLM-FSM."""

    def __init__(self, workflow_engine: 'WorkflowEngine', fsm_manager: FSMManager):
        """
        Initialize the event handler.

        Args:
            workflow_engine: WorkflowEngine instance
            fsm_manager: FSMManager instance
        """
        super().__init__(name="EventHandler", priority=20)
        self.workflow_engine = workflow_engine
        self.fsm_manager = fsm_manager

    def should_execute(self, timing, current_state, target_state, context, updated_keys=None):
        """
        Determine if this handler should execute.

        Args:
            timing: The timing point in the FSM lifecycle
            current_state: Current FSM state
            target_state: Target state if transitioning
            context: Current context data
            updated_keys: Keys being updated (for CONTEXT_UPDATE timing)

        Returns:
            True if the handler should execute
        """
        # Execute after transitioning to a state that waits for an event
        if timing == HandlerTiming.POST_TRANSITION:
            waiting_info = context.get("_waiting_info", {})
            return waiting_info.get("waiting_for_event", False)
        return False

    def execute(self, context):
        """
        Execute the handler logic.

        Args:
            context: Current context data

        Returns:
            Updated context
        """
        # Get the workflow information
        workflow_info = context.get("_workflow_info", {})
        workflow_instance_id = workflow_info.get("instance_id")

        if not workflow_instance_id:
            logger.error("No workflow instance ID found in context")
            return {}

        # Get event waiting information
        waiting_info = context.get("_waiting_info", {})
        event_type = waiting_info.get("event_type")

        if not event_type:
            logger.error("No event type specified in waiting info")
            return {}

        # Register the workflow as waiting for this event
        try:
            self.workflow_engine.register_event_listener(
                workflow_instance_id,
                event_type,
                waiting_info.get("success_state"),
                waiting_info.get("timeout_seconds"),
                waiting_info.get("timeout_state"),
                waiting_info.get("event_mapping", {})
            )
            return {"_event_listener_registered": True}
        except Exception as e:
            logger.error(f"Error registering event listener: {str(e)}")
            return {"_event_listener_error": str(e)}


class TimerHandler(BaseHandler):
    """Handler that processes timers in LLM-FSM."""

    def __init__(self, workflow_engine: 'WorkflowEngine', fsm_manager: FSMManager):
        """
        Initialize the timer handler.

        Args:
            workflow_engine: WorkflowEngine instance
            fsm_manager: FSMManager instance
        """
        super().__init__(name="TimerHandler", priority=30)
        self.workflow_engine = workflow_engine
        self.fsm_manager = fsm_manager

    def should_execute(self, timing, current_state, target_state, context, updated_keys=None):
        """
        Determine if this handler should execute.

        Args:
            timing: The timing point in the FSM lifecycle
            current_state: Current FSM state
            target_state: Target state if transitioning
            context: Current context data
            updated_keys: Keys being updated (for CONTEXT_UPDATE timing)

        Returns:
            True if the handler should execute
        """
        # Execute after transitioning to a state that sets a timer
        if timing == HandlerTiming.POST_TRANSITION:
            timer_info = context.get("_timer_info", {})
            return timer_info.get("waiting_for_timer", False)
        return False

    def execute(self, context):
        """
        Execute the handler logic.

        Args:
            context: Current context data

        Returns:
            Updated context
        """
        # Get the workflow information
        workflow_info = context.get("_workflow_info", {})
        workflow_instance_id = workflow_info.get("instance_id")

        if not workflow_instance_id:
            logger.error("No workflow instance ID found in context")
            return {}

        # Get timer information
        timer_info = context.get("_timer_info", {})
        delay_seconds = timer_info.get("delay_seconds")
        next_state = timer_info.get("next_state")

        if not delay_seconds or not next_state:
            logger.error("Invalid timer information")
            return {}

        # Schedule the timer
        try:
            asyncio.create_task(
                self.workflow_engine.schedule_timer(
                    workflow_instance_id,
                    delay_seconds,
                    next_state
                )
            )
            return {"_timer_scheduled": True}
        except Exception as e:
            logger.error(f"Error scheduling timer: {str(e)}")
            return {"_timer_error": str(e)}


# --------------------------------------------------------------
# Main Workflow Engine
# --------------------------------------------------------------

class WorkflowEngine:
    """
    Engine for executing workflow definitions using LLM-FSM.
    """

    def __init__(
            self,
            fsm_manager: Optional[FSMManager] = None,
            llm_interface=None,
            handler_system: Optional[HandlerSystem] = None
    ):
        """
        Initialize the workflow engine.

        Args:
            fsm_manager: Optional FSMManager instance
            llm_interface: Optional LLM interface
            handler_system: Optional handler system
        """
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

        # Dictionary of workflow definitions by ID
        self.workflow_definitions: Dict[str, WorkflowDefinition] = {}

        # Dictionary of workflow instances by ID
        self.workflow_instances: Dict[str, WorkflowInstance] = {}

        # Mapping of conversation IDs to workflow instances
        self.conversation_map: Dict[str, str] = {}

        # Dictionary of event listeners
        self.event_listeners: Dict[str, Dict[str, EventListener]] = {}

        # Dictionary of timers
        self.timers: Dict[str, Timer] = {}

        # Register workflow handlers
        self._register_workflow_handlers()

        logger.info("Workflow engine initialized")

    def _register_workflow_handlers(self) -> None:
        """Register handlers for workflow execution."""
        # Auto-transition handler
        self.handler_system.register_handler(
            AutoTransitionHandler(self, self.fsm_manager)
        )

        # Event handler
        self.handler_system.register_handler(
            EventHandler(self, self.fsm_manager)
        )

        # Timer handler
        self.handler_system.register_handler(
            TimerHandler(self, self.fsm_manager)
        )

    def register_workflow(self, workflow: WorkflowDefinition) -> None:
        """
        Register a workflow definition.

        Args:
            workflow: Workflow definition to register
        """
        self.workflow_definitions[workflow.workflow_id] = workflow
        logger.info(f"Registered workflow: {workflow.workflow_id}")

    async def start_workflow(
            self,
            workflow_id: str,
            initial_context: Optional[Dict[str, Any]] = None,
            instance_id: Optional[str] = None
    ) -> str:
        """
        Start a new workflow instance.

        Args:
            workflow_id: ID of the workflow definition
            initial_context: Optional initial context data
            instance_id: Optional instance ID

        Returns:
            The workflow instance ID

        Raises:
            ValueError: If the workflow definition is not found
        """
        # Check if the workflow exists
        if workflow_id not in self.workflow_definitions:
            raise ValueError(f"Workflow definition not found: {workflow_id}")

        # Get the workflow definition
        workflow_def = self.workflow_definitions[workflow_id]

        # Validate the workflow definition
        if not workflow_def.initial_step_id:
            raise ValueError(f"Workflow {workflow_id} does not have an initial step defined")

        # Create an instance ID if not provided
        instance_id = instance_id or str(uuid.uuid4())

        # Create a new workflow instance
        instance = WorkflowInstance(
            instance_id=instance_id,
            workflow_id=workflow_id,
            current_step_id=workflow_def.initial_step_id,
            context=initial_context or {}
        )

        # Add workflow information to the context
        instance.context["_workflow_info"] = {
            "workflow_id": workflow_id,
            "instance_id": instance_id,
            "auto_transition_states": {
                step_id: isinstance(step, AutoTransitionStep)
                for step_id, step in workflow_def.steps.items()
            }
        }

        # Update the status
        instance.update_status(WorkflowStatus.RUNNING)

        # Store the instance
        self.workflow_instances[instance_id] = instance

        logger.info(f"Started workflow instance: {instance_id} (workflow: {workflow_id})")

        # Execute the initial step
        await self._execute_workflow_step(instance)

        return instance_id

    async def _execute_workflow_step(self, instance: WorkflowInstance) -> None:
        """
        Execute the current step in a workflow.

        Args:
            instance: Workflow instance to execute
        """
        # Get the workflow definition and current step
        if not await self._validate_workflow_step(instance):
            return

        step = await self._get_current_step(instance)
        if not step:
            return

        logger.info(f"Executing step: {instance.current_step_id} (instance: {instance.instance_id})")

        try:
            # Execute the step
            result = await step.execute(instance.context)

            # Update context and history with result
            await self._update_context_and_history(instance, result)

            # Handle step result based on success/failure
            if not result.success:
                await self._handle_step_failure(instance, result)
                return

            # Process next steps based on result
            await self._process_step_result(instance, result)

        except Exception as e:
            await self._handle_step_exception(instance, e)

    async def _validate_workflow_step(self, instance: WorkflowInstance) -> bool:
        """
        Validate that the workflow and step exist.

        Args:
            instance: Workflow instance to validate

        Returns:
            True if valid, False otherwise
        """
        # Check if the workflow definition exists
        workflow_id = instance.workflow_id
        if workflow_id not in self.workflow_definitions:
            error_msg = f"Workflow definition not found: {workflow_id}"
            logger.error(error_msg)
            instance.update_status(WorkflowStatus.FAILED, ValueError(error_msg))
            return False

        # Check if the current step exists
        step_id = instance.current_step_id
        workflow_def = self.workflow_definitions[workflow_id]

        if not step_id or step_id not in workflow_def.steps:
            error_msg = f"Step not found: {step_id}"
            logger.error(error_msg)
            instance.update_status(WorkflowStatus.FAILED, ValueError(error_msg))
            return False

        return True

    async def _get_current_step(self, instance: WorkflowInstance) -> Optional[WorkflowStep]:
        """
        Get the current step for a workflow instance.

        Args:
            instance: Workflow instance

        Returns:
            The current step or None if not found
        """
        workflow_def = self.workflow_definitions[instance.workflow_id]
        return workflow_def.steps.get(instance.current_step_id)

    async def _update_context_and_history(self, instance: WorkflowInstance, result: WorkflowStepResult) -> None:
        """
        Update the instance context and history with step result.

        Args:
            instance: Workflow instance to update
            result: Result of the step execution
        """
        # Update the context with step result data
        if result.data:
            instance.context.update(result.data)

        # Add to history
        instance.add_history_entry(
            step_id=instance.current_step_id,
            message=result.message or "",
            data=result.data
        )

    async def _handle_step_failure(self, instance: WorkflowInstance, result: WorkflowStepResult) -> None:
        """
        Handle a failed step execution.

        Args:
            instance: Workflow instance
            result: Result of the step execution
        """
        logger.warning(f"Step failed: {instance.current_step_id} - {result.message}")

        # If there's an error and a next state is specified, transition to it
        if result.next_state:
            # Validate the next state
            workflow_def = self.workflow_definitions[instance.workflow_id]
            if result.next_state not in workflow_def.steps:
                error_msg = f"Invalid next state in error handler: {result.next_state}"
                logger.error(error_msg)
                instance.update_status(WorkflowStatus.FAILED, ValueError(error_msg))
                return

            # Transition to the error handler state
            logger.info(f"Transitioning to error handler state: {result.next_state}")
            instance.current_step_id = result.next_state
            instance.update_status(WorkflowStatus.RUNNING)

            # Execute the error handler step
            await self._execute_workflow_step(instance)
        else:
            # No error handler state specified, mark as failed
            error = ValueError(result.message) if not result.error else ValueError(result.error)
            instance.update_status(WorkflowStatus.FAILED, error)

    async def _handle_step_exception(self, instance: WorkflowInstance, exception: Exception) -> None:
        """
        Handle an exception during step execution.

        Args:
            instance: Workflow instance
            exception: The exception that occurred
        """
        error_msg = f"Error executing step {instance.current_step_id}: {str(exception)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        instance.update_status(WorkflowStatus.FAILED, exception)

    async def _process_step_result(self, instance: WorkflowInstance, result: WorkflowStepResult) -> None:
        """
        Process the result of a successful step execution.

        Args:
            instance: Workflow instance
            result: Result of the step execution
        """
        # Check if a next state was specified
        if result.next_state:
            # Validate the next state
            workflow_def = self.workflow_definitions[instance.workflow_id]
            if result.next_state not in workflow_def.steps:
                error_msg = f"Invalid next state: {result.next_state}"
                logger.error(error_msg)
                instance.update_status(WorkflowStatus.FAILED, ValueError(error_msg))
                return

            # Transition to the next state
            logger.info(f"Transitioning from {instance.current_step_id} to {result.next_state}")
            instance.current_step_id = result.next_state
            instance.update_status(WorkflowStatus.RUNNING)

            # Execute the next step
            await self._execute_workflow_step(instance)
        else:
            # Check if this is a waiting step
            waiting_info = instance.context.get("_waiting_info", {})
            timer_info = instance.context.get("_timer_info", {})

            if waiting_info.get("waiting_for_event") or timer_info.get("waiting_for_timer"):
                logger.info(f"Workflow instance {instance.instance_id} is waiting for an external event or timer")
                instance.update_status(WorkflowStatus.WAITING)
            else:
                # Check if this is a terminal step (no transitions)
                workflow_def = self.workflow_definitions[instance.workflow_id]
                current_step = workflow_def.steps[instance.current_step_id]

                # A step is terminal if it has no defined next_state and is not a waiting step
                is_terminal = True

                if isinstance(current_step, (AutoTransitionStep, TimerStep, LLMProcessingStep)):
                    is_terminal = False

                if is_terminal:
                    logger.info(f"Workflow instance {instance.instance_id} completed successfully")
                    instance.update_status(WorkflowStatus.COMPLETED)

    async def register_event_listener(
            self,
            instance_id: str,
            event_type: str,
            success_state: Optional[str] = None,
            timeout_seconds: Optional[int] = None,
            timeout_state: Optional[str] = None,
            event_mapping: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Register a workflow instance to listen for an event.

        Args:
            instance_id: Workflow instance ID
            event_type: Type of event to listen for
            success_state: State to transition to when event received
            timeout_seconds: Optional timeout in seconds
            timeout_state: State to transition to on timeout
            event_mapping: Map event payload to context keys
        """
        # Initialize the event listeners for this event type if needed
        if event_type not in self.event_listeners:
            self.event_listeners[event_type] = {}

        # Create the event listener
        listener = EventListener(
            instance_id=instance_id,
            success_state=success_state or "",
            event_mapping=event_mapping or {},
        )

        # Set timeout if specified
        if timeout_seconds:
            listener.timeout_at = datetime.now() + timedelta(seconds=timeout_seconds)

        # Store the listener
        self.event_listeners[event_type][instance_id] = listener

        logger.info(f"Registered event listener: instance {instance_id} for event {event_type}")

        # Set up a timeout if specified
        if timeout_seconds and timeout_state:
            # Schedule the timeout task
            timeout_task = asyncio.create_task(
                self._handle_timeout_task(instance_id, event_type, timeout_state, timeout_seconds)
            )

            # Store task reference to prevent garbage collection
            timer_key = f"{instance_id}_{event_type}_timeout"
            self.timers[timer_key] = Timer(
                instance_id=instance_id,
                next_state=timeout_state,
                expires_at=datetime.now() + timedelta(seconds=timeout_seconds),
                task=timeout_task
            )

            logger.info(f"Set up timeout for event listener: {timeout_seconds} seconds")

    async def _handle_timeout_task(
            self,
            instance_id: str,
            event_type: str,
            timeout_state: str,
            timeout_seconds: int
    ) -> None:
        """
        Task to handle timeouts for event listeners.

        Args:
            instance_id: Workflow instance ID
            event_type: Event type
            timeout_state: State to transition to on timeout
            timeout_seconds: Timeout duration in seconds
        """
        try:
            # Wait for the timeout period
            await asyncio.sleep(timeout_seconds)

            # Check if the instance still exists and is waiting for this event
            await self.handle_timeout(instance_id, event_type, timeout_state)
        except asyncio.CancelledError:
            # Task was cancelled, do nothing
            pass
        except Exception as e:
            logger.error(f"Error in timeout task: {str(e)}")

    async def handle_timeout(self, instance_id: str, event_type: str, timeout_state: str) -> None:
        """
        Handle a timeout for an event listener.

        Args:
            instance_id: Workflow instance ID
            event_type: Event type being waited for
            timeout_state: State to transition to
        """
        logger.info(f"Timeout for instance {instance_id} waiting for event {event_type}")

        # Check if the instance exists
        if instance_id not in self.workflow_instances:
            logger.warning(f"Instance not found: {instance_id}")
            return

        # Check if the instance is still waiting for this event
        if event_type in self.event_listeners and instance_id in self.event_listeners[event_type]:
            # Remove the listener
            del self.event_listeners[event_type][instance_id]

            # Update the instance
            instance = self.workflow_instances[instance_id]
            instance.current_step_id = timeout_state
            instance.context["_timeout"] = {
                "event_type": event_type,
                "timeout_at": datetime.now().isoformat()
            }

            # Execute the timeout step
            await self._execute_workflow_step(instance)

    async def schedule_timer(self, instance_id: str, delay_seconds: int, next_state: str) -> None:
        """
        Schedule a timer for a workflow instance.

        Args:
            instance_id: Workflow instance ID
            delay_seconds: Delay in seconds
            next_state: State to transition to after the delay
        """
        # Create a timer task
        timer_task = asyncio.create_task(
            self._timer_task(instance_id, delay_seconds, next_state)
        )

        # Store the timer
        expires_at = datetime.now() + timedelta(seconds=delay_seconds)
        timer_key = f"{instance_id}_timer"
        self.timers[timer_key] = Timer(
            instance_id=instance_id,
            next_state=next_state,
            expires_at=expires_at,
            task=timer_task
        )

        logger.info(f"Scheduled timer for instance {instance_id}: {delay_seconds} seconds")

    async def _timer_task(self, instance_id: str, delay_seconds: int, next_state: str) -> None:
        """
        Task to handle timer expirations.

        Args:
            instance_id: Workflow instance ID
            delay_seconds: Delay duration in seconds
            next_state: State to transition to
        """
        try:
            # Wait for the specified delay
            await asyncio.sleep(delay_seconds)

            # Handle the timer expiration
            await self.handle_timer_expiration(instance_id, next_state)
        except asyncio.CancelledError:
            # Task was cancelled, do nothing
            pass
        except Exception as e:
            logger.error(f"Error in timer task: {str(e)}")

    async def handle_timer_expiration(self, instance_id: str, next_state: str) -> None:
        """
        Handle a timer expiration.

        Args:
            instance_id: Workflow instance ID
            next_state: State to transition to
        """
        logger.info(f"Timer expired for instance {instance_id}")

        # Check if the instance exists
        if instance_id not in self.workflow_instances:
            logger.warning(f"Instance not found: {instance_id}")
            return

        # Update the instance
        instance = self.workflow_instances[instance_id]
        instance.current_step_id = next_state
        instance.context["_timer_expired"] = {
            "expired_at": datetime.now().isoformat()
        }

        # Clean up the timer
        timer_key = f"{instance_id}_timer"
        if timer_key in self.timers:
            del self.timers[timer_key]

        # Execute the next step
        await self._execute_workflow_step(instance)

    async def process_event(self, event: WorkflowEvent) -> List[str]:
        """
        Process an external event.

        Args:
            event: The event to process

        Returns:
            List of affected workflow instance IDs
        """
        event_type = event.event_type

        # Check if there are any listeners for this event type
        if event_type not in self.event_listeners:
            logger.debug(f"No listeners for event type: {event_type}")
            return []

        affected_instances = []

        # Process the event for each listener
        for instance_id, listener in list(self.event_listeners[event_type].items()):
            # Check if the instance exists
            if instance_id not in self.workflow_instances:
                logger.warning(f"Instance not found: {instance_id}")
                continue

            # Update the instance context with event data
            instance = self.workflow_instances[instance_id]

            # Map event payload to context according to mapping
            for context_key, payload_key in listener.event_mapping.items():
                if payload_key in event.payload:
                    instance.context[context_key] = event.payload[payload_key]

            # Also store the complete event
            instance.context["_last_event"] = event.model_dump()

            # Get the success state
            success_state = listener.success_state
            if success_state:
                # Transition to the success state
                instance.current_step_id = success_state

                # Execute the next step
                await self._execute_workflow_step(instance)

                # Remove the listener
                del self.event_listeners[event_type][instance_id]

                # Cancel any timeout
                timer_key = f"{instance_id}_{event_type}_timeout"
                if timer_key in self.timers and self.timers[timer_key].task:
                    self.timers[timer_key].task.cancel()
                    del self.timers[timer_key]

                affected_instances.append(instance_id)

        logger.info(f"Processed event {event_type}, affected instances: {len(affected_instances)}")

        return affected_instances

    async def advance_workflow(self, instance_id: str, user_input: str = "") -> bool:
        """
        Advance a workflow instance, optionally with user input.

        Args:
            instance_id: Workflow instance ID
            user_input: Optional user input

        Returns:
            True if the workflow was advanced, False otherwise
        """
        # Check if the instance exists
        if instance_id not in self.workflow_instances:
            logger.warning(f"Instance not found: {instance_id}")
            return False

        # Get the instance
        instance = self.workflow_instances[instance_id]

        # Check if the workflow is in a state that can be advanced
        if instance.status not in [WorkflowStatus.RUNNING, WorkflowStatus.WAITING]:
            logger.warning(f"Workflow cannot be advanced: {instance.status}")
            return False

        # Add user input to context if provided
        if user_input:
            instance.context["_user_input"] = user_input

        # Execute the current step
        await self._execute_workflow_step(instance)
        return True

    def get_workflow_instance(self, instance_id: str) -> Optional[WorkflowInstance]:
        """
        Get a workflow instance by ID.

        Args:
            instance_id: Workflow instance ID

        Returns:
            The workflow instance, or None if not found
        """
        return self.workflow_instances.get(instance_id)

    def get_workflow_definition(self, workflow_id: str) -> Optional[WorkflowDefinition]:
        """
        Get a workflow definition by ID.

        Args:
            workflow_id: Workflow definition ID

        Returns:
            The workflow definition, or None if not found
        """
        return self.workflow_definitions.get(workflow_id)

    def get_workflow_status(self, instance_id: str) -> Optional[WorkflowStatus]:
        """
        Get the status of a workflow instance.

        Args:
            instance_id: Workflow instance ID

        Returns:
            The workflow status, or None if not found
        """
        instance = self.get_workflow_instance(instance_id)
        return instance.status if instance else None

    def get_workflow_context(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the context of a workflow instance.

        Args:
            instance_id: Workflow instance ID

        Returns:
            The workflow context, or None if not found
        """
        instance = self.get_workflow_instance(instance_id)
        return instance.context if instance else None

    def get_active_workflows(self) -> List[str]:
        """
        Get a list of active workflow instance IDs.

        Returns:
            List of active workflow instance IDs
        """
        return [
            instance_id for instance_id, instance in self.workflow_instances.items()
            if instance.status in [WorkflowStatus.RUNNING, WorkflowStatus.WAITING]
        ]

    async def cancel_workflow(self, instance_id: str, reason: str = "Cancelled by user") -> bool:
        """
        Cancel a workflow instance.

        Args:
            instance_id: Workflow instance ID
            reason: Reason for cancellation

        Returns:
            True if successful, False otherwise
        """
        # Check if the instance exists
        if instance_id not in self.workflow_instances:
            logger.warning(f"Instance not found: {instance_id}")
            return False

        # Get the instance
        instance = self.workflow_instances[instance_id]

        # Update status
        instance.update_status(WorkflowStatus.CANCELLED)
        instance.context["_cancellation_reason"] = reason

        # Clean up any resources
        # Cancel any timers
        for timer_key in list(self.timers.keys()):
            if timer_key.startswith(f"{instance_id}_"):
                if self.timers[timer_key].task:
                    self.timers[timer_key].task.cancel()
                del self.timers[timer_key]

        # Remove any event listeners
        for event_type in list(self.event_listeners.keys()):
            if instance_id in self.event_listeners[event_type]:
                del self.event_listeners[event_type][instance_id]

        logger.info(f"Workflow instance {instance_id} cancelled: {reason}")
        return True

    def serialize_workflow_definition(self, workflow_id: str) -> Dict[str, Any]:
        """
        Serialize a workflow definition to a dictionary.

        Args:
            workflow_id: Workflow definition ID

        Returns:
            Serialized workflow definition

        Raises:
            ValueError: If the workflow is not found
        """
        workflow = self.get_workflow_definition(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow definition not found: {workflow_id}")

        # Convert to dict, but handle special cases like callables
        workflow_dict = workflow.model_dump(exclude={"steps"})

        # Handle steps separately
        steps_dict = {}
        for step_id, step in workflow.steps.items():
            step_type = step.__class__.__name__
            step_dict = step.model_dump(exclude={"action", "api_function", "condition", "aggregation_function"})
            step_dict["type"] = step_type
            steps_dict[step_id] = step_dict

        workflow_dict["steps"] = steps_dict
        return workflow_dict

    def serialize_workflow_instance(self, instance_id: str) -> Dict[str, Any]:
        """
        Serialize a workflow instance to a dictionary.

        Args:
            instance_id: Workflow instance ID

        Returns:
            Serialized workflow instance

        Raises:
            ValueError: If the instance is not found
        """
        instance = self.get_workflow_instance(instance_id)
        if not instance:
            raise ValueError(f"Workflow instance not found: {instance_id}")

        return instance.model_dump()


# --------------------------------------------------------------
# Helper Functions for DSL-like Usage
# --------------------------------------------------------------

def create_workflow(workflow_id: str, name: str, description: str = "") -> WorkflowDefinition:
    """
    Create a new workflow definition with a fluent API.

    Args:
        workflow_id: Unique identifier for the workflow
        name: Human-readable name
        description: Optional description

    Returns:
        A new workflow definition
    """
    return WorkflowDefinition(
        workflow_id=workflow_id,
        name=name,
        description=description
    )


def auto_step(step_id: str, name: str, next_state: str,
              action: Optional[Callable] = None, description: str = "") -> AutoTransitionStep:
    """
    Create an auto transition step.

    Args:
        step_id: Unique identifier for the step
        name: Human-readable name
        next_state: State to transition to
        action: Optional function to execute before transition
        description: Optional description

    Returns:
        A new auto transition step
    """
    return AutoTransitionStep(
        step_id=step_id,
        name=name,
        next_state=next_state,
        action=action,
        description=description
    )


def api_step(step_id: str, name: str, api_function: Callable,
             success_state: str, failure_state: str,
             input_mapping: Dict[str, str] = None,
             output_mapping: Dict[str, str] = None,
             description: str = "") -> APICallStep:
    """
    Create an API call step.

    Args:
        step_id: Unique identifier for the step
        name: Human-readable name
        api_function: Function to call the API
        success_state: State to transition to on success
        failure_state: State to transition to on failure
        input_mapping: Map context keys to API parameters
        output_mapping: Map API response to context keys
        description: Optional description

    Returns:
        A new API call step
    """
    return APICallStep(
        step_id=step_id,
        name=name,
        api_function=api_function,
        success_state=success_state,
        failure_state=failure_state,
        input_mapping=input_mapping or {},
        output_mapping=output_mapping or {},
        description=description
    )


def condition_step(step_id: str, name: str, condition: Callable,
                   true_state: str, false_state: str,
                   description: str = "") -> ConditionStep:
    """
    Create a condition step.

    Args:
        step_id: Unique identifier for the step
        name: Human-readable name
        condition: Function that returns True or False
        true_state: State to transition to if condition is True
        false_state: State to transition to if condition is False
        description: Optional description

    Returns:
        A new condition step
    """
    return ConditionStep(
        step_id=step_id,
        name=name,
        condition=condition,
        true_state=true_state,
        false_state=false_state,
        description=description
    )


def llm_step(step_id: str, name: str, llm_interface: Any,
             prompt_template: str, context_mapping: Dict[str, str],
             output_mapping: Dict[str, str], next_state: str,
             error_state: Optional[str] = None,
             description: str = "") -> LLMProcessingStep:
    """
    Create an LLM processing step.

    Args:
        step_id: Unique identifier for the step
        name: Human-readable name
        llm_interface: LLM interface to use
        prompt_template: Template string for the prompt
        context_mapping: Map context keys to prompt variables
        output_mapping: Map LLM response to context keys
        next_state: State to transition to on success
        error_state: State to transition to on error
        description: Optional description

    Returns:
        A new LLM processing step
    """
    return LLMProcessingStep(
        step_id=step_id,
        name=name,
        llm_interface=llm_interface,
        prompt_template=prompt_template,
        context_mapping=context_mapping,
        output_mapping=output_mapping,
        next_state=next_state,
        error_state=error_state,
        description=description
    )


def wait_event_step(step_id: str, name: str, event_type: str,
                    success_state: str, timeout_seconds: Optional[int] = None,
                    timeout_state: Optional[str] = None,
                    event_mapping: Dict[str, str] = None,
                    description: str = "") -> WaitForEventStep:
    """
    Create a wait for event step.

    Args:
        step_id: Unique identifier for the step
        name: Human-readable name
        event_type: Type of event to wait for
        success_state: State to transition to when event received
        timeout_seconds: Optional timeout in seconds
        timeout_state: State to transition to on timeout
        event_mapping: Map event payload to context keys
        description: Optional description

    Returns:
        A new wait for event step
    """
    return WaitForEventStep(
        step_id=step_id,
        name=name,
        description=description,
        config=WaitEventConfig(
            event_type=event_type,
            success_state=success_state,
            timeout_seconds=timeout_seconds,
            timeout_state=timeout_state,
            event_mapping=event_mapping or {}
        )
    )


def timer_step(step_id: str, name: str, delay_seconds: int,
               next_state: str, description: str = "") -> TimerStep:
    """
    Create a timer step.

    Args:
        step_id: Unique identifier for the step
        name: Human-readable name
        delay_seconds: Time to wait in seconds
        next_state: State to transition to after the delay
        description: Optional description

    Returns:
        A new timer step
    """
    return TimerStep(
        step_id=step_id,
        name=name,
        delay_seconds=delay_seconds,
        next_state=next_state,
        description=description
    )


def parallel_step(step_id: str, name: str, steps: List[WorkflowStep],
                  next_state: str, error_state: Optional[str] = None,
                  aggregation_function: Optional[Callable] = None,
                  description: str = "") -> ParallelStep:
    """
    Create a parallel step.

    Args:
        step_id: Unique identifier for the step
        name: Human-readable name
        steps: List of steps to execute in parallel
        next_state: State to transition to on success
        error_state: State to transition to on any error
        aggregation_function: Function to aggregate results
        description: Optional description

    Returns:
        A new parallel step
    """
    return ParallelStep(
        step_id=step_id,
        name=name,
        steps=steps,
        next_state=next_state,
        error_state=error_state,
        aggregation_function=aggregation_function,
        description=description
    )

# --------------------------------------------------------------
