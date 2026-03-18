"""
DSL helper functions for creating workflows with a fluent API.
"""

from typing import Dict, List, Any, Optional, Callable

# --------------------------------------------------------------
# local imports
# --------------------------------------------------------------

from .definitions import WorkflowDefinition
from .steps import (
    WorkflowStep, AutoTransitionStep, APICallStep, ConditionStep,
    LLMProcessingStep, WaitForEventStep, TimerStep, ParallelStep
)
from .models import WaitEventConfig

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

# --------------------------------------------------------------


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

# --------------------------------------------------------------


def api_step(step_id: str, name: str, api_function: Callable,
             success_state: str, failure_state: str,
             input_mapping: Optional[Dict[str, str]] = None,
             output_mapping: Optional[Dict[str, str]] = None,
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

# --------------------------------------------------------------


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

# --------------------------------------------------------------


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

# --------------------------------------------------------------


def wait_event_step(step_id: str, name: str, event_type: str,
                    success_state: str, timeout_seconds: Optional[int] = None,
                    timeout_state: Optional[str] = None,
                    event_mapping: Optional[Dict[str, str]] = None,
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

# --------------------------------------------------------------


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

# --------------------------------------------------------------

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


# Workflow builder class for even more fluent API
class WorkflowBuilder:
    """Builder class for creating workflows with a fluent API."""

    def __init__(self, workflow_id: str, name: str, description: str = ""):
        """Initialize the workflow builder."""
        self.workflow = WorkflowDefinition(
            workflow_id=workflow_id,
            name=name,
            description=description
        )

    def add_step(self, step: WorkflowStep) -> 'WorkflowBuilder':
        """Add a step to the workflow."""
        self.workflow.with_step(step)
        return self

    def set_initial_step(self, step: WorkflowStep) -> 'WorkflowBuilder':
        """Set the initial step of the workflow."""
        self.workflow.with_initial_step(step)
        return self

    def add_metadata(self, key: str, value: Any) -> 'WorkflowBuilder':
        """Add metadata to the workflow."""
        self.workflow.metadata[key] = value
        return self

    def build(self) -> WorkflowDefinition:
        """Build and return the workflow definition."""
        return self.workflow

# --------------------------------------------------------------


def workflow_builder(workflow_id: str, name: str, description: str = "") -> WorkflowBuilder:
    """
    Create a new workflow builder.

    Args:
        workflow_id: Unique identifier for the workflow
        name: Human-readable name
        description: Optional description

    Returns:
        A new workflow builder
    """
    return WorkflowBuilder(workflow_id, name, description)


# --------------------------------------------------------------

# Convenience functions for common workflow patterns
def linear_workflow(workflow_id: str, name: str, steps: List[WorkflowStep],
                    description: str = "") -> WorkflowDefinition:
    """
    Create a linear workflow where steps execute in sequence.

    Args:
        workflow_id: Unique identifier for the workflow
        name: Human-readable name
        steps: List of steps in execution order
        description: Optional description

    Returns:
        A workflow definition with linear step execution
    """
    if not steps:
        raise ValueError("Linear workflow must have at least one step")

    workflow = create_workflow(workflow_id, name, description)

    # Add all steps
    for step in steps:
        workflow.with_step(step)

    # Set the first step as initial
    workflow.initial_step_id = steps[0].step_id

    return workflow

# --------------------------------------------------------------


def conditional_workflow(workflow_id: str, name: str,
                         initial_step: WorkflowStep,
                         condition_step: ConditionStep,
                         true_branch: List[WorkflowStep],
                         false_branch: List[WorkflowStep],
                         description: str = "") -> WorkflowDefinition:
    """
    Create a conditional workflow with two branches.

    Args:
        workflow_id: Unique identifier for the workflow
        name: Human-readable name
        initial_step: The initial step
        condition_step: The condition step that determines branching
        true_branch: Steps to execute if condition is true
        false_branch: Steps to execute if condition is false
        description: Optional description

    Returns:
        A workflow definition with conditional branching
    """
    workflow = create_workflow(workflow_id, name, description)

    # Add initial step
    workflow.with_initial_step(initial_step)

    # Add condition step
    workflow.with_step(condition_step)

    # Add branch steps
    for step in true_branch + false_branch:
        workflow.with_step(step)

    return workflow

# --------------------------------------------------------------


def event_driven_workflow(workflow_id: str, name: str,
                          setup_steps: List[WorkflowStep],
                          event_step: WaitForEventStep,
                          processing_steps: List[WorkflowStep],
                          description: str = "") -> WorkflowDefinition:
    """
    Create an event-driven workflow that waits for external events.

    Args:
        workflow_id: Unique identifier for the workflow
        name: Human-readable name
        setup_steps: Steps to execute before waiting for events
        event_step: The step that waits for an event
        processing_steps: Steps to execute after receiving the event
        description: Optional description

    Returns:
        A workflow definition that waits for external events
    """
    workflow = create_workflow(workflow_id, name, description)

    # Add all steps
    all_steps = setup_steps + [event_step] + processing_steps
    for step in all_steps:
        workflow.with_step(step)

    # Set the first setup step as initial (or event step if no setup)
    if setup_steps:
        workflow.initial_step_id = setup_steps[0].step_id
    else:
        workflow.initial_step_id = event_step.step_id

    return workflow

# --------------------------------------------------------------