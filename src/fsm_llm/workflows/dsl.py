"""
DSL helper functions for creating workflows with a fluent API.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# --------------------------------------------------------------
# local imports
# --------------------------------------------------------------
from .definitions import WorkflowDefinition
from .models import WaitEventConfig
from .steps import (
    AgentStep,
    APICallStep,
    AutoTransitionStep,
    ConditionStep,
    ConversationStep,
    LLMProcessingStep,
    ParallelStep,
    RetryStep,
    SwitchStep,
    TimerStep,
    WaitForEventStep,
    WorkflowStep,
)

# --------------------------------------------------------------


def create_workflow(
    workflow_id: str, name: str, description: str = ""
) -> WorkflowDefinition:
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
        workflow_id=workflow_id, name=name, description=description
    )


# --------------------------------------------------------------


def auto_step(
    step_id: str,
    name: str,
    next_state: str,
    action: Callable | None = None,
    description: str = "",
    error_state: str | None = None,
    timeout: float | None = None,
) -> AutoTransitionStep:
    """
    Create an auto transition step.

    Args:
        step_id: Unique identifier for the step
        name: Human-readable name
        next_state: State to transition to
        action: Optional function (sync or async) to execute before transition;
            returns a dict merged into the context, or None
        description: Optional description
        error_state: State to transition to if the action fails (otherwise the
            instance FAILS)
        timeout: Optional per-step time limit in seconds

    Returns:
        A new auto transition step
    """
    return AutoTransitionStep(
        step_id=step_id,
        name=name,
        next_state=next_state,
        action=action,
        description=description,
        error_state=error_state,
        timeout=timeout,
    )


# --------------------------------------------------------------


def api_step(
    step_id: str,
    name: str,
    api_function: Callable,
    success_state: str,
    failure_state: str,
    input_mapping: dict[str, str] | None = None,
    output_mapping: dict[str, str] | None = None,
    description: str = "",
    timeout: float | None = None,
) -> APICallStep:
    """
    Create an API call step.

    Args:
        step_id: Unique identifier for the step
        name: Human-readable name
        api_function: Function to call the API
        success_state: State to transition to on success
        failure_state: State to transition to on failure
        input_mapping: ``{api_param: context_key}``
        output_mapping: ``{context_key: result_key}``; ``result_key`` may be a
            dotted path, or ``""`` for the whole result
        description: Optional description
        timeout: Optional per-step time limit in seconds

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
        description=description,
        timeout=timeout,
    )


# --------------------------------------------------------------


def condition_step(
    step_id: str,
    name: str,
    condition: Callable,
    true_state: str,
    false_state: str,
    description: str = "",
    error_state: str | None = None,
    timeout: float | None = None,
) -> ConditionStep:
    """
    Create a condition step.

    Args:
        step_id: Unique identifier for the step
        name: Human-readable name
        condition: Function that returns True or False
        true_state: State to transition to if condition is True
        false_state: State to transition to if condition is False
        description: Optional description
        error_state: State to transition to if the condition raises
            (otherwise the instance FAILS)
        timeout: Optional per-step time limit in seconds

    Returns:
        A new condition step
    """
    return ConditionStep(
        step_id=step_id,
        name=name,
        condition=condition,
        true_state=true_state,
        false_state=false_state,
        description=description,
        error_state=error_state,
        timeout=timeout,
    )


# --------------------------------------------------------------


def llm_step(
    step_id: str,
    name: str,
    llm_interface: Any,
    prompt_template: str,
    context_mapping: dict[str, str] | None = None,
    output_mapping: dict[str, str] | None = None,
    next_state: str = "",
    error_state: str | None = None,
    description: str = "",
    timeout: float | None = None,
    system_prompt: str | None = None,
) -> LLMProcessingStep:
    """
    Create an LLM processing step.

    Args:
        step_id: Unique identifier for the step
        name: Human-readable name
        llm_interface: A core ``LLMInterface`` or any object with
            ``generate(prompt)`` (sync or async)
        prompt_template: ``str.format`` template for the prompt
        context_mapping: INPUT mapping ``{prompt_var: context_key}``
        output_mapping: ``{context_key: regex}`` applied to the response
            (``""`` stores the whole response)
        next_state: State to transition to on success (``""`` = terminal)
        error_state: State to transition to on error (otherwise the
            instance FAILS)
        description: Optional description
        timeout: Optional per-step time limit in seconds
        system_prompt: System prompt used with a core ``LLMInterface``

    Returns:
        A new LLM processing step
    """
    extra: dict[str, Any] = {}
    if system_prompt is not None:
        extra["system_prompt"] = system_prompt
    return LLMProcessingStep(
        step_id=step_id,
        name=name,
        llm_interface=llm_interface,
        prompt_template=prompt_template,
        context_mapping=context_mapping or {},
        output_mapping=output_mapping or {},
        next_state=next_state,
        error_state=error_state,
        description=description,
        timeout=timeout,
        **extra,
    )


# --------------------------------------------------------------


def wait_event_step(
    step_id: str,
    name: str,
    event_type: str,
    success_state: str,
    timeout_seconds: float | None = None,
    timeout_state: str | None = None,
    event_mapping: dict[str, str] | None = None,
    description: str = "",
    correlation_key: str | None = None,
) -> WaitForEventStep:
    """
    Create a wait for event step.

    Args:
        step_id: Unique identifier for the step
        name: Human-readable name
        event_type: Type of event to wait for
        success_state: State to transition to when event received
            (``""`` completes the workflow)
        timeout_seconds: Optional timeout in seconds
        timeout_state: State to transition to on timeout (requires
            ``timeout_seconds``; without it a timeout FAILS the instance)
        event_mapping: ``{context_key: payload_key}``
        description: Optional description
        correlation_key: Only accept events whose ``payload[key]`` equals
            ``context[key]``

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
            event_mapping=event_mapping or {},
            correlation_key=correlation_key,
        ),
    )


# --------------------------------------------------------------


def timer_step(
    step_id: str,
    name: str,
    delay_seconds: float,
    next_state: str,
    description: str = "",
) -> TimerStep:
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
        description=description,
    )


# --------------------------------------------------------------


def parallel_step(
    step_id: str,
    name: str,
    steps: list[WorkflowStep],
    next_state: str,
    error_state: str | None = None,
    aggregation_function: Callable | None = None,
    description: str = "",
    timeout: float | None = None,
) -> ParallelStep:
    """
    Create a parallel step.

    Args:
        step_id: Unique identifier for the step
        name: Human-readable name
        steps: List of steps to execute in parallel
        next_state: State to transition to on success
        error_state: State to transition to on any error (otherwise the
            instance FAILS)
        aggregation_function: Function to aggregate results
        description: Optional description
        timeout: Optional time limit in seconds for all children together

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
        description=description,
        timeout=timeout,
    )


def conversation_step(
    step_id: str,
    name: str,
    success_state: str = "",
    fsm_file: str | None = None,
    fsm_definition: Any | None = None,
    model: str | None = None,
    initial_context: dict[str, str] | None = None,
    context_mapping: dict[str, str] | None = None,
    auto_messages: list[str] | None = None,
    max_turns: int = 20,
    error_state: str | None = None,
    description: str = "",
    conversation_timeout: float | None = None,
    timeout: float | None = None,
    require_completion: bool = False,
    use_user_input: bool = False,
) -> ConversationStep:
    """
    Create a conversation step that runs an FSM conversation within a workflow.

    This bridges workflows with FSM-LLM core and reasoning: a workflow step
    can invoke a full FSM conversation (including push/pop stacking for reasoning).

    Args:
        step_id: Unique identifier for the step
        name: Human-readable name
        success_state: State to transition to on success
        fsm_file: Path to FSM definition JSON file
        fsm_definition: FSM definition as a dict or ``FSMDefinition``
            (exactly one of ``fsm_file`` / ``fsm_definition``)
        model: LLM model to use
        initial_context: INPUT map ``{conversation_key: workflow_key}``
        context_mapping: OUTPUT map ``{workflow_key: conversation_key}``
        auto_messages: Messages to send to drive the conversation
        max_turns: Maximum conversation turns
        error_state: State to transition to on error (otherwise the
            instance FAILS)
        description: Optional description
        conversation_timeout: Time limit for the conversation in seconds
        timeout: Per-step time limit (the smaller limit wins)
        require_completion: Fail when the conversation has not ended
        use_user_input: Also send ``advance_workflow``'s user input

    Returns:
        A new conversation step
    """
    return ConversationStep(
        step_id=step_id,
        name=name,
        success_state=success_state,
        fsm_file=fsm_file,
        fsm_definition=fsm_definition,
        model=model,
        initial_context=initial_context or {},
        context_mapping=context_mapping or {},
        auto_messages=auto_messages or [],
        max_turns=max_turns,
        error_state=error_state,
        description=description,
        conversation_timeout=conversation_timeout,
        timeout=timeout,
        require_completion=require_completion,
        use_user_input=use_user_input,
    )


# --------------------------------------------------------------


# Workflow builder class for even more fluent API
class WorkflowBuilder:
    """Builder class for creating workflows with a fluent API."""

    def __init__(self, workflow_id: str, name: str, description: str = ""):
        """Initialize the workflow builder."""
        self.workflow = WorkflowDefinition(
            workflow_id=workflow_id, name=name, description=description
        )

    def add_step(self, step: WorkflowStep) -> WorkflowBuilder:
        """Add a step to the workflow."""
        self.workflow.with_step(step)
        return self

    def set_initial_step(self, step: WorkflowStep) -> WorkflowBuilder:
        """Set the initial step of the workflow."""
        self.workflow.with_initial_step(step)
        return self

    def add_metadata(self, key: str, value: Any) -> WorkflowBuilder:
        """Add metadata to the workflow."""
        self.workflow.metadata[key] = value
        return self

    def build(self, validate: bool = False) -> WorkflowDefinition:
        """Return the workflow definition (the builder's own object).

        Args:
            validate: Run ``WorkflowDefinition.validate()`` first (raises
                ``WorkflowValidationError``).
        """
        if validate:
            self.workflow.validate()
        return self.workflow


# --------------------------------------------------------------


def workflow_builder(
    workflow_id: str, name: str, description: str = ""
) -> WorkflowBuilder:
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
def linear_workflow(
    workflow_id: str, name: str, steps: list[WorkflowStep], description: str = ""
) -> WorkflowDefinition:
    """
    Register ``steps`` in a workflow and make the first one initial.

    This does NOT wire the steps together: each step must already name its
    successor (for example ``auto_step(..., next_state="<next id>")``), and
    the last one should end the workflow (``next_state=""``).

    Args:
        workflow_id: Unique identifier for the workflow
        name: Human-readable name
        steps: List of steps in execution order
        description: Optional description

    Returns:
        A workflow definition (not yet validated)
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


def conditional_workflow(
    workflow_id: str,
    name: str,
    initial_step: WorkflowStep,
    condition_step: ConditionStep,
    true_branch: list[WorkflowStep],
    false_branch: list[WorkflowStep],
    description: str = "",
) -> WorkflowDefinition:
    """
    Register an initial step, a condition step and two branches.

    This does NOT wire the steps together: ``initial_step`` must route to
    ``condition_step``, whose ``true_state``/``false_state`` must name the
    first step of each branch.

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


def event_driven_workflow(
    workflow_id: str,
    name: str,
    setup_steps: list[WorkflowStep],
    event_step: WaitForEventStep,
    processing_steps: list[WorkflowStep],
    description: str = "",
) -> WorkflowDefinition:
    """
    Register setup steps, an event wait and processing steps.

    This does NOT wire the steps together: the last setup step must route
    to ``event_step``, whose ``success_state`` must name the first
    processing step.

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
    all_steps = [*setup_steps, event_step, *processing_steps]
    for step in all_steps:
        workflow.with_step(step)

    # Set the first setup step as initial (or event step if no setup)
    if setup_steps:
        workflow.initial_step_id = setup_steps[0].step_id
    else:
        workflow.initial_step_id = event_step.step_id

    return workflow


# ------------------------------------------------------------------
# New step type factories
# ------------------------------------------------------------------


def agent_step(
    step_id: str,
    name: str,
    agent: Any,
    task_template: str = "{task}",
    success_state: str = "",
    context_mapping: dict[str, str] | None = None,
    error_state: str | None = None,
    description: str = "",
    input_mapping: dict[str, str] | None = None,
    timeout: float | None = None,
) -> AgentStep:
    """Create an ``AgentStep`` that runs an FSM-LLM agent.

    Args:
        step_id: Unique step identifier.
        name: Human-readable step name.
        agent: A ``BaseAgent`` instance (or anything with ``run(task)``).
        task_template: Format string for agent task (filled from context).
        success_state: State to transition to on success.
        context_mapping: OUTPUT ``{workflow_key: agent_result_key}`` mapping.
        error_state: State to transition to on failure (otherwise the
            instance FAILS).
        description: Step description.
        input_mapping: INPUT ``{agent_context_key: workflow_key}`` passed as
            ``agent.run(task, initial_context=...)``.
        timeout: Optional per-step time limit in seconds.

    Returns:
        Configured AgentStep instance.
    """
    return AgentStep(
        step_id=step_id,
        name=name,
        agent=agent,
        task_template=task_template,
        success_state=success_state,
        context_mapping=context_mapping or {},
        error_state=error_state,
        description=description,
        input_mapping=input_mapping or {},
        timeout=timeout,
    )


def retry_step(
    step_id: str,
    name: str,
    step: WorkflowStep,
    max_retries: int = 3,
    backoff_factor: float = 1.0,
    description: str = "",
    timeout: float | None = None,
) -> RetryStep:
    """Create a ``RetryStep`` that wraps another step with retry logic.

    Args:
        step_id: Unique step identifier.
        name: Human-readable step name.
        step: The inner step to retry on failure.
        max_retries: Maximum number of retry attempts.
        backoff_factor: Linear delay multiplier: retry ``n`` waits
            ``backoff_factor * n`` seconds.
        description: Step description.
        timeout: Optional time limit per attempt in seconds.

    Returns:
        Configured RetryStep instance.
    """
    return RetryStep(
        step_id=step_id,
        name=name,
        step=step,
        max_retries=max_retries,
        backoff_factor=backoff_factor,
        description=description,
        timeout=timeout,
    )


def switch_step(
    step_id: str,
    name: str,
    key: str,
    cases: dict[str, str],
    default_state: str | None = None,
    description: str = "",
) -> SwitchStep:
    """Create a ``SwitchStep`` for n-way branching on a context key.

    Args:
        step_id: Unique step identifier.
        name: Human-readable step name.
        key: Context key whose value determines routing.
        cases: ``{value: target_state}`` mapping.
        default_state: State when value matches no case. ``None`` (the
            default, same as ``SwitchStep``) FAILS the instance on an
            unmatched value; ``""`` completes it.
        description: Step description.

    Returns:
        Configured SwitchStep instance.
    """
    return SwitchStep(
        step_id=step_id,
        name=name,
        key=key,
        cases=cases,
        default_state=default_state,
        description=description,
    )


# --------------------------------------------------------------
