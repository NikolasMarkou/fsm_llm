"""
Factories of the step types that wrap an agent, another step, or a key.

``agent_step``, ``retry_step`` and ``switch_step`` build ``AgentStep``,
``RetryStep`` and ``SwitchStep``. They sit apart from ``dsl.py`` so that module
keeps headroom under the module-size cap (roadmap 0.18); the package
``fsm_llm.workflows`` re-exports them.
"""

from __future__ import annotations

from typing import Any

from .steps import AgentStep, RetryStep, SwitchStep, WorkflowStep


def agent_step(
    step_id: str,
    name: str,
    agent: Any,
    task_template: str = "{task}",
    success_step_id: str = "",
    context_mapping: dict[str, str] | None = None,
    error_step_id: str | None = None,
    description: str = "",
    input_mapping: dict[str, str] | None = None,
    timeout_seconds: float | None = None,
) -> AgentStep:
    """Create an ``AgentStep`` that runs an FSM-LLM agent.

    Args:
        step_id: Unique step identifier.
        name: Human-readable step name.
        agent: A ``BaseAgent`` instance (or anything with ``run(task)``).
        task_template: Format string for agent task (filled from context).
        success_step_id: State to transition to on success.
        context_mapping: OUTPUT ``{workflow_key: agent_result_key}`` mapping.
        error_step_id: State to transition to on failure (otherwise the
            instance FAILS).
        description: Step description.
        input_mapping: INPUT ``{agent_context_key: workflow_key}`` passed as
            ``agent.run(task, initial_context=...)``.
        timeout_seconds: Optional per-step time limit in seconds.

    Returns:
        Configured AgentStep instance.
    """
    return AgentStep(
        step_id=step_id,
        name=name,
        agent=agent,
        task_template=task_template,
        success_step_id=success_step_id,
        context_mapping=context_mapping or {},
        error_step_id=error_step_id,
        description=description,
        input_mapping=input_mapping or {},
        timeout_seconds=timeout_seconds,
    )


def retry_step(
    step_id: str,
    name: str,
    step: WorkflowStep,
    max_retries: int = 3,
    backoff_factor: float = 1.0,
    description: str = "",
    timeout_seconds: float | None = None,
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
        timeout_seconds: Optional time limit per attempt in seconds.

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
        timeout_seconds=timeout_seconds,
    )


def switch_step(
    step_id: str,
    name: str,
    key: str,
    cases: dict[str, str],
    default_step_id: str | None = None,
    description: str = "",
) -> SwitchStep:
    """Create a ``SwitchStep`` for n-way branching on a context key.

    Args:
        step_id: Unique step identifier.
        name: Human-readable step name.
        key: Context key whose value determines routing.
        cases: ``{value: target_state}`` mapping.
        default_step_id: State when value matches no case. ``None`` (the
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
        default_step_id=default_step_id,
        description=description,
    )
