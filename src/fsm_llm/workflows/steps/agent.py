"""The step that runs an agent: ``AgentStep``.

Part of the package (plan d454fd62 step 92; the old ``workflows/steps.py`` at the same dotted path).
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, field_validator

from fsm_llm.logging import logger

from ..exceptions import WorkflowStepError
from ..models import WorkflowStepResult
from .base import WorkflowStep
from .helpers import _call_user_callable, _error_text, _is_async_callable


class AgentStep(WorkflowStep):
    """Execute an FSM-LLM agent as a workflow step.

    Wraps any ``BaseAgent.run()`` call so agents can participate in
    workflows alongside other step types. ``agent.run`` must return an
    ``AgentResult``-like object (``answer``, ``success``, optional
    ``final_context`` / ``structured_output``) or a plain string answer.

    An agent whose class has its own ``async def arun`` is awaited on the loop (a real cancel at
    the time limit, no executor thread, plan ace7a8c8 X5.1); any other agent runs its ``run``
    through ``_call_user_callable`` (the engine ``executor=``), as before.

    Example::

        from fsm_llm.agents import ReactAgent
        agent = ReactAgent(tools=registry)
        step = AgentStep(
            step_id="research",
            name="Research Step",
            agent=agent,
            task_template="Research {topic}",
            success_step_id="analyze",
            context_mapping={"findings": "answer"},
        )

    Output keys: ``agent_answer``/``agent_success`` (latest agent step) and
    ``agent_<step_id>_answer``/``agent_<step_id>_success`` (per step). An
    agent that reports ``success=False`` fails the step.
    """

    agent: Any = Field(exclude=True)
    """A ``BaseAgent`` instance (or any object with a ``run(task)`` method)."""
    task_template: str = "{task}"
    """Format string for the agent task.  Placeholders are filled from context."""
    success_step_id: str = ""
    context_mapping: dict[str, str] = Field(default_factory=dict)
    """OUTPUT mapping ``{workflow_key: agent_result_key}``. ``agent_result_key``
    is looked up in ``final_context``; ``"answer"``, ``"success"`` and
    ``"structured_output"`` also map the matching result attribute."""
    input_mapping: dict[str, str] = Field(default_factory=dict)
    """INPUT mapping ``{agent_context_key: workflow_key}`` passed to
    ``agent.run(task, initial_context=...)`` (only when non-empty)."""
    error_step_id: str | None = None

    @field_validator("agent")
    @classmethod
    def _validate_agent(cls, v: Any) -> Any:
        if not callable(getattr(v, "run", None)):
            raise ValueError("agent must provide a run(task) method")
        return v

    async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
        """Run the agent and map its result back into workflow context."""
        try:
            # Format the task from context
            try:
                task = self.task_template.format(**context)
            except KeyError as e:
                raise WorkflowStepError(
                    f"Missing context key for task template: {e}", step_id=self.step_id
                ) from e

            kwargs: dict[str, Any] = {}
            if self.input_mapping:
                kwargs["initial_context"] = {
                    agent_key: context[wf_key]
                    for agent_key, wf_key in self.input_mapping.items()
                    if wf_key in context
                }

            arun: Any = getattr(self.agent, "arun", None)
            if _is_async_callable(arun):
                run = arun(task, **kwargs)
            else:
                run = _call_user_callable(self.agent.run, task, **kwargs)
            result = await self._with_timeout(run)

            answer: Any
            success: bool
            final_context: Any
            structured: Any
            if isinstance(result, str):
                answer, success, final_context, structured = result, True, {}, None
            else:
                answer = getattr(result, "answer", None)
                success = bool(getattr(result, "success", False))
                final_context = getattr(result, "final_context", None) or {}
                structured = getattr(result, "structured_output", None)

            attrs = {
                "answer": answer,
                "success": success,
                "structured_output": structured,
            }
            data: dict[str, Any] = {
                "agent_answer": answer,
                "agent_success": success,
                f"agent_{self.step_id}_answer": answer,
                f"agent_{self.step_id}_success": success,
            }
            for wf_key, agent_key in self.context_mapping.items():
                if isinstance(final_context, dict) and agent_key in final_context:
                    data[wf_key] = final_context[agent_key]
                elif agent_key in attrs:
                    data[wf_key] = attrs[agent_key]

            if not success:
                return WorkflowStepResult(
                    success=False,
                    data=data,
                    error=f"Agent reported failure: {str(answer or '')[:200]}",
                    next_step_id=self.error_step_id,
                    message="Agent step failed: agent reported success=False",
                )

            return WorkflowStepResult.success_result(
                data=data,
                next_step_id=self.success_step_id,
                message=f"Agent completed: {str(answer or '')[:100]}",
            )

        except Exception as e:
            logger.error(f"Agent step '{self.step_id}' failed: {e!s}")
            return WorkflowStepResult.failure_result(
                error=_error_text(e),
                next_step_id=self.error_step_id,
                message=f"Agent step failed: {e!s}",
            )
