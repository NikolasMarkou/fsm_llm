"""The steps that pick the next step: ``AutoStep``, ``ConditionStep`` and ``SwitchStep``.

Part of the package (plan d454fd62 step 92; the old ``workflows/steps.py`` at the same dotted path).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fsm_llm.logging import logger

from ..models import WorkflowStepResult
from .base import WorkflowStep
from .helpers import _call_user_callable, _error_text


class AutoStep(WorkflowStep):
    """A step that automatically transitions to the next state.

    ``action`` may return a dict (merged into the context) or ``None``. When
    the action fails, the step routes to ``error_step_id`` if one is set and
    returns a failure result with no route otherwise (the instance then
    FAILS).
    """

    next_step_id: str
    action: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    error_step_id: str | None = None

    async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
        """Execute the step and transition automatically."""
        try:
            data = {}
            if self.action:
                data = await self._with_timeout(
                    _call_user_callable(self.action, context)
                )
                if data is None:
                    data = {}
                elif not isinstance(data, dict):
                    raise TypeError(
                        f"action must return a dict or None, got {type(data).__name__}"
                    )

            return WorkflowStepResult.success_result(
                data=data,
                next_step_id=self.next_step_id,
                message=f"Auto-transitioned to {self.next_step_id}",
            )
        except Exception as e:
            logger.error(f"Error in auto transition step {self.step_id}: {e!s}")
            return WorkflowStepResult.failure_result(
                error=_error_text(e),
                next_step_id=self.error_step_id,
                message=f"Auto-transition failed: {e!s}",
            )


class ConditionStep(WorkflowStep):
    """A step that evaluates a condition and transitions accordingly.

    When the condition raises, the step routes to ``error_step_id`` if one is
    set and returns a failure result with no route otherwise.
    """

    condition: Callable[[dict[str, Any]], bool]
    true_step_id: str
    false_step_id: str
    error_step_id: str | None = None

    async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
        """Evaluate the condition and determine the next state."""
        try:
            result = await self._with_timeout(
                _call_user_callable(self.condition, context)
            )

            # Determine the next state
            next_step_id = self.true_step_id if result else self.false_step_id

            return WorkflowStepResult.success_result(
                next_step_id=next_step_id,
                message=f"Condition evaluated to {result}, transitioning to {next_step_id}",
                data={"condition_result": result},
            )
        except Exception as e:
            logger.error(f"Error in condition step {self.step_id}: {e!s}")
            return WorkflowStepResult.failure_result(
                error=_error_text(e),
                next_step_id=self.error_step_id,
                message=f"Condition evaluation failed: {e!s}",
            )


class SwitchStep(WorkflowStep):
    """Route to different states based on a context key value.

    Unlike ``ConditionStep`` which only supports binary branching,
    ``SwitchStep`` supports arbitrary n-way routing. The context value is
    compared as ``str(value)`` (so ``True`` matches the case ``"True"``);
    a missing key compares as ``""``.

    Example::

        step = SwitchStep(
            step_id="route",
            name="Route by intent",
            key="user_intent",
            cases={"buy": "checkout", "browse": "catalog", "help": "support"},
            default_step_id="fallback",
        )

    Output keys: ``switch_<step_id>_matched`` (the compared value) and
    ``switch_<step_id>_target`` (the chosen state).
    """

    key: str
    """Context key whose value determines the target state."""
    cases: dict[str, str]
    """Mapping of key values to target state IDs."""
    default_step_id: str | None = None
    """State to transition to when the key value doesn't match any case.
    ``None`` means no default (the instance FAILS). Use ``""`` for terminal."""

    def referenced_step_ids(self) -> set[str]:
        """Every case target and the default step id."""
        return {*self.cases.values(), *super().referenced_step_ids()} - {""}

    async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
        """Evaluate the context key and route to the matching state."""
        value = context.get(self.key)
        value_str = str(value) if value is not None else ""

        target = self.cases.get(value_str, self.default_step_id)
        if target is None:
            return WorkflowStepResult.failure_result(
                error=f"No matching case for {self.key}={value_str!r} and no default_step_id",
                next_step_id="",
                message=f"Switch step '{self.step_id}' has no route for value {value_str!r}",
            )

        return WorkflowStepResult.success_result(
            data={
                f"switch_{self.step_id}_matched": value_str,
                f"switch_{self.step_id}_target": target,
            },
            next_step_id=target,
            message=f"Routed to '{target}' (key={self.key}, value={value_str!r})",
        )
