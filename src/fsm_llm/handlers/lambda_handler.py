"""``LambdaHandler``: the concrete handler a ``HandlerBuilder`` builds.

Package ``fsm_llm.handlers`` (plan d454fd62, step 77: split out of the old module
``fsm_llm/handlers.py``).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .base import BaseHandler
from .errors import HandlerExecutionError
from .protocol import ConditionLambda, ExecutionLambda, HandlerTiming


def _as_set(names: Iterable[str] | str | None) -> set[str]:
    """``names`` as a set: a bare ``str`` is one name (not its characters),
    ``None`` is empty."""
    return {names} if isinstance(names, str) else set(names or ())


class LambdaHandler(BaseHandler):
    """
    Internal implementation of a handler using lambda functions.

    This class is the concrete implementation created by the HandlerBuilder.
    It provides all the conditional logic configured through the builder pattern.
    """

    def __init__(
        self,
        name: str,
        condition_lambdas: list[ConditionLambda],
        execution_lambda: ExecutionLambda,
        timings: set[HandlerTiming] | None,
        states: set[str],
        target_states: set[str],
        required_keys: set[str],
        updated_keys: set[str],
        priority: int = 100,
        not_states: set[str] | None = None,
        not_target_states: set[str] | None = None,
        critical: bool = False,
    ):
        """
        Initialize the lambda handler with all configuration from the builder.

        :param name: Name for the handler (used in logs and debugging)
        :type name: str
        :param condition_lambdas: List of condition functions that must all return True
        :type condition_lambdas: list[ConditionLambda]
        :param execution_lambda: The function to execute when conditions are met
        :type execution_lambda: ExecutionLambda
        :param timings: Set of timing points when this handler can execute
        :type timings: set[HandlerTiming]
        :param states: Set of current states that allow execution
        :type states: set[str]
        :param target_states: Set of target states that allow execution
        :type target_states: set[str]
        :param required_keys: Set of context keys that must be present
        :type required_keys: set[str]
        :param updated_keys: Set of context keys to watch for updates
        :type updated_keys: set[str]
        :param priority: Execution priority (lower numbers execute first)
        :type priority: int
        :param not_states: Set of current states that prevent execution
        :type not_states: set[str]
        :param not_target_states: Set of target states that prevent execution
        :type not_target_states: set[str]
        :param critical: If True, handler errors are raised even in "continue" error mode
        :type critical: bool
        """
        super().__init__(name=name, priority=priority, critical=critical)
        self.condition_lambdas = condition_lambdas
        self.execution_lambda = execution_lambda
        self.timings = timings
        self.states = _as_set(states)
        self.target_states = _as_set(target_states)
        self.required_keys = _as_set(required_keys)
        self.updated_keys = _as_set(updated_keys)
        self.not_states = _as_set(not_states)
        self.not_target_states = _as_set(not_target_states)

    def should_execute(
        self,
        timing: HandlerTiming,
        current_state: str,
        target_state: str | None,
        context: dict[str, Any],
        updated_keys: set[str] | None = None,
    ) -> bool:
        """
        Determine if this handler should execute based on builder configuration.

        :param timing: The lifecycle hook point being executed
        :type timing: HandlerTiming
        :param current_state: Current FSM state identifier
        :type current_state: str
        :param target_state: Target state identifier (None if not transitioning)
        :type target_state: str | None
        :param context: Current context data dictionary
        :type context: dict[str, Any]
        :param updated_keys: Set of context keys being updated
        :type updated_keys: set[str] | None
        :return: True if all conditions are met and handler should execute
        :rtype: bool
        """
        # Quick rejection tests first for optimal performance

        # Check timing constraints - None means "all timings", empty set means "no timings"
        if self.timings is not None and timing not in self.timings:
            return False

        # Check current state inclusion constraints
        if self.states and current_state not in self.states:
            return False

        # Check current state exclusion constraints
        if self.not_states and current_state in self.not_states:
            return False

        # Check target state inclusion constraints
        if self.target_states and (
            not target_state or target_state not in self.target_states
        ):
            return False

        # Check target state exclusion constraints
        if (
            self.not_target_states
            and target_state
            and target_state in self.not_target_states
        ):
            return False

        # Check required context keys constraints
        if self.required_keys and not all(key in context for key in self.required_keys):
            return False

        # Check updated keys constraints (for CONTEXT_UPDATE timing)
        if self.updated_keys and (
            not updated_keys
            or not any(key in updated_keys for key in self.updated_keys)
        ):
            return False

        # Evaluate custom condition lambdas - all must return True
        for condition in self.condition_lambdas:
            try:
                if not condition(
                    timing, current_state, target_state, context, updated_keys
                ):
                    return False
            except Exception as e:
                raise HandlerExecutionError(
                    self.name,
                    RuntimeError(
                        f"Condition lambda raised exception "
                        f"(timing={timing}, state={current_state}): {e!s}"
                    ),
                ) from e

        # All conditions passed - handler should execute
        return True

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Execute the handler's lambda function.

        :param context: Current context data dictionary
        :type context: dict[str, Any]
        Failures propagate UNWRAPPED and UNLOGGED.
        :class:`HandlerSystem.execute_handlers` is the single site that wraps a
        handler failure in :class:`HandlerExecutionError` and logs it with a
        traceback, for every handler type.

        :return: Dictionary containing context updates
        :rtype: dict[str, Any]
        :raises TypeError: If the lambda returns a value that is neither ``None``
            nor a ``dict``
        """
        # DECISION plan-2026-07-18T162030-a02151fe/D-013 [STALE]
        # Do NOT reintroduce a try/except HandlerExecutionError wrapper here, and
        # do NOT move the TypeError below back inside a try. This method used to
        # do both, and the result was a message wrapped 2x (raising lambda) or 3x
        # (non-dict return, because the raise was caught by its own except),
        # burying the real cause. execute_handlers is the ONE wrapping site --
        # which is what plain BaseHandler subclasses have always relied on. The
        # duplicate logger.error that lived here was removed for the same reason:
        # execute_handlers already logs this failure with a full traceback.
        result = self.execution_lambda(context)

        # Ensure we always return a dict
        if result is None:
            return {}
        if isinstance(result, dict):
            return result

        raise TypeError(f"Handler returned non-dict result: {type(result).__name__}")

    def __str__(self) -> str:
        """
        Return string representation for debugging and logging purposes.

        :return: String representation of the handler
        :rtype: str
        """
        return f"{self.name} (Lambda Handler)"
