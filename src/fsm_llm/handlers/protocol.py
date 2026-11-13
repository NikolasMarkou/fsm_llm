"""Hook points, error modes and the handler protocol.

Package ``fsm_llm.handlers`` (plan d454fd62, step 77: split out of the old module
``fsm_llm/handlers.py``). ``HandlerTiming`` and ``HandlerErrorMode`` are the two closed
vocabularies, ``FSMHandler`` is the structural interface every handler satisfies, and
``ExecutionLambda`` and ``ConditionLambda`` are the callable shapes of a lambda handler.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Any, Protocol


class HandlerTiming(str, Enum):
    """
    Enumeration defining hook points where handlers can be executed during FSM lifecycle.

    These timing points provide comprehensive coverage of the FSM execution flow,
    allowing handlers to intervene at precisely the right moments for their specific needs.
    """

    START_CONVERSATION = "start_conversation"
    PRE_PROCESSING = "pre_processing"
    POST_PROCESSING = "post_processing"
    PRE_TRANSITION = "pre_transition"
    POST_TRANSITION = "post_transition"
    CONTEXT_UPDATE = "context_update"
    END_CONVERSATION = "end_conversation"
    ERROR = "error"


class HandlerErrorMode(str, Enum):
    """What ``HandlerSystem`` does when a handler fails.

    ``CONTINUE`` logs the failure and moves on to the next handler; ``RAISE``
    raises ``HandlerExecutionError`` (a ``critical`` handler raises in both
    modes). A ``str`` enum, so ``"raise"`` and ``HandlerErrorMode.RAISE`` are
    interchangeable wherever ``handler_error_mode`` is accepted.
    """

    CONTINUE = "continue"
    RAISE = "raise"

    @classmethod
    def from_string(cls, value: HandlerErrorMode | str) -> HandlerErrorMode:
        """Return the member for a member or an exact value string.

        Raises ``ValueError`` naming ``handler_error_mode`` and the valid values
        for anything else (another string, ``None``, a number).
        """
        try:
            return cls(value)
        except ValueError:
            valid_values = [member.value for member in cls]
            raise ValueError(
                f"Invalid handler_error_mode: {value!r}. Must be one of {valid_values}"
            ) from None


# Type aliases for better code readability and type safety
ExecutionLambda = Callable[[dict[str, Any]], dict[str, Any]]
"""Type alias for execution lambda functions."""


ConditionLambda = Callable[
    [HandlerTiming, str, str | None, dict[str, Any], set[str] | None], bool
]
"""Type alias for condition evaluation lambda functions."""


class FSMHandler(Protocol):
    """
    Protocol defining the interface for self-determining FSM handlers.

    This protocol establishes the contract that all handlers must implement to participate
    in the FSM execution lifecycle. The key innovation is the ``should_execute`` method,
    which allows each handler to make autonomous decisions about when to run.

    Handlers are synchronous callables (``execute`` returns a dict, never a
    coroutine), ordered by priority for deterministic execution sequences.
    """

    @property
    def priority(self) -> int:
        """
        Get the execution priority of this handler.

        Lower numerical values indicate higher priority and earlier execution.
        Default priority is typically 100, allowing for both higher (< 100)
        and lower (> 100) priority handlers.

        :return: Priority value where lower numbers execute first
        :rtype: int
        """
        ...

    def should_execute(
        self,
        timing: HandlerTiming,
        current_state: str,
        target_state: str | None,
        context: dict[str, Any],
        updated_keys: set[str] | None = None,
    ) -> bool:
        """
        Determine if this handler should execute based on current FSM state and context.

        This method is the core of the self-determining architecture. Each handler
        evaluates the provided parameters to decide whether it should participate
        in the current execution cycle.

        :param timing: The lifecycle hook point being executed
        :type timing: HandlerTiming
        :param current_state: Current state identifier of the FSM
        :type current_state: str
        :param target_state: Target state identifier (None if not transitioning)
        :type target_state: str | None
        :param context: Current context data dictionary
        :type context: dict[str, Any]
        :param updated_keys: Set of context keys being updated (for CONTEXT_UPDATE timing)
        :type updated_keys: set[str] | None
        :return: True if the handler should execute, False otherwise
        :rtype: bool
        """
        ...

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Execute the handler's core logic and return context updates.

        This method performs the actual work of the handler. It receives the current
        context and returns a dictionary of updates to be merged back into the context.

        Handlers return their results directly as a dictionary of context updates.

        :param context: Current context data dictionary
        :type context: dict[str, Any]
        :return: Dictionary containing context updates to apply
        :rtype: dict[str, Any]
        :raises Exception: Any exception that occurs during handler execution
        """
        ...
