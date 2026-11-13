"""``BaseHandler``: the base class for hand-written handlers.

Package ``fsm_llm.handlers`` (plan d454fd62, step 77: split out of the old module
``fsm_llm/handlers.py``).
"""

from __future__ import annotations

from typing import Any

from .protocol import HandlerTiming


class BaseHandler:
    """
    Base class for implementing FSM handlers with self-contained execution conditions.

    This class provides a foundation for creating custom handlers by implementing
    the FSMHandler protocol. It includes common functionality like name management
    and priority handling, while leaving the core logic (should_execute and execute)
    for subclasses to implement.

    Subclasses must override:
    - ``should_execute()``: Define when the handler should run
    - ``execute()``: Implement the handler's core functionality
    """

    def __init__(
        self, name: str | None = None, priority: int = 100, critical: bool = False
    ):
        """
        Initialize the base handler with name and priority.

        :param name: Optional name for the handler (defaults to class name if None)
        :type name: str | None
        :param priority: Execution priority where lower values indicate higher priority
        :type priority: int
        :param critical: If True, handler errors are raised even in "continue" error mode
        :type critical: bool
        """
        self.name = name or self.__class__.__name__
        self._priority = priority
        self.critical = critical

    @property
    def priority(self) -> int:
        """
        Get the handler's execution priority.

        :return: Priority value where lower numbers execute first
        :rtype: int
        """
        return self._priority

    def should_execute(
        self,
        timing: HandlerTiming,
        current_state: str,
        target_state: str | None,
        context: dict[str, Any],
        updated_keys: set[str] | None = None,
    ) -> bool:
        """
        Determine if this handler should execute based on current conditions.

        Default implementation always returns False. Subclasses must override this
        method to implement their specific execution conditions.

        :param timing: The lifecycle hook point being executed
        :type timing: HandlerTiming
        :param current_state: Current FSM state identifier
        :type current_state: str
        :param target_state: Target state identifier (None if not transitioning)
        :type target_state: str | None
        :param context: Current context data dictionary
        :type context: dict[str, Any]
        :param updated_keys: Set of context keys being updated (for CONTEXT_UPDATE timing)
        :type updated_keys: set[str] | None
        :return: Always False in base implementation
        :rtype: bool
        """
        return False

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Execute the handler's core logic and return context updates.

        Default implementation does nothing and returns an empty dictionary.
        Subclasses must override this method to implement their specific functionality.

        :param context: Current context data dictionary
        :type context: dict[str, Any]
        :return: Empty dictionary in base implementation
        :rtype: dict[str, Any]
        """
        return {}
