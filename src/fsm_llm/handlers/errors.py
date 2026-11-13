"""Handler system exceptions.

Package ``fsm_llm.handlers`` (plan d454fd62, step 77: split out of the old module
``fsm_llm/handlers.py``). ``HandlerSystemError`` is the root of the handler errors and
``HandlerExecutionError`` wraps a failed handler.
"""

from __future__ import annotations

from typing import Any

from fsm_llm.core.definitions import FSMError


class HandlerSystemError(FSMError):
    """
    Base exception class for all handler system related errors.

    This serves as the root of the exception hierarchy for the handler system,
    allowing clients to catch all handler-related exceptions with a single except clause.
    Inherits from FSMError so that ``except FSMError`` catches handler errors too.
    """

    pass


class HandlerExecutionError(HandlerSystemError):
    """
    Exception raised when a handler execution fails during runtime.

    This exception wraps the original error that occurred during handler execution,
    providing additional context about which handler failed and preserving the
    original exception for debugging purposes.
    """

    def __init__(self, handler_name: str, original_error: Exception):
        """
        Initialize the handler execution error.

        :param handler_name: Name of the handler that failed
        :type handler_name: str
        :param original_error: The original exception that caused the failure
        :type original_error: Exception
        """
        self.handler_name = handler_name
        self.original_error = original_error
        # Updates from handlers that succeeded *before* this failure (see D-006).
        self.partial_context: dict[str, Any] = {}
        super().__init__(f"Error in handler {handler_name}: {original_error!s}")

    def __reduce__(self):
        # Exception pickling replays ``cls(*self.args)`` and ``args`` holds only
        # the formatted message, so the default protocol could not rebuild this
        # two-argument constructor (C6). Rebuild from the real arguments and
        # restore the instance attributes (``partial_context``, ``details``).
        return (
            self.__class__,
            (self.handler_name, self.original_error),
            self.__dict__.copy(),
        )
