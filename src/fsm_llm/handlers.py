from __future__ import annotations

"""
FSM Handler System: A Comprehensive Framework for Self-Determining Function Handlers in FSM-LLM.

This module provides a sophisticated and flexible architecture for executing custom functions
during Finite State Machine (FSM) execution. The key innovation is that each handler contains
its own logic for determining when it should run, making the system highly extensible and
maintainable.

Overview
--------
The handler system operates on the principle of self-determination: rather than having a
central dispatcher decide which handlers to run, each handler implements its own
``should_execute()`` method that evaluates the current FSM state and context to determine
if it should be activated.

Core Components
---------------
1. **HandlerTiming**: Enum defining execution hook points throughout the FSM lifecycle
2. **FSMHandler**: Protocol defining the interface for all handlers
3. **HandlerSystem**: Central orchestrator that manages and executes handlers
4. **BaseHandler**: Base implementation class for creating custom handlers
5. **HandlerBuilder**: Fluent API for creating handlers using lambda functions
6. **LambdaHandler**: Internal implementation for lambda-based handlers

Architecture
------------
The system follows these key architectural principles:

- **Self-Determination**: Each handler decides when it should execute
- **Priority-Based Execution**: Handlers execute in priority order (lower numbers first)
- **Error Isolation**: Handler failures don't break the entire system
- **Context Awareness**: Handlers have full access to FSM state and context
- **Flexible Conditions**: Support for complex execution conditions via lambdas

Handler Execution Flow
----------------------
1. **Registration**: Handlers are registered with the HandlerSystem
2. **Filtering**: At each timing point, potentially applicable handlers are filtered
3. **Condition Evaluation**: Each handler's ``should_execute()`` method is called
4. **Execution**: Qualifying handlers execute in priority order
5. **Context Update**: Handler results are merged into the FSM context
6. **Error Handling**: Failures are handled according to the configured error mode

Usage Patterns
--------------

Basic Handler Creation::

    class MyHandler(BaseHandler):
        def should_execute(self, timing, current_state, target_state, context, updated_keys):
            return timing == HandlerTiming.PRE_PROCESSING and current_state == "collecting_info"

        def execute(self, context):
            return {"processed": True}

Lambda-Based Handler Creation::

    handler = (create_handler("my_handler")
               .at(HandlerTiming.POST_TRANSITION)
               .on_state("completed")
               .do(lambda ctx: {"completion_time": datetime.now().isoformat()}))

Advanced Conditional Logic::

    handler = (create_handler("conditional_handler")
               .when(lambda timing, state, target, ctx, keys:
                     ctx.get("user_score", 0) > 80 and "premium" in ctx.get("features", []))
               .do(lambda ctx: enable_premium_features(ctx)))
"""

import concurrent.futures
import copy
import traceback
from collections.abc import Callable
from enum import Enum
from typing import Any, Protocol

# --------------------------------------------------------------
# Local imports
# --------------------------------------------------------------
from .logging import logger
from .types import FSMError

# --------------------------------------------------------------
# Enumerations and Type Definitions
# --------------------------------------------------------------


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


# Type aliases for better code readability and type safety
ExecutionLambda = Callable[[dict[str, Any]], dict[str, Any]]
"""Type alias for execution lambda functions."""

ConditionLambda = Callable[
    [HandlerTiming, str, str | None, dict[str, Any], set[str] | None], bool
]
"""Type alias for condition evaluation lambda functions."""


# --------------------------------------------------------------
# Protocol Definitions
# --------------------------------------------------------------


class FSMHandler(Protocol):
    """
    Protocol defining the interface for self-determining FSM handlers.

    This protocol establishes the contract that all handlers must implement to participate
    in the FSM execution lifecycle. The key innovation is the ``should_execute`` method,
    which allows each handler to make autonomous decisions about when to run.

    The protocol supports both synchronous and asynchronous execution patterns,
    with priority-based ordering for deterministic execution sequences.
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


# --------------------------------------------------------------
# Exception Classes
# --------------------------------------------------------------


class HandlerSystemError(FSMError):
    """
    Base exception class for all handler system related errors.

    This serves as the root of the exception hierarchy for the handler system,
    allowing clients to catch all handler-related exceptions with a single except clause.
    Inherits from FSMError so that ``except FSMError`` catches handler errors too.
    """

    pass


# --------------------------------------------------------------


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
        super().__init__(f"Error in handler {handler_name}: {original_error!s}")


# --------------------------------------------------------------
# Core System Classes
# --------------------------------------------------------------


class HandlerSystem:
    """
    Central orchestrator for executing custom functions during FSM execution.

    This class manages the registration and execution of handlers, providing error
    handling and context management.
    """

    def __init__(
        self, error_mode: str = "continue", handler_timeout: float | None = None
    ):
        """
        Initialize the handler system with specified error handling behavior.

        :param error_mode: How to handle errors during handler execution
        :type error_mode: str
        :param handler_timeout: Maximum seconds a handler may run before timeout.
            ``None`` disables timeout (default). Use
            ``constants.DEFAULT_HANDLER_TIMEOUT`` (30 s) for safety.
        :type handler_timeout: float | None
        :raises ValueError: If error_mode is not one of: continue, raise
        """
        self.handlers: list[FSMHandler] = []
        self.error_mode = error_mode
        self.handler_timeout = handler_timeout
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None

        # Validate error mode parameter
        valid_modes = ["continue", "raise"]
        if error_mode not in valid_modes:
            raise ValueError(
                f"Invalid error_mode: {error_mode}. Must be one of {valid_modes}"
            )

    def register_handler(self, handler: FSMHandler) -> None:
        """
        Register a new handler with the system and maintain priority ordering.

        :param handler: The handler instance to register
        :type handler: FSMHandler
        """
        self.handlers.append(handler)
        # Maintain sorted order by priority after adding new handler
        self.handlers.sort(key=lambda h: getattr(h, "priority", 100))

    def execute_handlers(
        self,
        timing: HandlerTiming,
        current_state: str,
        target_state: str | None,
        context: dict[str, Any],
        updated_keys: set[str] | None = None,
    ) -> dict[str, Any]:
        """
        Execute all qualifying handlers at the specified timing point.

        Properly cascade context updates between handlers, ensuring that each handler
        sees the cumulative changes made by previous handlers.

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
        :return: Dictionary containing all context updates from executed handlers
        :rtype: dict[str, Any]
        """
        updated_context = copy.deepcopy(context)
        output_context = {}

        # Pre-filter handlers to optimize performance by avoiding unnecessary should_execute calls
        potential_handlers = [
            h
            for h in self.handlers
            if not hasattr(h, "timings") or h.timings is None or timing in h.timings
        ]

        # Execute applicable handlers in priority order (lower priority numbers first)
        for handler in potential_handlers:
            handler_name = getattr(handler, "name", handler.__class__.__name__)

            try:
                # Check if this handler should execute based on current conditions
                if handler.should_execute(
                    timing, current_state, target_state, updated_context, updated_keys
                ):
                    logger.debug(f"Executing handler {handler_name} at {timing.name}")

                    # Execute the handler with optional timeout
                    result = self._execute_single_handler(
                        handler, updated_context, handler_name
                    )

                    # Update context with handler result if valid
                    if result and isinstance(result, dict):
                        updated_context.update(result)
                        output_context.update(result)

                    logger.debug(f"Handler {handler_name} completed successfully")

            except Exception as e:
                # Create structured error with context about the failed handler
                error = HandlerExecutionError(handler_name, e)
                logger.error(f"{error!s}\n{traceback.format_exc()}")

                # Handle the error according to the configured error mode
                # Critical handlers always raise, even in "continue" mode
                is_critical = getattr(handler, "critical", False)
                if self.error_mode == "raise" or is_critical:
                    raise error from e
                elif self.error_mode == "continue":
                    continue  # Log the error and continue to next handler

        return output_context

    def _execute_single_handler(
        self, handler: FSMHandler, context: dict[str, Any], handler_name: str
    ) -> dict[str, Any] | None:
        """Execute a single handler, optionally with timeout protection.

        When ``handler_timeout`` is set, the handler runs in a shared thread
        pool and is interrupted if it exceeds the timeout.
        """
        if self.handler_timeout is None:
            return handler.execute(context)

        if self._executor is None:
            self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        future = self._executor.submit(handler.execute, context)
        try:
            return future.result(timeout=self.handler_timeout)
        except concurrent.futures.TimeoutError as e:
            raise TimeoutError(
                f"Handler '{handler_name}' timed out after {self.handler_timeout}s"
            ) from e

    def close(self) -> None:
        """Shut down the handler executor pool, if active."""
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None


# --------------------------------------------------------------
# Base Handler Implementation
# --------------------------------------------------------------


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


# --------------------------------------------------------------
# Fluent Builder Interface
# --------------------------------------------------------------


class HandlerBuilder:
    """
    Fluent interface builder for creating FSM handlers using lambda functions.

    The HandlerBuilder provides a convenient and readable way to create handlers
    without needing to implement full classes. It supports complex conditional
    logic through method chaining and lambda functions.

    Key Features:
    - Fluent method chaining for readable configuration
    - Support for multiple condition types (timing, state, context, etc.)
    - Custom condition lambdas for complex logic
    - Both synchronous and asynchronous execution support
    - Priority-based execution control

    Example Usage::

        handler = (create_handler("data_validator")
                   .at(HandlerTiming.PRE_PROCESSING)
                   .on_state("collecting_data")
                   .when_context_has("user_input")
                   .with_priority(50)
                   .do(lambda ctx: validate_and_clean_data(ctx)))
    """

    def __init__(self, name: str = "LambdaHandler"):
        """
        Initialize the handler builder with default configuration.

        :param name: Name for the generated handler (used in logs and debugging)
        :type name: str
        """
        self.name = name
        self.condition_lambdas: list[ConditionLambda] = []
        self.execution_lambda: ExecutionLambda | None = None
        self.timings: set[HandlerTiming] | None = None
        self.states: set[str] = set()
        self.target_states: set[str] = set()
        self.required_keys: set[str] = set()
        self.updated_keys: set[str] = set()
        self.priority: int = 100
        self.not_states: set[str] = set()
        self.not_target_states: set[str] = set()

    def with_priority(self, priority: int) -> HandlerBuilder:
        """
        Set the handler's execution priority for controlling execution order.

        :param priority: Priority value where lower numbers execute first
        :type priority: int
        :return: Self for method chaining
        :rtype: HandlerBuilder
        """
        self.priority = priority
        return self

    def when(self, condition: ConditionLambda) -> HandlerBuilder:
        """
        Add a custom condition lambda for complex execution logic.

        The condition lambda receives all the context information and should
        return True when the handler should execute. Multiple conditions
        can be added and all must evaluate to True for execution.

        :param condition: Lambda function that returns True when handler should execute
        :type condition: ConditionLambda
        :return: Self for method chaining
        :rtype: HandlerBuilder
        """
        self.condition_lambdas.append(condition)
        return self

    def at(self, *timings: HandlerTiming) -> HandlerBuilder:
        """
        Specify one or more timing points when the handler should execute.

        :param timings: One or more HandlerTiming values
        :type timings: HandlerTiming
        :return: Self for method chaining
        :rtype: HandlerBuilder
        """
        if self.timings is None:
            self.timings = set()
        self.timings.update(timings)
        return self

    def on_state(self, *states: str) -> HandlerBuilder:
        """
        Execute only when the FSM is in one of the specified current states.

        :param states: State IDs to match against current_state
        :type states: str
        :return: Self for method chaining
        :rtype: HandlerBuilder
        """
        self.states.update(states)
        return self

    def not_on_state(self, *states: str) -> HandlerBuilder:
        """
        Do not execute when the FSM is in any of the specified current states.

        :param states: State IDs that should not match current_state
        :type states: str
        :return: Self for method chaining
        :rtype: HandlerBuilder
        """
        self.not_states.update(states)
        return self

    def on_target_state(self, *states: str) -> HandlerBuilder:
        """
        Execute only when transitioning to one of the specified target states.

        :param states: State IDs to match against target_state
        :type states: str
        :return: Self for method chaining
        :rtype: HandlerBuilder
        """
        self.target_states.update(states)
        return self

    def not_on_target_state(self, *states: str) -> HandlerBuilder:
        """
        Do not execute when transitioning to any of the specified target states.

        :param states: State IDs that should not match target_state
        :type states: str
        :return: Self for method chaining
        :rtype: HandlerBuilder
        """
        self.not_target_states.update(states)
        return self

    def when_context_has(self, *keys: str) -> HandlerBuilder:
        """
        Execute only when the context contains all of the specified keys.

        :param keys: Context keys that must be present for execution
        :type keys: str
        :return: Self for method chaining
        :rtype: HandlerBuilder
        """
        self.required_keys.update(keys)
        return self

    def when_keys_updated(self, *keys: str) -> HandlerBuilder:
        """
        Execute only when one or more of the specified context keys are being updated.

        This is particularly useful for CONTEXT_UPDATE timing to react to
        specific data changes.

        :param keys: Context keys to watch for updates
        :type keys: str
        :return: Self for method chaining
        :rtype: HandlerBuilder
        """
        self.updated_keys.update(keys)
        return self

    def on_state_entry(self, *states: str) -> HandlerBuilder:
        """
        Convenient shorthand for executing when entering specific states.

        Equivalent to calling ``.at(HandlerTiming.POST_TRANSITION).on_target_state(*states)``

        :param states: Target states that trigger execution upon entry
        :type states: str
        :return: Self for method chaining
        :rtype: HandlerBuilder
        """
        if self.timings is None:
            self.timings = set()
        self.timings.add(HandlerTiming.POST_TRANSITION)
        self.target_states.update(states)
        return self

    def on_state_exit(self, *states: str) -> HandlerBuilder:
        """
        Convenient shorthand for executing when exiting specific states.

        Equivalent to calling ``.at(HandlerTiming.PRE_TRANSITION).on_state(*states)``

        :param states: Current states that trigger execution upon exit
        :type states: str
        :return: Self for method chaining
        :rtype: HandlerBuilder
        """
        if self.timings is None:
            self.timings = set()
        self.timings.add(HandlerTiming.PRE_TRANSITION)
        self.states.update(states)
        return self

    def on_context_update(self, *keys: str) -> HandlerBuilder:
        """
        Convenient shorthand for executing when specific context keys are updated.

        Equivalent to calling ``.at(HandlerTiming.CONTEXT_UPDATE).when_keys_updated(*keys)``

        :param keys: Context keys to watch for updates
        :type keys: str
        :return: Self for method chaining
        :rtype: HandlerBuilder
        """
        if self.timings is None:
            self.timings = set()
        self.timings.add(HandlerTiming.CONTEXT_UPDATE)
        self.updated_keys.update(keys)
        return self

    def do(self, execution: ExecutionLambda) -> BaseHandler:
        """
        Set the execution lambda and build the final handler instance.

        This method completes the builder pattern by providing the actual execution
        logic and returning a configured handler ready for registration.

        :param execution: Lambda or function that performs the handler's work
        :type execution: ExecutionLambda
        :return: Configured LambdaHandler instance ready for use
        :rtype: LambdaHandler
        :raises ValueError: If called before setting execution logic
        """
        self.execution_lambda = execution
        return self.build()

    def build(self) -> BaseHandler:
        """
        Build a handler from the current configuration.

        This method creates the final handler instance based on all the configuration
        set through the fluent interface. It automatically detects whether the
        execution lambda is async and configures the handler appropriately.

        :return: Configured BaseHandler instance
        :rtype: BaseHandler
        :raises ValueError: If execution lambda is not set
        """
        if not self.execution_lambda:
            raise ValueError("Execution lambda is required - use .do() to set it")

        # Create a handler instance with all the configured parameters
        handler = LambdaHandler(
            name=self.name,
            condition_lambdas=self.condition_lambdas.copy(),
            execution_lambda=self.execution_lambda,
            timings=self.timings.copy() if self.timings is not None else None,
            states=self.states.copy(),
            target_states=self.target_states.copy(),
            required_keys=self.required_keys.copy(),
            updated_keys=self.updated_keys.copy(),
            priority=self.priority,
            not_states=self.not_states.copy(),
            not_target_states=self.not_target_states.copy(),
        )

        return handler


# --------------------------------------------------------------
# Convenience Functions
# --------------------------------------------------------------


def create_handler(name: str = "LambdaHandler") -> HandlerBuilder:
    """
    Create a new handler builder instance for fluent handler construction.

    This is the primary entry point for creating handlers using the builder pattern.
    It returns a HandlerBuilder that can be configured through method chaining.

    Example::

        handler = (create_handler("my_handler")
                   .at(HandlerTiming.PRE_PROCESSING)
                   .on_state("active")
                   .do(lambda ctx: {"processed": True}))

    :param name: Name for the generated handler (used in logs and debugging)
    :type name: str
    :return: New HandlerBuilder instance ready for configuration
    :rtype: HandlerBuilder
    """
    return HandlerBuilder(name)


# --------------------------------------------------------------
# Internal Implementation Classes
# --------------------------------------------------------------


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
        """
        super().__init__(name=name, priority=priority)
        self.condition_lambdas = condition_lambdas
        self.execution_lambda = execution_lambda
        self.timings = timings
        self.states = states
        self.target_states = target_states
        self.required_keys = required_keys
        self.updated_keys = updated_keys
        self.not_states = (
            {not_states}
            if isinstance(not_states, str)
            else set(not_states)
            if not_states is not None
            else set()
        )
        self.not_target_states = (
            {not_target_states}
            if isinstance(not_target_states, str)
            else set(not_target_states)
            if not_target_states is not None
            else set()
        )

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
        :return: Dictionary containing context updates
        :rtype: dict[str, Any]
        :raises HandlerExecutionError: If execution fails
        """
        try:
            result = self.execution_lambda(context)

            # Ensure we always return a dict
            if result is None:
                return {}
            elif isinstance(result, dict):
                return result
            else:
                raise HandlerExecutionError(
                    self.name,
                    TypeError(
                        f"Handler returned non-dict result: {type(result).__name__}"
                    ),
                )

        except Exception as e:
            logger.error(f"Error in {self.name}: {e!s}")
            raise HandlerExecutionError(self.name, e) from e

    def __str__(self) -> str:
        """
        Return string representation for debugging and logging purposes.

        :return: String representation of the handler
        :rtype: str
        """
        return f"{self.name} (Lambda Handler)"


# --------------------------------------------------------------
# R5 — Handlers as AST transformers
# --------------------------------------------------------------
#
# Per `plans/plan_2026-04-27_43d56276/plan.md` and D-PLAN-02, R5 reframes
# handler execution: rather than the FSM dialog pipeline calling
# ``HandlerSystem.execute_handlers`` as Python middleware around
# ``Executor.run`` (the pre-R5 model — see ``HandlerSystem.execute_handlers``
# above), the handler dispatch is **spliced into the compiled λ-term itself**
# at the appropriate structural seam, and runs inside the executor as a
# ``Combinator(op=HOST_CALL, ...)`` invocation. The host-callable bound at
# the env name :data:`HANDLER_RUNNER_VAR_NAME` is the bridge.
#
# This module section adds:
#
# 1. :data:`HANDLER_RUNNER_VAR_NAME` — the canonical env-binding name.
# 2. :func:`make_handler_runner` — factory producing the host-callable that
#    the AST splicer references. Internally delegates to
#    ``HandlerSystem.execute_handlers`` (the legacy middleware path) so the
#    semantics are byte-equivalent to pre-R5; only the call site moves from
#    Python middleware to AST-driven invocation.
# 3. :func:`compose` — top-level entry: ``compose(term, handlers, ...) ->
#    Term``. Applies all 8 splice functions in deterministic order to the
#    input term and returns a new term. Idempotent for an empty handler
#    list (returns the input term unchanged).
# 4. Eight ``_splice_<timing>`` AST rewriters — one per
#    :class:`HandlerTiming` value. Each takes a ``Term`` and returns a new
#    ``Term``. Five are real AST rewriters; three (START_CONVERSATION,
#    END_CONVERSATION, ERROR) are identity placeholders because those
#    timings fire on the **conversation lifecycle boundary**, not inside
#    the per-turn compiled term:
#
#    * ``START_CONVERSATION`` / ``END_CONVERSATION`` — host-side. The
#      pre-R5 dispatch sites are in ``dialog/fsm.py`` (start_conversation
#      / end_conversation methods); they fire once per conversation, not
#      per turn. The splice function is identity; the host invokes the
#      runner directly through ``make_handler_runner`` so the execution
#      path is unified.
#    * ``ERROR`` — host-trapped. The executor cannot trap exceptions
#      (would breach the kernel's purity invariants), so the host's
#      exception boundary catches and dispatches ERROR handlers via the
#      runner before re-raising. Splice function is identity.
#    * ``PRE_PROCESSING`` / ``POST_PROCESSING`` — turn-level wraps. Spliced
#      around the inner ``Case``-on-state body of the compiled FSM term
#      (i.e. inside the four-deep ``Abs`` chain, around the per-turn body).
#    * ``PRE_TRANSITION`` / ``POST_TRANSITION`` — transition-Case-level
#      wraps. Spliced inside each ``Case`` branch around the
#      transition-evaluation ``Let`` binding.
#    * ``CONTEXT_UPDATE`` — per-Let-binding wrap. Spliced after each
#      extraction-stage ``Let`` so handlers see freshly-bound context keys.
#
# The splicer is a pure AST → AST transformation; it does not import from
# the dialog package and does not depend on any FSM-specific knowledge
# beyond the structural conventions documented above. Steps 3 and 4 of
# plan_43d56276 wire ``compose`` into ``Program``/``API.register_handler``
# and replace the explicit ``self.execute_handlers(...)`` call sites in
# ``dialog/pipeline.py``.
#
# The pre-R5 ``HandlerSystem.execute_handlers`` middleware path remains in
# place above. After step 4 it is no longer called from ``pipeline.py``;
# however, it is still the implementation that ``make_handler_runner``
# delegates to — so the actual handler execution semantics (priority
# ordering, error_mode, timeout, ``should_execute`` filtering) are
# unchanged.

from .runtime.ast import Abs, Combinator, Term
from .runtime.dsl import host_call, let_, var

# Canonical env-binding name for the handler runner host-callable.
# The AST splicer emits ``host_call(HANDLER_RUNNER_VAR_NAME, <timing>, ...)``
# nodes. The dialog/runtime caller (Program/API in step 3, MessagePipeline
# in step 4) is responsible for binding this name in env to the value
# returned by :func:`make_handler_runner`.
HANDLER_RUNNER_VAR_NAME: str = "__fsm_handlers__"

# Sentinel name used as the input_var for synthesized Let bindings that
# discard the handler runner's return value. The runner is invoked for its
# side-effect on the FSM context (the host-callable mutates ``instance``);
# its return value is therefore discarded by the surrounding Let.
_DISCARD_VAR_PREFIX: str = "_fsm_handler_"

# Internal counter for generating fresh _DISCARD_VAR_PREFIX names per
# splice. The names need only be unique within a single ``compose`` call;
# we use a simple module-level counter wrapped by ``_fresh_discard_name``.
_DISCARD_COUNTER: list[int] = [0]


def _fresh_discard_name() -> str:
    """Generate a fresh discard-var name for a synthesized Let binding."""
    _DISCARD_COUNTER[0] += 1
    return f"{_DISCARD_VAR_PREFIX}{_DISCARD_COUNTER[0]}"


# --------------------------------------------------------------
# Public API: compose + handler runner factory
# --------------------------------------------------------------


def make_handler_runner(
    handler_system: HandlerSystem,
) -> Callable[..., dict[str, Any]]:
    """Build the host-callable that the AST splicer references.

    The returned callable has signature
    ``runner(timing_str, current_state, target_state, context, updated_keys)``
    and delegates to :meth:`HandlerSystem.execute_handlers` after coercing
    the ``timing_str`` argument into a :class:`HandlerTiming` enum value.

    The runner must be bound in the executor's env at name
    :data:`HANDLER_RUNNER_VAR_NAME` for spliced terms to evaluate. The
    binding is performed by the dialog-side caller (``Program`` / ``API``
    in step 3, ``MessagePipeline`` in step 4 of plan_43d56276).

    :param handler_system: The :class:`HandlerSystem` whose handlers should
        be invoked when the runner is called from a spliced term.
    :return: A Python callable suitable for binding in an Executor env.
    """

    def runner(
        timing_str: str,
        current_state: str,
        target_state: str | None,
        context: dict[str, Any],
        updated_keys: set[str] | None = None,
    ) -> dict[str, Any]:
        try:
            timing = HandlerTiming(timing_str)
        except ValueError as exc:
            raise HandlerSystemError(
                f"Unknown HandlerTiming value: {timing_str!r}"
            ) from exc
        return handler_system.execute_handlers(
            timing=timing,
            current_state=current_state,
            target_state=target_state,
            context=context,
            updated_keys=updated_keys,
        )

    return runner


def compose(
    term: Term,
    handlers: list[FSMHandler] | None,
    *,
    handler_runner_var: str = HANDLER_RUNNER_VAR_NAME,
) -> Term:
    """Splice handler-invocation seams into a compiled λ-term.

    For each :class:`HandlerTiming` value, the corresponding
    ``_splice_<timing>`` rewriter is applied to ``term`` in deterministic
    order (program-level outermost, then turn-level, then per-Case, then
    per-Let). The result is a new ``Term`` whose evaluation invokes the
    host-callable bound at ``handler_runner_var`` at every splice point.

    Idempotent for ``handlers in (None, [])``: returns ``term`` unchanged
    (zero AST mutation, zero new nodes). This is the back-compat path for
    FSMs registered with no handlers — the splicer must not perturb the
    AST shape.

    The ``handlers`` argument is **not** introspected to filter splice
    points; instead, every timing's splice is unconditionally applied,
    and the host-callable's :meth:`HandlerSystem.execute_handlers`
    delegates to ``should_execute`` per handler at runtime. This mirrors
    the pre-R5 middleware semantics. Per-timing pruning is a future
    optimization (deferred — see D-PLAN-02 trade-off).

    :param term: The compiled λ-term (typically the output of
        ``compile_fsm`` / ``compile_fsm_cached``).
    :param handlers: List of :class:`FSMHandler` instances. ``None`` or
        empty list short-circuits — splicer returns ``term`` unchanged.
    :param handler_runner_var: Env-binding name for the handler runner
        host-callable. Defaults to :data:`HANDLER_RUNNER_VAR_NAME`. Tests
        may override for isolation.
    :return: A new ``Term`` with handler splices, or ``term`` unchanged if
        no handlers were registered.
    """
    if not handlers:
        return term

    # Apply splices in a fixed order. Program-level splices wrap last
    # (outermost) so they fire first/last during evaluation. Inner splices
    # wrap first so they nest inside the program-level wrappers.
    spliced = term
    spliced = _splice_context_update(spliced, handler_runner_var)
    spliced = _splice_pre_transition(spliced, handler_runner_var)
    spliced = _splice_post_transition(spliced, handler_runner_var)
    spliced = _splice_pre_processing(spliced, handler_runner_var)
    spliced = _splice_post_processing(spliced, handler_runner_var)
    spliced = _splice_start_conversation(spliced, handler_runner_var)
    spliced = _splice_end_conversation(spliced, handler_runner_var)
    spliced = _splice_error(spliced, handler_runner_var)
    return spliced


# --------------------------------------------------------------
# Splice helpers
# --------------------------------------------------------------
#
# Each ``_splice_<timing>`` returns a Term. The shared building block
# below, :func:`_handler_invocation`, builds a single
# ``Combinator(HOST_CALL, ...)`` node bound in a Let around a body.
#
# Reserved env names referenced by the splicer (must be bound by the
# dialog-side caller before evaluating a spliced term):
#
# * ``HANDLER_RUNNER_VAR_NAME`` — the host-callable.
# * ``current_state_id`` — the current FSM state id (str).
# * ``target_state_id`` — the next state id (str | None) — only meaningful
#   under PRE/POST_TRANSITION; bound to ``None`` elsewhere.
# * ``context_data`` — the current context dict (dict[str, Any]).
# * ``updated_keys`` — set[str] | None — only meaningful under
#   CONTEXT_UPDATE; bound to ``None`` elsewhere.
#
# These names are documented here (not exported as constants) because
# their binding is the responsibility of the FSM compiler / dialog layer,
# which is the only consumer of the splicer for now. Stdlib factories do
# not use the handler splicer — they have no FSM-state semantics.

CURRENT_STATE_VAR: str = "current_state_id"
TARGET_STATE_VAR: str = "target_state_id"
CONTEXT_DATA_VAR: str = "context_data"
UPDATED_KEYS_VAR: str = "updated_keys"


def _handler_invocation(timing: HandlerTiming, *, runner_var: str) -> Combinator:
    """Build a ``host_call(runner, timing, state, target, ctx, keys)`` node."""
    # The runner expects positional args in the same order as
    # HandlerSystem.execute_handlers' kwargs:
    # (timing_str, current_state, target_state, context, updated_keys).
    # We pass the timing as a Var bound to a string literal in env.
    # Simpler: bind the timing string as a fresh Var whose name is the
    # timing value itself, since the splicer can pre-compute it as a
    # literal — but the executor only resolves Vars from env. To avoid
    # forcing the caller to bind every timing string, we use a tiny shim:
    # the timing is encoded as a Var whose name is a pre-reserved env
    # binding. That binding is added to env by make_handler_runner's
    # env_extension dict (see _splice helpers below).
    #
    # Actual encoding: the splicer emits the timing as a Var named
    # f"_handler_timing_{timing.value}", which the dialog-side caller
    # binds to the literal string ``timing.value`` before each call.
    # That keeps the AST free of Python literal embedding and reuses the
    # existing env-binding machinery.
    timing_var_name = _timing_var_name(timing)
    return host_call(
        runner_var,
        var(timing_var_name),
        var(CURRENT_STATE_VAR),
        var(TARGET_STATE_VAR),
        var(CONTEXT_DATA_VAR),
        var(UPDATED_KEYS_VAR),
    )


def _timing_var_name(timing: HandlerTiming) -> str:
    """Canonical Var name for a timing string literal in env.

    The dialog-side caller pre-binds ``_handler_timing_<value>`` →
    ``<value>`` for every :class:`HandlerTiming`. The splicer references
    these Vars rather than embedding string literals into the AST.
    """
    return f"_handler_timing_{timing.value}"


def required_env_bindings() -> dict[str, str]:
    """Return the static (timing-string) env bindings required by spliced terms.

    The dialog-side caller (Program/API/MessagePipeline) must merge this
    dict into the env passed to ``Executor.run``. The remaining bindings
    (CURRENT_STATE_VAR, TARGET_STATE_VAR, CONTEXT_DATA_VAR,
    UPDATED_KEYS_VAR, HANDLER_RUNNER_VAR_NAME) are runtime-dependent and
    are bound per-turn / per-call by the caller.

    :return: dict mapping ``_handler_timing_<value>`` → ``<value>`` for
        each :class:`HandlerTiming` value.
    """
    return {_timing_var_name(t): t.value for t in HandlerTiming}


# --------------------------------------------------------------
# Outer (program-level) splices — START/END_CONV, ERROR
# --------------------------------------------------------------


def _splice_start_conversation(term: Term, runner_var: str) -> Term:
    """Identity splice — START_CONVERSATION fires on the conversation
    lifecycle boundary, not inside the per-turn compiled term.

    The pre-R5 dispatch site lives in ``dialog/fsm.py:start_conversation``
    (lines ~267 — the `_execute_handlers(START_CONVERSATION, ...)` call).
    That call site is not part of the compiled λ-term — it runs once when
    a new conversation is created, before any turn is processed. Splicing
    it into the per-turn term would fire it on every turn, which is wrong.

    Step 4 of plan_43d56276 keeps the host-side ``_execute_handlers`` call
    at the lifecycle boundary; it routes through ``make_handler_runner``
    so the underlying execution path is unified, but the call site is
    host-side, not term-spliced. This splice function is therefore a
    structural placeholder: identity transform, exists so the 8-timing
    enumeration in :func:`required_env_bindings` and the orchestration in
    :func:`compose` is uniform.
    """
    return term


def _splice_end_conversation(term: Term, runner_var: str) -> Term:
    """Identity splice — END_CONVERSATION is host-side (see
    :func:`_splice_start_conversation` for rationale). Lifecycle boundary
    in ``dialog/fsm.py:end_conversation`` (lines ~292, ~305, ~567)."""
    return term


def _splice_error(term: Term, runner_var: str) -> Term:
    """Identity splice — ERROR handlers are host-trapped.

    The executor has no try/except machinery (would breach the kernel's
    purity invariants). The host (``dialog/fsm.py`` line ~417 catches at
    the orchestration boundary) catches the exception, then *separately*
    invokes the runner with timing=ERROR before re-raising. This splice
    is a structural placeholder so ``required_env_bindings()`` covers all
    8 timings; the actual ERROR dispatch lives in the host's exception
    handler (see D-PLAN-02 trade-off — host-side error boundary preserved).
    """
    return term


# --------------------------------------------------------------
# Turn-level splices — PRE/POST_PROCESSING
# --------------------------------------------------------------


def _splice_pre_processing(term: Term, runner_var: str) -> Term:
    """Wrap the inner per-turn body of ``term`` so PRE_PROCESSING fires first.

    The compiled FSM term shape (per ``dialog/compile_fsm.py:compile_fsm``)
    is::

        Abs(USER_MSG, Abs(STATE_ID, Abs(CONV_ID, Abs(INSTANCE, Case(...)))))

    The PRE/POST_PROCESSING splice points are inside the innermost ``Abs``
    body, around the ``Case``-on-state. We walk the four-deep Abs chain
    and rewrap the innermost body. If ``term`` is not an Abs (e.g. tests
    pass a bare term), we splice at the top — the caller is responsible
    for ensuring the term shape is appropriate.
    """
    return _wrap_innermost_abs_body(
        term,
        lambda inner: _wrap_pre(inner, HandlerTiming.PRE_PROCESSING, runner_var),
    )


def _splice_post_processing(term: Term, runner_var: str) -> Term:
    """Wrap the inner per-turn body of ``term`` so POST_PROCESSING fires last."""
    return _wrap_innermost_abs_body(
        term,
        lambda inner: _wrap_post(inner, HandlerTiming.POST_PROCESSING, runner_var),
    )


def _wrap_innermost_abs_body(term: Term, transform: Callable[[Term], Term]) -> Term:
    """Recursively walk an ``Abs`` chain and apply ``transform`` to its body.

    For ``Abs(p1, Abs(p2, ..., Abs(pN, body)))`` returns
    ``Abs(p1, Abs(p2, ..., Abs(pN, transform(body))))``. For non-``Abs``
    terms returns ``transform(term)``.

    Pure structural rewrite — does not interpret the body's shape.
    """
    if isinstance(term, Abs):
        return Abs(
            param=term.param,
            body=_wrap_innermost_abs_body(term.body, transform),
        )
    return transform(term)


def _wrap_pre(body: Term, timing: HandlerTiming, runner_var: str) -> Term:
    """Emit ``let _h_N = host_call(...) in body``."""
    h = _handler_invocation(timing, runner_var=runner_var)
    return let_(_fresh_discard_name(), h, body)


def _wrap_post(body: Term, timing: HandlerTiming, runner_var: str) -> Term:
    """Emit ``let _r = body in let _h_N = host_call(...) in _r``."""
    result_name = _fresh_discard_name() + "_result"
    h = _handler_invocation(timing, runner_var=runner_var)
    return let_(
        result_name,
        body,
        let_(_fresh_discard_name(), h, var(result_name)),
    )


# --------------------------------------------------------------
# Identity splices — PRE/POST_TRANSITION + CONTEXT_UPDATE (D-STEP-04-RESOLUTION)
# --------------------------------------------------------------
#
# These three timings are dispatched **host-side** in the R5-narrow scope
# (per Option gamma in plan_43d56276 D-STEP-04-RESOLUTION). The structural
# splicer approach (per-Case-branch wrap for PRE/POST_TRANSITION; per-Let
# wrap for CONTEXT_UPDATE) does not match the existing call-site cardinality
# and conditional gating semantics:
#
#   * PRE_TRANSITION / POST_TRANSITION — the host fires these only when an
#     actual transition is applied (and POST_TRANSITION has rollback-on-
#     failure semantics that require a kernel exception node we do not
#     emit). A per-Case-branch wrap would over-fire on every turn.
#   * CONTEXT_UPDATE — the host call is guarded by ``if extracted_data:``
#     and passes a per-call ``updated_keys`` set. A per-Let wrap cannot
#     thread per-Let key sets without structural changes.
#
# These splice functions therefore remain identity transforms — the
# call sites in ``dialog/pipeline.py`` (lines 629, 806, 1865, 1886) keep
# calling ``MessagePipeline.execute_handlers(...)`` directly. The
# unified execution path is preserved at the
# ``HandlerSystem.execute_handlers`` boundary.
#
# Refining the splicer to honour these cardinality + gating semantics is
# deferred to a follow-up plan with dedicated test coverage. The
# enumeration is kept here so :func:`compose`'s loop is uniform and
# :func:`required_env_bindings` covers all 8 timings without special-
# casing.


def _splice_pre_transition(term: Term, runner_var: str) -> Term:
    """Identity splice — see module docstring above."""
    return term


def _splice_post_transition(term: Term, runner_var: str) -> Term:
    """Identity splice — see module docstring above."""
    return term


def _splice_context_update(term: Term, runner_var: str) -> Term:
    """Identity splice — see module docstring above."""
    return term


# --------------------------------------------------------------
# Module exports — additive (R5)
# --------------------------------------------------------------

# The R5 surface is intentionally exported by name from the module rather
# than via ``__all__`` (this module has no existing ``__all__`` —
# preserving that convention). Public R5 names: ``compose``,
# ``make_handler_runner``, ``HANDLER_RUNNER_VAR_NAME``,
# ``required_env_bindings``. All ``_splice_*`` helpers are private.
