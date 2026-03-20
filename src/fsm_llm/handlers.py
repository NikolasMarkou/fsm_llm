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

import traceback
import concurrent.futures
from enum import Enum
from typing import Any, Callable, Protocol

# --------------------------------------------------------------
# Local imports
# --------------------------------------------------------------

from .logging import logger


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


class HandlerSystemError(Exception):
    """
    Base exception class for all handler system related errors.

    This serves as the root of the exception hierarchy for the handler system,
    allowing clients to catch all handler-related exceptions with a single except clause.
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
        super().__init__(f"Error in handler {handler_name}: {str(original_error)}")


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
        updated_context = context.copy()
        executed_handlers = []
        output_context = {}

        # Pre-filter handlers to optimize performance by avoiding unnecessary should_execute calls
        potential_handlers = [
            h
            for h in self.handlers
            if not hasattr(h, "timings")
            or not getattr(h, "timings")
            or timing in getattr(h, "timings")
        ]

        # Execute applicable handlers in priority order (lower priority numbers first)
        for handler in potential_handlers:
            handler_name = getattr(handler, "name", handler.__class__.__name__)

            try:
                # Check if this handler should execute based on current conditions
                if handler.should_execute(
                    timing, current_state, target_state, updated_context, updated_keys
                ):
                    # Log handler execution for debugging and monitoring
                    logger.debug(f"Executing handler {handler_name} at {timing.name}")

                    # Execute the handler with optional timeout
                    result = self._execute_single_handler(
                        handler, updated_context, handler_name
                    )

                    # Update context with handler result if valid
                    if result and isinstance(result, dict):
                        # CRITICAL FIX: Update the working context so later handlers see changes
                        updated_context.update(result)
                        output_context.update(result)

                    # Track executed handlers for debugging and audit purposes
                    executed_handlers.append(
                        {
                            "name": handler_name,
                            "updated_keys": list(result.keys())
                            if result and isinstance(result, dict)
                            else [],
                        }
                    )

                    logger.debug(f"Handler {handler_name} completed successfully")

            except Exception as e:
                # Create structured error with context about the failed handler
                error = HandlerExecutionError(handler_name, e)
                logger.error(f"{str(error)}\n{traceback.format_exc()}")

                # Handle the error according to the configured error mode
                # Critical handlers always raise, even in "continue" mode
                is_critical = getattr(handler, "critical", False)
                if self.error_mode == "raise" or is_critical:
                    raise error
                elif self.error_mode == "continue":
                    continue  # Log the error and continue to next handler

        # Track handler execution metadata internally (not in user context)
        if executed_handlers:
            if not hasattr(self, "_execution_metadata"):
                self._execution_metadata = {}
            self._execution_metadata[timing.name] = executed_handlers

        return output_context

    def _execute_single_handler(
        self, handler: FSMHandler, context: dict[str, Any], handler_name: str
    ) -> dict[str, Any] | None:
        """Execute a single handler, optionally with timeout protection.

        When ``handler_timeout`` is set, the handler runs in a thread pool
        and is interrupted if it exceeds the timeout.
        """
        if self.handler_timeout is None:
            return handler.execute(context)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(handler.execute, context)
            try:
                return future.result(timeout=self.handler_timeout)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(
                    f"Handler '{handler_name}' timed out after {self.handler_timeout}s"
                )


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
        self.timings: set[HandlerTiming] = set()
        self.states: set[str] = set()
        self.target_states: set[str] = set()
        self.required_keys: set[str] = set()
        self.updated_keys: set[str] = set()
        self.priority: int = 100
        self.not_states: set[str] = set()
        self.not_target_states: set[str] = set()

    def with_priority(self, priority: int) -> "HandlerBuilder":
        """
        Set the handler's execution priority for controlling execution order.

        :param priority: Priority value where lower numbers execute first
        :type priority: int
        :return: Self for method chaining
        :rtype: HandlerBuilder
        """
        self.priority = priority
        return self

    def when(self, condition: ConditionLambda) -> "HandlerBuilder":
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

    def at(self, *timings: HandlerTiming) -> "HandlerBuilder":
        """
        Specify one or more timing points when the handler should execute.

        :param timings: One or more HandlerTiming values
        :type timings: HandlerTiming
        :return: Self for method chaining
        :rtype: HandlerBuilder
        """
        self.timings.update(timings)
        return self

    def on_state(self, *states: str) -> "HandlerBuilder":
        """
        Execute only when the FSM is in one of the specified current states.

        :param states: State IDs to match against current_state
        :type states: str
        :return: Self for method chaining
        :rtype: HandlerBuilder
        """
        self.states.update(states)
        return self

    def not_on_state(self, *states: str) -> "HandlerBuilder":
        """
        Do not execute when the FSM is in any of the specified current states.

        :param states: State IDs that should not match current_state
        :type states: str
        :return: Self for method chaining
        :rtype: HandlerBuilder
        """
        self.not_states.update(states)
        return self

    def on_target_state(self, *states: str) -> "HandlerBuilder":
        """
        Execute only when transitioning to one of the specified target states.

        :param states: State IDs to match against target_state
        :type states: str
        :return: Self for method chaining
        :rtype: HandlerBuilder
        """
        self.target_states.update(states)
        return self

    def not_on_target_state(self, *states: str) -> "HandlerBuilder":
        """
        Do not execute when transitioning to any of the specified target states.

        :param states: State IDs that should not match target_state
        :type states: str
        :return: Self for method chaining
        :rtype: HandlerBuilder
        """
        self.not_target_states.update(states)
        return self

    def when_context_has(self, *keys: str) -> "HandlerBuilder":
        """
        Execute only when the context contains all of the specified keys.

        :param keys: Context keys that must be present for execution
        :type keys: str
        :return: Self for method chaining
        :rtype: HandlerBuilder
        """
        self.required_keys.update(keys)
        return self

    def when_keys_updated(self, *keys: str) -> "HandlerBuilder":
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

    def on_state_entry(self, *states: str) -> "HandlerBuilder":
        """
        Convenient shorthand for executing when entering specific states.

        Equivalent to calling ``.at(HandlerTiming.POST_TRANSITION).on_target_state(*states)``

        :param states: Target states that trigger execution upon entry
        :type states: str
        :return: Self for method chaining
        :rtype: HandlerBuilder
        """
        self.timings.add(HandlerTiming.POST_TRANSITION)
        self.target_states.update(states)
        return self

    def on_state_exit(self, *states: str) -> "HandlerBuilder":
        """
        Convenient shorthand for executing when exiting specific states.

        Equivalent to calling ``.at(HandlerTiming.PRE_TRANSITION).on_state(*states)``

        :param states: Current states that trigger execution upon exit
        :type states: str
        :return: Self for method chaining
        :rtype: HandlerBuilder
        """
        self.timings.add(HandlerTiming.PRE_TRANSITION)
        self.states.update(states)
        return self

    def on_context_update(self, *keys: str) -> "HandlerBuilder":
        """
        Convenient shorthand for executing when specific context keys are updated.

        Equivalent to calling ``.at(HandlerTiming.CONTEXT_UPDATE).when_keys_updated(*keys)``

        :param keys: Context keys to watch for updates
        :type keys: str
        :return: Self for method chaining
        :rtype: HandlerBuilder
        """
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
        :return: Configured BaseHandler instance ready for use
        :rtype: BaseHandler
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
            timings=self.timings.copy(),
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
        timings: set[HandlerTiming],
        states: set[str],
        target_states: set[str],
        required_keys: set[str],
        updated_keys: set[str],
        priority: int = 100,
        not_states: set[str] = None,
        not_target_states: set[str] = None,
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
        self.not_states = not_states or set()
        self.not_target_states = not_target_states or set()

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

        # Check timing constraints - if specified timings don't include current timing, reject
        if self.timings and timing not in self.timings:
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
                logger.warning(f"Error in condition lambda for {self.name}: {str(e)}")
                return False

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
                logger.error(
                    f"Handler {self.name} returned non-dict result: {type(result)}; discarding"
                )
                return {}

        except Exception as e:
            logger.error(f"Error in {self.name}: {str(e)}")
            raise HandlerExecutionError(self.name, e)

    def __str__(self) -> str:
        """
        Return string representation for debugging and logging purposes.

        :return: String representation of the handler
        :rtype: str
        """
        return f"{self.name} (Lambda Handler)"


# --------------------------------------------------------------
