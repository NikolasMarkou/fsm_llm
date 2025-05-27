"""
FSM Handler System: A Comprehensive Framework for Self-Determining Function Handlers in LLM-FSM.

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
6. **_LambdaHandler**: Internal implementation for lambda-based handlers

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

Error Handling Modes
--------------------
- **continue**: Log errors and continue with remaining handlers (default)
- **raise**: Stop execution and raise an exception on first error
- **skip**: Skip the failed handler and continue with others

Performance Considerations
--------------------------
- Handlers are sorted by priority once during registration
- Pre-filtering reduces unnecessary ``should_execute()`` calls
- Context updates are batched and applied efficiently
- Handler execution is tracked for debugging purposes

Thread Safety
-------------
The HandlerSystem is not inherently thread-safe. If used in multi-threaded
environments, external synchronization is required.

Examples
--------
See the module's test files and example implementations for comprehensive usage examples.

Notes
-----
This system is designed to integrate seamlessly with the LLM-FSM framework,
providing extension points for custom business logic, external integrations,
and advanced processing workflows.
"""

import asyncio
import inspect
import traceback
from enum import Enum, auto
from typing import Dict, Any, Callable, List, Optional, Union, Set, Protocol

# --------------------------------------------------------------
# Local imports
# --------------------------------------------------------------

from .logging import logger


# --------------------------------------------------------------
# Enumerations and Type Definitions
# --------------------------------------------------------------

class HandlerTiming(Enum):
    """
    Enumeration defining hook points where handlers can be executed during FSM lifecycle.

    These timing points provide comprehensive coverage of the FSM execution flow,
    allowing handlers to intervene at precisely the right moments for their specific needs.

    :cvar START_CONVERSATION: Triggered when a new conversation begins
    :cvar PRE_PROCESSING: Before the LLM processes user input
    :cvar POST_PROCESSING: After LLM response but before state transition
    :cvar PRE_TRANSITION: After LLM response and before state changes
    :cvar POST_TRANSITION: After the state has been successfully changed
    :cvar CONTEXT_UPDATE: When context is updated with new information
    :cvar END_CONVERSATION: When the conversation terminates
    :cvar ERROR: When an error occurs during FSM execution
    :cvar UNKNOWN: For any other unspecified timing points
    """
    START_CONVERSATION = auto()
    PRE_PROCESSING = auto()
    POST_PROCESSING = auto()
    PRE_TRANSITION = auto()
    POST_TRANSITION = auto()
    CONTEXT_UPDATE = auto()
    END_CONVERSATION = auto()
    ERROR = auto()
    UNKNOWN = auto()


# Type aliases for better code readability and type safety
ExecutionLambda = Callable[[Dict[str, Any]], Dict[str, Any]]
"""Type alias for synchronous execution lambda functions."""

AsyncExecutionLambda = Callable[[Dict[str, Any]], Dict[str, Any]]
"""Type alias for asynchronous execution lambda functions."""

ConditionLambda = Callable[[HandlerTiming, str, Optional[str], Dict[str, Any], Optional[Set[str]]], bool]
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

    def should_execute(self,
                       timing: HandlerTiming,
                       current_state: str,
                       target_state: Optional[str],
                       context: Dict[str, Any],
                       updated_keys: Optional[Set[str]] = None) -> bool:
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
        :type target_state: Optional[str]
        :param context: Current context data dictionary
        :type context: Dict[str, Any]
        :param updated_keys: Set of context keys being updated (for CONTEXT_UPDATE timing)
        :type updated_keys: Optional[Set[str]]
        :return: True if the handler should execute, False otherwise
        :rtype: bool
        """
        ...

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the handler's core logic and return context updates.

        This method performs the actual work of the handler. It receives the current
        context and returns a dictionary of updates to be merged back into the context.

        The method signature is async to support both synchronous and asynchronous
        operations. Synchronous handlers can simply return their results directly.

        :param context: Current context data dictionary
        :type context: Dict[str, Any]
        :return: Dictionary containing context updates to apply
        :rtype: Dict[str, Any]
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

    :param handler_name: Name of the handler that failed
    :type handler_name: str
    :param original_error: The original exception that caused the failure
    :type original_error: Exception
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

    Fixed version that properly cascades context updates between handlers.
    """

    def __init__(self, error_mode: str = "continue"):
        """Initialize the handler system with specified error handling behavior."""
        self.handlers: List[FSMHandler] = []
        self.error_mode = error_mode

        # Validate error mode parameter
        valid_modes = ["continue", "raise", "skip"]
        if error_mode not in valid_modes:
            raise ValueError(f"Invalid error_mode: {error_mode}. Must be one of {valid_modes}")

    def register_handler(self, handler: FSMHandler) -> None:
        """Register a new handler with the system and maintain priority ordering."""
        self.handlers.append(handler)
        # Maintain sorted order by priority after adding new handler
        self.handlers.sort(key=lambda h: getattr(h, 'priority', 100))

    def execute_handlers(self,
                         timing: HandlerTiming,
                         current_state: str,
                         target_state: Optional[str],
                         context: Dict[str, Any],
                         updated_keys: Optional[Set[str]] = None) -> Dict[str, Any]:
        """
        Execute all qualifying handlers at the specified timing point.

        Fixed to properly cascade context updates between handlers.
        """
        updated_context = context.copy()
        executed_handlers = []
        output_context = {}

        # Pre-filter handlers to optimize performance by avoiding unnecessary should_execute calls
        potential_handlers = [h for h in self.handlers
                              if not hasattr(h, 'timings') or
                              not getattr(h, 'timings') or
                              timing in getattr(h, 'timings')]

        # Execute applicable handlers in priority order (lower priority numbers first)
        for handler in potential_handlers:
            handler_name = getattr(handler, 'name', handler.__class__.__name__)

            try:
                # Check if this handler should execute based on current conditions
                if handler.should_execute(timing, current_state, target_state, updated_context, updated_keys):
                    # Log handler execution for debugging and monitoring
                    logger.debug(f"Executing handler {handler_name} at {timing.name}")

                    # Execute the handler and get its result
                    result = handler.execute(updated_context)

                    # Update context with handler result if valid
                    if result and isinstance(result, dict):
                        # CRITICAL FIX: Update the working context so later handlers see changes
                        updated_context.update(result)
                        output_context.update(result)

                        # Track keys that were updated by this handler for debugging
                        handler_updated_keys = set(result.keys())
                        if updated_keys is not None:
                            updated_keys.update(handler_updated_keys)

                    # Track executed handlers for debugging and audit purposes
                    executed_handlers.append({
                        'name': handler_name,
                        'updated_keys': list(result.keys()) if result and isinstance(result, dict) else []
                    })

                    logger.debug(f"Handler {handler_name} completed successfully")

            except Exception as e:
                # Create structured error with context about the failed handler
                error = HandlerExecutionError(handler_name, e)
                logger.error(f"{str(error)}\n{traceback.format_exc()}")

                # Handle the error according to the configured error mode
                if self.error_mode == "raise":
                    raise error
                elif self.error_mode == "continue":
                    continue  # Log the error and continue to next handler
                elif self.error_mode == "skip":
                    continue  # Skip this handler and continue with others

        # Add metadata about executed handlers to context for debugging and audit trails
        if executed_handlers:
            # CRITICAL FIX: Use output_context consistently
            if 'system' not in output_context:
                output_context['system'] = {}
            if 'handlers' not in output_context['system']:
                output_context['system']['handlers'] = {}

            output_context['system']['handlers'][timing.name] = executed_handlers

        return output_context


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

    :param name: Optional name for the handler (defaults to class name)
    :type name: Optional[str]
    :param priority: Execution priority (lower values execute first)
    :type priority: int
    """

    def __init__(self, name: str = None, priority: int = 100):
        """
        Initialize the base handler with name and priority.

        :param name: Optional name for the handler (defaults to class name if None)
        :type name: Optional[str]
        :param priority: Execution priority where lower values indicate higher priority
        :type priority: int
        """
        self.name = name or self.__class__.__name__
        self._priority = priority

    @property
    def priority(self) -> int:
        """
        Get the handler's execution priority.

        :return: Priority value where lower numbers execute first
        :rtype: int
        """
        return self._priority

    def should_execute(self,
                       timing: HandlerTiming,
                       current_state: str,
                       target_state: Optional[str],
                       context: Dict[str, Any],
                       updated_keys: Optional[Set[str]] = None) -> bool:
        """
        Determine if this handler should execute based on current conditions.

        Default implementation always returns False. Subclasses must override this
        method to implement their specific execution conditions.

        :param timing: The lifecycle hook point being executed
        :type timing: HandlerTiming
        :param current_state: Current FSM state identifier
        :type current_state: str
        :param target_state: Target state identifier (None if not transitioning)
        :type target_state: Optional[str]
        :param context: Current context data dictionary
        :type context: Dict[str, Any]
        :param updated_keys: Set of context keys being updated (for CONTEXT_UPDATE timing)
        :type updated_keys: Optional[Set[str]]
        :return: Always False in base implementation
        :rtype: bool
        """
        return False

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the handler's core logic and return context updates.

        Default implementation does nothing and returns an empty dictionary.
        Subclasses must override this method to implement their specific functionality.

        :param context: Current context data dictionary
        :type context: Dict[str, Any]
        :return: Empty dictionary in base implementation
        :rtype: Dict[str, Any]
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

    :param name: Name for the generated handler (used in logs and debugging)
    :type name: str
    """

    def __init__(self, name: str = "LambdaHandler"):
        """
        Initialize the handler builder with default configuration.

        :param name: Name for the generated handler (used in logs and debugging)
        :type name: str
        """
        self.name = name
        self.condition_lambdas: List[ConditionLambda] = []
        self.execution_lambda: Optional[Union[ExecutionLambda, AsyncExecutionLambda]] = None
        self.timings: Set[HandlerTiming] = set()
        self.states: Set[str] = set()
        self.target_states: Set[str] = set()
        self.required_keys: Set[str] = set()
        self.updated_keys: Set[str] = set()
        self.priority: int = 100
        self.not_states: Set[str] = set()
        self.not_target_states: Set[str] = set()

    def with_priority(self, priority: int) -> 'HandlerBuilder':
        """
        Set the handler's execution priority for controlling execution order.

        :param priority: Priority value where lower numbers execute first
        :type priority: int
        :return: Self for method chaining
        :rtype: HandlerBuilder
        """
        self.priority = priority
        return self

    def when(self, condition: ConditionLambda) -> 'HandlerBuilder':
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

    def at(self, *timings: HandlerTiming) -> 'HandlerBuilder':
        """
        Specify one or more timing points when the handler should execute.

        :param timings: One or more HandlerTiming values
        :type timings: HandlerTiming
        :return: Self for method chaining
        :rtype: HandlerBuilder
        """
        self.timings.update(timings)
        return self

    def on_state(self, *states: str) -> 'HandlerBuilder':
        """
        Execute only when the FSM is in one of the specified current states.

        :param states: State IDs to match against current_state
        :type states: str
        :return: Self for method chaining
        :rtype: HandlerBuilder
        """
        self.states.update(states)
        return self

    def not_on_state(self, *states: str) -> 'HandlerBuilder':
        """
        Do not execute when the FSM is in any of the specified current states.

        :param states: State IDs that should not match current_state
        :type states: str
        :return: Self for method chaining
        :rtype: HandlerBuilder
        """
        self.not_states.update(states)
        return self

    def on_target_state(self, *states: str) -> 'HandlerBuilder':
        """
        Execute only when transitioning to one of the specified target states.

        :param states: State IDs to match against target_state
        :type states: str
        :return: Self for method chaining
        :rtype: HandlerBuilder
        """
        self.target_states.update(states)
        return self

    def not_on_target_state(self, *states: str) -> 'HandlerBuilder':
        """
        Do not execute when transitioning to any of the specified target states.

        :param states: State IDs that should not match target_state
        :type states: str
        :return: Self for method chaining
        :rtype: HandlerBuilder
        """
        self.not_target_states.update(states)
        return self

    def when_context_has(self, *keys: str) -> 'HandlerBuilder':
        """
        Execute only when the context contains all of the specified keys.

        :param keys: Context keys that must be present for execution
        :type keys: str
        :return: Self for method chaining
        :rtype: HandlerBuilder
        """
        self.required_keys.update(keys)
        return self

    def when_keys_updated(self, *keys: str) -> 'HandlerBuilder':
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

    def on_state_entry(self, *states: str) -> 'HandlerBuilder':
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

    def on_state_exit(self, *states: str) -> 'HandlerBuilder':
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

    def on_context_update(self, *keys: str) -> 'HandlerBuilder':
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

    def do(self, execution: Union[ExecutionLambda, AsyncExecutionLambda]) -> BaseHandler:
        """
        Set the execution lambda and build the final handler instance.

        This method completes the builder pattern by providing the actual execution
        logic and returning a configured handler ready for registration.

        :param execution: Lambda or function that performs the handler's work
        :type execution: Union[ExecutionLambda, AsyncExecutionLambda]
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

        # Check if the execution lambda is async for proper handling
        is_async = inspect.iscoroutinefunction(self.execution_lambda)

        # Create a handler instance with all the configured parameters
        handler = LambdaHandler(
            name=self.name,
            condition_lambdas=self.condition_lambdas.copy(),
            execution_lambda=self.execution_lambda,
            is_async=is_async,
            timings=self.timings.copy(),
            states=self.states.copy(),
            target_states=self.target_states.copy(),
            required_keys=self.required_keys.copy(),
            updated_keys=self.updated_keys.copy(),
            priority=self.priority,
            not_states=self.not_states.copy(),
            not_target_states=self.not_target_states.copy()
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

    Fixed version that properly handles sync/async execution and returns dicts.
    """

    def __init__(
            self,
            name: str,
            condition_lambdas: List[ConditionLambda],
            execution_lambda: Union[ExecutionLambda, AsyncExecutionLambda],
            is_async: bool,
            timings: Set[HandlerTiming],
            states: Set[str],
            target_states: Set[str],
            required_keys: Set[str],
            updated_keys: Set[str],
            priority: int = 100,
            not_states: Set[str] = None,
            not_target_states: Set[str] = None
    ):
        """Initialize the lambda handler with all configuration from the builder."""
        super().__init__(name=name, priority=priority)
        self.condition_lambdas = condition_lambdas
        self.execution_lambda = execution_lambda
        self.is_async = is_async
        self.timings = timings
        self.states = states
        self.target_states = target_states
        self.required_keys = required_keys
        self.updated_keys = updated_keys
        self.not_states = not_states or set()
        self.not_target_states = not_target_states or set()

    def should_execute(self,
                       timing: HandlerTiming,
                       current_state: str,
                       target_state: Optional[str],
                       context: Dict[str, Any],
                       updated_keys: Optional[Set[str]] = None) -> bool:
        """Determine if this handler should execute based on builder configuration."""
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
        if self.target_states and (not target_state or target_state not in self.target_states):
            return False

        # Check target state exclusion constraints
        if self.not_target_states and target_state and target_state in self.not_target_states:
            return False

        # Check required context keys constraints
        if self.required_keys and not all(key in context for key in self.required_keys):
            return False

        # Check updated keys constraints (for CONTEXT_UPDATE timing)
        if self.updated_keys and (not updated_keys or not any(key in updated_keys for key in self.updated_keys)):
            return False

        # Evaluate custom condition lambdas - all must return True
        for condition in self.condition_lambdas:
            try:
                if not condition(timing, current_state, target_state, context, updated_keys):
                    return False
            except Exception as e:
                logger.warning(f"Error in condition lambda for {self.name}: {str(e)}")
                return False

        # All conditions passed - handler should execute
        return True

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the handler's lambda function with appropriate async handling.

        Fixed to properly handle async/sync execution and always return dict.
        """
        try:
            if self.is_async:
                # For async lambdas, we need to run them in an event loop
                # Check if there's already an event loop running
                try:
                    loop = asyncio.get_running_loop()
                    # We're already in an async context, create a task
                    future = asyncio.create_task(self.execution_lambda(context))
                    # Wait for it synchronously (blocking)
                    result = asyncio.run_coroutine_threadsafe(
                        self.execution_lambda(context), loop
                    ).result()
                except RuntimeError:
                    # No event loop running, create one
                    result = asyncio.run(self.execution_lambda(context))
            else:
                # Synchronous execution - just call it directly
                result = self.execution_lambda(context)

            # Ensure we always return a dict
            if result is None:
                return {}
            elif isinstance(result, dict):
                return result
            else:
                logger.warning(f"Handler {self.name} returned non-dict result: {type(result)}")
                return {}

        except Exception as e:
            logger.error(f"Error in {self.name}: {str(e)}")
            # Re-raise as HandlerExecutionError to be handled by HandlerSystem
            raise HandlerExecutionError(self.name, e)

    def __str__(self) -> str:
        """Return string representation for debugging and logging purposes."""
        return f"{self.name} (Lambda Handler)"

# --------------------------------------------------------------