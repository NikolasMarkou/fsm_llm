"""
FSM Handler System: A framework for adding self-determining function handlers to LLM-FSM.

This module provides a flexible architecture for executing custom functions
during FSM execution where each handler contains its own logic for when it should run.
"""

import asyncio
import inspect
import traceback
from enum import Enum, auto
from typing import Dict, Any, Callable, List, Optional, Union, Set, Protocol, Tuple

# --------------------------------------------------------------
# local imports
# --------------------------------------------------------------

from .logging import logger

# --------------------------------------------------------------

# Define hook points where handlers can be executed
class HandlerTiming(Enum):
    START_CONVERSATION = auto()  # When the conversation starts
    PRE_PROCESSING = auto()  # Before LLM processes the user input
    POST_PROCESSING = auto()  # After LLM has responded but before state transition
    PRE_TRANSITION = auto()  # After LLM response and before state changes
    POST_TRANSITION = auto()  # After state has changed
    CONTEXT_UPDATE = auto()  # When context is updated with new information
    END_CONVERSATION = auto()  # When the conversation ends
    ERROR = auto()  # When an error occurs during execution
    UNKNOWN = auto()  # For any other unknown timing points

# Type definitions for handler lambdas
ExecutionLambda = Callable[[Dict[str, Any]], Dict[str, Any]]
AsyncExecutionLambda = Callable[[Dict[str, Any]], Dict[str, Any]]
ConditionLambda = Callable[[HandlerTiming, str, Optional[str], Dict[str, Any], Optional[Set[str]]], bool]

# --------------------------------------------------------------

# Protocol for FSM Handlers with self-contained execution conditions
class FSMHandler(Protocol):
    """Protocol defining the interface for self-determining FSM handlers."""

    @property
    def priority(self) -> int:
        """Priority of this handler. Lower values indicate higher priority."""
        ...

    def should_execute(self,
                       timing: HandlerTiming,
                       current_state: str,
                       target_state: Optional[str],
                       context: Dict[str, Any],
                       updated_keys: Optional[Set[str]] = None) -> bool:
        """
        Determine if this handler should execute based on current FSM state.

        Args:
            timing: The hook point being executed
            current_state: Current state of the FSM
            target_state: Target state (if in transition)
            context: Current context data
            updated_keys: Set of keys being updated (for CONTEXT_UPDATE timing)

        Returns:
            True if the handler should execute, False otherwise
        """
        ...

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the handler's logic.

        Args:
            context: Current context data

        Returns:
            Dictionary with updates to add to context
        """
        ...


class HandlerSystemError(Exception):
    """Base exception for handler system errors."""
    pass


class HandlerExecutionError(HandlerSystemError):
    """Exception raised when a handler execution fails."""
    def __init__(self, handler_name: str, original_error: Exception):
        self.handler_name = handler_name
        self.original_error = original_error
        super().__init__(f"Error in handler {handler_name}: {str(original_error)}")


class HandlerSystem:
    """
    Handler system for executing custom functions during FSM execution.
    Handlers determine their own execution conditions.
    """

    def __init__(self, error_mode: str = "continue"):
        """
        Initialize the handler system.

        Args:
            error_mode: How to handle errors in handlers:
                - "continue": Log the error and continue (default)
                - "raise": Raise an exception and stop execution
                - "skip": Skip the handler but continue with others
        """
        self.handlers: List[FSMHandler] = []
        self.error_mode = error_mode

        # Validate error mode
        if error_mode not in ["continue", "raise", "skip"]:
            raise ValueError(f"Invalid error_mode: {error_mode}. Must be 'continue', 'raise', or 'skip'")

    def register_handler(self, handler: FSMHandler) -> None:
        """
        Register a new handler with the system.

        Args:
            handler: The handler to register
        """
        self.handlers.append(handler)
        # Sort handlers by priority after adding new one
        self.handlers.sort(key=lambda h: getattr(h, 'priority', 100))

    def execute_handlers(self,
                             timing: HandlerTiming,
                             current_state: str,
                             target_state: Optional[str],
                             context: Dict[str, Any],
                             updated_keys: Optional[Set[str]] = None) -> Dict[str, Any]:
        """
        Execute all handlers that should run at the specified timing.

        Args:
            timing: Which hook point is being executed
            current_state: Current FSM state
            target_state: Target state if in transition
            context: Current context data
            updated_keys: Set of context keys being updated (for CONTEXT_UPDATE)

        Returns:
            Updated context after all applicable handlers have executed

        Raises:
            HandlerExecutionError: If a handler execution fails and error_mode is "raise"
        """
        updated_context = context.copy()
        executed_handlers = []

        # Pre-filter handlers to avoid unnecessary should_execute calls
        potential_handlers = [h for h in self.handlers
                            if not hasattr(h, 'timings') or
                               not getattr(h, 'timings') or
                               timing in getattr(h, 'timings')]

        # Execute applicable handlers in priority order
        for handler in potential_handlers:
            handler_name = getattr(handler, 'name', handler.__class__.__name__)

            try:
                # Check if this handler should execute
                if handler.should_execute(timing, current_state, target_state, updated_context, updated_keys):
                    # Log handler execution for debugging
                    logger.debug(f"Executing handler {handler_name} at {timing.name}")

                    result = handler.execute(updated_context)

                    # Update context with handler result
                    if result and isinstance(result, dict):
                        updated_context.update(result)

                        # Track keys that were updated by this handler
                        handler_updated_keys = set(result.keys())
                        if updated_keys is not None:
                            updated_keys.update(handler_updated_keys)

                    # Track executed handlers for debugging
                    executed_handlers.append({
                        'name': handler_name,
                        'updated_keys': list(result.keys()) if result and isinstance(result, dict) else []
                    })

                    logger.debug(f"Handler {handler_name} executed")

            except Exception as e:
                error = HandlerExecutionError(handler_name, e)
                logger.error(f"{str(error)}\n{traceback.format_exc()}")

                if self.error_mode == "raise":
                    raise error
                elif self.error_mode == "continue":
                    continue  # Just log and continue to next handler
                elif self.error_mode == "skip":
                    continue  # Skip this handler and continue

        # Add metadata about executed handlers if any were executed
        if executed_handlers:
            if 'system' not in updated_context:
                updated_context['system'] = {}
            if 'handlers' not in updated_context['system']:
                updated_context['system']['handlers'] = {}

            updated_context['system']['handlers'][timing.name] = executed_handlers

        return updated_context


# Base class for creating custom handlers
class BaseHandler:
    """
    Base class for implementing FSM handlers with self-contained execution conditions.
    """

    def __init__(self, name: str = None, priority: int = 100):
        """
        Initialize the base handler.

        Args:
            name: Optional name for the handler (defaults to class name)
            priority: Execution priority (lower values = higher priority)
        """
        self.name = name or self.__class__.__name__
        self._priority = priority

    @property
    def priority(self) -> int:
        """Get the handler's priority."""
        return self._priority

    def should_execute(self,
                     timing: HandlerTiming,
                     current_state: str,
                     target_state: Optional[str],
                     context: Dict[str, Any],
                     updated_keys: Optional[Set[str]] = None) -> bool:
        """
        Determine if this handler should execute.
        Default implementation always returns False - override in subclasses.
        """
        return False

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute handler logic.
        Default implementation does nothing - override in subclasses.
        """
        return {}


class HandlerBuilder:
    """
    Builder for creating FSM handlers using lambdas.

    Provides a fluent interface to define when and how handlers execute.
    """

    def __init__(self, name: str = "LambdaHandler"):
        """
        Initialize the handler builder.

        Args:
            name: Name for the generated handler (used in logs)
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
        Set the handler's execution priority.

        Args:
            priority: Priority value (lower values execute first)

        Returns:
            Self for chaining
        """
        self.priority = priority
        return self

    def when(self, condition: ConditionLambda) -> 'HandlerBuilder':
        """
        Add a custom condition lambda.

        Args:
            condition: Lambda that returns True when handler should execute

        Returns:
            Self for chaining
        """
        self.condition_lambdas.append(condition)
        return self

    def at(self, *timings: HandlerTiming) -> 'HandlerBuilder':
        """
        Execute at specific timing points.

        Args:
            *timings: One or more HandlerTiming values

        Returns:
            Self for chaining
        """
        self.timings.update(timings)
        return self

    def on_state(self, *states: str) -> 'HandlerBuilder':
        """
        Execute when in specific states.

        Args:
            *states: State IDs to match against current_state

        Returns:
            Self for chaining
        """
        self.states.update(states)
        return self

    def not_on_state(self, *states: str) -> 'HandlerBuilder':
        """
        Do not execute when in specific states.

        Args:
            *states: State IDs that should not match current_state

        Returns:
            Self for chaining
        """
        self.not_states.update(states)
        return self

    def on_target_state(self, *states: str) -> 'HandlerBuilder':
        """
        Execute when transitioning to specific states.

        Args:
            *states: State IDs to match against target_state

        Returns:
            Self for chaining
        """
        self.target_states.update(states)
        return self

    def not_on_target_state(self, *states: str) -> 'HandlerBuilder':
        """
        Do not execute when transitioning to specific states.

        Args:
            *states: State IDs that should not match target_state

        Returns:
            Self for chaining
        """
        self.not_target_states.update(states)
        return self

    def when_context_has(self, *keys: str) -> 'HandlerBuilder':
        """
        Execute when context contains specific keys.

        Args:
            *keys: Context keys that must be present

        Returns:
            Self for chaining
        """
        self.required_keys.update(keys)
        return self

    def when_keys_updated(self, *keys: str) -> 'HandlerBuilder':
        """
        Execute when specific context keys are updated.

        Args:
            *keys: Context keys to watch for updates

        Returns:
            Self for chaining
        """
        self.updated_keys.update(keys)
        return self

    def on_state_entry(self, *states: str) -> 'HandlerBuilder':
        """
        Execute when entering specific states (shorthand).

        Args:
            *states: Target states that trigger execution

        Returns:
            Self for chaining
        """
        self.timings.add(HandlerTiming.POST_TRANSITION)
        self.target_states.update(states)
        return self

    def on_state_exit(self, *states: str) -> 'HandlerBuilder':
        """
        Execute when exiting specific states (shorthand).

        Args:
            *states: Current states that trigger execution

        Returns:
            Self for chaining
        """
        self.timings.add(HandlerTiming.PRE_TRANSITION)
        self.states.update(states)
        return self

    def on_context_update(self, *keys: str) -> 'HandlerBuilder':
        """
        Execute when specific context keys are updated (shorthand).

        Args:
            *keys: Context keys to watch for updates

        Returns:
            Self for chaining
        """
        self.timings.add(HandlerTiming.CONTEXT_UPDATE)
        self.updated_keys.update(keys)
        return self

    def do(self, execution: Union[ExecutionLambda, AsyncExecutionLambda]) -> BaseHandler:
        """
        Set the execution lambda and build the handler.

        Args:
            execution: Lambda/function that performs the handler's work

        Returns:
            Configured BaseHandler instance
        """
        self.execution_lambda = execution
        return self.build()

    def build(self) -> BaseHandler:
        """
        Build a handler from the current configuration.

        Returns:
            BaseHandler instance
        """
        if not self.execution_lambda:
            raise ValueError("Execution lambda is required - use .do() to set it")

        # Check if the execution lambda is async
        is_async = inspect.iscoroutinefunction(self.execution_lambda)

        # Create a handler class dynamically
        handler = _LambdaHandler(
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


def create_handler(name: str = "LambdaHandler") -> HandlerBuilder:
    """
    Create a new handler builder.

    Args:
        name: Name for the generated handler

    Returns:
        HandlerBuilder instance
    """
    return HandlerBuilder(name)


class _LambdaHandler(BaseHandler):
    """
    Internal implementation of a handler using lambdas.

    This class is created by the HandlerBuilder and shouldn't be used directly.
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
        """Initialize with all the builder's configuration."""
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
        """
        Determine if this handler should execute based on builder config.

        Evaluates all the conditions set in the builder.
        """
        # Quick rejection tests first for performance

        # If we specified timings, check if the current timing matches
        if self.timings and timing not in self.timings:
            return False

        # If we specified current states, check if current_state matches
        if self.states and current_state not in self.states:
            return False

        # If we specified states to avoid, check if current_state matches any
        if self.not_states and current_state in self.not_states:
            return False

        # If we specified target states, check if target_state matches
        if self.target_states and (not target_state or target_state not in self.target_states):
            return False

        # If we specified target states to avoid, check if target_state matches any
        if self.not_target_states and target_state and target_state in self.not_target_states:
            return False

        # If we specified required context keys, check if they're all present
        if self.required_keys and not all(key in context for key in self.required_keys):
            return False

        # If we specified updated keys, check if any are being updated
        if self.updated_keys and (not updated_keys or not any(key in updated_keys for key in self.updated_keys)):
            return False

        # Evaluate custom condition lambdas if any
        for condition in self.condition_lambdas:
            try:
                if not condition(timing, current_state, target_state, context, updated_keys):
                    return False
            except Exception as e:
                logger.warning(f"Error in condition lambda for {self.name}: {str(e)}")
                return False

        # All conditions passed
        return True

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the handler's logic.

        Handles both synchronous and asynchronous execution lambdas.
        """
        try:
            if self.is_async:
                # Execution lambda is already async
                return self.execution_lambda(context)
            else:
                # Execution lambda is synchronous - run in executor
                loop = asyncio.get_event_loop()
                result = loop.run_in_executor(None, self.execution_lambda, context)
                return result
        except Exception as e:
            logger.error(f"Error in {self.name}: {str(e)}")
            # Re-raise the exception to be handled by HandlerSystem
            raise HandlerExecutionError(self.name, e)

    def __str__(self):
        """String representation for debugging."""
        return f"{self.name} (Lambda Handler)"