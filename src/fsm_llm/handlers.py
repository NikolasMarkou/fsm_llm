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

from __future__ import annotations

import copy
import math
import numbers
import threading
import traceback
from collections.abc import Callable
from enum import Enum
from typing import Any, Protocol

# --------------------------------------------------------------
# Local imports
# --------------------------------------------------------------
from .definitions import FSMError
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
        # Rebound, never mutated in place (copy-on-write, see register_handler).
        self.handlers: list[FSMHandler] = []
        self._registration_lock = threading.Lock()
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
        # DECISION plan-2026-09-21T203800-8a03483a/D-019
        # Copy-on-write under a lock: build a NEW sorted list and rebind
        # `self.handlers` in one assignment. Do NOT go back to
        # `self.handlers.append(); self.handlers.sort()`: CPython empties a list
        # for the whole duration of an in-place sort, so a concurrent reader
        # (`handlers_at`, every turn) saw zero handlers and skipped them (C1).
        # Readers take one reference without the lock; do NOT lock them.
        # C6: reject an unorderable priority HERE, naming this handler. Left to
        # `sorted()`, a string priority registered first was accepted and the
        # TypeError surfaced on the NEXT (innocent) registration; a NaN never
        # raised and silently scrambled the order.
        priority = getattr(handler, "priority", 100)
        if isinstance(priority, bool) or not isinstance(priority, numbers.Real):
            raise TypeError(
                f"Handler {getattr(handler, 'name', handler)!r}: priority must be "
                f"a real number, got {type(priority).__name__}"
            )
        if math.isnan(priority):
            raise ValueError(
                f"Handler {getattr(handler, 'name', handler)!r}: priority is NaN"
            )
        with self._registration_lock:
            self.handlers = sorted(
                [*self.handlers, handler], key=lambda h: getattr(h, "priority", 100)
            )

    def handlers_at(self, timing: HandlerTiming) -> list[FSMHandler]:
        """Return the registered handlers that subscribe to ``timing``.

        Contract: pure and cheap (no ``should_execute`` call, no copying);
        returns a new list in priority order. A handler with no ``timings``
        attribute, or ``timings is None``, subscribes to every timing. An empty
        result means ``execute_handlers`` would do nothing at this timing.
        """
        registered = self.handlers  # one snapshot reference (D-019)
        return [
            h
            for h in registered
            if not hasattr(h, "timings") or h.timings is None or timing in h.timings
        ]

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
        output_context: dict[str, Any] = {}
        candidates = self.handlers_at(timing)
        if not candidates:
            return output_context

        # DECISION plan-2026-09-20T114608-a8e47b88/D-020
        # `copy.deepcopy(context)` used to run unconditionally, before checking
        # whether ANY handler at this timing would actually pass its own
        # `should_execute()` filter for the current state/target/updated_keys.
        # `should_execute()` never mutates `context` -- confirmed by reading
        # both implementations in this module, not assumed: `BaseHandler`'s
        # default always returns `False` without touching `context`, and
        # `LambdaHandler.should_execute` (the only other implementation here)
        # only ever READS `context` -- directly (`required_keys`/`updated_keys`
        # membership checks) and through user-supplied `condition_lambdas`,
        # which this system's documented contract treats as pure predicates,
        # same as every other condition check in that method. It is therefore
        # safe to probe every candidate's `should_execute()` against the
        # UNCOPIED `context` until the FIRST one that will actually run is
        # found, and pay for the deep copy exactly once, only then. Every
        # handler probed before that point still sees the same original
        # `context` a zero-registered-handler timing already skips entirely
        # via the `candidates` guard above (which complements, and does not
        # replace, the caller-side D-022 optimization in
        # `MessagePipeline.execute_handlers`, `pipeline.py`). Do NOT
        # deep-copy per-candidate inside this probe loop -- that would
        # reintroduce the exact cost removed here, just paid earlier and
        # more often.
        updated_context: dict[str, Any] | None = None

        def _report_and_maybe_raise(
            handler_name: str, handler: FSMHandler, exc: Exception
        ) -> None:
            """Log + apply the configured error_mode for a failure at ``handler``.

            Shared by the ``should_execute()`` probe and the actual handler
            execution below so both failure points get identical error-mode
            semantics without duplicating the raise/continue decision twice.
            Raises ``HandlerExecutionError`` (critical or error_mode="raise");
            otherwise returns normally (error_mode="continue").
            """
            # C6: a condition lambda's failure arrives already wrapped by
            # `LambdaHandler.should_execute`; wrapping it again doubled the
            # message and hid the real cause behind `original_error`.
            if isinstance(exc, HandlerExecutionError):
                error = exc
            else:
                error = HandlerExecutionError(handler_name, exc)
            logger.error(f"{error!s}\n{traceback.format_exc()}")
            is_critical = getattr(handler, "critical", False)
            if self.error_mode == "raise" or is_critical:
                error.partial_context = dict(output_context)
                if error is exc:
                    raise error
                raise error from exc

        # Execute applicable handlers in priority order (lower priority numbers first)
        for handler in candidates:
            handler_name = getattr(handler, "name", handler.__class__.__name__)
            probe_context = context if updated_context is None else updated_context

            try:
                # Check if this handler should execute based on current conditions
                should_run = handler.should_execute(
                    timing, current_state, target_state, probe_context, updated_keys
                )
            except Exception as e:
                _report_and_maybe_raise(handler_name, handler, e)
                continue  # error_mode == "continue": move to the next handler

            if not should_run:
                continue

            # DECISION plan-2026-09-20T114608-a8e47b88/D-025
            # copy.deepcopy(context) now runs OUTSIDE the handler-EXECUTION
            # try/except below (D-020 originally placed it inside that try,
            # alongside should_execute()). should_execute() has ALREADY
            # confirmed this handler will run by this point, so D-020's
            # "defer the copy until the first qualifying handler" benefit is
            # unchanged -- this only moves WHERE the copy is attempted, not
            # WHEN. A non-deep-copyable context value (e.g. a live
            # threading.Lock a caller left in context) is a caller bug, not a
            # handler-execution failure: pre-D-020 it raised a bare
            # TypeError straight out of execute_handlers. With the copy
            # inside the try, the default error_mode="continue" silently
            # swallowed it (empty dict, handler never ran, no exception) and
            # error_mode="raise" misattributed it to a handler that was never
            # actually invoked. Do NOT move this back inside the try below --
            # that reintroduces both regressions. See decisions.md D-025.
            if updated_context is None:
                updated_context = copy.deepcopy(context)

            logger.debug(f"Executing handler {handler_name} at {timing.name}")

            try:
                # Execute the handler with optional timeout
                result = self._execute_single_handler(
                    handler, updated_context, handler_name
                )

                # Update context with handler result if valid
                if isinstance(result, dict):
                    updated_context.update(result)
                    output_context.update(result)
                elif result is not None:
                    logger.warning(
                        f"Handler {handler_name} returned a non-dict result "
                        f"({type(result).__name__}); it was ignored"
                    )

                logger.debug(f"Handler {handler_name} completed successfully")

            except Exception as e:
                _report_and_maybe_raise(handler_name, handler, e)
                continue  # error_mode == "continue": move to the next handler

        return output_context

    def _execute_single_handler(
        self, handler: FSMHandler, context: dict[str, Any], handler_name: str
    ) -> dict[str, Any] | None:
        """Execute a single handler, optionally with timeout protection.

        When ``handler_timeout`` is set, the handler runs on a private copy of
        ``context`` in a daemon thread joined with the timeout. On timeout its
        result is discarded and ``TimeoutError`` is raised; the thread keeps
        running until the handler returns, but only ever touches its own copy.
        """
        if self.handler_timeout is None:
            return handler.execute(context)

        # DECISION plan-2026-09-21T203800-8a03483a/D-019
        # A timed handler gets its OWN deep copy and runs in a per-call DAEMON
        # thread. Do NOT hand it the shared `context` (a timed-out straggler
        # kept writing into the dict the next handler was reading, C2), and do
        # NOT go back to a ThreadPoolExecutor: its workers are non-daemon, so a
        # straggler blocked interpreter exit. Deepcopy failure falls back to a
        # shallow copy (top-level writes stay isolated) with a WARNING.
        try:
            private = copy.deepcopy(context)
        except Exception as exc:
            logger.warning(
                f"Handler '{handler_name}': context not deep-copyable ({exc!s}); "
                f"running on a shallow copy"
            )
            private = dict(context)

        outcome: dict[str, Any] = {}

        def _run() -> None:
            try:
                outcome["result"] = handler.execute(private)
            except BaseException as exc:  # re-raised on the caller's thread
                outcome["error"] = exc

        worker = threading.Thread(
            target=_run, name=f"fsm-handler-{handler_name}", daemon=True
        )
        worker.start()
        worker.join(self.handler_timeout)
        if worker.is_alive():
            raise TimeoutError(
                f"Handler '{handler_name}' timed out after {self.handler_timeout}s"
            )
        if "error" in outcome:
            raise outcome["error"]
        return outcome.get("result")

    def close(self) -> None:
        """Release handler-system resources. A safe no-op, kept for API
        compatibility: timed handlers run in per-call daemon threads."""


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
    - Critical handlers via ``.critical()`` (failures raise even in "continue" mode)

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
        # Named ``is_critical`` because ``critical`` is the fluent METHOD below.
        # Default stays False: adding the method must not change what existing
        # builder callers get.
        self.is_critical: bool = False

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

    def critical(self, value: bool = True) -> HandlerBuilder:
        """
        Mark the handler as critical.

        A critical handler's failure is raised as a ``HandlerExecutionError`` even
        when the ``HandlerSystem`` runs in ``error_mode="continue"``, where a
        non-critical failure would only be logged and skipped.

        Composes in any order with the other chain steps, and must be followed by
        ``.do()`` (or ``.build()``) like every other configuration step::

            handler = (create_handler("must_validate")
                       .at(HandlerTiming.PRE_PROCESSING)
                       .critical()
                       .do(lambda ctx: validate(ctx)))

        :param value: True to mark critical, False to clear it again
        :type value: bool
        :return: Self for method chaining
        :rtype: HandlerBuilder
        """
        self.is_critical = value
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
            timings=self.timings.copy() if self.timings is not None else None,
            states=self.states.copy(),
            target_states=self.target_states.copy(),
            required_keys=self.required_keys.copy(),
            updated_keys=self.updated_keys.copy(),
            priority=self.priority,
            not_states=self.not_states.copy(),
            not_target_states=self.not_target_states.copy(),
            critical=self.is_critical,
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


# --------------------------------------------------------------
