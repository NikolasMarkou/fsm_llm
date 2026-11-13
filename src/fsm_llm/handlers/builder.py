"""``HandlerBuilder`` (the fluent handler builder) and ``create_handler``.

Package ``fsm_llm.handlers`` (plan d454fd62, step 77: split out of the old module
``fsm_llm/handlers.py``).
"""

from __future__ import annotations

from fsm_llm.core.definitions import BuildError

from .base import BaseHandler
from .lambda_handler import LambdaHandler
from .protocol import ConditionLambda, ExecutionLambda, HandlerTiming


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
    - Synchronous execution only: an ``async def`` execution lambda builds
      but fails at run time ("non-dict result: coroutine")
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
        when the ``HandlerSystem`` runs in ``handler_error_mode="continue"``, where a
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
        return self.at(HandlerTiming.POST_TRANSITION).on_target_state(*states)

    def on_state_exit(self, *states: str) -> HandlerBuilder:
        """
        Convenient shorthand for executing when exiting specific states.

        Equivalent to calling ``.at(HandlerTiming.PRE_TRANSITION).on_state(*states)``

        :param states: Current states that trigger execution upon exit
        :type states: str
        :return: Self for method chaining
        :rtype: HandlerBuilder
        """
        return self.at(HandlerTiming.PRE_TRANSITION).on_state(*states)

    def on_context_update(self, *keys: str) -> HandlerBuilder:
        """
        Convenient shorthand for executing when specific context keys are updated.

        Equivalent to calling ``.at(HandlerTiming.CONTEXT_UPDATE).when_keys_updated(*keys)``

        :param keys: Context keys to watch for updates
        :type keys: str
        :return: Self for method chaining
        :rtype: HandlerBuilder
        """
        return self.at(HandlerTiming.CONTEXT_UPDATE).when_keys_updated(*keys)

    def do(self, execution: ExecutionLambda) -> BaseHandler:
        """
        Set the execution lambda and build the final handler instance.

        This method completes the builder pattern by providing the actual execution
        logic and returning a configured handler ready for registration.

        :param execution: Lambda or function that performs the handler's work
        :type execution: ExecutionLambda
        :return: Configured BaseHandler instance ready for use
        :rtype: BaseHandler
        :raises BuildError: If *execution* is missing or not callable (the
            check is ``build()``'s; ``do()`` records and then builds)
        """
        self.execution_lambda = execution
        return self.build()

    def build(self) -> BaseHandler:
        """
        Build a handler from the current configuration.

        This method creates the final handler instance based on all the configuration
        set through the fluent interface. It does not inspect the execution
        lambda beyond ``callable``: an ``async def`` one builds and fails when
        it runs (the handler gets a coroutine, not a dict).

        :return: Configured BaseHandler instance
        :rtype: BaseHandler
        :raises BuildError: (a ``ValueError``) If the execution lambda is not
            set or not callable
        """
        if not self.execution_lambda:
            raise BuildError("Execution lambda is required - use .do() to set it")
        if not callable(self.execution_lambda):
            raise BuildError(
                "Execution lambda must be callable, "
                f"got {type(self.execution_lambda).__name__}"
            )

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
