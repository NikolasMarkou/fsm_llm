"""
Structured Reasoning Engine for FSM-LLM
=======================================

Enhanced with loop prevention, context management, and standardized handling.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Generator
from contextlib import closing
from typing import Any

from fsm_llm import (
    API,
    FSMDefinition,
    FSMError,
    RunBudgetExceededError,
    clear_keys_on_entry,
)
from fsm_llm.engine.api.helpers import llm_settings_for
from fsm_llm.engine.executables import ExecutableRegistry
from fsm_llm.engine.fsm.budget import check_turn_limits
from fsm_llm.engine.kinds.call import chosen_target
from fsm_llm.handlers import HandlerTiming
from fsm_llm.llm import interface_model, resolve_model
from fsm_llm.logging import logger

from .choice import _classification_from, _eligible_choices, _strategy_for
from .constants import (
    RETRY_CLEARED_KEYS,
    SOLVE_DRIVER_KEYS,
    ClassifierStates,
    ContextKeys,
    Defaults,
    ErrorMessages,
    HandlerNames,
    LogMessages,
    OrchestratorStates,
    solve_step_budget,
)
from .definitions import ReasoningTrace
from .exceptions import ReasoningClassificationError, ReasoningExecutionError
from .handlers import (
    ContextManager,
    OutputFormatter,
    ReasoningHandlers,
    register_strategy_handlers,
    retries_exhausted,
)
from .selection import classification_intents
from .stream import SolveStream
from .utilities import get_reasoning_fsm

#: What ``ReasoningEngine.model`` shows when an injected interface states no model.
_INJECTED_MODEL_LABEL = "injected-llm-interface"


class ReasoningEngine:
    """
    Main reasoning engine with enhanced error handling and flow control.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        registry: ExecutableRegistry | None = None,
        **kwargs,
    ):
        """
        Initialize the reasoning engine.

        :param model: The LLM model to use; ``None`` reads env ``LLM_MODEL``
        :param registry: The ``ExecutableRegistry`` the solve chooses its strategy from: every
            eligible strategy in it (``selection.strategy_choices``) is a target of the
            orchestrator's call and an intent of the classifier (plan baa37f40 D-033); it must
            hold a strategy named ``analytical`` and at least one other. ``None`` is the nine
            built-ins. Read once, here: a later change of the registry does not reach this engine
        :param kwargs: Additional arguments for the API

        Raises ``ModelNotConfiguredError`` when no model is named (argument or
        ``LLM_MODEL``) and no ``llm_interface`` is injected: there is no
        built-in model. An injected interface owns its model, so none is
        needed then.

        Thread safety: ``solve_problem()`` is serialized via an internal lock
        so that only one solve runs at a time on a given engine instance.

        Raises ``TypeError`` for a ``registry`` that is not an ``ExecutableRegistry`` and for a
        ``targets=`` keyword (the targets of the orchestrator are the registry), and
        ``ReasoningRegistryError`` for a registry that cannot serve a solve.
        """
        # `targets=` would be swallowed by `**kwargs` into BOTH APIs below: refuse it by name
        if "targets" in kwargs:
            raise TypeError(
                "ReasoningEngine takes registry=, not targets=: the targets of the "
                "orchestrator are the strategies of its registry"
            )
        if registry is not None and not isinstance(registry, ExecutableRegistry):
            raise TypeError(
                f"registry must be an ExecutableRegistry, got {type(registry).__name__}"
            )
        self._solve_lock = threading.Lock()
        self.api_kwargs = kwargs.copy()
        # DECISION plan-2026-11-12T073150-d454fd62/D-005
        # The model is resolved through llm.resolve_model only, which raises
        # ModelNotConfiguredError when nothing is set. Do NOT give the
        # parameter a built-in default model again, and do NOT read the
        # environment here by hand. An injected interface needs no model.
        injected = kwargs.get("llm_interface")
        if injected is not None:
            self.model = interface_model(injected) or model or _INJECTED_MODEL_LABEL
        else:
            self.model = resolve_model(model)
        model = self.model

        # The strategies this engine may choose, read once (plan baa37f40 D-033)
        if registry is None:
            # lazy: `executables` imports this module
            from .executables import strategy_registry

            registry = strategy_registry()
        self.registry: ExecutableRegistry = registry
        self._choices, self._analytical = _eligible_choices(registry)
        self._names = {choice.target_id: choice.name for choice in self._choices}
        self._solution_keys = (
            ContextKeys.FINAL_SOLUTION,
            ContextKeys.PROPOSED_SOLUTION,
            *(choice.solution_key for choice in self._choices),
        )

        # Load FSM definitions
        self._load_fsm_definitions()

        # Initialize components
        self.handlers = ReasoningHandlers()
        self.context_manager = ContextManager()
        self.output_formatter = OutputFormatter()

        # Initialize APIs
        self._initialize_apis()

        logger.info(LogMessages.ENGINE_INITIALIZED.format(model=model))

    def _load_fsm_definitions(self):
        """Load every FSM definition (all are in the package, so a miss raises)."""
        self.main_fsm = get_reasoning_fsm("orchestrator")
        self.classifier_fsm = get_reasoning_fsm("classifier")
        # The closed list of the call is every eligible strategy of the registry, the fallback
        # the one named `analytical` (plan baa37f40 D-033): with the built-in registry this
        # writes the values the file holds, so no byte of the shipped definition moves.
        choose = self.main_fsm["states"][OrchestratorStates.EXECUTE_REASONING]["call"][
            "choose"
        ]
        choose["targets"] = [choice.target_id for choice in self._choices]
        choose["fallback"] = self._analytical.target_id
        # DECISION plan-2026-11-23T122249-baa37f40/D-034
        # The classifier's closed intents are derived from the SAME list as `choose.targets`
        # just above (`classification_intents(self._choices)`, at most 15, `analytical` kept):
        # with the built-in registry they equal the file, so no byte moves. Do NOT build the
        # intents from another list or filter them on this path (the model could pick a
        # strategy the call refuses, or miss one it lists), and do NOT type the strategy names
        # into `extraction_instructions` or `purpose` again.
        (entry,) = self.classifier_fsm["states"][ClassifierStates.RECOMMEND_STRATEGY][
            "classification_extractions"
        ]
        entry["intents"] = classification_intents(self._choices)
        entry["fallback_intent"] = self._analytical.name
        call = (
            FSMDefinition(**self.main_fsm)
            .states[OrchestratorStates.EXECUTE_REASONING]
            .call
        )
        if call is None:  # a packaging bug: the file of the orchestrator changed shape
            raise ReasoningExecutionError(
                f"state {OrchestratorStates.EXECUTE_REASONING!r} of the orchestrator "
                "file is not a call state"
            )
        if call.max_steps is None:  # a packaging bug too: the budget lives in the file
            raise ReasoningExecutionError(
                f"state {OrchestratorStates.EXECUTE_REASONING!r} of the orchestrator "
                "file declares no call max_steps"
            )
        self._call_config = call
        # the default step budget of a solve, derived from the file (plan baa37f40 D-032)
        self._solve_steps = solve_step_budget(call.max_steps)

    def _initialize_apis(self):
        """Initialize APIs with all necessary handlers."""
        # An injected `llm_interface` owns its model: core refuses another
        # one beside it, so `model` goes only to an interface core builds.
        settings = llm_settings_for(self.api_kwargs, model=self.model)
        # Main orchestrator API: the call state of `execute_reasoning` pushes a strategy FSM as
        # a frame of this API (plan baa37f40 D-031) and resolves its target in the registry
        self.orchestrator = API.from_definition(
            self.main_fsm, **settings, targets=self.registry, **self.api_kwargs
        )

        # Classification API
        self.classifier = API.from_definition(
            self.classifier_fsm, **settings, **self.api_kwargs
        )

        # Register handlers
        self._register_handlers()

    def _register_handlers(self):
        """Register all handlers with proper configuration."""
        # DECISION plan-2026-10-01T093600-944e2692/D-058: the classifier,
        # strategy selector, validator, retry limiter and hybrid loop counter
        # are critical: their failure stops the solve (core raises it through
        # the step, solve_problem wraps it as ReasoningExecutionError,
        # chained). Do NOT drop `.critical()`: core's default
        # handler_error_mode="continue" would log and skip the failure, so a
        # spent classifier budget would fall back to the model's strategy, a
        # missing strategy FSM would return a solve that ran no strategy, and
        # a validator crash would stall validate_refine until the step budget.
        # Do NOT set handler_error_mode="raise" on the APIs instead: the
        # pruner and tracer are best-effort and must not stop a solve.
        # Problem classifier
        self.orchestrator.register_handler(
            self.orchestrator.create_handler(HandlerNames.ORCHESTRATOR_CLASSIFIER)
            .at(HandlerTiming.CONTEXT_UPDATE)
            .when_keys_updated(ContextKeys.PROBLEM_TYPE)
            .critical()
            .do(self._classify_problem)
        )

        # DECISION plan-2026-10-01T093600-944e2692/D-053: clear the rejected
        # solution on EVERY entry to execute_reasoning so a retry re-runs
        # synthesis and validation (core extracts only unset keys). Do NOT
        # move the clear to synthesize_solution entry: that wipes the
        # proposed_solution the calculator strategy merged back on pop.
        self.orchestrator.register_handler(
            clear_keys_on_entry(
                RETRY_CLEARED_KEYS,
                state=OrchestratorStates.EXECUTE_REASONING,
                name=HandlerNames.RETRY_KEY_CLEARER,
            )
        )

        # Strategy selector (the call state itself pushes and pops the strategy FSM)
        self.orchestrator.register_handler(
            self.orchestrator.create_handler(HandlerNames.ORCHESTRATOR_SELECTOR)
            .on_state_entry(OrchestratorStates.EXECUTE_REASONING)
            .critical()
            .do(self._select_reasoning_type)
        )

        # Solution validator with retry logic
        self.orchestrator.register_handler(
            self.orchestrator.create_handler(HandlerNames.ORCHESTRATOR_VALIDATOR)
            .at(HandlerTiming.CONTEXT_UPDATE)
            .when_keys_updated(ContextKeys.PROPOSED_SOLUTION)
            .critical()
            .do(self.handlers.validate_when_insights_ready)
        )

        # Context pruner
        self.orchestrator.register_handler(
            self.orchestrator.create_handler(HandlerNames.CONTEXT_PRUNER)
            .at(HandlerTiming.PRE_TRANSITION)
            .do(self.handlers.prune_context)
        )

        # Reasoning tracer (the classifier's own trace is never read)
        self.orchestrator.register_handler(
            self.orchestrator.create_handler(HandlerNames.REASONING_TRACER)
            .at(HandlerTiming.POST_TRANSITION)
            .do(self.handlers.update_reasoning_trace)
        )

        # Retry limiter for validation loops
        self.orchestrator.register_handler(
            self.orchestrator.create_handler(HandlerNames.RETRY_LIMITER)
            .on_state_entry(OrchestratorStates.VALIDATE_REFINE)
            .critical()
            .do(self._check_retry_limit)
        )

        register_strategy_handlers(self.orchestrator)

    def _classify_problem(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Classify problem using the classification FSM.

        :param context: Current context
        :return: Classification results
        :raises ReasoningClassificationError: If classification fails
        """
        # Skip if already classified
        if context.get(ContextKeys.CLASSIFIED_PROBLEM_TYPE):
            return {}

        # Extract relevant context for classification
        classification_context = self.context_manager.extract_relevant_context(
            context,
            [
                ContextKeys.PROBLEM_STATEMENT,
                ContextKeys.PROBLEM_TYPE,
                ContextKeys.PROBLEM_COMPONENTS,
            ],
        )

        logger.info(
            LogMessages.CLASSIFICATION_STARTED.format(
                context=list(classification_context.keys())
            )
        )

        conversation_id = None
        try:
            # Run classification
            conversation_id, _ = self.classifier.start_conversation(
                initial_context=classification_context
            )

            # Message-free steps: the problem reaches the model through the
            # conversation context, never through a synthetic user message.
            self.classifier.run_until_terminal(
                conversation_id, max_steps=Defaults.MAX_CLASSIFICATION_STEPS
            )

            # Get results
            result = self.classifier.get_data(conversation_id)
            self.classifier.end_conversation(conversation_id)
            conversation_id = None  # Mark as cleaned up
        except RunBudgetExceededError as e:
            raise ReasoningClassificationError(
                "Classification did not converge after "
                f"{Defaults.MAX_CLASSIFICATION_STEPS} steps",
                details={"context_keys": list(classification_context.keys())},
            ) from e
        except Exception as e:
            raise ReasoningClassificationError(
                f"Problem classification failed: {e}",
                details={"context_keys": list(classification_context.keys())},
            ) from e
        finally:
            # Clean up classifier conversation on any error path
            if conversation_id is not None:
                _end_quietly(self.classifier, conversation_id)

        classification = _classification_from(result)

        logger.info(
            LogMessages.CLASSIFICATION_COMPLETE.format(
                type=classification.recommended_type
            )
        )

        # DECISION plan-2026-11-23T122249-baa37f40/D-031
        # The strategy is chosen HERE and only here, as the target id the call state
        # of `execute_reasoning` reads from the handler-only `classified_problem_type`
        # (the classified strategy, then `preferred_reasoning_type` when it names a
        # known one). Do NOT let a model reply choose it (`reasoning_strategy` is a
        # model extraction and is never read), do NOT make the key public to match the
        # roadmap's wording (a bulk reply could then pick the strategy, D-054), and do
        # NOT write the bare name: the call's closed list holds target ids.
        strategy = _strategy_for(
            classification.recommended_type,
            context.get(ContextKeys.PREFERRED_REASONING_TYPE),
            self._choices,
            self._analytical,
        )
        return {
            ContextKeys.CLASSIFIED_PROBLEM_TYPE: strategy.target_id,
            ContextKeys.CLASSIFICATION_JUSTIFICATION: classification.justification,
            ContextKeys.PROBLEM_DOMAIN: classification.domain,
            ContextKeys.ALTERNATIVE_APPROACHES: classification.alternatives,
        }

    # DECISION plan-2026-11-23T122249-baa37f40/D-031
    # The `execute_reasoning` call state decides on ARRIVAL, after the state-entry
    # handlers, so this handler runs first: it names the strategy for the trace and the
    # validator (`reasoning_type_selected`, by the SAME rule the call executor uses,
    # `chosen_target`, never a second one) and defaults `constraints` to `[]`, because
    # the executor refuses a declared input that is absent and nothing in the
    # orchestrator extracts `constraints`. Do NOT drop the default (every solve with no
    # caller constraints would end `failed/inputs_missing` and skip its strategy), do
    # NOT take the name from `reasoning_strategy` (a model value), and do NOT overwrite
    # a caller's `constraints`. D-043 (review iter1, concerns 1 and 2): when the classifier
    # never ran (a `problem_type` in the starting context is never an UPDATE) the choice is
    # written here, the preferred strategy else the fallback. Do NOT drop that (a caller's
    # `--type` would be ignored, as at HEAD before D-043) and do NOT take
    # `preferred_reasoning_type` out of the orchestrator's `handler_only_keys` (a bulk reply
    # could then pick the strategy over the classifier).
    def _select_reasoning_type(self, context: dict[str, Any]) -> dict[str, Any]:
        """On entry to ``execute_reasoning``: name the strategy the call will run.

        :param context: Current context
        :return: ``reasoning_type_selected`` (the strategy name of the target
            ``chosen_target`` picks), plus ``constraints = []`` when absent and
            ``classified_problem_type`` when the classifier did not run
        """
        delta: dict[str, Any] = {}
        if not context.get(ContextKeys.CLASSIFIED_PROBLEM_TYPE):
            chosen = _strategy_for(
                self._analytical.name,
                context.get(ContextKeys.PREFERRED_REASONING_TYPE),
                self._choices,
                self._analytical,
            )
            delta[ContextKeys.CLASSIFIED_PROBLEM_TYPE] = chosen.target_id
        target = chosen_target(self._call_config, {**context, **delta})
        selected = self._names.get(str(target), str(target))
        logger.info(LogMessages.STRATEGY_EXECUTING.format(type=selected))
        delta[ContextKeys.REASONING_TYPE_SELECTED] = selected
        if context.get(ContextKeys.CONSTRAINTS) is None:
            delta[ContextKeys.CONSTRAINTS] = []
        return delta

    def _check_retry_limit(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Give this attempt a verdict if it has none, then check the retry limit.

        :param context: Current context
        :return: Retry status, plus the verdict keys of
            ``ReasoningHandlers.validate_solution`` when no verdict was set
        """
        # DECISION plan-2026-10-01T093600-944e2692/D-054: the verdict is
        # handler-only, so an attempt that produced no proposed_solution (the
        # validator runs on its CONTEXT_UPDATE only) gets it here. Do NOT hand
        # the verdict back to the model: its bulk reply must not open the
        # gate, and with no verdict validate_refine would stay until the step
        # budget.
        verdict: dict[str, Any] = {}
        if context.get(ContextKeys.VALIDATION_RESULT) is None:
            verdict = self.handlers.validate_solution(context)
        retry_count = verdict.get(
            ContextKeys.RETRY_COUNT, context.get(ContextKeys.RETRY_COUNT, 0)
        )
        max_reached = retries_exhausted(retry_count)

        if max_reached:
            logger.warning(ErrorMessages.MAX_RETRIES_EXCEEDED)

        return {**verdict, ContextKeys.MAX_RETRIES_REACHED: max_reached}

    def close(self) -> None:
        """End every open conversation of both APIs; safe to call twice."""
        for api in (self.orchestrator, self.classifier):
            api.close()

    def __enter__(self) -> ReasoningEngine:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def solve_problem(
        self,
        problem: str,
        initial_context: dict[str, Any] | None = None,
        *,
        max_steps: int | None = None,
        max_seconds: float | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """
        Solve a problem using structured reasoning.

        This method is serialized: only one ``solve_problem`` call executes
        at a time per engine instance.

        :param problem: The problem statement
        :param initial_context: Optional initial context
        :param max_steps: Ceiling on the steps of the one orchestrator run (strategy
            frames included); ``None`` derives it from the orchestrator file
            (``solve_step_budget(call.max_steps)``, 170 for the shipped file)
        :param max_seconds: Wall-clock ceiling of that run; ``None`` is no limit
        :return: Tuple of (solution, trace_info)
        :raises ValueError: If a limit is not a valid run limit (before any model call)
        :raises ReasoningExecutionError: If reasoning execution fails, a spent budget
            included (chained from ``RunBudgetExceededError``)
        """
        check_turn_limits(max_steps, max_seconds)
        solve = self._solve(
            problem, initial_context, max_steps, max_seconds, stream=False
        )
        try:
            next(solve)
        except StopIteration as done:
            result: tuple[str, dict[str, Any]] = done.value
            return result
        raise AssertionError("a solve that streams nothing yielded")  # pragma: no cover

    def solve_problem_stream(
        self,
        problem: str,
        initial_context: dict[str, Any] | None = None,
        *,
        max_steps: int | None = None,
        max_seconds: float | None = None,
    ) -> SolveStream:
        """
        Solve a problem and stream its replies as they arrive (plan baa37f40 D-036).

        The same solve as ``solve_problem`` (one body), driven by ``run_until_terminal_stream``:
        iterating the returned ``SolveStream`` yields the start greeting and every chunk of
        every step, the replies of a pushed strategy FSM included, while the strategy runs;
        ``.result()`` is the ``(solution, trace_info)`` tuple once the iterator ended. The limits
        are checked now (``ValueError``); nothing else runs, and the solve lock of the engine is
        not taken, before the first ``next()``. The lock is held until the stream ends, fails or
        is closed (``close()``, a ``with`` block, or dropping the stream), and the orchestrator
        conversation is ended on each of those ways out.

        :param problem: The problem statement
        :param initial_context: Optional initial context
        :param max_steps: As in ``solve_problem``
        :param max_seconds: As in ``solve_problem``
        :return: The ``SolveStream``; iterating it raises what ``solve_problem`` raises
        :raises ValueError: If a limit is not a valid run limit (before any model call)
        """
        check_turn_limits(max_steps, max_seconds)
        return SolveStream(
            self._solve(problem, initial_context, max_steps, max_seconds, stream=True)
        )

    # DECISION plan-2026-11-23T122249-baa37f40/D-036
    # ONE body serves `solve_problem` (`stream=False`: never yields, `run_until_terminal`, the
    # request bytes of the parent) and `solve_problem_stream` (`stream=True`: yields the chunks of
    # `run_until_terminal_stream`). The lock is taken HERE, inside the generator, in a `with`: an
    # unstarted stream holds nothing, and every way a generator ends (exhaustion, an error,
    # `close()`, finalisation) releases it. Do NOT take or release the lock outside this `with`
    # (a hand-written release that is not in a `finally` leaks it on an abandoned stream), do NOT
    # end the conversation before the run stream is closed (it stays claimed until then, so
    # `_run_streaming` closes it first), do NOT narrow the last `except` back to the two
    # interrupts (a `GeneratorExit` from `close()` and a `CancelledError` from a model call must
    # end the conversation too), and do NOT give `solve_problem` another driver than
    # `run_until_terminal`.
    def _solve(
        self,
        problem: str,
        initial_context: dict[str, Any] | None,
        max_steps: int | None,
        max_seconds: float | None,
        *,
        stream: bool,
    ) -> Generator[str, None, tuple[str, dict[str, Any]]]:
        """The one solve body; returns ``(solution, trace_info)`` and yields chunks only when
        ``stream`` is true. Holds ``_solve_lock`` from its first step to its end."""
        steps = self._solve_steps if max_steps is None else max_steps
        with self._solve_lock:
            # Initialize context (copy to avoid mutating caller's dict)
            context = dict(initial_context) if initial_context else {}
            # The strategy keys are the engine's own (D-058, D-060).
            dropped = [
                key for key in SOLVE_DRIVER_KEYS if context.pop(key, None) is not None
            ]
            if dropped:
                logger.warning(
                    f"initial_context keys set by the engine only, ignored: {dropped}"
                )
            context[ContextKeys.PROBLEM_STATEMENT] = problem
            context[ContextKeys.REASONING_TRACE] = []
            context[ContextKeys.RETRY_COUNT] = 0

            started = time.monotonic()
            # Start orchestrator
            conversation_id, initial_response = self.orchestrator.start_conversation(
                context
            )
            log = _solve_log(conversation_id)
            log.info(f"Started reasoning process: {conversation_id}")

            try:
                if stream:
                    responses = yield from self._run_streaming(
                        conversation_id, initial_response, steps, max_seconds
                    )
                else:
                    # One message-free run drives the orchestrator, every strategy FSM its
                    # `execute_reasoning` call pushes (and pops) and every retry.
                    results = self.orchestrator.run_until_terminal(
                        conversation_id, max_steps=steps, max_seconds=max_seconds
                    )
                    # a pushed child's greeting rides on the reply of the step that pushed it
                    responses = [initial_response] + [
                        result.response
                        for result in results
                        if result.response is not None
                    ]
            except RunBudgetExceededError as e:
                # DECISION plan-2026-10-01T093600-944e2692/D-054 (D-014): an
                # unfinished solve raises, never returns a fallback string as the
                # solution.
                raise self._failed_solve(
                    conversation_id,
                    f"Reasoning did not finish within {e.limit} {e.budget}",
                ) from e
            except Exception as e:
                raise self._failed_solve(
                    conversation_id, f"Reasoning execution failed: {e}"
                ) from e
            except BaseException:
                # An interrupt, a cancel and the close of a stream reach the caller
                # unwrapped, but end what the solve opened.
                _end_quietly(self.orchestrator, conversation_id)
                raise

            # Get final context BEFORE ending the conversation (end_conversation
            # removes the instance, making get_data fail).
            try:
                final_context = self.orchestrator.get_data(conversation_id)
                solution = self.output_formatter.extract_final_solution(
                    final_context, self._solution_keys
                )

                # Build trace info
                trace_steps = final_context.get(ContextKeys.REASONING_TRACE, [])
                reasoning_types = self._extract_reasoning_types(
                    final_context, trace_steps
                )

                trace_info = ReasoningTrace(
                    steps=trace_steps,
                    reasoning_types_used=set(reasoning_types),
                    final_confidence=final_context.get(
                        ContextKeys.SOLUTION_CONFIDENCE, 0.0
                    ),
                    execution_time_seconds=time.monotonic() - started,
                )

                log.info(
                    LogMessages.PROBLEM_SOLVED.format(steps=trace_info.total_steps)
                )

                trace_dump = trace_info.model_dump()
                return solution, {
                    "reasoning_trace": trace_dump,
                    "summary": self.output_formatter.format_reasoning_summary(
                        trace_dump
                    ),
                    "final_context": final_context,
                    "all_responses": responses,
                }
            finally:
                # Always clean up the conversation, even if post-processing raises
                _end_quietly(self.orchestrator, conversation_id)

    def _run_streaming(
        self,
        conversation_id: str,
        initial_response: str,
        max_steps: int,
        max_seconds: float | None,
    ) -> Generator[str, None, list[str]]:
        """Drive the run with ``run_until_terminal_stream``, yielding every chunk.

        Contract: yields the start greeting when it is not empty, then each chunk of each step;
        returns the ``all_responses`` list of the solve: the greeting, then one text per step that
        spoke, the concatenation of that step's chunks (the steps are told apart by the
        ``before_step`` hook, one bucket per step number, so the reply and the greeting a call
        pushed at the end of the step, which the stream yields as ``"\n<greeting>"``, concatenate
        to the string ``advance`` returns). The run stream is closed before this generator ends,
        however it ends, so the conversation is free to be ended.
        """
        steps: dict[int, list[str]] = {}
        current: list[str] = []

        def begin_step(step: int) -> None:
            nonlocal current
            current = steps.setdefault(step, [])

        if initial_response:
            yield initial_response
        with closing(
            self.orchestrator.run_until_terminal_stream(
                conversation_id,
                max_steps=max_steps,
                max_seconds=max_seconds,
                before_step=begin_step,
            )
        ) as chunks:
            for chunk in chunks:
                current.append(chunk)
                yield chunk
        return [initial_response] + [
            "".join(texts) for texts in steps.values() if texts
        ]

    def _failed_solve(
        self, conversation_id: str, message: str
    ) -> ReasoningExecutionError:
        """End a solve whose run raised; return the error to raise.

        Contract: reads the partial context, ends the conversation (a failure
        to end is logged), and returns ``ReasoningExecutionError(message)``
        with ``details={conversation_id, responses_so_far, partial_context}``
        for the caller to raise ``from`` the run's error; ``partial_context``
        is ``None`` when core cannot read it (``_read_partial_context``).
        ``responses_so_far`` is the greeting plus the replies in the history of
        the frame on top (best effort: the run's results are lost with the error).
        """
        # DECISION plan-2026-10-01T093600-944e2692/D-058: every failed solve
        # (spent budget or any error, e.g. a prompt over core's cap on the
        # final step) carries what it had. Do NOT end the conversation before
        # reading it: the live read is the documented one, while after
        # end_conversation core answers get_data only from its bounded
        # ended-conversation cache, which may already have evicted it.
        partial_context = self._read_partial_context(conversation_id)
        replies = self._replies_so_far(conversation_id)
        _end_quietly(self.orchestrator, conversation_id)
        return ReasoningExecutionError(
            message,
            details={
                "conversation_id": conversation_id,
                "responses_so_far": 1 + replies,
                "partial_context": partial_context,
            },
        )

    def _replies_so_far(self, conversation_id: str) -> int:
        """The exchanges in the history of the frame on top, ``0`` when core cannot read it."""
        try:
            return len(self.orchestrator.get_conversation_history(conversation_id))
        except (FSMError, ValueError):
            return 0

    def _read_partial_context(self, conversation_id: str) -> dict[str, Any] | None:
        """The context of the frame on top of a solve that did not finish.

        Must run before the conversation is ended. Returns ``None`` (with a
        WARNING) when core cannot read it, so the caller's error still
        reports the unfinished solve and the conversation is still ended.
        """
        try:
            data: dict[str, Any] = self.orchestrator.get_data(conversation_id)
            return data
        except (FSMError, ValueError) as read_err:
            # get_data raises FSMError, or ValueError for a conversation
            # whose stack is gone (core's `api.handle_conversation_errors`).
            _solve_log(conversation_id).warning(
                f"Could not read the partial context of {conversation_id}: {read_err}"
            )
            return None

    def _extract_reasoning_types(
        self, final_context: dict[str, Any], trace_steps: list[dict[str, Any]]
    ) -> list[str]:
        """Extract unique reasoning types used."""
        types = set()

        # From final context
        if ContextKeys.REASONING_TYPE_SELECTED in final_context:
            types.add(final_context[ContextKeys.REASONING_TYPE_SELECTED])

        # From trace steps
        for step in trace_steps:
            snapshot = step.get("context_snapshot", {})
            if ContextKeys.REASONING_TYPE_SELECTED in snapshot:
                types.add(snapshot[ContextKeys.REASONING_TYPE_SELECTED])

        return list(types) if types else ["unknown"]


def _solve_log(conversation_id: str) -> Any:
    """The logger bound to one solve's orchestrator conversation."""
    return logger.bind(conversation_id=conversation_id, package="fsm_llm.reasoning")


def _end_quietly(api: API, conversation_id: str) -> None:
    """End ``conversation_id`` on ``api``; a failure is logged, not raised (it runs on
    paths that already return or raise a result)."""
    try:
        api.end_conversation(conversation_id)
    except Exception as cleanup_err:
        _solve_log(conversation_id).warning(
            f"Failed to clean up conversation {conversation_id}: {cleanup_err}"
        )
