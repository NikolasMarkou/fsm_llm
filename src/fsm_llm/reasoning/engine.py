"""
Structured Reasoning Engine for FSM-LLM
=======================================

Enhanced with loop prevention, context management, and standardized handling.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from fsm_llm import (
    API,
    FSMDefinition,
    FSMError,
    RunBudgetExceededError,
    clear_keys_on_entry,
)
from fsm_llm.engine.api.helpers import llm_settings_for
from fsm_llm.engine.fsm.budget import check_turn_limits
from fsm_llm.engine.kinds.call import chosen_target
from fsm_llm.handlers import HandlerTiming
from fsm_llm.llm import interface_model, resolve_model
from fsm_llm.logging import logger

from .constants import (
    RETRY_CLEARED_KEYS,
    SOLVE_DRIVER_KEYS,
    ContextKeys,
    Defaults,
    ErrorMessages,
    HandlerNames,
    LogMessages,
    OrchestratorStates,
    ReasoningType,
    solve_step_budget,
)
from .definitions import ReasoningClassificationResult, ReasoningTrace
from .exceptions import ReasoningClassificationError, ReasoningExecutionError
from .handlers import (
    ContextManager,
    OutputFormatter,
    ReasoningHandlers,
    register_strategy_handlers,
    retries_exhausted,
)
from .strategy_files import strategy_id, strategy_name
from .utilities import get_reasoning_fsm, lookup_reasoning_type, map_reasoning_type

#: What ``ReasoningEngine.model`` shows when an injected interface states no model.
_INJECTED_MODEL_LABEL = "injected-llm-interface"


class ReasoningEngine:
    """
    Main reasoning engine with enhanced error handling and flow control.
    """

    def __init__(self, model: str | None = None, **kwargs):
        """
        Initialize the reasoning engine.

        :param model: The LLM model to use; ``None`` reads env ``LLM_MODEL``
        :param kwargs: Additional arguments for the API

        Raises ``ModelNotConfiguredError`` when no model is named (argument or
        ``LLM_MODEL``) and no ``llm_interface`` is injected: there is no
        built-in model. An injected interface owns its model, so none is
        needed then.

        Thread safety: ``solve_problem()`` is serialized via an internal lock
        so that only one solve runs at a time on a given engine instance.
        """
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
        # The call state of the orchestrator pushes a strategy FSM as a frame of this API
        # (plan baa37f40 D-031): the nine built-in strategies are its targets. Imported
        # here because ``executables`` imports this module.
        from .executables import strategy_registry

        # Main orchestrator API
        self.orchestrator = API.from_definition(
            self.main_fsm, **settings, targets=strategy_registry(), **self.api_kwargs
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
        )
        return {
            ContextKeys.CLASSIFIED_PROBLEM_TYPE: strategy_id(strategy),
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
    # a caller's `constraints`.
    def _select_reasoning_type(self, context: dict[str, Any]) -> dict[str, Any]:
        """On entry to ``execute_reasoning``: name the strategy the call will run.

        :param context: Current context
        :return: ``reasoning_type_selected`` (the strategy name of the target
            ``chosen_target`` picks), plus ``constraints = []`` when absent
        """
        target = chosen_target(self._call_config, context)
        selected = strategy_name(target) or str(target)
        logger.info(LogMessages.STRATEGY_EXECUTING.format(type=selected))
        delta: dict[str, Any] = {ContextKeys.REASONING_TYPE_SELECTED: selected}
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
        with self._solve_lock:
            return self._solve_problem_locked(
                problem,
                initial_context,
                self._solve_steps if max_steps is None else max_steps,
                max_seconds,
            )

    def _solve_problem_locked(
        self,
        problem: str,
        initial_context: dict[str, Any] | None,
        max_steps: int,
        max_seconds: float | None,
    ) -> tuple[str, dict[str, Any]]:
        """Internal solve logic, must be called while holding ``_solve_lock``."""
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
            # One message-free run drives the orchestrator, every strategy FSM its
            # `execute_reasoning` call pushes (and pops) and every retry.
            results = self.orchestrator.run_until_terminal(
                conversation_id, max_steps=max_steps, max_seconds=max_seconds
            )
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
        except (KeyboardInterrupt, SystemExit):
            # The caller's signal reaches it unwrapped, but ends what it opened.
            _end_quietly(self.orchestrator, conversation_id)
            raise
        # a pushed child's greeting rides on the reply of the step that pushed it
        responses = [initial_response] + [
            result.response for result in results if result.response is not None
        ]

        # Get final context BEFORE ending the conversation (end_conversation
        # removes the instance, making get_data fail).
        try:
            final_context = self.orchestrator.get_data(conversation_id)
            solution = self.output_formatter.extract_final_solution(final_context)

            # Build trace info
            trace_steps = final_context.get(ContextKeys.REASONING_TRACE, [])
            reasoning_types = self._extract_reasoning_types(final_context, trace_steps)

            trace_info = ReasoningTrace(
                steps=trace_steps,
                reasoning_types_used=set(reasoning_types),
                final_confidence=final_context.get(
                    ContextKeys.SOLUTION_CONFIDENCE, 0.0
                ),
                execution_time_seconds=time.monotonic() - started,
            )

            log.info(LogMessages.PROBLEM_SOLVED.format(steps=trace_info.total_steps))

            trace_dump = trace_info.model_dump()
            return solution, {
                "reasoning_trace": trace_dump,
                "summary": self.output_formatter.format_reasoning_summary(trace_dump),
                "final_context": final_context,
                "all_responses": responses,
            }
        finally:
            # Always clean up the conversation, even if post-processing raises
            _end_quietly(self.orchestrator, conversation_id)

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


def _strategy_for(recommended: str, preferred: object) -> str:
    """The strategy name a solve runs: the classified one, then the preferred one.

    Contract: ``recommended`` is the classifier's (normalised) recommendation, mapped by
    ``map_reasoning_type`` (an unknown spelling is ``analytical``, with a WARNING).
    ``preferred`` is ``initial_context["preferred_reasoning_type"]``: when it names a known
    strategy (a value or an alias) it WINS over the classified one; when it names none, ONE
    WARNING and it is ignored. Never raises for any value.
    """
    chosen = map_reasoning_type(recommended)
    if preferred:
        named = lookup_reasoning_type(str(preferred))
        if named is None:
            logger.warning(
                f"preferred_reasoning_type {preferred!r} names no known strategy; ignoring"
            )
        else:
            logger.debug(f"Using preferred_reasoning_type from context: {named!r}")
            chosen = named
    return chosen


def _classification_from(result: dict[str, Any]) -> ReasoningClassificationResult:
    """The classification of a finished classifier conversation's ``result``.

    Contract: never raises for any JSON value the model wrote. A
    ``recommended_reasoning_type`` that is missing (a classification below
    its confidence threshold, a cut reply or a provider outage writes
    nothing), not a ``str`` or blank becomes ``"analytical"`` (an unknown
    name is mapped to analytical later, by ``map_reasoning_type``); a
    non-``str`` justification or domain becomes ``""``;
    ``alternative_approaches`` keeps its non-blank ``str`` items (a bare
    non-blank ``str`` becomes a one-item list), anything else is dropped.
    Each replaced value, a missing one included, is logged as one WARNING;
    well-shaped values pass unchanged.
    """
    # DECISION plan-2026-10-01T093600-944e2692/D-060: the classifier handler
    # is critical (D-058), so the model's values are normalised here. Do NOT
    # build ReasoningClassificationResult from the raw values again: a badly
    # shaped auxiliary field (a str alternative_approaches, seen live) would
    # fail the whole solve. Do NOT drop `.critical()` to absorb it instead:
    # that also hides a spent classifier budget or a provider error.
    recommended = result.get(ContextKeys.RECOMMENDED_REASONING_TYPE)
    if not (isinstance(recommended, str) and recommended.strip()):
        if recommended is None:
            logger.warning(
                "No recommended reasoning type was produced (low confidence, a cut "
                "reply or a failed classification call): using analytical"
            )
        else:
            logger.warning(f"Malformed recommended reasoning type: {recommended!r}")
        recommended = ReasoningType.ANALYTICAL.value
    texts: dict[str, str] = {}
    for key in (ContextKeys.STRATEGY_JUSTIFICATION, ContextKeys.PROBLEM_DOMAIN):
        value = result.get(key)
        if value is not None and not isinstance(value, str):
            logger.warning(f"Malformed classifier field {key} dropped: {value!r}")
        texts[key] = value if isinstance(value, str) else ""
    approaches = result.get(ContextKeys.ALTERNATIVE_APPROACHES)
    items = [approaches] if isinstance(approaches, str) else approaches
    alternatives = (
        [item for item in items if isinstance(item, str) and item.strip()]
        if isinstance(items, list)
        else []
    )
    if approaches is not None and alternatives != approaches:
        logger.warning(f"Malformed alternative approaches normalised: {approaches!r}")
    return ReasoningClassificationResult(
        recommended_type=recommended,
        justification=texts[ContextKeys.STRATEGY_JUSTIFICATION],
        domain=texts[ContextKeys.PROBLEM_DOMAIN],
        alternatives=alternatives,
    )


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
