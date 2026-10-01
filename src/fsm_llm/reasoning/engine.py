"""
Structured Reasoning Engine for FSM-LLM
=======================================

Enhanced with loop prevention, context management, and standardized handling.
"""

from __future__ import annotations

import threading
from typing import Any

from fsm_llm import (
    API,
    AdvanceResult,
    ContextMergeStrategy,
    FSMError,
    RunBudgetExceededError,
    clear_keys_on_entry,
)
from fsm_llm.api import llm_settings_for
from fsm_llm.handlers import HandlerTiming
from fsm_llm.logging import logger

from .constants import (
    HYBRID_EVALUATION_STATE,
    RETRY_CLEARED_KEYS,
    SOLVE_DRIVER_KEYS,
    ContextKeys,
    Defaults,
    ErrorMessages,
    HandlerNames,
    LogMessages,
    OrchestratorStates,
    ReasoningType,
)
from .definitions import ReasoningClassificationResult, ReasoningTrace
from .exceptions import ReasoningClassificationError, ReasoningExecutionError
from .handlers import ContextManager, OutputFormatter, ReasoningHandlers
from .utilities import load_fsm_definition, map_reasoning_type


class ReasoningEngine:
    """
    Main reasoning engine with enhanced error handling and flow control.
    """

    def __init__(self, model: str = Defaults.MODEL, **kwargs):
        """
        Initialize the reasoning engine.

        :param model: The LLM model to use
        :param kwargs: Additional arguments for the API

        Thread safety: ``solve_problem()`` is serialized via an internal lock
        so that only one solve runs at a time on a given engine instance.
        """
        self._solve_lock = threading.Lock()
        self.model = model
        self.api_kwargs = kwargs.copy()

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
        """Load all FSM definitions with error handling."""
        try:
            self.main_fsm = load_fsm_definition("orchestrator")
            self.classifier_fsm = load_fsm_definition("classifier")

            # Load reasoning FSMs
            self.reasoning_fsms = {}
            failed_fsms = []
            for reasoning_type in ReasoningType:
                try:
                    fsm = load_fsm_definition(reasoning_type.value)
                    self.reasoning_fsms[reasoning_type] = fsm
                except Exception as e:
                    failed_fsms.append(reasoning_type.value)
                    logger.warning(
                        f"Could not load FSM for {reasoning_type.value}: {e}"
                    )
            if failed_fsms:
                logger.error(
                    f"Failed to load {len(failed_fsms)} reasoning FSM(s): "
                    f"{', '.join(failed_fsms)}. These reasoning types will "
                    "fall back to analytical."
                )
        except Exception as e:
            logger.error(f"Failed to load core FSM definitions: {e}")
            raise

    def _initialize_apis(self):
        """Initialize APIs with all necessary handlers."""
        # An injected `llm_interface` owns its model: core refuses another
        # one beside it, so `model` goes only to an interface core builds.
        settings = llm_settings_for(self.api_kwargs, model=self.model)
        # Main orchestrator API
        self.orchestrator = API.from_definition(
            self.main_fsm, **settings, **self.api_kwargs
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
        # strategy executor, validator, retry limiter and hybrid loop counter
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

        # Strategy executor
        self.orchestrator.register_handler(
            self.orchestrator.create_handler(HandlerNames.ORCHESTRATOR_EXECUTOR)
            .on_state_entry(OrchestratorStates.EXECUTE_REASONING)
            .critical()
            .do(self._prepare_reasoning_execution)
        )

        # Solution validator with retry logic
        self.orchestrator.register_handler(
            self.orchestrator.create_handler(HandlerNames.ORCHESTRATOR_VALIDATOR)
            .at(HandlerTiming.CONTEXT_UPDATE)
            .when_keys_updated(ContextKeys.PROPOSED_SOLUTION)
            .critical()
            .do(self.handlers.validate_solution)
        )

        # Context pruner
        self.orchestrator.register_handler(
            self.orchestrator.create_handler(HandlerNames.CONTEXT_PRUNER)
            .at(HandlerTiming.PRE_TRANSITION)
            .do(self.handlers.prune_context)
        )

        # Reasoning tracer for both APIs
        tracer = (
            self.orchestrator.create_handler(HandlerNames.REASONING_TRACER)
            .at(HandlerTiming.POST_TRANSITION)
            .do(self.handlers.update_reasoning_trace)
        )

        self.orchestrator.register_handler(tracer)
        self.classifier.register_handler(tracer)

        # Retry limiter for validation loops
        self.orchestrator.register_handler(
            self.orchestrator.create_handler(HandlerNames.RETRY_LIMITER)
            .on_state_entry(OrchestratorStates.VALIDATE_REFINE)
            .critical()
            .do(self._check_retry_limit)
        )

        # Hybrid back-edge counter: strategy FSMs run on the orchestrator API,
        # and only the hybrid FSM has this state.
        self.orchestrator.register_handler(
            self.orchestrator.create_handler(HandlerNames.HYBRID_LOOP_COUNTER)
            .on_state_exit(HYBRID_EVALUATION_STATE)
            .critical()
            .do(self.handlers.count_hybrid_loop)
        )

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

        conv_id = None
        try:
            # Run classification
            conv_id, _ = self.classifier.start_conversation(
                initial_context=classification_context
            )

            # Message-free steps: the problem reaches the model through the
            # conversation context, never through a synthetic user message.
            self.classifier.run_until_terminal(
                conv_id, max_steps=Defaults.MAX_CLASSIFICATION_ITERATIONS
            )

            # Get results
            result = self.classifier.get_data(conv_id)
            self.classifier.end_conversation(conv_id)
            conv_id = None  # Mark as cleaned up
        except RunBudgetExceededError as e:
            raise ReasoningClassificationError(
                "Classification did not converge after "
                f"{Defaults.MAX_CLASSIFICATION_ITERATIONS} steps",
                details={"context_keys": list(classification_context.keys())},
            ) from e
        except Exception as e:
            raise ReasoningClassificationError(
                f"Problem classification failed: {e}",
                details={"context_keys": list(classification_context.keys())},
            ) from e
        finally:
            # Clean up classifier conversation on any error path
            if conv_id is not None:
                try:
                    self.classifier.end_conversation(conv_id)
                except Exception as cleanup_err:
                    logger.debug(f"Cleanup error (suppressed): {cleanup_err}")

        # Create classification result
        classification = ReasoningClassificationResult(
            recommended_type=result.get(
                ContextKeys.RECOMMENDED_REASONING_TYPE, "analytical"
            ),
            justification=result.get(ContextKeys.STRATEGY_JUSTIFICATION, ""),
            domain=result.get(ContextKeys.PROBLEM_DOMAIN, ""),
            alternatives=result.get(ContextKeys.ALTERNATIVE_APPROACHES, []),
        )

        logger.info(
            LogMessages.CLASSIFICATION_COMPLETE.format(
                type=classification.recommended_type
            )
        )

        return {
            ContextKeys.CLASSIFIED_PROBLEM_TYPE: classification.recommended_type,
            ContextKeys.CLASSIFICATION_JUSTIFICATION: classification.justification,
            ContextKeys.PROBLEM_DOMAIN: classification.domain,
            ContextKeys.ALTERNATIVE_APPROACHES: classification.alternatives,
        }

    def _prepare_reasoning_execution(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Prepare for reasoning execution with proper strategy selection.

        :param context: Current context
        :return: Execution preparation results
        """
        # Determine reasoning type
        orchestrator_strategy = context.get(ContextKeys.REASONING_STRATEGY)
        classified_type = context.get(ContextKeys.CLASSIFIED_PROBLEM_TYPE)

        # Priority: direct computation > classified type > orchestrator strategy
        if orchestrator_strategy == "direct computation":
            reasoning_type_str = "simple_calculator"
        elif classified_type:
            reasoning_type_str = map_reasoning_type(classified_type)
        elif orchestrator_strategy:
            reasoning_type_str = map_reasoning_type(orchestrator_strategy)
        else:
            reasoning_type_str = "analytical"  # Default

        # Allow CLI --type / initial_context override (R-ISSUE-003)
        # DECISION plan_2026-05-29_d9092060/D-004 [STALE]
        # NOTE: preferred_reasoning_type is checked AFTER the normal priority chain.
        # Overriding earlier would bypass the FSM's own classified type which the
        # orchestrator FSM has already validated; we only apply the preference when
        # it maps to a known ReasoningType value.
        preferred = context.get(ContextKeys.PREFERRED_REASONING_TYPE)
        if preferred:
            preferred_mapped = map_reasoning_type(str(preferred))
            try:
                ReasoningType(preferred_mapped)
                reasoning_type_str = preferred_mapped
                logger.debug(
                    f"Using preferred_reasoning_type from context: {reasoning_type_str!r}"
                )
            except ValueError:
                logger.warning(
                    f"preferred_reasoning_type {preferred!r} maps to unknown type "
                    f"{preferred_mapped!r}; ignoring"
                )

        # Get enum member
        try:
            reasoning_type = ReasoningType(reasoning_type_str)
        except ValueError:
            logger.warning(
                ErrorMessages.INVALID_REASONING_TYPE.format(type=reasoning_type_str)
            )
            reasoning_type = ReasoningType.ANALYTICAL

        # The strategy FSM is pushed by type from the before_step hook
        # (``_StrategyStack``); only the type and a flag go into context.
        fsm_def = self.reasoning_fsms.get(reasoning_type)

        if not fsm_def:
            logger.error(ErrorMessages.FSM_NOT_FOUND.format(name=reasoning_type.value))
            # DECISION plan-2026-09-12T065608-089d0ec7/D-009
            # Fallback ONLY to ANALYTICAL, never to an arbitrary other loaded
            # type. Do NOT restore the old `[ReasoningType.ANALYTICAL,
            # *self.reasoning_fsms]` loop: silently substituting, say,
            # "deductive" for a missing "creative" FSM would solve the
            # caller's problem with the wrong reasoning strategy while only
            # logging a warning — the caller has no structured signal that a
            # different strategy was used. See decisions.md D-009.
            fsm_def = self.reasoning_fsms.get(ReasoningType.ANALYTICAL)
            if fsm_def:
                reasoning_type = ReasoningType.ANALYTICAL
                logger.warning(
                    f"Falling back to {ReasoningType.ANALYTICAL.value} reasoning"
                )
            else:
                raise ReasoningExecutionError("No reasoning FSM definitions available")

        logger.info(LogMessages.STRATEGY_EXECUTING.format(type=reasoning_type.value))

        return {
            ContextKeys.REASONING_PUSH_PENDING: True,
            ContextKeys.REASONING_TYPE_SELECTED: reasoning_type.value,
            ContextKeys.CLASSIFICATION_JUSTIFICATION: context.get(
                ContextKeys.CLASSIFICATION_JUSTIFICATION, ""
            ),
        }

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
        max_reached = retry_count >= Defaults.MAX_RETRIES

        if max_reached:
            logger.warning(ErrorMessages.MAX_RETRIES_EXCEEDED)

        return {**verdict, ContextKeys.MAX_RETRIES_REACHED: max_reached}

    def solve_problem(
        self, problem: str, initial_context: dict[str, Any] | None = None
    ) -> tuple[str, dict[str, Any]]:
        """
        Solve a problem using structured reasoning.

        This method is serialized: only one ``solve_problem`` call executes
        at a time per engine instance.

        :param problem: The problem statement
        :param initial_context: Optional initial context
        :return: Tuple of (solution, trace_info)
        :raises ReasoningExecutionError: If reasoning execution fails
        """
        with self._solve_lock:
            return self._solve_problem_locked(problem, initial_context)

    def _solve_problem_locked(
        self, problem: str, initial_context: dict[str, Any] | None = None
    ) -> tuple[str, dict[str, Any]]:
        """Internal solve logic, must be called while holding ``_solve_lock``."""
        # Initialize context (copy to avoid mutating caller's dict)
        context = dict(initial_context) if initial_context else {}
        # The push hook's driver keys are the engine's own (D-058).
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

        # Start orchestrator
        conv_id, initial_response = self.orchestrator.start_conversation(context)
        log = _solve_log(conv_id)
        log.info(f"Started reasoning process: {conv_id}")

        stack = _StrategyStack(self, conv_id)
        try:
            # One message-free run drives the orchestrator, every pushed
            # strategy FSM and every retry; the hook pushes and pops.
            results = self.orchestrator.run_until_terminal(
                conv_id, max_steps=Defaults.MAX_SOLVE_STEPS, before_step=stack
            )
        except RunBudgetExceededError as e:
            # DECISION plan-2026-10-01T093600-944e2692/D-054 (D-014): an
            # unfinished solve raises, never returns a fallback string as the
            # solution.
            raise self._failed_solve(
                conv_id, stack, f"Reasoning did not finish within {e.limit} {e.budget}"
            ) from e
        except Exception as e:
            raise self._failed_solve(
                conv_id, stack, f"Reasoning execution failed: {e}"
            ) from e
        responses = _ordered_responses(initial_response, stack.responses, results)

        # Get final context BEFORE ending the conversation (end_conversation
        # removes the instance, making get_data fail).
        try:
            final_context = self.orchestrator.get_data(conv_id)
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
            self._end_quietly(conv_id)

    def _failed_solve(
        self, conv_id: str, stack: _StrategyStack, message: str
    ) -> ReasoningExecutionError:
        """End a solve whose run raised; return the error to raise.

        Contract: reads the partial context, ends the conversation (a failure
        to end is logged), and returns ``ReasoningExecutionError(message)``
        with ``details={conversation_id, responses_so_far, partial_context}``
        for the caller to raise ``from`` the run's error; ``partial_context``
        is ``None`` when core cannot read it (``_read_partial_context``).
        """
        # DECISION plan-2026-10-01T093600-944e2692/D-058: every failed solve
        # (spent budget or any error, e.g. a prompt over core's cap on the
        # final step) carries what it had. Do NOT end the conversation before
        # reading it: the live read is the documented one, while after
        # end_conversation core answers get_data only from its bounded
        # ended-conversation cache, which may already have evicted it.
        partial_context = self._read_partial_context(conv_id)
        self._end_quietly(conv_id)
        return ReasoningExecutionError(
            message,
            details={
                "conversation_id": conv_id,
                "responses_so_far": 1 + len(stack.responses),
                "partial_context": partial_context,
            },
        )

    def _read_partial_context(self, conv_id: str) -> dict[str, Any] | None:
        """The context of the frame on top of a solve that did not finish.

        Must run before the conversation is ended. Returns ``None`` (with a
        WARNING) when core cannot read it, so the caller's error still
        reports the unfinished solve.
        """
        try:
            data: dict[str, Any] = self.orchestrator.get_data(conv_id)
            return data
        except FSMError as read_err:
            _solve_log(conv_id).warning(
                f"Could not read the partial context of {conv_id}: {read_err}"
            )
            return None

    def _end_quietly(self, conv_id: str) -> None:
        """End the orchestrator conversation; a failure is logged, not raised
        (it runs on paths that already return or raise a result)."""
        try:
            self.orchestrator.end_conversation(conv_id)
        except Exception as cleanup_err:
            _solve_log(conv_id).warning(
                f"Failed to clean up conversation {conv_id}: {cleanup_err}"
            )

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


class _StrategyStack:
    """The ``before_step`` hook of one solve's orchestrator run.

    Contract: called by core ``run_until_terminal`` with the number of the
    next step (and once more, with the same number, when a pushed frame has
    ended, D-052). Per call, on conversation ``conversation_id`` of
    ``engine.orchestrator``:

    - a pushed strategy FSM that has ended is popped, its results merged into
      the orchestrator with ``ContextManager.merge_reasoning_results``;
    - a pushed strategy FSM still running after ``MAX_SUB_FSM_ITERATIONS``
      steps is force-popped the same way;
    - with only the orchestrator on the stack and ``reasoning_push_pending``
      set, the flag is cleared and the strategy FSM of
      ``reasoning_type_selected`` is pushed with the problem keys.

    ``responses`` collects ``(step, reply)`` for each push and pop. What a
    core call raises propagates unchanged (the run stops, no step runs).
    """

    def __init__(self, engine: ReasoningEngine, conversation_id: str) -> None:
        self._engine = engine
        self._api = engine.orchestrator
        self._conversation_id = conversation_id
        self._log = _solve_log(conversation_id)
        self.responses: list[tuple[int, str]] = []
        self._pushed_at = 0
        self._reasoning_type = ""
        self._orchestrator_context: dict[str, Any] = {}

    def __call__(self, step: int) -> None:
        conv_id = self._conversation_id
        if self._api.get_stack_depth(conv_id) > 1:
            if self._api.has_conversation_ended(conv_id):
                self._pop(step)
            # DECISION plan-2026-10-01T093600-944e2692/D-053: the sub-FSM
            # step count is arithmetic on core's step number (pushed before
            # step `_pushed_at`, so `step - _pushed_at` sub steps have run).
            # Do NOT count steps in a loop of the engine's own (D-013).
            elif step - self._pushed_at >= Defaults.MAX_SUB_FSM_ITERATIONS:
                self._log.error(
                    f"Sub-FSM exceeded {Defaults.MAX_SUB_FSM_ITERATIONS} steps; "
                    "forcing completion"
                )
                self._pop(step)
            return
        context = self._api.get_data(conv_id)
        if context.get(ContextKeys.REASONING_PUSH_PENDING):
            self._push(step, context)

    def _push(self, step: int, context: dict[str, Any]) -> None:
        conv_id = self._conversation_id
        reasoning_type = ReasoningType(context[ContextKeys.REASONING_TYPE_SELECTED])
        fsm_def = self._engine.reasoning_fsms[reasoning_type]
        self._api.update_context(conv_id, {ContextKeys.REASONING_PUSH_PENDING: None})
        sub_context = self._engine.context_manager.extract_relevant_context(
            context,
            [
                ContextKeys.PROBLEM_STATEMENT,
                ContextKeys.PROBLEM_COMPONENTS,
                ContextKeys.CONSTRAINTS,
                ContextKeys.PROBLEM_TYPE,
            ],
        )
        self._log.info(
            LogMessages.FSM_PUSHED.format(
                name=fsm_def.get("name"),
                depth=self._api.get_stack_depth(conv_id) + 1,
            )
        )
        response = self._api.push_fsm(
            conv_id, fsm_def, inherit_context=False, context_to_pass=sub_context
        )
        self.responses.append((step, response))
        self._pushed_at = step
        self._reasoning_type = reasoning_type.value
        self._orchestrator_context = context

    def _pop(self, step: int) -> None:
        conv_id = self._conversation_id
        # Read the strategy FSM's context while it is still on top.
        sub_context = self._api.get_data(conv_id)
        results = self._engine.context_manager.merge_reasoning_results(
            self._orchestrator_context, sub_context, self._reasoning_type
        )
        response = self._api.pop_fsm(
            conv_id,
            context_to_return=results,
            merge_strategy=ContextMergeStrategy.UPDATE,
        )
        self.responses.append((step, response))
        self._log.info(
            LogMessages.FSM_POPPED.format(
                name=self._reasoning_type,
                depth=self._api.get_stack_depth(conv_id),
            )
        )


def _ordered_responses(
    initial_response: str,
    hook_responses: list[tuple[int, str]],
    results: tuple[AdvanceResult, ...],
) -> list[str]:
    """Every reply of a solve, in the order it was produced.

    Contract: ``hook_responses`` holds ``(step, reply)`` pairs in call order,
    each produced before step ``step`` ran; ``results[i]`` is step ``i + 1``.
    Returns ``initial_response``, then the replies interleaved by step; a
    silent step (``response is None``) adds nothing. Never raises.
    """
    ordered = [initial_response]
    pending = iter(hook_responses)
    upcoming = next(pending, None)
    for step, result in enumerate(results, start=1):
        while upcoming is not None and upcoming[0] <= step:
            ordered.append(upcoming[1])
            upcoming = next(pending, None)
        if result.response is not None:
            ordered.append(result.response)
    while upcoming is not None:
        ordered.append(upcoming[1])
        upcoming = next(pending, None)
    return ordered
