"""The orchestrator handlers of ``ReasoningEngine``, moved text for text out of ``engine.py`` (D-003)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fsm_llm import API, RunBudgetExceededError, clear_keys_on_entry
from fsm_llm.core.definitions import CallStateConfig
from fsm_llm.engine.context import RunContext
from fsm_llm.engine.kinds.call import chosen_target
from fsm_llm.handlers import HandlerTiming
from fsm_llm.logging import logger

from .choice import _classification_from, _strategy_for
from .constants import (
    RETRY_CLEARED_KEYS,
    ContextKeys,
    Defaults,
    ErrorMessages,
    HandlerNames,
    LogMessages,
    OrchestratorStates,
)
from .exceptions import ReasoningClassificationError
from .handlers import (
    ContextManager,
    ReasoningHandlers,
    register_strategy_handlers,
    retries_exhausted,
)
from .selection import StrategyChoice


class EngineHandlers:
    """Mixin of ``ReasoningEngine``: registers and runs its handlers (state set by the engine)."""

    orchestrator: Any  # an API, as untyped as at base (`closing` in `_run_streaming`)
    classifier: API
    handlers: ReasoningHandlers
    context_manager: ContextManager
    _choices: list[StrategyChoice]
    _analytical: StrategyChoice
    _names: dict[str, str]
    _call_config: CallStateConfig
    _context: RunContext | None

    def _run_hook(self) -> Callable[[int], None] | None:
        """ONE fresh ``before_step`` for one run loop of this engine (the context's), or ``None``."""
        return self._context.step_hook() if self._context is not None else None

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
                conversation_id,
                max_steps=Defaults.MAX_CLASSIFICATION_STEPS,
                before_step=self._run_hook(),
            )

            # Get results
            result = self.classifier.get_data(conversation_id)
            self.classifier.end_conversation(conversation_id)
            conversation_id = None  # Mark as cleaned up
        except RunBudgetExceededError as e:
            if self._context is not None and self._context.budget_spent():
                raise  # the shared ledger, not the classifier's own steps
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
