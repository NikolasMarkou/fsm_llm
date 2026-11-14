"""The iteration limiters of the agent patterns (plan d454fd62 step 88): ``ThinkTurnLimiter``, which
counts think turns on the instance (the ReAct family), and ``make_iteration_limiter``, which counts in
context (the other patterns). The class and function bodies are the old ``agents/handlers.py`` ones."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fsm_llm.agents.constants import AgentStates, ContextKeys, Defaults, LogMessages
from fsm_llm.logging import logger

from .forced import concluded_on_evidence, recorded_forced_reason


class ThinkTurnLimiter:
    """The ReAct-family iteration limiter: counts think turns on the instance.

    Interface contract (callers: ``AgentHandlers``, which inherits it, and
    ``ParallelReactAgent``, which needs the limiter but runs its own tool
    batch and never passes ``gated=``): create one per ``run()`` call (never
    store it on ``self``, 089d0ec7 D-014) and register
    :meth:`check_iteration_limit` as the PRE_TRANSITION limiter. Construction
    never raises.
    """

    def __init__(self) -> None:
        self._current_iteration = 0
        self._transitions = 0

    def check_iteration_limit(self, context: dict[str, Any]) -> dict[str, Any]:
        """Count think turns and force the stop once the budget is spent.

        Called as a PRE_TRANSITION handler on every transition. ``max_iterations``
        (``_max_iterations`` in context) counts think turns: the count rises on
        each transition out of ``think``. The stop is forced (``max_iterations_
        reached`` and ``should_terminate`` True) when a later transition closes
        a cycle with ``count >= max - 1``, so the next think turn concludes: a
        limit of N gives N think turns and N - 1 tool turns, and no tool runs
        after the flag. An approved ``await_approval -> act`` exit does not close
        the cycle (its call still runs). Independently, the stop is forced
        ``FORCED_STOP_MARGIN`` transitions (at most ``max``) before the
        ``max * FSM_BUDGET_MULTIPLIER`` loop ceiling, so a run with long cycles
        concludes instead of raising ``BudgetExhaustedError``. A limit of 1
        behaves like 2 (the first cycle's tool always runs). A turn that
        concludes on its own evidence (:func:`concluded_on_evidence`) with no
        recorded forced reason is never forced, and withdraws an earlier flag
        (``max_iterations_reached`` back to False), so the last think turn's
        own conclusion reports ``success=True``. Never raises.
        """
        # DECISION plan-2026-09-29T103145-06a5ec0a/D-028
        # LOOP-04: count think exits here, on the instance. Do NOT move this
        # into make_iteration_limiter or give that factory a state filter
        # (3e4eb3e5 D-003, c1d5bfbc D-007: the other patterns count every turn).
        # Do NOT force the stop on the think exit itself: the think -> act
        # decision is already made, so the cycle's tool would be skipped and
        # max_iterations=1 would run no tool at all. Keep the `- 1` early rule:
        # the flag lands at the end of cycle max - 1, and think turn max
        # concludes on it. See decisions.md D-028.
        self._transitions += 1
        leaving = context.get(ContextKeys.CURRENT_STATE) or AgentStates.THINK
        if leaving == AgentStates.THINK:
            self._current_iteration += 1
        max_iterations = context.get(
            ContextKeys.ITERATION_BUDGET, Defaults.MAX_ITERATIONS
        )

        logger.debug(
            LogMessages.ITERATION.format(
                current=self._current_iteration, max=max_iterations
            )
        )

        # DECISION plan-2026-09-29T103145-06a5ec0a/D-051: a turn whose own
        # verdict concludes on evidence (think: should_terminate, which think
        # entry refreshed; Reflexion evaluate: evaluation_passed) is the
        # model's conclusion, not the budget's, even after the flag: the
        # flag is withdrawn so the run reports success. Do NOT drop the
        # recorded-reason check (a stall or reflection cap stays forced) and
        # do NOT read should_terminate on a state that does not refresh it.
        if (
            concluded_on_evidence(leaving, context)
            and recorded_forced_reason(context) is None
        ):
            update: dict[str, Any] = {
                ContextKeys.ITERATION_COUNT: self._current_iteration
            }
            if context.get(ContextKeys.MAX_ITERATIONS_REACHED) is True:
                logger.info("The model concluded on its own on the last turn")
                update[ContextKeys.MAX_ITERATIONS_REACHED] = False
            return update

        approved = (
            leaving == AgentStates.AWAIT_APPROVAL
            and context.get(ContextKeys.APPROVAL_GRANTED) is True
        )
        cycle_closed = leaving != AgentStates.THINK and not approved
        spent = cycle_closed and self._current_iteration >= max_iterations - 1
        margin = min(Defaults.FORCED_STOP_MARGIN, max_iterations)
        ceiling = max_iterations * Defaults.FSM_BUDGET_MULTIPLIER
        if spent or self._transitions >= ceiling - margin:
            return {
                ContextKeys.ITERATION_COUNT: self._current_iteration,
                ContextKeys.MAX_ITERATIONS_REACHED: True,
                ContextKeys.SHOULD_TERMINATE: True,
            }

        return {ContextKeys.ITERATION_COUNT: self._current_iteration}


# DECISION plan-2026-09-24T045559-3e4eb3e5/D-003: the 8 context-counting
# patterns build their limiter here; do NOT hand-roll a per-pattern copy (the
# copies drifted: 7 of 8 triggered at `>= max`, not `>= max - 1`), and do NOT
# fold this into AgentHandlers.check_iteration_limit (it counts on the
# instance, these count in context; merging changes ReAct semantics).
def make_iteration_limiter(
    max_iterations: int,
    forced: dict[str, Any],
    *,
    context_max_key: str | None = None,
    early: bool = True,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """
    Build a PRE_TRANSITION iteration limiter that counts in context.

    Args:
        max_iterations: The limit, or the fallback when ``context_max_key``
            is given but absent from the context.
        forced: Context updates returned verbatim (copied once) when the limit
            is hit, e.g. ``{SHOULD_TERMINATE: True}``.
        context_max_key: Optional context key whose value overrides
            ``max_iterations`` at call time.
        early: ``True`` (default) triggers at ``limit - 1``; ``False``
            triggers at ``limit`` (maker_checker only, see D-014).

    Returns:
        A handler returning ``{ITERATION_COUNT: count}``, plus ``forced`` once
        ``count >= limit - 1`` (``count >= limit`` when not ``early``). It
        never raises.

    The transition decision is already made before a PRE_TRANSITION handler
    fires, so with ``early`` the limiter triggers one iteration early
    (``>= max - 1``) and the forced transition fires on the next iteration
    rather than overshooting by 1. The forced keys only change routing where
    a transition reads them. PRE_TRANSITION handlers do not run on a BLOCKED
    turn, so the run's step ceiling (core ``max_steps``, set by
    ``BaseAgent._run_budgets``) is the backstop bound.
    """
    forced_updates = dict(forced)

    def check_iteration_limit(context: dict[str, Any]) -> dict[str, Any]:
        count = context.get(ContextKeys.ITERATION_COUNT, 0) + 1
        limit = (
            context.get(context_max_key, max_iterations)
            if context_max_key is not None
            else max_iterations
        )
        logger.debug(LogMessages.ITERATION.format(current=count, max=limit))
        if count >= (limit - 1 if early else limit):
            return {ContextKeys.ITERATION_COUNT: count, **forced_updates}
        return {ContextKeys.ITERATION_COUNT: count}

    return check_iteration_limit
