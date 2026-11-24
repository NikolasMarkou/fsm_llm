"""The worker-pattern factories of the handler library (split from ``library.py``, plan a555dcfe).

The orchestrator pair (``agents.orchestrator_delegator``, ``agents.orchestrator_decision_reset``), the hook
``agents.raise_run_ending_error`` and the REWOO executor and hook (``agents.rewoo_executor``,
``agents.rewoo_cut_off_answer``). The interface contract of a factory is in ``library.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from fsm_llm.agents.constants import (
    ContextKeys,
    Defaults,
    ErrorMessages,
    HandlerNames,
    HandlerPriorities,
    OrchestratorStates,
    REWOOStates,
)
from fsm_llm.core.definitions import AgentError
from fsm_llm.handlers import create_handler

from .context import RUN_RESULT, HandlerContext
from .library_support import _priority, _states
from .orchestrator import delegate_to_workers
from .rewoo import execute_plans
from .run_guards import RunEndingErrorHolder

if TYPE_CHECKING:
    from fsm_llm.core.definitions.executable_pattern import HandlerBinding
    from fsm_llm.handlers import BaseHandler

#: The ``scope`` key of the run's one ``RunEndingErrorHolder``.
_HOLDER = "run_ending_error_holder"


def _run_holder(ctx: HandlerContext) -> RunEndingErrorHolder:
    """The run's one error holder (in ``ctx.scope``, so the delegator and the hook share it)."""
    holder = ctx.scope.get(_HOLDER)
    if holder is None:
        holder = ctx.scope[_HOLDER] = RunEndingErrorHolder()
    result: RunEndingErrorHolder = holder
    return result


def orchestrator_delegator(
    ctx: HandlerContext, binding: HandlerBinding | None = None
) -> list[BaseHandler]:
    """``agents.orchestrator_delegator``: runs the callable ``worker_factory`` per subtask on entry to ``delegate``.

    Reads the option ``max_workers`` (an int of at least 1) and the optional callable ``worker_factory``
    (none: the model fills each subtask). A worker's run-ending error goes into the run's holder, which
    ``agents.raise_run_ending_error`` re-raises after the run.
    """
    max_workers = ctx.options.get("max_workers", Defaults.MAX_WORKERS)
    if (
        isinstance(max_workers, bool)
        or not isinstance(max_workers, int)
        or max_workers < 1
    ):
        raise AgentError(
            f"agents.orchestrator_delegator needs max_workers >= 1, got {max_workers!r}"
        )
    delegate = ctx.callables.get("worker_factory")
    holder = _run_holder(ctx)
    return [
        create_handler(HandlerNames.ORCHESTRATOR_DELEGATOR)
        .with_priority(_priority(binding, HandlerPriorities.TOOL_EXECUTOR))
        .on_state_entry(*_states(binding, (OrchestratorStates.DELEGATE,)))
        .do(lambda context: delegate_to_workers(context, delegate, max_workers, holder))
    ]


def orchestrator_decision_reset(
    ctx: HandlerContext, binding: HandlerBinding | None = None
) -> list[BaseHandler]:
    """``agents.orchestrator_decision_reset``: clears ``all_collected`` on entry to ``orchestrate``."""
    # DECISION plan-2026-09-24T091842-c1d5bfbc/D-009
    # Core extracts all_collected only while it is unset, so round 1's
    # False would stick and collect could never extract a later True.
    # Clear it on entry to orchestrate (the redo state), NOT on entry to
    # collect: that would erase the limiter's forced True (PRE_TRANSITION
    # runs before entry; plan-2026-09-24T045559-3e4eb3e5/D-013).
    return [
        create_handler(HandlerNames.ORCHESTRATOR_DECISION_RESET)
        .with_priority(_priority(binding, HandlerPriorities.TOOL_EXECUTOR))
        .on_state_entry(*_states(binding, (OrchestratorStates.ORCHESTRATE,)))
        .do(lambda _context: {ContextKeys.ALL_COLLECTED: None})
    ]


def raise_run_ending_error(
    ctx: HandlerContext, binding: HandlerBinding | None = None
) -> Callable[[], None]:
    """``agents.raise_run_ending_error`` (a HOOK, ``at: after_run``): re-raise what a worker left in the holder."""
    # DECISION plan-2026-09-29T103145-06a5ec0a/D-042: re-raise a worker's
    # AgentTimeoutError/BudgetExhaustedError HERE, after the run. Do NOT
    # re-raise inside the delegator (core's handler error mode swallows it)
    # and do NOT keep the holder on self (shared across runs, D-014).
    return _run_holder(ctx).raise_if_set


def rewoo_executor(
    ctx: HandlerContext, binding: HandlerBinding | None = None
) -> list[BaseHandler]:
    """``agents.rewoo_executor``: runs the plan with the run's tools on entry to ``execute_plans``."""
    tools = ctx.tools
    if tools is None:
        raise AgentError(
            "agents.rewoo_executor needs the run's tools and the context has none"
        )
    return [
        create_handler(HandlerNames.REWOO_EXECUTOR)
        .with_priority(_priority(binding, HandlerPriorities.TOOL_EXECUTOR))
        .on_state_entry(*_states(binding, (REWOOStates.EXECUTE_PLANS,)))
        .do(lambda context: execute_plans(context, tools))
    ]


def rewoo_cut_off_answer(
    ctx: HandlerContext, binding: HandlerBinding | None = None
) -> Callable[[], None]:
    """``agents.rewoo_cut_off_answer`` (a HOOK, ``at: after_run``): a plan cut off at ``max_tokens`` ships its cause."""

    # DECISION plan-2026-11-06T093029-0068b055/D-025: a plan cut off at
    # max_tokens ships the cause as the answer, not the `solve` state's
    # filler about evidence that does not exist. The `solve` call has
    # already been made by then (the FSM is linear; adding a branch for the
    # rare cut is not worth a second plan path). Do NOT run the cut plan's
    # prefix and do NOT retry at a higher cap here.
    def rewrite() -> None:
        result = ctx.scope.get(RUN_RESULT)
        if (
            result is not None
            and result.final_context.get(ContextKeys.PLAN_CUT_OFF) is True
        ):
            ctx.scope[RUN_RESULT] = result.model_copy(
                update={
                    "answer": ErrorMessages.REWOO_PLAN_CUT_OFF,
                    "structured_output": None,
                }
            )

    return rewrite
