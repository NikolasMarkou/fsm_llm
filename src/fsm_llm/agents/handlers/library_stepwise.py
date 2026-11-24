"""The step-wise pattern factories of the handler library (split from ``library.py``, plan a555dcfe).

The Plan-and-Execute four (``agents.plan_step_executor``, ``plan_step_fresh_keys``, ``plan_step_checker`` and
``plan_replan_counter``; bodies in ``plan_execute.py``) and the Maker-Checker five (``agents.revision_count``,
``maker_checker_checker``, ``revise_entry``, ``revise_exit`` and ``force_pass``; bodies in
``maker_checker.py``). The interface contract of a factory is in ``library.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fsm_llm.agents.constants import (
    ContextKeys,
    Defaults,
    HandlerNames,
    HandlerPriorities,
    MakerCheckerStates,
    PlanExecuteStates,
)
from fsm_llm.handlers import HandlerTiming, create_handler
from fsm_llm.tools import require_int_limit

from .context import HandlerContext
from .freshness import make_fresh_keys_handler
from .library_support import _keys, _on_entry, _priority, _states
from .maker_checker import (
    count_revision,
    force_pass_at_limit,
    make_revise_handlers,
    track_revisions,
)
from .plan_execute import (
    make_replan_handler,
    make_result_checker,
    make_step_tracker,
)

if TYPE_CHECKING:
    from fsm_llm.core.definitions.executable_pattern import HandlerBinding
    from fsm_llm.handlers import BaseHandler


def plan_step_executor(
    ctx: HandlerContext, binding: HandlerBinding | None = None
) -> list[BaseHandler]:
    """``agents.plan_step_executor``: bounds the plan and describes the current step on entry to ``execute_step``."""
    return _on_entry(
        HandlerNames.PLAN_STEP_EXECUTOR,
        binding,
        PlanExecuteStates.EXECUTE_STEP,
        HandlerPriorities.TOOL_EXECUTOR,
        make_step_tracker(),
    )


def plan_step_fresh_keys(
    ctx: HandlerContext, binding: HandlerBinding | None = None
) -> list[BaseHandler]:
    """``agents.plan_step_fresh_keys``: clears ``step_result`` on entry to ``execute_step`` (core extracts a key only while unset)."""
    return _on_entry(
        HandlerNames.PLAN_STEP_FRESH_KEYS,
        binding,
        PlanExecuteStates.EXECUTE_STEP,
        HandlerPriorities.TOOL_EXECUTOR,
        make_fresh_keys_handler(_keys(binding, (ContextKeys.STEP_RESULT,))),
    )


def plan_step_checker(
    ctx: HandlerContext, binding: HandlerBinding | None = None
) -> list[BaseHandler]:
    """``agents.plan_step_checker``: records the step and decides ``step_failed`` on entry to ``check_result``.

    The cap is the option ``max_replans`` (at least 0: no replans). It runs one priority after the
    tool executor, which writes the status it reads.
    """
    # DECISION plan-2026-09-29T103145-06a5ec0a/D-032
    # step_failed is decided on check_result ENTRY, from the status of the
    # tool that just ran (the executor above, D-024 order), and routes the
    # check_result turn that follows. Do NOT move it to PRE_TRANSITION on
    # check_result (core picks the edge before PRE_TRANSITION runs, D-031)
    # and do NOT let the model extract step_failed (a False seed is never
    # re-extracted, and a guessed verdict is not a tool outcome). The
    # max_replans check lives here too, so a failure past the cap routes to
    # synthesize instead of entering replan. See decisions.md D-032.
    limit = require_int_limit(
        "max_replans",
        ctx.options.get("max_replans", Defaults.MAX_REPLANS),
        minimum=0,
    )
    return _on_entry(
        HandlerNames.PLAN_STEP_CHECKER,
        binding,
        PlanExecuteStates.CHECK_RESULT,
        HandlerPriorities.TOOL_EXECUTOR + 1,
        make_result_checker(limit),
    )


def plan_replan_counter(
    ctx: HandlerContext, binding: HandlerBinding | None = None
) -> list[BaseHandler]:
    """``agents.plan_replan_counter``: counts the replan and reopens ``plan_steps`` on entry to ``replan``."""
    return _on_entry(
        HandlerNames.PLAN_REPLANNER,
        binding,
        PlanExecuteStates.REPLAN,
        HandlerPriorities.TOOL_EXECUTOR,
        make_replan_handler(),
    )


def revision_count(
    ctx: HandlerContext, binding: HandlerBinding | None = None
) -> list[BaseHandler]:
    """``agents.revision_count``: bumps ``revision_count`` on entry to ``check`` (D-009 of plan 5ed71cf2)."""
    return _on_entry(
        HandlerNames.MAKER_CHECKER_REVISION_COUNT,
        binding,
        MakerCheckerStates.CHECK,
        HandlerPriorities.TOOL_EXECUTOR,
        count_revision,
    )


def maker_checker_checker(
    ctx: HandlerContext, binding: HandlerBinding | None = None
) -> list[BaseHandler]:
    """``agents.maker_checker_checker``: the trace and the quality auto-pass, CONTEXT_UPDATE on ``check``.

    Fires only on a turn that extracted ``checker_passed`` or ``quality_score``. The options
    ``max_revisions`` and ``quality_threshold`` are the defaults for a context that carries neither.
    """
    max_revisions = ctx.options.get("max_revisions", Defaults.MAX_REVISIONS)
    threshold = ctx.options.get("quality_threshold", Defaults.QUALITY_THRESHOLD)
    return [
        create_handler(HandlerNames.MAKER_CHECKER_CHECKER)
        .with_priority(_priority(binding, HandlerPriorities.TOOL_EXECUTOR))
        .at(HandlerTiming.CONTEXT_UPDATE)
        .on_state(*_states(binding, (MakerCheckerStates.CHECK,)))
        .when_keys_updated(ContextKeys.CHECKER_PASSED, ContextKeys.QUALITY_SCORE)
        .do(lambda context: track_revisions(context, max_revisions, threshold))
    ]


def revise_entry(
    ctx: HandlerContext, binding: HandlerBinding | None = None
) -> list[BaseHandler]:
    """``agents.revise_entry``: on entry to ``revise`` the draft moves aside and a False verdict clears."""
    return _on_entry(
        HandlerNames.MAKER_CHECKER_REVISE_ENTRY,
        binding,
        MakerCheckerStates.REVISE,
        HandlerPriorities.TOOL_EXECUTOR,
        make_revise_handlers()[0],
    )


def revise_exit(
    ctx: HandlerContext, binding: HandlerBinding | None = None
) -> list[BaseHandler]:
    """``agents.revise_exit``: on exit from ``revise`` the consumed feedback clears (a missing draft is restored)."""
    return [
        create_handler(HandlerNames.MAKER_CHECKER_REVISE_EXIT)
        .with_priority(_priority(binding, HandlerPriorities.TOOL_EXECUTOR))
        .on_state_exit(*_states(binding, (MakerCheckerStates.REVISE,)))
        .do(make_revise_handlers()[1])
    ]


def force_pass(
    ctx: HandlerContext, binding: HandlerBinding | None = None
) -> list[BaseHandler]:
    """``agents.force_pass``: forces ``checker_passed`` on a ``check`` turn once the budget or the cap is spent."""
    return [
        create_handler(HandlerNames.MAKER_CHECKER_FORCE_PASS)
        .with_priority(_priority(binding, HandlerPriorities.TOOL_EXECUTOR))
        .at(HandlerTiming.PRE_TRANSITION)
        .on_state(*_states(binding, (MakerCheckerStates.CHECK,)))
        .do(force_pass_at_limit)
    ]
