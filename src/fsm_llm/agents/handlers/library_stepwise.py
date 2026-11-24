"""The step-wise pattern factories of the handler library (split from ``library.py``, plan a555dcfe).

The Plan-and-Execute four (``agents.plan_step_executor``, ``plan_step_fresh_keys``, ``plan_step_checker`` and
``plan_replan_counter``; bodies in ``plan_execute.py``), the Maker-Checker five (``agents.revision_count``,
``maker_checker_checker``, ``revise_entry``, ``revise_exit`` and ``force_pass``; bodies in
``maker_checker.py``), the Debate three (``agents.debate_judge``, ``debate_consensus_reset`` and
``debate_limiter``; bodies in ``debate.py``) and the Evaluator-Optimizer three (``agents.evalopt_evaluator``,
``refine_entry`` and ``refine_exit``; body in ``evaluator_optimizer.py``). The interface contract of a factory
is in ``library.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from fsm_llm.agents.constants import (
    ContextKeys,
    DebateStates,
    Defaults,
    EvalOptStates,
    HandlerNames,
    HandlerPriorities,
    MakerCheckerStates,
    PlanExecuteStates,
)
from fsm_llm.core.definitions import AgentError
from fsm_llm.handlers import HandlerTiming, create_handler
from fsm_llm.tools import require_int_limit

from .context import HandlerContext
from .debate import (
    ROUND_KEYS,
    make_judge_handler,
    make_propose_entry_handler,
    record_forced_consensus,
)
from .evaluator_optimizer import evaluate_output
from .freshness import make_fresh_keys_handler, make_redraft_handlers
from .library_react import iteration_check
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


def debate_judge(
    ctx: HandlerContext, binding: HandlerBinding | None = None
) -> list[BaseHandler]:
    """``agents.debate_judge``: records the round and forces the cap, PRE_TRANSITION on ``judge``.

    The cap is the option ``max_rounds`` (at least 1). It runs one priority before the iteration
    limiter, so a cap on the same turn is recorded as ``forced_pass``.
    """
    # DECISION plan-2026-11-05T134105-5ed71cf2/D-009
    # Round record + forced reason: PRE_TRANSITION on judge, NOT
    # CONTEXT_UPDATE. Core skips every CONTEXT_UPDATE handler on a turn
    # that committed no extracted key (a judge that returns null for the
    # verdict and consensus), which left the round uncounted and the loop
    # to the generic limiter (F-AM-01). The judge always transitions (its
    # p300 fallback edge), so this runs on every judge exit. It runs after
    # the gate, so it cannot route: the cap is the judge -> conclude
    # edge's own `current_round >= max_rounds` disjunct, and this handler
    # records the round and the forced reason. It runs before the
    # iteration limiter (lower priority number) so a cap on the same turn
    # is recorded as forced_pass, as the CONTEXT_UPDATE handler did.
    # Do NOT move this back to CONTEXT_UPDATE or key it on extraction.
    num_rounds = require_int_limit(
        "max_rounds",
        ctx.options.get("max_rounds", Defaults.MAX_DEBATE_ROUNDS),
        minimum=1,
    )
    return [
        create_handler(HandlerNames.DEBATE_JUDGE)
        .with_priority(_priority(binding, HandlerPriorities.ITERATION_LIMITER - 1))
        .at(HandlerTiming.PRE_TRANSITION)
        .on_state(*_states(binding, (DebateStates.JUDGE,)))
        .do(make_judge_handler(num_rounds))
    ]


def debate_consensus_reset(
    ctx: HandlerContext, binding: HandlerBinding | None = None
) -> list[BaseHandler]:
    """``agents.debate_consensus_reset``: clears the round keys and advances ``current_round`` on entry to ``propose``."""
    return _on_entry(
        HandlerNames.DEBATE_CONSENSUS_RESET,
        binding,
        DebateStates.PROPOSE,
        HandlerPriorities.TOOL_EXECUTOR,
        make_propose_entry_handler(_keys(binding, ROUND_KEYS)),
    )


def debate_iteration_check(
    ctx: HandlerContext, binding: HandlerBinding | None = None
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """The closure ``agents.debate_limiter`` registers: ``iteration_check`` with the forced stop recorded."""
    return record_forced_consensus(iteration_check(ctx, binding))


def debate_limiter(
    ctx: HandlerContext, binding: HandlerBinding | None = None
) -> list[BaseHandler]:
    """``agents.debate_limiter``: the iteration limiter whose forced consensus records its reason (``debate_iteration_check``)."""
    return [
        create_handler(HandlerNames.ITERATION_LIMITER)
        .with_priority(_priority(binding, HandlerPriorities.ITERATION_LIMITER))
        .at(HandlerTiming.PRE_TRANSITION)
        .do(debate_iteration_check(ctx, binding))
    ]


def evalopt_evaluator(
    ctx: HandlerContext, binding: HandlerBinding | None = None
) -> list[BaseHandler]:
    """``agents.evalopt_evaluator``: runs the host's ``evaluation_fn`` on entry to ``evaluate`` and records the verdict.

    Reads the required callable ``evaluation_fn`` (``ctx.callables``) and the option ``max_refinements`` (at
    least 0), the default for a context that carries no ``_max_refinements``. ``AgentError`` without the callable.
    """
    limit = require_int_limit(
        "max_refinements",
        ctx.options.get("max_refinements", Defaults.MAX_REFINEMENTS),
        minimum=0,
    )
    evaluation_fn = ctx.callables.get("evaluation_fn")
    if not callable(evaluation_fn):
        raise AgentError("agents.evalopt_evaluator needs the callable 'evaluation_fn'")
    return _on_entry(
        HandlerNames.EVAL_OPT_EVALUATOR,
        binding,
        EvalOptStates.EVALUATE,
        HandlerPriorities.TOOL_EXECUTOR,
        lambda context: evaluate_output(context, evaluation_fn, limit),
    )


def _redraft() -> tuple[
    Callable[[dict[str, Any]], dict[str, Any]],
    Callable[[dict[str, Any]], dict[str, Any]],
]:
    """The ``refine`` (entry, exit) pair over ``generated_output`` and ``previous_output``."""
    # DECISION plan-2026-09-24T045559-3e4eb3e5/D-013: refine must re-extract
    # generated_output (core skips a set key), so entering refine moves
    # the draft to previous_output and leaving it restores the draft if
    # none was produced. Do NOT drop the stash: refine's prompt reads it.
    # The evaluation keys need no clearing: the evaluator rewrites them.
    return make_redraft_handlers(
        ContextKeys.GENERATED_OUTPUT, ContextKeys.PREVIOUS_OUTPUT
    )


def refine_entry(
    ctx: HandlerContext, binding: HandlerBinding | None = None
) -> list[BaseHandler]:
    """``agents.refine_entry``: on entry to ``refine`` the draft moves to ``previous_output`` so it is extracted again."""
    return _on_entry(
        HandlerNames.EVAL_OPT_REFINE_ENTRY,
        binding,
        EvalOptStates.REFINE,
        HandlerPriorities.TOOL_EXECUTOR,
        _redraft()[0],
    )


def refine_exit(
    ctx: HandlerContext, binding: HandlerBinding | None = None
) -> list[BaseHandler]:
    """``agents.refine_exit``: on exit from ``refine`` the stashed draft is restored when none was produced."""
    return [
        create_handler(HandlerNames.EVAL_OPT_REFINE_EXIT)
        .with_priority(_priority(binding, HandlerPriorities.TOOL_EXECUTOR))
        .on_state_exit(*_states(binding, (EvalOptStates.REFINE,)))
        .do(_redraft()[1])
    ]
