"""The Plan-and-Execute handler bodies (plan a555dcfe step 15; moved from ``plan_execute.py``).

Interface contract: each ``make_*`` returns a ``(context) -> updates`` function that the library
factory of the same name wraps in a handler (``agents.plan_step_executor``, ``agents.plan_step_checker``
and ``agents.plan_replan_counter``). ``bounded_plan(value) -> list`` is the one rule for ``plan_steps``
the three share: a list is kept (cut to ``Defaults.MAX_PLAN_STEPS`` with a WARNING); a non-blank string
is ONE step, never a sequence of characters; any other value is an empty plan (a WARNING unless None).
``make_result_checker(max_replans)`` reads the ``ToolRunStatus`` the tool executor wrote on the same
entry. Never raises.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fsm_llm.agents.constants import (
    ContextKeys,
    Defaults,
    LogMessages,
    StopReason,
    ToolRunStatus,
)
from fsm_llm.logging import logger

__all__ = [
    "bounded_plan",
    "make_replan_handler",
    "make_result_checker",
    "make_step_tracker",
]

_Body = Callable[[dict[str, Any]], dict[str, Any]]


def bounded_plan(value: Any) -> list[Any]:
    """``plan_steps`` as a list of at most ``Defaults.MAX_PLAN_STEPS`` steps (see the module docstring)."""
    if isinstance(value, list):
        plan = value
    elif isinstance(value, str):
        plan = [value.strip()] if value.strip() else []
        logger.warning("plan_steps is a string, not a list: treated as one step")
    else:
        if value is not None:
            logger.warning(
                f"plan_steps is a {type(value).__name__}, not a list: ignored"
            )
        plan = []
    limit = Defaults.MAX_PLAN_STEPS
    if len(plan) > limit:
        logger.warning(f"Plan of {len(plan)} steps capped at {limit} steps")
        plan = plan[:limit]
    return plan


def make_step_tracker() -> _Body:
    """The execute_step entry handler: bound the plan and describe the current step."""

    def track_step(context: dict[str, Any]) -> dict[str, Any]:
        raw = context.get(ContextKeys.PLAN_STEPS, [])
        plan_steps = bounded_plan(raw)
        updates: dict[str, Any] = {}
        if plan_steps != raw:
            updates[ContextKeys.PLAN_STEPS] = plan_steps
        current_index = context.get(ContextKeys.CURRENT_STEP_INDEX, 0)
        if current_index < len(plan_steps):
            step_desc = plan_steps[current_index]
            total = len(plan_steps)
            logger.info(
                LogMessages.PLAN_STEP.format(
                    current=current_index + 1,
                    total=total,
                    description=str(step_desc)[:80],
                )
            )
            updates[ContextKeys.CURRENT_STEP_DESCRIPTION] = (
                f"Step {current_index + 1}/{total}: {step_desc}"
            )
        return updates

    return track_step


def make_result_checker(max_replans: int) -> _Body:
    """The check_result entry handler: record the step and decide ``step_failed`` from the tool status."""

    def check_result(context: dict[str, Any]) -> dict[str, Any]:
        plan_steps = bounded_plan(context.get(ContextKeys.PLAN_STEPS, []))
        current_index = context.get(ContextKeys.CURRENT_STEP_INDEX, 0)
        step_results = list(context.get(ContextKeys.STEP_RESULTS) or [])

        # DECISION plan_2026-05-31_f08da86d/D-001 [STALE]: tie the per-entry
        # `success` flag to a REAL tool execution for this step. A weak
        # model that NARRATES a step (zero tool calls) must not produce a
        # success=True entry that passes _has_execution_evidence. The
        # execute_tool handler runs just before this one on CHECK_RESULT
        # entry (D-024) and writes TOOL_STATUS "success"/"failed" ONLY when
        # a tool genuinely ran (else "skipped"/"rejected"). Do NOT edit
        # _has_execution_evidence instead (see decisions.md D-001).
        status = context.get(ContextKeys.TOOL_STATUS)
        tool_ran = status in ToolRunStatus.RAN
        step_failed = status == ToolRunStatus.FAILED
        # A tool step records the tool observation, not the pre-tool note.
        observation = context.get(ContextKeys.TOOL_RESULT) if tool_ran else None
        step_result = observation or context.get(ContextKeys.STEP_RESULT)
        if step_result:
            step_results.append(
                {
                    "step_index": current_index,
                    "result": str(step_result),
                    "success": status == ToolRunStatus.SUCCESS,
                }
            )

        updates: dict[str, Any] = {
            ContextKeys.STEP_RESULTS: step_results,
            ContextKeys.STEP_FAILED: step_failed,
        }
        # DECISION plan-2026-10-01T093600-944e2692/D-043: a timed-out step
        # (outcome unknown: it may still run and take effect) ends the
        # plan and goes to synthesis. Do NOT treat it as failed (a replan
        # would run the same side effect again) nor as done (the next
        # step would build on an outcome nobody knows).
        # DECISION plan-2026-10-01T093600-944e2692/D-047: the plan was cut
        # short, so the run did not reach its goal: record NO_RESULT (read
        # by `_run_outcome`, success=False). Do NOT leave it unrecorded
        # (an earlier successful step read as evidence, success=True) and
        # do NOT record MAX_ITERATIONS or STALLED (no budget ran out).
        if status == ToolRunStatus.UNKNOWN:
            logger.warning(
                f"Step {current_index + 1} timed out with an unknown "
                "outcome: synthesizing the results so far"
            )
            updates[ContextKeys.ALL_STEPS_COMPLETE] = True
            updates[ContextKeys.FORCED_STOP_REASON] = StopReason.NO_RESULT.value
        elif step_failed:
            if context.get(ContextKeys.REPLAN_COUNT, 0) >= max_replans:
                logger.warning(
                    f"Step {current_index + 1} failed with {max_replans} "
                    "replans used: synthesizing the results so far"
                )
                updates[ContextKeys.ALL_STEPS_COMPLETE] = True
                # DECISION plan-2026-11-05T134105-5ed71cf2/D-011
                # Same seam as the UNKNOWN branch (D-047): the cap cuts the
                # plan short, so the run did not reach its goal. Do NOT
                # leave it unrecorded (an earlier successful step reads as
                # evidence and the run ships success=True).
                updates[ContextKeys.FORCED_STOP_REASON] = StopReason.NO_RESULT.value
        else:
            next_index = current_index + 1
            updates[ContextKeys.CURRENT_STEP_INDEX] = next_index
            if next_index >= len(plan_steps):
                updates[ContextKeys.ALL_STEPS_COMPLETE] = True
        return updates

    return check_result


def make_replan_handler() -> _Body:
    """The replan entry handler: count the replan and reopen ``plan_steps``.

    The cap is enforced by the checker, so the Nth replan still runs. The ``[]`` reads as unset for
    an agent FSM (D-046), so replan extracts a new plan, which restarts at step 1; the old plan is
    stashed for the replan prompt.
    """

    def handle_replan(context: dict[str, Any]) -> dict[str, Any]:
        return {
            ContextKeys.REPLAN_COUNT: context.get(ContextKeys.REPLAN_COUNT, 0) + 1,
            ContextKeys.PREVIOUS_PLAN_STEPS: bounded_plan(
                context.get(ContextKeys.PLAN_STEPS, [])
            ),
            ContextKeys.PLAN_STEPS: [],
            ContextKeys.CURRENT_STEP_INDEX: 0,
            ContextKeys.STEP_FAILED: False,
        }

    return handle_replan
