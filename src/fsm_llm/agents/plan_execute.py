"""
PlanExecuteAgent — Plan-and-Execute agent implementation.

Separates strategic planning from tactical execution:
Plan -> Execute Step -> Check Result -> Synthesize (all done)
                                      -> Replan (step failed)
                                      -> Execute Step (next step)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fsm_llm import API
from fsm_llm.logging import logger

from .base import BaseAgent
from .constants import (
    ContextKeys,
    Defaults,
    HandlerNames,
    HandlerPriorities,
    LogMessages,
    PlanExecuteStates,
)
from .definitions import AgentConfig, AgentResult
from .fsm_definitions import build_plan_execute_fsm
from .handlers import AgentHandlers, make_fresh_keys_handler, make_iteration_limiter
from .tools import ToolRegistry

# Tool statuses that mean a tool really ran for the step (AgentHandlers).
_TOOL_RAN = frozenset({"success", "failed"})


def _bounded_plan(value: Any) -> list[Any]:
    """``plan_steps`` as a list of at most ``Defaults.MAX_PLAN_STEPS`` steps.

    Contract: a list is kept (truncated to the cap, with a WARNING); a
    non-blank string is ONE step, never a sequence of characters; any other
    value (None included) is an empty plan, with a WARNING unless None. Shared
    by the step tracker, the step checker and the replan stash. Never raises.
    """
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


class PlanExecuteAgent(BaseAgent):
    """
    Plan-and-Execute agent that separates planning from execution.

    First creates a plan, then executes each step sequentially. If a step
    fails, the agent can revise the remaining plan. When all steps are
    complete, results are synthesized into a final answer.

    Usage::

        agent = PlanExecuteAgent(tools=registry)
        result = agent.run("Compare the populations of France and Germany")
        print(result.answer)
    """

    def __init__(
        self,
        tools: ToolRegistry | None = None,
        config: AgentConfig | None = None,
        max_replans: int = Defaults.MAX_REPLANS,
        **api_kwargs: Any,
    ) -> None:
        """
        Initialize a Plan-and-Execute agent.

        :param tools: Optional tool registry (executor may use LLM only)
        :param config: Agent configuration (defaults to AgentConfig())
        :param max_replans: Maximum number of replan cycles
        :param api_kwargs: Additional kwargs passed to fsm_llm.API
        """
        super().__init__(config, **api_kwargs)
        self.tools = tools
        self._refuse_flagged_tools()
        self.max_replans = max_replans
        # DECISION plan-2026-09-12T135914-45a654de/D-012
        # No `self._handlers` here (matches react.py's D-014 pattern) — a
        # per-instance AgentHandlers shared across concurrent run() calls is
        # the D-004/D-014 race. Each call builds its own call-LOCAL
        # AgentHandlers (see run() below), still legitimately `None` in
        # tool-less mode. Do NOT reintroduce this assignment — see
        # decisions.md D-012.

        tool_count = len(tools) if tools is not None else 0
        logger.info(
            LogMessages.AGENT_STARTED.format(
                tool_count=tool_count, model=self.config.model
            )
        )

    def run(
        self,
        task: str,
        initial_context: dict[str, Any] | None = None,
    ) -> AgentResult:
        """
        Run the agent on a task.

        :param task: The task/question for the agent to solve
        :param initial_context: Optional initial context data
        :return: AgentResult with answer, trace, and metadata
        """
        self._refuse_flagged_tools()
        fsm_def = build_plan_execute_fsm(self.tools, task_description=task)

        # DECISION plan-2026-09-12T135914-45a654de/D-012
        # A call-LOCAL AgentHandlers (or None in tool-less mode), threaded
        # explicitly into `_register_handlers` via `_standard_run`'s
        # `handlers=` parameter. A fresh instance needs no `.reset()`. Do
        # NOT reintroduce `self._handlers = ...` — see decisions.md D-012.
        handlers = AgentHandlers(self.tools) if self.tools is not None else None

        # Build initial context
        context = self._init_context(
            task,
            initial_context,
            extra={
                ContextKeys.OBSERVATIONS: [],
                # DECISION plan-2026-09-19T175721-21cd7f8e/D-046: the `[]` seed
                # is the ONLY exit of the `plan` state. Its one transition is
                # `has_context: plan_steps`, which is key EXISTENCE, so when the
                # model cannot produce a plan an unseeded run loops on `plan`
                # and raises BudgetExhaustedError after 36 wasted asks instead of
                # ending as `success=False`. Do NOT remove it (iteration 3 did,
                # D-036). The seed is still extractable: the pipeline's
                # skip-if-set filter reads an empty container as unset for
                # agent-managed FSMs (D-046 in pipeline.py), so the plan is
                # asked for. Readers keep using `context.get(PLAN_STEPS, [])`.
                ContextKeys.PLAN_STEPS: [],
                ContextKeys.CURRENT_STEP_INDEX: 0,
                ContextKeys.STEP_RESULTS: [],
                ContextKeys.ALL_STEPS_COMPLETE: False,
                ContextKeys.STEP_FAILED: False,
                "_max_iterations": self.config.max_iterations,
                "_replan_count": 0,
            },
        )

        # DECISION plan_2026-05-31_cb91a9d5/D-001 [STALE]: require ≥1 executed step —
        # a run that never left `plan` (weak decomposition) must not pass as
        # success on synthesis prose alone.
        return self._standard_run(
            task,
            fsm_def,
            context,
            "plan_execute",
            handlers=handlers,
            execution_evidence_keys=[ContextKeys.STEP_RESULTS],
        )

    # DECISION plan-2026-09-12T135914-45a654de/D-012
    # `handlers` is legitimately `None` in tool-less mode — keep the
    # `if handlers is not None:` conditional guard. Do NOT copy react.py's
    # hard `AgentError`-on-`None` raise here (see decisions.md D-012 /
    # plan.md invariant 9 — that guard is only correct where tools are
    # mandatory).
    def _register_handlers(
        self, api: API, handlers: AgentHandlers | None = None
    ) -> None:
        """Register agent handlers with the API."""
        # DECISION plan-2026-09-24T091842-c1d5bfbc/D-024
        # Run the tool on CHECK_RESULT entry, before the checker (priority +1):
        # the step's tool selection is extracted during the execute_step turn.
        # Do NOT move it back to EXECUTE_STEP entry (step 1 never ran a tool,
        # step k ran step k-1's input) and do NOT touch the shared compactor.
        if handlers is not None:
            self._register_tool_executor(
                api, PlanExecuteStates.CHECK_RESULT, handlers.execute_tool
            )

        # Iteration limiter
        self._register_iteration_limiter(api, self._make_iteration_limiter())

        # Plan step tracker
        api.register_handler(
            api.create_handler(HandlerNames.PLAN_STEP_EXECUTOR)
            .with_priority(HandlerPriorities.TOOL_EXECUTOR)
            .on_state_entry(PlanExecuteStates.EXECUTE_STEP)
            .do(self._make_step_tracker())
        )
        # execute_step produces step_result: clear it on entry so each step
        # extracts its own (core extracts a key only while it is unset).
        api.register_handler(
            api.create_handler(HandlerNames.PLAN_STEP_FRESH_KEYS)
            .with_priority(HandlerPriorities.TOOL_EXECUTOR)
            .on_state_entry(PlanExecuteStates.EXECUTE_STEP)
            .do(make_fresh_keys_handler([ContextKeys.STEP_RESULT]))
        )

        # DECISION plan-2026-09-29T103145-06a5ec0a/D-032
        # step_failed is decided on check_result ENTRY, from the status of the
        # tool that just ran (the executor above, D-024 order), and routes the
        # check_result turn that follows. Do NOT move it to PRE_TRANSITION on
        # check_result (core picks the edge before PRE_TRANSITION runs, D-031)
        # and do NOT let the model extract step_failed (a False seed is never
        # re-extracted, and a guessed verdict is not a tool outcome). The
        # max_replans check lives here too, so a failure past the cap routes to
        # synthesize instead of entering replan. See decisions.md D-032.
        api.register_handler(
            api.create_handler(HandlerNames.PLAN_STEP_CHECKER)
            .with_priority(HandlerPriorities.TOOL_EXECUTOR + 1)
            .on_state_entry(PlanExecuteStates.CHECK_RESULT)
            .do(self._make_result_checker())
        )

        # Replan: count it and reopen the plan for re-extraction
        api.register_handler(
            api.create_handler(HandlerNames.PLAN_REPLANNER)
            .with_priority(HandlerPriorities.TOOL_EXECUTOR)
            .on_state_entry(PlanExecuteStates.REPLAN)
            .do(self._make_replan_handler())
        )

    def _make_iteration_limiter(self) -> Callable[[dict[str, Any]], dict[str, Any]]:
        """Create the iteration limiter handler (shared rule, see handlers)."""
        return make_iteration_limiter(
            self.config.max_iterations,
            {
                ContextKeys.MAX_ITERATIONS_REACHED: True,
                ContextKeys.SHOULD_TERMINATE: True,
            },
            context_max_key="_max_iterations",
        )

    def _make_step_tracker(self) -> Callable[[dict[str, Any]], dict[str, Any]]:
        """Create the execute_step entry handler that bounds the plan and
        describes the current step."""

        def track_step(context: dict[str, Any]) -> dict[str, Any]:
            raw = context.get(ContextKeys.PLAN_STEPS, [])
            plan_steps = _bounded_plan(raw)
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

    def _make_result_checker(self) -> Callable[[dict[str, Any]], dict[str, Any]]:
        """Create the check_result entry handler: record the step and decide
        ``step_failed`` from the tool status (D-032)."""
        max_replans = self.max_replans

        def check_result(context: dict[str, Any]) -> dict[str, Any]:
            plan_steps = _bounded_plan(context.get(ContextKeys.PLAN_STEPS, []))
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
            tool_ran = status in _TOOL_RAN
            step_failed = status == "failed"
            # A tool step records the tool observation, not the pre-tool note.
            observation = context.get(ContextKeys.TOOL_RESULT) if tool_ran else None
            step_result = observation or context.get(ContextKeys.STEP_RESULT)
            if step_result:
                step_results.append(
                    {
                        "step_index": current_index,
                        "result": str(step_result),
                        "success": tool_ran and not step_failed,
                    }
                )

            updates: dict[str, Any] = {
                ContextKeys.STEP_RESULTS: step_results,
                ContextKeys.STEP_FAILED: step_failed,
            }
            if step_failed:
                if context.get("_replan_count", 0) >= max_replans:
                    logger.warning(
                        f"Step {current_index + 1} failed with {max_replans} "
                        "replans used: synthesizing the results so far"
                    )
                    updates[ContextKeys.ALL_STEPS_COMPLETE] = True
            else:
                next_index = current_index + 1
                updates[ContextKeys.CURRENT_STEP_INDEX] = next_index
                if next_index >= len(plan_steps):
                    updates[ContextKeys.ALL_STEPS_COMPLETE] = True
            return updates

        return check_result

    def _make_replan_handler(self) -> Callable[[dict[str, Any]], dict[str, Any]]:
        """Create the replan entry handler.

        It counts the replan (the cap is enforced by the checker, so the Nth
        replan still runs) and reopens ``plan_steps``: the ``[]`` reads as
        unset for an agent FSM (D-046), so replan extracts a new plan, which
        restarts at step 1. The old plan is stashed for the replan prompt.
        """

        def handle_replan(context: dict[str, Any]) -> dict[str, Any]:
            return {
                "_replan_count": context.get("_replan_count", 0) + 1,
                ContextKeys.PREVIOUS_PLAN_STEPS: _bounded_plan(
                    context.get(ContextKeys.PLAN_STEPS, [])
                ),
                ContextKeys.PLAN_STEPS: [],
                ContextKeys.CURRENT_STEP_INDEX: 0,
                ContextKeys.STEP_FAILED: False,
            }

        return handle_replan
