"""
ParallelReactAgent — a ReAct variant that dispatches MULTIPLE tool calls per
step, concurrently.

The stock :class:`ReactAgent` FSM is strictly one-tool-per-turn (think selects a
single ``tool_name``, act runs it, loop). When a task needs several independent
lookups (search three sources, fetch two files), that serializes avoidable
latency. ``ParallelReactAgent`` extracts a *list* of tool calls in the think
state and runs them together in a thread pool.

Fully additive: a new self-contained FSM + ``BaseAgent`` subclass. The stock
ReAct path is untouched. Tools must be thread-safe to benefit (the same
requirement as any concurrent dispatch).

Example::

    from fsm_llm.agents import AgentConfig, ToolRegistry
    from fsm_llm.agents.parallel_react import ParallelReactAgent

    agent = ParallelReactAgent(tools=registry, config=AgentConfig(model=model),
                               max_parallel=4)
    result = agent.run("Compare the weather in Paris, Tokyo and Cairo.")
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, ClassVar

from fsm_llm import API
from fsm_llm.logging import logger

from .base import BaseAgent, caller_prompt_keys
from .constants import (
    REACT_THINK_FRESH_KEYS,
    AgentStates,
    ContextKeys,
    Defaults,
    LogMessages,
    ToolRunStatus,
)
from .definitions import AgentConfig, AgentResult, ToolCall
from .exceptions import require_int_limit
from .fsm_definitions import build_parallel_react_fsm
from .handlers import (
    ThinkTurnLimiter,
    call_label,
    forced_stop_skip,
    record_tool_step,
    with_feedback,
)
from .tools import ToolRegistry, normalize_tool_input, redact_secret_entries


class ParallelReactAgent(BaseAgent):
    """ReAct agent that dispatches multiple tool calls per step concurrently."""

    # Run outputs caller context may not seed (D-052 of plan 06a5ec0a):
    # `tool_calls` is the list `act` executes, and a seeded value is a set key
    # (skip-if-set), so the think extraction would be skipped and the caller's
    # calls would run with no model choice, also through AgentServer.
    _run_output_keys: ClassVar[frozenset[str]] = frozenset({ContextKeys.TOOL_CALLS})

    def __init__(
        self,
        tools: ToolRegistry,
        config: AgentConfig | None = None,
        max_parallel: int = 4,
        **api_kwargs: Any,
    ) -> None:
        self._require_tools(tools)
        max_parallel = require_int_limit("max_parallel", max_parallel, minimum=1)
        super().__init__(config, **api_kwargs)
        self.tools = tools
        self._refuse_flagged_tools()
        self.max_parallel = max_parallel
        # DECISION plan-2026-09-12T065608-089d0ec7/D-014
        # No `self._handlers` here (and none is ever assigned anywhere in this
        # class) — see the identical note in react.py's ReactAgent.__init__.
        # A per-instance AgentHandlers shared across concurrent run() calls
        # is the D-004/D-014 race; each call builds and uses its own
        # call-LOCAL AgentHandlers (see run() below). See decisions.md D-014.

        logger.info(
            LogMessages.AGENT_STARTED.format(
                tool_count=len(tools), model=self.config.model
            )
        )

    def run(
        self,
        task: str,
        initial_context: dict[str, Any] | None = None,
    ) -> AgentResult:
        # DECISION plan-2026-09-12T065608-089d0ec7/D-014 (supersedes D-004)
        # A call-LOCAL AgentHandlers, threaded explicitly into
        # `_register_handlers` via `_standard_run`'s `handlers=` parameter —
        # see the identical, fuller note in react.py's ReactAgent.run(). Two
        # overlapping run() calls on the SAME agent (AgentServer's
        # asyncio.to_thread dispatch) must never share one AgentHandlers
        # instance; D-004's `self._handlers = AgentHandlers(...)` reassignment
        # did not achieve that (both calls could still read back the SAME,
        # most-recently-assigned instance). Do NOT reintroduce
        # `self._handlers = AgentHandlers(...)` here. See decisions.md D-014.
        self._refuse_flagged_tools()
        # plan-2026-10-01T093600-944e2692/D-042: the limiter only. This agent
        # calls `execute(call)` without `gated`, so an older registry
        # override still runs here.
        handlers = ThinkTurnLimiter()
        fsm_def = build_parallel_react_fsm(
            self.tools,
            task_description=task[: Defaults.MAX_TASK_PREVIEW_LENGTH],
            output_schema=self.config.output_schema,
            context_keys=caller_prompt_keys(initial_context),
        )
        context = self._react_context(task, initial_context)
        return self._standard_run(
            task, fsm_def, context, "parallel_react", handlers=handlers
        )

    # DECISION plan-2026-09-12T065608-089d0ec7/D-014
    # See the identical note on ReactAgent._register_handlers (react.py) —
    # `handlers` is required in practice; the `| None = None` default only
    # satisfies BaseAgent's narrower abstract signature. Do NOT fall back to
    # `self._handlers` if `handlers` is None. See decisions.md D-014.
    def _register_handlers(
        self, api: API, handlers: ThinkTurnLimiter | None = None
    ) -> None:
        handlers = self._require_handlers(handlers)
        self._register_tool_executor(api, AgentStates.ACT, self._dispatch_parallel)
        self._register_iteration_limiter(api, handlers.check_iteration_limit)
        self._register_think_loop_handlers(api, REACT_THINK_FRESH_KEYS)

    def _normalize_calls(self, raw: Any) -> list[ToolCall]:
        """Coerce extracted ``tool_calls`` into a list of ToolCall objects."""
        if not isinstance(raw, list):
            return []
        calls: list[ToolCall] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            name = item.get("tool_name") or item.get("name")
            if not name or name == ContextKeys.NO_TOOL:
                continue
            params = normalize_tool_input(
                item.get("tool_input") or item.get("input") or {}
            )
            calls.append(ToolCall(tool_name=str(name), parameters=params))
        return calls

    def _dispatch_parallel(self, context: dict[str, Any]) -> dict[str, Any]:
        """Execute the extracted tool batch concurrently; record observations."""
        forced = forced_stop_skip(context)
        if forced is not None:  # LOOP-05: no tool after the forced stop
            return {**forced, ContextKeys.TOOL_CALLS: None}
        calls = self._normalize_calls(context.get(ContextKeys.TOOL_CALLS))
        if not calls:
            return with_feedback(
                "No tools were selected.",
                {
                    ContextKeys.TOOL_STATUS: ToolRunStatus.SKIPPED,
                    ContextKeys.TOOL_CALLS: None,
                },
            )

        # Submit in order; gather results in order for deterministic observations.
        workers = min(self.max_parallel, len(calls))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [(c, pool.submit(self.tools.execute, c)) for c in calls]
            results = [(c, f.result()) for c, f in futures]

        reasoning = str(context.get(ContextKeys.REASONING) or "")
        any_success = False
        recorded: dict[str, Any] = {}
        for call, result in results:
            any_success = any_success or result.success
            recorded = record_tool_step(
                {**context, **recorded},
                tool_name=call.tool_name,
                # plan-2026-09-29T103145-06a5ec0a/D-016: show a redacted copy.
                shown_input=redact_secret_entries(call.parameters),
                observation=result.observation,
                status=result.status,
                thought=reasoning,
                action=call_label(call.tool_name, call.parameters),
            )

        logger.info(
            f"Parallel dispatch executed {len(calls)} tool(s), "
            f"{sum(1 for _, r in results if r.success)} succeeded"
        )

        # plan-2026-10-01T093600-944e2692/D-043: one timed-out call makes the
        # batch's outcome unknown, whatever else succeeded.
        statuses = {result.status for _, result in results}
        if ToolRunStatus.UNKNOWN in statuses:
            batch_status = ToolRunStatus.UNKNOWN
        elif any_success:
            batch_status = ToolRunStatus.SUCCESS
        else:
            batch_status = ToolRunStatus.FAILED
        return {
            ContextKeys.TOOL_RESULT: f"Executed {len(calls)} tool(s).",
            ContextKeys.TOOL_STATUS: batch_status,
            **recorded,
            ContextKeys.TOOL_CALLS: None,
            ContextKeys.SHOULD_TERMINATE: None,
        }
