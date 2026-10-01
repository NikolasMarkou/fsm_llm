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

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fsm_llm import API
from fsm_llm.logging import logger

from .base import BaseAgent, caller_prompt_keys
from .constants import (
    FRAMEWORK_ONLY_KEYS,
    REACT_THINK_FRESH_KEYS,
    AgentStates,
    ContextKeys,
    Defaults,
    ToolRunStatus,
)
from .definitions import AgentConfig, AgentResult, AgentStep, ToolCall
from .exceptions import AgentError
from .fsm_definitions import _conclude_on_evidence_logic, _typed_field_extraction
from .handlers import (
    ThinkTurnLimiter,
    forced_stop_skip,
    next_step_number,
    with_feedback,
)
from .tools import ToolRegistry, normalize_tool_input, redact_secret_entries
from .truncation import smart_truncate

TOOL_CALLS_KEY = "tool_calls"


def _build_parallel_think_instructions(
    registry: ToolRegistry, task_description: str
) -> str:
    """Per-field instructions for the parallel-think ``tool_calls`` list."""
    return (
        f"You are solving this task: {task_description}\n\n"
        f"{registry.to_prompt_description()}\n\n"
        "Decide which tools to call NEXT. You may call SEVERAL independent tools "
        f"at once to work in parallel. '{TOOL_CALLS_KEY}' is a JSON list of tool "
        "calls, each an object with 'tool_name' (one of the available tools) "
        "and 'tool_input' (an object of arguments). Use an empty list if no "
        "tool is needed. Only batch tools whose inputs do not depend on each "
        "other's results."
    )


def build_parallel_react_fsm(
    registry: ToolRegistry,
    task_description: str = "",
    output_schema: type | None = None,
    context_keys: Sequence[str] = (),
) -> dict[str, Any]:
    """Build the think -> act(parallel) -> conclude FSM definition.

    ``think`` extracts only typed per-field values (``tool_calls`` list,
    ``should_terminate`` bool; D-009 of plan 06a5ec0a);
    their prompts list ``task``, ``observations``, ``agent_feedback`` and
    ``context_keys``, never ``agent_trace``.
    """
    from .prompts import (
        build_conclude_response_instructions,
        build_think_terminate_instructions,
    )

    extra = (ContextKeys.AGENT_FEEDBACK, *context_keys)

    persona = (
        "You are a methodical AI agent that solves tasks by using tools. "
        "You can call multiple independent tools at once to work in parallel. "
        "Review previous observations before deciding. Terminate when you have "
        "enough information to answer."
    )

    think_state: dict[str, Any] = {
        "id": AgentStates.THINK,
        "description": "Reason about the task and select one or more tools to run",
        "purpose": "Decide the next (possibly parallel) batch of tool calls",
        "required_context_keys": [TOOL_CALLS_KEY, "should_terminate"],
        "extraction_instructions": "",
        "field_extractions": [
            _typed_field_extraction(
                TOOL_CALLS_KEY,
                "list",
                _build_parallel_think_instructions(registry, task_description),
                extra_context_keys=extra,
            ),
            _typed_field_extraction(
                ContextKeys.SHOULD_TERMINATE,
                "bool",
                build_think_terminate_instructions(),
                extra_context_keys=extra,
                required=False,
            ),
        ],
        "response_instructions": "",
        "transitions": [
            {
                "target_state": AgentStates.CONCLUDE,
                "description": "Task can be answered with current observations",
                "priority": 10,
                "conditions": [
                    {
                        # DECISION plan-2026-09-24T091842-c1d5bfbc/D-008
                        # Do NOT revert to should_terminate alone: a turn-1
                        # terminate with no batch would answer from memory.
                        # See _conclude_on_evidence_logic.
                        "description": (
                            "Agent decided to terminate AND a tool has run "
                            "or termination is forced"
                        ),
                        "logic": _conclude_on_evidence_logic(),
                    }
                ],
            },
            # DECISION plan-2026-09-24T045559-3e4eb3e5/D-002
            # Unconditional lowest-priority fallback. Do NOT gate this edge on the
            # tool selection: a gated edge BLOCKS `think` on a null/unknown tool, and
            # no PRE_TRANSITION or `act`-entry handler runs on a BLOCKED turn, so the
            # run burns the 3x loop ceiling. `act` handles an empty batch.
            {
                "target_state": AgentStates.ACT,
                "description": "Execute the selected tool batch",
                "priority": 300,
            },
        ],
    }

    states: dict[str, Any] = {
        AgentStates.THINK: think_state,
        AgentStates.ACT: {
            "id": AgentStates.ACT,
            "description": "Execute the selected tools concurrently and observe",
            "purpose": "Run the tool batch and record observations",
            "response_instructions": "",
            "transitions": [
                {
                    "target_state": AgentStates.CONCLUDE,
                    "description": "Terminate when framework signals completion",
                    "priority": 1,
                    "conditions": [
                        {
                            "description": (
                                "Framework/agent decided to terminate AND a tool "
                                "has run or termination is forced"
                            ),
                            "logic": _conclude_on_evidence_logic(),
                        }
                    ],
                },
                {
                    "target_state": AgentStates.THINK,
                    "description": "Return to thinking with new observations",
                    "priority": 900,
                },
            ],
        },
        AgentStates.CONCLUDE: {
            "id": AgentStates.CONCLUDE,
            "description": "Formulate and present the final answer",
            "purpose": "Synthesize all observations into a complete answer",
            "required_context_keys": (
                list(output_schema.model_fields.keys())
                if output_schema and hasattr(output_schema, "model_fields")
                else []
            ),
            "response_instructions": build_conclude_response_instructions(),
            "transitions": [],
        },
    }

    return {
        "name": "ParallelReactAgent",
        "description": "ReAct agent with parallel tool dispatch",
        "initial_state": AgentStates.THINK,
        "persona": persona,
        "states": states,
        # D-051 of plan 06a5ec0a, as every _finalize_fsm builder.
        "handler_only_keys": list(FRAMEWORK_ONLY_KEYS),
    }


class ParallelReactAgent(BaseAgent):
    """ReAct agent that dispatches multiple tool calls per step concurrently."""

    def __init__(
        self,
        tools: ToolRegistry,
        config: AgentConfig | None = None,
        max_parallel: int = 4,
        **api_kwargs: Any,
    ) -> None:
        if len(tools) == 0:
            raise AgentError("Cannot create agent with empty tool registry")
        if max_parallel < 1:
            raise AgentError("max_parallel must be >= 1")
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
        context = self._init_context(
            task,
            initial_context,
            extra={
                ContextKeys.OBSERVATIONS: [],
                "_max_iterations": self.config.max_iterations,
            },
        )
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
        if handlers is None:
            raise AgentError(
                "ParallelReactAgent._register_handlers called without a "
                "handlers instance — this is a programming error, not a "
                "runtime condition; run() must always pass one."
            )
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
            return {**forced, TOOL_CALLS_KEY: None}
        calls = self._normalize_calls(context.get(TOOL_CALLS_KEY))
        observations = context.get(ContextKeys.OBSERVATIONS, []) or []
        if not isinstance(observations, list):
            observations = []
        trace = context.get(ContextKeys.AGENT_TRACE, []) or []
        if not isinstance(trace, list):
            trace = []

        if not calls:
            return with_feedback(
                "No tools were selected.",
                {ContextKeys.TOOL_STATUS: "skipped", TOOL_CALLS_KEY: None},
            )

        # Submit in order; gather results in order for deterministic observations.
        workers = min(self.max_parallel, len(calls))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [(c, pool.submit(self.tools.execute, c)) for c in calls]
            results = [(c, f.result()) for c, f in futures]

        reasoning = str(context.get(ContextKeys.REASONING) or "")
        any_success = False
        for call, result in results:
            any_success = any_success or result.success
            observation = result.observation
            step_num = next_step_number(trace)
            # plan-2026-09-29T103145-06a5ec0a/D-016: show a redacted copy.
            shown_input = redact_secret_entries(call.parameters)
            entry = smart_truncate(
                f"[Step {step_num}] Tool: {call.tool_name} | "
                f"Input: {shown_input} | Result: {observation}",
                Defaults.MAX_OBSERVATION_LENGTH,
            )
            observations.append(entry)
            trace_step = AgentStep(
                iteration=step_num,
                thought=reasoning,
                action=f"{call.tool_name}({shown_input})",
                observation=observation,
            ).model_dump(mode="json")
            trace_step["tool_input"] = shown_input
            trace_step[ContextKeys.TOOL_STATUS] = result.status
            trace.append(trace_step)

        if len(observations) > Defaults.MAX_OBSERVATIONS:
            observations = observations[-Defaults.MAX_OBSERVATIONS :]

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
            ContextKeys.OBSERVATIONS: observations,
            ContextKeys.OBSERVATION_COUNT: len(observations),
            ContextKeys.AGENT_TRACE: trace,
            TOOL_CALLS_KEY: None,
            ContextKeys.SHOULD_TERMINATE: None,
        }
