from __future__ import annotations

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

    from fsm_llm_agents import AgentConfig, ToolRegistry
    from fsm_llm_agents.parallel_react import ParallelReactAgent

    agent = ParallelReactAgent(tools=registry, config=AgentConfig(model=model),
                               max_parallel=4)
    result = agent.run("Compare the weather in Paris, Tokyo and Cairo.")
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fsm_llm import API
from fsm_llm.logging import logger

from .base import BaseAgent
from .constants import ContextKeys, Defaults
from .definitions import AgentConfig, AgentResult, AgentStep, ToolCall
from .exceptions import AgentError
from .handlers import AgentHandlers
from .tools import ToolRegistry, normalize_tool_input
from .truncation import smart_truncate

TOOL_CALLS_KEY = "tool_calls"


def _build_parallel_think_instructions(
    registry: ToolRegistry, task_description: str
) -> str:
    """Extraction instructions for the parallel-think state."""
    return (
        f"You are solving this task: {task_description}\n\n"
        f"{registry.to_prompt_description()}\n\n"
        "Decide which tools to call NEXT. You may call SEVERAL independent tools "
        "at once to work in parallel. Extract:\n"
        f"- '{TOOL_CALLS_KEY}': a JSON list of tool calls, each an object with "
        "'tool_name' (one of the available tools) and 'tool_input' (an object of "
        "arguments). Use an empty list if no tool is needed.\n"
        "- 'should_terminate': true ONLY when you have enough information to "
        "answer the task; otherwise false.\n"
        "Only batch tools whose inputs do not depend on each other's results."
    )


def build_parallel_react_fsm(
    registry: ToolRegistry,
    task_description: str = "",
    output_schema: type | None = None,
) -> dict[str, Any]:
    """Build the think -> act(parallel) -> conclude FSM definition."""
    from .prompts import (
        build_conclude_extraction_instructions,
        build_conclude_response_instructions,
    )

    persona = (
        "You are a methodical AI agent that solves tasks by using tools. "
        "You can call multiple independent tools at once to work in parallel. "
        "Review previous observations before deciding. Terminate when you have "
        "enough information to answer."
    )

    think_state: dict[str, Any] = {
        "id": "think",
        "description": "Reason about the task and select one or more tools to run",
        "purpose": "Decide the next (possibly parallel) batch of tool calls",
        "required_context_keys": [TOOL_CALLS_KEY, "should_terminate"],
        "extraction_instructions": _build_parallel_think_instructions(
            registry, task_description
        ),
        "response_instructions": "",
        "transitions": [
            {
                "target_state": "conclude",
                "description": "Task can be answered with current observations",
                "priority": 10,
                "conditions": [
                    {
                        "description": "Agent decided to terminate",
                        "logic": {"==": [{"var": "should_terminate"}, True]},
                    }
                ],
            },
            {
                "target_state": "act",
                "description": "Execute the selected tool batch",
                "priority": 300,
                "conditions": [
                    {
                        "description": "At least one tool was selected",
                        "logic": {"has_context": TOOL_CALLS_KEY},
                    }
                ],
            },
        ],
    }

    states: dict[str, Any] = {
        "think": think_state,
        "act": {
            "id": "act",
            "description": "Execute the selected tools concurrently and observe",
            "purpose": "Run the tool batch and record observations",
            "response_instructions": "",
            "transitions": [
                {
                    "target_state": "conclude",
                    "description": "Terminate when framework signals completion",
                    "priority": 1,
                    "conditions": [
                        {
                            "description": "Framework or agent decided to terminate",
                            "logic": {"==": [{"var": "should_terminate"}, True]},
                        }
                    ],
                },
                {
                    "target_state": "think",
                    "description": "Return to thinking with new observations",
                    "priority": 900,
                },
            ],
        },
        "conclude": {
            "id": "conclude",
            "description": "Formulate and present the final answer",
            "purpose": "Synthesize all observations into a complete answer",
            "required_context_keys": (
                ["final_answer"]
                + (
                    list(output_schema.model_fields.keys())
                    if output_schema and hasattr(output_schema, "model_fields")
                    else []
                )
            ),
            "extraction_instructions": build_conclude_extraction_instructions(
                output_schema
            ),
            "response_instructions": build_conclude_response_instructions(),
            "transitions": [],
        },
    }

    return {
        "name": "ParallelReactAgent",
        "description": "ReAct agent with parallel tool dispatch",
        "initial_state": "think",
        "persona": persona,
        "states": states,
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
        self.max_parallel = max_parallel
        self._handlers = AgentHandlers(tools)

    def run(
        self,
        task: str,
        initial_context: dict[str, Any] | None = None,
    ) -> AgentResult:
        self._handlers.reset()
        fsm_def = build_parallel_react_fsm(
            self.tools,
            task_description=task[: Defaults.MAX_TASK_PREVIEW_LENGTH],
            output_schema=self.config.output_schema,
        )
        context = self._init_context(
            task,
            initial_context,
            extra={
                ContextKeys.OBSERVATIONS: [],
                "_max_iterations": self.config.max_iterations,
            },
        )
        return self._standard_run(task, fsm_def, context, "parallel_react")

    def _register_handlers(self, api: API) -> None:
        self._register_tool_executor(api, "act", self._dispatch_parallel)
        self._register_iteration_limiter(api, self._handlers.check_iteration_limit)

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
        calls = self._normalize_calls(context.get(TOOL_CALLS_KEY))
        observations = context.get(ContextKeys.OBSERVATIONS, []) or []
        if not isinstance(observations, list):
            observations = []
        trace = context.get(ContextKeys.AGENT_TRACE, []) or []
        if not isinstance(trace, list):
            trace = []

        if not calls:
            return {
                ContextKeys.TOOL_RESULT: "No tools were selected.",
                ContextKeys.TOOL_STATUS: "skipped",
                TOOL_CALLS_KEY: None,
            }

        # Submit in order; gather results in order for deterministic observations.
        workers = min(self.max_parallel, len(calls))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [(c, pool.submit(self.tools.execute, c)) for c in calls]
            results = [(c, f.result()) for c, f in futures]

        any_success = False
        for call, result in results:
            any_success = any_success or result.success
            observation = result.summary
            if not result.success:
                observation = f"[TOOL FAILED] {observation}"
            step_num = len(observations) + 1
            entry = smart_truncate(
                f"[Step {step_num}] Tool: {call.tool_name} | "
                f"Input: {call.parameters} | Result: {observation}",
                Defaults.MAX_OBSERVATION_LENGTH,
            )
            observations.append(entry)
            trace_step = AgentStep(
                iteration=step_num,
                thought="",
                action=f"{call.tool_name}({call.parameters})",
                observation=observation,
            ).model_dump(mode="json")
            trace_step["tool_input"] = call.parameters
            trace.append(trace_step)

        if len(observations) > Defaults.MAX_OBSERVATIONS:
            observations = observations[-Defaults.MAX_OBSERVATIONS :]

        logger.info(
            f"Parallel dispatch executed {len(calls)} tool(s), "
            f"{sum(1 for _, r in results if r.success)} succeeded"
        )

        return {
            ContextKeys.TOOL_RESULT: f"Executed {len(calls)} tool(s).",
            ContextKeys.TOOL_STATUS: "success" if any_success else "failed",
            ContextKeys.OBSERVATIONS: observations,
            ContextKeys.OBSERVATION_COUNT: len(observations),
            ContextKeys.AGENT_TRACE: trace,
            TOOL_CALLS_KEY: None,
            ContextKeys.SHOULD_TERMINATE: None,
        }
