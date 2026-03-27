from __future__ import annotations

"""
REWOOAgent — Reasoning WithOut Observation agent implementation.

Uses exactly 2 LLM calls: plan all tool calls upfront (#E1, #E2 refs),
execute them sequentially (no LLM), then synthesize from evidence.
"""

import re
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
    REWOOStates,
)
from .definitions import AgentConfig, AgentResult, ToolCall
from .exceptions import AgentError
from .fsm_definitions import build_rewoo_fsm
from .tools import ToolRegistry


class REWOOAgent(BaseAgent):
    """
    REWOO agent that plans all tool calls upfront then executes them.

    Makes exactly 2 LLM calls: one to plan all tool calls with #E1/#E2
    variable references, one to synthesize the final answer from evidence.

    Usage::

        agent = REWOOAgent(tools=registry)
        result = agent.run("What is the population of France times 2?")
    """

    def __init__(
        self,
        tools: ToolRegistry,
        config: AgentConfig | None = None,
        **api_kwargs: Any,
    ) -> None:
        if len(tools) == 0:
            raise AgentError("Cannot create agent with empty tool registry")

        super().__init__(config, **api_kwargs)
        self.tools = tools

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
        """
        Run the agent on a task.

        :param task: The task/question for the agent to solve
        :param initial_context: Optional initial context data
        :return: AgentResult with answer, trace, and metadata
        """
        fsm_def = build_rewoo_fsm(self.tools, task_description=task[:Defaults.MAX_TASK_PREVIEW_LENGTH])

        context = self._init_context(
            task,
            initial_context,
            extra={
                ContextKeys.EVIDENCE: {},
            },
        )

        return self._standard_run(task, fsm_def, context, "rewoo")

    def _register_handlers(self, api: API) -> None:
        """Register agent handlers with the API."""
        api.register_handler(
            api.create_handler(HandlerNames.REWOO_EXECUTOR)
            .with_priority(HandlerPriorities.TOOL_EXECUTOR)
            .on_state_entry(REWOOStates.EXECUTE_PLANS)
            .do(self._execute_all_plans)
        )
        self._register_iteration_limiter(api, self._check_iteration_limit)

    def _execute_all_plans(self, context: dict[str, Any]) -> dict[str, Any]:
        """Execute all planned tool calls, substituting #EN variable references."""
        plan_blueprint = context.get(ContextKeys.PLAN_BLUEPRINT, [])
        if not isinstance(plan_blueprint, list):
            logger.warning("plan_blueprint is not a list, skipping execution")
            return {ContextKeys.EVIDENCE: {}}

        evidence: dict[str, str] = {}
        trace_entries: list[dict[str, Any]] = list(
            context.get(ContextKeys.AGENT_TRACE, [])
        )

        for step in plan_blueprint:
            if not isinstance(step, dict):
                continue

            plan_id = step.get("plan_id", 0)
            tool_name = step.get("tool_name", "")
            tool_input = step.get("tool_input", {})
            description = step.get("description", "")

            # Substitute #EN references in tool_input
            tool_input = self._substitute_evidence_refs(tool_input, evidence)

            logger.info(
                LogMessages.PLAN_STEP.format(
                    current=plan_id,
                    total=len(plan_blueprint),
                    description=description,
                )
            )

            # Normalize tool_input to dict
            if isinstance(tool_input, str):
                tool_input = {"input": tool_input}
            elif not isinstance(tool_input, dict):
                tool_input = {"input": str(tool_input)}

            # Execute tool
            tool_call = ToolCall(
                tool_name=tool_name,
                parameters=tool_input,
                reasoning=description,
            )
            result = self.tools.execute(tool_call)

            # Store evidence
            evidence_key = f"E{plan_id}"
            evidence[evidence_key] = result.summary

            if result.success:
                logger.info(LogMessages.TOOL_EXECUTED.format(name=tool_name))
            else:
                logger.warning(
                    LogMessages.TOOL_FAILED.format(name=tool_name, error=result.error)
                )

            # Record in trace (include "action" key for cross-agent consistency)
            trace_entries.append(
                {
                    "action": f"{tool_name}({plan_id})",
                    "thought": description,
                    "plan_id": plan_id,
                    "tool_name": tool_name,
                    "tool_input": tool_input,
                    "description": description,
                    "result": result.summary,
                    "success": result.success,
                }
            )

        return {
            ContextKeys.EVIDENCE: evidence,
            ContextKeys.AGENT_TRACE: trace_entries,
        }

    def _substitute_evidence_refs(
        self,
        value: Any,
        evidence: dict[str, str],
    ) -> Any:
        """Recursively substitute #EN references with evidence values."""
        if isinstance(value, str):
            # Replace #E1, #E2, etc. with actual evidence
            def replace_ref(match: re.Match[str]) -> str:
                ref_key = match.group(1)
                if ref_key not in evidence:
                    logger.warning(
                        f"Evidence reference #{ref_key} not found in evidence store"
                    )
                return evidence.get(ref_key, "[unavailable]")

            return re.sub(r"#(E\d+)", replace_ref, value)

        if isinstance(value, dict):
            return {
                k: self._substitute_evidence_refs(v, evidence) for k, v in value.items()
            }

        if isinstance(value, list):
            return [self._substitute_evidence_refs(item, evidence) for item in value]

        return value

    def _check_iteration_limit(self, context: dict[str, Any]) -> dict[str, Any]:
        """Check if the iteration limit has been reached."""
        iteration = context.get(ContextKeys.ITERATION_COUNT, 0) + 1

        if iteration >= self.config.max_iterations:
            return {
                ContextKeys.ITERATION_COUNT: iteration,
                ContextKeys.MAX_ITERATIONS_REACHED: True,
            }

        return {ContextKeys.ITERATION_COUNT: iteration}

    def _build_trace(self, final_context: dict[str, Any], iteration: int) -> Any:
        """Build agent trace from final context with REWOO-specific trace format."""
        from .definitions import AgentTrace

        trace_data = final_context.get(ContextKeys.AGENT_TRACE, [])
        trace = AgentTrace(
            tool_calls=[],
            total_iterations=final_context.get(ContextKeys.ITERATION_COUNT, iteration),
        )

        # Populate trace from stored tool calls
        for step in trace_data:
            if not isinstance(step, dict):
                continue
            tool_name = step.get("tool_name", "")
            if not tool_name:
                # Fall back to "action" key for cross-agent compatibility
                action = step.get("action", "")
                tool_name = action.split("(")[0] if action else ""
            if tool_name and tool_name != ContextKeys.NO_TOOL:
                trace.tool_calls.append(
                    ToolCall(
                        tool_name=tool_name,
                        parameters=step.get("tool_input", {}),
                        reasoning=str(step.get("thought", step.get("description", ""))),
                    )
                )

        return trace
