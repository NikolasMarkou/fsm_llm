from __future__ import annotations

"""
Agent-specific handlers for tool execution, iteration limiting,
observation tracking, and HITL gating.
"""

from typing import Any

from fsm_llm.logging import logger

from .constants import ContextKeys, Defaults, LogMessages
from .definitions import AgentStep, ToolCall
from .tools import ToolRegistry


class AgentHandlers:
    """Collection of handler functions for agent FSM operations."""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry
        self._current_iteration = 0

    def reset(self) -> None:
        """Reset handler state for a new run."""
        self._current_iteration = 0

    def execute_tool(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Execute the tool selected during the think state.

        Called as a POST_TRANSITION handler when entering the 'act' state.
        """
        tool_name = context.get(ContextKeys.TOOL_NAME)
        tool_input = context.get(ContextKeys.TOOL_INPUT) or {}
        reasoning = context.get(ContextKeys.REASONING, "")

        if not tool_name or tool_name == "none":
            return {
                ContextKeys.TOOL_RESULT: "No tool was selected.",
                ContextKeys.TOOL_STATUS: "skipped",
            }

        # Normalize tool_input to dict
        if isinstance(tool_input, str):
            tool_input = {"input": tool_input}
        elif not isinstance(tool_input, dict):
            tool_input = {"input": str(tool_input)}

        logger.info(LogMessages.TOOL_SELECTED.format(name=tool_name, input=tool_input))

        tool_call = ToolCall(
            tool_name=tool_name,
            parameters=tool_input,
            reasoning=reasoning,
        )

        result = self.registry.execute(tool_call)

        if result.success:
            logger.info(LogMessages.TOOL_EXECUTED.format(name=tool_name))
        else:
            logger.warning(
                LogMessages.TOOL_FAILED.format(name=tool_name, error=result.error)
            )

        # Build observation string
        observation = result.summary

        # Accumulate observations
        observations = context.get(ContextKeys.OBSERVATIONS, [])
        if not isinstance(observations, list):
            observations = []

        step_num = len(observations) + 1
        observation_entry = (
            f"[Step {step_num}] Tool: {tool_name} | "
            f"Input: {tool_input} | "
            f"Result: {observation}"
        )
        observations.append(observation_entry)

        # Prune if too many observations
        if len(observations) > Defaults.MAX_OBSERVATIONS:
            observations = observations[-Defaults.MAX_OBSERVATIONS :]

        # Track in agent trace
        trace = context.get(ContextKeys.AGENT_TRACE, [])
        if not isinstance(trace, list):
            trace = []
        trace.append(
            AgentStep(
                iteration=step_num,
                thought=reasoning,
                action=f"{tool_name}({tool_input})",
                observation=observation,
            ).model_dump(mode="json")
        )

        return {
            ContextKeys.TOOL_RESULT: observation,
            ContextKeys.TOOL_STATUS: "success" if result.success else "failed",
            ContextKeys.TOOL_ERROR: result.error,
            ContextKeys.OBSERVATIONS: observations,
            ContextKeys.OBSERVATION_COUNT: len(observations),
            ContextKeys.AGENT_TRACE: trace,
            # Clear tool selection for next iteration
            ContextKeys.TOOL_NAME: None,
            ContextKeys.TOOL_INPUT: None,
            ContextKeys.SHOULD_TERMINATE: None,
        }

    def check_iteration_limit(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Check if the iteration limit has been reached.

        Called as a PRE_TRANSITION handler.
        """
        self._current_iteration += 1
        max_iterations = context.get("_max_iterations", Defaults.MAX_ITERATIONS)

        logger.debug(
            LogMessages.ITERATION.format(
                current=self._current_iteration, max=max_iterations
            )
        )

        if self._current_iteration >= max_iterations:
            return {
                ContextKeys.ITERATION_COUNT: self._current_iteration,
                ContextKeys.MAX_ITERATIONS_REACHED: True,
                ContextKeys.SHOULD_TERMINATE: True,
            }

        return {ContextKeys.ITERATION_COUNT: self._current_iteration}

    def check_approval(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Check if the selected tool requires human approval.

        Called as a CONTEXT_UPDATE handler when tool_name is updated.
        Uses the approval_policy provided to HumanInTheLoop.
        """
        tool_name = context.get(ContextKeys.TOOL_NAME)
        if not tool_name or tool_name == "none":
            return {}

        if tool_name in self.registry:
            tool = self.registry.get(tool_name)
            if tool.requires_approval:
                return {ContextKeys.APPROVAL_REQUIRED: True}

        return {ContextKeys.APPROVAL_REQUIRED: False}
