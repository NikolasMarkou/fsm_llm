from __future__ import annotations

"""
Agent-specific handlers for tool execution, iteration limiting,
observation tracking, and HITL gating.
"""

from typing import Any

from fsm_llm.logging import logger

from .constants import ContextKeys, Defaults, LogMessages
from .definitions import AgentStep, ToolCall
from .tools import ToolRegistry, normalize_tool_input

# Parameter names where using the task string as a default value is reasonable.
# For domain-specific params (expression, city, filename), the tool should fail
# with an explicit error so the model learns to provide proper params.
_QUERY_LIKE_PARAMS = frozenset(
    {
        "query",
        "search",
        "text",
        "input",
        "prompt",
        "question",
        "message",
        "content",
        "description",
        "topic",
        "subject",
    }
)


class AgentHandlers:
    """Collection of handler functions for agent FSM operations."""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry
        self._current_iteration = 0
        self._consecutive_no_tool = 0

    def reset(self) -> None:
        """Reset handler state for a new run."""
        self._current_iteration = 0
        self._consecutive_no_tool = 0

    def execute_tool(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Execute the tool selected during the think state.

        Called as a POST_TRANSITION handler when entering the 'act' state.
        """
        tool_name = context.get(ContextKeys.TOOL_NAME)
        tool_input = context.get(ContextKeys.TOOL_INPUT)
        if tool_input is None:
            tool_input = {}
        reasoning = context.get(ContextKeys.REASONING, "")

        if not tool_name or tool_name == ContextKeys.NO_TOOL:
            # Skip stall detection for HITL agents awaiting approval
            if context.get(ContextKeys.APPROVAL_REQUIRED):
                return {
                    ContextKeys.TOOL_RESULT: "Awaiting approval.",
                    ContextKeys.TOOL_STATUS: "skipped",
                }

            # Block premature termination: must use at least one tool
            if self._current_iteration <= 1 and context.get(
                ContextKeys.SHOULD_TERMINATE
            ):
                tool_names = [t.name for t in self.registry.list_tools()]
                return {
                    ContextKeys.TOOL_RESULT: (
                        "You must use at least one tool before concluding. "
                        f"Available: {', '.join(tool_names)}"
                    ),
                    ContextKeys.TOOL_STATUS: "rejected",
                    ContextKeys.SHOULD_TERMINATE: False,
                }

            self._consecutive_no_tool += 1
            should_terminate = context.get(ContextKeys.SHOULD_TERMINATE)

            # Stall detection: force terminate after 2 consecutive no-tool cycles
            if self._consecutive_no_tool >= 2:
                logger.warning(
                    f"Stall detected: {self._consecutive_no_tool} consecutive "
                    f"iterations with no tool selected, forcing termination"
                )
                return {
                    ContextKeys.TOOL_RESULT: (
                        "No tool was called for 2 consecutive iterations. "
                        "Terminating — provide your best answer now."
                    ),
                    ContextKeys.TOOL_STATUS: "skipped",
                    ContextKeys.SHOULD_TERMINATE: True,
                }

            if not should_terminate:
                # Instructive warning to push model toward tool use
                tool_names = [t.name for t in self.registry.list_tools()]
                return {
                    ContextKeys.TOOL_RESULT: (
                        "WARNING: No tool was called but the task is not complete. "
                        f"You must select a tool from: {', '.join(tool_names)}. "
                        "Do not answer from memory — use a tool to gather information."
                    ),
                    ContextKeys.TOOL_STATUS: "skipped",
                }

            return {
                ContextKeys.TOOL_RESULT: "No tool was selected.",
                ContextKeys.TOOL_STATUS: "skipped",
            }

        # Reset stall counter — a tool was actually selected
        self._consecutive_no_tool = 0

        tool_input = normalize_tool_input(tool_input)

        # Recovery: if tool_input is empty, try to infer parameters from context.
        if not tool_input and tool_name in self.registry:
            tool_def = self.registry.get(tool_name)
            schema = tool_def.parameter_schema or {}
            props = schema.get("properties", {})
            # Handle flat {param: description} schemas (no "properties" wrapper)
            if not props and schema and "type" not in schema:
                if all(isinstance(v, str) for v in schema.values()):
                    props = {k: {} for k in schema}
            required = schema.get("required", list(props.keys()))

            # Single-param recovery: only for query-like parameters where the
            # task string is a reasonable value (search, lookup, text processing).
            # For domain-specific params (expression, city, filename), let the
            # tool fail so the model gets explicit feedback to retry with params.
            if len(required) == 1:
                param_name = required[0]
                if param_name.lower() in _QUERY_LIKE_PARAMS:
                    task = context.get(ContextKeys.TASK, "")
                    if task:
                        tool_input = {param_name: task}
                        logger.info(f"Recovered empty tool_input: {param_name}=<task>")
                else:
                    logger.info(
                        f"Empty tool_input for '{tool_name}', param "
                        f"'{param_name}' is not query-like — skipping "
                        f"auto-recovery to get explicit error feedback"
                    )

            # Multi-param recovery: provide a context-aware example so the
            # model can learn the expected format from the error feedback.
            elif len(required) > 1:
                example_params = {}
                for k in required:
                    prop_info = props.get(k, {})
                    ptype = prop_info.get("type", "string")
                    pdesc = prop_info.get("description", k)
                    example_params[k] = f"<{ptype}: {pdesc}>"
                tool_input = example_params
                logger.info(
                    f"Empty tool_input for multi-param tool '{tool_name}' "
                    f"({len(required)} required params) — using "
                    f"format example as input for error feedback"
                )

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

        # Build observation string — prefix failures so LLM can distinguish
        observation = result.summary
        if not result.success:
            observation = f"[TOOL FAILED] {observation}"

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
            dropped = len(observations) - Defaults.MAX_OBSERVATIONS
            logger.debug(
                f"Pruning {dropped} old observations "
                f"(keeping last {Defaults.MAX_OBSERVATIONS})"
            )
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

    def classification_tool_override(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Override should_terminate when classification confidently selects a tool.

        When use_classification=True, the pipeline stores the full classification
        result at ``_tool_name_classification``.  If the classifier picked a real
        tool but the free-form extraction set should_terminate=True, this handler
        clears should_terminate so the agent proceeds to ACT instead of CONCLUDE.
        """
        # Only act when should_terminate is True (potential conflict)
        if not context.get(ContextKeys.SHOULD_TERMINATE):
            return {}

        classification = context.get("_tool_name_classification")
        if not isinstance(classification, dict):
            return {}

        intent = classification.get("intent", ContextKeys.NO_TOOL)
        confidence = classification.get("confidence", 0.0)

        if intent != ContextKeys.NO_TOOL and confidence >= 0.5:
            logger.info(
                f"Classification override: tool={intent} "
                f"(confidence={confidence:.2f}), unsetting should_terminate"
            )
            return {ContextKeys.SHOULD_TERMINATE: False}

        return {}

    def check_iteration_limit(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Check if the iteration limit has been reached.

        Called as a PRE_TRANSITION handler. Since the transition decision
        is already made before this handler fires, we trigger one iteration
        early (>= max - 1) so the conclude transition fires on the next
        iteration rather than overshooting by 1.
        """
        self._current_iteration += 1
        max_iterations = context.get("_max_iterations", Defaults.MAX_ITERATIONS)

        logger.debug(
            LogMessages.ITERATION.format(
                current=self._current_iteration, max=max_iterations
            )
        )

        if self._current_iteration >= max_iterations - 1:
            return {
                ContextKeys.ITERATION_COUNT: self._current_iteration,
                ContextKeys.MAX_ITERATIONS_REACHED: True,
                ContextKeys.SHOULD_TERMINATE: True,
            }

        return {ContextKeys.ITERATION_COUNT: self._current_iteration}
