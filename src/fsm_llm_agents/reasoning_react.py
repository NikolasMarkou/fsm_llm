from __future__ import annotations

"""
ReasoningReactAgent — ReAct agent with integrated structured reasoning.

Extends the ReAct pattern with a pseudo-tool ``reason`` that invokes
FSM-LLM's reasoning engine via FSM stacking (push_fsm / pop_fsm).
The agent autonomously decides when to use structured reasoning versus
regular tools.
"""

from collections.abc import Callable
from typing import Any

from fsm_llm import API
from fsm_llm.logging import logger

from .base import BaseAgent
from .constants import (
    AgentStates,
    ContextKeys,
    Defaults,
    LogMessages,
    ReasoningIntegrationKeys,
)
from .definitions import AgentConfig, AgentResult, AgentStep
from .exceptions import AgentError
from .fsm_definitions import build_react_fsm
from .handlers import AgentHandlers
from .hitl import HumanInTheLoop, make_hitl_checker
from .tools import ToolRegistry

# Optional import — reasoning package may not be installed
try:
    from fsm_llm_reasoning import ReasoningEngine

    _HAS_REASONING = True
except ImportError:
    _HAS_REASONING = False


class ReasoningReactAgent(BaseAgent):
    """
    ReAct agent with integrated structured reasoning via FSM stacking.

    Auto-registers a ``reason`` pseudo-tool in the tool registry.
    When the LLM selects ``reason``, the agent pushes a reasoning FSM
    onto the stack (via ``ReasoningEngine``), executes it, and pops
    results back into the agent context under namespaced keys.

    Requires ``fsm_llm_reasoning`` to be installed. Raises
    ``AgentError`` at construction time if the package is missing.

    Usage::

        from fsm_llm_agents import ReasoningReactAgent, ToolRegistry

        registry = ToolRegistry()
        registry.register_function(search, name="search", description="Search")
        agent = ReasoningReactAgent(tools=registry)
        result = agent.run("Analyze: is 97 prime?")
    """

    def __init__(
        self,
        tools: ToolRegistry,
        config: AgentConfig | None = None,
        hitl: HumanInTheLoop | None = None,
        reasoning_model: str | None = None,
        **api_kwargs: Any,
    ) -> None:
        """
        Initialize a ReasoningReactAgent.

        :param tools: Tool registry (``reason`` is auto-registered)
        :param config: Agent configuration
        :param hitl: Optional HITL manager
        :param reasoning_model: Model for the reasoning engine (defaults to config.model)
        :param api_kwargs: Additional kwargs passed to fsm_llm.API
        """
        if not _HAS_REASONING:
            raise AgentError(
                "ReasoningReactAgent requires fsm_llm_reasoning. "
                "Install with: pip install fsm-llm[reasoning]"
            )

        super().__init__(config, **api_kwargs)
        self.hitl = hitl

        # Copy registry to avoid mutating the caller's ToolRegistry
        self.tools = ToolRegistry()
        for tool_def in tools._tools.values():
            self.tools.register(tool_def)
        reason_name = ReasoningIntegrationKeys.REASONING_TOOL_NAME
        if reason_name not in self.tools:
            self.tools.register_function(
                self._reason_placeholder,
                name=reason_name,
                description=(
                    "Use structured reasoning to analyze a complex problem. "
                    "Provide the problem statement as 'problem' parameter. "
                    "Use this when the task requires deep analytical, "
                    "deductive, or critical thinking."
                ),
                parameter_schema={
                    "properties": {
                        "problem": {
                            "type": "string",
                            "description": "The problem to reason about",
                        },
                    }
                },
            )

        if len(self.tools) == 0:
            raise AgentError("Cannot create agent with empty tool registry")

        # Create reasoning engine
        reasoning_model_name = reasoning_model or self.config.model
        self._reasoning_engine = ReasoningEngine(
            model=reasoning_model_name, **api_kwargs
        )
        self._handlers = AgentHandlers(self.tools)

        logger.info(
            LogMessages.AGENT_STARTED.format(
                tool_count=len(self.tools), model=self.config.model
            )
        )

    @staticmethod
    def _reason_placeholder(params: dict) -> str:
        """Placeholder — actual reasoning is intercepted in the run loop."""
        return "Reasoning executed via FSM stacking."

    def run(
        self,
        task: str,
        initial_context: dict[str, Any] | None = None,
    ) -> AgentResult:
        """
        Run the agent on a task.

        When the LLM selects the ``reason`` tool, the POST_TRANSITION
        handler intercepts the call and delegates to
        ``ReasoningEngine.solve_problem()`` instead of the placeholder.
        Results are stored under ``ReasoningIntegrationKeys`` in context.

        :param task: The task/question for the agent
        :param initial_context: Optional initial context data
        :return: AgentResult with answer, trace, and metadata
        """
        self._handlers.reset()

        # Build FSM from tool registry
        has_approval_tools = any(t.requires_approval for t in self.tools.list_tools())
        include_approval = (
            self.hitl is not None
            and self.hitl.has_approval_policy
            and has_approval_tools
        )

        fsm_def = build_react_fsm(
            self.tools,
            task_description=task[:Defaults.MAX_TASK_PREVIEW_LENGTH],
            include_approval_state=include_approval,
        )

        # Build initial context
        context = self._init_context(
            task,
            initial_context,
            extra={
                ContextKeys.OBSERVATIONS: [],
                "_max_iterations": self.config.max_iterations,
            },
        )

        return self._standard_run(task, fsm_def, context, "reasoning_react")

    def _make_reasoning_tool_executor(
        self,
    ) -> Callable[[dict[str, Any]], dict[str, Any]]:
        """Create tool executor that intercepts 'reason' and invokes the reasoning engine.

        For non-reason tools, delegates to the standard AgentHandlers.execute_tool.
        For the 'reason' tool, runs ReasoningEngine.solve_problem() and returns
        results under namespaced keys — all inside the POST_TRANSITION handler
        so the FSM pipeline processes them at the correct time.
        """
        base_handler = self._handlers
        reason_name = ReasoningIntegrationKeys.REASONING_TOOL_NAME
        engine = self._reasoning_engine

        def execute_tool_with_reasoning(context: dict[str, Any]) -> dict[str, Any]:
            tool_name = context.get(ContextKeys.TOOL_NAME)

            # Non-reason tools: delegate to standard handler
            if tool_name != reason_name:
                return base_handler.execute_tool(context)

            # Extract problem from tool input
            tool_input = context.get(ContextKeys.TOOL_INPUT) or {}
            if isinstance(tool_input, str):
                problem = tool_input
            elif isinstance(tool_input, dict):
                problem = tool_input.get("problem", str(tool_input))
            else:
                problem = str(tool_input)

            logger.info(f"ReasoningReactAgent: invoking reasoning for: {problem[:100]}")

            try:
                solution, trace_info = engine.solve_problem(problem)

                reasoning_type = trace_info.get("reasoning_trace", {}).get(
                    "reasoning_types_used", ["unknown"]
                )
                confidence = trace_info.get("reasoning_trace", {}).get(
                    "final_confidence", 0.0
                )

                observation = (
                    f"Structured reasoning result (type={reasoning_type}): {solution}"
                )

                # Accumulate in observations
                observations = context.get(ContextKeys.OBSERVATIONS, [])
                if not isinstance(observations, list):
                    observations = []
                step_num = len(observations) + 1
                observation_entry = (
                    f"[Step {step_num}] Tool: reason | "
                    f"Input: {problem[:100]} | "
                    f"Result: {observation}"
                )
                observations.append(observation_entry)

                # Track in agent trace
                trace = context.get(ContextKeys.AGENT_TRACE, [])
                if not isinstance(trace, list):
                    trace = []
                trace.append(
                    AgentStep(
                        iteration=step_num,
                        thought=context.get(ContextKeys.REASONING, ""),
                        action=f"reason({problem[:100]})",
                        observation=observation,
                    ).model_dump(mode="json")
                )

                return {
                    ContextKeys.TOOL_RESULT: observation,
                    ContextKeys.TOOL_STATUS: "success",
                    ContextKeys.TOOL_ERROR: None,
                    ContextKeys.OBSERVATIONS: observations,
                    ContextKeys.OBSERVATION_COUNT: len(observations),
                    ContextKeys.AGENT_TRACE: trace,
                    ContextKeys.TOOL_NAME: None,
                    ContextKeys.TOOL_INPUT: None,
                    ContextKeys.SHOULD_TERMINATE: None,
                    # Namespaced reasoning results
                    ReasoningIntegrationKeys.REASONING_RESULT: solution,
                    ReasoningIntegrationKeys.REASONING_TYPE_USED: str(reasoning_type),
                    ReasoningIntegrationKeys.REASONING_CONFIDENCE: confidence,
                }

            except Exception as e:
                logger.warning(f"Reasoning failed, recording error: {e}", exc_info=True)
                return {
                    ContextKeys.TOOL_RESULT: f"Reasoning failed: {e}",
                    ContextKeys.TOOL_STATUS: "failed",
                    ContextKeys.TOOL_ERROR: str(e),
                    ContextKeys.TOOL_NAME: None,
                    ContextKeys.TOOL_INPUT: None,
                }

        return execute_tool_with_reasoning

    def _register_handlers(self, api: API) -> None:
        """Register agent handlers with the API."""
        self._register_tool_executor(api, AgentStates.ACT, self._make_reasoning_tool_executor())

        self._register_iteration_limiter(api, self._handlers.check_iteration_limit)

        if self.hitl is not None and self.hitl.has_approval_policy:
            self._register_hitl_gate(api, make_hitl_checker(self.hitl))

