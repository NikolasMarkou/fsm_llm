from __future__ import annotations

"""
ReasoningReactAgent — ReAct agent with integrated structured reasoning.

Extends the ReAct pattern with a pseudo-tool ``reason`` that invokes
FSM-LLM's reasoning engine via FSM stacking (push_fsm / pop_fsm).
The agent autonomously decides when to use structured reasoning versus
regular tools.
"""

import time
from typing import Any

from fsm_llm import API
from fsm_llm.handlers import HandlerTiming
from fsm_llm.logging import logger

from .constants import (
    AgentStates,
    ContextKeys,
    Defaults,
    HandlerNames,
    LogMessages,
    ReasoningIntegrationKeys,
)
from .definitions import AgentConfig, AgentResult, AgentStep, AgentTrace, ToolCall
from .exceptions import AgentError, AgentTimeoutError, BudgetExhaustedError
from .fsm_definitions import build_react_fsm
from .handlers import AgentHandlers
from .hitl import HumanInTheLoop
from .tools import ToolRegistry

# Optional import — reasoning package may not be installed
try:
    from fsm_llm_reasoning import ReasoningEngine

    _HAS_REASONING = True
except ImportError:
    _HAS_REASONING = False


class ReasoningReactAgent:
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

        self.config = config or AgentConfig()
        self.hitl = hitl
        self._api_kwargs = api_kwargs

        # Auto-register the reason pseudo-tool
        self.tools = tools
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

        When the LLM selects the ``reason`` tool, the agent intercepts
        the call and delegates to ``ReasoningEngine.solve_problem()``
        instead of the placeholder function. Results are stored under
        ``ReasoningIntegrationKeys`` in the context.

        :param task: The task/question for the agent
        :param initial_context: Optional initial context data
        :return: AgentResult with answer, trace, and metadata
        """
        start_time = time.monotonic()
        reason_tool_name = ReasoningIntegrationKeys.REASONING_TOOL_NAME

        # Build FSM from tool registry
        has_approval_tools = any(t.requires_approval for t in self.tools.list_tools())
        include_approval = (
            self.hitl is not None
            and self.hitl.has_approval_policy
            and has_approval_tools
        )

        fsm_def = build_react_fsm(
            self.tools,
            task_description=task[:200],
            include_approval_state=include_approval,
        )

        # Create API instance
        api = API.from_definition(
            fsm_def,
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            **self._api_kwargs,
        )

        # Register handlers — tool executor will handle regular tools
        self._register_handlers(api)

        # Build initial context
        context: dict[str, Any] = dict(initial_context) if initial_context else {}
        context[ContextKeys.TASK] = task
        context[ContextKeys.OBSERVATIONS] = []
        context[ContextKeys.AGENT_TRACE] = []
        context[ContextKeys.ITERATION_COUNT] = 0
        context["_max_iterations"] = self.config.max_iterations

        # Start conversation
        conv_id, initial_response = api.start_conversation(context)
        log = logger.bind(
            conversation_id=conv_id, package="fsm_llm_agents", agent_type="reasoning_react"
        )

        try:
            responses = [initial_response]
            iteration = 0

            while not api.has_conversation_ended(conv_id):
                iteration += 1

                # Check time budget
                elapsed = time.monotonic() - start_time
                if elapsed > self.config.timeout_seconds:
                    raise AgentTimeoutError(self.config.timeout_seconds)

                # Check iteration budget
                if (
                    iteration
                    > self.config.max_iterations * Defaults.FSM_BUDGET_MULTIPLIER
                ):
                    raise BudgetExhaustedError("iterations", self.config.max_iterations)

                current_context = api.get_data(conv_id)

                # Intercept reason tool before the act state handler runs
                tool_name = current_context.get(ContextKeys.TOOL_NAME)
                if tool_name == reason_tool_name:
                    self._execute_reasoning(api, conv_id, current_context)

                response = api.converse(Defaults.CONTINUE_MESSAGE, conv_id)
                responses.append(response)

            # Extract final results
            final_context = api.get_data(conv_id)
            answer = self._extract_answer(final_context, responses)

            trace_data = final_context.get(ContextKeys.AGENT_TRACE, [])
            trace = AgentTrace(
                steps=[],
                tool_calls=[],
                total_iterations=final_context.get(
                    ContextKeys.ITERATION_COUNT, iteration
                ),
            )

            for step in trace_data:
                if isinstance(step, dict) and "action" in step:
                    action_name = step.get("action", "").split("(")[0]
                    if action_name and action_name != "none":
                        trace.tool_calls.append(
                            ToolCall(
                                tool_name=action_name,
                                parameters={},
                                reasoning=step.get("thought", ""),
                            )
                        )

            elapsed = time.monotonic() - start_time
            log.info(
                LogMessages.AGENT_COMPLETE.format(iterations=trace.total_iterations)
            )

            return AgentResult(
                answer=answer,
                success=True,
                trace=trace,
                final_context={
                    k: v for k, v in final_context.items() if not k.startswith("_")
                },
            )

        except (AgentTimeoutError, BudgetExhaustedError):
            raise
        except Exception as e:
            raise AgentError(
                f"Agent execution failed: {e}",
                details={"task": task, "iteration": iteration},
            ) from e
        finally:
            api.end_conversation(conv_id)

    def _execute_reasoning(
        self,
        api: API,
        conv_id: str,
        current_context: dict[str, Any],
    ) -> None:
        """
        Execute structured reasoning and inject results into context.

        Intercepts the ``reason`` tool call, runs ReasoningEngine.solve_problem(),
        and stores results under namespaced keys. The regular tool executor handler
        will see the pre-filled result and skip execution.
        """
        tool_input = current_context.get(ContextKeys.TOOL_INPUT) or {}
        if isinstance(tool_input, str):
            problem = tool_input
        elif isinstance(tool_input, dict):
            problem = tool_input.get("problem", str(tool_input))
        else:
            problem = str(tool_input)

        logger.info(f"ReasoningReactAgent: invoking reasoning for: {problem[:100]}")

        try:
            solution, trace_info = self._reasoning_engine.solve_problem(problem)

            # Build observation from reasoning result
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
            observations = current_context.get(ContextKeys.OBSERVATIONS, [])
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
            trace = current_context.get(ContextKeys.AGENT_TRACE, [])
            if not isinstance(trace, list):
                trace = []
            trace.append(
                AgentStep(
                    iteration=step_num,
                    thought=current_context.get(ContextKeys.REASONING, ""),
                    action=f"reason({problem[:100]})",
                    observation=observation,
                ).model_dump(mode="json")
            )

            # Update context with reasoning results + clear tool selection
            api.update_context(
                conv_id,
                {
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
                },
            )

        except Exception as e:
            logger.warning(f"Reasoning failed, falling back: {e}")
            # On failure, let the placeholder tool run
            api.update_context(
                conv_id,
                {
                    ContextKeys.TOOL_RESULT: f"Reasoning failed: {e}",
                    ContextKeys.TOOL_STATUS: "failed",
                    ContextKeys.TOOL_ERROR: str(e),
                    ContextKeys.TOOL_NAME: None,
                    ContextKeys.TOOL_INPUT: None,
                },
            )

    def _register_handlers(self, api: API) -> None:
        """Register agent handlers with the API."""
        api.register_handler(
            api.create_handler(HandlerNames.TOOL_EXECUTOR)
            .on_state_entry(AgentStates.ACT)
            .do(self._handlers.execute_tool)
        )

        api.register_handler(
            api.create_handler(HandlerNames.ITERATION_LIMITER)
            .at(HandlerTiming.PRE_TRANSITION)
            .do(self._handlers.check_iteration_limit)
        )

        if self.hitl is not None and self.hitl.has_approval_policy:
            api.register_handler(
                api.create_handler(HandlerNames.HITL_GATE)
                .at(HandlerTiming.CONTEXT_UPDATE)
                .when_keys_updated(ContextKeys.TOOL_NAME)
                .do(self._make_hitl_checker())
            )

    def _make_hitl_checker(self) -> Any:
        """Create a HITL approval checker handler function."""
        hitl = self.hitl

        def check_approval(context: dict[str, Any]) -> dict[str, Any]:
            tool_name = context.get(ContextKeys.TOOL_NAME)
            if not tool_name or tool_name == "none" or hitl is None:
                return {}
            tool_input = context.get(ContextKeys.TOOL_INPUT, {})
            reasoning = context.get(ContextKeys.REASONING, "")
            tool_call = ToolCall(
                tool_name=tool_name,
                parameters=tool_input
                if isinstance(tool_input, dict)
                else {"input": str(tool_input)},
                reasoning=reasoning,
            )
            if hitl.requires_approval(tool_call, context):
                return {ContextKeys.APPROVAL_REQUIRED: True}
            return {ContextKeys.APPROVAL_REQUIRED: False}

        return check_approval

    def _extract_answer(
        self,
        final_context: dict[str, Any],
        responses: list[str],
    ) -> str:
        """Extract the final answer from context or responses."""
        answer = final_context.get(ContextKeys.FINAL_ANSWER)
        if answer and isinstance(answer, str) and len(answer) > 5:
            return answer
        for response in reversed(responses):
            if response and len(response.strip()) > 5:
                return response.strip()
        return "Agent could not determine an answer."
