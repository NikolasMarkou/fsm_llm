from __future__ import annotations

"""
ReactAgent — ReAct (Reasoning + Acting) agent implementation.

Uses FSM-LLM's 2-pass architecture to implement the ReAct loop:
Think -> Act -> Observe -> Think -> ... -> Conclude
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
)
from .definitions import AgentConfig, AgentResult, AgentTrace, ToolCall
from .exceptions import AgentError, AgentTimeoutError, BudgetExhaustedError
from .fsm_definitions import build_react_fsm
from .handlers import AgentHandlers
from .hitl import HumanInTheLoop
from .tools import ToolRegistry


class ReactAgent:
    """
    ReAct agent that uses tools to solve tasks step by step.

    Combines FSM-LLM's core (states, transitions, handlers) with
    a tool registry and optional HITL support to implement the
    ReAct pattern.

    Usage::

        registry = ToolRegistry()
        registry.register_function(search, name="search", description="Search the web")
        registry.register_function(calculate, name="calculate", description="Do math")

        agent = ReactAgent(tools=registry)
        result = agent.run("What is the population of France times 2?")
        print(result.answer)
    """

    def __init__(
        self,
        tools: ToolRegistry,
        config: AgentConfig | None = None,
        hitl: HumanInTheLoop | None = None,
        **api_kwargs: Any,
    ) -> None:
        """
        Initialize a ReAct agent.

        :param tools: Tool registry with registered tools
        :param config: Agent configuration (defaults to AgentConfig())
        :param hitl: Optional HITL manager for approval gates
        :param api_kwargs: Additional kwargs passed to fsm_llm.API
        """
        if len(tools) == 0:
            raise AgentError("Cannot create agent with empty tool registry")

        self.tools = tools
        self.config = config or AgentConfig()
        self.hitl = hitl
        self._api_kwargs = api_kwargs
        self._handlers = AgentHandlers(tools)

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
        start_time = time.monotonic()
        self._handlers.reset()

        # Determine if we need HITL approval state
        has_approval_tools = any(t.requires_approval for t in self.tools.list_tools())
        include_approval = (
            self.hitl is not None
            and self.hitl.has_approval_policy
            and has_approval_tools
        )

        # Build FSM from tool registry
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

        # Register handlers
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
            conversation_id=conv_id, package="fsm_llm_agents", agent_type="react"
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

                # HITL: check if approval is needed before acting
                if (
                    self.hitl is not None
                    and current_context.get(ContextKeys.APPROVAL_REQUIRED)
                    and not current_context.get(ContextKeys.APPROVAL_GRANTED)
                ):
                    tool_name = current_context.get(ContextKeys.TOOL_NAME, "")
                    tool_input = current_context.get(ContextKeys.TOOL_INPUT, {})
                    reasoning = current_context.get(ContextKeys.REASONING, "")

                    tool_call = ToolCall(
                        tool_name=tool_name,
                        parameters=tool_input
                        if isinstance(tool_input, dict)
                        else {"input": str(tool_input)},
                        reasoning=reasoning,
                    )

                    approved = self.hitl.request_approval(tool_call, current_context)

                    api.update_context(
                        conv_id,
                        {
                            ContextKeys.APPROVAL_GRANTED: approved,
                            ContextKeys.APPROVAL_REQUIRED: False,
                        },
                    )

                    if not approved:
                        api.update_context(
                            conv_id,
                            {
                                ContextKeys.TOOL_NAME: None,
                                ContextKeys.TOOL_INPUT: None,
                            },
                        )

                response = api.converse(Defaults.CONTINUE_MESSAGE, conv_id)
                responses.append(response)

            # Extract final results
            final_context = api.get_data(conv_id)
            answer = self._extract_answer(final_context, responses)

            # Build trace
            trace_data = final_context.get(ContextKeys.AGENT_TRACE, [])
            trace = AgentTrace(
                tool_calls=[],
                total_iterations=final_context.get(
                    ContextKeys.ITERATION_COUNT, iteration
                ),
            )

            # Populate trace from stored data
            for step in trace_data:
                if isinstance(step, dict) and "action" in step:
                    tool_name = step.get("action", "").split("(")[0]
                    if tool_name and tool_name != "none":
                        trace.tool_calls.append(
                            ToolCall(
                                tool_name=tool_name,
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

    def _register_handlers(self, api: API) -> None:
        """Register agent handlers with the API."""
        # Tool executor: runs on entry to 'act' state
        api.register_handler(
            api.create_handler(HandlerNames.TOOL_EXECUTOR)
            .on_state_entry(AgentStates.ACT)
            .do(self._handlers.execute_tool)
        )

        # Iteration limiter: checks budget on every pre-transition
        api.register_handler(
            api.create_handler(HandlerNames.ITERATION_LIMITER)
            .at(HandlerTiming.PRE_TRANSITION)
            .do(self._handlers.check_iteration_limit)
        )

        # Approval checker: flags tools needing approval
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
        # Priority: explicit final_answer > last response
        answer = final_context.get(ContextKeys.FINAL_ANSWER)
        if answer and isinstance(answer, str) and len(answer) > 5:
            return str(answer)

        # Fall back to the last non-empty response
        for response in reversed(responses):
            if response and len(response.strip()) > 5:
                return response.strip()

        return "Agent could not determine an answer."
