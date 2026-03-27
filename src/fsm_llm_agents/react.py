from __future__ import annotations

"""
ReactAgent — ReAct (Reasoning + Acting) agent implementation.

Uses FSM-LLM's 2-pass architecture to implement the ReAct loop:
Think -> Act -> Observe -> Think -> ... -> Conclude
"""

from typing import Any

from fsm_llm import API
from fsm_llm.handlers import HandlerTiming
from fsm_llm.logging import logger

from .base import BaseAgent
from .constants import (
    AgentStates,
    ContextKeys,
    HandlerNames,
    HandlerPriorities,
    LogMessages,
)
from .definitions import AgentConfig, AgentResult, ToolCall
from .exceptions import AgentError
from .fsm_definitions import build_react_fsm
from .handlers import AgentHandlers
from .hitl import HumanInTheLoop
from .tools import ToolRegistry


class ReactAgent(BaseAgent):
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
        use_classification: bool = False,
        **api_kwargs: Any,
    ) -> None:
        if len(tools) == 0:
            raise AgentError("Cannot create agent with empty tool registry")

        super().__init__(config, **api_kwargs)
        self.tools = tools
        self.hitl = hitl
        self.use_classification = use_classification
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
        self._handlers.reset()

        # Determine if we need HITL approval state
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
            use_classification=self.use_classification,
        )

        context: dict[str, Any] = dict(initial_context) if initial_context else {}
        context[ContextKeys.TASK] = task
        context[ContextKeys.OBSERVATIONS] = []
        context[ContextKeys.AGENT_TRACE] = []
        context[ContextKeys.ITERATION_COUNT] = 0
        context["_max_iterations"] = self.config.max_iterations

        return self._standard_run(task, fsm_def, context, "react")

    def _on_loop_iteration(self, api: API, conv_id: str, iteration: int) -> None:
        """Handle HITL approval gates before each converse()."""
        if self.hitl is None:
            return

        current_context = api.get_data(conv_id)
        if not (
            current_context.get(ContextKeys.APPROVAL_REQUIRED)
            and not current_context.get(ContextKeys.APPROVAL_GRANTED)
        ):
            return

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

    def _register_handlers(self, api: API) -> None:
        """Register agent handlers with the API."""
        api.register_handler(
            api.create_handler(HandlerNames.TOOL_EXECUTOR)
            .with_priority(HandlerPriorities.TOOL_EXECUTOR)
            .on_state_entry(AgentStates.ACT)
            .do(self._handlers.execute_tool)
        )

        api.register_handler(
            api.create_handler(HandlerNames.ITERATION_LIMITER)
            .with_priority(HandlerPriorities.ITERATION_LIMITER)
            .at(HandlerTiming.PRE_TRANSITION)
            .do(self._handlers.check_iteration_limit)
        )

        if self.hitl is not None and self.hitl.has_approval_policy:
            api.register_handler(
                api.create_handler(HandlerNames.HITL_GATE)
                .with_priority(HandlerPriorities.HITL_GATE)
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
