from __future__ import annotations

"""
ReactAgent — ReAct (Reasoning + Acting) agent implementation.

Uses FSM-LLM's 2-pass architecture to implement the ReAct loop:
Think -> Act -> Observe -> Think -> ... -> Conclude
"""

from typing import Any

from fsm_llm import API
from fsm_llm.logging import logger

from .base import BaseAgent
from .constants import (
    AgentStates,
    ContextKeys,
    Defaults,
    HandlerPriorities,
    LogMessages,
)
from .definitions import AgentConfig, AgentResult
from .exceptions import AgentError
from .fsm_definitions import build_react_fsm
from .handlers import AgentHandlers
from .hitl import HumanInTheLoop, make_hitl_checker
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
            task_description=task[: Defaults.MAX_TASK_PREVIEW_LENGTH],
            include_approval_state=include_approval,
            use_classification=self.use_classification,
        )

        context = self._init_context(
            task,
            initial_context,
            extra={
                ContextKeys.OBSERVATIONS: [],
                "_max_iterations": self.config.max_iterations,
            },
        )

        return self._standard_run(task, fsm_def, context, "react")

    def _on_loop_iteration(self, api: API, conv_id: str, iteration: int) -> None:
        """Handle HITL approval gates before each converse()."""
        self._handle_hitl_approval(api, conv_id)

    def _register_handlers(self, api: API) -> None:
        """Register agent handlers with the API."""
        self._register_tool_executor(api, AgentStates.ACT, self._handlers.execute_tool)
        self._register_iteration_limiter(api, self._handlers.check_iteration_limit)

        if self.use_classification:
            from fsm_llm.handlers import HandlerTiming

            api.register_handler(
                api.create_handler("classification_tool_override")
                .with_priority(HandlerPriorities.TOOL_EXECUTOR)
                .at(HandlerTiming.POST_PROCESSING)
                .on_state(AgentStates.THINK)
                .do(self._handlers.classification_tool_override)
            )

        if self.hitl is not None and self.hitl.has_approval_policy:
            self._register_hitl_gate(api, make_hitl_checker(self.hitl))
