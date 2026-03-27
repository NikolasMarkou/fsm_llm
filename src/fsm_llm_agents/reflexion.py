from __future__ import annotations

"""
ReflexionAgent — ReAct with evaluation, verbal self-critique, and episodic memory.

Extends the ReAct loop with an evaluation gate and a reflection state:
Think -> Act -> Evaluate -> Reflect (if failed) -> Think (loop)
                          -> Conclude (if passed)
"""

from collections.abc import Callable
from typing import Any

from fsm_llm import API
from fsm_llm.handlers import HandlerTiming
from fsm_llm.logging import logger

from .base import BaseAgent
from .constants import (
    ContextKeys,
    Defaults,
    HandlerNames,
    HandlerPriorities,
    LogMessages,
    ReflexionStates,
)
from .definitions import (
    AgentConfig,
    AgentResult,
    EvaluationResult,
    ReflexionMemory,
    ToolCall,
)
from .exceptions import AgentError
from .fsm_definitions import build_reflexion_fsm
from .handlers import AgentHandlers
from .hitl import HumanInTheLoop
from .tools import ToolRegistry


class ReflexionAgent(BaseAgent):
    """
    Reflexion agent that evaluates and self-critiques its own outputs.

    Combines the ReAct tool-use loop with an evaluation gate and episodic
    memory. If evaluation fails, the agent reflects, stores lessons in
    episodic memory, and retries with an improved strategy.

    Usage::

        agent = ReflexionAgent(tools=registry, max_reflections=3)
        result = agent.run("What year was the Eiffel Tower built?")
        print(result.answer)
    """

    def __init__(
        self,
        tools: ToolRegistry,
        config: AgentConfig | None = None,
        evaluation_fn: Callable[[dict[str, Any]], EvaluationResult] | None = None,
        max_reflections: int = Defaults.MAX_REFLECTIONS,
        hitl: HumanInTheLoop | None = None,
        **api_kwargs: Any,
    ) -> None:
        """
        Initialize a Reflexion agent.

        :param tools: Tool registry with registered tools
        :param config: Agent configuration (defaults to AgentConfig())
        :param evaluation_fn: Optional external evaluation function.
            If None, the LLM self-evaluates via the FSM evaluate state.
        :param max_reflections: Maximum reflect->think cycles before forcing conclude
        :param hitl: Optional HITL manager for approval gates
        :param api_kwargs: Additional kwargs passed to fsm_llm.API
        """
        if len(tools) == 0:
            raise AgentError("Cannot create agent with empty tool registry")

        super().__init__(config, **api_kwargs)
        self.tools = tools
        self.evaluation_fn = evaluation_fn
        self.max_reflections = max_reflections
        self.hitl = hitl
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
        Run the Reflexion agent on a task.

        :param task: The task/question for the agent to solve
        :param initial_context: Optional initial context data
        :return: AgentResult with answer, trace, and metadata
        """
        self._handlers.reset()

        fsm_def = build_reflexion_fsm(
            self.tools,
            task_description=task[:200],
        )

        # Build initial context
        context: dict[str, Any] = dict(initial_context) if initial_context else {}
        context.update(
            {
                ContextKeys.TASK: task,
                ContextKeys.OBSERVATIONS: [],
                ContextKeys.AGENT_TRACE: [],
                ContextKeys.ITERATION_COUNT: 0,
                ContextKeys.EPISODIC_MEMORY: [],
                ContextKeys.REFLECTION_COUNT: 0,
                "_max_iterations": self.config.max_iterations,
            }
        )

        return self._standard_run(task, fsm_def, context, "reflexion")

    def _on_loop_iteration(self, api: API, conv_id: str, iteration: int) -> None:
        """Handle HITL approval gates before each converse()."""
        if self.hitl is None:
            return

        current_ctx = api.get_data(conv_id)
        if not (
            current_ctx.get(ContextKeys.APPROVAL_REQUIRED)
            and not current_ctx.get(ContextKeys.APPROVAL_GRANTED)
        ):
            return

        tool_name = current_ctx.get(ContextKeys.TOOL_NAME, "")
        tool_input = current_ctx.get(ContextKeys.TOOL_INPUT, {})
        reasoning = current_ctx.get(ContextKeys.REASONING, "")

        tool_call = ToolCall(
            tool_name=tool_name,
            parameters=tool_input
            if isinstance(tool_input, dict)
            else {"input": str(tool_input)},
            reasoning=reasoning,
        )

        approved = self.hitl.request_approval(tool_call, current_ctx)
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
            .on_state_entry(ReflexionStates.ACT)
            .do(self._handlers.execute_tool)
        )
        api.register_handler(
            api.create_handler(HandlerNames.ITERATION_LIMITER)
            .with_priority(HandlerPriorities.ITERATION_LIMITER)
            .at(HandlerTiming.PRE_TRANSITION)
            .do(self._handlers.check_iteration_limit)
        )
        api.register_handler(
            api.create_handler(HandlerNames.REFLEXION_REFLECTOR)
            .with_priority(HandlerPriorities.TOOL_EXECUTOR)
            .on_state_entry(ReflexionStates.REFLECT)
            .do(self._make_reflection_handler())
        )

        # External evaluation: override LLM self-evaluation with user-provided fn
        if self.evaluation_fn is not None:
            api.register_handler(
                api.create_handler("external_evaluator")
                .with_priority(HandlerPriorities.TOOL_EXECUTOR)
                .at(HandlerTiming.CONTEXT_UPDATE)
                .on_state(ReflexionStates.EVALUATE)
                .do(self._make_evaluation_handler())
            )

        # HITL: flag tools needing approval
        if self.hitl is not None and self.hitl.has_approval_policy:
            api.register_handler(
                api.create_handler(HandlerNames.HITL_GATE)
                .with_priority(HandlerPriorities.HITL_GATE)
                .at(HandlerTiming.CONTEXT_UPDATE)
                .when_keys_updated(ContextKeys.TOOL_NAME)
                .do(self._make_hitl_checker())
            )

    def _make_evaluation_handler(self) -> Callable[[dict[str, Any]], dict[str, Any]]:
        """Create handler that runs external evaluation_fn after LLM extraction."""
        if self.evaluation_fn is None:
            raise AgentError(
                "evaluation_fn must be set before creating evaluation handler"
            )
        evaluation_fn = self.evaluation_fn

        def handle_evaluation(context: dict[str, Any]) -> dict[str, Any]:
            eval_result = evaluation_fn(context)
            return {
                ContextKeys.EVALUATION_PASSED: eval_result.passed,
                ContextKeys.EVALUATION_SCORE: eval_result.score,
                ContextKeys.EVALUATION_FEEDBACK: eval_result.feedback,
            }

        return handle_evaluation

    def _make_reflection_handler(self) -> Callable[[dict[str, Any]], dict[str, Any]]:
        """Create the reflection and episodic memory handler."""
        max_reflections = self.max_reflections

        def handle_reflection(context: dict[str, Any]) -> dict[str, Any]:
            reflection_count = context.get(ContextKeys.REFLECTION_COUNT, 0) + 1
            episodic_memory = list(context.get(ContextKeys.EPISODIC_MEMORY, []))

            reflection_text = context.get("reflection", "")
            lessons = context.get("lessons", [])
            if not isinstance(lessons, list):
                lessons = [str(lessons)] if lessons else []

            memory_entry = ReflexionMemory(
                episode=reflection_count,
                task_summary=context.get(ContextKeys.TASK, ""),
                outcome=context.get(ContextKeys.EVALUATION_FEEDBACK, ""),
                reflection=str(reflection_text),
                lessons=lessons,
            )
            episodic_memory.append(memory_entry.model_dump(mode="json"))

            logger.info(
                LogMessages.REFLECTION.format(
                    current=reflection_count,
                    max=max_reflections,
                    summary=str(reflection_text)[:80],
                )
            )

            updates: dict[str, Any] = {
                ContextKeys.REFLECTION_COUNT: reflection_count,
                ContextKeys.EPISODIC_MEMORY: episodic_memory,
                ContextKeys.EVALUATION_PASSED: None,
                ContextKeys.EVALUATION_SCORE: None,
                ContextKeys.EVALUATION_FEEDBACK: None,
            }
            if reflection_count >= max_reflections:
                updates[ContextKeys.SHOULD_TERMINATE] = True
            return updates

        return handle_reflection

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
