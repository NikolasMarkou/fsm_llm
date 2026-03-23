from __future__ import annotations

"""
PlanExecuteAgent — Plan-and-Execute agent implementation.

Separates strategic planning from tactical execution:
Plan -> Execute Step -> Check Result -> Synthesize (all done)
                                      -> Replan (step failed)
                                      -> Execute Step (next step)
"""

import time
from collections.abc import Callable
from typing import Any

from fsm_llm import API
from fsm_llm.handlers import HandlerTiming
from fsm_llm.logging import logger

from .constants import (
    ContextKeys,
    Defaults,
    HandlerNames,
    LogMessages,
    PlanExecuteStates,
)
from .definitions import AgentConfig, AgentResult, AgentTrace, ToolCall
from .exceptions import AgentError, AgentTimeoutError, BudgetExhaustedError
from .fsm_definitions import build_plan_execute_fsm
from .handlers import AgentHandlers
from .tools import ToolRegistry


class PlanExecuteAgent:
    """
    Plan-and-Execute agent that separates planning from execution.

    First creates a plan, then executes each step sequentially. If a step
    fails, the agent can revise the remaining plan. When all steps are
    complete, results are synthesized into a final answer.

    Usage::

        agent = PlanExecuteAgent(tools=registry)
        result = agent.run("Compare the populations of France and Germany")
        print(result.answer)
    """

    def __init__(
        self,
        tools: ToolRegistry | None = None,
        config: AgentConfig | None = None,
        max_replans: int = Defaults.MAX_REPLANS,
        **api_kwargs: Any,
    ) -> None:
        """
        Initialize a Plan-and-Execute agent.

        :param tools: Optional tool registry (executor may use LLM only)
        :param config: Agent configuration (defaults to AgentConfig())
        :param max_replans: Maximum number of replan cycles
        :param api_kwargs: Additional kwargs passed to fsm_llm.API
        """
        self.tools = tools
        self.config = config or AgentConfig()
        self.max_replans = max_replans
        self._api_kwargs = api_kwargs
        self._handlers = AgentHandlers(tools) if tools is not None else None

        tool_count = len(tools) if tools is not None else 0
        logger.info(
            LogMessages.AGENT_STARTED.format(
                tool_count=tool_count, model=self.config.model
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

        fsm_def = build_plan_execute_fsm(self.tools, task_description=task[:200])
        api = API.from_definition(
            fsm_def,
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            **self._api_kwargs,
        )
        self._register_handlers(api)

        # Build initial context
        context: dict[str, Any] = dict(initial_context) if initial_context else {}
        context.update(
            {
                ContextKeys.TASK: task,
                ContextKeys.OBSERVATIONS: [],
                ContextKeys.AGENT_TRACE: [],
                ContextKeys.ITERATION_COUNT: 0,
                ContextKeys.PLAN_STEPS: [],
                ContextKeys.CURRENT_STEP_INDEX: 0,
                ContextKeys.STEP_RESULTS: [],
                ContextKeys.ALL_STEPS_COMPLETE: False,
                ContextKeys.STEP_FAILED: False,
                "_max_iterations": self.config.max_iterations,
                "_replan_count": 0,
            }
        )

        conv_id, initial_response = api.start_conversation(context)
        log = logger.bind(
            conversation_id=conv_id, package="fsm_llm_agents", agent_type="plan_execute"
        )

        try:
            responses = [initial_response]
            iteration = 0

            while not api.has_conversation_ended(conv_id):
                iteration += 1
                elapsed = time.monotonic() - start_time
                if elapsed > self.config.timeout_seconds:
                    raise AgentTimeoutError(self.config.timeout_seconds)
                if (
                    iteration
                    > self.config.max_iterations * Defaults.FSM_BUDGET_MULTIPLIER
                ):
                    raise BudgetExhaustedError("iterations", self.config.max_iterations)

                response = api.converse(Defaults.CONTINUE_MESSAGE, conv_id)
                responses.append(response)

            final_context = api.get_data(conv_id)
            answer = self._extract_answer(final_context, responses)
            trace = self._build_trace(final_context, iteration)

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
                f"Plan-Execute agent execution failed: {e}",
                details={"task": task, "iteration": iteration},
            ) from e
        finally:
            api.end_conversation(conv_id)

    def _register_handlers(self, api: API) -> None:
        """Register agent handlers with the API."""
        if self._handlers is not None:
            api.register_handler(
                api.create_handler(HandlerNames.TOOL_EXECUTOR)
                .on_state_entry(PlanExecuteStates.EXECUTE_STEP)
                .do(self._handlers.execute_tool)
            )

        # Iteration limiter
        api.register_handler(
            api.create_handler(HandlerNames.ITERATION_LIMITER)
            .at(HandlerTiming.PRE_TRANSITION)
            .do(self._make_iteration_limiter())
        )

        # Plan step tracker
        api.register_handler(
            api.create_handler(HandlerNames.PLAN_STEP_EXECUTOR)
            .on_state_entry(PlanExecuteStates.EXECUTE_STEP)
            .do(self._make_step_tracker())
        )

        # Step result checker
        api.register_handler(
            api.create_handler(HandlerNames.PLAN_STEP_CHECKER)
            .on_state_entry(PlanExecuteStates.CHECK_RESULT)
            .do(self._make_result_checker())
        )

        # Replan counter
        api.register_handler(
            api.create_handler("PlanReplanCounter")
            .on_state_entry(PlanExecuteStates.REPLAN)
            .do(self._make_replan_handler())
        )

    def _make_iteration_limiter(self) -> Callable[[dict[str, Any]], dict[str, Any]]:
        """Create an iteration limiter handler."""

        def check_iteration_limit(context: dict[str, Any]) -> dict[str, Any]:
            count = context.get(ContextKeys.ITERATION_COUNT, 0) + 1
            max_iters = context.get("_max_iterations", self.config.max_iterations)
            if count >= max_iters:
                return {
                    ContextKeys.ITERATION_COUNT: count,
                    ContextKeys.MAX_ITERATIONS_REACHED: True,
                    ContextKeys.SHOULD_TERMINATE: True,
                }
            return {ContextKeys.ITERATION_COUNT: count}

        return check_iteration_limit

    def _make_step_tracker(self) -> Callable[[dict[str, Any]], dict[str, Any]]:
        """Create the plan step tracking handler."""

        def track_step(context: dict[str, Any]) -> dict[str, Any]:
            plan_steps = context.get(ContextKeys.PLAN_STEPS, [])
            current_index = context.get(ContextKeys.CURRENT_STEP_INDEX, 0)
            if not isinstance(plan_steps, list) or not plan_steps:
                return {}
            if current_index < len(plan_steps):
                step_desc = plan_steps[current_index]
                total = len(plan_steps)
                logger.info(
                    LogMessages.PLAN_STEP.format(
                        current=current_index + 1,
                        total=total,
                        description=str(step_desc)[:80],
                    )
                )
                return {
                    "current_step_description": f"Step {current_index + 1}/{total}: {step_desc}"
                }
            return {}

        return track_step

    def _make_result_checker(self) -> Callable[[dict[str, Any]], dict[str, Any]]:
        """Create the step result checking handler."""

        def check_result(context: dict[str, Any]) -> dict[str, Any]:
            plan_steps = context.get(ContextKeys.PLAN_STEPS, [])
            current_index = context.get(ContextKeys.CURRENT_STEP_INDEX, 0)
            step_results = list(context.get(ContextKeys.STEP_RESULTS, []))
            step_failed = context.get(ContextKeys.STEP_FAILED, False)

            step_result = context.get("step_result", "")
            if step_result:
                step_results.append(
                    {
                        "step_index": current_index,
                        "result": str(step_result),
                        "success": not step_failed,
                    }
                )

            updates: dict[str, Any] = {
                ContextKeys.STEP_RESULTS: step_results,
                ContextKeys.STEP_FAILED: False,
            }
            if not step_failed:
                next_index = current_index + 1
                updates[ContextKeys.CURRENT_STEP_INDEX] = next_index
                if next_index >= len(plan_steps):
                    updates[ContextKeys.ALL_STEPS_COMPLETE] = True
            return updates

        return check_result

    def _make_replan_handler(self) -> Callable[[dict[str, Any]], dict[str, Any]]:
        """Create the replan counter handler."""
        max_replans = self.max_replans

        def handle_replan(context: dict[str, Any]) -> dict[str, Any]:
            replan_count = context.get("_replan_count", 0) + 1
            updates: dict[str, Any] = {
                "_replan_count": replan_count,
                ContextKeys.CURRENT_STEP_INDEX: 0,
                ContextKeys.STEP_FAILED: False,
            }
            if replan_count >= max_replans:
                updates[ContextKeys.ALL_STEPS_COMPLETE] = True
            return updates

        return handle_replan

    def _build_trace(self, final_context: dict[str, Any], iteration: int) -> AgentTrace:
        """Build agent trace from final context."""
        trace_data = final_context.get(ContextKeys.AGENT_TRACE, [])
        trace = AgentTrace(
            tool_calls=[],
            total_iterations=final_context.get(ContextKeys.ITERATION_COUNT, iteration),
        )
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
        return trace

    def _extract_answer(
        self, final_context: dict[str, Any], responses: list[str]
    ) -> str:
        """Extract the final answer from context or responses."""
        answer = final_context.get(ContextKeys.FINAL_ANSWER)
        if answer and isinstance(answer, str) and len(answer) > 5:
            return str(answer)
        for response in reversed(responses):
            if response and len(response.strip()) > 5:
                return response.strip()
        return "Agent could not determine an answer."
