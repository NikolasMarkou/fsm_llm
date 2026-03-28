from __future__ import annotations

"""
ADaPTAgent — Adaptive Decomposition and Planning for Tasks.

Attempts tasks directly first, decomposes on failure, recursion bounded by max_depth.
FSM: attempt -> assess -> combine | assess -> decompose -> [recursive run()] -> combine
"""

import time
from collections.abc import Callable
from typing import Any

from fsm_llm import API
from fsm_llm.handlers import HandlerTiming
from fsm_llm.logging import logger

from .base import BaseAgent
from .constants import (
    ADaPTStates,
    ContextKeys,
    Defaults,
    HandlerNames,
    HandlerPriorities,
    LogMessages,
)
from .definitions import AgentConfig, AgentResult
from .exceptions import AgentError
from .fsm_definitions import build_adapt_fsm
from .tools import ToolRegistry


class ADaPTAgent(BaseAgent):
    """
    ADaPT agent: attempt first, decompose recursively only on failure.

    Usage::

        agent = ADaPTAgent(max_depth=3)
        result = agent.run("Explain how neural networks learn")
        print(result.answer)
    """

    def __init__(
        self,
        tools: ToolRegistry | None = None,
        config: AgentConfig | None = None,
        max_depth: int = Defaults.MAX_DECOMPOSITION_DEPTH,
        **api_kwargs: Any,
    ) -> None:
        super().__init__(config, **api_kwargs)
        self.tools = tools
        self.max_depth = max_depth

        if tools is None:
            logger.info(
                f"ADaPTAgent started in LLM-only mode (no tools), "
                f"max_depth={max_depth}, model={self.config.model}"
            )
        else:
            logger.info(
                f"ADaPTAgent started with {len(tools)} tools, "
                f"max_depth={max_depth}, model={self.config.model}"
            )

    def run(
        self,
        task: str,
        initial_context: dict[str, Any] | None = None,
        _depth: int = 0,
        _start_time: float | None = None,
    ) -> AgentResult:
        """Run the ADaPT agent. _depth is internal recursion tracking."""
        start_time = _start_time or time.monotonic()
        self._current_start_time = start_time
        logger.debug(LogMessages.DECOMPOSITION.format(depth=_depth))

        fsm_def = build_adapt_fsm(
            registry=self.tools,
            task_description=task[: Defaults.MAX_TASK_PREVIEW_LENGTH],
            max_depth=self.max_depth,
        )

        # Create API instance
        api = self._create_api(fsm_def)

        # Register handlers (needs initial_context and depth for subtask executor)
        self._register_handlers(api, initial_context, _depth)

        context = self._init_context(
            task,
            initial_context,
            extra={
                ContextKeys.CURRENT_DEPTH: _depth,
                ContextKeys.SUBTASK_RESULTS: [],
                "_max_iterations": self.config.max_iterations,
            },
        )

        try:
            responses, final_context, iteration = self._run_conversation_loop(
                api, context, start_time, "adapt"
            )

            answer = self._extract_answer(final_context, responses)
            trace = self._build_trace(final_context, iteration)

            return AgentResult(
                answer=answer,
                success=True,
                trace=trace,
                final_context=self._filter_context(final_context),
            )

        except AgentError:
            raise
        except Exception as e:
            raise AgentError(
                f"ADaPT execution failed: {e}",
                details={"task": task, "depth": _depth},
            ) from e

    def _register_handlers(
        self,
        api: API,
        initial_context: dict[str, Any] | None = None,
        depth: int = 0,
    ) -> None:
        """Register ADaPT handlers with the API."""
        # Depth tracker: logs decomposition events
        api.register_handler(
            api.create_handler(HandlerNames.ADAPT_ASSESSOR)
            .with_priority(HandlerPriorities.TOOL_EXECUTOR)
            .on_state_entry(ADaPTStates.DECOMPOSE)
            .do(self._track_decomposition)
        )

        # Subtask executor: intercepts DECOMPOSE->COMBINE transition,
        # runs recursive subtasks, and injects results before COMBINE
        api.register_handler(
            api.create_handler("subtask_executor")
            .with_priority(HandlerPriorities.TOOL_EXECUTOR)
            .at(HandlerTiming.PRE_TRANSITION)
            .on_state(ADaPTStates.DECOMPOSE)
            .do(self._make_subtask_executor(initial_context, depth))
        )

        self._register_iteration_limiter(api, self._check_iteration_limit)

    def _execute_subtasks(
        self,
        subtasks: list[Any],
        operator: str,
        depth: int,
        initial_context: dict[str, Any] | None,
        start_time: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Recursively execute subtasks via self.run(). AND=all, OR=first success.

        Recursive self.run() creates a fresh FSM + handler set per subtask,
        ensuring proper isolation. Depth is bounded by max_depth.
        """
        results: list[dict[str, Any]] = []

        for i, subtask in enumerate(subtasks):
            subtask_str = str(subtask)
            logger.debug(
                f"ADaPT subtask {i + 1}/{len(subtasks)} at depth {depth}: "
                f"{subtask_str[:100]}"
            )

            try:
                sub_result = self.run(
                    task=subtask_str,
                    initial_context=initial_context,
                    _depth=depth,
                    _start_time=start_time or time.monotonic(),
                )
                results.append(
                    {
                        "subtask": subtask_str,
                        "answer": sub_result.answer,
                        "success": sub_result.success,
                        "depth": depth,
                    }
                )

                # For OR operator, stop on first success
                if operator == "OR" and sub_result.success:
                    break

            except (
                Exception
            ) as e:  # Broad catch: subtask failures must not crash parent
                logger.warning(
                    f"ADaPT subtask failed at depth {depth}: {e}", exc_info=True
                )
                results.append(
                    {
                        "subtask": subtask_str,
                        "answer": f"Subtask error: {e}",
                        "success": False,
                        "depth": depth,
                    }
                )

        return results

    def _make_subtask_executor(
        self,
        initial_context: dict[str, Any] | None,
        depth: int,
    ) -> Callable[[dict[str, Any]], dict[str, Any]]:
        """Create handler that executes subtasks during DECOMPOSE->COMBINE transition.

        Fires as a PRE_TRANSITION handler when leaving the DECOMPOSE state.
        If subtasks were extracted, runs them recursively and injects results
        into context so the COMBINE state can synthesize them.
        """
        agent = self

        def execute_subtasks(context: dict[str, Any]) -> dict[str, Any]:
            subtasks_raw = context.get(ContextKeys.SUBTASKS)
            current_depth = context.get(ContextKeys.CURRENT_DEPTH, depth)

            if (
                not subtasks_raw
                or not isinstance(subtasks_raw, list)
                or current_depth >= agent.max_depth
            ):
                return {}

            subtask_results = agent._execute_subtasks(
                subtasks=subtasks_raw,
                operator=context.get("operator", "AND"),
                depth=current_depth + 1,
                initial_context=initial_context,
                start_time=agent._current_start_time,
            )

            return {
                ContextKeys.SUBTASK_RESULTS: subtask_results,
                ContextKeys.SUBTASKS: None,
            }

        return execute_subtasks

    def _track_decomposition(self, context: dict[str, Any]) -> dict[str, Any]:
        """Track decomposition events. POST_TRANSITION on 'decompose'."""
        current_depth = context.get(ContextKeys.CURRENT_DEPTH, 0)

        logger.info(LogMessages.DECOMPOSITION.format(depth=current_depth))

        # Track in agent trace
        trace = context.get(ContextKeys.AGENT_TRACE, [])
        if not isinstance(trace, list):
            trace = []
        trace.append(
            {
                "action": "decompose",
                "depth": current_depth,
                "max_depth": self.max_depth,
            }
        )

        return {ContextKeys.AGENT_TRACE: trace}

    def _check_iteration_limit(self, context: dict[str, Any]) -> dict[str, Any]:
        """Check if the iteration limit has been reached."""
        count = context.get(ContextKeys.ITERATION_COUNT, 0) + 1
        max_iterations = context.get("_max_iterations", Defaults.MAX_ITERATIONS)

        logger.debug(LogMessages.ITERATION.format(current=count, max=max_iterations))

        if count >= max_iterations - 1:
            return {
                ContextKeys.ITERATION_COUNT: count,
                ContextKeys.SHOULD_TERMINATE: True,
                ContextKeys.ATTEMPT_SUCCEEDED: False,
            }

        return {ContextKeys.ITERATION_COUNT: count}

    def _extract_answer(
        self,
        final_context: dict[str, Any],
        responses: list[str],
        extra_keys: list[str] | None = None,
    ) -> str:
        """Extract the final answer from context or responses."""
        answer = final_context.get(ContextKeys.FINAL_ANSWER)
        if (
            answer
            and isinstance(answer, str)
            and len(answer) > Defaults.MIN_ANSWER_LENGTH
        ):
            return str(answer)

        # Fall back to attempt_result if available
        attempt_result = final_context.get(ContextKeys.ATTEMPT_RESULT)
        if (
            attempt_result
            and isinstance(attempt_result, str)
            and len(attempt_result) > Defaults.MIN_ANSWER_LENGTH
        ):
            return str(attempt_result)

        for response in reversed(responses):
            if response and len(response.strip()) > Defaults.MIN_ANSWER_LENGTH:
                return response.strip()

        return "ADaPT agent could not determine an answer."
