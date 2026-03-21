from __future__ import annotations

"""
ADaPTAgent — Adaptive Decomposition and Planning for Tasks.

Attempts tasks directly first, decomposes on failure, recursion bounded by max_depth.
FSM: attempt -> assess -> combine | assess -> decompose -> [recursive run()] -> combine
"""

import time
from typing import Any

from fsm_llm import API
from fsm_llm.handlers import HandlerTiming
from fsm_llm.logging import logger

from .constants import (
    ADaPTStates,
    ContextKeys,
    Defaults,
    HandlerNames,
    LogMessages,
)
from .definitions import AgentConfig, AgentResult, AgentTrace
from .exceptions import AgentError, AgentTimeoutError, BudgetExhaustedError
from .fsm_definitions import build_adapt_fsm
from .tools import ToolRegistry


class ADaPTAgent:
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
        self.tools = tools
        self.config = config or AgentConfig()
        self.max_depth = max_depth
        self._api_kwargs = api_kwargs

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
    ) -> AgentResult:
        """Run the ADaPT agent. _depth is internal recursion tracking."""
        start_time = time.monotonic()
        logger.debug(LogMessages.DECOMPOSITION.format(depth=_depth))

        fsm_def = build_adapt_fsm(
            registry=self.tools,
            task_description=task[:200],
            max_depth=self.max_depth,
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
        context[ContextKeys.CURRENT_DEPTH] = _depth
        context[ContextKeys.SUBTASK_RESULTS] = []
        context[ContextKeys.AGENT_TRACE] = []
        context[ContextKeys.ITERATION_COUNT] = 0
        context["_max_iterations"] = self.config.max_iterations

        # Start conversation
        conv_id, initial_response = api.start_conversation(context)

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
                if iteration > self.config.max_iterations * Defaults.FSM_BUDGET_MULTIPLIER:
                    raise BudgetExhaustedError("iterations", self.config.max_iterations)

                current_context = api.get_data(conv_id)

                # Check if decomposition was triggered — handle recursion
                subtasks_raw = current_context.get(ContextKeys.SUBTASKS)
                if (
                    subtasks_raw
                    and isinstance(subtasks_raw, list)
                    and len(subtasks_raw) > 0
                    and _depth < self.max_depth
                ):
                    subtask_results = self._execute_subtasks(
                        subtasks=subtasks_raw,
                        operator=current_context.get("operator", "AND"),
                        depth=_depth + 1,
                        initial_context=initial_context,
                    )

                    # Inject subtask results back into context
                    api.update_context(conv_id, {
                        ContextKeys.SUBTASK_RESULTS: subtask_results,
                        ContextKeys.SUBTASKS: None,
                    })

                response = api.converse(Defaults.CONTINUE_MESSAGE, conv_id)
                responses.append(response)

            # Extract final results
            final_context = api.get_data(conv_id)
            answer = self._extract_answer(final_context, responses)

            # Build trace
            trace = AgentTrace(
                steps=[],
                tool_calls=[],
                total_iterations=final_context.get(
                    ContextKeys.ITERATION_COUNT, iteration
                ),
            )

            elapsed = time.monotonic() - start_time
            logger.info(
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
                f"ADaPT execution failed: {e}",
                details={"task": task, "depth": _depth, "iteration": iteration},
            ) from e
        finally:
            api.end_conversation(conv_id)

    def _execute_subtasks(
        self,
        subtasks: list[Any],
        operator: str,
        depth: int,
        initial_context: dict[str, Any] | None,
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
                )
                results.append({
                    "subtask": subtask_str,
                    "answer": sub_result.answer,
                    "success": sub_result.success,
                    "depth": depth,
                })

                # For OR operator, stop on first success
                if operator == "OR" and sub_result.success:
                    break

            except Exception as e:
                logger.warning(
                    f"ADaPT subtask failed at depth {depth}: {e}"
                )
                results.append({
                    "subtask": subtask_str,
                    "answer": f"Subtask error: {e}",
                    "success": False,
                    "depth": depth,
                })

        return results

    def _register_handlers(self, api: API) -> None:
        """Register ADaPT handlers with the API."""
        # Depth tracker: logs decomposition events
        api.register_handler(
            api.create_handler(HandlerNames.ADAPT_ASSESSOR)
            .on_state_entry(ADaPTStates.DECOMPOSE)
            .do(self._track_decomposition)
        )

        # Iteration limiter: checks budget on every pre-transition
        api.register_handler(
            api.create_handler(HandlerNames.ITERATION_LIMITER)
            .at(HandlerTiming.PRE_TRANSITION)
            .do(self._check_iteration_limit)
        )

    def _track_decomposition(self, context: dict[str, Any]) -> dict[str, Any]:
        """Track decomposition events. POST_TRANSITION on 'decompose'."""
        current_depth = context.get(ContextKeys.CURRENT_DEPTH, 0)

        logger.info(
            LogMessages.DECOMPOSITION.format(depth=current_depth)
        )

        # Track in agent trace
        trace = context.get(ContextKeys.AGENT_TRACE, [])
        if not isinstance(trace, list):
            trace = []
        trace.append({
            "action": "decompose",
            "depth": current_depth,
            "max_depth": self.max_depth,
        })

        return {ContextKeys.AGENT_TRACE: trace}

    def _check_iteration_limit(self, context: dict[str, Any]) -> dict[str, Any]:
        """Check if the iteration limit has been reached."""
        count = context.get(ContextKeys.ITERATION_COUNT, 0) + 1
        max_iterations = context.get("_max_iterations", Defaults.MAX_ITERATIONS)

        logger.debug(
            LogMessages.ITERATION.format(current=count, max=max_iterations)
        )

        if count >= max_iterations:
            return {
                ContextKeys.ITERATION_COUNT: count,
                ContextKeys.ATTEMPT_SUCCEEDED: True,
            }

        return {ContextKeys.ITERATION_COUNT: count}

    def _extract_answer(
        self,
        final_context: dict[str, Any],
        responses: list[str],
    ) -> str:
        """Extract the final answer from context or responses."""
        answer = final_context.get(ContextKeys.FINAL_ANSWER)
        if answer and isinstance(answer, str) and len(answer) > 5:
            return answer

        # Fall back to attempt_result if available
        attempt_result = final_context.get(ContextKeys.ATTEMPT_RESULT)
        if attempt_result and isinstance(attempt_result, str) and len(attempt_result) > 5:
            return attempt_result

        for response in reversed(responses):
            if response and len(response.strip()) > 5:
                return response.strip()

        return "ADaPT agent could not determine an answer."
