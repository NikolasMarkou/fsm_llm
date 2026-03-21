from __future__ import annotations

"""
OrchestratorAgent — Dynamic multi-agent coordination.

Decomposes tasks into subtasks, delegates to worker agents (or internal LLM),
collects results, and synthesizes a final answer.

FSM flow: orchestrate -> delegate -> collect -> synthesize
                                             -> orchestrate (if more work needed)
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
    OrchestratorStates,
)
from .definitions import AgentConfig, AgentResult, AgentTrace
from .exceptions import AgentError, AgentTimeoutError, BudgetExhaustedError
from .fsm_definitions import build_orchestrator_fsm
from .tools import ToolRegistry


class OrchestratorAgent:
    """
    Orchestrator agent that decomposes tasks and delegates to workers.

    Combines FSM-LLM's core with a worker factory to implement the
    orchestrator-workers pattern. The orchestrator plans subtasks,
    delegates them to workers, collects results, and synthesizes.

    Usage::

        def my_worker(subtask: str) -> AgentResult:
            # ... solve subtask ...
            return AgentResult(answer="result", success=True)

        agent = OrchestratorAgent(worker_factory=my_worker)
        result = agent.run("Analyze the pros and cons of remote work")
        print(result.answer)
    """

    def __init__(
        self,
        worker_factory: Callable[[str], AgentResult] | None = None,
        tools: ToolRegistry | None = None,
        config: AgentConfig | None = None,
        max_workers: int = Defaults.MAX_WORKERS,
        **api_kwargs: Any,
    ) -> None:
        """
        Initialize an Orchestrator agent.

        :param worker_factory: Callable that takes a subtask string and returns AgentResult.
            If None, the LLM handles subtasks inline (no sub-agents spawned).
        :param tools: Optional ToolRegistry (reserved for worker_factory implementations
            that need tool access; not used directly by the orchestrator)
        :param config: Agent configuration (defaults to AgentConfig())
        :param max_workers: Maximum number of subtask delegations per round
        :param api_kwargs: Additional kwargs passed to fsm_llm.API
        """
        self.worker_factory = worker_factory
        self.tools = tools
        self.config = config or AgentConfig()
        self.max_workers = max_workers
        self._api_kwargs = api_kwargs

        logger.info(
            f"OrchestratorAgent started with max_workers={max_workers}, model={self.config.model}"
        )

    def run(
        self,
        task: str,
        initial_context: dict[str, Any] | None = None,
    ) -> AgentResult:
        """
        Run the orchestrator agent on a task.

        :param task: The task/question to solve via delegation
        :param initial_context: Optional initial context data
        :return: AgentResult with answer, trace, and metadata
        """
        start_time = time.monotonic()

        # Build FSM
        fsm_def = build_orchestrator_fsm(task_description=task[:200])

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
        context[ContextKeys.WORKER_RESULTS] = []
        context[ContextKeys.AGENT_TRACE] = []
        context[ContextKeys.ITERATION_COUNT] = 0
        context["_max_iterations"] = self.config.max_iterations

        # Start conversation
        conv_id, initial_response = api.start_conversation(context)
        log = logger.bind(
            conversation_id=conv_id, package="fsm_llm_agents", agent_type="orchestrator"
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
                if iteration > self.config.max_iterations * Defaults.FSM_BUDGET_MULTIPLIER:
                    raise BudgetExhaustedError("iterations", self.config.max_iterations)

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
                f"Orchestrator execution failed: {e}",
                details={"task": task, "iteration": iteration},
            ) from e
        finally:
            api.end_conversation(conv_id)

    def _register_handlers(self, api: API) -> None:
        """Register orchestrator handlers with the API."""
        # Worker delegator: runs on entry to 'delegate' state
        api.register_handler(
            api.create_handler(HandlerNames.ORCHESTRATOR_DELEGATOR)
            .on_state_entry(OrchestratorStates.DELEGATE)
            .do(self._delegate_to_workers)
        )

        # Iteration limiter: checks budget on every pre-transition
        api.register_handler(
            api.create_handler(HandlerNames.ITERATION_LIMITER)
            .at(HandlerTiming.PRE_TRANSITION)
            .do(self._check_iteration_limit)
        )

    def _delegate_to_workers(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Execute worker_factory for each subtask.

        Called as a POST_TRANSITION handler when entering the 'delegate' state.
        """
        subtasks = context.get(ContextKeys.SUBTASKS, [])
        if not isinstance(subtasks, list):
            subtasks = [str(subtasks)]

        # Limit to max_workers
        subtasks = subtasks[: self.max_workers]

        existing_results = context.get(ContextKeys.WORKER_RESULTS, [])
        if not isinstance(existing_results, list):
            existing_results = []

        new_results: list[dict[str, Any]] = []

        for i, subtask in enumerate(subtasks):
            subtask_str = str(subtask)
            logger.debug(f"Delegating subtask {i + 1}/{len(subtasks)}: {subtask_str[:100]}")

            if self.worker_factory is not None:
                try:
                    worker_result = self.worker_factory(subtask_str)
                    new_results.append({
                        "subtask": subtask_str,
                        "answer": worker_result.answer,
                        "success": worker_result.success,
                    })
                except Exception as e:
                    logger.warning(f"Worker failed for subtask: {e}")
                    new_results.append({
                        "subtask": subtask_str,
                        "answer": f"Worker error: {e}",
                        "success": False,
                    })
            else:
                # No worker_factory: store subtask for LLM inline processing
                new_results.append({
                    "subtask": subtask_str,
                    "answer": f"[Pending LLM processing: {subtask_str}]",
                    "success": True,
                })

        all_results = existing_results + new_results

        # Track in agent trace
        trace = context.get(ContextKeys.AGENT_TRACE, [])
        if not isinstance(trace, list):
            trace = []
        trace.append({
            "action": "delegate",
            "subtasks_delegated": len(subtasks),
            "results_count": len(new_results),
        })

        return {
            ContextKeys.WORKER_RESULTS: all_results,
            ContextKeys.AGENT_TRACE: trace,
            # Clear subtasks so orchestrate can set new ones if needed
            ContextKeys.SUBTASKS: None,
        }

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
                ContextKeys.ALL_COLLECTED: True,
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

        for response in reversed(responses):
            if response and len(response.strip()) > 5:
                return response.strip()

        return "Orchestrator could not determine an answer."
