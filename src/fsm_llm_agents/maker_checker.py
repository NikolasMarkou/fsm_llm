from __future__ import annotations

"""
MakerCheckerAgent — Two-persona quality loop agent implementation.

The maker creates content and the checker critiques it. The loop
continues until the checker approves or maximum revisions are reached.
"""

import time
from typing import Any

from fsm_llm import API
from fsm_llm.handlers import HandlerTiming
from fsm_llm.logging import logger

from .constants import (
    ContextKeys,
    Defaults,
    HandlerNames,
    LogMessages,
    MakerCheckerStates,
)
from .definitions import AgentConfig, AgentResult, AgentTrace
from .exceptions import AgentError, AgentTimeoutError, BudgetExhaustedError
from .fsm_definitions import build_maker_checker_fsm


class MakerCheckerAgent:
    """
    Agent that uses a maker-checker pattern for quality assurance.

    The maker persona generates or revises content, and the checker
    persona evaluates it against quality criteria. The loop continues
    until the checker approves or max revisions are reached.

    Usage::

        agent = MakerCheckerAgent(
            maker_instructions="Write a professional email apologizing for a delay",
            checker_instructions=(
                "Check for: professional tone, clear apology, "
                "concrete next steps, appropriate length"
            ),
            max_revisions=3,
            quality_threshold=0.7,
        )
        result = agent.run("Write an apology email to a client about a project delay")
        print(result.answer)
    """

    def __init__(
        self,
        maker_instructions: str,
        checker_instructions: str,
        config: AgentConfig | None = None,
        max_revisions: int = Defaults.MAX_REVISIONS,
        quality_threshold: float = Defaults.QUALITY_THRESHOLD,
        **api_kwargs: Any,
    ) -> None:
        """
        Initialize a Maker-Checker agent.

        :param maker_instructions: What the maker should produce
        :param checker_instructions: What the checker should evaluate
        :param config: Agent configuration (defaults to AgentConfig())
        :param max_revisions: Maximum number of revision cycles
        :param quality_threshold: Minimum quality score to pass (0.0-1.0)
        :param api_kwargs: Additional kwargs passed to fsm_llm.API
        """
        self.maker_instructions = maker_instructions
        self.checker_instructions = checker_instructions
        self.config = config or AgentConfig()
        self.max_revisions = max_revisions
        self.quality_threshold = quality_threshold
        self._api_kwargs = api_kwargs

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

        # Build FSM
        fsm_def = build_maker_checker_fsm(
            maker_instructions=self.maker_instructions,
            checker_instructions=self.checker_instructions,
            task_description=task[:200],
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
        context[ContextKeys.AGENT_TRACE] = []
        context[ContextKeys.ITERATION_COUNT] = 0
        context[ContextKeys.REVISION_COUNT] = 0
        context["_max_revisions"] = self.max_revisions
        context["_quality_threshold"] = self.quality_threshold

        # Start conversation
        conv_id, initial_response = api.start_conversation(context)
        log = logger.bind(
            conversation_id=conv_id, package="fsm_llm_agents", agent_type="maker_checker"
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

                # Hard ceiling on iterations
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
                f"Agent execution failed: {e}",
                details={"task": task, "iteration": iteration},
            ) from e
        finally:
            api.end_conversation(conv_id)

    def _register_handlers(self, api: API) -> None:
        """Register agent handlers with the API."""
        # Revision tracker: runs on entry to check state
        api.register_handler(
            api.create_handler(HandlerNames.MAKER_CHECKER_CHECKER)
            .on_state_entry(MakerCheckerStates.CHECK)
            .do(self._track_revisions)
        )

        # Iteration limiter
        api.register_handler(
            api.create_handler(HandlerNames.ITERATION_LIMITER)
            .at(HandlerTiming.PRE_TRANSITION)
            .do(self._check_iteration_limit)
        )

    def _track_revisions(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Track revision count and force pass if max revisions reached.

        Called as POST_TRANSITION handler on entry to the check state.
        Increments revision_count and forces checker_passed=True if
        the maximum number of revisions has been reached.
        """
        revision_count = context.get(ContextKeys.REVISION_COUNT, 0) + 1
        max_revisions = context.get("_max_revisions", self.max_revisions)
        quality_score = context.get("quality_score", 0.0)

        # Record in trace
        trace = list(context.get(ContextKeys.AGENT_TRACE, []))
        trace.append(
            {
                "type": "check",
                "revision_count": revision_count,
                "quality_score": quality_score,
            }
        )

        logger.info(
            LogMessages.EVALUATION.format(
                result=f"Revision {revision_count}/{max_revisions}",
                score=quality_score,
            )
        )

        result: dict[str, Any] = {
            ContextKeys.REVISION_COUNT: revision_count,
            ContextKeys.AGENT_TRACE: trace,
        }

        # Auto-pass if quality score meets threshold
        quality_threshold = context.get("_quality_threshold", self.quality_threshold)
        if isinstance(quality_score, (int, float)) and quality_score >= quality_threshold:
            logger.info(
                f"Quality score {quality_score:.2f} >= threshold {quality_threshold:.2f}, approving"
            )
            result[ContextKeys.CHECKER_PASSED] = True

        # Force checker_passed if max revisions reached
        elif revision_count >= max_revisions:
            logger.info(
                f"Max revisions ({max_revisions}) reached, forcing checker approval"
            )
            result[ContextKeys.CHECKER_PASSED] = True

        return result

    def _check_iteration_limit(self, context: dict[str, Any]) -> dict[str, Any]:
        """Check if the iteration limit has been reached."""
        iteration = context.get(ContextKeys.ITERATION_COUNT, 0) + 1

        if iteration >= self.config.max_iterations:
            return {
                ContextKeys.ITERATION_COUNT: iteration,
                ContextKeys.MAX_ITERATIONS_REACHED: True,
                ContextKeys.CHECKER_PASSED: True,
            }

        return {ContextKeys.ITERATION_COUNT: iteration}

    def _extract_answer(
        self,
        final_context: dict[str, Any],
        responses: list[str],
    ) -> str:
        """Extract the final answer from context or responses."""
        # Priority: final_answer > draft_output > last response
        answer = final_context.get(ContextKeys.FINAL_ANSWER)
        if answer and isinstance(answer, str) and len(answer) > 5:
            return answer

        draft = final_context.get(ContextKeys.DRAFT_OUTPUT)
        if draft and isinstance(draft, str) and len(draft) > 5:
            return draft

        for response in reversed(responses):
            if response and len(response.strip()) > 5:
                return response.strip()

        return "Agent could not determine an answer."
