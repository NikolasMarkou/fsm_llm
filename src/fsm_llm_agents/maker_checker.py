from __future__ import annotations

"""
MakerCheckerAgent — Two-persona quality loop agent implementation.

The maker creates content and the checker critiques it. The loop
continues until the checker approves or maximum revisions are reached.
"""

from typing import Any

from fsm_llm import API
from fsm_llm.logging import logger

from .base import BaseAgent
from .constants import (
    ContextKeys,
    Defaults,
    HandlerNames,
    HandlerPriorities,
    LogMessages,
    MakerCheckerStates,
)
from .definitions import AgentConfig, AgentResult
from .fsm_definitions import build_maker_checker_fsm


class MakerCheckerAgent(BaseAgent):
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
        super().__init__(config, **api_kwargs)
        self.maker_instructions = maker_instructions
        self.checker_instructions = checker_instructions
        self.max_revisions = max_revisions
        self.quality_threshold = quality_threshold

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
        # Build FSM
        fsm_def = build_maker_checker_fsm(
            maker_instructions=self.maker_instructions,
            checker_instructions=self.checker_instructions,
            task_description=task[: Defaults.MAX_TASK_PREVIEW_LENGTH],
        )

        # Build initial context
        context = self._init_context(
            task,
            initial_context,
            extra={
                ContextKeys.REVISION_COUNT: 0,
                "_max_revisions": self.max_revisions,
                "_quality_threshold": self.quality_threshold,
            },
        )

        return self._standard_run(
            task,
            fsm_def,
            context,
            "maker_checker",
            extra_answer_keys=[ContextKeys.DRAFT_OUTPUT],
        )

    def _register_handlers(self, api: API) -> None:
        """Register agent handlers with the API."""
        # Revision tracker: runs on entry to check state
        api.register_handler(
            api.create_handler(HandlerNames.MAKER_CHECKER_CHECKER)
            .with_priority(HandlerPriorities.TOOL_EXECUTOR)
            .on_state_entry(MakerCheckerStates.CHECK)
            .do(self._track_revisions)
        )

        # Iteration limiter
        self._register_iteration_limiter(api, self._check_iteration_limit)

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
        if (
            isinstance(quality_score, int | float)
            and quality_score >= quality_threshold
        ):
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
