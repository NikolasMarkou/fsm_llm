from __future__ import annotations

"""
DebateAgent -- Multi-perspective quality improvement through structured debate.

Implements a propose -> critique -> counter -> judge loop with
configurable personas and round limits.
"""

from typing import Any

from fsm_llm import API
from fsm_llm.logging import logger

from .base import BaseAgent
from .constants import (
    ContextKeys,
    DebateStates,
    Defaults,
    HandlerNames,
    HandlerPriorities,
    LogMessages,
)
from .definitions import AgentConfig, AgentResult, DebateRound
from .fsm_definitions import build_debate_fsm

_DEFAULT_PROPOSER_PERSONA = (
    "You are a constructive advocate who builds strong, well-reasoned arguments. "
    "Present your position clearly with supporting evidence."
)

_DEFAULT_CRITIC_PERSONA = (
    "You are a rigorous critic who identifies weaknesses and gaps in arguments. "
    "Be fair but thorough in your analysis."
)

_DEFAULT_JUDGE_PERSONA = (
    "You are an impartial judge who evaluates the quality of arguments. "
    "Determine whether a consensus has been reached or another round is needed."
)


class DebateAgent(BaseAgent):
    """
    Debate agent that improves answer quality through structured argumentation.

    Three personas (proposer, critic, judge) engage in multi-round debate
    to refine the answer. The judge decides when consensus is reached or
    the maximum number of rounds has been exhausted.

    Usage::

        from fsm_llm_agents import DebateAgent

        agent = DebateAgent(num_rounds=3)
        result = agent.run("Should cities ban cars from their centers?")
        print(result.answer)
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        num_rounds: int = Defaults.MAX_DEBATE_ROUNDS,
        proposer_persona: str = "",
        critic_persona: str = "",
        judge_persona: str = "",
        **api_kwargs: Any,
    ) -> None:
        """
        Initialize a debate agent.

        :param config: Agent configuration (defaults to AgentConfig())
        :param num_rounds: Maximum number of debate rounds
        :param proposer_persona: Persona for the proposer/counter role
        :param critic_persona: Persona for the critic role
        :param judge_persona: Persona for the judge role
        :param api_kwargs: Additional kwargs passed to fsm_llm.API
        """
        super().__init__(config, **api_kwargs)
        self.num_rounds = max(1, num_rounds)
        self.proposer_persona = proposer_persona or _DEFAULT_PROPOSER_PERSONA
        self.critic_persona = critic_persona or _DEFAULT_CRITIC_PERSONA
        self.judge_persona = judge_persona or _DEFAULT_JUDGE_PERSONA

        logger.info(
            f"DebateAgent initialized with {self.num_rounds} max rounds, model={self.config.model}"
        )

    def run(
        self,
        task: str,
        initial_context: dict[str, Any] | None = None,
    ) -> AgentResult:
        """
        Run the debate on a task.

        :param task: The task/question for the agent to debate
        :param initial_context: Optional initial context data
        :return: AgentResult with answer, trace, and metadata
        """
        # Build FSM
        fsm_def = build_debate_fsm(
            task_description=task[:Defaults.MAX_TASK_PREVIEW_LENGTH],
            proposer_persona=self.proposer_persona,
            critic_persona=self.critic_persona,
            judge_persona=self.judge_persona,
            max_rounds=self.num_rounds,
        )

        # Build initial context
        context = self._init_context(
            task,
            initial_context,
            extra={
                ContextKeys.DEBATE_ROUNDS: [],
                ContextKeys.CURRENT_ROUND: 1,
                ContextKeys.CONSENSUS_REACHED: False,
                "_max_rounds": self.num_rounds,
            },
        )

        # 4 states per round (propose/critique/counter/judge) + conclude;
        # multiplied by DEBATE_STATES_PER_ROUND to account for the
        # number of FSM transitions each debate round requires.
        max_fsm_iterations = (
            self.num_rounds * Defaults.FSM_BUDGET_MULTIPLIER * Defaults.DEBATE_STATES_PER_ROUND
        )

        return self._standard_run(
            task,
            fsm_def,
            context,
            "debate",
            max_iterations=max_fsm_iterations,
            extra_answer_keys=[ContextKeys.JUDGE_VERDICT],
        )

    def _register_handlers(self, api: API) -> None:
        """Register debate-specific handlers with the API."""
        # Round tracker + debate history: runs post-transition on judge state
        api.register_handler(
            api.create_handler(HandlerNames.DEBATE_JUDGE)
            .with_priority(HandlerPriorities.TOOL_EXECUTOR)
            .on_state_entry(DebateStates.JUDGE)
            .do(self._make_judge_handler())
        )

        # Iteration limiter
        self._register_iteration_limiter(api, self._make_iteration_limiter())

    def _make_judge_handler(self) -> Any:
        """Create a handler that tracks rounds and records debate history."""
        num_rounds = self.num_rounds

        def handle_judge(context: dict[str, Any]) -> dict[str, Any]:
            current_round = context.get(ContextKeys.CURRENT_ROUND, 1)

            logger.info(
                LogMessages.DEBATE_ROUND.format(current=current_round, max=num_rounds)
            )

            # Record this debate round
            debate_rounds = list(context.get(ContextKeys.DEBATE_ROUNDS, []))
            round_entry = DebateRound(
                round_num=current_round,
                proposition=context.get(ContextKeys.PROPOSITION, ""),
                critique=context.get(ContextKeys.CRITIQUE, ""),
                counter_argument=context.get(ContextKeys.COUNTER_ARGUMENT, ""),
                judge_verdict=context.get(ContextKeys.JUDGE_VERDICT, ""),
            )
            debate_rounds.append(round_entry.model_dump(mode="json"))

            updates: dict[str, Any] = {
                ContextKeys.DEBATE_ROUNDS: debate_rounds,
            }

            # Force consensus if max rounds reached
            if current_round >= num_rounds:
                updates[ContextKeys.CONSENSUS_REACHED] = True

            # Increment round for next cycle
            updates[ContextKeys.CURRENT_ROUND] = current_round + 1

            return updates

        return handle_judge

    def _make_iteration_limiter(self) -> Any:
        """Create an iteration limiter handler."""
        max_iters = self.num_rounds * Defaults.FSM_BUDGET_MULTIPLIER * 2

        def check_limit(context: dict[str, Any]) -> dict[str, Any]:
            count = context.get(ContextKeys.ITERATION_COUNT, 0) + 1
            result: dict[str, Any] = {ContextKeys.ITERATION_COUNT: count}
            if count >= max_iters:
                result[ContextKeys.SHOULD_TERMINATE] = True
            return result

        return check_limit
