"""
DebateAgent -- Multi-perspective quality improvement through structured debate.

Implements a propose -> critique -> counter -> judge loop with
configurable personas and round limits.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fsm_llm import API
from fsm_llm.handlers import HandlerTiming
from fsm_llm.logging import logger

from .base import BaseAgent, artifact_text
from .constants import (
    ContextKeys,
    DebateStates,
    Defaults,
    HandlerNames,
    HandlerPriorities,
    LogMessages,
    StopReason,
)
from .definitions import AgentConfig, AgentResult, DebateRound
from .fsm_definitions import build_debate_fsm
from .handlers import (
    make_fresh_keys_handler,
    make_iteration_limiter,
    recorded_forced_reason,
)

# Every key a debate round writes, cleared on propose entry.
_ROUND_KEYS: tuple[str, ...] = (
    ContextKeys.CONSENSUS_REACHED,
    ContextKeys.PROPOSITION,
    ContextKeys.CRITIQUE,
    ContextKeys.COUNTER_ARGUMENT,
    ContextKeys.JUDGE_VERDICT,
)

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

    Trust: ``proposer_persona``, ``critic_persona`` and ``judge_persona`` are
    developer-authored text interpolated unsanitized into the generated FSM's
    prompts, like an FSM's own ``persona``. Do not pass raw end-user input;
    put user-supplied content in the ``task`` instead.

    Usage::

        from fsm_llm.agents import DebateAgent

        agent = DebateAgent(num_rounds=3)
        result = agent.run("Should cities ban cars from their centers?")
        print(result.answer)
    """

    # Run outputs caller context may not seed (D-052 of plan 06a5ec0a).
    _run_output_keys = frozenset(
        {
            ContextKeys.PROPOSITION,
            ContextKeys.CRITIQUE,
            ContextKeys.COUNTER_ARGUMENT,
            ContextKeys.JUDGE_VERDICT,
            ContextKeys.CONSENSUS_REACHED,
            ContextKeys.DEBATE_ROUNDS,
            ContextKeys.CURRENT_ROUND,
        }
    )

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
            task_description=task,
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
                "_max_rounds": self.num_rounds,
            },
        )

        return self._standard_run(
            task,
            fsm_def,
            context,
            "debate",
            max_iterations=self._fsm_budget(),
            # Success key only; the answer is the conclude reply (see
            # _extract_answer, D-036 of plan 06a5ec0a).
            extra_answer_keys=[ContextKeys.PROPOSITION],
        )

    # DECISION plan-2026-09-29T103145-06a5ec0a/D-036
    # The answer is the terminal conclude reply, NOT a context key: do NOT
    # pass the round keys (or judge_verdict, PAT-03) to the base lookup, which
    # prefers any extra key over the reply. ``proposition`` is passed to
    # _standard_run only so a debate that produced no proposition reports
    # success=False (a reply argued from nothing is no result).
    def _extract_answer(
        self,
        final_context: dict[str, Any],
        responses: list[str],
        extra_keys: list[str] | None = None,
    ) -> str:
        """The conclude reply (no pattern answer key; ``extra_keys`` ignored)."""
        return super()._extract_answer(final_context, responses, None)

    def _register_handlers(self, api: API) -> None:
        """Register debate-specific handlers with the API."""
        # Round tracker + debate history: CONTEXT_UPDATE fires after the judge
        # state's extraction (current_state still JUDGE) and before transition
        # eval, so it records the current round's verdict and its max-round
        # consensus_reached is not overwritten by a later judge extraction.
        api.register_handler(
            api.create_handler(HandlerNames.DEBATE_JUDGE)
            .with_priority(HandlerPriorities.TOOL_EXECUTOR)
            .at(HandlerTiming.CONTEXT_UPDATE)
            .on_state(DebateStates.JUDGE)
            .do(self._make_judge_handler())
        )

        # DECISION plan-2026-09-24T091842-c1d5bfbc/D-009
        # Core extracts consensus_reached only while it is unset, so do NOT
        # seed it in run() and do NOT leave a round's False in place: the
        # judge could never extract a later True. Clear it on entry to
        # propose (the redo state), NOT on entry to judge: that would erase
        # the limiter's forced True (PRE_TRANSITION runs before entry;
        # plan-2026-09-24T045559-3e4eb3e5/D-013). The round values are
        # cleared there too (skip-if-set froze them after round 1, PAT-03);
        # the judge handler already recorded them in debate_rounds.
        api.register_handler(
            api.create_handler(HandlerNames.DEBATE_CONSENSUS_RESET)
            .with_priority(HandlerPriorities.TOOL_EXECUTOR)
            .on_state_entry(DebateStates.PROPOSE)
            .do(make_fresh_keys_handler(_ROUND_KEYS))
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
            updates: dict[str, Any] = {
                ContextKeys.DEBATE_ROUNDS: [
                    *debate_rounds,
                    round_entry.model_dump(mode="json"),
                ],
            }

            # DECISION plan-2026-09-29T103145-06a5ec0a/D-051: `proposition`
            # is the success key, and propose entry clears it every round.
            # Do NOT let a round whose proposer returned nothing erase the
            # debate's position: the last recorded proposition is restored
            # (the round entry still records this round's empty one).
            if not artifact_text(context.get(ContextKeys.PROPOSITION)).strip():
                earlier = [
                    r.get("proposition")
                    for r in debate_rounds
                    if isinstance(r, dict)
                    and artifact_text(r.get("proposition")).strip()
                ]
                if earlier:
                    updates[ContextKeys.PROPOSITION] = earlier[-1]

            # Force consensus if max rounds reached. D-051: a consensus the
            # judge did not reach itself is a forced pass (D-011), not success.
            if current_round >= num_rounds:
                if context.get(ContextKeys.CONSENSUS_REACHED) is not True:
                    updates[ContextKeys.FORCED_STOP_REASON] = StopReason.FORCED_PASS
                updates[ContextKeys.CONSENSUS_REACHED] = True

            # Increment round for next cycle
            updates[ContextKeys.CURRENT_ROUND] = current_round + 1

            return updates

        return handle_judge

    def _fsm_budget(self) -> int:
        """FSM budget: a pure function of constructor state, never stored on self.

        4 states per round (propose/critique/counter/judge) + conclude;
        multiplied by DEBATE_STATES_PER_ROUND to account for the number of
        FSM transitions each debate round requires.
        """
        return (
            self.num_rounds
            * Defaults.FSM_BUDGET_MULTIPLIER
            * Defaults.DEBATE_STATES_PER_ROUND
        )

    def _make_iteration_limiter(self) -> Callable[[dict[str, Any]], dict[str, Any]]:
        """Create the iteration limiter handler (shared rule, see handlers)."""
        # Same cap as the run's max_iterations to avoid premature termination.
        # Also set CONSENSUS_REACHED so the FSM transitions to CONCLUDE cleanly
        # (A-ISSUE-006).
        limiter = make_iteration_limiter(
            self._fsm_budget(),
            {
                ContextKeys.SHOULD_TERMINATE: True,
                ContextKeys.CONSENSUS_REACHED: True,
            },
        )

        # D-051 of plan 06a5ec0a: the limiter's consensus is a forced stop
        # unless the judge already reached one on this turn (propose entry
        # clears consensus_reached, and every forced True records a reason,
        # so an unrecorded True is the judge's own).
        def check_iteration_limit(context: dict[str, Any]) -> dict[str, Any]:
            update = limiter(context)
            if ContextKeys.CONSENSUS_REACHED not in update:
                return update
            genuine = context.get(ContextKeys.CONSENSUS_REACHED) is True
            if not genuine and recorded_forced_reason(context) is None:
                update[ContextKeys.FORCED_STOP_REASON] = StopReason.MAX_ITERATIONS
            return update

        return check_iteration_limit
