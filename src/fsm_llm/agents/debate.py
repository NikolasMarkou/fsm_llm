"""
DebateAgent -- Multi-perspective quality improvement through structured debate.

Implements a propose -> critique -> counter -> judge loop with
configurable personas and round limits.

The machine is the pattern file ``patterns/debate.json`` run by ``PatternAgent`` (plan a555dcfe
step 17); its handlers are the library factories ``agents.debate_judge``,
``agents.debate_consensus_reset`` and ``agents.debate_limiter``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from fsm_llm.agents.constants import ContextKeys, Defaults
from fsm_llm.agents.fsm_definitions.debate import debate_texts
from fsm_llm.tools import require_int_limit

from .definitions import AgentConfig
from .handlers.library_stepwise import debate_iteration_check
from .pattern_agent import PatternAgent

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


class DebateAgent(PatternAgent):
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

    # No `tools` argument: `create_agent` raises for tools given to this pattern (`accepts_tools`).
    takes_tools: ClassVar[bool] = False

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
        # DECISION plan-2026-11-06T093029-0068b055/D-008 (12.8): refuse, never
        # clamp. ``max(1, num_rounds)`` made 0, -3 and every typo a silent 1.
        num_rounds = require_int_limit("num_rounds", num_rounds, minimum=1)
        from .pattern_files import load_envelope  # lazy: reads the file

        super().__init__(
            load_envelope("debate"),
            config=config,
            options={
                "max_rounds": num_rounds,
                "max_iterations": _fsm_budget(num_rounds),
            },
            **api_kwargs,
        )
        self.num_rounds = num_rounds
        self.proposer_persona = proposer_persona or _DEFAULT_PROPOSER_PERSONA
        self.critic_persona = critic_persona or _DEFAULT_CRITIC_PERSONA
        self.judge_persona = judge_persona or _DEFAULT_JUDGE_PERSONA

    def _text_variables(self) -> dict[str, str]:
        """The persona and the field prompts around the three personas (read at run time)."""
        return debate_texts(
            self.proposer_persona,
            self.critic_persona,
            self.judge_persona,
            self.num_rounds,
        )

    def _context_seed(self) -> dict[str, Any]:
        """The file's seed (``debate_rounds``, ``current_round``) and the round cap the prompts read."""
        return {**super()._context_seed(), ContextKeys.MAX_ROUNDS: self.num_rounds}

    # DECISION plan-2026-09-29T103145-06a5ec0a/D-036
    # The answer is the terminal conclude reply, NOT a context key: do
    # NOT pass the round keys (or judge_verdict, PAT-03) as
    # extra_answer_keys, which the base lookup prefers to the reply.
    # ``proposition`` is the success key only, so a debate that
    # produced no proposition reports success=False (a reply argued
    # from nothing is no result). Both live in the run block of
    # patterns/debate.json (no ``answer_keys``, ``success_keys`` proposition).
    def _budget_kwargs(self) -> dict[str, Any]:
        """The hard step ceiling of a run (``_fsm_budget``)."""
        return {"max_iterations": self._fsm_budget()}

    def _fsm_budget(self) -> int:
        """FSM budget: a pure function of constructor state, never stored on self."""
        return _fsm_budget(self.num_rounds)

    def _make_iteration_limiter(self) -> Callable[[dict[str, Any]], dict[str, Any]]:
        """The closure ``agents.debate_limiter`` registers for this agent (the file's binding)."""
        binding = next(
            b
            for b in self.envelope.body.handlers
            if b.handler == "agents.debate_limiter"
        )
        return debate_iteration_check(self._handler_context(), binding)


def _fsm_budget(num_rounds: int) -> int:
    """4 states per round (propose/critique/counter/judge) + conclude; multiplied by
    DEBATE_STATES_PER_ROUND to account for the FSM transitions each round requires."""
    return (
        num_rounds * Defaults.FSM_BUDGET_MULTIPLIER * Defaults.DEBATE_STATES_PER_ROUND
    )
