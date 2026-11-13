"""Debate FSM (``build_debate_fsm``)."""

from __future__ import annotations

from typing import Any

from fsm_llm.agents.constants import ContextKeys, DebateStates
from fsm_llm.agents.prompts import (
    build_debate_conclude_response_instructions,
    build_debate_field_instructions,
)

from ._common import _by_id, _edge, _finalize_fsm, _is, _state, _when
from ._loop import _field


def build_debate_fsm(
    task_description: str = "",
    proposer_persona: str = "",
    critic_persona: str = "",
    judge_persona: str = "",
    max_rounds: int = 3,
) -> dict[str, Any]:
    """
    Build a Debate FSM definition.

    Implements a multi-round debate loop:
    propose -> critique -> counter -> judge -> propose (loop)
                                             -> conclude (consensus or max rounds)

    Each debate state extracts one typed field (judge: the verdict and the
    ``consensus_reached`` decision) whose prompt shows the task and this
    round's earlier values; state-level extraction and response instructions
    are empty, so no context-free bulk call runs and only ``conclude`` speaks.
    """
    # Use proposer persona as the top-level FSM persona since it starts
    persona = proposer_persona or (
        "You are a thoughtful AI that explores questions through structured debate. "
        "Multiple perspectives are considered to arrive at the best answer."
    )
    fields = build_debate_field_instructions(
        proposer_persona, critic_persona, judge_persona, max_rounds
    )
    round_values = (
        ContextKeys.PROPOSITION,
        ContextKeys.CRITIQUE,
        ContextKeys.COUNTER_ARGUMENT,
    )

    def _debate_field(
        name: str, extra: tuple[str, ...], *, required: bool = True
    ) -> dict[str, Any]:
        return _field(name, "str", fields[name], extra, required=required)

    states = _by_id(
        _state(
            DebateStates.PROPOSE,
            "Generate or refine a proposition for the task",
            "Present a well-reasoned argument or answer",
            [_edge(DebateStates.CRITIQUE, "Proposition ready for critique", 100)],
            fields=[
                _debate_field(ContextKeys.PROPOSITION, (ContextKeys.DEBATE_ROUNDS,))
            ],
        ),
        _state(
            DebateStates.CRITIQUE,
            "Critically analyze the current proposition",
            "Identify weaknesses, gaps, and counterpoints",
            [
                _edge(
                    DebateStates.COUNTER,
                    "Critique complete, allow counter-argument",
                    100,
                )
            ],
            fields=[_debate_field(ContextKeys.CRITIQUE, round_values[:1])],
        ),
        _state(
            DebateStates.COUNTER,
            "Address the critique with counter-arguments",
            "Strengthen the proposition by addressing criticisms",
            [_edge(DebateStates.JUDGE, "Counter-argument ready for judgment", 100)],
            fields=[_debate_field(ContextKeys.COUNTER_ARGUMENT, round_values[:2])],
        ),
        _state(
            DebateStates.JUDGE,
            "Evaluate the debate exchange and decide next action",
            "Determine whether consensus has been reached",
            [
                # DECISION plan-2026-09-24T091842-c1d5bfbc/D-009
                # Do NOT drop the round disjunct: consensus_reached
                # is now extracted every round, and before plan
                # 5ed71cf2/D-005 transition evaluation overlaid
                # this turn's extracted False on the judge
                # handler's max-round True. The cap reads
                # current_round, never an extracted value: the
                # judge's PRE_TRANSITION handler runs after this
                # gate, and a judge that extracts null must not
                # leave the cap unread. current_round advances on
                # propose entry, so at the judge of round N it is
                # N: the cap is `>=` (plan 5ed71cf2/D-009).
                _when(
                    DebateStates.CONCLUDE,
                    "Consensus reached or max rounds hit",
                    10,
                    "Consensus reached or max rounds hit",
                    {
                        "or": [
                            _is(ContextKeys.CONSENSUS_REACHED, True),
                            {">=": [{"var": ContextKeys.CURRENT_ROUND}, max_rounds]},
                        ]
                    },
                ),
                _edge(DebateStates.PROPOSE, "Another round of debate needed", 300),
            ],
            fields=[
                # The verdict is recorded, never routed on: a null costs no
                # retry call.
                _debate_field(ContextKeys.JUDGE_VERDICT, round_values, required=False),
                # Narrowed to this round's values, its number and the debate
                # history (fix 21.1): the old bool helper showed the whole
                # context, agent_trace included.
                # DECISION plan-2026-09-29T103145-06a5ec0a/D-054
                # Do NOT drop debate_rounds from this list: without the
                # earlier rounds the live judge declined consensus every
                # round (forced_pass). It is bounded by num_rounds, unlike
                # agent_trace, which must stay out.
                _field(
                    ContextKeys.CONSENSUS_REACHED,
                    "bool",
                    fields[ContextKeys.CONSENSUS_REACHED],
                    (
                        *round_values,
                        ContextKeys.CURRENT_ROUND,
                        ContextKeys.DEBATE_ROUNDS,
                    ),
                ),
            ],
        ),
        _state(
            DebateStates.CONCLUDE,
            "Produce the final answer from the debate",
            "Synthesize the debate into a definitive answer",
            response_instructions=build_debate_conclude_response_instructions(),
            # Terminal: core never extracts here; the reply is the answer.
            extraction_instructions="",
        ),
    )

    return _finalize_fsm(
        "debate_agent",
        task_description,
        "Debate agent",
        DebateStates.PROPOSE,
        persona,
        states,
    )
