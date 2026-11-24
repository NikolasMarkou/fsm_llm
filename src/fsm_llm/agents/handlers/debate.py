"""The bodies of the Debate handlers (plan a555dcfe step 17, D-028).

Moved from ``DebateAgent``; the library factories ``agents.debate_judge``, ``agents.debate_consensus_reset``
and ``agents.debate_limiter`` (``library_stepwise.py``) wrap them. Not in ``handlers.__all__``.
Interface contract: ``make_judge_handler(num_rounds)`` and ``make_propose_entry_handler(keys)`` return
``dict -> dict`` handler bodies; ``record_forced_consensus(limiter)`` returns the limiter closure wrapped
with the forced-stop record; none raises.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from fsm_llm.agents.constants import ContextKeys, LogMessages, StopReason
from fsm_llm.agents.definitions import DebateRound
from fsm_llm.logging import logger

from .forced import recorded_forced_reason
from .freshness import make_fresh_keys_handler
from .recording import artifact_text

__all__ = [
    "ROUND_KEYS",
    "make_judge_handler",
    "make_propose_entry_handler",
    "record_forced_consensus",
]

#: Every key a debate round writes, cleared on propose entry.
ROUND_KEYS: tuple[str, ...] = (
    ContextKeys.CONSENSUS_REACHED,
    ContextKeys.PROPOSITION,
    ContextKeys.CRITIQUE,
    ContextKeys.COUNTER_ARGUMENT,
    ContextKeys.JUDGE_VERDICT,
)


def make_judge_handler(num_rounds: int) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Create the judge-exit handler: record the round, force the cap.

    Does not advance ``current_round`` (propose entry does, so a null
    judge extraction still counts the round).
    """

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
                if isinstance(r, dict) and artifact_text(r.get("proposition")).strip()
            ]
            if earlier:
                updates[ContextKeys.PROPOSITION] = earlier[-1]

        # Force consensus if max rounds reached. D-051: a consensus the
        # judge did not reach itself is a forced pass (D-011), not success.
        if current_round >= num_rounds:
            if context.get(ContextKeys.CONSENSUS_REACHED) is not True:
                updates[ContextKeys.FORCED_STOP_REASON] = StopReason.FORCED_PASS.value
            updates[ContextKeys.CONSENSUS_REACHED] = True

        return updates

    return handle_judge


def make_propose_entry_handler(
    keys: Sequence[str],
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """The propose-entry handler: clear the round keys and advance the round number."""
    # DECISION plan-2026-09-24T091842-c1d5bfbc/D-009
    # Core extracts consensus_reached only while it is unset, so do NOT
    # seed it in run() and do NOT leave a round's False in place: the
    # judge could never extract a later True. Clear it on entry to
    # propose (the redo state), NOT on entry to judge: that would erase
    # the limiter's forced True (PRE_TRANSITION runs before entry;
    # plan-2026-09-24T045559-3e4eb3e5/D-013). The round values are
    # cleared there too (skip-if-set froze them after round 1, PAT-03);
    # the judge handler already recorded them in debate_rounds. The
    # round number advances here too (plan 5ed71cf2/D-009): the round
    # is already recorded at judge exit, and this entry runs whatever
    # the judge extracted.
    clear_round = make_fresh_keys_handler(keys)

    def on_propose_entry(context: dict[str, Any]) -> dict[str, Any]:
        delta = clear_round(context)
        delta[ContextKeys.CURRENT_ROUND] = context.get(ContextKeys.CURRENT_ROUND, 1) + 1
        return delta

    return on_propose_entry


def record_forced_consensus(
    limiter: Callable[[dict[str, Any]], dict[str, Any]],
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Wrap the iteration limiter: its consensus is a forced stop unless the judge reached one."""

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
            update[ContextKeys.FORCED_STOP_REASON] = StopReason.MAX_ITERATIONS.value
        return update

    return check_iteration_limit
