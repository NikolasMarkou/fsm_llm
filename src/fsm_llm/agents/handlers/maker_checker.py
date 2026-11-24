"""The bodies of the Maker-Checker handlers (plan a555dcfe step 16, D-027).

Moved from ``MakerCheckerAgent``; the library factories ``agents.revision_count``,
``agents.maker_checker_checker``, ``agents.revise_entry``, ``agents.revise_exit`` and
``agents.force_pass`` (``library.py``) wrap them, and the class keeps ``_track_revisions`` and
``_force_pass_at_limit`` as thin seams over the same functions. Not in ``handlers.__all__``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fsm_llm.agents.constants import ContextKeys, LogMessages, StopReason
from fsm_llm.logging import logger

from .freshness import make_redraft_handlers

__all__ = [
    "count_revision",
    "force_pass_at_limit",
    "make_revise_handlers",
    "track_revisions",
]


# DECISION plan-2026-11-05T134105-5ed71cf2/D-009
# revision_count advances on check ENTRY, NOT in the CONTEXT_UPDATE
# tracker below: core skips every CONTEXT_UPDATE handler on a turn
# that committed no extracted key (a checker that returns null for the
# score and the verdict), which left the count at 0 and the loop to
# max_iterations (F-AM-01). The cap itself is the check -> output
# edge's `revision_count >= max_revisions` disjunct plus the check
# PRE_TRANSITION handler (force_pass_at_limit), which records the
# forced pass; PRE_TRANSITION runs after the gate, so it cannot route.
# Do NOT move the count back into track_revisions or key it on
# extraction. Entry does not clear anything (D-013 below).
def count_revision(context: dict[str, Any]) -> dict[str, Any]:
    """Bump ``revision_count`` on entry to ``check``."""
    return {ContextKeys.REVISION_COUNT: context.get(ContextKeys.REVISION_COUNT, 0) + 1}


def track_revisions(
    context: dict[str, Any], max_revisions: int, quality_threshold: float
) -> dict[str, Any]:
    """Record the check in the trace and auto-pass on a high quality score.

    A CONTEXT_UPDATE handler on ``check`` (after its extraction, before the transition is
    evaluated), so ``checker_passed`` and ``quality_score`` are this turn's. The count is bumped by
    ``count_revision`` and the cap is enforced by the check edge and ``force_pass_at_limit``
    (plan 5ed71cf2/D-009); *max_revisions* and *quality_threshold* are the defaults for a context
    that carries neither key. A pass written here routes this turn: the gate reads the committed
    context, which holds this handler's delta (plan 5ed71cf2/D-005).
    """
    revision_count = context.get(ContextKeys.REVISION_COUNT, 0)
    # Typed float field (a dict-shaped checker_feedback is rejected by the str field).
    quality_score = context.get(ContextKeys.QUALITY_SCORE, 0.0)
    trace = list(context.get(ContextKeys.AGENT_TRACE, []))
    trace.append(
        {
            "type": "check",
            "revision_count": revision_count,
            ContextKeys.QUALITY_SCORE: quality_score,
        }
    )
    logger.info(
        LogMessages.EVALUATION.format(
            result=f"Revision {revision_count}/"
            f"{context.get(ContextKeys.MAX_REVISIONS, max_revisions)}",
            score=quality_score,
        )
    )
    result: dict[str, Any] = {ContextKeys.AGENT_TRACE: trace}
    threshold = context.get(ContextKeys.QUALITY_THRESHOLD, quality_threshold)
    if isinstance(quality_score, int | float) and quality_score >= threshold:
        logger.info(
            f"Quality score {quality_score:.2f} >= threshold {threshold:.2f}, approving"
        )
        result[ContextKeys.CHECKER_PASSED] = True
    return result


def make_revise_handlers() -> tuple[
    Callable[[dict[str, Any]], dict[str, Any]],
    Callable[[dict[str, Any]], dict[str, Any]],
]:
    """The ``(on_entry, on_exit)`` pair of ``revise``."""
    # DECISION plan-2026-09-24T045559-3e4eb3e5/D-013: re-judge every
    # round. Entering revise moves the draft aside and clears a False
    # verdict; leaving it clears the consumed feedback. Do NOT clear on
    # check entry (it would erase the limiter's forced pass) or clear
    # feedback on entry (revise must still see it). Do NOT clear a True
    # verdict: it is a forced pass (track_revisions or the limiter); kept,
    # it routes check -> output. (Before plan 5ed71cf2/D-005 the forced
    # pass lost to the extracted False on its own turn, because the
    # evaluator overlaid extracted data on context; the rule stays for a
    # pass written after the gate ran, e.g. the limiter.)
    # With a True verdict do NOT move the draft either: the judged draft
    # must ship, not an unjudged redraft (step 3.2).
    # DECISION plan-2026-09-24T045559-3e4eb3e5/D-017
    redraft_entry, on_exit = make_redraft_handlers(
        ContextKeys.DRAFT_OUTPUT,
        ContextKeys.PREVIOUS_DRAFT,
        clear_on_exit=(ContextKeys.CHECKER_FEEDBACK,),
    )

    def on_entry(context: dict[str, Any]) -> dict[str, Any]:
        if context.get(ContextKeys.CHECKER_PASSED) is True:
            return {}
        delta = redraft_entry(context)
        delta.update(
            dict.fromkeys((ContextKeys.CHECKER_PASSED, ContextKeys.QUALITY_SCORE))
        )
        return delta

    return on_entry, on_exit


# DECISION plan-2026-09-24T091842-c1d5bfbc/D-007: the budget's forced
# pass lands only on a check turn. Do NOT put checker_passed back in
# the global limiter's forced dict: at an odd budget the limit hits on
# a revise turn, the next check skips the already-set verdict (core
# extracts a key only while it is unset) and ships the unjudged
# redraft. Do NOT filter the counter by state either: it must advance
# every turn. Runs after the limiter (higher priority number) so it
# reads the flag the limiter set on this same turn.
def force_pass_at_limit(context: dict[str, Any]) -> dict[str, Any]:
    """Force ``checker_passed`` once the budget or the revision cap is spent.

    Registered on the check state only (D-007); returns ``{}`` otherwise. The cap is
    ``revision_count >= _max_revisions`` (the count is bumped on check entry, so a check turn whose
    extraction was null still counts; plan 5ed71cf2/D-009). A pass the checker (or its quality
    score) gave on this turn is not overridden, so it records no forced reason (D-051 of plan
    06a5ec0a).
    """
    if context.get(ContextKeys.CHECKER_PASSED) is True:
        return {}
    max_revisions = context.get(ContextKeys.MAX_REVISIONS)
    revisions_spent = (
        isinstance(max_revisions, int)
        and context.get(ContextKeys.REVISION_COUNT, 0) >= max_revisions
    )
    if context.get(ContextKeys.MAX_ITERATIONS_REACHED) is not True and not (
        revisions_spent
    ):
        return {}
    return {
        ContextKeys.CHECKER_PASSED: True,
        ContextKeys.FORCED_STOP_REASON: StopReason.FORCED_PASS.value,
    }
