"""The body of the Evaluator-Optimizer evaluation handler (plan a555dcfe step 18, D-029).

Moved from ``EvaluatorOptimizerAgent._run_evaluation``; the library factory ``agents.evalopt_evaluator``
(``library_stepwise.py``) wraps it and the class keeps ``_run_evaluation`` as a thin seam over it. Interface
contract: ``evaluate_output(context, evaluation_fn, max_refinements)`` returns a context delta;
*max_refinements* is the default for a context that carries no ``_max_refinements``. A raising or
wrong-typed *evaluation_fn* is a failed evaluation (``run_evaluation_fn``), never an error here. Not in
``handlers.__all__``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fsm_llm.agents.constants import ContextKeys, LogMessages, StopReason
from fsm_llm.agents.definitions import EvaluationResult
from fsm_llm.logging import logger

from .recording import artifact_text
from .run_guards import run_evaluation_fn

__all__ = ["evaluate_output"]


def evaluate_output(
    context: dict[str, Any],
    evaluation_fn: Callable[[str, dict[str, Any]], EvaluationResult],
    max_refinements: int,
) -> dict[str, Any]:
    """Run the external evaluation on the generated output and record the verdict.

    A pass sets ``evaluation_passed``; a failure below the cap sets it False with the feedback and
    counts a refinement; at the cap (or a spent iteration budget) the output ships with a forced pass.
    """
    generated_output = context.get(ContextKeys.GENERATED_OUTPUT, "")
    refinement_count = context.get(ContextKeys.REFINEMENT_COUNT, 0)
    max_refinements = context.get(ContextKeys.MAX_REFINEMENTS, max_refinements)

    eval_result = run_evaluation_fn(
        evaluation_fn, artifact_text(generated_output), context
    )
    logger.info(
        LogMessages.EVALUATION.format(
            result="PASSED" if eval_result.passed else "FAILED",
            score=eval_result.score,
        )
    )

    trace = list(context.get(ContextKeys.AGENT_TRACE, []))
    trace.append(
        {
            "type": "evaluation",
            "refinement_count": refinement_count,
            "passed": eval_result.passed,
            "score": eval_result.score,
            "feedback": eval_result.feedback,
        }
    )
    verdict = eval_result.model_dump(mode="json")

    if eval_result.passed:
        return {
            ContextKeys.EVALUATION_PASSED: True,
            ContextKeys.EVALUATION_RESULT: verdict,
            ContextKeys.AGENT_TRACE: trace,
        }

    # refinement_count is 0-indexed: exactly max_refinements attempts, then the
    # (max_refinements + 1)th evaluation forces the pass.
    # DECISION plan-2026-09-24T091842-c1d5bfbc/D-022: also honour the
    # limiter's max_iterations_reached here. Do NOT rely on the limiter's
    # forced evaluation_passed alone: this entry handler overwrites it with
    # False every round, so the run never stopped and hit the 3x ceiling.
    # Forcing here ships the output just evaluated, never an unjudged one.
    at_budget = context.get(ContextKeys.MAX_ITERATIONS_REACHED) is True
    if refinement_count >= max_refinements or at_budget:
        logger.info(
            f"Max refinements ({max_refinements}) reached, "
            "forcing output with best effort"
        )
        return {
            ContextKeys.EVALUATION_PASSED: True,
            ContextKeys.EVALUATION_RESULT: verdict,
            ContextKeys.REFINEMENT_FEEDBACK: eval_result.feedback,
            ContextKeys.AGENT_TRACE: trace,
            # The output ships, but the run reports success=False (D-011).
            ContextKeys.FORCED_STOP_REASON: StopReason.FORCED_PASS.value,
        }

    return {
        ContextKeys.EVALUATION_PASSED: False,
        ContextKeys.EVALUATION_RESULT: verdict,
        ContextKeys.REFINEMENT_FEEDBACK: eval_result.feedback,
        ContextKeys.REFINEMENT_COUNT: refinement_count + 1,
        ContextKeys.AGENT_TRACE: trace,
    }
