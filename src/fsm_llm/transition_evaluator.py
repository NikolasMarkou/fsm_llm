"""
Transition Evaluator Module for FSM-LLM: Intelligent State Transition Resolution.

This module implements the core transition evaluation logic for the FSM-LLM
architecture. It serves as the decision engine that determines whether state transitions can
be resolved deterministically based on context and conditions, or whether they require LLM
assistance for ambiguous cases.

Architecture Role
-----------------
The TransitionEvaluator is a critical component of the 2-pass processing model:

**Pass 1 (Analysis & Evaluation)**: Data extraction, context preparation, and
this module determines transition feasibility
**Pass 2 (Generation)**: Response generation based on evaluation results

This separation allows for:
- More efficient processing by avoiding unnecessary LLM calls
- Consistent transition logic independent of LLM interpretation
- Better debugging and validation of transition decisions
- Improved conversation flow predictability

Evaluation Outcomes
-------------------
The evaluator produces three distinct outcomes:

1. **DETERMINISTIC**: Single clear transition path identified
   - All conditions satisfied for target transition
   - Among the transitions whose conditions pass, the one with the unique
     lowest ``priority`` value wins outright. The gap between priorities and
     the number of conditions play no part.
   - Results in immediate transition without LLM consultation

2. **AMBIGUOUS**: Multiple valid transition paths detected
   - Two or more passing transitions share the lowest ``priority`` value
   - Requires LLM assistance to select appropriate path
   - Only the tied-lowest group is offered to the classifier: a transition
     with a higher priority value can never beat a lower one

The per-transition ``confidence`` (derived from priority plus a condition-count
boost) is still computed and reported for diagnostics, but it does not decide
the outcome. ``TransitionEvaluatorConfig.minimum_confidence`` and
``ambiguity_threshold`` are kept for backward compatibility and have no effect.

3. **BLOCKED**: No valid transition paths available
   - All transitions fail their required conditions
   - Context lacks necessary data for any path
   - May trigger error handling or user clarification prompts
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .constants import (
    CONDITION_SUCCESS_RATE_BOOST,
    MIN_BASE_CONFIDENCE,
    PRIORITY_SCALING_DIVISOR,
)
from .definitions import (
    FSMContext,
    State,
    Transition,
    TransitionCondition,
    TransitionEvaluation,
    TransitionEvaluationError,
    TransitionEvaluationResult,
    TransitionOption,
)
from .expressions import evaluate_logic, get_var

# --------------------------------------------------------------
# local imports
# --------------------------------------------------------------
from .logging import logger

# --------------------------------------------------------------
# Evaluation Configuration
# --------------------------------------------------------------


@dataclass
class TransitionEvaluatorConfig:
    """Configuration for transition evaluation behavior."""

    # DECISION plan-2026-09-21T203800-8a03483a/D-003: both thresholds are
    # deprecated no-ops. Ranking is by ``priority`` alone (unique lowest wins,
    # a tie at the lowest is AMBIGUOUS). Do NOT re-wire them into the outcome
    # and do NOT delete them: existing ``TransitionEvaluatorConfig(...)``
    # callers pass them. Supersedes the earlier D-003/D-004 note that kept
    # ``minimum_confidence`` as the multi-candidate gate.
    ambiguity_threshold: float = 0.1  # Deprecated: no effect on the outcome
    minimum_confidence: float = 0.5  # Deprecated: no effect on the outcome

    # Evaluation modes
    strict_condition_matching: bool = True  # Require all conditions to pass

    # Evidence weighting
    evidence_conditions_normalizer: float = (
        5.0  # Number of conditions for max evidence weight
    )

    # Debugging
    detailed_logging: bool = False  # Enable detailed evaluation logging


# --------------------------------------------------------------
# Transition Evaluator
# --------------------------------------------------------------


class TransitionEvaluator:
    """
    Evaluates state transitions to determine if they can be resolved deterministically.

    This class implements the core logic for the 2-pass architecture's second pass,
    deciding whether transitions can be handled automatically or need LLM assistance.
    """

    def __init__(self, config: TransitionEvaluatorConfig | None = None):
        """Initialize transition evaluator with configuration."""
        self.config = config or TransitionEvaluatorConfig()
        logger.debug(f"TransitionEvaluator initialized with config: {self.config}")

    def evaluate_transitions(
        self,
        current_state: State,
        context: FSMContext,
        extracted_data: dict[str, Any] | None = None,
    ) -> TransitionEvaluation:
        """
        Evaluate all possible transitions from current state.

        Args:
            current_state: Current state definition
            context: Current FSM context
            extracted_data: Data extracted from latest user interaction

        Returns:
            TransitionEvaluation with result and recommendations
        """
        logger.debug(f"Evaluating transitions from state: {current_state.id}")

        try:
            # Merge extracted data into working context
            working_context = self._prepare_working_context(context, extracted_data)

            # Evaluate each transition
            transition_scores = self._evaluate_individual_transitions(
                current_state.transitions, working_context
            )

            # Determine evaluation result
            return self._determine_evaluation_result(
                transition_scores, current_state, working_context
            )

        except Exception as e:
            error_msg = f"Error evaluating transitions from {current_state.id}: {e!s}"
            logger.error(error_msg)
            raise TransitionEvaluationError(error_msg) from e

    def _prepare_working_context(
        self, context: FSMContext, extracted_data: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Prepare working context for transition evaluation.

        Combines existing context with newly extracted data.
        """
        working_context = context.data.copy()

        if extracted_data:
            # Log context updates
            if self.config.detailed_logging:
                logger.debug(f"Merging extracted data: {list(extracted_data.keys())}")

            working_context.update(extracted_data)

        return working_context

    def _evaluate_individual_transitions(
        self, transitions: list[Transition], context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Evaluate each transition individually and calculate scores.

        Returns:
            List of transition evaluation results with scores and metadata
        """
        if not transitions:
            logger.debug("No transitions available from current state")
            return []

        results = []

        for transition in transitions:
            try:
                score_data = self._evaluate_single_transition(transition, context)
                results.append(score_data)

            except Exception as e:
                logger.warning(
                    f"Error evaluating transition to {transition.target_state}: {e}"
                )
                results.append(
                    {
                        "transition": transition,
                        "confidence": 0.0,
                        "passes_conditions": False,
                        "failed_conditions": [str(e)],
                        "evaluation_notes": [
                            f"Evaluation error: {type(e).__name__}: {e}"
                        ],
                    }
                )

        # Rank by priority only; the sort is stable, so ties keep definition
        # order. Confidence is diagnostic and must not reorder (D-003).
        results.sort(key=lambda x: x["transition"].priority)

        if self.config.detailed_logging:
            logger.debug(f"Evaluated {len(results)} transitions")
            for result in results[:3]:  # Log top 3 results
                logger.debug(
                    f"  {result['transition'].target_state}: "
                    f"confidence={result['confidence']:.2f}, "
                    f"passes={result['passes_conditions']}"
                )

        return results

    def _evaluate_single_transition(
        self, transition: Transition, context: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Evaluate a single transition against current context.

        Returns:
            Dictionary with evaluation results and metadata
        """
        evaluation_result: dict[str, Any] = {
            "transition": transition,
            "confidence": 0.0,
            "passes_conditions": True,
            "failed_conditions": [],
            "evaluation_notes": [],
        }

        # Base confidence from priority (inverted - lower priority = higher confidence)
        base_confidence = max(
            MIN_BASE_CONFIDENCE, 1.0 - (transition.priority / PRIORITY_SCALING_DIVISOR)
        )

        # Evaluate conditions if present
        if transition.conditions:
            condition_results = self._evaluate_transition_conditions(
                transition.conditions, context
            )

            evaluation_result["passes_conditions"] = condition_results["all_pass"]
            evaluation_result["failed_conditions"] = condition_results["failed"]
            evaluation_result["evaluation_notes"].extend(condition_results["notes"])

            # Adjust confidence based on condition results
            if condition_results["all_pass"]:
                # Additive boost that preserves gradient between transitions
                # Uses diminishing returns: boost * (1 - base) so high-confidence
                # transitions still differentiate instead of all collapsing to 1.0
                boost = condition_results["confidence_factor"]
                evaluation_result["confidence"] = min(
                    1.0, base_confidence + boost * (1.0 - base_confidence)
                )
            else:
                # Significantly reduce confidence for failed conditions
                evaluation_result["confidence"] = base_confidence * 0.1
        else:
            # No conditions - base confidence applies
            evaluation_result["confidence"] = base_confidence
            evaluation_result["evaluation_notes"].append("No conditions to evaluate")
        return evaluation_result

    def _evaluate_transition_conditions(
        self, conditions: list[TransitionCondition], context: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Evaluate all conditions for a transition.

        Returns:
            Dictionary with condition evaluation results
        """
        result: dict[str, Any] = {
            "all_pass": True,
            "failed": [],
            "notes": [],
            "confidence_factor": 0.0,
        }

        passed_conditions = 0
        total_conditions = len(conditions)

        # Sort conditions by evaluation priority
        sorted_conditions = sorted(conditions, key=lambda c: c.evaluation_priority)

        for condition in sorted_conditions:
            try:
                condition_passes = self._evaluate_single_condition(condition, context)

                if condition_passes:
                    passed_conditions += 1
                    result["notes"].append(f"✓ {condition.description}")
                else:
                    result["all_pass"] = False
                    result["failed"].append(condition.description)
                    result["notes"].append(f"✗ {condition.description}")

                    # Early exit if strict matching is enabled
                    if self.config.strict_condition_matching:
                        break

            except Exception as e:
                result["all_pass"] = False
                result["failed"].append(f"{condition.description} (error: {e!s})")
                logger.warning(f"Condition evaluation error: {e}")

                if self.config.strict_condition_matching:
                    break

        # Scale confidence boost by absolute condition count to differentiate
        # transitions. More conditions passing = richer evidence = higher boost.
        # A transition with 5 passing conditions gets a higher boost than one with 1.
        if total_conditions > 0 and result["all_pass"]:
            evidence_weight = min(
                1.0, total_conditions / self.config.evidence_conditions_normalizer
            )
            result["confidence_factor"] = CONDITION_SUCCESS_RATE_BOOST * (
                0.5 + 0.5 * evidence_weight
            )

        return result

    def _evaluate_single_condition(
        self, condition: TransitionCondition, context: dict[str, Any]
    ) -> bool:
        """
        Evaluate a single transition condition.

        Returns:
            True if condition passes, False otherwise
        """
        # Check required context keys first
        if condition.requires_context_keys:
            _not_found = object()
            missing_keys = [
                key
                for key in condition.requires_context_keys
                if get_var(context, key, _not_found) is _not_found
            ]

            if missing_keys:
                if self.config.detailed_logging:
                    logger.debug(f"Condition failed: missing keys {missing_keys}")
                return False

        # Evaluate JsonLogic if present
        if condition.logic:
            try:
                result = evaluate_logic(condition.logic, context)
                if self.config.detailed_logging:
                    logger.debug(
                        f"JsonLogic evaluation: {result} for {condition.description}"
                    )
                return bool(result)
            except Exception as e:
                logger.warning(
                    f"JsonLogic evaluation failed for condition '{condition.description}': {e}"
                )
                return False

        # If no logic specified, condition passes if required keys are present
        return True

    def _determine_evaluation_result(
        self,
        transition_scores: list[dict[str, Any]],
        current_state: State,
        context: dict[str, Any],
    ) -> TransitionEvaluation:
        """
        Determine the final evaluation result based on transition scores.

        Among the passing transitions, the unique lowest ``priority`` value is
        DETERMINISTIC. Two or more tied at the lowest priority are AMBIGUOUS,
        and only that tied group becomes the classifier's candidate set. None
        passing is BLOCKED. Confidence does not take part (D-003).

        Args:
            transition_scores: Evaluated transitions with scores
            current_state: Current state definition
            context: Working context

        Returns:
            TransitionEvaluation with result and recommendations
        """
        # Filter to only passing transitions
        passing_transitions = [
            score for score in transition_scores if score["passes_conditions"]
        ]

        if not passing_transitions:
            return self._create_blocked_result(transition_scores, current_state)

        # DECISION plan-2026-09-21T203800-8a03483a/D-003: priority is decisive.
        # Do NOT rank by confidence, compare confidence gaps against
        # ``ambiguity_threshold``, or offer higher-priority-value transitions
        # to the classifier: condition count and priority gaps once inverted
        # or blurred the documented "lower priority wins" rule (audit A2).
        lowest = min(score["transition"].priority for score in passing_transitions)
        tied = [
            score
            for score in passing_transitions
            if score["transition"].priority == lowest
        ]
        if len(tied) == 1:
            return self._create_deterministic_result(tied[0])

        return self._create_ambiguous_result(tied, current_state)

    def _create_deterministic_result(
        self, winner: dict[str, Any]
    ) -> TransitionEvaluation:
        """Create result for deterministic transition selection."""
        logger.debug(
            f"Deterministic transition selected: {winner['transition'].target_state}"
        )

        return TransitionEvaluation(
            result_type=TransitionEvaluationResult.DETERMINISTIC,
            deterministic_transition=winner["transition"].target_state,
            confidence=winner["confidence"],
        )

    def _create_ambiguous_result(
        self, passing_transitions: list[dict[str, Any]], current_state: State
    ) -> TransitionEvaluation:
        """Create result for ambiguous cases requiring LLM assistance."""
        logger.debug(
            f"Ambiguous transitions detected: {len(passing_transitions)} options"
        )

        # Create transition options for LLM evaluation
        options = []
        for score_data in passing_transitions:
            transition = score_data["transition"]

            # Use LLM-specific description if available, otherwise use regular description
            description = transition.llm_description or transition.description

            options.append(
                TransitionOption(
                    target_state=transition.target_state,
                    description=description,
                    priority=transition.priority,
                )
            )

        # Sort options by priority for consistent presentation
        options.sort(key=lambda opt: opt.priority)

        return TransitionEvaluation(
            result_type=TransitionEvaluationResult.AMBIGUOUS,
            available_options=options,
            confidence=max(score["confidence"] for score in passing_transitions),
        )

    def _create_blocked_result(
        self, all_transitions: list[dict[str, Any]], current_state: State
    ) -> TransitionEvaluation:
        """Create result for blocked transitions (no valid options)."""
        logger.warning(f"No valid transitions from state: {current_state.id}")

        # Collect reasons for blocking
        blocked_reasons = []
        for score_data in all_transitions:
            if score_data["failed_conditions"]:
                blocked_reasons.extend(score_data["failed_conditions"])

        reason_summary = (
            "; ".join(blocked_reasons) if blocked_reasons else "No conditions satisfied"
        )

        return TransitionEvaluation(
            result_type=TransitionEvaluationResult.BLOCKED,
            blocked_reason=reason_summary,
            confidence=0.0,
        )
