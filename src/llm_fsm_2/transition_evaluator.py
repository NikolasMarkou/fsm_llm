"""
Transition Evaluator Module for LLM-FSM: Intelligent State Transition Resolution.

This module implements the core transition evaluation logic for the LLM-FSM
architecture. It serves as the decision engine that determines whether state transitions can
be resolved deterministically based on context and conditions, or whether they require LLM
assistance for ambiguous cases.

Architecture Role
-----------------
The TransitionEvaluator is a critical component of the processing model:

**Pass 1 (Analysis)**: Data extraction and context preparation
**Pass 2 (Evaluation)**: This module determines transition feasibility
**Pass 3 (Generation)**: Response generation based on evaluation results

This separation allows for:
- More efficient processing by avoiding unnecessary LLM calls
- Consistent transition logic independent of LLM interpretation
- Better debugging and validation of transition decisions
- Improved conversation flow predictability

Evaluation Outcomes
-------------------
The evaluator produces three distinct outcomes:

1. **DETERMINISTIC**: Single clear transition path identified
   - High confidence score (≥ minimum_confidence threshold)
   - All conditions satisfied for target transition
   - Significant confidence gap from alternatives (≥ ambiguity_threshold)
   - Results in immediate transition without LLM consultation

2. **AMBIGUOUS**: Multiple valid transition paths detected
   - Several transitions pass their conditions
   - Insufficient confidence gap between top options
   - Requires LLM assistance to select appropriate path
   - Presents curated options to LLM for decision

3. **BLOCKED**: No valid transition paths available
   - All transitions fail their required conditions
   - Context lacks necessary data for any path
   - May trigger error handling or user clarification prompts
"""

from dataclasses import dataclass
from typing import Dict, List,  Any, Set

# --------------------------------------------------------------
# local imports
# --------------------------------------------------------------

from .logging import logger
from .expressions import evaluate_logic
from .definitions import (
    State,
    Transition,
    TransitionCondition,
    TransitionOption,
    TransitionEvaluation,
    TransitionEvaluationResult,
    TransitionEvaluationError,
    FSMContext
)


# --------------------------------------------------------------
# Evaluation Configuration
# --------------------------------------------------------------

@dataclass
class TransitionEvaluatorConfig:
    """Configuration for transition evaluation behavior."""

    # Evaluation thresholds
    ambiguity_threshold: float = 0.1  # Confidence difference threshold for ambiguity
    minimum_confidence: float = 0.5  # Minimum confidence for deterministic selection

    # Evaluation modes
    enable_priority_fallback: bool = True  # Use priority when conditions are equal
    enable_llm_fallback: bool = True  # Allow LLM assistance for ambiguous cases
    strict_condition_matching: bool = True  # Require all conditions to pass

    # Performance options
    early_termination: bool = False  # Stop on first high-confidence match
    parallel_evaluation: bool = False  # Enable parallel condition evaluation

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

    def __init__(self, config: TransitionEvaluatorConfig = None):
        """Initialize transition evaluator with configuration."""
        self.config = config or TransitionEvaluatorConfig()
        logger.debug(f"TransitionEvaluator initialized with config: {self.config}")

    def evaluate_transitions(
            self,
            current_state: State,
            context: FSMContext,
            extracted_data: Dict[str, Any] = None
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
                current_state.transitions,
                working_context
            )

            # Determine evaluation result
            return self._determine_evaluation_result(
                transition_scores,
                current_state,
                working_context
            )

        except Exception as e:
            error_msg = f"Error evaluating transitions from {current_state.id}: {str(e)}"
            logger.error(error_msg)
            raise TransitionEvaluationError(error_msg)

    def _prepare_working_context(
            self,
            context: FSMContext,
            extracted_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
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
            self,
            transitions: List[Transition],
            context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
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

                # Early termination if high confidence match found
                if (self.config.early_termination and
                        score_data['confidence'] >= 0.9 and
                        score_data['passes_conditions']):
                    logger.debug(f"Early termination: high confidence match for {transition.target_state}")
                    break

            except Exception as e:
                logger.warning(f"Error evaluating transition to {transition.target_state}: {e}")
                results.append({
                    'transition': transition,
                    'confidence': 0.0,
                    'passes_conditions': False,
                    'failed_conditions': [str(e)],
                    'evaluation_notes': ['Evaluation error occurred']
                })

        # Sort by confidence and priority
        results.sort(key=lambda x: (-x['confidence'], x['transition'].priority))

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
            self,
            transition: Transition,
            context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate a single transition against current context.

        Returns:
            Dictionary with evaluation results and metadata
        """
        evaluation_result = {
            'transition': transition,
            'confidence': 0.0,
            'passes_conditions': True,
            'failed_conditions': [],
            'evaluation_notes': []
        }

        # Base confidence from priority (inverted - lower priority = higher confidence)
        base_confidence = max(0.1, 1.0 - (transition.priority / 1000.0))

        # Evaluate conditions if present
        if transition.conditions:
            condition_results = self._evaluate_transition_conditions(
                transition.conditions,
                context
            )

            evaluation_result['passes_conditions'] = condition_results['all_pass']
            evaluation_result['failed_conditions'] = condition_results['failed']
            evaluation_result['evaluation_notes'].extend(condition_results['notes'])

            # Adjust confidence based on condition results
            if condition_results['all_pass']:
                # Boost confidence for passing conditions
                confidence_boost = condition_results['confidence_factor']
                evaluation_result['confidence'] = min(1.0, base_confidence * confidence_boost)
            else:
                # Significantly reduce confidence for failed conditions
                evaluation_result['confidence'] = base_confidence * 0.1
        else:
            # No conditions - base confidence applies
            evaluation_result['confidence'] = base_confidence
            evaluation_result['evaluation_notes'].append("No conditions to evaluate")
        logger.info("evaluation_result : {}".format(evaluation_result))
        return evaluation_result

    def _evaluate_transition_conditions(
            self,
            conditions: List[TransitionCondition],
            context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate all conditions for a transition.

        Returns:
            Dictionary with condition evaluation results
        """
        result = {
            'all_pass': True,
            'failed': [],
            'notes': [],
            'confidence_factor': 1.0
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
                    result['notes'].append(f"✓ {condition.description}")
                else:
                    result['all_pass'] = False
                    result['failed'].append(condition.description)
                    result['notes'].append(f"✗ {condition.description}")

                    # Early exit if strict matching is enabled
                    if self.config.strict_condition_matching:
                        break

            except Exception as e:
                result['all_pass'] = False
                result['failed'].append(f"{condition.description} (error: {str(e)})")
                logger.warning(f"Condition evaluation error: {e}")

        # Calculate confidence factor based on condition success rate
        if total_conditions > 0:
            success_rate = passed_conditions / total_conditions
            result['confidence_factor'] = 1.0 + (success_rate * 0.5)  # Boost up to 1.5x

        logger.info(f'evaluation_result: {result}\n context: {context}')

        return result

    def _evaluate_single_condition(
            self,
            condition: TransitionCondition,
            context: Dict[str, Any]
    ) -> bool:
        """
        Evaluate a single transition condition.

        Returns:
            True if condition passes, False otherwise
        """
        # Check required context keys first
        if condition.requires_context_keys:
            missing_keys = [
                key for key in condition.requires_context_keys
                if key not in context
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
                    logger.debug(f"JsonLogic evaluation: {result} for {condition.description}")
                return bool(result)
            except Exception as e:
                logger.warning(f"JsonLogic evaluation failed for condition '{condition.description}': {e}")
                return False

        # If no logic specified, condition passes if required keys are present
        return True

    def _determine_evaluation_result(
            self,
            transition_scores: List[Dict[str, Any]],
            current_state: State,
            context: Dict[str, Any]
    ) -> TransitionEvaluation:
        """
        Determine the final evaluation result based on transition scores.

        Args:
            transition_scores: Evaluated transitions with scores
            current_state: Current state definition
            context: Working context

        Returns:
            TransitionEvaluation with result and recommendations
        """
        # Filter to only passing transitions
        passing_transitions = [
            score for score in transition_scores
            if score['passes_conditions']
        ]

        if not passing_transitions:
            return self._create_blocked_result(transition_scores, current_state)

        # Check for single clear winner
        if len(passing_transitions) == 1:
            winner = passing_transitions[0]
            if winner['confidence'] >= self.config.minimum_confidence:
                return self._create_deterministic_result(winner)

        # Check for clear confidence leader
        if len(passing_transitions) > 1:
            top_two = passing_transitions[:2]
            confidence_gap = top_two[0]['confidence'] - top_two[1]['confidence']

            if (confidence_gap > self.config.ambiguity_threshold and
                    top_two[0]['confidence'] >= self.config.minimum_confidence):
                return self._create_deterministic_result(top_two[0])

        # Multiple viable options or low confidence - create ambiguous result
        return self._create_ambiguous_result(passing_transitions, current_state)

    def _create_deterministic_result(self, winner: Dict[str, Any]) -> TransitionEvaluation:
        """Create result for deterministic transition selection."""
        logger.debug(f"Deterministic transition selected: {winner['transition'].target_state}")

        return TransitionEvaluation(
            result_type=TransitionEvaluationResult.DETERMINISTIC,
            deterministic_transition=winner['transition'].target_state,
            confidence=winner['confidence']
        )

    def _create_ambiguous_result(
            self,
            passing_transitions: List[Dict[str, Any]],
            current_state: State
    ) -> TransitionEvaluation:
        """Create result for ambiguous cases requiring LLM assistance."""
        logger.debug(f"Ambiguous transitions detected: {len(passing_transitions)} options")

        # Create transition options for LLM evaluation
        options = []
        for score_data in passing_transitions:
            transition = score_data['transition']

            # Use LLM-specific description if available, otherwise use regular description
            description = transition.llm_description or transition.description

            options.append(TransitionOption(
                target_state=transition.target_state,
                description=description,
                priority=transition.priority
            ))

        # Sort options by priority for consistent presentation
        options.sort(key=lambda opt: opt.priority)

        return TransitionEvaluation(
            result_type=TransitionEvaluationResult.AMBIGUOUS,
            available_options=options,
            confidence=max(score['confidence'] for score in passing_transitions)
        )

    def _create_blocked_result(
            self,
            all_transitions: List[Dict[str, Any]],
            current_state: State
    ) -> TransitionEvaluation:
        """Create result for blocked transitions (no valid options)."""
        logger.warning(f"No valid transitions from state: {current_state.id}")

        # Collect reasons for blocking
        blocked_reasons = []
        for score_data in all_transitions:
            if score_data['failed_conditions']:
                blocked_reasons.extend(score_data['failed_conditions'])

        reason_summary = "; ".join(blocked_reasons) if blocked_reasons else "No conditions satisfied"

        return TransitionEvaluation(
            result_type=TransitionEvaluationResult.BLOCKED,
            blocked_reason=reason_summary,
            confidence=0.0
        )

    def validate_transition_target(
            self,
            target_state: str,
            available_states: Set[str]
    ) -> bool:
        """
        Validate that a selected transition target is valid.

        Args:
            target_state: Selected target state
            available_states: Set of valid state identifiers

        Returns:
            True if target is valid, False otherwise
        """
        is_valid = target_state in available_states

        if not is_valid:
            logger.error(f"Invalid transition target '{target_state}'. Valid options: {sorted(available_states)}")

        return is_valid

    def get_evaluation_summary(
            self,
            evaluation: TransitionEvaluation
    ) -> Dict[str, Any]:
        """
        Get a summary of the evaluation result for debugging.

        Args:
            evaluation: Transition evaluation result

        Returns:
            Dictionary with evaluation summary
        """
        summary = {
            'result_type': evaluation.result_type.value,
            'confidence': evaluation.confidence
        }

        if evaluation.result_type == TransitionEvaluationResult.DETERMINISTIC:
            summary['selected_state'] = evaluation.deterministic_transition

        elif evaluation.result_type == TransitionEvaluationResult.AMBIGUOUS:
            summary['options_count'] = len(evaluation.available_options)
            summary['options'] = [opt.target_state for opt in evaluation.available_options]

        elif evaluation.result_type == TransitionEvaluationResult.BLOCKED:
            summary['blocked_reason'] = evaluation.blocked_reason

        return summary