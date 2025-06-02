"""
Handler implementations for the reasoning engine.
"""
from typing import Dict, Any


from llm_fsm.logging import logger
from .constants import ContextKeys
from .models import ValidationResult


class ReasoningHandlers:
    """Collection of handlers for the reasoning engine."""

    @staticmethod
    def validate_solution(context: Dict[str, Any]) -> Dict[str, Any]:
        solution = context.get(ContextKeys.PROPOSED_SOLUTION, "")
        insights = context.get(ContextKeys.KEY_INSIGHTS, [])
        problem_type = context.get(ContextKeys.PROBLEM_TYPE, "").lower()  # Get problem_type

        is_simple_problem = "arithmetic" in problem_type or \
                            context.get("reasoning_strategy") == "direct computation" or \
                            context.get(ContextKeys.REASONING_STRATEGY) == "simple_calculator"

        sufficient_detail_check = True
        if not solution:  # A solution must exist
            sufficient_detail_check = False
        elif not is_simple_problem and len(str(solution)) <= 10:  # Allow very short for simple, else check length
            # Changed 50 to 10 for the check, adjust as needed
            sufficient_detail_check = False

        validation_checks = {
            "has_solution": bool(solution),
            "has_insights": bool(insights),  # Check if insights exist, not just len > 0 if empty list is possible
            "sufficient_detail": sufficient_detail_check,
            "addresses_problem": True  # This likely needs more sophisticated checking
        }

        is_overall_valid = all(validation_checks.values())
        confidence = sum(validation_checks.values()) / len(validation_checks) if validation_checks else 0.0

        validation = ValidationResult(
            is_valid=is_overall_valid,
            confidence=confidence,
            checks=validation_checks,
            issues=[k for k, v in validation_checks.items() if not v]
        )

        logger.info(
            f"Solution validation: {validation.is_valid} (confidence: {validation.confidence:.2f}) "
            f"for solution '{solution}'. Problem type: '{problem_type}'. Is simple: {is_simple_problem}")

        return {
            "validation_checks": validation.checks,
            "solution_valid": validation.is_valid,  # This is what the orchestrator condition will use
            ContextKeys.VALIDATION_RESULT: validation.is_valid,  # LLM might use this based on prompt
            ContextKeys.CONFIDENCE_LEVEL: validation.confidence  # LLM might use this
        }

    @staticmethod
    def update_reasoning_trace(context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the reasoning trace with the current step.

        :param context: Current conversation context
        :return: Dictionary with updated reasoning trace
        """
        current_state = context.get("_current_state", "")
        previous_state = context.get("_previous_state", "")

        trace = context.get(ContextKeys.REASONING_TRACE, [])

        if previous_state and current_state:
            snapshot_context_keys_to_exclude = [
                ContextKeys.REASONING_TRACE,
                "reasoning_fsm_to_push"
            ]
            step = {
                "from": previous_state,
                "to": current_state,
                "context_snapshot": {
                    k: v for k, v in context.items()
                    if not k.startswith("_") and k not in snapshot_context_keys_to_exclude
                }
            }
            trace.append(step)
            logger.debug(f"Added reasoning step: {previous_state} -> {current_state}")

        return {ContextKeys.REASONING_TRACE: trace}