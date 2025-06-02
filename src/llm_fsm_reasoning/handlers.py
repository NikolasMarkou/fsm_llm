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
        """
        Validate the proposed solution for logical consistency and completeness.

        :param context: Current conversation context
        :return: Dictionary with validation results
        """
        solution = context.get(ContextKeys.PROPOSED_SOLUTION, "")
        insights = context.get(ContextKeys.KEY_INSIGHTS, [])

        # Validation checks
        validation_checks = {
            "has_solution": bool(solution),
            "has_insights": len(insights) > 0,
            "sufficient_detail": len(solution) > 50,
            "addresses_problem": True  # Would need sophisticated check
        }

        validation = ValidationResult(
            is_valid=all(validation_checks.values()),
            confidence=sum(validation_checks.values()) / len(validation_checks),
            checks=validation_checks,
            issues=[k for k, v in validation_checks.items() if not v]
        )

        logger.info(f"Solution validation: {validation.is_valid} (confidence: {validation.confidence})")

        return {
            "validation_checks": validation.checks,
            "solution_valid": validation.is_valid,
            "solution_confidence": validation.confidence
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