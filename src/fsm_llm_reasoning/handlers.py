from __future__ import annotations

"""
Handler implementations for the reasoning engine.
Enhanced with retry logic, context pruning, and standardized handling.
"""
from typing import Any
import json

from fsm_llm.logging import logger
from .constants import (
    ContextKeys, Defaults, ErrorMessages, LogMessages,
    ReasoningType
)
from .definitions import ValidationResult


class ReasoningHandlers:
    """Collection of handlers for the reasoning engine."""

    @staticmethod
    def validate_solution(context: dict[str, Any]) -> dict[str, Any]:
        """
        Validate the proposed solution with proper handling for different problem types.

        :param context: Current conversation context
        :return: Validation results with retry handling
        """
        solution = context.get(ContextKeys.PROPOSED_SOLUTION, "")
        insights = context.get(ContextKeys.KEY_INSIGHTS, [])
        problem_type = context.get(ContextKeys.PROBLEM_TYPE, "").lower()
        reasoning_strategy = context.get(ContextKeys.REASONING_STRATEGY, "")
        retry_count = context.get(ContextKeys.RETRY_COUNT, 0)

        # Determine if this is a simple problem
        is_simple_problem = any([
            "arithmetic" in problem_type,
            "calculation" in problem_type,
            reasoning_strategy == "direct computation",
            reasoning_strategy == "simple_calculator",
            context.get(ContextKeys.REASONING_TYPE_SELECTED) == ReasoningType.SIMPLE_CALCULATOR.value
        ])

        # Validation checks
        has_solution = bool(solution)
        has_insights = bool(insights) or is_simple_problem  # Simple problems may not need insights

        # Check solution detail appropriately
        if is_simple_problem:
            # For simple problems, even a single number is sufficient
            sufficient_detail = has_solution
        else:
            # For complex problems, require a substantive solution (more than a trivial answer)
            solution_str = str(solution).strip()
            sufficient_detail = has_solution and len(solution_str) > Defaults.MIN_SOLUTION_LENGTH

        # Check if solution addresses the problem (keyword overlap check)
        problem_statement = context.get(ContextKeys.PROBLEM_STATEMENT, "")
        if has_solution and problem_statement:
            problem_words = set(problem_statement.lower().split())
            solution_words = set(str(solution).lower().split())
            # At least some overlap between problem and solution terms
            addresses_problem = len(problem_words & solution_words) > 0
        else:
            addresses_problem = has_solution

        validation_checks = {
            "has_solution": has_solution,
            "has_insights": has_insights,
            "sufficient_detail": sufficient_detail,
            "addresses_problem": addresses_problem
        }

        # Calculate overall validity
        is_valid = all(validation_checks.values())

        # Handle retry logic
        max_retries_reached = retry_count >= Defaults.MAX_RETRIES
        if not is_valid and not max_retries_reached:
            retry_count += 1
            logger.info(LogMessages.RETRY_ATTEMPT.format(
                current=retry_count,
                max=Defaults.MAX_RETRIES
            ))

        # Calculate confidence
        confidence = sum(validation_checks.values()) / len(validation_checks)

        validation = ValidationResult(
            is_valid=is_valid,
            confidence=confidence,
            checks=validation_checks,
            issues=[k for k, v in validation_checks.items() if not v]
        )

        logger.info(LogMessages.VALIDATION_RESULT.format(
            valid=validation.is_valid,
            confidence=validation.confidence
        ))

        return {
            ContextKeys.RETRY_COUNT: retry_count,
            ContextKeys.VALIDATION_CHECKS: validation.checks,
            ContextKeys.SOLUTION_VALID: validation.is_valid,
            ContextKeys.VALIDATION_RESULT: validation.is_valid,
            ContextKeys.CONFIDENCE_LEVEL: validation.confidence,
            ContextKeys.SOLUTION_CONFIDENCE: validation.confidence,
            ContextKeys.MAX_RETRIES_REACHED: max_retries_reached
        }

    @staticmethod
    def update_reasoning_trace(context: dict[str, Any]) -> dict[str, Any]:
        """
        Update reasoning trace with size management.

        :param context: Current conversation context
        :return: Updated trace with pruning if needed
        """
        current_state = context.get("_current_state", "")
        previous_state = context.get("_previous_state", "")

        trace = context.get(ContextKeys.REASONING_TRACE, [])

        # Prune old traces if getting too long
        if len(trace) > Defaults.MAX_TRACE_STEPS:
            # Keep first few and last many
            trace = trace[:5] + trace[-(Defaults.MAX_TRACE_STEPS - 5):]
            logger.debug(f"Pruned reasoning trace to {len(trace)} steps")

        if previous_state and current_state:
            # Create minimal context snapshot
            snapshot_keys_to_include = [
                ContextKeys.PROBLEM_TYPE,
                ContextKeys.REASONING_STRATEGY,
                ContextKeys.REASONING_TYPE_SELECTED,
                ContextKeys.SOLUTION_VALID,
                ContextKeys.RETRY_COUNT
            ]

            context_snapshot = {
                k: v for k, v in context.items()
                if k in snapshot_keys_to_include
            }

            step = {
                "from": previous_state,
                "to": current_state,
                "context_snapshot": context_snapshot
            }

            trace.append(step)
            logger.debug(f"Added reasoning step: {previous_state} -> {current_state}")

        return {ContextKeys.REASONING_TRACE: trace}

    @staticmethod
    def prune_context(context: dict[str, Any]) -> dict[str, Any]:
        """
        Prune context to prevent explosion.

        :param context: Current context
        :return: Pruning recommendations
        """
        # Estimate context size
        context_str = json.dumps(context, default=str)
        context_size = len(context_str)

        if context_size > Defaults.CONTEXT_PRUNE_THRESHOLD:
            # Keys to always preserve
            preserve_keys = {
                ContextKeys.PROBLEM_STATEMENT,
                ContextKeys.PROBLEM_TYPE,
                ContextKeys.REASONING_STRATEGY,
                ContextKeys.PROPOSED_SOLUTION,
                ContextKeys.SOLUTION_VALID,
                ContextKeys.RETRY_COUNT
            }

            # Keys that can be pruned if large
            prune_candidates = [
                ContextKeys.REASONING_TRACE,
                ContextKeys.LOGICAL_STEPS,
                ContextKeys.OBSERVATIONS,
                ContextKeys.CREATIVE_IDEAS
            ]

            pruned_updates = {}
            for key in prune_candidates:
                if key in context and key not in preserve_keys:
                    value = context.get(key)
                    if isinstance(value, list) and len(value) > Defaults.PRUNE_LIST_MAX_LENGTH:
                        pruned_updates[key] = value[-Defaults.PRUNE_LIST_MAX_LENGTH:]
                    elif isinstance(value, str) and len(value) > Defaults.PRUNE_STRING_MAX_LENGTH:
                        pruned_updates[key] = value[:Defaults.PRUNE_STRING_MAX_LENGTH] + "...[truncated]"

            if pruned_updates:
                new_size = len(json.dumps({**context, **pruned_updates}, default=str))
                logger.info(LogMessages.CONTEXT_PRUNED.format(
                    original=context_size,
                    new=new_size
                ))
                return pruned_updates

        return {}

class ContextManager:
    """Manages context size and content."""

    @staticmethod
    def extract_relevant_context(
        source_context: dict[str, Any],
        target_keys: list[str],
        max_size: int | None = None
    ) -> dict[str, Any]:
        """
        Extract only relevant context keys.

        :param source_context: Source context
        :param target_keys: Keys to extract
        :param max_size: Maximum size limit
        :return: Filtered context
        """
        filtered = {
            k: v for k, v in source_context.items()
            if k in target_keys and v is not None
        }

        # Enforce size limit by removing lowest-priority keys until under budget
        if max_size:
            size = len(json.dumps(filtered, default=str))
            if size > max_size:
                logger.warning(f"Context size {size} exceeds limit {max_size}, truncating")
                # Remove keys in reverse order (last-added = lowest priority) until under limit
                keys_by_priority = list(filtered.keys())
                while size > max_size and keys_by_priority:
                    removed_key = keys_by_priority.pop()
                    del filtered[removed_key]
                    size = len(json.dumps(filtered, default=str))

        return filtered

    @staticmethod
    def merge_reasoning_results(
        orchestrator_context: dict[str, Any],
        sub_fsm_context: dict[str, Any],
        reasoning_type: str
    ) -> dict[str, Any]:
        """
        Merge sub-FSM results back to orchestrator with proper mapping.

        :param orchestrator_context: Main orchestrator context
        :param sub_fsm_context: Sub-FSM final context
        :param reasoning_type: Type of reasoning executed
        :return: Mapped results for orchestrator
        """
        results = {}

        # Map based on reasoning type using constants
        if reasoning_type == ReasoningType.ANALYTICAL.value:
            results[ContextKeys.KEY_INSIGHTS] = sub_fsm_context.get(ContextKeys.KEY_INSIGHTS)
            results[ContextKeys.INTEGRATED_ANALYSIS] = sub_fsm_context.get(ContextKeys.INTEGRATED_ANALYSIS)

        elif reasoning_type == ReasoningType.DEDUCTIVE.value:
            results[ContextKeys.DEDUCTIVE_CONCLUSION] = sub_fsm_context.get(ContextKeys.CONCLUSION)
            results[ContextKeys.LOGICAL_VALIDITY] = sub_fsm_context.get(ContextKeys.LOGICAL_VALIDITY)

        elif reasoning_type == ReasoningType.INDUCTIVE.value:
            results[ContextKeys.INDUCTIVE_HYPOTHESIS] = sub_fsm_context.get(ContextKeys.HYPOTHESIS)
            results[ContextKeys.GENERALIZATION_STRENGTH] = sub_fsm_context.get(ContextKeys.GENERALIZATION_STRENGTH)

        elif reasoning_type == ReasoningType.CREATIVE.value:
            results[ContextKeys.BEST_CREATIVE_SOLUTION] = sub_fsm_context.get(ContextKeys.BEST_CREATIVE_SOLUTION)
            results[ContextKeys.INNOVATION_RATING] = sub_fsm_context.get(ContextKeys.INNOVATION_RATING)

        elif reasoning_type == ReasoningType.CRITICAL.value:
            results[ContextKeys.CRITICAL_ASSESSMENT] = sub_fsm_context.get(ContextKeys.CRITICAL_ASSESSMENT)
            results[ContextKeys.ASSESSMENT_CONFIDENCE] = sub_fsm_context.get(ContextKeys.CONFIDENCE_RATING)

        elif reasoning_type == ReasoningType.SIMPLE_CALCULATOR.value:
            # Direct mapping for calculator results
            calculation_result = sub_fsm_context.get(ContextKeys.CALCULATION_RESULT)
            if calculation_result is not None:
                results[ContextKeys.CALCULATION_RESULT] = calculation_result
                # Also set as proposed solution for consistency
                results[ContextKeys.PROPOSED_SOLUTION] = calculation_result

            if ContextKeys.CALCULATION_ERROR in sub_fsm_context:
                results[ContextKeys.CALCULATION_ERROR_DETAILS] = sub_fsm_context.get(ContextKeys.CALCULATION_ERROR)

        elif reasoning_type == ReasoningType.HYBRID.value:
            results[ContextKeys.FINAL_HYBRID_SOLUTION] = sub_fsm_context.get(ContextKeys.FINAL_HYBRID_SOLUTION)
            results[ContextKeys.HYBRID_SYNTHESIS_SUMMARY] = sub_fsm_context.get(ContextKeys.REASONING_SYNTHESIS)

        elif reasoning_type == ReasoningType.ABDUCTIVE.value:
            results[ContextKeys.BEST_EXPLANATION] = sub_fsm_context.get(ContextKeys.BEST_HYPOTHESIS)
            results[ContextKeys.EXPLANATION_CONFIDENCE] = sub_fsm_context.get(ContextKeys.CONFIDENCE_IN_EXPLANATION)

        elif reasoning_type == ReasoningType.ANALOGICAL.value:
            results[ContextKeys.ANALOGICAL_SOLUTION] = sub_fsm_context.get(ContextKeys.ADAPTED_SOLUTION_OR_UNDERSTANDING)
            results[ContextKeys.ANALOGY_CONFIDENCE] = sub_fsm_context.get(ContextKeys.ANALOGY_CONFIDENCE_RATING)

        # Filter out None values
        results = {k: v for k, v in results.items() if v is not None}

        # Add completion flag
        results[f"{reasoning_type}_reasoning_completed"] = True

        return results


class OutputFormatter:
    """Formats and extracts final outputs."""

    @staticmethod
    def extract_final_solution(context: dict[str, Any]) -> str:
        """
        Extract the final solution from context with proper fallbacks.

        :param context: Final context
        :return: Formatted solution string
        """
        # Priority order for solution extraction
        solution_keys = [
            ContextKeys.FINAL_SOLUTION,
            ContextKeys.PROPOSED_SOLUTION,
            ContextKeys.CALCULATION_RESULT,
            ContextKeys.INTEGRATED_ANALYSIS,
            ContextKeys.CONCLUSION,
            ContextKeys.BEST_CREATIVE_SOLUTION,
            ContextKeys.FINAL_HYBRID_SOLUTION
        ]

        for key in solution_keys:
            solution = context.get(key)
            if solution:
                return str(solution)

        # Check if max retries were hit
        if context.get(ContextKeys.MAX_RETRIES_REACHED):
            return ErrorMessages.MAX_RETRIES_EXCEEDED

        return "Solution process completed, but no explicit solution found."

    @staticmethod
    def format_reasoning_summary(trace_info: dict[str, Any]) -> str:
        """
        Format a human-readable reasoning summary.

        :param trace_info: Reasoning trace information
        :return: Formatted summary
        """
        lines = [
            "Reasoning Summary:",
            f"- Total steps: {trace_info.get('total_steps', 0)}",
            f"- Reasoning types: {', '.join(trace_info.get('reasoning_types_used', ['unknown']))}",
            f"- Confidence: {trace_info.get('final_confidence', 0):.2%}"
        ]

        return "\n".join(lines)
