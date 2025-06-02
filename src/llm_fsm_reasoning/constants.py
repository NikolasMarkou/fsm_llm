"""
Constants and enumerations for the reasoning engine.
"""
from enum import Enum


class ReasoningType(Enum):
    """Types of reasoning strategies available."""
    ANALYTICAL = "analytical"  # Breaking down complex problems
    DEDUCTIVE = "deductive"  # General to specific
    INDUCTIVE = "inductive"  # Specific to general
    ABDUCTIVE = "abductive"  # Best explanation
    ANALOGICAL = "analogical"  # Reasoning by analogy
    CREATIVE = "creative"  # Novel solutions
    CRITICAL = "critical"  # Evaluating arguments
    HYBRID = "hybrid"  # Combination of methods


class ContextKeys:
    """Standard context keys used across the reasoning engine."""
    # Problem analysis
    PROBLEM_STATEMENT = "problem_statement"
    PROBLEM_TYPE = "problem_type"
    PROBLEM_COMPONENTS = "problem_components"
    CONSTRAINTS = "constraints"

    # Strategy selection
    REASONING_STRATEGY = "reasoning_strategy"
    STRATEGY_RATIONALE = "strategy_rationale"
    CLASSIFIED_PROBLEM_TYPE = "classified_problem_type"

    # Solution synthesis
    PROPOSED_SOLUTION = "proposed_solution"
    KEY_INSIGHTS = "key_insights"
    FINAL_SOLUTION = "final_solution"

    # Validation
    VALIDATION_RESULT = "validation_result"
    CONFIDENCE_LEVEL = "confidence_level"

    # Reasoning trace
    REASONING_TRACE = "reasoning_trace"


class MergeStrategy:
    """Context merge strategies for FSM stacking."""
    UPDATE = "update"
    PRESERVE = "preserve"
    SELECTIVE = "selective"


# Default model configuration
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 2000