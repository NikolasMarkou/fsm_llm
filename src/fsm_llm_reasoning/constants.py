from __future__ import annotations

"""
Constants and enumerations for the reasoning engine.
All string literals and magic values are consolidated here.
"""
from enum import Enum

from fsm_llm.constants import DEFAULT_LLM_MODEL  # noqa: F401


class ReasoningType(str, Enum):
    """Types of reasoning strategies available."""
    SIMPLE_CALCULATOR = "simple_calculator"
    ANALYTICAL = "analytical"
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    CREATIVE = "creative"
    CRITICAL = "critical"
    HYBRID = "hybrid"


class OrchestratorStates:
    """States in the main orchestrator FSM.

    Note: These constants are also used as dict keys in reasoning_modes.py
    FSM definitions. Both locations must stay in sync when renaming states.
    """
    PROBLEM_ANALYSIS = "problem_analysis"
    STRATEGY_SELECTION = "strategy_selection"
    EXECUTE_REASONING = "execute_reasoning"
    SYNTHESIZE_SOLUTION = "synthesize_solution"
    VALIDATE_REFINE = "validate_refine"
    FINAL_ANSWER = "final_answer"


class ClassifierStates:
    """States in the classifier FSM."""
    ANALYZE_DOMAIN = "analyze_domain"
    ANALYZE_STRUCTURE = "analyze_structure"
    IDENTIFY_REASONING_NEEDS = "identify_reasoning_needs"
    RECOMMEND_STRATEGY = "recommend_strategy"


class ContextKeys:
    """Standard context keys used across the reasoning engine."""
    # Problem analysis
    PROBLEM_STATEMENT = "problem_statement"
    PROBLEM_TYPE = "problem_type"
    PROBLEM_COMPONENTS = "problem_components"
    CONSTRAINTS = "constraints"

    # Domain analysis
    PROBLEM_DOMAIN = "problem_domain"
    DOMAIN_INDICATORS = "domain_indicators"
    PROBLEM_STRUCTURE = "problem_structure"
    STRUCTURAL_ELEMENTS = "structural_elements"

    # Strategy selection
    REASONING_STRATEGY = "reasoning_strategy"
    STRATEGY_RATIONALE = "strategy_rationale"
    CLASSIFIED_PROBLEM_TYPE = "classified_problem_type"
    REASONING_REQUIREMENTS = "reasoning_requirements"
    KEY_CHALLENGES = "key_challenges"
    RECOMMENDED_REASONING_TYPE = "recommended_reasoning_type"
    STRATEGY_JUSTIFICATION = "strategy_justification"
    ALTERNATIVE_APPROACHES = "alternative_approaches"

    # Solution synthesis
    PROPOSED_SOLUTION = "proposed_solution"
    KEY_INSIGHTS = "key_insights"
    FINAL_SOLUTION = "final_solution"

    # Validation
    VALIDATION_RESULT = "validation_result"
    CONFIDENCE_LEVEL = "confidence_level"
    SOLUTION_VALID = "solution_valid"
    SOLUTION_CONFIDENCE = "solution_confidence"
    VALIDATION_CHECKS = "validation_checks"

    # Reasoning trace
    REASONING_TRACE = "reasoning_trace"

    # Execution control
    REASONING_FSM_TO_PUSH = "reasoning_fsm_to_push"
    REASONING_TYPE_SELECTED = "reasoning_type_selected"
    CLASSIFICATION_JUSTIFICATION = "classification_justification"
    RETRY_COUNT = "retry_count"
    MAX_RETRIES_REACHED = "max_retries_reached"

    # Simple calculator specific
    OPERAND1 = "operand1"
    OPERAND2 = "operand2"
    OPERATOR = "operator"
    CALCULATION_RESULT = "calculation_result"
    CALCULATION_ERROR = "calculation_error"

    # Analytical reasoning
    COMPONENTS = "components"
    ATTRIBUTES = "attributes"
    RELATIONSHIPS = "relationships"
    COMPONENT_ANALYSIS = "component_analysis"
    DATA_REQUIREMENTS = "data_requirements"
    PATTERNS = "patterns"
    CAUSAL_LINKS = "causal_links"
    DEPENDENCIES = "dependencies"
    INTEGRATED_ANALYSIS = "integrated_analysis"

    # Deductive reasoning
    PREMISES = "premises"
    ASSUMPTIONS = "assumptions"
    LOGICAL_STEPS = "logical_steps"
    INTERMEDIATE_CONCLUSIONS = "intermediate_conclusions"
    CONCLUSION = "conclusion"
    LOGICAL_VALIDITY = "logical_validity"

    # Inductive reasoning
    OBSERVATIONS = "observations"
    DATA_POINTS = "data_points"
    COMMONALITIES = "commonalities"
    TRENDS = "trends"
    HYPOTHESIS = "hypothesis"
    SUPPORTING_EVIDENCE = "supporting_evidence"
    TEST_RESULTS = "test_results"
    COUNTER_EXAMPLES = "counter_examples"
    GENERALIZATION_STRENGTH = "generalization_strength"

    # Creative reasoning
    PERSPECTIVES = "perspectives"
    REFRAMINGS = "reframings"
    CREATIVE_IDEAS = "creative_ideas"
    UNCONVENTIONAL_APPROACHES = "unconventional_approaches"
    COMBINATIONS = "combinations"
    NOVEL_SOLUTIONS = "novel_solutions"
    BEST_CREATIVE_SOLUTION = "best_creative_solution"
    INNOVATION_RATING = "innovation_rating"

    # Critical reasoning
    CLAIMS = "claims"
    ARGUMENTS = "arguments"
    EVIDENCE_QUALITY = "evidence_quality"
    EVIDENCE_GAPS = "evidence_gaps"
    LOGICAL_ANALYSIS = "logical_analysis"
    FALLACIES = "fallacies"
    ALTERNATIVE_EXPLANATIONS = "alternative_explanations"
    COUNTER_ARGUMENTS = "counter_arguments"
    CRITICAL_ASSESSMENT = "critical_assessment"
    CONFIDENCE_RATING = "confidence_rating"

    # Hybrid reasoning
    PROBLEM_ASPECTS = "problem_aspects"
    REASONING_MAP = "reasoning_map"
    ANALYTICAL_BREAKDOWN = "analytical_breakdown"
    COMPONENT_RELATIONSHIPS = "component_relationships"
    LOGICAL_CONCLUSIONS = "logical_conclusions"
    REASONING_CHAIN = "reasoning_chain"
    CREATIVE_INSIGHTS = "creative_insights"
    NOVEL_APPROACHES = "novel_approaches"
    EVALUATION_RESULTS = "evaluation_results"
    INTEGRATED_SOLUTION = "integrated_solution"
    REASONING_SYNTHESIS_NOTES = "reasoning_synthesis_notes"
    FINAL_HYBRID_SOLUTION = "final_hybrid_solution"
    REASONING_SYNTHESIS = "reasoning_synthesis"

    # Deductive sub-FSM result keys
    DEDUCTIVE_CONCLUSION = "deductive_conclusion"

    # Inductive sub-FSM result keys
    INDUCTIVE_HYPOTHESIS = "inductive_hypothesis"

    # Critical sub-FSM result keys
    ASSESSMENT_CONFIDENCE = "assessment_confidence"

    # Abductive reasoning
    SURPRISING_ELEMENTS = "surprising_elements"
    POTENTIAL_HYPOTHESES = "potential_hypotheses"
    HYPOTHESIS_RATIONALES = "hypothesis_rationales"
    HYPOTHESIS_EVALUATIONS = "hypothesis_evaluations"
    EVALUATION_CRITERIA = "evaluation_criteria"
    BEST_HYPOTHESIS = "best_hypothesis"
    SELECTION_JUSTIFICATION = "selection_justification"
    CONFIDENCE_IN_EXPLANATION = "confidence_in_explanation"
    NEXT_STEPS_FOR_VALIDATION = "next_steps_for_validation"

    # Abductive sub-FSM result keys
    BEST_EXPLANATION = "best_explanation"
    EXPLANATION_CONFIDENCE = "explanation_confidence"

    # Analogical reasoning
    TARGET_PROBLEM_DESCRIPTION = "target_problem_description"
    KEY_FEATURES_OF_TARGET = "key_features_of_target"
    POTENTIAL_ANALOGS = "potential_analogs"
    RATIONALE_FOR_CHOICE = "rationale_for_choice"
    SIMILARITY_CRITERIA_USED = "similarity_criteria_used"
    SELECTED_ANALOG = "selected_analog"
    ANALOGICAL_MAPPING = "analogical_mapping"
    IDENTIFIED_SIMILARITIES = "identified_similarities"
    IDENTIFIED_DIFFERENCES = "identified_differences"
    TRANSFERRED_INSIGHTS_OR_SOLUTIONS = "transferred_insights_or_solutions"
    POTENTIAL_INFERENCES = "potential_inferences"
    ANALOGY_STRENGTHS = "analogy_strengths"
    ANALOGY_WEAKNESSES_OR_LIMITATIONS = "analogy_weaknesses_or_limitations"
    ADAPTED_SOLUTION_OR_UNDERSTANDING = "adapted_solution_or_understanding"
    ANALOGY_CONFIDENCE_RATING = "analogy_confidence_rating"

    # Analogical sub-FSM result keys
    ANALOGICAL_SOLUTION = "analogical_solution"
    ANALOGY_CONFIDENCE = "analogy_confidence"

    # Hybrid sub-FSM result keys
    HYBRID_SYNTHESIS_SUMMARY = "hybrid_synthesis_summary"

    # Calculation error details
    CALCULATION_ERROR_DETAILS = "calculation_error_details"


class HandlerNames:
    """Handler names for registration."""
    ORCHESTRATOR_CLASSIFIER = "OrchestratorProblemClassifier"
    ORCHESTRATOR_EXECUTOR = "OrchestratorStrategyExecutor"
    ORCHESTRATOR_VALIDATOR = "OrchestratorSolutionValidator"
    REASONING_TRACER = "ReasoningTracer"
    CONTEXT_PRUNER = "ContextPruner"
    RETRY_LIMITER = "RetryLimiter"


class Defaults:
    """Default configuration values."""
    MODEL = DEFAULT_LLM_MODEL
    TEMPERATURE = 0.7
    MAX_TOKENS = 2000
    MAX_RETRIES = 3
    MAX_CONTEXT_SIZE = 10000  # characters
    MAX_TRACE_STEPS = 50
    CONTEXT_PRUNE_THRESHOLD = 8000  # Start pruning at 80% of max
    MIN_SOLUTION_LENGTH = 20  # Minimum chars for a substantive solution
    PRUNE_LIST_MAX_LENGTH = 10  # Keep last N items when pruning lists
    PRUNE_STRING_MAX_LENGTH = 1000  # Truncate strings beyond this length
    MAX_SUB_FSM_ITERATIONS = 30  # Limit iterations when executing a sub-FSM


class ErrorMessages:
    """Standard error messages."""
    MAX_RETRIES_EXCEEDED = "Maximum retry attempts exceeded"
    INVALID_REASONING_TYPE = "Invalid reasoning type: {type}"
    FSM_NOT_FOUND = "FSM definition not found: {name}"
    CONTEXT_TOO_LARGE = "Context size exceeds maximum allowed"
    CALCULATION_ERROR = "Calculation error: {error}"
    VALIDATION_FAILED = "Solution validation failed: {reason}"


class LogMessages:
    """Standard log message templates."""
    ENGINE_INITIALIZED = "Reasoning engine initialized with model: {model}"
    CLASSIFICATION_STARTED = "Starting classification with context: {context}"
    CLASSIFICATION_COMPLETE = "Classification complete. Recommended: {type}"
    STRATEGY_EXECUTING = "Executing {type} reasoning strategy"
    FSM_PUSHED = "Pushed FSM: {name}, stack depth: {depth}"
    FSM_POPPED = "Popped FSM: {name}, remaining depth: {depth}"
    VALIDATION_RESULT = "Validation: {valid} (confidence: {confidence})"
    RETRY_ATTEMPT = "Retry attempt {current}/{max} for validation"
    CONTEXT_PRUNED = "Pruned context from {original} to {new} characters"
    PROBLEM_SOLVED = "Problem solved in {steps} steps"
