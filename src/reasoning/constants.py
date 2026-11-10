"""
Constants and enumerations for the reasoning engine.
All string literals and magic values are consolidated here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from fsm_llm.constants import DEFAULT_LLM_MODEL


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
    # DECISION plan-2026-09-12T135914-45a654de/D-014
    # SOLUTION_VALID/CONFIDENCE_LEVEL were retired as duplicate aliases of
    # VALIDATION_RESULT/SOLUTION_CONFIDENCE respectively (finding 19). Do NOT
    # re-add either alias to backfill "backward compatibility" -- no external
    # consumer was found (git history, CLAUDE.md/docs, examples/reasoning/
    # math_tutor all checked); see decisions.md D-007 (PLAN) and D-014 (EXECUTE).
    VALIDATION_RESULT = "validation_result"
    SOLUTION_CONFIDENCE = "solution_confidence"
    VALIDATION_CHECKS = "validation_checks"

    # Reasoning trace
    REASONING_TRACE = "reasoning_trace"

    # Execution control
    # Set (True) by the execute_reasoning entry handler; the engine's
    # before_step hook clears it and pushes the strategy FSM of
    # REASONING_TYPE_SELECTED. Only a flag: never an FSM dict in context.
    REASONING_PUSH_PENDING = "reasoning_push_pending"
    REASONING_TYPE_SELECTED = "reasoning_type_selected"
    CLASSIFICATION_JUSTIFICATION = "classification_justification"
    RETRY_COUNT = "retry_count"
    MAX_RETRIES_REACHED = "max_retries_reached"

    # Hybrid strategy loop control (the critical_evaluation back edge)
    NEEDS_REFINEMENT = "needs_refinement"
    HYBRID_LOOP_COUNT = "hybrid_loop_count"

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

    # CLI --type flag: set in initial_context to override automatic type selection
    PREFERRED_REASONING_TYPE = "preferred_reasoning_type"


@dataclass(frozen=True)
class StrategySpec:
    """Everything the package knows about one reasoning strategy.

    Contract: ``description`` is the line ``--list-types`` shows; ``aliases``
    are further lowercase spellings ``map_reasoning_type`` accepts;
    ``results`` maps each orchestrator key the strategy writes back to the
    sub-FSM key it is read from (``merge_reasoning_results`` copies a value
    only when it is not ``None``); ``solution_key`` is the orchestrator key
    ``extract_final_solution`` reads for this strategy.
    """

    description: str
    aliases: tuple[str, ...]
    results: dict[str, str]
    solution_key: str


def _same(*keys: str) -> dict[str, str]:
    return {key: key for key in keys}


# The one table of strategies, keyed by ``ReasoningType`` value. A new strategy
# is an enum member, an FSM in ``reasoning_modes.ALL_REASONING_FSMS`` and one
# entry here. The order is the order ``extract_final_solution`` reads the
# solution keys after ``final_solution`` and ``proposed_solution``.
STRATEGIES: dict[str, StrategySpec] = {
    ReasoningType.SIMPLE_CALCULATOR.value: StrategySpec(
        "Direct arithmetic calculations",
        (
            "direct computation",
            "direct calculation",
            "arithmetic",
            "calculation",
            "math",
            "calculator",
            "compute",
            "numeric",
        ),
        {
            ContextKeys.CALCULATION_RESULT: ContextKeys.CALCULATION_RESULT,
            ContextKeys.PROPOSED_SOLUTION: ContextKeys.CALCULATION_RESULT,
            ContextKeys.CALCULATION_ERROR_DETAILS: ContextKeys.CALCULATION_ERROR,
        },
        ContextKeys.CALCULATION_RESULT,
    ),
    ReasoningType.ANALYTICAL.value: StrategySpec(
        "Breaking down complex problems into components",
        ("analyze", "analysis"),
        _same(ContextKeys.KEY_INSIGHTS, ContextKeys.INTEGRATED_ANALYSIS),
        ContextKeys.INTEGRATED_ANALYSIS,
    ),
    ReasoningType.DEDUCTIVE.value: StrategySpec(
        "Deriving specific conclusions from general principles",
        ("deduce", "deduction", "logic", "logical", "syllogism"),
        {
            ContextKeys.DEDUCTIVE_CONCLUSION: ContextKeys.CONCLUSION,
            ContextKeys.LOGICAL_VALIDITY: ContextKeys.LOGICAL_VALIDITY,
        },
        ContextKeys.DEDUCTIVE_CONCLUSION,
    ),
    ReasoningType.INDUCTIVE.value: StrategySpec(
        "Finding patterns from specific observations",
        ("induce", "induction", "pattern", "observation", "generalize"),
        {
            ContextKeys.INDUCTIVE_HYPOTHESIS: ContextKeys.HYPOTHESIS,
            ContextKeys.GENERALIZATION_STRENGTH: ContextKeys.GENERALIZATION_STRENGTH,
        },
        ContextKeys.INDUCTIVE_HYPOTHESIS,
    ),
    ReasoningType.CREATIVE.value: StrategySpec(
        "Generating novel solutions through divergent thinking",
        ("create", "innovation", "innovative", "brainstorm", "divergent"),
        _same(ContextKeys.BEST_CREATIVE_SOLUTION, ContextKeys.INNOVATION_RATING),
        ContextKeys.BEST_CREATIVE_SOLUTION,
    ),
    ReasoningType.CRITICAL.value: StrategySpec(
        "Evaluating arguments and evidence",
        ("critique", "evaluation", "evaluate", "assessment", "judge"),
        {
            ContextKeys.CRITICAL_ASSESSMENT: ContextKeys.CRITICAL_ASSESSMENT,
            ContextKeys.ASSESSMENT_CONFIDENCE: ContextKeys.CONFIDENCE_RATING,
        },
        ContextKeys.CRITICAL_ASSESSMENT,
    ),
    ReasoningType.HYBRID.value: StrategySpec(
        "Combining multiple reasoning approaches",
        ("mixed", "combined", "multi"),
        {
            ContextKeys.FINAL_HYBRID_SOLUTION: ContextKeys.FINAL_HYBRID_SOLUTION,
            ContextKeys.HYBRID_SYNTHESIS_SUMMARY: ContextKeys.REASONING_SYNTHESIS,
        },
        ContextKeys.FINAL_HYBRID_SOLUTION,
    ),
    ReasoningType.ABDUCTIVE.value: StrategySpec(
        "Finding the best explanation for observations",
        ("explain", "explanation", "hypothesis", "abduction", "diagnose"),
        {
            ContextKeys.BEST_EXPLANATION: ContextKeys.BEST_HYPOTHESIS,
            ContextKeys.EXPLANATION_CONFIDENCE: ContextKeys.CONFIDENCE_IN_EXPLANATION,
        },
        ContextKeys.BEST_EXPLANATION,
    ),
    ReasoningType.ANALOGICAL.value: StrategySpec(
        "Transferring insights through analogies",
        ("analogy", "comparison", "compare", "similar"),
        {
            ContextKeys.ANALOGICAL_SOLUTION: (
                ContextKeys.ADAPTED_SOLUTION_OR_UNDERSTANDING
            ),
            ContextKeys.ANALOGY_CONFIDENCE: ContextKeys.ANALOGY_CONFIDENCE_RATING,
        },
        ContextKeys.ANALOGICAL_SOLUTION,
    ),
}


# Keys cleared on every entry to execute_reasoning. Core extracts a key only
# while it is unset, so without the clear a retry would keep the rejected
# solution and never re-run synthesis or validation (the retry loop stalls).
RETRY_CLEARED_KEYS: tuple[str, ...] = (
    ContextKeys.PROPOSED_SOLUTION,
    ContextKeys.KEY_INSIGHTS,
    ContextKeys.VALIDATION_RESULT,
)


# Orchestrator keys only handlers, update_context and initial_context write
# (core ``handler_only_keys``): no extraction, bulk or per-field, may set them,
# so a model reply can never open the validation gate or pick the strategy.
ORCHESTRATOR_HANDLER_ONLY_KEYS: tuple[str, ...] = (
    ContextKeys.VALIDATION_RESULT,
    ContextKeys.VALIDATION_CHECKS,
    ContextKeys.MAX_RETRIES_REACHED,
    ContextKeys.RETRY_COUNT,
    ContextKeys.SOLUTION_CONFIDENCE,
    ContextKeys.REASONING_TYPE_SELECTED,
    ContextKeys.CLASSIFIED_PROBLEM_TYPE,
    ContextKeys.REASONING_PUSH_PENDING,
)

# Keys only the engine's own handlers write to choose and push the strategy
# (``_classify_problem``, ``_StrategyStack``): ``solve_problem`` drops them
# from a caller's ``initial_context``, so a caller cannot push a strategy
# before the orchestrator has chosen one, nor set a classification. This
# does NOT make a ``final_context`` passed back in start fresh: a carried
# ``problem_type`` keeps the classifier from running and a carried
# ``reasoning_strategy`` then picks the strategy (Known open, D-061).
SOLVE_DRIVER_KEYS: tuple[str, ...] = (
    ContextKeys.REASONING_PUSH_PENDING,
    ContextKeys.REASONING_TYPE_SELECTED,
    ContextKeys.CLASSIFIED_PROBLEM_TYPE,
)

# The hybrid strategy state whose exit the loop counter handler counts; its
# back edge to the first hybrid state runs at most Defaults.MAX_HYBRID_LOOPS
# times.
HYBRID_EVALUATION_STATE = "critical_evaluation"

# The first hybrid state, where the critical_evaluation back edge re-enters.
HYBRID_FIRST_STATE = "identify_components"

# Keys the looped hybrid states extract, plus the refinement flag. Core
# extracts a key only while it is unset, so without clearing them on every
# entry to HYBRID_FIRST_STATE the back edge re-runs no extraction and the
# flag stays True. ``hybrid_loop_count`` is NOT here: it is handler-only and
# must survive the loop (D-054). Keys built after the loop
# (``integrated_solution``, ``final_hybrid_solution``) are not here either.
HYBRID_CLEARED_KEYS: tuple[str, ...] = (
    ContextKeys.PROBLEM_ASPECTS,
    ContextKeys.REASONING_MAP,
    ContextKeys.ANALYTICAL_BREAKDOWN,
    ContextKeys.COMPONENT_RELATIONSHIPS,
    ContextKeys.LOGICAL_CONCLUSIONS,
    ContextKeys.REASONING_CHAIN,
    ContextKeys.CREATIVE_INSIGHTS,
    ContextKeys.NOVEL_APPROACHES,
    ContextKeys.EVALUATION_RESULTS,
    ContextKeys.NEEDS_REFINEMENT,
)


class HandlerNames:
    """Handler names for registration."""

    ORCHESTRATOR_CLASSIFIER = "OrchestratorProblemClassifier"
    ORCHESTRATOR_EXECUTOR = "OrchestratorStrategyExecutor"
    ORCHESTRATOR_VALIDATOR = "OrchestratorSolutionValidator"
    REASONING_TRACER = "ReasoningTracer"
    CONTEXT_PRUNER = "ContextPruner"
    RETRY_LIMITER = "RetryLimiter"
    RETRY_KEY_CLEARER = "RetryKeyClearer"
    HYBRID_LOOP_COUNTER = "HybridLoopCounter"
    HYBRID_KEY_CLEARER = "HybridKeyClearer"


class Defaults:
    """Default configuration values."""

    MODEL = DEFAULT_LLM_MODEL
    TEMPERATURE = 0.7
    MAX_TOKENS = 2000
    MAX_RETRIES = 3
    MAX_TRACE_STEPS = 50
    TRACE_KEEP_START = 5  # Oldest trace steps kept when the trace is pruned
    CONTEXT_PRUNE_THRESHOLD = 8000  # characters; prune the context above this
    MIN_SOLUTION_LENGTH = 20  # Minimum chars for a substantive solution
    PRUNE_LIST_MAX_LENGTH = 10  # Keep last N items when pruning lists
    PRUNE_STRING_MAX_LENGTH = 1000  # Truncate strings beyond this length
    MAX_SUB_FSM_ITERATIONS = 30  # Steps of a strategy FSM before a forced pop
    MAX_CLASSIFICATION_ITERATIONS = 10  # Steps budget of the classifier FSM run
    MAX_TOTAL_ITERATIONS = 50  # Orchestrator steps of one solve
    MAX_HYBRID_LOOPS = 2  # Times the hybrid critical_evaluation back edge runs
    # Steps budget of the one orchestrator run of a solve (D-013): the
    # orchestrator's own steps plus a full strategy run for the first attempt
    # and each retry. A pop of an ended strategy FSM is not a step (D-052).
    MAX_SOLVE_STEPS = MAX_TOTAL_ITERATIONS + (MAX_RETRIES + 1) * MAX_SUB_FSM_ITERATIONS


class ErrorMessages:
    """Standard error messages."""

    MAX_RETRIES_EXCEEDED = "Maximum retry attempts exceeded"
    INVALID_REASONING_TYPE = "Invalid reasoning type: {type}"
    FSM_NOT_FOUND = "FSM definition not found: {name}"


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
    CONTEXT_PRUNED = "Context pruned ({original} chars exceeded threshold)"
    PROBLEM_SOLVED = "Problem solved in {steps} steps"
