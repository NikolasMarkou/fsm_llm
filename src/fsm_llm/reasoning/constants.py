"""
Constants and enumerations for the reasoning engine.
All string literals and magic values are consolidated here.
"""

from __future__ import annotations

from enum import Enum


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

    Note: These constants spell the state names of ``strategies/orchestrator.json``.
    Both locations must stay in sync when renaming states.
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

    # Strategy selection
    REASONING_STRATEGY = "reasoning_strategy"
    CLASSIFIED_PROBLEM_TYPE = "classified_problem_type"
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
    # The strategy NAME the execute_reasoning call runs (written by the entry
    # handler); the call state chooses its target by CLASSIFIED_PROBLEM_TYPE,
    # which holds the target id, and writes its record to CALL_RESULT.
    REASONING_TYPE_SELECTED = "reasoning_type_selected"
    CALL_RESULT = "call_result"
    CLASSIFICATION_JUSTIFICATION = "classification_justification"
    RETRY_COUNT = "retry_count"
    MAX_RETRIES_REACHED = "max_retries_reached"

    # Hybrid strategy loop control (the critical_evaluation back edge)
    NEEDS_REFINEMENT = "needs_refinement"
    HYBRID_LOOP_COUNT = "hybrid_loop_count"

    # Simple calculator specific
    CALCULATION_EXPRESSION = "calculation_expression"
    CALCULATION_RESULT = "calculation_result"
    CALCULATION_ERROR = "calculation_error"

    # Analytical reasoning
    INTEGRATED_ANALYSIS = "integrated_analysis"

    # Deductive reasoning
    LOGICAL_STEPS = "logical_steps"
    CONCLUSION = "conclusion"

    # Inductive reasoning
    OBSERVATIONS = "observations"

    # Creative reasoning
    CREATIVE_IDEAS = "creative_ideas"
    BEST_CREATIVE_SOLUTION = "best_creative_solution"

    # Critical reasoning
    CRITICAL_ASSESSMENT = "critical_assessment"

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
    FINAL_HYBRID_SOLUTION = "final_hybrid_solution"

    # Deductive sub-FSM result keys
    DEDUCTIVE_CONCLUSION = "deductive_conclusion"

    # Inductive sub-FSM result keys
    INDUCTIVE_HYPOTHESIS = "inductive_hypothesis"

    # Abductive sub-FSM result keys
    BEST_EXPLANATION = "best_explanation"

    # Analogical sub-FSM result keys
    ANALOGICAL_SOLUTION = "analogical_solution"

    # Calculation error details
    CALCULATION_ERROR_DETAILS = "calculation_error_details"

    # CLI --type flag: set in initial_context to override automatic type selection
    PREFERRED_REASONING_TYPE = "preferred_reasoning_type"


# Keys cleared on every entry to execute_reasoning. Core extracts a key only
# while it is unset, so without the clear a retry would keep the rejected
# solution and never re-run synthesis or validation (the retry loop stalls).
RETRY_CLEARED_KEYS: tuple[str, ...] = (
    ContextKeys.PROPOSED_SOLUTION,
    ContextKeys.KEY_INSIGHTS,
    ContextKeys.VALIDATION_RESULT,
    ContextKeys.CALCULATION_RESULT,
    ContextKeys.CALCULATION_ERROR_DETAILS,
    ContextKeys.CALL_RESULT,
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
    ContextKeys.HYBRID_LOOP_COUNT,
    ContextKeys.CALCULATION_RESULT,
    ContextKeys.CALCULATION_ERROR_DETAILS,
    ContextKeys.PREFERRED_REASONING_TYPE,
)

# Keys only the engine's own handlers write to choose the strategy
# (``_classify_problem``, ``_select_reasoning_type``): ``solve_problem`` drops
# them from a caller's ``initial_context``, so a caller cannot name a strategy
# before the orchestrator has chosen one, nor set a classification. This
# does NOT make a ``final_context`` passed back in start fresh: a carried
# ``problem_type`` keeps the classifier from running and a carried
# ``reasoning_strategy`` then picks the strategy (Known open, D-061).
SOLVE_DRIVER_KEYS: tuple[str, ...] = (
    ContextKeys.REASONING_TYPE_SELECTED,
    ContextKeys.CLASSIFIED_PROBLEM_TYPE,
)

# The calculator state whose entry evaluates the extracted expression.
CALCULATION_STATE = "perform_calculation"

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
    ORCHESTRATOR_SELECTOR = "OrchestratorStrategySelector"
    ORCHESTRATOR_VALIDATOR = "OrchestratorSolutionValidator"
    REASONING_TRACER = "ReasoningTracer"
    CONTEXT_PRUNER = "ContextPruner"
    RETRY_LIMITER = "RetryLimiter"
    RETRY_KEY_CLEARER = "RetryKeyClearer"
    HYBRID_LOOP_COUNTER = "HybridLoopCounter"
    HYBRID_KEY_CLEARER = "HybridKeyClearer"
    CALCULATOR_EXECUTOR = "CalculatorExecutor"
    CALCULATOR_FINALIZER = "CalculatorFinalizer"


class Defaults:
    """Default configuration values."""

    TEMPERATURE = 0.7
    MAX_TOKENS = 2000
    MAX_RETRIES = 3
    MAX_TRACE_STEPS = 50
    TRACE_KEEP_START = 5  # Oldest trace steps kept when the trace is pruned
    CONTEXT_PRUNE_THRESHOLD = 8000  # characters; prune the context above this
    MIN_SOLUTION_LENGTH = 20  # Minimum chars for a substantive solution
    PRUNE_LIST_MAX_LENGTH = 10  # Keep last N items when pruning lists
    PRUNE_STRING_MAX_LENGTH = 1000  # Truncate strings beyond this length
    MAX_CLASSIFICATION_STEPS = 10  # Steps budget of the classifier FSM run
    MAX_TOTAL_STEPS = 50  # Orchestrator steps of one solve
    MAX_HYBRID_LOOPS = 2  # Times the hybrid critical_evaluation back edge runs
    # Characters of a strategy description kept as ONE line for a closed intent, a type listing
    # and the CLI (``fsm_llm.text.capped_line``); the llm_description precedent of core.
    STRATEGY_DESCRIPTION_CHARS = 300


def solve_step_budget(strategy_steps: int) -> int:
    """The step budget of the one orchestrator run of a solve (D-013, plan baa37f40 D-032).

    The orchestrator's own steps (``Defaults.MAX_TOTAL_STEPS``) plus a full strategy run for
    the first attempt and each retry, ``strategy_steps`` each: ``strategy_steps`` is
    ``call.max_steps`` of ``execute_reasoning`` in ``strategies/orchestrator.json`` (the
    engine reads it from the file it loaded, so the number lives once). A pop of an ended
    strategy FSM is not a step. With the shipped file (30) the result is 170.
    """
    return Defaults.MAX_TOTAL_STEPS + (Defaults.MAX_RETRIES + 1) * strategy_steps


class ErrorMessages:
    """Standard error messages."""

    MAX_RETRIES_EXCEEDED = "Maximum retry attempts exceeded"
    #: The answer of a calculator solve whose expression was refused on every
    #: attempt; ``{reason}`` is one of the calculator's fixed refusals.
    CALCULATION_REFUSED = "The calculation could not be completed: {reason}"
    FSM_NOT_FOUND = "FSM definition not found: {name}"


class LogMessages:
    """Standard log message templates."""

    ENGINE_INITIALIZED = "Reasoning engine initialized with model: {model}"
    CLASSIFICATION_STARTED = "Starting classification with context: {context}"
    CLASSIFICATION_COMPLETE = "Classification complete. Recommended: {type}"
    STRATEGY_EXECUTING = "Executing {type} reasoning strategy"
    VALIDATION_RESULT = "Validation: {valid} (confidence: {confidence})"
    RETRY_ATTEMPT = "Retry attempt {current}/{max} for validation"
    CONTEXT_PRUNED = "Context pruned ({original} chars exceeded threshold)"
    PROBLEM_SOLVED = "Problem solved in {steps} steps"
