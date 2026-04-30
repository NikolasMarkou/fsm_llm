"""
Unit tests for reasoning engine constants.
"""

import pytest

from fsm_llm.stdlib.reasoning.constants import (
    ClassifierStates,
    ContextKeys,
    Defaults,
    ErrorMessages,
    HandlerNames,
    LogMessages,
    OrchestratorStates,
    ReasoningType,
)


class TestReasoningType:
    """Test ReasoningType enum."""

    def test_all_values(self):
        expected = {
            "simple_calculator",
            "analytical",
            "deductive",
            "inductive",
            "abductive",
            "analogical",
            "creative",
            "critical",
            "hybrid",
        }
        actual = {rt.value for rt in ReasoningType}
        assert actual == expected

    def test_count(self):
        assert len(ReasoningType) == 9

    def test_string_enum(self):
        assert isinstance(ReasoningType.ANALYTICAL, str)
        assert ReasoningType.ANALYTICAL == "analytical"

    def test_from_value(self):
        assert ReasoningType("analytical") == ReasoningType.ANALYTICAL

    def test_invalid_value(self):
        with pytest.raises(ValueError):
            ReasoningType("nonexistent")


class TestOrchestratorStates:
    """Test OrchestratorStates constants."""

    def test_all_states_defined(self):
        assert OrchestratorStates.PROBLEM_ANALYSIS == "problem_analysis"
        assert OrchestratorStates.STRATEGY_SELECTION == "strategy_selection"
        assert OrchestratorStates.EXECUTE_REASONING == "execute_reasoning"
        assert OrchestratorStates.SYNTHESIZE_SOLUTION == "synthesize_solution"
        assert OrchestratorStates.VALIDATE_REFINE == "validate_refine"
        assert OrchestratorStates.FINAL_ANSWER == "final_answer"


class TestClassifierStates:
    """Test ClassifierStates constants."""

    def test_all_states_defined(self):
        assert ClassifierStates.ANALYZE_DOMAIN == "analyze_domain"
        assert ClassifierStates.ANALYZE_STRUCTURE == "analyze_structure"
        assert ClassifierStates.IDENTIFY_REASONING_NEEDS == "identify_reasoning_needs"
        assert ClassifierStates.RECOMMEND_STRATEGY == "recommend_strategy"


class TestContextKeys:
    """Test ContextKeys has all expected key groups."""

    def test_problem_keys(self):
        assert ContextKeys.PROBLEM_STATEMENT == "problem_statement"
        assert ContextKeys.PROBLEM_TYPE == "problem_type"
        assert ContextKeys.PROBLEM_COMPONENTS == "problem_components"

    def test_solution_keys(self):
        assert ContextKeys.PROPOSED_SOLUTION == "proposed_solution"
        assert ContextKeys.FINAL_SOLUTION == "final_solution"
        assert ContextKeys.KEY_INSIGHTS == "key_insights"

    def test_validation_keys(self):
        assert ContextKeys.VALIDATION_RESULT == "validation_result"
        assert ContextKeys.SOLUTION_VALID == "solution_valid"
        assert ContextKeys.SOLUTION_CONFIDENCE == "solution_confidence"

    def test_execution_control_keys(self):
        assert ContextKeys.REASONING_FSM_TO_PUSH == "reasoning_fsm_to_push"
        assert ContextKeys.REASONING_TYPE_SELECTED == "reasoning_type_selected"
        assert ContextKeys.RETRY_COUNT == "retry_count"
        assert ContextKeys.MAX_RETRIES_REACHED == "max_retries_reached"

    def test_calculator_keys(self):
        assert ContextKeys.OPERAND1 == "operand1"
        assert ContextKeys.OPERAND2 == "operand2"
        assert ContextKeys.OPERATOR == "operator"
        assert ContextKeys.CALCULATION_RESULT == "calculation_result"

    def test_reasoning_type_result_keys(self):
        """Each reasoning type should have result keys."""
        assert ContextKeys.DEDUCTIVE_CONCLUSION == "deductive_conclusion"
        assert ContextKeys.INDUCTIVE_HYPOTHESIS == "inductive_hypothesis"
        assert ContextKeys.BEST_CREATIVE_SOLUTION == "best_creative_solution"
        assert ContextKeys.CRITICAL_ASSESSMENT == "critical_assessment"
        assert ContextKeys.FINAL_HYBRID_SOLUTION == "final_hybrid_solution"
        assert ContextKeys.BEST_EXPLANATION == "best_explanation"
        assert ContextKeys.ANALOGICAL_SOLUTION == "analogical_solution"


class TestHandlerNames:
    """Test HandlerNames constants."""

    def test_all_names(self):
        assert HandlerNames.ORCHESTRATOR_CLASSIFIER == "OrchestratorProblemClassifier"
        assert HandlerNames.ORCHESTRATOR_EXECUTOR == "OrchestratorStrategyExecutor"
        assert HandlerNames.ORCHESTRATOR_VALIDATOR == "OrchestratorSolutionValidator"
        assert HandlerNames.REASONING_TRACER == "ReasoningTracer"
        assert HandlerNames.CONTEXT_PRUNER == "ContextPruner"
        assert HandlerNames.RETRY_LIMITER == "RetryLimiter"


class TestDefaults:
    """Test Defaults configuration values."""

    def test_values(self):
        assert isinstance(Defaults.MODEL, str)
        assert Defaults.TEMPERATURE == 0.7
        assert Defaults.MAX_TOKENS == 2000
        assert Defaults.MAX_RETRIES == 3
        assert Defaults.MAX_CONTEXT_SIZE == 10000
        assert Defaults.MAX_TRACE_STEPS == 50
        assert Defaults.CONTEXT_PRUNE_THRESHOLD == 8000

    def test_prune_threshold_below_max(self):
        """Prune threshold should be below max context size."""
        assert Defaults.CONTEXT_PRUNE_THRESHOLD < Defaults.MAX_CONTEXT_SIZE


class TestErrorMessages:
    """Test error message templates."""

    def test_format_strings(self):
        assert "{type}" in ErrorMessages.INVALID_REASONING_TYPE
        assert "{name}" in ErrorMessages.FSM_NOT_FOUND
        assert "{error}" in ErrorMessages.CALCULATION_ERROR
        assert "{reason}" in ErrorMessages.VALIDATION_FAILED

    def test_plain_messages(self):
        assert ErrorMessages.MAX_RETRIES_EXCEEDED
        assert ErrorMessages.CONTEXT_TOO_LARGE


class TestLogMessages:
    """Test log message templates."""

    def test_format_strings(self):
        assert "{model}" in LogMessages.ENGINE_INITIALIZED
        assert "{context}" in LogMessages.CLASSIFICATION_STARTED
        assert "{type}" in LogMessages.CLASSIFICATION_COMPLETE
        assert "{name}" in LogMessages.FSM_PUSHED
        assert "{steps}" in LogMessages.PROBLEM_SOLVED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
