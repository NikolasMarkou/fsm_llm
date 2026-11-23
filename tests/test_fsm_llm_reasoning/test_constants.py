"""
Unit tests for reasoning engine constants.
"""

from __future__ import annotations

import pytest

from fsm_llm.core.verify import FSMValidator
from fsm_llm.reasoning.constants import (
    ORCHESTRATOR_HANDLER_ONLY_KEYS,
    ClassifierStates,
    ContextKeys,
    Defaults,
    ErrorMessages,
    HandlerNames,
    LogMessages,
    OrchestratorStates,
    ReasoningType,
    solve_step_budget,
)
from fsm_llm.reasoning.handlers import ReasoningHandlers
from fsm_llm.reasoning.utilities import get_reasoning_fsm


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
        assert ContextKeys.SOLUTION_CONFIDENCE == "solution_confidence"

    def test_execution_control_keys(self):
        # the push flag and the pushed FSM are gone: the call state pushes (D-031)
        assert not hasattr(ContextKeys, "REASONING_PUSH_PENDING")
        assert not hasattr(ContextKeys, "REASONING_FSM_TO_PUSH")
        assert ContextKeys.CALL_RESULT == "call_result"
        assert ContextKeys.REASONING_TYPE_SELECTED == "reasoning_type_selected"
        assert ContextKeys.RETRY_COUNT == "retry_count"
        assert ContextKeys.MAX_RETRIES_REACHED == "max_retries_reached"

    def test_calculator_keys(self):
        assert ContextKeys.CALCULATION_EXPRESSION == "calculation_expression"
        assert ContextKeys.CALCULATION_RESULT == "calculation_result"
        assert ContextKeys.CALCULATION_ERROR == "calculation_error"

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
        assert HandlerNames.ORCHESTRATOR_SELECTOR == "OrchestratorStrategySelector"
        assert not hasattr(HandlerNames, "ORCHESTRATOR_EXECUTOR")
        assert HandlerNames.ORCHESTRATOR_VALIDATOR == "OrchestratorSolutionValidator"
        assert HandlerNames.REASONING_TRACER == "ReasoningTracer"
        assert HandlerNames.CONTEXT_PRUNER == "ContextPruner"
        assert HandlerNames.RETRY_LIMITER == "RetryLimiter"
        assert HandlerNames.RETRY_KEY_CLEARER == "RetryKeyClearer"
        assert HandlerNames.HYBRID_LOOP_COUNTER == "HybridLoopCounter"
        assert HandlerNames.CALCULATOR_EXECUTOR == "CalculatorExecutor"
        assert HandlerNames.CALCULATOR_FINALIZER == "CalculatorFinalizer"


class TestDefaults:
    """Test Defaults configuration values."""

    def test_values(self):
        assert Defaults.TEMPERATURE == 0.7
        assert Defaults.MAX_TOKENS == 2000
        assert Defaults.MAX_RETRIES == 3
        assert Defaults.MAX_TRACE_STEPS == 50
        assert Defaults.CONTEXT_PRUNE_THRESHOLD == 8000

    def test_solve_steps_budget_is_cumulative(self):
        """D-013: orchestrator steps plus one full strategy run per attempt. The strategy
        budget is ``call.max_steps`` of ``execute_reasoning`` in the orchestrator file
        (``MAX_SUB_FSM_STEPS`` is gone, D-031) and ``solve_step_budget`` derives the
        total (plan baa37f40 step 16, D-032: the literal ``MAX_SOLVE_STEPS`` is gone)."""
        call = get_reasoning_fsm("orchestrator")["states"]["execute_reasoning"]["call"]
        assert not hasattr(Defaults, "MAX_SUB_FSM_STEPS")
        assert not hasattr(Defaults, "MAX_SOLVE_STEPS")
        assert solve_step_budget(call["max_steps"]) == (
            Defaults.MAX_TOTAL_STEPS + (Defaults.MAX_RETRIES + 1) * call["max_steps"]
        )
        assert solve_step_budget(call["max_steps"]) == 170

    def test_hybrid_back_edge_limit(self):
        assert Defaults.MAX_HYBRID_LOOPS == 2


class TestHandlerOnlyKeys:
    """D-054 of plan 944e2692: the orchestrator's handler-owned keys."""

    def test_orchestrator_list(self):
        assert set(ORCHESTRATOR_HANDLER_ONLY_KEYS) == {
            "validation_result",
            "validation_checks",
            "max_retries_reached",
            "retry_count",
            "solution_confidence",
            "reasoning_type_selected",
            "classified_problem_type",
            "hybrid_loop_count",  # F3: a model claim must not reach final_context
            "calculation_result",  # D-052 of plan 51fde95b: no bulk claim is the answer
            "calculation_error_details",
        }

    def test_the_fsm_definitions_carry_them(self):
        assert get_reasoning_fsm("orchestrator")["handler_only_keys"] == list(
            ORCHESTRATOR_HANDLER_ONLY_KEYS
        )
        assert get_reasoning_fsm("hybrid")["handler_only_keys"] == [
            ContextKeys.HYBRID_LOOP_COUNT
        ]
        assert get_reasoning_fsm("simple_calculator")["handler_only_keys"] == [
            ContextKeys.CALCULATION_RESULT,
            ContextKeys.CALCULATION_ERROR,
        ]

    def test_the_calculator_fsm_validates_clean(self):
        """No D-054 warning: the handler-written keys are not ``required_context_keys``."""
        result = FSMValidator(get_reasoning_fsm("simple_calculator")).validate()
        assert result.is_valid
        assert not result.warnings


class TestErrorMessages:
    """Test error message templates."""

    def test_format_strings(self):
        assert not hasattr(
            ErrorMessages, "INVALID_REASONING_TYPE"
        )  # the dead branch is gone
        assert "{name}" in ErrorMessages.FSM_NOT_FOUND

    def test_plain_messages(self):
        assert ErrorMessages.MAX_RETRIES_EXCEEDED


class TestRemovedLegacyConstants:
    """Absence pins: each name existed (unread) on the commit before its removal."""

    @pytest.mark.parametrize("name", ["OPERAND1", "OPERAND2", "OPERATOR"])
    def test_context_key_is_gone(self, name):
        assert not hasattr(ContextKeys, name)

    @pytest.mark.parametrize("name", ["MAX_CONTEXT_SIZE"])
    def test_defaults_name_is_gone(self, name):
        assert not hasattr(Defaults, name)
        # The threshold the context pruner reads stays.
        assert Defaults.CONTEXT_PRUNE_THRESHOLD == 8000

    @pytest.mark.parametrize(
        "name", ["CONTEXT_TOO_LARGE", "VALIDATION_FAILED", "CALCULATION_ERROR"]
    )
    def test_error_message_name_is_gone(self, name):
        assert not hasattr(ErrorMessages, name)


class TestLogMessages:
    """Test log message templates."""

    def test_format_strings(self):
        assert "{model}" in LogMessages.ENGINE_INITIALIZED
        assert "{context}" in LogMessages.CLASSIFICATION_STARTED
        assert "{type}" in LogMessages.CLASSIFICATION_COMPLETE
        assert not hasattr(LogMessages, "FSM_PUSHED")  # the call state pushes (D-031)
        assert "{steps}" in LogMessages.PROBLEM_SOLVED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestMagicNumberConstants:
    """F-011: Magic numbers should be in Defaults, not hardcoded."""

    def test_min_solution_length_constant_exists(self):
        assert hasattr(Defaults, "MIN_SOLUTION_LENGTH")
        assert Defaults.MIN_SOLUTION_LENGTH == 20

    def test_prune_list_max_length_constant_exists(self):
        assert hasattr(Defaults, "PRUNE_LIST_MAX_LENGTH")
        assert Defaults.PRUNE_LIST_MAX_LENGTH == 10

    def test_prune_string_max_length_constant_exists(self):
        assert hasattr(Defaults, "PRUNE_STRING_MAX_LENGTH")
        assert Defaults.PRUNE_STRING_MAX_LENGTH == 1000

    def test_prune_context_uses_list_constant(self):
        """Verify prune_context respects PRUNE_LIST_MAX_LENGTH."""
        context = {
            ContextKeys.REASONING_TRACE: list(range(50)),
            ContextKeys.PROBLEM_STATEMENT: "test problem",
        }
        # Trigger pruning by making context large enough
        big_padding = {"_padding": "x" * (Defaults.CONTEXT_PRUNE_THRESHOLD + 1)}
        full_context = {**context, **big_padding}
        result = ReasoningHandlers.prune_context(full_context)
        if ContextKeys.REASONING_TRACE in result:
            assert (
                len(result[ContextKeys.REASONING_TRACE])
                <= Defaults.PRUNE_LIST_MAX_LENGTH
            )

    def test_validate_solution_uses_min_solution_length(self):
        """Verify validate_solution uses MIN_SOLUTION_LENGTH constant."""
        context = {
            ContextKeys.PROPOSED_SOLUTION: "short",  # < MIN_SOLUTION_LENGTH
            ContextKeys.KEY_INSIGHTS: "some insight",
            ContextKeys.RETRY_COUNT: 0,
        }
        result = ReasoningHandlers.validate_solution(context)
        assert result[ContextKeys.VALIDATION_RESULT] is False


class TestNoHardcodedKeys:
    """_classify_problem return dict must use ContextKeys constants,
    not raw strings."""

    def test_no_problem_domain_classified_string_in_source(self):
        import inspect

        from fsm_llm.reasoning.engine import ReasoningEngine

        source = inspect.getsource(ReasoningEngine._classify_problem)
        assert "problem_domain_classified" not in source

    def test_classify_return_uses_context_keys(self):
        """The return dict keys should match ContextKeys constants."""
        import inspect

        from fsm_llm.reasoning.engine import ReasoningEngine

        source = inspect.getsource(ReasoningEngine._classify_problem)
        assert "ContextKeys.PROBLEM_DOMAIN" in source
        assert "ContextKeys.ALTERNATIVE_APPROACHES" in source
