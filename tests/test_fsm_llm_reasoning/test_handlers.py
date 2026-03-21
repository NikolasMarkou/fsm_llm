"""
Unit tests for reasoning engine handlers, context manager, and output formatter.
"""
import pytest

from fsm_llm_reasoning.constants import (
    ContextKeys,
    Defaults,
    ErrorMessages,
    ReasoningType,
)
from fsm_llm_reasoning.handlers import (
    ContextManager,
    OutputFormatter,
    ReasoningHandlers,
)


class TestValidateSolution:
    """Test ReasoningHandlers.validate_solution."""

    def test_valid_complex_solution(self):
        ctx = {
            ContextKeys.PROPOSED_SOLUTION: "This is a detailed solution with multiple steps addressing the problem",
            ContextKeys.KEY_INSIGHTS: ["insight1", "insight2"],
            ContextKeys.PROBLEM_STATEMENT: "How to solve this complex problem?",
        }
        result = ReasoningHandlers.validate_solution(ctx)

        assert result[ContextKeys.SOLUTION_VALID] is True
        assert result[ContextKeys.SOLUTION_CONFIDENCE] == 1.0
        assert result[ContextKeys.VALIDATION_CHECKS]["has_solution"] is True
        assert result[ContextKeys.VALIDATION_CHECKS]["has_insights"] is True
        assert result[ContextKeys.VALIDATION_CHECKS]["sufficient_detail"] is True
        assert result[ContextKeys.VALIDATION_CHECKS]["addresses_problem"] is True

    def test_empty_solution(self):
        ctx = {
            ContextKeys.PROPOSED_SOLUTION: "",
            ContextKeys.KEY_INSIGHTS: [],
        }
        result = ReasoningHandlers.validate_solution(ctx)

        assert result[ContextKeys.SOLUTION_VALID] is False
        assert result[ContextKeys.VALIDATION_CHECKS]["has_solution"] is False

    def test_simple_calculator_relaxed_validation(self):
        """Simple calculator problems should accept short solutions."""
        ctx = {
            ContextKeys.PROPOSED_SOLUTION: "42",
            ContextKeys.PROBLEM_TYPE: "arithmetic",
            ContextKeys.REASONING_STRATEGY: "simple_calculator",
        }
        result = ReasoningHandlers.validate_solution(ctx)

        assert result[ContextKeys.SOLUTION_VALID] is True
        assert result[ContextKeys.VALIDATION_CHECKS]["sufficient_detail"] is True
        assert result[ContextKeys.VALIDATION_CHECKS]["has_insights"] is True  # relaxed for simple

    def test_retry_count_incremented_on_failure(self):
        ctx = {
            ContextKeys.PROPOSED_SOLUTION: "",
            ContextKeys.RETRY_COUNT: 0,
        }
        result = ReasoningHandlers.validate_solution(ctx)

        assert result[ContextKeys.RETRY_COUNT] == 1
        assert result[ContextKeys.MAX_RETRIES_REACHED] is False

    def test_max_retries_reached(self):
        ctx = {
            ContextKeys.PROPOSED_SOLUTION: "",
            ContextKeys.RETRY_COUNT: Defaults.MAX_RETRIES,
        }
        result = ReasoningHandlers.validate_solution(ctx)

        assert result[ContextKeys.MAX_RETRIES_REACHED] is True

    def test_addresses_problem_overlap(self):
        """Solution should have word overlap with problem."""
        ctx = {
            ContextKeys.PROPOSED_SOLUTION: "The algorithm has time complexity O(n log n) for sorting",
            ContextKeys.KEY_INSIGHTS: ["sorting is key"],
            ContextKeys.PROBLEM_STATEMENT: "What is the time complexity of merge sort algorithm?",
        }
        result = ReasoningHandlers.validate_solution(ctx)

        assert result[ContextKeys.VALIDATION_CHECKS]["addresses_problem"] is True

    def test_no_overlap_still_counted(self):
        """Solution with no word overlap should still count addresses_problem if solution exists."""
        ctx = {
            ContextKeys.PROPOSED_SOLUTION: "xyz abc completely unrelated long enough text here",
            ContextKeys.KEY_INSIGHTS: ["insight"],
            ContextKeys.PROBLEM_STATEMENT: "",
        }
        result = ReasoningHandlers.validate_solution(ctx)
        # Empty problem_statement -> addresses_problem defaults to has_solution
        assert result[ContextKeys.VALIDATION_CHECKS]["addresses_problem"] is True


class TestUpdateReasoningTrace:
    """Test ReasoningHandlers.update_reasoning_trace."""

    def test_adds_step(self):
        ctx = {
            "_current_state": "analyze",
            "_previous_state": "start",
            ContextKeys.REASONING_TRACE: [],
            ContextKeys.PROBLEM_TYPE: "technical",
        }
        result = ReasoningHandlers.update_reasoning_trace(ctx)

        trace = result[ContextKeys.REASONING_TRACE]
        assert len(trace) == 1
        assert trace[0]["from"] == "start"
        assert trace[0]["to"] == "analyze"
        assert ContextKeys.PROBLEM_TYPE in trace[0]["context_snapshot"]

    def test_no_step_without_states(self):
        ctx = {
            ContextKeys.REASONING_TRACE: [],
        }
        result = ReasoningHandlers.update_reasoning_trace(ctx)
        assert len(result[ContextKeys.REASONING_TRACE]) == 0

    def test_trace_pruning(self):
        """Trace should be pruned when exceeding MAX_TRACE_STEPS."""
        long_trace = [{"from": f"s{i}", "to": f"s{i+1}"} for i in range(Defaults.MAX_TRACE_STEPS + 10)]
        ctx = {
            "_current_state": "final",
            "_previous_state": "penultimate",
            ContextKeys.REASONING_TRACE: long_trace,
        }
        result = ReasoningHandlers.update_reasoning_trace(ctx)

        trace = result[ContextKeys.REASONING_TRACE]
        # Should be pruned to MAX_TRACE_STEPS (first 5 + last N) + 1 new step
        assert len(trace) <= Defaults.MAX_TRACE_STEPS + 1

    def test_snapshot_only_includes_specific_keys(self):
        """Only specific context keys should be in the snapshot."""
        ctx = {
            "_current_state": "b",
            "_previous_state": "a",
            ContextKeys.REASONING_TRACE: [],
            ContextKeys.PROBLEM_TYPE: "math",
            ContextKeys.REASONING_STRATEGY: "analytical",
            "random_key": "should not appear",
            "_internal_key": "should not appear",
        }
        result = ReasoningHandlers.update_reasoning_trace(ctx)
        snapshot = result[ContextKeys.REASONING_TRACE][0]["context_snapshot"]

        assert ContextKeys.PROBLEM_TYPE in snapshot
        assert ContextKeys.REASONING_STRATEGY in snapshot
        assert "random_key" not in snapshot
        assert "_internal_key" not in snapshot


class TestPruneContext:
    """Test ReasoningHandlers.prune_context."""

    def test_no_pruning_under_threshold(self):
        ctx = {"key": "small value"}
        result = ReasoningHandlers.prune_context(ctx)
        assert result == {}

    def test_pruning_large_lists(self):
        """Large lists should be truncated to last 10 items."""
        large_list = list(range(100))
        ctx = {
            ContextKeys.REASONING_TRACE: large_list,
            ContextKeys.PROBLEM_STATEMENT: "test",
        }
        # Make context large enough to trigger pruning
        ctx["_padding"] = "x" * (Defaults.CONTEXT_PRUNE_THRESHOLD + 1000)
        result = ReasoningHandlers.prune_context(ctx)

        if ContextKeys.REASONING_TRACE in result:
            assert len(result[ContextKeys.REASONING_TRACE]) <= 10

    def test_pruning_large_strings(self):
        """Large strings should be truncated to 1000 chars."""
        ctx = {
            ContextKeys.OBSERVATIONS: "x" * 5000,
            ContextKeys.PROBLEM_STATEMENT: "test",
            "_padding": "y" * (Defaults.CONTEXT_PRUNE_THRESHOLD + 1000),
        }
        result = ReasoningHandlers.prune_context(ctx)

        if ContextKeys.OBSERVATIONS in result:
            assert len(result[ContextKeys.OBSERVATIONS]) <= 1100  # 1000 + truncation marker

    def test_preserve_keys_not_pruned(self):
        """Keys in the preserve set should never be pruned."""
        ctx = {
            ContextKeys.PROBLEM_STATEMENT: "x" * 5000,
            ContextKeys.PROPOSED_SOLUTION: "y" * 5000,
            "_padding": "z" * (Defaults.CONTEXT_PRUNE_THRESHOLD + 1000),
        }
        result = ReasoningHandlers.prune_context(ctx)

        # Preserve keys should not be in pruned updates
        assert ContextKeys.PROBLEM_STATEMENT not in result
        assert ContextKeys.PROPOSED_SOLUTION not in result


class TestContextManager:
    """Test ContextManager utility class."""

    def test_extract_relevant_context(self):
        source = {"a": 1, "b": 2, "c": None, "d": 4}
        result = ContextManager.extract_relevant_context(source, ["a", "c", "d", "missing"])

        assert result == {"a": 1, "d": 4}  # c excluded (None), missing excluded
        assert "b" not in result

    def test_extract_with_max_size(self):
        """Max size is now enforced: oversized keys are removed to fit budget."""
        source = {"key": "x" * 10000}
        result = ContextManager.extract_relevant_context(source, ["key"], max_size=100)
        assert "key" not in result  # key removed because it exceeds budget

    def test_merge_analytical_results(self):
        results = ContextManager.merge_reasoning_results(
            {},
            {ContextKeys.KEY_INSIGHTS: ["insight1"], ContextKeys.INTEGRATED_ANALYSIS: "analysis"},
            ReasoningType.ANALYTICAL.value,
        )
        assert ContextKeys.KEY_INSIGHTS in results
        assert ContextKeys.INTEGRATED_ANALYSIS in results
        assert f"{ReasoningType.ANALYTICAL.value}_reasoning_completed" in results

    def test_merge_deductive_results(self):
        results = ContextManager.merge_reasoning_results(
            {},
            {ContextKeys.CONCLUSION: "therefore X", ContextKeys.LOGICAL_VALIDITY: True},
            ReasoningType.DEDUCTIVE.value,
        )
        assert ContextKeys.DEDUCTIVE_CONCLUSION in results
        assert ContextKeys.LOGICAL_VALIDITY in results

    def test_merge_simple_calculator_results(self):
        results = ContextManager.merge_reasoning_results(
            {},
            {ContextKeys.CALCULATION_RESULT: 42},
            ReasoningType.SIMPLE_CALCULATOR.value,
        )
        assert results[ContextKeys.CALCULATION_RESULT] == 42
        assert results[ContextKeys.PROPOSED_SOLUTION] == 42

    def test_merge_filters_none_values(self):
        results = ContextManager.merge_reasoning_results(
            {},
            {ContextKeys.KEY_INSIGHTS: None, ContextKeys.INTEGRATED_ANALYSIS: None},
            ReasoningType.ANALYTICAL.value,
        )
        assert ContextKeys.KEY_INSIGHTS not in results
        assert ContextKeys.INTEGRATED_ANALYSIS not in results

    def test_merge_all_reasoning_types(self):
        """All reasoning types should produce a completion flag."""
        for rt in ReasoningType:
            results = ContextManager.merge_reasoning_results({}, {}, rt.value)
            assert f"{rt.value}_reasoning_completed" in results


class TestOutputFormatter:
    """Test OutputFormatter utility class."""

    def test_extract_final_solution_priority(self):
        """Should follow priority order for solution keys."""
        ctx = {
            ContextKeys.FINAL_SOLUTION: "final",
            ContextKeys.PROPOSED_SOLUTION: "proposed",
        }
        assert OutputFormatter.extract_final_solution(ctx) == "final"

    def test_extract_fallback_to_proposed(self):
        ctx = {ContextKeys.PROPOSED_SOLUTION: "proposed"}
        assert OutputFormatter.extract_final_solution(ctx) == "proposed"

    def test_extract_calculation_result(self):
        ctx = {ContextKeys.CALCULATION_RESULT: "42"}
        assert OutputFormatter.extract_final_solution(ctx) == "42"

    def test_extract_max_retries_message(self):
        ctx = {ContextKeys.MAX_RETRIES_REACHED: True}
        result = OutputFormatter.extract_final_solution(ctx)
        assert result == ErrorMessages.MAX_RETRIES_EXCEEDED

    def test_extract_no_solution(self):
        result = OutputFormatter.extract_final_solution({})
        assert "no explicit solution" in result.lower()

    def test_format_reasoning_summary(self):
        trace_info = {
            "total_steps": 5,
            "reasoning_types_used": ["analytical", "deductive"],
            "final_confidence": 0.85,
        }
        summary = OutputFormatter.format_reasoning_summary(trace_info)

        assert "5" in summary
        assert "analytical" in summary
        assert "85" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
