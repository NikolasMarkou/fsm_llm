"""
Unit tests for the reasoning engine components.
"""

import pytest

from fsm_llm.stdlib.reasoning.constants import ContextKeys, ReasoningType
from fsm_llm.stdlib.reasoning.definitions import (
    ReasoningClassificationResult,
    ReasoningStep,
    ReasoningStepType,
    ValidationResult,
)
from fsm_llm.stdlib.reasoning.handlers import ReasoningHandlers
from fsm_llm.stdlib.reasoning.utilities import map_reasoning_type


class TestReasoningModels:
    """Test Pydantic models."""

    def test_validation_result(self):
        """Test ValidationResult model."""
        result = ValidationResult(
            is_valid=True,
            confidence=0.95,
            checks={"has_solution": True, "complete": True},
            issues=[],
        )

        assert result.is_valid is True
        assert result.confidence == 0.95
        assert len(result.issues) == 0

    def test_reasoning_step(self):
        """Test ReasoningStep model."""
        step = ReasoningStep(
            step_type=ReasoningStepType.ANALYSIS,
            content="Breaking down the problem",
            confidence=0.8,
            evidence=["observation1", "observation2"],
        )

        assert step.step_type == ReasoningStepType.ANALYSIS
        assert step.confidence == 0.8
        assert len(step.evidence) == 2

    def test_classification_result(self):
        """Test ReasoningClassificationResult model."""
        result = ReasoningClassificationResult(
            recommended_type="analytical",
            justification="Problem requires systematic analysis",
            domain="technical",
            alternatives=["deductive"],
            confidence="high",
        )

        assert result.recommended_type == "analytical"
        assert result.justification == "Problem requires systematic analysis"


class TestReasoningHandlers:
    """Test handler implementations."""

    def test_validate_solution(self):
        """Test solution validation handler."""
        handlers = ReasoningHandlers()

        # Valid solution
        context = {
            ContextKeys.PROPOSED_SOLUTION: "This is a detailed solution with multiple steps addressing the core problem",
            ContextKeys.KEY_INSIGHTS: ["insight1", "insight2"],
        }

        result = handlers.validate_solution(context)

        assert result[ContextKeys.SOLUTION_VALID] is True
        assert result[ContextKeys.SOLUTION_CONFIDENCE] == 1.0
        assert result[ContextKeys.VALIDATION_CHECKS]["has_solution"] is True
        assert result[ContextKeys.VALIDATION_CHECKS]["has_insights"] is True
        assert result[ContextKeys.VALIDATION_CHECKS]["sufficient_detail"] is True

    def test_update_reasoning_trace(self):
        """Test reasoning trace update."""
        handlers = ReasoningHandlers()

        context = {
            "_current_state": "analyze",
            "_previous_state": "start",
            ContextKeys.REASONING_TRACE: [],
            ContextKeys.PROBLEM_TYPE: "technical",
        }

        result = handlers.update_reasoning_trace(context)

        assert len(result[ContextKeys.REASONING_TRACE]) == 1
        trace_step = result[ContextKeys.REASONING_TRACE][0]
        assert trace_step["from"] == "start"
        assert trace_step["to"] == "analyze"
        # Only specific ContextKeys are snapshotted
        assert ContextKeys.PROBLEM_TYPE in trace_step["context_snapshot"]
        assert "_current_state" not in trace_step["context_snapshot"]
        assert "_previous_state" not in trace_step["context_snapshot"]


class TestReasoningUtils:
    """Test utility functions."""

    def test_map_reasoning_type(self):
        """Test reasoning type mapping."""
        assert map_reasoning_type("ANALYTICAL") == "analytical"
        assert map_reasoning_type("creative") == "creative"
        assert map_reasoning_type("unknown") == "analytical"  # default


class TestReasoningEngine:
    """Test main reasoning engine."""

    def test_reasoning_type_enum(self):
        """Test ReasoningType enum."""
        assert ReasoningType.ANALYTICAL.value == "analytical"
        assert ReasoningType.CREATIVE.value == "creative"

        # All reasoning types should have corresponding values
        expected_enum_values = [
            "simple_calculator",
            "analytical",
            "deductive",
            "inductive",
            "abductive",
            "analogical",
            "creative",
            "critical",
            "hybrid",
        ]
        actual_enum_values = [rt.value for rt in ReasoningType]
        for val in expected_enum_values:
            assert val in actual_enum_values
        assert len(actual_enum_values) == len(expected_enum_values)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
