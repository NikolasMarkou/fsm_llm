"""
Unit tests for reasoning engine Pydantic models.
"""
import pytest
from datetime import datetime

from fsm_llm_reasoning.definitions import (
    ReasoningStep,
    ReasoningStepType,
    ReasoningTrace,
    ValidationResult,
    ReasoningClassificationResult,
    ProblemContext,
    SolutionResult,
    ContextSnapshot,
    EngineStatus,
    ConfidenceLevel,
    ErrorReport,
    PerformanceMetrics,
    TimestampedModel,
    get_model_by_name,
    validate_model_data,
)


class TestTimestampedModel:
    """Test base timestamped model."""

    def test_auto_timestamp(self):
        m = TimestampedModel()
        assert isinstance(m.timestamp, datetime)
        assert m.timestamp.tzinfo is not None

    def test_age_seconds(self):
        m = TimestampedModel()
        assert m.age_seconds >= 0
        assert m.age_seconds < 5  # should be near-instant


class TestReasoningStep:
    """Test ReasoningStep model."""

    def test_basic_creation(self):
        step = ReasoningStep(
            step_type=ReasoningStepType.ANALYSIS,
            content="Breaking down the problem"
        )
        assert step.step_type == ReasoningStepType.ANALYSIS
        assert step.confidence == 0.0
        assert step.evidence == []

    def test_confidence_levels(self):
        low = ReasoningStep(step_type=ReasoningStepType.ANALYSIS, content="test", confidence=0.3)
        med = ReasoningStep(step_type=ReasoningStepType.ANALYSIS, content="test", confidence=0.6)
        high = ReasoningStep(step_type=ReasoningStepType.ANALYSIS, content="test", confidence=0.8)
        very_high = ReasoningStep(step_type=ReasoningStepType.ANALYSIS, content="test", confidence=0.95)

        assert low.confidence_level == ConfidenceLevel.LOW
        assert med.confidence_level == ConfidenceLevel.MEDIUM
        assert high.confidence_level == ConfidenceLevel.HIGH
        assert very_high.confidence_level == ConfidenceLevel.VERY_HIGH

    def test_has_evidence(self):
        no_ev = ReasoningStep(step_type=ReasoningStepType.ANALYSIS, content="test")
        has_ev = ReasoningStep(step_type=ReasoningStepType.ANALYSIS, content="test", evidence=["fact1"])
        assert no_ev.has_evidence is False
        assert has_ev.has_evidence is True

    def test_content_validation_empty(self):
        with pytest.raises(ValueError):
            ReasoningStep(step_type=ReasoningStepType.ANALYSIS, content="   ")

    def test_content_validation_stripped(self):
        step = ReasoningStep(step_type=ReasoningStepType.ANALYSIS, content="  hello  ")
        assert step.content == "hello"

    def test_evidence_cleaned(self):
        step = ReasoningStep(
            step_type=ReasoningStepType.ANALYSIS,
            content="test",
            evidence=["valid", "", "  ", "also valid"]
        )
        assert step.evidence == ["valid", "also valid"]

    def test_confidence_bounds(self):
        with pytest.raises(ValueError):
            ReasoningStep(step_type=ReasoningStepType.ANALYSIS, content="test", confidence=1.5)
        with pytest.raises(ValueError):
            ReasoningStep(step_type=ReasoningStepType.ANALYSIS, content="test", confidence=-0.1)


class TestValidationResult:
    """Test ValidationResult model."""

    def test_basic_valid(self):
        vr = ValidationResult(is_valid=True, confidence=0.9, checks={"a": True, "b": True})
        assert vr.passed_checks == 2
        assert vr.total_checks == 2
        assert vr.pass_rate == 1.0
        assert vr.has_issues is False
        assert "Valid" in vr.validation_summary

    def test_with_issues(self):
        vr = ValidationResult(
            is_valid=False,
            confidence=0.4,
            checks={"a": True, "b": False},
            issues=["b failed"]
        )
        assert vr.passed_checks == 1
        assert vr.pass_rate == 0.5
        assert vr.has_issues is True
        assert "Invalid" in vr.validation_summary

    def test_empty_checks(self):
        vr = ValidationResult(is_valid=True)
        assert vr.pass_rate == 0.0
        assert vr.total_checks == 0


class TestReasoningTrace:
    """Test ReasoningTrace model."""

    def test_empty_trace(self):
        trace = ReasoningTrace()
        assert trace.total_steps == 0
        assert trace.reasoning_complexity == "simple"
        assert trace.unique_states_visited == []
        assert trace.average_step_time is None

    def test_with_steps(self):
        steps = [
            {"from": "start", "to": "analyze"},
            {"from": "analyze", "to": "synthesize"},
        ]
        trace = ReasoningTrace(steps=steps, reasoning_types_used={"analytical"})
        assert trace.total_steps == 2
        assert set(trace.unique_states_visited) == {"start", "analyze", "synthesize"}

    def test_complexity_levels(self):
        simple = ReasoningTrace(steps=[{"x": 1}] * 3, reasoning_types_used={"a"})
        moderate = ReasoningTrace(steps=[{"x": 1}] * 8, reasoning_types_used={"a", "b"})
        complex_t = ReasoningTrace(steps=[{"x": 1}] * 15, reasoning_types_used={"a", "b", "c"})
        highly = ReasoningTrace(steps=[{"x": 1}] * 25, reasoning_types_used={"a", "b", "c", "d"})

        assert simple.reasoning_complexity == "simple"
        assert moderate.reasoning_complexity == "moderate"
        assert complex_t.reasoning_complexity == "complex"
        assert highly.reasoning_complexity == "highly_complex"

    def test_average_step_time(self):
        trace = ReasoningTrace(steps=[{"x": 1}] * 10, execution_time_seconds=5.0)
        assert trace.average_step_time == 0.5

    def test_computed_fields_stripped_from_input(self):
        """Computed fields in input dict should not cause validation errors."""
        trace = ReasoningTrace(
            steps=[],
            total_steps=999,  # should be stripped
            unique_states_visited=["fake"],  # should be stripped
        )
        assert trace.total_steps == 0


class TestReasoningClassificationResult:
    """Test classification result model."""

    def test_basic(self):
        r = ReasoningClassificationResult(
            recommended_type="analytical",
            justification="systematic analysis needed"
        )
        assert r.has_alternatives is False
        assert "analytical" in r.classification_summary

    def test_alternatives_deduped(self):
        r = ReasoningClassificationResult(
            recommended_type="analytical",
            justification="test",
            alternatives=["creative", "creative", "deductive"]
        )
        assert r.alternatives == ["creative", "deductive"]


class TestProblemContext:
    """Test ProblemContext model."""

    def test_basic(self):
        pc = ProblemContext(problem_statement="What is 2+2?")
        assert pc.has_constraints is False
        assert pc.is_high_priority is False
        assert pc.context_size > 0

    def test_validation_too_short(self):
        with pytest.raises(ValueError):
            ProblemContext(problem_statement="ab")

    def test_validation_empty(self):
        with pytest.raises(ValueError):
            ProblemContext(problem_statement="   ")

    def test_constraints_cleaned(self):
        pc = ProblemContext(
            problem_statement="test problem",
            constraints=["valid", "", "  ", "also valid"]
        )
        assert pc.constraints == ["valid", "also valid"]

    def test_high_priority(self):
        pc = ProblemContext(problem_statement="urgent task", priority="urgent")
        assert pc.is_high_priority is True


class TestSolutionResult:
    """Test SolutionResult model."""

    def test_basic(self):
        trace = ReasoningTrace(steps=[{"from": "a", "to": "b"}])
        sr = SolutionResult(
            solution="The answer is 42",
            confidence=0.95,
            reasoning_summary="Analytical reasoning applied",
            trace=trace,
        )
        assert sr.confidence_level == ConfidenceLevel.VERY_HIGH
        assert sr.is_high_confidence is True
        assert sr.reasoning_depth == 1
        assert sr.has_alternatives is False
        assert sr.is_validated is False

    def test_solution_not_empty(self):
        with pytest.raises(ValueError):
            SolutionResult(
                solution="  ",
                reasoning_summary="test summary here",
                trace=ReasoningTrace(),
            )


class TestContextSnapshot:
    """Test ContextSnapshot model."""

    def test_basic(self):
        cs = ContextSnapshot(state="analyzing", context_data={"key": "value"})
        assert cs.key_count == 1
        assert cs.has_important_keys is False

    def test_important_keys_validated(self):
        with pytest.raises(ValueError, match="Important keys not found"):
            ContextSnapshot(
                state="test",
                context_data={"a": 1},
                important_keys={"b"}
            )

    def test_context_density(self):
        cs = ContextSnapshot(
            state="test",
            context_data={"a": 1, "b": 2},
            important_keys={"a"}
        )
        assert cs.context_density == 0.5


class TestEngineStatus:
    """Test EngineStatus model."""

    def test_basic(self):
        es = EngineStatus(is_ready=True, model="gpt-4")
        assert es.has_active_conversations is False
        assert es.fsm_count == 0
        assert "Ready" in es.status_summary

    def test_not_ready(self):
        es = EngineStatus(is_ready=False, model="gpt-4")
        assert "Not Ready" in es.status_summary


class TestErrorReport:
    """Test ErrorReport model."""

    def test_critical(self):
        er = ErrorReport(error_type="RuntimeError", message="boom", severity="critical")
        assert er.is_critical is True

    def test_non_critical(self):
        er = ErrorReport(error_type="ValueError", message="minor")
        assert er.is_critical is False


class TestPerformanceMetrics:
    """Test PerformanceMetrics model."""

    def test_empty(self):
        pm = PerformanceMetrics()
        assert pm.has_performance_data is False
        assert pm.most_used_reasoning_type is None

    def test_with_data(self):
        pm = PerformanceMetrics(
            total_problems_solved=10,
            reasoning_type_usage={"analytical": 5, "creative": 3}
        )
        assert pm.has_performance_data is True
        assert pm.most_used_reasoning_type == "analytical"


class TestUtilityFunctions:
    """Test module-level utility functions."""

    def test_get_model_by_name(self):
        assert get_model_by_name("ReasoningStep") is ReasoningStep
        assert get_model_by_name("NonExistent") is None

    def test_validate_model_data(self):
        result = validate_model_data(ValidationResult, {"is_valid": True})
        assert isinstance(result, ValidationResult)

    def test_validate_model_data_invalid(self):
        with pytest.raises(ValueError):
            validate_model_data(ReasoningStep, {"step_type": "invalid_type"})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
