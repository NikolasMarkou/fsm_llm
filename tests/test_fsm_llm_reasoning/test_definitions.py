"""
Unit tests for reasoning engine Pydantic models.
"""

import json
from datetime import datetime

import pytest

from fsm_llm.context.filtering import redacting_json_default
from fsm_llm.reasoning.definitions import (
    ConfidenceLevel,
    ProblemContext,
    ReasoningClassificationResult,
    ReasoningStep,
    ReasoningStepType,
    ReasoningTrace,
    TimestampedModel,
    ValidationResult,
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
            step_type=ReasoningStepType.ANALYSIS, content="Breaking down the problem"
        )
        assert step.step_type == ReasoningStepType.ANALYSIS
        assert step.confidence == 0.0
        assert step.evidence == []

    def test_confidence_levels(self):
        low = ReasoningStep(
            step_type=ReasoningStepType.ANALYSIS, content="test", confidence=0.3
        )
        med = ReasoningStep(
            step_type=ReasoningStepType.ANALYSIS, content="test", confidence=0.6
        )
        high = ReasoningStep(
            step_type=ReasoningStepType.ANALYSIS, content="test", confidence=0.8
        )
        very_high = ReasoningStep(
            step_type=ReasoningStepType.ANALYSIS, content="test", confidence=0.95
        )

        assert low.confidence_level == ConfidenceLevel.LOW
        assert med.confidence_level == ConfidenceLevel.MEDIUM
        assert high.confidence_level == ConfidenceLevel.HIGH
        assert very_high.confidence_level == ConfidenceLevel.VERY_HIGH

    def test_has_evidence(self):
        no_ev = ReasoningStep(step_type=ReasoningStepType.ANALYSIS, content="test")
        has_ev = ReasoningStep(
            step_type=ReasoningStepType.ANALYSIS, content="test", evidence=["fact1"]
        )
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
            evidence=["valid", "", "  ", "also valid"],
        )
        assert step.evidence == ["valid", "also valid"]

    def test_confidence_bounds(self):
        with pytest.raises(ValueError):
            ReasoningStep(
                step_type=ReasoningStepType.ANALYSIS, content="test", confidence=1.5
            )
        with pytest.raises(ValueError):
            ReasoningStep(
                step_type=ReasoningStepType.ANALYSIS, content="test", confidence=-0.1
            )


class TestValidationResult:
    """Test ValidationResult model."""

    def test_basic_valid(self):
        vr = ValidationResult(
            is_valid=True, confidence=0.9, checks={"a": True, "b": True}
        )
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
            issues=["b failed"],
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
        complex_t = ReasoningTrace(
            steps=[{"x": 1}] * 15, reasoning_types_used={"a", "b", "c"}
        )
        highly = ReasoningTrace(
            steps=[{"x": 1}] * 25, reasoning_types_used={"a", "b", "c", "d"}
        )

        assert simple.reasoning_complexity == "simple"
        assert moderate.reasoning_complexity == "moderate"
        assert complex_t.reasoning_complexity == "complex"
        assert highly.reasoning_complexity == "highly_complex"

    def test_reasoning_types_dump_as_sorted_list(self):
        trace = ReasoningTrace(reasoning_types_used={"b", "a"})
        assert trace.model_dump()["reasoning_types_used"] == ["a", "b"]
        assert trace.model_dump(mode="json")["reasoning_types_used"] == ["a", "b"]

    def test_reasoning_types_dump_is_not_redacted_by_json_hook(self):
        dump = ReasoningTrace(reasoning_types_used={"analytical"}).model_dump()
        text = json.dumps(dump, default=redacting_json_default)
        assert "redacted" not in text
        assert json.loads(text)["reasoning_types_used"] == ["analytical"]

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
            recommended_type="analytical", justification="systematic analysis needed"
        )
        assert r.has_alternatives is False
        assert "analytical" in r.classification_summary

    def test_alternatives_deduped(self):
        r = ReasoningClassificationResult(
            recommended_type="analytical",
            justification="test",
            alternatives=["creative", "creative", "deductive"],
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
            constraints=["valid", "", "  ", "also valid"],
        )
        assert pc.constraints == ["valid", "also valid"]

    def test_high_priority(self):
        pc = ProblemContext(problem_statement="urgent task", priority="urgent")
        assert pc.is_high_priority is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


_SORTED_KEYS = ["alpha", "beta", "gamma", "mid", "zeta"]
_SET_PROBE = """
import json
from fsm_llm.reasoning.definitions import ReasoningStep
keys = {"alpha", "mid", "beta", "zeta", "gamma"}
step = ReasoningStep(step_type="analysis", content="a step", context_keys_used=keys)
print(json.dumps([
    json.loads(step.model_dump_json())["context_keys_used"],
    step.model_dump(mode="json")["context_keys_used"],
]))
"""


class TestSetFieldsSerializeSortedInJson:
    """F3 (reasoning audit 2.4c): a set of context keys is written to JSON as a
    sorted list, so two processes (two hash seeds) write the same bytes."""

    @pytest.mark.parametrize("seed", ["1", "2", "3"])
    def test_every_hash_seed_gives_the_same_sorted_list(self, seed):
        import os
        import subprocess
        import sys

        env = {**os.environ, "PYTHONHASHSEED": seed}
        done = subprocess.run(
            [sys.executable, "-c", _SET_PROBE],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )

        assert json.loads(done.stdout) == [_SORTED_KEYS] * 2

    def test_python_mode_keeps_the_set(self):
        from fsm_llm.reasoning.definitions import ReasoningStep

        keys = {"b", "a"}
        step = ReasoningStep(
            step_type="analysis", content="a step", context_keys_used=keys
        )

        assert step.model_dump()["context_keys_used"] == keys
