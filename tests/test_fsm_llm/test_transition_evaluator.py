"""
Dedicated unit tests for TransitionEvaluator.

Tests the core transition evaluation logic: DETERMINISTIC, AMBIGUOUS, and BLOCKED outcomes,
condition evaluation, confidence scoring, and configuration.
"""

import pytest

from fsm_llm._models import TransitionEvaluationResult
from fsm_llm.dialog.definitions import (
    FSMContext,
    State,
    Transition,
    TransitionCondition,
)
from fsm_llm.dialog.transition_evaluator import (
    TransitionEvaluator,
    TransitionEvaluatorConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_condition(description="cond", logic=None, requires_keys=None, priority=0):
    return TransitionCondition(
        description=description,
        logic=logic,
        requires_context_keys=requires_keys or [],
        evaluation_priority=priority,
    )


def _make_transition(
    target, priority=100, conditions=None, description="", llm_description=None
):
    return Transition(
        target_state=target,
        description=description or f"Go to {target}",
        priority=priority,
        conditions=conditions or [],
        llm_description=llm_description,
    )


def _make_state(state_id, transitions=None):
    return State(
        id=state_id,
        description=f"State {state_id}",
        purpose=f"Purpose of {state_id}",
        extraction_instructions="Extract data",
        response_instructions="Respond",
        transitions=transitions or [],
    )


def _make_context(data=None):
    ctx = FSMContext()
    if data:
        ctx.data.update(data)
    return ctx


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTransitionEvaluatorConfig:
    """Test configuration defaults and customisation."""

    def test_default_config(self):
        config = TransitionEvaluatorConfig()
        assert config.ambiguity_threshold == 0.1
        assert config.minimum_confidence == 0.5
        assert config.strict_condition_matching is True
        assert config.detailed_logging is False

    def test_custom_config(self):
        config = TransitionEvaluatorConfig(
            ambiguity_threshold=0.2,
            minimum_confidence=0.7,
            strict_condition_matching=False,
            detailed_logging=True,
        )
        assert config.ambiguity_threshold == 0.2
        assert config.minimum_confidence == 0.7


class TestDeterministicTransitions:
    """Single clear winner should yield DETERMINISTIC."""

    def test_single_transition_no_conditions(self):
        """One transition with no conditions -> DETERMINISTIC."""
        evaluator = TransitionEvaluator()
        state = _make_state("start", [_make_transition("end", priority=100)])
        result = evaluator.evaluate_transitions(state, _make_context())

        assert result.result_type == TransitionEvaluationResult.DETERMINISTIC
        assert result.deterministic_transition == "end"
        assert result.confidence > 0

    def test_single_transition_with_passing_condition(self):
        """Transition with satisfied condition -> DETERMINISTIC with boosted confidence."""
        cond = _make_condition(
            description="name exists",
            logic={"==": [{"var": "name"}, "Alice"]},
            requires_keys=["name"],
        )
        evaluator = TransitionEvaluator()
        state = _make_state("start", [_make_transition("end", conditions=[cond])])
        result = evaluator.evaluate_transitions(state, _make_context({"name": "Alice"}))

        assert result.result_type == TransitionEvaluationResult.DETERMINISTIC
        assert result.deterministic_transition == "end"
        assert result.confidence > 0.5

    def test_clear_winner_by_confidence_gap(self):
        """Two transitions but clear confidence gap -> DETERMINISTIC."""
        cond = _make_condition(logic={"==": [{"var": "x"}, 1]}, requires_keys=["x"])
        t1 = _make_transition("win", priority=100, conditions=[cond])
        t2 = _make_transition("lose", priority=500)  # lower priority = lower confidence
        evaluator = TransitionEvaluator()
        state = _make_state("start", [t1, t2])
        result = evaluator.evaluate_transitions(state, _make_context({"x": 1}))

        assert result.result_type == TransitionEvaluationResult.DETERMINISTIC
        assert result.deterministic_transition == "win"

    def test_priority_determines_winner_with_large_gap(self):
        """Two unconditioned transitions with large priority gap -> DETERMINISTIC."""
        t1 = _make_transition("first", priority=100)
        t2 = _make_transition("second", priority=900)  # much lower confidence
        evaluator = TransitionEvaluator()
        state = _make_state("start", [t1, t2])
        result = evaluator.evaluate_transitions(state, _make_context())

        assert result.result_type == TransitionEvaluationResult.DETERMINISTIC
        assert result.deterministic_transition == "first"


class TestAmbiguousTransitions:
    """Multiple valid options with similar confidence -> AMBIGUOUS."""

    def test_ambiguous_same_priority(self):
        """Two transitions with same priority and no distinguishing conditions."""
        t1 = _make_transition("a", priority=100)
        t2 = _make_transition("b", priority=100)
        evaluator = TransitionEvaluator()
        state = _make_state("start", [t1, t2])
        result = evaluator.evaluate_transitions(state, _make_context())

        assert result.result_type == TransitionEvaluationResult.AMBIGUOUS
        assert len(result.available_options) == 2
        targets = {opt.target_state for opt in result.available_options}
        assert targets == {"a", "b"}

    def test_ambiguous_close_confidence(self):
        """Transitions with close priority values -> AMBIGUOUS when gap < threshold."""
        t1 = _make_transition("a", priority=100)
        t2 = _make_transition("b", priority=110)
        evaluator = TransitionEvaluator(
            TransitionEvaluatorConfig(ambiguity_threshold=0.5)
        )
        state = _make_state("start", [t1, t2])
        result = evaluator.evaluate_transitions(state, _make_context())

        assert result.result_type == TransitionEvaluationResult.AMBIGUOUS

    def test_ambiguous_options_use_llm_description(self):
        """Ambiguous options should prefer llm_description when available."""
        t1 = _make_transition(
            "a", priority=100, description="regular", llm_description="LLM desc A"
        )
        t2 = _make_transition("b", priority=100, description="regular")
        evaluator = TransitionEvaluator()
        state = _make_state("start", [t1, t2])
        result = evaluator.evaluate_transitions(state, _make_context())

        assert result.result_type == TransitionEvaluationResult.AMBIGUOUS
        descriptions = {opt.description for opt in result.available_options}
        assert "LLM desc A" in descriptions


class TestBlockedTransitions:
    """All transitions fail -> BLOCKED."""

    def test_blocked_missing_required_key(self):
        """No transitions pass when required context key is missing."""
        cond = _make_condition(
            requires_keys=["missing_key"], logic={"==": [{"var": "missing_key"}, "x"]}
        )
        t = _make_transition("end", conditions=[cond])
        evaluator = TransitionEvaluator()
        state = _make_state("start", [t])
        result = evaluator.evaluate_transitions(state, _make_context())

        assert result.result_type == TransitionEvaluationResult.BLOCKED
        assert result.confidence == 0.0
        assert result.blocked_reason

    def test_blocked_failing_logic(self):
        """Transition blocked by failing JsonLogic condition."""
        cond = _make_condition(
            logic={"==": [{"var": "x"}, "expected"]}, requires_keys=["x"]
        )
        t = _make_transition("end", conditions=[cond])
        evaluator = TransitionEvaluator()
        state = _make_state("start", [t])
        result = evaluator.evaluate_transitions(state, _make_context({"x": "wrong"}))

        assert result.result_type == TransitionEvaluationResult.BLOCKED

    def test_blocked_no_transitions(self):
        """Terminal state with no transitions -> BLOCKED."""
        evaluator = TransitionEvaluator()
        state = _make_state("terminal", [])
        result = evaluator.evaluate_transitions(state, _make_context())

        assert result.result_type == TransitionEvaluationResult.BLOCKED

    def test_blocked_reason_includes_failed_conditions(self):
        """Blocked reason should describe which conditions failed."""
        cond = _make_condition(
            description="user provided email", requires_keys=["email"]
        )
        t = _make_transition("next", conditions=[cond])
        evaluator = TransitionEvaluator()
        state = _make_state("start", [t])
        result = evaluator.evaluate_transitions(state, _make_context())

        assert result.result_type == TransitionEvaluationResult.BLOCKED
        assert (
            "email" in result.blocked_reason.lower()
            or "user provided email" in result.blocked_reason
        )


class TestConditionEvaluation:
    """Test individual condition evaluation logic."""

    def test_condition_without_logic_passes_if_keys_present(self):
        """Condition with no logic should pass if required keys exist."""
        cond = _make_condition(requires_keys=["name"])
        t = _make_transition("next", conditions=[cond])
        evaluator = TransitionEvaluator()
        state = _make_state("start", [t])
        result = evaluator.evaluate_transitions(state, _make_context({"name": "test"}))

        assert result.result_type == TransitionEvaluationResult.DETERMINISTIC

    def test_condition_with_nested_var(self):
        """JsonLogic with nested variable access."""
        cond = _make_condition(logic={"==": [{"var": "user.age"}, 25]})
        t = _make_transition("next", conditions=[cond])
        evaluator = TransitionEvaluator()
        state = _make_state("start", [t])
        result = evaluator.evaluate_transitions(
            state, _make_context({"user": {"age": 25}})
        )

        assert result.result_type == TransitionEvaluationResult.DETERMINISTIC

    def test_strict_condition_matching_early_exit(self):
        """Strict mode should stop evaluating on first failure."""
        cond1 = _make_condition(
            description="always fails", requires_keys=["nonexistent"], priority=0
        )
        cond2 = _make_condition(description="always passes", priority=1)
        t = _make_transition("next", conditions=[cond1, cond2])
        evaluator = TransitionEvaluator(
            TransitionEvaluatorConfig(strict_condition_matching=True)
        )
        state = _make_state("start", [t])
        result = evaluator.evaluate_transitions(state, _make_context())

        assert result.result_type == TransitionEvaluationResult.BLOCKED

    def test_multiple_conditions_all_must_pass(self):
        """All conditions must pass for transition to be valid."""
        cond1 = _make_condition(requires_keys=["a"])
        cond2 = _make_condition(requires_keys=["b"])
        t = _make_transition("next", conditions=[cond1, cond2])
        evaluator = TransitionEvaluator()
        state = _make_state("start", [t])

        # Only one key -> BLOCKED
        result_partial = evaluator.evaluate_transitions(state, _make_context({"a": 1}))
        assert result_partial.result_type == TransitionEvaluationResult.BLOCKED

        # Both keys -> DETERMINISTIC
        result_full = evaluator.evaluate_transitions(
            state, _make_context({"a": 1, "b": 2})
        )
        assert result_full.result_type == TransitionEvaluationResult.DETERMINISTIC


class TestConfidenceScoring:
    """Test confidence calculation mechanics."""

    def test_lower_priority_gives_higher_confidence(self):
        """Lower priority value = higher base confidence."""
        evaluator = TransitionEvaluator()
        t_high = _make_transition("high", priority=100)
        t_low = _make_transition("low", priority=500)
        state = _make_state("start", [t_high, t_low])
        result = evaluator.evaluate_transitions(state, _make_context())

        # High priority transition should be selected
        assert result.deterministic_transition == "high"

    def test_condition_boost_increases_confidence(self):
        """Passing conditions should boost confidence above base."""
        cond = _make_condition(logic={"==": [{"var": "x"}, 1]}, requires_keys=["x"])
        # Priority 100 with condition boost: 0.9 * 1.5 = 1.0 (capped)
        t_with_cond = _make_transition("conditioned", priority=100, conditions=[cond])
        # Priority 500 without conditions: 0.5 base confidence
        t_no_cond = _make_transition("plain", priority=500)
        evaluator = TransitionEvaluator()
        state = _make_state("start", [t_with_cond, t_no_cond])
        result = evaluator.evaluate_transitions(state, _make_context({"x": 1}))

        # Condition boost on same-priority transition should clearly win
        assert result.result_type == TransitionEvaluationResult.DETERMINISTIC
        assert result.deterministic_transition == "conditioned"

    def test_failed_condition_severely_reduces_confidence(self):
        """Failed conditions should reduce confidence by 90%."""
        cond = _make_condition(requires_keys=["missing"])
        t_fail = _make_transition("fail", priority=100, conditions=[cond])
        t_pass = _make_transition("pass", priority=500)
        evaluator = TransitionEvaluator()
        state = _make_state("start", [t_fail, t_pass])
        result = evaluator.evaluate_transitions(state, _make_context())

        # The failing condition transition should lose to the plain one
        assert result.deterministic_transition == "pass"


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_evaluation_error_in_condition_continues(self):
        """Malformed logic should not crash the evaluator."""
        cond = _make_condition(logic={"invalid_op": "bad"})
        t = _make_transition("next", conditions=[cond])
        evaluator = TransitionEvaluator()
        state = _make_state("start", [t])
        # Should not raise - evaluator gracefully handles errors
        result = evaluator.evaluate_transitions(state, _make_context())
        assert result.result_type in (
            TransitionEvaluationResult.BLOCKED,
            TransitionEvaluationResult.DETERMINISTIC,
        )

    def test_empty_context(self):
        """Evaluator should work with empty context."""
        t = _make_transition("next", priority=100)
        evaluator = TransitionEvaluator()
        state = _make_state("start", [t])
        result = evaluator.evaluate_transitions(state, _make_context())

        assert result.result_type == TransitionEvaluationResult.DETERMINISTIC

    def test_extracted_data_merged_into_context(self):
        """Extracted data should be available during condition evaluation."""
        cond = _make_condition(
            logic={"==": [{"var": "extracted_key"}, "yes"]},
            requires_keys=["extracted_key"],
        )
        t = _make_transition("next", conditions=[cond])
        evaluator = TransitionEvaluator()
        state = _make_state("start", [t])
        result = evaluator.evaluate_transitions(
            state, _make_context(), extracted_data={"extracted_key": "yes"}
        )

        assert result.result_type == TransitionEvaluationResult.DETERMINISTIC
        assert result.deterministic_transition == "next"

    def test_detailed_logging_mode(self):
        """Detailed logging mode should not change evaluation results."""
        config = TransitionEvaluatorConfig(detailed_logging=True)
        evaluator = TransitionEvaluator(config)
        t = _make_transition("next", priority=100)
        state = _make_state("start", [t])
        result = evaluator.evaluate_transitions(state, _make_context())

        assert result.result_type == TransitionEvaluationResult.DETERMINISTIC

    def test_minimum_confidence_threshold(self):
        """Transitions below minimum confidence should not be DETERMINISTIC."""
        config = TransitionEvaluatorConfig(minimum_confidence=0.99)
        evaluator = TransitionEvaluator(config)
        # Two transitions with similar priority - neither above 0.99
        t1 = _make_transition("a", priority=100)
        t2 = _make_transition("b", priority=150)
        state = _make_state("start", [t1, t2])
        result = evaluator.evaluate_transitions(state, _make_context())

        # With very high minimum confidence, should be AMBIGUOUS
        assert result.result_type == TransitionEvaluationResult.AMBIGUOUS


class TestWorkingContextPreparation:
    """Test context merging for evaluation."""

    def test_extracted_data_overrides_existing_context(self):
        """Extracted data should override existing context values."""
        cond = _make_condition(
            logic={"==": [{"var": "status"}, "updated"]}, requires_keys=["status"]
        )
        t = _make_transition("next", conditions=[cond])
        evaluator = TransitionEvaluator()
        state = _make_state("start", [t])
        result = evaluator.evaluate_transitions(
            state,
            _make_context({"status": "old"}),
            extracted_data={"status": "updated"},
        )

        assert result.result_type == TransitionEvaluationResult.DETERMINISTIC

    def test_none_extracted_data(self):
        """None extracted_data should not cause errors."""
        t = _make_transition("next")
        evaluator = TransitionEvaluator()
        state = _make_state("start", [t])
        result = evaluator.evaluate_transitions(
            state, _make_context(), extracted_data=None
        )

        assert result.result_type == TransitionEvaluationResult.DETERMINISTIC


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
