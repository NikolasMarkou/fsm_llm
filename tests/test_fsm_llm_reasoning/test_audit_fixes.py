"""
Tests verifying fixes for audit findings in fsm_llm_reasoning.
Covers: F-003 (malformed user_message), F-005 (context pruning),
        F-011 (magic numbers extracted to constants).
"""
from __future__ import annotations

import json

from fsm_llm_reasoning.handlers import ContextManager, ReasoningHandlers
from fsm_llm_reasoning.constants import ContextKeys, Defaults


# ---------------------------------------------------------------------------
# F-003: Malformed user_message string
# ---------------------------------------------------------------------------

class TestMalformedUserMessage:
    """F-003: The sub-FSM continuation message must not contain ':{' artifact."""

    def test_continue_reasoning_message_is_clean(self):
        """Verify the malformed string was fixed in engine.py."""
        import inspect
        from fsm_llm_reasoning.engine import ReasoningEngine

        source = inspect.getsource(ReasoningEngine)
        # The old malformed string should NOT be present
        assert ':\\n:{' not in source
        assert 'Continue reasoning:\\n:{' not in source
        # The fixed string should be present
        assert 'Continue reasoning.' in source


# ---------------------------------------------------------------------------
# F-005: extract_relevant_context enforces max_size
# ---------------------------------------------------------------------------

class TestContextManagerEnforcement:
    """F-005: extract_relevant_context must enforce max_size, not just warn."""

    def test_enforce_max_size_removes_keys(self):
        """Context exceeding max_size should have keys removed."""
        source = {"key": "x" * 10000}
        result = ContextManager.extract_relevant_context(source, ["key"], max_size=100)
        size = len(json.dumps(result, default=str))
        assert size <= 100

    def test_enforce_max_size_keeps_fitting_keys(self):
        """Keys that fit within budget should be kept."""
        source = {"small": "ok", "big": "x" * 10000}
        result = ContextManager.extract_relevant_context(
            source, ["small", "big"], max_size=200
        )
        # small should fit, big should be removed
        assert "small" in result
        assert "big" not in result

    def test_no_max_size_returns_all(self):
        """Without max_size, all requested keys are returned."""
        source = {"a": "x" * 10000, "b": "y" * 10000}
        result = ContextManager.extract_relevant_context(source, ["a", "b"])
        assert "a" in result
        assert "b" in result

    def test_max_size_with_empty_context(self):
        """Empty context should return empty dict regardless of max_size."""
        result = ContextManager.extract_relevant_context({}, ["key"], max_size=10)
        assert result == {}


# ---------------------------------------------------------------------------
# F-011: Magic numbers extracted to constants
# ---------------------------------------------------------------------------

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
            assert len(result[ContextKeys.REASONING_TRACE]) <= Defaults.PRUNE_LIST_MAX_LENGTH

    def test_validate_solution_uses_min_solution_length(self):
        """Verify validate_solution uses MIN_SOLUTION_LENGTH constant."""
        context = {
            ContextKeys.PROPOSED_SOLUTION: "short",  # < MIN_SOLUTION_LENGTH
            ContextKeys.KEY_INSIGHTS: "some insight",
            ContextKeys.RETRY_COUNT: 0,
        }
        result = ReasoningHandlers.validate_solution(context)
        assert result[ContextKeys.SOLUTION_VALID] is False
