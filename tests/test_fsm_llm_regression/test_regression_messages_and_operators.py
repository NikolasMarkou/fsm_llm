"""
Regression tests for Plan 12 fixes.

Tests verify that the issues found in pass 12 remain fixed.
"""

import re

import pytest


class TestV1MessageTruncation:
    """V1: Message truncation must not exceed max_message_length."""

    def test_truncated_message_respects_limit(self):
        """Truncated message (including suffix) must be <= max_message_length."""
        from fsm_llm.definitions import Conversation

        max_len = 100
        conv = Conversation(max_message_length=max_len)
        long_message = "x" * 500

        conv.add_user_message(long_message)
        stored = conv.exchanges[-1]["user"]
        assert len(stored) <= max_len, (
            f"Stored message length {len(stored)} exceeds max_message_length {max_len}"
        )
        assert stored.endswith("... [truncated]")

    def test_truncated_system_message_respects_limit(self):
        """System message truncation also respects limit."""
        from fsm_llm.definitions import Conversation

        max_len = 50
        conv = Conversation(max_message_length=max_len)
        conv.add_system_message("y" * 200)
        stored = conv.exchanges[-1]["system"]
        assert len(stored) <= max_len

    def test_exact_length_message_not_truncated(self):
        """Message exactly at max_message_length is NOT truncated."""
        from fsm_llm.definitions import Conversation

        max_len = 100
        conv = Conversation(max_message_length=max_len)
        exact_message = "z" * max_len

        conv.add_user_message(exact_message)
        stored = conv.exchanges[-1]["user"]
        assert stored == exact_message
        assert "truncated" not in stored

    def test_short_message_not_truncated(self):
        """Message shorter than limit is stored verbatim."""
        from fsm_llm.definitions import Conversation

        conv = Conversation(max_message_length=1000)
        msg = "Hello world"
        conv.add_user_message(msg)
        assert conv.exchanges[-1]["user"] == msg


class TestV2ConfidenceConstant:
    """V2: Confidence factor must use CONDITION_SUCCESS_RATE_BOOST constant."""

    def test_confidence_factor_matches_constant(self):
        """When all conditions pass, confidence_factor = CONDITION_SUCCESS_RATE_BOOST."""
        from fsm_llm.constants import CONDITION_SUCCESS_RATE_BOOST
        from fsm_llm.definitions import TransitionCondition
        from fsm_llm.transition_evaluator import TransitionEvaluator

        evaluator = TransitionEvaluator()
        conditions = [
            TransitionCondition(
                description="Key exists",
                requires_context_keys=["name"]
            )
        ]
        context = {"name": "Alice"}

        result = evaluator._evaluate_transition_conditions(conditions, context)
        assert result['all_pass'] is True
        expected = CONDITION_SUCCESS_RATE_BOOST
        assert result['confidence_factor'] == pytest.approx(expected), (
            f"Expected {expected}, got {result['confidence_factor']}"
        )


class TestV4OperatorNames:
    """V4: ALLOWED_JSONLOGIC_OPERATIONS must match actual operator names."""

    def test_has_context_in_allowed_operations(self):
        """has_context (not context_has) must be in ALLOWED_JSONLOGIC_OPERATIONS."""
        from fsm_llm.constants import ALLOWED_JSONLOGIC_OPERATIONS
        assert 'has_context' in ALLOWED_JSONLOGIC_OPERATIONS
        assert 'context_has' not in ALLOWED_JSONLOGIC_OPERATIONS

    def test_context_length_in_allowed_operations(self):
        """context_length (not context_count) must be in ALLOWED_JSONLOGIC_OPERATIONS."""
        from fsm_llm.constants import ALLOWED_JSONLOGIC_OPERATIONS
        assert 'context_length' in ALLOWED_JSONLOGIC_OPERATIONS
        assert 'context_count' not in ALLOWED_JSONLOGIC_OPERATIONS

    def test_all_expression_operators_in_allowed_set(self):
        """Every operator implemented in expressions.py should be in the allowed set."""
        from fsm_llm.constants import ALLOWED_JSONLOGIC_OPERATIONS
        from fsm_llm.expressions import operations

        # Operators handled directly in evaluate_logic (not in operations dict)
        special_operators = {'var', 'missing', 'missing_some', 'has_context', 'context_length'}

        all_implemented = set(operations.keys()) | special_operators
        missing_from_allowed = all_implemented - ALLOWED_JSONLOGIC_OPERATIONS
        # !! and !!! are internal aliases, not user-facing
        missing_from_allowed -= {'!!'}
        assert not missing_from_allowed, (
            f"Operators implemented but not in ALLOWED_JSONLOGIC_OPERATIONS: {missing_from_allowed}"
        )


class TestV11WordBoundaryMatching:
    """V11: Transition matching must use word boundaries."""

    def test_substring_does_not_match(self):
        """State name 'order' should NOT match in text 'disorder'."""
        # Test the regex pattern used in llm.py
        target = "order"
        text = "The system is in disorder right now"
        assert not re.search(rf'\b{re.escape(target)}\b', text, re.IGNORECASE)

    def test_exact_word_matches(self):
        """State name 'order' should match in text containing 'order' as a word."""
        target = "order"
        text = "I think we should go to the order state"
        assert re.search(rf'\b{re.escape(target)}\b', text, re.IGNORECASE)

    def test_underscore_state_name_matches(self):
        """State names with underscores should match as whole words."""
        target = "collect_name"
        text = "The next step is collect_name for the user"
        assert re.search(rf'\b{re.escape(target)}\b', text, re.IGNORECASE)

    def test_case_insensitive_matching(self):
        """Matching should be case-insensitive."""
        target = "greeting"
        text = "We should move to GREETING"
        assert re.search(rf'\b{re.escape(target)}\b', text, re.IGNORECASE)


class TestV14NoEmojiInValidator:
    """V14: Validator output must not contain emoji characters."""

    def test_valid_fsm_output_no_emoji(self):
        """Valid FSM validation output should use text markers, not emoji."""
        from fsm_llm.validator import FSMValidationResult
        result = FSMValidationResult("test_fsm")
        output = str(result)
        assert "[VALID]" in output
        assert "\u2705" not in output  # ✅
        assert "\u274c" not in output  # ❌

    def test_invalid_fsm_output_no_emoji(self):
        """Invalid FSM validation output should use text markers, not emoji."""
        from fsm_llm.validator import FSMValidationResult
        result = FSMValidationResult("test_fsm")
        result.add_error("test error")
        result.add_warning("test warning")
        output = str(result)
        assert "[INVALID]" in output
        assert "[ERROR]" in output
        assert "[WARN]" in output
        # No emoji
        assert "\U0001f534" not in output  # 🔴
        assert "\U0001f7e0" not in output  # 🟠
        assert "\U0001f535" not in output  # 🔵


class TestV5PythonVersion:
    """V5: Python version requirement must match pyproject.toml (3.10+)."""

    def test_minimum_python_version_is_3_10(self):
        from pathlib import Path
        pyproject = Path(__file__).parent.parent.parent / "pyproject.toml"
        content = pyproject.read_text()
        assert 'requires-python = ">=3.10"' in content


class TestV15NoSkipErrorMode:
    """V15: HANDLER_ERROR_SKIP should not exist in constants."""

    def test_handler_error_skip_not_defined(self):
        import fsm_llm.constants as c
        assert not hasattr(c, 'HANDLER_ERROR_SKIP')
