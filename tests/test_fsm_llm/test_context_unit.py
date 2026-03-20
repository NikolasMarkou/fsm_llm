"""Unit tests for fsm_llm.context module."""
from __future__ import annotations

from fsm_llm.context import clean_context_keys


class TestCleanContextKeys:
    """Tests for clean_context_keys function."""

    def test_removes_none_values(self):
        """Keys with None values should be removed by default."""
        data = {"name": "Alice", "age": None, "city": "NYC"}
        result = clean_context_keys(data, "test-conv")
        assert result == {"name": "Alice", "city": "NYC"}

    def test_preserves_none_values_when_disabled(self):
        """None values should be preserved when remove_none_values=False."""
        data = {"name": "Alice", "age": None}
        result = clean_context_keys(data, "test-conv", remove_none_values=False)
        assert result == {"name": "Alice", "age": None}

    def test_removes_internal_prefix_underscore(self):
        """Keys starting with _ should be removed."""
        data = {"name": "Alice", "_internal": "hidden"}
        result = clean_context_keys(data, "test-conv")
        assert result == {"name": "Alice"}

    def test_removes_internal_prefix_system(self):
        """Keys starting with system_ should be removed."""
        data = {"name": "Alice", "system_state": "active"}
        result = clean_context_keys(data, "test-conv")
        assert result == {"name": "Alice"}

    def test_removes_internal_prefix_internal(self):
        """Keys starting with internal_ should be removed."""
        data = {"name": "Alice", "internal_id": "123"}
        result = clean_context_keys(data, "test-conv")
        assert result == {"name": "Alice"}

    def test_removes_internal_prefix_dunder(self):
        """Keys starting with __ should be removed."""
        data = {"name": "Alice", "__class": "user"}
        result = clean_context_keys(data, "test-conv")
        assert result == {"name": "Alice"}

    def test_preserves_empty_lists(self):
        """Empty lists are semantically meaningful and should be preserved."""
        data = {"allergies": [], "name": "Alice"}
        result = clean_context_keys(data, "test-conv")
        assert result == {"allergies": [], "name": "Alice"}

    def test_preserves_empty_strings(self):
        """Empty strings are preserved."""
        data = {"notes": "", "name": "Alice"}
        result = clean_context_keys(data, "test-conv")
        assert result == {"notes": "", "name": "Alice"}

    def test_preserves_zero_values(self):
        """Zero and False should be preserved."""
        data = {"count": 0, "active": False, "name": "Alice"}
        result = clean_context_keys(data, "test-conv")
        assert result == {"count": 0, "active": False, "name": "Alice"}

    def test_empty_dict(self):
        """Empty dict input returns empty dict."""
        result = clean_context_keys({}, "test-conv")
        assert result == {}

    def test_all_internal_keys(self):
        """Dict with only internal keys returns empty dict."""
        data = {"_a": 1, "system_b": 2, "__c": 3}
        result = clean_context_keys(data, "test-conv")
        assert result == {}

    def test_warns_on_forbidden_patterns(self):
        """Keys matching forbidden security patterns should trigger a warning."""
        data = {"password": "secret123", "name": "Alice"}
        # Should not raise, just warn
        result = clean_context_keys(data, "test-conv")
        # Forbidden keys are warned but NOT removed
        assert "password" in result
        assert "name" in result

    def test_warns_on_token_pattern(self):
        """Token-related keys should trigger a warning."""
        data = {"auth_token": "abc123"}
        result = clean_context_keys(data, "test-conv")
        assert "auth_token" in result

    def test_mixed_removal_and_preservation(self):
        """Combined scenario with internal, None, and valid keys."""
        data = {
            "name": "Alice",
            "age": None,
            "_internal": "hidden",
            "system_flag": True,
            "hobbies": [],
            "score": 0,
        }
        result = clean_context_keys(data, "test-conv")
        assert result == {"name": "Alice", "hobbies": [], "score": 0}
