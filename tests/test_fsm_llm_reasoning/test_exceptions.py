"""
Unit tests for reasoning engine exception hierarchy.
"""

import pytest

from fsm_llm_reasoning.exceptions import (
    ReasoningClassificationError,
    ReasoningEngineError,
    ReasoningExecutionError,
)


class TestReasoningEngineError:
    """Test base exception."""

    def test_basic(self):
        err = ReasoningEngineError("test error")
        assert str(err) == "test error"
        assert err.details == {}

    def test_with_details(self):
        err = ReasoningEngineError("error", details={"key": "value"})
        assert err.details == {"key": "value"}

    def test_is_exception(self):
        assert issubclass(ReasoningEngineError, Exception)


class TestReasoningExecutionError:
    """Test execution error."""

    def test_basic(self):
        err = ReasoningExecutionError("exec failed")
        assert str(err) == "exec failed"
        assert err.reasoning_type is None

    def test_with_reasoning_type(self):
        err = ReasoningExecutionError("failed", reasoning_type="analytical")
        assert err.reasoning_type == "analytical"

    def test_inherits_from_base(self):
        err = ReasoningExecutionError("test")
        assert isinstance(err, ReasoningEngineError)


class TestReasoningClassificationError:
    """Test classification error."""

    def test_basic(self):
        err = ReasoningClassificationError("classification failed")
        assert isinstance(err, ReasoningEngineError)

    def test_catch_as_base(self):
        """Should be catchable as ReasoningEngineError."""
        with pytest.raises(ReasoningEngineError):
            raise ReasoningClassificationError("test")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
