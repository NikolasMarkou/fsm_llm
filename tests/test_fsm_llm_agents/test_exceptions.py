from __future__ import annotations

"""Tests for fsm_llm_agents.exceptions module."""

import pytest

from fsm_llm.definitions import FSMError
from fsm_llm_agents.exceptions import (
    AgentError,
    AgentTimeoutError,
    ApprovalDeniedError,
    BudgetExhaustedError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolValidationError,
)


class TestExceptionHierarchy:
    """Tests for exception class hierarchy."""

    def test_agent_error_is_fsm_error(self):
        e = AgentError("test")
        assert isinstance(e, FSMError)

    def test_tool_execution_error_is_agent_error(self):
        e = ToolExecutionError("failed", tool_name="search")
        assert isinstance(e, AgentError)
        assert e.tool_name == "search"

    def test_tool_not_found_error(self):
        e = ToolNotFoundError("search")
        assert isinstance(e, AgentError)
        assert e.tool_name == "search"
        assert "search" in str(e)

    def test_tool_validation_error(self):
        e = ToolValidationError("search", "missing param")
        assert isinstance(e, AgentError)
        assert e.tool_name == "search"
        assert "missing param" in str(e)

    def test_budget_exhausted_error(self):
        e = BudgetExhaustedError("iterations", 10)
        assert isinstance(e, AgentError)
        assert e.budget_type == "iterations"
        assert e.limit == 10

    def test_approval_denied_error(self):
        e = ApprovalDeniedError("send_email")
        assert isinstance(e, AgentError)
        assert "send_email" in str(e)

    def test_agent_timeout_error(self):
        e = AgentTimeoutError(300.0)
        assert isinstance(e, AgentError)
        assert e.timeout_seconds == 300.0
        assert "300.0" in str(e)

    def test_agent_error_with_details(self):
        e = AgentError("test", details={"key": "value"})
        assert isinstance(e, FSMError)

    def test_all_exceptions_catchable_as_agent_error(self):
        exceptions = [
            ToolExecutionError("x"),
            ToolNotFoundError("x"),
            ToolValidationError("x", "y"),
            BudgetExhaustedError("x", 1),
            ApprovalDeniedError("x"),
            AgentTimeoutError(1.0),
        ]
        for exc in exceptions:
            with pytest.raises(AgentError):
                raise exc
