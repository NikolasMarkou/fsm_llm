from __future__ import annotations

"""Tests for fsm_llm_agents.definitions module."""

import pytest

from fsm_llm.stdlib.agents.definitions import (
    AgentConfig,
    AgentResult,
    AgentStep,
    AgentTrace,
    ApprovalRequest,
    ToolCall,
    ToolDefinition,
    ToolResult,
)


class TestToolDefinition:
    """Tests for ToolDefinition model."""

    def test_basic_creation(self):
        tool = ToolDefinition(name="search", description="Search the web")
        assert tool.name == "search"
        assert tool.description == "Search the web"
        assert tool.parameter_schema == {}
        assert tool.requires_approval is False
        assert tool.execute_fn is None

    def test_with_schema(self):
        schema = {"properties": {"query": {"type": "string"}}}
        tool = ToolDefinition(
            name="search",
            description="Search",
            parameter_schema=schema,
        )
        assert tool.parameter_schema == schema

    def test_with_execute_fn(self):
        fn = lambda params: "result"  # noqa: E731
        tool = ToolDefinition(
            name="search",
            description="Search",
            execute_fn=fn,
        )
        assert tool.execute_fn is fn

    def test_invalid_name_special_chars(self):
        with pytest.raises(ValueError, match="alphanumeric"):
            ToolDefinition(name="bad-name!", description="Bad")

    def test_valid_name_underscores(self):
        tool = ToolDefinition(name="web_search", description="Search")
        assert tool.name == "web_search"

    def test_requires_approval(self):
        tool = ToolDefinition(
            name="delete", description="Delete", requires_approval=True
        )
        assert tool.requires_approval is True


class TestToolCall:
    """Tests for ToolCall model."""

    def test_basic_creation(self):
        call = ToolCall(tool_name="search", parameters={"query": "test"})
        assert call.tool_name == "search"
        assert call.parameters == {"query": "test"}
        assert call.reasoning == ""

    def test_with_reasoning(self):
        call = ToolCall(
            tool_name="calc",
            parameters={"expr": "2+2"},
            reasoning="Need to compute",
        )
        assert call.reasoning == "Need to compute"


class TestToolResult:
    """Tests for ToolResult model."""

    def test_success_result(self):
        result = ToolResult(
            tool_name="search",
            success=True,
            result="Found: Paris has 2M people",
            execution_time_ms=150.0,
        )
        assert result.success is True
        assert "Paris" in result.summary

    def test_failure_result(self):
        result = ToolResult(
            tool_name="search",
            success=False,
            error="Network timeout",
        )
        assert result.success is False
        assert "Error: Network timeout" in result.summary

    def test_summary_truncation(self):
        long_result = "x" * 3000
        result = ToolResult(tool_name="t", success=True, result=long_result)
        assert len(result.summary) < 3000
        assert "truncated" in result.summary


class TestAgentStep:
    """Tests for AgentStep model."""

    def test_basic_creation(self):
        step = AgentStep(
            iteration=1,
            thought="Need to search",
            action="search({query: test})",
            observation="Found results",
        )
        assert step.iteration == 1
        assert step.timestamp is not None


class TestAgentTrace:
    """Tests for AgentTrace model."""

    def test_empty_trace(self):
        trace = AgentTrace()
        assert trace.total_iterations == 0
        assert trace.tools_used == []

    def test_tools_used(self):
        trace = AgentTrace(
            tool_calls=[
                ToolCall(tool_name="search", parameters={}),
                ToolCall(tool_name="calc", parameters={}),
                ToolCall(tool_name="search", parameters={"q": "2"}),
            ],
            total_iterations=3,
        )
        assert set(trace.tools_used) == {"search", "calc"}


class TestAgentConfig:
    """Tests for AgentConfig model."""

    def test_defaults(self):
        config = AgentConfig()
        assert config.max_iterations == 10
        assert config.timeout_seconds == 300.0
        assert config.temperature == 0.5

    def test_invalid_max_iterations(self):
        with pytest.raises(ValueError, match="at least 1"):
            AgentConfig(max_iterations=0)

    def test_custom_values(self):
        config = AgentConfig(model="gpt-4", max_iterations=20, timeout_seconds=60.0)
        assert config.model == "gpt-4"
        assert config.max_iterations == 20


class TestAgentResult:
    """Tests for AgentResult model."""

    def test_basic_result(self):
        result = AgentResult(answer="42", success=True)
        assert result.answer == "42"
        assert result.iterations_used == 0
        assert result.tools_used == []

    def test_with_trace(self):
        trace = AgentTrace(
            tool_calls=[ToolCall(tool_name="calc", parameters={})],
            total_iterations=3,
        )
        result = AgentResult(answer="42", success=True, trace=trace)
        assert result.iterations_used == 3
        assert result.tools_used == ["calc"]


class TestApprovalRequest:
    """Tests for ApprovalRequest model."""

    def test_basic_request(self):
        req = ApprovalRequest(
            tool_name="send_email",
            parameters={"to": "user@example.com"},
            reasoning="Need to notify",
        )
        assert req.tool_name == "send_email"
        assert req.parameters["to"] == "user@example.com"
