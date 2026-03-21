from __future__ import annotations

"""Tests for fsm_llm_agents.handlers module."""

from fsm_llm_agents.constants import ContextKeys, Defaults
from fsm_llm_agents.handlers import AgentHandlers
from fsm_llm_agents.tools import ToolRegistry


def _echo(params):
    """Echo the input."""
    return f"Echo: {params.get('input', '')}"


def _add(params):
    """Add a and b."""
    return params["a"] + params["b"]


def _make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_function(_echo, name="echo", description="Echo input")
    registry.register_function(_add, name="add", description="Add numbers")
    return registry


class TestAgentHandlers:
    """Tests for AgentHandlers."""

    def test_execute_tool_success(self):
        registry = _make_registry()
        handlers = AgentHandlers(registry)

        context = {
            ContextKeys.TOOL_NAME: "echo",
            ContextKeys.TOOL_INPUT: {"input": "hello"},
            ContextKeys.REASONING: "Testing echo",
            ContextKeys.OBSERVATIONS: [],
        }
        result = handlers.execute_tool(context)

        assert result[ContextKeys.TOOL_STATUS] == "success"
        assert "Echo: hello" in result[ContextKeys.TOOL_RESULT]
        assert len(result[ContextKeys.OBSERVATIONS]) == 1
        assert result[ContextKeys.OBSERVATION_COUNT] == 1

    def test_execute_tool_with_kwargs(self):
        registry = _make_registry()
        handlers = AgentHandlers(registry)

        context = {
            ContextKeys.TOOL_NAME: "add",
            ContextKeys.TOOL_INPUT: {"a": 3, "b": 7},
            ContextKeys.OBSERVATIONS: [],
        }
        result = handlers.execute_tool(context)

        assert result[ContextKeys.TOOL_STATUS] == "success"
        assert "10" in result[ContextKeys.TOOL_RESULT]

    def test_execute_tool_none_selected(self):
        registry = _make_registry()
        handlers = AgentHandlers(registry)

        context = {
            ContextKeys.TOOL_NAME: "none",
            ContextKeys.OBSERVATIONS: [],
        }
        result = handlers.execute_tool(context)
        assert result[ContextKeys.TOOL_STATUS] == "skipped"

    def test_execute_tool_no_tool_name(self):
        registry = _make_registry()
        handlers = AgentHandlers(registry)

        context = {ContextKeys.TOOL_NAME: None, ContextKeys.OBSERVATIONS: []}
        result = handlers.execute_tool(context)
        assert result[ContextKeys.TOOL_STATUS] == "skipped"

    def test_execute_tool_nonexistent(self):
        registry = _make_registry()
        handlers = AgentHandlers(registry)

        context = {
            ContextKeys.TOOL_NAME: "nonexistent",
            ContextKeys.TOOL_INPUT: {},
            ContextKeys.OBSERVATIONS: [],
        }
        result = handlers.execute_tool(context)
        assert result[ContextKeys.TOOL_STATUS] == "failed"

    def test_execute_tool_string_input_normalized(self):
        """String tool_input should be normalized to dict."""
        registry = _make_registry()
        handlers = AgentHandlers(registry)

        context = {
            ContextKeys.TOOL_NAME: "echo",
            ContextKeys.TOOL_INPUT: "hello world",
            ContextKeys.OBSERVATIONS: [],
        }
        result = handlers.execute_tool(context)
        assert result[ContextKeys.TOOL_STATUS] == "success"

    def test_execute_tool_clears_selection(self):
        """After execution, tool selection keys should be cleared."""
        registry = _make_registry()
        handlers = AgentHandlers(registry)

        context = {
            ContextKeys.TOOL_NAME: "echo",
            ContextKeys.TOOL_INPUT: {"input": "test"},
            ContextKeys.OBSERVATIONS: [],
        }
        result = handlers.execute_tool(context)
        assert result[ContextKeys.TOOL_NAME] is None
        assert result[ContextKeys.TOOL_INPUT] is None

    def test_execute_tool_accumulates_observations(self):
        registry = _make_registry()
        handlers = AgentHandlers(registry)

        existing = ["[Step 1] Previous observation"]
        context = {
            ContextKeys.TOOL_NAME: "echo",
            ContextKeys.TOOL_INPUT: {"input": "test"},
            ContextKeys.OBSERVATIONS: existing,
        }
        result = handlers.execute_tool(context)
        assert len(result[ContextKeys.OBSERVATIONS]) == 2

    def test_execute_tool_prunes_observations(self):
        registry = _make_registry()
        handlers = AgentHandlers(registry)

        # Create more than MAX_OBSERVATIONS existing observations
        existing = [f"[Step {i}] Obs {i}" for i in range(Defaults.MAX_OBSERVATIONS + 5)]
        context = {
            ContextKeys.TOOL_NAME: "echo",
            ContextKeys.TOOL_INPUT: {"input": "test"},
            ContextKeys.OBSERVATIONS: existing,
        }
        result = handlers.execute_tool(context)
        assert len(result[ContextKeys.OBSERVATIONS]) <= Defaults.MAX_OBSERVATIONS

    def test_execute_tool_builds_trace(self):
        registry = _make_registry()
        handlers = AgentHandlers(registry)

        context = {
            ContextKeys.TOOL_NAME: "echo",
            ContextKeys.TOOL_INPUT: {"input": "test"},
            ContextKeys.REASONING: "Testing",
            ContextKeys.OBSERVATIONS: [],
            ContextKeys.AGENT_TRACE: [],
        }
        result = handlers.execute_tool(context)
        trace = result[ContextKeys.AGENT_TRACE]
        assert len(trace) == 1
        assert trace[0]["action"] == "echo({'input': 'test'})"

    def test_check_iteration_limit_below(self):
        registry = _make_registry()
        handlers = AgentHandlers(registry)

        context = {"_max_iterations": 10}
        result = handlers.check_iteration_limit(context)
        assert ContextKeys.ITERATION_COUNT in result
        assert ContextKeys.MAX_ITERATIONS_REACHED not in result

    def test_check_iteration_limit_reached(self):
        registry = _make_registry()
        handlers = AgentHandlers(registry)

        # Simulate reaching the limit
        context = {"_max_iterations": 2}
        handlers.check_iteration_limit(context)  # iteration 1
        result = handlers.check_iteration_limit(context)  # iteration 2

        assert result[ContextKeys.MAX_ITERATIONS_REACHED] is True
        assert result[ContextKeys.SHOULD_TERMINATE] is True

    def test_check_approval_requires_approval(self):
        registry = ToolRegistry()
        registry.register_function(
            _echo,
            name="dangerous",
            description="Dangerous tool",
            requires_approval=True,
        )
        handlers = AgentHandlers(registry)

        context = {ContextKeys.TOOL_NAME: "dangerous"}
        result = handlers.check_approval(context)
        assert result[ContextKeys.APPROVAL_REQUIRED] is True

    def test_check_approval_not_required(self):
        registry = _make_registry()
        handlers = AgentHandlers(registry)

        context = {ContextKeys.TOOL_NAME: "echo"}
        result = handlers.check_approval(context)
        assert result[ContextKeys.APPROVAL_REQUIRED] is False

    def test_check_approval_no_tool(self):
        registry = _make_registry()
        handlers = AgentHandlers(registry)

        context = {ContextKeys.TOOL_NAME: None}
        result = handlers.check_approval(context)
        assert result == {}
