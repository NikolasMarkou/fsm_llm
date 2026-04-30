from __future__ import annotations

"""Tests for fsm_llm_agents.constants module."""

from fsm_llm.stdlib.agents.constants import (
    AgentStates,
    ContextKeys,
    Defaults,
    ErrorMessages,
    HandlerNames,
    LogMessages,
)


class TestAgentStates:
    """Tests for AgentStates constants."""

    def test_states_are_strings(self):
        assert isinstance(AgentStates.THINK, str)
        assert isinstance(AgentStates.ACT, str)
        assert isinstance(AgentStates.CONCLUDE, str)

    def test_states_are_unique(self):
        states = {
            AgentStates.THINK,
            AgentStates.ACT,
            AgentStates.CONCLUDE,
            AgentStates.AWAIT_APPROVAL,
        }
        assert len(states) == 4


class TestContextKeys:
    """Tests for ContextKeys constants."""

    def test_keys_are_strings(self):
        assert isinstance(ContextKeys.TASK, str)
        assert isinstance(ContextKeys.TOOL_NAME, str)
        assert isinstance(ContextKeys.FINAL_ANSWER, str)

    def test_no_duplicate_values(self):
        values = [
            v
            for k, v in vars(ContextKeys).items()
            if not k.startswith("_") and isinstance(v, str)
        ]
        assert len(values) == len(set(values)), "Duplicate context key values found"


class TestDefaults:
    """Tests for Defaults constants."""

    def test_max_iterations_positive(self):
        assert Defaults.MAX_ITERATIONS > 0

    def test_timeout_positive(self):
        assert Defaults.TIMEOUT_SECONDS > 0

    def test_temperature_valid(self):
        assert 0.0 <= Defaults.TEMPERATURE <= 2.0


class TestHandlerNames:
    """Tests for HandlerNames constants."""

    def test_names_are_unique(self):
        names = {
            HandlerNames.TOOL_EXECUTOR,
            HandlerNames.ITERATION_LIMITER,
            HandlerNames.HITL_GATE,
        }
        assert len(names) == 3


class TestErrorMessages:
    """Tests for ErrorMessages templates."""

    def test_format_strings_work(self):
        msg = ErrorMessages.BUDGET_EXHAUSTED.format(limit=10)
        assert "10" in msg

        msg = ErrorMessages.TOOL_NOT_FOUND.format(name="search")
        assert "search" in msg


class TestLogMessages:
    """Tests for LogMessages templates."""

    def test_format_strings_work(self):
        msg = LogMessages.AGENT_STARTED.format(tool_count=3, model="gpt-4")
        assert "3" in msg
        assert "gpt-4" in msg

        msg = LogMessages.TOOL_SELECTED.format(name="search", input="query")
        assert "search" in msg
