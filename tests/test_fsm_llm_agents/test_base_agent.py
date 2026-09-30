from __future__ import annotations

"""Tests for BaseAgent abstract class."""

import time
from unittest.mock import patch

import pytest

from fsm_llm.agents.base import BaseAgent
from fsm_llm.agents.definitions import AgentConfig, AgentResult
from fsm_llm.agents.exceptions import AgentTimeoutError, BudgetExhaustedError


class ConcreteAgent(BaseAgent):
    """Minimal concrete agent for testing base class."""

    def __init__(self, config=None, **api_kwargs):
        super().__init__(config, **api_kwargs)

    def run(self, task, initial_context=None):
        return AgentResult(answer=f"Answer: {task}", success=True)

    def _register_handlers(self, api):
        pass


class TestBaseAgentCallable:
    """Tests for __call__ support."""

    def test_call_delegates_to_run(self):
        agent = ConcreteAgent()
        result = agent("What is 2+2?")
        assert result.answer == "Answer: What is 2+2?"
        assert result.success is True

    def test_call_with_kwargs(self):
        class KwargsAgent(BaseAgent):
            def run(self, task, initial_context=None):
                ctx = initial_context or {}
                return AgentResult(
                    answer=f"{task} [{ctx.get('mode', 'default')}]", success=True
                )

            def _register_handlers(self, api):
                pass

        agent = KwargsAgent()
        result = agent("hello", initial_context={"mode": "fast"})
        assert "[fast]" in result.answer

    def test_str_representation(self):
        agent = ConcreteAgent()
        assert "ConcreteAgent" in str(agent)
        assert "model=" in str(agent)


class TestBaseAgentBudgets:
    """Tests for budget enforcement."""

    def test_check_budgets_timeout(self):
        agent = ConcreteAgent(config=AgentConfig(timeout_seconds=0.001))
        time.sleep(0.01)
        with pytest.raises(AgentTimeoutError):
            agent._check_budgets(time.monotonic() - 1.0)

    def test_check_budgets_ok(self):
        agent = ConcreteAgent(config=AgentConfig(max_iterations=10))
        # Should not raise
        agent._check_budgets(time.monotonic())

    def test_run_budgets_are_the_step_ceiling_and_the_remaining_time(self):
        agent = ConcreteAgent(config=AgentConfig(max_iterations=5, timeout_seconds=60))
        max_steps, max_seconds = agent._run_budgets(time.monotonic() - 20.0)
        assert max_steps == 15  # 5 * FSM_BUDGET_MULTIPLIER
        assert 39.0 < max_seconds <= 40.0
        assert agent._run_budgets(time.monotonic(), 2)[0] == 6

    def test_run_budgets_raise_timeout_when_the_clock_is_spent(self):
        agent = ConcreteAgent(config=AgentConfig(timeout_seconds=1.0))
        with pytest.raises(AgentTimeoutError):
            agent._run_budgets(time.monotonic() - 2.0)

    def test_budget_error_maps_the_core_budgets(self):
        from fsm_llm import RunBudgetExceededError

        agent = ConcreteAgent(config=AgentConfig(max_iterations=5, timeout_seconds=7))
        steps = agent._budget_error(RunBudgetExceededError("steps", 15, 15))
        assert isinstance(steps, BudgetExhaustedError)
        assert steps.limit == 15  # 5 * 3
        seconds = agent._budget_error(RunBudgetExceededError("seconds", 3.2, 4))
        assert isinstance(seconds, AgentTimeoutError)
        assert seconds.timeout_seconds == 7


class TestBaseAgentAnswerExtraction:
    """Tests for answer extraction."""

    def test_extract_from_final_answer(self):
        agent = ConcreteAgent()
        ctx = {"final_answer": "The answer is 42."}
        assert agent._extract_answer(ctx, []) == "The answer is 42."

    def test_extract_from_extra_keys(self):
        agent = ConcreteAgent()
        ctx = {"judge_verdict": "Consensus: yes, it is correct."}
        answer = agent._extract_answer(ctx, [], extra_keys=["judge_verdict"])
        assert "Consensus" in answer

    def test_extract_from_responses(self):
        agent = ConcreteAgent()
        ctx = {}
        answer = agent._extract_answer(ctx, ["short", "This is a longer response."])
        assert answer == "This is a longer response."

    def test_extract_default(self):
        agent = ConcreteAgent()
        assert "could not" in agent._extract_answer({}, [])

    def test_final_answer_priority(self):
        agent = ConcreteAgent()
        ctx = {
            "final_answer": "Primary answer text.",
            "judge_verdict": "Verdict text here.",
        }
        answer = agent._extract_answer(ctx, ["response"], extra_keys=["judge_verdict"])
        assert answer == "Primary answer text."


class TestBaseAgentTraceBuilding:
    """Tests for trace building."""

    def test_build_trace_empty(self):
        agent = ConcreteAgent()
        trace = agent._build_trace({}, 5)
        assert trace.total_iterations == 5
        assert trace.tool_calls == []

    def test_build_trace_with_steps(self):
        agent = ConcreteAgent()
        ctx = {
            "agent_trace": [
                {"action": "search(query)", "thought": "Need to search"},
                {"action": "none", "thought": "Done"},
                {"action": "calculate(expr)", "thought": "Math time"},
            ]
        }
        trace = agent._build_trace(ctx, 3)
        assert len(trace.tool_calls) == 2
        assert trace.tool_calls[0].tool_name == "search"
        assert trace.tool_calls[1].tool_name == "calculate"


class TestBaseAgentContextFiltering:
    """Tests for context filtering."""

    def test_filter_removes_private_keys(self):
        ctx = {"answer": "yes", "_internal": "secret", "_max": 10, "data": 42}
        filtered = BaseAgent._filter_context(ctx)
        assert "answer" in filtered
        assert "data" in filtered
        assert "_internal" not in filtered
        assert "_max" not in filtered

    def test_filter_removes_every_internal_prefix_case_insensitively(self):
        """`_filter_context` must strip the FULL internal-prefix class.

        Pins step 3 of plan-2026-07-20T040150-876e7164 (F-13). The previous
        `k.startswith("_")` check was case-sensitive and saw only the literal
        `_` prefix, so `system_password`, `internal_token` and `__dunder` all
        passed straight through into caller-visible output.
        """
        ctx = {
            # Must survive.
            "answer": "yes",
            "data": 42,
            # `_`-prefixed (already caught by the old check).
            "_internal": "secret",
            # The prefixes the old check MISSED entirely.
            "system_password": "hunter2",
            "internal_token": "tok",
            "__dunder": "no",
            # Case variants of the same prefixes.
            "System_Password": "hunter2",
            "SYSTEM_secret": "shh",
            "Internal_token": "tok",
            "INTERNAL_state": "x",
        }
        filtered = BaseAgent._filter_context(ctx)

        assert filtered == {"answer": "yes", "data": 42}

    def test_final_context_does_not_leak_system_prefixed_keys(self):
        """The filter's output is what reaches `AgentResult.final_context`.

        End-to-end assertion of the same defect at the surface a caller
        actually sees, not just at the static helper.
        """
        result = AgentResult(
            answer="done",
            success=True,
            final_context=BaseAgent._filter_context(
                {
                    "order_id": "A-1",
                    "system_password": "hunter2",
                    "Internal_token": "tok",
                }
            ),
        )

        assert result.final_context == {"order_id": "A-1"}
        assert "system_password" not in result.final_context
        assert "Internal_token" not in result.final_context


class TestBaseAgentCreateApi:
    """Tests for API factory helper."""

    @patch("fsm_llm.agents.base.API.from_definition")
    def test_create_api(self, mock_from_def):
        agent = ConcreteAgent(config=AgentConfig(model="test-model"))
        fsm_def = {"name": "test", "initial_state": "start", "states": {}}
        agent._create_api(fsm_def)

        mock_from_def.assert_called_once_with(
            fsm_def,
            model="test-model",
            temperature=agent.config.temperature,
            max_tokens=agent.config.max_tokens,
        )


class TestBaseAgentConfig:
    """Tests for configuration handling."""

    def test_default_config(self):
        agent = ConcreteAgent()
        assert agent.config.max_iterations > 0
        assert agent.config.timeout_seconds > 0

    def test_custom_config(self):
        config = AgentConfig(model="custom-model", max_iterations=20)
        agent = ConcreteAgent(config=config)
        assert agent.config.model == "custom-model"
        assert agent.config.max_iterations == 20

    def test_api_kwargs_stored(self):
        agent = ConcreteAgent(api_key="test-key", extra="value")
        assert agent._api_kwargs == {"api_key": "test-key", "extra": "value"}


class TestCreateAgent:
    """Tests for create_agent() convenience function."""

    def test_create_react_agent(self):
        from fsm_llm.agents import create_agent
        from fsm_llm.agents.tools import tool

        @tool
        def search(query: str) -> str:
            """Search the web."""
            return "results"

        agent = create_agent(tools=[search])
        from fsm_llm.agents import ReactAgent

        assert isinstance(agent, ReactAgent)

    def test_create_debate_agent(self):
        from fsm_llm.agents import DebateAgent, create_agent

        agent = create_agent(pattern="debate")
        assert isinstance(agent, DebateAgent)

    def test_create_with_registry(self):
        from fsm_llm.agents import ToolRegistry, create_agent

        registry = ToolRegistry()
        registry.register_function(lambda p: "ok", name="test", description="Test tool")
        agent = create_agent(tools=registry)
        assert len(agent.tools) == 1

    def test_unknown_pattern_raises(self):
        from fsm_llm.agents import create_agent

        with pytest.raises(ValueError, match="Unknown pattern"):
            create_agent(pattern="nonexistent")

    def test_create_with_config(self):
        from fsm_llm.agents import AgentConfig, create_agent

        config = AgentConfig(max_iterations=5)
        agent = create_agent(pattern="debate", config=config)
        assert agent.config.max_iterations == 5

    def test_create_parallel_react_agent(self):
        from fsm_llm.agents import ParallelReactAgent, create_agent
        from fsm_llm.agents.tools import tool

        @tool
        def search(query: str) -> str:
            """Search the web."""
            return "results"

        agent = create_agent(pattern="parallel_react", tools=[search])
        assert isinstance(agent, ParallelReactAgent)

    def test_create_native_fc_agent(self):
        from fsm_llm.agents import NativeFunctionCallingReactAgent, create_agent
        from fsm_llm.agents.tools import tool

        @tool
        def search(query: str) -> str:
            """Search the web."""
            return "results"

        agent = create_agent(pattern="native_fc", tools=[search])
        assert isinstance(agent, NativeFunctionCallingReactAgent)

    def test_create_verified_react_agent(self):
        from fsm_llm.agents import VerifiedReactAgent, create_agent
        from fsm_llm.agents.tools import tool

        @tool
        def search(query: str) -> str:
            """Search the web."""
            return "results"

        agent = create_agent(pattern="verified_react", tools=[search])
        assert isinstance(agent, VerifiedReactAgent)

    def test_create_auto_memory_agent(self):
        from fsm_llm.agents import AutoMemoryReactAgent, create_agent
        from fsm_llm.agents.tools import tool

        @tool
        def search(query: str) -> str:
            """Search the web."""
            return "results"

        agent = create_agent(pattern="auto_memory", tools=[search])
        assert isinstance(agent, AutoMemoryReactAgent)
