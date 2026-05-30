"""Tests for ParallelReactAgent: multi-tool-per-step concurrent dispatch."""

from __future__ import annotations

import threading
import time

import pytest

from fsm_llm_agents import (
    AgentConfig,
    ParallelReactAgent,
    ToolRegistry,
    build_parallel_react_fsm,
    tool,
)
from fsm_llm_agents.constants import ContextKeys
from fsm_llm_agents.parallel_react import TOOL_CALLS_KEY


@tool
def weather(city: str) -> str:
    """Get weather for a city."""
    return f"sunny in {city}"


@tool
def boom(city: str) -> str:
    """Always fails."""
    raise RuntimeError("kaboom")


def _registry():
    reg = ToolRegistry()
    reg.register(weather._tool_definition)
    reg.register(boom._tool_definition)
    return reg


def _agent(max_parallel=4):
    return ParallelReactAgent(
        tools=_registry(),
        config=AgentConfig(model="mock/model"),
        max_parallel=max_parallel,
    )


class TestConstruction:
    def test_empty_registry_rejected(self):
        with pytest.raises(Exception):
            ParallelReactAgent(tools=ToolRegistry())

    def test_invalid_max_parallel(self):
        with pytest.raises(Exception):
            ParallelReactAgent(tools=_registry(), max_parallel=0)

    def test_fsm_is_valid(self):
        # API.from_definition validates the FSM structure on construction.
        from unittest.mock import MagicMock

        from fsm_llm import API
        from fsm_llm.llm import LLMInterface

        fsm = build_parallel_react_fsm(_registry(), task_description="t")
        api = API.from_definition(fsm, llm_interface=MagicMock(spec=LLMInterface))
        assert api is not None


class TestNormalizeCalls:
    def test_parses_tool_name_and_input(self):
        agent = _agent()
        calls = agent._normalize_calls(
            [{"tool_name": "weather", "tool_input": {"city": "Paris"}}]
        )
        assert len(calls) == 1
        assert calls[0].tool_name == "weather"
        assert calls[0].parameters == {"city": "Paris"}

    def test_accepts_name_and_input_aliases(self):
        agent = _agent()
        calls = agent._normalize_calls([{"name": "weather", "input": {"city": "Rome"}}])
        assert calls[0].tool_name == "weather"

    def test_skips_non_dict_and_no_tool(self):
        agent = _agent()
        calls = agent._normalize_calls(
            ["junk", {"tool_name": "none"}, {"tool_name": "weather"}]
        )
        assert len(calls) == 1

    def test_non_list_returns_empty(self):
        assert _agent()._normalize_calls("nope") == []


class TestDispatch:
    def _ctx(self, tool_calls):
        return {
            TOOL_CALLS_KEY: tool_calls,
            ContextKeys.OBSERVATIONS: [],
            ContextKeys.AGENT_TRACE: [],
        }

    def test_runs_multiple_tools_and_orders_observations(self):
        agent = _agent()
        ctx = self._ctx(
            [
                {"tool_name": "weather", "tool_input": {"city": "Paris"}},
                {"tool_name": "weather", "tool_input": {"city": "Tokyo"}},
            ]
        )
        out = agent._dispatch_parallel(ctx)
        obs = out[ContextKeys.OBSERVATIONS]
        assert len(obs) == 2
        assert "Paris" in obs[0]
        assert "Tokyo" in obs[1]
        assert out[TOOL_CALLS_KEY] is None  # cleared

    def test_partial_failure_records_both(self):
        agent = _agent()
        ctx = self._ctx(
            [
                {"tool_name": "weather", "tool_input": {"city": "Paris"}},
                {"tool_name": "boom", "tool_input": {"city": "X"}},
            ]
        )
        out = agent._dispatch_parallel(ctx)
        obs = out[ContextKeys.OBSERVATIONS]
        assert len(obs) == 2
        assert any("TOOL FAILED" in o for o in obs)
        assert out[ContextKeys.TOOL_STATUS] == "success"  # at least one succeeded

    def test_empty_calls_skipped(self):
        agent = _agent()
        out = agent._dispatch_parallel(self._ctx([]))
        assert out[ContextKeys.TOOL_STATUS] == "skipped"

    def test_actually_concurrent(self):
        """Two slow tools overlap when max_parallel >= 2."""
        active = {"now": 0, "max": 0}
        lock = threading.Lock()

        def slow(city: str) -> str:
            with lock:
                active["now"] += 1
                active["max"] = max(active["max"], active["now"])
            time.sleep(0.05)
            with lock:
                active["now"] -= 1
            return city

        reg = ToolRegistry()
        reg.register_function(
            slow,
            name="slow",
            description="slow",
            parameter_schema={
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        )
        agent = ParallelReactAgent(
            tools=reg, config=AgentConfig(model="mock/model"), max_parallel=3
        )
        ctx = {
            TOOL_CALLS_KEY: [
                {"tool_name": "slow", "tool_input": {"city": c}}
                for c in ("a", "b", "c")
            ],
            ContextKeys.OBSERVATIONS: [],
            ContextKeys.AGENT_TRACE: [],
        }
        agent._dispatch_parallel(ctx)
        assert active["max"] >= 2
