"""Tests for ParallelReactAgent: multi-tool-per-step concurrent dispatch."""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from fsm_llm.definitions import (
    FieldExtractionRequest,
    FieldExtractionResponse,
    ResponseGenerationRequest,
    ResponseGenerationResponse,
)
from fsm_llm.expressions import evaluate_logic
from fsm_llm.llm import LLMInterface
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


# ---------------------------------------------------------------------------
# D-008: conclude needs tool evidence (plan-2026-09-24T091842-c1d5bfbc)
# ---------------------------------------------------------------------------


class _BatchLLM(LLMInterface):
    """Mock LLM: ``batches[i]`` is the i-th think's (tool_calls, should_terminate).

    A think turn starts at its ``tool_calls`` request (extracted before
    ``should_terminate``); the last entry repeats once the list is exhausted.
    """

    def __init__(self, batches: list[tuple[list[dict[str, Any]], bool]]) -> None:
        self.model = "mock-model"
        self.batches = batches
        self.thinks = 0

    def extract_field(self, request: FieldExtractionRequest) -> FieldExtractionResponse:
        name = request.field_name
        if name == TOOL_CALLS_KEY:
            self.thinks += 1
        calls, terminate = self.batches[
            min(max(self.thinks - 1, 0), len(self.batches) - 1)
        ]
        value: Any = {
            TOOL_CALLS_KEY: calls,
            ContextKeys.SHOULD_TERMINATE: terminate,
            ContextKeys.FINAL_ANSWER: "sunny",
        }.get(name)
        return FieldExtractionResponse(
            field_name=name,
            value=value,
            confidence=0.9 if value is not None else 0.0,
            reasoning="mock",
            is_valid=value is not None,
        )

    def generate_response(
        self, request: ResponseGenerationRequest
    ) -> ResponseGenerationResponse:
        return ResponseGenerationResponse(
            message="final answer", message_type="response", reasoning="mock"
        )


def _run_recording_states(agent: Any) -> tuple[Any, list[str]]:
    """Run the agent; return (result, state before each loop turn)."""
    states: list[str] = []
    real_hook = agent._on_loop_iteration

    def _hook(api: Any, conv_id: str, iteration: int) -> None:
        states.append(api.get_current_state(conv_id))
        real_hook(api, conv_id, iteration)

    agent._on_loop_iteration = _hook
    return agent.run("Weather in Paris?"), states


def _conclude_passes(state: str, ctx: dict[str, Any]) -> bool:
    fsm = build_parallel_react_fsm(_registry())
    edge = next(
        t
        for t in fsm["states"][state]["transitions"]
        if t["target_state"] == "conclude"
    )
    return all(bool(evaluate_logic(c["logic"], ctx)) for c in edge["conditions"])


_PARIS = [{"tool_name": "weather", "tool_input": {"city": "Paris"}}]


class TestParallelConcludeNeedsEvidence:
    """Turn-1 ``should_terminate=True`` with no tool must not conclude (D-008)."""

    def _agent(self, batches) -> ParallelReactAgent:
        return ParallelReactAgent(
            tools=_registry(),
            config=AgentConfig(model="mock/model", max_iterations=6),
            llm_interface=_BatchLLM(batches),
        )

    def test_turn_one_terminate_without_tool_goes_to_act(self):
        result, states = _run_recording_states(self._agent([([], True)]))

        assert len(states) >= 2, "ParallelReact concluded on turn 1 with no tool"
        assert states[1] == "act"
        assert result.answer
        assert len(states) <= 6 + 2

    def test_tool_then_terminate_still_concludes(self):
        result, states = _run_recording_states(
            self._agent([(_PARIS, False), ([], True)])
        )

        assert states[:3] == ["think", "act", "think"]
        assert len(states) == 3
        assert "weather" in result.tools_used

    @pytest.mark.parametrize("state", ["think", "act"])
    def test_conclude_edge_needs_observation_or_forced_stop(self, state):
        terminate = {ContextKeys.SHOULD_TERMINATE: True}

        assert not _conclude_passes(state, terminate)
        assert _conclude_passes(state, {**terminate, ContextKeys.OBSERVATION_COUNT: 1})
        assert _conclude_passes(
            state, {**terminate, ContextKeys.MAX_ITERATIONS_REACHED: True}
        )
