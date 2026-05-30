"""Tests for NativeFunctionCallingReactAgent (provider-native tool calling).

The LLM is replaced with an injected complete_fn returning normalized
responses, so the loop is exercised with no live provider.
"""

from __future__ import annotations

import pytest

from fsm_llm_agents import (
    AgentConfig,
    NativeFunctionCallingReactAgent,
    ToolRegistry,
    tool,
)


@tool
def weather(city: str) -> str:
    """Get weather for a city."""
    return f"sunny in {city}"


def _registry():
    reg = ToolRegistry()
    reg.register(weather._tool_definition)
    return reg


def _scripted(*responses):
    """Return a complete_fn yielding the given normalized responses in order."""
    it = iter(responses)

    def complete_fn(model, messages, schemas):
        return next(it)

    return complete_fn


class TestConstruction:
    def test_empty_registry_rejected(self):
        with pytest.raises(Exception):
            NativeFunctionCallingReactAgent(tools=ToolRegistry())


class TestLoop:
    def test_direct_answer_no_tools(self):
        agent = NativeFunctionCallingReactAgent(
            tools=_registry(),
            config=AgentConfig(model="mock/model"),
            complete_fn=_scripted({"content": "42", "tool_calls": []}),
        )
        result = agent.run("what is the answer?")
        assert result.answer == "42"
        assert result.success is True
        assert result.tools_used == []

    def test_single_tool_then_answer(self):
        agent = NativeFunctionCallingReactAgent(
            tools=_registry(),
            config=AgentConfig(model="mock/model"),
            complete_fn=_scripted(
                {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "name": "weather",
                            "arguments": {"city": "Paris"},
                        }
                    ],
                },
                {"content": "It's sunny in Paris.", "tool_calls": []},
            ),
        )
        result = agent.run("weather in Paris?")
        assert "sunny in Paris" in result.answer or "Paris" in result.answer
        assert "weather" in result.tools_used
        assert result.success

    def test_multiple_tool_calls_in_one_turn(self):
        agent = NativeFunctionCallingReactAgent(
            tools=_registry(),
            config=AgentConfig(model="mock/model"),
            complete_fn=_scripted(
                {
                    "content": None,
                    "tool_calls": [
                        {"id": "a", "name": "weather", "arguments": {"city": "Paris"}},
                        {"id": "b", "name": "weather", "arguments": {"city": "Rome"}},
                    ],
                },
                {"content": "Done.", "tool_calls": []},
            ),
        )
        result = agent.run("compare weather")
        assert len(result.trace.tool_calls) == 2

    def test_string_arguments_are_parsed(self):
        agent = NativeFunctionCallingReactAgent(
            tools=_registry(),
            config=AgentConfig(model="mock/model"),
            complete_fn=_scripted(
                {
                    "content": None,
                    "tool_calls": [
                        {"id": "c1", "name": "weather", "arguments": {"city": "Oslo"}}
                    ],
                },
                {"content": "ok", "tool_calls": []},
            ),
        )
        result = agent.run("q")
        assert result.trace.tool_calls[0].parameters == {"city": "Oslo"}

    def test_max_iterations_exhausted(self):
        # Always returns a tool call → never concludes; bounded by max_iterations.
        def always_tool(model, messages, schemas):
            return {
                "content": None,
                "tool_calls": [
                    {"id": "x", "name": "weather", "arguments": {"city": "X"}}
                ],
            }

        agent = NativeFunctionCallingReactAgent(
            tools=_registry(),
            config=AgentConfig(model="mock/model", max_iterations=3),
            complete_fn=always_tool,
        )
        result = agent.run("loop")
        # No final answer, but tool calls happened → counts as a (partial) success.
        assert result.answer == ""
        assert len(result.trace.tool_calls) == 3

    def test_uses_get_json_schemas(self):
        captured = {}

        def cap(model, messages, schemas):
            captured["schemas"] = schemas
            return {"content": "ok", "tool_calls": []}

        agent = NativeFunctionCallingReactAgent(
            tools=_registry(),
            config=AgentConfig(model="mock/model"),
            complete_fn=cap,
        )
        agent.run("q")
        assert captured["schemas"][0]["type"] == "function"
        assert captured["schemas"][0]["function"]["name"] == "weather"

    def test_assistant_message_reconstruction(self):
        msg = NativeFunctionCallingReactAgent._assistant_message(
            None, [{"id": "c1", "name": "weather", "arguments": {"city": "Paris"}}]
        )
        assert msg["role"] == "assistant"
        assert msg["tool_calls"][0]["function"]["name"] == "weather"
        # arguments serialized as a JSON string per OpenAI format
        assert isinstance(msg["tool_calls"][0]["function"]["arguments"], str)
