"""Tests for NativeFunctionCallingReactAgent (provider-native tool calling).

The LLM is replaced with an injected complete_fn returning normalized
responses, so the loop is exercised with no live provider.
"""

from __future__ import annotations

import pytest

from fsm_llm.ollama import apply_ollama_params, prepare_ollama_messages
from fsm_llm_agents import (
    AgentConfig,
    NativeFunctionCallingReactAgent,
    ToolRegistry,
    tool,
)
from fsm_llm_agents.exceptions import AgentError


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


# ---------------------------------------------------------------------------
# Minimal litellm response stand-ins for the `_litellm_complete` path.
# ---------------------------------------------------------------------------


class _FakeFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, id, name, arguments):
        self.id = id
        self.function = _FakeFunction(name, arguments)


class _FakeMessage:
    def __init__(self, content=None, tool_calls=None, reasoning_content=None):
        self.content = content
        self.tool_calls = tool_calls
        self.reasoning_content = reasoning_content


class _FakeResponse:
    def __init__(self, message):
        self.choices = [type("_Choice", (), {"message": message})()]


def _stub_completion(monkeypatch, message, captured=None):
    """Patch ``litellm.completion`` to return *message*, recording kwargs."""

    def fake(**kwargs):
        if captured is not None:
            captured.update(kwargs)
        return _FakeResponse(message)

    monkeypatch.setattr("litellm.completion", fake)


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
        # No final answer; the tool calls still land in the trace. Whether that
        # counts as success is D-005's question, asserted in TestSuccessSignal.
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


class TestLitellmBoundaryWrap:
    """F-03 / SC-10 — `_litellm_complete` is the agent's raw provider boundary.
    A provider failure must surface as an ``AgentError`` (the package root)
    with the provider exception preserved as ``__cause__``.

    DECISION plan-2026-07-20T040150-876e7164/D-006.
    """

    @staticmethod
    def _agent():
        # complete_fn=None so `_complete` routes to the real litellm path.
        return NativeFunctionCallingReactAgent(
            tools=_registry(),
            config=AgentConfig(model="mock/model"),
        )

    def test_provider_failure_raises_agent_error_chained(self, monkeypatch):
        provider_error = RuntimeError("provider timeout")

        def explode(**kwargs):
            raise provider_error

        monkeypatch.setattr("litellm.completion", explode)

        with pytest.raises(AgentError) as excinfo:
            self._agent()._litellm_complete([{"role": "user", "content": "q"}], [])

        assert excinfo.value.__cause__ is provider_error
        assert not isinstance(excinfo.value, RuntimeError)

    def test_wrap_reaches_the_run_loop(self, monkeypatch):
        """The wrap is on the path `run()` actually takes when no complete_fn
        is injected — not only on a directly-called private helper."""

        def explode(**kwargs):
            raise RuntimeError("provider down")

        monkeypatch.setattr("litellm.completion", explode)

        with pytest.raises(AgentError):
            self._agent()._complete([{"role": "user", "content": "q"}], [])

    def test_parsing_errors_are_not_relabelled_as_provider_failures(self, monkeypatch):
        """Only the network call is inside the try. A malformed response object
        must raise as itself, not be reported as an LLM call failure."""

        def bad_shape(**kwargs):
            return object()  # no `.choices`

        monkeypatch.setattr("litellm.completion", bad_shape)

        with pytest.raises(AttributeError):
            self._agent()._litellm_complete([{"role": "user", "content": "q"}], [])


class TestOllamaHelperGating:
    """DECISION plan-2026-07-21-bf7ffe24/D-003 — the Ollama call-shape helpers
    run inside ``_litellm_complete``, gated by ``is_ollama_model``. Off Ollama
    the blast radius must be exactly zero: the helpers are never even called.
    """

    @staticmethod
    def _agent(model):
        return NativeFunctionCallingReactAgent(
            tools=_registry(), config=AgentConfig(model=model)
        )

    @staticmethod
    def _spy_helpers(monkeypatch):
        seen = {"params": [], "messages": []}

        def spy_params(call_params, model, *, structured=True):
            seen["params"].append((dict(call_params), model, structured))
            return apply_ollama_params(call_params, model, structured=structured)

        def spy_messages(messages, model, response_format=None):
            seen["messages"].append((messages, model, response_format))
            return prepare_ollama_messages(messages, model, response_format)

        monkeypatch.setattr("fsm_llm_agents.native_fc.apply_ollama_params", spy_params)
        monkeypatch.setattr(
            "fsm_llm_agents.native_fc.prepare_ollama_messages", spy_messages
        )
        return seen

    def test_non_ollama_model_never_calls_the_helpers(self, monkeypatch):
        """Criterion 10 / the no-op proof: for a non-Ollama model neither helper
        is invoked at all, so no Ollama-shaped param can leak to a provider."""
        seen = self._spy_helpers(monkeypatch)
        captured: dict = {}
        _stub_completion(monkeypatch, _FakeMessage(content="hi"), captured)

        out = self._agent("mock/model")._litellm_complete(
            [{"role": "user", "content": "q"}], []
        )

        assert seen["params"] == []
        assert seen["messages"] == []
        assert "reasoning_effort" not in captured
        assert out["content"] == "hi"

    def test_ollama_model_applies_both_helpers(self, monkeypatch):
        seen = self._spy_helpers(monkeypatch)
        captured: dict = {}
        _stub_completion(monkeypatch, _FakeMessage(content="hi"), captured)

        self._agent("ollama_chat/qwen3.5:4b")._litellm_complete(
            [{"role": "user", "content": "q"}], []
        )

        assert len(seen["params"]) == 1
        assert len(seen["messages"]) == 1
        # D-002: the tool turn carries no response_format — one native tool call
        # OR one constrained payload per turn, never stacked.
        assert seen["messages"][0][2] is None
        # structured=False mirrors llm.py:642 for free-text replies: the user's
        # temperature is preserved, only thinking is disabled.
        assert seen["params"][0][2] is False
        assert captured["reasoning_effort"] == "none"
        assert captured["messages"][-1]["content"].startswith("/nothink")

    def test_ollama_preparation_does_not_mutate_the_caller_messages(self, monkeypatch):
        _stub_completion(monkeypatch, _FakeMessage(content="hi"))
        messages = [{"role": "user", "content": "q"}]

        self._agent("ollama_chat/qwen3.5:4b")._litellm_complete(messages, [])

        assert messages == [{"role": "user", "content": "q"}]


class TestReasoningTraceRecovery:
    """DECISION plan-2026-07-21-bf7ffe24/D-003 — an empty ``content`` with no
    tool calls is a reasoning-only reply; recover it through the SHARED
    resolver. With tool calls present, ``content=None`` is the normal shape and
    no recovery may be attempted.
    """

    @staticmethod
    def _agent():
        return NativeFunctionCallingReactAgent(
            tools=_registry(), config=AgentConfig(model="ollama_chat/qwen3.5:4b")
        )

    def test_empty_content_without_tool_calls_recovers_from_trace(self, monkeypatch):
        _stub_completion(
            monkeypatch,
            _FakeMessage(content=None, reasoning_content="the real answer"),
        )

        out = self._agent()._litellm_complete([{"role": "user", "content": "q"}], [])

        assert out["content"] == "the real answer"

    def test_empty_content_with_tool_calls_attempts_no_recovery(self, monkeypatch):
        calls = []

        def spy(message):
            calls.append(message)
            return "SHOULD NOT BE USED"

        monkeypatch.setattr("fsm_llm_agents.native_fc._resolve_reasoning_trace", spy)
        _stub_completion(
            monkeypatch,
            _FakeMessage(
                content=None,
                tool_calls=[_FakeToolCall("c1", "weather", '{"city": "Paris"}')],
                reasoning_content="noise",
            ),
        )

        out = self._agent()._litellm_complete([{"role": "user", "content": "q"}], [])

        assert calls == []
        assert not out["content"]
        assert out["tool_calls"][0]["name"] == "weather"
        assert out["tool_calls"][0]["arguments"] == {"city": "Paris"}


class TestSuccessSignal:
    """DECISION plan-2026-07-21-bf7ffe24/D-005 — ``success`` must distinguish a
    working run from a doomed one. The old ``bool(answer) or bool(trace_calls)``
    reported True on three live runs that wrote nothing and answered nothing.
    """

    def test_tool_calls_without_a_final_answer_are_not_success(self):
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
                {"content": "", "tool_calls": []},
            ),
        )
        result = agent.run("q")
        assert result.tools_used == ["weather"]
        assert result.answer == ""
        assert result.success is False

    def test_exhausted_max_iterations_is_not_success(self):
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
        assert len(result.trace.tool_calls) == 3
        assert result.success is False

    def test_final_answer_after_tool_use_is_success(self):
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
                {"content": "It is sunny.", "tool_calls": []},
            ),
        )
        result = agent.run("q")
        assert result.answer == "It is sunny."
        assert result.success is True
