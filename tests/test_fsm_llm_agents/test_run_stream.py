"""Tests for streaming agent output via run_stream / _standard_run_stream."""

from __future__ import annotations

from unittest.mock import patch

from fsm_llm_agents import AgentConfig, ReactAgent, ToolRegistry, tool


@tool
def echo(query: str) -> str:
    """Echo."""
    return query


def _registry():
    reg = ToolRegistry()
    reg.register(echo._tool_definition)
    return reg


class _FakeAPI:
    """Minimal API stand-in driving one streaming turn."""

    def __init__(self, tokens, turns=1, initial=""):
        self._tokens = tokens
        self._turns_left = turns
        self._initial = initial

    def start_conversation(self, context):
        return ("cid", self._initial)

    def has_conversation_ended(self, cid):
        return self._turns_left <= 0

    def converse_stream(self, msg, cid):
        self._turns_left -= 1
        yield from self._tokens

    def end_conversation(self, cid):
        pass

    # Handler registration is a no-op chain for the fake.
    def register_handler(self, handler):
        pass

    def create_handler(self, name):
        return _Chain()


class _Chain:
    def __getattr__(self, _name):
        def _f(*a, **k):
            return self

        return _f


class TestRunStream:
    def test_streams_tokens(self):
        agent = ReactAgent(tools=_registry(), config=AgentConfig(model="mock/model"))
        fake = _FakeAPI(["Hello", ", ", "world"], turns=1)
        with patch.object(agent, "_create_api", return_value=fake):
            out = list(agent.run_stream("hi"))
        assert "".join(out) == "Hello, world"

    def test_yields_initial_response(self):
        agent = ReactAgent(tools=_registry(), config=AgentConfig(model="mock/model"))
        fake = _FakeAPI(["x"], turns=1, initial="GREET")
        with patch.object(agent, "_create_api", return_value=fake):
            out = list(agent.run_stream("hi"))
        assert out[0] == "GREET"
        assert "".join(out) == "GREETx"

    def test_multiple_turns(self):
        agent = ReactAgent(tools=_registry(), config=AgentConfig(model="mock/model"))
        fake = _FakeAPI(["tok"], turns=3)
        with patch.object(agent, "_create_api", return_value=fake):
            out = list(agent.run_stream("hi"))
        assert out == ["tok", "tok", "tok"]

    def test_returns_iterator_lazily(self):
        agent = ReactAgent(tools=_registry(), config=AgentConfig(model="mock/model"))
        gen = agent.run_stream("hi")
        # It's a generator — nothing runs until iterated.
        assert hasattr(gen, "__next__")

    def test_run_stream_exists_on_react(self):
        assert hasattr(ReactAgent, "run_stream")
