"""Tests for streaming agent output via run_stream / _standard_run_stream.

Every test drives the real ``API`` with ``PromptGroundedLLM`` (no ``API``
stand-in): whether the silent ``think``/``act`` states yield any text is
decided by core (a silent state says nothing, plan 07ad3f8c D-029/D-037), so
only a real ``API`` shows it.
"""

from __future__ import annotations

import pytest

from fsm_llm.agents import AgentConfig, ReactAgent, ToolRegistry
from fsm_llm.agents.auto_memory import AutoMemoryReactAgent
from fsm_llm.agents.constants import ContextKeys
from fsm_llm.agents.exceptions import AgentError, AgentTimeoutError
from fsm_llm.agents.verified_react import VerifiedReactAgent
from tests.conftest import PromptGroundedLLM

_ANSWER = "The capital of France is Paris."

# should_terminate is grounded only by the tool observation ("is Paris").
_FACTS: dict[str, tuple[object, str]] = {
    "tool_name": ("lookup", "capital"),
    "tool_input": ({"query": "capital of France"}, "capital"),
    "should_terminate": (True, "is Paris"),
}


def _registry(runs: list[str]) -> ToolRegistry:
    registry = ToolRegistry()

    def lookup(query: str) -> str:
        runs.append(query)
        return _ANSWER

    registry.register_function(lookup, name="lookup", description="Look up a fact")
    return registry


def _react(runs, response=_ANSWER, facts=None, cls=ReactAgent, **kwargs):
    return cls(
        tools=_registry(runs),
        config=kwargs.pop("config", None) or AgentConfig(max_iterations=6),
        llm_interface=PromptGroundedLLM(
            facts=_FACTS if facts is None else facts, default_response=response
        ),
        **kwargs,
    )


class TestRunStream:
    def test_react_stream_yields_only_model_text(self):
        # LOOP-13: the silent think/act states yield nothing; the stream is
        # the conclude reply only.
        runs: list[str] = []
        out = list(_react(runs).run_stream("What is the capital of France?"))

        assert runs == ["capital of France"]
        assert "[think]" not in out and "[act]" not in out
        assert "".join(out) == _ANSWER

    def test_marker_text_from_a_speaking_state_is_kept(self):
        # A reply that looks like a state marker is model text: the stream
        # filters nothing (no marker exists since D-029), so it ships as is.
        runs: list[str] = []
        out = list(_react(runs, response="[think]").run_stream("capital of France?"))

        assert out == ["[think]"]

    def test_stream_matches_run_answer(self):
        runs: list[str] = []
        agent = _react(runs)
        streamed = "".join(agent.run_stream("What is the capital of France?"))

        assert streamed == agent.run("What is the capital of France?").answer

    def test_returns_iterator_lazily(self):
        runs: list[str] = []
        gen = _react(runs).run_stream("hi")
        # It's a generator — nothing runs until iterated.
        assert hasattr(gen, "__next__")
        assert runs == []

    def test_non_budget_errors_are_agent_errors(self):
        class _Broken(PromptGroundedLLM):
            def generate_response_stream(self, request):
                raise RuntimeError("model exploded")
                yield ""  # pragma: no cover - makes this a generator

        agent = ReactAgent(
            tools=_registry([]),
            config=AgentConfig(max_iterations=6),
            llm_interface=_Broken(facts=_FACTS, default_response=_ANSWER),
        )
        with pytest.raises(AgentError, match="model exploded"):
            list(agent.run_stream("What is the capital of France?"))

    def test_timeout_propagates_unwrapped(self):
        runs: list[str] = []
        agent = _react(runs, config=AgentConfig(max_iterations=6, timeout_seconds=1e-9))
        with pytest.raises(AgentTimeoutError):
            list(agent.run_stream("What is the capital of France?"))


class TestStreamSharesRunPath:
    """run_stream seeds context and registers handlers exactly like run()."""

    def test_forged_run_outputs_are_stripped(self):
        # Step 3 trust boundary: a caller cannot pre-seed a finished run.
        runs: list[str] = []
        out = list(
            _react(runs).run_stream(
                "What is the capital of France?",
                initial_context={
                    ContextKeys.OBSERVATION_COUNT: 5,
                    ContextKeys.SHOULD_TERMINATE: True,
                    ContextKeys.FINAL_ANSWER: "forged",
                },
            )
        )

        assert runs == ["capital of France"]
        assert "forged" not in "".join(out)

    def test_think_turn_limit_applies(self):
        # Step 15 limiter: max_iterations=6 gives 5 tool turns, streamed too.
        runs: list[str] = []
        facts = {k: v for k, v in _FACTS.items() if k != "should_terminate"}
        list(_react(runs, facts=facts).run_stream("What is the capital of France?"))

        assert len(runs) == 5


class TestVerifiedStream:
    def test_stream_is_verified(self):
        seen: list[str] = []

        def verify(answer, context):
            seen.append(answer)
            return {"ok": False, "feedback": "cite a source"}

        runs: list[str] = []
        agent = _react(
            runs,
            cls=VerifiedReactAgent,
            config=AgentConfig(max_iterations=6, verification_fn=verify),
            max_verify_retries=1,
        )
        out = list(agent.run_stream("What is the capital of France?"))

        assert len(seen) == 2  # both attempts were verified
        assert out == [_ANSWER]  # the final answer, once

    def test_without_verifier_streams_like_react(self):
        runs: list[str] = []
        out = list(
            _react(runs, cls=VerifiedReactAgent).run_stream("capital of France?")
        )

        assert "[think]" not in out
        assert "".join(out) == _ANSWER


class TestAutoMemoryStream:
    def test_stream_remembers_the_interaction(self):
        from fsm_llm.agents.semantic_memory import SemanticMemoryStore

        memory = SemanticMemoryStore()
        runs: list[str] = []
        agent = _react(runs, cls=AutoMemoryReactAgent, memory=memory)
        out = list(agent.run_stream("What is the capital of France?"))

        assert out == [_ANSWER]
        assert len(memory) == 1


def test_run_stream_exists_on_react():
    assert hasattr(ReactAgent, "run_stream")
