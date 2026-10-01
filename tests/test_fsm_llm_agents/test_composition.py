"""Tests for composition helpers: react_worker_factory + default_llm_judge."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from fsm_llm.agents import (
    AgentConfig,
    ToolRegistry,
    default_llm_judge,
    react_worker_factory,
    tool,
)
from fsm_llm.agents.composition import _default_complete
from fsm_llm.agents.definitions import AgentResult, AgentTrace, EvaluationResult
from fsm_llm.agents.exceptions import AgentError, EvaluationError
from fsm_llm.definitions import LLMResponseError


@tool
def noop(query: str) -> str:
    """No-op."""
    return query


def _registry():
    reg = ToolRegistry()
    reg.register(noop._tool_definition)
    return reg


class TestReactWorkerFactory:
    def test_returns_callable(self):
        worker = react_worker_factory(_registry(), AgentConfig(model="mock/model"))
        assert callable(worker)

    def test_worker_runs_react_per_subtask(self, monkeypatch):
        seen = []

        def fake_run(self, task, initial_context=None):
            seen.append(task)
            return AgentResult(
                answer=f"done:{task}",
                success=True,
                trace=AgentTrace(tool_calls=[], total_iterations=1),
            )

        monkeypatch.setattr("fsm_llm.agents.react.ReactAgent.run", fake_run)
        worker = react_worker_factory(_registry(), AgentConfig(model="mock/model"))
        r = worker("subtask-A")
        assert isinstance(r, AgentResult)
        assert r.answer == "done:subtask-A"
        assert seen == ["subtask-A"]

    def test_fresh_agent_each_call(self, monkeypatch):
        instances = []

        orig_init = __import__(
            "fsm_llm.agents.react", fromlist=["ReactAgent"]
        ).ReactAgent.__init__

        def tracking_init(self, *a, **k):
            instances.append(self)
            orig_init(self, *a, **k)

        def fake_run(self, task, initial_context=None):
            return AgentResult(
                answer="x",
                success=True,
                trace=AgentTrace(tool_calls=[], total_iterations=1),
            )

        monkeypatch.setattr("fsm_llm.agents.react.ReactAgent.__init__", tracking_init)
        monkeypatch.setattr("fsm_llm.agents.react.ReactAgent.run", fake_run)
        worker = react_worker_factory(_registry(), AgentConfig(model="mock/model"))
        worker("a")
        worker("b")
        assert len(instances) == 2  # a fresh agent per subtask


class TestDefaultLlmJudge:
    def test_passes_above_threshold(self):
        def fake_complete(model, prompt):
            return '{"score": 0.9, "feedback": "great"}'

        judge = default_llm_judge(threshold=0.7, complete_fn=fake_complete)
        result = judge("the answer", {"task": "do X"})
        assert isinstance(result, EvaluationResult)
        assert result.passed is True
        assert result.score == 0.9
        assert result.feedback == "great"

    def test_fails_below_threshold(self):
        judge = default_llm_judge(
            threshold=0.7,
            complete_fn=lambda m, p: '{"score": 0.3, "feedback": "weak"}',
        )
        result = judge("bad", {})
        assert result.passed is False
        assert result.score == 0.3

    def test_clamps_out_of_range_score(self):
        judge = default_llm_judge(complete_fn=lambda m, p: '{"score": 5}')
        assert judge("x", {}).score == 1.0

    def test_handles_unparseable_response(self):
        judge = default_llm_judge(complete_fn=lambda m, p: "not json at all")
        result = judge("x", {})
        assert result.passed is False
        assert result.score == 0.0

    def test_handles_complete_exception(self):
        def boom(m, p):
            raise RuntimeError("llm down")

        result = default_llm_judge(complete_fn=boom)("x", {})
        assert result.passed is False
        assert "judge error" in result.feedback

    def test_criteria_in_prompt(self):
        captured = {}

        def cap(model, prompt):
            captured["prompt"] = prompt
            return '{"score": 1.0}'

        default_llm_judge(criteria="cites sources", complete_fn=cap)(
            "ans", {"task": "T"}
        )
        assert "cites sources" in captured["prompt"]
        assert "T" in captured["prompt"]

    def test_compatible_with_evaluator_optimizer_signature(self):
        # evaluation_fn is called as fn(output_str, context_dict)
        judge = default_llm_judge(complete_fn=lambda m, p: '{"score": 0.8}')
        out = judge("output text", {"task": "t"})
        assert isinstance(out, EvaluationResult)


def _judge_reply(content):
    """A provider reply of the shape litellm returns (one choice, no tools)."""
    message = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7, total_tokens=18),
    )


class _ProviderStub:
    """Stands in for both provider bindings: records every request's kwargs."""

    def __init__(self, reply=None, error=None):
        self.requests: list[dict] = []
        self._reply = reply
        self._error = error

    def __call__(self, **kwargs):
        self.requests.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._reply


def _install(monkeypatch, stub):
    # Both bindings: the core send path and the provider module, so a judge
    # that bypassed core would still be caught (and never reach the network).
    monkeypatch.setattr("fsm_llm.llm.completion", stub)
    monkeypatch.setattr("litellm.completion", stub)


class TestDefaultCompleteOnCore:
    """The default judge sends through core's ``LiteLLMInterface.complete``.

    DECISION plan-2026-10-01T093600-944e2692/D-007.
    """

    def test_one_user_turn_at_temperature_zero(self, monkeypatch):
        stub = _ProviderStub(reply=_judge_reply('{"score": 0.9}'))
        _install(monkeypatch, stub)

        assert _default_complete("gpt-4o-mini", "grade this") == '{"score": 0.9}'

        assert len(stub.requests) == 1
        request = stub.requests[0]
        assert request["model"] == "gpt-4o-mini"
        assert request["messages"] == [{"role": "user", "content": "grade this"}]
        assert request["temperature"] == 0.0
        assert "tools" not in request
        assert "response_format" not in request
        assert "reasoning_effort" not in request

    def test_ollama_model_gets_core_ollama_preparation(self, monkeypatch):
        """RED on the parent: the judge called the provider directly with no
        Ollama preparation, so `ollama_chat` judges ran with thinking on."""
        stub = _ProviderStub(reply=_judge_reply('{"score": 0.2}'))
        _install(monkeypatch, stub)

        _default_complete("ollama_chat/qwen3.5:4b", "grade this")

        request = stub.requests[0]
        assert request["reasoning_effort"] == "none"
        assert request["temperature"] == 0
        assert request["messages"][-1]["role"] == "user"
        assert request["messages"][-1]["content"].startswith("/nothink")
        assert request["messages"][-1]["content"].endswith("grade this")

    def test_no_reply_text_is_empty_string(self, monkeypatch):
        _install(monkeypatch, _ProviderStub(reply=_judge_reply(None)))
        assert _default_complete("gpt-4o-mini", "grade this") == ""


class TestDefaultCompleteBoundary:
    """F-03 / SC-10 — `_default_complete` is the judge's LLM boundary. A
    provider failure must leave it as an ``AgentError`` subclass with the
    provider exception preserved in the cause chain, never as core's
    ``LLMResponseError``, a raw provider exception, or an empty answer.

    DECISION plan-2026-07-20T040150-876e7164/D-006 [STALE].
    """

    def test_provider_failure_raises_evaluation_error(self, monkeypatch):
        provider_error = RuntimeError("provider connection reset")
        _install(monkeypatch, _ProviderStub(error=provider_error))

        with pytest.raises(EvaluationError) as excinfo:
            _default_complete("mock/model", "grade this")

        # Wrapped to the package root, not to core's FSMError surface (I-5).
        assert isinstance(excinfo.value, AgentError)
        # Core's boundary error, chained from the provider's (SC-10).
        assert isinstance(excinfo.value.__cause__, LLMResponseError)
        assert excinfo.value.__cause__.__cause__ is provider_error
        assert "provider connection reset" in str(excinfo.value)

    def test_raw_provider_exception_does_not_escape(self, monkeypatch):
        _install(monkeypatch, _ProviderStub(error=RuntimeError("rate limited")))

        with pytest.raises(AgentError):
            _default_complete("mock/model", "grade this")

    def test_judge_still_degrades_gracefully_over_the_wrap(self, monkeypatch):
        """The wrap must not change `judge()`'s contract — it already catches
        broadly and reports not-passed."""
        _install(monkeypatch, _ProviderStub(error=RuntimeError("provider down")))

        result = default_llm_judge(model="mock/model")("some answer", {"task": "T"})

        assert result.passed is False
        assert result.score == 0.0
        assert "judge error" in result.feedback
        # The provider's own message reaches the feedback via the chained message.
        assert "provider down" in result.feedback
