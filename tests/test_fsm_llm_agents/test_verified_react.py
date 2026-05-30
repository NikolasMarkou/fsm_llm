"""Tests for VerifiedReactAgent: verify-and-retry + periodic reflection."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from fsm_llm_agents import AgentConfig, ToolRegistry, VerifiedReactAgent, tool
from fsm_llm_agents.definitions import AgentResult, AgentTrace


@tool
def noop(query: str) -> str:
    """No-op."""
    return query


def _registry():
    reg = ToolRegistry()
    reg.register(noop._tool_definition)
    return reg


def _result(answer, success=True):
    return AgentResult(
        answer=answer,
        success=success,
        trace=AgentTrace(tool_calls=[], total_iterations=1),
    )


def _agent(config, max_verify_retries=1):
    return VerifiedReactAgent(
        tools=_registry(), config=config, max_verify_retries=max_verify_retries
    )


class TestConstruction:
    def test_invalid_retries(self):
        with pytest.raises(Exception):
            VerifiedReactAgent(tools=_registry(), max_verify_retries=-1)


class TestVerification:
    def test_no_verification_fn_delegates_once(self, monkeypatch):
        calls = {"n": 0}

        def fake_run(self, task, initial_context=None):
            calls["n"] += 1
            return _result("ans")

        monkeypatch.setattr("fsm_llm_agents.react.ReactAgent.run", fake_run)
        agent = _agent(AgentConfig(model="mock/model"))
        r = agent.run("q")
        assert r.answer == "ans"
        assert calls["n"] == 1

    def test_passes_first_try(self, monkeypatch):
        calls = {"n": 0}

        def fake_run(self, task, initial_context=None):
            calls["n"] += 1
            return _result("good")

        monkeypatch.setattr("fsm_llm_agents.react.ReactAgent.run", fake_run)
        cfg = AgentConfig(
            model="mock/model",
            verification_fn=lambda a, c: {"ok": a == "good", "feedback": "x"},
        )
        agent = _agent(cfg)
        agent.run("q")
        assert calls["n"] == 1

    def test_retries_with_feedback_then_passes(self, monkeypatch):
        tasks = []
        answers = iter(["bad", "good"])

        def fake_run(self, task, initial_context=None):
            tasks.append(task)
            return _result(next(answers))

        monkeypatch.setattr("fsm_llm_agents.react.ReactAgent.run", fake_run)
        cfg = AgentConfig(
            model="mock/model",
            verification_fn=lambda a, c: {
                "ok": a == "good",
                "feedback": "needs to say good",
            },
        )
        agent = _agent(cfg, max_verify_retries=2)
        r = agent.run("q")
        assert r.answer == "good"
        assert len(tasks) == 2
        # Feedback folded into the second attempt's task.
        assert "needs to say good" in tasks[1]

    def test_exhausts_retries_returns_last(self, monkeypatch):
        calls = {"n": 0}

        def fake_run(self, task, initial_context=None):
            calls["n"] += 1
            return _result("never-good")

        monkeypatch.setattr("fsm_llm_agents.react.ReactAgent.run", fake_run)
        cfg = AgentConfig(
            model="mock/model", verification_fn=lambda a, c: {"ok": False}
        )
        agent = _agent(cfg, max_verify_retries=2)
        r = agent.run("q")
        assert r.answer == "never-good"
        assert calls["n"] == 3  # 1 + 2 retries

    def test_bool_verdict_supported(self, monkeypatch):
        def fake_run(self, task, initial_context=None):
            return _result("ok")

        monkeypatch.setattr("fsm_llm_agents.react.ReactAgent.run", fake_run)
        cfg = AgentConfig(model="mock/model", verification_fn=lambda a, c: True)
        agent = _agent(cfg)
        assert agent.run("q").answer == "ok"

    def test_verification_exception_treated_as_pass(self, monkeypatch):
        calls = {"n": 0}

        def fake_run(self, task, initial_context=None):
            calls["n"] += 1
            return _result("x")

        def boom(a, c):
            raise RuntimeError("verifier crashed")

        monkeypatch.setattr("fsm_llm_agents.react.ReactAgent.run", fake_run)
        agent = _agent(AgentConfig(model="mock/model", verification_fn=boom))
        agent.run("q")
        assert calls["n"] == 1  # not retried


class TestReflection:
    def _api(self):
        api = MagicMock()
        api.get_data.return_value = {"observations": ["[Step 1] ..."]}
        return api

    def test_injects_reflection_on_cadence(self):
        agent = _agent(AgentConfig(model="mock/model", reflect_every_n=2))
        api = self._api()
        agent._on_loop_iteration(api, "cid", 2)
        assert api.update_context.called
        # update_context(conv_id, {...}) — second positional arg
        payload = api.update_context.call_args[0][1]
        assert any("Reflection" in o for o in payload["observations"])

    def test_no_injection_off_cadence(self):
        agent = _agent(AgentConfig(model="mock/model", reflect_every_n=3))
        api = self._api()
        agent._on_loop_iteration(api, "cid", 2)  # 2 % 3 != 0
        assert not api.update_context.called

    def test_no_injection_when_disabled(self):
        agent = _agent(AgentConfig(model="mock/model"))  # reflect_every_n None
        api = self._api()
        agent._on_loop_iteration(api, "cid", 4)
        assert not api.update_context.called
