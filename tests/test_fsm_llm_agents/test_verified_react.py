"""Tests for VerifiedReactAgent: verify-and-retry + periodic reflection."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from fsm_llm.agents import AgentConfig, ToolRegistry, VerifiedReactAgent, tool
from fsm_llm.agents.constants import ContextKeys, StopReason
from fsm_llm.agents.definitions import AgentResult, AgentTrace


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

        monkeypatch.setattr("fsm_llm.agents.react.ReactAgent.run", fake_run)
        agent = _agent(AgentConfig(model="mock/model"))
        r = agent.run("q")
        assert r.answer == "ans"
        assert calls["n"] == 1

    def test_passes_first_try(self, monkeypatch):
        calls = {"n": 0}

        def fake_run(self, task, initial_context=None):
            calls["n"] += 1
            return _result("good")

        monkeypatch.setattr("fsm_llm.agents.react.ReactAgent.run", fake_run)
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

        monkeypatch.setattr("fsm_llm.agents.react.ReactAgent.run", fake_run)
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

        monkeypatch.setattr("fsm_llm.agents.react.ReactAgent.run", fake_run)
        cfg = AgentConfig(
            model="mock/model", verification_fn=lambda a, c: {"ok": False}
        )
        agent = _agent(cfg, max_verify_retries=2)
        r = agent.run("q")
        assert r.answer == "never-good"
        assert calls["n"] == 3  # 1 + 2 retries
        # REACT-04: the answer ships, but it is not a success.
        assert r.success is False
        assert r.stop_reason == StopReason.VERIFICATION_FAILED

    def test_passing_answer_keeps_its_outcome(self, monkeypatch):
        def fake_run(self, task, initial_context=None):
            return _result("good")

        monkeypatch.setattr("fsm_llm.agents.react.ReactAgent.run", fake_run)
        cfg = AgentConfig(model="mock/model", verification_fn=lambda a, c: True)
        r = _agent(cfg).run("q")
        assert r.success is True

    def test_bool_verdict_supported(self, monkeypatch):
        def fake_run(self, task, initial_context=None):
            return _result("ok")

        monkeypatch.setattr("fsm_llm.agents.react.ReactAgent.run", fake_run)
        cfg = AgentConfig(model="mock/model", verification_fn=lambda a, c: True)
        agent = _agent(cfg)
        assert agent.run("q").answer == "ok"

    def test_verification_exception_is_rejection(self, monkeypatch):
        """REACT-04: a raising verifier fails closed, it never passes."""
        tasks: list[str] = []

        def fake_run(self, task, initial_context=None):
            tasks.append(task)
            return _result("x")

        def boom(a, c):
            raise RuntimeError("verifier crashed")

        monkeypatch.setattr("fsm_llm.agents.react.ReactAgent.run", fake_run)
        agent = _agent(AgentConfig(model="mock/model", verification_fn=boom))
        r = agent.run("q")
        assert len(tasks) == 2  # retried like any rejection
        assert "verifier crashed" in tasks[1]
        assert r.answer == "x"
        assert r.success is False
        assert r.stop_reason == StopReason.VERIFICATION_FAILED


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
        assert "Reflection" in payload[ContextKeys.AGENT_FEEDBACK]

    def test_reflection_is_not_an_observation(self):
        """REACT-04: a note in observations counts as evidence (observation_count)."""
        agent = _agent(AgentConfig(model="mock/model", reflect_every_n=1))
        api = self._api()
        agent._on_loop_iteration(api, "cid", 1)
        payload = api.update_context.call_args[0][1]
        assert ContextKeys.OBSERVATIONS not in payload
        assert ContextKeys.OBSERVATION_COUNT not in payload

    def test_reflection_keeps_pending_feedback(self):
        agent = _agent(AgentConfig(model="mock/model", reflect_every_n=1))
        api = MagicMock()
        api.get_data.return_value = {ContextKeys.AGENT_FEEDBACK: "tool was denied"}
        agent._on_loop_iteration(api, "cid", 1)
        note = api.update_context.call_args[0][1][ContextKeys.AGENT_FEEDBACK]
        assert note.startswith("tool was denied") and "Reflection" in note

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
