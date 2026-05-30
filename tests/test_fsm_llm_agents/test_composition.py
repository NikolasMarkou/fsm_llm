"""Tests for composition helpers: react_worker_factory + default_llm_judge."""

from __future__ import annotations

from fsm_llm_agents import (
    AgentConfig,
    ToolRegistry,
    default_llm_judge,
    react_worker_factory,
    tool,
)
from fsm_llm_agents.definitions import AgentResult, AgentTrace, EvaluationResult


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

        monkeypatch.setattr("fsm_llm_agents.react.ReactAgent.run", fake_run)
        worker = react_worker_factory(_registry(), AgentConfig(model="mock/model"))
        r = worker("subtask-A")
        assert isinstance(r, AgentResult)
        assert r.answer == "done:subtask-A"
        assert seen == ["subtask-A"]

    def test_fresh_agent_each_call(self, monkeypatch):
        instances = []

        orig_init = __import__(
            "fsm_llm_agents.react", fromlist=["ReactAgent"]
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

        monkeypatch.setattr("fsm_llm_agents.react.ReactAgent.__init__", tracking_init)
        monkeypatch.setattr("fsm_llm_agents.react.ReactAgent.run", fake_run)
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
