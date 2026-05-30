"""Tests for auto-memory: recall-before / remember-after at the run() boundary."""

from __future__ import annotations

from fsm_llm_agents import (
    AgentConfig,
    AutoMemoryReactAgent,
    SemanticMemoryStore,
    ToolRegistry,
    augment_task_with_memories,
    remember_interaction,
    tool,
)
from fsm_llm_agents.definitions import AgentResult, AgentTrace


def _fake_embed(text: str) -> list[float]:
    t = text.lower()
    return [
        float(t.count("python") + t.count("language") + t.count("code")),
        float(t.count("cat") + t.count("pet")),
    ]


def _store():
    return SemanticMemoryStore(embed_fn=_fake_embed)


@tool
def noop(query: str) -> str:
    """No-op."""
    return query


def _registry():
    reg = ToolRegistry()
    reg.register(noop._tool_definition)
    return reg


class TestHelpers:
    def test_augment_injects_relevant_memory(self):
        s = _store()
        s.add("My favorite language is Python")
        s.add("I have a pet cat")
        out = augment_task_with_memories("what programming language do I like?", s)
        assert "Python" in out
        assert out.endswith("what programming language do I like?")

    def test_augment_noop_when_empty(self):
        s = _store()
        task = "anything"
        assert augment_task_with_memories(task, s) == task

    def test_augment_survives_search_failure(self):
        class Broken:
            def add(self, *a, **k):
                pass

            def search(self, *a, **k):
                raise RuntimeError("down")

        task = "hello"
        assert augment_task_with_memories(task, Broken()) == task

    def test_remember_interaction_stores(self):
        s = _store()
        remember_interaction(s, "Q?", "the answer")
        assert len(s) == 1

    def test_remember_skips_empty_answer(self):
        s = _store()
        remember_interaction(s, "Q?", "   ")
        assert len(s) == 0


class TestAutoMemoryReactAgent:
    def _agent(self, monkeypatch, store, **kw):
        agent = AutoMemoryReactAgent(
            tools=_registry(),
            config=AgentConfig(model="mock/model"),
            memory=store,
            **kw,
        )
        captured = {}

        def fake_run(self, task, initial_context=None):
            captured["task"] = task
            return AgentResult(
                answer="Python",
                success=True,
                trace=AgentTrace(tool_calls=[], total_iterations=1),
            )

        # Patch the ReactAgent.run that super().run() resolves to.
        monkeypatch.setattr("fsm_llm_agents.react.ReactAgent.run", fake_run)
        return agent, captured

    def test_recall_injected_into_task(self, monkeypatch):
        store = _store()
        store.add("My favorite language is Python")
        agent, captured = self._agent(monkeypatch, store)
        agent.run("what language do I like?")
        assert "Python" in captured["task"]

    def test_auto_remember_persists_interaction(self, monkeypatch):
        store = _store()
        agent, _ = self._agent(monkeypatch, store)
        agent.run("tell me about python")
        assert len(store) == 1

    def test_no_remember_when_disabled(self, monkeypatch):
        store = _store()
        agent, _ = self._agent(monkeypatch, store, auto_remember=False)
        agent.run("tell me about python")
        assert len(store) == 0

    def test_defaults_to_semantic_store(self):
        agent = AutoMemoryReactAgent(
            tools=_registry(), config=AgentConfig(model="mock/model")
        )
        assert isinstance(agent.memory, SemanticMemoryStore)

    def test_remember_only_on_success(self, monkeypatch):
        store = _store()
        agent = AutoMemoryReactAgent(
            tools=_registry(),
            config=AgentConfig(model="mock/model"),
            memory=store,
        )

        def fake_run(self, task, initial_context=None):
            return AgentResult(
                answer="garbage",
                success=False,
                trace=AgentTrace(tool_calls=[], total_iterations=1),
            )

        monkeypatch.setattr("fsm_llm_agents.react.ReactAgent.run", fake_run)
        agent.run("q")
        assert len(store) == 0  # failed run not stored
