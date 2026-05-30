"""Tests for auto-memory: recall-before / remember-after at the run() boundary."""

from __future__ import annotations

import pytest

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

    def _patch_fail(self, monkeypatch):
        def fake_run(self, task, initial_context=None):
            return AgentResult(
                answer="garbage",
                success=False,
                trace=AgentTrace(tool_calls=[], total_iterations=1),
            )

        monkeypatch.setattr("fsm_llm_agents.react.ReactAgent.run", fake_run)

    def test_default_persists_unsuccessful_run(self, monkeypatch):
        # New default (remember_only_on_success=False): conversational turns
        # that the guard flags success=False MUST still persist.
        store = _store()
        agent = AutoMemoryReactAgent(
            tools=_registry(), config=AgentConfig(model="mock/model"), memory=store
        )
        self._patch_fail(monkeypatch)
        agent.run("remember: my name is Nikolas")
        assert len(store) == 1

    def test_remember_only_on_success_opt_in(self, monkeypatch):
        store = _store()
        agent = AutoMemoryReactAgent(
            tools=_registry(),
            config=AgentConfig(model="mock/model"),
            memory=store,
            remember_only_on_success=True,  # explicit opt-in
        )
        self._patch_fail(monkeypatch)
        agent.run("q")
        assert len(store) == 0  # failed run not stored when opted in

    def test_persists_task_when_run_raises(self, monkeypatch):
        # If the underlying run raises (budget/timeout), the user-stated fact
        # must not be lost — persisted via the finally path.
        store = _store()
        agent = AutoMemoryReactAgent(
            tools=_registry(), config=AgentConfig(model="mock/model"), memory=store
        )

        def boom(self, task, initial_context=None):
            raise RuntimeError("budget exhausted")

        monkeypatch.setattr("fsm_llm_agents.react.ReactAgent.run", boom)
        with pytest.raises(RuntimeError):
            agent.run("remember: my name is Nikolas")
        assert len(store) == 1  # fact preserved despite the raise

    def test_no_persist_on_raise_when_disabled(self, monkeypatch):
        store = _store()
        agent = AutoMemoryReactAgent(
            tools=_registry(),
            config=AgentConfig(model="mock/model"),
            memory=store,
            auto_remember=False,
        )

        def boom(self, task, initial_context=None):
            raise RuntimeError("x")

        monkeypatch.setattr("fsm_llm_agents.react.ReactAgent.run", boom)
        with pytest.raises(RuntimeError):
            agent.run("q")
        assert len(store) == 0
