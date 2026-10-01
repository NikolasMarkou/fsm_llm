"""Tests for SemanticMemoryStore + create_semantic_memory_tools.

Embeddings are injected via embed_fn so tests are deterministic and need no LLM.
"""

from __future__ import annotations

from typing import Any

import pytest

from fsm_llm.agents import (
    SemanticMemoryStore,
    create_semantic_memory_tools,
)
from fsm_llm.agents.definitions import ToolCall
from fsm_llm.agents.tools import ToolRegistry


def _fake_embed(text: str) -> list[float]:
    """Toy embedding: counts of a few keywords → a 3-dim vector."""
    t = text.lower()
    return [
        float(t.count("python") + t.count("code")),
        float(t.count("cat") + t.count("dog") + t.count("pet")),
        float(t.count("paris") + t.count("france") + t.count("travel")),
    ]


def _store():
    return SemanticMemoryStore(embed_fn=_fake_embed)


class TestSemanticMemoryStore:
    def test_add_and_len(self):
        s = _store()
        eid = s.add("I love Python code")
        assert eid
        assert len(s) == 1

    def test_empty_text_rejected(self):
        with pytest.raises(ValueError):
            _store().add("   ")

    def test_semantic_search_ranks_by_meaning(self):
        s = _store()
        s.add("My favorite programming language is Python")
        s.add("I have a pet cat named Whiskers")
        s.add("I want to travel to Paris, France")

        top = s.search("tell me about coding", k=1)
        assert len(top) == 1
        assert "Python" in top[0][0]

        top_pet = s.search("animals and pets", k=1)
        assert "cat" in top_pet[0][0]

    def test_search_respects_k(self):
        s = _store()
        for i in range(5):
            s.add(f"python fact {i}")
        assert len(s.search("python", k=3)) == 3

    def test_search_empty_store(self):
        assert _store().search("anything") == []

    def test_forget(self):
        s = _store()
        eid = s.add("python rocks")
        assert s.forget(eid) is True
        assert len(s) == 0
        assert s.forget("nonexistent") is False

    def test_substring_fallback_when_embedding_unavailable(self):
        # embed_fn returns None -> no embeddings -> substring fallback
        s = SemanticMemoryStore(
            embed_fn=lambda t: (_ for _ in ()).throw(RuntimeError())
        )
        s.add("The capital of France is Paris")
        s.add("Unrelated fact about dogs")
        results = s.search("france")
        assert any("France" in r[0] for r in results)

    def test_clear(self):
        s = _store()
        s.add("a python thing")
        s.clear()
        assert len(s) == 0


class TestPersistence:
    def test_roundtrip_to_dict(self):
        s = _store()
        s.add("python", metadata={"src": "test"})
        data = s.to_dict()
        restored = SemanticMemoryStore.from_dict(data, embed_fn=_fake_embed)
        assert len(restored) == 1
        assert restored.all_entries()[0].metadata == {"src": "test"}

    def test_save_load_file(self, tmp_path):
        path = str(tmp_path / "mem.json")
        s = SemanticMemoryStore(embed_fn=_fake_embed, persist_path=path)
        s.add("My favorite language is Python")
        s.add("I have a cat")

        # auto-persist already wrote the file; load it back
        loaded = SemanticMemoryStore.load(path, embed_fn=_fake_embed)
        assert len(loaded) == 2
        # cached embeddings survive → semantic search works without re-embedding
        top = loaded.search("coding", k=1)
        assert "Python" in top[0][0]

    def test_save_requires_path(self):
        with pytest.raises(ValueError):
            _store().save()


class TestSemanticMemoryTools:
    def test_creates_remember_and_recall(self):
        tools = create_semantic_memory_tools(_store())
        names = {t.name for t in tools}
        assert names == {"remember", "recall"}

    def test_tools_drive_the_store(self):
        store = _store()
        registry = ToolRegistry()
        for t in create_semantic_memory_tools(store):
            registry.register(t)

        r = registry.execute(
            ToolCall(
                tool_name="remember",
                parameters={"content": "My favorite language is Python"},
            )
        )
        assert r.success
        assert len(store) == 1

        recall = registry.execute(
            ToolCall(tool_name="recall", parameters={"query": "coding languages"})
        )
        assert recall.success
        assert "Python" in recall.result

    def test_recall_empty(self):
        store = _store()
        registry = ToolRegistry()
        for t in create_semantic_memory_tools(store):
            registry.register(t)
        recall = registry.execute(
            ToolCall(tool_name="recall", parameters={"query": "anything"})
        )
        assert "No memories" in recall.result


class _EmbeddingProvider:
    """A fake provider for both embedding bindings (core's and litellm's).

    Answers one ``_fake_embed`` vector per input text in a real
    ``EmbeddingResponse``, or raises when ``fail``. ``requests`` holds the
    keyword arguments of every request.
    """

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.requests: list[dict[str, Any]] = []

    def __call__(self, **params: Any) -> Any:
        from litellm.types.utils import EmbeddingResponse

        self.requests.append(params)
        if self.fail:
            raise ConnectionError("embedding provider down")
        return EmbeddingResponse(
            model=params["model"],
            data=[
                {"object": "embedding", "index": i, "embedding": _fake_embed(t)}
                for i, t in enumerate(params["input"])
            ],
        )


@pytest.fixture
def provider(monkeypatch) -> _EmbeddingProvider:
    """Route core's embedding binding and litellm's module function to a fake."""
    import litellm

    import fsm_llm.llm

    fake = _EmbeddingProvider()
    monkeypatch.setattr(fsm_llm.llm, "embedding", fake)
    monkeypatch.setattr(litellm, "embedding", fake)
    return fake


class TestDefaultEmbedder:
    """Plan 944e2692 step 8 (D-005): with no ``embed_fn`` the store embeds
    through core's ``LiteLLMEmbedder`` (its meter counts every request)."""

    def test_add_and_search_send_through_core_embedder(self, provider):
        store = SemanticMemoryStore(embedding_model="ollama/qwen3-embedding:0.6b")
        store.add("My cat sleeps all day")
        store.add("Paris is my favourite travel spot")
        hits = store.search("tell me about my pet", k=1)

        assert hits[0][0] == "My cat sleeps all day"
        assert [r["input"] for r in provider.requests] == [
            ["My cat sleeps all day"],
            ["Paris is my favourite travel spot"],
            ["tell me about my pet"],
        ]
        assert {r["model"] for r in provider.requests} == {
            "ollama/qwen3-embedding:0.6b"
        }
        usage = store._embed_texts.__self__.usage()  # the store's LiteLLMEmbedder
        assert usage.calls == 3
        assert usage.errors == 0

    def test_provider_outage_stores_unembedded_and_recalls_by_substring(self, provider):
        provider.fail = True
        store = SemanticMemoryStore()
        store.add("Remember the dentist on Friday")

        assert store.all_entries()[0].embedding is None
        hits = store.search("dentist")
        assert [h[0] for h in hits] == ["Remember the dentist on Friday"]
        assert hits[0][1] == 1.0
        assert len(provider.requests) == 2

    def test_embed_fn_bypasses_the_provider(self, provider):
        store = SemanticMemoryStore(embed_fn=_fake_embed)
        store.add("I write Python code")
        store.search("python")

        assert provider.requests == []
