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


class TestLegacyFileWithoutModel:
    """Plan 0068b055 step 0 (D-015): a file written before ``embedding_model``
    was stored loads with the current default embedder; a stored name wins."""

    def test_legacy_file_without_model_uses_new_default(self):
        from fsm_llm.llm import DEFAULT_EMBEDDING_MODEL

        data = SemanticMemoryStore(embed_fn=_fake_embed).to_dict()
        data.pop("embedding_model", None)
        restored = SemanticMemoryStore.from_dict(data, embed_fn=_fake_embed)
        assert restored._embedding_model == DEFAULT_EMBEDDING_MODEL
        assert DEFAULT_EMBEDDING_MODEL == "ollama/embeddinggemma-2:740m"

    def test_stored_model_is_kept(self):
        store = SemanticMemoryStore(embedding_model="x/old", embed_fn=_fake_embed)
        restored = SemanticMemoryStore.from_dict(store.to_dict(), embed_fn=_fake_embed)
        assert restored._embedding_model == "x/old"


def _saved_store(path: str, model: str) -> None:
    """A persisted store whose file records ``model`` (stub embedder, 3-wide)."""
    store = SemanticMemoryStore(
        embedding_model=model, persist_path=path, embed_fn=_fake_embed
    )
    store.add("My cat sleeps all day")
    store.add("Paris is my favourite travel spot")


class TestConstructorAdoptsRecordedModel:
    """Plan 0068b055 D-018 (pass-2 concern 3): ``embedding_model`` defaults to
    ``None``; a persisted store reopened through its constructor keeps the model
    its file recorded, like ``load`` and ``from_dict``. An explicit model wins."""

    def test_default_is_none(self):
        import inspect

        default = inspect.signature(SemanticMemoryStore).parameters["embedding_model"]
        assert default.default is None

    def test_reopened_store_keeps_the_recorded_model_and_searches(
        self, tmp_path, provider
    ):
        path = str(tmp_path / "mem.json")
        _saved_store(path, "ollama/qwen3-embedding:0.6b")

        reopened = SemanticMemoryStore(persist_path=path)

        assert reopened._embedding_model == "ollama/qwen3-embedding:0.6b"
        hits = reopened.search("tell me about my pet", k=1)
        assert hits[0][0] == "My cat sleeps all day"
        assert hits[0][1] > 0.0
        assert {r["model"] for r in provider.requests} == {
            "ollama/qwen3-embedding:0.6b"
        }

    def test_explicit_model_wins_and_warns(self, tmp_path):
        from fsm_llm.logging import logger

        path = str(tmp_path / "mem.json")
        _saved_store(path, "model-A")

        logger.enable("fsm_llm")
        captured: list[str] = []
        sink_id = logger.add(lambda msg: captured.append(str(msg)), level="WARNING")
        try:
            reopened = SemanticMemoryStore(
                embedding_model="model-B", persist_path=path, embed_fn=_fake_embed
            )
        finally:
            logger.remove(sink_id)
            logger.disable("fsm_llm")

        assert reopened._embedding_model == "model-B"
        joined = "\n".join(captured)
        assert "mismatch" in joined.lower()
        assert "model-A" in joined and "model-B" in joined

    def test_matching_recorded_model_does_not_warn(self, tmp_path):
        from fsm_llm.logging import logger

        path = str(tmp_path / "mem.json")
        _saved_store(path, "model-A")

        logger.enable("fsm_llm")
        captured: list[str] = []
        sink_id = logger.add(lambda msg: captured.append(str(msg)), level="WARNING")
        try:
            SemanticMemoryStore(persist_path=path, embed_fn=_fake_embed)
        finally:
            logger.remove(sink_id)
            logger.disable("fsm_llm")

        assert not any("mismatch" in m.lower() for m in captured)

    @pytest.mark.parametrize(
        "content",
        [
            None,  # the file does not exist
            "",  # a blank file
            '{"counter": 0, "entries": []}',  # a legacy file: no recorded model
            '{"embedding_model": "", "entries": []}',
            '{"embedding_model": 7, "entries": []}',
        ],
        ids=["missing", "blank", "legacy", "blank-model", "non-string-model"],
    )
    def test_no_usable_recorded_model_uses_the_default(self, tmp_path, content):
        from fsm_llm.llm import DEFAULT_EMBEDDING_MODEL

        path = tmp_path / "mem.json"
        if content is not None:
            path.write_text(content, encoding="utf-8")

        store = SemanticMemoryStore(persist_path=str(path), embed_fn=_fake_embed)

        assert store._embedding_model == DEFAULT_EMBEDDING_MODEL

    def test_no_persist_path_uses_the_default(self):
        from fsm_llm.llm import DEFAULT_EMBEDDING_MODEL

        assert (
            SemanticMemoryStore(embed_fn=_fake_embed)._embedding_model
            == DEFAULT_EMBEDDING_MODEL
        )

    def test_the_file_is_read_once(self, tmp_path, monkeypatch):
        import fsm_llm.agents.semantic_memory as module

        path = str(tmp_path / "mem.json")
        _saved_store(path, "model-A")
        reads: list[str] = []
        real = module._read_payload
        monkeypatch.setattr(
            module, "_read_payload", lambda p: reads.append(p) or real(p)
        )

        store = SemanticMemoryStore(persist_path=path, embed_fn=_fake_embed)

        assert len(reads) == 1
        assert len(store) == 2

    def test_invalid_file_still_raises_and_is_left_untouched(self, tmp_path):
        target = tmp_path / "mem.json"
        target.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(ValueError, match=r"mem\.json"):
            SemanticMemoryStore(persist_path=str(target), embed_fn=_fake_embed)
        assert target.read_text(encoding="utf-8") == "[1, 2, 3]"


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
