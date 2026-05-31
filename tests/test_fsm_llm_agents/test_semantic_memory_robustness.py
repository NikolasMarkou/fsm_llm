"""Robustness tests for SemanticMemoryStore (Step 4 of plan_2026-05-31_f08da86d).

Covers:
- 4a: in-process thread safety (a ``threading.Lock`` guards add/forget/clear/save;
  N concurrent ``add()`` calls yield N entries with distinct ids).
- 4b: ``from_dict`` warns (loguru) when the stored embedding model differs from
  the active one.
- 4c: optional ``max_entries`` FIFO cap evicts oldest; default is unbounded.

Embeddings are injected via ``embed_fn`` so tests are deterministic and need no LLM.
"""

from __future__ import annotations

import threading

from fsm_llm.logging import logger
from fsm_llm_agents import SemanticMemoryStore


def _fake_embed(text: str) -> list[float]:
    """Deterministic toy embedding; length-based 1-dim vector."""
    return [float(len(text))]


class TestConcurrentAdd:
    def test_lock_present(self):
        store = SemanticMemoryStore(embed_fn=_fake_embed)
        assert hasattr(store, "_lock")

    def test_concurrent_add_yields_n_distinct_entries(self):
        store = SemanticMemoryStore(embed_fn=_fake_embed)
        n = 8
        barrier = threading.Barrier(n)

        def worker(i: int) -> None:
            # Align thread starts to maximize contention on _counter/_entries.
            barrier.wait()
            store.add(f"fact number {i}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(store) == n
        ids = [e.id for e in store.all_entries()]
        assert len(set(ids)) == n  # all distinct, no lost/duplicated counter


class TestEmbeddingModelMismatch:
    def test_from_dict_warns_on_mismatch(self):
        store = SemanticMemoryStore(embedding_model="model-A", embed_fn=_fake_embed)
        store.add("hello world")
        data = store.to_dict()
        assert data["embedding_model"] == "model-A"

        # Library logging is opt-in (logger.disable("fsm_llm") at import); enable
        # the package prefix so the WARNING reaches our sink, then restore.
        logger.enable("fsm_llm")
        captured: list[str] = []
        sink_id = logger.add(lambda msg: captured.append(str(msg)), level="WARNING")
        try:
            # Active model "model-B" differs from stored "model-A" -> warn.
            SemanticMemoryStore.from_dict(
                data, embed_fn=_fake_embed, embedding_model="model-B"
            )
        finally:
            logger.remove(sink_id)
            logger.disable("fsm_llm")

        joined = "\n".join(captured)
        assert "mismatch" in joined.lower()
        assert "model-A" in joined
        assert "model-B" in joined

    def test_from_dict_no_warn_on_match(self):
        store = SemanticMemoryStore(embedding_model="model-A", embed_fn=_fake_embed)
        store.add("hello world")
        data = store.to_dict()

        logger.enable("fsm_llm")
        captured: list[str] = []
        sink_id = logger.add(lambda msg: captured.append(str(msg)), level="WARNING")
        try:
            # No active model passed -> stored model reused -> no mismatch warning
            # (prior behavior preserved).
            restored = SemanticMemoryStore.from_dict(data, embed_fn=_fake_embed)
        finally:
            logger.remove(sink_id)
            logger.disable("fsm_llm")

        assert restored._embedding_model == "model-A"
        assert not any("mismatch" in m.lower() for m in captured)


class TestMaxEntriesCap:
    def test_cap_evicts_oldest_keeps_newest(self):
        store = SemanticMemoryStore(embed_fn=_fake_embed, max_entries=3)
        for i in range(5):
            store.add(f"entry-{i}")
        assert len(store) == 3
        texts = [e.text for e in store.all_entries()]
        # FIFO: oldest two (entry-0, entry-1) evicted; 3 newest retained in order.
        assert texts == ["entry-2", "entry-3", "entry-4"]

    def test_default_unbounded(self):
        store = SemanticMemoryStore(embed_fn=_fake_embed)
        for i in range(5):
            store.add(f"entry-{i}")
        assert len(store) == 5
