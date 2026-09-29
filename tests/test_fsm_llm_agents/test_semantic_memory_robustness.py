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

import pytest

from fsm_llm.agents import SemanticMemoryStore
from fsm_llm.logging import logger


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


class TestCrossInstanceAtomicSave:
    """SC-8 / F-06: two INDEPENDENT stores persisting to ONE path.

    ``SemanticMemoryStore._lock`` is *per-instance*, so two instances sharing a
    ``persist_path`` (two processes, or two stores built from one file) have
    zero mutual exclusion. With the old fixed ``f"{target}.tmp"`` temp name both
    writers open the SAME temp path and whichever ``os.replace`` lands first
    consumes it out from under the other -- measured at 1190/1200 (99.2%)
    ``FileNotFoundError``. A single-instance probe cannot reproduce this, which
    is why both clauses below run across two stores.
    """

    @staticmethod
    def _run_trials(tmp_path, trials: int) -> tuple[list[BaseException], list[str]]:
        """Run ``trials`` rounds of two stores saving to one path simultaneously.

        Returns ``(exceptions_raised, leftover_tmp_filenames)``.
        """
        target = str(tmp_path / "mem.json")
        stores = []
        for tag in ("a", "b"):
            store = SemanticMemoryStore(persist_path=target, embed_fn=_fake_embed)
            # Enough entries that json.dump holds the temp file open long
            # enough for the two writers to actually overlap.
            for i in range(30):
                store.add(f"{tag} fact number {i}")
            stores.append(store)

        errors: list[BaseException] = []
        errors_lock = threading.Lock()

        def worker(store: SemanticMemoryStore, barrier: threading.Barrier) -> None:
            barrier.wait()
            try:
                store.save()
            except Exception as e:  # the failure rate IS the measurement
                with errors_lock:
                    errors.append(e)

        for _ in range(trials):
            barrier = threading.Barrier(len(stores))
            threads = [
                threading.Thread(target=worker, args=(s, barrier)) for s in stores
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        residue = sorted(p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp"))
        return errors, residue

    def test_concurrent_cross_instance_save_is_atomic(self, tmp_path):
        """SC-8, both clauses: 0 errors AND 0 ``.tmp`` residue over 200 trials.

        The residue clause is not decoration -- a "fix" that gives each write a
        unique temp name but forgets the ``finally``-unlink would satisfy the
        no-error clause while littering the directory on every failed write.
        """
        errors, residue = self._run_trials(tmp_path, trials=200)

        assert errors == [], (
            f"{len(errors)}/400 concurrent saves failed; "
            f"first: {type(errors[0]).__name__}: {errors[0]}"
        )
        assert residue == [], f"temp files left behind: {residue}"

    def test_save_leaves_no_temp_file_when_serialization_fails(self, tmp_path):
        """The ``finally``-unlink runs on non-OSError failures too.

        Pins the reason ``session.py``'s D-011 rejected ``except OSError``: a
        ``TypeError`` out of ``json.dump`` must not leak the temp file either.
        """
        target = str(tmp_path / "mem.json")
        store = SemanticMemoryStore(persist_path=target, embed_fn=_fake_embed)
        store.add("a fact")

        class _Unserializable:
            def __repr__(self) -> str:
                return "<unserializable>"

        store._entries[0].metadata = {"bad": _Unserializable()}

        with pytest.raises(TypeError):
            store.save()

        residue = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
        assert residue == [], f"temp files left behind after a failed save: {residue}"


def _capture_warnings(fn):
    """Run ``fn`` with fsm_llm logging enabled; return (result, WARNING texts)."""
    logger.enable("fsm_llm")
    captured: list[str] = []
    sink_id = logger.add(lambda msg: captured.append(str(msg)), level="WARNING")
    try:
        result = fn()
    finally:
        logger.remove(sink_id)
        logger.disable("fsm_llm")
    return result, captured


class TestPersistPathLoadOnInit:
    """MEM-01: a store built on an existing ``persist_path`` keeps prior sessions."""

    def test_second_store_sees_first_store_entries(self, tmp_path):
        target = str(tmp_path / "mem.json")
        first = SemanticMemoryStore(persist_path=target, embed_fn=_fake_embed)
        first.add("my favourite language is Python")
        first.add("I have a cat")

        second = SemanticMemoryStore(persist_path=target, embed_fn=_fake_embed)
        assert [e.text for e in second.all_entries()] == [
            "my favourite language is Python",
            "I have a cat",
        ]

        # The first add of the new session keeps the old entries on disk and
        # continues the id counter instead of reusing mem-1.
        eid = second.add("I live in Athens")
        assert eid == "mem-3"
        reloaded = SemanticMemoryStore.load(target, embed_fn=_fake_embed)
        assert len(reloaded) == 3

    def test_load_on_init_makes_no_embedding_call(self, tmp_path):
        target = str(tmp_path / "mem.json")
        SemanticMemoryStore(persist_path=target, embed_fn=_fake_embed).add("a fact")

        calls: list[str] = []

        def counting_embed(text: str) -> list[float]:
            calls.append(text)
            return _fake_embed(text)

        store = SemanticMemoryStore(persist_path=target, embed_fn=counting_embed)
        assert len(store) == 1
        assert calls == []

    def test_missing_file_starts_empty(self, tmp_path):
        store = SemanticMemoryStore(
            persist_path=str(tmp_path / "absent.json"), embed_fn=_fake_embed
        )
        assert len(store) == 0

    def test_empty_file_starts_empty(self, tmp_path):
        target = tmp_path / "mem.json"
        target.write_text("  \n", encoding="utf-8")
        store = SemanticMemoryStore(persist_path=str(target), embed_fn=_fake_embed)
        assert len(store) == 0

    def test_corrupt_file_raises_and_is_not_clobbered(self, tmp_path):
        target = tmp_path / "mem.json"
        target.write_text("{not json", encoding="utf-8")
        with pytest.raises(ValueError, match=r"mem\.json"):
            SemanticMemoryStore(persist_path=str(target), embed_fn=_fake_embed)
        assert target.read_text(encoding="utf-8") == "{not json"

    def test_non_store_json_raises(self, tmp_path):
        target = tmp_path / "mem.json"
        target.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(ValueError, match=r"mem\.json"):
            SemanticMemoryStore(persist_path=str(target), embed_fn=_fake_embed)

    def test_init_load_warns_on_model_mismatch(self, tmp_path):
        target = str(tmp_path / "mem.json")
        SemanticMemoryStore(
            embedding_model="model-A", persist_path=target, embed_fn=_fake_embed
        ).add("hello world")

        _, captured = _capture_warnings(
            lambda: SemanticMemoryStore(
                embedding_model="model-B", persist_path=target, embed_fn=_fake_embed
            )
        )
        joined = "\n".join(captured)
        assert "mismatch" in joined.lower()
        assert "model-A" in joined and "model-B" in joined


class TestUnembeddedEntriesSearchable:
    """MEM-02: entries whose embedding failed stay reachable by substring."""

    def test_unembedded_entry_found_beside_embedded_ones(self):
        def flaky_embed(text: str) -> list[float]:
            if text == "apple pie recipe":
                raise RuntimeError("provider down")
            return _fake_embed(text)

        store = SemanticMemoryStore(embed_fn=flaky_embed)
        store.add("banana bread")
        store.add("apple pie recipe")
        assert store.all_entries()[1].embedding is None

        texts = [text for text, _score, _meta in store.search("apple", k=5)]
        assert "apple pie recipe" in texts
        assert "banana bread" in texts

    def test_unembedded_non_matching_entry_not_returned(self):
        def flaky_embed(text: str) -> list[float]:
            if text == "carrot cake":
                raise RuntimeError("provider down")
            return _fake_embed(text)

        store = SemanticMemoryStore(embed_fn=flaky_embed)
        store.add("banana bread")
        store.add("carrot cake")
        texts = [text for text, _score, _meta in store.search("apple", k=5)]
        assert texts == ["banana bread"]


class TestMaxEntriesRoundTrip:
    """MEM-03: ``max_entries`` survives to_dict/from_dict, save/load and init load."""

    def test_to_dict_from_dict_round_trips_max_entries(self):
        store = SemanticMemoryStore(embed_fn=_fake_embed, max_entries=3)
        data = store.to_dict()
        assert data["max_entries"] == 3
        restored = SemanticMemoryStore.from_dict(data, embed_fn=_fake_embed)
        assert restored._max_entries == 3

    def test_from_dict_without_max_entries_is_unbounded(self):
        restored = SemanticMemoryStore.from_dict(
            {"embedding_model": "m", "counter": 0, "entries": []}, embed_fn=_fake_embed
        )
        assert restored._max_entries is None

    def test_load_restores_cap_and_evicts_on_next_add(self, tmp_path):
        target = str(tmp_path / "mem.json")
        store = SemanticMemoryStore(
            persist_path=target, embed_fn=_fake_embed, max_entries=2
        )
        store.add("one")
        store.add("two")
        loaded = SemanticMemoryStore.load(target, embed_fn=_fake_embed)
        loaded.add("three")
        assert [e.text for e in loaded.all_entries()] == ["two", "three"]

    def test_init_load_uses_stored_cap_unless_given(self, tmp_path):
        target = str(tmp_path / "mem.json")
        SemanticMemoryStore(
            persist_path=target, embed_fn=_fake_embed, max_entries=2
        ).add("one")
        assert (
            SemanticMemoryStore(persist_path=target, embed_fn=_fake_embed)._max_entries
            == 2
        )
        assert (
            SemanticMemoryStore(
                persist_path=target, embed_fn=_fake_embed, max_entries=5
            )._max_entries
            == 5
        )


class TestSaveDurability:
    """MEM-03: save creates the parent directory and fsyncs before replacing."""

    def test_missing_parent_dir_is_created(self, tmp_path):
        target = tmp_path / "nested" / "deeper" / "mem.json"
        store = SemanticMemoryStore(persist_path=str(target), embed_fn=_fake_embed)
        store.add("a fact")
        assert target.exists()
        assert len(SemanticMemoryStore.load(str(target), embed_fn=_fake_embed)) == 1

    def test_temp_file_fsynced_before_replace(self, tmp_path, monkeypatch):
        import os

        order: list[str] = []
        real_fsync, real_replace = os.fsync, os.replace

        def spy_fsync(fd):
            order.append("fsync")
            return real_fsync(fd)

        def spy_replace(src, dst):
            order.append("replace")
            return real_replace(src, dst)

        monkeypatch.setattr(os, "fsync", spy_fsync)
        monkeypatch.setattr(os, "replace", spy_replace)
        SemanticMemoryStore(embed_fn=_fake_embed).save(str(tmp_path / "mem.json"))
        assert order[:2] == ["fsync", "replace"]


class TestCosineDimensionMismatch:
    """MEM-04: vectors of different width score 0.0 with a WARNING."""

    def test_mismatch_scores_zero_and_warns(self):
        from fsm_llm.agents.semantic_tools import _cosine_similarity

        score, captured = _capture_warnings(
            lambda: _cosine_similarity([1.0, 0.0], [1.0, 0.0, 5.0])
        )
        assert score == 0.0
        assert any("dimension" in m.lower() for m in captured)

    def test_search_ranks_mismatched_entry_last(self):
        vectors = {"python code": [1.0, 0.0], "old model fact": [1.0, 0.0, 9.0]}
        store = SemanticMemoryStore(embed_fn=lambda t: vectors.get(t, [1.0, 0.0]))
        store.add("old model fact")
        store.add("python code")
        results = store.search("coding", k=2)
        assert results[0][0] == "python code"
        assert results[1][1] == 0.0
