"""
Tests for ``fsm_llm.synthesis.library``: the content-addressed procedure store,
its lifecycle edge table, atomic writes, fail-loud corruption handling and the
signature-first retrieval.
"""

import contextlib
import threading
import time
from pathlib import Path
from typing import Any, ClassVar

import pytest

from fsm_llm import API
from fsm_llm.definitions import FSMDefinition
from fsm_llm.synthesis import SynthesisError, library
from fsm_llm.synthesis.exceptions import LibraryError
from fsm_llm.synthesis.library import FSMLibrary, digest_of

GOAL = "book a table for dinner"


def _fsm(name: str = "Collect", tool: str | None = None) -> FSMDefinition:
    first: dict[str, Any] = {
        "id": "a",
        "description": "a",
        "purpose": "p",
        "transitions": [{"target_state": "b", "description": "go"}],
    }
    if tool:
        first["completion"] = {
            "tools": [{"type": "function", "function": {"name": tool}}]
        }
    return FSMDefinition(
        name=name,
        description="d",
        initial_state="a",
        states={
            "a": first,
            "b": {"id": "b", "description": "b", "purpose": "p"},
        },
    )


def _put(lib: FSMLibrary, name: str = "Collect", **kw: Any) -> library.LibraryRecord:
    kw.setdefault("goal", GOAL)
    kw.setdefault("inputs_required", ["party"])
    kw.setdefault("return_keys", ["slot"])
    return lib.put(_fsm(name), **kw)


@pytest.fixture
def lib(tmp_path: Path) -> FSMLibrary:
    return FSMLibrary(tmp_path / "lib")


def _files(lib: FSMLibrary) -> list[str]:
    return sorted(
        str(p.relative_to(lib.root)) for p in lib.root.rglob("*") if p.is_file()
    )


class TestDigest:
    def test_parity_with_api_fsm_id(self):
        d = _fsm()
        digest = digest_of(d)
        assert len(digest) == 64
        assert digest[:8] == API.process_fsm_definition(d)[1].rsplit("_", 1)[1]

    def test_content_not_construction_decides(self):
        assert digest_of(_fsm()) == digest_of(FSMDefinition(**_fsm().model_dump()))
        assert digest_of(_fsm("A")) != digest_of(_fsm("B"))


class TestPut:
    def test_record_and_round_trip(self, lib: FSMLibrary):
        rec = _put(lib, "Tooled", goal="x" * 900, inputs_required=["b", "a", "a"])
        assert lib.get_record(rec.digest) == rec
        assert rec.digest == digest_of(lib.get_definition(rec.digest))
        assert (rec.version, rec.lifecycle, rec.stats) == (
            "1.0",
            "draft",
            {"runs": 0, "successes": 0},
        )
        assert rec.signature["inputs_required"] == ["a", "b"]
        assert len(str(rec.provenance["goal"])) == 500
        assert str(rec.provenance["created_at"]).endswith("+00:00")
        assert _files(lib) == [
            f"definitions/{rec.digest}.json",
            f"records/{rec.digest}.json",
        ]

    def test_tools_used_comes_from_the_definition(self, lib: FSMLibrary):
        rec = lib.put(
            _fsm(tool="lookup"), goal=GOAL, inputs_required=[], return_keys=["k"]
        )
        assert rec.signature["tools_used"] == ["lookup"]

    def test_idempotent_and_distinct(self, lib: FSMLibrary):
        first = _put(lib)
        lib.record_run(first.digest, True)
        again = _put(lib, goal="another goal entirely")
        assert again.stats["runs"] == 1  # the existing record, unchanged
        assert again.provenance["goal"] == GOAL
        other = _put(lib, "Other")
        assert other.digest != first.digest and len(_files(lib)) == 4

    def test_approved_by_is_recorded_and_added_once_to_an_existing_record(
        self, lib: FSMLibrary
    ):
        first = _put(lib, lifecycle="verified")
        assert first.provenance["approved_by"] is None
        added = _put(lib, lifecycle="verified", approved_by="reviewer")
        assert added.provenance["approved_by"] == "reviewer"
        assert added.lifecycle == "verified" and added.stats == first.stats
        kept = _put(lib, approved_by="someone else")  # never overwritten
        assert kept.provenance["approved_by"] == "reviewer"
        assert lib.get_record(first.digest) == added
        fresh = _put(lib, "Other", approved_by="  x\x00y  ")
        assert fresh.provenance["approved_by"] == "xy" and fresh.lifecycle == "draft"

    def test_parent_bumps_minor_and_must_exist(self, lib: FSMLibrary):
        parent = _put(lib)
        child = _put(lib, "Fixed", parent=parent.digest)
        grand = _put(lib, "Fixed2", parent=child.digest)
        assert [child.version, grand.version] == ["1.1", "1.2"]
        assert child.provenance["parent"] == parent.digest
        with pytest.raises(LibraryError):
            _put(lib, "Orphan", parent="0" * 64)

    def test_new_entry_lifecycle_is_draft_or_verified(self, lib: FSMLibrary):
        assert _put(lib, lifecycle="verified").lifecycle == "verified"
        for bad in ("trusted", "deprecated", "nope"):
            with pytest.raises(SynthesisError):
                _put(lib, "Other", lifecycle=bad)


class TestLifecycle:
    _OPS = ("verify", "approve", "deprecate")
    _LEGAL: ClassVar[dict[tuple[str, str], str]] = {
        ("draft", "verify"): "verified",
        ("verified", "approve"): "trusted",
        ("draft", "deprecate"): "deprecated",
        ("verified", "deprecate"): "deprecated",
        ("trusted", "deprecate"): "deprecated",
    }

    @staticmethod
    def _call(lib: FSMLibrary, digest: str, op: str):
        return getattr(lib, op)(digest, *(("alice",) if op == "approve" else ()))

    @pytest.mark.parametrize("state", ["draft", "verified", "trusted", "deprecated"])
    @pytest.mark.parametrize("op", _OPS)
    def test_edge_table(self, lib: FSMLibrary, state: str, op: str):
        rec = _put(lib)
        for step in {
            "draft": [],
            "verified": ["verify"],
            "trusted": ["verify", "approve"],
            "deprecated": ["deprecate"],
        }[state]:
            self._call(lib, rec.digest, step)
        before = (lib.root / "records" / f"{rec.digest}.json").read_bytes()
        target = self._LEGAL.get((state, op))
        if target is None:
            with pytest.raises(LibraryError, match="illegal lifecycle edge"):
                self._call(lib, rec.digest, op)
            assert (lib.root / "records" / f"{rec.digest}.json").read_bytes() == before
        else:
            assert self._call(lib, rec.digest, op).lifecycle == target
            assert lib.get_record(rec.digest).lifecycle == target

    def test_approve_records_who_and_needs_a_name(self, lib: FSMLibrary):
        rec = _put(lib, lifecycle="verified")
        with pytest.raises(SynthesisError):
            lib.approve(rec.digest, "  ")
        assert lib.approve(rec.digest, "alice").provenance["approved_by"] == "alice"

    def test_record_run_trusts_a_verified_entry_at_the_threshold(self, lib: FSMLibrary):
        v = _put(lib, lifecycle="verified")
        d = _put(lib, "Draft")
        for _ in range(2):
            lib.record_run(v.digest, True, trust_after_successes=3)
            lib.record_run(d.digest, True, trust_after_successes=3)
        lib.record_run(v.digest, False, trust_after_successes=3)
        assert lib.get_record(v.digest).lifecycle == "verified"
        done = lib.record_run(v.digest, True, trust_after_successes=3)
        assert (done.lifecycle, done.stats) == ("trusted", {"runs": 4, "successes": 3})
        assert (
            lib.record_run(d.digest, True, trust_after_successes=1).lifecycle == "draft"
        )

    def test_record_run_without_a_threshold_never_trusts(self, lib: FSMLibrary):
        rec = _put(lib, lifecycle="verified")
        for _ in range(5):
            out = lib.record_run(rec.digest, True)
        assert (out.lifecycle, out.stats["successes"]) == ("verified", 5)
        lib.deprecate(rec.digest)
        assert lib.record_run(rec.digest, False).stats["runs"] == 6

    @pytest.mark.parametrize("bad", [0, -1, True, 1.5, "2"])
    def test_bad_arguments(self, lib: FSMLibrary, bad: Any):
        rec = _put(lib)
        with pytest.raises(SynthesisError):
            lib.record_run(rec.digest, True, trust_after_successes=bad)
        with pytest.raises(SynthesisError):
            lib.record_run(rec.digest, "yes")  # type: ignore[arg-type]


class TestSafety:
    @pytest.mark.parametrize(
        "bad", ["../x", "A" * 64, "a" * 63, "a" * 65, "", None, 123, "a" * 63 + "/"]
    )
    @pytest.mark.parametrize("method", ["get_record", "get_definition", "deprecate"])
    def test_ids_are_refused_before_any_path_is_built(
        self, lib: FSMLibrary, method: str, bad: Any
    ):
        with pytest.raises(LibraryError, match="invalid digest"):
            getattr(lib, method)(bad)
        assert not lib.root.exists()

    def test_tilde_is_expanded_once_in_the_constructor(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        home = FSMLibrary("~/procs")
        assert home.root == (tmp_path / "procs").resolve()
        _put(home)
        assert (tmp_path / "procs" / "records").is_dir()

    def test_root_that_is_a_file_is_refused(self, tmp_path: Path):
        (tmp_path / "f").write_text("x")
        with pytest.raises(LibraryError):
            FSMLibrary(tmp_path / "f")

    @pytest.mark.parametrize(
        ("fail", "raised"),
        [(OSError("disk full"), LibraryError), (RuntimeError, RuntimeError)],
    )
    def test_a_failed_replace_leaves_no_partial_or_temp_file(
        self, lib: FSMLibrary, monkeypatch, fail, raised
    ):
        def boom(*_a: Any) -> None:
            raise fail if isinstance(fail, BaseException) else fail()

        monkeypatch.setattr(library.os, "replace", boom)
        with pytest.raises(raised):
            _put(lib)
        assert _files(lib) == []

    def test_a_failed_second_write_keeps_the_first_file_whole(
        self, lib: FSMLibrary, monkeypatch
    ):
        rec = _put(lib)
        before = (lib.root / "records" / f"{rec.digest}.json").read_bytes()
        monkeypatch.setattr(library.os, "replace", lambda *_a: 1 / 0)
        with pytest.raises(ZeroDivisionError):
            lib.record_run(rec.digest, True)
        monkeypatch.undo()
        assert (lib.root / "records" / f"{rec.digest}.json").read_bytes() == before
        assert not [f for f in _files(lib) if f.endswith(".tmp")]

    def test_a_corrupt_record_is_never_overwritten(self, lib: FSMLibrary):
        rec = _put(lib)
        path = lib.root / "records" / f"{rec.digest}.json"
        path.write_text('{"digest": "torn')
        for act in (
            lambda: _put(lib),
            lambda: lib.record_run(rec.digest, True),
            lambda: lib.get_record(rec.digest),
            lambda: lib.search(
                GOAL,
                inputs_available=[],
                return_keys=[],
                tools_allowed=[],
                min_lifecycle="draft",
            ),
            lambda: _put(lib, "Child", parent=rec.digest),
        ):
            with pytest.raises(LibraryError):
                act()
        assert path.read_text() == '{"digest": "torn'

    @pytest.mark.parametrize(
        "damage",
        [
            lambda r: r.replace('"draft"', '"bogus"'),
            lambda r: r.replace('"version": "1.0"', '"version": "v1"'),
            lambda r: r[:-2] + ', "extra": 1}',
            lambda r: r.replace('"runs": 0', '"runs": -1'),
            lambda r: r.replace('"approved_by": null', '"approved_by": null, "x": "y"'),
        ],
    )
    def test_an_invalid_record_shape_is_refused(self, lib: FSMLibrary, damage):
        rec = _put(lib)
        path = lib.root / "records" / f"{rec.digest}.json"
        path.write_text(damage(path.read_text()))
        with pytest.raises(LibraryError):
            lib.get_record(rec.digest)

    def test_a_record_filed_under_another_digest_is_refused(self, lib: FSMLibrary):
        a, b = _put(lib), _put(lib, "Other")
        text = (lib.root / "records" / f"{a.digest}.json").read_text()
        (lib.root / "records" / f"{b.digest}.json").write_text(text)
        with pytest.raises(LibraryError, match="another digest"):
            lib.get_record(b.digest)

    def test_a_definition_that_does_not_hash_to_its_name_is_refused(
        self, lib: FSMLibrary
    ):
        rec = _put(lib)
        path = lib.root / "definitions" / f"{rec.digest}.json"
        path.write_text(path.read_text().replace('"Collect"', '"Hacked"'))
        with pytest.raises(LibraryError, match="does not match"):
            lib.get_definition(rec.digest)
        with pytest.raises(LibraryError):
            _put(lib)  # same digest again: the bad file is not repaired by a put
        assert '"Hacked"' in path.read_text()

    def test_a_definition_that_is_not_a_definition_is_refused(self, lib: FSMLibrary):
        import hashlib

        raw = b'{"name": "x"}'
        digest = hashlib.sha256(raw).hexdigest()
        (lib.root / "definitions").mkdir(parents=True)
        (lib.root / "definitions" / f"{digest}.json").write_bytes(raw)
        with pytest.raises(LibraryError, match="corrupt definition"):
            lib.get_definition(digest)

    def test_unknown_digest_is_a_library_error(self, lib: FSMLibrary):
        for get in (lib.get_record, lib.get_definition):
            with pytest.raises(LibraryError, match="no "):
                get("0" * 64)

    def test_symlinked_entries_are_refused(self, lib: FSMLibrary, tmp_path: Path):
        rec = _put(lib)
        path = lib.root / "records" / f"{rec.digest}.json"
        elsewhere = tmp_path / "copy.json"
        elsewhere.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(elsewhere)
        with pytest.raises(LibraryError, match="symlink"):
            lib.get_record(rec.digest)

    def test_a_directory_symlinked_outside_the_root_is_refused(
        self, lib: FSMLibrary, tmp_path: Path
    ):
        outside = tmp_path / "outside"
        outside.mkdir()
        lib.root.mkdir()
        (lib.root / "records").symlink_to(outside)
        with pytest.raises(LibraryError):
            _put(lib)
        assert list(outside.iterdir()) == []

    def test_an_unreadable_records_dir_is_a_library_error(self, lib: FSMLibrary):
        lib.root.mkdir()
        (lib.root / "records").write_text("not a dir")
        with pytest.raises(LibraryError):
            lib.search(
                GOAL,
                inputs_available=[],
                return_keys=[],
                tools_allowed=[],
                min_lifecycle="draft",
            )

    def test_a_stray_json_that_is_no_digest_fails_loud(self, lib: FSMLibrary):
        _put(lib)
        (lib.root / "records" / "notes.json").write_text("{}")
        with pytest.raises(LibraryError, match="invalid digest"):
            lib.search(
                GOAL,
                inputs_available=["party"],
                return_keys=[],
                tools_allowed=[],
                min_lifecycle="draft",
            )


class TestConcurrency:
    """Eight writers on one record; the lock makes the count exact."""

    @staticmethod
    def _hammer(lib: FSMLibrary, digest: str, monkeypatch) -> int:
        real = FSMLibrary._read_record

        def slow(self, d):  # widen the read-modify-write window
            out = real(self, d)
            time.sleep(0.002)
            return out

        monkeypatch.setattr(FSMLibrary, "_read_record", slow)
        threads = [
            threading.Thread(
                target=lambda: [lib.record_run(digest, True) for _ in range(5)]
            )
            for _ in range(8)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        monkeypatch.undo()
        return lib.get_record(digest).stats["runs"]

    def test_eight_writers_lose_no_count(self, lib: FSMLibrary, monkeypatch):
        rec = _put(lib)
        assert self._hammer(lib, rec.digest, monkeypatch) == 40

    def test_negative_control_without_the_lock_counts_are_lost(
        self, lib: FSMLibrary, monkeypatch
    ):
        rec = _put(lib)
        lib._lock = contextlib.nullcontext()  # type: ignore[assignment]
        assert self._hammer(lib, rec.digest, monkeypatch) < 40

    def test_double_put_from_threads_is_idempotent(self, lib: FSMLibrary):
        out: list[str] = []
        threads = [
            threading.Thread(target=lambda: out.append(_put(lib).digest))
            for _ in range(8)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(set(out)) == 1 and len(_files(lib)) == 2


class TestSearch:
    @staticmethod
    def _q(lib: FSMLibrary, goal: str = GOAL, **kw: Any):
        kw.setdefault("inputs_available", ["party", "date"])
        kw.setdefault("return_keys", ["slot"])
        kw.setdefault("tools_allowed", [])
        kw.setdefault("min_lifecycle", "draft")
        return lib.search(goal, **kw)

    def test_empty_library(self, lib: FSMLibrary):
        assert self._q(lib) == []

    def test_word_overlap_ranks_and_ties_break_by_digest(self, lib: FSMLibrary):
        near = _put(lib, "Near", goal="book a dinner table")
        far = _put(lib, "Far", goal="reset a password")
        hits = self._q(lib, "book dinner table")
        assert [h.record.digest for h in hits] == [near.digest, far.digest]
        assert hits[0].score > hits[1].score == 0
        tie = [h.record.digest for h in self._q(lib, "zzz")]
        assert tie == sorted(tie)
        assert len(self._q(lib, top_k=1)) == 1

    @pytest.mark.parametrize(
        ("put_kw", "query_kw"),
        [
            ({"inputs_required": ["secret"]}, {}),
            ({"return_keys": ["other"]}, {}),
            ({"tool": "lookup"}, {}),
            ({}, {"min_lifecycle": "verified"}),
            ({"lifecycle": "verified", "deprecate": True}, {}),
        ],
    )
    def test_signature_filter_beats_a_higher_score(
        self, lib: FSMLibrary, put_kw: dict, query_kw: dict
    ):
        put_kw = dict(put_kw)
        tool, dep = put_kw.pop("tool", None), put_kw.pop("deprecate", False)
        kw = {
            "goal": GOAL,
            "inputs_required": ["party"],
            "return_keys": ["slot"],
            **put_kw,
        }
        excluded = lib.put(_fsm("Excluded", tool=tool), **kw)
        if dep:
            lib.deprecate(excluded.digest)
        allowed = _put(lib, "Allowed", goal="something unrelated", lifecycle="verified")

        def embed(texts):  # the excluded entry is a perfect match
            return [[1.0, 0.0] if t == GOAL else [0.0, 1.0] for t in texts]

        for e in (embed, None):
            hits = self._q(lib, GOAL, embed=e, **query_kw)
            assert [h.record.digest for h in hits] == [allowed.digest]

    def test_tools_allowed_admits_a_tool_procedure(self, lib: FSMLibrary):
        rec = lib.put(
            _fsm(tool="lookup"), goal=GOAL, inputs_required=[], return_keys=["slot"]
        )
        assert [h.record.digest for h in self._q(lib, tools_allowed=["lookup"])] == [
            rec.digest
        ]

    def test_min_lifecycle_orders_draft_verified_trusted(self, lib: FSMLibrary):
        d, v = _put(lib), _put(lib, "V", lifecycle="verified")
        t = _put(lib, "T", lifecycle="verified")
        lib.approve(t.digest, "alice")
        got = lambda m: {h.record.digest for h in self._q(lib, min_lifecycle=m)}  # noqa: E731
        assert got("draft") == {d.digest, v.digest, t.digest}
        assert got("verified") == {v.digest, t.digest}
        assert got("trusted") == {t.digest}

    def test_cosine_ranks_through_one_batch_call(self, lib: FSMLibrary):
        a = _put(lib, "A", goal="alpha")
        b = _put(lib, "B", goal="beta")
        calls: list[list[str]] = []

        def embed(texts):
            calls.append(list(texts))
            vec = {"query": [1.0, 0.0], "alpha": [0.0, 1.0], "beta": [0.9, 0.1]}
            return [vec[t] for t in texts]

        hits = self._q(lib, "query", embed=embed)
        assert [h.record.digest for h in hits] == [b.digest, a.digest]
        assert hits[0].score == pytest.approx(0.9 / (0.81 + 0.01) ** 0.5)
        assert len(calls) == 1 and calls[0][0] == "query" and len(calls[0]) == 3

    @pytest.mark.parametrize(
        "embed",
        [
            lambda t: 1 / 0,
            lambda t: [[1.0]],
            lambda t: [[1.0, 0.0]] * (len(t) - 1) + [[1.0]],
            lambda t: [[float("nan"), 0.0]] * len(t),
            lambda t: [[float("inf"), 0.0]] * len(t),
            lambda t: [[]] * len(t),
            lambda t: [["x", "y"]] * len(t),
            lambda t: None,
        ],
        ids=[
            "raises",
            "wrong-count",
            "uneven",
            "nan",
            "inf",
            "empty",
            "garbage",
            "none",
        ],
    )
    def test_a_bad_embedder_falls_back_to_word_overlap_with_one_warning(
        self, lib: FSMLibrary, monkeypatch, embed
    ):
        near = _put(lib, "Near", goal="book a dinner table")
        _put(lib, "Far", goal="reset a password")
        seen: list[str] = []

        class Rec:
            def warning(self, msg: str) -> None:
                seen.append(msg)

        monkeypatch.setattr(library, "logger", Rec())
        hits = self._q(lib, "book dinner table", embed=embed)
        assert hits[0].record.digest == near.digest
        assert hits[0].score == pytest.approx(3 / 4)  # Jaccard, not a cosine
        assert len(seen) == 1 and "word overlap" in seen[0]

    @pytest.mark.parametrize("kw", [{"min_lifecycle": "deprecated"}, {"top_k": 0}])
    def test_bad_search_arguments(self, lib: FSMLibrary, kw: dict):
        with pytest.raises(SynthesisError):
            self._q(lib, **kw)
