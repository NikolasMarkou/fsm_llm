"""
Tests for ``fsm_llm.synthesis.library``: the content-addressed procedure store,
its lifecycle edge table, atomic writes, fail-loud corruption handling and the
signature-first retrieval.
"""

import contextlib
import inspect
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, ClassVar

import pytest

from fsm_llm import API
from fsm_llm.definitions import FSMDefinition
from fsm_llm.synthesis import RunTrace, SynthesisError, library
from fsm_llm.synthesis.constants import DEFAULT_MAX_TRACES
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

    def test_gates_passed_gains_only_the_gates_the_record_lacks(self, lib: FSMLibrary):
        first = _put(lib, gates_passed=["G1", "G2"])
        more = _put(lib, gates_passed=["G1", "G2", "G6b"])
        assert more.provenance["gates_passed"] == ["G1", "G2", "G6b"]
        assert lib.get_record(first.digest) == more
        assert _put(lib, gates_passed=["G1"]) == more  # nothing to add, unchanged

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

    def test_a_repaired_child_is_never_trusted_by_count_only_approve_does(
        self, lib: FSMLibrary
    ):
        """D-038 item 4, D-062: a record with `repair` provenance."""
        parent = _put(lib, lifecycle="verified")
        child = lib.put(
            _fsm("Child"),
            goal=GOAL,
            inputs_required=["party"],
            return_keys=["slot"],
            lifecycle="verified",
            parent=parent.digest,
            repair="ab" * 32,
        )
        for _ in range(4):
            out = lib.record_run(child.digest, True, trust_after_successes=1)
        assert (out.lifecycle, out.stats["successes"]) == ("verified", 4)
        assert out.provenance["approved_by"] is None
        assert lib.approve(child.digest, "alice").lifecycle == "trusted"
        # the control: the parent, with no repair provenance, still earns it
        assert (
            lib.record_run(parent.digest, True, trust_after_successes=1).lifecycle
            == "trusted"
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
        assert hits[0].score == pytest.approx(3 / 4 / 2)  # half a Jaccard
        assert not hits[0].embedded
        assert len(seen) == 1 and "word overlap" in seen[0]

    @pytest.mark.parametrize(
        ("query", "exact"),
        [
            ("book a table for dinner", True),
            ("  Book   A\tTABLE \n for  dinner ", True),  # case and whitespace only
            ("book a table, for dinner!!", False),  # punctuation is part of the goal
            ("book a table for dinner tonight", False),  # an extra word
            ("book a table for dinner now please", False),
            ("do NOT book a table for dinner", False),  # a negation
            ("dinner for a table book", False),  # the same words, reordered
            ("cancel a table for dinner", False),  # one word swapped
        ],
    )
    def test_the_word_fallback_scores_one_only_for_the_same_goal(
        self, lib: FSMLibrary, query: str, exact: bool
    ):
        _put(lib)
        (hit,) = self._q(lib, query)
        assert not hit.embedded
        assert (hit.score == 1.0) is exact
        assert exact or 0 < hit.score <= 0.5  # it ranks, it can never be reused

    @pytest.mark.parametrize(
        ("stored", "query"),
        [
            ("count answers > 10", "count answers < 10"),
            ("count answers > 10", "count answers >= 10"),
            ("learn C++", "learn C#"),
            ("balance of -5", "balance of 5"),
            ("pay 1.5 euro", "pay 1 5 euro"),
            ("?", "!!!"),
            ("?", "?!"),
        ],
    )
    def test_goals_that_differ_in_a_symbol_are_not_the_same_goal(
        self, lib: FSMLibrary, stored: str, query: str
    ):
        _put(lib, "Sym", goal=stored)
        (hit,) = self._q(lib, query)
        assert not hit.embedded and hit.score < 1.0
        (same,) = self._q(lib, stored.upper() + "  ")
        assert same.score == 1.0  # case and trailing whitespace only

    @pytest.mark.parametrize("empty", ["", " ", " \t\n "])
    def test_an_empty_goal_never_matches_exactly(self, lib: FSMLibrary, empty: str):
        _put(lib, "Blank", goal="x")
        (hit,) = self._q(lib, empty)
        assert hit.score == 0.0
        # the stored side may be empty too (a hand-written record): still no match
        _put(lib, "Empty", goal="  ")
        assert all(h.score == 0.0 for h in self._q(lib, empty))
        assert all(h.score < 1.0 for h in self._q(lib, "  "))

    def test_an_embedder_score_is_flagged_as_a_cosine(self, lib: FSMLibrary):
        _put(lib)
        (hit,) = self._q(lib, "anything", embed=lambda t: [[1.0, 0.0]] * len(t))
        assert hit.embedded and hit.score == pytest.approx(1.0)

    @pytest.mark.parametrize("kw", [{"min_lifecycle": "deprecated"}, {"top_k": 0}])
    def test_bad_search_arguments(self, lib: FSMLibrary, kw: dict):
        with pytest.raises(SynthesisError):
            self._q(lib, **kw)


D = "c" * 64


def _tr(i: int = 0, digest: str = D, **kw: Any) -> RunTrace:
    return RunTrace(digest=digest, steps=i, states=("a", "b"), **kw)


def _trace_file(lib: FSMLibrary, digest: str = D) -> Path:
    return lib.root / "traces" / f"{digest}.jsonl"


class TestTraces:
    def test_add_and_read_oldest_first(self, lib: FSMLibrary):
        for i in range(4):
            lib.add_trace(_tr(i), max_traces=10)
        found, skipped = lib.read_traces(D)
        assert [t.steps for t in found] == [0, 1, 2, 3] and skipped == 0
        assert [t.steps for t in lib.read_traces(D, limit=2)[0]] == [2, 3]
        assert lib.read_traces("d" * 64) == ((), 0)

    def test_every_row_carries_the_version(self, lib: FSMLibrary):
        lib.add_trace(_tr(), max_traces=10)
        assert json.loads(_trace_file(lib).read_text())["v"] == 1

    def test_the_ring_cap_trims_the_oldest(self, lib: FSMLibrary):
        for i in range(7):
            lib.add_trace(_tr(i), max_traces=3)
        assert [t.steps for t in lib.read_traces(D)[0]] == [4, 5, 6]
        assert len(_trace_file(lib).read_text().splitlines()) == 3

    def test_a_trace_is_filed_under_its_own_digest(self, lib: FSMLibrary):
        lib.add_trace(_tr(1, "d" * 64), max_traces=3)
        lib.add_trace(_tr(2), max_traces=3)
        assert [t.steps for t in lib.read_traces(D)[0]] == [2]

    def test_the_ring_cap_has_one_default(self, lib: FSMLibrary):
        """``TraceRecorder(lib.add_trace)`` is the one wiring line (D-064)."""
        default = inspect.signature(FSMLibrary.add_trace).parameters["max_traces"]
        assert default.default == DEFAULT_MAX_TRACES == 200
        lib.add_trace(_tr(1))
        assert [t.steps for t in lib.read_traces(D)[0]] == [1]

    def test_retire_traces_moves_the_file_aside_and_keeps_it(self, lib: FSMLibrary):
        for i in range(3):
            lib.add_trace(_tr(i), max_traces=10)
        child = "e" * 64
        lib.retire_traces(D, child)

        assert lib.read_traces(D) == ((), 0)  # a later read sees only newer ones
        kept = lib.root / "traces" / f"{D}.{child}.jsonl"
        assert len(kept.read_text().splitlines()) == 3
        lib.add_trace(_tr(9), max_traces=10)
        assert [t.steps for t in lib.read_traces(D)[0]] == [9]
        assert len(kept.read_text().splitlines()) == 3  # never touched again
        lib.retire_traces("d" * 64, child)  # no file: nothing to move, no error

    @pytest.mark.parametrize("bad", ["../x", "A" * 64, "", None, 5])
    def test_retire_traces_refuses_a_bad_digest(self, lib: FSMLibrary, bad):
        lib.add_trace(_tr(), max_traces=3)
        with pytest.raises(LibraryError, match="invalid digest"):
            lib.retire_traces(D, bad)
        with pytest.raises(LibraryError, match="invalid digest"):
            lib.retire_traces(bad, "e" * 64)
        assert len(lib.read_traces(D)[0]) == 1

    @pytest.mark.parametrize("bad", ["../x", "A" * 64, "a" * 63, "", None, 5])
    def test_a_bad_id_is_refused_before_a_path_is_built(self, lib: FSMLibrary, bad):
        with pytest.raises(LibraryError, match="invalid digest"):
            lib.read_traces(bad)
        if isinstance(bad, str):
            with pytest.raises(LibraryError, match="invalid digest"):
                lib.add_trace(RunTrace(digest=bad), max_traces=3)
        assert not lib.root.exists()

    @pytest.mark.parametrize("bad", [0, -1, True, 2.0, "3", None, 10**30])
    def test_a_bad_cap_or_limit_is_refused(self, lib: FSMLibrary, bad):
        with pytest.raises(SynthesisError):
            lib.add_trace(_tr(), max_traces=bad)
        if bad is not None:  # no limit means every kept trace
            with pytest.raises(SynthesisError):
                lib.read_traces(D, limit=bad)

    def test_a_torn_last_line_is_skipped_counted_and_healed(self, lib: FSMLibrary):
        lib.add_trace(_tr(1), max_traces=10)
        with _trace_file(lib).open("a") as f:
            f.write('{"v": 1, "digest": "' + D + '", "states": ["a"')  # no newline
        found, skipped = lib.read_traces(D)
        assert [t.steps for t in found] == [1] and skipped == 1

        lib.add_trace(_tr(2), max_traces=10)  # the next add keeps only whole rows
        found, skipped = lib.read_traces(D)
        assert [t.steps for t in found] == [1, 2] and skipped == 0

    def test_a_hostile_file_never_crashes_a_reader(self, lib: FSMLibrary):
        good = json.dumps(_tr(9).to_row())
        wrong_digest = json.dumps(_tr(1, "d" * 64).to_row())
        lines = [
            b"\xff\xfe not utf-8",
            b"[" * 5000,  # deep nesting, under the row size
            json.dumps(
                {**_tr(7).to_row(), "reason": "x" * 20000}
            ).encode(),  # valid, long
            b"[]",
            b"null",
            b'"text"',
            b"{}",
            b'{"v": 2}',
            wrong_digest.encode(),
            json.dumps({**_tr().to_row(), "status": "running"}).encode(),
            json.dumps({**_tr().to_row(), "states": [{"a": 1}]}).encode(),
            good.encode(),
        ]
        path = _trace_file(lib)
        path.parent.mkdir(parents=True)
        path.write_bytes(b"\n".join(lines) + b"\n")
        found, skipped = lib.read_traces(D)
        assert [t.steps for t in found] == [9] and skipped == len(lines) - 1

    def test_a_huge_file_is_read_by_its_tail_only(self, lib: FSMLibrary):
        row = (json.dumps(_tr(5).to_row()) + "\n").encode()
        path = _trace_file(lib)
        path.parent.mkdir(parents=True)
        path.write_bytes(row * (6 * 2**20 // len(row)))
        found, skipped = lib.read_traces(D)
        assert found and skipped == 0 and len(found) < 6 * 2**20 // len(row)

    def test_a_symlinked_or_unreadable_trace_file_is_a_library_error(
        self, lib: FSMLibrary, tmp_path: Path
    ):
        path = _trace_file(lib)
        path.parent.mkdir(parents=True)
        (tmp_path / "x").write_text("")
        os.symlink(tmp_path / "x", path)
        with pytest.raises(LibraryError):
            lib.read_traces(D)
        with pytest.raises(LibraryError):
            lib.add_trace(_tr(), max_traces=3)
        path.unlink()
        path.mkdir()  # a directory where the file should be
        with pytest.raises(LibraryError):
            lib.read_traces(D)

    def test_search_lists_only_records(self, lib: FSMLibrary):
        rec = _put(lib)
        lib.add_trace(_tr(1, rec.digest), max_traces=3)
        lib.add_scenarios(rec.digest, [{"kind": "happy", "turns": ["hi"]}])
        hits = TestSearch._q(lib)
        assert [h.record.digest for h in hits] == [rec.digest]

    @staticmethod
    def _hammer(lib: FSMLibrary, monkeypatch) -> int:
        real = FSMLibrary._write

        def slow(self, path, text):  # widen the read-modify-write window
            time.sleep(0.002)
            real(self, path, text)

        monkeypatch.setattr(FSMLibrary, "_write", slow)
        threads = [
            threading.Thread(
                target=lambda: [lib.add_trace(_tr(), max_traces=1000) for _ in range(5)]
            )
            for _ in range(8)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        monkeypatch.undo()
        return len(lib.read_traces(D)[0])

    def test_eight_writers_lose_no_trace(self, lib: FSMLibrary, monkeypatch):
        assert self._hammer(lib, monkeypatch) == 40

    def test_negative_control_without_the_lock_traces_are_lost(
        self, lib: FSMLibrary, monkeypatch
    ):
        lib._lock = contextlib.nullcontext()  # type: ignore[assignment]
        assert self._hammer(lib, monkeypatch) < 40


class TestScenarios:
    def test_round_trip_and_replace(self, lib: FSMLibrary):
        assert lib.read_scenarios(D) == []
        lib.add_scenarios(D, [{"kind": "happy", "turns": ["hi", "yes"]}])
        assert lib.read_scenarios(D) == [{"kind": "happy", "turns": ["hi", "yes"]}]
        lib.add_scenarios(D, [{"kind": "off_topic", "turns": ["x"]}])
        assert [s["kind"] for s in lib.read_scenarios(D)] == ["off_topic"]

    def test_text_is_stripped_and_capped(self, lib: FSMLibrary):
        lib.add_scenarios(
            D,
            [
                {"kind": "k" * 200, "turns": ["a\x1b[31m\x00b\nc"] + ["t" * 1000] * 20}
                for _ in range(10)
            ],
        )
        got = lib.read_scenarios(D)
        assert len(got) == 6 and all(len(s["turns"]) == 12 for s in got)
        assert got[0]["kind"] == "k" * 120
        assert got[0]["turns"][0] == "ab c" and got[0]["turns"][1] == "t" * 300

    @pytest.mark.parametrize(
        "bad",
        [
            "turns",
            None,
            [None],
            [{"kind": "x"}],
            [{"turns": ["a"]}],
            [{"kind": 5, "turns": ["a"]}],
            [{"kind": "x", "turns": "abc"}],
            [{"kind": "x", "turns": [1]}],
            [{"kind": "x", "turns": [True]}],
        ],
    )
    def test_a_bad_shape_is_refused_and_nothing_is_written(self, lib: FSMLibrary, bad):
        with pytest.raises(SynthesisError):
            lib.add_scenarios(D, bad)
        assert not lib.root.exists()

    def test_a_bad_id_is_refused(self, lib: FSMLibrary):
        with pytest.raises(LibraryError, match="invalid digest"):
            lib.add_scenarios("x", [])
        with pytest.raises(LibraryError, match="invalid digest"):
            lib.read_scenarios("x")

    @pytest.mark.parametrize(
        "text",
        ["{", "[1]", '{"a": 1}', '[{"kind": "x"}]', '[{"kind": 1, "turns": []}]'],
    )
    def test_a_corrupt_file_is_refused_never_overwritten(self, lib: FSMLibrary, text):
        path = lib.root / "scenarios" / f"{D}.json"
        path.parent.mkdir(parents=True)
        path.write_text(text)
        with pytest.raises(LibraryError):
            lib.read_scenarios(D)
        assert path.read_text() == text


def _child(lib: FSMLibrary, parent: library.LibraryRecord, **kw: Any):
    """A second definition stored as a repair of ``parent`` (same goal)."""
    kw.setdefault("goal", GOAL)
    return lib.put(
        _fsm("Collect2"),
        inputs_required=["party"],
        return_keys=["slot"],
        lifecycle="verified",
        parent=parent.digest,
        **kw,
    )


class TestRepairLineage:
    def test_a_tie_returns_the_child_of_a_crash_between_the_two_writes(self, lib):
        parent = _put(lib, lifecycle="verified")
        child = _child(lib, parent)
        assert (parent.version, child.version) == ("1.0", "1.1")
        hits = TestSearch._q(lib, min_lifecycle="verified")
        assert [h.record.digest for h in hits] == [child.digest, parent.digest]
        assert hits[0].score == hits[1].score

    def test_lifecycle_rank_outranks_the_minor_version_in_a_tie(self, lib):
        parent = _put(lib, lifecycle="verified")
        child = _child(lib, parent)
        lib.approve(parent.digest, "boss")
        hits = TestSearch._q(lib, min_lifecycle="verified")
        assert [h.record.digest for h in hits] == [parent.digest, child.digest]

    def test_the_version_is_compared_as_numbers(self, lib):
        rec = _put(lib, "V0", lifecycle="verified")
        for i in range(1, 11):
            rec = lib.put(
                _fsm(f"V{i}"),
                goal=GOAL,
                inputs_required=["party"],
                return_keys=["slot"],
                lifecycle="verified",
                parent=rec.digest,
            )
        assert rec.version == "1.10"
        hits = TestSearch._q(lib, min_lifecycle="verified")
        assert hits[0].record.version == "1.10"  # "1.9" > "1.10" as text

    def test_a_better_score_still_beats_rank_and_version(self, lib):
        far = _put(lib, "Far", goal="reset a password", lifecycle="verified")
        _child(lib, far, goal="reset a password")
        near = _put(lib, "Near", goal=GOAL)
        hits = TestSearch._q(lib, min_lifecycle="draft")
        assert hits[0].record.digest == near.digest

    def test_provenance_holds_the_repair_and_replay_strings(self, lib):
        parent = _put(lib, lifecycle="verified")
        child = _child(lib, parent, repair="a" * 64, replay="3/4 vs 1/4")
        assert child.provenance["repair"] == "a" * 64
        assert child.provenance["replay"] == "3/4 vs 1/4"
        assert lib.get_record(child.digest) == child
        assert _put(lib, "Collect").provenance.get("repair") is None

    def test_add_patch_is_atomic_json_under_patches(self, lib, monkeypatch):
        rec = _put(lib, lifecycle="verified")
        ops = [{"op": "replace", "path": "/states/a/purpose", "value": "x"}]
        seen: list[str] = []
        real = FSMLibrary._write

        def spy(self, path, text):
            seen.append(path.parent.name)
            real(self, path, text)

        monkeypatch.setattr(FSMLibrary, "_write", spy)
        lib.add_patch(rec.digest, ops)
        assert seen == ["patches"]
        stored = json.loads((lib.root / "patches" / f"{rec.digest}.json").read_text())
        assert stored == ops
        assert not [f for f in _files(lib) if f.endswith(".tmp")]
        with pytest.raises(LibraryError):
            lib.add_patch("../x", ops)
        assert [h.record.digest for h in TestSearch._q(lib)] == [rec.digest]

    def test_deprecate_can_leave_a_trusted_record_alone(self, lib):
        rec = _put(lib, lifecycle="verified")
        lib.approve(rec.digest, "boss")
        assert lib.deprecate(rec.digest, keep_trusted=True).lifecycle == "trusted"
        assert lib.get_record(rec.digest).lifecycle == "trusted"
        other = _put(lib, "Other", lifecycle="verified")
        assert lib.deprecate(other.digest, keep_trusted=True).lifecycle == "deprecated"
        assert lib.deprecate(rec.digest).lifecycle == "deprecated"  # explicit call

    def test_deprecate_keep_trusted_is_idempotent_on_a_deprecated_record(self, lib):
        rec = _put(lib, lifecycle="verified")
        lib.deprecate(rec.digest)
        assert lib.deprecate(rec.digest, keep_trusted=True).lifecycle == "deprecated"
        with pytest.raises(LibraryError):  # the plain call still refuses the edge
            lib.deprecate(rec.digest)

    def _kid(self, lib, name, parent, i):
        return lib.put(
            _fsm(name),
            goal=GOAL,
            inputs_required=["party"],
            return_keys=["slot"],
            lifecycle="verified",
            parent=parent.digest,
            repair=f"{i}" * 64,
        )

    def test_count_lineage_counts_every_repair_of_the_tree_from_the_root(self, lib):
        root = _put(lib, lifecycle="verified")
        assert lib.count_lineage(root.digest) == 0
        kids = [self._kid(lib, f"Kid{i}", root, i) for i in range(3)]
        grand = self._kid(lib, "Grand", kids[1], 4)
        lib.deprecate(kids[0].digest)  # a retired child still counts
        _put(lib, "Stranger", lifecycle="verified")
        # the same tree whichever node asks: 3 children and 1 grandchild
        for node in (root, *kids, grand):
            assert lib.count_lineage(node.digest) == 4
        with pytest.raises(LibraryError, match="invalid digest"):
            lib.count_lineage("../x")

    def test_count_lineage_stops_at_a_parent_that_is_not_stored(self, lib):
        root = _put(lib, lifecycle="verified")
        kid = self._kid(lib, "Kid", root, 1)
        self._kid(lib, "Grand", kid, 2)
        (lib.root / "records" / f"{root.digest}.json").unlink()
        assert lib.count_lineage(kid.digest) == 1  # the kid is the root now

    def test_count_lineage_survives_a_cycle_in_a_corrupt_store(self, lib):
        root = _put(lib, lifecycle="verified")
        kid = self._kid(lib, "Kid", root, 1)
        path = lib.root / "records" / f"{root.digest}.json"
        data = json.loads(path.read_text())
        data["provenance"]["parent"] = kid.digest
        path.write_text(json.dumps(data))
        assert lib.count_lineage(kid.digest) == 1

    def test_clean_scenarios_is_the_public_cleaner(self):
        out = FSMLibrary.clean_scenarios(
            [{"kind": "happy", "turns": ["a\x00b" + "x" * 400]}] * 9
        )
        assert len(out) == 6 and len(out[0]["turns"][0]) <= 300
        assert "\x00" not in out[0]["turns"][0]
        with pytest.raises(ValueError):
            FSMLibrary.clean_scenarios("nope")
