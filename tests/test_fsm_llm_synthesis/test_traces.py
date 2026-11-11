"""
Tests for ``fsm_llm.synthesis.traces``: the value-free ``RunTrace``, the
``TraceRecorder`` that turns runner events into one, and the converter for a
simulation trial. The library's trace store is tested in ``test_library.py``.
"""

import dataclasses
from types import SimpleNamespace
from typing import Any

import pytest

from fsm_llm.synthesis import RunTrace, TraceRecorder
from fsm_llm.synthesis.constants import TRACE_KINDS
from fsm_llm.synthesis.traces import trace_from_trial
from tests.test_fsm_llm_synthesis.conftest import ScriptedRunnerLLM
from tests.test_fsm_llm_synthesis.test_runner import _rig

D1 = "a" * 64
D2 = "b" * 64


def _trace(**kw: Any) -> RunTrace:
    kw.setdefault("digest", D1)
    return RunTrace(**kw)


class TestRunTrace:
    def test_a_row_round_trips(self):
        t = _trace(
            version="1.2",
            mode="converse",
            scenario="happy",
            states=("a", "b"),
            steps=2,
            keys=("k",),
            kinds=(("k", "int"),),
            status="aborted",
            reason="budget_spent",
            stuck="a",
            outcome="blocked",
        )
        row = t.to_row()
        assert row["v"] == 1 and row["states"] == ["a", "b"]
        assert RunTrace.from_row(row) == t

    def test_it_is_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            _trace().steps = 3  # type: ignore[misc]

    @pytest.mark.parametrize(
        "bad",
        [
            {"mode": "fast"},
            {"status": "running"},
            {"outcome": "maybe"},
            {"digest": 7},
            {"states": ("a", 1)},
            {"states": "abc"},
            {"keys": (["k"],)},
            {"kinds": (("k", "object"),)},
            {"kinds": (("k",),)},
            {"steps": -1},
            {"steps": True},
            {"steps": 1.5},
            {"reason": 5},
            {"stuck": 5},
        ],
    )
    def test_a_wrong_shape_is_refused(self, bad: dict[str, Any]):
        with pytest.raises((TypeError, ValueError)):
            _trace(**bad)

    def test_every_closed_type_name_is_accepted(self):
        names = (*TRACE_KINDS.values(), "other")
        assert _trace(kinds=tuple(("k", n) for n in names)).kinds

    def test_text_is_stripped_and_capped_and_the_tuples_are_cut(self):
        t = _trace(
            states=tuple(f"s{i}" for i in range(500)),
            keys=tuple(f"k{i}" for i in range(500)),
            kinds=tuple((f"k{i}", "int") for i in range(500)),
            reason="x" * 500,
            stuck="line\x1b[31m\x00\nbreak",
        )
        assert (len(t.states), len(t.keys), len(t.kinds)) == (100, 64, 64)
        assert t.states[-1] == "s499"  # the last states are the ones that matter
        assert t.reason == "x" * 80
        assert t.stuck == "line break"

    @pytest.mark.parametrize("row", [None, [], "x", 5, {}, {"v": 1}, {"digest": D1}])
    def test_from_row_refuses_anything_that_is_not_a_trace_row(self, row: Any):
        with pytest.raises((TypeError, ValueError)):
            RunTrace.from_row(row)

    def test_from_row_refuses_another_version(self):
        row = _trace().to_row()
        row["v"] = 2
        with pytest.raises(ValueError):
            RunTrace.from_row(row)


def _events(
    recorder: TraceRecorder, *events: tuple[str, dict[str, Any]], conv: str = "c1"
) -> None:
    for name, fields in events:
        recorder(name, {"conv": conv, **fields})


class TestRecorder:
    def _run(self, **popped: Any) -> tuple[list[RunTrace], TraceRecorder]:
        got: list[RunTrace] = []
        rec = TraceRecorder(got.append)
        _events(
            rec,
            ("obtained", {"digest": D1, "version": "1.3", "source": "compiled"}),
            ("pushed", {"digest": D1, "step": 3}),
            (
                "stepped",
                {"digest": D1, "step": 1, "state": "ask", "keys": "topic", "kinds": ""},
            ),
            (
                "stepped",
                {
                    "digest": D1,
                    "step": 2,
                    "state": "done",
                    "keys": "answer,topic",
                    "kinds": "answer:str,topic:int",
                },
            ),
            (
                "popped",
                {
                    "digest": D1,
                    "status": "ok",
                    "reason": None,
                    "step": 5,
                    **popped,
                },
            ),
        )
        return got, rec

    def test_an_ok_run_yields_one_trace(self):
        got, _ = self._run()
        assert got == [
            RunTrace(
                digest=D1,
                version="1.3",
                mode="run",
                states=("ask", "done"),
                steps=2,
                keys=("answer", "topic"),
                kinds=(("answer", "str"), ("topic", "int")),
                status="ok",
            )
        ]

    def test_converse_mode_is_read_from_a_step_of_none(self):
        got: list[RunTrace] = []
        rec = TraceRecorder(got.append)
        _events(
            rec,
            ("pushed", {"digest": D1, "step": None}),
            ("popped", {"digest": D1, "status": "ok", "reason": None, "step": None}),
        )
        assert got[0].mode == "converse" and got[0].steps == 0

    def test_an_abort_names_the_stuck_state_and_outcome(self):
        got: list[RunTrace] = []
        rec = TraceRecorder(got.append)
        _events(
            rec,
            ("pushed", {"digest": D1, "step": 2}),
            (
                "stepped",
                {"digest": D1, "step": 1, "state": "ask", "keys": "", "kinds": ""},
            ),
            (
                "aborted",
                {
                    "digest": D1,
                    "step": 4,
                    "state": "ask",
                    "outcome": "ambiguous",
                    "keys": "topic",
                    "kinds": "topic:str",
                },
            ),
            (
                "popped",
                {
                    "digest": D1,
                    "status": "aborted",
                    "reason": "budget_spent",
                    "step": 4,
                },
            ),
        )
        (t,) = got
        assert (t.status, t.reason) == ("aborted", "budget_spent")
        assert (t.stuck, t.outcome) == ("ask", "ambiguous")
        assert (t.keys, t.kinds) == (("topic",), (("topic", "str"),))

    def test_a_failure_after_a_push_keeps_what_was_seen(self):
        got: list[RunTrace] = []
        rec = TraceRecorder(got.append)
        _events(
            rec,
            ("obtained", {"digest": D1, "version": "1.0", "source": "library"}),
            ("pushed", {"digest": D1, "step": 2}),
            ("failed", {"digest": D1, "reason": "unexpected_error", "step": 2}),
        )
        (t,) = got
        assert (t.status, t.reason, t.version) == ("failed", "unexpected_error", "1.0")

    def test_a_refused_push_is_a_failed_trace_and_a_compile_failure_is_none(self):
        got: list[RunTrace] = []
        rec = TraceRecorder(got.append)
        _events(
            rec,
            ("obtained", {"digest": D1, "version": "1.0", "source": "library"}),
            ("failed", {"digest": D1, "reason": "lifecycle_too_low", "step": None}),
            ("failed", {"digest": None, "reason": "compile_failed", "step": 3}),
        )
        assert [(t.status, t.mode, t.reason) for t in got] == [
            ("failed", "converse", "lifecycle_too_low")
        ]

    def test_two_conversations_interleaved_are_separate_runs(self):
        got: list[RunTrace] = []
        rec = TraceRecorder(got.append)
        for conv, digest in (("c1", D1), ("c2", D2)):
            _events(rec, ("pushed", {"digest": digest, "step": 1}), conv=conv)
        _events(
            rec,
            (
                "stepped",
                {"digest": D1, "step": 1, "state": "x", "keys": "", "kinds": ""},
            ),
            conv="c1",
        )
        for conv, digest in (("c2", D2), ("c1", D1)):
            _events(
                rec,
                (
                    "popped",
                    {"digest": digest, "status": "ok", "reason": None, "step": 2},
                ),
                conv=conv,
            )
        assert [(t.digest, t.states) for t in got] == [(D2, ()), (D1, ("x",))]

    def test_a_run_is_forgotten_once_it_is_yielded(self):
        got, rec = self._run()
        _events(
            rec,
            ("pushed", {"digest": D1, "step": 9}),
            ("popped", {"digest": D1, "status": "failed", "reason": "x", "step": 9}),
        )
        assert got[1].states == () and got[1].version is None

    def test_a_pop_with_no_earlier_event_still_yields(self):
        got: list[RunTrace] = []
        _events(
            TraceRecorder(got.append),
            ("popped", {"digest": D1, "status": "ok", "reason": None, "step": 4}),
        )
        assert got[0].mode == "run" and got[0].states == ()

    @pytest.mark.parametrize(
        "event",
        [
            ("nonsense", {"digest": D1}),
            ("stepped", {}),
            ("popped", {}),
            ("popped", {"digest": None, "status": "ok"}),
            ("pushed", {"digest": D1, "step": "x"}),
            ("stepped", {"digest": D1, "step": "x", "state": 7, "keys": 5}),
        ],
    )
    def test_a_malformed_event_never_raises(self, event: tuple[str, dict[str, Any]]):
        got: list[RunTrace] = []
        rec = TraceRecorder(got.append)
        rec(event[0], dict(event[1]))
        rec(event[0], {"conv": "c1", **event[1]})
        assert all(isinstance(t, RunTrace) for t in got)

    def test_unfinished_runs_are_bounded(self):
        got: list[RunTrace] = []
        rec = TraceRecorder(got.append)
        step = {"digest": D1, "step": 1, "state": "x", "keys": "", "kinds": ""}
        _events(
            rec, ("pushed", {"digest": D1, "step": 1}), ("stepped", step), conv="c0"
        )
        for i in range(300):
            _events(rec, ("pushed", {"digest": D1, "step": 1}), conv=f"n{i}")
        _events(
            rec,
            ("popped", {"digest": D1, "status": "ok", "reason": None, "step": 1}),
            conv="c0",
        )
        assert got[0].states == ()  # c0 was the oldest and is gone

    def test_a_sink_that_raises_leaves_the_recorder_usable(self):
        calls: list[RunTrace] = []

        def sink(trace: RunTrace) -> None:
            calls.append(trace)
            raise RuntimeError("disk full")

        rec = TraceRecorder(sink)
        for _ in range(2):
            with pytest.raises(RuntimeError):
                _events(
                    rec,
                    ("pushed", {"digest": D1, "step": 1}),
                    (
                        "stepped",
                        {
                            "digest": D1,
                            "step": 1,
                            "state": "x",
                            "keys": "",
                            "kinds": "",
                        },
                    ),
                    (
                        "popped",
                        {"digest": D1, "status": "ok", "reason": None, "step": 1},
                    ),
                )
        assert len(calls) == 2 and calls[1].states == ("x",)

    def test_a_sink_must_be_callable(self):
        with pytest.raises(TypeError):
            TraceRecorder("not callable")  # type: ignore[arg-type]


CANARIES = ("CANARY-CONTEXT", "CANARY-USER", "CANARY-MODEL")


class TestOnARealRunner:
    """The recorder as the runner's ``trace_sink``, into a real library."""

    def test_no_value_reaches_the_trace_file(self, valid_fsm, tmp_path):
        got: list[RunTrace] = []
        rig = _rig(
            valid_fsm,
            tmp_path,
            extraction={"answer": CANARIES[2]},
            context={"topic": CANARIES[0]},
            sink=lambda e, f: recorder(e, f),
        )
        recorder = TraceRecorder(
            lambda t: (got.append(t), rig.library.add_trace(t, max_traces=10))  # type: ignore[union-attr]
        )

        assert rig.runner.converse(rig.conv, "hello") == "What is the answer?"
        rig.runner.converse(rig.conv, CANARIES[1])

        assert rig.library is not None and len(got) == 1
        t = got[0]
        assert (t.mode, t.status) == ("converse", "ok")
        assert rig.library.get_record(t.digest).stats["runs"] == 1
        assert t.states and t.steps >= 1 and "answer" in t.keys
        found, skipped = rig.library.read_traces(t.digest)
        assert found == (t,) and skipped == 0
        blob = "".join(p.read_text() for p in rig.library.root.rglob("*.jsonl"))
        assert blob and not any(c in blob for c in CANARIES)

    def test_an_aborted_run_names_its_stuck_state(self, valid_fsm):
        from fsm_llm.synthesis import SynthesisPolicy

        got: list[RunTrace] = []
        rig = _rig(
            valid_fsm,
            extraction={},
            policy=SynthesisPolicy(max_turns=1),
            sink=lambda e, f: recorder(e, f),
        )
        recorder = TraceRecorder(got.append)
        rig.runner.converse(rig.conv, "hello")
        rig.runner.converse(rig.conv, "one")

        (t,) = got
        assert (t.status, t.reason, t.stuck) == ("aborted", "budget_spent", "ask")
        assert t.outcome == "blocked" and isinstance(rig.cheap, ScriptedRunnerLLM)


def _trial(**kw: Any) -> Any:
    base: dict[str, Any] = {
        "passed": True,
        "error": None,
        "final_state": "done",
        "visited_states": ["ask", "done"],
        "context": {"answer": "CANARY-CONTEXT", "invented one": 1, "_x": 2},
        "ended": True,
        "turns_sent": 2,
    }
    return SimpleNamespace(**{**base, **kw})


class TestFromTrial:
    def _make(self, trial: Any) -> RunTrace:
        return trace_from_trial(
            trial, D1, scenario="correction", declared=("answer", "topic")
        )

    def test_a_passing_trial(self):
        t = self._make(_trial())
        assert (t.mode, t.scenario, t.status, t.reason) == (
            "sim",
            "correction",
            "ok",
            None,
        )
        assert (t.states, t.steps, t.stuck, t.digest) == (("ask", "done"), 2, None, D1)

    def test_only_declared_names_are_kept_never_a_value(self):
        t = self._make(_trial())
        assert t.keys == ("answer",) and t.kinds == ()
        assert "CANARY" not in repr(t)

    def test_not_ended_is_stuck_in_the_final_state(self):
        t = self._make(_trial(passed=False, ended=False, final_state="ask"))
        assert (t.status, t.stuck, t.reason) == ("failed", "ask", "check_failed")

    def test_an_error_and_an_abort(self):
        err = self._make(_trial(passed=False, error="boom CANARY", ended=False))
        assert (err.status, err.reason) == ("failed", "error")
        assert "CANARY" not in repr(err)
        abort = self._make(_trial(passed=False, final_state="abort"))
        assert (abort.status, abort.reason) == ("aborted", "budget_spent")
        flagged = self._make(
            _trial(passed=False, context={"procedure_abort": True}, ended=False)
        )
        assert flagged.status == "aborted"
