"""The signals of a child frame on its run trace (step 8 of plan c720601d, decision D-012).

Real rows drive PRODUCT code (``make_rig`` runs of a real ``API`` and ``SynthesizeExecutor`` with
the event log on at both privacy levels); the pure rows hand-build the events and rows a real run
cannot be made to write. What is pinned: the log ON gives a trace that carries the failure codes
and the clean-window evidence; the log OFF gives the row the trace always was (no new key); an old
row loads and re-saves unchanged; ``RunTrace`` stays value-free; a lost reader (a restart, a fork)
gives empty signals and no error; the signals are read once per run, from the position taken at
the push.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar
from unittest import mock

import pytest

from fsm_llm.core.codes import LlmFailureCode
from fsm_llm.core.definitions import LLMResponseError
from fsm_llm.engine.kinds.requests import ChildView
from fsm_llm.runtime import EventLog
from fsm_llm.runtime.signals import FrameSignals, frame_signals
from fsm_llm.synthesis import RunTrace, SynthesisPolicy, TraceRecorder
from fsm_llm.synthesis import traces as traces_module
from fsm_llm.synthesis.traces import signal_fields
from tests.test_fsm_llm_synthesis.conftest import ScriptedRunnerLLM, good_fsm
from tests.test_fsm_llm_synthesis.rig import REPLIES, Rig, make_rig, named_error

LEVELS = ("values", "shape")
NEW = ("failures", "misses", "scopes", "bad_extractions")
D1 = "a" * 64
CANARY = "CANARY-7f3a91"


class _Down(ScriptedRunnerLLM):
    """A cheap model whose every field extraction fails with a provider error of a named class."""

    def __init__(self, name: str) -> None:
        super().__init__({}, dict(REPLIES))
        self.name = name

    def extract_field(self, request: Any) -> Any:
        raise LLMResponseError("provider down") from named_error(self.name)


def _run(
    tmp_path: Any, cheap: ScriptedRunnerLLM, level: str | None, *, hook: Any = None
) -> tuple[list[RunTrace], Rig]:
    """One synthesized child run to its end; the traces the recorder folded."""
    traces: list[RunTrace] = []
    recorder = TraceRecorder(traces.append)

    def sink(event: str, fields: dict[str, Any]) -> None:
        if hook is not None:
            hook(event)
        recorder(event, fields)

    log = None if level is None else EventLog(tmp_path / "ev", privacy=level)
    rig = make_rig(
        good_fsm(),
        cheap=cheap,
        policy=SynthesisPolicy(max_turns=3),
        sink=sink,
        event_log=log,
    )
    try:
        rig.run(30)
    finally:
        if log is not None:
            log.close()
    return traces, rig


@pytest.mark.parametrize("level", LEVELS)
class TestALoggedRunLeavesItsSignalsOnTheTrace:
    @pytest.mark.parametrize(
        ("name", "code"),
        [
            ("RateLimitError", LlmFailureCode.RATE_LIMITED),
            ("Timeout", LlmFailureCode.TIMEOUT),
            ("APIConnectionError", LlmFailureCode.TRANSPORT),
            ("AuthenticationError", LlmFailureCode.AUTH),
        ],
    )
    def test_an_outage_carries_its_failure_code_and_no_evidence(
        self, tmp_path: Any, level: str, name: str, code: LlmFailureCode
    ):
        traces, _ = _run(tmp_path, _Down(name), level)

        (trace,) = traces
        assert trace.status == "aborted" and trace.reason == "provider_error"
        assert [c for c, _ in trace.failures] == [code.value]
        assert trace.failures[0][1] >= 3
        assert (trace.misses, trace.scopes, trace.bad_extractions) == ((), (), ())

    def test_a_stuck_run_carries_the_clean_miss_and_no_failure(
        self, tmp_path: Any, level: str
    ):
        traces, _ = _run(tmp_path, ScriptedRunnerLLM({}, dict(REPLIES)), level)

        (trace,) = traces
        assert trace.reason == "budget_spent" and trace.stuck == "ask"
        assert trace.misses == (("ask", "answer", 3),)
        assert (trace.failures, trace.scopes, trace.bad_extractions) == ((), (), ())

    def test_the_row_of_a_logged_run_differs_from_the_unlogged_one_by_the_new_keys(
        self, tmp_path: Any, level: str
    ):
        on, _ = _run(tmp_path, ScriptedRunnerLLM({}, dict(REPLIES)), level)
        off, _ = _run(tmp_path, ScriptedRunnerLLM({}, dict(REPLIES)), None)

        row_on, row_off = on[0].to_row(), off[0].to_row()
        assert set(row_on) - set(row_off) == {"misses"}
        assert {k: v for k, v in row_on.items() if k != "misses"} == row_off

    def test_the_canary_of_a_user_value_is_in_no_row(self, tmp_path: Any, level: str):
        cheap = ScriptedRunnerLLM(
            {"town": CANARY}, dict(REPLIES)
        )  # a wrong key: a miss

        traces, rig = _run(tmp_path, cheap, level)

        assert traces[0].misses == (("ask", "answer", 3),)
        assert CANARY not in json.dumps(traces[0].to_row())
        assert CANARY not in repr(rig.stream)


class TestALogThatIsOffChangesNothing:
    def test_a_run_with_the_log_off_has_no_signal_key_in_its_row(self, tmp_path: Any):
        for cheap in (
            ScriptedRunnerLLM({"answer": "42"}, dict(REPLIES)),
            ScriptedRunnerLLM({}, dict(REPLIES)),
            _Down("RateLimitError"),
        ):
            traces, rig = _run(tmp_path, cheap, None)

            assert not set(NEW) & set(traces[0].to_row())
            assert not [f for e, f in rig.stream if set(NEW) & set(f)]

    def test_a_run_with_the_log_on_but_nothing_to_report_writes_no_key(
        self, tmp_path: Any
    ):
        traces, _ = _run(
            tmp_path, ScriptedRunnerLLM({"answer": "42"}, REPLIES), "values"
        )

        assert traces[0].status == "ok"
        assert not set(NEW) & set(traces[0].to_row())


class TestARowOfBeforeStillLoads:
    OLD: ClassVar[dict[str, Any]] = {
        "v": 1,
        "digest": D1,
        "version": None,
        "mode": "run",
        "states": ["ask"],
        "steps": 3,
        "keys": ["answer"],
        "kinds": [["answer", "str"]],
        "status": "aborted",
        "reason": "budget_spent",
        "stuck": "ask",
        "outcome": "blocked",
    }

    def test_an_old_row_loads_with_empty_signals_and_saves_unchanged(self):
        trace = RunTrace.from_row(dict(self.OLD))

        assert [getattr(trace, name) for name in NEW] == [(), (), (), ()]
        assert trace.to_row() == self.OLD

    def test_a_row_with_signals_round_trips_and_the_version_stays_one(self):
        trace = RunTrace(
            digest=D1,
            failures=(("timeout", 2),),
            misses=(("ask", "answer", 3),),
            scopes=(("ask", "intent", 1),),
            bad_extractions=(("ask", 4),),
        )
        row = json.loads(json.dumps(trace.to_row()))

        assert row["v"] == traces_module.TRACE_VERSION == 1
        assert RunTrace.from_row(row) == trace


class TestARunTraceStaysValueFree:
    @pytest.mark.parametrize(
        ("field", "entry"),
        [
            ("failures", ("rate limited", 1)),  # a name with a space
            ("failures", (CANARY + " text", 1)),
            ("failures", (3, 1)),  # not a name
            ("misses", ("ask", "an answer", 1)),
            ("misses", ("ask", "answer", -1)),  # a negative count
            ("scopes", ("ask", "intent")),  # a missing count
            ("scopes", ("ask", "intent", 1.5)),
            ("scopes", ("ask", "intent", True)),
            ("bad_extractions", ("ask", "x", 1)),  # a wrong width
            ("bad_extractions", ("a:b", 1)),  # the wire separator
            ("bad_extractions", ("x" * 81, 1)),  # an oversized name
        ],
    )
    def test_a_value_bearing_or_malformed_entry_is_refused(
        self, field: str, entry: tuple[Any, ...]
    ):
        with pytest.raises(ValueError):
            RunTrace(digest=D1, **{field: (entry,)})
        with pytest.raises(ValueError):
            RunTrace.from_row({"v": 1, "digest": D1, field: [list(entry)]})

    def test_a_field_that_is_not_a_sequence_is_refused(self):
        with pytest.raises(ValueError):
            RunTrace(digest=D1, misses="ask:answer:1")  # type: ignore[arg-type]

    def test_an_oversized_field_is_capped(self):
        trace = RunTrace(
            digest=D1, bad_extractions=tuple((f"s{i}", 1) for i in range(1000))
        )

        assert len(trace.bad_extractions) == 64

    def test_a_code_outside_the_closed_set_is_unknown(self):
        trace = RunTrace(digest=D1, failures=(("from_a_newer_version", 2),))

        assert trace.failures == (("unknown", 2),)

    def test_every_member_of_the_closed_set_is_kept_as_it_is(self):
        entries = tuple((c.value, 1) for c in LlmFailureCode)

        assert RunTrace(digest=D1, failures=entries).failures == entries


class TestTheWireOfTheSignals:
    def test_the_signals_become_scalars_and_come_back_equal(self):
        signals = FrameSignals(
            failures=(("rate_limited", 2),),
            misses=(("ask", "answer", 3),),
            out_of_scope=(("ask", "intent", 1),),
            extraction_failed=(("ask", 2), ("wait", 1)),
            cut=(("ask", 1),),
            errors=2,
        )
        fields = signal_fields(signals)
        assert all(isinstance(v, str) for v in fields.values())
        assert (
            fields["bad_extractions"] == "ask:3,wait:1"
        )  # failed and cut add up per state

        recorder_rows: list[RunTrace] = []
        recorder = TraceRecorder(recorder_rows.append)
        recorder("pushed", {"conv": "c", "digest": D1, "step": 0})
        recorder("aborted", {"conv": "c", "digest": D1, "state": "ask", **fields})
        recorder("popped", {"conv": "c", "digest": D1, "status": "aborted"})

        (trace,) = recorder_rows
        assert trace.failures == (("rate_limited", 2),)
        assert trace.misses == (("ask", "answer", 3),)
        assert trace.scopes == (("ask", "intent", 1),)
        assert trace.bad_extractions == (("ask", 3), ("wait", 1))

    def test_empty_signals_give_no_field(self):
        assert signal_fields(FrameSignals()) == {}

    @pytest.mark.parametrize(
        "text",
        [
            "a b:3",
            "x:y:z:q",
            "ask:-1",
            "ask:" + "9" * 5000,
            ":",
            ",,",
            "ask",
            "a:b:c:d:1",
        ],
    )
    def test_a_hostile_field_is_dropped_and_never_loses_the_run(self, text: str):
        rows: list[RunTrace] = []
        recorder = TraceRecorder(rows.append)
        recorder("pushed", {"conv": "c", "digest": D1, "step": 0})
        recorder(
            "aborted",
            {
                "conv": "c",
                "digest": D1,
                "failures": text,
                "misses": text,
                "bad_extractions": text,
            },
        )
        recorder("popped", {"conv": "c", "digest": D1, "status": "aborted"})

        assert len(rows) == 1  # the run still ends in a trace
        assert (rows[0].failures, rows[0].misses, rows[0].bad_extractions) == (
            (),
            (),
            (),
        )

    def test_a_non_string_field_is_ignored(self):
        rows: list[RunTrace] = []
        recorder = TraceRecorder(rows.append)
        recorder("pushed", {"conv": "c", "digest": D1, "step": 0})
        recorder("stepped", {"conv": "c", "digest": D1, "step": 1, "misses": ["ask"]})
        recorder("popped", {"conv": "c", "digest": D1, "status": "ok"})

        assert rows[0].misses == ()


class TestTheReaderIsTakenAtThePushAndReadOnce:
    def test_the_signals_are_read_once_per_run_from_the_position_of_the_push(
        self, tmp_path: Any
    ):
        taken: list[Any] = []
        from fsm_llm.runtime import usage

        real = usage.watch_frame_signals

        def counting(log: Any, root: str, frame: int) -> Any:
            read = real(log, root, frame)

            def counted() -> FrameSignals:
                got = read()
                taken.append(got)
                return got

            return counted

        with mock.patch("fsm_llm.engine.api.calls.watch_frame_signals", counting):
            traces, _ = _run(tmp_path, ScriptedRunnerLLM({}, dict(REPLIES)), "values")

        assert len(taken) == 1  # not per tick: the child took three turns
        assert taken[0].misses == (("ask", "answer", 3),)
        assert traces[0].misses == (("ask", "answer", 3),)

    def test_the_events_before_the_push_are_not_the_childs(self, tmp_path: Any):
        # the parent's own turns wrote events before the push: they are in no signal
        traces, rig = _run(tmp_path, ScriptedRunnerLLM({}, dict(REPLIES)), "values")
        log = EventLog(tmp_path / "ev")
        try:
            whole = frame_signals(log.read(rig.conv), 1)
        finally:
            log.close()

        assert whole.misses == traces[0].misses


class TestALostReaderGivesEmptySignalsAndNoError:
    """A restart or a fork loses the reader (G2 of the map): the child's signals are empty and
    every rule that needs them is inert; the run itself is unchanged."""

    @pytest.mark.parametrize("level", LEVELS)
    def test_a_reader_lost_after_the_push_leaves_no_signal_and_no_error(
        self, tmp_path: Any, level: str
    ):
        holder: list[Rig] = []

        def lose(event: str) -> None:
            if event == "pushed" and holder:
                holder[0].api._call_marks.signals.clear()

        traces: list[RunTrace] = []
        recorder = TraceRecorder(traces.append)
        log = EventLog(tmp_path / "ev", privacy=level)
        rig = make_rig(
            good_fsm(),
            cheap=ScriptedRunnerLLM({}, dict(REPLIES)),
            policy=SynthesisPolicy(max_turns=3),
            sink=lambda e, f: (lose(e), recorder(e, f)),
            event_log=log,
        )
        holder.append(rig)
        try:
            rig.run(30)
        finally:
            log.close()

        (trace,) = traces
        assert trace.status == "aborted" and trace.reason == "budget_spent"
        assert [getattr(trace, name) for name in NEW] == [(), (), (), ()]
        assert not set(NEW) & set(trace.to_row())


class TestTheViewOfTheChildHoldsCopiesOnly:
    def test_the_view_carries_a_frozen_value_and_defaults_to_none(self):
        view = ChildView("c", None, "ask", {}, True, None, 1)
        assert view.signals is None
        with_signals = ChildView(
            "c", None, "ask", {}, True, None, 1, signals=FrameSignals(errors=1)
        )
        assert hash(with_signals.signals) == hash(FrameSignals(errors=1))
        with pytest.raises(AttributeError):
            with_signals.signals = None  # type: ignore[misc]
