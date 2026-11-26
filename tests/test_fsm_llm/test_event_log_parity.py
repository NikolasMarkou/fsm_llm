"""The extended parity matrix of the event log (plan 3f4530d3 step 9; roadmap rows 1.1, 1.3, 1.18).

The later flip of the log default is one line only if a logged run is, to a caller, the run it
would have been without the log. This file proves that for the five product paths no earlier
parity row covers together (a registry tool in a handler, a ``ReactAgent`` run with a tool, the
greeting, an approval that is granted and then denied, a ``call`` run), through the SYNC entry
points (no event loop) and the ASYNC ones (lane threads write the boundaries), with the log off,
at privacy ``values`` and at ``shape``. The cells live in ``event_log_parity_cells.py`` and run on a
REAL ``LiteLLMInterface`` whose provider is scripted, so the bytes compared are the bytes a host
would send. Per cell the rows pin:

1. log OFF equals the golden ``fixtures/event_log_parity_golden.json``, written by the cells module
   on a ``git archive`` of the plan base 7b1206e3 (before any producer existed): what the product
   did before the log writes anything, byte for byte;
2. log ON (``values`` and ``shape``) equals log OFF: replies, ``get_data``, history, the saved
   snapshot, the usage meter and the provider request kwargs, and the logged run holds the event
   types its path must write;
3. the rebuild: the stored prompt blob of EVERY ``llm/request`` equals the bytes the provider
   received, in order, the greeting included (no slicing), on both paths;
4. the sync log and the async log of one cell hold the same event types in the same order, and
   the same prompt bytes (a lane hop moves no byte, no event);
5. the final produced set is exactly the types of the plan, and the one declared type with no
   producer is ``compaction/epoch`` (roadmap row 2.5), so a future type cannot slip in unseen.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from fsm_llm.llm.observer import OBSERVED_CALL_KEYS
from fsm_llm.runtime import DECLARED_TYPES, PRODUCED_TYPES, EventLog, canonical_json
from fsm_llm.runtime.stores.jsonl import JsonlWriter
from tests.test_fsm_llm import event_log_parity_cells as cells
from tests.test_fsm_llm.event_log_support import tree_bytes, tree_files

LEVELS = ("off", "values", "shape")
GOLDEN = json.loads(cells.GOLDEN_PATH.read_text(encoding="utf-8"))
#: the event types each cell's logged run must hold, besides the turn plumbing
_MUST = {
    "greeting": {"llm/request", "llm/response", "turn/start", "turn/end", "stack/push",
                 "stack/pop", "context/record", "transition"},
    "tool": {"tool/call", "tool/result", "handler", "llm/request", "turn/end"},
    "react": {"tool/call", "tool/result", "extraction", "handler", "llm/request"},
    "approval": {"approval/request", "approval/outcome", "tool/call", "tool/result",
                 "extraction", "handler"},
    "call": {"executable/resolve", "stack/push", "stack/pop", "handler", "llm/request"},
}  # fmt: skip
#: the types the product writes; the one declared type nothing writes yet
PRODUCED_FINAL = frozenset(
    {
        "turn/start", "turn/end", "llm/request", "llm/response", "provider/error",
        "tool/call", "tool/result", "approval/request", "approval/outcome", "transition",
        "decision", "executor/start", "executor/result", "stack/push", "stack/pop",
        "context/record", "extraction", "extraction/miss", "intent/out_of_scope",
        "handler", "executable/resolve", "run/start", "run/end",
    }
)  # fmt: skip
NOT_YET_PRODUCED_FINAL = frozenset({"compaction/epoch"})


@dataclass
class Got:
    """One cell run at one level: the run, its events per root id and the bytes of its log."""

    run: cells.Run
    events: dict[str, list[dict[str, Any]]]
    prompts: list[bytes]
    tree: bytes
    files: list[Path]

    @property
    def flat(self) -> list[dict[str, Any]]:
        """Every event, the root of the cell first (``seq`` orders one conversation only)."""
        first = [self.run.cid] if self.run.cid in self.events else []
        ids = first + [i for i in sorted(self.events) if i not in first]
        return [e for i in ids for e in self.events[i]]

    @property
    def types(self) -> list[tuple[str, int]]:
        return [(e["type"], e["frame"]) for e in self.flat]


def _got(run: cells.Run, log: EventLog | None) -> Got:
    if log is None:
        return Got(run, {}, [], b"", [])
    events = {cid: list(log.read(cid)) for cid in log.conversation_ids()}
    ordered = Got(run, events, [], b"", [])
    prompts = [
        log.get_blob(e["prompt_ref"])
        for e in ordered.flat
        if e["type"] == "llm/request" and "prompt_ref" in e
    ]
    return Got(run, events, prompts, tree_bytes(log.root), tree_files(log.root))


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> dict[tuple[str, str, str], Got]:
    """Every (cell, mode, level) run once; a run is deterministic, so rows read this table."""
    table: dict[tuple[str, str, str], Got] = {}
    root = tmp_path_factory.mktemp("parity")
    for name, cell in cells.CELLS.items():
        for mode, use_async in cells.MODES.items():
            for level in LEVELS:
                work = root / f"{name}-{mode}-{level}"
                (work / "store").mkdir(parents=True)
                if level == "off":
                    table[name, mode, level] = _got(
                        cell(None, use_async, work / "store"), None
                    )
                    continue
                with EventLog(work / "log", privacy=level) as log:
                    table[name, mode, level] = _got(
                        cell(log, use_async, work / "store"), log
                    )
    return table


def _scrubbed(received: list[dict[str, Any]]) -> list[Any]:
    return [cells.scrub(kwargs) for kwargs in received]


def _every() -> list[Any]:
    return [pytest.param(n, m, id=f"{n}-{m}") for n in cells.CELLS for m in cells.MODES]


class TestLogOffIsTheParentGolden:
    def test_the_golden_holds_every_cell_in_both_modes(self) -> None:
        assert set(GOLDEN) == {*cells.CELLS, cells.SWEEP_VIEWS}
        assert all(set(GOLDEN[name]) == set(cells.MODES) for name in GOLDEN)

    @pytest.mark.parametrize(("name", "mode"), _every())
    def test_a_run_with_the_log_off_is_what_the_product_did_before_any_producer(
        self, runs: dict[tuple[str, str, str], Got], name: str, mode: str
    ) -> None:
        off = runs[name, mode, "off"].run.outcome
        want = GOLDEN[name][mode]
        assert list(off) == list(want) or set(off) == set(want)
        for key in want:
            assert off[key] == want[key], (
                f"{name}/{mode}: {key} differs from the parent"
            )

    @pytest.mark.parametrize(("name", "mode"), _every())
    def test_each_cell_really_reaches_the_provider(
        self, runs: dict[tuple[str, str, str], Got], name: str, mode: str
    ) -> None:
        run = runs[name, mode, "off"].run
        assert len(run.received) == len(run.outcome["requests"]) >= 3
        assert run.outcome["usage"]["calls"] == len(run.received)


class TestLogOnEqualsLogOff:
    @pytest.mark.parametrize("level", ["values", "shape"])
    @pytest.mark.parametrize(("name", "mode"), _every())
    def test_replies_data_history_snapshot_usage_and_provider_bytes_are_equal(
        self, runs: dict[tuple[str, str, str], Got], name: str, mode: str, level: str
    ) -> None:
        off, on = runs[name, mode, "off"].run, runs[name, mode, level].run
        assert on.outcome == off.outcome
        # every kwarg of every request, not a digest (an agent prompt names the wall clock of its
        # tool trace, so the two runs are compared with clocks and ids scrubbed)
        assert _scrubbed(on.received) == _scrubbed(off.received)

    @pytest.mark.parametrize("level", ["values", "shape"])
    @pytest.mark.parametrize(("name", "mode"), _every())
    def test_the_logged_run_holds_the_events_its_path_must_write(
        self, runs: dict[tuple[str, str, str], Got], name: str, mode: str, level: str
    ) -> None:
        got = runs[name, mode, level]
        seen = {e["type"] for e in got.flat}
        assert _MUST[name] <= seen, (name, sorted(_MUST[name] - seen))
        assert seen <= {t.value for t in PRODUCED_TYPES}
        assert [
            e["seq"] for e in got.events[got.run.cid or sorted(got.events)[0]]
        ] == list(range(1, len(got.events[got.run.cid or sorted(got.events)[0]]) + 1))

    def test_the_comparison_can_fail(
        self, runs: dict[tuple[str, str, str], Got]
    ) -> None:
        off = runs["tool", "sync", "off"].run
        on = runs["tool", "sync", "values"].run
        changed = {**on.outcome, "requests": on.outcome["requests"][:-1]}
        assert changed != off.outcome  # a dropped request
        bytes_moved = [dict(k, temperature=0.123) for k in on.received]
        assert _scrubbed(bytes_moved) != _scrubbed(off.received)  # one changed kwarg
        assert on.outcome == off.outcome  # and the real pair
        assert _scrubbed(on.received) == _scrubbed(off.received)


class TestTheRebuild:
    """Any past prompt rebuilds byte-identical: the stored blob IS the bytes the provider got."""

    @pytest.mark.parametrize(("name", "mode"), _every())
    def test_every_request_blob_equals_the_provider_bytes_in_order(
        self, runs: dict[tuple[str, str, str], Got], name: str, mode: str
    ) -> None:
        got = runs[name, mode, "values"]
        sent = got.run.received
        requests = [e for e in got.flat if e["type"] == "llm/request"]
        # ALL calls, the greeting first: no slicing anywhere, and not vacuously
        assert len(requests) == len(got.prompts) == len(sent) >= 3
        for event, blob, kwargs in zip(requests, got.prompts, sent, strict=True):
            expected = canonical_json(
                {k: kwargs[k] for k in OBSERVED_CALL_KEYS if k in kwargs}
            )
            assert blob == expected, (name, mode, event["call_type"], event["seq"])
            assert (event["source"], event["model"]) == ("provider", "gpt-4o")
        responses = [
            e for e in got.flat if e["type"] in {"llm/response", "provider/error"}
        ]
        assert sorted(e["request_seq"] for e in responses) == sorted(
            e["seq"] for e in requests
        )  # every request has its one outcome

    def test_the_greeting_is_the_first_request_of_its_conversation(
        self, runs: dict[tuple[str, str, str], Got]
    ) -> None:
        for mode in cells.MODES:
            got = runs["greeting", mode, "values"]
            root = got.events[got.run.cid]
            first = next(e for e in root if e["type"] == "llm/request")
            assert first["frame"] == 0
            assert first["seq"] < next(
                e["seq"] for e in root if e["type"] == "turn/start"
            )
            # a pushed child's greeting is a request under its own frame, before its first turn
            child = next(
                e for e in root if e["type"] == "llm/request" and e["frame"] == 1
            )
            assert child["seq"] < next(
                e["seq"] for e in root if e["type"] == "turn/start" and e["frame"] == 1
            )
            silent = [i for i in got.events if i != got.run.cid]
            assert len(silent) == 1  # the silent start made no greeting request
            quiet = got.events[silent[0]]
            assert next(e for e in quiet if e["type"] == "llm/request")["seq"] > next(
                e["seq"] for e in quiet if e["type"] == "turn/start"
            )

    def test_a_tool_turn_and_an_agent_turn_are_rebuilt_too(
        self, runs: dict[tuple[str, str, str], Got]
    ) -> None:
        for name in ("tool", "react", "approval"):
            for mode in cells.MODES:
                got = runs[name, mode, "values"]
                kinds = {e["call_type"] for e in got.flat if e["type"] == "llm/request"}
                assert "response_generation" in kinds, (name, mode)
        react = runs["react", "sync", "values"]
        assert {"field_extraction"} <= {
            e["call_type"] for e in react.flat if e["type"] == "llm/request"
        }

    def test_the_comparison_can_fail(
        self, runs: dict[tuple[str, str, str], Got]
    ) -> None:
        got = runs["tool", "async", "values"]
        kwargs = dict(got.run.received[0], temperature=0.123)
        wrong = canonical_json(
            {k: kwargs[k] for k in OBSERVED_CALL_KEYS if k in kwargs}
        )
        assert wrong != got.prompts[0]


class TestSyncAndAsyncWriteTheSameLog:
    @pytest.mark.parametrize("level", ["values", "shape"])
    @pytest.mark.parametrize("name", list(cells.CELLS))
    def test_the_lane_hop_moves_no_event_and_no_prompt_byte(
        self, runs: dict[tuple[str, str, str], Got], name: str, level: str
    ) -> None:
        sync, aync = runs[name, "sync", level], runs[name, "async", level]
        assert sync.types == aync.types
        # prompt bytes: an agent prompt names the wall clock of its tool trace, so the blobs are
        # read back and compared with clocks scrubbed (each blob is exact against its own run in
        # the rebuild rows above)
        assert [cells.scrub(json.loads(b)) for b in sync.prompts] == [
            cells.scrub(json.loads(b)) for b in aync.prompts
        ]
        # the fields two runs never share: a clock, a random call id, the blobs that hold a
        # provider reply, a prompt or a context with a clock (their NAMES are still compared)
        volatile = {"ts", "elapsed_ms", "call_id", "conversation_id"}

        def stable(event: dict[str, Any]) -> dict[str, Any]:
            return {
                k: "<ref>" if k.endswith("_ref") else cells.scrub(v)
                for k, v in event.items()
                if k not in volatile
            }

        assert [stable(e) for e in sync.flat] == [stable(e) for e in aync.flat]


class TestPrivacyLevels:
    @pytest.mark.parametrize(("name", "mode"), _every())
    def test_at_shape_no_blob_and_no_reference_is_written(
        self, runs: dict[tuple[str, str, str], Got], name: str, mode: str
    ) -> None:
        got = runs[name, mode, "shape"]
        assert not any("blobs" in p.parts for p in got.files)
        assert [k for e in got.flat for k in e if k.endswith("_ref")] == []
        assert got.prompts == []

    @pytest.mark.parametrize(("name", "mode"), _every())
    def test_at_values_the_prompt_blobs_exist_and_no_grant_is_anywhere(
        self, runs: dict[tuple[str, str, str], Got], name: str, mode: str
    ) -> None:
        got = runs[name, mode, "values"]
        assert any("blobs" in p.parts for p in got.files)
        for level in ("values", "shape"):
            logged = runs[name, mode, level]
            assert b"sk-parity" not in logged.tree, (
                name,
                mode,
                level,
            )  # the key never lands
            for path in (
                logged.files
            ):  # a blob never holds the grant key; an approval line neither
                text = path.read_bytes().decode("utf-8", "replace")
                if path.suffix != ".jsonl":
                    assert "_approval_granted" not in text, (
                        name,
                        mode,
                        level,
                        path.name,
                    )
            for event in logged.flat:
                if event["type"].startswith("approval/"):
                    assert "_approval_granted" not in json.dumps(event), (
                        name,
                        mode,
                        level,
                    )


class TestTheApprovalCell:
    @pytest.mark.parametrize("level", ["values", "shape"])
    @pytest.mark.parametrize("mode", list(cells.MODES))
    def test_a_granted_then_a_denied_ask_each_write_a_pair_and_change_no_verdict(
        self, runs: dict[tuple[str, str, str], Got], mode: str, level: str
    ) -> None:
        off, on = runs["approval", mode, "off"].run, runs["approval", mode, level].run
        assert [a["tool_name"] for a in off.outcome["asks"]] == ["danger", "danger"]
        assert off.outcome["runs"] == [
            "danger"
        ]  # the tool ran once: the denied ask ran nothing
        assert (
            on.outcome["asks"] == off.outcome["asks"]
        )  # the approver saw the same RAW request
        events = [
            e
            for e in runs["approval", mode, level].flat
            if e["type"].startswith("approval/")
        ]
        assert [e["type"] for e in events] == [
            "approval/request",
            "approval/outcome",
        ] * 2
        assert [e["approved"] for e in events if e["type"] == "approval/outcome"] == [
            True,
            False,
        ]
        assert [e["reason"] for e in events if e["type"] == "approval/outcome"] == [
            "approved",
            "denied",
        ]
        tools = [
            e for e in runs["approval", mode, level].flat if e["type"] == "tool/call"
        ]
        assert (
            len(tools) == 1 and tools[0]["gated"] is True
        )  # the denied call is no tool/call


#: the table bounds that must change nothing: unbounded (the writer of before) and 0 (every append
#: closes its writer, so EVERY append re-opens the log); the default bound is the ``runs`` table
BOUNDS = (None, 0)


@dataclass
class Bounded:
    """One cell run on a bounded log: the run, the writers opened and the writers left open."""

    got: Got
    opens: int
    left: int


@pytest.fixture(scope="module")
def bounded_runs(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[tuple[str, str, str, int | None], Bounded]:
    """Every (cell, mode, level, bound) run once, at the two logged levels."""
    table: dict[tuple[str, str, str, int | None], Bounded] = {}
    root = tmp_path_factory.mktemp("bounded-parity")
    real = JsonlWriter._open
    for bound in BOUNDS:
        for name, cell in cells.CELLS.items():
            for mode, use_async in cells.MODES.items():
                for level in ("values", "shape"):
                    work = root / f"{name}-{mode}-{level}-{bound}"
                    (work / "store").mkdir(parents=True)
                    opened: list[int] = []

                    def counting(self: JsonlWriter, _seen: list[int] = opened) -> None:
                        _seen.append(1)
                        real(self)

                    with (
                        mock.patch.object(JsonlWriter, "_open", counting),
                        EventLog(
                            work / "log", privacy=level, max_open_writers=bound
                        ) as log,
                    ):
                        run = cell(log, use_async, work / "store")
                        left = log.open_writer_count()
                        got = _got(run, log)
                    table[name, mode, level, bound] = Bounded(got, len(opened), left)
    return table


#: the fields two runs never share (a clock, a random call id) and the blobs that hold them
_VOLATILE = {"ts", "elapsed_ms", "call_id", "conversation_id"}


def _stable_event(event: dict[str, Any]) -> dict[str, Any]:
    return {
        k: "<ref>" if k.endswith("_ref") else cells.scrub(v)
        for k, v in event.items()
        if k not in _VOLATILE
    }


class TestEveryAppendMayReopen:
    """G1 (plan 77e5fef9): the bound changes nothing a caller or a reader can see. With the
    table unbounded and with bound 0 (every append closes and re-opens its writer) a cell gives
    the replies, data, history, snapshot, usage and provider bytes of the log-off golden and
    writes the log the default bound writes, event for event."""

    @pytest.mark.parametrize("bound", BOUNDS)
    @pytest.mark.parametrize("level", ["values", "shape"])
    @pytest.mark.parametrize(("name", "mode"), _every())
    def test_a_bounded_run_is_the_run_of_the_golden_and_the_log_of_the_default(
        self,
        runs: dict[tuple[str, str, str], Got],
        bounded_runs: dict[tuple[str, str, str, int | None], Bounded],
        name: str,
        mode: str,
        level: str,
        bound: int | None,
    ) -> None:
        got = bounded_runs[name, mode, level, bound].got
        off, default = runs[name, mode, "off"].run, runs[name, mode, level]
        assert got.run.outcome == off.outcome
        assert _scrubbed(got.run.received) == _scrubbed(off.received)
        assert got.types == default.types
        assert [_stable_event(e) for e in got.flat] == [
            _stable_event(e) for e in default.flat
        ]
        assert [cells.scrub(json.loads(b)) for b in got.prompts] == [
            cells.scrub(json.loads(b)) for b in default.prompts
        ]
        for events in got.events.values():
            assert [e["seq"] for e in events] == list(range(1, len(events) + 1))

    @pytest.mark.parametrize("level", ["values", "shape"])
    @pytest.mark.parametrize(("name", "mode"), _every())
    def test_bound_zero_reopens_every_append_and_none_opens_once_per_conversation(
        self,
        bounded_runs: dict[tuple[str, str, str, int | None], Bounded],
        name: str,
        mode: str,
        level: str,
    ) -> None:
        """Kills: 0 read as unbounded, a bound ignored (non-vacuity of the whole class)."""
        unbounded = bounded_runs[name, mode, level, None]
        every = bounded_runs[name, mode, level, 0]
        conversations = len(every.got.events)
        assert unbounded.opens == conversations  # one open per tree, never a re-open
        assert every.left == 0
        assert (
            every.opens > 3 * conversations
        )  # many appends per conversation, each opens

    def test_the_comparison_can_fail(
        self,
        runs: dict[tuple[str, str, str], Got],
        bounded_runs: dict[tuple[str, str, str, int | None], Bounded],
    ) -> None:
        got = bounded_runs["tool", "sync", "values", 0].got
        default = runs["tool", "sync", "values"]
        dropped = [_stable_event(e) for e in got.flat][:-1]
        assert dropped != [_stable_event(e) for e in default.flat]
        reset = [
            dict(_stable_event(e), seq=1) for e in got.flat
        ]  # a re-open restarting seq
        assert reset != [_stable_event(e) for e in default.flat]
        assert [_stable_event(e) for e in got.flat] == [
            _stable_event(e) for e in default.flat
        ]


class TestTheFinalProducedSet:
    def test_the_product_writes_exactly_these_types_and_nothing_else_is_declared_but_one(
        self,
    ) -> None:
        assert {t.value for t in PRODUCED_TYPES} == PRODUCED_FINAL
        assert {
            t.value for t in DECLARED_TYPES - PRODUCED_TYPES
        } == NOT_YET_PRODUCED_FINAL
        assert {
            t.value for t in DECLARED_TYPES
        } == PRODUCED_FINAL | NOT_YET_PRODUCED_FINAL

    def test_the_cells_together_write_every_produced_type_the_matrix_can_reach(
        self, runs: dict[tuple[str, str, str], Got]
    ) -> None:
        seen = {e["type"] for got in runs.values() for e in got.flat}
        # what only another scenario reaches: a provider failure, a spent budget, a delegated
        # run, a miss and an out-of-scope intent (test_event_log_producers.py holds each)
        elsewhere = {
            "provider/error", "decision", "run/start", "run/end", "extraction/miss",
            "intent/out_of_scope",
        }  # fmt: skip
        assert seen == PRODUCED_FINAL - elsewhere


class TestTheSweepsCoverTheTypes:
    """A type the product writes is killed at every offset by at least one sweep, so a new
    producer cannot join ``PRODUCED_TYPES`` without a sweep that dies inside it (the two
    types left out below, ``run/start`` and ``run/end``, are written by a delegated run of a ``call``
    state: its worker is a thread outside the engine, and no per-offset kill sweep exists for it yet,
    a known gap recorded in the plan notes)."""

    UNSWEPT = frozenset({"run/start", "run/end"})

    def test_the_union_of_the_per_offset_sweeps_holds_every_other_produced_type(
        self, tmp_path: Path
    ) -> None:
        from tests.test_fsm_llm import test_call_log as call_sweep
        from tests.test_fsm_llm import test_event_log_crash as crash
        from tests.test_fsm_llm import test_event_log_stack_crash as stack_sweep

        swept: dict[str, set[str]] = {}
        for index, scenario in enumerate(
            (
                crash.TOOLS,
                crash.CHAT,
                crash.PRODUCTS,
                crash.PRODUCTS_ASYNC,
                crash.BUDGET,
            )
        ):
            golden = crash._golden(scenario, tmp_path / f"c{index}")
            swept[scenario.name] = {e["type"] for e in golden.events}
        swept["stack"] = {
            e["type"] for e in stack_sweep._run_golden(tmp_path / "stack").events
        }
        swept["call"] = {
            e["type"] for e in call_sweep._run_golden(tmp_path / "call").events
        }
        union = set().union(*swept.values())
        assert union == PRODUCED_FINAL - self.UNSWEPT, sorted(
            PRODUCED_FINAL - self.UNSWEPT ^ union
        )
        # the async sweep holds the same types as the sync one: the lane hop adds none
        sync, aync = (
            swept[n] for n in (crash.PRODUCTS.name, crash.PRODUCTS_ASYNC.name)
        )
        assert sync == aync
        assert {"executable/resolve"} <= swept["call"]
