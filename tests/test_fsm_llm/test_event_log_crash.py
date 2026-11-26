"""Crash proofs of the event log: a killed process, every death offset, the fsync policy.

Plan 21e6f00d step 11 (roadmap row 1.3). Three proofs, none timing-based (``seq`` is the order,
no test reads or asserts on ``ts``):

1. One real child process, killed with ``Popen.kill()`` while a model call is blocked: the log
   is readable, its tail is the open turn with its ``llm/request`` last, the dead holder's lock
   is free, and ``EventLogSessionStore`` restores the last committed turn with the transcript,
   the in-flight tool as ``unknown`` and nothing retried. This test is NOT marked ``slow`` or
   ``integration`` (CI deselects both), and a collection test pins that no test of this file
   carries a marker.
2. The deterministic variant every leg runs: the module-level ``jsonl._write`` is patched to
   die (a ``BaseException``) at the k-th event, whole or torn to half its bytes, and to drop
   every later write, for every k over a full scripted run of a completion-state tool exchange
   and of a Pass-2 failure that is rolled back and then killed. For every k the log is exactly
   the acknowledged prefix, the model double ran only calls whose ``llm/request`` is on disk,
   and the restored conversation is the last committed state (the pre-turn state inside an
   open turn), never a half state.
3. The product paths (plan 3f4530d3 step 9): the same sweep over a scripted run that goes through
   the producers (a registry tool, an approval, the extraction family, a handler, the greeting),
   through the sync and the async entry points (in async mode a death lands in a lane thread), and
   over a spent turn budget. The golden is the LIVE view at each boundary and equals what the parent
   archive showed with the log off (``fixtures/event_log_parity_golden.json``).
4. The fsync policy: exactly ``llm/request``, a side-effecting ``tool/call`` and ``turn/end``
   are followed by one ``fsync``, counted through the ``_fsync`` indirection.

Unverified until the Windows run: that ``Popen.kill()`` (``TerminateProcess``) on a Windows
child leaves a log the parent can read and a lock the parent can take, that the child imports
``fsm_llm`` there within the deadline (its environment is ``SYSTEMROOT`` and ``PATH`` only),
and that ``os.replace`` of the ready file is seen by a polling parent.
"""

from __future__ import annotations

import ast
import copy
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from fsm_llm import API
from fsm_llm.core.definitions import FSMError
from fsm_llm.engine.session import EventLogSessionStore
from fsm_llm.runtime import EventLog, is_checkpoint, unresolved_tool_calls
from fsm_llm.runtime.stores import JsonlWriter, fold_session, jsonl, read_events
from fsm_llm.testing import ScriptedLLM, final_reply
from tests.conftest import current_timing_factor
from tests.test_fsm_llm import event_log_child as child
from tests.test_fsm_llm import event_log_parity_cells as cells
from tests.test_fsm_llm.event_log_support import (
    CHILD_MODULE,
    _Dead,
    child_env,
    child_process,
    reopen_after_kill,
    wait_ready,
    writer_bound,  # noqa: F401  (autouse: every test runs at bound None and 0)
)
from tests.test_fsm_llm.test_event_log_jsonl import can_lock_in_child
from tests.test_fsm_llm.test_event_log_sessions import _CHAT


class _Crash:
    """Stand-in for ``jsonl._write``: *survive* writes land, the next one dies, later ones drop.

    Interface contract: ``_Crash(survive, torn=False)``; ``survive=None`` never dies (a counting
    pass). The write that dies lands nothing, or with ``torn`` its first half (a torn last
    line), then raises ``_Dead``; every write after that returns as if it had landed but writes
    nothing (the process is gone, whatever still runs is a ghost). ``writes`` counts the writes
    that landed. One write is one event in every scripted run here.
    """

    def __init__(self, survive: int | None, torn: bool = False) -> None:
        self.survive, self.torn = survive, torn
        self.writes = 0
        self.dead = False
        self._real = jsonl._write

    def __call__(self, fd: int, data: Any) -> int:
        if self.dead:
            return len(data)
        if self.survive is not None and self.writes >= self.survive:
            self.dead = True
            if self.torn:
                self._real(fd, data[: len(data) // 2])
            raise _Dead
        self.writes += 1
        return self._real(fd, data)


@dataclass(frozen=True)
class _Scenario:
    """A scripted run the sweep kills at every event.

    ``make_llm`` builds the double, ``play(log, llm, record)`` runs the whole conversation on a
    log and calls ``record(api, conversation_id)`` after every step that wrote events,
    ``resume(store, llm)`` builds the API that restores from a store, ``view`` reads what a
    restore must reproduce. Every model call is logged before it is made, the root greeting
    included (plan 3f4530d3 step 2 dropped ``greeting_calls``, the calls the log did not hold).
    """

    name: str
    make_llm: Callable[[], ScriptedLLM]
    play: Callable[[EventLog, Any, Callable[[API, str], None]], None]
    resume: Callable[[EventLogSessionStore, Any], API]
    view: Callable[[API, str], dict[str, Any]]


def _play_tools(log: EventLog, llm: Any, record: Callable[[API, str], None]) -> None:
    api = API.from_definition(child.tool_fsm(), llm_interface=llm, event_log=log)
    cid = child.start(api)
    record(api, cid)
    child.drive_exchange(api, log, cid, mark=lambda: record(api, cid))
    api.advance(cid)  # the model answers: call_model -> done
    record(api, cid)


TOOLS = _Scenario(
    name="a completion state with a tool exchange",
    make_llm=lambda: child.ChildLLM(child.scripted_replies()),
    play=_play_tools,
    resume=lambda store, llm: API.from_definition(
        child.tool_fsm(), llm_interface=llm, session_store=store
    ),
    view=child.view,
)


class _Flaky(ScriptedLLM):
    """Pass 2 fails while ``down``; the failed call is still a call the double received."""

    down = False

    def generate_response(self, request: Any) -> Any:
        if self.down:
            self.call_history.append(("generate_response", request))
            raise RuntimeError("provider down")
        return super().generate_response(request)


def _chat_llm() -> ScriptedLLM:
    return _Flaky(
        [],
        extraction_data={"name": "Ann"},
        responses={"intake": "hi Ann", "talk": "talking"},
    )


def _chat_view(api: API, cid: str) -> dict[str, Any]:
    snapshot = api.fsm_manager.get_conversation_snapshot(cid)
    return {
        "current_state": snapshot["current_state"],
        "data": api.get_data(cid),
        "history": snapshot["conversation_history"],
    }


def _play_chat(log: EventLog, llm: Any, record: Callable[[API, str], None]) -> None:
    api = API(copy.deepcopy(_CHAT), llm_interface=llm, event_log=log)
    cid, _ = api.start_conversation()
    record(api, cid)
    llm.down = True
    try:
        api.converse(
            "I am Ann", cid
        )  # Pass 1 commits the name, Pass 2 fails: rolled back
    except FSMError:
        pass
    else:
        raise AssertionError("the scripted Pass-2 failure did not happen")
    record(api, cid)
    llm.down = False
    api.converse("I am Ann", cid)
    record(api, cid)
    api.converse("more", cid)
    record(api, cid)


CHAT = _Scenario(
    name="a Pass-2 failure rolled back and then a crash",
    make_llm=_chat_llm,
    play=_play_chat,
    resume=lambda store, llm: API(
        copy.deepcopy(_CHAT), llm_interface=llm, session_store=store
    ),
    view=_chat_view,
)


def _play_budget(log: EventLog, llm: Any, record: Callable[[API, str], None]) -> None:
    api = API(copy.deepcopy(_CHAT), llm_interface=llm, event_log=log, max_steps=1)
    cid, _ = api.start_conversation()
    record(api, cid)
    api.converse("I am Ann", cid)
    record(api, cid)
    with pytest.raises(
        FSMError
    ):  # the spent turn budget refuses the next turn: a decision
        api.converse("more", cid)
    record(api, cid)


BUDGET = _Scenario(
    name="a spent turn budget refuses a turn and then a crash",
    make_llm=_chat_llm,
    play=_play_budget,
    resume=CHAT.resume,
    view=_chat_view,
)


def _only_log(log: EventLog) -> str:
    ids = log.conversation_ids()
    assert len(ids) == 1, ids
    return ids[0]


@dataclass(frozen=True)
class _Golden:
    """The uncrashed run: its events, and a view at each boundary keyed by writes so far."""

    events: list[dict[str, Any]]
    views: list[tuple[int, dict[str, Any]]]

    @property
    def writes(self) -> int:
        return len(self.events)

    def view_after(self, writes: int) -> dict[str, Any] | None:
        """The last whole state at a boundary of at most *writes* events, if any."""
        wanted = None
        for at, view in self.views:
            if at <= writes:
                wanted = view
        return wanted


def _golden(scenario: _Scenario, root: Path) -> _Golden:
    views: list[tuple[int, dict[str, Any]]] = []
    crash = _Crash(None)
    with EventLog(root) as log, mock.patch.object(jsonl, "_write", crash):
        scenario.play(
            log,
            scenario.make_llm(),
            lambda api, cid: views.append((crash.writes, scenario.view(api, cid))),
        )
        events = list(log.read(_only_log(log)))
    assert len(events) == crash.writes  # one write is one event
    return _Golden(events, views)


def _kill_at(scenario: _Scenario, root: Path, survive: int, torn: bool) -> Any:
    """Run the scenario until the (survive+1)-th write dies; return the double it used."""
    llm = scenario.make_llm()
    crash = _Crash(survive, torn)
    with EventLog(root) as log, mock.patch.object(jsonl, "_write", crash):
        with pytest.raises(_Dead):
            scenario.play(log, llm, lambda api, cid: None)
    assert crash.dead
    return llm


def _check_death(
    scenario: _Scenario, golden: _Golden, root: Path, survive: int, torn: bool
) -> bool:
    """The assertions for one death offset; True when a tool call was left unknown."""
    llm = _kill_at(scenario, root, survive, torn)
    reopened = EventLog(root)  # the dead writer's lock is free once its fd is closed
    try:
        cid = _only_log(reopened)
        path = root / "logs" / f"{cid}.jsonl"
        events = list(read_events(path))
        where = f"{scenario.name}, death at event {survive + 1} (torn={torn})"
        assert [e["seq"] for e in events] == list(range(1, survive + 1)), where
        assert [e["type"] for e in events] == [
            e["type"] for e in golden.events[:survive]
        ], where  # the log is exactly what was acknowledged, nothing more
        wanted = [
            e["seq"] for e in golden.events[:survive] if is_checkpoint(e["type"], e)
        ]
        assert wanted == [e["seq"] for e in events if is_checkpoint(e["type"], e)], (
            where
        )
        asked = sum(e["type"] == "llm/request" for e in events)
        # a model call ran only after its llm/request was on disk, and every durable request ran
        assert len(llm.call_history) == asked, where
        raw = path.read_bytes()
        assert (
            (not raw.endswith(b"\n")) if torn else (raw == b"" or raw.endswith(b"\n"))
        ), where
        store = EventLogSessionStore(reopened)
        resumed_llm = scenario.make_llm()
        api = scenario.resume(store, resumed_llm)
        restored = api.restore_session(cid)
        want = golden.view_after(survive)
        if want is None:
            assert restored is None, where
            return False
        assert restored is not None, where
        new_id, state = restored
        assert scenario.view(api, new_id) == want, (
            where
        )  # a whole state, never a half one
        unknown = state.metadata.get("unknown_tool_calls", [])
        open_calls = unresolved_tool_calls(events)
        assert [c["call_id"] for c in unknown] == [c["call_id"] for c in open_calls], (
            where
        )
        assert {c["status"] for c in unknown} <= {"unknown"}, where
        assert resumed_llm.call_history == [], (
            where
        )  # a restore calls nothing, retries nothing
        return bool(unknown)
    finally:
        reopened.close()
    # (torn tails are repaired by the next writer, proven in test_event_log_jsonl.py)


def _sweep(scenario: _Scenario, tmp_path: Path, *, torn: bool) -> tuple[_Golden, int]:
    golden = _golden(scenario, tmp_path / "golden")
    unknown_seen = 0
    for survive in range(golden.writes):  # every k; the full run is the golden itself
        root = tmp_path / f"k{survive}"
        unknown_seen += _check_death(scenario, golden, root, survive, torn)
    return golden, unknown_seen


class TestEveryDeathOffset:
    def test_a_tool_exchange_dies_at_every_event_and_restores_a_whole_state(
        self, tmp_path: Path
    ) -> None:
        golden, unknown_seen = _sweep(TOOLS, tmp_path, torn=False)
        types = [e["type"] for e in golden.events]
        # non-vacuity: the run holds what the sweep claims to cover, and the sweep saw it
        assert golden.writes >= 25
        assert {"llm/request", "tool/call", "tool/result", "turn/end"} <= set(types)
        assert types.count("llm/request") == 2 and types.count("turn/start") == 4
        assert unknown_seen == 1  # the one offset between tool/call and tool/result
        assert golden.view_after(golden.writes)["current_state"] == "done"
        assert golden.views[0][1]["current_state"] == "intake"

    def test_the_same_deaths_with_a_torn_last_line(self, tmp_path: Path) -> None:
        golden, unknown_seen = _sweep(TOOLS, tmp_path, torn=True)
        assert golden.writes >= 25 and unknown_seen == 1

    def test_a_rolled_back_turn_then_a_crash_restores_the_rolled_back_state(
        self, tmp_path: Path
    ) -> None:
        golden, _ = _sweep(CHAT, tmp_path, torn=False)
        events = golden.events
        ends = [e for e in events if e["type"] == "turn/end"]
        assert [e["outcome"] for e in ends] == ["rolled_back", "committed", "committed"]
        failed = next(e for e in events if e["type"] == "turn/start")
        inside = [e for e in events if failed["seq"] < e["seq"] < ends[0]["seq"]]
        # non-vacuity: Pass 1 changed the state inside the failed turn (name set, moved to
        # talk), so a restore that kept it would differ from the pre-turn view
        assert any(e["type"] == "transition" and e["executed"] for e in inside)
        pre_turn, after_failure = golden.views[0][1], golden.views[1][1]
        assert pre_turn["current_state"] == "intake" and pre_turn == after_failure
        assert golden.views[2][1]["current_state"] == "talk"
        assert any(e["type"] == "provider/error" for e in inside)

    def test_the_rolled_back_deaths_with_a_torn_last_line(self, tmp_path: Path) -> None:
        golden, _ = _sweep(CHAT, tmp_path, torn=True)
        assert golden.writes >= 20

    def test_a_torn_tail_is_repaired_by_the_next_writer_with_the_acked_seq(
        self, tmp_path: Path
    ) -> None:
        golden = _golden(TOOLS, tmp_path / "golden")
        survive = next(
            i for i, e in enumerate(golden.events) if e["type"] == "llm/request"
        )  # die inside the first model request
        _kill_at(TOOLS, tmp_path / "dead", survive, torn=True)
        path = next((tmp_path / "dead" / "logs").glob("*.jsonl"))
        assert not path.read_bytes().endswith(b"\n")
        writer = JsonlWriter(path)  # the lock holder cuts the tail
        try:
            assert writer.last_seq == survive
        finally:
            writer.close()
        assert path.read_bytes().endswith(b"\n")
        assert len(list(read_events(path))) == survive


def _products(use_async: bool) -> _Scenario:
    return _Scenario(
        name=f"the product paths, {'async' if use_async else 'sync'} entry points",
        make_llm=cells.product_llm,
        play=lambda log, llm, record: cells.product_play(log, llm, record, use_async),
        resume=lambda store, llm: API(
            cells.product_fsm(), llm_interface=llm, session_store=store
        ),
        view=cells.product_view,
    )


PRODUCTS, PRODUCTS_ASYNC = _products(False), _products(True)
#: what the product scenario writes besides the turn plumbing: a registry tool, an approval,
#: the extraction family, a handler and the greeting (plan 3f4530d3 step 9)
_PRODUCT_TYPES = {
    "tool/call", "tool/result", "approval/request", "approval/outcome", "handler",
    "extraction", "extraction/miss", "intent/out_of_scope", "llm/request", "llm/response",
}  # fmt: skip


class TestTheProductPaths:
    """Every death offset of a run through the producers of plan 3f4530d3: a registry tool, an
    approval, the extraction family, a handler and the greeting, through the sync and the async
    entry points (in async mode every boundary write is a job of a lane thread, so a death lands
    in a lane thread and reaches the awaiting turn as the same ``_Dead``). The golden view is the
    LIVE view at each boundary and equals the view the PARENT archive showed with the log off
    (``fixtures/event_log_parity_golden.json``, written on the plan base), never a fold."""

    def _sweep_the_products(
        self, tmp_path: Path, scenario: _Scenario, *, torn: bool
    ) -> None:
        golden, unknown_seen = _sweep(scenario, tmp_path, torn=torn)
        types = [e["type"] for e in golden.events]
        assert _PRODUCT_TYPES <= set(
            types
        )  # non-vacuity: the run holds what the sweep claims
        assert golden.writes >= 50
        side = [
            e for e in golden.events if e["type"] == "tool/call" and e["side_effecting"]
        ]
        reads = [e for e in golden.events if e["type"] == "tool/call"]
        assert len(side) == 2 and len(reads) == 4  # add and peek, in two turns
        # one offset between each tool/call (read-only ones too: the fold lists every call
        # without a result) and its tool/result
        assert unknown_seen == len(reads)
        outcomes = [
            e["approved"] for e in golden.events if e["type"] == "approval/outcome"
        ]
        assert outcomes == [True, True]
        assert golden.view_after(golden.writes)["current_state"] == "talk"
        # the golden is the parent's live view at the same boundaries, state for state
        wanted = cells.golden_views()["async" if scenario is PRODUCTS_ASYNC else "sync"]
        assert [view for _, view in golden.views] == wanted

    def test_the_sync_entry_points_die_at_every_offset_and_restore_a_whole_state(
        self, tmp_path: Path
    ) -> None:
        self._sweep_the_products(tmp_path, PRODUCTS, torn=False)

    def test_the_async_entry_points_die_at_every_offset_in_a_lane_thread(
        self, tmp_path: Path
    ) -> None:
        self._sweep_the_products(tmp_path, PRODUCTS_ASYNC, torn=False)

    def test_the_same_sync_deaths_with_a_torn_last_line(self, tmp_path: Path) -> None:
        self._sweep_the_products(tmp_path, PRODUCTS, torn=True)

    def test_the_same_async_deaths_with_a_torn_last_line(self, tmp_path: Path) -> None:
        self._sweep_the_products(tmp_path, PRODUCTS_ASYNC, torn=True)

    def test_a_spent_budget_decision_dies_at_every_offset_too(
        self, tmp_path: Path
    ) -> None:
        golden, _ = _sweep(BUDGET, tmp_path, torn=False)
        decisions = [e for e in golden.events if e["type"] == "decision"]
        assert [e["kind"] for e in decisions] == ["budget_spent"]
        assert golden.events[-1]["type"] in {"decision", "turn/end"}

    def test_the_sync_and_the_async_run_write_the_same_events(
        self, tmp_path: Path
    ) -> None:
        sync = _golden(PRODUCTS, tmp_path / "s")
        aync = _golden(PRODUCTS_ASYNC, tmp_path / "a")
        assert [(e["type"], e["frame"]) for e in sync.events] == [
            (e["type"], e["frame"]) for e in aync.events
        ]
        assert [view for _, view in sync.views] == [view for _, view in aync.views]


class TestFsyncPolicy:
    def test_only_a_request_a_side_effecting_tool_call_and_a_turn_end_are_fsynced(
        self, tmp_path: Path
    ) -> None:
        order: list[tuple[str, ...]] = []
        real_write, real_fsync = jsonl._write, jsonl._fsync

        def write(fd: int, data: Any) -> int:
            for line in bytes(data).split(b"\n")[:-1]:
                event = json.loads(line)
                flag = "side" if event.get("side_effecting", True) else "read"
                order.append(("write", event["type"], flag))
            return real_write(fd, data)

        def fsync(fd: int) -> None:
            order.append(("fsync",))
            real_fsync(fd)

        with (
            mock.patch.object(jsonl, "_write", write),
            mock.patch.object(jsonl, "_fsync", fsync),
            EventLog(tmp_path / "root") as log,
        ):
            TOOLS.play(log, TOOLS.make_llm(), lambda api, cid: None)
            cid = _only_log(log)
            with log.tool_call(
                cid, name="peek", call_id="r1", arguments={}, side_effecting=False
            ):
                pass
        writes = [step for step in order if step[0] == "write"]
        fsyncs = [i for i, step in enumerate(order) if step[0] == "fsync"]
        due = [
            i
            for i, step in enumerate(order)
            if step[0] == "write"
            and (
                step[1] in {"llm/request", "turn/end"}
                or (step[1] == "tool/call" and step[2] == "side")
            )
        ]
        # each fsync directly follows its checkpoint event, and each checkpoint event is
        # directly followed by one: the sets match offset for offset
        assert [i - 1 for i in fsyncs] == due
        assert (
            len(fsyncs) == 7
        )  # 2 llm/request + 1 side-effecting tool/call + 4 turn/end
        assert (
            len(writes) == 28 + 2
        )  # the run plus the read-only tool/call and its result
        assert ("write", "tool/call", "read") in order


class TestTheKilledChild:
    def test_the_capability_probe_says_run_on_every_ci_platform(self) -> None:
        # Non-vacuity: a probe that always said False would skip the child test everywhere.
        # The skip count per leg is recorded in progress.md (expected 0 on all seven).
        if sys.platform not in {"linux", "win32", "darwin"}:
            pytest.skip("not a platform CI runs")
        assert can_lock_in_child() is True

    def test_the_child_environment_has_no_model_and_no_forced_utf8(self) -> None:
        env = child_env()
        assert "PYTHONUTF8" not in env and "LLM_MODEL" not in env
        assert env["OLLAMA_API_BASE"] == env["OLLAMA_HOST"] == "http://127.0.0.1:9"
        assert set(env) >= {"PATH", "LITELLM_LOCAL_MODEL_COST_MAP"}

    def test_a_kill_in_the_middle_of_a_turn_restores_the_last_committed_turn(
        self, tmp_path: Path
    ) -> None:
        if not can_lock_in_child():
            pytest.skip("this platform cannot start a child that locks a file")
        root, ready = tmp_path / "log", tmp_path / "ready.json"
        _run_until_blocked_then_kill(root, ready, tmp_path / "child.err")
        announced = json.loads(ready.read_text(encoding="utf-8"))
        cid, expected = announced["conversation_id"], announced["expected"]
        path = root / "logs" / f"{cid}.jsonl"
        events = list(
            read_events(path)
        )  # the kill may tear the tail; readers ignore it
        types = [e["type"] for e in events]
        # an open turn with its request last: the second model call was written (and fsynced)
        # before the double ran, and nothing after it
        assert types[-5:] == [
            "turn/end",
            "tool/call",
            "turn/start",
            "executor/start",
            "llm/request",
        ], types
        assert events[-1]["source"] == "engine" and events[-1]["seq"] == len(events)
        assert [e["call_id"] for e in unresolved_tool_calls(events)] == ["in-flight"]
        with EventLog(root) as reopened:
            projection = fold_session(reopened.read(cid), reopened.get_json)
            assert projection is not None
            assert projection.current_state == expected["current_state"]
            assert projection.context_data == expected["data"]
            assert projection.conversation_history == expected["history"]
            assert projection.completion_context == {
                child.MESSAGES: expected["transcript"]
            }
            assert _last_seq(path) == len(events)  # the dead holder's lock is free
        _assert_resumes_and_continues(root, cid, expected)
        # Unverified until the Windows run: msvcrt lock release and a readable log after
        # TerminateProcess; the Linux run proves the shape.


def _last_seq(path: Path) -> int:
    """Open a writer on the log (this takes its lock) and close it again."""
    writer = reopen_after_kill(lambda: JsonlWriter(path))
    try:
        return writer.last_seq
    finally:
        writer.close()


def _run_until_blocked_then_kill(root: Path, ready: Path, err: Path) -> None:
    """Start the child, wait for its ready file, kill it with ``Popen.kill()``.

    The wait is a bounded number of polls (scaled by the timing factor), not a clock read; a
    child that exits early fails the test with the tail of its stderr.
    """
    with child_process([CHILD_MODULE, str(root), str(ready)], err) as proc:
        wait_ready(proc, ready, err)
        proc.kill()  # SIGKILL on POSIX, TerminateProcess on Windows: the portable kill
        proc.wait(timeout=60 * current_timing_factor())


def _assert_resumes_and_continues(
    root: Path, cid: str, expected: dict[str, Any]
) -> None:
    with EventLog(root) as reopened:
        llm = child.ChildLLM([final_reply("5")])
        api = API.from_definition(
            child.tool_fsm(),
            llm_interface=llm,
            session_store=EventLogSessionStore(reopened),
            event_log=reopened,
        )
        restored = api.restore_session(cid)
        assert restored is not None
        new_id, state = restored
        assert (
            child.view(api, new_id) == expected
        )  # the last committed turn, transcript included
        unknown = state.metadata["unknown_tool_calls"]
        assert [(c["call_id"], c["status"]) for c in unknown] == [
            ("in-flight", "unknown")
        ]
        assert llm.call_history == []  # the restore retried nothing
        step = api.advance(
            new_id
        )  # the conversation continues from the stored transcript
        assert step.state_after == "done" and api.has_conversation_ended(new_id)
        assert [r.messages[1:] for r in llm.requests] == [expected["transcript"]]


class TestThisFile:
    def test_every_child_process_of_the_log_tests_gets_the_shared_environment(
        self,
    ) -> None:
        here, started = Path(__file__).parent, 0
        for name in (
            "test_event_log_jsonl.py",
            "test_event_log_blobs.py",
            "test_event_log_stack_crash.py",
            "event_log_support.py",
        ):
            tree = ast.parse((here / name).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and ast.unparse(node.func) in {
                    "subprocess.Popen",
                    "subprocess.run",
                }:
                    envs = {
                        ast.unparse(k.value) for k in node.keywords if k.arg == "env"
                    }
                    # cost-map flag, temp and home variables
                    assert envs == {"child_env()"}, (name, node.lineno)
                    started += 1
        assert started >= 4  # none missed, none vacuous

    def test_no_test_of_this_file_carries_a_marker(self) -> None:
        # CI deselects slow, integration and real_llm: a marked kill test would never run there
        module = sys.modules[__name__]
        found: list[str] = []
        assert not getattr(module, "pytestmark", [])
        for name, obj in vars(module).items():
            if name.startswith("Test") and isinstance(obj, type):
                assert not getattr(obj, "pytestmark", []), name
                for attr, method in vars(obj).items():
                    if attr.startswith("test_"):
                        assert not getattr(method, "pytestmark", []), (name, attr)
                        found.append(f"{name}.{attr}")
            elif name.startswith("test_") and callable(obj):
                assert not getattr(obj, "pytestmark", []), name
                found.append(name)
        tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
        defined = [
            f"{cls.name}.{fn.name}"
            for cls in tree.body
            if isinstance(cls, ast.ClassDef)
            for fn in cls.body
            if isinstance(fn, ast.FunctionDef) and fn.name.startswith("test_")
        ]
        assert (
            sorted(found) == sorted(defined) and len(found) >= 8
        )  # none missed, none vacuous
        banned = {"os.kill", "signal.SIGKILL", "time.perf_counter"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                dotted = ast.unparse(node)
                assert dotted not in banned and not dotted.startswith("pytest.mark"), (
                    dotted
                )
