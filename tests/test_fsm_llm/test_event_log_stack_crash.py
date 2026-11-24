"""Crash proofs for the stack in the event log (plan 5a6a1f80 step 6, roadmap row 1.4).

The recovery rule (D-004, written before the sweep): replaying a prefix in ``seq`` order, a frame's
state is the last state-carrying event of that frame (``turn/start``, ``turn/end``, a
``context/record`` with a state or delta, the creating ``stack/push``, for the parent the
``stack/pop``); a push of depth d replaces frame d-1 and is the only commit of a frame; a pop
truncates and a missing pop is inferred from an event at a lower frame; an open turn contributes its
pre-turn state; a crash before a child's push loses the child, before the pop restores the pre-pop
stack. The GOLDEN VIEW is observed from the LIVE ``API`` right after every state-carrying append
(``_Watched``), never computed by a fold. Proofs (``seq`` is the order, no test reads ``ts``):

1. every death offset k (a ``_Dead`` at the (k+1)-th ``_write``): the log is the acknowledged
   prefix; fold, ``fork`` at every finished turn and ``resume`` equal the golden view; a fork's own
   log folds alone and names its lineage; the lock comes before any read; no model call; the
   in-flight tool is reported once;
2. a torn tail of every length of every push and pop line (``tried`` equals the line length) and
   half of every other line;
3. a lost or replayed push or pop, which a counting fold cannot survive;
4. a real ``Popen.kill()`` at frame 1 (resume, continue, pop, finish) and a second process that
   holds the writer (a resumer is refused, a fork still reads). Neither is marked: CI deselects
   ``slow`` and ``integration``.

Unverified until the Windows run: the lock release after ``TerminateProcess``.
"""

from __future__ import annotations

import ast
import contextlib
import json
import shutil
import subprocess
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from fsm_llm import API
from fsm_llm.core.definitions import FSMError
from fsm_llm.engine.api import forks
from fsm_llm.runtime import EventLog, LogCorruptError, LogLockedError
from fsm_llm.runtime.stores import fold_stack, jsonl, read_events
from fsm_llm.testing import final_reply
from tests.conftest import current_timing_factor
from tests.test_fsm_llm import event_log_child as child
from tests.test_fsm_llm.event_log_support import (
    CHILD_MODULE,
    _Dead,
    child_process,
    reopen_after_kill,
    wait_ready,
)
from tests.test_fsm_llm.test_event_log_crash import _Crash
from tests.test_fsm_llm.test_event_log_jsonl import can_lock_in_child
from tests.test_fsm_llm.test_event_log_stack import _FOLD_TYPES, _folded_view, _Watched

_CARRY = {kind.value for kind in _FOLD_TYPES}
_View = tuple[int, list[tuple[str, dict[str, Any]]]]


@dataclass(frozen=True)
class _Golden:
    """The uncrashed run: its directory and root id, its events, and the live view at each
    state-carrying event keyed by ``seq``."""

    root: Path
    cid: str
    events: list[dict[str, Any]]
    views: dict[int, _View]

    def of(self, events: list[dict[str, Any]]) -> _View | None:
        """The golden view of a log holding *events*: the live view at the last state-carrying
        event (a replayed event, one whose ``seq`` came before, changes nothing)."""
        seen: set[int] = set()
        last = None
        for event in events:
            if event["seq"] not in seen and event["type"] in _CARRY:
                last = event["seq"]
            seen.add(event["seq"])
        return None if last is None else self.views[last]

    def at(self, cut: int) -> _View | None:
        return self.of(self.events[:cut])

    def cut(self, turn: int) -> int:
        """The ``seq`` a fork at *turn* stops at, read from the events (not from ``turn_cut``)."""
        if turn == 0:
            return next(e["seq"] for e in self.events if e["type"] == "turn/start") - 1
        return [e["seq"] for e in self.events if e["type"] == "turn/end"][turn - 1]

    def types(self, kind: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["type"] == kind]


def _stacked_api(log: EventLog, llm: Any = None) -> API:
    llm = llm or child.stack_llm(child.scripted_replies())
    return API(child.chat_fsm(), llm_interface=llm, event_log=log)


def _run_golden(root: Path) -> _Golden:
    with _Watched(root) as log:
        api = _stacked_api(log)
        log.api = api  # the hook reads the live view after each state-carrying append
        cid = child.stacked_start(api, log)
        child.stacked_finish(api, cid)
        log.api = None
        events = list(log.read(cid))
        views = dict(log.seen)
    return _Golden(root, cid, events, views)


@pytest.fixture(scope="module")
def golden(tmp_path_factory: pytest.TempPathFactory) -> _Golden:
    return _run_golden(tmp_path_factory.mktemp("golden"))


def _open_calls(events: list[dict[str, Any]], cut: int) -> list[str]:
    """Side-effecting tool calls of the first *cut* events that have no result yet."""
    first = [e for e in events if e["seq"] <= cut]
    done = {e["call_id"] for e in first if e["type"] == "tool/result"}
    return [
        e["call_id"]
        for e in first
        if e["type"] == "tool/call"
        and e.get("side_effecting", True)
        and e["call_id"] not in done
    ]


@contextlib.contextmanager
def _watching(steps: list[str]) -> Iterator[None]:
    """Append to *steps*, in order, ``lock`` for ``open_existing`` and ``read`` for every read,
    query and fold ``resume`` makes."""

    def spy(name: str, real: Any) -> Any:
        def wrapper(*a: Any, **k: Any) -> Any:
            steps.append(name)
            return real(*a, **k)

        return wrapper

    with contextlib.ExitStack() as stack:
        for owner, attr, name in (
            (EventLog, "open_existing", "lock"),
            (EventLog, "read", "read"),
            (EventLog, "query", "read"),
            (forks, "fold_stack", "read"),
        ):
            real = getattr(owner, attr)
            stack.enter_context(mock.patch.object(owner, attr, spy(name, real)))
        yield


def _check_cut(
    root: Path, golden: _Golden, cid: str, cut: int, *, forks_too: bool
) -> int:
    """Fold, fork and resume the log in *root* (the first *cut* events, maybe a torn tail after
    them) and compare each with the golden view; return the number of unknown tool calls."""
    where = f"after {cut} events"
    with EventLog(root) as log:
        events = list(log.read(cid))
        assert [e["seq"] for e in events] == list(range(1, cut + 1)), where
        want = golden.at(cut)
        folded = fold_stack(iter(events), log.get_json)
        assert (None if folded is None else _folded_view(folded)) == want, where
        llm = child.stack_llm([])
        api = _stacked_api(log, llm)
        ends = sum(e["type"] == "turn/end" for e in events)
        if forks_too:
            _check_forks(api, log, golden, cid, ends, want, where)
        if want is None:
            with pytest.raises(FSMError):
                api.resume(cid)
            return 0
        order: list[str] = []
        with _watching(order):
            resumed, state = api.resume(cid)
        assert order[0] == "lock" and "read" in order, where
        assert resumed == cid and child.stack_view(api, cid) == want, where
        open_calls = _open_calls(golden.events, cut)
        unknown = state.metadata.get("unknown_tool_calls", [])
        assert [c["call_id"] for c in unknown] == open_calls, where
        assert {c["status"] for c in unknown} <= {"unknown"}, where
        assert llm.call_history == [], where  # nothing is retried, no model is asked
        after = list(log.read(cid))
        assert after[:cut] == events and [e["seq"] for e in after] == list(
            range(1, len(after) + 1)
        ), where
        assert sorted(e["type"] for e in after[cut:]) == sorted(
            ["tool/result"] * len(open_calls) + ["context/record"]
        ), where
        folded = fold_stack(iter(after), log.get_json)
        assert folded is not None and _folded_view(folded) == want, where
        api.end_conversation(cid)
        return len(open_calls)


def _check_forks(
    api: API,
    log: EventLog,
    golden: _Golden,
    cid: str,
    ends: int,
    want: _View | None,
    where: str,
) -> None:
    if want is None:
        with pytest.raises(FSMError):
            api.fork(cid, at_turn=0)
        return
    for turn in range(ends + 1):
        cut = golden.cut(turn)
        wanted = golden.at(cut)
        new_id, state = api.fork(cid, at_turn=turn)
        assert child.stack_view(api, new_id) == wanted, f"{where}, fork at {turn}"
        assert state.stack_depth == wanted[0]  # type: ignore[index]
        mine = list(log.read(new_id))
        # lineage: the fork says where it came from, and its own log stands alone
        assert mine[0]["source"] == "fork", where
        assert mine[0]["forked_from"] == {
            "conversation_id": cid,
            "seq": cut,
            "turn": turn,
        }, where
        folded = fold_stack(iter(mine), log.get_json)
        assert folded is not None and _folded_view(folded) == wanted, where
        api.end_conversation(new_id)
    with pytest.raises(ValueError):  # past the last finished turn
        api.fork(cid, at_turn=ends + 1)


def _die_after(root: Path, survive: int) -> str:
    """Run the scenario until the (survive+1)-th write dies; return the root id of the log."""
    crash = _Crash(survive)
    with EventLog(root) as log, mock.patch.object(jsonl, "_write", crash):
        api = _stacked_api(log)
        with pytest.raises(_Dead):
            child.stacked_finish(api, child.stacked_start(api, log))
    assert crash.dead
    with EventLog(
        root
    ) as reopened:  # the dead writer's lock is free once its fd is closed
        (cid,) = reopened.conversation_ids()
    return cid


class TestTheScenario:
    def test_the_scenario_holds_what_the_sweep_claims_to_cover(
        self, golden: _Golden
    ) -> None:
        pushes, pops = golden.types("stack/push"), golden.types("stack/pop")
        assert [p["depth"] for p in pushes + pops] == [2, 2, 1]
        # 50 events: the root greeting is logged (a request and its reply, plan 3f4530d3 step 2)
        assert len(golden.events) == 50 and len(golden.types("turn/end")) == 6
        # frame 1 has turns, two deltas and a tool exchange; the second push is left open
        top = [e["type"] for e in golden.events if e["frame"] == 1]
        assert top.count("tool/call") == 1 and top.count("context/record") == 2
        assert {view[0] for view in golden.views.values()} == {1, 2}
        assert golden.at(len(golden.events))[0] == 2  # type: ignore[index]
        # the pop merged the child's return into the parent: the log has it only in the pop
        assert golden.views[pops[0]["seq"]][1][0][1]["result"] == 5


class TestEveryDeathOffset:
    def test_each_offset_folds_forks_and_resumes_to_the_golden_view(
        self, golden: _Golden, tmp_path: Path
    ) -> None:
        covered: set[int] = set()
        unknown = 0
        for survive in range(len(golden.events)):
            root = tmp_path / f"k{survive}"
            cid = _die_after(root, survive)
            with EventLog(root) as log:
                types = [(e["type"], e["frame"]) for e in log.read(cid)]
            # the log is exactly the acknowledged prefix of the golden run, nothing more
            assert types == [(e["type"], e["frame"]) for e in golden.events[:survive]]
            unknown += _check_cut(root, golden, cid, survive, forks_too=True)
            covered.add(survive)
        full = tmp_path / "full"
        shutil.copytree(golden.root, full)
        unknown += _check_cut(
            full, golden, golden.cid, len(golden.events), forks_too=True
        )
        covered.add(len(golden.events))
        assert covered == set(range(len(golden.events) + 1))  # EVERY k, including 0
        assert unknown == 1  # the one offset between the tool/call and its tool/result


class TestEveryTornTail:
    @staticmethod
    def _lines(golden: _Golden) -> list[bytes]:
        raw = (golden.root / "logs" / f"{golden.cid}.jsonl").read_bytes()
        lines = raw.splitlines(keepends=True)
        assert len(lines) == len(golden.events) and b"".join(lines) == raw
        return lines

    def test_every_length_of_every_push_and_pop_line_and_half_of_the_others(
        self, golden: _Golden, tmp_path: Path
    ) -> None:
        lines = self._lines(golden)
        scratch = tmp_path / "scratch"
        shutil.copytree(golden.root, scratch)
        path = scratch / "logs" / f"{golden.cid}.jsonl"
        stack_lines = {
            e["seq"] for e in golden.events if e["type"] in {"stack/push", "stack/pop"}
        }
        tried: dict[int, int] = {}
        for seq, line in enumerate(lines, start=1):
            acked = b"".join(lines[: seq - 1])
            lengths = range(len(line)) if seq in stack_lines else [len(line) // 2]
            for length in lengths:  # every proper prefix of the line, the newline last
                path.write_bytes(acked + line[:length])
                _check_cut(scratch, golden, golden.cid, seq - 1, forks_too=False)
                data = path.read_bytes()  # the writer cut the tail, acked bytes stand
                assert data.startswith(acked) and (data == b"" or data.endswith(b"\n"))
                tried[seq] = tried.get(seq, 0) + 1
        assert {seq: tried[seq] for seq in stack_lines} == {
            seq: len(lines[seq - 1]) for seq in stack_lines
        }  # non-vacuity: tried == len(line) for each stack-carrying line
        assert sum(tried.values()) > 1500 and set(tried) == set(range(1, 51))


def _variants(golden: _Golden) -> list[tuple[str, list[dict[str, Any]]]]:
    events, out = golden.events, []
    for index, event in enumerate(events):
        if event["type"] in {"stack/push", "stack/pop"}:
            name = f"{event['type']} {event['seq']}"
            out.append((f"lost {name}", events[:index] + events[index + 1 :]))
            out.append(
                (
                    f"replayed {name}",
                    [*events[: index + 1], event, *events[index + 1 :]],
                )
            )
    return out


class TestALostOrReplayedRecord:
    def test_no_prefix_of_a_log_that_lost_or_repeated_a_stack_record_shifts_the_stack(
        self, golden: _Golden
    ) -> None:
        with EventLog(golden.root) as log:
            variants = _variants(golden)
            first_push = golden.types("stack/push")[0]["seq"]
            assert len(variants) == 6
            checked = corrupt = 0
            for name, events in variants:
                lost = name == f"lost stack/push {first_push}"
                for end in range(1, len(events) + 1):
                    prefix = events[:end]
                    if lost and any(
                        e["frame"] >= 1 and e["type"] in _CARRY for e in prefix
                    ):
                        # the push is the only commit of a frame: a state event of an uncovered
                        # frame is refused, never guessed
                        with pytest.raises(LogCorruptError, match="no stack/push"):
                            fold_stack(iter(prefix), log.get_json)
                        corrupt += 1
                        continue
                    folded = fold_stack(iter(prefix), log.get_json)
                    got = None if folded is None else _folded_view(folded)
                    assert got == golden.of(prefix), f"{name}, {end} events"
                    checked += 1
            assert corrupt > 30 and checked > 200  # non-vacuity: both kinds really ran

    def test_a_push_at_an_occupied_depth_replaces_the_finished_child(
        self, golden: _Golden
    ) -> None:
        pop, push = golden.types("stack/pop")[0], golden.types("stack/push")[1]
        # the pop and the root turn after it are lost: the next push finds depth 2 occupied
        kept = [e for e in golden.events if not pop["seq"] <= e["seq"] < push["seq"]]
        with EventLog(golden.root) as log:
            folded = fold_stack(iter(kept), log.get_json)
        assert folded is not None
        root_before_the_merge = golden.views[pop["seq"] - 1][1][0]
        second_child = golden.views[push["seq"]][1][1]
        assert _folded_view(folded) == (2, [root_before_the_merge, second_child])


def _normal(view: Any) -> Any:
    return json.loads(json.dumps(view))


class TestRealProcesses:
    def test_the_capability_probe_says_run_on_every_ci_platform(self) -> None:
        # Non-vacuity: a probe that always said False would skip both process tests everywhere.
        if not can_lock_in_child():
            pytest.fail("the probe says this platform cannot run the child tests")

    def test_a_process_killed_mid_turn_at_frame_one_is_resumed_and_finishes(
        self, golden: _Golden, tmp_path: Path
    ) -> None:
        if not can_lock_in_child():
            pytest.skip("this platform cannot start a child that locks a file")
        root, ready, err = tmp_path / "log", tmp_path / "ready.json", tmp_path / "err"
        args = [CHILD_MODULE, str(root), str(ready), "stacked", "run"]
        with child_process(args, err) as proc:
            wait_ready(proc, ready, err)
            proc.kill()  # SIGKILL on POSIX, TerminateProcess on Windows
            proc.wait(timeout=60 * current_timing_factor())
        announced = json.loads(ready.read_text(encoding="utf-8"))
        cid, expected = announced["conversation_id"], announced["expected"]
        events = list(
            read_events(root / "logs" / f"{cid}.jsonl")
        )  # a torn tail is ignored
        assert [e["type"] for e in events[-5:]] == [
            "turn/end",
            "tool/call",
            "turn/start",
            "executor/start",
            "llm/request",
        ]
        assert {e["frame"] for e in events[-5:]} == {
            1
        }  # killed inside the child's turn
        with EventLog(root) as log:
            llm = child.stack_llm([final_reply("5")])
            api = _stacked_api(log, llm)
            resumed, state = reopen_after_kill(lambda: api.resume(cid))
            assert resumed == cid
            assert _normal(child.stack_view(api, cid)) == expected  # the pre-turn state
            top = api.conversation_stacks[cid][1]
            assert (top.shared_context_keys, top.preserve_history) == (["name"], True)
            assert top.return_context == {"answer": 42}
            assert [
                (c["call_id"], c["status"])
                for c in state.metadata["unknown_tool_calls"]
            ] == [("in-flight", "unknown")]
            assert (
                llm.call_history == []
            )  # the in-flight tool and its model call: not re-run
            child.stacked_finish(
                api, cid
            )  # the child answers, it pops, the root goes on
            assert child.stack_view(api, cid) == golden.at(len(golden.events))
            after = list(log.read(cid))
            assert [e["seq"] for e in after] == list(range(1, len(after) + 1))
            assert [
                e["source"]
                for e in after
                if e["type"] == "context/record" and e.get("source") == "resume"
            ] == ["resume"]
            assert [
                e["status"]
                for e in after
                if e["type"] == "tool/result" and e["call_id"] == "in-flight"
            ] == ["unknown"]
            folded = fold_stack(iter(after), log.get_json)
            assert folded is not None and _folded_view(folded) == child.stack_view(
                api, cid
            )
        # Unverified until the Windows run: msvcrt lock release after TerminateProcess.

    def test_a_log_held_by_another_process_refuses_a_resumer_and_still_forks(
        self, golden: _Golden, tmp_path: Path
    ) -> None:
        if not can_lock_in_child():
            pytest.skip("this platform cannot start a child that locks a file")
        root, ready, err = tmp_path / "log", tmp_path / "ready.json", tmp_path / "err"
        shutil.copytree(golden.root, root)
        cid, final = golden.cid, golden.at(len(golden.events))
        args = [CHILD_MODULE, str(root), str(ready), "stacked", "resume", cid]
        source = root / "logs" / f"{cid}.jsonl"
        with child_process(args, err, stdin=subprocess.PIPE) as proc:
            wait_ready(proc, ready, err)
            assert json.loads(ready.read_text(encoding="utf-8"))["expected"] == _normal(
                final
            )
            held = source.read_bytes()  # what the winner (the child) left
            with EventLog(root) as log:
                api = _stacked_api(log)
                reads: list[str] = []
                real_read = EventLog.read

                def spy(self: EventLog, *a: Any, **k: Any) -> Any:
                    reads.append("read")
                    return real_read(self, *a, **k)

                with mock.patch.object(EventLog, "read", spy):
                    with pytest.raises(LogLockedError):
                        api.resume(cid)  # exactly one process owns the log
                assert reads == [] and api.list_active_conversations() == []
                for turn in (
                    0,
                    2,
                    4,
                    6,
                ):  # a fork takes no writer lock: it never blocks
                    new_id, _ = api.fork(cid, at_turn=turn)
                    assert child.stack_view(api, new_id) == golden.at(golden.cut(turn))
                    api.end_conversation(new_id)
                assert source.read_bytes() == held  # the source log was not touched
            assert proc.stdin is not None
            proc.stdin.write(b"\n")
            proc.stdin.close()
            assert proc.wait(timeout=60 * current_timing_factor()) == 0
        with EventLog(root) as log:  # the winner is gone: the next resumer owns it
            api = _stacked_api(log)
            reopen_after_kill(lambda: api.resume(cid))
            assert child.stack_view(api, cid) == final
            sources = [
                e["source"] for e in log.read(cid) if e.get("source") == "resume"
            ]
            assert sources == ["resume", "resume"]  # the child's, then this one


class _Gate(EventLog):
    """A log that holds the writing thread right after one event until the test lets it go."""

    def __init__(self, root: Path, stop_after: int) -> None:
        super().__init__(root)
        self.stop_after = stop_after
        self.reached, self.release_it = threading.Event(), threading.Event()

    def append(self, conversation_id: str, type: Any, **kw: Any) -> int:  # type: ignore[override]
        seq = super().append(conversation_id, type, **kw)
        if seq == self.stop_after:
            self.reached.set()
            assert self.release_it.wait(60 * current_timing_factor())
        return seq


class TestAForkWhileTheSourceWriterAppends:
    def test_a_fork_is_the_same_before_during_and_after_the_writer_goes_on(
        self, golden: _Golden, tmp_path: Path
    ) -> None:
        stop = next(
            e["seq"]
            for e in golden.events
            if e["type"] == "turn/start" and e["seq"] > 27
        )
        failures: list[BaseException] = []
        with _Gate(tmp_path / "events", stop) as log:
            api = _stacked_api(log)
            ids: list[str] = []

            def source() -> None:
                try:
                    ids.append(child.stacked_start(api, log))
                    child.stacked_finish(api, ids[0])
                except BaseException as exc:  # reported by the main thread
                    failures.append(exc)
                    log.reached.set()

            thread = threading.Thread(target=source)
            thread.start()
            assert log.reached.wait(60 * current_timing_factor()) and not failures
            (cid,) = log.conversation_ids()
            done = [e for e in golden.types("turn/end") if e["seq"] < stop]
            for turn in range(len(done) + 1):  # the writer is mid-turn at frame 1
                new_id, _ = api.fork(cid, at_turn=turn)
                assert child.stack_view(api, new_id) == golden.at(golden.cut(turn))
                api.end_conversation(new_id)
            with pytest.raises(ValueError):  # the open turn is not a finished turn
                api.fork(cid, at_turn=len(done) + 1)
            log.release_it.set()
            thread.join(60 * current_timing_factor())
            assert not thread.is_alive() and not failures
            for turn in range(len(golden.types("turn/end")) + 1):
                new_id, _ = api.fork(cid, at_turn=turn)
                assert child.stack_view(api, new_id) == golden.at(golden.cut(turn))
                api.end_conversation(new_id)
            assert [(e["type"], e["frame"]) for e in log.read(cid)] == [
                (e["type"], e["frame"]) for e in golden.events
            ]  # the forks added nothing to the source log


class TestThisFile:
    def test_no_test_of_this_file_carries_a_marker(self) -> None:
        # CI deselects slow, integration and real_llm: a marked kill test would never run there
        tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
        marks = {
            n.attr
            for n in ast.walk(tree)
            if isinstance(n, ast.Attribute) and ast.unparse(n.value) == "pytest.mark"
        }
        names = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        assert not marks & {"slow", "integration", "real_llm", "examples"}
        assert not any(
            isinstance(n, ast.Name) and n.id == "pytestmark" for n in ast.walk(tree)
        )
        assert sum("killed" in name or "held_by" in name for name in names) == 2
