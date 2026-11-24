"""A delegated call run across a crash, a fork, a restore and an exit (plan baa37f40, step K7.6).

A delegated run lives in a thread of THIS process, so nothing of it is in the log but the parent's
own events: the ``running`` record (a handle and a deadline), ``run/start`` (fsynced before the
thread exists) and, once the committed record is final, ``run/end`` (written by settle on the
engine thread, from an in-memory mark). The rule of the design (D-016, roadmap 1.3): a run in
flight at a crash is UNKNOWN, never retried. These rows pin what that means on every path that
rebuilds a conversation without the host that owned the run:

1. a fork inside the running window and a snapshot restore end ``failed/run_lost`` and start
   nothing (the host scopes a handle to the conversation that reserved it, so even the SAME API
   and the SAME host adopt nothing); a fork after the final record is a normal fork;
2. a crash at EVERY event offset of one delegated run (the harness of ``test_call_log.py``): a
   fresh API (a fresh host) resumes the prefix to the view the live API showed, and the run goes
   on to a final record that is ``ok`` or ``failed/run_lost``, with at most ONE ``run/start`` and
   one ``run/end`` in the log, no second start and no run that was started and not announced;
3. an interpreter that exits with a delegated run thread alive and the log open leaves a log that
   reads clean (no torn line, no ``LogCorruptError``).

KNOWN LIMIT pinned by row 2 (decision D-054): the mark that owes ``run/end`` is in memory, so a run
whose ``run/start`` is on disk and whose process died ends in the log as ``run/start`` with no
``run/end`` (the 1.3 reading: in flight at the crash means unknown), even after the resumed
process wrote the final ``failed/run_lost`` record.

Rules of the rows (LESSONS ace7a8c8, D-048): no ``Thread.join``, events and bounded waits, no
sleep as synchronization, no row depends on the order of a scripted reply (the double answers by
default response), and the subprocess row has its own timeout.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

import fsm_llm
from fsm_llm import API, FileSessionStore
from fsm_llm.engine.api import delegated
from fsm_llm.runtime import EventLog, EventType, LogCorruptError
from fsm_llm.runtime.stores import fold_stack, jsonl
from tests.test_fsm_llm.event_log_child import stack_view as _view
from tests.test_fsm_llm.event_log_support import _Dead
from tests.test_fsm_llm.test_call_delegated_settle import (
    _BOUND,
    _assemble,
    _Probe,
)
from tests.test_fsm_llm.test_event_log_crash import _Crash
from tests.test_fsm_llm.test_event_log_stack import _FOLD_TYPES, _folded_view, _Watched

_ROOT = Path(__file__).resolve().parents[2]
_CARRY = {kind.value for kind in _FOLD_TYPES}


def test_the_module_under_test_is_this_checkout() -> None:
    assert Path(fsm_llm.__file__).resolve().is_relative_to(_ROOT / "src")


@pytest.fixture(autouse=True)
def _short_wait(monkeypatch):
    """The wait at the end of ``aadvance`` is short (a held run is waited for in vain)."""
    monkeypatch.setattr(delegated, "DELEGATED_WAIT_SECONDS", 0.05)


def _types(log: EventLog, cid: str) -> list[str]:
    return [e["type"] for e in log.read(cid)]


def _count(log: EventLog, cid: str, kind: EventType) -> int:
    return sum(t == kind.value for t in _types(log, cid))


def _record(api: API, cid: str) -> dict[str, Any]:
    frame = api.conversation_stacks[cid][0].conversation_id
    record = api.fsm_manager.copy_raw_context_data(frame).get("call_result")
    assert isinstance(record, dict)
    return record


def _state(api: API, cid: str) -> str:
    return api.fsm_manager.get_conversation_state(
        api.conversation_stacks[cid][0].conversation_id
    )


class TestAForkAndARestoreInsideTheRunningWindow:
    def test_a_fork_inside_the_window_ends_run_lost_and_starts_nothing(
        self, tmp_path: Path
    ) -> None:
        """Turn 1 ends with the ``running`` record and the run is held. A fork at turn 1 on the
        SAME API (the host that owns the run is right there) is a new conversation: the host
        scopes the handle to the conversation that reserved it, so the fork adopts nothing and
        ends ``failed/run_lost`` in its first turn. The original run is untouched and finishes.
        Kills: a poll or a start that is scoped to the handle alone."""
        probe = _Probe(hold=True)
        api, _targets, log, _, _llm, _extra = _assemble(tmp_path, probe=probe)
        assert log is not None
        try:
            cid, _ = api.start_conversation({"topic": "why"})
            api.advance(cid)  # turn 1: the request, then settle starts the run
            assert probe.entered.wait(_BOUND)
            assert _record(api, cid)["status"] == "running"
            new_id, state = api.fork(cid, at_turn=1)
            assert new_id != cid and state.current_state == "work"
            fork_frame = api.conversation_stacks[new_id][0].conversation_id
            assert (
                api.fsm_manager.copy_raw_context_data(fork_frame)["call_result"][
                    "status"
                ]
                == "running"
            )
            step = api.advance(new_id)
            record = _record(api, new_id)
            assert (record["status"], record["reason"]) == ("failed", "run_lost")
            assert step.state_after == "apology"
            assert len(probe.runs) == 1  # nothing was started for the fork
            assert _count(log, new_id, EventType.RUN_START) == 0
            assert _count(log, new_id, EventType.RUN_END) == 0
            # the source: still running, still the same run, finishes on its own
            assert _record(api, cid)["status"] == "running"
            probe.release.set()
            api.run_until_terminal(cid, max_steps=20)
            assert _record(api, cid)["status"] == "ok" and _state(api, cid) == "done"
            assert len(probe.runs) == 1
            assert (
                _record(api, new_id)["reason"] == "run_lost"
            )  # and the fork stays lost
            assert _count(log, cid, EventType.RUN_START) == 1
            assert _count(log, cid, EventType.RUN_END) == 1
        finally:
            probe.release.set()
            api.close()
            log.close()

    def test_a_fork_after_the_final_record_is_a_normal_fork(
        self, tmp_path: Path
    ) -> None:
        """The turn that ended with the final record carries the merged returns and the success
        edge: a fork there has nothing to poll and nothing to lose."""
        probe = _Probe()
        api, _targets, log, _, _llm, _extra = _assemble(tmp_path, probe=probe)
        assert log is not None
        try:
            cid, _ = api.start_conversation({"topic": "why"})
            api.run_until_terminal(cid, max_steps=20)
            turns = sum(t == EventType.TURN_END.value for t in _types(log, cid))
            assert turns == 2 and len(probe.runs) == 1
            new_id, state = api.fork(cid, at_turn=turns)
            assert state.current_state == "done"
            data = api.get_data(new_id)
            assert data["answer"] == "the answer" and data["found"] == ["f1"]
            assert _record(api, new_id)["status"] == "ok"
            assert len(probe.runs) == 1  # no second run for the fork
            assert _count(log, new_id, EventType.RUN_START) == 0
        finally:
            api.close()
            log.close()

    @pytest.mark.parametrize("same_api", [True, False])
    def test_a_snapshot_restore_inside_the_window_ends_run_lost(
        self, tmp_path: Path, same_api: bool
    ) -> None:
        """A snapshot saved while the run is in flight restores the record with a handle no host
        holds for the NEW conversation (a fresh API: a fresh host; the same API: the host of the
        original, which scopes the handle to the original). Either way the first turn ends
        ``failed/run_lost`` and nothing starts."""
        store = FileSessionStore(tmp_path / "sessions")
        probe = _Probe(hold=True)
        api, _targets, log, _, _llm, _extra = _assemble(tmp_path, probe=probe)
        assert log is not None
        other_log: EventLog | None = None
        other: API | None = None
        try:
            api._session_store = store  # the snapshot goes to this store
            cid, _ = api.start_conversation({"topic": "why"})
            api.advance(cid)
            assert probe.entered.wait(_BOUND)
            api.save_session(cid)
            saved = store.load(cid)
            assert saved is not None
            assert saved.context_data["call_result"]["status"] == "running"
            fresh = _Probe()
            if same_api:
                target_api = api
            else:
                other_log = EventLog(tmp_path / "other")
                other, *_rest = _assemble(tmp_path, probe=fresh, existing_log=other_log)
                other._session_store = store
                target_api = other
            restored = target_api.restore_session(cid)
            assert restored is not None
            new_id, _state_after = restored
            step = target_api.advance(new_id)
            record = _record(target_api, new_id)
            assert (record["status"], record["reason"]) == ("failed", "run_lost")
            assert step.state_after == "apology"
            assert len(probe.runs) == 1 and fresh.runs == []
        finally:
            probe.release.set()
            api.close()
            log.close()
            if other is not None:
                other.close()
            if other_log is not None:
                other_log.close()


# ---- the crash sweep -----------------------------------------------------------------------------


def _play(api: API) -> str:
    """Start, then run: the request, the start of the run, one poll, the final record."""
    cid, _ = api.start_conversation({"topic": "why"})
    api.run_until_terminal(cid, max_steps=20)
    return cid


class _Died:
    """What one dying incarnation left: the log's directory, root id, the views it showed live
    (by ``seq``) and the probe that its host ran."""

    def __init__(
        self, root: Path, cid: str | None, views: dict[int, Any], probe: _Probe
    ) -> None:
        self.root, self.cid, self.views, self.probe = root, cid, views, probe


def _die_after(root: Path, survive: int) -> _Died:
    """Run the scenario until the (survive+1)-th write dies; the log is the acknowledged prefix."""
    crash = _Crash(survive)
    probe = _Probe()
    with _Watched(root) as log, mock.patch.object(jsonl, "_write", crash):
        api, _targets, _log, _probe, _llm, _extra = _assemble(
            Path("."), probe=probe, existing_log=log
        )
        log.api = api
        with pytest.raises(_Dead):
            _play(api)
        log.api = None
        views = dict(log.seen)
    assert crash.dead
    with EventLog(
        root
    ) as reopened:  # the dead writer's lock is free once its fd is closed
        ids = reopened.conversation_ids()
    return _Died(root, ids[0] if ids else None, views, probe)


def _golden(root: Path) -> list[str]:
    """The event types of the run no crash touched (the sweep's yardstick)."""
    with EventLog(root) as log:
        api, _targets, _log, probe, _llm, _extra = _assemble(
            Path("."), existing_log=log
        )
        try:
            cid = _play(api)
            assert _record(api, cid)["status"] == "ok" and len(probe.runs) == 1
            return _types(log, cid)
        finally:
            api.close()


@pytest.fixture(scope="module")
def golden_types(tmp_path_factory: pytest.TempPathFactory) -> list[str]:
    return _golden(tmp_path_factory.mktemp("delegated_golden"))


class TestTheScenario:
    def test_the_run_holds_what_the_sweep_claims_to_cover(
        self, golden_types: list[str]
    ) -> None:
        """Non-vacuity: the order the sweep relies on. The record is committed (``turn/end``)
        BEFORE ``run/start``; ``run/start`` is before the second turn; ``run/end`` is last, after
        the turn that committed the final record."""
        start = golden_types.index(EventType.RUN_START.value)
        end = golden_types.index(EventType.RUN_END.value)
        first_end = golden_types.index(EventType.TURN_END.value)
        assert first_end < start < golden_types.index(EventType.TURN_START.value, start)
        assert end == len(golden_types) - 1
        assert golden_types.count(EventType.TURN_END.value) == 2
        assert golden_types[end - 1] == EventType.TURN_END.value


def _resume_and_finish(root: Path, cid: str, died: _Died, cut: int) -> dict[str, Any]:
    """Resume the prefix in a FRESH API (a fresh host), check the view, go on to the end."""
    where = f"after {cut} events"
    fresh = _Probe()
    with EventLog(root) as log:
        events = list(log.read(cid))
        assert [e["seq"] for e in events] == list(range(1, cut + 1)), where
        last = max((e["seq"] for e in events if e["type"] in _CARRY), default=None)
        folded = fold_stack(iter(events), log.get_json)
        want = None if last is None else died.views[last]
        assert (None if folded is None else _folded_view(folded)) == want, where
        api, _targets, _log, _probe, llm, _extra = _assemble(
            Path("."), probe=fresh, existing_log=log
        )
        try:
            assert want is not None, where
            resumed, _state_after = api.resume(cid)
            assert resumed == cid and _view(api, cid) == want, where
            assert llm.call_history == [], where  # resuming asks no model
            seen = want[1][0][1].get("call_result")
            api.run_until_terminal(cid, max_steps=20)
            record = _record(api, cid)
            starts = _count(log, cid, EventType.RUN_START)
            ends = _count(log, cid, EventType.RUN_END)
            folded_end = fold_stack(log.read(cid), log.get_json)
            assert folded_end is not None and folded_end.depth == 1, where
            return {
                "seen": seen,
                "record": record,
                "state": _state(api, cid),
                "starts": starts,
                "ends": ends,
                "fresh": fresh,
            }
        finally:
            api.close()


class TestEveryDeathOffset:
    def test_each_offset_resumes_to_a_final_record_with_no_second_start(
        self, golden_types: list[str], tmp_path: Path, monkeypatch
    ) -> None:
        """A death after EVERY landed event of one delegated run, then a resume in a fresh API:
        the log is the acknowledged prefix; the view equals what the live API showed; the call
        ends ``ok`` or ``failed/run_lost`` and is never started a second time:

        - no record committed yet (the request was in flight): the resumed process starts the
          run ONCE (the only start in the log);
        - a ``running`` record committed (with or without ``run/start`` on disk): the fresh host
          never saw the handle, so the record ends ``failed/run_lost``, nothing is re-run, and
          ``run/start`` is in the log exactly when the dead process started the thread;
        - the final record committed: a normal finish.

        The known limit (D-054): after a death with ``run/start`` on disk there is NO ``run/end``,
        even after the resumed process wrote the final record. Kills: a start from the record
        (``settle`` never asks the reservation), a restore that re-runs, a host that adopts a
        handle another process made."""
        # the wait ends as soon as the (instant) run finishes, so a long bound costs nothing and
        # keeps the order of events the same on every run
        monkeypatch.setattr(delegated, "DELEGATED_WAIT_SECONDS", _BOUND)
        kinds: set[str] = set()
        total = len(golden_types)
        for survive in range(total):
            root = tmp_path / f"k{survive}"
            died = _die_after(root, survive)
            where = f"after {survive} events"
            if survive == 0:  # nothing landed: nothing to resume
                continue
            assert died.cid is not None, where
            cid = died.cid
            with EventLog(root) as log:
                assert _types(log, cid) == golden_types[:survive], where
            out = _resume_and_finish(root, cid, died, survive)
            started_before = golden_types[:survive].count(EventType.RUN_START.value)
            if started_before:  # the dead process really started the thread
                assert died.probe.entered.wait(_BOUND), where
            assert len(died.probe.runs) == started_before, where
            fresh_runs = len(out["fresh"].runs)
            # never a second start, and every start was announced (and every announce a start)
            assert fresh_runs + len(died.probe.runs) <= 1, where
            assert out["starts"] == fresh_runs + len(died.probe.runs), where
            assert out["starts"] <= 1 and out["ends"] <= 1, where
            seen = out["seen"]
            record = out["record"]
            if seen is None:
                kinds.add("request in flight")
                assert fresh_runs == 1, where
                assert (record["status"], out["state"]) == ("ok", "done"), where
                assert out["ends"] == 1, where  # the resumed process owns run/end
            elif seen["status"] == "running":
                kinds.add("running record")
                assert fresh_runs == 0, where
                assert (record["status"], record["reason"]) == ("failed", "run_lost")
                assert out["state"] == "apology", where
                assert out["ends"] == 0, (
                    where
                )  # the known limit: the mark was in memory
            else:
                kinds.add("final record")
                assert fresh_runs == 0 and record["status"] == "ok", where
                assert out["state"] == "done", where
        full = tmp_path / "full"
        full.mkdir()
        with EventLog(full) as log:
            api, _targets, _log, probe, _llm, _extra = _assemble(
                Path("."), existing_log=log
            )
            try:
                cid = _play(api)
            finally:
                api.close()
            assert _types(log, cid) == golden_types
            assert _count(log, cid, EventType.RUN_END) == 1 and len(probe.runs) == 1
        # non-vacuity: every kind of offset was visited
        assert kinds == {"request in flight", "running record", "final record"}, kinds


# ---- an exit with a run thread alive and the log open ----------------------------------------------

_CHILD = textwrap.dedent(
    """
    import sys, time
    from pathlib import Path

    sys.path.insert(0, sys.argv[2])
    sys.path.insert(0, sys.argv[3])

    from fsm_llm import API
    from fsm_llm.aio import aoffload
    from fsm_llm.core.codes import Status
    from fsm_llm.core.definitions.executable import Contract
    from fsm_llm.core.results import RunResult
    from fsm_llm.engine.context import RunContext
    from fsm_llm.engine.executables import ExecutableRegistry
    from fsm_llm.engine.executables.code_backed import CodeBackedExecutable
    from fsm_llm.engine.executables.runs import ExecutableTargets
    from fsm_llm.runtime import EventLog
    from fsm_llm.testing import ScriptedLLM

    from tests.test_fsm_llm.test_call_delegated import _initial_call
    from tests.test_fsm_llm.test_call_state import _call, _parent


    class Hold(CodeBackedExecutable):
        def __init__(self):
            super().__init__(
                id="hold.run",
                kind="pattern",
                impl="child.Hold",
                contract=Contract(inputs=["topic"], outputs=[]),
            )

        async def execute(self, inputs, objects, context, identity):
            # a pattern runs its agent on an executor thread (aoffload) and keeps going
            await aoffload(time.sleep, 1.0)
            return RunResult(status=Status.OK, answer="late", identity=identity)


    log = EventLog(Path(sys.argv[1]))
    registry = ExecutableRegistry()
    registry.register(Hold())
    targets = ExecutableTargets(registry, RunContext(event_log=log))
    definition = _parent(
        _call(target="hold.run", inputs=["topic"], returns={"answer": "answer"}),
        response_instructions="",
    )
    api = API.from_definition(
        _initial_call(definition),
        llm_interface=ScriptedLLM(default_response="r"),
        targets=targets,
        event_log=log,
    )
    cid, _ = api.start_conversation({"topic": "why"})
    api.advance(cid)  # the record, run/start, the thread
    print(cid, flush=True)
    # exit now: no close(), the daemon thread and the open writer are still alive
    """
)


class TestAnExitWithARunThreadAlive:
    def test_the_log_reads_clean_after_an_exit_with_a_live_run_thread(
        self, tmp_path: Path
    ) -> None:
        """The child starts a delegated run, then the interpreter exits with the run thread alive
        and the log writer open. The run thread writes no event (settle alone writes run events),
        so the log is whatever the engine thread appended before the exit: every line a whole
        record, the last one ``run/start``, and the whole file folds with no ``LogCorruptError``.
        Measured on 3.12 (scratch probe, progress.md): the exit WAITS for the executor thread of
        a pattern that is still running its agent (``aoffload`` uses the loop's default executor,
        which the interpreter joins at exit), so a detached pattern delays process exit by what
        is left of its agent run; a run that only awaits is killed with its daemon thread."""
        root = tmp_path / "child"
        done = subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                _CHILD,
                str(root),
                str(_ROOT),
                str(_ROOT / "src"),
            ],
            capture_output=True,
            text=True,
            timeout=240,
            cwd=_ROOT,
            env={
                "PATH": os.environ.get("PATH", ""),
                "OLLAMA_API_BASE": "http://127.0.0.1:9",
                "LITELLM_LOCAL_MODEL_COST_MAP": "True",
                **(
                    {"SYSTEMROOT": os.environ["SYSTEMROOT"]}
                    if "SYSTEMROOT" in os.environ
                    else {}
                ),
            },
        )
        assert done.returncode == 0, done.stderr
        cid = done.stdout.strip().splitlines()[-1]
        path = root / "logs" / f"{cid}.jsonl"
        raw = path.read_bytes()
        assert raw.endswith(b"\n")  # no torn last line
        lines = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
        assert [e["seq"] for e in lines] == list(range(1, len(lines) + 1))
        assert lines[-1]["type"] == EventType.RUN_START.value
        assert not any(e["type"] == EventType.RUN_END.value for e in lines)
        with EventLog(root) as log:
            try:
                folded = fold_stack(log.read(cid), log.get_json)
            except (
                LogCorruptError
            ) as error:  # pragma: no cover - the failure being probed
                pytest.fail(f"the log is corrupt after the exit: {error}")
            assert folded is not None and folded.depth == 1
