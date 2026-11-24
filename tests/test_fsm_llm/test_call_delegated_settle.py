"""``settle`` for a delegated call, the run events and the wait (plan baa37f40, step K7.4, D-052).

A delegated request (``test_call_delegated.py``) is started by ``settle`` AFTER the turn commit,
from the parent's own definition and context, through the run host: the host's ``start`` consumes
the reservation of Pass 1 (a forged record has none and starts nothing), ``run/start`` is
appended with ``checkpoint=True`` BEFORE the daemon thread exists, ``run/end`` is written by
settle on the ENGINE thread when the committed record is final (a worker never writes a run
event), and ``aadvance`` makes ONE bounded wait at its end, outside every lock, so a run loop does
not spin. Real ``ExecutableTargets`` and a real ``RunHost``: the target is a code-backed stub
that records what it was given and parks until released. Rules of the rows (LESSONS ace7a8c8,
D-048): no ``Thread.join``, events and bounded waits instead of sleeps, a thread-order row never
depends on a scripted reply order, the one stress-free design. Each guard row names the mutant it
kills (census D1, D3, D7, D8).
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

import fsm_llm
from fsm_llm import API
from fsm_llm.aio import SYNC, aoffload
from fsm_llm.core.codes import Status
from fsm_llm.core.definitions import FSMError
from fsm_llm.core.definitions.executable import Contract
from fsm_llm.core.results import Identity, RunResult
from fsm_llm.engine.api import delegated
from fsm_llm.engine.api.calls import settle
from fsm_llm.engine.context import RunContext
from fsm_llm.engine.executables import ExecutableRegistry
from fsm_llm.engine.executables.code_backed import CodeBackedExecutable
from fsm_llm.engine.executables.runs import ExecutableTargets, RunHost
from fsm_llm.engine.kinds.targets import (
    DEFAULT_DELEGATED_MAX_SECONDS,
    MAX_DELEGATED_MAX_SECONDS,
    RunPoll,
)
from fsm_llm.runtime import (
    PRODUCED_TYPES,
    EventLog,
    EventType,
    PrivacyLevel,
)
from fsm_llm.runtime.constants import ENVELOPE_KEYS, LEVEL_KEY
from fsm_llm.runtime.privacy import SHAPE_FIELDS
from fsm_llm.runtime.stores import fold_stack
from tests.conftest import current_timing_factor
from tests.test_fsm_llm.test_async_cancel import _gate_is_free
from tests.test_fsm_llm.test_call_delegated import _initial_call
from tests.test_fsm_llm.test_call_settle import _Gate
from tests.test_fsm_llm.test_call_state import _call, _parent

# seconds: a wait that would hang fails instead (scaled under a tracer)
_BOUND = 20 * current_timing_factor()
_TARGET = "probe.run"
_ROOT = Path(__file__).resolve().parents[2]
_RUN_TYPES = (EventType.RUN_START.value, EventType.RUN_END.value)


def test_the_module_under_test_is_this_checkout() -> None:
    assert Path(fsm_llm.__file__).resolve().is_relative_to(_ROOT / "src")


@pytest.fixture(autouse=True)
def _short_wait(monkeypatch):
    """The wait at the end of ``aadvance`` is short in a test (its value is pinned by a row)."""
    monkeypatch.setattr(delegated, "DELEGATED_WAIT_SECONDS", 0.05)


class _Probe(CodeBackedExecutable):
    """A code-backed pattern: records what it was given, parks until released, then returns."""

    def __init__(
        self,
        id: str = _TARGET,
        *,
        hold: bool = False,
        raises: BaseException | None = None,
        result: RunResult | None = None,
    ) -> None:
        super().__init__(
            id=id,
            kind="pattern",
            impl="tests.probe.Probe",
            contract=Contract(inputs=["topic"], outputs=["findings"]),
        )
        self.hold = hold
        self.raises = raises
        self.result = result
        self.runs: list[dict[str, Any]] = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.watch: tuple[EventLog, str] | None = None

    async def execute(self, inputs, objects, context, identity: Identity) -> RunResult:
        seen: dict[str, Any] = {
            "inputs": dict(inputs),
            "context": context,
            "thread": threading.current_thread().name,
            "sync": SYNC.get(),
            "log_types": None,
        }
        if self.watch is not None:  # what is on disk when the worker starts
            log, root = self.watch
            seen["log_types"] = [e["type"] for e in log.read(root)]
        self.runs.append(seen)
        self.entered.set()
        if self.hold:
            await aoffload(self.release.wait, _BOUND)
        if self.raises is not None:
            raise self.raises
        if self.result is not None:
            return self.result
        return RunResult(
            status=Status.OK,
            outputs={"findings": ["f1"]},
            answer="the answer",
            identity=identity,
        )


@dataclass
class Flow:
    api: API
    cid: str
    frame: str
    probe: _Probe
    targets: ExecutableTargets
    log: EventLog | None
    llm: Any
    root: Path
    extra: list[_Probe] = field(default_factory=list)

    def raw(self) -> dict[str, Any]:
        return self.api.fsm_manager.copy_raw_context_data(self.frame)

    def record(self) -> Any:
        return self.raw().get("call_result")

    def events(self, *types: str) -> list[dict[str, Any]]:
        assert self.log is not None
        return [e for e in self.log.read(self.cid) if e["type"] in types]

    def seed(self, record: dict[str, Any]) -> None:
        self.api.fsm_manager.update_conversation_context(
            self.frame, {"call_result": record}
        )

    async def step(self):
        return await self.api.aadvance(self.cid)

    def release_all(self) -> None:
        for probe in (self.probe, *self.extra):
            probe.release.set()


def _forged(handle: str, **over: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "status": "running",
        "state": "work",
        "target": _TARGET,
        "template_identity": "forged",
        "instance_identity": None,
        "mode": "delegated",
        "handle": handle,
        "deadline": time.time() + 600,
        "request": {
            "target": _TARGET,
            "inputs": {"topic": "why"},
            "max_steps": None,
            "max_seconds": None,
        },
    }
    record.update(over)
    return record


def _assemble(
    tmp_path: Path,
    *,
    probe: _Probe | None = None,
    extra: tuple[_Probe, ...] = (),
    log: bool = True,
    existing_log: EventLog | None = None,
    privacy: PrivacyLevel | str = PrivacyLevel.VALUES,
    config: dict[str, Any] | None = None,
    llm: Any = None,
    **state_over: Any,
):
    probe = _Probe() if probe is None else probe
    registry = ExecutableRegistry()
    for executable in (probe, *extra):
        registry.register(executable)
    event_log = (
        existing_log
        if existing_log is not None
        else EventLog(tmp_path / "logs", privacy=privacy)
        if log
        else None
    )
    # the host context carries the log so a worker that reached for it could write (D8)
    targets = ExecutableTargets(registry, RunContext(event_log=event_log))
    llm = _Gate(default_response="r") if llm is None else llm
    state_over.setdefault("response_instructions", "")
    definition = _parent(
        _call(
            target=_TARGET,
            inputs=["topic"],
            returns={"answer": "answer", "found": "findings"},
        )
        if config is None
        else config,
        **state_over,
    )
    api = API.from_definition(
        _initial_call(definition),
        llm_interface=llm,
        targets=targets,
        **({} if event_log is None else {"event_log": event_log}),
    )
    return api, targets, event_log, probe, llm, extra


def run_delegated(log: EventLog) -> str:
    """One delegated call run to its end on *log* (the sync names); returns the root id. The
    scenario the event-log tests add to their union of produced types (``run/start`` and
    ``run/end``)."""
    api, _targets, _log, _probe, _llm, _extra = _assemble(Path("."), existing_log=log)
    try:
        cid, _ = api.start_conversation({"topic": "why"})
        api.run_until_terminal(cid, max_steps=30)
        return cid
    finally:
        api.close()  # the shared log stays open: the API does not own it


@pytest.fixture
def make_flow(tmp_path):
    made: list[Flow] = []

    async def build(**kw: Any) -> Flow:
        context = kw.pop("context", None)
        api, targets, log, probe, llm, extra = _assemble(tmp_path, **kw)
        cid, _ = await api.astart_conversation({"topic": "why", **(context or {})})
        frame = api.conversation_stacks[cid][0].conversation_id
        if log is not None:
            probe.watch = (log, cid)
        flow = Flow(
            api, cid, frame, probe, targets, log, llm, tmp_path / "logs", list(extra)
        )
        made.append(flow)
        return flow

    yield build
    for flow in made:
        flow.release_all()
        try:
            flow.api.close()
        finally:
            if flow.log is not None:
                flow.log.close()


class _Spy:
    """Records every ``EventLog.append`` of a run type: the type, the thread and the checkpoint
    flag; ``order`` keeps every append of any type."""

    def __init__(self, log: EventLog) -> None:
        self.rows: list[tuple[str, str, Any]] = []
        self.order: list[str] = []
        real = log.append

        def append(conversation_id, type, **kwargs):
            value = getattr(type, "value", type)
            self.order.append(value)
            if value in _RUN_TYPES:
                self.rows.append(
                    (value, threading.current_thread().name, kwargs.get("checkpoint"))
                )
            return real(conversation_id, type, **kwargs)

        log.append = append  # type: ignore[method-assign]


class TestTheHappyPath:
    async def test_the_run_goes_through_settle_and_the_returns_land_once(
        self, make_flow
    ):
        """Kills: a delegated record nothing starts (the run loop would wait for ever)."""
        flow = await make_flow()
        results = await flow.api.arun_until_terminal(flow.cid, max_steps=30)
        assert results[-1].ended
        data = flow.api.get_data(flow.cid)
        assert data["answer"] == "the answer" and data["found"] == ["f1"]
        assert data["call_result"]["status"] == "ok"
        assert len(flow.probe.runs) == 1
        assert flow.probe.runs[0]["inputs"] == {"topic": "why"}
        assert [e["type"] for e in flow.events(*_RUN_TYPES)] == list(_RUN_TYPES)

    async def test_no_frame_is_ever_pushed_for_a_delegated_record(
        self, make_flow, monkeypatch
    ):
        """Rule (b) re-requests only ``frame`` records: a delegated one is never a crashed apply.
        Kills (D7): ``_is_request`` accepting the delegated mode (a child pushed over the run)."""
        flow = await make_flow()

        async def boom(*a, **k):
            raise AssertionError("a delegated call pushes no frame")

        monkeypatch.setattr(flow.api, "apush_fsm", boom)
        depths = []
        for _ in range(4):
            await flow.step()
            depths.append(flow.api.get_stack_depth(flow.cid))
            if flow.record()["status"] != "running":
                break
        assert set(depths) == {1}

    def test_the_sync_names_drive_the_same_run_with_no_event_loop(self, tmp_path):
        api, _targets, log, probe, _llm, _extra = _assemble(tmp_path)
        try:
            cid, _ = api.start_conversation({"topic": "why"})
            results = api.run_until_terminal(cid, max_steps=30)
            assert results[-1].ended
            assert api.get_data(cid)["answer"] == "the answer"
            assert probe.runs[0]["sync"] is False  # the worker has its own loop
            assert probe.runs[0]["thread"].startswith("fsm-llm-run-")
        finally:
            api.close()
            assert log is not None
            log.close()

    async def test_the_run_works_with_the_log_off(self, make_flow):
        flow = await make_flow(log=False)
        await flow.api.arun_until_terminal(flow.cid, max_steps=30)
        assert flow.api.get_data(flow.cid)["answer"] == "the answer"

    async def test_a_record_of_an_unknown_mode_is_still_refused(self, make_flow):
        flow = await make_flow()
        flow.seed(_forged("run-x", mode="weird"))
        step = await flow.step()
        assert flow.record()["reason"] == "unknown_target"
        assert step.state_after == "apology" and flow.probe.runs == []


class TestTheStartIsAfterTheCommit:
    async def test_a_rolled_back_turn_starts_nothing_and_its_reservation_never_starts(
        self, make_flow, monkeypatch
    ):
        """A Pass 2 failure after the record: no record, nothing started, no run event; the stale
        reservation of the rolled-back turn is never started and the next turn mints a new
        handle. Kills: a start from the reservation, or one before the turn committed."""
        flow = await make_flow(response_instructions="Say something")
        claimed: list[str] = []
        real_claim = flow.targets.claim

        def claim(handle, *, conversation_id):
            claimed.append(handle)
            return real_claim(handle, conversation_id=conversation_id)

        monkeypatch.setattr(flow.targets, "claim", claim)
        real = flow.llm.agenerate_response

        async def boom(request):
            raise FSMError("provider is down")

        monkeypatch.setattr(flow.llm, "agenerate_response", boom)
        with pytest.raises(FSMError):
            await flow.step()
        assert "call_result" not in flow.raw()
        assert flow.probe.runs == [] and flow.events(*_RUN_TYPES) == []
        monkeypatch.setattr(flow.llm, "agenerate_response", real)
        await flow.step()
        assert len(claimed) == 2 and claimed[0] != claimed[1]
        assert flow.record()["handle"] == claimed[1]
        assert flow.probe.entered.wait(_BOUND)
        assert flow.targets.host.poll(claimed[0], conversation_id=flow.frame) == (
            RunPoll(False, None)
        )  # reserved, never started
        assert len(flow.probe.runs) == 1

    async def test_a_cancelled_turn_starts_nothing_and_the_next_turn_runs_it(
        self, make_flow
    ):
        """A cancel inside the turn (the model call of Pass 2) rolls everything back: no run
        starts, the gate is free and the next entry starts the run once."""
        flow = await make_flow(response_instructions="Say something")
        flow.llm.arm()
        task = asyncio.create_task(flow.step())
        await asyncio.wait_for(flow.llm.entered.wait(), _BOUND)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert flow.probe.runs == [] and flow.events(*_RUN_TYPES) == []
        assert _gate_is_free(flow.api, flow.cid)
        await flow.step()
        assert flow.probe.entered.wait(_BOUND)
        assert len(flow.probe.runs) == 1

    async def test_a_cancel_between_the_commit_and_the_start_heals_on_the_next_entry(
        self, make_flow, monkeypatch
    ):
        """The reservation of Pass 1 survives a settle that raised before the start; the next
        entry's settle starts the SAME handle once. The probe parks (``hold``), so the record
        is still the running one when it is read: a probe that returns at once can land the
        final record, which carries no handle, before the read (a CI-only race)."""
        flow = await make_flow(probe=_Probe(hold=True))
        real = delegated.settle_delegated
        calls = {"n": 0}

        async def once_then_real(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise asyncio.CancelledError
            return await real(*args, **kwargs)

        monkeypatch.setattr("fsm_llm.engine.api.calls.settle_delegated", once_then_real)
        with pytest.raises(asyncio.CancelledError):
            await flow.step()
        handle = flow.record()["handle"]
        assert flow.probe.runs == []
        await flow.step()
        assert flow.probe.entered.wait(_BOUND)
        assert flow.record()["handle"] == handle and len(flow.probe.runs) == 1

    async def test_the_start_comes_from_the_parents_own_state_not_the_record(
        self, make_flow
    ):
        """A forged record names another target and a handle this conversation really holds:
        the run that starts is the one the STATE declares. Kills (D3): a start from the record's
        target."""
        evil = _Probe("evil.run")
        flow = await make_flow(extra=(evil,))
        assert flow.targets.claim("run-mine", conversation_id=flow.frame)
        flow.seed(_forged("run-mine", target="evil.run"))
        await flow.step()
        assert flow.probe.entered.wait(_BOUND)
        assert len(flow.probe.runs) == 1 and evil.runs == []

    async def test_a_forged_record_with_an_unreserved_handle_starts_nothing(
        self, make_flow
    ):
        """Kills (D3): ``settle`` starting from the record's handle with no reservation (the
        host's ``start`` skipping its check). The record ends ``failed/run_lost`` in the turn."""
        flow = await make_flow()
        flow.seed(_forged("run-" + "f" * 32))
        step = await flow.step()
        record = flow.record()
        assert (record["status"], record["reason"]) == ("failed", "run_lost")
        assert flow.probe.runs == [] and flow.events(*_RUN_TYPES) == []
        assert step.state_after == "apology"

    async def test_a_forged_record_copying_another_conversations_handle_adopts_nothing(
        self, make_flow
    ):
        """Kills (D3): an unscoped reservation or poll. The other conversation's reservation is
        intact afterwards and still starts."""
        flow = await make_flow()
        assert flow.targets.host.claim("run-stolen", conversation_id="someone-else")
        flow.seed(_forged("run-stolen"))
        await flow.step()
        assert flow.record()["reason"] == "run_lost" and flow.probe.runs == []
        assert flow.targets.host.poll(
            "run-stolen", conversation_id="someone-else"
        ) == RunPoll(False, None)

        async def body(token):
            return RunResult(status=Status.OK)

        assert flow.targets.host.start(
            "run-stolen", body, conversation_id="someone-else"
        )

    async def test_a_restore_style_record_with_no_host_state_is_run_lost(
        self, make_flow
    ):
        """A restart, a restored snapshot or a fork: the host never saw the handle."""
        flow = await make_flow()
        await flow.step()  # a real request and a started run
        flow.probe.release.set()
        handle = flow.record()["handle"]
        fresh = RunHost()
        flow.targets._host = fresh  # the "restarted" host knows nothing
        assert fresh.poll(handle, conversation_id=flow.frame) is None
        flow.api._call_marks.runs.clear()  # and the process forgot its marks
        step = await flow.step()
        assert flow.record()["reason"] == "run_lost" and step.state_after == "apology"

    async def test_a_started_run_whose_mark_is_gone_is_left_alone(self, make_flow):
        """The host still holds the run (the process lost only the in-memory mark): settle does
        not start it twice and does not end it. Kills: ``start`` false read as a lost run."""
        flow = await make_flow(probe=_Probe(hold=True))
        await flow.step()
        assert flow.probe.entered.wait(_BOUND)
        flow.api._call_marks.runs.clear()
        await settle(flow.api, flow.cid)
        assert flow.record()["status"] == "running"
        assert len(flow.probe.runs) == 1

    async def test_a_target_that_no_longer_resolves_ends_unknown_target_and_starts_nothing(
        self, make_flow, monkeypatch
    ):
        """The executor resolved it in Pass 1; settle resolves it AGAIN from the parent's own
        state, and a registry that no longer answers ends the call (the reservation is never
        started)."""
        flow = await make_flow()
        real = flow.targets.resolve_run
        asked = {"n": 0}

        def once(*args, **kwargs):
            asked["n"] += 1
            return real(*args, **kwargs) if asked["n"] == 1 else None

        monkeypatch.setattr(flow.targets, "resolve_run", once)
        await flow.step()
        assert asked["n"] >= 2  # the executor, then settle
        assert flow.record()["reason"] == "unknown_target"
        assert flow.probe.runs == [] and flow.events(*_RUN_TYPES) == []


class TestWhatTheChildReceives:
    async def test_the_child_receives_only_the_declared_inputs(self, make_flow):
        """Kills (D1): inputs built from the whole parent context (the child would see
        ``_approval_granted`` and the secret)."""
        flow = await make_flow(
            context={
                "_approval_granted": {"call": "x"},
                "api_key": "sk-secret-value",
                "unrelated": "do not copy me",
            }
        )
        await flow.api.arun_until_terminal(flow.cid, max_steps=30)
        assert len(flow.probe.runs) == 1
        given = flow.probe.runs[0]["inputs"]
        assert given == {"topic": "why"}
        assert "sk-secret-value" not in repr(flow.probe.runs[0]["context"])

    async def test_the_child_runs_under_its_own_token_and_the_bounded_limit(
        self, make_flow
    ):
        flow = await make_flow(
            config=_call(
                target=_TARGET,
                inputs=["topic"],
                returns={"answer": "answer"},
                max_seconds=30,
            )
        )
        base = flow.targets.context
        await flow.api.arun_until_terminal(flow.cid, max_steps=30)
        context = flow.probe.runs[0]["context"]
        assert context.parent == flow.frame
        assert context.max_seconds == 30.0
        assert context.cancel is not base.cancel and not base.cancel.cancelled

    @pytest.mark.parametrize(
        ("given", "want"),
        [
            (None, DEFAULT_DELEGATED_MAX_SECONDS),
            (10**7, MAX_DELEGATED_MAX_SECONDS),
        ],
        ids=["default", "capped"],
    )
    async def test_the_default_and_the_cap_reach_the_child(
        self, make_flow, given, want
    ):
        extra = {} if given is None else {"max_seconds": given}
        flow = await make_flow(
            config=_call(
                target=_TARGET,
                inputs=["topic"],
                returns={"answer": "answer"},
                **extra,
            )
        )
        await flow.api.arun_until_terminal(flow.cid, max_steps=30)
        assert flow.probe.runs[0]["context"].max_seconds == want


class TestTheRunEvents:
    async def test_run_start_is_a_checkpoint_on_disk_before_the_worker_runs(
        self, make_flow
    ):
        """Kills (D8): ``run/start`` written after the thread started, or with no fsync."""
        flow = await make_flow()
        spy = _Spy(flow.log)
        await flow.step()
        assert flow.probe.entered.wait(_BOUND)
        assert flow.probe.runs[0]["log_types"].count("run/start") == 1
        starts = [r for r in spy.rows if r[0] == "run/start"]
        assert starts == [("run/start", threading.current_thread().name, True)]
        [event] = flow.events("run/start")
        assert event["handle"] == flow.record()["handle"]
        assert event["source"] == "run" and event["parent"] == flow.frame
        assert event["target"] == _TARGET and event["frame"] == 0
        assert event["template_identity"] == flow.probe.template_identity
        assert event["instance_identity"] is None

    async def test_a_worker_never_writes_a_run_event(self, make_flow):
        """Every ``run/*`` append happens on the engine thread. Kills (D8): a worker writing
        ``run/end`` (here through the host context's log)."""
        flow = await make_flow()
        spy = _Spy(flow.log)
        await flow.api.arun_until_terminal(flow.cid, max_steps=30)
        assert [r[0] for r in spy.rows] == list(_RUN_TYPES)
        names = {r[1] for r in spy.rows}
        assert not any(n.startswith("fsm-llm-run-") for n in names)
        assert [e["type"] for e in flow.events(*_RUN_TYPES)] == list(_RUN_TYPES)

    async def test_run_end_is_written_once_when_the_committed_record_is_final(
        self, make_flow
    ):
        flow = await make_flow()
        await flow.api.arun_until_terminal(flow.cid, max_steps=30)
        [start] = flow.events("run/start")
        [end] = flow.events("run/end")
        assert end["handle"] == start["handle"] and end["status"] == "ok"
        assert end["target"] == _TARGET and end["reason"] is None
        assert end["seq"] > start["seq"]
        for _ in range(3):  # more settles write nothing: the mark is gone
            await settle(flow.api, flow.cid)
        assert len(flow.events("run/end")) == 1

    async def test_a_turn_that_rolls_back_writes_no_run_end(
        self, make_flow, monkeypatch
    ):
        """The finishing turn's Pass 2 fails: the record is back to ``running``, so nothing is
        logged; the next turn lands it and run/end is written exactly once."""
        flow = await make_flow()
        await flow.step()  # the run starts
        assert flow.probe.entered.wait(_BOUND)
        assert flow.targets.host.wait(flow.record()["handle"], _BOUND)
        real = flow.llm.agenerate_response

        async def boom(request):
            raise FSMError("provider is down")

        monkeypatch.setattr(flow.llm, "agenerate_response", boom)
        with pytest.raises(FSMError):
            await flow.step()
        assert flow.record()["status"] == "running"
        assert flow.events("run/end") == []
        monkeypatch.setattr(flow.llm, "agenerate_response", real)
        await flow.step()
        assert flow.record()["status"] == "ok"
        assert len(flow.events("run/end")) == 1

    async def test_a_frame_with_a_turn_in_flight_is_not_judged(
        self, make_flow, monkeypatch
    ):
        """A record that looks final while a turn of the frame runs may still roll back: the end
        is judged after the turn. Kills: a flush that reads the context of a frame in flight."""
        flow = await make_flow(probe=_Probe(hold=True))
        await flow.step()
        assert flow.probe.entered.wait(_BOUND)
        flow.seed({"status": "ok", "state": "work", "reason": None, "target": _TARGET})
        real = flow.api.fsm_manager.peek_turn
        monkeypatch.setattr(
            flow.api.fsm_manager,
            "peek_turn",
            lambda frame_id: (lambda peek: None if peek is None else (peek[0], True))(
                real(frame_id)
            ),
        )
        delegated.flush_run_ends(flow.api, flow.cid)
        assert flow.events("run/end") == []
        monkeypatch.setattr(flow.api.fsm_manager, "peek_turn", real)
        delegated.flush_run_ends(flow.api, flow.cid)
        assert [e["status"] for e in flow.events("run/end")] == ["ok"]

    async def test_a_deadline_aborts_the_call_cancels_the_run_and_logs_the_end(
        self, make_flow
    ):
        flow = await make_flow(
            probe=_Probe(hold=True),
            config=_call(
                target=_TARGET,
                inputs=["topic"],
                returns={"answer": "answer"},
                max_seconds=30,
            ),
        )
        clock = SimpleNamespace(now=time.time())
        with patch(
            "fsm_llm.engine.kinds.call.time", SimpleNamespace(time=lambda: clock.now)
        ):
            await flow.step()
            assert flow.probe.entered.wait(_BOUND)
            clock.now += 31.0
            step = await flow.step()
        record = flow.record()
        assert (record["status"], record["reason"]) == ("aborted", "budget_spent")
        assert step.state_after == "apology"
        assert flow.probe.runs[0]["context"].cancel.cancelled
        [end] = flow.events("run/end")
        assert (end["status"], end["reason"]) == ("aborted", "budget_spent")

    async def test_a_worker_error_ends_failed_with_the_type_only(self, make_flow):
        """Kills (D6): the exception text reaching the record or the log."""
        flow = await make_flow(probe=_Probe(raises=ValueError("secret sk-123456")))
        await flow.api.arun_until_terminal(flow.cid, max_steps=30)
        assert not flow.api.get_data(flow.cid).get("answer")
        record = flow.api.get_data(flow.cid)["call_result"]
        assert (record["status"], record["reason"]) == ("failed", "unexpected_error")
        [end] = flow.events("run/end")
        assert (end["status"], end["reason"]) == ("failed", "unexpected_error")
        log_text = "\n".join(
            p.read_text(encoding="utf-8") for p in (flow.root / "logs").glob("*.jsonl")
        )
        assert "sk-123456" not in log_text and "sk-123456" not in repr(record)

    async def test_the_events_sit_at_the_top_frame_and_the_fold_is_unchanged(
        self, make_flow
    ):
        flow = await make_flow()
        await flow.api.arun_until_terminal(flow.cid, max_steps=30)
        events = list(flow.log.read(flow.cid))
        assert {e["frame"] for e in events if e["type"] in _RUN_TYPES} == {0}
        folded = fold_stack(iter(events), flow.log.get_json)
        assert folded is not None and folded.depth == 1

    def test_the_run_types_are_produced_and_have_a_shape_allow_list(self):
        for kind in (EventType.RUN_START, EventType.RUN_END):
            assert kind in PRODUCED_TYPES and kind in SHAPE_FIELDS
            assert not any(name.endswith("_ref") for name in SHAPE_FIELDS[kind])
        assert EventType.EXECUTABLE_RESOLVE not in PRODUCED_TYPES

    async def test_at_the_shape_level_a_run_event_carries_only_its_allow_list(
        self, make_flow
    ):
        flow = await make_flow(privacy=PrivacyLevel.SHAPE)
        await flow.api.arun_until_terminal(flow.cid, max_steps=30)
        envelope = {*ENVELOPE_KEYS, LEVEL_KEY}
        for event in flow.events(*_RUN_TYPES):
            kind = EventType.from_string(event["type"])
            assert set(event) - envelope <= SHAPE_FIELDS[kind]
            assert not any(k.endswith("_ref") for k in event)


def _lock_free(lock: Any) -> bool:
    """True when ANOTHER thread could take *lock* now (a re-entrant lock held by the caller's own
    thread would let the caller in, so the probe runs on a worker thread)."""
    if lock.acquire(blocking=False):
        lock.release()
        return True
    return False


class TestTheBoundedWait:
    async def test_aadvance_waits_once_bounded_outside_every_lock(
        self, make_flow, monkeypatch
    ):
        """One ``RunHost.wait`` per ``aadvance`` that leaves a started run behind, with the
        named bound, on a worker thread, holding no stack lock, no manager lock and no turn
        gate. Kills (D7): the wait inside a lock."""
        flow = await make_flow(probe=_Probe(hold=True))
        seen: list[dict[str, Any]] = []
        real = RunHost.wait
        main = threading.current_thread()

        def spy(self, handle, timeout_seconds):
            seen.append(
                {
                    "timeout": timeout_seconds,
                    "stack": _lock_free(flow.api._stack_lock),
                    "manager": _lock_free(flow.api.fsm_manager._lock),
                    "gate": _gate_is_free(flow.api, flow.cid),
                    "worker": threading.current_thread() is not main,
                }
            )
            return real(self, handle, timeout_seconds)

        monkeypatch.setattr(RunHost, "wait", spy)
        await flow.step()
        assert len(seen) == 1
        assert seen[0] == {
            "timeout": delegated.DELEGATED_WAIT_SECONDS,
            "stack": True,
            "manager": True,
            "gate": True,
            "worker": True,
        }
        await flow.step()  # a poll turn
        assert len(seen) == 2

    async def test_a_message_turn_never_waits(self, make_flow, monkeypatch):
        flow = await make_flow(probe=_Probe(hold=True))
        await flow.step()
        waits = []
        monkeypatch.setattr(
            RunHost, "wait", lambda self, h, t: waits.append(h) or False
        )
        await flow.api.aconverse("hello", flow.cid)
        assert waits == []

    async def test_no_wait_without_a_started_run_or_for_a_launcher_with_no_wait(
        self, make_flow, monkeypatch
    ):
        waits = []
        monkeypatch.setattr(
            RunHost, "wait", lambda self, h, t: waits.append(h) or False
        )
        flow = await make_flow()
        flow.seed({"status": "ok", "state": "work", "reason": None, "target": _TARGET})
        await flow.step()  # a final record: nothing started, nothing to wait for
        assert waits == []
        monkeypatch.delattr(ExecutableTargets, "wait")
        flow2 = await make_flow(probe=_Probe(hold=True))
        await flow2.step()  # a launcher with no `wait` is never waited for
        assert waits == []

    async def test_the_wait_does_not_block_the_event_loop(self, make_flow, monkeypatch):
        """Kills (D7): the blocking wait run inline on the loop thread (the ticker starves)."""
        monkeypatch.setattr(delegated, "DELEGATED_WAIT_SECONDS", 0.4)
        flow = await make_flow(probe=_Probe(hold=True))
        ticks = 0
        running = True

        async def ticker():
            nonlocal ticks
            while running:
                ticks += 1
                await asyncio.sleep(0.01)

        task = asyncio.create_task(ticker())
        try:
            await flow.step()
        finally:
            running = False
            await task
        assert ticks >= 5

    async def test_a_parked_run_does_not_make_aadvance_hang(self, make_flow):
        flow = await make_flow(probe=_Probe(hold=True))
        started = time.monotonic()
        await flow.step()
        await flow.step()
        assert time.monotonic() - started < _BOUND
