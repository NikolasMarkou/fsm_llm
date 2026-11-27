"""A cancel at EVERY lane hop of a logged async turn (plan 3f4530d3 step 8b, decisions D-015
and D-016, roadmap row 1.18).

A hop is one ``Lanes.lane_call`` of the engine: the first writer open and ``turn/start``, every
model request and reply, the extraction blob, ``turn/end``. A clean run counts them (N); for each
k in 1..N the lane job of the k-th hop is PARKED on an ``Event``, the turn task is cancelled while
it waits there, and the job is then released. The rows assert what the cancel left: the state of
the conversation, the log (a rolled-back turn, no request without its end, ``turn/end`` after
``turn/start``), the gate, a working second turn and that the orphan job RAN after the cancel.
The last hop (``turn/end`` of a committed turn) raises the cancel too: the turn stays committed and
the caller gets no reply (D-019; D-016 absorbed it, and the cancel was then lost for the rest of a
run).
Nothing reads a clock: a wait on a thread is an ``Event`` with a cap that only ends a hang.
"""

from __future__ import annotations

import asyncio
import copy
import json
import threading
from collections.abc import Awaitable
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from fsm_llm import API
from fsm_llm.core.definitions import FSMDefinition, FSMError
from fsm_llm.engine import events as engine_events
from fsm_llm.llm import LiteLLMInterface
from fsm_llm.runtime import EventLog, LogWriteError
from fsm_llm.runtime import scope as scope_module
from fsm_llm.runtime.lanes import Lanes
from fsm_llm.runtime.stores import fold_session, read_events
from fsm_llm.runtime.stores.jsonl import JsonlWriter
from fsm_llm.testing import ScriptedLLM
from tests.conftest import current_timing_factor
from tests.test_fsm_llm import event_log_child as child
from tests.test_fsm_llm import kind_corpus as kc
from tests.test_fsm_llm.event_log_support import (
    CHILD_MODULE,
    _reply,
    child_process,
    reopen_after_kill,
    wait_ready,
)
from tests.test_fsm_llm.test_async_cancel import (
    _cancel,
    _gate_is_free,
    _next,
    _Parking,
    _prepare,
    _seen,
)
from tests.test_fsm_llm.test_async_engine import (
    _CID,
    _SCENARIOS,
    _collect,
    _stream_api,
)
from tests.test_fsm_llm.test_event_log_census import (
    _TALKING_KID,
    _chat,
    _llm,
    _push_api,
)
from tests.test_fsm_llm.test_event_log_jsonl import can_lock_in_child
from tests.test_fsm_llm.test_event_log_lanes import (
    _pool_of,
    _queued,
    _run_in_thread,
    _until,
)

CAP_SECONDS = 20.0  # only ends a hang


class Hops:
    """Counts the ``lane_call`` hops of every log and parks the ``park_at``-th one's job."""

    def __init__(self) -> None:
        self.count = 0
        self.park_at = 0
        self.entered, self.gate, self.ran = (threading.Event() for _ in range(3))

    def arm(self, park_at: int) -> None:
        self.count, self.park_at = 0, park_at
        for event in (self.entered, self.gate, self.ran):
            event.clear()

    @contextmanager
    def installed(self):
        real = Lanes.lane_call
        hops = self

        async def lane_call(self_: Lanes, root_id: str, fn: Any, /, *args: Any) -> Any:
            hops.count += 1
            if hops.count == hops.park_at:
                inner = fn

                def parked(*a: Any) -> Any:
                    hops.entered.set()
                    hops.gate.wait(CAP_SECONDS)
                    try:
                        return inner(*a)
                    finally:
                        hops.ran.set()

                return await real(self_, root_id, parked, *args)
            return await real(self_, root_id, fn, *args)

        with mock.patch.object(Lanes, "lane_call", lane_call):
            yield self


async def _barrier(log: EventLog, root_id: str) -> None:
    """Return once every job queued so far on the lane of *root_id* has run (FIFO)."""
    await log.lane_call(root_id, lambda: None)


def _events(api, root_id: str = _CID) -> list[dict[str, Any]]:
    return list(api.event_log.read(root_id))


def _every_request_has_its_end(kinds: list[str]) -> None:
    assert kinds.count("llm/request") == (
        kinds.count("llm/response") + kinds.count("provider/error")
    ), kinds


async def _cancel_at(hops: Hops, work: Awaitable[Any]) -> asyncio.Task:
    """Run *work*, cancel its task while the parked job waits, release the job, return the task."""
    task = asyncio.ensure_future(work)
    try:
        assert await asyncio.to_thread(hops.entered.wait, CAP_SECONDS), "never parked"
        task.cancel()
        await asyncio.sleep(0)  # the cancel lands at the awaited hop
    finally:
        hops.gate.set()
    await asyncio.wait({task}, timeout=CAP_SECONDS)
    assert task.done()
    return task


async def _turn_hops(tmp_path: Path) -> tuple[int, str]:
    """A clean logged turn of the ``plain`` scenario: its hop count N and its reply."""
    hops = Hops()
    with hops.installed():
        api, _, cid = await _prepare("plain", 0, tmp_path / "count")
        hops.arm(-1)
        reply = await api.aconverse(_SCENARIOS["plain"].turns[0], cid)
        hops_of_the_turn = hops.count
    api.close()
    return hops_of_the_turn, reply


class TestTheSweepOfTheHops:
    async def test_a_cancel_parked_at_every_hop_of_a_turn(self, tmp_path: Path) -> None:
        """Kills: a hop with no cancel arm (a request with no ``provider/error``), a lane job
        cancelled with its await (the orphan never ran), a ``turn/end`` that is not queued
        behind ``turn/start``, a gate that stays held, a rollback that misses an orphan write,
        and an end hop that absorbs the cancel (D-019)."""
        message = _SCENARIOS["plain"].turns[0]
        n, expected = await _turn_hops(tmp_path)
        assert n >= 8, n  # begin, requests, replies, the extraction blob, end
        hops = Hops()
        for k in range(1, n + 1):
            with hops.installed():
                api, _, cid = await _prepare("plain", 0, tmp_path / f"k{k}")
                before = _seen(api, cid)
                hops.arm(k)
                task = await _cancel_at(hops, api.aconverse(message, cid))
                await _barrier(api.event_log, _CID)
                assert hops.ran.is_set(), f"k={k}: the orphan job never ran"
                kinds = [e["type"] for e in _events(api)]
                _every_request_has_its_end(kinds)
                assert _gate_is_free(api, cid), f"k={k}"
                if k < n:
                    assert task.cancelled(), f"k={k}"
                    assert _seen(api, cid) == before, f"k={k}"
                    self._the_log_shows_a_rolled_back_turn(api, before)
                    assert await api.aconverse(message, cid) == expected, f"k={k}"
                else:  # the end write of a committed turn: the cancel raises, the turn stays (D-019)
                    assert task.cancelled(), f"k={k}"
                    ends = [e for e in _events(api) if e["type"] == "turn/end"]
                    assert [e["outcome"] for e in ends][-1] == "committed"
                    assert _seen(api, cid)["state"] != before["state"]
            api.close()

    @staticmethod
    def _the_log_shows_a_rolled_back_turn(api, before: dict[str, Any]) -> None:
        events = _events(api)
        kinds = [e["type"] for e in events]
        starts = [e["seq"] for e in events if e["type"] == "turn/start"]
        ends = [e["seq"] for e in events if e["type"] == "turn/end"]
        assert len(starts) == len(ends) == 1 and starts[0] < ends[0], kinds
        assert [e["outcome"] for e in events if e["type"] == "turn/end"] == [
            "rolled_back"
        ]
        folded = fold_session(events, api.event_log.get_json)
        assert folded is not None
        assert folded.current_state == before["state"]
        assert folded.conversation_history == before["history"]


_PROVIDER = "fsm_llm.llm.providers.litellm"
_OBSERVED_FSM = kc._fsm(
    "Observed",
    "chat",
    kc._state(
        "chat",
        extraction_instructions="Extract the name",
        required_context_keys=["name"],
        response_instructions="Greet the user.",
        transitions=[
            kc._go("bye", conditions=kc._when("name", {"!=": [{"var": "name"}, None]}))
        ],
    ),
    kc._state("bye", response_instructions="Say goodbye."),
)


async def _acompletion(**kwargs: Any) -> Any:
    if kwargs.get("response_format"):
        return _reply(
            json.dumps({"extracted_data": {"name": "Ada"}, "confidence": 0.9})
        )
    return _reply("Hi")


@contextmanager
def _provider():
    from unittest.mock import AsyncMock

    with (
        mock.patch(f"{_PROVIDER}.acompletion", new_callable=AsyncMock) as acompletion,
        mock.patch(f"{_PROVIDER}.get_supported_openai_params", return_value=[]),
    ):
        acompletion.side_effect = _acompletion
        yield acompletion


async def _observed_api(root: Path) -> tuple[API, str]:
    api = API.from_definition(
        _OBSERVED_FSM,
        llm_interface=LiteLLMInterface(model="gpt-4o", api_key="sk-unused"),
        event_log=EventLog(root),
    )
    cid, _ = await api.astart_conversation(_conversation_id=_CID)
    return api, cid


class TestTheSweepOfTheObserverHops:
    """The same sweep over a ``LiteLLMInterface``: its request and reply are written by the
    request observer's async twins (``abegin_send``, ``aend_send``) from ``_asend``."""

    async def test_a_cancel_parked_at_every_hop_of_an_observed_turn(
        self, tmp_path: Path
    ) -> None:
        """Kills: an ``abegin_send`` whose cancel leaves an ``llm/request`` with no
        ``provider/error`` (the request job ran after the cancel), a begin hop the meter counts
        as a sent request, a reply job that survives with no request."""
        message = "my name is Ada"
        hops = Hops()
        with hops.installed(), _provider():
            api, cid = await _observed_api(tmp_path / "count")
            hops.arm(-1)
            expected = await api.aconverse(message, cid)
            n = hops.count
            api.close()
        assert n >= 6, n
        for k in range(1, n + 1):
            with hops.installed(), _provider():
                api, cid = await _observed_api(tmp_path / f"k{k}")
                before = _seen(api, cid)
                hops.arm(k)
                task = await _cancel_at(hops, api.aconverse(message, cid))
                await _barrier(api.event_log, _CID)
                assert hops.ran.is_set(), f"k={k}"
                kinds = [e["type"] for e in _events(api)]
                _every_request_has_its_end(kinds)
                assert _gate_is_free(api, cid), f"k={k}"
                if k < n:
                    assert task.cancelled(), f"k={k}"
                    assert _seen(api, cid) == before, f"k={k}"
                    TestTheSweepOfTheHops._the_log_shows_a_rolled_back_turn(api, before)
                    assert await api.aconverse(message, cid) == expected, f"k={k}"
                else:  # the end write of a committed turn: the cancel raises (D-019)
                    assert task.cancelled(), f"k={k}"
                    ends = [
                        e["outcome"] for e in _events(api) if e["type"] == "turn/end"
                    ]
                    assert ends[-1] == "committed" and _seen(api, cid) != before
            api.close()


def _plain_start_api(root: Path) -> API:
    return API.from_definition(
        _chat(silent=False), llm_interface=_llm(), event_log=EventLog(root)
    )


class TestTheSweepOfTheStartHops:
    async def test_a_cancel_parked_at_every_hop_of_a_start_leaves_no_conversation(
        self, tmp_path: Path
    ) -> None:
        """Kills: a cancel after the greeting that leaves a conversation nobody holds (the
        record hop is a cancel point past the manager's start), a writer that stays locked, an
        orphan greeting request with no ``provider/error``."""
        hops = Hops()
        with hops.installed():
            api = _plain_start_api(tmp_path / "count")
            hops.arm(-1)
            await api.astart_conversation()
            n = hops.count
            api.close()
            assert n >= 3, n  # request, reply, the start record
            for k in range(1, n + 1):
                api = _plain_start_api(tmp_path / f"k{k}")
                hops.arm(k)
                task = await _cancel_at(hops, api.astart_conversation())
                assert task.cancelled(), f"k={k}"
                assert await asyncio.to_thread(hops.ran.wait, CAP_SECONDS), f"k={k}"
                (root,) = api.event_log.conversation_ids()  # the id the start minted
                await _barrier(api.event_log, root)
                _every_request_has_its_end([e["type"] for e in _events(api, root)])
                manager = api.fsm_manager
                assert manager.instances == {}, f"k={k}"
                assert manager._conversation_locks == {}, f"k={k}"
                assert api.conversation_stacks == {}, f"k={k}"
                await _barrier(
                    api.event_log, root
                )  # the release queued behind the jobs ran
                assert not api.event_log.has_writer(root), f"k={k}"
                hops.arm(-1)
                await api.astart_conversation()
                api.close()


class TestAnOrphanAndTheRelease:
    async def test_a_release_queued_behind_an_orphan_job_closes_the_writer_it_wrote_with(
        self, tmp_path: Path
    ) -> None:
        """The turn is cancelled while its first job (``turn/start``, the writer open) is
        parked; ``release`` is queued behind the orphan and the ``turn/end`` job.
        Kills: a release that runs inline and closes nothing (the orphan then reopens a writer
        and keeps its lock for good), a release that overtakes the orphan job (D8)."""
        hops = Hops()
        root = tmp_path / "log"
        with hops.installed():
            api, _, cid = await _prepare("plain", 0, root)
            hops.arm(1)
            task = asyncio.ensure_future(
                api.aconverse(_SCENARIOS["plain"].turns[0], cid)
            )
            try:
                assert await asyncio.to_thread(hops.entered.wait, CAP_SECONDS)
                task.cancel()
                await asyncio.sleep(0)
                # the cancelled task has queued its ``turn/end`` behind the parked job; the
                # release lands behind both (an ended conversation would be refused: busy) and,
                # off a lane thread, waits for them: a thread of its own keeps the loop free
                pool = _pool_of(api.event_log, _CID)
                _until(lambda: _queued(pool) == 1, "the turn/end job to queue")
                releaser, _ = _run_in_thread(lambda: api.event_log.release(_CID))
                _until(lambda: _queued(pool) == 2, "the release job to queue")
                assert (
                    releaser.is_alive()
                )  # it waits for the parked job and the turn/end
            finally:
                hops.gate.set()
            _until(lambda: not releaser.is_alive(), "the release to return")
            await asyncio.wait({task}, timeout=CAP_SECONDS)
            await _barrier(api.event_log, _CID)
            assert hops.ran.is_set()
        assert not api.event_log.has_writer(_CID)
        with EventLog(root) as other:  # the lock is free: a second log takes the writer
            assert other.open_existing(_CID) >= 1
        api.close()


class TestAKillAfterTheRequestHop:
    @pytest.mark.parametrize("bound", ["0", "1"])
    def test_a_kill_blocked_in_an_async_model_call_leaves_the_fsynced_request(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bound: str
    ) -> None:
        """The child writes ``llm/request`` through a lane job, then blocks inside the model
        call on the loop thread; ``Popen.kill()`` there must leave the request on disk and
        the log resumable at ``call_model``. Linux proven; the Windows legs run it in CI.
        Kills: a request written after the call, a request that stayed in a queue, a lane
        job whose fsync was dropped."""
        if not can_lock_in_child():
            pytest.skip("this platform cannot start a child that locks a file")
        monkeypatch.setenv(
            child.BOUND_ENV, bound
        )  # 0: every append re-opens by tail read
        root, ready, err = tmp_path / "log", tmp_path / "ready.json", tmp_path / "err"
        with child_process([CHILD_MODULE, str(root), str(ready), "async"], err) as proc:
            wait_ready(proc, ready, err)
            proc.kill()  # SIGKILL on POSIX, TerminateProcess on Windows
            proc.wait(timeout=60 * current_timing_factor())
        cid = json.loads(ready.read_text(encoding="utf-8"))["conversation_id"]
        events = list(read_events(root / "logs" / f"{cid}.jsonl"))
        kinds = [e["type"] for e in events]
        assert kinds[-1] == "llm/request", kinds  # the request is the last line on disk
        assert events[-1]["call_type"] == "completion"
        assert "llm/response" not in kinds
        last_start = len(kinds) - 1 - kinds[::-1].index("turn/start")
        assert "turn/end" not in kinds[last_start:]  # this turn never ended
        assert "turn/end" in kinds[:last_start]  # the silent step before it did
        with reopen_after_kill(lambda: EventLog(root)) as reopened:
            api = API.from_definition(
                child.tool_fsm(), llm_interface=child.ChildLLM([]), event_log=reopened
            )
            _, state = api.resume(cid)
            assert state.current_state == "call_model"


class TestTheObserverTwins:
    """``llm.observer`` awaits an observer's ``abegin_send`` and ``aend_send`` when its CLASS
    defines them and calls the sync hooks inline otherwise (a bare mock is no twin)."""

    @staticmethod
    def _install(observer: Any):
        from fsm_llm.llm import observer as obs

        return obs, obs.install_observer(observer)

    async def test_an_observer_with_no_twins_is_called_inline(self) -> None:
        seen: list[str] = []

        class Plain:
            def begin_send(self, call, *, call_type):
                seen.append("begin")
                return "t"

            def end_send(self, token, response):
                seen.append("end")

            def fail_send(self, token, error):
                seen.append("fail")

        obs, token = self._install(Plain())
        try:
            handle = await obs.abegin_send({"model": "m", "x": 1}, call_type="c")
            await obs.aend_send(handle, "reply")
        finally:
            obs.reset_observer(token)
        assert seen == ["begin", "end"]

    async def test_an_observer_with_twins_has_only_the_twins_awaited(self) -> None:
        seen: list[Any] = []

        class Twinned:
            def begin_send(self, call, *, call_type):
                seen.append("sync-begin")

            def end_send(self, token, response):
                seen.append("sync-end")

            def fail_send(self, token, error):
                seen.append("fail")

            async def abegin_send(self, call, *, call_type):
                await asyncio.sleep(0)
                seen.append(("begin", sorted(call), call_type))
                return "tok"

            async def aend_send(self, token, response):
                await asyncio.sleep(0)
                seen.append(("end", token, response))

        obs, token = self._install(Twinned())
        try:
            handle = await obs.abegin_send({"model": "m", "secret": 1}, call_type="c")
            await obs.aend_send(handle, "reply")
        finally:
            obs.reset_observer(token)
        assert seen == [("begin", ["model"], "c"), ("end", "tok", "reply")]

    async def test_a_failing_end_twin_is_a_warning_and_never_raises(self) -> None:
        class Broken:
            def begin_send(self, call, *, call_type):
                return None

            async def aend_send(self, token, response):
                raise RuntimeError("boom")

        obs, token = self._install(Broken())
        try:
            handle = await obs.abegin_send({"model": "m"}, call_type="c")
            await obs.aend_send(handle, "reply")  # does not raise
        finally:
            obs.reset_observer(token)
        assert handle is not None

    async def test_a_bare_mock_observer_is_no_twin(self) -> None:
        observer = mock.MagicMock()
        observer.begin_send.return_value = "t"
        obs, token = self._install(observer)
        try:
            handle = await obs.abegin_send({"model": "m"}, call_type="c")
            await obs.aend_send(handle, "r")
        finally:
            obs.reset_observer(token)
        observer.begin_send.assert_called_once()
        observer.end_send.assert_called_once_with("t", "r")


class TestASecondCancel:
    async def test_a_second_cancel_returns_at_once_and_the_end_job_still_completes(
        self, tmp_path: Path
    ) -> None:
        """The turn was cancelled at its first hop and waits in ``finally`` for its ``turn/end``
        job, queued behind the parked one. A SECOND cancel ends the task at once (it never waits
        on a disk), the gate is free, and the job still writes ``turn/end`` when released.
        Kills: an end write shielded from the caller, one that swallows a cancel on the error
        path (D-016), a ``release`` that runs before the job."""
        hops = Hops()
        with hops.installed():
            api, _, cid = await _prepare("plain", 0, tmp_path / "log")
            hops.arm(1)
            task = asyncio.ensure_future(
                api.aconverse(_SCENARIOS["plain"].turns[0], cid)
            )
            try:
                assert await asyncio.to_thread(hops.entered.wait, CAP_SECONDS)
                task.cancel()
                await asyncio.sleep(0)
                assert (
                    not task.done()
                )  # waiting for its turn/end, behind the parked job
                task.cancel()
                await asyncio.wait({task}, timeout=CAP_SECONDS)
                assert task.cancelled()
                assert _gate_is_free(api, cid)
            finally:
                hops.gate.set()
            await _barrier(api.event_log, _CID)
        ends = [e["outcome"] for e in _events(api) if e["type"] == "turn/end"]
        assert ends == ["rolled_back"]
        api.close()


class TestACancelOnTheEndWriteOfAFailedTurn:
    async def test_the_cancel_propagates_when_the_turn_failed(
        self, tmp_path: Path
    ) -> None:
        """Pass 2 fails, so the turn ends with an error; a cancel that lands on its ``turn/end``
        write is the caller's request to stop and propagates (D-016 swallows it only for a turn
        that ended WITHOUT an error). Kills: a swallow on the error path."""
        message = _SCENARIOS["pass2_failure"].turns[0]
        hops = Hops()
        with hops.installed():
            api, _, cid = await _prepare("pass2_failure", 0, tmp_path / "count")
            hops.arm(-1)
            with pytest.raises(FSMError):
                await api.aconverse(message, cid)
            n = hops.count
            api.close()
            api, _, cid = await _prepare("pass2_failure", 0, tmp_path / "k")
            hops.arm(n)  # the last hop of the turn is its turn/end write
            task = await _cancel_at(hops, api.aconverse(message, cid))
            await _barrier(api.event_log, _CID)
        assert task.cancelled()
        ends = [e["outcome"] for e in _events(api) if e["type"] == "turn/end"]
        assert ends == ["rolled_back"]
        api.close()


class TestATurnThatNeverStarted:
    async def test_a_refused_turn_start_writes_no_turn_end(
        self, tmp_path: Path
    ) -> None:
        """The log cannot write ``turn/start``: the turn raises before it begins (as before the
        lanes) and its ``turn/end`` job, queued behind the failed one, writes nothing. Kills: a
        ``turn/end`` for a turn that has no ``turn/start``, a turn that runs unlogged."""
        message = _SCENARIOS["plain"].turns[0]
        api, _, cid = await _prepare("plain", 0, tmp_path / "log")
        before = _seen(api, cid)
        real = EventLog.append

        def append(self, conversation_id, type, **fields):
            if str(getattr(type, "value", type)) == "turn/start":
                raise LogWriteError("disk full")
            return real(self, conversation_id, type, **fields)

        with mock.patch.object(EventLog, "append", append):
            with pytest.raises(LogWriteError):
                await api.aconverse(message, cid)
        await _barrier(api.event_log, _CID)
        kinds = [e["type"] for e in _events(api)]
        assert "turn/start" not in kinds and "turn/end" not in kinds, kinds
        assert _seen(api, cid) == before and _gate_is_free(api, cid)
        assert await api.aconverse(message, cid)  # the log works again: the turn runs
        api.close()


class TestAnEndedLoggedConversationHoldsNoWriter:
    """Review 1 of plan 3f4530d3: ``end_conversation`` and ``close`` return with no writer, file
    or lock left, so the next step of the same caller (``resume``, a new ``EventLog`` on the
    directory) never meets it."""

    async def test_resume_right_after_an_async_end_conversation_works(
        self, tmp_path: Path
    ) -> None:
        """Kills: a ``release`` that queues the close on the lane and returns (``resume`` then
        raised ``LogLockedError ... already has a live writer``, 30 of 30 runs)."""
        for n in range(3):
            api, _, cid = await _prepare("plain", 1, tmp_path / f"log{n}")
            state = api.get_current_state(cid)
            api.end_conversation(cid)
            assert not api.event_log.has_writer(cid), n
            api.resume(cid)
            assert api.get_current_state(cid) == state, n
            api.close()

    async def test_a_new_event_log_on_the_directory_resumes_right_after_a_close(
        self, tmp_path: Path
    ) -> None:
        """Kills: a ``close`` that returns before the lane released the advisory lock (a second
        ``EventLog`` then met ``held by another writer``, 7 of 30 runs). The close is slowed so
        the window is certain; nothing asserts a duration."""
        root = tmp_path / "log"
        api, _, cid = await _prepare("plain", 1, root)
        real_close = JsonlWriter.close

        def slow_close(self: JsonlWriter) -> None:
            threading.Event().wait(0.05)
            real_close(self)

        with mock.patch.object(JsonlWriter, "close", slow_close):
            api.event_log.close()
        again = EventLog(root)
        try:
            assert again.open_existing(cid) >= 1  # the lock is free
        finally:
            again.close()
        api.close()


def _chain(length: int = 8) -> FSMDefinition:
    """``s0 -> s1 -> ... `` of silent states: a run loop takes one turn (one ``turn/end``) per step."""
    states: dict[str, Any] = {}
    for i in range(length):
        state: dict[str, Any] = {
            "id": f"s{i}",
            "description": f"s{i}",
            "purpose": "p",
            "response_instructions": "",
        }
        if i < length - 1:
            state["transitions"] = [
                {
                    "target_state": f"s{i + 1}",
                    "description": "go",
                    "priority": 1,
                    "conditions": [{"description": "always", "logic": {"==": [1, 1]}}],
                }
            ]
        states[f"s{i}"] = state
    return FSMDefinition(
        name="Chain", description="chain", initial_state="s0", states=states
    )


class _EndParks:
    """Parks the ``at``-th ``turn/end`` job (``engine.events._write_end``) on an Event, on its
    lane thread; every other job runs as written."""

    def __init__(self, at: int) -> None:
        self.at, self.calls = at, 0
        self.entered, self.gate = threading.Event(), threading.Event()

    @contextmanager
    def installed(self):
        real = engine_events._write_end
        parks = self

        def parked(handle: Any, blob: Any, error: Any) -> Any:
            parks.calls += 1
            if parks.calls == parks.at:
                parks.entered.set()
                parks.gate.wait(CAP_SECONDS)
            return real(handle, blob, error)

        with mock.patch.object(engine_events, "_write_end", parked):
            yield self


async def _chain_api(root: Path) -> tuple[API, str]:
    api = API.from_definition(
        _chain(), llm_interface=ScriptedLLM(responses={}), event_log=EventLog(root)
    )
    cid, _ = await api.astart_conversation()
    return api, cid


class TestACancelOnTheEndWriteOfACommittedTurn:
    """Review 2 of plan 3f4530d3 (D-019 supersedes D-016): a cancel that lands on the ``turn/end``
    write of a committed turn is raised, never absorbed. The probe was a chain of silent states
    under ``arun_until_terminal``: the run took six more steps after the cancel, ``cancelling()``
    stayed 1, and a ``TaskGroup`` or ``timeout`` around it never saw its cancel."""

    async def test_a_cancel_on_the_second_end_write_stops_the_run_loop_at_once(
        self, tmp_path: Path
    ) -> None:
        """Kills: an end hop that swallows the cancel (the run finished at ``s7``)."""
        parks = _EndParks(2)
        with parks.installed():
            api, cid = await _chain_api(tmp_path / "log")
            task = asyncio.ensure_future(api.arun_until_terminal(cid, max_steps=50))
            try:
                assert await asyncio.to_thread(parks.entered.wait, CAP_SECONDS)
                task.cancel()
                await asyncio.sleep(0)
            finally:
                parks.gate.set()
            await asyncio.wait({task}, timeout=CAP_SECONDS)
            await _barrier(api.event_log, cid)
        assert task.cancelled() and task.cancelling() >= 1
        assert (
            api.get_current_state(cid) == "s2"
        )  # the committed turn stays, nothing more ran
        ends = [e["outcome"] for e in _events(api, cid) if e["type"] == "turn/end"]
        assert ends == ["committed", "committed"]  # the parked job still wrote its line
        api.close()

    async def test_a_task_group_whose_sibling_fails_stops_the_run_at_the_end_write(
        self, tmp_path: Path
    ) -> None:
        """Kills: an absorbed cancel under a ``TaskGroup`` (the body kept running to ``s7``)."""
        parks = _EndParks(2)
        with parks.installed():
            api, cid = await _chain_api(tmp_path / "log")

            async def sibling() -> None:
                assert await asyncio.to_thread(parks.entered.wait, CAP_SECONDS)
                raise ValueError("sibling failed")

            try:
                with pytest.raises(ExceptionGroup) as raised:
                    async with asyncio.TaskGroup() as group:
                        group.create_task(sibling())
                        await api.arun_until_terminal(cid, max_steps=50)
            finally:
                parks.gate.set()
            await _barrier(api.event_log, cid)
        assert raised.group_contains(ValueError)
        assert api.get_current_state(cid) == "s2"
        api.close()

    async def test_a_timeout_that_fires_on_the_end_write_raises_timeout_error(
        self, tmp_path: Path
    ) -> None:
        """Kills: an absorbed cancel under ``asyncio.timeout`` (the block ended normally)."""
        parks = _EndParks(2)
        with parks.installed():
            api, cid = await _chain_api(tmp_path / "log")
            loop = asyncio.get_running_loop()
            try:
                with pytest.raises(TimeoutError):
                    async with asyncio.timeout(None) as scope:

                        async def expire() -> None:
                            assert await asyncio.to_thread(
                                parks.entered.wait, CAP_SECONDS
                            )
                            scope.reschedule(loop.time())

                        trip = asyncio.ensure_future(expire())
                        await api.arun_until_terminal(cid, max_steps=50)
            finally:
                parks.gate.set()
            await trip
            await _barrier(api.event_log, cid)
        assert api.get_current_state(cid) == "s2"
        api.close()

    async def test_the_reply_comes_only_after_the_turn_end_is_written(
        self, tmp_path: Path
    ) -> None:
        """Kills: an end write that is queued and not awaited (the reply, and so a kill, could
        come before the checkpoint of the turn the caller already holds)."""
        order: list[str] = []
        real = engine_events._write_end

        def spying(handle: Any, blob: Any, error: Any) -> Any:
            threading.Event().wait(0.02)  # the job is slower than the caller
            real(handle, blob, error)
            order.append("turn/end written")

        with mock.patch.object(engine_events, "_write_end", spying):
            api, _, cid = await _prepare("plain", 0, tmp_path / "log")
            await api.aconverse(_SCENARIOS["plain"].turns[0], cid)
            order.append("reply")
        assert order == ["turn/end written", "reply"]
        api.close()


class TestACloseWithJobsStillQueued:
    """Review 3 of plan 3f4530d3: ``close`` drops no write that was already queued."""

    async def test_a_close_leaves_the_end_of_a_held_request_and_its_turn_end_in_the_log(
        self, tmp_path: Path
    ) -> None:
        """A request job is held on its lane, the turn is cancelled twice (its ``provider/error``
        and ``turn/end`` are queued behind the held job) and the host closes the log. Kills: a
        ``close`` that marks the log closed and empties the table first (the queued jobs found
        no writer and the log ended with an ``llm/request`` and no end)."""
        entered, gate = threading.Event(), threading.Event()
        armed = [True]
        real = scope_module.log_request

        def parked(scope: Any, build: Any, **fields: Any) -> int:
            seq = real(scope, build, **fields)
            if armed[0]:
                armed[0] = False
                entered.set()
                gate.wait(CAP_SECONDS)
            return seq

        root = tmp_path / "log"
        with mock.patch.object(scope_module, "log_request", parked):
            api, _, cid = await _prepare("plain", 0, root)
            armed[0] = True
            task = asyncio.ensure_future(
                api.aconverse(_SCENARIOS["plain"].turns[0], cid)
            )
            closer = None
            try:
                assert await asyncio.to_thread(entered.wait, CAP_SECONDS)
                task.cancel()
                await asyncio.sleep(0)
                task.cancel()  # the second cancel ends the task; its turn/end stays queued
                await asyncio.wait({task}, timeout=CAP_SECONDS)
                assert task.cancelled()
                pool = _pool_of(api.event_log, _CID)
                closer, _ = _run_in_thread(api.event_log.close)
                _until(lambda: _queued(pool) >= 3, "the close job to queue")
                assert (
                    closer.is_alive()
                )  # it waits for the held job and what is behind it
            finally:
                gate.set()
            _until(lambda: not (closer and closer.is_alive()), "close to return")
        events = list(read_events(root / "logs" / f"{_CID}.jsonl"))
        kinds = [e["type"] for e in events]
        _every_request_has_its_end(kinds)
        assert kinds[-2:] == ["provider/error", "turn/end"], kinds
        assert not api.event_log.has_writer(_CID)
        with EventLog(root) as other:  # no writer, file or lock is left
            assert other.open_existing(_CID) >= 1


# ----------------------------------------------------------------------------------------------
# Plan 77e5fef9 step 5 (S2a, D-007): the joined reply of a streamed turn is a deferred lane job
# ----------------------------------------------------------------------------------------------

#: the lane hops of one streamed turn of the chat FSM (the greeting excluded): turn/start, the
#: extraction record, the stream request, turn/end and one more. The reply write is a deferred job
#: and has no hop of its own.
STREAM_TURN_HOPS = 5


async def _stream_logged(root: Path, cls: type = _Parking) -> tuple[API, Any, str]:
    api, llm = _stream_api(cls, event_log=EventLog(root))
    cid, _ = await api.astart_conversation(_conversation_id=_CID)
    return api, llm, cid


def _stream_ends(api: API) -> list[dict[str, Any]]:
    return [e for e in _events(api) if e["type"] == "llm/response" and e.get("stream")]


class TestTheHopsOfAStreamedTurn:
    async def test_a_streamed_turn_has_the_hops_it_had_before_the_deferral(
        self, tmp_path: Path
    ) -> None:
        """Kills (M26): an awaited ``lane_call`` in the ``finally`` of the stream (a new hop is
        a new cancel point in a generator whose cancel rules are pinned)."""
        hops = Hops()
        with hops.installed():
            api, _, cid = await _stream_logged(tmp_path)
            hops.arm(-1)
            assert await _collect(api.aconverse_stream("hi", cid)) == ["a", "b", "c"]
            assert hops.count == STREAM_TURN_HOPS
            api.close()

    async def test_a_cancel_parked_at_every_hop_of_a_streamed_turn(
        self, tmp_path: Path
    ) -> None:
        """The sweep of the plain turn over a stream: every request keeps its end, the gate is
        free, a rolled-back turn is rolled back whole, a committed turn stays committed, and the
        NEXT streamed turn of the same caller works and logs its own reply."""
        hops = Hops()
        for k in range(1, STREAM_TURN_HOPS + 1):
            with hops.installed():
                api, _, cid = await _stream_logged(tmp_path / f"k{k}")
                before = _seen(api, cid)
                hops.arm(k)
                task = await _cancel_at(hops, _collect(api.aconverse_stream("hi", cid)))
                await _barrier(api.event_log, _CID)
                assert hops.ran.is_set(), f"k={k}: the orphan job never ran"
                _every_request_has_its_end([e["type"] for e in _events(api)])
                assert _gate_is_free(api, cid), f"k={k}"
                assert task.cancelled(), f"k={k}"
                if k < STREAM_TURN_HOPS:
                    assert _seen(api, cid) == before, f"k={k}"
                    TestTheSweepOfTheHops._the_log_shows_a_rolled_back_turn(api, before)
                else:  # the end write of a committed turn: the cancel raises, the turn stays
                    ends = [e for e in _events(api) if e["type"] == "turn/end"]
                    assert ends[-1]["outcome"] == "committed", f"k={k}"
                hops.arm(-1)
                before_second = len(_stream_ends(api))
                assert await _collect(api.aconverse_stream("hi", cid)) == [
                    "a",
                    "b",
                    "c",
                ]
                await _barrier(api.event_log, _CID)
                assert len(_stream_ends(api)) == before_second + 1, f"k={k}"
                _every_request_has_its_end([e["type"] for e in _events(api)])
            api.close()


class TestAStreamedReplyKeepsItsEnd:
    """Every ``llm/request`` of a stream has an end after a drain, however the stream ended,
    sync twin and async: finished, ``aclose``, a cancelled consumer, ``athrow(CancelledError)``,
    a consumer that breaks, a provider error. The flags are the base's (D-007 e)."""

    @staticmethod
    def _stream_rows(api: API) -> list[tuple[str, bool | None]]:
        return [
            (e["type"], e.get("abandoned"))
            for e in _events(api)
            if e.get("stream") and e["type"] != "llm/request"
        ]

    @pytest.mark.parametrize(
        ("how", "ending"),
        [
            ("finish", [("llm/response", False)]),
            ("aclose", [("llm/response", True)]),
            ("break", [("llm/response", True)]),
            ("athrow", [("llm/response", True)]),
            ("cancel", [("llm/response", True)]),
        ],
    )
    async def test_every_way_a_stream_ends_leaves_one_end_per_request(
        self, tmp_path: Path, how: str, ending: list[tuple[str, bool | None]]
    ) -> None:
        """Kills (M24, M27): ``_done`` flipped inside the job (an ``llm/response`` AND a
        ``provider/error``), the write dropped on abandonment or on a rolled-back cancel."""
        api, llm, cid = await _stream_logged(tmp_path)
        stream = api.aconverse_stream("hi", cid)
        if how == "finish":
            assert await _collect(stream) == ["a", "b", "c"]
        elif how == "cancel":
            assert await stream.__anext__() == "a"
            llm.park(1)
            await _cancel(llm, _next(stream))
        else:
            assert await stream.__anext__() == "a"
            if how == "athrow":
                with pytest.raises((asyncio.CancelledError, StopAsyncIteration)):
                    await stream.athrow(asyncio.CancelledError())
            if how == "break":
                async for _ in stream:
                    break
            await stream.aclose()
        await _barrier(api.event_log, _CID)
        kinds = [e["type"] for e in _events(api)]
        _every_request_has_its_end(kinds)
        assert self._stream_rows(api) == ending, kinds
        assert _gate_is_free(api, cid)
        api.close()

    async def test_a_stream_that_raised_ends_in_a_provider_error_only(
        self, tmp_path: Path
    ) -> None:
        class Broken(_Parking):
            async def agenerate_response_stream(self, request):
                yield "a"
                raise RuntimeError("PRIVATE")

        api, _, cid = await _stream_logged(tmp_path, Broken)
        with pytest.raises(FSMError):
            await _collect(api.aconverse_stream("hi", cid))
        await _barrier(api.event_log, _CID)
        _every_request_has_its_end([e["type"] for e in _events(api)])
        assert self._stream_rows(api) == [("provider/error", None)]
        api.close()

    @pytest.mark.parametrize("how", ["finish", "close"])
    async def test_the_sync_stream_writes_the_same_ends(
        self, tmp_path: Path, how: str
    ) -> None:
        """The sync twin (``iterate_sync``, ``SYNC`` set: inline, no lane): the same rows."""
        api, _ = _stream_api(event_log=EventLog(tmp_path))

        def drive() -> None:
            cid, _ = api.start_conversation(_conversation_id=_CID)
            stream = api.converse_stream("hi", cid)
            if how == "finish":
                assert list(stream) == ["a", "b", "c"]
            else:
                assert next(stream) == "a"
                stream.close()

        await asyncio.to_thread(drive)
        _every_request_has_its_end([e["type"] for e in _events(api)])
        assert self._stream_rows(api) == [("llm/response", how == "close")]
        api.close()


class TestTheNextCallSeesTheReply:
    """Deferral changes WHEN, never WHETHER or in what ORDER (LESSONS 3f4530d3): the next call
    of the same caller meets the earlier stream reply."""

    async def test_a_turn_right_after_a_stream_logs_the_reply_before_its_turn_start(
        self, tmp_path: Path
    ) -> None:
        api, _, cid = await _stream_logged(tmp_path)
        for n in range(8):
            await _collect(api.aconverse_stream("hi", cid))
            await api.aconverse("again", cid)
            events = _events(api)
            responses = [e["seq"] for e in _stream_ends(api)]
            starts = [e["seq"] for e in events if e["type"] == "turn/start"]
            ends = [e["seq"] for e in events if e["type"] == "turn/end"]
            assert len(responses) == n + 1 and len(starts) == len(ends) == 2 * (n + 1)
            assert starts[2 * n] < responses[n] < ends[2 * n] < starts[2 * n + 1], n
        api.close()

    async def test_resume_right_after_end_conversation_finds_the_reply(
        self, tmp_path: Path
    ) -> None:
        """Kills (M28): a ``release`` that does not queue behind the reply job (the writer is
        gone before the line, or the lock is still held: the 30-of-30 failure of plan 3f4530d3)."""
        for n in range(30):
            api, _, cid = await _stream_logged(tmp_path / f"log{n}")
            await _collect(api.aconverse_stream("hi", cid))
            state = api.get_current_state(cid)
            api.end_conversation(cid)
            assert not api.event_log.has_writer(cid), n
            api.resume(cid)
            assert api.get_current_state(cid) == state, n
            assert len(_stream_ends(api)) == 1, n
            api.close()

    async def test_a_new_event_log_reads_the_reply_right_after_close(
        self, tmp_path: Path
    ) -> None:
        api, _, cid = await _stream_logged(tmp_path)
        await _collect(api.aconverse_stream("hi", cid))
        api.event_log.close()
        again = EventLog(tmp_path)
        try:
            ends = [e["type"] for e in again.read(cid)]
            assert ends[-2:] == ["llm/response", "turn/end"], ends
        finally:
            again.close()
        api.close()


PUSH_HOPS = (
    2  # the greeting of the pushed child: its request and its reply; the push adds none
)


async def _push_logged(root: Path) -> tuple[API, str]:
    api = _push_api(root)
    cid, _ = await api.astart_conversation()
    return api, cid


def _kinds(api: API, cid: str, prefix: str = "") -> list[str]:
    return [e["type"] for e in _events(api, cid) if e["type"].startswith(prefix)]


class TestTheHopsOfAnAsyncPush:
    """Plan 77e5fef9 step 6 (S2b, D-008): ``stack/push`` is deferred, never awaited, so
    ``apush_fsm`` has the hops it had (D-020 of plan baa37f40: no await after the frame append)."""

    async def test_a_push_has_the_hops_it_had_before_the_deferral(
        self, tmp_path: Path
    ) -> None:
        """Kills (M26 for the push): an awaited ``lane_call`` after the frame append (a new
        cancel point on a pushed frame)."""
        hops = Hops()
        with hops.installed():
            api, cid = await _push_logged(tmp_path)
            hops.arm(-1)
            await api.apush_fsm(cid, copy.deepcopy(_TALKING_KID))
            assert hops.count == PUSH_HOPS
            api.close()

    async def test_a_cancel_parked_at_every_hop_of_a_push(self, tmp_path: Path) -> None:
        """A cancel in the greeting leaves no child, no frame, no in-flight id and NO push line,
        the orphan job still ran, the gate is free, and the NEXT push of the caller works and
        folds. Kills: a push line queued before the greeting settles (a push for a child that
        does not exist), a frame left by the cancel."""
        from fsm_llm.runtime.stores import fold_stack

        hops = Hops()
        for k in range(1, PUSH_HOPS + 1):
            with hops.installed():
                api, cid = await _push_logged(tmp_path / f"k{k}")
                before = _seen(api, cid)
                hops.arm(k)
                task = await _cancel_at(
                    hops, api.apush_fsm(cid, copy.deepcopy(_TALKING_KID))
                )
                await _barrier(api.event_log, cid)
                assert hops.ran.is_set() and task.cancelled(), f"k={k}"
                assert api.get_stack_depth(cid) == 1, f"k={k}"
                assert not api._pending_push_ids, f"k={k}"
                assert _gate_is_free(api, cid), f"k={k}"
                assert _seen(api, cid) == before, f"k={k}"
                assert _kinds(api, cid, "stack/") == [], f"k={k}"
                _every_request_has_its_end(_kinds(api, cid))
                hops.arm(-1)
                await api.apush_fsm(cid, copy.deepcopy(_TALKING_KID))
                api.event_log.release(cid)
                assert _kinds(api, cid, "stack/") == ["stack/push"], f"k={k}"
                log = api.event_log
                folded = fold_stack(log.read(cid), log.get_json)
                assert folded is not None and folded.depth == 2, f"k={k}"
            api.close()
