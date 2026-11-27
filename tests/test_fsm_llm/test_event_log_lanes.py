"""Lane threads and the fsync outside the append guard (plan 3f4530d3 step 8a, decisions D-013
and D-014, roadmap row 1.18).

``Lanes`` runs ONE boundary write as one job on a FIFO lane thread; ``JsonlWriter`` fsyncs a
checkpoint after ``_guard`` is released. Every row is deterministic: a job or an fsync parks on
an ``Event`` the test releases, nothing asserts on a clock (an upper bound on a wait only caps a
hang), and a join of a thread that may be blocked on a lock is a bounded ``is_alive`` poll.

Nothing in the engine calls a lane yet (step 8b wires it), so the existing sync rows of the event
log run unedited; this file pins the lane and the writer alone.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import gc
import json
import threading
from functools import partial
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

import fsm_llm.runtime.log as log_module
from fsm_llm.aio import run_sync
from fsm_llm.runtime import EventLog, LogWriteError, bind, llm_call
from fsm_llm.runtime import lanes as lanes_module
from fsm_llm.runtime.calls import LlmCall, allm_call
from fsm_llm.runtime.lanes import Lanes
from fsm_llm.runtime.stores import jsonl, read_events
from fsm_llm.runtime.stores.jsonl import JsonlWriter
from fsm_llm.testing import ScriptedLLM
from tests.conftest import current_timing_factor
from tests.test_fsm_llm.event_log_support import greeting_request

CAP_SECONDS = 20.0  # only caps a hang; no assertion reads a duration
LANE_PREFIX = "fsm-llm-log-lane"


def _lane_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name.startswith(LANE_PREFIX)]


def _until(predicate, what: str) -> None:
    """Poll *predicate* in bounded ticks; never a join of a thread that may hold a lock."""
    ticker = threading.Event()
    for _ in range(int(CAP_SECONDS * 200)):
        if predicate():
            return
        ticker.wait(0.005)
    raise AssertionError(f"timed out waiting for {what}")


def _event(n: int = 0) -> dict[str, object]:
    return {
        "ts": "t",
        "conversation_id": "root",
        "frame": 0,
        "type": "context/record",
        "n": n,
    }


@pytest.fixture
def log(tmp_path: Path):
    event_log = EventLog(tmp_path)
    try:
        yield event_log
    finally:
        event_log.close()


class TestLaneCall:
    async def test_the_jobs_of_one_root_run_in_fifo_order_on_one_thread(self, log):
        order: list[int] = []
        threads: set[int] = set()

        def job(n: int) -> int:
            order.append(n)
            threads.add(threading.get_ident())
            return n

        results = await asyncio.gather(
            *(log.lane_call("root", job, n) for n in range(200))
        )
        assert order == list(range(200))
        assert results == list(range(200))
        assert len(threads) == 1
        assert threading.get_ident() not in threads

    async def test_a_root_always_maps_to_the_same_lane_thread(self):
        lanes = Lanes(count=4)
        before = set(
            _lane_threads()
        )  # pools other tests left to the collector are not ours
        try:
            seen: dict[str, set[int]] = {}
            for _ in range(3):
                for root in ("a", "b", "c", "d", "e"):
                    ident = await lanes.lane_call(root, threading.get_ident)
                    seen.setdefault(root, set()).add(ident)
            assert all(len(idents) == 1 for idents in seen.values())
            assert len(set(_lane_threads()) - before) <= 4
        finally:
            lanes.shutdown()

    def test_inline_in_sync_mode_and_no_thread_is_started(self, log):
        before = len(_lane_threads())
        ran_on: list[int] = []

        async def go() -> str:
            return await log.lane_call(
                "root", lambda: ran_on.append(threading.get_ident()) or "done"
            )

        assert run_sync(go()) == "done"
        assert ran_on == [threading.get_ident()]
        assert len(_lane_threads()) == before
        assert all(pool is None for pool in log._lanes._pools)

    async def test_the_job_sees_a_copy_of_the_callers_context(self, log):
        var: contextvars.ContextVar[str] = contextvars.ContextVar("lane_row")
        var.set("from the caller")

        def job() -> str:
            seen = var.get()
            var.set("set by the job")  # a copy: the caller must not see this
            return seen

        assert await log.lane_call("root", job) == "from the caller"
        assert var.get() == "from the caller"

    async def test_an_exception_of_the_job_reaches_the_awaiter(self, log):
        def boom() -> None:
            raise KeyError("boom")

        with pytest.raises(KeyError):
            await log.lane_call("root", boom)
        assert await log.lane_call("root", lambda: "next job still runs") == (
            "next job still runs"
        )

    async def test_a_job_still_runs_after_its_await_was_cancelled(self, log):
        started, release, finished = (threading.Event() for _ in range(3))

        def job() -> str:
            started.set()
            release.wait(CAP_SECONDS)
            finished.set()
            return "late"

        task = asyncio.ensure_future(log.lane_call("root", job))
        while not started.is_set():
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not finished.is_set()  # the await returned at once; the job is parked
        release.set()
        _until(finished.is_set, "the orphan job to finish")

    async def test_a_job_cancelled_while_queued_is_not_dropped(self, log):
        gate_started, gate = threading.Event(), threading.Event()
        ran: list[str] = []

        def blocker() -> None:
            gate_started.set()
            gate.wait(CAP_SECONDS)

        first = asyncio.ensure_future(log.lane_call("root", blocker))
        while not gate_started.is_set():
            await asyncio.sleep(0)
        queued = asyncio.ensure_future(log.lane_call("root", ran.append, "queued"))
        await asyncio.sleep(0)
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        gate.set()
        await first
        _until(lambda: ran == ["queued"], "the cancelled-while-queued job to run")

    async def test_a_failing_orphan_job_is_not_reported_as_a_lost_exception(self, log):
        started, release, finished = (threading.Event() for _ in range(3))
        problems: list[dict] = []
        asyncio.get_running_loop().set_exception_handler(
            lambda loop, ctx: problems.append(ctx)
        )

        def job() -> None:
            started.set()
            release.wait(CAP_SECONDS)
            try:
                raise RuntimeError("late failure")
            finally:
                finished.set()

        task = asyncio.ensure_future(log.lane_call("root", job))
        while not started.is_set():
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        _until(finished.is_set, "the orphan job to finish")
        for _ in range(10):
            await asyncio.sleep(0)
        gc.collect()
        assert problems == []

    async def test_a_closed_log_refuses_a_lane_call_and_sync_mode_still_runs(
        self, tmp_path
    ):
        closed = EventLog(tmp_path)
        closed.close()
        with pytest.raises(LogWriteError):
            await closed.lane_call("root", lambda: None)

    def test_count_must_be_positive(self):
        with pytest.raises(ValueError):
            Lanes(count=0)


def _pool_of(log: EventLog, root: str):
    """The lane pool of *root* as it stands now (taken BEFORE a ``close`` drops the table)."""
    return log._lanes._pools[log._lanes.lane_of(root)]


def _queued(pool) -> int:
    """Jobs waiting in *pool* (the job in flight is not counted)."""
    return pool._work_queue.qsize()


class TestCloseAndRelease:
    async def test_close_waits_for_the_job_in_flight_and_closes_the_writer_behind_it(
        self, log
    ):
        log.append("root", "context/record", n=1)
        writer = log._writers["root"]
        started, release, finished = (threading.Event() for _ in range(3))

        def job() -> None:
            started.set()
            release.wait(CAP_SECONDS)
            writer.append(_event(2))  # the writer is still open here, behind this job
            finished.set()

        task = asyncio.ensure_future(log.lane_call("root", job))
        while not started.is_set():
            await asyncio.sleep(0)
        threads = _lane_threads()
        assert threads
        pool = _pool_of(log, "root")
        closer, out = _run_in_thread(log.close)
        _until(lambda: _queued(pool) >= 1, "the close job to queue behind the job")
        assert writer._fd >= 0 and closer.is_alive()  # waiting, not returned
        release.set()
        await task
        _until(lambda: not closer.is_alive(), "close to return")
        assert out == [None] and writer._fd < 0 and finished.is_set()
        _until(
            lambda: not any(t.is_alive() for t in threads), "the lane threads to end"
        )
        assert [e["n"] for e in read_events(log.root / "logs" / "root.jsonl")] == [1, 2]

    async def test_release_runs_behind_the_queued_jobs_of_its_tree_and_waits_for_them(
        self, log
    ):
        log.append("root", "context/record", n=1)
        started, release = threading.Event(), threading.Event()
        outcome: list[object] = []

        def first() -> None:
            started.set()
            release.wait(CAP_SECONDS)

        def second() -> None:
            try:
                outcome.append(log.append("root", "context/record", n=2))
            except Exception as exc:  # a reopened writer would meet its own lock
                outcome.append(exc)

        t1 = asyncio.ensure_future(log.lane_call("root", first))
        while not started.is_set():
            await asyncio.sleep(0)
        t2 = asyncio.ensure_future(log.lane_call("root", second))
        await asyncio.sleep(0)  # the task has submitted its job
        pool = _pool_of(log, "root")
        releaser, out = _run_in_thread(lambda: log.release("root"))
        _until(lambda: _queued(pool) == 2, "the release job to queue behind both")
        assert log.has_writer("root") and releaser.is_alive()  # it waits for them
        release.set()
        await asyncio.gather(t1, t2)
        _until(lambda: not releaser.is_alive(), "release to return")
        assert out == [None] and not log.has_writer("root")
        assert outcome == [2]
        log.append("root", "context/record", n=3)  # a later append reopens the writer
        assert [e["n"] for e in read_events(log.root / "logs" / "root.jsonl")] == [
            1,
            2,
            3,
        ]

    async def test_release_returns_only_after_the_writer_is_closed_and_the_lock_is_free(
        self, log
    ):
        """Kills: a release that returns at once while the close runs on the lane (review 1:
        ``resume`` after an async ``end_conversation`` met its own writer)."""
        await log.lane_call("root", partial(log.append, "root", "context/record", n=1))
        writer = log._writers["root"]
        real_close = JsonlWriter.close

        def slow_close(self) -> None:
            threading.Event().wait(
                0.05
            )  # widens the window; no assertion reads a clock
            real_close(self)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(JsonlWriter, "close", slow_close)
            log.release("root")
            assert writer._fd < 0 and not log.has_writer("root")
        with EventLog(log.root) as other:  # the advisory lock is free again
            assert other.open_existing("root") >= 1

    async def test_close_returns_only_after_every_writer_is_closed_and_the_lock_is_free(
        self, tmp_path
    ):
        """Kills: a close that returns before the lane closed the writers (review 1: a new
        ``EventLog`` on the directory met ``held by another writer``)."""
        first = EventLog(tmp_path)
        await first.lane_call(
            "root", partial(first.append, "root", "context/record", n=1)
        )
        real_close = JsonlWriter.close

        def slow_close(self) -> None:
            threading.Event().wait(0.05)
            real_close(self)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(JsonlWriter, "close", slow_close)
            first.close()
        with EventLog(tmp_path) as second:
            assert second.open_existing("root") >= 1

    async def test_a_wait_is_bounded_when_the_lane_is_stuck(self, log, monkeypatch):
        """Kills: a wait with no bound (a hung disk would pin the caller, and the loop in async
        mode, for good). The job parks until the test ends; the release gives up and warns."""
        monkeypatch.setattr(lanes_module, "LANE_WAIT_SECONDS", 0.05)
        log.append("root", "context/record", n=1)
        started, release = threading.Event(), threading.Event()

        def stuck() -> None:
            started.set()
            release.wait(
                3 * CAP_SECONDS
            )  # outlives the poll below: an unbounded wait fails it

        task = asyncio.ensure_future(log.lane_call("root", stuck))
        while not started.is_set():
            await asyncio.sleep(0)
        waiter, out = _run_in_thread(lambda: log.release("root"))
        _until(lambda: not waiter.is_alive(), "a bounded release to return")
        assert out == [None] and log.has_writer("root")  # the job still owns the close
        release.set()
        await task
        _until(lambda: not log.has_writer("root"), "the queued close to run")

    async def test_a_release_and_a_close_called_on_a_lane_thread_never_wait(
        self, log, monkeypatch
    ):
        """Kills: a wait on a lane thread (the job would wait for its own queue: a deadlock
        until the bound). A job of one lane ends its tree, another tree on another lane and then
        closes the log; it returns, and every writer is closed once the lanes drained."""
        monkeypatch.setattr(lanes_module, "LANE_WAIT_SECONDS", 2 * CAP_SECONDS)
        other = next(
            name
            for name in (f"tree-{n}" for n in range(64))
            if log._lanes.lane_of(name) != log._lanes.lane_of("root")
        )
        await log.lane_call("root", partial(log.append, "root", "context/record", n=1))
        await log.lane_call(other, partial(log.append, other, "context/record", n=1))
        writers = [log._writers["root"], log._writers[other]]
        verdicts: list[object] = []

        def job() -> None:
            verdicts.append(lanes_module.on_lane_thread())
            log.release("root")  # own lane: queued behind this very job
            log.release(other)  # another lane
            log.close()

        await asyncio.wait_for(log.lane_call("root", job), CAP_SECONDS)
        assert verdicts == [True] and not lanes_module.on_lane_thread()
        _until(lambda: all(w._fd < 0 for w in writers), "the queued closes to run")

    def test_release_and_close_are_inline_when_no_lane_ever_ran(self, log):
        log.append("root", "context/record", n=1)
        writer = log._writers["root"]
        log.release("root")
        assert writer._fd < 0 and not log.has_writer("root")
        log.append("root", "context/record", n=2)
        writer2 = log._writers["root"]
        log.close()
        assert writer2._fd < 0
        log.close()  # idempotent


def _run_in_thread(fn) -> tuple[threading.Thread, list[object]]:
    out: list[object] = []

    def target() -> None:
        try:
            out.append(fn())
        except BaseException as exc:
            out.append(exc)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread, out


class TestFsyncOutsideTheGuard:
    def test_a_checkpoint_append_returns_only_after_its_fsync_that_follows_the_write(
        self, tmp_path, monkeypatch
    ):
        order: list[str] = []
        real_write, real_fsync = jsonl._write, jsonl._fsync
        monkeypatch.setattr(
            jsonl,
            "_write",
            lambda fd, data: (order.append("write"), real_write(fd, data))[1],
        )
        monkeypatch.setattr(
            jsonl, "_fsync", lambda fd: (order.append("fsync"), real_fsync(fd))[1]
        )
        with JsonlWriter(tmp_path / "logs" / "root.jsonl") as writer:
            order.clear()
            writer.append(_event(1), checkpoint=True)
            order.append("returned")
            writer.append(_event(2))
            order.append("returned")
        assert order == ["write", "fsync", "returned", "write", "returned"]

    def test_the_append_is_parked_in_its_fsync_until_the_disk_answers(
        self, tmp_path, monkeypatch
    ):
        entered, release = threading.Event(), threading.Event()
        monkeypatch.setattr(
            jsonl, "_fsync", lambda fd: (entered.set(), release.wait(CAP_SECONDS))[1]
        )
        with JsonlWriter(tmp_path / "logs" / "root.jsonl") as writer:
            thread, out = _run_in_thread(
                lambda: writer.append(_event(1), checkpoint=True)
            )
            assert entered.wait(CAP_SECONDS)
            assert out == []  # not acknowledged: the fsync has not returned
            release.set()
            _until(lambda: out == [1], "the append to return after its fsync")
            thread.join(CAP_SECONDS)

    def test_a_plain_append_is_not_blocked_by_a_checkpoint_parked_in_fsync(
        self, tmp_path, monkeypatch
    ):
        entered, release = threading.Event(), threading.Event()
        monkeypatch.setattr(
            jsonl, "_fsync", lambda fd: (entered.set(), release.wait(CAP_SECONDS))[1]
        )
        path = tmp_path / "logs" / "root.jsonl"
        with JsonlWriter(path) as writer:
            thread, out = _run_in_thread(
                lambda: writer.append(_event(0), checkpoint=True)
            )
            assert entered.wait(CAP_SECONDS)
            # the guard is free: appends from two threads land while the fsync is parked
            pair = [
                _run_in_thread(
                    lambda k=k: [writer.append(_event(k * 100 + i)) for i in range(25)]
                )
                for k in (1, 2)
            ]
            for thread2, _ in pair:
                _until(lambda t=thread2: not t.is_alive(), "an appender to finish")
            assert out == []  # the checkpoint is still parked
            release.set()
            _until(lambda: out == [1], "the checkpoint to be acknowledged")
            thread.join(CAP_SECONDS)
        seqs = [e["seq"] for e in read_events(path)]
        assert seqs == list(range(1, 52))  # unique, contiguous, every line whole
        lines = path.read_bytes().split(b"\n")
        assert lines[-1] == b"" and all(json.loads(line) for line in lines[:-1])

    @pytest.mark.skipif(
        current_timing_factor() > 1,
        reason="a thread stress row is skipped under a tracer",
    )
    def test_two_threads_appending_while_a_third_syncs_keep_seq_unique_and_lines_whole(
        self, tmp_path
    ):
        path = tmp_path / "logs" / "root.jsonl"
        with JsonlWriter(path) as writer:
            workers = [
                _run_in_thread(
                    lambda k=k, cp=cp: [
                        writer.append(
                            _event(k * 1000 + i), checkpoint=cp and i % 5 == 0
                        )
                        for i in range(100)
                    ]
                )
                for k, cp in ((1, False), (2, False), (3, True))
            ]
            for thread, out in workers:
                _until(lambda t=thread: not t.is_alive(), "a worker to finish")
                assert not isinstance(out[0], BaseException), out
        events = list(read_events(path))
        assert [e["seq"] for e in events] == list(range(1, 301))
        assert len({e["n"] for e in events}) == 300

    def test_a_failed_fsync_poisons_the_writer_and_the_next_append_refuses(
        self, tmp_path, monkeypatch
    ):
        def broken(fd: int) -> None:
            raise OSError(5, "disk gone")

        monkeypatch.setattr(jsonl, "_fsync", broken)
        with JsonlWriter(tmp_path / "logs" / "root.jsonl") as writer:
            with pytest.raises(LogWriteError, match="failed"):
                writer.append(_event(1), checkpoint=True)
            with pytest.raises(LogWriteError, match="poisoned"):
                writer.append(_event(2))
            with pytest.raises(LogWriteError, match="poisoned"):
                writer.append(_event(3), checkpoint=True)

    def test_an_interrupted_fsync_poisons_the_writer_and_the_interrupt_propagates(
        self, tmp_path, monkeypatch
    ):
        def interrupted(fd: int) -> None:
            raise KeyboardInterrupt

        monkeypatch.setattr(jsonl, "_fsync", interrupted)
        with JsonlWriter(tmp_path / "logs" / "root.jsonl") as writer:
            with pytest.raises(KeyboardInterrupt):
                writer.append(_event(1), checkpoint=True)
            with pytest.raises(LogWriteError, match="poisoned"):
                writer.append(_event(2))

    def test_a_close_waits_for_the_fsync_in_flight_and_the_append_then_succeeds(
        self, tmp_path, monkeypatch
    ):
        entered, release = threading.Event(), threading.Event()
        monkeypatch.setattr(
            jsonl, "_fsync", lambda fd: (entered.set(), release.wait(CAP_SECONDS))[1]
        )
        writer = JsonlWriter(tmp_path / "logs" / "root.jsonl")
        attempts: list[int] = []

        class _Watched:
            """The sync guard, noting each thread that reaches for it."""

            def __init__(self, lock: Any) -> None:
                self._lock = lock

            def __enter__(self) -> Any:
                attempts.append(threading.get_ident())
                return self._lock.__enter__()

            def __exit__(self, *exc_info: Any) -> Any:
                return self._lock.__exit__(*exc_info)

        writer._sync_guard = _Watched(writer._sync_guard)
        _thread, out = _run_in_thread(lambda: writer.append(_event(1), checkpoint=True))
        assert entered.wait(CAP_SECONDS)
        closer, _ = _run_in_thread(writer.close)
        # the closer has reached the guard the fsync holds (no sleep, no guess): it is blocked there
        _until(lambda: len(attempts) == 2, "the closer to reach the sync guard")
        assert closer.is_alive() and writer._fd >= 0  # not closed under an fsync
        release.set()
        _until(
            lambda: out == [1] and not closer.is_alive(), "append and close to finish"
        )
        assert writer._fd < 0

    def test_the_locks_are_taken_in_one_order_and_the_write_path_never_nests_them(self):
        source = Path(jsonl.__file__).read_text(encoding="utf-8")
        # the fsync is outside the guard: no `_fsync(` call sits inside the `with self._guard:` of
        # append_many (its body ends before `_flush` is called)
        body = source.split("def append_many", 1)[1].split("def _flush", 1)[0]
        assert "_fsync(" not in body
        assert "self._flush()" in body


class TestShutdownRunsTheFinalJobBehindTheQueue:
    async def test_the_final_job_follows_the_jobs_of_a_started_lane_and_runs_inline_for_idle_ones(
        self,
    ):
        """Kills: a final job that overtakes a job already queued (a writer closed under it) and
        a lane never started that is skipped (its inline writers stay open)."""
        lanes = Lanes(count=4)
        order: list[str] = []
        started, release = threading.Event(), threading.Event()

        def held() -> None:
            started.set()
            release.wait(CAP_SECONDS)
            order.append("job")

        task = asyncio.ensure_future(lanes.lane_call("a", held))
        while not started.is_set():
            await asyncio.sleep(0)
        busy = lanes.lane_of("a")
        jobs = lanes.shutdown(lambda lane: order.append(f"final {lane}"))
        assert len(jobs) == 1 and sorted(order) == [
            f"final {i}" for i in range(4) if i != busy
        ]  # the idle lanes ran inline, the busy lane's final is still queued
        release.set()
        await task
        _until(lambda: jobs[0].done(), "the final job of the busy lane")
        assert order[-2:] == ["job", f"final {busy}"]
        with pytest.raises(LogWriteError):
            await lanes.lane_call("a", lambda: None)


ROOT = "stream-root"


def _responses(log: EventLog) -> list[dict[str, Any]]:
    return [e for e in log.read(ROOT) if e["type"] == "llm/response"]


async def _park(log: EventLog) -> tuple[asyncio.Future[Any], threading.Event]:
    """Hold the lane of ROOT with a job that waits on the returned gate (set it to release)."""
    entered, gate = threading.Event(), threading.Event()

    def hold() -> None:
        entered.set()
        gate.wait(CAP_SECONDS)

    task = asyncio.ensure_future(log.lane_call(ROOT, hold))
    assert await asyncio.to_thread(entered.wait, CAP_SECONDS), "the lane never parked"
    return task, gate


class TestADeferredStreamReply:
    """Plan 77e5fef9 step 5 (S2a, D-007): the joined reply of a streamed call is ONE job queued
    behind the tree's lane jobs. Every row parks the lane on an ``Event`` (no clock) and reads
    what the log holds while the job waits."""

    async def test_the_reply_is_queued_behind_a_parked_job_and_written_on_a_lane(
        self, log: EventLog
    ) -> None:
        """Kills: the blob and the line written inline on the loop (a line in the log while
        the lane is parked, a loop thread name), a write on any thread but the lane."""
        threads: list[str] = []
        real = log.append

        def recording(*args: Any, **kwargs: Any) -> int:
            if args[1] == "llm/response":
                threads.append(threading.current_thread().name)
            return real(*args, **kwargs)

        llm = ScriptedLLM()
        with bind(log, ROOT):
            async with allm_call(
                llm, "response_stream", greeting_request(), stream=True
            ) as call:
                parked, gate = await _park(log)
                pool = _pool_of(log, ROOT)
                log.append = recording  # type: ignore[method-assign]
                call.stream_done("abc", abandoned=False)
                assert _responses(log) == [], "the reply was written on the loop"
                assert _queued(pool) == 1
                assert list(log._deferred) == [ROOT]  # an ordered job (D-009)
                gate.set()
                await parked
                await log.lane_call(ROOT, lambda: None)  # FIFO: the job has run
        (response,) = _responses(log)
        assert [e["type"] for e in log.read(ROOT)] == ["llm/request", "llm/response"]
        assert response["abandoned"] is False and response["stream"] is True
        assert response["request_seq"] == 1
        assert log.get_json(response["response_ref"])["text"] == "abc"
        assert len(threads) == 1 and threads[0].startswith(LANE_PREFIX), threads

    async def test_the_flags_and_the_link_are_decided_before_the_job_is_queued(
        self, log: EventLog
    ) -> None:
        """Kills: ``abandoned`` or the request link read inside the job (the loop has moved
        on: the observer's last request and the exhausted flag change before it runs)."""
        from fsm_llm.llm import LiteLLMInterface

        llm = LiteLLMInterface(model="gpt-4o", api_key="sk-unused")
        await log.lane_call(ROOT, lambda: None)  # a lane thread has served the tree
        with bind(log, ROOT) as scope:
            with llm_call(
                llm, "response_stream", greeting_request(), stream=True
            ) as call:
                scope.observer.last_request_seq = 7
                parked, gate = await _park(log)
                call.stream_done("ab")  # abandoned None: read from the chunk state NOW
                call._exhausted = True  # what a late read would see
                scope.observer.last_request_seq = 9999
                gate.set()
                await parked
                await log.lane_call(ROOT, lambda: None)
        (response,) = _responses(log)
        assert response["abandoned"] is True and response["request_seq"] == 7

    async def test_the_job_runs_in_a_copy_of_the_callers_context(
        self, tmp_path: Path
    ) -> None:
        """Kills: a bare ``lane_after`` job (the redaction hook of the reply blob would not see
        a ContextVar the caller set, as the hook of an ``aresponse`` job does)."""
        tenant: contextvars.ContextVar[str] = contextvars.ContextVar("tenant")
        seen: list[tuple[str, str]] = []

        def hook(kind: object, value: Any) -> Any:
            seen.append((str(getattr(kind, "value", kind)), tenant.get("<none>")))
            return value

        with EventLog(tmp_path, redact=hook) as redacting:
            tenant.set("acme")
            with bind(redacting, ROOT):
                async with allm_call(
                    ScriptedLLM(), "response_stream", greeting_request(), stream=True
                ) as call:
                    call.stream_done("abc", abandoned=False)
                    await redacting.lane_call(ROOT, lambda: None)
        assert seen and all(who == "acme" for _, who in seen), seen
        assert any(kind == "response" for kind, _ in seen), seen

    @pytest.mark.parametrize("how", ["release", "close"])
    async def test_release_and_close_wait_for_the_queued_reply(
        self, tmp_path: Path, how: str
    ) -> None:
        """Kills: a ``release`` or ``close`` that does not queue behind the job (it returns
        while the lane is parked and the writer is gone before the line: the end-then-resume
        loss of plan 3f4530d3)."""
        event_log = EventLog(tmp_path)
        with bind(event_log, ROOT):
            async with allm_call(
                ScriptedLLM(), "response_stream", greeting_request(), stream=True
            ) as call:
                parked, gate = await _park(event_log)
                pool = _pool_of(event_log, ROOT)
                call.stream_done("abc", abandoned=False)
                finished = threading.Event()

                def finish() -> None:
                    getattr(event_log, how)(*([ROOT] if how == "release" else []))
                    finished.set()

                thread, out = _run_in_thread(finish)
                _until(lambda: finished.is_set() or _queued(pool) >= 2, "the queue")
                assert not finished.is_set(), (
                    f"{how} returned before the reply was written"
                )
                assert (
                    _queued(pool) >= 2
                )  # the reply job, then the release or close job
                gate.set()
                await parked
                _until(finished.is_set, f"{how} to return")
                thread.join(CAP_SECONDS)
        assert out == [None]
        again = EventLog(tmp_path)
        try:
            assert [e["type"] for e in again.read(ROOT)] == [
                "llm/request",
                "llm/response",
            ]
        finally:
            again.close()

    async def test_a_log_closed_while_the_job_waits_still_gets_the_line(
        self, tmp_path: Path
    ) -> None:
        """``close`` queues its writers' close behind the jobs, so a queued reply still meets its
        writer. Kills: a job that is dropped when the log is closed (a request without its end)."""
        event_log = EventLog(tmp_path)
        with bind(event_log, ROOT):
            async with allm_call(
                ScriptedLLM(), "response_stream", greeting_request(), stream=True
            ) as call:
                parked, gate = await _park(event_log)
                call.stream_done("abc", abandoned=True)
                closing, out = _run_in_thread(event_log.close)
                _until(lambda: event_log._closed, "the log to be marked closed")
                gate.set()
                await parked
                _until(lambda: not closing.is_alive(), "close to return")
        assert out == [None]
        again = EventLog(tmp_path)
        try:
            assert [e["type"] for e in again.read(ROOT)] == [
                "llm/request",
                "llm/response",
            ]
        finally:
            again.close()

    async def test_a_reply_after_the_log_is_closed_is_a_warning_not_an_error(
        self, tmp_path: Path
    ) -> None:
        """The closed log refuses the line; the finally of the stream must raise nothing."""
        from unittest.mock import patch

        event_log = EventLog(tmp_path)
        with bind(event_log, ROOT):
            async with allm_call(
                ScriptedLLM(), "response_stream", greeting_request(), stream=True
            ) as call:
                event_log.close()
                with patch("fsm_llm.runtime.log.logger") as warned:
                    call.stream_done("abc", abandoned=False)
        assert warned.warning.called
        assert "llm/response not written" in warned.warning.call_args.args[0]

    async def test_a_reply_blob_that_cannot_be_written_is_an_unrecorded_line_and_a_warning(
        self, log: EventLog
    ) -> None:
        """The failure rules of the inline write hold on the lane: ``response_fields`` turns a
        failed blob into ``unrecorded`` with one WARNING and the stream's ``finally`` raises
        nothing. Kills: a job whose failure escapes (a ``Future`` exception, a lost line)."""
        from unittest.mock import patch

        with bind(log, ROOT):
            async with allm_call(
                ScriptedLLM(), "response_stream", greeting_request(), stream=True
            ) as call:
                with (
                    patch.object(log, "put_json", side_effect=OSError("disk full")),
                    patch("fsm_llm.runtime.scope.logger") as warned,
                ):
                    call.stream_done("abc", abandoned=False)
                    await log.lane_call(ROOT, lambda: None)
        (response,) = _responses(log)
        assert response["unrecorded"] is True and "response_ref" not in response
        assert "reply not recorded" in warned.warning.call_args.args[0]

    async def test_a_stream_before_any_lane_job_of_the_tree_runs_inline_a_named_limit(
        self, log: EventLog
    ) -> None:
        """``Lanes.after`` runs a function inline when no lane thread ever served the tree: a
        turn never gets here (its request is a lane job); a bare call does and the line is on
        disk at once, on the calling thread."""
        threads: list[str] = []
        real = log.append

        def recording(*args: Any, **kwargs: Any) -> int:
            threads.append(threading.current_thread().name)
            return real(*args, **kwargs)

        log.append = recording  # type: ignore[method-assign]
        with bind(log, ROOT):
            with llm_call(
                ScriptedLLM(), "response_stream", greeting_request(), stream=True
            ) as c:
                c.stream_done("abc", abandoned=False)
        assert [e["type"] for e in log.read(ROOT)] == ["llm/request", "llm/response"]
        assert set(threads) == {threading.current_thread().name}
        assert _pool_of(log, ROOT) is None  # no lane thread was started for the tree

    async def test_a_sync_mode_caller_writes_inline_even_with_a_started_lane(
        self, log: EventLog
    ) -> None:
        """``SYNC`` (a ``run_sync`` caller) never queues. The lane is parked, so a queued job
        could not have written the line. Kills: a deferral that ignores ``SYNC``."""
        await log.lane_call(ROOT, lambda: None)
        parked, gate = await _park(log)

        def drive() -> None:
            async def body() -> None:
                with bind(log, ROOT):
                    with llm_call(
                        ScriptedLLM(),
                        "response_stream",
                        greeting_request(),
                        stream=True,
                    ) as call:
                        call.stream_done("abc", abandoned=False)

            run_sync(body())

        try:
            await asyncio.to_thread(drive)
            assert len(_responses(log)) == 1, "the sync caller's reply was queued"
        finally:
            gate.set()
            await parked

    def test_no_public_name_was_added(self) -> None:
        """A review rule as a row: the deferral is a closure handed to ``lane_after``."""
        public = {n for n in dir(LlmCall) if not n.startswith("_")}
        assert public == {"response", "aresponse", "stream", "astream", "stream_done"}
        assert {n for n in dir(EventLog) if n.startswith("lane")} == {
            "lane_call",
            "lane_after",
        }


def _labels(log: EventLog) -> list[str]:
    return [e["label"] for e in log.read(ROOT) if "label" in e]


def _deferred_write(log: EventLog, label: str) -> Any:
    """A deferred job: one ``context/record`` named *label*, appended on the lane thread."""
    return lambda: log.append(ROOT, "context/record", label=label)


class TestADeferredJobOrdersTheInlineWritesOfItsTree:
    """Plan 77e5fef9 step 6 (D-008): an inline ``append`` of a tree waits for the tree's last
    deferred job, so a ``stack/pop``, a ``run/end`` or an ``update_context`` record written right
    after a deferred ``stack/push`` cannot land before it. Every row parks the lane on an
    ``Event`` and spies on the wait itself (no clock decides)."""

    @staticmethod
    @contextlib.contextmanager
    def _spy():
        """Set ``reached`` when an inline writer enters its wait (the ``settle`` of the log)."""
        reached = threading.Event()
        real = log_module.settle

        def settle(jobs: Any) -> bool:
            reached.set()
            return real(jobs)

        with mock.patch.object(log_module, "settle", settle):
            yield reached

    @pytest.mark.parametrize("how", ["append", "append_batch"])
    async def test_an_inline_write_waits_for_the_deferred_job_and_lands_after_it(
        self, log: EventLog, how: str
    ) -> None:
        """Kills (M31): an ``append`` or ``append_batch`` that does not wait (it returns while
        the lane is parked and its line precedes the deferred one: the inverted push and pop)."""
        await log.lane_call(ROOT, lambda: None)  # a lane thread serves the tree
        parked, gate = await _park(log)
        log.lane_after(ROOT, _deferred_write(log, "deferred"), ordered=True)
        write = {
            "append": lambda: log.append(ROOT, "context/record", label="inline"),
            "append_batch": lambda: log.append_batch(
                ROOT, [{"type": "context/record", "label": "inline"}]
            ),
        }[how]
        with self._spy() as reached:
            thread, out = _run_in_thread(write)
            _until(reached.is_set, "the inline write to reach its wait")
            assert thread.is_alive() and _labels(log) == []
            gate.set()
            await parked
            _until(lambda: not thread.is_alive(), "the inline write to finish")
        assert _labels(log) == ["deferred", "inline"] and len(out) == 1
        assert not isinstance(out[0], BaseException)

    async def test_the_last_of_several_deferred_jobs_is_what_it_waits_for(
        self, log: EventLog
    ) -> None:
        """A lane is FIFO, so the last job ends last. Kills: a wait for the FIRST job only."""
        await log.lane_call(ROOT, lambda: None)
        parked, gate = await _park(log)
        for label in ("one", "two", "three"):
            log.lane_after(ROOT, _deferred_write(log, label), ordered=True)
        with self._spy() as reached:
            thread, _ = _run_in_thread(
                lambda: log.append(ROOT, "context/record", label="inline")
            )
            _until(reached.is_set, "the inline write to reach its wait")
            gate.set()
            await parked
            _until(lambda: not thread.is_alive(), "the inline write to finish")
        assert _labels(log) == ["one", "two", "three", "inline"]

    async def test_another_tree_never_waits(self, log: EventLog) -> None:
        """The wait is per tree. Kills: a wait on any deferred job of the log."""
        await log.lane_call(ROOT, lambda: None)
        parked, gate = await _park(log)
        log.lane_after(ROOT, _deferred_write(log, "deferred"), ordered=True)
        try:
            with self._spy() as reached:
                thread, out = _run_in_thread(
                    lambda: log.append("another-root", "context/record", label="other")
                )
                _until(
                    lambda: not thread.is_alive(), "the other tree's write to return"
                )
            assert len(out) == 1 and not isinstance(out[0], BaseException)
            assert not reached.is_set(), "the other tree's write waited"
            assert _labels(log) == []  # the deferred job still waits behind the park
        finally:
            gate.set()
            await parked
            await log.lane_call(ROOT, lambda: None)

    async def test_a_deferred_job_does_not_wait_for_itself(self, log: EventLog) -> None:
        """The job's own append runs on a lane thread, where ``settle`` never waits. Kills: a
        wait on a lane thread (it would wait for the job it runs in, until the bound: a WARNING)."""
        await log.lane_call(ROOT, lambda: None)
        with (
            mock.patch.object(lanes_module, "LANE_WAIT_SECONDS", 0.2),
            mock.patch.object(lanes_module, "logger") as warned,
        ):
            log.lane_after(ROOT, _deferred_write(log, "deferred"), ordered=True)
            await log.lane_call(ROOT, lambda: None)  # FIFO: the job has run
        assert _labels(log) == ["deferred"]
        assert not warned.warning.called, warned.warning.call_args_list

    async def test_the_wait_is_bounded_and_the_write_goes_on_with_a_warning(
        self, log: EventLog
    ) -> None:
        """A hung lane: past ``LANE_WAIT_SECONDS`` the inline write goes on with a WARNING and
        may precede the job (the named limit of D-008 (d)). Kills: a wait with no bound."""
        await log.lane_call(ROOT, lambda: None)
        parked, gate = await _park(log)
        log.lane_after(ROOT, _deferred_write(log, "deferred"), ordered=True)
        try:
            with (
                mock.patch.object(lanes_module, "LANE_WAIT_SECONDS", 0.05),
                mock.patch.object(lanes_module, "logger") as warned,
            ):
                thread, out = _run_in_thread(
                    lambda: log.append(ROOT, "context/record", label="inline")
                )
                _until(lambda: not thread.is_alive(), "the bounded wait to give up")
            assert len(out) == 1 and not isinstance(out[0], BaseException)
            assert _labels(log) == ["inline"]  # the job is still parked
            assert "still running" in warned.warning.call_args.args[0]
        finally:
            gate.set()
            await parked
            await log.lane_call(ROOT, lambda: None)
        assert _labels(log) == ["inline", "deferred"]

    async def test_a_job_that_is_not_ordered_orders_nothing(
        self, log: EventLog
    ) -> None:
        """The cancel arms (``provider/error`` of a cancelled call) keep their pre-plan behaviour:
        an inline append right after one never waits for the lane (D-009, probe p11: 1.45 s on
        the loop at the parent). Kills: a registry that keeps every ``lane_after`` job."""
        await log.lane_call(ROOT, lambda: None)
        parked, gate = await _park(log)
        log.lane_after(ROOT, _deferred_write(log, "deferred"))
        try:
            assert log._deferred == {}
            with (
                self._spy() as reached,
                mock.patch.object(lanes_module, "LANE_WAIT_SECONDS", 0.2),
            ):
                thread, out = _run_in_thread(
                    lambda: log.append(ROOT, "context/record", label="inline")
                )
                _until(lambda: not thread.is_alive(), "the inline write to return")
            assert len(out) == 1 and not isinstance(out[0], BaseException), out
            assert not reached.is_set(), "an unordered job made an inline write wait"
            assert _labels(log) == ["inline"]
        finally:
            gate.set()
            await parked
            await log.lane_call(ROOT, lambda: None)

    def test_only_the_two_deferrals_of_the_plan_are_ordered(self) -> None:
        """A source scan (D-009): ``lane_after`` is called with ``ordered=True`` by
        ``LlmCall.stream_done`` and ``record_push`` and by nothing else; the cancel arms
        (``alog_request``, ``_recorded``) keep their pre-plan behaviour. Kills: a cancel arm that
        registers (an inline append after a cancelled call blocks the loop behind the stuck
        request job), a deferral that stops registering (a pop could precede its push)."""
        import ast

        import fsm_llm

        found: dict[tuple[str, str], bool] = {}
        for path in sorted(Path(fsm_llm.__file__).parent.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for fn in ast.walk(tree):
                if not isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef):
                    continue
                for call in ast.walk(fn):
                    if (
                        isinstance(call, ast.Call)
                        and isinstance(call.func, ast.Attribute)
                        and call.func.attr == "lane_after"
                    ):
                        flag = {k.arg: k.value for k in call.keywords}.get("ordered")
                        found[(path.name, fn.name)] = (
                            isinstance(flag, ast.Constant) and flag.value is True
                        )
        ordered = {key for key, flag in found.items() if flag}
        assert ordered == {("calls.py", "stream_done"), ("events.py", "record_push")}
        assert found[("scope.py", "alog_request")] is False
        assert found[("calls.py", "_recorded")] is False

    async def test_a_finished_job_leaves_nothing_behind(self, log: EventLog) -> None:
        """Kills: a registry that keeps a finished job (a leak per conversation, a wait on a
        done future). A job that ran inline (no lane thread ever served the tree) registers none."""
        log.lane_after(
            ROOT, _deferred_write(log, "inline"), ordered=True
        )  # no lane thread yet: inline
        assert log._deferred == {} and _pool_of(log, ROOT) is None
        await log.lane_call(ROOT, lambda: None)
        parked, gate = await _park(log)
        log.lane_after(ROOT, _deferred_write(log, "queued"), ordered=True)
        assert list(log._deferred) == [ROOT]
        gate.set()
        await parked
        await log.lane_call(ROOT, lambda: None)
        _until(lambda: log._deferred == {}, "the finished job to be forgotten")
        assert _labels(log) == ["inline", "queued"]
