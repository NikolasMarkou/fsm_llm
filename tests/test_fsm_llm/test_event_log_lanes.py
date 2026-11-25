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
import contextvars
import gc
import json
import threading
from pathlib import Path

import pytest

from fsm_llm.aio import run_sync
from fsm_llm.runtime import EventLog, LogWriteError
from fsm_llm.runtime.lanes import Lanes
from fsm_llm.runtime.stores import jsonl, read_events
from fsm_llm.runtime.stores.jsonl import JsonlWriter
from tests.conftest import current_timing_factor

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
        try:
            seen: dict[str, set[int]] = {}
            for _ in range(3):
                for root in ("a", "b", "c", "d", "e"):
                    ident = await lanes.lane_call(root, threading.get_ident)
                    seen.setdefault(root, set()).add(ident)
            assert all(len(idents) == 1 for idents in seen.values())
            assert len(_lane_threads()) <= 4
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


class TestCloseAndRelease:
    async def test_close_drains_a_job_in_flight_before_the_writer_is_closed(self, log):
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
        log.close()  # returns at once
        assert writer._fd >= 0  # queued behind the job in flight
        release.set()
        await task
        _until(lambda: writer._fd < 0, "the writer to be closed behind the job")
        _until(
            lambda: not any(t.is_alive() for t in threads), "the lane threads to end"
        )
        assert finished.is_set()
        assert [e["n"] for e in read_events(log.root / "logs" / "root.jsonl")] == [1, 2]

    async def test_release_runs_behind_the_queued_jobs_of_its_tree(self, log):
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
        log.release("root")  # queued behind both jobs
        assert log.has_writer("root")
        release.set()
        await asyncio.gather(t1, t2)
        _until(lambda: not log.has_writer("root"), "the release job to run")
        assert outcome == [2]
        log.append("root", "context/record", n=3)  # a later append reopens the writer
        assert [e["n"] for e in read_events(log.root / "logs" / "root.jsonl")] == [
            1,
            2,
            3,
        ]

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
        _thread, out = _run_in_thread(lambda: writer.append(_event(1), checkpoint=True))
        assert entered.wait(CAP_SECONDS)
        closer, _ = _run_in_thread(writer.close)
        assert writer._fd >= 0  # the descriptor is not closed under an fsync
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
