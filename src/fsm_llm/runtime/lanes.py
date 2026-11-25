"""Lane threads of the event log: where a boundary write runs so it does not block the loop.

A logged async turn fsyncs about 28 times (the log, the blobs and their directories), and each
fsync stalls the event loop for milliseconds. ``Lanes`` runs one boundary write as ONE job on a
lane thread and the caller awaits it, so the write is today's sync function, unchanged, only on
another thread. Nothing is queued as data: a line is still written and fsynced by the job
itself, never buffered (see the anchor).

Interface contract (callers: ``EventLog.lane_call``, ``EventLog.close`` and ``release``):
    - ``Lanes(count=LANE_COUNT)``: no thread starts until the first ``lane_call``; a sync-only
      host never starts one. One ``Lanes`` belongs to one ``EventLog``.
    - ``await lanes.lane_call(root_id, fn, *args)``: returns what ``fn(*args)`` returns and
      raises what it raises. With ``aio.SYNC`` set (a ``run_sync`` caller, no loop) ``fn`` runs
      inline on the caller's thread. Otherwise the job runs on the lane of *root_id* with a
      copy of the caller's Context (a ContextVar set by the caller is visible there). The
      lane is a stable hash of the ROOT id over ``count`` single-thread executors, so the jobs
      of one tree run in FIFO order. A cancelled await returns at once and the job STILL runs
      (an orphan job: its result is dropped); a job that raises after its await was cancelled
      is not reported. ``LogWriteError`` after ``shutdown``.
    - ``lanes.after(root_id, fn)``: runs ``fn`` after every job already queued on the lane of
      *root_id*, without waiting for it (a failure is a WARNING with the exception type); when
      no job ever ran on that lane, or after ``shutdown``, ``fn`` runs inline.
    - ``lanes.shutdown()``: idempotent, never waits; queued jobs still run, then the threads
      end, and no new job is accepted.
Imports ``aio`` and ``runtime.events`` only.
"""

from __future__ import annotations

import asyncio
import contextvars
import threading
import zlib
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, TypeVar

from fsm_llm.aio import SYNC
from fsm_llm.logging import logger
from fsm_llm.runtime.constants import LANE_COUNT
from fsm_llm.runtime.events import LogWriteError

T = TypeVar("T")


def _consume(future: asyncio.Future[Any]) -> None:
    """Mark the outcome of an orphaned job as seen (nobody awaits it any more)."""
    if not future.cancelled():
        future.exception()


def _warn_on_failure(future: Future[Any]) -> None:
    exc = future.exception()
    if exc is not None:
        logger.warning(f"event log lane job failed ({type(exc).__name__})")


# DECISION plan-2026-11-24T193347-3f4530d3/D-013
# The lanes carry whole boundary writes; they are NOT a queue of appends. Do NOT buffer a line
# for a writer task (a SIGKILL once lost 10 to 22 acknowledged events, D-011 of plan 21e6f00d),
# do NOT split a job so its blob and its line run on different threads or reorder them (blob
# first, line next, fsync last), do NOT put a ``Lanes`` in a module global (the lanes belong
# to one EventLog and end with its ``close``), do NOT use the default executor (sync-only model
# calls fill it and would queue every checkpoint behind them), and do NOT cancel the future of
# a job whose await was cancelled (the job must still run: a request line without its
# ``provider/error`` is worse than a late write). A lane thread never takes a manager lock.
class Lanes:
    """At most ``count`` single-thread FIFO executors, started lazily (see the module doc)."""

    def __init__(self, count: int = LANE_COUNT) -> None:
        if count < 1:
            raise ValueError(f"count is 1 or more, got {count}")
        self._pools: list[ThreadPoolExecutor | None] = [None] * count
        self._guard = threading.Lock()
        self._closed = False

    def _submit(
        self, root_id: str, create: bool, fn: Callable[..., T], *args: Any
    ) -> Future[T] | None:
        index = zlib.crc32(root_id.encode("utf-8")) % len(self._pools)
        with self._guard:
            pool = self._pools[index]
            if self._closed:
                if create:
                    raise LogWriteError("the event log is closed")
                return None
            if pool is None:
                if not create:
                    return None
                pool = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix=f"fsm-llm-log-lane-{index}"
                )
                self._pools[index] = pool
            return pool.submit(fn, *args)

    async def lane_call(self, root_id: str, fn: Callable[..., T], /, *args: Any) -> T:
        """Run ``fn(*args)`` on the lane of *root_id* and await it (inline in sync mode)."""
        if SYNC.get():
            return fn(*args)
        job = self._submit(root_id, True, contextvars.copy_context().run, fn, *args)
        assert job is not None  # create=True always submits or raises
        inner = asyncio.wrap_future(job)
        inner.add_done_callback(_consume)
        return await asyncio.shield(inner)

    def after(self, root_id: str, fn: Callable[[], object]) -> None:
        """Run *fn* behind the queued jobs of *root_id*'s lane, else inline (see the module doc)."""
        job = self._submit(root_id, False, fn)
        if job is None:
            fn()
        else:
            job.add_done_callback(_warn_on_failure)

    def shutdown(self) -> None:
        """Stop accepting jobs; queued ones still run and the threads then end. Never waits."""
        with self._guard:
            self._closed = True
            pools, self._pools = self._pools, [None] * len(self._pools)
        for pool in pools:
            if pool is not None:
                pool.shutdown(wait=False)
