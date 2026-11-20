"""CPU-time stopwatch for tests (plan d454fd62 step 67.1).

Lives beside ``tests/conftest.py``, which re-exports ``CpuTimer`` and
``cpu_timer`` (import them as ``from tests.conftest import cpu_timer``):
``conftest.py`` defines no class of its own (``test_doubles.py`` pins that).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

__all__ = ["CpuTimer", "cpu_timer"]


class CpuTimer:
    """CPU seconds spent inside one ``with cpu_timer():`` block.

    ``elapsed`` reads the seconds so far while the block is open and the final
    total after it closed (a block that raised still sets it).
    """

    def __init__(self, read_clock: Callable[[], float]) -> None:
        self._read = read_clock
        self._start = read_clock()
        self._stop: float | None = None

    @property
    def elapsed(self) -> float:
        end = self._read() if self._stop is None else self._stop
        return end - self._start


@contextmanager
def cpu_timer(*, thread: bool = False) -> Iterator[CpuTimer]:
    """Measure the CPU time of a computation, not the wall time it took.

    Interface contract (every test whose budget is about CPU work: a linear-time
    pin on a hostile input, a dry-run speed, a "completes under N seconds" check
    on pure computation):
        - ``with cpu_timer() as t: work()`` then ``assert t.elapsed < budget *
          timing_factor``. The clock is ``time.process_time()`` (user plus system
          CPU of the whole process, all threads: a computation run in a worker
          thread that the test joins is counted); ``thread=True`` reads
          ``time.thread_time()`` instead, for code that runs on the calling thread
          only (use it inside a worker thread body to time that worker alone).
        - Returns a :class:`CpuTimer`; never raises on its own, an exception of
          the block propagates and ``elapsed`` is still set.
        - Why not ``time.perf_counter()``: wall time includes every moment the
          scheduler gives the CPU to another process, so on a host that also runs
          a model server a 0.8 s computation reads 2+ s and trips a hard ceiling
          (plan d454fd62 step 67.1: ``test_dryrun`` failed 3 of 3 at load 30).
          CPU time does not grow while the process waits for a CPU. It is NOT
          immune to a tracer or to cache contention, so keep multiplying the
          budget by ``timing_factor`` (a tracer inflates CPU time too).
        - Use wall time only where wall time IS the property: a timeout or
          deadline that must fire, a sleep, a lock wait, concurrency overlap. A
          computation that sleeps or waits on I/O has CPU time near zero: never
          use this timer to bound a wait.
        - A ratio of two timings (a quadratic-versus-linear sweep) is scale-free
          and stays on whatever clock the sweep uses (``time.perf_counter`` stays
          legitimate there).
    """
    read_clock = time.thread_time if thread else time.process_time
    timer = CpuTimer(read_clock)
    try:
        yield timer
    finally:
        timer._stop = read_clock()
