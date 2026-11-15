"""The ``cpu_timer`` helper (``tests/conftest.py``, plan d454fd62 step 67.1).

A budget about CPU work is measured in CPU seconds: wall time inflates when the
host is busy (a local model server, a parallel build), CPU time does not. The
tests here burn CPU and sleep to show the two clocks apart, and simulate a
loaded host by making ``time.perf_counter`` run 10 times fast.
"""

from __future__ import annotations

import ast
import threading
import time
from pathlib import Path

import pytest

from tests import conftest
from tests.conftest import CpuTimer, cpu_timer


def _burn(cpu_seconds: float) -> None:
    """Spin until this thread has used *cpu_seconds* of CPU (not wall) time."""
    end = time.thread_time() + cpu_seconds
    while time.thread_time() < end:
        pass


class _InflatedWallClock:
    """``time.perf_counter`` as seen on a loaded host: every real second shows
    as *factor* seconds (the scheduler gave the CPU to another process)."""

    def __init__(self, real, factor: float) -> None:
        self._real = real
        self._factor = factor
        self._origin = real()

    def __call__(self) -> float:
        return self._origin + (self._real() - self._origin) * self._factor


class TestCpuTimer:
    def test_it_counts_cpu_burnt_in_the_block(self):
        with cpu_timer() as timer:
            _burn(0.05)
        assert 0.05 <= timer.elapsed < 1.0

    def test_it_does_not_count_a_sleep(self):
        with cpu_timer() as timer:
            time.sleep(0.2)
        assert timer.elapsed < 0.1

    def test_the_wall_clock_counts_the_sleep_and_the_cpu_timer_does_not(self):
        wall_start = time.perf_counter()
        with cpu_timer() as timer:
            time.sleep(0.2)
        assert time.perf_counter() - wall_start >= 0.2
        assert timer.elapsed < 0.1

    def test_process_time_counts_the_cpu_of_a_joined_worker_thread(self):
        worker = threading.Thread(target=_burn, args=(0.05,))
        with cpu_timer() as timer:
            worker.start()
            worker.join()
        assert timer.elapsed >= 0.05

    def test_a_blocked_main_thread_burns_nothing_while_a_worker_sleeps(self):
        worker = threading.Thread(target=time.sleep, args=(0.2,))
        with cpu_timer() as timer:
            worker.start()
            worker.join()
        assert timer.elapsed < 0.1

    def test_thread_mode_counts_this_thread_only(self):
        sleeper = threading.Thread(target=time.sleep, args=(0.1,))
        sleeper.start()
        with cpu_timer(thread=True) as timer:
            _burn(0.03)
        sleeper.join()
        assert 0.03 <= timer.elapsed < 1.0

    def test_thread_mode_inside_a_worker_times_the_worker(self):
        box: dict[str, float] = {}

        def _work() -> None:
            with cpu_timer(thread=True) as timer:
                _burn(0.03)
            box["elapsed"] = timer.elapsed

        worker = threading.Thread(target=_work)
        worker.start()
        worker.join()
        assert 0.03 <= box["elapsed"] < 1.0

    def test_elapsed_can_be_read_inside_the_block_and_grows(self):
        with cpu_timer() as timer:
            first = timer.elapsed
            _burn(0.03)
            second = timer.elapsed
        assert second >= first + 0.03

    def test_elapsed_is_frozen_after_the_block(self):
        with cpu_timer() as timer:
            _burn(0.02)
        frozen = timer.elapsed
        _burn(0.03)
        assert timer.elapsed == frozen

    def test_an_exception_propagates_and_elapsed_is_still_set(self):
        with pytest.raises(ZeroDivisionError):
            with cpu_timer() as timer:
                _burn(0.02)
                raise ZeroDivisionError
        assert timer.elapsed >= 0.02
        settled = timer.elapsed
        _burn(0.01)
        assert timer.elapsed == settled  # closed, not still running

    def test_the_clock_is_read_at_call_time_not_at_import(self, monkeypatch):
        reads = iter([10.0, 12.5])
        monkeypatch.setattr(time, "process_time", lambda: next(reads))
        with cpu_timer() as timer:
            pass
        assert timer.elapsed == pytest.approx(2.5)
        assert isinstance(timer, CpuTimer)


class TestUnderASimulatedLoadedHost:
    """RED/GREEN for the root cause: the old shape (wall time around CPU work)
    fails when the wall clock runs 10x fast; the new shape does not."""

    @pytest.fixture
    def loaded_host(self, monkeypatch):
        monkeypatch.setattr(
            time, "perf_counter", _InflatedWallClock(time.perf_counter, 10.0)
        )

    def test_the_old_shape_fails_when_wall_time_is_inflated(self, loaded_host):
        start = time.perf_counter()
        _burn(0.05)
        took = time.perf_counter() - start
        with pytest.raises(AssertionError):
            assert took < 0.2  # 0.05 s of work reads as 0.5 s

    def test_the_new_shape_passes_when_wall_time_is_inflated(self, loaded_host):
        with cpu_timer() as timer:
            _burn(0.05)
        assert timer.elapsed < 0.2

    def test_the_new_shape_still_fails_on_real_cpu_work_over_the_budget(
        self, loaded_host
    ):
        with cpu_timer() as timer:
            _burn(0.06)
        with pytest.raises(AssertionError):
            assert timer.elapsed < 0.05

    def test_a_scaled_budget_still_holds_the_budget_meaningful(
        self, loaded_host, timing_factor
    ):
        with cpu_timer() as timer:
            _burn(0.02)
        assert timer.elapsed < 0.5 * timing_factor


#: Files that may still call ``time.perf_counter``: only this file (it simulates
#: the wall clock). The scale-free spread sweeps of ``test_context_unit.py`` read
#: CPU time too since plan d454fd62 step 120: a wall-clock ratio is not immune to
#: a loaded host (the scheduler stretches the sizes unevenly), and a ratio bound
#: is never scaled (see ``TestNoRatioBoundIsScaled``).
_WALL_CLOCK_FILES = {
    "test_cpu_timer.py": None,
}
_TEST_ROOT = Path(conftest.__file__).parent


def wall_clock_calls(source: str) -> list[tuple[int, str]]:
    """``(line, enclosing function)`` of every ``perf_counter`` reference."""
    tree = ast.parse(source)
    hits: list[tuple[int, str]] = []

    def walk(node: ast.AST, scope: str) -> None:
        for child in ast.iter_child_nodes(node):
            inner = (
                child.name
                if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef)
                else scope
            )
            if (isinstance(child, ast.Attribute) and child.attr == "perf_counter") or (
                isinstance(child, ast.Name) and child.id == "perf_counter"
            ):
                hits.append((child.lineno, scope))
            walk(child, inner)

    walk(tree, "<module>")
    return hits


class TestNoCpuBudgetIsMeasuredOnTheWallClock:
    """A budget about CPU work is measured with ``cpu_timer``: a test that times
    pure computation with ``time.perf_counter`` flakes on a loaded host."""

    def test_the_scan_reports_the_enclosing_function(self):
        source = (
            "import time\n"
            "def test_a():\n    s = time.perf_counter()\n"
            "def helper():\n    def measure():\n        return time.perf_counter()\n"
        )
        assert wall_clock_calls(source) == [(3, "test_a"), (6, "measure")]

    def test_the_scan_sees_a_from_import(self):
        assert wall_clock_calls("from time import perf_counter\nperf_counter()\n") == [
            (2, "<module>")
        ]

    def test_only_the_listed_functions_of_the_listed_files_use_perf_counter(self):
        offenders = {}
        for path in sorted(_TEST_ROOT.rglob("*.py")):
            hits = wall_clock_calls(path.read_text(encoding="utf-8"))
            if not hits:
                continue
            allowed = _WALL_CLOCK_FILES.get(path.name, set())
            if allowed is None:
                continue
            bad = [(n, f) for n, f in hits if f not in (allowed or set())]
            if bad:
                offenders[str(path.relative_to(_TEST_ROOT))] = bad
        assert offenders == {}, (
            "measure CPU work with tests.conftest.cpu_timer (CPU time), not "
            "time.perf_counter (wall time inflates on a loaded host)"
        )
