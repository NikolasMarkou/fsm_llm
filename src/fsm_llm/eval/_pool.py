"""
Thread-pool fan-out shared by both runners, with Ctrl-C and SIGTERM handling,
and the subprocess runner whose children an interrupt ends.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar

from .constants import CHILD_TERMINATE_GRACE_SECONDS

T = TypeVar("T")

_POSIX = os.name == "posix"
#: ``signal.SIGKILL`` where it exists (POSIX); Windows kills with ``Popen.kill``.
_SIGKILL = getattr(signal, "SIGKILL", signal.SIGTERM)


@contextmanager
def sigterm_as_interrupt() -> Iterator[None]:
    """For the duration of the block, SIGTERM raises ``KeyboardInterrupt``.

    Interface contract (caller: :func:`run_interruptible`): ``timeout(1)``, a
    service manager and ``kill`` send SIGTERM, which would end the process
    before the partial report is written; as a ``KeyboardInterrupt`` it takes
    the Ctrl-C path (queued work cancelled, report written, exit 130). The
    previous SIGTERM handler is restored on exit, also when the block raises.
    Outside the main thread nothing is installed (``signal.signal`` only works
    there), and a SIGTERM the process deliberately ignores (``SIG_IGN``) stays
    ignored, so the block just runs. Never raises by itself.
    """
    if (
        threading.current_thread() is not threading.main_thread()
        or signal.getsignal(signal.SIGTERM) is signal.SIG_IGN
    ):
        yield
        return

    def _raise_interrupt(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, _raise_interrupt)
    try:
        yield
    finally:
        # ``None``: the old handler was installed from C, so it cannot be put back.
        signal.signal(signal.SIGTERM, signal.SIG_DFL if previous is None else previous)


class PoolInterrupted(RuntimeError):
    """A child was refused because the run it belongs to was interrupted."""


def _signal_child(proc: subprocess.Popen[str], sig: signal.Signals) -> None:
    """Send ``sig`` to the child's whole process group (POSIX) or to the child.

    ``run_tracked`` starts every child in its own session, so its process group
    id is its pid and the example's own subprocesses go with it. A child that is
    already reaped is left alone (its pid could have been reused). Never raises.
    """
    if proc.returncode is not None:
        return
    try:
        if _POSIX:
            os.killpg(proc.pid, sig)
        elif sig == _SIGKILL:
            proc.kill()
        else:
            proc.terminate()
    except (ProcessLookupError, PermissionError, OSError):
        pass


class _ChildScope:
    """The example subprocesses of ONE ``run_interruptible`` call.

    Interface contract (caller: ``run_interruptible``; workers reach it through
    :func:`run_tracked`): ``add`` registers a running child and raises
    :class:`PoolInterrupted` once ``terminate_all`` ran, so a child spawned by a
    worker that was mid-spawn at the interrupt is refused, not left running
    (the caller kills it). ``terminate_all`` SIGTERMs every registered child's
    process group, waits up to ``grace`` seconds for them to exit, SIGKILLs the
    rest and returns. Never raises. Thread-safe.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._children: set[subprocess.Popen[str]] = set()
        self._closed = False

    def add(self, proc: subprocess.Popen[str]) -> None:
        with self._lock:
            if self._closed:
                raise PoolInterrupted("the run was interrupted")
            self._children.add(proc)

    def discard(self, proc: subprocess.Popen[str]) -> None:
        with self._lock:
            self._children.discard(proc)

    def terminate_all(self, grace: float = CHILD_TERMINATE_GRACE_SECONDS) -> None:
        with self._lock:
            self._closed = True
            children = list(self._children)
        for proc in children:
            _signal_child(proc, signal.SIGTERM)
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline and any(
            proc.poll() is None for proc in children
        ):
            time.sleep(0.02)
        for proc in children:
            if proc.poll() is None:
                _signal_child(proc, _SIGKILL)


_active = threading.local()


def run_tracked(
    args: Sequence[str],
    *,
    cwd: str | Path,
    env: Mapping[str, str],
    input: str | None,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    """``subprocess.run(args, capture_output, utf-8 text)`` whose child an interrupt ends.

    Interface contract (caller: ``examples.run_example``): same arguments and
    results as ``subprocess.run`` with ``capture_output=True, text=True,
    encoding="utf-8", errors="replace", check=False``: stdin is piped only when
    ``input`` is not ``None`` (else inherited), ``subprocess.TimeoutExpired``
    carries the partial output, any other failure to start raises as ``Popen``
    does. Differences: on POSIX the child runs in its own session, so a timeout
    or an interrupt ends its whole process group, and when called by a worker of
    ``run_interruptible`` the child is registered with that run, so an
    interrupt terminates it (SIGTERM, grace, SIGKILL). Called outside a pool the
    child is simply not registered. Raises :class:`PoolInterrupted` for a child
    refused after an interrupt.
    """
    scope: _ChildScope | None = getattr(_active, "scope", None)
    with subprocess.Popen(
        list(args),
        cwd=str(cwd),
        env=dict(env),
        stdin=subprocess.PIPE if input is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=_POSIX,
    ) as proc:
        try:
            if scope is not None:
                scope.add(proc)
            try:
                stdout, stderr = proc.communicate(input, timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                _signal_child(proc, _SIGKILL)
                if _POSIX:
                    proc.wait()
                else:  # pragma: no cover  (as subprocess.run does on Windows)
                    exc.stdout, exc.stderr = proc.communicate()  # type: ignore[assignment]
                raise
            except BaseException:
                _signal_child(proc, _SIGKILL)
                raise
        finally:
            if scope is not None:
                scope.discard(proc)
    return subprocess.CompletedProcess(list(args), proc.returncode, stdout, stderr)


def run_interruptible(
    work: Callable[[T], Any],
    items: Iterable[T],
    workers: int,
    on_done: Callable[[T, Future[Any]], None],
) -> bool:
    """Run ``work(item)`` for every item on ``workers`` threads.

    Interface contract (callers: ``examples.run_examples``, ``cases.run_cases``):
        - SIGTERM during the run counts as Ctrl-C (:func:`sigterm_as_interrupt`;
          only from the main thread).
        - ``on_done(item, future)`` runs on the calling thread, in completion
          order; the future is done, so ``future.result()`` does not block.
        - Returns ``False`` when every item finished, ``True`` when a
          ``KeyboardInterrupt`` (Ctrl-C or SIGTERM) arrived: queued items are
          cancelled and never reach ``on_done``, and the pool is shut down
          without waiting (items already running finish in the background,
          unreported). The subprocesses those items started through
          :func:`run_tracked` are terminated first (SIGTERM to the process
          group, ``CHILD_TERMINATE_GRACE_SECONDS``, then SIGKILL), so their
          threads end at once and the process can exit. Work that runs in the
          calling process (the case runner's trials) cannot be stopped: it
          finishes in the background.
        - Any other exception from ``on_done`` cancels the queue, ends the
          children the same way and propagates.
    """
    scope = _ChildScope()

    def scoped(item: T) -> Any:
        _active.scope = scope
        try:
            return work(item)
        finally:
            _active.scope = None

    with sigterm_as_interrupt():
        pool = ThreadPoolExecutor(max_workers=workers)
        finished = False
        try:
            futures = {pool.submit(scoped, item): item for item in items}
            for future in as_completed(futures):
                on_done(futures[future], future)
            finished = True
        except KeyboardInterrupt:
            return True
        finally:
            pool.shutdown(wait=finished, cancel_futures=not finished)
            if not finished:
                # DECISION plan-2026-11-06T093029-0068b055/D-028 (12.34): end
                # the running example subprocesses here, after the queue was
                # cancelled. Do NOT leave them to finish: the pool threads are
                # non-daemon and joined at interpreter exit, so the process
                # lingered up to the 360 s harness timeout after SIGTERM. Do
                # NOT signal only the child pid (its own subprocesses survive)
                # or track children in a module global (a pool-scoped registry
                # lets two runs, or a direct run_example, never see each other).
                scope.terminate_all()
        return False
