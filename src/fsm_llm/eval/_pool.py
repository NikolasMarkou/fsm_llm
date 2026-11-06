"""
Thread-pool fan-out shared by both runners, with Ctrl-C and SIGTERM handling.
"""

from __future__ import annotations

import signal
import threading
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import Any, TypeVar

T = TypeVar("T")


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
          unreported).
        - Any other exception from ``on_done`` cancels the queue and propagates.
    """
    with sigterm_as_interrupt():
        pool = ThreadPoolExecutor(max_workers=workers)
        finished = False
        try:
            futures = {pool.submit(work, item): item for item in items}
            for future in as_completed(futures):
                on_done(futures[future], future)
            finished = True
        except KeyboardInterrupt:
            return True
        finally:
            pool.shutdown(wait=finished, cancel_futures=not finished)
        return False
