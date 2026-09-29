"""
Thread-pool fan-out shared by both runners, with Ctrl-C handling.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any, TypeVar

T = TypeVar("T")


def run_interruptible(
    work: Callable[[T], Any],
    items: Iterable[T],
    workers: int,
    on_done: Callable[[T, Future[Any]], None],
) -> bool:
    """Run ``work(item)`` for every item on ``workers`` threads.

    Interface contract (callers: ``examples.run_examples``, ``cases.run_cases``):
        - ``on_done(item, future)`` runs on the calling thread, in completion
          order; the future is done, so ``future.result()`` does not block.
        - Returns ``False`` when every item finished, ``True`` when a
          ``KeyboardInterrupt`` arrived: queued items are cancelled and never
          reach ``on_done``, and the pool is shut down without waiting (items
          already running finish in the background, unreported).
        - Any other exception from ``on_done`` cancels the queue and propagates.
    """
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
