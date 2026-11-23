"""
The stream of one reasoning solve
=================================

``ReasoningEngine.solve_problem_stream`` returns a ``SolveStream`` (plan baa37f40 step 19,
D-036): the reply text of one solve as it arrives, strategy replies included, and the
``(solution, trace_info)`` tuple of ``solve_problem`` once the stream is exhausted.
"""

from __future__ import annotations

import threading
from collections.abc import Generator, Iterator
from types import TracebackType
from typing import Any

from .exceptions import ReasoningExecutionError

#: The ``(solution, trace_info)`` tuple ``solve_problem`` returns.
SolveResult = tuple[str, dict[str, Any]]


# DECISION plan-2026-11-23T122249-baa37f40/D-046
# The solve lock is held across the yields of a stream, so the thread that holds it must never wait
# for it: it is told, by name. Do NOT go back to a bare `threading.Lock` (a consumer that solves
# again inside its own `for` loop hung forever with no error) and do NOT take the lock from
# another thread's id: only the owner reads its own id, so no second lock guards `_owner`.
class SolveLock:
    """The lock that serialises the solves of one engine, with the thread that holds it.

    Interface contract: ``acquire(blocking=True) -> bool``, ``release()``, ``locked() -> bool`` and
    ``with lock:`` as for ``threading.Lock``. One difference: a BLOCKING ``acquire`` (and ``with``)
    by the thread that already holds it raises ``ReasoningExecutionError`` instead of waiting for
    itself; ``acquire(blocking=False)`` by that thread answers ``False`` as a ``Lock`` does; another
    thread waits its turn. ``release`` by a thread that does not hold the lock is the ``Lock``
    error (``RuntimeError``).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._owner: int | None = None

    def acquire(self, blocking: bool = True) -> bool:
        if blocking and self._owner == threading.get_ident():
            raise ReasoningExecutionError(
                "a solve of this engine is still open on this thread (an unclosed solve "
                "stream?): close it before solving again"
            )
        taken = self._lock.acquire(blocking)
        if taken:
            self._owner = threading.get_ident()
        return taken

    def release(self) -> None:
        self._owner = None
        self._lock.release()

    def locked(self) -> bool:
        return self._lock.locked()

    def __enter__(self) -> SolveLock:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()


class SolveStream(Iterator[str]):
    """An iterator over the reply text of one solve, with the result at its end.

    Interface contract: made by ``ReasoningEngine.solve_problem_stream`` only. Iterating yields
    the start greeting (when it is not empty) and then every chunk of every step, a pushed
    strategy FSM's included, in order; nothing runs before the first ``next()`` and no thread
    is started. ``result()`` returns the ``(solution, trace_info)`` tuple of ``solve_problem``
    once the iterator ended, the same object every time, and raises ``ReasoningExecutionError``
    before that, after ``close()``, and after the solve failed (chained from the error the
    iteration raised). The engine's solve lock is held from the first ``next()`` until the
    stream ends, fails or is closed; ``close()``, leaving a ``with`` block and dropping the
    stream all close it (as for any generator), and a stream that is only abandoned while a
    reference to it lives keeps the lock. One thread at a time may iterate it.
    """

    def __init__(self, solve: Generator[str, None, SolveResult]) -> None:
        self._solve = solve
        self._result: SolveResult | None = None
        self._error: BaseException | None = None
        self._closed = False

    def __next__(self) -> str:
        if self._result is not None or self._error is not None or self._closed:
            raise StopIteration
        try:
            return next(self._solve)
        except StopIteration as done:
            self._result = done.value
            raise
        except BaseException as error:
            self._error = error
            raise

    def result(self) -> SolveResult:
        """The ``(solution, trace_info)`` tuple, once the stream has been exhausted."""
        if self._result is not None:
            return self._result
        if self._error is not None:
            raise ReasoningExecutionError(
                "the solve failed, so the stream has no result"
            ) from self._error
        reason = (
            "was closed before the solve finished"
            if self._closed
            else "is not finished"
        )
        raise ReasoningExecutionError(
            f"the solve stream {reason}: iterate it to its end before asking for the result"
        )

    def close(self) -> None:
        """End the solve now: release the engine lock and end the orchestrator conversation."""
        if self._result is None and self._error is None:
            self._closed = True
        self._solve.close()

    def __enter__(self) -> SolveStream:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
