"""Write a pipeline once as ``async def`` and run it from sync code with no event loop.

Kernel module (standard library only). A coroutine that never really suspends is stepped
with ``send(None)`` on the caller's thread: :func:`run_sync` and :func:`iterate_sync` bind the
``SYNC`` flag while they step, and :func:`acall` reads it. In sync mode ``acall`` calls the
SYNC method of an object (so a patch of that method still intercepts); in async mode it
awaits the ``a`` twin, else runs the sync method on a worker thread. A coroutine that really
suspends in sync mode is a bug and raises :class:`SuspendedInSyncModeError`.

Known limit: ``SYNC`` is a ``ContextVar``, so whatever copies the current Context while it is
set takes the sync path: ``asyncio.run`` or ``create_task`` called inside a sync-mode step and
``copy_context().run`` do (it works, and blocks that loop). A plain thread starts with an
empty Context (probed on 3.11, 3.14 and 3.15), so it does not.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    Coroutine,
    Iterable,
    Iterator,
)
from contextvars import ContextVar
from typing import Any, TypeVar, cast

__all__ = [
    "SYNC",
    "SuspendedInSyncModeError",
    "acall",
    "aiterate",
    "async_name",
    "iterate_sync",
    "run_sync",
]

T = TypeVar("T")

#: True only while ``run_sync`` or ``iterate_sync`` is stepping, never across a ``yield``.
SYNC: ContextVar[bool] = ContextVar("fsm_llm_sync", default=False)

_DONE = object()


class SuspendedInSyncModeError(RuntimeError):
    """A coroutine driven by ``run_sync`` really suspended: a bug, never a state."""


def async_name(name: str) -> str:
    """The async twin of a method name: ``generate_response`` to ``agenerate_response``,
    ``_send`` to ``_asend``."""
    bare = name.lstrip("_")
    return name[: len(name) - len(bare)] + "a" + bare


async def acall(obj: Any, name: str, /, *args: Any, **kwargs: Any) -> Any:
    """Call ``obj.<name>`` through the transport the current mode asks for.

    Sync mode (``SYNC`` set): ``obj.<name>`` itself, no suspension. Async mode: ``await
    obj.<async_name(name)>`` when it exists, else ``obj.<name>`` on a worker thread
    (``asyncio.to_thread``; a cancel returns at once while the worker keeps running).
    """
    if SYNC.get():
        return getattr(obj, name)(*args, **kwargs)
    twin = getattr(obj, async_name(name), None)
    if twin is not None:
        return await twin(*args, **kwargs)
    return await asyncio.to_thread(getattr(obj, name), *args, **kwargs)


def _step(awaitable: Any) -> Any:
    """Run an awaitable to its end on this thread with no loop; raise if it suspends."""
    try:
        suspended_on = awaitable.send(None)
    except StopIteration as done:
        return done.value
    # DECISION plan-2026-11-19T025425-51fde95b/D-004: end a suspended step with
    # throw(GeneratorExit), never close(). On Python 3.11 and 3.12 close() on an asend()
    # awaitable leaves the async generator "already running" (fixed in 3.14). Raise the
    # error after this cleanup and outside any finally, so it hides no other error.
    try:
        awaitable.throw(GeneratorExit)
    except (GeneratorExit, StopIteration, StopAsyncIteration):
        pass
    raise SuspendedInSyncModeError(
        f"{awaitable!r} suspended on {suspended_on!r} in sync mode"
    )


def _drive(awaitable: Any) -> Any:
    token = SYNC.set(True)
    try:
        return _step(awaitable)
    finally:
        SYNC.reset(token)


def run_sync(work: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine to completion on the caller's thread with no event loop.

    Returns what it returns and raises what it raises (``KeyboardInterrupt`` and
    ``SystemExit`` too); ``SYNC`` is restored. Anything that is not a coroutine (a test
    double's plain value) is returned as is. Raises :class:`SuspendedInSyncModeError` if
    the coroutine really suspends, after running its cleanup.
    """
    if not inspect.iscoroutine(work):
        return cast(T, work)
    try:
        return cast(T, _drive(work))
    finally:
        if inspect.getcoroutinestate(work) == inspect.CORO_CREATED:
            work.close()  # never started: no "never awaited" warning


def iterate_sync(agen: AsyncGenerator[T, None] | Iterator[T]) -> Iterator[T]:
    """Drive an async generator one ``next()`` at a time on the caller's thread.

    Lazy: nothing runs before the first ``next()``. ``SYNC`` is bound only WHILE a step
    runs, so a consumer may hop Contexts between steps. ``close()`` (or dropping the
    iterator) runs the generator's ``GeneratorExit`` arm and ``finally``. Anything that is
    not an async generator is iterated as is.
    """
    if not isinstance(agen, AsyncGenerator):
        yield from agen
        return
    try:
        while True:
            try:
                item = _drive(agen.asend(None))
            except StopAsyncIteration:
                return
            yield item
    finally:
        _drive(agen.aclose())


class _Stepper:
    """Steps a sync iterator and closes it once no worker step is inside it.

    Interface contract: ``step()`` returns the next item or the ``_DONE`` marker (it raises
    what the iterator raises) and may run inline or on a worker thread; ``close()`` runs on
    any thread, once, and never raises ``ValueError``: a generator cannot be closed while a
    worker is inside it, so a ``close()`` during a step is deferred to the end of that step.
    """

    def __init__(self, it: Iterator[Any]) -> None:
        self._it = it
        self._lock = threading.Lock()
        self._running = False
        self._closing = False
        self._closed = False

    def step(self) -> Any:
        with self._lock:
            if self._closed:
                return _DONE
            self._running = True
        try:
            return next(self._it, _DONE)
        finally:
            with self._lock:
                self._running = False
                deferred = self._closing
            if deferred:
                self._close_iterator()

    def close(self) -> None:
        with self._lock:
            if self._running:
                self._closing = True
                return
            self._closed = True
        self._close_iterator()

    def _close_iterator(self) -> None:
        close = getattr(self._it, "close", None)
        if close is not None:
            close()


async def aiterate(source: Iterable[T] | AsyncIterable[T]) -> AsyncGenerator[T, None]:
    """Iterate a sync or async iterable from async code, closing a sync source on exit.

    An async iterable is iterated natively and ``aclose``d on exit. A sync one is stepped
    inline in sync mode and one ``next`` per worker thread in async mode, so the loop is
    never blocked. A source that is both (litellm's stream wrapper) follows the mode: its
    sync side in sync mode, where there is no loop to run its async side on. A cancel or a
    timeout during a worker step stays that exception at once; the sync source is closed
    when that step ends (a worker is not stopped, as for any ``to_thread`` call).
    """
    if isinstance(source, AsyncIterable) and not (
        SYNC.get() and isinstance(source, Iterable)
    ):
        ait = source.__aiter__()
        try:
            async for item in ait:
                yield item
        finally:
            aclose = getattr(ait, "aclose", None)
            if aclose is not None:
                await aclose()
        return
    # DECISION plan-2026-11-19T025425-51fde95b/D-043: never close a sync source from the loop
    # thread while a worker step is inside it (ValueError "generator already executing" would
    # replace the cancel), and never await that step to end it (a cancel would wait for the
    # source). `_Stepper` defers the close to the end of the step. See decisions.md D-043.
    stepper = _Stepper(iter(source))
    try:
        while True:
            step = (
                stepper.step() if SYNC.get() else await asyncio.to_thread(stepper.step)
            )
            if step is _DONE:
                return
            yield cast(T, step)
    finally:
        stepper.close()
