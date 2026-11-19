"""Write a pipeline once as ``async def`` and run it from sync code with no event loop.

Kernel module (standard library only). A coroutine that never really suspends is stepped
with ``send(None)`` on the caller's thread: :func:`run_sync` and :func:`iterate_sync` bind the
``SYNC`` flag while they step, and :func:`acall` reads it. In sync mode ``acall`` calls the
SYNC method of an object (so a patch of that method still intercepts); in async mode it
awaits the ``a`` twin, else runs the sync method on a worker thread. A coroutine that really
suspends in sync mode is a bug and raises :class:`SuspendedInSyncModeError`.

Known limit: a thread started while ``SYNC`` is set copies the flag, so an ``asyncio.run``
inside it takes the sync path (it works, and blocks that loop).
"""

from __future__ import annotations

import asyncio
import inspect
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


def _next(it: Iterator[Any]) -> Any:
    return next(it, _DONE)


async def aiterate(source: Iterable[T] | AsyncIterable[T]) -> AsyncGenerator[T, None]:
    """Iterate a sync or async iterable from async code, closing a sync source on exit.

    An async iterable is iterated natively. A sync one is stepped inline in sync mode and
    one ``next`` per worker thread in async mode, so the loop is never blocked.
    """
    if isinstance(source, AsyncIterable):
        async for item in source:
            yield item
        return
    it = iter(source)
    try:
        while True:
            step = _next(it) if SYNC.get() else await asyncio.to_thread(_next, it)
            if step is _DONE:
                return
            yield cast(T, step)
    finally:
        close = getattr(it, "close", None)
        if close is not None:
            close()
