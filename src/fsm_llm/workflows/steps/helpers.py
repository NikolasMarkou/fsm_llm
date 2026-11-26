"""What every step module shares: the step executor, the user-callable call and the error text.

``_STEP_EXECUTOR`` is the one ``ContextVar`` the engine driver sets around a step and
``_call_user_callable`` and ``ConversationStep`` read: it is defined here once and never rebound,
so every reader sees the same object. Imports no other module of the package.

Part of the package (plan d454fd62 step 92; the old ``workflows/steps.py`` at the same dotted path).
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import inspect
from collections.abc import Callable, Iterator
from concurrent.futures import Executor
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fsm_llm.engine.context import RunContext

#: Executor used for synchronous user callables. The engine sets it for the
#: duration of a step when it was constructed with ``executor=...``; ``None``
#: means the event loop's default executor.
_STEP_EXECUTOR: contextvars.ContextVar[Executor | None] = contextvars.ContextVar(
    "fsm_llm_workflows_step_executor", default=None
)

# DECISION plan-2026-11-25T195353-7d9c0955/D-007
# The run context of the run that owns a step, read only by `ExecutableStep` (it derives its child from
# it) and set only through `run_context_scope`, which resets in `finally`. Do NOT build a bare
# RunContext in the step when it is set, give the child a fresh CancelToken, set it anywhere else,
# or widen a limit past the parent's (derive raises, no clamp).
_RUN_CONTEXT: contextvars.ContextVar[RunContext | None] = contextvars.ContextVar(
    "fsm_llm_workflows_run_context", default=None
)


#: How many adapter runs enclose the code that reads it: `run_context_scope` is one level.
_RUN_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar(
    "fsm_llm_workflows_run_depth", default=0
)


@contextmanager
def run_context_scope(context: RunContext | None) -> Iterator[None]:
    """Make ``context`` the parent of every ``ExecutableStep`` run inside the block (one level deeper).

    ``None`` is a no-op: an ambient outer scope stays as it is, at its own depth.
    """
    if context is None:
        yield
        return
    token, depth = _RUN_CONTEXT.set(context), _RUN_DEPTH.set(_RUN_DEPTH.get() + 1)
    try:
        yield
    finally:
        _RUN_DEPTH.reset(depth)
        _RUN_CONTEXT.reset(token)


def _is_async_callable(fn: Any) -> bool:
    """True for ``async def`` functions (incl. partials) and objects whose
    ``__call__`` is ``async def``."""
    if inspect.iscoroutinefunction(fn):
        return True
    call = getattr(fn, "__call__", None)  # noqa: B004
    return call is not None and inspect.iscoroutinefunction(call)


async def _call_user_callable(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call a user callable without blocking the event loop.

    Async callables are awaited directly. Everything else runs in the step
    executor, and if it RETURNS an awaitable (a lambda wrapping an async
    function, a sync factory returning a coroutine) that awaitable is awaited
    too, so a coroutine object is never mistaken for a plain value.

    # DECISION plan-2026-09-27T120000-5d1e7a3b/D-004
    # Do NOT go back to `inspect.iscoroutinefunction(fn)` alone: a lambda that
    # returns a coroutine was run in the executor and its (truthy, never
    # awaited) coroutine object was used as the result, so a ConditionStep
    # always took `true_step_id` and an APIStep silently mapped nothing.

    Note: a synchronous callable that is still running when a timeout fires
    cannot be stopped (Python threads are not cancellable); it keeps running
    in its worker thread. Keep sync callables short or make them async.
    """
    if _is_async_callable(fn):
        result = fn(*args, **kwargs)
    else:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            _STEP_EXECUTOR.get(), functools.partial(fn, *args, **kwargs)
        )
    if inspect.isawaitable(result):
        result = await result
    return result


def _error_text(exc: BaseException) -> str:
    """Non-empty error text for an exception (``str()`` can be empty)."""
    text = str(exc)
    return text if text else type(exc).__name__
