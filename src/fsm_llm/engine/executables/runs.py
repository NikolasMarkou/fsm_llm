"""The run host: one daemon thread per delegated run, and the targets object that owns it.

Plan baa37f40 phase K7 (D-050, roadmap E3.2 second half). A ``call`` state in delegated mode runs a
code-backed ``pattern`` or ``workflow`` OFF the parent's turn, on a thread the engine does not own,
and polls it on later turns. This module holds the thread side only: it never logs, never reads the
parent's context and never calls back into an ``API`` (the engine writes ``run/start`` and
``run/end`` itself, on its own thread, through the ``announce`` hook of ``start``).

Interface contract:

- ``RunHost(*, max_runs=DEFAULT_MAX_RUNS)``: the thread-per-run host. Handles are non-blank
  ``str`` of at most 128 characters; a conversation id is a non-blank ``str``; a violation raises
  ``ValueError``. Every method is safe from any thread and never raises for an unknown handle.
  - ``claim(handle, *, conversation_id) -> bool``: reserve a handle for a conversation (bounded:
    at most ``MAX_RESERVED_RUNS``, the oldest reservation is evicted when full, so the stale
    reservation of a rolled-back turn costs nothing lasting). ``False`` for a handle already known
    and while ``max_runs`` runs are live (the call's ``failed/push_failed``). Makes the run's
    ``CancelToken``.
  - ``start(handle, body, *, conversation_id, announce=None) -> bool``: consume the reservation of
    THAT conversation (``False``: no reservation, another conversation's, already started, or
    ``max_runs`` reached by then), call ``announce()`` OUTSIDE the lock (an exception drops the run
    and propagates: no thread was started), then start one DAEMON thread that runs
    ``asyncio.run(body(token))`` on a fresh empty ``contextvars.Context`` (so ``aio.SYNC`` and a
    turn-gate ticket of the starting thread never reach it). ``body`` is ``async (CancelToken) ->
    RunResult``. A thread that cannot start drops the run and returns ``False``.
  - ``poll(handle, *, conversation_id) -> RunPoll | None``: ``RunPoll(False, None)`` while reserved
    or running, ``RunPoll(True, result)`` once finished, ``None`` when the host never saw the handle
    FOR THAT CONVERSATION (or it was evicted or dropped): the ``run_lost`` answer.
  - ``cancel(handle, *, conversation_id=None) -> bool``: set the token of a reserved or running
    run (a run is checked BETWEEN steps; a pattern is not interruptible and may keep running: it is
    detached, never joined). ``False`` for an unknown, finished or foreign handle.
  - ``cancel_for(conversation_id) -> int``: set the token of every running run of the conversation,
    drop its reservations and finished results, and orphan the running ones (their result is
    dropped at the finish). Returns the number of running runs told to stop.
  - ``wait(handle, timeout_seconds) -> bool``: one bounded ``Event.wait`` for a run to finish,
    ``False`` on a timeout or an unknown handle. It never joins a thread.
- A worker exception of ANY kind (the body's, ``CancelledError`` and a wrong return value included)
  is stored as ``RunResult(failed, unexpected_error, detail={"error": <exception type name>})``:
  the TYPE only, never its text or arguments (census D6).
- ``ExecutableTargets(registry, context=None, *, host=None)``: the object behind the one
  ``API(targets=)``. It is a ``TargetResolver`` (``resolve_target`` is the registry's frame
  resolution) and a ``RunLauncher`` (``resolve_run`` is ``registry.resolve_delegated`` judged
  against ``context``, the HOST context: its ``callables``, limits and model; the others go to
  the ``RunHost``). ``start(handle, target, inputs, *, conversation_id, max_seconds=None,
  announce=None)`` runs ``target.executable`` in the worker: it instantiates it there and runs it
  with ``context.derive(parent=conversation_id, max_seconds=<the narrower of the call's and the
  host's>, cancel=<the run's own token>)``. The caller passes ``inputs`` as a private copy (the
  worker never shares a mutable value with the parent's context). A host context that sets
  ``max_steps`` makes every pattern fail with the mapped type of its ``ValueError``.
  ``wait(handle, timeout_seconds) -> bool`` is the host's bounded wait, offered so ``aadvance``
  can pace a run loop over a delegated parent (K7.4); it is not part of the ``RunLauncher`` Protocol.

Concurrency: the host lock is a LEAF. It guards dict and counter operations only: no log write, no
``announce``, no ``Thread.start``, no ``Event.set``, no manager lock and no model call run under
it, and this module imports no ``engine.api``, ``engine.fsm`` or ``engine.turn`` module. Nothing
here joins a thread: not at ``close()``, not in a test.

Imported by name (``from fsm_llm.engine.executables import ExecutableTargets``);
``import fsm_llm`` loads none of it.
"""

from __future__ import annotations

import asyncio
import contextvars
import threading
from collections import OrderedDict
from collections.abc import Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass, field
from threading import Event, Thread
from typing import Any

from fsm_llm.core.codes import Reason, Status
from fsm_llm.core.results import RunResult
from fsm_llm.engine.context import CancelToken, RunContext
from fsm_llm.engine.kinds.targets import DelegatedTarget, ResolvedTarget, RunPoll
from fsm_llm.logging import logger

from .registry import ExecutableRegistry

__all__ = [
    "DEFAULT_MAX_RUNS",
    "MAX_FINISHED_RUNS",
    "MAX_HANDLE_LENGTH",
    "MAX_RESERVED_RUNS",
    "ExecutableTargets",
    "RunHost",
]

#: The default of ``RunHost(max_runs=)``: delegated runs live at once (running threads).
DEFAULT_MAX_RUNS = 32
#: Hard cap on reservations that never started (a rolled-back turn leaves one); the oldest goes.
MAX_RESERVED_RUNS = 1024
#: Hard cap on finished results kept for a poll; the oldest goes (its poll is then ``None``).
MAX_FINISHED_RUNS = 256
#: The longest handle the host accepts.
MAX_HANDLE_LENGTH = 128

_Body = Callable[[CancelToken], Coroutine[Any, Any, RunResult]]


@dataclass(eq=False)
class _Run:
    """One run's record: owner, token, the event of its end and its result."""

    handle: str
    conversation_id: str
    token: CancelToken = field(default_factory=CancelToken)
    finished: Event = field(default_factory=Event)
    result: RunResult | None = None
    orphaned: bool = False


def _check_name(value: object, what: str, *, limit: int | None = None) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{what} must be a non-blank str, got {value!r}")
    if limit is not None and len(value) > limit:
        raise ValueError(f"{what} is longer than {limit} characters")
    return value


# DECISION plan-2026-11-23T122249-baa37f40/D-050
# The host is a fire-and-cancel design with NO join. Its lock is a leaf (dict and counter work
# only: the `announce` log write, `Thread.start` and `Event.set` all run outside it), the thread is
# a daemon that runs on a fresh empty Context, and a stop is the per-run token, never a wait for
# the thread. Do NOT call `Thread.join` (a timed join on a thread that may wait on a lock the
# joiner holds hung Windows 3.15 in plan ace7a8c8), do NOT nest a manager lock or import
# `engine.api`/`engine.fsm` here (a worker that calls back into the parent API breaks the one-turn
# rule), do NOT store an exception's text (type only), and do NOT start the thread before
# `announce` (the `run/start` record must be on disk first).
class RunHost:
    """One daemon thread per delegated run. See the module docstring."""

    def __init__(self, *, max_runs: int = DEFAULT_MAX_RUNS) -> None:
        if isinstance(max_runs, bool) or not isinstance(max_runs, int) or max_runs < 1:
            raise ValueError("max_runs must be an integer >= 1")
        self._max_runs = max_runs
        self._lock = threading.Lock()
        self._reserved: OrderedDict[str, _Run] = OrderedDict()
        self._running: dict[str, _Run] = {}
        self._finished: OrderedDict[str, _Run] = OrderedDict()

    # ---- the engine side --------------------------------------------------------------

    def claim(self, handle: str, *, conversation_id: str) -> bool:
        """Reserve *handle* for *conversation_id* (see the module docstring)."""
        _check_name(handle, "handle", limit=MAX_HANDLE_LENGTH)
        _check_name(conversation_id, "conversation_id")
        with self._lock:
            if self._known_locked(handle) or len(self._running) >= self._max_runs:
                return False
            self._reserved[handle] = _Run(handle, conversation_id)
            while len(self._reserved) > MAX_RESERVED_RUNS:
                self._reserved.popitem(last=False)
        return True

    def start(
        self,
        handle: str,
        body: _Body,
        *,
        conversation_id: str,
        announce: Callable[[], None] | None = None,
    ) -> bool:
        """Consume the reservation, announce, then start the daemon thread."""
        with self._lock:
            run = self._reserved.get(handle)
            if run is None or run.conversation_id != conversation_id:
                return False
            del self._reserved[handle]
            if len(self._running) >= self._max_runs:
                return False
            self._running[handle] = run
        try:
            if announce is not None:
                announce()
        except BaseException:
            self._drop_running(run)
            raise
        thread = Thread(
            target=self._work,
            args=(run, body),
            name=f"fsm-llm-run-{handle}",
            daemon=True,
        )
        try:
            thread.start()
        except RuntimeError:  # the process cannot start another thread
            self._drop_running(run)
            logger.warning("a delegated run could not start its thread")
            return False
        return True

    def poll(self, handle: str, *, conversation_id: str) -> RunPoll | None:
        """What the host knows of *handle* for this conversation, or ``None``."""
        with self._lock:
            run = self._lookup_locked(handle)
            if run is None or run.conversation_id != conversation_id:
                return None
            finished = handle in self._finished
            result = run.result
        return RunPoll(True, result) if finished else RunPoll(False, None)

    def cancel(self, handle: str, *, conversation_id: str | None = None) -> bool:
        """Set the token of a reserved or running run of that conversation."""
        with self._lock:
            run = self._running.get(handle) or self._reserved.get(handle)
            if run is None or (
                conversation_id is not None and run.conversation_id != conversation_id
            ):
                return False
        run.token.cancel()
        return True

    def cancel_for(self, conversation_id: str) -> int:
        """Stop and forget every run of a conversation that ended; the number stopped."""
        tokens: list[CancelToken] = []
        with self._lock:
            for table in (self._reserved, self._finished):
                for handle in [
                    h for h, r in table.items() if r.conversation_id == conversation_id
                ]:
                    del table[handle]
            for run in self._running.values():
                if run.conversation_id == conversation_id:
                    run.orphaned = True
                    tokens.append(run.token)
        for (
            token
        ) in tokens:  # the Event of a token has its own lock: set it outside ours
            token.cancel()
        return len(tokens)

    def wait(self, handle: str, timeout_seconds: float) -> bool:
        """One bounded wait for the run to finish; ``False`` on a timeout or an unknown handle."""
        with self._lock:
            run = self._lookup_locked(handle)
        if run is None:
            return False
        return run.finished.wait(timeout_seconds)

    # ---- the worker side --------------------------------------------------------------

    def _work(self, run: _Run, body: _Body) -> None:
        contextvars.Context().run(self._drive, run, body)

    def _drive(self, run: _Run, body: _Body) -> None:
        try:
            result = asyncio.run(body(run.token))
            if not isinstance(result, RunResult):
                raise TypeError("a run body returns a RunResult")
        except BaseException as error:
            result = RunResult(
                status=Status.FAILED,
                reason=Reason.UNEXPECTED_ERROR,
                detail={"error": type(error).__name__},
            )
        self._finish(run, result)

    def _finish(self, run: _Run, result: RunResult) -> None:
        with self._lock:
            self._running.pop(run.handle, None)
            if not run.orphaned:
                run.result = result
                self._finished[run.handle] = run
                while len(self._finished) > MAX_FINISHED_RUNS:
                    self._finished.popitem(last=False)
        run.finished.set()

    def _drop_running(self, run: _Run) -> None:
        with self._lock:
            self._running.pop(run.handle, None)
        run.finished.set()

    def _known_locked(self, handle: str) -> bool:
        return (
            handle in self._reserved
            or handle in self._running
            or handle in self._finished
        )

    def _lookup_locked(self, handle: str) -> _Run | None:
        return (
            self._finished.get(handle)
            or self._running.get(handle)
            or self._reserved.get(handle)
        )


class ExecutableTargets:
    """The registry and the run host behind the one ``API(targets=)``."""

    def __init__(
        self,
        registry: ExecutableRegistry,
        context: RunContext | None = None,
        *,
        host: RunHost | None = None,
    ) -> None:
        if not isinstance(registry, ExecutableRegistry):
            raise TypeError(
                f"registry must be an ExecutableRegistry, got {type(registry).__name__}"
            )
        if context is not None and not isinstance(context, RunContext):
            raise TypeError(
                f"context must be a RunContext, got {type(context).__name__}"
            )
        if host is not None and not isinstance(host, RunHost):
            raise TypeError(f"host must be a RunHost, got {type(host).__name__}")
        self._registry = registry
        self._context = RunContext() if context is None else context
        self._host = RunHost() if host is None else host

    @property
    def registry(self) -> ExecutableRegistry:
        return self._registry

    @property
    def context(self) -> RunContext:
        return self._context

    @property
    def host(self) -> RunHost:
        return self._host

    # ---- TargetResolver ---------------------------------------------------------------

    def resolve_target(
        self, target_id: str, *, conversation_id: str | None = None
    ) -> ResolvedTarget | None:
        """Frame resolution: the registry's, unchanged."""
        return self._registry.resolve_target(target_id, conversation_id=conversation_id)

    # ---- RunLauncher ------------------------------------------------------------------

    def resolve_run(
        self,
        target_id: str,
        *,
        input_keys: Sequence[str] = (),
        max_steps: int | None = None,
        conversation_id: str | None = None,
    ) -> DelegatedTarget | None:
        """Delegated resolution, judged against this object's host context."""
        return self._registry.resolve_delegated(
            target_id,
            launcher=self,
            context=self._context,
            input_keys=input_keys,
            max_steps=max_steps,
        )

    def claim(self, handle: str, *, conversation_id: str) -> bool:
        return self._host.claim(handle, conversation_id=conversation_id)

    def start(
        self,
        handle: str,
        target: DelegatedTarget,
        inputs: Mapping[str, Any],
        *,
        conversation_id: str,
        max_seconds: float | None = None,
        announce: Callable[[], None] | None = None,
    ) -> bool:
        if not isinstance(target, DelegatedTarget):
            raise TypeError(
                f"target must be a DelegatedTarget, got {type(target).__name__}"
            )
        base = self._context
        private = dict(inputs)
        limit = max_seconds
        if limit is not None and base.max_seconds is not None:
            limit = min(limit, base.max_seconds)
        executable = target.executable

        async def body(token: CancelToken) -> RunResult:
            context = base.derive(
                parent=conversation_id, max_seconds=limit, cancel=token
            )
            instance = executable.instantiate()
            result: RunResult = await instance.arun(private, context=context)
            return result

        return self._host.start(
            handle, body, conversation_id=conversation_id, announce=announce
        )

    def poll(self, handle: str, *, conversation_id: str) -> RunPoll | None:
        return self._host.poll(handle, conversation_id=conversation_id)

    def wait(self, handle: str, timeout_seconds: float) -> bool:
        """One bounded wait for a run to finish (``RunHost.wait``): the optional method of a
        launcher that ``aadvance`` waits through once per step (``engine/api/delegated.py``)."""
        return self._host.wait(handle, timeout_seconds)

    def cancel(self, handle: str, *, conversation_id: str | None = None) -> bool:
        return self._host.cancel(handle, conversation_id=conversation_id)

    def cancel_for(self, conversation_id: str) -> int:
        return self._host.cancel_for(conversation_id)
