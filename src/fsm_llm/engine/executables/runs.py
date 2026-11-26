"""The run host: one daemon thread per delegated run, and the targets object that owns it.

Plan baa37f40 phase K7 (D-050, roadmap E3.2 second half). A ``call`` state in delegated mode runs a
code-backed ``pattern`` OFF the parent's turn, on a thread the engine does not own,
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
    ``max_runs`` reached by then), call ``announce()`` OUTSIDE the lock (an exception puts the
    reservation back and propagates: no thread was started, the next ``start`` of the handle may
    try again, D-057), then start one DAEMON thread that runs
    ``asyncio.run(body(token))`` on a fresh empty ``contextvars.Context`` (so ``aio.SYNC`` and a
    turn-gate ticket of the starting thread never reach it). ``body`` is ``async (CancelToken) ->
    RunResult``. A thread that cannot start drops the run and returns ``False``.
  - ``poll(handle, *, conversation_id) -> RunPoll | None``: ``RunPoll(False, None)`` while reserved
    or running, ``RunPoll(True, result)`` once finished, ``None`` when the host never saw the handle
    FOR THAT CONVERSATION (or it was evicted or dropped): the ``run_lost`` answer.
  - ``cancel(handle, *, conversation_id=None) -> bool``: set the token of a reserved or running
    run (the token of the context derived for the run: an agent class that takes a context, all but
    ``self_consistency``, ``swarm`` and ``meta_builder``, and ``reasoning.solve`` read it between
    their own steps and end ``aborted/cancelled`` at the next one, never mid model call or tool;
    the three others check it ONCE, before the run starts, so a cancel after the start only
    detaches them: they may keep running and are never joined; D-058). ``False`` for an unknown, finished or foreign handle.
  - ``cancel_for(conversation_id) -> int``: set the token of every running run of the conversation,
    drop its reservations and finished results, and orphan the running ones (their result is
    dropped at the finish). Returns the number of running runs told to stop.
  - ``forget(handle, *, conversation_id) -> bool``: drop the FINISHED result of a run whose record
    landed (``False``: unknown, not finished or another conversation's); called by ``settle``
    through ``ExecutableTargets.forget`` once the committed record is final (D-057).
  - ``wait(handle, timeout_seconds) -> bool``: one bounded ``Event.wait` for a run to finish,
    ``False`` on a timeout or an unknown handle. It never joins a thread.
- A worker exception of ANY kind (the body's, ``CancelledError`` and a wrong return value included)
  is stored as ``RunResult(failed, unexpected_error, detail={"error": <exception type name>})``:
  the TYPE only, never its text or arguments (census D6).
- ``ExecutableTargets(registry, context=None, *, host=None)``: the object behind the one
  ``API(targets=)``. It is a ``TargetResolver`` (``resolve_target`` is the registry's frame
  resolution) and a ``RunLauncher`` (``resolve_run`` is ``registry.resolve_delegated`` judged
  against ``context``, the HOST context: its ``callables``, limits and model; the others go to
  the ``RunHost``). ``start(handle, target, inputs, *, conversation_id, max_seconds=None,
  max_steps=None, announce=None)`` runs ``target.executable`` in the worker: it instantiates it there and runs it
  with ``context.derive(parent=conversation_id, max_steps=<the narrower of the call's and the
  host's>, max_seconds=<the narrower of the call's and the host's>, cancel=<the run's own
  token>)`` (D-055: the call's step budget reaches an executable that accepts one). The caller
  passes ``inputs`` as a private copy (the worker never shares a mutable value with the parent's
  context). A host context that sets
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
from fsm_llm.engine.kinds.call import bounded_value
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
#: Hard cap on finished results kept until their record lands (``forget``) or their conversation
#: ends; results not landed yet count toward it and the oldest goes (its poll is then ``None``).
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
    identity: tuple[str | None, str | None] = (None, None)


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
# `announce` (the `run/start` record must be on disk first). When `announce` raises, the
# reservation goes BACK (D-057: do NOT drop the run, a transient log failure would end a call that
# never started as `run_lost`), and a start that cannot happen (full, no thread) stores a finished
# `failed/push_failed` result (D-058: do NOT drop it silently, the parent would read `run_lost`).
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
        identity: tuple[str | None, str | None] = (None, None),
        announce: Callable[[], None] | None = None,
    ) -> bool:
        """Consume the reservation, announce, then start the daemon thread."""
        with self._lock:
            run = self._reserved.get(handle)
            if run is None or run.conversation_id != conversation_id:
                return False
            del self._reserved[handle]
            run.identity = identity
            if len(self._running) >= self._max_runs:
                self._refuse_locked(run)
                full = True
            else:
                self._running[handle] = run
                full = False
        if full:
            run.finished.set()  # a poll answers the finished refusal (push_failed)
            return False
        try:
            if announce is not None:
                announce()
        except BaseException:
            self._unstart(run)  # nothing started: the reservation goes back (D-057)
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
            with self._lock:
                self._running.pop(run.handle, None)
                self._refuse_locked(run)
            run.finished.set()
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
            template_identity, instance_identity = run.identity
        return RunPoll(
            finished, result if finished else None, template_identity, instance_identity
        )

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

    def forget(self, handle: str, *, conversation_id: str) -> bool:
        """Drop the FINISHED result of *handle* once its record landed; ``False`` for a handle
        that is unknown, not finished or another conversation's."""
        with self._lock:
            run = self._finished.get(handle)
            if run is None or run.conversation_id != conversation_id:
                return False
            del self._finished[handle]
        return True

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

    def _refuse_locked(self, run: _Run) -> None:
        """Store a FINISHED ``failed/push_failed`` result for a run that could not start (a full
        host, a thread the process cannot start): the poll answers it, settle ends the call at
        once and nothing is ever logged as started (D-058). Caller holds the lock."""
        run.result = RunResult(status=Status.FAILED, reason=Reason.PUSH_FAILED)
        self._finished[run.handle] = run
        while len(self._finished) > MAX_FINISHED_RUNS:
            self._finished.popitem(last=False)

    def _unstart(self, run: _Run) -> None:
        """An ``announce`` that failed: the run never started, so it is reserved again."""
        with self._lock:
            self._running.pop(run.handle, None)
            self._reserved[run.handle] = run
            while len(self._reserved) > MAX_RESERVED_RUNS:
                self._reserved.popitem(last=False)

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


def _bounded_result(result: RunResult) -> RunResult:
    """The result a host stores: every value of ``outputs`` and ``detail`` and the ``answer`` go
    through ``bounded_value`` (the 8000 character rule of the merge), so the finished table holds
    small results. Contract: never raises for a ``RunResult``; a value under the cap is a
    JSON-safe equal copy."""
    return result.model_copy(
        update={
            "outputs": {k: bounded_value(v) for k, v in result.outputs.items()},
            "answer": None if result.answer is None else bounded_value(result.answer),
            "detail": {k: bounded_value(v) for k, v in result.detail.items()},
        }
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
        max_steps: int | None = None,
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
        steps = max_steps
        if steps is not None and base.max_steps is not None:
            steps = min(steps, base.max_steps)
        executable = target.executable

        async def body(token: CancelToken) -> RunResult:
            context = base.derive(
                parent=conversation_id,
                max_steps=steps,
                max_seconds=limit,
                cancel=token,
            )
            instance = executable.instantiate()
            result: RunResult = await instance.arun(private, context=context)
            return _bounded_result(result)  # capped BEFORE the host stores it (D-057)

        return self._host.start(
            handle,
            body,
            conversation_id=conversation_id,
            identity=(target.template_identity, target.instance_identity),
            announce=announce,
        )

    def poll(self, handle: str, *, conversation_id: str) -> RunPoll | None:
        return self._host.poll(handle, conversation_id=conversation_id)

    def wait(self, handle: str, timeout_seconds: float) -> bool:
        """One bounded wait for a run to finish (``RunHost.wait``): the optional method of a
        launcher that ``aadvance`` waits through once per step (``engine/api/delegated.py``)."""
        return self._host.wait(handle, timeout_seconds)

    def forget(self, handle: str, *, conversation_id: str) -> bool:
        """Drop the finished result of a landed run (``RunHost.forget``): the optional method of a
        launcher that ``flush_run_ends`` calls once the committed record is final (D-057)."""
        return self._host.forget(handle, conversation_id=conversation_id)

    def cancel(self, handle: str, *, conversation_id: str | None = None) -> bool:
        return self._host.cancel(handle, conversation_id=conversation_id)

    def cancel_for(self, conversation_id: str) -> int:
        return self._host.cancel_for(conversation_id)
