"""``TurnGate``: the per-conversation lock owned by a ticket (plan f150ec28, steps 1.2 and 1.3).

The ``threading.RLock`` it replaces is owned by a thread, so every task of one loop is "the same
owner". The rows state who the owner is: a child task, a ``to_thread`` worker and a
``copy_context().run`` thread re-enter; a sibling task and a plain thread do not. Each row names
the mutant of the gate it kills: ownership by thread id (``sibling_task``), exclusion removed
(the second-owner and the waiting rows), a release that needs the current Context (the
release-by-ticket row), a wait on the loop thread (the fail-fast row; a watchdog frees the
holder so the mutant ends the test instead of hanging it), a ticket reset by token (the
asyncgen finalizer row), a binding never pruned (the pile-up row). The ``TestManagerLevel`` rows
have the shape of ``findings/probes/p1_reentry.py``. No sleeps and no clock assertion: events and
bounded joins only (Windows ticks 15.6 ms).
"""

from __future__ import annotations

import asyncio
import contextvars
import gc
import json
import sys
import threading
from collections.abc import Callable
from typing import Any

import pytest

from fsm_llm.core.definitions import ConversationBusyError, FSMDefinition, FSMError
from fsm_llm.engine.fsm import gate as gate_module
from fsm_llm.engine.fsm.gate import TurnGate
from fsm_llm.engine.fsm.manager import FSMManager
from fsm_llm.engine.fsm.turns import _claim_turn, _release_turn
from fsm_llm.handlers import HandlerTiming, create_handler
from fsm_llm.testing import ScriptedLLM

BOUND_SECONDS = 10.0
#: A build whose plain threads copy the Context cannot show the plain-thread row.
THREADS_INHERIT_CONTEXT = bool(getattr(sys.flags, "thread_inherit_context", 0))


def _in_thread(
    fn: Callable[..., Any], *args: Any, ctx: contextvars.Context | None = None
) -> Any:
    """Run ``fn(*args)`` on a new thread (in ``ctx`` when given) and return its result.

    A plain thread starts with an empty Context. The join is bounded: a hang fails the row.
    """
    box: dict[str, Any] = {}

    def body() -> None:
        try:
            box["value"] = fn(*args)
        except BaseException as exc:  # reported on the calling thread
            box["error"] = exc

    target = body if ctx is None else (lambda: ctx.run(body))
    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(BOUND_SECONDS)
    assert not worker.is_alive(), "the worker did not finish within the bound"
    if "error" in box:
        raise box["error"]
    return box.get("value")


def _probe(gate: TurnGate) -> bool:
    """A non-blocking acquire from the caller's Context; releases when it got in."""
    got = gate.acquire(blocking=False)
    if got:
        gate.release()
    return got


class _Watchdog:
    """Frees a holder after a bound, so a mutant that waits forever fails the row instead of hanging."""

    def __init__(self, gate: TurnGate, tickets: list[Any]) -> None:
        self._timer = threading.Timer(5.0, lambda: gate.release(tickets[0]))
        self._timer.daemon = True

    def __enter__(self) -> _Watchdog:
        self._timer.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._timer.cancel()


class TestExclusionAndReentry:
    def test_a_second_owner_is_refused_at_once(self):
        gate = TurnGate()
        assert gate.acquire(blocking=False)
        assert _in_thread(gate.acquire, False) is False
        assert _in_thread(gate.acquire, True, 0) is False
        assert gate.locked()
        gate.release()
        assert _in_thread(_probe, gate) is True  # free again: the same call now gets in
        assert not gate.locked()

    def test_the_owner_reenters_and_every_hold_needs_its_release(self):
        gate = TurnGate()
        assert gate.acquire() and gate.acquire(blocking=False)
        gate.release()
        assert gate.locked()
        gate.release()
        assert not gate.locked()
        with pytest.raises(RuntimeError):
            gate.release()

    def test_the_rlock_subset_the_lock_order_wrapper_needs(self):
        gate = TurnGate()
        assert gate.acquire(False) and gate.acquire(True, 0.01)
        gate.release()
        gate.release()
        with gate:
            assert gate.locked() and gate.current_ticket() is not None
        assert not gate.locked() and gate.current_ticket() is None


class TestWhoInheritsTheTicket:
    @pytest.mark.parametrize(
        ("kind", "re_enters"),
        [
            ("child_task", True),
            ("to_thread", True),
            ("copy_context_thread", True),
            ("sibling_task", False),
            ("plain_thread", False),
        ],
    )
    async def test_who_inherits_the_ticket(self, kind: str, re_enters: bool):
        if kind == "plain_thread" and THREADS_INHERIT_CONTEXT:
            pytest.skip("this build copies the Context into new threads")
        gate = TurnGate()
        ready, done = asyncio.Event(), asyncio.Event()
        out: dict[str, bool] = {}

        async def owner() -> None:
            assert gate.acquire(blocking=False)
            ctx = contextvars.copy_context()
            if kind == "child_task":

                async def child() -> bool:
                    return _probe(gate)

                out["got"] = await asyncio.create_task(child())
            elif kind == "to_thread":
                out["got"] = await asyncio.to_thread(_probe, gate)
            elif kind == "copy_context_thread":
                out["got"] = _in_thread(_probe, gate, ctx=ctx)
            elif kind == "plain_thread":
                out["got"] = _in_thread(_probe, gate)
            ready.set()
            await done.wait()
            gate.release()

        async def sibling() -> None:
            await ready.wait()
            if kind == "sibling_task":
                out["got"] = _probe(gate)
            done.set()

        await asyncio.gather(owner(), sibling())
        assert out["got"] is re_enters
        assert not gate.locked()


class TestBlockingAcquire:
    def test_a_blocking_acquire_waits_for_the_holder_and_wakes_at_release(self):
        gate = TurnGate()
        assert gate.acquire()
        entered = threading.Event()
        result: dict[str, bool] = {}

        def waiter() -> None:
            entered.set()
            result["got"] = gate.acquire(timeout=BOUND_SECONDS)
            gate.release()

        worker = threading.Thread(target=waiter, daemon=True)
        worker.start()
        assert entered.wait(BOUND_SECONDS)
        worker.join(0.1)  # negative evidence: it is still waiting
        assert worker.is_alive() and "got" not in result
        gate.release()
        worker.join(BOUND_SECONDS)
        assert not worker.is_alive() and result == {"got": True}

    def test_a_blocking_acquire_honours_its_timeout(self):
        gate = TurnGate()
        assert gate.acquire()
        assert _in_thread(gate.acquire, True, 0.05) is False
        assert _in_thread(gate.acquire, True, 0) is False
        gate.release()

    async def test_a_blocking_acquire_on_the_loop_thread_fails_fast(self):
        gate = TurnGate("c1")
        ready, done = asyncio.Event(), asyncio.Event()
        tickets: list[Any] = []

        async def holder() -> None:
            assert gate.acquire(blocking=False)
            tickets.append(gate.current_ticket())
            ready.set()
            await done.wait()
            if gate.locked():
                gate.release()

        async def reader() -> None:
            await ready.wait()
            with _Watchdog(gate, tickets):
                with pytest.raises(ConversationBusyError, match="c1") as refused:
                    gate.acquire()
                assert refused.value.conversation_id == "c1"
                with pytest.raises(ConversationBusyError):
                    gate.acquire(timeout=30)  # a timeout does not make it wait
            assert gate.locked()  # the refusal changed nothing
            done.set()

        await asyncio.gather(holder(), reader())
        assert not gate.locked()

    async def test_a_loop_thread_waits_for_a_thread_holder_and_times_out(self):
        gate = TurnGate()
        started, stop = threading.Event(), threading.Event()

        def hold() -> None:
            gate.acquire()
            started.set()
            stop.wait(BOUND_SECONDS)
            gate.release()

        worker = threading.Thread(target=hold, daemon=True)
        worker.start()
        assert started.wait(BOUND_SECONDS)
        assert (
            gate.acquire(timeout=0.05) is False
        )  # a thread will release: not a refusal
        stop.set()
        worker.join(BOUND_SECONDS)
        assert not worker.is_alive() and not gate.locked()


class TestRelease:
    def test_release_by_ticket_works_from_any_context_and_thread(self):
        gate = TurnGate()
        assert gate.acquire()
        ticket = gate.current_ticket()
        _in_thread(gate.release, ticket)  # a plain thread: an empty Context
        assert not gate.locked()
        assert gate.acquire()
        contextvars.Context().run(gate.release, gate.current_ticket())
        assert not gate.locked()

    def test_a_release_by_a_non_owner_raises_and_changes_nothing(self):
        gate, other = TurnGate(), TurnGate()
        with pytest.raises(RuntimeError):
            gate.release()
        assert gate.acquire()
        with pytest.raises(RuntimeError):
            _in_thread(gate.release)  # a thread that holds nothing
        assert other.acquire()
        foreign = other.current_ticket()
        other.release()
        with pytest.raises(RuntimeError):
            gate.release(foreign)
        assert gate.locked()
        gate.release()

    def test_released_tickets_do_not_pile_up_in_a_long_lived_context(self):
        gate = TurnGate()
        for _ in range(50):
            assert gate.acquire()
            contextvars.Context().run(gate.release, gate.current_ticket())
        assert gate.acquire()
        assert len(gate_module._HELD.get()) <= 1
        gate.release()


class TestAsyncGeneratorFinalizer:
    async def test_an_asyncgen_finalizer_releases_without_a_loop_error(self):
        """A stream abandoned without ``aclose``: its ``finally`` runs in the finalizer task,
        another Context. The release goes by ticket and a token reset would raise there."""
        gate = TurnGate()
        loop = asyncio.get_running_loop()
        errors: list[Any] = []
        loop.set_exception_handler(lambda _loop, context: errors.append(context))

        async def stream():
            assert gate.acquire(blocking=False)
            ticket = gate.current_ticket()
            try:
                yield 1
                yield 2
            finally:
                gate.release(ticket)

        agen = stream()
        async for _ in agen:
            break  # abandoned: never closed
        del agen
        gc.collect()
        for _ in range(100):
            if not gate.locked():
                break
            await asyncio.sleep(0)
        assert not gate.locked()
        assert errors == []
        assert gate.acquire(
            blocking=False
        )  # a stale binding grants nothing and blocks nothing
        gate.release()


_FSM = json.loads(
    '{"name": "GateFsm", "description": "d", "initial_state": "chat", "states": {'
    '"chat": {"id": "chat", "description": "c", "purpose": "p", "response_instructions": "r",'
    ' "transitions": [{"target_state": "done", "description": "never", "conditions":'
    ' [{"description": "n", "logic": {"==": [1, 2]}}]}]},'
    ' "done": {"id": "done", "description": "t", "purpose": "e", "response_instructions": "bye"}}}'
)


def _manager() -> tuple[FSMManager, str]:
    definition = FSMDefinition(**_FSM)
    manager = FSMManager(
        fsm_loader=lambda _id: definition,
        llm_interface=ScriptedLLM(default_response="hi"),
    )
    conversation_id, _ = manager.start_conversation("x")
    return manager, conversation_id


async def _while_a_task_holds(
    manager: FSMManager, cid: str, probe: Callable[[], None]
) -> None:
    """Hold a turn claim in one task, suspended, and run ``probe`` from a sibling task."""
    started, release = asyncio.Event(), asyncio.Event()

    async def turn() -> None:
        claim = _claim_turn(manager, cid)  # what a turn does first
        started.set()
        await release.wait()  # suspended mid-turn, the gate still held
        _release_turn(manager, cid, *claim)

    async def sibling() -> None:
        await started.wait()
        try:
            probe()
        finally:
            release.set()

    await asyncio.gather(turn(), sibling())


class TestManagerLevel:
    async def test_a_sibling_task_cannot_read_or_end_a_conversation_mid_turn(self):
        manager, cid = _manager()

        def probe() -> None:
            for call in (
                lambda: manager.get_conversation_state(cid),
                lambda: manager.end_conversation(cid),
            ):
                with pytest.raises(ConversationBusyError, match=cid):
                    call()
            assert cid in manager.instances  # nothing was ended

        await _while_a_task_holds(manager, cid, probe)
        assert not manager._conversation_locks[cid].locked()
        assert manager.get_conversation_state(cid) == "chat"  # the holder let go
        manager.end_conversation(cid)
        assert cid not in manager.instances

    async def test_a_sync_turn_from_the_loop_thread_is_refused_while_a_task_holds(self):
        manager, cid = _manager()

        def probe() -> None:
            with pytest.raises(FSMError, match="already being processed"):
                manager.process_message(cid, "hello")

        await _while_a_task_holds(manager, cid, probe)
        assert manager.process_message(cid, "hello") == "hi"  # free again

    def test_a_copy_context_handler_thread_reads_during_the_turn(self):
        manager, cid = _manager()
        seen: dict[str, Any] = {}

        def reads_through_the_manager() -> None:
            ctx = contextvars.copy_context()  # what the handler runner does
            seen["state"] = _in_thread(manager.get_conversation_state, cid, ctx=ctx)

        manager.handler_system.register_handler(
            create_handler("reader")
            .at(HandlerTiming.POST_PROCESSING)
            .do(lambda context: reads_through_the_manager() or {})
        )
        assert manager.process_message(cid, "hello") == "hi"
        assert seen == {"state": "chat"}
