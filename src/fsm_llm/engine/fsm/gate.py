"""``TurnGate``: the per-conversation lock, owned by a ticket and not by a thread.

Layer ``engine`` (plan f150ec28, step 1.2, roadmap row 1.5 part B). The ``threading.RLock`` it
replaces is owned by a THREAD, so under asyncio every task of one loop looks like the same owner
and a reader or ``end_conversation`` of a sibling task passes straight through a suspended turn.
Here the owner is a ticket that rides in a ``ContextVar`` tuple: a thread, a child task, an
``asyncio.to_thread`` worker and a ``copy_context().run`` thread (a handler or a tool thread)
inherit it and re-enter; a sibling task and a plain ``Thread`` (empty Context) do not.

Interface contract:
    - ``TurnGate(conversation_id=None)``: the id only names the conversation in a refusal.
    - ``acquire(blocking=True, timeout=-1) -> bool``: the owner re-enters (a depth count) and gets
      True. A second owner with ``blocking=False`` gets False at once and never waits. A blocking
      second owner on a plain thread waits for the holder to release, bounded by ``timeout`` when
      it is not negative (False on expiry). A blocking second owner on the thread of an event loop
      whose holder is a task of that loop raises ``ConversationBusyError`` at once, whatever the
      timeout: the wait could never end, the loop is what would release it. There is no async
      acquire: a turn that finds the gate held is refused by its caller. Waiting threads are
      served in arrival order and the LAST release hands the gate to the first of them directly
      (it is never free in between), so the thread that just released cannot take it back
      before a waiter ran: a lock released and retaken in a tight loop must not starve a reader
      (the ``threading.RLock`` this replaces handed over at C level, without the GIL).
    - ``release(ticket=None) -> None``: one hold. With no ticket it is the hold of the current
      Context. With the ticket ``current_ticket()`` returned after the ``acquire`` it works from
      any Context and any thread, so a stream's ``finally`` may run in an asyncgen finalizer.
      Failure: ``RuntimeError`` when the Context (or the ticket) is not the owner.
    - ``locked() -> bool``; ``current_ticket() -> GateTicket | None`` (this Context's owner
      ticket); ``with gate:`` is a blocking ``acquire`` then ``release``.
    - Nothing here logs, imports a conversation or calls out while the mutex is held. The binding
      is only ever ``set`` to a new tuple, never reset with a token (a token raises ``ValueError``
      in another Context), and a released ticket drops out of the tuple at the next ``acquire``.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from contextvars import ContextVar

from fsm_llm.core.definitions import ConversationBusyError

__all__ = ["GateTicket", "TurnGate"]


class GateTicket:
    """One owner of a gate: its number of holds and the loop thread it was taken on, if any."""

    __slots__ = ("depth", "loop_thread")

    def __init__(self, loop_thread: int | None) -> None:
        self.depth = 1
        self.loop_thread = loop_thread


class _Waiter:
    """A blocked acquirer: the ticket ``release`` handed it (``None`` until then)."""

    __slots__ = ("loop_thread", "ticket")

    def __init__(self) -> None:
        self.loop_thread = _running_loop_thread()
        self.ticket: GateTicket | None = None


#: The tickets this Context holds, newest last (only ever ``set`` to a new tuple).
_HELD: ContextVar[tuple[GateTicket, ...]] = ContextVar(
    "fsm_llm_gate_tickets", default=()
)


def _running_loop_thread() -> int | None:
    """The ident of this thread when an event loop is running on it, else ``None``."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return None
    return threading.get_ident()


class TurnGate:
    """A re-entrant lock owned by a ticket (the module docstring is the contract)."""

    def __init__(self, conversation_id: str | None = None) -> None:
        self._conversation_id = conversation_id  # only for the refusal message
        self._cond = threading.Condition(threading.Lock())  # a leaf mutex
        self._owner: GateTicket | None = None
        self._queue: deque[_Waiter] = deque()  # blocked acquirers, oldest first

    def _mine(self) -> GateTicket | None:
        owner = self._owner
        return owner if owner is not None and owner in _HELD.get() else None

    def _bind(self, ticket: GateTicket) -> None:
        live = tuple(t for t in _HELD.get() if t.depth > 0)
        _HELD.set((*live, ticket))

    def _hand_over(self) -> None:
        """The gate has no holder: give it to the oldest waiter, else leave it free."""
        # DECISION plan-2026-11-19T193704-f150ec28/D-015
        # The LAST release hands the gate to the oldest waiter DIRECTLY: it makes that
        # waiter's ticket the owner, so the gate is never free in between. Do NOT write
        # this as `self._owner = None; self._cond.notify_all()` (the releaser, a loop of
        # turns holding the GIL, retakes the gate before the woken waiter runs, and a
        # blocked reader starves: 98 reads against about 3,000 in the seam test), do NOT
        # replace the queue by a reserved flag (a waiter that timed out leaves the gate
        # reserved for nobody), and do NOT let the releaser's own non-blocking claim
        # succeed while a waiter is queued. See decisions.md D-015.
        if not self._queue:
            self._owner = None
            return
        waiter = self._queue.popleft()
        self._owner = waiter.ticket = GateTicket(waiter.loop_thread)
        self._cond.notify_all()

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        deadline = None if timeout < 0 else time.monotonic() + timeout
        with self._cond:
            mine = self._mine()
            if mine is not None:
                mine.depth += 1
                return True
            if self._owner is None:
                self._owner = ticket = GateTicket(_running_loop_thread())
                self._bind(ticket)
                return True
            if not blocking:
                return False
            if self._owner.loop_thread == threading.get_ident():
                raise ConversationBusyError(
                    f"Conversation {self._conversation_id} is held by a turn of this "
                    "event loop, so a blocking call on the loop thread would never "
                    "return: call it from another thread or after the turn ends",
                    conversation_id=self._conversation_id,
                )
            waiter = _Waiter()
            self._queue.append(waiter)
            try:
                while waiter.ticket is None:
                    remaining = (
                        None if deadline is None else deadline - time.monotonic()
                    )
                    if remaining is not None and remaining <= 0:
                        break
                    self._cond.wait(remaining)
            except BaseException:
                self._abandon(waiter)
                raise
            if waiter.ticket is None:
                self._abandon(waiter)
                return False
            self._bind(waiter.ticket)
            return True

    def _abandon(self, waiter: _Waiter) -> None:
        """A waiter stops waiting (timeout, interrupt): leave the queue, or, when a release
        already handed it the gate, hand the gate on."""
        if waiter.ticket is None:
            self._queue.remove(waiter)
        else:
            self._hand_over()

    def release(self, ticket: GateTicket | None = None) -> None:
        with self._cond:
            held = self._mine() if ticket is None else ticket
            if held is None or held is not self._owner:
                raise RuntimeError("cannot release an un-owned TurnGate")
            held.depth -= 1
            if held.depth > 0:
                return
            self._hand_over()
        _HELD.set(tuple(t for t in _HELD.get() if t is not held))

    def locked(self) -> bool:
        return self._owner is not None

    def current_ticket(self) -> GateTicket | None:
        return self._mine()

    def __enter__(self) -> TurnGate:
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()
