"""The lock order of ``FSMManager`` (the three locks and the order they are taken in).

``FSMManager`` owns three kinds of lock: ``_lock`` (the dicts of instances and
per-conversation locks), one ``conv_lock`` per conversation (an ``RLock``) and
``_fsm_cache_lock`` (the definition cache). The rules, each anchored at the code that
keeps it (``engine/fsm/turns.py``, ``snapshots.py``, ``manager.py``):

- ``_lock`` is taken BEFORE a ``conv_lock``: every writer takes ``conv_lock`` while it
  holds ``_lock`` and only without waiting (``acquire(blocking=False)``), and a reader
  releases ``_lock`` before it blocks on ``conv_lock``. A thread never blocks on a
  ``conv_lock`` while it holds ``_lock``, and never takes ``_lock`` under a ``conv_lock``
  except in the one place that is bounded and documented: ``end_conversation`` removes the
  instance under its bounded (``timeout=``) hold of the lock;
- ``_fsm_cache_lock`` is a LEAF: while it is held no other lock is taken (the loader
  runs outside it);
- every lock is released when an operation returns, whether it succeeds or fails.

The manager's methods live in three modules (``TurnsMixin``, ``SnapshotsMixin`` and the
shell), so a change to one of them can break the order without touching the others.
This module records the real acquire and release calls of one thread through wrapper
locks and checks each operation of the manager against the rules; for the turn path,
the snapshot reads and the end path it also pins the exact order of ``_lock`` and
``conv_lock`` calls. The cache lock is left out of the pinned sequences because how
often a turn resolves the definition belongs to the pipeline, not to the manager.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator

import pytest

from fsm_llm.core.definitions import FSMDefinition, FSMError
from fsm_llm.engine.fsm.manager import FSMManager
from fsm_llm.testing import ScriptedLLM

_FSM = {
    "name": "LockOrder",
    "description": "Two states for the lock-order test",
    "initial_state": "chat",
    "states": {
        "chat": {
            "id": "chat",
            "description": "Chat",
            "purpose": "Answer",
            "response_instructions": "Reply briefly",
            "transitions": [
                {
                    "target_state": "done",
                    "description": "Never taken here",
                    "conditions": [{"description": "never", "logic": {"==": [1, 2]}}],
                }
            ],
        },
        "done": {
            "id": "done",
            "description": "Terminal",
            "purpose": "End",
            "response_instructions": "Say goodbye",
        },
    },
}


class _RecLock:
    """A lock wrapper that appends every acquire and release to a shared list.

    Interface contract: ``name`` is the role (``lock``, ``cache`` or ``conv``); ``events``
    is the shared list of ``(name, what, thread id)`` triples, ``what`` being ``acquire`` (blocking),
    ``acquire-nb`` (``blocking=False``), ``acquire-timeout``, ``acquire-failed`` (a
    non-blocking or timed acquire that did not get the lock) or ``release``. Everything
    else is delegated to the real lock, so the manager cannot tell the difference.
    """

    def __init__(self, name: str, inner, events: list[tuple[str, str, int]]):
        self._name = name
        self._inner = inner
        self._events = events

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        what = "acquire"
        if not blocking:
            what = "acquire-nb"
        elif timeout != -1:
            what = "acquire-timeout"
        got = self._inner.acquire(blocking, timeout)
        self._events.append(
            (self._name, what if got else "acquire-failed", threading.get_ident())
        )
        return got

    def release(self) -> None:
        self._events.append((self._name, "release", threading.get_ident()))
        self._inner.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    def locked(self) -> bool:
        return self._inner.locked()


def order_problems(events: list[tuple[str, str, int]]) -> list[str]:
    """Every rule the sequence breaks, as text; an empty list means it keeps them all.

    Held locks are tracked per thread (a suspended stream holds its ``conv_lock`` on the
    thread that opened it; another thread's call is not nested under it).
    """
    problems: list[str] = []
    held: dict[int, list[tuple[str, str]]] = {}  # tid -> (name, how it was acquired)
    for i, (name, what, tid) in enumerate(events):
        mine = held.setdefault(tid, [])
        if what == "acquire-failed":
            continue
        names = [n for n, _ in mine]
        if what == "release":
            if name not in names:
                problems.append(f"event {i}: {name} released but not held")
            else:
                mine.pop(len(names) - 1 - names[::-1].index(name))
            continue
        if "cache" in names:
            problems.append(
                f"event {i}: {name} taken while the leaf cache lock is held"
            )
        if name == "conv" and what != "acquire-nb" and "lock" in names:
            problems.append(f"event {i}: conv_lock awaited while _lock is held")
        if name == "lock" and any(
            n == "conv" and how != "acquire-timeout" for n, how in mine
        ):
            problems.append(f"event {i}: _lock taken while a conv_lock is held")
        mine.append((name, what))
    for mine in held.values():
        if mine:
            problems.append(f"still held at the end: {[n for n, _ in mine]}")
    return problems


def _short(events: list[tuple[str, str, int]], names=("lock", "conv")) -> list[str]:
    """The pinned form: ``lock+`` (acquire), ``conv+nb``, ``conv+t``, ``conv-`` (release)."""
    suffix = {
        "acquire": "+",
        "acquire-nb": "+nb",
        "acquire-timeout": "+t",
        "acquire-failed": "+failed",
        "release": "-",
    }
    return [f"{n}{suffix[w]}" for n, w, _ in events if n in names]


def _build() -> tuple[FSMManager, str, list[tuple[str, str, int]]]:
    llm = ScriptedLLM(responses={"chat": "ok", "done": "bye"}, stream_chunks=["o", "k"])
    manager = FSMManager(
        fsm_loader=lambda fsm_id: FSMDefinition(**_FSM), llm_interface=llm
    )
    conversation_id, _ = manager.start_conversation("fsm")
    events: list[tuple[str, str, int]] = []
    manager._lock = _RecLock("lock", manager._lock, events)  # type: ignore[assignment]
    manager._fsm_cache_lock = _RecLock("cache", manager._fsm_cache_lock, events)  # type: ignore[assignment]
    manager._conversation_locks[conversation_id] = _RecLock(  # type: ignore[assignment]
        "conv", manager._conversation_locks[conversation_id], events
    )
    return manager, conversation_id, events


def _drain(stream: Iterator[str]) -> list[str]:
    return list(stream)


# One entry per manager operation that takes a lock: name, the call, and whether it ends
# the conversation (so the next operation cannot reuse it).
_OPERATIONS: dict[str, Callable[[FSMManager, str], object]] = {
    "process_message": lambda m, c: m.process_message(c, "hi"),
    "advance": lambda m, c: m.advance(c),
    "process_message_stream": lambda m, c: _drain(m.process_message_stream(c, "hi")),
    "advance_stream": lambda m, c: _drain(m.advance_stream(c)),
    "has_conversation_ended": lambda m, c: m.has_conversation_ended(c),
    "get_conversation_data": lambda m, c: m.get_conversation_data(c),
    "get_conversation_state": lambda m, c: m.get_conversation_state(c),
    "get_conversation_history": lambda m, c: m.get_conversation_history(c),
    "get_end_snapshot": lambda m, c: m.get_end_snapshot(c),
    "get_conversation_snapshot": lambda m, c: m.get_conversation_snapshot(c),
    "get_complete_conversation": lambda m, c: m.get_complete_conversation(c),
    "set_conversation_state": lambda m, c: m.set_conversation_state(c, "done"),
    "seed_restored_conversation": lambda m, c: m.seed_restored_conversation(
        c, summary="s", history=[{"user": "u"}], provenance={}, working_memory=None
    ),
    "has_instance": lambda m, c: m.has_instance(c),
    "copy_raw_context_data": lambda m, c: m.copy_raw_context_data(c),
    "update_conversation_context": lambda m, c: m.update_conversation_context(
        c, {"k": 1}
    ),
    "prune_orphaned_locks": lambda m, c: m.prune_orphaned_locks(),
    "get_fsm_definition": lambda m, c: m.get_fsm_definition("fsm"),
    "end_conversation": lambda m, c: m.end_conversation(c),
}

# The exact order of ``_lock`` and ``conv_lock`` calls of the operations that matter for the
# order: ``+`` blocking acquire, ``+nb`` non-blocking, ``+t`` timed, ``-`` release. A change
# here is a change of the lock protocol: read the anchors in ``engine/fsm/`` first.
_PINNED: dict[str, list[str]] = {
    # the turn guard: _lock held while the per-conversation lock is taken without waiting
    "process_message": ["lock+", "conv+nb", "lock-", "conv-"],
    "advance": ["lock+", "conv+nb", "lock-", "conv-"],
    # the stream claims at its FIRST next(): an eager existence check under _lock, then the
    # same guard inside the nested closure
    "process_message_stream": ["lock+", "lock-", "lock+", "conv+nb", "lock-", "conv-"],
    "advance_stream": ["lock+", "lock-", "lock+", "conv+nb", "lock-", "conv-"],
    # a read: _lock for the lookup only, RELEASED before the blocking conv_lock
    "get_conversation_data": ["lock+", "lock-", "conv+", "conv-"],
    "get_conversation_snapshot": ["lock+", "lock-", "conv+", "conv-"],
    "get_complete_conversation": ["lock+", "lock-", "conv+", "conv-"],
    # the end path's read is bounded
    "get_end_snapshot": ["lock+", "lock-", "conv+t", "conv-"],
    # the validated setter resolves the definition AFTER _lock is released, before conv_lock
    "set_conversation_state": ["lock+", "lock-", "conv+", "conv-"],
    # a write: conv_lock taken without waiting while _lock is still held
    "update_conversation_context": ["lock+", "conv+nb", "lock-", "conv-"],
    # end: _lock for the lookup, the bounded conv_lock, then _lock again for the cleanup UNDER
    # that bounded hold (the one documented place where _lock is taken under a conv_lock: every
    # other conv_lock taker is non-blocking, so nobody waits on conv_lock while holding _lock)
    "end_conversation": ["lock+", "lock-", "conv+t", "lock+", "lock-", "conv-"],
}


class TestOperationsKeepTheOrder:
    @pytest.mark.parametrize("operation", sorted(_OPERATIONS))
    def test_the_operation_follows_the_lock_rules(self, operation: str):
        manager, conversation_id, events = _build()
        _OPERATIONS[operation](manager, conversation_id)
        assert order_problems(events) == []
        # not vacuous: the operation touched at least one of the three locks, unless it is
        # the cache read (cache only) or one of the two ``_lock``-only lookups
        assert events, operation

    @pytest.mark.parametrize("operation", sorted(_PINNED))
    def test_the_lock_and_conv_lock_calls_are_in_the_pinned_order(self, operation: str):
        manager, conversation_id, events = _build()
        _OPERATIONS[operation](manager, conversation_id)
        assert _short(events) == _PINNED[operation]

    def test_a_failed_turn_releases_everything(self):
        manager, conversation_id, events = _build()
        manager.llm_interface.responses.clear()  # type: ignore[attr-defined]
        with pytest.raises(FSMError):
            manager.process_message(conversation_id, "hi")
        assert order_problems(events) == []
        assert _short(events)[-1] == "conv-"

    def test_a_refused_second_turn_takes_no_lock_it_cannot_release(self):
        manager, conversation_id, events = _build()
        stream = manager.process_message_stream(conversation_id, "hi")
        next(stream)  # the stream now holds the turn guard and conv_lock (this thread)
        outcome: list[BaseException | None] = []

        def second_turn() -> None:
            try:
                manager.process_message(conversation_id, "again")
                outcome.append(None)
            except BaseException as e:
                outcome.append(e)

        thread = threading.Thread(target=second_turn, daemon=True)
        thread.start()
        thread.join(5)
        assert not thread.is_alive()
        assert len(outcome) == 1 and isinstance(outcome[0], FSMError)
        assert "already being processed" in str(outcome[0])
        stream.close()
        assert order_problems(events) == []

    def test_the_loader_runs_outside_the_cache_lock(self):
        # ``_fsm_cache_lock`` is a leaf: a custom loader may do anything, so it never runs
        # under it (a miss takes the lock to look, releases it, loads, takes it to insert).
        seen: list[bool] = []
        box: dict[str, FSMManager] = {}

        def loader(fsm_id: str) -> FSMDefinition:
            seen.append(box["manager"]._fsm_cache_lock.locked())
            if fsm_id == "broken":
                raise ValueError("no such definition")
            return FSMDefinition(**_FSM)

        manager = FSMManager(fsm_loader=loader, llm_interface=ScriptedLLM())
        box["manager"] = manager
        events: list[tuple[str, str, int]] = []
        manager._fsm_cache_lock = _RecLock("cache", manager._fsm_cache_lock, events)  # type: ignore[assignment]
        manager.get_fsm_definition("one")
        with pytest.raises(ValueError):
            manager.get_fsm_definition("broken")
        assert seen == [False, False]
        assert not manager._fsm_cache_lock.locked()
        assert order_problems(events) == []

    def test_an_unknown_conversation_takes_no_conv_lock(self):
        manager, _, events = _build()
        with pytest.raises(FSMError):
            manager.process_message("no-such-id", "hi")
        assert order_problems(events) == []
        assert not [e for e in events if e[0] == "conv"]

    def test_the_busy_conv_lock_refuses_without_waiting(self):
        manager, conversation_id, events = _build()
        lock = manager._conversation_locks[conversation_id]
        held, release = threading.Event(), threading.Event()

        def holder() -> None:
            with lock:
                held.set()
                release.wait(5)

        thread = threading.Thread(target=holder, daemon=True)
        thread.start()
        try:
            assert held.wait(5)
            with pytest.raises(FSMError, match="another thread"):
                manager.process_message(conversation_id, "hi")
        finally:
            release.set()
            thread.join(5)
        assert not thread.is_alive()
        failed = [(n, w) for n, w, _ in events if w == "acquire-failed"]
        assert failed == [("conv", "acquire-failed")]


class TestTheCheckerSeesViolations:
    """Guards the guard: each rule is broken in a synthetic sequence and in a real manager."""

    def test_lock_taken_under_a_conv_lock(self):
        events = [
            ("conv", "acquire", 1),
            ("lock", "acquire", 1),
            ("lock", "release", 1),
            ("conv", "release", 1),
        ]
        assert order_problems(events) == [
            "event 1: _lock taken while a conv_lock is held"
        ]

    def test_lock_under_the_bounded_conv_hold_is_the_documented_exception(self):
        events = [
            ("conv", "acquire-timeout", 1),
            ("lock", "acquire", 1),
            ("lock", "release", 1),
            ("conv", "release", 1),
        ]
        assert order_problems(events) == []

    def test_conv_lock_awaited_while_holding_lock(self):
        events = [
            ("lock", "acquire", 1),
            ("conv", "acquire", 1),
            ("conv", "release", 1),
            ("lock", "release", 1),
        ]
        assert order_problems(events) == [
            "event 1: conv_lock awaited while _lock is held"
        ]

    def test_anything_taken_under_the_leaf_cache_lock(self):
        events = [
            ("cache", "acquire", 1),
            ("lock", "acquire", 1),
            ("lock", "release", 1),
            ("cache", "release", 1),
        ]
        assert order_problems(events) == [
            "event 1: lock taken while the leaf cache lock is held"
        ]

    def test_a_lock_left_held(self):
        assert order_problems([("lock", "acquire", 1)]) == [
            "still held at the end: ['lock']"
        ]

    def test_another_threads_hold_is_not_nesting(self):
        events = [
            ("conv", "acquire-nb", 1),
            ("lock", "acquire", 2),
            ("lock", "release", 2),
            ("conv", "release", 1),
        ]
        assert order_problems(events) == []

    def test_the_right_order_is_clean(self):
        events = [
            ("lock", "acquire", 1),
            ("conv", "acquire-nb", 1),
            ("lock", "release", 1),
            ("conv", "release", 1),
        ]
        assert order_problems(events) == []

    def test_a_manager_that_inverts_the_order_is_caught(self):
        class _Inverted(FSMManager):
            def get_conversation_state(self, conversation_id: str) -> str:
                with self._conversation_locks[conversation_id]:
                    with self._lock:
                        return self.instances[conversation_id].current_state

        llm = ScriptedLLM(responses={"chat": "ok"})
        manager = _Inverted(
            fsm_loader=lambda fsm_id: FSMDefinition(**_FSM), llm_interface=llm
        )
        conversation_id, _ = manager.start_conversation("fsm")
        events: list[tuple[str, str, int]] = []
        manager._lock = _RecLock("lock", manager._lock, events)  # type: ignore[assignment]
        manager._conversation_locks[conversation_id] = _RecLock(  # type: ignore[assignment]
            "conv", manager._conversation_locks[conversation_id], events
        )
        manager.get_conversation_state(conversation_id)
        assert order_problems(events) == [
            "event 1: _lock taken while a conv_lock is held"
        ]


class _Recording(dict):
    """The manager's per-conversation lock table, wrapping every lock it holds or gets in a
    ``_RecLock`` (a pushed frame's lock is created at the push, so the table wraps on write)."""

    def __init__(self, items, events) -> None:
        self._events = events
        super().__init__()
        for key, lock in items.items():
            self[key] = lock

    def __setitem__(self, key, lock) -> None:
        if not isinstance(lock, _RecLock):
            lock = _RecLock("conv", lock, self._events)
        super().__setitem__(key, lock)


class TestSettleKeepsTheOrder:
    """``settle`` (``engine/api/calls.py``, plan baa37f40 D-023) runs after the turn released its
    conversation lock and reads through ``peek_turn``: a whole call flow (the request, the push,
    the child's turns, the pop with its merge) takes no manager lock under a conversation lock
    and awaits no conversation lock while ``_lock`` is held."""

    async def test_a_whole_call_flow_keeps_the_lock_rules(self):
        from tests.test_fsm_llm.test_call_settle import _flow

        flow = await _flow(steps=2)
        manager = flow.api.fsm_manager
        events: list[tuple[str, str, int]] = []
        manager._lock = _RecLock("lock", manager._lock, events)  # type: ignore[assignment]
        manager._fsm_cache_lock = _RecLock("cache", manager._fsm_cache_lock, events)  # type: ignore[assignment]
        manager._conversation_locks = _Recording(manager._conversation_locks, events)  # type: ignore[assignment]
        await flow.api.arun_until_terminal(flow.cid, max_steps=10)
        assert flow.record() is not None and flow.record()["status"] == "ok"
        assert order_problems(events) == []
        # not vacuous: both frames' conversation locks and `_lock` were taken
        assert {name for name, _, _ in events} >= {"lock", "conv"}
        assert sum(1 for n, w, _ in events if n == "conv" and w == "acquire-nb") >= 3
