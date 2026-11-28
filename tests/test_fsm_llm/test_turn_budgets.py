"""Core turn budgets (plan 5a6a1f80 step 7, roadmap row 1.7).

A lifetime budget per manager conversation: ``max_steps`` and ``max_seconds`` on the ``API`` (a
default for the root and every frame), ``API.start_conversation``, ``push_fsm`` and
``FSMManager.start_conversation``. The record is private to the manager, charged in
``_enter_turn`` after the claim and before any model call, and a spent budget raises the
existing ``RunBudgetExceededError`` on the NEXT turn with the claim undone. Default ``None``:
nothing changes. Each test names a mutant it kills in its docstring where that is not obvious.
"""

from __future__ import annotations

import copy
import threading
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import pytest

from fsm_llm import API, APIBuilder, FileSessionStore
from fsm_llm.core.definitions import BuildError, FSMDefinition, FSMError
from fsm_llm.core.definitions import RunBudgetExceededError as Spent
from fsm_llm.engine.fsm.manager import FSMManager
from fsm_llm.handlers import HandlerTiming
from fsm_llm.runtime import EventLog, EventType
from fsm_llm.testing import ScriptedLLM
from tests.test_fsm_llm.event_log_support import tree_bytes
from tests.test_fsm_llm.support import _state
from tests.test_fsm_llm.test_fsm_lock_order import (
    _FSM,
    _OPERATIONS,
    _PINNED,
    _RecLock,
    _short,
    order_problems,
)

_NEVER = [
    {
        "target_state": "end",
        "description": "never",
        "conditions": [{"description": "n", "logic": {"==": [1, 2]}}],
    }
]
_CHAT: dict[str, Any] = {
    "name": "Budget",
    "description": "One speaking state that never ends",
    "initial_state": "chat",
    "states": {
        "chat": _state("chat", response_instructions="Reply", transitions=_NEVER),
        "end": _state("end"),
    },
}
_ONCE: dict[str, Any] = {
    "name": "Once",
    "description": "The first turn ends the conversation",
    "initial_state": "chat",
    "states": {
        "chat": _state(
            "chat",
            response_instructions="Reply",
            transitions=[{"target_state": "done", "description": "go"}],
        ),
        "done": _state("done", response_instructions="Bye"),
    },
}
_KID: dict[str, Any] = {
    "name": "Kid",
    "description": "A pushed child that never ends",
    "initial_state": "hello",
    "states": {
        "hello": _state("hello", response_instructions="Kid", transitions=_NEVER),
        "end": _state("end"),
    },
}
_TURNS: dict[str, Callable[[API, str], object]] = {
    "converse": lambda api, cid: api.converse("hi", cid),
    "advance": lambda api, cid: api.advance(cid),
    "converse_stream": lambda api, cid: list(api.converse_stream("hi", cid)),
    "advance_stream": lambda api, cid: list(api.advance_stream(cid)),
}
_ENTRY = pytest.mark.parametrize("entry", list(_TURNS))


class _Failing(ScriptedLLM):
    """Answers until ``down`` is set, then every Pass-2 call raises."""

    down = False

    def generate_response(self, request: Any) -> Any:
        if self.down:
            raise RuntimeError("provider down")
        return super().generate_response(request)


class _Clock:
    """A settable stand-in for the budget clock."""

    now = 100.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    fake = _Clock()
    monkeypatch.setattr("fsm_llm.engine.fsm.budget._monotonic", fake)
    return fake


def _llm() -> ScriptedLLM:
    return ScriptedLLM(default_response="ok", stream_chunks=["o", "k"])


def _api(definition: dict[str, Any] | None = None, **kw: Any) -> tuple[API, str]:
    kw.setdefault("llm_interface", _llm())
    api = API(copy.deepcopy(definition or _CHAT), **kw)
    cid, _ = api.start_conversation()
    return api, cid


def _taken(api: API, cid: str) -> int:
    return api.fsm_manager._turn_budgets[cid].steps_taken


def _top(api: API, cid: str) -> str:
    return api.conversation_stacks[cid][-1].conversation_id


def _lock_is_free(manager: FSMManager, cid: str) -> bool:
    """True when ANOTHER thread takes the conversation lock at once (an RLock re-enters)."""
    got: list[bool] = []

    def probe() -> None:
        lock = manager._conversation_locks[cid]
        got.append(lock.acquire(blocking=False))
        if got[-1]:
            lock.release()

    thread = threading.Thread(target=probe)
    thread.start()
    thread.join(5)
    return got == [True]


class TestOff:
    def test_no_limit_means_no_record_and_every_turn_runs(self) -> None:
        api, cid = _api()
        for turn in _TURNS.values():
            for _ in range(3):
                turn(api, cid)
        assert api.fsm_manager._turn_budgets == {}

    def test_the_manager_is_called_as_before_when_nothing_is_set(self) -> None:
        api = API(copy.deepcopy(_CHAT), llm_interface=_llm())
        with patch.object(
            api.fsm_manager,
            "start_conversation",
            wraps=api.fsm_manager.start_conversation,
        ) as start:
            api.start_conversation()
        assert set(start.call_args.kwargs) == {"initial_context", "suppress_start"}


class TestCounting:
    @_ENTRY
    def test_turn_n_plus_one_raises_with_the_ids_of_the_budget(self, entry: str):
        api, cid = _api(max_steps=3)
        for _ in range(3):
            _TURNS[entry](api, cid)
        with pytest.raises(Spent) as refused:
            _TURNS[entry](api, cid)
        error = refused.value
        assert (error.budget, error.limit, error.steps_done) == ("steps", 3, 3)
        assert isinstance(error, FSMError) and "steps budget of 3" in str(error)

    def test_the_four_entries_share_one_count(self) -> None:
        api, cid = _api(max_steps=4)
        for turn in _TURNS.values():
            turn(api, cid)
        assert _taken(api, cid) == 4
        for turn in _TURNS.values():
            with pytest.raises(Spent):
                turn(api, cid)

    def test_the_greeting_is_not_a_turn(self) -> None:
        api, cid = _api(max_steps=1)
        assert _taken(api, cid) == 0
        api.converse("one", cid)
        with pytest.raises(Spent):
            api.converse("two", cid)

    def test_a_stream_counts_at_its_first_next_never_at_the_call(self) -> None:
        """Mutant: the stream is not counted, or is counted when it is opened."""
        api, cid = _api(max_steps=1)
        stream = api.converse_stream("a", cid)
        assert _taken(api, cid) == 0
        api.converse("b", cid)
        with pytest.raises(Spent):
            list(stream)

    def test_a_rolled_back_turn_still_counts(self) -> None:
        """Mutant: the counter is refunded on rollback (kept in the snapshot)."""
        llm = _Failing(default_response="ok")
        api, cid = _api(max_steps=2, llm_interface=llm)
        greeting = api.get_conversation_history(cid)
        llm.down = True
        with pytest.raises(FSMError) as failed:
            api.converse("lost", cid)
        assert not isinstance(failed.value, Spent) and _taken(api, cid) == 1
        assert api.get_conversation_history(cid) == greeting
        llm.down = False
        api.converse("kept", cid)
        with pytest.raises(Spent):
            api.converse("third", cid)

    def test_a_turn_refused_as_terminal_still_counts(self) -> None:
        api, cid = _api(_ONCE, max_steps=2)
        api.converse("go", cid)
        with pytest.raises(FSMError, match="ended") as ended:
            api.converse("more", cid)
        assert not isinstance(ended.value, Spent)
        with pytest.raises(Spent):
            api.converse("again", cid)

    def test_the_public_reader_gives_the_charged_count_and_none_without_a_limit(
        self,
    ) -> None:
        """The runner traces the engine's count instead of keeping its own: the reader is
        the record's ``steps_taken``, ``None`` for no limit and for an unknown id."""
        api, cid = _api(max_steps=3)
        manager = api.fsm_manager
        assert manager.turn_budget_steps(cid) == 0
        api.converse("one", cid)
        api.advance(cid)
        assert manager.turn_budget_steps(cid) == _taken(api, cid) == 2
        assert manager.turn_budget_steps("nobody") is None
        free, free_cid = _api()
        free.converse("one", free_cid)
        assert free.fsm_manager.turn_budget_steps(free_cid) is None

    def test_a_turn_the_guard_refuses_is_not_charged(self) -> None:
        api, cid = _api(max_steps=1)
        manager = api.fsm_manager
        manager._active_turns.add(cid)
        with pytest.raises(FSMError, match="already being processed"):
            manager.process_message(cid, "busy")
        manager._active_turns.discard(cid)
        assert _taken(api, cid) == 0


class TestRefusal:
    @_ENTRY
    def test_a_refusal_leaves_nothing_claimed(self, entry: str) -> None:
        """Mutant: no undo of the claim on refusal (a leaked guard or a held lock)."""
        api, cid = _api(max_steps=1)
        _TURNS[entry](api, cid)
        for _ in range(
            2
        ):  # a leaked guard would turn the second into "already processed"
            with pytest.raises(Spent):
                _TURNS[entry](api, cid)
        manager = api.fsm_manager
        assert manager._active_turns == set() and _lock_is_free(manager, cid)
        api.end_conversation(cid)
        assert manager._turn_budgets == {}

    @_ENTRY
    def test_a_refusal_calls_no_model_and_changes_nothing(self, entry: str) -> None:
        """Mutant: the charge runs after the model call."""
        llm = _llm()
        api, cid = _api(max_steps=1, llm_interface=llm)
        _TURNS[entry](api, cid)
        calls = len(llm.call_history)
        before = (api.get_conversation_history(cid), api.get_data(cid))
        with pytest.raises(Spent):
            _TURNS[entry](api, cid)
        assert len(llm.call_history) == calls
        assert (api.get_conversation_history(cid), api.get_data(cid)) == before
        assert _taken(api, cid) == 1  # a refusal never counts itself


class TestFrames:
    def test_a_pushed_frame_has_its_own_record(self) -> None:
        """Mutant: one record shared by the parent and the child."""
        api, cid = _api(max_steps=3)
        api.converse("p1", cid)
        api.converse("p2", cid)
        api.push_fsm(cid, copy.deepcopy(_KID))
        kid = _top(api, cid)
        records = api.fsm_manager._turn_budgets
        assert records[kid] is not records[cid]
        for _ in range(3):
            api.converse("k", cid)
        with pytest.raises(Spent):
            api.converse("k4", cid)
        api.pop_fsm(cid)
        assert kid not in records and records[cid].steps_taken == 2
        api.converse("p3", cid)
        with pytest.raises(Spent):
            api.converse("p4", cid)

    def test_a_push_argument_wins_over_the_api_default(self) -> None:
        api, cid = _api()
        api.push_fsm(cid, copy.deepcopy(_KID), max_steps=1)
        api.converse("k", cid)
        with pytest.raises(Spent) as refused:
            api.converse("k2", cid)
        assert refused.value.limit == 1
        api.pop_fsm(cid)
        for _ in range(5):  # the root has no limit
            api.converse("p", cid)

    def test_a_start_argument_wins_over_the_api_default(self) -> None:
        api = API(copy.deepcopy(_CHAT), llm_interface=_llm(), max_steps=5)
        cid, _ = api.start_conversation(max_steps=1)
        api.converse("a", cid)
        with pytest.raises(Spent):
            api.converse("b", cid)

    def test_the_record_is_dropped_with_the_conversation(self) -> None:
        """Mutant: the record is not dropped at end_conversation."""
        api, cid = _api(max_steps=2)
        api.push_fsm(cid, copy.deepcopy(_KID))
        manager = api.fsm_manager
        assert len(manager._turn_budgets) == 2
        api.end_conversation(cid)
        assert manager._turn_budgets == {} and manager.instances == {}

    def test_a_failed_start_leaves_no_record(self) -> None:
        api = API(
            copy.deepcopy(_CHAT),
            llm_interface=_llm(),
            max_steps=2,
            handler_error_mode="raise",
        )
        api.create_handler(
            "boom",
            timing=HandlerTiming.START_CONVERSATION,
            action=lambda ctx: 1 / 0,
        )
        with pytest.raises(FSMError):
            api.start_conversation()
        assert api.fsm_manager._turn_budgets == {}


class TestNoForgery:
    def test_a_handler_and_a_driver_write_cannot_move_the_count(self) -> None:
        api, cid = _api(max_steps=2)
        forged = {"steps_taken": 0, "max_steps": 99, "turn_budget": None, "turns": 0}
        api.create_handler(
            "forge",
            timing=HandlerTiming.POST_PROCESSING,
            action=lambda ctx: dict(forged),
        )
        for _ in range(2):
            api.converse("hi", cid)
            api.update_context(cid, forged)
        assert _taken(api, cid) == 2
        with pytest.raises(Spent):
            api.converse("hi", cid)

    def test_context_metadata_and_snapshot_carry_no_budget(self, tmp_path) -> None:
        """Mutant: the counter is kept in the context. Compared against an unbudgeted twin."""
        shapes = []
        for options in ({}, {"max_steps": 5, "max_seconds": 1000.0}):
            store = FileSessionStore(tmp_path / str(len(shapes)))
            api, cid = _api(session_store=store, **options)
            api.converse("hi", cid)
            api.advance(cid)
            api.save_session(cid)
            snapshot = api.fsm_manager.get_conversation_snapshot(cid)
            instance = api.fsm_manager.instances[cid]
            shapes.append(
                (
                    sorted(instance.context.data),
                    sorted(instance.context.metadata),
                    sorted(snapshot),
                    sorted(snapshot["context_data"]),
                )
            )
            saved = tree_bytes(tmp_path / str(len(shapes) - 1))
            for word in (b"budget", b"steps_taken", b"max_steps", b"max_seconds"):
                assert word not in saved
        assert shapes[0] == shapes[1]


class TestSeconds:
    def test_a_spent_clock_refuses_the_next_turn(self, clock: _Clock) -> None:
        api, cid = _api(max_seconds=10.0)
        clock.now = 109.0
        api.converse("a", cid)
        clock.now = 110.0
        with pytest.raises(Spent) as refused:
            api.converse("b", cid)
        error = refused.value
        assert (error.budget, error.limit, error.steps_done) == ("seconds", 10.0, 1)
        assert api.fsm_manager.turn_budget_spent(cid) == "seconds"
        assert api.fsm_manager.turn_budget_spent("nobody") is None
        assert api.fsm_manager._active_turns == set()

    def test_a_stalled_or_backwards_clock_never_spends(self, clock: _Clock) -> None:
        api, cid = _api(max_seconds=10.0)
        for _ in range(20):  # stalled
            api.converse("a", cid)
        clock.now = 50.0  # backwards
        api.converse("b", cid)
        assert api.fsm_manager.turn_budget_spent(cid) is None

    def test_a_turn_in_flight_is_never_cut_short(self, clock: _Clock) -> None:
        class _Slow(ScriptedLLM):
            slow = False

            def generate_response(self, request: Any) -> Any:
                if self.slow:
                    clock.now = 500.0
                return super().generate_response(request)

        slow = _Slow(default_response="ok")
        api, cid = _api(max_seconds=10.0, llm_interface=slow)
        slow.slow = True
        assert api.converse("a", cid) == "ok"
        with pytest.raises(Spent):
            api.converse("b", cid)

    def test_both_spent_reports_the_seconds(self, clock: _Clock) -> None:
        api, cid = _api(max_steps=1, max_seconds=5.0)
        api.converse("a", cid)
        assert api.fsm_manager.turn_budget_spent(cid) == "steps"
        clock.now = 200.0
        with pytest.raises(Spent) as refused:
            api.converse("b", cid)
        assert refused.value.budget == "seconds"


_BAD_STEPS = [0, -1, True, 1.5, "3"]
_BAD_SECONDS = [0, -1.0, True, float("nan"), "5"]


class TestSurface:
    @pytest.mark.parametrize(
        "kw",
        [{"max_steps": v} for v in _BAD_STEPS]
        + [{"max_seconds": v} for v in _BAD_SECONDS],
    )
    def test_a_bad_limit_is_refused_before_anything_is_registered(
        self, kw: dict[str, Any]
    ) -> None:
        with pytest.raises(ValueError, match="max_s"):
            API(copy.deepcopy(_CHAT), llm_interface=_llm(), **kw)
        api, cid = _api()
        manager = api.fsm_manager
        with pytest.raises(ValueError, match="max_s"):
            api.start_conversation(**kw)
        with pytest.raises(ValueError, match="max_s"):
            api.push_fsm(cid, copy.deepcopy(_KID), **kw)
        with pytest.raises(ValueError, match="max_s"):
            manager.start_conversation(api.fsm_id, **kw)
        assert list(manager.instances) == [cid] and api.get_stack_depth(cid) == 1
        assert api._temp_fsm_definitions == {} and api._pending_push_ids == {}

    def test_the_builder_sets_the_defaults_and_build_refuses_a_bad_one(self) -> None:
        builder = APIBuilder().set_definition(_CHAT).set_llm_interface(_llm())
        api = builder.set_max_steps(2).set_max_seconds(30.0).build()
        cid, _ = api.start_conversation()
        api.converse("a", cid)
        api.converse("b", cid)
        with pytest.raises(Spent):
            api.converse("c", cid)
        assert api.fsm_manager._turn_budgets[cid].max_seconds == 30.0
        for bad in (
            builder.set_max_steps(0),
            builder.set_max_steps(2).set_max_seconds(0),
        ):
            with pytest.raises(BuildError):
                bad.build()
        builder.set_max_steps(None).set_max_seconds(None).build()


class TestEventLog:
    def test_one_decision_per_refusal_and_no_user_text(self, log: EventLog) -> None:
        api, cid = _api(event_log=log, max_steps=2)
        api.converse("first", cid)
        api.advance(cid)
        for _ in range(2):
            with pytest.raises(Spent):
                api.converse("SECRET-REFUSED-TEXT", cid)
        events = list(log.read(cid))
        decisions = [e for e in events if e["type"] == "decision"]
        assert len(decisions) == 2
        for event in decisions:
            payload = {k: v for k, v in event.items() if k not in ("v", "seq", "ts")}
            assert payload == {
                "conversation_id": cid,
                "frame": 0,
                "type": "decision",
                "kind": "budget_spent",
                "budget": "steps",
                "limit": 2,
                "steps_done": 2,
            }
        starts = [e["type"] for e in events].count("turn/start")
        assert starts == 2  # none for a refusal
        assert b"SECRET-REFUSED-TEXT" not in tree_bytes(log.root)
        last_end = max(e["seq"] for e in events if e["type"] == "turn/end")
        assert decisions[0]["seq"] > last_end

    def test_the_decision_carries_the_frame_of_a_refused_child(
        self, log: EventLog
    ) -> None:
        api, cid = _api(event_log=log, max_steps=1)
        api.push_fsm(cid, copy.deepcopy(_KID))
        api.converse("k", cid)
        with pytest.raises(Spent):
            api.converse("k2", cid)
        (event,) = [e for e in log.read(cid) if e["type"] == "decision"]
        assert event["frame"] == 1

    def test_the_log_off_writes_nothing(self) -> None:
        api, cid = _api(max_steps=1)
        api.converse("a", cid)
        with patch.object(EventLog, "append") as append, pytest.raises(Spent):
            api.converse("b", cid)
        assert not append.called

    def test_a_log_that_cannot_write_the_decision_still_raises_the_budget(
        self, log: EventLog
    ) -> None:
        api, cid = _api(event_log=log, max_steps=1)
        api.converse("a", cid)
        real = log.append

        def refuse(conversation_id: str, kind: Any, **fields: Any) -> int:
            if EventType.from_string(kind) is EventType.DECISION:
                raise OSError("disk full")
            return real(conversation_id, kind, **fields)

        with patch.object(log, "append", refuse), pytest.raises(Spent):
            api.converse("b", cid)
        assert api.fsm_manager._active_turns == set()

    def test_fork_and_resume_start_a_fresh_budget_from_the_api_default(
        self, log: EventLog
    ) -> None:
        api, cid = _api(event_log=log, max_steps=2)
        api.converse("p", cid)
        api.push_fsm(cid, copy.deepcopy(_KID))
        api.converse("k1", cid)
        api.converse("k2", cid)
        with pytest.raises(Spent):
            api.converse("k3", cid)
        forked, state = api.fork(cid, at_turn=3)
        assert state.stack_depth == 2
        records = api.fsm_manager._turn_budgets
        assert {
            records[forked].steps_taken,
            records[_top(api, forked)].steps_taken,
        } == {0}
        api.converse("f1", forked)
        api.converse("f2", forked)
        with pytest.raises(Spent):
            api.converse("f3", forked)
        log.close()
        with EventLog(log.root) as again:
            second = API(
                copy.deepcopy(_CHAT), llm_interface=_llm(), event_log=again, max_steps=2
            )
            second.resume(cid)  # the log holds a decision event: the fold ignores it
            assert _taken(second, _top(second, cid)) == 0
            second.converse("r1", cid)
            second.converse("r2", cid)
            with pytest.raises(Spent):
                second.converse("r3", cid)

    def test_a_restored_session_starts_a_fresh_budget(self, tmp_path) -> None:
        store = FileSessionStore(tmp_path)
        api, cid = _api(session_store=store, max_steps=1)
        api.converse("a", cid)
        with pytest.raises(Spent):
            api.converse("b", cid)
        restored = api.restore_session(cid)
        assert restored is not None
        new_id = restored[0]
        api.converse("c", new_id)
        with pytest.raises(Spent):
            api.converse("d", new_id)


class TestLockSequence:
    @staticmethod
    def _manager(**budget: Any) -> tuple[FSMManager, str, list[tuple[str, str, int]]]:
        llm = ScriptedLLM(
            responses={"chat": "ok", "done": "bye"}, stream_chunks=["o", "k"]
        )
        manager = FSMManager(
            fsm_loader=lambda fsm_id: FSMDefinition(**_FSM), llm_interface=llm
        )
        cid, _ = manager.start_conversation("fsm", **budget)
        events: list[tuple[str, str, int]] = []
        manager._lock = _RecLock("lock", manager._lock, events)  # type: ignore[assignment]
        manager._conversation_locks[cid] = _RecLock(  # type: ignore[assignment]
            "conv", manager._conversation_locks[cid], events
        )
        return manager, cid, events

    @pytest.mark.parametrize(
        "operation",
        ["process_message", "advance", "process_message_stream", "advance_stream"],
    )
    def test_a_budget_leaves_the_pinned_lock_sequence_alone(
        self, operation: str
    ) -> None:
        manager, cid, events = self._manager(max_steps=5, max_seconds=1000.0)
        _OPERATIONS[operation](manager, cid)
        assert _short(events) == _PINNED[operation]
        assert order_problems(events) == []

    @pytest.mark.parametrize("operation", ["process_message", "advance_stream"])
    def test_a_refusal_undoes_its_claim_under_the_lock_and_takes_no_other(
        self, operation: str
    ) -> None:
        manager, cid, events = self._manager(max_steps=1)
        _OPERATIONS[operation](manager, cid)
        events.clear()
        with pytest.raises(Spent):
            _OPERATIONS[operation](manager, cid)
        want = {
            "process_message": ["lock+", "conv+nb", "conv-", "lock-"],
            "advance_stream": ["lock+", "lock-", "lock+", "conv+nb", "conv-", "lock-"],
        }[operation]
        assert _short(events) == want
        assert order_problems(events) == []


class TestRuns:
    def test_a_per_call_run_budget_does_not_reset_the_lifetime_one(self) -> None:
        """The reason the synthesis runner kept its own counter: a call's counters restart."""
        llm = _llm()
        api, cid = _api(max_steps=3, llm_interface=llm)
        with pytest.raises(Spent) as refused:
            api.run_until_terminal(cid, max_steps=10)
        assert (refused.value.limit, refused.value.steps_done) == (3, 3)
        calls = len(llm.call_history)
        with pytest.raises(Spent) as again:
            api.run_until_terminal(cid, max_steps=2)
        assert again.value.steps_done == 3 and len(llm.call_history) == calls
        assert api.fsm_manager._active_turns == set()
