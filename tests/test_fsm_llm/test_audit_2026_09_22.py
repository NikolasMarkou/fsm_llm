"""Regression tests for the 2026-09-22 fsm_llm core audit.

One class per behaviour step of plan-2026-09-22T080837-8b258a25 (steps 2-14).
Each test was run RED against the parent commit of its step before the fix
landed, and names the audit id it pins (``test_p0_1_*``, ``test_p1_4_*``, ...).
"""

from __future__ import annotations

import threading
import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from fsm_llm.api import API
from fsm_llm.definitions import (
    ConversationBusyError,
    FieldExtractionResponse,
    FSMDefinition,
    FSMError,
    State,
    Transition,
    TransitionCondition,
)
from fsm_llm.llm import LLMInterface

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_JOIN_TIMEOUT_S = 10.0


def _mock_llm() -> MagicMock:
    """A spec'd mock LLM: field extraction returns nothing, Pass 2 says "ok"."""
    llm = MagicMock(spec=LLMInterface)
    llm.model = "gpt-4"

    def _extract_field(request):
        return FieldExtractionResponse(
            field_name=request.field_name,
            value=None,
            confidence=0.0,
            reasoning="mock",
            is_valid=False,
        )

    llm.extract_field.side_effect = _extract_field
    llm.generate_response.return_value = MagicMock(
        message="ok", message_type="response", reasoning="mock"
    )
    llm.generate_response_stream.side_effect = lambda request: iter(["o", "k"])
    return llm


def _blocked_fsm() -> FSMDefinition:
    """``start`` waits on a key nobody sets, so every turn is BLOCKED."""
    never = TransitionCondition(
        description="never set",
        logic={"==": [{"var": "never_set_flag"}, True]},
    )
    states = {
        "start": State(
            id="start",
            description="Start",
            purpose="Wait",
            response_instructions="Respond",
            transitions=[
                Transition(target_state="end", description="Done", conditions=[never])
            ],
        ),
        "end": State(id="end", description="End", purpose="End", transitions=[]),
    }
    return FSMDefinition(
        name="audit_2026_09_22",
        description="Blocked FSM for audit regression tests",
        initial_state="start",
        states=states,
    )


def _api() -> API:
    return API.from_definition(_blocked_fsm(), llm_interface=_mock_llm())


def _join_all(threads: list[threading.Thread]) -> None:
    for t in threads:
        t.join(timeout=_JOIN_TIMEOUT_S)
    assert not any(t.is_alive() for t in threads), "a thread hung"


# ---------------------------------------------------------------------------
# Step 2: ended-conversation cache concurrency and getter consistency
# ---------------------------------------------------------------------------


class _ParkOnWrite(dict):
    """Parks the cache write for ``key`` until ``release`` is set."""

    def __init__(self, key: str) -> None:
        super().__init__()
        self.key = key
        self.entered = threading.Event()
        self.release = threading.Event()

    def __setitem__(self, k, v) -> None:
        if k == self.key:
            self.entered.set()
            self.release.wait(timeout=_JOIN_TIMEOUT_S)
        super().__setitem__(k, v)


class _MeetOnEvict(dict):
    """The first two evictions meet on a barrier before popping, so two
    unsynchronised ends pick the same oldest key."""

    def __init__(self) -> None:
        super().__init__()
        self.barrier = threading.Barrier(2, timeout=0.5)

    def pop(self, *args):
        try:
            self.barrier.wait()
        except threading.BrokenBarrierError:
            pass
        return super().pop(*args)


class TestStep2EndedCacheRace:
    """P0-1: ``_ended_conversations`` is written, evicted and read only under
    ``API._stack_lock`` (plan-2026-09-22T080837-8b258a25/D-003)."""

    def test_p0_1_reader_never_sees_the_gap_between_drop_and_cache_write(self):
        api = _api()
        conv_id, _ = api.start_conversation(initial_context={"note": "n"})
        cache = _ParkOnWrite(conv_id)
        api._ended_conversations = cache

        errors: list[BaseException] = []
        results: dict[str, Any] = {}

        def _end() -> None:
            try:
                api.end_conversation(conv_id)
            except BaseException as e:  # pragma: no cover - failure path
                errors.append(e)

        def _read(name: str) -> None:
            try:
                results[name] = getattr(api, name)(conv_id)
            except BaseException as e:
                results[name] = e

        ender = threading.Thread(target=_end)
        ender.start()
        assert cache.entered.wait(timeout=_JOIN_TIMEOUT_S)
        # The end is parked inside the cache write. Readers started now either
        # wait for it (fixed) or run in the gap and see neither the live
        # conversation nor its cache entry (the P0-1 race).
        getters = (
            "get_data",
            "has_conversation_ended",
            "get_current_state",
            "get_conversation_history",
        )
        readers = [threading.Thread(target=_read, args=(g,)) for g in getters]
        for r in readers:
            r.start()
        for r in readers:
            r.join(timeout=0.5)
        cache.release.set()
        _join_all([ender, *readers])

        assert errors == []
        assert results["get_data"] == {"note": "n"}
        assert results["has_conversation_ended"] is True
        assert results["get_current_state"] == "start"
        assert isinstance(results["get_conversation_history"], list)

    def test_p0_1_concurrent_evictions_never_pop_the_same_key(self):
        api = _api()
        ids = [api.start_conversation()[0] for _ in range(2)]
        cache = _MeetOnEvict()
        dict.__setitem__(cache, "oldest", {"data": {}, "state": "s", "history": []})
        api._ended_conversations = cache
        api._MAX_ENDED_CACHE = 1

        errors: list[BaseException] = []

        def _end(cid: str) -> None:
            try:
                api.end_conversation(cid)
            except BaseException as e:
                errors.append(e)

        enders = [threading.Thread(target=_end, args=(cid,)) for cid in ids]
        for t in enders:
            t.start()
        _join_all(enders)

        assert errors == []
        assert len(cache) == 1
        assert next(iter(cache)) in ids
        assert api.list_active_conversations() == []


class TestStep2GetterFallbacks:
    """P1-4 / P1-5: all four conversation getters share one fallback rule
    (plan-2026-09-22T080837-8b258a25/D-004)."""

    def test_p1_4_history_after_end_comes_from_the_cache(self):
        api = _api()
        conv_id, _ = api.start_conversation()
        api.converse("hello", conv_id)
        live = api.get_conversation_history(conv_id)
        assert live
        api.end_conversation(conv_id)
        assert api.get_conversation_history(conv_id) == live

    @pytest.mark.parametrize(
        "getter", ["get_data", "get_current_state", "get_conversation_history"]
    )
    def test_p1_4_unknown_id_still_raises_value_error(self, getter):
        api = _api()
        with pytest.raises(ValueError):
            getattr(api, getter)("never-started")

    def test_p1_4_has_conversation_ended_unknown_id_is_false(self):
        assert _api().has_conversation_ended("never-started") is False

    @pytest.mark.parametrize(
        ("getter", "manager_method", "expected"),
        [
            ("get_data", "get_conversation_data", {"note": "cached"}),
            ("has_conversation_ended", "has_conversation_ended", True),
            ("get_current_state", "get_conversation_state", "end"),
            (
                "get_conversation_history",
                "get_conversation_history",
                [{"user": "hi"}],
            ),
        ],
    )
    def test_p1_5_not_found_fsm_error_falls_back_to_the_cache(
        self, getter, manager_method, expected
    ):
        api = _api()
        conv_id, _ = api.start_conversation()
        api._ended_conversations[conv_id] = {
            "data": {"note": "cached"},
            "state": "end",
            "history": [{"user": "hi"}],
        }
        err = FSMError(f"Conversation {conv_id} not found")
        with patch.object(api.fsm_manager, manager_method, side_effect=err):
            assert getattr(api, getter)(conv_id) == expected

    @pytest.mark.parametrize(
        ("getter", "manager_method"),
        [
            ("get_data", "get_conversation_data"),
            ("has_conversation_ended", "has_conversation_ended"),
            ("get_current_state", "get_conversation_state"),
            ("get_conversation_history", "get_conversation_history"),
        ],
    )
    def test_p1_5_busy_refusal_propagates_despite_a_cache_entry(
        self, getter, manager_method
    ):
        api = _api()
        conv_id, _ = api.start_conversation()
        api._ended_conversations[conv_id] = {
            "data": {},
            "state": "end",
            "history": [],
        }
        busy = ConversationBusyError("still running a turn")
        with patch.object(api.fsm_manager, manager_method, side_effect=busy):
            with pytest.raises(ConversationBusyError):
                getattr(api, getter)(conv_id)

    def test_p1_5_not_found_fsm_error_without_cache_reraises_original(self):
        api = _api()
        conv_id, _ = api.start_conversation()
        err = FSMError(f"Conversation {conv_id} not found")
        with patch.object(api.fsm_manager, "get_conversation_data", side_effect=err):
            with pytest.raises(FSMError) as info:
                api.get_data(conv_id)
        assert info.value is err


# ---------------------------------------------------------------------------
# Step 3: restore_session lock order (P0-3, C-NEW-007)
# ---------------------------------------------------------------------------


class _LockOrderTracker:
    """Records, per thread, which tracked locks are held, and every time the
    manager ``_lock`` is acquired while a ``conv_lock`` is held."""

    def __init__(self) -> None:
        self._local = threading.local()
        self.violations: list[str] = []

    def held(self) -> list[str]:
        if not hasattr(self._local, "held"):
            self._local.held = []
        return self._local.held


class _RecordingLock:
    """Wraps a Lock/RLock; blocking acquires are bounded so a real deadlock
    fails the test instead of hanging the suite."""

    def __init__(self, inner: Any, name: str, tracker: _LockOrderTracker) -> None:
        self._inner = inner
        self._name = name
        self._tracker = tracker

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        held = self._tracker.held()
        if self._name == "manager" and any(h.startswith("conv:") for h in held):
            self._tracker.violations.append(f"_lock under {held}")
        if blocking and timeout == -1:
            timeout = _JOIN_TIMEOUT_S
        got = self._inner.acquire(blocking, timeout)
        if blocking and not got:
            raise AssertionError(f"{self._name} not acquired: possible deadlock")
        if got:
            held.append(self._name)
        return got

    def release(self) -> None:
        self._tracker.held().remove(self._name)
        self._inner.release()

    def __enter__(self) -> bool:
        return self.acquire()

    def __exit__(self, *exc: Any) -> None:
        self.release()


class _WrapNewLocks(dict):
    """``_conversation_locks`` replacement that wraps each new conv_lock."""

    def __init__(self, tracker: _LockOrderTracker) -> None:
        super().__init__()
        self._tracker = tracker

    def __setitem__(self, key, value) -> None:
        super().__setitem__(key, _RecordingLock(value, f"conv:{key}", self._tracker))


class _OneSessionStore:
    """In-memory store that hands back one prepared ``SessionState``."""

    def __init__(self, state: Any) -> None:
        self.state = state

    def save(self, session_id: str, state: Any) -> None:
        self.state = state

    def load(self, session_id: str) -> Any:
        return self.state

    def delete(self, session_id: str) -> bool:
        return True


def _saved_state(api: API, **overrides: Any) -> Any:
    from fsm_llm.session import SessionState

    fields: dict[str, Any] = {
        "conversation_id": "saved",
        "fsm_id": api.fsm_id,
        "current_state": "start",
        "context_data": {"name": "Ada"},
        "conversation_history": [{"user": "hi"}, {"system": "hello"}],
        "conversation_summary": "earlier: talked about tea",
        "metadata": {"pipeline_extracted": {"name": {"turn": 1}}},
        "working_memory": {"buffers": {"core": {"k": "v"}}},
    }
    fields.update(overrides)
    return SessionState(**fields)


def _restore_api(**overrides: Any) -> tuple[API, _OneSessionStore]:
    api = _api()
    store = _OneSessionStore(None)
    api._session_store = store
    store.state = _saved_state(api, **overrides)
    return api, store


def _restored_instance(api: API, conv_id: str) -> Any:
    fsm_id = api._get_current_fsm_conversation_id(conv_id)
    return api.fsm_manager.instances[fsm_id]


class TestStep3RestoreLockOrder:
    def test_p0_3_restore_never_takes_manager_lock_under_conv_lock(self):
        api, _ = _restore_api()
        tracker = _LockOrderTracker()
        manager = api.fsm_manager
        manager._lock = _RecordingLock(manager._lock, "manager", tracker)
        manager._conversation_locks = _WrapNewLocks(tracker)

        outcome: dict[str, Any] = {}

        def _run() -> None:
            try:
                outcome["result"] = api.restore_session("saved")
            except BaseException as exc:  # surfaced below
                outcome["error"] = exc

        worker = threading.Thread(target=_run, daemon=True)
        worker.start()
        _join_all([worker])
        assert "error" not in outcome, outcome.get("error")
        assert tracker.violations == []

        conv_id, _ = outcome["result"]
        inst = _restored_instance(api, conv_id)
        assert inst.context.conversation.summary == "earlier: talked about tea"
        assert api.get_conversation_history(conv_id) == [
            {"user": "hi"},
            {"system": "hello"},
        ]
        from fsm_llm.pipeline import _PROVENANCE_KEY

        assert inst.context.metadata[_PROVENANCE_KEY] == {"name": {"turn": 1}}
        assert inst.context.working_memory.get("core", "k") == "v"

    def test_p0_3_api_has_no_manager_lock_reach_in(self):
        import ast
        import inspect

        from fsm_llm import api as api_module

        tree = ast.parse(inspect.getsource(api_module))
        reach_ins = [
            f"{node.lineno}: fsm_manager.{node.attr}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and node.attr in {"_lock", "_conversation_locks", "instances"}
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "fsm_manager"
        ]
        assert reach_ins == []

    def test_p0_3_seed_on_unknown_conversation_raises_fsm_error(self):
        api = _api()
        with pytest.raises(FSMError, match="not found"):
            api.fsm_manager.seed_restored_conversation(
                "no-such-id",
                summary=None,
                history=[{"user": "hi"}],
                provenance=None,
                working_memory=None,
            )

    def test_p0_3_missing_summary_leaves_fresh_summary(self):
        api, _ = _restore_api(conversation_summary=None)
        conv_id, _ = api.restore_session("saved")
        assert _restored_instance(api, conv_id).context.conversation.summary is None

    def test_p0_3_absent_working_memory_keeps_fresh_default(self):
        api, _ = _restore_api(working_memory=None)
        conv_id, _ = api.restore_session("saved")
        fresh = api.start_conversation()[0]
        restored_wm = _restored_instance(api, conv_id).context.working_memory
        fresh_wm = _restored_instance(api, fresh).context.working_memory
        assert type(restored_wm) is type(fresh_wm)
        if fresh_wm is not None:
            assert restored_wm.list_buffers() == fresh_wm.list_buffers()

    def test_p0_3_null_buffers_mean_default_buffers(self):
        from fsm_llm.memory import DEFAULT_BUFFERS

        api, _ = _restore_api(working_memory={"buffers": None})
        conv_id, _ = api.restore_session("saved")
        wm = _restored_instance(api, conv_id).context.working_memory
        assert set(wm.list_buffers()) == set(DEFAULT_BUFFERS)

    def test_p0_3_explicit_empty_buffers_are_honoured(self):
        api, _ = _restore_api(working_memory={"buffers": {}, "hidden_buffers": []})
        conv_id, _ = api.restore_session("saved")
        wm = _restored_instance(api, conv_id).context.working_memory
        assert wm.list_buffers() == []
        assert wm._hidden_buffers == frozenset()

    def test_p0_3_seed_failure_tears_down_the_new_conversation(self):
        api, _ = _restore_api()
        before = set(api.active_conversations)
        with patch.object(
            api.fsm_manager,
            "seed_restored_conversation",
            side_effect=FSMError("seed failed"),
            create=True,
        ):
            with pytest.raises(FSMError, match="seed failed"):
                api.restore_session("saved")
        assert set(api.active_conversations) == before


# ---------------------------------------------------------------------------
# Step 4: handler/API config passthrough and read-only should_execute probe
# ---------------------------------------------------------------------------


def _sleeping_handler(seconds: float) -> Any:
    import time

    from fsm_llm.handlers import HandlerTiming, create_handler

    def _sleep(ctx: dict[str, Any]) -> dict[str, Any]:
        time.sleep(seconds)
        return {"slept": True}

    return create_handler("sleeper").at(HandlerTiming.PRE_PROCESSING).do(_sleep)


class TestStep4ConfigPassthrough:
    def test_p0_4_handler_timeout_reaches_handler_system(self):
        from fsm_llm.handlers import HandlerExecutionError, HandlerTiming

        api = API.from_definition(
            _blocked_fsm(),
            llm_interface=_mock_llm(),
            handler_error_mode="raise",
            handler_timeout=0.05,
        )
        assert api.handler_system.handler_timeout == 0.05
        api.handler_system.register_handler(_sleeping_handler(0.3))
        with pytest.raises(HandlerExecutionError) as exc_info:
            api.handler_system.execute_handlers(
                HandlerTiming.PRE_PROCESSING, "start", None, {}
            )
        assert isinstance(exc_info.value.original_error, TimeoutError)

    def test_p0_4_default_api_has_no_handler_timeout(self):
        from fsm_llm.handlers import HandlerTiming

        api = API.from_definition(
            _blocked_fsm(), llm_interface=_mock_llm(), handler_error_mode="raise"
        )
        assert api.handler_system.handler_timeout is None
        api.handler_system.register_handler(_sleeping_handler(0.1))
        result = api.handler_system.execute_handlers(
            HandlerTiming.PRE_PROCESSING, "start", None, {}
        )
        assert result == {"slept": True}

    def test_p0_4_max_fsm_cache_size_reaches_manager_and_evicts(self):
        api = API.from_definition(
            _blocked_fsm(), llm_interface=_mock_llm(), max_fsm_cache_size=1
        )
        manager = api.fsm_manager
        api._temp_fsm_definitions["other"] = _blocked_fsm()
        manager.fsm_cache.clear()
        manager.get_fsm_definition(api.fsm_id)
        manager.get_fsm_definition("other")
        assert list(manager.fsm_cache) == ["other"]

    def test_p0_4_default_fsm_cache_size_is_unchanged(self):
        from fsm_llm.constants import DEFAULT_MAX_FSM_CACHE_SIZE
        from fsm_llm.fsm import FSMManager

        api = _api()
        assert api.fsm_manager._max_fsm_cache_size == 64
        assert DEFAULT_MAX_FSM_CACHE_SIZE == 64
        assert FSMManager(llm_interface=_mock_llm())._max_fsm_cache_size == 64

    def test_p0_4_inert_default_handler_timeout_is_gone(self):
        from fsm_llm import constants

        assert not hasattr(constants, "DEFAULT_HANDLER_TIMEOUT")


class TestStep4ReadOnlyProbe:
    @staticmethod
    def _mutating_condition(t, s, ts, ctx, uk) -> bool:
        ctx["probed"] = True
        return True

    def test_p1_3_mutating_condition_raises_and_leaves_context_alone(self):
        from fsm_llm.handlers import (
            HandlerExecutionError,
            HandlerSystem,
            HandlerTiming,
            create_handler,
        )

        system = HandlerSystem(error_mode="raise")
        system.register_handler(
            create_handler("mutator")
            .when(self._mutating_condition)
            .do(lambda ctx: {"ran": True})
        )
        context = {"existing": 1}
        with pytest.raises(HandlerExecutionError) as exc_info:
            system.execute_handlers(HandlerTiming.PRE_PROCESSING, "s", None, context)
        assert context == {"existing": 1}
        assert not isinstance(exc_info.value.original_error, HandlerExecutionError)
        assert isinstance(exc_info.value.__cause__, TypeError)

    def test_p1_3_continue_mode_skips_the_mutating_handler(self):
        from fsm_llm.handlers import HandlerSystem, HandlerTiming, create_handler

        system = HandlerSystem(error_mode="continue")
        system.register_handler(
            create_handler("mutator")
            .with_priority(10)
            .when(self._mutating_condition)
            .do(lambda ctx: {"ran": True})
        )
        system.register_handler(
            create_handler("reader")
            .with_priority(20)
            .when(lambda t, s, ts, ctx, uk: ctx.get("existing") == 1 and "x" not in ctx)
            .do(lambda ctx: {"read": sorted(ctx)})
        )
        context = {"existing": 1}
        result = system.execute_handlers(
            HandlerTiming.PRE_PROCESSING, "s", None, context
        )
        assert result == {"read": ["existing"]}
        assert context == {"existing": 1}

    def test_p1_3_probe_after_first_runner_sees_cascaded_dict(self):
        from fsm_llm.handlers import HandlerSystem, HandlerTiming, create_handler

        seen: list[Any] = []

        def _record(t, s, ts, ctx, uk) -> bool:
            seen.append((type(ctx), dict(ctx)))
            return True

        system = HandlerSystem()
        system.register_handler(
            create_handler("first")
            .with_priority(10)
            .when(_record)
            .do(lambda ctx: {"a": 1})
        )
        system.register_handler(
            create_handler("second").with_priority(20).when(_record).do(lambda ctx: {})
        )
        system.execute_handlers(HandlerTiming.PRE_PROCESSING, "s", None, {"z": 0})
        assert seen[0] == (types.MappingProxyType, {"z": 0})
        assert seen[1] == (dict, {"z": 0, "a": 1})
