"""Regression tests for the 2026-09-22 fsm_llm core audit.

One class per behaviour step of plan-2026-09-22T080837-8b258a25 (steps 2-14).
Each test was run RED against the parent commit of its step before the fix
landed, and names the audit id it pins (``test_p0_1_*``, ``test_p1_4_*``, ...).
"""

from __future__ import annotations

import datetime
import decimal
import json
import threading
import types
import uuid
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
        assert wm.hidden_buffers == frozenset()

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


# ---------------------------------------------------------------------------
# Step 5: classification context_snapshot never carries a secret
# ---------------------------------------------------------------------------


def _snapshot_fsm(context_keys: list[str]) -> FSMDefinition:
    """``triage`` owns one classification field whose record snapshots
    ``context_keys``; ``browse`` is the fallback, so the turn stays put."""
    from fsm_llm.definitions import ClassificationExtractionConfig, IntentDefinition

    triage = State(
        id="triage",
        description="Triage",
        purpose="Route the shopper",
        response_instructions="Respond",
        classification_extractions=[
            ClassificationExtractionConfig(
                field_name="intent",
                intents=[
                    IntentDefinition(name="buy", description="wants to buy"),
                    IntentDefinition(name="browse", description="just looking"),
                ],
                fallback_intent="browse",
                confidence_threshold=0.5,
                context_keys=context_keys,
            )
        ],
        transitions=[
            Transition(
                target_state="done",
                description="Buy",
                conditions=[
                    TransitionCondition(
                        description="buy", logic={"==": [{"var": "intent"}, "buy"]}
                    )
                ],
            )
        ],
    )
    done = State(id="done", description="Done", purpose="End", transitions=[])
    return FSMDefinition(
        name="snapshot_fsm",
        description="Snapshot FSM",
        initial_state="triage",
        states={"triage": triage, "done": done},
    )


def _classify_browse() -> Any:
    """Patch the classifier's litellm boundary to answer ``browse``."""
    import json

    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = json.dumps(
        {"reasoning": "r", "intent": "browse", "confidence": 0.95, "entities": {}}
    )
    return (
        patch("fsm_llm.classification.completion", return_value=resp),
        patch("fsm_llm.classification.get_supported_openai_params", return_value=[]),
    )


_TOP_SECRET = "sk-live-TOPSECRET123456"
_NESTED_SECRET = "sk-proj-NESTEDSECRET987"
_LIST_SECRET = "rt-LISTSECRET-9f8e7d6c5b4a"
_DEEP_SECRET = "hunter2-DEEPSECRET"


class TestStep5ContextSnapshotFilter:
    """Security (D-007): ``classification_results[f]["context_snapshot"]`` is
    built from security-filtered values: a forbidden entry is dropped at every
    depth (top level, nested dict, dict inside a list), allowed values are kept
    exactly. Driven through ``API.converse`` and
    ``FSMManager.get_complete_conversation`` (the monitor's read path)."""

    def _run(self, context: dict[str, Any], keys: list[str]) -> tuple[API, str]:
        api = API.from_definition(_snapshot_fsm(keys), llm_interface=_mock_llm())
        conv_id, _ = api.start_conversation(initial_context=context)
        completion, params = _classify_browse()
        with completion, params:
            api.converse("just looking", conv_id)
        return api, conv_id

    def _context(self) -> dict[str, Any]:
        return {
            "api_key": _TOP_SECRET,
            "topic": "running shoes",
            "profile": {
                "session_token": _NESTED_SECRET,
                "tier": "gold",
                "history": [{"refresh_token": _LIST_SECRET, "label": "a"}, 7],
                "deep": {"password": _DEEP_SECRET, "ok": 1},
            },
        }

    def test_secret_absent_from_snapshot_at_every_depth(self):
        import json

        api, conv_id = self._run(
            self._context(), ["api_key", "profile", "topic", "missing"]
        )
        # The fixture reaches the branch: the secrets are really in context.
        raw = api.fsm_manager.instances[conv_id].context.data
        assert raw["api_key"] == _TOP_SECRET
        assert raw["profile"]["session_token"] == _NESTED_SECRET

        metadata = api.fsm_manager.get_complete_conversation(conv_id)["metadata"]
        snapshot = metadata["classification_results"]["intent"]["context_snapshot"]
        assert snapshot == {
            "topic": "running shoes",
            "profile": {
                "tier": "gold",
                "history": [{"label": "a"}, 7],
                "deep": {"ok": 1},
            },
        }
        dumped = json.dumps(metadata)
        for secret in (_TOP_SECRET, _NESTED_SECRET, _LIST_SECRET, _DEEP_SECRET):
            assert secret not in dumped
        for name in ("api_key", "session_token", "refresh_token", "password"):
            assert name not in dumped

    def test_allowed_values_kept_exactly(self):
        context = {
            "topic": "running shoes",
            "prefs": {"sizes": [9, 9.5], "colour": None, "tags": ("a", "b")},
        }
        api, conv_id = self._run(context, ["topic", "prefs"])
        metadata = api.fsm_manager.get_complete_conversation(conv_id)["metadata"]
        snapshot = metadata["classification_results"]["intent"]["context_snapshot"]
        # JSON-native round trip unchanged (tuple -> list), nothing dropped.
        assert snapshot == {
            "topic": "running shoes",
            "prefs": {"sizes": [9, 9.5], "colour": None, "tags": ["a", "b"]},
        }

    def test_cyclic_value_still_omitted(self):
        cyclic: dict[str, Any] = {"tier": "gold"}
        cyclic["self"] = cyclic
        api, conv_id = self._run({"topic": "t", "loop": cyclic}, ["topic", "loop"])
        metadata = api.fsm_manager.get_complete_conversation(conv_id)["metadata"]
        snapshot = metadata["classification_results"]["intent"]["context_snapshot"]
        assert snapshot == {"topic": "t"}


class TestStep5NamedConstants:
    """The two cross-module context/metadata keys are named constants."""

    def test_output_response_format_constant(self):
        from fsm_llm.constants import CONTEXT_KEY_OUTPUT_RESPONSE_FORMAT

        assert CONTEXT_KEY_OUTPUT_RESPONSE_FORMAT == "_output_response_format"

    def test_provenance_constant_and_pipeline_alias(self):
        from fsm_llm import fsm, pipeline
        from fsm_llm.constants import PROVENANCE_METADATA_KEY

        assert PROVENANCE_METADATA_KEY == "_pipeline_extracted"
        assert pipeline._PROVENANCE_KEY is PROVENANCE_METADATA_KEY
        assert fsm.PROVENANCE_METADATA_KEY is PROVENANCE_METADATA_KEY


# ---------------------------------------------------------------------------
# Step 6: FileSessionStore.save never calls str() on an arbitrary object
# ---------------------------------------------------------------------------


class _Leaky:
    """An object whose ``__str__`` exposes a secret (the D-008 leak path)."""

    def __str__(self) -> str:
        return "password=zzS3CRETzz"

    __repr__ = __str__


class _SubDatetime(datetime.datetime):
    """Not the exact scalar type, so it may override ``__str__``."""

    def __str__(self) -> str:
        return "sub=zzS3CRETzz"


def _save_with(tmp_path: Any, values: dict[str, Any]) -> tuple[API, str, str]:
    """Start a conversation, put *values* into its context, save to disk."""
    from fsm_llm.session import FileSessionStore

    store = FileSessionStore(tmp_path)
    api = API.from_definition(
        _blocked_fsm(), llm_interface=_mock_llm(), session_store=store
    )
    conv_id, _ = api.start_conversation()
    api.fsm_manager.instances[conv_id].context.data.update(values)
    api.save_session(conv_id)
    text = (tmp_path / f"{conv_id}.json").read_text()
    return api, conv_id, text


class TestStep6SessionSaveDefaultHook:
    """P1 session: the ``default=`` hook redacts every non-scalar object."""

    def test_leaky_object_saved_as_placeholder(self, tmp_path):
        _, _, text = _save_with(
            tmp_path, {"profile": _Leaky(), "nested": {"deep": [_Leaky()]}}
        )
        assert "zzS3CRETzz" not in text
        saved = json.loads(text)["context_data"]
        assert saved["profile"] == "<redacted:_Leaky>"
        assert saved["nested"] == {"deep": ["<redacted:_Leaky>"]}

    def test_set_bytes_frozenset_and_scalar_subclass_are_placeholders(self, tmp_path):
        _, _, text = _save_with(
            tmp_path,
            {
                "tags": {"a"},
                "raw": b"zzS3CRETzz",
                "frozen": frozenset({"b"}),
                "sub": _SubDatetime(2026, 1, 2),
            },
        )
        assert "zzS3CRETzz" not in text
        saved = json.loads(text)["context_data"]
        assert saved["tags"] == "<redacted:set>"
        assert saved["raw"] == "<redacted:bytes>"
        assert saved["frozen"] == "<redacted:frozenset>"
        assert saved["sub"] == "<redacted:_SubDatetime>"

    def test_exact_value_scalars_byte_identical_to_default_str(self, tmp_path):
        from fsm_llm.session import SessionState

        scalars = {
            "when": datetime.datetime(2026, 1, 2, 3, 4, 5),
            "day": datetime.date(2026, 1, 2),
            "at": datetime.time(3, 4, 5),
            "span": datetime.timedelta(seconds=90),
            "amount": decimal.Decimal("9.99"),
            "uid": uuid.UUID("12345678-1234-5678-1234-567812345678"),
        }
        _, _, text = _save_with(tmp_path, scalars)
        state = SessionState.model_validate_json(text)
        assert state.context_data == {k: str(v) for k, v in scalars.items()}
        # The whole file is exactly what the old `default=str` would write.
        state.context_data.update(scalars)
        old_bytes = json.dumps(state.model_dump(), indent=2, default=str)
        assert text == old_bytes

    def test_restore_round_trip(self, tmp_path):
        from fsm_llm.session import FileSessionStore

        when = datetime.datetime(2026, 1, 2, 3, 4, 5)
        uid = uuid.UUID("12345678-1234-5678-1234-567812345678")
        _, conv_id, _ = _save_with(
            tmp_path,
            {
                "when": when,
                "amount": decimal.Decimal("9.99"),
                "uid": uid,
                "profile": _Leaky(),
            },
        )
        api2 = API.from_definition(
            _blocked_fsm(),
            llm_interface=_mock_llm(),
            session_store=FileSessionStore(tmp_path),
        )
        rid, _ = api2.restore_session(conv_id)
        data = api2.get_data(rid)
        assert data["when"] == str(when)
        assert data["amount"] == "9.99"
        assert data["uid"] == str(uid)
        assert data["profile"] == "<redacted:_Leaky>"


# ---------------------------------------------------------------------------
# Step 7: definitions and misc correctness (P1-2, P1-6, typed id error,
# shared entity coercer, FSMStackFrame narrowing, version architecture)
# ---------------------------------------------------------------------------


class TestStep7HistorySizeZeroSummary:
    """P1-2: ``max_history_size=0`` digests exchanges before clearing them."""

    def test_p1_2_size_zero_summarises_before_clearing(self):
        from fsm_llm.definitions import Conversation

        conv = Conversation(max_history_size=0)
        conv.add_user_message("hello")
        conv.add_system_message("hi there")
        conv.add_user_message("how are you")
        conv.add_system_message("good")
        assert conv.exchanges == []
        assert conv.summary is not None
        assert "user: hello" in conv.summary
        assert "system: good" in conv.summary

    def test_p1_2_size_zero_keeps_the_summary_cap(self):
        from fsm_llm.definitions import Conversation

        conv = Conversation(max_history_size=0, summary="x" * 1990)
        conv.add_user_message("hello")
        conv.add_system_message("hi there")
        assert conv.summary is not None
        assert len(conv.summary) == 2000
        assert conv.summary.startswith("x" * 1990 + " | user: ")

    def test_p1_2_summary_already_at_cap_stays_at_cap(self):
        from fsm_llm.definitions import Conversation

        conv = Conversation(max_history_size=0, summary="y" * 2000)
        conv.add_user_message("hello")
        conv.add_system_message("hi")
        assert conv.summary == "y" * 2000
        assert conv.exchanges == []


class TestStep7FSMDefinitionNotFoundError:
    """Typed error for the registry-less id branch of ``load_fsm_definition``."""

    def test_id_branch_raises_typed_error_that_is_fsm_and_value_error(self):
        from fsm_llm import FSMDefinitionNotFoundError
        from fsm_llm.utilities import load_fsm_definition

        with pytest.raises(FSMDefinitionNotFoundError) as exc:
            load_fsm_definition("no-such-id")
        err = exc.value
        assert isinstance(err, FSMError)
        assert isinstance(err, ValueError)
        assert err.fsm_id == "no-such-id"
        msg = str(err)
        assert "Unknown FSM ID" in msg
        assert "no FSM registry" in msg
        assert "'no-such-id'" in msg

    def test_error_pickles_with_its_attributes(self):
        import pickle

        from fsm_llm.definitions import FSMDefinitionNotFoundError

        err = FSMDefinitionNotFoundError("abc123")
        clone = pickle.loads(pickle.dumps(err))
        assert type(clone) is FSMDefinitionNotFoundError
        assert str(clone) == str(err)
        assert clone.fsm_id == "abc123"
        assert clone.details == {"fsm_id": "abc123"}

    def test_api_loader_fallback_raises_the_typed_error(self):
        from fsm_llm.definitions import FSMDefinitionNotFoundError

        api = _api()
        with pytest.raises(FSMDefinitionNotFoundError):
            api.fsm_manager.fsm_loader("evicted-content-hash")

    def test_exported_in_all(self):
        import fsm_llm

        assert "FSMDefinitionNotFoundError" in fsm_llm.__all__


class TestStep7SharedEntityCoercer:
    """Both entity validators delegate to one module-level coercer."""

    def test_both_validators_delegate_to_the_shared_function(self):
        import fsm_llm.definitions as d

        real = d._coerce_entity_values
        with patch.object(d, "_coerce_entity_values", wraps=real) as spy:
            d.IntentScore(intent="a", confidence=0.5, entities={"k": 1})
            d.ClassificationResult(
                reasoning="r", intent="a", confidence=0.5, entities={"k": [1, 2]}
            )
        assert [c.args for c in spy.call_args_list] == [({"k": 1},), ({"k": [1, 2]},)]

    def test_shared_function_behaviour(self):
        from fsm_llm.definitions import _coerce_entity_values

        assert _coerce_entity_values({"a": None, "b": 3, "c": ["x", 1]}) == {
            "a": None,
            "b": "3",
            "c": "x, 1",
        }
        assert _coerce_entity_values("junk") == {}


class TestStep7StackFrameNarrowing:
    """``FSMStackFrame.fsm_definition`` accepts only an ``FSMDefinition``."""

    def test_raw_string_definition_is_rejected(self):
        from pydantic import ValidationError

        from fsm_llm.api import FSMStackFrame

        with pytest.raises(ValidationError):
            FSMStackFrame(fsm_definition="some-id", conversation_id="c1")

    def test_definition_object_is_accepted(self):
        from fsm_llm.api import FSMStackFrame

        frame = FSMStackFrame(fsm_definition=_blocked_fsm(), conversation_id="c1")
        assert isinstance(frame.fsm_definition, FSMDefinition)


class TestStep7VersionAndSessionDocs:
    def test_architecture_is_2_pass(self):
        from fsm_llm import get_version_info

        assert get_version_info()["architecture"] == "2-pass"

    def test_p1_6_stack_depth_documented_as_advisory(self):
        from fsm_llm.session import SessionState

        description = SessionState.model_fields["stack_depth"].description or ""
        assert "advisory" in description
        assert "never read by restore" in description


# ---------------------------------------------------------------------------
# Step 8: load-time validation of prompt_config / transition_classification
# ---------------------------------------------------------------------------


def _step8_fsm_data(
    *,
    prompt_config: Any = None,
    transition_classification: Any = None,
) -> dict[str, Any]:
    """A valid FSM dict carrying the two configs under test on state ``start``."""
    classification: dict[str, Any] = {
        "field_name": "intent",
        "intents": [
            {"name": "buy", "description": "Wants to purchase."},
            {"name": "browse", "description": "Just looking."},
        ],
        "fallback_intent": "browse",
    }
    if prompt_config is not None:
        classification["prompt_config"] = prompt_config
    start: dict[str, Any] = {
        "id": "start",
        "description": "Start",
        "purpose": "Start",
        "classification_extractions": [classification],
        "transitions": [{"target_state": "done", "description": "Go", "priority": 100}],
    }
    if transition_classification is not None:
        start["transition_classification"] = transition_classification
    return {
        "name": "Step8",
        "description": "Step 8 fixture",
        "initial_state": "start",
        "states": {
            "start": start,
            "done": {"id": "done", "description": "End", "purpose": "End"},
        },
    }


def _loader_rejects(data: dict[str, Any]) -> bool:
    from pydantic import ValidationError

    try:
        FSMDefinition(**data)
    except ValidationError:
        return True
    return False


def _validator_rejects(data: dict[str, Any]) -> bool:
    from fsm_llm.validator import FSMValidator

    return FSMValidator(data).validate().is_valid is False


_STEP8_BAD_PROMPT_CONFIGS = [
    pytest.param({"max_intent": 2}, id="unknown-key-typo"),
    pytest.param({"max_intents": 99}, id="max-intents-over-cap"),
    pytest.param({"max_intents": 0}, id="max-intents-zero"),
    pytest.param({"max_tokens": 0}, id="max-tokens-zero"),
    pytest.param({"temperature": 2.5}, id="temperature-over"),
    pytest.param({"max_tokens": "5"}, id="max-tokens-str"),
]

_STEP8_BAD_TRANSITION_CLASSIFICATIONS = [
    pytest.param({"done": "str"}, id="entry-not-dict"),
    pytest.param({"confidence_threshold": 1.5}, id="threshold-over"),
    pytest.param({"confidence_threshold": -0.1}, id="threshold-under"),
    pytest.param({"confidence_threshold": True}, id="threshold-bool"),
    pytest.param({"confidence_threshold": "0.7"}, id="threshold-str"),
    pytest.param({"done": {"desc": "typo"}}, id="entry-unknown-key"),
    pytest.param({"done": {"description": 3}}, id="description-not-str"),
]


class TestStep8LoadTimeValidation:
    @pytest.mark.parametrize("prompt_config", _STEP8_BAD_PROMPT_CONFIGS)
    def test_bad_prompt_config_fails_at_load_and_in_validator(self, prompt_config):
        data = _step8_fsm_data(prompt_config=prompt_config)
        assert _loader_rejects(data)
        assert _validator_rejects(data)

    @pytest.mark.parametrize("config", _STEP8_BAD_TRANSITION_CLASSIFICATIONS)
    def test_bad_transition_classification_fails_at_load_and_in_validator(self, config):
        data = _step8_fsm_data(transition_classification=config)
        assert _loader_rejects(data)
        assert _validator_rejects(data)

    @pytest.mark.parametrize(
        "config",
        [
            pytest.param({}, id="empty"),
            pytest.param({"confidence_threshold": 0}, id="threshold-0"),
            pytest.param({"confidence_threshold": 1}, id="threshold-1"),
            pytest.param({"confidence_threshold": 0.7}, id="threshold-float"),
            pytest.param({"done": {"description": None}}, id="description-none"),
            pytest.param({"done": {}}, id="entry-empty"),
            pytest.param(
                {"done": {"description": "Go"}, "confidence_threshold": 0.5},
                id="full",
            ),
        ],
    )
    def test_valid_transition_classification_loads(self, config):
        data = _step8_fsm_data(transition_classification=config)
        assert not _loader_rejects(data)
        assert not _validator_rejects(data)

    def test_auto_mode_none_loads(self):
        state = State(
            id="s", description="d", purpose="p", transition_classification=None
        )
        assert state.transition_classification is None

    @pytest.mark.parametrize(
        "prompt_config",
        [
            {},
            {"max_intents": 5, "temperature": 2.0, "max_tokens": 1},
            {"include_reasoning": False, "multi_intent": True},
        ],
    )
    def test_valid_prompt_config_loads_unchanged(self, prompt_config):
        data = _step8_fsm_data(prompt_config=dict(prompt_config))
        fsm = FSMDefinition(**data)
        loaded = fsm.states["start"].classification_extractions[0].prompt_config
        assert loaded == prompt_config
        assert not _validator_rejects(data)

    @pytest.mark.parametrize(
        "prompt_config",
        [
            {"max_intents": 3},
            {"max_intents": 6},
            {"bogus": True},
            {"temperature": -0.01},
            {"max_tokens": None},
        ],
    )
    def test_load_verdict_equals_extraction_time_verdict(self, prompt_config):
        """Single source: load rejects exactly what the dataclass rejects."""
        from fsm_llm.prompts import ClassificationPromptConfig

        try:
            ClassificationPromptConfig(**prompt_config)
            extraction_rejects = False
        except (TypeError, ValueError):
            extraction_rejects = True
        data = _step8_fsm_data(prompt_config=prompt_config)
        assert _loader_rejects(data) is extraction_rejects

    def test_shipped_manual_mode_example_loads(self):
        from pathlib import Path

        from fsm_llm.utilities import load_fsm_from_file

        path = (
            Path(__file__).resolve().parents[2]
            / "examples/classification/classified_transitions/fsm_manual.json"
        )
        fsm = load_fsm_from_file(str(path))
        config = fsm.states["greeting"].transition_classification
        assert config is not None
        assert config["confidence_threshold"] == 0.7


# ---------------------------------------------------------------------------
# Step 9.1: dead core symbols removed; DEFAULT_TEMPERATURE is the one source
# ---------------------------------------------------------------------------


class TestStep9DeadCoreSymbols:
    @pytest.mark.parametrize(
        "name", ["DomainSchema", "LLMRequestType", "validate_json_structure"]
    )
    def test_removed_public_names_are_gone(self, name):
        import fsm_llm

        assert name not in fsm_llm.__all__
        assert not hasattr(fsm_llm, name)

    @pytest.mark.parametrize(
        "name",
        [
            "LOG_FIELD_TIMESTAMP",
            "LOG_FIELD_PACKAGE",
            "LOG_MESSAGE_PREVIEW_LENGTH",
            "LOG_RESPONSE_PREVIEW_LENGTH",
            "DEFAULT_STEP_TIMEOUT",
        ],
    )
    def test_removed_constants_are_gone(self, name):
        from fsm_llm import constants

        assert not hasattr(constants, name)

    def test_context_compactor_has_no_summarize_on_trim(self):
        from fsm_llm.context import ContextCompactor

        with pytest.raises(TypeError):
            ContextCompactor(summarize_on_trim=True)
        assert not hasattr(ContextCompactor(), "summarize_on_trim")


class TestStep9DefaultTemperature:
    def test_api_default_temperature_comes_from_constant(self, monkeypatch):
        from fsm_llm import api as api_module
        from fsm_llm.constants import DEFAULT_TEMPERATURE

        api = API(fsm_definition=_step9_fsm_dict(), model="gpt-4o-mini")
        assert api.llm_interface.temperature == DEFAULT_TEMPERATURE
        # The API reads the name, not a literal: rebinding it changes the default.
        monkeypatch.setattr(api_module, "DEFAULT_TEMPERATURE", 0.25)
        api = API(fsm_definition=_step9_fsm_dict(), model="gpt-4o-mini")
        assert api.llm_interface.temperature == 0.25

    def test_litellm_interface_default_is_the_constant(self):
        import inspect

        from fsm_llm.constants import DEFAULT_TEMPERATURE
        from fsm_llm.llm import LiteLLMInterface

        param = inspect.signature(LiteLLMInterface.__init__).parameters["temperature"]
        assert param.default is DEFAULT_TEMPERATURE
        assert LiteLLMInterface(model="gpt-4o-mini").temperature == DEFAULT_TEMPERATURE


class TestStep9LeafDeadCode:
    def test_short_circuit_operators_never_reach_operations(self):
        from fsm_llm.expressions import _SHORT_CIRCUIT_OPERATORS, operations

        assert _SHORT_CIRCUIT_OPERATORS == {"and", "or", "if"}
        assert _SHORT_CIRCUIT_OPERATORS.isdisjoint(operations)

    def test_every_allowed_operator_has_exactly_one_dispatcher(self):
        from fsm_llm.constants import ALLOWED_JSONLOGIC_OPERATIONS
        from fsm_llm.expressions import (
            _SHORT_CIRCUIT_OPERATORS,
            _data_operators,
            operations,
        )

        tables = [set(operations), set(_data_operators), set(_SHORT_CIRCUIT_OPERATORS)]
        assert set().union(*tables) == ALLOWED_JSONLOGIC_OPERATIONS
        assert sum(len(t) for t in tables) == len(ALLOWED_JSONLOGIC_OPERATIONS)

    def test_short_circuit_still_lazy_and_if_still_works(self):
        from fsm_llm.expressions import evaluate_logic

        guard = {
            "and": [{"!=": [{"var": "x"}, 0]}, {"<": [{"/": [1, {"var": "x"}]}, 5]}]
        }
        assert evaluate_logic(guard, {"x": 0}) is False
        assert evaluate_logic({"or": [0, "", "last"]}) == "last"
        assert evaluate_logic({"if": [False, "a", True, "b", "c"]}) == "b"
        assert evaluate_logic({"if": [False, "a", "default"]}) == "default"

    def test_if_condition_is_gone(self):
        import fsm_llm.expressions as expressions

        assert not hasattr(expressions, "if_condition")

    def test_ollama_transition_schema_is_gone(self):
        import fsm_llm.ollama as ollama

        assert not hasattr(ollama, "TRANSITION_JSON_SCHEMA")
        assert "transition_decision" not in ollama._CALL_TYPE_SCHEMAS
        assert ollama.build_ollama_response_format("transition_decision") is None

    def test_visualizer_unused_glyphs_are_gone(self):
        from fsm_llm.visualizer import ARROW_STYLES, BOX_STYLES, ICONS

        assert "note" not in ICONS
        assert "section" not in BOX_STYLES
        for key in ("down_arrow", "right_arrow", "diamond"):
            assert key not in ARROW_STYLES

    def test_prompt_helper_params_are_gone(self):
        import inspect

        from fsm_llm.prompts import BasePromptBuilder

        est = inspect.signature(BasePromptBuilder._estimate_token_count).parameters
        assert list(est) == ["self", "text"]
        fmt = inspect.signature(BasePromptBuilder._build_response_format).parameters
        assert "field_heading" not in fmt
        lines = BasePromptBuilder._build_response_format("{}", ["a: b"])
        assert "Where:" in lines

    def test_additional_info_needed_is_gone(self):
        from fsm_llm.definitions import DataExtractionResponse

        assert "additional_info_needed" not in DataExtractionResponse.model_fields
        # Pydantic's default extra="ignore": an old payload still validates.
        resp = DataExtractionResponse.model_validate({"additional_info_needed": True})
        assert not hasattr(resp, "additional_info_needed")


def _step9_fsm_dict() -> dict[str, Any]:
    return {
        "name": "t",
        "description": "d",
        "initial_state": "start",
        "states": {
            "start": {
                "id": "start",
                "description": "s",
                "purpose": "p",
                "response_instructions": "r",
            }
        },
    }


# ---------------------------------------------------------------------------
# Step 10: transition confidence removed, no-op config fields deprecated
# ---------------------------------------------------------------------------


def _step10_state(priorities: list[int], passing: list[bool]) -> State:
    return State(
        id="s",
        description="d",
        purpose="p",
        transitions=[
            Transition(
                target_state=f"t{i}",
                description=f"to t{i}",
                priority=priority,
                conditions=[
                    TransitionCondition(
                        description=f"c{i}",
                        logic={"==": [{"var": "go"}, ok]},
                    )
                ],
            )
            for i, (priority, ok) in enumerate(zip(priorities, passing, strict=True))
        ],
    )


def _step10_evaluate(state: State, config: Any = None) -> Any:
    from fsm_llm.definitions import FSMContext
    from fsm_llm.transition_evaluator import TransitionEvaluator

    return TransitionEvaluator(config).evaluate_transitions(
        state, FSMContext(data={"go": True})
    )


class TestStep10ConfidenceRemoved:
    @pytest.mark.parametrize(
        "name",
        [
            "MIN_BASE_CONFIDENCE",
            "PRIORITY_SCALING_DIVISOR",
            "CONDITION_SUCCESS_RATE_BOOST",
        ],
    )
    def test_confidence_constants_are_gone(self, name):
        from fsm_llm import constants

        assert not hasattr(constants, name)

    def test_transition_evaluation_has_no_confidence_field(self):
        from fsm_llm.definitions import TransitionEvaluation

        assert "confidence" not in TransitionEvaluation.model_fields
        ev = _step10_evaluate(_step10_state([100], [True]))
        assert not hasattr(ev, "confidence")
        assert "confidence" not in ev.model_dump()

    def test_per_transition_scores_carry_no_confidence(self):
        from fsm_llm.transition_evaluator import TransitionEvaluator

        evaluator = TransitionEvaluator()
        state = _step10_state([100, 200], [True, False])
        scores = evaluator._evaluate_individual_transitions(
            state.transitions, {"go": True}
        )
        assert all("confidence" not in s for s in scores)
        cond = evaluator._evaluate_transition_conditions(
            state.transitions[0].conditions, {"go": True}
        )
        assert "confidence_factor" not in cond

    def test_priority_outcomes_unchanged(self):
        from fsm_llm.definitions import TransitionEvaluationResult as R

        det = _step10_evaluate(_step10_state([200, 100], [True, True]))
        assert det.result_type == R.DETERMINISTIC
        assert det.deterministic_transition == "t1"
        amb = _step10_evaluate(_step10_state([100, 100, 300], [True, True, True]))
        assert amb.result_type == R.AMBIGUOUS
        assert [o.target_state for o in amb.available_options] == ["t0", "t1"]
        blocked = _step10_evaluate(_step10_state([100], [False]))
        assert blocked.result_type == R.BLOCKED
        assert blocked.blocked_reason == "c0"


class TestStep10ConfigDeprecation:
    def test_defaults_are_silent(self):
        import warnings

        from fsm_llm.transition_evaluator import TransitionEvaluatorConfig

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            TransitionEvaluatorConfig()
            TransitionEvaluatorConfig(
                ambiguity_threshold=0.1,
                minimum_confidence=0.5,
                evidence_conditions_normalizer=5.0,
                strict_condition_matching=False,
                detailed_logging=True,
            )

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("minimum_confidence", 0.9),
            ("ambiguity_threshold", 0.5),
            ("evidence_conditions_normalizer", 2.0),
        ],
    )
    def test_non_default_deprecated_field_warns_at_caller(self, field, value):
        from fsm_llm.transition_evaluator import TransitionEvaluatorConfig

        with pytest.warns(DeprecationWarning, match="no effect") as record:
            TransitionEvaluatorConfig(**{field: value})
        assert len(record) == 1
        assert field in str(record[0].message)
        assert "1.0" in str(record[0].message)
        assert record[0].filename == __file__

    def test_deprecated_fields_do_not_change_outcomes(self):
        from fsm_llm.definitions import TransitionEvaluationResult as R
        from fsm_llm.transition_evaluator import TransitionEvaluatorConfig

        with pytest.warns(DeprecationWarning):
            config = TransitionEvaluatorConfig(
                ambiguity_threshold=0.9,
                minimum_confidence=0.99,
                evidence_conditions_normalizer=1.0,
            )
        state = _step10_state([100, 101], [True, True])
        ev = _step10_evaluate(state, config)
        assert ev.result_type == R.DETERMINISTIC
        assert ev.deterministic_transition == "t0"
        assert ev == _step10_evaluate(state)


# ---------------------------------------------------------------------------
# Step 11: skip_generation flag and the write-only request fields (D-034)
# ---------------------------------------------------------------------------


class _SkipRecordingLLM(LLMInterface):
    """A custom interface that decides the skip from ``skip_generation`` only."""

    def __init__(self) -> None:
        self.requests: list[Any] = []
        self.stream_requests: list[Any] = []

    def generate_response(self, request):
        from fsm_llm.definitions import ResponseGenerationResponse

        self.requests.append(request)
        skipped = getattr(request, "skip_generation", False)
        return ResponseGenerationResponse(message="" if skipped else "real reply")

    def generate_response_stream(self, request):
        self.stream_requests.append(request)
        yield "real reply"


def _step11_fsm(initial_instructions: str) -> FSMDefinition:
    """``start`` moves unconditionally to ``think`` (empty instructions)."""
    states = {
        "start": State(
            id="start",
            description="Start",
            purpose="Start",
            response_instructions=initial_instructions,
            transitions=[Transition(target_state="think", description="Always")],
        ),
        "think": State(
            id="think",
            description="Think",
            purpose="Think",
            response_instructions="",
            transitions=[Transition(target_state="end", description="Done")],
        ),
        "end": State(id="end", description="End", purpose="End", transitions=[]),
    }
    return FSMDefinition(
        name="step11",
        description="Pass-2 skip sites",
        initial_state="start",
        states=states,
    )


def _step11_request(**overrides: Any) -> Any:
    from fsm_llm.definitions import ResponseGenerationRequest

    fields: dict[str, Any] = {"system_prompt": "A real prompt", "user_message": "hi"}
    fields.update(overrides)
    return ResponseGenerationRequest(**fields)


class TestStep11SkipGeneration:
    def test_greeting_skip_site_sets_flag(self):
        llm = _SkipRecordingLLM()
        api = API.from_definition(_step11_fsm(""), llm_interface=llm)
        _, greeting = api.start_conversation()
        assert greeting == "[start]"
        assert len(llm.requests) == 1
        assert llm.requests[0].skip_generation is True
        assert llm.requests[0].system_prompt == "."

    def test_sync_skip_site_sets_flag(self):
        llm = _SkipRecordingLLM()
        api = API.from_definition(_step11_fsm("Greet"), llm_interface=llm)
        conv_id, greeting = api.start_conversation()
        assert greeting == "real reply"
        assert llm.requests[0].skip_generation is False
        assert api.converse("go", conv_id) == "[think]"
        assert len(llm.requests) == 2
        assert llm.requests[1].skip_generation is True
        assert llm.requests[1].system_prompt == "."

    def test_stream_skip_site_makes_no_call(self):
        """The stream skip site builds no request (D-034): nothing to mark."""
        llm = _SkipRecordingLLM()
        api = API.from_definition(_step11_fsm("Greet"), llm_interface=llm)
        conv_id, _ = api.start_conversation()
        assert list(api.converse_stream("go", conv_id)) == ["[think]"]
        assert len(llm.requests) == 1
        assert llm.stream_requests == []

    def test_normal_pass2_request_is_not_marked(self):
        llm = _SkipRecordingLLM()
        api = API.from_definition(_step11_fsm("Greet"), llm_interface=llm)
        api.start_conversation()
        assert llm.requests[0].skip_generation is False
        assert llm.requests[0].system_prompt != "."

    @pytest.mark.parametrize(
        "overrides",
        [
            {"skip_generation": True},
            {"system_prompt": "."},
            {"system_prompt": ".", "skip_generation": True},
        ],
    )
    def test_litellm_skips_on_either_signal(self, overrides):
        from fsm_llm.llm import LiteLLMInterface

        llm = LiteLLMInterface(model="gpt-4o-mini")
        request = _step11_request(**overrides)
        with patch("fsm_llm.llm.completion") as completion:
            response = llm.generate_response(request)
            chunks = list(llm.generate_response_stream(request))
        completion.assert_not_called()
        assert response.message == ""
        assert chunks == [""]

    def test_write_only_fields_are_gone_and_old_kwargs_are_ignored(self):
        request = _step11_request(
            extracted_data={"a": 1}, context={"b": 2}, previous_state="s"
        )
        for name in ("extracted_data", "context", "previous_state"):
            assert not hasattr(request, name)
        assert request.skip_generation is False


# ---------------------------------------------------------------------------
# Step 12.1: prune_orphaned_locks / is_below_default_threshold renames
# ---------------------------------------------------------------------------


def _step12_manager():
    from fsm_llm.fsm import FSMManager

    manager = FSMManager(llm_interface=_mock_llm(), fsm_loader=lambda x: None)
    manager._conversation_locks["orphan"] = threading.Lock()
    manager._conversation_locks["active"] = threading.Lock()
    manager.instances["active"] = MagicMock()
    return manager


def _step12_result(confidence: float):
    from fsm_llm.definitions import ClassificationResult

    return ClassificationResult(reasoning="r", intent="a", confidence=confidence)


class TestStep12Renames:
    def test_prune_orphaned_locks_prunes_only_orphans(self):
        manager = _step12_manager()
        assert manager.prune_orphaned_locks() == ["orphan"]
        assert set(manager._conversation_locks) == {"active"}

    def test_old_manager_name_warns_at_caller_and_still_prunes(self):
        manager = _step12_manager()
        with pytest.warns(DeprecationWarning, match="prune_orphaned_locks") as rec:
            assert manager.cleanup_stale_conversations() == ["orphan"]
        assert rec[0].filename == __file__
        assert set(manager._conversation_locks) == {"active"}

    def test_api_cleanup_stale_conversations_is_not_deprecated(self):
        import warnings

        api = API.from_definition(_step12_fsm_dict(), llm_interface=_mock_llm())
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            assert api.cleanup_stale_conversations(max_idle_seconds=3600) == []

    @pytest.mark.parametrize(
        ("confidence", "expected"),
        [(0.0, True), (0.59, True), (0.6, False), (1.0, False)],
    )
    def test_is_below_default_threshold(self, confidence, expected):
        import warnings

        result = _step12_result(confidence)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            assert result.is_below_default_threshold is expected

    def test_old_result_property_warns_at_caller_and_matches(self):
        for confidence in (0.1, 0.9):
            result = _step12_result(confidence)
            with pytest.warns(
                DeprecationWarning, match="is_below_default_threshold"
            ) as rec:
                old = result.is_low_confidence
            assert rec[0].filename == __file__
            assert old is result.is_below_default_threshold

    def test_classifier_method_keeps_its_name_without_warning(self):
        import warnings

        from fsm_llm.classification import Classifier
        from fsm_llm.definitions import ClassificationSchema, IntentDefinition

        clf = Classifier(
            schema=ClassificationSchema(
                intents=[
                    IntentDefinition(name="a", description="A"),
                    IntentDefinition(name="b", description="B"),
                ],
                fallback_intent="b",
                confidence_threshold=0.8,
            ),
            model="gpt-4o",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            assert clf.is_low_confidence(_step12_result(0.7)) is True
            assert clf.is_low_confidence(_step12_result(0.8)) is False

    def test_model_dump_does_not_expose_either_property(self):
        dumped = _step12_result(0.5).model_dump()
        assert "is_low_confidence" not in dumped
        assert "is_below_default_threshold" not in dumped


def _step12_fsm_dict() -> dict:
    return {
        "name": "s12",
        "description": "d",
        "initial_state": "start",
        "states": {
            "start": {
                "id": "start",
                "description": "d",
                "purpose": "p",
                "response_instructions": "",
            }
        },
    }


def _module_source(module: Any) -> str:
    import inspect

    return inspect.getsource(module)


class TestStep12Accessors:
    """Step 12.2: public accessors replace three private reach-ins
    (``WorkingMemory._hidden_buffers``, the logging handler registry, and a
    prompt builder's ``_sanitize_text_for_prompt``)."""

    def test_hidden_buffers_returns_the_same_frozenset_object(self):
        from fsm_llm.memory import WorkingMemory

        wm = WorkingMemory(hidden_buffers={"metadata", "audit"})
        assert type(wm.hidden_buffers) is frozenset
        assert wm.hidden_buffers is wm._hidden_buffers
        assert wm.hidden_buffers == frozenset({"metadata", "audit"})

    def test_hidden_buffers_is_read_only(self):
        from fsm_llm.memory import WorkingMemory

        wm = WorkingMemory()
        before = wm.hidden_buffers
        with pytest.raises(AttributeError):
            wm.hidden_buffers = frozenset({"x"})  # type: ignore[misc]
        with pytest.raises(AttributeError):
            del wm.hidden_buffers  # type: ignore[misc]
        assert WorkingMemory.hidden_buffers.fset is None
        assert wm.hidden_buffers is before

    @pytest.mark.parametrize("how", ["copy", "deepcopy", "pickle"])
    def test_hidden_buffers_stays_a_frozenset_on_every_copy_path(self, how):
        import copy
        import pickle

        from fsm_llm.memory import WorkingMemory

        wm = WorkingMemory(hidden_buffers={"metadata"})
        clone = {
            "copy": copy.copy,
            "deepcopy": copy.deepcopy,
            "pickle": lambda m: pickle.loads(pickle.dumps(m)),
        }[how](wm)
        assert type(clone.hidden_buffers) is frozenset
        assert clone.hidden_buffers == frozenset({"metadata"})

    def test_snapshot_reads_the_public_property(self):
        from fsm_llm.memory import WorkingMemory

        class _Override(WorkingMemory):
            @property
            def hidden_buffers(self) -> frozenset[str]:
                return frozenset({"via_property"})

        api = _api()
        conv_id, _ = api.start_conversation()
        instance = api.fsm_manager.instances[conv_id]
        instance.context.working_memory = _Override()
        snapshot = api.fsm_manager.get_conversation_snapshot(conv_id)
        assert snapshot["hidden_buffers"] == ["via_property"]

    def test_snapshot_keeps_a_default_for_foreign_memory_objects(self):
        class _Foreign:
            def to_dict(self) -> dict[str, Any]:
                return {"core": {}}

        api = _api()
        conv_id, _ = api.start_conversation()
        api.fsm_manager.instances[conv_id].context.working_memory = _Foreign()
        snapshot = api.fsm_manager.get_conversation_snapshot(conv_id)
        assert snapshot["hidden_buffers"] == []

    def test_fsm_source_has_no_hidden_buffers_reach_in(self):
        import fsm_llm.fsm as fsm_module

        assert "_hidden_buffers" not in _module_source(fsm_module)

    def test_reset_handlers_drops_library_state_and_keeps_user_handlers(self):
        import io

        from loguru import logger

        import fsm_llm.logging as log_module
        from fsm_llm.constants import LOG_SINK_STDOUT

        user_buf = io.StringIO()
        user_id = logger.add(user_buf, level="DEBUG")
        try:
            lib_id = log_module.setup_logging(sink=LOG_SINK_STDOUT)
            assert lib_id in log_module._library_handler_ids
            assert log_module._stream_handler_ids
            log_module._file_handler_initialized = True
            ids_list = log_module._library_handler_ids
            stream_dict = log_module._stream_handler_ids

            log_module.reset_handlers()

            assert log_module._library_handler_ids == []
            assert log_module._stream_handler_ids == {}
            assert log_module._file_handler_initialized is False
            # Cleared in place: tests import these containers by reference.
            assert log_module._library_handler_ids is ids_list
            assert log_module._stream_handler_ids is stream_dict
            with pytest.raises(ValueError):
                logger.remove(lib_id)
            logger.info("user handler survives")
            assert "user handler survives" in user_buf.getvalue()
        finally:
            logger.remove(user_id)
            log_module.reset_handlers()

    def test_reset_handlers_tolerates_an_already_removed_handler(self):
        from loguru import logger

        import fsm_llm.logging as log_module
        from fsm_llm.constants import LOG_SINK_STDOUT

        lib_id = log_module.setup_logging(sink=LOG_SINK_STDOUT)
        logger.remove(lib_id)
        log_module.reset_handlers()
        assert log_module._library_handler_ids == []

    def test_enable_debug_logging_dedup_is_unchanged(self):
        import fsm_llm.logging as log_module
        from fsm_llm import enable_debug_logging
        from fsm_llm.constants import LOG_FORMAT_HUMAN, LOG_SINK_STDERR

        try:
            enable_debug_logging()
            count = len(log_module._library_handler_ids)
            assert count == 1
            assert (
                log_module.setup_logging(sink=LOG_SINK_STDERR, format=LOG_FORMAT_HUMAN)
                == -1
            )
            assert len(log_module._library_handler_ids) == count
            # A different handler shape still registers.
            assert (
                log_module.setup_logging(
                    sink=LOG_SINK_STDERR, format=LOG_FORMAT_HUMAN, context=True
                )
                != -1
            )
        finally:
            log_module.reset_handlers()

    def test_init_does_not_touch_logging_private_state(self):
        import fsm_llm

        source = _module_source(fsm_llm)
        for name in (
            "_library_handler_ids",
            "_stream_handler_ids",
            "_file_handler_initialized",
        ):
            assert name not in source, name

    @pytest.mark.parametrize(
        "text",
        [
            None,
            "",
            "plain text",
            "line one\nline two\r\n",
            "<task>do it</task>",
            "bye </task",
            "latency < threshold",
            "<b>bold</b> and <i>it</i>",
            "<b </task>",
            "< name" + "x" * 300 + ">",
            "<!-- c --> <![CDATA[x]]> <?xml ?>",
            "<a<a<a<a",
        ],
    )
    def test_sanitize_text_for_prompt_matches_the_builder(self, text):
        from fsm_llm.prompts import (
            DataExtractionPromptBuilder,
            ResponseGenerationPromptBuilder,
            sanitize_text_for_prompt,
        )

        expected = DataExtractionPromptBuilder()._sanitize_text_for_prompt(text)
        assert sanitize_text_for_prompt(text) == expected
        assert (
            ResponseGenerationPromptBuilder()._sanitize_text_for_prompt(text)
            == expected
        )

    def test_pipeline_source_has_no_private_sanitiser_reach_in(self):
        import fsm_llm.pipeline as pipeline_module

        assert "._sanitize_text_for_prompt" not in _module_source(pipeline_module)
