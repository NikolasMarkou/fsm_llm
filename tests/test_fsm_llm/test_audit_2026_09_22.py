"""Regression tests for the 2026-09-22 fsm_llm core audit.

One class per behaviour step of plan-2026-09-22T080837-8b258a25 (steps 2-14).
Each test was run RED against the parent commit of its step before the fix
landed, and names the audit id it pins (``test_p0_1_*``, ``test_p1_4_*``, ...).
"""

from __future__ import annotations

import threading
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
