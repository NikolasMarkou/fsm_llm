"""Integration-seam regression tests for the FSM stack lifecycle (`API.push_fsm`/`API.pop_fsm`).

T1 (DECISION plan-2026-07-18-80b0bd4d/D-013): `pop_fsm` calls the blocking
`fsm_manager.end_conversation` with `_stack_lock` DELIBERATELY released (see the D-001 narrow-lock
comment in `api.py`), then re-acquires the lock to remove the frame. The old removal predicate was
positional (`inner_stack[-1].conversation_id == current_frame.conversation_id`), so a concurrent
`push_fsm` on the same root id during that window appended a new top frame, the predicate went
False, and the already-ended frame was stranded in the stack forever — permanently bricking the
conversation (`converse` and every later `pop_fsm` fail, since
`_get_current_fsm_conversation_id` keeps returning a torn-down FSMManager conversation).

Determinism contract for the threaded tests in this file:
- NO `time.sleep()` anywhere. Ordering is forced purely with `threading.Barrier` /
  `threading.Event`, both of which the monkeypatched `end_conversation` participates in.
- EVERY `wait()` and `join()` carries an explicit timeout, and the test asserts the wait actually
  returned rather than timing out — so a regression fails loudly instead of hanging CI.
"""

import threading
import time as _real_time

import pytest

from fsm_llm import api as api_module
from fsm_llm.api import API
from fsm_llm.definitions import FSMDefinition
from tests.conftest import MockLLM2Interface

# Generous relative to the work being done (all in-process, mocked LLM); small enough that a
# genuine regression surfaces as a fast failure rather than a CI hang.
WAIT_TIMEOUT = 10.0


def _make_fsm(name: str) -> FSMDefinition:
    """Two-state FSM: `greeting` -> `farewell` once `user_name` is in context."""
    return FSMDefinition.model_validate(
        {
            "name": name,
            "description": "stack lifecycle seam test FSM",
            "version": "4.1",
            "initial_state": "greeting",
            "states": {
                "greeting": {
                    "id": "greeting",
                    "description": "greeting state",
                    "purpose": "greet the user",
                    "transitions": [
                        {
                            "target_state": "farewell",
                            "description": "user identified",
                            "priority": 100,
                            "conditions": [
                                {
                                    "description": "user_name known",
                                    "requires_context_keys": ["user_name"],
                                }
                            ],
                        }
                    ],
                },
                "farewell": {
                    "id": "farewell",
                    "description": "farewell state",
                    "purpose": "say goodbye",
                    "transitions": [],
                },
            },
        }
    )


@pytest.fixture
def api():
    """API on a mocked 2-pass LLM, closed deterministically after the test."""
    instance = API(fsm_definition=_make_fsm("main"), llm_interface=MockLLM2Interface())
    try:
        yield instance
    finally:
        instance.close()


def _frame_ids(api_instance: API, conversation_id: str) -> list[str]:
    with api_instance._stack_lock:
        return [
            f.conversation_id for f in api_instance.conversation_stacks[conversation_id]
        ]


class TestPopPushRace:
    """SC-1: a forced pop/push interleaving must not strand the ended frame."""

    def test_concurrent_push_during_pop_does_not_strand_ended_frame(self, api):
        conv_id, _ = api.start_conversation()
        api.push_fsm(conv_id, _make_fsm("sub1"))
        assert api.get_stack_depth(conv_id) == 2

        # The frame pop_fsm is about to end — captured by identity, which is exactly the
        # property the fix relies on.
        with api._stack_lock:
            doomed_frame = api.conversation_stacks[conv_id][-1]
        doomed_id = doomed_frame.conversation_id

        original_end_conversation = api.fsm_manager.end_conversation
        # Rendezvous: pop's end_conversation blocks here until the push thread arrives, so the
        # push provably happens inside the lock-released window. No sleep required.
        window_open = threading.Barrier(2, timeout=WAIT_TIMEOUT)
        push_done = threading.Event()

        def blocking_end_conversation(*args, **kwargs):
            window_open.wait()
            assert push_done.wait(timeout=WAIT_TIMEOUT), (
                "push thread did not finish inside the end_conversation window"
            )
            return original_end_conversation(*args, **kwargs)

        api.fsm_manager.end_conversation = blocking_end_conversation

        errors: list[BaseException] = []

        def do_push():
            try:
                window_open.wait()
                api.push_fsm(conv_id, _make_fsm("sub2"))
            except BaseException as exc:
                errors.append(exc)
            finally:
                push_done.set()

        push_thread = threading.Thread(target=do_push, name="race-push")
        push_thread.start()
        try:
            api.pop_fsm(conv_id)
        finally:
            push_thread.join(timeout=WAIT_TIMEOUT)
            api.fsm_manager.end_conversation = original_end_conversation

        assert not push_thread.is_alive(), (
            "push thread hung — interleaving is not deterministic"
        )
        assert errors == [], f"push thread raised: {errors}"

        # The ended frame must be gone, from wherever in the stack it ended up.
        remaining = _frame_ids(api, conv_id)
        assert doomed_id not in remaining, (
            f"ended frame {doomed_id} stranded in stack {remaining}"
        )
        # Root frame + the concurrently pushed frame.
        assert api.get_stack_depth(conv_id) == 2

        # The conversation must still be usable — this is what the defect destroyed.
        assert isinstance(api.converse("hello there", conv_id), str)

        # And the stack must still be poppable down to the root.
        assert isinstance(api.pop_fsm(conv_id), str)
        assert api.get_stack_depth(conv_id) == 1

    def test_removal_is_by_identity_not_value_equality(self, api):
        """FSMStackFrame has pydantic VALUE equality — `==`/list.remove could delete the wrong
        frame. Pin that two value-equal frames are distinguishable and that `pop_fsm` removes
        exactly the one it ended."""
        conv_id, _ = api.start_conversation()
        with api._stack_lock:
            root = api.conversation_stacks[conv_id][0]
        twin = root.model_copy(deep=True)
        assert twin == root, "precondition: FSMStackFrame compares by value"
        assert twin is not root, "precondition: value equality does not imply identity"

        api.push_fsm(conv_id, _make_fsm("sub1"))
        with api._stack_lock:
            doomed_id = api.conversation_stacks[conv_id][-1].conversation_id
        api.pop_fsm(conv_id)
        assert _frame_ids(api, conv_id) == [conv_id]
        assert doomed_id not in _frame_ids(api, conv_id)


class TestSingleThreadedStackNonRegression:
    """Ordinary, uncontended push/pop behaviour must be completely unchanged."""

    def test_push_then_pop_resumes_previous_conversation(self, api):
        conv_id, _ = api.start_conversation()
        assert api.get_stack_depth(conv_id) == 1

        api.push_fsm(conv_id, _make_fsm("sub1"))
        assert api.get_stack_depth(conv_id) == 2

        response = api.pop_fsm(conv_id)
        assert response == "Resumed previous conversation."
        assert api.get_stack_depth(conv_id) == 1
        assert _frame_ids(api, conv_id) == [conv_id]

    def test_pop_with_only_root_frame_still_raises(self, api):
        from fsm_llm.definitions import FSMError

        conv_id, _ = api.start_conversation()
        with pytest.raises(FSMError, match="only one FSM remaining"):
            api.pop_fsm(conv_id)
        assert api.get_stack_depth(conv_id) == 1

    def test_nested_push_pop_unwinds_in_order(self, api):
        conv_id, _ = api.start_conversation()
        api.push_fsm(conv_id, _make_fsm("sub1"))
        api.push_fsm(conv_id, _make_fsm("sub2"))
        assert api.get_stack_depth(conv_id) == 3

        ids_before = _frame_ids(api, conv_id)
        api.pop_fsm(conv_id)
        assert _frame_ids(api, conv_id) == ids_before[:2]
        api.pop_fsm(conv_id)
        assert _frame_ids(api, conv_id) == ids_before[:1]


# ==========================================================================================
# H1 — a pushed sub-FSM definition must survive LRU eviction of the FSM cache
#
# `push_fsm` used to pop `_temp_fsm_definitions[processed_fsm_id]` right after
# starting the sub-conversation, so the sub-FSM survived ONLY in the 64-entry LRU
# `fsm_cache`. Once ~65 distinct FSM ids were loaded, the pushed def was evicted
# and `custom_fsm_loader` fell through to `load_fsm_definition`, which raises
# `ValueError: Unknown FSM ID` for a non-path content-hash id — permanently
# bricking every stacked sub-conversation. See decisions.md D-011.
# ==========================================================================================


class TestPushedFsmSurvivesLruEviction:
    def test_sub_conversation_survives_cache_eviction(self, api):
        """A pushed sub-FSM must stay operable after its cache entry is evicted."""
        # Squeeze the LRU cache so a single unrelated load evicts the sub-FSM.
        api.fsm_manager._max_fsm_cache_size = 1

        conv_a, _ = api.start_conversation()
        api.push_fsm(conv_a, _make_fsm("sub1"))
        assert api.get_stack_depth(conv_a) == 2

        # Loading a *different* FSM id (the root, for a second conversation) into
        # the 1-entry cache evicts the pushed sub-FSM definition.
        api.start_conversation()

        # Operating on the sub-conversation must NOT raise
        # `ValueError: Unknown FSM ID` — the def has to be re-resolvable.
        assert isinstance(api.get_current_state(conv_a), str)
        assert isinstance(api.converse("hello there", conv_a), str)

    def test_temp_definitions_empty_after_full_teardown(self, api):
        """No leak: every pushed def is released once its frames are gone."""
        conv_id, _ = api.start_conversation()
        api.push_fsm(conv_id, _make_fsm("sub1"))
        api.push_fsm(conv_id, _make_fsm("sub2"))
        with api._stack_lock:
            assert api._temp_fsm_definitions, "pushed defs should be registered"

        # Pop one frame: its def is released, the other pushed def stays.
        api.pop_fsm(conv_id)
        with api._stack_lock:
            assert len(api._temp_fsm_definitions) == 1

        # Ending the conversation releases everything.
        api.end_conversation(conv_id)
        with api._stack_lock:
            assert api._temp_fsm_definitions == {}

    def test_shared_sub_fsm_survives_one_conversations_pop(self, api):
        """A sub-FSM SHARED by two conversations (same content hash) must
        survive one conversation's pop_fsm and remain operable for the other."""
        api.fsm_manager._max_fsm_cache_size = 1
        shared = _make_fsm("shared_sub")

        conv_a, _ = api.start_conversation()
        conv_b, _ = api.start_conversation()
        api.push_fsm(conv_a, shared)
        api.push_fsm(conv_b, shared)  # identical content hash -> same fsm_id

        # Exactly one shared temp entry backs both frames.
        with api._stack_lock:
            assert len(api._temp_fsm_definitions) == 1

        # conv_a pops its frame; the shared def must NOT be released — conv_b
        # still references it.
        api.pop_fsm(conv_a)
        with api._stack_lock:
            assert len(api._temp_fsm_definitions) == 1, (
                "shared sub-FSM def evicted while conv_b still references it"
            )

        # Force cache eviction, then drive conv_b's still-live sub-conversation.
        api.start_conversation()
        assert isinstance(api.converse("hello there", conv_b), str)


# ==========================================================================================
# H1 (race) — a concurrent cleanup during a push's register->append window must NOT
# evict the in-flight pushed def.
#
# `push_fsm` registers the temp def under `_stack_lock`, RELEASES the lock to run
# `start_conversation` (which may do a greeting LLM call), then re-acquires the lock
# to append the referencing frame. In that window the def is registered but NO frame
# references it, so a concurrent `pop_fsm`/`end_conversation` on ANOTHER conversation
# runs `_release_unreferenced_temp_definitions`, sees the id as unreferenced, and
# evicts it -> the in-flight sub-conversation bricks with `ValueError: Unknown FSM ID`.
# The `_pending_push_ids` guard treats in-flight pushes as referenced. See D-011 / D-013.
# ==========================================================================================


class TestPushRegisterAppendRace:
    def test_cleanup_during_push_start_does_not_evict_pending_def(self, api):
        """Deterministically drive a concurrent cleanup INTO the register->append
        window by hooking `start_conversation` (called by push after the temp def
        is registered but before the frame is appended). Without the pending guard
        the def is evicted mid-push and the sub-conversation bricks."""
        # Squeeze the LRU cache so an eviction of the temp entry is unrecoverable:
        # once dropped from `_temp_fsm_definitions`, the loader falls through to
        # `load_fsm_definition`, which raises for a content-hash id.
        api.fsm_manager._max_fsm_cache_size = 1

        conv_id, _ = api.start_conversation()

        original_start = api.fsm_manager.start_conversation
        armed = {"on": False}

        def start_with_concurrent_cleanup(*args, **kwargs):
            # This runs INSIDE push_fsm, after the temp def is registered and the
            # `_stack_lock` has been released, and before the frame is appended —
            # exactly the race window. Simulate another conversation's cleanup.
            if armed["on"]:
                armed["on"] = False
                api._release_unreferenced_temp_definitions()
            return original_start(*args, **kwargs)

        api.fsm_manager.start_conversation = start_with_concurrent_cleanup
        try:
            armed["on"] = True
            api.push_fsm(conv_id, _make_fsm("sub_raced"))
        finally:
            api.fsm_manager.start_conversation = original_start

        # The pushed sub def must have survived the mid-push cleanup. The top
        # (pushed) frame carries the content-hash id registered in
        # `_temp_fsm_definitions`; the root frame's id is the main FSM id and is
        # resolved directly, never a temp entry.
        with api._stack_lock:
            sub_fsm_id = api.conversation_stacks[conv_id][-1].fsm_id
            assert sub_fsm_id is not None, "sub frame lost its fsm_id"
            assert sub_fsm_id in api._temp_fsm_definitions, (
                "in-flight pushed def was evicted by the concurrent cleanup"
            )

        # And the sub-conversation must still resolve/operate (loader can find it).
        assert isinstance(api.get_current_state(conv_id), str)
        assert isinstance(api.converse("hello there", conv_id), str)

    def test_pending_push_id_survives_direct_cleanup(self, api):
        """Unit-level: an id marked in-flight in `_pending_push_ids` must NOT be
        dropped by `_release_unreferenced_temp_definitions`, even with no frame
        referencing it yet."""
        pending_id = "pending-hash-xyz"
        with api._stack_lock:
            api._temp_fsm_definitions[pending_id] = _make_fsm("pending")
            api._pending_push_ids.add(pending_id)

        api._release_unreferenced_temp_definitions()

        with api._stack_lock:
            assert pending_id in api._temp_fsm_definitions, (
                "pending in-flight def was evicted by cleanup"
            )

        # Once no longer pending and unreferenced, it IS releasable.
        with api._stack_lock:
            api._pending_push_ids.discard(pending_id)
        api._release_unreferenced_temp_definitions()
        with api._stack_lock:
            assert pending_id not in api._temp_fsm_definitions

    def test_pending_guard_is_cleared_after_push_completes(self, api):
        """No leak: a completed push must leave `_pending_push_ids` empty so the
        def stays releasable by ordinary reference counting afterwards."""
        conv_id, _ = api.start_conversation()
        api.push_fsm(conv_id, _make_fsm("sub1"))
        with api._stack_lock:
            assert api._pending_push_ids == set(), "push left a leaked pending-id entry"
        api.end_conversation(conv_id)
        with api._stack_lock:
            assert api._temp_fsm_definitions == {}
            assert api._pending_push_ids == set()


# ==========================================================================================
# T2 — cleanup_stale_conversations must not evict actively-driven conversations
# ==========================================================================================

MAX_IDLE = 0.05


class _FakeClock:
    """Stand-in for the `time` module inside `fsm_llm.api` only.

    `cleanup_stale_conversations` decides eviction by pure arithmetic on `time.monotonic()`
    reads (`api.py`: `now - last_access > max_idle_seconds`), so a controllable clock makes the
    decision FULLY deterministic — no `time.sleep()`, no wall-clock race, no flake surface.
    Every other attribute delegates to the real module, so nothing else in `api.py` changes
    behaviour, and only `fsm_llm.api`'s own `time` binding is replaced (never the global
    `time` module, which would leak into unrelated tests).
    """

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def __getattr__(self, name):
        return getattr(_real_time, name)


@pytest.fixture
def clock(monkeypatch):
    fake = _FakeClock()
    monkeypatch.setattr(api_module, "time", fake)
    return fake


class TestStaleCleanupRespectsNonConverseActivity:
    """SC-3: a conversation driven ONLY through push_fsm/get_data/update_context is active.

    `_last_accessed` used to be written exclusively by start_conversation/converse/
    converse_stream, so a caller using the documented stacking + polling APIs was force-ended
    by the documented periodic-maintenance API, after which `converse` raised
    `ValueError: Unknown conversation ID` forever.
    """

    def test_push_get_update_activity_prevents_eviction(self, api, clock):
        conv_id, _ = api.start_conversation()

        # Idle far longer than the threshold, so the ONLY thing that can save this
        # conversation is the activity below refreshing its timestamp.
        clock.advance(1.0)

        api.push_fsm(conv_id, _make_fsm("sub1"))
        assert api.get_data(conv_id) is not None
        api.update_context(conv_id, {"user_name": "ada"})

        # Only a sliver of idle time since that activity.
        clock.advance(MAX_IDLE / 5)

        assert api.cleanup_stale_conversations(MAX_IDLE) == []
        assert conv_id in api.list_active_conversations()
        # The real damage the defect caused: converse permanently broken afterwards.
        assert isinstance(api.converse("hello there", conv_id), str)

    def test_get_stack_depth_alone_prevents_eviction(self, api, clock):
        """`get_stack_depth` reads `conversation_stacks` directly rather than going through
        `_get_current_fsm_conversation_id`, so it needs its own refresh."""
        conv_id, _ = api.start_conversation()
        clock.advance(1.0)
        assert api.get_stack_depth(conv_id) == 1
        clock.advance(MAX_IDLE / 5)

        assert api.cleanup_stale_conversations(MAX_IDLE) == []
        assert conv_id in api.list_active_conversations()

    def test_pop_fsm_alone_prevents_eviction(self, api, clock):
        """`pop_fsm`'s opening block is the other direct `conversation_stacks` reader."""
        conv_id, _ = api.start_conversation()
        api.push_fsm(conv_id, _make_fsm("sub1"))
        clock.advance(1.0)
        api.pop_fsm(conv_id)
        clock.advance(MAX_IDLE / 5)

        assert api.cleanup_stale_conversations(MAX_IDLE) == []
        assert conv_id in api.list_active_conversations()


class TestStaleCleanupStillEvicts:
    """SC-4: the inverse. Without these, a fix of `return []` would pass every test above."""

    def test_genuinely_idle_conversation_is_still_evicted(self, api, clock):
        conv_id, _ = api.start_conversation()
        clock.advance(1.0)  # no API activity whatsoever

        assert api.cleanup_stale_conversations(MAX_IDLE) == [conv_id]
        assert conv_id not in api.list_active_conversations()
        with pytest.raises(ValueError, match="Unknown conversation ID"):
            api.converse("hello there", conv_id)

    def test_only_the_idle_conversation_of_two_is_evicted(self, api, clock):
        """Eviction stays per-conversation: refreshing one must not shield the other."""
        active_id, _ = api.start_conversation()
        idle_id, _ = api.start_conversation()

        clock.advance(1.0)
        api.get_data(active_id)  # touches active_id only
        clock.advance(MAX_IDLE / 5)

        assert api.cleanup_stale_conversations(MAX_IDLE) == [idle_id]
        assert api.list_active_conversations() == [active_id]

    def test_activity_then_real_idleness_is_evicted(self, api, clock):
        """Activity buys exactly one idle window, not permanent immunity."""
        conv_id, _ = api.start_conversation()
        clock.advance(1.0)
        api.get_data(conv_id)

        assert api.cleanup_stale_conversations(MAX_IDLE) == []
        clock.advance(1.0)
        assert api.cleanup_stale_conversations(MAX_IDLE) == [conv_id]


class TestStaleCleanupBookkeeping:
    """Edge cases the refresh must not break."""

    def test_refresh_keys_by_root_id_and_creates_no_extra_entries(self, api, clock):
        """The sub-FSM conversation id must NEVER appear in `_last_accessed` — cleanup
        iterates that dict and calls `end_conversation` on every key it holds."""
        conv_id, _ = api.start_conversation()
        api.push_fsm(conv_id, _make_fsm("sub1"))
        # Only meaningful once a sub-FSM is actually on the stack; before the push this
        # returns the root id itself.
        sub_id = api.get_sub_conversation_id(conv_id)
        assert sub_id != conv_id
        api.get_data(conv_id)
        api.update_context(conv_id, {"user_name": "ada"})

        with api._stack_lock:
            tracked = set(api._last_accessed)
        assert tracked == {conv_id}
        assert sub_id not in tracked

    def test_unknown_conversation_id_creates_no_entry(self, api, clock):
        """Refreshing must not resurrect eviction bookkeeping for a nonexistent id."""
        conv_id, _ = api.start_conversation()

        with pytest.raises(ValueError, match="Unknown conversation ID"):
            api.get_stack_depth("no-such-conversation")
        with pytest.raises(ValueError, match="Unknown conversation ID"):
            api.get_data("no-such-conversation")

        with api._stack_lock:
            assert set(api._last_accessed) == {conv_id}

    def test_ended_conversation_is_untracked(self, api, clock):
        """`end_conversation` resolves the current FSM (which refreshes) before popping the
        entry — the pop must still win, or a dead id would linger in the eviction map."""
        conv_id, _ = api.start_conversation()
        api.end_conversation(conv_id)

        with api._stack_lock:
            assert conv_id not in api._last_accessed

    def test_cleanup_uses_the_same_clock_as_the_refresh(self, api, clock):
        """Mixing `time.time()` into either side would silently corrupt the arithmetic: the
        two clocks differ by many orders of magnitude, so eviction would become all-or-nothing.
        Pin that a refresh writes the monotonic clock the cleanup sweep reads."""
        conv_id, _ = api.start_conversation()
        clock.advance(1.0)
        api.get_data(conv_id)

        with api._stack_lock:
            assert api._last_accessed[conv_id] == clock.now
