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

import pytest

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
