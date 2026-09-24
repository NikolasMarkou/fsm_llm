"""Deterministic same-conversation turn guard (two threads, one conversation).

Thread 1 enters a turn and is held inside the LLM call on a ``threading.Event``;
thread 2 then calls ``converse`` on the same conversation and must be refused
with ``FSMError`` ("already being processed") instead of running or waiting.
Synchronisation is by events and bounded joins only (no sleeps).

Mutation-checked: with ``FSMManager._enter_turn``'s ``_active_turns`` check
removed and its non-blocking ``conv_lock`` acquire made blocking, this test fails
(thread 2 waits for thread 1 instead of being refused).
"""

from __future__ import annotations

import threading

from fsm_llm import API
from fsm_llm.definitions import FSMError, ResponseGenerationResponse
from tests.conftest import MockLLM2Interface

_WAIT = 5.0  # bound on every wait/join; a healthy run never gets near it


def _fsm() -> dict:
    return {
        "name": "TurnGuard",
        "description": "Single chat state for the turn-guard test",
        "initial_state": "chat",
        "states": {
            "chat": {
                "id": "chat",
                "description": "Chat",
                "purpose": "Answer the user",
                "response_instructions": "Reply briefly",
                "transitions": [
                    {
                        "target_state": "done",
                        "description": "Never taken in this test",
                        "conditions": [
                            {
                                "description": "never",
                                "logic": {"==": [1, 2]},
                            }
                        ],
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


class _BlockingLLM(MockLLM2Interface):
    """Mock LLM whose first armed ``generate_response`` blocks on ``release``.

    ``entered`` is set from inside that call, so the test knows the calling
    thread is provably inside its turn. Later calls return at once.
    """

    def __init__(self) -> None:
        super().__init__(response_text="ok")
        self.armed = False
        self.entered = threading.Event()
        self.release = threading.Event()
        self._blocked_once = False
        self._guard = threading.Lock()

    def generate_response(self, request) -> ResponseGenerationResponse:
        with self._guard:
            block = self.armed and not self._blocked_once
            if block:
                self._blocked_once = True
        if block:
            self.entered.set()
            if not self.release.wait(timeout=_WAIT * 2):
                raise RuntimeError("release event never set")
        return super().generate_response(request)


class TestSameConversationTurnGuard:
    def test_second_thread_is_refused_while_first_is_inside_llm_call(self):
        llm = _BlockingLLM()
        api = API.from_definition(_fsm(), llm_interface=llm)
        try:
            conv_id, _ = api.start_conversation()
            llm.armed = True

            outcomes: dict[str, object] = {}

            def worker(name: str, msg: str) -> None:
                try:
                    outcomes[name] = api.converse(msg, conv_id)
                except BaseException as e:  # record, assert in the main thread
                    outcomes[name] = e

            t1 = threading.Thread(target=worker, args=("t1", "hello"), daemon=True)
            t2 = threading.Thread(target=worker, args=("t2", "hi"), daemon=True)
            try:
                t1.start()
                assert llm.entered.wait(timeout=_WAIT), (
                    "thread 1 never reached the LLM call"
                )
                t2.start()
                t2.join(timeout=_WAIT)
                t2_finished_while_t1_held = not t2.is_alive()
            finally:
                llm.release.set()
            t1.join(timeout=_WAIT)
            t2.join(timeout=_WAIT)
            assert not t1.is_alive() and not t2.is_alive(), outcomes

            assert t2_finished_while_t1_held, (
                "thread 2 waited for thread 1 instead of being refused"
            )
            assert outcomes["t1"] == "ok", outcomes
            err = outcomes["t2"]
            assert isinstance(err, FSMError), outcomes
            assert "already being processed" in str(err)

            successes = [v for v in outcomes.values() if v == "ok"]
            refused = [
                v
                for v in outcomes.values()
                if isinstance(v, FSMError) and "already being processed" in str(v)
            ]
            assert len(successes) == 1 and len(refused) == 1, outcomes

            # the refused turn left no trace and the conversation is usable again
            users = [
                m["user"] for m in api.get_conversation_history(conv_id) if "user" in m
            ]
            assert users == ["hello"]
            assert api.converse("again", conv_id) == "ok"
        finally:
            llm.release.set()
            api.close()
