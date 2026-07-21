"""Integration-seam tests for streaming conversation-history shape (T3 / D-015).

Drives ``API.converse_stream`` through the real seam
(``API`` -> ``FSMManager.process_message_stream`` -> ``MessagePipeline.process_stream``
-> ``MessagePipeline._stream_response_generation_pass``) with a mock LLM whose
stream fails part-way through.

The contract under test:

* a **backend stream error** must leave history in the same shape as the
  identical synchronous failure — no truncated assistant turn, and the user turn
  rolled back;
* **client abandonment** (``GeneratorExit``) must keep its existing, deliberately
  documented behaviour (``fsm.py:329-338``) — the partial reply the user actually
  saw IS persisted, and the user turn stays.

No ``time.sleep`` anywhere.  The mock stream is fully deterministic, and every
consumption loop carries an explicit iteration bound so a runaway generator
fails the test instead of hanging the gate.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from fsm_llm.api import API
from fsm_llm.definitions import (
    FieldExtractionResponse,
    FSMError,
    ResponseGenerationRequest,
    ResponseGenerationResponse,
)
from fsm_llm.llm import LiteLLMInterface, LLMInterface

# Hard bound on every generator consumption loop: the FSM under test yields at
# most 3 chunks, so anything beyond this is a runaway and must fail, not hang.
_MAX_CHUNKS = 50


def _streaming_fsm_dict() -> dict:
    """Minimal FSM with a non-terminal start state (streaming requires one)."""
    return {
        "name": "stream_history_test",
        "description": "FSM for streaming history-shape tests",
        "initial_state": "start",
        "persona": "Test bot",
        "states": {
            "start": {
                "id": "start",
                "description": "Start state",
                "purpose": "Begin conversation",
                "response_instructions": "Say hello",
                "transitions": [
                    {
                        "target_state": "end",
                        "description": "Move to end",
                        "conditions": [
                            {
                                "description": "Never fires",
                                "logic": {"==": [1, 2]},
                            }
                        ],
                    }
                ],
            },
            "end": {
                "id": "end",
                "description": "End state",
                "purpose": "End conversation",
                "response_instructions": "Say goodbye",
                "transitions": [],
            },
        },
    }


class _ScriptedStreamLLM(LLMInterface):
    """LLM whose stream yields ``chunks`` then optionally raises.

    ``fail_after`` is the number of chunks to yield before raising
    ``RuntimeError``; ``None`` means the stream completes normally.
    """

    def __init__(
        self,
        chunks: list[str],
        fail_after: int | None = None,
        greeting: str = "Greetings!",
    ) -> None:
        self.chunks = chunks
        self.fail_after = fail_after
        self.greeting = greeting
        self.stream_calls = 0

    def generate_response(
        self, request: ResponseGenerationRequest
    ) -> ResponseGenerationResponse:
        # Only used for the initial greeting (start_conversation).
        return ResponseGenerationResponse(
            message=self.greeting,
            message_type="response",
            reasoning="mock",
        )

    def generate_response_stream(
        self, request: ResponseGenerationRequest
    ) -> Iterator[str]:
        self.stream_calls += 1
        for i, chunk in enumerate(self.chunks):
            if self.fail_after is not None and i >= self.fail_after:
                raise RuntimeError("simulated LLM stream failure")
            yield chunk
        if self.fail_after is not None and self.fail_after >= len(self.chunks):
            raise RuntimeError("simulated LLM stream failure")

    def extract_field(self, request) -> FieldExtractionResponse:
        return FieldExtractionResponse(
            field_name=request.field_name,
            value=None,
            confidence=0.0,
            reasoning="mock",
            is_valid=False,
        )


def _make_api(llm: LLMInterface) -> tuple[API, str]:
    api = API(fsm_definition=_streaming_fsm_dict(), llm_interface=llm)
    conv_id, _ = api.start_conversation()
    return api, conv_id


def _drain(gen: Iterator[str], limit: int = _MAX_CHUNKS) -> list[str]:
    """Consume a generator with a hard iteration bound (no wall-clock timeout)."""
    out: list[str] = []
    for chunk in gen:
        out.append(chunk)
        assert len(out) <= limit, f"runaway stream: exceeded {limit} chunks"
    return out


class TestMidStreamBackendError:
    """SC-5: a backend error must not leave orphaned history entries."""

    def test_no_partial_assistant_turn_and_user_turn_rolled_back(self):
        llm = _ScriptedStreamLLM(["Hel", "lo ", "world"], fail_after=2)
        api, conv_id = _make_api(llm)

        before = list(api.get_conversation_history(conv_id))

        delivered: list[str] = []
        with pytest.raises(FSMError):
            for chunk in api.converse_stream("hello", conv_id):
                delivered.append(chunk)
                assert len(delivered) <= _MAX_CHUNKS

        # The caller genuinely saw a partial reply before the failure.
        assert delivered == ["Hel", "lo "]

        after = api.get_conversation_history(conv_id)

        # 1. No truncated assistant turn was persisted.
        assert not any("system" in e and e.get("system") == "Hello " for e in after), (
            f"truncated assistant reply persisted: {after}"
        )
        # 2. The user turn was rolled back too (this is the part that silently
        #    no-op'd while the partial system message masked it).
        assert not any("user" in e for e in after), f"orphaned user turn: {after}"
        # 3. History is byte-identical to its pre-call shape.
        assert after == before

    def test_error_before_any_chunk_leaves_history_untouched(self):
        """Edge case from the plan: zero chunks received before the error."""
        llm = _ScriptedStreamLLM(["Hel", "lo "], fail_after=0)
        api, conv_id = _make_api(llm)

        before = list(api.get_conversation_history(conv_id))

        with pytest.raises(FSMError):
            _drain(api.converse_stream("hello", conv_id))

        assert api.get_conversation_history(conv_id) == before

    def test_conversation_remains_usable_after_a_failed_stream(self):
        """The failed turn must not brick the conversation for a retry."""
        llm = _ScriptedStreamLLM(["Hel", "lo ", "world"], fail_after=2)
        api, conv_id = _make_api(llm)

        with pytest.raises(FSMError):
            _drain(api.converse_stream("hello", conv_id))

        # Retry with a healthy stream on the SAME conversation.
        llm.fail_after = None
        chunks = _drain(api.converse_stream("hello again", conv_id))
        assert "".join(chunks) == "Hello world"

        history = api.get_conversation_history(conv_id)
        assert {"user": "hello again"} in history
        assert {"system": "Hello world"} in history
        # The failed turn left nothing behind: exactly one user turn total.
        assert sum(1 for e in history if "user" in e) == 1


class TestClientAbandonment:
    """SC-6: the ``GeneratorExit`` contract (``fsm.py:329-338``) is unchanged.

    This is the guard against over-broadening the D-015 fix: if
    ``except BaseException`` had been written before ``except GeneratorExit``,
    the partial would no longer be persisted and this class would fail.
    """

    def test_partial_is_still_persisted_when_the_consumer_walks_away(self):
        llm = _ScriptedStreamLLM(["Hel", "lo ", "world"])
        api, conv_id = _make_api(llm)

        gen = api.converse_stream("hello", conv_id)
        delivered: list[str] = []
        for chunk in gen:
            delivered.append(chunk)
            if len(delivered) == 2:
                break
            assert len(delivered) <= _MAX_CHUNKS
        gen.close()  # deterministic GeneratorExit injection (no GC reliance)

        assert delivered == ["Hel", "lo "]

        history = api.get_conversation_history(conv_id)
        # The partial reply the user ACTUALLY SAW is preserved ...
        assert {"system": "Hello "} in history, f"abandonment partial lost: {history}"
        # ... and so is the user turn that produced it.
        assert {"user": "hello"} in history, f"user turn lost on abandonment: {history}"

    def test_abandonment_before_any_chunk_persists_nothing(self):
        """Zero chunks consumed: nothing to persist, but no crash either."""
        llm = _ScriptedStreamLLM(["Hel", "lo "])
        api, conv_id = _make_api(llm)

        before = list(api.get_conversation_history(conv_id))
        gen = api.converse_stream("hello", conv_id)
        gen.close()  # never iterated: generator body never ran

        assert api.get_conversation_history(conv_id) == before


class TestSuccessfulStreamUnchanged:
    """Non-regression: the happy path still records the full reply."""

    def test_full_reply_recorded(self):
        llm = _ScriptedStreamLLM(["Hel", "lo ", "world"])
        api, conv_id = _make_api(llm)

        chunks = _drain(api.converse_stream("hello", conv_id))
        assert "".join(chunks) == "Hello world"

        history = api.get_conversation_history(conv_id)
        assert {"user": "hello"} in history
        assert {"system": "Hello world"} in history


class _ThinkingStreamLLM(_ScriptedStreamLLM):
    """Pass 1 stays mocked; Pass 2 runs the REAL `LiteLLMInterface` stream code.

    ``self.chunks`` is reinterpreted as the sequence of ``delta.thinking`` values
    for a stream whose every ``delta.content`` is ``""`` — the shape that used to
    reach the user as a completely silent reply.
    """

    def generate_response_stream(self, request: ResponseGenerationRequest):
        self.stream_calls += 1
        raw = [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="", thinking=t))]
            )
            for t in self.chunks
        ]
        real = LiteLLMInterface(model="test-model")
        with (
            patch("fsm_llm.llm.completion", return_value=iter(raw)),
            patch("fsm_llm.llm.get_supported_openai_params", return_value=[]),
        ):
            yield from real.generate_response_stream(request)


class TestEmptyStreamIsNotPersisted:
    """SC-2 (F-02): an all-empty stream must never become an assistant turn.

    The criterion is the user-visible history shape, NOT "no exception raised":
    a blank turn recorded without error is the strictly WORSE outcome the old
    code produced.
    """

    def test_thinking_answer_is_recovered_and_no_blank_turn_is_stored(self):
        api, conv_id = _make_api(_ThinkingStreamLLM(["", "Hello there!"]))

        chunks = _drain(api.converse_stream("hello", conv_id))
        history = api.get_conversation_history(conv_id)

        assert "".join(chunks) == "Hello there!", f"thinking answer lost: {chunks}"
        assert {"system": ""} not in history, f"blank assistant turn stored: {history}"
        assert {"system": "Hello there!"} in history, f"answer not stored: {history}"

    def test_chunk_list_of_empty_strings_stores_nothing(self):
        """The pipeline guard alone: `['', '']` is a TRUTHY list but empty content.

        Uses a plain scripted LLM (no `llm.py` involvement) so this pins
        `pipeline.py`'s joined-content guard independently of the interface fix.
        """
        api, conv_id = _make_api(_ScriptedStreamLLM(["", ""]))

        chunks = _drain(api.converse_stream("hello", conv_id))
        history = api.get_conversation_history(conv_id)

        assert chunks == ["", ""]
        assert {"system": ""} not in history, f"blank assistant turn stored: {history}"

    def test_genuine_partial_on_abandonment_still_persists(self):
        """Regression guard on the UNTOUCHED contract (D-015).

        Empty-content suppression and `GeneratorExit` partial-persist are two
        distinct conditions; collapsing them would silently drop real partials.
        """
        api, conv_id = _make_api(_ScriptedStreamLLM(["", "Hel", "lo "]))

        gen = api.converse_stream("hello", conv_id)
        delivered: list[str] = []
        for chunk in gen:
            delivered.append(chunk)
            if len(delivered) == 2:
                break
            assert len(delivered) <= _MAX_CHUNKS
        gen.close()

        history = api.get_conversation_history(conv_id)
        assert {"system": "Hel"} in history, f"genuine partial lost: {history}"


class TestEagerPrologueOnCall:
    """G1: the streaming entry points must run their existence check and
    ``_last_accessed`` staleness refresh at CALL time — before the returned
    generator is ever iterated — so a created-but-deferred stream cannot skip
    validation or be reaped as stale mid-flight.

    Every assertion here is made on a generator that is created but NOT
    iterated (no ``next()``/``for``/``list()``), which is exactly what these
    tests defend: on the pre-split lazy-generator code the prologue only runs
    at first ``next()``, so all of these are RED before the wrapper/inner split.
    """

    def test_fsm_manager_stream_not_found_raises_at_call_time(self):
        """``FSMManager.process_message_stream`` on an unknown conversation must
        raise ``FSMError`` at CALL time, not deferred to first iteration."""
        api, _conv_id = _make_api(_ScriptedStreamLLM(["Hi"]))

        # Bare call, no iteration: on the pre-split code this returns a generator
        # and raises nothing until iterated (RED). After the split the eager
        # existence check fires at call time.
        with pytest.raises(FSMError):
            api.fsm_manager.process_message_stream("no-such-conversation", "hi")

    def test_api_converse_stream_unknown_conversation_raises_at_call_time(self):
        """``API.converse_stream`` resolves the conversation id (which validates
        existence) at CALL time — an unknown id raises before iteration."""
        api, _conv_id = _make_api(_ScriptedStreamLLM(["Hi"]))

        # ``_get_current_fsm_conversation_id`` raises ``ValueError`` for an
        # unknown id; the wrapper re-raises it unwrapped. Pre-split this is
        # deferred to first ``next()`` (RED).
        with pytest.raises(ValueError):
            api.converse_stream("hi", "no-such-conversation")

    def test_api_converse_stream_refreshes_last_accessed_at_call_time(self):
        """Creating (not iterating) the stream must refresh ``_last_accessed`` so
        ``cleanup_stale_conversations`` cannot reap an in-flight stream."""
        api, conv_id = _make_api(_ScriptedStreamLLM(["Hel", "lo"]))

        # Force the conversation to look stale, then create the generator WITHOUT
        # iterating it.  On the pre-split code the refresh is deferred to the
        # first ``next()``, so the stale timestamp survives (RED).
        stale = time.monotonic() - 10_000.0
        api._last_accessed[conv_id] = stale

        gen = api.converse_stream("hello", conv_id)
        try:
            assert api._last_accessed[conv_id] > stale, (
                "staleness refresh was deferred to iteration — an un-iterated "
                "stream can be reaped by cleanup_stale_conversations"
            )
        finally:
            gen.close()

    def test_uniterated_generator_does_not_hold_conv_lock(self):
        """Invariant (STOP-IF guard): the eager prologue must be LOCK-FREE.
        conv_lock acquisition stays LAZY inside the inner generator, so a stream
        that is created and abandoned WITHOUT iteration never leaks the lock."""
        api, conv_id = _make_api(_ScriptedStreamLLM(["Hel", "lo"]))
        fsm_conv_id = api._get_current_fsm_conversation_id(conv_id)
        conv_lock = api.fsm_manager._conversation_locks[fsm_conv_id]

        gen = api.converse_stream("hello", conv_id)
        try:
            # Another thread must be able to take the lock: a never-iterated
            # generator has NOT acquired conv_lock.
            acquired = conv_lock.acquire(blocking=False)
            assert acquired, (
                "an un-iterated stream generator is holding conv_lock — the "
                "eager wrapper must not acquire it (lock-leak on abandonment)"
            )
            conv_lock.release()
        finally:
            gen.close()

    def test_already_being_processed_guard_trips_lazily_on_iteration(self):
        """The 'already being processed' guard still fires — conv_lock is
        acquired LAZILY on first iteration, and if another thread holds it the
        first ``next()`` raises ``FSMError``."""
        api, conv_id = _make_api(_ScriptedStreamLLM(["Hel", "lo"]))
        fsm_conv_id = api._get_current_fsm_conversation_id(conv_id)
        conv_lock = api.fsm_manager._conversation_locks[fsm_conv_id]

        holding = threading.Event()
        release = threading.Event()

        def _hog():
            conv_lock.acquire()
            holding.set()
            release.wait(timeout=5.0)
            conv_lock.release()

        t = threading.Thread(target=_hog)
        t.start()
        try:
            assert holding.wait(timeout=5.0)
            gen = api.fsm_manager.process_message_stream(fsm_conv_id, "hi")
            # Call-time succeeded (existence check passed); the guard trips on
            # first iteration when the lazy conv_lock acquire fails.
            with pytest.raises(FSMError, match="already being processed"):
                next(gen)
        finally:
            release.set()
            t.join(timeout=5.0)
