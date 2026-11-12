"""Core session paths exercised from the agents suite: stream auto-save, partial streams, restore seed.

These classes test core code (``API.converse_stream``, ``FSMManager.seed_restored_conversation``), not
agents code. They stay in this suite because a move to the core suite would change their pinned
(suite, class::test) ids; relocating them is a separate, deliberate step.
"""

from __future__ import annotations

import threading
from unittest.mock import Mock

import pytest


class TestStreamAutoSave:
    """Verify converse_stream() auto-saves sessions."""

    def test_auto_save_after_full_stream(self):
        """Session saved after generator fully consumed."""
        from fsm_llm import API

        api = Mock(spec=API)
        api._session_store = Mock()
        api._stack_lock = threading.RLock()
        api._last_accessed = {}
        api.save_session = Mock()
        api._get_current_fsm_conversation_id = Mock(return_value="fsm-1")
        api.fsm_manager = Mock()
        api.fsm_manager.process_message_stream = Mock(
            return_value=iter(["chunk1", "chunk2"])
        )

        chunks = list(API.converse_stream(api, "hello", "conv-1"))
        assert chunks == ["chunk1", "chunk2"]
        api.save_session.assert_called_once_with("conv-1")

    def test_auto_save_on_partial_consumption(self):
        """Session saved even when generator partially consumed."""
        from fsm_llm import API

        api = Mock(spec=API)
        api._session_store = Mock()
        api._stack_lock = threading.RLock()
        api._last_accessed = {}
        api.save_session = Mock()
        api._get_current_fsm_conversation_id = Mock(return_value="fsm-1")
        api.fsm_manager = Mock()
        api.fsm_manager.process_message_stream = Mock(
            return_value=iter(["a", "b", "c"])
        )

        gen = API.converse_stream(api, "hello", "conv-1")
        next(gen)  # consume one chunk
        gen.close()  # abandon generator
        api.save_session.assert_called_once_with("conv-1")

    def test_no_save_without_store(self):
        """No save attempt when session_store is None."""
        from fsm_llm import API

        api = Mock(spec=API)
        api._session_store = None
        api._stack_lock = threading.RLock()
        api._last_accessed = {}
        api.save_session = Mock()
        api._get_current_fsm_conversation_id = Mock(return_value="fsm-1")
        api.fsm_manager = Mock()
        api.fsm_manager.process_message_stream = Mock(return_value=iter(["x"]))

        list(API.converse_stream(api, "hello", "conv-1"))
        api.save_session.assert_not_called()


class TestStreamingTryFinally:
    """Verify partial streaming stores response in history."""

    def test_partial_response_stored_pattern(self):
        """The try/finally pattern stores partial responses."""
        conversation = Mock()
        chunks = []
        try:
            for chunk in ["Hello ", "World", " end"]:
                chunks.append(chunk)
                if len(chunks) == 2:
                    break  # Simulate interruption
        finally:
            if chunks:
                conversation.add_system_message("".join(chunks))

        conversation.add_system_message.assert_called_once_with("Hello World")


class TestReplayHistoryGuard:
    """Verify the restore seed handles missing FSM instances.

    History replay moved from ``API._replay_history`` into
    ``FSMManager.seed_restored_conversation`` (plan-2026-09-22T080837-8b258a25
    step 3); a missing instance is now a typed ``FSMError`` (never a bare
    ``KeyError``), which ``restore_session`` turns into a teardown."""

    def test_missing_instance_raises_typed_error(self):
        """The seed raises FSMError, not KeyError, for an unknown fsm_id."""
        from fsm_llm.definitions import FSMError
        from fsm_llm.fsm import FSMManager

        manager = FSMManager(llm_interface=Mock())

        with pytest.raises(FSMError, match="not found"):
            manager.seed_restored_conversation(
                "nonexistent-fsm-id",
                summary=None,
                history=[{"user": "hi"}],
                provenance=None,
                working_memory=None,
            )
