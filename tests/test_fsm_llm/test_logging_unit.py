"""Unit tests for fsm_llm.logging module."""

from __future__ import annotations

import json

import pytest

from fsm_llm.engine.api.helpers import handle_conversation_errors
from fsm_llm.engine.fsm.helpers import with_conversation_context


class TestWithConversationContext:
    """Tests for the with_conversation_context decorator."""

    def test_injects_log_kwarg(self):
        """Decorator should inject a 'log' keyword argument."""
        received_log = None

        class FakeManager:
            @with_conversation_context
            def my_method(self, conversation_id, log=None):
                nonlocal received_log
                received_log = log
                return "ok"

        mgr = FakeManager()
        result = mgr.my_method("conv-123")
        assert result == "ok"
        assert received_log is not None

    def test_preserves_function_name(self):
        """Decorator should preserve the original function's name."""

        class FakeManager:
            @with_conversation_context
            def my_method(self, conversation_id, log=None):
                pass

        assert FakeManager.my_method.__name__ == "my_method"

    def test_passes_additional_args(self):
        """Decorator should forward extra args and kwargs."""
        received_args = None

        class FakeManager:
            @with_conversation_context
            def my_method(self, conversation_id, message, option=None, log=None):
                nonlocal received_args
                received_args = (message, option)
                return "ok"

        mgr = FakeManager()
        mgr.my_method("conv-123", "hello", option="fast")
        assert received_args == ("hello", "fast")


class TestHandleConversationErrors:
    """Tests for the handle_conversation_errors decorator."""

    def test_passes_through_on_success(self):
        """Successful calls should pass through unchanged."""

        class FakeAPI:
            @handle_conversation_errors
            def my_method(self, conversation_id):
                return "result"

        api = FakeAPI()
        assert api.my_method("conv-123") == "result"

    def test_reraises_value_error(self):
        """ValueError should be re-raised as-is."""

        class FakeAPI:
            @handle_conversation_errors
            def my_method(self, conversation_id):
                raise ValueError("not found")

        api = FakeAPI()
        with pytest.raises(ValueError, match="not found"):
            api.my_method("conv-123")

    def test_reraises_fsm_error(self):
        """FSMError should be re-raised as-is."""
        from fsm_llm.core.definitions import FSMError

        class FakeAPI:
            @handle_conversation_errors
            def my_method(self, conversation_id):
                raise FSMError("fsm issue")

        api = FakeAPI()
        with pytest.raises(FSMError, match="fsm issue"):
            api.my_method("conv-123")

    def test_wraps_generic_exception_as_fsm_error(self):
        """Other exceptions should be wrapped in FSMError."""
        from fsm_llm.core.definitions import FSMError

        class FakeAPI:
            @handle_conversation_errors
            def my_method(self, conversation_id):
                raise RuntimeError("something broke")

        api = FakeAPI()
        with pytest.raises(FSMError, match="something broke"):
            api.my_method("conv-123")

    def test_custom_error_message(self):
        """Custom error message should be used when provided."""
        from fsm_llm.core.definitions import FSMError

        class FakeAPI:
            @handle_conversation_errors("Custom failure")
            def my_method(self, conversation_id):
                raise RuntimeError("boom")

        api = FakeAPI()
        with pytest.raises(FSMError, match="Custom failure"):
            api.my_method("conv-123")

    def test_preserves_function_name(self):
        """Decorator should preserve the original function's name."""

        class FakeAPI:
            @handle_conversation_errors
            def my_method(self, conversation_id):
                pass

        assert FakeAPI.my_method.__name__ == "my_method"


class TestHandlerCriticalFlag:
    """Tests for the critical handler flag added in step 7."""

    def test_critical_handler_raises_in_continue_mode(self):
        """Critical handlers should raise even when handler_error_mode='continue'."""
        from fsm_llm.handlers import (
            BaseHandler,
            HandlerExecutionError,
            HandlerSystem,
            HandlerTiming,
        )

        class FailingHandler(BaseHandler):
            def __init__(self):
                super().__init__(name="Failing", critical=True)

            def should_execute(
                self, timing, current_state, target_state, context, updated_keys=None
            ):
                return True

            def execute(self, context):
                raise RuntimeError("critical failure")

        system = HandlerSystem(handler_error_mode="continue")
        system.register_handler(FailingHandler())

        with pytest.raises(HandlerExecutionError, match="critical failure"):
            system.execute_handlers(
                timing=HandlerTiming.PRE_PROCESSING,
                current_state="start",
                target_state=None,
                context={},
            )

    def test_non_critical_handler_continues_in_continue_mode(self):
        """Non-critical handlers should be swallowed in continue mode."""
        from fsm_llm.handlers import BaseHandler, HandlerSystem, HandlerTiming

        class FailingHandler(BaseHandler):
            def __init__(self):
                super().__init__(name="Failing", critical=False)

            def should_execute(
                self, timing, current_state, target_state, context, updated_keys=None
            ):
                return True

            def execute(self, context):
                raise RuntimeError("non-critical failure")

        system = HandlerSystem(handler_error_mode="continue")
        system.register_handler(FailingHandler())

        # Should not raise
        result = system.execute_handlers(
            timing=HandlerTiming.PRE_PROCESSING,
            current_state="start",
            target_state=None,
            context={},
        )
        # Result may be empty dict since handler failed
        assert isinstance(result, dict)


class TestWorkflowStatusTransitions:
    """Tests for workflow status transition validation."""

    def test_valid_transition_pending_to_running(self):
        """PENDING → RUNNING should succeed."""
        from fsm_llm.workflows.models import WorkflowInstance, WorkflowStatus

        instance = WorkflowInstance(
            instance_id="test",
            workflow_id="wf",
            current_step_id="start",
        )
        instance.update_status(WorkflowStatus.RUNNING)
        assert instance.status == WorkflowStatus.RUNNING

    def test_invalid_transition_completed_to_running(self):
        """COMPLETED → RUNNING should raise WorkflowStateError."""
        from fsm_llm.workflows.exceptions import WorkflowStateError
        from fsm_llm.workflows.models import WorkflowInstance, WorkflowStatus

        instance = WorkflowInstance(
            instance_id="test",
            workflow_id="wf",
            current_step_id="start",
        )
        instance.update_status(WorkflowStatus.RUNNING)
        instance.update_status(WorkflowStatus.COMPLETED)
        with pytest.raises(WorkflowStateError, match="Invalid status transition"):
            instance.update_status(WorkflowStatus.RUNNING)

    def test_same_status_is_allowed(self):
        """Setting the same status should not raise."""
        from fsm_llm.workflows.models import WorkflowInstance, WorkflowStatus

        instance = WorkflowInstance(
            instance_id="test",
            workflow_id="wf",
            current_step_id="start",
        )
        instance.update_status(WorkflowStatus.RUNNING)
        # Same status should not raise
        instance.update_status(WorkflowStatus.RUNNING)


class TestConversationLockCleanup:
    """Tests for conversation lock cleanup."""

    def test_prune_orphaned_locks(self):
        """prune_orphaned_locks should remove orphaned locks."""
        import threading
        from unittest.mock import MagicMock

        from fsm_llm.engine.fsm.manager import FSMManager

        mock_llm = MagicMock()
        manager = FSMManager(llm_interface=mock_llm, fsm_loader=lambda x: None)

        # Manually add orphaned locks (no matching instance)
        manager._conversation_locks["orphan1"] = threading.Lock()
        manager._conversation_locks["orphan2"] = threading.Lock()

        cleaned = manager.prune_orphaned_locks()
        assert set(cleaned) == {"orphan1", "orphan2"}
        assert "orphan1" not in manager._conversation_locks
        assert "orphan2" not in manager._conversation_locks

    def test_cleanup_preserves_active_locks(self):
        """prune_orphaned_locks should keep locks for active instances."""
        import threading
        from unittest.mock import MagicMock

        from fsm_llm.engine.fsm.manager import FSMManager

        mock_llm = MagicMock()
        manager = FSMManager(llm_interface=mock_llm, fsm_loader=lambda x: None)

        # Add a lock with a matching instance
        manager._conversation_locks["active"] = threading.Lock()
        manager.instances["active"] = MagicMock()

        cleaned = manager.prune_orphaned_locks()
        assert cleaned == []
        assert "active" in manager._conversation_locks


class TestJsonExtractionValidation:
    """Tests for JSON regex fallback validation."""

    def test_regex_returns_none_for_only_auxiliary_keys(self):
        """Regex fallback should return None if only auxiliary keys found."""
        from fsm_llm.llm.json_extraction import extract_json_from_text

        # Text that only has auxiliary keys, no meaningful ones
        text = 'Some text "status": "ok" more text'
        result = extract_json_from_text(text)
        # Should return None since no meaningful keys
        assert result is None

    def test_regex_returns_data_with_extracted_data(self):
        """Regex fallback should return data when extracted_data is found."""
        from fsm_llm.llm.json_extraction import extract_json_from_text

        # Use text where brace matching fails but regex can find extracted_data
        text = 'Here is the data "extracted_data": {name: Alice} end'
        result = extract_json_from_text(text)
        # The brace match finds {name: Alice} but json.loads fails on it
        # The regex pattern r'"extracted_data"\s*:\s*(\{[^}]*\})' should catch it
        assert result is not None
        assert "extracted_data" in result

    def test_regex_returns_data_with_selected_transition(self):
        """Regex fallback should return data when selected_transition is found."""
        from fsm_llm.llm.json_extraction import extract_json_from_text

        text = 'blah "selected_transition": "next_state" blah'
        result = extract_json_from_text(text)
        assert result is not None
        assert result["selected_transition"] == "next_state"


@pytest.fixture
def _clean_logging():
    """Snapshot and restore every piece of global logging state touched here."""
    from loguru import logger

    import fsm_llm.logging as log_module

    before_ids = list(log_module._library_handler_ids)
    before_streams = dict(log_module._stream_handler_ids)
    before_flag = log_module._file_handler_initialized
    log_module._file_handler_initialized = False
    added: list[int] = []
    yield added
    for hid in added:
        try:
            logger.remove(hid)
        except ValueError:
            pass
    for hid in list(log_module._library_handler_ids):
        if hid not in before_ids:
            try:
                logger.remove(hid)
            except ValueError:
                pass
    log_module._library_handler_ids[:] = before_ids
    log_module._stream_handler_ids.clear()
    log_module._stream_handler_ids.update(before_streams)
    log_module._file_handler_initialized = before_flag


class TestJsonFileSink:
    """F-HM-02: the JSON file sink writes one valid JSON object per line."""

    def test_markup_like_text_and_exception_stay_valid_jsonl(
        self, tmp_path, capfd, _clean_logging
    ):
        from loguru import logger

        from fsm_llm.logging import setup_logging

        _clean_logging.append(
            setup_logging(
                sink="file", format="json", level="INFO", log_dir=str(tmp_path)
            )
        )

        logger.info("tag <current_context> inside")
        logger.info("backslash \\<x> and <red>not a colour</red>")
        (lambda: logger.info("from a lambda"))()
        try:
            raise RuntimeError("boom <b>")
        except RuntimeError:
            logger.exception("boom")
        logger.complete()

        files = list(tmp_path.iterdir())
        assert files
        lines = [
            line
            for f in files
            for line in f.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        entries = [json.loads(line) for line in lines]
        messages = [e["message"] for e in entries]
        assert "tag <current_context> inside" in messages
        assert "backslash \\<x> and <red>not a colour</red>" in messages
        assert "from a lambda" in messages
        assert "boom" in messages
        lambda_entry = next(e for e in entries if e["message"] == "from a lambda")
        assert lambda_entry["function"] == "<lambda>"
        boom = next(e for e in entries if e["message"] == "boom")
        assert boom["exception_type"] == "RuntimeError"
        assert boom["exception_message"] == "boom <b>"
        assert not any("Traceback" in line for line in lines)
        assert "Logging error" not in capfd.readouterr().err
