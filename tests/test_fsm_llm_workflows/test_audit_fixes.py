"""
Tests verifying fixes for audit findings in fsm_llm_workflows.
Covers: F-001 (ParallelStep), F-002 (event race), F-004 (ConversationStep),
        F-007 (LLMProcessingStep template), F-010 (dead code removal).
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from fsm_llm_workflows.steps import (
    ParallelStep,
    AutoTransitionStep,
    LLMProcessingStep,
    ConversationStep,
)
from fsm_llm_workflows.engine import WorkflowEngine
from fsm_llm_workflows.models import WorkflowEvent


# ---------------------------------------------------------------------------
# F-001: ParallelStep must return failure when substeps fail
# ---------------------------------------------------------------------------

class TestParallelStepErrorHandling:
    """F-001: ParallelStep should report failure even without error_state."""

    def test_parallel_step_fails_when_substeps_fail_no_error_state(self):
        """ParallelStep with no error_state must still return failure if substeps fail."""
        failing_step = AutoTransitionStep(
            step_id="fail", name="Failing",
            next_state="done",
            action=lambda ctx: (_ for _ in ()).throw(ValueError("boom")),
        )
        ok_step = AutoTransitionStep(
            step_id="ok", name="OK", next_state="done",
        )

        parallel = ParallelStep(
            step_id="par", name="Parallel",
            steps=[ok_step, failing_step],
            next_state="next",
            error_state=None,  # No error_state configured
        )

        result = asyncio.get_event_loop().run_until_complete(parallel.execute({}))

        # Must be failure, NOT success
        assert result.success is False
        assert "error" in result.message.lower() or "error" in (result.error or "").lower()

    def test_parallel_step_fails_with_error_state(self):
        """ParallelStep with error_state transitions to it on failure."""
        failing_step = AutoTransitionStep(
            step_id="fail", name="Failing",
            next_state="done",
            action=lambda ctx: (_ for _ in ()).throw(RuntimeError("fail")),
        )

        parallel = ParallelStep(
            step_id="par", name="Parallel",
            steps=[failing_step],
            next_state="next",
            error_state="error_handler",
        )

        result = asyncio.get_event_loop().run_until_complete(parallel.execute({}))

        assert result.success is False
        assert result.next_state == "error_handler"

    def test_parallel_step_succeeds_when_all_ok(self):
        """ParallelStep returns success when all substeps succeed."""
        step_a = AutoTransitionStep(step_id="a", name="A", next_state="done")
        step_b = AutoTransitionStep(step_id="b", name="B", next_state="done")

        parallel = ParallelStep(
            step_id="par", name="Parallel",
            steps=[step_a, step_b],
            next_state="next",
        )

        result = asyncio.get_event_loop().run_until_complete(parallel.execute({}))

        assert result.success is True
        assert result.next_state == "next"


# ---------------------------------------------------------------------------
# F-002: Event listener race condition — pop instead of del
# ---------------------------------------------------------------------------

class TestEventListenerRaceCondition:
    """F-002: process_event must not crash if listener was already removed."""

    def test_pop_instead_of_del_in_process_event(self):
        """Verify engine uses .pop() for listener cleanup (race-safe)."""
        import inspect
        source = inspect.getsource(WorkflowEngine.process_event)
        # Must NOT use bare del on event_listeners
        assert "del self.event_listeners" not in source
        # Should use .pop() for safe removal
        assert ".pop(" in source

    def test_process_event_no_listeners(self):
        """process_event with no matching listeners should return empty list."""
        engine = WorkflowEngine()
        event = WorkflowEvent(event_type="nonexistent_event", payload={})
        result = asyncio.get_event_loop().run_until_complete(engine.process_event(event))
        assert result == []

    def test_process_event_empty_listener_dict(self):
        """process_event with empty listener dict for event type should return empty."""
        engine = WorkflowEngine()
        engine.event_listeners["test_event"] = {}
        event = WorkflowEvent(event_type="test_event", payload={})
        result = asyncio.get_event_loop().run_until_complete(engine.process_event(event))
        assert result == []


# ---------------------------------------------------------------------------
# F-004: ConversationStep resource leak
# ---------------------------------------------------------------------------

class TestConversationStepResourceCleanup:
    """F-004: end_conversation must be called even on exception."""

    def test_end_conversation_called_on_exception(self):
        """Verify end_conversation is called in finally block."""
        step = ConversationStep(
            step_id="conv", name="Conv",
            fsm_file="fake.json",
            success_state="done",
            auto_messages=["msg1"],
        )

        mock_api = MagicMock()
        mock_api.start_conversation.return_value = ("conv-1", "Hello")
        mock_api.has_conversation_ended.return_value = False
        mock_api.converse.side_effect = RuntimeError("LLM failed")
        mock_api.end_conversation = MagicMock()

        # API is imported inside execute(), so we patch it at the source module
        mock_api_class = MagicMock()
        mock_api_class.from_file.return_value = mock_api

        with patch.dict("sys.modules", {}):
            with patch("fsm_llm.API", mock_api_class):
                with patch("fsm_llm.api.API", mock_api_class):
                    result = asyncio.get_event_loop().run_until_complete(step.execute({}))

        # end_conversation MUST be called despite the exception
        mock_api.end_conversation.assert_called_once_with("conv-1")
        # Result should be a failure
        assert result.success is False


# ---------------------------------------------------------------------------
# F-007: LLMProcessingStep template KeyError
# ---------------------------------------------------------------------------

class TestLLMProcessingStepTemplateError:
    """F-007: Missing template variables should raise WorkflowStepError, not KeyError."""

    def test_missing_template_variable_raises_step_error(self):
        """Template with {missing_var} should raise WorkflowStepError."""
        mock_llm = MagicMock()

        step = LLMProcessingStep(
            step_id="llm", name="LLM Step",
            llm_interface=mock_llm,
            prompt_template="Hello {name}, your order {order_id} is ready",
            context_mapping={"name": "user_name"},  # order_id not mapped
            output_mapping={"result": ".*"},
            next_state="done",
        )

        # Context has user_name but template needs order_id too
        context = {"user_name": "Alice"}

        result = asyncio.get_event_loop().run_until_complete(step.execute(context))
        # Should fail gracefully instead of crashing with raw KeyError
        assert result.success is False
        assert "order_id" in result.error.lower() or "template" in result.error.lower()


# ---------------------------------------------------------------------------
# F-010: Dead conversation_map removed
# ---------------------------------------------------------------------------

class TestDeadCodeRemoved:
    """F-010: conversation_map should not exist on WorkflowEngine."""

    def test_no_conversation_map(self):
        """WorkflowEngine should not have conversation_map attribute."""
        engine = WorkflowEngine()
        assert not hasattr(engine, "conversation_map")
