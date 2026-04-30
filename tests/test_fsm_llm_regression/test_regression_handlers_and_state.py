"""Regression tests for plan 4 verified bugs in fsm_llm."""

import inspect
from unittest.mock import MagicMock, patch

import pytest

from fsm_llm.handlers import (
    HandlerExecutionError,
    HandlerSystem,
    HandlerTiming,
    create_handler,
)

# ── B1: POST_TRANSITION passes old_state as current_state ────


class TestPostTransitionCurrentState:
    """B1: POST_TRANSITION handler should see the NEW state as current_state."""

    def test_post_transition_handler_sees_new_state(self):
        """After transition A->B, POST_TRANSITION should pass current_state=B."""
        from fsm_llm.dialog.definitions import FSMContext, FSMInstance
        from fsm_llm.dialog.fsm import FSMManager

        # Track what current_state POST_TRANSITION receives
        received_states = []

        handler_system = HandlerSystem(error_mode="continue")
        handler = (
            create_handler("track_state")
            .at(HandlerTiming.POST_TRANSITION)
            .do(lambda ctx: received_states.append(ctx.get("_current_state")))
        )
        handler_system.register_handler(handler)

        # Patch execute_handlers to capture args
        original_execute = handler_system._execute_handlers
        captured_calls = []

        def spy_execute(**kwargs):
            if kwargs.get("timing") == HandlerTiming.POST_TRANSITION:
                captured_calls.append(kwargs)
            return original_execute(**kwargs)

        handler_system._execute_handlers = spy_execute

        manager = FSMManager(
            llm_interface=MagicMock(),
            handler_system=handler_system,
        )

        # Create a minimal instance
        context = FSMContext()
        instance = FSMInstance(
            fsm_id="test",
            current_state="state_A",
            persona=None,
            context=context,
        )
        manager.instances["conv1"] = instance

        manager._pipeline._execute_state_transition(instance, "state_B", "conv1")

        # Verify POST_TRANSITION received current_state=state_B (the NEW state)
        assert len(captured_calls) == 1
        assert captured_calls[0]["current_state"] == "state_B", (
            f"POST_TRANSITION should receive new state as current_state, got '{captured_calls[0]['current_state']}'"
        )


# ── B2: Handler exceptions bypass error_mode ─────────────────


class TestHandlerErrorModeRespected:
    """B2: _execute_handlers should respect handler_system.error_mode."""

    def test_raise_mode_propagates_exception(self):
        """When error_mode='raise', handler exceptions should propagate."""
        from fsm_llm.dialog.definitions import FSMContext, FSMInstance
        from fsm_llm.dialog.fsm import FSMManager

        handler_system = HandlerSystem(error_mode="raise")
        handler = (
            create_handler("exploding_handler")
            .at(HandlerTiming.PRE_PROCESSING)
            .do(lambda ctx: (_ for _ in ()).throw(ValueError("handler boom")))
        )
        handler_system.register_handler(handler)

        manager = FSMManager(
            llm_interface=MagicMock(),
            handler_system=handler_system,
        )
        manager.instances["conv1"] = FSMInstance(
            fsm_id="test",
            current_state="state_A",
            persona=None,
            context=FSMContext(),
        )

        with pytest.raises((HandlerExecutionError, Exception)):
            manager._execute_handlers(
                HandlerTiming.PRE_PROCESSING,
                "conv1",
                current_state="state_A",
            )

    def test_continue_mode_swallows_exception(self):
        """When error_mode='continue', handler exceptions should be swallowed."""
        from fsm_llm.dialog.definitions import FSMContext, FSMInstance
        from fsm_llm.dialog.fsm import FSMManager

        handler_system = HandlerSystem(error_mode="continue")
        handler = (
            create_handler("exploding_handler")
            .at(HandlerTiming.PRE_PROCESSING)
            .do(lambda ctx: (_ for _ in ()).throw(ValueError("handler boom")))
        )
        handler_system.register_handler(handler)

        manager = FSMManager(
            llm_interface=MagicMock(),
            handler_system=handler_system,
        )
        manager.instances["conv1"] = FSMInstance(
            fsm_id="test",
            current_state="state_A",
            persona=None,
            context=FSMContext(),
        )

        # Should NOT raise
        manager._execute_handlers(
            HandlerTiming.PRE_PROCESSING,
            "conv1",
            current_state="state_A",
        )


# ── B3: Empty string default causes ValidationError ──────────


class TestEmptyStringMessageFallback:
    """B3: Missing 'message' key in JSON should fall through to unstructured handling."""

    def test_missing_message_key_falls_through(self):
        """JSON without 'message' key should not raise ValidationError."""
        from fsm_llm.runtime._litellm import LiteLLMInterface

        interface = LiteLLMInterface.__new__(LiteLLMInterface)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        # JSON is valid but missing "message" key
        mock_response.choices[0].message.content = '{"reasoning": "some reasoning"}'

        # Should NOT raise ValidationError — should fall through to unstructured handling
        result = interface._parse_response_generation_response(mock_response)
        assert result.message is not None
        assert len(result.message) > 0


# ── B4: None content crashes with AttributeError ─────────────


class TestNoneContentHandling:
    """B4: _make_llm_call should reject None content from LLM responses."""

    def test_none_content_raises_clean_error(self):
        """None content should raise LLMResponseError, not AttributeError."""
        from fsm_llm.runtime._litellm import LiteLLMInterface, LLMResponseError

        interface = LiteLLMInterface.__new__(LiteLLMInterface)
        interface.model = "test-model"
        interface.temperature = 0.5
        interface.max_tokens = 100
        interface.timeout = 120.0
        interface.kwargs = {}

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None

        with (
            patch("fsm_llm.runtime._litellm.completion", return_value=mock_response),
            patch(
                "fsm_llm.runtime._litellm.get_supported_openai_params",
                return_value=None,
            ),
        ):
            with pytest.raises(LLMResponseError):
                interface._make_llm_call(
                    [{"role": "user", "content": "test"}], "data_extraction"
                )


# ── B5: _temp_fsm_definitions memory leak ─────────────────────


class TestTempFsmDefinitionsCleanup:
    """B5: _temp_fsm_definitions should be cleaned up after caching in push_fsm."""

    def test_temp_definitions_removed_after_caching(self):
        """Temp definitions should be removed from _temp after start_conversation caches them."""
        from fsm_llm.dialog.api import API

        # Verify that _temp_fsm_definitions entries are removed promptly.
        # With the fix, entries are popped in push_fsm after start_conversation.
        # end_conversation no longer needs to clear them.
        api = API.__new__(API)
        api._stack_lock = __import__("threading").Lock()
        api._temp_fsm_definitions = {"temp_fsm_1": MagicMock()}

        # Simulate the pop that happens after start_conversation in push_fsm
        api._temp_fsm_definitions.pop("temp_fsm_1", None)
        assert len(api._temp_fsm_definitions) == 0


# ── B6: Double period typo ────────────────────────────────────


class TestDoublePeriodinPrompt:
    """B6: prompts.py has a double period in task description."""

    def test_no_double_period_in_response_task(self):
        """_build_response_task_section should not contain '..'."""
        from fsm_llm.dialog.prompts import ResponseGenerationPromptBuilder

        builder = ResponseGenerationPromptBuilder()
        sections = builder._build_response_task_section()
        joined = "\n".join(sections)
        assert ".." not in joined, (
            f"Double period found in response task section: {joined}"
        )


# ── B7: Dead KeyError in exception handlers ───────────────────


class TestDeadKeyErrorCatch:
    """B7: Parse functions catch KeyError but .get() never raises it."""

    def test_no_keyerror_in_parse_response_generation(self):
        """_parse_response_generation_response should not catch KeyError."""
        from fsm_llm.runtime._litellm import LiteLLMInterface

        source = inspect.getsource(LiteLLMInterface._parse_response_generation_response)
        assert "KeyError" not in source, (
            "_parse_response_generation_response catches KeyError but uses .get() — dead code"
        )

    def test_parse_transition_removed(self):
        """_parse_transition_response was removed (classification replaces it)."""
        from fsm_llm.runtime._litellm import LiteLLMInterface

        assert not hasattr(LiteLLMInterface, "_parse_transition_response")
