"""
Unit tests for the MessagePipeline class.

Tests cover: pipeline creation, get_state resolution, execute_handlers context
management, generate_initial_response, full 2-pass process(), and
_clean_empty_context_keys delegation.
"""

from unittest.mock import MagicMock

import pytest


def configure_mock_extract_field(mock_llm, mock_data=None):
    """Configure a mock LLM with extract_field support."""
    from fsm_llm.definitions import FieldExtractionResponse

    data = mock_data or {}

    def _side_effect(request):
        value = data.get(request.field_name)
        return FieldExtractionResponse(
            field_name=request.field_name,
            value=value,
            confidence=1.0 if value is not None else 0.0,
            reasoning="Mock field extraction",
            is_valid=value is not None,
        )

    mock_llm.extract_field.side_effect = _side_effect
    return mock_llm


from fsm_llm.definitions import (
    DataExtractionResponse,
    FSMContext,
    FSMDefinition,
    FSMInstance,
    ResponseGenerationResponse,
    State,
    StateNotFoundError,
    Transition,
)
from fsm_llm.handlers import BaseHandler, HandlerSystem, HandlerTiming
from fsm_llm.llm import LLMInterface
from fsm_llm.pipeline import MessagePipeline
from fsm_llm.prompts import (
    DataExtractionPromptBuilder,
    ResponseGenerationPromptBuilder,
)
from fsm_llm.transition_evaluator import TransitionEvaluator

# ── Helpers ───────────────────────────────────────────────────


def _make_state(state_id, transitions=None, purpose="Test purpose"):
    """Create a minimal State object."""
    return State(
        id=state_id,
        description=f"{state_id} description",
        purpose=purpose,
        transitions=transitions or [],
    )


def _make_fsm_definition(states_dict, initial_state="start"):
    """Create a minimal FSMDefinition from a dict of {id: State}."""
    return FSMDefinition(
        name="test_fsm",
        description="Test FSM for pipeline tests",
        initial_state=initial_state,
        states=states_dict,
    )


def _make_instance(fsm_id="test_fsm", current_state="start", context_data=None):
    """Create a minimal FSMInstance."""
    ctx = FSMContext(data=context_data or {})
    return FSMInstance(
        fsm_id=fsm_id,
        current_state=current_state,
        context=ctx,
    )


def _make_mock_llm():
    """Create a mock LLM interface with default responses."""
    llm = MagicMock(spec=LLMInterface)
    llm.extract_data.return_value = DataExtractionResponse(
        extracted_data={},
        confidence=1.0,
        reasoning="mock extraction",
    )
    llm.generate_response.return_value = ResponseGenerationResponse(
        message="Hello from mock LLM",
        message_type="response",
        reasoning="mock response",
    )
    llm.decide_transition.side_effect = NotImplementedError(
        "decide_transition is deprecated"
    )
    configure_mock_extract_field(llm)
    return llm


def _make_pipeline(
    fsm_def=None,
    llm=None,
    handler_system=None,
    error_mode="continue",
):
    """Create a MessagePipeline with defaults wired up."""
    if fsm_def is None:
        fsm_def = _make_fsm_definition(
            {
                "start": _make_state(
                    "start",
                    transitions=[
                        Transition(
                            target_state="end",
                            description="Go to end",
                            priority=100,
                        )
                    ],
                ),
                "end": _make_state("end"),
            }
        )

    if llm is None:
        llm = _make_mock_llm()

    if handler_system is None:
        handler_system = HandlerSystem(error_mode=error_mode)

    # Real prompt builders (they only build strings)
    dep = DataExtractionPromptBuilder()
    rgp = ResponseGenerationPromptBuilder()
    te = TransitionEvaluator()

    def resolver(fsm_id):
        return fsm_def

    return MessagePipeline(
        llm_interface=llm,
        data_extraction_prompt_builder=dep,
        response_generation_prompt_builder=rgp,
        transition_evaluator=te,
        handler_system=handler_system,
        fsm_resolver=resolver,
    )


# ══════════════════════════════════════════════════════════════
# 1. Pipeline creation
# ══════════════════════════════════════════════════════════════


class TestPipelineCreation:
    """Tests for MessagePipeline construction."""

    def test_create_with_all_dependencies(self):
        pipeline = _make_pipeline()
        assert pipeline.llm_interface is not None
        assert pipeline.handler_system is not None
        assert pipeline.fsm_resolver is not None

    def test_stores_prompt_builders(self):
        pipeline = _make_pipeline()
        assert isinstance(
            pipeline.data_extraction_prompt_builder, DataExtractionPromptBuilder
        )
        assert isinstance(
            pipeline.response_generation_prompt_builder, ResponseGenerationPromptBuilder
        )

    def test_stores_transition_evaluator(self):
        pipeline = _make_pipeline()
        assert isinstance(pipeline.transition_evaluator, TransitionEvaluator)


# ══════════════════════════════════════════════════════════════
# 2. get_state()
# ══════════════════════════════════════════════════════════════


class TestGetState:
    """Tests for MessagePipeline.get_state()."""

    def test_resolves_valid_state(self):
        pipeline = _make_pipeline()
        instance = _make_instance()
        state = pipeline.get_state(instance)
        assert state.id == "start"

    def test_resolves_end_state(self):
        pipeline = _make_pipeline()
        instance = _make_instance(current_state="end")
        state = pipeline.get_state(instance)
        assert state.id == "end"

    def test_raises_state_not_found_for_invalid_state(self):
        pipeline = _make_pipeline()
        instance = _make_instance(current_state="nonexistent")
        with pytest.raises(StateNotFoundError, match="nonexistent"):
            pipeline.get_state(instance)

    def test_passes_conversation_id_for_logging(self):
        pipeline = _make_pipeline()
        instance = _make_instance()
        # Should not raise when conversation_id is provided
        state = pipeline.get_state(instance, conversation_id="conv-123")
        assert state.id == "start"

    def test_works_without_conversation_id(self):
        pipeline = _make_pipeline()
        instance = _make_instance()
        state = pipeline.get_state(instance, conversation_id=None)
        assert state.id == "start"


# ══════════════════════════════════════════════════════════════
# 3. execute_handlers()
# ══════════════════════════════════════════════════════════════


class _ContextCapturingHandler(BaseHandler):
    """Handler that records the context it received and returns updates."""

    def __init__(self, name="capturing", result=None):
        super().__init__(name=name, priority=100)
        self._result = result or {}
        self.captured_context = None

    def should_execute(
        self, timing, current_state, target_state, context, updated_keys=None
    ):
        return True

    def execute(self, context):
        self.captured_context = context.copy()
        return self._result.copy()


class TestExecuteHandlers:
    """Tests for MessagePipeline.execute_handlers()."""

    def test_deep_copies_context_before_passing_to_handlers(self):
        """Handler mutations should not affect the original instance context."""
        handler = _ContextCapturingHandler(result={"added_key": "value"})
        hs = HandlerSystem(error_mode="continue")
        hs.register_handler(handler)

        pipeline = _make_pipeline(handler_system=hs)
        instance = _make_instance(context_data={"original": "data"})

        pipeline.execute_handlers(
            instance, HandlerTiming.PRE_PROCESSING, "conv-1", current_state="start"
        )

        # Handler should have seen the original data
        assert handler.captured_context["original"] == "data"

    def test_merges_handler_results_into_instance_context(self):
        handler = _ContextCapturingHandler(result={"new_key": "new_value"})
        hs = HandlerSystem(error_mode="continue")
        hs.register_handler(handler)

        pipeline = _make_pipeline(handler_system=hs)
        instance = _make_instance(context_data={"existing": "data"})

        pipeline.execute_handlers(
            instance, HandlerTiming.PRE_PROCESSING, "conv-1", current_state="start"
        )

        assert instance.context.data["new_key"] == "new_value"
        assert instance.context.data["existing"] == "data"

    def test_none_value_deletes_key_from_context(self):
        """A handler returning None for a key requests deletion."""
        handler = _ContextCapturingHandler(result={"to_delete": None})
        hs = HandlerSystem(error_mode="continue")
        hs.register_handler(handler)

        pipeline = _make_pipeline(handler_system=hs)
        instance = _make_instance(
            context_data={"to_delete": "old_value", "keep": "yes"}
        )

        pipeline.execute_handlers(
            instance, HandlerTiming.PRE_PROCESSING, "conv-1", current_state="start"
        )

        assert "to_delete" not in instance.context.data
        assert instance.context.data["keep"] == "yes"

    def test_none_value_for_nonexistent_key_is_harmless(self):
        handler = _ContextCapturingHandler(result={"nonexistent": None})
        hs = HandlerSystem(error_mode="continue")
        hs.register_handler(handler)

        pipeline = _make_pipeline(handler_system=hs)
        instance = _make_instance()

        # Should not raise
        pipeline.execute_handlers(
            instance, HandlerTiming.PRE_PROCESSING, "conv-1", current_state="start"
        )
        assert "nonexistent" not in instance.context.data

    def test_error_context_merged_before_handler_call(self):
        """error_context dict should be visible to handlers."""
        handler = _ContextCapturingHandler()
        hs = HandlerSystem(error_mode="continue")
        hs.register_handler(handler)

        pipeline = _make_pipeline(handler_system=hs)
        instance = _make_instance(context_data={"base": "data"})

        pipeline.execute_handlers(
            instance,
            HandlerTiming.ERROR,
            "conv-1",
            current_state="start",
            error_context={"error_type": "TestError", "error_msg": "something failed"},
        )

        assert handler.captured_context["error_type"] == "TestError"
        assert handler.captured_context["error_msg"] == "something failed"
        assert handler.captured_context["base"] == "data"

    def test_handler_exception_swallowed_in_continue_mode(self):
        class FailHandler(BaseHandler):
            def should_execute(self, *args, **kwargs):
                return True

            def execute(self, context):
                raise RuntimeError("handler boom")

        hs = HandlerSystem(error_mode="continue")
        hs.register_handler(FailHandler(name="fail"))

        pipeline = _make_pipeline(handler_system=hs)
        instance = _make_instance()

        # Should not raise
        pipeline.execute_handlers(
            instance, HandlerTiming.PRE_PROCESSING, "conv-1", current_state="start"
        )

    def test_handler_exception_propagated_in_raise_mode(self):
        class FailHandler(BaseHandler):
            def should_execute(self, *args, **kwargs):
                return True

            def execute(self, context):
                raise RuntimeError("handler boom")

        hs = HandlerSystem(error_mode="raise")
        hs.register_handler(FailHandler(name="fail"))

        pipeline = _make_pipeline(handler_system=hs)
        instance = _make_instance()

        with pytest.raises(Exception):
            pipeline.execute_handlers(
                instance, HandlerTiming.PRE_PROCESSING, "conv-1", current_state="start"
            )

    def test_uses_instance_current_state_when_current_state_not_provided(self):
        handler = _ContextCapturingHandler()
        hs = HandlerSystem(error_mode="continue")
        hs.register_handler(handler)

        pipeline = _make_pipeline(handler_system=hs)
        instance = _make_instance(current_state="start")

        pipeline.execute_handlers(instance, HandlerTiming.PRE_PROCESSING, "conv-1")

        # Handler should have been called (indicating current_state was resolved)
        assert handler.captured_context is not None

    def test_passes_updated_keys_to_handler_system(self):
        """updated_keys should be forwarded to handler system."""
        # Use a handler that only fires on specific updated_keys
        h = _ContextCapturingHandler(result={"saw_update": True})

        # Override should_execute to check updated_keys
        def custom_should(
            timing, current_state, target_state, context, updated_keys=None
        ):
            if updated_keys and "user_name" in updated_keys:
                return True
            return False

        h.should_execute = custom_should

        hs = HandlerSystem(error_mode="continue")
        hs.register_handler(h)
        pipeline = _make_pipeline(handler_system=hs)
        instance = _make_instance()

        pipeline.execute_handlers(
            instance,
            HandlerTiming.CONTEXT_UPDATE,
            "conv-1",
            current_state="start",
            updated_keys={"user_name"},
        )

        assert instance.context.data.get("saw_update") is True


# ══════════════════════════════════════════════════════════════
# 4. generate_initial_response()
# ══════════════════════════════════════════════════════════════


class TestGenerateInitialResponse:
    """Tests for MessagePipeline.generate_initial_response()."""

    def test_calls_llm_generate_response(self):
        llm = _make_mock_llm()
        pipeline = _make_pipeline(llm=llm)
        instance = _make_instance()

        response = pipeline.generate_initial_response(instance, "conv-1")

        assert response == "Hello from mock LLM"
        llm.generate_response.assert_called_once()

    def test_stores_response_on_instance(self):
        llm = _make_mock_llm()
        pipeline = _make_pipeline(llm=llm)
        instance = _make_instance()

        pipeline.generate_initial_response(instance, "conv-1")

        assert instance.last_response_generation is not None
        assert instance.last_response_generation.message == "Hello from mock LLM"

    def test_adds_response_to_conversation_history(self):
        llm = _make_mock_llm()
        pipeline = _make_pipeline(llm=llm)
        instance = _make_instance()

        pipeline.generate_initial_response(instance, "conv-1")

        exchanges = instance.context.conversation.exchanges
        assert len(exchanges) == 1
        assert exchanges[0].get("system") == "Hello from mock LLM"

    def test_returns_response_message_string(self):
        llm = _make_mock_llm()
        llm.generate_response.return_value = ResponseGenerationResponse(
            message="Welcome to the conversation!",
            message_type="response",
        )
        pipeline = _make_pipeline(llm=llm)
        instance = _make_instance()

        result = pipeline.generate_initial_response(instance, "conv-1")

        assert isinstance(result, str)
        assert result == "Welcome to the conversation!"

    def test_raises_state_not_found_for_invalid_instance_state(self):
        pipeline = _make_pipeline()
        instance = _make_instance(current_state="nonexistent")

        with pytest.raises(StateNotFoundError):
            pipeline.generate_initial_response(instance, "conv-1")


# ══════════════════════════════════════════════════════════════
# 5. process() — full 2-pass pipeline
# ══════════════════════════════════════════════════════════════


class TestProcess:
    """Tests for MessagePipeline.process() — the full 2-pass orchestration."""

    def test_returns_response_string(self):
        llm = _make_mock_llm()
        pipeline = _make_pipeline(llm=llm)
        instance = _make_instance()

        result = pipeline.process(instance, "Hello", "conv-1")

        assert isinstance(result, str)
        assert result == "Hello from mock LLM"

    def test_calls_extract_data_then_generate_response(self):
        """Pipeline calls extract_field per field then generate_response."""
        llm = _make_mock_llm()
        configure_mock_extract_field(llm, {"user_name": "Alice"})
        fsm_def = _make_fsm_definition(
            {
                "start": State(
                    id="start",
                    description="start description",
                    purpose="Test purpose",
                    required_context_keys=["user_name"],
                    transitions=[
                        Transition(
                            target_state="end",
                            description="Go to end",
                            priority=100,
                        )
                    ],
                ),
                "end": _make_state("end"),
            }
        )
        pipeline = _make_pipeline(llm=llm, fsm_def=fsm_def)
        instance = _make_instance()

        pipeline.process(instance, "Hello", "conv-1")

        # Pass 1 calls extract_field, Pass 2 calls generate_response
        assert llm.extract_field.called
        llm.generate_response.assert_called_once()

    def test_pre_processing_handlers_run_before_extraction(self):
        """PRE_PROCESSING handlers should run before data extraction."""
        call_order = []

        class OrderTracker(BaseHandler):
            def __init__(self, label, timing):
                super().__init__(name=label, priority=100)
                self._timing = timing

            def should_execute(self, timing, *args, **kwargs):
                return timing == self._timing

            def execute(self, context):
                call_order.append(self.name)
                return {}

        hs = HandlerSystem(error_mode="continue")
        hs.register_handler(OrderTracker("pre_proc", HandlerTiming.PRE_PROCESSING))
        hs.register_handler(OrderTracker("post_proc", HandlerTiming.POST_PROCESSING))

        llm = _make_mock_llm()
        pipeline = _make_pipeline(llm=llm, handler_system=hs)
        instance = _make_instance()

        pipeline.process(instance, "Hello", "conv-1")

        assert "pre_proc" in call_order
        assert "post_proc" in call_order
        assert call_order.index("pre_proc") < call_order.index("post_proc")

    def test_extracted_data_updates_context(self):
        """Data extracted by LLM should be merged into instance context."""
        llm = _make_mock_llm()
        configure_mock_extract_field(llm, {"user_name": "Alice"})
        fsm_def = _make_fsm_definition(
            {
                "start": State(
                    id="start",
                    description="start description",
                    purpose="Test purpose",
                    required_context_keys=["user_name"],
                    transitions=[
                        Transition(
                            target_state="end",
                            description="Go to end",
                            priority=100,
                        )
                    ],
                ),
                "end": _make_state("end"),
            }
        )
        pipeline = _make_pipeline(llm=llm, fsm_def=fsm_def)
        instance = _make_instance()

        pipeline.process(instance, "My name is Alice", "conv-1")

        assert instance.context.data.get("user_name") == "Alice"

    def test_stores_extraction_response_on_instance(self):
        llm = _make_mock_llm()
        configure_mock_extract_field(llm, {"key": "value"})
        fsm_def = _make_fsm_definition(
            {
                "start": State(
                    id="start",
                    description="start description",
                    purpose="Test purpose",
                    required_context_keys=["key"],
                    transitions=[
                        Transition(
                            target_state="end",
                            description="Go to end",
                            priority=100,
                        )
                    ],
                ),
                "end": _make_state("end"),
            }
        )
        pipeline = _make_pipeline(llm=llm, fsm_def=fsm_def)
        instance = _make_instance()

        pipeline.process(instance, "test message", "conv-1")

        assert instance.last_extraction_response is not None

    def test_stores_response_generation_on_instance(self):
        llm = _make_mock_llm()
        pipeline = _make_pipeline(llm=llm)
        instance = _make_instance()

        pipeline.process(instance, "test message", "conv-1")

        assert instance.last_response_generation is not None
        assert instance.last_response_generation.message == "Hello from mock LLM"

    def test_no_extraction_data_skips_context_update(self):
        """When no fields are configured, context should not change."""
        llm = _make_mock_llm()
        pipeline = _make_pipeline(llm=llm)
        instance = _make_instance(context_data={"original": "data"})

        pipeline.process(instance, "Hello", "conv-1")

        # Original data should be preserved, no new keys added
        # (internal keys like _previous_state may not be added if no transition)
        assert instance.context.data.get("original") == "data"

    def test_process_with_terminal_state_no_transition(self):
        """In a terminal state (no transitions), no transition should occur."""
        llm = _make_mock_llm()
        pipeline = _make_pipeline(llm=llm)
        instance = _make_instance(current_state="end")

        result = pipeline.process(instance, "Hello", "conv-1")

        assert isinstance(result, str)
        assert instance.current_state == "end"  # Stays in terminal state
        # decide_transition should not be called for terminal state
        llm.decide_transition.assert_not_called()

    def test_process_response_added_to_conversation(self):
        llm = _make_mock_llm()
        pipeline = _make_pipeline(llm=llm)
        instance = _make_instance()

        pipeline.process(instance, "Hello", "conv-1")

        exchanges = instance.context.conversation.exchanges
        assert any("system" in e for e in exchanges)


# ══════════════════════════════════════════════════════════════
# 6. _clean_empty_context_keys()
# ══════════════════════════════════════════════════════════════


class TestCleanEmptyContextKeys:
    """Tests for MessagePipeline._clean_empty_context_keys static method."""

    def test_delegates_to_context_module(self):
        """Should call clean_context_keys from the context module."""
        data = {"user_name": "Alice", "_internal": "hidden"}
        result = MessagePipeline._clean_empty_context_keys(
            data=data, conversation_id="conv-1"
        )
        # Internal keys should be removed
        assert "_internal" not in result
        assert result["user_name"] == "Alice"

    def test_removes_none_values_by_default(self):
        data = {"name": "Alice", "age": None}
        result = MessagePipeline._clean_empty_context_keys(
            data=data, conversation_id="conv-1"
        )
        assert "age" not in result
        assert result["name"] == "Alice"

    def test_preserves_none_when_disabled(self):
        data = {"name": "Alice", "age": None}
        result = MessagePipeline._clean_empty_context_keys(
            data=data, conversation_id="conv-1", remove_none_values=False
        )
        assert result["age"] is None

    def test_removes_internal_prefix_keys(self):
        data = {
            "user_data": "keep",
            "_private": "remove",
            "system_info": "remove",
            "internal_state": "remove",
            "__dunder": "remove",
        }
        result = MessagePipeline._clean_empty_context_keys(
            data=data, conversation_id="conv-1"
        )
        assert "user_data" in result
        assert "_private" not in result
        assert "system_info" not in result
        assert "internal_state" not in result
        assert "__dunder" not in result

    def test_removes_empty_string_keys(self):
        data = {"": "empty_key_value", "valid": "data"}
        result = MessagePipeline._clean_empty_context_keys(
            data=data, conversation_id="conv-1"
        )
        assert "" not in result
        assert result["valid"] == "data"

    def test_preserves_empty_lists_and_strings(self):
        """Empty lists and empty string values are semantically meaningful."""
        data = {"allergies": [], "note": "", "name": "Alice"}
        result = MessagePipeline._clean_empty_context_keys(
            data=data, conversation_id="conv-1"
        )
        assert result["allergies"] == []
        assert result["note"] == ""
        assert result["name"] == "Alice"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
