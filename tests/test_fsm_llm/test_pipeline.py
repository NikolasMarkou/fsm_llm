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
    FieldExtractionConfig,
    FieldExtractionResponse,
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
    llm.generate_response.return_value = ResponseGenerationResponse(
        message="Hello from mock LLM",
        message_type="response",
        reasoning="mock response",
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

    def test_process_response_added_to_conversation(self):
        llm = _make_mock_llm()
        pipeline = _make_pipeline(llm=llm)
        instance = _make_instance()

        pipeline.process(instance, "Hello", "conv-1")

        exchanges = instance.context.conversation.exchanges
        assert any("system" in e for e in exchanges)


class TestProcessPass2FailureIsAtomic:
    """H2: a Pass-2 (response-generation) failure must not leave a half-applied
    turn. Pass 1 may commit a transition + extracted data; if Pass 2 raises,
    the caller-visible state must be restored to pre-turn so a retry evaluates
    against the state the user was actually responded from. See decisions.md D-012.
    """

    def _transition_fsm(self):
        from fsm_llm.definitions import TransitionCondition

        return _make_fsm_definition(
            {
                "start": State(
                    id="start",
                    description="start description",
                    purpose="Collect the name",
                    required_context_keys=["user_name"],
                    transitions=[
                        Transition(
                            target_state="end",
                            description="name captured",
                            priority=100,
                            conditions=[
                                TransitionCondition(
                                    description="user_name present",
                                    requires_context_keys=["user_name"],
                                )
                            ],
                        )
                    ],
                ),
                "end": _make_state("end"),
            }
        )

    def test_pass2_failure_reverts_transition_and_extraction(self):
        llm = _make_mock_llm()
        configure_mock_extract_field(llm, {"user_name": "Alice"})
        # Pass 2 blows up AFTER Pass 1 has committed the transition + extraction.
        llm.generate_response.side_effect = RuntimeError("pass-2 boom")

        pipeline = _make_pipeline(llm=llm, fsm_def=self._transition_fsm())
        instance = _make_instance()

        # The exception must propagate to the caller.
        with pytest.raises(RuntimeError, match="pass-2 boom"):
            pipeline.process(instance, "My name is Alice", "conv-1")

        # ...but the turn must be atomic: state and context reverted to pre-turn.
        assert instance.current_state == "start", (
            "Pass-2 failure left the Pass-1 transition committed"
        )
        assert "user_name" not in instance.context.data, (
            "Pass-2 failure left the Pass-1 extraction committed"
        )

    def test_pass2_failure_with_no_transition_leaves_state_intact(self):
        """No-transition / no-extraction turn: the restore is a correct no-op."""
        llm = _make_mock_llm()
        llm.generate_response.side_effect = RuntimeError("pass-2 boom")

        pipeline = _make_pipeline(llm=llm)  # no required_context_keys, no extraction
        instance = _make_instance(
            current_state="end", context_data={"original": "data"}
        )

        with pytest.raises(RuntimeError, match="pass-2 boom"):
            pipeline.process(instance, "Hello", "conv-1")

        assert instance.current_state == "end"
        assert instance.context.data.get("original") == "data"

    def test_pass2_failure_reverts_working_memory_and_metadata(self):
        """H2 completeness (D-013): `context.working_memory` and `context.metadata`
        are separate handler-mutable fields; a Pass-2 failure must revert them too,
        not just current_state + context.data. A handler that writes a working-memory
        buffer value and a metadata key during the turn must see both rolled back
        when Pass 2 raises."""
        from fsm_llm.memory import WorkingMemory

        instance = _make_instance()
        instance.context.working_memory = WorkingMemory()
        # Pre-turn baselines: neither the buffer value nor the metadata key exist.
        assert instance.context.working_memory.get("scratch", "turn_key") is None
        assert "turn_meta" not in instance.context.metadata

        captured = instance  # closure ref for the mutating handler

        class _WmMetaMutatingHandler(BaseHandler):
            """Writes the two separate FSMContext fields directly (the delta
            mechanism only reaches context.data)."""

            def __init__(self):
                super().__init__(name="wm_meta_mutator", priority=100)

            def should_execute(
                self, timing, current_state, target_state, context, updated_keys=None
            ):
                return True

            def execute(self, context):
                captured.context.working_memory.set(
                    "scratch", "turn_key", "turn_value"
                )
                captured.context.metadata["turn_meta"] = "turn_value"
                return {}

        hs = HandlerSystem(error_mode="continue")
        hs.register_handler(_WmMetaMutatingHandler())

        llm = _make_mock_llm()
        llm.generate_response.side_effect = RuntimeError("pass-2 boom")
        pipeline = _make_pipeline(llm=llm, handler_system=hs)

        with pytest.raises(RuntimeError, match="pass-2 boom"):
            pipeline.process(instance, "Hello", "conv-1")

        # The handler ran (mutated the live fields), but the Pass-2 failure must
        # have restored the pre-turn snapshot of BOTH fields.
        assert instance.context.working_memory.get("scratch", "turn_key") is None, (
            "Pass-2 failure left the working_memory mutation committed"
        )
        assert "turn_meta" not in instance.context.metadata, (
            "Pass-2 failure left the metadata mutation committed"
        )


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


# ══════════════════════════════════════════════════════════════
# Context Scope Tests
# ══════════════════════════════════════════════════════════════


class TestApplyContextScope:
    """Tests for _apply_context_scope static method."""

    def test_no_scope_returns_full_context(self):
        """When context_scope is None, all context is returned."""
        state = _make_state("start")
        context = {"name": "Alice", "email": "alice@test.com", "age": 30}
        result = MessagePipeline._apply_context_scope(context, state, "conv-1")
        assert result == context

    def test_scope_filters_to_read_keys(self):
        state = _make_state("start")
        state.context_scope = {"read_keys": ["name", "email"]}
        context = {"name": "Alice", "email": "alice@test.com", "age": 30}
        result = MessagePipeline._apply_context_scope(context, state, "conv-1")
        assert result == {"name": "Alice", "email": "alice@test.com"}

    def test_scope_missing_keys_ignored(self):
        state = _make_state("start")
        state.context_scope = {"read_keys": ["name", "missing_key"]}
        context = {"name": "Alice", "age": 30}
        result = MessagePipeline._apply_context_scope(context, state, "conv-1")
        assert result == {"name": "Alice"}

    def test_scope_empty_read_keys_returns_all(self):
        state = _make_state("start")
        state.context_scope = {"read_keys": []}
        context = {"name": "Alice"}
        result = MessagePipeline._apply_context_scope(context, state, "conv-1")
        assert result == context

    def test_scope_no_read_keys_in_dict_returns_all(self):
        state = _make_state("start")
        state.context_scope = {"write_keys": ["output"]}
        context = {"name": "Alice"}
        result = MessagePipeline._apply_context_scope(context, state, "conv-1")
        assert result == context

    def test_scope_with_response_generation(self):
        """Integration: verify context_scope is respected during response gen."""
        llm = _make_mock_llm()
        state = _make_state("start")
        state.context_scope = {"read_keys": ["visible_key"]}
        fsm_def = _make_fsm_definition({"start": state})
        pipeline = _make_pipeline(llm=llm, fsm_def=fsm_def)
        instance = _make_instance(
            context_data={"visible_key": "show", "hidden_key": "hide"}
        )

        pipeline.process(instance, "test", "conv-1")

        # Verify the LLM was called with scoped context
        gen_call = llm.generate_response.call_args
        request = gen_call[0][0] if gen_call[0] else gen_call.kwargs.get("request")
        assert "visible_key" in request.context
        assert "hidden_key" not in request.context


class TestContextScopeInFSMDefinition:
    """Test that context_scope works in FSM JSON definitions."""

    def test_state_accepts_context_scope(self):
        state = State(
            id="test",
            description="Test state",
            purpose="Testing",
            context_scope={"read_keys": ["name"], "write_keys": ["output"]},
        )
        assert state.context_scope == {"read_keys": ["name"], "write_keys": ["output"]}

    def test_state_context_scope_none_by_default(self):
        state = State(
            id="test",
            description="Test state",
            purpose="Testing",
        )
        assert state.context_scope is None

    def test_fsm_definition_with_scoped_states(self):
        """Full FSM definition with context_scope parses correctly."""
        fsm_dict = {
            "name": "test",
            "description": "Test FSM",
            "initial_state": "start",
            "states": {
                "start": {
                    "id": "start",
                    "description": "Start state",
                    "purpose": "Collect name",
                    "context_scope": {
                        "read_keys": ["greeting"],
                        "write_keys": ["name"],
                    },
                    "transitions": [],
                }
            },
        }
        fsm_def = FSMDefinition(**fsm_dict)
        assert fsm_def.states["start"].context_scope["read_keys"] == ["greeting"]


# ══════════════════════════════════════════════════════════════
# Additive bulk extraction (DECISION plan_2026-05-30_26c9510a/D-001)
# ══════════════════════════════════════════════════════════════


def _mock_bulk_llm_call(extracted: dict):
    """Build a MagicMock _make_llm_call returning a bulk-extraction response."""
    import json as _json

    msg = MagicMock()
    msg.content = _json.dumps({"extracted_data": extracted, "confidence": 0.9})
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return MagicMock(return_value=resp)


class TestAdditiveBulkExtraction:
    """A field named only in extraction_instructions (not in
    required_context_keys / transition conditions) must still be extracted
    via an additive bulk pass — covering the ~50% multi-field drop."""

    def _state_with_instruction_only_field(self):
        # policyholder_name has a config (required key); policy_number is
        # mentioned ONLY in extraction_instructions → no per-field config.
        return State(
            id="start",
            description="claim intake",
            purpose="collect claim info",
            required_context_keys=["policyholder_name"],
            extraction_instructions=(
                "Extract the policyholder_name and the policy_number."
            ),
            transitions=[],
        )

    def test_instruction_only_field_is_extracted(self):
        state = self._state_with_instruction_only_field()
        fsm_def = _make_fsm_definition({"start": state})
        llm = _make_mock_llm()
        configure_mock_extract_field(llm, {"policyholder_name": "David Wilson"})
        # Inject _make_llm_call (not on the LLMInterface ABC spec).
        llm._make_llm_call = _mock_bulk_llm_call({"policy_number": "GS-2024-88431"})

        pipeline = _make_pipeline(fsm_def=fsm_def, llm=llm)
        instance = _make_instance(current_state="start")

        resp = pipeline._execute_data_extraction(instance, "...", "conv-1")

        # Per-field config field captured AND instruction-only field merged in.
        assert resp.extracted_data.get("policyholder_name") == "David Wilson"
        assert resp.extracted_data.get("policy_number") == "GS-2024-88431"

    def test_bulk_does_not_overwrite_extracted_field(self):
        state = self._state_with_instruction_only_field()
        fsm_def = _make_fsm_definition({"start": state})
        llm = _make_mock_llm()
        configure_mock_extract_field(llm, {"policyholder_name": "David Wilson"})
        # Bulk tries to clobber the per-field value — must be ignored.
        llm._make_llm_call = _mock_bulk_llm_call(
            {"policyholder_name": "WRONG", "policy_number": "GS-1"}
        )

        pipeline = _make_pipeline(fsm_def=fsm_def, llm=llm)
        instance = _make_instance(current_state="start")

        resp = pipeline._execute_data_extraction(instance, "...", "conv-1")

        assert resp.extracted_data["policyholder_name"] == "David Wilson"
        assert resp.extracted_data["policy_number"] == "GS-1"

    def test_bulk_does_not_overwrite_handler_set_context(self):
        state = self._state_with_instruction_only_field()
        fsm_def = _make_fsm_definition({"start": state})
        llm = _make_mock_llm()
        configure_mock_extract_field(llm, {"policyholder_name": "David Wilson"})
        llm._make_llm_call = _mock_bulk_llm_call({"policy_number": "BULK"})

        pipeline = _make_pipeline(fsm_def=fsm_def, llm=llm)
        # policy_number already set (e.g. by a handler) → bulk must not touch it.
        instance = _make_instance(
            current_state="start", context_data={"policy_number": "PRESET"}
        )

        resp = pipeline._execute_data_extraction(instance, "...", "conv-1")

        assert resp.extracted_data.get("policy_number") != "BULK"

    def test_no_bulk_call_without_extraction_instructions(self):
        # No extraction_instructions → additive bulk pass must not fire.
        state = State(
            id="start",
            description="d",
            purpose="p",
            required_context_keys=["policyholder_name"],
            transitions=[],
        )
        fsm_def = _make_fsm_definition({"start": state})
        llm = _make_mock_llm()
        configure_mock_extract_field(llm, {"policyholder_name": "David Wilson"})
        llm._make_llm_call = _mock_bulk_llm_call({"policy_number": "SHOULD_NOT_APPEAR"})

        pipeline = _make_pipeline(fsm_def=fsm_def, llm=llm)
        instance = _make_instance(current_state="start")

        resp = pipeline._execute_data_extraction(instance, "...", "conv-1")

        assert "policy_number" not in resp.extracted_data
        llm._make_llm_call.assert_not_called()


class TestFieldTypeCoercionRejectsWrongTypes:
    """T6 / D-018: `list`/`dict` coercers must fail loudly like their 4 siblings.

    Both used to end in a bare `return v`, so a wrong-typed value passed through
    unconverted, the call site's `except (ValueError, TypeError, JSONDecodeError)`
    never fired, and the field was recorded `is_valid=True` with a wrong-typed value
    written into FSM context. Every assertion below drives the real
    `_validate_field_extraction` seam, never a coercer in isolation.
    """

    @staticmethod
    def _validate(field_type: str, value):
        """Drive the real validation seam the pipeline uses for every extracted field."""
        return MessagePipeline._validate_field_extraction(
            FieldExtractionResponse(
                field_name="items",
                value=value,
                confidence=1.0,
                reasoning="r",
                is_valid=True,
            ),
            FieldExtractionConfig(
                field_name="items",
                field_type=field_type,
                extraction_instructions="extract the items",
                confidence_threshold=0.0,
            ),
        )

    # --- the defect: wrong-typed input must be marked invalid, not accepted ---

    @pytest.mark.parametrize("bad", [5, 3.5, {"a": 1}, (1, 2), object()])
    def test_list_field_rejects_wrong_typed_values(self, bad):
        resp = self._validate("list", bad)
        assert resp.is_valid is False
        assert resp.validation_error
        assert "list" in resp.validation_error

    @pytest.mark.parametrize("bad", [5, 3.5, [1, 2], (1, 2), object()])
    def test_dict_field_rejects_wrong_typed_values(self, bad):
        resp = self._validate("dict", bad)
        assert resp.is_valid is False
        assert resp.validation_error
        assert "dict" in resp.validation_error

    def test_json_string_of_the_wrong_shape_is_still_rejected(self):
        """Pre-existing behavior, pinned: a parsed str of the wrong shape already raised."""
        assert self._validate("list", '{"a": 1}').is_valid is False
        assert self._validate("dict", "[1, 2]").is_valid is False

    def test_malformed_json_string_is_rejected(self):
        """JSONDecodeError is caught by the same already-wired except tuple."""
        assert self._validate("list", "[1, 2").is_valid is False

    # --- non-regression: valid inputs still coerce exactly as before ---

    def test_list_passthrough_and_json_parse_still_work(self):
        assert self._validate("list", [1, 2]).value == [1, 2]
        assert self._validate("list", "[1, 2]").value == [1, 2]

    def test_dict_passthrough_and_json_parse_still_work(self):
        assert self._validate("dict", {"a": 1}).value == {"a": 1}
        assert self._validate("dict", '{"a": 1}').value == {"a": 1}

    def test_none_never_reaches_the_coercer(self):
        """The call site returns early on a None value, so raising cannot break it."""
        resp = self._validate("list", None)
        assert resp.is_valid is True
        assert resp.value is None

    @pytest.mark.parametrize(
        ("field_type", "value", "expected"),
        [
            ("int", "5", 5),
            ("float", "2.5", 2.5),
            ("str", 5, "5"),
            ("bool", "yes", True),
            ("any", {"anything": 1}, {"anything": 1}),
        ],
    )
    def test_sibling_coercers_are_unchanged(self, field_type, value, expected):
        resp = self._validate(field_type, value)
        assert resp.is_valid is True
        assert resp.value == expected

    def test_str_and_bool_stay_total_by_design(self):
        """Not a third instance of the defect: str()/bool() cannot fail, so they
        have nothing to raise. Pinned so nobody 'propagates' the T6 fix to them."""
        assert self._validate("str", [1, 2]).is_valid is True
        assert self._validate("bool", [1, 2]).is_valid is True

    def test_int_and_float_already_failed_loudly(self):
        """The shape T6 brings list/dict into line with. Pinned as the reference."""
        assert self._validate("int", "abc").is_valid is False
        assert self._validate("float", {"a": 1}).is_valid is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
