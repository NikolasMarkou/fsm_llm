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


# TestProcess (legacy MessagePipeline.process) — deleted in S11 (D-S11-00).
# The legacy pipeline.process method was retired along with use_compiled.
# Coverage for the 2-pass orchestration is provided by the
# TestPipelineProcessCompiled* classes below and by the FSMManager-level
# integration tests in test_fsm.py / test_api.py.


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
        """Integration: verify context_scope is respected during response gen.

        Post-R9a (plan_2026-04-27_32652286): the terminal cohort path emits a
        ``Leaf`` and routes through ``oracle.invoke`` rather than the legacy
        ``CB_RESPOND`` callback. ``oracle.invoke`` constructs a synthetic
        ``ResponseGenerationRequest(system_prompt=..., user_message="")``
        whose ``context`` field is intentionally empty — the prompt is the
        substrate by then. So we assert the scope is honoured *inside the
        rendered system_prompt* (the actual oracle input), which is the
        behavioural contract that survives R9a/R9b/R9c.
        """
        llm = _make_mock_llm()
        state = _make_state("start")
        state.context_scope = {"read_keys": ["visible_key"]}
        fsm_def = _make_fsm_definition({"start": state})
        pipeline = _make_pipeline(llm=llm, fsm_def=fsm_def)
        instance = _make_instance(
            context_data={"visible_key": "show", "hidden_key": "hide"}
        )

        pipeline.process_compiled(instance, "test", "conv-1", tier=3)

        # Verify the LLM was called and the rendered system_prompt reflects
        # the scope (visible included; hidden excluded).
        gen_call = llm.generate_response.call_args
        assert gen_call is not None, "LLM was not called"
        request = gen_call[0][0] if gen_call[0] else gen_call.kwargs.get("request")
        assert "visible_key" in request.system_prompt
        assert "hidden_key" not in request.system_prompt


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
# S8-probe: process_compiled routes response-only FSMs through the
# compiled λ-term. Zero regression risk — legacy process untouched.
# ══════════════════════════════════════════════════════════════


def _make_response_only_fsm(initial_state: str = "hello") -> FSMDefinition:
    """FSM with a single state: no transitions, no extractions — the
    S8-probe cohort. Ref D-S8-01."""
    states = {
        initial_state: State(
            id=initial_state,
            description="respond-only state",
            purpose="greet",
            response_instructions="Say hi.",
            transitions=[],
        )
    }
    return FSMDefinition(
        name="probe_fsm",
        description="probe",
        initial_state=initial_state,
        states=states,
    )


class TestPipelineProcessCompiledProbe:
    """S8-probe: response-only FSMs route through the compiled λ-term."""

    def test_method_exists_and_returns_str(self) -> None:
        fsm = _make_response_only_fsm()
        pipeline = _make_pipeline(fsm_def=fsm)
        instance = _make_instance(current_state="hello")

        assert hasattr(pipeline, "process_compiled"), (
            "MessagePipeline must expose process_compiled"
        )
        result = pipeline.process_compiled(instance, "hi there", "conv-1")
        assert isinstance(result, str)

    def test_returns_expected_mock_response(self) -> None:
        fsm = _make_response_only_fsm()
        pipeline = _make_pipeline(fsm_def=fsm)
        instance = _make_instance(current_state="hello")

        result = pipeline.process_compiled(instance, "whatever", "conv-1")
        assert result == "Hello from mock LLM"

    # test_equivalence_with_legacy_process_single_state — deleted in S11
    # (D-S11-00). Legacy `pipeline.process` is gone; there is no longer a
    # second oracle to compare against. Correctness of `process_compiled`
    # is covered by the dispatch / branch / handler tests below.

    def test_dispatch_routes_to_current_state_branch(self) -> None:
        """Case scrutinee = instance.current_state. We bypass
        FSMDefinition's reachability validator (which otherwise
        orphans any multi-state response-only FSM — no transitions
        means no reach) via object.__setattr__, then verify dispatch
        fires for both states."""
        fsm = _make_response_only_fsm(initial_state="alpha")
        # Inject a second reachable-in-name-only state to exercise the
        # Case's multi-branch dispatch. FSMDefinition.validate already
        # ran; we mutate the frozen states dict in place.
        beta_state = State(
            id="beta",
            description="beta state",
            purpose="beta",
            response_instructions="Beta response.",
            transitions=[],
        )
        object.__setattr__(fsm, "states", {**fsm.states, "beta": beta_state})
        pipeline = _make_pipeline(fsm_def=fsm)

        inst_alpha = _make_instance(current_state="alpha")
        inst_beta = _make_instance(current_state="beta")

        r_alpha = pipeline.process_compiled(inst_alpha, "x", "c1")
        r_beta = pipeline.process_compiled(inst_beta, "x", "c2")

        assert r_alpha == "Hello from mock LLM"
        assert r_beta == "Hello from mock LLM"

    def test_cohort_guard_rejects_fsm_with_transitions(self) -> None:
        """FSM with any state having transitions → ValueError."""
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
        pipeline = _make_pipeline(fsm_def=fsm_def)
        instance = _make_instance(current_state="start")

        with pytest.raises(ValueError, match=r"cohort violation.*transitions"):
            pipeline.process_compiled(instance, "m", "c1")

    def test_cohort_guard_rejects_fsm_with_extractions(self) -> None:
        """FSM with extraction_instructions → ValueError."""
        start = State(
            id="start",
            description="d",
            purpose="p",
            response_instructions="r",
            extraction_instructions="extract stuff",
            transitions=[],
        )
        fsm_def = FSMDefinition(
            name="extract_fsm",
            description="extract test",
            initial_state="start",
            states={"start": start},
        )
        pipeline = _make_pipeline(fsm_def=fsm_def)
        instance = _make_instance(current_state="start")

        with pytest.raises(
            ValueError, match=r"cohort violation.*extraction_instructions"
        ):
            pipeline.process_compiled(instance, "m", "c1")

    def test_pre_and_post_processing_handlers_fire_once(self) -> None:
        """PRE_PROCESSING and POST_PROCESSING each fire exactly once per
        process_compiled call. Validates the cross-cutting handler
        wrapping around executor.run."""
        counts = {"pre": 0, "post": 0}

        class _TimingCounter(BaseHandler):
            """Counter that only runs at a specific timing point."""

            def __init__(self, target_timing: HandlerTiming, key: str) -> None:
                super().__init__(name=f"{key}_counter")
                self._target_timing = target_timing
                self._key = key

            def should_execute(
                self, timing, current_state, target_state, context, updated_keys=None
            ):
                return timing is self._target_timing

            def execute(self, context):
                counts[self._key] += 1
                return {}

        hs = HandlerSystem(error_mode="raise")
        hs.register_handler(_TimingCounter(HandlerTiming.PRE_PROCESSING, "pre"))
        hs.register_handler(_TimingCounter(HandlerTiming.POST_PROCESSING, "post"))

        fsm = _make_response_only_fsm()
        pipeline = _make_pipeline(fsm_def=fsm, handler_system=hs)
        instance = _make_instance(current_state="hello")

        pipeline.process_compiled(instance, "m", "c1")
        assert counts == {"pre": 1, "post": 1}

    def test_current_state_must_match_a_branch(self) -> None:
        """If instance.current_state is not a valid branch key, the Case
        dispatcher raises ASTConstructionError. This is analogous to
        StateNotFoundError semantics."""
        from fsm_llm.lam.errors import ASTConstructionError

        fsm = _make_response_only_fsm()
        pipeline = _make_pipeline(fsm_def=fsm)
        instance = _make_instance(current_state="missing_state")

        with pytest.raises(ASTConstructionError, match="case: scrutinee"):
            pipeline.process_compiled(instance, "m", "c1")


# ══════════════════════════════════════════════════════════════
# S8b T1 cohort: extractions-only FSMs route through the compiled
# λ-term with CB_EXTRACT / CB_FIELD_EXTRACT / CB_CLASS_EXTRACT all
# bound to the legacy dispatcher (first-to-fire runs, rest are
# no-ops). Verifies legacy-semantic equivalence.
# ══════════════════════════════════════════════════════════════


def _make_extract_only_fsm(
    with_extraction_instructions: bool = False,
    with_field_extractions: bool = False,
    with_classification_extractions: bool = False,
) -> FSMDefinition:
    """FSM with a single terminal state + optional extractions. No transitions."""
    from fsm_llm.definitions import (
        ClassificationExtractionConfig,
        FieldExtractionConfig,
        IntentDefinition,
    )

    extraction_instructions = (
        "Extract any user data." if with_extraction_instructions else None
    )
    field_extractions = (
        [
            FieldExtractionConfig(
                field_name="user_name",
                field_type="str",
                extraction_instructions="Extract the user's name.",
                required=False,
            )
        ]
        if with_field_extractions
        else []
    )
    classification_extractions = (
        [
            ClassificationExtractionConfig(
                field_name="user_mood",
                intents=[
                    IntentDefinition(name="happy", description="User is happy"),
                    IntentDefinition(name="sad", description="User is sad"),
                ],
                fallback_intent="happy",
                confidence_threshold=0.5,
            )
        ]
        if with_classification_extractions
        else []
    )
    state = State(
        id="hello",
        description="hello state",
        purpose="greet + extract",
        response_instructions="Respond warmly.",
        extraction_instructions=extraction_instructions,
        field_extractions=field_extractions,
        classification_extractions=classification_extractions,
        transitions=[],
    )
    return FSMDefinition(
        name="extract_fsm",
        description="extract test",
        initial_state="hello",
        states={"hello": state},
    )


class TestPipelineProcessCompiledExtractions:
    """S8b T1: extractions-only FSMs. CB_EXTRACT, CB_FIELD_EXTRACT,
    CB_CLASS_EXTRACT all share the dispatcher binding (first-to-fire
    runs the full dispatcher; subsequent are no-ops)."""

    def _run_compiled_tier1(self, pipeline, instance, msg="hello", conv="c1"):
        return pipeline.process_compiled(instance, msg, conv, tier=1)

    def test_tier1_with_extraction_instructions_fires_bulk(self) -> None:
        """T1.a — CB_EXTRACT fires for state with extraction_instructions."""
        fsm = _make_extract_only_fsm(with_extraction_instructions=True)
        pipeline = _make_pipeline(fsm_def=fsm)

        # Mock bulk extraction: return a small dict

        pipeline.llm_interface.generate_response.return_value = (
            ResponseGenerationResponse(
                message="Hello from mock LLM",
                message_type="response",
                reasoning="r",
            )
        )
        # patch _bulk_extract_from_instructions
        calls = {"n": 0}

        def _fake_bulk(*args, **kwargs):
            calls["n"] += 1
            return {"some_key": "some_value"}

        pipeline._bulk_extract_from_instructions = _fake_bulk  # type: ignore

        instance = _make_instance(current_state="hello")
        result = self._run_compiled_tier1(pipeline, instance)
        assert result == "Hello from mock LLM"
        assert calls["n"] == 1
        assert instance.context.data.get("some_key") == "some_value"

    def test_tier1_with_field_extractions_fires_field(self) -> None:
        """T1.b — CB_FIELD_EXTRACT fires for state with field_extractions."""
        fsm = _make_extract_only_fsm(with_field_extractions=True)
        llm = _make_mock_llm()
        configure_mock_extract_field(llm, {"user_name": "Alice"})
        pipeline = _make_pipeline(fsm_def=fsm, llm=llm)

        instance = _make_instance(current_state="hello")
        result = self._run_compiled_tier1(pipeline, instance, msg="I'm Alice")
        assert result == "Hello from mock LLM"
        assert instance.context.data.get("user_name") == "Alice"

    def test_tier1_with_classification_extractions(self) -> None:
        """T1.c — CB_CLASS_EXTRACT path does not crash at tier=1."""
        fsm = _make_extract_only_fsm(with_classification_extractions=True)
        pipeline = _make_pipeline(fsm_def=fsm)

        # Mock _execute_classification_extractions to avoid real classifier
        def _fake_class(*args, **kwargs):
            return {"user_mood": "happy"}

        pipeline._execute_classification_extractions = _fake_class  # type: ignore

        instance = _make_instance(current_state="hello")
        result = self._run_compiled_tier1(pipeline, instance)
        assert result == "Hello from mock LLM"
        assert instance.context.data.get("user_mood") == "happy"

    def test_tier1_rejects_state_with_transitions(self) -> None:
        """T1.d — cohort guard rejects transitions at tier=1."""
        fsm_def = _make_fsm_definition(
            {
                "start": _make_state(
                    "start",
                    transitions=[
                        Transition(target_state="end", description="d", priority=1)
                    ],
                ),
                "end": _make_state("end"),
            }
        )
        pipeline = _make_pipeline(fsm_def=fsm_def)
        instance = _make_instance(current_state="start")
        with pytest.raises(ValueError, match=r"has transitions"):
            pipeline.process_compiled(instance, "m", "c1", tier=1)

    def test_tier1_context_update_handler_fires_with_correct_keys(self) -> None:
        """T1.e — CONTEXT_UPDATE handler fires with the extracted-keys set."""
        fsm = _make_extract_only_fsm(with_field_extractions=True)
        llm = _make_mock_llm()
        configure_mock_extract_field(llm, {"user_name": "Bob"})

        seen_updates: list[set] = []

        class _CtxUpdateCounter(BaseHandler):
            def __init__(self):
                super().__init__(name="ctx_counter")
                self._last_updated_keys = None

            def should_execute(
                self, timing, current_state, target_state, context, updated_keys=None
            ):
                if timing is HandlerTiming.CONTEXT_UPDATE:
                    self._last_updated_keys = updated_keys
                    return True
                return False

            def execute(self, context):
                if self._last_updated_keys:
                    seen_updates.append(set(self._last_updated_keys))
                return {}

        hs = HandlerSystem(error_mode="raise")
        hs.register_handler(_CtxUpdateCounter())
        pipeline = _make_pipeline(fsm_def=fsm, llm=llm, handler_system=hs)

        instance = _make_instance(current_state="hello")
        pipeline.process_compiled(instance, "hi", "c1", tier=1)
        # Exactly one CONTEXT_UPDATE fire, with user_name in the set
        assert len(seen_updates) == 1
        assert "user_name" in seen_updates[0]

    def test_tier1_turn_state_accumulates_extraction_response(self) -> None:
        """T1.f — _TurnState.extraction_response is set; CB_RESPOND sees it."""
        fsm = _make_extract_only_fsm(with_field_extractions=True)
        llm = _make_mock_llm()
        configure_mock_extract_field(llm, {"user_name": "Carol"})
        pipeline = _make_pipeline(fsm_def=fsm, llm=llm)

        seen_requests: list = []
        original = pipeline._execute_response_generation_pass

        def _spy(inst, msg, extraction, transition_occurred, prev, conv):
            seen_requests.append(
                {
                    "extraction": extraction,
                    "transition_occurred": transition_occurred,
                    "previous_state": prev,
                }
            )
            return original(inst, msg, extraction, transition_occurred, prev, conv)

        pipeline._execute_response_generation_pass = _spy  # type: ignore

        instance = _make_instance(current_state="hello")
        pipeline.process_compiled(instance, "hi", "c1", tier=1)
        assert len(seen_requests) == 1
        extraction = seen_requests[0]["extraction"]
        assert extraction is not None
        assert extraction.extracted_data.get("user_name") == "Carol"
        assert seen_requests[0]["transition_occurred"] is False
        assert seen_requests[0]["previous_state"] is None

    # test_tier1_equivalence_with_legacy_process — deleted in S11
    # (D-S11-00). Legacy `pipeline.process` is gone.

    def test_tier1_multiple_extraction_slots_single_dispatch(self) -> None:
        """Even when compiler emits BOTH CB_EXTRACT and CB_FIELD_EXTRACT
        (state has both extraction_instructions and field_extractions),
        the dispatcher runs exactly once."""
        fsm = _make_extract_only_fsm(
            with_extraction_instructions=True, with_field_extractions=True
        )
        llm = _make_mock_llm()
        configure_mock_extract_field(llm, {"user_name": "Eve"})
        pipeline = _make_pipeline(fsm_def=fsm, llm=llm)

        dispatch_calls = {"n": 0}
        original = pipeline._execute_data_extraction

        def _spy(*args, **kwargs):
            dispatch_calls["n"] += 1
            return original(*args, **kwargs)

        pipeline._execute_data_extraction = _spy  # type: ignore

        instance = _make_instance(current_state="hello")
        pipeline.process_compiled(instance, "hi", "c1", tier=1)
        assert dispatch_calls["n"] == 1


# ══════════════════════════════════════════════════════════════
# S8b T2 cohort: FSMs with deterministic transitions. CB_EVAL_TRANSIT
# wired; CB_RESOLVE_AMBIG is sentinel (fails loud on AMBIGUOUS — D-S8b-02).
# Post-transition re-extraction fires for non-agent FSMs with a
# successful transition and missing fields in the new state.
# ══════════════════════════════════════════════════════════════


def _make_deterministic_transition_fsm() -> FSMDefinition:
    """FSM with a single deterministic transition from start → end.
    The condition fires when `has_name` is True in context."""
    return FSMDefinition(
        name="det_fsm",
        description="deterministic transition test",
        initial_state="start",
        states={
            "start": State(
                id="start",
                description="start",
                purpose="greet",
                response_instructions="Respond.",
                transitions=[
                    Transition(
                        target_state="end",
                        description="go to end when has_name",
                        priority=100,
                        conditions=[
                            {
                                "description": "has_name is truthy",
                                "requires_context_keys": ["has_name"],
                                "logic": {"==": [{"var": "has_name"}, True]},
                            }
                        ],
                    )
                ],
            ),
            "end": State(
                id="end",
                description="end",
                purpose="goodbye",
                response_instructions="Say goodbye.",
                transitions=[],
            ),
        },
    )


class TestPipelineProcessCompiledDeterministic:
    """S8b T2: CB_EVAL_TRANSIT wired. Deterministic/blocked transitions."""

    def test_tier2_deterministic_transition_advances(self) -> None:
        """T2.a — DETERMINISTIC → current_state mutated; 'advanced' branch."""
        fsm = _make_deterministic_transition_fsm()
        pipeline = _make_pipeline(fsm_def=fsm)
        # Seed the context so the transition condition fires
        instance = _make_instance(
            current_state="start", context_data={"has_name": True}
        )
        result = pipeline.process_compiled(instance, "hi", "c1", tier=2)
        assert result == "Hello from mock LLM"
        assert instance.current_state == "end"

    def test_tier2_blocked_transition_stays(self) -> None:
        """T2.b — no transition condition fires → 'blocked' branch, stays."""
        fsm = _make_deterministic_transition_fsm()
        pipeline = _make_pipeline(fsm_def=fsm)
        instance = _make_instance(current_state="start")  # has_name absent
        result = pipeline.process_compiled(instance, "hi", "c1", tier=2)
        assert result == "Hello from mock LLM"
        assert instance.current_state == "start"

    def test_tier2_pre_post_transition_handlers_fire(self) -> None:
        """T2.c — PRE_TRANSITION and POST_TRANSITION fire on transition."""
        fsm = _make_deterministic_transition_fsm()

        counts = {"pre": 0, "post": 0}

        class _TC(BaseHandler):
            def __init__(self, target_timing, key):
                super().__init__(name=f"{key}_counter")
                self._t = target_timing
                self._k = key

            def should_execute(
                self, timing, current_state, target_state, context, updated_keys=None
            ):
                return timing is self._t

            def execute(self, context):
                counts[self._k] += 1
                return {}

        hs = HandlerSystem(error_mode="raise")
        hs.register_handler(_TC(HandlerTiming.PRE_TRANSITION, "pre"))
        hs.register_handler(_TC(HandlerTiming.POST_TRANSITION, "post"))
        pipeline = _make_pipeline(fsm_def=fsm, handler_system=hs)
        instance = _make_instance(
            current_state="start", context_data={"has_name": True}
        )
        pipeline.process_compiled(instance, "hi", "c1", tier=2)
        assert counts == {"pre": 1, "post": 1}

    def test_tier2_post_transition_reextract_fires(self) -> None:
        """T2.d — post-transition re-extraction runs for non-agent FSM
        when transition occurred and new state has missing fields."""
        from fsm_llm.definitions import FieldExtractionConfig

        # new state (end) has a field_extraction that is initially missing
        fsm = _make_deterministic_transition_fsm()
        object.__setattr__(
            fsm.states["end"],
            "field_extractions",
            [
                FieldExtractionConfig(
                    field_name="goodbye_note",
                    field_type="str",
                    extraction_instructions="Extract goodbye note.",
                    required=False,
                )
            ],
        )
        llm = _make_mock_llm()
        configure_mock_extract_field(llm, {"goodbye_note": "bye!"})
        pipeline = _make_pipeline(fsm_def=fsm, llm=llm)

        instance = _make_instance(
            current_state="start", context_data={"has_name": True}
        )
        pipeline.process_compiled(instance, "hi", "c1", tier=2)
        assert instance.current_state == "end"
        # Post-transition re-extract populated goodbye_note
        assert instance.context.data.get("goodbye_note") == "bye!"

    def test_tier2_post_transition_reextract_skipped_for_agent_fsm(self) -> None:
        """T2.e — post-transition re-extraction skipped when agent_trace
        is in context."""
        from fsm_llm.definitions import FieldExtractionConfig

        fsm = _make_deterministic_transition_fsm()
        object.__setattr__(
            fsm.states["end"],
            "field_extractions",
            [
                FieldExtractionConfig(
                    field_name="goodbye_note",
                    field_type="str",
                    extraction_instructions="Extract goodbye note.",
                    required=False,
                )
            ],
        )
        llm = _make_mock_llm()
        configure_mock_extract_field(llm, {"goodbye_note": "bye!"})
        pipeline = _make_pipeline(fsm_def=fsm, llm=llm)

        instance = _make_instance(
            current_state="start",
            context_data={"has_name": True, "agent_trace": ["t1"]},
        )
        pipeline.process_compiled(instance, "hi", "c1", tier=2)
        assert instance.current_state == "end"
        # Re-extract skipped — goodbye_note NOT set
        assert "goodbye_note" not in instance.context.data

    # test_tier2_equivalence_with_legacy_process — deleted in S11
    # (D-S11-00). Legacy `pipeline.process` is gone.

    def test_tier2_transition_occurred_flows_to_cb_respond(self) -> None:
        """_TurnState.transition_occurred/previous_state feed into
        _execute_response_generation_pass (CB_RESPOND reads them).

        Post-A.M3c (plan_2026-04-29_0f87b9c4) the default lifts non-cohort
        responses to a D2 Leaf (no longer routes through CB_RESPOND →
        ``_execute_response_generation_pass``); explicitly disable the
        Leaf path to assert the legacy CB_RESPOND wiring still threads
        ``(transition_occurred, previous_state)`` correctly. Removed when
        the field is retired in M3d-wide."""
        fsm = _make_deterministic_transition_fsm()
        # Explicit-False: keep the legacy CB_RESPOND wire shape this test
        # was authored against. M3d-wide retirement will remove the field
        # AND this regression-coverage test together.
        for s in fsm.states.values():
            s._emit_response_leaf_for_non_cohort = False
        pipeline = _make_pipeline(fsm_def=fsm)
        seen: list = []
        original = pipeline._execute_response_generation_pass

        def _spy(inst, msg, extr, transitioned, prev, conv):
            seen.append((transitioned, prev))
            return original(inst, msg, extr, transitioned, prev, conv)

        pipeline._execute_response_generation_pass = _spy  # type: ignore

        instance = _make_instance(
            current_state="start", context_data={"has_name": True}
        )
        pipeline.process_compiled(instance, "hi", "c1", tier=2)
        assert seen == [(True, "start")]

    def test_tier2_rejects_structurally_ambiguous_fsm(self) -> None:
        """Tier-2 statically rejects FSMs with structurally ambiguous
        transitions (D-S9-07 — D-S8b-02 revisit: graduated from runtime
        sentinel to compile-time `_state_may_be_ambiguous` check)."""

        # Build an FSM with a state that has 2 competing transitions both
        # firing on the same condition → AMBIGUOUS.
        fsm = FSMDefinition(
            name="amb_fsm",
            description="ambiguous test",
            initial_state="start",
            states={
                "start": State(
                    id="start",
                    description="start",
                    purpose="choose",
                    response_instructions="r",
                    transitions=[
                        Transition(
                            target_state="left",
                            description="left path",
                            priority=100,
                        ),
                        Transition(
                            target_state="right",
                            description="right path",
                            priority=100,
                        ),
                    ],
                ),
                "left": State(
                    id="left",
                    description="left",
                    purpose="left",
                    response_instructions="r",
                    transitions=[],
                ),
                "right": State(
                    id="right",
                    description="right",
                    purpose="right",
                    response_instructions="r",
                    transitions=[],
                ),
            },
        )
        pipeline = _make_pipeline(fsm_def=fsm)
        instance = _make_instance(current_state="start")
        with pytest.raises(ValueError, match=r"tier=2 cohort violation.*AMBIGUOUS"):
            pipeline.process_compiled(instance, "hi", "c1", tier=2)


# ══════════════════════════════════════════════════════════════
# S8b T3 cohort: full pipeline. CB_RESOLVE_AMBIG wired (curried).
# Ambiguous transitions resolve via the classifier stub; turn-state
# continuity (last_evaluation populated by CB_EVAL_TRANSIT, read by
# CB_RESOLVE_AMBIG) is asserted.
# ══════════════════════════════════════════════════════════════


def _make_ambiguous_fsm(n_branches: int = 2) -> FSMDefinition:
    """FSM with N competing transitions from `start` to `t{i}` — AMBIGUOUS.
    No conditions → all transitions fire → evaluator returns AMBIGUOUS."""
    transitions = [
        Transition(
            target_state=f"t{i}",
            description=f"branch {i}",
            priority=100,
        )
        for i in range(n_branches)
    ]
    states: dict[str, State] = {
        "start": State(
            id="start",
            description="start",
            purpose="choose path",
            response_instructions="Respond.",
            transitions=transitions,
        ),
    }
    for i in range(n_branches):
        states[f"t{i}"] = State(
            id=f"t{i}",
            description=f"target {i}",
            purpose=f"t{i}",
            response_instructions="Respond.",
            transitions=[],
        )
    return FSMDefinition(
        name="amb_fsm",
        description="ambiguous test",
        initial_state="start",
        states=states,
    )


class TestPipelineProcessCompiledAmbiguous:
    """S8b T3: CB_RESOLVE_AMBIG curried closure resolves ambiguity via
    classifier; turn-state threads last_evaluation from CB_EVAL_TRANSIT
    to CB_RESOLVE_AMBIG (no duplicate evaluation)."""

    def test_tier3_ambiguous_resolves_and_applies(self) -> None:
        """T3.a — AMBIGUOUS: classifier resolves; target_state applied."""
        fsm = _make_ambiguous_fsm(n_branches=3)
        pipeline = _make_pipeline(fsm_def=fsm)

        # Stub _resolve_ambiguous_transition to pick "t1"
        def _fake_resolve(evaluation, msg, extr, instance, conv_id):
            return "t1"

        pipeline._resolve_ambiguous_transition = _fake_resolve  # type: ignore

        instance = _make_instance(current_state="start")
        result = pipeline.process_compiled(instance, "pick one", "c1", tier=3)
        assert result == "Hello from mock LLM"
        assert instance.current_state == "t1"

    def test_tier3_ambiguous_fallback_no_state_change(self) -> None:
        """T3.b — classifier returns current state → no state change."""
        fsm = _make_ambiguous_fsm(n_branches=2)
        pipeline = _make_pipeline(fsm_def=fsm)

        def _fake_resolve(evaluation, msg, extr, instance, conv_id):
            return "start"  # fallback to current

        pipeline._resolve_ambiguous_transition = _fake_resolve  # type: ignore

        instance = _make_instance(current_state="start")
        result = pipeline.process_compiled(instance, "m", "c1", tier=3)
        assert result == "Hello from mock LLM"
        assert instance.current_state == "start"

    def test_tier3_classifier_exception_falls_through(self) -> None:
        """T3.c — if _resolve_ambiguous_transition internally catches and
        returns current state, compiled path still succeeds."""
        fsm = _make_ambiguous_fsm(n_branches=2)
        pipeline = _make_pipeline(fsm_def=fsm)

        def _fake_resolve(evaluation, msg, extr, instance, conv_id):
            # Simulating the fallback: returns current_state
            return instance.current_state

        pipeline._resolve_ambiguous_transition = _fake_resolve  # type: ignore

        instance = _make_instance(current_state="start")
        pipeline.process_compiled(instance, "m", "c1", tier=3)
        assert instance.current_state == "start"

    def test_tier3_turn_state_last_evaluation_populated(self) -> None:
        """T3.d — _TurnState.last_evaluation is written by CB_EVAL_TRANSIT
        and read (non-None) by CB_RESOLVE_AMBIG. No duplicate eval."""
        fsm = _make_ambiguous_fsm(n_branches=2)
        pipeline = _make_pipeline(fsm_def=fsm)

        eval_calls = {"n": 0}
        original_eval = pipeline.transition_evaluator.evaluate_transitions

        def _count_eval(*args, **kwargs):
            eval_calls["n"] += 1
            return original_eval(*args, **kwargs)

        pipeline.transition_evaluator.evaluate_transitions = _count_eval  # type: ignore

        seen_evaluations: list = []

        def _fake_resolve(evaluation, msg, extr, instance, conv_id):
            # Assert we got the SAME evaluation object from CB_EVAL_TRANSIT
            seen_evaluations.append(evaluation)
            assert evaluation is not None
            return "t0"

        pipeline._resolve_ambiguous_transition = _fake_resolve  # type: ignore

        instance = _make_instance(current_state="start")
        pipeline.process_compiled(instance, "m", "c1", tier=3)
        # evaluate_transitions called exactly once (no duplicate)
        assert eval_calls["n"] == 1
        # And the TransitionEvaluation object flowed through
        assert len(seen_evaluations) == 1
        assert seen_evaluations[0] is not None

    # test_tier3_equivalence_with_legacy_on_6way — deleted in S11
    # (D-S11-00). Legacy `pipeline.process` is gone. Coverage for the
    # 6-way ambiguous path is retained in the other tier=3 tests in this
    # class plus the form_filling example's integration test.

    def test_tier3_curried_cb_resolve_ambig_shape(self) -> None:
        """T3.f — CB_RESOLVE_AMBIG factory returns a curried 2-level callable."""
        from fsm_llm.pipeline import _TurnState

        fsm = _make_ambiguous_fsm(n_branches=2)
        pipeline = _make_pipeline(fsm_def=fsm)
        ts = _TurnState()
        factory = pipeline._make_cb_resolve_ambig(
            _make_instance(current_state="start"), "c1", ts
        )
        # factory(inst) → callable; callable(msg) → anything
        inst = _make_instance(current_state="start")
        curried = factory(inst)
        assert callable(curried)
        # Without last_evaluation set, inner must raise LLMResponseError
        from fsm_llm.definitions import LLMResponseError

        with pytest.raises(LLMResponseError, match=r"contract violation"):
            curried("msg")

    def test_tier3_full_cohort_on_simple_response_fsm(self) -> None:
        """T3 full cohort works on trivial response-only FSM (degenerate
        case — no transitions, no extractions). Used to exercise the
        tier=3 default path when compiled_term_resolver is wired."""
        fsm = _make_response_only_fsm()
        pipeline = _make_pipeline(fsm_def=fsm)
        instance = _make_instance(current_state="hello")
        result = pipeline.process_compiled(instance, "m", "c1", tier=3)
        assert result == "Hello from mock LLM"


# ══════════════════════════════════════════════════════════════
# S8b scaffold: compiled_term_resolver ctor arg + _TurnState +
# parameterized cohort gate. No behavior change at tier=0.
# ══════════════════════════════════════════════════════════════


class TestS8bScaffold:
    """Step-1 scaffold: ctor arg, _TurnState dataclass, cohort tiers."""

    def test_ctor_accepts_compiled_term_resolver(self) -> None:
        """MessagePipeline accepts compiled_term_resolver=callable."""
        from fsm_llm.lam.fsm_compile import compile_fsm

        fsm = _make_response_only_fsm()

        def resolver(fsm_id):
            return compile_fsm(fsm)

        pipeline = _make_pipeline(fsm_def=fsm)
        # Re-construct to exercise the ctor arg path
        pipeline2 = MessagePipeline(
            llm_interface=pipeline.llm_interface,
            data_extraction_prompt_builder=pipeline.data_extraction_prompt_builder,
            response_generation_prompt_builder=pipeline.response_generation_prompt_builder,
            transition_evaluator=pipeline.transition_evaluator,
            handler_system=pipeline.handler_system,
            fsm_resolver=pipeline.fsm_resolver,
            field_extraction_prompt_builder=pipeline.field_extraction_prompt_builder,
            compiled_term_resolver=resolver,
        )
        assert pipeline2.compiled_term_resolver is resolver

    def test_ctor_default_is_none(self) -> None:
        """Omitted compiled_term_resolver defaults to None (back-compat)."""
        pipeline = _make_pipeline()
        assert pipeline.compiled_term_resolver is None

    def test_turn_state_dataclass_default_shape(self) -> None:
        """_TurnState dataclass exists with the fields documented in D-S8b-01."""
        from fsm_llm.pipeline import _TurnState

        ts = _TurnState()
        assert ts.extraction_response is None
        assert ts.last_evaluation is None
        assert ts.transition_occurred is False
        assert ts.previous_state is None
        assert ts.extraction_dispatcher_ran is False

    def test_check_compiled_cohort_tier0_matches_probe(self) -> None:
        """tier=0 rejects transitions, extractions — same as the probe."""
        pipeline = _make_pipeline()
        fsm_with_trans = _make_fsm_definition(
            {
                "start": _make_state(
                    "start",
                    transitions=[
                        Transition(target_state="end", description="d", priority=1)
                    ],
                ),
                "end": _make_state("end"),
            }
        )
        with pytest.raises(ValueError, match=r"tier=0 cohort violation"):
            pipeline._check_compiled_cohort(fsm_with_trans, tier=0)

    def test_check_compiled_cohort_tier1_allows_extractions(self) -> None:
        """tier=1 admits extractions; still rejects transitions."""
        pipeline = _make_pipeline()
        from fsm_llm.definitions import State

        fsm_extract_only = FSMDefinition(
            name="e",
            description="d",
            initial_state="start",
            states={
                "start": State(
                    id="start",
                    description="d",
                    purpose="p",
                    response_instructions="r",
                    extraction_instructions="extract",
                    transitions=[],
                )
            },
        )
        # Should NOT raise at tier=1
        pipeline._check_compiled_cohort(fsm_extract_only, tier=1)
        # But tier=0 should still reject
        with pytest.raises(ValueError, match=r"extraction_instructions"):
            pipeline._check_compiled_cohort(fsm_extract_only, tier=0)

    def test_check_compiled_cohort_tier2_allows_transitions(self) -> None:
        """tier=2 admits transitions; tier=1 rejects them."""
        pipeline = _make_pipeline()
        fsm_with_trans = _make_fsm_definition(
            {
                "start": _make_state(
                    "start",
                    transitions=[
                        Transition(target_state="end", description="d", priority=1)
                    ],
                ),
                "end": _make_state("end"),
            }
        )
        # tier=2 allows
        pipeline._check_compiled_cohort(fsm_with_trans, tier=2)
        # tier=1 rejects
        with pytest.raises(ValueError, match=r"has transitions"):
            pipeline._check_compiled_cohort(fsm_with_trans, tier=1)

    def test_check_compiled_cohort_tier3_admits_everything(self) -> None:
        """tier=3 is full cohort — nothing rejected."""
        pipeline = _make_pipeline()
        fsm_with_trans = _make_fsm_definition(
            {
                "start": _make_state(
                    "start",
                    transitions=[
                        Transition(target_state="end", description="d", priority=1)
                    ],
                ),
                "end": _make_state("end"),
            }
        )
        pipeline._check_compiled_cohort(fsm_with_trans, tier=3)

    def test_check_compiled_cohort_invalid_tier(self) -> None:
        """Unknown tier raises ValueError."""
        pipeline = _make_pipeline()
        fsm = _make_response_only_fsm()
        with pytest.raises(ValueError, match=r"invalid cohort tier"):
            pipeline._check_compiled_cohort(fsm, tier=99)

    def test_probe_cohort_backcompat_delegates(self) -> None:
        """_check_probe_cohort is preserved; delegates to tier=0."""
        pipeline = _make_pipeline()
        fsm = _make_response_only_fsm()
        pipeline._check_probe_cohort(fsm)  # no raise

    def test_process_compiled_routes_through_resolver_when_wired(self) -> None:
        """If compiled_term_resolver is supplied, process_compiled consults it."""
        from fsm_llm.lam.fsm_compile import compile_fsm

        fsm = _make_response_only_fsm()
        calls = {"n": 0}

        def resolver(fsm_id):
            calls["n"] += 1
            return compile_fsm(fsm)

        base = _make_pipeline(fsm_def=fsm)
        pipeline = MessagePipeline(
            llm_interface=base.llm_interface,
            data_extraction_prompt_builder=base.data_extraction_prompt_builder,
            response_generation_prompt_builder=base.response_generation_prompt_builder,
            transition_evaluator=base.transition_evaluator,
            handler_system=base.handler_system,
            fsm_resolver=base.fsm_resolver,
            field_extraction_prompt_builder=base.field_extraction_prompt_builder,
            compiled_term_resolver=resolver,
        )
        inst = _make_instance(current_state="hello")
        result = pipeline.process_compiled(inst, "m", "c1")
        assert result == "Hello from mock LLM"
        assert calls["n"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
