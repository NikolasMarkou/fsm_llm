"""
Tests for multi-pass extraction in the MessagePipeline.

Covers: _build_field_configs_from_state auto-conversion, field-level retry
for missing keys, no retry when disabled, merge logic, max retries respected,
refinement prompt content, and field-level confidence threshold behavior.
"""

from unittest.mock import MagicMock

import pytest

from fsm_llm.definitions import (
    DataExtractionResponse,
    FieldExtractionResponse,
    FSMContext,
    FSMDefinition,
    FSMInstance,
    ResponseGenerationResponse,
    State,
)
from fsm_llm.handlers import HandlerSystem
from fsm_llm.llm import LLMInterface
from fsm_llm.pipeline import MessagePipeline
from fsm_llm.prompts import (
    DataExtractionPromptBuilder,
    ResponseGenerationPromptBuilder,
)
from fsm_llm.transition_evaluator import TransitionEvaluator


def configure_mock_extract_field(mock_llm, mock_data=None, confidence=1.0):
    """Configure a mock LLM with extract_field support.

    Parameters
    ----------
    mock_data : dict | None
        Mapping of field_name -> value.  Fields not in the dict return
        ``value=None, confidence=0.0, is_valid=False``.
    confidence : float
        Confidence to use for fields that *are* found in mock_data.
    """
    data = mock_data or {}
    def _side_effect(request):
        value = data.get(request.field_name)
        return FieldExtractionResponse(
            field_name=request.field_name,
            value=value,
            confidence=confidence if value is not None else 0.0,
            reasoning="Mock field extraction",
            is_valid=value is not None,
        )
    mock_llm.extract_field.side_effect = _side_effect
    return mock_llm

# ── Helpers ───────────────────────────────────────────────────


def _make_state(
    state_id,
    transitions=None,
    purpose="Test purpose",
    required_context_keys=None,
    extraction_retries=0,
    extraction_confidence_threshold=0.0,
    extraction_instructions=None,
):
    return State(
        id=state_id,
        description=f"{state_id} description",
        purpose=purpose,
        transitions=transitions or [],
        required_context_keys=required_context_keys,
        extraction_retries=extraction_retries,
        extraction_confidence_threshold=extraction_confidence_threshold,
        extraction_instructions=extraction_instructions,
    )


def _make_single_state_fsm(state):
    """Create a valid FSM with a single terminal state (no orphan issues)."""
    return FSMDefinition(
        name="test_fsm",
        description="Test FSM",
        initial_state=state.id,
        states={state.id: state},
    )


def _make_fsm_definition(states_dict, initial_state="start"):
    return FSMDefinition(
        name="test_fsm",
        description="Test FSM",
        initial_state=initial_state,
        states=states_dict,
    )


def _make_instance(fsm_id="test_fsm", current_state="start", context_data=None):
    ctx = FSMContext(data=context_data or {})
    return FSMInstance(
        fsm_id=fsm_id,
        current_state=current_state,
        context=ctx,
    )


def _make_mock_llm():
    llm = MagicMock(spec=LLMInterface)
    llm.extract_data.return_value = DataExtractionResponse(
        extracted_data={},
        confidence=1.0,
        reasoning="mock",
    )
    llm.generate_response.return_value = ResponseGenerationResponse(
        message="OK", message_type="response", reasoning="mock"
    )
    llm.decide_transition.side_effect = NotImplementedError(
        "decide_transition is deprecated"
    )
    configure_mock_extract_field(llm)
    return llm


def _make_pipeline(fsm_def=None, llm=None):
    if fsm_def is None:
        from fsm_llm.definitions import Transition

        fsm_def = _make_fsm_definition(
            {
                "start": _make_state(
                    "start",
                    transitions=[
                        Transition(
                            target_state="end",
                            description="go",
                            priority=100,
                        )
                    ],
                ),
                "end": _make_state("end"),
            }
        )
    if llm is None:
        llm = _make_mock_llm()

    def resolver(fsm_id):
        return fsm_def

    return MessagePipeline(
        llm_interface=llm,
        data_extraction_prompt_builder=DataExtractionPromptBuilder(),
        response_generation_prompt_builder=ResponseGenerationPromptBuilder(),
        transition_evaluator=TransitionEvaluator(),
        handler_system=HandlerSystem(),
        fsm_resolver=resolver,
    )


# ══════════════════════════════════════════════════════════════
# 1. _build_field_configs_from_state
# ══════════════════════════════════════════════════════════════


class TestBuildFieldConfigsFromState:
    """Tests for MessagePipeline._build_field_configs_from_state."""

    def test_no_configs_when_no_required_keys(self):
        state = _make_state("s")
        configs = MessagePipeline._build_field_configs_from_state(state)
        assert configs == []

    def test_creates_one_config_per_required_key(self):
        state = _make_state("s", required_context_keys=["name", "email"])
        configs = MessagePipeline._build_field_configs_from_state(state)
        assert len(configs) == 2
        names = {c.field_name for c in configs}
        assert names == {"name", "email"}

    def test_configs_are_required(self):
        state = _make_state("s", required_context_keys=["name"])
        configs = MessagePipeline._build_field_configs_from_state(state)
        assert configs[0].required is True

    def test_configs_use_extraction_instructions(self):
        state = _make_state(
            "s",
            required_context_keys=["name"],
            extraction_instructions="Get the user's full name",
        )
        configs = MessagePipeline._build_field_configs_from_state(state)
        assert "Get the user's full name" in configs[0].extraction_instructions

    def test_configs_use_confidence_threshold(self):
        state = _make_state(
            "s",
            required_context_keys=["name"],
            extraction_confidence_threshold=0.8,
        )
        configs = MessagePipeline._build_field_configs_from_state(state)
        assert configs[0].confidence_threshold == 0.8

    def test_default_field_type_is_any(self):
        state = _make_state("s", required_context_keys=["name"])
        configs = MessagePipeline._build_field_configs_from_state(state)
        assert configs[0].field_type == "any"


# ══════════════════════════════════════════════════════════════
# 2. Multi-pass extraction in _execute_data_extraction
# ══════════════════════════════════════════════════════════════


class TestMultiPassExtraction:
    """Tests for field-level multi-pass extraction via _execute_data_extraction.

    The pipeline now calls ``llm.extract_field()`` per field (driven by
    ``required_context_keys``), not ``llm.extract_data()``.
    """

    def test_no_extraction_when_no_required_keys(self):
        """No fields configured -> no extract_field calls."""
        llm = _make_mock_llm()
        fsm_def = _make_fsm_definition(
            {"start": _make_state("start", extraction_retries=0)}
        )
        pipeline = _make_pipeline(fsm_def=fsm_def, llm=llm)
        instance = _make_instance()

        result = pipeline._execute_data_extraction(instance, "hello", "conv1")
        llm.extract_field.assert_not_called()
        assert result.extracted_data == {}

    def test_single_pass_extracts_all_fields(self):
        """Pass 1 extracts all required fields -- no retry needed."""
        llm = _make_mock_llm()
        configure_mock_extract_field(llm, {"name": "Alice", "email": "a@b.com"})
        fsm_def = _make_fsm_definition(
            {
                "start": _make_state(
                    "start",
                    required_context_keys=["name", "email"],
                    extraction_retries=1,
                ),
            }
        )
        pipeline = _make_pipeline(fsm_def=fsm_def, llm=llm)
        instance = _make_instance()

        result = pipeline._execute_data_extraction(instance, "I'm Alice", "conv1")
        # 2 fields extracted in pass 1, no retry needed
        assert llm.extract_field.call_count == 2
        assert result.extracted_data == {"name": "Alice", "email": "a@b.com"}

    def test_no_retry_when_retries_zero(self):
        """extraction_retries=0: only one pass, even if fields are missing."""
        llm = _make_mock_llm()
        configure_mock_extract_field(llm, {"name": "Alice"})  # email missing
        fsm_def = _make_fsm_definition(
            {
                "start": _make_state(
                    "start",
                    required_context_keys=["name", "email"],
                    extraction_retries=0,
                ),
            }
        )
        pipeline = _make_pipeline(fsm_def=fsm_def, llm=llm)
        instance = _make_instance()

        result = pipeline._execute_data_extraction(instance, "hello", "conv1")
        # 2 fields attempted in pass 1, no retry pass
        assert llm.extract_field.call_count == 2
        assert result.extracted_data == {"name": "Alice"}

    def test_retry_on_missing_keys(self):
        """Retry when required_context_keys are missing after pass 1."""
        llm = _make_mock_llm()
        call_count = {"n": 0}

        def _side_effect(request):
            call_count["n"] += 1
            if request.field_name == "name":
                return FieldExtractionResponse(
                    field_name="name", value="Alice",
                    confidence=1.0, reasoning="ok", is_valid=True,
                )
            if request.field_name == "email":
                # First call: fail; second call (retry): succeed
                if call_count["n"] <= 2:
                    return FieldExtractionResponse(
                        field_name="email", value=None,
                        confidence=0.0, reasoning="not found", is_valid=False,
                    )
                return FieldExtractionResponse(
                    field_name="email", value="alice@example.com",
                    confidence=1.0, reasoning="ok", is_valid=True,
                )
            return FieldExtractionResponse(
                field_name=request.field_name, value=None,
                confidence=0.0, reasoning="unknown", is_valid=False,
            )

        llm.extract_field.side_effect = _side_effect

        fsm_def = _make_fsm_definition(
            {
                "start": _make_state(
                    "start",
                    required_context_keys=["name", "email"],
                    extraction_retries=1,
                ),
            }
        )
        pipeline = _make_pipeline(fsm_def=fsm_def, llm=llm)
        instance = _make_instance()

        result = pipeline._execute_data_extraction(instance, "I'm Alice", "conv1")
        # Pass 1: name + email (2 calls), retry: email only (1 call) = 3 total
        assert llm.extract_field.call_count == 3
        assert result.extracted_data == {"name": "Alice", "email": "alice@example.com"}

    def test_no_retry_when_key_in_existing_context(self):
        """Keys already in instance context don't trigger retry."""
        llm = _make_mock_llm()
        configure_mock_extract_field(llm, {"name": "Alice"})  # email not extracted
        fsm_def = _make_fsm_definition(
            {
                "start": _make_state(
                    "start",
                    required_context_keys=["name", "email"],
                    extraction_retries=1,
                ),
            }
        )
        pipeline = _make_pipeline(fsm_def=fsm_def, llm=llm)
        # email already in context
        instance = _make_instance(context_data={"email": "a@b.com"})

        result = pipeline._execute_data_extraction(instance, "hello", "conv1")
        # Pass 1: 2 fields. email in context -> no retry needed
        assert llm.extract_field.call_count == 2
        assert result.extracted_data["name"] == "Alice"

    def test_merge_preserves_existing_keys(self):
        """Retry should add new keys without removing pass-1 data."""
        llm = _make_mock_llm()
        call_count = {"n": 0}

        def _side_effect(request):
            call_count["n"] += 1
            if request.field_name == "a":
                return FieldExtractionResponse(
                    field_name="a", value="1", confidence=1.0,
                    reasoning="ok", is_valid=True,
                )
            if request.field_name == "b":
                return FieldExtractionResponse(
                    field_name="b", value="2", confidence=1.0,
                    reasoning="ok", is_valid=True,
                )
            if request.field_name == "c":
                # Fail on pass 1 (calls 1-3), succeed on retry (call 4)
                if call_count["n"] <= 3:
                    return FieldExtractionResponse(
                        field_name="c", value=None, confidence=0.0,
                        reasoning="not found", is_valid=False,
                    )
                return FieldExtractionResponse(
                    field_name="c", value="3", confidence=1.0,
                    reasoning="ok", is_valid=True,
                )
            return FieldExtractionResponse(
                field_name=request.field_name, value=None,
                confidence=0.0, reasoning="unknown", is_valid=False,
            )

        llm.extract_field.side_effect = _side_effect

        fsm_def = _make_fsm_definition(
            {
                "start": _make_state(
                    "start",
                    required_context_keys=["a", "b", "c"],
                    extraction_retries=1,
                ),
            }
        )
        pipeline = _make_pipeline(fsm_def=fsm_def, llm=llm)
        instance = _make_instance()

        result = pipeline._execute_data_extraction(instance, "test", "conv1")
        assert result.extracted_data == {"a": "1", "b": "2", "c": "3"}

    def test_max_retries_respected(self):
        """Should not retry more than extraction_retries times."""
        llm = _make_mock_llm()
        # All fields always fail -- should retry exactly 2 times
        configure_mock_extract_field(llm, {})  # all fields return None
        fsm_def = _make_fsm_definition(
            {
                "start": _make_state(
                    "start",
                    required_context_keys=["name"],
                    extraction_retries=2,
                ),
            }
        )
        pipeline = _make_pipeline(fsm_def=fsm_def, llm=llm)
        instance = _make_instance()

        pipeline._execute_data_extraction(instance, "test", "conv1")
        # 1 initial call + 2 retry calls = 3 total extract_field calls
        assert llm.extract_field.call_count == 3

    def test_stops_early_when_satisfied(self):
        """Should not do extra retries if all required keys are found."""
        llm = _make_mock_llm()
        configure_mock_extract_field(llm, {"name": "Alice", "email": "a@b.com"})
        fsm_def = _make_fsm_definition(
            {
                "start": _make_state(
                    "start",
                    required_context_keys=["name", "email"],
                    extraction_retries=3,  # allows up to 3, but all found in pass 1
                ),
            }
        )
        pipeline = _make_pipeline(fsm_def=fsm_def, llm=llm)
        instance = _make_instance()

        result = pipeline._execute_data_extraction(instance, "test", "conv1")
        # Only pass 1: 2 fields, no retries
        assert llm.extract_field.call_count == 2
        assert result.extracted_data == {"name": "Alice", "email": "a@b.com"}


# ══════════════════════════════════════════════════════════════
# 3. Refinement prompt builder
# ══════════════════════════════════════════════════════════════


class TestRefinementPromptBuilder:
    def test_refinement_prompt_contains_previously_extracted(self):
        builder = DataExtractionPromptBuilder()
        state = _make_state(
            "s",
            extraction_instructions="Extract name and email",
        )
        fsm_def = _make_single_state_fsm(state)
        instance = _make_instance(current_state="s")

        prompt = builder.build_refinement_prompt(
            instance=instance,
            state=state,
            fsm_definition=fsm_def,
            previous_extraction={"name": "Alice"},
            missing_keys=["email"],
        )

        assert "previously_extracted" in prompt
        assert "Alice" in prompt

    def test_refinement_prompt_contains_missing_keys(self):
        builder = DataExtractionPromptBuilder()
        state = _make_state("s")
        fsm_def = _make_single_state_fsm(state)
        instance = _make_instance(current_state="s")

        prompt = builder.build_refinement_prompt(
            instance=instance,
            state=state,
            fsm_definition=fsm_def,
            previous_extraction={},
            missing_keys=["email", "phone"],
        )

        assert "still_missing" in prompt
        assert "email" in prompt
        assert "phone" in prompt

    def test_refinement_prompt_mentions_refinement(self):
        builder = DataExtractionPromptBuilder()
        state = _make_state("s")
        fsm_def = _make_single_state_fsm(state)
        instance = _make_instance(current_state="s")

        prompt = builder.build_refinement_prompt(
            instance=instance,
            state=state,
            fsm_definition=fsm_def,
            previous_extraction={},
            missing_keys=["x"],
        )

        assert "REFINEMENT" in prompt
        assert "ONLY" in prompt

    def test_refinement_prompt_includes_extraction_instructions(self):
        builder = DataExtractionPromptBuilder()
        state = _make_state(
            "s",
            extraction_instructions="Extract user_name and user_email as JSON",
        )
        fsm_def = _make_single_state_fsm(state)
        instance = _make_instance(current_state="s")

        prompt = builder.build_refinement_prompt(
            instance=instance,
            state=state,
            fsm_definition=fsm_def,
            previous_extraction={"user_name": "Bob"},
            missing_keys=["user_email"],
        )

        assert "extraction_instructions" in prompt
        assert "user_name and user_email" in prompt

    def test_refinement_prompt_empty_previous_extraction(self):
        builder = DataExtractionPromptBuilder()
        state = _make_state("s")
        fsm_def = _make_single_state_fsm(state)
        instance = _make_instance(current_state="s")

        prompt = builder.build_refinement_prompt(
            instance=instance,
            state=state,
            fsm_definition=fsm_def,
            previous_extraction={},
            missing_keys=["name"],
        )

        assert "nothing was extracted" in prompt


# ══════════════════════════════════════════════════════════════
# 4. State model fields
# ══════════════════════════════════════════════════════════════


class TestStateExtractionFields:
    def test_default_values(self):
        state = _make_state("s")
        assert state.extraction_retries == 0
        assert state.extraction_confidence_threshold == 0.0

    def test_custom_values(self):
        state = _make_state(
            "s", extraction_retries=2, extraction_confidence_threshold=0.8
        )
        assert state.extraction_retries == 2
        assert state.extraction_confidence_threshold == 0.8

    def test_retries_validation_max(self):
        with pytest.raises(ValueError):
            _make_state("s", extraction_retries=4)

    def test_retries_validation_min(self):
        with pytest.raises(ValueError):
            _make_state("s", extraction_retries=-1)

    def test_threshold_validation_max(self):
        with pytest.raises(ValueError):
            _make_state("s", extraction_confidence_threshold=1.5)

    def test_backward_compatible_no_fields(self):
        """States without the new fields should work as before."""
        state = State(
            id="s",
            description="desc",
            purpose="test",
        )
        assert state.extraction_retries == 0
        assert state.extraction_confidence_threshold == 0.0
