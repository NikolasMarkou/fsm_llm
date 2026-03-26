"""
Tests for multi-pass extraction in the MessagePipeline.

Covers: retry triggered on low confidence, retry triggered on missing keys,
no retry when disabled, merge logic, max retries respected, refinement
prompt content, and the _extraction_needs_retry helper.
"""

from unittest.mock import MagicMock

import pytest

from fsm_llm.definitions import (
    DataExtractionResponse,
    FSMContext,
    FSMDefinition,
    FSMInstance,
    ResponseGenerationResponse,
    State,
    TransitionDecisionResponse,
)
from fsm_llm.handlers import HandlerSystem
from fsm_llm.llm import LLMInterface
from fsm_llm.pipeline import MessagePipeline
from fsm_llm.prompts import (
    DataExtractionPromptBuilder,
    ResponseGenerationPromptBuilder,
    TransitionPromptBuilder,
)
from fsm_llm.transition_evaluator import TransitionEvaluator

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
    llm.decide_transition.return_value = TransitionDecisionResponse(
        selected_transition="end", reasoning="mock"
    )
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
        transition_prompt_builder=TransitionPromptBuilder(),
        transition_evaluator=TransitionEvaluator(),
        handler_system=HandlerSystem(),
        fsm_resolver=resolver,
    )


# ══════════════════════════════════════════════════════════════
# 1. _extraction_needs_retry helper
# ══════════════════════════════════════════════════════════════


class TestExtractionNeedsRetry:
    def test_no_retry_when_all_keys_present(self):
        state = _make_state("s", required_context_keys=["name"], extraction_retries=1)
        instance = _make_instance(context_data={})
        response = DataExtractionResponse(
            extracted_data={"name": "Alice"}, confidence=0.9
        )
        pipeline = _make_pipeline()
        needs, missing = pipeline._extraction_needs_retry(response, state, instance)
        assert not needs
        assert missing == []

    def test_retry_when_key_missing_from_extraction(self):
        state = _make_state(
            "s",
            required_context_keys=["name", "email"],
            extraction_retries=1,
        )
        instance = _make_instance(context_data={})
        response = DataExtractionResponse(
            extracted_data={"name": "Alice"}, confidence=0.9
        )
        pipeline = _make_pipeline()
        needs, missing = pipeline._extraction_needs_retry(response, state, instance)
        assert needs
        assert "email" in missing

    def test_no_retry_when_key_in_existing_context(self):
        state = _make_state(
            "s",
            required_context_keys=["name", "email"],
            extraction_retries=1,
        )
        instance = _make_instance(context_data={"email": "a@b.com"})
        response = DataExtractionResponse(
            extracted_data={"name": "Alice"}, confidence=0.9
        )
        pipeline = _make_pipeline()
        needs, _missing = pipeline._extraction_needs_retry(response, state, instance)
        assert not needs

    def test_retry_on_low_confidence(self):
        state = _make_state(
            "s",
            extraction_retries=1,
            extraction_confidence_threshold=0.8,
        )
        instance = _make_instance()
        response = DataExtractionResponse(extracted_data={"x": "y"}, confidence=0.3)
        pipeline = _make_pipeline()
        needs, _missing = pipeline._extraction_needs_retry(response, state, instance)
        assert needs

    def test_no_retry_when_confidence_meets_threshold(self):
        state = _make_state(
            "s",
            extraction_retries=1,
            extraction_confidence_threshold=0.5,
        )
        instance = _make_instance()
        response = DataExtractionResponse(extracted_data={"x": "y"}, confidence=0.8)
        pipeline = _make_pipeline()
        needs, _missing = pipeline._extraction_needs_retry(response, state, instance)
        assert not needs

    def test_retry_on_additional_info_needed(self):
        state = _make_state("s", extraction_retries=1)
        instance = _make_instance()
        response = DataExtractionResponse(
            extracted_data={}, confidence=0.5, additional_info_needed=True
        )
        pipeline = _make_pipeline()
        needs, _ = pipeline._extraction_needs_retry(response, state, instance)
        assert needs

    def test_no_retry_when_retries_disabled(self):
        """Even if keys are missing, retries=0 means no retry at pipeline level."""
        state = _make_state(
            "s",
            required_context_keys=["name"],
            extraction_retries=0,
        )
        instance = _make_instance()
        response = DataExtractionResponse(extracted_data={}, confidence=0.1)
        pipeline = _make_pipeline()
        # _extraction_needs_retry itself doesn't check retries count,
        # but _execute_data_extraction skips the loop when retries=0
        needs, missing = pipeline._extraction_needs_retry(response, state, instance)
        # It reports needs=True, but the caller won't retry
        assert needs
        assert "name" in missing


# ══════════════════════════════════════════════════════════════
# 2. Multi-pass extraction in _execute_data_extraction
# ══════════════════════════════════════════════════════════════


class TestMultiPassExtraction:
    def test_no_retry_when_retries_zero(self):
        """Default: extraction_retries=0, only one LLM call."""
        llm = _make_mock_llm()
        llm.extract_data.return_value = DataExtractionResponse(
            extracted_data={"name": "Alice"}, confidence=0.5
        )
        fsm_def = _make_fsm_definition(
            {"start": _make_state("start", extraction_retries=0)}
        )
        pipeline = _make_pipeline(fsm_def=fsm_def, llm=llm)
        instance = _make_instance()

        result = pipeline._execute_data_extraction(instance, "hello", "conv1")
        assert llm.extract_data.call_count == 1
        assert result.extracted_data == {"name": "Alice"}

    def test_retry_on_missing_keys(self):
        """Retry when required_context_keys are missing after pass 1."""
        llm = _make_mock_llm()
        # Pass 1: extracts name but not email
        # Pass 2 (refinement): extracts email
        llm.extract_data.side_effect = [
            DataExtractionResponse(extracted_data={"name": "Alice"}, confidence=0.7),
            DataExtractionResponse(
                extracted_data={"email": "alice@example.com"}, confidence=0.9
            ),
        ]
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
        assert llm.extract_data.call_count == 2
        assert result.extracted_data == {"name": "Alice", "email": "alice@example.com"}

    def test_retry_on_low_confidence(self):
        """Retry when confidence is below threshold."""
        llm = _make_mock_llm()
        llm.extract_data.side_effect = [
            DataExtractionResponse(extracted_data={"x": "y"}, confidence=0.2),
            DataExtractionResponse(extracted_data={"x": "better_y"}, confidence=0.9),
        ]
        fsm_def = _make_fsm_definition(
            {
                "start": _make_state(
                    "start",
                    extraction_retries=1,
                    extraction_confidence_threshold=0.7,
                ),
            }
        )
        pipeline = _make_pipeline(fsm_def=fsm_def, llm=llm)
        instance = _make_instance()

        result = pipeline._execute_data_extraction(instance, "test", "conv1")
        assert llm.extract_data.call_count == 2
        # Refinement value overrides
        assert result.extracted_data["x"] == "better_y"
        # Confidence updated to the higher value
        assert result.confidence == 0.9

    def test_merge_preserves_existing_keys(self):
        """Refinement should not clear keys from pass 1."""
        llm = _make_mock_llm()
        llm.extract_data.side_effect = [
            DataExtractionResponse(extracted_data={"a": "1", "b": "2"}, confidence=0.5),
            DataExtractionResponse(extracted_data={"c": "3"}, confidence=0.8),
        ]
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

    def test_merge_does_not_override_with_none(self):
        """Refinement None values should not clear existing data."""
        llm = _make_mock_llm()
        llm.extract_data.side_effect = [
            DataExtractionResponse(extracted_data={"name": "Alice"}, confidence=0.5),
            DataExtractionResponse(
                extracted_data={"name": None, "email": "a@b.com"}, confidence=0.8
            ),
        ]
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

        result = pipeline._execute_data_extraction(instance, "test", "conv1")
        # name should NOT be overwritten with None
        assert result.extracted_data["name"] == "Alice"
        assert result.extracted_data["email"] == "a@b.com"

    def test_max_retries_respected(self):
        """Should not retry more than extraction_retries times."""
        llm = _make_mock_llm()
        # Always return low confidence — should retry exactly 2 times
        llm.extract_data.return_value = DataExtractionResponse(
            extracted_data={}, confidence=0.1
        )
        fsm_def = _make_fsm_definition(
            {
                "start": _make_state(
                    "start",
                    extraction_retries=2,
                    extraction_confidence_threshold=0.9,
                ),
            }
        )
        pipeline = _make_pipeline(fsm_def=fsm_def, llm=llm)
        instance = _make_instance()

        pipeline._execute_data_extraction(instance, "test", "conv1")
        # 1 initial + 2 retries = 3 calls
        assert llm.extract_data.call_count == 3

    def test_stops_early_when_satisfied(self):
        """Should not do extra retries if requirements are met."""
        llm = _make_mock_llm()
        llm.extract_data.side_effect = [
            DataExtractionResponse(extracted_data={"name": "Alice"}, confidence=0.5),
            DataExtractionResponse(extracted_data={"email": "a@b.com"}, confidence=0.9),
            # This should never be called
            DataExtractionResponse(extracted_data={"extra": "val"}, confidence=1.0),
        ]
        fsm_def = _make_fsm_definition(
            {
                "start": _make_state(
                    "start",
                    required_context_keys=["name", "email"],
                    extraction_retries=3,  # allows up to 3, but should stop at 1
                ),
            }
        )
        pipeline = _make_pipeline(fsm_def=fsm_def, llm=llm)
        instance = _make_instance()

        result = pipeline._execute_data_extraction(instance, "test", "conv1")
        # 1 initial + 1 retry = 2 (not 4)
        assert llm.extract_data.call_count == 2
        assert "extra" not in result.extracted_data


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
