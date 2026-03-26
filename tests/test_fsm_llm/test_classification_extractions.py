"""Tests for classification-based extraction (classification_extractions on State)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from fsm_llm.classification import Classifier
from fsm_llm.constants import CLASSIFICATION_EXTRACTION_RESULT_SUFFIX
from fsm_llm.definitions import (
    ClassificationExtractionConfig,
    ClassificationResult,
    FSMContext,
    FSMDefinition,
    FSMInstance,
    IntentDefinition,
    State,
    Transition,
)
from fsm_llm.handlers import HandlerSystem
from fsm_llm.pipeline import MessagePipeline
from fsm_llm.prompts import (
    DataExtractionPromptBuilder,
    FieldExtractionPromptBuilder,
    ResponseGenerationPromptBuilder,
)
from fsm_llm.transition_evaluator import TransitionEvaluator

# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------


def _make_intents(*names: str) -> list[IntentDefinition]:
    return [IntentDefinition(name=n, description=f"The {n} intent") for n in names]


def _make_config(
    field_name: str = "sentiment",
    intents: list[str] | None = None,
    fallback: str = "neutral",
    **kwargs,
) -> ClassificationExtractionConfig:
    intent_names = intents or ["positive", "negative", "neutral"]
    return ClassificationExtractionConfig(
        field_name=field_name,
        intents=_make_intents(*intent_names),
        fallback_intent=fallback,
        **kwargs,
    )


def _make_state(
    state_id: str = "triage",
    classification_extractions: list[ClassificationExtractionConfig] | None = None,
    transitions: list[dict] | None = None,
) -> State:
    trans = transitions or [
        {"target_state": "end", "description": "Done"},
    ]
    return State(
        id=state_id,
        description=f"State {state_id}",
        purpose=f"Purpose of {state_id}",
        extraction_instructions="Extract data",
        response_instructions="Respond",
        transitions=[Transition(**t) for t in trans],
        classification_extractions=classification_extractions,
    )


def _terminal_state(state_id: str = "end") -> State:
    return State(
        id=state_id,
        description=f"Terminal state {state_id}",
        purpose="End",
        response_instructions="Goodbye",
        transitions=[],
    )


def _make_fsm(states: dict[str, State] | None = None) -> FSMDefinition:
    if states is None:
        states = {
            "triage": _make_state("triage"),
            "end": _terminal_state("end"),
        }
    return FSMDefinition(
        name="test",
        description="test FSM",
        states=states,
        initial_state="triage",
    )


def _make_pipeline(
    fsm: FSMDefinition | None = None,
    states: dict[str, State] | None = None,
) -> MessagePipeline:
    if fsm is None:
        fsm = _make_fsm(states)
    mock_llm = MagicMock()
    mock_llm.model = "test-model"
    mock_llm.extract_field.return_value = MagicMock(
        field_name="dummy", value=None, confidence=0.0, is_valid=False
    )
    return MessagePipeline(
        llm_interface=mock_llm,
        data_extraction_prompt_builder=DataExtractionPromptBuilder(),
        response_generation_prompt_builder=ResponseGenerationPromptBuilder(),
        transition_evaluator=TransitionEvaluator(),
        handler_system=HandlerSystem(),
        fsm_resolver=lambda fid: fsm,
        field_extraction_prompt_builder=FieldExtractionPromptBuilder(),
    )


def _make_instance(fsm_id: str = "test", state: str = "triage") -> FSMInstance:
    return FSMInstance(
        fsm_id=fsm_id,
        current_state=state,
        context=FSMContext(),
    )


def _mock_classify_result(
    intent: str = "negative",
    confidence: float = 0.9,
    reasoning: str = "test",
    entities: dict | None = None,
) -> ClassificationResult:
    return ClassificationResult(
        reasoning=reasoning,
        intent=intent,
        confidence=confidence,
        entities=entities or {},
    )


# ----------------------------------------------------------
# Model tests
# ----------------------------------------------------------


class TestClassificationExtractionConfig:
    def test_valid_config(self):
        config = _make_config()
        assert config.field_name == "sentiment"
        assert len(config.intents) == 3
        assert config.fallback_intent == "neutral"
        assert config.confidence_threshold == 0.6

    def test_fallback_not_in_intents_raises(self):
        with pytest.raises(ValueError, match="Fallback intent"):
            _make_config(fallback="nonexistent")

    def test_min_intents_enforced(self):
        with pytest.raises(ValueError):
            _make_config(intents=["only_one"])

    def test_defaults(self):
        config = _make_config()
        assert config.required is False
        assert config.confidence_threshold == 0.6
        assert config.model is None
        assert config.prompt_config is None
        assert config.context_keys is None

    def test_custom_model_and_prompt_config(self):
        config = _make_config(
            model="gpt-4o",
            prompt_config={"temperature": 0.1, "max_tokens": 256},
        )
        assert config.model == "gpt-4o"
        assert config.prompt_config["temperature"] == 0.1

    def test_duplicate_intent_names_raises(self):
        with pytest.raises(ValueError, match="unique"):
            ClassificationExtractionConfig(
                field_name="test",
                intents=[
                    IntentDefinition(name="a", description="A"),
                    IntentDefinition(name="a", description="A duplicate"),
                ],
                fallback_intent="a",
            )


class TestStateIntegration:
    def test_classification_extractions_default_none(self):
        state = _make_state()
        assert state.classification_extractions is None

    def test_classification_extractions_roundtrip(self):
        config = _make_config()
        state = _make_state(classification_extractions=[config])
        assert len(state.classification_extractions) == 1
        assert state.classification_extractions[0].field_name == "sentiment"

    def test_state_json_roundtrip(self):
        config = _make_config()
        state = _make_state(classification_extractions=[config])
        data = state.model_dump()
        restored = State(**data)
        assert len(restored.classification_extractions) == 1
        assert restored.classification_extractions[0].field_name == "sentiment"


# ----------------------------------------------------------
# Pipeline _execute_classification_extractions tests
# ----------------------------------------------------------


class TestExecuteClassificationExtractions:
    def test_success(self):
        config = _make_config()
        state = _make_state(classification_extractions=[config])
        fsm = _make_fsm({"triage": state, "end": _terminal_state("end")})
        pipeline = _make_pipeline(fsm)
        instance = _make_instance()

        result = _mock_classify_result("negative", 0.9)

        with patch.object(Classifier, "classify", return_value=result):
            data = pipeline._execute_classification_extractions(
                state, "I'm so frustrated!", instance, "conv1"
            )

        assert data["sentiment"] == "negative"
        full_key = f"_sentiment{CLASSIFICATION_EXTRACTION_RESULT_SUFFIX}"
        assert full_key in data
        assert data[full_key]["intent"] == "negative"
        assert data[full_key]["confidence"] == 0.9

    def test_below_threshold_skipped(self):
        config = _make_config(confidence_threshold=0.8)
        state = _make_state(classification_extractions=[config])
        pipeline = _make_pipeline()
        instance = _make_instance()

        result = _mock_classify_result("negative", 0.5)

        with patch.object(Classifier, "classify", return_value=result):
            data = pipeline._execute_classification_extractions(
                state, "hmm", instance, "conv1"
            )

        assert "sentiment" not in data

    def test_fallback_intent_skipped(self):
        config = _make_config()
        state = _make_state(classification_extractions=[config])
        pipeline = _make_pipeline()
        instance = _make_instance()

        result = _mock_classify_result("neutral", 0.95)  # neutral is fallback

        with patch.object(Classifier, "classify", return_value=result):
            data = pipeline._execute_classification_extractions(
                state, "what time is it", instance, "conv1"
            )

        assert "sentiment" not in data

    def test_exception_handling(self):
        config = _make_config()
        state = _make_state(classification_extractions=[config])
        pipeline = _make_pipeline()
        instance = _make_instance()

        with patch.object(Classifier, "classify", side_effect=RuntimeError("boom")):
            data = pipeline._execute_classification_extractions(
                state, "test", instance, "conv1"
            )

        assert data == {}

    def test_no_model_available(self):
        config = _make_config()
        state = _make_state(classification_extractions=[config])
        pipeline = _make_pipeline()
        pipeline.llm_interface = MagicMock(spec=[])  # no .model attribute
        instance = _make_instance()

        data = pipeline._execute_classification_extractions(
            state, "test", instance, "conv1"
        )
        assert data == {}

    def test_model_override(self):
        config = _make_config(model="gpt-4o")
        state = _make_state(classification_extractions=[config])
        pipeline = _make_pipeline()
        instance = _make_instance()

        result = _mock_classify_result("positive", 0.9)

        with patch.object(Classifier, "__init__", return_value=None) as mock_init:
            with patch.object(Classifier, "classify", return_value=result):
                # Need to set up the mock so __init__ doesn't fail
                mock_init.return_value = None
                pipeline._execute_classification_extractions(
                    state, "test", instance, "conv1"
                )

            # Verify model override was used
            call_kwargs = mock_init.call_args
            assert call_kwargs.kwargs.get("model") == "gpt-4o" or (
                call_kwargs and call_kwargs[1].get("model") == "gpt-4o"
            )

    def test_context_keys_snapshot(self):
        config = _make_config(context_keys=["user_name", "missing_key"])
        state = _make_state(classification_extractions=[config])
        pipeline = _make_pipeline()
        instance = _make_instance()
        instance.context.data["user_name"] = "Alice"

        result = _mock_classify_result("positive", 0.9)

        with patch.object(Classifier, "classify", return_value=result):
            data = pipeline._execute_classification_extractions(
                state, "test", instance, "conv1"
            )

        full_key = f"_sentiment{CLASSIFICATION_EXTRACTION_RESULT_SUFFIX}"
        assert data[full_key]["context_snapshot"] == {"user_name": "Alice"}

    def test_multiple_classification_extractions(self):
        configs = [
            _make_config(field_name="sentiment"),
            _make_config(
                field_name="urgency",
                intents=["high", "low"],
                fallback="low",
            ),
        ]
        state = _make_state(classification_extractions=configs)
        pipeline = _make_pipeline()
        instance = _make_instance()

        results = iter(
            [
                _mock_classify_result("negative", 0.9),
                _mock_classify_result("high", 0.85),
            ]
        )

        with patch.object(
            Classifier, "classify", side_effect=lambda msg: next(results)
        ):
            data = pipeline._execute_classification_extractions(
                state, "urgent complaint", instance, "conv1"
            )

        assert data["sentiment"] == "negative"
        assert data["urgency"] == "high"

    def test_empty_configs_returns_empty(self):
        state = _make_state()  # no classification_extractions
        pipeline = _make_pipeline()
        instance = _make_instance()

        data = pipeline._execute_classification_extractions(
            state, "test", instance, "conv1"
        )
        assert data == {}


# ----------------------------------------------------------
# Integration with _execute_data_extraction
# ----------------------------------------------------------


class TestDataExtractionIntegration:
    def test_classification_extraction_in_data_extraction(self):
        config = _make_config()
        state = _make_state(classification_extractions=[config])
        fsm = _make_fsm({"triage": state, "end": _terminal_state("end")})
        pipeline = _make_pipeline(fsm)
        instance = _make_instance()

        result = _mock_classify_result("negative", 0.9)

        with patch.object(Classifier, "classify", return_value=result):
            response = pipeline._execute_data_extraction(
                instance, "I'm frustrated!", "conv1"
            )

        assert response.extracted_data["sentiment"] == "negative"

    def test_no_extractions_returns_empty(self):
        state = _make_state()  # no field or classification extractions
        fsm = _make_fsm({"triage": state, "end": _terminal_state("end")})
        pipeline = _make_pipeline(fsm)
        instance = _make_instance()

        response = pipeline._execute_data_extraction(instance, "hello", "conv1")
        assert response.extracted_data == {}
        assert response.confidence == 1.0


# ----------------------------------------------------------
# FSM JSON format test
# ----------------------------------------------------------


class TestFSMJsonFormat:
    def test_fsm_with_classification_extractions_from_dict(self):
        """Verify a full FSM definition with classification_extractions parses."""
        fsm_dict = {
            "name": "sentiment_bot",
            "description": "Routes by sentiment",
            "initial_state": "triage",
            "states": {
                "triage": {
                    "id": "triage",
                    "description": "Classify sentiment",
                    "purpose": "Determine mood",
                    "response_instructions": "Respond",
                    "classification_extractions": [
                        {
                            "field_name": "sentiment",
                            "intents": [
                                {"name": "positive", "description": "Happy"},
                                {"name": "negative", "description": "Angry"},
                                {"name": "neutral", "description": "Neutral"},
                            ],
                            "fallback_intent": "neutral",
                            "confidence_threshold": 0.7,
                        }
                    ],
                    "transitions": [
                        {
                            "target_state": "escalate",
                            "description": "Negative sentiment",
                            "conditions": [
                                {
                                    "description": "Sentiment is negative",
                                    "requires_context_keys": ["sentiment"],
                                    "logic": {"==": [{"var": "sentiment"}, "negative"]},
                                }
                            ],
                        },
                        {
                            "target_state": "done",
                            "description": "Any other sentiment",
                        },
                    ],
                },
                "escalate": {
                    "id": "escalate",
                    "description": "Escalation",
                    "purpose": "Handle complaints",
                    "response_instructions": "Apologize",
                    "transitions": [],
                },
                "done": {
                    "id": "done",
                    "description": "Done",
                    "purpose": "Wrap up",
                    "response_instructions": "Goodbye",
                    "transitions": [],
                },
            },
        }

        fsm = FSMDefinition(**fsm_dict)
        triage = fsm.states["triage"]
        assert len(triage.classification_extractions) == 1
        assert triage.classification_extractions[0].field_name == "sentiment"
        assert len(triage.classification_extractions[0].intents) == 3
