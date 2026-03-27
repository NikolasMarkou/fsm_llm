"""
Tests for classification-aware transition resolution.

Tests that when a State has ``transition_classification`` enabled, the
MessagePipeline uses Classifier (from fsm_llm_classification) to resolve
AMBIGUOUS transitions instead of the raw LLM prompt.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch


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


from fsm_llm.constants import (
    CONTEXT_KEY_CLASSIFICATION_RESULT,
    DEFAULT_TRANSITION_CLASSIFICATION_CONFIDENCE,
    TRANSITION_CLASSIFICATION_FALLBACK_INTENT,
)
from fsm_llm.definitions import (
    DataExtractionResponse,
    FSMContext,
    FSMDefinition,
    FSMInstance,
    State,
    Transition,
    TransitionEvaluation,
    TransitionEvaluationResult,
    TransitionOption,
)
from fsm_llm.handlers import HandlerSystem
from fsm_llm.llm import LLMInterface
from fsm_llm.pipeline import MessagePipeline
from fsm_llm.prompts import (
    DataExtractionPromptBuilder,
    ResponseGenerationPromptBuilder,
)
from fsm_llm.transition_evaluator import TransitionEvaluator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(
    state_id: str,
    transitions: list[Transition] | None = None,
    transition_classification: bool | dict[str, Any] | None = None,
) -> State:
    return State(
        id=state_id,
        description=f"State {state_id}",
        purpose=f"Purpose of {state_id}",
        extraction_instructions="Extract data",
        response_instructions="Respond",
        transitions=transitions or [],
        transition_classification=transition_classification,
    )


def _make_transition(
    target: str,
    description: str = "",
    priority: int = 100,
    llm_description: str | None = None,
) -> Transition:
    return Transition(
        target_state=target,
        description=description or f"Go to {target}",
        priority=priority,
        llm_description=llm_description,
    )


def _make_option(
    target: str, description: str = "", priority: int = 100
) -> TransitionOption:
    return TransitionOption(
        target_state=target,
        description=description or f"Go to {target}",
        priority=priority,
    )


def _make_ambiguous_evaluation(*targets: str) -> TransitionEvaluation:
    options = [_make_option(t, f"Go to {t}") for t in targets]
    return TransitionEvaluation(
        result_type=TransitionEvaluationResult.AMBIGUOUS,
        available_options=options,
        confidence=0.5,
    )


def _make_fsm_definition(states: dict[str, State]) -> FSMDefinition:
    """Create a minimal FSM definition with given states.

    Automatically creates terminal placeholder states for any transition
    targets not already defined in ``states``.
    """
    all_states = dict(states)

    # Collect all transition targets and create missing states as terminals
    for state in states.values():
        for t in state.transitions:
            if t.target_state not in all_states:
                all_states[t.target_state] = State(
                    id=t.target_state,
                    description=f"Terminal {t.target_state}",
                    purpose="End",
                    transitions=[],
                )

    # Ensure at least one terminal state
    if not any(not s.transitions for s in all_states.values()):
        all_states["terminal"] = State(
            id="terminal", description="Terminal", purpose="End", transitions=[]
        )

    initial = next(iter(states))
    return FSMDefinition(
        name="test_fsm",
        description="Test FSM",
        initial_state=initial,
        states=all_states,
    )


def _make_pipeline(
    llm_interface: LLMInterface,
    fsm_definition: FSMDefinition,
) -> MessagePipeline:
    return MessagePipeline(
        llm_interface=llm_interface,
        data_extraction_prompt_builder=DataExtractionPromptBuilder(),
        response_generation_prompt_builder=ResponseGenerationPromptBuilder(),
        transition_evaluator=TransitionEvaluator(),
        handler_system=HandlerSystem(),
        fsm_resolver=lambda fsm_id: fsm_definition,
    )


def _make_instance(
    fsm_id: str = "test_fsm",
    current_state: str = "start",
) -> FSMInstance:
    return FSMInstance(
        fsm_id=fsm_id,
        current_state=current_state,
        context=FSMContext(),
    )


def _mock_classifier_result(intent: str, confidence: float = 0.9):
    """Create a mock ClassificationResult."""
    result = MagicMock()
    result.intent = intent
    result.confidence = confidence
    result.reasoning = f"Selected {intent}"
    result.entities = {}
    result.is_low_confidence = confidence < DEFAULT_TRANSITION_CLASSIFICATION_CONFIDENCE
    return result


# ---------------------------------------------------------------------------
# Tests: State model field
# ---------------------------------------------------------------------------


class TestStateTransitionClassificationField:
    """Test the transition_classification field on State model."""

    def test_default_is_none(self):
        state = _make_state("s1")
        assert state.transition_classification is None

    def test_set_to_dict(self):
        config = {
            "billing": {"description": "User has billing questions"},
            "support": {"description": "User needs technical support"},
        }
        state = _make_state("s1", transition_classification=config)
        assert state.transition_classification == config

    def test_fsm_definition_with_classification_field(self):
        """FSMDefinition accepts states with transition_classification."""
        states = {
            "start": _make_state(
                "start",
                transitions=[_make_transition("terminal")],
                transition_classification=None,
            ),
            "terminal": State(
                id="terminal", description="End", purpose="End", transitions=[]
            ),
        }
        fsm = FSMDefinition(
            name="test",
            description="Test",
            initial_state="start",
            states=states,
        )
        assert fsm.states["start"].transition_classification is None

    def test_json_roundtrip(self):
        """transition_classification survives JSON serialization."""
        state = _make_state("s1", transition_classification=None)
        data = state.model_dump()
        restored = State.model_validate(data)
        assert restored.transition_classification is None

    def test_json_roundtrip_dict(self):
        config = {"target_a": {"description": "Option A"}}
        state = _make_state("s1", transition_classification=config)
        data = state.model_dump()
        restored = State.model_validate(data)
        assert restored.transition_classification == config


# ---------------------------------------------------------------------------
# Tests: Auto-mode classification
# ---------------------------------------------------------------------------


class TestClassificationAutoMode:
    """Test transition_classification=None (auto-generate schema from transitions).

    Classification is always-on and part of core. None means auto-generation
    from transition descriptions.
    """

    def test_auto_mode_uses_classifier(self):
        """Ambiguous transitions use Classifier (classification is always-on)."""
        states = {
            "start": _make_state(
                "start",
                transitions=[
                    _make_transition("billing", "User asks about billing"),
                    _make_transition("support", "User needs tech support"),
                    _make_transition("terminal"),
                ],
                transition_classification=None,
            ),
        }
        fsm_def = _make_fsm_definition(states)
        mock_llm = MagicMock(spec=LLMInterface)
        configure_mock_extract_field(mock_llm)
        mock_llm.model = "gpt-4"
        pipeline = _make_pipeline(mock_llm, fsm_def)
        instance = _make_instance(current_state="start")

        evaluation = _make_ambiguous_evaluation("billing", "support")
        mock_result = _mock_classifier_result("billing", 0.92)

        with patch("fsm_llm.pipeline.Classifier") as mock_cls_cls:
            mock_classifier_instance = MagicMock()
            mock_classifier_instance.classify.return_value = mock_result
            mock_cls_cls.return_value = mock_classifier_instance

            result = pipeline._resolve_ambiguous_transition(
                evaluation,
                "I have a billing question",
                DataExtractionResponse(),
                instance,
                "conv-1",
            )

        assert result == "billing"


class TestClassificationManualMode:
    """Test transition_classification={...} (user-provided intent descriptions)."""

    def test_manual_mode_custom_descriptions(self):
        """Manual mode uses user-provided descriptions for classification schema."""
        config = {
            "billing": {
                "description": "User has questions about invoices, payments, or charges"
            },
            "support": {"description": "User needs help with technical issues or bugs"},
        }
        states = {
            "start": _make_state(
                "start",
                transitions=[
                    _make_transition("billing"),
                    _make_transition("support"),
                    _make_transition("terminal"),
                ],
                transition_classification=config,
            ),
        }

        options = [_make_option("billing"), _make_option("support")]

        schema = MessagePipeline._build_transition_classification_schema(
            states["start"], options
        )

        # Verify intents were created with custom descriptions
        intent_map = {i.name: i.description for i in schema.intents}
        assert "User has questions about invoices" in intent_map["billing"]
        assert "User needs help with technical" in intent_map["support"]
        assert TRANSITION_CLASSIFICATION_FALLBACK_INTENT in intent_map

    def test_manual_mode_custom_confidence_threshold(self):
        """Manual mode respects custom confidence_threshold."""
        config = {
            "billing": {"description": "Billing stuff"},
            "support": {"description": "Support stuff"},
            "confidence_threshold": 0.8,
        }
        states = {
            "start": _make_state(
                "start",
                transitions=[
                    _make_transition("billing"),
                    _make_transition("support"),
                    _make_transition("terminal"),
                ],
                transition_classification=config,
            ),
        }

        options = [_make_option("billing"), _make_option("support")]

        schema = MessagePipeline._build_transition_classification_schema(
            states["start"], options
        )

        assert schema.confidence_threshold == 0.8


# ---------------------------------------------------------------------------
# Tests: Schema generation
# ---------------------------------------------------------------------------


class TestBuildTransitionClassificationSchema:
    """Test _build_transition_classification_schema static method."""

    def test_auto_mode_generates_intents_from_options(self):
        state = _make_state("s", transition_classification=None)
        options = [
            _make_option("billing", "User has billing questions"),
            _make_option("support", "User needs technical support"),
        ]

        schema = MessagePipeline._build_transition_classification_schema(state, options)

        # Should create intents for each option + fallback
        assert len(schema.intents) == 3  # billing, support, fallback

        # Schema should use default confidence
        assert (
            schema.confidence_threshold == DEFAULT_TRANSITION_CLASSIFICATION_CONFIDENCE
        )
        assert schema.fallback_intent == TRANSITION_CLASSIFICATION_FALLBACK_INTENT

    def test_auto_mode_uses_option_description(self):
        state = _make_state("s", transition_classification=None)
        options = [
            _make_option("order_status", "User wants to check their order"),
            _make_option("returns", "User wants to return a product"),
        ]

        schema = MessagePipeline._build_transition_classification_schema(state, options)

        intent_map = {i.name: i.description for i in schema.intents}
        assert intent_map["order_status"] == "User wants to check their order"
        assert intent_map["returns"] == "User wants to return a product"

    def test_auto_mode_uses_existing_descriptions(self):
        """TransitionOption always has a description; auto-mode passes it through."""
        state = _make_state("s", transition_classification=None)
        options = [
            _make_option("a", "Go to a"),
            _make_option("b", "Go to b"),
        ]

        schema = MessagePipeline._build_transition_classification_schema(state, options)

        intent_map = {i.name: i.description for i in schema.intents}
        assert intent_map["a"] == "Go to a"
        assert intent_map["b"] == "Go to b"

    def test_manual_mode_merges_custom_and_default_descriptions(self):
        config = {
            "billing": {"description": "Custom billing description"},
            # "support" not in config — should use option description
        }
        state = _make_state("s", transition_classification=config)
        options = [
            _make_option("billing", "Default billing desc"),
            _make_option("support", "Default support desc"),
        ]

        schema = MessagePipeline._build_transition_classification_schema(state, options)

        intent_map = {i.name: i.description for i in schema.intents}
        assert intent_map["billing"] == "Custom billing description"
        assert intent_map["support"] == "Default support desc"


# ---------------------------------------------------------------------------
# Tests: Fallback and edge cases
# ---------------------------------------------------------------------------


class TestClassificationFallbackBehavior:
    """Test behavior when classification is disabled or returns fallback.

    Note: Classification is now always-on (absorbed into core). The old
    LLM decide_transition fallback path has been removed. These tests
    verify the current classification-based behavior.
    """

    def test_no_classification_field_uses_classification(self):
        """States without transition_classification use classification by default."""
        states = {
            "start": _make_state(
                "start",
                transitions=[
                    _make_transition("a"),
                    _make_transition("b"),
                    _make_transition("terminal"),
                ],
                # No transition_classification — classification is always-on
            ),
        }
        fsm_def = _make_fsm_definition(states)
        mock_llm = MagicMock(spec=LLMInterface)
        configure_mock_extract_field(mock_llm)
        mock_llm.model = "gpt-4"
        pipeline = _make_pipeline(mock_llm, fsm_def)
        instance = _make_instance(current_state="start")

        evaluation = _make_ambiguous_evaluation("a", "b")
        mock_result = _mock_classifier_result("a", 0.9)

        with patch("fsm_llm.pipeline.Classifier") as mock_cls:
            mock_classifier_instance = MagicMock()
            mock_classifier_instance.classify.return_value = mock_result
            mock_cls.return_value = mock_classifier_instance

            result = pipeline._resolve_ambiguous_transition(
                evaluation, "message", DataExtractionResponse(), instance, "conv-1"
            )

        assert result == "a"


# ---------------------------------------------------------------------------
# Tests: Context storage
# ---------------------------------------------------------------------------


class TestClassificationContextStorage:
    """Test that classification results are stored in instance context."""

    def test_classification_result_stored_in_context(self):
        """Successful classification stores result in context for debugging."""
        states = {
            "start": _make_state(
                "start",
                transitions=[
                    _make_transition("billing"),
                    _make_transition("support"),
                    _make_transition("terminal"),
                ],
                transition_classification=None,
            ),
        }
        fsm_def = _make_fsm_definition(states)
        mock_llm = MagicMock(spec=LLMInterface)
        configure_mock_extract_field(mock_llm)
        mock_llm.model = "gpt-4"
        pipeline = _make_pipeline(mock_llm, fsm_def)
        instance = _make_instance(current_state="start")

        evaluation = _make_ambiguous_evaluation("billing", "support")
        mock_result = _mock_classifier_result("billing", 0.95)

        with patch("fsm_llm.pipeline.Classifier") as mock_cls:
            mock_classifier_instance = MagicMock()
            mock_classifier_instance.classify.return_value = mock_result
            mock_cls.return_value = mock_classifier_instance

            result = pipeline._resolve_ambiguous_transition(
                evaluation,
                "billing question",
                DataExtractionResponse(),
                instance,
                "conv-1",
            )

        assert result == "billing"
        assert CONTEXT_KEY_CLASSIFICATION_RESULT in instance.context.data
        stored = instance.context.data[CONTEXT_KEY_CLASSIFICATION_RESULT]
        assert stored["intent"] == "billing"
        assert stored["confidence"] == 0.95
        assert stored["reasoning"] == "Selected billing"


# ---------------------------------------------------------------------------
# Tests: Constants
# ---------------------------------------------------------------------------


class TestClassificationTransitionConstants:
    """Test classification transition constants."""

    def test_default_confidence(self):
        assert DEFAULT_TRANSITION_CLASSIFICATION_CONFIDENCE == 0.6

    def test_fallback_intent_is_internal(self):
        assert TRANSITION_CLASSIFICATION_FALLBACK_INTENT.startswith("_")

    def test_context_key_is_internal(self):
        assert CONTEXT_KEY_CLASSIFICATION_RESULT.startswith("_")
