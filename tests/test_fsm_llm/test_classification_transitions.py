"""
Tests for classification-aware transition resolution.

Tests that when a State has ``transition_classification`` enabled, the
MessagePipeline uses Classifier (from fsm_llm_classification) to resolve
AMBIGUOUS transitions instead of the raw LLM prompt.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

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
    TransitionDecisionResponse,
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
    TransitionPromptBuilder,
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


def _make_option(target: str, description: str = "", priority: int = 100) -> TransitionOption:
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
        transition_prompt_builder=TransitionPromptBuilder(),
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

    def test_set_to_true(self):
        state = _make_state("s1", transition_classification=True)
        assert state.transition_classification is True

    def test_set_to_false(self):
        state = _make_state("s1", transition_classification=False)
        assert state.transition_classification is False

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
                transition_classification=True,
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
        assert fsm.states["start"].transition_classification is True

    def test_json_roundtrip(self):
        """transition_classification survives JSON serialization."""
        state = _make_state("s1", transition_classification=True)
        data = state.model_dump()
        restored = State.model_validate(data)
        assert restored.transition_classification is True

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
    """Test transition_classification=True (auto-generate schema from transitions)."""

    def test_auto_mode_uses_classifier(self):
        """When transition_classification=True, ambiguous transitions use Classifier."""
        states = {
            "start": _make_state(
                "start",
                transitions=[
                    _make_transition("billing", "User asks about billing"),
                    _make_transition("support", "User needs tech support"),
                    _make_transition("terminal"),
                ],
                transition_classification=True,
            ),
        }
        fsm_def = _make_fsm_definition(states)
        mock_llm = MagicMock(spec=LLMInterface)
        mock_llm.model = "gpt-4"
        pipeline = _make_pipeline(mock_llm, fsm_def)
        instance = _make_instance(current_state="start")

        evaluation = _make_ambiguous_evaluation("billing", "support")
        mock_result = _mock_classifier_result("billing", 0.92)

        with patch("fsm_llm.pipeline.Classifier", create=True) as mock_cls_cls, \
             patch.dict("sys.modules", {"fsm_llm_classification": MagicMock()}):
            # Patch the lazy import inside the method
            mock_classifier_instance = MagicMock()
            mock_classifier_instance.classify.return_value = mock_result
            mock_cls_cls.return_value = mock_classifier_instance

            # We need to patch the actual import inside _try_classify_transition
            with patch(
                "fsm_llm.pipeline.MessagePipeline._try_classify_transition"
            ) as mock_try:
                mock_try.return_value = "billing"
                result = pipeline._resolve_ambiguous_transition(
                    evaluation, "I have a billing question", DataExtractionResponse(),
                    instance, "conv-1"
                )

            assert result == "billing"

    def test_auto_mode_fallback_to_llm_on_import_error(self):
        """When classification package missing, falls back to raw LLM."""
        states = {
            "start": _make_state(
                "start",
                transitions=[
                    _make_transition("billing", "Billing"),
                    _make_transition("support", "Support"),
                    _make_transition("terminal"),
                ],
                transition_classification=True,
            ),
        }
        fsm_def = _make_fsm_definition(states)
        mock_llm = MagicMock(spec=LLMInterface)
        mock_llm.model = "gpt-4"
        mock_llm.decide_transition.return_value = TransitionDecisionResponse(
            selected_transition="billing", reasoning="LLM decided"
        )
        pipeline = _make_pipeline(mock_llm, fsm_def)
        instance = _make_instance(current_state="start")

        evaluation = _make_ambiguous_evaluation("billing", "support")

        # Simulate classification package not installed
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "fsm_llm_classification":
                raise ImportError("No module named 'fsm_llm_classification'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = pipeline._resolve_ambiguous_transition(
                evaluation, "billing question", DataExtractionResponse(),
                instance, "conv-1"
            )

        assert result == "billing"
        # Should have called LLM decide_transition as fallback
        mock_llm.decide_transition.assert_called_once()


class TestClassificationManualMode:
    """Test transition_classification={...} (user-provided intent descriptions)."""

    def test_manual_mode_custom_descriptions(self):
        """Manual mode uses user-provided descriptions for classification schema."""
        config = {
            "billing": {"description": "User has questions about invoices, payments, or charges"},
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
        fsm_def = _make_fsm_definition(states)
        mock_llm = MagicMock(spec=LLMInterface)
        mock_llm.model = "gpt-4"
        pipeline = _make_pipeline(mock_llm, fsm_def)

        options = [_make_option("billing"), _make_option("support")]

        # Test schema generation directly
        mock_schema_cls = MagicMock()
        mock_intent_cls = MagicMock()
        mock_intent_cls.side_effect = lambda name, description: MagicMock(
            name=name, description=description
        )

        pipeline._build_transition_classification_schema(
            states["start"], options, mock_schema_cls, mock_intent_cls
        )

        # Verify intents were created with custom descriptions
        calls = mock_intent_cls.call_args_list
        descriptions = {c.kwargs["name"]: c.kwargs["description"] for c in calls}
        assert "User has questions about invoices" in descriptions["billing"]
        assert "User needs help with technical" in descriptions["support"]
        assert TRANSITION_CLASSIFICATION_FALLBACK_INTENT in descriptions

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
        fsm_def = _make_fsm_definition(states)
        mock_llm = MagicMock(spec=LLMInterface)
        pipeline = _make_pipeline(mock_llm, fsm_def)

        options = [_make_option("billing"), _make_option("support")]
        mock_schema_cls = MagicMock()
        mock_intent_cls = MagicMock()

        pipeline._build_transition_classification_schema(
            states["start"], options, mock_schema_cls, mock_intent_cls
        )

        # Schema should be created with confidence_threshold=0.8
        mock_schema_cls.assert_called_once()
        call_kwargs = mock_schema_cls.call_args.kwargs
        assert call_kwargs["confidence_threshold"] == 0.8


# ---------------------------------------------------------------------------
# Tests: Schema generation
# ---------------------------------------------------------------------------


class TestBuildTransitionClassificationSchema:
    """Test _build_transition_classification_schema static method."""

    def test_auto_mode_generates_intents_from_options(self):
        state = _make_state("s", transition_classification=True)
        options = [
            _make_option("billing", "User has billing questions"),
            _make_option("support", "User needs technical support"),
        ]

        mock_schema_cls = MagicMock()
        mock_intent_cls = MagicMock()
        mock_intent_cls.side_effect = lambda name, description: MagicMock(
            name=name, description=description
        )

        MessagePipeline._build_transition_classification_schema(
            state, options, mock_schema_cls, mock_intent_cls
        )

        # Should create intents for each option + fallback
        assert mock_intent_cls.call_count == 3  # billing, support, fallback

        # Schema should use default confidence
        mock_schema_cls.assert_called_once()
        assert (
            mock_schema_cls.call_args.kwargs["confidence_threshold"]
            == DEFAULT_TRANSITION_CLASSIFICATION_CONFIDENCE
        )
        assert (
            mock_schema_cls.call_args.kwargs["fallback_intent"]
            == TRANSITION_CLASSIFICATION_FALLBACK_INTENT
        )

    def test_auto_mode_uses_option_description(self):
        state = _make_state("s", transition_classification=True)
        options = [
            _make_option("order_status", "User wants to check their order"),
            _make_option("returns", "User wants to return a product"),
        ]

        intent_descriptions = {}

        def capture_intent(name, description):
            intent_descriptions[name] = description
            return MagicMock(name=name, description=description)

        mock_schema_cls = MagicMock()
        mock_intent_cls = MagicMock(side_effect=capture_intent)

        MessagePipeline._build_transition_classification_schema(
            state, options, mock_schema_cls, mock_intent_cls
        )

        assert intent_descriptions["order_status"] == "User wants to check their order"
        assert intent_descriptions["returns"] == "User wants to return a product"

    def test_auto_mode_uses_existing_descriptions(self):
        """TransitionOption always has a description; auto-mode passes it through."""
        state = _make_state("s", transition_classification=True)
        options = [
            _make_option("a", "Go to a"),
            _make_option("b", "Go to b"),
        ]

        intent_descriptions = {}

        def capture_intent(name, description):
            intent_descriptions[name] = description
            return MagicMock(name=name, description=description)

        mock_schema_cls = MagicMock()
        mock_intent_cls = MagicMock(side_effect=capture_intent)

        MessagePipeline._build_transition_classification_schema(
            state, options, mock_schema_cls, mock_intent_cls
        )

        assert intent_descriptions["a"] == "Go to a"
        assert intent_descriptions["b"] == "Go to b"

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

        intent_descriptions = {}

        def capture_intent(name, description):
            intent_descriptions[name] = description
            return MagicMock(name=name, description=description)

        mock_schema_cls = MagicMock()
        mock_intent_cls = MagicMock(side_effect=capture_intent)

        MessagePipeline._build_transition_classification_schema(
            state, options, mock_schema_cls, mock_intent_cls
        )

        assert intent_descriptions["billing"] == "Custom billing description"
        assert intent_descriptions["support"] == "Default support desc"


# ---------------------------------------------------------------------------
# Tests: Fallback and edge cases
# ---------------------------------------------------------------------------


class TestClassificationFallbackBehavior:
    """Test graceful degradation when classification is unavailable or fails."""

    def test_no_classification_field_uses_llm(self):
        """States without transition_classification always use raw LLM."""
        states = {
            "start": _make_state(
                "start",
                transitions=[
                    _make_transition("a"),
                    _make_transition("b"),
                    _make_transition("terminal"),
                ],
                # No transition_classification
            ),
        }
        fsm_def = _make_fsm_definition(states)
        mock_llm = MagicMock(spec=LLMInterface)
        mock_llm.model = "gpt-4"
        mock_llm.decide_transition.return_value = TransitionDecisionResponse(
            selected_transition="a", reasoning="Mock"
        )
        pipeline = _make_pipeline(mock_llm, fsm_def)
        instance = _make_instance(current_state="start")

        evaluation = _make_ambiguous_evaluation("a", "b")
        result = pipeline._resolve_ambiguous_transition(
            evaluation, "message", DataExtractionResponse(), instance, "conv-1"
        )

        assert result == "a"
        mock_llm.decide_transition.assert_called_once()

    def test_classification_false_uses_llm(self):
        """transition_classification=False explicitly disables classification."""
        states = {
            "start": _make_state(
                "start",
                transitions=[
                    _make_transition("a"),
                    _make_transition("b"),
                    _make_transition("terminal"),
                ],
                transition_classification=False,
            ),
        }
        fsm_def = _make_fsm_definition(states)
        mock_llm = MagicMock(spec=LLMInterface)
        mock_llm.model = "gpt-4"
        mock_llm.decide_transition.return_value = TransitionDecisionResponse(
            selected_transition="a", reasoning="Mock"
        )
        pipeline = _make_pipeline(mock_llm, fsm_def)
        instance = _make_instance(current_state="start")

        evaluation = _make_ambiguous_evaluation("a", "b")
        result = pipeline._resolve_ambiguous_transition(
            evaluation, "message", DataExtractionResponse(), instance, "conv-1"
        )

        # False is falsy but not None — check implementation handles this
        # transition_classification=False should NOT trigger classification
        # because `False is not None` is True but bool False should be treated as disabled
        assert result == "a"

    def test_fallback_intent_triggers_llm_fallback(self):
        """When classification returns fallback intent, fall back to raw LLM."""
        states = {
            "start": _make_state(
                "start",
                transitions=[
                    _make_transition("a"),
                    _make_transition("b"),
                    _make_transition("terminal"),
                ],
                transition_classification=True,
            ),
        }
        fsm_def = _make_fsm_definition(states)
        mock_llm = MagicMock(spec=LLMInterface)
        mock_llm.model = "gpt-4"
        mock_llm.decide_transition.return_value = TransitionDecisionResponse(
            selected_transition="a", reasoning="LLM fallback"
        )
        pipeline = _make_pipeline(mock_llm, fsm_def)
        instance = _make_instance(current_state="start")

        evaluation = _make_ambiguous_evaluation("a", "b")
        mock_result = _mock_classifier_result(
            TRANSITION_CLASSIFICATION_FALLBACK_INTENT, 0.3
        )

        import builtins
        original_import = builtins.__import__

        mock_classifier_instance = MagicMock()
        mock_classifier_instance.classify.return_value = mock_result

        mock_classification_module = MagicMock()
        mock_classification_module.Classifier.return_value = mock_classifier_instance
        mock_classification_module.ClassificationSchema = MagicMock()
        mock_classification_module.IntentDefinition = MagicMock(
            side_effect=lambda name, description: MagicMock(name=name, description=description)
        )
        mock_classification_module.ClassificationResult = MagicMock

        def mock_import(name, *args, **kwargs):
            if name == "fsm_llm_classification":
                return mock_classification_module
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = pipeline._resolve_ambiguous_transition(
                evaluation, "unclear message", DataExtractionResponse(),
                instance, "conv-1"
            )

        assert result == "a"
        # Should have called LLM as fallback
        mock_llm.decide_transition.assert_called_once()

    def test_classification_exception_falls_back_to_llm(self):
        """If classification raises an exception, fall back to raw LLM."""
        states = {
            "start": _make_state(
                "start",
                transitions=[
                    _make_transition("a"),
                    _make_transition("b"),
                    _make_transition("terminal"),
                ],
                transition_classification=True,
            ),
        }
        fsm_def = _make_fsm_definition(states)
        mock_llm = MagicMock(spec=LLMInterface)
        mock_llm.model = "gpt-4"
        mock_llm.decide_transition.return_value = TransitionDecisionResponse(
            selected_transition="b", reasoning="LLM fallback after error"
        )
        pipeline = _make_pipeline(mock_llm, fsm_def)
        instance = _make_instance(current_state="start")

        evaluation = _make_ambiguous_evaluation("a", "b")

        import builtins
        original_import = builtins.__import__

        mock_classifier_instance = MagicMock()
        mock_classifier_instance.classify.side_effect = RuntimeError("LLM call failed")

        mock_classification_module = MagicMock()
        mock_classification_module.Classifier.return_value = mock_classifier_instance
        mock_classification_module.ClassificationSchema = MagicMock()
        mock_classification_module.IntentDefinition = MagicMock(
            side_effect=lambda name, description: MagicMock(name=name, description=description)
        )
        mock_classification_module.ClassificationResult = MagicMock

        def mock_import(name, *args, **kwargs):
            if name == "fsm_llm_classification":
                return mock_classification_module
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = pipeline._resolve_ambiguous_transition(
                evaluation, "message", DataExtractionResponse(),
                instance, "conv-1"
            )

        assert result == "b"
        mock_llm.decide_transition.assert_called_once()

    def test_no_model_attribute_falls_back_to_llm(self):
        """If LLM interface has no model attribute, fall back to raw LLM."""
        states = {
            "start": _make_state(
                "start",
                transitions=[
                    _make_transition("a"),
                    _make_transition("b"),
                    _make_transition("terminal"),
                ],
                transition_classification=True,
            ),
        }
        fsm_def = _make_fsm_definition(states)
        mock_llm = MagicMock(spec=LLMInterface)
        # Remove model attribute
        del mock_llm.model
        mock_llm.decide_transition.return_value = TransitionDecisionResponse(
            selected_transition="a", reasoning="No model fallback"
        )
        pipeline = _make_pipeline(mock_llm, fsm_def)
        instance = _make_instance(current_state="start")

        evaluation = _make_ambiguous_evaluation("a", "b")

        import builtins
        original_import = builtins.__import__

        mock_classification_module = MagicMock()

        def mock_import(name, *args, **kwargs):
            if name == "fsm_llm_classification":
                return mock_classification_module
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = pipeline._resolve_ambiguous_transition(
                evaluation, "message", DataExtractionResponse(),
                instance, "conv-1"
            )

        assert result == "a"
        mock_llm.decide_transition.assert_called_once()


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
                transition_classification=True,
            ),
        }
        fsm_def = _make_fsm_definition(states)
        mock_llm = MagicMock(spec=LLMInterface)
        mock_llm.model = "gpt-4"
        pipeline = _make_pipeline(mock_llm, fsm_def)
        instance = _make_instance(current_state="start")

        evaluation = _make_ambiguous_evaluation("billing", "support")
        mock_result = _mock_classifier_result("billing", 0.95)

        import builtins
        original_import = builtins.__import__

        mock_classifier_instance = MagicMock()
        mock_classifier_instance.classify.return_value = mock_result

        mock_classification_module = MagicMock()
        mock_classification_module.Classifier.return_value = mock_classifier_instance
        mock_classification_module.ClassificationSchema = MagicMock()
        mock_classification_module.IntentDefinition = MagicMock(
            side_effect=lambda name, description: MagicMock(name=name, description=description)
        )
        mock_classification_module.ClassificationResult = MagicMock

        def mock_import(name, *args, **kwargs):
            if name == "fsm_llm_classification":
                return mock_classification_module
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = pipeline._resolve_ambiguous_transition(
                evaluation, "billing question", DataExtractionResponse(),
                instance, "conv-1"
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
