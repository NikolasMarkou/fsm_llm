"""
Tests for classification-aware transition resolution.

Tests that when a State has ``transition_classification`` enabled, the
MessagePipeline uses Classifier (from fsm_llm.classification) to resolve
AMBIGUOUS transitions instead of the raw LLM prompt.
"""

from __future__ import annotations

import json
import sys
import threading
from typing import Any
from unittest.mock import MagicMock, patch

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


from fsm_llm.api import API
from fsm_llm.classification import Classifier
from fsm_llm.constants import (
    DEFAULT_LLM_MODEL,
    DEFAULT_TRANSITION_CLASSIFICATION_CONFIDENCE,
    MAX_CLASSIFIER_CACHE_SIZE,
    METADATA_KEY_TRANSITION_CLASSIFICATION,
    TRANSITION_CLASSIFICATION_FALLBACK_INTENT,
)
from fsm_llm.definitions import (
    ClassificationError,
    ClassificationExtractionConfig,
    ClassificationResult,
    ClassificationSchema,
    CompletionRequest,
    CompletionResponse,
    DataExtractionResponse,
    FieldExtractionRequest,
    FieldExtractionResponse,
    FSMContext,
    FSMDefinition,
    FSMError,
    FSMInstance,
    IntentDefinition,
    ResponseGenerationRequest,
    ResponseGenerationResponse,
    State,
    Transition,
    TransitionEvaluation,
    TransitionEvaluationResult,
    TransitionOption,
)
from fsm_llm.handlers import HandlerSystem, HandlerTiming
from fsm_llm.llm import LiteLLMInterface, LLMInterface
from fsm_llm.pipeline import MessagePipeline
from fsm_llm.prompts import (
    ClassificationPromptConfig,
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
        stored = instance.context.metadata[METADATA_KEY_TRANSITION_CLASSIFICATION]
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


# ---------------------------------------------------------------------------
# Tests: exception discipline at the ambiguous-transition classifier call
# ---------------------------------------------------------------------------


def _ambiguous_fsm() -> FSMDefinition:
    """A start state with two unconditioned transitions -> AMBIGUOUS each turn."""
    return _make_fsm_definition(
        {
            "start": _make_state(
                "start",
                transitions=[_make_transition("a"), _make_transition("b")],
            ),
        }
    )


def _api_with_failing_classifier(exc: BaseException):
    """Build an API on the ambiguous FSM whose transition classifier raises ``exc``.

    Returns ``(api, conv_id, patcher)``; the caller enters ``patcher`` around
    ``api.converse`` so the exception is raised inside the public turn path.
    """
    mock_llm = MagicMock(spec=LLMInterface)
    configure_mock_extract_field(mock_llm)
    mock_llm.model = "gpt-4"
    mock_llm.generate_response.return_value = MagicMock(
        message="ok", message_type="response", reasoning="mock"
    )
    api = API.from_definition(_ambiguous_fsm(), llm_interface=mock_llm)
    conv_id, _ = api.start_conversation()
    patcher = patch("fsm_llm.pipeline.Classifier")
    return api, conv_id, patcher, exc


class TestAmbiguousTransitionExceptionDiscipline:
    """``_resolve_ambiguous_transition`` degrades to a stay ONLY for the shared
    soft-fail tuple; programming errors and BaseException propagate out of
    ``API.converse``. Pins plan-2026-09-20T165703-0d9c218e D-001.
    """

    @pytest.mark.parametrize(
        "exc",
        [
            ClassificationError("classifier outage"),
            ValueError("bad payload"),
            RuntimeError("transport hiccup"),
        ],
        ids=["ClassificationError", "ValueError", "RuntimeError"],
    )
    def test_soft_fail_classes_stay_in_state_with_fallback_marker(self, exc):
        api, conv_id, patcher, exc = _api_with_failing_classifier(exc)
        with patcher as mock_cls:
            mock_cls.return_value.classify.side_effect = exc
            response = api.converse("which one?", conv_id)

        assert mock_cls.return_value.classify.called
        assert isinstance(response, str)
        assert api.get_current_state(conv_id) == "start"
        instance = api.fsm_manager.instances[conv_id]
        stored = instance.context.metadata[METADATA_KEY_TRANSITION_CLASSIFICATION]
        assert stored["fallback"] is True
        assert str(exc) in stored["error"]

    @pytest.mark.parametrize(
        "exc",
        [AttributeError("no such attr"), ZeroDivisionError("division by zero")],
        ids=["AttributeError", "ZeroDivisionError"],
    )
    def test_programming_errors_propagate_out_of_converse(self, exc):
        """Not swallowed into a stay. ``FSMManager.process_message`` wraps any
        non-FSMError into ``FSMError`` (``fsm.py`` ``raise FSMError(...) from e``),
        so the public surface is an ``FSMError`` whose ``__cause__`` is the
        original programming error.
        """
        api, conv_id, patcher, exc = _api_with_failing_classifier(exc)
        with patcher as mock_cls:
            mock_cls.return_value.classify.side_effect = exc
            with pytest.raises(FSMError) as info:
                api.converse("which one?", conv_id)

        assert info.value.__cause__ is exc
        assert mock_cls.return_value.classify.called
        assert api.get_current_state(conv_id) == "start"
        instance = api.fsm_manager.instances[conv_id]
        # No fallback marker: the failure was not degraded to a stay.
        assert METADATA_KEY_TRANSITION_CLASSIFICATION not in instance.context.metadata

    def test_construction_failure_degrades_to_stay(self):
        """A ``Classifier(...)`` CONSTRUCTION failure at the transition site is
        covered by the same soft-fail try as ``classify()``: the turn stays in
        state instead of escaping as ``FSMError``. RED on the pre-step-2 code,
        where the ``_get_classifier`` call sat outside the try (review W2).
        """
        exc = ValueError("schema rejected at construction")
        api, conv_id, patcher, exc = _api_with_failing_classifier(exc)
        with patcher as mock_cls:
            mock_cls.side_effect = exc
            response = api.converse("which one?", conv_id)

        assert mock_cls.called
        assert isinstance(response, str)
        assert api.get_current_state(conv_id) == "start"
        instance = api.fsm_manager.instances[conv_id]
        stored = instance.context.metadata[METADATA_KEY_TRANSITION_CLASSIFICATION]
        assert stored["fallback"] is True
        assert str(exc) in stored["error"]

    def test_keyboard_interrupt_propagates_bare(self):
        api, conv_id, patcher, exc = _api_with_failing_classifier(KeyboardInterrupt())
        with patcher as mock_cls:
            mock_cls.return_value.classify.side_effect = exc
            with pytest.raises(KeyboardInterrupt):
                api.converse("which one?", conv_id)

        assert mock_cls.return_value.classify.called


# ---------------------------------------------------------------------------
# Tests: bounded per-pipeline Classifier cache
# ---------------------------------------------------------------------------


def _stay_result() -> ClassificationResult:
    """A real ``ClassificationResult`` that keeps the conversation in its state."""
    return ClassificationResult(
        reasoning="mock",
        intent=TRANSITION_CLASSIFICATION_FALLBACK_INTENT,
        confidence=0.9,
    )


def _api_on(fsm_def: FSMDefinition):
    mock_llm = MagicMock(spec=LLMInterface)
    configure_mock_extract_field(mock_llm)
    mock_llm.model = "gpt-4"
    mock_llm.generate_response.return_value = MagicMock(
        message="ok", message_type="response", reasoning="mock"
    )
    api = API.from_definition(fsm_def, llm_interface=mock_llm)
    conv_id, _ = api.start_conversation()
    return api, conv_id


def _schema(*names: str) -> ClassificationSchema:
    intents = [IntentDefinition(name=n, description=f"Intent {n}") for n in names]
    return ClassificationSchema(
        intents=intents, fallback_intent=names[0], confidence_threshold=0.6
    )


class TestClassifierCache:
    """``MessagePipeline._get_classifier`` reuses one ``Classifier`` per key
    (schema + prompt config + override model + the identity of the injected
    conversation interface), bounded at ``MAX_CLASSIFIER_CACHE_SIZE``. Pins
    plan-2026-09-20T165703-0d9c218e D-001 as amended by D-006 of plan
    944e2692 (connection kwargs left the key with the private interface).
    """

    def test_same_state_over_two_turns_constructs_once(self):
        api, conv_id = _api_on(_ambiguous_fsm())
        with patch("fsm_llm.pipeline.Classifier") as mock_cls:
            mock_cls.return_value.classify.return_value = _stay_result()
            api.converse("first", conv_id)
            api.converse("second", conv_id)

        assert mock_cls.call_count == 1
        assert mock_cls.return_value.classify.call_count == 2
        assert api.get_current_state(conv_id) == "start"

    def test_extra_intent_in_schema_constructs_again(self):
        fsm_def = _ambiguous_fsm()
        mock_llm = MagicMock(spec=LLMInterface)
        mock_llm.model = "gpt-4"
        pipeline = _make_pipeline(mock_llm, fsm_def)
        instance = _make_instance(current_state="start")

        with patch("fsm_llm.pipeline.Classifier") as mock_cls:
            mock_cls.return_value.classify.return_value = _stay_result()
            for evaluation in (
                _make_ambiguous_evaluation("a", "b"),
                _make_ambiguous_evaluation("a", "b", "c"),
                _make_ambiguous_evaluation("a", "b"),
            ):
                pipeline._resolve_ambiguous_transition(
                    evaluation, "msg", DataExtractionResponse(), instance, "conv-1"
                )

        assert mock_cls.call_count == 2
        assert len(pipeline._classifier_cache) == 2

    def test_model_override_constructs_again(self):
        intents = [
            IntentDefinition(name="yes", description="Yes"),
            IntentDefinition(name="no", description="No"),
        ]
        base = ClassificationExtractionConfig(
            field_name="answer", intents=intents, fallback_intent="no"
        )
        override = ClassificationExtractionConfig(
            field_name="answer_alt",
            intents=intents,
            fallback_intent="no",
            model="gpt-4o",
        )
        state = State(
            id="start",
            description="Start",
            purpose="Ask",
            classification_extractions=[base, override],
        )
        fsm_def = _make_fsm_definition({"start": state})
        mock_llm = MagicMock(spec=LLMInterface)
        mock_llm.model = "gpt-4"
        pipeline = _make_pipeline(mock_llm, fsm_def)
        instance = _make_instance(current_state="start")

        with patch("fsm_llm.pipeline.Classifier") as mock_cls:
            mock_cls.return_value.classify.return_value = ClassificationResult(
                reasoning="mock", intent="yes", confidence=0.9
            )
            pipeline._execute_classification_extractions(state, "yes", instance, "c")
            pipeline._execute_classification_extractions(state, "yes", instance, "c")

        assert mock_cls.call_count == 2
        # The entry without an override sends through the conversation's
        # interface; the other model gets its own, built from the name only.
        assert [
            (c.kwargs.get("model"), c.kwargs.get("llm"))
            for c in mock_cls.call_args_list
        ] == [(None, mock_llm), ("gpt-4o", None)]
        assert mock_cls.return_value.classify.call_count == 4

    def test_cache_is_bounded_and_evicts_oldest(self):
        mock_llm = MagicMock(spec=LLMInterface)
        mock_llm.model = "gpt-4"
        pipeline = _make_pipeline(mock_llm, _ambiguous_fsm())
        assert MAX_CLASSIFIER_CACHE_SIZE == 64

        with patch("fsm_llm.pipeline.Classifier") as mock_cls:
            mock_cls.side_effect = lambda **kwargs: MagicMock(name="clf")
            first = pipeline._get_classifier(_schema("i0", "j0"), None)
            first_key = next(iter(pipeline._classifier_cache))
            for n in range(1, MAX_CLASSIFIER_CACHE_SIZE + 1):
                pipeline._get_classifier(_schema(f"i{n}", f"j{n}"), None)

        assert mock_cls.call_count == MAX_CLASSIFIER_CACHE_SIZE + 1
        assert len(pipeline._classifier_cache) == MAX_CLASSIFIER_CACHE_SIZE
        assert first_key not in pipeline._classifier_cache
        assert first not in pipeline._classifier_cache.values()

    def test_eviction_race_two_threads(self):
        """Concurrent misses past the bound must not raise. RED on the
        pre-step-2 code: the unlocked ``cache.pop(next(iter(cache)))`` raced a
        concurrent insert into ``RuntimeError: dictionary changed size during
        iteration`` (review W2). Pins the ``_classifier_cache_lock``.
        """
        mock_llm = MagicMock(spec=LLMInterface)
        mock_llm.model = "gpt-4"
        pipeline = _make_pipeline(mock_llm, _ambiguous_fsm())
        errors: list[BaseException] = []
        n_threads, n_calls = 4, 2000

        def worker(tid: int) -> None:
            try:
                for n in range(n_calls):
                    pipeline._get_classifier(
                        _schema(f"t{tid}_i{n}", f"t{tid}_j{n}"), None
                    )
            except BaseException as e:  # collected for the assert below
                errors.append(e)

        old_interval = sys.getswitchinterval()
        try:
            with patch("fsm_llm.pipeline.Classifier") as mock_cls:
                mock_cls.side_effect = lambda **kwargs: object()
                for n in range(MAX_CLASSIFIER_CACHE_SIZE):
                    pipeline._get_classifier(_schema(f"p{n}", f"q{n}"), None)
                assert len(pipeline._classifier_cache) == MAX_CLASSIFIER_CACHE_SIZE
                sys.setswitchinterval(1e-6)
                threads = [
                    threading.Thread(target=worker, args=(t,)) for t in range(n_threads)
                ]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join(timeout=10)
        finally:
            sys.setswitchinterval(old_interval)

        # Pre-Mortem 2: a deadlocked worker must fail the test, not hang it.
        assert all(not t.is_alive() for t in threads)
        assert errors == []
        assert len(pipeline._classifier_cache) == MAX_CLASSIFIER_CACHE_SIZE

    def test_both_sites_share_one_cache(self):
        """An extraction entry whose schema equals the auto-built transition
        schema of the same state yields ONE construction across both sites."""
        extraction = ClassificationExtractionConfig(
            field_name="route",
            intents=[
                IntentDefinition(name="a", description="Go to a"),
                IntentDefinition(name="b", description="Go to b"),
                IntentDefinition(
                    name=TRANSITION_CLASSIFICATION_FALLBACK_INTENT,
                    description="None of the above options clearly match the user's intent",
                ),
            ],
            fallback_intent=TRANSITION_CLASSIFICATION_FALLBACK_INTENT,
            confidence_threshold=DEFAULT_TRANSITION_CLASSIFICATION_CONFIDENCE,
        )
        state = State(
            id="start",
            description="Start",
            purpose="Route",
            extraction_instructions="Extract data",
            response_instructions="Respond",
            transitions=[_make_transition("a"), _make_transition("b")],
            classification_extractions=[extraction],
        )
        api, conv_id = _api_on(_make_fsm_definition({"start": state}))
        with patch("fsm_llm.pipeline.Classifier") as mock_cls:
            mock_cls.return_value.classify.return_value = _stay_result()
            api.converse("which one?", conv_id)

        assert mock_cls.return_value.classify.call_count == 2
        assert mock_cls.call_count == 1
        assert len(api.fsm_manager._pipeline._classifier_cache) == 1

    def test_same_interface_caches(self):
        """Two lookups over the same conversation interface: one
        construction, one entry, the classifier holds that interface."""
        mock_llm = MagicMock(spec=LLMInterface)
        mock_llm.model = "gpt-4"
        pipeline = _make_pipeline(mock_llm, _ambiguous_fsm())
        schema = _schema("a", "b")

        with patch("fsm_llm.pipeline.Classifier") as mock_cls:
            mock_cls.side_effect = lambda **kwargs: MagicMock(name="clf")
            first = pipeline._get_classifier(schema, None)
            second = pipeline._get_classifier(schema, None)

        assert mock_cls.call_count == 1
        assert first is second
        assert mock_cls.call_args.kwargs["llm"] is mock_llm
        assert len(pipeline._classifier_cache) == 1

    def test_prompt_config_change_constructs_again(self):
        """Same schema and model, ``temperature=0.1`` vs ``temperature=0.2``
        -> two constructions and two entries. PASSES by design today (the
        key already hashes ``dataclasses.asdict(prompt_config)``, D-001);
        this pin guards a future key simplification that drops the prompt
        config and would silently reuse a classifier built for the old
        temperature (review W4).
        """
        mock_llm = MagicMock(spec=LLMInterface)
        mock_llm.model = "gpt-4"
        pipeline = _make_pipeline(mock_llm, _ambiguous_fsm())
        schema = _schema("a", "b")

        with patch("fsm_llm.pipeline.Classifier") as mock_cls:
            mock_cls.side_effect = lambda **kwargs: MagicMock(name="clf")
            first = pipeline._get_classifier(
                schema, ClassificationPromptConfig(temperature=0.1)
            )
            second = pipeline._get_classifier(
                schema, ClassificationPromptConfig(temperature=0.2)
            )

        assert mock_cls.call_count == 2
        assert first is not second
        assert [c.kwargs["config"].temperature for c in mock_cls.call_args_list] == [
            0.1,
            0.2,
        ]
        assert len(pipeline._classifier_cache) == 2

    def test_rebound_interface_constructs_again(self):
        """Same schema and config, ``pipeline.llm_interface`` rebound to another
        interface -> a second construction over the NEW interface, never the
        classifier cached over the old one. A per-entry other model is keyed
        on its name only, so it survives the rebinding."""
        old_llm = MagicMock(spec=LLMInterface)
        old_llm.model = "gpt-4"
        new_llm = MagicMock(spec=LLMInterface)
        new_llm.model = "gpt-4"
        pipeline = _make_pipeline(old_llm, _ambiguous_fsm())
        schema = _schema("a", "b")

        with patch("fsm_llm.pipeline.Classifier") as mock_cls:
            mock_cls.side_effect = lambda **kwargs: MagicMock(name="clf")
            first = pipeline._get_classifier(schema, None)
            other_first = pipeline._get_classifier(schema, None, model="gpt-4o")
            pipeline.llm_interface = new_llm
            second = pipeline._get_classifier(schema, None)
            other_second = pipeline._get_classifier(schema, None, model="gpt-4o")

        assert first is not second
        assert other_first is other_second
        assert [c.kwargs.get("llm") for c in mock_cls.call_args_list] == [
            old_llm,
            None,
            new_llm,
        ]
        assert len(pipeline._classifier_cache) == 3

    def test_same_model_override_uses_the_conversation_interface(self):
        """An entry ``model`` equal to the interface's is no override: the
        classifier sends through the conversation's interface, one entry."""
        mock_llm = MagicMock(spec=LLMInterface)
        mock_llm.model = "gpt-4"
        pipeline = _make_pipeline(mock_llm, _ambiguous_fsm())
        schema = _schema("a", "b")

        with patch("fsm_llm.pipeline.Classifier") as mock_cls:
            mock_cls.side_effect = lambda **kwargs: MagicMock(name="clf")
            first = pipeline._get_classifier(schema, None, model="gpt-4")
            second = pipeline._get_classifier(schema, None)

        assert first is second
        assert mock_cls.call_count == 1
        assert mock_cls.call_args.kwargs == {
            "schema": schema,
            "llm": mock_llm,
            "config": None,
        }


# ---------------------------------------------------------------------------
# Plan 07ad3f8c step 4: classification with no user message
# ---------------------------------------------------------------------------


def _route_fsm() -> tuple[FSMDefinition, State]:
    """Initial ``start`` and a second state ``route`` with two tied exits."""
    route = _make_state(
        "route",
        transitions=[
            _make_transition("billing", "The account has an unpaid invoice"),
            _make_transition("support", "The device reports a fault"),
        ],
    )
    start = _make_state("start", transitions=[_make_transition("route")])
    return _make_fsm_definition({"start": start, "route": route}), route


class _StructuredProvider:
    """Scripted ``fsm_llm.llm.completion`` that records each request."""

    def __init__(self, intent: str):
        self.calls: list[dict[str, Any]] = []
        self._intent = intent

    def completion(self, **kwargs: Any) -> MagicMock:
        import json

        self.calls.append(kwargs)
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = json.dumps(
            {"reasoning": "r", "intent": self._intent, "confidence": 0.95}
        )
        response.choices[0].message.reasoning_content = None
        return response

    def __enter__(self) -> _StructuredProvider:
        self._patches = [
            patch("fsm_llm.llm.completion", side_effect=self.completion),
            patch(
                "fsm_llm.llm.get_supported_openai_params",
                return_value=["response_format"],
            ),
        ]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc: object) -> None:
        for p in reversed(self._patches):
            p.stop()


def _provider_backed_mock(model: str) -> MagicMock:
    """A spec'd mock conversation interface whose ``complete`` is a real
    ``LiteLLMInterface``'s for ``model``: the classifier sends through the
    conversation's interface (D-006 of plan 944e2692), so this is how its
    request reaches the binding ``_StructuredProvider`` scripts."""
    mock_llm = MagicMock(spec=LLMInterface)
    mock_llm.model = model
    mock_llm.complete.side_effect = LiteLLMInterface(model=model).complete
    return mock_llm


class TestAmbiguousTransitionWithoutUserMessage:
    """A turn with no user message resolves a tie from the context alone."""

    def _instance(self) -> FSMInstance:
        instance = _make_instance(current_state="route")
        instance.context.data.update(
            {
                "invoice_status": "unpaid",
                "api_key": "sk-live-abcdef0123456789abcdef",
                "_approval_granted": {"tool": "refund"},
            }
        )
        instance.context.conversation.add_user_message("my bill looks wrong")
        return instance

    def test_classifier_receives_none_and_the_context(self):
        fsm_def, _ = _route_fsm()
        mock_llm = MagicMock(spec=LLMInterface)
        mock_llm.model = "gpt-4"
        pipeline = _make_pipeline(mock_llm, fsm_def)
        instance = self._instance()

        with patch("fsm_llm.pipeline.Classifier") as mock_cls_cls:
            classifier = MagicMock()
            classifier.classify.return_value = _mock_classifier_result("billing")
            mock_cls_cls.return_value = classifier
            result = pipeline._resolve_ambiguous_transition(
                _make_ambiguous_evaluation("billing", "support"),
                None,
                DataExtractionResponse(),
                instance,
                "conv-1",
            )

        assert result == "billing"
        args, kwargs = classifier.classify.call_args
        assert args == (None,)
        # No exchange is in flight, so the last user entry is history.
        assert kwargs["context"]["history"] == [{"user": "my bill looks wrong"}]
        assert kwargs["context"]["data"]["invoice_status"] == "unpaid"

    def test_message_turn_still_drops_the_in_flight_user_entry(self):
        fsm_def, route = _route_fsm()
        mock_llm = MagicMock(spec=LLMInterface)
        mock_llm.model = "gpt-4"
        pipeline = _make_pipeline(mock_llm, fsm_def)
        context = pipeline._build_classifier_context(
            self._instance(), route, "conv-1", user_message="my bill looks wrong"
        )
        assert context["history"] == []

    def test_provider_request_is_context_only(self):
        from fsm_llm.constants import NEUTRAL_USER_TURN

        fsm_def, _ = _route_fsm()
        mock_llm = _provider_backed_mock("gpt-4o")
        pipeline = _make_pipeline(mock_llm, fsm_def)
        instance = self._instance()

        with _StructuredProvider("billing") as provider:
            result = pipeline._resolve_ambiguous_transition(
                _make_ambiguous_evaluation("billing", "support"),
                None,
                DataExtractionResponse(),
                instance,
                "conv-1",
            )

        assert result == "billing"
        (call,) = provider.calls
        system, user = call["messages"]
        prompt = system["content"]
        assert "Analyze the user's message" not in prompt
        assert "classify the message itself" not in prompt
        assert "there is no user message" in prompt
        assert "There is no user message. Classify from this context." in prompt
        assert "unpaid" in prompt
        assert "my bill looks wrong" in prompt
        assert "sk-live-abcdef0123456789abcdef" not in prompt
        assert "_approval_granted" not in prompt
        assert "Continue" not in prompt
        assert user == {"role": "user", "content": NEUTRAL_USER_TURN}

    def test_fallback_intent_means_stay(self):
        fsm_def, _ = _route_fsm()
        mock_llm = _provider_backed_mock("gpt-4o")
        pipeline = _make_pipeline(mock_llm, fsm_def)

        with _StructuredProvider(TRANSITION_CLASSIFICATION_FALLBACK_INTENT) as provider:
            result = pipeline._resolve_ambiguous_transition(
                _make_ambiguous_evaluation("billing", "support"),
                None,
                DataExtractionResponse(),
                self._instance(),
                "conv-1",
            )

        assert result is None
        assert len(provider.calls) == 1  # the classifier really answered


class TestClassificationExtractionWithoutUserMessage:
    """``classification_extractions`` on a turn with no user message."""

    def test_context_only_prompt_and_cached_message_prompt_untouched(self):
        from fsm_llm.classification import Classifier

        schema = _schema("act", "done")
        with _StructuredProvider("act") as provider:
            classifier = Classifier(schema, model="gpt-4o")
            cached = classifier._system_prompt
            first = classifier.classify(None, context={"data": {"task": "sum"}})
            second = classifier.classify("please add", context={"data": {"t": 1}})

        assert first.intent == second.intent == "act"
        no_message, with_message = provider.calls
        assert "there is no user message" in no_message["messages"][0]["content"]
        assert with_message["messages"][0]["content"].startswith(cached)
        assert "Analyze the user's message" in cached
        assert with_message["messages"][1]["content"] == "please add"
        assert classifier._system_prompt == cached

    def test_pipeline_extraction_site_passes_none(self):
        state = State(
            id="route",
            description="d",
            purpose="Pick the next move",
            classification_extractions=[
                ClassificationExtractionConfig(
                    field_name="next_move",
                    intents=[
                        IntentDefinition(name="act", description="Call a tool"),
                        IntentDefinition(name="done", description="Answer now"),
                    ],
                    fallback_intent="done",
                )
            ],
            transitions=[],
        )
        start = _make_state("start", transitions=[_make_transition("route")])
        fsm_def = _make_fsm_definition({"start": start, "route": state})
        mock_llm = _provider_backed_mock("gpt-4o")
        pipeline = _make_pipeline(mock_llm, fsm_def)
        instance = _make_instance(current_state="route")
        instance.context.data["task"] = "sum 2 and 3"

        with _StructuredProvider("act") as provider:
            data = pipeline._execute_classification_extractions(
                state, None, instance, "conv-1"
            )

        assert data == {"next_move": "act"}
        (call,) = provider.calls
        assert "there is no user message" in call["messages"][0]["content"]
        assert "sum 2 and 3" in call["messages"][0]["content"]


# ---------------------------------------------------------------------------
# Plan 944e2692 step 5 (D-006): the classifier uses the conversation's own
# LLM interface
# ---------------------------------------------------------------------------


class _RecordingInterface(LLMInterface):
    """A custom conversation interface (not a ``LiteLLMInterface``): Pass 1
    finds nothing, Pass 2 says "ok", ``complete`` answers classifier
    requests by schema (``mood`` intents -> ``angry``, transition intents ->
    ``billing``) and records every request."""

    model = "custom/model"

    def __init__(self) -> None:
        self.requests: list[CompletionRequest] = []

    def generate_response(
        self, request: ResponseGenerationRequest
    ) -> ResponseGenerationResponse:
        return ResponseGenerationResponse(
            message="ok", message_type="response", reasoning="r"
        )

    def extract_field(self, request: FieldExtractionRequest) -> FieldExtractionResponse:
        return FieldExtractionResponse(
            field_name=request.field_name, value=None, confidence=0.0, is_valid=False
        )

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        import json

        self.requests.append(request)
        schema = json.dumps(request.response_format)
        intent = "angry" if '"angry"' in schema else "billing"
        return CompletionResponse(
            kind="final",
            text=json.dumps({"reasoning": "r", "intent": intent, "confidence": 0.95}),
        )


def _injected_fsm(mood_model: str | None = None) -> dict[str, Any]:
    """``start`` (initial) moves to ``route`` deterministically; ``route``
    owns a ``mood`` classification field and two tied, unconditioned exits
    (AMBIGUOUS on every turn)."""
    mood: dict[str, Any] = {
        "field_name": "mood",
        "intents": [
            {"name": "calm", "description": "The user is calm"},
            {"name": "angry", "description": "The user is upset"},
        ],
        "fallback_intent": "calm",
        "confidence_threshold": 0.5,
    }
    if mood_model is not None:
        mood["model"] = mood_model
    return {
        "name": "injected",
        "description": "Classifier injection",
        "initial_state": "start",
        "states": {
            "start": {
                "id": "start",
                "description": "Start",
                "purpose": "Greet",
                "response_instructions": "Greet",
                "transitions": [{"target_state": "route", "description": "Begin"}],
            },
            "route": {
                "id": "route",
                "description": "Route",
                "purpose": "Route the request",
                "response_instructions": "Respond",
                "classification_extractions": [mood],
                "transitions": [
                    {"target_state": "billing", "description": "A billing issue"},
                    {"target_state": "support", "description": "A device fault"},
                ],
            },
            "billing": {"id": "billing", "description": "Billing", "purpose": "End"},
            "support": {"id": "support", "description": "Support", "purpose": "End"},
        },
    }


def _injected_run(
    llm: Any, fsm: dict[str, Any], provider: Any
) -> tuple[API, str, list]:
    """Two turns on ``fsm``: into ``route``, then one turn in ``route``. A
    POST_TRANSITION handler records each transition; ``provider`` is the
    side effect of the patched ``fsm_llm.llm.completion``."""
    transitions: list[tuple[str, str]] = []
    api = API.from_definition(fsm, llm_interface=llm)
    api.register_handler(
        api.create_handler("record_transitions")
        .at(HandlerTiming.POST_TRANSITION)
        .do(
            lambda ctx: (
                transitions.append(
                    (ctx.get("_previous_state"), ctx.get("_current_state"))
                )
                or {}
            )
        )
    )
    with (
        patch("fsm_llm.llm.completion", side_effect=provider),
        patch(
            "fsm_llm.llm.get_supported_openai_params",
            return_value=["response_format"],
        ),
    ):
        conv_id, _ = api.start_conversation()
        api.converse("hello", conv_id)
        api.converse("I was charged twice and I am furious", conv_id)
    return api, conv_id, transitions


def _provider_must_not_be_called(**kwargs: Any) -> Any:
    raise AssertionError(f"provider binding reached for {kwargs.get('model')!r}")


class TestClassifierUsesConversationInterface:
    """A custom ``llm_interface`` given to ``API`` receives every classifier
    request (AMBIGUOUS transitions and ``classification_extractions``); only
    an entry naming another model gets its own interface, which inherits
    nothing. RED on the parent: the classifier built a private
    ``LiteLLMInterface`` and called the provider binding."""

    def test_ambiguous_transition_goes_to_the_custom_interface(self):
        llm = _RecordingInterface()
        api, conv_id, transitions = _injected_run(
            llm, _injected_fsm(), _provider_must_not_be_called
        )

        assert api.get_current_state(conv_id) == "billing"
        assert transitions == [("start", "route"), ("route", "billing")]
        transition_requests = [
            r for r in llm.requests if '"billing"' in json.dumps(r.response_format)
        ]
        assert len(transition_requests) == 1
        assert transition_requests[0].call_type == "classification"

    def test_classification_extraction_goes_to_the_custom_interface(self):
        llm = _RecordingInterface()
        api, conv_id, _ = _injected_run(
            llm, _injected_fsm(), _provider_must_not_be_called
        )

        assert api.get_data(conv_id)["mood"] == "angry"
        mood_requests = [
            r for r in llm.requests if '"angry"' in json.dumps(r.response_format)
        ]
        assert mood_requests
        assert all(r.call_type == "classification" for r in llm.requests)

    def test_entry_with_another_model_gets_its_own_interface_and_no_settings(self):
        provider_calls: list[dict[str, Any]] = []

        def provider(**kwargs: Any) -> Any:
            provider_calls.append(kwargs)
            response = MagicMock()
            response.choices = [MagicMock()]
            response.choices[0].message.content = json.dumps(
                {"reasoning": "r", "intent": "angry", "confidence": 0.9}
            )
            response.choices[0].message.reasoning_content = None
            response.choices[0].message.tool_calls = None
            return response

        llm = _RecordingInterface()
        llm.kwargs = {"api_key": "sk-conversation-secret", "api_base": "http://p"}
        llm.timeout = 7
        api, conv_id, _ = _injected_run(
            llm, _injected_fsm("anthropic/claude-3-haiku"), provider
        )

        # The tie still goes to the conversation's interface ...
        assert api.get_current_state(conv_id) == "billing"
        assert all('"angry"' not in json.dumps(r.response_format) for r in llm.requests)
        # ... the other model's entry to its own interface, settings not inherited.
        assert api.get_data(conv_id)["mood"] == "angry"
        assert provider_calls
        for call in provider_calls:
            assert call["model"] == "anthropic/claude-3-haiku"
            assert "api_key" not in call
            assert "api_base" not in call
            assert call["timeout"] == 120.0

    def test_usage_is_counted_on_the_conversation_interface(self):
        provider_calls: list[dict[str, Any]] = []

        def provider(**kwargs: Any) -> Any:
            provider_calls.append(kwargs)
            response = MagicMock()
            response.choices = [MagicMock()]
            if kwargs.get("response_format") is not None and '"intent"' in json.dumps(
                kwargs["response_format"]
            ):
                intents = json.dumps(kwargs["response_format"])
                intent = "angry" if '"angry"' in intents else "billing"
                content = {"reasoning": "r", "intent": intent, "confidence": 0.95}
            else:
                content = {"message": "ok", "reasoning": ""}
            response.choices[0].message.content = json.dumps(content)
            response.choices[0].message.reasoning_content = None
            response.choices[0].message.tool_calls = None
            response.usage = None
            return response

        llm = LiteLLMInterface(model="gpt-4o")
        api, conv_id, _ = _injected_run(llm, _injected_fsm(), provider)

        assert api.get_current_state(conv_id) == "billing"
        classifier_calls = [
            c
            for c in provider_calls
            if c.get("response_format") is not None
            and '"intent"' in json.dumps(c["response_format"])
        ]
        assert classifier_calls
        usage = llm.usage()
        assert usage.by_kind["classify"].calls == len(classifier_calls)
        assert usage.calls == len(provider_calls)

    def test_direct_classifier_with_llm_refuses_connection_settings(self):
        llm = _RecordingInterface()
        schema = _schema("a", "b")
        with pytest.raises(ValueError, match="connection"):
            Classifier(schema, llm=llm, api_key="k")
        with pytest.raises(ValueError, match="connection"):
            Classifier(schema, llm=llm, timeout=5)
        with pytest.raises(ValueError, match="different model"):
            Classifier(schema, model="gpt-4o", llm=llm)
        classifier = Classifier(schema, llm=llm)
        assert classifier.model == "custom/model"
        assert classifier.classify("hi").intent == "a"  # "billing" is unknown
        (request,) = llm.requests
        assert request.messages[1] == {"role": "user", "content": "hi"}

    def test_direct_classifier_without_llm_is_unchanged(self):
        classifier = Classifier(_schema("a", "b"))
        assert classifier.model == DEFAULT_LLM_MODEL
        assert isinstance(classifier._llm, LiteLLMInterface)
        assert classifier._llm.timeout == 120.0
