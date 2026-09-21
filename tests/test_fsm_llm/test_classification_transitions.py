"""
Tests for classification-aware transition resolution.

Tests that when a State has ``transition_classification`` enabled, the
MessagePipeline uses Classifier (from fsm_llm_classification) to resolve
AMBIGUOUS transitions instead of the raw LLM prompt.
"""

from __future__ import annotations

import sys
import threading
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr


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
from fsm_llm.constants import (
    CONTEXT_KEY_CLASSIFICATION_RESULT,
    DEFAULT_TRANSITION_CLASSIFICATION_CONFIDENCE,
    MAX_CLASSIFIER_CACHE_SIZE,
    TRANSITION_CLASSIFICATION_FALLBACK_INTENT,
)
from fsm_llm.definitions import (
    ClassificationError,
    ClassificationExtractionConfig,
    ClassificationResult,
    ClassificationSchema,
    DataExtractionResponse,
    FSMContext,
    FSMDefinition,
    FSMError,
    FSMInstance,
    IntentDefinition,
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
        # get_data() strips internal keys; read the raw instance context.
        instance = api.fsm_manager.instances[conv_id]
        stored = instance.context.data[CONTEXT_KEY_CLASSIFICATION_RESULT]
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
        assert CONTEXT_KEY_CLASSIFICATION_RESULT not in instance.context.data

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
        stored = instance.context.data[CONTEXT_KEY_CLASSIFICATION_RESULT]
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
    """``MessagePipeline._get_classifier`` reuses one ``Classifier`` per content
    key (schema + model + prompt config + connection kwargs), bounded at
    ``MAX_CLASSIFIER_CACHE_SIZE``. Pins plan-2026-09-20T165703-0d9c218e D-001.
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
        assert {c.kwargs["model"] for c in mock_cls.call_args_list} == {
            "gpt-4",
            "gpt-4o",
        }
        assert mock_cls.return_value.classify.call_count == 4

    def test_cache_is_bounded_and_evicts_oldest(self):
        mock_llm = MagicMock(spec=LLMInterface)
        mock_llm.model = "gpt-4"
        pipeline = _make_pipeline(mock_llm, _ambiguous_fsm())
        assert MAX_CLASSIFIER_CACHE_SIZE == 64

        with patch("fsm_llm.pipeline.Classifier") as mock_cls:
            mock_cls.side_effect = lambda **kwargs: MagicMock(name="clf")
            first = pipeline._get_classifier(_schema("i0", "j0"), "gpt-4", None, {})
            first_key = next(iter(pipeline._classifier_cache))
            for n in range(1, MAX_CLASSIFIER_CACHE_SIZE + 1):
                pipeline._get_classifier(_schema(f"i{n}", f"j{n}"), "gpt-4", None, {})

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
                        _schema(f"t{tid}_i{n}", f"t{tid}_j{n}"), "gpt-4", None, {}
                    )
            except BaseException as e:  # collected for the assert below
                errors.append(e)

        old_interval = sys.getswitchinterval()
        try:
            with patch("fsm_llm.pipeline.Classifier") as mock_cls:
                mock_cls.side_effect = lambda **kwargs: object()
                for n in range(MAX_CLASSIFIER_CACHE_SIZE):
                    pipeline._get_classifier(
                        _schema(f"p{n}", f"q{n}"), "gpt-4", None, {}
                    )
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

    def test_non_json_native_connection_kwarg_bypasses_cache(self):
        """Two distinct ``SecretStr`` api keys must yield two constructions and
        insert nothing. RED on the pre-step-3 code: ``json.dumps(...,
        default=str)`` digested both as ``'**********'`` so the second key hit
        the first key's classifier (review W1). Pins the JSON-native
        cacheability guard: a non-native value bypasses the cache entirely.
        """
        mock_llm = MagicMock(spec=LLMInterface)
        mock_llm.model = "gpt-4"
        pipeline = _make_pipeline(mock_llm, _ambiguous_fsm())
        schema = _schema("a", "b")

        with patch("fsm_llm.pipeline.Classifier") as mock_cls:
            mock_cls.side_effect = lambda **kwargs: MagicMock(name="clf")
            first = pipeline._get_classifier(
                schema, "gpt-4", None, {"api_key": SecretStr("A")}
            )
            second = pipeline._get_classifier(
                schema, "gpt-4", None, {"api_key": SecretStr("B")}
            )

        assert mock_cls.call_count == 2
        assert first is not second
        assert [
            c.kwargs["api_key"].get_secret_value() for c in mock_cls.call_args_list
        ] == ["A", "B"]
        assert len(pipeline._classifier_cache) == 0

    def test_json_native_connection_kwarg_still_caches(self):
        """The guard fires only on non-JSON-native values: a plain-string
        ``api_key`` still yields one construction over two calls and one entry."""
        mock_llm = MagicMock(spec=LLMInterface)
        mock_llm.model = "gpt-4"
        pipeline = _make_pipeline(mock_llm, _ambiguous_fsm())
        schema = _schema("a", "b")

        with patch("fsm_llm.pipeline.Classifier") as mock_cls:
            mock_cls.side_effect = lambda **kwargs: MagicMock(name="clf")
            first = pipeline._get_classifier(schema, "gpt-4", None, {"api_key": "k"})
            second = pipeline._get_classifier(schema, "gpt-4", None, {"api_key": "k"})

        assert mock_cls.call_count == 1
        assert first is second
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
                schema, "gpt-4", ClassificationPromptConfig(temperature=0.1), {}
            )
            second = pipeline._get_classifier(
                schema, "gpt-4", ClassificationPromptConfig(temperature=0.2), {}
            )

        assert mock_cls.call_count == 2
        assert first is not second
        assert [c.kwargs["config"].temperature for c in mock_cls.call_args_list] == [
            0.1,
            0.2,
        ]
        assert len(pipeline._classifier_cache) == 2

    def test_connection_kwargs_change_constructs_again(self):
        """Same schema, model and config, ``api_key="k1"`` vs ``api_key="k2"``
        -> two constructions and two entries. Both values are JSON-native, so
        both calls go THROUGH the cache (the complement of
        ``test_json_native_connection_kwarg_still_caches``: different key ->
        different instance). PASSES by design today (the key already hashes
        ``connection_kwargs``, D-001); this pin guards a future key
        simplification that drops the connection kwargs (review W4).
        """
        mock_llm = MagicMock(spec=LLMInterface)
        mock_llm.model = "gpt-4"
        pipeline = _make_pipeline(mock_llm, _ambiguous_fsm())
        schema = _schema("a", "b")

        with patch("fsm_llm.pipeline.Classifier") as mock_cls:
            mock_cls.side_effect = lambda **kwargs: MagicMock(name="clf")
            first = pipeline._get_classifier(schema, "gpt-4", None, {"api_key": "k1"})
            second = pipeline._get_classifier(schema, "gpt-4", None, {"api_key": "k2"})

        assert mock_cls.call_count == 2
        assert first is not second
        assert [c.kwargs["api_key"] for c in mock_cls.call_args_list] == ["k1", "k2"]
        assert len(pipeline._classifier_cache) == 2
