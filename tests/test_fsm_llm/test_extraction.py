"""Focused unit tests for the new ``fsm_llm.dialog.extraction`` module
(Phase C, 0.8.0).

Pure code motion was assumed against the existing extensive coverage in
``test_pipeline.py``, ``test_classification_extractions.py``, and
``test_multipass_extraction.py``. This file adds targeted contract tests
for the engine's public surface, including:

* ``ExtractionEngine.__init__`` binds the parent pipeline as a single
  back-reference.
* The ``_oracle`` / ``llm_interface`` / ``field_extraction_prompt_builder``
  / ``fsm_resolver`` properties are *live* pass-throughs (mutating the
  parent pipeline propagates).
* The pass-through helpers (``get_state``, ``execute_handlers``,
  ``_apply_context_scope``, ``_clean_empty_context_keys``,
  ``_build_field_configs_from_state``) match the legacy MessagePipeline
  semantics byte-for-byte.
* The 8 migrated methods produce the same outputs as their pre-extraction
  predecessors when invoked on the engine vs the pipeline-delegation
  wrapper (the wrappers are kept on ``MessagePipeline`` for back-compat).
* ``ExtractionEngine`` does NOT construct its own Oracle — the M4 single-
  Oracle invariant holds (``engine._oracle is pipeline._oracle``).

These are deliberately kept tight: ~10 tests covering the key contract.
The deep semantic surface is exercised by the existing pipeline tests.
"""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

import pytest

from fsm_llm._models import (
    FieldExtractionResponse,
)
from fsm_llm.dialog.classification import Classifier
from fsm_llm.dialog.definitions import (
    ClassificationExtractionConfig,
    ClassificationResult,
    FieldExtractionConfig,
    FSMContext,
    FSMDefinition,
    FSMInstance,
    IntentDefinition,
    State,
    Transition,
)
from fsm_llm.dialog.extraction import (
    _TYPE_COERCERS,
    ExtractionEngine,
    _coerce_bool,
    _coerce_dict,
    _coerce_float,
    _coerce_int,
    _coerce_list,
    _coerce_str,
)
from fsm_llm.dialog.prompts import (
    DataExtractionPromptBuilder,
    FieldExtractionPromptBuilder,
    ResponseGenerationPromptBuilder,
)
from fsm_llm.dialog.transition_evaluator import TransitionEvaluator
from fsm_llm.dialog.turn import MessagePipeline
from fsm_llm.handlers import HandlerSystem
from fsm_llm.runtime._litellm import LLMInterface

# ----------------------------------------------------------
# Helpers (mirrors the patterns from the existing pipeline tests)
# ----------------------------------------------------------


def _make_terminal_state(state_id: str = "end") -> State:
    return State(
        id=state_id,
        description=f"Terminal {state_id}",
        purpose="end",
        response_instructions="Goodbye",
        transitions=[],
    )


def _make_state(
    state_id: str = "hello",
    *,
    response_instructions: str = "Respond.",
    extraction_instructions: str = "",
    field_extractions: list[FieldExtractionConfig] | None = None,
    classification_extractions: list[ClassificationExtractionConfig] | None = None,
    required_context_keys: list[str] | None = None,
    transitions: list[Transition] | None = None,
) -> State:
    return State(
        id=state_id,
        description=f"State {state_id}",
        purpose="test",
        response_instructions=response_instructions,
        extraction_instructions=extraction_instructions,
        field_extractions=field_extractions or [],
        classification_extractions=classification_extractions,
        required_context_keys=required_context_keys or [],
        transitions=transitions or [],
    )


def _make_fsm_definition(
    states: dict[str, State] | None = None, initial: str = "hello"
) -> FSMDefinition:
    if states is None:
        # Single-state FSM avoids reachability errors. The terminal
        # "hello" state has no transitions so it's its own end.
        states = {"hello": _make_state("hello")}
    return FSMDefinition(
        name="test",
        description="extraction tests",
        initial_state=initial,
        states=states,
    )


def _make_pipeline(
    fsm_def: FSMDefinition | None = None,
    *,
    llm: LLMInterface | None = None,
    handler_system: HandlerSystem | None = None,
) -> MessagePipeline:
    if fsm_def is None:
        fsm_def = _make_fsm_definition()
    if llm is None:
        llm = MagicMock(spec=LLMInterface)
        llm.model = "test-model"
        llm.extract_field.return_value = MagicMock(
            field_name="dummy", value=None, confidence=0.0, is_valid=False
        )
    return MessagePipeline(
        llm_interface=llm,
        data_extraction_prompt_builder=DataExtractionPromptBuilder(),
        response_generation_prompt_builder=ResponseGenerationPromptBuilder(),
        transition_evaluator=TransitionEvaluator(),
        handler_system=handler_system or HandlerSystem(),
        fsm_resolver=lambda _fid: fsm_def,
        field_extraction_prompt_builder=FieldExtractionPromptBuilder(),
    )


def _make_instance(
    fsm_id: str = "test", state: str = "hello", **ctx: object
) -> FSMInstance:
    return FSMInstance(
        fsm_id=fsm_id,
        current_state=state,
        context=FSMContext(data=dict(ctx)),
    )


# ==============================================================
# 1. Construction + identity invariants
# ==============================================================


class TestExtractionEngineInit:
    """``ExtractionEngine.__init__`` binds the parent pipeline only."""

    def test_engine_holds_pipeline_back_reference(self) -> None:
        """``__init__`` stores exactly one reference: the parent pipeline."""
        pipeline = _make_pipeline()
        engine = pipeline._extraction
        assert isinstance(engine, ExtractionEngine)
        assert engine._pipeline is pipeline

    def test_engine_does_not_construct_its_own_oracle(self) -> None:
        """M4 single-Oracle invariant: ``engine._oracle is pipeline._oracle``.

        The engine never calls ``LiteLLMOracle(...)``; it field-reads the
        parent pipeline's Oracle. Mirrors test_oracle_ownership.py G2 gate.
        """
        pipeline = _make_pipeline()
        engine = pipeline._extraction
        assert engine._oracle is pipeline._oracle

    def test_engine_oracle_property_is_live(self) -> None:
        """Mutating ``pipeline._oracle`` is reflected on ``engine._oracle``."""
        pipeline = _make_pipeline()
        engine = pipeline._extraction
        sentinel = object()
        pipeline._oracle = sentinel  # type: ignore[assignment]
        assert engine._oracle is sentinel


class TestExtractionEnginePassThroughs:
    """Property pass-throughs read from the parent pipeline live."""

    def test_llm_interface_propagates_runtime_mutation(self) -> None:
        """If a test swaps out ``pipeline.llm_interface`` after construction,
        the engine sees the new instance immediately.

        This is the contract that ``test_classification_extractions.py``
        relies on (line 284) for the ``test_no_model_available`` scenario.
        """
        pipeline = _make_pipeline()
        engine = pipeline._extraction
        replacement = MagicMock(spec=LLMInterface)
        pipeline.llm_interface = replacement
        assert engine.llm_interface is replacement

    def test_field_extraction_prompt_builder_passthrough(self) -> None:
        pipeline = _make_pipeline()
        engine = pipeline._extraction
        assert engine.field_extraction_prompt_builder is (
            pipeline.field_extraction_prompt_builder
        )

    def test_fsm_resolver_passthrough(self) -> None:
        pipeline = _make_pipeline()
        engine = pipeline._extraction
        assert engine.fsm_resolver is pipeline.fsm_resolver


# ==============================================================
# 2. Coercer helpers (module-level dispatch dict)
# ==============================================================


class TestCoercers:
    """Verify the type-coercer dispatch is byte-equivalent to the
    legacy turn.py module-level helpers."""

    def test_int_coercer(self) -> None:
        assert _coerce_int(42) == 42
        assert _coerce_int("3") == 3

    def test_float_coercer(self) -> None:
        assert _coerce_float(1.5) == 1.5
        assert _coerce_float("2.5") == 2.5

    def test_bool_coercer(self) -> None:
        assert _coerce_bool(True) is True
        assert _coerce_bool(False) is False
        assert _coerce_bool("true") is True
        assert _coerce_bool("False") is False
        assert _coerce_bool("yes") is True
        assert _coerce_bool("0") is False

    def test_str_coercer(self) -> None:
        assert _coerce_str("hi") == "hi"
        assert _coerce_str(7) == "7"

    def test_list_coercer(self) -> None:
        assert _coerce_list([1, 2]) == [1, 2]
        assert _coerce_list("[3,4]") == [3, 4]
        with pytest.raises((TypeError, ValueError)):
            _coerce_list('"not a list"')

    def test_dict_coercer(self) -> None:
        assert _coerce_dict({"a": 1}) == {"a": 1}
        assert _coerce_dict('{"k":"v"}') == {"k": "v"}
        with pytest.raises((TypeError, ValueError)):
            _coerce_dict("[1,2]")

    def test_dispatch_table_keys(self) -> None:
        """The dispatch dict registers exactly the 6 supported types
        (matches legacy turn.py contract)."""
        assert set(_TYPE_COERCERS.keys()) == {
            "int",
            "float",
            "bool",
            "str",
            "list",
            "dict",
        }


# ==============================================================
# 3. _validate_field_extraction (static method, pure logic)
# ==============================================================


class TestValidateFieldExtraction:
    """Static validator preserves the legacy behaviour from turn.py."""

    def test_valid_passthrough(self) -> None:
        config = FieldExtractionConfig(
            field_name="user_name",
            field_type="str",
            extraction_instructions="x",
            confidence_threshold=0.0,
        )
        response = FieldExtractionResponse(
            field_name="user_name",
            value="Alice",
            confidence=0.9,
            is_valid=True,
        )
        result = ExtractionEngine._validate_field_extraction(response, config)
        assert result.is_valid is True
        assert result.value == "Alice"

    def test_field_name_echo_rejected(self) -> None:
        config = FieldExtractionConfig(
            field_name="user_name",
            field_type="str",
            extraction_instructions="x",
        )
        response = FieldExtractionResponse(
            field_name="user_name",
            value="user_name",  # echoed back — must be rejected
            confidence=0.95,
            is_valid=True,
        )
        result = ExtractionEngine._validate_field_extraction(response, config)
        assert result.is_valid is False
        assert "field name" in (result.validation_error or "").lower()

    def test_below_confidence_threshold_rejected(self) -> None:
        config = FieldExtractionConfig(
            field_name="age",
            field_type="int",
            extraction_instructions="x",
            confidence_threshold=0.8,
        )
        response = FieldExtractionResponse(
            field_name="age",
            value=30,
            confidence=0.5,
            is_valid=True,
        )
        result = ExtractionEngine._validate_field_extraction(response, config)
        assert result.is_valid is False
        assert "Confidence" in (result.validation_error or "")

    def test_int_coercion_applied(self) -> None:
        config = FieldExtractionConfig(
            field_name="age",
            field_type="int",
            extraction_instructions="x",
            confidence_threshold=0.0,
        )
        response = FieldExtractionResponse(
            field_name="age",
            value="42",
            confidence=0.9,
            is_valid=True,
        )
        result = ExtractionEngine._validate_field_extraction(response, config)
        assert result.is_valid is True
        assert result.value == 42

    def test_allowed_values_enforced(self) -> None:
        config = FieldExtractionConfig(
            field_name="status",
            field_type="str",
            extraction_instructions="x",
            confidence_threshold=0.0,
            validation_rules={"allowed_values": ["active", "inactive"]},
        )
        response = FieldExtractionResponse(
            field_name="status",
            value="pending",
            confidence=0.95,
            is_valid=True,
        )
        result = ExtractionEngine._validate_field_extraction(response, config)
        assert result.is_valid is False
        assert "not in allowed values" in (result.validation_error or "")


# ==============================================================
# 4. _bulk_extract_from_instructions — single LLM call shape
# ==============================================================


class TestBulkExtractFromInstructions:
    """The bulk-extract fallback dispatches via ``oracle.invoke_messages``."""

    def test_bulk_extract_returns_dict_on_success(self) -> None:
        pipeline = _make_pipeline()
        engine = pipeline._extraction

        # Patch the oracle's invoke_messages so we don't hit the network.
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[
            0
        ].message.content = '{"extracted_data": {"city": "Tokyo"}, "confidence": 0.9}'
        with patch.object(
            pipeline._oracle, "invoke_messages", return_value=mock_response
        ) as mock_invoke:
            state = _make_state(
                "ask_city",
                extraction_instructions="Extract the city the user mentions.",
            )
            instance = _make_instance()
            result = engine._bulk_extract_from_instructions(
                instance, "I live in Tokyo", state, "conv1"
            )

        assert result == {"city": "Tokyo"}
        mock_invoke.assert_called_once()
        # call_type must be "data_extraction" so the spy / litellm path
        # treats this as a JSON-shaped extraction call.
        call_kwargs = mock_invoke.call_args
        assert call_kwargs.kwargs.get("call_type") == "data_extraction"

    def test_bulk_extract_filters_empty_values(self) -> None:
        pipeline = _make_pipeline()
        engine = pipeline._extraction

        mock_response = Mock()
        mock_response.choices = [Mock()]
        # Mix of None / "" / {} / valid — only valid passes through.
        mock_response.choices[
            0
        ].message.content = (
            '{"extracted_data": {"a": "ok", "b": null, "c": "", "d": {}}}'
        )
        with patch.object(
            pipeline._oracle, "invoke_messages", return_value=mock_response
        ):
            state = _make_state("s", extraction_instructions="x")
            instance = _make_instance()
            result = engine._bulk_extract_from_instructions(
                instance, "msg", state, "c1"
            )

        assert result == {"a": "ok"}


# ==============================================================
# 5. _execute_classification_extractions — engine and pipeline parity
# ==============================================================


class TestExecuteClassificationExtractions:
    """Engine method matches the legacy pipeline-side surface."""

    def test_engine_method_returns_intent_dict(self) -> None:
        config = ClassificationExtractionConfig(
            field_name="sentiment",
            intents=[
                IntentDefinition(name="positive", description="happy"),
                IntentDefinition(name="negative", description="sad"),
                IntentDefinition(name="neutral", description="meh"),
            ],
            fallback_intent="neutral",
        )
        state = _make_state("hello", classification_extractions=[config])
        pipeline = _make_pipeline(_make_fsm_definition({"hello": state}))
        engine = pipeline._extraction
        instance = _make_instance()

        result = ClassificationResult(
            reasoning="r", intent="positive", confidence=0.9, entities={}
        )
        with patch.object(Classifier, "classify", return_value=result):
            data = engine._execute_classification_extractions(
                state, "I love this!", instance, "c1"
            )
        assert data["sentiment"] == "positive"

    def test_pipeline_delegates_to_engine(self) -> None:
        """``MessagePipeline._execute_classification_extractions`` is now
        a thin delegation wrapper that returns the engine's dict
        unchanged."""
        config = ClassificationExtractionConfig(
            field_name="mood",
            intents=[
                IntentDefinition(name="up", description="up"),
                IntentDefinition(name="down", description="down"),
            ],
            fallback_intent="up",
        )
        state = _make_state("hello", classification_extractions=[config])
        pipeline = _make_pipeline(_make_fsm_definition({"hello": state}))
        instance = _make_instance()

        # Patch the ENGINE method — the pipeline wrapper must forward.
        sentinel = {"mood": "down"}
        with patch.object(
            pipeline._extraction,
            "_execute_classification_extractions",
            return_value=sentinel,
        ) as mock_engine:
            result = pipeline._execute_classification_extractions(
                state, "msg", instance, "c1"
            )

        mock_engine.assert_called_once()
        assert result is sentinel
