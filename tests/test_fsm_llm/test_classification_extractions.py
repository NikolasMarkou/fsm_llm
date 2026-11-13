"""Tests for classification-based extraction (classification_extractions on State)."""

from __future__ import annotations

import json
import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from litellm.exceptions import RateLimitError

from fsm_llm.constants import (
    MAX_MULTI_INTENTS,
    METADATA_KEY_CLASSIFICATION_RESULTS,
)
from fsm_llm.context.layout import (
    DataExtractionPromptBuilder,
    FieldExtractionPromptBuilder,
    ResponseGenerationPromptBuilder,
)
from fsm_llm.core.definitions import (
    ClassificationError,
    ClassificationExtractionConfig,
    ClassificationPromptConfig,
    ClassificationResponseError,
    ClassificationResult,
    ClassificationSchema,
    FSMContext,
    FSMDefinition,
    FSMError,
    FSMInstance,
    HierarchicalSchema,
    IntentDefinition,
    LLMResponseError,
    MultiClassificationResult,
    State,
    Transition,
    TransitionCondition,
)
from fsm_llm.core.transition_evaluator import TransitionEvaluator
from fsm_llm.engine.api.conversation import API
from fsm_llm.engine.classification import Classifier, HierarchicalClassifier
from fsm_llm.engine.kinds.conversation.executor import ConversationPass1
from fsm_llm.engine.turn.message_pipeline import MessagePipeline
from fsm_llm.handlers import HandlerSystem
from fsm_llm.llm import LiteLLMInterface, LLMInterface
from fsm_llm.llm.interface import _completion_response
from fsm_llm.logging import logger
from tests.test_fsm_llm.support import (
    _A4_TRANSITION_KEY,
    _a3_api,
    _a3_extraction_fsm,
    _ambiguous_then_blocked_fsm,
    _api,
    _ClassBulkProv,
    _classification,
    _classified_bulk_fsm,
    _classified_fsm,
    _ClassifierCapture,
    _fake_response,
    _mock_llm,
    _raw_data,
)

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
    mock_llm = MagicMock(spec=LLMInterface)
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


class TestIntentsCapMatchesRuntimeSibling:
    """F-09 / SC-13.

    `ClassificationExtractionConfig.intents` had no upper bound while the
    `ClassificationSchema` that `ConversationPass1._execute_classification_extractions`
    builds FROM it at runtime caps at 15. A 16-intent state therefore loaded
    clean and failed on first ENTRY to the state.
    """

    @staticmethod
    def _names(n: int) -> list[str]:
        return [f"intent_{i}" for i in range(n)]

    def test_sixteen_intents_raise_at_construction(self):
        """The failure moved from first-entry to construction time."""
        with pytest.raises(ValueError, match="at most 15"):
            _make_config(intents=self._names(16), fallback="intent_0")

    def test_fifteen_intents_still_accepted(self):
        """Vacuity guard: the cap is 15, not "any list is rejected"."""
        config = _make_config(intents=self._names(15), fallback="intent_0")
        assert len(config.intents) == 15

    def test_cap_is_identical_to_the_runtime_sibling(self):
        """The lockstep invariant D-013 records, asserted rather than trusted.

        Reads both caps off the pydantic schemas so the test fails if EITHER
        side is changed alone -- which is the whole failure mode F-09 was.
        """
        config_cap = ClassificationExtractionConfig.model_fields[
            "intents"
        ].metadata  # constraint objects
        schema_cap = ClassificationSchema.model_fields["intents"].metadata

        def max_len(metadata):
            # Default None (not StopIteration) so an ABSENT cap fails this as a
            # clean assertion rather than an opaque generator error.
            return next(
                (m.max_length for m in metadata if getattr(m, "max_length", None)),
                None,
            )

        assert max_len(config_cap) == max_len(schema_cap) == 15

    def test_load_time_rejection_reaches_api_from_file(self, tmp_path):
        """SC-13 end-to-end: `API.from_file`, not just the model constructor."""
        fsm = {
            "name": "capped",
            "description": "F-09 reproducer",
            "initial_state": "start",
            "states": {
                "start": {
                    "id": "start",
                    "description": "Start",
                    "purpose": "Begin",
                    "classification_extractions": [
                        {
                            "field_name": "user_intent",
                            "intents": [
                                {"name": n, "description": f"The {n} intent"}
                                for n in self._names(16)
                            ],
                            "fallback_intent": "intent_0",
                        }
                    ],
                    "transitions": [
                        {"target_state": "end", "description": "Go to end"}
                    ],
                },
                "end": {
                    "id": "end",
                    "description": "End",
                    "purpose": "Finish",
                    "transitions": [],
                },
            },
        }
        path = tmp_path / "capped.json"
        path.write_text(json.dumps(fsm), encoding="utf-8")

        with pytest.raises(ValueError, match="at most 15"):
            API.from_file(str(path))


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
            data = ConversationPass1(pipeline)._execute_classification_extractions(
                state, "I'm so frustrated!", instance, "conv1"
            )

        # A4 (D-005): the returned delta carries only the intent; the full
        # result lives in context.metadata, not in a dropped `_` data key.
        assert data == {"sentiment": "negative"}
        full = instance.context.metadata[METADATA_KEY_CLASSIFICATION_RESULTS]
        assert full["sentiment"]["intent"] == "negative"
        assert full["sentiment"]["confidence"] == 0.9

    def test_below_threshold_skipped(self):
        config = _make_config(confidence_threshold=0.8)
        state = _make_state(classification_extractions=[config])
        pipeline = _make_pipeline()
        instance = _make_instance()

        result = _mock_classify_result("negative", 0.5)

        with patch.object(Classifier, "classify", return_value=result):
            data = ConversationPass1(pipeline)._execute_classification_extractions(
                state, "hmm", instance, "conv1"
            )

        assert "sentiment" not in data

    def test_fallback_intent_stored(self):
        """Fallback intent is always stored so downstream JsonLogic has the key."""
        config = _make_config()
        state = _make_state(classification_extractions=[config])
        pipeline = _make_pipeline()
        instance = _make_instance()

        result = _mock_classify_result("neutral", 0.95)  # neutral is fallback

        with patch.object(Classifier, "classify", return_value=result):
            data = ConversationPass1(pipeline)._execute_classification_extractions(
                state, "what time is it", instance, "conv1"
            )

        assert data["sentiment"] == "neutral"

    def test_exception_handling(self):
        config = _make_config()
        state = _make_state(classification_extractions=[config])
        pipeline = _make_pipeline()
        instance = _make_instance()

        with patch.object(
            Classifier, "classify", side_effect=ClassificationError("boom")
        ):
            data = ConversationPass1(pipeline)._execute_classification_extractions(
                state, "test", instance, "conv1"
            )

        assert data == {}

    @pytest.mark.parametrize(
        "error", [RuntimeError("bug"), ValueError("bug"), KeyError("bug")]
    )
    def test_an_error_outside_the_classifier_family_propagates(self, error: Exception):
        """Only ``ClassificationError`` (an outage or an unreadable reply) is
        soft; anything else is a bug and propagates (D-037 of plan 944e2692)."""
        config = _make_config()
        state = _make_state(classification_extractions=[config])
        pipeline = _make_pipeline()
        instance = _make_instance()

        with (
            patch.object(Classifier, "classify", side_effect=error),
            pytest.raises(type(error)),
        ):
            ConversationPass1(pipeline)._execute_classification_extractions(
                state, "test", instance, "conv1"
            )

    @pytest.mark.parametrize("required", [False, True])
    def test_interface_without_complete_builds_no_classifier(self, required):
        """The classifier sends through the conversation's interface (D-006
        of plan 944e2692). One that is not an ``LLMInterface`` implementing
        ``complete`` cannot classify (``API`` refuses the pair at
        construction, see test_classification_transitions.py::TestClassifierNeedsAnInterfaceWithComplete); a pipeline reached
        without that check builds and caches nothing, and the field fails
        with ``ClassificationError`` as D-020 states: skipped, or raised when
        required (D-046 of plan 944e2692; it raised ``ValueError`` at
        f85e07a)."""
        config = _make_config(required=required)
        state = _make_state(classification_extractions=[config])
        pipeline = _make_pipeline()
        pipeline.llm_interface = MagicMock(spec=[])  # no .model, no .complete
        instance = _make_instance()

        with patch("fsm_llm.llm.providers.litellm.completion") as provider:
            if required:
                with pytest.raises(ClassificationError, match="complete"):
                    ConversationPass1(pipeline)._execute_classification_extractions(
                        state, "test", instance, "conv1"
                    )
            else:
                assert (
                    ConversationPass1(pipeline)._execute_classification_extractions(
                        state, "test", instance, "conv1"
                    )
                    == {}
                )
        provider.assert_not_called()
        assert pipeline._classifier_cache == {}

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
                ConversationPass1(pipeline)._execute_classification_extractions(
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
            data = ConversationPass1(pipeline)._execute_classification_extractions(
                state, "test", instance, "conv1"
            )

        assert data == {"sentiment": "positive"}
        full = instance.context.metadata[METADATA_KEY_CLASSIFICATION_RESULTS]
        assert full["sentiment"]["context_snapshot"] == {"user_name": "Alice"}

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
            Classifier, "classify", side_effect=lambda msg, context=None: next(results)
        ):
            data = ConversationPass1(pipeline)._execute_classification_extractions(
                state, "urgent complaint", instance, "conv1"
            )

        assert data["sentiment"] == "negative"
        assert data["urgency"] == "high"

    def test_empty_configs_returns_empty(self):
        state = _make_state()  # no classification_extractions
        pipeline = _make_pipeline()
        instance = _make_instance()

        data = ConversationPass1(pipeline)._execute_classification_extractions(
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
            response = ConversationPass1(pipeline)._execute_data_extraction(
                instance, "I'm frustrated!", "conv1"
            )

        assert response.extracted_data["sentiment"] == "negative"

    def test_no_extractions_returns_empty(self):
        state = _make_state()  # no field or classification extractions
        fsm = _make_fsm({"triage": state, "end": _terminal_state("end")})
        pipeline = _make_pipeline(fsm)
        instance = _make_instance()

        response = ConversationPass1(pipeline)._execute_data_extraction(
            instance, "hello", "conv1"
        )
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


# ----------------------------------------------------------
# F-03: litellm boundary wrapping in Classifier
# ----------------------------------------------------------


def _rate_limit_error() -> RateLimitError:
    """A REAL litellm transient error.

    Deliberately not a RuntimeError stand-in: RateLimitError's MRO is
    openai.APIStatusError -> APIError -> OpenAIError -> Exception, which
    matches none of the types in the pipeline's classification except tuple.
    A RuntimeError stand-in is caught by that tuple already and would make
    this whole section pass against un-fixed code.
    """
    return RateLimitError(message="rate limited", llm_provider="openai", model="gpt-4o")


def _classifier() -> Classifier:
    return Classifier(
        schema=ClassificationSchema(
            intents=_make_intents("positive", "negative", "neutral"),
            fallback_intent="neutral",
        ),
        model="gpt-4o",
    )


class TestClassifierLLMBoundary:
    def test_ratelimit_during_extraction_still_completes_the_turn(
        self, mock_llm2_interface
    ):
        """SC-3: the user-visible property -- the turn returns a response.

        Asserted end-to-end through API.converse(), not at the classifier, so
        that a fix which merely stops the classifier raising but leaves the
        turn broken cannot satisfy it.
        """
        config = _make_config()  # required=False -- designed to fail soft
        fsm = _make_fsm(
            {
                "triage": _make_state(classification_extractions=[config]),
                "end": _terminal_state("end"),
            }
        )
        # The classifier sends through the conversation's interface (D-006 of
        # plan 944e2692). MockLLM2Interface has no `complete`, and without one
        # every classifier call fails soft before the provider boundary (see
        # test_interface_without_complete_fails_soft) -- the turn would then
        # complete for a reason unrelated to this fix. Forward `complete` to a
        # real LiteLLMInterface so the patched binding below is reached.
        mock_llm2_interface.model = "gpt-4o"
        mock_llm2_interface.complete = LiteLLMInterface(model="gpt-4o").complete
        api = API.from_definition(fsm, llm_interface=mock_llm2_interface)
        conv_id, _ = api.start_conversation()

        with patch(
            "fsm_llm.llm.providers.litellm.completion", side_effect=_rate_limit_error()
        ) as mock_completion:
            response = api.converse("I am furious", conv_id)

        assert mock_completion.called, "classification boundary was never reached"
        assert isinstance(response, str)
        assert response.strip()
        # The soft-failing field must simply be absent, not half-written.
        assert "sentiment" not in api.get_data(conv_id)

    def test_litellm_exception_is_wrapped_into_the_fsmerror_hierarchy(self):
        with patch(
            "fsm_llm.llm.providers.litellm.completion", side_effect=_rate_limit_error()
        ):
            with pytest.raises(ClassificationError) as exc:
                _classifier().classify("hello")

        assert isinstance(exc.value, FSMError)
        # The LLM layer wraps the provider error (LLMResponseError from
        # `complete`, plan 944e2692 D-002); the provider error stays chained.
        assert isinstance(exc.value.__cause__, LLMResponseError)
        assert isinstance(exc.value.__cause__.__cause__, RateLimitError)

    def test_wrapped_error_is_caught_by_the_pipelines_existing_except_tuple(self):
        """A-3: wrapping at the source is enough; the tuple is NOT widened."""
        config = _make_config()
        state = _make_state(classification_extractions=[config])
        pipeline = _make_pipeline()
        instance = _make_instance()

        with patch(
            "fsm_llm.llm.providers.litellm.completion", side_effect=_rate_limit_error()
        ):
            data = ConversationPass1(pipeline)._execute_classification_extractions(
                state, "test", instance, "conv1"
            )

        assert data == {}

    def test_keyboard_interrupt_still_propagates_bare(self):
        """BaseException must never be wrapped -- HARD system invariant."""
        with patch(
            "fsm_llm.llm.providers.litellm.completion", side_effect=KeyboardInterrupt()
        ):
            with pytest.raises(KeyboardInterrupt):
                _classifier().classify("hello")

    def test_system_exit_still_propagates_bare(self):
        with patch(
            "fsm_llm.llm.providers.litellm.completion", side_effect=SystemExit()
        ):
            with pytest.raises(SystemExit):
                _classifier().classify("hello")

    def test_hierarchical_classifier_inherits_the_wrapping(self):
        """Blast radius the plan flagged UNVERIFIED: it composes Classifier."""
        domain = ClassificationSchema(
            intents=_make_intents("billing", "support"),
            fallback_intent="support",
        )
        hier = HierarchicalClassifier(
            schema=HierarchicalSchema(
                domain_schema=domain,
                intent_schemas={
                    "billing": ClassificationSchema(
                        intents=_make_intents("refund", "invoice"),
                        fallback_intent="invoice",
                    )
                },
            ),
            model="gpt-4o",
        )

        with patch(
            "fsm_llm.llm.providers.litellm.completion", side_effect=_rate_limit_error()
        ):
            with pytest.raises(ClassificationError):
                hier.classify("where is my refund")


class TestClassifierMalformedResponseShape:
    """RED-before regression for review N10 (plan-2026-09-20T165703-0d9c218e).

    `Classifier._call_llm` chased `response.choices[0].message.content` bare,
    so a `completion()` return that does not raise but carries a malformed
    shape leaked `AttributeError` -- a type NOT in the pipeline's
    `_CLASSIFICATION_SOFT_FAIL_EXCEPTIONS`, so a `required=False` field
    would crash the turn instead of failing soft.

    Since plan 944e2692 (D-002) the reply is read by the LLM layer's
    `complete`, which raises `LLMResponseError` for every unreadable shape,
    and the classifier turns every failure of that call into
    `ClassificationError` (a member of the soft-fail tuple, WITHOUT widening
    it). The shape error stays chained under the `LLMResponseError`.

    Not reachable through genuine litellm 1.x objects (`Choices`/`Message`
    always carry `.message`/`.content`); reachable via a patched/mocked
    `completion` or a future litellm type change. Driven with bare
    `SimpleNamespace` objects on purpose: a `MagicMock` auto-creates any
    attribute and can never exercise the missing-attribute path.
    """

    def test_choice_without_message_raises_classification_error(self):
        """RED on the N10 code: AttributeError('... no attribute message')."""
        response = types.SimpleNamespace(choices=[types.SimpleNamespace()])
        with patch("fsm_llm.llm.providers.litellm.completion", return_value=response):
            with pytest.raises(ClassificationError, match="Malformed") as exc:
                _classifier().classify("x")

        assert isinstance(exc.value.__cause__, LLMResponseError)
        assert isinstance(exc.value.__cause__.__cause__, AttributeError)

    def test_message_without_content_raises_classification_error(self):
        response = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace())]
        )
        with patch("fsm_llm.llm.providers.litellm.completion", return_value=response):
            with pytest.raises(ClassificationError, match="Malformed"):
                _classifier().classify("x")

    def test_empty_choices_guard_keeps_its_original_message(self):
        """The empty-reply guard keeps its wording, now in the LLM layer."""
        response = types.SimpleNamespace(choices=[])
        with patch("fsm_llm.llm.providers.litellm.completion", return_value=response):
            with pytest.raises(ClassificationError, match="Empty response"):
                _classifier().classify("x")

    def test_parse_failure_inside_extract_response_passes_through_unchanged(self):
        """A parse failure raised inside `_extract_response` keeps its own
        type and message (it is not a failure of the call)."""
        response = types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(message=types.SimpleNamespace(content="not json"))
            ]
        )
        with patch("fsm_llm.llm.providers.litellm.completion", return_value=response):
            with pytest.raises(ClassificationResponseError, match="Failed to parse"):
                _classifier().classify("x")

    def test_choices_as_dict_is_a_classification_error(self):
        """`choices` as a non-empty dict makes `choices[0]` a `KeyError`. The
        LLM layer reads it as an unreadable shape like the others (it used to
        escape the classifier bare and was caught only because `KeyError` is
        in the pipeline's soft-fail tuple)."""
        response = types.SimpleNamespace(choices={"a": 1})
        with patch("fsm_llm.llm.providers.litellm.completion", return_value=response):
            with pytest.raises(ClassificationError, match="Malformed") as exc:
                _classifier().classify("x")

        assert isinstance(exc.value.__cause__.__cause__, KeyError)


# ----------------------------------------------------------
# F-16: multi-intent truncation observability
# ----------------------------------------------------------


class TestMultiIntentTruncationWarning:
    @staticmethod
    def _parse(n_intents: int, max_intents: int) -> list:
        names = [f"intent{i}" for i in range(n_intents)]
        clf = Classifier(
            schema=ClassificationSchema(
                intents=_make_intents(*names),
                fallback_intent=names[0],
            ),
            model="gpt-4o",
            config=ClassificationPromptConfig(max_intents=max_intents),
        )
        data = {
            "intents": [
                # Distinct descending confidences so the cut is unambiguous.
                {"intent": name, "confidence": 0.99 - i * 0.05}
                for i, name in enumerate(names)
            ]
        }

        records: list = []
        sink_id = logger.add(
            lambda message: records.append(message.record), level="WARNING"
        )
        # fsm_llm calls logger.disable("fsm_llm") at import; without this the
        # sink receives nothing and the negative case passes for the wrong reason.
        logger.enable("fsm_llm")
        try:
            result = clf._parse_multi(data)
        finally:
            logger.remove(sink_id)

        assert len(result.intents) == min(n_intents, MAX_MULTI_INTENTS)
        return [r["message"] for r in records]

    # max_intents is now bounded at construction (1..MAX_MULTI_INTENTS), so the
    # over-return this class exercises is the LLM ignoring the prompt's cap, not
    # a config asking for more than the result model holds.
    def test_warns_naming_the_discarded_count_when_truncating(self):
        messages = self._parse(n_intents=8, max_intents=5)

        truncation = [m for m in messages if "truncated" in m]
        assert len(truncation) == 1
        # The count actually discarded (8 - 5), the total, and the request.
        assert "discarding 3 of 8" in truncation[0]
        assert "max_intents=5" in truncation[0]

    def test_silent_when_within_the_cap(self):
        assert self._parse(n_intents=5, max_intents=5) == []

    def test_silent_when_below_the_cap(self):
        assert self._parse(n_intents=3, max_intents=5) == []


class TestClassificationPromptConfigValidation:
    """Classification audit #3 / #9: bounds checked at construction.

    `max_intents` above the `MultiClassificationResult.intents` cap used to be
    accepted and silently truncated at parse time; `max_tokens=0` and an
    out-of-range temperature surfaced only as a provider error (or, on Ollama,
    not at all, since `apply_ollama_params` forces temperature=0).
    """

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("max_intents", 0),
            ("max_intents", 6),
            ("max_tokens", 0),
            ("temperature", -0.1),
            ("temperature", 2.1),
        ],
    )
    def test_out_of_range_raises_naming_the_field(self, field, value):
        with pytest.raises(ValueError, match=field):
            ClassificationPromptConfig(**{field: value})

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("max_intents", 1),
            ("max_intents", 5),
            ("max_tokens", 1),
            ("temperature", 0.0),
            ("temperature", 2.0),
        ],
    )
    def test_boundary_values_accepted(self, field, value):
        assert getattr(ClassificationPromptConfig(**{field: value}), field) == value

    def test_max_intents_bound_is_the_result_model_cap(self):
        """The lockstep, asserted: the bound IS the pydantic cap, one constant."""
        metadata = MultiClassificationResult.model_fields["intents"].metadata
        cap = next(
            (m.max_length for m in metadata if getattr(m, "max_length", None)),
            None,
        )
        assert cap == MAX_MULTI_INTENTS
        ClassificationPromptConfig(max_intents=cap)
        with pytest.raises(ValueError, match="max_intents"):
            ClassificationPromptConfig(max_intents=cap + 1)

    def test_bad_prompt_config_is_rejected_at_load(self):
        """Step 8 (D-011): the same bound now fails when the config is built."""
        with pytest.raises(ValueError, match="max_intents"):
            _make_config(prompt_config={"max_intents": 0})

    def test_bad_prompt_config_on_a_state_fails_the_turn(self, mock_llm2_interface):
        """Defence in depth: a bad `prompt_config` that bypassed load
        validation (plain attribute assignment) on a soft field.

        A broken configuration is not a classifier outage: since D-037 of plan
        944e2692 only `ClassificationError` is soft, so the turn fails (it was
        a logged warning with the key unset) and no classifier is built.
        """
        config = _make_config()
        config.prompt_config = {"max_intents": 0}
        fsm = _make_fsm(
            {
                "triage": _make_state(classification_extractions=[config]),
                "end": _terminal_state("end"),
            }
        )
        mock_llm2_interface.model = "gpt-4o"
        # An FSM that classifies needs an interface with `complete` (D-029 of
        # plan 944e2692); the failure under test is the prompt config's.
        mock_llm2_interface.complete = LiteLLMInterface(model="gpt-4o").complete
        api = API.from_definition(fsm, llm_interface=mock_llm2_interface)
        conv_id, _ = api.start_conversation()

        with (
            patch("fsm_llm.llm.providers.litellm.completion") as mock_completion,
            pytest.raises(FSMError, match="max_intents"),
        ):
            api.converse("I am furious", conv_id)

        assert not mock_completion.called, "classifier must not be built"
        assert "sentiment" not in api.get_data(conv_id)


_NON_ASCII_NAMES = ["café", "名前", "naïve_intent", "буя", "intent_µ"]
_ASCII_OK_NAMES = ["buy", "buy_now", "_private", "Intent2", "a", "A_1_b"]
_ASCII_BAD_NAMES = ["2fast", "buy-now", "buy now", "buy!", ""]


class TestIntentNameCharsetMatchesSiblingIdentifiers:
    """F-21 / SC-22.

    `IntentDefinition.name` used `self.name.replace("_", "").isalnum()`, which is
    Unicode-aware, while `State.id` and `Transition.target_state` use a strict
    ASCII `pattern=`. So `café`/名前 passed as an intent name and failed as a
    state id -- two identifier policies in one schema. The census that gated this
    change found zero non-ASCII intent names in the repo, so the hard alignment
    was taken.

    These assert LOCKSTEP with the sibling rather than mere rejection: a test that
    only said "IntentDefinition rejects 'café'" would still pass if the two rules
    drifted apart again in the other direction.
    """

    @staticmethod
    def _intent_accepts(name: str) -> bool:
        try:
            IntentDefinition(name=name, description="d")
            return True
        except ValueError:
            return False

    @staticmethod
    def _state_accepts(name: str) -> bool:
        try:
            State(id=name, description="d", purpose="p")
            return True
        except ValueError:
            return False

    @pytest.mark.parametrize("name", _NON_ASCII_NAMES)
    def test_non_ascii_intent_name_is_rejected(self, name):
        with pytest.raises(ValueError, match="alphanumeric ASCII"):
            IntentDefinition(name=name, description="d")

    @pytest.mark.parametrize(
        "name", _NON_ASCII_NAMES + _ASCII_OK_NAMES + _ASCII_BAD_NAMES
    )
    def test_intent_name_policy_is_identical_to_state_id_policy(self, name):
        assert self._intent_accepts(name) == self._state_accepts(name), (
            f"{name!r}: IntentDefinition and State.id disagree on the same identifier"
        )

    @pytest.mark.parametrize("name", _ASCII_OK_NAMES)
    def test_conformant_names_still_accepted(self, name):
        assert self._intent_accepts(name) is True

    def test_rejection_is_a_value_error_so_the_validator_reports_it_as_an_error(self):
        """The enforcement MECHANISM is load-bearing, not incidental.

        A `model_validator` raising ValueError yields `value_error`, which
        validator.py's ALLOW-list promotes to ERROR tier, so `fsm-llm-validate`
        and `API.from_file` agree on a non-conformant intent name. This test
        fails if someone "simplifies" the rule into a `Field(pattern=)`.

        # DECISION plan-2026-07-20T040150-876e7164/D-001 [STALE]
        The original rationale here was that `Field(pattern=)` emits
        `string_pattern_mismatch`, "which validator.py's ALLOW-list (D-013)
        deliberately excludes". That is NO LONGER TRUE: the D-013 carve-out was
        measured to be a live false green and `string_pattern_mismatch` is now
        promoted, so a `pattern=` rewrite would no longer break agreement. The
        `model_validator` shape is still preferred -- it produces the specific,
        actionable message this test asserts, rather than a raw regex dump -- but
        the agreement argument no longer applies and has been removed rather than
        left standing as a falsified comment.
        """
        import pydantic

        with pytest.raises(pydantic.ValidationError) as exc:
            IntentDefinition(name="café", description="d")
        assert [e["type"] for e in exc.value.errors()] == ["value_error"]


# ----------------------------------------------------------
# NL1: classification thinking-recovery must read reasoning_content
# ----------------------------------------------------------


class TestClassifierReasoningContentRecovery:
    """RED-before regression for NL1 (plan-2026-07-21T072826-e3131cc2).

    `Classifier._extract_response` recovered reasoning-only replies with
    `getattr(msg, "thinking", None)` only. The installed litellm range RENAMES
    the raw `thinking` field to `reasoning_content` and DELETES `thinking`
    before building the Message (see llm.py D-002 / C2). So on the project's own
    `DEFAULT_LLM_MODEL = ollama_chat/qwen3.5:4b`, a classification reply that
    puts its JSON in `reasoning_content` with empty `content` used to raise
    `ClassificationResponseError("LLM returned empty content")` instead of being
    recovered — crashing AMBIGUOUS-transition resolution.

    Driven by a REAL `litellm.types.utils.Message`, never a `.thinking` stub:
    a stub is GREEN on the broken pre-fix code (it never probes the gap), which
    is exactly how the original C2 bug shipped. See llm.py TestReasoningContentRecovery.
    """

    def test_classify_recovers_intent_from_reasoning_content(self):
        """SC-1: real Message(content="", reasoning_content=intent-json) → recovered."""
        from litellm.types.utils import Message

        msg = Message(
            content="",
            reasoning_content='{"intent": "positive", "confidence": 0.9, "reasoning": "clearly happy"}',
        )
        # Contract precondition: a real litellm Message has no `.thinking` attr,
        # so the pre-fix `getattr(msg, "thinking", None)` gate returns None here.
        assert not hasattr(msg, "thinking"), (
            "real litellm Message must not expose a `.thinking` attr — if this "
            "fails the negative control no longer documents the NL1 gap"
        )
        choice = MagicMock()
        choice.message = msg
        response = MagicMock()
        response.choices = [choice]

        with patch(
            "fsm_llm.llm.providers.litellm.completion", return_value=response
        ) as mock_completion:
            result = _classifier().classify("I am thrilled")

        assert mock_completion.called
        assert isinstance(result, ClassificationResult)
        assert result.intent == "positive"
        assert result.confidence == pytest.approx(0.9)

    def test_extract_response_recovers_from_reasoning_content(self):
        """SC-1 (unit): the LLM layer's reply reader feeds `_extract_response`."""
        from litellm.types.utils import Message

        msg = Message(content="", reasoning_content='{"intent": "negative"}')
        assert not hasattr(msg, "thinking")
        choice = MagicMock()
        choice.message = msg
        response = MagicMock()
        response.choices = [choice]

        data = Classifier._extract_response(_completion_response(response))
        assert data == {"intent": "negative"}


class TestClassificationNaNConfidence:
    """G6: a ``NaN`` confidence must NOT be silently promoted to ``1.0`` (max
    certainty), which would defeat ``is_below_default_threshold``. ``float('nan')`` is
    what real ``json.loads('{"confidence": NaN}')`` yields, so it is passed to
    the already-parsed ``data`` dict exactly as production would.
    """

    def test_parse_single_nan_confidence_is_low_confidence(self):
        clf = _classifier()
        result = clf._parse_single({"intent": "positive", "confidence": float("nan")})
        assert result.confidence == 0.0
        assert result.is_below_default_threshold

    def test_parse_single_infinity_confidence_is_low_confidence(self):
        clf = _classifier()
        result = clf._parse_single({"intent": "positive", "confidence": float("inf")})
        assert result.confidence == 0.0

    def test_parse_multi_nan_confidence_is_low_confidence(self):
        clf = _classifier()
        result = clf._parse_multi(
            {"intents": [{"intent": "positive", "confidence": float("nan")}]}
        )
        assert result.intents[0].confidence == 0.0

    def test_parse_single_valid_confidence_preserved(self):
        clf = _classifier()
        result = clf._parse_single({"intent": "positive", "confidence": 0.7})
        assert result.confidence == 0.7

    def test_parse_single_non_numeric_confidence_still_defaults_zero(self):
        clf = _classifier()
        result = clf._parse_single({"intent": "positive", "confidence": {"bad": 1}})
        assert result.confidence == 0.0


class _ClassifiedHarness:
    """`API.converse` with a scripted classifier and a spying fake completion.

    ``plain_calls`` records the system prompt of every per-field extraction
    call (the ones that carry a ``response_format`` and name a field).
    """

    def __init__(self, fsm: dict, intent: str, confidence: float, plain_value="buy"):
        self.fsm = fsm
        self.intent = intent
        self.confidence = confidence
        self.plain_value = plain_value
        self.plain_calls: list[str] = []

    def _completion(self, **kwargs):
        system = kwargs["messages"][0]["content"]
        if kwargs.get("response_format") is not None:
            self.plain_calls.append(system)
            return _fake_response(
                json.dumps(
                    {"field_name": "x", "value": self.plain_value, "confidence": 0.95}
                )
            )
        return _fake_response(json.dumps({"message": "ok", "reasoning": ""}))

    def run(self):
        from fsm_llm.core.definitions import ClassificationResult

        classifier = MagicMock()
        classifier.classify.return_value = ClassificationResult(
            reasoning="r", intent=self.intent, confidence=self.confidence
        )
        with (
            patch(
                "fsm_llm.llm.providers.litellm.completion", side_effect=self._completion
            ),
            patch(
                "fsm_llm.llm.providers.litellm.get_supported_openai_params",
                return_value=["response_format"],
            ),
            patch(
                "fsm_llm.engine.turn.transitions.Classifier", return_value=classifier
            ),
        ):
            api = API.from_definition(self.fsm, model="gpt-4o", api_key="test")
            cid, _ = api.start_conversation()
            api.converse("I would like to buy a phone", cid)
            return api.get_data(cid), api.get_current_state(cid)


class TestClassificationOwnedKeyIsNotPlainExtracted:
    def test_no_plain_extract_call_for_classification_key(self):
        h = _ClassifiedHarness(_classified_fsm(), "buy", 0.95)
        data, state = h.run()
        assert h.plain_calls == []
        assert data.get("intent") == "buy"
        assert state == "shop"

    def test_below_threshold_classification_leaves_key_unset_and_gate_closed(self):
        h = _ClassifiedHarness(_classified_fsm(), "buy", 0.2)
        data, state = h.run()
        assert h.plain_calls == []
        assert "intent" not in data
        assert state == "triage"

    def test_fallback_intent_is_still_stored(self):
        h = _ClassifiedHarness(_classified_fsm(), "browse", 0.1)
        data, state = h.run()
        assert data.get("intent") == "browse"
        assert state == "triage"

    def test_ordinary_required_key_is_still_plain_extracted(self):
        """Vacuity guard: the exclusion is per-key, not a disabled extractor."""
        h = _ClassifiedHarness(_classified_fsm(topic=True), "buy", 0.95, "phones")
        data, _ = h.run()
        assert len(h.plain_calls) == 1
        assert "'topic'" in h.plain_calls[0]
        assert "'intent'" not in h.plain_calls[0]
        assert data.get("topic") == "phones"

    def test_explicit_field_extraction_with_same_name_still_wins(self):
        h = _ClassifiedHarness(
            _classified_fsm(explicit_field_extraction=True), "buy", 0.2
        )
        data, state = h.run()
        assert len(h.plain_calls) == 1
        assert data.get("intent") == "buy"
        assert state == "shop"


class TestBulkPassRespectsClassificationOwnership:
    def test_bulk_cannot_fill_a_classifier_owned_key_below_threshold(self):
        with _ClassBulkProv(_classified_bulk_fsm(), "buy", 0.2) as p:
            data = p.say("hmm maybe something", bulk={"intent": "buy"})
            state = p.api.get_current_state(p.cid)
        assert "intent" not in data
        assert state == "triage"

    def test_ordinary_bulk_keys_still_added_next_to_a_classifier_owned_key(self):
        with _ClassBulkProv(_classified_bulk_fsm(), "buy", 0.2) as p:
            data = p.say("hmm maybe something", bulk={"intent": "buy", "topic": "cats"})
        assert data.get("topic") == "cats"
        assert "intent" not in data

    def test_classifier_above_threshold_still_transitions(self):
        """Vacuity guard: the classification path itself is intact."""
        with _ClassBulkProv(_classified_bulk_fsm(), "buy", 0.95) as p:
            data = p.say("I want to buy a phone", bulk={"topic": "phones"})
            state = p.api.get_current_state(p.cid)
        assert data.get("intent") == "buy"
        assert state == "shop"

    def test_agent_fsm_keeps_the_bulk_fill_for_a_classification_owned_key(self):
        """Agent FSMs (`use_classification=True`) rely on the bulk fill when the
        classifier is below threshold; the `agent_trace` marker keeps it."""
        with _ClassBulkProv(
            _classified_bulk_fsm(), "buy", 0.2, initial_context={"agent_trace": []}
        ) as p:
            data = p.say("hmm maybe something", bulk={"intent": "buy"})
        assert data.get("intent") == "buy"


_A4_RESULTS_KEY = "classification_results"


# The documented metadata keys are pinned as literals: they are the public
# contract a monitor or debugger reads through get_complete_conversation.
def _a4_metadata(api: API, conv_id: str) -> dict[str, Any]:
    return api.fsm_manager.get_complete_conversation(conv_id)["metadata"]


class TestStep04A4:
    """A4: every classification-extraction result is recorded in full under
    ``context.metadata["classification_results"][field_name]`` and the
    transition-classification record is mirrored to
    ``context.metadata["transition_classification"]``. Both are readable via
    ``get_complete_conversation``, stay out of ``get_data`` and ``context.data``
    (except the back-compat transition key), are JSON-native, and roll back
    with the turn.
    """

    def test_a4_classification_result_visible_via_get_complete_conversation(self):
        api, conv_id = _a3_api(_a3_extraction_fsm(context_keys=["topic"]))
        with _ClassifierCapture("buy", 0.9):
            api.converse("I want to buy these", conv_id)

        metadata = _a4_metadata(api, conv_id)
        assert metadata[_A4_RESULTS_KEY]["intent"] == {
            "intent": "buy",
            "confidence": 0.9,
            "reasoning": "r",
            "entities": {},
            "context_snapshot": {"topic": "running shoes"},
        }
        # JSON-native: the whole metadata mapping serialises without default=.
        json.dumps(metadata)

    def test_a4_result_not_in_get_data(self):
        api, conv_id = _a3_api(_a3_extraction_fsm())
        with _ClassifierCapture("buy", 0.9):
            api.converse("I want to buy these", conv_id)

        data = api.get_data(conv_id)
        assert data["intent"] == "buy"
        assert _A4_RESULTS_KEY not in data
        assert not any("classification" in k for k in _raw_data(api, conv_id))

    def test_a4_fallback_and_low_confidence_results_are_recorded(self):
        api, conv_id = _a3_api(_a3_extraction_fsm())
        with _ClassifierCapture("browse", 0.95):
            api.converse("just looking", conv_id)
        record = _a4_metadata(api, conv_id)[_A4_RESULTS_KEY]["intent"]
        assert record["intent"] == "browse"
        assert "low_confidence" not in record

        with _ClassifierCapture("buy", 0.2):
            api.converse("maybe buy?", conv_id)
        record = _a4_metadata(api, conv_id)[_A4_RESULTS_KEY]["intent"]
        assert record["intent"] == "buy"
        assert record["low_confidence"] is True
        # The discarded result never became the field value.
        assert api.get_data(conv_id)["intent"] == "browse"
        assert api.get_current_state(conv_id) == "triage"

    def test_a4_rollback_restores_metadata_result(self):
        llm = _mock_llm()
        api = API.from_definition(_a3_extraction_fsm(), llm_interface=llm)
        conv_id, _ = api.start_conversation()
        with _ClassifierCapture("browse", 0.95):
            api.converse("just looking", conv_id)
        prior = _a4_metadata(api, conv_id)[_A4_RESULTS_KEY]

        llm.generate_response.side_effect = RuntimeError("pass 2 down")
        with _ClassifierCapture("buy", 0.9), pytest.raises(FSMError):
            api.converse("buy it now", conv_id)

        assert api.get_current_state(conv_id) == "triage"
        assert _a4_metadata(api, conv_id)[_A4_RESULTS_KEY] == prior
        assert prior["intent"]["intent"] == "browse"

    def test_a4_earlier_snapshot_not_mutated_by_later_turn(self):
        api, conv_id = _a3_api(_a3_extraction_fsm())
        with _ClassifierCapture("browse", 0.95):
            api.converse("just looking", conv_id)
        earlier = _a4_metadata(api, conv_id)

        with _ClassifierCapture("browse", 0.7):
            api.converse("still looking", conv_id)

        assert earlier[_A4_RESULTS_KEY]["intent"]["confidence"] == 0.95
        later = _a4_metadata(api, conv_id)
        assert later[_A4_RESULTS_KEY]["intent"]["confidence"] == 0.7

    def test_a4_context_snapshot_keeps_json_native_values_only(self):
        api = API.from_definition(
            _a3_extraction_fsm(context_keys=["topic", "handle"]),
            llm_interface=_mock_llm(),
        )
        conv_id, _ = api.start_conversation(
            initial_context={"topic": "running shoes", "handle": object()}
        )
        with _ClassifierCapture("browse", 0.95):
            api.converse("just looking", conv_id)

        record = _a4_metadata(api, conv_id)[_A4_RESULTS_KEY]["intent"]
        assert record["context_snapshot"] == {"topic": "running shoes"}
        json.dumps(record)

    def test_a4_context_snapshot_drops_secret_shaped_entries_at_every_depth(self):
        """D-031 of plan 8b258a25 (step 80 pins it: a scratch mutant that returned the
        unfiltered copy survived every test): the stored snapshot is filtered with the
        never-truncating data walker, so a secret-looking entry never reaches
        ``metadata``, which ``get_complete_conversation`` and the session file expose."""
        api = API.from_definition(
            _a3_extraction_fsm(context_keys=["topic", "profile", "api_key"]),
            llm_interface=_mock_llm(),
        )
        conv_id, _ = api.start_conversation(
            initial_context={
                "topic": "running shoes",
                "profile": {"name": "Ada", "password": "hunter2hunter2"},
                "api_key": "sk-abcdefghijklmnopqrstuvwxyz123456",
            }
        )
        with _ClassifierCapture("browse", 0.95):
            api.converse("just looking", conv_id)

        record = _a4_metadata(api, conv_id)[_A4_RESULTS_KEY]["intent"]
        assert record["context_snapshot"] == {
            "topic": "running shoes",
            "profile": {"name": "Ada"},
        }

    def test_a4_transition_record_in_metadata(self):
        api, conv_id, _ = _api(_ambiguous_then_blocked_fsm())
        with patch("fsm_llm.engine.turn.transitions.Classifier") as mock_cls:
            mock_cls.return_value.classify.return_value = _classification("a", 0.95)
            api.converse("a please", conv_id)

        assert api.get_current_state(conv_id) == "a"
        mirrored = _a4_metadata(api, conv_id)[_A4_TRANSITION_KEY]
        assert "_transition_classification_result" not in _raw_data(api, conv_id)
        assert mirrored["intent"] == "a"
        assert mirrored["confidence"] == pytest.approx(0.95)
        json.dumps(mirrored)

        # A8's turn-start clear covers the metadata mirror too.
        api.converse("still here", conv_id)
        assert _A4_TRANSITION_KEY not in _a4_metadata(api, conv_id)

    def test_a4_transition_low_confidence_and_failure_mirrored(self):
        api, conv_id, _ = _api(
            _ambiguous_then_blocked_fsm({"confidence_threshold": 0.9})
        )
        # One patch for both turns: the pipeline caches the classifier.
        with patch("fsm_llm.engine.turn.transitions.Classifier") as mock_cls:
            mock_cls.return_value.classify.side_effect = [
                _classification("a", 0.05),
                ClassificationError("down"),
            ]
            api.converse("maybe a?", conv_id)
            assert _a4_metadata(api, conv_id)[_A4_TRANSITION_KEY]["low_confidence"]
            api.converse("again", conv_id)
        mirrored = _a4_metadata(api, conv_id)[_A4_TRANSITION_KEY]
        assert mirrored == {"error": "down", "fallback": True}

    def test_a4_session_save_restore_unaffected(self, tmp_path):
        from fsm_llm.engine.session import FileSessionStore

        api = API.from_definition(
            _a3_extraction_fsm(),
            llm_interface=_mock_llm(),
            session_store=FileSessionStore(tmp_path),
        )
        conv_id, _ = api.start_conversation()
        with _ClassifierCapture("browse", 0.95):
            api.converse("just looking", conv_id)
        api.save_session(conv_id)

        restored = api.restore_session(conv_id)
        assert restored is not None
        new_conv_id, state = restored
        assert api.get_data(new_conv_id)["intent"] == "browse"
        # Per-turn debug records are not session state; only provenance is.
        assert set(state.metadata) == {"pipeline_extracted"}


def _a5_config(field_name: str) -> ClassificationExtractionConfig:
    return ClassificationExtractionConfig(
        field_name=field_name,
        intents=[
            IntentDefinition(name="buy", description="wants to buy"),
            IntentDefinition(name="browse", description="just looking"),
        ],
        fallback_intent="browse",
        confidence_threshold=0.5,
    )


def _a5_fsm(
    fields: tuple[str, ...] = ("intent",),
    handler_only_keys: list[str] | None = None,
) -> FSMDefinition:
    """``greet`` owns no extraction and always moves to ``triage`` (one
    unconditioned transition, DETERMINISTIC); ``triage`` owns one
    classification field per name in ``fields`` and never leaves (its one
    transition is gated on a key nobody sets), so no second transition runs."""
    states = {
        "greet": State(
            id="greet",
            description="Greet",
            purpose="Say hello",
            response_instructions="Respond",
            transitions=[Transition(target_state="triage", description="Always")],
        ),
        "triage": State(
            id="triage",
            description="Triage",
            purpose="Route the shopper",
            response_instructions="Respond",
            classification_extractions=[_a5_config(f) for f in fields],
            transitions=[
                Transition(
                    target_state="done",
                    description="Never",
                    conditions=[
                        TransitionCondition(
                            description="never set",
                            logic={"==": [{"var": "never_set_flag"}, True]},
                        )
                    ],
                )
            ],
        ),
        "done": State(id="done", description="Done", purpose="End", transitions=[]),
    }
    return FSMDefinition(
        name="a5_fsm",
        description="A5 FSM",
        initial_state="greet",
        states=states,
        handler_only_keys=handler_only_keys or [],
    )


def _a5_self_loop_fsm() -> FSMDefinition:
    """``triage`` owns ``intent``; its only passing transition is an
    unconditioned self-loop (the edge to ``done`` is gated on a key nobody
    sets)."""
    never = TransitionCondition(
        description="never set", logic={"==": [{"var": "never_set_flag"}, True]}
    )
    states = {
        "triage": State(
            id="triage",
            description="Triage",
            purpose="Route the shopper",
            response_instructions="Respond",
            classification_extractions=[_a5_config("intent")],
            transitions=[
                Transition(target_state="triage", description="Again"),
                Transition(
                    target_state="done", description="Never", conditions=[never]
                ),
            ],
        ),
        "done": State(id="done", description="Done", purpose="End", transitions=[]),
    }
    return FSMDefinition(
        name="a5_loop", description="A5 loop", initial_state="triage", states=states
    )


class TestStep05A5:
    """A5: after a transition into a different state, each of that state's
    classification fields that is still unset (absent or None) is classified
    on the same message, one classifier call per field. Self-loops,
    agent-managed FSMs, already-set fields and ``handler_only_keys`` cost no
    call (plan-2026-09-21T203800-8a03483a/D-006)."""

    def test_a5_intent_for_next_state_is_classified_after_transition(self):
        api, conv_id, _ = _api(_a5_fsm())
        with _ClassifierCapture("buy", 0.9) as cap:
            api.converse("hi, I want to buy running shoes", conv_id)

        assert api.get_current_state(conv_id) == "triage"
        assert api.get_data(conv_id)["intent"] == "buy"
        assert len(cap.calls) == 1
        record = _a4_metadata(api, conv_id)[_A4_RESULTS_KEY]["intent"]
        assert record["intent"] == "buy"

    def test_a5_stream_path_classifies_too(self):
        api, conv_id, _ = _api(_a5_fsm())
        with _ClassifierCapture("buy", 0.9) as cap:
            assert "".join(api.converse_stream("hi, buy please", conv_id)) == "ok"

        assert api.get_current_state(conv_id) == "triage"
        assert api.get_data(conv_id)["intent"] == "buy"
        assert len(cap.calls) == 1

    def test_a5_exactly_one_extra_call_per_unset_field(self):
        api, conv_id, _ = _api(_a5_fsm(fields=("intent", "mood")))
        with _ClassifierCapture("buy", 0.9) as cap:
            api.converse("hi, I want to buy", conv_id)

        assert len(cap.calls) == 2
        data = api.get_data(conv_id)
        assert data["intent"] == "buy"
        assert data["mood"] == "buy"

    def test_a5_already_set_field_costs_no_call(self):
        api = API.from_definition(
            _a5_fsm(fields=("intent", "mood")), llm_interface=_mock_llm()
        )
        conv_id, _ = api.start_conversation(initial_context={"intent": "browse"})
        with _ClassifierCapture("buy", 0.9) as cap:
            api.converse("hi, I want to buy", conv_id)

        assert len(cap.calls) == 1
        data = api.get_data(conv_id)
        assert data["intent"] == "browse"
        assert data["mood"] == "buy"

    def test_a5_self_loop_costs_no_call(self):
        """Guard (passes on the pre-step source): the state's own pass just
        classified this message; a below-threshold result leaves the field
        unset, and the self-loop must not buy a second call for it."""
        api, conv_id, _ = _api(_a5_self_loop_fsm())
        with _ClassifierCapture("buy", 0.2) as cap:
            api.converse("maybe buy?", conv_id)

        assert len(cap.calls) == 1
        assert "intent" not in api.get_data(conv_id)

    def test_a5_handler_only_key_not_classified(self):
        """Guard (passes on the pre-step source)."""
        api, conv_id, _ = _api(_a5_fsm(handler_only_keys=["intent"]))
        with _ClassifierCapture("buy", 0.9) as cap:
            api.converse("hi, I want to buy", conv_id)

        assert api.get_current_state(conv_id) == "triage"
        assert cap.calls == []
        assert "intent" not in api.get_data(conv_id)

    def test_a5_agent_managed_fsm_costs_no_call(self):
        """Guard (passes on the pre-step source)."""
        api, conv_id, _ = _api(_a5_fsm())
        api.update_context(conv_id, {"agent_trace": []})
        with _ClassifierCapture("buy", 0.9) as cap:
            api.converse("hi, I want to buy", conv_id)

        assert api.get_current_state(conv_id) == "triage"
        assert cap.calls == []
