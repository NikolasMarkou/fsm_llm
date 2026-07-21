"""Tests for classification-based extraction (classification_extractions on State)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from litellm.exceptions import RateLimitError

from fsm_llm.api import API
from fsm_llm.classification import Classifier, HierarchicalClassifier
from fsm_llm.constants import CLASSIFICATION_EXTRACTION_RESULT_SUFFIX
from fsm_llm.definitions import (
    ClassificationError,
    ClassificationExtractionConfig,
    ClassificationResult,
    ClassificationSchema,
    FSMContext,
    FSMDefinition,
    FSMError,
    FSMInstance,
    HierarchicalSchema,
    IntentDefinition,
    State,
    Transition,
)
from fsm_llm.handlers import HandlerSystem
from fsm_llm.logging import logger
from fsm_llm.pipeline import MessagePipeline
from fsm_llm.prompts import (
    ClassificationPromptConfig,
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


class TestIntentsCapMatchesRuntimeSibling:
    """F-09 / SC-13.

    `ClassificationExtractionConfig.intents` had no upper bound while the
    `ClassificationSchema` that `MessagePipeline._execute_classification_extractions`
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
        path.write_text(json.dumps(fsm))

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

    def test_fallback_intent_stored(self):
        """Fallback intent is always stored so downstream JsonLogic has the key."""
        config = _make_config()
        state = _make_state(classification_extractions=[config])
        pipeline = _make_pipeline()
        instance = _make_instance()

        result = _mock_classify_result("neutral", 0.95)  # neutral is fallback

        with patch.object(Classifier, "classify", return_value=result):
            data = pipeline._execute_classification_extractions(
                state, "what time is it", instance, "conv1"
            )

        assert data["sentiment"] == "neutral"

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
        # MockLLM2Interface carries no .model, and without one the pipeline
        # skips classification entirely (see test_no_model_available) -- the
        # turn would then complete for a reason unrelated to this fix.
        mock_llm2_interface.model = "gpt-4o"
        api = API.from_definition(fsm, llm_interface=mock_llm2_interface)
        conv_id, _ = api.start_conversation()

        with patch(
            "fsm_llm.classification.completion", side_effect=_rate_limit_error()
        ) as mock_completion:
            response = api.converse("I am furious", conv_id)

        assert mock_completion.called, "classification boundary was never reached"
        assert isinstance(response, str)
        assert response.strip()
        # The soft-failing field must simply be absent, not half-written.
        assert "sentiment" not in api.get_data(conv_id)

    def test_litellm_exception_is_wrapped_into_the_fsmerror_hierarchy(self):
        with patch(
            "fsm_llm.classification.completion", side_effect=_rate_limit_error()
        ):
            with pytest.raises(ClassificationError) as exc:
                _classifier().classify("hello")

        assert isinstance(exc.value, FSMError)
        assert isinstance(exc.value.__cause__, RateLimitError)

    def test_wrapped_error_is_caught_by_the_pipelines_existing_except_tuple(self):
        """A-3: wrapping at the source is enough; the tuple is NOT widened."""
        config = _make_config()
        state = _make_state(classification_extractions=[config])
        pipeline = _make_pipeline()
        instance = _make_instance()

        with patch(
            "fsm_llm.classification.completion", side_effect=_rate_limit_error()
        ):
            data = pipeline._execute_classification_extractions(
                state, "test", instance, "conv1"
            )

        assert data == {}

    def test_keyboard_interrupt_still_propagates_bare(self):
        """BaseException must never be wrapped -- HARD system invariant."""
        with patch(
            "fsm_llm.classification.completion", side_effect=KeyboardInterrupt()
        ):
            with pytest.raises(KeyboardInterrupt):
                _classifier().classify("hello")

    def test_system_exit_still_propagates_bare(self):
        with patch("fsm_llm.classification.completion", side_effect=SystemExit()):
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
            "fsm_llm.classification.completion", side_effect=_rate_limit_error()
        ):
            with pytest.raises(ClassificationError):
                hier.classify("where is my refund")


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

        assert len(result.intents) == min(n_intents, 5)
        return [r["message"] for r in records]

    def test_warns_naming_the_discarded_count_when_truncating(self):
        messages = self._parse(n_intents=8, max_intents=10)

        truncation = [m for m in messages if "truncated" in m]
        assert len(truncation) == 1
        # The count actually discarded (8 - 5), the total, and the request.
        assert "discarding 3 of 8" in truncation[0]
        assert "max_intents=10" in truncation[0]

    def test_silent_when_within_the_cap(self):
        assert self._parse(n_intents=5, max_intents=10) == []

    def test_silent_when_below_the_cap(self):
        assert self._parse(n_intents=3, max_intents=10) == []


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

        # DECISION plan-2026-07-20T040150-876e7164/D-001
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
            "fsm_llm.classification.completion", return_value=response
        ) as mock_completion:
            result = _classifier().classify("I am thrilled")

        assert mock_completion.called
        assert isinstance(result, ClassificationResult)
        assert result.intent == "positive"
        assert result.confidence == pytest.approx(0.9)

    def test_extract_response_recovers_from_reasoning_content(self):
        """SC-1 (unit): drive `_extract_response` directly at the seam."""
        from litellm.types.utils import Message

        msg = Message(content="", reasoning_content='{"intent": "negative"}')
        assert not hasattr(msg, "thinking")
        choice = MagicMock()
        choice.message = msg
        response = MagicMock()
        response.choices = [choice]

        data = Classifier._extract_response("", response)
        assert data == {"intent": "negative"}
