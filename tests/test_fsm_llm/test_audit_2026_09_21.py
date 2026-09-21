"""Regression tests for the 2026-09-21 fsm_llm core audit.

One class per plan step (plan-2026-09-21T203800-8a03483a). Every test is named
after the audit id it pins (``test_a1_*``, ``test_a8_*``, ...) and was run RED
against the pre-step source before the fix landed.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from fsm_llm.api import API
from fsm_llm.constants import (
    CONTEXT_KEY_CLASSIFICATION_RESULT,
    DEFAULT_TRANSITION_CLASSIFICATION_CONFIDENCE,
)
from fsm_llm.definitions import (
    ClassificationResult,
    FieldExtractionResponse,
    FSMDefinition,
    FSMError,
    State,
    Transition,
    TransitionCondition,
)
from fsm_llm.handlers import HandlerTiming
from fsm_llm.llm import LLMInterface

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _mock_llm() -> MagicMock:
    """A spec'd mock LLM: field extraction returns nothing, Pass 2 says "ok"."""
    llm = MagicMock(spec=LLMInterface)
    llm.model = "gpt-4"

    def _extract_field(request):
        return FieldExtractionResponse(
            field_name=request.field_name,
            value=None,
            confidence=0.0,
            reasoning="mock",
            is_valid=False,
        )

    llm.extract_field.side_effect = _extract_field
    llm.generate_response.return_value = MagicMock(
        message="ok", message_type="response", reasoning="mock"
    )
    llm.generate_response_stream.side_effect = lambda request: iter(["o", "k"])
    return llm


def _classification(intent: str, confidence: float) -> ClassificationResult:
    return ClassificationResult(
        reasoning=f"picked {intent}", intent=intent, confidence=confidence
    )


def _ambiguous_then_blocked_fsm(
    transition_classification: dict[str, Any] | None = None,
) -> FSMDefinition:
    """``start`` has two unconditioned, equal-priority transitions (AMBIGUOUS
    every turn). ``a`` has one transition gated on a key nobody sets, so a turn
    in ``a`` is BLOCKED and never reaches the classifier.
    """
    never = TransitionCondition(
        description="never set",
        logic={"==": [{"var": "never_set_flag"}, True]},
    )
    states = {
        "start": State(
            id="start",
            description="Start",
            purpose="Pick a branch",
            response_instructions="Respond",
            transitions=[
                Transition(target_state="a", description="Go to a"),
                Transition(target_state="b", description="Go to b"),
            ],
            transition_classification=transition_classification,
        ),
        "a": State(
            id="a",
            description="Branch a",
            purpose="Wait",
            response_instructions="Respond",
            transitions=[
                Transition(target_state="end", description="Done", conditions=[never])
            ],
        ),
        "b": State(id="b", description="Branch b", purpose="End", transitions=[]),
        "end": State(id="end", description="End", purpose="End", transitions=[]),
    }
    return FSMDefinition(
        name="audit_fsm",
        description="Audit FSM",
        initial_state="start",
        states=states,
    )


def _api(fsm_def: FSMDefinition) -> tuple[API, str, MagicMock]:
    llm = _mock_llm()
    api = API.from_definition(fsm_def, llm_interface=llm)
    conv_id, _ = api.start_conversation()
    return api, conv_id, llm


def _raw_data(api: API, conv_id: str) -> dict[str, Any]:
    """The unfiltered instance context (``get_data`` strips internal keys)."""
    return api.fsm_manager.instances[conv_id].context.data


# ---------------------------------------------------------------------------
# Step 1: A1 (transition-classification threshold) + A8 (stale result)
# ---------------------------------------------------------------------------


class TestStep01A1A8:
    """A1: a transition classification below ``schema.confidence_threshold``
    means "stay" (no transition, no transition handlers). A8: the
    ``_transition_classification_result`` record of an earlier turn is cleared
    at turn start, inside the turn snapshot, on both the sync and stream paths.
    """

    def test_a1_low_confidence_stays_and_fires_no_transition_handlers(self):
        api, conv_id, _ = _api(
            _ambiguous_then_blocked_fsm({"confidence_threshold": 0.9})
        )
        pre_transition_calls: list[dict[str, Any]] = []
        api.create_handler(
            "count_pre_transition",
            HandlerTiming.PRE_TRANSITION,
            lambda ctx: pre_transition_calls.append(dict(ctx)) or {},
        )

        with patch("fsm_llm.pipeline.Classifier") as mock_cls:
            mock_cls.return_value.classify.return_value = _classification("a", 0.05)
            response = api.converse("maybe a?", conv_id)

        assert mock_cls.return_value.classify.called
        assert isinstance(response, str)
        assert api.get_current_state(conv_id) == "start"
        assert pre_transition_calls == []
        record = _raw_data(api, conv_id)[CONTEXT_KEY_CLASSIFICATION_RESULT]
        assert record["intent"] == "a"
        assert record["confidence"] == pytest.approx(0.05)
        assert record["low_confidence"] is True

    def test_a1_confidence_at_threshold_transitions(self):
        api, conv_id, _ = _api(
            _ambiguous_then_blocked_fsm({"confidence_threshold": 0.9})
        )
        with patch("fsm_llm.pipeline.Classifier") as mock_cls:
            mock_cls.return_value.classify.return_value = _classification("a", 0.9)
            api.converse("a please", conv_id)

        # The comparison is strict: confidence == threshold is confident.
        assert api.get_current_state(conv_id) == "a"
        record = _raw_data(api, conv_id)[CONTEXT_KEY_CLASSIFICATION_RESULT]
        assert record.get("low_confidence") is not True

    def test_a1_default_threshold_applies_without_config(self):
        below = DEFAULT_TRANSITION_CLASSIFICATION_CONFIDENCE - 0.1
        api, conv_id, _ = _api(_ambiguous_then_blocked_fsm(None))
        with patch("fsm_llm.pipeline.Classifier") as mock_cls:
            mock_cls.return_value.classify.return_value = _classification("a", below)
            api.converse("maybe a?", conv_id)

        assert api.get_current_state(conv_id) == "start"
        record = _raw_data(api, conv_id)[CONTEXT_KEY_CLASSIFICATION_RESULT]
        assert record["low_confidence"] is True

    def test_a8_stale_result_absent_after_deterministic_turn(self):
        api, conv_id, _ = _api(_ambiguous_then_blocked_fsm())
        with patch("fsm_llm.pipeline.Classifier") as mock_cls:
            mock_cls.return_value.classify.return_value = _classification("a", 0.95)
            api.converse("a please", conv_id)
        assert api.get_current_state(conv_id) == "a"
        assert CONTEXT_KEY_CLASSIFICATION_RESULT in _raw_data(api, conv_id)

        # Turn 2 in "a" is BLOCKED: no classifier runs, so no record may remain.
        with patch("fsm_llm.pipeline.Classifier") as mock_cls:
            api.converse("still here", conv_id)
            assert not mock_cls.return_value.classify.called

        assert api.get_current_state(conv_id) == "a"
        assert CONTEXT_KEY_CLASSIFICATION_RESULT not in _raw_data(api, conv_id)

    def test_a8_rollback_restores_prior_result(self):
        api, conv_id, llm = _api(_ambiguous_then_blocked_fsm())
        with patch("fsm_llm.pipeline.Classifier") as mock_cls:
            mock_cls.return_value.classify.return_value = _classification("a", 0.95)
            api.converse("a please", conv_id)
        prior = dict(_raw_data(api, conv_id)[CONTEXT_KEY_CLASSIFICATION_RESULT])

        seen_during_turn: list[bool] = []
        api.create_handler(
            "observe_record",
            HandlerTiming.POST_PROCESSING,
            lambda ctx: (
                seen_during_turn.append(CONTEXT_KEY_CLASSIFICATION_RESULT in ctx) or {}
            ),
        )
        llm.generate_response.side_effect = RuntimeError("pass 2 down")

        with pytest.raises(FSMError):
            api.converse("still here", conv_id)

        # Cleared during the turn, restored by the turn rollback.
        assert seen_during_turn == [False]
        assert _raw_data(api, conv_id)[CONTEXT_KEY_CLASSIFICATION_RESULT] == prior

    def test_a8_stream_path_clears_too(self):
        api, conv_id, _ = _api(_ambiguous_then_blocked_fsm())
        with patch("fsm_llm.pipeline.Classifier") as mock_cls:
            mock_cls.return_value.classify.return_value = _classification("a", 0.95)
            "".join(api.converse_stream("a please", conv_id))
        assert api.get_current_state(conv_id) == "a"
        assert CONTEXT_KEY_CLASSIFICATION_RESULT in _raw_data(api, conv_id)

        reply = "".join(api.converse_stream("still here", conv_id))

        assert reply == "ok"
        assert api.get_current_state(conv_id) == "a"
        assert CONTEXT_KEY_CLASSIFICATION_RESULT not in _raw_data(api, conv_id)
