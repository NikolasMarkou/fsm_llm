"""
Tests verifying fixes for audit findings in fsm_llm_classification.
Covers: F-006 (confidence clamping), F-008 (is_low_confidence threshold),
        F-010 (dead code removal).
"""
from __future__ import annotations

from fsm_llm_classification.definitions import (
    ClassificationResult,
    ClassificationSchema,
    IntentDefinition,
)


def _make_schema(**overrides):
    defaults = dict(
        intents=[
            IntentDefinition(name="greet", description="Greetings"),
            IntentDefinition(name="bye", description="Farewell"),
        ],
        fallback_intent="greet",
    )
    defaults.update(overrides)
    return ClassificationSchema(**defaults)


# ---------------------------------------------------------------------------
# F-006: Confidence clamping
# ---------------------------------------------------------------------------

class TestConfidenceClamping:
    """F-006: Out-of-range confidence values must be clamped to [0, 1]."""

    def test_classifier_clamps_high_confidence(self):
        """Confidence > 1.0 from LLM should be clamped to 1.0."""
        from fsm_llm_classification.classifier import Classifier

        schema = _make_schema()
        clf = Classifier(schema=schema, model="test")

        data = {"reasoning": "test", "intent": "greet", "confidence": 1.5}
        result = clf._parse_single(data)
        assert result.confidence == 1.0

    def test_classifier_clamps_negative_confidence(self):
        """Confidence < 0.0 from LLM should be clamped to 0.0."""
        from fsm_llm_classification.classifier import Classifier

        schema = _make_schema()
        clf = Classifier(schema=schema, model="test")

        data = {"reasoning": "test", "intent": "greet", "confidence": -0.3}
        result = clf._parse_single(data)
        assert result.confidence == 0.0

    def test_classifier_preserves_valid_confidence(self):
        """Valid confidence should pass through unchanged."""
        from fsm_llm_classification.classifier import Classifier

        schema = _make_schema()
        clf = Classifier(schema=schema, model="test")

        data = {"reasoning": "test", "intent": "greet", "confidence": 0.85}
        result = clf._parse_single(data)
        assert result.confidence == 0.85

    def test_multi_intent_clamps_confidence(self):
        """Multi-intent parsing should also clamp confidence."""
        from fsm_llm_classification.classifier import Classifier

        schema = _make_schema()
        clf = Classifier(schema=schema, model="test")

        data = {
            "reasoning": "test",
            "intents": [
                {"intent": "greet", "confidence": 2.0, "entities": {}},
                {"intent": "bye", "confidence": -1.0, "entities": {}},
            ],
        }
        result = clf._parse_multi(data)
        assert result.intents[0].confidence == 1.0
        assert result.intents[1].confidence == 0.0


# ---------------------------------------------------------------------------
# F-008: is_low_confidence uses configurable threshold
# ---------------------------------------------------------------------------

class TestIsLowConfidenceThreshold:
    """F-008: is_low_confidence should use class-level constant, not hardcoded 0.6."""

    def test_default_threshold_is_06(self):
        """Default threshold should be 0.6 (backwards compatible)."""
        result = ClassificationResult(
            reasoning="", intent="greet", confidence=0.59
        )
        assert result.is_low_confidence is True

        result2 = ClassificationResult(
            reasoning="", intent="greet", confidence=0.61
        )
        assert result2.is_low_confidence is False

    def test_threshold_is_configurable_via_class_attribute(self):
        """DEFAULT_CONFIDENCE_THRESHOLD should be a class attribute."""
        assert hasattr(ClassificationResult, "DEFAULT_CONFIDENCE_THRESHOLD")
        assert ClassificationResult.DEFAULT_CONFIDENCE_THRESHOLD == 0.6


# ---------------------------------------------------------------------------
# F-010: Dead code removed
# ---------------------------------------------------------------------------

class TestDeadCodeRemoved:
    """F-010: intent_enum and get_intent should be removed from ClassificationSchema."""

    def test_no_intent_enum(self):
        """ClassificationSchema should not have intent_enum property."""
        schema = _make_schema()
        assert not hasattr(type(schema), "intent_enum") or not callable(getattr(type(schema), "intent_enum", None))

    def test_no_get_intent(self):
        """ClassificationSchema should not have get_intent method."""
        schema = _make_schema()
        assert not hasattr(schema, "get_intent")

    def test_intent_names_still_works(self):
        """intent_names property should still work after dead code removal."""
        schema = _make_schema()
        assert schema.intent_names == ["greet", "bye"]
