"""Tests for classification schema definitions and result models."""

import pytest
from pydantic import ValidationError

from fsm_llm import (
    ClassificationResult,
    ClassificationSchema,
    HierarchicalSchema,
    IntentDefinition,
    IntentScore,
    MultiClassificationResult,
)

# --------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------


def make_schema(**overrides):
    defaults = dict(
        intents=[
            IntentDefinition(name="order_status", description="About orders"),
            IntentDefinition(name="product_info", description="About products"),
            IntentDefinition(name="general_support", description="Fallback"),
        ],
        fallback_intent="general_support",
    )
    defaults.update(overrides)
    return ClassificationSchema(**defaults)


# --------------------------------------------------------------
# IntentDefinition
# --------------------------------------------------------------


class TestIntentDefinition:
    def test_valid_name(self):
        intent = IntentDefinition(name="order_status", description="d")
        assert intent.name == "order_status"

    def test_invalid_name_rejects_special_chars(self):
        with pytest.raises(ValidationError):
            IntentDefinition(name="order-status!", description="d")


# --------------------------------------------------------------
# ClassificationSchema
# --------------------------------------------------------------


class TestClassificationSchema:
    def test_valid_schema(self):
        schema = make_schema()
        assert len(schema.intents) == 3
        assert schema.intent_names == [
            "order_status",
            "product_info",
            "general_support",
        ]

    def test_fallback_must_be_in_intents(self):
        with pytest.raises(ValidationError, match="Fallback intent"):
            make_schema(fallback_intent="nonexistent")

    def test_duplicate_names_rejected(self):
        with pytest.raises(ValidationError, match="unique"):
            make_schema(
                intents=[
                    IntentDefinition(name="a", description="x"),
                    IntentDefinition(name="a", description="y"),
                ],
                fallback_intent="a",
            )

    def test_minimum_two_intents(self):
        with pytest.raises(ValidationError):
            make_schema(
                intents=[
                    IntentDefinition(name="a", description="x"),
                ],
                fallback_intent="a",
            )


# --------------------------------------------------------------
# ClassificationResult
# --------------------------------------------------------------


class TestClassificationResult:
    def test_basic_result(self):
        r = ClassificationResult(
            reasoning="test",
            intent="order_status",
            confidence=0.95,
            entities={"order_id": "123"},
        )
        assert r.intent == "order_status"
        assert not r.is_low_confidence

    def test_low_confidence(self):
        r = ClassificationResult(
            reasoning="",
            intent="x",
            confidence=0.3,
        )
        assert r.is_low_confidence

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            ClassificationResult(
                reasoning="",
                intent="x",
                confidence=1.5,
            )


# --------------------------------------------------------------
# MultiClassificationResult
# --------------------------------------------------------------


class TestMultiClassificationResult:
    def test_primary(self):
        r = MultiClassificationResult(
            reasoning="test",
            intents=[
                IntentScore(intent="a", confidence=0.9),
                IntentScore(intent="b", confidence=0.5),
            ],
        )
        assert r.primary.intent == "a"

    def test_at_least_one_intent(self):
        with pytest.raises(ValidationError):
            MultiClassificationResult(reasoning="test", intents=[])


# --------------------------------------------------------------
# HierarchicalSchema
# --------------------------------------------------------------


class TestHierarchicalSchema:
    def test_missing_domain_schema_rejected(self):
        domain = make_schema(
            intents=[
                IntentDefinition(name="billing", description="b"),
                IntentDefinition(name="shipping", description="s"),
                IntentDefinition(name="other", description="o"),
            ],
            fallback_intent="other",
        )
        # Missing "shipping" in intent_schemas
        with pytest.raises(ValidationError, match="Missing intent schemas"):
            HierarchicalSchema(
                domain_schema=domain,
                intent_schemas={
                    "billing": make_schema(),
                },
            )

    def test_valid_hierarchical(self):
        domain = make_schema(
            intents=[
                IntentDefinition(name="billing", description="b"),
                IntentDefinition(name="other", description="o"),
            ],
            fallback_intent="other",
        )
        h = HierarchicalSchema(
            domain_schema=domain,
            intent_schemas={"billing": make_schema()},
        )
        assert "billing" in h.intent_schemas
