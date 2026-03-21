"""Tests for intent routing."""

import pytest

from fsm_llm_classification import (
    ClassificationError,
    ClassificationResult,
    ClassificationSchema,
    IntentDefinition,
    IntentRouter,
    IntentScore,
    MultiClassificationResult,
)


def _schema():
    return ClassificationSchema(
        intents=[
            IntentDefinition(name="order_status", description="About orders"),
            IntentDefinition(name="general_support", description="Fallback"),
        ],
        fallback_intent="general_support",
        confidence_threshold=0.6,
    )


def _result(intent="order_status", confidence=0.9):
    return ClassificationResult(
        reasoning="test",
        intent=intent,
        confidence=confidence,
        entities={},
    )


class TestIntentRouter:
    def test_routes_to_correct_handler(self):
        router = IntentRouter(_schema())
        router.register("order_status", lambda msg, ent: "order_handler")
        router.register("general_support", lambda msg, ent: "fallback_handler")

        assert router.route("test", _result()) == "order_handler"

    def test_low_confidence_triggers_clarification(self):
        router = IntentRouter(_schema())
        router.register("order_status", lambda msg, ent: "order_handler")
        router.register("general_support", lambda msg, ent: "fallback_handler")

        resp = router.route("test", _result(confidence=0.3))
        assert "not sure" in resp.lower()

    def test_custom_clarification_handler(self):
        router = IntentRouter(
            _schema(),
            clarification_handler=lambda msg, ent: "custom_clarify",
        )
        router.register("general_support", lambda msg, ent: "fb")

        assert router.route("test", _result(confidence=0.3)) == "custom_clarify"

    def test_unknown_handler_falls_to_fallback(self):
        router = IntentRouter(_schema())
        router.register("general_support", lambda msg, ent: "fallback_handler")

        assert router.route("test", _result()) == "fallback_handler"

    def test_no_fallback_raises(self):
        router = IntentRouter(_schema())

        with pytest.raises(ClassificationError, match="No handler"):
            router.route("test", _result())

    def test_register_unknown_intent_raises(self):
        router = IntentRouter(_schema())

        with pytest.raises(ValueError, match="Unknown intent"):
            router.register("nonexistent", lambda msg, ent: None)

    def test_register_many(self):
        router = IntentRouter(_schema())
        router.register_many({
            "order_status": lambda msg, ent: "a",
            "general_support": lambda msg, ent: "b",
        })

        assert router.route("test", _result()) == "a"

    def test_chaining(self):
        router = IntentRouter(_schema())
        r = router.register("order_status", lambda msg, ent: "a")
        assert r is router

    def test_route_multi(self):
        router = IntentRouter(_schema())
        router.register("order_status", lambda msg, ent: "order")
        router.register("general_support", lambda msg, ent: "general")

        multi = MultiClassificationResult(
            reasoning="test",
            intents=[
                IntentScore(intent="order_status", confidence=0.9),
                IntentScore(intent="general_support", confidence=0.4),
            ],
        )
        results = router.route_multi("test", multi)
        # Only the high-confidence intent should be routed
        assert results == ["order"]
