"""
Regression tests for S5: `IntentScore` must preserve `None` entity values.

The defect: `IntentScore.coerce_entity_values` coerced a `None` entity value into
the literal string `"None"` via a bare `str(val)`, while its sibling
`ClassificationResult.coerce_entity_values` explicitly preserved `None`.

Why this matters at the seam rather than at the model: `IntentScore` backs
`MultiClassificationResult.intents`, and `IntentRouter.route_multi` passes
`scored.entities` straight into user handler functions. A handler written as
`if entities.get("foo"):` sees the TRUTHY string `"None"` instead of a falsy
absent value, so it takes the wrong branch on data the LLM explicitly reported
as absent. A direct model-construction assertion alone would not prove that
handler contract, so the primary test below drives a real
`MultiClassificationResult` through `IntentRouter.route_multi`.

See decisions.md D-010 (and D-004 for the plan-time framing).
"""

from __future__ import annotations

from typing import Any

import pytest

from fsm_llm.classification import IntentRouter
from fsm_llm.definitions import (
    ClassificationError,
    ClassificationResult,
    ClassificationSchema,
    IntentDefinition,
    IntentScore,
    MultiClassificationResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _schema() -> ClassificationSchema:
    """Minimal two-intent schema (pydantic enforces min_length=2 on intents)."""
    return ClassificationSchema(
        intents=[
            IntentDefinition(name="order_status", description="Check an order"),
            IntentDefinition(name="unclear", description="Fallback"),
        ],
        fallback_intent="unclear",
        confidence_threshold=0.5,
    )


# ---------------------------------------------------------------------------
# PRIMARY: the integration seam (IntentRouter.route_multi -> handler)
# ---------------------------------------------------------------------------


class TestNoneEntityThroughRouteMulti:
    """The seam the defect actually lives on."""

    def test_handler_sees_falsy_branch_for_none_entity(self) -> None:
        """A handler's `if entities.get(k):` must take the FALSY branch.

        This is the assertion that fails on the unfixed code: `"None"` is a
        non-empty string and therefore truthy.
        """
        observed: dict[str, Any] = {}

        def handle_order_status(user_message: str, entities: Any) -> str:
            observed["raw"] = entities.get("order_id")
            if entities.get("order_id"):
                return "TRUTHY"
            return "FALSY"

        router = IntentRouter(_schema())
        router.register("order_status", handle_order_status)

        result = MultiClassificationResult(
            reasoning="user asked about an order but gave no id",
            intents=[
                IntentScore(
                    intent="order_status",
                    confidence=0.9,
                    entities={"order_id": None},
                )
            ],
        )

        outputs = router.route_multi("where is my order?", result)

        assert outputs == ["FALSY"], (
            "handler took the truthy branch on an absent entity; "
            f"it observed {observed['raw']!r}"
        )
        assert observed["raw"] is None
        assert observed["raw"] != "None"

    def test_route_multi_preserves_none_alongside_real_values(self) -> None:
        """Preserving `None` must not disturb real string/list coercion."""
        seen: dict[str, Any] = {}

        def handler(user_message: str, entities: Any) -> None:
            seen.update(entities)

        router = IntentRouter(_schema())
        router.register("order_status", handler)

        result = MultiClassificationResult(
            reasoning="mixed entity payload",
            intents=[
                IntentScore(
                    intent="order_status",
                    confidence=0.9,
                    entities={
                        "order_id": None,
                        "customer": "ada",
                        "items": ["a", "b"],
                        "count": 3,
                    },
                )
            ],
        )
        router.route_multi("hi", result)

        assert seen == {
            "order_id": None,
            "customer": "ada",
            "items": "a, b",
            "count": "3",
        }


# ---------------------------------------------------------------------------
# The two seams route_multi does NOT cover: route() and the clarification path
# ---------------------------------------------------------------------------


class TestNoneEntityThroughRoute:
    """Pins the widened `HandlerFn` contract at its two uncovered call sites.

    `route_multi` (above) covers `classification.py:516`. The remaining two
    places a `dict[str, str | None]` reaches user code are `route()`'s normal
    dispatch (`:483`) and its low-confidence clarification branch (`:470`).
    Handlers here are annotated `dict[str, str | None]` on purpose: that is the
    contract `HandlerFn` now declares (see decisions.md D-001), and these tests
    are what would fail if a future edit narrowed it back.
    """

    def test_route_handler_receives_none_not_the_string_none(self) -> None:
        """`route()` must hand the handler a real `None`, not `"None"`."""
        observed: dict[str, Any] = {}

        def handle(user_message: str, entities: dict[str, str | None]) -> str:
            observed["entities"] = entities
            return "TRUTHY" if entities.get("order_id") else "FALSY"

        router = IntentRouter(_schema())
        router.register("order_status", handle)
        result = ClassificationResult(
            reasoning="user asked about an order but gave no id",
            intent="order_status",
            confidence=0.9,
            entities={"order_id": None},
        )

        assert router.route("where is my order?", result) == "FALSY", (
            "handler took the truthy branch on an absent entity; "
            f"it observed {observed['entities']['order_id']!r}"
        )
        assert observed["entities"]["order_id"] is None
        assert observed["entities"]["order_id"] != "None"

    def test_clarification_handler_receives_none_and_its_return_is_routed(
        self,
    ) -> None:
        """The low-confidence branch must pass `None` through AND return its value.

        Two properties in one drive because they share a single call: the
        clarification handler sees the real `None`, and `route()` returns what
        that handler returned (not the default clarification string, and not the
        normal handler's value).
        """
        observed: dict[str, Any] = {}

        def clarify(user_message: str, entities: dict[str, str | None]) -> str:
            observed["entities"] = entities
            return "CLARIFY-SENTINEL"

        def never_called(user_message: str, entities: dict[str, str | None]) -> str:
            raise AssertionError("normal handler must not run below threshold")

        schema = _schema()  # confidence_threshold=0.5
        router = IntentRouter(schema, clarification_handler=clarify)
        router.register("order_status", never_called)
        result = ClassificationResult(
            reasoning="ambiguous, and no order id was given",
            intent="order_status",
            confidence=0.1,
            entities={"order_id": None},
        )
        assert result.confidence < schema.confidence_threshold, (
            "test setup no longer exercises the clarification branch"
        )

        assert router.route("uh, my thing?", result) == "CLARIFY-SENTINEL"
        assert observed["entities"]["order_id"] is None
        assert observed["entities"]["order_id"] != "None"
        assert not observed["entities"]["order_id"]


# ---------------------------------------------------------------------------
# validate() and shared handler resolution (route / route_multi)
# ---------------------------------------------------------------------------


def _three_intent_schema() -> ClassificationSchema:
    """Three intents so validate() ordering and partial coverage are visible."""
    return ClassificationSchema(
        intents=[
            IntentDefinition(name="order_status", description="Check an order"),
            IntentDefinition(name="product_info", description="Product details"),
            IntentDefinition(name="unclear", description="Fallback"),
        ],
        fallback_intent="unclear",
        confidence_threshold=0.5,
    )


def _noop(user_message: str, entities: dict[str, str | None]) -> str:
    return "noop"


class TestIntentRouterValidateAndDispatch:
    """`validate()` reports missing handlers once each, in schema order, and
    `route` / `route_multi` share one fallback-or-raise resolution path."""

    def test_validate_all_covered_returns_empty(self) -> None:
        router = IntentRouter(_three_intent_schema())
        router.register_many(
            {"order_status": _noop, "product_info": _noop, "unclear": _noop}
        )
        assert router.validate() == []

    def test_validate_reports_missing_in_schema_order(self) -> None:
        router = IntentRouter(_three_intent_schema())
        router.register("product_info", _noop)
        assert router.validate() == ["order_status", "unclear"]

    def test_validate_lists_missing_fallback_exactly_once(self) -> None:
        schema = _three_intent_schema()
        router = IntentRouter(schema)
        router.register("order_status", _noop)
        router.register("product_info", _noop)
        missing = router.validate()
        assert missing == [schema.fallback_intent]
        assert missing.count(schema.fallback_intent) == 1

    def test_route_falls_back_when_intent_handler_missing(self) -> None:
        calls: list[tuple[str, dict[str, str | None]]] = []

        def fallback(user_message: str, entities: dict[str, str | None]) -> str:
            calls.append((user_message, entities))
            return "FALLBACK"

        router = IntentRouter(_three_intent_schema())
        router.register("unclear", fallback)
        result = ClassificationResult(
            reasoning="asked about a product",
            intent="product_info",
            confidence=0.9,
            entities={"product": "widget"},
        )
        assert router.route("tell me about widget", result) == "FALLBACK"
        assert calls == [("tell me about widget", {"product": "widget"})]

    def test_route_multi_falls_back_when_intent_handler_missing(self) -> None:
        calls: list[tuple[str, dict[str, str | None]]] = []

        def fallback(user_message: str, entities: dict[str, str | None]) -> str:
            calls.append((user_message, entities))
            return "FALLBACK"

        router = IntentRouter(_three_intent_schema())
        router.register("unclear", fallback)
        result = MultiClassificationResult(
            reasoning="asked about a product",
            intents=[
                IntentScore(
                    intent="product_info",
                    confidence=0.9,
                    entities={"product": "widget"},
                )
            ],
        )
        assert router.route_multi("tell me about widget", result) == ["FALLBACK"]
        assert calls == [("tell me about widget", {"product": "widget"})]

    def test_route_and_route_multi_raise_same_message_without_fallback(
        self,
    ) -> None:
        router = IntentRouter(_three_intent_schema())
        router.register("order_status", _noop)

        single = ClassificationResult(
            reasoning="asked about a product",
            intent="product_info",
            confidence=0.9,
            entities={},
        )
        multi = MultiClassificationResult(
            reasoning="asked about a product",
            intents=[IntentScore(intent="product_info", confidence=0.9, entities={})],
        )

        with pytest.raises(ClassificationError) as single_exc:
            router.route("tell me about widget", single)
        with pytest.raises(ClassificationError) as multi_exc:
            router.route_multi("tell me about widget", multi)

        assert str(single_exc.value) == str(multi_exc.value)
        assert "No handler for intent 'product_info'" in str(single_exc.value)
        assert "no fallback handler registered for 'unclear'" in str(single_exc.value)


# ---------------------------------------------------------------------------
# Direct model assertion (SC-7 first half)
# ---------------------------------------------------------------------------


class TestIntentScoreNonePreservation:
    def test_none_entity_value_is_preserved(self) -> None:
        score = IntentScore(intent="a", confidence=0.9, entities={"foo": None})
        assert score.entities == {"foo": None}

    def test_none_entity_value_is_not_the_string_none(self) -> None:
        """Explicit anti-regression pin on the exact corrupted value."""
        score = IntentScore(intent="a", confidence=0.9, entities={"foo": None})
        assert score.entities["foo"] is None

    def test_non_dict_entities_still_coerce_to_empty(self) -> None:
        """Pre-existing tolerance for garbage LLM output is unchanged."""
        assert IntentScore(intent="a", confidence=0.5, entities="junk").entities == {}
        assert IntentScore(intent="a", confidence=0.5, entities=None).entities == {}


# ---------------------------------------------------------------------------
# ANTI-DIVERGENCE PIN: the two sibling coercers must agree
# ---------------------------------------------------------------------------


class TestSiblingCoercersAgree:
    """Fails loudly if a future edit re-splits the two coercers.

    `IntentScore.coerce_entity_values` and
    `ClassificationResult.coerce_entity_values` are intentionally textually
    identical modulo the return annotation. This is the DRY convergence S5
    established; the original defect was precisely their divergence.
    """

    @pytest.mark.parametrize(
        "payload",
        [
            {"foo": None},
            {"foo": "bar"},
            {"foo": ["a", "b"]},
            {"foo": 3},
            {"foo": False},
            {"foo": None, "bar": "x", "baz": ["y", "z"]},
            {},
            "not-a-dict",
            None,
            123,
        ],
    )
    def test_identical_output_for_same_input(self, payload: Any) -> None:
        score = IntentScore(intent="a", confidence=0.9, entities=payload)
        single = ClassificationResult(
            reasoning="r", intent="a", confidence=0.9, entities=payload
        )
        assert score.entities == single.entities, (
            "IntentScore and ClassificationResult entity coercion have DIVERGED; "
            "they must remain textually identical (see decisions.md D-010)"
        )
