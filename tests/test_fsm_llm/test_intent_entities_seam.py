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
