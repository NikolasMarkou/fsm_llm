"""
Intent-to-handler routing for classified results.

Provides a registry that maps intent names to handler callables,
with built-in low-confidence fallback and missing-handler safety.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fsm_llm.logging import logger

from .definitions import (
    ClassificationError,
    ClassificationResult,
    ClassificationSchema,
    MultiClassificationResult,
)

HandlerFn = Callable[[str, dict[str, str]], Any]


class IntentRouter:
    """
    Maps classified intents to handler functions.

    Usage::

        router = IntentRouter(schema)
        router.register("order_status", handle_order_status)
        router.register("product_info", handle_product_info)

        result = classifier.classify(user_message)
        response = router.route(user_message, result)
    """

    def __init__(
        self,
        schema: ClassificationSchema,
        *,
        clarification_handler: HandlerFn | None = None,
    ) -> None:
        self.schema = schema
        self._handlers: dict[str, HandlerFn] = {}
        self._clarification_handler = clarification_handler or self._default_clarify

    # ----------------------------------------------------------
    # Registration
    # ----------------------------------------------------------

    def register(self, intent: str, handler: HandlerFn) -> IntentRouter:
        """
        Register a handler for an intent. Returns self for chaining.

        Raises ValueError if ``intent`` is not in the schema.
        """
        if intent not in self.schema.intent_names:
            raise ValueError(
                f"Unknown intent '{intent}'. "
                f"Valid intents: {self.schema.intent_names}"
            )
        self._handlers[intent] = handler
        return self

    def register_many(self, mapping: dict[str, HandlerFn]) -> IntentRouter:
        """Register multiple handlers at once."""
        for intent, handler in mapping.items():
            self.register(intent, handler)
        return self

    # ----------------------------------------------------------
    # Routing
    # ----------------------------------------------------------

    def route(
        self,
        user_message: str,
        result: ClassificationResult,
    ) -> Any:
        """
        Route a classified result to the appropriate handler.

        If confidence is below the schema threshold, the clarification
        handler is called instead. If no handler is registered for the
        intent, the fallback intent handler is used.
        """
        if result.confidence < self.schema.confidence_threshold:
            logger.info(
                f"Low confidence ({result.confidence:.2f} < "
                f"{self.schema.confidence_threshold}), requesting clarification"
            )
            return self._clarification_handler(user_message, result.entities)

        handler = self._handlers.get(result.intent)
        if handler is None:
            fallback = self._handlers.get(self.schema.fallback_intent)
            if fallback is None:
                raise ClassificationError(
                    f"No handler for intent '{result.intent}' and no fallback "
                    f"handler registered for '{self.schema.fallback_intent}'"
                )
            logger.warning(
                f"No handler for '{result.intent}', using fallback"
            )
            handler = fallback

        return handler(user_message, result.entities)

    def route_multi(
        self,
        user_message: str,
        result: MultiClassificationResult,
    ) -> list[Any]:
        """
        Route each intent in a multi-intent result.

        Returns a list of handler results, one per detected intent.
        Low-confidence intents are skipped.
        """
        outputs: list[Any] = []
        for scored in result.intents:
            as_single = ClassificationResult(
                reasoning=result.reasoning,
                intent=scored.intent,
                confidence=scored.confidence,
                entities=scored.entities,
            )
            if scored.confidence < self.schema.confidence_threshold:
                logger.debug(
                    f"Skipping low-confidence intent '{scored.intent}' "
                    f"({scored.confidence:.2f})"
                )
                continue
            outputs.append(self.route(user_message, as_single))
        return outputs

    # ----------------------------------------------------------
    # Defaults
    # ----------------------------------------------------------

    def validate(self) -> list[str]:
        """Check that all schema intents have registered handlers.

        Returns a list of intent names that lack handlers (empty if all covered).
        """
        missing = [
            name for name in self.schema.intent_names
            if name not in self._handlers
        ]
        if missing:
            logger.warning(f"Intents without handlers: {missing}")
        return missing

    @staticmethod
    def _default_clarify(user_message: str, entities: dict[str, str]) -> str:
        return (
            "I'm not sure I understand your request. "
            "Could you please rephrase or provide more details?"
        )
