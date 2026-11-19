"""IntentRouter: maps classified intents to handler functions.

Layer ``engine`` (plan 51fde95b, step 6.2): moved out of ``engine/classification.py``, a pure
move that keeps the class body unchanged and the module under its size line. ``HandlerFn``
stays defined in ``engine/classification.py``.
"""

from __future__ import annotations

from typing import Any

from fsm_llm.core.definitions import (
    ClassificationError,
    ClassificationResult,
    ClassificationSchema,
    MultiClassificationResult,
)
from fsm_llm.logging import logger

from .classification import HandlerFn


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
                f"Unknown intent '{intent}'. Valid intents: {self.schema.intent_names}"
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
        handler is called instead.
        """
        if result.confidence < self.schema.confidence_threshold:
            logger.info(
                f"Low confidence ({result.confidence:.2f} < "
                f"{self.schema.confidence_threshold}), requesting clarification"
            )
            return self._clarification_handler(user_message, result.entities)

        handler = self._resolve_handler(result.intent)
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
        skipped = 0
        for scored in result.intents:
            if scored.confidence < self.schema.confidence_threshold:
                logger.debug(
                    f"Skipping low-confidence intent '{scored.intent}' "
                    f"({scored.confidence:.2f})"
                )
                skipped += 1
                continue

            handler = self._resolve_handler(scored.intent)
            outputs.append(handler(user_message, scored.entities))

        if not outputs and skipped > 0:
            logger.warning(
                f"All {skipped} intents were below confidence threshold "
                f"({self.schema.confidence_threshold}); no handlers invoked"
            )
        return outputs

    def _resolve_handler(self, intent: str) -> HandlerFn:
        """Return the handler to invoke for ``intent``.

        Contract: returns the handler registered for ``intent``; if none,
        returns the handler registered for ``schema.fallback_intent`` and logs
        a WARNING; if neither is registered, raises ``ClassificationError``.
        Shared by ``route`` and ``route_multi`` so both paths fall back and
        fail identically.
        """
        handler = self._handlers.get(intent)
        if handler is not None:
            return handler
        fallback = self._handlers.get(self.schema.fallback_intent)
        if fallback is None:
            raise ClassificationError(
                f"No handler for intent '{intent}' and no fallback "
                f"handler registered for '{self.schema.fallback_intent}'"
            )
        logger.warning(f"No handler for '{intent}', using fallback")
        return fallback

    # ----------------------------------------------------------
    # Validation & Defaults
    # ----------------------------------------------------------

    def validate(self) -> list[str]:
        """Check that all schema intents (including fallback) have registered handlers.

        Returns a list of intent names that lack handlers (empty if all covered).
        """
        # ClassificationSchema.validate_schema guarantees fallback_intent is in
        # intent_names, so the comprehension already covers the fallback.
        missing = [
            name for name in self.schema.intent_names if name not in self._handlers
        ]
        if missing:
            logger.warning(f"Intents without handlers: {missing}")
        return missing

    @staticmethod
    def _default_clarify(user_message: str, entities: dict[str, str | None]) -> str:
        return (
            "I'm not sure I understand your request. "
            "Could you please rephrase or provide more details?"
        )
