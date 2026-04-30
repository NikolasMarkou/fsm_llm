"""
Core classification runtime: Classifier, HierarchicalClassifier, and IntentRouter.

Provides LLM-backed intent classification for both standalone use and
as the transition resolution mechanism within the FSM pipeline.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from litellm import completion, get_supported_openai_params

from ..constants import DEFAULT_LLM_MODEL
from ..logging import logger
from ..ollama import apply_ollama_params, prepare_ollama_messages
from .._models import (
    ClassificationError,
    ClassificationResponseError,
)
from ..utilities import extract_json_from_text
from .definitions import (
    ClassificationResult,
    ClassificationSchema,
    HierarchicalResult,
    HierarchicalSchema,
    IntentScore,
    MultiClassificationResult,
)
from .prompts import (
    ClassificationPromptConfig,
    build_classification_json_schema,
    build_classification_system_prompt,
)

# Type alias for intent handler functions
HandlerFn = Callable[[str, dict[str, str]], Any]


# --------------------------------------------------------------
# Classifier
# --------------------------------------------------------------


class Classifier:
    """
    LLM-backed intent classifier.

    Wraps a ClassificationSchema and an LLM model to provide a simple
    ``classify()`` / ``classify_multi()`` interface.  Follows the same
    LiteLLM integration pattern used by ``fsm_llm.LiteLLMInterface``.
    """

    def __init__(
        self,
        schema: ClassificationSchema,
        model: str = DEFAULT_LLM_MODEL,
        *,
        api_key: str | None = None,
        config: ClassificationPromptConfig | None = None,
        **litellm_kwargs,
    ) -> None:
        if not model or not model.strip():
            raise ValueError("model must be a non-empty string")

        self.schema = schema
        self.model = model
        self.config = config or ClassificationPromptConfig()
        self._kwargs: dict = {**litellm_kwargs}
        if api_key:
            self._kwargs["api_key"] = api_key

        # Pre-build prompts so they're not reconstructed on every call.
        single_config = replace(self.config, multi_intent=False)
        multi_config = replace(self.config, multi_intent=True)

        self._system_prompt = build_classification_system_prompt(schema, single_config)
        self._multi_system_prompt = build_classification_system_prompt(
            schema, multi_config
        )

        self._json_schema = build_classification_json_schema(
            schema,
            multi_intent=False,
            include_reasoning=self.config.include_reasoning,
            include_entities=self.config.include_entities,
        )
        self._multi_json_schema = build_classification_json_schema(
            schema,
            multi_intent=True,
            max_intents=self.config.max_intents,
            include_reasoning=self.config.include_reasoning,
            include_entities=self.config.include_entities,
        )

        logger.bind(package="fsm_llm.classification").info(
            f"Classifier initialized: model={model}, "
            f"intents={len(schema.intents)}, "
            f"threshold={schema.confidence_threshold}"
        )

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def classify(self, user_message: str) -> ClassificationResult:
        """
        Classify a single user message into one intent.

        Returns:
            ClassificationResult with intent, confidence, reasoning, and entities.

        Raises:
            ClassificationResponseError: If the LLM response cannot be parsed.
        """
        raw = self._call_llm(user_message, multi_intent=False)
        return self._parse_single(raw)

    def classify_multi(self, user_message: str) -> MultiClassificationResult:
        """
        Classify a message that may contain multiple intents.

        Returns:
            MultiClassificationResult with a ranked list of IntentScores.
        """
        raw = self._call_llm(user_message, multi_intent=True)
        return self._parse_multi(raw)

    def is_low_confidence(self, result: ClassificationResult) -> bool:
        """Check if a result falls below the schema's confidence threshold."""
        return result.confidence < self.schema.confidence_threshold

    # ----------------------------------------------------------
    # LLM Communication
    # ----------------------------------------------------------

    def _call_llm(self, user_message: str, *, multi_intent: bool) -> dict:
        """Make the LLM call and return the parsed JSON dict."""
        start = time.time()

        system_prompt = (
            self._multi_system_prompt if multi_intent else self._system_prompt
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        reserved = {"model", "messages", "temperature", "max_tokens"}
        safe_kwargs = {k: v for k, v in self._kwargs.items() if k not in reserved}
        call_params = {
            **safe_kwargs,
            "model": self.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        # Use structured output when the provider supports it
        supported = get_supported_openai_params(model=self.model)
        if supported and "response_format" in supported:
            target_schema = (
                self._multi_json_schema if multi_intent else self._json_schema
            )
            call_params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "intent_classification",
                    "schema": target_schema,
                },
            }

        # Ollama: disable thinking mode and force temperature=0
        apply_ollama_params(call_params, self.model, structured=True)

        # Ollama: prepend /nothink and embed schema in prompt
        messages = prepare_ollama_messages(
            messages, self.model, call_params.get("response_format")
        )
        call_params["messages"] = messages

        response = completion(**call_params)
        elapsed = time.time() - start

        if not response or not getattr(response, "choices", None):
            raise ClassificationResponseError("Empty response from LLM")

        content = response.choices[0].message.content
        logger.debug(f"Classification call completed in {elapsed:.2f}s")

        return self._extract_response(content, response)

    # ----------------------------------------------------------
    # Response Extraction
    # ----------------------------------------------------------

    @staticmethod
    def _extract_response(content: Any, response: Any) -> dict:
        """
        Extract a JSON dict from the LLM response content.

        Handles three scenarios:
        1. Normal response -- content is a string or dict with JSON.
        2. Thinking model -- content is empty but the thinking field
           contains the answer (e.g., some Ollama models with qwen3).
        3. Structured output -- content is already a dict.
        """
        # Already a dict (some providers return parsed JSON directly)
        if isinstance(content, dict):
            return content

        # Normal case -- content is a non-empty string
        if content:
            data = extract_json_from_text(content)
            if data is not None:
                return data
            raise ClassificationResponseError(
                f"Failed to parse LLM JSON.\nResponse: {content[:200]}"
            )

        # Thinking model fallback -- content is empty/None, check thinking field
        msg = response.choices[0].message
        thinking = getattr(msg, "thinking", None)
        if thinking:
            logger.warning(
                "LLM returned empty content with non-empty thinking field; "
                "extracting classification from thinking content"
            )
            data = extract_json_from_text(thinking)
            if data is not None:
                return data

        raise ClassificationResponseError("LLM returned empty content")

    # ----------------------------------------------------------
    # Response Parsing
    # ----------------------------------------------------------

    def _parse_single(self, data: dict) -> ClassificationResult:
        intent = data.get("intent", "")
        if intent not in self.schema.intent_names:
            logger.warning(
                f"LLM returned unknown intent '{intent}', "
                f"falling back to '{self.schema.fallback_intent}'"
            )
            intent = self.schema.fallback_intent

        raw_confidence = data.get("confidence", 0.0)
        try:
            confidence = float(raw_confidence)
        except (ValueError, TypeError):
            logger.warning(
                f"Invalid confidence value {raw_confidence!r}, defaulting to 0.0"
            )
            confidence = 0.0
        return ClassificationResult(
            reasoning=data.get("reasoning", ""),
            intent=intent,
            confidence=max(0.0, min(1.0, confidence)),
            entities=data.get("entities", {})
            if isinstance(data.get("entities"), dict)
            else {},
        )

    def _parse_multi(self, data: dict) -> MultiClassificationResult:
        raw_intents = data.get("intents", [])
        if not raw_intents:
            raise ClassificationResponseError(
                "Multi-intent response contained no intents"
            )

        valid_names = set(self.schema.intent_names)
        scored: list[IntentScore] = []
        for item in raw_intents:
            if not isinstance(item, dict):
                logger.warning(
                    f"Skipping non-dict item in multi-intent response: {item!r}"
                )
                continue
            name = item.get("intent", "")
            if name not in valid_names:
                logger.warning(
                    f"LLM returned unknown intent '{name}' in multi-intent response, "
                    f"falling back to '{self.schema.fallback_intent}'"
                )
                name = self.schema.fallback_intent
            raw_confidence = item.get("confidence", 0.0)
            try:
                confidence = float(raw_confidence)
            except (ValueError, TypeError):
                logger.warning(
                    f"Invalid confidence value {raw_confidence!r}, defaulting to 0.0"
                )
                confidence = 0.0
            scored.append(
                IntentScore(
                    intent=name,
                    confidence=max(0.0, min(1.0, confidence)),
                    entities=item.get("entities", {})
                    if isinstance(item.get("entities"), dict)
                    else {},
                )
            )

        # Deduplicate intents (fallback remapping can create duplicates),
        # keeping the highest-confidence entry for each intent name,
        # then re-sort by confidence descending.
        seen: dict[str, int] = {}
        for i, s in enumerate(scored):
            if s.intent not in seen or s.confidence > scored[seen[s.intent]].confidence:
                seen[s.intent] = i
        scored = sorted(
            [scored[i] for i in seen.values()],
            key=lambda s: s.confidence,
            reverse=True,
        )

        if not scored:
            raise ClassificationResponseError(
                "Multi-intent response contained no valid intents after filtering"
            )

        return MultiClassificationResult(
            reasoning=data.get("reasoning", ""),
            intents=scored,
        )


# --------------------------------------------------------------
# Hierarchical Classifier
# --------------------------------------------------------------


class HierarchicalClassifier:
    """
    Two-stage classifier for large intent sets (>15 classes).

    Stage 1 classifies the domain, stage 2 classifies the intent within
    that domain using a domain-specific schema.
    """

    def __init__(
        self,
        schema: HierarchicalSchema,
        model: str = DEFAULT_LLM_MODEL,
        *,
        api_key: str | None = None,
        config: ClassificationPromptConfig | None = None,
        **litellm_kwargs,
    ) -> None:
        self.schema = schema
        shared = dict(model=model, api_key=api_key, config=config, **litellm_kwargs)

        self._domain_classifier = Classifier(schema=schema.domain_schema, **shared)
        self._intent_classifiers: dict[str, Classifier] = {
            domain: Classifier(schema=intent_schema, **shared)
            for domain, intent_schema in schema.intent_schemas.items()
        }

    def classify(self, user_message: str) -> HierarchicalResult:
        """
        Run two-stage classification: domain then intent.

        If the domain result maps to the fallback and no sub-classifier exists,
        the intent result mirrors the domain result.
        """
        domain_result = self._domain_classifier.classify(user_message)

        sub = self._intent_classifiers.get(domain_result.intent)
        if sub is None:
            logger.warning(
                f"No sub-classifier for domain '{domain_result.intent}', "
                f"mirroring domain result as intent"
            )
            return HierarchicalResult(
                domain=domain_result,
                intent=domain_result,
            )

        intent_result = sub.classify(user_message)
        return HierarchicalResult(
            domain=domain_result,
            intent=intent_result,
        )


# --------------------------------------------------------------
# Intent Router
# --------------------------------------------------------------


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

        handler = self._handlers.get(result.intent)
        if handler is None:
            fallback = self._handlers.get(self.schema.fallback_intent)
            if fallback is None:
                raise ClassificationError(
                    f"No handler for intent '{result.intent}' and no fallback "
                    f"handler registered for '{self.schema.fallback_intent}'"
                )
            logger.warning(f"No handler for '{result.intent}', using fallback")
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
        skipped = 0
        for scored in result.intents:
            if scored.confidence < self.schema.confidence_threshold:
                logger.debug(
                    f"Skipping low-confidence intent '{scored.intent}' "
                    f"({scored.confidence:.2f})"
                )
                skipped += 1
                continue

            handler = self._handlers.get(scored.intent)
            if handler is None:
                handler = self._handlers.get(self.schema.fallback_intent)
                if handler is None:
                    raise ClassificationError(
                        f"No handler for intent '{scored.intent}' and no fallback "
                        f"handler registered for '{self.schema.fallback_intent}'"
                    )
                logger.warning(f"No handler for '{scored.intent}', using fallback")
            outputs.append(handler(user_message, scored.entities))

        if not outputs and skipped > 0:
            logger.warning(
                f"All {skipped} intents were below confidence threshold "
                f"({self.schema.confidence_threshold}); no handlers invoked"
            )
        return outputs

    # ----------------------------------------------------------
    # Validation & Defaults
    # ----------------------------------------------------------

    def validate(self) -> list[str]:
        """Check that all schema intents (including fallback) have registered handlers.

        Returns a list of intent names that lack handlers (empty if all covered).
        """
        missing = [
            name for name in self.schema.intent_names if name not in self._handlers
        ]
        if (
            self.schema.fallback_intent not in self._handlers
            and self.schema.fallback_intent not in missing
        ):
            missing.append(self.schema.fallback_intent)
        if missing:
            logger.warning(f"Intents without handlers: {missing}")
        return missing

    @staticmethod
    def _default_clarify(user_message: str, entities: dict[str, str]) -> str:
        return (
            "I'm not sure I understand your request. "
            "Could you please rephrase or provide more details?"
        )
