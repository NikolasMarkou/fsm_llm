"""
Core classifier implementation.

Uses LiteLLM (via the fsm_llm LLM interface pattern) to classify user input
against a predefined schema. Supports single-intent, multi-intent, and
hierarchical (two-stage) classification.
"""

from __future__ import annotations

import json
import time
from typing import Optional

from litellm import completion, get_supported_openai_params
from loguru import logger

from .definitions import (
    ClassificationSchema,
    ClassificationResult,
    ClassificationResponseError,
    HierarchicalResult,
    HierarchicalSchema,
    IntentScore,
    MultiClassificationResult,
)
from .prompts import (
    ClassificationPromptConfig,
    build_json_schema,
    build_system_prompt,
)


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
        model: str = "gpt-4o-mini",
        *,
        api_key: Optional[str] = None,
        config: Optional[ClassificationPromptConfig] = None,
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

        # Pre-build prompts so they're not reconstructed on every call
        self._system_prompt = build_system_prompt(schema, self.config)
        self._json_schema = build_json_schema(
            schema,
            multi_intent=False,
            include_reasoning=self.config.include_reasoning,
            include_entities=self.config.include_entities,
        )
        self._multi_json_schema = build_json_schema(
            schema,
            multi_intent=True,
            max_intents=self.config.max_intents,
            include_reasoning=self.config.include_reasoning,
            include_entities=self.config.include_entities,
        )

        logger.info(
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

        messages = [
            {"role": "system", "content": self._system_prompt},
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

        response = completion(**call_params)
        elapsed = time.time() - start

        if not response or not getattr(response, "choices", None):
            raise ClassificationResponseError("Empty response from LLM")

        content = response.choices[0].message.content
        if content is None:
            raise ClassificationResponseError("LLM returned null content")

        logger.debug(f"Classification call completed in {elapsed:.2f}s")

        if isinstance(content, dict):
            return content

        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ClassificationResponseError(
                f"Failed to parse LLM JSON: {exc}\nResponse: {content[:200]}"
            ) from exc

        if not isinstance(data, dict):
            raise ClassificationResponseError(
                f"Expected JSON object, got {type(data).__name__}"
            )
        return data

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

        return ClassificationResult(
            reasoning=data.get("reasoning", ""),
            intent=intent,
            confidence=float(data.get("confidence", 0.0)),
            entities=data.get("entities", {}),
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
            name = item.get("intent", "")
            if name not in valid_names:
                name = self.schema.fallback_intent
            scored.append(
                IntentScore(
                    intent=name,
                    confidence=float(item.get("confidence", 0.0)),
                    entities=item.get("entities", {}),
                )
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
        model: str = "gpt-4o-mini",
        *,
        api_key: Optional[str] = None,
        config: Optional[ClassificationPromptConfig] = None,
        **litellm_kwargs,
    ) -> None:
        self.schema = schema
        shared = dict(model=model, api_key=api_key, config=config, **litellm_kwargs)

        self._domain_classifier = Classifier(
            schema=schema.domain_schema, **shared
        )
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
            return HierarchicalResult(
                domain=domain_result,
                intent=domain_result,
            )

        intent_result = sub.classify(user_message)
        return HierarchicalResult(
            domain=domain_result,
            intent=intent_result,
        )
