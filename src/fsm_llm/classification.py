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

from .constants import (
    DEFAULT_LLM_MODEL,
    MAX_MULTI_INTENTS,
    RESERVED_LLM_CALL_KWARGS,
)
from .definitions import (
    ClassificationError,
    ClassificationResponseError,
    ClassificationResult,
    ClassificationSchema,
    CompletionRequest,
    CompletionResponse,
    HierarchicalResult,
    HierarchicalSchema,
    IntentScore,
    MultiClassificationResult,
)
from .llm import LiteLLMInterface, LLMInterface
from .logging import logger
from .prompts import (
    ClassificationPromptConfig,
    build_classification_context_block,
    build_classification_json_schema,
    build_classification_system_prompt,
)
from .utilities import (
    coerce_confidence,
    extract_json_from_text,
)

# Type alias for intent handler functions
HandlerFn = Callable[[str, dict[str, str | None]], Any]


def _reasoning_text(value: Any) -> str:
    """A model-supplied ``reasoning`` as the required ``str`` field.

    A ``str`` is returned as-is; ``null`` or any other type becomes ``""``
    (audit D8: it used to raise a raw pydantic ``ValidationError`` out of
    ``classify``). Never raises.
    """
    return value if isinstance(value, str) else ""


# --------------------------------------------------------------
# Classifier
# --------------------------------------------------------------


class Classifier:
    """
    LLM-backed intent classifier.

    Wraps a ClassificationSchema and an LLM interface to provide a simple
    ``classify()`` / ``classify_multi()`` interface. Every request is one
    ``complete`` call (a ``json_schema`` response format, the prompt config's
    ``temperature`` and ``max_tokens``) on one ``LLMInterface``:

    - ``llm`` given: that interface (its model, connection settings, timeout
      and usage meter). ``model`` defaults to the interface's ``model``
      attribute; a ``model`` that differs from it, an ``api_key`` or any
      ``llm_kwargs`` are refused (``ValueError``): the interface owns the
      model and the connection. An interface that does not implement
      ``complete`` makes every call fail with ``ClassificationError``.
    - ``llm`` omitted: a ``fsm_llm.LiteLLMInterface`` constructed here from
      ``model`` (default ``DEFAULT_LLM_MODEL``), ``api_key`` and
      ``llm_kwargs`` (``timeout`` defaults to 120 seconds; ``retries`` has
      that interface's meaning; names in ``RESERVED_LLM_CALL_KWARGS`` are
      ignored, the prompt config owns ``temperature`` and ``max_tokens``).
    """

    def __init__(
        self,
        schema: ClassificationSchema,
        model: str | None = None,
        *,
        llm: LLMInterface | None = None,
        api_key: str | None = None,
        config: ClassificationPromptConfig | None = None,
        **llm_kwargs,
    ) -> None:
        if model is not None and not model.strip():
            raise ValueError("model must be a non-empty string")

        self.schema = schema
        self.config = config or ClassificationPromptConfig()
        # DECISION plan-2026-10-01T093600-944e2692/D-002 (supersedes
        # 07ad3f8c/D-022): the classifier sends one `LLMInterface.complete`
        # request per call and reads the typed reply. Do NOT import the
        # provider SDK here, rebuild call params (model kwargs, timeout,
        # response_format, Ollama preparation, the neutral user turn) or read
        # provider objects: llm.py is the one request path and the one reply
        # reader. The 120s default bounds a stalled provider so it cannot hold
        # the conversation thread and its conv_lock (CA3-003). See decisions.md
        # D-002.
        # DECISION plan-2026-10-01T093600-944e2692/D-020: an injected `llm` is
        # used as is. Do NOT merge `api_key`/`llm_kwargs` into it or build a
        # second interface beside it (two LLM paths for one classifier), and
        # do NOT fall back to a private LiteLLMInterface when the injected one
        # lacks `complete`: that is the bypass D-006 closes. See D-020.
        self._llm: LLMInterface
        if llm is not None:
            if api_key is not None or llm_kwargs:
                raise ValueError(
                    "Classifier(llm=...) takes no api_key or connection kwargs: "
                    "the injected interface owns its connection settings"
                )
            interface_model = getattr(llm, "model", None)
            if not isinstance(interface_model, str):
                interface_model = None
            if model is not None and interface_model not in (None, model):
                raise ValueError(
                    f"Classifier(model={model!r}, llm=...) names a different "
                    f"model than the injected interface ({interface_model!r}); "
                    "omit llm to classify with another model"
                )
            self._llm = llm
            self.model: str | None = model if model is not None else interface_model
        else:
            self.model = model if model is not None else DEFAULT_LLM_MODEL
            connection: dict[str, Any] = {
                "timeout": 120.0,
                **{
                    k: v
                    for k, v in llm_kwargs.items()
                    if k not in RESERVED_LLM_CALL_KWARGS
                },
            }
            self._llm = LiteLLMInterface(
                self.model,
                api_key=api_key,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                **connection,
            )

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
            f"Classifier initialized: model={self.model}, "
            f"interface={type(self._llm).__name__}, "
            f"intents={len(schema.intents)}, "
            f"threshold={schema.confidence_threshold}"
        )

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def classify(
        self, user_message: str | None, context: dict[str, Any] | None = None
    ) -> ClassificationResult:
        """
        Classify a single user message into one intent.

        Args:
            user_message: The message to classify (sent as the user turn), or
                ``None`` when there is no user message: the system prompt
                then asks for a classification of ``context`` alone and the
                user turn is left empty for the LLM layer to fill.
            context: Optional per-call context ``{"history", "purpose",
                "data"}`` rendered, sanitized and security-filtered, into the
                system prompt (see ``build_classification_context_block``).
                None or ``{}`` sends exactly the context-free prompt.

        Returns:
            ClassificationResult with intent, confidence, reasoning, and entities.

        Raises:
            ClassificationResponseError: If the LLM response cannot be parsed.
        """
        raw = self._call_llm(user_message, multi_intent=False, context=context)
        return self._parse_single(raw)

    def classify_multi(
        self, user_message: str | None, context: dict[str, Any] | None = None
    ) -> MultiClassificationResult:
        """
        Classify a message that may contain multiple intents.

        ``user_message`` and ``context`` have the meaning they have in
        ``classify``.

        Returns:
            MultiClassificationResult with a ranked list of IntentScores.
        """
        raw = self._call_llm(user_message, multi_intent=True, context=context)
        return self._parse_multi(raw)

    def is_low_confidence(self, result: ClassificationResult) -> bool:
        """Check if a result falls below the schema's confidence threshold."""
        return result.confidence < self.schema.confidence_threshold

    # ----------------------------------------------------------
    # LLM Communication
    # ----------------------------------------------------------

    def _call_llm(
        self,
        user_message: str | None,
        *,
        multi_intent: bool,
        context: dict[str, Any] | None = None,
    ) -> dict:
        """Make the LLM call and return the parsed JSON dict.

        ``user_message`` of ``None`` selects the context-only system prompt
        and reaches the LLM layer as ``None`` (no user message); a string,
        ``""`` included, is sent as the user's message.
        """
        start = time.time()

        # DECISION plan-2026-09-21T203800-8a03483a/D-004: per-call context is
        # appended HERE, to the cached base prompt, never baked into
        # __init__. Do NOT move it into the constructor or the pipeline's
        # classifier cache key: one Classifier serves every turn of a state, so
        # a constructor input would either go stale or bust the content-keyed
        # cache on every turn. Do NOT render context text without the shared
        # prompts.py sanitizer/filter: history and data are user-controlled.
        if user_message is None:
            # Built per call: the cached prompts are the message wording, and
            # a message-free call is the rarer path.
            base_prompt = build_classification_system_prompt(
                self.schema,
                replace(self.config, multi_intent=multi_intent),
                user_message=None,
            )
        else:
            base_prompt = (
                self._multi_system_prompt if multi_intent else self._system_prompt
            )
        system_prompt = base_prompt + build_classification_context_block(
            context, user_message=user_message
        )
        request = CompletionRequest(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "intent_classification",
                    "schema": (
                        self._multi_json_schema if multi_intent else self._json_schema
                    ),
                },
            },
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            call_type="classification",
        )
        # Every failure of the call (an outage, an unreadable reply: both
        # LLMResponseError from `complete`) is a ClassificationError, a member
        # of the pipeline's soft-fail tuple, never a bare AttributeError
        # (review N10, plan-2026-09-20T165703-0d9c218e).
        try:
            response = self._llm.complete(request)
        except Exception as e:
            raise ClassificationError(f"Classification LLM call failed: {e!s}") from e
        logger.debug(f"Classification call completed in {time.time() - start:.2f}s")
        return self._extract_response(response)

    # ----------------------------------------------------------
    # Response Extraction
    # ----------------------------------------------------------

    @staticmethod
    def _extract_response(response: CompletionResponse) -> dict:
        """
        Extract a JSON dict from a classification reply.

        ``response.text`` is the reply content, or the model's reasoning trace
        when the content was empty (recovered by the LLM layer for a reply
        with no tool call). Raises ``ClassificationResponseError`` when there
        is no text or no JSON object in it.
        """
        text = response.text
        if response.kind != "final" or not text:
            raise ClassificationResponseError("LLM returned empty content")
        data = extract_json_from_text(text)
        if data is not None:
            return data
        raise ClassificationResponseError(
            f"Failed to parse LLM JSON.\nResponse: {text[:200]}"
        )

    # ----------------------------------------------------------
    # Response Parsing
    # ----------------------------------------------------------

    def _resolve_intent(self, raw: Any) -> str | None:
        """Map a model-supplied intent onto a declared intent name.

        Contract: an exact match wins; otherwise a case- and
        surrounding-whitespace-insensitive match is accepted only when it
        folds onto exactly ONE declared name (``"BUY"`` -> ``"buy"``). Returns
        None for a non-``str`` value, an unknown name, or an ambiguous fold,
        so the caller applies the fallback intent. Never raises.
        """
        if not isinstance(raw, str):
            return None
        names = self.schema.intent_names
        if raw in names:
            return raw
        folded = raw.strip().casefold()
        matches = [n for n in names if n.casefold() == folded]
        return matches[0] if len(matches) == 1 else None

    def _parse_single(self, data: dict) -> ClassificationResult:
        raw_intent = data.get("intent", "")
        intent = self._resolve_intent(raw_intent)
        if intent is None:
            logger.warning(
                f"LLM returned unknown intent {raw_intent!r}, "
                f"falling back to '{self.schema.fallback_intent}'"
            )
            intent = self.schema.fallback_intent

        raw_confidence = data.get("confidence", 0.0)
        try:
            # coerce_confidence maps NaN/±inf → 0.0 (so is_low_confidence fires,
            # NOT 1.0) and clamps; a `{...}`/`null` still raises for the existing
            # warn+default-0.0 rung below (D-001, utilities.py).
            confidence = coerce_confidence(raw_confidence, 0.0)
        except (ValueError, TypeError):
            logger.warning(
                f"Invalid confidence value {raw_confidence!r}, defaulting to 0.0"
            )
            confidence = 0.0
        return ClassificationResult(
            reasoning=_reasoning_text(data.get("reasoning")),
            intent=intent,
            confidence=confidence,
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

        scored: list[IntentScore] = []
        for item in raw_intents:
            if not isinstance(item, dict):
                logger.warning(
                    f"Skipping non-dict item in multi-intent response: {item!r}"
                )
                continue
            raw_name = item.get("intent", "")
            name = self._resolve_intent(raw_name)
            if name is None:
                logger.warning(
                    f"LLM returned unknown intent {raw_name!r} in multi-intent "
                    f"response, falling back to '{self.schema.fallback_intent}'"
                )
                name = self.schema.fallback_intent
            raw_confidence = item.get("confidence", 0.0)
            try:
                # NaN/±inf → 0.0 via the shared coercer; `{...}`/`null` still
                # raises for the warn+default-0.0 rung (D-001, utilities.py).
                confidence = coerce_confidence(raw_confidence, 0.0)
            except (ValueError, TypeError):
                logger.warning(
                    f"Invalid confidence value {raw_confidence!r}, defaulting to 0.0"
                )
                confidence = 0.0
            scored.append(
                IntentScore(
                    intent=name,
                    confidence=confidence,
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

        # MultiClassificationResult.intents enforces max_length=MAX_MULTI_INTENTS;
        # keep the highest-scored intents to avoid an uncaught pydantic
        # ValidationError.
        if len(scored) > MAX_MULTI_INTENTS:
            logger.warning(
                "Multi-intent response truncated: discarding "
                f"{len(scored) - MAX_MULTI_INTENTS} of {len(scored)} valid intents "
                f"to fit the max_length={MAX_MULTI_INTENTS} cap "
                f"(max_intents={self.config.max_intents} was requested)"
            )
        scored = scored[:MAX_MULTI_INTENTS]

        return MultiClassificationResult(
            reasoning=_reasoning_text(data.get("reasoning")),
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
        **llm_kwargs,
    ) -> None:
        self.schema = schema
        shared = dict(model=model, api_key=api_key, config=config, **llm_kwargs)

        self._domain_classifier = Classifier(schema=schema.domain_schema, **shared)
        self._intent_classifiers: dict[str, Classifier] = {
            domain: Classifier(schema=intent_schema, **shared)
            for domain, intent_schema in schema.intent_schemas.items()
        }

    def classify(
        self, user_message: str | None, context: dict[str, Any] | None = None
    ) -> HierarchicalResult:
        """
        Run two-stage classification: domain then intent.

        If the domain result maps to the fallback and no sub-classifier exists,
        the intent result mirrors the domain result. ``context`` is forwarded
        to both stages (see ``Classifier.classify``).
        """
        domain_result = self._domain_classifier.classify(user_message, context)

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

        intent_result = sub.classify(user_message, context)
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
