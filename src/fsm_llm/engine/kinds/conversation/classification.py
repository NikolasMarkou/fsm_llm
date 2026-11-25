"""Classification extraction of the conversation kind (``ConversationClassificationMixin``):
the ``classification_extractions`` of a state, one classifier call per entry.

Layer ``engine`` (plan d454fd62): split out of the old ``fsm_llm.pipeline`` at step 79 and moved
here at step 80 (D-030), where the conversation executor owns it. The method and function bodies
are unchanged except that a service of the pipeline is read as ``self._services.<name>`` (the
``_PipelineServices`` protocol, ``engine/kinds/services.py``) instead of ``self.<name>``.
"""

from __future__ import annotations

import math
from typing import Any

from fsm_llm.aio import acall, run_sync
from fsm_llm.core.definitions import (
    ClassificationError,
    ClassificationExtractionConfig,
    ClassificationPromptConfig,
    ClassificationResult,
    ClassificationSchema,
    FSMInstance,
    State,
)
from fsm_llm.logging import logger
from fsm_llm.runtime import (
    EventType,
    current_scope,
    emit,
    visible_entries,
    warn_on_failure,
)

from ..records import (
    _CLASSIFICATION_SOFT_FAIL_EXCEPTIONS,
    _classification_record,
    _json_native_values,
    _record_classification_result,
)
from .surface import _ConversationSurface


def _record_out_of_scope(
    state: State,
    config: ClassificationExtractionConfig,
    result: ClassificationResult,
    *,
    low_confidence: bool,
) -> None:
    """Append ``intent/out_of_scope`` (a fallback or a discarded low-confidence intent)."""
    if current_scope() is None or not visible_entries({config.field_name: None}):
        return
    confidence = (
        {"confidence": result.confidence} if math.isfinite(result.confidence) else {}
    )
    with warn_on_failure(EventType.INTENT_OUT_OF_SCOPE):
        emit(
            EventType.INTENT_OUT_OF_SCOPE,
            state=state.id,
            field=config.field_name,
            intent=result.intent,
            low_confidence=low_confidence,
            **confidence,
        )


class ConversationClassificationMixin(_ConversationSurface):
    """Classification extraction of the conversation kind."""

    def _execute_classification_extractions(
        self,
        current_state: State,
        user_message: str | None,
        instance: FSMInstance,
        conversation_id: str,
        *,
        configs_override: list[ClassificationExtractionConfig] | None = None,
    ) -> dict[str, Any]:
        """``run_sync`` over ``_aexecute_classification_extractions``, the one body."""
        return run_sync(
            self._aexecute_classification_extractions(
                current_state,
                user_message,
                instance,
                conversation_id,
                configs_override=configs_override,
            )
        )

    async def _aexecute_classification_extractions(
        self,
        current_state: State,
        user_message: str | None,
        instance: FSMInstance,
        conversation_id: str,
        *,
        configs_override: list[ClassificationExtractionConfig] | None = None,
    ) -> dict[str, Any]:
        """Run classification extractions and return extracted data.

        For each :class:`ClassificationExtractionConfig`, builds a
        :class:`ClassificationSchema`, creates a :class:`Classifier`,
        and stores the result in two places:

        - ``field_name`` → intent string in the returned dict (simple,
          JsonLogic-friendly; fallback always, other intents only at or above
          ``confidence_threshold``)
        - ``context.metadata["classification_results"][field_name]`` → full
          result dict (intent, confidence, reasoning, entities, plus
          ``low_confidence`` when below ``confidence_threshold`` (a non-fallback
          intent is then discarded) and ``context_snapshot`` of the
          JSON-native ``context_keys`` values), written for every result
          (plan-2026-09-21T203800-8a03483a/D-005). Readable via
          ``FSMManager.get_complete_conversation``.

        Args:
            current_state: Current state (for config lookup).
            user_message: User input to classify, or ``None`` when the turn
                has no user message (the classifier then decides from the
                per-call context alone).
            instance: FSM instance (per-call context and the result record).
            conversation_id: Logging context.
            configs_override: If provided, run only these configs
                (used during retry).

        Returns:
            Dict of extracted key-value pairs to merge into context.
        """
        log = logger.bind(conversation_id=conversation_id)
        configs = configs_override or self._classification_configs(
            current_state, instance
        )
        if not configs:
            return {}

        extracted: dict[str, Any] = {}

        for config in configs:
            try:
                schema = ClassificationSchema(
                    intents=config.intents,
                    fallback_intent=config.fallback_intent,
                    confidence_threshold=config.confidence_threshold,
                )

                prompt_config = None
                if config.prompt_config:
                    prompt_config = ClassificationPromptConfig(**config.prompt_config)

                classifier = self._services._get_classifier(
                    schema, prompt_config, model=config.model
                )

                result: ClassificationResult = await acall(
                    classifier,
                    "classify",
                    user_message,
                    context=self._services._build_classifier_context(
                        instance,
                        current_state,
                        conversation_id,
                        config.context_keys,
                        user_message=user_message,
                    ),
                )

                log.debug(
                    f"Classification extraction '{config.field_name}': "
                    f"intent={result.intent}, confidence={result.confidence:.2f}"
                )

                # plan-2026-09-21T203800-8a03483a/D-005 (A4): the full result
                # of EVERY classification of this field (fallback and
                # discarded low-confidence ones too) is the inspectable
                # record; see _record_classification_result.
                full_result = _classification_record(
                    result, config.confidence_threshold
                )
                if config.context_keys:
                    full_result["context_snapshot"] = _json_native_values(
                        instance.context.data, config.context_keys
                    )
                _record_classification_result(instance, config.field_name, full_result)

                # Always store fallback intent so the context key exists
                # for downstream JsonLogic conditions
                low = "low_confidence" in full_result
                if result.intent == config.fallback_intent:
                    extracted[config.field_name] = result.intent
                    _record_out_of_scope(
                        current_state, config, result, low_confidence=low
                    )
                    log.debug(
                        f"Classification extraction '{config.field_name}': "
                        f"fallback intent '{result.intent}' stored"
                    )
                    continue

                # Skip low confidence
                if low:
                    _record_out_of_scope(
                        current_state, config, result, low_confidence=True
                    )
                    # LV2-05 (D-037): an accepted stall (D-016), now visible: a
                    # gated key left unset here holds the state with no other trace.
                    log.warning(
                        f"Classification extraction '{config.field_name}': "
                        f"intent '{result.intent}' at confidence "
                        f"{result.confidence:.2f} is below threshold "
                        f"{config.confidence_threshold}, discarded; the key stays "
                        "unset and a transition gated on it will not fire"
                    )
                    continue

                # Store simple value (user-visible, works with JsonLogic)
                extracted[config.field_name] = result.intent

                log.info(
                    f"Classification extraction '{config.field_name}' = "
                    f"'{result.intent}' (confidence={result.confidence:.2f})"
                )

            except _CLASSIFICATION_SOFT_FAIL_EXCEPTIONS as e:
                if config.required:
                    raise ClassificationError(
                        f"Required classification extraction '{config.field_name}' "
                        f"failed: {e}"
                    ) from e
                log.warning(
                    f"Classification extraction '{config.field_name}' failed: {e}"
                )
                continue

        return extracted
