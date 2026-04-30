from __future__ import annotations

"""ExtractionEngine: Pass-1 extraction logic for the FSM dialog turn.

Houses the data extraction + field extraction + classification extraction
cluster originally embedded in :class:`MessagePipeline` (see
:mod:`fsm_llm.dialog.turn`). The engine is owned by the parent pipeline
(``MessagePipeline._extraction``) and delegates back to the pipeline for
shared host-side concerns (``get_state``, ``execute_handlers``, the
``_apply_context_scope`` / ``_clean_empty_context_keys`` /
``_build_field_configs_from_state`` static helpers, plus ``llm_interface``
and the prompt builders).

# DECISION (Phase C, 0.8.0) — pure code motion. The 8 methods preserve
# their docstrings, comments, decorators, and error semantics verbatim.
# The single-Oracle invariant (M4 / test_oracle_ownership.py) is preserved
# by reading ``self._pipeline._oracle`` at call time — the engine never
# constructs an Oracle of its own.
"""

import json
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..constants import CLASSIFICATION_EXTRACTION_RESULT_SUFFIX
from ..handlers import HandlerTiming
from ..logging import logger
from .._models import (
    ClassificationError,
    DataExtractionResponse,
    FieldExtractionRequest,
    FieldExtractionResponse,
)
from .classification import Classifier
from .definitions import (
    ClassificationExtractionConfig,
    ClassificationResult,
    ClassificationSchema,
    FieldExtractionConfig,
    FSMInstance,
    State,
)
from .prompts import ClassificationPromptConfig

if TYPE_CHECKING:
    from .turn import MessagePipeline, _TurnState


# --- Type coercion dispatch for field extraction validation ---


def _coerce_int(v: Any) -> int:
    return v if isinstance(v, int) else int(v)


def _coerce_float(v: Any) -> float:
    return v if isinstance(v, float) else float(v)


def _coerce_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("true", "1", "yes")
    return bool(v)


def _coerce_str(v: Any) -> str:
    return v if isinstance(v, str) else str(v)


def _coerce_list(v: Any) -> Any:
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        parsed = json.loads(v)
        if not isinstance(parsed, list):
            raise TypeError("not a list")
        return parsed
    return v


def _coerce_dict(v: Any) -> Any:
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        parsed = json.loads(v)
        if not isinstance(parsed, dict):
            raise TypeError("not a dict")
        return parsed
    return v


_TYPE_COERCERS: dict[str, Callable[[Any], Any]] = {
    "int": _coerce_int,
    "float": _coerce_float,
    "bool": _coerce_bool,
    "str": _coerce_str,
    "list": _coerce_list,
    "dict": _coerce_dict,
    # "any" — no coercion, not in dispatch dict
}


class ExtractionEngine:
    """Pass-1 extraction cluster.

    Owned by :class:`MessagePipeline` as ``self._extraction``. The engine
    holds a back-reference to the parent pipeline so it can read live
    attributes (``llm_interface``, ``_oracle``, prompt builders,
    ``fsm_resolver``) and call back to host-side helpers
    (``get_state``, ``execute_handlers``, ``_apply_context_scope``,
    ``_clean_empty_context_keys``, ``_build_field_configs_from_state``).

    Parameters
    ----------
    pipeline:
        The parent :class:`MessagePipeline`. The engine never constructs
        its own :class:`~fsm_llm.runtime.oracle.Oracle`; calls go through
        ``self._pipeline._oracle`` so the M4 single-Oracle invariant
        holds end-to-end.
    """

    def __init__(self, pipeline: MessagePipeline) -> None:
        self._pipeline = pipeline

    # ----------------------------------------------------------
    # Pipeline-attribute pass-throughs
    # ----------------------------------------------------------
    # These are *read* at call time so runtime mutations on the parent
    # pipeline (e.g. tests that swap out ``llm_interface``) propagate.

    @property
    def llm_interface(self) -> Any:
        return self._pipeline.llm_interface

    @property
    def field_extraction_prompt_builder(self) -> Any:
        return self._pipeline.field_extraction_prompt_builder

    @property
    def fsm_resolver(self) -> Any:
        return self._pipeline.fsm_resolver

    @property
    def _oracle(self) -> Any:
        return self._pipeline._oracle

    # ----------------------------------------------------------
    # Pipeline-helper pass-throughs (delegation back to host)
    # ----------------------------------------------------------

    def get_state(
        self, instance: FSMInstance, conversation_id: str | None = None
    ) -> State:
        return self._pipeline.get_state(instance, conversation_id)

    def execute_handlers(
        self,
        instance: FSMInstance,
        timing: HandlerTiming,
        conversation_id: str,
        current_state: str | None = None,
        target_state: str | None = None,
        updated_keys: set[str] | None = None,
        error_context: dict[str, Any] | None = None,
    ) -> None:
        self._pipeline.execute_handlers(
            instance,
            timing,
            conversation_id,
            current_state=current_state,
            target_state=target_state,
            updated_keys=updated_keys,
            error_context=error_context,
        )

    @staticmethod
    def _build_field_configs_from_state(state: State) -> list[FieldExtractionConfig]:
        # Static helper lives on MessagePipeline; defer to it so the
        # legacy ``MessagePipeline._build_field_configs_from_state(state)``
        # call shape used by the test suite stays intact.
        from .turn import MessagePipeline

        return MessagePipeline._build_field_configs_from_state(state)

    @staticmethod
    def _apply_context_scope(
        context: dict[str, Any],
        state: State,
        conversation_id: str,
    ) -> dict[str, Any]:
        from .turn import MessagePipeline

        return MessagePipeline._apply_context_scope(context, state, conversation_id)

    @staticmethod
    def _clean_empty_context_keys(
        data: dict[str, Any], conversation_id: str, remove_none_values: bool = True
    ) -> dict[str, Any]:
        from .turn import MessagePipeline

        return MessagePipeline._clean_empty_context_keys(
            data, conversation_id, remove_none_values
        )

    # ----------------------------------------------------------
    # Compiled-path callback factory
    # ----------------------------------------------------------

    def _make_cb_extract(
        self,
        instance: FSMInstance,
        message: str,
        conversation_id: str,
        turn_state: _TurnState,
    ) -> Callable[[FSMInstance], Any]:
        """`CB_EXTRACT` / `CB_FIELD_EXTRACT` / `CB_CLASS_EXTRACT` binding.

        All three slots share this implementation — the FIRST extraction
        callback to fire delegates to `_execute_data_extraction` and fires
        CONTEXT_UPDATE; subsequent calls within the same turn are no-ops.

        Why: the compiler emits separate Lets for bulk / field / class
        based on state configuration, but `_execute_data_extraction`
        coordinates them in a single pass (with cross-stage behaviors like
        skip-if-in-context and multi-pass retry). Per-callback primitives
        would diverge semantically — assumption A3 in plan.md. Using a
        single dispatched entry point guarded by
        `extraction_dispatcher_ran` preserves byte-for-byte equivalence
        with the pre-compiled 2-pass flow (retired in S11) at the cost of
        the λ-kernel's per-callback granularity (acceptable — the tiered
        cohort test suite is the compliance gate, not formal
        single-responsibility per slot).
        """

        def _extract(_inst: FSMInstance) -> Any:
            if turn_state.extraction_dispatcher_ran:
                return None
            turn_state.extraction_dispatcher_ran = True

            extraction_response = self._execute_data_extraction(
                instance, message, conversation_id
            )
            turn_state.extraction_response = extraction_response

            # Mirror the context-update + CONTEXT_UPDATE handler fire used
            # by the pre-compiled 2-pass flow (retired in S11).
            if extraction_response.extracted_data:
                extraction_response.extracted_data = self._clean_empty_context_keys(
                    data=extraction_response.extracted_data,
                    conversation_id=conversation_id,
                )
                if extraction_response.extracted_data:
                    instance.context.update(extraction_response.extracted_data)
                    self.execute_handlers(
                        instance,
                        HandlerTiming.CONTEXT_UPDATE,
                        conversation_id,
                        current_state=instance.current_state,
                        updated_keys=set(extraction_response.extracted_data.keys()),
                    )
            return None

        return _extract

    # ----------------------------------------------------------
    # Post-transition re-extraction
    # ----------------------------------------------------------

    def _post_transition_reextract(
        self,
        instance: FSMInstance,
        user_message: str,
        turn_state: _TurnState,
        conversation_id: str,
    ) -> None:
        """Post-transition re-extraction outer wrap (S8b step 3).

        Factored from the pre-compiled 2-pass flow (retired in S11).
        Preserves exception-swallow-with-warning and the `missing_configs`
        filter. Caller guards on `turn_state.transition_occurred` and the
        `agent_trace` check.
        """
        log = logger.bind(conversation_id=conversation_id)
        new_state = self.get_state(instance, conversation_id)
        new_configs = self._build_field_configs_from_state(new_state)
        missing_configs = [
            c
            for c in new_configs
            if c.field_name not in instance.context.data
            or instance.context.data.get(c.field_name) is None
        ]
        if not missing_configs:
            return

        log.debug(
            f"Post-transition extraction in "
            f"'{instance.current_state}' for "
            f"{[c.field_name for c in missing_configs]}"
        )
        try:
            post_results = self._execute_field_extractions(
                instance, user_message, missing_configs, conversation_id
            )
            post_data: dict[str, Any] = {}
            for result in post_results:
                if result.is_valid and result.value is not None:
                    post_data[result.field_name] = result.value

            if post_data:
                post_data = self._clean_empty_context_keys(
                    data=post_data, conversation_id=conversation_id
                )
                if post_data:
                    instance.context.update(post_data)
                    if turn_state.extraction_response is not None:
                        turn_state.extraction_response.extracted_data.update(post_data)
                    self.execute_handlers(
                        instance,
                        HandlerTiming.CONTEXT_UPDATE,
                        conversation_id,
                        current_state=instance.current_state,
                        updated_keys=set(post_data.keys()),
                    )
        except Exception as e:
            log.warning(f"Post-transition extraction failed (non-fatal): {e}")

    # ----------------------------------------------------------
    # Bulk extraction fallback
    # ----------------------------------------------------------

    def _bulk_extract_from_instructions(
        self,
        instance: FSMInstance,
        user_message: str,
        state: State,
        conversation_id: str,
    ) -> dict[str, Any]:
        """Bulk-extract data when a state has extraction_instructions but no
        explicit required_context_keys or field_extractions.

        Uses a single LLM call with a simple prompt to extract whatever
        data the instructions describe.  Returns a dict of extracted
        key-value pairs (may be empty).
        """
        log = logger.bind(conversation_id=conversation_id)

        prompt = (
            f"Extract information from the user's message.\n\n"
            f"Instructions: {state.extraction_instructions}\n\n"
            f"User message: {user_message}\n\n"
            f'Respond with JSON: {{"extracted_data": {{"key": "value", ...}}, '
            f'"confidence": 0.95, "reasoning": "..."}}\n\n'
            f"Only include keys for information actually present in the "
            f"user's message. Use descriptive snake_case key names."
        )

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_message},
        ]

        try:
            # DECISION D-PIVOT-1-CALLSITE (step 11, plan_2026-04-27_32652286):
            # bulk-extract site rewired through `oracle.invoke_messages` —
            # the new pre-built-message-array surface added in step 10
            # (D-PIVOT-1-ORACLE). Returns the raw litellm response so the
            # inline <think>/markdown-fence/extracted_data parsing below
            # is byte-equivalent to the legacy `_make_llm_call(messages,
            # "data_extraction")` path. Replaces the deferred-site marker
            # at D-R10-7.1.
            # M4 — Program-owned Oracle field-read.
            _oracle = self._oracle
            response = _oracle.invoke_messages(messages, call_type="data_extraction")
            content = response.choices[0].message.content
            if isinstance(content, str):
                content = re.sub(
                    r"<think>.*?</think>", "", content, flags=re.DOTALL
                ).strip()
                content = re.sub(
                    r"^```(?:json)?\s*\n?", "", content, flags=re.MULTILINE
                )
                content = re.sub(r"\n?```\s*$", "", content).strip()

            if isinstance(content, str):
                import json as json_mod

                data = json_mod.loads(content)
            elif isinstance(content, dict):
                data = content
            else:
                return {}

            extracted = data.get("extracted_data", data)
            if isinstance(extracted, dict):
                # Filter out None/empty values
                return {
                    k: v
                    for k, v in extracted.items()
                    if v is not None and v != "" and v != {}
                }
        except Exception as e:
            log.warning(f"Bulk extraction fallback failed: {e}")

        return {}

    # ----------------------------------------------------------
    # Pass 1: Field-based extraction (replaces bulk extract_data)
    # ----------------------------------------------------------

    def _execute_data_extraction(
        self, instance: FSMInstance, user_message: str, conversation_id: str
    ) -> DataExtractionResponse:
        """Execute data extraction via per-field ``extract_field`` calls
        and classification extractions.

        Builds a unified list of ``FieldExtractionConfig`` from both
        legacy ``required_context_keys`` and explicit ``field_extractions``,
        then extracts each field individually.  Also runs any
        ``classification_extractions`` declared on the state.  Supports
        multi-pass retry for missing required fields (up to ``extraction_retries``).
        """
        log = logger.bind(conversation_id=conversation_id)
        log.debug("Executing field-based data extraction")

        current_state = self.get_state(instance, conversation_id)

        # Build unified field configs
        all_configs = self._build_field_configs_from_state(current_state)

        has_field_configs = bool(all_configs)
        has_classification_configs = bool(current_state.classification_extractions)

        has_extraction_instructions = bool(current_state.extraction_instructions)

        if not has_field_configs and not has_classification_configs:
            if has_extraction_instructions:
                # Fallback: bulk extraction for states with instructions
                # but no explicit field configs.  Uses a single LLM call
                # to extract any relevant data the instructions describe.
                log.debug(
                    "No field configs but extraction_instructions present; "
                    "using bulk extraction fallback"
                )
                bulk_data = self._bulk_extract_from_instructions(
                    instance, user_message, current_state, conversation_id
                )
                # Don't overwrite values already set in context (e.g. by
                # handlers) — bulk extraction is best-effort for NEW data.
                if bulk_data:
                    existing = instance.context.data
                    bulk_data = {
                        k: v for k, v in bulk_data.items() if existing.get(k) is None
                    }
                if bulk_data:
                    response = DataExtractionResponse(
                        extracted_data=bulk_data,
                        confidence=0.8,
                    )
                    instance.last_extraction_response = response
                    return response

            log.debug("No fields or classifications to extract for this state")
            response = DataExtractionResponse(extracted_data={}, confidence=1.0)
            instance.last_extraction_response = response
            return response

        extracted_data: dict[str, Any] = {}
        confidences: list[float] = []

        # --- Field extractions ---
        if has_field_configs:
            # Skip fields already set in context (e.g. by handlers)
            existing = instance.context.data
            all_configs = [c for c in all_configs if existing.get(c.field_name) is None]
            results = self._execute_field_extractions(
                instance, user_message, all_configs, conversation_id
            )
            for result in results:
                if result.is_valid and result.value is not None:
                    extracted_data[result.field_name] = result.value
                    confidences.append(result.confidence)

            log.debug(
                f"Field extraction pass 1: "
                f"{list(extracted_data.keys()) or 'no data'}, "
                f"min_confidence={min(confidences) if confidences else 0.0:.2f}"
            )

        # --- Classification extractions ---
        if has_classification_configs:
            classification_data = self._execute_classification_extractions(
                current_state, user_message, instance, conversation_id
            )
            extracted_data.update(classification_data)

        # --- Multi-pass retry for missing required fields ---
        max_retries = current_state.extraction_retries
        if max_retries > 0:
            for retry_num in range(1, max_retries + 1):
                existing_context = instance.context.data

                # Find missing required field configs
                missing_field_configs = (
                    [
                        cfg
                        for cfg in all_configs
                        if cfg.required
                        and cfg.field_name not in extracted_data
                        and cfg.field_name not in existing_context
                    ]
                    if has_field_configs
                    else []
                )

                # Find missing required classification configs
                missing_class_configs = (
                    [
                        cfg
                        for cfg in (current_state.classification_extractions or [])
                        if cfg.required
                        and cfg.field_name not in extracted_data
                        and cfg.field_name not in existing_context
                    ]
                    if has_classification_configs
                    else []
                )

                if not missing_field_configs and not missing_class_configs:
                    break

                missing_names = [c.field_name for c in missing_field_configs] + [
                    c.field_name for c in missing_class_configs
                ]
                log.info(
                    f"Extraction retry {retry_num}/{max_retries}: "
                    f"missing={missing_names}"
                )

                if missing_field_configs:
                    retry_results = self._execute_field_extractions(
                        instance, user_message, missing_field_configs, conversation_id
                    )
                    for result in retry_results:
                        if result.is_valid and result.value is not None:
                            extracted_data[result.field_name] = result.value
                            confidences.append(result.confidence)

                if missing_class_configs:
                    retry_class_data = self._execute_classification_extractions(
                        current_state,
                        user_message,
                        instance,
                        conversation_id,
                        configs_override=missing_class_configs,
                    )
                    extracted_data.update(retry_class_data)

        # Build final response — check all sources for missing required fields
        all_required_names: list[str] = []
        if has_field_configs:
            all_required_names.extend(
                cfg.field_name for cfg in all_configs if cfg.required
            )
        if has_classification_configs:
            all_required_names.extend(
                cfg.field_name
                for cfg in (current_state.classification_extractions or [])
                if cfg.required
            )

        min_confidence = min(confidences) if confidences else 0.0
        response = DataExtractionResponse(
            extracted_data=extracted_data,
            confidence=min_confidence,
            additional_info_needed=any(
                name not in extracted_data and name not in instance.context.data
                for name in all_required_names
            ),
        )
        instance.last_extraction_response = response

        log.debug(
            f"Data extraction complete: "
            f"{list(extracted_data.keys())}, "
            f"confidence={min_confidence:.2f}"
        )
        return response

    def _execute_field_extractions(
        self,
        instance: FSMInstance,
        user_message: str,
        field_configs: list[FieldExtractionConfig],
        conversation_id: str,
    ) -> list[FieldExtractionResponse]:
        """Execute targeted field extractions for a list of configs.

        Runs one LLM call per field.  Each config specifies its own
        instructions, dynamic context selection, and validation rules.

        Previously extracted values are added to the dynamic context
        for subsequent extractions, enabling dependent field extraction
        (e.g., tool_input can see that tool_name was already extracted).
        """
        log = logger.bind(conversation_id=conversation_id)
        results: list[FieldExtractionResponse] = []
        # Accumulate extracted values so later fields can see earlier ones
        extracted_so_far: dict[str, Any] = {}

        for field_config in field_configs:
            log.debug(
                f"Extracting field '{field_config.field_name}' "
                f"(type={field_config.field_type})"
            )

            # Build dynamic context from config.context_keys
            if field_config.context_keys is not None:
                dynamic_context = {
                    k: v
                    for k, v in instance.context.data.items()
                    if k in field_config.context_keys
                }
            else:
                # Apply state-level context_scope as default filter
                current_state = self.get_state(instance, conversation_id)
                dynamic_context = self._apply_context_scope(
                    instance.context.get_user_visible_data(),
                    current_state,
                    conversation_id,
                )

            # Include previously extracted fields so the LLM can use them
            if extracted_so_far:
                dynamic_context.update(extracted_so_far)

            # Build prompt
            system_prompt = (
                self.field_extraction_prompt_builder.build_field_extraction_prompt(
                    instance=instance,
                    field_config=field_config,
                    user_message=user_message,
                    dynamic_context=dynamic_context,
                )
            )

            # Build request
            request = FieldExtractionRequest(
                system_prompt=system_prompt,
                user_message=user_message,
                field_name=field_config.field_name,
                field_type=field_config.field_type,
                context=dynamic_context,
                validation_rules=field_config.validation_rules,
            )

            # Call LLM
            try:
                # DECISION D-PIVOT-1-CALLSITE (step 11, plan_2026-04-27_32652286):
                # field-extraction site rewired through `oracle.invoke_field`
                # — direct passthrough to LLMInterface.extract_field that
                # preserves the legacy outer-envelope schema (distinct from
                # oracle._invoke_structured's D-008 bare-schema path).
                # Replaces the deferred-site marker at D-R10-7.2.
                # M4 — Program-owned Oracle field-read.
                response = self._oracle.invoke_field(request)
            except Exception as e:
                log.warning(
                    f"Field extraction failed for '{field_config.field_name}': {e}"
                )
                response = FieldExtractionResponse(
                    field_name=field_config.field_name,
                    value=None,
                    confidence=0.0,
                    is_valid=False,
                    validation_error=f"LLM call failed: {e}",
                )

            # Validate and coerce
            response = self._validate_field_extraction(response, field_config)

            log.debug(
                f"Field '{field_config.field_name}': "
                f"value={response.value!r}, confidence={response.confidence:.2f}, "
                f"valid={response.is_valid}"
            )
            results.append(response)

            # Feed successful extractions into context for subsequent fields
            if response.is_valid and response.value is not None:
                extracted_so_far[field_config.field_name] = response.value

        return results

    @staticmethod
    def _validate_field_extraction(
        response: FieldExtractionResponse,
        config: FieldExtractionConfig,
    ) -> FieldExtractionResponse:
        """Validate and type-coerce a field extraction response."""
        # Skip validation if already failed
        if not response.is_valid or response.value is None:
            return response

        # Reject values that are obviously the field name echoed back —
        # small models sometimes confuse the JSON template keys with values.
        if isinstance(response.value, str) and response.value.strip().lower() in (
            config.field_name.lower(),
            "field_name",
            "value",
        ):
            return FieldExtractionResponse(
                field_name=response.field_name,
                value=None,
                confidence=0.0,
                reasoning="Model echoed field name instead of extracting a value",
                is_valid=False,
                validation_error="Extracted value matches field name (model confusion)",
            )

        # Confidence threshold check
        if (
            config.confidence_threshold > 0.0
            and response.confidence < config.confidence_threshold
        ):
            return FieldExtractionResponse(
                field_name=response.field_name,
                value=response.value,
                confidence=response.confidence,
                reasoning=response.reasoning,
                is_valid=False,
                validation_error=(
                    f"Confidence {response.confidence:.2f} below threshold "
                    f"{config.confidence_threshold:.2f}"
                ),
            )

        # Type coercion via dispatch
        value = response.value
        try:
            coercer = _TYPE_COERCERS.get(config.field_type)
            if coercer is not None:
                value = coercer(value)
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            return FieldExtractionResponse(
                field_name=response.field_name,
                value=response.value,
                confidence=response.confidence,
                reasoning=response.reasoning,
                is_valid=False,
                validation_error=(f"Type coercion to {config.field_type} failed: {e}"),
            )

        # Validation rules
        rules = config.validation_rules or {}
        if "allowed_values" in rules:
            if value not in rules["allowed_values"]:
                return FieldExtractionResponse(
                    field_name=response.field_name,
                    value=value,
                    confidence=response.confidence,
                    reasoning=response.reasoning,
                    is_valid=False,
                    validation_error=(
                        f"Value {value!r} not in allowed values: "
                        f"{rules['allowed_values']}"
                    ),
                )

        if "min_length" in rules and isinstance(value, str):
            if len(value) < rules["min_length"]:
                return FieldExtractionResponse(
                    field_name=response.field_name,
                    value=value,
                    confidence=response.confidence,
                    reasoning=response.reasoning,
                    is_valid=False,
                    validation_error=(
                        f"Value length {len(value)} below minimum {rules['min_length']}"
                    ),
                )

        if "max_length" in rules and isinstance(value, str):
            if len(value) > rules["max_length"]:
                return FieldExtractionResponse(
                    field_name=response.field_name,
                    value=value,
                    confidence=response.confidence,
                    reasoning=response.reasoning,
                    is_valid=False,
                    validation_error=(
                        f"Value length {len(value)} exceeds maximum "
                        f"{rules['max_length']}"
                    ),
                )

        if "pattern" in rules and isinstance(value, str):
            import re

            try:
                if not re.match(rules["pattern"], value):
                    return FieldExtractionResponse(
                        field_name=response.field_name,
                        value=value,
                        confidence=response.confidence,
                        reasoning=response.reasoning,
                        is_valid=False,
                        validation_error=(
                            f"Value does not match pattern: {rules['pattern']}"
                        ),
                    )
            except re.error as e:
                logger.error(f"Invalid regex pattern {rules['pattern']!r}: {e}")
                return FieldExtractionResponse(
                    field_name=response.field_name,
                    value=value,
                    confidence=0.0,
                    reasoning=f"Invalid regex pattern: {e}",
                    is_valid=False,
                    validation_error=f"Invalid regex pattern: {e}",
                )

        # All checks passed — return with coerced value
        return FieldExtractionResponse(
            field_name=response.field_name,
            value=value,
            confidence=response.confidence,
            reasoning=response.reasoning,
            is_valid=True,
        )

    # ----------------------------------------------------------
    # Classification-based extraction
    # ----------------------------------------------------------

    def _execute_classification_extractions(
        self,
        current_state: State,
        user_message: str,
        instance: FSMInstance,
        conversation_id: str,
        *,
        configs_override: list[ClassificationExtractionConfig] | None = None,
    ) -> dict[str, Any]:
        """Run classification extractions and return extracted data.

        For each :class:`ClassificationExtractionConfig`, builds a
        :class:`ClassificationSchema`, creates a :class:`Classifier`,
        and stores the result in two context keys:

        - ``field_name`` → intent string (simple, JsonLogic-friendly)
        - ``_{field_name}_classification`` → full result dict (debugging)

        Args:
            current_state: Current state (for config lookup).
            user_message: User input to classify.
            instance: FSM instance (for model fallback).
            conversation_id: Logging context.
            configs_override: If provided, run only these configs
                (used during retry).

        Returns:
            Dict of extracted key-value pairs to merge into context.
        """
        log = logger.bind(conversation_id=conversation_id)
        configs = configs_override or current_state.classification_extractions or []
        if not configs:
            return {}

        model = getattr(self.llm_interface, "model", None)
        extracted: dict[str, Any] = {}

        for config in configs:
            effective_model = config.model or model
            if not effective_model:
                if config.required:
                    raise ClassificationError(
                        f"Required classification extraction '{config.field_name}': "
                        "no LLM model available"
                    )
                log.warning(
                    f"Classification extraction '{config.field_name}': "
                    "no LLM model available, skipping"
                )
                continue

            try:
                schema = ClassificationSchema(
                    intents=config.intents,
                    fallback_intent=config.fallback_intent,
                    confidence_threshold=config.confidence_threshold,
                )

                prompt_config = None
                if config.prompt_config:
                    prompt_config = ClassificationPromptConfig(**config.prompt_config)

                classifier = Classifier(
                    schema=schema,
                    model=effective_model,
                    config=prompt_config,
                )

                result: ClassificationResult = classifier.classify(user_message)

                log.debug(
                    f"Classification extraction '{config.field_name}': "
                    f"intent={result.intent}, confidence={result.confidence:.2f}"
                )

                # Always store fallback intent so the context key exists
                # for downstream JsonLogic conditions
                if result.intent == config.fallback_intent:
                    extracted[config.field_name] = result.intent
                    log.debug(
                        f"Classification extraction '{config.field_name}': "
                        f"fallback intent '{result.intent}' stored"
                    )
                    continue

                # Skip low confidence
                if result.confidence < config.confidence_threshold:
                    log.debug(
                        f"Classification extraction '{config.field_name}': "
                        f"confidence {result.confidence:.2f} below threshold "
                        f"{config.confidence_threshold}, skipping"
                    )
                    continue

                # Store simple value (user-visible, works with JsonLogic)
                extracted[config.field_name] = result.intent

                # Store full result (internal key, debugging)
                suffix = CLASSIFICATION_EXTRACTION_RESULT_SUFFIX
                full_key = f"_{config.field_name}{suffix}"
                full_result: dict[str, Any] = {
                    "intent": result.intent,
                    "confidence": result.confidence,
                    "reasoning": result.reasoning,
                    "entities": result.entities,
                }
                # Include context snapshot if configured
                if config.context_keys:
                    full_result["context_snapshot"] = {
                        k: instance.context.data.get(k)
                        for k in config.context_keys
                        if k in instance.context.data
                    }
                extracted[full_key] = full_result

                log.info(
                    f"Classification extraction '{config.field_name}' = "
                    f"'{result.intent}' (confidence={result.confidence:.2f})"
                )

            except (
                ClassificationError,
                ValueError,
                TypeError,
                KeyError,
                RuntimeError,
                OSError,
            ) as e:
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
