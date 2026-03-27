from __future__ import annotations

"""
MessagePipeline: The 2-pass message processing engine.

Encapsulates all LLM-driven processing logic extracted from FSMManager:
- Pass 1: Data extraction + transition evaluation + state transition
- Pass 2: Response generation from final state
- Handler execution bridge (deep-copy context, merge deltas)

FSMManager delegates to this class for all message processing.
The pipeline does not own instances or locks — those remain in FSMManager.
"""

import copy
import json
import time
from collections.abc import Callable
from typing import Any

from .classification import Classifier
from .constants import (
    CLASSIFICATION_EXTRACTION_RESULT_SUFFIX,
    CONTEXT_KEY_CLASSIFICATION_RESULT,
    DEFAULT_TRANSITION_CLASSIFICATION_CONFIDENCE,
    TRANSITION_CLASSIFICATION_FALLBACK_INTENT,
)
from .context import clean_context_keys
from .definitions import (
    ClassificationError,
    ClassificationExtractionConfig,
    ClassificationResult,
    ClassificationSchema,
    DataExtractionResponse,
    FieldExtractionConfig,
    FieldExtractionRequest,
    FieldExtractionResponse,
    FSMDefinition,
    FSMInstance,
    IntentDefinition,
    InvalidTransitionError,
    ResponseGenerationRequest,
    State,
    StateNotFoundError,
    TransitionEvaluation,
    TransitionEvaluationResult,
    TransitionOption,
)
from .handlers import HandlerSystem, HandlerTiming
from .llm import LLMInterface
from .logging import logger
from .prompts import (
    ClassificationPromptConfig,
    DataExtractionPromptBuilder,
    FieldExtractionPromptBuilder,
    ResponseGenerationPromptBuilder,
)
from .transition_evaluator import TransitionEvaluator


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


def _coerce_list(v: Any) -> list:
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        parsed = json.loads(v)
        if not isinstance(parsed, list):
            raise TypeError("not a list")
        return parsed
    return v


def _coerce_dict(v: Any) -> dict:
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


class MessagePipeline:
    """2-pass message processing pipeline.

    Handles data extraction, transition evaluation, state transitions,
    response generation, and handler execution. Stateless with respect
    to conversation instances — all state is passed as parameters.
    """

    def __init__(
        self,
        llm_interface: LLMInterface,
        data_extraction_prompt_builder: DataExtractionPromptBuilder,
        response_generation_prompt_builder: ResponseGenerationPromptBuilder,
        transition_evaluator: TransitionEvaluator,
        handler_system: HandlerSystem,
        fsm_resolver: Callable[[str], FSMDefinition],
        field_extraction_prompt_builder: FieldExtractionPromptBuilder | None = None,
    ):
        self.llm_interface = llm_interface
        self.data_extraction_prompt_builder = data_extraction_prompt_builder
        self.response_generation_prompt_builder = response_generation_prompt_builder
        self.transition_evaluator = transition_evaluator
        self.handler_system = handler_system
        self.fsm_resolver = fsm_resolver
        self.field_extraction_prompt_builder = (
            field_extraction_prompt_builder or FieldExtractionPromptBuilder()
        )

    def get_state(
        self, instance: FSMInstance, conversation_id: str | None = None
    ) -> State:
        """Resolve current State from FSM definition."""
        log = (
            logger.bind(conversation_id=conversation_id) if conversation_id else logger
        )

        fsm_def = self.fsm_resolver(instance.fsm_id)
        if instance.current_state not in fsm_def.states:
            error_msg = (
                f"State '{instance.current_state}' not found in FSM '{instance.fsm_id}'"
            )
            log.error(error_msg)
            raise StateNotFoundError(error_msg)

        return fsm_def.states[instance.current_state]

    # ----------------------------------------------------------
    # Handler execution bridge
    # ----------------------------------------------------------

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
        """Execute handlers at specified timing point.

        Deep-copies instance context before passing to handlers, then merges
        the delta dict back into the instance. A handler returning a key with
        value ``None`` requests deletion of that key.
        """
        context = copy.deepcopy(instance.context.data)

        if error_context:
            context.update(error_context)

        try:
            updated_context = self.handler_system.execute_handlers(
                timing=timing,
                current_state=current_state or instance.current_state,
                target_state=target_state,
                context=context,
                updated_keys=updated_keys,
            )

            if updated_context:
                for key, value in updated_context.items():
                    if value is None:
                        instance.context.data.pop(key, None)
                    else:
                        instance.context.data[key] = value

        except Exception as e:
            logger.error(f"Handler execution error at {timing.name}: {e!s}")
            if self.handler_system.error_mode == "raise":
                raise

    # ----------------------------------------------------------
    # Full 2-pass processing
    # ----------------------------------------------------------

    def process(self, instance: FSMInstance, message: str, conversation_id: str) -> str:
        """Execute the full 2-pass message processing pipeline.

        Pass 1: PRE_PROCESSING handlers → data extraction → context update →
                transition evaluation → state transition
        Pass 2: POST_PROCESSING handlers → response generation

        Args:
            instance: The FSM instance (already validated as non-terminal).
            message: User message to process.
            conversation_id: Conversation identifier.

        Returns:
            Generated response message.
        """
        # Contextualize propagates conversation_id to all downstream logger
        # calls on this thread (llm.py, transition_evaluator.py, etc.)
        with logger.contextualize(conversation_id=conversation_id, package="fsm_llm"):
            # Execute pre-processing handlers
            self.execute_handlers(
                instance,
                HandlerTiming.PRE_PROCESSING,
                conversation_id,
                current_state=instance.current_state,
            )

            # Pass 1: Data extraction + transition evaluation + execution
            extraction_response, transition_occurred, previous_state = (
                self._execute_extraction_and_transition_pass(
                    instance, message, conversation_id
                )
            )

            # Execute post-processing handlers (after potential transition)
            self.execute_handlers(
                instance,
                HandlerTiming.POST_PROCESSING,
                conversation_id,
                current_state=instance.current_state,
            )

            # Pass 2: Response generation based on final state
            return self._execute_response_generation_pass(
                instance,
                message,
                extraction_response,
                transition_occurred,
                previous_state,
                conversation_id,
            )

    # ----------------------------------------------------------
    # Initial response generation
    # ----------------------------------------------------------

    def generate_initial_response(
        self, instance: FSMInstance, conversation_id: str
    ) -> str:
        """Generate initial response for conversation start (no extraction/transition)."""
        log = logger.bind(conversation_id=conversation_id)

        current_state = self.get_state(instance, conversation_id)
        fsm_def = self.fsm_resolver(instance.fsm_id)

        system_prompt = self.response_generation_prompt_builder.build_response_prompt(
            instance=instance,
            state=current_state,
            fsm_definition=fsm_def,
            extracted_data={},
            transition_occurred=False,
            previous_state=None,
            user_message="",
        )

        request = ResponseGenerationRequest(
            system_prompt=system_prompt,
            user_message="",
            extracted_data={},
            context=instance.context.get_user_visible_data(),
            transition_occurred=False,
            previous_state=None,
        )

        response = self.llm_interface.generate_response(request)
        instance.last_response_generation = response
        instance.context.conversation.add_system_message(response.message)

        log.info("Generated initial response")
        return response.message

    # ----------------------------------------------------------
    # Pass 1: Data extraction + transition
    # ----------------------------------------------------------

    def _execute_extraction_and_transition_pass(
        self, instance: FSMInstance, user_message: str, conversation_id: str
    ) -> tuple[DataExtractionResponse, bool, str | None]:
        """Execute Pass 1: Data Extraction + Transition Evaluation + Execution."""
        log = logger.bind(conversation_id=conversation_id)
        log.debug("Executing data extraction and transition pass")

        # Step 1: Unified field-based extraction (auto-converts legacy
        # required_context_keys and merges with explicit field_extractions)
        extraction_response = self._execute_data_extraction(
            instance, user_message, conversation_id
        )

        # Step 2: Update context with extracted data
        if extraction_response.extracted_data:
            extraction_response.extracted_data = self._clean_empty_context_keys(
                data=extraction_response.extracted_data, conversation_id=conversation_id
            )

            if extraction_response.extracted_data:
                instance.context.update(extraction_response.extracted_data)

                # Notify handlers about context updates
                self.execute_handlers(
                    instance,
                    HandlerTiming.CONTEXT_UPDATE,
                    conversation_id,
                    current_state=instance.current_state,
                    updated_keys=set(extraction_response.extracted_data.keys()),
                )

        # Step 3: Transition Evaluation and Execution
        transition_occurred, previous_state = (
            self._execute_transition_evaluation_and_execution(
                instance, user_message, extraction_response, conversation_id
            )
        )

        log.debug("Data extraction and transition pass completed")
        return extraction_response, transition_occurred, previous_state

    # ----------------------------------------------------------
    # Pass 1: Field-based extraction (replaces bulk extract_data)
    # ----------------------------------------------------------

    @staticmethod
    def _build_field_configs_from_state(state: State) -> list[FieldExtractionConfig]:
        """Auto-convert legacy state fields to FieldExtractionConfig list.

        Translates ``required_context_keys`` + ``extraction_instructions``
        into per-field configs so the pipeline can use the unified
        ``extract_field`` primitive for all extraction.  Explicit
        ``field_extractions`` on the state are appended after the
        auto-generated ones.
        """
        configs: list[FieldExtractionConfig] = []

        # Auto-convert required_context_keys → one config per key
        if state.required_context_keys:
            instructions = (
                state.extraction_instructions
                or "Extract the value of this field from the user's input."
            )
            for key in state.required_context_keys:
                configs.append(
                    FieldExtractionConfig(
                        field_name=key,
                        field_type="any",
                        extraction_instructions=(
                            f"Extract the '{key}' field. {instructions}"
                        ),
                        context_keys=None,  # all context
                        required=True,
                        confidence_threshold=state.extraction_confidence_threshold,
                    )
                )

        # Append explicit field_extractions (user-defined, take priority)
        if state.field_extractions:
            # Avoid duplicates: explicit configs override auto-generated ones
            explicit_names = {fc.field_name for fc in state.field_extractions}
            configs = [c for c in configs if c.field_name not in explicit_names]
            configs.extend(state.field_extractions)

        return configs

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

        if not has_field_configs and not has_classification_configs:
            log.debug("No fields or classifications to extract for this state")
            response = DataExtractionResponse(extracted_data={}, confidence=1.0)
            instance.last_extraction_response = response
            return response

        extracted_data: dict[str, Any] = {}
        confidences: list[float] = []

        # --- Field extractions ---
        if has_field_configs:
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
        """
        log = logger.bind(conversation_id=conversation_id)
        results: list[FieldExtractionResponse] = []

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
                response = self.llm_interface.extract_field(request)
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

        # All checks passed — return with coerced value
        return FieldExtractionResponse(
            field_name=response.field_name,
            value=value,
            confidence=response.confidence,
            reasoning=response.reasoning,
            is_valid=True,
        )

    # ----------------------------------------------------------
    # Pass 1: Transition evaluation and execution
    # ----------------------------------------------------------

    def _execute_transition_evaluation_and_execution(
        self,
        instance: FSMInstance,
        user_message: str,
        extraction_response: DataExtractionResponse,
        conversation_id: str,
    ) -> tuple[bool, str | None]:
        """Evaluate transitions and execute if one is selected."""
        log = logger.bind(conversation_id=conversation_id)
        log.debug("Executing transition evaluation and execution")

        current_state = self.get_state(instance, conversation_id)

        if not current_state.transitions:
            log.debug("Terminal state reached - no transitions to evaluate")
            return False, None

        previous_state_id = instance.current_state

        evaluation = self.transition_evaluator.evaluate_transitions(
            current_state, instance.context, extraction_response.extracted_data
        )

        target_state = None

        if evaluation.result_type == TransitionEvaluationResult.DETERMINISTIC:
            target_state = evaluation.deterministic_transition
            log.info(f"Deterministic transition selected: {target_state}")

        elif evaluation.result_type == TransitionEvaluationResult.AMBIGUOUS:
            target_state = self._resolve_ambiguous_transition(
                evaluation, user_message, extraction_response, instance, conversation_id
            )
            log.info(f"LLM-assisted transition selected: {target_state}")

        elif evaluation.result_type == TransitionEvaluationResult.BLOCKED:
            log.warning(f"Transitions blocked: {evaluation.blocked_reason}")
            return False, None

        if target_state:
            self._execute_state_transition(instance, target_state, conversation_id)
            return True, previous_state_id

        return False, None

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

                # Skip fallback intent
                if result.intent == config.fallback_intent:
                    log.debug(
                        f"Classification extraction '{config.field_name}': "
                        "returned fallback intent, skipping"
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
                ClassificationError, ValueError, TypeError, KeyError,
                RuntimeError, OSError,
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

    def _resolve_ambiguous_transition(
        self,
        evaluation: TransitionEvaluation,
        user_message: str,
        extraction_response: DataExtractionResponse,
        instance: FSMInstance,
        conversation_id: str,
    ) -> str:
        """Resolve ambiguous transition using classification.

        Classification is always-on for ambiguous transitions. Builds a
        ClassificationSchema from available transition options and uses
        the Classifier to make a structured, confidence-scored decision.
        """
        log = logger.bind(conversation_id=conversation_id)
        log.debug(
            f"Resolving ambiguous transition with {len(evaluation.available_options)} options"
        )

        current_state = self.get_state(instance, conversation_id)

        schema = self._build_transition_classification_schema(
            current_state,
            evaluation.available_options,
        )

        model = getattr(self.llm_interface, "model", None)
        if model is None:
            raise InvalidTransitionError(
                "Cannot determine LLM model for classification-based "
                "transition resolution"
            )

        classifier = Classifier(
            schema=schema,
            model=model,
        )

        try:
            result: ClassificationResult = classifier.classify(user_message)
        except (ClassificationError, Exception) as e:
            log.warning(
                f"Classification failed during ambiguous transition resolution: {e}"
            )
            log.warning("Falling back to current state (no transition)")
            instance.context.data[CONTEXT_KEY_CLASSIFICATION_RESULT] = {
                "error": str(e),
                "fallback": True,
            }
            return instance.current_state

        log.debug(
            f"Classification result: intent={result.intent}, "
            f"confidence={result.confidence:.2f}, reasoning={result.reasoning}"
        )

        # Store classification result in context for debugging
        instance.context.data[CONTEXT_KEY_CLASSIFICATION_RESULT] = {
            "intent": result.intent,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
            "entities": result.entities,
        }

        # Store as transition decision for debugging
        instance.last_transition_decision = result

        # Handle fallback intent (low confidence or unknown) — stay in current state
        if result.intent == TRANSITION_CLASSIFICATION_FALLBACK_INTENT:
            log.info(
                "Classification returned fallback intent — staying in current state"
            )
            return instance.current_state

        # Validate the classified intent is a valid target state
        valid_targets = {opt.target_state for opt in evaluation.available_options}
        if result.intent not in valid_targets:
            raise InvalidTransitionError(
                f"Classification returned unknown target '{result.intent}'. "
                f"Valid options: {sorted(valid_targets)}"
            )

        log.info(
            f"Classification-based transition selected: {result.intent} "
            f"(confidence={result.confidence:.2f})"
        )
        return result.intent

    # ----------------------------------------------------------
    # Classification schema builder
    # ----------------------------------------------------------

    @staticmethod
    def _build_transition_classification_schema(
        state: State,
        options: list[TransitionOption],
    ) -> ClassificationSchema:
        """Build a ClassificationSchema from transition options.

        If the state has a custom ``transition_classification`` dict config,
        merges user-provided descriptions and thresholds. Otherwise
        auto-generates intents from transition descriptions.
        """
        config = state.transition_classification

        if isinstance(config, dict):
            # Manual mode: user provides intent descriptions
            intents = []
            for opt in options:
                custom = config.get(opt.target_state, {})
                description = (
                    custom.get("description")
                    or opt.description
                    or f"Transition to {opt.target_state}"
                )
                intents.append(
                    IntentDefinition(name=opt.target_state, description=description)
                )
            confidence_threshold = config.get(
                "confidence_threshold",
                DEFAULT_TRANSITION_CLASSIFICATION_CONFIDENCE,
            )
        else:
            # Auto mode: generate from transition option descriptions
            intents = []
            for opt in options:
                description = opt.description or f"Transition to {opt.target_state}"
                intents.append(
                    IntentDefinition(name=opt.target_state, description=description)
                )
            confidence_threshold = DEFAULT_TRANSITION_CLASSIFICATION_CONFIDENCE

        # Add fallback intent for low-confidence cases
        intents.append(
            IntentDefinition(
                name=TRANSITION_CLASSIFICATION_FALLBACK_INTENT,
                description="None of the above options clearly match the user's intent",
            )
        )

        return ClassificationSchema(
            intents=intents,
            fallback_intent=TRANSITION_CLASSIFICATION_FALLBACK_INTENT,
            confidence_threshold=confidence_threshold,
        )

    def _execute_state_transition(
        self, instance: FSMInstance, target_state: str, conversation_id: str
    ) -> None:
        """Execute state transition with PRE/POST handler integration and rollback."""
        log = logger.bind(conversation_id=conversation_id)
        old_state = instance.current_state

        self.execute_handlers(
            instance,
            HandlerTiming.PRE_TRANSITION,
            conversation_id,
            current_state=old_state,
            target_state=target_state,
        )

        # Deep-copy full context for rollback if POST_TRANSITION handlers fail
        old_context_snapshot = copy.deepcopy(instance.context.data)

        instance.current_state = target_state
        instance.context.data.update(
            {
                "_previous_state": old_state,
                "_current_state": target_state,
                "_transition_timestamp": time.time(),
            }
        )

        try:
            self.execute_handlers(
                instance,
                HandlerTiming.POST_TRANSITION,
                conversation_id,
                current_state=target_state,
                target_state=target_state,
            )
        except Exception as handler_err:
            log.warning(
                f"POST_TRANSITION handler failed ({type(handler_err).__name__}: {handler_err}), rolling back state from {target_state} to {old_state}"
            )
            instance.current_state = old_state
            if old_context_snapshot is not None:
                instance.context.data.clear()
                instance.context.data.update(old_context_snapshot)
            else:
                log.error("Rollback snapshot was None, cannot safely restore context")
            raise

        log.info(f"State transition executed: {old_state} -> {target_state}")

    # ----------------------------------------------------------
    # Pass 2: Response generation
    # ----------------------------------------------------------

    def _execute_response_generation_pass(
        self,
        instance: FSMInstance,
        user_message: str,
        extraction_response: DataExtractionResponse,
        transition_occurred: bool,
        previous_state: str | None,
        conversation_id: str,
    ) -> str:
        """Execute Pass 2: Response Generation based on final state."""
        log = logger.bind(conversation_id=conversation_id)
        log.debug("Executing response generation pass")

        current_state = self.get_state(instance, conversation_id)
        fsm_def = self.fsm_resolver(instance.fsm_id)

        system_prompt = self.response_generation_prompt_builder.build_response_prompt(
            instance=instance,
            state=current_state,
            fsm_definition=fsm_def,
            extracted_data=extraction_response.extracted_data,
            transition_occurred=transition_occurred,
            previous_state=previous_state,
            user_message=user_message,
        )

        # Apply context scoping if the state defines read_keys
        context_for_llm = self._apply_context_scope(
            instance.context.get_user_visible_data(),
            current_state,
            conversation_id,
        )

        request = ResponseGenerationRequest(
            system_prompt=system_prompt,
            user_message=user_message,
            extracted_data=extraction_response.extracted_data,
            context=context_for_llm,
            transition_occurred=transition_occurred,
            previous_state=previous_state,
        )

        response = self.llm_interface.generate_response(request)
        instance.last_response_generation = response
        instance.context.conversation.add_system_message(response.message)

        log.debug("Response generation pass completed")
        return response.message

    # ----------------------------------------------------------
    # Utilities
    # ----------------------------------------------------------

    @staticmethod
    def _apply_context_scope(
        context: dict[str, Any],
        state: State,
        conversation_id: str,
    ) -> dict[str, Any]:
        """Filter context by state's context_scope if defined.

        If the state has ``context_scope`` with ``read_keys``, returns
        only the keys listed. Missing keys are silently skipped (states
        may be entered before all keys are populated).

        If ``context_scope`` is ``None``, returns the full context
        unchanged (backward-compatible default).
        """
        if state.context_scope is None:
            return context

        read_keys = state.context_scope.get("read_keys")
        if not read_keys:
            return context

        scoped = {k: v for k, v in context.items() if k in read_keys}
        missing = [k for k in read_keys if k not in context]
        if missing:
            log = logger.bind(conversation_id=conversation_id)
            log.debug(
                f"Context scope: state '{state.id}' requested keys "
                f"{missing} but they are not in context"
            )
        return scoped

    @staticmethod
    def _clean_empty_context_keys(
        data: dict[str, Any], conversation_id: str, remove_none_values: bool = True
    ) -> dict[str, Any]:
        """Clean invalid keys from context data. Delegates to context module."""
        return clean_context_keys(data, conversation_id, remove_none_values)
