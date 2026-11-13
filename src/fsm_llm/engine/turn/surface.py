"""The declarations the mixins of ``MessagePipeline`` share (mypy only).

Layer ``engine`` (plan d454fd62, step 79): split out of the old ``fsm_llm.pipeline``; the
method and function bodies are unchanged.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator, Mapping
from typing import TYPE_CHECKING, Any

from fsm_llm.classification import Classifier
from fsm_llm.context.layout import (
    DataExtractionPromptBuilder,
    FieldExtractionPromptBuilder,
    ResponseGenerationPromptBuilder,
)
from fsm_llm.core.definitions import (
    ClassificationExtractionConfig,
    ClassificationPromptConfig,
    ClassificationSchema,
    DataExtractionResponse,
    FieldExtractionConfig,
    FieldExtractionResponse,
    FSMDefinition,
    FSMInstance,
    State,
)
from fsm_llm.core.transition_evaluator import TransitionEvaluator
from fsm_llm.executors import StateExecutor
from fsm_llm.handlers import HandlerSystem, HandlerTiming
from fsm_llm.llm import LLMInterface

from .records import _TurnRecord


class _PipelineSurface:
    """What every mixin of ``MessagePipeline`` may rely on: the attributes of the shell and the
    methods of the other mixins, declared for mypy only (a runtime no-op). A mixin that defines
    one of these is checked against the declaration here.
    """

    if TYPE_CHECKING:
        llm_interface: LLMInterface
        data_extraction_prompt_builder: DataExtractionPromptBuilder
        response_generation_prompt_builder: ResponseGenerationPromptBuilder
        transition_evaluator: TransitionEvaluator
        handler_system: HandlerSystem
        fsm_resolver: Callable[[str], FSMDefinition]
        field_extraction_prompt_builder: FieldExtractionPromptBuilder
        _classifier_cache: dict[str, Classifier]
        _classifier_cache_lock: threading.Lock
        _executors: Mapping[str, StateExecutor]

        @staticmethod
        def _apply_context_scope(
            context: dict[str, Any],
            state: State,
            conversation_id: str,
        ) -> dict[str, Any]: ...

        def _build_classifier_context(
            self,
            instance: FSMInstance,
            state: State,
            conversation_id: str,
            context_keys: list[str] | None = None,
            *,
            user_message: str | None,
        ) -> dict[str, Any]: ...

        @staticmethod
        def _build_field_configs_from_state(
            state: State,
        ) -> list[FieldExtractionConfig]: ...

        def _bulk_extract_from_instructions(
            self,
            instance: FSMInstance,
            user_message: str | None,
            state: State,
            conversation_id: str,
        ) -> dict[str, Any]: ...

        def _execute_classification_extractions(
            self,
            current_state: State,
            user_message: str | None,
            instance: FSMInstance,
            conversation_id: str,
            *,
            configs_override: list[ClassificationExtractionConfig] | None = None,
        ) -> dict[str, Any]: ...

        def _execute_data_extraction(
            self, instance: FSMInstance, user_message: str | None, conversation_id: str
        ) -> DataExtractionResponse: ...

        def _execute_extraction_and_transition_pass(
            self,
            instance: FSMInstance,
            user_message: str | None,
            conversation_id: str,
            *,
            turn: _TurnRecord | None = None,
        ) -> tuple[DataExtractionResponse, bool, str | None]: ...

        def _execute_field_extractions(
            self,
            instance: FSMInstance,
            user_message: str | None,
            field_configs: list[FieldExtractionConfig],
            conversation_id: str,
            memo: dict[tuple[str, str], FieldExtractionResponse] | None = None,
        ) -> list[FieldExtractionResponse]: ...

        def _execute_response_generation_pass(
            self,
            instance: FSMInstance,
            user_message: str | None,
            extraction_response: DataExtractionResponse,
            transition_occurred: bool,
            previous_state: str | None,
            conversation_id: str,
        ) -> str | None: ...

        def _execute_transition_evaluation_and_execution(
            self,
            instance: FSMInstance,
            user_message: str | None,
            conversation_id: str,
            *,
            turn: _TurnRecord | None = None,
        ) -> tuple[bool, str | None]: ...

        def _get_classifier(
            self,
            schema: ClassificationSchema,
            prompt_config: ClassificationPromptConfig | None,
            *,
            model: str | None = None,
        ) -> Classifier: ...

        def _handler_only_keys(self, instance: FSMInstance) -> frozenset[str]: ...

        def _stream_response_generation_pass(
            self,
            instance: FSMInstance,
            user_message: str | None,
            extraction_response: DataExtractionResponse,
            transition_occurred: bool,
            previous_state: str | None,
            conversation_id: str,
        ) -> Iterator[str]: ...

        @staticmethod
        def _validate_field_extraction(
            response: FieldExtractionResponse,
            config: FieldExtractionConfig,
        ) -> FieldExtractionResponse: ...

        def execute_handlers(
            self,
            instance: FSMInstance,
            timing: HandlerTiming,
            *,
            current_state: str | None = None,
            target_state: str | None = None,
            updated_keys: set[str] | None = None,
            error_context: dict[str, Any] | None = None,
        ) -> None: ...

        def get_state(
            self, instance: FSMInstance, conversation_id: str | None = None
        ) -> State: ...
