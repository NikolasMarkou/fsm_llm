"""The declarations the mixins of ``MessagePipeline`` share (mypy only).

Layer ``engine`` (plan d454fd62, step 79): split out of the old ``fsm_llm.pipeline``; the
method and function bodies are unchanged. Step 80 moved the declarations of the conversation
Pass-1 methods to ``engine/kinds/conversation/surface.py``; what is left is what the turn mixins
share, which is also what ``engine/kinds/services.py`` (``_PipelineServices``) needs of this class.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator, Mapping
from typing import TYPE_CHECKING, Any

from fsm_llm.context.layout import (
    DataExtractionPromptBuilder,
    FieldExtractionPromptBuilder,
    ResponseGenerationPromptBuilder,
)
from fsm_llm.core.definitions import (
    ClassificationPromptConfig,
    ClassificationSchema,
    DataExtractionResponse,
    FSMDefinition,
    FSMInstance,
    State,
)
from fsm_llm.core.transition_evaluator import TransitionEvaluator
from fsm_llm.engine.classification import Classifier
from fsm_llm.engine.kinds import StateExecutor
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

        def _execute_extraction_and_transition_pass(
            self,
            instance: FSMInstance,
            user_message: str | None,
            conversation_id: str,
            *,
            turn: _TurnRecord | None = None,
        ) -> tuple[DataExtractionResponse, bool, str | None]: ...

        async def _aexecute_extraction_and_transition_pass(
            self,
            instance: FSMInstance,
            user_message: str | None,
            conversation_id: str,
            *,
            turn: _TurnRecord | None = None,
        ) -> tuple[DataExtractionResponse, bool, str | None]: ...

        def _execute_response_generation_pass(
            self,
            instance: FSMInstance,
            user_message: str | None,
            extraction_response: DataExtractionResponse,
            transition_occurred: bool,
            previous_state: str | None,
            conversation_id: str,
        ) -> str | None: ...

        def _get_classifier(
            self,
            schema: ClassificationSchema,
            prompt_config: ClassificationPromptConfig | None,
            *,
            model: str | None = None,
        ) -> Classifier: ...

        def _stream_response_generation_pass(
            self,
            instance: FSMInstance,
            user_message: str | None,
            extraction_response: DataExtractionResponse,
            transition_occurred: bool,
            previous_state: str | None,
            conversation_id: str,
        ) -> Iterator[str]: ...

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
