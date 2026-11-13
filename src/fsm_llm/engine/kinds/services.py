"""What an executor may ask of the turn pipeline: ``_PipelineServices`` (a ``Protocol``).

Layer ``engine`` (plan d454fd62, step 80, D-030). The kind executors live BELOW
``engine/turn/`` (the pipeline imports them, never the reverse), so an executor cannot name
``MessagePipeline``. It receives the pipeline as the ``pipeline`` argument of ``extract`` and
``after_transition`` and reads exactly the members listed here, nothing else. Pass 2 and the
transition code share the same members with the conversation kind; they stay on the pipeline.

Interface contract: this is a structural type, never instantiated and never imported at run
time by ``engine/turn/`` (mypy checks that ``MessagePipeline`` satisfies it, because the
turn calls ``executor.extract(self, ...)``). A member is added here only when an executor
reads it; a member nothing reads is removed. ``_apply_context_scope`` is a static method of the
pipeline (Pass 2 owns it); the others are the pipeline's attributes and methods of the same
name.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from fsm_llm.classification import Classifier
from fsm_llm.context.layout import (
    DataExtractionPromptBuilder,
    FieldExtractionPromptBuilder,
)
from fsm_llm.core.definitions import (
    ClassificationPromptConfig,
    ClassificationSchema,
    FSMDefinition,
    FSMInstance,
    State,
)
from fsm_llm.handlers import HandlerTiming
from fsm_llm.llm import LLMInterface


class _PipelineServices(Protocol):
    """The services of the turn pipeline that the kind executors use."""

    llm_interface: LLMInterface
    data_extraction_prompt_builder: DataExtractionPromptBuilder
    field_extraction_prompt_builder: FieldExtractionPromptBuilder
    fsm_resolver: Callable[[str], FSMDefinition]

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

    def _get_classifier(
        self,
        schema: ClassificationSchema,
        prompt_config: ClassificationPromptConfig | None,
        *,
        model: str | None = None,
    ) -> Classifier: ...

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
