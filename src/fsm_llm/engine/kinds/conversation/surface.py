"""The declarations the mixins of ``ConversationPass1`` share (mypy only).

Layer ``engine`` (plan d454fd62, step 80, D-030): the conversation half of the old
``engine/turn/surface.py``. A class whose body is one ``if TYPE_CHECKING:`` block: the
``_services`` attribute and the methods that one conversation mixin calls on another, so each mixin
is checked against the declarations (a runtime no-op, the base is empty).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fsm_llm.core.definitions import (
    ClassificationExtractionConfig,
    DataExtractionResponse,
    FieldExtractionConfig,
    FieldExtractionResponse,
    FSMInstance,
    State,
)

from ..services import _PipelineServices


class _ConversationSurface:
    """What every mixin of the conversation Pass-1 machinery may rely on: the services of the
    pipeline it is bound to and the methods of the other mixins, declared for mypy only.
    """

    if TYPE_CHECKING:
        _services: _PipelineServices

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

        def _classification_configs(
            self, state: State, instance: FSMInstance
        ) -> list[ClassificationExtractionConfig]: ...

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

        def _execute_field_extractions(
            self,
            instance: FSMInstance,
            user_message: str | None,
            field_configs: list[FieldExtractionConfig],
            conversation_id: str,
            memo: dict[tuple[str, str], FieldExtractionResponse] | None = None,
        ) -> list[FieldExtractionResponse]: ...

        def _handler_only_keys(self, instance: FSMInstance) -> frozenset[str]: ...

        @staticmethod
        def _validate_field_extraction(
            response: FieldExtractionResponse,
            config: FieldExtractionConfig,
        ) -> FieldExtractionResponse: ...
