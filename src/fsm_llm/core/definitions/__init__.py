"""
Enhanced FSM-LLM definitions with the 2-pass architecture.

This package defines the core data structures for a refined 2-pass FSM-LLM system:
1. Pass 1: Data extraction + transition evaluation
2. Pass 2: Response generation based on final state

Key Changes:
- Separate data extraction from response generation
- Response generation occurs after transition evaluation
- Enhanced request/response models for each pass

The package is the one import surface of the layer (`from fsm_llm.core.definitions import
State`); each name is defined in the module named in `module_path_map.json` and re-exported
here, so a patch or a private-name import aims at the defining module.
"""

from __future__ import annotations

from .classification import (
    ClassificationExtractionConfig,
    ClassificationPromptConfig,
    ClassificationResult,
    ClassificationSchema,
    HierarchicalResult,
    HierarchicalSchema,
    IntentDefinition,
    IntentScore,
    MultiClassificationResult,
)
from .completion import (
    CONTEXT_KEY_CUT_OFF,
    FINISH_REASON_LENGTH,
    CompletionRequest,
    CompletionResponse,
    LLMCallCounts,
    LLMUsage,
    ModelToolCall,
)
from .errors import (
    BuildError,
    ClassificationError,
    ClassificationResponseError,
    ConversationBusyError,
    FSMDefinitionNotFoundError,
    FSMError,
    InvalidTransitionError,
    LLMResponseError,
    ModelNotConfiguredError,
    OKFError,
    RunBudgetExceededError,
    SkillError,
    StateNotFoundError,
    TransitionEvaluationError,
)
from .extraction import (
    FieldExtractionConfig,
    FieldType,
    TypedFieldType,
    checked_key_names,
    typed_field_extraction,
)
from .fsm import (
    Conversation,
    FSMContext,
    FSMDefinition,
    FSMInstance,
    dangling_transition_message,
    exchange_preview_lines,
    handler_only_names,
    orphaned_states_message,
    reachable_from,
    unknown_initial_state_message,
)
from .kinds import (
    CompletionStateConfig,
    ContextScope,
    SynthesizeConfig,
    is_public_result_key,
)
from .limits import (
    ASCII_IDENTIFIER_PATTERN,
    CONVERSATION_SUMMARY_MAX_CHARS,
    DEFAULT_SYNTHESIS_RESULT_KEY,
    MAX_CLASSIFICATION_INTENTS,
    MAX_PRIORITY,
    RESPONSE_TEXT_MAX_LENGTH,
    SYNTHESIZE_GOAL_MAX_LENGTH,
    SYNTHESIZE_KEY_MAX_LENGTH,
)
from .pass_models import (
    PROMPT_MAX_CHARS,
    USER_MESSAGE_MAX_CHARS,
    BulkExtractionRequest,
    DataExtractionResponse,
    FieldExtractionRequest,
    FieldExtractionResponse,
    ResponseGenerationRequest,
    ResponseGenerationResponse,
)
from .results import (
    AdvanceResult,
    TransitionEvaluation,
    TransitionEvaluationResult,
)
from .state import (
    State,
)
from .transition import (
    Transition,
    TransitionCondition,
    TransitionOption,
    logic_has_unresolvable_reads,
    logic_referenced_keys,
)

__all__ = [
    "ASCII_IDENTIFIER_PATTERN",
    "AdvanceResult",
    "BuildError",
    "BulkExtractionRequest",
    "CONTEXT_KEY_CUT_OFF",
    "CONVERSATION_SUMMARY_MAX_CHARS",
    "ClassificationError",
    "ClassificationExtractionConfig",
    "ClassificationPromptConfig",
    "ClassificationResponseError",
    "ClassificationResult",
    "ClassificationSchema",
    "CompletionRequest",
    "CompletionResponse",
    "CompletionStateConfig",
    "ContextScope",
    "Conversation",
    "ConversationBusyError",
    "DEFAULT_SYNTHESIS_RESULT_KEY",
    "DataExtractionResponse",
    "FINISH_REASON_LENGTH",
    "FSMContext",
    "FSMDefinition",
    "FSMDefinitionNotFoundError",
    "FSMError",
    "FSMInstance",
    "FieldExtractionConfig",
    "FieldExtractionRequest",
    "FieldExtractionResponse",
    "FieldType",
    "HierarchicalResult",
    "HierarchicalSchema",
    "IntentDefinition",
    "IntentScore",
    "InvalidTransitionError",
    "LLMCallCounts",
    "LLMResponseError",
    "LLMUsage",
    "MAX_CLASSIFICATION_INTENTS",
    "MAX_PRIORITY",
    "ModelNotConfiguredError",
    "ModelToolCall",
    "MultiClassificationResult",
    "OKFError",
    "PROMPT_MAX_CHARS",
    "RESPONSE_TEXT_MAX_LENGTH",
    "ResponseGenerationRequest",
    "ResponseGenerationResponse",
    "RunBudgetExceededError",
    "SYNTHESIZE_GOAL_MAX_LENGTH",
    "SYNTHESIZE_KEY_MAX_LENGTH",
    "SkillError",
    "State",
    "StateNotFoundError",
    "SynthesizeConfig",
    "Transition",
    "TransitionCondition",
    "TransitionEvaluation",
    "TransitionEvaluationError",
    "TransitionEvaluationResult",
    "TransitionOption",
    "TypedFieldType",
    "USER_MESSAGE_MAX_CHARS",
    "checked_key_names",
    "dangling_transition_message",
    "exchange_preview_lines",
    "handler_only_names",
    "is_public_result_key",
    "logic_has_unresolvable_reads",
    "logic_referenced_keys",
    "orphaned_states_message",
    "reachable_from",
    "typed_field_extraction",
    "unknown_initial_state_message",
]
