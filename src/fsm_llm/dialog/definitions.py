from __future__ import annotations

"""
Enhanced FSM-LLM definitions with improved 2-pass architecture.

This module defines the core data structures for a refined 2-pass FSM-LLM system:
1. Pass 1: Data extraction + transition evaluation
2. Pass 2: Response generation based on final state

Key Changes:
- Separate data extraction from response generation
- Response generation occurs after transition evaluation
- Enhanced request/response models for each pass
"""

from collections import deque
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from ..constants import (
    DEFAULT_MAX_HISTORY_SIZE,
    DEFAULT_MAX_MESSAGE_LENGTH,
    INTERNAL_KEY_PREFIXES,
    MESSAGE_TRUNCATION_SUFFIX,
)

# --------------------------------------------------------------
# local imports
# --------------------------------------------------------------
from ..logging import logger

if TYPE_CHECKING:
    from .memory import WorkingMemory  # noqa: F401

# --------------------------------------------------------------
# Enums for LLM Request Types
# --------------------------------------------------------------


class LLMRequestType(str, Enum):
    """Types of requests that can be sent to LLM."""

    DATA_EXTRACTION = "data_extraction"
    RESPONSE_GENERATION = "response_generation"
    CLASSIFICATION = "classification"
    FIELD_EXTRACTION = "field_extraction"


class TransitionEvaluationResult(str, Enum):
    """Results of transition evaluation."""

    DETERMINISTIC = "deterministic"  # Clear single transition
    AMBIGUOUS = "ambiguous"  # Multiple valid transitions, need LLM
    BLOCKED = "blocked"  # No valid transitions available


# --------------------------------------------------------------
# Data Extraction Models (Pass 1)
# --------------------------------------------------------------


class DataExtractionResponse(BaseModel):
    """
    Response from data extraction containing only extracted information.

    No user-facing message is generated at this stage.
    """

    extracted_data: dict[str, Any] = Field(
        default_factory=dict, description="Data extracted from user input"
    )

    confidence: float = Field(
        default=1.0,
        description="Confidence in the extraction (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )

    reasoning: str | None = Field(
        None,
        description="Internal reasoning for debugging (not shown to user)",
        max_length=5000,
    )

    additional_info_needed: bool | None = Field(
        default=None,
        description="Whether additional information is needed from the user",
    )


# --------------------------------------------------------------
# Response Generation Models (Pass 2)
# --------------------------------------------------------------


class ResponseGenerationRequest(BaseModel):
    """
    Request for generating user-facing response based on final state.

    This request generates the actual message shown to users after
    data extraction and transition evaluation are complete.
    """

    system_prompt: str = Field(
        ...,
        description="Prompt focused on response generation for current state",
        min_length=1,
        max_length=30000,
    )

    user_message: str = Field(
        ...,
        description="Original user message for context",
        min_length=0,
        max_length=10000,
    )

    extracted_data: dict[str, Any] = Field(
        default_factory=dict, description="Data extracted in Pass 1"
    )

    context: dict[str, Any] = Field(
        default_factory=dict, description="Current conversation context"
    )

    transition_occurred: bool = Field(
        default=False, description="Whether a state transition occurred"
    )

    previous_state: str | None = Field(
        None, description="Previous state if transition occurred"
    )

    response_format: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional response format for constrained decoding. "
            "When set, passed to the LLM provider to enforce structured output "
            "(e.g., JSON schema). Falls back to free-text if provider unsupported."
        ),
    )


class ResponseGenerationResponse(BaseModel):
    """
    Response containing the final user-facing message.

    Generated after all data extraction and transitions are complete.
    """

    message: str = Field(..., description="Final user-facing message", max_length=5000)

    reasoning: str | None = Field(
        None, description="Internal reasoning for debugging", max_length=5000
    )

    message_type: str = Field(
        default="response", description="Type of the response message"
    )


# --------------------------------------------------------------
# Enhanced Transition Decision Models
# --------------------------------------------------------------


class TransitionOption(BaseModel):
    """
    A possible transition option for LLM evaluation.

    Contains minimal information needed for transition decisions
    without exposing internal FSM structure.
    """

    target_state: str = Field(
        ...,
        description="Target state identifier",
        min_length=1,
        max_length=100,
        pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$",
    )

    description: str = Field(
        ...,
        description="Human-readable description of when this transition applies",
        min_length=1,
        max_length=500,
    )

    priority: int = Field(
        default=100,
        description="Priority for this transition (lower = higher priority)",
        ge=0,
        le=1000,
    )


# --------------------------------------------------------------
# Field Extraction Models (targeted single-field extraction)
# --------------------------------------------------------------


class FieldExtractionConfig(BaseModel):
    """Configuration for targeted extraction of a single field.

    Declared on a :class:`State` via ``field_extractions`` to run focused
    extraction after the bulk ``extract_data`` pass completes.
    """

    field_name: str = Field(
        ...,
        description="Context key to extract into",
        min_length=1,
        max_length=100,
    )

    field_type: Literal["str", "int", "float", "bool", "list", "dict", "any"] = Field(
        default="str",
        description="Expected type: str, int, float, bool, list, dict, any",
    )

    extraction_instructions: str = Field(
        ...,
        description="Focused instructions for extracting this specific field",
        min_length=1,
        max_length=5000,
    )

    context_keys: list[str] | None = Field(
        default=None,
        description=(
            "Context keys to include as dynamic context in the extraction prompt. "
            "If None, all user-visible context is passed."
        ),
    )

    validation_rules: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Validation rules: allowed_values, min_length, max_length, "
            "min_value, max_value, pattern (regex)"
        ),
    )

    required: bool = Field(
        default=True,
        description="Whether extraction failure should be treated as an error",
    )

    confidence_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for the extraction to be accepted",
    )

    _ALLOWED_VALIDATION_RULE_KEYS: ClassVar[set[str]] = {
        "allowed_values",
        "min_length",
        "max_length",
        "min_value",
        "max_value",
        "pattern",
    }

    @model_validator(mode="after")
    def validate_validation_rule_keys(self) -> FieldExtractionConfig:
        """Reject unknown keys in validation_rules to catch typos early."""
        if self.validation_rules:
            unknown = set(self.validation_rules) - self._ALLOWED_VALIDATION_RULE_KEYS
            if unknown:
                raise ValueError(
                    f"Unknown validation_rules keys: {sorted(unknown)}. "
                    f"Allowed: {sorted(self._ALLOWED_VALIDATION_RULE_KEYS)}"
                )
        return self


class FieldExtractionRequest(BaseModel):
    """Request for extracting a single specific field from user input."""

    system_prompt: str = Field(
        ...,
        description="Focused prompt for single-field extraction",
        min_length=1,
        max_length=30000,
    )

    user_message: str = Field(
        ...,
        description="User input to extract the field from",
        min_length=0,
        max_length=10000,
    )

    field_name: str = Field(
        ...,
        description="Name of the field to extract",
        min_length=1,
        max_length=100,
    )

    field_type: Literal["str", "int", "float", "bool", "list", "dict", "any"] = Field(
        default="str",
        description="Expected type of the extracted value",
    )

    context: dict[str, Any] | None = Field(
        default=None,
        description="Dynamic context to guide extraction",
    )

    validation_rules: dict[str, Any] | None = Field(
        default=None,
        description="Validation rules for the extracted value",
    )


class FieldExtractionResponse(BaseModel):
    """Response from single-field extraction."""

    field_name: str = Field(
        ...,
        description="Name of the extracted field",
        min_length=1,
        max_length=100,
    )

    value: Any = Field(
        default=None,
        description="The extracted value (typed according to field_type)",
    )

    confidence: float = Field(
        default=1.0,
        description="Confidence in the extraction (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )

    reasoning: str | None = Field(
        default=None,
        description="Reasoning for the extraction decision",
        max_length=5000,
    )

    is_valid: bool = Field(
        default=True,
        description="Whether the value passed validation rules",
    )

    validation_error: str | None = Field(
        default=None,
        description="Validation error message if is_valid is False",
    )


# --------------------------------------------------------------
# Classification Extraction Models
# --------------------------------------------------------------


class ClassificationExtractionConfig(BaseModel):
    """Configuration for classification-based extraction of a categorical value.

    Declared on a :class:`State` via ``classification_extractions`` to classify
    user input into predefined categories and store the result in context.
    Runs during Pass 1 alongside field extractions, before transition evaluation.
    """

    field_name: str = Field(
        ...,
        description="Context key to store the classified intent in",
        min_length=1,
        max_length=100,
    )

    intents: list[IntentDefinition] = Field(
        min_length=2,
        description="Classification categories (2-15 recommended)",
    )

    fallback_intent: str = Field(
        description="Intent to use when classification is ambiguous or low-confidence",
    )

    confidence_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Below this threshold, extraction is treated as failed",
    )

    required: bool = Field(
        default=False,
        description="If True, failed classification triggers retry logic",
    )

    model: str | None = Field(
        default=None,
        description="Override LLM model for this classification (None = use pipeline's)",
    )

    prompt_config: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Override ClassificationPromptConfig fields as a dict. "
            "Keys: include_reasoning, max_tokens, temperature, include_entities."
        ),
    )

    context_keys: list[str] | None = Field(
        default=None,
        description=(
            "Context keys to snapshot alongside the classification result. "
            "If None, no context snapshot is stored."
        ),
    )

    @model_validator(mode="after")
    def validate_fallback_in_intents(self) -> ClassificationExtractionConfig:
        names = [i.name for i in self.intents]
        if self.fallback_intent not in names:
            raise ValueError(
                f"Fallback intent '{self.fallback_intent}' must be in the intent list"
            )
        if len(names) != len(set(names)):
            raise ValueError("Intent names must be unique")
        return self


# --------------------------------------------------------------
# Enhanced Transition Condition Models
# --------------------------------------------------------------


class TransitionCondition(BaseModel):
    """
    Enhanced transition condition with evaluation capabilities.

    Supports both simple key-based and complex JsonLogic conditions.
    """

    description: str = Field(
        ...,
        description="Human-readable description of this condition",
        min_length=1,
        max_length=500,
    )

    requires_context_keys: list[str] | None = Field(
        default=None, description="Context keys required for evaluation"
    )

    logic: dict[str, Any] | None = Field(
        default=None, description="JsonLogic expression for complex evaluation"
    )

    evaluation_priority: int = Field(
        default=100,
        description="Priority for condition evaluation (lower = earlier)",
        ge=0,
        le=1000,
    )


class Transition(BaseModel):
    """
    Enhanced transition definition with evaluation metadata.

    Supports priority-based and condition-based transition logic.
    """

    target_state: str = Field(
        ...,
        description="Target state identifier",
        min_length=1,
        max_length=100,
        pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$",
    )

    description: str = Field(
        ...,
        description="When this transition should occur",
        min_length=1,
        max_length=500,
    )

    conditions: list[TransitionCondition] | None = Field(
        default=None, description="Conditions that must be satisfied"
    )

    priority: int = Field(
        default=100, description="Priority for transition selection", ge=0, le=1000
    )

    llm_description: str | None = Field(
        None,
        description="Description for LLM when choosing between transitions",
        max_length=300,
    )


# --------------------------------------------------------------
# State Definition Models
# --------------------------------------------------------------


class State(BaseModel):
    """
    Enhanced state definition for improved 2-pass architecture.

    Separates data extraction concerns from response generation.
    """

    id: str = Field(
        ...,
        description="Unique state identifier",
        min_length=1,
        max_length=100,
        pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$",
    )

    description: str = Field(
        ...,
        description="Human-readable state description",
        min_length=1,
        max_length=300,
    )

    purpose: str = Field(
        ...,
        description="What should be accomplished in this state",
        min_length=1,
        max_length=500,
    )

    extraction_instructions: str | None = Field(
        None,
        description="Instructions for data extraction in this state",
        max_length=5000,
    )

    response_instructions: str | None = Field(
        None,
        description="Instructions for response generation in this state",
        max_length=5000,
    )

    transitions: list[Transition] = Field(
        default_factory=list, description="Available transitions from this state"
    )

    required_context_keys: list[str] | None = Field(
        default=None, description="Context keys that should be collected"
    )

    extraction_retries: int = Field(
        default=1,
        ge=0,
        le=3,
        description=(
            "Number of refinement extraction passes when confidence is low "
            "or required_context_keys are missing. 0 disables multi-pass."
        ),
    )

    extraction_confidence_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum confidence for extraction to be accepted without retry. "
            "0.0 disables confidence-based retry (only missing-key retry applies)."
        ),
    )

    transition_classification: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Custom classification config for ambiguous transition resolution. "
            "dict: user-provided config with custom intent descriptions and thresholds. "
            "None: auto-generate classification schema from transition descriptions. "
            "Classification is always-on for ambiguous transitions."
        ),
    )

    field_extractions: list[FieldExtractionConfig] | None = Field(
        default=None,
        description=(
            "Targeted field extractions to run after bulk extraction. "
            "Each entry extracts a single named field with custom instructions, "
            "dynamic context selection, and validation rules."
        ),
    )

    classification_extractions: list[ClassificationExtractionConfig] | None = Field(
        default=None,
        description=(
            "Classification-based extractions to run during Pass 1. "
            "Each entry classifies user input into predefined categories "
            "and stores the result in context for transition evaluation."
        ),
    )

    context_scope: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional context scoping for this state. Controls which "
            "context keys are injected into LLM prompts. "
            "Keys: 'read_keys' (list[str]) — keys to include in prompts; "
            "'write_keys' (list[str]) — keys this state is expected to produce. "
            "When None, all user-visible context is injected (default behavior)."
        ),
    )

    # M3a (merge spec §3 I6) — private scaffolding field for the
    # response-Leaf emission rollout. When False (the default),
    # compile_fsm._compile_state takes the legacy
    # ``App(CB_RESPOND, instance)`` host-callback path for non-cohort
    # states (byte-equivalent to pre-M3 behavior). When True, M3b's
    # Leaf-emission branch will fire — a non-cohort state's response
    # generation lifts to a real ``Leaf`` so Theorem-2 strict equality
    # ``Executor.oracle_calls == plan(...).predicted_calls`` holds for
    # non-cohort FSM states too. The field is underscore-prefixed
    # because it is INTERNAL: it controls compile-time emission, not
    # FSM authoring semantics, and is expected to be flipped to True
    # globally in M3c then removed in M3d (the field's existence is
    # transitional). FSM JSON authors should never set this. Pydantic
    # accepts the field via populate_by_name but it is not part of the
    # documented v4.1 schema.
    _emit_response_leaf_for_non_cohort: bool = False

    # A.D5 (merge spec §4 CAND-C; plan_2026-04-28_90d0824f step 1) —
    # opt-in compile-time pathway for terminal non-cohort states to
    # emit a real ``Leaf(schema_ref=...)`` instead of the conservative
    # D3 fallback ``App(CB_RESPOND, instance)``. When set, the compiler
    # routes the terminal-non-cohort branch to ``leaf(template,
    # schema_ref=<dotted-path>, streaming=False)`` so the executor
    # enforces structured output via the kernel's ``_invoke_structured``
    # path and Theorem-2 strict equality holds end-to-end. When None
    # (the default), the legacy D3 ``App(CB_RESPOND, instance)``
    # survives — preserves byte-equivalent behaviour for every
    # Category-A FSM that has not opted in (which is every FSM at HEAD;
    # stdlib agents migration to set this field at FSM-construction
    # time is deferred per D-002 in plan_2026-04-28_ca542489).
    # Compile-time mutual exclusion with ``streaming=True`` per D-005 —
    # the streaming-Leaf path in ``process_stream_compiled`` degrades
    # structured terminal opt-in responses to a single-chunk iterator
    # via the entry-point ``iter([result])`` normalisation.
    #
    # Accepts EITHER a Pydantic ``BaseModel`` subclass (preferred —
    # auto-converted to ``f"{cls.__module__}.{cls.__qualname__}"``
    # at compile time per D-007-SURPRISE in plan_90d0824f) OR a
    # pre-formatted dotted-path string ``"module.Class"`` (for callers
    # that already hold the resolved path). Validation happens at
    # ``compile_fsm._compile_state``; runtime resolution happens in
    # ``runtime/oracle.py:_resolve_schema``.
    #
    # Typed as ``Any`` (not ``type[BaseModel] | str``) because Pydantic
    # v2 rejects bare class-type annotations on a non-frozen
    # ``BaseModel`` without ``arbitrary_types_allowed=True``; we keep
    # ``State``'s config minimal and validate the constraint at the
    # compile-time gate.
    output_schema_ref: Any = None


# --------------------------------------------------------------
# FSM Definition Models
# --------------------------------------------------------------


class FSMDefinition(BaseModel):
    """
    Complete FSM definition for improved 2-pass architecture.

    Enhanced with separate extraction and response capabilities.
    """

    name: str = Field(
        ..., description="FSM name identifier", min_length=1, max_length=100
    )

    description: str = Field(
        ...,
        description="FSM purpose and functionality description",
        min_length=1,
        max_length=1000,
    )

    states: dict[str, State] = Field(
        ..., description="All states in the FSM", min_length=1
    )

    initial_state: str = Field(
        ..., description="Starting state identifier", min_length=1
    )

    version: str = Field(
        default="4.1", description="FSM definition version", min_length=1, max_length=20
    )

    persona: str | None = Field(
        None, description="Conversation persona for response generation", max_length=500
    )

    @model_validator(mode="after")
    def validate_fsm_structure(self) -> FSMDefinition:
        """Comprehensive FSM validation for improved 2-pass architecture."""
        logger.debug(f"Validating FSM: {self.name}")

        # Basic structure validation
        if self.initial_state not in self.states:
            raise ValueError(
                f"Initial state '{self.initial_state}' not found in states"
            )

        # Validate state.id matches dict key
        for state_id, state in self.states.items():
            if state.id != state_id:
                raise ValueError(
                    f"State id '{state.id}' does not match dict key '{state_id}'"
                )

        # Validate all transitions
        for state_id, state in self.states.items():
            for transition in state.transitions:
                if transition.target_state not in self.states:
                    raise ValueError(
                        f"Invalid transition from '{state_id}' to non-existent state '{transition.target_state}'"
                    )

        # Terminal state validation
        terminal_states = {
            state_id for state_id, state in self.states.items() if not state.transitions
        }

        if not terminal_states:
            raise ValueError("FSM must have at least one terminal state")

        # Reachability validation
        reachable_states = self._calculate_reachable_states()
        orphaned_states = set(self.states.keys()) - reachable_states

        if orphaned_states:
            raise ValueError(f"Orphaned states detected: {sorted(orphaned_states)}")

        # Validate terminal state reachability
        reachable_terminals = terminal_states.intersection(reachable_states)
        if not reachable_terminals:
            raise ValueError("No terminal states are reachable from initial state")

        unreachable_terminals = terminal_states - reachable_states
        if unreachable_terminals:
            logger.warning(
                f"Unreachable terminal states in FSM '{self.name}': "
                f"{sorted(unreachable_terminals)}"
            )

        logger.debug(f"FSM '{self.name}' validation successful")
        return self

    def _calculate_reachable_states(self) -> set:
        """Calculate all states reachable from initial state."""
        reachable = {self.initial_state}
        to_process = deque([self.initial_state])

        while to_process:
            current = to_process.popleft()
            current_state = self.states[current]

            for transition in current_state.transitions:
                if transition.target_state not in reachable:
                    reachable.add(transition.target_state)
                    to_process.append(transition.target_state)

        return reachable


# --------------------------------------------------------------
# Context and Instance Models (Enhanced)
# --------------------------------------------------------------


class Conversation(BaseModel):
    """Enhanced conversation management for improved 2-pass architecture."""

    exchanges: list[dict[str, str]] = Field(
        default_factory=list, description="Conversation history in chronological order"
    )

    max_history_size: int = Field(
        default=DEFAULT_MAX_HISTORY_SIZE,
        description="Maximum conversation exchanges to retain",
        ge=0,
        le=1000,
    )

    max_message_length: int = Field(
        default=DEFAULT_MAX_MESSAGE_LENGTH,
        description="Maximum message length in characters",
        ge=1,
        le=50000,
    )

    summary: str | None = Field(
        default=None,
        description=(
            "Compressed summary of older exchanges that were trimmed from history. "
            "Populated automatically when history exceeds max_history_size, or "
            "explicitly via ContextCompactor.summarize()."
        ),
    )

    def add_user_message(self, message: str) -> None:
        """Add user message with automatic truncation."""
        if len(message) > self.max_message_length:
            suffix = MESSAGE_TRUNCATION_SUFFIX
            if self.max_message_length <= len(suffix):
                message = message[: self.max_message_length]
            else:
                message = message[: self.max_message_length - len(suffix)] + suffix

        self.exchanges.append({"user": message})
        self._maintain_history_size()

    def add_system_message(self, message: str) -> None:
        """Add system message with automatic truncation."""
        if len(message) > self.max_message_length:
            suffix = MESSAGE_TRUNCATION_SUFFIX
            if self.max_message_length <= len(suffix):
                message = message[: self.max_message_length]
            else:
                message = message[: self.max_message_length - len(suffix)] + suffix

        self.exchanges.append({"system": message})
        self._maintain_history_size()

    def get_recent(self, n: int | None = None) -> list[dict[str, str]]:
        """Get recent conversation messages.

        Args:
            n: Number of exchanges (user+system pairs) to return.
               Defaults to max_history_size.
        """
        if n is None:
            n = self.max_history_size

        if n <= 0:
            return []

        # Each exchange is assumed to be a user+system pair (2 messages).
        # If the conversation has an odd number of messages (e.g., user sent
        # but system hasn't replied yet), this may return a partial pair.
        return self.exchanges[-n * 2 :]

    def search(self, query: str, limit: int = 5) -> list[dict[str, str]]:
        """Search conversation history for exchanges matching a query.

        Performs case-insensitive substring matching on all exchange
        messages (both user and system). Also searches the summary if
        one exists.

        Args:
            query: Search string.
            limit: Maximum number of matching exchanges to return.

        Returns:
            List of matching exchange dicts, most recent first.
        """
        if not query:
            return []

        query_lower = query.lower()
        matches: list[dict[str, str]] = []

        # Search exchanges in reverse (most recent first)
        for exchange in reversed(self.exchanges):
            if len(matches) >= limit:
                break
            for value in exchange.values():
                if query_lower in value.lower():
                    matches.append(exchange)
                    break

        return matches

    def get_summary_and_recent(
        self, n: int | None = None
    ) -> tuple[str | None, list[dict[str, str]]]:
        """Get conversation summary (if any) and recent exchanges.

        Convenience method for prompt builders that want to include
        both the compressed summary of older history and the recent
        exchange window.

        Args:
            n: Number of recent exchanges. Defaults to max_history_size.

        Returns:
            Tuple of (summary_text_or_None, recent_exchanges).
        """
        return self.summary, self.get_recent(n)

    def _maintain_history_size(self) -> None:
        """Trim history to max_history_size exchanges (each = 2 messages).

        When trimming, captures a simple text summary of the removed
        exchanges to preserve context that would otherwise be lost.
        """
        limit = self.max_history_size * 2
        if limit == 0:
            self.exchanges.clear()
            return
        if len(self.exchanges) > limit:
            trimmed = self.exchanges[:-limit]
            self._append_to_summary(trimmed)
            self.exchanges = self.exchanges[-limit:]

    def _append_to_summary(self, trimmed_exchanges: list[dict[str, str]]) -> None:
        """Append trimmed exchanges to the conversation summary.

        Produces a compact text representation of the trimmed exchanges
        and appends it to the existing summary (if any). Caps total
        summary length at 2000 characters.
        """
        if not trimmed_exchanges:
            return

        lines: list[str] = []
        for exchange in trimmed_exchanges:
            for role, message in exchange.items():
                # Compact: first 100 chars of each message
                preview = message[:100]
                if len(message) > 100:
                    preview += "..."
                lines.append(f"{role}: {preview}")

        new_text = " | ".join(lines)

        if self.summary:
            combined = f"{self.summary} | {new_text}"
        else:
            combined = new_text

        # Cap summary length
        max_summary = 2000
        if len(combined) > max_summary:
            combined = combined[:max_summary]

        self.summary = combined


class FSMContext(BaseModel):
    """Enhanced context management for improved 2-pass architecture.

    Supports an optional ``working_memory`` for structured buffer-based
    context management. When set, ``get_user_visible_data()`` includes
    data from all working memory buffers (flattened). The flat ``data``
    dict remains the primary storage for backward compatibility.
    """

    model_config = {"arbitrary_types_allowed": True}

    data: dict[str, Any] = Field(
        default_factory=dict, description="Conversation context data"
    )

    conversation: Conversation = Field(
        default_factory=Conversation, description="Conversation history management"
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict, description="System metadata and operational data"
    )

    working_memory: Any = (
        Field(  # Runtime: WorkingMemory | None (avoids circular import)
            default=None,
            description=(
                "Optional WorkingMemory instance for structured buffer-based "
                "context management. When set, get_user_visible_data() merges "
                "data from working memory buffers. Import from fsm_llm.memory."
            ),
            exclude=True,
        )
    )

    def __init__(self, **data):
        """Initialize with optional conversation configuration."""
        if "conversation" not in data:
            data = dict(data)
            max_history = data.pop("max_history_size", DEFAULT_MAX_HISTORY_SIZE)
            max_message_length = data.pop(
                "max_message_length", DEFAULT_MAX_MESSAGE_LENGTH
            )

            data["conversation"] = Conversation(
                max_history_size=max_history, max_message_length=max_message_length
            )

        super().__init__(**data)

    def update(self, new_data: dict[str, Any]) -> None:
        """Update context with new data."""
        if new_data:
            for key in new_data:
                if any(key.startswith(p) for p in INTERNAL_KEY_PREFIXES):
                    logger.warning(
                        f"Context update contains internal-prefix key: {key!r}"
                    )
            logger.debug(f"Updating context with keys: {list(new_data.keys())}")
            self.data.update(new_data)

    def get_user_visible_data(self) -> dict[str, Any]:
        """Get context data filtered for user visibility.

        When ``working_memory`` is set, merges flattened buffer data
        with the flat ``data`` dict. The flat ``data`` dict takes
        precedence on key collisions.
        """
        # Start with working memory data if available
        if self.working_memory is not None and hasattr(
            self.working_memory, "get_all_data"
        ):
            merged = self.working_memory.get_all_data()
            # Flat data dict overrides working memory on collision
            merged.update(self.data)
        else:
            merged = self.data

        return {
            key: value
            for key, value in merged.items()
            if not any(key.startswith(p) for p in INTERNAL_KEY_PREFIXES)
            and key != "system"
        }


class FSMInstance(BaseModel):
    """Enhanced FSM instance for improved 2-pass architecture."""

    fsm_id: str = Field(
        ...,
        description="FSM definition identifier",
        min_length=1,
    )

    current_state: str = Field(
        ..., description="Current state identifier", min_length=1, max_length=100
    )

    context: FSMContext = Field(
        default_factory=FSMContext, description="Conversation context and history"
    )

    persona: str | None = Field(
        default="Helpful AI assistant",
        description="Conversation persona",
        max_length=500,
    )

    last_extraction_response: DataExtractionResponse | None = Field(
        None, description="Last data extraction response for debugging"
    )

    last_transition_decision: ClassificationResult | None = Field(
        None, description="Last classification result for transition debugging"
    )

    last_response_generation: ResponseGenerationResponse | None = Field(
        None, description="Last response generation for debugging"
    )


# --------------------------------------------------------------
# Transition Evaluation Models
# --------------------------------------------------------------


class TransitionEvaluation(BaseModel):
    """
    Result of evaluating possible transitions from current state.

    Used by the transition evaluator to determine next steps.
    """

    result_type: TransitionEvaluationResult = Field(
        ..., description="Type of evaluation result"
    )

    deterministic_transition: str | None = Field(
        None, description="Target state if deterministically determined"
    )

    available_options: list[TransitionOption] = Field(
        default_factory=list, description="Available transition options if ambiguous"
    )

    blocked_reason: str | None = Field(
        None, description="Reason if transitions are blocked"
    )

    confidence: float = Field(
        default=0.0,
        description="Confidence in the evaluation result (0.0-1.0, or -1.0 for evaluation errors)",
        ge=-1.0,
        le=1.0,
    )

    @model_validator(mode="after")
    def validate_result_consistency(self) -> TransitionEvaluation:
        """Ensure populated fields match result_type."""
        if self.result_type == TransitionEvaluationResult.DETERMINISTIC:
            if (
                not self.deterministic_transition
                or not self.deterministic_transition.strip()
            ):
                raise ValueError(
                    "DETERMINISTIC result requires non-empty deterministic_transition"
                )
        elif self.result_type == TransitionEvaluationResult.AMBIGUOUS:
            if not self.available_options:
                raise ValueError(
                    "AMBIGUOUS result requires non-empty available_options"
                )
        elif self.result_type == TransitionEvaluationResult.BLOCKED:
            if self.blocked_reason is None:
                raise ValueError("BLOCKED result requires blocked_reason")
        return self


# --------------------------------------------------------------
# Classification Models
# --------------------------------------------------------------


class IntentDefinition(BaseModel):
    """A single intent class within a classification schema."""

    name: str = Field(description="Snake_case identifier used for handler routing")
    description: str = Field(description="Human-readable description shown to the LLM")

    @model_validator(mode="after")
    def validate_name_format(self) -> IntentDefinition:
        if not self.name.replace("_", "").isalnum():
            raise ValueError(
                f"Intent name must be alphanumeric with underscores, got '{self.name}'"
            )
        if self.name[0].isdigit():
            raise ValueError(
                f"Intent name must start with a letter or underscore, got '{self.name}'"
            )
        return self


class ClassificationSchema(BaseModel):
    """
    Defines the complete set of intents for a classifier.

    Enforces mutual exclusivity guidelines: max 15 intents per schema
    and a mandatory fallback class.
    """

    intents: list[IntentDefinition] = Field(
        min_length=2,
        max_length=15,
        description="List of intent definitions (2-15)",
    )
    fallback_intent: str = Field(
        description="Name of the fallback intent for ambiguous inputs"
    )
    confidence_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Below this threshold, the classifier signals low confidence",
    )

    @model_validator(mode="after")
    def validate_schema(self) -> ClassificationSchema:
        names = [i.name for i in self.intents]
        if len(names) != len(set(names)):
            raise ValueError("Intent names must be unique")
        if self.fallback_intent not in names:
            raise ValueError(
                f"Fallback intent '{self.fallback_intent}' must be in the intent list"
            )
        return self

    @property
    def intent_names(self) -> list[str]:
        return [i.name for i in self.intents]


class IntentScore(BaseModel):
    """A single scored intent within a classification result."""

    intent: str = Field(description="The classified intent name")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Model confidence in this classification"
    )
    entities: dict[str, str] = Field(
        default_factory=dict, description="Extracted entities relevant to this intent"
    )

    @field_validator("entities", mode="before")
    @classmethod
    def coerce_entity_values(cls, v: Any) -> dict[str, str]:
        if not isinstance(v, dict):
            return {}
        return {
            k: ", ".join(str(i) for i in val) if isinstance(val, list) else str(val)
            for k, val in v.items()
        }


class ClassificationResult(BaseModel):
    """Result of a single-intent classification."""

    reasoning: str = Field(
        description="Chain-of-thought explanation preceding the classification"
    )
    intent: str = Field(description="The primary classified intent")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Model confidence in this classification"
    )
    entities: dict[str, str | None] = Field(
        default_factory=dict, description="Extracted entities relevant to the intent"
    )

    @field_validator("entities", mode="before")
    @classmethod
    def coerce_entity_values(cls, v: Any) -> dict[str, str | None]:
        if not isinstance(v, dict):
            return {}
        return {
            k: (
                ", ".join(str(i) for i in val)
                if isinstance(val, list)
                else (str(val) if val is not None else None)
            )
            for k, val in v.items()
        }

    #: Default threshold for is_low_confidence when no schema is available.
    #: For schema-aware checks, use Classifier.is_low_confidence() instead.
    DEFAULT_CONFIDENCE_THRESHOLD: ClassVar[float] = 0.6

    @property
    def is_low_confidence(self) -> bool:
        """Check against the default threshold. Use schema-aware check in Classifier."""
        return self.confidence < self.DEFAULT_CONFIDENCE_THRESHOLD


class MultiClassificationResult(BaseModel):
    """Result of a multi-intent classification (compound queries)."""

    reasoning: str = Field(
        description="Chain-of-thought explanation preceding the classification"
    )
    intents: list[IntentScore] = Field(
        min_length=1,
        max_length=5,
        description="Ranked list of detected intents, most probable first",
    )

    @property
    def primary(self) -> IntentScore:
        return self.intents[0]


class DomainSchema(BaseModel):
    """
    Maps a domain to its intent sub-schema for hierarchical classification.

    Use when the total intent count exceeds ~15. Stage 1 classifies domain,
    stage 2 classifies intent within that domain.
    """

    domain: str = Field(description="Domain identifier (snake_case)")
    intent_schema: ClassificationSchema = Field(
        description="Intent schema for this domain"
    )


class HierarchicalSchema(BaseModel):
    """Top-level schema for two-stage hierarchical classification."""

    domain_schema: ClassificationSchema = Field(
        description="Stage 1: domain-level classification schema"
    )
    intent_schemas: dict[str, ClassificationSchema] = Field(
        description="Stage 2: domain -> intent schema mapping"
    )

    @model_validator(mode="after")
    def validate_domain_coverage(self) -> HierarchicalSchema:
        domain_names = set(self.domain_schema.intent_names)
        schema_keys = set(self.intent_schemas.keys())
        missing = domain_names - schema_keys - {self.domain_schema.fallback_intent}
        if missing:
            raise ValueError(f"Missing intent schemas for domains: {missing}")
        return self


class HierarchicalResult(BaseModel):
    """Result of a hierarchical (two-stage) classification."""

    domain: ClassificationResult = Field(description="Stage 1 domain classification")
    intent: ClassificationResult = Field(
        description="Stage 2 intent classification within the domain"
    )


# --------------------------------------------------------------
# Exception Classes
# --------------------------------------------------------------


class FSMError(Exception):
    """Base exception for FSM-related errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.details = details or {}


class StateNotFoundError(FSMError):
    """Exception for non-existent state references."""

    def __init__(self, message: str, state_id: str | None = None, **kwargs):
        super().__init__(message, **kwargs)
        self.state_id = state_id


class InvalidTransitionError(FSMError):
    """Exception for invalid state transitions."""

    def __init__(
        self,
        message: str,
        source_state: str | None = None,
        target_state: str | None = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.source_state = source_state
        self.target_state = target_state


class LLMResponseError(FSMError):
    """Exception for LLM response processing errors."""

    pass


class TransitionEvaluationError(FSMError):
    """Exception for transition evaluation errors."""

    def __init__(self, message: str, state_id: str | None = None, **kwargs):
        super().__init__(message, **kwargs)
        self.state_id = state_id


class ClassificationError(FSMError):
    """Base exception for classification operations."""

    pass


class SchemaValidationError(ClassificationError):
    """Raised when a classification schema is invalid."""

    pass


class ClassificationResponseError(ClassificationError):
    """Raised when the LLM returns an unparseable classification."""

    pass
