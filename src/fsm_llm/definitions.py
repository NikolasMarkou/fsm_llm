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

from __future__ import annotations

import re
from collections import deque
from collections.abc import Iterator
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .constants import (
    ALLOWED_JSONLOGIC_OPERATIONS,
    DEFAULT_MAX_HISTORY_SIZE,
    DEFAULT_MAX_MESSAGE_LENGTH,
    JSONLOGIC_RAW_ARGUMENT_OPERATIONS,
    MAX_JSONLOGIC_DEPTH,
    MAX_MULTI_INTENTS,
    MESSAGE_TRUNCATION_SUFFIX,
    TRANSITION_CLASSIFICATION_THRESHOLD_KEY,
    has_internal_prefix,
)

# --------------------------------------------------------------
# local imports
# --------------------------------------------------------------
from .logging import logger

if TYPE_CHECKING:
    from .memory import WorkingMemory  # noqa: F401

# --------------------------------------------------------------
# Identifier charset policy
# --------------------------------------------------------------

# The single source of truth for every identifier-shaped field in this module:
# `State.id`, `Transition.target_state`, `FSMDefinition.initial_state` and
# `IntentDefinition.name`. ASCII-only, letter-or-underscore first. Two spellings
# because pydantic's `Field(pattern=)` wants the source string while
# `IntentDefinition.validate_name_format` needs a compiled object -- see the
# D-025 anchor there for why that field must NOT migrate to `pattern=`.
ASCII_IDENTIFIER_PATTERN = r"^[a-zA-Z_][a-zA-Z0-9_]*$"
_ASCII_IDENTIFIER = re.compile(ASCII_IDENTIFIER_PATTERN)

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


class BulkExtractionRequest(BaseModel):
    """Request for free-form bulk data extraction from a state's
    ``extraction_instructions`` (no per-field schema).

    Deliberately minimal (only the two fields the prompt needs): unlike
    ``FieldExtractionRequest`` this is not targeting one named field, so a
    required ``field_name``/``field_type`` would not fit. See
    ``LLMInterface.extract_bulk_data`` for the ABC contract this feeds.
    """

    system_prompt: str = Field(
        ...,
        description="Prompt describing what free-form data to extract",
        min_length=1,
        max_length=30000,
    )

    user_message: str = Field(
        ...,
        description="User input to extract data from",
        min_length=0,
        max_length=10000,
    )


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

    rejected_corrections: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Values the bulk extraction proposed for an already-stored key that "
            "the provenance rule refused to apply and that the user's message "
            "contains (D-032); empty on every other turn"
        ),
    )

    extraction_failed: bool = Field(
        default=False,
        description=(
            "True when the bulk extraction call of this turn raised, so a value "
            "the user restated may not have been stored (D-050)"
        ),
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
        pattern=ASCII_IDENTIFIER_PATTERN,
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
            self._check_validation_rule_types(self.validation_rules)
        return self

    @staticmethod
    def _check_validation_rule_types(rules: dict[str, Any]) -> None:
        """Type-check the rules the extractor compares against at runtime.

        Raises ``ValueError`` naming the offending key, so a bad rule fails at
        load instead of as a ``TypeError`` on the first extraction turn (D-023).
        ``min_value``/``max_value`` stay ``float()``-coerced at use and are not
        checked here.
        """
        for key in ("min_length", "max_length"):
            v = rules.get(key)
            if key in rules and (not isinstance(v, int) or isinstance(v, bool)):
                raise ValueError(f"validation_rules {key!r} must be an int, got {v!r}")
        allowed = rules.get("allowed_values")
        if "allowed_values" in rules and not isinstance(allowed, (list, tuple, set)):
            raise ValueError(
                f"validation_rules 'allowed_values' must be a list, got {allowed!r}"
            )
        if "pattern" in rules:
            try:
                re.compile(rules["pattern"])
            except (re.error, TypeError) as e:
                raise ValueError(
                    f"validation_rules 'pattern' must be a valid regex string: {e}"
                ) from e


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

    # DECISION plan-2026-07-19T191147-4b664252/D-013 [STALE]: this cap MUST stay in
    # lockstep with its sibling `ClassificationSchema.intents` (same file,
    # `max_length=15`), which `MessagePipeline._execute_classification_extractions`
    # builds FROM this config at conversation runtime. Leaving this side
    # uncapped is F-09: the FSM passes `API.from_file` and `fsm-llm-validate`
    # clean, then fails on first ENTRY to the state -- silently skipped when
    # `required=False`, a mid-conversation `ClassificationError` when True.
    # Do NOT relax this back to a docstring "recommended"; if the sibling's cap
    # ever moves, move this one in the same commit. See decisions.md D-013.
    intents: list[IntentDefinition] = Field(
        min_length=2,
        max_length=15,
        description="Classification categories (2-15)",
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
            "Keys: include_reasoning, max_tokens, temperature, include_entities, "
            "multi_intent, max_intents. Bounds (max_tokens >= 1, "
            "0.0 <= temperature <= 2.0, 1 <= max_intents <= 5) and the key set "
            "are checked at load: an unknown key or out-of-range value raises."
        ),
    )

    context_keys: list[str] | None = Field(
        default=None,
        description=(
            "Context keys the classifier sees as context data and that are "
            "snapshotted alongside the classification result. If None, the "
            "classifier sees the state's read_keys-scoped visible context and "
            "no snapshot is stored."
        ),
    )

    @field_validator("prompt_config")
    @classmethod
    def _validate_prompt_config(
        cls, value: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """Reject a ``prompt_config`` that ``ClassificationPromptConfig`` refuses.

        # DECISION plan-2026-09-22T080837-8b258a25/D-011
        # The verdict IS `ClassificationPromptConfig(**value)`, the object the
        # extraction site builds. Do NOT copy its keys or bounds into Field
        # constraints here: a second copy drifts and load and extraction would
        # disagree. Lazy import because prompts imports this module.
        """
        if value is None:
            return value
        from .prompts import ClassificationPromptConfig

        try:
            ClassificationPromptConfig(**value)
        except TypeError as e:
            raise ValueError(f"invalid prompt_config: {e}") from e
        return value

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


def _walk_logic_nodes(node: Any, _depth: int = 0) -> Iterator[tuple[str, Any]]:
    """Yield every `(operator, arguments)` pair in a JsonLogic `node`, visiting
    exactly the positions `evaluate_logic` evaluates as logic.

    A JsonLogic object is a dict whose single key is an operator mapping to its
    argument value(s): a list/tuple of arguments, or one bare argument. Each
    argument is evaluated as logic one level deeper. A list or tuple met AS an
    argument is DATA (`evaluate_logic` returns it unevaluated), so its elements,
    dicts included, are never walked; nor are the raw arguments of the
    `JSONLOGIC_RAW_ARGUMENT_OPERATIONS` (`var`, `missing`, `missing_some`).

    # DECISION plan-2026-09-21T203800-8a03483a/D-014
    # This supersedes D-007's "recurse into list/tuple ELEMENTS" clause below
    # (B9): walking data-list elements rejected `{"in":[{"var":"x"},[{"foo":1}]]}`,
    # which evaluates fine. Do NOT recurse into data lists or raw-argument
    # operators again, and do NOT drop the depth or single-key checks (B5):
    # without them over-deep or multi-key logic loads clean and is silently
    # False at every turn (`evaluate_logic` raises, the evaluator swallows it).
    # The depth bound is `evaluate_logic`'s own (every evaluated position at
    # depth > MAX_JSONLOGIC_DEPTH raises there), so load and runtime agree.
    # D-007's empty-dict rejection is kept. See decisions.md D-014.

    # DECISION plan-2026-07-21T045419-9925aa3a/D-007
    # An EMPTY dict is malformed JsonLogic (an operator object with zero operator
    # keys), so this RAISES ValueError — that is how H6 rejects both a top-level
    # `logic: {}` and a nested `{"and": [{}]}`. Do NOT "soften" this to silently
    # skip empty dicts: `logic: {}` previously meant "always fail" while
    # `logic: null` means "always pass", a silent divergence this guard closes.
    # Do NOT recurse into a dict's KEYS (they are operators, already yielded) —
    # only into its VALUES and into list/tuple ELEMENTS [superseded by D-014:
    # only an operator's ARGUMENT list is walked, never a data list]. Primitives,
    # including the literal strings in a data-list arg such as
    # `{"in":[{"var":"x"},["a","b"]]}`, are DATA not operators and are ignored
    # (false-reject trap). This
    # helper lives here, NOT in expressions.py, because expressions.py imports
    # FROM definitions.py — the reverse edge would be a circular import.
    # See decisions.md D-007.

    Contract:
        node: a JsonLogic position that `evaluate_logic` evaluates (the root
            logic dict, or one operator argument).
        _depth: the `evaluate_logic` depth of `node` (0 for the root).
    Yields: each `(operator key, raw argument value)` pair encountered, in
        traversal order. Callers: `_walk_logic_operators` (load-time operator
        allow-list) and `logic_referenced_keys` (validator gating checks).
    Raises: ValueError if an evaluated position is deeper than
        MAX_JSONLOGIC_DEPTH, or an operator dict has zero keys (D-007) or more
        than one key.
    """
    if _depth > MAX_JSONLOGIC_DEPTH:
        raise ValueError(
            f"JsonLogic nesting exceeds the maximum depth ({MAX_JSONLOGIC_DEPTH})"
        )
    if not isinstance(node, dict):
        # Primitives and lists at an evaluated position are data.
        return
    if not node:
        raise ValueError("Empty JsonLogic object '{}' is not a valid condition")
    if len(node) != 1:
        raise ValueError(
            f"JsonLogic object has keys {sorted(map(str, node))}; each operator "
            "object must have exactly one key"
        )
    operator, arguments = next(iter(node.items()))
    yield operator, arguments
    if operator in JSONLOGIC_RAW_ARGUMENT_OPERATIONS:
        return
    if not isinstance(arguments, (list, tuple)):
        arguments = [arguments]
    for argument in arguments:
        yield from _walk_logic_nodes(argument, _depth + 1)


def _walk_logic_operators(node: Any, _depth: int = 0) -> Iterator[str]:
    """Yield every operator key in a JsonLogic `node` (see `_walk_logic_nodes`
    for which positions are visited and what raises)."""
    for operator, _arguments in _walk_logic_nodes(node, _depth):
        yield operator


def logic_referenced_keys(logic: Any) -> set[str]:
    """Return the top-level context keys a JsonLogic expression reads.

    A key counts when it is named literally by `var` (a dotted path counts as
    its first segment: `profile.email` reads `profile`), by `missing` /
    `missing_some`, or by the one-argument `has_context` shorthand. A value
    that only appears as a compared literal is not a reference, and a computed
    name (`{"var": {"cat": ...}}`) cannot be resolved statically and is skipped.

    Contract:
        logic: a `TransitionCondition.logic` value (dict, or None/falsy).
    Returns: the set of referenced first-segment key names (empty for None).
    Raises: ValueError exactly when `_walk_logic_nodes` does (malformed logic,
        which the loader already rejects).
    """
    keys: set[str] = set()
    if not logic:
        return keys

    def _add(name: Any) -> None:
        if isinstance(name, str) and name:
            keys.add(name.split(".", 1)[0])

    for operator, arguments in _walk_logic_nodes(logic):
        args = list(arguments) if isinstance(arguments, (list, tuple)) else [arguments]
        if operator == "var" and args:
            _add(args[0])
        elif operator == "missing":
            for arg in args:
                for name in arg if isinstance(arg, (list, tuple)) else [arg]:
                    _add(name)
        elif operator == "missing_some" and len(args) == 2:
            names = args[1]
            for name in names if isinstance(names, (list, tuple)) else [names]:
                _add(name)
        elif operator == "has_context" and len(args) == 1:
            _add(args[0])
    return keys


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
        default=None,
        description=(
            "Context keys required for evaluation; a key that is absent, None "
            "or an empty string is missing and fails the condition"
        ),
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

    @model_validator(mode="after")
    def _validate_logic(self) -> TransitionCondition:
        """Reject an empty/nested-empty (H6), non-allow-listed-operator (H8),
        multi-key or over-deep (B5, D-014) `logic` dict at LOAD time, so `API.from_file` and `fsm-llm-validate`
        both fail on a malformed condition instead of blowing up mid-conversation.

        # DECISION plan-2026-07-21T045419-9925aa3a/D-007
        # Raise a plain ValueError (surfaces as pydantic error type `value_error`,
        # already promoted to ERROR by validator.py's allow-list) so the
        # validator/loader agreement IFF is preserved. Do NOT switch this to a
        # different exception type or add `extra="forbid"` — either would break
        # `test_constraint_sweep_validator_agrees_with_loader`. `logic is None`
        # (absent) stays valid: it means "pass if required keys present".
        # See decisions.md D-007.
        """
        if self.logic is None:
            return self
        unknown = sorted(
            {
                op
                for op in _walk_logic_operators(self.logic)
                if op not in ALLOWED_JSONLOGIC_OPERATIONS
            }
        )
        if unknown:
            raise ValueError(
                f"Unsupported JsonLogic operator(s): {unknown}. "
                f"Allowed: {sorted(ALLOWED_JSONLOGIC_OPERATIONS)}"
            )
        return self


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
        pattern=ASCII_IDENTIFIER_PATTERN,
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


class ContextScope(BaseModel):
    """Which context keys a state's prompts see.

    # DECISION plan-2026-09-21T203800-8a03483a/D-015
    # A typed model with `extra="forbid"`, not a free-form dict. Do NOT loosen
    # it back to `dict[str, Any]` or to `extra="ignore"`: a bare-string
    # `read_keys` made scoping a substring test (`"u" in "username"`), and a
    # misspelt key (`read_key`) silently disabled scoping. `State.context_scope`
    # still accepts a plain dict, which pydantic validates into this model.
    # See decisions.md D-015.
    """

    model_config = ConfigDict(extra="forbid")

    read_keys: list[str] | None = Field(
        default=None,
        description=(
            "Keys included in this state's prompts; None or [] means all "
            "user-visible context"
        ),
    )
    write_keys: list[str] | None = Field(
        default=None,
        description="Keys this state is expected to produce (advisory, not enforced)",
    )


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
        pattern=ASCII_IDENTIFIER_PATTERN,
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

    context_scope: ContextScope | None = Field(
        default=None,
        description=(
            "Optional context scoping for this state (a ContextScope or a dict "
            "with only 'read_keys' and/or 'write_keys', each a list of strings). "
            "Controls which context keys are injected into LLM prompts. "
            "When None, all user-visible context is injected (default behavior)."
        ),
    )

    @field_validator("transition_classification")
    @classmethod
    def _validate_transition_classification(
        cls, value: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """Check the shape ``_build_transition_classification_schema`` reads.

        The reserved key ``TRANSITION_CLASSIFICATION_THRESHOLD_KEY`` maps to a
        real number (not a bool) in [0, 1]. Every other key names a target state
        and maps to a dict whose only allowed key is ``description`` (a str or
        None). None (auto mode) and an empty dict pass.
        """
        if value is None:
            return value
        for key, entry in value.items():
            if key == TRANSITION_CLASSIFICATION_THRESHOLD_KEY:
                if (
                    isinstance(entry, bool)
                    or not isinstance(entry, int | float)
                    or not 0.0 <= entry <= 1.0
                ):
                    raise ValueError(
                        f"transition_classification {key!r} must be a number "
                        f"within 0.0..1.0, got {entry!r}"
                    )
                continue
            if not isinstance(entry, dict):
                raise ValueError(
                    f"transition_classification {key!r} must be a dict like "
                    f'{{"description": "..."}}, got {entry!r}'
                )
            unknown = sorted(str(k) for k in set(entry) - {"description"})
            if unknown:
                raise ValueError(
                    f"transition_classification {key!r} has unknown keys "
                    f"{unknown}; only 'description' is allowed"
                )
            description = entry.get("description")
            if description is not None and not isinstance(description, str):
                raise ValueError(
                    f"transition_classification {key!r} 'description' must be "
                    f"a str, got {description!r}"
                )
        return value

    @model_validator(mode="after")
    def _validate_extraction_configs(self) -> State:
        """Reject state configs that can never work (B10).

        # DECISION plan-2026-09-21T203800-8a03483a/D-016
        # A plain ValueError (pydantic `value_error`, promoted to ERROR by
        # validator.py) so loader and `fsm-llm-validate` agree. Do NOT downgrade
        # these to warnings: two configs of one channel writing one
        # `field_name` race for the same context key, and an empty or
        # internal-prefixed required key is never extracted (internal keys are
        # filtered out of extraction), so the state can never be satisfied.
        # DECISION plan-2026-09-21T203800-8a03483a/D-033: duplicates are checked
        # per list. Do NOT extend the check ACROSS field_extractions and
        # classification_extractions: one explicit extraction named like a
        # classification field is the supported below-threshold fallback of
        # plan-2026-09-19T175721-21cd7f8e/D-006 (pipeline.py). See decisions.md
        # D-016, D-033.
        """
        for channel, configs in (
            ("field_extractions", self.field_extractions),
            ("classification_extractions", self.classification_extractions),
        ):
            names = [c.field_name for c in configs or []]
            duplicates = sorted({name for name in names if names.count(name) > 1})
            if duplicates:
                raise ValueError(
                    f"State '{self.id}': field_name(s) {duplicates} are declared "
                    f"more than once in {channel}"
                )
        bad_keys = [
            key
            for key in self.required_context_keys or []
            if not key.strip() or has_internal_prefix(key)
        ]
        if bad_keys:
            raise ValueError(
                f"State '{self.id}': required_context_keys {bad_keys} are empty or "
                "internal-prefixed and can never be extracted"
            )
        return self


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

    # DECISION plan-2026-09-19T175721-21cd7f8e/D-033: opt-in and FSM-wide. The
    # default stays empty because a default-closed rule (auto-minting every
    # key a transition reads) would freeze 36 of 49 shipped example FSMs. The
    # pipeline consults it at three filters (bulk return, per-field configs,
    # post-transition configs). A listed key that is also a
    # classification_extractions field is NOT covered (classification writes
    # are a separate channel).
    handler_only_keys: list[str] = Field(
        default_factory=list,
        description=(
            "Context keys only handlers, update_context and initial_context may "
            "write; never extracted from user text (opt-in, default empty)"
        ),
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
        # Trimming is deferred to add_system_message (turn completion) so a
        # rolled-back user turn isn't prematurely compressed into the summary.

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
            n: Number of exchange pairs (each pair = 1 user + 1 system message)
               to return. Returns up to ``n * 2`` individual messages. If fewer
               than ``n * 2`` messages exist, all are returned. Defaults to
               max_history_size.
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
            self._append_to_summary(self.exchanges)
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
    data from all non-hidden working memory buffers (flattened). The flat
    ``data`` dict remains the primary storage for backward compatibility.

    Prompt reach: non-hidden buffer data sits under ``data`` (``data`` wins
    on collision) in ``get_merged_data()``, which feeds the transition
    evaluator's working context and the Pass-2 ``<current_context>`` block at
    all three sites (``_execute_response_generation_pass``,
    ``_stream_response_generation_pass``, ``generate_initial_response``),
    scoped by ``read_keys``. ``get_user_visible_data()`` (the same merge
    without internal keys) feeds the Pass-1 per-field extraction prompt (a
    field whose ``context_keys`` is left at its default), both classifier
    call sites (through ``MessagePipeline._build_classifier_context``), and
    ``ResponseGenerationRequest.context``. The bulk extraction pass sees
    ``data`` only. ``update_context`` and handlers write ``data`` only. See
    ``fsm_llm.memory.WorkingMemory`` for the full statement.
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

    # Runtime: WorkingMemory | None (typed Any to avoid a circular import).
    # `exclude=True` means model_dump()/model_dump_json() omit this field
    # ENTIRELY (no key, not even None); model_copy(deep=True) and copy.deepcopy
    # do carry it. Persist working memory via WorkingMemory.to_dict(), as
    # save_session does -- not via a context dump.
    working_memory: Any = Field(
        default=None,
        description=(
            "Optional WorkingMemory instance for structured buffer-based "
            "context management. Non-hidden buffer data is merged under "
            "the flat data dict (data wins) for the transition evaluator, "
            "the Pass-2 prompt, the per-field extraction prompt and the "
            "classifiers; hidden buffers are never merged. Import from "
            "fsm_llm.memory."
        ),
        exclude=True,
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
                if has_internal_prefix(key):
                    logger.warning(
                        f"Context update contains internal-prefix key: {key!r}"
                    )
            logger.debug(f"Updating context with keys: {list(new_data.keys())}")
            self.data.update(new_data)

    def get_merged_data(self) -> dict[str, Any]:
        """Non-hidden WorkingMemory buffers under the flat ``data`` dict.

        Returns a new dict: ``working_memory.get_all_data()`` (hidden buffers
        never included) overlaid by ``data`` (``data`` wins on collision).
        Internal-prefix keys are NOT stripped, so the result is a drop-in
        for ``data`` wherever a consumer applies its own filter (the
        transition evaluator, the Pass-2 prompt builder). Without working
        memory, or with empty buffers, it equals ``dict(data)`` in content
        and key order. Never raises.
        """
        # DECISION plan-2026-09-21T203800-8a03483a/D-008: the one WM-under-data
        # merge for every decision point. Do NOT let a buffer value override
        # context.data (existing FSMs that never write WM must stay
        # byte-identical) and do NOT merge hidden buffers (get_all_data skips
        # them; they carry orchestration metadata that must never reach an LLM).
        if self.working_memory is not None and hasattr(
            self.working_memory, "get_all_data"
        ):
            merged: dict[str, Any] = self.working_memory.get_all_data()
            merged.update(self.data)
            return merged
        return dict(self.data)

    def get_user_visible_data(self) -> dict[str, Any]:
        """``get_merged_data()`` without internal-prefix keys or ``system``."""
        return {
            key: value
            for key, value in self.get_merged_data().items()
            if not has_internal_prefix(key) and key != "system"
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
        # DECISION plan-2026-07-19T191147-4b664252/D-025 [STALE]: this check is a
        # `model_validator` raising ValueError ON PURPOSE. Do NOT "simplify" it to
        # `Field(pattern=_ASCII_IDENTIFIER.pattern)` to match `State.id` /
        # `Transition.target_state`, even though that reads cleaner and enforces the
        # identical charset. A `pattern=` violation is a `string_pattern_mismatch`
        # error, and `validator.py`'s ALLOW-list (see D-013) deliberately EXCLUDES
        # that type -- so the swap would make `fsm-llm-validate` report is_valid=True
        # on an FSM that `API.from_file` then refuses to load. A ValueError from a
        # `model_validator` is a `value_error`, which the ALLOW-list DOES promote to
        # ERROR tier. The regex is shared; the enforcement mechanism must not be.
        if not _ASCII_IDENTIFIER.match(self.name):
            raise ValueError(
                f"Intent name must be alphanumeric ASCII with underscores, "
                f"got '{self.name}'"
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


def _coerce_entity_values(v: Any) -> dict[str, str | None]:
    """Coerce raw LLM ``entities`` into ``{name: str | None}``.

    Shared by the ``IntentScore`` and ``ClassificationResult`` validators.
    A non-dict becomes ``{}``; a list joins with ``", "``; ``None`` stays
    ``None``; anything else is ``str()``'d. Never raises.
    """
    # DECISION plan-2026-09-22T080837-8b258a25/D-010 (supersedes 80b0bd4d D-010)
    # Both entity validators MUST call this one function; do not inline a copy.
    # Do NOT map None to str(None): "None" is truthy and defeats a handler's
    # `if entities.get(k):` check. Do NOT narrow `entities` to dict[str, str].
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


class IntentScore(BaseModel):
    """A single scored intent within a classification result."""

    intent: str = Field(description="The classified intent name")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Model confidence in this classification"
    )
    entities: dict[str, str | None] = Field(
        default_factory=dict, description="Extracted entities relevant to this intent"
    )

    @field_validator("entities", mode="before")
    @classmethod
    def coerce_entity_values(cls, v: Any) -> dict[str, str | None]:
        return _coerce_entity_values(v)


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
        return _coerce_entity_values(v)

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
        max_length=MAX_MULTI_INTENTS,
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


class ConversationBusyError(FSMError):
    """A turn held the conversation's lock past
    ``END_CONVERSATION_LOCK_TIMEOUT_SECONDS``, so ending it was refused.
    Nothing was changed; retry after the turn completes."""

    def __init__(self, message: str, conversation_id: str | None = None, **kwargs):
        super().__init__(message, **kwargs)
        self.conversation_id = conversation_id


class FSMDefinitionNotFoundError(FSMError, ValueError):
    """No FSM definition is resolvable for a non-path id.

    Raised by ``utilities.load_fsm_definition`` (and so by the ``API`` loader
    fallback) for an id that is neither a file path nor cached: there is no
    FSM registry. Also a ``ValueError`` so existing handlers keep catching it.
    """

    def __init__(self, fsm_id: str):
        super().__init__(
            f"Unknown FSM ID '{fsm_id}': no FSM registry: only file paths are "
            f"loadable; id '{fsm_id}' is not cached",
            details={"fsm_id": fsm_id},
        )
        self.fsm_id = fsm_id

    def __reduce__(self):
        # Pickling replays ``cls(*self.args)``; ``args`` holds the formatted
        # message, not ``fsm_id``. Rebuild from the real argument.
        return (self.__class__, (self.fsm_id,), self.__dict__.copy())


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
