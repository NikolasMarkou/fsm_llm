from __future__ import annotations

"""Private runtime-touching Pydantic models + enums.

Hosts the request/response models and enums that the λ-runtime kernel,
handlers, and stdlib subpackages share. Renamed from ``fsm_llm.types`` at
0.9.0 (which itself was introduced at 0.7.0 to dissolve the kernel↔dialog
import coupling).

This module is **private**. Do not import from it directly:

- For exceptions: use ``fsm_llm.errors`` (or the top-level ``fsm_llm.FSMError`` /
  ``fsm_llm.LambdaError`` for the roots).
- For request/response models: import them from where the consumer lives —
  ``fsm_llm.runtime`` for kernel-touching models, or the appropriate stdlib
  subpackage for stdlib-touching ones. Direct imports of these models from
  user code are unusual; they appear in oracle invocations and the FSM 2-pass
  pipeline as internal plumbing.

What's in this module:

- **Enums**: ``LLMRequestType``, ``TransitionEvaluationResult``.
- **Request/response models**: ``DataExtractionResponse``,
  ``ResponseGenerationRequest``, ``ResponseGenerationResponse``,
  ``FieldExtractionRequest``, ``FieldExtractionResponse``.
- **Exception hierarchy**: ``FSMError`` and its eight subclasses. These
  are re-exported from ``fsm_llm.errors`` (the public path).
"""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Data Extraction Models (Pass 1)
# ---------------------------------------------------------------------------


class DataExtractionResponse(BaseModel):
    """Response from data extraction containing only extracted information."""

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


# ---------------------------------------------------------------------------
# Response Generation Models (Pass 2)
# ---------------------------------------------------------------------------


class ResponseGenerationRequest(BaseModel):
    """Request for generating user-facing response based on final state."""

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
    """Response containing the final user-facing message."""

    message: str = Field(..., description="Final user-facing message", max_length=5000)

    reasoning: str | None = Field(
        None, description="Internal reasoning for debugging", max_length=5000
    )

    message_type: str = Field(
        default="response", description="Type of the response message"
    )


# ---------------------------------------------------------------------------
# Field Extraction Request/Response (the runtime-touching pair).
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Exception Hierarchy
# ---------------------------------------------------------------------------


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


__all__ = [
    # Enums
    "LLMRequestType",
    "TransitionEvaluationResult",
    # Request/response models
    "DataExtractionResponse",
    "ResponseGenerationRequest",
    "ResponseGenerationResponse",
    "FieldExtractionRequest",
    "FieldExtractionResponse",
    # Exception hierarchy
    "FSMError",
    "StateNotFoundError",
    "InvalidTransitionError",
    "LLMResponseError",
    "TransitionEvaluationError",
    "ClassificationError",
    "SchemaValidationError",
    "ClassificationResponseError",
]
