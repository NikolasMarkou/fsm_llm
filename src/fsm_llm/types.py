from __future__ import annotations

"""Neutral types layer for fsm_llm — runtime-safe Pydantic models + exceptions.

Hosts the request/response models, enums, and exception hierarchy that the
λ-runtime kernel (``fsm_llm.runtime.oracle``, ``runtime._litellm``,
``runtime.errors``), the handler system (``fsm_llm.handlers``), and the
stdlib subpackages (``fsm_llm.stdlib.{agents,reasoning,workflows}.exceptions``,
``stdlib.agents.meta_builder``, ``stdlib.agents.meta_builders``) all need to
share — without forcing those layers to reach into ``fsm_llm.dialog``.

Introduced in 0.7.0 to dissolve the kernel↔dialog import coupling that the
layering audit (``tests/test_fsm_llm/test_layering.py``) had to allow-list at
five entries through 0.6.x. After this move, ``runtime/`` and ``stdlib/``
import models from ``fsm_llm.types`` directly; ``fsm_llm.dialog.definitions``
re-exports the same names for byte-equivalent back-compat — the canonical
home is here.

What's in this module:

- **Enums**: ``LLMRequestType``, ``TransitionEvaluationResult``.
- **Request/response models** (used by ``runtime.oracle``, ``runtime._litellm``,
  the prompt builders, and the FSM compiler): ``DataExtractionResponse``,
  ``ResponseGenerationRequest``, ``ResponseGenerationResponse``,
  ``FieldExtractionRequest``, ``FieldExtractionResponse``.
- **Exception hierarchy** (used by the kernel, handlers, and every stdlib
  subpackage's ``exceptions.py``): ``FSMError`` and its eight subclasses.

What stays in ``fsm_llm.dialog.definitions``:

- The FSM authoring schema (``State``, ``Transition``, ``TransitionCondition``,
  ``FSMDefinition``, ``FSMContext``, ``FSMInstance``, ``Conversation``,
  ``TransitionEvaluation``, ``TransitionOption``).
- Field/classification extraction *config* models that depend on FSM-domain
  models (``FieldExtractionConfig``, ``ClassificationExtractionConfig``,
  ``IntentDefinition``, ``ClassificationSchema``, ``ClassificationResult``,
  ``IntentScore``, ``MultiClassificationResult``, ``DomainSchema``,
  ``HierarchicalSchema``, ``HierarchicalResult``).

These dialog-domain models continue to live in ``dialog/definitions.py`` —
they are a *consumer* of this types layer, not a peer.
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


# ---------------------------------------------------------------------------
# Response Generation Models (Pass 2)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Field Extraction Request/Response (the runtime-touching pair).
#
# ``FieldExtractionConfig`` (the FSM authoring-time config) stays in
# ``dialog/definitions.py`` — it's an FSM-domain model authored on a State.
# The Request/Response pair below is what the runtime oracle actually sees.
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
