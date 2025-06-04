"""
Enhanced LLM-FSM definitions with improved 2-pass architecture.

This module defines the core data structures for a refined 2-pass LLM-FSM system:
1. Pass 1: Data extraction + transition evaluation
2. Pass 2: Response generation based on final state

Key Changes:
- Separate data extraction from response generation
- Response generation occurs after transition evaluation
- Enhanced request/response models for each pass
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, model_validator

# --------------------------------------------------------------
# Local imports
# --------------------------------------------------------------

from .logging import logger
from .constants import (
    DEFAULT_MAX_HISTORY_SIZE,
    DEFAULT_MAX_MESSAGE_LENGTH
)


# --------------------------------------------------------------
# Enums for LLM Request Types
# --------------------------------------------------------------

class LLMRequestType(str, Enum):
    """Types of requests that can be sent to LLM."""
    DATA_EXTRACTION = "data_extraction"
    RESPONSE_GENERATION = "response_generation"
    TRANSITION_DECISION = "transition_decision"


class TransitionEvaluationResult(str, Enum):
    """Results of transition evaluation."""
    DETERMINISTIC = "deterministic"  # Clear single transition
    AMBIGUOUS = "ambiguous"  # Multiple valid transitions, need LLM
    BLOCKED = "blocked"  # No valid transitions available


# --------------------------------------------------------------
# Data Extraction Models (Pass 1)
# --------------------------------------------------------------

class DataExtractionRequest(BaseModel):
    """
    Request for extracting data from user input without generating response.

    This request focuses purely on understanding and extracting information
    from user input without generating any user-facing content.
    """

    system_prompt: str = Field(
        ...,
        description="Prompt focused on data extraction and understanding",
        min_length=1,
        max_length=15000
    )

    user_message: str = Field(
        ...,
        description="User input to analyze and extract data from",
        min_length=0,
        max_length=10000
    )

    context: Optional[Dict[str, Any]] = Field(
        None,
        description="Current conversation context for extraction guidance"
    )


class DataExtractionResponse(BaseModel):
    """
    Response from data extraction containing only extracted information.

    No user-facing message is generated at this stage.
    """

    extracted_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Data extracted from user input"
    )

    confidence: float = Field(
        default=1.0,
        description="Confidence in the extraction (0.0-1.0)",
        ge=0.0,
        le=1.0
    )

    reasoning: Optional[str] = Field(
        None,
        description="Internal reasoning for debugging (not shown to user)",
        max_length=1000
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
        max_length=20000
    )

    user_message: str = Field(
        ...,
        description="Original user message for context",
        max_length=10000
    )

    extracted_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Data extracted in Pass 1"
    )

    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Current conversation context"
    )

    transition_occurred: bool = Field(
        default=False,
        description="Whether a state transition occurred"
    )

    previous_state: Optional[str] = Field(
        None,
        description="Previous state if transition occurred"
    )


class ResponseGenerationResponse(BaseModel):
    """
    Response containing the final user-facing message.

    Generated after all data extraction and transitions are complete.
    """

    message: str = Field(
        ...,
        description="Final user-facing message",
        min_length=1,
        max_length=5000
    )

    reasoning: Optional[str] = Field(
        None,
        description="Internal reasoning for debugging",
        max_length=1000
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
        max_length=100
    )

    description: str = Field(
        ...,
        description="Human-readable description of when this transition applies",
        min_length=1,
        max_length=500
    )

    priority: int = Field(
        default=100,
        description="Priority for this transition (lower = higher priority)",
        ge=0,
        le=1000
    )


class TransitionDecisionRequest(BaseModel):
    """
    Request for deciding between multiple valid transition options.

    Used only when deterministic evaluation results in ambiguity.
    """

    system_prompt: str = Field(
        ...,
        description="Focused prompt for transition decision making only",
        min_length=1,
        max_length=10000
    )

    current_state: str = Field(
        ...,
        description="Current state identifier",
        min_length=1,
        max_length=100
    )

    available_transitions: List[TransitionOption] = Field(
        ...,
        description="Valid transition options to choose from",
        min_length=1
    )

    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Relevant context for transition decision"
    )

    user_message: str = Field(
        ...,
        description="Original user message that triggered evaluation",
        max_length=10000
    )

    extracted_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Data extracted from user input"
    )


class TransitionDecisionResponse(BaseModel):
    """
    Response containing transition decision from LLM.

    Simple, focused response for transition selection only.
    """

    selected_transition: str = Field(
        ...,
        description="Target state identifier for selected transition",
        min_length=1,
        max_length=100
    )

    reasoning: Optional[str] = Field(
        None,
        description="Reasoning for transition choice (for debugging)",
        max_length=1000
    )


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
        max_length=500
    )

    requires_context_keys: Optional[List[str]] = Field(
        default=None,
        description="Context keys required for evaluation"
    )

    logic: Optional[Dict[str, Any]] = Field(
        default=None,
        description="JsonLogic expression for complex evaluation"
    )

    evaluation_priority: int = Field(
        default=100,
        description="Priority for condition evaluation (lower = earlier)",
        ge=0,
        le=1000
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
        max_length=100
    )

    description: str = Field(
        ...,
        description="When this transition should occur",
        min_length=1,
        max_length=500
    )

    conditions: Optional[List[TransitionCondition]] = Field(
        default=None,
        description="Conditions that must be satisfied"
    )

    priority: int = Field(
        default=100,
        description="Priority for transition selection",
        ge=0,
        le=1000
    )

    is_deterministic: bool = Field(
        default=True,
        description="Whether this transition can be evaluated deterministically"
    )

    llm_description: Optional[str] = Field(
        None,
        description="Description for LLM when choosing between transitions",
        max_length=300
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
        pattern=r'^[a-zA-Z_][a-zA-Z0-9_]*$'
    )

    description: str = Field(
        ...,
        description="Human-readable state description",
        min_length=1,
        max_length=300
    )

    purpose: str = Field(
        ...,
        description="What should be accomplished in this state",
        min_length=1,
        max_length=500
    )

    extraction_instructions: Optional[str] = Field(
        None,
        description="Instructions for data extraction in this state",
        max_length=2000
    )

    response_instructions: Optional[str] = Field(
        None,
        description="Instructions for response generation in this state",
        max_length=2000
    )

    transitions: List[Transition] = Field(
        default_factory=list,
        description="Available transitions from this state"
    )

    required_context_keys: Optional[List[str]] = Field(
        default=None,
        description="Context keys that should be collected"
    )

    auto_transition_threshold: Optional[float] = Field(
        None,
        description="Confidence threshold for automatic transitions (0.0-1.0)",
        ge=0.0,
        le=1.0
    )

    response_type: str = Field(
        default="conversational",
        description="Type of response to generate: conversational, form, confirmation, etc."
    )


# --------------------------------------------------------------
# FSM Definition Models
# --------------------------------------------------------------

class FSMDefinition(BaseModel):
    """
    Complete FSM definition for improved 2-pass architecture.

    Enhanced with separate extraction and response capabilities.
    """

    name: str = Field(
        ...,
        description="FSM name identifier",
        min_length=1,
        max_length=100
    )

    description: str = Field(
        ...,
        description="FSM purpose and functionality description",
        min_length=1,
        max_length=1000
    )

    states: Dict[str, State] = Field(
        ...,
        description="All states in the FSM",
        min_length=1
    )

    initial_state: str = Field(
        ...,
        description="Starting state identifier",
        min_length=1
    )

    version: str = Field(
        default="4.1",
        description="FSM definition version",
        min_length=1,
        max_length=20
    )

    persona: Optional[str] = Field(
        None,
        description="Conversation persona for response generation",
        max_length=500
    )

    transition_evaluation_mode: str = Field(
        default="hybrid",
        description="Transition evaluation strategy: 'deterministic', 'llm', or 'hybrid'"
    )

    @model_validator(mode='after')
    def validate_fsm_structure(self) -> 'FSMDefinition':
        """Comprehensive FSM validation for improved 2-pass architecture."""
        logger.debug(f"Validating FSM: {self.name}")

        # Basic structure validation
        if self.initial_state not in self.states:
            raise ValueError(f"Initial state '{self.initial_state}' not found in states")

        # Validate all transitions
        for state_id, state in self.states.items():
            for transition in state.transitions:
                if transition.target_state not in self.states:
                    raise ValueError(
                        f"Invalid transition from '{state_id}' to non-existent state '{transition.target_state}'"
                    )

        # Terminal state validation
        terminal_states = {
            state_id for state_id, state in self.states.items()
            if not state.transitions
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
        to_process = [self.initial_state]

        while to_process:
            current = to_process.pop(0)
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

    exchanges: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Conversation history in chronological order"
    )

    max_history_size: int = Field(
        default=DEFAULT_MAX_HISTORY_SIZE,
        description="Maximum conversation exchanges to retain",
        ge=0,
        le=1000
    )

    max_message_length: int = Field(
        default=DEFAULT_MAX_MESSAGE_LENGTH,
        description="Maximum message length in characters",
        ge=1,
        le=50000
    )

    def add_user_message(self, message: str) -> None:
        """Add user message with automatic truncation."""
        if len(message) > self.max_message_length:
            message = message[:self.max_message_length] + "... [truncated]"

        self.exchanges.append({"user": message})
        self._maintain_history_size()

    def add_system_message(self, message: str) -> None:
        """Add system message with automatic truncation."""
        if len(message) > self.max_message_length:
            message = message[:self.max_message_length] + "... [truncated]"

        self.exchanges.append({"system": message})
        self._maintain_history_size()

    def get_recent(self, n: Optional[int] = None) -> List[Dict[str, str]]:
        """Get recent conversation exchanges."""
        if n is None:
            n = self.max_history_size

        if n <= 0:
            return []

        return self.exchanges[-n * 2:] if n * 2 > 0 else []

    def _maintain_history_size(self) -> None:
        """Maintain conversation history within size limits."""
        limit = self.max_history_size * 2
        if len(self.exchanges) > limit:
            excess = len(self.exchanges) - limit
            self.exchanges = self.exchanges[excess:]


class FSMContext(BaseModel):
    """Enhanced context management for improved 2-pass architecture."""

    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Conversation context data"
    )

    conversation: Conversation = Field(
        default_factory=Conversation,
        description="Conversation history management"
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="System metadata and operational data"
    )

    def __init__(self, **data):
        """Initialize with optional conversation configuration."""
        if "conversation" not in data:
            max_history = data.pop("max_history_size", DEFAULT_MAX_HISTORY_SIZE)
            max_message_length = data.pop("max_message_length", DEFAULT_MAX_MESSAGE_LENGTH)

            data["conversation"] = Conversation(
                max_history_size=max_history,
                max_message_length=max_message_length
            )

        super().__init__(**data)

    def update(self, new_data: Dict[str, Any]) -> None:
        """Update context with new data."""
        if new_data:
            logger.debug(f"Updating context with keys: {list(new_data.keys())}")
            self.data.update(new_data)

    def has_keys(self, keys: List[str]) -> bool:
        """Check if all specified keys exist."""
        if not keys:
            return True
        return all(key in self.data for key in keys)

    def get_missing_keys(self, keys: List[str]) -> List[str]:
        """Get list of missing required keys."""
        if not keys:
            return []
        return [key for key in keys if key not in self.data]

    def get_user_visible_data(self) -> Dict[str, Any]:
        """Get context data filtered for user visibility."""
        return {
            key: value for key, value in self.data.items()
            if not key.startswith('_') and not key.startswith('system_')
        }


class FSMInstance(BaseModel):
    """Enhanced FSM instance for improved 2-pass architecture."""

    fsm_id: str = Field(
        ...,
        description="FSM definition identifier",
        min_length=1,
        max_length=100
    )

    current_state: str = Field(
        ...,
        description="Current state identifier",
        min_length=1,
        max_length=100
    )

    context: FSMContext = Field(
        default_factory=FSMContext,
        description="Conversation context and history"
    )

    persona: Optional[str] = Field(
        default="Helpful AI assistant",
        description="Conversation persona",
        max_length=500
    )

    last_extraction_response: Optional[DataExtractionResponse] = Field(
        None,
        description="Last data extraction response for debugging"
    )

    last_transition_decision: Optional[TransitionDecisionResponse] = Field(
        None,
        description="Last transition decision for debugging"
    )

    last_response_generation: Optional[ResponseGenerationResponse] = Field(
        None,
        description="Last response generation for debugging"
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
        ...,
        description="Type of evaluation result"
    )

    deterministic_transition: Optional[str] = Field(
        None,
        description="Target state if deterministically determined"
    )

    available_options: List[TransitionOption] = Field(
        default_factory=list,
        description="Available transition options if ambiguous"
    )

    blocked_reason: Optional[str] = Field(
        None,
        description="Reason if transitions are blocked"
    )

    confidence: float = Field(
        default=0.0,
        description="Confidence in the evaluation result (0.0-1.0)",
        ge=0.0,
        le=1.0
    )


# --------------------------------------------------------------
# Exception Classes
# --------------------------------------------------------------

class FSMError(Exception):
    """Base exception for FSM-related errors."""
    pass


class StateNotFoundError(FSMError):
    """Exception for non-existent state references."""
    pass


class InvalidTransitionError(FSMError):
    """Exception for invalid state transitions."""
    pass


class LLMResponseError(FSMError):
    """Exception for LLM response processing errors."""
    pass


class TransitionEvaluationError(FSMError):
    """Exception for transition evaluation errors."""
    pass