"""
LLM-FSM Version 3: Improved Finite State Machine implementation for Large Language Models.
Now with LiteLLM integration for simplified access to multiple LLM providers.
Includes logging for better debugging and monitoring.

This module provides the core framework for implementing FSMs with LLMs,
leveraging the LLM's natural language understanding capabilities.
"""

import json
from pydantic import BaseModel, Field, model_validator
from typing import Dict, List, Optional, Any, Union, Callable, Tuple

# --------------------------------------------------------------
# local imports
# --------------------------------------------------------------

from .logging import logger
from .constants import (
    DEFAULT_MESSAGE_TRUNCATE_LENGTH,
    DEFAULT_MAX_HISTORY_SIZE,
    DEFAULT_MAX_MESSAGE_LENGTH
)

# --------------------------------------------------------------


class TransitionCondition(BaseModel):
    """
    Defines a condition for a state transition.

    Attributes:
        description: Human-readable description of the condition
        requires_context_keys: List of context keys that must be present
        logic: JsonLogic expression to evaluate against context data
    """
    description: str = Field(..., description="Human-readable description of the condition")
    requires_context_keys: Optional[List[str]] = Field(
        default=None,
        description="Context keys required to be present"
    )
    logic: Optional[Dict[str, Any]] = Field(
        default=None,
        description="JsonLogic expression to evaluate against context"
    )


class Emission(BaseModel):
    """
    Defines an emission that the LLM should output when in a particular state.

    Attributes:
        description: Human-readable description of what should be emitted
        instruction: Additional instruction for the LLM about the emission
    """
    description: str = Field(..., description="Description of the emission")
    instruction: Optional[str] = Field(None, description="Additional instruction for the LLM")


class Transition(BaseModel):
    """
    Defines a transition between states.

    Attributes:
        target_state: The state to transition to
        description: Human-readable description of when this transition should occur
        conditions: Optional conditions that must be met
        priority: Priority of this transition (lower numbers have higher priority)
    """
    target_state: str = Field(..., description="Target state identifier")
    description: str = Field(..., description="Description of when this transition should occur")
    conditions: Optional[List[TransitionCondition]] = Field(
        default=None,
        description="Conditions for transition"
    )
    priority: int = Field(default=100, description="Priority (lower = higher)")


class State(BaseModel):
    """
    Defines a state in the FSM.

    Attributes:
        id: Unique identifier for the state
        description: Human-readable description of the state
        purpose: The purpose of this state (what information to collect or action to take)
        transitions: Available transitions from this state
        required_context_keys: Context keys that should be collected in this state
        instructions: Instructions for the LLM in this state
        example_dialogue: Example dialogue for this state
    """
    id: str = Field(..., description="Unique identifier for the state")
    description: str = Field(..., description="Human-readable description of the state")
    purpose: str = Field(..., description="The purpose of this state")
    transitions: List[Transition] = Field(default_factory=list, description="Available transitions")
    required_context_keys: Optional[List[str]] = Field(
        default=None,
        description="Context keys to collect"
    )
    instructions: Optional[str] = Field(None, description="Instructions for the LLM")
    example_dialogue: Optional[List[Dict[str, str]]] = Field(None, description="Example dialogue")


class FunctionHandler(BaseModel):
    """Definition of a function handler that can be triggered during FSM execution.

    Function handlers allow integrating external systems with the FSM,
    such as databases, APIs, or validation services.

    :param name: Unique identifier for the function handler
    :param description: Human-readable description of the handler's purpose
    :param trigger_on: List of events that will trigger this handler
    :param states: States where this handler applies (None means all states)
    :param function: Callable function to execute when triggered
    """
    name: str
    description: str
    trigger_on: List[str]  # List of events: "pre_transition", "post_transition", "context_update"
    states: Optional[List[str]] = None  # If None, applies to all states
    function: Optional[Callable] = None  # Not stored in JSON definition

    class Config:
        arbitrary_types_allowed = True


class FSMDefinition(BaseModel):
    """
    Complete definition of a Finite State Machine.

    Attributes:
        name: Name of the FSM
        description: Human-readable description
        states: All states in the FSM
        initial_state: The starting state
        version: Version of the FSM definition
        persona: Optional description of the persona/tone to use throughout the conversation
        function_handlers: Optional list of function handlers for this FSM
    """
    name: str = Field(..., description="Name of the FSM")
    description: str = Field(..., description="Human-readable description")
    states: Dict[str, State] = Field(..., description="All states in the FSM")
    initial_state: str = Field(..., description="The starting state identifier")
    version: str = Field(default="3.0", description="Version of the FSM definition")
    persona: Optional[str] = Field(None, description="Optional persona/tone to use for responses")
    function_handlers: Optional[List[FunctionHandler]] = Field([], description="Optional list of function handlers for this FSM")

    @model_validator(mode='after')
    def validate_states(self) -> 'FSMDefinition':
        """
        Validates that:
        1. The initial state exists
        2. All target states in transitions exist
        3. No orphaned states
        4. At least one terminal state exists
        5. At least one terminal state is reachable from the initial state

        Returns:
            The validated FSM definition

        Raises:
            ValueError: If any validation fails
        """
        logger.debug(f"Validating FSM definition: {self.name}")

        # Check initial state exists
        if self.initial_state not in self.states:
            error_msg = f"Initial state '{self.initial_state}' not found in states"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Identify terminal states (those with no outgoing transitions)
        terminal_states = {
            state_id for state_id, state in self.states.items()
            if not state.transitions
        }

        # Check that at least one terminal state exists
        if not terminal_states:
            error_msg = "FSM has no terminal states. At least one state must have no outgoing transitions."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Find all states reachable from the initial state
        reachable_states = {self.initial_state}
        change_made = True
        while change_made:
            change_made = False
            for state_id, state in self.states.items():
                if state_id in reachable_states:
                    for transition in state.transitions:
                        if transition.target_state not in reachable_states:
                            reachable_states.add(transition.target_state)
                            change_made = True

        # Check all target states exist
        for state_id, state in self.states.items():
            for transition in state.transitions:
                if transition.target_state not in self.states:
                    error_msg = f"Transition from '{state_id}' to non-existent state '{transition.target_state}'"
                    logger.error(error_msg)
                    raise ValueError(error_msg)

        # Check for orphaned states (states not reachable from initial state)
        orphaned_states = set(self.states.keys()) - reachable_states
        if orphaned_states:
            states_str = ", ".join(orphaned_states)
            error_msg = f"Orphaned states detected: {states_str}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Check that at least one terminal state is reachable
        reachable_terminal_states = terminal_states.intersection(reachable_states)
        if not reachable_terminal_states:
            terminal_str = ", ".join(terminal_states)
            error_msg = f"No terminal states are reachable from the initial state. Terminal states: {terminal_str}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"FSM definition validated successfully: {self.name}")
        logger.info(f"Reachable terminal states: {', '.join(reachable_terminal_states)}")
        return self


class Conversation(BaseModel):
    """
    Conversation history.

    Attributes:
        exchanges: List of conversation exchanges
        max_history_size: Maximum number of exchanges to track in history
        max_message_length: Maximum length of a message (soft cap)
    """
    exchanges: List[Dict[str, str]] = Field(default_factory=list, description="Conversation exchanges")
    max_history_size: int = Field(default=DEFAULT_MAX_HISTORY_SIZE, description="Maximum number of exchanges to keep in history")
    max_message_length: int = Field(default=DEFAULT_MAX_MESSAGE_LENGTH, description="Maximum length of a message")

    def add_user_message(self, message: str) -> None:
        """
        Add a user message to the conversation, truncating if needed.

        Args:
            message: The user's message
        """
        truncated = False
        if len(message) > self.max_message_length:
            message = message[:self.max_message_length] + "... [truncated]"
            truncated = True

        logger.debug(f"Adding user message: {message[:DEFAULT_MESSAGE_TRUNCATE_LENGTH]}{'...' if len(message) > DEFAULT_MESSAGE_TRUNCATE_LENGTH else ''}")
        if truncated:
            logger.debug(f"Message was truncated to {self.max_message_length} characters")

        self.exchanges.append({"user": message})

        # Trim history if it exceeds the maximum size
        if len(self.exchanges) > self.max_history_size * 2:  # *2 because each exchange has user and system
            excess = len(self.exchanges) - self.max_history_size * 2
            self.exchanges = self.exchanges[excess:]
            logger.debug(f"Trimmed {excess} old messages from conversation history")

    def add_system_message(self, message: str) -> None:
        """
        Add a system message to the conversation, truncating if needed.

        Args:
            message: The system's message
        """
        truncated = False
        if len(message) > self.max_message_length:
            message = message[:self.max_message_length] + "... [truncated]"
            truncated = True

        logger.debug(f"Adding system message: {message[:50]}{'...' if len(message) > 50 else ''}")
        if truncated:
            logger.debug(f"Message was truncated to {self.max_message_length} characters")

        self.exchanges.append({"system": message})

    def get_recent(self, n: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Get the n most recent exchanges.

        Args:
            n: Number of exchanges to retrieve (uses max_history_size if None)

        Returns:
            List of recent exchanges
        """
        if n is None:
            n = self.max_history_size

        return self.exchanges[-n * 2:] if n > 0 else []  # *2 to account for user+system pairs

class FSMContext(BaseModel):
    """
    Runtime context for an FSM instance.

    Attributes:
        data: Context data collected during the conversation
        conversation: Conversation history
        metadata: Additional metadata
    """
    data: Dict[str, Any] = Field(default_factory=dict, description="Context data")
    conversation: Conversation = Field(default_factory=Conversation, description="Conversation history")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def __init__(self, **data):
        # Set defaults for conversation configuration if not provided
        if "conversation" not in data:
            max_history = data.pop("max_history_size", DEFAULT_MAX_HISTORY_SIZE)
            max_message_length = data.pop("max_message_length", DEFAULT_MAX_MESSAGE_LENGTH)
            data["conversation"] = Conversation(
                max_history_size=max_history,
                max_message_length=max_message_length
            )
        super().__init__(**data)

    def update(self, new_data: Dict[str, Any]) -> None:
        """
        Update the context data.

        Args:
            new_data: New data to add to the context
        """
        if new_data:
            logger.debug(f"Updating context with new data: {json.dumps(new_data)}")
            self.data.update(new_data)

    def has_keys(self, keys: List[str]) -> bool:
        """
        Check if all specified keys exist in the context data.

        Args:
            keys: List of keys to check

        Returns:
            True if all keys exist, False otherwise
        """
        if not keys:
            return True
        result = all(key in self.data for key in keys)
        logger.debug(f"Checking context for keys: {keys} - Result: {result}")
        return result

    def get_missing_keys(self, keys: List[str]) -> List[str]:
        """
        Get keys that are missing from the context data.

        Args:
            keys: List of keys to check

        Returns:
            List of missing keys
        """
        if not keys:
            return []
        missing = [key for key in keys if key not in self.data]
        if missing:
            logger.debug(f"Missing context keys: {missing}")
        return missing


class FSMInstance(BaseModel):
    """
    Runtime instance of an FSM.

    Attributes:
        fsm_id: ID of the FSM definition
        current_state: Current state identifier
        context: Runtime context
        persona: Optional persona description from the FSM definition
    """
    fsm_id: str = Field(..., description="ID of the FSM definition")
    current_state: str = Field(..., description="Current state identifier")
    context: FSMContext = Field(default_factory=FSMContext, description="Runtime context")
    persona: Optional[str] = Field(None, description="Optional persona for response tone/style")


class StateTransition(BaseModel):
    """
    Defines a state transition decision.

    Attributes:
        target_state: The state to transition to
        context_update: Updates to the context data
    """
    target_state: str = Field(..., description="Target state identifier")
    context_update: Dict[str, Any] = Field(
        default_factory=dict,
        description="Updates to the context data"
    )

class LLMRequest(BaseModel):
    """
    A request to the LLM.

    Attributes:
        system_prompt: The system prompt
        user_message: The user's message
        context: Optional context information
    """
    system_prompt: str = Field(..., description="System prompt for the LLM")
    user_message: str = Field(..., description="User message")
    context: Optional[Dict[str, Any]] = Field(None, description="Optional context information")


class LLMResponse(BaseModel):
    """
    A response from the LLM.

    Attributes:
        transition: The state transition to perform
        message: The message for the user
        reasoning: Explanation of the decision
    """
    transition: StateTransition = Field(..., description="State transition to perform")
    message: str = Field(..., description="Message for the user")
    reasoning: Optional[str] = Field(None, description="Explanation of the decision")


class LLMResponseSchema(BaseModel):
    """
    Schema for the structured JSON output from the LLM.

    This is used with LiteLLM's json_schema support to ensure
    consistent parsing of LLM outputs.
    """
    transition: StateTransition = Field(..., description="State transition to perform")
    message: str = Field(..., description="Message for the user")
    reasoning: Optional[str] = Field(None, description="Explanation of the decision")

class FSMError(Exception):
    """Base exception for FSM errors."""
    pass


class StateNotFoundError(FSMError):
    """Exception raised when a state is not found."""
    pass


class InvalidTransitionError(FSMError):
    """Exception raised when a transition is invalid."""
    pass


class LLMResponseError(FSMError):
    """Exception raised when an LLM response is invalid."""
    pass