"""
LLM-FSM Version 3: Improved Finite State Machine implementation for Large Language Models.
Now with LiteLLM integration for simplified access to multiple LLM providers.
Includes logging for better debugging and monitoring.

This module provides the core framework for implementing FSMs with LLMs,
leveraging the LLM's natural language understanding capabilities.
"""

import re
import os
import time
import json
import uuid
import litellm
from pydantic import BaseModel, Field, model_validator
from litellm import completion, get_supported_openai_params
from typing import Dict, List, Optional, Any, Union, Callable, Tuple


from .logging import logger


class TransitionCondition(BaseModel):
    """
    Defines a condition for a state transition.

    Attributes:
        description: Human-readable description of the condition
        requires_context_keys: List of context keys that must be present
    """
    description: str = Field(..., description="Human-readable description of the condition")
    requires_context_keys: Optional[List[str]] = Field(
        default=None,
        description="Context keys required to be present"
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


class FSMDefinition(BaseModel):
    """
    Complete definition of a Finite State Machine.

    Attributes:
        name: Name of the FSM
        description: Human-readable description
        states: All states in the FSM
        initial_state: The starting state
        version: Version of the FSM definition
    """
    name: str = Field(..., description="Name of the FSM")
    description: str = Field(..., description="Human-readable description")
    states: Dict[str, State] = Field(..., description="All states in the FSM")
    initial_state: str = Field(..., description="The starting state identifier")
    version: str = Field(default="3.0", description="Version of the FSM definition")

    @model_validator(mode='after')
    def validate_states(self) -> 'FSMDefinition':
        """
        Validates that:
        1. The initial state exists
        2. All target states in transitions exist
        3. No orphaned states

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

        # Check all transition target states exist
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

        # Check for orphaned states
        orphaned_states = set(self.states.keys()) - reachable_states
        if orphaned_states:
            states_str = ", ".join(orphaned_states)
            error_msg = f"Orphaned states detected: {states_str}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"FSM definition validated successfully: {self.name}")
        return self


class Conversation(BaseModel):
    """
    Conversation history.

    Attributes:
        exchanges: List of conversation exchanges
    """
    exchanges: List[Dict[str, str]] = Field(default_factory=list, description="Conversation exchanges")

    def add_user_message(self, message: str) -> None:
        """
        Add a user message to the conversation.

        Args:
            message: The user's message
        """
        logger.debug(f"Adding user message: {message[:50]}{'...' if len(message) > 50 else ''}")
        self.exchanges.append({"user": message})

    def add_system_message(self, message: str) -> None:
        """
        Add a system message to the conversation.

        Args:
            message: The system's message
        """
        logger.debug(f"Adding system message: {message[:50]}{'...' if len(message) > 50 else ''}")
        self.exchanges.append({"system": message})

    def get_recent(self, n: int = 5) -> List[Dict[str, str]]:
        """
        Get the n most recent exchanges.

        Args:
            n: Number of exchanges to retrieve

        Returns:
            List of recent exchanges
        """
        return self.exchanges[-n:] if n > 0 else []


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
    """
    fsm_id: str = Field(..., description="ID of the FSM definition")
    current_state: str = Field(..., description="Current state identifier")
    context: FSMContext = Field(default_factory=FSMContext, description="Runtime context")


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