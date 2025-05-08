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


class LLMInterface:
    """
    Interface for communicating with LLMs.
    """

    def send_request(self, request: LLMRequest) -> LLMResponse:
        """
        Send a request to the LLM and get the response.

        Args:
            request: The LLM request

        Returns:
            The LLM's response

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement send_request")


class LiteLLMInterface(LLMInterface):
    """
    Implementation of LLMInterface using LiteLLM.

    This class uses LiteLLM to send requests to various LLM providers
    while maintaining a consistent interface.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        enable_json_validation: bool = True,
        **kwargs
    ):
        """
        Initialize the LiteLLM interface.

        Args:
            model: The model to use (e.g., "gpt-4", "claude-3-opus")
            api_key: Optional API key (will use environment variables if not provided)
            enable_json_validation: Whether to enable JSON schema validation
            **kwargs: Additional arguments to pass to LiteLLM
        """
        self.model = model
        self.kwargs = kwargs

        logger.info(f"Initializing LiteLLMInterface with model: {model}")

        # Extract provider from model name for API key setting
        if api_key:
            # Simple provider detection, can be expanded
            if "gpt" in model.lower() or "openai" in model.lower():
                os.environ["OPENAI_API_KEY"] = api_key
                logger.debug("Setting OPENAI_API_KEY environment variable")
            elif "claude" in model.lower() or "anthropic" in model.lower():
                os.environ["ANTHROPIC_API_KEY"] = api_key
                logger.debug("Setting ANTHROPIC_API_KEY environment variable")
            else:
                # For other providers, we'll need to determine the right env var
                # or pass it directly to LiteLLM
                self.kwargs["api_key"] = api_key
                logger.debug("Using API key directly in LiteLLM kwargs")
        else:
            logger.debug("No API key provided, assuming it's set in environment variables")

        # Enable JSON schema validation if needed
        if enable_json_validation:
            litellm.enable_json_schema_validation = True
            logger.debug("Enabled JSON schema validation in LiteLLM")

    def send_request(self, request: LLMRequest) -> LLMResponse:
        """
        Send a request to the LLM using LiteLLM and get the response.

        Args:
            request: The LLM request

        Returns:
            The LLM's response

        Raises:
            LLMResponseError: If there's an error processing the LLM response
        """
        try:
            start_time = time.time()

            # Log the request (truncated for brevity)
            logger.info(f"Sending request to {self.model}")
            logger.debug(f"User message: {request.user_message[:50]}{'...' if len(request.user_message) > 50 else ''}")

            # Prepare messages for LiteLLM
            messages = [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_message}
            ]

            # Check if the model supports structured output (response_format or json_schema)
            supported_params = get_supported_openai_params(model=self.model)
            logger.debug(f"Supported parameters for {self.model}: {', '.join(supported_params)}")

            # Decide on the response format approach
            if "response_format" in supported_params:
                # The model supports the OpenAI-style response_format
                logger.debug(f"Using response_format for {self.model}")
                response = completion(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    **self.kwargs
                )
            else:
                # For other models, try to use json_schema if possible
                # or fall back to parsing from unstructured output
                try:
                    if litellm.supports_response_schema(model=self.model):
                        logger.debug(f"Using json_schema for {self.model}")
                        response = completion(
                            model=self.model,
                            messages=messages,
                            response_format=LLMResponseSchema,
                            **self.kwargs
                        )
                    else:
                        # Fall back to unstructured output
                        # We'll add instruction to return JSON in the system prompt
                        logger.debug(f"Using enhanced prompt with JSON instructions for {self.model}")
                        enhanced_prompt = (
                            f"{request.system_prompt}\n\n"
                            "IMPORTANT: You must respond with a valid JSON object that follows this schema:\n"
                            "{\n"
                            '  "transition": {\n'
                            '    "target_state": "state_id",\n'
                            '    "context_update": {"key1": "value1", "key2": "value2"}\n'
                            '  },\n'
                            '  "message": "Your message to the user",\n'
                            '  "reasoning": "Your reasoning for this decision"\n'
                            "}\n"
                        )

                        # Update system message with enhanced prompt
                        messages[0] = {"role": "system", "content": enhanced_prompt}

                        response = completion(
                            model=self.model,
                            messages=messages,
                            **self.kwargs
                        )
                except Exception as schema_error:
                    # If schema approach fails, fall back to unstructured output
                    # with manual JSON instructions
                    logger.warning(f"JSON schema approach failed: {str(schema_error)}")
                    logger.debug("Falling back to basic approach with JSON instructions")

                    enhanced_prompt = (
                        f"{request.system_prompt}\n\n"
                        "IMPORTANT: You must respond with a valid JSON object that follows this schema:\n"
                        "{\n"
                        '  "transition": {\n'
                        '    "target_state": "state_id",\n'
                        '    "context_update": {"key1": "value1", "key2": "value2"}\n'
                        '  },\n'
                        '  "message": "Your message to the user",\n'
                        '  "reasoning": "Your reasoning for this decision"\n'
                        "}\n"
                    )

                    # Update system message with enhanced prompt
                    messages[0] = {"role": "system", "content": enhanced_prompt}

                    response = completion(
                        model=self.model,
                        messages=messages,
                        **self.kwargs
                    )

            # Calculate response time
            response_time = time.time() - start_time
            logger.info(f"Received response from {self.model} in {response_time:.2f}s")

            # Extract the response content
            content = response.choices[0].message.content

            # Handle different response types
            if hasattr(content, "model_dump"):
                # This is already a Pydantic model (likely from json_schema)
                logger.debug("Response is a Pydantic model")
                response_data = content.model_dump()
            else:
                # This is a string, try to parse as JSON
                try:
                    logger.debug("Parsing response as JSON")
                    response_data = json.loads(content)
                except json.JSONDecodeError:
                    # If not valid JSON, try to extract JSON from the text
                    logger.warning("Response is not valid JSON, attempting to extract JSON from text")
                    extracted_json = extract_json_from_text(content)
                    if not extracted_json:
                        error_msg = f"Could not parse JSON from LLM response: {content[:100]}..."
                        logger.error(error_msg)
                        raise LLMResponseError(error_msg)
                    response_data = extracted_json
                    logger.debug("Successfully extracted JSON from text")

            # Create a StateTransition from the response
            transition_data = response_data.get("transition", {})
            transition = StateTransition(
                target_state=transition_data.get("target_state", ""),
                context_update=transition_data.get("context_update", {})
            )

            # Log the transition
            logger.info(f"Transition to: {transition.target_state}")
            if transition.context_update:
                logger.debug(f"Context updates: {json.dumps(transition.context_update)}")

            # Create and return the LLMResponse
            return LLMResponse(
                transition=transition,
                message=response_data.get("message", ""),
                reasoning=response_data.get("reasoning", None)
            )

        except Exception as e:
            # Handle exceptions
            error_msg = f"Error processing LLM response: {str(e)}"
            logger.error(error_msg)
            raise LLMResponseError(error_msg)


class PromptBuilder:
    """
    Builder for creating prompts for the LLM.
    """

    def build_system_prompt(self, instance: FSMInstance, state: State) -> str:
        """
        Build a system prompt for the current state with clearer instructions about valid transitions.

        Args:
            instance: The FSM instance
            state: The current state

        Returns:
            A system prompt string
        """
        logger.debug(f"Building system prompt for state: {state.id}")

        # Start with the basic prompt
        prompt_parts = [
            f"# {instance.fsm_id}",
            f"## Current State: {state.id}",
            f"Description: {state.description}",
            f"Purpose: {state.purpose}"
        ]

        # Add instructions if available
        if state.instructions:
            prompt_parts.append(f"\n## Instructions:\n{state.instructions}")

        # Add required context keys with enhanced instructions
        if state.required_context_keys:
            keys_str = ", ".join(state.required_context_keys)
            prompt_parts.append(f"\n## Information to collect:\n{keys_str}")

            # Special handling for name collection
            if "name" in state.required_context_keys:
                prompt_parts.append("\n## EXTRACTION INSTRUCTIONS:")
                prompt_parts.append("When the user provides their name, you MUST:")
                prompt_parts.append("1. Extract the name explicitly mentioned (e.g., 'My name is John' → 'John')")
                prompt_parts.append("2. Extract implicit name mentions (e.g., 'Call me John' → 'John')")
                prompt_parts.append(
                    "3. Store the extracted name in the context_update field as: {\"name\": \"ExtractedName\"}")
                prompt_parts.append(
                    "4. Only transition to the next state if you have successfully extracted and stored the name")
                prompt_parts.append(
                    "\nIncorrect: Responding with 'Nice to meet you, John!' but not adding {\"name\": \"John\"} to context_update")
                prompt_parts.append(
                    "Correct: Adding {\"name\": \"John\"} to context_update AND responding with 'Nice to meet you, John!'")

        # Add available transitions with explicit instructions
        available_states = [t.target_state for t in state.transitions]
        prompt_parts.append("\n## Available Transitions:")

        if available_states:
            for i, transition in enumerate(state.transitions):
                prompt_parts.append(f"{i + 1}. To '{transition.target_state}': {transition.description}")

            # Add EXPLICIT instructions about valid transitions
            prompt_parts.append("\n## IMPORTANT TRANSITION RULES:")
            prompt_parts.append("1. You MUST ONLY choose from the following valid target states:")
            prompt_parts.append("   " + ", ".join([f"'{state}'" for state in available_states]))
            prompt_parts.append("2. Do NOT invent or create new states that are not in the above list.")
            prompt_parts.append("3. If you're unsure which state to transition to, stay in the current state.")
            prompt_parts.append(f"4. The current state is '{state.id}' - you can choose to stay here if needed.")
        else:
            prompt_parts.append("This state has no outgoing transitions. Stay in the current state.")

        # Add current context with clearer formatting
        if instance.context.data:
            prompt_parts.append("\n## Current Context:")
            for key, value in instance.context.data.items():
                prompt_parts.append(f"- {key}: {value}")
        else:
            prompt_parts.append("\n## Current Context: None (empty)")

        # Add conversation history
        recent_exchanges = instance.context.conversation.get_recent(5)
        if recent_exchanges:
            prompt_parts.append("\n## Recent conversation history:")
            for exchange in recent_exchanges:
                for role, text in exchange.items():
                    prompt_parts.append(f"{role.capitalize()}: {text}")

        # Add example dialogue if available
        if state.example_dialogue:
            prompt_parts.append("\n## Example dialogue for this state:")
            for exchange in state.example_dialogue:
                for role, text in exchange.items():
                    prompt_parts.append(f"{role.capitalize()}: {text}")

        available_states = ', '.join([f"{state}" for state in available_states])
        # Add response format instructions
        prompt_parts.extend([
            "\n## Response Format:",
            "Respond with a JSON object with the following structure:",
            "```json",
            "{",
            '  "transition": {',
            '    "target_state": "state_id",',
            '    "context_update": {"key1": "value1", "key2": "value2"}',
            '  },',
            '  "message": "Your message to the user",',
            '  "reasoning": "Your reasoning for this decision"',
            "}",
            "```",
            "\nImportant:",
            "1. Collect all required information from the user's message",
            "2. Only transition to a new state if all required information is collected or another transition is appropriate",
            "3. Your message should be conversational and natural",
            "4. Don't mention states, transitions, or context keys to the user",
            f"5. Remember, you can ONLY choose from these valid target states: {available_states}"
        ])

        prompt = "\n".join(prompt_parts)
        logger.debug(f"System prompt length: {len(prompt)} characters")

        return prompt


class FSMManager:
    """
    Manager for LLM-based finite state machines.
    """

    def __init__(
        self,
        fsm_loader: Callable[[str], FSMDefinition],
        llm_interface: LLMInterface,
        prompt_builder: Optional[PromptBuilder] = None
    ):
        """
        Initialize the FSM Manager.

        Args:
            fsm_loader: A function that loads an FSM definition by ID
            llm_interface: Interface for communicating with LLMs
            prompt_builder: Builder for creating prompts (optional)
        """
        self.fsm_loader = fsm_loader
        self.llm_interface = llm_interface
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.fsm_cache: Dict[str, FSMDefinition] = {}

        logger.info("FSM Manager initialized")

    def get_fsm_definition(self, fsm_id: str) -> FSMDefinition:
        """
        Get an FSM definition, using cache if available.

        Args:
            fsm_id: The ID of the FSM definition

        Returns:
            The FSM definition
        """
        if fsm_id not in self.fsm_cache:
            logger.info(f"Loading FSM definition: {fsm_id}")
            self.fsm_cache[fsm_id] = self.fsm_loader(fsm_id)
        else:
            logger.debug(f"Using cached FSM definition: {fsm_id}")
        return self.fsm_cache[fsm_id]

    def create_instance(self, fsm_id: str) -> FSMInstance:
        """
        Create a new FSM instance.

        Args:
            fsm_id: The ID of the FSM definition

        Returns:
            A new FSM instance
        """
        fsm_def = self.get_fsm_definition(fsm_id)
        logger.info(f"Creating new FSM instance for {fsm_id}, starting at state: {fsm_def.initial_state}")
        return FSMInstance(
            fsm_id=fsm_id,
            current_state=fsm_def.initial_state
        )

    def get_current_state(self, instance: FSMInstance) -> State:
        """
        Get the current state for an FSM instance.

        Args:
            instance: The FSM instance

        Returns:
            The current state

        Raises:
            ValueError: If the state is not found
        """
        fsm_def = self.get_fsm_definition(instance.fsm_id)
        if instance.current_state not in fsm_def.states:
            error_msg = f"State '{instance.current_state}' not found in FSM '{instance.fsm_id}'"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.debug(f"Current state: {instance.current_state}")
        return fsm_def.states[instance.current_state]

    def validate_transition(
        self,
        instance: FSMInstance,
        target_state: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate a state transition.

        Args:
            instance: The FSM instance
            target_state: The target state

        Returns:
            Tuple of (is_valid, error_message)
        """
        logger.debug(f"Validating transition from {instance.current_state} to {target_state}")

        fsm_def = self.get_fsm_definition(instance.fsm_id)
        current_state = self.get_current_state(instance)

        # Check if the target state exists
        if target_state not in fsm_def.states:
            error_msg = f"Target state '{target_state}' does not exist"
            logger.warning(error_msg)
            return False, error_msg

        # If staying in the same state, always valid
        if target_state == instance.current_state:
            logger.debug("Staying in the same state - valid")
            return True, None

        # Check if there's a transition to the target state
        valid_transitions = [t.target_state for t in current_state.transitions]
        if target_state not in valid_transitions:
            error_msg = f"No transition from '{current_state.id}' to '{target_state}'"
            logger.warning(error_msg)
            return False, error_msg

        # Get the transition definition
        transition = next(t for t in current_state.transitions if t.target_state == target_state)

        # Check conditions if any
        if transition.conditions:
            for condition in transition.conditions:
                if condition.requires_context_keys and not instance.context.has_keys(condition.requires_context_keys):
                    missing = instance.context.get_missing_keys(condition.requires_context_keys)
                    error_msg = f"Missing required context keys: {', '.join(missing)}"
                    logger.warning(error_msg)
                    return False, error_msg

        logger.debug(f"Transition from {instance.current_state} to {target_state} is valid")
        return True, None

    def process_user_input(
            self,
            instance: FSMInstance,
            user_input: str
    ) -> Tuple[FSMInstance, str]:
        """
        Process user input and update the FSM state, with improved context extraction.

        Args:
            instance: The FSM instance
            user_input: The user's input text

        Returns:
            A tuple of (updated instance, response message)
        """
        logger.info(f"Processing user input in state: {instance.current_state}")

        # Add the user message to the conversation
        instance.context.conversation.add_user_message(user_input)

        # Get the current state
        current_state = self.get_current_state(instance)

        # Generate the system prompt
        system_prompt = self.prompt_builder.build_system_prompt(instance, current_state)

        # Create the LLM request
        request = LLMRequest(
            system_prompt=system_prompt,
            user_message=user_input
        )

        # Get the LLM response
        response = self.llm_interface.send_request(request)

        # IMPORTANT: First update the context with extracted data
        # This ensures we capture any information even if transition validation fails
        if response.transition.context_update:
            logger.info(f"Updating context with: {json.dumps(response.transition.context_update)}")
            instance.context.update(response.transition.context_update)

        # Now validate the transition after context has been updated
        is_valid, error = self.validate_transition(
            instance,
            response.transition.target_state
        )

        if not is_valid:
            # Handle ANY invalid transition by staying in the current state
            logger.warning(f"Invalid transition detected: {error}")
            logger.info(f"Staying in current state '{instance.current_state}' and processing response")

            # If the target state doesn't exist, modify the response to stay in current state
            if "does not exist" in error or "No transition from" in error:
                logger.warning(f"LLM attempted to transition to invalid state: {response.transition.target_state}")

                # Modify the transition to stay in the current state
                response.transition.target_state = instance.current_state

                # Log this modification
                logger.info(f"Modified transition to stay in current state: {instance.current_state}")

            # Add the system response to the conversation
            instance.context.conversation.add_system_message(response.message)

            # Return without changing state
            return instance, response.message

        # Log the state transition
        old_state = instance.current_state
        instance.current_state = response.transition.target_state
        logger.info(f"State transition: {old_state} -> {instance.current_state}")

        # Add the system response to the conversation
        instance.context.conversation.add_system_message(response.message)

        return instance, response.message


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract a JSON object from text.

    Args:
        text: The text to extract from

    Returns:
        The extracted JSON or None
    """
    logger.debug("Attempting to extract JSON from text")

    # Try to find JSON between code blocks first
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if json_match:
        try:
            json_str = json_match.group(1)
            logger.debug("Found JSON in code block, parsing...")
            return json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from code block")
            pass

    # Try to find any JSON object in the text
    json_pattern = r'{[\s\S]*}'
    json_match = re.search(json_pattern, text)
    if json_match:
        try:
            json_str = json_match.group(0)
            logger.debug("Found JSON pattern in text, parsing...")
            return json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from text pattern")
            pass

    logger.warning("Could not extract valid JSON from text")
    return None


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