import json
import uuid
from typing import Dict, List, Optional, Any, Union, Callable, Tuple


from .logging import logger
from .llm import LLMInterface, PromptBuilder
from .definitions import FSMDefinition, FSMInstance,State, LLMRequest


class FSMManager:
    """
    Manager for LLM-based finite state machines with improved API.
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
        # Store instances by conversation ID
        self.instances: Dict[str, FSMInstance] = {}

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

    def _create_instance(self, fsm_id: str) -> FSMInstance:
        """
        Create a new FSM instance (private method).

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

    def _process_user_input(
            self,
            instance: FSMInstance,
            user_input: str
    ) -> Tuple[FSMInstance, str]:
        """
        Internal method to process user input and update the FSM state.

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

    def start_conversation(
            self,
            fsm_id: str,
            initial_context: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
        """
        Start a new conversation with the specified FSM.

        Args:
            fsm_id: The ID of the FSM definition or path to FSM file
            initial_context: Optional initial context data for user personalization

        Returns:
            Tuple of (conversation_id, initial_response)
        """
        # Create a new instance
        instance = self._create_instance(fsm_id)

        # Add initial context if provided
        if initial_context:
            instance.context.update(initial_context)
            logger.info(f"Added initial context with keys: {', '.join(initial_context.keys())}")

        # Generate a unique conversation ID
        conversation_id = str(uuid.uuid4())

        # Store the instance
        self.instances[conversation_id] = instance

        logger.info(f"Started new conversation {conversation_id} with FSM {fsm_id}")

        # Process an empty input to get the initial response
        instance, response = self._process_user_input(instance, "")

        # Update the stored instance
        self.instances[conversation_id] = instance

        return conversation_id, response

    def process_message(self, conversation_id: str, message: str) -> str:
        """
        Process a user message in an existing conversation.

        Args:
            conversation_id: The conversation ID
            message: The user's message

        Returns:
            The system's response

        Raises:
            ValueError: If the conversation ID is not found
        """
        # Get the instance
        if conversation_id not in self.instances:
            error_msg = f"Conversation {conversation_id} not found"
            logger.error(error_msg)
            raise ValueError(error_msg)

        instance = self.instances[conversation_id]

        # Process the message
        updated_instance, response = self._process_user_input(instance, message)

        # Update the stored instance
        self.instances[conversation_id] = updated_instance

        return response

    def is_conversation_ended(self, conversation_id: str) -> bool:
        """
        Check if a conversation has reached an end state.

        Args:
            conversation_id: The conversation ID

        Returns:
            True if the conversation has ended, False otherwise

        Raises:
            ValueError: If the conversation ID is not found
        """
        # Get the instance
        if conversation_id not in self.instances:
            error_msg = f"Conversation {conversation_id} not found"
            logger.error(error_msg)
            raise ValueError(error_msg)

        instance = self.instances[conversation_id]

        # Check if the current state is a terminal state
        current_state = self.get_current_state(instance)
        return len(current_state.transitions) == 0

    def get_conversation_data(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get the collected data from a conversation.

        Args:
            conversation_id: The conversation ID

        Returns:
            The context data collected during the conversation

        Raises:
            ValueError: If the conversation ID is not found
        """
        # Get the instance
        if conversation_id not in self.instances:
            error_msg = f"Conversation {conversation_id} not found"
            logger.error(error_msg)
            raise ValueError(error_msg)

        instance = self.instances[conversation_id]

        # Return a copy of the context data
        return dict(instance.context.data)

    def get_conversation_state(self, conversation_id: str) -> str:
        """
        Get the current state of a conversation.

        Args:
            conversation_id: The conversation ID

        Returns:
            The current state ID

        Raises:
            ValueError: If the conversation ID is not found
        """
        # Get the instance
        if conversation_id not in self.instances:
            error_msg = f"Conversation {conversation_id} not found"
            logger.error(error_msg)
            raise ValueError(error_msg)

        instance = self.instances[conversation_id]

        return instance.current_state

    def end_conversation(self, conversation_id: str) -> None:
        """
        Explicitly end a conversation and clean up resources.

        Args:
            conversation_id: The conversation ID

        Raises:
            ValueError: If the conversation ID is not found
        """
        # Check if the conversation exists
        if conversation_id not in self.instances:
            error_msg = f"Conversation {conversation_id} not found"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Remove the instance
        del self.instances[conversation_id]

        logger.info(f"Ended conversation {conversation_id}")

    def get_complete_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """
        Extract all data from a conversation, including the complete history,
        collected data, state transitions, and metadata.

        Args:
            conversation_id: The conversation ID

        Returns:
            A dictionary containing all conversation data

        Raises:
            ValueError: If the conversation ID is not found
        """
        # Get the instance
        if conversation_id not in self.instances:
            error_msg = f"Conversation {conversation_id} not found"
            logger.error(error_msg)
            raise ValueError(error_msg)

        instance = self.instances[conversation_id]

        # Extract the conversation history
        conversation_history = [
            exchange for exchange in instance.context.conversation.exchanges
        ]

        # Get the current state information
        current_state = self.get_current_state(instance)

        # Compile all data
        result = {
            "id": conversation_id,
            "fsm_id": instance.fsm_id,
            "current_state": {
                "id": instance.current_state,
                "description": current_state.description,
                "purpose": current_state.purpose,
                "is_terminal": len(current_state.transitions) == 0
            },
            "collected_data": dict(instance.context.data),
            "conversation_history": conversation_history,
            "metadata": dict(instance.context.metadata)
        }

        logger.info(f"Extracted complete data for conversation {conversation_id}")
        return result