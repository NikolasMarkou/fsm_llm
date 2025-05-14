import json
import uuid
import time
import traceback
from datetime import datetime
from typing import Dict, Optional, Any, Callable, Tuple

# --------------------------------------------------------------
# local imports
# --------------------------------------------------------------

from .llm import LLMInterface
from .prompts import PromptBuilder
from .expressions import evaluate_logic
from .utilities import load_fsm_definition
from .logging import logger, with_conversation_context
from .handler_system import HandlerSystem, HandlerTiming
from .constants import DEFAULT_MAX_HISTORY_SIZE, DEFAULT_MAX_MESSAGE_LENGTH
from .definitions import FSMDefinition, FSMContext, FSMInstance, State, LLMRequest


# --------------------------------------------------------------

class FSMManager:
    """
    Manager for LLM-based finite state machines with integrated handler system.
    """

    def __init__(
            self,
            fsm_loader: Callable[[str], FSMDefinition] = load_fsm_definition,
            llm_interface: LLMInterface = None,
            prompt_builder: Optional[PromptBuilder] = None,
            max_history_size: int = DEFAULT_MAX_HISTORY_SIZE,
            max_message_length: int = DEFAULT_MAX_MESSAGE_LENGTH,
            handler_system: Optional[HandlerSystem] = None,
            handler_error_mode: str = "continue"
    ):
        """
        Initialize the FSM Manager.

        Args:
            fsm_loader: A function that loads an FSM definition by ID
            llm_interface: Interface for communicating with LLMs
            prompt_builder: Builder for creating prompts (optional)
            max_history_size: Maximum number of conversation exchanges to keep in history
            max_message_length: Maximum length of a message in characters
            handler_system: Optional handler system for function handlers
            handler_error_mode: How to handle errors in handlers ("continue", "raise", or "skip")
        """
        self.fsm_loader = fsm_loader
        self.llm_interface = llm_interface
        self.prompt_builder = prompt_builder or PromptBuilder(max_history_size=max_history_size)
        self.fsm_cache: Dict[str, FSMDefinition] = {}
        # Store instances by conversation ID
        self.instances: Dict[str, FSMInstance] = {}
        self.max_history_size = max_history_size
        self.max_message_length = max_message_length

        # Add handler system
        self.handler_system = handler_system or HandlerSystem(error_mode=handler_error_mode)

        logger.info(
            f"FSM Manager initialized with max_history_size={max_history_size}, max_message_length={max_message_length}")

    def register_handler(self, handler):
        """
        Register a handler with the system.

        Args:
            handler: The handler to register
        """
        self.handler_system.register_handler(handler)
        logger.info(f"Registered handler: {getattr(handler, 'name', handler.__class__.__name__)}")

    def get_logger_for_conversation(self, conversation_id: str):
        """
        Get a logger instance bound to a specific conversation ID.

        Args:
            conversation_id: The conversation ID to bind to the logger

        Returns:
            A logger instance with the conversation ID bound to it
        """
        return logger.bind(conversation_id=conversation_id)

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

        # Create context with configured conversation parameters
        context = FSMContext(
            max_history_size=self.max_history_size,
            max_message_length=self.max_message_length
        )

        return FSMInstance(
            fsm_id=fsm_id,
            current_state=fsm_def.initial_state,
            persona=fsm_def.persona,
            context=context
        )

    def get_current_state(self, instance: FSMInstance, conversation_id: Optional[str] = None) -> State:
        """
        Get the current state for an FSM instance.

        Args:
            instance: The FSM instance
            conversation_id: Optional conversation ID for logging

        Returns:
            The current state

        Raises:
            ValueError: If the state is not found
        """
        # Use conversation-specific logger if ID provided
        log = self.get_logger_for_conversation(conversation_id) if conversation_id else logger

        fsm_def = self.get_fsm_definition(instance.fsm_id)
        if instance.current_state not in fsm_def.states:
            error_msg = f"State '{instance.current_state}' not found in FSM '{instance.fsm_id}'"
            log.error(error_msg)
            raise ValueError(error_msg)

        log.debug(f"Current state: {instance.current_state}")
        return fsm_def.states[instance.current_state]

    def validate_transition(
            self,
            instance: FSMInstance,
            target_state: str,
            conversation_id: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate a state transition.

        Args:
            instance: The FSM instance
            target_state: The target state
            conversation_id: Optional conversation ID for logging

        Returns:
            Tuple of (is_valid, error_message)
        """
        log = self.get_logger_for_conversation(conversation_id) if conversation_id else logger
        log.debug(f"Validating transition from {instance.current_state} to {target_state}")

        fsm_def = self.get_fsm_definition(instance.fsm_id)
        current_state = self.get_current_state(instance, conversation_id)

        # Check if the target state exists
        if target_state not in fsm_def.states:
            error_msg = f"Target state '{target_state}' does not exist"
            log.warning(error_msg)
            return False, error_msg

        # If staying in the same state, always valid
        if target_state == instance.current_state:
            log.debug("Staying in the same state - valid")
            return True, None

        # Check if there's a transition to the target state
        valid_transitions = [t.target_state for t in current_state.transitions]
        if target_state not in valid_transitions:
            error_msg = f"No transition from '{current_state.id}' to '{target_state}'"
            log.warning(error_msg)
            return False, error_msg

        # Get the transition definition
        transition = next(t for t in current_state.transitions if t.target_state == target_state)

        # Check conditions if any
        if transition.conditions:
            for condition in transition.conditions:
                # check if required context keys exist
                if condition.requires_context_keys and not instance.context.has_keys(condition.requires_context_keys):
                    missing = instance.context.get_missing_keys(condition.requires_context_keys)
                    error_msg = f"Missing required context keys: {', '.join(missing)}"
                    log.warning(error_msg)
                    return False, error_msg
                # Check logic condition if present
                if condition.logic:
                    try:
                        result = evaluate_logic(condition.logic, instance.context.data)
                        if not result:
                            error_msg = f"Condition '{condition.description}' evaluated to false"
                            logger.warning(error_msg)
                            return False, error_msg
                    except Exception as e:
                        error_msg = f"Error evaluating condition logic: {str(e)}"
                        logger.error(error_msg)
                        return False, error_msg

        log.debug(f"Transition from {instance.current_state} to {target_state} is valid")
        return True, None

    def _process_user_input(
            self,
            instance: FSMInstance,
            user_input: str,
            conversation_id: str,
            skip_transition: bool = False
    ) -> Tuple[FSMInstance, str]:
        """
        Internal method to process user input and update the FSM state.

        Args:
            instance: The FSM instance
            user_input: The user's input text
            conversation_id: The conversation ID for logging
            skip_transition: if true do not update the FSM state (used for first message)

        Returns:
            A tuple of (updated instance, response message)
        """
        log = self.get_logger_for_conversation(conversation_id)
        log.info(f"Processing user input in state: {instance.current_state}")

        current_state_id = instance.current_state

        # Add the user message to the conversation
        instance.context.conversation.add_user_message(user_input)

        # Add conversation_id to context for handlers
        instance.context.data["_conversation_id"] = conversation_id

        try:
            # Execute PRE_PROCESSING handlers
            updated_context = self.handler_system.execute_handlers(
                timing=HandlerTiming.PRE_PROCESSING,
                current_state=current_state_id,
                target_state=None,
                context=instance.context.data
            )

            # Update context with handler results
            instance.context.data.update(updated_context)

            # Get the current state
            current_state = self.get_current_state(instance, conversation_id)

            # Generate the system prompt
            system_prompt = self.prompt_builder.build_system_prompt(instance, current_state)

            # Create the LLM request
            request = LLMRequest(
                system_prompt=system_prompt,
                user_message=user_input
            )

            log.debug(f"system_prompt:\n{system_prompt}")

            # Get the LLM response
            response = self.llm_interface.send_request(request)

            log.debug(f"system_response:\n{response.model_dump_json(indent=2)})")

            # Get the set of keys that will be updated
            updated_keys = set(
                response.transition.context_update.keys()) if response.transition.context_update else set()

            # Execute CONTEXT_UPDATE handlers before updating context
            updated_context = self.handler_system.execute_handlers(
                timing=HandlerTiming.CONTEXT_UPDATE,
                current_state=current_state_id,
                target_state=response.transition.target_state,
                context=instance.context.data,
                updated_keys=updated_keys
            )

            # IMPORTANT: First update the context with extracted data
            # This ensures we capture any information even if transition validation fails
            if response.transition.context_update:
                log.info(f"Updating context with: {json.dumps(response.transition.context_update)}")
                # Update with a merge of LLM's updates and handler's updates
                final_updates = {**response.transition.context_update, **updated_context}
                instance.context.update(final_updates)
            else:
                # Just update with handler results if LLM provided no updates
                instance.context.update(updated_context)

            # Execute POST_PROCESSING handlers
            updated_context = self.handler_system.execute_handlers(
                timing=HandlerTiming.POST_PROCESSING,
                current_state=current_state_id,
                target_state=response.transition.target_state,
                context=instance.context.data
            )

            # Update context with handler results
            instance.context.data.update(updated_context)

            # Now validate the transition after context has been updated
            is_valid, error = self.validate_transition(
                instance,
                response.transition.target_state,
                conversation_id
            )

            if not is_valid:
                # Handle ANY invalid transition by staying in the current state
                log.warning(f"Invalid transition detected: {error}")
                log.info(f"Staying in current state '{instance.current_state}' and processing response")

                # If the target state doesn't exist, modify the response to stay in current state
                if "does not exist" in error or "No transition from" in error:
                    log.warning(f"LLM attempted to transition to invalid state: {response.transition.target_state}")

                    # Modify the transition to stay in the current state
                    response.transition.target_state = instance.current_state

                    # Log this modification
                    log.info(f"Modified transition to stay in current state: {instance.current_state}")

                # Add the system response to the conversation
                instance.context.conversation.add_system_message(response.message)

                # Return without changing state
                return instance, response.message

            # Execute PRE_TRANSITION handlers
            updated_context = self.handler_system.execute_handlers(
                timing=HandlerTiming.PRE_TRANSITION,
                current_state=current_state_id,
                target_state=response.transition.target_state,
                context=instance.context.data
            )

            # Update context with handler results
            instance.context.data.update(updated_context)

            # Execute state transition if not skipping
            if not skip_transition:
                old_state = instance.current_state
                instance.current_state = response.transition.target_state
                log.info(f"State transition: {old_state} -> {instance.current_state}")

                # Add state transition metadata to context
                instance.context.data["_previous_state"] = old_state
                instance.context.data["_current_state"] = instance.current_state

                # Track state transitions for metrics
                if "_state_transitions" not in instance.context.data:
                    instance.context.data["_state_transitions"] = []

                instance.context.data["_state_transitions"].append({
                    "from": old_state,
                    "to": instance.current_state,
                    "timestamp": instance.context.data.get("_timestamp", None)
                })

                # Execute POST_TRANSITION handlers
                updated_context = self.handler_system.execute_handlers(
                    timing=HandlerTiming.POST_TRANSITION,
                    current_state=old_state,
                    target_state=instance.current_state,
                    context=instance.context.data
                )

                # Update context with handler results
                instance.context.data.update(updated_context)

            # Add the system response to the conversation
            instance.context.conversation.add_system_message(response.message)

            return instance, response.message

        except Exception as e:
            logger.error(f"Error processing user input: {str(e)}\n{traceback.format_exc()}")

            # Execute ERROR handlers
            try:
                error_context = {
                    **instance.context.data,
                    "error": {
                        "message": str(e),
                        "type": e.__class__.__name__,
                        "traceback": traceback.format_exc()
                    }
                }

                updated_context = self.handler_system.execute_handlers(
                    timing=HandlerTiming.ERROR,
                    current_state=current_state_id,
                    target_state=None,
                    context=error_context
                )

                # Update context with error handler results
                instance.context.data.update(updated_context)

                # Check if error handlers provided a fallback response
                fallback_response = updated_context.get("_fallback_response")
                if fallback_response:
                    # Add fallback response to conversation
                    instance.context.conversation.add_system_message(fallback_response)
                    return instance, fallback_response

            except Exception as handler_error:
                logger.error(f"Error in error handlers: {str(handler_error)}")

            # Re-raise the original exception if no fallback provided
            raise

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

        # Generate a unique conversation ID
        conversation_id = str(uuid.uuid4())

        # Get a conversation-specific logger
        log = self.get_logger_for_conversation(conversation_id)

        # Add initial context if provided
        if initial_context:
            instance.context.update(initial_context)
            log.info(f"Added initial context with keys: {', '.join(initial_context.keys())}")

        instance.context.data["_conversation_id"] = conversation_id
        instance.context.data["_conversation_start"] = datetime.now().isoformat()
        instance.context.data["_timestamp"] = time.time()
        instance.context.data["_fsm_id"] = fsm_id

        # Store the instance
        self.instances[conversation_id] = instance

        log.info(f"Started new conversation {conversation_id} with FSM {fsm_id}")

        # Execute START_CONVERSATION handlers
        updated_context = self.handler_system.execute_handlers(
            timing=HandlerTiming.START_CONVERSATION,
            current_state=None,
            target_state=instance.current_state,
            context=instance.context.data
        )

        # Update context with handler results
        instance.context.data.update(updated_context)

        # Process an empty input to get the initial response
        instance, response = self._process_user_input(
            instance, "", conversation_id, skip_transition=True
        )

        # Update the stored instance
        self.instances[conversation_id] = instance

        return conversation_id, response

    @with_conversation_context
    def process_message(self, conversation_id: str, message: str, log=None) -> str:
        """
        Process a user message in an existing conversation.

        Args:
            conversation_id: The conversation ID
            message: The user's message
            log: Logger instance (injected by decorator)

        Returns:
            The system's response

        Raises:
            ValueError: If the conversation ID is not found
        """
        # Get the instance
        if conversation_id not in self.instances:
            error_msg = f"Conversation {conversation_id} not found"
            log.error(error_msg)
            raise ValueError(error_msg)

        instance = self.instances[conversation_id]

        log.info(f"Processing message: {message[:50]}{'...' if len(message) > 50 else ''}")

        # Process the message
        updated_instance, response = (
            self._process_user_input(instance, message, conversation_id, skip_transition=False)
        )

        # Update the stored instance
        self.instances[conversation_id] = updated_instance

        return response

    @with_conversation_context
    def is_conversation_ended(self, conversation_id: str, log=None) -> bool:
        """
        Check if a conversation has reached an end state.

        Args:
            conversation_id: The conversation ID
            log: Logger instance (injected by decorator)

        Returns:
            True if the conversation has ended, False otherwise

        Raises:
            ValueError: If the conversation ID is not found
        """
        # Get the instance
        if conversation_id not in self.instances:
            error_msg = f"Conversation {conversation_id} not found"
            log.error(error_msg)
            raise ValueError(error_msg)

        instance = self.instances[conversation_id]

        # Check if the current state is a terminal state
        current_state = self.get_current_state(instance, conversation_id)
        is_ended = len(current_state.transitions) == 0

        if is_ended:
            log.info(f"Conversation has reached terminal state: {instance.current_state}")

        return is_ended

    @with_conversation_context
    def get_conversation_data(self, conversation_id: str, log=None) -> Dict[str, Any]:
        """
        Get the collected data from a conversation.

        Args:
            conversation_id: The conversation ID
            log: Logger instance (injected by decorator)

        Returns:
            The context data collected during the conversation

        Raises:
            ValueError: If the conversation ID is not found
        """
        # Get the instance
        if conversation_id not in self.instances:
            error_msg = f"Conversation {conversation_id} not found"
            log.error(error_msg)
            raise ValueError(error_msg)

        instance = self.instances[conversation_id]

        log.debug(f"Retrieving collected data with keys: {', '.join(instance.context.data.keys())}")

        # Return a copy of the context data
        return dict(instance.context.data)

    @with_conversation_context
    def get_conversation_state(self, conversation_id: str, log=None) -> str:
        """
        Get the current state of a conversation.

        Args:
            conversation_id: The conversation ID
            log: Logger instance (injected by decorator)

        Returns:
            The current state ID

        Raises:
            ValueError: If the conversation ID is not found
        """
        # Get the instance
        if conversation_id not in self.instances:
            error_msg = f"Conversation {conversation_id} not found"
            log.error(error_msg)
            raise ValueError(error_msg)

        instance = self.instances[conversation_id]

        log.debug(f"Current conversation state: {instance.current_state}")

        return instance.current_state

    @with_conversation_context
    def end_conversation(self, conversation_id: str, log=None) -> None:
        """
        Explicitly end a conversation and clean up resources.

        Args:
            conversation_id: The conversation ID
            log: Logger instance (injected by decorator)

        Raises:
            ValueError: If the conversation ID is not found
        """
        # Check if the conversation exists
        if conversation_id not in self.instances:
            error_msg = f"Conversation {conversation_id} not found"
            log.error(error_msg)
            raise ValueError(error_msg)

        log.info(f"Ending conversation {conversation_id}")

        # Remove the instance
        del self.instances[conversation_id]

    @with_conversation_context
    def get_complete_conversation(self, conversation_id: str, log=None) -> Dict[str, Any]:
        """
        Extract all data from a conversation, including the complete history,
        collected data, state transitions, and metadata.

        Args:
            conversation_id: The conversation ID
            log: Logger instance (injected by decorator)

        Returns:
            A dictionary containing all conversation data

        Raises:
            ValueError: If the conversation ID is not found
        """
        # Get the instance
        if conversation_id not in self.instances:
            error_msg = f"Conversation {conversation_id} not found"
            log.error(error_msg)
            raise ValueError(error_msg)

        instance = self.instances[conversation_id]

        # Extract the conversation history
        conversation_history = [
            exchange for exchange in instance.context.conversation.exchanges
        ]

        # Get the current state information
        current_state = self.get_current_state(instance, conversation_id)

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
            "metadata": dict(instance.context.metadata),
            "state_transitions": instance.context.data.get("_state_transitions", [])
        }

        log.info(f"Extracted complete data for conversation {conversation_id}")
        return result