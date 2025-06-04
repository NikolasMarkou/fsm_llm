"""
Enhanced FSM Manager implementing improved 2-Pass Architecture.

This module orchestrates the improved 2-pass LLM-FSM execution:
1. Pass 1: Data extraction + transition evaluation
2. Pass 2: Response generation based on final state

Key Features:
- Separation of data extraction from response generation
- Response generation occurs after transition evaluation
- Enhanced context management and handler integration
- Comprehensive logging and error handling
"""

import uuid
import time
import traceback
from datetime import datetime
from typing import Dict, Optional, Any, Callable, Tuple, List

# --------------------------------------------------------------
# Local imports
# --------------------------------------------------------------

from .llm import LLMInterface
from .prompts import (
    DataExtractionPromptBuilder,
    ResponseGenerationPromptBuilder,
    TransitionPromptBuilder
)
from .transition_evaluator import TransitionEvaluator, TransitionEvaluatorConfig
from .utilities import load_fsm_definition
from .handlers import HandlerSystem, HandlerTiming
from .logging import logger, with_conversation_context
from .constants import DEFAULT_MAX_HISTORY_SIZE, DEFAULT_MAX_MESSAGE_LENGTH
from .definitions import (
    FSMDefinition,
    FSMContext,
    FSMInstance,
    State,
    DataExtractionRequest,
    DataExtractionResponse,
    ResponseGenerationRequest,
    ResponseGenerationResponse,
    TransitionDecisionRequest,
    TransitionDecisionResponse,
    TransitionEvaluation,
    TransitionEvaluationResult,
    FSMError,
    StateNotFoundError,
    InvalidTransitionError,
    LLMResponseError
)


# --------------------------------------------------------------
# FSM Manager for Improved 2-Pass Architecture
# --------------------------------------------------------------

class FSMManager:
    """
    Enhanced FSM Manager implementing improved 2-pass architecture.

    This manager orchestrates the separation between data extraction,
    transition logic, and response generation, ensuring responses are
    generated with full context of the final state.
    """

    def __init__(
            self,
            fsm_loader: Callable[[str], FSMDefinition] = load_fsm_definition,
            llm_interface: LLMInterface = None,
            data_extraction_prompt_builder: Optional[DataExtractionPromptBuilder] = None,
            response_generation_prompt_builder: Optional[ResponseGenerationPromptBuilder] = None,
            transition_prompt_builder: Optional[TransitionPromptBuilder] = None,
            transition_evaluator: Optional[TransitionEvaluator] = None,
            max_history_size: int = DEFAULT_MAX_HISTORY_SIZE,
            max_message_length: int = DEFAULT_MAX_MESSAGE_LENGTH,
            handler_system: Optional[HandlerSystem] = None,
            handler_error_mode: str = "continue"
    ):
        """
        Initialize the enhanced FSM Manager.

        Args:
            fsm_loader: Function to load FSM definitions
            llm_interface: Interface for LLM communication
            data_extraction_prompt_builder: Builder for data extraction prompts
            response_generation_prompt_builder: Builder for response generation prompts
            transition_prompt_builder: Builder for transition decision prompts
            transition_evaluator: Evaluator for transition decisions
            max_history_size: Maximum conversation history size
            max_message_length: Maximum message length
            handler_system: Optional handler system for custom logic
            handler_error_mode: Handler error handling mode
        """
        self.fsm_loader = fsm_loader
        self.llm_interface = llm_interface

        # Initialize prompt builders
        self.data_extraction_prompt_builder = data_extraction_prompt_builder or DataExtractionPromptBuilder()
        self.response_generation_prompt_builder = response_generation_prompt_builder or ResponseGenerationPromptBuilder()
        self.transition_prompt_builder = transition_prompt_builder or TransitionPromptBuilder()

        # Initialize transition evaluator
        self.transition_evaluator = transition_evaluator or TransitionEvaluator()

        # Cache and instance management
        self.fsm_cache: Dict[str, FSMDefinition] = {}
        self.instances: Dict[str, FSMInstance] = {}

        # Configuration
        self.max_history_size = max_history_size
        self.max_message_length = max_message_length

        # Handler system
        self.handler_system = handler_system or HandlerSystem(error_mode=handler_error_mode)

        logger.info(
            f"Enhanced FSM Manager initialized with improved 2-pass architecture - "
            f"history_size={max_history_size}, message_length={max_message_length}"
        )

    def register_handler(self, handler):
        """Register a handler with the system."""
        self.handler_system.register_handler(handler)

    def get_fsm_definition(self, fsm_id: str) -> FSMDefinition:
        """Get FSM definition with caching."""
        if fsm_id not in self.fsm_cache:
            logger.info(f"Loading FSM definition: {fsm_id}")
            self.fsm_cache[fsm_id] = self.fsm_loader(fsm_id)
        return self.fsm_cache[fsm_id]

    def _create_instance(self, fsm_id: str) -> FSMInstance:
        """Create new FSM instance."""
        fsm_def = self.get_fsm_definition(fsm_id)
        logger.info(f"Creating FSM instance for {fsm_id}, initial state: {fsm_def.initial_state}")

        # Create context with configuration
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
        """Get current state definition for an instance."""
        log = logger.bind(conversation_id=conversation_id) if conversation_id else logger

        fsm_def = self.get_fsm_definition(instance.fsm_id)
        if instance.current_state not in fsm_def.states:
            error_msg = f"State '{instance.current_state}' not found in FSM '{instance.fsm_id}'"
            log.error(error_msg)
            raise StateNotFoundError(error_msg)

        return fsm_def.states[instance.current_state]

    def start_conversation(
            self,
            fsm_id: str,
            initial_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str]:
        """
        Start new conversation with improved 2-pass architecture.

        Args:
            fsm_id: FSM definition identifier
            initial_context: Optional initial context data

        Returns:
            Tuple of (conversation_id, initial_response)
        """
        # Create instance
        instance = self._create_instance(fsm_id)
        conversation_id = str(uuid.uuid4())

        # Setup initial context
        if initial_context:
            instance.context.update(initial_context)
            logger.info(f"Added initial context: {list(initial_context.keys())}")

        # Add system metadata
        instance.context.data.update({
            "_conversation_id": conversation_id,
            "_conversation_start": datetime.now().isoformat(),
            "_timestamp": time.time(),
            "_fsm_id": fsm_id
        })

        # Store instance
        self.instances[conversation_id] = instance

        # Execute start conversation handlers
        self._execute_handlers(
            HandlerTiming.START_CONVERSATION,
            conversation_id,
            current_state=None,
            target_state=instance.current_state
        )

        logger.info(f"Started conversation [{conversation_id}] with FSM [{fsm_id}]")

        # Generate initial response using response generation (no user input)
        try:
            response = self._generate_initial_response(instance, conversation_id)

            # Update stored instance
            self.instances[conversation_id] = instance

            return conversation_id, response

        except Exception as e:
            logger.error(f"Error generating initial response: {str(e)}")
            raise FSMError(f"Failed to start conversation: {str(e)}")

    def _generate_initial_response(self, instance: FSMInstance, conversation_id: str) -> str:
        """Generate initial response for conversation start."""
        log = logger.bind(conversation_id=conversation_id)

        # Get current state
        current_state = self.get_current_state(instance, conversation_id)
        fsm_def = self.get_fsm_definition(instance.fsm_id)

        # Build response generation prompt (no data extraction or transitions for initial)
        system_prompt = self.response_generation_prompt_builder.build_response_prompt(
            instance=instance,
            state=current_state,
            fsm_definition=fsm_def,
            extracted_data={},
            transition_occurred=False,
            previous_state=None,
            user_message=""
        )

        # Create response generation request
        request = ResponseGenerationRequest(
            system_prompt=system_prompt,
            user_message="",  # Empty for initial response
            extracted_data={},
            context=instance.context.get_user_visible_data(),
            transition_occurred=False,
            previous_state=None
        )

        # Generate response
        response = self.llm_interface.generate_response(request)

        # Store response for debugging
        instance.last_response_generation = response

        # Add to conversation history
        instance.context.conversation.add_system_message(response.message)

        log.info("Generated initial response")
        return response.message

    @with_conversation_context
    def process_message(self, conversation_id: str, message: str, log=None) -> str:
        """
        Process user message with improved 2-pass architecture.

        Pass 1: Data extraction + transition evaluation + transition execution
        Pass 2: Response generation based on final state

        Args:
            conversation_id: Conversation identifier
            message: User message to process
            log: Logger instance (injected by decorator)

        Returns:
            System response message
        """
        if conversation_id not in self.instances:
            raise ValueError(f"Conversation {conversation_id} not found")

        instance = self.instances[conversation_id]
        log.info(f"Processing message in state: {instance.current_state}")

        try:
            # Add user message to history
            instance.context.conversation.add_user_message(message)

            # Execute pre-processing handlers
            self._execute_handlers(
                HandlerTiming.PRE_PROCESSING,
                conversation_id,
                current_state=instance.current_state
            )

            # Pass 1: Data extraction + transition evaluation + execution
            extraction_response, transition_occurred, previous_state = self._execute_extraction_and_transition_pass(
                instance, message, conversation_id
            )

            # Execute post-processing handlers (after potential transition)
            self._execute_handlers(
                HandlerTiming.POST_PROCESSING,
                conversation_id,
                current_state=instance.current_state
            )

            # Pass 2: Response generation based on final state
            response_message = self._execute_response_generation_pass(
                instance, message, extraction_response, transition_occurred, previous_state, conversation_id
            )

            # Update stored instance
            self.instances[conversation_id] = instance

            return response_message

        except Exception as e:
            log.error(f"Error processing message: {str(e)}\n{traceback.format_exc()}")

            # Execute error handlers
            self._execute_handlers(
                HandlerTiming.ERROR,
                conversation_id,
                current_state=instance.current_state,
                error_context={"error": str(e), "traceback": traceback.format_exc()}
            )

            raise FSMError(f"Failed to process message: {str(e)}")

    def _execute_extraction_and_transition_pass(
            self,
            instance: FSMInstance,
            user_message: str,
            conversation_id: str
    ) -> Tuple[DataExtractionResponse, bool, Optional[str]]:
        """
        Execute Pass 1: Data Extraction + Transition Evaluation + Execution.

        Returns:
            Tuple of (extraction_response, transition_occurred, previous_state)
        """
        log = logger.bind(conversation_id=conversation_id)
        log.debug("Executing data extraction and transition pass")

        # Step 1: Data Extraction
        extraction_response = self._execute_data_extraction(instance, user_message, conversation_id)

        # Step 2: Update context with extracted data
        if extraction_response.extracted_data:
            extraction_response.extracted_data = (
                self._clean_empty_context_keys(
                    data=extraction_response.extracted_data,
                    conversation_id=conversation_id
                )
            )
            instance.context.update(extraction_response.extracted_data)

            # Execute context update handlers
            self._execute_handlers(
                HandlerTiming.CONTEXT_UPDATE,
                conversation_id,
                current_state=instance.current_state,
                updated_keys=set(extraction_response.extracted_data.keys())
            )

        # Step 3: Transition Evaluation and Execution
        transition_occurred, previous_state = self._execute_transition_evaluation_and_execution(
            instance, user_message, extraction_response, conversation_id
        )

        log.debug("Data extraction and transition pass completed")
        return extraction_response, transition_occurred, previous_state

    def _execute_data_extraction(
            self,
            instance: FSMInstance,
            user_message: str,
            conversation_id: str
    ) -> DataExtractionResponse:
        """Execute data extraction from user input."""
        log = logger.bind(conversation_id=conversation_id)
        log.debug("Executing data extraction")

        # Get current state and FSM definition
        current_state = self.get_current_state(instance, conversation_id)
        fsm_def = self.get_fsm_definition(instance.fsm_id)

        # Build data extraction prompt
        system_prompt = self.data_extraction_prompt_builder.build_extraction_prompt(
            instance, current_state, fsm_def
        )

        # Create data extraction request
        request = DataExtractionRequest(
            system_prompt=system_prompt,
            user_message=user_message,
            context=instance.context.get_user_visible_data()
        )

        # Extract data
        response = self.llm_interface.extract_data(request)

        # Store response for debugging
        instance.last_extraction_response = response

        log.debug(f"Data extraction completed: {list(response.extracted_data.keys()) if response.extracted_data else 'no data'}")
        return response

    def _execute_transition_evaluation_and_execution(
            self,
            instance: FSMInstance,
            user_message: str,
            extraction_response: DataExtractionResponse,
            conversation_id: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Execute transition evaluation and execution.

        Returns:
            Tuple of (transition_occurred, previous_state)
        """
        log = logger.bind(conversation_id=conversation_id)
        log.debug("Executing transition evaluation and execution")

        # Get current state
        current_state = self.get_current_state(instance, conversation_id)

        # Skip transition evaluation for terminal states
        if not current_state.transitions:
            log.debug("Terminal state reached - no transitions to evaluate")
            return False, None

        # Store current state before potential transition
        previous_state_id = instance.current_state

        # Evaluate transitions
        evaluation = self.transition_evaluator.evaluate_transitions(
            current_state,
            instance.context,
            extraction_response.extracted_data
        )

        # Handle evaluation result
        target_state = None

        if evaluation.result_type == TransitionEvaluationResult.DETERMINISTIC:
            target_state = evaluation.deterministic_transition
            log.info(f"Deterministic transition selected: {target_state}")

        elif evaluation.result_type == TransitionEvaluationResult.AMBIGUOUS:
            target_state = self._resolve_ambiguous_transition(
                evaluation, user_message, extraction_response, instance, conversation_id
            )
            log.info(f"LLM-assisted transition selected: {target_state}")

        elif evaluation.result_type == TransitionEvaluationResult.BLOCKED:
            log.warning(f"Transitions blocked: {evaluation.blocked_reason}")
            return False, None

        # Execute transition if target determined
        if target_state and target_state != instance.current_state:
            self._execute_state_transition(instance, target_state, conversation_id)
            return True, previous_state_id

        return False, None

    def _resolve_ambiguous_transition(
            self,
            evaluation: TransitionEvaluation,
            user_message: str,
            extraction_response: DataExtractionResponse,
            instance: FSMInstance,
            conversation_id: str
    ) -> str:
        """
        Resolve ambiguous transition using LLM assistance.

        Args:
            evaluation: Ambiguous transition evaluation
            user_message: Original user message
            extraction_response: Data extraction response
            instance: FSM instance
            conversation_id: Conversation identifier

        Returns:
            Selected target state
        """
        log = logger.bind(conversation_id=conversation_id)
        log.debug(f"Resolving ambiguous transition with {len(evaluation.available_options)} options")

        # Build transition decision prompt
        system_prompt = self.transition_prompt_builder.build_transition_prompt(
            current_state=instance.current_state,
            available_transitions=evaluation.available_options,
            context=instance.context.get_user_visible_data(),
            user_message=user_message,
            extracted_data=extraction_response.extracted_data
        )

        # Create transition decision request
        request = TransitionDecisionRequest(
            system_prompt=system_prompt,
            current_state=instance.current_state,
            available_transitions=evaluation.available_options,
            context=instance.context.get_user_visible_data(),
            user_message=user_message,
            extracted_data=extraction_response.extracted_data
        )

        # Get LLM decision
        response = self.llm_interface.decide_transition(request)

        # Store decision for debugging
        instance.last_transition_decision = response

        # Validate selected transition
        valid_targets = {opt.target_state for opt in evaluation.available_options}
        if response.selected_transition not in valid_targets:
            raise InvalidTransitionError(
                f"LLM selected invalid transition '{response.selected_transition}'. "
                f"Valid options: {sorted(valid_targets)}"
            )

        return response.selected_transition

    def _execute_state_transition(
            self,
            instance: FSMInstance,
            target_state: str,
            conversation_id: str
    ) -> None:
        """
        Execute state transition with handler integration.

        Args:
            instance: FSM instance
            target_state: Target state to transition to
            conversation_id: Conversation identifier
        """
        log = logger.bind(conversation_id=conversation_id)
        old_state = instance.current_state

        # Execute pre-transition handlers
        self._execute_handlers(
            HandlerTiming.PRE_TRANSITION,
            conversation_id,
            current_state=old_state,
            target_state=target_state
        )

        # Perform state transition
        instance.current_state = target_state

        # Update context metadata
        instance.context.data.update({
            "_previous_state": old_state,
            "_current_state": target_state,
            "_transition_timestamp": time.time()
        })

        # Execute post-transition handlers
        self._execute_handlers(
            HandlerTiming.POST_TRANSITION,
            conversation_id,
            current_state=old_state,
            target_state=target_state
        )

        log.info(f"State transition executed: {old_state} -> {target_state}")

    def _execute_response_generation_pass(
            self,
            instance: FSMInstance,
            user_message: str,
            extraction_response: DataExtractionResponse,
            transition_occurred: bool,
            previous_state: Optional[str],
            conversation_id: str
    ) -> str:
        """
        Execute Pass 2: Response Generation based on final state.

        Args:
            instance: FSM instance
            user_message: Original user message
            extraction_response: Data extraction response
            transition_occurred: Whether a transition occurred
            previous_state: Previous state if transition occurred
            conversation_id: Conversation identifier

        Returns:
            Generated user-facing message
        """
        log = logger.bind(conversation_id=conversation_id)
        log.debug("Executing response generation pass")

        # Get current (final) state and FSM definition
        current_state = self.get_current_state(instance, conversation_id)
        fsm_def = self.get_fsm_definition(instance.fsm_id)

        # Build response generation prompt with full context
        system_prompt = self.response_generation_prompt_builder.build_response_prompt(
            instance=instance,
            state=current_state,
            fsm_definition=fsm_def,
            extracted_data=extraction_response.extracted_data,
            transition_occurred=transition_occurred,
            previous_state=previous_state,
            user_message=user_message
        )

        # Create response generation request
        request = ResponseGenerationRequest(
            system_prompt=system_prompt,
            user_message=user_message,
            extracted_data=extraction_response.extracted_data,
            context=instance.context.get_user_visible_data(),
            transition_occurred=transition_occurred,
            previous_state=previous_state
        )

        # Generate response
        response = self.llm_interface.generate_response(request)

        # Store response for debugging
        instance.last_response_generation = response

        # Add response to conversation history
        instance.context.conversation.add_system_message(response.message)

        log.debug("Response generation pass completed")
        return response.message

    def _execute_handlers(
            self,
            timing: HandlerTiming,
            conversation_id: str,
            current_state: Optional[str] = None,
            target_state: Optional[str] = None,
            updated_keys: Optional[set] = None,
            error_context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Execute handlers at specified timing point."""
        if conversation_id not in self.instances:
            return

        instance = self.instances[conversation_id]
        context = instance.context.data.copy()

        # Add error context if provided
        if error_context:
            context.update(error_context)

        # Execute handlers
        try:
            updated_context = self.handler_system.execute_handlers(
                timing=timing,
                current_state=current_state or instance.current_state,
                target_state=target_state,
                context=context,
                updated_keys=updated_keys
            )

            # Update instance context with handler results
            if updated_context:
                instance.context.data.update(updated_context)

        except Exception as e:
            logger.error(f"Handler execution error at {timing.name}: {str(e)}")

    def _clean_empty_context_keys(
            self,
            data: Dict[str, Any],
            conversation_id: str,
            remove_empty_strings: bool = True,
            remove_empty_collections: bool = True,
            remove_none_values: bool = True
    ) -> Dict[str, Any]:
        """
        Clean empty/invalid keys from context data.

        Args:
            data: Dictionary to clean
            conversation_id: For logging context
            remove_empty_strings: Remove keys with empty string values
            remove_empty_collections: Remove keys with empty lists/dicts
            remove_none_values: Remove keys with None values

        Returns:
            Cleaned dictionary with empty keys removed
        """
        log = logger.bind(conversation_id=conversation_id)
        cleaned = {}
        removed_keys = []

        for key, value in data.items():
            should_remove = False
            removal_reason = ""

            # Check for None values
            if remove_none_values and value is None:
                should_remove = True
                removal_reason = "None value"

            # Check for empty strings
            elif remove_empty_strings and isinstance(value, str) and value.strip() == "":
                should_remove = True
                removal_reason = "empty string"

            # Check for empty collections
            elif remove_empty_collections and isinstance(value, (list, dict, set, tuple)) and len(value) == 0:
                should_remove = True
                removal_reason = f"empty {type(value).__name__}"

            # Keep the key-value pair
            if not should_remove:
                cleaned[key] = value
            else:
                removed_keys.append(f"{key} ({removal_reason})")

        if removed_keys:
            log.debug(f"Removed empty context keys: {removed_keys}")

        return cleaned

    # ==========================================
    # CONVERSATION MANAGEMENT METHODS
    # ==========================================

    @with_conversation_context
    def has_conversation_ended(self, conversation_id: str, log=None) -> bool:
        """Check if conversation has reached terminal state."""
        if conversation_id not in self.instances:
            raise ValueError(f"Conversation {conversation_id} not found")

        instance = self.instances[conversation_id]
        current_state = self.get_current_state(instance, conversation_id)

        is_ended = len(current_state.transitions) == 0
        if is_ended:
            log.info(f"Conversation reached terminal state: {instance.current_state}")

        return is_ended

    @with_conversation_context
    def get_conversation_data(self, conversation_id: str, log=None) -> Dict[str, Any]:
        """Get collected context data from conversation."""
        if conversation_id not in self.instances:
            raise ValueError(f"Conversation {conversation_id} not found")

        instance = self.instances[conversation_id]
        return dict(instance.context.data)

    @with_conversation_context
    def get_conversation_state(self, conversation_id: str, log=None) -> str:
        """Get current state of conversation."""
        if conversation_id not in self.instances:
            raise ValueError(f"Conversation {conversation_id} not found")

        return self.instances[conversation_id].current_state

    @with_conversation_context
    def get_conversation_history(self, conversation_id: str, log=None) -> List[Dict[str, str]]:
        """Get conversation history."""
        if conversation_id not in self.instances:
            raise ValueError(f"Conversation {conversation_id} not found")

        instance = self.instances[conversation_id]
        return instance.context.conversation.get_recent()

    @with_conversation_context
    def update_conversation_context(
            self,
            conversation_id: str,
            context_update: Dict[str, Any],
            log=None
    ) -> None:
        """Update conversation context data."""
        if conversation_id not in self.instances:
            raise ValueError(f"Conversation {conversation_id} not found")

        if not isinstance(context_update, dict):
            raise TypeError("context_update must be a dictionary")

        instance = self.instances[conversation_id]

        if context_update:
            log.info(f"Updating context with keys: {list(context_update.keys())}")
            instance.context.update(context_update)

            # Execute context update handlers
            self._execute_handlers(
                HandlerTiming.CONTEXT_UPDATE,
                conversation_id,
                updated_keys=set(context_update.keys())
            )

    @with_conversation_context
    def end_conversation(self, conversation_id: str, log=None) -> None:
        """End conversation and clean up resources."""
        if conversation_id not in self.instances:
            raise ValueError(f"Conversation {conversation_id} not found")

        # Execute end conversation handlers
        self._execute_handlers(
            HandlerTiming.END_CONVERSATION,
            conversation_id
        )

        # Remove instance
        del self.instances[conversation_id]
        log.info(f"Conversation {conversation_id} ended")

    @with_conversation_context
    def get_complete_conversation(self, conversation_id: str, log=None) -> Dict[str, Any]:
        """Get complete conversation data for analysis."""
        if conversation_id not in self.instances:
            raise ValueError(f"Conversation {conversation_id} not found")

        instance = self.instances[conversation_id]
        current_state = self.get_current_state(instance, conversation_id)

        return {
            "id": conversation_id,
            "fsm_id": instance.fsm_id,
            "current_state": {
                "id": instance.current_state,
                "description": current_state.description,
                "purpose": current_state.purpose,
                "is_terminal": len(current_state.transitions) == 0
            },
            "collected_data": dict(instance.context.data),
            "conversation_history": instance.context.conversation.get_recent(),
            "metadata": dict(instance.context.metadata),
            "last_extraction_response": instance.last_extraction_response.model_dump() if instance.last_extraction_response else None,
            "last_transition_decision": instance.last_transition_decision.model_dump() if instance.last_transition_decision else None,
            "last_response_generation": instance.last_response_generation.model_dump() if instance.last_response_generation else None
        }