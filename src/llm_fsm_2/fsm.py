# /fsm.py

"""
Enhanced FSM Manager implementing 2-Pass Architecture.

This module orchestrates the 2-pass LLM-FSM execution:
1. Pass 1: Content generation focused on current state only
2. Pass 2: Transition evaluation (deterministic or LLM-assisted)

Key Features:
- Separation of content generation from transition logic
- Deterministic transition evaluation with LLM fallback
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
from .prompts import ContentPromptBuilder, TransitionPromptBuilder
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
    ContentGenerationRequest,
    ContentGenerationResponse,
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
# FSM Manager for 2-Pass Architecture
# --------------------------------------------------------------

class FSMManager:
    """
    Enhanced FSM Manager implementing 2-pass architecture.

    This manager orchestrates the separation between content generation
    and transition logic, ensuring FSM structure doesn't leak into conversations.
    """

    def __init__(
            self,
            fsm_loader: Callable[[str], FSMDefinition] = load_fsm_definition,
            llm_interface: LLMInterface = None,
            content_prompt_builder: Optional[ContentPromptBuilder] = None,
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
            content_prompt_builder: Builder for content generation prompts
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
        self.content_prompt_builder = content_prompt_builder or ContentPromptBuilder()
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
            f"Enhanced FSM Manager initialized with 2-pass architecture - "
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
        Start new conversation with 2-pass architecture.

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

        # Generate initial response using content generation only
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

        # Build content generation prompt
        system_prompt = self.content_prompt_builder.build_content_prompt(
            instance, current_state, fsm_def
        )

        # Create content generation request
        request = ContentGenerationRequest(
            system_prompt=system_prompt,
            user_message="",  # Empty for initial response
            context=instance.context.get_user_visible_data()
        )

        # Generate content
        response = self.llm_interface.generate_content(request)

        # Store response for debugging
        instance.last_content_response = response

        # Update context with extracted data
        if response.extracted_data:
            instance.context.update(response.extracted_data)

        # Add to conversation history
        instance.context.conversation.add_system_message(response.message)

        log.info("Generated initial response")
        return response.message

    @with_conversation_context
    def process_message(self, conversation_id: str, message: str, log=None) -> str:
        """
        Process user message with 2-pass architecture.

        Pass 1: Generate content response
        Pass 2: Evaluate and execute transitions

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

            # Pass 1: Generate content response
            content_response = self._execute_content_generation_pass(
                instance, message, conversation_id
            )

            # Execute post-processing handlers
            self._execute_handlers(
                HandlerTiming.POST_PROCESSING,
                conversation_id,
                current_state=instance.current_state
            )

            # Pass 2: Evaluate and execute transitions
            self._execute_transition_evaluation_pass(
                instance, message, content_response, conversation_id
            )

            # Update stored instance
            self.instances[conversation_id] = instance

            return content_response.message

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

    def _execute_content_generation_pass(
            self,
            instance: FSMInstance,
            user_message: str,
            conversation_id: str
    ) -> ContentGenerationResponse:
        """
        Execute Pass 1: Content Generation.

        Generates user-facing response without exposing FSM structure.
        """
        log = logger.bind(conversation_id=conversation_id)
        log.debug("Executing content generation pass")

        # Get current state and FSM definition
        current_state = self.get_current_state(instance, conversation_id)
        fsm_def = self.get_fsm_definition(instance.fsm_id)

        # Build content generation prompt
        system_prompt = self.content_prompt_builder.build_content_prompt(
            instance, current_state, fsm_def
        )

        # Create request
        request = ContentGenerationRequest(
            system_prompt=system_prompt,
            user_message=user_message,
            context=instance.context.get_user_visible_data()
        )

        # Generate content
        response = self.llm_interface.generate_content(request)

        # Store response for debugging
        instance.last_content_response = response

        # Update context with extracted data
        if response.extracted_data:
            instance.context.update(response.extracted_data)

            # Execute context update handlers
            self._execute_handlers(
                HandlerTiming.CONTEXT_UPDATE,
                conversation_id,
                current_state=instance.current_state,
                updated_keys=set(response.extracted_data.keys())
            )

        # Add response to conversation history
        instance.context.conversation.add_system_message(response.message)

        log.debug("Content generation pass completed")
        return response

    def _execute_transition_evaluation_pass(
            self,
            instance: FSMInstance,
            user_message: str,
            content_response: ContentGenerationResponse,
            conversation_id: str
    ) -> None:
        """
        Execute Pass 2: Transition Evaluation.

        Evaluates possible transitions and executes them deterministically
        or with LLM assistance for ambiguous cases.
        """
        log = logger.bind(conversation_id=conversation_id)
        log.debug("Executing transition evaluation pass")

        # Get current state
        current_state = self.get_current_state(instance, conversation_id)

        # Skip transition evaluation for terminal states
        if not current_state.transitions:
            log.debug("Terminal state reached - no transitions to evaluate")
            return

        # Evaluate transitions
        evaluation = self.transition_evaluator.evaluate_transitions(
            current_state,
            instance.context,
            content_response.extracted_data
        )

        # Handle evaluation result
        target_state = None

        if evaluation.result_type == TransitionEvaluationResult.DETERMINISTIC:
            target_state = evaluation.deterministic_transition
            log.info(f"Deterministic transition selected: {target_state}")

        elif evaluation.result_type == TransitionEvaluationResult.AMBIGUOUS:
            target_state = self._resolve_ambiguous_transition(
                evaluation, user_message, instance, conversation_id
            )
            log.info(f"LLM-assisted transition selected: {target_state}")

        elif evaluation.result_type == TransitionEvaluationResult.BLOCKED:
            log.warning(f"Transitions blocked: {evaluation.blocked_reason}")
            # Stay in current state
            return

        # Execute transition if target determined
        if target_state and target_state != instance.current_state:
            self._execute_state_transition(instance, target_state, conversation_id)

    def _resolve_ambiguous_transition(
            self,
            evaluation: TransitionEvaluation,
            user_message: str,
            instance: FSMInstance,
            conversation_id: str
    ) -> str:
        """
        Resolve ambiguous transition using LLM assistance.

        Args:
            evaluation: Ambiguous transition evaluation
            user_message: Original user message
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
            user_message=user_message
        )

        # Create transition decision request
        request = TransitionDecisionRequest(
            system_prompt=system_prompt,
            current_state=instance.current_state,
            available_transitions=evaluation.available_options,
            context=instance.context.get_user_visible_data(),
            user_message=user_message
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
            "last_content_response": instance.last_content_response.model_dump() if instance.last_content_response else None,
            "last_transition_decision": instance.last_transition_decision.model_dump() if instance.last_transition_decision else None
        }