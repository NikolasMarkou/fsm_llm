from __future__ import annotations

"""
FSM Manager Module for FSM-LLM: Core Orchestration Engine with Enhanced 2-Pass Architecture.

This module implements the central orchestration engine for the FSM-LLM library, managing the
complete lifecycle of FSM-driven conversations. The FSMManager class serves as the primary
coordinator between data extraction, transition evaluation, state management, and response
generation, implementing a sophisticated 2-pass architecture that separates concerns for
improved conversation quality and system reliability.

Enhanced 2-Pass Architecture
----------------------------
The FSMManager orchestrates a refined 2-pass processing model that fundamentally changes
how conversational AI systems handle user interactions:

**Traditional Single-Pass Approach**:
User Input → LLM → State Transition + Response (combined, potentially inconsistent)

**Enhanced 2-Pass Architecture**:
**Pass 1 - Analysis & Transition**:
1. Data Extraction: Extract structured information from user input
2. Context Integration: Merge extracted data with existing conversation context
3. Transition Evaluation: Determine next state using rule-based or LLM-assisted logic
4. State Transition: Execute validated state change with handler integration

**Pass 2 - Response Generation**:
5. Response Synthesis: Generate contextually appropriate response based on final state
6. History Management: Update conversation history with consistent messaging

This separation ensures:
- **Consistency**: Responses always reflect the final state after transitions
- **Efficiency**: Rule-based transitions avoid unnecessary LLM calls
- **Quality**: Responses are generated with complete context awareness
- **Debuggability**: Each pass can be independently analyzed and optimized

Core Responsibilities
---------------------
The FSMManager serves as the central coordination hub for:

**FSM Lifecycle Management**:
- FSM definition loading and caching
- Conversation instance creation and tracking
- State transition validation and execution
- Terminal state detection and cleanup

**Component Integration**:
- LLM interface coordination for different request types
- Prompt builder orchestration for specialized prompts
- Transition evaluator integration for intelligent path selection
- Handler system execution at critical lifecycle points

**Context & History Management**:
- Conversation context maintenance and updates
- Message history tracking with configurable limits
- Metadata management for debugging and analytics
- Context cleaning and validation

**Error Handling & Recovery**:
- Comprehensive exception handling with detailed logging
- Handler-based error recovery mechanisms
- Conversation state preservation during failures
- Graceful degradation for edge cases

Conversation Flow Architecture
------------------------------
The enhanced conversation flow follows this detailed sequence:

```
User Message Input
        ↓
[START: Pre-Processing Handlers]
        ↓
[PASS 1: Analysis & State Management]
├─ Data Extraction (LLM)
│  └─ Extract structured data from user input
├─ Context Integration (Rules)
│  └─ Merge extracted data with conversation context
├─ Transition Evaluation (Rules/LLM)
│  ├─ Rule-based evaluation for deterministic cases
│  └─ LLM-assisted selection for ambiguous cases
└─ State Transition (Rules)
   └─ Execute validated transition with handlers
        ↓
[PASS 2: Response Generation]
├─ Response Generation (LLM)
│  └─ Generate response based on final state and context
└─ History Update (Rules)
   └─ Add response to conversation history
        ↓
[END: Post-Processing Handlers]
        ↓
Response to User
```

Usage Examples
--------------
Basic FSM manager initialization and usage:

.. code-block:: python

    from fsm_llm.fsm import FSMManager
    from fsm_llm.llm import LiteLLMInterface

    # Initialize components
    llm_interface = LiteLLMInterface(model="gpt-4", temperature=0.7)

    # Create FSM manager with enhanced 2-pass architecture
    fsm_manager = FSMManager(
        llm_interface=llm_interface,
        max_history_size=10,
        max_message_length=1000
    )

    # Start conversation
    conversation_id, initial_response = fsm_manager.start_conversation(
        fsm_id="customer_service_fsm",
        initial_context={"customer_tier": "premium"}
    )

    # Process user messages
    response = fsm_manager.process_message(conversation_id, "I need help with my order")

Advanced configuration with custom components:

.. code-block:: python

    from fsm_llm.transition_evaluator import TransitionEvaluator, TransitionEvaluatorConfig
    from fsm_llm.handlers import HandlerSystem

    # Configure transition evaluator for strict evaluation
    evaluator_config = TransitionEvaluatorConfig(
        ambiguity_threshold=0.3,
        minimum_confidence=0.8,
        strict_condition_matching=True
    )
    transition_evaluator = TransitionEvaluator(evaluator_config)

    # Create handler system with custom error handling
    handler_system = HandlerSystem(error_mode="raise")

    # Initialize with custom components
    fsm_manager = FSMManager(
        llm_interface=llm_interface,
        transition_evaluator=transition_evaluator,
        handler_system=handler_system,
        max_history_size=20
    )

Conversation monitoring and debugging:

.. code-block:: python

    # Get comprehensive conversation state
    conversation_data = fsm_manager.get_complete_conversation(conversation_id)

    # Monitor conversation progress
    current_state = fsm_manager.get_conversation_state(conversation_id)
    collected_data = fsm_manager.get_conversation_data(conversation_id)
    is_terminal = fsm_manager.has_conversation_ended(conversation_id)

    # Update context programmatically
    fsm_manager.update_conversation_context(
        conversation_id,
        {"user_verified": True, "priority_level": "high"}
    )

Performance Optimization Features
---------------------------------
The FSMManager includes several optimization strategies:

**Caching Strategy**:
- FSM definition caching prevents repeated file I/O
- Instance pooling for conversation management
- Context optimization with configurable limits

**Selective Processing**:
- Terminal state detection skips unnecessary processing
- Early termination in transition evaluation
- Conditional handler execution based on timing and state

**Memory Management**:
- Configurable history size limits prevent memory bloat
- Context cleaning removes empty/invalid data
- Automatic cleanup on conversation termination

**Concurrent Processing Support**:
- Thread-safe instance management (per conversation)
- Parallel-ready handler execution
- Async-compatible architecture foundation
"""

import copy
import re
import uuid
import time
import threading
import traceback
from datetime import datetime
from typing import Any, Callable
from collections import defaultdict

# --------------------------------------------------------------
# Local imports
# --------------------------------------------------------------

from .llm import LLMInterface
from .prompts import (
    DataExtractionPromptBuilder,
    ResponseGenerationPromptBuilder,
    TransitionPromptBuilder
)
from .transition_evaluator import TransitionEvaluator
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
    TransitionDecisionRequest,
    TransitionEvaluation,
    TransitionEvaluationResult,
    FSMError,
    StateNotFoundError,
    InvalidTransitionError
)


# --------------------------------------------------------------
# FSM Manager
# --------------------------------------------------------------

class FSMManager:
    """
    This manager orchestrates the separation between data extraction,
    transition logic, and response generation, ensuring responses are
    generated with full context of the final state.
    """

    def __init__(
            self,
            fsm_loader: Callable[[str], FSMDefinition] = load_fsm_definition,
            llm_interface: LLMInterface = None,
            data_extraction_prompt_builder: DataExtractionPromptBuilder | None = None,
            response_generation_prompt_builder: ResponseGenerationPromptBuilder | None = None,
            transition_prompt_builder: TransitionPromptBuilder | None = None,
            transition_evaluator: TransitionEvaluator | None = None,
            max_history_size: int = DEFAULT_MAX_HISTORY_SIZE,
            max_message_length: int = DEFAULT_MAX_MESSAGE_LENGTH,
            handler_system: HandlerSystem | None = None,
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
        if llm_interface is None:
            raise ValueError("llm_interface is required and cannot be None")

        self.fsm_loader = fsm_loader
        self.llm_interface = llm_interface

        # Initialize prompt builders
        self.data_extraction_prompt_builder = data_extraction_prompt_builder or DataExtractionPromptBuilder()
        self.response_generation_prompt_builder = response_generation_prompt_builder or ResponseGenerationPromptBuilder()
        self.transition_prompt_builder = transition_prompt_builder or TransitionPromptBuilder()

        # Initialize transition evaluator
        self.transition_evaluator = transition_evaluator or TransitionEvaluator()

        # Lock for thread-safe access to shared class-level dicts
        self._lock = threading.Lock()
        # Per-conversation locks to prevent concurrent mutations on the same instance
        self._conversation_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)

        # Cache and instance management
        self.fsm_cache: dict[str, FSMDefinition] = {}
        self.instances: dict[str, FSMInstance] = {}

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
        with self._lock:
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

    def get_current_state(self, instance: FSMInstance, conversation_id: str | None = None) -> State:
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
            initial_context: dict[str, Any] | None = None
    ) -> tuple[str, str]:
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
        with self._lock:
            self.instances[conversation_id] = instance

        # Execute start conversation handlers
        try:
            self._execute_handlers(
                HandlerTiming.START_CONVERSATION,
                conversation_id,
                current_state=None,
                target_state=instance.current_state
            )
        except Exception as e:
            with self._lock:
                self.instances.pop(conversation_id, None)
            logger.error(f"START_CONVERSATION handler failed: {str(e)}")
            raise FSMError(f"Failed to start conversation: {str(e)}") from e

        logger.info(f"Started conversation [{conversation_id}] with FSM [{fsm_id}]")

        # Generate initial response using response generation (no user input)
        try:
            response = self._generate_initial_response(instance, conversation_id)
            return conversation_id, response

        except Exception as e:
            # Fire END_CONVERSATION handlers to balance START_CONVERSATION
            try:
                self._execute_handlers(
                    HandlerTiming.END_CONVERSATION,
                    conversation_id,
                    current_state=instance.current_state
                )
            except Exception as cleanup_err:
                logger.warning(f"END_CONVERSATION handler failed during cleanup: {cleanup_err}")
            with self._lock:
                self.instances.pop(conversation_id, None)
            logger.error(f"Error generating initial response: {str(e)}")
            raise FSMError(f"Failed to start conversation: {str(e)}") from e

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

        # Acquire per-conversation lock to prevent concurrent mutations
        conv_lock = self._conversation_locks[conversation_id]
        if not conv_lock.acquire(blocking=False):
            raise FSMError(
                f"Conversation {conversation_id} is already being processed by another thread"
            )
        try:
            return self._process_message_locked(conversation_id, message, log)
        finally:
            conv_lock.release()

    def _process_message_locked(self, conversation_id: str, message: str, log) -> str:
        """Process message while holding the per-conversation lock."""
        instance = self.instances[conversation_id]
        log.info(f"Processing message in state: {instance.current_state}")

        # Check for terminal state before processing
        current_state = self.get_current_state(instance, conversation_id)
        if not current_state.transitions:
            raise FSMError(f"Conversation has ended - current state '{instance.current_state}' is terminal")

        # Add user message to history before processing
        instance.context.conversation.add_user_message(message)

        try:
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

            return response_message

        except FSMError:
            self._rollback_user_message(instance, message, log)
            raise
        except Exception as e:
            log.error(f"Error processing message: {str(e)}\n{traceback.format_exc()}")
            self._rollback_user_message(instance, message, log)

            # Execute error handlers (protected from masking original exception)
            try:
                self._execute_handlers(
                    HandlerTiming.ERROR,
                    conversation_id,
                    current_state=instance.current_state,
                    error_context={"error": str(e), "traceback": traceback.format_exc()}
                )
            except Exception as handler_err:
                log.warning(f"Error handler raised an exception, preserving original error: {handler_err}")

            raise FSMError(f"Failed to process message: {str(e)}") from e

    @staticmethod
    def _rollback_user_message(instance: FSMInstance, message: str, log) -> None:
        """Remove user message from history to avoid duplicates on retry."""
        try:
            exchanges = instance.context.conversation.exchanges
            if exchanges and "user" in exchanges[-1] and exchanges[-1]["user"] == message:
                exchanges.pop()
        except Exception as rollback_err:
            log.warning(f"Failed to rollback user message from conversation history: {rollback_err}")

    def _execute_extraction_and_transition_pass(
            self,
            instance: FSMInstance,
            user_message: str,
            conversation_id: str
    ) -> tuple[DataExtractionResponse, bool, str | None]:
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

            # Re-check after cleaning — all keys may have been removed
            if extraction_response.extracted_data:
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
    ) -> tuple[bool, str | None]:
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

        # Execute transition if target determined (including self-transitions)
        if target_state:
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

        # Execute pre-transition handlers.
        # Raising from a PRE_TRANSITION handler will block the transition
        # and propagate the exception to the caller.
        self._execute_handlers(
            HandlerTiming.PRE_TRANSITION,
            conversation_id,
            current_state=old_state,
            target_state=target_state
        )

        # Perform state transition and post-transition handlers atomically:
        # if post-transition handlers fail, rollback the state change.
        old_context_meta = {
            "_previous_state": instance.context.data.get("_previous_state"),
            "_current_state": instance.context.data.get("_current_state"),
            "_transition_timestamp": instance.context.data.get("_transition_timestamp"),
        }

        instance.current_state = target_state
        instance.context.data.update({
            "_previous_state": old_state,
            "_current_state": target_state,
            "_transition_timestamp": time.time()
        })

        try:
            # Execute post-transition handlers (current_state is the NEW state after transition)
            self._execute_handlers(
                HandlerTiming.POST_TRANSITION,
                conversation_id,
                current_state=target_state,
                target_state=target_state
            )
        except Exception as handler_err:
            # Rollback state change on post-transition handler failure
            log.warning(f"POST_TRANSITION handler failed ({type(handler_err).__name__}: {handler_err}), rolling back state from {target_state} to {old_state}")
            instance.current_state = old_state
            instance.context.data.update(old_context_meta)
            raise

        log.info(f"State transition executed: {old_state} -> {target_state}")

    def _execute_response_generation_pass(
            self,
            instance: FSMInstance,
            user_message: str,
            extraction_response: DataExtractionResponse,
            transition_occurred: bool,
            previous_state: str | None,
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
            current_state: str | None = None,
            target_state: str | None = None,
            updated_keys: set[str] | None = None,
            error_context: dict[str, Any] | None = None
    ) -> None:
        """Execute handlers at specified timing point."""
        if conversation_id not in self.instances:
            return

        instance = self.instances[conversation_id]
        context = copy.deepcopy(instance.context.data)

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

            # Merge handler results into instance context.
            # Handlers return a delta dict of keys to update.
            # Convention: a handler returning a key with value None requests deletion.
            if updated_context:
                for key, value in updated_context.items():
                    if value is None:
                        instance.context.data.pop(key, None)
                    else:
                        instance.context.data[key] = value

        except Exception as e:
            logger.error(f"Handler execution error at {timing.name}: {str(e)}")
            if self.handler_system.error_mode == "raise":
                raise

    def _clean_empty_context_keys(
            self,
            data: dict[str, Any],
            conversation_id: str,
            remove_none_values: bool = True
    ) -> dict[str, Any]:
        """
        Clean invalid keys from context data.

        Only strips None values and keys with internal prefix patterns.
        Empty lists and empty strings are preserved as they can be
        semantically meaningful (e.g., ``{"allergies": []}`` means "no allergies").

        Args:
            data: Dictionary to clean
            conversation_id: For logging context
            remove_none_values: Remove keys with None values

        Returns:
            Cleaned dictionary with invalid keys removed
        """
        from .constants import INTERNAL_KEY_PREFIXES, FORBIDDEN_CONTEXT_PATTERNS

        log = logger.bind(conversation_id=conversation_id)
        cleaned = {}
        removed_keys = []
        warned_keys = []

        for key, value in data.items():
            should_remove = False
            removal_reason = ""

            # Check for None values
            if remove_none_values and value is None:
                should_remove = True
                removal_reason = "None value"

            # Check for internal prefix patterns
            elif any(key.startswith(prefix) for prefix in INTERNAL_KEY_PREFIXES):
                should_remove = True
                removal_reason = "internal key prefix"

            # Keep the key-value pair
            if not should_remove:
                cleaned[key] = value
                # Warn on forbidden context patterns (security)
                if any(re.match(pattern, key, re.IGNORECASE) for pattern in FORBIDDEN_CONTEXT_PATTERNS):
                    warned_keys.append(key)
            else:
                removed_keys.append(f"{key} ({removal_reason})")

        if warned_keys:
            log.warning(
                f"Context contains keys matching forbidden security patterns: {warned_keys}. "
                "Storing sensitive data (passwords, secrets, tokens, API keys) in FSM context "
                "is a security risk."
            )

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

        is_ended = not current_state.transitions
        if is_ended:
            log.info(f"Conversation reached terminal state: {instance.current_state}")

        return is_ended

    @with_conversation_context
    def get_conversation_data(self, conversation_id: str, log=None) -> dict[str, Any]:
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
    def get_conversation_history(self, conversation_id: str, log=None) -> list[dict[str, str]]:
        """Get conversation history."""
        if conversation_id not in self.instances:
            raise ValueError(f"Conversation {conversation_id} not found")

        instance = self.instances[conversation_id]
        return instance.context.conversation.get_recent()

    @with_conversation_context
    def update_conversation_context(
            self,
            conversation_id: str,
            context_update: dict[str, Any],
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
        with self._lock:
            self.instances.pop(conversation_id, None)
        log.info(f"Conversation {conversation_id} ended")

    @with_conversation_context
    def get_complete_conversation(self, conversation_id: str, log=None) -> dict[str, Any]:
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
                "is_terminal": not current_state.transitions
            },
            "collected_data": dict(instance.context.data),
            "conversation_history": instance.context.conversation.get_recent(),
            "metadata": dict(instance.context.metadata),
            "last_extraction_response": instance.last_extraction_response.model_dump() if instance.last_extraction_response else None,
            "last_transition_decision": instance.last_transition_decision.model_dump() if instance.last_transition_decision else None,
            "last_response_generation": instance.last_response_generation.model_dump() if instance.last_response_generation else None
        }
