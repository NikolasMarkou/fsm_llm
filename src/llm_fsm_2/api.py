"""
Enhanced API Module for LLM-FSM: Stateful Conversational AI

This module implements the main API interface for the LLM-FSM library, providing developers
with a powerful framework for building stateful conversational AI applications. The enhanced
implementation features a sophisticated 2-pass architecture that separates data extraction,
transition evaluation, and response generation for improved conversation quality and consistency.

Core Architecture
-----------------
The enhanced API implements a 2-pass processing model:

1. **First Pass - Analysis & Transition**:
   - Data extraction from user input using specialized prompts
   - Transition evaluation using configurable logic
   - State management and context updates

2. **Second Pass - Response Generation**:
   - Response generation based on new state and updated context
   - Enhanced prompt building with rich context awareness
   - Consistent, contextually-appropriate responses

Key Features
------------
- **FSM Stacking**: Modular conversation design with push/pop FSM operations
- **Enhanced Context Management**: Sophisticated context inheritance and merging strategies
- **Flexible Handler System**: Extensible event-driven handler architecture
- **Advanced Transition Logic**: JsonLogic-based conditional transitions with custom evaluators
- **Multi-LLM Support**: Pluggable LLM interfaces with default LiteLLM integration
- **Conversation Persistence**: Comprehensive conversation state and history management
- **Error Handling**: Robust error handling with detailed logging and recovery mechanisms

Usage Examples
--------------
Basic conversation with single FSM:

.. code-block:: python

    from llm_fsm import API

    # Initialize from FSM definition file
    api = API.from_file("conversation_fsm.json", model="gpt-4")

    # Start conversation
    conversation_id, initial_response = api.start_conversation()
    print(f"Bot: {initial_response}")

    # Process user messages
    response = api.converse("Hello there!", conversation_id)
    print(f"Bot: {response}")

Advanced FSM stacking for modular conversations:

.. code-block:: python

    # Start main conversation
    conversation_id, response = api.start_conversation()

    # Push specialized FSM for address collection
    address_response = api.push_fsm(
        conversation_id=conversation_id,
        new_fsm_definition="address_collection_fsm.json",
        shared_context_keys=["user_name", "email"],
        preserve_history=True
    )

    # Collect address information...
    # When done, pop back to main FSM
    resume_response = api.pop_fsm(
        conversation_id=conversation_id,
        merge_strategy="update"  # Merge collected address data
    )

Custom handler integration:

.. code-block:: python

    # Create and register custom handler
    validation_handler = (api.create_handler("AddressValidator")
                         .on_timing(HandlerTiming.POST_TRANSITION)
                         .when_state("address_confirmation")
                         .execute(validate_address_function)
                         .build())

    api.register_handler(validation_handler)
"""

import os
import json
import hashlib
from enum import Enum
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, Tuple, List, Union

# --------------------------------------------------------------
# local imports
# --------------------------------------------------------------

from .fsm import FSMManager
from .llm import LiteLLMInterface, LLMInterface
from .prompts import (
    DataExtractionPromptBuilder,
    ResponseGenerationPromptBuilder,
    TransitionPromptBuilder
)
from .definitions import FSMDefinition, FSMError
from .logging import logger, handle_conversation_errors
from .transition_evaluator import TransitionEvaluator, TransitionEvaluatorConfig
from .handlers import HandlerSystem, FSMHandler, HandlerBuilder, create_handler

# --------------------------------------------------------------


class FSMStackFrame(BaseModel):
    """Represents a single FSM in the conversation stack."""

    fsm_definition: Union[FSMDefinition, Dict[str, Any], str]
    conversation_id: str
    return_context: Dict[str, Any] = Field(default_factory=dict)
    entry_point: Optional[str] = None
    shared_context_keys: List[str] = Field(default_factory=list)
    preserve_history: bool = False

    class Config:
        arbitrary_types_allowed = True

# --------------------------------------------------------------


class ContextMergeStrategy(Enum):
    """Context merge strategies for FSM stack operations."""

    UPDATE = "update"
    PRESERVE = "preserve"
    SELECTIVE = "selective"

    @classmethod
    def from_string(cls, value: Union[str, 'ContextMergeStrategy']) -> 'ContextMergeStrategy':
        """Convert string or enum to ContextMergeStrategy."""
        if isinstance(value, cls):
            return value

        if isinstance(value, str):
            try:
                return cls(value.lower().strip())
            except ValueError:
                valid_values = [e.value for e in cls]
                raise ValueError(f"Invalid merge strategy '{value}'. Must be one of: {valid_values}")

        raise ValueError(f"Invalid type for merge strategy: {type(value)}")


# --------------------------------------------------------------
# Main API Class
# --------------------------------------------------------------

class API:
    """
    Enhanced API for Improved 2-Pass LLM-FSM Architecture.

    This class provides a backward-compatible interface while internally
    implementing the improved 2-pass architecture for better conversation quality
    and response generation after transition evaluation.
    """

    def __init__(self,
                 fsm_definition: Union[FSMDefinition, Dict[str, Any], str],
                 llm_interface: Optional[LLMInterface] = None,
                 model: Optional[str] = None,
                 api_key: Optional[str] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None,
                 max_history_size: int = 5,
                 max_message_length: int = 1000,
                 handlers: Optional[List[FSMHandler]] = None,
                 handler_error_mode: str = "continue",
                 transition_config: Optional[TransitionEvaluatorConfig] = None,
                 **llm_kwargs):
        """
        Initialize API with improved 2-pass architecture.

        Args:
            fsm_definition: FSM definition (object, dict, or file path)
            llm_interface: Optional custom LLM interface
            model: LLM model name (if using default interface)
            api_key: API key (if using default interface)
            temperature: LLM temperature parameter
            max_tokens: Maximum tokens for LLM responses
            max_history_size: Maximum conversation history size
            max_message_length: Maximum message length
            handlers: Optional list of handlers
            handler_error_mode: Handler error handling mode
            transition_config: Configuration for transition evaluation
            **llm_kwargs: Additional LLM parameters
        """
        # Handle LLM interface initialization
        if llm_interface is not None:
            if not isinstance(llm_interface, LLMInterface):
                raise ValueError(f"llm_interface must be an instance of LLMInterface")
            self.llm_interface = llm_interface
            logger.info(f"API initialized with custom LLM interface: {type(llm_interface).__name__}")
        else:
            # Create default interface
            model = model or "gpt-4o-mini"
            temperature = temperature if temperature is not None else 0.5
            max_tokens = max_tokens if max_tokens is not None else 1000

            self.llm_interface = LiteLLMInterface(
                model=model,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                **llm_kwargs
            )
            logger.info(f"API initialized with default LiteLLM interface, model={model}")

        # Process FSM definition
        self.fsm_definition, self.fsm_id = self.process_fsm_definition(fsm_definition)

        # Create enhanced prompt builders
        data_extraction_prompt_builder = DataExtractionPromptBuilder()
        response_generation_prompt_builder = ResponseGenerationPromptBuilder()
        transition_prompt_builder = TransitionPromptBuilder()

        # Create transition evaluator
        evaluator_config = transition_config or TransitionEvaluatorConfig()
        transition_evaluator = TransitionEvaluator(evaluator_config)

        # Create custom FSM loader
        def custom_fsm_loader(fsm_id: str) -> FSMDefinition:
            if fsm_id == self.fsm_id:
                return self.fsm_definition
            elif hasattr(self, '_temp_fsm_definitions') and fsm_id in self._temp_fsm_definitions:
                return self._temp_fsm_definitions[fsm_id]
            else:
                from .utilities import load_fsm_definition
                return load_fsm_definition(fsm_id)

        # Initialize enhanced FSM manager with improved 2-pass architecture
        self.fsm_manager = FSMManager(
            fsm_loader=custom_fsm_loader,
            llm_interface=self.llm_interface,
            data_extraction_prompt_builder=data_extraction_prompt_builder,
            response_generation_prompt_builder=response_generation_prompt_builder,
            transition_prompt_builder=transition_prompt_builder,
            transition_evaluator=transition_evaluator,
            max_history_size=max_history_size,
            max_message_length=max_message_length
        )

        # Initialize handler system
        self.handler_system = HandlerSystem(error_mode=handler_error_mode)

        # Register provided handlers
        if handlers:
            for handler in handlers:
                self.register_handler(handler)

        # Set handler system in FSM manager
        if hasattr(self.fsm_manager, 'handler_system'):
            self.fsm_manager.handler_system = self.handler_system

        # FSM stacking support
        self.active_conversations: Dict[str, bool] = {}
        self.conversation_stacks: Dict[str, List[FSMStackFrame]] = {}
        self._temp_fsm_definitions: Dict[str, FSMDefinition] = {}

        logger.info(f"Enhanced API fully initialized with improved 2-pass architecture")

    @classmethod
    def process_fsm_definition(
            cls,
            fsm_definition: Union[FSMDefinition, Dict[str, Any], str]
    ) -> Tuple[FSMDefinition, str]:
        """Process FSM definition input and return standardized format."""
        if isinstance(fsm_definition, FSMDefinition):
            fsm_def = fsm_definition
            fsm_id = f"fsm_def_{fsm_def.name}_{hash(str(fsm_def.model_dump()))}"

        elif isinstance(fsm_definition, dict):
            try:
                fsm_def = FSMDefinition(**fsm_definition)
                content_hash = hashlib.md5(
                    json.dumps(fsm_definition, sort_keys=True).encode()
                ).hexdigest()[:8]
                fsm_id = f"fsm_dict_{fsm_def.name}_{content_hash}"
            except Exception as e:
                raise ValueError(f"Invalid FSM definition dictionary: {str(e)}")

        elif isinstance(fsm_definition, str):
            try:
                from .utilities import load_fsm_from_file
                fsm_def = load_fsm_from_file(fsm_definition)
                fsm_id = f"fsm_file_{fsm_definition}"
            except Exception as e:
                raise ValueError(f"Failed to load FSM from file '{fsm_definition}': {str(e)}")

        else:
            raise ValueError(
                f"Invalid FSM definition type: {type(fsm_definition)}. "
                f"Must be FSMDefinition, dict, or str"
            )

        return fsm_def, fsm_id

    @classmethod
    def from_file(cls, path: Union[Path, str], **kwargs) -> 'API':
        """Create API instance from FSM definition file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"FSM definition file not found: {path}")
        return cls(fsm_definition=path, **kwargs)

    @classmethod
    def from_definition(cls, fsm_definition: Union[FSMDefinition, Dict[str, Any]], **kwargs) -> 'API':
        """Create API instance from FSM definition object or dictionary."""
        return cls(fsm_definition=fsm_definition, **kwargs)

    def start_conversation(self, initial_context: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
        """
        Start new conversation with improved 2-pass architecture.

        Args:
            initial_context: Optional initial context data

        Returns:
            Tuple of (conversation_id, initial_response)
        """
        try:
            # Start conversation using enhanced FSM manager
            conversation_id, response = self.fsm_manager.start_conversation(
                self.fsm_id,
                initial_context=initial_context
            )

            # Track conversation and initialize stack
            self.active_conversations[conversation_id] = True
            self.conversation_stacks[conversation_id] = [
                FSMStackFrame(
                    fsm_definition=self.fsm_definition,
                    conversation_id=conversation_id
                )
            ]

            return conversation_id, response

        except Exception as e:
            logger.error(f"Error starting conversation: {str(e)}")
            raise FSMError(f"Failed to start conversation: {str(e)}")

    def converse(self, user_message: str, conversation_id: str) -> str:
        """
        Process message using improved 2-pass architecture.

        Args:
            user_message: User's message
            conversation_id: Existing conversation ID

        Returns:
            System response
        """
        try:
            current_fsm_id = self._get_current_fsm_conversation_id(conversation_id)
            response = self.fsm_manager.process_message(current_fsm_id, user_message)
            return response
        except ValueError:
            logger.error(f"Invalid conversation ID: {conversation_id}")
            raise ValueError(f"Conversation not found: {conversation_id}")
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            raise FSMError(f"Failed to process message: {str(e)}")

    # ==========================================
    # FSM STACKING METHODS (Enhanced)
    # ==========================================

    def push_fsm(self,
                 conversation_id: str,
                 new_fsm_definition: Union[FSMDefinition, Dict[str, Any], str],
                 context_to_pass: Optional[Dict[str, Any]] = None,
                 return_context: Optional[Dict[str, Any]] = None,
                 entry_point: Optional[str] = None,
                 shared_context_keys: Optional[List[str]] = None,
                 preserve_history: bool = False,
                 inherit_context: bool = True) -> str:
        """Push new FSM onto conversation stack with enhanced context management."""
        if conversation_id not in self.active_conversations:
            raise ValueError(f"Conversation not found: {conversation_id}")

        try:
            # Process new FSM definition
            processed_fsm_def, processed_fsm_id = self.process_fsm_definition(new_fsm_definition)
            self._temp_fsm_definitions[processed_fsm_id] = processed_fsm_def

            # Get current FSM context for inheritance
            current_fsm_id = self._get_current_fsm_conversation_id(conversation_id)
            inherited_context = {}

            if inherit_context:
                try:
                    inherited_context = self.fsm_manager.get_conversation_data(current_fsm_id)
                except Exception as e:
                    logger.warning(f"Could not inherit context: {str(e)}")

            # Merge contexts
            initial_context = {}
            if inherited_context:
                initial_context.update(inherited_context)
            if context_to_pass:
                initial_context.update(context_to_pass)

            # Handle history preservation
            if preserve_history:
                try:
                    history = self.fsm_manager.get_conversation_history(current_fsm_id)
                    initial_context['_inherited_history'] = history
                except Exception as e:
                    logger.warning(f"Could not preserve history: {str(e)}")

            # Start new FSM conversation
            new_conversation_id, response = self.fsm_manager.start_conversation(
                processed_fsm_id,
                initial_context=initial_context
            )

            # Create and push new stack frame
            new_frame = FSMStackFrame(
                fsm_definition=processed_fsm_def,
                conversation_id=new_conversation_id,
                return_context=return_context or {},
                entry_point=entry_point,
                shared_context_keys=shared_context_keys or [],
                preserve_history=preserve_history
            )

            self.conversation_stacks[conversation_id].append(new_frame)

            logger.info(
                f"Pushed new FSM onto conversation {conversation_id}, "
                f"stack depth: {len(self.conversation_stacks[conversation_id])}"
            )

            return response

        except Exception as e:
            logger.error(f"Error pushing FSM: {str(e)}")
            raise FSMError(f"Failed to push FSM: {str(e)}")

    def pop_fsm(self,
                conversation_id: str,
                context_to_return: Optional[Dict[str, Any]] = None,
                merge_strategy: Union[str, ContextMergeStrategy] = ContextMergeStrategy.UPDATE) -> str:
        """Pop current FSM from stack and return to previous with enhanced context handling."""
        if conversation_id not in self.active_conversations:
            raise ValueError(f"Conversation not found: {conversation_id}")

        stack = self.conversation_stacks[conversation_id]
        if len(stack) <= 1:
            raise ValueError("Cannot pop from FSM stack: only one FSM remaining")

        try:
            merge_strategy_enum = ContextMergeStrategy.from_string(merge_strategy)

            # Get current and previous frames
            current_frame = stack[-1]
            previous_frame = stack[-2]

            # Get current FSM context
            current_fsm_context = {}
            try:
                current_fsm_context = self.fsm_manager.get_conversation_data(current_frame.conversation_id)
            except Exception as e:
                logger.warning(f"Could not get current FSM context: {str(e)}")

            # Prepare context to merge back
            context_to_merge = {}

            if current_frame.return_context:
                context_to_merge.update(current_frame.return_context)

            if context_to_return:
                context_to_merge.update(context_to_return)

            if current_frame.shared_context_keys:
                for key in current_frame.shared_context_keys:
                    if key in current_fsm_context:
                        context_to_merge[key] = current_fsm_context[key]

            # Apply merge strategy
            if context_to_merge:
                self._merge_context_with_strategy(
                    previous_frame.conversation_id,
                    context_to_merge,
                    merge_strategy_enum
                )

            # Handle history preservation
            if current_frame.preserve_history:
                try:
                    current_history = self.fsm_manager.get_conversation_history(current_frame.conversation_id)
                    summary_context = {
                        '_sub_conversation_summary': {
                            'fsm_type': str(current_frame.fsm_definition),
                            'final_context': current_fsm_context,
                            'exchange_count': len(current_history)
                        }
                    }
                    self._merge_context_with_strategy(
                        previous_frame.conversation_id,
                        summary_context,
                        ContextMergeStrategy.UPDATE
                    )
                except Exception as e:
                    logger.warning(f"Could not preserve sub-conversation summary: {str(e)}")

            # Remove current FSM from stack
            stack.pop()

            # End the popped FSM conversation
            self.fsm_manager.end_conversation(current_frame.conversation_id)

            # Generate resume message
            response = self._generate_resume_message(previous_frame, context_to_merge)

            logger.info(f"Popped FSM from conversation {conversation_id}, stack depth: {len(stack)}")

            return response

        except Exception as e:
            logger.error(f"Error popping FSM: {str(e)}")
            raise FSMError(f"Failed to pop FSM: {str(e)}")

    def _merge_context_with_strategy(
            self,
            conversation_id: str,
            context_to_merge: Dict[str, Any],
            strategy: ContextMergeStrategy = ContextMergeStrategy.UPDATE
    ) -> None:
        """Merge context using specified strategy."""
        if not context_to_merge:
            return

        try:
            current_context = self.fsm_manager.get_conversation_data(conversation_id)
        except Exception:
            current_context = {}

        if strategy == ContextMergeStrategy.UPDATE:
            merged_context = {**current_context, **context_to_merge}
        elif strategy == ContextMergeStrategy.PRESERVE:
            merged_context = current_context.copy()
            for key, value in context_to_merge.items():
                if key not in current_context:
                    merged_context[key] = value
        elif strategy == ContextMergeStrategy.SELECTIVE:
            merged_context = current_context.copy()
            if conversation_id in self.conversation_stacks:
                stack = self.conversation_stacks[conversation_id]
                if stack:
                    current_frame = stack[-1]
                    shared_keys = current_frame.shared_context_keys
                    for key in shared_keys:
                        if key in context_to_merge:
                            merged_context[key] = context_to_merge[key]
            else:
                merged_context = {**current_context, **context_to_merge}
        else:
            raise ValueError(f"Unknown merge strategy {strategy}")

        self.fsm_manager.update_conversation_context(conversation_id, merged_context)

    def _generate_resume_message(self, previous_frame: FSMStackFrame, merged_context: Dict[str, Any]) -> str:
        """Generate message for resuming previous FSM."""
        if merged_context:
            context_summary = ", ".join([f"{k}={v}" for k, v in list(merged_context.items())[:3]])
            if len(merged_context) > 3:
                context_summary += f"... (+{len(merged_context) - 3} more)"
            return f"Resumed previous conversation with updated context: {context_summary}"
        else:
            return "Resumed previous conversation."

    def _get_current_fsm_conversation_id(self, conversation_id: str) -> str:
        """Get conversation ID of current active FSM (top of stack)."""
        if conversation_id not in self.conversation_stacks:
            return conversation_id

        stack = self.conversation_stacks[conversation_id]
        return stack[-1].conversation_id if stack else conversation_id

    # ==========================================
    # HANDLER MANAGEMENT METHODS
    # ==========================================

    def register_handler(self, handler: FSMHandler) -> None:
        """Register handler with the system."""
        self.handler_system.register_handler(handler)

    def register_handlers(self, handlers: List[FSMHandler]) -> None:
        """Register multiple handlers."""
        for handler in handlers:
            self.register_handler(handler)

    def create_handler(self, name: str = "CustomHandler") -> HandlerBuilder:
        """Create new handler using fluent builder."""
        return create_handler(name)

    # ==========================================
    # CONVERSATION MANAGEMENT METHODS
    # ==========================================

    @handle_conversation_errors
    def get_data(self, conversation_id: str) -> Dict[str, Any]:
        """Get collected data from current FSM."""
        current_fsm_id = self._get_current_fsm_conversation_id(conversation_id)
        return self.fsm_manager.get_conversation_data(current_fsm_id)

    @handle_conversation_errors
    def has_conversation_ended(self, conversation_id: str) -> bool:
        """Check if current FSM has ended."""
        current_fsm_id = self._get_current_fsm_conversation_id(conversation_id)
        return self.fsm_manager.has_conversation_ended(current_fsm_id)

    @handle_conversation_errors
    def get_current_state(self, conversation_id: str) -> str:
        """Get current state of active FSM."""
        current_fsm_id = self._get_current_fsm_conversation_id(conversation_id)
        return self.fsm_manager.get_conversation_state(current_fsm_id)

    @handle_conversation_errors
    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, str]]:
        """Get conversation history for current FSM."""
        current_fsm_id = self._get_current_fsm_conversation_id(conversation_id)
        return self.fsm_manager.get_conversation_history(current_fsm_id)

    @handle_conversation_errors("Failed to end conversation")
    def end_conversation(self, conversation_id: str) -> None:
        """End conversation and clean up all FSMs in stack."""
        if conversation_id in self.conversation_stacks:
            for frame in self.conversation_stacks[conversation_id]:
                try:
                    self.fsm_manager.end_conversation(frame.conversation_id)
                except Exception as e:
                    logger.warning(f"Error ending FSM {frame.conversation_id}: {str(e)}")
        else:
            self.fsm_manager.end_conversation(conversation_id)

        if conversation_id in self.active_conversations:
            del self.active_conversations[conversation_id]

    def list_active_conversations(self) -> List[str]:
        """List all active conversation IDs."""
        return list(self.active_conversations.keys())

    def get_llm_interface(self) -> LLMInterface:
        """Get current LLM interface."""
        return self.llm_interface

    def close(self) -> None:
        """Clean up all active conversations."""
        for conversation_id in list(self.active_conversations.keys()):
            try:
                self.end_conversation(conversation_id)
            except Exception as e:
                logger.warning(f"Error ending conversation {conversation_id} during cleanup: {str(e)}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.close()

# --------------------------------------------------------------
