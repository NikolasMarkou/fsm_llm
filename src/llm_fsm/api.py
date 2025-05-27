"""
LLM-FSM Simplified API: A comprehensive framework for building stateful conversational AI
systems using Finite State Machines powered by Large Language Models.

This module serves as the primary entry point for the LLM-FSM framework, providing a
simplified yet powerful interface for creating, managing, and orchestrating complex
conversational flows with support for FSM stacking, context handover, and multi-turn
state management.

Overview
--------
The LLM-FSM framework addresses the fundamental challenge of statelessness in Large
Language Models by combining the structural predictability of Finite State Machines
with the natural language understanding capabilities of modern LLMs. This hybrid
approach enables developers to build robust conversational applications with:

- **Predictable Conversation Flows**: Clear state transitions and conversation paths
- **Persistent Context Management**: Maintain information across multiple interactions
- **Dynamic State Transitions**: LLM-driven decision making for state changes
- **Natural Language Understanding**: Leverage LLM capabilities for user intent recognition
- **Flexible Response Generation**: Context-aware response generation with persona support

Key Features
------------
**FSM Stacking & Context Handover**
    - Stack multiple FSMs to create complex conversational workflows
    - Comprehensive context inheritance between parent and child FSMs
    - Flexible merge strategies for context synchronization
    - Selective context sharing with fine-grained control
    - Conversation history preservation across FSM transitions

**Advanced State Management**
    - Support for conditional transitions with JsonLogic expressions
    - Custom handler system for integrating external logic
    - Hierarchical state machine support for complex workflows
    - Real-time state validation and error handling

**LLM Integration**
    - Provider-agnostic LLM interface supporting OpenAI, Anthropic, and others
    - Custom LLM interface support for specialized implementations
    - Configurable parameters (temperature, max_tokens, etc.)
    - Built-in prompt engineering for optimal FSM interaction

**Production-Ready Features**
    - Comprehensive logging and debugging support
    - Conversation persistence and resumption
    - Memory-efficient conversation history management
    - Context manager support for resource cleanup
    - Thread-safe conversation management

Architecture
------------
The API class serves as the orchestrator for multiple components:

1. **FSM Manager**: Handles individual FSM instances and their lifecycle
2. **LLM Interface**: Abstracts communication with various LLM providers
3. **Context Manager**: Manages conversation context and history
4. **Stack Manager**: Orchestrates FSM stacking and context handover
5. **Handler System**: Integrates custom logic at various execution points

Usage Patterns
--------------
**Simple Conversational Flow**::

    api = API.from_file("greeting.json", model="gpt-4o")
    conv_id, response = api.start_conversation()

    while not api.has_conversation_ended(conv_id):
        user_input = input("You: ")
        response = api.converse(user_input, conv_id)
        print(f"Bot: {response}")

**Complex Multi-FSM Workflow**::

    # Start main conversation
    api = API.from_file("main.json", model="gpt-4o")
    conv_id, response = api.start_conversation({"user_id": "123"})

    # Push specialized sub-FSM for detailed information gathering
    api.push_fsm(
        conv_id,
        "details_form.json",
        shared_context_keys=["user_id", "preferences"],
        preserve_history=True,
        inherit_context=True
    )

    # Interact with sub-FSM
    response = api.converse("I need to update my profile", conv_id)

    # Return to main FSM with collected data
    api.pop_fsm(conv_id, context_to_return={"profile_updated": True})

**Custom LLM Integration**::

    class CustomLLMInterface(LLMInterface):
        def send_request(self, system_prompt, user_message):
            # Custom implementation
            return {"response": "..."}

    api = API.from_file("fsm.json", llm_interface=CustomLLMInterface())

Context Management
------------------
The framework provides sophisticated context management with multiple strategies:

- **Inheritance**: Child FSMs automatically inherit parent context
- **Selective Sharing**: Specify exact keys to share between FSMs
- **Merge Strategies**: Control how context is merged when returning from sub-FSMs
- **History Preservation**: Optionally maintain conversation history across transitions
- **Automatic Synchronization**: Keep shared context keys in sync across the stack

Error Handling
--------------
Comprehensive error handling with specific exceptions:

- **FSMError**: General FSM-related errors
- **ValueError**: Invalid conversation IDs or parameters
- **FileNotFoundError**: Missing FSM definition files
- **IOError**: File system operations failures

All errors include detailed logging and context information for debugging.

Thread Safety
-------------
The API class is designed to be thread-safe for concurrent conversation management.
Each conversation is isolated and can be safely managed from different threads.

Performance Considerations
--------------------------
- Efficient memory management with configurable history limits
- Lazy loading of FSM definitions
- Optimized context serialization for large datasets
- Minimal overhead for non-stacked conversations

Examples
--------
See the examples/ directory in the repository for comprehensive usage examples,
including basic greetings, form filling, product recommendations, and complex
multi-step workflows.

Dependencies
------------
- pydantic: For data validation and serialization
- typing: For type hints and annotations
- Standard library modules: os, json for file operations

Notes
-----
This module requires Python 3.8+ and is compatible with the broader LLM-FSM
ecosystem including the workflows extension for building automated processes.

For detailed information about FSM definition formats, see the documentation
in the docs/ directory of the repository.
"""

import os
import json
import hashlib
from enum import Enum
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, Tuple, List, Union, Callable

# --------------------------------------------------------------
# local imports
# --------------------------------------------------------------

from .fsm import FSMManager
from .llm import LiteLLMInterface, LLMInterface
from .definitions import FSMDefinition, FSMError
from .logging import logger, handle_conversation_errors
from .handlers import HandlerSystem, FSMHandler, HandlerBuilder, create_handler, HandlerTiming

# --------------------------------------------------------------


class FSMStackFrame(BaseModel):
    """
    Represents a single FSM in the conversation stack.

    Attributes:
        fsm_definition: The FSM definition for this frame
        conversation_id: The conversation ID for this FSM instance
        return_context: Context to pass back when returning from a sub-FSM
        entry_point: Optional state to resume when returning to this FSM
        shared_context_keys: Keys that should be automatically shared between FSMs
        preserve_history: Whether to preserve conversation history when stacking
    """
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
    """
    Enumeration of available context merge strategies when popping FSMs from the stack.

    Attributes:
        UPDATE: Update parent context with child context (child values take precedence)
        PRESERVE: Preserve parent context, only add new keys from child
        SELECTIVE: Only merge keys specified in shared_context_keys
    """
    UPDATE = "update"
    PRESERVE = "preserve"
    SELECTIVE = "selective"

    @classmethod
    def from_string(cls, value: Union[str, 'ContextMergeStrategy']) -> 'ContextMergeStrategy':
        """
        Convert string or enum to ContextMergeStrategy enum.

        Args:
            value: String representation or existing enum value

        Returns:
            ContextMergeStrategy enum value

        Raises:
            ValueError: If the string value is not a valid strategy
        """
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


class API:
    """
    Main entry point for working with LLM-FSM.

    This class provides a simplified interface for using the LLM-FSM framework,
    hiding the complexity of the underlying components while maintaining all
    the power and flexibility of the full system.

    It can handle multiple conversations, each one accessed through the conversation_id,
    with support for FSM stacking and comprehensive context handover between FSMs.
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
                 **llm_kwargs):
        """
        Initialize an LLM-FSM instance.

        Args:
            fsm_definition: FSM definition as an object, dictionary, or path to file
            llm_interface: Optional custom LLM interface instance. If provided, other LLM parameters are ignored.
            model: The LLM model to use (e.g., "gpt-4o", "claude-3-opus"). Used only if llm_interface is None.
            api_key: Optional API key (will use environment variables if not provided). Used only if llm_interface is None.
            temperature: LLM temperature parameter (0.0-1.0). Used only if llm_interface is None.
            max_tokens: Maximum tokens for LLM responses. Used only if llm_interface is None.
            max_history_size: Maximum number of exchanges to keep in conversation history
            max_message_length: Maximum length of messages in characters
            handlers: Optional list of handlers to register with the FSM manager
            handler_error_mode: How to handle handler errors ("continue", "raise", "skip")
            **llm_kwargs: Additional keyword arguments to pass to LiteLLMInterface constructor if llm_interface is None

        Raises:
            ValueError: If neither llm_interface nor model is provided
        """
        # Handle LLM interface initialization
        if llm_interface is not None:
            # Use provided custom LLM interface
            if not isinstance(llm_interface, LLMInterface):
                raise ValueError(f"llm_interface {type(llm_interface).__name__} must be an instance of LLMInterface")
            self.llm_interface = llm_interface
            logger.info(f"LLM_FSM initialized with custom LLM interface: {type(llm_interface).__name__}")
        else:
            # Create default LiteLLMInterface with provided parameters
            # Set defaults for LiteLLMInterface parameters
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
            logger.info(f"LLM_FSM initialized with default LiteLLM interface, model={model}")

        # Process and store the FSM definition
        self.fsm_definition, self.fsm_id = self.process_fsm_definition(fsm_definition)

        # Create a custom FSM loader that can handle our processed definition
        def custom_fsm_loader(fsm_id: str) -> FSMDefinition:
            if fsm_id == self.fsm_id:
                return self.fsm_definition
            elif hasattr(self, '_temp_fsm_definitions') and fsm_id in self._temp_fsm_definitions:
                return self._temp_fsm_definitions[fsm_id]
            else:
                # Fallback to default loading for other IDs (in case of stacking)
                from .utilities import load_fsm_definition
                return load_fsm_definition(fsm_id)

        # Initialize FSM manager
        self.fsm_manager = FSMManager(
            fsm_loader=custom_fsm_loader,
            llm_interface=self.llm_interface,
            max_history_size=max_history_size,
            max_message_length=max_message_length
        )

        # Initialize and configure handler system
        self.handler_system = HandlerSystem(error_mode=handler_error_mode)

        # Register provided handlers
        if handlers:
            for handler in handlers:
                self.register_handler(handler)

        # Set the handler system in FSM manager
        if hasattr(self.fsm_manager, 'handler_system'):
            self.fsm_manager.handler_system = self.handler_system

        # Store active conversations and their FSM stacks
        self.active_conversations: Dict[str, bool] = {}
        self.conversation_stacks: Dict[str, List[FSMStackFrame]] = {}

        # Store temporary FSM definitions for stacking
        self._temp_fsm_definitions: Dict[str, FSMDefinition] = {}

        logger.info(f"LLM_FSM fully initialized with max_history_size={max_history_size}")

    @classmethod
    def process_fsm_definition(
            cls,
            fsm_definition: Union[FSMDefinition, Dict[str, Any], str]) -> Tuple[FSMDefinition, str]:
        """
        Process the FSM definition input and return a standardized FSMDefinition object and ID.

        Args:
            fsm_definition: FSM definition in various formats

        Returns:
            Tuple of (FSMDefinition object, unique ID for caching)

        Raises:
            ValueError: If the FSM definition is invalid
        """
        if isinstance(fsm_definition, FSMDefinition):
            # Already an FSMDefinition object
            fsm_def = fsm_definition
            fsm_id = f"fsm_def_{fsm_def.name}_{hash(str(fsm_def.model_dump()))}"

        elif isinstance(fsm_definition, dict):
            # Dictionary definition - convert to FSMDefinition
            try:
                fsm_def = FSMDefinition(**fsm_definition)
                # Create a unique ID based on the content
                content_hash = hashlib.md5(json.dumps(fsm_definition, sort_keys=True).encode()).hexdigest()[:8]
                fsm_id = f"fsm_dict_{fsm_def.name}_{content_hash}"
            except Exception as e:
                raise ValueError(f"Invalid FSM definition dictionary: {str(e)}")

        elif isinstance(fsm_definition, str):
            # String path - load from file
            try:
                from .utilities import load_fsm_from_file
                fsm_def = load_fsm_from_file(fsm_definition)
                fsm_id = f"fsm_file_{fsm_definition}"
            except Exception as e:
                raise ValueError(f"Failed to load FSM from file '{fsm_definition}': {str(e)}")

        else:
            raise ValueError(
                f"Invalid FSM definition type: {type(fsm_definition)}. Must be FSMDefinition, dict, or str")

        return fsm_def, fsm_id

    @classmethod
    def from_file(
            cls,
            path: Union[Path, str], **kwargs) -> 'API':
        """
        Create an LLM-FSM instance from a JSON file.

        Args:
            path: Path to the FSM definition JSON file
            **kwargs: Additional arguments to pass to the constructor (llm_interface, model, api_key, handlers, etc.)

        Returns:
            An initialized API instance

        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"FSM definition file not found: {path}")

        return cls(fsm_definition=path, **kwargs)

    @classmethod
    def from_definition(
            cls,
            fsm_definition: Union[FSMDefinition, Dict[str, Any]],
            **kwargs) -> 'API':
        """
        Create an LLM-FSM instance from an FSM definition object or dictionary.

        Args:
            fsm_definition: FSM definition as an object or dictionary
            **kwargs: Additional arguments to pass to the constructor (llm_interface, model, api_key, handlers, etc.)

        Returns:
            An initialized API instance
        """
        return cls(fsm_definition=fsm_definition, **kwargs)

    def start_conversation(
            self,
            initial_context: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
        """
        Start a new conversation with the FSM.

        This method initializes a new conversation and returns the initial response from the FSM.

        Args:
            initial_context: Optional initial context data for user personalization

        Returns:
            A tuple of (conversation_id, initial_response)

        Raises:
            FSMError: If there's an error starting the conversation
        """
        try:
            # Start new conversation through the FSM manager using the processed FSM ID
            conversation_id, response = self.fsm_manager.start_conversation(
                self.fsm_id,
                initial_context=initial_context
            )

            # Track the conversation and initialize its FSM stack
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

    def converse(self,
                 user_message: str,
                 conversation_id: str) -> str:
        """
        Process a message and return the response,
        this must be an existing already started conversation

        Args:
            user_message: The user's message
            conversation_id: ID for an existing conversation

        Returns:
            response

        Raises:
            ValueError: If the conversation ID is not found or invalid
            FSMError: If there's an error in the FSM processing
        """
        try:
            # Get the current active FSM (top of stack)
            current_fsm_id = self._get_current_fsm_conversation_id(conversation_id)
            response = self.fsm_manager.process_message(current_fsm_id, user_message)
            return response
        except ValueError:
            logger.error(f"Invalid conversation ID: {conversation_id}")
            raise ValueError(f"Conversation not found: {conversation_id}")
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            raise FSMError(f"Failed to process message: {str(e)}")

    def push_fsm(self,
                 conversation_id: str,
                 new_fsm_definition: Union[FSMDefinition, Dict[str, Any], str],
                 context_to_pass: Optional[Dict[str, Any]] = None,
                 return_context: Optional[Dict[str, Any]] = None,
                 entry_point: Optional[str] = None,
                 shared_context_keys: Optional[List[str]] = None,
                 preserve_history: bool = False,
                 inherit_context: bool = True) -> str:
        """
        Push a new FSM onto the conversation stack with comprehensive context handover.

        Args:
            conversation_id: The main conversation ID
            new_fsm_definition: FSM definition for the new FSM to stack
            context_to_pass: Explicit context to pass to the new FSM
            return_context: Context to store for when returning to current FSM
            entry_point: Optional state to resume when returning to current FSM
            shared_context_keys: Keys that should be automatically synced between FSMs
            preserve_history: Whether to copy conversation history to the new FSM
            inherit_context: Whether to inherit all context from current FSM

        Returns:
            Response from the new FSM's initialization

        Raises:
            ValueError: If the conversation ID is not found
            FSMError: If there's an error starting the new FSM
        """
        if conversation_id not in self.active_conversations:
            raise ValueError(f"Conversation not found: {conversation_id}")

        try:
            # Process the new FSM definition
            processed_fsm_def, processed_fsm_id = self.process_fsm_definition(new_fsm_definition)

            # Store the processed FSM definition for the loader
            self._temp_fsm_definitions[processed_fsm_id] = processed_fsm_def

            # Get current FSM's context for inheritance
            current_fsm_id = self._get_current_fsm_conversation_id(conversation_id)
            inherited_context = {}

            if inherit_context:
                try:
                    inherited_context = self.fsm_manager.get_conversation_data(current_fsm_id)
                except Exception as e:
                    logger.warning(f"Could not inherit context from current FSM: {str(e)}")

            # Merge contexts (explicit context_to_pass takes precedence)
            initial_context = {}
            if inherited_context:
                initial_context.update(inherited_context)
            if context_to_pass:
                initial_context.update(context_to_pass)

            # Handle conversation history preservation
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

            # Create new stack frame and push to stack
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
                f"Pushed new FSM onto conversation {conversation_id}, stack depth: {len(self.conversation_stacks[conversation_id])}")
            logger.debug(f"Context passed to new FSM: {list(initial_context.keys())}")

            return response

        except Exception as e:
            logger.error(f"Error pushing FSM: {str(e)}")
            raise FSMError(f"Failed to push FSM: {str(e)}")

    def pop_fsm(self,
                conversation_id: str,
                context_to_return: Optional[Dict[str, Any]] = None,
                merge_strategy: Union[str, ContextMergeStrategy] = ContextMergeStrategy.UPDATE) -> str:
        """
        Pop the current FSM from the stack and return to the previous one with context handover.

        Args:
            conversation_id: The main conversation ID
            context_to_return: Context to merge back into the previous FSM
            merge_strategy: How to merge contexts (ContextMergeStrategy enum or string)
                - UPDATE: Update parent context with returned context
                - PRESERVE: Only add new keys, don't overwrite existing
                - SELECTIVE: Only merge keys specified in shared_context_keys

        Returns:
            Response from resuming the previous FSM

        Raises:
            ValueError: If the conversation ID is not found, stack is empty, or invalid merge strategy
            FSMError: If there's an error popping the FSM
        """
        if conversation_id not in self.active_conversations:
            raise ValueError(f"Conversation not found: {conversation_id}")

        stack = self.conversation_stacks[conversation_id]
        if len(stack) <= 1:
            raise ValueError("Cannot pop from FSM stack: only one FSM remaining")

        try:
            # Convert merge strategy to enum
            merge_strategy_enum = ContextMergeStrategy.from_string(merge_strategy)

            # Get current and previous frames
            current_frame = stack[-1]
            previous_frame = stack[-2]

            # Get current FSM's final context
            current_fsm_context = {}
            try:
                current_fsm_context = self.fsm_manager.get_conversation_data(current_frame.conversation_id)
            except Exception as e:
                logger.warning(f"Could not get current FSM context: {str(e)}")

            # Prepare context to merge back
            context_to_merge = {}

            # Add stored return context
            if current_frame.return_context:
                context_to_merge.update(current_frame.return_context)

            # Add explicitly provided return context
            if context_to_return:
                context_to_merge.update(context_to_return)

            # Add shared context keys from current FSM
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

            # Handle history preservation if it was enabled
            if current_frame.preserve_history:
                try:
                    current_history = self.fsm_manager.get_conversation_history(current_frame.conversation_id)
                    # Add current sub-conversation to parent's context as a summary
                    summary_context = {'_sub_conversation_summary': {
                        'fsm_type': str(current_frame.fsm_definition),
                        'final_context': current_fsm_context,
                        'exchange_count': len(current_history)
                    }}
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
            logger.debug(f"Context merged back: {list(context_to_merge.keys())}")

            return response

        except Exception as e:
            logger.error(f"Error popping FSM: {str(e)}")
            raise FSMError(f"Failed to pop FSM: {str(e)}")

    def sync_shared_context(self, conversation_id: str) -> None:
        """
        Synchronize shared context keys between all FSMs in the stack.

        This method propagates changes in shared context keys from the current FSM
        to all other FSMs in the stack, ensuring consistency.

        Args:
            conversation_id: The conversation ID

        Raises:
            ValueError: If the conversation ID is not found
        """
        if conversation_id not in self.conversation_stacks:
            return  # No stacking, nothing to sync

        stack = self.conversation_stacks[conversation_id]
        if len(stack) <= 1:
            return  # Only one FSM, nothing to sync

        try:
            # Get current FSM context
            current_frame = stack[-1]
            current_context = self.fsm_manager.get_conversation_data(current_frame.conversation_id)

            # Sync with all other FSMs in stack
            for frame in stack[:-1]:  # Exclude current FSM
                if frame.shared_context_keys:
                    context_update = {}
                    for key in frame.shared_context_keys:
                        if key in current_context:
                            context_update[key] = current_context[key]

                    if context_update:
                        self._merge_context_with_strategy(
                            frame.conversation_id,
                            context_update,
                            ContextMergeStrategy.UPDATE
                        )

            logger.debug(f"Synchronized shared context for conversation {conversation_id}")

        except Exception as e:
            logger.warning(f"Error synchronizing shared context: {str(e)}")

    def get_context_flow(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get a summary of how context has flowed between FSMs in the stack.

        Args:
            conversation_id: The conversation ID

        Returns:
            A dictionary showing context flow information

        Raises:
            ValueError: If the conversation ID is not found
        """
        if conversation_id not in self.conversation_stacks:
            return {"stack_depth": 1, "context_flow": "No stacking"}

        stack = self.conversation_stacks[conversation_id]
        flow_info = {
            "stack_depth": len(stack),
            "frames": []
        }

        for i, frame in enumerate(stack):
            try:
                frame_context = self.fsm_manager.get_conversation_data(frame.conversation_id)
                frame_info = {
                    "level": i,
                    "fsm_definition": str(frame.fsm_definition)[:50] + "..." if len(
                        str(frame.fsm_definition)) > 50 else str(frame.fsm_definition),
                    "context_keys": list(frame_context.keys()),
                    "shared_context_keys": frame.shared_context_keys,
                    "return_context_keys": list(frame.return_context.keys()),
                    "preserve_history": frame.preserve_history
                }
                flow_info["frames"].append(frame_info)
            except Exception as e:
                flow_info["frames"].append({
                    "level": i,
                    "error": f"Could not get frame info: {str(e)}"
                })

        return flow_info

    def get_stack_depth(self, conversation_id: str) -> int:
        """
        Get the current FSM stack depth for a conversation.

        Args:
            conversation_id: The conversation ID

        Returns:
            The number of FSMs in the stack

        Raises:
            ValueError: If the conversation ID is not found
        """
        if conversation_id not in self.active_conversations:
            raise ValueError(f"Conversation not found: {conversation_id}")

        return len(self.conversation_stacks[conversation_id])

    def register_handler(self, handler: FSMHandler) -> None:
        """
        Register a handler with the FSM system.

        Args:
            handler: The handler to register
        """
        self.handler_system.register_handler(handler)

    def register_handlers(self, handlers: List[FSMHandler]) -> None:
        """
        Register multiple handlers with the FSM system.

        Args:
            handlers: List of handlers to register
        """
        for handler in handlers:
            self.register_handler(handler)

    def create_handler(self, name: str = "CustomHandler") -> HandlerBuilder:
        """
        Create a new handler using the fluent builder interface.

        Args:
            name: Name for the handler (used in logging)

        Returns:
            HandlerBuilder instance for method chaining
        """
        return create_handler(name)

    def get_registered_handlers(self) -> List[str]:
        """
        Get names of all registered handlers.

        Returns:
            List of handler names
        """
        return [getattr(h, 'name', h.__class__.__name__) for h in self.handler_system.handlers]

    def set_handler_error_mode(self, mode: str) -> None:
        """
        Set how the handler system handles errors.

        Args:
            mode: Error handling mode ("continue", "raise", "skip")

        Raises:
            ValueError: If mode is not valid
        """
        if mode not in ["continue", "raise", "skip"]:
            raise ValueError(f"Invalid error mode: {mode}. Must be 'continue', 'raise', or 'skip'")
        self.handler_system.error_mode = mode

    def add_logging_handler(self,
                            log_timings: Optional[List[HandlerTiming]] = None,
                            log_states: Optional[List[str]] = None,
                            priority: int = 10) -> None:
        """
        Add a convenient logging handler for debugging FSM execution.

        Args:
            log_timings: Specific timing points to log (default: all major timings)
            log_states: Specific states to log (default: all states)
            priority: Handler priority (lower = higher priority)
        """
        if log_timings is None:
            log_timings = [
                HandlerTiming.START_CONVERSATION,
                HandlerTiming.PRE_PROCESSING,
                HandlerTiming.POST_PROCESSING,
                HandlerTiming.PRE_TRANSITION,
                HandlerTiming.POST_TRANSITION,
                HandlerTiming.END_CONVERSATION
            ]

        def log_execution(context: Dict[str, Any]) -> Dict[str, Any]:
            """Log FSM execution details."""
            import datetime
            log_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "context_keys": list(context.keys()),
                "context_size": len(context)
            }
            logger.info(f"FSM Execution Log: {log_entry}")
            return {"_last_logged": log_entry["timestamp"]}

        builder = self.create_handler("FSMLogger").with_priority(priority).at(*log_timings)

        if log_states:
            builder = builder.on_state(*log_states)

        handler = builder.do(log_execution)
        self.register_handler(handler)

    def add_context_validator_handler(self,
                                      required_keys: List[str],
                                      timing: HandlerTiming = HandlerTiming.PRE_PROCESSING,
                                      priority: int = 5) -> None:
        """
        Add a handler that validates required context keys are present.

        Args:
            required_keys: List of context keys that must be present
            timing: When to validate (default: PRE_PROCESSING)
            priority: Handler priority (lower = higher priority)
        """

        def validate_context(context: Dict[str, Any]) -> Dict[str, Any]:
            """Validate required context keys."""
            missing_keys = [key for key in required_keys if key not in context]
            if missing_keys:
                logger.warning(f"Missing required context keys: {missing_keys}")
                return {"_validation_warnings": missing_keys}
            return {"_validation_passed": True}

        handler = self.create_handler("ContextValidator") \
            .with_priority(priority) \
            .at(timing) \
            .do(validate_context)

        self.register_handler(handler)

    def add_state_entry_handler(self,
                                state: str,
                                handler_func: Callable[[Dict[str, Any]], Dict[str, Any]],
                                priority: int = 50) -> None:
        """
        Add a handler that executes when entering a specific state.

        Args:
            state: State ID to watch for entry
            handler_func: Function to execute on state entry
            priority: Handler priority (lower = higher priority)
        """
        handler = self.create_handler(f"StateEntry_{state}") \
            .with_priority(priority) \
            .on_state_entry(state) \
            .do(handler_func)

        self.register_handler(handler)

    def add_context_update_handler(self,
                                   watched_keys: List[str],
                                   handler_func: Callable[[Dict[str, Any]], Dict[str, Any]],
                                   priority: int = 50) -> None:
        """
        Add a handler that executes when specific context keys are updated.

        Args:
            watched_keys: Context keys to watch for updates
            handler_func: Function to execute on key updates
            priority: Handler priority (lower = higher priority)
        """
        handler = self.create_handler(f"ContextUpdate_{'_'.join(watched_keys)}") \
            .with_priority(priority) \
            .on_context_update(*watched_keys) \
            .do(handler_func)

        self.register_handler(handler)

    @handle_conversation_errors
    def get_data(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get collected data from the current active FSM in the conversation.

        Args:
            conversation_id: The conversation ID

        Returns:
            The context data collected during the current FSM conversation

        Raises:
            ValueError: If the conversation ID is not found
        """
        current_fsm_id = self._get_current_fsm_conversation_id(conversation_id)
        return self.fsm_manager.get_conversation_data(current_fsm_id)

    def get_all_stack_data(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Get collected data from all FSMs in the conversation stack.

        Args:
            conversation_id: The conversation ID

        Returns:
            A list of context data from each FSM in the stack (bottom to top)

        Raises:
            ValueError: If the conversation ID is not found
        """
        if conversation_id not in self.conversation_stacks:
            # Fallback for non-stacked conversations
            return [self.get_data(conversation_id)]

        stack_data = []
        for frame in self.conversation_stacks[conversation_id]:
            try:
                data = self.fsm_manager.get_conversation_data(frame.conversation_id)
                stack_data.append(data)
            except Exception as e:
                logger.warning(f"Could not get data for FSM {frame.conversation_id}: {str(e)}")
                stack_data.append({})

        return stack_data

    @handle_conversation_errors
    def has_conversation_ended(self, conversation_id: str) -> bool:
        """
        Check if the current active FSM has ended (reached a terminal state).

        Args:
            conversation_id: The conversation ID

        Returns:
            True if the current FSM has ended, False otherwise

        Raises:
            ValueError: If the conversation ID is not found
        """
        current_fsm_id = self._get_current_fsm_conversation_id(conversation_id)
        return self.fsm_manager.has_conversation_ended(current_fsm_id)

    @handle_conversation_errors
    def get_current_state(self, conversation_id: str) -> str:
        """
        Get the current state of the active FSM in the conversation.

        Args:
            conversation_id: The conversation ID

        Returns:
            The current state ID of the active FSM

        Raises:
            ValueError: If the conversation ID is not found
        """
        current_fsm_id = self._get_current_fsm_conversation_id(conversation_id)
        return self.fsm_manager.get_conversation_state(current_fsm_id)

    @handle_conversation_errors("Failed to end conversation")
    def end_conversation(self, conversation_id: str) -> None:
        """
        Explicitly end a conversation and clean up all FSMs in the stack.

        This marks the conversation as completed but retains the data.
        Use delete_conversation to completely remove all data.

        Args:
            conversation_id: The conversation ID

        Raises:
            ValueError: If the conversation ID is not found
        """
        # End all FSMs in the stack
        if conversation_id in self.conversation_stacks:
            for frame in self.conversation_stacks[conversation_id]:
                try:
                    self.fsm_manager.end_conversation(frame.conversation_id)
                except Exception as e:
                    logger.warning(f"Error ending FSM {frame.conversation_id}: {str(e)}")
        else:
            # Fallback for non-stacked conversations
            self.fsm_manager.end_conversation(conversation_id)

        # Remove from active conversations
        if conversation_id in self.active_conversations:
            del self.active_conversations[conversation_id]

    @handle_conversation_errors("Failed to delete conversation")
    def delete_conversation(self, conversation_id: str) -> None:
        """
        Completely delete a conversation and remove all associated data from all FSMs.

        Unlike end_conversation which just marks a conversation as ended,
        this method completely removes all data related to the conversation
        from memory.

        Args:
            conversation_id: The conversation ID

        Raises:
            ValueError: If the conversation ID is not found
        """
        # First end the conversation (which validates the ID exists)
        self.end_conversation(conversation_id)

        # Remove all FSM instances from the stack
        if conversation_id in self.conversation_stacks:
            for frame in self.conversation_stacks[conversation_id]:
                if frame.conversation_id in self.fsm_manager.instances:
                    del self.fsm_manager.instances[frame.conversation_id]

            # Remove the stack itself
            del self.conversation_stacks[conversation_id]
        else:
            # Fallback for non-stacked conversations
            if conversation_id in self.fsm_manager.instances:
                del self.fsm_manager.instances[conversation_id]

        logger.info(f"Conversation {conversation_id} and all stacked FSMs completely deleted")

    @handle_conversation_errors
    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, str]]:
        """
        Get the conversation history for the current active FSM.

        Args:
            conversation_id: The conversation ID

        Returns:
            A list of conversation exchanges (user and system messages)

        Raises:
            ValueError: If the conversation ID is not found
        """
        current_fsm_id = self._get_current_fsm_conversation_id(conversation_id)
        complete_data = self.fsm_manager.get_complete_conversation(current_fsm_id)
        return complete_data.get("conversation_history", [])

    @handle_conversation_errors("Failed to save conversation")
    def save_conversation(self, conversation_id: str, path: str) -> None:
        """
        Save a conversation's state to a file for later resumption.

        Args:
            conversation_id: The conversation ID
            path: The file path to save to

        Raises:
            ValueError: If the conversation ID is not found
            IOError: If the file cannot be written
        """
        try:
            # Get complete conversation data including stack information
            save_data = {
                "main_conversation_id": conversation_id,
                "stack_depth": self.get_stack_depth(conversation_id),
                "all_stack_data": self.get_all_stack_data(conversation_id),
                "context_flow": self.get_context_flow(conversation_id)
            }

            # Save to file
            with open(path, 'w') as f:
                json.dump(save_data, f, indent=2)

            logger.info(f"Conversation {conversation_id} saved to {path}")

        except Exception as e:
            logger.error(f"Error saving conversation: {str(e)}")
            raise IOError(f"Failed to save conversation: {str(e)}")

    def list_active_conversations(self) -> List[str]:
        """
        List all active conversation IDs.

        Returns:
            A list of active conversation IDs
        """
        return list(self.active_conversations.keys())

    def get_llm_interface(self) -> LLMInterface:
        """
        Get the currently used LLM interface.

        Returns:
            The LLM interface instance being used by this API instance
        """
        return self.llm_interface

    def execute_handlers_manually(self,
                                  timing: HandlerTiming,
                                  conversation_id: str,
                                  target_state: Optional[str] = None,
                                  updated_keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Manually execute handlers for testing or custom workflows.

        Args:
            timing: The timing point to execute handlers for
            conversation_id: The conversation ID to get context from
            target_state: Optional target state for transition handlers
            updated_keys: Optional list of keys being updated

        Returns:
            Updated context after handler execution

        Raises:
            ValueError: If conversation ID is not found
        """
        if conversation_id not in self.active_conversations:
            raise ValueError(f"Conversation not found: {conversation_id}")

        current_fsm_id = self._get_current_fsm_conversation_id(conversation_id)
        current_state = self.fsm_manager.get_conversation_state(current_fsm_id)
        context = self.fsm_manager.get_conversation_data(current_fsm_id)

        updated_keys_set = set(updated_keys) if updated_keys else None

        return self.handler_system.execute_handlers(
            timing=timing,
            current_state=current_state,
            target_state=target_state,
            context=context,
            updated_keys=updated_keys_set
        )

    def close(self) -> None:
        """
        Clean up all active conversations and resources.

        This method can be called explicitly to clean up when done
        using the API instance.
        """
        for conversation_id in list(self.active_conversations.keys()):
            try:
                self.end_conversation(conversation_id)
            except Exception as e:
                logger.warning(f"Error ending conversation {conversation_id} during cleanup: {str(e)}")

    def _get_current_fsm_conversation_id(self, conversation_id: str) -> str:
        """
        Get the conversation ID of the currently active FSM (top of stack).

        Args:
            conversation_id: The main conversation ID

        Returns:
            The conversation ID of the current active FSM
        """
        if conversation_id not in self.conversation_stacks:
            return conversation_id  # Fallback to original behavior

        stack = self.conversation_stacks[conversation_id]
        return stack[-1].conversation_id if stack else conversation_id

    def _merge_context_with_strategy(self,
                                     conversation_id: str,
                                     context_to_merge: Dict[str, Any],
                                     strategy: ContextMergeStrategy = ContextMergeStrategy.UPDATE) -> None:
        """
        Merge context using the specified strategy.

        Fixed to properly implement SELECTIVE strategy.

        Args:
            conversation_id: The conversation ID to update
            context_to_merge: Context to merge
            strategy: Merge strategy enum
        """
        if not context_to_merge:
            return

        try:
            current_context = self.fsm_manager.get_conversation_data(conversation_id)
        except Exception:
            current_context = {}

        # Fallback to UPDATE strategy
        if strategy is None:
            strategy = ContextMergeStrategy.UPDATE

        if strategy == ContextMergeStrategy.UPDATE:
            # Update existing context with new values
            merged_context = {**current_context, **context_to_merge}
        elif strategy == ContextMergeStrategy.PRESERVE:
            # Only add new keys, don't overwrite existing
            # FIXED: Correct logic - new keys from child that don't exist in parent
            merged_context = current_context.copy()
            for key, value in context_to_merge.items():
                if key not in current_context:
                    merged_context[key] = value
        elif strategy == ContextMergeStrategy.SELECTIVE:
            # FIXED: Properly implement selective merge
            # Only merge keys that are in the shared_context_keys of the frame
            merged_context = current_context.copy()

            # Get the current frame from the stack to access shared_context_keys
            if conversation_id in self.conversation_stacks:
                stack = self.conversation_stacks[conversation_id]
                if stack:
                    current_frame = stack[-1]
                    shared_keys = current_frame.shared_context_keys

                    # Only merge keys that are in the shared list
                    for key in shared_keys:
                        if key in context_to_merge:
                            merged_context[key] = context_to_merge[key]
            else:
                # Fallback to UPDATE if we can't find the frame
                merged_context = {**current_context, **context_to_merge}
        else:
            raise ValueError(f"Unknown merge strategy {strategy}")

        # Update the conversation context
        self.fsm_manager.update_conversation_context(conversation_id, merged_context)

    def _generate_resume_message(self,
                                 previous_frame: FSMStackFrame,
                                 merged_context: Dict[str, Any]) -> str:
        """
        Generate a message for when resuming a previous FSM.

        Args:
            previous_frame: The FSM frame being resumed
            merged_context: Context that was merged back

        Returns:
            A resume message
        """
        if merged_context:
            context_summary = ", ".join([f"{k}={v}" for k, v in list(merged_context.items())[:3]])
            if len(merged_context) > 3:
                context_summary += f"... (+{len(merged_context) - 3} more)"
            return f"Resumed previous conversation with updated context: {context_summary}"
        else:
            return "Resumed previous conversation."

    def __enter__(self):
        """Context manager entry point."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit - clean up all active conversations.

        This ensures all conversations are properly ended when using
        the API class in a 'with' statement.
        """
        self.close()

# --------------------------------------------------------------