from __future__ import annotations

"""
Enhanced API Module for FSM-LLM: Stateful Conversational AI

This module implements the main API interface for the FSM-LLM library, providing developers
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

    from fsm_llm import API

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
                         .at(HandlerTiming.POST_TRANSITION)
                         .on_state("address_confirmation")
                         .do(validate_address_function))

    api.register_handler(validation_handler)
"""

import hashlib
import json
import os
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .constants import DEFAULT_LLM_MODEL, DEFAULT_MAX_STACK_DEPTH, FSM_ID_HASH_LENGTH
from .definitions import FSMDefinition, FSMError

# --------------------------------------------------------------
# local imports
# --------------------------------------------------------------
from .fsm import FSMManager
from .handlers import (
    FSMHandler,
    HandlerBuilder,
    HandlerSystem,
    HandlerTiming,
    create_handler,
)
from .llm import LiteLLMInterface, LLMInterface
from .logging import handle_conversation_errors, logger
from .prompts import (
    DataExtractionPromptBuilder,
    FieldExtractionPromptBuilder,
    ResponseGenerationPromptBuilder,
)
from .transition_evaluator import TransitionEvaluator, TransitionEvaluatorConfig

# --------------------------------------------------------------


class FSMStackFrame(BaseModel):
    """Represents a single FSM in the conversation stack."""

    fsm_definition: FSMDefinition | dict[str, Any] | str
    conversation_id: str
    return_context: dict[str, Any] = Field(default_factory=dict)
    shared_context_keys: list[str] = Field(default_factory=list)
    preserve_history: bool = False

    model_config = {"arbitrary_types_allowed": True}


# --------------------------------------------------------------


class ContextMergeStrategy(str, Enum):
    """Context merge strategies for FSM stack operations."""

    UPDATE = "update"
    PRESERVE = "preserve"

    @classmethod
    def from_string(cls, value: str | ContextMergeStrategy) -> ContextMergeStrategy:
        """Convert string or enum to ContextMergeStrategy."""
        if isinstance(value, cls):
            return value

        if isinstance(value, str):
            try:
                return cls(value.lower().strip())
            except ValueError as e:
                valid_values = [e_member.value for e_member in cls]
                raise ValueError(
                    f"Invalid merge strategy '{value}'. Must be one of: {valid_values}"
                ) from e

        raise ValueError(f"Invalid type for merge strategy: {type(value)}")


# --------------------------------------------------------------
# Main API Class
# --------------------------------------------------------------


class API:
    """
    Enhanced API for Improved 2-Pass FSM-LLM Architecture.

    This class provides a backward-compatible interface while internally
    implementing the improved 2-pass architecture for better conversation quality
    and response generation after transition evaluation.
    """

    def __init__(
        self,
        fsm_definition: FSMDefinition | dict[str, Any] | str,
        llm_interface: LLMInterface | None = None,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_history_size: int = 5,
        max_message_length: int = 1000,
        handlers: list[FSMHandler] | None = None,
        handler_error_mode: str = "continue",
        transition_config: TransitionEvaluatorConfig | None = None,
        **llm_kwargs,
    ):
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
                raise ValueError("llm_interface must be an instance of LLMInterface")
            self.llm_interface = llm_interface
            logger.info(
                f"API initialized with custom LLM interface: {type(llm_interface).__name__}"
            )
        else:
            # Create default interface
            model = model or os.environ.get("LLM_MODEL", DEFAULT_LLM_MODEL)
            temperature = temperature if temperature is not None else 0.5
            max_tokens = max_tokens if max_tokens is not None else 1000

            self.llm_interface = LiteLLMInterface(
                model=model,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                **llm_kwargs,
            )
            logger.info(
                f"API initialized with default LiteLLM interface, model={model}"
            )

        # Process FSM definition
        self.fsm_definition, self.fsm_id = self.process_fsm_definition(fsm_definition)

        # Create enhanced prompt builders
        data_extraction_prompt_builder = DataExtractionPromptBuilder()
        response_generation_prompt_builder = ResponseGenerationPromptBuilder()
        field_extraction_prompt_builder = FieldExtractionPromptBuilder()

        # Create transition evaluator
        evaluator_config = transition_config or TransitionEvaluatorConfig()
        transition_evaluator = TransitionEvaluator(evaluator_config)

        # FSM stacking support (initialized before FSMManager so closure can access it)
        self._temp_fsm_definitions: dict[str, FSMDefinition] = {}

        # Create custom FSM loader
        def custom_fsm_loader(fsm_id: str) -> FSMDefinition:
            if fsm_id == self.fsm_id:
                return self.fsm_definition
            elif fsm_id in self._temp_fsm_definitions:
                return self._temp_fsm_definitions[fsm_id]
            else:
                from .utilities import load_fsm_definition

                return load_fsm_definition(fsm_id)

        # Initialize handler system (single instance shared with FSMManager)
        self.handler_system = HandlerSystem(error_mode=handler_error_mode)

        # Initialize enhanced FSM manager with improved 2-pass architecture
        self.fsm_manager = FSMManager(
            fsm_loader=custom_fsm_loader,
            llm_interface=self.llm_interface,
            data_extraction_prompt_builder=data_extraction_prompt_builder,
            response_generation_prompt_builder=response_generation_prompt_builder,
            field_extraction_prompt_builder=field_extraction_prompt_builder,
            transition_evaluator=transition_evaluator,
            max_history_size=max_history_size,
            max_message_length=max_message_length,
            handler_system=self.handler_system,
        )

        # Register provided handlers
        if handlers:
            for handler in handlers:
                self.register_handler(handler)

        # FSM stacking support
        self._stack_lock = threading.Lock()
        self.active_conversations: dict[str, bool] = {}
        self.conversation_stacks: dict[str, list[FSMStackFrame]] = {}
        self._last_accessed: dict[str, float] = {}

        logger.info("Enhanced API fully initialized with improved 2-pass architecture")

    @classmethod
    def process_fsm_definition(
        cls, fsm_definition: FSMDefinition | dict[str, Any] | str
    ) -> tuple[FSMDefinition, str]:
        """Process FSM definition input and return standardized format."""
        if isinstance(fsm_definition, FSMDefinition):
            fsm_def = fsm_definition
            content_hash = hashlib.sha256(
                json.dumps(fsm_def.model_dump(), sort_keys=True).encode()
            ).hexdigest()[:FSM_ID_HASH_LENGTH]
            fsm_id = f"fsm_def_{fsm_def.name}_{content_hash}"

        elif isinstance(fsm_definition, dict):
            try:
                fsm_def = FSMDefinition(**fsm_definition)
                content_hash = hashlib.sha256(
                    json.dumps(fsm_definition, sort_keys=True).encode()
                ).hexdigest()[:FSM_ID_HASH_LENGTH]
                fsm_id = f"fsm_dict_{fsm_def.name}_{content_hash}"
            except Exception as e:
                raise ValueError(f"Invalid FSM definition dictionary: {e!s}") from e

        elif isinstance(fsm_definition, str):
            try:
                from .utilities import load_fsm_from_file

                fsm_def = load_fsm_from_file(fsm_definition)
                fsm_id = f"fsm_file_{fsm_definition}"
            except Exception as e:
                raise ValueError(
                    f"Failed to load FSM from file '{fsm_definition}': {e!s}"
                ) from e

        else:
            raise ValueError(
                f"Invalid FSM definition type: {type(fsm_definition)}. "
                f"Must be FSMDefinition, dict, or str"
            )

        return fsm_def, fsm_id

    @classmethod
    def from_file(cls, path: Path | str, **kwargs: Any) -> API:
        """Create API instance from FSM definition file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"FSM definition file not found: {path}")
        return cls(fsm_definition=str(path), **kwargs)

    @classmethod
    def from_definition(
        cls,
        fsm_definition: FSMDefinition | dict[str, Any] | None = None,
        **kwargs,
    ) -> API:
        """Create API instance from FSM definition object or dictionary.

        Accepts ``fsm_definition`` positionally or as keyword.  The alias
        ``definition`` is also accepted for convenience.
        """
        if fsm_definition is None:
            fsm_definition = kwargs.pop("definition", None)
        if fsm_definition is None:
            raise TypeError("from_definition() requires an fsm_definition argument")
        return cls(fsm_definition=fsm_definition, **kwargs)

    def start_conversation(
        self, initial_context: dict[str, Any] | None = None
    ) -> tuple[str, str]:
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
                self.fsm_id, initial_context=initial_context
            )

            # Track conversation and initialize stack
            with self._stack_lock:
                self.active_conversations[conversation_id] = True
                self.conversation_stacks[conversation_id] = [
                    FSMStackFrame(
                        fsm_definition=self.fsm_definition,
                        conversation_id=conversation_id,
                    )
                ]
                self._last_accessed[conversation_id] = time.monotonic()

            return conversation_id, response

        except FSMError:
            raise
        except Exception as e:
            logger.error(f"Error starting conversation: {e!s}")
            raise FSMError(f"Failed to start conversation: {e!s}") from e

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
            with self._stack_lock:
                self._last_accessed[conversation_id] = time.monotonic()
            response: str = self.fsm_manager.process_message(
                current_fsm_id, user_message
            )
            return response
        except (ValueError, FSMError):
            raise
        except Exception as e:
            logger.error(f"Error processing message: {e!s}")
            raise FSMError(f"Failed to process message: {e!s}") from e

    # ==========================================
    # FSM STACKING METHODS (Enhanced)
    # ==========================================

    def push_fsm(
        self,
        conversation_id: str,
        new_fsm_definition: FSMDefinition | dict[str, Any] | str,
        context_to_pass: dict[str, Any] | None = None,
        return_context: dict[str, Any] | None = None,
        shared_context_keys: list[str] | None = None,
        preserve_history: bool = False,
        inherit_context: bool = True,
    ) -> str:
        """Push new FSM onto conversation stack with enhanced context management."""
        with self._stack_lock:
            if conversation_id not in self.active_conversations:
                raise FSMError(f"Conversation not found: {conversation_id}")
            self._validate_stack_depth(conversation_id)

        processed_fsm_id = None
        new_conversation_id = None
        push_succeeded = False
        try:
            processed_fsm_def, processed_fsm_id = self.process_fsm_definition(
                new_fsm_definition
            )
            with self._stack_lock:
                self._temp_fsm_definitions[processed_fsm_id] = processed_fsm_def

            current_fsm_id = self._get_current_fsm_conversation_id(conversation_id)
            initial_context = self._build_push_context(
                current_fsm_id, context_to_pass, preserve_history, inherit_context
            )

            new_conversation_id, response = self.fsm_manager.start_conversation(
                processed_fsm_id, initial_context=initial_context
            )
            with self._stack_lock:
                self._temp_fsm_definitions.pop(processed_fsm_id, None)
                # Re-validate depth atomically with frame addition
                self._validate_stack_depth(conversation_id)

                new_frame = FSMStackFrame(
                    fsm_definition=processed_fsm_def,
                    conversation_id=new_conversation_id,
                    return_context=return_context or {},
                    shared_context_keys=shared_context_keys or [],
                    preserve_history=preserve_history,
                )
                self.conversation_stacks[conversation_id].append(new_frame)
                push_succeeded = True

            with self._stack_lock:
                stack_depth = len(self.conversation_stacks.get(conversation_id, []))
            logger.info(
                f"Pushed new FSM onto conversation {conversation_id}, "
                f"stack depth: {stack_depth}"
            )
            return response

        except (FSMError, ValueError):
            raise
        except Exception as e:
            logger.error(f"Error pushing FSM: {e!s}")
            raise FSMError(f"Failed to push FSM: {e!s}") from e
        finally:
            if not push_succeeded:
                try:
                    self._rollback_push(processed_fsm_id, new_conversation_id)
                except Exception as rollback_err:
                    logger.error(
                        f"Rollback after failed push_fsm also failed: {rollback_err}"
                    )

    def _validate_stack_depth(self, conversation_id: str) -> None:
        """Raise FSMError if FSM stack depth limit is reached."""
        current_depth = len(self.conversation_stacks.get(conversation_id, []))
        if current_depth >= DEFAULT_MAX_STACK_DEPTH:
            raise FSMError(
                f"FSM stack depth limit ({DEFAULT_MAX_STACK_DEPTH}) reached for "
                f"conversation {conversation_id}. Cannot push more FSMs."
            )

    def _build_push_context(
        self,
        current_fsm_id: str,
        context_to_pass: dict[str, Any] | None,
        preserve_history: bool,
        inherit_context: bool,
    ) -> dict[str, Any]:
        """Build initial context for pushed FSM from inheritance and passed context."""
        initial_context: dict[str, Any] = {}

        if inherit_context:
            try:
                initial_context.update(
                    self.fsm_manager.get_conversation_data(current_fsm_id)
                )
            except KeyError as e:
                logger.warning(f"Could not inherit context (missing key): {e!s}")

        if context_to_pass:
            initial_context.update(context_to_pass)

        if preserve_history:
            try:
                history = self.fsm_manager.get_conversation_history(current_fsm_id)
                initial_context["_inherited_history"] = history
            except (FSMError, ValueError) as e:
                logger.warning(f"Could not preserve history: {e!s}")

        return initial_context

    def _rollback_push(
        self, processed_fsm_id: str | None, new_conversation_id: str | None
    ) -> None:
        """Clean up resources after a failed push_fsm attempt."""
        if processed_fsm_id:
            self._temp_fsm_definitions.pop(processed_fsm_id, None)
        if new_conversation_id:
            try:
                self.fsm_manager.end_conversation(new_conversation_id)
            except Exception as cleanup_err:
                logger.debug(
                    f"Failed to clean up orphaned conversation {new_conversation_id}: {cleanup_err}"
                )

    def pop_fsm(
        self,
        conversation_id: str,
        context_to_return: dict[str, Any] | None = None,
        merge_strategy: str | ContextMergeStrategy = ContextMergeStrategy.UPDATE,
    ) -> str:
        """Pop current FSM from stack and return to previous with enhanced context handling."""
        # Hold _stack_lock for the entire pop operation to prevent concurrent
        # push/pop from corrupting the stack.
        with self._stack_lock:
            if conversation_id not in self.active_conversations:
                raise FSMError(f"Conversation not found: {conversation_id}")
            stack = self.conversation_stacks[conversation_id]
            if len(stack) <= 1:
                raise FSMError("Cannot pop from FSM stack: only one FSM remaining")
            current_frame = stack[-1]
            previous_frame = stack[-2]

            try:
                merge_strategy_enum = ContextMergeStrategy.from_string(merge_strategy)

                current_fsm_context = self._get_frame_context(current_frame)
                context_to_merge = self._collect_pop_context(
                    current_frame, current_fsm_context, context_to_return
                )

                if context_to_merge:
                    self._merge_context_with_strategy(
                        previous_frame.conversation_id,
                        context_to_merge,
                        merge_strategy_enum,
                    )

                if current_frame.preserve_history:
                    self._preserve_sub_conversation_summary(
                        current_frame, previous_frame, current_fsm_context
                    )

                try:
                    self.fsm_manager.end_conversation(current_frame.conversation_id)
                finally:
                    stack.pop()

                response = self._generate_resume_message(
                    previous_frame, context_to_merge
                )
                stack_depth = len(self.conversation_stacks.get(conversation_id, []))
                logger.info(
                    f"Popped FSM from conversation {conversation_id}, "
                    f"stack depth: {stack_depth}"
                )
                return response

            except (FSMError, ValueError):
                raise
            except Exception as e:
                logger.error(f"Error popping FSM: {e!s}")
                raise FSMError(f"Failed to pop FSM: {e!s}") from e

    def _get_frame_context(self, frame: FSMStackFrame) -> dict[str, Any]:
        """Get conversation data for a stack frame.

        Raises:
            FSMError: If context retrieval fails (orphaned conversation, FSM corruption).
        """
        try:
            result: dict[str, Any] = self.fsm_manager.get_conversation_data(
                frame.conversation_id
            )
            return result
        except (FSMError, ValueError) as e:
            raise FSMError(
                f"Context retrieval failed for frame {frame.conversation_id}: {e!s}. "
                f"Sub-FSM context would be lost during pop."
            ) from e

    def _collect_pop_context(
        self,
        frame: FSMStackFrame,
        fsm_context: dict[str, Any],
        context_to_return: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Collect context to merge back when popping an FSM."""
        result: dict[str, Any] = {}
        if frame.return_context:
            result.update(frame.return_context)
        if context_to_return:
            result.update(context_to_return)
        missing_keys = []
        for key in frame.shared_context_keys or []:
            if key in fsm_context:
                result[key] = fsm_context[key]
            else:
                missing_keys.append(key)
        if missing_keys:
            logger.warning(
                f"Shared context keys not found in sub-FSM: {missing_keys}. "
                f"These will not be merged back to parent FSM."
            )
        return result

    def _preserve_sub_conversation_summary(
        self,
        current_frame: FSMStackFrame,
        previous_frame: FSMStackFrame,
        current_fsm_context: dict[str, Any],
    ) -> None:
        """Preserve sub-conversation summary in parent frame context."""
        try:
            current_history = self.fsm_manager.get_conversation_history(
                current_frame.conversation_id
            )
            summary_context = {
                "_sub_conversation_summary": {
                    "fsm_type": str(current_frame.fsm_definition),
                    "final_context": current_fsm_context,
                    "exchange_count": len(current_history),
                }
            }
            self._merge_context_with_strategy(
                previous_frame.conversation_id,
                summary_context,
                ContextMergeStrategy.UPDATE,
            )
        except (FSMError, ValueError) as e:
            logger.warning(f"Could not preserve sub-conversation summary: {e!s}")

    def _merge_context_with_strategy(
        self,
        conversation_id: str,
        context_to_merge: dict[str, Any],
        strategy: ContextMergeStrategy = ContextMergeStrategy.UPDATE,
    ) -> None:
        """Merge context using specified strategy."""
        if not context_to_merge:
            return

        try:
            current_context = self.fsm_manager.get_conversation_data(conversation_id)
        except (FSMError, ValueError) as ctx_err:
            logger.warning(
                f"Could not retrieve context for {conversation_id}: {ctx_err}"
            )
            current_context = {}

        if strategy == ContextMergeStrategy.UPDATE:
            merged_context = {**current_context, **context_to_merge}
        elif strategy == ContextMergeStrategy.PRESERVE:
            merged_context = current_context.copy()
            for key, value in context_to_merge.items():
                if key not in current_context:
                    merged_context[key] = value
        else:
            raise ValueError(f"Unknown merge strategy {strategy}")

        # Only pass changed keys to avoid triggering handlers for unchanged data
        diff = {
            k: v
            for k, v in merged_context.items()
            if k not in current_context or current_context[k] != v
        }
        if diff:
            self.fsm_manager.update_conversation_context(conversation_id, diff)

    def _generate_resume_message(
        self, previous_frame: FSMStackFrame, merged_context: dict[str, Any]
    ) -> str:
        """Generate message for resuming previous FSM."""
        if merged_context:
            context_keys = list(merged_context.keys())[:3]
            context_summary = ", ".join(context_keys)
            if len(merged_context) > 3:
                context_summary += f"... (+{len(merged_context) - 3} more)"
            return f"Resumed previous conversation. Updated fields: {context_summary}"
        else:
            return "Resumed previous conversation."

    def _get_current_fsm_conversation_id(self, conversation_id: str) -> str:
        """Get conversation ID of current active FSM (top of stack)."""
        with self._stack_lock:
            if conversation_id not in self.conversation_stacks:
                raise ValueError(
                    f"Unknown conversation ID: {conversation_id}. "
                    f"Call start_conversation() first or check list_active_conversations()."
                )

            stack = self.conversation_stacks[conversation_id]
            if not stack:
                raise ValueError(
                    f"Conversation stack is empty for {conversation_id}. "
                    f"The conversation may have been corrupted."
                )
            return stack[-1].conversation_id

    # ==========================================
    # CONTEXT AND STACK MANAGEMENT METHODS
    # ==========================================

    def update_context(
        self, conversation_id: str, context_update: dict[str, Any]
    ) -> None:
        """
        Update context data for the current FSM in conversation.

        Args:
            conversation_id: Root conversation ID
            context_update: Dictionary of context keys to update
        """
        if not isinstance(context_update, dict):
            raise TypeError("context_update must be a dictionary")
        if not context_update:
            return
        current_fsm_id = self._get_current_fsm_conversation_id(conversation_id)
        self.fsm_manager.update_conversation_context(current_fsm_id, context_update)

    def get_stack_depth(self, conversation_id: str) -> int:
        """
        Get the current FSM stack depth for a conversation.

        Args:
            conversation_id: Root conversation ID

        Returns:
            Number of FSMs on the stack (1 = base FSM only)
        """
        with self._stack_lock:
            if conversation_id not in self.conversation_stacks:
                raise ValueError(
                    f"Unknown conversation ID: {conversation_id}. "
                    f"Call start_conversation() first."
                )
            return len(self.conversation_stacks[conversation_id])

    def get_sub_conversation_id(self, conversation_id: str) -> str:
        """
        Get the internal conversation ID of the current (top-of-stack) sub-FSM.

        Useful for extensions that need to track sub-FSM identity
        across push/pop operations.

        Args:
            conversation_id: Root conversation ID

        Returns:
            The internal conversation ID of the active sub-FSM
        """
        return self._get_current_fsm_conversation_id(conversation_id)

    # ==========================================
    # HANDLER MANAGEMENT METHODS
    # ==========================================

    def register_handler(self, handler: FSMHandler) -> None:
        """Register handler with the system."""
        self.handler_system.register_handler(handler)

    def register_handlers(self, handlers: list[FSMHandler]) -> None:
        """Register multiple handlers."""
        for handler in handlers:
            self.register_handler(handler)

    def create_handler(
        self,
        name: str = "CustomHandler",
        timing: HandlerTiming | None = None,
        action: Any | None = None,
    ) -> HandlerBuilder:
        """Create new handler using fluent builder.

        When *timing* and *action* are both provided the handler is built
        and registered automatically, providing a convenient shorthand::

            fsm.create_handler(
                name="on_start",
                timing=HandlerTiming.START_CONVERSATION,
                action=lambda ctx: print("started"),
            )
        """
        builder = create_handler(name)
        if timing is not None:
            builder = builder.at(timing)
        if action is not None:
            builder = builder.do(action)
        if timing is not None and action is not None:
            self.register_handler(builder)
        return builder

    # ==========================================
    # CONVERSATION MANAGEMENT METHODS
    # ==========================================

    @handle_conversation_errors
    def get_data(self, conversation_id: str) -> dict[str, Any]:
        """Get collected data from current FSM."""
        current_fsm_id = self._get_current_fsm_conversation_id(conversation_id)
        data: dict[str, Any] = self.fsm_manager.get_conversation_data(current_fsm_id)
        return data

    @handle_conversation_errors
    def has_conversation_ended(self, conversation_id: str) -> bool:
        """Check if current FSM has ended."""
        current_fsm_id = self._get_current_fsm_conversation_id(conversation_id)
        ended: bool = self.fsm_manager.has_conversation_ended(current_fsm_id)
        return ended

    @handle_conversation_errors
    def get_current_state(self, conversation_id: str) -> str:
        """Get current state of active FSM."""
        current_fsm_id = self._get_current_fsm_conversation_id(conversation_id)
        state: str = self.fsm_manager.get_conversation_state(current_fsm_id)
        return state

    @handle_conversation_errors
    def get_conversation_history(self, conversation_id: str) -> list[dict[str, str]]:
        """Get conversation history for current FSM."""
        current_fsm_id = self._get_current_fsm_conversation_id(conversation_id)
        history: list[dict[str, str]] = self.fsm_manager.get_conversation_history(
            current_fsm_id
        )
        return history

    @handle_conversation_errors("Failed to end conversation")
    def end_conversation(self, conversation_id: str) -> None:
        """End conversation and clean up all FSMs in stack."""
        with self._stack_lock:
            stack = self.conversation_stacks.pop(conversation_id, None)
            self.active_conversations.pop(conversation_id, None)
            self._last_accessed.pop(conversation_id, None)

        if stack:
            for frame in reversed(stack):
                try:
                    self.fsm_manager.end_conversation(frame.conversation_id)
                except Exception as e:
                    logger.warning(f"Error ending FSM {frame.conversation_id}: {e!s}")
        else:
            self.fsm_manager.end_conversation(conversation_id)

        # No need to clear _temp_fsm_definitions — entries are removed after caching in push_fsm

    def list_active_conversations(self) -> list[str]:
        """List all active conversation IDs."""
        with self._stack_lock:
            return list(self.active_conversations.keys())

    def cleanup_stale_conversations(
        self, max_idle_seconds: float = 3600.0
    ) -> list[str]:
        """End conversations that have been idle longer than max_idle_seconds.

        This method should be called periodically by the application to prevent
        indefinite memory accumulation from abandoned conversations.

        Args:
            max_idle_seconds: Maximum idle time before a conversation is cleaned up.
                Defaults to 3600 (1 hour).

        Returns:
            List of conversation IDs that were cleaned up.
        """
        now = time.monotonic()
        stale_ids: list[str] = []
        with self._stack_lock:
            for conv_id, last_access in self._last_accessed.items():
                if now - last_access > max_idle_seconds:
                    stale_ids.append(conv_id)

        cleaned: list[str] = []
        for conv_id in stale_ids:
            try:
                self.end_conversation(conv_id)
                cleaned.append(conv_id)
            except Exception as e:
                logger.warning(
                    f"Failed to clean up stale conversation {conv_id}: {e!s}"
                )

        if cleaned:
            logger.info(f"Cleaned up {len(cleaned)} stale conversations")
        return cleaned

    def get_llm_interface(self) -> LLMInterface:
        """Get current LLM interface."""
        return self.llm_interface

    def close(self) -> None:
        """Clean up all active conversations."""
        for conversation_id in list(self.active_conversations.keys()):
            try:
                self.end_conversation(conversation_id)
            except Exception as e:
                logger.warning(
                    f"Error ending conversation {conversation_id} during cleanup: {e!s}"
                )

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.close()


# --------------------------------------------------------------
