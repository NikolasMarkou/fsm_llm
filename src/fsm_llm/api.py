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
from collections.abc import Iterator
from enum import Enum
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, Field

from .constants import DEFAULT_LLM_MODEL, DEFAULT_MAX_STACK_DEPTH, FSM_ID_HASH_LENGTH
from .definitions import FSMDefinition, FSMError

# --------------------------------------------------------------
# local imports
# --------------------------------------------------------------
from .fsm import FSMManager
from .handlers import (
    BaseHandler,
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
from .session import SessionState, SessionStore
from .transition_evaluator import TransitionEvaluator, TransitionEvaluatorConfig

# --------------------------------------------------------------


class FSMStackFrame(BaseModel):
    """Represents a single FSM in the conversation stack."""

    fsm_definition: FSMDefinition | dict[str, Any] | str
    conversation_id: str
    return_context: dict[str, Any] = Field(default_factory=dict)
    shared_context_keys: list[str] = Field(default_factory=list)
    preserve_history: bool = False
    # DECISION plan-2026-07-21T045419-9925aa3a/D-011
    # The content-hash id of this frame's FSM definition. OPTIONAL with a safe
    # default so serialized frames / restore_session predating this field still
    # load. `_release_unreferenced_temp_definitions` counts refs across all live
    # frames by this id: a frame that leaves it None UNDER-COUNTS and can let
    # cleanup evict a def another frame still needs. If you add a THIRD
    # FSMStackFrame construction site, you MUST set fsm_id there too. See D-011.
    fsm_id: str | None = None

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
        session_store: SessionStore | None = None,
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
        # DECISION plan-2026-07-21T045419-9925aa3a/D-011
        # In-flight-push guard: ids registered in `_temp_fsm_definitions` but not
        # yet referenced by a live FSMStackFrame. `push_fsm` releases `_stack_lock`
        # between registering the temp def and appending the frame (to run
        # `start_conversation`, which may make a greeting LLM call). Without this
        # set, a concurrent `pop_fsm`/`end_conversation` on ANOTHER conversation
        # calls `_release_unreferenced_temp_definitions` in that window, sees the
        # id as unreferenced (no frame yet), and evicts it — bricking the in-flight
        # sub-conversation with `ValueError: Unknown FSM ID`. Cleanup treats
        # pending ids as referenced. See D-011.
        self._pending_push_ids: set[str] = set()

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
        self._ended_conversations: dict[str, dict[str, Any]] = {}
        self._MAX_ENDED_CACHE: int = 10_000

        # Session persistence
        self._session_store = session_store

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
        self,
        initial_context: dict[str, Any] | None = None,
        *,
        _suppress_start: bool = False,
    ) -> tuple[str, str]:
        """
        Start new conversation with improved 2-pass architecture.

        Args:
            initial_context: Optional initial context data
            _suppress_start: Internal resume flag forwarded to
                ``FSMManager.start_conversation(suppress_start=...)``. When True
                the START_CONVERSATION handlers and the Pass-2 greeting are
                skipped (used by ``restore_session``). Not part of the public
                API; default False preserves normal start behavior.

        Returns:
            Tuple of (conversation_id, initial_response)
        """
        try:
            # Start conversation using enhanced FSM manager
            conversation_id, response = self.fsm_manager.start_conversation(
                self.fsm_id,
                initial_context=initial_context,
                suppress_start=_suppress_start,
            )

            # Track conversation and initialize stack
            with self._stack_lock:
                self.active_conversations[conversation_id] = True
                self.conversation_stacks[conversation_id] = [
                    FSMStackFrame(
                        fsm_definition=self.fsm_definition,
                        conversation_id=conversation_id,
                        fsm_id=self.fsm_id,
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
            # D-014: _get_current_fsm_conversation_id already refreshed _last_accessed.
            current_fsm_id = self._get_current_fsm_conversation_id(conversation_id)
            response: str = self.fsm_manager.process_message(
                current_fsm_id, user_message
            )
            # Auto-save session if store is configured
            if self._session_store is not None:
                try:
                    self.save_session(conversation_id)
                except Exception as e:
                    logger.warning(f"Auto-save session failed: {e!s}")
            return response
        except (ValueError, FSMError):
            raise
        except Exception as e:
            logger.error(f"Error processing message: {e!s}")
            raise FSMError(f"Failed to process message: {e!s}") from e

    def converse_stream(self, user_message: str, conversation_id: str) -> Iterator[str]:
        """Process message, streaming the response tokens.

        Pass 1 (extraction + transitions) runs fully.  Pass 2 yields
        response tokens as they arrive from the LLM.

        This is a thin NON-generator wrapper: it resolves the current FSM
        conversation id at CALL time — which validates existence AND refreshes
        ``_last_accessed`` — then returns a lazy nested-closure generator that
        streams the reply.  Resolving eagerly means a created-but-not-yet-iterated
        stream cannot skip validation or be reaped as stale mid-flight by
        ``cleanup_stale_conversations``.

        Args:
            user_message: User's message.
            conversation_id: Existing conversation ID.

        Yields:
            String chunks of the response as they arrive.
        """
        try:
            # D-014: _get_current_fsm_conversation_id already refreshed _last_accessed.
            # Runs at CALL time (this is a plain function, not a generator).
            current_fsm_id = self._get_current_fsm_conversation_id(conversation_id)
        except (ValueError, FSMError):
            raise
        except Exception as e:
            logger.error(f"Error streaming message: {e!s}")
            raise FSMError(f"Failed to stream message: {e!s}") from e

        # DECISION plan-2026-07-21T082818-4c63deac/D-002
        # conv_lock acquisition stays LAZY inside this nested-closure generator
        # (via ``process_message_stream``'s own inner generator), so an abandoned
        # (never-iterated) stream cannot leak a lock and the ``_lock -> conv_lock``
        # order is preserved.  The generator is a NESTED CLOSURE (capturing
        # ``self``/args from scope) and deliberately NOT a ``self._..._inner``
        # method: a bound-method self-dispatch resolves to a Mock under
        # ``Mock(spec=API)`` unbound-self tests and regressed the agents
        # auto-save suite (D-003/CF1).  Do NOT hoist conv_lock into the eager
        # prologue and do NOT reintroduce a ``self.``-attribute inner generator.
        def _stream() -> Iterator[str]:
            try:
                try:
                    yield from self.fsm_manager.process_message_stream(
                        current_fsm_id, user_message
                    )
                finally:
                    # Auto-save session after stream completes or is abandoned
                    if self._session_store is not None:
                        try:
                            self.save_session(conversation_id)
                        except Exception as save_err:
                            logger.warning(f"Auto-save session failed: {save_err!s}")
            except (ValueError, FSMError):
                raise
            except Exception as e:
                logger.error(f"Error streaming message: {e!s}")
                raise FSMError(f"Failed to stream message: {e!s}") from e

        return _stream()

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
                # DECISION plan-2026-07-21T045419-9925aa3a/D-011
                # Mark this id in-flight BEFORE releasing the lock for
                # start_conversation. Cleared atomically with frame append below;
                # this closes the register→append window where a concurrent
                # cleanup would otherwise evict the still-unreferenced def.
                self._pending_push_ids.add(processed_fsm_id)

            current_fsm_id = self._get_current_fsm_conversation_id(conversation_id)
            initial_context = self._build_push_context(
                current_fsm_id, context_to_pass, preserve_history, inherit_context
            )

            new_conversation_id, response = self.fsm_manager.start_conversation(
                processed_fsm_id, initial_context=initial_context
            )
            with self._stack_lock:
                # DECISION plan-2026-07-21T045419-9925aa3a/D-011
                # Do NOT re-add `_temp_fsm_definitions.pop(processed_fsm_id, None)`
                # here. The pushed def used to survive only in the LRU `fsm_cache`;
                # once ~65 distinct FSM ids loaded it was evicted and the loader
                # fell through to `load_fsm_definition`, which raises
                # `ValueError: Unknown FSM ID` for a content-hash id — bricking the
                # sub-conversation. The def must stay registered for the frame's
                # LIFETIME; `_release_unreferenced_temp_definitions` (called on
                # pop_fsm / end_conversation / _rollback_push) drops it once the
                # last referencing frame is gone. See D-011.
                # Re-validate depth atomically with frame addition
                self._validate_stack_depth(conversation_id)

                new_frame = FSMStackFrame(
                    fsm_definition=processed_fsm_def,
                    conversation_id=new_conversation_id,
                    return_context=return_context or {},
                    shared_context_keys=shared_context_keys or [],
                    preserve_history=preserve_history,
                    fsm_id=processed_fsm_id,
                )
                self.conversation_stacks[conversation_id].append(new_frame)
                # Frame now references the def; clear the in-flight guard
                # atomically with the append (same _stack_lock hold). See D-011.
                self._pending_push_ids.discard(processed_fsm_id)
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
        # DECISION plan-2026-07-21T045419-9925aa3a/D-011
        # Do NOT unconditionally pop `processed_fsm_id` here. A failed push can
        # share its content-hash id with a live frame in ANOTHER conversation
        # (two conversations pushing the same sub-FSM); an unconditional pop
        # would evict a def that conversation still needs. Release by reference
        # instead — the helper drops the id iff no live frame references it.
        if new_conversation_id:
            try:
                self.fsm_manager.end_conversation(new_conversation_id)
            except Exception as cleanup_err:
                logger.debug(
                    f"Failed to clean up orphaned conversation {new_conversation_id}: {cleanup_err}"
                )
        # A failed push never appended a frame, so clear its in-flight guard so
        # the id can be released below (otherwise it leaks as permanently
        # "pending" and its temp def is never freed). See D-011.
        if processed_fsm_id is not None:
            with self._stack_lock:
                self._pending_push_ids.discard(processed_fsm_id)
        self._release_unreferenced_temp_definitions()

    def _release_unreferenced_temp_definitions(self) -> None:
        """Drop temp FSM definitions no live stack frame still references.

        DECISION plan-2026-07-21T045419-9925aa3a/D-011
        Reference-aware cleanup: a content-hash FSM id is SHARED across
        conversations on one API instance, so a single temp def may back
        several live frames. Compute the set of ``fsm_id``s used by every frame
        in every live stack under ``_stack_lock``, then drop only temp entries
        absent from that set. Do NOT drop a temp entry merely because one
        referencing conversation ended — that reintroduces
        ``ValueError: Unknown FSM ID`` for the conversations still using it.

        Contract: takes no args; acquires ``_stack_lock`` internally (callers
        must NOT already hold it); mutates ``_temp_fsm_definitions`` in place;
        returns None; never raises.
        """
        with self._stack_lock:
            # DECISION plan-2026-07-21T045419-9925aa3a/D-011
            # Treat in-flight pushes (registered temp def, frame not yet
            # appended) as referenced. Without `| self._pending_push_ids` a
            # cleanup racing a concurrent push evicts a def that push is about
            # to reference — the exact register→append window H1 must survive.
            used = {
                frame.fsm_id
                for stack in self.conversation_stacks.values()
                for frame in stack
                if frame.fsm_id is not None
            } | self._pending_push_ids
            stale = [
                fsm_id for fsm_id in self._temp_fsm_definitions if fsm_id not in used
            ]
            for fsm_id in stale:
                self._temp_fsm_definitions.pop(fsm_id, None)

    def pop_fsm(
        self,
        conversation_id: str,
        context_to_return: dict[str, Any] | None = None,
        merge_strategy: str | ContextMergeStrategy = ContextMergeStrategy.UPDATE,
    ) -> str:
        """Pop current FSM from stack and return to previous with enhanced context handling."""
        # DECISION plan_2026-05-29_d9092060/D-001
        # Narrow lock scope: snapshot frame references under _stack_lock, then release
        # the lock before calling fsm_manager methods (which acquire per-conversation
        # RLocks and can block). Re-acquire _stack_lock only to pop the stack entry.
        # Do NOT revert to a single wide `with self._stack_lock:` covering the whole
        # method — that blocks list_active_conversations/push_fsm on other conversations
        # for the full duration of FSM teardown I/O.
        with self._stack_lock:
            if conversation_id not in self.active_conversations:
                raise FSMError(f"Conversation not found: {conversation_id}")
            stack = self.conversation_stacks[conversation_id]
            if len(stack) <= 1:
                raise FSMError("Cannot pop from FSM stack: only one FSM remaining")
            # D-014: pop_fsm reads conversation_stacks directly instead of going
            # through _get_current_fsm_conversation_id, so it needs its own refresh.
            self._last_accessed[conversation_id] = time.monotonic()
            # Snapshot frame references — do NOT call fsm_manager inside this lock
            current_frame = stack[-1]
            previous_frame = stack[-2]
            merge_strategy_enum = ContextMergeStrategy.from_string(merge_strategy)

        # FSM manager operations outside the lock (they acquire per-conversation RLocks)
        try:
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
                # Re-acquire lock only to mutate the stack
                # DECISION plan-2026-07-18T051819-80b0bd4d/D-013
                # Remove the frame we just ended BY OBJECT IDENTITY, at whatever
                # index it now holds. Do NOT restore the old
                # `inner_stack[-1].conversation_id == current_frame.conversation_id`
                # top-of-stack test: end_conversation runs with _stack_lock RELEASED
                # (D-001 above, deliberately), so a concurrent push_fsm on the same
                # root id can append a new top frame in that window. The positional
                # test then fails and the already-ended frame is stranded in the
                # stack forever — permanently bricking the conversation, because
                # _get_current_fsm_conversation_id keeps returning a conversation
                # FSMManager has already torn down (converse and pop_fsm then fail
                # on every subsequent call).
                # Use `is`, NOT `==` / list.remove(): FSMStackFrame is a pydantic
                # model with VALUE equality, so two structurally-identical frames
                # are indistinguishable by `==` and the wrong one could be deleted.
                # Absent frame (double pop / concurrent end_conversation) is a no-op,
                # never an IndexError.
                # Do NOT "simplify" this by widening _stack_lock over
                # end_conversation — that stalls unrelated conversations for the
                # duration of FSM teardown I/O, which is exactly what D-001 rejected.
                with self._stack_lock:
                    inner_stack = self.conversation_stacks.get(conversation_id) or []
                    for idx in range(len(inner_stack) - 1, -1, -1):
                        if inner_stack[idx] is current_frame:
                            del inner_stack[idx]
                            break
                    else:
                        logger.debug(
                            f"pop_fsm: frame {current_frame.conversation_id} already "
                            f"absent from stack {conversation_id}; nothing to remove"
                        )

            # The popped frame is gone; release any temp def it was the last
            # frame to reference (D-011). Reference-aware, so a sub-FSM shared
            # with another live conversation survives this pop.
            self._release_unreferenced_temp_definitions()

            response = self._generate_resume_message(previous_frame, context_to_merge)
            with self._stack_lock:
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
        # Explicitly-requested shared keys are honored regardless of prefix:
        # fall back to the sub-FSM's raw context.data for internal-prefixed
        # keys that get_conversation_data() filters out.
        raw_context: dict[str, Any] = {}
        with self.fsm_manager._lock:
            instance = self.fsm_manager.instances.get(frame.conversation_id)
            if instance is not None:
                raw_context = dict(instance.context.data)
        missing_keys = []
        for key in frame.shared_context_keys or []:
            if key in fsm_context:
                result[key] = fsm_context[key]
            elif key in raw_context:
                result[key] = raw_context[key]
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
            # DECISION plan-2026-07-18T051819-80b0bd4d/D-014
            # THIS is the single idleness-refresh point. Every public method that
            # resolves "which FSM is current" funnels through here (converse,
            # converse_stream, get_data, update_context, get_current_state,
            # get_conversation_history, get_sub_conversation_id,
            # has_conversation_ended, push_fsm, save_session, end_conversation), so
            # one write under the ALREADY-HELD lock covers all of them.
            # Do NOT re-add explicit `self._last_accessed[...] = ...` lines in
            # converse/converse_stream: they were deleted precisely because they
            # duplicated this write one line later, and two writers of the same
            # eviction bookkeeping is how cleanup_stale_conversations came to
            # force-end conversations that were driven purely through
            # push_fsm/get_data/update_context (which never called converse).
            # Do NOT extract this into a _touch() helper that acquires the lock:
            # _stack_lock is a plain non-reentrant threading.Lock, so a helper
            # called from here would self-deadlock.
            # Must use time.monotonic() — cleanup_stale_conversations reads the
            # same clock; mixing in time.time() silently corrupts the arithmetic.
            self._last_accessed[conversation_id] = time.monotonic()
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
            # D-014: second method that bypasses _get_current_fsm_conversation_id.
            self._last_accessed[conversation_id] = time.monotonic()
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
        # mypy: `do()` returns a built BaseHandler (not the builder), so the local
        # is annotated as the union. register_handler receives a BaseHandler at
        # runtime (only reachable after `.do()` ran); cast narrows for the Protocol.
        # The declared `-> HandlerBuilder` return is pre-existing and unchanged;
        # cast preserves the exact runtime object returned. Annotation-only.
        builder: HandlerBuilder | BaseHandler = create_handler(name)
        if timing is not None:
            builder = cast(HandlerBuilder, builder).at(timing)
        if action is not None:
            builder = cast(HandlerBuilder, builder).do(action)
        if timing is not None and action is not None:
            self.register_handler(cast(FSMHandler, builder))
        return cast(HandlerBuilder, builder)

    # ==========================================
    # CONVERSATION MANAGEMENT METHODS
    # ==========================================

    @handle_conversation_errors
    def get_data(self, conversation_id: str) -> dict[str, Any]:
        """Get collected data from current FSM."""
        try:
            current_fsm_id = self._get_current_fsm_conversation_id(conversation_id)
            data: dict[str, Any] = self.fsm_manager.get_conversation_data(
                current_fsm_id
            )
            return data
        except (ValueError, KeyError):
            # Conversation ended — return cached data if available
            cached = self._ended_conversations.get(conversation_id)
            if cached:
                return cast(dict[str, Any], cached.get("data", {}))
            raise

    @handle_conversation_errors
    def has_conversation_ended(self, conversation_id: str) -> bool:
        """Check if current FSM has ended."""
        try:
            current_fsm_id = self._get_current_fsm_conversation_id(conversation_id)
            ended: bool = self.fsm_manager.has_conversation_ended(current_fsm_id)
            return ended
        except (ValueError, KeyError):
            # Conversation ended — check cache
            return conversation_id in self._ended_conversations

    @handle_conversation_errors
    def get_current_state(self, conversation_id: str) -> str:
        """Get current state of active FSM."""
        try:
            current_fsm_id = self._get_current_fsm_conversation_id(conversation_id)
            state: str = self.fsm_manager.get_conversation_state(current_fsm_id)
            return state
        except (ValueError, KeyError):
            cached = self._ended_conversations.get(conversation_id)
            if cached:
                return cast(str, cached.get("state", "unknown"))
            raise

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
        # Cache data and state before cleanup so get_data/get_current_state
        # still work after the conversation ends.
        try:
            current_fsm_id = self._get_current_fsm_conversation_id(conversation_id)
            self._ended_conversations[conversation_id] = {
                "data": self.fsm_manager.get_conversation_data(current_fsm_id),
                "state": self.fsm_manager.get_conversation_state(current_fsm_id),
                "history": self.fsm_manager.get_conversation_history(current_fsm_id),
            }
            if len(self._ended_conversations) > self._MAX_ENDED_CACHE:
                self._ended_conversations.pop(next(iter(self._ended_conversations)))
        except Exception:
            pass  # best-effort cache

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

        # D-011: pushed sub-FSM defs now live in _temp_fsm_definitions for the
        # frame's lifetime (the post-push pop was removed). This conversation's
        # frames are already out of conversation_stacks, so release any temp def
        # no OTHER live conversation still references.
        self._release_unreferenced_temp_definitions()

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
                # TOCTOU: conversation may have been ended by another thread between the
                # stale-ID collection above and this call. The FSMError is caught below.
                self.end_conversation(conv_id)
                cleaned.append(conv_id)
            except Exception as e:
                logger.warning(
                    f"Failed to clean up stale conversation {conv_id}: {e!s}"
                )

        if cleaned:
            logger.info(f"Cleaned up {len(cleaned)} stale conversations")
        return cleaned

    # ==========================================
    # SESSION PERSISTENCE METHODS
    # ==========================================

    def save_session(self, conversation_id: str) -> None:
        """Save conversation state to the session store.

        Args:
            conversation_id: Root conversation ID to save.

        Raises:
            FSMError: If no session store is configured or save fails.
        """
        if self._session_store is None:
            raise FSMError("No session store configured")

        current_fsm_id = self._get_current_fsm_conversation_id(conversation_id)
        state = SessionState(
            conversation_id=conversation_id,
            fsm_id=self.fsm_id,
            current_state=self.fsm_manager.get_conversation_state(current_fsm_id),
            context_data=self.fsm_manager.get_conversation_data(current_fsm_id),
            conversation_history=self.fsm_manager.get_conversation_history(
                current_fsm_id
            ),
            stack_depth=self.get_stack_depth(conversation_id),
        )

        # H10: the flat context_data does not carry WorkingMemory, so persist it
        # separately. Read the live instance's WorkingMemory reference under
        # _lock (consistent with the _replay_history reach-in); to_dict() is
        # itself internally lock-guarded. hidden_buffers are carried explicitly so
        # a custom hidden-buffer set survives the round-trip (D-032 downgrade).
        with self.fsm_manager._lock:
            instance = self.fsm_manager.instances.get(current_fsm_id)
            wm_obj = instance.context.working_memory if instance is not None else None
        if wm_obj is not None and hasattr(wm_obj, "to_dict"):
            state.working_memory = {
                "buffers": wm_obj.to_dict(),
                "hidden_buffers": sorted(
                    getattr(wm_obj, "_hidden_buffers", frozenset())
                ),
            }

        self._session_store.save(conversation_id, state)

    def load_session(self, session_id: str) -> SessionState | None:
        """Load a previously saved session state.

        Note: this returns the saved state for inspection. To fully
        restore a conversation, use ``restore_session()``.

        Args:
            session_id: Session identifier to load.

        Returns:
            Session state if found, None otherwise.

        Raises:
            FSMError: If no session store is configured.
        """
        if self._session_store is None:
            raise FSMError("No session store configured")
        return self._session_store.load(session_id)

    def restore_session(self, session_id: str) -> tuple[str, SessionState] | None:
        """Restore a conversation from a saved session.

        Starts a new conversation pre-populated with the saved context
        and conversation history. Note: the persisted JSON round-trip is
        lossy for non-JSON-native context values (datetime/set/custom
        objects are restored as strings, not their original type).

        Args:
            session_id: Session identifier to restore.

        Returns:
            Tuple of (conversation_id, session_state) if found, None if
            no saved session exists.

        Raises:
            FSMError: If no session store is configured or restore fails.
        """
        if self._session_store is None:
            raise FSMError("No session store configured")

        state = self._session_store.load(session_id)
        if state is None:
            return None

        # Start conversation with saved context. H9: _suppress_start=True skips
        # the START_CONVERSATION handlers and the Pass-2 greeting so a resume
        # does not re-fire start-of-conversation side effects.
        conv_id, _ = self.start_conversation(
            initial_context=state.context_data, _suppress_start=True
        )

        # Restore conversation history (and H10: WorkingMemory) under the
        # per-conversation lock.
        # NOTE (C-NEW-007): restore_session acquires conv_lock then
        # _replay_history briefly takes _lock (conv_lock → _lock), the reverse
        # of the codebase's canonical _lock → conv_lock order. This is a LATENT
        # ordering inversion only — restore_session operates on a freshly
        # created conversation id whose conv_lock no other thread can yet hold,
        # so a circular wait is not currently reachable. Do NOT change the
        # _replay_history signature/guard without updating its regression test.
        current_fsm_id = self._get_current_fsm_conversation_id(conv_id)
        conv_lock = self.fsm_manager._conversation_locks.get(current_fsm_id)

        def _replay_and_restore_wm() -> None:
            self._replay_history(current_fsm_id, state.conversation_history)
            if state.working_memory:
                # Single definition (not duplicated across the if/else arms):
                # duplicating the hidden_buffers carry risks fixing one arm and
                # not the other, re-opening the D-032 hidden-buffer downgrade.
                from .memory import WorkingMemory

                with self.fsm_manager._lock:
                    wm_instance = self.fsm_manager.instances.get(current_fsm_id)
                if wm_instance is not None:
                    wm = state.working_memory
                    wm_instance.context.working_memory = WorkingMemory.from_dict(
                        wm.get("buffers") or {},
                        hidden_buffers=frozenset(wm.get("hidden_buffers") or []),
                    )

        # DECISION plan-2026-07-21T072826-e3131cc2/D-003: restore_session must
        # NOT leak a half-registered conversation on partial-setup failure. Once
        # start_conversation(_suppress_start=True) has registered conv_id in
        # active_conversations / conversation_stacks / fsm_manager.instances,
        # ANY exception in history replay, WorkingMemory restore, or the
        # set_conversation_state validation below (which RAISES FSMError for a
        # corrupted/foreign/redeployed FSM whose saved current_state no longer
        # exists) leaves a fully-initialized conversation loaded with someone
        # else's history — invisible to the caller (conv_id is never returned)
        # and reclaimed only after the idle timeout. Do NOT drop this try/except
        # "to simplify"; it mirrors push_fsm's _rollback_push teardown-on-partial-
        # setup-failure (prior plan H1/D-011). The nested try/except around the
        # teardown is load-bearing (Pre-Mortem 3): end_conversation may itself
        # raise on a half-initialized conversation, and teardown must NEVER mask
        # the original error — swallow-and-log, then re-raise the original.
        # NOTE (intentional): end_conversation fires END_CONVERSATION handlers on
        # this teardown even though START was suppressed. This is DELIBERATE and
        # consistent with fsm.py:_cleanup_after_failed_start (prior plan D-006):
        # any failed conversation setup fires END on teardown. Do NOT suppress END
        # handlers here — that would diverge from the failed-start precedent.
        try:
            if conv_lock is not None:
                with conv_lock:
                    _replay_and_restore_wm()
            else:
                _replay_and_restore_wm()

            # C3: reinstate the saved current_state AFTER the conv_lock block.
            # set_conversation_state takes _lock then conv_lock, so calling it
            # OUTSIDE the conv_lock block here preserves the canonical
            # _lock → conv_lock order and does not weaken C-NEW-007. It also
            # validates the state against the FSM def, raising FSMError for a
            # corrupted/foreign session.
            self.fsm_manager.set_conversation_state(current_fsm_id, state.current_state)
        except Exception:
            # Best-effort teardown of the just-created conversation. end_conversation
            # removes conv_id from active_conversations / conversation_stacks and
            # ends the FSM instance (fsm_manager.instances). Swallow any teardown
            # error so it never masks the original failure.
            try:
                self.end_conversation(conv_id)
            except Exception as teardown_error:
                logger.warning(
                    "restore_session teardown of half-restored conversation "
                    f"'{conv_id}' failed: {teardown_error!s}"
                )
            raise

        return conv_id, state

    def _replay_history(self, fsm_id: str, history: list[dict[str, str]]) -> None:
        """Replay saved conversation history into an FSM instance."""
        with self.fsm_manager._lock:
            if fsm_id not in self.fsm_manager.instances:
                logger.warning(
                    f"Cannot replay history: FSM instance '{fsm_id}' not found"
                )
                return
            instance = self.fsm_manager.instances[fsm_id]
        for exchange in history:
            if "user" in exchange:
                instance.context.conversation.add_user_message(exchange["user"])
            if "system" in exchange:
                instance.context.conversation.add_system_message(exchange["system"])

    def get_llm_interface(self) -> LLMInterface:
        """Get current LLM interface."""
        return self.llm_interface

    def close(self) -> None:
        """Clean up all active conversations and release resources."""
        self.handler_system.close()
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
