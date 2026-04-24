from __future__ import annotations

"""
FSMManager: Conversation lifecycle orchestrator for the 2-pass architecture.

Manages conversation instances, thread-safe locking, and FSM definition caching.
Delegates all message processing to MessagePipeline.

Thread-safe per-conversation. See docs/architecture.md for details.
"""

import threading
import time
import traceback
import uuid
from collections import OrderedDict
from collections.abc import Callable, Iterator
from datetime import datetime
from typing import Any

from .constants import (
    DEFAULT_MAX_HISTORY_SIZE,
    DEFAULT_MAX_MESSAGE_LENGTH,
    INTERNAL_KEY_PREFIXES,
)
from .definitions import (
    FSMContext,
    FSMDefinition,
    FSMError,
    FSMInstance,
    State,
)
from .handlers import HandlerSystem, HandlerTiming
from .lam.ast import Term
from .lam.fsm_compile import compile_fsm

# --------------------------------------------------------------
# Local imports
# --------------------------------------------------------------
from .llm import LLMInterface
from .logging import logger, with_conversation_context
from .pipeline import MessagePipeline
from .prompts import (
    DataExtractionPromptBuilder,
    FieldExtractionPromptBuilder,
    ResponseGenerationPromptBuilder,
)
from .transition_evaluator import TransitionEvaluator
from .utilities import load_fsm_definition

# --------------------------------------------------------------
# FSM Manager
# --------------------------------------------------------------


class FSMManager:
    """Conversation lifecycle orchestrator.

    Manages instances, thread-safe locking, FSM definition caching, and
    (as of M2 S7) compiled λ-term caching for the lambda substrate.
    Delegates 2-pass message processing to :class:`MessagePipeline`.

    Two parallel LRU caches share the same ``max_fsm_cache_size`` bound
    and ``self._lock``:

    - ``self.fsm_cache`` — ``fsm_id → FSMDefinition``, lazy-populated by
      :meth:`get_fsm_definition` on first request.
    - ``self._compiled_terms`` — ``fsm_id → lam.ast.Term``, lazy-populated
      by :meth:`get_compiled_term` on first request. Used by the S8+
      compiled pipeline path; not exercised by the legacy 2-pass
      :class:`MessagePipeline`.

    The caches may drift (one populated, the other not) and are
    independent — eviction of one does not evict the other. See D-S7-01
    through D-S7-03 for design rationale.
    """

    def __init__(
        self,
        fsm_loader: Callable[[str], FSMDefinition] = load_fsm_definition,
        llm_interface: LLMInterface | None = None,
        data_extraction_prompt_builder: DataExtractionPromptBuilder | None = None,
        response_generation_prompt_builder: ResponseGenerationPromptBuilder
        | None = None,
        field_extraction_prompt_builder: FieldExtractionPromptBuilder | None = None,
        transition_evaluator: TransitionEvaluator | None = None,
        max_history_size: int = DEFAULT_MAX_HISTORY_SIZE,
        max_message_length: int = DEFAULT_MAX_MESSAGE_LENGTH,
        handler_system: HandlerSystem | None = None,
        handler_error_mode: str = "continue",
        max_fsm_cache_size: int = 64,
    ):
        if llm_interface is None:
            raise ValueError("llm_interface is required and cannot be None")

        self.fsm_loader = fsm_loader
        self.llm_interface = llm_interface

        # Lock for thread-safe access to shared class-level dicts
        self._lock = threading.Lock()
        # Per-conversation locks to prevent concurrent mutations on the same instance
        self._conversation_locks: dict[str, threading.RLock] = {}

        # Cache and instance management
        self.fsm_cache: OrderedDict[str, FSMDefinition] = OrderedDict()
        self._max_fsm_cache_size = max_fsm_cache_size
        self.instances: dict[str, FSMInstance] = {}
        # DECISION D-S7-01 — compile cache lives on the manager, not the
        # FSMInstance. Compiled terms are pure derivatives of the
        # immutable FSMDefinition; multiple instances with the same
        # fsm_id share the same term, so per-instance caching would
        # duplicate memory for no behavioral gain. LRU mirrors
        # self.fsm_cache (D-S7-02); lock discipline releases
        # self._lock before get_fsm_definition to avoid reentrant
        # deadlock (D-S7-03). See plans/plan_2026-04-24_91a46fbe/decisions.md.
        self._compiled_terms: OrderedDict[str, Term] = OrderedDict()

        # Configuration
        self.max_history_size = max_history_size
        self.max_message_length = max_message_length

        # Handler system
        self.handler_system = handler_system or HandlerSystem(
            error_mode=handler_error_mode
        )

        # Prompt builders (stored for attribute access by tests/API)
        self.data_extraction_prompt_builder = (
            data_extraction_prompt_builder or DataExtractionPromptBuilder()
        )
        self.response_generation_prompt_builder = (
            response_generation_prompt_builder or ResponseGenerationPromptBuilder()
        )
        self.field_extraction_prompt_builder = (
            field_extraction_prompt_builder or FieldExtractionPromptBuilder()
        )

        # Transition evaluator
        self.transition_evaluator = transition_evaluator or TransitionEvaluator()

        # Message processing pipeline
        self._pipeline = MessagePipeline(
            llm_interface=self.llm_interface,
            data_extraction_prompt_builder=self.data_extraction_prompt_builder,
            response_generation_prompt_builder=self.response_generation_prompt_builder,
            transition_evaluator=self.transition_evaluator,
            handler_system=self.handler_system,
            fsm_resolver=self.get_fsm_definition,
            field_extraction_prompt_builder=self.field_extraction_prompt_builder,
            # S8b: plug the S7 compiled-term LRU cache into the pipeline
            # so `process_compiled` avoids recompiling per turn.
            compiled_term_resolver=self.get_compiled_term,
        )

        logger.info(
            f"Enhanced FSM Manager initialized with improved 2-pass architecture - "
            f"history_size={max_history_size}, message_length={max_message_length}"
        )

    # ----------------------------------------------------------
    # Handler registration
    # ----------------------------------------------------------

    def register_handler(self, handler):
        """Register a handler with the system."""
        self.handler_system.register_handler(handler)

    # ----------------------------------------------------------
    # FSM definition cache
    # ----------------------------------------------------------

    def get_fsm_definition(self, fsm_id: str) -> FSMDefinition:
        """Get FSM definition with caching and LRU eviction."""
        with self._lock:
            if fsm_id not in self.fsm_cache:
                logger.info(f"Loading FSM definition: {fsm_id}")
                if len(self.fsm_cache) >= self._max_fsm_cache_size:
                    evicted_key, _ = self.fsm_cache.popitem(last=False)
                    logger.debug(f"Evicted FSM definition from cache: {evicted_key}")
                self.fsm_cache[fsm_id] = self.fsm_loader(fsm_id)
            else:
                self.fsm_cache.move_to_end(fsm_id)
            return self.fsm_cache[fsm_id]

    def get_compiled_term(self, fsm_id: str) -> Term:
        """Get compiled λ-term for ``fsm_id`` with LRU caching.

        Mirrors :meth:`get_fsm_definition`'s cache policy on
        ``self._compiled_terms``. Calls :func:`compile_fsm` at most once
        per ``fsm_id`` while the term stays resident. Thread-safe; see
        D-S7-03 for lock discipline.

        Not invoked by any production code path as of S7 — reserved for
        the S8 :class:`MessagePipeline` integration.
        """
        # First check under the lock (fast path for cache hits).
        with self._lock:
            if fsm_id in self._compiled_terms:
                self._compiled_terms.move_to_end(fsm_id)
                return self._compiled_terms[fsm_id]

        # Cold miss: resolve the definition OUTSIDE the lock — otherwise
        # nested acquisition of self._lock (a non-reentrant Lock)
        # deadlocks against get_fsm_definition (D-S7-03).
        defn = self.get_fsm_definition(fsm_id)
        compiled = compile_fsm(defn)

        # Re-acquire to commit; another thread may have won the race.
        with self._lock:
            if fsm_id in self._compiled_terms:
                # Loser path: discard our work, reuse the winner.
                self._compiled_terms.move_to_end(fsm_id)
                return self._compiled_terms[fsm_id]
            if len(self._compiled_terms) >= self._max_fsm_cache_size:
                evicted_key, _ = self._compiled_terms.popitem(last=False)
                logger.debug(f"Evicted compiled term from cache: {evicted_key}")
            self._compiled_terms[fsm_id] = compiled
            return compiled

    # ----------------------------------------------------------
    # Instance lifecycle
    # ----------------------------------------------------------

    def _create_instance(self, fsm_id: str) -> FSMInstance:
        """Create new FSM instance."""
        fsm_def = self.get_fsm_definition(fsm_id)
        logger.info(
            f"Creating FSM instance for {fsm_id}, initial state: {fsm_def.initial_state}"
        )

        context = FSMContext(
            max_history_size=self.max_history_size,
            max_message_length=self.max_message_length,
        )

        return FSMInstance(
            fsm_id=fsm_id,
            current_state=fsm_def.initial_state,
            persona=fsm_def.persona,
            context=context,
        )

    def get_current_state(
        self, instance: FSMInstance, conversation_id: str | None = None
    ) -> State:
        """Get current state definition for an instance."""
        return self._pipeline.get_state(instance, conversation_id)

    def start_conversation(
        self, fsm_id: str, initial_context: dict[str, Any] | None = None
    ) -> tuple[str, str]:
        """Start new conversation.

        Returns:
            Tuple of (conversation_id, initial_response)
        """
        instance = self._create_instance(fsm_id)
        conversation_id = str(uuid.uuid4())
        log = logger.bind(conversation_id=conversation_id, package="fsm_llm")

        if initial_context:
            if not isinstance(initial_context, dict):
                raise FSMError(
                    f"initial_context must be a dict, got {type(initial_context).__name__}"
                )
            instance.context.update(initial_context)
            log.info(f"Added initial context: {list(initial_context.keys())}")

        instance.context.data.update(
            {
                "_conversation_id": conversation_id,
                "_conversation_start": datetime.now().isoformat(),
                "_timestamp": time.time(),
                "_fsm_id": fsm_id,
            }
        )

        with self._lock:
            self.instances[conversation_id] = instance
            self._conversation_locks[conversation_id] = threading.RLock()

        # Execute start conversation handlers
        try:
            self._execute_handlers(
                HandlerTiming.START_CONVERSATION,
                conversation_id,
                current_state=None,
                target_state=instance.current_state,
            )
        except FSMError:
            self._cleanup_conversation_resources(conversation_id)
            raise
        except Exception as e:
            self._cleanup_conversation_resources(conversation_id)
            log.error(f"START_CONVERSATION handler failed: {e!s}")
            raise FSMError(f"Failed to start conversation: {e!s}") from e

        log.info(f"Started conversation [{conversation_id}] with FSM [{fsm_id}]")

        # Generate initial response
        try:
            response = self._pipeline.generate_initial_response(
                instance, conversation_id
            )
            return conversation_id, response

        except FSMError:
            try:
                self._execute_handlers(
                    HandlerTiming.END_CONVERSATION,
                    conversation_id,
                    current_state=instance.current_state,
                )
            except Exception as cleanup_err:
                log.warning(
                    f"END_CONVERSATION handler failed during cleanup: {cleanup_err}"
                )
            self._cleanup_conversation_resources(conversation_id)
            raise
        except Exception as e:
            try:
                self._execute_handlers(
                    HandlerTiming.END_CONVERSATION,
                    conversation_id,
                    current_state=instance.current_state,
                )
            except Exception as cleanup_err:
                log.warning(
                    f"END_CONVERSATION handler failed during cleanup: {cleanup_err}"
                )
            self._cleanup_conversation_resources(conversation_id)
            log.error(f"Error generating initial response: {e!s}")
            raise FSMError(f"Failed to start conversation: {e!s}") from e

    # ----------------------------------------------------------
    # Message processing
    # ----------------------------------------------------------

    @with_conversation_context
    def process_message(
        self, conversation_id: str, message: str, log: Any = None
    ) -> str:
        """Process user message with the 2-pass architecture.

        Pass 1: Data extraction + transition evaluation + transition execution
        Pass 2: Response generation based on final state
        """
        with self._lock:
            if conversation_id not in self.instances:
                raise FSMError(f"Conversation {conversation_id} not found")
            conv_lock = self._conversation_locks[conversation_id]
            if not conv_lock.acquire(blocking=False):
                raise FSMError(
                    f"Conversation {conversation_id} is already being processed by another thread"
                )
        try:
            return self._process_message_locked(conversation_id, message, log)
        finally:
            conv_lock.release()

    @with_conversation_context
    def process_message_stream(
        self, conversation_id: str, message: str, log: Any = None
    ) -> Iterator[str]:
        """Process user message, streaming the Pass 2 response.

        Pass 1 (extraction + transitions) runs fully.  Pass 2 yields
        response tokens as they arrive from the LLM.
        """
        with self._lock:
            if conversation_id not in self.instances:
                raise FSMError(f"Conversation {conversation_id} not found")
            conv_lock = self._conversation_locks[conversation_id]
            if not conv_lock.acquire(blocking=False):
                raise FSMError(
                    f"Conversation {conversation_id} is already being processed by another thread"
                )
        try:
            instance = self.instances[conversation_id]
            current_state = self.get_current_state(instance, conversation_id)
            if not current_state.transitions:
                raise FSMError(
                    f"Conversation has ended - current state '{instance.current_state}' is terminal"
                )
            instance.context.conversation.add_user_message(message)
            try:
                yield from self._pipeline.process_stream(
                    instance, message, conversation_id
                )
            except FSMError:
                self._rollback_user_message(instance, message, log)
                raise
            except Exception as e:
                self._rollback_user_message(instance, message, log)
                raise FSMError(f"Failed to process message: {e!s}") from e
        finally:
            conv_lock.release()

    def _process_message_locked(
        self, conversation_id: str, message: str, log: Any
    ) -> str:
        """Process message while holding the per-conversation lock."""
        instance = self.instances[conversation_id]
        log.info(f"Processing message in state: {instance.current_state}")

        current_state = self.get_current_state(instance, conversation_id)
        if not current_state.transitions:
            raise FSMError(
                f"Conversation has ended - current state '{instance.current_state}' is terminal"
            )

        instance.context.conversation.add_user_message(message)

        try:
            return self._pipeline.process(instance, message, conversation_id)

        except FSMError:
            self._rollback_user_message(instance, message, log)
            raise
        except Exception as e:
            log.error(f"Error processing message: {e!s}\n{traceback.format_exc()}")
            self._rollback_user_message(instance, message, log)

            try:
                self._pipeline.execute_handlers(
                    instance,
                    HandlerTiming.ERROR,
                    conversation_id,
                    current_state=instance.current_state,
                    error_context={
                        "_error": str(e),
                        "_traceback": traceback.format_exc(),
                    },
                )
            except Exception as handler_err:
                log.warning(
                    f"Error handler raised an exception, preserving original error: {handler_err}"
                )

            raise FSMError(f"Failed to process message: {e!s}") from e

    @staticmethod
    def _rollback_user_message(instance: FSMInstance, message: str, log: Any) -> None:
        """Remove last user message from history to avoid duplicates on retry.

        Called immediately after ``add_user_message`` in error paths, so the
        last exchange is always the one just added.  We verify it is a user
        message (not a system reply) before popping.
        """
        try:
            exchanges = instance.context.conversation.exchanges
            if exchanges and "user" in exchanges[-1]:
                exchanges.pop()
        except Exception as rollback_err:
            log.warning(
                f"Failed to rollback user message from conversation history: {rollback_err}"
            )

    # ----------------------------------------------------------
    # Conversation query methods
    # ----------------------------------------------------------

    @with_conversation_context
    def has_conversation_ended(self, conversation_id: str, log: Any = None) -> bool:
        """Check if conversation has reached terminal state."""
        if conversation_id not in self.instances:
            raise FSMError(f"Conversation {conversation_id} not found")

        instance = self.instances[conversation_id]
        current_state = self.get_current_state(instance, conversation_id)

        is_ended = not current_state.transitions
        if is_ended:
            log.info(f"Conversation reached terminal state: {instance.current_state}")

        return is_ended

    @with_conversation_context
    def get_conversation_data(
        self, conversation_id: str, log: Any = None
    ) -> dict[str, Any]:
        """Get collected context data (internal metadata keys filtered out)."""
        if conversation_id not in self.instances:
            raise FSMError(f"Conversation {conversation_id} not found")

        instance = self.instances[conversation_id]
        return {
            k: v
            for k, v in instance.context.data.items()
            if not any(k.startswith(p) for p in INTERNAL_KEY_PREFIXES)
        }

    @with_conversation_context
    def get_conversation_state(self, conversation_id: str, log: Any = None) -> str:
        """Get current state of conversation."""
        if conversation_id not in self.instances:
            raise FSMError(f"Conversation {conversation_id} not found")

        return self.instances[conversation_id].current_state

    @with_conversation_context
    def get_conversation_history(
        self, conversation_id: str, log: Any = None
    ) -> list[dict[str, str]]:
        """Get conversation history."""
        if conversation_id not in self.instances:
            raise FSMError(f"Conversation {conversation_id} not found")

        instance = self.instances[conversation_id]
        return instance.context.conversation.get_recent()

    @with_conversation_context
    def update_conversation_context(
        self, conversation_id: str, context_update: dict[str, Any], log: Any = None
    ) -> None:
        """Update conversation context data."""
        if not isinstance(context_update, dict):
            raise FSMError("context_update must be a dictionary")

        with self._lock:
            if conversation_id not in self.instances:
                raise FSMError(f"Conversation {conversation_id} not found")
            conv_lock = self._conversation_locks.get(conversation_id)
            instance = self.instances[conversation_id]

            # Acquire per-conversation lock while still holding _lock to prevent
            # the conversation from being cleaned up between lookup and acquire.
            if conv_lock is not None and not conv_lock.acquire(blocking=False):
                raise FSMError(
                    f"Conversation {conversation_id} is already being processed "
                    "by another thread"
                )
        try:
            if context_update:
                log.info(f"Updating context with keys: {list(context_update.keys())}")
                instance.context.update(context_update)

                self._execute_handlers(
                    HandlerTiming.CONTEXT_UPDATE,
                    conversation_id,
                    updated_keys=set(context_update.keys()),
                )
        finally:
            if conv_lock is not None:
                try:
                    conv_lock.release()
                except RuntimeError:
                    pass  # Lock was already released or cleaned up

    # ----------------------------------------------------------
    # Conversation lifecycle (end / cleanup)
    # ----------------------------------------------------------

    def _cleanup_conversation_resources(self, conversation_id: str) -> None:
        """Remove instance and per-conversation lock. Thread-safe."""
        with self._lock:
            self.instances.pop(conversation_id, None)
            self._conversation_locks.pop(conversation_id, None)

    @with_conversation_context
    def end_conversation(self, conversation_id: str, log: Any = None) -> None:
        """End conversation and clean up resources."""
        with self._lock:
            if conversation_id not in self.instances:
                raise FSMError(f"Conversation {conversation_id} not found")
            conv_lock = self._conversation_locks.get(conversation_id)

        # Acquire per-conversation lock to ensure no concurrent process_message.
        # Use timeout to prevent indefinite blocking if a thread is stuck.
        if conv_lock is not None:
            if not conv_lock.acquire(timeout=30):
                logger.warning(
                    f"Timed out waiting for conversation lock on {conversation_id}, "
                    "proceeding with cleanup"
                )
        try:
            self._execute_handlers(HandlerTiming.END_CONVERSATION, conversation_id)
        finally:
            # Clean up while holding conv_lock to prevent race with process_message.
            # Lock ordering is safe: process_message uses non-blocking conv_lock
            # acquisition, so _lock→conv_lock never creates circular wait.
            self._cleanup_conversation_resources(conversation_id)
            if conv_lock is not None:
                try:
                    conv_lock.release()
                except RuntimeError:
                    pass  # Lock was not acquired (timeout) or already released
        log.info(f"Conversation {conversation_id} ended")

    def cleanup_stale_conversations(self) -> list[str]:
        """Remove locks for conversations that no longer have active instances."""
        with self._lock:
            stale_ids = [
                cid for cid in self._conversation_locks if cid not in self.instances
            ]
            for cid in stale_ids:
                self._conversation_locks.pop(cid, None)
        if stale_ids:
            logger.info(f"Cleaned up {len(stale_ids)} stale conversation locks")
        return stale_ids

    @with_conversation_context
    def get_complete_conversation(
        self, conversation_id: str, log: Any = None
    ) -> dict[str, Any]:
        """Get complete conversation data for analysis."""
        if conversation_id not in self.instances:
            raise FSMError(f"Conversation {conversation_id} not found")

        instance = self.instances[conversation_id]
        current_state = self.get_current_state(instance, conversation_id)

        return {
            "id": conversation_id,
            "fsm_id": instance.fsm_id,
            "current_state": {
                "id": instance.current_state,
                "description": current_state.description,
                "purpose": current_state.purpose,
                "is_terminal": not current_state.transitions,
            },
            "collected_data": dict(instance.context.data),
            "conversation_history": instance.context.conversation.get_recent(),
            "metadata": dict(instance.context.metadata),
            "last_extraction_response": instance.last_extraction_response.model_dump()
            if instance.last_extraction_response
            else None,
            "last_transition_decision": instance.last_transition_decision.model_dump()
            if instance.last_transition_decision
            else None,
            "last_response_generation": instance.last_response_generation.model_dump()
            if instance.last_response_generation
            else None,
        }

    # ----------------------------------------------------------
    # Pipeline delegation
    # ----------------------------------------------------------

    def _execute_handlers(
        self,
        timing: HandlerTiming,
        conversation_id: str,
        current_state: str | None = None,
        target_state: str | None = None,
        updated_keys: set[str] | None = None,
        error_context: dict[str, Any] | None = None,
    ) -> None:
        """Execute handlers at specified timing point."""
        if conversation_id not in self.instances:
            return
        self._pipeline.execute_handlers(
            self.instances[conversation_id],
            timing,
            conversation_id,
            current_state=current_state,
            target_state=target_state,
            updated_keys=updated_keys,
            error_context=error_context,
        )
