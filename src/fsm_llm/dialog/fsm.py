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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..runtime.oracle import LiteLLMOracle

from ..constants import (
    DEFAULT_MAX_HISTORY_SIZE,
    DEFAULT_MAX_MESSAGE_LENGTH,
    INTERNAL_KEY_PREFIXES,
)
from ..handlers import HandlerSystem, HandlerTiming
from ..logging import logger, with_conversation_context

# --------------------------------------------------------------
# Local imports
# --------------------------------------------------------------
from ..runtime._litellm import LLMInterface
from ..runtime.ast import Term
from .._models import FSMError
from ..utilities import load_fsm_definition
from .compile_fsm import compile_fsm_cached
from .definitions import (
    FSMContext,
    FSMDefinition,
    FSMInstance,
    State,
)
from .prompts import (
    DataExtractionPromptBuilder,
    FieldExtractionPromptBuilder,
    ResponseGenerationPromptBuilder,
)
from .transition_evaluator import TransitionEvaluator
from .turn import MessagePipeline

# --------------------------------------------------------------
# FSM Manager
# --------------------------------------------------------------


class FSMManager:
    """Conversation lifecycle orchestrator.

    Manages instances, thread-safe locking, FSM definition caching, and
    (as of M2 S7) compiled λ-term caching for the lambda substrate.
    Delegates 2-pass message processing to :class:`MessagePipeline`.

    A single per-manager FSM-definition cache lives here:

    - ``self.fsm_cache`` — ``fsm_id → FSMDefinition``, lazy-populated by
      :meth:`get_fsm_definition` on first request.

    The compiled-λ-term cache lives in the kernel as of R2 (plan v3
    step 8): :func:`fsm_llm.lam.compile_fsm_cached` is the canonical
    front-door, backed by ``@lru_cache(maxsize=64)`` on
    ``_compile_fsm_by_id(fsm_id, fsm_json)``. :meth:`get_compiled_term`
    is now a 3-line shim — see ``# DECISION D-002`` in
    ``lam/fsm_compile.py`` and plan v3 D-PLAN-07 for rationale.

    The previous per-manager ``_compiled_terms`` OrderedDict and its
    ~30 LOC of bookkeeping (D-S7-01..D-S7-03) were removed in R2. The
    ``self._lock`` and ``self._conversation_locks`` remain — they
    protect the ``fsm_cache`` and per-conversation instance state,
    not the compiled-term cache.
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
        oracle: LiteLLMOracle | None = None,
    ):
        if llm_interface is None:
            raise ValueError("llm_interface is required and cannot be None")

        self.fsm_loader = fsm_loader
        self.llm_interface = llm_interface
        # M4 (merge spec §3 I2) — single Oracle threaded from Program → API
        # → here → MessagePipeline. When None (back-compat for direct
        # FSMManager construction without oracle=), wrap llm_interface.
        if oracle is None:
            from ..runtime.oracle import LiteLLMOracle

            oracle = LiteLLMOracle(llm_interface)
        self._oracle = oracle
        # DECISION D-S11-00 — post-S11 the compiled λ-term is the only
        # message-processing path. The prior routing opt-out (D-S9-01,
        # D-S10-00) has been removed along with the legacy pipeline methods.

        # Lock for thread-safe access to shared class-level dicts
        self._lock = threading.Lock()
        # Per-conversation locks to prevent concurrent mutations on the same instance
        self._conversation_locks: dict[str, threading.RLock] = {}

        # Cache and instance management
        self.fsm_cache: OrderedDict[str, FSMDefinition] = OrderedDict()
        self._max_fsm_cache_size = max_fsm_cache_size
        self.instances: dict[str, FSMInstance] = {}
        # R2 (plan v3 step 9): the per-manager `_compiled_terms`
        # OrderedDict was removed. The compiled-term cache now lives
        # in `fsm_llm.lam.compile_fsm_cached` (lru_cache(maxsize=64)).
        # `get_compiled_term` is a 3-line shim — see D-PLAN-07.

        # Configuration
        self.max_history_size = max_history_size
        self.max_message_length = max_message_length

        # Handler system
        self.handler_system = handler_system or HandlerSystem(
            error_mode=handler_error_mode
        )
        # R5 step 3 — composed-term cache. Keyed on
        # (fsm_id, handlers_version) where handlers_version is a
        # monotonically increasing counter incremented on every
        # register_handler. Step 4 will switch the pipeline.py compiled-
        # term consumer over to get_composed_term; step 3 only adds the
        # cache + the get_composed_term method without making it
        # load-bearing for the existing pipeline path.
        self._handlers_version: int = 0
        self._composed_term_cache: dict[tuple[str, int], Term] = {}

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
            # R5 step 4 (D-STEP-04-RESOLUTION) — switched from
            # `get_compiled_term` (base term, no handler splices) to
            # `get_composed_term` (handler-spliced term). When no
            # handlers are registered, `compose` is identity and the
            # composed-term cache returns the same Term object as the
            # base — no behavior change for handler-free FSMs. When
            # PRE/POST_PROCESSING handlers register, the composed term
            # carries the splice and the pipeline's env extension binds
            # the runner that fires them.
            compiled_term_resolver=self.get_composed_term,
            # M4 — pass the Program-owned Oracle through identity-preserving.
            oracle=self._oracle,
        )

        logger.info(
            f"Enhanced FSM Manager initialized with improved 2-pass architecture - "
            f"history_size={max_history_size}, message_length={max_message_length}"
        )

    # ----------------------------------------------------------
    # Handler registration
    # ----------------------------------------------------------

    def register_handler(self, handler):
        """Register a handler with the system.

        R5 step 3 — increments ``_handlers_version`` so the composed-term
        cache (:meth:`get_composed_term`) returns a fresh composition on
        the next call. The pre-R5 ``HandlerSystem._execute_handlers``
        middleware path is also updated (handlers are appended to
        ``handler_system.handlers``) — both paths see the new handler
        until step 4 deletes the middleware call sites in
        ``dialog/pipeline.py``.
        """
        with self._lock:
            self.handler_system.register_handler(handler)
            self._handlers_version += 1

    def get_composed_term(self, fsm_id: str) -> Term:
        """Return the compiled λ-term for ``fsm_id`` with handlers spliced in.

        R5 step 3 entry point. Looks up the base term via
        :func:`compile_fsm_cached` (kernel-level cache, keyed on FSM JSON),
        then composes registered handlers via
        :func:`fsm_llm.handlers.compose`. The composed result is cached on
        ``(fsm_id, _handlers_version)`` — when a new handler registers,
        the version increments, the next call recomposes, and stale
        composed terms become unreachable (eligible for GC).

        When no handlers are registered (``handler_system.handlers`` is
        empty), :func:`compose` is idempotent and returns the base term
        unchanged — the cache stores that identity for cheap reuse.

        Step 3 ships this method but does **not** wire it into the
        per-turn processing path. The existing
        ``MessagePipeline.process_compiled`` consumer continues to use
        :meth:`get_compiled_term` (the base, uncomposed term) and the
        legacy ``HandlerSystem._execute_handlers`` middleware. Step 4
        flips that switch.
        """
        with self._lock:
            key = (fsm_id, self._handlers_version)
            cached = self._composed_term_cache.get(key)
            if cached is not None:
                return cached

        # Resolve outside the lock — base compile is already cached at
        # the kernel level and is thread-safe.
        from ..handlers import compose

        base = self.get_compiled_term(fsm_id)
        handlers = list(self.handler_system.handlers)
        composed = compose(base, handlers)

        with self._lock:
            # Re-check under lock in case another thread populated.
            existing = self._composed_term_cache.get(key)
            if existing is not None:
                return existing
            # Bound the cache so a long-running manager with many handler
            # versions doesn't grow unboundedly. Drop the lowest-version
            # entry per fsm_id when over a soft cap (mirrors the kernel
            # compile cache's lru_cache(maxsize=64) discipline).
            if len(self._composed_term_cache) >= 128:
                # Evict the oldest entry deterministically (FIFO over keys).
                oldest_key = next(iter(self._composed_term_cache))
                self._composed_term_cache.pop(oldest_key, None)
            self._composed_term_cache[key] = composed
            return composed

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
        """Get compiled λ-term for ``fsm_id`` (R2 — kernel cache shim).

        Routes to :func:`fsm_llm.lam.compile_fsm_cached` after resolving
        the FSMDefinition through this manager's loader cache. Cache
        identity is keyed on ``(fsm_id, fsm.model_dump_json())`` per
        ``# DECISION D-002`` in ``lam/fsm_compile.py``.

        See plan v3 D-PLAN-07 for rationale. Pre-R2 this method
        maintained a per-manager ``_compiled_terms`` OrderedDict; that
        bookkeeping has been removed.
        """
        defn = self.get_fsm_definition(fsm_id)
        return compile_fsm_cached(defn, fsm_id=fsm_id)

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
                # DECISION D-S11-00 — streaming routes unconditionally through
                # the compiled λ-term at tier=3. The legacy `process_stream`
                # wrapper was deleted in S11; no silent fallback (D-S8b-02).
                yield from self._pipeline.process_stream_compiled(
                    instance, message, conversation_id, tier=3
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
            # DECISION D-S11-00 — non-streaming routes unconditionally through
            # the compiled λ-term at tier=3. The legacy `process` method was
            # deleted in S11; no silent fallback (D-S8b-02).
            return self._pipeline.process_compiled(
                instance, message, conversation_id, tier=3
            )

        except FSMError:
            self._rollback_user_message(instance, message, log)
            raise
        except Exception as e:
            log.error(f"Error processing message: {e!s}\n{traceback.format_exc()}")
            self._rollback_user_message(instance, message, log)

            try:
                self._pipeline._execute_handlers(
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
        self._pipeline._execute_handlers(
            self.instances[conversation_id],
            timing,
            conversation_id,
            current_state=current_state,
            target_state=target_state,
            updated_keys=updated_keys,
            error_context=error_context,
        )
