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
    has_internal_prefix,
)
from .definitions import (
    FSMContext,
    FSMDefinition,
    FSMError,
    FSMInstance,
    State,
)
from .handlers import HandlerSystem, HandlerTiming

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

    Manages instances, thread-safe locking, and FSM definition caching.
    Delegates 2-pass message processing to :class:`MessagePipeline`.
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

        except FSMError as e:
            self._cleanup_after_failed_start(conversation_id, instance, e, log)
            raise
        except Exception as e:
            self._cleanup_after_failed_start(conversation_id, instance, e, log)
            log.error(f"Error generating initial response: {e!s}")
            raise FSMError(f"Failed to start conversation: {e!s}") from e

    def _cleanup_after_failed_start(
        self,
        conversation_id: str,
        instance: FSMInstance,
        original: BaseException,
        log: Any,
    ) -> None:
        """Fire END_CONVERSATION and release resources after a failed start.

        Both `start_conversation` failure arms share this. Contract:

        - `instances` and `_conversation_locks` are ALWAYS emptied, even when
          the handler pass raises or is interrupted (the `finally`).
        - Returns normally when the handler pass succeeds or when the handler
          system chose to swallow the failure; the caller then re-raises
          `original`.
        - Raises the handler's exception, chained from `original`, when the
          handler system decided it must propagate.

        Args:
            conversation_id: Conversation being torn down.
            instance: Its instance, for the handlers' `current_state`.
            original: The failure `start_conversation` is already unwinding from.
            log: Conversation-bound logger.
        """
        try:
            self._execute_handlers(
                HandlerTiming.END_CONVERSATION,
                conversation_id,
                current_state=instance.current_state,
            )
        except Exception as cleanup_err:
            # DECISION plan-2026-07-19T191147-4b664252/D-006
            # The CLEANUP exception wins; `original` becomes its `__cause__`.
            # Do NOT "fix" this back to logging `cleanup_err` and re-raising
            # `original`: that was the F-07 defect. A `critical=True` handler
            # always raising is a package-wide contract, and this was the one
            # firing site of eight where it did not hold.
            # Do NOT narrow this to `getattr(handler, "critical", False)` either.
            # Whatever escapes `_execute_handlers` has ALREADY been through the
            # handler system's swallow decision (non-critical + "continue" never
            # reaches here), so re-raising it unconditionally is the same rule
            # every other firing site follows.
            # `original` is logged, not just chained, because a traceback is not
            # guaranteed to reach the operator; the two failures have different
            # root causes and both are needed to diagnose.
            log.error(
                "END_CONVERSATION handler failed while cleaning up a failed "
                f"start_conversation: {cleanup_err} "
                f"(original failure being unwound: {original!s})"
            )
            raise cleanup_err from original
        finally:
            self._cleanup_conversation_resources(conversation_id)

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
            except GeneratorExit:
                # Consumer abandoned the iterator before Pass 2 finished (e.g. a
                # client timeout before the first token). GeneratorExit is a
                # BaseException, so it bypasses the handlers below; roll back the
                # just-added user message so history is not left with an orphaned
                # user turn (CA3-002). _rollback_user_message only pops when the
                # last exchange is a user message, so a partial reply already
                # added by the pipeline is preserved.
                self._rollback_user_message(instance, message, log)
                raise
            except (KeyboardInterrupt, SystemExit):
                # DECISION plan-2026-07-18T162030-a02151fe/D-014
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
        except (KeyboardInterrupt, SystemExit):
            # DECISION plan-2026-07-18T162030-a02151fe/D-014
            # This clause must stay BEFORE `except Exception` and must re-raise
            # BARE. Do NOT merge it into the Exception clause below and do NOT
            # wrap it in FSMError: KeyboardInterrupt/SystemExit derive from
            # BaseException precisely so ordinary handlers cannot swallow them,
            # and wrapping would hand them to every `except Exception` upstream.
            # Without the clause the signal skips _rollback_user_message and
            # leaves an orphaned {'user': ...} entry with no matching reply.
            # No ERROR-timing handlers run here either -- an interrupt is not an
            # application error, and running user code while unwinding a signal
            # is how a Ctrl-C becomes unkillable.
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
                # DECISION plan-2026-07-20T040150-876e7164/D-009
                # The HANDLER exception wins; the failure being unwound becomes
                # its `__cause__`. This is the same shape (and the same rule) as
                # `_cleanup_after_failed_start` 150 lines above, which chains
                # `raise cleanup_err from original` for a structurally identical
                # unwind-time handler pass.
                # Do NOT restore the `log.warning(...)`-and-continue swallow that
                # used to be here. Its stated justification was that promoting the
                # reporter's own failure destroys the diagnosis of the failure
                # being reported — that is FALSE for `raise X from Y`, which keeps
                # BOTH: `handler_err` reaches the caller (a `critical=True`
                # handler always raising is a package-wide contract) and `e`
                # survives as `handler_err.__cause__`, printed in any default
                # traceback. The `log.error` below states both failures explicitly
                # as well, because a traceback is not guaranteed to reach the
                # operator.
                # Do NOT narrow this to `getattr(handler, "critical", False)`:
                # whatever escapes `execute_handlers` has ALREADY passed through
                # the handler system's swallow decision (a non-critical handler
                # under "continue" never reaches here).
                log.error(
                    "ERROR-timing handler failed while unwinding a failed "
                    f"process_message: {handler_err} "
                    f"(original failure being unwound: {e!s})"
                )
                raise handler_err from e

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
            k: v for k, v in instance.context.data.items() if not has_internal_prefix(k)
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

                # DECISION plan-2026-07-20T040150-876e7164/D-009
                # Scoped rollback — shape (c) of `pipeline.py`'s three different
                # partial-commit contracts (D-005 of
                # plan-2026-07-19T191147-4b664252). Snapshot EXACTLY the keys this
                # call is about to write, commit, and restore only those keys if
                # the CONTEXT_UPDATE pass raises. Without it a caller that catches
                # the handler failure and reports "update rejected" was simply
                # wrong: the update had already landed permanently.
                # Do NOT "align" this with a full `context.data` deepcopy: that
                # would also discard the deltas of CONTEXT_UPDATE handlers that
                # already SUCCEEDED, which D-006 guarantees survive at this timing
                # point. Precedence note: if a handler delta touched one of
                # `context_update`'s own keys, the rollback wins.
                # BOUNDARY (known, deliberate): the snapshot is SHALLOW. Key
                # coverage is exact — `FSMContext.update` writes precisely
                # `new_data`'s keys and derives nothing — but `pre_commit` holds
                # the SAME objects as `context.data[key]`. A handler that mutates
                # a pre-existing dict/list value IN PLACE before failing is
                # therefore NOT restored. Do not "fix" this with a deep copy:
                # D-005/D-020 rejected exactly that, because handlers are
                # contractually supposed to return a delta rather than reach into
                # `context.data`, and the copy cost would be paid on every call.
                pre_commit = {
                    key: instance.context.data[key]
                    for key in context_update
                    if key in instance.context.data
                }
                instance.context.update(context_update)

                try:
                    self._execute_handlers(
                        HandlerTiming.CONTEXT_UPDATE,
                        conversation_id,
                        updated_keys=set(context_update.keys()),
                    )
                except Exception as handler_err:
                    log.error(
                        f"CONTEXT_UPDATE handler failed "
                        f"({type(handler_err).__name__}: {handler_err}), rolling "
                        f"back keys {sorted(context_update)}"
                    )
                    for key in context_update:
                        if key in pre_commit:
                            instance.context.data[key] = pre_commit[key]
                        else:
                            instance.context.data.pop(key, None)
                    raise
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
