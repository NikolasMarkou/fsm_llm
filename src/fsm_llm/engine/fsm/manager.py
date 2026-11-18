"""
FSMManager: Conversation lifecycle orchestrator for the 2-pass architecture.

Manages conversation instances, thread-safe locking, and FSM definition caching.
Delegates all message processing to MessagePipeline.

Thread-safe per-conversation. See docs/architecture.md for details.

Layer ``engine`` (plan d454fd62, step 78): the shell of the old ``fsm_llm.fsm``: the
constructor, the definition cache, start, end and context update. The turn path is
``TurnsMixin`` (``turns.py``), the reads, snapshots and restore seed are
``SnapshotsMixin`` (``snapshots.py``).
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any

from pydantic import ValidationError

from fsm_llm.constants import (
    DEFAULT_MAX_FSM_CACHE_SIZE,
    DEFAULT_MAX_HISTORY_SIZE,
    DEFAULT_MAX_MESSAGE_LENGTH,
)
from fsm_llm.context import commit_with_key_rollback
from fsm_llm.context.layout import (
    DataExtractionPromptBuilder,
    DataExtractionPromptConfig,
    FieldExtractionPromptBuilder,
    FieldExtractionPromptConfig,
    ResponseGenerationPromptBuilder,
    ResponsePromptConfig,
)
from fsm_llm.core.definitions import (
    Conversation,
    FSMContext,
    FSMDefinition,
    FSMError,
    FSMInstance,
    State,
)
from fsm_llm.core.loaders import load_fsm_definition
from fsm_llm.core.transition_evaluator import TransitionEvaluator
from fsm_llm.engine.kinds import StateExecutor
from fsm_llm.engine.turn import MessagePipeline
from fsm_llm.handlers import HandlerErrorMode, HandlerSystem, HandlerTiming
from fsm_llm.llm import LLMInterface
from fsm_llm.logging import logger
from fsm_llm.runtime.stores import check_conversation_id

from .budget import TurnBudget, check_turn_limits, open_budget
from .helpers import _conversation_entry, with_conversation_context
from .snapshots import SnapshotsMixin
from .turns import TurnsMixin


class FSMManager(TurnsMixin, SnapshotsMixin):
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
        handler_error_mode: HandlerErrorMode | str = HandlerErrorMode.CONTINUE,
        max_fsm_cache_size: int = DEFAULT_MAX_FSM_CACHE_SIZE,
        *,
        executors: Mapping[str, StateExecutor] | None = None,
    ):
        """Create the manager.

        ``executors`` is a ``{kind_name: StateExecutor}`` map for THIS manager's
        pipeline only (it overrides the module table of ``engine/kinds/registry.py``; see
        ``engine.kinds.check_executors``: ``TypeError`` for a bad value,
        ``ValueError`` for an unknown kind name). Keyword-only, internal until
        the kind extension point is public.
        """
        if llm_interface is None:
            raise ValueError("llm_interface is required and cannot be None")
        # DECISION plan-2026-09-22T080837-8b258a25/D-041
        # Do NOT relax this to `>= 0` to mean "caching disabled": the eviction
        # branch in `_load_fsm_definition` runs `self.fsm_cache.popitem()` once
        # the size bound is reached, so a bound of 0 (or negative) pops an empty
        # OrderedDict and every start_conversation dies with a cryptic
        # "dictionary is empty" KeyError. Disabling the cache needs the eviction
        # branch fixed first. See D-041.
        if max_fsm_cache_size < 1:
            raise ValueError(
                f"max_fsm_cache_size must be >= 1, got {max_fsm_cache_size}"
            )

        # The history and length limits have one definition, the ``Conversation``
        # fields; validating through them here fails at construction, naming the
        # argument, instead of at every ``start_conversation``.
        try:
            limits = Conversation(
                max_history_size=max_history_size,
                max_message_length=max_message_length,
            )
        except ValidationError as e:
            raise ValueError(
                "; ".join(
                    f"{err['loc'][0]}: {err['msg']}, got {err['input']!r}"
                    for err in e.errors()
                )
            ) from None

        self.fsm_loader = fsm_loader
        self.llm_interface = llm_interface

        # Lock for thread-safe access to shared class-level dicts
        self._lock = threading.Lock()
        # Per-conversation locks to prevent concurrent mutations on the same instance
        self._conversation_locks: dict[str, threading.RLock] = {}
        # Conversation ids with a turn in flight (D-018 re-entrancy guard). Added
        # under ``_lock`` after ``conv_lock`` is acquired; discarded WITHOUT any
        # lock in the ``finally`` (a lone ``set.discard`` is atomic under the GIL).
        self._active_turns: set[str] = set()
        # Lifetime turn budgets (plan 5a6a1f80 D-010, ``budget.py``): one record per manager
        # conversation id that was given a limit. Written under ``_lock`` with the instance,
        # charged in ``_enter_turn``, dropped with the instance; a conversation with no limit
        # has no entry.
        self._turn_budgets: dict[str, TurnBudget] = {}

        # Cache and instance management
        self.fsm_cache: OrderedDict[str, FSMDefinition] = OrderedDict()
        self._max_fsm_cache_size = max_fsm_cache_size
        # Leaf lock for ``fsm_cache`` only (see the anchor in get_fsm_definition).
        self._fsm_cache_lock = threading.Lock()
        self.instances: dict[str, FSMInstance] = {}

        # Configuration (the validated, coerced values: 2.0 and "5" mean 2 and 5)
        max_history_size = limits.max_history_size
        self.max_history_size = max_history_size
        self.max_message_length = max_message_length

        # Handler system
        self.handler_system = handler_system or HandlerSystem(
            handler_error_mode=handler_error_mode
        )

        # Prompt builders (stored for attribute access by tests/API)
        # DECISION plan-2026-11-06T093029-0068b055/D-004
        # The default builders show `max_history_size` messages: the number is
        # passed UNCHANGED as `max_history_messages` (the unit of the default 5,
        # so default prompts stay byte-identical). Do NOT pass `2 * max_history_size`
        # (the honest exchange unit): it grows every default prompt from 5 to 10
        # messages and moves the eval baselines (carried, needs a live run), and
        # D-038 / test_bugs.py pin the unmultiplied number. Do NOT leave the
        # defaults config-less either: then the knob only sizes storage and the
        # model never sees a larger window. Pass the VALIDATED value (the one
        # `Conversation` coerced above), never the raw argument: 2.0 or "5" would
        # reach the slice or comparison in the prompt builder as a float or str.
        # engine/api builds no builders of its own so this stays the one site.
        self.data_extraction_prompt_builder = (
            data_extraction_prompt_builder
            or DataExtractionPromptBuilder(
                DataExtractionPromptConfig(max_history_messages=max_history_size)
            )
        )
        self.response_generation_prompt_builder = (
            response_generation_prompt_builder
            or ResponseGenerationPromptBuilder(
                ResponsePromptConfig(max_history_messages=max_history_size)
            )
        )
        self.field_extraction_prompt_builder = (
            field_extraction_prompt_builder
            or FieldExtractionPromptBuilder(
                FieldExtractionPromptConfig(max_history_messages=max_history_size)
            )
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
            executors=executors,
        )

        logger.info(
            f"Enhanced FSM Manager initialized with the 2-pass architecture - "
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
        """Get FSM definition with caching and LRU eviction.

        Takes only ``_fsm_cache_lock`` (never ``_lock``); the loader runs
        outside it. When two threads miss at once, both load and the first
        inserted definition wins; the second load is discarded.
        """
        # DECISION plan-2026-09-22T080837-8b258a25/D-036
        # _fsm_cache_lock is a LEAF lock: while holding it do NOT take `_lock`,
        # a conv_lock, log, or call fsm_loader (a custom loader may do anything).
        # Do NOT go back to `_lock` here: every turn resolves its def through this
        # method, so that serialised all conversations on one lock. See D-036.
        with self._fsm_cache_lock:
            if fsm_id in self.fsm_cache:
                self.fsm_cache.move_to_end(fsm_id)
                return self.fsm_cache[fsm_id]

        logger.info(f"Loading FSM definition: {fsm_id}")
        loaded = self.fsm_loader(fsm_id)

        evicted_key = None
        with self._fsm_cache_lock:
            if fsm_id in self.fsm_cache:
                self.fsm_cache.move_to_end(fsm_id)
                return self.fsm_cache[fsm_id]
            if len(self.fsm_cache) >= self._max_fsm_cache_size:
                evicted_key, _ = self.fsm_cache.popitem(last=False)
            self.fsm_cache[fsm_id] = loaded
        if evicted_key is not None:
            logger.debug(f"Evicted FSM definition from cache: {evicted_key}")
        return loaded

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

    # DECISION plan-2026-09-12T135914-45a654de/D-019
    # Do NOT rename this back to (or add a sibling method named)
    # `get_current_state` on FSMManager. `API.get_current_state(conversation_id)
    # -> str` already owns that name as a public, string-returning contract;
    # this method returns the full `State` object. The two previously shared
    # the same name with incompatible return types (finding 16) -- reusing
    # `get_current_state` here would silently reintroduce that collision. See
    # decisions.md D-019.
    def resolve_state_definition(
        self, instance: FSMInstance, conversation_id: str | None = None
    ) -> State:
        """Resolve the full ``State`` definition for an instance's current state.

        Internal-facing counterpart to ``API.get_current_state(conversation_id)
        -> str``, which returns only the state ID string for public consumers.
        This method returns the full ``State`` object (description, purpose,
        transitions, etc.) and is used internally by ``FSMManager``/pipeline
        code that needs more than the bare state ID. The two methods
        deliberately have different names (this one was renamed from
        ``get_current_state``) to avoid the same-name/incompatible-return-type
        collision that previously existed between them.
        """
        return self._pipeline.get_state(instance, conversation_id)

    def start_conversation(
        self,
        fsm_id: str,
        initial_context: dict[str, Any] | None = None,
        *,
        suppress_start: bool = False,
        conversation_id: str | None = None,
        max_steps: int | None = None,
        max_seconds: float | None = None,
    ) -> tuple[str, str]:
        """Start new conversation.

        Args:
            fsm_id: FSM definition id to instantiate.
            initial_context: Optional initial context data.
            suppress_start: Resume path. When True the instance is created,
                registered and internal-key-seeded exactly as normal, but the
                START_CONVERSATION handler pass AND the Pass-2 greeting are
                skipped, returning ``(conversation_id, "")``. Used by session
                restore so resuming does not re-fire start-of-conversation side
                effects or burn an LLM greeting. Default False keeps the normal
                start path byte-for-byte unchanged.
            conversation_id: Keep this id instead of minting a ``uuid4`` (the seam
                ``API.resume`` uses, plan 5a6a1f80 D-008). It must pass
                ``check_conversation_id`` (``EventPayloadError``, an ``FSMError``,
                otherwise) and not be live in ``instances`` (``FSMError``): nothing is
                registered when it is refused. Default ``None`` mints the id as always.
            max_steps: Lifetime turn budget of THIS conversation id (``converse``,
                ``advance`` and both streams each count one, a rolled-back or refused-as-
                terminal turn included, the greeting not): turn ``max_steps + 1`` raises
                ``RunBudgetExceededError("steps", max_steps, max_steps)``. ``None`` (default)
                is no limit. An int of at least 1, else ``ValueError`` before anything starts.
            max_seconds: Lifetime wall-clock budget from the start of this conversation,
                checked at the start of each turn (never mid-turn): a turn that starts
                after it is spent raises ``RunBudgetExceededError("seconds", ...)``. ``None``
                is no limit; a number above 0, else ``ValueError``.

        Returns:
            Tuple of (conversation_id, initial_response)
        """
        check_turn_limits(max_steps, max_seconds)
        instance = self._create_instance(fsm_id)
        explicit_id = conversation_id is not None
        if conversation_id is None:
            conversation_id = str(uuid.uuid4())
        else:
            check_conversation_id(conversation_id)
        log = logger.bind(conversation_id=conversation_id, package="fsm_llm")

        if initial_context is not None and not isinstance(initial_context, dict):
            raise FSMError(
                f"initial_context must be a dict, got {type(initial_context).__name__}"
            )
        if initial_context:
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
            if explicit_id and conversation_id in self.instances:
                raise FSMError(f"conversation id '{conversation_id}' is already live")
            self.instances[conversation_id] = instance
            self._conversation_locks[conversation_id] = threading.RLock()
            budget = open_budget(max_steps, max_seconds)
            if budget is not None:
                self._turn_budgets[conversation_id] = budget

        if suppress_start:
            # Resume path: skip the START_CONVERSATION handler pass and the
            # Pass-2 greeting. Instance creation, lock registration and
            # internal-key seeding above are unchanged. Returns before touching
            # the default path below, so that path stays byte-for-byte the same.
            log.info(
                f"Started conversation [{conversation_id}] with FSM [{fsm_id}] "
                "(suppress_start: resume)"
            )
            return conversation_id, ""

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
        except (KeyboardInterrupt, SystemExit):
            # An interrupt is the caller's signal, never wrapped and no END handler runs on it
            # (as in `turns.py`), but nothing may stay registered for an id nobody holds.
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
        except (KeyboardInterrupt, SystemExit):
            self._cleanup_conversation_resources(conversation_id)
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
            # DECISION plan-2026-07-19T191147-4b664252/D-006 [STALE]
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

    @with_conversation_context
    def update_conversation_context(
        self, conversation_id: str, context_update: dict[str, Any], log: Any = None
    ) -> None:
        """Update conversation context data."""
        if not isinstance(context_update, dict):
            raise FSMError("context_update must be a dictionary")

        with self._lock:
            # L11: an instance present with no per-conversation lock is a broken
            # create-together / remove-together invariant. Fail loud instead of
            # silently proceeding with an UNLOCKED write (which would race the
            # 2-pass write path): missing_lock="raise", as in _read_under_lock.
            instance, conv_lock = _conversation_entry(
                self, conversation_id, missing_lock="raise"
            )

            # Acquire per-conversation lock while still holding _lock to prevent
            # the conversation from being cleaned up between lookup and acquire.
            if not conv_lock.acquire(blocking=False):
                raise FSMError(
                    f"Conversation {conversation_id} is already being processed "
                    "by another thread"
                )
        try:
            if context_update:
                log.info(f"Updating context with keys: {list(context_update.keys())}")

                # DECISION plan-2026-07-20T040150-876e7164/D-009 [STALE]
                # Scoped rollback - shape (c) of the turn pipeline's (`engine/turn/`) three different
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
                # The snapshot is SHALLOW on purpose: see
                # `commit_with_key_rollback` (context/commit.py), which holds the
                # BOUNDARY note and does the commit.
                commit_with_key_rollback(
                    instance.context,
                    context_update,
                    lambda: self._execute_handlers(
                        HandlerTiming.CONTEXT_UPDATE,
                        conversation_id,
                        updated_keys=set(context_update.keys()),
                    ),
                    lambda handler_err: log.error(
                        f"CONTEXT_UPDATE handler failed "
                        f"({type(handler_err).__name__}: {handler_err}), rolling "
                        f"back keys {sorted(context_update)}"
                    ),
                )
        finally:
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
            self._turn_budgets.pop(conversation_id, None)

    def turn_budget_spent(self, conversation_id: str) -> str | None:
        """Name the lifetime budget that forbids the next turn of ``conversation_id``.

        Contract: ``"seconds"`` or ``"steps"`` (seconds first when both are spent), or ``None``
        for a conversation with no limit, an unknown or ended id, or a budget with room left.
        Reads one record and takes no lock; the answer is a fact of the record, not of the
        text of any ``RunBudgetExceededError``, so a driver can tell its own frame's budget from
        another one's.
        """
        budget = self._turn_budgets.get(conversation_id)
        return None if budget is None else budget.spent()

    def turn_budget_steps(self, conversation_id: str) -> int | None:
        """Count the turns charged to ``conversation_id`` so far (its lifetime count).

        Contract: the number of turns the budget has counted (a rolled-back turn included,
        the greeting never), or ``None`` for a conversation with no limit, an unknown or
        ended id. Reads one record and takes no lock; the same fact ``turn_budget_spent``
        judges, so a driver that traces or reports steps reads it here and keeps no count.
        """
        budget = self._turn_budgets.get(conversation_id)
        return None if budget is None else budget.steps_taken

    @with_conversation_context
    def end_conversation(self, conversation_id: str, log: Any = None) -> None:
        """End conversation and clean up resources."""
        with self._lock:
            _, conv_lock = _conversation_entry(
                self, conversation_id, missing_lock="unlocked"
            )

        # Acquire per-conversation lock to ensure no concurrent process_message.
        # Use timeout to prevent indefinite blocking if a thread is stuck.
        # DECISION plan-2026-09-21T203800-8a03483a/D-036
        # On timeout, REFUSE: log and raise FSMError, leaving the instance and
        # its lock untouched. Do NOT go back to "proceeding with cleanup": a
        # turn still holds the lock, so END handlers and the instance teardown
        # ran under a live turn (C12), and the `finally` then released a lock
        # this thread never acquired. The caller may retry once the turn ends.
        if conv_lock is not None:
            self._acquire_or_refuse(conv_lock, conversation_id)
        try:
            self._execute_handlers(HandlerTiming.END_CONVERSATION, conversation_id)
        finally:
            # Clean up while holding conv_lock to prevent race with process_message.
            # Lock ordering is safe: process_message uses non-blocking conv_lock
            # acquisition, so _lock→conv_lock never creates circular wait.
            self._cleanup_conversation_resources(conversation_id)
            if conv_lock is not None:
                conv_lock.release()
        log.info(f"Conversation {conversation_id} ended")

    def prune_orphaned_locks(self) -> list[str]:
        """Remove locks for conversations that no longer have active instances.

        Pure lock bookkeeping: no handler runs and no conversation ends. Not
        to be confused with ``API.cleanup_stale_conversations``, the idle-time
        reaper that ends conversations. Returns the pruned conversation ids.
        """
        with self._lock:
            stale_ids = [
                cid for cid in self._conversation_locks if cid not in self.instances
            ]
            for cid in stale_ids:
                self._conversation_locks.pop(cid, None)
        if stale_ids:
            logger.info(f"Cleaned up {len(stale_ids)} stale conversation locks")
        return stale_ids

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
            current_state=current_state,
            target_state=target_state,
            updated_keys=updated_keys,
            error_context=error_context,
        )
