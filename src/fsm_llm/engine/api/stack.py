"""FSM stacking of ``API`` (``StackMixin``): ``push_fsm``, ``pop_fsm``, the context merge on pop,
the top-of-stack resolver, and the two public stack types ``FSMStackFrame`` and ``ContextMergeStrategy``.

Layer ``engine`` (plan d454fd62, step 81): split out of the old ``fsm_llm.api``; the method bodies
are unchanged. ``_stack_lock`` is a plain non-reentrant ``Lock``, never held across an ``FSMManager``
call (the anchors below).
"""

from __future__ import annotations

import os
import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from fsm_llm.constants import DEFAULT_MAX_STACK_DEPTH
from fsm_llm.core.definitions import FSMDefinition, FSMError
from fsm_llm.core.state_kinds import kind_of
from fsm_llm.logging import logger

from .budget import budget_kwargs
from .events import (
    child_event_scope,
    record_context_update,
    record_pop,
    record_push,
)
from .helpers import _require_classifier_interface, _unexpected_error
from .surface import _ApiSurface


class FSMStackFrame(BaseModel):
    """Represents a single FSM in the conversation stack."""

    fsm_definition: FSMDefinition
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


class StackMixin(_ApiSurface):
    """The FSM stack of one ``API``: push, pop, merge, and the top-of-stack lookups."""

    # ==========================================
    # FSM STACKING METHODS (Enhanced)
    # ==========================================

    def push_fsm(
        self,
        conversation_id: str,
        new_fsm_definition: FSMDefinition | dict[str, Any] | str | os.PathLike[str],
        context_to_pass: dict[str, Any] | None = None,
        return_context: dict[str, Any] | None = None,
        shared_context_keys: list[str] | None = None,
        preserve_history: bool = False,
        inherit_context: bool = True,
        *,
        max_steps: int | None = None,
        max_seconds: float | None = None,
    ) -> str:
        """Push new FSM onto conversation stack with enhanced context management.

        ``max_steps`` and ``max_seconds`` are the lifetime turn budget of the NEW frame (its
        own manager conversation, never shared with the parent), instead of the ``API``
        default (``None``: the default, itself no limit); ``ValueError`` before anything is
        registered when invalid. The push greeting is not a turn and is not counted.
        """
        budget = budget_kwargs(self, max_steps, max_seconds)
        with self._stack_lock:
            if conversation_id not in self.conversation_stacks:
                raise FSMError(f"Conversation not found: {conversation_id}")
            self._validate_stack_depth(conversation_id)

        processed_fsm_id = None
        new_conversation_id = None
        push_succeeded = False
        try:
            processed_fsm_def, processed_fsm_id = self.process_fsm_definition(
                new_fsm_definition
            )
            _require_classifier_interface(processed_fsm_def, self.llm_interface)
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
                current_fsm_id,
                context_to_pass,
                preserve_history,
                inherit_context,
                child_definition=processed_fsm_def,
            )

            with child_event_scope(self, conversation_id):
                new_conversation_id, response = self.fsm_manager.start_conversation(
                    processed_fsm_id, initial_context=initial_context, **budget
                )
            with self._stack_lock:
                # DECISION plan-2026-07-21T045419-9925aa3a/D-011
                # Do NOT re-add `_temp_fsm_definitions.pop(processed_fsm_id, None)`
                # here. The pushed def used to survive only in the LRU `fsm_cache`;
                # once ~65 distinct FSM ids loaded it was evicted and the loader
                # fell through to `load_fsm_definition`, which raises
                # `FSMDefinitionNotFoundError` (a `ValueError`) for a content-hash
                # id — bricking the
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
                stack = self.conversation_stacks[conversation_id]
                stack.append(new_frame)
                # The depth this push logs is the stack length at ITS append, read in the same
                # hold: a second push between two holds made the first one log the other's.
                stack_depth = len(stack)
                # Frame now references the def; clear the in-flight guard
                # atomically with the append (same _stack_lock hold). See D-011.
                self._pending_push_ids.discard(processed_fsm_id)
                push_succeeded = True

            record_push(
                self,
                conversation_id,
                new_conversation_id,
                processed_fsm_id,
                processed_fsm_def,
                stack_depth,
                return_context=new_frame.return_context,
                shared_context_keys=new_frame.shared_context_keys,
                preserve_history=new_frame.preserve_history,
                limits=budget,
            )
            logger.info(
                f"Pushed new FSM onto conversation {conversation_id}, "
                f"stack depth: {stack_depth}"
            )
            return response

        except (FSMError, ValueError):
            raise
        except Exception as e:
            raise _unexpected_error(e, "Error pushing FSM", "Failed to push FSM") from e
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
        *,
        child_definition: FSMDefinition,
    ) -> dict[str, Any]:
        """Build initial context for pushed FSM from inheritance and passed context.

        The inherited part never carries the ``result_key`` of a completion or
        synthesize state of ``child_definition``: that key belongs to the
        child's own calls or run. ``context_to_pass`` is applied after and may
        set it.
        """
        initial_context: dict[str, Any] = {}

        if inherit_context:
            try:
                initial_context.update(
                    self.fsm_manager.get_conversation_data(current_fsm_id)
                )
            except KeyError as e:
                logger.warning(f"Could not inherit context (missing key): {e!s}")
            # DECISION plan-2026-10-01T093600-944e2692/D-029: an inherited
            # completion result made the child's completion state skip its
            # call (skip-if-set) and "answer" with the parent's reply. Do NOT
            # inherit a key a child completion state owns (D-021), and do NOT
            # drop it from an explicit `context_to_pass`. See D-029.
            # DECISION plan-2026-11-12T073150-d454fd62/D-022: the same drop for
            # a synthesize state's result key, for a different reason. A child
            # pushed here with `inherit_context=True` routed on the parent's
            # stale `synthesis_result` (its edges must read that key) before
            # any run wrote one. It is NOT that a runner skips the child: the
            # runner pushes its own children with `inherit_context=False` and
            # never serves a child pushed by a user. Do NOT restore that
            # "skips its own run" reason, and do NOT drop the key from an
            # explicit `context_to_pass`.
            for state in child_definition.states.values():
                spec = kind_of(state)
                if not spec.inherits_result_on_push:
                    key = spec.result_key(state)
                    if key is not None:
                        initial_context.pop(key, None)

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
        ``FSMDefinitionNotFoundError`` (a ``ValueError``) for the conversations
        still using it.

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
        # DECISION plan_2026-05-29_d9092060/D-001 [STALE]
        # Narrow lock scope: snapshot frame references under _stack_lock, then release
        # the lock before calling fsm_manager methods (which acquire per-conversation
        # RLocks and can block). Re-acquire _stack_lock only to pop the stack entry.
        # Do NOT revert to a single wide `with self._stack_lock:` covering the whole
        # method — that blocks list_active_conversations/push_fsm on other conversations
        # for the full duration of FSM teardown I/O.
        with self._stack_lock:
            if conversation_id not in self.conversation_stacks:
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

            refused = False
            try:
                self.fsm_manager.end_conversation(current_frame.conversation_id)
            except Exception:
                # A REFUSED end (`ConversationBusyError`: a turn holds the child's lock past the
                # timeout) tears nothing down, so the child instance is still registered: keep
                # its frame, as `API.end_conversation` does (D-050 and D-053 of plan 8a03483a,
                # anchored in lifecycle.py). Dropping it orphans a live child that no stack
                # entry reaches and that ending the root never ends. `has_instance` is read
                # HERE, outside `_stack_lock` (never held across a manager call).
                refused = self.fsm_manager.has_instance(current_frame.conversation_id)
                raise
            finally:
                if not refused:
                    # Re-acquire lock only to mutate the stack
                    with self._stack_lock:
                        inner_stack = (
                            self.conversation_stacks.get(conversation_id) or []
                        )
                        for idx in range(len(inner_stack) - 1, -1, -1):
                            if inner_stack[idx] is current_frame:
                                del inner_stack[idx]
                                break
                        else:
                            logger.debug(
                                f"pop_fsm: frame {current_frame.conversation_id} already "
                                f"absent from stack {conversation_id}; nothing to remove"
                            )
                        # the depth this pop logs: the stack right after ITS removal
                        stack_depth = len(inner_stack)

            # The popped frame is gone; release any temp def it was the last
            # frame to reference (D-011). Reference-aware, so a sub-FSM shared
            # with another live conversation survives this pop.
            self._release_unreferenced_temp_definitions()

            response = self._generate_resume_message(context_to_merge)
            record_pop(
                self,
                conversation_id,
                current_frame.conversation_id,
                current_frame.fsm_id,
                merge_strategy_enum.value,
                len(context_to_merge or {}),
                stack_depth,
                parent_id=previous_frame.conversation_id,
            )
            logger.info(
                f"Popped FSM from conversation {conversation_id}, "
                f"stack depth: {stack_depth}"
            )
            return response

        except (FSMError, ValueError):
            raise
        except Exception as e:
            raise _unexpected_error(e, "Error popping FSM", "Failed to pop FSM") from e

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
        raw_context = (
            self.fsm_manager.copy_raw_context_data(frame.conversation_id) or {}
        )
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
            # C11: the definition's NAME, not `str(definition)` (the whole
            # pydantic repr: every state, persona and instruction text).
            summary_context = {
                "_sub_conversation_summary": {
                    "fsm_type": current_frame.fsm_definition.name,
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
        else:  # ContextMergeStrategy.PRESERVE: the enum has two members
            merged_context = current_context.copy()
            for key, value in context_to_merge.items():
                if key not in current_context:
                    merged_context[key] = value

        # Only pass changed keys to avoid triggering handlers for unchanged data
        diff = {
            k: v
            for k, v in merged_context.items()
            if k not in current_context or current_context[k] != v
        }
        if diff:
            self.fsm_manager.update_conversation_context(conversation_id, diff)

    def _generate_resume_message(self, merged_context: dict[str, Any]) -> str:
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
        record_context_update(self, conversation_id, current_fsm_id, context_update)

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
