"""The conversation reads and the end of a conversation for ``API`` (``LifecycleMixin``): the
ended-conversation cache, the four getters, ``end_conversation``, ``list_active_conversations``
and ``cleanup_stale_conversations``.

Layer ``engine`` (plan d454fd62, step 81): split out of the old ``fsm_llm.api``; the method bodies
are unchanged. The order in ``end_conversation`` (cache read, then the frames, then the bookkeeping
drop under one ``_stack_lock`` hold) is the contract of the anchors below.
"""

from __future__ import annotations

import time
from typing import Any

from fsm_llm.core.definitions import ConversationBusyError, FSMError
from fsm_llm.logging import logger

from .helpers import _read_or_ended, handle_conversation_errors
from .surface import _ApiSurface

# Bound of the per-API cache of ended-conversation snapshots (oldest evicted).
MAX_ENDED_CONVERSATIONS_CACHE = 10_000


class LifecycleMixin(_ApiSurface):
    """The ended-conversation cache, the getters and the end of one ``API``'s conversations."""

    # ==========================================
    # CONVERSATION MANAGEMENT METHODS
    # ==========================================

    def _ended_cache_entry(self, conversation_id: str) -> dict[str, Any] | None:
        """The ended-conversation cache entry for ``conversation_id``, or None.

        Contract: reads ``_ended_conversations`` under ``_stack_lock`` (held for
        the dict read only; never call it with ``_stack_lock`` held). Returns
        the cached ``{"data", "state", "history"}`` snapshot; never raises.
        """
        with self._stack_lock:
            return self._ended_conversations.get(conversation_id)

    def _conversation_gone(self, conversation_id: str) -> bool:
        """True when ``conversation_id`` no longer resolves to a live FSM
        instance (its stack is gone, or the top frame's instance was torn
        down). Contract: the licence for a getter's ended-cache fallback;
        takes ``_stack_lock`` then ``fsm_manager._lock`` (each for one read,
        never nested); never raises."""
        try:
            current_fsm_id = self._get_current_fsm_conversation_id(conversation_id)
        except ValueError:
            return True
        return not self.fsm_manager.has_instance(current_fsm_id)

    @handle_conversation_errors
    def get_data(self, conversation_id: str) -> dict[str, Any]:
        """Get collected data from current FSM."""
        return _read_or_ended(
            self,
            conversation_id,
            lambda manager, frame_id: manager.get_conversation_data(frame_id),
            "data",
            {},
        )

    @handle_conversation_errors
    def has_conversation_ended(self, conversation_id: str) -> bool:
        """Check if current FSM has ended.

        Returns ``True`` for a terminal state and for a conversation that has
        ended and been closed (its snapshot is cached). An ID that was never
        started, or whose snapshot is gone, answers ``False`` rather than
        raising ``ValueError`` like the other getters.
        """
        try:
            current_fsm_id = self._get_current_fsm_conversation_id(conversation_id)
            ended: bool = self.fsm_manager.has_conversation_ended(current_fsm_id)
            return ended
        except ConversationBusyError:
            raise
        except (ValueError, KeyError, FSMError):
            if self._conversation_gone(conversation_id):
                return self._ended_cache_entry(conversation_id) is not None
            raise

    @handle_conversation_errors
    def get_current_state(self, conversation_id: str) -> str:
        """Get current state of active FSM."""
        return _read_or_ended(
            self,
            conversation_id,
            lambda manager, frame_id: manager.get_conversation_state(frame_id),
            "state",
            "unknown",
        )

    @handle_conversation_errors
    def get_conversation_history(self, conversation_id: str) -> list[dict[str, str]]:
        """Get conversation history for current FSM (cached history once ended)."""
        return _read_or_ended(
            self,
            conversation_id,
            lambda manager, frame_id: manager.get_conversation_history(frame_id),
            "history",
            [],
        )

    def _frame_instance_present(self, frame_id: str) -> bool:
        """True if the FSMManager still holds ``frame_id``'s instance. After a
        failed end this means the end was refused and nothing was torn down."""
        return self.fsm_manager.has_instance(frame_id)

    def _drop_ended_frames(self, conversation_id: str, ended: set[str]) -> None:
        """Remove already-ended frames from a conversation's live stack."""
        if not ended:
            return
        with self._stack_lock:
            stack = self.conversation_stacks.get(conversation_id)
            if stack is not None:
                stack[:] = [f for f in stack if f.conversation_id not in ended]

    @handle_conversation_errors("Failed to end conversation")
    def end_conversation(self, conversation_id: str) -> None:
        """End conversation and clean up all FSMs in stack.

        If a turn holds a frame's lock past
        ``END_CONVERSATION_LOCK_TIMEOUT_SECONDS``, raises
        ``ConversationBusyError`` (an ``FSMError``) and the conversation stays
        active with its remaining frames; retry after the turn. Frames above
        the refusing one that were already ended are removed from the stack.
        Any other failure while reading the ended-conversation cache is
        logged and the end proceeds without a cache entry.
        """
        # DECISION plan-2026-09-21T203800-8a03483a/D-050
        # Read the cache, then end the frames, and only THEN drop the API
        # bookkeeping. Do NOT pop conversation_stacks /
        # _last_accessed first: a refused end (C12) then vanished from the API
        # while its FSMManager instance stayed allocated, and no sweep could
        # find it again (review W2). Do NOT swallow a refusal: a frame whose
        # end raised while its instance is still present was refused and
        # nothing was torn down, so the caller must see the FSMError. Do NOT
        # use the unbounded getters for the cache: they waited forever.
        try:
            current_fsm_id: str | None = self._get_current_fsm_conversation_id(
                conversation_id
            )
        except ValueError:
            current_fsm_id = None
        ended_cache: dict[str, Any] | None = None
        if current_fsm_id is not None:
            # DECISION plan-2026-09-21T203800-8a03483a/D-053
            # Only the lock-timeout refusal blocks the end. Do NOT re-raise
            # every snapshot error while the instance is present: a context the
            # data walker refuses (`ContextFilterError`) or any other read
            # failure then made the conversation impossible to end or sweep
            # (review pass 2). The ended cache is best-effort. See D-053.
            try:
                ended_cache = self.fsm_manager.get_end_snapshot(current_fsm_id)
            except ConversationBusyError:
                raise  # refused: a turn is running; nothing changed
            except Exception as e:
                logger.warning(
                    f"Ending {conversation_id} without an ended-conversation "
                    f"cache entry: snapshot failed ({type(e).__name__}: {e!s})"
                )
                ended_cache = None

        with self._stack_lock:
            stack = list(self.conversation_stacks.get(conversation_id) or [])
        frame_ids = [f.conversation_id for f in reversed(stack)] or [conversation_id]
        ended: set[str] = set()
        unstacked_error: Exception | None = None
        for frame_id in frame_ids:
            try:
                self.fsm_manager.end_conversation(frame_id)
                ended.add(frame_id)
            except Exception as e:
                if self._frame_instance_present(frame_id):
                    self._drop_ended_frames(conversation_id, ended)
                    raise
                ended.add(frame_id)
                if not stack:
                    unstacked_error = e
                else:
                    logger.warning(f"Error ending FSM {frame_id}: {e!s}")

        # DECISION plan-2026-09-22T080837-8b258a25/D-003
        # The cache write and FIFO eviction share the bookkeeping drop's hold.
        # Do NOT move them after this block: a reader in between saw neither
        # the live conversation nor its cache entry, and two ends evicting at
        # once popped the same key (KeyError). Only dict ops here: never call
        # FSMManager under _stack_lock (the snapshot was taken above). D-003.
        with self._stack_lock:
            self.conversation_stacks.pop(conversation_id, None)
            self._last_accessed.pop(conversation_id, None)
            if ended_cache is not None:
                self._ended_conversations[conversation_id] = ended_cache
                if len(self._ended_conversations) > self._MAX_ENDED_CACHE:
                    self._ended_conversations.pop(next(iter(self._ended_conversations)))
        if unstacked_error is not None:
            raise unstacked_error

        # D-011: pushed sub-FSM defs now live in _temp_fsm_definitions for the
        # frame's lifetime (the post-push pop was removed). This conversation's
        # frames are already out of conversation_stacks, so release any temp def
        # no OTHER live conversation still references.
        self._release_unreferenced_temp_definitions()

    def list_active_conversations(self) -> list[str]:
        """List all active conversation IDs."""
        with self._stack_lock:
            return list(self.conversation_stacks)

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
