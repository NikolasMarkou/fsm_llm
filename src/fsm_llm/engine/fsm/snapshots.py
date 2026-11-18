"""Reads, snapshots and the restore seed of ``FSMManager`` (``SnapshotsMixin``).

Layer ``engine`` (plan d454fd62, step 78): split out of the old ``fsm_llm.fsm``; the
method bodies are unchanged. ``_read_under_lock`` and ``_acquire_or_refuse`` are the
one place a read takes ``conv_lock`` (the anchors below say why ``_lock`` is released
first).
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from fsm_llm.constants import (
    END_CONVERSATION_LOCK_TIMEOUT_SECONDS,
    PROVENANCE_METADATA_KEY,
)
from fsm_llm.core.definitions import (
    ConversationBusyError,
    FSMError,
    FSMInstance,
    StateNotFoundError,
)
from fsm_llm.logging import logger

from .helpers import (
    _conversation_entry,
    _strip_internal_mapping,
    with_conversation_context,
)

if TYPE_CHECKING:
    from fsm_llm.core.definitions import FSMDefinition, State


# Return type of a read snapshot (see FSMManager._read_under_lock).
_SnapshotT = TypeVar("_SnapshotT")


def snapshot_from_instance(inst: FSMInstance) -> dict[str, Any]:
    """The fields of ``SnapshotsMixin.get_conversation_snapshot`` read from ONE instance.

    Interface contract: the caller holds the instance's ``conv_lock`` (or owns the instance),
    because every field is read in this one call (the atomicity of the snapshot is the
    caller's hold, see ``get_conversation_snapshot``). Fast in-memory work only: no
    definition lookup, no manager lock. Returns the dict that method documents. Never
    raises for content. Shared with the event log's state blob (``engine/events.py``),
    which reads the same fields so a logged state and a saved session cannot drift.
    """
    wm_obj = inst.context.working_memory
    working_memory: dict[str, dict[str, Any]] | None = None
    hidden_buffers: list[str] = []
    # WorkingMemory guards its own buffers with its OWN internal
    # lock (memory.py, independent of `_lock`/`conv_lock`), so
    # calling `to_dict()` here -- already inside `conv_lock` -- does
    # not nest under/invert either FSMManager lock. See plan.md
    # Assumptions.
    if wm_obj is not None and hasattr(wm_obj, "to_dict"):
        working_memory = wm_obj.to_dict()
        hidden_buffers = sorted(getattr(wm_obj, "hidden_buffers", frozenset()))
    return {
        "current_state": inst.current_state,
        "context_data": _strip_internal_mapping(inst.context.data),
        "conversation_history": inst.context.conversation.get_recent(),
        "working_memory": working_memory,
        "hidden_buffers": hidden_buffers,
        "provenance": dict(inst.context.metadata.get(PROVENANCE_METADATA_KEY) or {}),
        # DECISION plan-2026-09-21T203800-8a03483a/D-007 (A6): the
        # summary is read in this SAME hold as the history it
        # complements (a trim moves exchanges from one to the other).
        # Do NOT read it through a separate call in save_session.
        "conversation_summary": inst.context.conversation.summary,
    }


class SnapshotsMixin:
    """Reads, snapshots and the restore seed of ``FSMManager``, all under ``conv_lock``.

    ``_lock`` is taken only for the dict lookup and released before ``conv_lock`` is
    acquired (the ``_lock`` then ``conv_lock`` order, never the reverse).
    """

    if TYPE_CHECKING:
        # Provided by ``FSMManager`` (declared for mypy only).
        _lock: threading.Lock
        _conversation_locks: dict[str, threading.RLock]
        instances: dict[str, FSMInstance]

        def get_fsm_definition(self, fsm_id: str) -> FSMDefinition: ...

        def resolve_state_definition(
            self, instance: FSMInstance, conversation_id: str | None = None
        ) -> State: ...

    @staticmethod
    def _acquire_or_refuse(conv_lock: Any, conversation_id: str) -> None:
        """Acquire ``conv_lock`` for an end-of-conversation step, bounded.

        Interface contract:
          - Parameters: the conversation's lock and its id (for the message).
          - Returns: None with the lock HELD; the caller must release it.
          - Failure: if the lock is not acquired within
            ``END_CONVERSATION_LOCK_TIMEOUT_SECONDS`` (read at call time), logs
            ERROR and raises ``ConversationBusyError`` (an ``FSMError``, "...
            still processing a turn ..."); the lock is not held and nothing
            has been changed.
        """
        if not conv_lock.acquire(timeout=END_CONVERSATION_LOCK_TIMEOUT_SECONDS):
            logger.error(
                f"Timed out after {END_CONVERSATION_LOCK_TIMEOUT_SECONDS}s waiting "
                f"for conversation lock on {conversation_id}; a turn is still "
                "running, conversation NOT ended"
            )
            raise ConversationBusyError(
                f"Conversation {conversation_id} is still processing a turn; "
                "end_conversation refused (retry after the turn completes)",
                conversation_id=conversation_id,
            )

    def _read_under_lock(
        self,
        conversation_id: str,
        snapshot_fn: Callable[[FSMInstance], _SnapshotT],
        *,
        bounded: bool = False,
    ) -> _SnapshotT:
        """Run ``snapshot_fn`` against a conversation under its ``conv_lock``.

        Interface contract:
          - Parameters: ``conversation_id`` (must be an active conversation) and
            ``snapshot_fn``, a callable taking the ``FSMInstance`` and returning
            a point-in-time in-memory snapshot. ``bounded=True`` waits for the
            lock at most ``END_CONVERSATION_LOCK_TIMEOUT_SECONDS`` (the end
            path); the default waits as long as the running turn takes.
          - Returns: whatever ``snapshot_fn`` returns.
          - Failure: raises ``FSMError`` if the conversation is unknown, or if
            its instance is present but its per-conversation lock is missing
            (a broken create-together / remove-together invariant — L11), or
            (``bounded`` only) if the lock is not acquired in time.

        HARD RULE: ``snapshot_fn`` runs while holding ``conv_lock`` and MUST do
        only fast in-memory work. It MUST NOT call ``resolve_state_definition`` /
        ``get_fsm_definition`` (a cache miss runs ``fsm_loader``, arbitrary and
        possibly slow code, and would hold ``conv_lock`` across it), nor anything
        that takes ``self._lock`` (a lock-order inversion). Resolve state OUTSIDE
        this call, after it returns. See decisions.md D-005.
        """
        # DECISION plan-2026-07-21T045419-9925aa3a/D-005
        # The write path acquires conv_lock while holding _lock (non-blocking), so
        # blocking on _lock while holding conv_lock would close a circular wait ->
        # deadlock. Therefore: take _lock ONLY for the dict lookup, RELEASE it, THEN
        # acquire conv_lock (blocking) for the snapshot. Do NOT move any
        # resolve_state_definition / get_fsm_definition call into snapshot_fn: it no
        # longer takes _lock (leaf _fsm_cache_lock, D-036 of 8b258a25), but a cache
        # miss runs fsm_loader (arbitrary code) under conv_lock. See D-005.
        with self._lock:
            instance, conv_lock = _conversation_entry(
                self, conversation_id, missing_lock="raise"
            )

        if not bounded:
            with conv_lock:
                return snapshot_fn(instance)
        self._acquire_or_refuse(conv_lock, conversation_id)
        try:
            return snapshot_fn(instance)
        finally:
            conv_lock.release()

    def get_end_snapshot(self, conversation_id: str) -> dict[str, Any]:
        """Read what ``API.end_conversation`` caches, under ONE bounded hold.

        Interface contract:
          - Parameters: ``conversation_id`` (an active conversation).
          - Returns: ``{"data": <get_conversation_data view>, "state": str,
            "history": <get_conversation_history view>}``, one consistent
            point-in-time read.
          - Failure: ``FSMError`` for an unknown conversation, or when a turn
            holds the lock past ``END_CONVERSATION_LOCK_TIMEOUT_SECONDS`` (the
            same refusal ``end_conversation`` raises).
        """
        # DECISION plan-2026-09-21T203800-8a03483a/D-050
        # The end path's cache read is BOUNDED like the end itself. Do NOT
        # switch API.end_conversation back to get_conversation_data/state/
        # history: those wait for the turn indefinitely, so the documented
        # 30 s refusal was never what an API caller saw (review W2). Do NOT
        # bound the ordinary getters: a read during a long turn must wait.
        return self._read_under_lock(
            conversation_id,
            lambda inst: {
                "data": _strip_internal_mapping(inst.context.data),
                "state": inst.current_state,
                "history": inst.context.conversation.get_recent(),
            },
            bounded=True,
        )

    @with_conversation_context
    def has_conversation_ended(self, conversation_id: str, log: Any = None) -> bool:
        """Check if conversation has reached terminal state."""
        instance = self._read_under_lock(conversation_id, lambda inst: inst)

        # State resolution may run fsm_loader on a cache miss, so it happens
        # OUTSIDE conv_lock (the _read_under_lock snapshot has already returned).
        # See D-005.
        current_state = self.resolve_state_definition(instance, conversation_id)

        is_ended = not current_state.transitions
        if is_ended:
            log.info(f"Conversation reached terminal state: {instance.current_state}")

        return is_ended

    def get_conversation_data(self, conversation_id: str) -> dict[str, Any]:
        """Get collected context data (internal metadata keys filtered out).

        The internal-prefix filter is applied at every nesting level -- inside
        nested dicts and inside dicts nested in lists/tuples -- so
        ``{"user": {"_internal_note": "x"}}`` is filtered like its flat
        equivalent. Matched keys are DROPPED (not redacted). Recursion is
        bounded at ``MAX_CONTEXT_FILTER_DEPTH``; a container deeper than that is
        dropped rather than returned unfiltered. See D-010.
        """
        # Run the filter UNDER conv_lock: it rebuilds the whole structure, so a
        # single pass yields an independent, fully-filtered snapshot that a
        # concurrent turn can neither tear nor resize mid-iteration (C1).
        return self._read_under_lock(
            conversation_id,
            lambda inst: _strip_internal_mapping(inst.context.data),
        )

    def get_conversation_state(self, conversation_id: str) -> str:
        """Get current state of conversation."""
        return self._read_under_lock(conversation_id, lambda inst: inst.current_state)

    def set_conversation_state(self, conversation_id: str, state_name: str) -> None:
        """Set an existing conversation's current state to a validated FSM state.

        Interface contract:
          - Parameters: `conversation_id` (an existing conversation), `state_name`
            (a state id that MUST exist in that conversation's FSM definition).
          - Returns: None; mutates `instance.current_state` in place.
          - Failure: raises `FSMError` if the conversation is unknown OR
            `state_name` is not a state in its FSM definition (never silently
            mis-sets).

        Used by session restore to reinstate the saved `current_state` after the
        conversation has been (re)created at `initial_state`.
        """
        # DECISION plan-2026-07-21T045419-9925aa3a/D-004
        # A validated, locked setter belongs in the manager (which owns FSM-def
        # access), NOT as a raw `instance.current_state = ...` poke in engine/api/:
        # the FSM-def membership check is the invariant that turns a corrupted or
        # foreign session naming a nonexistent state into a loud FSMError instead
        # of silent corruption. Do NOT inline this back into restore_session, and
        # do NOT invert the lock order: `_lock` MUST be released before resolving
        # the def / acquiring `conv_lock`. `get_fsm_definition` takes only its leaf
        # `_fsm_cache_lock` now (D-036 of 8b258a25), but a cache miss runs
        # `fsm_loader` (arbitrary code), which must not run under `_lock`.
        # Acquiring `conv_lock` only AFTER `_lock` is released preserves the
        # canonical `_lock -> conv_lock` order. See decisions.md D-004.
        with self._lock:
            instance, conv_lock = _conversation_entry(
                self, conversation_id, missing_lock="unlocked"
            )

        fsm_def = self.get_fsm_definition(instance.fsm_id)
        if state_name not in fsm_def.states:
            raise FSMError(f"'{state_name}' is not a state in FSM '{instance.fsm_id}'")

        if conv_lock is not None:
            with conv_lock:
                instance.current_state = state_name
        else:
            instance.current_state = state_name

    def get_conversation_history(self, conversation_id: str) -> list[dict[str, str]]:
        """Get conversation history."""
        # get_recent() returns a NEW list; taking it under conv_lock keeps it
        # consistent with a concurrent turn's exchanges mutation (C1).
        return self._read_under_lock(
            conversation_id,
            lambda inst: inst.context.conversation.get_recent(),
        )

    # DECISION plan-2026-09-20T114608-a8e47b88/D-005
    def get_conversation_snapshot(self, conversation_id: str) -> dict[str, Any]:
        """Get one point-in-time-consistent read of everything ``save_session``
        needs, under a SINGLE ``_read_under_lock`` hold.

        Interface contract:
          - Parameters: ``conversation_id`` (an active conversation).
          - Returns: a dict with keys ``current_state`` (str),
            ``context_data`` (internal-key-filtered, see
            ``get_conversation_data``), ``conversation_history`` (see
            ``get_conversation_history``), ``working_memory`` (the buffer
            dict from ``WorkingMemory.to_dict()``, or ``None`` if the
            conversation has no working memory configured),
            ``hidden_buffers`` (sorted list of hidden buffer names, ``[]``
            if none/none configured), ``provenance`` (a shallow copy of
            ``context.metadata[PROVENANCE_METADATA_KEY]``, ``{}`` if absent), and
            ``conversation_summary`` (``Conversation.summary``, str or
            ``None``).
          - Failure: same as ``_read_under_lock`` (``FSMError`` for an
            unknown conversation or a missing per-conversation lock).

        Atomicity guarantee (why this method exists -- G2): every field
        above is read from the SAME ``FSMInstance`` while holding its
        ``conv_lock`` for the entire snapshot, so a concurrent turn can
        neither complete nor partially mutate the instance between two of
        these reads. Composing the equivalent result from separate calls
        (``get_conversation_state`` + ``get_conversation_data`` + ...) each
        acquire and release ``conv_lock`` independently, leaving a gap a
        concurrent turn can land in and produce a torn combination -- e.g.
        pre-transition state paired with post-transition data. See
        decisions.md D-005 (this plan) / `_read_under_lock`'s own D-005
        (plan-2026-07-21T045419-9925aa3a) for the underlying lock-order
        invariant this method reuses unchanged.

        Shallow-copy boundary (review-iter-1.md NOTE 7): this guarantee
        covers the READ only, not deep isolation of every field afterward --
        ``conversation_history`` (``Conversation.get_recent()``) and
        ``working_memory`` (``WorkingMemory.to_dict()``) return a new outer
        container whose NESTED mutable values still alias the live
        instance's objects, so a caller that holds onto and later mutates
        those nested values can still observe/cause a change after
        ``conv_lock`` is released. Only ``context_data`` is genuinely
        rebuilt end-to-end (via ``_strip_internal_mapping``). In practice
        ``save_session`` only reads these fields once and serializes them
        immediately, so this is a latent caveat for any FUTURE caller, not a
        known bug in the current one.

        Deliberately EXCLUDES stack depth: ``API.get_stack_depth`` reads
        ``API.conversation_stacks`` under a *different* lock
        (``API._stack_lock``), a structural property (which FSMs are
        pushed) that does not change within a single turn's Pass-1/Pass-2
        commit, so folding it in here would not close any tear and would
        only pull an unrelated lock into this hold. It stays a separate,
        non-atomic call, by design -- see plan.md's explicit scope note.
        """
        return self._read_under_lock(conversation_id, snapshot_from_instance)

    def seed_restored_conversation(
        self,
        conversation_id: str,
        *,
        summary: str | None,
        history: list[dict[str, str]],
        provenance: dict[str, Any] | None,
        working_memory: Any,
        internal_context: Mapping[str, Any] | None = None,
    ) -> None:
        """Apply a restored session's seeds under ONE ``conv_lock`` hold.

        Interface contract:
          - Parameters: ``conversation_id`` (an active conversation, freshly
            started by ``API.restore_session``); ``summary`` (set when not
            ``None``); ``history`` (``{"user": ...}`` / ``{"system": ...}``
            exchanges, appended in order); ``provenance`` (a shallow copy is
            stored under ``PROVENANCE_METADATA_KEY`` when not ``None``);
            ``working_memory`` (a built ``WorkingMemory``, assigned when not
            ``None``); ``internal_context`` (internal-prefixed context entries
            written as they are when not ``None``: the CALLER vets the keys, only
            ``API.restore_session`` calls this and passes just the names the
            running definition's completion states declare, plan 21e6f00d D-009).
            ``None`` means "keep the fresh conversation's value".
          - Returns: None; mutates the instance in place.
          - Failure: same as ``_read_under_lock`` (``FSMError`` for an unknown
            conversation or a missing per-conversation lock).

        The restore-side twin of ``get_conversation_snapshot``: it reuses
        ``_read_under_lock`` (``_lock`` for the lookup only, released before
        ``conv_lock``), so the seed body does in-memory work only.
        """

        def _seed(inst: FSMInstance) -> None:
            conversation = inst.context.conversation
            # DECISION plan-2026-09-21T203800-8a03483a/D-007 (A6, order per
            # D-029): seed the saved summary BEFORE replaying history, not
            # after. If this API keeps fewer exchanges than the saver did, the
            # replay trims and `_append_to_summary` appends the trimmed
            # exchanges after the saved digest, in order; setting it after the
            # replay would overwrite those.
            if summary is not None:
                conversation.summary = summary
            for exchange in history:
                if "user" in exchange:
                    conversation.add_user_message(exchange["user"])
                if "system" in exchange:
                    conversation.add_system_message(exchange["system"])
            if provenance is not None:
                inst.context.metadata[PROVENANCE_METADATA_KEY] = dict(provenance)
            if working_memory is not None:
                inst.context.working_memory = working_memory
            if internal_context:
                inst.context.update(dict(internal_context))

        self._read_under_lock(conversation_id, _seed)

    def has_instance(self, conversation_id: str) -> bool:
        """True if an FSM instance is registered for ``conversation_id``
        (read under ``_lock``; never raises)."""
        with self._lock:
            return conversation_id in self.instances

    def copy_raw_context_data(self, conversation_id: str) -> dict[str, Any] | None:
        """A shallow copy of the UNFILTERED ``context.data`` (internal keys
        included), taken under ``_lock`` only; ``None`` for an unknown
        conversation. For explicit key requests that bypass the internal-key
        filter (``API`` pop-context merge); never returned to a caller as-is.

        An unlocked point-in-time copy, on purpose: ``dict(data)`` of one dict is a
        single C call (no resize error, no torn dict), but a copy can land between two
        writes of a running turn. It is NOT taken through ``_read_under_lock``: a timed
        handler thread runs while the turn thread holds ``conv_lock`` and joins it, so
        a handler reading through this API with a blocking ``conv_lock`` would deadlock
        until its timeout."""
        with self._lock:
            instance = self.instances.get(conversation_id)
            return None if instance is None else dict(instance.context.data)

    def get_complete_conversation(self, conversation_id: str) -> dict[str, Any]:
        """Get complete conversation data for analysis.

        ``metadata`` is a shallow copy of ``context.metadata``. Besides
        ``_pipeline_extracted`` (provenance digests) it carries
        ``classification_results`` ({field_name: latest full classification
        record}) and, for the latest turn only, ``transition_classification``
        (the ambiguous-transition classifier record). Both are JSON-native and
        replaced, never mutated in place, so a returned snapshot stays stable.
        """
        # Snapshot every mutable container (+ fsm_id/state_id and the debug-model
        # refs) in ONE pass under conv_lock, so collected_data / history / metadata
        # AND the current_state sub-dict are internally consistent and cannot be
        # torn by a concurrent turn (C1). The
        # last_* refs are replaced by assignment on the write path (never mutated
        # in place), so calling model_dump() on them OUTSIDE the lock is safe and
        # keeps the lock-hold short.
        # A positional tuple keeps each element's static type precise (a mixed
        # dict would widen every value to a union).
        (
            fsm_id,
            state_id,
            collected_data,
            history,
            metadata,
            last_extraction,
            last_transition,
            last_response,
        ) = self._read_under_lock(
            conversation_id,
            lambda inst: (
                inst.fsm_id,
                inst.current_state,
                dict(inst.context.data),
                inst.context.conversation.get_recent(),
                dict(inst.context.metadata),
                inst.last_extraction_response,
                inst.last_transition_decision,
                inst.last_response_generation,
            ),
        )

        # State resolution may run fsm_loader on a cache miss, so it happens
        # OUTSIDE conv_lock (the snapshot has already returned). See D-005. Resolve the State from the
        # fsm_id + state_id CAPTURED in the snapshot -- NOT via
        # resolve_state_definition(instance), which re-reads inst.current_state fresh: a
        # concurrent transition between the snapshot and that re-read would yield a
        # torn current_state sub-dict (id from the snapshot, description/purpose/
        # is_terminal from a newer state). Keying off the captured id keeps the
        # whole sub-dict internally consistent with the snapshot.
        fsm_def = self.get_fsm_definition(fsm_id)
        if state_id not in fsm_def.states:
            raise StateNotFoundError(f"State '{state_id}' not found in FSM '{fsm_id}'")
        current_state = fsm_def.states[state_id]

        return {
            "id": conversation_id,
            "fsm_id": fsm_id,
            "current_state": {
                "id": state_id,
                "description": current_state.description,
                "purpose": current_state.purpose,
                "is_terminal": not current_state.transitions,
            },
            "collected_data": collected_data,
            "conversation_history": history,
            "metadata": metadata,
            "last_extraction_response": last_extraction.model_dump()
            if last_extraction
            else None,
            "last_transition_decision": last_transition.model_dump()
            if last_transition
            else None,
            "last_response_generation": last_response.model_dump()
            if last_response
            else None,
        }
