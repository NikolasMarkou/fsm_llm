"""``API.fork`` and ``API.resume`` (``ForkMixin``): a new conversation tree from the state of an
old one at the end of its Nth turn, and the SAME tree continued after a crash or a close, both
read from the event log (plan 5a6a1f80, steps 4 and 5, roadmap row 1.4).

Layer ``engine``. The log is the only source; nothing is read from a live conversation. ``fork``
takes no ``conv_lock`` and no writer lock and never writes the source log. ``resume`` takes the
writer lock FIRST and writes the log it continues. Both rebuild the stack through the ONE restore
body (``SessionsMixin._restore_conversation``) and ``_add_restored_frame``: ``resume`` is not a
second restore path.

Interface contract:
    - ``fork(conversation_id, *, at_turn) -> (new_conversation_id, SessionState)``: *at_turn* counts
      the ``turn/end`` events of every frame in log order (one ``converse`` or ``advance`` call is
      one turn, a rolled-back turn counts, the greeting is not a turn); ``0`` is the state before
      the first turn. The new conversation has a minted id, the state, history and summary of the
      root and the whole stack the source had at the cut. The returned ``SessionState`` is the
      folded root (``stack_depth`` is the real depth). Needs an event log (``FSMError`` otherwise).
    - Failure: ``FSMError`` for an id with no log or no state at the cut; ``ValueError`` for an
      ``at_turn`` that is not an int of 0 or more, is past the last finished turn or names a turn
      that has not finished; ``LogCorruptError`` for a frame whose definition the log cannot give
      back (unrecorded, deleted, tampered, not loadable, or with an ``fsm_id`` that is not the one
      logged). Any failure after the first conversation started ends everything built (as
      ``restore_session``) and releases every temporary definition; nothing stays registered.
    - The fork's own log is self-sufficient: a ``fork`` record with the root state and, per frame
      above the root, a ``restored`` ``stack/push`` (D-006 of the plan), so a fork of a fork and a
      ``restore_session`` of it need no help from the source.
    - ``resume(conversation_id) -> (conversation_id, SessionState)``: the SAME id and the SAME log,
      the whole stack rebuilt with its own ids (root and frames keep them). Order: the id must not
      be live in this ``API``; ``EventLog.open_existing`` takes the writer lock BEFORE anything is
      read (an unknown id raises ``FSMError`` and creates no file and no ``.lock``; another
      process or ``EventLog`` holding the lock, or another ``API`` of the same ``EventLog``, is
      ``LogLockedError``); the whole log is folded (a torn tail is cut by the writer, an open turn
      contributes its pre-turn state and is never closed by a ``turn/end`` of ours); then a
      ``context/record`` ``source: "resume"`` with ``resumed_from: {seq}`` and, for each tool call
      the log could not resolve, a ``tool/result`` ``status: "unknown"``. Those calls are listed
      ONCE in ``state.metadata["unknown_tool_calls"]`` and are never re-run; the next resume
      reports none. A conversation in a terminal state resumes as an ended one
      (``has_conversation_ended``); an ended-conversation cache entry of the id is dropped. Needs
      an event log. Failure: ``FSMError`` (no log, no state, live id), ``LogLockedError``,
      ``LogCorruptError`` as ``fork``; any failure after the lock releases the writer and ends
      what was built. A budget is not persisted: the resumed tree starts with a fresh one (D-011).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from fsm_llm.constants import DEFAULT_COMPLETION_MESSAGES_KEY
from fsm_llm.core.definitions import FSMDefinition, FSMError
from fsm_llm.core.redaction import REDACTED
from fsm_llm.engine.events import declared_internal_keys
from fsm_llm.engine.session import SessionState, session_state_from_projection
from fsm_llm.logging import logger
from fsm_llm.runtime import EventLog, LogCorruptError
from fsm_llm.runtime.stores import FrameProjection, fold_stack, turn_cut

from .events import record_forked, record_restored_frame, record_resumed
from .helpers import _require_classifier_interface
from .stack import FSMStackFrame
from .surface import _ApiSurface


@dataclass(frozen=True)
class _Prepared:
    """A frame of the log made ready to rebuild: its definition loaded and checked, and only the
    entries the trust rule lets through."""

    frame: FrameProjection
    definition: FSMDefinition
    fsm_id: str
    internal: dict[str, Any]
    return_context: dict[str, Any]


def _frame_internal(
    definition: FSMDefinition, carried: Mapping[str, Any]
) -> dict[str, Any]:
    """The internal context entries a rebuilt frame above the root may be seeded with."""
    # DECISION plan-2026-11-18T071726-5a6a1f80/D-009
    # A frame above the root is rebuilt from a definition that came out of the LOG, which is a
    # file and not trusted. Its own declarations are therefore NOT an allow-list: a forged
    # definition can declare `messages_key="_approval_granted"` (or the same as
    # `response_format_key`) and `declared_internal_keys(definition)` would then admit a forged
    # driver grant. Only the DEFAULT transcript name `_completion_messages` is restored, and only
    # when the definition declares it; no `response_format_key` value is restored at all. Do NOT
    # seed `declared_internal_keys(definition)`, do NOT widen this to a deny-list of known grants
    # (the grant names belong to agents, and the list would go stale), and do NOT read the keys
    # from a delta or a blob of the root here (the root uses the CALLER's own definition).
    # See decisions.md D-009.
    key = DEFAULT_COMPLETION_MESSAGES_KEY
    if key not in declared_internal_keys(definition) or key not in carried:
        return {}
    return {key: carried[key]}


def _prepare(api: _ApiSurface, log: EventLog, frame: FrameProjection) -> _Prepared:
    """Load and check the definition of *frame* BEFORE anything is started (see the contract)."""
    where = f"the frame of child '{frame.child_conversation_id}'"
    try:
        definition, fsm_id = api.process_fsm_definition(
            log.get_json(frame.definition_ref)
        )
    except ValueError:  # the loader's own message echoes the blob: say what failed only
        raise LogCorruptError(
            f"{where} has a definition that is not a loadable FSM"
        ) from None
    if fsm_id != frame.fsm_id:
        raise LogCorruptError(
            f"{where} has a definition whose id is not the recorded one (the log or its "
            "blob was altered)"
        )
    _require_classifier_interface(definition, api.llm_interface)
    kept = {k: v for k, v in frame.return_context.items() if v != REDACTED}
    if len(kept) != len(frame.return_context):
        logger.warning(
            f"fork: dropped {len(frame.return_context) - len(kept)} redacted entr(ies) "
            f"of the return context of {where}: a placeholder must not merge into the parent"
        )
    return _Prepared(
        frame=frame,
        definition=definition,
        fsm_id=fsm_id,
        internal=_frame_internal(definition, frame.state.completion_context),
        return_context=kept,
    )


class ForkMixin(_ApiSurface):
    """Fork a conversation of this ``API``'s event log into a new tree."""

    def fork(self, conversation_id: str, *, at_turn: int) -> tuple[str, SessionState]:
        """Start a NEW conversation from *conversation_id* as it stood at the end of turn *at_turn*.

        See the module docstring for the contract. The source may be live (this API or another
        process): only committed lines are read, through a separate descriptor.

        Args:
            conversation_id: Root id of the source tree (its log must exist).
            at_turn: The turn to fork at; ``0`` is the state before the first turn.

        Returns:
            ``(new_conversation_id, state)``, as ``restore_session``.

        Raises:
            FSMError: no event log, no log for the id, or no state at that point.
            ValueError: *at_turn* is invalid or not a finished turn.
            LogCorruptError: a damaged log or a frame the log cannot give back.
        """
        log = self._event_log
        if log is None:
            raise FSMError("fork needs an event log: build the API with event_log=")
        # Each read is its own generator over its own descriptor, closed when it is dropped.
        if next(log.read(conversation_id), None) is None:
            raise FSMError(f"no event log for conversation '{conversation_id}'")
        cut = turn_cut(log.read(conversation_id), at_turn)
        projection = fold_stack(log.query(conversation_id, until_seq=cut), log.get_json)
        if projection is None:
            raise FSMError(
                f"conversation '{conversation_id}' has no recorded state at turn {at_turn}"
            )
        prepared = [_prepare(self, log, frame) for frame in projection.frames]
        state = session_state_from_projection(
            projection.root, stack_depth=projection.depth
        )
        if state.fsm_id != self.fsm_id:
            logger.warning(
                f"fork: '{conversation_id}' was recorded under fsm_id='{state.fsm_id}' but "
                f"this API instance is fsm_id='{self.fsm_id}': forking anyway"
            )
        lineage = {"conversation_id": conversation_id, "seq": cut, "turn": at_turn}

        def record(new_id: str) -> None:
            record_forked(self, new_id, lineage)
            for frame in prepared:
                self._add_restored_frame(new_id, frame)

        return self._restore_conversation(state, record), state

    def resume(self, conversation_id: str) -> tuple[str, SessionState]:
        """Continue *conversation_id* on its own log after a crash or a close.

        See the module docstring for the contract.

        Args:
            conversation_id: Root id of a tree whose log exists and is not held by a writer.

        Returns:
            ``(conversation_id, state)``: the same id and the folded root; ``state.metadata
            ["unknown_tool_calls"]`` lists the in-flight tool calls, once.

        Raises:
            FSMError: no event log, no log for the id, no state in it, or the id is live here.
            LogLockedError: another writer holds the log.
            LogCorruptError: a damaged log or a frame the log cannot give back.
        """
        log = self._event_log
        if log is None:
            raise FSMError("resume needs an event log: build the API with event_log=")
        with self._stack_lock:
            live = conversation_id in self.conversation_stacks
        if live:
            raise FSMError(f"conversation '{conversation_id}' is live in this API")
        try:
            log.open_existing(conversation_id)  # the lock comes BEFORE any read
        except FileNotFoundError:
            raise FSMError(
                f"no event log for conversation '{conversation_id}'"
            ) from None
        resumed = False
        try:
            projection = fold_stack(log.read(conversation_id), log.get_json)
            if projection is None:
                raise FSMError(
                    f"conversation '{conversation_id}' has no recorded state"
                )
            prepared = [_prepare(self, log, frame) for frame in projection.frames]
            state = session_state_from_projection(
                projection.root, stack_depth=projection.depth
            )
            if state.fsm_id != self.fsm_id:
                logger.warning(
                    f"resume: '{conversation_id}' was recorded under fsm_id='{state.fsm_id}' "
                    f"but this API instance is fsm_id='{self.fsm_id}': resuming anyway"
                )
            lineage = {"seq": projection.root.last_seq}
            unknown = projection.root.unknown_tool_calls

            def record(root_id: str) -> None:
                for frame in prepared:
                    self._add_restored_frame(root_id, frame, keep_id=True)
                with self._stack_lock:
                    top = self.conversation_stacks[root_id]
                    top_id, top_frame = top[-1].conversation_id, len(top) - 1
                record_resumed(self, root_id, top_id, top_frame, lineage, unknown)

            self._restore_conversation(state, record, conversation_id=conversation_id)
            resumed = True
        finally:
            # a no-op when the restore teardown already ended the conversation and freed it
            if not resumed:
                log.release(conversation_id)
        with self._stack_lock:
            self._ended_conversations.pop(conversation_id, None)
        return conversation_id, state

    def _add_restored_frame(
        self, root_id: str, prepared: _Prepared, *, keep_id: bool = False
    ) -> None:
        """Rebuild one frame on top of the stack of *root_id*, exactly as ``push_fsm`` adds one.

        *keep_id* (``resume``) starts the frame under the id the log gave it and writes no push
        record (the log has it); ``fork`` mints an id and records a ``restored`` push.

        The temporary definition and the in-flight id are registered BEFORE the start, the frame
        is appended and the id released from "pending" in ONE ``_stack_lock`` hold, and a failure
        before the append releases by reference (``_rollback_push``). ``_stack_lock`` is never
        held across a manager call. A failure after the append is the caller's: ending the root
        ends every frame in the stack.
        """
        state, fsm_id = prepared.frame.state, prepared.fsm_id
        manager = self.fsm_manager
        child_id: str | None = None
        added = False
        try:
            with self._stack_lock:
                self._temp_fsm_definitions[fsm_id] = prepared.definition
                self._pending_push_ids.add(fsm_id)
            kept = (
                {"conversation_id": prepared.frame.child_conversation_id}
                if keep_id
                else {}
            )
            child_id, _ = manager.start_conversation(
                fsm_id,
                initial_context=state.context_data,
                suppress_start=True,
                **kept,
            )
            manager.seed_restored_conversation(
                child_id,
                summary=state.conversation_summary or None,
                history=state.conversation_history,
                provenance=state.provenance or None,
                working_memory=self._restored_working_memory(state.working_memory),
                internal_context=prepared.internal or None,
            )
            manager.set_conversation_state(child_id, state.current_state)
            with self._stack_lock:
                if root_id not in self.conversation_stacks:
                    raise FSMError(f"Conversation not found: {root_id}")
                stack = self.conversation_stacks[root_id]
                stack.append(
                    FSMStackFrame(
                        fsm_definition=prepared.definition,
                        conversation_id=child_id,
                        return_context=prepared.return_context,
                        shared_context_keys=prepared.frame.shared_context_keys,
                        preserve_history=prepared.frame.preserve_history,
                        fsm_id=fsm_id,
                    )
                )
                self._pending_push_ids.discard(fsm_id)
                depth = len(stack)
            added = True
        finally:
            if not added:
                self._rollback_push(fsm_id, child_id)
        if keep_id:
            return
        record_restored_frame(
            self,
            root_id,
            child_id,
            fsm_id,
            prepared.definition,
            depth,
            return_context=prepared.return_context,
            shared_context_keys=prepared.frame.shared_context_keys,
            preserve_history=prepared.frame.preserve_history,
        )
