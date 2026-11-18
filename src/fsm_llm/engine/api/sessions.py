"""Session persistence of ``API`` (``SessionsMixin``): ``save_session``, ``load_session`` and
``restore_session`` with the ``WorkingMemory`` rebuild.

Layer ``engine`` (plan d454fd62, step 81): split out of the old ``fsm_llm.api``; the method bodies
are unchanged. ``restore_session`` takes no ``conv_lock`` (the anchors below name the seed call
that does).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fsm_llm.core.definitions import FSMError
from fsm_llm.engine.events import declared_internal_keys
from fsm_llm.engine.session import (
    METADATA_COMPLETION_CONTEXT,
    METADATA_LAST_SEQ,
    METADATA_PROVENANCE,
    SessionState,
)
from fsm_llm.logging import logger
from fsm_llm.memory import WorkingMemory
from fsm_llm.runtime.stores.session import public_context

from .events import record_restored
from .surface import _ApiSurface


class SessionsMixin(_ApiSurface):
    """Save, load and restore one ``API``'s conversations through its session store."""

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

        # Validation and the D-014 idle refresh go through the top-of-stack
        # resolver; the state itself is read from the ROOT frame.
        self._get_current_fsm_conversation_id(conversation_id)
        # DECISION plan-2026-09-19T175721-21cd7f8e/D-022
        # Save the ROOT frame, not the top of the stack. A stacked save used to
        # write the CHILD's state and data under the root id, so restore_session
        # (which rebuilds only the root FSM and ignores stack_depth) hit a state
        # that does not exist in the root definition and auto-save then
        # overwrote the last good session. Do NOT "fix" this by saving the child
        # under the child id: restore has no path that rebuilds stacks.
        with self._stack_lock:
            current_fsm_id = self.conversation_stacks[conversation_id][
                0
            ].conversation_id

        # DECISION plan-2026-09-20T114608-a8e47b88/D-005
        # ONE atomic read (state + data + history + working memory +
        # provenance) instead of 4 separately-locked reads: a concurrent turn
        # used to be able to land in the gap between two of those reads and
        # persist a torn snapshot (e.g. pre-transition state paired with
        # post-transition data -- G2). Do NOT decompose this back into
        # individual get_conversation_state/get_conversation_data/
        # get_conversation_history/working-memory-reach-in calls -- that
        # reintroduces the exact race this closes. See decisions.md D-005 and
        # `FSMManager.get_conversation_snapshot`'s docstring.
        snapshot = self.fsm_manager.get_conversation_snapshot(current_fsm_id)

        state = SessionState(
            conversation_id=conversation_id,
            fsm_id=self.fsm_id,
            current_state=snapshot["current_state"],
            context_data=snapshot["context_data"],
            conversation_history=snapshot["conversation_history"],
            # get_stack_depth is intentionally OUTSIDE the atomic snapshot: it
            # is structural (which FSMs are pushed), not turn-mutated per
            # message, and reads API.conversation_stacks under a different
            # lock (_stack_lock) -- folding it in would not close any tear and
            # would only pull an unrelated lock into the hold. See plan.md's
            # explicit scope note.
            stack_depth=self.get_stack_depth(conversation_id),
            conversation_summary=snapshot["conversation_summary"],
        )

        # H10: the flat context_data does not carry WorkingMemory, so persist it
        # separately. hidden_buffers are carried explicitly so a custom
        # hidden-buffer set survives the round-trip (D-032 downgrade).
        # DECISION plan-2026-09-19T175721-21cd7f8e/D-031: persist the D-015
        # provenance digests so a correction still lands after a restart
        # (RB-05: iteration 2 failed closed and every restored key was
        # frozen with no log). Digests are JSON-native strings. Do NOT
        # re-seed provenance for keys the file never carried: a value the
        # store has no digest for stays handler-seeded (never overwritten).
        if snapshot["provenance"]:
            state.metadata[METADATA_PROVENANCE] = snapshot["provenance"]
        if snapshot["working_memory"] is not None:
            state.working_memory = {
                "buffers": snapshot["working_memory"],
                "hidden_buffers": snapshot["hidden_buffers"],
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
        lossy for non-JSON-native context values, per
        ``context.filtering.redacting_json_default``: an exact stdlib scalar
        (datetime/date/time/timedelta/Decimal/UUID) comes back as its
        ``str()``, while anything else (set, bytes, Path, Enum, a custom
        object) comes back as the placeholder ``"<redacted:TypeName>"``,
        not as its ``str()`` text.

        Note: the ``fsm_id``-mismatch WARNING this method logs (D-011) is
        only observable if the ``fsm_llm`` logger namespace has been
        enabled -- ``logging.py`` calls ``logger.disable("fsm_llm")`` at
        import time, so a caller must have called ``setup_logging()`` /
        ``enable_debug_logging()`` (or otherwise re-enabled the namespace)
        before this WARNING is visible on any sink.

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

        # DECISION plan-2026-09-20T114608-a8e47b88/D-011
        # A saved session's fsm_id legitimately drifts across additive schema
        # upgrades (e.g. handler_only_keys: [] added to model_dump(), see
        # CHANGELOG precedent) -- so this is a WARNING, not a hard-fail. Do
        # NOT raise here: `set_conversation_state` below already guards the
        # case that actually matters (a saved current_state that does not
        # exist in THIS fsm_id's definition), and two definitions can share a
        # state name (e.g. both have "start") without being the same FSM, so
        # a mismatch can restore "successfully" onto semantically different
        # states. This log is the operator-visible signal for that case --
        # but only if the "fsm_llm" logger namespace is enabled (disabled by
        # default at import, see this method's docstring).
        if state.fsm_id != self.fsm_id:
            logger.warning(
                f"restore_session: saved session '{session_id}' was recorded "
                f"under fsm_id='{state.fsm_id}' but this API instance is "
                f"fsm_id='{self.fsm_id}' -- restoring anyway; verify this is "
                "the intended FSM definition."
            )

        conv_id = self._restore_conversation(
            state,
            lambda new_id: record_restored(
                self, new_id, session_id, state.metadata.get(METADATA_LAST_SEQ)
            ),
        )

        return conv_id, state

    def _restore_conversation(
        self,
        state: SessionState,
        record: Callable[[str], None],
        *,
        conversation_id: str | None = None,
    ) -> str:
        """Start a conversation seeded from *state* and return its id: the ONE restore body.

        Interface contract (callers: ``restore_session`` now, ``fork`` and ``resume``
        next, plan 5a6a1f80): *state* is the saved or folded root session; *record*
        takes the new conversation id and writes the lineage record (it runs inside the
        teardown guard, so a raise ends the half-built conversation); *conversation_id*
        keeps an id (``resume``) and is forwarded as ``_conversation_id`` only when
        given. Returns the new id. Any exception after the start ends the conversation
        (END handlers fire on purpose) and re-raises the original; a failed start raises
        as ``start_conversation`` does and leaves nothing behind.
        """
        # Start conversation with saved context. H9: _suppress_start=True skips
        # the START_CONVERSATION handlers and the Pass-2 greeting so a resume
        # does not re-fire start-of-conversation side effects.
        kept_id: dict[str, Any] = (
            {} if conversation_id is None else {"_conversation_id": conversation_id}
        )
        # A snapshot is a file: only its public keys seed the context (`public_context`, the
        # one rule of the fold), never an internal one such as a forged `_approval_granted`
        # (the declared transcript keys come through `_restorable_internal` below). The filter
        # sits HERE, at the restore, and never in `start_conversation`: agents seed
        # `_max_iterations` there on purpose (D-026 of plan 0068b055).
        conv_id, _ = self.start_conversation(
            initial_context=public_context(state.context_data),
            _suppress_start=True,
            **kept_id,
        )

        current_fsm_id = self._get_current_fsm_conversation_id(conv_id)

        # DECISION plan-2026-07-21T072826-e3131cc2/D-003: restore_session must
        # NOT leak a half-registered conversation on partial-setup failure. Once
        # start_conversation(_suppress_start=True) has registered conv_id in
        # conversation_stacks / fsm_manager.instances,
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
        # consistent with engine/fsm/manager.py:_cleanup_after_failed_start (prior plan D-006):
        # any failed conversation setup fires END on teardown. Do NOT suppress END
        # handlers here — that would diverge from the failed-start precedent.
        try:
            # DECISION plan-2026-09-22T080837-8b258a25/D-005
            # All four seeds (summary, history, provenance, H10 WorkingMemory)
            # go through ONE manager call that takes `_lock` only for the
            # lookup, then `conv_lock`. Do NOT take `conv_lock` here, and do NOT
            # reach into `fsm_manager._lock`/`.instances` in a seed: that was
            # the C-NEW-007 inversion (`_lock` taken under `conv_lock`).
            saved_prov = state.metadata.get(METADATA_PROVENANCE)
            self.fsm_manager.seed_restored_conversation(
                current_fsm_id,
                summary=state.conversation_summary or None,
                history=state.conversation_history,
                # D-031: re-seed provenance; an old file has no key -> empty map
                provenance=(
                    saved_prov if isinstance(saved_prov, dict) and saved_prov else None
                ),
                working_memory=self._restored_working_memory(state.working_memory),
                internal_context=self._restorable_internal(state),
            )

            # C3: reinstate the saved current_state AFTER the seeds.
            # set_conversation_state takes _lock then conv_lock (canonical
            # order). It also validates the state against the FSM def,
            # raising FSMError for a corrupted/foreign session.
            self.fsm_manager.set_conversation_state(current_fsm_id, state.current_state)

            # Event log on: the restored state is the first record of the new log (lineage
            # names the old id and the last event the fold read). A write failure raises
            # here, so the teardown below ends the conversation.
            record(conv_id)
        except Exception:
            # Best-effort teardown of the just-created conversation. end_conversation
            # removes conv_id from conversation_stacks and
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

        return conv_id

    def _restorable_internal(self, state: SessionState) -> dict[str, Any] | None:
        """The internal context entries a restore may seed: only the keys the RUNNING
        definition's completion states declare (``declared_internal_keys``).

        Everything else a session carries under ``completion_context`` is refused, above all the
        driver-only grant ``_approval_granted``: a forged log or snapshot must never hand a
        restored conversation an approval (plan 21e6f00d D-009, a RED test pins it).
        """
        saved = state.metadata.get(METADATA_COMPLETION_CONTEXT)
        if not isinstance(saved, dict) or not saved:
            return None
        allowed = declared_internal_keys(self.fsm_definition)
        kept = {name: value for name, value in saved.items() if name in allowed}
        if len(kept) != len(saved):
            logger.warning(
                f"restore_session: refused {len(saved) - len(kept)} undeclared internal "
                "context entr(ies) of the saved session"
            )
        return kept or None

    @staticmethod
    def _restored_working_memory(wm: dict[str, Any] | None) -> Any:
        """Build the H10 ``WorkingMemory`` a saved session carries, or ``None``
        when it carries none (the fresh conversation's value is kept)."""
        if not wm:
            return None
        # Single definition (not duplicated across if/else arms): duplicating
        # the hidden_buffers carry risks fixing one arm and not the other,
        # re-opening the D-032 hidden-buffer downgrade.
        # Review N8 (plan-2026-09-20T165703-0d9c218e): no `buffers` signal (key
        # missing or JSON null) means the DEFAULT buffers; an explicit `{}`
        # keeps meaning zero buffers (the tested `from_dict({})` contract). Do
        # not collapse both into `wm.get("buffers") or {}` again.
        buffers_raw = wm.get("buffers")
        # DECISION plan-2026-09-20T165703-0d9c218e/D-001
        # Same None-vs-empty rule for `hidden_buffers`: a missing or null key
        # passes `None` so `WorkingMemory` applies `DEFAULT_HIDDEN_BUFFERS`
        # ({"metadata"}) and `from_dict` lets the embedded `_hidden_buffers` key
        # decide; an explicit list (including `[]`) is honoured verbatim. Do NOT
        # write `frozenset(wm.get("hidden_buffers") or [])`: that collapsed
        # "absent" into "none hidden" and leaked `metadata` into the aggregate
        # views (review-iter-2 concern 3). See decisions.md D-001.
        hidden_raw = wm.get("hidden_buffers")
        hidden = None if hidden_raw is None else frozenset(hidden_raw)
        if buffers_raw is None:
            return WorkingMemory(hidden_buffers=hidden)
        return WorkingMemory.from_dict(buffers_raw, hidden_buffers=hidden)
