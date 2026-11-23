"""The turn entries of ``API`` (``TurnEntriesMixin``): ``start_conversation``, ``converse``,
``advance`` and the two streams, with the two helpers every entry shares.

Layer ``engine``. A pure move out of ``conversation.py`` (plan f150ec28, step 1.1): that module
was at the size limit and the async entries of the same turns are added here beside their sync
names. Each body is the old one; the only text changes are the type of the ``api`` parameter of
``_auto_save_session`` (the mypy-only ``_ApiSurface``, as in the other mixins) and two em dashes
of a docstring. ``API`` in ``conversation.py`` inherits the mixin, so ``API.converse`` is
``TurnEntriesMixin.converse``.

The module-level helpers stay module-level on purpose (anchor 07ad3f8c/D-021 below), and the
streams stay lazy nested closures (anchor 4c63deac/D-002 below).
"""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator, Iterator
from contextlib import aclosing, contextmanager
from typing import Any

from fsm_llm.aio import acall, acall_stream, iterate_sync, run_sync
from fsm_llm.core.definitions import AdvanceResult, FSMError
from fsm_llm.logging import logger

from .budget import budget_kwargs
from .calls import Settled, settle
from .events import event_scope, event_stream, record_started
from .helpers import _unexpected_error
from .stack import FSMStackFrame
from .surface import _ApiSurface

# DECISION plan-2026-09-30T062855-07ad3f8c/D-021
# The two helpers below are what every turn entry of ``API`` shares (sync and
# stream). They are MODULE-LEVEL functions on purpose: do NOT turn them into
# ``API`` methods called through ``self``. Under the ``Mock(spec=API)``
# unbound-self tests a ``self._helper()`` call resolves to a Mock, so the error
# mapping and the auto-save would silently not run (the same trap the
# plan-2026-07-21T082818-4c63deac/D-002 anchor in ``converse_stream`` records).
# Do NOT merge them into one wrapper either: the sync turn saves only after a
# successful turn, the stream saves in a ``finally`` (also when abandoned).


@contextmanager
def _turn_errors(doing: str, do: str) -> Iterator[None]:
    """Map what escapes a turn entry of ``API`` to the public error contract.

    Contract: ``doing`` / ``do`` name the entry in the log line and the error
    message (``"processing"`` / ``"process"``, ``"streaming"`` / ``"stream"``).
    ``ValueError`` and ``FSMError`` pass through unchanged; any other
    ``Exception`` is logged and re-raised as ``FSMError`` chained from it.
    ``BaseException`` (``GeneratorExit``, ``KeyboardInterrupt``) is untouched.
    """
    try:
        yield
    except (ValueError, FSMError):
        raise
    except Exception as e:
        raise _unexpected_error(
            e, f"Error {doing} message", f"Failed to {do} message"
        ) from e


def _with_greetings(
    reply: str | None, before: list[str], after: list[str]
) -> str | None:
    """The reply of a step between the greetings of the children a call state pushed.

    Contract: ``before`` are the greetings of the START settle (a child pushed before the step
    ran) and ``after`` those of the END settle. ``reply`` unchanged when both are empty; else
    the greetings of ``before``, the non-empty reply and the greetings of ``after``, joined by
    one newline (D-023 item 9, order of D-026 (4) of plan baa37f40). Never raises.
    """
    if not before and not after:
        return reply
    return "\n".join([*before, *([reply] if reply else []), *after])


# DECISION plan-2026-11-23T122249-baa37f40/D-026
# The settle AFTER a committed turn must never lose that turn: the reply is the caller's and the
# turn is already in the history. Do NOT let an `Exception` of the END settle (a refused end, a
# second driver, a push failure) escape this guard (the reply would be lost, the session never
# saved, and a caller that retries would resend the message to the child), do NOT save the
# session only after the settle (an error or a cancel there would skip it), and do NOT catch
# `BaseException` or `CancelledError` here: a cancel after the commit leaves the turn committed
# with the request `running`, which the next entry heals (D-023). A START settle is not guarded:
# it runs before the turn, so a raise loses nothing and a retry is safe.
async def _settle_after(api: _ApiSurface, conversation_id: str) -> Settled:
    """The END settle of a turn entry, after the committed turn was saved.

    Contract: returns the greetings of the children pushed (empty when none or when settle
    raised an ``Exception``, which is logged at WARNING); saves the session again when settle
    changed something. A ``BaseException`` propagates.
    """
    try:
        settled = await settle(api, conversation_id)
    except Exception as error:
        logger.warning(
            f"Post-turn settle of conversation {conversation_id} failed, the next entry "
            f"retries it: {type(error).__name__}: {error}"
        )
        return Settled()
    if settled.changed:
        _auto_save_session(api, conversation_id)
    return settled


def _auto_save_session(api: _ApiSurface, conversation_id: str) -> None:
    """Save the session after a turn when ``api`` has a session store.

    Contract: no-op without a store. Never raises for an ``Exception``: a
    failed save is logged at WARNING and the turn's result stands.
    """
    if api._session_store is None:
        return
    try:
        api.save_session(conversation_id)
    except Exception as e:
        logger.warning(f"Auto-save session failed: {e!s}")


def _open_stream(
    api: _ApiSurface,
    conversation_id: str,
    doing: str,
    do: str,
    manager_name: str,
    *args: Any,
) -> AsyncGenerator[str, None]:
    """The one stream entry of ``API``: the eager id check, then the lazy async generator.

    Contract: the id is resolved NOW (``ValueError`` if unknown), so a stream never iterated
    holds nothing. The generator streams ``fsm_manager.<manager_name>(id, *args)`` through
    ``acall_stream`` under the event log scope and saves the session in its ``finally``
    (also when closed early). Module level on purpose: see the D-002 anchor below.
    """
    with _turn_errors(doing, do):
        # D-014: _get_current_fsm_conversation_id already refreshed _last_accessed.
        current_fsm_id = api._get_current_fsm_conversation_id(conversation_id)

    # DECISION plan-2026-07-21T082818-4c63deac/D-002
    # conv_lock acquisition stays LAZY inside the manager's stream (its first step), so an
    # abandoned (never-iterated) stream cannot leak a lock and the ``_lock -> conv_lock`` order
    # is preserved. This generator is a MODULE-LEVEL function taking the API (and the sync
    # ``converse_stream`` is ``iterate_sync`` over it), deliberately NOT a ``self._..._inner``
    # method and not ``self.aconverse_stream(...)``: a bound-method self-dispatch resolves to
    # a Mock under ``Mock(spec=API)`` unbound-self tests and regressed the agents auto-save
    # suite (D-003/CF1). Do NOT hoist conv_lock into the eager prologue and do NOT
    # reintroduce a ``self.``-attribute inner generator.
    async def _stream() -> AsyncGenerator[str, None]:
        with _turn_errors(doing, do):
            try:
                sent = False
                frame_id = current_fsm_id
                # A pending request (an abandoned stream, a restore, a resume, a failed
                # greeting) is applied BEFORE the frame is read, so the message goes to the
                # child it pushes (D-026 (3)); its greetings come first.
                started = await settle(api, conversation_id)
                for greeting in started:
                    yield greeting
                    sent = True
                if started.changed:
                    frame_id = api._get_current_fsm_conversation_id(conversation_id)
                lead = "\n" if sent else ""
                async with aclosing(
                    event_stream(
                        api,
                        conversation_id,
                        acall_stream(api.fsm_manager, manager_name, frame_id, *args),
                    )
                ) as chunks:
                    async for chunk in chunks:
                        yield lead + chunk
                        lead = ""
                        sent = True
                # Only at the NORMAL end (never in the `finally`: an abandoned or cancelled
                # stream defers to the next entry's settle). A pushed child's greeting is one
                # more chunk. See the D-023 anchor of `calls.py`.
                for greeting in await _settle_after(api, conversation_id):
                    yield f"\n{greeting}" if sent else greeting
                    sent = True
            finally:
                # Auto-save session after stream completes or is abandoned
                _auto_save_session(api, conversation_id)

    return _stream()


class TurnEntriesMixin(_ApiSurface):
    """The turn entries of ``API``: start a conversation and run one turn, sync, as a stream."""

    def start_conversation(
        self,
        initial_context: dict[str, Any] | None = None,
        *,
        max_steps: int | None = None,
        max_seconds: float | None = None,
        _suppress_start: bool = False,
        _conversation_id: str | None = None,
    ) -> tuple[str, str]:
        """
        Start new conversation with the 2-pass architecture.

        Args:
            initial_context: Optional initial context data
            max_steps: Lifetime turn budget of this conversation, instead of the
                ``API`` default (``None``: the default, itself no limit).
            max_seconds: Lifetime wall-clock budget, as ``max_steps``.
            _suppress_start: Internal resume flag forwarded to
                ``FSMManager.start_conversation(suppress_start=...)``. When True
                the START_CONVERSATION handlers and the Pass-2 greeting are
                skipped (used by ``restore_session``). Not part of the public
                API; default False preserves normal start behavior.
            _conversation_id: Internal, beside ``_suppress_start``: keep this id
                instead of a minted one (``FSMManager.start_conversation(conversation_id=)``;
                ``resume``). Sent to the manager only when given, so the default
                call is unchanged. Not part of the public API.

        Returns:
            Tuple of (conversation_id, initial_response). The response is
            the empty string when the initial state is silent (empty
            ``response_instructions``): no LLM call, nothing in the history.

        Raises:
            FSMError: ``initial_context`` is neither ``None`` nor a dict, or the
                start failed (any other unexpected error is wrapped in it).
            ValueError: an invalid ``max_steps`` or ``max_seconds``; passes through
                unchanged, as at ``push_fsm`` and ``pop_fsm``.

        ``run_sync`` over ``astart_conversation``, the one body.
        """
        return run_sync(
            self.astart_conversation(
                initial_context,
                max_steps=max_steps,
                max_seconds=max_seconds,
                _suppress_start=_suppress_start,
                _conversation_id=_conversation_id,
            )
        )

    async def astart_conversation(
        self,
        initial_context: dict[str, Any] | None = None,
        *,
        max_steps: int | None = None,
        max_seconds: float | None = None,
        _suppress_start: bool = False,
        _conversation_id: str | None = None,
    ) -> tuple[str, str]:
        """The async twin of ``start_conversation``: the greeting awaits its model call."""
        try:
            # Start conversation using enhanced FSM manager
            kept_id: dict[str, Any] = (
                {}
                if _conversation_id is None
                else {"conversation_id": _conversation_id}
            )
            conversation_id, response = await acall(
                self.fsm_manager,
                "start_conversation",
                self.fsm_id,
                initial_context=initial_context,
                suppress_start=_suppress_start,
                **kept_id,
                **budget_kwargs(self, max_steps, max_seconds),
            )

            # Track conversation and initialize stack
            with self._stack_lock:
                self.conversation_stacks[conversation_id] = [
                    FSMStackFrame(
                        fsm_definition=self.fsm_definition,
                        conversation_id=conversation_id,
                        fsm_id=self.fsm_id,
                    )
                ]
                self._last_accessed[conversation_id] = time.monotonic()

            if not _suppress_start:  # a restore writes its own record (restore_session)
                record_started(self, conversation_id)
            return conversation_id, response

        except (ValueError, FSMError):
            raise
        except Exception as e:
            raise _unexpected_error(
                e, "Error starting conversation", "Failed to start conversation"
            ) from e

    def converse(self, user_message: str, conversation_id: str) -> str:
        """
        Process message using the 2-pass architecture.

        Args:
            user_message: User's message
            conversation_id: Existing conversation ID

        Returns:
            System response; the empty string when the turn ended on a silent
            state (empty ``response_instructions``), which makes no Pass-2
            call and records the user message only.

        ``run_sync`` over ``aconverse``, the one body.
        """
        return run_sync(self.aconverse(user_message, conversation_id))

    async def aconverse(self, user_message: str, conversation_id: str) -> str:
        """The async twin of ``converse``: the event loop runs other tasks while the model
        calls are awaited. A second turn on a busy conversation is refused, never waited for."""
        with _turn_errors("processing", "process"):
            # A call state's request (a cancelled settle, a restore) is applied BEFORE the turn
            # resolves its frame, so the message goes to the child it pushes.
            before = await settle(self, conversation_id)
            # D-014: _get_current_fsm_conversation_id already refreshed _last_accessed.
            current_fsm_id = self._get_current_fsm_conversation_id(conversation_id)
            with event_scope(self, conversation_id):
                response: str = await acall(
                    self.fsm_manager, "process_message", current_fsm_id, user_message
                )
            # Auto-save session if store is configured: the committed turn is saved BEFORE
            # the end settle (an error or a cancel there must not skip it), see `_settle_after`
            _auto_save_session(self, conversation_id)
            after = await _settle_after(self, conversation_id)
            joined = _with_greetings(response, before, after)
            return response if joined is None else joined

    def advance(self, conversation_id: str) -> AdvanceResult:
        """Run one step of the current state with no user message.

        The same turn as ``converse`` (one turn at a time per conversation,
        rollback on failure, every handler timing, extraction, transition
        evaluation, Pass 2 from the post-transition state, session auto-save),
        resolved on the top of the FSM stack. Differences from ``converse``:
        nothing is appended to the history for a user and prompts carry no
        user message.

        Args:
            conversation_id: Existing conversation ID.

        Returns:
            ``AdvanceResult`` with the state before and after, the transition
            outcome, the reply (``None`` for a silent state) and ``ended``.

        Raises:
            ValueError: unknown conversation ID.
            FSMError: the conversation has ended (terminal state), a turn is
                already running for it (a handler calling back, or another
                thread), or the step failed. A failed step is rolled back as a
                ``converse`` turn is: a Pass-2 or POST_PROCESSING failure
                restores the whole turn; a failure in a Pass-1 handler
                (CONTEXT_UPDATE, PRE_TRANSITION, POST_TRANSITION) follows the
                same partial-commit rules as ``converse`` (keys extracted
                before the failure can stay).

        ``run_sync`` over ``aadvance``, the one body.
        """
        return run_sync(self.aadvance(conversation_id))

    async def aadvance(self, conversation_id: str) -> AdvanceResult:
        """The async twin of ``advance`` (see ``aconverse``)."""
        with _turn_errors("advancing without a", "advance without a"):
            before = await settle(self, conversation_id)
            current_fsm_id = self._get_current_fsm_conversation_id(conversation_id)
            with event_scope(self, conversation_id):
                result: AdvanceResult = await acall(
                    self.fsm_manager, "advance", current_fsm_id
                )
            _auto_save_session(self, conversation_id)
            after = await _settle_after(self, conversation_id)
            if before or after:
                result = result.model_copy(
                    update={"response": _with_greetings(result.response, before, after)}
                )
            return result

    def converse_stream(self, user_message: str, conversation_id: str) -> Iterator[str]:
        """Process message, streaming the response tokens.

        Pass 1 (extraction + transitions) runs fully.  Pass 2 yields
        response tokens as they arrive from the LLM.

        This is a thin NON-generator wrapper: it resolves the current FSM
        conversation id at CALL time, which validates existence AND refreshes
        ``_last_accessed``, then returns a lazy nested-closure generator that
        streams the reply.  Resolving eagerly means a created-but-not-yet-iterated
        stream cannot skip validation or be reaped as stale mid-flight by
        ``cleanup_stale_conversations``.

        Args:
            user_message: User's message.
            conversation_id: Existing conversation ID.

        Yields:
            String chunks of the response as they arrive; nothing when the
            turn ended on a silent state.
        """
        return iterate_sync(
            _open_stream(
                self,
                conversation_id,
                "streaming",
                "stream",
                "process_message_stream",
                user_message,
            )
        )

    def aconverse_stream(
        self, user_message: str, conversation_id: str
    ) -> AsyncGenerator[str, None]:
        """The async twin of ``converse_stream``: a plain function (the id is checked now)
        that returns a lazy async generator; ``async for`` over it yields the same chunks."""
        return _open_stream(
            self,
            conversation_id,
            "streaming",
            "stream",
            "process_message_stream",
            user_message,
        )

    def advance_stream(self, conversation_id: str) -> Iterator[str]:
        """Run one step of the current state with no user message, streaming
        the reply.

        The stream form of ``advance``: the same turn, resolved on the top of
        the FSM stack at CALL time, run lazily. Only reply text is yielded; a
        silent state (empty ``response_instructions``) yields no chunk, makes
        no Pass-2 LLM call and appends nothing to the history.
        Read the outcome of the step after the stream is exhausted with
        ``get_current_state`` and ``has_conversation_ended``. The conversation
        is claimed at the first ``next()`` and released when the stream
        finishes, fails or is closed; a stream closed early keeps the partial
        reply in the history and keeps the step (as ``converse_stream`` does).

        Args:
            conversation_id: Existing conversation ID.

        Yields:
            String chunks of the reply as they arrive.

        Raises:
            ValueError: unknown conversation ID (at call time).
            FSMError: at the first ``next()``, when the conversation has ended
                (terminal state) or a turn is already running for it; or when
                the step failed, after its rollback.
        """
        return iterate_sync(
            _open_stream(
                self,
                conversation_id,
                "streaming a step without a",
                "stream a step without a",
                "advance_stream",
            )
        )

    def aadvance_stream(self, conversation_id: str) -> AsyncGenerator[str, None]:
        """The async twin of ``advance_stream`` (see ``aconverse_stream``)."""
        return _open_stream(
            self,
            conversation_id,
            "streaming a step without a",
            "stream a step without a",
            "advance_stream",
        )
