"""The turn path of ``FSMManager`` (``TurnsMixin``): the turn guard, the locked sync
turn, the lazy locked stream and the ERROR handlers.

Layer ``engine`` (plan d454fd62, step 78): split out of the old ``fsm_llm.fsm``; the
method bodies are unchanged. The ``_lock`` then ``conv_lock`` order and the lazy
``conv_lock`` of the nested stream closure are the contracts of the anchors below.
"""

from __future__ import annotations

import threading
import traceback
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, TypeVar

from fsm_llm.core.definitions import AdvanceResult, FSMError, FSMInstance
from fsm_llm.handlers import HandlerTiming

from .helpers import (
    _conversation_entry,
    _unexpected_turn_failure,
    with_conversation_context,
)

if TYPE_CHECKING:
    from fsm_llm.core.definitions import State
    from fsm_llm.engine.turn import MessagePipeline


# Return type of one sync turn: ``str`` for a message, ``AdvanceResult`` for a step.
_TurnT = TypeVar("_TurnT")


class TurnsMixin:
    """The turn path of ``FSMManager``: claim the conversation, run one sync or streamed turn, release it.

    Lock order: ``_lock`` then ``conv_lock``, never the reverse; ``conv_lock`` is only ever
    taken non-blocking here.
    """

    if TYPE_CHECKING:
        # Provided by ``FSMManager`` (declared for mypy only).
        _lock: threading.Lock
        _conversation_locks: dict[str, threading.RLock]
        _active_turns: set[str]
        instances: dict[str, FSMInstance]
        _pipeline: MessagePipeline

        def resolve_state_definition(
            self, instance: FSMInstance, conversation_id: str | None = None
        ) -> State: ...

    def _enter_turn(self, conversation_id: str, conv_lock: threading.RLock) -> None:
        """Claim ``conversation_id`` for one turn; caller holds ``self._lock``.

        Contract: raises ``FSMError`` when a turn is already in flight for this
        conversation on ANY thread (same-thread re-entry included, which the RLock
        alone would let through) or when ``conv_lock`` is held elsewhere. On
        success ``conv_lock`` is held and the id is in ``_active_turns``; the
        caller MUST release both in a ``finally`` (``_active_turns.discard`` first,
        taking no lock, then ``conv_lock.release()``).
        """
        # DECISION plan-2026-09-19T175721-21cd7f8e/D-018
        # Turn-level guard, not an ERROR-handler flag: a handler at ANY timing that
        # calls converse() on its own conversation nested one turn per call (99
        # provider calls for one user turn in repro1), and conv_lock is an RLock so
        # the owning thread always re-acquires it. Do NOT make conv_lock a plain
        # Lock (readers and update_context legitimately re-enter it) and do NOT
        # take ``_lock`` in the release path: that would nest ``_lock`` under
        # ``conv_lock`` and invert the hard ``_lock -> conv_lock`` order.
        if conversation_id in self._active_turns:
            raise FSMError(f"Conversation {conversation_id} is already being processed")
        if not conv_lock.acquire(blocking=False):
            raise FSMError(
                f"Conversation {conversation_id} is already being processed by another thread"
            )
        self._active_turns.add(conversation_id)

    @with_conversation_context
    def process_message(
        self, conversation_id: str, message: str, log: Any = None
    ) -> str:
        """Process user message with the 2-pass architecture.

        Pass 1: Data extraction + transition evaluation + transition execution
        Pass 2: Response generation based on final state
        """
        return self._run_locked_turn(
            conversation_id,
            message,
            log,
            lambda instance: self._pipeline.process(instance, message, conversation_id),
        )

    @with_conversation_context
    def advance(self, conversation_id: str, log: Any = None) -> AdvanceResult:
        """Run one turn of the current state with no user message.

        The same locked turn as ``process_message`` (turn guard, terminal
        check, ERROR handlers, error mapping). No user exchange is appended to
        the history, so a failed step pops none.

        Args:
            conversation_id: The conversation to advance.

        Returns:
            The ``AdvanceResult`` of the step.

        Raises:
            FSMError: unknown conversation, a turn already in flight for it
                (re-entrant or concurrent), a terminal current state
                ("Conversation has ended ..."), or a failed step.
        """
        return self._run_locked_turn(
            conversation_id,
            None,
            log,
            lambda instance: self._pipeline.advance(instance, conversation_id),
        )

    def _run_locked_turn(
        self,
        conversation_id: str,
        message: str | None,
        log: Any,
        run: Callable[[FSMInstance], _TurnT],
    ) -> _TurnT:
        """Claim the conversation, run one sync turn, release it.

        Contract: ``message`` is the user's text, or ``None`` for a turn with
        no user message; ``run`` is the pipeline entry for that turn and its
        return value is returned. Raises ``FSMError`` for an unknown
        conversation, a turn already in flight, a terminal state, or a failed
        turn (see ``_process_message_locked``). The turn guard and
        ``conv_lock`` are always released.
        """
        # DECISION plan-2026-09-30T062855-07ad3f8c/D-024
        # ONE locked sync turn for a message turn and a message-free step; the
        # pipeline entry is the only thing the two callers pass in. Do NOT copy
        # this block or `_process_message_locked` for the no-message case: the
        # turn guard (21cd7f8e/D-018) and the error-clause order (21cd7f8e/D-009,
        # a02151fe/D-014) would then have to hold in two places.
        with self._lock:
            _, conv_lock = _conversation_entry(
                self, conversation_id, missing_lock="keyerror"
            )
            self._enter_turn(conversation_id, conv_lock)
        try:
            return self._process_message_locked(conversation_id, message, log, run)
        finally:
            self._active_turns.discard(conversation_id)
            conv_lock.release()

    @with_conversation_context
    def process_message_stream(
        self, conversation_id: str, message: str, log: Any = None
    ) -> Iterator[str]:
        """Process user message, streaming the Pass 2 response.

        Pass 1 (extraction + transitions) runs fully.  Pass 2 yields
        response tokens as they arrive from the LLM.

        This is a thin NON-generator wrapper: it runs the EAGER, lock-free
        existence check at CALL time (so a created-but-not-yet-iterated stream
        cannot skip validation) and returns a lazy nested-closure generator that
        does the actual Pass-1/Pass-2 work.  Because this is a plain function (no
        ``yield`` in its body), ``@with_conversation_context`` runs the prologue
        eagerly, which is exactly what call-time validation requires.
        """
        return self._open_locked_stream(
            conversation_id,
            message,
            log,
            lambda instance: self._pipeline.process_stream(
                instance, message, conversation_id
            ),
        )

    @with_conversation_context
    def advance_stream(self, conversation_id: str, log: Any = None) -> Iterator[str]:
        """Run one turn of the current state with no user message, streaming
        the Pass 2 reply.

        The same locked stream turn as ``process_message_stream`` (eager
        existence check, lazy turn guard, terminal check, ERROR handlers,
        error mapping). No user exchange is appended to the history, so a
        failed or abandoned step pops none. A silent state yields no chunk.

        Args:
            conversation_id: The conversation to advance.

        Returns:
            A lazy iterator over the reply chunks.

        Raises:
            FSMError: at call time for an unknown conversation; at the first
                ``next()`` for a turn already in flight (re-entrant or
                concurrent), a terminal current state ("Conversation has
                ended ..."), or a failed step.
        """
        return self._open_locked_stream(
            conversation_id,
            None,
            log,
            lambda instance: self._pipeline.advance_stream(instance, conversation_id),
        )

    def _open_locked_stream(
        self,
        conversation_id: str,
        message: str | None,
        log: Any,
        run: Callable[[FSMInstance], Iterator[str]],
    ) -> Iterator[str]:
        """Validate the conversation now, return the lazy locked stream turn.

        Contract: ``message`` is the user's text, or ``None`` for a turn with
        no user message; ``run`` is the pipeline stream entry for that turn.
        Raises ``FSMError`` at CALL time for an unknown conversation. The
        returned generator claims the turn guard and ``conv_lock`` at its
        first ``next()`` and releases both when it finishes, fails or is
        closed; it raises ``FSMError`` for a turn already in flight, a
        terminal state, or a failed turn. A never-iterated generator holds
        nothing.
        """
        # DECISION plan-2026-09-30T062855-07ad3f8c/D-025
        # ONE locked stream turn for a message turn and a message-free step,
        # the stream counterpart of `_run_locked_turn` (D-024). Do NOT copy
        # the closure below for the no-message case. This method is a plain
        # function called by both entries; what 4c63deac/D-002 forbids (a lock
        # acquire outside the closure, a `self.`-attribute generator) still
        # holds inside it.
        with self._lock:
            if conversation_id not in self.instances:
                raise FSMError(f"Conversation {conversation_id} not found")

        # DECISION plan-2026-07-21T082818-4c63deac/D-002
        # The eager prologue above is LOCK-FREE by design: it touches only
        # ``self._lock``/``self.instances`` for the existence check and NEVER
        # ``conv_lock``.  conv_lock acquisition stays inside this NESTED-CLOSURE
        # generator so an abandoned (never-iterated) generator cannot leak a
        # lock, and the ``_lock -> conv_lock`` order is preserved.  The generator
        # is a nested closure (capturing ``self``/args from scope) and
        # deliberately NOT a ``self._..._inner`` method: a bound-method
        # self-dispatch resolves to a Mock under ``Mock(spec=FSMManager)``
        # unbound-self tests (D-003/CF1).  Do NOT hoist ``conv_lock.acquire``
        # into the wrapper and do NOT reintroduce a ``self.``-attribute inner
        # generator.
        def _stream() -> Iterator[str]:
            # conv_lock is acquired non-blocking at the FIRST ``next()`` (lazy),
            # preserving the no-leak-on-abandonment property.  The existence
            # check runs again implicitly when this grabs
            # ``self._conversation_locks[conversation_id]`` under ``self._lock`` —
            # cheap, and it correctly handles a conversation that vanished
            # between call and first iteration.
            with self._lock:
                _, conv_lock = _conversation_entry(
                    self, conversation_id, missing_lock="keyerror"
                )
                self._enter_turn(conversation_id, conv_lock)
            try:
                instance = self.instances[conversation_id]
                current_state = self.resolve_state_definition(instance, conversation_id)
                if not current_state.transitions:
                    raise FSMError(
                        f"Conversation has ended - current state '{instance.current_state}' is terminal"
                    )
                if message is not None:
                    instance.context.conversation.add_user_message(message)
                try:
                    yield from run(instance)
                except FSMError as e:
                    self._rollback_user_message(instance, message, log)
                    # DECISION plan-2026-09-19T175721-21cd7f8e/D-009
                    # ERROR handlers run here too (they used to never run on the
                    # stream path). GeneratorExit / KeyboardInterrupt / SystemExit
                    # are separate arms below and deliberately fire nothing.
                    self._fire_error_handlers(instance, e, log)
                    raise
                except GeneratorExit:
                    # Consumer abandoned the iterator before Pass 2 finished
                    # (e.g. a client timeout before the first token).
                    # GeneratorExit is a BaseException, so it bypasses the
                    # handlers below; roll back the just-added user message so
                    # history is not left with an orphaned user turn (CA3-002).
                    # _rollback_user_message only pops when the last exchange is
                    # a user message, so a partial reply already added by the
                    # pipeline is preserved.
                    self._rollback_user_message(instance, message, log)
                    raise
                except (KeyboardInterrupt, SystemExit):
                    # DECISION plan-2026-07-18T162030-a02151fe/D-014 [STALE]
                    self._rollback_user_message(instance, message, log)
                    raise
                except Exception as e:
                    failure = _unexpected_turn_failure(log, message, e)
                    self._rollback_user_message(instance, message, log)
                    self._fire_error_handlers(instance, e, log)
                    raise failure from e
            finally:
                self._active_turns.discard(conversation_id)
                conv_lock.release()

        return _stream()

    def _process_message_locked(
        self,
        conversation_id: str,
        message: str | None,
        log: Any,
        run: Callable[[FSMInstance], _TurnT],
    ) -> _TurnT:
        """Run one turn while holding the per-conversation lock.

        ``message`` of ``None`` is a turn with no user message: no user
        exchange is appended, so none is popped when the turn fails.
        """
        instance = self.instances[conversation_id]
        log.info(f"Processing turn in state: {instance.current_state}")

        current_state = self.resolve_state_definition(instance, conversation_id)
        if not current_state.transitions:
            raise FSMError(
                f"Conversation has ended - current state '{instance.current_state}' is terminal"
            )

        if message is not None:
            instance.context.conversation.add_user_message(message)

        try:
            return run(instance)

        except FSMError as e:
            self._rollback_user_message(instance, message, log)
            # DECISION plan-2026-09-19T175721-21cd7f8e/D-009
            # An FSMError (notably LLMResponseError, the wrapped provider outage)
            # fires the ERROR-timing handlers like any other failure, then is
            # re-raised UNWRAPPED. Do NOT restore the bare rollback-and-raise: it
            # made ERROR handlers unreachable for the main real failure while
            # docs/handlers.md promises "ERROR -- On any exception".
            self._fire_error_handlers(instance, e, log)
            raise
        except (KeyboardInterrupt, SystemExit):
            # DECISION plan-2026-07-18T162030-a02151fe/D-014 [STALE]
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
            failure = _unexpected_turn_failure(log, message, e)
            self._rollback_user_message(instance, message, log)
            self._fire_error_handlers(instance, e, log)
            raise failure from e

    def _fire_error_handlers(
        self,
        instance: FSMInstance,
        exc: BaseException,
        log: Any,
    ) -> None:
        """Run the ERROR-timing handlers for ``exc`` (called from an ``except`` arm).

        Contract: call only from inside an ``except Exception`` / ``except
        FSMError`` arm, after ``_rollback_user_message``, on the thread that
        already holds this conversation's ``conv_lock``. Takes no lock of its own
        (``HandlerSystem`` does its own locking), so the ``_lock -> conv_lock``
        order is untouched. Never call it for ``KeyboardInterrupt`` /
        ``SystemExit`` / ``GeneratorExit``. Returns ``None`` when the handlers
        ran (or none are registered); if an ERROR handler raises, that exception
        is raised chained ``from exc`` and REPLACES ``exc`` as what the caller
        sees. The caller re-raises ``exc`` itself when this returns.
        """
        try:
            self._pipeline.execute_handlers(
                instance,
                HandlerTiming.ERROR,
                current_state=instance.current_state,
                error_context={
                    "_error": str(exc),
                    "_traceback": traceback.format_exc(),
                },
            )
        except Exception as handler_err:
            # DECISION plan-2026-07-20T040150-876e7164/D-009 [STALE]
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
            # handler always raising is a package-wide contract) and `exc`
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
                f"turn: {handler_err} "
                f"(original failure being unwound: {exc!s})"
            )
            raise handler_err from exc

    @staticmethod
    def _rollback_user_message(
        instance: FSMInstance, message: str | None, log: Any
    ) -> None:
        """Remove last user message from history to avoid duplicates on retry.

        Called immediately after ``add_user_message`` in error paths, so the
        last exchange is always the one just added.  We verify it is a user
        message (not a system reply) before popping. ``message`` of ``None``
        (a turn that appended no user exchange) pops nothing: the last entry,
        whatever it is, belongs to an earlier turn.
        """
        if message is None:
            return
        try:
            exchanges = instance.context.conversation.exchanges
            if exchanges and "user" in exchanges[-1]:
                exchanges.pop()
        except Exception as rollback_err:
            log.warning(
                f"Failed to rollback user message from conversation history: {rollback_err}"
            )
