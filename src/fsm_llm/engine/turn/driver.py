"""The turn drivers of ``MessagePipeline`` (``DriverMixin``): one sync or streamed turn, for a
message (``process``) or without one (``advance``).

Layer ``engine`` (plan d454fd62, step 79): split out of the old ``fsm_llm.pipeline``; the
method and function bodies are unchanged.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Iterator
from contextlib import aclosing

from fsm_llm.aio import asteps, iterate_sync, run_sync
from fsm_llm.constants import METADATA_KEY_TRANSITION_CLASSIFICATION
from fsm_llm.core.definitions import (
    CONTEXT_KEY_CUT_OFF,
    AdvanceResult,
    FSMError,
    FSMInstance,
)
from fsm_llm.handlers import HandlerTiming
from fsm_llm.logging import logger

from .records import _TurnRecord, _TurnSnapshot
from .surface import _PipelineSurface


def _logged_as(
    conversation_id: str, steps: AsyncGenerator[str, None]
) -> AsyncGenerator[str, None]:
    """``steps`` with the turn's logger context (``conversation_id``, ``package``) bound only
    WHILE a step runs: loguru resets its token in the Context that set it, and a stream may be
    closed in another one (the asyncgen finalizer)."""
    return asteps(
        lambda: logger.contextualize(
            conversation_id=conversation_id, package="fsm_llm"
        ),
        steps,
    )


class DriverMixin(_PipelineSurface):
    """The sync and stream turn drivers."""

    # ----------------------------------------------------------
    # Full 2-pass processing
    # ----------------------------------------------------------

    def process(self, instance: FSMInstance, message: str, conversation_id: str) -> str:
        """``run_sync`` over ``aprocess``, the one body."""
        return run_sync(self.aprocess(instance, message, conversation_id))

    async def aprocess(
        self, instance: FSMInstance, message: str, conversation_id: str
    ) -> str:
        """Execute the full 2-pass message processing pipeline.

        Pass 1: PRE_PROCESSING handlers → data extraction → context update →
                transition evaluation → state transition
        Pass 2: POST_PROCESSING handlers → response generation

        Args:
            instance: The FSM instance (already validated as non-terminal).
            message: User message to process.
            conversation_id: Conversation identifier.

        Returns:
            Generated response message; the empty string when the turn ended
            on a silent state (empty ``response_instructions``).
        """
        turn = await self._run_turn(instance, message, conversation_id)
        return turn.response or ""

    def advance(self, instance: FSMInstance, conversation_id: str) -> AdvanceResult:
        """``run_sync`` over ``aadvance``, the one body."""
        return run_sync(self.aadvance(instance, conversation_id))

    async def aadvance(
        self, instance: FSMInstance, conversation_id: str
    ) -> AdvanceResult:
        """Run one turn of the current state with no user message.

        The same turn body as ``process`` (snapshot and rollback, handler
        timings, extraction, transition evaluation, Pass 2). Differences: the
        prompts carry no user message, and a silent state (empty
        ``response_instructions``) makes no Pass-2 LLM call and appends nothing
        to the history.

        Args:
            instance: The FSM instance (already validated as non-terminal).
            conversation_id: Conversation identifier.

        Returns:
            The ``AdvanceResult`` of the step.

        Raises:
            FSMError: the current state is terminal (no transition evaluation
                can run), or anything the turn body raises, after its rollback.
        """
        self._require_non_terminal(instance, conversation_id)
        turn = await self._run_turn(instance, None, conversation_id)
        if turn.transition_outcome is None:
            raise FSMError(f"Cannot advance from terminal state '{turn.state_before}'")
        return AdvanceResult(
            state_before=turn.state_before,
            state_after=turn.state_after,
            transition_outcome=turn.transition_outcome,
            response=turn.response,
            ended=not self.get_state(instance, conversation_id).transitions,
        )

    def _require_non_terminal(
        self, instance: FSMInstance, conversation_id: str
    ) -> None:
        """Raise ``FSMError`` before any work when the current state is terminal.

        Contract: the one guard of ``advance`` and ``advance_stream``, so the
        two agree: a terminal state runs no extraction, no handler and no LLM
        call, and leaves the history alone.
        """
        if not self.get_state(instance, conversation_id).transitions:
            raise FSMError(
                f"Cannot advance from terminal state '{instance.current_state}'"
            )

    async def _run_turn(
        self, instance: FSMInstance, user_message: str | None, conversation_id: str
    ) -> _TurnRecord:
        """Run one turn of the current state: the single turn body, an ``async def`` that
        ``process`` and ``advance`` drive through ``run_sync`` (it has no sync name: the
        anchors below name it).

        Contract: ``user_message`` is the user's text, or ``None`` for a turn
        with no user message. ``process`` is the entry for a message,
        ``advance`` the entry for ``None``. Returns the ``_TurnRecord`` of a
        successful turn; on failure the pre-turn snapshot is restored as
        described below and the exception propagates.
        """
        # DECISION plan-2026-09-30T062855-07ad3f8c/D-007
        # One turn body for a message turn and a message-free step. Do NOT add
        # a second method that repeats this snapshot / PRE_PROCESSING / Pass 1 /
        # POST_PROCESSING / Pass 2 / restore sequence for the no-message case:
        # every anchor below would then have to hold in two places.
        # Contextualize propagates conversation_id to all downstream logger
        # calls on this thread (llm/, core/transition_evaluator.py, etc.)
        with logger.contextualize(conversation_id=conversation_id, package="fsm_llm"):
            # DECISION plan-2026-07-21T045419-9925aa3a/D-012
            # Turn-level atomicity snapshot, taken BEFORE Pass 1. A Pass-2
            # failure must not leave a committed Pass-1 transition + extraction
            # visible to the caller: a retry would then evaluate against a state
            # the LLM never responded from. This guard wraps AROUND the Pass-1
            # partial-commit contracts (D-005 scoped rollback, D-006 handler
            # re-raise, _execute_state_transition's POST_TRANSITION rollback) —
            # do NOT push it INTO those helpers, and do NOT drop the deepcopy
            # for a shallow copy: Pass 1 mutates context.data values in place.
            # The deepcopy mirrors the per-transition snapshot already taken at
            # _execute_state_transition (search `old_context_snapshot`), so it
            # is not a new cost class. Snapshot ALL mutable instance state that
            # a handler (CONTEXT_UPDATE/POST_TRANSITION/POST_PROCESSING) can
            # touch — current_state, context.data, context.working_memory AND
            # context.metadata — not just state+data: the latter two are
            # separate fields, so restoring only data leaves the turn NON-atomic.
            # working_memory may be None; deepcopy handles that (its
            # __getstate__/__setstate__ drop the lock, so a WorkingMemory copies
            # safely). Boundary (same as D-005): already-run handler EXTERNAL
            # side effects (I/O) cannot be undone — only in-memory instance
            # state (current_state, context.data, working_memory, metadata) is
            # restored on Pass-2 failure. See D-012.
            snapshot = _TurnSnapshot.capture(instance)
            turn = _TurnRecord(state_before=snapshot.state, state_after=snapshot.state)

            # DECISION plan-2026-09-21T203800-8a03483a/D-002 (A8): a transition
            # classification record belongs to the turn that produced it. Clear
            # it AFTER the snapshot so a rolled-back turn restores the prior
            # record; do NOT move this before the snapshot, and do NOT drop it
            # from process_stream() (same clear there). The record lives in
            # metadata only (A4, plan-2026-09-21T203800-8a03483a/D-005).
            instance.context.metadata.pop(METADATA_KEY_TRANSITION_CLASSIFICATION, None)
            # D-025: the cut-off record, like the classification record, belongs
            # to the turn that produced it (after the snapshot: a rolled-back
            # turn restores the prior one).
            instance.context.data.pop(CONTEXT_KEY_CUT_OFF, None)

            # DECISION plan-2026-09-12T065608-089d0ec7/D-002
            # Widened turn-atomicity: PRE_PROCESSING and POST_PROCESSING handler
            # calls are now EACH individually wrapped in their own restore-on-
            # exception block (reusing the D-012 `_TurnSnapshot` of records.py, no
            # new snapshot logic). Two SEPARATE try/except blocks, not one block
            # spanning PRE_PROCESSING through Pass 2: Pass 1
            # (_execute_extraction_and_transition_pass) deliberately stays
            # OUTSIDE both — it owns its own internal partial-commit contracts
            # (D-005 scoped CONTEXT_UPDATE rollback, D-006 handler re-raise,
            # _execute_state_transition's POST_TRANSITION rollback), and folding
            # it into this outer restore would wipe data Pass 1 legitimately
            # committed before a later failure (see
            # TestPostTransitionHandlerFailure /
            # TestPreTransitionContextUpdateRollback in
            # tests/test_fsm_llm/test_pipeline_handler_contract.py, which pin
            # exactly that scoped behavior). This narrows D-006's documented
            # "survives for PRE_PROCESSING / POST_PROCESSING" guarantee for the
            # case where the SAME handler invocation that produced the partial
            # merge is also what raises: process() now immediately restores the
            # pre-turn snapshot in that case. See D-002 in decisions.md.
            #
            # DECISION plan-2026-09-12T065608-089d0ec7/D-017 [clarifies D-002]:
            # the two try/except blocks below are NOT symmetric in scope. The
            # PRE_PROCESSING block (immediately below) is its OWN, separate
            # try/except -- Pass 1 runs AFTER it, unwrapped, so a PRE_PROCESSING
            # failure can only ever roll back state PRE_PROCESSING itself
            # touched (nothing yet, structurally, since Pass 1 hasn't run).
            # The POST_PROCESSING block further down is DELIBERATELY FOLDED
            # INTO THE SAME try as Pass 2 (_execute_response_generation_pass) --
            # one shared restore-on-exception block, not two. Do NOT split it
            # into its own try/except "to match PRE_PROCESSING's shape": Pass 1
            # (extraction + transition evaluation + the state transition itself)
            # has ALREADY run and committed by the time POST_PROCESSING starts,
            # so a POST_PROCESSING handler failure intentionally rolls back
            # BOTH its own partial merge AND Pass 1's already-committed
            # transition + extracted data -- the whole pre-turn snapshot, the
            # same as a Pass-2 failure would. This is turn atomicity applied
            # consistently: a half-turn (new state kept, response never
            # generated) is not a better outcome than a whole-turn rollback,
            # exactly the same reasoning D-002 already applied to
            # POST_TRANSITION and PRE_PROCESSING. Pinned by
            # TestPostProcessingHandlerFailureRollsBackTransition in
            # tests/test_fsm_llm/test_pipeline_handler_contract.py. See D-017
            # in decisions.md (this entry does not edit or replace D-002 --
            # D-002's own text under-described this scope; D-017 clarifies it).
            try:
                # Execute pre-processing handlers
                self.execute_handlers(
                    instance,
                    HandlerTiming.PRE_PROCESSING,
                    current_state=instance.current_state,
                )
            except Exception:
                # Restore the pre-turn in-memory state so the turn is atomic.
                # Covers all handler-mutable fields, not just state+data (D-012).
                snapshot.restore(instance)
                raise

            # Pass 1: Data extraction + transition evaluation + execution
            (
                extraction_response,
                transition_occurred,
                previous_state,
            ) = await self._aexecute_extraction_and_transition_pass(
                instance, user_message, conversation_id, turn=turn
            )

            try:
                # Execute post-processing handlers (after potential transition)
                #
                # DECISION plan-2026-09-12T065608-089d0ec7/D-017 [clarifies D-002]:
                # a failure here shares the except block below with Pass 2 --
                # see the D-017 note above this method's PRE_PROCESSING try for
                # the full rationale. A raise here rolls back Pass 1's
                # already-committed transition too, not just this call's own
                # partial merge.
                self.execute_handlers(
                    instance,
                    HandlerTiming.POST_PROCESSING,
                    current_state=instance.current_state,
                )

                # Pass 2: Response generation based on final state
                turn.response = await self._execute_response_generation_pass(
                    instance,
                    user_message,
                    extraction_response,
                    transition_occurred,
                    previous_state,
                    conversation_id,
                )
                turn.state_after = instance.current_state
                return turn
            except Exception:
                # Restore the pre-turn in-memory state so the turn is atomic.
                # Covers all handler-mutable fields, not just state+data (D-012).
                # D-017: this ALSO undoes Pass 1's already-committed state
                # transition + extracted data when the exception originated in
                # POST_PROCESSING (see the D-017 comment above), not only when
                # it originated in Pass 2.
                snapshot.restore(instance)
                raise

    def process_stream(
        self, instance: FSMInstance, message: str, conversation_id: str
    ) -> Iterator[str]:
        """Execute 2-pass processing, streaming Pass 2 tokens.

        Pass 1 runs fully (extraction + transition). Pass 2 yields
        response tokens as they arrive from the LLM. ``iterate_sync`` over
        ``aprocess_stream``, the one body.

        Args:
            instance: The FSM instance (already validated as non-terminal).
            message: User message to process.
            conversation_id: Conversation identifier.

        Yields:
            String chunks of the response as they arrive.
        """
        return iterate_sync(self.aprocess_stream(instance, message, conversation_id))

    def aprocess_stream(
        self, instance: FSMInstance, message: str, conversation_id: str
    ) -> AsyncGenerator[str, None]:
        """The async twin of ``process_stream``: an async generator over the stream turn body."""
        return _logged_as(
            conversation_id, self._stream_turn(instance, message, conversation_id)
        )

    def advance_stream(
        self, instance: FSMInstance, conversation_id: str
    ) -> Iterator[str]:
        """Run one turn of the current state with no user message, streaming Pass 2.

        The same stream turn body as ``process_stream``. Differences: the
        prompts carry no user message, and a silent state (empty
        ``response_instructions``) yields no chunk, makes no Pass-2 LLM call
        and appends nothing to the history. ``iterate_sync`` over
        ``aadvance_stream``.

        Args:
            instance: The FSM instance.
            conversation_id: Conversation identifier.

        Yields:
            String chunks of the reply as they arrive; nothing for a silent
            state.

        Raises:
            FSMError: at the first ``next()`` when the current state is
                terminal, or anything the turn body raises, after its rollback.
        """
        return iterate_sync(self.aadvance_stream(instance, conversation_id))

    async def aadvance_stream(
        self, instance: FSMInstance, conversation_id: str
    ) -> AsyncGenerator[str, None]:
        """The async twin of ``advance_stream`` (the terminal check runs at the first step)."""
        self._require_non_terminal(instance, conversation_id)
        async with aclosing(
            _logged_as(
                conversation_id, self._stream_turn(instance, None, conversation_id)
            )
        ) as chunks:
            async for chunk in chunks:
                yield chunk

    async def _stream_turn(
        self, instance: FSMInstance, user_message: str | None, conversation_id: str
    ) -> AsyncGenerator[str, None]:
        """Run one turn of the current state, streaming Pass 2: the single
        stream turn body (an async generator; the sync stream is ``iterate_sync`` over it).

        Contract: ``user_message`` is the user's text, or ``None`` for a turn
        with no user message. ``process_stream`` is the entry for a message,
        ``advance_stream`` the entry for ``None``. Yields the reply chunks. On
        an ``Exception`` the pre-turn snapshot is restored as described below
        and the exception propagates; ``GeneratorExit`` restores nothing. The
        entries wrap it in ``_logged_as`` (the logger context is bound per step).
        """
        # DECISION plan-2026-09-30T062855-07ad3f8c/D-025
        # One STREAM turn body for a message turn and a message-free step,
        # the stream counterpart of `_run_turn` (D-007). Do NOT add a second
        # generator that repeats this sequence for the no-message case, and do
        # NOT fold this body into `_run_turn`: e3131cc2/D-004 below keeps the
        # sync and the stream guards as two separate bodies.
        # DECISION plan-2026-07-21T072826-e3131cc2/D-004
        # Turn-level atomicity snapshot for the STREAMING path — the exact
        # mirror of process()'s H2 guard (plan-2026-07-21T045419-9925aa3a/
        # D-012/D-013). process_stream is a SEPARATE entry point; a Pass-2
        # failure here must not leave a committed Pass-1 transition +
        # extraction visible to a retry any more than it may in process().
        # Snapshot ALL four handler-mutable fields before Pass 1 (deepcopy:
        # Pass 1 mutates context.data values in place; working_memory may be
        # None and deepcopy handles that).
        #
        # Why this generator try/except is NOT redundant:
        #  (a) NOT redundant with process(): that guard lives in a different
        #      method and never runs for the streaming caller. Do NOT delete
        #      this as "already covered by process()".
        #  (b) NOT redundant with _stream_response_generation_pass's
        #      `persist=False` path: that flag ONLY suppresses appending the
        #      accumulated assistant message to conversation history — it
        #      never restores current_state / context.data / working_memory /
        #      metadata. Those four survive a mid-stream failure uncorrupted
        #      without this block.
        #  (c) `except Exception` (NOT BaseException) is deliberate, matching
        #      process() and the D-015 streaming-abandonment contract:
        #      GeneratorExit / KeyboardInterrupt from a consumer that stopped
        #      iterating must NOT trigger rollback — the partial reply the
        #      user already saw is kept (documented boundary; a yielded chunk
        #      cannot be un-yielded, same as H2's "external side effects not
        #      undone").
        snapshot = _TurnSnapshot.capture(instance)

        # DECISION plan-2026-09-21T203800-8a03483a/D-002 (A8): streaming
        # mirror of the per-turn clear in process(); after the snapshot.
        instance.context.metadata.pop(METADATA_KEY_TRANSITION_CLASSIFICATION, None)
        # D-025: the cut-off record, like the classification record, belongs
        # to the turn that produced it (after the snapshot: a rolled-back
        # turn restores the prior one).
        instance.context.data.pop(CONTEXT_KEY_CUT_OFF, None)

        # DECISION plan-2026-09-12T065608-089d0ec7/D-002
        # Widened turn-atomicity, streaming mirror of process()'s guard
        # above (same D-002 rationale: two separate try/except blocks,
        # Pass 1 stays unwrapped). See the comment in process() for the
        # full explanation; not repeated here to avoid drift between the
        # two copies.
        #
        # DECISION plan-2026-09-12T065608-089d0ec7/D-017 [clarifies D-002]:
        # the two blocks below are NOT symmetric in scope -- see the D-017
        # comment in process() (same file, above) for the full rationale.
        # In short: PRE_PROCESSING (immediately below) is its own separate
        # try/except; POST_PROCESSING (further down) is deliberately
        # folded into the SAME try as Pass 2's streaming call
        # (_stream_response_generation_pass), so a POST_PROCESSING
        # handler failure here also rolls back Pass 1's already-committed
        # transition + extracted data, not just its own partial merge.
        try:
            # Execute pre-processing handlers
            self.execute_handlers(
                instance,
                HandlerTiming.PRE_PROCESSING,
                current_state=instance.current_state,
            )
        except Exception:
            # Restore the pre-turn in-memory state so the streaming turn is
            # atomic. Covers all handler-mutable fields (D-004, mirrors D-012/
            # D-013). GeneratorExit deliberately does NOT reach here.
            snapshot.restore(instance)
            raise

        # Pass 1: Data extraction + transition (runs fully)
        (
            extraction_response,
            transition_occurred,
            previous_state,
        ) = await self._aexecute_extraction_and_transition_pass(
            instance, user_message, conversation_id
        )

        try:
            # Execute post-processing handlers
            #
            # D-017: a failure here shares the except block below with
            # Pass 2's streaming call -- see the D-017 note above this
            # method's PRE_PROCESSING try.
            self.execute_handlers(
                instance,
                HandlerTiming.POST_PROCESSING,
                current_state=instance.current_state,
            )

            # Pass 2: Stream response generation
            async with aclosing(
                self._stream_response_generation_pass(
                    instance,
                    user_message,
                    extraction_response,
                    transition_occurred,
                    previous_state,
                    conversation_id,
                )
            ) as chunks:
                async for chunk in chunks:
                    yield chunk
        except Exception:
            # Restore the pre-turn in-memory state so the streaming turn is
            # atomic. Covers all handler-mutable fields (D-004, mirrors D-012/
            # D-013). GeneratorExit deliberately does NOT reach here.
            snapshot.restore(instance)
            raise
