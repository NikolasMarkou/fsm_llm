"""Pass 2 of ``MessagePipeline`` (``Pass2Mixin``): the reply from the post-transition state, the
greeting, the stream variant and the response request builder.

Layer ``engine`` (plan d454fd62, step 79): split out of the old ``fsm_llm.pipeline``; the
method and function bodies are unchanged.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import Any

from fsm_llm.aio import acall, acall_stream, run_sync
from fsm_llm.constants import CONTEXT_KEY_OUTPUT_RESPONSE_FORMAT
from fsm_llm.context import record_cut_off, was_cut_off
from fsm_llm.core.definitions import (
    CONTEXT_KEY_CUT_OFF,
    DataExtractionResponse,
    FSMInstance,
    ResponseGenerationRequest,
    ResponseGenerationResponse,
    State,
)
from fsm_llm.logging import logger
from fsm_llm.runtime import allm_call

from .records import _is_silent, _response_temperature
from .surface import _PipelineSurface


class Pass2Mixin(_PipelineSurface):
    """Pass 2: the reply generation and its request builder."""

    async def _stream_response_generation_pass(
        self,
        instance: FSMInstance,
        user_message: str | None,
        extraction_response: DataExtractionResponse,
        transition_occurred: bool,
        previous_state: str | None,
        conversation_id: str,
    ) -> AsyncGenerator[str, None]:
        """Stream Pass 2: yield response tokens as they arrive (an async generator; the
        sync stream is ``iterate_sync`` over it).

        Yields nothing, and makes no LLM call, for a turn that ended on a
        silent state.
        """
        log = logger.bind(conversation_id=conversation_id)

        current_state = self.get_state(instance, conversation_id)

        if _is_silent(current_state):
            # Stream mirror of the D-037 branch in
            # `_execute_response_generation_pass`: no chunk, no call, nothing
            # appended to the history.
            log.debug("Silent state: no response")
            return

        request = self._build_response_request(
            instance,
            current_state,
            conversation_id,
            user_message,
            extraction_response,
            transition_occurred,
            previous_state,
            stream=True,
        )

        # Accumulate chunks to store in conversation history
        chunks: list[str] = []
        persist = True
        # [SUPERSEDED BY plan-2026-11-19T193704-f150ec28/D-002 on the cancel clause: the paragraph
        # on a KeyboardInterrupt below stands for interrupts; a CancelledError now has the arms
        # of the driver and of `turns.py` (the whole turn is rolled back, no partial reply).]
        # DECISION plan-2026-07-18T051819-80b0bd4d/D-015 [STALE]: client abandonment and a backend
        # stream error MUST diverge here, and the ORDER of these except clauses is
        # the whole mechanism.
        #   * GeneratorExit (the consumer stopped iterating / the generator was
        #     closed or GC'd) keeps `persist = True`, so the partial reply the user
        #     ACTUALLY SAW is written to history.  engine/fsm/turns.py (`_open_locked_stream`) documents that
        #     contract and depends on it: its _rollback_user_message call then sees
        #     a trailing {"system": partial} entry and correctly no-ops, leaving the
        #     user turn in place.  Do NOT "simplify" this clause away.
        #   * Any other exception (a backend error mid-stream) clears `persist`, so
        #     no truncated assistant turn is stored.  That leaves a bare
        #     {"user": ...} as the last exchange, which is what
        #     _rollback_user_message needs in order to pop it.
        # BaseException, not Exception: KeyboardInterrupt/SystemExit must not
        # silently persist a truncated turn either.  Both clauses bare-`raise`, so
        # nothing is swallowed.  Writing `except BaseException` FIRST would swallow
        # the abandonment semantics and silently flip the engine/fsm/turns.py contract.
        #
        # SCOPE OF THE PARITY CLAIM — corrected in step 8; measure, do not assume.
        # Full parity with the synchronous path (_execute_response_generation_pass,
        # which never calls add_system_message when generate_response raises) holds
        # for **Exception subclasses only**.  For a true BaseException
        # (KeyboardInterrupt / SystemExit) this clause suppresses the partial reply
        # correctly, but the orphaned user turn SURVIVES: the rollback lives in
        # engine/fsm/turns.py (`_open_locked_stream`), whose clauses are FSMError / GeneratorExit / Exception:
        # there is no BaseException clause, so _rollback_user_message never runs.
        # Probed on this tree: ^C mid-stream leaves
        # [{'system': 'Greetings!'}, {'user': 'hi'}].  Do NOT "fix" this by
        # widening the engine/fsm/turns.py clause: whether a KeyboardInterrupt should roll back
        # is a design call, not a defect fix.  Open item, recorded in
        # verification.md § Not Verified.
        #
        # .throw() vs .close(): only GeneratorExit is the documented abandonment
        # signal, so a thrown Exception is deliberately treated as an ERROR (partial
        # dropped, user turn rolled back) even though the consumer may already have
        # seen chunks.  That asymmetry is intended, not an oversight.
        async with allm_call(
            self.llm_interface, "response_stream", request, stream=True
        ) as call:
            try:
                async with aclosing(
                    call.astream(
                        acall_stream(
                            self.llm_interface, "generate_response_stream", request
                        )
                    )
                ) as stream:
                    async for chunk in stream:
                        chunks.append(chunk)
                        yield chunk
            except GeneratorExit:
                raise
            except BaseException:
                persist = False
                raise
            finally:
                # Store the accumulated response when the stream completed OR the
                # consumer abandoned it — but never when it errored (see D-015).
                #
                # DECISION plan-2026-07-19T191147-4b664252/D-003 [STALE]
                # TWO DISTINCT suppression conditions, deliberately not merged:
                #   * `persist` is False only for a mid-stream ERROR (D-015 above).
                #   * `full_message` is empty when the provider produced no real text
                #     (e.g. every delta.content was ""), which `if chunks` could not
                #     see — ['', ''] is a truthy list, so a blank assistant turn was
                #     persisted permanently. Test on the JOINED content, never on the
                #     chunk list.
                # Do NOT collapse these into one flag: a GeneratorExit carrying a
                # GENUINE partial must still persist (that is the engine/fsm/turns.py contract),
                # and only the empty-content case is suppressed here.
                full_message = "".join(chunks)
                call.stream_done(full_message)
                if full_message and persist:
                    instance.context.conversation.add_system_message(full_message)
                    log.debug("Streaming response generation completed")

    # ----------------------------------------------------------
    # Initial response generation
    # ----------------------------------------------------------

    def generate_initial_response(
        self, instance: FSMInstance, conversation_id: str
    ) -> str:
        """``run_sync`` over ``agenerate_initial_response``, the one body."""
        return run_sync(self.agenerate_initial_response(instance, conversation_id))

    async def agenerate_initial_response(
        self, instance: FSMInstance, conversation_id: str
    ) -> str:
        """Generate initial response for conversation start (no extraction/transition).

        Returns the empty string, with no LLM call and nothing appended to the
        history, when the initial state is silent.
        """
        log = logger.bind(conversation_id=conversation_id)

        instance.context.data.pop(CONTEXT_KEY_CUT_OFF, None)  # D-025
        current_state = self.get_state(instance, conversation_id)

        if _is_silent(current_state):
            # D-037 (see `_execute_response_generation_pass`): a silent initial
            # state makes no LLM call, appends nothing and returns no text. A
            # ReAct-style `think` state is the initial state of every agent
            # FSM; a Pass 2 here put completion prose in history before any
            # tool ran (F-LIVE-01).
            log.debug("Silent initial state: no response")
            return ""

        # The greeting has no extraction response and no user message: the
        # builder takes `None` for both (07ad3f8c/D-035, D-044 inside it).
        request = self._build_response_request(
            instance, current_state, conversation_id, None, None, False, None
        )

        async with allm_call(self.llm_interface, "greeting", request) as call:
            response: ResponseGenerationResponse = await acall(
                self.llm_interface, "generate_response", request
            )
            await call.aresponse(response)
        instance.last_response_generation = response
        instance.context.conversation.add_system_message(response.message)
        if was_cut_off(response):
            record_cut_off(instance, reply=True)

        log.info("Generated initial response")
        return response.message

    # ----------------------------------------------------------
    # Pass 2: Response generation
    # ----------------------------------------------------------

    async def _execute_response_generation_pass(
        self,
        instance: FSMInstance,
        user_message: str | None,
        extraction_response: DataExtractionResponse,
        transition_occurred: bool,
        previous_state: str | None,
        conversation_id: str,
    ) -> str | None:
        """Execute Pass 2: Response Generation based on final state.

        Returns the reply text, or ``None`` when the turn ended on a silent
        state (no LLM call is made and nothing is appended to the history).
        """
        log = logger.bind(conversation_id=conversation_id)
        log.debug("Executing response generation pass")

        current_state = self.get_state(instance, conversation_id)

        if _is_silent(current_state):
            # DECISION plan-2026-09-30T062855-07ad3f8c/D-037
            # A silent state (empty `response_instructions`) says nothing on
            # every entry: no LLM interface call, no text, nothing appended to
            # the history. Do NOT bring back a skip request (`system_prompt="."`
            # or a `skip_generation` flag: a provider-less call with no
            # product), and do NOT return or record a `[state]` placeholder:
            # every later prompt rendered it as assistant text. Supersedes
            # 0d9c218e/D-006 and 8b258a25/D-034. The same rule holds at the
            # stream site and the greeting. See decisions.md D-037, D-029.
            log.debug("Silent state: no response")
            return None

        request = self._build_response_request(
            instance,
            current_state,
            conversation_id,
            user_message,
            extraction_response,
            transition_occurred,
            previous_state,
        )

        async with allm_call(
            self.llm_interface, "response_generation", request
        ) as call:
            response: ResponseGenerationResponse = await acall(
                self.llm_interface, "generate_response", request
            )
            await call.aresponse(response)
        instance.last_response_generation = response
        instance.context.conversation.add_system_message(response.message)
        if was_cut_off(response):
            record_cut_off(instance, reply=True)

        log.debug("Response generation pass completed")
        return response.message

    def _build_response_request(
        self,
        instance: FSMInstance,
        current_state: State,
        conversation_id: str,
        user_message: str | None,
        extraction_response: DataExtractionResponse | None,
        transition_occurred: bool,
        previous_state: str | None,
        *,
        stream: bool = False,
    ) -> ResponseGenerationRequest:
        """Build the Pass-2 request of the turn, the stream turn or the greeting.

        Interface contract (private; three callers: ``_execute_response_generation_pass``,
        ``_stream_response_generation_pass``, ``agenerate_initial_response``):
        the caller has already ruled out a silent state. ``user_message`` is the
        turn's text, ``None`` for a step without one. ``extraction_response`` is
        the Pass-1 result, or ``None`` for the greeting, which then has no
        extracted data, no rejected corrections, a ``""`` message in the
        prompt and ``None`` in the request. A terminal state carries the
        structured output format, the greeting of a terminal initial state
        included. ``stream`` asks for plain text unless that format is set. Returns the ``ResponseGenerationRequest``; raises what the
        prompt builder, ``fsm_resolver`` or the request model raise. It makes no
        LLM call and changes no instance state.
        """
        fsm_def = self.fsm_resolver(instance.fsm_id)
        greeting = extraction_response is None

        # Only enforce structured output format on terminal states (no
        # outgoing transitions).  Applying it on intermediate states forces
        # the model to produce JSON when the prompt asks for free-form text,
        # which can cause small models to hang or produce garbage. The greeting
        # reads it too: a terminal initial state's greeting is the only reply
        # that FSM ever gives.
        output_response_format = None
        if not current_state.transitions:
            output_response_format = instance.context.data.get(
                CONTEXT_KEY_OUTPUT_RESPONSE_FORMAT
            )

        extracted_data: dict[str, Any] = {}
        extraction_failed = False
        if extraction_response is not None:
            extracted_data = extraction_response.extracted_data
            extraction_failed = extraction_response.extraction_failed

        # DECISION plan-2026-09-19T175721-21cd7f8e/D-005: scope the
        # PROMPT, the only context channel (request.context is gone). Do NOT
        # revert to full instance.context.data: read_keys would then
        # promise scoping the prompt never delivered (hidden values leaked).
        # DECISION plan-2026-09-21T203800-8a03483a/D-008: the prompt gets
        # WorkingMemory under data (get_merged_data), scoped the same way,
        # at all three Pass-2 sites (sync, stream, greeting; now this one
        # builder). Do NOT feed WM through a request field only, and do NOT
        # merge hidden buffers.
        context = self._apply_context_scope(
            instance.context.get_merged_data(), current_state, conversation_id
        )
        rejected_corrections = None
        if extraction_response is not None and extraction_response.rejected_corrections:
            # DECISION plan-2026-09-19T175721-21cd7f8e/D-032
            # read_keys scopes THIS channel too: a refused value is a context
            # value. Do NOT pass the raw dict (final review concern 3): the key
            # the state hides from <current_context> then leaks here.
            rejected_corrections = self._apply_context_scope(
                extraction_response.rejected_corrections, current_state, conversation_id
            )

        # DECISION plan-2026-09-19T175721-21cd7f8e/D-003: stream plain text
        # unless the terminal state carries a structured output format.  The
        # sync prompt (JSON envelope) is unchanged; the stream yields and
        # persists raw deltas, so the envelope must never be requested here.
        system_prompt = self.response_generation_prompt_builder.build_response_prompt(
            instance=instance,
            state=current_state,
            fsm_definition=fsm_def,
            extracted_data=extracted_data,
            transition_occurred=transition_occurred,
            previous_state=previous_state,
            # None (a step without a user message) selects the prompt's
            # no-message wording (07ad3f8c/D-035); the greeting keeps the
            # conversational wording it is built with for "".
            user_message="" if greeting else user_message,
            plain_text_response=stream and output_response_format is None,
            context=context,
            rejected_corrections=rejected_corrections,
            extraction_failed=extraction_failed,
        )

        # The greeting has no user message: the request says so with None
        # (07ad3f8c/D-044).
        return ResponseGenerationRequest.checked(
            prompt_context=context,
            system_prompt=system_prompt,
            user_message=None if greeting else user_message,
            transition_occurred=transition_occurred,
            response_format=output_response_format,
            temperature=_response_temperature(instance),
        )

    # ----------------------------------------------------------
    # Utilities
    # ----------------------------------------------------------

    @staticmethod
    def _apply_context_scope(
        context: dict[str, Any],
        state: State,
        conversation_id: str,
    ) -> dict[str, Any]:
        """Filter context by state's context_scope if defined.

        If the state has ``context_scope`` with ``read_keys``, returns
        only the keys listed. Missing keys are silently skipped (states
        may be entered before all keys are populated).

        If ``context_scope`` is ``None``, returns the full context
        unchanged.
        """
        if state.context_scope is None:
            return context

        read_keys = state.context_scope.read_keys
        if not read_keys:
            return context

        scoped = {k: v for k, v in context.items() if k in read_keys}
        missing = [k for k in read_keys if k not in context]
        if missing:
            log = logger.bind(conversation_id=conversation_id)
            log.debug(
                f"Context scope: state '{state.id}' requested keys "
                f"{missing} but they are not in context"
            )
        return scoped
