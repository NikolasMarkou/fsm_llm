"""Bulk extraction of the conversation kind (``ConversationBulkMixin``): the one call that
extracts every field of a state from its extraction instructions.

Layer ``engine`` (plan d454fd62): split out of the old ``fsm_llm.pipeline`` at step 79 and moved
here at step 80 (D-030), where the conversation executor owns it. The method and function bodies
are unchanged except that a service of the pipeline is read as ``self._services.<name>`` (the
``_PipelineServices`` protocol, ``engine/kinds/services.py``) instead of ``self.<name>``.
"""

from __future__ import annotations

from typing import Any

from fsm_llm.aio import acall, run_sync
from fsm_llm.constants import CONTEXT_KEY_AGENT_TRACE, is_forbidden_context_entry
from fsm_llm.context.layout import sanitize_text_for_prompt
from fsm_llm.core.definitions import BulkExtractionRequest, FSMInstance, State
from fsm_llm.logging import logger
from fsm_llm.runtime import allm_call

from .surface import _ConversationSurface


class _BulkFailed(dict):
    """Empty marker returned by ``_bulk_extract_from_instructions`` when the
    bulk call raised. It compares equal to ``{}`` and is falsy, so every caller
    and the two pinned private call shapes still see "nothing extracted"; the
    two ``_execute_data_extraction`` call sites read ``isinstance`` first.

    DECISION plan-2026-09-19T175721-21cd7f8e/D-050
    Do NOT return ``(dict, bool)`` or add an out-parameter (both break the
    pinned 4-argument spy signatures), do NOT stash the flag on
    ``context.metadata`` (it is serialised into the session file) and do NOT
    encode it as a key inside ``rejected_corrections`` (braids two meanings).
    """


class ConversationBulkMixin(_ConversationSurface):
    """Bulk extraction of the conversation kind."""

    # ----------------------------------------------------------
    # Pass 1: Field-based extraction (replaces bulk extract_data)
    # ----------------------------------------------------------

    def _bulk_extract_from_instructions(
        self,
        instance: FSMInstance,
        user_message: str | None,
        state: State,
        conversation_id: str,
    ) -> dict[str, Any]:
        """``run_sync`` over ``_abulk_extract_from_instructions``, the one body."""
        return run_sync(
            self._abulk_extract_from_instructions(
                instance, user_message, state, conversation_id
            )
        )

    async def _abulk_extract_from_instructions(
        self,
        instance: FSMInstance,
        user_message: str | None,
        state: State,
        conversation_id: str,
    ) -> dict[str, Any]:
        """Bulk-extract data when a state has extraction_instructions but no
        explicit required_context_keys or field_extractions.

        Uses a single LLM call with a simple prompt to extract whatever
        data the instructions describe.  Returns a dict of extracted
        key-value pairs (may be empty).

        ``user_message`` of ``None`` (a turn with no user message) builds the
        prompt from the context the state may read and the recent
        conversation instead of a message; the request carries ``None``
        (the LLM layer sends the neutral instruction), never ``""`` (an empty
        string is a message the user sent empty, D-044).

        Failure: any exception from the call is swallowed here (a WARNING) and
        the return is the ``_BulkFailed`` marker, so a turn goes on with
        ``extraction_failed=True``. That includes the ``LLMResponseError`` that
        ``extract_bulk_data`` raises for a reply cut off at ``max_tokens`` that
        is not valid JSON: the turn does NOT fail. A cut reply that still
        parses is committed as is. Neither case is recorded in the cut-off
        record (``record_cut_off``): only Pass-2 replies, field extractions and
        completion-state calls are (D-025); a bulk cut stays unflagged
        (CHANGELOG Known open).
        """
        log = logger.bind(conversation_id=conversation_id)
        prompt_context: dict[str, Any] | None = None

        if user_message is None:
            # DECISION plan-2026-09-30T062855-07ad3f8c/D-023
            # With no message the model extracts from the context, so the
            # context is rendered INTO this prompt. Do NOT dump
            # instance.context.data here (json.dumps or an f-string): it must
            # pass the state's read_keys scope and then the prompt security
            # filter inside build_extraction_source_sections, the same two
            # gates <current_context> has in Pass 2. Do NOT skip the bulk call
            # on a message-free turn either: a bulk-only state would then
            # never extract anything. See decisions.md D-023.
            prompt_context = self._services._apply_context_scope(
                instance.context.get_user_visible_data(),
                state,
                conversation_id,
            )
            source_sections = self._services.data_extraction_prompt_builder.build_extraction_source_sections(
                instance, prompt_context
            )
            source = "\n".join(source_sections).rstrip() or "(no context available)"
            subject = "the current context and the recent conversation below"
            source_block = (
                "There is no user message. Extract only from this context "
                f"and conversation:\n{source}"
            )
            present_in = "the context or the conversation above"
        else:
            subject = "the user's message"
            source_block = f"User message: {sanitize_text_for_prompt(user_message)}"
            present_in = "the user's message"
        prompt = (
            f"Extract information from {subject}.\n\n"
            f"Instructions: {state.extraction_instructions}\n\n"
            f"{source_block}\n\n"
            f'Respond with JSON: {{"extracted_data": {{"key": "value", ...}}, '
            f'"confidence": 0.95, "reasoning": "..."}}\n\n'
            f"Only include keys for information actually present in "
            f"{present_in}. Use descriptive snake_case key names."
        )

        # DECISION plan-2026-09-12T135914-45a654de/D-015
        # Previously reached past the LLMInterface ABC via
        # `cast(LiteLLMInterface, self._services.llm_interface)._make_llm_call(...)`:
        # a private method of ONE concrete implementation, called from
        # outside that class. Now goes through the public ABC surface
        # (`extract_bulk_data`, llm/interface.py), which does its own parsing via the
        # shared `strip_think_and_fences` helper (llm/text.py): the
        # <think>-tag/fence-stripping regex that used to be hand-duplicated
        # here is gone, not reconciled. Do NOT reintroduce a cast+private-call
        # here; if a future LLMInterface subclass needs bulk extraction, it
        # implements `extract_bulk_data` (default raises NotImplementedError,
        # mirroring `extract_field`). See decisions.md D-015.
        try:
            request = BulkExtractionRequest.checked(
                system_prompt=prompt,
                user_message=user_message,
                prompt_context=prompt_context,
            )
            llm = self._services.llm_interface
            async with allm_call(llm, "data_extraction", request) as call:
                response = await acall(llm, "extract_bulk_data", request)
                await call.aresponse(response)
            # Filter out None/empty values — extract_bulk_data returns the
            # extraction verbatim; this merge-time filtering is this
            # caller's own concern, not the LLM interface's.
            # DECISION plan-2026-09-19T175721-21cd7f8e/D-019: this channel
            # is steerable by user text, so it may never plant the agent
            # marker or a secret-named key (both call sites read this
            # return). Do NOT exempt declared names: a declared field has
            # the per-field channel, and an exemption would re-open the
            # no-config fallback. `is_admin`-style undeclared gate keys are
            # closed ONLY when the author lists them in `handler_only_keys`
            # (D-033, opt-in; LV2-04 stays the default).
            handler_only = self._handler_only_keys(instance)
            return {
                k: v
                for k, v in response.extracted_data.items()
                if v is not None
                and v != ""
                and v != {}
                and k != CONTEXT_KEY_AGENT_TRACE
                and k not in handler_only
                and not is_forbidden_context_entry(k, v)
            }
        except Exception as e:
            log.warning(f"Bulk extraction fallback failed: {e}")
            # DECISION plan-2026-09-19T175721-21cd7f8e/D-050: a failed call is
            # not "nothing to extract"; see _BulkFailed.
            return _BulkFailed()
