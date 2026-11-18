"""The post-transition extraction of the conversation kind (``ConversationPostTransitionMixin``):
what the NEW state's conversation executor runs after a transition.

Layer ``engine`` (plan d454fd62): split out of the old ``fsm_llm.pipeline`` at step 79 and moved
here at step 80 (D-030), where the conversation executor owns it. The method and function bodies
are unchanged except that a service of the pipeline is read as ``self._services.<name>`` (the
``_PipelineServices`` protocol, ``engine/kinds/services.py``) instead of ``self.<name>``.
"""

from __future__ import annotations

from typing import Any

from fsm_llm.constants import PROVENANCE_METADATA_KEY
from fsm_llm.context import clean_context_keys
from fsm_llm.core.definitions import (
    DataExtractionResponse,
    FSMInstance,
    State,
    result_key_names,
)
from fsm_llm.handlers import HandlerExecutionError, HandlerTiming
from fsm_llm.logging import logger

from ..records import _record_provenance
from .surface import _ConversationSurface


class ConversationPostTransitionMixin(_ConversationSurface):
    """The extraction the conversation executor runs after a transition."""

    def _extract_after_transition(
        self,
        instance: FSMInstance,
        user_message: str | None,
        conversation_id: str,
        extraction_response: DataExtractionResponse,
        previous_state: str | None,
        new_state: State,
    ) -> None:
        """Post-transition extraction of a conversation state (the conversation
        executor's ``after_transition``).

        Contract: called once, after a transition into ``new_state`` that is not
        an agent-managed FSM's. Re-extracts in ``new_state`` what the same user
        message holds for it (missing field keys, unset classification fields,
        or a full re-run on a revisit) and folds the result into
        ``extraction_response`` (its ``extracted_data``, ``rejected_corrections``
        and ``extraction_failed``) and the context. Returns ``None``. An
        extraction failure is logged and swallowed; a ``HandlerExecutionError``
        propagates (the transition is already committed).
        """
        log = logger.bind(conversation_id=conversation_id)
        # DECISION plan-2026-09-19T175721-21cd7f8e/D-033: the target state
        # may read a handler-only key (`gate` reads `is_admin`); the
        # post-transition pass must not ask the extractor for it either.
        handler_only = self._handler_only_keys(instance)
        new_configs = [
            c
            for c in self._build_field_configs_from_state(new_state)
            if c.field_name not in handler_only
        ]
        missing_configs = [
            c for c in new_configs if instance.context.data.get(c.field_name) is None
        ]
        # DECISION plan-2026-09-19T175721-21cd7f8e/D-034: a transition into
        # a state whose own keys the pipeline already filled (a back edge)
        # re-runs the target's whole Pass-1 extraction, so a correction in
        # THIS message reaches the D-015 bulk path (LV2-03). The trigger is
        # DATA (a config-covered key that is set AND has provenance), not
        # graph shape. Do NOT clear the target's keys on a "backward" edge:
        # "go back" alone would erase them, and a DFS back edge also fires
        # for every agent loop. Do NOT relax the provenance requirement:
        # that reopens repro5 (handler-seeded gate value). States that own
        # classification_extractions are excluded so the call budget stays
        # 1 bulk + 1 retry per still-null required key [budget SUPERSEDED
        # by plan-2026-09-21T203800-8a03483a/D-006 below: the exclusion
        # stays; +1 classifier call per unset classification field of the
        # new state is added]. A self-loop is not a
        # revisit (the state's own extraction just ran on this message; a
        # re-run would let the bulk overwrite a same-turn per-field value
        # and cost +1 call on every chatty turn). The re-run replaces the
        # missing-key ask below (it already covers those keys), so no
        # per-field call is duplicated.
        prov_now = instance.context.metadata.get(PROVENANCE_METADATA_KEY, {})
        revisit = (
            previous_state != instance.current_state
            and bool(new_state.extraction_instructions)
            and not new_state.classification_extractions
            and any(
                c.field_name in prov_now
                and instance.context.data.get(c.field_name) is not None
                for c in new_configs
            )
        )
        # DECISION plan-2026-09-21T203800-8a03483a/D-006 [supersedes the
        # budget sentence of plan-2026-09-19T175721-21cd7f8e/D-034 above]:
        # intent stated for the NEW state in this message was lost because
        # this pass only built field configs. Classify each of the new
        # state's classification fields that is still unset (absent or
        # None), once, through the same commit below. Budget: +1 classifier
        # call per unset classification field of a different, non-agent new
        # state. Do NOT include set fields, self-loops (the state's own pass
        # just classified this message) or agent FSMs (guarded above); the
        # clause "or `handler_only_keys` (D-033)" is SUPERSEDED by the anchor
        # below. Do NOT drop the `revisit` exclusion of classification-owning
        # states: a full re-run there would buy the bulk call on top of these.
        # DECISION plan-2026-11-18T194536-52ce5464/D-002: the owner approved
        # option A on 2026-11-18 ("approve option A"). A framework `result_key`
        # (completion or synthesize state) is closed to EVERY classification
        # channel, here too; an author-listed `handler_only_keys` entry is NOT:
        # the classifier may set it, in Pass 1 and now after a transition (+1
        # classifier call per unset listed field of the new state). Do NOT filter
        # by `handler_only` here again (that re-closes the listed key and makes
        # Pass 1 and this pass disagree) and do NOT drop the result-key filter
        # (a classified `completion_result` starves the completion call). The
        # same rule filters Pass 1: `_classification_configs` in extraction.py.
        closed = result_key_names(self._services.fsm_resolver(instance.fsm_id))
        class_configs = (
            [
                c
                for c in new_state.classification_extractions or []
                if c.field_name not in closed
                and instance.context.data.get(c.field_name) is None
            ]
            if previous_state != instance.current_state
            else []
        )
        if revisit or missing_configs or class_configs:
            log.debug(
                f"Post-transition extraction in "
                f"'{instance.current_state}' for "
                f"{'re-run' if revisit else [c.field_name for c in missing_configs]}"
                f" + classification {[c.field_name for c in class_configs]}"
            )
            try:
                post_data: dict[str, Any] = {}
                if revisit:
                    again = self._execute_data_extraction(
                        instance, user_message, conversation_id
                    )
                    post_data = dict(again.extracted_data)
                    extraction_response.rejected_corrections.update(
                        again.rejected_corrections
                    )
                    extraction_response.extraction_failed |= again.extraction_failed
                    # the re-run overwrote the turn's response on the instance
                    instance.last_extraction_response = extraction_response
                elif missing_configs:
                    post_results = self._execute_field_extractions(
                        instance, user_message, missing_configs, conversation_id
                    )
                    for result in post_results:
                        if result.is_valid and result.value is not None:
                            post_data[result.field_name] = result.value
                if class_configs:
                    # plan-2026-09-21T203800-8a03483a/D-006
                    post_data.update(
                        self._execute_classification_extractions(
                            new_state,
                            user_message,
                            instance,
                            conversation_id,
                            configs_override=class_configs,
                        )
                    )

                if post_data:
                    post_data = clean_context_keys(post_data, conversation_id)
                    if post_data:
                        instance.context.update(post_data)
                        # DECISION plan-2026-09-19T175721-21cd7f8e/D-015
                        _record_provenance(instance, post_data)
                        extraction_response.extracted_data.update(post_data)
                        self._services.execute_handlers(
                            instance,
                            HandlerTiming.CONTEXT_UPDATE,
                            current_state=instance.current_state,
                            updated_keys=set(post_data.keys()),
                        )
                # DECISION plan-2026-09-19T175721-21cd7f8e/D-052 (LV6-01):
                # the source state's bulk may have refused a value that
                # this re-extraction just APPLIED (no config there, one
                # here); Pass 2 must not be told it was not changed. Drop
                # an entry only when the STORED value now equals it under
                # the merge point's own comparison. Do NOT drop by key: a
                # value a handler edited (or that never landed) is still
                # genuinely rejected and stays listed.
                rej = extraction_response.rejected_corrections
                for key in [
                    k
                    for k, v in rej.items()
                    if k in instance.context.data
                    and str(v).strip().lower()
                    == str(instance.context.data[k]).strip().lower()
                ]:
                    del rej[key]
            except HandlerExecutionError:
                # DECISION plan-2026-07-18T051819-80b0bd4d/D-012 [STALE]: a handler failure that
                # escaped MessagePipeline.execute_handlers must propagate REGARDLESS
                # of which call site fired it. The "non-fatal" tolerance below exists
                # for EXTRACTION failures (LLM parse errors, malformed field values),
                # NOT for HANDLER failures. Before this split, the same critical=True
                # CONTEXT_UPDATE handler failed the turn when it fired above (no
                # transition) but was silently swallowed here (transition occurred) —
                # the contract's outcome depended on FSM topology. Do NOT fold this
                # clause back into the broad `except Exception`.
                #
                # FAILURE SHAPE (differs from POST_TRANSITION — verify before relying
                # on it): the transition is ALREADY COMMITTED when this fires. There is
                # no rollback here, unlike _execute_state_transition. The caller sees
                # the conversation left in the TARGET state, the user message popped
                # from history, no assistant reply, and — if the target is terminal —
                # has_conversation_ended() == True. Raising is still correct (a critical
                # handler must not fail silently), but callers recovering from this
                # exception must not assume the turn was atomic.
                raise
            except Exception as e:
                log.warning(f"Post-transition extraction failed (non-fatal): {e}")
