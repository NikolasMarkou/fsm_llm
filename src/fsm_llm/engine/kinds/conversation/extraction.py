"""Data extraction of the conversation kind (``ConversationExtractionMixin``): the bulk-then-
field-based Pass 1 extraction and the handler-only key set.

Layer ``engine`` (plan d454fd62): split out of the old ``fsm_llm.pipeline`` at step 79 and moved
here at step 80 (D-030), where the conversation executor owns it. The method and function bodies
are unchanged except that a service of the pipeline is read as ``self._services.<name>`` (the
``_PipelineServices`` protocol, ``engine/kinds/services.py``) instead of ``self.<name>``.
"""

from __future__ import annotations

import re
from typing import Any

from fsm_llm.constants import PROVENANCE_METADATA_KEY
from fsm_llm.core.definitions import (
    ClassificationExtractionConfig,
    DataExtractionResponse,
    FieldExtractionConfig,
    FieldExtractionResponse,
    FSMInstance,
    State,
    handler_only_names,
    result_key_names,
)
from fsm_llm.llm.ollama import is_ollama_model
from fsm_llm.logging import logger

from ..records import _is_agent_managed, _value_digest
from .bulk import _BulkFailed
from .surface import _ConversationSurface


class ConversationExtractionMixin(_ConversationSurface):
    """Pass 1 data extraction of the conversation kind."""

    def _handler_only_keys(self, instance: FSMInstance) -> frozenset[str]:
        """Keys no extraction may write (D-033): the FSM author's
        ``handler_only_keys`` plus the ``result_key`` of every completion state
        (only that state's own completion call writes it, D-021) and of every
        synthesize state (only the runner writes it, D-002 of plan 2cefeb61).

        The result keys come from the kind registry through ``handler_only_names``.
        Resolved per instance through ``fsm_resolver`` so a stacked child uses
        its own list, not the root's. Returns an empty set for the default.
        """
        return handler_only_names(self._services.fsm_resolver(instance.fsm_id))

    def _classification_configs(
        self, state: State, instance: FSMInstance
    ) -> list[ClassificationExtractionConfig]:
        """The classification entries of ``state`` that may run: those not named
        like a framework ``result_key`` (completion or synthesize), which only its
        own kind writes. The ONE filter of Pass 1, its retry loop and the
        classification loop; the pass after a transition applies the same rule
        (``post_transition.py``, where the decision is anchored). An author-listed
        ``handler_only_keys`` entry stays classifiable. Returns a new list."""
        closed = result_key_names(self._services.fsm_resolver(instance.fsm_id))
        return [
            c
            for c in state.classification_extractions or []
            if c.field_name not in closed
        ]

    def _execute_data_extraction(
        self, instance: FSMInstance, user_message: str | None, conversation_id: str
    ) -> DataExtractionResponse:
        """Execute data extraction via per-field ``extract_field`` calls
        and classification extractions (the conversation executor's ``extract``).

        Builds a unified list of ``FieldExtractionConfig`` from both
        ``required_context_keys`` and explicit ``field_extractions``,
        then extracts each field individually.  Also runs any
        ``classification_extractions`` declared on the state.  Supports
        multi-pass retry for missing required fields (up to ``extraction_retries``).
        """
        log = logger.bind(conversation_id=conversation_id)
        log.debug("Executing field-based data extraction")

        current_state = self._services.get_state(instance, conversation_id)

        # Build unified field configs. DECISION
        # plan-2026-09-19T175721-21cd7f8e/D-033: a `handler_only_keys` name
        # gets no config even when a transition reads it; a key the LLM may
        # not write must not be minted into a per-field extraction.
        handler_only = self._handler_only_keys(instance)
        all_configs = [
            c
            for c in self._build_field_configs_from_state(current_state)
            if c.field_name not in handler_only
        ]

        has_field_configs = bool(all_configs)
        has_classification_configs = bool(
            self._classification_configs(current_state, instance)
        )

        has_extraction_instructions = bool(current_state.extraction_instructions)

        bulk_failed = False
        if not has_field_configs and not has_classification_configs:
            if has_extraction_instructions:
                # Fallback: bulk extraction for states with instructions
                # but no explicit field configs.  Uses a single LLM call
                # to extract any relevant data the instructions describe.
                log.debug(
                    "No field configs but extraction_instructions present; "
                    "using bulk extraction fallback"
                )
                bulk_data = self._bulk_extract_from_instructions(
                    instance, user_message, current_state, conversation_id
                )
                bulk_failed = isinstance(bulk_data, _BulkFailed)
                # Don't overwrite values already set in context (e.g. by
                # handlers) — bulk extraction is best-effort for NEW data.
                if bulk_data:
                    existing = instance.context.data
                    bulk_data = {
                        k: v for k, v in bulk_data.items() if existing.get(k) is None
                    }
                if bulk_data:
                    response = DataExtractionResponse(
                        extracted_data=bulk_data,
                        confidence=0.8,
                    )
                    instance.last_extraction_response = response
                    return response

            log.debug("No fields or classifications to extract for this state")
            response = DataExtractionResponse(
                extracted_data={}, confidence=1.0, extraction_failed=bulk_failed
            )
            instance.last_extraction_response = response
            return response

        extracted_data: dict[str, Any] = {}
        rejected: dict[str, Any] = {}
        confidences: list[float] = []
        cfg_by_name: dict[str, FieldExtractionConfig] = {}
        # Bound on every path: the retry loop below reads it.
        memo: dict[tuple[str, str], FieldExtractionResponse] | None = None

        # --- Field extractions ---
        if has_field_configs:
            # Skip fields already set in context (e.g. by handlers). The names
            # are captured first: the additive bulk pass below needs the
            # config-covered set, not the filtered one (D-004, D-015).
            cfg_by_name = {c.field_name: c for c in all_configs}
            # DECISION plan-2026-09-19T175721-21cd7f8e/D-035: identical null
            # extractions are memoised ONLY on Ollama (temperature 0, so an
            # identical prompt returns an identical result and a retry is
            # pure waste). Elsewhere a retry is a deliberate resample: do NOT
            # widen this to every provider, and do NOT memoise a success. The
            # `isinstance(str)` test is load-bearing: a Mock interface's
            # `.model` is truthy and must not enable it.
            model = getattr(self._services.llm_interface, "model", None)
            memo = {} if isinstance(model, str) and is_ollama_model(model) else None
            existing = instance.context.data
            # DECISION plan-2026-09-19T175721-21cd7f8e/D-046: for an agent FSM an
            # EMPTY list/dict counts as unset. A builder seeds `[]` so a
            # has_context gate can pass (plan_execute `plan_steps`); without this
            # the seed hides the key from extraction. Non-agent FSMs keep "empty
            # is set": do NOT widen this to every FSM (D-036 rejected that).
            agent_managed = _is_agent_managed(existing)
            all_configs = [
                c
                for c in all_configs
                if existing.get(c.field_name) is None
                or (agent_managed and existing[c.field_name] in ([], {}))
            ]
            results = self._execute_field_extractions(
                instance, user_message, all_configs, conversation_id, memo
            )
            for result in results:
                if result.is_valid and result.value is not None:
                    extracted_data[result.field_name] = result.value
                    confidences.append(result.confidence)

            log.debug(
                f"Field extraction pass 1: "
                f"{list(extracted_data.keys()) or 'no data'}, "
                f"min_confidence={min(confidences) if confidences else 0.0:.2f}"
            )

        # --- Classification extractions ---
        if has_classification_configs:
            classification_data = self._execute_classification_extractions(
                current_state, user_message, instance, conversation_id
            )
            extracted_data.update(classification_data)

        # --- Multi-pass retry for missing required fields ---
        max_retries = current_state.extraction_retries
        if max_retries > 0:
            for retry_num in range(1, max_retries + 1):
                existing_context = instance.context.data

                # Find missing required field configs
                missing_field_configs = (
                    [
                        cfg
                        for cfg in all_configs
                        if cfg.required
                        and cfg.field_name not in extracted_data
                        and cfg.field_name not in existing_context
                    ]
                    if has_field_configs
                    else []
                )

                # Find missing required classification configs
                missing_class_configs = (
                    [
                        cfg
                        for cfg in self._classification_configs(current_state, instance)
                        if cfg.required
                        and cfg.field_name not in extracted_data
                        and cfg.field_name not in existing_context
                    ]
                    if has_classification_configs
                    else []
                )

                if not missing_field_configs and not missing_class_configs:
                    break

                missing_names = [c.field_name for c in missing_field_configs] + [
                    c.field_name for c in missing_class_configs
                ]
                log.info(
                    f"Extraction retry {retry_num}/{max_retries}: "
                    f"missing={missing_names}"
                )

                if missing_field_configs:
                    retry_results = self._execute_field_extractions(
                        instance,
                        user_message,
                        missing_field_configs,
                        conversation_id,
                        memo,
                    )
                    for result in retry_results:
                        if result.is_valid and result.value is not None:
                            extracted_data[result.field_name] = result.value
                            confidences.append(result.confidence)

                if missing_class_configs:
                    retry_class_data = self._execute_classification_extractions(
                        current_state,
                        user_message,
                        instance,
                        conversation_id,
                        configs_override=missing_class_configs,
                    )
                    extracted_data.update(retry_class_data)

        # --- Additive bulk extraction for instruction-only fields ---
        # DECISION plan-2026-09-19T175721-21cd7f8e/D-015 [supersedes
        # plan-2026-09-19T175721-21cd7f8e/D-004's overwrite condition and
        # plan_2026-05-30_26c9510a/D-001 "merge ONLY keys still absent"]:
        # fields named only in extraction_instructions never get a
        # FieldExtractionConfig, so the per-field passes silently miss them
        # (~50% extraction on multi-field states); this best-effort bulk pass
        # adds them. It is also the only channel for a later-turn correction
        # (LV-03), so a config-covered key may be overwritten, but ONLY while
        # it still holds exactly what the pipeline extracted (digest recorded
        # at the two commit sites). Do NOT go back to D-004's `key in
        # config_names` rule: it let one bulk turn flip a handler-seeded gate
        # value (repro5). A config-covered bulk value is coerced/validated
        # first (never a dict in a str key, never 24 -> "24"); instruction-only
        # keys stay raw and skip-if-set. Agent FSMs (`agent_trace`) keep
        # handler-set state in config-covered keys, so they never overwrite.
        # Do NOT re-extract per field every turn (1+ LLM call per field per
        # turn). Provenance is persisted by save_session and re-seeded by
        # restore_session as JSON-native digests (D-031 supersedes clause (3)
        # "not persisted"): a file without it restores an empty map and fails
        # closed, and the digest is still compared to the stored value, so a
        # handler-seeded or update_context value is never overwritten. Bulk
        # values validate at a fixed confidence 1.0, a classification-owned key
        # is not correctable, and the digest map is visible in
        # get_complete_conversation()['metadata'] and the session file (same
        # trust domain as context_data). See decisions.md D-015, D-031. The
        # no-config fallback above keeps skip-if-set.
        if has_extraction_instructions and (
            has_field_configs or has_classification_configs
        ):
            bulk_data = self._bulk_extract_from_instructions(
                instance, user_message, current_state, conversation_id
            )
            bulk_failed = isinstance(bulk_data, _BulkFailed)
            if bulk_data:
                existing = instance.context.data
                agent_managed = _is_agent_managed(existing)
                prov = instance.context.metadata.get(PROVENANCE_METADATA_KEY, {})
                # DECISION plan-2026-09-19T175721-21cd7f8e/D-019: a
                # classification-owned key absent because the classifier was
                # below threshold must stay absent (ra02: bulk "buy" bypassed
                # the 0.7 gate). Do NOT apply this to agent FSMs: with
                # use_classification=True `tool_name` is classification-owned
                # AND relies on this fill when the classifier declines.
                owned = (
                    set()
                    if agent_managed
                    else {
                        c.field_name
                        for c in (current_state.classification_extractions or [])
                    }
                )
                for key, value in bulk_data.items():
                    if value is None or key in extracted_data or key in owned:
                        continue
                    cfg = cfg_by_name.get(key)
                    if cfg is not None:
                        checked = self._validate_field_extraction(
                            FieldExtractionResponse(
                                field_name=key, value=value, confidence=1.0
                            ),
                            cfg,
                        )
                        value = checked.value
                        if (
                            not checked.is_valid
                            or value is None
                            or (isinstance(value, dict) and cfg.field_type != "dict")
                        ):
                            continue
                    current = existing.get(key)
                    if current is None:
                        extracted_data[key] = value
                        log.debug(f"Bulk extraction added missing field: {key}")
                    elif (
                        user_message is not None
                        and not agent_managed
                        and str(value).strip().lower() != str(current).strip().lower()
                    ):
                        # plan-2026-09-30T062855-07ad3f8c/D-023: a correction
                        # is something a user message asks for. With no
                        # message a set key is neither overwritten nor
                        # reported as a rejected correction.
                        # DECISION plan-2026-09-19T175721-21cd7f8e/D-052 (4):
                        # an instruction-only key (no config) is never
                        # corrected (skip-if-set) but IS reported below under
                        # the same grounding test. Do NOT correct it here.
                        # LV6-01: a back edge that then applies the value
                        # prunes this entry (post_transition.py, LV6-01).
                        if cfg is not None and prov.get(key) == _value_digest(current):
                            extracted_data[key] = value
                            log.debug(f"Bulk extraction corrected field: {key}")
                        else:
                            # DECISION plan-2026-09-19T175721-21cd7f8e/D-032:
                            # a refused overwrite (handler-set, update_context
                            # or unpersisted value, D-015) that the USER asked
                            # for is carried to Pass 2 so the reply does not
                            # claim a change the store never made (LV3-01).
                            # Do NOT report every differing bulk value: an
                            # ungrounded one ("navy" after "thanks") would put
                            # a phantom correction in the prompt.
                            # DECISION plan-2026-09-19T175721-21cd7f8e/D-049
                            # The value must be a WHOLE token of the message and
                            # at least 3 characters: a substring test let `red`
                            # ground on "bored now" and `500` on "1500 items".
                            # Do NOT use plain `in` again, and do NOT use `\b`
                            # (a value like `c++` ends in a non-word character;
                            # lookarounds work for it). Cost: a real 1-2
                            # character correction (`US`, `42`) never produces
                            # the block, the quiet pre-D-032 behaviour.
                            needle = str(value).strip().lower()
                            if len(needle) >= 3 and re.search(
                                rf"(?<!\w){re.escape(needle)}(?!\w)",
                                user_message.lower(),
                            ):
                                rejected[key] = value
                                log.debug(f"Bulk correction rejected: {key}")

        min_confidence = min(confidences) if confidences else 0.0
        response = DataExtractionResponse(
            extracted_data=extracted_data,
            confidence=min_confidence,
            rejected_corrections=rejected,
            extraction_failed=bulk_failed,
        )
        instance.last_extraction_response = response

        log.debug(
            f"Data extraction complete: "
            f"{list(extracted_data.keys())}, "
            f"confidence={min_confidence:.2f}"
        )
        return response
