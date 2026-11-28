"""Transition evaluation of ``MessagePipeline`` (``TransitionsMixin``): rule evaluation, the
classifier tie-break on AMBIGUOUS, the classifier cache and the state transition.

Layer ``engine`` (plan d454fd62, step 79): split out of the old ``fsm_llm.pipeline``; the
method and function bodies are unchanged.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from typing import Any

from pydantic import ValidationError

from fsm_llm.aio import acall, run_sync
from fsm_llm.constants import (
    CLASSIFIER_HISTORY_EXCHANGES,
    DEFAULT_TRANSITION_CLASSIFICATION_CONFIDENCE,
    MAX_CLASSIFIER_CACHE_SIZE,
    TRANSITION_CLASSIFICATION_FALLBACK_INTENT,
    TRANSITION_CLASSIFICATION_THRESHOLD_KEY,
)
from fsm_llm.core.codes import Purpose
from fsm_llm.core.definitions import (
    ClassificationError,
    ClassificationPromptConfig,
    ClassificationResult,
    ClassificationSchema,
    FSMInstance,
    IntentDefinition,
    InvalidTransitionError,
    State,
    TransitionEvaluation,
    TransitionEvaluationResult,
    TransitionOption,
)
from fsm_llm.engine.classification import Classifier
from fsm_llm.engine.kinds.records import (
    _CLASSIFICATION_SOFT_FAIL_EXCEPTIONS,
    _classification_record,
)
from fsm_llm.handlers import HandlerTiming
from fsm_llm.llm import implements_complete
from fsm_llm.llm.purpose import purpose_scope
from fsm_llm.logging import logger

from .events import record_transition
from .records import (
    _record_transition_classification,
    _TurnRecord,
    _TurnSnapshot,
)
from .surface import _PipelineSurface


class TransitionsMixin(_PipelineSurface):
    """Transition evaluation, the classifier cache and the state transition."""

    # ----------------------------------------------------------
    # Pass 1: Transition evaluation and execution
    # ----------------------------------------------------------

    def _execute_transition_evaluation_and_execution(
        self,
        instance: FSMInstance,
        user_message: str | None,
        conversation_id: str,
        *,
        turn: _TurnRecord | None = None,
    ) -> tuple[bool, str | None]:
        """``run_sync`` over ``_aexecute_transition_evaluation_and_execution``, the one body."""
        return run_sync(
            self._aexecute_transition_evaluation_and_execution(
                instance, user_message, conversation_id, turn=turn
            )
        )

    async def _aexecute_transition_evaluation_and_execution(
        self,
        instance: FSMInstance,
        user_message: str | None,
        conversation_id: str,
        *,
        turn: _TurnRecord | None = None,
    ) -> tuple[bool, str | None]:
        """Evaluate transitions and execute if one is selected.

        ``turn``, when given, receives ``evaluation.result_type``.
        """
        log = logger.bind(conversation_id=conversation_id)
        log.debug("Executing transition evaluation and execution")

        current_state = self.get_state(instance, conversation_id)

        if not current_state.transitions:
            log.debug("Terminal state reached - no transitions to evaluate")
            return False, None

        previous_state_id = instance.current_state

        # DECISION plan-2026-11-05T134105-5ed71cf2/D-005
        # The gate reads the COMMITTED context only. Everything extracted this
        # turn was committed by `_execute_extraction_and_transition_pass`
        # (None and empty values dropped first), and the CONTEXT_UPDATE
        # handlers have run on it, so `instance.context` is the one source of
        # truth. Do NOT pass `extraction_response.extracted_data` as a third
        # argument: the evaluator would re-apply the model's value over a
        # handler's overwrite or delete, and the gate would decide on a value
        # the store no longer holds (a validator or veto could not guard the
        # transition it exists for). The evaluator keeps the parameter for
        # direct callers; the pipeline does not use it.
        evaluation = self.transition_evaluator.evaluate_transitions(
            current_state, instance.context
        )
        # DECISION plan-2026-09-30T062855-07ad3f8c/D-020
        # The outcome travels on the per-turn record. Do NOT widen this
        # method's or the Pass-1 method's return tuple to carry it: regression
        # tests replace and unpack both with the (bool, previous_state) and
        # 3-tuple shapes. Do NOT store it on ``FSMInstance`` or the pipeline
        # (a public model field, or state shared between conversations).
        if turn is not None:
            turn.transition_outcome = evaluation.result_type

        target_state = None

        if evaluation.result_type == TransitionEvaluationResult.DETERMINISTIC:
            target_state = evaluation.deterministic_transition
            log.info(f"Deterministic transition selected: {target_state}")

        elif evaluation.result_type == TransitionEvaluationResult.AMBIGUOUS:
            target_state = await self._aresolve_ambiguous_transition(
                evaluation, user_message, instance, conversation_id
            )
            log.info(
                f"LLM-assisted transition selected: {target_state}"
                if target_state
                else "LLM-assisted resolution selected no transition (stay)"
            )

        elif evaluation.result_type == TransitionEvaluationResult.BLOCKED:
            log.warning(f"Transitions blocked: {evaluation.blocked_reason}")
            record_transition(current_state, evaluation, None, executed=False)
            return False, None

        if target_state:
            self._execute_state_transition(instance, target_state, conversation_id)
            record_transition(current_state, evaluation, target_state, executed=True)
            return True, previous_state_id

        record_transition(current_state, evaluation, None, executed=False)
        return False, None

    # ----------------------------------------------------------
    # Classification-based extraction
    # ----------------------------------------------------------

    def _get_classifier(
        self,
        schema: ClassificationSchema,
        prompt_config: ClassificationPromptConfig | None,
        *,
        model: str | None = None,
    ) -> Classifier:
        """Return the cached ``Classifier`` for this exact configuration, or build it.

        Contract: ``schema`` is the intent schema, ``prompt_config`` the
        per-entry prompt override (or ``None``) and ``model`` a per-entry model
        override (or ``None``). A ``model`` that is ``None`` or equal to the
        conversation interface's ``model`` gives a classifier that sends
        through ``self.llm_interface`` (a custom interface is honoured, its
        meter counts the call). A ``model`` that differs gives a classifier
        with its own interface for that model, built from the model name only
        (no api_key, api_base or timeout of the conversation's interface is
        inherited). Returns a classifier whose inputs are content-equal to the
        arguments and, when it sends through the conversation's interface,
        built over the interface object bound now; it never returns one built
        for another schema, prompt config, model or interface. Cache misses
        construct via the module-level ``Classifier`` symbol (so
        ``patch("fsm_llm.engine.turn.transitions.Classifier")`` keeps observing construction)
        and evict the oldest entry when the cache holds
        ``MAX_CLASSIFIER_CACHE_SIZE`` entries. Raises ``ClassificationError``
        (soft at both call sites) when the classifier would send through a
        conversation interface that does not implement ``complete``; other
        construction errors propagate to the caller unchanged.

        Thread safety: a ``MessagePipeline`` is shared across conversations,
        so the hit check, the construction, the eviction and the insert run
        under ``self._classifier_cache_lock`` as one unit. The unlocked
        version raced ``cache.pop(next(iter(cache)))`` against a concurrent
        insert into ``RuntimeError: dictionary changed size during
        iteration`` (reproduced by the iteration-1 review, W2, and pinned by
        ``test_eviction_race_two_threads``). Holding the lock across
        ``Classifier(...)`` is cheap: construction is pure CPU (prompt and
        schema building, no network call), and it also removes the
        double-construct on a shared miss.
        """
        # DECISION plan-2026-10-01T093600-944e2692/D-006 (supersedes
        # 21cd7f8e/D-008 and the identity rule of 0d9c218e/D-001): the
        # classifier sends through the conversation's own `llm_interface`
        # unless the entry names a different model. Do NOT rebuild a private
        # interface from the conversation interface's kwargs (that bypassed a
        # custom `llm_interface` and its meter), and do NOT pass that
        # interface's api_key/api_base/timeout to a classifier for another
        # model (credentials would reach another provider). The key is the
        # content (schema, prompt config, override model) plus `id()` of the
        # injected interface: the cached Classifier holds that interface, so
        # its id cannot be reused by another object while the entry lives,
        # and rebinding `self.llm_interface` misses the cache instead of
        # returning a classifier over the old interface. Do NOT put per-call
        # prompt or context in the key (8a03483a/D-004) or key on the
        # state/field names, and do NOT put connection settings in it (none
        # reach the classifier, and secrets must stay out of keys and logs).
        # The check/construct/evict/insert sequence is ONE critical section
        # under `_classifier_cache_lock`: do NOT narrow the lock to the dict
        # writes, the FIFO `next(iter())` eviction is what raced (review W2).
        # See decisions.md D-006.
        llm = self.llm_interface
        interface_model = getattr(llm, "model", None)
        own_model = model if model and model != interface_model else None
        # DECISION plan-2026-10-01T093600-944e2692/D-046: an interface without
        # `complete` cannot classify, which is an outage of this turn's
        # classification, not a bug: ClassificationError (a tie stays, a field
        # is skipped, WARNING). Do NOT let Classifier's ValueError escape (a
        # tie state became a dead end, review pass 11 W1) and do NOT widen the
        # soft-fail tuple (D-039: a bug inside a real `complete` propagates).
        payload = json.dumps(
            {
                "schema": schema.model_dump(),
                "config": (
                    dataclasses.asdict(prompt_config) if prompt_config else None
                ),
                "model": own_model,
                "interface": None if own_model else id(llm),
            },
            sort_keys=True,
        )
        key = hashlib.sha256(payload.encode()).hexdigest()
        cache = self._classifier_cache
        with self._classifier_cache_lock:
            hit = cache.get(key)
            if hit is not None:
                return hit
            if own_model:
                classifier = Classifier(
                    schema=schema, model=own_model, config=prompt_config
                )
            else:
                try:
                    classifier = Classifier(
                        schema=schema, llm_interface=llm, config=prompt_config
                    )
                except ValueError as e:
                    if implements_complete(llm):
                        raise
                    raise ClassificationError(
                        f"{type(llm).__name__} does not implement complete "
                        "(LLMInterface.complete), so it cannot classify"
                    ) from e
            if len(cache) >= MAX_CLASSIFIER_CACHE_SIZE:
                cache.pop(next(iter(cache)), None)
            cache[key] = classifier
        return classifier

    def _resolve_ambiguous_transition(
        self,
        evaluation: TransitionEvaluation,
        user_message: str | None,
        instance: FSMInstance,
        conversation_id: str,
    ) -> str | None:
        """``run_sync`` over ``_aresolve_ambiguous_transition``, the one body."""
        return run_sync(
            self._aresolve_ambiguous_transition(
                evaluation, user_message, instance, conversation_id
            )
        )

    async def _aresolve_ambiguous_transition(
        self,
        evaluation: TransitionEvaluation,
        user_message: str | None,
        instance: FSMInstance,
        conversation_id: str,
    ) -> str | None:
        """Resolve ambiguous transition using classification.

        Classification is always-on for ambiguous transitions. Builds a
        ClassificationSchema from available transition options and uses
        the Classifier to make a structured, confidence-scored decision.
        ``user_message`` of ``None`` (no user message) makes the classifier
        decide among the tied options from the per-call context alone.

        Returns the selected target state id, or ``None`` when no transition
        should happen (classifier failure, or the fallback intent). The caller
        treats any truthy return as a transition, so "stay" must be ``None``
        and never ``instance.current_state``.
        """
        log = logger.bind(conversation_id=conversation_id)
        log.debug(
            f"Resolving ambiguous transition with {len(evaluation.available_options)} options"
        )

        current_state = self.get_state(instance, conversation_id)

        try:
            # Inside the try, as at the extraction site: a construction
            # failure degrades to "stay" like a classify() failure (review W2).
            # The schema build is in it too: a tie set the schema cannot hold
            # arrives as ClassificationError (plan-2026-11-05T134105-5ed71cf2/
            # D-006).
            schema = self._build_transition_classification_schema(
                current_state,
                evaluation.available_options,
            )
            classifier = self._get_classifier(schema, None)
            # the tie-break shares the `classification` call type with the classifier of
            # `classification_extractions`: only this scope tells them apart in the log
            with purpose_scope(Purpose.TIE_BREAK):
                result: ClassificationResult = await acall(
                    classifier,
                    "classify",
                    user_message,
                    context=self._build_classifier_context(
                        instance,
                        current_state,
                        conversation_id,
                        user_message=user_message,
                    ),
                )
        except _CLASSIFICATION_SOFT_FAIL_EXCEPTIONS as e:
            # DECISION plan-2026-09-20T165703-0d9c218e/D-001: "stay" is the
            # degrade ONLY for the classes the shared tuple names (the D-004
            # set of plan-2026-07-19T191147-4b664252, reused unchanged).
            # A programming error (AttributeError etc.) propagates out of the
            # turn; it is not a classifier outage. Do not widen this back to
            # `except Exception` -- the old broad catch predated D-004 and no
            # test pinned it. See decisions.md D-001.
            log.warning(
                f"Classification failed during ambiguous transition resolution: {e}"
            )
            log.warning("Falling back to current state (no transition)")
            _record_transition_classification(
                instance, {"error": str(e), "fallback": True}
            )
            # DECISION plan-2026-09-19T175721-21cd7f8e/D-007: "stay" is None,
            # not instance.current_state. Returning the current state made the
            # caller run PRE/POST_TRANSITION handlers and tell Pass 2 a
            # transition occurred. Do NOT "fix" this with a
            # `target != current_state` guard at the caller: declared explicit
            # self-loops are design and must keep firing.
            return None

        log.debug(
            f"Classification result: intent={result.intent}, "
            f"confidence={result.confidence:.2f}, reasoning={result.reasoning}"
        )

        # DECISION plan-2026-09-21T203800-8a03483a/D-002: a result below
        # schema.confidence_threshold means "stay" (A1). Compare against the
        # schema built above, the same `<` rule as Classifier.is_low_confidence
        # and the extraction site; do NOT call classifier.is_low_confidence()
        # here, because a patched Classifier returns a truthy mock and would
        # silently turn every mocked transition into a stay. Do NOT route this
        # through an exception: the soft-fail tuple above stays closed.
        record = _classification_record(result, schema.confidence_threshold)
        low_confidence = "low_confidence" in record

        # Store the classification record in metadata
        # (plan-2026-09-21T203800-8a03483a/D-005).
        _record_transition_classification(instance, record)

        # Store as transition decision for debugging
        instance.last_transition_decision = result

        if low_confidence:
            log.warning(
                f"Transition classification below threshold in state "
                f"'{current_state.id}': intent={result.intent}, "
                f"confidence={result.confidence:.2f}, "
                f"threshold={schema.confidence_threshold} -- staying in state"
            )
            # D-007 of plan-2026-09-19T175721-21cd7f8e: stay is None.
            return None

        # Handle fallback intent (low confidence or unknown) — stay in current state
        if result.intent == TRANSITION_CLASSIFICATION_FALLBACK_INTENT:
            log.info(
                "Classification returned fallback intent — staying in current state"
            )
            # DECISION plan-2026-09-19T175721-21cd7f8e/D-007: None, not the
            # current state (see the classifier-failure return above).
            return None

        # Validate the classified intent is a valid target state
        valid_targets = {opt.target_state for opt in evaluation.available_options}
        if result.intent not in valid_targets:
            raise InvalidTransitionError(
                f"Classification returned unknown target '{result.intent}'. "
                f"Valid options: {sorted(valid_targets)}"
            )

        log.info(
            f"Classification-based transition selected: {result.intent} "
            f"(confidence={result.confidence:.2f})"
        )
        return result.intent

    def _build_classifier_context(
        self,
        instance: FSMInstance,
        state: State,
        conversation_id: str,
        context_keys: list[str] | None = None,
        *,
        user_message: str | None,
    ) -> dict[str, Any]:
        """Per-call classifier context: ``{"history", "purpose", "data"}``.

        ``history`` is the last ``CLASSIFIER_HISTORY_EXCHANGES`` exchanges
        without the in-flight user message (the classifier receives that as
        its user turn). With ``user_message`` of ``None`` no exchange is in
        flight, so a trailing user entry is a past message and stays. ``data`` is ``get_user_visible_data()`` (WorkingMemory
        included, data wins) scoped by the state's ``read_keys`` and then, when
        given, narrowed to ``context_keys`` (the intersection: ``context_keys``
        can never expose a key ``read_keys`` hides). Security filtering,
        sanitization and the per-line history cap happen where it is rendered
        (``prompts.build_classification_context_block``).
        """
        # DECISION plan-2026-09-21T203800-8a03483a/D-004: one builder for
        # both classifier call sites (extraction and ambiguous transition).
        # Do NOT pass raw instance.context.data or skip the scope: read_keys
        # must hide from the classifier what it hides from Pass 2, and the
        # result is per-call input to classify(), never a cache-key input.
        history = instance.context.conversation.get_recent(CLASSIFIER_HISTORY_EXCHANGES)
        if user_message is not None and history and "user" in history[-1]:
            history = history[:-1]
        # DECISION plan-2026-09-21T203800-8a03483a/D-047: read_keys ALWAYS
        # applies first; context_keys only narrows it. Do NOT return to
        # "context_keys replaces read_keys" (it sent a read_keys-hidden salary
        # to the classifier). See decisions.md D-047.
        data = self._apply_context_scope(
            instance.context.get_user_visible_data(),
            state,
            conversation_id,
            context_keys,
        )
        return {"history": history, "purpose": state.purpose, "data": data}

    # ----------------------------------------------------------
    # Classification schema builder
    # ----------------------------------------------------------

    @staticmethod
    def _build_transition_classification_schema(
        state: State,
        options: list[TransitionOption],
    ) -> ClassificationSchema:
        """Build a ClassificationSchema from transition options.

        If the state has a custom ``transition_classification`` dict config,
        merges user-provided descriptions and thresholds. Otherwise
        auto-generates intents from transition descriptions.
        """
        config = state.transition_classification

        if isinstance(config, dict):
            # Manual mode: user provides intent descriptions
            intents = []
            for opt in options:
                custom = config.get(opt.target_state, {})
                description = (
                    custom.get("description")
                    or opt.description
                    or f"Transition to {opt.target_state}"
                )
                intents.append(
                    IntentDefinition(name=opt.target_state, description=description)
                )
            confidence_threshold = config.get(
                TRANSITION_CLASSIFICATION_THRESHOLD_KEY,
                DEFAULT_TRANSITION_CLASSIFICATION_CONFIDENCE,
            )
        else:
            # Auto mode: generate from transition option descriptions
            intents = []
            for opt in options:
                description = opt.description or f"Transition to {opt.target_state}"
                intents.append(
                    IntentDefinition(name=opt.target_state, description=description)
                )
            confidence_threshold = DEFAULT_TRANSITION_CLASSIFICATION_CONFIDENCE

        # Add fallback intent for low-confidence cases
        intents.append(
            IntentDefinition(
                name=TRANSITION_CLASSIFICATION_FALLBACK_INTENT,
                description="None of the above options clearly match the user's intent",
            )
        )

        # DECISION plan-2026-11-05T134105-5ed71cf2/D-006: a tie set the schema
        # cannot hold (a target named like the fallback intent, 15 or more
        # distinct tied targets) is a ClassificationError, so the turn stays
        # with a WARNING instead of failing. Do NOT truncate to 14 candidates
        # (it picks winners by definition order) and do NOT add ValidationError
        # to _CLASSIFICATION_SOFT_FAIL_EXCEPTIONS (0d9c218e/D-001: the tuple
        # stays closed; a programming error must still propagate).
        try:
            return ClassificationSchema(
                intents=intents,
                fallback_intent=TRANSITION_CLASSIFICATION_FALLBACK_INTENT,
                confidence_threshold=confidence_threshold,
            )
        except ValidationError as e:
            raise ClassificationError(
                f"Cannot classify the tied transitions from state '{state.id}': {e}"
            ) from e

    def _execute_state_transition(
        self, instance: FSMInstance, target_state: str, conversation_id: str
    ) -> None:
        """Execute state transition with PRE/POST handler integration and rollback."""
        log = logger.bind(conversation_id=conversation_id)
        old_state = instance.current_state

        self.execute_handlers(
            instance,
            HandlerTiming.PRE_TRANSITION,
            current_state=old_state,
            target_state=target_state,
        )

        # Deep-copy full context for rollback if POST_TRANSITION handlers fail.
        #
        # DECISION plan-2026-09-20T114608-a8e47b88/D-024
        # Snapshot AND restore instance.context.metadata alongside
        # instance.context.data, mirroring the turn-atomicity rollback pattern
        # already used correctly at the turn-level sites (search
        # `_TurnSnapshot` in `records.py`: its `restore` clear()+update()s data
        # AND metadata together from one pre-captured deepcopy pair). Before D-018, `merge_delta` never
        # touched `metadata`, so a data-only snapshot here was safe -- D-018
        # made `merge_delta` ALSO pop a deleted key's provenance digest out of
        # `context.metadata[PROVENANCE_METADATA_KEY]` on a None-delta deletion. So a
        # POST_TRANSITION handler chain where an earlier handler deletes a
        # provenanced key (clearing its digest) and a LATER handler at the
        # same timing raises left the plaintext key restored on rollback but
        # its digest permanently gone -- data and metadata desynced. Do NOT
        # snapshot `data` only "because metadata isn't touched here" -- that
        # premise is no longer true. See decisions.md D-024.
        snapshot = _TurnSnapshot.capture(instance, whole_turn=False)

        instance.current_state = target_state
        instance.context.data.update(
            {
                "_previous_state": old_state,
                "_current_state": target_state,
                "_transition_timestamp": time.time(),
            }
        )

        try:
            self.execute_handlers(
                instance,
                HandlerTiming.POST_TRANSITION,
                current_state=target_state,
                target_state=target_state,
            )
        except Exception as handler_err:
            log.warning(
                f"POST_TRANSITION handler failed ({type(handler_err).__name__}: {handler_err}), rolling back state from {target_state} to {old_state}"
            )
            snapshot.restore(instance)
            raise

        log.info(f"State transition executed: {old_state} -> {target_state}")
