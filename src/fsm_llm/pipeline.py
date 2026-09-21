"""
MessagePipeline: The 2-pass message processing engine.

Encapsulates all LLM-driven processing logic extracted from FSMManager:
- Pass 1: Data extraction + transition evaluation + state transition
- Pass 2: Response generation from final state
- Handler execution bridge (deep-copy context, merge deltas)

FSMManager delegates to this class for all message processing.
The pipeline does not own instances or locks — those remain in FSMManager.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import re
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

from .classification import Classifier
from .constants import (
    CLASSIFIER_HISTORY_EXCHANGES,
    CONTEXT_KEY_AGENT_TRACE,
    CONTEXT_KEY_CLASSIFICATION_RESULT,
    DEFAULT_TRANSITION_CLASSIFICATION_CONFIDENCE,
    MAX_CLASSIFIER_CACHE_SIZE,
    METADATA_KEY_CLASSIFICATION_RESULTS,
    METADATA_KEY_TRANSITION_CLASSIFICATION,
    TRANSITION_CLASSIFICATION_FALLBACK_INTENT,
    is_forbidden_context_entry,
)
from .context import clean_context_keys
from .definitions import (
    BulkExtractionRequest,
    ClassificationError,
    ClassificationExtractionConfig,
    ClassificationResult,
    ClassificationSchema,
    DataExtractionResponse,
    FieldExtractionConfig,
    FieldExtractionRequest,
    FieldExtractionResponse,
    FSMDefinition,
    FSMInstance,
    IntentDefinition,
    InvalidTransitionError,
    ResponseGenerationRequest,
    State,
    StateNotFoundError,
    TransitionEvaluation,
    TransitionEvaluationResult,
    TransitionOption,
)
from .handlers import HandlerExecutionError, HandlerSystem, HandlerTiming
from .llm import LLMInterface
from .logging import logger
from .ollama import is_ollama_model
from .prompts import (
    ClassificationPromptConfig,
    DataExtractionPromptBuilder,
    FieldExtractionPromptBuilder,
    ResponseGenerationPromptBuilder,
)
from .transition_evaluator import TransitionEvaluator

# --- Type coercion dispatch for field extraction validation ---


def _coerce_int(v: Any) -> int:
    return v if isinstance(v, int) else int(v)


def _coerce_float(v: Any) -> float:
    return v if isinstance(v, float) else float(v)


# DECISION plan-2026-09-19T175721-21cd7f8e/D-002
# `_coerce_str`/`_coerce_bool` DO raise `TypeError` on a dict or list. This supersedes
# the last clause of plan-2026-07-18T051819-80b0bd4d/D-018 (below), which said they
# "deliberately do NOT raise". Do NOT restore total `str(v)`/`bool(v)` on containers:
# live (LV-01) qwen3.5:9b returned `{"blue": "blue"}` for a str field, `str()` turned
# it into the "valid" value `"{'blue': 'blue'}"`, and `bool({...})` is True for any
# non-empty object, so junk was stored and drove gated transitions. D-018's premise
# ("coercion failed is not a reachable outcome") held for scalars only. The raise
# reuses the already-wired `except (ValueError, TypeError, json.JSONDecodeError)` in
# `_validate_field_extraction` (no sentinel, no new exception type). See decisions.md D-002.
def _coerce_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("true", "1", "yes")
    if isinstance(v, (dict, list)):
        raise TypeError(f"expected bool, got {type(v).__name__}")
    return bool(v)


def _coerce_str(v: Any) -> str:
    if isinstance(v, (dict, list)):
        raise TypeError(f"expected str, got {type(v).__name__}")
    return v if isinstance(v, str) else str(v)


# DECISION plan-2026-07-18T051819-80b0bd4d/D-018 [STALE]
# The trailing `raise TypeError` in each of the two coercers below is LOAD-BEARING.
# Do NOT "restore" a `return v` passthrough for wrong-typed input. Both functions used
# to end in a bare `return v`, so an int or a dict handed to a `field_type="list"` field
# sailed through unconverted, `_validate_field_extraction`'s
# `except (ValueError, TypeError, json.JSONDecodeError)` (see the sole call site, further
# down this file) never fired, and the field was recorded `is_valid=True` with a
# wrong-typed value written straight into FSM context. Raising here reuses that
# ALREADY-WIRED error protocol — do not invent a new exception type, a sentinel return,
# or a second validation path. This makes `list`/`dict` behave like their 4 siblings.
# [SUPERSEDED by plan-2026-09-19T175721-21cd7f8e/D-002, above, for dict/list input:]
# `_coerce_str`/`_coerce_bool` deliberately do NOT raise: `str()`/`bool()` are TOTAL, so
# "coercion failed" is not a reachable outcome for them, not a missing guard.
# `None` never reaches any coercer (the call site returns early on a None value).
# See decisions.md D-018.
def _coerce_list(v: Any) -> Any:
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        parsed = json.loads(v)
        if not isinstance(parsed, list):
            raise TypeError("not a list")
        return parsed
    raise TypeError(f"expected list, got {type(v).__name__}")


def _coerce_dict(v: Any) -> Any:
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        parsed = json.loads(v)
        if not isinstance(parsed, dict):
            raise TypeError("not a dict")
        return parsed
    raise TypeError(f"expected dict, got {type(v).__name__}")


_TYPE_COERCERS: dict[str, Callable[[Any], Any]] = {
    "int": _coerce_int,
    "float": _coerce_float,
    "bool": _coerce_bool,
    "str": _coerce_str,
    "list": _coerce_list,
    "dict": _coerce_dict,
    # "any" — no coercion, not in dispatch dict
}

# Keyword names `Classifier.__init__` binds itself; an interface kwarg with one
# of these names must never be spread into `Classifier(...)`.
_CLASSIFIER_BOUND_NAMES = frozenset({"schema", "model", "config"})

# `context.metadata` key holding {context key: _value_digest(value)} for every
# value the pipeline itself extracted (never the values). D-015.
_PROVENANCE_KEY = "_pipeline_extracted"

# DECISION plan-2026-09-20T165703-0d9c218e/D-001
# The ONE exception tuple both classifier call sites degrade on (D-004 of
# plan-2026-07-19T191147-4b664252 chose it for
# `_execute_classification_extractions`; this plan's D-001 extends it to
# `_resolve_ambiguous_transition`). A soft failure means: the classification
# extraction is skipped (or raised as ClassificationError when required), and an
# ambiguous transition becomes a "stay". Programming errors (AttributeError,
# NameError, ZeroDivisionError, ...) and BaseException are NOT in the tuple and
# propagate. Do not widen either site back to `except Exception`, and do not
# inline a second copy of this tuple -- add classes here or nowhere.
_CLASSIFICATION_SOFT_FAIL_EXCEPTIONS: tuple[type[Exception], ...] = (
    ClassificationError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    OSError,
)


def _value_digest(value: Any) -> str:
    """Short stable digest of a context value, for provenance comparison.

    Args:
        value: any context value (JSON-native, or anything ``str()``-able).

    Returns:
        First 16 hex chars of the sha256 of the value's sorted-key JSON; a
        ``repr`` fallback covers keys that cannot be sorted. Never raises for
        ordinary values.
    """
    try:
        payload = json.dumps(value, sort_keys=True, default=str)
    except (TypeError, ValueError):
        payload = repr(value)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _record_provenance(instance: FSMInstance, committed: dict[str, Any]) -> None:
    """Record that the pipeline itself extracted ``committed`` (digests only)."""
    prov = instance.context.metadata.setdefault(_PROVENANCE_KEY, {})
    for key, value in committed.items():
        prov[key] = _value_digest(value)


# DECISION plan-2026-09-21T203800-8a03483a/D-005 (A4): full classification
# records live in `context.metadata`, NOT in a `_<field>_classification` data
# key: `clean_context_keys` drops every internal-prefixed key before commit,
# and `has_internal_prefix` must not be special-cased per field. Writes are
# copy-on-write (a NEW dict replaces the metadata entry): do NOT switch to
# `metadata.setdefault(...)[field] = record`, because
# `get_complete_conversation` returns a SHALLOW copy of metadata and a later
# turn would mutate a snapshot a caller already holds. Records hold
# JSON-native values only (`_json_native_values`).
def _record_classification_result(
    instance: FSMInstance, field_name: str, record: dict[str, Any]
) -> None:
    """Store ``record`` as the latest result of classification field
    ``field_name`` under ``metadata[METADATA_KEY_CLASSIFICATION_RESULTS]``."""
    metadata = instance.context.metadata
    results = dict(metadata.get(METADATA_KEY_CLASSIFICATION_RESULTS) or {})
    results[field_name] = record
    metadata[METADATA_KEY_CLASSIFICATION_RESULTS] = results


def _record_transition_classification(
    instance: FSMInstance, record: dict[str, Any]
) -> None:
    """Store the turn's transition-classification record in the back-compat
    ``context.data`` key and its ``context.metadata`` mirror
    (plan-2026-09-21T203800-8a03483a/D-005); the mirror is a separate copy so
    neither aliases the other."""
    instance.context.data[CONTEXT_KEY_CLASSIFICATION_RESULT] = record
    instance.context.metadata[METADATA_KEY_TRANSITION_CLASSIFICATION] = copy.deepcopy(
        record
    )


def _json_native_values(data: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    """Independent JSON-native copies of ``data[k]`` for each present ``k``;
    a value that does not survive ``json.dumps`` is left out (never
    stringified, so no object repr lands in metadata)."""
    out: dict[str, Any] = {}
    for key in keys:
        if key in data:
            try:
                out[key] = json.loads(json.dumps(data[key], allow_nan=False))
            except (TypeError, ValueError, RecursionError):
                continue
    return out


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


class MessagePipeline:
    """2-pass message processing pipeline.

    Handles data extraction, transition evaluation, state transitions,
    response generation, and handler execution. Stateless with respect
    to conversation instances — all state is passed as parameters.
    """

    def __init__(
        self,
        llm_interface: LLMInterface,
        data_extraction_prompt_builder: DataExtractionPromptBuilder,
        response_generation_prompt_builder: ResponseGenerationPromptBuilder,
        transition_evaluator: TransitionEvaluator,
        handler_system: HandlerSystem,
        fsm_resolver: Callable[[str], FSMDefinition],
        field_extraction_prompt_builder: FieldExtractionPromptBuilder | None = None,
    ):
        self.llm_interface = llm_interface
        self.data_extraction_prompt_builder = data_extraction_prompt_builder
        self.response_generation_prompt_builder = response_generation_prompt_builder
        self.transition_evaluator = transition_evaluator
        self.handler_system = handler_system
        self.fsm_resolver = fsm_resolver
        self.field_extraction_prompt_builder = (
            field_extraction_prompt_builder or FieldExtractionPromptBuilder()
        )
        # Content-keyed, bounded; see `_get_classifier`. The lock guards
        # check/construct/evict/insert as one unit (review W2, step 2).
        self._classifier_cache: dict[str, Classifier] = {}
        self._classifier_cache_lock = threading.Lock()

    def get_state(
        self, instance: FSMInstance, conversation_id: str | None = None
    ) -> State:
        """Resolve current State from FSM definition."""
        log = (
            logger.bind(conversation_id=conversation_id) if conversation_id else logger
        )

        fsm_def = self.fsm_resolver(instance.fsm_id)
        if instance.current_state not in fsm_def.states:
            error_msg = (
                f"State '{instance.current_state}' not found in FSM '{instance.fsm_id}'"
            )
            log.error(error_msg)
            raise StateNotFoundError(error_msg)

        return fsm_def.states[instance.current_state]

    # ----------------------------------------------------------
    # Handler execution bridge
    # ----------------------------------------------------------

    def execute_handlers(
        self,
        instance: FSMInstance,
        timing: HandlerTiming,
        conversation_id: str,
        current_state: str | None = None,
        target_state: str | None = None,
        updated_keys: set[str] | None = None,
        error_context: dict[str, Any] | None = None,
    ) -> None:
        """Execute handlers at specified timing point.

        Deep-copies instance context before passing to handlers, then merges
        the delta dict back into the instance. A handler returning a key with
        value ``None`` requests deletion of that key.
        """
        # DECISION plan-2026-09-19T175721-21cd7f8e/D-022
        # Zero handlers at this timing: nothing can read or return a delta, so
        # skip the deep copy (14 -> ~4 copies on a zero-handler advance turn, and
        # a non-copyable context value no longer crashes a handler timing). Do NOT
        # extend this to the D-012 pre-turn snapshots below: rollback needs them.
        # A Mock(spec=HandlerSystem) returns a truthy Mock here and keeps the path.
        #
        # DECISION plan-2026-09-19T175721-21cd7f8e/D-029
        # `handlers_at` is an OPTIONAL fast-path hook, not part of the public
        # `handler_system=` seam: a duck-typed system with only
        # `execute_handlers` must keep working. Do NOT call it unguarded (RB-11).
        handlers_at = getattr(self.handler_system, "handlers_at", None)
        if callable(handlers_at) and not handlers_at(timing):
            return
        context = copy.deepcopy(instance.context.data)

        if error_context:
            context.update(error_context)

        def merge_delta(delta: dict[str, Any]) -> None:
            """Apply a handler delta dict to the instance context (None = delete)."""
            # DECISION plan-2026-09-19T175721-21cd7f8e/D-018
            # An ERROR handler runs after the turn was rolled back; merging its
            # returned dict would write into that rolled-back state (RA-07).
            # update_context is the supported write path. Do NOT re-enable.
            if timing is HandlerTiming.ERROR:
                if delta:
                    logger.debug(
                        f"ERROR handler delta not merged (turn rolled back): "
                        f"{sorted(delta)}"
                    )
                return
            # DECISION plan-2026-09-20T114608-a8e47b88/D-018
            # This is the ONE place a handler-returned `None` deletes a
            # context key -- `ContextCompactor.compact`/`prune` included.
            # Those two only ever see `instance.context.data` (a plain dict,
            # via this method's own `context` variable above); they cannot
            # reach `.metadata` themselves and never delete anything
            # directly, so the fix cannot live inside them without changing
            # their public `context: dict[str, Any]` signature. Every
            # None-delta deletion, from ANY handler, passes through here --
            # do NOT special-case `ContextCompactor` with a second deletion
            # site; that would leave the exact same stale-digest gap for
            # every other handler using this same convention. A deleted
            # key's provenance digest must go with it or a pruned
            # low-entropy value's digest can outlive its plaintext in a
            # persisted session file (findings/residual-findings-verify.md
            # #8). See decisions.md D-018.
            for key, value in delta.items():
                if value is None:
                    instance.context.data.pop(key, None)
                    prov = instance.context.metadata.get(_PROVENANCE_KEY)
                    if prov is not None:
                        prov.pop(key, None)
                else:
                    instance.context.data[key] = value

        try:
            updated_context = self.handler_system.execute_handlers(
                timing=timing,
                current_state=current_state or instance.current_state,
                target_state=target_state,
                context=context,
                updated_keys=updated_keys,
            )

            if updated_context:
                merge_delta(updated_context)

        except Exception as e:
            # DECISION plan-2026-07-18T051819-80b0bd4d/D-006 [STALE]: re-raise UNCONDITIONALLY.
            # HandlerSystem.execute_handlers raises only when error_mode == "raise" OR
            # the failing handler is critical=True, so anything escaping it already
            # means "must propagate". Do NOT reintroduce an `if error_mode == "raise"`
            # re-gate here: it swallowed every critical failure under "continue".
            #
            # SCOPE OF THE partial-context MERGE BELOW: it preserves what earlier
            # handlers at this timing point already produced, and that survives for
            # CONTEXT_UPDATE. It does NOT survive at POST_TRANSITION:
            # _execute_state_transition intentionally restores the pre-transition
            # snapshot (see the rollback below) taken BEFORE these handlers ran, so
            # the rollback overwrites this merge. That override is deliberate and
            # correct — a half-applied transition is worse than a lost partial
            # delta. Pinned by tests/test_fsm_llm/test_pipeline_handler_contract.py::
            # TestPostTransitionHandlerFailure.  It also does not survive for a key
            # that the pre-transition CONTEXT_UPDATE rollback owns (D-005 shape (c));
            # every other key at that timing point is preserved as stated.
            #
            # UPDATE (D-002, plan-2026-09-12T065608-089d0ec7): for PRE_PROCESSING
            # and POST_PROCESSING specifically, this merge no longer survives a
            # raise that escapes to MessagePipeline.process()/process_stream():
            # those two call sites are now each individually wrapped in a
            # restore-on-exception block that reverts the whole pre-turn snapshot
            # (turn-atomicity), which runs immediately after this merge and undoes
            # it. The merge below still happens (so a caller catching the
            # exception at THIS layer, below process(), still sees the partial
            # delta applied to `instance.context.data` for one statement), but
            # process()/process_stream() wipe it before the exception reaches
            # their caller. See D-002 in decisions.md and
            # test_pipeline_handler_contract.py::TestPartialHandlerResultsPreserved
            # (updated by the same step) for the narrowed contract.
            merge_delta(getattr(e, "partial_context", None) or {})
            logger.error(f"Handler execution error at {timing.name}: {e!s}")
            raise

    # ----------------------------------------------------------
    # Full 2-pass processing
    # ----------------------------------------------------------

    def process(self, instance: FSMInstance, message: str, conversation_id: str) -> str:
        """Execute the full 2-pass message processing pipeline.

        Pass 1: PRE_PROCESSING handlers → data extraction → context update →
                transition evaluation → state transition
        Pass 2: POST_PROCESSING handlers → response generation

        Args:
            instance: The FSM instance (already validated as non-terminal).
            message: User message to process.
            conversation_id: Conversation identifier.

        Returns:
            Generated response message.
        """
        # Contextualize propagates conversation_id to all downstream logger
        # calls on this thread (llm.py, transition_evaluator.py, etc.)
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
            pre_turn_state = instance.current_state
            pre_turn_data = copy.deepcopy(instance.context.data)
            pre_turn_wm = copy.deepcopy(instance.context.working_memory)
            pre_turn_metadata = copy.deepcopy(instance.context.metadata)

            # DECISION plan-2026-09-21T203800-8a03483a/D-002 (A8): a transition
            # classification record belongs to the turn that produced it. Clear
            # it AFTER the snapshot so a rolled-back turn restores the prior
            # record; do NOT move this before the snapshot, and do NOT drop it
            # from process_stream() (same clear there). The metadata mirror
            # (A4, plan-2026-09-21T203800-8a03483a/D-005) is part of the same
            # record and is cleared with it.
            instance.context.data.pop(CONTEXT_KEY_CLASSIFICATION_RESULT, None)
            instance.context.metadata.pop(METADATA_KEY_TRANSITION_CLASSIFICATION, None)

            # DECISION plan-2026-09-12T065608-089d0ec7/D-002
            # Widened turn-atomicity: PRE_PROCESSING and POST_PROCESSING handler
            # calls are now EACH individually wrapped in their own restore-on-
            # exception block (reusing the D-012 pre_turn_* snapshots above, no
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
                    conversation_id,
                    current_state=instance.current_state,
                )
            except Exception:
                # Restore the pre-turn in-memory state so the turn is atomic.
                # Covers all handler-mutable fields, not just state+data (D-012).
                instance.current_state = pre_turn_state
                instance.context.data.clear()
                instance.context.data.update(pre_turn_data)
                instance.context.working_memory = pre_turn_wm
                instance.context.metadata.clear()
                instance.context.metadata.update(pre_turn_metadata)
                raise

            # Pass 1: Data extraction + transition evaluation + execution
            extraction_response, transition_occurred, previous_state = (
                self._execute_extraction_and_transition_pass(
                    instance, message, conversation_id
                )
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
                    conversation_id,
                    current_state=instance.current_state,
                )

                # Pass 2: Response generation based on final state
                return self._execute_response_generation_pass(
                    instance,
                    message,
                    extraction_response,
                    transition_occurred,
                    previous_state,
                    conversation_id,
                )
            except Exception:
                # Restore the pre-turn in-memory state so the turn is atomic.
                # Covers all handler-mutable fields, not just state+data (D-012).
                # D-017: this ALSO undoes Pass 1's already-committed state
                # transition + extracted data when the exception originated in
                # POST_PROCESSING (see the D-017 comment above), not only when
                # it originated in Pass 2.
                instance.current_state = pre_turn_state
                instance.context.data.clear()
                instance.context.data.update(pre_turn_data)
                instance.context.working_memory = pre_turn_wm
                instance.context.metadata.clear()
                instance.context.metadata.update(pre_turn_metadata)
                raise

    def process_stream(
        self, instance: FSMInstance, message: str, conversation_id: str
    ) -> Iterator[str]:
        """Execute 2-pass processing, streaming Pass 2 tokens.

        Pass 1 runs fully (extraction + transition). Pass 2 yields
        response tokens as they arrive from the LLM.

        Args:
            instance: The FSM instance (already validated as non-terminal).
            message: User message to process.
            conversation_id: Conversation identifier.

        Yields:
            String chunks of the response as they arrive.
        """
        with logger.contextualize(conversation_id=conversation_id, package="fsm_llm"):
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
            pre_turn_state = instance.current_state
            pre_turn_data = copy.deepcopy(instance.context.data)
            pre_turn_wm = copy.deepcopy(instance.context.working_memory)
            pre_turn_metadata = copy.deepcopy(instance.context.metadata)

            # DECISION plan-2026-09-21T203800-8a03483a/D-002 (A8): streaming
            # mirror of the per-turn clear in process(); after the snapshot.
            instance.context.data.pop(CONTEXT_KEY_CLASSIFICATION_RESULT, None)
            instance.context.metadata.pop(METADATA_KEY_TRANSITION_CLASSIFICATION, None)

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
                    conversation_id,
                    current_state=instance.current_state,
                )
            except Exception:
                # Restore the pre-turn in-memory state so the streaming turn is
                # atomic. Covers all handler-mutable fields (D-004, mirrors D-012/
                # D-013). GeneratorExit deliberately does NOT reach here.
                instance.current_state = pre_turn_state
                instance.context.data.clear()
                instance.context.data.update(pre_turn_data)
                instance.context.working_memory = pre_turn_wm
                instance.context.metadata.clear()
                instance.context.metadata.update(pre_turn_metadata)
                raise

            # Pass 1: Data extraction + transition (runs fully)
            extraction_response, transition_occurred, previous_state = (
                self._execute_extraction_and_transition_pass(
                    instance, message, conversation_id
                )
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
                    conversation_id,
                    current_state=instance.current_state,
                )

                # Pass 2: Stream response generation
                yield from self._stream_response_generation_pass(
                    instance,
                    message,
                    extraction_response,
                    transition_occurred,
                    previous_state,
                    conversation_id,
                )
            except Exception:
                # Restore the pre-turn in-memory state so the streaming turn is
                # atomic. Covers all handler-mutable fields (D-004, mirrors D-012/
                # D-013). GeneratorExit deliberately does NOT reach here.
                instance.current_state = pre_turn_state
                instance.context.data.clear()
                instance.context.data.update(pre_turn_data)
                instance.context.working_memory = pre_turn_wm
                instance.context.metadata.clear()
                instance.context.metadata.update(pre_turn_metadata)
                raise

    def _stream_response_generation_pass(
        self,
        instance: FSMInstance,
        user_message: str,
        extraction_response: DataExtractionResponse,
        transition_occurred: bool,
        previous_state: str | None,
        conversation_id: str,
    ) -> Iterator[str]:
        """Stream Pass 2: yield response tokens as they arrive."""
        log = logger.bind(conversation_id=conversation_id)

        current_state = self.get_state(instance, conversation_id)

        # Fast-path for empty response_instructions
        if (
            current_state.response_instructions is not None
            and not current_state.response_instructions
        ):
            synthetic = f"[{current_state.id}]"
            instance.context.conversation.add_system_message(synthetic)
            yield synthetic
            return

        fsm_def = self.fsm_resolver(instance.fsm_id)

        # Only enforce structured output format on terminal states (no
        # outgoing transitions).  Applying it on intermediate states forces
        # the model to produce JSON when the prompt asks for free-form text,
        # which can cause small models to hang or produce garbage.
        output_response_format = None
        if not current_state.transitions:
            output_response_format = instance.context.data.get(
                "_output_response_format"
            )

        # DECISION plan-2026-09-19T175721-21cd7f8e/D-003: stream plain text
        # unless the terminal state carries a structured output format.  The
        # sync prompt (JSON envelope) is unchanged; the stream yields and
        # persists raw deltas, so the envelope must never be requested here.
        system_prompt = self.response_generation_prompt_builder.build_response_prompt(
            instance=instance,
            state=current_state,
            fsm_definition=fsm_def,
            extracted_data=extraction_response.extracted_data,
            transition_occurred=transition_occurred,
            previous_state=previous_state,
            user_message=user_message,
            plain_text_response=output_response_format is None,
            context=self._apply_context_scope(
                instance.context.get_merged_data(), current_state, conversation_id
            ),
            # DECISION plan-2026-09-19T175721-21cd7f8e/D-032
            # read_keys scopes THIS channel too: a refused value is a context
            # value. Do NOT pass the raw dict (final review concern 3): the key
            # the state hides from <current_context> then leaks here.
            rejected_corrections=self._apply_context_scope(
                extraction_response.rejected_corrections, current_state, conversation_id
            ),
            extraction_failed=extraction_response.extraction_failed,
        )

        context_for_llm = self._apply_context_scope(
            instance.context.get_user_visible_data(),
            current_state,
            conversation_id,
        )

        request = ResponseGenerationRequest(
            system_prompt=system_prompt,
            user_message=user_message,
            extracted_data=extraction_response.extracted_data,
            context=context_for_llm,
            transition_occurred=transition_occurred,
            previous_state=previous_state,
            response_format=output_response_format,
        )

        # Accumulate chunks to store in conversation history
        chunks: list[str] = []
        persist = True
        # DECISION plan-2026-07-18T051819-80b0bd4d/D-015 [STALE]: client abandonment and a backend
        # stream error MUST diverge here, and the ORDER of these except clauses is
        # the whole mechanism.
        #   * GeneratorExit (the consumer stopped iterating / the generator was
        #     closed or GC'd) keeps `persist = True`, so the partial reply the user
        #     ACTUALLY SAW is written to history.  fsm.py:329-338 documents that
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
        # the abandonment semantics and silently flip the fsm.py contract.
        #
        # SCOPE OF THE PARITY CLAIM — corrected in step 8; measure, do not assume.
        # Full parity with the synchronous path (_execute_response_generation_pass,
        # which never calls add_system_message when generate_response raises) holds
        # for **Exception subclasses only**.  For a true BaseException
        # (KeyboardInterrupt / SystemExit) this clause suppresses the partial reply
        # correctly, but the orphaned user turn SURVIVES: the rollback lives in
        # fsm.py:326-341, whose clauses are FSMError / GeneratorExit / Exception —
        # there is no BaseException clause, so _rollback_user_message never runs.
        # Probed on this tree: ^C mid-stream leaves
        # [{'system': 'Greetings!'}, {'user': 'hi'}].  Do NOT "fix" this by
        # widening fsm.py's clause — whether a KeyboardInterrupt should roll back
        # is a design call, not a defect fix.  Open item, recorded in
        # verification.md § Not Verified.
        #
        # .throw() vs .close(): only GeneratorExit is the documented abandonment
        # signal, so a thrown Exception is deliberately treated as an ERROR (partial
        # dropped, user turn rolled back) even though the consumer may already have
        # seen chunks.  That asymmetry is intended, not an oversight.
        try:
            for chunk in self.llm_interface.generate_response_stream(request):
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
            # GENUINE partial must still persist (that is the fsm.py contract),
            # and only the empty-content case is suppressed here.
            full_message = "".join(chunks)
            if full_message and persist:
                instance.context.conversation.add_system_message(full_message)
                log.debug("Streaming response generation completed")

    # ----------------------------------------------------------
    # Initial response generation
    # ----------------------------------------------------------

    def generate_initial_response(
        self, instance: FSMInstance, conversation_id: str
    ) -> str:
        """Generate initial response for conversation start (no extraction/transition)."""
        log = logger.bind(conversation_id=conversation_id)

        current_state = self.get_state(instance, conversation_id)

        # DECISION plan-2026-09-20T165703-0d9c218e/D-006
        # Fast-path for an initial state with empty response_instructions (a
        # ReAct-style `think` state is the initial state of every agent FSM).
        # Without this skip the greeting runs a full Pass 2 and the model's
        # completion prose lands in history BEFORE any tool runs, poisoning
        # every later `tool_name` extraction (F-LIVE-01). This block is an
        # EXACT mirror of the sync-turn site in
        # `_execute_response_generation_pass`:
        # the `"."` system_prompt is the sentinel `LiteLLMInterface` uses to
        # return a synthetic response WITHOUT calling litellm, so the call
        # below IS the skip mechanism. Do NOT drop the `generate_response`
        # call (a custom interface must see the same call count at the
        # greeting as on every turn), do NOT set
        # `instance.last_response_generation` (the sibling sites do not), and
        # do NOT fold the three copies into a shared helper (D-006).
        if (
            current_state.response_instructions is not None
            and not current_state.response_instructions
        ):
            request = ResponseGenerationRequest(
                system_prompt=".",
                user_message="",
                extracted_data={},
                context={},
                transition_occurred=False,
                previous_state=None,
            )
            self.llm_interface.generate_response(request)
            synthetic = f"[{current_state.id}]"
            instance.context.conversation.add_system_message(synthetic)
            log.debug(
                "Skipped initial response generation (empty response_instructions)"
            )
            return synthetic

        fsm_def = self.fsm_resolver(instance.fsm_id)

        system_prompt = self.response_generation_prompt_builder.build_response_prompt(
            instance=instance,
            state=current_state,
            fsm_definition=fsm_def,
            extracted_data={},
            transition_occurred=False,
            previous_state=None,
            user_message="",
            context=self._apply_context_scope(
                instance.context.get_merged_data(), current_state, conversation_id
            ),
        )

        request = ResponseGenerationRequest(
            system_prompt=system_prompt,
            user_message="",
            extracted_data={},
            context=self._apply_context_scope(
                instance.context.get_user_visible_data(),
                current_state,
                conversation_id,
            ),
            transition_occurred=False,
            previous_state=None,
        )

        response = self.llm_interface.generate_response(request)
        instance.last_response_generation = response
        instance.context.conversation.add_system_message(response.message)

        log.info("Generated initial response")
        return response.message

    # ----------------------------------------------------------
    # Pass 1: Data extraction + transition
    # ----------------------------------------------------------

    def _execute_extraction_and_transition_pass(
        self, instance: FSMInstance, user_message: str, conversation_id: str
    ) -> tuple[DataExtractionResponse, bool, str | None]:
        """Execute Pass 1: Data Extraction + Transition Evaluation + Execution."""
        log = logger.bind(conversation_id=conversation_id)
        log.debug("Executing data extraction and transition pass")

        # Step 1: Unified field-based extraction (auto-converts legacy
        # required_context_keys and merges with explicit field_extractions)
        extraction_response = self._execute_data_extraction(
            instance, user_message, conversation_id
        )

        # Step 2: Update context with extracted data
        if extraction_response.extracted_data:
            extraction_response.extracted_data = self._clean_empty_context_keys(
                data=extraction_response.extracted_data, conversation_id=conversation_id
            )

            if extraction_response.extracted_data:
                committed = extraction_response.extracted_data

                # DECISION plan-2026-07-19T191147-4b664252/D-005 [STALE]
                # SHAPE (c) of three DIFFERENT partial-commit contracts in this file.
                # Roll back ONLY the keys this statement commits, then re-raise:
                #   (a) _execute_state_transition — FULL context snapshot restore, because
                #       a half-applied transition is worse than a lost handler delta.
                #   (b) post-transition CONTEXT_UPDATE (below) — NO rollback; the
                #       transition is already committed and cannot be undone here.
                #   (c) HERE — scoped rollback. Do NOT "align" this with (a) by
                #       deep-copying all of context.data: that would also discard the
                #       deltas of CONTEXT_UPDATE handlers that already SUCCEEDED, which
                #       D-006 guarantees survive at this timing point (pinned by
                #       test_pipeline_handler_contract.py::TestPartialHandlerResultsPreserved).
                # Without this, a permanently-failing critical handler wrote the field
                # once and then blocked every later turn, while the user turn was popped
                # from history — get_data() and history disagreed, and the extraction
                # skip-guard below meant a retry never re-ran. Precedence note: if a
                # handler delta touched one of `committed`'s keys, the rollback wins.
                # BOUNDARY (known, deliberate): the snapshot is SHALLOW. Key coverage
                # is exact — `FSMContext.update` writes precisely `new_data`'s keys and
                # derives nothing, so no extraction-written key escapes the restore —
                # but `pre_commit` holds the SAME object as `context.data[key]`. A
                # CONTEXT_UPDATE handler that mutates a pre-existing dict/list value
                # IN PLACE before failing is therefore NOT restored. Do not "fix" this
                # with a deep copy: that reintroduces exactly the (a)-shaped cost this
                # decision rejected, on every turn, to undo a mutation that handlers
                # are contractually supposed to make by returning a delta rather than
                # by reaching into `context.data`.
                pre_commit = {
                    key: instance.context.data[key]
                    for key in committed
                    if key in instance.context.data
                }
                instance.context.update(committed)

                # Notify handlers about context updates
                try:
                    self.execute_handlers(
                        instance,
                        HandlerTiming.CONTEXT_UPDATE,
                        conversation_id,
                        current_state=instance.current_state,
                        updated_keys=set(committed.keys()),
                    )
                except Exception as handler_err:
                    log.warning(
                        f"CONTEXT_UPDATE handler failed "
                        f"({type(handler_err).__name__}: {handler_err}), rolling back "
                        f"extracted keys {sorted(committed)}"
                    )
                    for key in committed:
                        if key in pre_commit:
                            instance.context.data[key] = pre_commit[key]
                        else:
                            instance.context.data.pop(key, None)
                    raise
                # DECISION plan-2026-09-19T175721-21cd7f8e/D-015: recorded only
                # after the handlers succeeded, so a rollback leaves no stale
                # provenance; a handler edit of a key fails the digest compare.
                _record_provenance(instance, committed)

        # Step 3: Transition Evaluation and Execution
        transition_occurred, previous_state = (
            self._execute_transition_evaluation_and_execution(
                instance, user_message, extraction_response, conversation_id
            )
        )

        # Step 4: Post-transition extraction — re-extract in the new state
        # if a transition occurred.  This handles the common case where
        # the user provides data relevant to the *next* state in the same
        # message (e.g., providing email+age when the FSM just collected
        # the name).  Skipped for agent-managed FSMs (detected by
        # "agent_trace" context key) to avoid extra LLM calls.
        is_agent_fsm = CONTEXT_KEY_AGENT_TRACE in instance.context.data
        if transition_occurred and not is_agent_fsm:
            new_state = self.get_state(instance, conversation_id)
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
                c
                for c in new_configs
                if c.field_name not in instance.context.data
                or instance.context.data.get(c.field_name) is None
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
            prov_now = instance.context.metadata.get(_PROVENANCE_KEY, {})
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
            # just classified this message), agent FSMs (guarded above) or
            # `handler_only_keys` (D-033). Do NOT drop the `revisit` exclusion
            # of classification-owning states: a full re-run there would buy
            # the bulk call on top of these.
            class_configs = (
                [
                    c
                    for c in new_state.classification_extractions or []
                    if c.field_name not in handler_only
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
                        post_data = self._clean_empty_context_keys(
                            data=post_data, conversation_id=conversation_id
                        )
                        if post_data:
                            instance.context.update(post_data)
                            # DECISION plan-2026-09-19T175721-21cd7f8e/D-015
                            _record_provenance(instance, post_data)
                            extraction_response.extracted_data.update(post_data)
                            self.execute_handlers(
                                instance,
                                HandlerTiming.CONTEXT_UPDATE,
                                conversation_id,
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

        log.debug("Data extraction and transition pass completed")
        return extraction_response, transition_occurred, previous_state

    # ----------------------------------------------------------
    # Pass 1: Field-based extraction (replaces bulk extract_data)
    # ----------------------------------------------------------

    def _bulk_extract_from_instructions(
        self,
        instance: FSMInstance,
        user_message: str,
        state: State,
        conversation_id: str,
    ) -> dict[str, Any]:
        """Bulk-extract data when a state has extraction_instructions but no
        explicit required_context_keys or field_extractions.

        Uses a single LLM call with a simple prompt to extract whatever
        data the instructions describe.  Returns a dict of extracted
        key-value pairs (may be empty).
        """
        log = logger.bind(conversation_id=conversation_id)

        safe_message = self.data_extraction_prompt_builder._sanitize_text_for_prompt(
            user_message
        )
        prompt = (
            f"Extract information from the user's message.\n\n"
            f"Instructions: {state.extraction_instructions}\n\n"
            f"User message: {safe_message}\n\n"
            f'Respond with JSON: {{"extracted_data": {{"key": "value", ...}}, '
            f'"confidence": 0.95, "reasoning": "..."}}\n\n'
            f"Only include keys for information actually present in the "
            f"user's message. Use descriptive snake_case key names."
        )

        # DECISION plan-2026-09-12T135914-45a654de/D-015
        # Previously reached past the LLMInterface ABC via
        # `cast(LiteLLMInterface, self.llm_interface)._make_llm_call(...)` —
        # a private method of ONE concrete implementation, called from
        # outside that class. Now goes through the public ABC surface
        # (`extract_bulk_data`, llm.py), which does its own parsing via the
        # shared `strip_think_and_fences` helper (utilities.py) — the
        # <think>-tag/fence-stripping regex that used to be hand-duplicated
        # here is gone, not reconciled. Do NOT reintroduce a cast+private-call
        # here; if a future LLMInterface subclass needs bulk extraction, it
        # implements `extract_bulk_data` (default raises NotImplementedError,
        # mirroring `extract_field`). See decisions.md D-015.
        try:
            request = BulkExtractionRequest(
                system_prompt=prompt, user_message=user_message
            )
            response = self.llm_interface.extract_bulk_data(request)
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

        return {}

    def _handler_only_keys(self, instance: FSMInstance) -> frozenset[str]:
        """Keys the FSM author reserved for handlers (D-033).

        Resolved per instance through ``fsm_resolver`` so a stacked child uses
        its own list, not the root's. Returns an empty set for the default.
        """
        return frozenset(self.fsm_resolver(instance.fsm_id).handler_only_keys)

    @staticmethod
    def _build_field_configs_from_state(state: State) -> list[FieldExtractionConfig]:
        """Auto-convert legacy state fields to FieldExtractionConfig list.

        Translates ``required_context_keys`` + ``extraction_instructions``
        into per-field configs so the pipeline can use the unified
        ``extract_field`` primitive for all extraction.  Explicit
        ``field_extractions`` on the state are appended after the
        auto-generated ones.
        """
        configs: list[FieldExtractionConfig] = []

        # Collect all required keys: from state-level AND from transition
        # conditions' requires_context_keys.  This ensures fields that
        # transitions depend on are extracted even if the state doesn't
        # list them in its own required_context_keys.
        all_required_keys: list[str] = list(state.required_context_keys or [])
        if state.transitions:
            for transition in state.transitions:
                if transition.conditions:
                    for condition in transition.conditions:
                        if condition.requires_context_keys:
                            for key in condition.requires_context_keys:
                                if key not in all_required_keys:
                                    all_required_keys.append(key)

        # DECISION plan-2026-09-19T175721-21cd7f8e/D-006: a key owned by a
        # classification_extractions entry is NOT auto-minted as a plain
        # extraction. Minting it let the plain extractor fill a key the
        # classifier had just rejected below its confidence threshold (the
        # gated transition then fired anyway) and cost one wasted LLM call per
        # turn. Do NOT gate this in the transition evaluator (wrong layer).
        # Explicit ``field_extractions`` with the same name still win below.
        classification_owned = {
            c.field_name for c in (state.classification_extractions or [])
        }
        all_required_keys = [
            k for k in all_required_keys if k not in classification_owned
        ]

        # Auto-convert required keys → one config per key
        if all_required_keys:
            instructions = (
                state.extraction_instructions
                or "Extract the value of this field from the user's input."
            )
            for key in all_required_keys:
                configs.append(
                    FieldExtractionConfig(
                        field_name=key,
                        field_type="any",
                        extraction_instructions=(
                            f"Extract the '{key}' field. {instructions}"
                        ),
                        context_keys=None,  # all context
                        required=True,
                        confidence_threshold=state.extraction_confidence_threshold,
                    )
                )

        # Append explicit field_extractions (user-defined, take priority)
        if state.field_extractions:
            # Avoid duplicates: explicit configs override auto-generated ones
            explicit_names = {fc.field_name for fc in state.field_extractions}
            configs = [c for c in configs if c.field_name not in explicit_names]
            configs.extend(state.field_extractions)

        return configs

    def _execute_data_extraction(
        self, instance: FSMInstance, user_message: str, conversation_id: str
    ) -> DataExtractionResponse:
        """Execute data extraction via per-field ``extract_field`` calls
        and classification extractions.

        Builds a unified list of ``FieldExtractionConfig`` from both
        legacy ``required_context_keys`` and explicit ``field_extractions``,
        then extracts each field individually.  Also runs any
        ``classification_extractions`` declared on the state.  Supports
        multi-pass retry for missing required fields (up to ``extraction_retries``).
        """
        log = logger.bind(conversation_id=conversation_id)
        log.debug("Executing field-based data extraction")

        current_state = self.get_state(instance, conversation_id)

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
        has_classification_configs = bool(current_state.classification_extractions)

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
            model = getattr(self.llm_interface, "model", None)
            memo: dict[tuple[str, str], FieldExtractionResponse] | None = (
                {} if isinstance(model, str) and is_ollama_model(model) else None
            )
            existing = instance.context.data
            # DECISION plan-2026-09-19T175721-21cd7f8e/D-046: for an agent FSM an
            # EMPTY list/dict counts as unset. A builder seeds `[]` so a
            # has_context gate can pass (plan_execute `plan_steps`); without this
            # the seed hides the key from extraction. Non-agent FSMs keep "empty
            # is set": do NOT widen this to every FSM (D-036 rejected that).
            agent_managed = CONTEXT_KEY_AGENT_TRACE in existing
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
                        for cfg in (current_state.classification_extractions or [])
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
                agent_managed = CONTEXT_KEY_AGENT_TRACE in existing
                prov = instance.context.metadata.get(_PROVENANCE_KEY, {})
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
                        not agent_managed
                        and str(value).strip().lower() != str(current).strip().lower()
                    ):
                        # DECISION plan-2026-09-19T175721-21cd7f8e/D-052 (4):
                        # an instruction-only key (no config) is never
                        # corrected (skip-if-set) but IS reported below under
                        # the same grounding test. Do NOT correct it here.
                        # LV6-01: a back edge that then applies the value
                        # prunes this entry (~line 1009).
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

        # Build final response — check all sources for missing required fields
        all_required_names: list[str] = []
        if has_field_configs:
            all_required_names.extend(
                cfg.field_name for cfg in all_configs if cfg.required
            )
        if has_classification_configs:
            all_required_names.extend(
                cfg.field_name
                for cfg in (current_state.classification_extractions or [])
                if cfg.required
            )

        min_confidence = min(confidences) if confidences else 0.0
        response = DataExtractionResponse(
            extracted_data=extracted_data,
            confidence=min_confidence,
            additional_info_needed=any(
                name not in extracted_data and name not in instance.context.data
                for name in all_required_names
            ),
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

    def _execute_field_extractions(
        self,
        instance: FSMInstance,
        user_message: str,
        field_configs: list[FieldExtractionConfig],
        conversation_id: str,
        memo: dict[tuple[str, str], FieldExtractionResponse] | None = None,
    ) -> list[FieldExtractionResponse]:
        """Execute targeted field extractions for a list of configs.

        Runs one LLM call per field.  Each config specifies its own
        instructions, dynamic context selection, and validation rules.

        Previously extracted values are added to the dynamic context
        for subsequent extractions, enabling dependent field extraction
        (e.g., tool_input can see that tool_name was already extracted).

        ``memo`` (D-035, ``None`` = disabled) maps ``(field_name, built prompt
        + "|" + message)`` to a previous NULL result; it is only ever handed an
        Ollama interface's dict by ``_execute_data_extraction``.
        """
        log = logger.bind(conversation_id=conversation_id)
        results: list[FieldExtractionResponse] = []
        # Accumulate extracted values so later fields can see earlier ones
        extracted_so_far: dict[str, Any] = {}

        for field_config in field_configs:
            log.debug(
                f"Extracting field '{field_config.field_name}' "
                f"(type={field_config.field_type})"
            )

            # Build dynamic context from config.context_keys
            if field_config.context_keys is not None:
                dynamic_context = {
                    k: v
                    for k, v in instance.context.data.items()
                    if k in field_config.context_keys
                }
            else:
                # Apply state-level context_scope as default filter
                current_state = self.get_state(instance, conversation_id)
                dynamic_context = self._apply_context_scope(
                    instance.context.get_user_visible_data(),
                    current_state,
                    conversation_id,
                )

            # Include previously extracted fields so the LLM can use them
            if extracted_so_far:
                dynamic_context.update(extracted_so_far)

            # Build prompt
            system_prompt = (
                self.field_extraction_prompt_builder.build_field_extraction_prompt(
                    instance=instance,
                    field_config=field_config,
                    user_message=user_message,
                    dynamic_context=dynamic_context,
                )
            )

            # Build request
            request = FieldExtractionRequest(
                system_prompt=system_prompt,
                user_message=user_message,
                field_name=field_config.field_name,
                field_type=field_config.field_type,
                context=dynamic_context,
                validation_rules=field_config.validation_rules,
            )

            # Call LLM (or reuse an identical null, D-035)
            memo_key = (
                field_config.field_name,
                f"{system_prompt}|{user_message}",
            )
            try:
                if memo is not None and memo_key in memo:
                    response = memo[memo_key]
                else:
                    response = self.llm_interface.extract_field(request)
                    if memo is not None and response.value is None:
                        memo[memo_key] = response
            except Exception as e:
                log.warning(
                    f"Field extraction failed for '{field_config.field_name}': {e}"
                )
                response = FieldExtractionResponse(
                    field_name=field_config.field_name,
                    value=None,
                    confidence=0.0,
                    is_valid=False,
                    validation_error=f"LLM call failed: {e}",
                )

            # Validate and coerce
            response = self._validate_field_extraction(response, field_config)

            log.debug(
                f"Field '{field_config.field_name}': "
                f"value={response.value!r}, confidence={response.confidence:.2f}, "
                f"valid={response.is_valid}"
            )
            results.append(response)

            # Feed successful extractions into context for subsequent fields
            if response.is_valid and response.value is not None:
                extracted_so_far[field_config.field_name] = response.value

        return results

    @staticmethod
    def _validate_field_extraction(
        response: FieldExtractionResponse,
        config: FieldExtractionConfig,
    ) -> FieldExtractionResponse:
        """Validate and type-coerce a field extraction response."""
        # Skip validation if already failed
        if not response.is_valid or response.value is None:
            return response

        # Reject values that are obviously the field name echoed back —
        # small models sometimes confuse the JSON template keys with values.
        if isinstance(response.value, str) and response.value.strip().lower() in (
            config.field_name.lower(),
            "field_name",
            "value",
        ):
            return FieldExtractionResponse(
                field_name=response.field_name,
                value=None,
                confidence=0.0,
                reasoning="Model echoed field name instead of extracting a value",
                is_valid=False,
                validation_error="Extracted value matches field name (model confusion)",
            )

        # A self-reported confidence of exactly 0.0 means the model could not
        # ground the value (LV-01: dict-wrapped junk came back with low/zero
        # confidence); treat it as "not extracted" regardless of the configured
        # threshold (which defaults can leave at 0.0). decisions.md D-002.
        # DECISION plan-2026-09-19T175721-21cd7f8e/D-016
        # Do NOT make this threshold-driven (`confidence <= threshold`): with
        # the default threshold 0.0 it is the same rule, and an author-set
        # threshold above 0 already rejects 0.0 below. Do NOT delete it either:
        # it is what rejects LV-01 dict junk that comes back at confidence 0.
        # Known cliff: a correct value the model scored exactly 0 is dropped
        # while 0.05 is kept. See decisions.md D-016.
        if response.confidence == 0.0:
            return FieldExtractionResponse(
                field_name=response.field_name,
                value=response.value,
                confidence=0.0,
                reasoning=response.reasoning,
                is_valid=False,
                validation_error="Model reported zero confidence in the value",
            )

        # Confidence threshold check
        if (
            config.confidence_threshold > 0.0
            and response.confidence < config.confidence_threshold
        ):
            return FieldExtractionResponse(
                field_name=response.field_name,
                value=response.value,
                confidence=response.confidence,
                reasoning=response.reasoning,
                is_valid=False,
                validation_error=(
                    f"Confidence {response.confidence:.2f} below threshold "
                    f"{config.confidence_threshold:.2f}"
                ),
            )

        # Type coercion via dispatch
        value = response.value
        try:
            coercer = _TYPE_COERCERS.get(config.field_type)
            if coercer is not None:
                value = coercer(value)
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            return FieldExtractionResponse(
                field_name=response.field_name,
                value=response.value,
                confidence=response.confidence,
                reasoning=response.reasoning,
                is_valid=False,
                validation_error=(f"Type coercion to {config.field_type} failed: {e}"),
            )

        # Validation rules
        rules = config.validation_rules or {}
        if "allowed_values" in rules:
            if value not in rules["allowed_values"]:
                return FieldExtractionResponse(
                    field_name=response.field_name,
                    value=value,
                    confidence=response.confidence,
                    reasoning=response.reasoning,
                    is_valid=False,
                    validation_error=(
                        f"Value {value!r} not in allowed values: "
                        f"{rules['allowed_values']}"
                    ),
                )

        if "min_length" in rules and isinstance(value, str):
            if len(value) < rules["min_length"]:
                return FieldExtractionResponse(
                    field_name=response.field_name,
                    value=value,
                    confidence=response.confidence,
                    reasoning=response.reasoning,
                    is_valid=False,
                    validation_error=(
                        f"Value length {len(value)} below minimum {rules['min_length']}"
                    ),
                )

        if "max_length" in rules and isinstance(value, str):
            if len(value) > rules["max_length"]:
                return FieldExtractionResponse(
                    field_name=response.field_name,
                    value=value,
                    confidence=response.confidence,
                    reasoning=response.reasoning,
                    is_valid=False,
                    validation_error=(
                        f"Value length {len(value)} exceeds maximum "
                        f"{rules['max_length']}"
                    ),
                )

        if "min_value" in rules:
            try:
                if float(value) < float(rules["min_value"]):
                    return FieldExtractionResponse(
                        field_name=response.field_name,
                        value=value,
                        confidence=response.confidence,
                        reasoning=response.reasoning,
                        is_valid=False,
                        validation_error=(
                            f"Value {value!r} is below minimum {rules['min_value']}"
                        ),
                    )
            except (TypeError, ValueError):
                pass  # non-numeric values skip numeric range check

        if "max_value" in rules:
            try:
                if float(value) > float(rules["max_value"]):
                    return FieldExtractionResponse(
                        field_name=response.field_name,
                        value=value,
                        confidence=response.confidence,
                        reasoning=response.reasoning,
                        is_valid=False,
                        validation_error=(
                            f"Value {value!r} exceeds maximum {rules['max_value']}"
                        ),
                    )
            except (TypeError, ValueError):
                pass  # non-numeric values skip numeric range check

        if "pattern" in rules and isinstance(value, str):
            import re

            try:
                if not re.match(rules["pattern"], value):
                    return FieldExtractionResponse(
                        field_name=response.field_name,
                        value=value,
                        confidence=response.confidence,
                        reasoning=response.reasoning,
                        is_valid=False,
                        validation_error=(
                            f"Value does not match pattern: {rules['pattern']}"
                        ),
                    )
            except re.error as e:
                logger.error(f"Invalid regex pattern {rules['pattern']!r}: {e}")
                return FieldExtractionResponse(
                    field_name=response.field_name,
                    value=value,
                    confidence=0.0,
                    reasoning=f"Invalid regex pattern: {e}",
                    is_valid=False,
                    validation_error=f"Invalid regex pattern: {e}",
                )

        # All checks passed — return with coerced value
        return FieldExtractionResponse(
            field_name=response.field_name,
            value=value,
            confidence=response.confidence,
            reasoning=response.reasoning,
            is_valid=True,
        )

    # ----------------------------------------------------------
    # Pass 1: Transition evaluation and execution
    # ----------------------------------------------------------

    def _execute_transition_evaluation_and_execution(
        self,
        instance: FSMInstance,
        user_message: str,
        extraction_response: DataExtractionResponse,
        conversation_id: str,
    ) -> tuple[bool, str | None]:
        """Evaluate transitions and execute if one is selected."""
        log = logger.bind(conversation_id=conversation_id)
        log.debug("Executing transition evaluation and execution")

        current_state = self.get_state(instance, conversation_id)

        if not current_state.transitions:
            log.debug("Terminal state reached - no transitions to evaluate")
            return False, None

        previous_state_id = instance.current_state

        evaluation = self.transition_evaluator.evaluate_transitions(
            current_state, instance.context, extraction_response.extracted_data
        )

        target_state = None

        if evaluation.result_type == TransitionEvaluationResult.DETERMINISTIC:
            target_state = evaluation.deterministic_transition
            log.info(f"Deterministic transition selected: {target_state}")

        elif evaluation.result_type == TransitionEvaluationResult.AMBIGUOUS:
            target_state = self._resolve_ambiguous_transition(
                evaluation, user_message, extraction_response, instance, conversation_id
            )
            log.info(
                f"LLM-assisted transition selected: {target_state}"
                if target_state
                else "LLM-assisted resolution selected no transition (stay)"
            )

        elif evaluation.result_type == TransitionEvaluationResult.BLOCKED:
            log.warning(f"Transitions blocked: {evaluation.blocked_reason}")
            return False, None

        if target_state:
            self._execute_state_transition(instance, target_state, conversation_id)
            return True, previous_state_id

        return False, None

    # ----------------------------------------------------------
    # Classification-based extraction
    # ----------------------------------------------------------

    def _classifier_connection_kwargs(
        self, config_model: str | None = None
    ) -> dict[str, Any]:
        """Connection settings the classifier must inherit from the LLM interface.

        Contract: ``config_model`` is a per-config model override (or ``None``).
        Returns ``dict(interface.kwargs)`` (``api_key``, ``api_base``, ...) plus
        ``timeout``, to be spread into ``Classifier(...)``, minus the names
        ``Classifier.__init__`` binds itself (``schema``, ``model``,
        ``config``). An interface lacking the attributes
        (``Mock(spec=LLMInterface)``, a custom interface) contributes ``{}``.
        Returns ``{}`` when ``config_model`` differs from the interface's
        model, so a key is never sent to another provider. The key is never
        logged.
        """
        # DECISION plan-2026-09-19T175721-21cd7f8e/D-008: guarded getattr, NOT
        # passing the interface to Classifier (it calls litellm.completion
        # directly) and NOT unconditional attribute access (the pipeline is
        # LLM-interface-agnostic; a bare interface must yield {}). Do NOT
        # inherit for a config `model` that differs from the interface's: that
        # would send the interface's api_key/api_base to a different provider.
        llm = self.llm_interface
        if config_model and config_model != getattr(llm, "model", None):
            return {}
        connection: dict[str, Any] = {}
        kwargs = getattr(llm, "kwargs", None)
        if isinstance(kwargs, dict):
            # DECISION plan-2026-09-19T175721-21cd7f8e/D-021: drop the names
            # Classifier binds itself; spreading them is a TypeError that the
            # extraction path swallows (silent classification skip). Do NOT
            # widen this to a filter on "known good" keys: litellm accepts
            # arbitrary provider kwargs.
            connection.update(
                {k: v for k, v in kwargs.items() if k not in _CLASSIFIER_BOUND_NAMES}
            )
        timeout = getattr(llm, "timeout", None)
        if isinstance(timeout, int | float) and not isinstance(timeout, bool):
            connection["timeout"] = timeout
        return connection

    def _get_classifier(
        self,
        schema: ClassificationSchema,
        model: str,
        prompt_config: ClassificationPromptConfig | None,
        connection_kwargs: dict[str, Any],
    ) -> Classifier:
        """Return the cached ``Classifier`` for this exact configuration, or build it.

        Contract: ``schema`` is the intent schema, ``model`` the resolved model
        name, ``prompt_config`` the per-entry prompt override (or ``None``) and
        ``connection_kwargs`` the output of ``_classifier_connection_kwargs``.
        Returns a ``Classifier`` whose four inputs are content-equal to the
        arguments; it never returns a classifier built for a different
        schema/model/config/connection. Cache misses construct via the
        module-level ``Classifier`` symbol (so ``patch("fsm_llm.pipeline.
        Classifier")`` keeps observing construction) and evict the oldest entry
        when the cache holds ``MAX_CLASSIFIER_CACHE_SIZE`` entries. When
        ``connection_kwargs`` holds a value ``json.dumps`` cannot serialise
        (e.g. a pydantic ``SecretStr``), the call bypasses the cache: a fresh
        ``Classifier`` is built and returned without a lookup, an insert or
        the lock. Never raises on its own; construction errors propagate to
        the caller unchanged.

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
        # DECISION plan-2026-09-20T165703-0d9c218e/D-001: the key is a content
        # hash of schema + model + prompt config + connection kwargs, NOT
        # `(state_id, field_name)`. A per-entry `model` override, a changed
        # `api_base`/`timeout`, or an edited prompt_config would otherwise reuse
        # a stale instance built for the old settings. The content hash makes
        # staleness impossible ONLY when every connection-kwarg value is
        # JSON-native (schema and prompt_config always are); a non-native
        # value (e.g. pydantic `SecretStr`, whose `str()` elides its state so
        # two different keys would collide, review W1) BYPASSES the cache:
        # construct fresh, never insert. Do NOT reintroduce a `str()` default
        # or a custom encoder: a redacting `__str__` cannot be made unique, so
        # bypassing is the only honest key. Do NOT key on identity or on the
        # state/field names. Cached instances retain `api_key`/connection
        # credentials for the pipeline's lifetime (up to
        # `MAX_CLASSIFIER_CACHE_SIZE` copies) by design (review N11). The
        # check/construct/evict/insert sequence below is ONE critical section
        # under `_classifier_cache_lock`: do NOT narrow the lock to the dict
        # writes only, the FIFO `next(iter())` eviction is what raced (review
        # W2). See decisions.md D-001.
        try:
            payload = json.dumps(
                {
                    "schema": schema.model_dump(),
                    "model": model,
                    "config": (
                        dataclasses.asdict(prompt_config) if prompt_config else None
                    ),
                    "conn": connection_kwargs,
                },
                sort_keys=True,
            )
        except (TypeError, ValueError):
            # Names only, never the values: a connection kwarg may be a secret.
            logger.debug(
                "Classifier cache bypassed: a connection kwarg is not "
                "JSON-native (kwargs: {}); constructing a fresh instance",
                sorted(connection_kwargs),
            )
            return Classifier(
                schema=schema,
                model=model,
                config=prompt_config,
                **connection_kwargs,
            )
        key = hashlib.sha256(payload.encode()).hexdigest()
        cache = self._classifier_cache
        with self._classifier_cache_lock:
            hit = cache.get(key)
            if hit is not None:
                return hit
            classifier = Classifier(
                schema=schema,
                model=model,
                config=prompt_config,
                **connection_kwargs,
            )
            if len(cache) >= MAX_CLASSIFIER_CACHE_SIZE:
                cache.pop(next(iter(cache)), None)
            cache[key] = classifier
        return classifier

    def _execute_classification_extractions(
        self,
        current_state: State,
        user_message: str,
        instance: FSMInstance,
        conversation_id: str,
        *,
        configs_override: list[ClassificationExtractionConfig] | None = None,
    ) -> dict[str, Any]:
        """Run classification extractions and return extracted data.

        For each :class:`ClassificationExtractionConfig`, builds a
        :class:`ClassificationSchema`, creates a :class:`Classifier`,
        and stores the result in two places:

        - ``field_name`` → intent string in the returned dict (simple,
          JsonLogic-friendly; fallback always, other intents only at or above
          ``confidence_threshold``)
        - ``context.metadata["classification_results"][field_name]`` → full
          result dict (intent, confidence, reasoning, entities, plus
          ``low_confidence`` when below ``confidence_threshold`` (a non-fallback
          intent is then discarded) and ``context_snapshot`` of the
          JSON-native ``context_keys`` values), written for every result
          (plan-2026-09-21T203800-8a03483a/D-005). Readable via
          ``FSMManager.get_complete_conversation``.

        Args:
            current_state: Current state (for config lookup).
            user_message: User input to classify.
            instance: FSM instance (for model fallback).
            conversation_id: Logging context.
            configs_override: If provided, run only these configs
                (used during retry).

        Returns:
            Dict of extracted key-value pairs to merge into context.
        """
        log = logger.bind(conversation_id=conversation_id)
        configs = configs_override or current_state.classification_extractions or []
        if not configs:
            return {}

        model = getattr(self.llm_interface, "model", None)
        extracted: dict[str, Any] = {}

        for config in configs:
            effective_model = config.model or model
            if not effective_model:
                if config.required:
                    raise ClassificationError(
                        f"Required classification extraction '{config.field_name}': "
                        "no LLM model available"
                    )
                log.warning(
                    f"Classification extraction '{config.field_name}': "
                    "no LLM model available, skipping"
                )
                continue

            try:
                schema = ClassificationSchema(
                    intents=config.intents,
                    fallback_intent=config.fallback_intent,
                    confidence_threshold=config.confidence_threshold,
                )

                prompt_config = None
                if config.prompt_config:
                    prompt_config = ClassificationPromptConfig(**config.prompt_config)

                classifier = self._get_classifier(
                    schema,
                    effective_model,
                    prompt_config,
                    self._classifier_connection_kwargs(config.model),
                )

                result: ClassificationResult = classifier.classify(
                    user_message,
                    context=self._build_classifier_context(
                        instance, current_state, conversation_id, config.context_keys
                    ),
                )

                log.debug(
                    f"Classification extraction '{config.field_name}': "
                    f"intent={result.intent}, confidence={result.confidence:.2f}"
                )

                # plan-2026-09-21T203800-8a03483a/D-005 (A4): the full result
                # of EVERY classification of this field (fallback and
                # discarded low-confidence ones too) is the inspectable
                # record; see _record_classification_result.
                full_result: dict[str, Any] = {
                    "intent": result.intent,
                    "confidence": result.confidence,
                    "reasoning": result.reasoning,
                    "entities": dict(result.entities),
                }
                if result.confidence < config.confidence_threshold:
                    full_result["low_confidence"] = True
                if config.context_keys:
                    full_result["context_snapshot"] = _json_native_values(
                        instance.context.data, config.context_keys
                    )
                _record_classification_result(instance, config.field_name, full_result)

                # Always store fallback intent so the context key exists
                # for downstream JsonLogic conditions
                if result.intent == config.fallback_intent:
                    extracted[config.field_name] = result.intent
                    log.debug(
                        f"Classification extraction '{config.field_name}': "
                        f"fallback intent '{result.intent}' stored"
                    )
                    continue

                # Skip low confidence
                if result.confidence < config.confidence_threshold:
                    # LV2-05 (D-037): an accepted stall (D-016), now visible: a
                    # gated key left unset here holds the state with no other trace.
                    log.warning(
                        f"Classification extraction '{config.field_name}': "
                        f"intent '{result.intent}' at confidence "
                        f"{result.confidence:.2f} is below threshold "
                        f"{config.confidence_threshold}, discarded; the key stays "
                        "unset and a transition gated on it will not fire"
                    )
                    continue

                # Store simple value (user-visible, works with JsonLogic)
                extracted[config.field_name] = result.intent

                log.info(
                    f"Classification extraction '{config.field_name}' = "
                    f"'{result.intent}' (confidence={result.confidence:.2f})"
                )

            except _CLASSIFICATION_SOFT_FAIL_EXCEPTIONS as e:
                if config.required:
                    raise ClassificationError(
                        f"Required classification extraction '{config.field_name}' "
                        f"failed: {e}"
                    ) from e
                log.warning(
                    f"Classification extraction '{config.field_name}' failed: {e}"
                )
                continue

        return extracted

    def _resolve_ambiguous_transition(
        self,
        evaluation: TransitionEvaluation,
        user_message: str,
        extraction_response: DataExtractionResponse,
        instance: FSMInstance,
        conversation_id: str,
    ) -> str | None:
        """Resolve ambiguous transition using classification.

        Classification is always-on for ambiguous transitions. Builds a
        ClassificationSchema from available transition options and uses
        the Classifier to make a structured, confidence-scored decision.

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

        schema = self._build_transition_classification_schema(
            current_state,
            evaluation.available_options,
        )

        model = getattr(self.llm_interface, "model", None)
        if model is None:
            raise InvalidTransitionError(
                "Cannot determine LLM model for classification-based "
                "transition resolution"
            )

        try:
            # Inside the try, as at the extraction site: a construction
            # failure degrades to "stay" like a classify() failure (review W2).
            classifier = self._get_classifier(
                schema, model, None, self._classifier_connection_kwargs()
            )
            result: ClassificationResult = classifier.classify(
                user_message,
                context=self._build_classifier_context(
                    instance, current_state, conversation_id
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
        low_confidence = result.confidence < schema.confidence_threshold

        # Store the classification record: data key + metadata mirror
        # (plan-2026-09-21T203800-8a03483a/D-005).
        record: dict[str, Any] = {
            "intent": result.intent,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
            "entities": dict(result.entities),
        }
        if low_confidence:
            record["low_confidence"] = True
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
    ) -> dict[str, Any]:
        """Per-call classifier context: ``{"history", "purpose", "data"}``.

        ``history`` is the last ``CLASSIFIER_HISTORY_EXCHANGES`` exchanges
        without the in-flight user message (the classifier receives that as
        its user turn). ``data`` is ``get_user_visible_data()`` (WorkingMemory
        included, data wins) scoped by ``context_keys`` when given, else by
        the state's ``read_keys``. Security filtering and sanitization happen
        where it is rendered (``prompts.build_classification_context_block``).
        """
        # DECISION plan-2026-09-21T203800-8a03483a/D-004: one builder for
        # both classifier call sites (extraction and ambiguous transition).
        # Do NOT pass raw instance.context.data or skip the scope: read_keys
        # must hide from the classifier what it hides from Pass 2, and the
        # result is per-call input to classify(), never a cache-key input.
        history = instance.context.conversation.get_recent(CLASSIFIER_HISTORY_EXCHANGES)
        if history and "user" in history[-1]:
            history = history[:-1]
        visible = instance.context.get_user_visible_data()
        if context_keys is not None:
            data = {k: v for k, v in visible.items() if k in context_keys}
        else:
            data = self._apply_context_scope(visible, state, conversation_id)
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
                "confidence_threshold",
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

        return ClassificationSchema(
            intents=intents,
            fallback_intent=TRANSITION_CLASSIFICATION_FALLBACK_INTENT,
            confidence_threshold=confidence_threshold,
        )

    def _execute_state_transition(
        self, instance: FSMInstance, target_state: str, conversation_id: str
    ) -> None:
        """Execute state transition with PRE/POST handler integration and rollback."""
        log = logger.bind(conversation_id=conversation_id)
        old_state = instance.current_state

        self.execute_handlers(
            instance,
            HandlerTiming.PRE_TRANSITION,
            conversation_id,
            current_state=old_state,
            target_state=target_state,
        )

        # Deep-copy full context for rollback if POST_TRANSITION handlers fail.
        #
        # DECISION plan-2026-09-20T114608-a8e47b88/D-024
        # Snapshot AND restore instance.context.metadata alongside
        # instance.context.data, mirroring the turn-atomicity rollback pattern
        # already used correctly at the pre_turn_* sites (search
        # `pre_turn_metadata` in this file: pipeline.py:475-479/521-526/
        # 606-611/647-652 all clear()+update() data AND metadata together from
        # one pre-captured deepcopy pair). Before D-018, `merge_delta` never
        # touched `metadata`, so a data-only snapshot here was safe -- D-018
        # made `merge_delta` ALSO pop a deleted key's provenance digest out of
        # `context.metadata[_PROVENANCE_KEY]` on a None-delta deletion. So a
        # POST_TRANSITION handler chain where an earlier handler deletes a
        # provenanced key (clearing its digest) and a LATER handler at the
        # same timing raises left the plaintext key restored on rollback but
        # its digest permanently gone -- data and metadata desynced. Do NOT
        # snapshot `data` only "because metadata isn't touched here" -- that
        # premise is no longer true. See decisions.md D-024.
        old_context_snapshot = copy.deepcopy(instance.context.data)
        old_metadata_snapshot = copy.deepcopy(instance.context.metadata)

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
                conversation_id,
                current_state=target_state,
                target_state=target_state,
            )
        except Exception as handler_err:
            log.warning(
                f"POST_TRANSITION handler failed ({type(handler_err).__name__}: {handler_err}), rolling back state from {target_state} to {old_state}"
            )
            instance.current_state = old_state
            instance.context.data.clear()
            instance.context.data.update(old_context_snapshot)
            instance.context.metadata.clear()
            instance.context.metadata.update(old_metadata_snapshot)
            raise

        log.info(f"State transition executed: {old_state} -> {target_state}")

    # ----------------------------------------------------------
    # Pass 2: Response generation
    # ----------------------------------------------------------

    def _execute_response_generation_pass(
        self,
        instance: FSMInstance,
        user_message: str,
        extraction_response: DataExtractionResponse,
        transition_occurred: bool,
        previous_state: str | None,
        conversation_id: str,
    ) -> str:
        """Execute Pass 2: Response Generation based on final state."""
        log = logger.bind(conversation_id=conversation_id)
        log.debug("Executing response generation pass")

        current_state = self.get_state(instance, conversation_id)

        # Fast-path for states with empty response_instructions (e.g. agent
        # intermediate states).  We build a minimal prompt and let the LLM
        # interface decide whether to skip the API call (LiteLLMInterface
        # returns a synthetic response for short system prompts).
        if (
            current_state.response_instructions is not None
            and not current_state.response_instructions
        ):
            request = ResponseGenerationRequest(
                system_prompt=".",
                user_message=user_message,
                extracted_data=extraction_response.extracted_data,
                context={},
                transition_occurred=transition_occurred,
                previous_state=previous_state,
            )
            response = self.llm_interface.generate_response(request)
            synthetic = f"[{current_state.id}]"
            instance.context.conversation.add_system_message(synthetic)
            log.debug("Skipped response generation (empty response_instructions)")
            return synthetic

        fsm_def = self.fsm_resolver(instance.fsm_id)

        system_prompt = self.response_generation_prompt_builder.build_response_prompt(
            instance=instance,
            state=current_state,
            fsm_definition=fsm_def,
            extracted_data=extraction_response.extracted_data,
            transition_occurred=transition_occurred,
            previous_state=previous_state,
            user_message=user_message,
            # DECISION plan-2026-09-19T175721-21cd7f8e/D-005: scope the
            # PROMPT, not only request.context (llm.py never reads it). Do NOT
            # revert to full instance.context.data: read_keys would then
            # promise scoping the prompt never delivered (hidden values leaked).
            # DECISION plan-2026-09-21T203800-8a03483a/D-008: the prompt gets
            # WorkingMemory under data (get_merged_data), scoped the same way,
            # at all three Pass-2 sites (sync, stream, greeting). Do NOT feed
            # WM through request.context only, and do NOT merge hidden buffers.
            context=self._apply_context_scope(
                instance.context.get_merged_data(), current_state, conversation_id
            ),
            # DECISION plan-2026-09-19T175721-21cd7f8e/D-032
            # read_keys scopes THIS channel too: a refused value is a context
            # value. Do NOT pass the raw dict (final review concern 3): the key
            # the state hides from <current_context> then leaks here.
            rejected_corrections=self._apply_context_scope(
                extraction_response.rejected_corrections, current_state, conversation_id
            ),
            extraction_failed=extraction_response.extraction_failed,
        )

        # Apply context scoping if the state defines read_keys
        context_for_llm = self._apply_context_scope(
            instance.context.get_user_visible_data(),
            current_state,
            conversation_id,
        )

        # Only enforce structured output format on terminal states
        output_response_format = None
        if not current_state.transitions:
            output_response_format = instance.context.data.get(
                "_output_response_format"
            )

        request = ResponseGenerationRequest(
            system_prompt=system_prompt,
            user_message=user_message,
            extracted_data=extraction_response.extracted_data,
            context=context_for_llm,
            transition_occurred=transition_occurred,
            previous_state=previous_state,
            response_format=output_response_format,
        )

        response = self.llm_interface.generate_response(request)
        instance.last_response_generation = response
        instance.context.conversation.add_system_message(response.message)

        log.debug("Response generation pass completed")
        return response.message

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
        unchanged (backward-compatible default).
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

    @staticmethod
    def _clean_empty_context_keys(
        data: dict[str, Any], conversation_id: str, remove_none_values: bool = True
    ) -> dict[str, Any]:
        """Clean invalid keys from context data. Delegates to context module."""
        return clean_context_keys(data, conversation_id, remove_none_values)
