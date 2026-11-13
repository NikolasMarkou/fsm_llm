"""Module-level records and helpers of the turn pipeline: the turn snapshot and record, the
provenance, classification and cut-off helpers, the silent-state test.

Layer ``engine`` (plan d454fd62, step 79): split out of the old ``fsm_llm.pipeline``; the
method and function bodies are unchanged.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from typing import Any

from fsm_llm.constants import (
    CONTEXT_KEY_AGENT_TRACE,
    CONTEXT_KEY_RESPONSE_TEMPERATURE,
    MAX_CONTEXT_FILTER_DEPTH,
    METADATA_KEY_CLASSIFICATION_RESULTS,
    METADATA_KEY_TRANSITION_CLASSIFICATION,
    PROVENANCE_METADATA_KEY,
    is_forbidden_context_entry,
)
from fsm_llm.context.filtering import filter_context_tree
from fsm_llm.core.definitions import (
    ClassificationError,
    ClassificationResult,
    FSMInstance,
    State,
    TransitionEvaluationResult,
)

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
# DECISION plan-2026-10-01T093600-944e2692/D-039 (narrows the class list of
# D-001 above; the one-tuple rule stands): only the classifier's own error
# family is soft. Since D-006 the conversation's own, possibly custom,
# interface answers here, and the classifier turns every provider failure
# (LLMResponseError, `complete` being total) and every unreadable reply into
# a ClassificationError. Do NOT add ValueError, TypeError, KeyError,
# RuntimeError or OSError back: a bug in a custom `complete`, a broken
# prompt_config or a classifier refused at construction (no `complete`, a
# model mismatch) became a silent "stay" or a skipped field. See D-039.
# A missing `complete` is soft: `_get_classifier` raises ClassificationError
# for it (plan-2026-10-01T093600-944e2692/D-046).
_CLASSIFICATION_SOFT_FAIL_EXCEPTIONS: tuple[type[Exception], ...] = (
    ClassificationError,
)


def _response_temperature(instance: FSMInstance) -> float | None:
    """The conversation's Pass-2 sampling temperature, or ``None``.

    Contract: returns ``context.data[CONTEXT_KEY_RESPONSE_TEMPERATURE]``
    unchanged (``None`` when absent). Every Pass-2 request (sync, stream,
    greeting) carries it, and ``ResponseGenerationRequest`` refuses a value
    that is not a number in [0, 2] (a bool or a string included), so a bad
    value fails the turn instead of being dropped or coerced.
    """
    # DECISION plan-2026-10-01T093600-944e2692/D-038: a per-conversation
    # Pass-2 temperature travels on the request, so it applies through an
    # injected interface too. Do NOT read it at one Pass-2 site only, do NOT
    # coerce or drop a bad value, and do NOT pass it as an `API(temperature=)`
    # beside an injected interface (refused). See decisions.md D-038.
    value: float | None = instance.context.data.get(CONTEXT_KEY_RESPONSE_TEMPERATURE)
    return value


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
    prov = instance.context.metadata.setdefault(PROVENANCE_METADATA_KEY, {})
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
    """Store the turn's transition-classification record under
    ``metadata[METADATA_KEY_TRANSITION_CLASSIFICATION]``."""
    # DECISION plan-2026-09-30T062855-07ad3f8c/D-038: metadata is the ONLY
    # home of this record. Do NOT also write it to a ``context.data`` key
    # (the removed ``_transition_classification_result`` copy): two stores of
    # one record drift, and an internal-prefixed data key is dropped by
    # ``clean_context_keys`` anyway. Supersedes the data-key half of
    # plan-2026-09-21T203800-8a03483a/D-005.
    instance.context.metadata[METADATA_KEY_TRANSITION_CLASSIFICATION] = record


def _is_agent_managed(data: dict[str, Any]) -> bool:
    """True when ``data`` is the context of an agent-managed FSM (it carries the
    ``agent_trace`` key); the pipeline then skips the extra extraction calls."""
    return CONTEXT_KEY_AGENT_TRACE in data


def _classification_record(
    result: ClassificationResult, confidence_threshold: float
) -> dict[str, Any]:
    """The inspectable record of one classification: ``intent``, ``confidence``,
    ``reasoning``, ``entities``, plus ``low_confidence: True`` when the
    confidence is below ``confidence_threshold`` (the same ``<`` rule at every
    site). Returns a new dict the caller may extend."""
    record: dict[str, Any] = {
        "intent": result.intent,
        "confidence": result.confidence,
        "reasoning": result.reasoning,
        "entities": dict(result.entities),
    }
    if result.confidence < confidence_threshold:
        record["low_confidence"] = True
    return record


def _drop_forbidden_entry(key: Any, value: Any, _full_key: str) -> str | None:
    """``filter_context_tree`` predicate: drop a secret-shaped entry."""
    if isinstance(key, str) and is_forbidden_context_entry(key, value):
        return "forbidden security pattern"
    return None


def _json_native_values(data: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    """Independent JSON-native copies of ``data[k]`` for each present ``k``;
    a value that does not survive ``json.dumps`` is left out (never
    stringified, so no object repr lands in metadata). Secret-shaped entries
    are then dropped at every depth (``is_forbidden_context_entry``)."""
    out: dict[str, Any] = {}
    for key in keys:
        if key in data:
            try:
                out[key] = json.loads(json.dumps(data[key], allow_nan=False))
            except (TypeError, ValueError, RecursionError):
                continue
    # DECISION plan-2026-09-22T080837-8b258a25/D-031
    # Filter the STORED JSON copy with the never-truncating data walker. Do NOT
    # use the prompt walker (its node budget truncates), filter only the top
    # level (nested secrets leak to the monitor), or filter before the round
    # trip (a cycle would then be cut and stored instead of omitted). D-007.
    return filter_context_tree(out, MAX_CONTEXT_FILTER_DEPTH, _drop_forbidden_entry)


def _is_silent(state: State) -> bool:
    """True when ``state`` says nothing: ``response_instructions`` is ``""``.

    ``None`` (the field left out) is not silent: that state gets a default
    reply prompt. Read by the three Pass-2 entries (sync, stream, greeting).
    """
    return state.response_instructions is not None and not state.response_instructions


_NOT_CAPTURED: Any = object()


@dataclasses.dataclass(slots=True)
class _TurnSnapshot:
    """The in-memory instance state a failed turn rolls back (internal).

    Interface contract: ``capture(instance, working_memory=True)`` deep-copies
    ``context.data`` and ``context.metadata`` (and ``context.working_memory``
    unless ``working_memory=False``, the per-transition snapshot) and records
    ``current_state``; it returns the snapshot. ``restore(instance)`` puts them
    back: ``data`` and ``metadata`` are cleared and refilled in place,
    ``working_memory`` is rebound to the copy (skipped when it was not
    captured), ``current_state`` is reassigned. Neither call raises beyond what
    ``copy.deepcopy`` raises. The one list of what a turn can roll back lives
    here; the callers keep their own try shapes (07ad3f8c/D-007, D-025).
    """

    state: str
    data: dict[str, Any]
    metadata: dict[str, Any]
    working_memory: Any

    @classmethod
    def capture(
        cls, instance: FSMInstance, *, working_memory: bool = True
    ) -> _TurnSnapshot:
        return cls(
            state=instance.current_state,
            data=copy.deepcopy(instance.context.data),
            working_memory=(
                copy.deepcopy(instance.context.working_memory)
                if working_memory
                else _NOT_CAPTURED
            ),
            metadata=copy.deepcopy(instance.context.metadata),
        )

    def restore(self, instance: FSMInstance) -> None:
        instance.current_state = self.state
        instance.context.data.clear()
        instance.context.data.update(self.data)
        if self.working_memory is not _NOT_CAPTURED:
            instance.context.working_memory = self.working_memory
        instance.context.metadata.clear()
        instance.context.metadata.update(self.metadata)


@dataclasses.dataclass(slots=True)
class _TurnRecord:
    """What one turn of ``MessagePipeline._run_turn`` did (internal, not exported).

    Contract: created by ``_run_turn`` with ``state_before`` set and filled as
    the turn runs; returned only when the turn succeeded. ``transition_outcome``
    is ``None`` when no evaluation ran (terminal state). ``response`` is the
    Pass-2 text, or ``None`` when the turn ended on a silent state. A failed
    turn raises and returns no record.
    """

    state_before: str
    state_after: str
    transition_outcome: TransitionEvaluationResult | None = None
    response: str | None = None
