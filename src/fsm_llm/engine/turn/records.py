"""Module-level records and helpers of the turn pipeline: the turn snapshot and record, the
Pass-2 temperature, the transition-classification record, the silent-state test.

Layer ``engine`` (plan d454fd62, step 79): split out of the old ``fsm_llm.pipeline``; the
method and function bodies are unchanged. Step 80 (D-030) moved the helpers the kind executors
also use (provenance, classification record, agent-managed test, JSON-native copy) down to
``engine/kinds/records.py``.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

from fsm_llm.constants import (
    CONTEXT_KEY_RESPONSE_TEMPERATURE,
    METADATA_KEY_TRANSITION_CLASSIFICATION,
)
from fsm_llm.core.definitions import (
    FSMInstance,
    State,
    TransitionEvaluationResult,
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
    ``current_state``; it returns the snapshot. The whole-turn form also keeps
    the three introspection records ``last_extraction_response``,
    ``last_transition_decision`` and ``last_response_generation`` (references:
    a turn replaces them by assignment, never in place) and the conversation
    history: a copy of ``exchanges`` and the ``summary`` as they stand at the
    capture (the manager has already added the user message of a message turn).
    A copy and not a length, because adding a reply trims the oldest exchanges
    into the summary. ``restore(instance)`` puts them back: ``data`` and
    ``metadata`` are cleared and refilled in place, ``working_memory``, the three
    records and the history are rebound to the captured values (``exchanges``
    refilled in place; all skipped when ``working_memory=False``: the
    per-transition rollback keeps the records the turn has already set and
    touches no history), ``current_state`` is reassigned. A restore leaves that
    user message in place: the manager arms pop it.
    Neither call raises beyond what ``copy.deepcopy`` raises. The one list of
    what a turn can roll back lives here; the callers keep their own try shapes
    (07ad3f8c/D-007, D-025).
    """

    state: str
    data: dict[str, Any]
    metadata: dict[str, Any]
    working_memory: Any
    last_extraction_response: Any = _NOT_CAPTURED
    last_transition_decision: Any = _NOT_CAPTURED
    last_response_generation: Any = _NOT_CAPTURED
    exchanges: Any = _NOT_CAPTURED
    summary: Any = _NOT_CAPTURED

    @classmethod
    def capture(
        cls, instance: FSMInstance, *, working_memory: bool = True
    ) -> _TurnSnapshot:
        whole_turn = working_memory
        return cls(
            state=instance.current_state,
            data=copy.deepcopy(instance.context.data),
            working_memory=(
                copy.deepcopy(instance.context.working_memory)
                if whole_turn
                else _NOT_CAPTURED
            ),
            metadata=copy.deepcopy(instance.context.metadata),
            last_extraction_response=(
                instance.last_extraction_response if whole_turn else _NOT_CAPTURED
            ),
            last_transition_decision=(
                instance.last_transition_decision if whole_turn else _NOT_CAPTURED
            ),
            last_response_generation=(
                instance.last_response_generation if whole_turn else _NOT_CAPTURED
            ),
            # The whole-turn snapshot holds the HISTORY too (D-002 of plan 8525c797, review
            # round): an interrupt that lands after Pass 2 added the reply restored the state
            # and kept both the user message and the reply, so a retry ran on a stale reply.
            # Do NOT capture a length and truncate to it (adding a reply trims the oldest
            # exchanges into `summary`, so a length cuts the wrong entries and the summary keeps
            # the trimmed text: copy the list and the summary), do NOT restore it in the
            # per-transition snapshot, and do NOT pop the user message here (the manager arms
            # do, only while it is the last entry). GeneratorExit reaches no restore, so an
            # abandoned stream keeps its partial reply.
            exchanges=(
                [dict(e) for e in instance.context.conversation.exchanges]
                if whole_turn
                else _NOT_CAPTURED
            ),
            summary=(
                instance.context.conversation.summary if whole_turn else _NOT_CAPTURED
            ),
        )

    def restore(self, instance: FSMInstance) -> None:
        instance.current_state = self.state
        instance.context.data.clear()
        instance.context.data.update(self.data)
        if self.working_memory is not _NOT_CAPTURED:
            instance.context.working_memory = self.working_memory
            instance.last_extraction_response = self.last_extraction_response
            instance.last_transition_decision = self.last_transition_decision
            instance.last_response_generation = self.last_response_generation
            conversation = instance.context.conversation
            conversation.exchanges[:] = self.exchanges
            conversation.summary = self.summary
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
