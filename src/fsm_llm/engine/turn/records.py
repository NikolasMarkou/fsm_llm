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
