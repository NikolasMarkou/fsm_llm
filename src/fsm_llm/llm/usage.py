"""Usage counters: the per-instance provider-call ledger and ``_Metered`` (``usage``,
``reset_usage``) shared by ``LiteLLMInterface`` and ``LiteLLMEmbedder``.

Layer ``llm`` (plan d454fd62, step 73): split out of the old ``fsm_llm/llm/__init__.py`` (the old ``llm.py``).
Plan 5a6a1f80, step 9 (roadmap row 1.13): the meter is a ledger with a marker and the module is
the home of the ONE record shape and the ONE fold that the event log projection
(``fsm_llm.runtime.usage``) shares.

Interface contract (public names; ``_UsageMeter`` and ``_Metered`` stay private):
    - ``UsageRecord(kind, ok, counts)``: one provider call. ``ok`` false is a call that raised;
      ``counts`` is ``(prompt, completion, total)`` tokens of an answered call, ``None`` when the
      reply carried no usage (it then counts as ``usage_missing``).
    - ``fold(records) -> LLMUsage``: totals plus ``by_kind`` of any iterable of records; the
      only place an ``LLMUsage`` is built. Never raises on a record it can read.
    - ``usage_counts(reply)``: the one defensive token reader, ``None`` when absent.
    - ``usage_kind(call_type, stream)``: the kind of a call, the one rule of ``_send`` and of the
      projection.
    - ``ledger_mark(owner)`` and ``ledger_since(owner, mark)``: a position in the ledger of a
      metered interface (``None`` for any object that has none) and the usage since it. A mark
      survives ``reset_usage()``.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping
from typing import Any, NamedTuple

from fsm_llm.constants import (
    USAGE_KIND_BY_CALL_TYPE,
    USAGE_KIND_COMPLETE,
    USAGE_KIND_STREAM,
)
from fsm_llm.core.definitions import LLMCallCounts, LLMUsage

# The counters of one kind: exactly the fields of ``LLMCallCounts``.
_COUNTER_FIELDS = tuple(LLMCallCounts.model_fields)


def _reply_field(obj: Any, name: str) -> Any:
    """``obj[name]`` for a dict, ``obj.name`` otherwise; ``None`` when absent."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _token_count(value: Any) -> int:
    """A token count as an ``int``; anything that is not a non-negative int is 0."""
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return 0


def usage_counts(response: Any) -> tuple[int, int, int] | None:
    """(prompt, completion, total) tokens of a provider reply; ``None`` if absent.

    Read defensively (the rules of the deleted agents bench reader):
    ``usage`` may be a dict or an object, may be missing (streamed replies
    carry none), and its fields may be missing or not ints. A missing
    ``total_tokens`` is the sum of the other two.
    """
    usage = _reply_field(response, "usage")
    if usage is None:
        return None
    prompt = _token_count(_reply_field(usage, "prompt_tokens"))
    completion_tokens = _token_count(_reply_field(usage, "completion_tokens"))
    total = _token_count(_reply_field(usage, "total_tokens"))
    return prompt, completion_tokens, total or prompt + completion_tokens


class UsageRecord(NamedTuple):
    """One provider call, as the meter and the event log both see it (see the module docstring)."""

    kind: str
    ok: bool
    counts: tuple[int, int, int] | None = None


#: The counters of every kind: ``{kind: {counter name: n}}`` (a mark is a copy of it).
UsageMark = dict[str, dict[str, int]]


def usage_kind(call_type: str, stream: bool) -> str:
    """The usage kind of a call: ``stream`` when streamed, else by call type, else ``complete``."""
    if stream:
        return USAGE_KIND_STREAM
    return USAGE_KIND_BY_CALL_TYPE.get(call_type, USAGE_KIND_COMPLETE)


def _record_deltas(record: UsageRecord) -> dict[str, int]:
    """What one record adds to the counters of its kind (the one rule of meter and log)."""
    if not record.ok:
        return {"calls": 1, "errors": 1}
    if record.counts is None:
        return {"calls": 1, "usage_missing": 1}
    prompt, completion_tokens, total = record.counts
    return {
        "calls": 1,
        "prompt_tokens": prompt,
        "completion_tokens": completion_tokens,
        "total_tokens": total,
    }


def _usage_from(per_kind: Mapping[str, Mapping[str, int]]) -> LLMUsage:
    """The one builder of an ``LLMUsage``: totals and ``by_kind`` of per-kind counters.

    A kind with every counter at zero is left out (``by_kind`` lists kinds with a call).
    """
    kept = {
        kind: counts
        for kind, counts in per_kind.items()
        if any(counts[name] for name in _COUNTER_FIELDS)
    }
    totals = {
        name: sum(counts[name] for counts in kept.values()) for name in _COUNTER_FIELDS
    }
    return LLMUsage(
        **totals,
        by_kind={
            kind: LLMCallCounts(**{name: counts[name] for name in _COUNTER_FIELDS})
            for kind, counts in sorted(kept.items())
        },
    )


def fold(records: Iterable[UsageRecord]) -> LLMUsage:
    """The usage of *records*: totals plus ``by_kind`` (see the module docstring)."""
    per_kind: dict[str, dict[str, int]] = {}
    for record in records:
        counts = per_kind.setdefault(record.kind, dict.fromkeys(_COUNTER_FIELDS, 0))
        for name, delta in _record_deltas(record).items():
            counts[name] += delta
    return _usage_from(per_kind)


def _minus(
    now: Mapping[str, Mapping[str, int]], then: Mapping[str, Mapping[str, int]]
) -> dict[str, dict[str, int]]:
    """The counters of *now* less those of *then* (a kind *then* lacks counts from zero)."""
    return {
        kind: {name: counts[name] - then.get(kind, {}).get(name, 0) for name in counts}
        for kind, counts in now.items()
    }


# DECISION plan-2026-11-18T071726-5a6a1f80/D-014: the ledger only grows and ``reset_usage`` moves
# a marker. This SUPERSEDES D-004 of plan 944e2692 on "a reset clears the counters": a reader
# that holds a position (``mark`` and ``since``, the synthesis runner's outage check) must
# outlive a ``reset_usage()`` another reader made. Do NOT clear ``_counts`` on reset, do NOT
# share a ledger or a marker between interface instances (a per-instance marker keeps two
# readers apart) and NEVER add the ledger to a projection of the event log (one outcome event
# exists per call: a reader picks ONE source). The rest of D-004 stands: fed in ``_send`` only.
# See decisions.md D-014 and D-027.
class _UsageMeter:
    """The provider-call ledger of one interface instance: counters by kind, plus a marker.

    Interface contract (shared by every provider call site of ``llm/providers/litellm.py``):
    - ``record(kind, response)``: one answered call of ``kind``; its token
      usage is read with ``usage_counts`` (a streamed call passes ``None`` and
      counts as usage-missing). Never raises on a strange reply shape.
    - ``record_error(kind)``: one call of ``kind`` that raised.
    - ``snapshot(reset=...)``: a frozen ``LLMUsage`` of the calls since the marker, totals
      plus ``by_kind``; with ``reset=True`` the marker moves to now in the same critical
      section, so no call is lost between the read and the reset and nothing is forgotten.
    - ``mark()`` and ``since(mark)``: a copy of the ledger now and the usage since a copy.
    Every method holds one lock: safe when one interface is shared by
    concurrent conversations. The lock is never held across a provider call.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: dict[str, dict[str, int]] = {}  # the ledger: never cleared
        self._marker: dict[str, dict[str, int]] = {}  # the ledger at the last reset

    def _bump(self, kind: str, **deltas: int) -> None:
        with self._lock:
            counts = self._counts.setdefault(kind, dict.fromkeys(_COUNTER_FIELDS, 0))
            for name, delta in deltas.items():
                counts[name] += delta

    def record(self, kind: str, response: Any) -> None:
        record = UsageRecord(kind, True, usage_counts(response))
        self._bump(kind, **_record_deltas(record))

    def record_error(self, kind: str) -> None:
        self._bump(kind, **_record_deltas(UsageRecord(kind, False)))

    def mark(self) -> UsageMark:
        with self._lock:
            return {kind: dict(counts) for kind, counts in self._counts.items()}

    def since(self, mark: UsageMark) -> LLMUsage:
        return _usage_from(_minus(self.mark(), mark))

    def snapshot(self, *, reset: bool = False) -> LLMUsage:
        with self._lock:
            ledger = {kind: dict(counts) for kind, counts in self._counts.items()}
            view = _minus(ledger, self._marker)
            if reset:
                self._marker = ledger
        return _usage_from(view)


# Guards the lazy creation of a meter, so two threads making an instance's
# first calls at once share one meter instead of each creating its own.
_METER_CREATION_LOCK = threading.Lock()


def _meter_of(owner: Any) -> _UsageMeter:
    """The ``_usage_meter`` of ``owner``, created on first use.

    Lazy so that an instance built with ``__new__`` (no ``__init__``) still
    counts. ``owner`` is any object that declares a class attribute
    ``_usage_meter: _UsageMeter | None = None``.
    """
    meter: _UsageMeter | None = owner._usage_meter
    if meter is None:
        with _METER_CREATION_LOCK:
            meter = owner._usage_meter
            if meter is None:
                meter = _UsageMeter()
                owner._usage_meter = meter
    return meter


def _ledger_of(owner: object) -> _UsageMeter | None:
    """The ledger of *owner*: only an interface whose ``usage`` is the one ``_Metered`` gives.

    An object that overrides ``usage()`` reports its own counters, which this module cannot
    see, so it has no ledger here.
    """
    if isinstance(owner, _Metered) and type(owner).usage is _Metered.usage:
        return _meter_of(owner)
    return None


def ledger_mark(owner: object) -> UsageMark | None:
    """A position in the ledger of *owner*, or ``None`` when it has no ledger of its own."""
    meter = _ledger_of(owner)
    return None if meter is None else meter.mark()


def ledger_since(owner: object, mark: UsageMark) -> LLMUsage:
    """The usage of *owner* since *mark* (``ValueError`` for an owner with no ledger)."""
    meter = _ledger_of(owner)
    if meter is None:
        raise ValueError(f"{type(owner).__name__} has no usage ledger")
    return meter.since(mark)


class _Metered:
    """The provider-call ledger shared by ``LiteLLMInterface`` and the embedder.

    Interface contract: a subclass sends every provider request through
    ``_meter_of(self)`` and gets ``usage`` and ``reset_usage`` for free. The
    ``_usage_meter`` class attribute (``None``) is what lets an instance built
    with ``__new__`` count too.
    """

    # Provider-call counters, created on first use (see _meter_of).
    _usage_meter: _UsageMeter | None = None

    def usage(self) -> LLMUsage:
        """A frozen snapshot of this object's provider-call counters.

        Counts every provider request it sent since it was built or since the
        last ``reset_usage``: totals plus ``by_kind`` (see ``LLMUsage``). Safe
        to call while other threads use it.
        """
        return _meter_of(self).snapshot()

    def reset_usage(self) -> LLMUsage:
        """Return the snapshot of the counters and start counting from zero.

        The read and the new start are one atomic step, so a request made
        concurrently is counted either in the returned snapshot or in the
        next one, never lost. The ledger itself is kept (a marker moves), so a
        position taken before the reset still reads the calls made since.
        """
        return _meter_of(self).snapshot(reset=True)
