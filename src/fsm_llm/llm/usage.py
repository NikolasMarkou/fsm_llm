"""Usage counters: the per-instance provider-call meter and ``_Metered`` (``usage``,
``reset_usage``) shared by ``LiteLLMInterface`` and ``LiteLLMEmbedder``.

Layer ``llm`` (plan d454fd62, step 73): split out of the old ``fsm_llm/llm/__init__.py`` (the old ``llm.py``); definitions unchanged.
"""

from __future__ import annotations

import threading
from typing import Any

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


def _usage_of(response: Any) -> tuple[int, int, int] | None:
    """(prompt, completion, total) tokens of a provider reply; ``None`` if absent.

    Read defensively (the rules of ``scripts/agents_bench.py``'s reader):
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


class _UsageMeter:
    """Per-kind provider-call counters of one interface instance.

    Interface contract (shared by every provider call site of ``llm/providers/litellm.py``):
    - ``record(kind, response)``: one answered call of ``kind``; its token
      usage is read with ``_usage_of`` (a streamed call passes ``None`` and
      counts as usage-missing). Never raises on a strange reply shape.
    - ``record_error(kind)``: one call of ``kind`` that raised.
    - ``snapshot(reset=...)``: a frozen ``LLMUsage`` copy, totals plus
      ``by_kind``; with ``reset=True`` the counters are cleared in the same
      critical section, so no call is lost between the read and the reset.
    Every method holds one lock: safe when one interface is shared by
    concurrent conversations. The lock is never held across a provider call.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: dict[str, dict[str, int]] = {}

    def _bump(self, kind: str, **deltas: int) -> None:
        with self._lock:
            counts = self._counts.setdefault(kind, dict.fromkeys(_COUNTER_FIELDS, 0))
            for name, delta in deltas.items():
                counts[name] += delta

    def record(self, kind: str, response: Any) -> None:
        usage = _usage_of(response)
        if usage is None:
            self._bump(kind, calls=1, usage_missing=1)
            return
        prompt, completion_tokens, total = usage
        self._bump(
            kind,
            calls=1,
            prompt_tokens=prompt,
            completion_tokens=completion_tokens,
            total_tokens=total,
        )

    def record_error(self, kind: str) -> None:
        self._bump(kind, calls=1, errors=1)

    def snapshot(self, *, reset: bool = False) -> LLMUsage:
        with self._lock:
            per_kind = {kind: dict(counts) for kind, counts in self._counts.items()}
            if reset:
                self._counts = {}
        totals = {
            name: sum(counts[name] for counts in per_kind.values())
            for name in _COUNTER_FIELDS
        }
        return LLMUsage(
            **totals,
            by_kind={
                kind: LLMCallCounts(**counts)
                for kind, counts in sorted(per_kind.items())
            },
        )


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


class _Metered:
    """The provider-call counters shared by ``LiteLLMInterface`` and the embedder.

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
        """Clear the counters and return the snapshot they held.

        The read and the clear are one atomic step, so a request made
        concurrently is counted either in the returned snapshot or in the
        next one, never lost.
        """
        return _meter_of(self).snapshot(reset=True)
