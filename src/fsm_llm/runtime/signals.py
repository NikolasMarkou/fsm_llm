"""The signals of one frame of an event log (roadmap row P7.3).

Interface contract (callers: ``runtime.usage.watch_frame_signals``, tests):
    - ``FrameSignals``: frozen, plain tuples sorted by name, the count last. ``failures``:
      ``(code, n)`` of every ``provider/error`` (a missing or foreign code is ``unknown``).
      ``misses`` and ``out_of_scope``: ``(state, key or field, n)``. ``extraction_failed`` and
      ``cut``: ``(state, n)``. ``errors``: how many ``provider/error`` events were read.
    - ``frame_signals(events, frame) -> FrameSignals``: one pass over any iterable of events
      (``log.query(...)``). Failures count EVERY provider error of the frame. The rest comes only
      from a CLEAN turn window: ``turn/start`` to ``turn/end``, the outcome not ``rolled_back``
      and no provider error inside. No ``turn/end`` (a killed turn), a rolled-back or a tainted
      window gives nothing. ``seq`` is the order. Never reads a blob, a ``source`` or a ``ts``,
      never raises on an event it can read. Absence proves nothing (writes are best effort).
    - ``is_signal_name(value) -> bool``: the one rule for a state or key name a signal may carry
      (an ASCII identifier of at most ``SIGNAL_NAME_MAX`` characters). An event with any other
      name is not recorded: a model-written key such as ``a,b:c`` cannot shape a wire string.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from fsm_llm.core.codes import LlmFailureCode, Purpose
from fsm_llm.core.definitions import ASCII_IDENTIFIER_PATTERN
from fsm_llm.runtime.events import EventType, TurnOutcome

__all__ = ["SIGNAL_NAME_MAX", "FrameSignals", "frame_signals", "is_signal_name"]

SIGNAL_NAME_MAX = 80
_NAME = re.compile(ASCII_IDENTIFIER_PATTERN)

_KNOWN_CODES = frozenset(code.value for code in LlmFailureCode)
_CUT_PURPOSES = frozenset({Purpose.EXTRACT.value, Purpose.CLASSIFY.value})


@dataclass(frozen=True)
class FrameSignals:
    failures: tuple[tuple[str, int], ...] = ()
    misses: tuple[tuple[str, str, int], ...] = ()
    out_of_scope: tuple[tuple[str, str, int], ...] = ()
    extraction_failed: tuple[tuple[str, int], ...] = ()
    cut: tuple[tuple[str, int], ...] = ()
    errors: int = 0


def is_signal_name(value: object) -> bool:
    """Whether *value* may be a name in a signal (see the module docstring)."""
    return (
        isinstance(value, str)
        and len(value) <= SIGNAL_NAME_MAX
        and _NAME.fullmatch(value) is not None
    )


def _name(value: Any) -> str | None:
    return value if is_signal_name(value) else None


def _window(
    window: list[Mapping[str, Any]], end: Mapping[str, Any], found: dict[str, Counter]
) -> None:
    """Add the evidence of one finished turn window (nothing when it is not clean)."""
    kinds = {e.get("type") for e in window}
    if (
        end.get("outcome") == TurnOutcome.ROLLED_BACK.value
        or EventType.PROVIDER_ERROR.value in kinds
    ):
        return
    state = _name(window[0].get("state"))
    for event in window:
        kind, here = event.get("type"), _name(event.get("state"))
        if kind == EventType.EXECUTOR_RESULT.value and here:
            if event.get("extraction_failed") is True:
                found["bad"][(here,)] += 1
        elif kind == EventType.EXTRACTION_MISS.value and here:
            keys = event.get("keys")
            for key in keys if isinstance(keys, list) else ():
                if _name(key):
                    found["miss"][(here, key)] += 1
        elif kind == EventType.INTENT_OUT_OF_SCOPE.value and here:
            if _name(event.get("field")):
                found["scope"][(here, event["field"])] += 1
        elif (
            kind == EventType.LLM_RESPONSE.value
            and event.get("finish_reason") == "length"
            and event.get("purpose") in _CUT_PURPOSES
            and state
        ):
            found["cut"][(state,)] += 1


# DECISION plan-2026-11-25T074601-c720601d/D-012
# Failures count EVERY provider error of the frame; evidence comes only from clean windows (an
# outage turn also writes a miss and a blocked edge and ends rolled_back). Do NOT count failures
# only from clean windows, read a blob (a shape log has none), fold the session or stack, read
# `source` or `ts`, or take empty signals for "no outage". See decisions.md D-012.
def frame_signals(events: Iterable[Mapping[str, Any]], frame: int) -> FrameSignals:
    """The signals of *frame* in *events* (see the module docstring)."""
    found: dict[str, Counter] = {k: Counter() for k in ("bad", "miss", "scope", "cut")}
    failures: Counter = Counter()
    window: list[Mapping[str, Any]] | None = None
    for event in events:
        if event.get("frame") != frame:
            continue
        kind = event.get("type")
        if kind == EventType.PROVIDER_ERROR.value:
            code = event.get("failure_code")
            failures[
                code if code in _KNOWN_CODES else LlmFailureCode.UNKNOWN.value
            ] += 1
        if kind == EventType.TURN_START.value:
            window = [event]  # a start with no end before it drops the killed turn
        elif window is not None:
            window.append(event)
            if kind == EventType.TURN_END.value:
                _window(window, event, found)
                window = None

    def flat(name: str) -> tuple[tuple[Any, ...], ...]:
        return tuple(sorted((*key, n) for key, n in found[name].items()))

    return FrameSignals(
        failures=tuple(sorted(failures.items())),
        misses=flat("miss"),
        out_of_scope=flat("scope"),
        extraction_failed=flat("bad"),
        cut=flat("cut"),
        errors=sum(failures.values()),
    )
