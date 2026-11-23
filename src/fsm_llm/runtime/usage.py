"""Usage as a projection of the event log (roadmap row 1.13, plan 5a6a1f80, step 9).

The meter of an interface (``fsm_llm.llm.usage``) and the events of a log are two records of the
same calls. They fold through ONE function and ONE record shape (``UsageRecord``, ``fold``), so
a count means the same thing in both. A reader picks ONE source: the ledger and the events are
never added (one outcome event exists per call, so a sum would count twice). Cost is not here
(row 2.1 owns a price source); tokens, calls and errors are.

Interface contract (callers: ``engine.api.calls`` (settle) for the outage reader, tests for the rest):
    - ``usage_from_events(events) -> LLMUsage``: folds the ``llm/response`` and ``provider/error``
      events of any iterable (``log.read(id)``, ``log.query(...)``); other events are skipped
      and a ``llm/request`` is never counted. An event is one call: a response is an answered
      call whose three inline token integers (all present) are its usage, any other response
      (an engine-logged reply, a stream, ``unrecorded: true``) counts as ``usage_missing``, an
      error is a call that raised. The kind is ``usage_kind(call_type, stream)``. Never reads a
      blob and never raises on an event it can read. One documented difference from the meter:
      a stream that raises while it is read is one error here, where the meter (which counts at
      the send) saw an answered call.
    - ``log_position(log, root_id) -> int``: the last ``seq`` of the tree (0 for none).
    - ``provider_errors_since(log, root_id, frame, after_seq) -> int``: the ``provider/error``
      events of that frame with ``seq > after_seq``. ``seq`` is the order, never a time.
    - ``watch_provider_errors(interface, log, root_id, frame) -> Callable[[], int]``: takes a
      position NOW and returns a reader of the provider errors made since. The sources, in
      order, each tried at read time and the next used when one raises: the event log (when
      *log* is given: the errors of *frame* after the position), the ledger of *interface*
      (a position that survives ``reset_usage()``), the ``usage().errors`` count of an
      interface that has its own ``usage()`` (read before and after, as the synthesis runner, since deleted, always did),
      and 0. An interface with no meter and no log reads 0 (documented limit).

Imports ``llm`` and ``runtime`` modules only.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any

from fsm_llm.core.definitions import LLMUsage
from fsm_llm.llm.usage import (
    UsageRecord,
    fold,
    ledger_mark,
    ledger_since,
    usage_kind,
)
from fsm_llm.runtime.constants import USAGE_TOKEN_FIELDS
from fsm_llm.runtime.events import EventType
from fsm_llm.runtime.log import EventLog

_RESPONSE = EventType.LLM_RESPONSE.value
_ERROR = EventType.PROVIDER_ERROR.value


def _tokens_of(event: Mapping[str, Any]) -> tuple[int, int, int] | None:
    """The three inline token integers of a response, or ``None`` unless all are present."""
    values = tuple(event.get(name) for name in USAGE_TOKEN_FIELDS)
    if all(type(v) is int and v >= 0 for v in values):
        return values  # type: ignore[return-value]
    return None


def _records(events: Iterable[Mapping[str, Any]]) -> Iterator[UsageRecord]:
    for event in events:
        kind = event.get("type")
        if kind != _RESPONSE and kind != _ERROR:
            continue
        call_type = event.get("call_type")
        yield UsageRecord(
            usage_kind(
                call_type if isinstance(call_type, str) else "",
                event.get("stream") is True,
            ),
            kind == _RESPONSE,
            _tokens_of(event) if kind == _RESPONSE else None,
        )


def usage_from_events(events: Iterable[Mapping[str, Any]]) -> LLMUsage:
    """The usage the ``llm/response`` and ``provider/error`` events of *events* describe."""
    return fold(_records(events))


def log_position(log: EventLog, root_id: str) -> int:
    """The last ``seq`` of the tree *root_id* (0 for an empty or missing log)."""
    last = 0
    for event in log.read(root_id):
        last = event["seq"]
    return last


def provider_errors_since(
    log: EventLog, root_id: str, frame: int, after_seq: int
) -> int:
    """How many ``provider/error`` events *frame* of the tree wrote after ``seq`` *after_seq*."""
    return sum(
        1
        for _ in log.query(
            root_id, types=[EventType.PROVIDER_ERROR], frame=frame, since_seq=after_seq
        )
    )


def _own_errors(interface: object) -> int:
    """The ``errors`` of an interface's own ``usage()`` (any failure is the caller's to catch)."""
    return int(interface.usage().errors)  # type: ignore[attr-defined]


def watch_provider_errors(
    interface: object, log: EventLog | None, root_id: str, frame: int
) -> Callable[[], int]:
    """A reader of the provider errors made since now (see the module docstring)."""
    readers: list[Callable[[], int]] = []
    if log is not None:
        try:
            after = log_position(log, root_id)
            readers.append(lambda: provider_errors_since(log, root_id, frame, after))
        except Exception:
            pass  # a source that cannot start is skipped; the next one answers
    try:
        mark = ledger_mark(interface)
        if mark is not None:
            readers.append(lambda: ledger_since(interface, mark).errors)
        else:
            before = _own_errors(interface)
            readers.append(lambda: max(_own_errors(interface) - before, 0))
    except Exception:
        pass  # no meter at all: only the log (or nothing) can answer

    def errors() -> int:
        for read in readers:
            try:
                return read()
            except Exception:
                continue  # fall back one level
        return 0

    return errors
