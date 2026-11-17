"""The checkpoint set and the recovery rule of the event log.

A checkpoint event is appended and then ``fsync``ed before the thing it announces happens: the
model call (``llm/request``), a side-effecting tool call (``tool/call``) and the end of a turn
(``turn/end``). One ``fsync`` flushes every earlier append of the file, so the other events
cost no ``fsync`` of their own. Power-loss durability is argued from that rule, not tested.

The recovery rule: a ``tool/call`` with no ``tool/result`` of the same ``call_id`` was in
flight when the process died (or was interrupted). Whether the tool ran, ran halfway or never
started cannot be known from the log, so it is reported as ``unknown`` and nothing retries it.

Imports ``runtime.events`` only.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from fsm_llm.runtime.events import EventType

#: Event types that are a checkpoint whatever their fields. A ``tool/call`` is one when it is
#: side-effecting (see ``is_checkpoint``).
CHECKPOINT_TYPES: frozenset[EventType] = frozenset(
    {EventType.LLM_REQUEST, EventType.TURN_END}
)

#: The ``status`` of a tool call the log cannot resolve.
UNKNOWN_STATUS = "unknown"


def is_checkpoint(event_type: EventType | str, fields: Mapping[str, Any]) -> bool:
    """Return True when an event of *event_type* with *fields* must be fsynced.

    ``CHECKPOINT_TYPES`` always are. A ``tool/call`` is unless its ``side_effecting`` field is
    exactly ``False`` (unknown means side-effecting: a call that does not say is assumed to
    change the world). Raises ``ValueError`` for a type outside the vocabulary.
    """
    kind = EventType.from_string(event_type)
    if kind in CHECKPOINT_TYPES:
        return True
    return kind is EventType.TOOL_CALL and fields.get("side_effecting") is not False


def unresolved_tool_calls(events: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return the tool calls of *events* that have no result, each with ``status: "unknown"``.

    *events* are decoded envelopes in ``seq`` order (``EventLog.read``). A ``tool/result``
    resolves the OLDEST still-open ``tool/call`` of its ``call_id`` (a reused id is matched
    one result per call). Each report is a plain dict with ``status`` (``"unknown"``), ``seq``,
    ``frame``, ``call_id``, ``name``, ``side_effecting`` and the call's ``arguments`` or
    ``arguments_ref`` when it has one; reports are in ``seq`` order. Never executes or retries
    anything, and never raises on an event of another type.
    """
    open_calls: dict[Any, list[Mapping[str, Any]]] = {}
    for event in events:
        kind = event.get("type")
        if kind == EventType.TOOL_CALL.value:
            open_calls.setdefault(event.get("call_id"), []).append(event)
        elif kind == EventType.TOOL_RESULT.value:
            pending = open_calls.get(event.get("call_id"))
            if pending:
                pending.pop(0)
    unresolved = [call for calls in open_calls.values() for call in calls]
    unresolved.sort(key=lambda call: call["seq"])
    return [_report(call) for call in unresolved]


def _report(call: Mapping[str, Any]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "status": UNKNOWN_STATUS,
        "seq": call["seq"],
        "frame": call["frame"],
        "call_id": call.get("call_id"),
        "name": call.get("name"),
        "side_effecting": call.get("side_effecting") is not False,
    }
    for key in ("arguments", "arguments_ref"):
        if key in call:
            report[key] = call[key]
    return report
