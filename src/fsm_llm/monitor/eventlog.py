"""Read-only browse of a durable ``EventLog`` for the dashboard (roadmap row 1.10, first slice).

Only the public readers are used (``conversation_ids``, ``query``, ``read``,
``usage_from_events``): nothing here appends, forks, resumes or prunes, and a
log the host owns is never locked. A shown event is its envelope (``seq``,
``ts``, ``type``, ``frame``) plus the fields ``SHAPE_FIELDS`` allows for its
type (ids, names, counts, outcomes, codes): never a ``*_ref``, a blob, a
prompt, a reply or user text, whatever the privacy level of the log. A log
file that is a symbolic link is skipped (the reader would follow it).
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any

from fsm_llm.runtime import EventLog, EventType
from fsm_llm.runtime.constants import LOG_DIR
from fsm_llm.runtime.privacy import shape_only
from fsm_llm.runtime.stores.jsonl import LOG_SUFFIX
from fsm_llm.runtime.usage import usage_from_events

from .collector import redact_context
from .constants import MAX_EVENTLOG_CONVERSATIONS, MAX_EVENTLOG_EVENTS


def _regular_ids(log: EventLog) -> list[str]:
    """Ids whose log file is a regular file (a link is skipped, never followed)."""
    found = []
    for conversation_id in log.conversation_ids():
        path = Path(log.root) / LOG_DIR / f"{conversation_id}{LOG_SUFFIX}"
        try:
            if stat.S_ISREG(os.lstat(path).st_mode):
                found.append(conversation_id)
        except OSError:
            continue
    return found


def list_logs(log: EventLog) -> list[str]:
    """The ids of the first ``MAX_EVENTLOG_CONVERSATIONS`` logs, sorted."""
    return _regular_ids(log)[:MAX_EVENTLOG_CONVERSATIONS]


def _shown(event: dict[str, Any]) -> dict[str, Any]:
    try:
        fields = shape_only(EventType.from_string(str(event.get("type"))), event)
    except ValueError:  # a type this version does not know: envelope only
        fields = {}
    envelope = {k: event.get(k) for k in ("seq", "ts", "type", "frame")}
    return redact_context({**envelope, **fields})  # type: ignore[no-any-return]


def timeline(
    log: EventLog, conversation_id: str, *, since_seq: int = 0, limit: int = 200
) -> dict[str, Any]:
    """``{conversation_id, events, next_seq, usage}`` for one tree.

    At most ``MAX_EVENTLOG_EVENTS`` events after ``since_seq`` (``next_seq`` is
    the last ``seq`` shown when more may follow, else None). ``usage`` is the
    fold of the whole log (``runtime.usage``), shown beside the meter, never
    added to it. Raises ``FileNotFoundError`` (404) for an id with no regular
    log file; no path is built from the caller's text.
    """
    if conversation_id not in _regular_ids(log):
        raise FileNotFoundError(conversation_id)
    limit = max(1, min(limit, MAX_EVENTLOG_EVENTS))
    events = [
        _shown(e) for e in log.query(conversation_id, since_seq=since_seq, limit=limit)
    ]
    return {
        "conversation_id": conversation_id,
        "events": events,
        "next_seq": events[-1]["seq"] if len(events) == limit else None,
        "usage": usage_from_events(log.read(conversation_id)).model_dump(),
    }
