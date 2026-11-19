"""Read-only browse of a durable ``EventLog`` for the dashboard (roadmap row 1.10, first slice).

Only the public readers are used (``conversation_ids``, ``query``, ``read``,
``usage_from_events``): nothing here appends, forks, resumes or prunes, and a
log the host owns is never locked. A shown event is its envelope (``seq``,
``ts``, ``type``, ``frame``) plus the fields ``SHAPE_FIELDS`` allows for its
type (ids, names, counts, outcomes, codes): never a ``*_ref``, a blob, a
prompt, a reply or user text, whatever the privacy level of the log. A log
file that is a symbolic link is skipped (the reader would follow it).

A page costs one ``lstat`` of its own log and a cached usage fold (D-056 of plan
51fde95b): the fold of an unchanged file is computed once, keyed by path,
``st_mtime_ns`` and ``st_size``, at most ``MAX_EVENTLOG_USAGE_CACHE`` kept. A log
with a damaged line is shown up to it (``skipped`` 1, ``corrupt_after_seq``).
"""

from __future__ import annotations

import copy
import os
import stat
import threading
from collections import OrderedDict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from fsm_llm.runtime import EventLog, EventType
from fsm_llm.runtime.constants import LOG_DIR
from fsm_llm.runtime.events import EventPayloadError, LogCorruptError
from fsm_llm.runtime.privacy import shape_only
from fsm_llm.runtime.stores.jsonl import LOG_SUFFIX, check_conversation_id
from fsm_llm.runtime.usage import usage_from_events

from .collector import redact_context
from .constants import (
    MAX_EVENTLOG_CONVERSATIONS,
    MAX_EVENTLOG_EVENTS,
    MAX_EVENTLOG_USAGE_CACHE,
)

# (log root, conversation id) -> ((st_mtime_ns, st_size), usage dict, corrupt_after_seq)
_USAGE_CACHE: OrderedDict[
    tuple[str, str], tuple[tuple[int, int], dict[str, Any], int | None]
] = OrderedDict()
_USAGE_LOCK = threading.Lock()


def _is_regular(path: Path) -> bool:
    """Whether *path* is a regular file (a link is skipped, never followed)."""
    try:
        return stat.S_ISREG(os.lstat(path).st_mode)
    except OSError:
        return False


def list_logs(log: EventLog) -> list[str]:
    """The ids of the first ``MAX_EVENTLOG_CONVERSATIONS`` regular logs, sorted
    (the scan stops there: the rest of the directory is not ``lstat``ed)."""
    found: list[str] = []
    for conversation_id in log.conversation_ids():
        if _is_regular(Path(log.root) / LOG_DIR / f"{conversation_id}{LOG_SUFFIX}"):
            found.append(conversation_id)
            if len(found) == MAX_EVENTLOG_CONVERSATIONS:
                break
    return found


def _fold(log: EventLog, conversation_id: str) -> tuple[dict[str, Any], int | None]:
    """``(usage, corrupt_after_seq)`` of the readable prefix of one log: the fold
    stops at the first line the runtime refuses and says the last good ``seq``."""
    last, damaged = 0, False

    def events() -> Iterator[dict[str, Any]]:
        nonlocal last, damaged
        try:
            for event in log.read(conversation_id):
                last = event["seq"]
                yield event
        except LogCorruptError:
            damaged = True

    usage = usage_from_events(events()).model_dump()
    return usage, (last if damaged else None)


def _usage(
    log: EventLog, conversation_id: str, seen: os.stat_result
) -> tuple[dict[str, Any], int | None]:
    """``_fold`` cached by ``(root, id)`` and validated by the file's mtime and
    size, so an unchanged log is folded once and a grown one again (D-056)."""
    key = (str(log.root), conversation_id)
    stamp = (seen.st_mtime_ns, seen.st_size)
    with _USAGE_LOCK:
        hit = _USAGE_CACHE.get(key)
        if hit is not None and hit[0] == stamp:
            _USAGE_CACHE.move_to_end(key)
            return copy.deepcopy(hit[1]), hit[2]
    usage, corrupt_after = _fold(log, conversation_id)
    with _USAGE_LOCK:
        _USAGE_CACHE[key] = (stamp, usage, corrupt_after)
        _USAGE_CACHE.move_to_end(key)
        while len(_USAGE_CACHE) > MAX_EVENTLOG_USAGE_CACHE:
            _USAGE_CACHE.popitem(last=False)
    return copy.deepcopy(usage), corrupt_after


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
    """``{conversation_id, events, next_seq, usage, skipped, corrupt_after_seq}``.

    At most ``MAX_EVENTLOG_EVENTS`` events after ``since_seq`` (``next_seq`` is
    the last ``seq`` shown when more may follow, else None). ``usage`` is the
    fold of the whole readable log (``runtime.usage``, cached), shown beside the
    meter, never added to it. A damaged line stops the reading there: ``skipped``
    is 1 and ``corrupt_after_seq`` the last readable ``seq`` (else 0 and None).
    Raises ``FileNotFoundError`` (404) for an id the runtime refuses or with no
    regular log file; the id is checked by ``check_conversation_id`` before the
    one ``lstat`` of its own path, so no path is built from unchecked text.
    """
    try:
        check_conversation_id(conversation_id)
    except EventPayloadError:
        raise FileNotFoundError(conversation_id) from None
    path = Path(log.root) / LOG_DIR / f"{conversation_id}{LOG_SUFFIX}"
    try:
        seen = os.lstat(path)
    except OSError:
        raise FileNotFoundError(conversation_id) from None
    if not stat.S_ISREG(seen.st_mode):
        raise FileNotFoundError(conversation_id)
    limit = max(1, min(limit, MAX_EVENTLOG_EVENTS))
    events: list[dict[str, Any]] = []
    try:
        for event in log.query(conversation_id, since_seq=since_seq, limit=limit):
            events.append(_shown(event))
    except LogCorruptError:
        pass  # the page ends at the damage; _usage says where
    usage, corrupt_after = _usage(log, conversation_id, seen)
    return {
        "conversation_id": conversation_id,
        "events": events,
        "next_seq": events[-1]["seq"] if len(events) == limit else None,
        "usage": usage,
        "skipped": 0 if corrupt_after is None else 1,
        "corrupt_after_seq": corrupt_after,
    }
