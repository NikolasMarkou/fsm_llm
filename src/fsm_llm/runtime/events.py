"""The event vocabulary, the error family and the safe encoder of the event log.

One event is one line: compact JSON, ASCII only (``ensure_ascii=True``), keys sorted, ending in
one ``b"\\n"``. Newline is the only commit marker, so the encoding never lets a newline or a
Unicode line separator (U+2028, U+0085) appear inside a line, and readers split on the byte
``b"\\n"`` and never on ``str.splitlines``. Nesting depth is bounded by ``MAX_PAYLOAD_DEPTH``
on write and on read, iteratively: ``json.loads`` and ``json.dumps`` recurse (or, with
``indent``, allocate gigabytes) on a hostile value, and the limit differs per Python version.

Imports ``core`` (``FSMError``), the kernel and ``runtime.constants`` only.
"""

from __future__ import annotations

import datetime
import decimal
import json
import math
import uuid
from enum import Enum
from typing import Any, cast

from fsm_llm.core.definitions import FSMError
from fsm_llm.runtime.constants import (
    ENVELOPE_KEYS,
    LOG_FORMAT_VERSION,
    MAX_EVENT_BYTES,
    MAX_INT_BITS,
    MAX_PAYLOAD_DEPTH,
)


class EventType(str, Enum):
    """The closed vocabulary of event types.

    The values are the roadmap's wire strings verbatim (they contain ``/``, which the
    repository's snake_case habit would not; D-004 of plan 21e6f00d records why), the
    members are UPPER_SNAKE. ``PRODUCED_TYPES`` names the ones something writes today.
    """

    TURN_START = "turn/start"
    LLM_REQUEST = "llm/request"
    LLM_RESPONSE = "llm/response"
    PROVIDER_ERROR = "provider/error"
    EXTRACTION = "extraction"
    EXTRACTION_MISS = "extraction/miss"
    INTENT_OUT_OF_SCOPE = "intent/out_of_scope"
    TRANSITION = "transition"
    DECISION = "decision"
    HANDLER = "handler"
    TOOL_CALL = "tool/call"
    TOOL_RESULT = "tool/result"
    APPROVAL_REQUEST = "approval/request"
    APPROVAL_OUTCOME = "approval/outcome"
    EXECUTOR_START = "executor/start"
    EXECUTOR_RESULT = "executor/result"
    STACK_PUSH = "stack/push"
    STACK_POP = "stack/pop"
    CONTEXT_RECORD = "context/record"
    COMPACTION_EPOCH = "compaction/epoch"
    TURN_END = "turn/end"
    # The run vocabulary of E2.1 (plan ace7a8c8 D-059). ``run/start`` and ``run/end`` are written
    # by ``settle`` for a delegated ``call`` (plan baa37f40 K7.4, D-052); ``executable/resolve`` is
    # still declared with no producer. Unknown types are refused on write and tolerated on read,
    # so adding members changes no old log and needs no ``LOG_FORMAT_VERSION`` bump. A ``run/*``
    # event is written at the TOP frame of its owner. ``run/start`` (an fsynced checkpoint, before
    # the thread exists) carries ``handle``, ``target``, the template and instance identity, the
    # ``source`` of the request (a ``MessageSource`` value), the ``parent`` frame id and the
    # inputs by reference (``inputs_ref``, level ``values`` only); ``run/end`` carries ``handle``,
    # ``target``, ``status``, ``reason`` and the identities of the final record, written by the
    # engine thread only (never by a worker).
    EXECUTABLE_RESOLVE = "executable/resolve"
    RUN_START = "run/start"
    RUN_END = "run/end"

    @classmethod
    def from_string(cls, value: EventType | str) -> EventType:
        """Return the member for a member or an exact wire string.

        Raises ``ValueError`` naming ``event type`` and the valid values for anything else
        (a member name such as ``"llm_request"``, another string, ``None``, a number).
        """
        try:
            return cls(value)
        except ValueError:
            valid_values = [member.value for member in cls]
            raise ValueError(
                f"Invalid event type: {value!r}. Must be one of {valid_values}"
            ) from None


class TurnOutcome(str, Enum):
    """How a turn ended, as the ``outcome`` of its ``turn/end`` event."""

    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    ABANDONED = "abandoned"
    INTERRUPTED = "interrupted"

    @classmethod
    def from_string(cls, value: TurnOutcome | str) -> TurnOutcome:
        """Return the member for a member or an exact value string (``ValueError`` else)."""
        try:
            return cls(value)
        except ValueError:
            valid_values = [member.value for member in cls]
            raise ValueError(
                f"Invalid turn outcome: {value!r}. Must be one of {valid_values}"
            ) from None


#: Every type the format declares (the 21 of roadmap row 1.1 and the three run types).
DECLARED_TYPES: frozenset[EventType] = frozenset(EventType)

#: The types product code writes today (15: no producer of ``tool/call`` exists yet, and ``tool/result``
#: is written only as the ``unknown`` record of ``API.resume``). The set test builds the table of
#: scenarios from product paths only, never from ``EventLog.tool_call`` or ``EventLog.append``.
PRODUCED_TYPES: frozenset[EventType] = frozenset(
    {
        EventType.TURN_START,
        EventType.TURN_END,
        EventType.LLM_REQUEST,
        EventType.LLM_RESPONSE,
        EventType.PROVIDER_ERROR,
        EventType.TOOL_RESULT,
        EventType.CONTEXT_RECORD,
        EventType.STACK_PUSH,
        EventType.STACK_POP,
        EventType.EXECUTOR_START,
        EventType.EXECUTOR_RESULT,
        EventType.TRANSITION,
        EventType.DECISION,
        EventType.RUN_START,
        EventType.RUN_END,
    }
)


class EventLogError(FSMError):
    """Root of the event log errors."""


class LogCorruptError(EventLogError):
    """A log line or blob cannot be trusted: over the size cap, not JSON, too deep, a missing
    envelope key, a gapped ``seq`` or a blob whose bytes do not hash to its name."""


class LogLockedError(EventLogError):
    """Another writer holds the log; a lock is refused, never waited for."""


class LogWriteError(EventLogError):
    """An append failed (``OSError``). The writer is poisoned: later appends refuse, so a
    checkpoint that did not reach the disk never lets the model call it protects run."""


class LogPrivacyError(EventLogError):
    """A log written at a privacy level without values (``shape``) was folded, forked, resumed
    or restored: it holds no state to give back, and saying "unknown id" would lie."""


class EventPayloadError(EventLogError):
    """A payload cannot be written: too deep, too large, not JSON, or a bad envelope."""


def check_depth(value: Any, limit: int = MAX_PAYLOAD_DEPTH) -> bool:
    """Return True when no container in *value* nests deeper than *limit*.

    A dict, list or tuple is a container; the outermost is depth 1 and a scalar adds none.
    Iterative (an explicit stack, no recursion), so a 100,000-deep value cannot raise
    ``RecursionError``. A reference cycle nests without end and returns False. A container
    reached again at a depth no deeper than the deepest it was walked at is not walked again
    (a deeper meeting is, since its subtree now ends deeper), so a value with shared children
    costs its distinct nodes, not its paths. Never raises.
    """
    stack: list[tuple[Any, int]] = [(value, 0)]
    seen: dict[int, int] = {}
    while stack:
        node, depth = stack.pop()
        if isinstance(node, dict):
            children: Any = node.values()
        elif isinstance(node, (list, tuple)):
            children = node
        else:
            continue
        depth += 1
        if depth > limit:
            return False
        previous = seen.get(id(node))
        if previous is not None and previous >= depth:
            continue
        seen[id(node)] = depth
        stack.extend((child, depth) for child in children)
    return True


def _marker(value: object) -> str:
    """The typed marker of a leaf JSON cannot hold: its type name, never its ``repr``."""
    return f"<{type(value).__name__}>"


def _int_marker(value: int) -> str | None:
    """The marker of an int over ``MAX_INT_BITS`` (``json.dumps`` refuses its digits), else None."""
    bits = value.bit_length()
    return f"<int:{bits} bits>" if bits > MAX_INT_BITS else None


def _leaf(value: Any) -> Any:
    """Return the JSON form of a leaf, or the typed marker when it has none."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return _int_marker(value) or value
    if isinstance(value, float):
        return value if math.isfinite(value) else f"<float:{value!r}>"
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    if isinstance(value, (decimal.Decimal, uuid.UUID)):
        return str(value)
    if isinstance(value, (bytes, bytearray)):
        return f"<bytes:{len(value)}>"
    return _marker(value)


def _key(key: Any) -> str:
    """A JSON object key: a string as is, a number or bool as its text, else a typed marker."""
    if isinstance(key, str):
        return key
    if isinstance(key, int) and not isinstance(key, bool):
        return _int_marker(key) or str(key)
    if key is None or isinstance(key, (bool, float)):
        return str(key)
    return _marker(key)


def to_jsonable(value: Any, *, max_depth: int = MAX_PAYLOAD_DEPTH - 1) -> Any:
    """Return a copy of *value* that ``encode_event`` accepts, or the value itself if a leaf.

    Containers (dict, list, tuple) are copied, a tuple becoming a list. A leaf JSON cannot
    hold becomes a typed marker string such as ``"<set>"`` (``datetime`` and ``date`` become
    ISO text, ``Decimal`` and ``UUID`` their text, bytes ``"<bytes:N>"``, a non-finite float
    ``"<float:nan>"``, an int over ``MAX_INT_BITS`` bits ``"<int:N bits>"``: its digits would make
    ``json.dumps`` raise). A container nested deeper than *max_depth* and a reference cycle become
    ``"<truncated:depth>"`` and ``"<cycle>"``. The default depth leaves room for the envelope
    (depth 1), so a result stored directly as an event field always encodes; a caller that
    nests it deeper passes a smaller *max_depth*. Iterative, so depth cannot raise
    ``RecursionError``. Deliberately not ``default=str``: that hides a ``repr`` of an object
    (a secret in an attribute) in the log; a marker names the type and nothing else.
    """
    holder: list[Any] = [None]
    active: set[int] = set()
    # (source, destination container, slot in it, depth); a source of None with depth -1 is
    # the "leave" marker of the container whose id sits in the slot.
    stack: list[tuple[Any, Any, Any, int]] = [(value, holder, 0, 1)]
    while stack:
        source, dest, slot, depth = stack.pop()
        if depth < 0:
            active.discard(slot)
            continue
        if not isinstance(source, (dict, list, tuple)):
            dest[slot] = _leaf(source)
            continue
        if depth > max_depth:
            dest[slot] = "<truncated:depth>"
            continue
        if id(source) in active:
            dest[slot] = "<cycle>"
            continue
        active.add(id(source))
        stack.append((None, None, id(source), -1))
        if isinstance(source, dict):
            out_map: dict[str, Any] = {}
            dest[slot] = out_map
            # pushed in reverse so the copy keeps the source's key order when popped
            for key, child in reversed(list(source.items())):
                stack.append((child, out_map, _key(key), depth + 1))
        else:
            out_list: list[Any] = [None] * len(source)
            dest[slot] = out_list
            for index, child in enumerate(source):
                stack.append((child, out_list, index, depth + 1))
    return holder[0]


def _check_envelope(event: object, error: type[EventLogError], what: str) -> None:
    """Raise *error* unless *event* is a dict with every ``ENVELOPE_KEYS`` key."""
    if not isinstance(event, dict):
        raise error(f"{what} is not a JSON object")
    missing = [key for key in ENVELOPE_KEYS if key not in event]
    if missing:
        raise error(f"{what} lacks envelope keys {missing}")


def canonical_json(value: Any, *, what: str = "payload") -> bytes:
    """Return the canonical ASCII JSON bytes of *value*: compact, keys sorted, no newline.

    The one place the log's ``json.dumps`` arguments live; ``encode_event`` (a line) and the
    blob store (``BlobStore.put_json``) both call it, so an event and a blob never encode
    differently. Raises ``EventPayloadError`` (naming *what*) when *value* nests deeper than
    ``MAX_PAYLOAD_DEPTH`` or holds a cycle (checked before ``json.dumps`` runs), or holds a
    value JSON cannot (pass it through ``to_jsonable`` first; NaN and infinity are refused).
    No size cap here: the caller applies its own.
    """
    if not check_depth(value, MAX_PAYLOAD_DEPTH):
        raise EventPayloadError(
            f"{what} nests deeper than {MAX_PAYLOAD_DEPTH} levels or contains a cycle"
        )
    try:
        text = json.dumps(
            value,
            separators=(",", ":"),
            ensure_ascii=True,
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise EventPayloadError(f"{what} is not JSON serializable: {exc}") from None
    return text.encode("ascii")


def encode_event(envelope: dict[str, Any]) -> bytes:
    """Return the log line of *envelope*: compact ASCII JSON with sorted keys and one ``b"\\n"``.

    Raises ``EventPayloadError`` when the envelope lacks a key of ``ENVELOPE_KEYS``, nests
    deeper than ``MAX_PAYLOAD_DEPTH``, holds a value JSON cannot (pass a payload through
    ``to_jsonable`` first), or encodes to more than ``MAX_EVENT_BYTES`` bytes including the
    newline (put a large payload in a blob). The depth is checked before ``json.dumps`` runs.
    """
    _check_envelope(envelope, EventPayloadError, "event")
    data = canonical_json(envelope, what="event") + b"\n"
    if len(data) > MAX_EVENT_BYTES:
        raise EventPayloadError(
            f"event is {len(data)} bytes, over the {MAX_EVENT_BYTES} byte cap; "
            "store a large payload as a blob"
        )
    return data


def decode_event(line: bytes) -> dict[str, Any]:
    """Return the event of one log *line* (a trailing ``b"\\n"`` is allowed).

    Raises ``LogCorruptError`` for a line over ``MAX_EVENT_BYTES`` (checked before any
    parsing), not ASCII, not a JSON object, so deep that ``json.loads`` gives up
    (``RecursionError`` and ``MemoryError`` are corruption, not crashes), nested deeper than
    ``MAX_PAYLOAD_DEPTH`` once parsed, lacking an envelope key, or of a newer format version.
    """
    if len(line) > MAX_EVENT_BYTES:
        raise LogCorruptError(
            f"log line is {len(line)} bytes, over the {MAX_EVENT_BYTES} byte cap"
        )
    try:
        event = json.loads(line.removesuffix(b"\n").decode("ascii"))
    except (ValueError, RecursionError, MemoryError) as exc:
        raise LogCorruptError(
            f"log line is not valid JSON ({type(exc).__name__})"
        ) from None
    _check_envelope(event, LogCorruptError, "log line")
    if not check_depth(event, MAX_PAYLOAD_DEPTH):
        raise LogCorruptError(f"log line nests deeper than {MAX_PAYLOAD_DEPTH} levels")
    version = event["v"]
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or not 1 <= version <= LOG_FORMAT_VERSION
    ):
        raise LogCorruptError(f"log line has unsupported format version {version!r}")
    return cast("dict[str, Any]", event)  # _check_envelope proved it is a dict
