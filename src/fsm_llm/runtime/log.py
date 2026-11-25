"""The event log: one append-only JSONL file per conversation tree, plus the shared blobs.

``EventLog(root)`` owns the layout ``<root>/logs/<root conversation id>.jsonl`` and
``<root>/blobs/``. It keeps one ``JsonlWriter`` per conversation id (opened on first use, so
nothing is created until something is written) and adds what the writer does not know: the
envelope (``ts``, ``conversation_id``, ``frame``, ``type``), the checkpoint rule, the
write-time secret filter, reads and queries, blobs, and the checkpoint API of a side-effecting
tool call (``tool_call``).

Privacy (D-003 of plan 21e6f00d): structured state and context payloads go through
``fsm_llm.runtime.filtering.filter_context`` (``core.redaction.redact_context``, then
internal-prefixed keys dropped except a caller's allow-list, then ``to_jsonable``); free text
(a prompt, a reply, a user message) is stored as the model saw it, because the repository
already keeps secret-looking context entries out of prompts. Keep the log directory as private
as a session directory.
``EventLog(root, privacy=, redact=)`` adds the ``shape`` level and the user's redaction hook
(``runtime.privacy``, plan 5a6a1f80: read its limits, a free-text secret reaches disk at level
``values`` unless the hook handles it).

The leaf lock ``_guard`` protects the table of writers. It is never held across an append or
a call into another component, so the log adds no edge to the lock order of ``FSMManager``;
it IS held while a new writer is built (taking the file lock, opening the file and scanning
it for a torn tail), so the first append of a large log makes the first append of another
conversation on the same ``EventLog`` wait for that scan. Imports ``core``, the kernel and
``runtime`` modules only. Windows behavior is that of ``stores.jsonl`` and ``blobs``
(unverified until the Windows run).
"""

from __future__ import annotations

import datetime
import json
import os
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar

from fsm_llm.core.redaction import redact_context
from fsm_llm.logging import logger
from fsm_llm.runtime.blobs import BlobStore, blob_ref
from fsm_llm.runtime.checkpoint import is_checkpoint, unresolved_tool_calls
from fsm_llm.runtime.constants import (
    ENVELOPE_KEYS,
    LEVEL_KEY,
    LOG_DIR,
    MAX_INLINE_BYTES,
    MAX_PAYLOAD_DEPTH,
)
from fsm_llm.runtime.events import (
    EventPayloadError,
    EventType,
    LogCorruptError,
    LogLockedError,
    LogWriteError,
    canonical_json,
    check_depth,
    to_jsonable,
)
from fsm_llm.runtime.lanes import Lanes
from fsm_llm.runtime.privacy import (
    PrivacyLevel,
    RedactionHook,
    RedactionKind,
    run_redaction,
    shape_only,
)
from fsm_llm.runtime.retention import PruneReport, prune_root
from fsm_llm.runtime.stores.jsonl import (
    LOG_SUFFIX,
    JsonlWriter,
    check_conversation_id,
    read_events,
)

_T = TypeVar("_T")

#: Envelope keys the caller never supplies as a field: the log assigns them (the level marker
#: too: a producer cannot say its own event is a ``values`` one).
_RESERVED_FIELDS = frozenset(ENVELOPE_KEYS) | {LEVEL_KEY}


@contextmanager
def warn_on_failure(kind: EventType | str) -> Iterator[None]:
    """Run a write that comes AFTER the thing it records; an ``Exception`` is a WARNING.

    Interface contract (callers: the turn scope, the stack events, ``end_turn``): the body is
    one or more appends; an ``Exception`` out of it is swallowed and logged as ``event log:
    <kind> not written (<ExceptionType>)`` (the type name only, never the message), because the
    outcome it describes already happened and must not be hidden by the log. A
    ``BaseException`` that is not an ``Exception`` (an interrupt, ``SystemExit``) propagates.
    """
    try:
        yield
    except Exception as exc:
        name = EventType.from_string(kind).value
        logger.warning(f"event log: {name} not written ({type(exc).__name__})")


def _timestamp() -> str:
    """The informational ``ts``: ISO 8601 UTC with milliseconds and ``Z``, from ``time.time()``.

    Read through the module-level ``time`` so a test can bind a stand-in clock. Never the
    order of anything (``seq`` is), so a clock that stands still, runs backwards or returns
    nonsense changes nothing but this text; an unreadable value becomes the epoch.
    """
    try:
        moment = datetime.datetime.fromtimestamp(time.time(), tz=datetime.UTC)
    except (OverflowError, OSError, ValueError):
        moment = datetime.datetime.fromtimestamp(0, tz=datetime.UTC)
    return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")


class EventLog:
    """The durable event log of one directory (see the module docstring).

    Interface contract (callers: ``runtime.scope``, ``engine`` and its session store in later
    steps; agents and handlers through ``read``, ``query`` and ``tool_call``):
        - ``EventLog(root)``: *root* holds ``logs/`` and ``blobs/``. No I/O until first use.
          Usable as a context manager; ``close()`` releases every writer and is idempotent.
        - ``append(conversation_id, type, *, frame=0, checkpoint=None, **fields) -> int``:
          appends one event and returns its ``seq`` (1-based, contiguous per id). *type* is an
          ``EventType`` or its wire string (``ValueError`` otherwise). *conversation_id* is the
          ROOT id of the tree (``check_conversation_id``); *frame* the stack index (root 0).
          *fields* are the per-type payload; they pass through ``to_jsonable`` (markers for
          non-JSON leaves, depth cut) but NOT through the secret filter: pass structured
          context through ``filter_context`` first. ``v``, ``seq``, ``ts``, ``conversation_id``,
          ``frame`` and ``type`` are never fields (``EventPayloadError``). *checkpoint* ``None``
          takes ``is_checkpoint``; ``True`` or ``False`` forces it. A checkpoint is fsynced
          before ``append`` returns. Failure: ``LogWriteError`` (the disk, a closed log, a
          poisoned writer: nothing after it is written, so the model call it guards must not
          run), ``LogLockedError`` (another writer holds this id), ``EventPayloadError``,
          ``LogCorruptError`` (a damaged existing log).
        - ``append_batch(conversation_id, events, *, frame=0, checkpoint=None) -> list[int]``:
          each of *events* is a mapping with ``type``, an optional ``frame`` and fields; all go
          out in ONE write with contiguous ``seq`` and at most one fsync (when any is a
          checkpoint, or *checkpoint* is True). NOT crash atomic: after a kill a whole-line
          prefix of the batch may exist.
        - ``read(conversation_id, *, since_seq=0) -> Iterator[dict]`` and ``query(...)``: see
          their docstrings. An id with no log reads as empty. ``LogCorruptError`` for a damaged
          log; never repairs, never writes.
        - ``put_blob(bytes) -> ref``, ``put_json(value, *, kind=None) -> ref``,
          ``get_blob(ref) -> bytes``, ``get_json(ref)``: ``BlobStore`` through the log root (blobs
          are shared by every tree). *kind* (a ``RedactionKind``) is required while a hook is set.
        - ``EventLog(root, *, privacy=PrivacyLevel.VALUES, redact=None)``: ``privacy`` ``"shape"``
          writes a per-type field allow-list (``privacy.SHAPE_FIELDS``), no blob and no ``*_ref``,
          and marks every event ``lvl: "shape"``: ``put_json`` and ``put_blob`` refuse
          (``EventPayloadError``), the producers read ``stores_values`` and build no blob, and a
          fold of such a log raises ``LogPrivacyError``. *redact* is the hook ``(kind, value) ->
          value`` of ``privacy.run_redaction``: it runs inside ``put_json`` and on the inline
          arguments of a tool call, after the built-in filters; ``put_blob`` is refused while it
          is set; any failure is ``EventPayloadError`` and the checkpoint it guards fails.
          ``digest_json(value) -> ref`` is the reference ``put_json`` would give, storing nothing.
        - ``tool_begin(...)`` and ``tool_end(...)``: the two emitters of ``tool/call`` and
          ``tool/result`` (what ``runtime.scope`` writes for ``ToolRegistry.execute``);
          ``tool_call(conversation_id, *, name, call_id, arguments, ...)``: the same pair as one
          context manager, see its docstring. ``unknown_tool_calls(conversation_id)`` reports
          the unresolved ones.
        - ``conversation_ids() -> list[str]``, ``release(conversation_id)``.
        - Long-running hosts: every conversation id that is appended to keeps one writer open,
          two file descriptors and the file's advisory lock, until ``release(conversation_id)``
          (``API`` does it when a root conversation ends) or ``close()``; nothing caps the
          number of writers. A host that starts many conversations must end them
          (``API.end_conversation``) or close the API or the log. A writer or log that is
          dropped without either is closed by the finalizer of ``JsonlWriter``, which frees
          the descriptors and the lock when the object is collected.
        - Recovery: a failed write POISONS the writer of that id, and every later append of
          that tree raises ``LogWriteError`` until ``release(conversation_id)`` (the next
          append reopens the file, repairs a torn tail and continues the ``seq``) or
          ``close()`` of the log. A late append to a released id (a straggler thread of a
          timed-out handler) reopens a writer; it is closed again by the next ``release`` or
          ``close``.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        privacy: PrivacyLevel | str = PrivacyLevel.VALUES,
        redact: RedactionHook | None = None,
    ) -> None:
        self._privacy = PrivacyLevel.from_string(privacy)
        if redact is not None and not callable(redact):
            raise TypeError(
                f"redact is a callable or None, got {type(redact).__name__}"
            )
        self._redact = redact
        self.root = Path(root)
        self._logs = self.root / LOG_DIR
        self._blobs = BlobStore(self.root)
        self._guard = threading.Lock()  # leaf lock: the table of writers only
        self._writers: dict[str, JsonlWriter] = {}
        self._closed = False
        self._lanes = Lanes()  # no thread starts until the first lane_call

    @property
    def privacy(self) -> PrivacyLevel:
        """The level this log writes at."""
        return self._privacy

    @property
    def stores_values(self) -> bool:
        """True at level ``values``; a producer builds no blob when this is False."""
        return self._privacy is PrivacyLevel.VALUES

    def _writer(self, conversation_id: str) -> JsonlWriter:
        check_conversation_id(conversation_id)
        with self._guard:
            if self._closed:
                raise LogWriteError("the event log is closed")
            writer = self._writers.get(conversation_id)
            if writer is None:
                writer = JsonlWriter(self._logs / f"{conversation_id}{LOG_SUFFIX}")
                self._writers[conversation_id] = writer
            return writer

    async def lane_call(
        self, conversation_id: str, fn: Callable[..., _T], /, *args: Any
    ) -> _T:
        """Run one boundary write on the lane thread of the tree *conversation_id* (the ROOT id).

        ``fn(*args)`` is today's sync function, unchanged; see ``runtime.lanes.Lanes.lane_call``
        (inline in sync mode, FIFO per tree, a cancelled await leaves the job running).
        """
        return await self._lanes.lane_call(conversation_id, fn, *args)

    def has_writer(self, conversation_id: str) -> bool:
        """True while this log holds the writer (and so the lock) of *conversation_id*."""
        with self._guard:
            return conversation_id in self._writers

    def open_existing(self, conversation_id: str) -> int:
        """Take the writer of an EXISTING log and return its last ``seq`` (``resume`` reads after).

        Interface contract (caller: ``API.resume``, plan 5a6a1f80): the lock is taken here, BEFORE
        the caller reads anything, so what it then reads is what it will continue. A torn tail is
        cut and the ``seq`` continues, by the one code of ``JsonlWriter``. The writer stays until
        ``release`` or ``close``. ``FileNotFoundError`` for an id with no log, and then no log, no
        ``.lock`` and no directory was created. ``LogLockedError`` when another process or
        ``EventLog`` holds the lock, and when THIS log already holds a writer for the id (two
        ``API`` objects share one ``EventLog`` and so one writer: the lock cannot tell them
        apart). ``LogCorruptError`` for a damaged log (the lock is released again),
        ``EventPayloadError`` for a bad id, ``LogWriteError`` for a closed log or an OS failure.
        """
        check_conversation_id(conversation_id)
        # DECISION plan-2026-11-18T071726-5a6a1f80/D-007
        # The writer lock is taken HERE, before the caller reads a byte of the log: what it then
        # folds is exactly what it will continue. Do NOT fold first and lock after (another
        # resumer would fold the same log and both would continue it: a TOCTOU on a file with
        # one writer), do NOT take the lock in the caller after a check of "no live writer" (the
        # check and the lock are one step or none), and do NOT create the file, the `.lock` or
        # the directory for an unknown id (it must raise and leave nothing). A log that is a
        # link is refused inside `JsonlWriter`, also before the lock. See decisions.md D-007.
        with self._guard:
            if self._closed:
                raise LogWriteError("the event log is closed")
            if conversation_id in self._writers:
                raise LogLockedError(
                    f"log {conversation_id}{LOG_SUFFIX} already has a live writer in this EventLog"
                )
            writer = JsonlWriter(
                self._logs / f"{conversation_id}{LOG_SUFFIX}", create=False
            )
            self._writers[conversation_id] = writer
            return writer.last_seq

    def _envelope(
        self, conversation_id: str, kind: Any, frame: Any, fields: Mapping[str, Any]
    ) -> dict[str, Any]:
        event_type = EventType.from_string(kind)
        if type(frame) is not int or frame < 0:
            raise EventPayloadError(f"frame is an int of 0 or more, got {frame!r:.40}")
        reserved = sorted(_RESERVED_FIELDS.intersection(fields))
        if reserved:
            raise EventPayloadError(f"the log assigns {reserved}; they are not fields")
        # DECISION plan-2026-11-18T071726-5a6a1f80/D-016
        # EVERY event of every append path passes here, so the privacy level is applied HERE:
        # at `shape` the payload is cut to the per-type allow-list (`privacy.SHAPE_FIELDS`) and
        # the event is marked `lvl: "shape"`. Do NOT filter in the producers instead (a new
        # field would leak until someone remembers), do NOT drop only the large fields or keep
        # a `*_ref` (a short user message hashes to a recoverable ref), and do NOT infer the
        # level from "no state_ref" (it cannot tell a values log a shape writer continued).
        # The marker is what makes a fold refuse a shape log loudly instead of returning None
        # ("unknown id"). The default stays `values` (a shape log cannot restore anything).
        # See decisions.md D-016.
        payload = dict(fields) if self.stores_values else shape_only(event_type, fields)
        envelope = {
            **to_jsonable(payload),
            "ts": _timestamp(),
            "conversation_id": conversation_id,
            "frame": frame,
            "type": event_type.value,
        }
        if not self.stores_values:
            envelope[LEVEL_KEY] = self._privacy.value
        return envelope

    def append(
        self,
        conversation_id: str,
        type: EventType | str,
        *,
        frame: int = 0,
        checkpoint: bool | None = None,
        **fields: Any,
    ) -> int:
        """Append one event and return its ``seq`` (see the class contract)."""
        event = self._envelope(conversation_id, type, frame, fields)
        writer = self._writer(conversation_id)  # checks the id
        if checkpoint is None:
            checkpoint = is_checkpoint(event["type"], event)
        return writer.append(event, checkpoint=checkpoint)

    def append_batch(
        self,
        conversation_id: str,
        events: Sequence[Mapping[str, Any]],
        *,
        frame: int = 0,
        checkpoint: bool | None = None,
    ) -> list[int]:
        """Append *events* in one write and return their ``seq`` numbers (see the class contract)."""
        # nothing is created until something is written; a bad id still raises
        if not events:
            check_conversation_id(conversation_id)
            return []
        envelopes = []
        for item in events:
            if "type" not in item:
                raise EventPayloadError("a batch event needs a type")
            fields = {k: v for k, v in item.items() if k not in ("type", "frame")}
            envelopes.append(
                self._envelope(
                    conversation_id, item["type"], item.get("frame", frame), fields
                )
            )
        if checkpoint is None:
            checkpoint = any(is_checkpoint(e["type"], e) for e in envelopes)
        return self._writer(conversation_id).append_many(
            envelopes, checkpoint=checkpoint
        )

    def read(
        self, conversation_id: str, *, since_seq: int = 0
    ) -> Iterator[dict[str, Any]]:
        """Yield the committed events of one tree with ``seq > since_seq``, in ``seq`` order.

        Missing log: yields nothing. A newline-less tail is ignored, a damaged log raises
        ``LogCorruptError`` (see ``read_events``). Opens a separate descriptor, so it is safe
        beside an open writer of the same process.
        """
        check_conversation_id(conversation_id)
        try:
            for event in read_events(self._logs / f"{conversation_id}{LOG_SUFFIX}"):
                if event["seq"] > since_seq:
                    yield event
        except FileNotFoundError:
            return

    def query(
        self,
        conversation_id: str,
        *,
        types: Iterable[EventType | str] | None = None,
        frame: int | None = None,
        since_seq: int = 0,
        until_seq: int | None = None,
        limit: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield the events of one tree that match every given filter, in ``seq`` order.

        *types* (members or wire strings, ``ValueError`` for an unknown one), *frame*, the
        range ``since_seq < seq <= until_seq`` and at most *limit* events. Read-only and
        lazy: the scan stops at *until_seq* or *limit*. ``None`` means no filter.
        """
        wanted = (
            None
            if types is None
            else frozenset(EventType.from_string(t).value for t in types)
        )
        count = 0
        if limit is not None and limit <= 0:
            return
        for event in self.read(conversation_id, since_seq=since_seq):
            if until_seq is not None and event["seq"] > until_seq:
                return
            if wanted is not None and event["type"] not in wanted:
                continue
            if frame is not None and event["frame"] != frame:
                continue
            count += 1
            yield event
            if limit is not None and count >= limit:
                return

    def conversation_ids(self) -> list[str]:
        """Return the ids that have a log file, sorted; foreign file names are ignored."""
        try:
            names = os.listdir(self._logs)
        except FileNotFoundError:
            return []
        stems = (n[: -len(LOG_SUFFIX)] for n in names if n.endswith(LOG_SUFFIX))
        return sorted(s for s in stems if _is_valid_id(s))

    def _refuse_values(self, what: str) -> None:
        if not self.stores_values:
            raise EventPayloadError(f"this log is at level shape and stores no {what}")

    def _redacted(self, kind: RedactionKind | str | None, value: Any) -> Any:
        """*value* through the hook when one is set (``kind`` is then required), else as it is."""
        if self._redact is None:
            return value
        if kind is None:
            raise EventPayloadError(
                "a redaction hook is set: name the kind of the value"
            )
        return run_redaction(self._redact, kind, value)

    def put_blob(self, data: bytes) -> str:
        """Store *data* in the blob store and return its ``sha256:`` reference.

        Refused (``EventPayloadError``) at level ``shape`` and while a redaction hook is set:
        bytes cannot be redacted, and a producer must reach blobs through ``put_json``.
        """
        self._refuse_values("blob")
        if self._redact is not None:
            raise EventPayloadError(
                "a redaction hook is set: store JSON, not raw bytes"
            )
        return self._blobs.put(data)

    def put_json(self, value: Any, *, kind: RedactionKind | str | None = None) -> str:
        """Store the canonical JSON of *value* (``EventPayloadError`` when too deep or not JSON).

        With a hook the stored value is what the hook returned for *kind* (see the class
        contract); at level ``shape`` it refuses.
        """
        self._refuse_values("blob")
        return self._blobs.put_json(self._redacted(kind, value))

    def digest_json(self, value: Any) -> str:
        """Return the reference ``put_json`` gives *value* with no hook, storing nothing.

        What a ``shape`` log uses to tell a rolled-back turn from an abandoned one (equal
        digests): the digest is compared in memory and never written.
        """
        return blob_ref(canonical_json(value, what="blob payload"))

    def get_blob(self, ref: str) -> bytes:
        """Return the bytes of *ref*, re-hashed (``LogCorruptError`` when missing or altered)."""
        return self._blobs.get(ref)

    def get_json(self, ref: str) -> Any:
        """Return the JSON value of blob *ref*; ``LogCorruptError`` if it is not JSON or too deep.

        ``RecursionError`` and ``MemoryError`` of the parser count as corruption, and the depth
        is checked (``MAX_PAYLOAD_DEPTH``) after parsing, as for a log line.
        """
        data = self._blobs.get(ref)
        try:
            value = json.loads(data.decode("utf-8"))
        except (ValueError, RecursionError, MemoryError) as exc:
            raise LogCorruptError(
                f"blob {ref[:19]} is not valid JSON ({type(exc).__name__})"
            ) from None
        if not check_depth(value, MAX_PAYLOAD_DEPTH):
            raise LogCorruptError(
                f"blob {ref[:19]} nests deeper than {MAX_PAYLOAD_DEPTH} levels"
            )
        return value

    def unknown_tool_calls(self, conversation_id: str) -> list[dict[str, Any]]:
        """Return the tool calls of one tree that have no result, as ``unknown`` (never retried)."""
        return unresolved_tool_calls(self.read(conversation_id))

    def tool_begin(
        self,
        conversation_id: str,
        *,
        name: str,
        call_id: str,
        arguments: Mapping[str, Any],
        side_effecting: bool = True,
        gated: bool | None = None,
        frame: int = 0,
    ) -> None:
        """Append ``tool/call`` BEFORE a tool body runs (the one emitter of that event).

        Fields: ``name``, ``call_id``, ``side_effecting``, ``gated`` when given, and the
        *arguments* after ``redact_context``: inline when small, else ``arguments_ref`` to a
        blob. A side-effecting call is fsynced first (``is_checkpoint``), so if
        ``LogWriteError`` is raised the body must not run. With a redaction hook the arguments
        pass through it (kind ``arguments``) after ``redact_context``, and a failing hook raises
        before the event. At level ``shape`` the event carries ``argument_keys`` (the sorted
        names) and no value. *side_effecting* defaults to True: a tool is side-effecting unless
        it says it only reads.
        """
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(call_id, str)
            or not call_id
        ):
            raise EventPayloadError("a tool call needs a non-empty name and call_id")
        self._append_call(
            EventType.TOOL_CALL,
            conversation_id,
            frame,
            arguments,
            gated,
            name=name,
            call_id=call_id,
            side_effecting=bool(side_effecting),
        )

    def _append_call(
        self,
        kind: EventType,
        conversation_id: str,
        frame: int,
        arguments: Mapping[str, Any],
        gated: bool | None = None,
        **fields: Any,
    ) -> int:
        """Append *kind* with *fields* and the *arguments* as ``arguments``, ``arguments_ref`` or (at
        shape) ``argument_keys``, after ``redact_context`` and the user hook (kind ``arguments``)."""
        if not isinstance(arguments, Mapping):
            raise EventPayloadError("tool call arguments are a mapping")
        safe = to_jsonable(redact_context(dict(arguments)))
        where: dict[str, Any]
        if not self.stores_values:
            where = {"argument_keys": sorted(str(name) for name in arguments)}
        else:
            safe = self._redacted(RedactionKind.ARGUMENTS, safe)
            # inline only what the event field can hold whole: the envelope is one level and
            # `to_jsonable` cuts at MAX_PAYLOAD_DEPTH - 1, so a hook output of depth 31 or 32
            # (which a blob keeps) would be cut inline
            if (
                check_depth(safe, MAX_PAYLOAD_DEPTH - 2)
                and len(canonical_json(safe, what="tool arguments")) <= MAX_INLINE_BYTES
            ):
                where = {"arguments": safe}
            else:  # the hook already ran: store the blob as it is
                where = {"arguments_ref": self._blobs.put_json(safe)}
        if gated is not None:
            where["gated"] = bool(gated)
        return self.append(conversation_id, kind, frame=frame, **fields, **where)

    def tool_end(
        self,
        conversation_id: str,
        *,
        name: str,
        call_id: str,
        status: str,
        frame: int = 0,
        error_type: str | None = None,
        error_code: str | None = None,
        truncated: bool | None = None,
        elapsed_ms: float | None = None,
    ) -> None:
        """Append ``tool/result`` for a ``tool_begin`` (the one emitter of that event).

        ``status`` is ``ok``, ``failed`` or ``unknown``; ``error_type`` is a class name, never
        message text; ``error_code`` a ``ToolErrorCode`` value; a field left ``None`` is
        omitted. No result value is ever written.
        """
        extra: dict[str, Any] = {
            key: value
            for key, value in (
                ("error_type", error_type),
                ("error_code", error_code),
                ("truncated", truncated),
                ("elapsed_ms", elapsed_ms),
            )
            if value is not None
        }
        self.append(
            conversation_id,
            EventType.TOOL_RESULT,
            frame=frame,
            name=name,
            call_id=call_id,
            status=status,
            **extra,
        )

    def approval_request(
        self,
        conversation_id: str,
        *,
        tool_name: str,
        arguments: Mapping[str, Any],
        has_callback: bool,
        frame: int = 0,
    ) -> int:
        """Append ``approval/request`` BEFORE the ask and return its ``seq``: ``tool_name``,
        ``has_callback`` and the arguments as ``tool/call`` has them; never the context or a grant."""
        return self._append_call(
            EventType.APPROVAL_REQUEST,
            conversation_id,
            frame,
            arguments,
            tool_name=tool_name,
            has_callback=bool(has_callback),
        )

    @contextmanager
    def tool_call(
        self,
        conversation_id: str,
        *,
        name: str,
        call_id: str,
        arguments: Mapping[str, Any],
        side_effecting: bool = True,
        frame: int = 0,
    ) -> Iterator[None]:
        """Checkpoint a tool call: ``with log.tool_call(...): run_the_tool()``.

        ``tool_begin`` BEFORE the body (see it), then on a normal exit ``tool/result`` with
        ``status: "ok"``; when the body raises an ``Exception`` it appends ``status: "failed"``
        with the exception's type name (never its message) and re-raises (a failing result
        append is logged, not raised, so the tool's own error is not masked). When the process
        dies inside, or the body is interrupted by a ``BaseException`` that is not an
        ``Exception`` (an interrupt, ``SystemExit``), no result is written: the call stays
        unresolved and a reader reports it as ``unknown``, because the log cannot tell whether
        the tool ran. Nothing retries it. ``ToolRegistry.execute`` does not use this manager:
        it reaches the log through ``core.tool_observer``.
        """
        self.tool_begin(
            conversation_id,
            name=name,
            call_id=call_id,
            arguments=arguments,
            side_effecting=side_effecting,
            frame=frame,
        )
        try:
            yield
        except Exception as exc:
            try:
                self.tool_end(
                    conversation_id,
                    name=name,
                    call_id=call_id,
                    status="failed",
                    frame=frame,
                    error_type=type(exc).__name__,
                )
            except Exception as log_exc:
                logger.warning(
                    f"tool result of {name!r} not logged ({type(log_exc).__name__}); "
                    "the call stays unknown"
                )
            raise
        self.tool_end(
            conversation_id, name=name, call_id=call_id, status="ok", frame=frame
        )

    def prune(
        self,
        *,
        max_age_seconds: float | None = None,
        max_logs: int | None = None,
        blob_grace_seconds: float = 3600.0,
        dry_run: bool = False,
    ) -> PruneReport:
        """Delete old logs, then the blobs no remaining log names; the default deletes nothing.

        Explicit only (``runtime.retention``, plan 5a6a1f80): a log is a candidate when its file
        mtime (never ``ts``) is *max_age_seconds* old or more than *max_logs* newer logs exist,
        and one whose writer lock is held is skipped (``logs_busy``). Then blobs no remaining
        log or live blob names and older than *blob_grace_seconds* go. *dry_run* deletes
        nothing but still takes and gives back the non-blocking writer lock of each candidate
        that has a ``.lock`` (a writer that opens at that instant can see ``LogLockedError``).
        Meant for a root with NO active writer on the same blobs: a re-put in the microseconds
        between the last check and the delete can still lose the blob (the reader then raises
        ``LogCorruptError``). An empty ``.lock`` stays for every pruned log and the mark phase
        reads every live blob in full. ``ValueError`` for a negative limit.
        """
        return prune_root(
            self.root,
            max_age_seconds=max_age_seconds,
            max_logs=max_logs,
            blob_grace_seconds=blob_grace_seconds,
            dry_run=dry_run,
        )

    def release(self, conversation_id: str) -> None:
        """Close and forget the writer of one id (a finished tree); a later append reopens it.

        This is also the recovery of a poisoned writer (see the class contract). While a lane
        thread serves this tree the release runs on it, behind the jobs already queued (so each
        meets the writer it was queued against) and returns before the writer is closed.
        """
        self._lanes.after(conversation_id, lambda: self._release_now(conversation_id))

    def _release_now(self, conversation_id: str) -> None:
        with self._guard:
            writer = self._writers.pop(conversation_id, None)
        if writer is not None:
            writer.close()

    def close(self) -> None:
        """Close every writer and refuse later appends. Idempotent; the files stay.

        A writer is closed behind the jobs already queued on its lane (a job in flight
        finishes first) and the lane threads end once those ran; ``close`` does not wait.
        """
        with self._guard:
            self._closed = True
            writers, self._writers = list(self._writers.items()), {}
        for conversation_id, writer in writers:
            self._lanes.after(conversation_id, writer.close)
        self._lanes.shutdown()

    def __enter__(self) -> EventLog:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _is_valid_id(candidate: str) -> bool:
    try:
        check_conversation_id(candidate)
    except EventPayloadError:
        return False
    return True
