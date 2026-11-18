"""Session projections of the event log: the versioned ``SessionProjection``, its one adjacent
migration, the reader of old ``FileSessionStore`` files and the fold of a log (plan 21e6f00d,
roadmap row 1.2).

Layer ``runtime``: plain dicts and the module's own pydantic model, never ``engine.SessionState``
(``engine`` sits above this layer; ``engine.session.EventLogSessionStore`` adapts the two).

Interface contract:
    - ``SessionProjection`` (``v`` is 2, ``extra="forbid"``): ``conversation_id``, ``fsm_id``,
      ``current_state``, ``context_data`` (public keys only), ``conversation_history``,
      ``conversation_summary``, ``working_memory`` (``{"buffers", "hidden_buffers"}`` or None, the
      shape of ``SessionState.working_memory``), ``provenance``, ``completion_context`` (the
      transcript and response format values the log carried, NOT yet checked against a
      definition: the restore does that), ``unknown_tool_calls`` and ``last_seq``.
    - ``migrate_session(payload) -> SessionProjection``: ``v`` absent is the old ``SessionState``
      JSON (v1) and migrates to v2 (``metadata["pipeline_extracted"]`` becomes ``provenance``;
      ``stack_depth``, ``saved_at`` and every other ``metadata`` key are dropped: advisory; the
      internal-prefixed keys of ``context_data`` are dropped, a legacy file is not trusted); ``v``
      2 is validated as it is; a larger ``v`` raises ``SessionVersionError``, any other malformed
      payload ``LogCorruptError``. A payload nested past ``MAX_PAYLOAD_DEPTH`` is refused.
    - ``projection_from_legacy_file(path) -> SessionProjection``: a bounded binary read
      (``MAX_SESSION_FILE_BYTES``, refused unread above it) of an old ``<id>.json``, parsed with
      ``RecursionError`` and ``MemoryError`` counted as corruption, then ``migrate_session``.
      ``FileNotFoundError`` for a missing file.
    - ``fold_session(events, load_blob) -> SessionProjection | None``: *events* are decoded
      envelopes in ``seq`` order (``EventLog.read``), *load_blob* maps a ``sha256:`` reference to
      its JSON value (``EventLog.get_json``). Reads FRAME 0 only. The state is that of the last
      state-carrying event (``turn/start``, ``turn/end`` and ``context/record`` with a
      ``state_ref``); a ``turn/end`` with outcome ``rolled_back`` returns to the state of its own
      ``turn/start`` whatever it carries; a ``context/record`` with a ``delta_ref`` (the
      ``update_context`` write) is applied onto the state in force (a ``None`` value is set,
      not deleted: ``API.update_context`` keeps the key). A crash leaves no ``turn/end``, so
      the open turn's ``turn/start`` state stands. An unknown event type is skipped. ``tool/call`` events of ANY frame without a ``tool/result``
      become ``unknown_tool_calls``; nothing is executed or retried. Returns None when no event
      carries a state. A bad event stream (a corrupt middle line) raises from *events*; a state
      or delta blob that is missing, tampered or of the wrong shape raises ``LogCorruptError``.
      ``LogPrivacyError`` as soon as ANY event of any frame carries the ``lvl`` marker (a log
      written at level ``shape``, or a ``values`` log a ``shape`` writer continued).
      Internal-prefixed keys of ``context_data`` are dropped here, whatever a blob says.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from fsm_llm.constants import has_internal_prefix
from fsm_llm.runtime.checkpoint import unresolved_tool_calls
from fsm_llm.runtime.constants import MAX_PAYLOAD_DEPTH, MAX_SESSION_FILE_BYTES
from fsm_llm.runtime.events import (
    EventLogError,
    EventType,
    LogCorruptError,
    TurnOutcome,
    check_depth,
)
from fsm_llm.runtime.privacy import refuse_shape

#: The version this module writes and the one a reader accepts.
SESSION_PROJECTION_VERSION = 2

#: The ``metadata`` key of an old ``SessionState`` that holds the provenance digests: the same
#: string as ``engine.session.METADATA_PROVENANCE``, which this layer cannot import (the engine
#: sits above it).
_LEGACY_PROVENANCE_KEY = "pipeline_extracted"


class SessionVersionError(EventLogError, ValueError):
    """A session payload carries a ``v`` newer than this reader understands."""


class SessionProjection(BaseModel):
    """The state of one conversation as the log (or an old session file) says it was."""

    model_config = ConfigDict(extra="forbid")

    v: Literal[2] = 2
    conversation_id: str
    fsm_id: str
    current_state: str
    context_data: dict[str, Any] = Field(default_factory=dict)
    conversation_history: list[dict[str, str]] = Field(default_factory=list)
    conversation_summary: str | None = None
    working_memory: dict[str, Any] | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)
    completion_context: dict[str, Any] = Field(default_factory=dict)
    unknown_tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    last_seq: int = 0


def _validated(data: Mapping[str, Any]) -> SessionProjection:
    """Validate *data*; the error names the failing fields and never echoes a value."""
    try:
        return SessionProjection.model_validate(dict(data))
    except ValidationError as exc:
        fields = sorted({".".join(str(p) for p in e["loc"]) for e in exc.errors()})
        raise LogCorruptError(
            f"session payload is not a valid v2 projection (fields: {', '.join(fields)})"
        ) from None


def _public(context_data: Any) -> Any:
    """*context_data* without its internal-prefixed top-level keys (``has_internal_prefix``).

    The one rule of the projection layer, shared by the fold and both versions of the
    migration; anything that is not a dict is returned as it is for the validation to refuse.
    """
    if not isinstance(context_data, dict):
        return context_data
    return {
        key: value
        for key, value in context_data.items()
        if not (isinstance(key, str) and has_internal_prefix(key))
    }


def _migrate_1_to_2(payload: Mapping[str, Any]) -> dict[str, Any]:
    """The one adjacent migration: an old ``SessionState`` dict to the v2 field names."""
    # DECISION plan-2026-11-17T100529-21e6f00d/D-009
    # An old file is NOT trusted to hold only public keys: `context_data` is filtered by
    # `_public` here, exactly as `build_projection` filters the fold, so a forged `_approval_granted` (or
    # any internal key) in a legacy file never reaches a restore. Do NOT copy `context_data`
    # through unfiltered, and do NOT carry internal keys by prefix or deny-list: the only
    # internal keys a restore takes are a definition's own transcript keys (`completion_context`,
    # checked against the current definition by the restore). See decisions.md D-009.
    metadata = payload.get("metadata")
    provenance = (
        metadata.get(_LEGACY_PROVENANCE_KEY) if isinstance(metadata, dict) else None
    )
    return {
        "v": 2,
        "conversation_id": payload.get("conversation_id"),
        "fsm_id": payload.get("fsm_id"),
        "current_state": payload.get("current_state"),
        "context_data": _public(payload.get("context_data", {})),
        "conversation_history": payload.get("conversation_history", []),
        "conversation_summary": payload.get("conversation_summary"),
        "working_memory": payload.get("working_memory"),
        "provenance": provenance if isinstance(provenance, dict) else {},
    }


def migrate_session(payload: Mapping[str, Any]) -> SessionProjection:
    """Return the v2 projection of *payload* (see the module docstring)."""
    if not isinstance(payload, Mapping):
        raise LogCorruptError("session payload is not a JSON object")
    if not check_depth(payload, MAX_PAYLOAD_DEPTH):
        raise LogCorruptError(f"session payload nests deeper than {MAX_PAYLOAD_DEPTH}")
    version = payload.get("v", 1)
    if isinstance(version, bool) or not isinstance(version, int):
        raise LogCorruptError("session payload has a non-integer version")
    if version > SESSION_PROJECTION_VERSION:
        raise SessionVersionError(
            f"session payload is version {version}; this reader understands up to "
            f"{SESSION_PROJECTION_VERSION}"
        )
    if version == 1:
        return _validated(_migrate_1_to_2(payload))
    if version == SESSION_PROJECTION_VERSION:
        return _validated(
            {**payload, "context_data": _public(payload.get("context_data", {}))}
        )
    raise LogCorruptError(f"session payload has an unknown version {version}")


def projection_from_legacy_file(path: str | os.PathLike[str]) -> SessionProjection:
    """Read an old ``FileSessionStore`` file and migrate it (see the module docstring)."""
    with open(path, "rb") as handle:
        data = handle.read(MAX_SESSION_FILE_BYTES + 1)
    if len(data) > MAX_SESSION_FILE_BYTES:
        raise LogCorruptError(
            f"session file is larger than {MAX_SESSION_FILE_BYTES} bytes"
        )
    try:
        payload = json.loads(data.decode("utf-8"))
    except (ValueError, RecursionError, MemoryError) as exc:
        raise LogCorruptError(
            f"session file is not valid JSON ({type(exc).__name__})"
        ) from None
    return migrate_session(payload)


def _apply_delta(context: dict[str, Any], internal: dict[str, Any], delta: Any) -> None:
    """Apply one ``update_context`` delta blob; a ``None`` value is a value, as in the write."""
    if not isinstance(delta, dict):
        raise LogCorruptError("a context delta blob is not a JSON object")
    for target, part, keep_internal in (
        (context, delta.get("context"), False),
        (internal, delta.get("internal"), True),
    ):
        if not isinstance(part, dict):
            continue
        for key, value in part.items():
            if not keep_internal and has_internal_prefix(key):
                continue
            target[key] = value


def fold_session(
    events: Iterable[Mapping[str, Any]], load_blob: Callable[[str], Any]
) -> SessionProjection | None:
    """Return the projection of a log (see the module docstring).

    Tracks block references only; the blobs of the winning state and of the deltas after it are
    read once, at the end.
    """
    ref: str | None = None  # the state blob in force
    deltas: list[str] = []  # the delta blobs written after it
    turn_start: str | None = None  # the state blob the open turn began with
    tools: list[Mapping[str, Any]] = []
    conversation_id: Any = None
    last_seq = 0
    for event in events:
        refuse_shape(event)  # a log without values holds no state: never "unknown id"
        last_seq = event["seq"]
        kind = event.get("type")
        if kind in (EventType.TOOL_CALL.value, EventType.TOOL_RESULT.value):
            tools.append(event)
        if event.get("frame") != 0:
            continue
        conversation_id = event.get("conversation_id")
        state_ref = event.get("state_ref")
        state_ref = state_ref if isinstance(state_ref, str) else None
        if kind == EventType.TURN_START.value and state_ref:
            ref, deltas = state_ref, []
            turn_start = state_ref
        elif kind == EventType.TURN_END.value:
            if event.get("outcome") == TurnOutcome.ROLLED_BACK.value:
                state_ref = turn_start or state_ref
            if state_ref:
                ref, deltas = state_ref, []
            turn_start = None
        elif kind == EventType.CONTEXT_RECORD.value:
            if state_ref:
                ref, deltas = state_ref, []
            elif ref and isinstance(event.get("delta_ref"), str):
                deltas.append(event["delta_ref"])
    if ref is None:
        return None
    return build_projection(ref, deltas, load_blob, conversation_id, last_seq, tools)


def _mapping(blob: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = blob.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise LogCorruptError(f"state blob field {key!r} is not a JSON object")
    return value


def build_projection(
    ref: str,
    deltas: list[str],
    load_blob: Callable[[str], Any],
    conversation_id: Any,
    last_seq: int,
    tools: list[Mapping[str, Any]],
) -> SessionProjection:
    """Read the winning state blob, apply the deltas after it and validate the result.

    The one builder of a state projection: ``fold_session`` calls it for the root and
    ``stores.stack.fold_stack`` for every frame (*conversation_id* is then the frame's own id).
    """
    # DECISION plan-2026-11-17T100529-21e6f00d/D-009
    # The fold drops internal-prefixed keys of `context_data` WHATEVER a blob says (`_public`),
    # and a delta writes internal keys only into `completion_context`, never into the context.
    # Do NOT "simplify" this to trust the blob: a log is a file, and a forged blob or delta
    # carrying `_approval_granted` would otherwise be a forgery path into the HITL gate at
    # restore. The same rule guards the migration (`_migrate_1_to_2`). See decisions.md D-009.
    blob = load_blob(ref)
    if not isinstance(blob, dict):
        raise LogCorruptError("a state blob is not a JSON object")
    context = _public(_mapping(blob, "context_data"))
    internal = dict(_mapping(blob, "internal"))
    for delta_ref in deltas:
        _apply_delta(context, internal, load_blob(delta_ref))
    memory = blob.get("working_memory")
    return _validated(
        {
            "conversation_id": conversation_id,
            "fsm_id": blob.get("fsm_id"),
            "current_state": blob.get("current_state"),
            "context_data": context,
            "conversation_history": blob.get("conversation_history") or [],
            "conversation_summary": blob.get("conversation_summary"),
            "working_memory": (
                None
                if memory is None
                else {
                    "buffers": memory,
                    "hidden_buffers": blob.get("hidden_buffers") or [],
                }
            ),
            "provenance": _mapping(blob, "provenance"),
            "completion_context": internal,
            "unknown_tool_calls": unresolved_tool_calls(tools),
            "last_seq": last_seq,
        }
    )
