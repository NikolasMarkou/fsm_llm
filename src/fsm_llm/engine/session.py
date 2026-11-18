"""
Session persistence for FSM-LLM conversations.

Provides a ``SessionStore`` interface and ``FileSessionStore``
implementation for saving and restoring conversation state across
process restarts.

Usage::

    from fsm_llm import API, FileSessionStore

    store = FileSessionStore("./sessions")
    api = API.from_file("bot.json", model="gpt-4", session_store=store)

    # Start or resume conversation
    conv_id, response = api.start_conversation()

    # State is auto-saved after each converse() call
    response = api.converse("Hello!", conv_id)

    # Explicit save/load
    api.save_session(conv_id)
    api.load_session(conv_id)
"""

from __future__ import annotations

import abc
import json
import os
import re
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from fsm_llm.context.filtering import redacting_json_default
from fsm_llm.logging import logger
from fsm_llm.runtime import EventLog, EventLogError, EventPayloadError
from fsm_llm.runtime.stores import (
    SessionProjection,
    check_conversation_id,
    fold_session,
    projection_from_legacy_file,
)

# One id rule for _path, list_sessions and ``MemorySessionStore``, matched with
# ``fullmatch``: ``$`` in a ``re.match`` pattern also matches before a trailing
# newline ("abc\n").
SESSION_ID_RE = re.compile(r"[a-zA-Z0-9_\-]+")

# The ``SessionState.metadata`` keys of a session loaded from the event log (the shape is
# documented on ``EventLogSessionStore``; ``SessionState`` itself has no field for them).
METADATA_PROVENANCE = "pipeline_extracted"
METADATA_COMPLETION_CONTEXT = "completion_context"
METADATA_UNKNOWN_TOOL_CALLS = "unknown_tool_calls"
METADATA_LAST_SEQ = "last_seq"


class SessionState(BaseModel):
    """Serializable snapshot of a conversation's state."""

    conversation_id: str
    fsm_id: str
    current_state: str
    context_data: dict[str, Any] = Field(default_factory=dict)
    conversation_history: list[dict[str, str]] = Field(default_factory=list)
    stack_depth: int = Field(
        default=1,
        description=(
            "advisory; written by save_session, never read by restore "
            "(stacks are not restorable; only the root frame is saved)."
        ),
    )
    # Optional carrier for a conversation's WorkingMemory. When populated the
    # shape is {"buffers": {name: {k: v}, ..., "_hidden_buffers": [name, ...]},
    # "hidden_buffers": [name, ...]}. "buffers" is WorkingMemory.to_dict()'s
    # OWN return verbatim (D-021, memory.py), which since D-021 embeds a
    # "_hidden_buffers" list INSIDE that dict alongside the real buffer-name
    # keys (never a buffer's own contents itself -- see memory.py D-026,
    # which makes that name unusable as a real buffer name). The outer
    # sibling "hidden_buffers" key (engine/api/sessions.py's save_session) carries the exact
    # same list explicitly, so the two are redundant on this specific path
    # (engine/api/sessions.py always passes hidden_buffers= explicitly to
    # WorkingMemory.from_dict on restore, so the embedded copy inside
    # "buffers" is read but never relied on here) -- see decisions.md D-021
    # for why the embedded key exists at all (a DIFFERENT caller,
    # fsm_llm/agents/memory_persistence.py, has no sibling key and depends on
    # it). The flat context_data does NOT carry WorkingMemory, so it is
    # persisted here. Default None keeps old session files (written before
    # this field existed) loadable unchanged.
    working_memory: dict[str, Any] | None = None
    # A6: `Conversation.summary`, the digest of exchanges trimmed out of
    # `conversation_history`. Default None keeps session files written
    # before this field existed loadable unchanged.
    conversation_summary: str | None = None
    saved_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionStore(abc.ABC):
    """Abstract interface for conversation session persistence."""

    @abc.abstractmethod
    def save(self, session_id: str, state: SessionState) -> None:
        """Save session state.

        Args:
            session_id: Unique session identifier.
            state: Session state to persist.
        """
        ...

    @abc.abstractmethod
    def load(self, session_id: str) -> SessionState | None:
        """Load session state.

        Args:
            session_id: Unique session identifier.

        Returns:
            Session state if found, None otherwise.
        """
        ...

    @abc.abstractmethod
    def delete(self, session_id: str) -> bool:
        """Delete a saved session.

        Args:
            session_id: Unique session identifier.

        Returns:
            True if session existed and was deleted.
        """
        ...

    @abc.abstractmethod
    def list_sessions(self) -> list[str]:
        """List all saved session IDs.

        Returns:
            List of session identifiers.
        """
        ...

    def exists(self, session_id: str) -> bool:
        """Check whether a session exists in the store.

        Default implementation uses ``load()``.  Subclasses may override
        for efficiency.
        """
        return self.load(session_id) is not None


class FileSessionStore(SessionStore):
    """File-based session store using JSON files.

    Each session is stored as a separate JSON file in the given
    directory.  Thread-safe for distinct session IDs (no locking
    across files).

    Args:
        directory: Path to the directory for session files.
            Created automatically if it does not exist.
    """

    def __init__(self, directory: str | Path) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"FileSessionStore initialized at {self._dir}")

    def _path(self, session_id: str) -> Path:
        # Validate session_id to prevent path traversal
        if not SESSION_ID_RE.fullmatch(session_id):
            raise ValueError(
                f"Invalid session_id: {session_id!r}. "
                "Only alphanumeric characters, hyphens, and underscores are allowed."
            )
        return self._dir / f"{session_id}.json"

    def save(self, session_id: str, state: SessionState) -> None:
        """Persist a session via an atomic temp-file write + rename.

        Note: the JSON round-trip is lossy for non-JSON-native context
        values. A tuple loads back as a list. A value whose EXACT type is
        ``datetime``/``date``/``time``/``timedelta``/``Decimal``/``UUID`` is
        written as its ``str()`` and loads back as that string. Anything else
        that is not JSON-native (set, bytes, a subclass of those scalars, a
        custom object) is written as ``"<redacted:TypeName>"``; its
        ``str()`` is never called (see ``context.filtering.redacting_json_default``). The file
        still holds the FULL context: secret-looking KEYS are removed only
        from LLM prompts. Keep the session directory as private as the data
        it stores.
        """
        path = self._path(session_id)
        data = state.model_dump()
        # Atomic write via a unique temp file + rename. A unique temp name
        # avoids corruption when two saves for the same id race (auto-save).
        fd, tmp_name = tempfile.mkstemp(dir=str(self._dir), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(json.dumps(data, indent=2, default=redacting_json_default))
                f.flush()
                os.fsync(f.fileno())
            # Windows only: while another writer's os.replace holds the same target, this one
            # raises PermissionError (WinError 5); retry with a short backoff (about 0.3 s in all).
            # POSIX never retries: there PermissionError is a real permission problem. Written out
            # in each of the three save sites on purpose (plan d7719b03, D-010): the anchors of
            # these sites refuse a shared atomic-save helper without a maintainer decision.
            delays = iter(
                (0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.05, 0.05, 0.05, 0.05, 0.05)
            )
            while True:
                try:
                    os.replace(tmp_name, str(path))
                    break
                except PermissionError:
                    delay = next(delays, None) if sys.platform == "win32" else None
                    if delay is None:
                        raise
                    time.sleep(delay)
            logger.debug(f"Session saved: {session_id}")
        finally:
            # DECISION plan-2026-07-18T162030-a02151fe/D-011 [STALE]
            # Do NOT "simplify" this back to `except OSError: ... raise`. That
            # shape leaked the temp file on every non-OSError exit (RuntimeError,
            # MemoryError, a TypeError out of json.dumps, KeyboardInterrupt).
            # Do NOT use `except BaseException:` either — forbidden pattern.
            # The existence check is what makes this a no-op after a successful
            # os.replace, which consumed tmp_name.
            if os.path.exists(tmp_name):
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass

    def load(self, session_id: str) -> SessionState | None:
        path = self._path(session_id)
        try:
            if not path.exists():
                return None
            data = json.loads(path.read_text(encoding="utf-8"))
            return SessionState.model_validate(data)
        except (json.JSONDecodeError, ValueError, OSError, RecursionError) as e:
            # OSError covers PermissionError, ENAMETOOLONG from exists() and the
            # TOCTOU window where the file is deleted between exists() and
            # read_text(); RecursionError is a JSON too deep for a count-based
            # parser (3.11 to 3.13), not a ValueError; contract is to return
            # None on any unreadable/invalid session, never raise (CB3-001).
            logger.warning(f"Failed to load session {session_id}: {e}")
            return None

    def delete(self, session_id: str) -> bool:
        path = self._path(session_id)
        # Unlink directly: an exists()-then-unlink() pair races a concurrent
        # delete. A missing file is False; any other OSError is logged, False.
        try:
            path.unlink()
        except FileNotFoundError:
            return False
        except OSError as e:
            logger.warning(f"Failed to delete session {session_id}: {e}")
            return False
        logger.debug(f"Session deleted: {session_id}")
        return True

    def list_sessions(self) -> list[str]:
        # Only ids _path accepts, so every listed id round-trips through load.
        return [
            p.stem
            for p in self._dir.glob("*.json")
            if SESSION_ID_RE.fullmatch(p.stem) and p.is_file()
        ]

    def exists(self, session_id: str) -> bool:
        path = self._path(session_id)
        try:
            return path.exists()
        except OSError as e:
            logger.warning(f"Failed to check session {session_id}: {e}")
            return False


def session_state_from_projection(
    projection: SessionProjection, *, stack_depth: int = 1
) -> SessionState:
    """The ``SessionState`` of a projection; the log-only parts ride ``metadata``.

    *stack_depth* fills the advisory ``SessionState.stack_depth`` (``fork`` passes the depth of
    the stack it folded; the root-only store leaves 1).
    """
    metadata: dict[str, Any] = {}
    if projection.provenance:
        metadata[METADATA_PROVENANCE] = projection.provenance
    if projection.completion_context:
        metadata[METADATA_COMPLETION_CONTEXT] = projection.completion_context
    if projection.unknown_tool_calls:
        metadata[METADATA_UNKNOWN_TOOL_CALLS] = projection.unknown_tool_calls
    if projection.last_seq:
        metadata[METADATA_LAST_SEQ] = projection.last_seq
    return SessionState(
        conversation_id=projection.conversation_id,
        fsm_id=projection.fsm_id,
        current_state=projection.current_state,
        context_data=projection.context_data,
        conversation_history=projection.conversation_history,
        working_memory=projection.working_memory,
        conversation_summary=projection.conversation_summary,
        stack_depth=stack_depth,
        metadata=metadata,
    )


class EventLogSessionStore(SessionStore):
    """A read-only session store over an ``EventLog``: what a conversation looked like when the
    process that wrote its log stopped (plan 21e6f00d, D-006).

    Resume with the existing verb: ``API(session_store=EventLogSessionStore(log))
    .restore_session(old_id)`` starts a NEW conversation seeded from the log of *old_id*.

    Interface contract:
        - ``EventLogSessionStore(log, *, legacy_dir=None)``: *log* is the ``EventLog`` to read;
          *legacy_dir*, when given, is a ``FileSessionStore`` directory (created if missing)
          consulted for an id the log has no state for, through the versioned projection
          (``projection_from_legacy_file``), so old and new sessions restore through one path.
        - ``load(id)``: the fold of the log of root conversation *id* (the state of its last
          committed turn, or the pre-turn state of an open one, frame 0 only), as a
          ``SessionState`` whose ``metadata`` holds ``pipeline_extracted`` (provenance),
          ``completion_context`` (the transcript and response format values the log carried),
          ``unknown_tool_calls`` (tool calls the log cannot resolve, never retried) and
          ``last_seq`` (the last event the fold read); a key is absent when it is empty.
          ``None`` when neither the log nor the legacy directory knows *id*. A corrupt log raises
          ``LogCorruptError``; it is never skipped past. An id outside the log's id rule raises
          ``EventPayloadError`` unless a legacy directory can hold it.
        - ``exists``, ``list_sessions`` (the log ids and the legacy ids, sorted): reads.
        - ``save``: a documented no-op (the engine writes the log itself, ``API`` auto-save
          still calls it). ``delete`` raises ``EventLogError``: the log is append-only.
    """

    def __init__(self, log: EventLog, *, legacy_dir: str | Path | None = None) -> None:
        self._log = log
        self._legacy = None if legacy_dir is None else FileSessionStore(legacy_dir)

    def _projection(self, session_id: str) -> SessionProjection | None:
        try:
            check_conversation_id(session_id)
        except EventPayloadError:
            if self._legacy is None:
                raise
        else:
            projection = fold_session(self._log.read(session_id), self._log.get_json)
            if projection is not None:
                return projection
        if self._legacy is None:
            return None
        try:
            return projection_from_legacy_file(self._legacy._path(session_id))
        except FileNotFoundError:
            return None

    def load(self, session_id: str) -> SessionState | None:
        projection = self._projection(session_id)
        return None if projection is None else session_state_from_projection(projection)

    def save(self, session_id: str, state: SessionState) -> None:
        """No-op: the log is written by the engine, never by a snapshot."""

    def delete(self, session_id: str) -> bool:
        raise EventLogError(
            "EventLogSessionStore cannot delete: the event log is append-only"
        )

    def list_sessions(self) -> list[str]:
        ids = set(self._log.conversation_ids())
        if self._legacy is not None:
            ids.update(self._legacy.list_sessions())
        return sorted(ids)
