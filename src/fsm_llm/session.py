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
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .logging import logger
from .utilities import redact_non_json_leaf

# One id rule for _path and list_sessions, matched with fullmatch: ``$`` in a
# ``re.match`` pattern also matches before a trailing newline ("abc\n").
_SESSION_ID_RE = re.compile(r"[a-zA-Z0-9_\-]+")


def _session_json_default(value: Any) -> str:
    """``json.dumps`` ``default=`` hook for ``FileSessionStore.save``.

    Returns ``str(value)`` only for an exact stdlib value scalar (the
    ``utilities.redact_non_json_leaf`` keep-set; json never calls ``default``
    for its other kept types), else ``"<redacted:TypeName>"``. Never raises.
    """
    # DECISION plan-2026-09-22T080837-8b258a25/D-008
    # Do NOT go back to `default=str`: an object's `__str__` wrote its secret
    # to disk. Do NOT copy the scalar type set here; reuse the leaf hook.
    redacted = redact_non_json_leaf(value)
    return str(value) if redacted is value else redacted


class SessionState(BaseModel):
    """Serializable snapshot of a conversation's state."""

    conversation_id: str
    fsm_id: str
    current_state: str
    context_data: dict[str, Any] = Field(default_factory=dict)
    conversation_history: list[dict[str, str]] = Field(default_factory=list)
    stack_depth: int = 1
    # Optional carrier for a conversation's WorkingMemory. When populated the
    # shape is {"buffers": {name: {k: v}, ..., "_hidden_buffers": [name, ...]},
    # "hidden_buffers": [name, ...]}. "buffers" is WorkingMemory.to_dict()'s
    # OWN return verbatim (D-021, memory.py), which since D-021 embeds a
    # "_hidden_buffers" list INSIDE that dict alongside the real buffer-name
    # keys (never a buffer's own contents itself -- see memory.py D-026,
    # which makes that name unusable as a real buffer name). The outer
    # sibling "hidden_buffers" key (api.py's save_session) carries the exact
    # same list explicitly, so the two are redundant on this specific path
    # (api.py always passes hidden_buffers= explicitly to
    # WorkingMemory.from_dict on restore, so the embedded copy inside
    # "buffers" is read but never relied on here) -- see decisions.md D-021
    # for why the embedded key exists at all (a DIFFERENT caller,
    # fsm_llm_agents/memory_persistence.py, has no sibling key and depends on
    # it). The flat context_data does NOT carry WorkingMemory, so it is
    # persisted here. Default None keeps old session files (written before
    # this field existed) loadable unchanged.
    working_memory: dict[str, Any] | None = None
    # A6: `Conversation.summary`, the digest of exchanges trimmed out of
    # `conversation_history`. Default None keeps session files written
    # before this field existed loadable unchanged.
    conversation_summary: str | None = None
    saved_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
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
        if not _SESSION_ID_RE.fullmatch(session_id):
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
        ``str()`` is never called (see ``_session_json_default``). The file
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
            with os.fdopen(fd, "w") as f:
                f.write(json.dumps(data, indent=2, default=_session_json_default))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, str(path))
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
            data = json.loads(path.read_text())
            return SessionState.model_validate(data)
        except (json.JSONDecodeError, ValueError, OSError) as e:
            # OSError covers PermissionError, ENAMETOOLONG from exists() and the
            # TOCTOU window where the file is deleted between exists() and
            # read_text(); contract is to return
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
            if _SESSION_ID_RE.fullmatch(p.stem) and p.is_file()
        ]

    def exists(self, session_id: str) -> bool:
        path = self._path(session_id)
        try:
            return path.exists()
        except OSError as e:
            logger.warning(f"Failed to check session {session_id}: {e}")
            return False
