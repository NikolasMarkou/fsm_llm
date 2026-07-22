from __future__ import annotations

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


class SessionState(BaseModel):
    """Serializable snapshot of a conversation's state."""

    conversation_id: str
    fsm_id: str
    current_state: str
    context_data: dict[str, Any] = Field(default_factory=dict)
    conversation_history: list[dict[str, str]] = Field(default_factory=list)
    stack_depth: int = 1
    # Optional carrier for a conversation's WorkingMemory. When populated the
    # shape is {"buffers": {name: {k: v}}, "hidden_buffers": [name, ...]}. The
    # flat context_data does NOT carry WorkingMemory, so it is persisted here.
    # Default None keeps old session files (written before this field existed)
    # loadable unchanged.
    working_memory: dict[str, Any] | None = None
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
        if not re.match(r"^[a-zA-Z0-9_\-]+$", session_id):
            raise ValueError(
                f"Invalid session_id: {session_id!r}. "
                "Only alphanumeric characters, hyphens, and underscores are allowed."
            )
        return self._dir / f"{session_id}.json"

    def save(self, session_id: str, state: SessionState) -> None:
        """Persist a session via an atomic temp-file write + rename.

        Note: the JSON round-trip is lossy for non-JSON-native context
        values (datetime/set/custom objects are coerced to strings via
        ``default=str`` and load back as strings, not their original type).
        """
        path = self._path(session_id)
        data = state.model_dump()
        # Atomic write via a unique temp file + rename. A unique temp name
        # avoids corruption when two saves for the same id race (auto-save).
        fd, tmp_name = tempfile.mkstemp(dir=str(self._dir), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(json.dumps(data, indent=2, default=str))
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
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return SessionState.model_validate(data)
        except (json.JSONDecodeError, ValueError, OSError) as e:
            # OSError covers PermissionError and the TOCTOU window where the file
            # is deleted between exists() and read_text(); contract is to return
            # None on any unreadable/invalid session, never raise (CB3-001).
            logger.warning(f"Failed to load session {session_id}: {e}")
            return None

    def delete(self, session_id: str) -> bool:
        path = self._path(session_id)
        if path.exists():
            path.unlink()
            logger.debug(f"Session deleted: {session_id}")
            return True
        return False

    def list_sessions(self) -> list[str]:
        return [p.stem for p in self._dir.glob("*.json") if p.is_file()]

    def exists(self, session_id: str) -> bool:
        return self._path(session_id).exists()
