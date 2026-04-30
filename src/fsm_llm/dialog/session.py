from __future__ import annotations

"""
Session persistence for FSM-LLM conversations.

Provides a ``SessionStore`` interface and ``FileSessionStore``
implementation for saving and restoring conversation state across
process restarts.

Usage::

    from fsm_llm.dialog.api import API
    from fsm_llm import FileSessionStore

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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..logging import logger


class SessionState(BaseModel):
    """Serializable snapshot of a conversation's state."""

    conversation_id: str
    fsm_id: str
    current_state: str
    context_data: dict[str, Any] = Field(default_factory=dict)
    conversation_history: list[dict[str, str]] = Field(default_factory=list)
    stack_depth: int = 1
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
        path = self._path(session_id)
        data = state.model_dump()
        # Atomic write via temp file + rename
        tmp_path = path.with_suffix(".tmp")
        try:
            tmp_path.write_text(json.dumps(data, indent=2, default=str))
            os.replace(str(tmp_path), str(path))
            logger.debug(f"Session saved: {session_id}")
        except OSError:
            tmp_path.unlink(missing_ok=True)
            raise

    def load(self, session_id: str) -> SessionState | None:
        path = self._path(session_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return SessionState.model_validate(data)
        except (json.JSONDecodeError, ValueError) as e:
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
