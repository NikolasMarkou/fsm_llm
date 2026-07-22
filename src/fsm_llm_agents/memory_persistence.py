from __future__ import annotations

"""
Persistence for agent WorkingMemory.

``fsm_llm.session.FileSessionStore`` persists FSM conversation state but NOT an
agent's :class:`~fsm_llm.memory.WorkingMemory` (only the flat ``context_data``
dict is saved). So even when an agent correctly calls ``remember``, those facts
are lost on process restart. This module closes that gap additively:

- :func:`save_working_memory` / :func:`load_working_memory` — file helpers
  (atomic JSON, lossless for JSON-native buffer values).
- :class:`MemorySessionStore` — a ``SessionStore`` that wraps any base store and
  co-persists a ``WorkingMemory`` per session, so FSM state and working memory
  are saved/restored together through one object.

Nothing imports this by default; it is opt-in.
"""

import json
import os
import re
import tempfile
from pathlib import Path

from fsm_llm.memory import WorkingMemory
from fsm_llm.session import FileSessionStore, SessionState, SessionStore

# Mirror FileSessionStore's id policy (path-traversal protection).
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def _check_id(session_id: str) -> None:
    if not session_id or not _SAFE_ID.match(session_id):
        raise ValueError(
            f"invalid session id {session_id!r}: only [A-Za-z0-9_-] allowed"
        )


def save_working_memory(memory: WorkingMemory, path: str) -> None:
    """Atomically write a WorkingMemory snapshot to a JSON file."""
    target = os.path.expanduser(path)
    parent = os.path.dirname(target)
    if parent:
        os.makedirs(parent, exist_ok=True)
    # DECISION plan-2026-07-19T191147-4b664252/D-008 [STALE] (re-applies plan-2026-07-18/D-011)
    # Do NOT collapse this back to a fixed `f"{target}.tmp"` name. That shape
    # was measured at 562/1200 (47%) FileNotFoundError under 4 threads saving
    # one session: two writers share the temp path, and the first os.replace
    # consumes it out from under the second. `fsm_llm.session.FileSessionStore.
    # save` is the reference implementation — keep the two in lockstep; this
    # file already regressed that fix once. The finally-with-exists-check (not
    # `except OSError`) is what cleans up on ANY failure, not just OSError.
    # `parent or "."`: the temp file MUST land on the same filesystem as the
    # target or os.replace raises EXDEV. A bare filename means the cwd.
    fd, tmp = tempfile.mkstemp(dir=parent or ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(memory.to_dict(), fh, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def load_working_memory(path: str) -> WorkingMemory:
    """Load a WorkingMemory snapshot previously written by save_working_memory."""
    with open(os.path.expanduser(path), encoding="utf-8") as fh:
        data = json.load(fh)
    return WorkingMemory.from_dict(data)


class MemorySessionStore(SessionStore):
    """A ``SessionStore`` that co-persists a ``WorkingMemory`` per session.

    Delegates all FSM session-state persistence to a wrapped base store
    (default: a :class:`FileSessionStore`), and stores each session's
    ``WorkingMemory`` as a sibling JSON file under ``<directory>/_memory/``.

    Use :meth:`save_session_with_memory` / :meth:`load_session_with_memory` to
    persist and restore both together.

    Args:
        base: Base store for ``SessionState``. Defaults to a
            ``FileSessionStore(directory)``.
        directory: Root directory. Required when ``base`` is omitted; also the
            location of the ``_memory`` subdir for WorkingMemory snapshots.
    """

    def __init__(
        self,
        base: SessionStore | None = None,
        directory: str | Path | None = None,
    ) -> None:
        if base is None and directory is None:
            raise ValueError("provide either a base store or a directory")
        self._base = base or FileSessionStore(directory)  # type: ignore[arg-type]
        root = Path(directory) if directory is not None else Path("./sessions")
        self._mem_dir = root / "_memory"

    # --- SessionStore interface (delegates to base) -------------------
    def save(self, session_id: str, state: SessionState) -> None:
        self._base.save(session_id, state)

    def load(self, session_id: str) -> SessionState | None:
        return self._base.load(session_id)

    def delete(self, session_id: str) -> bool:
        removed = self._base.delete(session_id)
        mem_path = self._mem_path(session_id)
        if mem_path.exists():
            mem_path.unlink()
            removed = True
        return removed

    def list_sessions(self) -> list[str]:
        return self._base.list_sessions()

    # --- WorkingMemory co-persistence ---------------------------------
    def _mem_path(self, session_id: str) -> Path:
        _check_id(session_id)
        return self._mem_dir / f"{session_id}.mem.json"

    def save_memory(self, session_id: str, memory: WorkingMemory) -> None:
        """Persist a WorkingMemory for ``session_id``."""
        save_working_memory(memory, str(self._mem_path(session_id)))

    def load_memory(self, session_id: str) -> WorkingMemory | None:
        """Load the WorkingMemory for ``session_id`` (None if absent)."""
        path = self._mem_path(session_id)
        if not path.exists():
            return None
        return load_working_memory(str(path))

    def save_session_with_memory(
        self,
        session_id: str,
        state: SessionState,
        memory: WorkingMemory,
    ) -> None:
        """Persist both FSM session state and working memory atomically-ish."""
        self.save(session_id, state)
        self.save_memory(session_id, memory)

    def load_session_with_memory(
        self, session_id: str
    ) -> tuple[SessionState | None, WorkingMemory | None]:
        """Restore both FSM session state and working memory."""
        return self.load(session_id), self.load_memory(session_id)
