"""Tests for WorkingMemory persistence + MemorySessionStore."""

from __future__ import annotations

import pytest

from fsm_llm.memory import BUFFER_CORE, WorkingMemory
from fsm_llm.session import SessionState
from fsm_llm_agents import (
    MemorySessionStore,
    load_working_memory,
    save_working_memory,
)


def _memory():
    m = WorkingMemory()
    m.set(BUFFER_CORE, "name", "Ada")
    m.set(BUFFER_CORE, "lang", "Python")
    return m


def _state(sid="sess1"):
    return SessionState(
        conversation_id=sid,
        fsm_id="fsm",
        current_state="start",
        context_data={"k": "v"},
    )


class TestWorkingMemoryFileHelpers:
    def test_save_and_load_roundtrip(self, tmp_path):
        path = str(tmp_path / "mem.json")
        save_working_memory(_memory(), path)
        loaded = load_working_memory(path)
        assert loaded.get(BUFFER_CORE, "name") == "Ada"
        assert loaded.get(BUFFER_CORE, "lang") == "Python"

    def test_save_creates_parent_dirs(self, tmp_path):
        path = str(tmp_path / "nested" / "deep" / "mem.json")
        save_working_memory(_memory(), path)
        assert load_working_memory(path).get(BUFFER_CORE, "name") == "Ada"


class TestMemorySessionStore:
    def test_requires_base_or_directory(self):
        with pytest.raises(ValueError):
            MemorySessionStore()

    def test_session_state_roundtrip(self, tmp_path):
        store = MemorySessionStore(directory=str(tmp_path))
        store.save("sess1", _state())
        loaded = store.load("sess1")
        assert loaded is not None
        assert loaded.context_data == {"k": "v"}

    def test_memory_roundtrip(self, tmp_path):
        store = MemorySessionStore(directory=str(tmp_path))
        store.save_memory("sess1", _memory())
        mem = store.load_memory("sess1")
        assert mem is not None
        assert mem.get(BUFFER_CORE, "lang") == "Python"

    def test_load_memory_absent_returns_none(self, tmp_path):
        store = MemorySessionStore(directory=str(tmp_path))
        assert store.load_memory("nope") is None

    def test_save_and_load_together(self, tmp_path):
        store = MemorySessionStore(directory=str(tmp_path))
        store.save_session_with_memory("sess1", _state(), _memory())
        state, mem = store.load_session_with_memory("sess1")
        assert state is not None and mem is not None
        assert state.current_state == "start"
        assert mem.get(BUFFER_CORE, "name") == "Ada"

    def test_delete_removes_both(self, tmp_path):
        store = MemorySessionStore(directory=str(tmp_path))
        store.save_session_with_memory("sess1", _state(), _memory())
        assert store.delete("sess1") is True
        assert store.load("sess1") is None
        assert store.load_memory("sess1") is None

    def test_list_sessions_delegates(self, tmp_path):
        store = MemorySessionStore(directory=str(tmp_path))
        store.save("sess1", _state("sess1"))
        store.save("sess2", _state("sess2"))
        assert set(store.list_sessions()) == {"sess1", "sess2"}

    def test_rejects_unsafe_session_id(self, tmp_path):
        store = MemorySessionStore(directory=str(tmp_path))
        with pytest.raises(ValueError):
            store.save_memory("../evil", _memory())
