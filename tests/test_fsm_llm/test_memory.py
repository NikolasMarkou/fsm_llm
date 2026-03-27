"""Tests for WorkingMemory structured buffer system."""

import pytest

from fsm_llm.memory import (
    BUFFER_CORE,
    BUFFER_ENVIRONMENT,
    BUFFER_REASONING,
    BUFFER_SCRATCH,
    DEFAULT_BUFFERS,
    WorkingMemory,
)


class TestWorkingMemoryInit:
    """Test WorkingMemory construction."""

    def test_default_buffers(self):
        memory = WorkingMemory()
        assert set(memory.list_buffers()) == set(DEFAULT_BUFFERS)

    def test_custom_buffers(self):
        memory = WorkingMemory(buffers=["a", "b"])
        assert memory.list_buffers() == ["a", "b"]

    def test_initial_data_goes_to_core(self):
        memory = WorkingMemory(initial_data={"name": "Alice"})
        assert memory.get(BUFFER_CORE, "name") == "Alice"

    def test_initial_data_ignored_if_no_core_buffer(self):
        memory = WorkingMemory(buffers=["custom"], initial_data={"name": "Alice"})
        assert not memory.has_buffer(BUFFER_CORE)

    def test_empty_memory(self):
        memory = WorkingMemory()
        assert len(memory) == 0


class TestWorkingMemoryCRUD:
    """Test get/set/delete operations."""

    def test_set_and_get(self):
        memory = WorkingMemory()
        memory.set(BUFFER_CORE, "name", "Alice")
        assert memory.get(BUFFER_CORE, "name") == "Alice"

    def test_get_default(self):
        memory = WorkingMemory()
        assert memory.get(BUFFER_CORE, "missing") is None
        assert memory.get(BUFFER_CORE, "missing", "default") == "default"

    def test_get_nonexistent_buffer_raises(self):
        memory = WorkingMemory()
        with pytest.raises(KeyError, match="does not exist"):
            memory.get("nonexistent", "key")

    def test_set_creates_buffer_if_missing(self):
        memory = WorkingMemory()
        memory.set("new_buffer", "key", "value")
        assert memory.has_buffer("new_buffer")
        assert memory.get("new_buffer", "key") == "value"

    def test_delete_existing_key(self):
        memory = WorkingMemory()
        memory.set(BUFFER_CORE, "name", "Alice")
        assert memory.delete(BUFFER_CORE, "name") is True
        assert memory.get(BUFFER_CORE, "name") is None

    def test_delete_missing_key(self):
        memory = WorkingMemory()
        assert memory.delete(BUFFER_CORE, "missing") is False

    def test_delete_missing_buffer(self):
        memory = WorkingMemory()
        assert memory.delete("nonexistent", "key") is False

    def test_overwrite_value(self):
        memory = WorkingMemory()
        memory.set(BUFFER_CORE, "name", "Alice")
        memory.set(BUFFER_CORE, "name", "Bob")
        assert memory.get(BUFFER_CORE, "name") == "Bob"

    def test_different_types(self):
        memory = WorkingMemory()
        memory.set(BUFFER_CORE, "str_val", "hello")
        memory.set(BUFFER_CORE, "int_val", 42)
        memory.set(BUFFER_CORE, "list_val", [1, 2, 3])
        memory.set(BUFFER_CORE, "dict_val", {"nested": True})
        assert memory.get(BUFFER_CORE, "str_val") == "hello"
        assert memory.get(BUFFER_CORE, "int_val") == 42
        assert memory.get(BUFFER_CORE, "list_val") == [1, 2, 3]
        assert memory.get(BUFFER_CORE, "dict_val") == {"nested": True}


class TestWorkingMemoryBuffers:
    """Test buffer-level operations."""

    def test_get_buffer_copy(self):
        memory = WorkingMemory()
        memory.set(BUFFER_CORE, "name", "Alice")
        buf = memory.get_buffer(BUFFER_CORE)
        assert buf == {"name": "Alice"}
        # Verify it's a copy
        buf["name"] = "Bob"
        assert memory.get(BUFFER_CORE, "name") == "Alice"

    def test_get_buffer_nonexistent_raises(self):
        memory = WorkingMemory()
        with pytest.raises(KeyError, match="does not exist"):
            memory.get_buffer("nonexistent")

    def test_clear_buffer(self):
        memory = WorkingMemory()
        memory.set(BUFFER_CORE, "a", 1)
        memory.set(BUFFER_CORE, "b", 2)
        memory.clear_buffer(BUFFER_CORE)
        assert memory.get_buffer(BUFFER_CORE) == {}

    def test_clear_nonexistent_buffer_raises(self):
        memory = WorkingMemory()
        with pytest.raises(KeyError, match="does not exist"):
            memory.clear_buffer("nonexistent")

    def test_has_buffer(self):
        memory = WorkingMemory()
        assert memory.has_buffer(BUFFER_CORE) is True
        assert memory.has_buffer("nonexistent") is False

    def test_create_buffer(self):
        memory = WorkingMemory()
        memory.create_buffer("custom")
        assert memory.has_buffer("custom")

    def test_create_buffer_idempotent(self):
        memory = WorkingMemory()
        memory.set(BUFFER_CORE, "key", "value")
        memory.create_buffer(BUFFER_CORE)  # Should not clear
        assert memory.get(BUFFER_CORE, "key") == "value"

    def test_update_buffer(self):
        memory = WorkingMemory()
        memory.set(BUFFER_CORE, "a", 1)
        memory.update_buffer(BUFFER_CORE, {"b": 2, "c": 3})
        assert memory.get(BUFFER_CORE, "a") == 1
        assert memory.get(BUFFER_CORE, "b") == 2
        assert memory.get(BUFFER_CORE, "c") == 3

    def test_update_buffer_creates_if_missing(self):
        memory = WorkingMemory()
        memory.update_buffer("new", {"key": "val"})
        assert memory.has_buffer("new")
        assert memory.get("new", "key") == "val"

    def test_import_flat_data(self):
        memory = WorkingMemory()
        memory.import_flat_data({"a": 1, "b": 2})
        assert memory.get(BUFFER_CORE, "a") == 1
        assert memory.get(BUFFER_CORE, "b") == 2

    def test_import_flat_data_custom_buffer(self):
        memory = WorkingMemory()
        memory.import_flat_data({"result": "ok"}, target_buffer=BUFFER_ENVIRONMENT)
        assert memory.get(BUFFER_ENVIRONMENT, "result") == "ok"


class TestWorkingMemoryGetAllData:
    """Test flattened view generation."""

    def test_single_buffer(self):
        memory = WorkingMemory()
        memory.set(BUFFER_CORE, "a", 1)
        memory.set(BUFFER_CORE, "b", 2)
        assert memory.get_all_data() == {"a": 1, "b": 2}

    def test_multiple_buffers_no_collision(self):
        memory = WorkingMemory()
        memory.set(BUFFER_CORE, "name", "Alice")
        memory.set(BUFFER_ENVIRONMENT, "result", "ok")
        memory.set(BUFFER_REASONING, "step", "1")
        data = memory.get_all_data()
        assert data == {"name": "Alice", "result": "ok", "step": "1"}

    def test_core_wins_on_collision(self):
        memory = WorkingMemory()
        memory.set(BUFFER_CORE, "key", "core_value")
        memory.set(BUFFER_ENVIRONMENT, "key", "env_value")
        assert memory.get_all_data()["key"] == "core_value"

    def test_empty_buffers(self):
        memory = WorkingMemory()
        assert memory.get_all_data() == {}


class TestWorkingMemoryScopedView:
    """Test scoped view generation."""

    def test_scoped_view_filters_keys(self):
        memory = WorkingMemory()
        memory.set(BUFFER_CORE, "name", "Alice")
        memory.set(BUFFER_CORE, "email", "alice@example.com")
        memory.set(BUFFER_CORE, "age", 30)
        scoped = memory.to_scoped_view(["name", "email"])
        assert scoped == {"name": "Alice", "email": "alice@example.com"}

    def test_scoped_view_missing_keys_ignored(self):
        memory = WorkingMemory()
        memory.set(BUFFER_CORE, "name", "Alice")
        scoped = memory.to_scoped_view(["name", "missing"])
        assert scoped == {"name": "Alice"}

    def test_scoped_view_empty_keys(self):
        memory = WorkingMemory()
        memory.set(BUFFER_CORE, "name", "Alice")
        assert memory.to_scoped_view([]) == {}

    def test_scoped_view_cross_buffer(self):
        memory = WorkingMemory()
        memory.set(BUFFER_CORE, "name", "Alice")
        memory.set(BUFFER_ENVIRONMENT, "search_result", "found")
        scoped = memory.to_scoped_view(["name", "search_result"])
        assert scoped == {"name": "Alice", "search_result": "found"}


class TestWorkingMemorySearch:
    """Test search across buffers."""

    def test_search_by_key_name(self):
        memory = WorkingMemory()
        memory.set(BUFFER_CORE, "user_name", "Alice")
        results = memory.search("user_name")
        assert len(results) == 1
        assert results[0] == (BUFFER_CORE, "user_name", "Alice")

    def test_search_by_string_value(self):
        memory = WorkingMemory()
        memory.set(BUFFER_CORE, "name", "Alice")
        results = memory.search("Alice")
        assert len(results) == 1
        assert results[0] == (BUFFER_CORE, "name", "Alice")

    def test_search_case_insensitive(self):
        memory = WorkingMemory()
        memory.set(BUFFER_CORE, "name", "Alice")
        results = memory.search("alice")
        assert len(results) == 1

    def test_search_across_buffers(self):
        memory = WorkingMemory()
        memory.set(BUFFER_CORE, "user_name", "Alice")
        memory.set(BUFFER_ENVIRONMENT, "user_result", "found")
        results = memory.search("user")
        assert len(results) == 2

    def test_search_limit(self):
        memory = WorkingMemory()
        for i in range(10):
            memory.set(BUFFER_CORE, f"item_{i}", f"value_{i}")
        results = memory.search("item", limit=3)
        assert len(results) == 3

    def test_search_empty_query(self):
        memory = WorkingMemory()
        memory.set(BUFFER_CORE, "name", "Alice")
        assert memory.search("") == []

    def test_search_no_matches(self):
        memory = WorkingMemory()
        memory.set(BUFFER_CORE, "name", "Alice")
        assert memory.search("xyz") == []

    def test_search_non_string_value(self):
        memory = WorkingMemory()
        memory.set(BUFFER_CORE, "count", 42)
        results = memory.search("42")
        assert len(results) == 1

    def test_search_core_first(self):
        memory = WorkingMemory()
        memory.set(BUFFER_CORE, "core_item", "value")
        memory.set(BUFFER_ENVIRONMENT, "env_item", "value")
        results = memory.search("item")
        assert results[0][0] == BUFFER_CORE


class TestWorkingMemorySerialization:
    """Test serialization round-trip."""

    def test_to_dict(self):
        memory = WorkingMemory()
        memory.set(BUFFER_CORE, "name", "Alice")
        memory.set(BUFFER_SCRATCH, "temp", True)
        d = memory.to_dict()
        assert d[BUFFER_CORE] == {"name": "Alice"}
        assert d[BUFFER_SCRATCH] == {"temp": True}

    def test_from_dict(self):
        data = {
            "core": {"name": "Alice"},
            "custom": {"x": 1},
        }
        memory = WorkingMemory.from_dict(data)
        assert memory.get("core", "name") == "Alice"
        assert memory.get("custom", "x") == 1
        assert memory.list_buffers() == ["core", "custom"]

    def test_round_trip(self):
        memory = WorkingMemory()
        memory.set(BUFFER_CORE, "name", "Alice")
        memory.set(BUFFER_ENVIRONMENT, "result", [1, 2, 3])

        restored = WorkingMemory.from_dict(memory.to_dict())
        assert restored.get(BUFFER_CORE, "name") == "Alice"
        assert restored.get(BUFFER_ENVIRONMENT, "result") == [1, 2, 3]


class TestWorkingMemoryLen:
    """Test __len__ and __repr__."""

    def test_len(self):
        memory = WorkingMemory()
        assert len(memory) == 0
        memory.set(BUFFER_CORE, "a", 1)
        assert len(memory) == 1
        memory.set(BUFFER_ENVIRONMENT, "b", 2)
        assert len(memory) == 2

    def test_repr(self):
        memory = WorkingMemory()
        memory.set(BUFFER_CORE, "a", 1)
        r = repr(memory)
        assert "WorkingMemory" in r
        assert "core" in r


class TestFSMContextWithWorkingMemory:
    """Test WorkingMemory integration with FSMContext."""

    def test_fsm_context_default_no_memory(self):
        from fsm_llm.definitions import FSMContext

        ctx = FSMContext()
        assert ctx.working_memory is None

    def test_fsm_context_with_working_memory(self):
        from fsm_llm.definitions import FSMContext

        memory = WorkingMemory(initial_data={"name": "Alice"})
        ctx = FSMContext(working_memory=memory)
        data = ctx.get_user_visible_data()
        assert data["name"] == "Alice"

    def test_fsm_context_flat_data_overrides_memory(self):
        from fsm_llm.definitions import FSMContext

        memory = WorkingMemory(initial_data={"name": "Memory"})
        ctx = FSMContext(data={"name": "FlatDict"}, working_memory=memory)
        assert ctx.get_user_visible_data()["name"] == "FlatDict"

    def test_fsm_context_memory_data_merged(self):
        from fsm_llm.definitions import FSMContext

        memory = WorkingMemory()
        memory.set(BUFFER_CORE, "from_memory", "value1")
        ctx = FSMContext(data={"from_flat": "value2"}, working_memory=memory)
        data = ctx.get_user_visible_data()
        assert data["from_memory"] == "value1"
        assert data["from_flat"] == "value2"

    def test_fsm_context_memory_internal_keys_filtered(self):
        from fsm_llm.definitions import FSMContext

        memory = WorkingMemory()
        memory.set(BUFFER_CORE, "_internal", "hidden")
        memory.set(BUFFER_CORE, "visible", "shown")
        ctx = FSMContext(working_memory=memory)
        data = ctx.get_user_visible_data()
        assert "_internal" not in data
        assert data["visible"] == "shown"

    def test_fsm_context_without_memory_unchanged(self):
        """Verify that FSMContext without working_memory behaves exactly as before."""
        from fsm_llm.definitions import FSMContext

        ctx = FSMContext(data={"name": "Alice", "_internal": "hidden"})
        data = ctx.get_user_visible_data()
        assert data == {"name": "Alice"}
