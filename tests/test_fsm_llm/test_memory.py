"""Tests for WorkingMemory structured buffer system."""

import threading
from typing import Any

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


class TestWorkingMemoryConcurrency:
    """F-05: buffer mutation and iteration must be mutually exclusive.

    Pre-fix, ``search()``/``get_all_data()`` iterated ``self._buffers[...]``
    with no lock, so a concurrent ``set()`` raised
    ``RuntimeError: dictionary changed size during iteration``
    in 199-200 of 200 reads.
    """

    def test_concurrent_set_during_search_and_get_all_data(self):
        """SC-7: 200 concurrent reads under a concurrent writer produce 0 raises."""
        memory = WorkingMemory()
        # A large seeded buffer plus a non-matching query forces `search` to scan
        # every key, widening the iteration window the writer must race into.
        # A small buffer (or an early `limit` cutoff) makes the race rare enough
        # that the test would pass against un-fixed code.
        for i in range(2000):
            memory.set(BUFFER_CORE, f"seed_{i}", f"value_{i}")

        stop = threading.Event()
        writes = 0
        raises: list[str] = []

        def writer():
            nonlocal writes
            while not stop.is_set():
                memory.set(BUFFER_CORE, f"k_{writes}", writes)
                writes += 1

        def reader():
            for n in range(200):
                try:
                    if n % 2 == 0:
                        memory.search("zzz-no-match")
                    else:
                        memory.get_all_data()
                except Exception as e:  # the defect under test
                    raises.append(f"{type(e).__name__}: {e}")

        writer_thread = threading.Thread(target=writer, daemon=True)
        reader_thread = threading.Thread(target=reader)
        writer_thread.start()
        reader_thread.start()
        reader_thread.join(timeout=60)
        stop.set()
        writer_thread.join(timeout=10)

        assert not reader_thread.is_alive(), "reader thread deadlocked"
        # Vacuity guard: a writer that never ran would make the 0-raises
        # assertion meaningless.
        assert writes > 0, "writer never mutated the buffer; probe was vacuous"
        assert raises == [], f"{len(raises)}/200 reads raised; first: {raises[0]}"

    def test_public_methods_do_not_self_deadlock(self):
        """The lock is non-reentrant: no public method may nest an acquisition."""
        memory = WorkingMemory(initial_data={"name": "Alice"})
        done = threading.Event()
        error: list[BaseException] = []

        def exercise():
            try:
                memory.set(BUFFER_CORE, "k", "v")
                memory.get(BUFFER_CORE, "k")
                memory.get_buffer(BUFFER_CORE)
                memory.list_buffers()
                memory.has_buffer(BUFFER_CORE)
                memory.create_buffer("extra")
                memory.update_buffer("extra", {"a": 1})
                memory.import_flat_data({"b": 2})
                memory.get_all_data()
                memory.to_scoped_view(["name", "a", "b"])
                memory.search("Alice")
                memory.to_dict()
                len(memory)
                repr(memory)
                memory.delete(BUFFER_CORE, "k")
                memory.clear_buffer("extra")
            except BaseException as e:  # reported to the main thread
                error.append(e)
            finally:
                done.set()

        thread = threading.Thread(target=exercise, daemon=True)
        thread.start()
        assert done.wait(timeout=10), "a public method self-deadlocked on the lock"
        assert error == [], f"public method raised: {error[0]!r}"

    def test_search_survives_a_value_whose_str_reenters_the_same_memory(self):
        """SC-15: `str(value)` runs OUTSIDE the lock, so a re-entrant `__str__`
        cannot deadlock `search` (D-011).

        The pre-fix failure mode is a HANG, not an exception — probed at a 5s
        timeout with the thread still alive. So this runs on a worker with a
        hard `join(timeout=...)` watchdog: without it, a regression would wedge
        the whole suite instead of failing this one test.
        """
        memory = WorkingMemory()

        class ReentrantStr:
            """A third-party-shaped value that summarises the memory holding it."""

            def __init__(self, owner: WorkingMemory) -> None:
                self._owner = owner

            def __str__(self) -> str:
                # Re-enters the SAME instance from inside `search`'s coercion.
                return f"summary of {self._owner.get(BUFFER_CORE, 'plain')}"

        memory.set(BUFFER_CORE, "plain", "alpha")
        memory.set(BUFFER_CORE, "reentrant", ReentrantStr(memory))

        done = threading.Event()
        outcome: list[Any] = []
        error: list[BaseException] = []

        def exercise():
            try:
                outcome.append(memory.search("summary"))
            except BaseException as e:  # reported to the main thread
                error.append(e)
            finally:
                done.set()

        thread = threading.Thread(target=exercise, daemon=True)
        thread.start()

        assert done.wait(timeout=10), (
            "search() deadlocked: str(value) re-entered the non-reentrant lock"
        )
        assert error == [], f"search raised: {error[0]!r}"
        # Not merely "did not hang" — the re-entrant value must actually MATCH,
        # or a fix that skipped coercion entirely would pass the watchdog.
        assert [(b, k) for b, k, _ in outcome[0]] == [(BUFFER_CORE, "reentrant")]

    def test_int_subclass_with_a_reentrant_str_is_still_deferred(self):
        """`_LOCK_SAFE_STR_TYPES` is matched by EXACT type, not `isinstance`.

        An `int` subclass that overrides `__str__` is the hole an `isinstance`
        check would leave open — it would be coerced under the lock and deadlock
        exactly like the plain object case.
        """
        memory = WorkingMemory()

        class SneakyInt(int):
            def __str__(self) -> str:
                return f"sneaky {memory.get(BUFFER_CORE, 'plain')}"

        memory.set(BUFFER_CORE, "plain", "alpha")
        # The key must NOT contain the query, or the key-name match short-
        # circuits before any coercion and the test is vacuous.
        memory.set(BUFFER_CORE, "wrapped", SneakyInt(7))

        done = threading.Event()
        outcome: list[Any] = []

        def exercise():
            try:
                outcome.append(memory.search("sneaky"))
            finally:
                done.set()

        thread = threading.Thread(target=exercise, daemon=True)
        thread.start()

        assert done.wait(timeout=10), (
            "an int subclass with a re-entrant __str__ was coerced under the lock"
        )
        assert [(b, k) for b, k, _ in outcome[0]] == [(BUFFER_CORE, "wrapped")]

    def test_str_subclass_value_with_a_reentrant_lower_is_still_deferred(self):
        """D-016. The `isinstance(value, str)` arm was the SAME hole one line
        above the frozenset written to close it: D-011 stated the rule
        ("EXACT type, not `isinstance`") and then did not apply it to the arm
        directly above. `str.lower` is overridable, so a `str` subclass ran
        third-party code under `self._lock` and deadlocked — demonstrated by
        adversarial review, which hung the full 5s watchdog.

        Note the value must be a `str` SUBCLASS, not a plain object: that is
        the whole point. `test_search_survives_...` covers the non-`str` path
        and passed throughout, which is why this hole survived.
        """
        memory = WorkingMemory()

        class ReentrantStr(str):
            def lower(self) -> str:
                # Re-enters the SAME instance from inside the locked scan.
                memory.get(BUFFER_CORE, "plain")
                return str.lower(self)

        memory.set(BUFFER_CORE, "plain", "alpha")
        # The key must NOT contain the query, or the key-name match
        # short-circuits before `.lower()` is ever reached and this is vacuous.
        memory.set(BUFFER_CORE, "wrapped", ReentrantStr("sneaky payload"))

        done = threading.Event()
        outcome: list[Any] = []

        def exercise():
            try:
                outcome.append(memory.search("sneaky"))
            finally:
                done.set()

        thread = threading.Thread(target=exercise, daemon=True)
        thread.start()

        assert done.wait(timeout=10), (
            "search() deadlocked: a str subclass's .lower() ran under the lock"
        )
        # Not merely "did not hang": the deferred value must still MATCH, or a
        # fix that simply skipped `str` subclasses would pass the watchdog.
        assert [(b, k) for b, k, _ in outcome[0]] == [(BUFFER_CORE, "wrapped")]

    def test_str_subclass_key_with_a_reentrant_lower_is_not_called(self):
        """D-016, the sibling hole. `key.lower()` on the key-name arm is
        reachable by a `str`-subclass KEY for exactly the same reason, so it
        goes through the unbound `str.lower`, which a subclass cannot override.
        """
        memory = WorkingMemory()

        class ReentrantKey(str):
            def lower(self) -> str:  # pragma: no cover - must never be called
                memory.get(BUFFER_CORE, "plain")
                return str.lower(self)

        memory.set(BUFFER_CORE, "plain", "alpha")
        memory.set(BUFFER_CORE, ReentrantKey("sneaky_key"), "value")

        done = threading.Event()
        outcome: list[Any] = []

        def exercise():
            try:
                outcome.append(memory.search("sneaky"))
            finally:
                done.set()

        thread = threading.Thread(target=exercise, daemon=True)
        thread.start()

        assert done.wait(timeout=10), (
            "search() deadlocked: a str subclass KEY's .lower() ran under the lock"
        )
        assert [(b, k) for b, k, _ in outcome[0]] == [(BUFFER_CORE, "sneaky_key")]

    def test_lock_is_not_reentrant(self):
        """I-6: the D-007 non-reentrant lock must not be "simplified" to an
        `RLock` to make a future self-deadlock go away. Asserted on the object,
        not by grepping the file, so a re-export or alias cannot dodge it."""
        memory = WorkingMemory()

        assert type(memory._lock) is type(threading.Lock()), (
            f"WorkingMemory._lock is {type(memory._lock).__name__}, not a plain Lock"
        )
        assert not isinstance(memory._lock, type(threading.RLock()))

    def test_search_result_order_and_limit_are_unchanged_by_the_hoist(self):
        """The snapshot must not reorder results or change which entries the
        `limit` cutoff considers — buffer order, then insertion order."""
        memory = WorkingMemory()
        memory.set(BUFFER_SCRATCH, "s_match_1", "hit")
        memory.set(BUFFER_SCRATCH, "s_match_2", "hit")
        memory.set(BUFFER_CORE, "c_match_1", "hit")
        memory.set(BUFFER_CORE, "c_match_2", 12345)  # matches via str() coercion

        assert [(b, k) for b, k, _ in memory.search("hit")] == [
            (BUFFER_CORE, "c_match_1"),
            (BUFFER_SCRATCH, "s_match_1"),
            (BUFFER_SCRATCH, "s_match_2"),
        ]
        # Core is searched first, so the limit cuts the scratch matches.
        assert [(b, k) for b, k, _ in memory.search("hit", limit=1)] == [
            (BUFFER_CORE, "c_match_1")
        ]
        # Coercion-only match still reached outside the lock.
        assert [(b, k) for b, k, _ in memory.search("2345")] == [
            (BUFFER_CORE, "c_match_2")
        ]

    def test_import_flat_data_still_merges_after_lock_split(self):
        """`import_flat_data` routes through `_update_buffer_locked`, not `update_buffer`."""
        memory = WorkingMemory()
        memory.set(BUFFER_CORE, "existing", 1)
        memory.import_flat_data({"imported": 2})
        assert memory.get(BUFFER_CORE, "existing") == 1
        assert memory.get(BUFFER_CORE, "imported") == 2

    def test_to_scoped_view_still_reads_through_locked_body(self):
        """`to_scoped_view` reuses `_get_all_data_locked`, preserving core-wins."""
        memory = WorkingMemory()
        memory.set(BUFFER_SCRATCH, "shared", "scratch")
        memory.set(BUFFER_CORE, "shared", "core")
        assert memory.to_scoped_view(["shared"]) == {"shared": "core"}


class TestWorkingMemoryCopySemantics:
    """D-018: the D-007 lock made WorkingMemory un-deepcopy-able/un-picklable."""

    @pytest.mark.parametrize("how", ["copy", "deepcopy", "pickle"])
    def test_hidden_buffers_stays_a_frozenset_across_every_copy_path(self, how):
        """D-032. `_hidden_buffers` is declared `frozenset[str]` because it gates
        what reaches an LLM prompt, so its immutability is a control rather than
        an incidental choice. D-027 rebuilt it as a mutable `set` inside
        `__getstate__` while fixing a different defect, and every copy path then
        silently downgraded the type -- a 138-line test addition specifically
        about copy semantics could not see it, because nothing asserted the TYPE.
        mypy cannot see it either: `__setstate__` reads from a `dict[str, Any]`.
        """
        import copy as copy_mod
        import pickle

        memory = WorkingMemory()
        memory.set(BUFFER_CORE, "k", "v")
        assert memory._hidden_buffers, (
            "vacuity guard: an empty hidden set makes the mutation assertion "
            "below meaningless"
        )

        clone = {
            "copy": lambda: copy_mod.copy(memory),
            "deepcopy": lambda: copy_mod.deepcopy(memory),
            "pickle": lambda: pickle.loads(pickle.dumps(memory)),
        }[how]()

        assert type(clone._hidden_buffers) is frozenset, (
            f"{how} downgraded _hidden_buffers to "
            f"{type(clone._hidden_buffers).__name__}; the declared type is "
            "frozenset[str] and it gates prompt visibility"
        )
        with pytest.raises(AttributeError):
            clone._hidden_buffers.add("injected")

    def test_a_legacy_pickle_carrying_a_mutable_set_is_normalised(self):
        """Over-correction guard AND a real compatibility case: pickles written
        by the D-027 build carry a plain `set`. `__setstate__` must re-wrap
        rather than trust `__getstate__`, or those restore as mutable."""
        memory = WorkingMemory()
        memory.__setstate__({"_buffers": {}, "_hidden_buffers": {"metadata"}})

        assert type(memory._hidden_buffers) is frozenset
        assert memory._hidden_buffers == frozenset({"metadata"})

    def test_deepcopy_succeeds_and_is_independent(self):
        import copy

        memory = WorkingMemory()
        memory.set(BUFFER_CORE, "nested", {"n": 1})
        clone = copy.deepcopy(memory)
        clone.set(BUFFER_CORE, "nested", {"n": 2})

        assert memory.get(BUFFER_CORE, "nested") == {"n": 1}
        assert clone.get(BUFFER_CORE, "nested") == {"n": 2}

    def test_deepcopy_gets_its_own_lock(self):
        """Sharing the original's lock would serialize two independent objects."""
        import copy

        memory = WorkingMemory()
        clone = copy.deepcopy(memory)
        assert clone._lock is not memory._lock

    def test_pickle_roundtrip_preserves_buffers_and_hidden_set(self):
        import pickle

        memory = WorkingMemory(hidden_buffers={"metadata", "audit"})
        memory.set(BUFFER_CORE, "a", 1)
        restored = pickle.loads(pickle.dumps(memory))

        assert restored.get(BUFFER_CORE, "a") == 1
        assert restored._hidden_buffers == memory._hidden_buffers

    def test_restored_instance_is_still_usable(self):
        """A rebuilt lock must actually work, not just exist."""
        import copy

        memory = WorkingMemory()
        memory.set(BUFFER_CORE, "a", 1)
        clone = copy.deepcopy(memory)
        clone.set(BUFFER_SCRATCH, "b", 2)
        assert len(clone) == 2
        assert clone.get_all_data() == {"a": 1, "b": 2}

    def test_deepcopy_of_fsm_context_carrying_working_memory(self):
        """The real reachability path: WorkingMemory is a sibling of `data`."""
        import copy

        from fsm_llm.definitions import FSMContext

        context = FSMContext(working_memory=WorkingMemory())
        context.working_memory.set(BUFFER_CORE, "a", 1)
        clone = copy.deepcopy(context)
        assert clone.working_memory.get(BUFFER_CORE, "a") == 1


class TestWorkingMemoryCopyUnderConcurrency:
    """D-027: D-018's first `__getstate__` held the lock but returned a LIVE
    `_buffers` reference, so the copier walked the dict AFTER the lock was
    released. That reintroduced SC-7's exact failure class
    (``RuntimeError: dictionary changed size during iteration``) on the code path
    D-018 had just created, and it also made `copy.copy` share `_buffers` while
    holding a DIFFERENT lock -- two objects mutating one dict under two locks.
    The five tests in `TestWorkingMemoryCopySemantics` all passed throughout:
    none of them exercised concurrency, and none exercised `copy.copy` at all.
    """

    def test_concurrent_set_during_deepcopy_produces_no_raises(self):
        """Mirrors SC-7's shape and scale: a large seeded buffer widens the
        iteration window so the race is reliable, not incidental. Against the
        live-reference `__getstate__` this measured 48/1200 raises here and
        1200/1200 under the reviewer's heavier writer; either way, non-zero."""
        import copy

        memory = WorkingMemory()
        # A small buffer makes the copy window too short to race into, which
        # would let this test pass against the un-fixed code.
        for i in range(2000):
            memory.set(BUFFER_CORE, f"seed_{i}", f"value_{i}")

        stop = threading.Event()
        writes = 0
        raises: list[str] = []

        def writer():
            # set-then-delete keeps the buffer SIZE stable. An append-only
            # writer would grow it without bound, and since each copy walks the
            # whole buffer the run time would go quadratic.
            nonlocal writes
            while not stop.is_set():
                memory.set(BUFFER_CORE, f"k_{writes}", writes)
                memory.delete(BUFFER_CORE, f"k_{writes}")
                writes += 1

        def copier():
            for _ in range(200):
                try:
                    copy.deepcopy(memory)
                except Exception as e:  # the defect under test
                    raises.append(f"{type(e).__name__}: {e}")

        writer_thread = threading.Thread(target=writer, daemon=True)
        copier_thread = threading.Thread(target=copier)
        writer_thread.start()
        copier_thread.start()
        copier_thread.join(timeout=120)
        stop.set()
        writer_thread.join(timeout=10)

        assert not copier_thread.is_alive(), "copier thread deadlocked"
        # Vacuity guard: a writer that never ran makes 0-raises meaningless.
        assert writes > 0, "writer never mutated the buffer; probe was vacuous"
        assert raises == [], f"{len(raises)}/200 deepcopies raised; first: {raises[0]}"

    def test_concurrent_set_during_pickle_produces_no_raises(self):
        """`pickle` goes through the same `__getstate__`, and it is the path
        session persistence uses -- so it needs its own pin, not an inference."""
        import pickle

        memory = WorkingMemory()
        for i in range(2000):
            memory.set(BUFFER_CORE, f"seed_{i}", f"value_{i}")

        stop = threading.Event()
        writes = 0
        raises: list[str] = []

        def writer():
            # set-then-delete: see the deepcopy test above.
            nonlocal writes
            while not stop.is_set():
                memory.set(BUFFER_CORE, f"k_{writes}", writes)
                memory.delete(BUFFER_CORE, f"k_{writes}")
                writes += 1

        def dumper():
            for _ in range(200):
                try:
                    pickle.dumps(memory)
                except Exception as e:
                    raises.append(f"{type(e).__name__}: {e}")

        writer_thread = threading.Thread(target=writer, daemon=True)
        dumper_thread = threading.Thread(target=dumper)
        writer_thread.start()
        dumper_thread.start()
        dumper_thread.join(timeout=120)
        stop.set()
        writer_thread.join(timeout=10)

        assert not dumper_thread.is_alive(), "dumper thread deadlocked"
        assert writes > 0, "writer never mutated the buffer; probe was vacuous"
        assert raises == [], f"{len(raises)}/200 pickles raised; first: {raises[0]}"

    def test_shallow_copy_is_a_snapshot_not_a_shared_alias(self):
        """D-027 defines `copy.copy` as a SNAPSHOT. This is the assertion that
        makes the docstring and the behaviour agree -- D-018 claimed
        "shallow-copy semantics are unchanged" while shipping a shared-buffer,
        separate-lock object, and no test contradicted it because no test
        touched `copy.copy`."""
        import copy

        memory = WorkingMemory()
        memory.set(BUFFER_CORE, "a", 1)
        clone = copy.copy(memory)

        # The MUTABLE mappings must be distinct objects...
        assert clone._buffers is not memory._buffers
        assert clone._buffers[BUFFER_CORE] is not memory._buffers[BUFFER_CORE]
        # ...but `_hidden_buffers` is deliberately NOT asserted to be a distinct
        # object (D-032). It is a `frozenset`, so `frozenset(x)` returns `x`
        # itself and sharing it is safe by construction -- there is no writer.
        # The original form of this line demanded non-identity, which is only
        # meaningful for a MUTABLE container; satisfying it literally is what
        # pushed D-027 into rebuilding the frozenset as a `set` and silently
        # downgrading a declared-immutable prompt-visibility control on every
        # copy. The invariant that actually matters -- the copy's hidden set
        # cannot be mutated at all, so it can never diverge from or corrupt the
        # original's -- is stronger, and is pinned by
        # `test_hidden_buffers_stays_a_frozenset_across_every_copy_path`.
        assert clone._hidden_buffers == memory._hidden_buffers
        with pytest.raises(AttributeError):
            clone._hidden_buffers.add("injected")
        # ...and the lock must be distinct too. Shared STATE plus separate LOCKS
        # is the combination that removes mutual exclusion outright.
        assert clone._lock is not memory._lock

        # Structural writes on one side must not be visible on the other.
        clone.set(BUFFER_CORE, "b", 2)
        assert memory.get(BUFFER_CORE, "b") is None
        memory.set(BUFFER_CORE, "c", 3)
        assert clone.get(BUFFER_CORE, "c") is None

    def test_shallow_copy_still_shares_values_by_reference(self):
        """The other half of "shallow": the snapshot rebuilds the MAPPINGS but
        must not deep-copy the VALUES, or `copy.copy` would silently become
        `copy.deepcopy` and cost O(payload) on every call."""
        import copy

        payload = {"n": 1}
        memory = WorkingMemory()
        memory.set(BUFFER_CORE, "nested", payload)
        clone = copy.copy(memory)

        assert clone.get(BUFFER_CORE, "nested") is payload
