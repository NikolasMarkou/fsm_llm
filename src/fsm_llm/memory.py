from __future__ import annotations

"""
Working memory management for FSM-LLM.

Provides structured, named memory buffers for organizing agent context
by function (core data, scratch/transient, environment/tool results,
reasoning traces) rather than a flat key-value bag.

Inspired by CoALA (Cognitive Architectures for Language Agents) and
Cognitive Workspace (CW) research on structured working memory.
"""

import builtins
import threading
from typing import Any

from .logging import logger

# Default buffer names
BUFFER_CORE = "core"
BUFFER_SCRATCH = "scratch"
BUFFER_ENVIRONMENT = "environment"
BUFFER_REASONING = "reasoning"
BUFFER_METADATA = "metadata"

DEFAULT_BUFFERS = (BUFFER_CORE, BUFFER_SCRATCH, BUFFER_ENVIRONMENT, BUFFER_REASONING)

# Hidden buffers are excluded from aggregate views (get_all_data,
# to_scoped_view) and search.  They carry orchestration metadata
# (user permissions, retry counts, routing decisions, billing tier)
# that should flow through multi-agent pipelines without polluting
# LLM context.
DEFAULT_HIDDEN_BUFFERS = frozenset({BUFFER_METADATA})

# Types whose `str()` is a built-in that cannot re-enter `WorkingMemory`, so
# `search` may coerce them while holding the lock. Membership is tested with
# `type(value) in`, never `isinstance` — a subclass can override `__str__`.
# Containers (`list`/`dict`/`tuple`/`set`) are deliberately ABSENT: their `str()`
# recurses into elements of arbitrary type. See `search` and decisions.md D-011.
_LOCK_SAFE_STR_TYPES = frozenset(
    {int, float, bool, complex, bytes, bytearray, type(None)}
)


class WorkingMemory:
    """Structured working memory with named buffers.

    Organizes context data into functionally distinct buffers:

    - **core**: Primary conversation data (user info, extracted fields,
      accumulated facts). Persists across state transitions.
    - **scratch**: Transient data for the current processing step.
      Intended to be cleared between turns.
    - **environment**: Tool call results, API responses, file system
      state. Updated by tool execution handlers.
    - **reasoning**: Agent reasoning traces, hypotheses, plan state.
      Used by reasoning-heavy agent patterns.

    Custom buffers can be added at construction or dynamically.

    Example::

        memory = WorkingMemory()
        memory.set("core", "user_name", "Alice")
        memory.set("environment", "search_result", {"title": "..."})

        # Flat view for backward compat
        all_data = memory.get_all_data()  # {"user_name": "Alice", "search_result": {...}}

        # Scoped view for LLM prompts
        scoped = memory.to_scoped_view(["user_name"])  # {"user_name": "Alice"}

        # Search across all buffers
        results = memory.search("Alice")  # [("core", "user_name", "Alice")]
    """

    def __init__(
        self,
        buffers: tuple[str, ...] | list[str] | None = None,
        initial_data: dict[str, Any] | None = None,
        hidden_buffers: frozenset[str] | set[str] | None = None,
    ) -> None:
        """Initialize working memory with named buffers.

        Args:
            buffers: Buffer names to create. Defaults to
                (core, scratch, environment, reasoning).
            initial_data: Data to populate the core buffer with.
            hidden_buffers: Buffer names to exclude from aggregate views
                (``get_all_data``, ``to_scoped_view``) and ``search``.
                Hidden buffers carry orchestration metadata that should
                not appear in LLM prompts.  Defaults to ``{"metadata"}``.
        """
        buffer_names = buffers or DEFAULT_BUFFERS
        # DECISION plan-2026-07-19T191147-4b664252/D-007: single NON-REENTRANT lock guarding
        # every read and write of `_buffers`. Acquisitions must NEVER nest: a public
        # method that needs another's body calls the lock-free `_*_locked()` twin
        # (`_get_all_data_locked`, `_update_buffer_locked`), mirroring
        # `fsm_llm_agents/semantic_memory.py`'s `_save_locked` pattern. Do NOT
        # "simplify" this to an RLock to allow nesting — the shell/`_locked()` split
        # is what makes the lock scope auditable at each call site.
        self._lock = threading.Lock()
        self._buffers: dict[str, dict[str, Any]] = {name: {} for name in buffer_names}
        self._hidden_buffers: frozenset[str] = frozenset(
            hidden_buffers if hidden_buffers is not None else DEFAULT_HIDDEN_BUFFERS
        )

        if initial_data and BUFFER_CORE in self._buffers:
            self._buffers[BUFFER_CORE].update(initial_data)

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    def get(self, buffer: str, key: str, default: Any = None) -> Any:
        """Get a value from a specific buffer.

        Args:
            buffer: Buffer name.
            key: Key to look up.
            default: Value to return if key not found.

        Returns:
            The value, or *default* if not found.

        Raises:
            KeyError: If the buffer does not exist.
        """
        with self._lock:
            if buffer not in self._buffers:
                raise KeyError(f"Buffer '{buffer}' does not exist")
            return self._buffers[buffer].get(key, default)

    def set(self, buffer: str, key: str, value: Any) -> None:
        """Set a value in a specific buffer.

        Creates the buffer if it does not exist.

        Args:
            buffer: Buffer name.
            key: Key to store.
            value: Value to store.
        """
        with self._lock:
            if buffer not in self._buffers:
                self._buffers[buffer] = {}
                logger.debug(f"Working memory: created buffer '{buffer}'")
            self._buffers[buffer][key] = value

    def delete(self, buffer: str, key: str) -> bool:
        """Delete a key from a specific buffer.

        Args:
            buffer: Buffer name.
            key: Key to delete.

        Returns:
            True if the key existed and was deleted, False otherwise.
        """
        with self._lock:
            if buffer not in self._buffers:
                return False
            if key in self._buffers[buffer]:
                del self._buffers[buffer][key]
                return True
            return False

    def get_buffer(self, name: str) -> dict[str, Any]:
        """Get a copy of all data in a buffer.

        Args:
            name: Buffer name.

        Returns:
            Copy of the buffer contents.

        Raises:
            KeyError: If the buffer does not exist.
        """
        with self._lock:
            if name not in self._buffers:
                raise KeyError(f"Buffer '{name}' does not exist")
            return dict(self._buffers[name])

    def clear_buffer(self, name: str) -> None:
        """Clear all data from a buffer without removing it.

        Args:
            name: Buffer name.

        Raises:
            KeyError: If the buffer does not exist.
        """
        with self._lock:
            if name not in self._buffers:
                raise KeyError(f"Buffer '{name}' does not exist")
            self._buffers[name].clear()

    def list_buffers(self) -> list[str]:
        """Return the names of all buffers."""
        with self._lock:
            return list(self._buffers.keys())

    def has_buffer(self, name: str) -> bool:
        """Check whether a buffer exists."""
        with self._lock:
            return name in self._buffers

    def create_buffer(self, name: str) -> None:
        """Create a new empty buffer.

        No-op if the buffer already exists.
        """
        with self._lock:
            if name not in self._buffers:
                self._buffers[name] = {}

    # ------------------------------------------------------------------
    # Aggregate views
    # ------------------------------------------------------------------

    def get_all_data(self) -> dict[str, Any]:
        """Flatten all buffers into a single dict for backward compat.

        When the same key exists in multiple buffers, the **core**
        buffer wins, followed by buffers in creation order.

        Returns:
            Merged dictionary of all buffer contents.
        """
        with self._lock:
            return self._get_all_data_locked()

    def _get_all_data_locked(self) -> dict[str, Any]:
        """Lock-free body of :meth:`get_all_data`.

        Caller MUST already hold ``self._lock``. Exists so :meth:`to_scoped_view`
        can reuse the merge without re-acquiring the non-reentrant lock.
        """
        merged: dict[str, Any] = {}
        shadowed: list[str] = []

        # Merge non-core buffers first (in order), skipping hidden buffers
        for name, data in self._buffers.items():
            if name == BUFFER_CORE or name in self._hidden_buffers:
                continue
            for key, value in data.items():
                if key in merged:
                    shadowed.append(f"{key} (in {name})")
                merged[key] = value

        # Core buffer wins on collision
        core = self._buffers.get(BUFFER_CORE, {})
        for key, value in core.items():
            if key in merged:
                shadowed.append(f"{key} (core overrides)")
            merged[key] = value

        if shadowed:
            logger.debug(f"Working memory: shadowed keys during flatten: {shadowed}")

        return merged

    def to_scoped_view(self, read_keys: list[str]) -> dict[str, Any]:
        """Return a subset of all data matching the given keys.

        Searches across all buffers. If a key exists in multiple
        buffers, core wins (same as ``get_all_data``).

        Args:
            read_keys: Keys to include in the view.

        Returns:
            Dict containing only the requested keys that exist.
        """
        with self._lock:
            all_data = self._get_all_data_locked()
        return {key: all_data[key] for key in read_keys if key in all_data}

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, limit: int = 20) -> list[tuple[str, str, Any]]:
        """Search across all buffers for keys or values matching a query.

        Performs case-insensitive substring matching on:
        - Key names
        - String values
        - String representations of non-string values

        Args:
            query: Search string.
            limit: Maximum number of results to return.

        Returns:
            List of (buffer_name, key, value) tuples, ordered by
            buffer priority (core first).
        """
        if not query:
            return []

        query_lower = query.lower()
        results: list[tuple[str, str, Any]] = []

        # DECISION plan-2026-07-20T040150-876e7164/D-011
        # `str(value)` must NOT run while `self._lock` is held. It calls
        # arbitrary third-party `__str__`, and an object that re-enters this
        # same WorkingMemory from its `__str__` (e.g. an agent object that
        # summarises the memory it holds a reference to) DEADLOCKS on the
        # non-reentrant lock — measured as a HANG, not an exception, so nothing
        # reports it. Values whose coercion cannot be decided under the lock are
        # collected as `already_matched=False` and resolved in pass 2, after the
        # release. Do NOT move that coercion back inside the `with` block.
        #
        # Do NOT "fix" the self-deadlock by making `_lock` an `RLock` either.
        # D-007 chose a non-reentrant lock deliberately so lock scope stays
        # answerable by inspection; the sanctioned escape when a critical
        # section needs another method's body is the shell/`_locked()` split
        # (`_get_all_data_locked`, `_update_buffer_locked`). Here even that
        # would not help — the callee is not ours, so the only correct answer is
        # to not be holding the lock when it runs.
        #
        # Do NOT go further and snapshot every entry to match entirely outside
        # the lock. That is the obvious-looking shape and it is WRONG: holding
        # the lock across the scan is what backpressures a concurrent writer,
        # and dropping it makes the deferred pass a lock-free window
        # proportional to the number of deferred values, which a hot writer
        # exploits. Measured on
        # `test_concurrent_set_during_search_and_get_all_data`: the full-snapshot
        # shape went superlinear (807k-1.25M keys after 20 reads vs. 82k for the
        # pre-hoist code) and never finished. `_LOCK_SAFE_STR_TYPES` is what
        # keeps that window near-zero for ordinary payloads.
        #
        # The early cutoff counts DECIDED matches only, so the scanned range is
        # a superset of the pre-hoist one and pass 2 re-applies the true `limit`
        # in scan order — the returned list is identical, entry for entry.
        # See decisions.md D-011.
        with self._lock:
            # Search core first, then other buffers in order (skip hidden)
            search_order = []
            if BUFFER_CORE in self._buffers:
                search_order.append(BUFFER_CORE)
            for name in self._buffers:
                if name != BUFFER_CORE and name not in self._hidden_buffers:
                    search_order.append(name)

            # (buffer, key, value, already_matched); `False` means "undecided,
            # needs `str(value)`", which must not happen under the lock.
            pending: list[tuple[str, str, Any, bool]] = []
            decided = 0

            for buffer_name in search_order:
                if decided >= limit:
                    break
                for key, value in self._buffers[buffer_name].items():
                    if decided >= limit:
                        break

                    # Match on key name
                    if query_lower in key.lower():
                        pending.append((buffer_name, key, value, True))
                        decided += 1
                        continue

                    # Match on string value
                    if isinstance(value, str):
                        if query_lower in value.lower():
                            pending.append((buffer_name, key, value, True))
                            decided += 1
                        continue

                    # A built-in scalar's `str()` cannot run third-party code,
                    # so it is decided here and never enlarges the lock-free
                    # window below. EXACT type, not `isinstance`: a subclass may
                    # override `__str__`, which is precisely the hole.
                    if type(value) in _LOCK_SAFE_STR_TYPES:
                        if query_lower in str(value).lower():
                            pending.append((buffer_name, key, value, True))
                            decided += 1
                        continue

                    # Anything else: deciding it needs a `str()` that may call
                    # back into this instance. Defer past the release.
                    pending.append((buffer_name, key, value, False))

        for buffer_name, key, value, already_matched in pending:
            if len(results) >= limit:
                break

            if already_matched:
                results.append((buffer_name, key, value))
                continue

            # Match on string representation of non-string values
            try:
                value_str = str(value)
                if query_lower in value_str.lower():
                    results.append((buffer_name, key, value))
            except Exception:
                pass

        return results[:limit]

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def update_buffer(self, buffer: str, data: dict[str, Any]) -> None:
        """Merge data into a buffer.

        Creates the buffer if it does not exist.

        Args:
            buffer: Buffer name.
            data: Key-value pairs to merge.
        """
        with self._lock:
            self._update_buffer_locked(buffer, data)

    def _update_buffer_locked(self, buffer: str, data: dict[str, Any]) -> None:
        """Lock-free body of :meth:`update_buffer`.

        Caller MUST already hold ``self._lock``. Exists so :meth:`import_flat_data`
        can reuse the merge without re-acquiring the non-reentrant lock.
        """
        if buffer not in self._buffers:
            self._buffers[buffer] = {}
        self._buffers[buffer].update(data)

    def import_flat_data(
        self, data: dict[str, Any], target_buffer: str = BUFFER_CORE
    ) -> None:
        """Import a flat dict into a buffer.

        Useful for migrating existing FSMContext.data into structured memory.

        Args:
            data: Flat dictionary to import.
            target_buffer: Which buffer to import into.
        """
        with self._lock:
            self._update_buffer_locked(target_buffer, data)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """Serialize to a plain dict of dicts.

        Returns:
            Dictionary mapping buffer names to their contents.
        """
        with self._lock:
            return {name: dict(data) for name, data in self._buffers.items()}

    @classmethod
    def from_dict(
        cls,
        data: dict[str, dict[str, Any]],
        # `builtins.set` disambiguates from `WorkingMemory.set` (the method), which
        # mypy otherwise resolves in classmethod scope under `from __future__
        # import annotations`. Same type as before — annotation-only.
        hidden_buffers: frozenset[str] | builtins.set[str] | None = None,
    ) -> WorkingMemory:
        """Deserialize from a plain dict of dicts.

        Args:
            data: Dictionary mapping buffer names to their contents.
            hidden_buffers: Buffer names to exclude from aggregate views.
                Defaults to ``DEFAULT_HIDDEN_BUFFERS``.

        Returns:
            New WorkingMemory instance.
        """
        buffer_names = list(data.keys())
        memory = cls(buffers=buffer_names, hidden_buffers=hidden_buffers)
        for name, contents in data.items():
            memory._buffers[name] = dict(contents)
        return memory

    # DECISION plan-2026-07-19T191147-4b664252/D-018
    # `threading.Lock` is not picklable, so introducing `self._lock` in D-007
    # silently made `WorkingMemory` un-`deepcopy`-able and un-`pickle`-able
    # (`TypeError: cannot pickle '_thread.lock' object`). Nothing in-tree hit it
    # -- the live sites copy `context.data`, not `context` -- but
    # `FSMContext.working_memory` is a sibling of `data`, so any caller doing
    # `deepcopy(instance.context)`, `FSMContext.model_copy(deep=True)`, or
    # putting a `WorkingMemory` into `context.data` would have broken.
    # These two hooks serve `copy.copy`, `copy.deepcopy` AND `pickle` at once,
    # which is why they are preferred over a bare `__deepcopy__`: a
    # `__deepcopy__` alone would have left `pickle` (hence session persistence)
    # still failing. The lock is EXCLUDED from the state and rebuilt fresh in
    # `__setstate__` -- a copy must never share the original's lock, or two
    # independent objects would serialize against each other. Do NOT add
    # `_lock` back into the returned state.
    # See decisions.md D-018.
    #
    # DECISION plan-2026-07-19T191147-4b664252/D-027 (CORRECTS D-018)
    # D-018's first version held the lock but returned LIVE references:
    #     with self._lock:
    #         return {"_buffers": self._buffers, ...}
    # Holding the lock there buys NOTHING. `__getstate__` returns before the
    # copier walks the dict, so the lock is already released by the time
    # `deepcopy`/`pickle` iterates -- and a concurrent `set()` then raises
    # `RuntimeError: dictionary changed size during iteration`. That is SC-7's
    # exact failure class, reintroduced inside the fix for a different concern
    # (measured: 48/1200 deepcopies raised, and 1200/1200 under the reviewer's
    # heavier writer). The copy must therefore be taken INSIDE the lock, not
    # merely started inside it.
    # Second defect, same cause: because `_buffers` was a live reference,
    # `copy.copy` produced an object SHARING the buffers while `__setstate__`
    # gave it a DIFFERENT lock -- two objects mutating one dict under two locks,
    # i.e. no mutual exclusion at all, which falsified D-018's own "shallow-copy
    # semantics are unchanged" claim.
    # RESOLUTION -- `copy.copy` is defined as a SNAPSHOT, and the docstrings say
    # so. The buffer MAPPINGS are rebuilt (so no writer can mutate a copy's
    # structure, and no copy can mutate the original's); the buffered VALUES are
    # still shared by reference, which is what makes this a shallow copy rather
    # than a deep one. `deepcopy` layers its own value-copying on top of this
    # snapshot and is therefore safe by construction.
    # The alternative -- sharing the lock alongside the buffers so `copy.copy`
    # stays a true alias -- was rejected: `__getstate__` also serves `pickle`,
    # where a shared lock is impossible, so that route would need two divergent
    # code paths and would still leave `deepcopy` racing.
    # Do NOT "optimise" the dict rebuild below back into a bare reference.
    # See decisions.md D-027.
    def __getstate__(self) -> dict[str, Any]:
        """Return a picklable SNAPSHOT: everything except the unpicklable lock.

        The snapshot is materialised while the lock is held, so a concurrent
        writer can neither corrupt the copier's iteration nor be observed
        half-applied. Buffered values are shared by reference; only the
        mappings are rebuilt.
        """
        with self._lock:
            return {
                "_buffers": {
                    name: dict(contents) for name, contents in self._buffers.items()
                },
                "_hidden_buffers": frozenset(self._hidden_buffers),
            }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore state and give the copy its OWN fresh lock."""
        self._buffers = state["_buffers"]
        # DECISION plan-2026-07-19T191147-4b664252/D-032
        # `frozenset(...)`, NOT a bare assignment, and NOT `set(...)`.
        # `_hidden_buffers` is declared `frozenset[str]` (see the attribute above)
        # because it gates `get_all_data`/`to_scoped_view`/`search` -- i.e. what
        # reaches an LLM prompt -- so its immutability is a deliberate control,
        # not an incidental choice. D-027 rebuilt it as a mutable `set` while
        # fixing the live-reference race, and EVERY copy/deepcopy/pickle then
        # silently downgraded the type: a copy's prompt-visibility set became
        # mutable in place. mypy cannot see this, because `state` is `dict[str,
        # Any]`; only an explicit type assertion in the copy-semantics tests can.
        # The re-wrap here (rather than trusting `__getstate__` alone) also
        # normalises pickles written by the D-027 build, which carry a `set`.
        # See decisions.md D-032.
        self._hidden_buffers = frozenset(state["_hidden_buffers"])
        self._lock = threading.Lock()

    def __len__(self) -> int:
        """Total number of keys across all buffers."""
        with self._lock:
            return sum(len(b) for b in self._buffers.values())

    def __repr__(self) -> str:
        with self._lock:
            buffer_sizes = {n: len(d) for n, d in self._buffers.items()}
        return f"WorkingMemory(buffers={buffer_sizes})"
