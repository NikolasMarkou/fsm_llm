from __future__ import annotations

"""
Working memory management for FSM-LLM.

Provides structured, named memory buffers for organizing agent context
by function (core data, scratch/transient, environment/tool results,
reasoning traces) rather than a flat key-value bag.

Inspired by CoALA (Cognitive Architectures for Language Agents) and
Cognitive Workspace (CW) research on structured working memory.
"""

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
        if name not in self._buffers:
            raise KeyError(f"Buffer '{name}' does not exist")
        self._buffers[name].clear()

    def list_buffers(self) -> list[str]:
        """Return the names of all buffers."""
        return list(self._buffers.keys())

    def has_buffer(self, name: str) -> bool:
        """Check whether a buffer exists."""
        return name in self._buffers

    def create_buffer(self, name: str) -> None:
        """Create a new empty buffer.

        No-op if the buffer already exists.
        """
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
        all_data = self.get_all_data()
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

        # Search core first, then other buffers in order (skip hidden)
        search_order = []
        if BUFFER_CORE in self._buffers:
            search_order.append(BUFFER_CORE)
        for name in self._buffers:
            if name != BUFFER_CORE and name not in self._hidden_buffers:
                search_order.append(name)

        for buffer_name in search_order:
            for key, value in self._buffers[buffer_name].items():
                if len(results) >= limit:
                    break

                # Match on key name
                if query_lower in key.lower():
                    results.append((buffer_name, key, value))
                    continue

                # Match on string value
                if isinstance(value, str) and query_lower in value.lower():
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
        self.update_buffer(target_buffer, data)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """Serialize to a plain dict of dicts.

        Returns:
            Dictionary mapping buffer names to their contents.
        """
        return {name: dict(data) for name, data in self._buffers.items()}

    @classmethod
    def from_dict(
        cls,
        data: dict[str, dict[str, Any]],
        hidden_buffers: frozenset[str] | set[str] | None = None,
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

    def __len__(self) -> int:
        """Total number of keys across all buffers."""
        return sum(len(b) for b in self._buffers.values())

    def __repr__(self) -> str:
        buffer_sizes = {n: len(d) for n, d in self._buffers.items()}
        return f"WorkingMemory(buffers={buffer_sizes})"
