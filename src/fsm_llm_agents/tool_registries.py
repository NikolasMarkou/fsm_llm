from __future__ import annotations

"""
Drop-in :class:`ToolRegistry` subclasses that add cross-cutting execution
behavior (result caching, retry-on-failure) without changing the dispatch
contract.

Both classes are 100% additive: they override only :meth:`ToolRegistry.execute`,
preserve its ``ToolCall -> ToolResult`` signature, and inherit every other
method (registration, schema generation, prompt description). Any agent that
accepts a ``ToolRegistry`` accepts these unchanged.

Example::

    from fsm_llm_agents import ReactAgent, AgentConfig
    from fsm_llm_agents.tool_registries import CachingToolRegistry

    registry = CachingToolRegistry()        # behaves like ToolRegistry
    registry.register(search._tool_definition)
    agent = ReactAgent(tools=registry, config=AgentConfig(model=model))
"""

import threading
import time
from typing import Any

from fsm_llm.logging import logger

from .definitions import ToolCall, ToolResult
from .tools import ToolRegistry


def _freeze(value: Any) -> Any:
    """Return a hashable, order-stable representation of ``value``."""
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze(v) for v in value))
    return value


class CachingToolRegistry(ToolRegistry):
    """A :class:`ToolRegistry` that memoizes successful tool results.

    Identical ``(tool_name, parameters)`` calls return the cached
    :class:`ToolResult` instead of re-invoking the tool. Only *successful*
    results are cached — failures always re-execute so transient errors are
    retryable. This is a pure latency/cost optimization for idempotent tools
    (search, lookups); do NOT use it for tools with side effects whose result
    depends on external mutable state.

    Args:
        max_entries: Upper bound on cached entries (insertion-ordered eviction).
            ``None`` means unbounded.
    """

    def __init__(self, max_entries: int | None = 256) -> None:
        super().__init__()
        self._cache: dict[tuple[str, Any], ToolResult] = {}
        self._cache_lock = threading.Lock()
        self._max_entries = max_entries
        self.cache_hits = 0
        self.cache_misses = 0

    def _cache_key(self, tool_call: ToolCall) -> tuple[str, Any]:
        return (tool_call.tool_name, _freeze(tool_call.parameters))

    def execute(self, tool_call: ToolCall) -> ToolResult:
        key = self._cache_key(tool_call)
        with self._cache_lock:
            cached = self._cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            logger.debug(f"Tool cache hit: {tool_call.tool_name}")
            # Return a copy so callers mutating the result don't corrupt the cache.
            return cached.model_copy(deep=True)

        self.cache_misses += 1
        result = super().execute(tool_call)
        if result.success:
            with self._cache_lock:
                if self._max_entries is not None:
                    while len(self._cache) >= self._max_entries:
                        # Evict oldest insertion.
                        self._cache.pop(next(iter(self._cache)))
                self._cache[key] = result.model_copy(deep=True)
        return result

    def clear_cache(self) -> None:
        """Drop all cached results."""
        with self._cache_lock:
            self._cache.clear()


class RetryingToolRegistry(ToolRegistry):
    """A :class:`ToolRegistry` that retries failed tool executions.

    ``ToolRegistry.execute`` never raises — it returns
    ``ToolResult(success=False)`` on error. This subclass re-invokes up to
    ``max_retries`` additional times (with optional backoff) whenever a result
    comes back unsuccessful, returning the first success or the last failure.

    Args:
        max_retries: Additional attempts after the first (total tries =
            ``max_retries + 1``).
        backoff_seconds: Base sleep between attempts; attempt *n* sleeps
            ``backoff_seconds * n`` (linear). ``0`` disables sleeping.
    """

    def __init__(self, max_retries: int = 2, backoff_seconds: float = 0.0) -> None:
        super().__init__()
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if backoff_seconds < 0:
            raise ValueError("backoff_seconds must be >= 0")
        self._max_retries = max_retries
        self._backoff = backoff_seconds

    def execute(self, tool_call: ToolCall) -> ToolResult:
        result = super().execute(tool_call)
        attempt = 0
        while not result.success and attempt < self._max_retries:
            attempt += 1
            if self._backoff:
                time.sleep(self._backoff * attempt)
            logger.debug(
                f"Retrying tool '{tool_call.tool_name}' "
                f"(attempt {attempt + 1}/{self._max_retries + 1})"
            )
            result = super().execute(tool_call)
        return result
