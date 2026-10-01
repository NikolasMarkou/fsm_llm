"""
Drop-in :class:`ToolRegistry` subclasses that add cross-cutting execution
behavior (result caching, retry-on-failure) without changing the dispatch
contract.

Both classes override only :meth:`ToolRegistry.execute`,
preserve its ``(ToolCall, *, gated=False) -> ToolResult`` signature, and inherit every other
method (registration, schema generation, prompt description). Any agent that
accepts a ``ToolRegistry`` accepts these unchanged.

Example::

    from fsm_llm.agents import ReactAgent, AgentConfig
    from fsm_llm.agents.tool_registries import CachingToolRegistry

    registry = CachingToolRegistry()        # behaves like ToolRegistry
    registry.register(search._tool_definition)
    agent = ReactAgent(tools=registry, config=AgentConfig(model=model))
"""

from __future__ import annotations

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
    :class:`ToolResult` instead of re-invoking the tool; a granted (gated)
    call is never served from or stored in the cache. Only *successful*
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

    def execute(self, tool_call: ToolCall, *, gated: bool = False) -> ToolResult:
        """Return a cached success for an identical call, else run it.

        A ``gated`` call (one an approver granted) always runs and is never
        cached: one approval covers exactly one execution.
        """
        # DECISION plan-2026-10-01T093600-944e2692/D-033: a granted call
        # bypasses the cache both ways. Do NOT serve it from the cache (the
        # approver approved an execution that would never happen, and the
        # model reports it as done) and do NOT store its result (a later
        # identical call, granted or not, would replay the side effect's
        # result without running). See D-052 of plan 06a5ec0a, D-008, D-033.
        if gated:
            return super().execute(tool_call, gated=True)
        key = self._cache_key(tool_call)
        with self._cache_lock:
            cached = self._cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            logger.debug(f"Tool cache hit: {tool_call.tool_name}")
            # Return a copy so callers mutating the result don't corrupt the cache.
            return cached.model_copy(deep=True)

        self.cache_misses += 1
        result = super().execute(tool_call, gated=gated)
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

    Only a tool whose annotations say ``idempotent=True`` or
    ``read_only=True`` (``ToolAnnotations.retry_safe``) is retried. A tool
    registered with ``requires_approval=True``, and any execution an approver
    granted (``execute(..., gated=True)``, which covers policy-gated tools), is
    never retried: one approval covers exactly one execution.
    """

    def __init__(self, max_retries: int = 2, backoff_seconds: float = 0.0) -> None:
        super().__init__()
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if backoff_seconds < 0:
            raise ValueError("backoff_seconds must be >= 0")
        self._max_retries = max_retries
        self._backoff = backoff_seconds

    def _retryable(self, tool_call: ToolCall, *, gated: bool) -> bool:
        """True when a failed *tool_call* may be re-run. Never raises."""
        # DECISION plan-2026-09-29T103145-06a5ec0a/D-052: one approval = one
        # call (D-015). The approval gate and grant spend run once per executor
        # turn, so a retry here would re-run an approved side effect with no
        # second approval. Do NOT retry a requires_approval tool, and do NOT
        # retry any granted execution (the executor passes `gated=True`, D-008
        # of plan 944e2692; the registry cannot see the policy itself).
        if gated:
            return False
        with self._tools_lock:
            tool = self._tools.get(tool_call.tool_name)
        if tool is None or tool.requires_approval:
            return False
        return tool.annotations.retry_safe

    def execute(self, tool_call: ToolCall, *, gated: bool = False) -> ToolResult:
        """Run *tool_call*, re-running a failed retry-safe, ungated call.

        ``gated=True`` (a call an approver granted) is never retried.
        """
        result = super().execute(tool_call, gated=gated)
        if not result.success and not self._retryable(tool_call, gated=gated):
            return result
        attempt = 0
        while not result.success and attempt < self._max_retries:
            attempt += 1
            if self._backoff:
                time.sleep(self._backoff * attempt)
            logger.debug(
                f"Retrying tool '{tool_call.tool_name}' "
                f"(attempt {attempt + 1}/{self._max_retries + 1})"
            )
            result = super().execute(tool_call, gated=gated)
        return result
