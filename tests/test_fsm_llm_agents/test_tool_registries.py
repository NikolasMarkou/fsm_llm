"""Tests for get_json_schemas() + CachingToolRegistry + RetryingToolRegistry."""

from __future__ import annotations

from typing import Annotated

import pytest

from fsm_llm.agents import CachingToolRegistry, RetryingToolRegistry, tool
from fsm_llm.tools import ToolAnnotations, ToolCall, ToolRegistry


@tool
def search(query: Annotated[str, "the search query"]) -> str:
    """Search the web."""
    return f"results for {query}"


@tool
def add(a: int, b: int) -> str:
    """Add two numbers."""
    return str(a + b)


class TestGetJsonSchemas:
    def test_shape_is_openai_compatible(self):
        reg = ToolRegistry()
        reg.register(search._tool_definition)
        schemas = reg.get_json_schemas()
        assert len(schemas) == 1
        entry = schemas[0]
        assert entry["type"] == "function"
        fn = entry["function"]
        assert fn["name"] == "search"
        assert fn["description"] == "Search the web."
        assert fn["parameters"]["type"] == "object"
        assert "query" in fn["parameters"]["properties"]
        assert fn["parameters"]["required"] == ["query"]

    def test_empty_registry_returns_empty_list(self):
        assert ToolRegistry().get_json_schemas() == []

    def test_no_schema_tool_gets_empty_properties(self):
        reg = ToolRegistry()
        reg.register_function(lambda: "x", name="noop", description="does nothing")
        schemas = reg.get_json_schemas()
        assert schemas[0]["function"]["parameters"] == {
            "type": "object",
            "properties": {},
        }

    def test_multi_param_required_list(self):
        reg = ToolRegistry()
        reg.register(add._tool_definition)
        params = reg.get_json_schemas()[0]["function"]["parameters"]
        assert set(params["required"]) == {"a", "b"}


class TestCachingToolRegistry:
    def test_is_a_tool_registry(self):
        assert issubclass(CachingToolRegistry, ToolRegistry)

    def test_caches_successful_results(self):
        calls = {"n": 0}

        def counter(query: str) -> str:
            calls["n"] += 1
            return f"call {calls['n']}"

        reg = CachingToolRegistry()
        reg.register_function(
            counter,
            name="counter",
            description="counts",
            parameter_schema={
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
        tc = ToolCall(tool_name="counter", parameters={"query": "x"})
        r1 = reg.execute(tc)
        r2 = reg.execute(ToolCall(tool_name="counter", parameters={"query": "x"}))
        assert r1.result == r2.result  # cached
        assert calls["n"] == 1  # underlying fn invoked once
        assert reg.cache_hits == 1
        assert reg.cache_misses == 1

    def test_different_params_not_cached_together(self):
        def echo(query: str) -> str:
            return query

        reg = CachingToolRegistry()
        reg.register_function(
            echo,
            name="echo",
            description="echo",
            parameter_schema={
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
        a = reg.execute(ToolCall(tool_name="echo", parameters={"query": "a"}))
        b = reg.execute(ToolCall(tool_name="echo", parameters={"query": "b"}))
        assert a.result == "a"
        assert b.result == "b"

    def test_failures_not_cached(self):
        state = {"fail": True}

        def flaky(query: str) -> str:
            if state["fail"]:
                raise RuntimeError("boom")
            return "ok"

        reg = CachingToolRegistry()
        reg.register_function(
            flaky,
            name="flaky",
            description="flaky",
            parameter_schema={
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
        r1 = reg.execute(ToolCall(tool_name="flaky", parameters={"query": "x"}))
        assert not r1.success
        state["fail"] = False
        r2 = reg.execute(ToolCall(tool_name="flaky", parameters={"query": "x"}))
        assert r2.success  # not served from cache
        assert r2.result == "ok"

    def test_clear_cache(self):
        calls = {"n": 0}

        def counter(query: str) -> str:
            calls["n"] += 1
            return "x"

        reg = CachingToolRegistry()
        reg.register_function(
            counter,
            name="counter",
            description="c",
            parameter_schema={
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
        reg.execute(ToolCall(tool_name="counter", parameters={"query": "x"}))
        reg.clear_cache()
        reg.execute(ToolCall(tool_name="counter", parameters={"query": "x"}))
        assert calls["n"] == 2

    def test_max_entries_eviction(self):
        def echo(query: str) -> str:
            return query

        reg = CachingToolRegistry(max_entries=2)
        reg.register_function(
            echo,
            name="echo",
            description="e",
            parameter_schema={
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
        for q in ["a", "b", "c"]:
            reg.execute(ToolCall(tool_name="echo", parameters={"query": q}))
        assert len(reg._cache) == 2


class TestRetryingToolRegistry:
    def test_is_a_tool_registry(self):
        assert issubclass(RetryingToolRegistry, ToolRegistry)

    def test_retries_until_success(self):
        attempts = {"n": 0}

        def flaky(query: str) -> str:
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise RuntimeError("transient")
            return "ok"

        reg = RetryingToolRegistry(max_retries=3)
        reg.register_function(
            flaky,
            name="flaky",
            description="f",
            annotations=ToolAnnotations(idempotent=True),
            parameter_schema={
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
        result = reg.execute(ToolCall(tool_name="flaky", parameters={"query": "x"}))
        assert result.success
        assert result.result == "ok"
        assert attempts["n"] == 3

    def test_returns_last_failure_when_exhausted(self):
        def always_fail(query: str) -> str:
            raise RuntimeError("permanent")

        reg = RetryingToolRegistry(max_retries=2)
        reg.register_function(
            always_fail,
            name="boom",
            description="b",
            parameter_schema={
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
        result = reg.execute(ToolCall(tool_name="boom", parameters={"query": "x"}))
        assert not result.success
        assert "permanent" in (result.error or "")

    def test_success_first_try_no_retry(self):
        attempts = {"n": 0}

        def ok(query: str) -> str:
            attempts["n"] += 1
            return "fine"

        reg = RetryingToolRegistry(max_retries=5)
        reg.register_function(
            ok,
            name="ok",
            description="o",
            parameter_schema={
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
        reg.execute(ToolCall(tool_name="ok", parameters={"query": "x"}))
        assert attempts["n"] == 1

    def test_invalid_args_rejected(self):
        with pytest.raises(ValueError):
            RetryingToolRegistry(max_retries=-1)
        with pytest.raises(ValueError):
            RetryingToolRegistry(backoff_seconds=-1)


class TestCacheSkipsGatedCalls:
    def _registry(self, paid: list[int]) -> CachingToolRegistry:
        def pay(amount: int) -> str:
            paid.append(amount)
            return f"paid {amount}"

        registry = CachingToolRegistry()
        registry.register_function(pay, requires_approval=True)
        return registry

    def test_each_granted_call_runs(self):
        paid: list[int] = []
        registry = self._registry(paid)
        call = ToolCall(tool_name="pay", parameters={"amount": 10})
        for _ in range(2):
            assert registry.execute(call, gated=True).success
        assert paid == [10, 10]
        assert registry.cache_hits == 0

    def test_a_granted_result_is_never_served_to_a_later_call(self):
        paid: list[int] = []
        registry = self._registry(paid)
        call = ToolCall(tool_name="pay", parameters={"amount": 10})
        registry.execute(call, gated=True)
        registry.execute(call)
        assert paid == [10, 10]

    def test_ungated_calls_are_still_cached(self):
        paid: list[int] = []
        registry = self._registry(paid)
        call = ToolCall(tool_name="pay", parameters={"amount": 10})
        registry.execute(call)
        registry.execute(call)
        assert paid == [10]
        assert registry.cache_hits == 1


class TestRetryingRegistryRespectsApproval:
    @staticmethod
    def _failing(
        registry: RetryingToolRegistry, runs: list[str], flagged: bool
    ) -> None:
        def pay(amount: str) -> str:
            runs.append(amount)
            raise RuntimeError("gateway down")

        # Idempotent, so only the approval flag decides (step 5 retry rule).
        registry.register_function(
            pay,
            name="pay",
            description="d",
            requires_approval=flagged,
            annotations=ToolAnnotations(idempotent=True),
        )

    def test_flagged_tool_runs_once(self):
        registry = RetryingToolRegistry(max_retries=3)
        runs: list[str] = []
        self._failing(registry, runs, flagged=True)
        result = registry.execute(ToolCall(tool_name="pay", parameters={"amount": "9"}))
        assert result.success is False
        assert runs == ["9"]

    def test_unflagged_idempotent_tool_is_still_retried(self):
        registry = RetryingToolRegistry(max_retries=3)
        runs: list[str] = []
        self._failing(registry, runs, flagged=False)
        registry.execute(ToolCall(tool_name="pay", parameters={"amount": "9"}))
        assert runs == ["9"] * 4
