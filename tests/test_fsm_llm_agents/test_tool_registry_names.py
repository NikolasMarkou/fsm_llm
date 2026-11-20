"""``ToolRegistry`` on ``NameRegistry`` (plan ace7a8c8 step 13, roadmap 0.17, decision D-049).

A duplicate tool name raises unless ``replace=True``; registration order is kept; the one leaf
lock is the name registry's; no subclass reads a private dict; ``execute`` still runs the
CURRENT definition (c1d5bfbc/D-024, 06a5ec0a/D-052 live view).
"""

from __future__ import annotations

import re
import threading
from pathlib import Path

import pytest

import fsm_llm
from fsm_llm.agents.semantic_tools import SemanticToolRegistry
from fsm_llm.agents.tool_registries import RetryingToolRegistry
from fsm_llm.core.names import NameCollisionError, NameRegistry
from fsm_llm.tools import (
    READ_ONLY_ANNOTATIONS,
    ToolCall,
    ToolDefinition,
    ToolRegistry,
    registry_from,
)

_SRC = Path(fsm_llm.__file__).resolve().parent


def _fn_one(params):
    return "one"


def _fn_two(params):
    return "two"


def _tool(name: str, fn=_fn_one, description: str = "d") -> ToolDefinition:
    return ToolDefinition(name=name, description=description, execute_fn=fn)


class TestDuplicateNames:
    def test_a_duplicate_raises_and_keeps_the_first(self):
        registry = ToolRegistry()
        first = _tool("add", description="first")
        registry.register(first)
        with pytest.raises(NameCollisionError, match="add"):
            registry.register(_tool("add", description="second"))
        assert registry.get("add") is first
        assert len(registry) == 1

    def test_replace_true_replaces_and_chains(self):
        registry = ToolRegistry()
        registry.register(_tool("add", description="first"))
        second = _tool("add", description="second")
        assert registry.register(second, replace=True) is registry
        assert registry.get("add") is second
        assert len(registry) == 1

    def test_replace_keeps_the_first_position(self):
        registry = ToolRegistry()
        for name in ("zeta", "alpha", "mid"):
            registry.register(_tool(name))
        registry.register(_tool("zeta", description="again"), replace=True)
        assert registry.tool_names == ["zeta", "alpha", "mid"]
        assert [t.name for t in registry.list_tools()] == ["zeta", "alpha", "mid"]

    def test_register_function_refuses_a_duplicate_and_replaces_on_request(self):
        registry = ToolRegistry()
        registry.register_function(_fn_one, name="t", description="d")
        with pytest.raises(NameCollisionError):
            registry.register_function(_fn_two, name="t", description="d")
        registry.register_function(_fn_two, name="t", description="d", replace=True)
        result = registry.execute(ToolCall(tool_name="t"))
        assert result.success and result.result == "two"

    def test_register_python_tool_passes_replace_through(self):
        class _Py:
            def __init__(self, definition):
                self._definition = definition

            def to_tool_definition(self):
                return self._definition

        registry = ToolRegistry()
        registry.register_python_tool(_Py(_tool("p")))
        with pytest.raises(NameCollisionError):
            registry.register_python_tool(_Py(_tool("p")))
        registry.register_python_tool(_Py(_tool("p", _fn_two)), replace=True)
        assert registry.execute(ToolCall(tool_name="p")).result == "two"

    def test_register_agent_passes_replace_through(self):
        class _Agent:
            def run(self, task):
                return "answer"

        registry = ToolRegistry()
        registry.register_agent(_Agent(), "sub", "d")
        with pytest.raises(NameCollisionError):
            registry.register_agent(_Agent(), "sub", "d")
        registry.register_agent(_Agent(), "sub", "d", replace=True)

    def test_registry_from_refuses_a_duplicate_name(self):
        with pytest.raises(NameCollisionError):
            registry_from([_tool("x"), _tool("x")])

    def test_a_refused_duplicate_does_not_change_what_execute_runs(self):
        registry = ToolRegistry()
        registry.register(_tool("t", _fn_one))
        with pytest.raises(NameCollisionError):
            registry.register(_tool("t", _fn_two))
        assert registry.execute(ToolCall(tool_name="t")).result == "one"

    def test_execute_runs_the_current_definition_after_a_replace(self):
        registry = ToolRegistry()
        registry.register(_tool("t", _fn_one))
        registry.register(_tool("t", _fn_two), replace=True)
        assert registry.execute(ToolCall(tool_name="t")).result == "two"

    def test_the_reserved_name_and_a_missing_function_are_still_refused(self):
        registry = ToolRegistry()
        with pytest.raises(ValueError, match="reserved"):
            registry.register(_tool("none"))
        with pytest.raises(ValueError, match="execute_fn"):
            registry.register(ToolDefinition(name="bare", description="d"))


class TestBuiltOnNameRegistry:
    def test_the_registry_holds_a_name_registry_of_namespace_tools(self):
        registry = ToolRegistry()
        assert isinstance(registry._registry, NameRegistry)
        assert registry._registry.namespace == "tools"
        assert not hasattr(registry, "_tools")
        assert not hasattr(registry, "_tools_lock")

    def test_the_leaf_lock_is_free_while_a_tool_runs(self):
        release = threading.Event()
        started = threading.Event()
        seen: list[bool] = []

        def hold(params):
            started.set()
            seen.append(registry._registry._lock.locked())
            release.wait(5.0)
            return "ok"

        registry = ToolRegistry()
        registry.register(_tool("hold", hold))
        results = []
        caller = threading.Thread(
            target=lambda: results.append(registry.execute(ToolCall(tool_name="hold")))
        )
        caller.start()
        try:
            assert started.wait(5.0)
            assert registry._registry._lock.acquire(timeout=1.0)
            registry._registry._lock.release()
            registry.register(_tool("other"))
        finally:
            release.set()
            caller.join(5.0)
        assert seen == [False]
        assert results[0].success

    def test_no_subclass_module_reads_a_private_tool_dict(self):
        for relative in (
            "agents/tool_registries.py",
            "agents/semantic_tools.py",
            "agents/reasoning_react.py",
        ):
            text = (_SRC / relative).read_text(encoding="utf-8")
            assert not re.search(r"\._tools\b|_tools_lock", text), relative

    def test_the_retrying_registry_still_decides_by_the_current_definition(self):
        registry = RetryingToolRegistry(max_retries=2)
        calls: list[int] = []

        def flaky(params):
            calls.append(1)
            raise RuntimeError("boom")

        registry.register(
            ToolDefinition(
                name="flaky",
                description="d",
                execute_fn=flaky,
                annotations=READ_ONLY_ANNOTATIONS,
            )
        )
        assert not registry.execute(ToolCall(tool_name="flaky")).success
        assert len(calls) == 3
        assert not registry.execute(ToolCall(tool_name="missing")).success


class TestSemanticRegistry:
    @staticmethod
    def _registry():
        embedded: list[str] = []

        def embed(texts):
            embedded.extend(texts)
            return [[float(len(t)), 1.0] for t in texts]

        return SemanticToolRegistry(embed_fn=embed), embedded

    def test_a_refused_duplicate_embeds_nothing(self):
        registry, embedded = self._registry()
        registry.register(_tool("a", description="first"))
        before = len(embedded)
        with pytest.raises(NameCollisionError):
            registry.register(_tool("a", description="second"))
        assert len(embedded) == before

    def test_replace_embeds_the_new_tool(self):
        registry, embedded = self._registry()
        registry.register(_tool("a", description="first"))
        before = len(embedded)
        registry.register(_tool("a", description="second one"), replace=True)
        assert len(embedded) > before
