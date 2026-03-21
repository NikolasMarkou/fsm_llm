from __future__ import annotations

"""Tests for fsm_llm_agents.tools module."""

import pytest

from fsm_llm_agents.definitions import ToolCall, ToolDefinition
from fsm_llm_agents.exceptions import ToolNotFoundError
from fsm_llm_agents.tools import ToolRegistry, tool


def _add(params):
    """Add two numbers."""
    return params["a"] + params["b"]


def _greet():
    """Greet the user."""
    return "Hello!"


def _failing_tool(params):
    """Tool that always fails."""
    raise RuntimeError("Intentional failure")


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_register_tool(self):
        registry = ToolRegistry()
        tool_def = ToolDefinition(
            name="add",
            description="Add numbers",
            execute_fn=_add,
        )
        registry.register(tool_def)
        assert "add" in registry
        assert len(registry) == 1

    def test_register_function(self):
        registry = ToolRegistry()
        registry.register_function(_add, name="add", description="Add two numbers")
        assert "add" in registry

    def test_register_function_chaining(self):
        registry = ToolRegistry()
        result = registry.register_function(_add, name="add", description="Add")
        assert result is registry

    def test_register_without_execute_fn_raises(self):
        registry = ToolRegistry()
        tool_def = ToolDefinition(name="bad", description="No fn")
        with pytest.raises(ValueError, match="execute_fn"):
            registry.register(tool_def)

    def test_get_existing_tool(self):
        registry = ToolRegistry()
        registry.register_function(_add, name="add", description="Add")
        tool = registry.get("add")
        assert tool.name == "add"

    def test_get_nonexistent_tool(self):
        registry = ToolRegistry()
        with pytest.raises(ToolNotFoundError):
            registry.get("nonexistent")

    def test_list_tools(self):
        registry = ToolRegistry()
        registry.register_function(_add, name="add", description="Add")
        registry.register_function(_greet, name="greet", description="Greet")
        tools = registry.list_tools()
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert names == {"add", "greet"}

    def test_tool_names(self):
        registry = ToolRegistry()
        registry.register_function(_add, name="add", description="Add")
        assert registry.tool_names == ["add"]

    def test_contains(self):
        registry = ToolRegistry()
        registry.register_function(_add, name="add", description="Add")
        assert "add" in registry
        assert "missing" not in registry

    def test_execute_success(self):
        registry = ToolRegistry()
        registry.register_function(_add, name="add", description="Add")

        call = ToolCall(tool_name="add", parameters={"a": 2, "b": 3})
        result = registry.execute(call)

        assert result.success is True
        assert result.result == 5
        assert result.execution_time_ms > 0

    def test_execute_no_params(self):
        registry = ToolRegistry()
        registry.register_function(_greet, name="greet", description="Greet")

        call = ToolCall(tool_name="greet", parameters={})
        result = registry.execute(call)

        assert result.success is True
        assert result.result == "Hello!"

    def test_execute_nonexistent_tool(self):
        registry = ToolRegistry()
        call = ToolCall(tool_name="missing", parameters={})
        result = registry.execute(call)

        assert result.success is False
        assert "not found" in result.error.lower()

    def test_execute_failing_tool(self):
        registry = ToolRegistry()
        registry.register_function(_failing_tool, name="fail", description="Fails")

        call = ToolCall(tool_name="fail", parameters={})
        result = registry.execute(call)

        assert result.success is False
        assert "Intentional failure" in result.error

    def test_to_prompt_description(self):
        registry = ToolRegistry()
        registry.register_function(
            _add,
            name="add",
            description="Add two numbers",
            parameter_schema={
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"},
                }
            },
        )
        desc = registry.to_prompt_description()
        assert "add" in desc
        assert "Add two numbers" in desc
        assert "number" in desc

    def test_to_prompt_description_empty(self):
        registry = ToolRegistry()
        assert "No tools" in registry.to_prompt_description()

    def test_to_classification_schema(self):
        registry = ToolRegistry()
        registry.register_function(_add, name="add", description="Add")
        registry.register_function(_greet, name="greet", description="Greet")

        schema = registry.to_classification_schema()
        assert schema["fallback_intent"] == "none"
        intent_names = {i["name"] for i in schema["intents"]}
        assert "add" in intent_names
        assert "greet" in intent_names
        assert "none" in intent_names


class TestToolDecorator:
    """Tests for the @tool decorator."""

    def test_basic_decorator(self):
        @tool(description="Add two numbers")
        def add(a: int, b: int) -> int:
            return a + b

        assert hasattr(add, "_tool_definition")
        defn = add._tool_definition
        assert defn.name == "add"
        assert defn.description == "Add two numbers"
        assert defn.execute_fn is add

    def test_decorator_with_custom_name(self):
        @tool(name="web_search", description="Search the internet")
        def search(query: str) -> str:
            return f"Results for: {query}"

        assert search._tool_definition.name == "web_search"

    def test_decorator_with_approval(self):
        @tool(description="Delete a file", requires_approval=True)
        def delete_file(path: str) -> bool:
            return True

        assert delete_file._tool_definition.requires_approval is True

    def test_register_decorated_function(self):
        @tool(description="Multiply")
        def multiply(a: int, b: int) -> int:
            return a * b

        registry = ToolRegistry()
        registry.register(multiply._tool_definition)
        assert "multiply" in registry
