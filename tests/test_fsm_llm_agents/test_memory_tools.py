"""Tests for agent memory management tools."""

from fsm_llm.memory import BUFFER_CORE, BUFFER_ENVIRONMENT, WorkingMemory
from fsm_llm_agents.memory_tools import create_memory_tools


class TestCreateMemoryTools:
    """Test create_memory_tools factory."""

    def test_creates_four_tools(self):
        memory = WorkingMemory()
        tools = create_memory_tools(memory)
        assert len(tools) == 4

    def test_tool_names(self):
        memory = WorkingMemory()
        tools = create_memory_tools(memory)
        names = {t.name for t in tools}
        assert names == {"remember", "recall", "forget", "list_memories"}

    def test_all_have_execute_fn(self):
        memory = WorkingMemory()
        tools = create_memory_tools(memory)
        for tool in tools:
            assert tool.execute_fn is not None

    def test_all_have_schemas(self):
        memory = WorkingMemory()
        tools = create_memory_tools(memory)
        for tool in tools:
            assert tool.parameter_schema is not None

    def test_all_have_descriptions(self):
        memory = WorkingMemory()
        tools = create_memory_tools(memory)
        for tool in tools:
            assert tool.description
            assert len(tool.description) > 10


class TestRememberTool:
    """Test the remember tool."""

    def _get_tool(self, memory):
        tools = create_memory_tools(memory)
        return next(t for t in tools if t.name == "remember")

    def test_remember_basic(self):
        memory = WorkingMemory()
        tool = self._get_tool(memory)
        result = tool.execute_fn(key="user_name", value="Alice")
        assert "Remembered" in result
        assert memory.get(BUFFER_CORE, "user_name") == "Alice"

    def test_remember_custom_buffer(self):
        memory = WorkingMemory()
        tool = self._get_tool(memory)
        result = tool.execute_fn(key="result", value="ok", buffer="environment")
        assert "environment" in result
        assert memory.get(BUFFER_ENVIRONMENT, "result") == "ok"

    def test_remember_overwrite(self):
        memory = WorkingMemory()
        tool = self._get_tool(memory)
        tool.execute_fn(key="name", value="Alice")
        tool.execute_fn(key="name", value="Bob")
        assert memory.get(BUFFER_CORE, "name") == "Bob"


class TestRecallTool:
    """Test the recall tool."""

    def _get_tool(self, memory):
        tools = create_memory_tools(memory)
        return next(t for t in tools if t.name == "recall")

    def test_recall_found(self):
        memory = WorkingMemory()
        memory.set(BUFFER_CORE, "user_name", "Alice")
        tool = self._get_tool(memory)
        result = tool.execute_fn(query="user")
        assert "user_name" in result
        assert "Alice" in result

    def test_recall_not_found(self):
        memory = WorkingMemory()
        tool = self._get_tool(memory)
        result = tool.execute_fn(query="nothing")
        assert "No memories found" in result

    def test_recall_multiple_results(self):
        memory = WorkingMemory()
        memory.set(BUFFER_CORE, "user_name", "Alice")
        memory.set(BUFFER_CORE, "user_email", "alice@test.com")
        tool = self._get_tool(memory)
        result = tool.execute_fn(query="user")
        assert "2 matching" in result


class TestForgetTool:
    """Test the forget tool."""

    def _get_tool(self, memory):
        tools = create_memory_tools(memory)
        return next(t for t in tools if t.name == "forget")

    def test_forget_existing(self):
        memory = WorkingMemory()
        memory.set(BUFFER_CORE, "temp", "data")
        tool = self._get_tool(memory)
        result = tool.execute_fn(key="temp")
        assert "Forgot" in result
        assert memory.get(BUFFER_CORE, "temp") is None

    def test_forget_not_found(self):
        memory = WorkingMemory()
        tool = self._get_tool(memory)
        result = tool.execute_fn(key="missing")
        assert "not found" in result


class TestListMemoriesTool:
    """Test the list_memories tool."""

    def _get_tool(self, memory):
        tools = create_memory_tools(memory)
        return next(t for t in tools if t.name == "list_memories")

    def test_list_all_empty(self):
        memory = WorkingMemory()
        tool = self._get_tool(memory)
        result = tool.execute_fn()
        assert "empty" in result

    def test_list_all_with_data(self):
        memory = WorkingMemory()
        memory.set(BUFFER_CORE, "name", "Alice")
        memory.set(BUFFER_ENVIRONMENT, "result", "ok")
        tool = self._get_tool(memory)
        result = tool.execute_fn()
        assert "2 total" in result
        assert "name" in result
        assert "result" in result

    def test_list_specific_buffer(self):
        memory = WorkingMemory()
        memory.set(BUFFER_CORE, "name", "Alice")
        memory.set(BUFFER_ENVIRONMENT, "result", "ok")
        tool = self._get_tool(memory)
        result = tool.execute_fn(buffer="core")
        assert "name" in result
        assert "result" not in result

    def test_list_empty_buffer(self):
        memory = WorkingMemory()
        tool = self._get_tool(memory)
        result = tool.execute_fn(buffer="scratch")
        assert "empty" in result

    def test_list_nonexistent_buffer(self):
        memory = WorkingMemory()
        tool = self._get_tool(memory)
        result = tool.execute_fn(buffer="nonexistent")
        assert "does not exist" in result


class TestMemoryToolsWithRegistry:
    """Test memory tools work with ToolRegistry."""

    def test_register_memory_tools(self):
        from fsm_llm_agents.tools import ToolRegistry

        memory = WorkingMemory()
        tools = create_memory_tools(memory)
        registry = ToolRegistry()
        for tool in tools:
            registry.register(tool)
        assert len(registry) == 4
        assert "remember" in registry
        assert "recall" in registry

    def test_execute_through_registry(self):
        from fsm_llm_agents.definitions import ToolCall
        from fsm_llm_agents.tools import ToolRegistry

        memory = WorkingMemory()
        tools = create_memory_tools(memory)
        registry = ToolRegistry()
        for tool in tools:
            registry.register(tool)

        call = ToolCall(
            tool_name="remember",
            parameters={"key": "test_key", "value": "test_value"},
        )
        result = registry.execute(call)
        assert result.success
        assert memory.get(BUFFER_CORE, "test_key") == "test_value"
