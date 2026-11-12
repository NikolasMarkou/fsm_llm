"""MCPToolProvider with mocked sessions: result formatting, reconnect per call, timeouts, name collisions.

Needs no ``mcp`` SDK (sessions are stand-ins); the real stdio round trip is ``test_mcp_stdio.py``.
The async tests rely on ``asyncio_mode = "auto"`` in ``pyproject.toml``.
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel

from fsm_llm.agents.mcp import _format_mcp_result


class _FakeResult(BaseModel):
    content: list[Any] = []
    isError: bool = False
    structuredContent: dict[str, Any] | None = None


class TestFormatMcpResultWithoutContent:
    """F-AI-14: an empty ``content`` showed the model a pydantic repr."""

    def test_structured_content_is_json(self):
        text = _format_mcp_result(_FakeResult(structuredContent={"answer": 42}))
        assert json.loads(text) == {"answer": 42}

    def test_nothing_at_all_is_empty_text(self):
        assert _format_mcp_result(_FakeResult()) == ""

    def test_content_still_wins(self):
        result = SimpleNamespace(
            content=[SimpleNamespace(text="hi")], structuredContent={"a": 1}
        )
        assert _format_mcp_result(result) == "hi"


class TestMCPSessionLifecycle:
    """Verify MCP executors reconnect per-call instead of using closed session."""

    def test_executor_does_not_capture_session(self):
        """Executor closure captures server_params, not session."""
        from fsm_llm.agents.mcp import MCPToolProvider

        provider = MCPToolProvider.__new__(MCPToolProvider)
        provider._server_params = "mock_params"
        provider._server_url = None

        mock_tool = Mock()
        mock_tool.name = "test-tool"
        mock_tool.description = "Test"
        mock_tool.inputSchema = None

        tool_def = provider._convert_mcp_tool(mock_tool)
        assert tool_def.execute_fn is not None
        assert tool_def.name == "test-tool"

    def test_format_mcp_result_with_text(self):
        """_format_mcp_result extracts text from content items."""
        from fsm_llm.agents.mcp import _format_mcp_result

        result = Mock()
        item1 = Mock()
        item1.text = "Hello"
        item2 = Mock()
        item2.text = "World"
        result.content = [item1, item2]

        assert _format_mcp_result(result) == "Hello\nWorld"

    def test_format_mcp_result_fallback(self):
        """_format_mcp_result falls back to str() for non-content results."""
        from fsm_llm.agents.mcp import _format_mcp_result

        assert _format_mcp_result("plain string") == "plain string"

    @pytest.mark.parametrize("flag", ["isError", "is_error"])
    def test_format_mcp_result_error_flag_raises(self, flag):
        """An MCP error result (1.x isError, 2.x is_error) is a failed call (D-026)."""
        from types import SimpleNamespace

        from fsm_llm.agents.exceptions import ToolExecutionError
        from fsm_llm.agents.mcp import _format_mcp_result

        result = SimpleNamespace(content=[SimpleNamespace(text="boom")], **{flag: True})
        with pytest.raises(ToolExecutionError, match="boom") as exc:
            _format_mcp_result(result, "t")
        assert exc.value.tool_name == "t"
        ok = SimpleNamespace(content=[SimpleNamespace(text="fine")], **{flag: False})
        assert _format_mcp_result(ok, "t") == "fine"

    @pytest.mark.parametrize("flag", ["isError", "is_error"])
    def test_mcp_error_result_is_failed_tool_result(self, flag):
        """Through the executor and ToolRegistry: success False, error text kept."""
        from contextlib import asynccontextmanager
        from types import SimpleNamespace

        from fsm_llm.agents import mcp as mcp_mod
        from fsm_llm.agents.definitions import ToolCall
        from fsm_llm.agents.tools import ToolRegistry

        err = SimpleNamespace(
            content=[SimpleNamespace(text="bad input")], **{flag: True}
        )

        @asynccontextmanager
        async def fake_stdio(_params):
            yield (None, None)

        class FakeSession:
            def __init__(self, *_a):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_a):
                return False

            async def initialize(self):
                return None

            async def call_tool(self, name, arguments):
                return err

        provider = mcp_mod.MCPToolProvider.__new__(mcp_mod.MCPToolProvider)
        provider._server_params = "params"
        provider._server_url = None
        # A one-property schema keeps this off the zero-argument call path (W5).
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        tool = SimpleNamespace(name="t", description="d", inputSchema=schema)
        with (
            patch.object(mcp_mod, "stdio_client", fake_stdio, create=True),
            patch.object(mcp_mod, "ClientSession", FakeSession, create=True),
        ):
            registry = ToolRegistry()
            registry.register(provider._convert_mcp_tool(tool))
            result = registry.execute(ToolCall(tool_name="t", parameters={"x": 1}))
        assert not result.success
        assert "bad input" in (result.error or "")


class TestMCPToolProvider:
    """Tests for MCPToolProvider."""

    def test_import_without_mcp(self):
        """MCPToolProvider module imports even without mcp SDK."""
        from fsm_llm.agents.mcp import MCPToolProvider

        assert MCPToolProvider is not None

    def test_create_mock_tool(self):
        """create_mock_tool creates valid ToolDefinitions."""
        from fsm_llm.agents.mcp import MCPToolProvider

        tool = MCPToolProvider.create_mock_tool(
            name="test-tool",
            description="A test tool",
            parameter_schema={"properties": {"query": {"type": "string"}}},
        )
        assert tool.name == "test-tool"
        assert tool.description == "A test tool"
        assert tool.execute_fn is not None

    def test_create_mock_tool_execution(self):
        """Mock tool executes and returns JSON."""
        from fsm_llm.agents.mcp import MCPToolProvider

        tool = MCPToolProvider.create_mock_tool(
            name="echo",
            description="Echo input",
            execute_fn=lambda **kwargs: f"echoed: {kwargs}",
        )
        result = tool.execute_fn(query="hello")
        assert "hello" in result

    def test_register_mock_tools_with_registry(self):
        """Mock tools can be registered with ToolRegistry."""
        from fsm_llm.agents.mcp import MCPToolProvider
        from fsm_llm.agents.tools import ToolRegistry

        provider = MCPToolProvider.__new__(MCPToolProvider)
        provider._tools = [
            MCPToolProvider.create_mock_tool("tool-a", "Tool A"),
            MCPToolProvider.create_mock_tool("tool-b", "Tool B"),
        ]

        registry = ToolRegistry()
        count = provider.register_tools(registry)
        assert count == 2
        assert "tool-a" in registry
        assert "tool-b" in registry

    def test_schema_conversion(self):
        """MCP input schema converts to ToolDefinition parameter_schema."""
        from fsm_llm.agents.mcp import _mcp_schema_to_parameter_schema

        schema = {
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        result = _mcp_schema_to_parameter_schema(schema)
        assert result["properties"]["query"]["type"] == "string"
        assert "query" in result["required"]

    def test_empty_schema_conversion(self):
        """Empty schema converts to empty dict."""
        from fsm_llm.agents.mcp import _mcp_schema_to_parameter_schema

        assert _mcp_schema_to_parameter_schema({}) == {}
        assert _mcp_schema_to_parameter_schema(None) == {}

    def test_get_tool_names(self):
        """get_tool_names returns names of discovered tools."""
        from fsm_llm.agents.mcp import MCPToolProvider

        provider = MCPToolProvider.__new__(MCPToolProvider)
        provider._tools = [
            MCPToolProvider.create_mock_tool("alpha", "Alpha tool"),
            MCPToolProvider.create_mock_tool("beta", "Beta tool"),
        ]
        assert provider.get_tool_names() == ["alpha", "beta"]


class _FakeMCPSession:
    """Stand-in for ``mcp.ClientSession``; sleeps are set per test."""

    init_delay = 0.0
    call_delay = 0.0

    def __init__(self, read, write):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def initialize(self):
        await asyncio.sleep(self.init_delay)

    async def list_tools(self):
        tool = SimpleNamespace(
            name="echo",
            description="Echo tool",
            inputSchema={"properties": {"text": {"type": "string"}}},
        )
        return SimpleNamespace(tools=[tool])

    async def call_tool(self, name, arguments=None):
        await asyncio.sleep(self.call_delay)
        return SimpleNamespace(content=[SimpleNamespace(text=f"{name}:ok")])


@asynccontextmanager
async def _fake_stdio_client(params):
    yield ("read", "write")


class TestMCPTimeouts:
    """MM-06: MCP discovery and tool calls are bounded by a timeout."""

    HANG = 5.0

    def _provider(self, monkeypatch, init_delay=0.0, call_delay=0.0, **kwargs):
        import fsm_llm.agents.mcp as mcp_mod

        session_cls = type(
            "Session",
            (_FakeMCPSession,),
            {"init_delay": init_delay, "call_delay": call_delay},
        )
        monkeypatch.setattr(mcp_mod, "_HAS_MCP", True)
        monkeypatch.setattr(mcp_mod, "stdio_client", _fake_stdio_client, raising=False)
        monkeypatch.setattr(mcp_mod, "ClientSession", session_cls, raising=False)
        return mcp_mod.MCPToolProvider(server_params=object(), **kwargs)

    async def test_hung_discovery_raises_agent_timeout(
        self, monkeypatch, timing_factor
    ):
        from fsm_llm.agents.exceptions import AgentTimeoutError

        provider = self._provider(
            monkeypatch, init_delay=self.HANG * timing_factor, timeout=0.05
        )
        start = time.monotonic()
        with pytest.raises(AgentTimeoutError) as info:
            await asyncio.wait_for(provider.discover_tools(), 1.0 * timing_factor)
        assert time.monotonic() - start < 2.0 * timing_factor
        assert isinstance(info.value.__cause__, asyncio.TimeoutError)

    async def test_fast_discovery_and_call_complete(self, monkeypatch, timing_factor):
        provider = self._provider(monkeypatch, timeout=0.5 * timing_factor)
        tools = await asyncio.wait_for(provider.discover_tools(), 1.0 * timing_factor)
        assert [t.name for t in tools] == ["echo"]
        out = await asyncio.wait_for(
            tools[0].execute_fn(text="hi"), 1.0 * timing_factor
        )
        assert out == "echo:ok"

    async def test_timeout_none_is_unbounded(self, monkeypatch, timing_factor):
        provider = self._provider(
            monkeypatch, init_delay=0.1 * timing_factor, timeout=None
        )
        tools = await asyncio.wait_for(provider.discover_tools(), 1.0 * timing_factor)
        assert [t.name for t in tools] == ["echo"]

    async def test_hung_call_raises_tool_execution_error(
        self, monkeypatch, timing_factor
    ):
        from fsm_llm.agents.exceptions import ToolExecutionError

        provider = self._provider(monkeypatch, timeout=0.05)
        tools = await provider.discover_tools()
        import fsm_llm.agents.mcp as mcp_mod

        # Discovery was fast; now make the call hang (per-test subclass).
        mcp_mod.ClientSession.call_delay = self.HANG * timing_factor
        start = time.monotonic()
        with pytest.raises(ToolExecutionError) as info:
            await asyncio.wait_for(tools[0].execute_fn(text="hi"), 1.0 * timing_factor)
        assert time.monotonic() - start < 2.0 * timing_factor
        assert info.value.tool_name == "echo"

    def test_hung_call_via_registry_is_failed_result(self, monkeypatch, timing_factor):
        import fsm_llm.agents.mcp as mcp_mod
        from fsm_llm.agents.definitions import ToolCall
        from fsm_llm.agents.tools import ToolRegistry

        provider = self._provider(monkeypatch, timeout=0.05)
        asyncio.run(provider.discover_tools())
        mcp_mod.ClientSession.call_delay = self.HANG * timing_factor
        registry = ToolRegistry()
        provider.register_tools(registry)
        start = time.monotonic()
        result = registry.execute(ToolCall(tool_name="echo", parameters={"text": "hi"}))
        assert time.monotonic() - start < 2.0 * timing_factor
        assert result.success is False
        assert "echo" in (result.error or "")

    def test_new_built_instance_has_default_timeout(self):
        from fsm_llm.agents.constants import Defaults
        from fsm_llm.agents.mcp import MCPToolProvider

        provider = MCPToolProvider.__new__(MCPToolProvider)
        assert provider._timeout == Defaults.MCP_TIMEOUT_SECONDS


class TestMCPRegisterNameCollision:
    """plan-2026-11-05T134105-5ed71cf2/D-014: skip, warn, count only registered."""

    @staticmethod
    def _provider(*tools):
        from fsm_llm.agents.mcp import MCPToolProvider

        provider = MCPToolProvider.__new__(MCPToolProvider)
        provider._tools = list(tools)
        return provider

    def test_mcp_tool_does_not_replace_gated_local_tool(self):
        from fsm_llm.agents.definitions import ToolDefinition
        from fsm_llm.agents.mcp import MCPToolProvider
        from fsm_llm.agents.tools import ToolRegistry

        local = ToolDefinition(
            name="delete_file",
            description="local",
            requires_approval=True,
            execute_fn=lambda **kw: "local",
        )
        registry = ToolRegistry().register(local)
        provider = self._provider(
            MCPToolProvider.create_mock_tool("delete_file", "remote"),
            MCPToolProvider.create_mock_tool("other", "remote"),
        )
        with pytest.warns(UserWarning, match="delete_file"):
            count = provider.register_tools(registry)
        assert count == 1
        assert registry.get("delete_file") is local
        assert registry.get("delete_file").requires_approval is True
        assert "other" in registry
        assert provider.skipped_tools == ("delete_file",)

    def test_two_providers_with_colliding_name_keep_the_first(self):
        from fsm_llm.agents.mcp import MCPToolProvider
        from fsm_llm.agents.tools import ToolRegistry

        first = MCPToolProvider.create_mock_tool(
            "search", "first", execute_fn=lambda **kw: "first"
        )
        second = MCPToolProvider.create_mock_tool(
            "search", "second", execute_fn=lambda **kw: "second"
        )
        registry = ToolRegistry()
        assert self._provider(first).register_tools(registry) == 1
        with pytest.warns(UserWarning, match="search"):
            assert self._provider(second).register_tools(registry) == 0
        assert registry.get("search") is first

    def test_no_collision_emits_no_warning_and_empty_skipped(self):
        import warnings

        from fsm_llm.agents.mcp import MCPToolProvider
        from fsm_llm.agents.tools import ToolRegistry

        provider = self._provider(MCPToolProvider.create_mock_tool("a", "A"))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert provider.register_tools(ToolRegistry()) == 1
        assert provider.skipped_tools == ()
