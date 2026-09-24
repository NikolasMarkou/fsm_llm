"""
MCP (Model Context Protocol) Tool Integration.

Connects to MCP servers and converts their tool definitions into
ToolDefinition objects compatible with ToolRegistry.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fsm_llm.logging import logger

from .constants import Defaults
from .definitions import ToolDefinition
from .exceptions import AgentTimeoutError, ToolExecutionError

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    _HAS_MCP = True
except ImportError:
    _HAS_MCP = False


def _require_mcp() -> None:
    if not _HAS_MCP:
        raise ImportError(
            "MCP support requires the 'mcp' package. "
            "Install it with: pip install fsm-llm[mcp] or pip install mcp"
        )


def _mcp_schema_to_parameter_schema(input_schema: dict[str, Any]) -> dict[str, Any]:
    """Convert an MCP tool's input_schema to ToolDefinition parameter_schema."""
    if not input_schema:
        return {}
    schema: dict[str, Any] = {}
    if "properties" in input_schema:
        schema["properties"] = input_schema["properties"]
    if "required" in input_schema:
        schema["required"] = input_schema["required"]
    return schema


def _format_mcp_result(result: Any) -> str:
    """Format an MCP tool result into a string."""
    if hasattr(result, "content") and result.content:
        parts = []
        for item in result.content:
            if hasattr(item, "text"):
                parts.append(item.text)
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(result)


class MCPToolProvider:
    """Connects to an MCP server and provides tools as ToolDefinitions.

    The provider discovers tools from an MCP server and converts them
    into ToolDefinition objects that can be registered with a ToolRegistry.

    Example::

        from fsm_llm_agents import ToolRegistry
        from fsm_llm_agents.mcp import MCPToolProvider

        provider = MCPToolProvider.from_stdio("npx", ["-y", "@modelcontextprotocol/server-everything"])
        registry = ToolRegistry()
        provider.register_tools(registry)

    Discovery and every tool call are bounded by ``timeout`` seconds
    (``None`` = unbounded). Expiry raises ``AgentTimeoutError`` from
    ``discover_tools`` and ``ToolExecutionError`` from a tool call.
    """

    # Class-level default so instances built via ``__new__`` still have one.
    _timeout: float | None = Defaults.MCP_TIMEOUT_SECONDS

    def __init__(
        self,
        server_params: Any | None = None,
        server_url: str | None = None,
        timeout: float | None = Defaults.MCP_TIMEOUT_SECONDS,
    ) -> None:
        _require_mcp()
        self._server_params = server_params
        self._server_url = server_url
        self._timeout = timeout
        self._tools: list[ToolDefinition] = []
        self._session: Any | None = None

    @classmethod
    def from_stdio(
        cls,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = Defaults.MCP_TIMEOUT_SECONDS,
    ) -> MCPToolProvider:
        """Create a provider that connects via stdio transport.

        Args:
            command: The command to run the MCP server.
            args: Arguments to pass to the command.
            env: Environment variables for the server process.
            timeout: Seconds per discovery and per tool call; None = unbounded.
        """
        _require_mcp()
        params = StdioServerParameters(
            command=command,
            args=args or [],
            env=env,
        )
        return cls(server_params=params, timeout=timeout)

    @classmethod
    def from_url(
        cls, url: str, timeout: float | None = Defaults.MCP_TIMEOUT_SECONDS
    ) -> MCPToolProvider:
        """Create a provider that connects via HTTP/SSE transport.

        Args:
            url: The URL of the MCP server.
            timeout: Seconds per discovery and per tool call; None = unbounded.
        """
        _require_mcp()
        return cls(server_url=url, timeout=timeout)

    async def discover_tools(self) -> list[ToolDefinition]:
        """Connect to the MCP server and discover available tools.

        Returns a list of ToolDefinition objects.

        Raises:
            AgentTimeoutError: discovery did not finish within the timeout.
        """
        _require_mcp()

        if self._server_params is not None:
            discovery = self._discover_stdio()
        elif self._server_url is not None:
            discovery = self._discover_http()
        else:
            raise ValueError("No server_params or server_url configured")
        try:
            return await asyncio.wait_for(discovery, timeout=self._timeout)
        # asyncio.TimeoutError, not the builtin: they differ on Python 3.10.
        except asyncio.TimeoutError as e:
            raise AgentTimeoutError(self._timeout or 0.0) from e

    async def _discover_stdio(self) -> list[ToolDefinition]:
        """Discover tools via stdio transport."""
        tools: list[ToolDefinition] = []
        async with stdio_client(self._server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                for mcp_tool in result.tools:
                    tool_def = self._convert_mcp_tool(mcp_tool)
                    tools.append(tool_def)
        self._tools = tools
        logger.info(f"Discovered {len(tools)} tools from MCP stdio server")
        return tools

    async def _discover_http(self) -> list[ToolDefinition]:
        """Discover tools via HTTP/SSE transport."""
        try:
            from mcp.client.sse import sse_client
        except ImportError:
            raise ImportError(
                "SSE client requires additional MCP dependencies"
            ) from None

        tools: list[ToolDefinition] = []
        async with sse_client(self._server_url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                for mcp_tool in result.tools:
                    tool_def = self._convert_mcp_tool(mcp_tool)
                    tools.append(tool_def)
        self._tools = tools
        logger.info(f"Discovered {len(tools)} tools from MCP HTTP server")
        return tools

    def _convert_mcp_tool(self, mcp_tool: Any) -> ToolDefinition:
        """Convert an MCP tool object into a ToolDefinition.

        Executors reconnect to the MCP server per-call rather than
        capturing a session reference that may be closed.
        """
        # DECISION plan-2026-09-24T091842-c1d5bfbc/D-025
        # mcp 2.x renamed Tool.inputSchema to input_schema; the 2.x name is read
        # only when the 1.x one is absent. Do NOT read just one: a tool with no
        # schema is called with one positional dict and every call fails.
        raw_schema = getattr(
            mcp_tool, "inputSchema", getattr(mcp_tool, "input_schema", None)
        )
        input_schema: dict[str, Any] = {}
        if raw_schema:
            input_schema = (
                raw_schema if isinstance(raw_schema, dict) else raw_schema.model_dump()
            )

        param_schema = _mcp_schema_to_parameter_schema(input_schema)

        def make_executor(
            tool_name: str,
            server_params: Any,
            server_url: str | None,
            timeout: float | None,
        ):
            """Create an executor that reconnects per call."""

            async def call(**kwargs: Any) -> str:
                if server_params is not None:
                    async with stdio_client(server_params) as (rd, wr):
                        async with ClientSession(rd, wr) as sess:
                            await sess.initialize()
                            result = await sess.call_tool(tool_name, arguments=kwargs)
                            return _format_mcp_result(result)
                elif server_url is not None:
                    from mcp.client.sse import sse_client

                    async with sse_client(server_url) as (rd, wr):
                        async with ClientSession(rd, wr) as sess:
                            await sess.initialize()
                            result = await sess.call_tool(tool_name, arguments=kwargs)
                            return _format_mcp_result(result)
                else:
                    raise ValueError("No server_params or server_url configured")

            async def execute(**kwargs: Any) -> str:
                try:
                    return await asyncio.wait_for(call(**kwargs), timeout=timeout)
                except asyncio.TimeoutError as e:
                    raise ToolExecutionError(
                        f"MCP tool '{tool_name}' timed out after {timeout}s",
                        tool_name=tool_name,
                    ) from e

            return execute

        return ToolDefinition(
            name=mcp_tool.name,
            description=getattr(mcp_tool, "description", "")
            or f"MCP tool: {mcp_tool.name}",
            parameter_schema=param_schema,
            execute_fn=make_executor(
                mcp_tool.name, self._server_params, self._server_url, self._timeout
            ),
        )

    def register_tools(self, registry: Any) -> int:
        """Register all discovered tools with a ToolRegistry.

        Must call discover_tools() (async) first.

        Args:
            registry: A ToolRegistry instance.

        Returns:
            Number of tools registered.
        """
        count = 0
        for tool in self._tools:
            try:
                registry.register(tool)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to register MCP tool '{tool.name}': {e}")
        return count

    @property
    def tools(self) -> list[ToolDefinition]:
        """Return the list of discovered tools."""
        return list(self._tools)

    def get_tool_names(self) -> list[str]:
        """Return names of all discovered tools."""
        return [t.name for t in self._tools]

    @staticmethod
    def create_mock_tool(
        name: str,
        description: str,
        parameter_schema: dict[str, Any] | None = None,
        execute_fn: Any = None,
    ) -> ToolDefinition:
        """Create a mock MCP-style tool for testing.

        Args:
            name: Tool name.
            description: Tool description.
            parameter_schema: JSON schema for parameters.
            execute_fn: Function to call when tool is executed.
        """
        return ToolDefinition(
            name=name,
            description=description,
            parameter_schema=parameter_schema or {},
            execute_fn=execute_fn or (lambda **kwargs: json.dumps(kwargs)),
        )
