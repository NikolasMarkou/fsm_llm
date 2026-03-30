from __future__ import annotations

"""
MCP (Model Context Protocol) Tool Integration.

Connects to MCP servers and converts their tool definitions into
ToolDefinition objects compatible with ToolRegistry.
"""

import json
from typing import Any

from fsm_llm.logging import logger

from .definitions import ToolDefinition

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
    """

    def __init__(
        self,
        server_params: Any | None = None,
        server_url: str | None = None,
    ) -> None:
        _require_mcp()
        self._server_params = server_params
        self._server_url = server_url
        self._tools: list[ToolDefinition] = []
        self._session: Any | None = None

    @classmethod
    def from_stdio(
        cls,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> MCPToolProvider:
        """Create a provider that connects via stdio transport.

        Args:
            command: The command to run the MCP server.
            args: Arguments to pass to the command.
            env: Environment variables for the server process.
        """
        _require_mcp()
        params = StdioServerParameters(
            command=command,
            args=args or [],
            env=env,
        )
        return cls(server_params=params)

    @classmethod
    def from_url(cls, url: str) -> MCPToolProvider:
        """Create a provider that connects via HTTP/SSE transport.

        Args:
            url: The URL of the MCP server.
        """
        _require_mcp()
        return cls(server_url=url)

    async def discover_tools(self) -> list[ToolDefinition]:
        """Connect to the MCP server and discover available tools.

        Returns a list of ToolDefinition objects.
        """
        _require_mcp()

        if self._server_params is not None:
            return await self._discover_stdio()
        elif self._server_url is not None:
            return await self._discover_http()
        else:
            raise ValueError("No server_params or server_url configured")

    async def _discover_stdio(self) -> list[ToolDefinition]:
        """Discover tools via stdio transport."""
        tools: list[ToolDefinition] = []
        async with stdio_client(self._server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                for mcp_tool in result.tools:
                    tool_def = self._convert_mcp_tool(mcp_tool, session)
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
                    tool_def = self._convert_mcp_tool(mcp_tool, session)
                    tools.append(tool_def)
        self._tools = tools
        logger.info(f"Discovered {len(tools)} tools from MCP HTTP server")
        return tools

    def _convert_mcp_tool(self, mcp_tool: Any, session: Any) -> ToolDefinition:
        """Convert an MCP tool object into a ToolDefinition."""
        input_schema = {}
        if hasattr(mcp_tool, "inputSchema") and mcp_tool.inputSchema:
            input_schema = (
                mcp_tool.inputSchema
                if isinstance(mcp_tool.inputSchema, dict)
                else mcp_tool.inputSchema.model_dump()
            )

        param_schema = _mcp_schema_to_parameter_schema(input_schema)

        def make_executor(tool_name: str, sess: Any):
            """Create an executor closure for an MCP tool."""

            async def execute(**kwargs: Any) -> str:
                result = await sess.call_tool(tool_name, arguments=kwargs)
                if hasattr(result, "content") and result.content:
                    parts = []
                    for item in result.content:
                        if hasattr(item, "text"):
                            parts.append(item.text)
                        else:
                            parts.append(str(item))
                    return "\n".join(parts)
                return str(result)

            return execute

        return ToolDefinition(
            name=mcp_tool.name,
            description=getattr(mcp_tool, "description", "") or f"MCP tool: {mcp_tool.name}",
            parameter_schema=param_schema,
            execute_fn=make_executor(mcp_tool.name, session),
        )

    def register_tools(self, registry: Any) -> int:
        """Register all discovered tools with a ToolRegistry.

        Must call discover_tools() first, or use register_tools_sync().

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
