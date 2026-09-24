"""Stdio MCP server used by ``test_mcp_stdio.py`` (not a test module).

Usage: ``python mcp_fixture_server.py <pid_file> [--hang]``

Writes its own PID to ``pid_file`` on start. ``--hang`` sleeps before
serving, so the client's ``initialize`` never gets an answer. Tools:
``add`` (fast) and ``slow`` (sleeps far longer than any test timeout).
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

try:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server
except ImportError:  # mcp 2.x renamed FastMCP to MCPServer
    from mcp.server.mcpserver import MCPServer as _Server

SLOW_TOOL_SECONDS = 120

server = _Server("fsm-llm-test-fixture")


@server.tool()
def add(a: int, b: int) -> str:
    """Add two integers."""
    return str(a + b)


@server.tool()
async def slow(seconds: int = SLOW_TOOL_SECONDS) -> str:
    """Sleep for a long time, then answer."""
    await asyncio.sleep(seconds)
    return "done"


def main(argv: list[str]) -> None:
    with open(argv[1], "w", encoding="utf-8") as fh:
        fh.write(str(os.getpid()))
    if "--hang" in argv[2:]:
        time.sleep(SLOW_TOOL_SECONDS)
    server.run(transport="stdio")


if __name__ == "__main__":
    main(sys.argv)
