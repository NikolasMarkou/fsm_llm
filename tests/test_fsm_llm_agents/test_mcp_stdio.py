"""MCPToolProvider against a real stdio MCP server (``mcp_fixture_server.py``).

Skipped when ``mcp`` is not installed, which includes CI and the project
``.venv`` (D-014, D-025 of plan-2026-09-24T091842-c1d5bfbc: installing mcp
upgrades shared dependencies, so these run only in dedicated venvs).
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from fsm_llm_agents.definitions import ToolCall
from fsm_llm_agents.exceptions import AgentTimeoutError
from fsm_llm_agents.mcp import MCPToolProvider
from fsm_llm_agents.tools import ToolRegistry

_FIXTURE = Path(__file__).with_name("mcp_fixture_server.py")
# Discovery spawns a Python child that imports mcp; keep it well clear of that.
_DISCOVERY_TIMEOUT = 30.0
_EXIT_GRACE_SECONDS = 5.0


def _provider(pid_file: Path, timeout: float, *extra: str) -> MCPToolProvider:
    return MCPToolProvider.from_stdio(
        sys.executable, [str(_FIXTURE), str(pid_file), *extra], timeout=timeout
    )


def _read_pid(pid_file: Path) -> int:
    deadline = time.monotonic() + _EXIT_GRACE_SECONDS
    while time.monotonic() < deadline:
        text = pid_file.read_text(encoding="utf-8") if pid_file.exists() else ""
        if text.strip():
            return int(text)
        time.sleep(0.05)
    raise AssertionError(f"fixture server never wrote {pid_file}")


def _gone_within(pid: int, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


class TestMCPStdio:
    def test_discover_register_and_call(self, tmp_path: Path) -> None:
        provider = _provider(tmp_path / "pid", _DISCOVERY_TIMEOUT)
        tools = asyncio.run(provider.discover_tools())
        assert sorted(t.name for t in tools) == ["add", "slow"]
        # mcp 2.x names the field input_schema (D-025); both must yield it.
        add_tool = next(t for t in tools if t.name == "add")
        assert add_tool.parameter_schema["required"] == ["a", "b"]

        registry = ToolRegistry()
        assert provider.register_tools(registry) == 2

        result = registry.execute(
            ToolCall(tool_name="add", parameters={"a": 2, "b": 3})
        )
        assert result.success, result.error
        assert result.result == "5"

    def test_slow_call_times_out_and_child_exits(self, tmp_path: Path) -> None:
        pid_file = tmp_path / "pid"
        # Each call spawns its own server, so the timeout covers server start;
        # 8 s leaves room for start while staying far under the 120 s sleep.
        provider = _provider(pid_file, 8.0)
        asyncio.run(provider.discover_tools())
        registry = ToolRegistry()
        provider.register_tools(registry)
        pid_file.unlink()

        result = registry.execute(ToolCall(tool_name="slow", parameters={}))

        assert not result.success
        assert "timed out" in (result.error or "")
        pid = _read_pid(pid_file)
        assert _gone_within(pid, _EXIT_GRACE_SECONDS), f"MCP child {pid} still alive"

    def test_hung_server_discovery_raises_agent_timeout(self, tmp_path: Path) -> None:
        pid_file = tmp_path / "pid"
        provider = _provider(pid_file, 5.0, "--hang")

        with pytest.raises(AgentTimeoutError):
            asyncio.run(provider.discover_tools())

        pid = _read_pid(pid_file)
        assert _gone_within(pid, _EXIT_GRACE_SECONDS), f"MCP child {pid} still alive"
