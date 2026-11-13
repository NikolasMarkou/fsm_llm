"""The two errors of the tool layer: a failed tool run and a tool that is not registered."""

from __future__ import annotations

from typing import Any

from fsm_llm.core.definitions import AgentError


class ToolExecutionError(AgentError):
    """Error during tool execution."""

    def __init__(self, message: str, tool_name: str | None = None, **kwargs: Any):
        super().__init__(message, **kwargs)
        self.tool_name = tool_name


class ToolNotFoundError(AgentError):
    """Requested tool does not exist in the registry."""

    def __init__(self, tool_name: str):
        super().__init__(f"Tool not found: {tool_name}")
        self.tool_name = tool_name
