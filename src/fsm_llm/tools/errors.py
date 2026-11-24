"""The errors of the tool layer: a failed tool run, a bind failure, and a tool that is not registered."""

from __future__ import annotations

from typing import Any

from fsm_llm.core.definitions import AgentError


class ToolExecutionError(AgentError):
    """Error during tool execution."""

    def __init__(self, message: str, tool_name: str | None = None, **kwargs: Any):
        super().__init__(message, **kwargs)
        self.tool_name = tool_name


class ToolArgumentError(ToolExecutionError):
    """The arguments of a call did not bind to the tool function (the body never ran).

    Raised by the argument binder of ``ToolRegistry``; its text is the bind error's text, so a
    model sees the same message as before. The registry maps it to ``ToolErrorCode.INVALID_ARGS``.
    """


class ToolNotFoundError(AgentError):
    """Requested tool does not exist in the registry."""

    def __init__(self, tool_name: str):
        super().__init__(f"Tool not found: {tool_name}")
        self.tool_name = tool_name
