from __future__ import annotations

"""
Exception hierarchy for the agents package.
"""

from typing import Any

from fsm_llm.definitions import FSMError


class AgentError(FSMError):
    """Base exception for all agent-related errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, details=details)


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


class ToolValidationError(AgentError):
    """Tool parameter validation failed."""

    def __init__(self, tool_name: str, reason: str):
        super().__init__(f"Validation failed for tool '{tool_name}': {reason}")
        self.tool_name = tool_name


class BudgetExhaustedError(AgentError):
    """Agent exceeded its iteration/token/time budget."""

    def __init__(self, budget_type: str, limit: int | float):
        super().__init__(
            f"Agent budget exhausted: {budget_type} limit ({limit}) reached"
        )
        self.budget_type = budget_type
        self.limit = limit


class ApprovalDeniedError(AgentError):
    """Human denied approval for an agent action."""

    def __init__(self, action_description: str):
        super().__init__(f"Approval denied for action: {action_description}")
        self.action_description = action_description


class AgentTimeoutError(AgentError):
    """Agent exceeded its time budget."""

    def __init__(self, timeout_seconds: float):
        super().__init__(f"Agent timed out after {timeout_seconds:.1f} seconds")
        self.timeout_seconds = timeout_seconds
