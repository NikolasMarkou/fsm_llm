"""The approval vocabulary of the policy layer.

``ApprovalRequest`` is what an approver is shown for one tool call, ``ApprovalPolicy``
decides whether a call needs approval, ``ApprovalCallback`` asks a human and
``ApprovalDeniedError`` is what a missing or refusing approval raises. The manager
that uses them is :class:`fsm_llm.policy.hitl.HumanInTheLoop`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from fsm_llm.core.definitions import AgentError
from fsm_llm.tools import ToolCall


class ApprovalRequest(BaseModel):
    """A request for human approval before an action."""

    tool_name: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    reasoning: str = ""
    context_summary: dict[str, Any] = Field(default_factory=dict)


# Type aliases
ApprovalCallback = Callable[[ApprovalRequest], bool]
ApprovalPolicy = Callable[[ToolCall, dict[str, Any]], bool]


class ApprovalDeniedError(AgentError):
    """Human denied approval for an agent action."""

    def __init__(self, action_description: str):
        super().__init__(f"Approval denied for action: {action_description}")
        self.action_description = action_description
