from __future__ import annotations

"""
Human-in-the-Loop (HITL) support for agent patterns.

Provides approval gates, confidence-based escalation, and human override
mechanisms that integrate with the agent FSM via handlers and context.
"""

from collections.abc import Callable
from typing import Any

from fsm_llm.logging import logger

from .constants import Defaults, LogMessages
from .definitions import ApprovalRequest, ToolCall
from .exceptions import AgentError

# Type aliases
ApprovalCallback = Callable[[ApprovalRequest], bool]
EscalationCallback = Callable[[str, dict[str, Any]], None]
ApprovalPolicy = Callable[[ToolCall, dict[str, Any]], bool]


class HumanInTheLoop:
    """
    Human-in-the-Loop manager for agent approval and escalation.

    Integrates with ReactAgent to gate tool executions that require
    human approval, escalate when confidence is low, and allow
    human overrides.

    Usage::

        hitl = HumanInTheLoop(
            approval_policy=lambda call, ctx: call.tool_name in ["send_email"],
            approval_callback=my_approval_handler,
            confidence_threshold=0.3,
        )
        agent = ReactAgent(tools=registry, hitl=hitl)
    """

    def __init__(
        self,
        approval_policy: ApprovalPolicy | None = None,
        approval_callback: ApprovalCallback | None = None,
        on_escalation: EscalationCallback | None = None,
        confidence_threshold: float = Defaults.CONFIDENCE_THRESHOLD,
    ) -> None:
        """
        Initialize HITL manager.

        :param approval_policy: Function that decides if a tool call needs approval.
            Receives (tool_call, context) and returns True if approval is needed.
        :param approval_callback: Function that requests approval from a human.
            Receives ApprovalRequest and returns True if approved.
        :param on_escalation: Callback when the agent escalates to a human.
        :param confidence_threshold: Escalate if agent confidence falls below this.
        """
        self._approval_policy = approval_policy
        self._approval_callback = approval_callback
        self._on_escalation = on_escalation
        self.confidence_threshold = confidence_threshold

    def requires_approval(self, tool_call: ToolCall, context: dict[str, Any]) -> bool:
        """Check if a tool call requires human approval."""
        if self._approval_policy is None:
            return False
        return self._approval_policy(tool_call, context)

    def request_approval(
        self,
        tool_call: ToolCall,
        context: dict[str, Any],
    ) -> bool:
        """
        Request approval from a human.

        Returns True if approved, False if denied.
        If no callback is set, defaults to auto-approve.
        """
        logger.info(
            LogMessages.APPROVAL_REQUESTED.format(
                action=f"{tool_call.tool_name}({tool_call.parameters})"
            )
        )

        if self._approval_callback is None:
            raise AgentError(
                "No approval callback configured — cannot approve tool call. "
                "Set an approval_callback on HumanInTheLoop before using approval gates."
            )

        request = ApprovalRequest(
            tool_name=tool_call.tool_name,
            parameters=tool_call.parameters,
            reasoning=tool_call.reasoning,
            context_summary={
                k: v
                for k, v in context.items()
                if not k.startswith("_") and k != "observations"
            },
        )

        approved = self._approval_callback(request)

        logger.info(
            LogMessages.APPROVAL_RESULT.format(
                result="APPROVED" if approved else "DENIED",
                action=f"{tool_call.tool_name}",
            )
        )

        return approved

    def escalate(self, reason: str, context: dict[str, Any]) -> None:
        """Escalate to a human with a reason and context."""
        logger.info(LogMessages.ESCALATION.format(reason=reason))

        if self._on_escalation is not None:
            self._on_escalation(reason, context)

    def should_escalate_on_confidence(self, confidence: float) -> bool:
        """Check if confidence is low enough to warrant escalation."""
        return confidence < self.confidence_threshold

    @property
    def has_approval_policy(self) -> bool:
        """Whether an approval policy is configured."""
        return self._approval_policy is not None

    @property
    def has_approval_callback(self) -> bool:
        """Whether an approval callback is configured."""
        return self._approval_callback is not None
