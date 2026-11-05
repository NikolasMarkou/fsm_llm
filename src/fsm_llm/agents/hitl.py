"""
Human-in-the-Loop (HITL) support for agent patterns.

Provides approval gates, confidence-based escalation, and human override
mechanisms that integrate with the agent FSM via handlers and context.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fsm_llm.constants import has_internal_prefix, is_forbidden_context_entry
from fsm_llm.logging import logger

from .constants import ContextKeys, Defaults, LogMessages
from .definitions import ApprovalRequest, ToolCall
from .exceptions import ApprovalDeniedError
from .tools import (
    ToolTimedOut,
    call_with_timeout,
    redact_secret_entries,
    tool_call_from_context,
)

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
        approval_timeout: float | None = None,
    ) -> None:
        """
        Initialize HITL manager.

        :param approval_policy: Function that decides if a tool call needs approval.
            Receives (tool_call, context) and returns True if approval is needed.
        :param approval_callback: Function that requests approval from a human.
            Receives ApprovalRequest and returns True if approved.
        :param on_escalation: Callback when the agent escalates to a human.
        :param confidence_threshold: Escalate if agent confidence falls below this.
        :param approval_timeout: Max seconds to wait for approval. None = no limit.
            If exceeded, the tool call is treated as denied. A value that is not
            a finite number in ``(0, Defaults.MAX_TOOL_TIMEOUT_S]`` makes
            ``request_approval`` raise ``ToolExecutionError`` (not a silent
            denial of every request).
        """
        self._approval_policy = approval_policy
        self._approval_timeout = approval_timeout
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

        Returns True if approved, False if denied (a timeout counts as denied).
        Raises ``ApprovalDeniedError`` if no callback is set.
        """
        logger.info(
            LogMessages.APPROVAL_REQUESTED.format(
                action=f"{tool_call.tool_name}"
                f"({redact_secret_entries(tool_call.parameters)})"
            )
        )

        callback = self._approval_callback
        if callback is None:
            raise ApprovalDeniedError(
                "No approval callback configured; cannot request approval"
            )

        # DECISION plan-2026-09-29T103145-06a5ec0a/D-016
        # `parameters` stays RAW: the approver must see the exact call it
        # authorises. Do NOT redact it. The context summary is background, so
        # secret-looking entries are dropped (top level) and redacted (nested).
        request = ApprovalRequest(
            tool_name=tool_call.tool_name,
            parameters=tool_call.parameters,
            reasoning=tool_call.reasoning,
            # DECISION plan-2026-07-20T040150-876e7164/D-003 [STALE]
            # This summary is handed to the approval callback (a human-facing
            # surface). Do NOT re-inline `k.startswith("_")`: it misses
            # `system_`/`internal_`/`__` and is case-sensitive. See D-003.
            context_summary=redact_secret_entries(
                {
                    k: v
                    for k, v in context.items()
                    if not has_internal_prefix(k)
                    and k != "observations"
                    and not is_forbidden_context_entry(k, v)
                }
            ),
        )

        if self._approval_timeout is None:
            approved = callback(request)
        else:
            # The callback runs as a timed tool does: in a copy of the caller's
            # context variables, its exception (SystemExit too) raised here, a
            # late one logged, a bad timeout refused (F-AB-04).
            try:
                approved = call_with_timeout(
                    lambda: callback(request),
                    "approval_callback",
                    self._approval_timeout,
                )
            except ToolTimedOut:
                logger.warning(
                    f"Approval timed out after {self._approval_timeout}s, "
                    "treating as denied"
                )
                approved = False

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


def make_hitl_checker(
    hitl: HumanInTheLoop,
    policy: ApprovalPolicy | None = None,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Create a reusable HITL approval checker handler.

    Returns a handler function that checks whether the current tool call
    requires human approval based on *policy* (default: the HITL policy,
    ``hitl.requires_approval``).  Agents pass their shared approval predicate
    so the gate, the ``await_approval`` state and the refusal agree.  Intended
    to be registered as a ``CONTEXT_UPDATE`` handler on the ``TOOL_NAME`` key.
    """
    decide: ApprovalPolicy = policy if policy is not None else hitl.requires_approval

    def check_approval(context: dict[str, Any]) -> dict[str, Any]:
        tool_name = context.get(ContextKeys.TOOL_NAME)
        if not tool_name or tool_name == ContextKeys.NO_TOOL:
            return {}

        if decide(tool_call_from_context(context), context):
            return {ContextKeys.APPROVAL_REQUIRED: True}

        return {ContextKeys.APPROVAL_REQUIRED: False}

    return check_approval
