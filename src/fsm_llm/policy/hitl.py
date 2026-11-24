"""
Human-in-the-Loop (HITL) manager of the policy layer.

Provides approval gates, confidence-based escalation, and human override
mechanisms: ``HumanInTheLoop`` decides whether a tool call needs approval, asks a
human and escalates. Agents wire it into their FSM through handlers and context
(``fsm_llm.agents``); this module knows nothing about an agent run.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fsm_llm.constants import has_internal_prefix, is_forbidden_context_entry
from fsm_llm.core.tool_observer import begin_approval, end_approval
from fsm_llm.logging import logger
from fsm_llm.tools import (
    ToolCall,
    ToolTimedOut,
    call_with_timeout,
    check_timeout,
    redact_secret_entries,
)

from .approval import (
    ApprovalCallback,
    ApprovalDeniedError,
    ApprovalPolicy,
    ApprovalRequest,
)
from .constants import Defaults, LogMessages

# Type aliases
EscalationCallback = Callable[[str, dict[str, Any]], None]


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
            a finite number in ``(0, Defaults.MAX_TOOL_TIMEOUT_S]`` raises
            ``ToolExecutionError`` here, at construction (``request_approval``
            keeps the same refusal as a backstop; neither is a silent denial
            of every request).
        """
        if approval_timeout is not None:
            check_timeout("approval_callback", approval_timeout)
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
        # DECISION plan-2026-11-24T193347-3f4530d3/D-007
        # The ask is evidence only: `approval/request` BEFORE the callback is asked, `approval/outcome`
        # after, both best effort (a failed write changes no verdict). Do NOT read an event or handle
        # back into a decision, write a grant or the context into an event, or move the request
        # after the callback. See decisions.md D-007.
        handle = begin_approval(
            tool_call.tool_name, tool_call.parameters, has_callback=callback is not None
        )
        if callback is None:
            end_approval(handle, approved=False, reason="no_callback")
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

        reason = None
        try:
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
                    reason = "timeout"
        except BaseException as exc:
            end_approval(
                handle, approved=False, reason="error", error_type=type(exc).__name__
            )
            raise
        reason = reason or ("approved" if approved is True else "denied")
        end_approval(handle, approved=approved is True, reason=reason)

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
