from __future__ import annotations

"""Tests for fsm_llm_agents.hitl module."""

from fsm_llm_agents.definitions import ApprovalRequest, ToolCall
from fsm_llm_agents.hitl import HumanInTheLoop


def _always_approve(request: ApprovalRequest) -> bool:
    return True


def _always_deny(request: ApprovalRequest) -> bool:
    return False


def _approve_safe_only(call: ToolCall, ctx: dict) -> bool:
    """Policy: only dangerous tools need approval."""
    return call.tool_name in ("delete", "send_email")


class TestHumanInTheLoop:
    """Tests for HumanInTheLoop class."""

    def test_default_no_policy(self):
        hitl = HumanInTheLoop()
        call = ToolCall(tool_name="search", parameters={})
        assert hitl.requires_approval(call, {}) is False

    def test_has_approval_policy(self):
        hitl = HumanInTheLoop(approval_policy=_approve_safe_only)
        assert hitl.has_approval_policy is True

    def test_no_approval_policy(self):
        hitl = HumanInTheLoop()
        assert hitl.has_approval_policy is False

    def test_has_approval_callback(self):
        hitl = HumanInTheLoop(approval_callback=_always_approve)
        assert hitl.has_approval_callback is True

    def test_requires_approval_with_policy(self):
        hitl = HumanInTheLoop(approval_policy=_approve_safe_only)

        safe_call = ToolCall(tool_name="search", parameters={})
        assert hitl.requires_approval(safe_call, {}) is False

        dangerous_call = ToolCall(tool_name="delete", parameters={})
        assert hitl.requires_approval(dangerous_call, {}) is True

    def test_request_approval_auto_approve(self):
        """No callback means auto-approve."""
        hitl = HumanInTheLoop()
        call = ToolCall(tool_name="search", parameters={})
        assert hitl.request_approval(call, {}) is True

    def test_request_approval_approved(self):
        hitl = HumanInTheLoop(approval_callback=_always_approve)
        call = ToolCall(tool_name="delete", parameters={})
        assert hitl.request_approval(call, {}) is True

    def test_request_approval_denied(self):
        hitl = HumanInTheLoop(approval_callback=_always_deny)
        call = ToolCall(tool_name="delete", parameters={})
        assert hitl.request_approval(call, {}) is False

    def test_escalation_callback(self):
        escalated = []

        def on_escalation(reason, ctx):
            escalated.append((reason, ctx))

        hitl = HumanInTheLoop(on_escalation=on_escalation)
        hitl.escalate("Low confidence", {"task": "test"})

        assert len(escalated) == 1
        assert escalated[0][0] == "Low confidence"

    def test_escalation_no_callback(self):
        """Escalation without callback should not raise."""
        hitl = HumanInTheLoop()
        hitl.escalate("test", {})  # Should not raise

    def test_confidence_threshold(self):
        hitl = HumanInTheLoop(confidence_threshold=0.5)
        assert hitl.should_escalate_on_confidence(0.3) is True
        assert hitl.should_escalate_on_confidence(0.5) is False
        assert hitl.should_escalate_on_confidence(0.8) is False

    def test_default_confidence_threshold(self):
        hitl = HumanInTheLoop()
        assert hitl.confidence_threshold == 0.3

    def test_approval_request_context_filtering(self):
        """Approval request should filter internal keys."""
        hitl = HumanInTheLoop(approval_callback=_always_approve)
        call = ToolCall(tool_name="send_email", parameters={"to": "user@test.com"})
        ctx = {
            "task": "Send email",
            "_internal": "should be filtered",
            "observations": ["long list"],
        }
        # Should not raise
        result = hitl.request_approval(call, ctx)
        assert result is True
