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

    def test_request_approval_no_callback_raises(self):
        """No callback raises AgentError instead of silently auto-approving."""
        import pytest

        from fsm_llm_agents.exceptions import AgentError

        hitl = HumanInTheLoop()
        call = ToolCall(tool_name="search", parameters={})
        with pytest.raises(AgentError, match="No approval callback configured"):
            hitl.request_approval(call, {})

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


class TestApprovalTimeoutCallbackErrors:
    """CR-05: a callback exception after the timeout is logged, not lost."""

    @staticmethod
    def _capture_warnings():
        from fsm_llm.logging import logger

        # Library logging is opt-in (logger.disable("fsm_llm") at import).
        logger.enable("fsm_llm")
        captured: list[str] = []
        sink_id = logger.add(lambda msg: captured.append(str(msg)), level="WARNING")
        return logger, captured, sink_id

    def test_exception_before_timeout_propagates(self):
        import pytest

        def _boom(request: ApprovalRequest) -> bool:
            raise RuntimeError("early failure")

        hitl = HumanInTheLoop(approval_callback=_boom, approval_timeout=5.0)
        call = ToolCall(tool_name="delete", parameters={})
        with pytest.raises(RuntimeError, match="early failure"):
            hitl.request_approval(call, {})

    def test_late_exception_after_timeout_is_logged(self):
        import threading
        import time

        threads: list[threading.Thread] = []

        def _slow_boom(request: ApprovalRequest) -> bool:
            threads.append(threading.current_thread())
            time.sleep(0.3)
            raise RuntimeError("late failure")

        hitl = HumanInTheLoop(approval_callback=_slow_boom, approval_timeout=0.05)
        call = ToolCall(tool_name="delete", parameters={})
        logger, captured, sink_id = self._capture_warnings()
        try:
            assert hitl.request_approval(call, {}) is False
            assert threads, "callback thread never started"
            threads[0].join(timeout=5.0)
            assert not threads[0].is_alive()
        finally:
            logger.remove(sink_id)
            logger.disable("fsm_llm")

        late = [m for m in captured if "late failure" in m]
        assert late, captured
        assert "RuntimeError" in late[0]

    def test_late_return_after_timeout_logs_no_error(self):
        import threading
        import time

        threads: list[threading.Thread] = []

        def _slow_ok(request: ApprovalRequest) -> bool:
            threads.append(threading.current_thread())
            time.sleep(0.3)
            return True

        hitl = HumanInTheLoop(approval_callback=_slow_ok, approval_timeout=0.05)
        call = ToolCall(tool_name="delete", parameters={})
        logger, captured, sink_id = self._capture_warnings()
        try:
            assert hitl.request_approval(call, {}) is False
            threads[0].join(timeout=5.0)
        finally:
            logger.remove(sink_id)
            logger.disable("fsm_llm")

        assert not any("raised after" in m for m in captured), captured
