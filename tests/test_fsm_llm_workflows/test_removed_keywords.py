"""A removed keyword spelling is refused, never silently dropped.

Pydantic ignores unknown keywords by default, so before the models set
``extra="forbid"`` a caller still passing ``error_state=`` got a step with no
error route and ``timeout=`` a step with no time limit. One test per removed
spelling: the old keyword raises (a ``ValidationError`` is a ``ValueError``) and
the new keyword works.
"""

from __future__ import annotations

import pytest

from fsm_llm.workflows.models import (
    EventListener,
    WaitEventConfig,
    WorkflowInstance,
    WorkflowStepResult,
)
from fsm_llm.workflows.steps import AutoStep, SwitchStep


class TestRemovedKeywordsRaise:
    def test_auto_step_error_state(self):
        with pytest.raises(ValueError, match="error_state"):
            AutoStep(step_id="a", name="A", next_step_id="b", error_state="err")
        step = AutoStep(step_id="a", name="A", next_step_id="b", error_step_id="err")
        assert step.error_step_id == "err"

    def test_auto_step_timeout(self):
        with pytest.raises(ValueError, match="timeout"):
            AutoStep(step_id="a", name="A", next_step_id="b", timeout=0.01)
        step = AutoStep(step_id="a", name="A", next_step_id="b", timeout_seconds=0.01)
        assert step.timeout_seconds == 0.01

    def test_switch_step_default_state(self):
        with pytest.raises(ValueError, match="default_state"):
            SwitchStep(
                step_id="s", name="S", key="k", cases={"x": "y"}, default_state="d"
            )
        step = SwitchStep(
            step_id="s", name="S", key="k", cases={"x": "y"}, default_step_id="d"
        )
        assert step.default_step_id == "d"

    def test_wait_event_config_timeout_state(self):
        with pytest.raises(ValueError, match="timeout_state"):
            WaitEventConfig(
                event_type="e",
                success_step_id="ok",
                timeout_seconds=5,
                timeout_state="t",
            )
        config = WaitEventConfig(
            event_type="e",
            success_step_id="ok",
            timeout_seconds=5,
            timeout_step_id="t",
        )
        assert config.timeout_step_id == "t"

    def test_wait_event_config_success_state(self):
        with pytest.raises(ValueError, match="success_state"):
            WaitEventConfig(event_type="e", success_step_id="ok", success_state="old")

    def test_event_listener_success_state(self):
        with pytest.raises(ValueError, match="success_state"):
            EventListener(instance_id="i", success_step_id="ok", success_state="old")
        listener = EventListener(instance_id="i", success_step_id="ok")
        assert listener.success_step_id == "ok"

    def test_workflow_step_result_next_state(self):
        with pytest.raises(ValueError, match="next_state"):
            WorkflowStepResult(success=True, next_state="b")
        result = WorkflowStepResult(success=True, next_step_id="b")
        assert result.next_step_id == "b"

    def test_workflow_instance_workflow_timeout(self):
        with pytest.raises(ValueError, match="workflow_timeout"):
            WorkflowInstance(
                instance_id="i",
                workflow_id="w",
                current_step_id="s",
                workflow_timeout=5.0,
            )
        instance = WorkflowInstance(
            instance_id="i", workflow_id="w", current_step_id="s", max_seconds=5.0
        )
        assert instance.max_seconds == 5.0
