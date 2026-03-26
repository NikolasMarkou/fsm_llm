"""
Basic tests for the fsm_llm_workflows extension package.
Tests models, exceptions, definitions, and DSL without requiring
async infrastructure.
"""

from datetime import datetime, timedelta, timezone

import pytest

# ----------------------------------------------------------------
# Exception tests
# ----------------------------------------------------------------


class TestWorkflowExceptions:
    """Test exception hierarchy and attributes."""

    def test_workflow_error_base(self):
        from fsm_llm_workflows.exceptions import WorkflowError

        err = WorkflowError("test error", details={"key": "val"})
        assert str(err) == "test error"
        assert err.details == {"key": "val"}

    def test_workflow_error_default_details(self):
        from fsm_llm_workflows.exceptions import WorkflowError

        err = WorkflowError("test")
        assert err.details == {}

    def test_workflow_definition_error(self):
        from fsm_llm_workflows.exceptions import WorkflowDefinitionError

        err = WorkflowDefinitionError("wf-1", "missing step")
        assert "wf-1" in str(err)
        assert "missing step" in str(err)
        assert err.workflow_id == "wf-1"

    def test_workflow_step_error_with_cause(self):
        from fsm_llm_workflows.exceptions import WorkflowStepError

        cause = ValueError("bad value")
        err = WorkflowStepError("step-1", "failed", cause=cause)
        assert err.step_id == "step-1"
        assert err.cause is cause
        assert err.details["cause"] == "bad value"
        assert err.details["cause_type"] == "ValueError"

    def test_workflow_instance_error(self):
        from fsm_llm_workflows.exceptions import WorkflowInstanceError

        err = WorkflowInstanceError("inst-1", "not found")
        assert err.instance_id == "inst-1"
        assert "inst-1" in str(err)

    def test_workflow_timeout_error(self):
        from fsm_llm_workflows.exceptions import WorkflowTimeoutError

        err = WorkflowTimeoutError("process", 30)
        assert err.operation == "process"
        assert err.timeout_seconds == 30
        assert err.details["timeout_seconds"] == 30

    def test_workflow_validation_error(self):
        from fsm_llm_workflows.exceptions import WorkflowValidationError

        err = WorkflowValidationError(["error 1", "error 2"])
        assert len(err.validation_errors) == 2
        assert "2 error(s)" in str(err)

    def test_workflow_state_error(self):
        from fsm_llm_workflows.exceptions import WorkflowStateError

        err = WorkflowStateError("running", "cancel", "cannot cancel")
        assert err.current_state == "running"
        assert err.operation == "cancel"

    def test_workflow_event_error(self):
        from fsm_llm_workflows.exceptions import WorkflowEventError

        err = WorkflowEventError("click", "unhandled")
        assert err.event_type == "click"

    def test_workflow_resource_error(self):
        from fsm_llm_workflows.exceptions import WorkflowResourceError

        err = WorkflowResourceError("timer", "t-1", "expired")
        assert err.resource_type == "timer"
        assert err.resource_id == "t-1"

    def test_exception_hierarchy(self):
        from fsm_llm_workflows.exceptions import (
            WorkflowDefinitionError,
            WorkflowError,
            WorkflowEventError,
            WorkflowInstanceError,
            WorkflowResourceError,
            WorkflowStateError,
            WorkflowStepError,
            WorkflowTimeoutError,
            WorkflowValidationError,
        )

        # All should be subclasses of WorkflowError
        for exc_cls in [
            WorkflowDefinitionError,
            WorkflowStepError,
            WorkflowInstanceError,
            WorkflowTimeoutError,
            WorkflowValidationError,
            WorkflowStateError,
            WorkflowEventError,
            WorkflowResourceError,
        ]:
            assert issubclass(exc_cls, WorkflowError)
            assert issubclass(exc_cls, Exception)


# ----------------------------------------------------------------
# Model tests
# ----------------------------------------------------------------


class TestWorkflowModels:
    """Test Pydantic models."""

    def test_workflow_status_enum(self):
        from fsm_llm_workflows.models import WorkflowStatus

        assert WorkflowStatus.PENDING.value == "pending"
        assert WorkflowStatus.RUNNING.value == "running"
        assert WorkflowStatus.COMPLETED.value == "completed"
        assert WorkflowStatus.FAILED.value == "failed"
        assert WorkflowStatus.CANCELLED.value == "cancelled"

    def test_workflow_event_creation(self):
        from fsm_llm_workflows.models import WorkflowEvent

        event = WorkflowEvent(event_type="user_click", payload={"x": 10})
        assert event.event_type == "user_click"
        assert event.payload == {"x": 10}
        assert event.event_id  # auto-generated UUID
        assert isinstance(event.timestamp, datetime)

    def test_workflow_event_serialization(self):
        from fsm_llm_workflows.models import WorkflowEvent

        event = WorkflowEvent(event_type="test")
        data = event.model_dump()
        assert isinstance(data["timestamp"], str)  # datetime -> ISO string

    def test_step_result_success(self):
        from fsm_llm_workflows.models import WorkflowStepResult

        result = WorkflowStepResult.success_result(
            data={"key": "val"}, next_state="done", message="ok"
        )
        assert result.success is True
        assert result.data == {"key": "val"}
        assert result.next_state == "done"
        assert result.error is None

    def test_step_result_failure(self):
        from fsm_llm_workflows.models import WorkflowStepResult

        result = WorkflowStepResult.failure_result(
            error="something broke", next_state="error_state"
        )
        assert result.success is False
        assert result.error == "something broke"
        assert result.next_state == "error_state"
        assert "failed" in result.message.lower()

    def test_step_result_exception_conversion(self):
        from fsm_llm_workflows.models import WorkflowStepResult

        result = WorkflowStepResult(success=False, error=ValueError("bad"))
        assert result.error == "bad"  # Exception converted to string

    def test_workflow_instance_lifecycle(self):
        from fsm_llm_workflows.models import WorkflowInstance, WorkflowStatus

        instance = WorkflowInstance(
            instance_id="i-1", workflow_id="w-1", current_step_id="start"
        )
        assert instance.status == WorkflowStatus.PENDING
        assert instance.is_active() is False
        assert instance.is_terminal() is False

        instance.update_status(WorkflowStatus.RUNNING)
        assert instance.is_active() is True
        assert instance.is_terminal() is False

        instance.update_status(WorkflowStatus.COMPLETED)
        assert instance.is_terminal() is True
        assert instance.completed_at is not None
        assert len(instance.history) == 2

    def test_workflow_instance_error_tracking(self):
        from fsm_llm_workflows.models import WorkflowInstance, WorkflowStatus

        instance = WorkflowInstance(
            instance_id="i-1", workflow_id="w-1", current_step_id="start"
        )
        instance.update_status(WorkflowStatus.FAILED, error=RuntimeError("crash"))
        assert instance.error == "crash"
        assert instance.is_terminal() is True

    def test_event_listener_expiry(self):
        from fsm_llm_workflows.models import EventListener

        # Not expired
        listener = EventListener(
            instance_id="i-1",
            success_state="done",
            timeout_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        assert listener.is_expired() is False

        # Expired
        listener_expired = EventListener(
            instance_id="i-1",
            success_state="done",
            timeout_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        assert listener_expired.is_expired() is True

        # No timeout
        listener_no_timeout = EventListener(instance_id="i-1", success_state="done")
        assert listener_no_timeout.is_expired() is False

    def test_wait_event_config_validation(self):
        from fsm_llm_workflows.models import WaitEventConfig

        # Valid
        config = WaitEventConfig(
            event_type="click", success_state="done", timeout_seconds=30
        )
        assert config.timeout_seconds == 30

        # Invalid timeout
        with pytest.raises(ValueError, match="positive"):
            WaitEventConfig(
                event_type="click", success_state="done", timeout_seconds=-1
            )


# ----------------------------------------------------------------
# Definition tests
# ----------------------------------------------------------------


class TestWorkflowDefinition:
    """Test workflow definition and validation."""

    def test_create_definition(self):
        from fsm_llm_workflows.definitions import WorkflowDefinition

        wf = WorkflowDefinition(workflow_id="wf-1", name="Test Workflow")
        assert wf.workflow_id == "wf-1"
        assert wf.steps == {}

    def test_add_step(self):
        from fsm_llm_workflows.definitions import WorkflowDefinition
        from fsm_llm_workflows.steps import AutoTransitionStep

        wf = WorkflowDefinition(workflow_id="wf-1", name="Test")
        step = AutoTransitionStep(step_id="s1", name="Step 1", next_state="s2")
        wf.with_step(step, is_initial=True)
        assert "s1" in wf.steps
        assert wf.initial_step_id == "s1"

    def test_validate_empty_workflow(self):
        from fsm_llm_workflows.definitions import WorkflowDefinition
        from fsm_llm_workflows.exceptions import WorkflowValidationError

        wf = WorkflowDefinition(workflow_id="wf-1", name="Test")
        with pytest.raises(WorkflowValidationError):
            wf.validate()

    def test_validate_missing_initial_step(self):
        from fsm_llm_workflows.definitions import WorkflowDefinition
        from fsm_llm_workflows.exceptions import WorkflowValidationError
        from fsm_llm_workflows.steps import AutoTransitionStep

        wf = WorkflowDefinition(workflow_id="wf-1", name="Test")
        step = AutoTransitionStep(step_id="s1", name="S1", next_state="end")
        wf.with_step(step)
        with pytest.raises(WorkflowValidationError):
            wf.validate()


# ----------------------------------------------------------------
# DSL tests
# ----------------------------------------------------------------


class TestWorkflowDSL:
    """Test DSL helper functions."""

    def test_create_workflow(self):
        from fsm_llm_workflows.dsl import create_workflow

        wf = create_workflow("wf-1", "Test", "A test workflow")
        assert wf.workflow_id == "wf-1"
        assert wf.name == "Test"
        assert wf.description == "A test workflow"

    def test_auto_step(self):
        from fsm_llm_workflows.dsl import auto_step

        step = auto_step("s1", "Step 1", next_state="s2")
        assert step.step_id == "s1"
        assert step.next_state == "s2"

    def test_condition_step(self):
        from fsm_llm_workflows.dsl import condition_step

        step = condition_step(
            "s1",
            "Check",
            condition=lambda ctx: True,
            true_state="yes",
            false_state="no",
        )
        assert step.step_id == "s1"
        assert step.true_state == "yes"
        assert step.false_state == "no"

    def test_fluent_workflow_building(self):
        from fsm_llm_workflows.dsl import auto_step, create_workflow

        wf = (
            create_workflow("wf-1", "Pipeline")
            .with_initial_step(auto_step("start", "Start", next_state="end"))
            .with_step(auto_step("end", "End", next_state="end"))
        )
        assert wf.initial_step_id == "start"
        assert len(wf.steps) == 2


# ----------------------------------------------------------------
# Package import tests
# ----------------------------------------------------------------


class TestPackageImports:
    """Test that the package exports are correct."""

    def test_import_main_package(self):
        import fsm_llm_workflows

        assert hasattr(fsm_llm_workflows, "__version__")
        assert hasattr(fsm_llm_workflows, "__all__")

    def test_import_all_exports(self):
        import fsm_llm_workflows

        for name in fsm_llm_workflows.__all__:
            assert hasattr(fsm_llm_workflows, name), f"Missing export: {name}"

    def test_version_matches_main_package(self):
        import fsm_llm
        import fsm_llm_workflows

        assert fsm_llm_workflows.__version__ == fsm_llm.__version__
