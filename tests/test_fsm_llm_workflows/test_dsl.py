"""
Unit tests for the workflow DSL factory functions and WorkflowBuilder.
"""
import pytest

from fsm_llm_workflows.definitions import WorkflowDefinition
from fsm_llm_workflows.dsl import (
    api_step,
    auto_step,
    condition_step,
    conditional_workflow,
    create_workflow,
    event_driven_workflow,
    linear_workflow,
    llm_step,
    parallel_step,
    timer_step,
    wait_event_step,
    workflow_builder,
)
from fsm_llm_workflows.steps import (
    APICallStep,
    AutoTransitionStep,
    ConditionStep,
    LLMProcessingStep,
    ParallelStep,
    TimerStep,
    WaitForEventStep,
)


class TestCreateWorkflow:
    """Test create_workflow factory."""

    def test_basic(self):
        wf = create_workflow("wf-1", "My Workflow", "desc")
        assert isinstance(wf, WorkflowDefinition)
        assert wf.workflow_id == "wf-1"
        assert wf.name == "My Workflow"
        assert wf.description == "desc"

    def test_default_description(self):
        wf = create_workflow("wf-1", "Test")
        assert wf.description == ""


class TestAutoStep:
    """Test auto_step factory."""

    def test_basic(self):
        step = auto_step("s1", "Step 1", "next_state")
        assert isinstance(step, AutoTransitionStep)
        assert step.step_id == "s1"
        assert step.name == "Step 1"
        assert step.next_state == "next_state"
        assert step.action is None

    def test_with_action(self):
        def fn(ctx):
            return {"result": True}
        step = auto_step("s1", "Step 1", "next", action=fn)
        assert step.action is fn

    def test_with_description(self):
        step = auto_step("s1", "Step 1", "next", description="does stuff")
        assert step.description == "does stuff"


class TestApiStep:
    """Test api_step factory."""

    def test_basic(self):
        def fn(**kwargs):
            return {"ok": True}
        step = api_step("s1", "API Call", fn, "success", "failure")
        assert isinstance(step, APICallStep)
        assert step.success_state == "success"
        assert step.failure_state == "failure"
        assert step.input_mapping == {}
        assert step.output_mapping == {}

    def test_with_mappings(self):
        def fn(**kwargs):
            return {}
        step = api_step(
            "s1", "API Call", fn, "ok", "err",
            input_mapping={"param": "ctx_key"},
            output_mapping={"result": "resp_key"},
        )
        assert step.input_mapping == {"param": "ctx_key"}
        assert step.output_mapping == {"result": "resp_key"}


class TestConditionStep:
    """Test condition_step factory."""

    def test_basic(self):
        def fn(ctx):
            return ctx.get("ready", False)
        step = condition_step("s1", "Check", fn, "yes", "no")
        assert isinstance(step, ConditionStep)
        assert step.true_state == "yes"
        assert step.false_state == "no"


class TestLlmStep:
    """Test llm_step factory."""

    def test_basic(self):
        mock_llm = object()
        step = llm_step(
            "s1", "LLM Process", mock_llm,
            prompt_template="Hello {name}",
            context_mapping={"name": "user_name"},
            output_mapping={"response": "(.*)"},
            next_state="done",
        )
        assert isinstance(step, LLMProcessingStep)
        assert step.prompt_template == "Hello {name}"
        assert step.next_state == "done"
        assert step.error_state is None

    def test_with_error_state(self):
        mock_llm = object()
        step = llm_step(
            "s1", "LLM", mock_llm,
            prompt_template="test",
            context_mapping={},
            output_mapping={},
            next_state="done",
            error_state="error",
        )
        assert step.error_state == "error"


class TestWaitEventStep:
    """Test wait_event_step factory."""

    def test_basic(self):
        step = wait_event_step("s1", "Wait", "payment_received", "paid")
        assert isinstance(step, WaitForEventStep)
        assert step.config.event_type == "payment_received"
        assert step.config.success_state == "paid"
        assert step.config.timeout_seconds is None

    def test_with_timeout(self):
        step = wait_event_step(
            "s1", "Wait", "event",
            success_state="ok",
            timeout_seconds=30,
            timeout_state="timed_out",
        )
        assert step.config.timeout_seconds == 30
        assert step.config.timeout_state == "timed_out"

    def test_with_event_mapping(self):
        step = wait_event_step(
            "s1", "Wait", "event",
            success_state="ok",
            event_mapping={"amount": "payment_amount"},
        )
        assert step.config.event_mapping == {"amount": "payment_amount"}


class TestTimerStep:
    """Test timer_step factory."""

    def test_basic(self):
        step = timer_step("s1", "Delay", 60, "next")
        assert isinstance(step, TimerStep)
        assert step.delay_seconds == 60
        assert step.next_state == "next"


class TestParallelStep:
    """Test parallel_step factory."""

    def test_basic(self):
        s1 = auto_step("sub1", "Sub 1", "done")
        s2 = auto_step("sub2", "Sub 2", "done")
        step = parallel_step("p1", "Parallel", [s1, s2], "merged")
        assert isinstance(step, ParallelStep)
        assert len(step.steps) == 2
        assert step.next_state == "merged"
        assert step.error_state is None

    def test_with_error_state(self):
        step = parallel_step("p1", "P", [], "ok", error_state="err")
        assert step.error_state == "err"

    def test_with_aggregation(self):
        def fn(results):
            return {"merged": True}
        step = parallel_step("p1", "P", [], "ok", aggregation_function=fn)
        assert step.aggregation_function is fn


class TestWorkflowBuilder:
    """Test WorkflowBuilder fluent API."""

    def test_build_empty(self):
        wf = workflow_builder("wf-1", "Test").build()
        assert isinstance(wf, WorkflowDefinition)
        assert wf.workflow_id == "wf-1"

    def test_add_step(self):
        s = auto_step("s1", "Step 1", "done")
        wf = workflow_builder("wf-1", "Test").add_step(s).build()
        assert "s1" in wf.steps

    def test_set_initial_step(self):
        s = auto_step("s1", "Step 1", "done")
        wf = workflow_builder("wf-1", "Test").set_initial_step(s).build()
        assert wf.initial_step_id == "s1"

    def test_add_metadata(self):
        wf = (workflow_builder("wf-1", "Test")
              .add_metadata("version", "1.0")
              .build())
        assert wf.metadata["version"] == "1.0"

    def test_chaining(self):
        s1 = auto_step("s1", "Step 1", "s2")
        s2 = auto_step("s2", "Step 2", "done")
        wf = (workflow_builder("wf-1", "Test")
              .set_initial_step(s1)
              .add_step(s2)
              .add_metadata("key", "val")
              .build())
        assert wf.initial_step_id == "s1"
        assert "s2" in wf.steps


class TestLinearWorkflow:
    """Test linear_workflow factory."""

    def test_basic(self):
        s1 = auto_step("s1", "Step 1", "s2")
        s2 = auto_step("s2", "Step 2", "done")
        wf = linear_workflow("wf-1", "Linear", [s1, s2])
        assert wf.initial_step_id == "s1"
        assert "s1" in wf.steps
        assert "s2" in wf.steps

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one step"):
            linear_workflow("wf-1", "Empty", [])


class TestConditionalWorkflow:
    """Test conditional_workflow factory."""

    def test_basic(self):
        init = auto_step("init", "Init", "check")
        check = condition_step("check", "Check", lambda ctx: True, "yes", "no")
        yes_step = auto_step("yes", "Yes", "done")
        no_step = auto_step("no", "No", "done")

        wf = conditional_workflow("wf-1", "Cond", init, check, [yes_step], [no_step])
        assert wf.initial_step_id == "init"
        assert "check" in wf.steps
        assert "yes" in wf.steps
        assert "no" in wf.steps


class TestEventDrivenWorkflow:
    """Test event_driven_workflow factory."""

    def test_with_setup(self):
        setup = auto_step("setup", "Setup", "wait")
        wait = wait_event_step("wait", "Wait", "event", "process")
        process = auto_step("process", "Process", "done")

        wf = event_driven_workflow("wf-1", "Event", [setup], wait, [process])
        assert wf.initial_step_id == "setup"
        assert "wait" in wf.steps
        assert "process" in wf.steps

    def test_without_setup(self):
        wait = wait_event_step("wait", "Wait", "event", "process")
        process = auto_step("process", "Process", "done")

        wf = event_driven_workflow("wf-1", "Event", [], wait, [process])
        assert wf.initial_step_id == "wait"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
