"""
Unit tests for the workflow DSL factory functions and WorkflowBuilder.
"""

import pytest

from fsm_llm import BuildError
from fsm_llm.workflows.definitions import WorkflowDefinition
from fsm_llm.workflows.dsl import (
    WorkflowBuilder,
    api_step,
    auto_step,
    condition_step,
    conditional_workflow,
    conversation_step,
    create_workflow,
    event_driven_workflow,
    linear_workflow,
    llm_step,
    parallel_step,
    timer_step,
    wait_event_step,
    workflow_builder,
)
from fsm_llm.workflows.exceptions import (
    WorkflowDefinitionError,
    WorkflowValidationError,
)
from fsm_llm.workflows.steps import (
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
            "s1",
            "API Call",
            fn,
            "ok",
            "err",
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
            "s1",
            "LLM Process",
            mock_llm,
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
            "s1",
            "LLM",
            mock_llm,
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
            "s1",
            "Wait",
            "event",
            success_state="ok",
            timeout_seconds=30,
            timeout_state="timed_out",
        )
        assert step.config.timeout_seconds == 30
        assert step.config.timeout_state == "timed_out"

    def test_with_event_mapping(self):
        step = wait_event_step(
            "s1",
            "Wait",
            "event",
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


def _end():
    """A terminal step (no outgoing state) so a built workflow validates."""
    return conversation_step("end", "End", fsm_file="x.json")


def _valid_builder():
    return (
        workflow_builder("wf-1", "Test")
        .set_initial_step(auto_step("s1", "Step 1", "end"))
        .add_step(_end())
    )


class TestWorkflowBuilder:
    """Test WorkflowBuilder fluent API (build() always validates)."""

    def test_build_empty_is_refused(self):
        with pytest.raises(BuildError) as ei:
            workflow_builder("wf-1", "Test").build()
        assert isinstance(ei.value.__cause__, WorkflowValidationError)

    def test_build_returns_definition(self):
        wf = _valid_builder().build()
        assert isinstance(wf, WorkflowDefinition)
        assert wf.workflow_id == "wf-1"

    def test_add_step(self):
        wf = (
            workflow_builder("wf-1", "Test")
            .set_initial_step(auto_step("s1", "S1", "s2"))
            .add_step(auto_step("s2", "S2", "end"))
            .add_step(_end())
            .build()
        )
        assert set(wf.steps) == {"s1", "s2", "end"}

    def test_set_initial_step(self):
        wf = _valid_builder().build()
        assert wf.initial_step_id == "s1"

    def test_add_metadata(self):
        wf = _valid_builder().add_metadata("version", "1.0").build()
        assert wf.metadata["version"] == "1.0"

    def test_chaining(self):
        s1 = auto_step("s1", "Step 1", "s2")
        s2 = auto_step("s2", "Step 2", "end")
        wf = (
            workflow_builder("wf-1", "Test")
            .set_initial_step(s1)
            .add_step(s2)
            .add_step(_end())
            .add_metadata("key", "val")
            .build()
        )
        assert wf.initial_step_id == "s1"
        assert "s2" in wf.steps

    def test_build_has_no_validate_parameter(self):
        with pytest.raises(TypeError):
            _valid_builder().build(validate=False)  # type: ignore[call-arg]


class TestWorkflowBuilderCallOrder:
    """set_initial_step keeps the step in call order; the last call is initial."""

    def test_two_initial_calls_keep_both_steps_in_call_order(self):
        a = auto_step("a", "A", "end")
        b = auto_step("b", "B", "a")
        wf = (
            workflow_builder("wf-1", "Test")
            .set_initial_step(a)
            .set_initial_step(b)
            .add_step(_end())
            .build()
        )
        assert list(wf.steps) == ["a", "b", "end"]
        assert wf.initial_step_id == "b"

    def test_initial_step_keeps_its_call_position(self):
        wf = (
            workflow_builder("wf-1", "Test")
            .add_step(_end())
            .set_initial_step(auto_step("s1", "S1", "end"))
            .build()
        )
        assert list(wf.steps) == ["end", "s1"]
        assert wf.initial_step_id == "s1"


class TestWorkflowBuilderBuildErrors:
    """Every failure inside build() is a BuildError chained from its cause."""

    def test_invalid_definition_fields_raise_build_error(self):
        from pydantic import ValidationError

        with pytest.raises(BuildError) as ei:
            WorkflowBuilder(None, "W").build()  # type: ignore[arg-type]
        assert isinstance(ei.value.__cause__, ValidationError)

    def test_non_step_raises_build_error_naming_position(self):
        with pytest.raises(BuildError, match=r"position 1.*NoneType"):
            (
                workflow_builder("wf-1", "Test")
                .add_step(_end())
                .add_step(None)  # type: ignore[arg-type]
                .build()
            )

    def test_non_step_initial_raises_build_error(self):
        with pytest.raises(BuildError):
            workflow_builder("wf-1", "Test").set_initial_step(
                "nope"  # type: ignore[arg-type]
            ).build()


class TestWorkflowBuilderConvention:
    """Mutators record, build() copies what the builder owns and validates."""

    def test_mutators_return_same_builder(self):
        b = workflow_builder("wf-1", "Test")
        s = auto_step("s1", "S1", "end")
        assert b.add_step(s) is b
        assert b.set_initial_step(s) is b
        assert b.add_metadata("k", "v") is b

    def test_builder_mutation_after_build_does_not_touch_product(self):
        b = _valid_builder().add_metadata("k", "v")
        wf = b.build()
        b.add_step(auto_step("late", "Late", "end")).add_metadata("k2", "v2")
        assert "late" not in wf.steps
        assert "k2" not in wf.metadata

    def test_product_mutation_does_not_touch_next_build(self):
        b = _valid_builder().add_metadata("k", "v")
        wf = b.build()
        wf.metadata["k"] = "changed"
        wf.steps.pop("end")
        again = b.build()
        assert again.metadata["k"] == "v"
        assert "end" in again.steps

    def test_second_build_is_independent(self):
        b = _valid_builder()
        first, second = b.build(), b.build()
        assert first is not second
        assert first.steps is not second.steps
        assert first.metadata is not second.metadata

    def test_steps_are_shared_by_reference(self):
        s = auto_step("s1", "S1", "end")
        wf = workflow_builder("wf-1", "T").set_initial_step(s).add_step(_end()).build()
        assert wf.steps["s1"] is s

    def test_duplicate_step_id_is_a_build_error(self):
        b = (
            workflow_builder("wf-1", "Test")
            .add_step(auto_step("s1", "A", "end"))
            .add_step(auto_step("s1", "B", "end"))
        )
        with pytest.raises(BuildError) as ei:
            b.build()
        assert isinstance(ei.value.__cause__, WorkflowDefinitionError)

    def test_invalid_workflow_is_a_build_error_with_errors(self):
        b = workflow_builder("wf-1", "Test").set_initial_step(
            auto_step("s1", "S1", "nowhere")
        )
        with pytest.raises(BuildError) as ei:
            b.build()
        cause = ei.value.__cause__
        assert isinstance(cause, WorkflowValidationError)
        assert ei.value.errors == cause.validation_errors
        assert any("nowhere" in e for e in ei.value.errors)


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
