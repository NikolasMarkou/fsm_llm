"""
Unit tests for workflow step types.
Tests step creation, async execution, and error handling.
"""
import pytest

from fsm_llm_workflows.steps import (
    AutoTransitionStep,
    APICallStep,
    ConditionStep,
    WaitForEventStep,
    TimerStep,
    ParallelStep,
)
from fsm_llm_workflows.models import WaitEventConfig
from fsm_llm_workflows.exceptions import WorkflowStepError


# ---------------------------------------------------------------------------
# AutoTransitionStep
# ---------------------------------------------------------------------------

class TestAutoTransitionStep:
    """Test AutoTransitionStep execution."""

    @pytest.mark.asyncio
    async def test_execute_no_action(self):
        step = AutoTransitionStep(step_id="s1", name="Test", next_state="done")
        result = await step.execute({})

        assert result.success is True
        assert result.next_state == "done"

    @pytest.mark.asyncio
    async def test_execute_with_sync_action(self):
        def my_action(ctx):
            return {"computed": ctx.get("x", 0) * 2}

        step = AutoTransitionStep(step_id="s1", name="Test", next_state="done", action=my_action)
        result = await step.execute({"x": 5})

        assert result.success is True
        assert result.data == {"computed": 10}

    @pytest.mark.asyncio
    async def test_execute_with_async_action(self):
        async def my_action(ctx):
            return {"async_result": True}

        step = AutoTransitionStep(step_id="s1", name="Test", next_state="done", action=my_action)
        result = await step.execute({})

        assert result.success is True
        assert result.data == {"async_result": True}

    @pytest.mark.asyncio
    async def test_execute_action_error(self):
        def bad_action(ctx):
            raise ValueError("boom")

        step = AutoTransitionStep(step_id="s1", name="Test", next_state="done", action=bad_action)
        with pytest.raises(WorkflowStepError):
            await step.execute({})


# ---------------------------------------------------------------------------
# ConditionStep
# ---------------------------------------------------------------------------

class TestConditionStep:
    """Test ConditionStep execution."""

    @pytest.mark.asyncio
    async def test_condition_true(self):
        step = ConditionStep(
            step_id="s1", name="Check",
            condition=lambda ctx: ctx.get("ready", False),
            true_state="go", false_state="wait",
        )
        result = await step.execute({"ready": True})

        assert result.success is True
        assert result.next_state == "go"
        assert result.data["condition_result"] is True

    @pytest.mark.asyncio
    async def test_condition_false(self):
        step = ConditionStep(
            step_id="s1", name="Check",
            condition=lambda ctx: ctx.get("ready", False),
            true_state="go", false_state="wait",
        )
        result = await step.execute({"ready": False})

        assert result.next_state == "wait"
        assert result.data["condition_result"] is False

    @pytest.mark.asyncio
    async def test_condition_async(self):
        async def check(ctx):
            return True

        step = ConditionStep(
            step_id="s1", name="Check",
            condition=check, true_state="go", false_state="wait",
        )
        result = await step.execute({})
        assert result.next_state == "go"

    @pytest.mark.asyncio
    async def test_condition_error(self):
        def bad_check(ctx):
            raise RuntimeError("check failed")

        step = ConditionStep(
            step_id="s1", name="Check",
            condition=bad_check, true_state="go", false_state="wait",
        )
        with pytest.raises(WorkflowStepError):
            await step.execute({})


# ---------------------------------------------------------------------------
# APICallStep
# ---------------------------------------------------------------------------

class TestAPICallStep:
    """Test APICallStep execution."""

    @pytest.mark.asyncio
    async def test_success(self):
        def my_api(**params):
            return {"status": "ok", "result": params.get("query", "")}

        step = APICallStep(
            step_id="s1", name="API",
            api_function=my_api,
            success_state="done",
            failure_state="error",
            input_mapping={"query": "search_term"},
            output_mapping={"response": "result"},
        )
        result = await step.execute({"search_term": "hello"})

        assert result.success is True
        assert result.next_state == "done"
        assert result.data.get("response") == "hello"

    @pytest.mark.asyncio
    async def test_failure(self):
        def failing_api(**params):
            raise ConnectionError("timeout")

        step = APICallStep(
            step_id="s1", name="API",
            api_function=failing_api,
            success_state="done",
            failure_state="error",
        )
        result = await step.execute({})

        assert result.success is False
        assert result.next_state == "error"

    @pytest.mark.asyncio
    async def test_async_api(self):
        async def my_api(**params):
            return {"data": "async result"}

        step = APICallStep(
            step_id="s1", name="API",
            api_function=my_api,
            success_state="done",
            failure_state="error",
            output_mapping={"result": "data"},
        )
        result = await step.execute({})

        assert result.success is True
        assert result.data.get("result") == "async result"


# ---------------------------------------------------------------------------
# WaitForEventStep
# ---------------------------------------------------------------------------

class TestWaitForEventStep:
    """Test WaitForEventStep execution."""

    @pytest.mark.asyncio
    async def test_execute(self):
        step = WaitForEventStep(
            step_id="s1", name="Wait",
            config=WaitEventConfig(
                event_type="payment",
                success_state="paid",
                timeout_seconds=30,
            ),
        )
        result = await step.execute({})

        assert result.success is True
        assert "_waiting_info" in result.data
        info = result.data["_waiting_info"]
        assert info["event_type"] == "payment"
        assert info["timeout_seconds"] == 30
        assert info["waiting_for_event"] is True


# ---------------------------------------------------------------------------
# TimerStep
# ---------------------------------------------------------------------------

class TestTimerStep:
    """Test TimerStep execution."""

    @pytest.mark.asyncio
    async def test_execute(self):
        step = TimerStep(step_id="s1", name="Delay", delay_seconds=60, next_state="next")
        result = await step.execute({})

        assert result.success is True
        assert "_timer_info" in result.data
        info = result.data["_timer_info"]
        assert info["delay_seconds"] == 60
        assert info["next_state"] == "next"
        assert info["waiting_for_timer"] is True


# ---------------------------------------------------------------------------
# ParallelStep
# ---------------------------------------------------------------------------

class TestParallelStep:
    """Test ParallelStep execution."""

    @pytest.mark.asyncio
    async def test_all_succeed(self):
        s1 = AutoTransitionStep(step_id="a", name="A", next_state="done",
                                action=lambda ctx: {"val": 1})
        s2 = AutoTransitionStep(step_id="b", name="B", next_state="done",
                                action=lambda ctx: {"val": 2})

        step = ParallelStep(step_id="p1", name="Parallel", steps=[s1, s2], next_state="merged")
        result = await step.execute({})

        assert result.success is True
        assert result.next_state == "merged"
        # Default aggregation prefixes by step index
        assert "step_0_val" in result.data
        assert "step_1_val" in result.data

    @pytest.mark.asyncio
    async def test_with_error_state(self):
        def bad_action(ctx):
            raise ValueError("fail")

        s1 = AutoTransitionStep(step_id="a", name="A", next_state="done", action=bad_action)
        s2 = AutoTransitionStep(step_id="b", name="B", next_state="done")

        step = ParallelStep(
            step_id="p1", name="Parallel",
            steps=[s1, s2], next_state="ok", error_state="err",
        )
        result = await step.execute({})

        assert result.success is False
        assert result.next_state == "err"

    @pytest.mark.asyncio
    async def test_custom_aggregation(self):
        s1 = AutoTransitionStep(step_id="a", name="A", next_state="done",
                                action=lambda ctx: {"count": 1})
        s2 = AutoTransitionStep(step_id="b", name="B", next_state="done",
                                action=lambda ctx: {"count": 2})

        def aggregate(results):
            total = sum(r.data.get("count", 0) for r in results if r.data)
            return {"total": total}

        step = ParallelStep(
            step_id="p1", name="Parallel",
            steps=[s1, s2], next_state="merged",
            aggregation_function=aggregate,
        )
        result = await step.execute({})

        assert result.data["total"] == 3

    @pytest.mark.asyncio
    async def test_context_isolation(self):
        """Each parallel step should get its own context copy."""
        def mutate(ctx):
            ctx["mutated"] = True
            return {}

        s1 = AutoTransitionStep(step_id="a", name="A", next_state="done", action=mutate)
        s2 = AutoTransitionStep(step_id="b", name="B", next_state="done",
                                action=lambda ctx: {"saw_mutation": ctx.get("mutated", False)})

        step = ParallelStep(step_id="p1", name="Parallel", steps=[s1, s2], next_state="done")
        original_ctx = {"original": True}
        await step.execute(original_ctx)

        # Original context should not be mutated
        assert "mutated" not in original_ctx


# ---------------------------------------------------------------------------
# WorkflowStep (abstract)
# ---------------------------------------------------------------------------

class TestWorkflowStepBase:
    """Test base WorkflowStep properties."""

    def test_step_model_fields(self):
        step = AutoTransitionStep(step_id="s1", name="Test", next_state="next")
        assert step.step_id == "s1"
        assert step.name == "Test"
        assert step.description == ""

    def test_step_with_description(self):
        step = AutoTransitionStep(step_id="s1", name="Test", next_state="next", description="desc")
        assert step.description == "desc"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
