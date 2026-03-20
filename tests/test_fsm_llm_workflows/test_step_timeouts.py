"""
Unit tests for workflow step timeout functionality.

Tests cover: step-level timeout via _with_timeout(), DEFAULT_STEP_TIMEOUT
constant, and timeout behavior for AutoTransitionStep, APICallStep,
ConditionStep, and ParallelStep.
"""
import asyncio
import pytest

from fsm_llm_workflows.steps import (
    AutoTransitionStep,
    APICallStep,
    ConditionStep,
    ParallelStep,
)
from fsm_llm_workflows.exceptions import WorkflowStepError
from fsm_llm.constants import DEFAULT_STEP_TIMEOUT


# ══════════════════════════════════════════════════════════════
# 1. DEFAULT_STEP_TIMEOUT constant
# ══════════════════════════════════════════════════════════════


class TestDefaultStepTimeout:
    """Verify the DEFAULT_STEP_TIMEOUT constant exists and is sensible."""

    def test_constant_exists(self):
        assert DEFAULT_STEP_TIMEOUT is not None

    def test_constant_is_positive_float(self):
        assert isinstance(DEFAULT_STEP_TIMEOUT, float)
        assert DEFAULT_STEP_TIMEOUT > 0

    def test_constant_value(self):
        assert DEFAULT_STEP_TIMEOUT == 120.0


# ══════════════════════════════════════════════════════════════
# 2. Steps without timeout (None) work normally
# ══════════════════════════════════════════════════════════════


class TestStepsWithoutTimeout:
    """Steps with timeout=None should work without any timeout enforcement."""

    @pytest.mark.asyncio
    async def test_auto_transition_no_timeout(self):
        step = AutoTransitionStep(
            step_id="s1", name="Test", next_state="done", timeout=None
        )
        result = await step.execute({})
        assert result.success is True
        assert result.next_state == "done"

    @pytest.mark.asyncio
    async def test_auto_transition_default_timeout_is_none(self):
        step = AutoTransitionStep(step_id="s1", name="Test", next_state="done")
        assert step.timeout is None

    @pytest.mark.asyncio
    async def test_condition_no_timeout(self):
        step = ConditionStep(
            step_id="s1", name="Check",
            condition=lambda ctx: True,
            true_state="yes", false_state="no",
            timeout=None,
        )
        result = await step.execute({})
        assert result.success is True
        assert result.next_state == "yes"

    @pytest.mark.asyncio
    async def test_api_call_no_timeout(self):
        async def my_api(**params):
            return {"data": "result"}

        step = APICallStep(
            step_id="s1", name="API",
            api_function=my_api,
            success_state="done", failure_state="error",
            output_mapping={"result": "data"},
            timeout=None,
        )
        result = await step.execute({})
        assert result.success is True
        assert result.data.get("result") == "result"

    @pytest.mark.asyncio
    async def test_parallel_no_timeout(self):
        s1 = AutoTransitionStep(
            step_id="a", name="A", next_state="done",
            action=lambda ctx: {"v": 1},
        )
        step = ParallelStep(
            step_id="p1", name="Parallel", steps=[s1], next_state="merged",
            timeout=None,
        )
        result = await step.execute({})
        assert result.success is True


# ══════════════════════════════════════════════════════════════
# 3. AutoTransitionStep with timeout
# ══════════════════════════════════════════════════════════════


class TestAutoTransitionStepTimeout:
    """Timeout behavior for AutoTransitionStep with async actions."""

    @pytest.mark.asyncio
    async def test_fast_async_action_succeeds(self):
        async def fast_action(ctx):
            await asyncio.sleep(0.01)
            return {"result": "fast"}

        step = AutoTransitionStep(
            step_id="s1", name="Fast",
            next_state="done",
            action=fast_action,
            timeout=1.0,
        )
        result = await step.execute({})
        assert result.success is True
        assert result.data.get("result") == "fast"

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_slow_async_action_raises_workflow_step_error(self):
        async def slow_action(ctx):
            await asyncio.sleep(5.0)
            return {"result": "never"}

        step = AutoTransitionStep(
            step_id="s1", name="Slow",
            next_state="done",
            action=slow_action,
            timeout=0.1,
        )

        with pytest.raises(WorkflowStepError) as exc_info:
            await step.execute({})

        # The outer exception wraps the timeout; check the cause chain
        assert exc_info.value.cause is not None
        assert "timed out" in str(exc_info.value.cause)

    @pytest.mark.asyncio
    async def test_sync_action_not_affected_by_timeout(self):
        """Sync actions are not wrapped with _with_timeout, so they run directly."""
        def sync_action(ctx):
            return {"sync": True}

        step = AutoTransitionStep(
            step_id="s1", name="Sync",
            next_state="done",
            action=sync_action,
            timeout=0.1,
        )
        result = await step.execute({})
        assert result.success is True
        assert result.data.get("sync") is True

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_timeout_error_includes_step_id(self):
        async def slow_action(ctx):
            await asyncio.sleep(5.0)
            return {}

        step = AutoTransitionStep(
            step_id="my_step_id", name="Slow",
            next_state="done",
            action=slow_action,
            timeout=0.1,
        )

        with pytest.raises(WorkflowStepError) as exc_info:
            await step.execute({})

        assert exc_info.value.step_id == "my_step_id"

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_timeout_error_includes_duration(self):
        async def slow_action(ctx):
            await asyncio.sleep(5.0)
            return {}

        step = AutoTransitionStep(
            step_id="s1", name="Slow",
            next_state="done",
            action=slow_action,
            timeout=0.1,
        )

        with pytest.raises(WorkflowStepError) as exc_info:
            await step.execute({})

        # Duration appears in the cause (the inner timeout WorkflowStepError)
        assert exc_info.value.cause is not None
        assert "0.1" in str(exc_info.value.cause)


# ══════════════════════════════════════════════════════════════
# 4. APICallStep with timeout
# ══════════════════════════════════════════════════════════════


class TestAPICallStepTimeout:
    """Timeout behavior for APICallStep with async api_function."""

    @pytest.mark.asyncio
    async def test_fast_async_api_succeeds(self):
        async def fast_api(**params):
            await asyncio.sleep(0.01)
            return {"status": "ok"}

        step = APICallStep(
            step_id="s1", name="API",
            api_function=fast_api,
            success_state="done", failure_state="error",
            output_mapping={"status_result": "status"},
            timeout=1.0,
        )
        result = await step.execute({})
        assert result.success is True
        assert result.data.get("status_result") == "ok"

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_slow_async_api_returns_failure(self):
        """APICallStep catches exceptions and returns failure results."""
        async def slow_api(**params):
            await asyncio.sleep(5.0)
            return {"status": "ok"}

        step = APICallStep(
            step_id="s1", name="API",
            api_function=slow_api,
            success_state="done", failure_state="error",
            timeout=0.1,
        )
        # APICallStep catches exceptions in execute() and returns failure result
        result = await step.execute({})
        assert result.success is False
        assert result.next_state == "error"


# ══════════════════════════════════════════════════════════════
# 5. ConditionStep with timeout
# ══════════════════════════════════════════════════════════════


class TestConditionStepTimeout:
    """Timeout behavior for ConditionStep with async conditions."""

    @pytest.mark.asyncio
    async def test_fast_async_condition_succeeds(self):
        async def fast_check(ctx):
            await asyncio.sleep(0.01)
            return True

        step = ConditionStep(
            step_id="s1", name="Check",
            condition=fast_check,
            true_state="yes", false_state="no",
            timeout=1.0,
        )
        result = await step.execute({})
        assert result.success is True
        assert result.next_state == "yes"

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_slow_async_condition_raises_workflow_step_error(self):
        async def slow_check(ctx):
            await asyncio.sleep(5.0)
            return True

        step = ConditionStep(
            step_id="s1", name="Check",
            condition=slow_check,
            true_state="yes", false_state="no",
            timeout=0.1,
        )

        with pytest.raises(WorkflowStepError) as exc_info:
            await step.execute({})

        # The outer exception wraps the timeout; check the cause chain
        assert exc_info.value.cause is not None
        assert "timed out" in str(exc_info.value.cause)

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_slow_condition_error_includes_step_id(self):
        async def slow_check(ctx):
            await asyncio.sleep(5.0)
            return True

        step = ConditionStep(
            step_id="cond_step", name="SlowCheck",
            condition=slow_check,
            true_state="yes", false_state="no",
            timeout=0.1,
        )

        with pytest.raises(WorkflowStepError) as exc_info:
            await step.execute({})

        assert exc_info.value.step_id == "cond_step"


# ══════════════════════════════════════════════════════════════
# 6. ParallelStep with timeout
# ══════════════════════════════════════════════════════════════


class TestParallelStepTimeout:
    """Timeout behavior for ParallelStep."""

    @pytest.mark.asyncio
    async def test_fast_parallel_steps_succeed(self):
        async def fast_action(ctx):
            await asyncio.sleep(0.01)
            return {"done": True}

        s1 = AutoTransitionStep(
            step_id="a", name="A", next_state="done", action=fast_action,
        )
        s2 = AutoTransitionStep(
            step_id="b", name="B", next_state="done", action=fast_action,
        )

        step = ParallelStep(
            step_id="p1", name="Parallel",
            steps=[s1, s2], next_state="merged",
            timeout=2.0,
        )
        result = await step.execute({})
        assert result.success is True
        assert result.next_state == "merged"

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_slow_parallel_steps_returns_failure(self):
        """ParallelStep wraps gather with _with_timeout; timeout goes through exception handling."""
        async def slow_action(ctx):
            await asyncio.sleep(5.0)
            return {"done": True}

        s1 = AutoTransitionStep(
            step_id="a", name="A", next_state="done", action=slow_action,
        )

        step = ParallelStep(
            step_id="p1", name="Parallel",
            steps=[s1], next_state="ok", error_state="err",
            timeout=0.1,
        )
        # ParallelStep catches exceptions in its execute() and returns failure
        result = await step.execute({})
        assert result.success is False
        assert result.next_state == "err"

    @pytest.mark.asyncio
    async def test_parallel_with_one_slow_and_one_fast_within_timeout(self):
        """Both steps finish within timeout."""
        async def fast_action(ctx):
            await asyncio.sleep(0.01)
            return {"fast": True}

        async def medium_action(ctx):
            await asyncio.sleep(0.05)
            return {"medium": True}

        s1 = AutoTransitionStep(
            step_id="a", name="Fast", next_state="done", action=fast_action,
        )
        s2 = AutoTransitionStep(
            step_id="b", name="Medium", next_state="done", action=medium_action,
        )

        step = ParallelStep(
            step_id="p1", name="Parallel",
            steps=[s1, s2], next_state="merged",
            timeout=2.0,
        )
        result = await step.execute({})
        assert result.success is True


# ══════════════════════════════════════════════════════════════
# 7. _with_timeout method directly
# ══════════════════════════════════════════════════════════════


class TestWithTimeoutMethod:
    """Test the _with_timeout helper method on WorkflowStep."""

    @pytest.mark.asyncio
    async def test_with_timeout_returns_result_when_fast(self):
        step = AutoTransitionStep(
            step_id="s1", name="Test", next_state="done", timeout=1.0,
        )

        async def fast_coro():
            return "result"

        result = await step._with_timeout(fast_coro())
        assert result == "result"

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_with_timeout_raises_workflow_step_error_when_slow(self):
        step = AutoTransitionStep(
            step_id="s1", name="Test", next_state="done", timeout=0.1,
        )

        async def slow_coro():
            await asyncio.sleep(5.0)
            return "never"

        with pytest.raises(WorkflowStepError, match="timed out"):
            await step._with_timeout(slow_coro())

    @pytest.mark.asyncio
    async def test_with_timeout_none_skips_timeout(self):
        step = AutoTransitionStep(
            step_id="s1", name="Test", next_state="done", timeout=None,
        )

        async def coro():
            return "no timeout"

        result = await step._with_timeout(coro())
        assert result == "no timeout"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
