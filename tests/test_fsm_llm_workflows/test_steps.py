"""
Unit tests for workflow step types.
Tests step creation, async execution, and error handling.
"""

from unittest.mock import MagicMock, patch

import pytest

from fsm_llm_workflows.exceptions import WorkflowStepError
from fsm_llm_workflows.models import WaitEventConfig
from fsm_llm_workflows.steps import (
    APICallStep,
    AutoTransitionStep,
    ConditionStep,
    ConversationStep,
    ParallelStep,
    TimerStep,
    WaitForEventStep,
)

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

        step = AutoTransitionStep(
            step_id="s1", name="Test", next_state="done", action=my_action
        )
        result = await step.execute({"x": 5})

        assert result.success is True
        assert result.data == {"computed": 10}

    @pytest.mark.asyncio
    async def test_execute_with_async_action(self):
        async def my_action(ctx):
            return {"async_result": True}

        step = AutoTransitionStep(
            step_id="s1", name="Test", next_state="done", action=my_action
        )
        result = await step.execute({})

        assert result.success is True
        assert result.data == {"async_result": True}

    @pytest.mark.asyncio
    async def test_execute_action_error(self):
        def bad_action(ctx):
            raise ValueError("boom")

        step = AutoTransitionStep(
            step_id="s1", name="Test", next_state="done", action=bad_action
        )
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
            step_id="s1",
            name="Check",
            condition=lambda ctx: ctx.get("ready", False),
            true_state="go",
            false_state="wait",
        )
        result = await step.execute({"ready": True})

        assert result.success is True
        assert result.next_state == "go"
        assert result.data["condition_result"] is True

    @pytest.mark.asyncio
    async def test_condition_false(self):
        step = ConditionStep(
            step_id="s1",
            name="Check",
            condition=lambda ctx: ctx.get("ready", False),
            true_state="go",
            false_state="wait",
        )
        result = await step.execute({"ready": False})

        assert result.next_state == "wait"
        assert result.data["condition_result"] is False

    @pytest.mark.asyncio
    async def test_condition_async(self):
        async def check(ctx):
            return True

        step = ConditionStep(
            step_id="s1",
            name="Check",
            condition=check,
            true_state="go",
            false_state="wait",
        )
        result = await step.execute({})
        assert result.next_state == "go"

    @pytest.mark.asyncio
    async def test_condition_error(self):
        def bad_check(ctx):
            raise RuntimeError("check failed")

        step = ConditionStep(
            step_id="s1",
            name="Check",
            condition=bad_check,
            true_state="go",
            false_state="wait",
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
            step_id="s1",
            name="API",
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
            step_id="s1",
            name="API",
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
            step_id="s1",
            name="API",
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
            step_id="s1",
            name="Wait",
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
        step = TimerStep(
            step_id="s1", name="Delay", delay_seconds=60, next_state="next"
        )
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
        s1 = AutoTransitionStep(
            step_id="a", name="A", next_state="done", action=lambda ctx: {"val": 1}
        )
        s2 = AutoTransitionStep(
            step_id="b", name="B", next_state="done", action=lambda ctx: {"val": 2}
        )

        step = ParallelStep(
            step_id="p1", name="Parallel", steps=[s1, s2], next_state="merged"
        )
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

        s1 = AutoTransitionStep(
            step_id="a", name="A", next_state="done", action=bad_action
        )
        s2 = AutoTransitionStep(step_id="b", name="B", next_state="done")

        step = ParallelStep(
            step_id="p1",
            name="Parallel",
            steps=[s1, s2],
            next_state="ok",
            error_state="err",
        )
        result = await step.execute({})

        assert result.success is False
        assert result.next_state == "err"

    @pytest.mark.asyncio
    async def test_custom_aggregation(self):
        s1 = AutoTransitionStep(
            step_id="a", name="A", next_state="done", action=lambda ctx: {"count": 1}
        )
        s2 = AutoTransitionStep(
            step_id="b", name="B", next_state="done", action=lambda ctx: {"count": 2}
        )

        def aggregate(results):
            total = sum(r.data.get("count", 0) for r in results if r.data)
            return {"total": total}

        step = ParallelStep(
            step_id="p1",
            name="Parallel",
            steps=[s1, s2],
            next_state="merged",
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
        s2 = AutoTransitionStep(
            step_id="b",
            name="B",
            next_state="done",
            action=lambda ctx: {"saw_mutation": ctx.get("mutated", False)},
        )

        step = ParallelStep(
            step_id="p1", name="Parallel", steps=[s1, s2], next_state="done"
        )
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
        step = AutoTransitionStep(
            step_id="s1", name="Test", next_state="next", description="desc"
        )
        assert step.description == "desc"


# ---------------------------------------------------------------------------
# ConversationStep
# ---------------------------------------------------------------------------


class TestConversationStep:
    """Test ConversationStep FSM-workflow integration."""

    def _mock_api(self, collected_data=None, responses=None):
        """Create a mock API instance for ConversationStep tests."""
        mock = MagicMock()
        mock.start_conversation.return_value = ("conv-1", "Hello!")
        mock.converse.side_effect = responses or ["Got it."]
        mock.has_conversation_ended.return_value = False
        mock.get_data.return_value = collected_data or {"name": "Alice"}
        mock.end_conversation.return_value = None
        return mock

    @pytest.mark.asyncio
    async def test_requires_fsm_file_or_definition(self):
        """ConversationStep must fail if neither fsm_file nor fsm_definition is given."""
        step = ConversationStep(step_id="conv1", name="Conv", success_state="done")
        with pytest.raises(WorkflowStepError, match="requires either"):
            await step.execute({})

    @pytest.mark.asyncio
    async def test_success_with_fsm_definition(self):
        """ConversationStep runs a full conversation and returns collected data."""
        mock_api = self._mock_api(
            collected_data={"name": "Alice", "age": "30"}, responses=["Thanks!"]
        )

        step = ConversationStep(
            step_id="conv1",
            name="Collect Info",
            fsm_definition={"name": "test", "initial_state": "start", "states": {}},
            success_state="process",
            auto_messages=["My name is Alice"],
            context_mapping={"user_name": "name", "user_age": "age"},
        )

        with patch("fsm_llm.API") as MockAPI:
            MockAPI.from_definition.return_value = mock_api
            result = await step.execute({})

        assert result.success is True
        assert result.next_state == "process"
        assert result.data["user_name"] == "Alice"
        assert result.data["user_age"] == "30"

    @pytest.mark.asyncio
    async def test_context_mapping_input(self):
        """Initial context maps workflow keys to conversation keys."""
        mock_api = self._mock_api()

        step = ConversationStep(
            step_id="conv1",
            name="Conv",
            fsm_definition={"name": "test", "initial_state": "s", "states": {}},
            success_state="done",
            initial_context={"conv_user": "workflow_user"},
        )

        with patch("fsm_llm.API") as MockAPI:
            MockAPI.from_definition.return_value = mock_api
            await step.execute({"workflow_user": "Bob"})

        # Verify start_conversation was called with mapped context
        call_kwargs = mock_api.start_conversation.call_args
        initial_ctx = call_kwargs[1].get(
            "initial_context", call_kwargs[0][0] if call_kwargs[0] else {}
        )
        assert initial_ctx.get("conv_user") == "Bob"

    @pytest.mark.asyncio
    async def test_error_returns_failure_result(self):
        """Errors during conversation produce a failure result, not an exception."""
        step = ConversationStep(
            step_id="conv1",
            name="Conv",
            fsm_definition={"name": "test", "initial_state": "s", "states": {}},
            success_state="done",
            error_state="error_state",
        )

        with patch("fsm_llm.API") as MockAPI:
            MockAPI.from_definition.side_effect = Exception("LLM unavailable")
            result = await step.execute({})

        assert result.success is False
        assert result.next_state == "error_state"
        assert "LLM unavailable" in result.message

    @pytest.mark.asyncio
    async def test_max_turns_respected(self):
        """Conversation stops after max_turns even if not ended."""
        mock_api = self._mock_api(responses=["r1", "r2", "r3", "r4", "r5"])
        mock_api.has_conversation_ended.return_value = False

        step = ConversationStep(
            step_id="conv1",
            name="Conv",
            fsm_definition={"name": "test", "initial_state": "s", "states": {}},
            success_state="done",
            max_turns=2,
            auto_messages=["m1", "m2", "m3", "m4", "m5"],
        )

        with patch("fsm_llm.API") as MockAPI:
            MockAPI.from_definition.return_value = mock_api
            await step.execute({})

        # Should only call converse max_turns times
        assert mock_api.converse.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
