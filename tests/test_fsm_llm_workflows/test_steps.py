"""
Unit tests for workflow step types.
Tests step creation, async execution, and error handling.
"""

from unittest.mock import MagicMock, patch

import pytest

from fsm_llm.workflows.models import WaitEventConfig
from fsm_llm.workflows.steps import (
    APIStep,
    AutoStep,
    ConditionStep,
    ConversationStep,
    ParallelStep,
    TimerStep,
    WaitEventStep,
)

# ---------------------------------------------------------------------------
# AutoStep
# ---------------------------------------------------------------------------


class TestAutoStep:
    """Test AutoStep execution."""

    @pytest.mark.asyncio
    async def test_execute_no_action(self):
        step = AutoStep(step_id="s1", name="Test", next_step_id="done")
        result = await step.execute({})

        assert result.success is True
        assert result.next_step_id == "done"

    @pytest.mark.asyncio
    async def test_execute_with_sync_action(self):
        def my_action(ctx):
            return {"computed": ctx.get("x", 0) * 2}

        step = AutoStep(
            step_id="s1", name="Test", next_step_id="done", action=my_action
        )
        result = await step.execute({"x": 5})

        assert result.success is True
        assert result.data == {"computed": 10}

    @pytest.mark.asyncio
    async def test_execute_with_async_action(self):
        async def my_action(ctx):
            return {"async_result": True}

        step = AutoStep(
            step_id="s1", name="Test", next_step_id="done", action=my_action
        )
        result = await step.execute({})

        assert result.success is True
        assert result.data == {"async_result": True}

    @pytest.mark.asyncio
    async def test_execute_action_error(self):
        def bad_action(ctx):
            raise ValueError("boom")

        step = AutoStep(
            step_id="s1", name="Test", next_step_id="done", action=bad_action
        )
        result = await step.execute({})
        assert result.success is False
        assert result.next_step_id is None
        assert result.error == "boom"


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
            true_step_id="go",
            false_step_id="wait",
        )
        result = await step.execute({"ready": True})

        assert result.success is True
        assert result.next_step_id == "go"
        assert result.data["condition_result"] is True

    @pytest.mark.asyncio
    async def test_condition_false(self):
        step = ConditionStep(
            step_id="s1",
            name="Check",
            condition=lambda ctx: ctx.get("ready", False),
            true_step_id="go",
            false_step_id="wait",
        )
        result = await step.execute({"ready": False})

        assert result.next_step_id == "wait"
        assert result.data["condition_result"] is False

    @pytest.mark.asyncio
    async def test_condition_async(self):
        async def check(ctx):
            return True

        step = ConditionStep(
            step_id="s1",
            name="Check",
            condition=check,
            true_step_id="go",
            false_step_id="wait",
        )
        result = await step.execute({})
        assert result.next_step_id == "go"

    @pytest.mark.asyncio
    async def test_condition_error(self):
        def bad_check(ctx):
            raise RuntimeError("check failed")

        step = ConditionStep(
            step_id="s1",
            name="Check",
            condition=bad_check,
            true_step_id="go",
            false_step_id="wait",
        )
        result = await step.execute({})
        assert result.success is False
        assert result.next_step_id is None
        assert result.error == "check failed"


# ---------------------------------------------------------------------------
# APIStep
# ---------------------------------------------------------------------------


class TestAPIStep:
    """Test APIStep execution."""

    @pytest.mark.asyncio
    async def test_success(self):
        def my_api(**params):
            return {"status": "ok", "result": params.get("query", "")}

        step = APIStep(
            step_id="s1",
            name="API",
            api_function=my_api,
            success_step_id="done",
            failure_step_id="error",
            input_mapping={"query": "search_term"},
            output_mapping={"response": "result"},
        )
        result = await step.execute({"search_term": "hello"})

        assert result.success is True
        assert result.next_step_id == "done"
        assert result.data.get("response") == "hello"

    @pytest.mark.asyncio
    async def test_failure(self):
        def failing_api(**params):
            raise ConnectionError("timeout")

        step = APIStep(
            step_id="s1",
            name="API",
            api_function=failing_api,
            success_step_id="done",
            failure_step_id="error",
        )
        result = await step.execute({})

        assert result.success is False
        assert result.next_step_id == "error"

    @pytest.mark.asyncio
    async def test_async_api(self):
        async def my_api(**params):
            return {"data": "async result"}

        step = APIStep(
            step_id="s1",
            name="API",
            api_function=my_api,
            success_step_id="done",
            failure_step_id="error",
            output_mapping={"result": "data"},
        )
        result = await step.execute({})

        assert result.success is True
        assert result.data.get("result") == "async result"


# ---------------------------------------------------------------------------
# WaitEventStep
# ---------------------------------------------------------------------------


class TestWaitEventStep:
    """Test WaitEventStep execution."""

    @pytest.mark.asyncio
    async def test_execute(self):
        step = WaitEventStep(
            step_id="s1",
            name="Wait",
            config=WaitEventConfig(
                event_type="payment",
                success_step_id="paid",
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
            step_id="s1", name="Delay", delay_seconds=60, next_step_id="next"
        )
        result = await step.execute({})

        assert result.success is True
        assert "_timer_info" in result.data
        info = result.data["_timer_info"]
        assert info["delay_seconds"] == 60
        assert info["next_step_id"] == "next"
        assert info["waiting_for_timer"] is True


# ---------------------------------------------------------------------------
# ParallelStep
# ---------------------------------------------------------------------------


class TestParallelStep:
    """Test ParallelStep execution."""

    @pytest.mark.asyncio
    async def test_all_succeed(self):
        s1 = AutoStep(
            step_id="a", name="A", next_step_id="done", action=lambda ctx: {"val": 1}
        )
        s2 = AutoStep(
            step_id="b", name="B", next_step_id="done", action=lambda ctx: {"val": 2}
        )

        step = ParallelStep(
            step_id="p1", name="Parallel", steps=[s1, s2], next_step_id="merged"
        )
        result = await step.execute({})

        assert result.success is True
        assert result.next_step_id == "merged"
        # Default aggregation prefixes by step index
        assert "step_0_val" in result.data
        assert "step_1_val" in result.data

    @pytest.mark.asyncio
    async def test_with_error_step_id(self):
        def bad_action(ctx):
            raise ValueError("fail")

        s1 = AutoStep(step_id="a", name="A", next_step_id="done", action=bad_action)
        s2 = AutoStep(step_id="b", name="B", next_step_id="done")

        step = ParallelStep(
            step_id="p1",
            name="Parallel",
            steps=[s1, s2],
            next_step_id="ok",
            error_step_id="err",
        )
        result = await step.execute({})

        assert result.success is False
        assert result.next_step_id == "err"

    @pytest.mark.asyncio
    async def test_custom_aggregation(self):
        s1 = AutoStep(
            step_id="a", name="A", next_step_id="done", action=lambda ctx: {"count": 1}
        )
        s2 = AutoStep(
            step_id="b", name="B", next_step_id="done", action=lambda ctx: {"count": 2}
        )

        def aggregate(results):
            total = sum(r.data.get("count", 0) for r in results if r.data)
            return {"total": total}

        step = ParallelStep(
            step_id="p1",
            name="Parallel",
            steps=[s1, s2],
            next_step_id="merged",
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

        s1 = AutoStep(step_id="a", name="A", next_step_id="done", action=mutate)
        s2 = AutoStep(
            step_id="b",
            name="B",
            next_step_id="done",
            action=lambda ctx: {"saw_mutation": ctx.get("mutated", False)},
        )

        step = ParallelStep(
            step_id="p1", name="Parallel", steps=[s1, s2], next_step_id="done"
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
        step = AutoStep(step_id="s1", name="Test", next_step_id="next")
        assert step.step_id == "s1"
        assert step.name == "Test"
        assert step.description == ""

    def test_step_with_description(self):
        step = AutoStep(
            step_id="s1", name="Test", next_step_id="next", description="desc"
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
        """ConversationStep must be rejected at construction if neither
        fsm_file nor fsm_definition is given (not only when it runs)."""
        with pytest.raises(ValueError, match="requires either"):
            ConversationStep(step_id="conv1", name="Conv", success_step_id="done")

    def test_rejects_both_fsm_file_and_definition(self):
        with pytest.raises(ValueError, match="exactly one"):
            ConversationStep(
                step_id="conv1",
                name="Conv",
                fsm_file="x.json",
                fsm_definition={"name": "t"},
            )

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
            success_step_id="process",
            auto_messages=["My name is Alice"],
            context_mapping={"user_name": "name", "user_age": "age"},
        )

        with patch("fsm_llm.workflows.steps.API") as MockAPI:
            MockAPI.from_definition.return_value = mock_api
            result = await step.execute({})

        assert result.success is True
        assert result.next_step_id == "process"
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
            success_step_id="done",
            initial_context={"conv_user": "workflow_user"},
        )

        with patch("fsm_llm.workflows.steps.API") as MockAPI:
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
            success_step_id="done",
            error_step_id="error_step_id",
        )

        with patch("fsm_llm.workflows.steps.API") as MockAPI:
            MockAPI.from_definition.side_effect = Exception("LLM unavailable")
            result = await step.execute({})

        assert result.success is False
        assert result.next_step_id == "error_step_id"
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
            success_step_id="done",
            max_turns=2,
            auto_messages=["m1", "m2", "m3", "m4", "m5"],
        )

        with patch("fsm_llm.workflows.steps.API") as MockAPI:
            MockAPI.from_definition.return_value = mock_api
            await step.execute({})

        # Should only call converse max_turns times
        assert mock_api.converse.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
