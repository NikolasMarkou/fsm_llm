"""
Unit tests for workflow step types.
Tests step creation, async execution, and error handling.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import Field, ValidationError

from fsm_llm.workflows import (
    WorkflowEngine,
    auto_step,
    create_workflow,
    llm_step,
    switch_step,
)
from fsm_llm.workflows.models import WaitEventConfig, WorkflowStatus, WorkflowStepResult
from fsm_llm.workflows.steps import (
    AgentStep,
    APIStep,
    AutoStep,
    ConditionStep,
    ConversationStep,
    LLMStep,
    ParallelStep,
    RetryStep,
    SwitchStep,
    TimerStep,
    WaitEventStep,
    WorkflowStep,
)
from tests.test_fsm_llm_workflows.support import _CoreLikeLLM

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

        with patch("fsm_llm.workflows.steps.conversation.API") as MockAPI:
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

        with patch("fsm_llm.workflows.steps.conversation.API") as MockAPI:
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

        with patch("fsm_llm.workflows.steps.conversation.API") as MockAPI:
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

        with patch("fsm_llm.workflows.steps.conversation.API") as MockAPI:
            MockAPI.from_definition.return_value = mock_api
            await step.execute({})

        # Should only call converse max_turns times
        assert mock_api.converse.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class _AgentResult:
    def __init__(self, answer, success=True, final_context=None):
        self.answer = answer
        self.success = success
        self.final_context = final_context or {}
        self.structured_output = None


class _RecordingAgent:
    def __init__(self, result):
        self.result = result
        self.calls: list[tuple[str, dict]] = []

    def run(self, task, **kwargs):
        self.calls.append((task, kwargs))
        return self.result


class TestLLMStepInterfaces:
    async def test_core_llm_interface_generate_response(self):
        llm = _CoreLikeLLM("ANSWER: 42")
        step = LLMStep(
            step_id="llm",
            name="LLM",
            llm_interface=llm,
            prompt_template="Q: {q}",
            context_mapping={"q": "question"},
            output_mapping={"answer": r"ANSWER: (\d+)", "raw": ""},
            next_step_id="done",
        )
        result = await step.execute({"question": "6*7"})

        assert result.success is True
        assert result.data == {"answer": "42", "raw": "ANSWER: 42"}
        assert llm.requests[0].user_message == "Q: 6*7"
        assert llm.requests[0].system_prompt

    async def test_sync_generate_returning_str(self):
        class _SyncLLM:
            def generate(self, prompt):
                return f"echo {prompt}"

        step = LLMStep(
            step_id="llm",
            name="LLM",
            llm_interface=_SyncLLM(),
            prompt_template="hi",
            output_mapping={"out": ""},
            next_step_id="done",
        )
        result = await step.execute({})
        assert result.success is True
        assert result.data["out"] == "echo hi"

    async def test_regex_miss_leaves_key_unset(self):
        step = LLMStep(
            step_id="llm",
            name="LLM",
            llm_interface=_CoreLikeLLM("no number here"),
            prompt_template="hi",
            output_mapping={"n": r"(\d+)"},
            next_step_id="done",
        )
        result = await step.execute({})
        assert result.success is True
        assert "n" not in result.data

    def test_invalid_regex_rejected_at_construction(self):
        with pytest.raises(ValueError, match="valid regex"):
            LLMStep(
                step_id="llm",
                name="LLM",
                llm_interface=_CoreLikeLLM(""),
                prompt_template="hi",
                output_mapping={"n": "(unclosed"},
                next_step_id="done",
            )


class TestFailClosedRouting:
    async def test_llm_failure_without_error_step_id_fails_instance(self):
        class _Broken:
            def generate(self, prompt):
                raise RuntimeError("provider down")

        wf = create_workflow("wf", "wf")
        wf.with_initial_step(
            LLMStep(
                step_id="llm",
                name="LLM",
                llm_interface=_Broken(),
                prompt_template="hi",
                next_step_id="after",
            )
        )
        wf.with_step(auto_step("after", "After", next_step_id=""))
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start("wf")

        inst = engine.workflow_instances[iid]
        assert inst.status == WorkflowStatus.FAILED
        assert inst.current_step_id == "llm"
        assert "provider down" in (inst.error or "")

    async def test_parallel_failure_without_error_step_id_has_no_route(self):
        bad = AutoStep(
            step_id="bad",
            name="Bad",
            next_step_id="x",
            action=lambda ctx: 1 / 0,
        )
        step = ParallelStep(step_id="p", name="P", steps=[bad], next_step_id="next")
        result = await step.execute({})
        assert result.success is False
        assert result.next_step_id is None

    async def test_agent_reporting_failure_fails_step(self):
        agent = _RecordingAgent(_AgentResult("gave up", success=False))
        step = AgentStep(
            step_id="a",
            name="A",
            agent=agent,
            success_step_id="next",
            error_step_id="err",
        )
        result = await step.execute({"task": "t"})
        assert result.success is False
        assert result.next_step_id == "err"
        assert result.data["agent_success"] is False

    async def test_conversation_failure_without_error_step_id_has_no_route(self):
        step = ConversationStep(
            step_id="c",
            name="C",
            fsm_definition={"name": "x"},
            success_step_id="next",
        )
        from unittest.mock import patch

        with patch("fsm_llm.workflows.steps.conversation.API") as mock_api:
            mock_api.from_definition.side_effect = RuntimeError("no model")
            result = await step.execute({})
        assert result.success is False
        assert result.next_step_id is None


class TestParallelEmptyErrors:
    async def test_exception_with_empty_message_is_a_failure(self):
        class _Raises(WorkflowStep):
            async def execute(self, context):
                raise RuntimeError()

        class _SilentFail(WorkflowStep):
            async def execute(self, context):
                return WorkflowStepResult(success=False)

        for child in (
            _Raises(step_id="c", name="C"),
            _SilentFail(step_id="c", name="C"),
        ):
            step = ParallelStep(
                step_id="p",
                name="P",
                steps=[child],
                next_step_id="n",
                error_step_id="e",
            )
            result = await step.execute({})
            assert result.success is False, type(child).__name__
            assert result.next_step_id == "e"
            assert result.error


class TestAwaitableReturningCallables:
    async def test_condition_lambda_returning_coroutine(self):
        async def is_big(ctx):
            return ctx["n"] > 10

        step = ConditionStep(
            step_id="c",
            name="C",
            condition=lambda ctx: is_big(ctx),
            true_step_id="big",
            false_step_id="small",
        )
        assert (await step.execute({"n": 1})).next_step_id == "small"
        assert (await step.execute({"n": 99})).next_step_id == "big"

    async def test_api_lambda_returning_coroutine(self):
        async def fetch(**kwargs):
            return {"value": 7}

        step = APIStep(
            step_id="api",
            name="API",
            api_function=lambda **kw: fetch(**kw),
            success_step_id="ok",
            failure_step_id="bad",
            output_mapping={"v": "value"},
        )
        result = await step.execute({})
        assert result.success is True
        assert result.data == {"v": 7}

    async def test_async_callable_object(self):
        class _AsyncAction:
            async def __call__(self, ctx):
                return {"done": True}

        step = AutoStep(step_id="a", name="A", next_step_id="n", action=_AsyncAction())
        assert (await step.execute({})).data == {"done": True}


class TestSwitchDefaults:
    async def test_dsl_default_fails_on_unmatched_value(self):
        wf = create_workflow("sw", "sw")
        wf.with_initial_step(switch_step("route", "Route", "k", cases={"a": "done"}))
        wf.with_step(auto_step("done", "Done", next_step_id=""))
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start("sw", {"k": "zzz"})
        assert engine.get_status(iid) == WorkflowStatus.FAILED

    async def test_switch_outputs_reach_context(self):
        wf = create_workflow("sw", "sw")
        wf.with_initial_step(switch_step("route", "Route", "k", cases={"a": "done"}))
        wf.with_step(auto_step("done", "Done", next_step_id=""))
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start("sw", {"k": "a"})
        ctx = engine.get_context(iid)
        assert ctx["switch_route_matched"] == "a"
        assert ctx["switch_route_target"] == "done"


class TestParallelChildren:
    async def test_internal_keys_do_not_leak_through_prefix(self):
        child = AutoStep(
            step_id="c",
            name="C",
            next_step_id="x",
            action=lambda ctx: {"_secret": 1, "system_x": 2, "ok": 3},
        )
        step = ParallelStep(step_id="p", name="P", steps=[child], next_step_id="n")
        result = await step.execute({})
        assert result.data == {"step_0_ok": 3}

    def test_timer_child_rejected(self):
        with pytest.raises(ValueError, match="cannot run inside a ParallelStep"):
            ParallelStep(
                step_id="p",
                name="P",
                steps=[
                    TimerStep(step_id="t", name="T", delay_seconds=1, next_step_id="x")
                ],
                next_step_id="n",
            )

    async def test_partial_data_kept_on_failure(self):
        good = AutoStep(
            step_id="g", name="G", next_step_id="x", action=lambda ctx: {"v": 1}
        )
        bad = AutoStep(
            step_id="b", name="B", next_step_id="x", action=lambda ctx: 1 / 0
        )
        step = ParallelStep(
            step_id="p",
            name="P",
            steps=[good, bad],
            next_step_id="n",
            error_step_id="e",
        )
        result = await step.execute({})
        assert result.success is False
        assert result.data == {"step_0_v": 1}

    async def test_uncopyable_context_falls_back_to_shallow_copy(self):
        import threading

        child = AutoStep(step_id="c", name="C", next_step_id="x")
        step = ParallelStep(step_id="p", name="P", steps=[child], next_step_id="n")
        result = await step.execute({"lock": threading.Lock()})
        assert result.success is True


class TestAgentStepMapping:
    async def test_answer_mapped_with_empty_final_context(self):
        agent = _RecordingAgent(_AgentResult("42"))
        step = AgentStep(
            step_id="a",
            name="A",
            agent=agent,
            success_step_id="next",
            context_mapping={"result": "answer"},
        )
        result = await step.execute({"task": "t"})
        assert result.data["result"] == "42"
        assert result.data["agent_a_answer"] == "42"

    async def test_input_mapping_passes_initial_context(self):
        agent = _RecordingAgent(_AgentResult("ok"))
        step = AgentStep(
            step_id="a",
            name="A",
            agent=agent,
            success_step_id="next",
            input_mapping={"customer": "cust_id"},
        )
        await step.execute({"task": "t", "cust_id": "C-1"})
        assert agent.calls == [("t", {"initial_context": {"customer": "C-1"}})]

    async def test_plain_string_result(self):
        class _StrAgent:
            def run(self, task):
                return f"done: {task}"

        step = AgentStep(step_id="a", name="A", agent=_StrAgent(), success_step_id="n")
        result = await step.execute({"task": "t"})
        assert result.success is True
        assert result.data["agent_answer"] == "done: t"

    def test_agent_without_run_rejected(self):
        with pytest.raises(ValueError, match="run"):
            AgentStep(step_id="a", name="A", agent=object())


class TestConversationStepFixes:
    @staticmethod
    def _mock_api(ended: bool = False, delay: float = 0.0):
        from unittest.mock import MagicMock

        api = MagicMock()

        def start(initial_context=None):
            if delay:
                import time

                time.sleep(delay)
            return ("conv", "hello")

        api.start_conversation.side_effect = start
        api.converse.return_value = "reply"
        api.has_conversation_ended.return_value = ended
        api.get_data.return_value = {"k": "v"}
        return api

    async def test_base_timeout_is_honoured(self, timing_factor):
        from unittest.mock import patch

        step = ConversationStep(
            step_id="c",
            name="C",
            fsm_definition={"name": "x"},
            timeout_seconds=0.05 * timing_factor,
            error_step_id="err",
        )
        with patch("fsm_llm.workflows.steps.conversation.API") as mock_cls:
            mock_cls.from_definition.return_value = self._mock_api(
                delay=0.3 * timing_factor
            )
            result = await step.execute({})
        assert result.success is False
        assert result.next_step_id == "err"
        assert "timed out" in result.error

    async def test_ended_flag_and_require_completion(self):
        from unittest.mock import patch

        step = ConversationStep(
            step_id="c",
            name="C",
            fsm_definition={"name": "x"},
            success_step_id="next",
            error_step_id="err",
            require_completion=True,
        )
        with patch("fsm_llm.workflows.steps.conversation.API") as mock_cls:
            mock_cls.from_definition.return_value = self._mock_api(ended=False)
            result = await step.execute({})
        assert result.success is False
        assert result.next_step_id == "err"
        assert result.data["conversation_c_ended"] is False

    async def test_user_input_is_sent_when_enabled(self):
        from unittest.mock import patch

        api = self._mock_api()
        step = ConversationStep(
            step_id="c",
            name="C",
            fsm_definition={"name": "x"},
            use_user_input=True,
        )
        with patch("fsm_llm.workflows.steps.conversation.API") as mock_cls:
            mock_cls.from_definition.return_value = api
            await step.execute({"_user_input": "hello there"})
        api.converse.assert_called_once_with(
            user_message="hello there", conversation_id="conv"
        )

    def test_accepts_fsm_definition_model_object(self):
        class _FakeFSMDefinition:
            def model_dump(self):
                return {}

        obj = _FakeFSMDefinition()
        step = ConversationStep(step_id="c", name="C", fsm_definition=obj)
        assert step.fsm_definition is obj


class TestLLMInterfaceValidatedBehaviour:
    def test_object_without_generate_is_refused(self):
        with pytest.raises(ValueError, match="generate"):
            llm_step("l", "L", object(), "hello")

    def test_none_is_refused(self):
        with pytest.raises(ValueError, match="generate"):
            llm_step("l", "L", None, "hello")

    def test_generate_and_generate_response_are_accepted(self):
        class _Gen:
            def generate(self, prompt: str) -> str:
                return "x"

        class _Resp:
            def generate_response(self, request: Any) -> Any:
                return "x"

        assert llm_step("l", "L", _Gen(), "hello").step_id == "l"
        assert llm_step("l", "L", _Resp(), "hello").step_id == "l"


class TestParallelStepErrorHandling:
    """F-001: ParallelStep should report failure even without error_step_id."""

    async def test_parallel_step_fails_when_substeps_fail_no_error_step_id(self):
        """ParallelStep with no error_step_id must still return failure if substeps fail."""
        failing_step = AutoStep(
            step_id="fail",
            name="Failing",
            next_step_id="done",
            action=lambda ctx: (_ for _ in ()).throw(ValueError("boom")),
        )
        ok_step = AutoStep(
            step_id="ok",
            name="OK",
            next_step_id="done",
        )

        parallel = ParallelStep(
            step_id="par",
            name="Parallel",
            steps=[ok_step, failing_step],
            next_step_id="next",
            error_step_id=None,  # No error_step_id configured
        )

        result = await parallel.execute({})

        # Must be failure, NOT success
        assert result.success is False
        assert (
            "error" in result.message.lower() or "error" in (result.error or "").lower()
        )

    async def test_parallel_step_fails_with_error_step_id(self):
        """ParallelStep with error_step_id transitions to it on failure."""
        failing_step = AutoStep(
            step_id="fail",
            name="Failing",
            next_step_id="done",
            action=lambda ctx: (_ for _ in ()).throw(RuntimeError("fail")),
        )

        parallel = ParallelStep(
            step_id="par",
            name="Parallel",
            steps=[failing_step],
            next_step_id="next",
            error_step_id="error_handler",
        )

        result = await parallel.execute({})

        assert result.success is False
        assert result.next_step_id == "error_handler"

    async def test_parallel_step_succeeds_when_all_ok(self):
        """ParallelStep returns success when all substeps succeed."""
        step_a = AutoStep(step_id="a", name="A", next_step_id="done")
        step_b = AutoStep(step_id="b", name="B", next_step_id="done")

        parallel = ParallelStep(
            step_id="par",
            name="Parallel",
            steps=[step_a, step_b],
            next_step_id="next",
        )

        result = await parallel.execute({})

        assert result.success is True
        assert result.next_step_id == "next"


class TestConversationStepResourceCleanup:
    """F-004: end_conversation must be called even on exception."""

    async def test_end_conversation_called_on_exception(self):
        """Verify end_conversation is called in finally block."""
        step = ConversationStep(
            step_id="conv",
            name="Conv",
            fsm_file="fake.json",
            success_step_id="done",
            auto_messages=["msg1"],
        )

        mock_api = MagicMock()
        mock_api.start_conversation.return_value = ("conv-1", "Hello")
        mock_api.has_conversation_ended.return_value = False
        mock_api.converse.side_effect = RuntimeError("LLM failed")
        mock_api.end_conversation = MagicMock()

        # API is imported inside execute(), so we patch it at the source module
        mock_api_class = MagicMock()
        mock_api_class.from_file.return_value = mock_api

        with patch.dict("sys.modules", {}):
            with patch("fsm_llm.workflows.steps.conversation.API", mock_api_class):
                with patch("fsm_llm.engine.api.conversation.API", mock_api_class):
                    result = await step.execute({})

        # end_conversation MUST be called despite the exception
        mock_api.end_conversation.assert_called_once_with("conv-1")
        # Result should be a failure
        assert result.success is False


class TestLLMStepTemplateError:
    """F-007: Missing template variables should raise WorkflowStepError, not KeyError."""

    async def test_missing_template_variable_raises_step_error(self):
        """Template with {missing_var} should raise WorkflowStepError."""
        mock_llm = MagicMock()

        step = LLMStep(
            step_id="llm",
            name="LLM Step",
            llm_interface=mock_llm,
            prompt_template="Hello {name}, your order {order_id} is ready",
            context_mapping={"name": "user_name"},  # order_id not mapped
            output_mapping={"result": ".*"},
            next_step_id="done",
        )

        # Context has user_name but template needs order_id too
        context = {"user_name": "Alice"}

        result = await step.execute(context)
        # Should fail gracefully instead of crashing with raw KeyError
        assert result.success is False
        assert "order_id" in result.error.lower() or "template" in result.error.lower()


# ---------------------------------------------------------------------------
# package layout (step 92 of plan d454fd62)
# ---------------------------------------------------------------------------

# class name -> the module of ``fsm_llm.workflows.steps`` that defines it
_STEP_MODULE_OF = {
    "WorkflowStep": "base",
    "AutoStep": "routing",
    "ConditionStep": "routing",
    "SwitchStep": "routing",
    "APIStep": "calls",
    "LLMStep": "calls",
    "WaitEventStep": "waits",
    "TimerStep": "waits",
    "ConversationStep": "conversation",
    "AgentStep": "agent",
    "ParallelStep": "composite",
    "RetryStep": "composite",
}


def _minimal_step_kwargs() -> dict[str, dict[str, Any]]:
    """Valid keyword arguments for each of the 11 step kinds."""
    inner = AutoStep(step_id="inner", name="Inner", next_step_id="")
    return {
        "AutoStep": {"next_step_id": "n"},
        "ConditionStep": {
            "condition": lambda c: True,
            "true_step_id": "a",
            "false_step_id": "b",
        },
        "SwitchStep": {"key": "k", "cases": {"a": "x"}},
        "APIStep": {
            "api_function": lambda: 1,
            "success_step_id": "a",
            "failure_step_id": "b",
        },
        "LLMStep": {
            "llm_interface": MagicMock(),
            "prompt_template": "hi",
            "next_step_id": "n",
        },
        "WaitEventStep": {
            "config": WaitEventConfig(event_type="e", success_step_id="a")
        },
        "TimerStep": {"delay_seconds": 1, "next_step_id": "n"},
        "ConversationStep": {"fsm_definition": {"x": 1}},
        "AgentStep": {"agent": MagicMock()},
        "ParallelStep": {"steps": [inner], "next_step_id": "n"},
        "RetryStep": {"step": inner},
    }


class TestWorkflowStepsPackageLayout:
    """``workflows/steps.py`` is a package at the same dotted path: the base in
    one module, the 11 kinds in six modules by what they do."""

    def test_the_facade_exports_exactly_the_twelve_public_names(self):
        import fsm_llm.workflows.steps as facade

        assert sorted(facade.__all__) == sorted(_STEP_MODULE_OF)
        namespace: dict = {}
        exec("from fsm_llm.workflows.steps import *", namespace)
        assert set(_STEP_MODULE_OF) <= set(namespace)
        for private in (
            "_STEP_EXECUTOR",
            "_call_user_callable",
            "_is_async_callable",
            "_error_text",
            "_lookup_path",
        ):
            assert private not in namespace
            assert not hasattr(facade, private), private
        # `API` is read by `ConversationStep` in its own module, never the facade
        assert not hasattr(facade, "API")

    def test_every_class_is_defined_in_the_module_that_owns_it(self):
        import importlib

        import fsm_llm.workflows.steps as facade

        for name, module in _STEP_MODULE_OF.items():
            cls = getattr(facade, name)
            assert cls.__module__ == f"fsm_llm.workflows.steps.{module}", name
            defining = importlib.import_module(f"fsm_llm.workflows.steps.{module}")
            assert vars(defining)[name] is cls

    def test_every_kind_inherits_the_forbidding_base_and_refuses_an_unknown_keyword(
        self,
    ):
        import fsm_llm.workflows.steps as facade

        kwargs = _minimal_step_kwargs()
        assert set(kwargs) == set(_STEP_MODULE_OF) - {"WorkflowStep"}
        assert WorkflowStep.model_config["extra"] == "forbid"
        for name, required in kwargs.items():
            cls = getattr(facade, name)
            assert issubclass(cls, WorkflowStep), name
            assert cls.model_config["extra"] == "forbid", name
            cls(step_id="s", name="S", **required)
            with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
                cls(step_id="s", name="S", next_state="x", **required)

    def test_serialize_tags_are_the_class_names_not_a_module_path(self):
        import fsm_llm.workflows.steps as facade
        from fsm_llm.workflows import WorkflowDefinition

        for name, required in _minimal_step_kwargs().items():
            step = getattr(facade, name)(step_id="s", name="S", **required)
            definition = WorkflowDefinition(
                workflow_id="w", name="W", steps={"s": step}, initial_step_id="s"
            )
            dumped = definition.serialize()["steps"]["s"]
            assert dumped["type"] == name
            assert "fsm_llm" not in str(dumped["type"])

    def test_the_step_executor_is_one_object_defined_in_one_module(self):
        import ast
        from pathlib import Path

        import fsm_llm.workflows.steps as facade
        from fsm_llm.workflows.engine import driver
        from fsm_llm.workflows.steps import conversation, helpers

        assert driver._STEP_EXECUTOR is helpers._STEP_EXECUTOR
        assert conversation._STEP_EXECUTOR is helpers._STEP_EXECUTOR
        assert helpers._STEP_EXECUTOR.name == "fsm_llm_workflows_step_executor"
        defined_in = []
        for path in sorted(Path(facade.__file__).parent.glob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                targets = []
                if isinstance(node, ast.Assign):
                    targets = [t for t in node.targets if isinstance(t, ast.Name)]
                elif isinstance(node, ast.AnnAssign) and isinstance(
                    node.target, ast.Name
                ):
                    targets = [node.target]
                if any(t.id == "_STEP_EXECUTOR" for t in targets):
                    defined_in.append(path.stem)
        assert defined_in == ["helpers"]

    async def test_the_executor_the_driver_sets_is_the_one_the_steps_use(self):
        import threading
        from concurrent.futures import ThreadPoolExecutor

        seen: list[str] = []

        def action(ctx):
            seen.append(threading.current_thread().name)
            return {}

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="step92")
        try:
            engine = WorkflowEngine(executor=executor)
            engine.register_workflow(
                create_workflow("wf", "W").with_initial_step(
                    auto_step("a", "A", next_step_id="", action=action)
                )
            )
            await engine.start("wf")
        finally:
            executor.shutdown()
        assert seen and all(name.startswith("step92") for name in seen)

    def test_a_patch_of_api_aims_at_the_module_that_reads_it(self):
        import fsm_llm.workflows.steps as facade
        from fsm_llm.workflows.steps import conversation

        assert conversation.API.__name__ == "API"
        with pytest.raises(AttributeError):
            with patch("fsm_llm.workflows.steps.API"):
                pass  # pragma: no cover
        assert not hasattr(facade, "API")

    async def test_a_patch_at_the_conversation_module_reaches_the_step(self):
        mock_api = MagicMock()
        mock_api.start_conversation.return_value = ("c1", "hello")
        mock_api.has_conversation_ended.return_value = True
        mock_api.get_data.return_value = {"k": "v"}
        step = ConversationStep(
            step_id="c",
            name="C",
            fsm_definition={"x": 1},
            context_mapping={"out": "k"},
            success_step_id="done",
        )
        with patch("fsm_llm.workflows.steps.conversation.API") as api_class:
            api_class.from_definition.return_value = mock_api
            result = await step.execute({})
        api_class.from_definition.assert_called_once()
        assert result.success is True
        assert result.data["out"] == "v"

    def test_every_decision_anchor_sits_with_the_code_it_constrains(self):
        from pathlib import Path

        import fsm_llm.workflows.steps as facade

        root = Path(facade.__file__).parent
        expected = {
            "helpers": ("5d1e7a3b/D-004",),
            "composite": ("5d1e7a3b/D-003", "089d0ec7/D-010"),
            "conversation": ("07ad3f8c/D-039",),
        }
        for module, tokens in expected.items():
            text = (root / f"{module}.py").read_text(encoding="utf-8")
            for token in tokens:
                assert token in text, (module, token)
        assert (root / "composite.py").read_text(encoding="utf-8").count(
            "5d1e7a3b/D-003"
        ) == 2

    async def test_a_log_record_is_named_after_the_module_that_logs_it(self):
        from loguru import logger

        records: list[tuple[str, str]] = []
        extras: list[tuple[str, dict]] = []
        logger.enable("fsm_llm")
        sink = logger.add(
            lambda m: (
                records.append((m.record["name"], m.record["message"])),
                extras.append((m.record["name"], dict(m.record["extra"]))),
            ),
            level="DEBUG",
            format="{message}",
        )

        def boom(ctx):
            raise ValueError("boom")

        try:
            await AutoStep(
                step_id="a", name="A", next_step_id="n", action=boom
            ).execute({})
            await APIStep(
                step_id="b",
                name="B",
                api_function=lambda: {},
                success_step_id="x",
                failure_step_id="y",
                output_mapping={"o": "missing"},
            ).execute({})
            await AgentStep(
                step_id="g", name="G", agent=MagicMock(run=boom), task_template="t"
            ).execute({})
        finally:
            logger.remove(sink)
            logger.disable("fsm_llm")

        def logged_by(prefix: str) -> str:
            return next(n for n, message in records if message.startswith(prefix))

        assert logged_by("Error in auto transition step") == (
            "fsm_llm.workflows.steps.routing"
        )
        assert logged_by("APIStep [b]") == "fsm_llm.workflows.steps.calls"
        assert logged_by("Agent step 'g' failed") == "fsm_llm.workflows.steps.agent"
        # no step binds extra context onto its records: besides the two keys the
        # library's own record patcher adds once a test has set logging up, a
        # record of a step carries nothing
        step_extras = [e for n, e in extras if n.startswith("fsm_llm.workflows.steps")]
        assert len(step_extras) == 3
        assert all(set(e) <= {"conversation_id", "package"} for e in step_extras)


# ---------------------------------------------------------------------------
# edges of each step module that the split of step 92 found unpinned
# ---------------------------------------------------------------------------


class _NoExecutor:
    """An executor that fails when asked to run anything."""

    def submit(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise AssertionError("an async callable must not run in the executor")


class TestStepModuleHelpers:
    """``steps/helpers.py`` and the private helper of ``steps/calls.py``."""

    def test_error_text_is_never_empty(self):
        from fsm_llm.workflows.steps.helpers import _error_text

        assert _error_text(ValueError("x")) == "x"
        assert _error_text(KeyError("k")) == "'k'"
        assert _error_text(TimeoutError()) == "TimeoutError"
        assert _error_text(RuntimeError("")) == "RuntimeError"

    def test_which_callables_count_as_async(self):
        import functools

        from fsm_llm.workflows.steps.helpers import _is_async_callable

        async def coro():
            return 1

        class AsyncCallable:
            async def __call__(self):
                return 1

        class SyncCallable:
            def __call__(self):
                return 1

        assert _is_async_callable(coro) is True
        assert _is_async_callable(AsyncCallable()) is True
        assert _is_async_callable(functools.partial(coro)) is True
        assert _is_async_callable(lambda: 1) is False
        assert _is_async_callable(SyncCallable()) is False
        assert _is_async_callable(functools.partial(lambda: 1)) is False
        assert _is_async_callable(5) is False
        assert _is_async_callable(None) is False

    async def test_a_callable_object_with_an_async_call_is_awaited_in_place(self):
        from fsm_llm.workflows.steps.helpers import _STEP_EXECUTOR, _call_user_callable

        class AsyncCallable:
            async def __call__(self, a, b=0):
                return ("obj", a, b)

        token = _STEP_EXECUTOR.set(_NoExecutor())  # type: ignore[arg-type]
        try:
            assert await _call_user_callable(AsyncCallable(), 1, b=2) == ("obj", 1, 2)
        finally:
            _STEP_EXECUTOR.reset(token)

    async def test_a_sync_callable_runs_in_the_step_executor_with_its_arguments(self):
        import threading
        from concurrent.futures import ThreadPoolExecutor

        from fsm_llm.workflows.steps.helpers import _STEP_EXECUTOR, _call_user_callable

        names: list[str] = []

        def sync(a, b=0):
            names.append(threading.current_thread().name)
            return (a, b)

        async def later():
            return "awaited"

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="step92b")
        token = _STEP_EXECUTOR.set(executor)
        try:
            assert await _call_user_callable(sync, 1, b=2) == (1, 2)
            # a sync factory that RETURNS an awaitable is awaited too
            assert await _call_user_callable(lambda: later()) == "awaited"
        finally:
            _STEP_EXECUTOR.reset(token)
            executor.shutdown()
        assert names == ["step92b_0"]

    def test_lookup_path_resolves_a_result_key(self):
        from fsm_llm.workflows.steps.calls import _lookup_path

        nested = {"a": {"b": {"c": [1]}}, "a.b": "exact"}
        assert _lookup_path(nested, "a.b") == (True, "exact")  # an exact key wins
        assert _lookup_path({"a": {"b": 1}}, "a.b") == (True, 1)
        assert _lookup_path({"a": {"b": {"c": [1]}}}, "a.b.c") == (True, [1])
        assert _lookup_path(5, "") == (True, 5)  # "" is the whole result
        assert _lookup_path([1], "x") == (False, None)  # not a dict
        assert _lookup_path({"a": None}, "a.b") == (False, None)
        assert _lookup_path({"a": {"b": 1}}, "a.") == (False, None)


class TestStepKindEdges:
    """Edges of the routing, call, wait and composite modules."""

    async def test_a_switch_compares_a_missing_or_none_value_as_empty_text(self):
        step = SwitchStep(
            step_id="s",
            name="S",
            key="k",
            cases={"": "empty", "None": "text"},
            default_step_id="other",
        )
        for context in ({}, {"k": None}):
            result = await step.execute(context)
            assert result.next_step_id == "empty", context
            assert result.data["switch_s_matched"] == ""
        assert (await step.execute({"k": "None"})).next_step_id == "text"

    async def test_a_switch_matches_case_sensitively(self):
        step = SwitchStep(
            step_id="s", name="S", key="k", cases={"a": "x"}, default_step_id="d"
        )
        assert (await step.execute({"k": "a"})).next_step_id == "x"
        assert (await step.execute({"k": "A"})).next_step_id == "d"

    def test_a_switch_references_every_target_but_no_empty_marker(self):
        step = SwitchStep(
            step_id="s",
            name="S",
            key="k",
            cases={"a": "x", "b": "", "c": "y"},
            default_step_id="",
        )
        assert step.referenced_step_ids() == {"x", "y"}
        with_default = SwitchStep(
            step_id="s", name="S", key="k", cases={"a": "x"}, default_step_id="d"
        )
        assert with_default.referenced_step_ids() == {"x", "d"}

    async def test_an_api_output_key_that_is_an_exact_key_beats_a_dotted_path(self):
        step = APIStep(
            step_id="a",
            name="A",
            api_function=lambda: {"user.name": "exact", "user": {"name": "nested"}},
            success_step_id="ok",
            failure_step_id="bad",
            output_mapping={"who": "user.name"},
        )
        assert (await step.execute({})).data == {"who": "exact"}

    async def test_an_api_input_key_missing_from_the_context_is_omitted(self):
        seen: list[dict[str, Any]] = []

        def api(**params):
            seen.append(params)
            return {}

        step = APIStep(
            step_id="a",
            name="A",
            api_function=api,
            success_step_id="ok",
            failure_step_id="bad",
            input_mapping={"p": "present", "q": "absent"},
        )
        result = await step.execute({"present": 1})
        assert result.success is True and result.next_step_id == "ok"
        assert seen == [{"p": 1}]

    async def test_an_llm_regex_matches_across_lines(self):
        step = LLMStep(
            step_id="l",
            name="L",
            llm_interface=_CoreLikeLLM("first\nsecond"),
            prompt_template="go",
            output_mapping={"both": "first.second", "head": r"(\w+)\n"},
            next_step_id="n",
        )
        result = await step.execute({})
        assert result.data == {"both": "first\nsecond", "head": "first"}

    async def test_an_empty_llm_system_prompt_falls_back_to_the_default_text(self):
        llm = _CoreLikeLLM("done")
        step = LLMStep(
            step_id="l",
            name="L",
            llm_interface=llm,
            prompt_template="go",
            system_prompt="",
            next_step_id="n",
        )
        await step.execute({})
        assert llm.requests[0].system_prompt == "Follow the user's instructions."


class _Outcome(WorkflowStep):
    """Does what ``kind`` says: ``ok`` (with ``data``), ``fail``, ``raise`` or ``garbage``."""

    kind: str = "ok"
    data: dict[str, Any] = Field(default_factory=dict)
    calls: int = 0

    async def execute(self, context: dict[str, Any]) -> Any:
        self.calls += 1
        if self.kind == "raise":
            raise ValueError("boom")
        if self.kind == "garbage":
            return "text"
        if self.kind == "fail":
            return WorkflowStepResult.failure_result(error="bad")
        return WorkflowStepResult.success_result(data=dict(self.data))


def _conversation_api(ended: bool = True) -> MagicMock:
    api = MagicMock()
    api.start_conversation.return_value = ("c1", "hello")
    api.has_conversation_ended.return_value = ended
    api.converse.return_value = "reply"
    api.get_data.return_value = {}
    return api


class TestStepKindEdgesWaitsAndConversation:
    """Edges of ``steps/waits.py``, ``steps/conversation.py`` and ``steps/agent.py``."""

    async def test_a_wait_step_reports_the_whole_wait_and_when_it_began(self):
        from datetime import datetime

        config = WaitEventConfig(
            event_type="e",
            success_step_id="a",
            timeout_seconds=2.0,
            timeout_step_id="t",
            event_mapping={"g": "v"},
            correlation_key="c",
        )
        result = await WaitEventStep(step_id="w", name="W", config=config).execute({})
        info = result.data["_waiting_info"]
        since = info.pop("waiting_since")
        assert datetime.fromisoformat(since).tzinfo is not None
        assert info == {
            "waiting_for_event": True,
            "event_type": "e",
            "timeout_seconds": 2.0,
            "timeout_step_id": "t",
            "success_step_id": "a",
            "event_mapping": {"g": "v"},
            "correlation_key": "c",
        }
        assert result.next_step_id is None

    async def test_a_timer_step_reports_when_the_timer_starts_and_ends(self):
        from datetime import datetime

        result = await TimerStep(
            step_id="t", name="T", delay_seconds=90, next_step_id="n"
        ).execute({})
        info = result.data["_timer_info"]
        start = datetime.fromisoformat(info.pop("timer_start"))
        end = datetime.fromisoformat(info.pop("timer_end"))
        assert 89.0 < (end - start).total_seconds() < 91.0
        assert info == {
            "waiting_for_timer": True,
            "delay_seconds": 90,
            "next_step_id": "n",
        }

    def test_a_conversation_has_the_smaller_of_its_two_time_limits(self):
        def limit(own, conversation):
            return ConversationStep(
                step_id="c",
                name="C",
                fsm_definition={"x": 1},
                timeout_seconds=own,
                conversation_timeout=conversation,
            )._effective_timeout()

        assert limit(None, None) is None
        assert limit(5, 2) == 2
        assert limit(2, 5) == 2
        assert limit(None, 3) == 3
        assert limit(3, None) == 3

    def test_a_conversation_needs_at_least_one_turn(self):
        for bad in (0, -1):
            with pytest.raises(ValidationError, match="max_turns must be >= 1"):
                ConversationStep(
                    step_id="c", name="C", fsm_definition={"x": 1}, max_turns=bad
                )
        assert (
            ConversationStep(
                step_id="c", name="C", fsm_definition={"x": 1}, max_turns=1
            ).max_turns
            == 1
        )

    async def test_the_user_input_is_sent_only_when_the_step_asks_for_it(self):
        for use_user_input, sent in ((False, []), (True, ["hi"])):
            api = _conversation_api(ended=False)
            step = ConversationStep(
                step_id="c",
                name="C",
                fsm_definition={"x": 1},
                use_user_input=use_user_input,
            )
            with patch("fsm_llm.workflows.steps.conversation.API") as api_class:
                api_class.from_definition.return_value = api
                await step.execute({"_user_input": "hi"})
            assert [c.kwargs["user_message"] for c in api.converse.call_args_list] == (
                sent
            ), use_user_input

    async def test_the_model_is_passed_to_the_api(self):
        for kwargs, factory in (
            ({"fsm_definition": {"x": 1}}, "from_definition"),
            ({"fsm_file": "x.json"}, "from_file"),
        ):
            step = ConversationStep(step_id="c", name="C", model="some-model", **kwargs)
            with patch("fsm_llm.workflows.steps.conversation.API") as api_class:
                getattr(api_class, factory).return_value = _conversation_api()
                await step.execute({})
            assert getattr(api_class, factory).call_args.kwargs == {
                "model": "some-model"
            }

    async def test_the_conversation_runs_in_the_executor_the_driver_set(self):
        import threading
        from concurrent.futures import ThreadPoolExecutor

        from fsm_llm.workflows.steps.helpers import _STEP_EXECUTOR

        names: list[str] = []
        api = _conversation_api()
        api.start_conversation.side_effect = lambda **kw: (
            names.append(threading.current_thread().name) or ("c1", "hello")
        )
        step = ConversationStep(step_id="c", name="C", fsm_definition={"x": 1})
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="conv92")
        token = _STEP_EXECUTOR.set(executor)
        try:
            with patch("fsm_llm.workflows.steps.conversation.API") as api_class:
                api_class.from_definition.return_value = api
                await step.execute({})
        finally:
            _STEP_EXECUTOR.reset(token)
            executor.shutdown()
        assert names == ["conv92_0"]

    async def test_an_agent_result_without_a_success_flag_is_a_failure(self):
        class _Result:
            answer = "partial"

        agent = MagicMock()
        agent.run.return_value = _Result()
        result = await AgentStep(step_id="a", name="A", agent=agent).execute(
            {"task": "go"}
        )
        assert result.success is False
        assert "Agent reported failure: partial" in str(result.error)

    async def test_a_missing_task_variable_is_named_in_the_error(self):
        step = AgentStep(
            step_id="a", name="A", agent=MagicMock(), task_template="do {topic}"
        )
        result = await step.execute({})
        assert result.success is False
        assert "Missing context key for task template: 'topic'" in str(result.error)

    async def test_a_missing_prompt_variable_is_named_in_the_error(self):
        step = LLMStep(
            step_id="l",
            name="L",
            llm_interface=_CoreLikeLLM("x"),
            prompt_template="hi {nope}",
            next_step_id="n",
        )
        result = await step.execute({})
        assert result.success is False
        assert "Prompt template variable 'nope' not found" in str(result.error)


class TestStepKindEdgesComposite:
    """Edges of ``steps/composite.py``."""

    def test_a_wait_or_timer_child_is_refused_however_deeply_it_is_wrapped(self):
        wait = WaitEventStep(
            step_id="w",
            name="W",
            config=WaitEventConfig(event_type="e", success_step_id="a"),
        )
        timer = TimerStep(step_id="t", name="T", delay_seconds=1, next_step_id="n")
        wrapped = [
            RetryStep(step_id="r", name="R", step=wait),
            RetryStep(
                step_id="r2",
                name="R2",
                step=RetryStep(step_id="r1", name="R1", step=timer),
            ),
            timer,
        ]
        for child in wrapped:
            with pytest.raises(ValidationError, match="cannot run inside a Parallel"):
                ParallelStep(step_id="p", name="P", steps=[child], next_step_id="n")
        ParallelStep(
            step_id="p",
            name="P",
            steps=[
                RetryStep(step_id="r", name="R", step=_Outcome(step_id="o", name="O"))
            ],
            next_step_id="n",
        )

    async def test_a_raising_child_does_not_cancel_the_others(self):
        ok = _Outcome(step_id="a", name="A", data={"v": 1})
        boom = _Outcome(step_id="b", name="B", kind="raise")
        step = ParallelStep(step_id="p", name="P", steps=[ok, boom], next_step_id="n")
        result = await step.execute({})
        assert result.success is False
        assert result.data == {"step_0_v": 1}
        assert "boom" in str(result.error)
        assert ok.calls == boom.calls == 1

    async def test_a_child_that_returns_no_result_object_fails_the_step(self):
        step = ParallelStep(
            step_id="p",
            name="P",
            steps=[_Outcome(step_id="g", name="G", kind="garbage")],
            next_step_id="n",
            error_step_id="err",
        )
        result = await step.execute({})
        assert result.success is False
        assert result.next_step_id == "err"
        assert "returned str, not a WorkflowStepResult" in str(result.error)

    async def test_the_context_copy_warning_starts_above_the_threshold(self):
        from loguru import logger

        from fsm_llm.workflows.constants import PARALLEL_DEEPCOPY_WARNING_THRESHOLD

        warnings: list[str] = []
        logger.enable("fsm_llm")
        sink = logger.add(
            lambda m: (
                warnings.append(m.record["message"])
                if m.record["name"] == "fsm_llm.workflows.steps.composite"
                else None
            ),
            level="WARNING",
            format="{message}",
        )
        try:
            for count in (
                PARALLEL_DEEPCOPY_WARNING_THRESHOLD,
                PARALLEL_DEEPCOPY_WARNING_THRESHOLD + 1,
            ):
                warnings.clear()
                children = [_Outcome(step_id=f"c{i}", name="C") for i in range(count)]
                await ParallelStep(
                    step_id="p", name="P", steps=children, next_step_id="n"
                ).execute({})
                assert bool(warnings) is (count > PARALLEL_DEEPCOPY_WARNING_THRESHOLD)
        finally:
            logger.remove(sink)
            logger.disable("fsm_llm")

    def test_a_retry_step_takes_no_negative_count_or_backoff(self):
        inner = _Outcome(step_id="o", name="O")
        with pytest.raises(ValidationError, match="max_retries must be >= 0"):
            RetryStep(step_id="r", name="R", step=inner, max_retries=-1)
        with pytest.raises(ValidationError, match="backoff_factor must be >= 0"):
            RetryStep(step_id="r", name="R", step=inner, backoff_factor=-0.5)
        step = RetryStep(
            step_id="r", name="R", step=inner, max_retries=0, backoff_factor=0
        )
        assert (step.max_retries, step.backoff_factor) == (0, 0)

    async def test_a_retry_waits_the_backoff_between_failed_results(self):
        delays: list[float] = []

        async def fake_sleep(delay: float) -> None:
            delays.append(delay)

        inner = _Outcome(step_id="o", name="O", kind="fail")
        step = RetryStep(
            step_id="r", name="R", step=inner, max_retries=2, backoff_factor=0.5
        )
        with patch("fsm_llm.workflows.steps.composite.asyncio.sleep", fake_sleep):
            result = await step.execute({})
        assert result.success is False and result.error == "bad"
        assert inner.calls == 3
        assert delays == [0.5, 1.0]

    async def test_a_retry_waits_the_backoff_between_raised_errors_too(self):
        delays: list[float] = []

        async def fake_sleep(delay: float) -> None:
            delays.append(delay)

        inner = _Outcome(step_id="o", name="O", kind="raise")
        step = RetryStep(
            step_id="r", name="R", step=inner, max_retries=2, backoff_factor=2.0
        )
        with patch("fsm_llm.workflows.steps.composite.asyncio.sleep", fake_sleep):
            with pytest.raises(ValueError, match="boom"):
                await step.execute({})
        assert inner.calls == 3
        assert delays == [2.0, 4.0]
