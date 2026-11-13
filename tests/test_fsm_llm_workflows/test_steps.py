"""
Unit tests for workflow step types.
Tests step creation, async execution, and error handling.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

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

        with patch("fsm_llm.workflows.steps.API") as mock_api:
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
        with patch("fsm_llm.workflows.steps.API") as mock_cls:
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
        with patch("fsm_llm.workflows.steps.API") as mock_cls:
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
        with patch("fsm_llm.workflows.steps.API") as mock_cls:
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
            with patch("fsm_llm.workflows.steps.API", mock_api_class):
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
