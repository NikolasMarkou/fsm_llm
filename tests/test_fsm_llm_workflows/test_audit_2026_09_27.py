"""
Regression tests for the 2026-09-27 fsm_llm_workflows audit.

Each class pins one finding (H = high, M = medium, L = low). Every test here
failed on the pre-fix code.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import textwrap
from typing import Any

import pytest

from fsm_llm_workflows import (
    DependencyResolver,
    WorkflowEngine,
    WorkflowHistoryEntry,
    auto_step,
    condition_step,
    create_workflow,
    switch_step,
    timer_step,
    wait_event_step,
)
from fsm_llm_workflows.definitions import WorkflowDefinition
from fsm_llm_workflows.exceptions import (
    WorkflowDefinitionError,
    WorkflowEventError,
    WorkflowInstanceError,
    WorkflowResourceError,
    WorkflowStepError,
    WorkflowTimeoutError,
    WorkflowValidationError,
)
from fsm_llm_workflows.models import (
    WaitEventConfig,
    WorkflowEvent,
    WorkflowStatus,
    WorkflowStepResult,
)
from fsm_llm_workflows.steps import (
    AgentStep,
    APICallStep,
    AutoTransitionStep,
    ConditionStep,
    ConversationStep,
    LLMProcessingStep,
    ParallelStep,
    RetryStep,
    TimerStep,
    WorkflowStep,
)

# ---------------------------------------------------------------
# helpers
# ---------------------------------------------------------------


def _wait_then_done(
    workflow_id: str,
    event_type: str = "go",
    **wait_kwargs: Any,
) -> WorkflowDefinition:
    """wait(event) -> done (terminal)."""
    wf = create_workflow(workflow_id, workflow_id)
    wf.with_initial_step(
        wait_event_step("wait", "Wait", event_type, success_state="done", **wait_kwargs)
    )
    wf.with_step(auto_step("done", "Done", next_state=""))
    return wf


async def _settle(seconds: float = 0.05) -> None:
    await asyncio.sleep(seconds)


class _DynamicLoopStep(WorkflowStep):
    """A custom step that always routes back to itself at runtime (the
    static graph cannot see it)."""

    async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
        return WorkflowStepResult.success_result(next_state=self.step_id)


class _TerminalStep(WorkflowStep):
    async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
        return WorkflowStepResult.success_result()


class _FakeLLMResponse:
    def __init__(self, message: str) -> None:
        self.message = message


class _CoreLikeLLM:
    """Shaped like fsm_llm.LLMInterface: sync generate_response only."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.requests: list[Any] = []

    def generate_response(self, request: Any) -> _FakeLLMResponse:
        self.requests.append(request)
        return _FakeLLMResponse(self.reply)


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


# ---------------------------------------------------------------
# H1 / M11: one instance's failure must not break event delivery
# ---------------------------------------------------------------


class TestProcessEventIsolation:
    async def test_expired_deadline_on_one_instance_does_not_strand_others(self):
        engine = WorkflowEngine()
        engine.register_workflow(_wait_then_done("wf"))
        # A: tiny workflow timeout; B: none. A is registered first.
        a = await engine.start_workflow("wf", workflow_timeout=5)
        b = await engine.start_workflow("wf")
        engine.workflow_instances[a].deadline = engine.workflow_instances[
            a
        ].created_at.replace(year=2000)

        affected = await engine.process_event(WorkflowEvent(event_type="go"))

        assert b in affected
        assert engine.get_workflow_status(b) == WorkflowStatus.COMPLETED
        assert engine.get_workflow_status(a) == WorkflowStatus.FAILED

    async def test_non_waiting_instance_is_skipped_and_not_mutated(self):
        engine = WorkflowEngine()
        engine.register_workflow(_wait_then_done("wf", event_mapping={"got": "value"}))
        a = await engine.start_workflow("wf")
        b = await engine.start_workflow("wf")
        # A leaves WAITING without its listener being cleaned up (the race
        # between a cancel and a concurrent process_event).
        engine.workflow_instances[a].status = WorkflowStatus.CANCELLED

        affected = await engine.process_event(
            WorkflowEvent(event_type="go", payload={"value": 1})
        )

        assert affected == [b]
        assert "got" not in engine.workflow_instances[a].context
        assert engine.get_workflow_status(a) == WorkflowStatus.CANCELLED
        assert engine.get_workflow_status(b) == WorkflowStatus.COMPLETED


# ---------------------------------------------------------------
# H2: the follow-up wait's timeout must survive event delivery
# ---------------------------------------------------------------


class TestRewaitTimeoutSurvives:
    async def test_second_wait_on_same_event_still_times_out(self):
        wf = create_workflow("rewait", "Rewait")
        wf.with_initial_step(wait_event_step("w1", "W1", "go", success_state="w2"))
        wf.with_step(
            wait_event_step(
                "w2",
                "W2",
                "go",
                success_state="done",
                timeout_seconds=0.05,
                timeout_state="timed_out",
            )
        )
        wf.with_step(auto_step("done", "Done", next_state=""))
        wf.with_step(auto_step("timed_out", "Timed out", next_state=""))
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start_workflow("rewait")

        await engine.process_event(WorkflowEvent(event_type="go"))
        assert engine.workflow_instances[iid].current_step_id == "w2"
        await _settle(0.2)

        inst = engine.workflow_instances[iid]
        assert inst.current_step_id == "timed_out"
        assert inst.status == WorkflowStatus.COMPLETED


# ---------------------------------------------------------------
# H3: a wait never stays WAITING forever
# ---------------------------------------------------------------


class TestWaitTimeoutsAndTerminalWaits:
    async def test_timeout_without_timeout_state_fails_instance(self):
        engine = WorkflowEngine()
        engine.register_workflow(_wait_then_done("wf", timeout_seconds=0.05))
        iid = await engine.start_workflow("wf")
        await _settle(0.2)

        inst = engine.workflow_instances[iid]
        assert inst.status == WorkflowStatus.FAILED
        assert "timed out" in (inst.error or "")
        assert engine.timers == {}

    async def test_empty_success_state_completes_on_event(self):
        wf = create_workflow("wf", "wf")
        wf.with_initial_step(wait_event_step("wait", "Wait", "go", success_state=""))
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start_workflow("wf")

        assert await engine.process_event(WorkflowEvent(event_type="go")) == [iid]
        assert engine.get_workflow_status(iid) == WorkflowStatus.COMPLETED

    def test_timeout_state_without_timeout_seconds_rejected(self):
        with pytest.raises(ValueError, match="timeout_state requires"):
            WaitEventConfig(event_type="go", success_state="x", timeout_state="t")

    def test_empty_event_type_rejected(self):
        with pytest.raises(ValueError):
            WaitEventConfig(event_type="", success_state="x")


# ---------------------------------------------------------------
# H4: LLMProcessingStep works with a core LLMInterface
# ---------------------------------------------------------------


class TestLLMProcessingStepInterfaces:
    async def test_core_llm_interface_generate_response(self):
        llm = _CoreLikeLLM("ANSWER: 42")
        step = LLMProcessingStep(
            step_id="llm",
            name="LLM",
            llm_interface=llm,
            prompt_template="Q: {q}",
            context_mapping={"q": "question"},
            output_mapping={"answer": r"ANSWER: (\d+)", "raw": ""},
            next_state="done",
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

        step = LLMProcessingStep(
            step_id="llm",
            name="LLM",
            llm_interface=_SyncLLM(),
            prompt_template="hi",
            output_mapping={"out": ""},
            next_state="done",
        )
        result = await step.execute({})
        assert result.success is True
        assert result.data["out"] == "echo hi"

    async def test_regex_miss_leaves_key_unset(self):
        step = LLMProcessingStep(
            step_id="llm",
            name="LLM",
            llm_interface=_CoreLikeLLM("no number here"),
            prompt_template="hi",
            output_mapping={"n": r"(\d+)"},
            next_state="done",
        )
        result = await step.execute({})
        assert result.success is True
        assert "n" not in result.data

    def test_invalid_regex_rejected_at_construction(self):
        with pytest.raises(ValueError, match="valid regex"):
            LLMProcessingStep(
                step_id="llm",
                name="LLM",
                llm_interface=_CoreLikeLLM(""),
                prompt_template="hi",
                output_mapping={"n": "(unclosed"},
                next_state="done",
            )


# ---------------------------------------------------------------
# H5: failures never continue down the success route
# ---------------------------------------------------------------


class TestFailClosedRouting:
    async def test_llm_failure_without_error_state_fails_instance(self):
        class _Broken:
            def generate(self, prompt):
                raise RuntimeError("provider down")

        wf = create_workflow("wf", "wf")
        wf.with_initial_step(
            LLMProcessingStep(
                step_id="llm",
                name="LLM",
                llm_interface=_Broken(),
                prompt_template="hi",
                next_state="after",
            )
        )
        wf.with_step(auto_step("after", "After", next_state=""))
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start_workflow("wf")

        inst = engine.workflow_instances[iid]
        assert inst.status == WorkflowStatus.FAILED
        assert inst.current_step_id == "llm"
        assert "provider down" in (inst.error or "")

    async def test_parallel_failure_without_error_state_has_no_route(self):
        bad = AutoTransitionStep(
            step_id="bad",
            name="Bad",
            next_state="x",
            action=lambda ctx: 1 / 0,
        )
        step = ParallelStep(step_id="p", name="P", steps=[bad], next_state="next")
        result = await step.execute({})
        assert result.success is False
        assert result.next_state is None

    async def test_agent_reporting_failure_fails_step(self):
        agent = _RecordingAgent(_AgentResult("gave up", success=False))
        step = AgentStep(
            step_id="a",
            name="A",
            agent=agent,
            success_state="next",
            error_state="err",
        )
        result = await step.execute({"task": "t"})
        assert result.success is False
        assert result.next_state == "err"
        assert result.data["agent_success"] is False

    async def test_conversation_failure_without_error_state_has_no_route(self):
        step = ConversationStep(
            step_id="c",
            name="C",
            fsm_definition={"name": "x"},
            success_state="next",
        )
        from unittest.mock import patch

        with patch("fsm_llm.API") as mock_api:
            mock_api.from_definition.side_effect = RuntimeError("no model")
            result = await step.execute({})
        assert result.success is False
        assert result.next_state is None


# ---------------------------------------------------------------
# H6: a failed parallel child always counts
# ---------------------------------------------------------------


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
                step_id="p", name="P", steps=[child], next_state="n", error_state="e"
            )
            result = await step.execute({})
            assert result.success is False, type(child).__name__
            assert result.next_state == "e"
            assert result.error


# ---------------------------------------------------------------
# H7: callables that return awaitables are awaited
# ---------------------------------------------------------------


class TestAwaitableReturningCallables:
    async def test_condition_lambda_returning_coroutine(self):
        async def is_big(ctx):
            return ctx["n"] > 10

        step = ConditionStep(
            step_id="c",
            name="C",
            condition=lambda ctx: is_big(ctx),
            true_state="big",
            false_state="small",
        )
        assert (await step.execute({"n": 1})).next_state == "small"
        assert (await step.execute({"n": 99})).next_state == "big"

    async def test_api_lambda_returning_coroutine(self):
        async def fetch(**kwargs):
            return {"value": 7}

        step = APICallStep(
            step_id="api",
            name="API",
            api_function=lambda **kw: fetch(**kw),
            success_state="ok",
            failure_state="bad",
            output_mapping={"v": "value"},
        )
        result = await step.execute({})
        assert result.success is True
        assert result.data == {"v": 7}

    async def test_async_callable_object(self):
        class _AsyncAction:
            async def __call__(self, ctx):
                return {"done": True}

        step = AutoTransitionStep(
            step_id="a", name="A", next_state="n", action=_AsyncAction()
        )
        assert (await step.execute({})).data == {"done": True}


# ---------------------------------------------------------------
# H8 / H9: instance ids
# ---------------------------------------------------------------


class TestInstanceIds:
    async def test_cancel_does_not_touch_timers_of_prefix_sharing_ids(self):
        wf = create_workflow("t", "t")
        wf.with_initial_step(timer_step("wait", "Wait", 60, "done"))
        wf.with_step(auto_step("done", "Done", next_state=""))
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        await engine.start_workflow("t", instance_id="order")
        await engine.start_workflow("t", instance_id="order_2")

        await engine.cancel_workflow("order")

        assert "order_2_timer" in engine.timers
        assert not engine.timers["order_2_timer"].task.cancelled()
        await engine.shutdown()

    async def test_duplicate_instance_id_rejected(self):
        engine = WorkflowEngine()
        engine.register_workflow(_wait_then_done("wf"))
        await engine.start_workflow("wf", instance_id="same")
        with pytest.raises(WorkflowInstanceError, match="already in use"):
            await engine.start_workflow("wf", instance_id="same")
        assert engine.get_workflow_status("same") == WorkflowStatus.WAITING


# ---------------------------------------------------------------
# M1: deadlines
# ---------------------------------------------------------------


class TestWorkflowDeadline:
    async def test_deadline_fails_waiting_instance(self):
        engine = WorkflowEngine()
        engine.register_workflow(_wait_then_done("wf"))
        iid = await engine.start_workflow("wf", workflow_timeout=0.05)
        assert engine.get_workflow_status(iid) == WorkflowStatus.WAITING
        await _settle(0.2)

        assert engine.get_workflow_status(iid) == WorkflowStatus.FAILED
        assert engine.event_listeners["go"] == {}

    async def test_timeout_error_names_instance(self):
        class _Slow(WorkflowStep):
            async def execute(self, context):
                await asyncio.sleep(1)
                return WorkflowStepResult.success_result(next_state="")

        wf = create_workflow("slow", "slow")
        wf.with_initial_step(_Slow(step_id="s", name="S"))
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        with pytest.raises(WorkflowTimeoutError) as exc_info:
            await engine.start_workflow("slow", workflow_timeout=0.05)

        iid = exc_info.value.instance_id
        assert iid is not None
        assert engine.get_workflow_status(iid) == WorkflowStatus.FAILED


# ---------------------------------------------------------------
# M2: long chains, loops through waits, step budget
# ---------------------------------------------------------------


class TestLoopsAndStepBudget:
    async def test_long_acyclic_chain_completes(self):
        wf = create_workflow("long", "long")
        n = 30
        for i in range(n):
            nxt = f"s{i + 1}" if i < n - 1 else ""
            wf.with_step(auto_step(f"s{i}", f"S{i}", next_state=nxt), is_initial=i == 0)
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start_workflow("long")
        assert engine.get_workflow_status(iid) == WorkflowStatus.COMPLETED

    async def test_polling_loop_through_timer_is_allowed(self):
        polls = {"n": 0}

        def poll(ctx):
            polls["n"] += 1
            return {"ready": polls["n"] >= 3}

        wf = create_workflow("poll", "poll")
        wf.with_initial_step(
            auto_step("check", "Check", next_state="route", action=poll)
        )
        wf.with_step(
            condition_step(
                "route",
                "Ready?",
                condition=lambda ctx: ctx["ready"],
                true_state="done",
                false_state="sleep",
            )
        )
        wf.with_step(timer_step("sleep", "Sleep", 0, "check"))
        wf.with_step(auto_step("done", "Done", next_state=""))
        assert wf.has_cycles() is True
        assert wf.has_synchronous_cycles() is False

        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start_workflow("poll")
        await _settle(0.2)

        assert engine.get_workflow_status(iid) == WorkflowStatus.COMPLETED
        assert polls["n"] == 3

    def test_synchronous_cycle_rejected(self):
        wf = create_workflow("cyc", "cyc")
        wf.with_initial_step(auto_step("a", "A", next_state="b"))
        wf.with_step(auto_step("b", "B", next_state="a"))
        with pytest.raises(WorkflowValidationError, match="synchronous cycle"):
            wf.validate()

    async def test_dynamic_loop_hits_step_budget(self):
        wf = create_workflow("dyn", "dyn")
        wf.with_initial_step(_DynamicLoopStep(step_id="loop", name="Loop"))
        engine = WorkflowEngine(max_steps_per_run=5)
        engine.register_workflow(wf)
        iid = await engine.start_workflow("dyn")

        inst = engine.workflow_instances[iid]
        assert inst.status == WorkflowStatus.FAILED
        assert "Step budget exceeded" in (inst.error or "")


# ---------------------------------------------------------------
# M3: correlation, targeting, buffering
# ---------------------------------------------------------------


class TestEventCorrelation:
    async def test_correlation_key_routes_to_matching_instance(self):
        engine = WorkflowEngine()
        engine.register_workflow(_wait_then_done("wf", correlation_key="order_id"))
        one = await engine.start_workflow("wf", {"order_id": 1})
        two = await engine.start_workflow("wf", {"order_id": 2})

        affected = await engine.process_event(
            WorkflowEvent(event_type="go", payload={"order_id": 2})
        )

        assert affected == [two]
        assert engine.get_workflow_status(one) == WorkflowStatus.WAITING
        assert engine.get_workflow_status(two) == WorkflowStatus.COMPLETED

    async def test_targeted_event_only_reaches_target(self):
        engine = WorkflowEngine()
        engine.register_workflow(_wait_then_done("wf"))
        one = await engine.start_workflow("wf")
        two = await engine.start_workflow("wf")

        affected = await engine.process_event(
            WorkflowEvent(event_type="go", instance_id=one)
        )

        assert affected == [one]
        assert engine.get_workflow_status(two) == WorkflowStatus.WAITING

    async def test_early_targeted_event_is_buffered(self):
        wf = create_workflow("wf", "wf")
        wf.with_initial_step(timer_step("pause", "Pause", 60, "wait"))
        wf.with_step(
            wait_event_step(
                "wait", "Wait", "paid", success_state="done", event_mapping={"amt": "a"}
            )
        )
        wf.with_step(auto_step("done", "Done", next_state=""))
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start_workflow("wf")

        # Arrives while the instance is still in the timer, not the wait.
        assert (
            await engine.process_event(
                WorkflowEvent(event_type="paid", payload={"a": 5}, instance_id=iid)
            )
            == []
        )
        await engine._handle_timer_expiration(iid, "wait")

        inst = engine.workflow_instances[iid]
        assert inst.status == WorkflowStatus.COMPLETED
        assert inst.context["amt"] == 5


# ---------------------------------------------------------------
# M4: SwitchStep defaults and outputs
# ---------------------------------------------------------------


class TestSwitchDefaults:
    async def test_dsl_default_fails_on_unmatched_value(self):
        wf = create_workflow("sw", "sw")
        wf.with_initial_step(switch_step("route", "Route", "k", cases={"a": "done"}))
        wf.with_step(auto_step("done", "Done", next_state=""))
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start_workflow("sw", {"k": "zzz"})
        assert engine.get_workflow_status(iid) == WorkflowStatus.FAILED

    async def test_switch_outputs_reach_context(self):
        wf = create_workflow("sw", "sw")
        wf.with_initial_step(switch_step("route", "Route", "k", cases={"a": "done"}))
        wf.with_step(auto_step("done", "Done", next_state=""))
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start_workflow("sw", {"k": "a"})
        ctx = engine.get_workflow_context(iid)
        assert ctx["switch_route_matched"] == "a"
        assert ctx["switch_route_target"] == "done"


# ---------------------------------------------------------------
# M5: ParallelStep children
# ---------------------------------------------------------------


class TestParallelChildren:
    async def test_internal_keys_do_not_leak_through_prefix(self):
        child = AutoTransitionStep(
            step_id="c",
            name="C",
            next_state="x",
            action=lambda ctx: {"_secret": 1, "system_x": 2, "ok": 3},
        )
        step = ParallelStep(step_id="p", name="P", steps=[child], next_state="n")
        result = await step.execute({})
        assert result.data == {"step_0_ok": 3}

    def test_timer_child_rejected(self):
        with pytest.raises(ValueError, match="cannot run inside a ParallelStep"):
            ParallelStep(
                step_id="p",
                name="P",
                steps=[
                    TimerStep(step_id="t", name="T", delay_seconds=1, next_state="x")
                ],
                next_state="n",
            )

    async def test_partial_data_kept_on_failure(self):
        good = AutoTransitionStep(
            step_id="g", name="G", next_state="x", action=lambda ctx: {"v": 1}
        )
        bad = AutoTransitionStep(
            step_id="b", name="B", next_state="x", action=lambda ctx: 1 / 0
        )
        step = ParallelStep(
            step_id="p", name="P", steps=[good, bad], next_state="n", error_state="e"
        )
        result = await step.execute({})
        assert result.success is False
        assert result.data == {"step_0_v": 1}

    async def test_uncopyable_context_falls_back_to_shallow_copy(self):
        import threading

        child = AutoTransitionStep(step_id="c", name="C", next_state="x")
        step = ParallelStep(step_id="p", name="P", steps=[child], next_state="n")
        result = await step.execute({"lock": threading.Lock()})
        assert result.success is True


# ---------------------------------------------------------------
# M6: AgentStep mapping
# ---------------------------------------------------------------


class TestAgentStepMapping:
    async def test_answer_mapped_with_empty_final_context(self):
        agent = _RecordingAgent(_AgentResult("42"))
        step = AgentStep(
            step_id="a",
            name="A",
            agent=agent,
            success_state="next",
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
            success_state="next",
            input_mapping={"customer": "cust_id"},
        )
        await step.execute({"task": "t", "cust_id": "C-1"})
        assert agent.calls == [("t", {"initial_context": {"customer": "C-1"}})]

    async def test_plain_string_result(self):
        class _StrAgent:
            def run(self, task):
                return f"done: {task}"

        step = AgentStep(step_id="a", name="A", agent=_StrAgent(), success_state="n")
        result = await step.execute({"task": "t"})
        assert result.success is True
        assert result.data["agent_answer"] == "done: t"

    def test_agent_without_run_rejected(self):
        with pytest.raises(ValueError, match="run"):
            AgentStep(step_id="a", name="A", agent=object())


# ---------------------------------------------------------------
# M7: ConversationStep
# ---------------------------------------------------------------


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

    async def test_base_timeout_is_honoured(self):
        from unittest.mock import patch

        step = ConversationStep(
            step_id="c",
            name="C",
            fsm_definition={"name": "x"},
            timeout=0.05,
            error_state="err",
        )
        with patch("fsm_llm.API") as mock_cls:
            mock_cls.from_definition.return_value = self._mock_api(delay=0.3)
            result = await step.execute({})
        assert result.success is False
        assert result.next_state == "err"
        assert "timed out" in result.error

    async def test_ended_flag_and_require_completion(self):
        from unittest.mock import patch

        step = ConversationStep(
            step_id="c",
            name="C",
            fsm_definition={"name": "x"},
            success_state="next",
            error_state="err",
            require_completion=True,
        )
        with patch("fsm_llm.API") as mock_cls:
            mock_cls.from_definition.return_value = self._mock_api(ended=False)
            result = await step.execute({})
        assert result.success is False
        assert result.next_state == "err"
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
        with patch("fsm_llm.API") as mock_cls:
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


# ---------------------------------------------------------------
# M9: resources are released
# ---------------------------------------------------------------


class TestResourceRelease:
    async def test_fired_event_timeout_leaves_no_timer(self):
        wf = create_workflow("wf", "wf")
        wf.with_initial_step(
            wait_event_step(
                "wait",
                "Wait",
                "go",
                success_state="done",
                timeout_seconds=0.05,
                timeout_state="late",
            )
        )
        wf.with_step(auto_step("done", "Done", next_state=""))
        wf.with_step(auto_step("late", "Late", next_state=""))
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start_workflow("wf")
        await _settle(0.2)

        assert engine.get_workflow_status(iid) == WorkflowStatus.COMPLETED
        assert engine.timers == {}
        assert engine.get_statistics()["active_timers"] == 0

    def test_default_retention_is_bounded(self):
        assert WorkflowEngine().max_completed_instances is not None

    async def test_failed_instances_are_purged_too(self):
        wf = create_workflow("f", "f")
        wf.with_initial_step(
            auto_step("boom", "Boom", next_state="", action=lambda ctx: 1 / 0)
        )
        engine = WorkflowEngine(max_completed_instances=2)
        engine.register_workflow(wf)
        ids = [await engine.start_workflow("f") for _ in range(4)]
        assert all(
            engine.get_workflow_status(i) in (WorkflowStatus.FAILED, None) for i in ids
        )
        assert len(engine.workflow_instances) == 2

    def test_history_is_capped(self):
        from fsm_llm_workflows.models import WorkflowInstance

        inst = WorkflowInstance(
            instance_id="i", workflow_id="w", current_step_id="s", max_history_entries=3
        )
        for i in range(10):
            inst.add_history_entry(step_id="s", message=str(i))
        assert [e.message for e in inst.history] == ["7", "8", "9"]


# ---------------------------------------------------------------
# M10: definitions are pinned
# ---------------------------------------------------------------


class TestDefinitionPinning:
    async def test_reregistration_does_not_affect_running_instance(self):
        engine = WorkflowEngine()
        engine.register_workflow(_wait_then_done("wf"))
        iid = await engine.start_workflow("wf")

        replacement = create_workflow("wf", "v2")
        replacement.with_initial_step(auto_step("other", "Other", next_state=""))
        engine.register_workflow(replacement)
        await engine.process_event(WorkflowEvent(event_type="go"))

        inst = engine.workflow_instances[iid]
        assert inst.status == WorkflowStatus.COMPLETED
        assert inst.current_step_id == "done"

    def test_mutating_after_registration_does_not_change_registered(self):
        engine = WorkflowEngine()
        wf = _wait_then_done("wf")
        engine.register_workflow(wf)
        wf.with_step(auto_step("extra", "Extra", next_state=""))
        assert "extra" not in engine.get_workflow_definition("wf").steps


# ---------------------------------------------------------------
# L1 / L2 / L17: hooks, write-only keys, background start, context copy
# ---------------------------------------------------------------


class TestEngineApi:
    async def test_hooks_see_steps_and_status_changes(self):
        seen: list[tuple[str, dict]] = []
        engine = WorkflowEngine()
        engine.add_hook(lambda name, inst, data: seen.append((name, data)))
        engine.add_hook(lambda *a: 1 / 0)  # a broken hook must not matter
        wf = create_workflow("h", "h")
        wf.with_initial_step(auto_step("a", "A", next_state=""))
        engine.register_workflow(wf)
        iid = await engine.start_workflow("h")

        assert engine.get_workflow_status(iid) == WorkflowStatus.COMPLETED
        names = [n for n, _ in seen]
        assert names == ["step_started", "step_completed", "status_changed"]
        assert seen[-1][1]["status"] == "completed"

    async def test_user_input_removed_after_advance(self):
        engine = WorkflowEngine()
        engine.register_workflow(_wait_then_done("wf"))
        iid = await engine.start_workflow("wf")
        assert await engine.advance_workflow(iid, "hi") is True
        assert "_user_input" not in engine.workflow_instances[iid].context
        assert (
            "auto_transition_states"
            not in engine.workflow_instances[iid].context["_workflow_info"]
        )

    async def test_background_start(self):
        engine = WorkflowEngine()
        wf = create_workflow("bg", "bg")
        wf.with_initial_step(auto_step("a", "A", next_state=""))
        engine.register_workflow(wf)
        iid = await engine.start_workflow("bg", wait=False)
        await _settle()
        assert engine.get_workflow_status(iid) == WorkflowStatus.COMPLETED

    async def test_context_getter_returns_copy_and_initial_context_is_copied(self):
        engine = WorkflowEngine()
        engine.register_workflow(_wait_then_done("wf"))
        initial = {"a": 1}
        iid = await engine.start_workflow("wf", initial)
        engine.get_workflow_context(iid)["a"] = 999
        assert engine.workflow_instances[iid].context["a"] == 1
        assert "_workflow_info" not in initial

    async def test_shutdown_refuses_new_starts_and_clears_listeners(self):
        engine = WorkflowEngine()
        engine.register_workflow(_wait_then_done("wf", timeout_seconds=60))
        await engine.start_workflow("wf")
        await engine.shutdown()
        assert engine.timers == {}
        assert engine.event_listeners == {}
        with pytest.raises(WorkflowResourceError):
            await engine.start_workflow("wf")


# ---------------------------------------------------------------
# L3: logging is off by default
# ---------------------------------------------------------------


class TestLoggingOffByDefault:
    def test_workflow_logs_follow_library_logging_switch(self):
        script = textwrap.dedent(
            """
            import asyncio
            from loguru import logger
            import fsm_llm_workflows as wfl
            from fsm_llm.logging import enable_library_logging

            lines = []
            logger.add(lambda m: lines.append(str(m)), level="DEBUG")

            async def run():
                engine = wfl.WorkflowEngine()
                wf = wfl.create_workflow("w", "w")
                wf.with_initial_step(wfl.auto_step("a", "A", next_state=""))
                engine.register_workflow(wf)
                await engine.start_workflow("w")

            asyncio.run(run())
            before = sum("Executing step" in line for line in lines)
            enable_library_logging()
            asyncio.run(run())
            after = sum("Executing step" in line for line in lines)
            print(before, after)
            """
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=120,
            env=dict(os.environ),
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert proc.stdout.split() == ["0", "1"]


# ---------------------------------------------------------------
# L5 / L7 / L9 / L10 / L12: smaller fixes
# ---------------------------------------------------------------


class TestSmallerFixes:
    async def test_auto_and_condition_error_state(self):
        auto = AutoTransitionStep(
            step_id="a",
            name="A",
            next_state="n",
            error_state="e",
            action=lambda c: 1 / 0,
        )
        cond = ConditionStep(
            step_id="c",
            name="C",
            condition=lambda c: 1 / 0,
            true_state="t",
            false_state="f",
            error_state="e",
        )
        for step in (auto, cond):
            result = await step.execute({})
            assert result.success is False
            assert result.next_state == "e"

    def test_non_dict_action_result_is_a_clear_error(self):
        step = AutoTransitionStep(
            step_id="a", name="A", next_state="n", action=lambda c: ["x"]
        )
        with pytest.raises(WorkflowStepError) as exc_info:
            asyncio.run(step.execute({}))
        assert "must return a dict" in str(exc_info.value.cause)

    async def test_history_records_step_error_without_redundant_entries(self):
        wf = create_workflow("h", "h")
        wf.with_initial_step(auto_step("a", "A", next_state="b"))
        wf.with_step(switch_step("b", "B", "k", cases={"x": "done"}))
        wf.with_step(auto_step("done", "Done", next_state=""))
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start_workflow("h")

        history = engine.workflow_instances[iid].history
        assert isinstance(history[0], WorkflowHistoryEntry)
        failed = [e for e in history if e.step_id == "b" and e.error]
        assert failed and "No matching case" in failed[0].error
        running = [e for e in history if e.message == "Status changed to running"]
        assert len(running) == 1

    def test_resolver_unknown_dependency_is_not_a_cycle(self):
        resolver = DependencyResolver().add_step("a", depends_on=["ghost"])
        assert resolver.has_cycles() is False
        with pytest.raises(WorkflowValidationError, match="unknown"):
            resolver.resolve()
        cyclic = DependencyResolver.from_dict({"a": ["b"], "b": ["a"]})
        assert cyclic.has_cycles() is True

    def test_exception_details_are_not_mutated(self):
        details = {"k": 1}
        WorkflowStepError("s", "m", cause=ValueError("x"), details=details)
        WorkflowTimeoutError("op", 1.0, details=details)
        assert details == {"k": 1}

    async def test_empty_event_type_listener_rejected(self):
        engine = WorkflowEngine()
        engine.register_workflow(_wait_then_done("wf"))
        iid = await engine.start_workflow("wf")
        with pytest.raises(WorkflowEventError):
            await engine.register_event_listener(iid, "")

    def test_duplicate_step_id_rejected(self):
        wf = create_workflow("d", "d")
        wf.with_step(auto_step("a", "A", next_state=""))
        with pytest.raises(WorkflowDefinitionError, match="Duplicate"):
            wf.with_step(auto_step("a", "Other", next_state=""))

    async def test_api_output_paths(self):
        step = APICallStep(
            step_id="api",
            name="API",
            api_function=lambda: {"user": {"name": "Ada"}, "items": [1, 2]},
            success_state="ok",
            failure_state="bad",
            output_mapping={"name": "user.name", "all": ""},
        )
        result = await step.execute({})
        assert result.data["name"] == "Ada"
        assert result.data["all"]["items"] == [1, 2]

        list_step = APICallStep(
            step_id="api2",
            name="API2",
            api_function=lambda: [1, 2, 3],
            success_state="ok",
            failure_state="bad",
            output_mapping={"rows": ""},
        )
        assert (await list_step.execute({})).data == {"rows": [1, 2, 3]}

    def test_float_delays_accepted(self):
        assert timer_step("t", "T", 0.25, "x").delay_seconds == 0.25
        cfg = WaitEventConfig(event_type="e", success_state="x", timeout_seconds=0.5)
        assert cfg.timeout_seconds == 0.5

    def test_serialize_strips_custom_callables(self):
        class _WithCallback(WorkflowStep):
            callback: Any = None

            async def execute(self, context):
                return WorkflowStepResult.success_result(next_state="")

        wf = create_workflow("s", "s")
        wf.with_initial_step(_WithCallback(step_id="a", name="A", callback=print))
        assert "callback" not in wf.serialize()["steps"]["a"]

    async def test_retry_wrapping_timer_counts_as_pausing(self):
        wf = create_workflow("r", "r")
        inner = TimerStep(step_id="t_inner", name="T", delay_seconds=0, next_state="a")
        wf.with_initial_step(auto_step("a", "A", next_state="t"))
        wf.with_step(RetryStep(step_id="t", name="T", step=inner))
        assert wf.has_synchronous_cycles() is False


# ---------------------------------------------------------------
# M8: injectable executor for synchronous callables
# ---------------------------------------------------------------


class TestStepExecutor:
    async def test_sync_callables_run_on_engine_executor(self):
        import threading
        from concurrent.futures import ThreadPoolExecutor

        wf = create_workflow("ex", "ex")
        wf.with_initial_step(
            auto_step(
                "a",
                "A",
                next_state="",
                action=lambda ctx: {"thread": threading.current_thread().name},
            )
        )
        with ThreadPoolExecutor(thread_name_prefix="wf-exec") as pool:
            engine = WorkflowEngine(executor=pool)
            engine.register_workflow(wf)
            iid = await engine.start_workflow("ex")
        assert engine.get_workflow_context(iid)["thread"].startswith("wf-exec")


# ---------------------------------------------------------------
# Follow-up review: delivery to the right wait, cancellation safety,
# background start guard, long prompts
# ---------------------------------------------------------------


class _SlowStep(WorkflowStep):
    next_state: str = ""
    delay: float = 0.3

    async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
        await asyncio.sleep(self.delay)
        return WorkflowStepResult.success_result(next_state=self.next_state)


def _two_waits_workflow() -> WorkflowDefinition:
    """wa (A, times out to wb) -> slow -> "" ; wb waits for B."""
    wf = create_workflow("tw", "tw")
    wf.with_initial_step(
        wait_event_step(
            "wa",
            "WA",
            "A",
            success_state="slow",
            timeout_seconds=0.1,
            timeout_state="wb",
        )
    )
    wf.with_step(_SlowStep(step_id="slow", name="Slow"))
    wf.with_step(wait_event_step("wb", "WB", "B", success_state="got_b"))
    wf.with_step(auto_step("got_b", "Got B", next_state=""))
    return wf


class TestReviewFollowUp:
    async def test_event_not_delivered_to_a_different_wait(self):
        engine = WorkflowEngine()
        engine.register_workflow(_two_waits_workflow())
        first = await engine.start_workflow("tw")
        second = await engine.start_workflow("tw")

        # Delivery to `first` runs the slow step (0.3 s) while `second`
        # times out into `wb`; the A event must not then move `second`.
        affected = await engine.process_event(WorkflowEvent(event_type="A"))

        assert first in affected
        assert second not in affected
        inst = engine.workflow_instances[second]
        assert inst.current_step_id == "wb"
        assert inst.status == WorkflowStatus.WAITING

    async def test_cancelled_delivery_restores_undelivered_listeners(self):
        wf = create_workflow("cd", "cd")
        wf.with_initial_step(wait_event_step("wa", "WA", "A", success_state="slow"))
        wf.with_step(_SlowStep(step_id="slow", name="Slow", delay=0.5))
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        await engine.start_workflow("cd")
        second = await engine.start_workflow("cd")

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                engine.process_event(WorkflowEvent(event_type="A")), 0.1
            )

        assert second in engine.event_listeners["A"]
        assert second in await engine.process_event(
            WorkflowEvent(event_type="A", instance_id=second)
        )

    async def test_leaving_a_wait_drops_its_listener(self):
        wf = create_workflow("lw", "lw")
        wf.with_initial_step(wait_event_step("wa", "WA", "A", success_state="wb"))
        wf.with_step(wait_event_step("wb", "WB", "B", success_state="done"))
        wf.with_step(auto_step("done", "Done", next_state=""))
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start_workflow("lw", workflow_timeout=60)

        # advance_workflow re-registers the A listener while the event is
        # being delivered; leaving `wa` must drop it.
        await asyncio.gather(
            engine.advance_workflow(iid),
            engine.process_event(WorkflowEvent(event_type="A")),
        )

        assert engine.workflow_instances[iid].current_step_id == "wb"
        assert iid not in engine.event_listeners.get("A", {})
        assert await engine.process_event(WorkflowEvent(event_type="A")) == []
        await engine.shutdown()

    async def test_background_start_skips_cancelled_instance(self):
        ran: list[str] = []
        wf = create_workflow("bgc", "bgc")
        wf.with_initial_step(
            auto_step("charge", "Charge", next_state="", action=lambda c: ran.append(1))
        )
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start_workflow("bgc", wait=False)
        assert await engine.cancel_workflow(iid) is True
        await _settle()

        assert ran == []
        assert engine.get_workflow_status(iid) == WorkflowStatus.CANCELLED

    async def test_long_prompt_fits_core_request_limits(self):
        llm = _CoreLikeLLM("ok")
        step = LLMProcessingStep(
            step_id="llm",
            name="LLM",
            llm_interface=llm,
            prompt_template="x" * 12000,
            output_mapping={"out": ""},
            next_state="done",
        )
        result = await step.execute({})
        assert result.success is True
        assert len(llm.requests[0].user_message) <= 10000
        assert "x" * 12000 in llm.requests[0].system_prompt

    async def test_event_completion_clears_waiting_info(self):
        wf = create_workflow("wc", "wc")
        wf.with_initial_step(wait_event_step("wait", "Wait", "go", success_state=""))
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start_workflow("wc")
        await engine.process_event(WorkflowEvent(event_type="go"))
        assert "_waiting_info" not in engine.workflow_instances[iid].context
