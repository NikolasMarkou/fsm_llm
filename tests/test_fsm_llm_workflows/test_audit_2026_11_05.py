"""
Regression tests for the 2026-11-05 fsm_llm.workflows audit (plan 5980fbb7 step 19).

Classes named ``...Behaviour`` pin a behaviour change (they failed on the parent
commit); the others pin behaviour the refactors of the same step must keep.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from fsm_llm.workflows import (
    WorkflowEngine,
    WorkflowEvent,
    WorkflowStatus,
    api_step,
    auto_step,
    condition_step,
    create_workflow,
    llm_step,
    parallel_step,
    retry_step,
    switch_step,
    timer_step,
    wait_event_step,
)
from fsm_llm.workflows.exceptions import WorkflowInstanceError, WorkflowValidationError

# ---------------------------------------------------------------
# helpers
# ---------------------------------------------------------------


def _record_hooks(engine: WorkflowEngine) -> list[tuple[str, dict[str, Any]]]:
    seen: list[tuple[str, dict[str, Any]]] = []
    engine.add_hook(lambda name, inst, data: seen.append((name, dict(data))))
    return seen


def _single(step) -> Any:
    wf = create_workflow("wf", "wf")
    wf.with_initial_step(step)
    return wf


# ---------------------------------------------------------------
# F-WE-01: a step that fails by raising is recorded like one that returns a failure
# ---------------------------------------------------------------


def _boom(_ctx: dict[str, Any]) -> Any:
    return 1 / 0


class TestRaisingStepsRecordTheFailureBehaviour:
    @pytest.mark.parametrize(
        "make_step",
        [
            lambda: auto_step("s", "S", next_state="", action=_boom),
            lambda: condition_step("s", "S", _boom, true_state="", false_state=""),
        ],
        ids=["auto", "condition"],
    )
    async def test_hook_history_and_cause_text(self, make_step):
        engine = WorkflowEngine()
        seen = _record_hooks(engine)
        engine.register_workflow(_single(make_step()))
        iid = await engine.start_workflow("wf")

        instance = engine.workflow_instances[iid]
        assert instance.status == WorkflowStatus.FAILED
        assert "division by zero" in (instance.error or "")
        completed = [d for n, d in seen if n == "step_completed"]
        assert len(completed) == 1
        assert completed[0]["success"] is False
        assert completed[0]["step_id"] == "s"
        entries = [e for e in instance.history if e.step_id == "s" and e.error]
        assert entries and "division by zero" in entries[0].error

    async def test_execute_returns_a_failure_result_not_an_exception(self):
        step = auto_step("s", "S", next_state="", action=_boom)
        result = await step.execute({})
        assert result.success is False
        assert result.next_state is None
        assert result.error == "division by zero"

    async def test_error_state_route_is_unchanged(self):
        wf = create_workflow("wf", "wf")
        wf.with_initial_step(
            auto_step("s", "S", next_state="ok", action=_boom, error_state="bad")
        )
        wf.with_step(auto_step("ok", "Ok", next_state=""))
        wf.with_step(auto_step("bad", "Bad", next_state=""))
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start_workflow("wf")
        instance = engine.workflow_instances[iid]
        assert instance.status == WorkflowStatus.COMPLETED
        assert next(e.step_id for e in instance.history if e.error) == "s"
        assert "bad" in {e.step_id for e in instance.history}

    async def test_retry_still_retries_a_raising_step(self):
        calls: list[int] = []

        def flaky(ctx: dict[str, Any]) -> dict[str, Any]:
            calls.append(1)
            if len(calls) < 3:
                raise RuntimeError("not yet")
            return {"ok": True}

        wf = _single(
            retry_step(
                "r",
                "R",
                auto_step("s", "S", next_state="", action=flaky),
                max_retries=3,
                backoff_factor=0,
            )
        )
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start_workflow("wf")
        assert engine.get_workflow_status(iid) == WorkflowStatus.COMPLETED
        assert len(calls) == 3


# ---------------------------------------------------------------
# F-WE-06: a TimeoutError raised by user code is not the step's time limit
# ---------------------------------------------------------------


class TestUserTimeoutErrorBehaviour:
    async def test_user_timeout_keeps_its_own_text(self):
        def upstream() -> Any:
            raise asyncio.TimeoutError("upstream socket timed out")

        step = api_step("s", "S", upstream, "", "", timeout_seconds=5)
        result = await step.execute({})
        assert result.success is False
        assert "upstream socket timed out" in (result.error or "")
        assert "Step timed out" not in (result.error or "")

    async def test_the_step_limit_still_reports_the_step_timeout(self):
        async def slow() -> Any:
            await asyncio.sleep(5)

        step = api_step("s", "S", slow, "", "", timeout_seconds=0.05)
        result = await step.execute({})
        assert result.success is False
        assert "timed out after 0.05s" in (result.error or "")


# ---------------------------------------------------------------
# F-WE-07: an async aggregation_function is accepted like every user callable
# ---------------------------------------------------------------


class TestAsyncAggregationBehaviour:
    @pytest.mark.parametrize("use_async", [True, False], ids=["async", "sync"])
    async def test_aggregation_function(self, use_async):
        if use_async:

            async def agg(results: list[Any]) -> dict[str, Any]:
                await asyncio.sleep(0)
                return {"total": len(results)}

        else:

            def agg(results: list[Any]) -> dict[str, Any]:  # type: ignore[misc]
                return {"total": len(results)}

        children = [
            auto_step("c1", "C1", next_state="", action=lambda c: {"x": 1}),
            auto_step("c2", "C2", next_state="", action=lambda c: {"y": 2}),
        ]
        wf = _single(
            parallel_step("p", "P", children, next_state="", aggregation_function=agg)
        )
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start_workflow("wf")
        assert engine.get_workflow_status(iid) == WorkflowStatus.COMPLETED
        assert engine.get_workflow_context(iid)["total"] == 2


# ---------------------------------------------------------------
# F-WE-10: schedule_timer and register_event_listener agree on terminal instances
# ---------------------------------------------------------------


class TestTerminalInstanceTimerBehaviour:
    async def test_schedule_timer_refuses_a_terminal_instance(self):
        engine = WorkflowEngine()
        engine.register_workflow(_single(auto_step("a", "A", next_state="")))
        iid = await engine.start_workflow("wf")
        assert engine.get_workflow_status(iid) == WorkflowStatus.COMPLETED

        with pytest.raises(WorkflowInstanceError, match="completed"):
            await engine.schedule_timer(iid, 60, "a")
        assert engine.timers == {}

    async def test_both_entry_points_refuse_the_same_way(self):
        engine = WorkflowEngine()
        engine.register_workflow(_single(auto_step("a", "A", next_state="")))
        iid = await engine.start_workflow("wf")
        with pytest.raises(WorkflowInstanceError) as listen:
            await engine.register_event_listener(iid, "go")
        with pytest.raises(WorkflowInstanceError) as timer:
            await engine.schedule_timer(iid, 1, "a")
        assert listen.value.instance_id == iid
        assert timer.value.instance_id == iid

    async def test_unknown_instance_is_still_an_instance_error(self):
        engine = WorkflowEngine()
        with pytest.raises(WorkflowInstanceError, match="not found"):
            await engine.schedule_timer("nope", 1, "a")


# ---------------------------------------------------------------
# F-WE-13: the initial PENDING -> RUNNING move is announced like every other
# ---------------------------------------------------------------


class TestInitialRunningIsAnnouncedBehaviour:
    async def test_first_hook_event_is_running(self):
        engine = WorkflowEngine()
        seen: list[tuple[str, str | None, bool]] = []
        engine.add_hook(
            lambda name, inst, data: seen.append(
                (
                    name,
                    data.get("status"),
                    inst.instance_id in engine.workflow_instances,
                )
            )
        )
        engine.register_workflow(_single(auto_step("a", "A", next_state="")))
        await engine.start_workflow("wf")

        assert [n for n, _, _ in seen] == [
            "status_changed",
            "step_started",
            "step_completed",
            "status_changed",
        ]
        assert seen[0][1] == "running"
        assert seen[0][2] is True  # the engine already holds the instance
        assert seen[-1][1] == "completed"


# ---------------------------------------------------------------
# F-WE-15: LLMProcessingStep validates llm_interface at construction
# ---------------------------------------------------------------


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


# ---------------------------------------------------------------
# F-WE-05: every step type's referenced states are validated
# ---------------------------------------------------------------


class _Gen:
    def generate(self, prompt: str) -> str:
        return "x"


class TestReferencedStatesAreValidated:
    @pytest.mark.parametrize(
        "make_step",
        [
            lambda: auto_step("s", "S", next_state="nope"),
            lambda: auto_step("s", "S", next_state="", error_state="nope"),
            lambda: api_step("s", "S", print, "", "nope"),
            lambda: condition_step("s", "S", bool, "", "nope"),
            lambda: condition_step("s", "S", bool, "", "", error_state="nope"),
            lambda: llm_step("s", "S", _Gen(), "p", next_state="nope"),
            lambda: timer_step("s", "S", 1, "nope"),
            lambda: parallel_step("s", "S", [], next_state="", error_state="nope"),
            lambda: wait_event_step("s", "S", "e", success_state="nope"),
            lambda: wait_event_step(
                "s", "S", "e", success_state="", timeout_seconds=1, timeout_state="nope"
            ),
            lambda: switch_step("s", "S", "k", cases={"a": "nope"}),
            lambda: switch_step("s", "S", "k", cases={}, default_state="nope"),
            lambda: retry_step("s", "S", auto_step("i", "I", next_state="nope")),
        ],
        ids=[
            "auto-next",
            "auto-error",
            "api-failure",
            "condition-false",
            "condition-error",
            "llm-next",
            "timer",
            "parallel-error",
            "wait-success",
            "wait-timeout",
            "switch-case",
            "switch-default",
            "retry-inner",
        ],
    )
    def test_unknown_state_is_reported(self, make_step):
        wf = _single(make_step())
        with pytest.raises(WorkflowValidationError) as err:
            wf.validate()
        assert any("unknown state: 'nope'" in e for e in err.value.validation_errors)

    def test_terminal_markers_are_not_references(self):
        wf = create_workflow("wf", "wf")
        wf.with_initial_step(wait_event_step("w", "W", "e", success_state=""))
        wf.validate()
        assert wf.get_terminal_states() == {"w"}

    def test_unreachable_step_is_reported(self):
        wf = create_workflow("wf", "wf")
        wf.with_initial_step(auto_step("a", "A", next_state=""))
        wf.with_step(auto_step("lonely", "Lonely", next_state=""))
        with pytest.raises(WorkflowValidationError, match="not reachable"):
            wf.validate()


# ---------------------------------------------------------------
# F-WE-02 / F-WE-03 / F-WE-04: timers, completing from a wait, resource release
# ---------------------------------------------------------------


class TestTimerKeysAndRearm:
    async def test_the_three_timer_keys(self):
        wf = create_workflow("wf", "wf")
        wf.with_initial_step(
            wait_event_step(
                "w", "W", "go", success_state="t", timeout_seconds=60, timeout_state="t"
            )
        )
        wf.with_step(timer_step("t", "T", 60, ""))
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start_workflow("wf", max_seconds=120)
        assert set(engine.timers) == {f"{iid}_go_timeout", f"{iid}_deadline"}

        await engine.process_event(WorkflowEvent(event_type="go"))
        assert set(engine.timers) == {f"{iid}_timer", f"{iid}_deadline"}
        assert engine.get_workflow_status(iid) == WorkflowStatus.WAITING

    async def test_rearming_replaces_and_cancels_the_old_task(self):
        wf = create_workflow("wf", "wf")
        wf.with_initial_step(timer_step("t", "T", 60, ""))
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start_workflow("wf")
        first = engine.timers[f"{iid}_timer"]

        assert await engine.advance_workflow(iid) is True
        second = engine.timers[f"{iid}_timer"]
        await asyncio.sleep(0)
        assert second is not first
        assert first.task is not None and first.task.cancelled()
        assert second.task is not None and not second.task.done()
        await engine.shutdown()

    async def test_timer_fires_and_forgets_its_entry(self):
        wf = create_workflow("wf", "wf")
        wf.with_initial_step(timer_step("t", "T", 0.01, "done"))
        wf.with_step(auto_step("done", "Done", next_state=""))
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start_workflow("wf")
        await asyncio.sleep(0.1)
        assert engine.get_workflow_status(iid) == WorkflowStatus.COMPLETED
        assert engine.timers == {}

    async def test_event_timeout_without_state_fails_and_forgets(self):
        wf = create_workflow("wf", "wf")
        wf.with_initial_step(
            wait_event_step("w", "W", "go", success_state="", timeout_seconds=0.01)
        )
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start_workflow("wf")
        await asyncio.sleep(0.1)
        assert engine.get_workflow_status(iid) == WorkflowStatus.FAILED
        assert engine.timers == {}
        assert engine.event_listeners.get("go", {}) == {}

    async def test_deadline_fails_a_waiting_instance_and_forgets_its_entry(self):
        wf = create_workflow("wf", "wf")
        wf.with_initial_step(wait_event_step("w", "W", "go", success_state=""))
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start_workflow("wf", max_seconds=0.02)
        await asyncio.sleep(0.15)
        assert engine.get_workflow_status(iid) == WorkflowStatus.FAILED
        assert engine.timers == {}


class TestCompletingFromAWait:
    @staticmethod
    def _markers(engine: WorkflowEngine, iid: str) -> tuple[bool, bool]:
        ctx = engine.workflow_instances[iid].context
        return "_waiting_info" in ctx, "_timer_info" in ctx

    async def test_event_without_success_state(self):
        wf = create_workflow("wf", "wf")
        wf.with_initial_step(
            wait_event_step("w", "W", "go", success_state="", timeout_seconds=60)
        )
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start_workflow("wf")
        assert await engine.process_event(WorkflowEvent(event_type="go")) == [iid]

        assert engine.get_workflow_status(iid) == WorkflowStatus.COMPLETED
        assert self._markers(engine, iid) == (False, False)
        assert engine.timers == {}
        assert engine.event_listeners.get("go", {}) == {}

    async def test_timer_with_empty_target(self):
        wf = create_workflow("wf", "wf")
        wf.with_initial_step(timer_step("t", "T", 0.01, ""))
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start_workflow("wf")
        await asyncio.sleep(0.1)

        assert engine.get_workflow_status(iid) == WorkflowStatus.COMPLETED
        assert self._markers(engine, iid) == (False, False)
        assert engine.timers == {}

    async def test_buffered_event_without_success_state(self):
        wf = create_workflow("wf", "wf")
        wf.with_initial_step(timer_step("pause", "Pause", 60, "wait"))
        wf.with_step(wait_event_step("wait", "Wait", "paid", success_state=""))
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start_workflow("wf")
        await engine.process_event(WorkflowEvent(event_type="paid", instance_id=iid))
        await engine._handle_timer_expiration(iid, "wait")

        assert engine.get_workflow_status(iid) == WorkflowStatus.COMPLETED
        assert self._markers(engine, iid) == (False, False)

    async def test_transition_out_of_a_wait_clears_both_markers(self):
        wf = create_workflow("wf", "wf")
        wf.with_initial_step(timer_step("t", "T", 0.01, "w"))
        wf.with_step(wait_event_step("w", "W", "go", success_state="done"))
        wf.with_step(auto_step("done", "Done", next_state=""))
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start_workflow("wf")
        await asyncio.sleep(0.1)
        assert engine.get_workflow_status(iid) == WorkflowStatus.WAITING
        assert self._markers(engine, iid) == (True, False)
        await engine.process_event(WorkflowEvent(event_type="go"))
        assert engine.get_workflow_status(iid) == WorkflowStatus.COMPLETED
        assert self._markers(engine, iid) == (False, False)


class TestCancelReleasesEverything:
    async def test_cancel_a_waiting_instance(self):
        wf = create_workflow("wf", "wf")
        wf.with_initial_step(
            wait_event_step("w", "W", "go", success_state="", timeout_seconds=60)
        )
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start_workflow("wf", max_seconds=120)
        assert engine.timers and engine.event_listeners["go"]

        assert await engine.cancel_workflow(iid) is True
        assert engine.get_workflow_status(iid) == WorkflowStatus.CANCELLED
        assert engine.timers == {}
        assert engine.event_listeners.get("go", {}) == {}

    async def test_a_raised_transition_error_fails_the_instance_once(self):
        # A step routing to an unknown state at run time FAILS the instance
        # (the failure path both _execute_workflow_step sites share).
        from fsm_llm.workflows.models import WorkflowStepResult
        from fsm_llm.workflows.steps import WorkflowStep

        class _Stray(WorkflowStep):
            async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
                return WorkflowStepResult.success_result(next_state="nowhere")

        wf = create_workflow("wf", "wf")
        wf.with_initial_step(_Stray(step_id="s", name="S"))
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start_workflow("wf")
        instance = engine.workflow_instances[iid]
        assert instance.status == WorkflowStatus.FAILED
        assert "Invalid next state" in (instance.error or "")
