"""Tests moved from dated audit files."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from fsm_llm.workflows import (
    WorkflowEngine,
    WorkflowEvent,
    WorkflowStatus,
    auto_step,
    create_workflow,
    timer_step,
    wait_event_step,
)
from fsm_llm.workflows.definitions import WorkflowDefinition
from fsm_llm.workflows.exceptions import WorkflowInstanceError
from fsm_llm.workflows.models import WaitEventConfig, WorkflowStepResult
from fsm_llm.workflows.steps import LLMStep, WorkflowStep
from tests.test_fsm_llm_workflows.support import (
    _CoreLikeLLM,
    _settle,
    _single,
    _wait_then_done,
)


class TestProcessEventIsolation:
    async def test_expired_deadline_on_one_instance_does_not_strand_others(self):
        engine = WorkflowEngine()
        engine.register_workflow(_wait_then_done("wf"))
        # A: tiny workflow timeout; B: none. A is registered first.
        a = await engine.start("wf", max_seconds=5)
        b = await engine.start("wf")
        engine.workflow_instances[a].deadline = engine.workflow_instances[
            a
        ].created_at.replace(year=2000)

        affected = await engine.process_event(WorkflowEvent(event_type="go"))

        assert b in affected
        assert engine.get_status(b) == WorkflowStatus.COMPLETED
        assert engine.get_status(a) == WorkflowStatus.FAILED

    async def test_non_waiting_instance_is_skipped_and_not_mutated(self):
        engine = WorkflowEngine()
        engine.register_workflow(_wait_then_done("wf", event_mapping={"got": "value"}))
        a = await engine.start("wf")
        b = await engine.start("wf")
        # A leaves WAITING without its listener being cleaned up (the race
        # between a cancel and a concurrent process_event).
        engine.workflow_instances[a].status = WorkflowStatus.CANCELLED

        affected = await engine.process_event(
            WorkflowEvent(event_type="go", payload={"value": 1})
        )

        assert affected == [b]
        assert "got" not in engine.workflow_instances[a].context
        assert engine.get_status(a) == WorkflowStatus.CANCELLED
        assert engine.get_status(b) == WorkflowStatus.COMPLETED


class TestRewaitTimeoutSurvives:
    async def test_second_wait_on_same_event_still_times_out(self, timing_factor):
        wf = create_workflow("rewait", "Rewait")
        wf.with_initial_step(wait_event_step("w1", "W1", "go", success_step_id="w2"))
        wf.with_step(
            wait_event_step(
                "w2",
                "W2",
                "go",
                success_step_id="done",
                timeout_seconds=0.05 * timing_factor,
                timeout_step_id="timed_out",
            )
        )
        wf.with_step(auto_step("done", "Done", next_step_id=""))
        wf.with_step(auto_step("timed_out", "Timed out", next_step_id=""))
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start("rewait")

        await engine.process_event(WorkflowEvent(event_type="go"))
        assert engine.workflow_instances[iid].current_step_id == "w2"
        await _settle(0.2 * timing_factor)

        inst = engine.workflow_instances[iid]
        assert inst.current_step_id == "timed_out"
        assert inst.status == WorkflowStatus.COMPLETED


class TestWaitTimeoutsAndTerminalWaits:
    async def test_timeout_without_timeout_step_id_fails_instance(self, timing_factor):
        engine = WorkflowEngine()
        engine.register_workflow(
            _wait_then_done("wf", timeout_seconds=0.05 * timing_factor)
        )
        iid = await engine.start("wf")
        await _settle(0.2 * timing_factor)

        inst = engine.workflow_instances[iid]
        assert inst.status == WorkflowStatus.FAILED
        assert "timed out" in (inst.error or "")
        assert engine.timers == {}

    async def test_empty_success_step_id_completes_on_event(self):
        wf = create_workflow("wf", "wf")
        wf.with_initial_step(wait_event_step("wait", "Wait", "go", success_step_id=""))
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start("wf")

        assert await engine.process_event(WorkflowEvent(event_type="go")) == [iid]
        assert engine.get_status(iid) == WorkflowStatus.COMPLETED

    def test_timeout_step_id_without_timeout_seconds_rejected(self):
        with pytest.raises(ValueError, match="timeout_step_id requires"):
            WaitEventConfig(event_type="go", success_step_id="x", timeout_step_id="t")

    def test_empty_event_type_rejected(self):
        with pytest.raises(ValueError):
            WaitEventConfig(event_type="", success_step_id="x")


class TestEventCorrelation:
    async def test_correlation_key_routes_to_matching_instance(self):
        engine = WorkflowEngine()
        engine.register_workflow(_wait_then_done("wf", correlation_key="order_id"))
        one = await engine.start("wf", {"order_id": 1})
        two = await engine.start("wf", {"order_id": 2})

        affected = await engine.process_event(
            WorkflowEvent(event_type="go", payload={"order_id": 2})
        )

        assert affected == [two]
        assert engine.get_status(one) == WorkflowStatus.WAITING
        assert engine.get_status(two) == WorkflowStatus.COMPLETED

    async def test_targeted_event_only_reaches_target(self):
        engine = WorkflowEngine()
        engine.register_workflow(_wait_then_done("wf"))
        one = await engine.start("wf")
        two = await engine.start("wf")

        affected = await engine.process_event(
            WorkflowEvent(event_type="go", instance_id=one)
        )

        assert affected == [one]
        assert engine.get_status(two) == WorkflowStatus.WAITING

    async def test_early_targeted_event_is_buffered(self):
        wf = create_workflow("wf", "wf")
        wf.with_initial_step(timer_step("pause", "Pause", 60, "wait"))
        wf.with_step(
            wait_event_step(
                "wait",
                "Wait",
                "paid",
                success_step_id="done",
                event_mapping={"amt": "a"},
            )
        )
        wf.with_step(auto_step("done", "Done", next_step_id=""))
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start("wf")

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


class _SlowStep(WorkflowStep):
    next_step_id: str = ""
    delay: float = 0.3

    async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
        await asyncio.sleep(self.delay)
        return WorkflowStepResult.success_result(next_step_id=self.next_step_id)


def _two_waits_workflow(timing_factor: float = 1.0) -> WorkflowDefinition:
    """wa (A, times out to wb) -> slow -> "" ; wb waits for B.

    The 0.1 s timeout and the 0.3 s slow step scale together by *timing_factor*,
    so the order they must keep survives a traced or loaded run.
    """
    wf = create_workflow("tw", "tw")
    wf.with_initial_step(
        wait_event_step(
            "wa",
            "WA",
            "A",
            success_step_id="slow",
            timeout_seconds=0.1 * timing_factor,
            timeout_step_id="wb",
        )
    )
    wf.with_step(_SlowStep(step_id="slow", name="Slow", delay=0.3 * timing_factor))
    wf.with_step(wait_event_step("wb", "WB", "B", success_step_id="got_b"))
    wf.with_step(auto_step("got_b", "Got B", next_step_id=""))
    return wf


class TestReviewFollowUp:
    async def test_event_not_delivered_to_a_different_wait(self, timing_factor):
        engine = WorkflowEngine()
        engine.register_workflow(_two_waits_workflow(timing_factor))
        first = await engine.start("tw")
        second = await engine.start("tw")

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
        wf.with_initial_step(wait_event_step("wa", "WA", "A", success_step_id="slow"))
        wf.with_step(_SlowStep(step_id="slow", name="Slow", delay=0.5))
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        await engine.start("cd")
        second = await engine.start("cd")

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
        wf.with_initial_step(wait_event_step("wa", "WA", "A", success_step_id="wb"))
        wf.with_step(wait_event_step("wb", "WB", "B", success_step_id="done"))
        wf.with_step(auto_step("done", "Done", next_step_id=""))
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start("lw", max_seconds=60)

        # advance re-registers the A listener while the event is
        # being delivered; leaving `wa` must drop it.
        await asyncio.gather(
            engine.advance(iid),
            engine.process_event(WorkflowEvent(event_type="A")),
        )

        assert engine.workflow_instances[iid].current_step_id == "wb"
        assert iid not in engine.event_listeners.get("A", {})
        assert await engine.process_event(WorkflowEvent(event_type="A")) == []
        await engine.close()

    async def test_background_start_skips_cancelled_instance(self):
        ran: list[str] = []
        wf = create_workflow("bgc", "bgc")
        wf.with_initial_step(
            auto_step(
                "charge", "Charge", next_step_id="", action=lambda c: ran.append(1)
            )
        )
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start("bgc", wait=False)
        assert await engine.cancel(iid) is True
        await _settle()

        assert ran == []
        assert engine.get_status(iid) == WorkflowStatus.CANCELLED

    async def test_long_prompt_fits_core_request_limits(self):
        llm = _CoreLikeLLM("ok")
        step = LLMStep(
            step_id="llm",
            name="LLM",
            llm_interface=llm,
            prompt_template="x" * 12000,
            output_mapping={"out": ""},
            next_step_id="done",
        )
        result = await step.execute({})
        assert result.success is True
        assert len(llm.requests[0].user_message) <= 10000
        assert "x" * 12000 in llm.requests[0].system_prompt

    async def test_event_completion_clears_waiting_info(self):
        wf = create_workflow("wc", "wc")
        wf.with_initial_step(wait_event_step("wait", "Wait", "go", success_step_id=""))
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start("wc")
        await engine.process_event(WorkflowEvent(event_type="go"))
        assert "_waiting_info" not in engine.workflow_instances[iid].context


class TestTerminalInstanceTimerBehaviour:
    async def test_schedule_timer_refuses_a_terminal_instance(self):
        engine = WorkflowEngine()
        engine.register_workflow(_single(auto_step("a", "A", next_step_id="")))
        iid = await engine.start("wf")
        assert engine.get_status(iid) == WorkflowStatus.COMPLETED

        with pytest.raises(WorkflowInstanceError, match="completed"):
            await engine.schedule_timer(iid, 60, "a")
        assert engine.timers == {}

    async def test_both_entry_points_refuse_the_same_way(self):
        engine = WorkflowEngine()
        engine.register_workflow(_single(auto_step("a", "A", next_step_id="")))
        iid = await engine.start("wf")
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


class TestTimerKeysAndRearm:
    async def test_the_three_timer_keys(self):
        wf = create_workflow("wf", "wf")
        wf.with_initial_step(
            wait_event_step(
                "w",
                "W",
                "go",
                success_step_id="t",
                timeout_seconds=60,
                timeout_step_id="t",
            )
        )
        wf.with_step(timer_step("t", "T", 60, ""))
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start("wf", max_seconds=120)
        assert set(engine.timers) == {f"{iid}_go_timeout", f"{iid}_deadline"}

        await engine.process_event(WorkflowEvent(event_type="go"))
        assert set(engine.timers) == {f"{iid}_timer", f"{iid}_deadline"}
        assert engine.get_status(iid) == WorkflowStatus.WAITING

    async def test_rearming_replaces_and_cancels_the_old_task(self):
        wf = create_workflow("wf", "wf")
        wf.with_initial_step(timer_step("t", "T", 60, ""))
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start("wf")
        first = engine.timers[f"{iid}_timer"]

        assert await engine.advance(iid) is True
        second = engine.timers[f"{iid}_timer"]
        await asyncio.sleep(0)
        assert second is not first
        assert first.task is not None and first.task.cancelled()
        assert second.task is not None and not second.task.done()
        await engine.close()

    async def test_timer_fires_and_forgets_its_entry(self, timing_factor):
        wf = create_workflow("wf", "wf")
        wf.with_initial_step(timer_step("t", "T", 0.01 * timing_factor, "done"))
        wf.with_step(auto_step("done", "Done", next_step_id=""))
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start("wf")
        await asyncio.sleep(0.1 * timing_factor)
        assert engine.get_status(iid) == WorkflowStatus.COMPLETED
        assert engine.timers == {}

    async def test_event_timeout_without_state_fails_and_forgets(self, timing_factor):
        wf = create_workflow("wf", "wf")
        wf.with_initial_step(
            wait_event_step(
                "w", "W", "go", success_step_id="", timeout_seconds=0.01 * timing_factor
            )
        )
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start("wf")
        await asyncio.sleep(0.1 * timing_factor)
        assert engine.get_status(iid) == WorkflowStatus.FAILED
        assert engine.timers == {}
        assert engine.event_listeners.get("go", {}) == {}

    async def test_deadline_fails_a_waiting_instance_and_forgets_its_entry(
        self, timing_factor
    ):
        wf = create_workflow("wf", "wf")
        wf.with_initial_step(wait_event_step("w", "W", "go", success_step_id=""))
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start("wf", max_seconds=0.02 * timing_factor)
        await asyncio.sleep(0.15 * timing_factor)
        assert engine.get_status(iid) == WorkflowStatus.FAILED
        assert engine.timers == {}


class TestCompletingFromAWait:
    @staticmethod
    def _markers(engine: WorkflowEngine, iid: str) -> tuple[bool, bool]:
        ctx = engine.workflow_instances[iid].context
        return "_waiting_info" in ctx, "_timer_info" in ctx

    async def test_event_without_success_step_id(self):
        wf = create_workflow("wf", "wf")
        wf.with_initial_step(
            wait_event_step("w", "W", "go", success_step_id="", timeout_seconds=60)
        )
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start("wf")
        assert await engine.process_event(WorkflowEvent(event_type="go")) == [iid]

        assert engine.get_status(iid) == WorkflowStatus.COMPLETED
        assert self._markers(engine, iid) == (False, False)
        assert engine.timers == {}
        assert engine.event_listeners.get("go", {}) == {}

    async def test_timer_with_empty_target(self, timing_factor):
        wf = create_workflow("wf", "wf")
        wf.with_initial_step(timer_step("t", "T", 0.01 * timing_factor, ""))
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start("wf")
        await asyncio.sleep(0.1 * timing_factor)

        assert engine.get_status(iid) == WorkflowStatus.COMPLETED
        assert self._markers(engine, iid) == (False, False)
        assert engine.timers == {}

    async def test_buffered_event_without_success_step_id(self):
        wf = create_workflow("wf", "wf")
        wf.with_initial_step(timer_step("pause", "Pause", 60, "wait"))
        wf.with_step(wait_event_step("wait", "Wait", "paid", success_step_id=""))
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start("wf")
        await engine.process_event(WorkflowEvent(event_type="paid", instance_id=iid))
        await engine._handle_timer_expiration(iid, "wait")

        assert engine.get_status(iid) == WorkflowStatus.COMPLETED
        assert self._markers(engine, iid) == (False, False)

    async def test_transition_out_of_a_wait_clears_both_markers(self, timing_factor):
        wf = create_workflow("wf", "wf")
        wf.with_initial_step(timer_step("t", "T", 0.01 * timing_factor, "w"))
        wf.with_step(wait_event_step("w", "W", "go", success_step_id="done"))
        wf.with_step(auto_step("done", "Done", next_step_id=""))
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start("wf")
        await asyncio.sleep(0.1 * timing_factor)
        assert engine.get_status(iid) == WorkflowStatus.WAITING
        assert self._markers(engine, iid) == (True, False)
        await engine.process_event(WorkflowEvent(event_type="go"))
        assert engine.get_status(iid) == WorkflowStatus.COMPLETED
        assert self._markers(engine, iid) == (False, False)


class TestEventListenerRaceCondition:
    """F-002: process_event must not crash if listener was already removed."""

    def test_pop_instead_of_del_in_process_event(self):
        """Verify engine uses .pop() for listener cleanup (race-safe)."""
        import inspect

        source = inspect.getsource(WorkflowEngine.process_event)
        # Must NOT use bare del on event_listeners
        assert "del self.event_listeners" not in source
        # Should use .pop() for safe removal
        assert ".pop(" in source

    async def test_process_event_no_listeners(self):
        """process_event with no matching listeners should return empty list."""
        engine = WorkflowEngine()
        event = WorkflowEvent(event_type="nonexistent_event", payload={})
        result = await engine.process_event(event)
        assert result == []

    async def test_process_event_empty_listener_dict(self):
        """process_event with empty listener dict for event type should return empty."""
        engine = WorkflowEngine()
        engine.event_listeners["test_event"] = {}
        event = WorkflowEvent(event_type="test_event", payload={})
        result = await engine.process_event(event)
        assert result == []
