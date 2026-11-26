"""Workflow engine lifecycle: instance ids, deadlines, loop and step budgets, resource release, definition pinning, step failures, cancellation and instance-lock concurrency."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import textwrap
from datetime import UTC
from typing import Any

import pytest

from fsm_llm.workflows import (
    WorkflowEngine,
    WorkflowStatus,
    auto_step,
    condition_step,
    create_workflow,
    parallel_step,
    retry_step,
    timer_step,
    wait_event_step,
)
from fsm_llm.workflows.exceptions import (
    WorkflowInstanceError,
    WorkflowResourceError,
    WorkflowTimeoutError,
    WorkflowValidationError,
)
from fsm_llm.workflows.models import WorkflowEvent, WorkflowStepResult
from fsm_llm.workflows.steps import WorkflowStep
from tests.test_fsm_llm_workflows.support import _settle, _single, _wait_then_done


class _DynamicLoopStep(WorkflowStep):
    """A custom step that always routes back to itself at runtime (the
    static graph cannot see it)."""

    async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
        return WorkflowStepResult.success_result(next_step_id=self.step_id)


class TestInstanceIds:
    async def test_cancel_does_not_touch_timers_of_prefix_sharing_ids(self):
        wf = create_workflow("t", "t")
        wf.with_initial_step(timer_step("wait", "Wait", 60, "done"))
        wf.with_step(auto_step("done", "Done", next_step_id=""))
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        await engine.start("t", instance_id="order")
        await engine.start("t", instance_id="order_2")

        await engine.cancel("order")

        assert "order_2_timer" in engine.timers
        assert not engine.timers["order_2_timer"].task.cancelled()
        await engine.close()

    async def test_duplicate_instance_id_rejected(self):
        engine = WorkflowEngine()
        engine.register_workflow(_wait_then_done("wf"))
        await engine.start("wf", instance_id="same")
        with pytest.raises(WorkflowInstanceError, match="already in use"):
            await engine.start("wf", instance_id="same")
        assert engine.get_status("same") == WorkflowStatus.WAITING


class TestWorkflowDeadline:
    async def test_deadline_fails_waiting_instance(self, timing_factor):
        engine = WorkflowEngine()
        engine.register_workflow(_wait_then_done("wf"))
        iid = await engine.start("wf", max_seconds=0.05 * timing_factor)
        assert engine.get_status(iid) == WorkflowStatus.WAITING
        await _settle(0.2 * timing_factor)

        assert engine.get_status(iid) == WorkflowStatus.FAILED
        assert engine.event_listeners["go"] == {}

    async def test_timeout_error_names_instance(self, timing_factor):
        class _Slow(WorkflowStep):
            async def execute(self, context):
                await asyncio.sleep(1 * timing_factor)
                return WorkflowStepResult.success_result(next_step_id="")

        wf = create_workflow("slow", "slow")
        wf.with_initial_step(_Slow(step_id="s", name="S"))
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        with pytest.raises(WorkflowTimeoutError) as exc_info:
            await engine.start("slow", max_seconds=0.05 * timing_factor)

        iid = exc_info.value.instance_id
        assert iid is not None
        assert engine.get_status(iid) == WorkflowStatus.FAILED


class TestLoopsAndStepBudget:
    async def test_long_acyclic_chain_completes(self):
        wf = create_workflow("long", "long")
        n = 30
        for i in range(n):
            nxt = f"s{i + 1}" if i < n - 1 else ""
            wf.with_step(
                auto_step(f"s{i}", f"S{i}", next_step_id=nxt), is_initial=i == 0
            )
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start("long")
        assert engine.get_status(iid) == WorkflowStatus.COMPLETED

    async def test_polling_loop_through_timer_is_allowed(self, timing_factor):
        polls = {"n": 0}

        def poll(ctx):
            polls["n"] += 1
            return {"ready": polls["n"] >= 3}

        wf = create_workflow("poll", "poll")
        wf.with_initial_step(
            auto_step("check", "Check", next_step_id="route", action=poll)
        )
        wf.with_step(
            condition_step(
                "route",
                "Ready?",
                condition=lambda ctx: ctx["ready"],
                true_step_id="done",
                false_step_id="sleep",
            )
        )
        wf.with_step(timer_step("sleep", "Sleep", 0, "check"))
        wf.with_step(auto_step("done", "Done", next_step_id=""))
        assert wf.has_cycles() is True
        assert wf.has_synchronous_cycles() is False

        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start("poll")
        await _settle(0.2 * timing_factor)

        assert engine.get_status(iid) == WorkflowStatus.COMPLETED
        assert polls["n"] == 3

    def test_synchronous_cycle_rejected(self):
        wf = create_workflow("cyc", "cyc")
        wf.with_initial_step(auto_step("a", "A", next_step_id="b"))
        wf.with_step(auto_step("b", "B", next_step_id="a"))
        with pytest.raises(WorkflowValidationError, match="synchronous cycle"):
            wf.validate_structure()

    async def test_dynamic_loop_hits_step_budget(self):
        wf = create_workflow("dyn", "dyn")
        wf.with_initial_step(_DynamicLoopStep(step_id="loop", name="Loop"))
        engine = WorkflowEngine(max_steps=5)
        engine.register_workflow(wf)
        iid = await engine.start("dyn")

        inst = engine.workflow_instances[iid]
        assert inst.status == WorkflowStatus.FAILED
        assert "Step budget exceeded" in (inst.error or "")


class TestResourceRelease:
    async def test_fired_event_timeout_leaves_no_timer(self, timing_factor):
        wf = create_workflow("wf", "wf")
        wf.with_initial_step(
            wait_event_step(
                "wait",
                "Wait",
                "go",
                success_step_id="done",
                timeout_seconds=0.05 * timing_factor,
                timeout_step_id="late",
            )
        )
        wf.with_step(auto_step("done", "Done", next_step_id=""))
        wf.with_step(auto_step("late", "Late", next_step_id=""))
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start("wf")
        await _settle(0.2 * timing_factor)

        assert engine.get_status(iid) == WorkflowStatus.COMPLETED
        assert engine.timers == {}
        assert engine.get_statistics()["active_timers"] == 0

    def test_default_retention_is_bounded(self):
        assert WorkflowEngine().max_completed_instances is not None

    async def test_failed_instances_are_purged_too(self):
        wf = create_workflow("f", "f")
        wf.with_initial_step(
            auto_step("boom", "Boom", next_step_id="", action=lambda ctx: 1 / 0)
        )
        engine = WorkflowEngine(max_completed_instances=2)
        engine.register_workflow(wf)
        ids = [await engine.start("f") for _ in range(4)]
        assert all(engine.get_status(i) in (WorkflowStatus.FAILED, None) for i in ids)
        assert len(engine.workflow_instances) == 2

    def test_history_is_capped(self):
        from fsm_llm.workflows.models import WorkflowInstance

        inst = WorkflowInstance(
            instance_id="i", workflow_id="w", current_step_id="s", max_history_entries=3
        )
        for i in range(10):
            inst.add_history_entry(step_id="s", message=str(i))
        assert [e.message for e in inst.history] == ["7", "8", "9"]


class TestDefinitionPinning:
    async def test_reregistration_does_not_affect_running_instance(self):
        engine = WorkflowEngine()
        engine.register_workflow(_wait_then_done("wf"))
        iid = await engine.start("wf")

        replacement = create_workflow("wf", "v2")
        replacement.with_initial_step(auto_step("other", "Other", next_step_id=""))
        engine.register_workflow(replacement)
        await engine.process_event(WorkflowEvent(event_type="go"))

        inst = engine.workflow_instances[iid]
        assert inst.status == WorkflowStatus.COMPLETED
        assert inst.current_step_id == "done"

    def test_mutating_after_registration_does_not_change_registered(self):
        engine = WorkflowEngine()
        wf = _wait_then_done("wf")
        engine.register_workflow(wf)
        wf.with_step(auto_step("extra", "Extra", next_step_id=""))
        assert "extra" not in engine.get_definition("wf").steps


class TestEngineApi:
    async def test_hooks_see_steps_and_status_changes(self):
        seen: list[tuple[str, dict]] = []
        engine = WorkflowEngine()
        engine.add_hook(lambda name, inst, data: seen.append((name, data)))
        engine.add_hook(lambda *a: 1 / 0)  # a broken hook must not matter
        wf = create_workflow("h", "h")
        wf.with_initial_step(auto_step("a", "A", next_step_id=""))
        engine.register_workflow(wf)
        iid = await engine.start("h")

        assert engine.get_status(iid) == WorkflowStatus.COMPLETED
        names = [n for n, _ in seen]
        assert names == [
            "status_changed",
            "step_started",
            "step_completed",
            "status_changed",
        ]
        assert seen[0][1]["status"] == "running"
        assert seen[-1][1]["status"] == "completed"

    async def test_user_input_removed_after_advance(self):
        engine = WorkflowEngine()
        engine.register_workflow(_wait_then_done("wf"))
        iid = await engine.start("wf")
        assert await engine.advance(iid, "hi") is True
        assert "_user_input" not in engine.workflow_instances[iid].context
        assert (
            "auto_transition_states"
            not in engine.workflow_instances[iid].context["_workflow_info"]
        )

    async def test_background_start(self):
        engine = WorkflowEngine()
        wf = create_workflow("bg", "bg")
        wf.with_initial_step(auto_step("a", "A", next_step_id=""))
        engine.register_workflow(wf)
        iid = await engine.start("bg", wait=False)
        await _settle()
        assert engine.get_status(iid) == WorkflowStatus.COMPLETED

    async def test_context_getter_returns_copy_and_initial_context_is_copied(self):
        engine = WorkflowEngine()
        engine.register_workflow(_wait_then_done("wf"))
        initial = {"a": 1}
        iid = await engine.start("wf", initial)
        engine.get_context(iid)["a"] = 999
        assert engine.workflow_instances[iid].context["a"] == 1
        assert "_workflow_info" not in initial

    async def test_close_refuses_new_starts_and_clears_listeners(self):
        engine = WorkflowEngine()
        engine.register_workflow(_wait_then_done("wf", timeout_seconds=60))
        await engine.start("wf")
        await engine.close()
        assert engine.timers == {}
        assert engine.event_listeners == {}
        with pytest.raises(WorkflowResourceError):
            await engine.start("wf")


class TestLoggingOffByDefault:
    def test_workflow_logs_follow_library_logging_switch(self):
        script = textwrap.dedent(
            """
            import asyncio
            from loguru import logger
            import fsm_llm.workflows as wfl
            from fsm_llm.logging import enable_library_logging

            lines = []
            logger.add(lambda m: lines.append(str(m)), level="DEBUG")

            async def run():
                engine = wfl.WorkflowEngine()
                wf = wfl.create_workflow("w", "w")
                wf.with_initial_step(wfl.auto_step("a", "A", next_step_id=""))
                engine.register_workflow(wf)
                await engine.start("w")

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


class TestStepExecutor:
    async def test_sync_callables_run_on_engine_executor(self):
        import threading
        from concurrent.futures import ThreadPoolExecutor

        wf = create_workflow("ex", "ex")
        wf.with_initial_step(
            auto_step(
                "a",
                "A",
                next_step_id="",
                action=lambda ctx: {"thread": threading.current_thread().name},
            )
        )
        with ThreadPoolExecutor(thread_name_prefix="wf-exec") as pool:
            engine = WorkflowEngine(executor=pool)
            engine.register_workflow(wf)
            iid = await engine.start("ex")
        assert engine.get_context(iid)["thread"].startswith("wf-exec")


def _record_hooks(engine: WorkflowEngine) -> list[tuple[str, dict[str, Any]]]:
    seen: list[tuple[str, dict[str, Any]]] = []
    engine.add_hook(lambda name, inst, data: seen.append((name, dict(data))))
    return seen


def _boom(_ctx: dict[str, Any]) -> Any:
    return 1 / 0


class TestRaisingStepsRecordTheFailureBehaviour:
    @pytest.mark.parametrize(
        "make_step",
        [
            lambda: auto_step("s", "S", next_step_id="", action=_boom),
            lambda: condition_step("s", "S", _boom, true_step_id="", false_step_id=""),
        ],
        ids=["auto", "condition"],
    )
    async def test_hook_history_and_cause_text(self, make_step):
        engine = WorkflowEngine()
        seen = _record_hooks(engine)
        engine.register_workflow(_single(make_step()))
        iid = await engine.start("wf")

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
        step = auto_step("s", "S", next_step_id="", action=_boom)
        result = await step.execute({})
        assert result.success is False
        assert result.next_step_id is None
        assert result.error == "division by zero"

    async def test_error_step_id_route_is_unchanged(self):
        wf = create_workflow("wf", "wf")
        wf.with_initial_step(
            auto_step("s", "S", next_step_id="ok", action=_boom, error_step_id="bad")
        )
        wf.with_step(auto_step("ok", "Ok", next_step_id=""))
        wf.with_step(auto_step("bad", "Bad", next_step_id=""))
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start("wf")
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
                auto_step("s", "S", next_step_id="", action=flaky),
                max_retries=3,
                backoff_factor=0,
            )
        )
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start("wf")
        assert engine.get_status(iid) == WorkflowStatus.COMPLETED
        assert len(calls) == 3


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
            auto_step("c1", "C1", next_step_id="", action=lambda c: {"x": 1}),
            auto_step("c2", "C2", next_step_id="", action=lambda c: {"y": 2}),
        ]
        wf = _single(
            parallel_step("p", "P", children, next_step_id="", aggregation_function=agg)
        )
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start("wf")
        assert engine.get_status(iid) == WorkflowStatus.COMPLETED
        assert engine.get_context(iid)["total"] == 2


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
        engine.register_workflow(_single(auto_step("a", "A", next_step_id="")))
        await engine.start("wf")

        assert [n for n, _, _ in seen] == [
            "status_changed",
            "step_started",
            "step_completed",
            "status_changed",
        ]
        assert seen[0][1] == "running"
        assert seen[0][2] is True  # the engine already holds the instance
        assert seen[-1][1] == "completed"


class TestCancelReleasesEverything:
    async def test_cancel_a_waiting_instance(self):
        wf = create_workflow("wf", "wf")
        wf.with_initial_step(
            wait_event_step("w", "W", "go", success_step_id="", timeout_seconds=60)
        )
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start("wf", max_seconds=120)
        assert engine.timers and engine.event_listeners["go"]

        assert await engine.cancel(iid) is True
        assert engine.get_status(iid) == WorkflowStatus.CANCELLED
        assert engine.timers == {}
        assert engine.event_listeners.get("go", {}) == {}

    async def test_a_raised_transition_error_fails_the_instance_once(self):
        # A step routing to an unknown state at run time FAILS the instance
        # (the failure path both _execute_workflow_step sites share).
        from fsm_llm.workflows.models import WorkflowStepResult
        from fsm_llm.workflows.steps import WorkflowStep

        class _Stray(WorkflowStep):
            async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
                return WorkflowStepResult.success_result(next_step_id="nowhere")

        wf = create_workflow("wf", "wf")
        wf.with_initial_step(_Stray(step_id="s", name="S"))
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start("wf")
        instance = engine.workflow_instances[iid]
        assert instance.status == WorkflowStatus.FAILED
        assert "Invalid next state" in (instance.error or "")


class TestDeadCodeRemoved:
    """F-010: conversation_map should not exist on WorkflowEngine."""

    def test_no_conversation_map(self):
        """WorkflowEngine should not have conversation_map attribute."""
        engine = WorkflowEngine()
        assert not hasattr(engine, "conversation_map")


class TestInstanceLockConcurrency:
    """F4: concurrent advance/cancel calls on the SAME
    instance must not corrupt current_step_id/status/context, and must not
    execute the same step body twice for a single logical advance.

    Regression test for plan-2026-09-12T065608-089d0ec7 step 2 (D-003).
    """

    @staticmethod
    def _build_counting_terminal_step(step_id: str, counter: list):
        """A step that sleeps briefly (to widen the race window), then
        completes with no next_step_id (terminal), incrementing ``counter``
        each time it actually runs."""
        import asyncio as _asyncio

        from fsm_llm.workflows.models import WorkflowStepResult
        from fsm_llm.workflows.steps import WorkflowStep

        class _CountingTerminalStep(WorkflowStep):
            async def execute(self, context):
                counter.append(1)
                await _asyncio.sleep(0.05)
                return WorkflowStepResult.success_result(
                    data={"ran": True}, next_step_id=None, message="done"
                )

        return _CountingTerminalStep(step_id=step_id, name="Counting")

    @staticmethod
    def _build_counting_waiting_step(step_id: str, counter: list):
        """A step that sleeps briefly, then parks the workflow WAITING for
        an event (not terminal), incrementing ``counter`` each run."""
        import asyncio as _asyncio

        from fsm_llm.workflows.models import WorkflowStepResult
        from fsm_llm.workflows.steps import WorkflowStep

        class _CountingWaitingStep(WorkflowStep):
            async def execute(self, context):
                counter.append(1)
                await _asyncio.sleep(0.05)
                return WorkflowStepResult.success_result(
                    data={
                        "_waiting_info": {
                            "waiting_for_event": True,
                            "event_type": "resume",
                        }
                    },
                    next_step_id=None,
                    message="waiting",
                )

        return _CountingWaitingStep(step_id=step_id, name="CountingWaiting")

    async def test_concurrent_advance_calls_do_not_double_execute_step(self):
        """Two concurrent advance() calls on the same RUNNING
        instance must only ever execute the step body once: whichever call
        loses the lock race must observe the (by-then) terminal instance via
        is_active() and no-op, rather than re-running the step concurrently."""
        from fsm_llm.workflows.definitions import WorkflowDefinition
        from fsm_llm.workflows.models import WorkflowInstance, WorkflowStatus

        for _ in range(5):
            counter: list = []
            engine = WorkflowEngine()
            step = self._build_counting_terminal_step("slow", counter)
            definition = WorkflowDefinition(
                workflow_id="wf-race-advance",
                name="Race",
                steps={"slow": step},
                initial_step_id="slow",
            )
            engine.register_workflow(definition)

            instance = WorkflowInstance(
                instance_id="race-advance-1",
                workflow_id="wf-race-advance",
                current_step_id="slow",
                status=WorkflowStatus.RUNNING,
            )
            engine.workflow_instances["race-advance-1"] = instance

            results = await asyncio.gather(
                engine.advance("race-advance-1"),
                engine.advance("race-advance-1"),
            )

            # Exactly one step execution, regardless of which call won the
            # per-instance lock race.
            assert len(counter) == 1
            assert sorted(results) == [False, True]
            assert instance.status == WorkflowStatus.COMPLETED
            assert instance.current_step_id == "slow"
            assert instance.context.get("ran") is True

    async def test_concurrent_advance_and_cancel_leave_consistent_state(self):
        """A concurrent advance() + cancel() on the same
        instance must always end CANCELLED, must never execute the step body
        more than once, and must never raise (WAITING->CANCELLED is always a
        valid transition, so no ordering of the two calls can hit an invalid
        status transition)."""
        from fsm_llm.workflows.definitions import WorkflowDefinition
        from fsm_llm.workflows.models import WorkflowInstance, WorkflowStatus

        for _ in range(5):
            counter: list = []
            engine = WorkflowEngine()
            step = self._build_counting_waiting_step("slow", counter)
            definition = WorkflowDefinition(
                workflow_id="wf-race-cancel",
                name="Race",
                steps={"slow": step},
                initial_step_id="slow",
            )
            engine.register_workflow(definition)

            instance = WorkflowInstance(
                instance_id="race-cancel-1",
                workflow_id="wf-race-cancel",
                current_step_id="slow",
                status=WorkflowStatus.RUNNING,
            )
            engine.workflow_instances["race-cancel-1"] = instance

            _advance_result, cancel_result = await asyncio.gather(
                engine.advance("race-cancel-1"),
                engine.cancel("race-cancel-1"),
            )

            # The step never ran more than once no matter the interleaving.
            assert len(counter) <= 1
            # cancel always succeeds: RUNNING->CANCELLED and
            # WAITING->CANCELLED are both valid transitions.
            assert cancel_result is True
            assert instance.status == WorkflowStatus.CANCELLED
            assert instance.current_step_id == "slow"
            assert instance.context.get("_cancellation_reason") == "Cancelled by user"

    def test_instance_lock_removed_on_remove_instance(self):
        """F4 cleanup: remove_instance must also drop the per-instance lock
        so _instance_locks does not grow unbounded."""
        from fsm_llm.workflows.models import WorkflowInstance, WorkflowStatus

        engine = WorkflowEngine()
        instance = WorkflowInstance(
            instance_id="done-1",
            workflow_id="wf-1",
            current_step_id="done",
            status=WorkflowStatus.COMPLETED,
        )
        engine.workflow_instances["done-1"] = instance
        engine._get_instance_lock("done-1")
        assert "done-1" in engine._instance_locks

        assert engine.remove_instance("done-1") is True
        assert "done-1" not in engine._instance_locks

    def test_instance_lock_removed_on_purge(self):
        """F4 cleanup: _purge_oldest_terminal_instances must also drop the
        per-instance locks of purged instances."""
        from datetime import datetime

        from fsm_llm.workflows.models import WorkflowInstance, WorkflowStatus

        engine = WorkflowEngine(max_completed_instances=1)
        for i in range(3):
            iid = f"purge-{i}"
            engine.workflow_instances[iid] = WorkflowInstance(
                instance_id=iid,
                workflow_id="wf-1",
                current_step_id="done",
                status=WorkflowStatus.COMPLETED,
                completed_at=datetime(2026, 1, 1 + i, tzinfo=UTC),
                updated_at=datetime(2026, 1, 1 + i, tzinfo=UTC),
            )
            engine._get_instance_lock(iid)

        engine._purge_oldest_terminal_instances()

        assert "purge-0" not in engine._instance_locks
        assert "purge-1" not in engine._instance_locks
        assert "purge-2" in engine._instance_locks


class TestHandleStepExceptionTerminalGuard:
    """F5: an exception raised while the instance is already terminal
    (CANCELLED/COMPLETED) must not be masked by a WorkflowStateError raised
    from inside the engine's own update_status(FAILED, ...) call (`_fail`)
    -- the original exception must always propagate unchanged.

    Regression test for plan-2026-09-12T065608-089d0ec7 step 4 (D-005).
    """

    @staticmethod
    def _make_instance(status, workflow_id="wf-terminal"):
        from fsm_llm.workflows.models import WorkflowInstance

        return WorkflowInstance(
            instance_id="terminal-1",
            workflow_id=workflow_id,
            current_step_id="slow",
            status=status,
        )

    async def test_noop_when_instance_already_cancelled(self):
        """Calling _fail on an already-CANCELLED instance
        must not raise WorkflowStateError and must not overwrite the status."""
        from fsm_llm.workflows.models import WorkflowStatus

        engine = WorkflowEngine()
        instance = self._make_instance(WorkflowStatus.CANCELLED)

        # Before the fix, this line raised WorkflowStateError from inside
        # update_status (CANCELLED -> FAILED is not a valid transition).
        engine._fail(instance, ValueError("boom"))

        assert instance.status == WorkflowStatus.CANCELLED

    async def test_noop_when_instance_already_completed(self):
        """Same guard, COMPLETED variant."""
        from fsm_llm.workflows.models import WorkflowStatus

        engine = WorkflowEngine()
        instance = self._make_instance(WorkflowStatus.COMPLETED)

        engine._fail(instance, ValueError("boom"))

        assert instance.status == WorkflowStatus.COMPLETED

    async def test_original_exception_surfaces_not_workflow_state_error(self):
        """_execute_workflow_step's except block must propagate the ORIGINAL
        exception even when the instance is already terminal, not the
        WorkflowStateError that update_status(FAILED, ...) would have raised
        pre-fix. Uses WorkflowTimeoutError since it is the one exception type
        _execute_workflow_step re-raises unconditionally (pre-existing,
        out-of-scope non-timeout-exception swallowing is unaffected by this
        fix)."""
        from fsm_llm.workflows.definitions import WorkflowDefinition
        from fsm_llm.workflows.models import WorkflowStatus
        from fsm_llm.workflows.steps import WorkflowStep

        class _RaisingStep(WorkflowStep):
            async def execute(self, context):
                raise WorkflowTimeoutError(
                    "boom", operation="step boom", timeout_seconds=1.0
                )

        engine = WorkflowEngine()
        step = _RaisingStep(step_id="slow", name="Raising")
        definition = WorkflowDefinition(
            workflow_id="wf-terminal-exc",
            name="TerminalExc",
            steps={"slow": step},
            initial_step_id="slow",
        )
        engine.register_workflow(definition)

        instance = self._make_instance(
            WorkflowStatus.CANCELLED, workflow_id="wf-terminal-exc"
        )
        engine.workflow_instances["terminal-1"] = instance

        with pytest.raises(WorkflowTimeoutError):
            await engine._execute_workflow_step(instance)

        # Status must remain CANCELLED (not overwritten to FAILED, and
        # certainly not masked by a WorkflowStateError).
        assert instance.status == WorkflowStatus.CANCELLED


class TestCancelWorkflowTerminalGuard:
    """F5 completion-fix (step 4.1): cancel's own update_status
    call must not raise WorkflowStateError when the instance is already
    terminal by the time cancel acquires the per-instance lock --
    the same terminal-transition hazard D-005 fixed for
    _handle_step_exception, but left unguarded on cancel itself.
    Reproduced by the iter-1 REFLECT reviewer as
    `[True, WorkflowStateError("... completed -> cancelled")]`
    (findings/review-iter-1.md concern 3).

    Regression test for plan-2026-09-12T065608-089d0ec7 step 4.1 (D-015).
    """

    @staticmethod
    def _make_instance(status, workflow_id="wf-cancel-terminal"):
        from fsm_llm.workflows.models import WorkflowInstance

        return WorkflowInstance(
            instance_id="cancel-terminal-1",
            workflow_id=workflow_id,
            current_step_id="slow",
            status=status,
        )

    async def test_cancel_already_completed_instance_returns_false_no_raise(self):
        from fsm_llm.workflows.models import WorkflowStatus

        engine = WorkflowEngine()
        instance = self._make_instance(WorkflowStatus.COMPLETED)
        engine.workflow_instances[instance.instance_id] = instance

        # Before the fix, this raised WorkflowStateError from inside
        # update_status (COMPLETED -> CANCELLED is not a valid transition).
        result = await engine.cancel(instance.instance_id)

        assert result is False
        assert instance.status == WorkflowStatus.COMPLETED

    async def test_cancel_already_cancelled_instance_returns_false_no_raise(self):
        from fsm_llm.workflows.models import WorkflowStatus

        engine = WorkflowEngine()
        instance = self._make_instance(WorkflowStatus.CANCELLED)
        engine.workflow_instances[instance.instance_id] = instance

        result = await engine.cancel(instance.instance_id)

        assert result is False
        assert instance.status == WorkflowStatus.CANCELLED

    async def test_cancel_already_failed_instance_returns_false_no_raise(self):
        from fsm_llm.workflows.models import WorkflowStatus

        engine = WorkflowEngine()
        instance = self._make_instance(WorkflowStatus.FAILED)
        engine.workflow_instances[instance.instance_id] = instance

        result = await engine.cancel(instance.instance_id)

        assert result is False
        assert instance.status == WorkflowStatus.FAILED

    async def test_concurrent_advance_completes_and_cancel_races_the_lock(self):
        """Direct reproduction of the reviewer's finding: a step that
        completes the instance (terminal, non-WAITING) races a concurrent
        cancel() waiting on the same per-instance lock. Neither call
        may ever raise, regardless of which one wins the lock race:

        - if cancel wins first, RUNNING->CANCELLED is a valid
          transition, so it succeeds normally (cancel_result True, counter
          stays 0, advance's own is_active() guard then makes it a
          clean no-op returning False);
        - if advance wins first, it runs the step to completion
          (COMPLETED) before cancel gets the lock, and this fix's
          terminal guard makes cancel a no-op returning False
          instead of raising WorkflowStateError (the reviewer's exact
          repro).
        """
        from fsm_llm.workflows.definitions import WorkflowDefinition
        from fsm_llm.workflows.models import WorkflowInstance, WorkflowStatus

        for _ in range(20):
            counter: list = []
            engine = WorkflowEngine()
            step = TestInstanceLockConcurrency._build_counting_terminal_step(
                "slow", counter
            )
            definition = WorkflowDefinition(
                workflow_id="wf-cancel-vs-complete",
                name="CancelVsComplete",
                steps={"slow": step},
                initial_step_id="slow",
            )
            engine.register_workflow(definition)

            instance = WorkflowInstance(
                instance_id="cancel-vs-complete-1",
                workflow_id="wf-cancel-vs-complete",
                current_step_id="slow",
                status=WorkflowStatus.RUNNING,
            )
            engine.workflow_instances["cancel-vs-complete-1"] = instance

            advance_result, cancel_result = await asyncio.gather(
                engine.advance("cancel-vs-complete-1"),
                engine.cancel("cancel-vs-complete-1"),
                return_exceptions=True,
            )

            # Neither call may raise -- this is the exact hazard the
            # reviewer reported ([True, WorkflowStateError(...)]).
            assert not isinstance(advance_result, BaseException), advance_result
            assert not isinstance(cancel_result, BaseException), cancel_result
            assert len(counter) <= 1
            assert instance.status in (
                WorkflowStatus.COMPLETED,
                WorkflowStatus.CANCELLED,
            )
            if instance.status == WorkflowStatus.COMPLETED:
                # advance won the lock race and ran the step first.
                assert cancel_result is False
                assert len(counter) == 1
            else:
                # cancel won the lock race before the step ran.
                assert cancel_result is True
                assert len(counter) == 0


class TestFloatTimeoutSecondsEndToEnd:
    """F7 regression: a sub-second max_seconds must report its real
    float value (e.g. 0.5) in the raised WorkflowTimeoutError, not `0` from
    a stale `int(...)` truncation in engine/driver.py's `_timeout_seconds()`
    helper. See decisions.md D-011."""

    async def test_expired_deadline_reports_subsecond_float_timeout(self):
        from datetime import datetime, timedelta

        from fsm_llm.workflows.definitions import WorkflowDefinition
        from fsm_llm.workflows.models import WorkflowInstance, WorkflowStatus
        from fsm_llm.workflows.steps import AutoStep, WorkflowStep

        class _TerminalStep(WorkflowStep):
            async def execute(self, context):
                return None

        engine = WorkflowEngine()
        step = AutoStep(step_id="only", name="Only", next_step_id="term")
        term_step = _TerminalStep(step_id="term", name="Term")
        definition = WorkflowDefinition(
            workflow_id="wf-float-timeout",
            name="FloatTimeout",
            steps={"only": step, "term": term_step},
            initial_step_id="only",
        )
        engine.register_workflow(definition)

        instance = WorkflowInstance(
            instance_id="float-timeout-1",
            workflow_id="wf-float-timeout",
            current_step_id="only",
            status=WorkflowStatus.RUNNING,
            max_seconds=0.5,
            deadline=datetime.now(UTC) - timedelta(seconds=1),
        )
        engine.workflow_instances[instance.instance_id] = instance

        with pytest.raises(WorkflowTimeoutError) as exc_info:
            await engine._execute_workflow_step(instance)

        assert exc_info.value.timeout_seconds == 0.5
        assert "0.5 seconds" in str(exc_info.value)


class TestCancelMidChain:
    """F-WE-01 (plan 5ed71cf2 step 15, D-017): the entry point holds the
    instance lock across a whole chain of back-to-back steps, so a cancel
    used to wait for the end of the run, find COMPLETED and return False.
    The cancel now registers a request before awaiting the lock; the chain
    loop honours it between steps. Gates are events, not sleeps, so the
    ordering does not depend on timing.
    """

    @staticmethod
    def _build(chain: list[str], ran: list[str], gates: dict[str, asyncio.Event]):
        """A chain ``chain[0] -> chain[1] -> ...`` of gated async steps.

        Each step appends its id to ``ran``, signals ``<id>_started`` and
        waits on its gate (when it has one) before routing on.
        """
        from fsm_llm.workflows.definitions import WorkflowDefinition
        from fsm_llm.workflows.models import WorkflowStepResult
        from fsm_llm.workflows.steps import WorkflowStep

        class _GatedStep(WorkflowStep):
            next_step_id: str = ""

            async def execute(self, context):
                ran.append(self.step_id)
                started = gates.get(f"{self.step_id}_started")
                if started is not None:
                    started.set()
                gate = gates.get(self.step_id)
                if gate is not None:
                    await gate.wait()
                return WorkflowStepResult.success_result(
                    data={},
                    next_step_id=self.next_step_id or None,
                    message=self.step_id,
                )

        steps = {}
        for i, sid in enumerate(chain):
            nxt = chain[i + 1] if i + 1 < len(chain) else ""
            steps[sid] = _GatedStep(step_id=sid, name=sid, next_step_id=nxt)
        return WorkflowDefinition(
            workflow_id="wf-cancel-chain",
            name="CancelChain",
            steps=steps,
            initial_step_id=chain[0],
        )

    async def test_cancel_between_steps_cancels_and_returns_true(self):
        from fsm_llm.workflows.constants import CONTEXT_KEY_CANCELLATION_REASON
        from fsm_llm.workflows.models import WorkflowStatus

        ran: list[str] = []
        gates = {"a": asyncio.Event(), "a_started": asyncio.Event()}
        engine = WorkflowEngine()
        engine.register_workflow(self._build(["a", "b", "c"], ran, gates))

        iid = await engine.start("wf-cancel-chain", wait=False)
        await asyncio.wait_for(gates["a_started"].wait(), 5)
        # Step a is in flight; the chain holds the instance lock.
        cancel = asyncio.ensure_future(engine.cancel(iid, "stop now"))
        await asyncio.sleep(0.02)  # let cancel register and block on the lock
        gates["a"].set()

        assert await asyncio.wait_for(cancel, 5) is True
        instance = engine.workflow_instances[iid]
        assert instance.status == WorkflowStatus.CANCELLED
        assert instance.context[CONTEXT_KEY_CANCELLATION_REASON] == "stop now"
        assert ran == ["a"]  # b and c never ran
        assert engine._cancel_requests == {}
        await asyncio.gather(*engine._background_tasks)

    async def test_cancel_racing_a_chain_that_finishes_first_returns_false(self):
        from fsm_llm.workflows.models import WorkflowStatus

        ran: list[str] = []
        gates = {"a": asyncio.Event(), "a_started": asyncio.Event()}
        engine = WorkflowEngine()
        engine.register_workflow(self._build(["a"], ran, gates))

        iid = await engine.start("wf-cancel-chain", wait=False)
        await asyncio.wait_for(gates["a_started"].wait(), 5)
        cancel = asyncio.ensure_future(engine.cancel(iid))
        await asyncio.sleep(0.02)
        gates["a"].set()  # the only step ends the workflow; no later iteration

        # D-015: a natural completion before the loop saw the request is
        # not a cancel that took effect.
        assert await asyncio.wait_for(cancel, 5) is False
        assert engine.workflow_instances[iid].status == WorkflowStatus.COMPLETED
        assert engine._cancel_requests == {}

    async def test_cancel_of_completed_instance_returns_false(self):
        from fsm_llm.workflows.models import WorkflowStatus

        ran: list[str] = []
        engine = WorkflowEngine()
        engine.register_workflow(self._build(["a", "b"], ran, {}))

        iid = await engine.start("wf-cancel-chain", wait=True)
        assert engine.workflow_instances[iid].status == WorkflowStatus.COMPLETED

        assert await engine.cancel(iid) is False
        assert engine.workflow_instances[iid].status == WorkflowStatus.COMPLETED
        assert engine._cancel_requests == {}

    async def test_cancel_of_unknown_instance_leaves_no_request(self):
        engine = WorkflowEngine()
        assert await engine.cancel("nope") is False
        assert engine._cancel_requests == {}

    async def test_cancel_of_waiting_instance_still_works(self):
        from fsm_llm.workflows.constants import CONTEXT_KEY_CANCELLATION_REASON
        from fsm_llm.workflows.definitions import WorkflowDefinition
        from fsm_llm.workflows.dsl import wait_event_step
        from fsm_llm.workflows.models import WorkflowStatus

        engine = WorkflowEngine()
        engine.register_workflow(
            WorkflowDefinition(
                workflow_id="wf-cancel-wait",
                name="CancelWait",
                steps={"w": wait_event_step("w", "w", "go", "")},
                initial_step_id="w",
            )
        )
        iid = await engine.start("wf-cancel-wait")
        assert engine.workflow_instances[iid].status == WorkflowStatus.WAITING

        assert await engine.cancel(iid, "bye") is True
        instance = engine.workflow_instances[iid]
        assert instance.status == WorkflowStatus.CANCELLED
        assert instance.context[CONTEXT_KEY_CANCELLATION_REASON] == "bye"
        assert engine._cancel_requests == {}


_ENGINE_MIXIN_MODULE_OF = {
    "StatusMixin": "status",
    "LifecycleMixin": "lifecycle",
    "DriverMixin": "driver",
    "EventsMixin": "events",
    "TimersMixin": "timers",
    "ResourcesMixin": "resources",
    "QueriesMixin": "queries",
}

_ENGINE_METHODS_OF = {
    "status": {
        "add_hook",
        "remove_hook",
        "_emit",
        "_get_instance_lock",
        "_set_status",
        "_fail",
    },
    "lifecycle": {
        "close",
        "register_workflow",
        "start",
        "_run_in_background",
        "_get_workflow_definition",
        "_definition_for",
        "_create_workflow_instance",
        "advance",
        "cancel",
    },
    "driver": {
        "_execute_workflow_step",
        "_context_stops",
        "_run_current_step",
        "_timeout_seconds",
        "_get_current_step",
        "_handle_successful_step",
        "_handle_failed_step",
        "_prepare_transition",
        "_leave_wait",
        "_transition_to_state",
        "_handle_step_without_transition",
    },
    "events": {
        "register_event_listener",
        "_handle_event_timeout",
        "process_event",
        "_restore_listeners",
        "_deliver_event",
        "_correlates",
        "_apply_event",
        "_buffer_event",
        "_take_buffered_event",
        "_cancel_event_timeout",
    },
    "timers": {
        "_live_instance",
        "_arm_timer",
        "_forget_timer",
        "schedule_timer",
        "_handle_timer_expiration",
        "_arm_deadline",
        "_handle_deadline",
    },
    "resources": {
        "_drop_wait_resources",
        "_release_instance_resources",
        "remove_instance",
        "_forget_instance",
        "_purge_oldest_terminal_instances",
    },
    "queries": {
        "get_instance",
        "get_definition",
        "get_status",
        "get_context",
        "list_active_instances",
        "get_statistics",
    },
}


class TestWorkflowEnginePackageLayout:
    """``workflows/engine.py`` is a package at the same dotted path (step 91 of
    plan d454fd62): ``WorkflowEngine`` is a shell over seven mixins."""

    def test_the_facade_exports_exactly_the_four_names(self):
        import fsm_llm.workflows.engine as facade
        from fsm_llm.workflows import Timer
        from fsm_llm.workflows.constants import DEFAULT_MAX_STEPS

        assert facade.__all__ == [
            "DEFAULT_MAX_STEPS",
            "LifecycleHook",
            "Timer",
            "WorkflowEngine",
        ]
        assert facade.WorkflowEngine is WorkflowEngine
        assert facade.Timer is Timer
        assert facade.DEFAULT_MAX_STEPS == DEFAULT_MAX_STEPS == 1000
        namespace: dict = {}
        exec("from fsm_llm.workflows.engine import *", namespace)
        assert {"WorkflowEngine", "Timer", "LifecycleHook"} <= set(namespace)
        assert "_timer_key" not in namespace

    def test_the_engine_is_a_shell_over_seven_mixins_each_in_its_module(self):
        mro = WorkflowEngine.__mro__
        assert WorkflowEngine.__module__ == "fsm_llm.workflows.engine.engine"
        mixins = mro[1:-2]
        assert [c.__name__ for c in mixins] == list(_ENGINE_MIXIN_MODULE_OF)
        for cls in mixins:
            module = _ENGINE_MIXIN_MODULE_OF[cls.__name__]
            assert cls.__module__ == f"fsm_llm.workflows.engine.{module}", cls.__name__
        surface = mro[-2]
        assert surface.__name__ == "_EngineSurface"
        # the declarations are for mypy only: a runtime no-op
        assert [n for n in vars(surface) if not n.startswith("__")] == []

    def test_the_shell_defines_only_its_constructor(self):
        assert {n for n in vars(WorkflowEngine) if not n.startswith("__")} == set()
        assert "__init__" in vars(WorkflowEngine)

    def test_every_method_is_defined_by_the_mixin_that_owns_it(self):
        import importlib

        for module, names in _ENGINE_METHODS_OF.items():
            mixin = next(
                getattr(
                    importlib.import_module(f"fsm_llm.workflows.engine.{module}"), c
                )
                for c, m in _ENGINE_MIXIN_MODULE_OF.items()
                if m == module
            )
            assert {n for n in vars(mixin) if not n.startswith("__")} == names, module

    def test_a_private_name_is_imported_from_its_module_not_the_facade(self):
        import fsm_llm.workflows.engine as facade
        from fsm_llm.workflows.engine import helpers

        assert helpers._timer_key("i1", "timeout", "e1") == "i1_e1_timeout"
        assert helpers._timer_key("i1", "deadline") == "i1_deadline"
        assert helpers._timeout_error("op", 0.5, "i1").timeout_seconds == 0.5
        assert helpers._CancelRequest("why").honored is False
        for name in (
            "_timer_key",
            "_timeout_error",
            "_CancelRequest",
            "logger",
            "asyncio",
        ):
            assert not hasattr(facade, name), name

    def test_every_decision_anchor_sits_with_the_code_it_constrains(self):
        from pathlib import Path

        import fsm_llm.workflows.engine as facade

        root = Path(facade.__file__).parent
        expected = {
            "driver": (
                "5d1e7a3b/D-002",
                "5ed71cf2/D-017",
                "089d0ec7/D-011",
                "876e7164/D-003",
                "45a654de/D-013",
                "45a654de/D-020",
            ),
            "events": ("5d1e7a3b/D-001", "5b2fbb09/D-001"),
            "lifecycle": ("5d1e7a3b/D-005", "089d0ec7/D-015"),
            "status": ("089d0ec7/D-003", "089d0ec7/D-005"),
            "resources": ("5d1e7a3b/D-005",),
            "timers": ("5ed71cf2/D-002",),
        }
        for module, tokens in expected.items():
            text = (root / f"{module}.py").read_text(encoding="utf-8")
            for token in tokens:
                assert token in text, (module, token)

    async def test_a_log_record_is_named_after_the_module_that_logs_it(self):
        from loguru import logger

        records: list[tuple[str, str]] = []
        logger.enable("fsm_llm")
        sink = logger.add(
            lambda m: records.append((m.record["name"], m.record["message"])),
            level="DEBUG",
            format="{message}",
        )
        try:
            engine = WorkflowEngine()
            engine.register_workflow(
                create_workflow("wf-names", "Names").with_initial_step(
                    auto_step("a", "A", next_step_id="")
                )
            )
            iid = await engine.start("wf-names")
            assert engine.remove_instance(iid) is True
        finally:
            logger.remove(sink)
            logger.disable("fsm_llm")

        names = {name for name, _ in records}
        assert names == {
            "fsm_llm.workflows.engine.engine",
            "fsm_llm.workflows.engine.lifecycle",
            "fsm_llm.workflows.engine.driver",
            "fsm_llm.workflows.engine.resources",
        }

        def logged_by(prefix: str) -> str:
            return next(n for n, message in records if message.startswith(prefix))

        assert logged_by("Workflow engine initialized").endswith(".engine.engine")
        assert logged_by("Started workflow instance").endswith(".lifecycle")
        assert logged_by("Executing step").endswith(".driver")
        assert logged_by("Removed terminal instance").endswith(".resources")


class _RunsOnceStep(WorkflowStep):
    """Counts its runs, routes to ``b`` in its definition but returns no route:
    a non-terminal step with no transition, which leaves the instance RUNNING."""

    next_step_id: str = "b"
    runs: int = 0

    async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
        self.runs += 1
        return WorkflowStepResult(success=True)


class _FailsWithErrorStep(WorkflowStep):
    async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
        return WorkflowStepResult.failure_result(error="E!", message="M")


class TestEngineEdgesOfEachModule:
    """Edges of the engine modules that the step 91 mutation proof found no suite test
    for (each was seen only by the differential, or by nothing): the exact limits, the
    emission rules of the hooks, the guards under the lock and the buffers."""

    async def test_a_hook_is_told_only_about_a_real_status_change(self):
        engine = WorkflowEngine()
        engine.register_workflow(_wait_then_done("wf"))
        iid = await engine.start("wf")
        instance = engine.workflow_instances[iid]
        seen: list[tuple[str, str]] = []
        engine.add_hook(lambda ev, inst, data: seen.append((ev, data["status"])))

        engine._set_status(instance, WorkflowStatus.WAITING)  # unchanged
        assert seen == []
        engine._set_status(instance, WorkflowStatus.RUNNING)
        assert seen == [("status_changed", "running")]
        await engine.close()

    async def test_an_awaitable_hook_result_runs_as_a_background_task(self):
        engine = WorkflowEngine()
        engine.register_workflow(_single_auto("wf"))
        events: list[str] = []

        async def hook(event: str, instance: Any, data: dict[str, Any]) -> None:
            events.append(event)

        engine.add_hook(hook)
        await engine.start("wf")
        assert engine._background_tasks  # scheduled, not awaited inline
        await _settle(0.05)
        assert events == [
            "status_changed",
            "step_started",
            "step_completed",
            "status_changed",
        ]
        assert engine._background_tasks == set()

    def test_remove_hook_reports_whether_the_hook_was_registered(self):
        engine = WorkflowEngine()

        def hook(*args: Any) -> None:
            return None

        engine.add_hook(hook)
        assert engine.remove_hook(hook) is True
        assert engine.remove_hook(hook) is False
        assert engine._hooks == []

    async def test_the_concurrency_limit_is_exact(self):
        engine = WorkflowEngine(max_concurrent_workflows=2)
        engine.register_workflow(_wait_then_done("wf"))
        await engine.start("wf", instance_id="a")
        await engine.start("wf", instance_id="b")
        with pytest.raises(WorkflowResourceError) as info:
            await engine.start("wf", instance_id="c")
        assert info.value.resource_id == "concurrent_limit"
        assert sorted(engine.list_active_instances()) == ["a", "b"]
        await engine.close()

    async def test_a_background_start_skips_an_instance_another_entry_point_drove(self):
        step = _RunsOnceStep(step_id="a", name="A")
        wf = create_workflow("wf", "wf")
        wf.with_initial_step(step)
        wf.with_step(auto_step("b", "B", next_step_id=""))
        engine = WorkflowEngine()
        engine.register_workflow(wf)

        iid = await engine.start("wf", wait=False)
        assert await engine.advance(iid) is True  # drives the step before the task runs
        await _settle(0.05)

        assert step.runs == 1  # the background task saw the history move and returned
        assert engine.get_status(iid) == WorkflowStatus.RUNNING

    async def test_the_initial_context_dict_of_the_caller_is_never_shared(self):
        engine = WorkflowEngine()
        engine.register_workflow(_wait_then_done("wf"))
        initial: dict[str, Any] = {"k": 1}
        iid = await engine.start("wf", initial)

        assert initial == {"k": 1}  # no `_workflow_info` written into it
        engine.workflow_instances[iid].context["added"] = 2
        assert "added" not in initial
        await engine.close()

    async def test_close_drops_buffered_events(self):
        engine = WorkflowEngine()
        engine.register_workflow(_wait_then_done("wf"))
        iid = await engine.start("wf")
        woke = await engine.process_event(
            WorkflowEvent(event_type="other", payload={}, instance_id=iid)
        )
        assert woke == []
        assert engine.get_statistics()["buffered_events"] == 1

        await engine.close()
        assert engine.get_statistics()["buffered_events"] == 0

    async def test_the_step_budget_is_exactly_max_steps_per_call(self):
        def chain(workflow_id: str, length: int):
            wf = create_workflow(workflow_id, workflow_id)
            for i in range(length):
                nxt = f"s{i + 1}" if i < length - 1 else ""
                wf.with_step(
                    auto_step(f"s{i}", f"S{i}", next_step_id=nxt), is_initial=i == 0
                )
            return wf

        engine = WorkflowEngine(max_steps=3)
        engine.register_workflow(chain("fits", 3))
        engine.register_workflow(chain("over", 4))
        fits = await engine.start("fits")
        over = await engine.start("over")

        assert engine.get_status(fits) == WorkflowStatus.COMPLETED
        assert engine.get_status(over) == WorkflowStatus.FAILED
        assert "Step budget exceeded: 3 steps" in str(
            engine.workflow_instances[over].error
        )

    async def test_a_cancel_request_never_cancels_an_instance_that_is_terminal(self):
        from fsm_llm.workflows.engine.helpers import _CancelRequest

        engine = WorkflowEngine()
        engine.register_workflow(_single_auto("wf"))
        iid = await engine.start("wf")
        instance = engine.workflow_instances[iid]
        assert instance.status == WorkflowStatus.COMPLETED
        request = _CancelRequest("late")
        engine._cancel_requests[iid] = request

        await engine._execute_workflow_step(instance)  # must not raise

        assert instance.status == WorkflowStatus.COMPLETED
        assert request.honored is False

    async def test_a_failed_step_with_no_route_reports_message_and_error(self):
        wf = create_workflow("wf", "wf")
        wf.with_initial_step(_FailsWithErrorStep(step_id="s", name="S"))
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start("wf")

        instance = engine.workflow_instances[iid]
        assert instance.status == WorkflowStatus.FAILED
        assert "M (E!)" in str(instance.error)

    async def test_the_step_executor_does_not_leak_into_the_callers_context(self):
        from concurrent.futures import ThreadPoolExecutor

        from fsm_llm.workflows.steps.helpers import _STEP_EXECUTOR

        executor = ThreadPoolExecutor(max_workers=1)
        try:
            engine = WorkflowEngine(executor=executor)
            engine.register_workflow(_single_auto("wf"))
            await engine.start("wf")
            assert _STEP_EXECUTOR.get() is None
        finally:
            executor.shutdown()

    async def test_a_step_that_returns_no_result_object_fails_the_instance(self):
        class _ReturnsText(WorkflowStep):
            async def execute(self, context: dict[str, Any]) -> Any:
                return "done"

        wf = create_workflow("wf", "wf")
        wf.with_initial_step(_ReturnsText(step_id="s", name="S"))
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start("wf")

        instance = engine.workflow_instances[iid]
        assert instance.status == WorkflowStatus.FAILED
        assert "returned str, expected a WorkflowStepResult" in str(instance.error)

    async def test_an_event_is_not_delivered_to_a_listener_whose_timeout_has_passed(
        self,
    ):
        from datetime import datetime, timedelta

        engine = WorkflowEngine()
        engine.register_workflow(_wait_then_done("wf", timeout_seconds=30))
        iid = await engine.start("wf")
        engine.event_listeners["go"][iid].timeout_at = datetime.now(UTC) - timedelta(
            seconds=1
        )

        woke = await engine.process_event(WorkflowEvent(event_type="go", payload={}))

        assert woke == []
        assert engine.get_status(iid) == WorkflowStatus.WAITING
        assert iid in engine.event_listeners["go"]  # skipped, not consumed
        await engine.close()

    async def test_a_delivery_keeps_the_timeout_of_the_next_wait_on_the_same_event(
        self,
    ):
        wf = create_workflow("wf", "wf")
        wf.with_initial_step(
            wait_event_step("w1", "W1", "go", "w2", timeout_seconds=30)
        )
        wf.with_step(wait_event_step("w2", "W2", "go", "done", timeout_seconds=30))
        wf.with_step(auto_step("done", "Done", next_step_id=""))
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start("wf")
        first = engine.timers[f"{iid}_go_timeout"].task

        woke = await engine.process_event(WorkflowEvent(event_type="go", payload={}))

        assert woke == [iid]
        assert engine.workflow_instances[iid].current_step_id == "w2"
        second = engine.timers[f"{iid}_go_timeout"].task
        assert second is not first
        assert not second.cancelled()
        await engine.close()

    async def test_a_delivered_event_is_recorded_in_the_context(self):
        engine = WorkflowEngine()
        engine.register_workflow(_wait_then_done("wf"))
        iid = await engine.start("wf")

        await engine.process_event(
            WorkflowEvent(event_type="go", payload={"a": 1}, instance_id=iid)
        )

        last = engine.get_context(iid)["_last_event"]
        assert last["event_type"] == "go"
        assert last["payload"] == {"a": 1}

    async def test_the_buffer_of_targeted_events_keeps_the_newest(self):
        from fsm_llm.workflows.constants import MAX_BUFFERED_EVENTS_PER_INSTANCE

        engine = WorkflowEngine()
        engine.register_workflow(_wait_then_done("wf"))
        iid = await engine.start("wf")
        limit = MAX_BUFFERED_EVENTS_PER_INSTANCE
        for n in range(limit + 5):
            await engine.process_event(
                WorkflowEvent(event_type="other", payload={"n": n}, instance_id=iid)
            )

        buffer = engine._event_buffers[iid]
        assert len(buffer) == limit
        assert buffer[0].payload["n"] == 5
        assert buffer[-1].payload["n"] == limit + 4
        await engine.close()

    async def test_a_buffered_event_satisfies_one_wait_only(self):
        wf = create_workflow("wf", "wf")
        wf.with_initial_step(wait_event_step("w0", "W0", "start", "w1"))
        wf.with_step(wait_event_step("w1", "W1", "go", "w2"))
        wf.with_step(wait_event_step("w2", "W2", "go", "done"))
        wf.with_step(auto_step("done", "Done", next_step_id=""))
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start("wf")
        await engine.process_event(
            WorkflowEvent(event_type="go", payload={}, instance_id=iid)
        )
        assert len(engine._event_buffers[iid]) == 1

        await engine.process_event(
            WorkflowEvent(event_type="start", payload={}, instance_id=iid)
        )

        # w1 took the buffered event; w2 must wait for another one
        instance = engine.workflow_instances[iid]
        assert instance.status == WorkflowStatus.WAITING
        assert instance.current_step_id == "w2"
        assert len(engine._event_buffers[iid]) == 0
        await engine.close()

    async def test_an_event_timeout_does_nothing_to_an_instance_that_is_not_waiting(
        self,
    ):
        step = _RunsOnceStep(step_id="a", name="A")
        wf = create_workflow("wf", "wf")
        wf.with_initial_step(step)
        wf.with_step(auto_step("b", "B", next_step_id=""))
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start("wf")
        instance = engine.workflow_instances[iid]
        assert instance.status == WorkflowStatus.RUNNING

        await engine._handle_event_timeout(iid, "go", "b")

        assert instance.status == WorkflowStatus.RUNNING
        assert instance.current_step_id == "a"
        assert "_timeout" not in instance.context

    async def test_a_timer_expiry_and_a_deadline_do_nothing_to_an_instance_that_is_not_waiting(
        self,
    ):
        step = _RunsOnceStep(step_id="a", name="A")
        wf = create_workflow("wf", "wf")
        wf.with_initial_step(step)
        wf.with_step(auto_step("b", "B", next_step_id=""))
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start("wf")
        instance = engine.workflow_instances[iid]

        await engine._handle_timer_expiration(iid, "b")
        await engine._handle_deadline(iid)

        assert instance.status == WorkflowStatus.RUNNING
        assert instance.current_step_id == "a"
        assert "_timer_expired" not in instance.context

    async def test_an_event_wait_that_times_out_reports_the_configured_seconds(
        self, timing_factor
    ):
        engine = WorkflowEngine()
        engine.register_workflow(
            _wait_then_done("wf", timeout_seconds=0.05 * timing_factor)
        )
        iid = await engine.start("wf")
        await _settle(0.25 * timing_factor)

        instance = engine.workflow_instances[iid]
        assert instance.status == WorkflowStatus.FAILED
        assert f"timed out after {0.05 * timing_factor} seconds" in str(instance.error)
        assert "wait for event 'go'" in str(instance.error)

    async def test_a_delivered_event_cancels_the_timeout_task_of_its_wait(self):
        engine = WorkflowEngine()
        engine.register_workflow(_wait_then_done("wf", timeout_seconds=30))
        iid = await engine.start("wf")
        task = engine.timers[f"{iid}_go_timeout"].task

        await engine.process_event(WorkflowEvent(event_type="go", payload={}))
        await _settle(0)

        assert task.cancelled()
        assert f"{iid}_go_timeout" not in engine.timers

    async def test_a_targeted_event_for_a_finished_instance_is_dropped_not_buffered(
        self,
    ):
        engine = WorkflowEngine()
        engine.register_workflow(_single_auto("wf"))
        iid = await engine.start("wf")
        assert engine.get_status(iid) == WorkflowStatus.COMPLETED

        woke = await engine.process_event(
            WorkflowEvent(event_type="go", payload={}, instance_id=iid)
        )

        assert woke == []
        assert engine._event_buffers == {}

    async def test_a_listener_of_an_unknown_instance_is_removed_by_process_event(self):
        from fsm_llm.workflows.models import EventListener

        engine = WorkflowEngine()
        engine.event_listeners["go"] = {
            "ghost": EventListener(instance_id="ghost", success_step_id="done")
        }

        woke = await engine.process_event(WorkflowEvent(event_type="go", payload={}))

        assert woke == []
        assert engine.event_listeners["go"] == {}

    async def test_a_timer_task_logs_the_error_of_its_handler_and_swallows_a_cancel(
        self,
    ):
        from loguru import logger

        engine = WorkflowEngine()
        messages: list[str] = []
        logger.enable("fsm_llm")
        sink = logger.add(
            lambda m: messages.append(m.record["message"]),
            level="ERROR",
            format="{message}",
        )

        async def broken() -> None:
            raise RuntimeError("handler broke")

        async def never() -> None:
            await asyncio.sleep(30)

        try:
            engine._arm_timer("k1", "i1", None, 0, broken, "demo")
            await _settle(0.05)
            assert messages == ["Error in demo task: handler broke"]

            engine._arm_timer("k2", "i1", None, 0, never, "never")
            await _settle(0.01)
            engine.timers["k2"].task.cancel()
            await _settle(0.01)
            assert messages == ["Error in demo task: handler broke"]
        finally:
            logger.remove(sink)
            logger.disable("fsm_llm")
            await engine.close()

    async def test_a_deadline_already_past_arms_a_watchdog_that_fires_at_once(self):
        from datetime import datetime, timedelta

        engine = WorkflowEngine()
        engine.register_workflow(_wait_then_done("wf"))
        iid = await engine.start("wf")
        instance = engine.workflow_instances[iid]
        instance.deadline = datetime.now(UTC) - timedelta(seconds=5)
        before = datetime.now(UTC)

        engine._arm_deadline(instance)

        assert engine.timers[f"{iid}_deadline"].expires_at >= before
        await engine.close()

    async def test_a_deadline_that_fires_on_a_running_instance_leaves_no_timer_entry(
        self, timing_factor
    ):
        step = _RunsOnceStep(step_id="a", name="A")
        wf = create_workflow("wf", "wf")
        wf.with_initial_step(step)
        wf.with_step(auto_step("b", "B", next_step_id=""))
        engine = WorkflowEngine()
        engine.register_workflow(wf)
        iid = await engine.start("wf", max_seconds=0.05 * timing_factor)
        assert f"{iid}_deadline" in engine.timers

        await _settle(0.25 * timing_factor)

        assert engine.get_status(iid) == WorkflowStatus.RUNNING
        assert f"{iid}_deadline" not in engine.timers
        await engine.close()

    async def test_forgetting_an_instance_drops_everything_the_engine_keeps_for_it(
        self,
    ):
        from fsm_llm.workflows.engine.helpers import _CancelRequest

        engine = WorkflowEngine()
        engine.register_workflow(_single_auto("wf"))
        iid = await engine.start("wf")
        engine._get_instance_lock(iid)
        engine._cancel_requests[iid] = _CancelRequest("x")
        assert iid in engine._instance_definitions

        assert engine.remove_instance(iid) is True

        assert iid not in engine.workflow_instances
        assert iid not in engine._instance_locks
        assert iid not in engine._instance_definitions
        assert engine._cancel_requests == {}

    async def test_the_statistics_count_live_timers_and_every_status(self):
        engine = WorkflowEngine()
        engine.register_workflow(_wait_then_done("waits", timeout_seconds=30))
        engine.register_workflow(_single_auto("ends"))
        waiting = await engine.start("waits")
        await engine.start("ends")
        await engine.process_event(
            WorkflowEvent(event_type="other", payload={}, instance_id=waiting)
        )

        async def quick() -> None:
            return None

        engine._arm_timer("quick", waiting, None, 0, quick, "quick")
        await _settle(0.02)  # the quick timer's task is done, the event timeout is not

        stats = engine.get_statistics()
        assert stats["total_workflows"] == 2
        assert stats["active_workflows"] == 1
        assert stats["registered_definitions"] == 2
        assert stats["event_listeners"] == 1
        assert stats["active_timers"] == 1
        assert stats["buffered_events"] == 1
        assert stats["status_breakdown"] == {"waiting": 1, "completed": 1}
        await engine.close()

    async def test_a_timer_never_cancels_the_task_that_runs_it(self):
        from datetime import datetime

        from fsm_llm.workflows import Timer

        async def inner() -> int:
            current = asyncio.current_task()
            assert current is not None
            Timer("i", None, datetime.now(UTC), current).cancel()
            return current.cancelling()

        assert await asyncio.ensure_future(inner()) == 0

        other = asyncio.ensure_future(asyncio.sleep(30))
        Timer("i", None, datetime.now(UTC), other).cancel()
        await _settle(0)
        assert other.cancelled()
        Timer("i", None, datetime.now(UTC), None).cancel()  # no task: a no-op

    async def test_a_terminal_instance_releases_its_buffered_events(self):
        engine = WorkflowEngine()
        engine.register_workflow(_wait_then_done("wf"))
        iid = await engine.start("wf")
        await engine.process_event(
            WorkflowEvent(event_type="other", payload={}, instance_id=iid)
        )
        assert iid in engine._event_buffers

        assert await engine.cancel(iid) is True

        assert iid not in engine._event_buffers

    def test_max_steps_of_one_is_accepted_and_below_one_is_refused(self):
        assert WorkflowEngine(max_steps=1).max_steps == 1
        for bad in (0, -3):
            with pytest.raises(ValueError, match="max_steps must be >= 1"):
                WorkflowEngine(max_steps=bad)


def _single_auto(workflow_id: str):
    wf = create_workflow(workflow_id, workflow_id)
    wf.with_initial_step(auto_step("only", "Only", next_step_id=""))
    return wf
