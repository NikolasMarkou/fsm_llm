"""
Tests verifying fixes for audit findings in fsm_llm_workflows.
Covers: F-001 (ParallelStep), F-002 (event race), F-004 (ConversationStep),
        F-007 (LLMProcessingStep template), F-010 (dead code removal).
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from fsm_llm_workflows.engine import WorkflowEngine
from fsm_llm_workflows.exceptions import WorkflowTimeoutError
from fsm_llm_workflows.models import WorkflowEvent
from fsm_llm_workflows.steps import (
    AutoTransitionStep,
    ConversationStep,
    LLMProcessingStep,
    ParallelStep,
)

# ---------------------------------------------------------------------------
# F-001: ParallelStep must return failure when substeps fail
# ---------------------------------------------------------------------------


class TestParallelStepErrorHandling:
    """F-001: ParallelStep should report failure even without error_state."""

    async def test_parallel_step_fails_when_substeps_fail_no_error_state(self):
        """ParallelStep with no error_state must still return failure if substeps fail."""
        failing_step = AutoTransitionStep(
            step_id="fail",
            name="Failing",
            next_state="done",
            action=lambda ctx: (_ for _ in ()).throw(ValueError("boom")),
        )
        ok_step = AutoTransitionStep(
            step_id="ok",
            name="OK",
            next_state="done",
        )

        parallel = ParallelStep(
            step_id="par",
            name="Parallel",
            steps=[ok_step, failing_step],
            next_state="next",
            error_state=None,  # No error_state configured
        )

        result = await parallel.execute({})

        # Must be failure, NOT success
        assert result.success is False
        assert (
            "error" in result.message.lower() or "error" in (result.error or "").lower()
        )

    async def test_parallel_step_fails_with_error_state(self):
        """ParallelStep with error_state transitions to it on failure."""
        failing_step = AutoTransitionStep(
            step_id="fail",
            name="Failing",
            next_state="done",
            action=lambda ctx: (_ for _ in ()).throw(RuntimeError("fail")),
        )

        parallel = ParallelStep(
            step_id="par",
            name="Parallel",
            steps=[failing_step],
            next_state="next",
            error_state="error_handler",
        )

        result = await parallel.execute({})

        assert result.success is False
        assert result.next_state == "error_handler"

    async def test_parallel_step_succeeds_when_all_ok(self):
        """ParallelStep returns success when all substeps succeed."""
        step_a = AutoTransitionStep(step_id="a", name="A", next_state="done")
        step_b = AutoTransitionStep(step_id="b", name="B", next_state="done")

        parallel = ParallelStep(
            step_id="par",
            name="Parallel",
            steps=[step_a, step_b],
            next_state="next",
        )

        result = await parallel.execute({})

        assert result.success is True
        assert result.next_state == "next"


# ---------------------------------------------------------------------------
# F-002: Event listener race condition — pop instead of del
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# F-004: ConversationStep resource leak
# ---------------------------------------------------------------------------


class TestConversationStepResourceCleanup:
    """F-004: end_conversation must be called even on exception."""

    async def test_end_conversation_called_on_exception(self):
        """Verify end_conversation is called in finally block."""
        step = ConversationStep(
            step_id="conv",
            name="Conv",
            fsm_file="fake.json",
            success_state="done",
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
            with patch("fsm_llm.API", mock_api_class):
                with patch("fsm_llm.api.API", mock_api_class):
                    result = await step.execute({})

        # end_conversation MUST be called despite the exception
        mock_api.end_conversation.assert_called_once_with("conv-1")
        # Result should be a failure
        assert result.success is False


# ---------------------------------------------------------------------------
# F-007: LLMProcessingStep template KeyError
# ---------------------------------------------------------------------------


class TestLLMProcessingStepTemplateError:
    """F-007: Missing template variables should raise WorkflowStepError, not KeyError."""

    async def test_missing_template_variable_raises_step_error(self):
        """Template with {missing_var} should raise WorkflowStepError."""
        mock_llm = MagicMock()

        step = LLMProcessingStep(
            step_id="llm",
            name="LLM Step",
            llm_interface=mock_llm,
            prompt_template="Hello {name}, your order {order_id} is ready",
            context_mapping={"name": "user_name"},  # order_id not mapped
            output_mapping={"result": ".*"},
            next_state="done",
        )

        # Context has user_name but template needs order_id too
        context = {"user_name": "Alice"}

        result = await step.execute(context)
        # Should fail gracefully instead of crashing with raw KeyError
        assert result.success is False
        assert "order_id" in result.error.lower() or "template" in result.error.lower()


# ---------------------------------------------------------------------------
# F-010: Dead conversation_map removed
# ---------------------------------------------------------------------------


class TestDeadCodeRemoved:
    """F-010: conversation_map should not exist on WorkflowEngine."""

    def test_no_conversation_map(self):
        """WorkflowEngine should not have conversation_map attribute."""
        engine = WorkflowEngine()
        assert not hasattr(engine, "conversation_map")


# ---------------------------------------------------------------------------
# F4: Per-instance lock for WorkflowInstance mutation
# ---------------------------------------------------------------------------


class TestInstanceLockConcurrency:
    """F4: concurrent advance_workflow/cancel_workflow calls on the SAME
    instance must not corrupt current_step_id/status/context, and must not
    execute the same step body twice for a single logical advance.

    Regression test for plan-2026-09-12T065608-089d0ec7 step 2 (D-003).
    """

    @staticmethod
    def _build_counting_terminal_step(step_id: str, counter: list):
        """A step that sleeps briefly (to widen the race window), then
        completes with no next_state (terminal), incrementing ``counter``
        each time it actually runs."""
        import asyncio as _asyncio

        from fsm_llm_workflows.models import WorkflowStepResult
        from fsm_llm_workflows.steps import WorkflowStep

        class _CountingTerminalStep(WorkflowStep):
            async def execute(self, context):
                counter.append(1)
                await _asyncio.sleep(0.05)
                return WorkflowStepResult.success_result(
                    data={"ran": True}, next_state=None, message="done"
                )

        return _CountingTerminalStep(step_id=step_id, name="Counting")

    @staticmethod
    def _build_counting_waiting_step(step_id: str, counter: list):
        """A step that sleeps briefly, then parks the workflow WAITING for
        an event (not terminal), incrementing ``counter`` each run."""
        import asyncio as _asyncio

        from fsm_llm_workflows.models import WorkflowStepResult
        from fsm_llm_workflows.steps import WorkflowStep

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
                    next_state=None,
                    message="waiting",
                )

        return _CountingWaitingStep(step_id=step_id, name="CountingWaiting")

    async def test_concurrent_advance_calls_do_not_double_execute_step(self):
        """Two concurrent advance_workflow() calls on the same RUNNING
        instance must only ever execute the step body once: whichever call
        loses the lock race must observe the (by-then) terminal instance via
        is_active() and no-op, rather than re-running the step concurrently."""
        from fsm_llm_workflows.definitions import WorkflowDefinition
        from fsm_llm_workflows.models import WorkflowInstance, WorkflowStatus

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
                engine.advance_workflow("race-advance-1"),
                engine.advance_workflow("race-advance-1"),
            )

            # Exactly one step execution, regardless of which call won the
            # per-instance lock race.
            assert len(counter) == 1
            assert sorted(results) == [False, True]
            assert instance.status == WorkflowStatus.COMPLETED
            assert instance.current_step_id == "slow"
            assert instance.context.get("ran") is True

    async def test_concurrent_advance_and_cancel_leave_consistent_state(self):
        """A concurrent advance_workflow() + cancel_workflow() on the same
        instance must always end CANCELLED, must never execute the step body
        more than once, and must never raise (WAITING->CANCELLED is always a
        valid transition, so no ordering of the two calls can hit an invalid
        status transition)."""
        from fsm_llm_workflows.definitions import WorkflowDefinition
        from fsm_llm_workflows.models import WorkflowInstance, WorkflowStatus

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
                engine.advance_workflow("race-cancel-1"),
                engine.cancel_workflow("race-cancel-1"),
            )

            # The step never ran more than once no matter the interleaving.
            assert len(counter) <= 1
            # cancel_workflow always succeeds: RUNNING->CANCELLED and
            # WAITING->CANCELLED are both valid transitions.
            assert cancel_result is True
            assert instance.status == WorkflowStatus.CANCELLED
            assert instance.current_step_id == "slow"
            assert instance.context.get("_cancellation_reason") == "Cancelled by user"

    def test_instance_lock_removed_on_remove_instance(self):
        """F4 cleanup: remove_instance must also drop the per-instance lock
        so _instance_locks does not grow unbounded."""
        from fsm_llm_workflows.models import WorkflowInstance, WorkflowStatus

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
        from datetime import datetime, timezone

        from fsm_llm_workflows.models import WorkflowInstance, WorkflowStatus

        engine = WorkflowEngine(max_completed_instances=1)
        for i in range(3):
            iid = f"purge-{i}"
            engine.workflow_instances[iid] = WorkflowInstance(
                instance_id=iid,
                workflow_id="wf-1",
                current_step_id="done",
                status=WorkflowStatus.COMPLETED,
                completed_at=datetime(2026, 1, 1 + i, tzinfo=timezone.utc),
                updated_at=datetime(2026, 1, 1 + i, tzinfo=timezone.utc),
            )
            engine._get_instance_lock(iid)

        engine._purge_oldest_terminal_instances()

        assert "purge-0" not in engine._instance_locks
        assert "purge-1" not in engine._instance_locks
        assert "purge-2" in engine._instance_locks


class TestHandleStepExceptionTerminalGuard:
    """F5: an exception raised while the instance is already terminal
    (CANCELLED/COMPLETED) must not be masked by a WorkflowStateError raised
    from inside _handle_step_exception's own update_status(FAILED, ...) call
    -- the original exception must always propagate unchanged.

    Regression test for plan-2026-09-12T065608-089d0ec7 step 4 (D-005).
    """

    @staticmethod
    def _make_instance(status, workflow_id="wf-terminal"):
        from fsm_llm_workflows.models import WorkflowInstance

        return WorkflowInstance(
            instance_id="terminal-1",
            workflow_id=workflow_id,
            current_step_id="slow",
            status=status,
        )

    async def test_noop_when_instance_already_cancelled(self):
        """Calling _handle_step_exception on an already-CANCELLED instance
        must not raise WorkflowStateError and must not overwrite the status."""
        from fsm_llm_workflows.models import WorkflowStatus

        engine = WorkflowEngine()
        instance = self._make_instance(WorkflowStatus.CANCELLED)

        # Before the fix, this line raised WorkflowStateError from inside
        # update_status (CANCELLED -> FAILED is not a valid transition).
        await engine._handle_step_exception(instance, ValueError("boom"))

        assert instance.status == WorkflowStatus.CANCELLED

    async def test_noop_when_instance_already_completed(self):
        """Same guard, COMPLETED variant."""
        from fsm_llm_workflows.models import WorkflowStatus

        engine = WorkflowEngine()
        instance = self._make_instance(WorkflowStatus.COMPLETED)

        await engine._handle_step_exception(instance, ValueError("boom"))

        assert instance.status == WorkflowStatus.COMPLETED

    async def test_original_exception_surfaces_not_workflow_state_error(self):
        """_execute_workflow_step's except block must propagate the ORIGINAL
        exception even when the instance is already terminal, not the
        WorkflowStateError that update_status(FAILED, ...) would have raised
        pre-fix. Uses WorkflowTimeoutError since it is the one exception type
        _execute_workflow_step re-raises unconditionally (pre-existing,
        out-of-scope non-timeout-exception swallowing is unaffected by this
        fix)."""
        from fsm_llm_workflows.definitions import WorkflowDefinition
        from fsm_llm_workflows.models import WorkflowStatus
        from fsm_llm_workflows.steps import WorkflowStep

        class _RaisingStep(WorkflowStep):
            async def execute(self, context):
                raise WorkflowTimeoutError(operation="step boom", timeout_seconds=1.0)

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
    """F5 completion-fix (step 4.1): cancel_workflow's own update_status
    call must not raise WorkflowStateError when the instance is already
    terminal by the time cancel_workflow acquires the per-instance lock --
    the same terminal-transition hazard D-005 fixed for
    _handle_step_exception, but left unguarded on cancel_workflow itself.
    Reproduced by the iter-1 REFLECT reviewer as
    `[True, WorkflowStateError("... completed -> cancelled")]`
    (findings/review-iter-1.md concern 3).

    Regression test for plan-2026-09-12T065608-089d0ec7 step 4.1 (D-015).
    """

    @staticmethod
    def _make_instance(status, workflow_id="wf-cancel-terminal"):
        from fsm_llm_workflows.models import WorkflowInstance

        return WorkflowInstance(
            instance_id="cancel-terminal-1",
            workflow_id=workflow_id,
            current_step_id="slow",
            status=status,
        )

    async def test_cancel_already_completed_instance_returns_false_no_raise(self):
        from fsm_llm_workflows.models import WorkflowStatus

        engine = WorkflowEngine()
        instance = self._make_instance(WorkflowStatus.COMPLETED)
        engine.workflow_instances[instance.instance_id] = instance

        # Before the fix, this raised WorkflowStateError from inside
        # update_status (COMPLETED -> CANCELLED is not a valid transition).
        result = await engine.cancel_workflow(instance.instance_id)

        assert result is False
        assert instance.status == WorkflowStatus.COMPLETED

    async def test_cancel_already_cancelled_instance_returns_false_no_raise(self):
        from fsm_llm_workflows.models import WorkflowStatus

        engine = WorkflowEngine()
        instance = self._make_instance(WorkflowStatus.CANCELLED)
        engine.workflow_instances[instance.instance_id] = instance

        result = await engine.cancel_workflow(instance.instance_id)

        assert result is False
        assert instance.status == WorkflowStatus.CANCELLED

    async def test_cancel_already_failed_instance_returns_false_no_raise(self):
        from fsm_llm_workflows.models import WorkflowStatus

        engine = WorkflowEngine()
        instance = self._make_instance(WorkflowStatus.FAILED)
        engine.workflow_instances[instance.instance_id] = instance

        result = await engine.cancel_workflow(instance.instance_id)

        assert result is False
        assert instance.status == WorkflowStatus.FAILED

    async def test_concurrent_advance_completes_and_cancel_races_the_lock(self):
        """Direct reproduction of the reviewer's finding: a step that
        completes the instance (terminal, non-WAITING) races a concurrent
        cancel_workflow() waiting on the same per-instance lock. Neither call
        may ever raise, regardless of which one wins the lock race:

        - if cancel_workflow wins first, RUNNING->CANCELLED is a valid
          transition, so it succeeds normally (cancel_result True, counter
          stays 0, advance_workflow's own is_active() guard then makes it a
          clean no-op returning False);
        - if advance_workflow wins first, it runs the step to completion
          (COMPLETED) before cancel_workflow gets the lock, and this fix's
          terminal guard makes cancel_workflow a no-op returning False
          instead of raising WorkflowStateError (the reviewer's exact
          repro).
        """
        from fsm_llm_workflows.definitions import WorkflowDefinition
        from fsm_llm_workflows.models import WorkflowInstance, WorkflowStatus

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
                engine.advance_workflow("cancel-vs-complete-1"),
                engine.cancel_workflow("cancel-vs-complete-1"),
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
                # advance_workflow won the lock race and ran the step first.
                assert cancel_result is False
                assert len(counter) == 1
            else:
                # cancel_workflow won the lock race before the step ran.
                assert cancel_result is True
                assert len(counter) == 0


class TestFloatTimeoutSecondsEndToEnd:
    """F7 regression: a sub-second workflow_timeout must report its real
    float value (e.g. 0.5) in the raised WorkflowTimeoutError, not `0` from
    a stale `int(...)` truncation in engine.py's `_timeout_seconds()`
    helper. See decisions.md D-011."""

    async def test_expired_deadline_reports_subsecond_float_timeout(self):
        from datetime import datetime, timedelta, timezone

        from fsm_llm_workflows.definitions import WorkflowDefinition
        from fsm_llm_workflows.models import WorkflowInstance, WorkflowStatus
        from fsm_llm_workflows.steps import AutoTransitionStep, WorkflowStep

        class _TerminalStep(WorkflowStep):
            async def execute(self, context):
                return None

        engine = WorkflowEngine()
        step = AutoTransitionStep(step_id="only", name="Only", next_state="term")
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
            workflow_timeout=0.5,
            deadline=datetime.now(timezone.utc) - timedelta(seconds=1),
        )
        engine.workflow_instances[instance.instance_id] = instance

        with pytest.raises(WorkflowTimeoutError) as exc_info:
            await engine._execute_workflow_step(instance)

        assert exc_info.value.timeout_seconds == 0.5
        assert "0.5 seconds" in str(exc_info.value)
