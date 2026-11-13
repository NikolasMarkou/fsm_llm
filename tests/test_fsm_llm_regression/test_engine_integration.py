"""
Integration tests for ReasoningEngine and WorkflowEngine.

These tests verify end-to-end engine behavior that component-level tests miss.
They exercise actual engine orchestration, FSM stacking, context flow, and
the interaction between extensions and core.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

import pytest

from fsm_llm.core.definitions import FSMDefinition
from fsm_llm.reasoning.constants import ContextKeys, ReasoningType
from fsm_llm.reasoning.handlers import OutputFormatter
from fsm_llm.reasoning.utilities import map_reasoning_type
from fsm_llm.workflows.dsl import (
    auto_step,
    create_workflow,
    timer_step,
    wait_event_step,
)
from fsm_llm.workflows.engine import WorkflowEngine
from fsm_llm.workflows.models import WorkflowEvent, WorkflowStatus
from tests.test_fsm_llm_reasoning.test_engine_scripted import (
    _PROBLEM,
    _VALID_SCRIPT,
    _ScriptedLLM,
)

# ============================================================================
# Reasoning Engine Integration Tests
# ============================================================================


class TestReasoningEngineIntegration:
    """Integration tests for the reasoning engine end-to-end.

    Driven by a scripted LLM that answers every extraction, so the solve
    really walks the orchestrator, the classifier and a pushed strategy FSM
    to ``final_answer`` (the old mock answered no extraction, so the solve
    only ever stopped on the iteration cap and asserted nothing).
    """

    def test_solve_problem_returns_solution(self):
        """solve_problem() returns the solution the script wrote."""
        from fsm_llm.reasoning import ReasoningEngine

        llm = _ScriptedLLM(_VALID_SCRIPT)
        engine = ReasoningEngine(model="mock", llm_interface=llm)
        solution, trace_info = engine.solve_problem(_PROBLEM)

        assert solution == _VALID_SCRIPT["final_solution"]
        assert isinstance(trace_info, dict)
        assert "reasoning_trace" in trace_info
        assert "summary" in trace_info
        assert llm.requests, "the scripted LLM must have been called"

    def test_solve_problem_completes_within_iteration_limit(self):
        """solve_problem() finishes, message-free, in a bounded number of calls."""
        from fsm_llm.reasoning import ReasoningEngine

        llm = _ScriptedLLM(_VALID_SCRIPT)
        engine = ReasoningEngine(model="mock", llm_interface=llm)
        engine.solve_problem(_PROBLEM)

        assert 0 < len(llm.requests) < 100
        assert all(request.user_message is None for request in llm.requests)

    def test_solve_problem_does_not_mutate_initial_context(self):
        """F-002 (iter-2): solve_problem() copies the caller's initial_context."""
        from fsm_llm.reasoning import ReasoningEngine

        llm = _ScriptedLLM(_VALID_SCRIPT)
        engine = ReasoningEngine(model="mock", llm_interface=llm)
        initial_context = {"domain": "math", "difficulty": "easy"}
        engine.solve_problem(_PROBLEM, initial_context)

        assert initial_context == {"domain": "math", "difficulty": "easy"}
        assert llm.requests

    def test_classification_guard_prevents_reclassification(self):
        """Classification should skip if already classified."""
        from fsm_llm.reasoning import ReasoningEngine
        from fsm_llm.reasoning.constants import ContextKeys

        llm = _ScriptedLLM(_VALID_SCRIPT)
        engine = ReasoningEngine(model="mock", llm_interface=llm)

        # If already classified, _classify_problem should return empty dict
        context_with_classification = {
            ContextKeys.CLASSIFIED_PROBLEM_TYPE: "analytical",
            ContextKeys.PROBLEM_STATEMENT: "test",
            ContextKeys.PROBLEM_TYPE: "test",
        }
        result = engine._classify_problem(context_with_classification)
        assert result == {}
        assert llm.requests == []

        # Without classification, the classifier FSM runs and recommends.
        context_without_classification = {
            ContextKeys.PROBLEM_STATEMENT: _PROBLEM,
            ContextKeys.PROBLEM_TYPE: "logic puzzle",
        }
        result = engine._classify_problem(context_without_classification)
        assert result[ContextKeys.CLASSIFIED_PROBLEM_TYPE] == "deductive"
        assert llm.field_requests(ContextKeys.RECOMMENDED_REASONING_TYPE)

    def test_fsm_definitions_have_extraction_instructions(self):
        """All reasoning FSM states must have extraction_instructions populated."""
        from fsm_llm.reasoning.reasoning_modes import ALL_REASONING_FSMS

        for fsm_name, fsm_dict in ALL_REASONING_FSMS.items():
            fsm_def = FSMDefinition(**fsm_dict)
            for state_id, state in fsm_def.states.items():
                assert state.extraction_instructions is not None, (
                    f"FSM '{fsm_name}' state '{state_id}' has no extraction_instructions"
                )
                assert len(state.extraction_instructions.strip()) > 0, (
                    f"FSM '{fsm_name}' state '{state_id}' has empty extraction_instructions"
                )

    def test_fsm_definitions_have_response_instructions(self):
        """All reasoning FSM states must have response_instructions populated."""
        from fsm_llm.reasoning.reasoning_modes import ALL_REASONING_FSMS

        for fsm_name, fsm_dict in ALL_REASONING_FSMS.items():
            fsm_def = FSMDefinition(**fsm_dict)
            for state_id, state in fsm_def.states.items():
                assert state.response_instructions is not None, (
                    f"FSM '{fsm_name}' state '{state_id}' has no response_instructions"
                )

    def test_no_bare_instructions_field(self):
        """No reasoning FSM should use the bare 'instructions' field (silently ignored)."""
        from fsm_llm.reasoning.reasoning_modes import ALL_REASONING_FSMS

        for fsm_name, fsm_dict in ALL_REASONING_FSMS.items():
            for state_id, state_dict in fsm_dict.get("states", {}).items():
                assert "instructions" not in state_dict, (
                    f"FSM '{fsm_name}' state '{state_id}' uses bare 'instructions' field "
                    "(silently ignored by Pydantic State model — use extraction_instructions "
                    "and response_instructions instead)"
                )

    def test_max_total_iterations_constant_exists(self):
        """Defaults.MAX_TOTAL_STEPS must exist to prevent infinite loops."""
        from fsm_llm.reasoning.constants import Defaults

        assert hasattr(Defaults, "MAX_TOTAL_STEPS")
        assert isinstance(Defaults.MAX_TOTAL_STEPS, int)
        assert Defaults.MAX_TOTAL_STEPS > 0


# ============================================================================
# Workflow Engine Integration Tests
# ============================================================================


class TestWorkflowEngineIntegration:
    """Integration tests for the workflow engine end-to-end."""

    @pytest.mark.asyncio
    async def test_linear_workflow_completes(self):
        """A simple linear workflow should execute to completion."""
        from fsm_llm.workflows import (
            WorkflowEngine,
            auto_step,
            create_workflow,
        )
        from fsm_llm.workflows.models import WorkflowStatus

        actions_called = []

        def step_action(ctx):
            actions_called.append("step1")
            return {"processed": True}

        workflow = create_workflow("test_linear", "Test Linear")
        workflow.with_initial_step(
            auto_step("start", "Start", next_step_id="end", action=step_action)
        )
        workflow.with_step(auto_step("end", "End", next_step_id=""))

        engine = WorkflowEngine()
        engine.register_workflow(workflow)
        instance_id = await engine.start("test_linear")

        instance = engine.get_instance(instance_id)
        assert instance is not None
        assert instance.status == WorkflowStatus.COMPLETED
        assert "step1" in actions_called

    @pytest.mark.asyncio
    async def test_condition_branching(self):
        """Condition steps should route to the correct branch."""
        from fsm_llm.workflows import (
            WorkflowEngine,
            auto_step,
            condition_step,
            create_workflow,
        )
        from fsm_llm.workflows.models import WorkflowStatus

        workflow = create_workflow("test_cond", "Test Condition")
        workflow.with_initial_step(
            condition_step(
                "check",
                "Check",
                condition=lambda ctx: ctx.get("value", 0) > 5,
                true_step_id="high",
                false_step_id="low",
            )
        )
        workflow.with_step(auto_step("high", "High", next_step_id=""))
        workflow.with_step(auto_step("low", "Low", next_step_id=""))

        engine = WorkflowEngine()
        engine.register_workflow(workflow)

        # Test false branch
        instance_id = await engine.start("test_cond", initial_context={"value": 3})
        instance = engine.get_instance(instance_id)
        assert instance.status == WorkflowStatus.COMPLETED
        assert instance.current_step_id == "low"

        # Test true branch
        instance_id = await engine.start("test_cond", initial_context={"value": 10})
        instance = engine.get_instance(instance_id)
        assert instance.status == WorkflowStatus.COMPLETED
        assert instance.current_step_id == "high"

    @pytest.mark.asyncio
    async def test_api_step_context_mapping(self):
        """API steps should correctly map input/output context."""
        from fsm_llm.workflows import (
            WorkflowEngine,
            api_step,
            auto_step,
            create_workflow,
        )
        from fsm_llm.workflows.models import WorkflowStatus

        async def mock_api(name: str = "", **kwargs):
            return {"greeting": f"Hello, {name}!"}

        workflow = create_workflow("test_api", "Test API")
        workflow.with_initial_step(
            api_step(
                "call_api",
                "Call API",
                api_function=mock_api,
                success_step_id="done",
                failure_step_id="error",
                input_mapping={"name": "user_name"},
                output_mapping={"result_greeting": "greeting"},
            )
        )
        workflow.with_step(auto_step("done", "Done", next_step_id=""))
        workflow.with_step(auto_step("error", "Error", next_step_id=""))

        engine = WorkflowEngine()
        engine.register_workflow(workflow)
        instance_id = await engine.start(
            "test_api", initial_context={"user_name": "Alice"}
        )

        instance = engine.get_instance(instance_id)
        assert instance.status == WorkflowStatus.COMPLETED
        assert instance.current_step_id == "done"
        assert instance.context.get("result_greeting") == "Hello, Alice!"

    @pytest.mark.asyncio
    async def test_parallel_step_executes_concurrently(self):
        """Parallel steps should all execute and merge results."""
        from fsm_llm.workflows import (
            WorkflowEngine,
            auto_step,
            create_workflow,
            parallel_step,
        )
        from fsm_llm.workflows.models import WorkflowStatus

        results = []

        def make_action(name):
            def action(ctx):
                results.append(name)
                return {f"{name}_done": True}

            return action

        sub_steps = [
            auto_step(
                f"sub_{i}", f"Sub {i}", next_step_id="", action=make_action(f"sub_{i}")
            )
            for i in range(3)
        ]

        workflow = create_workflow("test_parallel", "Test Parallel")
        workflow.with_initial_step(
            parallel_step("par", "Parallel", steps=sub_steps, next_step_id="done")
        )
        workflow.with_step(auto_step("done", "Done", next_step_id=""))

        engine = WorkflowEngine()
        engine.register_workflow(workflow)
        instance_id = await engine.start("test_parallel")

        instance = engine.get_instance(instance_id)
        assert instance.status == WorkflowStatus.COMPLETED
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_terminal_step_with_empty_next_step_id(self):
        """auto_step with next_step_id='' should be a valid terminal step."""
        from fsm_llm.workflows import (
            WorkflowEngine,
            auto_step,
            create_workflow,
        )
        from fsm_llm.workflows.models import WorkflowStatus

        workflow = create_workflow("test_terminal", "Test Terminal")
        workflow.with_initial_step(auto_step("start", "Start", next_step_id="done"))
        workflow.with_step(auto_step("done", "Done", next_step_id=""))

        # Should not raise during validation
        workflow.validate_structure()

        engine = WorkflowEngine()
        engine.register_workflow(workflow)
        instance_id = await engine.start("test_terminal")

        instance = engine.get_instance(instance_id)
        assert instance.status == WorkflowStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_event_workflow_clears_waiting_info(self):
        """After processing an event, _waiting_info should be cleared."""
        from fsm_llm.workflows import (
            WorkflowEngine,
            auto_step,
            create_workflow,
            wait_event_step,
        )
        from fsm_llm.workflows.models import WorkflowEvent, WorkflowStatus

        workflow = create_workflow("test_event", "Test Event")
        workflow.with_initial_step(
            wait_event_step(
                "wait",
                "Wait for Event",
                event_type="test_event",
                success_step_id="done",
            )
        )
        workflow.with_step(auto_step("done", "Done", next_step_id=""))

        engine = WorkflowEngine()
        engine.register_workflow(workflow)
        instance_id = await engine.start("test_event")

        # Should be waiting
        instance = engine.get_instance(instance_id)
        assert instance.status == WorkflowStatus.WAITING

        # Register listener and process event
        await engine.register_event_listener(
            instance_id, "test_event", success_step_id="done"
        )
        event = WorkflowEvent(event_type="test_event", payload={"data": "test"})
        await engine.process_event(event)

        # Should be completed, not stuck at WAITING
        instance = engine.get_instance(instance_id)
        assert instance.status == WorkflowStatus.COMPLETED
        assert "_waiting_info" not in instance.context

    @pytest.mark.asyncio
    async def test_custom_step_subclass_validates(self):
        """Custom WorkflowStep subclasses should pass validation."""
        from fsm_llm.workflows import (
            WorkflowEngine,
            auto_step,
            create_workflow,
        )
        from fsm_llm.workflows.models import WorkflowStatus, WorkflowStepResult
        from fsm_llm.workflows.steps import WorkflowStep

        class CustomStep(WorkflowStep):
            """A custom step with state references."""

            success_step_id: str = ""
            error_step_id: str = ""

            async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
                return WorkflowStepResult.success_result(
                    data={"custom": True},
                    next_step_id=self.success_step_id if self.success_step_id else None,
                )

        workflow = create_workflow("test_custom", "Test Custom Step")
        workflow.with_initial_step(
            CustomStep(
                step_id="custom",
                name="Custom",
                success_step_id="done",
                error_step_id="error",
            )
        )
        workflow.with_step(auto_step("done", "Done", next_step_id=""))
        workflow.with_step(auto_step("error", "Error", next_step_id=""))

        # Should not raise
        workflow.validate_structure()

        engine = WorkflowEngine()
        engine.register_workflow(workflow)
        instance_id = await engine.start("test_custom")

        instance = engine.get_instance(instance_id)
        assert instance.status == WorkflowStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_concurrent_workflows_no_leaks(self):
        """Multiple concurrent workflows should all complete with no resource leaks."""
        from fsm_llm.workflows import (
            WorkflowEngine,
            auto_step,
            create_workflow,
        )
        from fsm_llm.workflows.models import WorkflowStatus

        workflow = create_workflow("test_concurrent", "Test Concurrent")
        workflow.with_initial_step(auto_step("a", "A", next_step_id="b"))
        workflow.with_step(auto_step("b", "B", next_step_id="c"))
        workflow.with_step(auto_step("c", "C", next_step_id=""))

        engine = WorkflowEngine()
        engine.register_workflow(workflow)

        instance_ids = []
        for _ in range(10):
            iid = await engine.start("test_concurrent")
            instance_ids.append(iid)

        for iid in instance_ids:
            instance = engine.get_instance(iid)
            assert instance.status == WorkflowStatus.COMPLETED

        stats = engine.get_statistics()
        assert stats["active_workflows"] == 0


class TestFalsySolutionExtraction:
    """ED-001: extract_final_solution must use `is not None` instead of truthiness."""

    def test_calculation_result_zero_is_returned(self):
        """CALCULATION_RESULT=0 must be returned, not dropped."""
        context = {ContextKeys.CALCULATION_RESULT: 0}
        result = OutputFormatter.extract_final_solution(context)
        assert result == "0"

    def test_calculation_result_false_is_returned(self):
        """CALCULATION_RESULT=False must be returned as 'False'."""
        context = {ContextKeys.CALCULATION_RESULT: False}
        result = OutputFormatter.extract_final_solution(context)
        assert result == "False"

    def test_proposed_solution_zero_is_returned(self):
        """PROPOSED_SOLUTION=0 must be returned."""
        context = {ContextKeys.PROPOSED_SOLUTION: 0}
        result = OutputFormatter.extract_final_solution(context)
        assert result == "0"

    def test_empty_string_still_falls_through(self):
        """Empty string is a genuinely absent solution — should fall through."""
        context = {ContextKeys.FINAL_SOLUTION: ""}
        result = OutputFormatter.extract_final_solution(context)
        assert "Solution process completed" in result

    def test_none_still_falls_through(self):
        """None is a genuinely absent solution — should fall through."""
        context = {ContextKeys.FINAL_SOLUTION: None}
        result = OutputFormatter.extract_final_solution(context)
        assert "Solution process completed" in result

    def test_priority_order_respected_with_zero(self):
        """FINAL_SOLUTION=0 takes priority over PROPOSED_SOLUTION='real answer'."""
        context = {
            ContextKeys.FINAL_SOLUTION: 0,
            ContextKeys.PROPOSED_SOLUTION: "real answer",
        }
        result = OutputFormatter.extract_final_solution(context)
        assert result == "0"

    def test_fallback_chain_skips_none_finds_zero(self):
        """First key is None, second key is 0 — should return '0'."""
        context = {
            ContextKeys.FINAL_SOLUTION: None,
            ContextKeys.PROPOSED_SOLUTION: None,
            ContextKeys.CALCULATION_RESULT: 0,
        }
        result = OutputFormatter.extract_final_solution(context)
        assert result == "0"


class TestEventAutoRegistration:
    """ED-002: Engine must auto-register event listeners from _waiting_info."""

    async def test_wait_event_step_auto_registers_listener(self):
        """After WaitEventStep, engine should auto-register event listener."""
        engine = WorkflowEngine()

        wf = create_workflow("evt_auto", "Event Auto")
        wf.with_initial_step(auto_step("setup", "Setup", "wait"))
        wf.with_step(
            wait_event_step(
                "wait",
                "Wait",
                event_type="order_confirmed",
                success_step_id="process",
                event_mapping={"received_id": "order_id"},
            )
        )
        wf.with_step(auto_step("process", "Process", ""))
        engine.register_workflow(wf)

        iid = await engine.start("evt_auto", {})
        inst = engine.get_instance(iid)
        assert inst.status == WorkflowStatus.WAITING

        # Listener should have been auto-registered
        assert "order_confirmed" in engine.event_listeners
        assert iid in engine.event_listeners["order_confirmed"]

    async def test_event_fires_without_manual_registration(self):
        """After auto-registration, sending an event should advance the workflow."""
        engine = WorkflowEngine()

        wf = create_workflow("evt_fire", "Event Fire")
        wf.with_initial_step(auto_step("setup", "Setup", "wait"))
        wf.with_step(
            wait_event_step(
                "wait",
                "Wait",
                event_type="payment_done",
                success_step_id="confirm",
                event_mapping={"amount": "paid_amount"},
            )
        )
        wf.with_step(auto_step("confirm", "Confirm", ""))
        engine.register_workflow(wf)

        iid = await engine.start("evt_fire", {})

        # Send event — should transition automatically
        event = WorkflowEvent(event_type="payment_done", payload={"paid_amount": 99})
        await engine.process_event(event)

        inst = engine.get_instance(iid)
        assert inst.status == WorkflowStatus.COMPLETED
        assert inst.current_step_id == "confirm"
        assert inst.context.get("amount") == 99

    async def test_event_mapping_applied_correctly(self):
        """Event payload should be mapped to context via event_mapping."""
        engine = WorkflowEngine()

        wf = create_workflow("evt_map", "Event Map")
        wf.with_initial_step(
            wait_event_step(
                "wait",
                "Wait",
                event_type="data_ready",
                success_step_id="done",
                event_mapping={"local_key": "remote_key"},
            )
        )
        wf.with_step(auto_step("done", "Done", ""))
        engine.register_workflow(wf)

        iid = await engine.start("evt_map", {})

        event = WorkflowEvent(
            event_type="data_ready",
            payload={"remote_key": "hello_world"},
        )
        await engine.process_event(event)

        inst = engine.get_instance(iid)
        assert inst.context.get("local_key") == "hello_world"


class TestTimerAutoScheduling:
    """ED-002: Engine must auto-schedule timers from _timer_info."""

    async def test_timer_step_auto_schedules(self):
        """After TimerStep, engine should auto-schedule the timer."""
        engine = WorkflowEngine()

        wf = create_workflow("tmr_auto", "Timer Auto")
        wf.with_initial_step(
            timer_step("wait", "Wait", delay_seconds=1, next_step_id="done")
        )
        wf.with_step(auto_step("done", "Done", ""))
        engine.register_workflow(wf)

        iid = await engine.start("tmr_auto", {})

        inst = engine.get_instance(iid)
        assert inst.status == WorkflowStatus.WAITING

        # Timer should have been auto-scheduled
        timer_key = f"{iid}_timer"
        assert timer_key in engine.timers

    async def test_timer_fires_and_advances_workflow(self, timing_factor):
        """Auto-scheduled timer should transition the workflow after delay."""
        engine = WorkflowEngine()

        wf = create_workflow("tmr_fire", "Timer Fire")
        wf.with_initial_step(
            timer_step("wait", "Wait", delay_seconds=0, next_step_id="done")
        )
        wf.with_step(auto_step("done", "Done", ""))
        engine.register_workflow(wf)

        iid = await engine.start("tmr_fire", {})

        # Give the timer task a moment to fire (delay_seconds=0 but
        # asyncio.sleep(0) still yields to event loop once)
        await asyncio.sleep(0.1 * timing_factor)

        inst = engine.get_instance(iid)
        assert inst.status == WorkflowStatus.COMPLETED
        assert inst.current_step_id == "done"


class TestTypeMappingAliases:
    """ED-003: Natural language aliases must map to correct reasoning types."""

    def test_math_maps_to_calculator(self):
        assert map_reasoning_type("math") == ReasoningType.SIMPLE_CALCULATOR.value

    def test_calculator_maps_to_calculator(self):
        assert map_reasoning_type("calculator") == ReasoningType.SIMPLE_CALCULATOR.value

    def test_compute_maps_to_calculator(self):
        assert map_reasoning_type("compute") == ReasoningType.SIMPLE_CALCULATOR.value

    def test_logic_maps_to_deductive(self):
        assert map_reasoning_type("logic") == ReasoningType.DEDUCTIVE.value

    def test_logical_maps_to_deductive(self):
        assert map_reasoning_type("logical") == ReasoningType.DEDUCTIVE.value

    def test_pattern_maps_to_inductive(self):
        assert map_reasoning_type("pattern") == ReasoningType.INDUCTIVE.value

    def test_observation_maps_to_inductive(self):
        assert map_reasoning_type("observation") == ReasoningType.INDUCTIVE.value

    def test_innovative_maps_to_creative(self):
        assert map_reasoning_type("innovative") == ReasoningType.CREATIVE.value

    def test_brainstorm_maps_to_creative(self):
        assert map_reasoning_type("brainstorm") == ReasoningType.CREATIVE.value

    def test_evaluate_maps_to_critical(self):
        assert map_reasoning_type("evaluate") == ReasoningType.CRITICAL.value

    def test_assessment_maps_to_critical(self):
        assert map_reasoning_type("assessment") == ReasoningType.CRITICAL.value

    def test_hypothesis_maps_to_abductive(self):
        assert map_reasoning_type("hypothesis") == ReasoningType.ABDUCTIVE.value

    def test_abduction_maps_to_abductive(self):
        assert map_reasoning_type("abduction") == ReasoningType.ABDUCTIVE.value

    def test_compare_maps_to_analogical(self):
        assert map_reasoning_type("compare") == ReasoningType.ANALOGICAL.value

    def test_case_insensitive(self):
        assert map_reasoning_type("MATH") == ReasoningType.SIMPLE_CALCULATOR.value
        assert map_reasoning_type("Logic") == ReasoningType.DEDUCTIVE.value

    def test_whitespace_stripped(self):
        assert map_reasoning_type("  math  ") == ReasoningType.SIMPLE_CALCULATOR.value

    def test_unknown_still_defaults_to_analytical(self):
        assert (
            map_reasoning_type("quantum_teleportation")
            == ReasoningType.ANALYTICAL.value
        )

    def test_all_canonical_names_map_to_themselves(self):
        """Every ReasoningType enum value should map to itself."""
        for rt in ReasoningType:
            assert map_reasoning_type(rt.value) == rt.value


class TestWaitEventStepValidation:
    """F-004: WaitEventStep success/timeout states must be validated."""

    def test_wait_event_step_states_are_validated(self):
        """The referenced states of a WaitEventStep include both states."""
        try:
            from fsm_llm.workflows.models import WaitEventConfig
            from fsm_llm.workflows.steps import WaitEventStep

            step = WaitEventStep(
                step_id="wait_payment",
                name="Wait for Payment",
                config=WaitEventConfig(
                    event_type="payment_received",
                    success_step_id="payment_done",
                    timeout_seconds=300,
                    timeout_step_id="payment_timeout",
                ),
            )

            referenced = step.referenced_step_ids()
            assert "payment_done" in referenced
            assert "payment_timeout" in referenced
        except ImportError:
            import pytest

            pytest.skip("fsm_llm.workflows not installed")


class TestContextPruningLog:
    """C2: prune_context should compute actual new size."""

    def test_prune_reports_different_sizes(self):
        from fsm_llm.reasoning.constants import ContextKeys, Defaults
        from fsm_llm.reasoning.handlers import ReasoningHandlers

        # Create context large enough to trigger pruning
        large_list = [f"item_{i}" for i in range(50)]
        context = {
            ContextKeys.PROBLEM_STATEMENT: "test",
            ContextKeys.REASONING_TRACE: large_list,
            ContextKeys.LOGICAL_STEPS: large_list.copy(),
            ContextKeys.OBSERVATIONS: large_list.copy(),
        }
        # Pad to exceed threshold
        context["_padding"] = "x" * (Defaults.CONTEXT_PRUNE_THRESHOLD + 1000)

        result = ReasoningHandlers.prune_context(context)

        # Should have pruned something
        assert len(result) > 0
        # At least one list should be truncated to 10
        for key in result:
            if isinstance(result[key], list):
                assert len(result[key]) <= 10


class TestContextKeysConstants:
    """C3: merge_reasoning_results should only use ContextKeys constants."""

    def test_no_raw_strings_in_merge(self):
        """All keys in merge_reasoning_results should come from ContextKeys."""
        from fsm_llm.reasoning.handlers import ContextManager

        source = inspect.getsource(ContextManager.merge_reasoning_results)

        # These raw strings should no longer appear as dict keys
        forbidden_raw_keys = [
            '"deductive_conclusion"',
            '"best_explanation"',
            '"adapted_solution_or_understanding"',
            '"analogy_confidence"',
            '"assessment_confidence"',
            '"calculation_error_details"',
            '"hybrid_synthesis_summary"',
            '"explanation_confidence"',
            '"analogical_solution"',
            '"inductive_hypothesis"',
        ]

        for key in forbidden_raw_keys:
            # Check it's not used as a dict key assignment (results[key] = ...)
            pattern = f"results[{key}]"
            assert pattern not in source, (
                f"Raw string {key} found in merge_reasoning_results — use ContextKeys constant"
            )

    def test_all_reasoning_types_covered(self):
        """merge_reasoning_results should handle all ReasoningType values."""
        from fsm_llm.reasoning.constants import ReasoningType
        from fsm_llm.reasoning.handlers import ContextManager

        for rt in ReasoningType:
            # Should not crash for any reasoning type
            result = ContextManager.merge_reasoning_results(
                orchestrator_context={}, sub_fsm_context={}, reasoning_type=rt.value
            )
            # Should always have a completion flag
            assert f"{rt.value}_reasoning_completed" in result


class TestNoMergeStrategyAlias:
    """H4: reasoning engine should use ContextMergeStrategy directly."""

    def test_engine_imports_context_merge_strategy(self):
        from fsm_llm.reasoning import engine

        source = inspect.getsource(engine)
        # Should import from public API (not internal fsm_llm.api)
        assert "ContextMergeStrategy" in source
        # Must come from fsm_llm, not fsm_llm.api
        assert "from fsm_llm import" in source
        # Should not import the alias from constants
        assert "from .constants import" in source
        import_line = next(
            line for line in source.split("\n") if "from .constants import" in line
        )
        assert "MergeStrategy" not in import_line
