"""
Regression tests for issues found during epistemic deconstruction analysis.
Covers:
  ED-001: OutputFormatter falsy check loses CALCULATION_RESULT=0
  ED-002: WaitForEventStep / TimerStep auto-registration in engine
  ED-003: Type mapping aliases incomplete
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

from fsm_llm_reasoning.constants import ContextKeys, ReasoningType
from fsm_llm_reasoning.handlers import ContextManager, OutputFormatter, ReasoningHandlers
from fsm_llm_reasoning.utilities import map_reasoning_type
from fsm_llm_workflows.dsl import auto_step, create_workflow, timer_step, wait_event_step
from fsm_llm_workflows.engine import WorkflowEngine
from fsm_llm_workflows.models import WorkflowEvent, WorkflowStatus


# ---------------------------------------------------------------------------
# ED-001: OutputFormatter must not discard falsy solutions (0, False)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# ED-002: WaitForEventStep / TimerStep auto-registration
# ---------------------------------------------------------------------------


class TestEventAutoRegistration:
    """ED-002: Engine must auto-register event listeners from _waiting_info."""

    def test_wait_event_step_auto_registers_listener(self):
        """After WaitForEventStep, engine should auto-register event listener."""
        engine = WorkflowEngine()

        wf = create_workflow("evt_auto", "Event Auto")
        wf.with_initial_step(auto_step("setup", "Setup", "wait"))
        wf.with_step(
            wait_event_step(
                "wait", "Wait",
                event_type="order_confirmed",
                success_state="process",
                event_mapping={"received_id": "order_id"},
            )
        )
        wf.with_step(auto_step("process", "Process", ""))
        engine.register_workflow(wf)

        iid = asyncio.get_event_loop().run_until_complete(
            engine.start_workflow("evt_auto", {})
        )
        inst = engine.get_workflow_instance(iid)
        assert inst.status == WorkflowStatus.WAITING

        # Listener should have been auto-registered
        assert "order_confirmed" in engine.event_listeners
        assert iid in engine.event_listeners["order_confirmed"]

    def test_event_fires_without_manual_registration(self):
        """After auto-registration, sending an event should advance the workflow."""
        engine = WorkflowEngine()

        wf = create_workflow("evt_fire", "Event Fire")
        wf.with_initial_step(auto_step("setup", "Setup", "wait"))
        wf.with_step(
            wait_event_step(
                "wait", "Wait",
                event_type="payment_done",
                success_state="confirm",
                event_mapping={"amount": "paid_amount"},
            )
        )
        wf.with_step(auto_step("confirm", "Confirm", ""))
        engine.register_workflow(wf)

        loop = asyncio.get_event_loop()
        iid = loop.run_until_complete(engine.start_workflow("evt_fire", {}))

        # Send event — should transition automatically
        event = WorkflowEvent(event_type="payment_done", payload={"paid_amount": 99})
        loop.run_until_complete(engine.process_event(event))

        inst = engine.get_workflow_instance(iid)
        assert inst.status == WorkflowStatus.COMPLETED
        assert inst.current_step_id == "confirm"
        assert inst.context.get("amount") == 99

    def test_event_mapping_applied_correctly(self):
        """Event payload should be mapped to context via event_mapping."""
        engine = WorkflowEngine()

        wf = create_workflow("evt_map", "Event Map")
        wf.with_initial_step(
            wait_event_step(
                "wait", "Wait",
                event_type="data_ready",
                success_state="done",
                event_mapping={"local_key": "remote_key"},
            )
        )
        wf.with_step(auto_step("done", "Done", ""))
        engine.register_workflow(wf)

        loop = asyncio.get_event_loop()
        iid = loop.run_until_complete(engine.start_workflow("evt_map", {}))

        event = WorkflowEvent(
            event_type="data_ready",
            payload={"remote_key": "hello_world"},
        )
        loop.run_until_complete(engine.process_event(event))

        inst = engine.get_workflow_instance(iid)
        assert inst.context.get("local_key") == "hello_world"


class TestTimerAutoScheduling:
    """ED-002: Engine must auto-schedule timers from _timer_info."""

    def test_timer_step_auto_schedules(self):
        """After TimerStep, engine should auto-schedule the timer."""
        engine = WorkflowEngine()

        wf = create_workflow("tmr_auto", "Timer Auto")
        wf.with_initial_step(
            timer_step("wait", "Wait", delay_seconds=1, next_state="done")
        )
        wf.with_step(auto_step("done", "Done", ""))
        engine.register_workflow(wf)

        loop = asyncio.get_event_loop()
        iid = loop.run_until_complete(engine.start_workflow("tmr_auto", {}))

        inst = engine.get_workflow_instance(iid)
        assert inst.status == WorkflowStatus.WAITING

        # Timer should have been auto-scheduled
        timer_key = f"{iid}_timer"
        assert timer_key in engine.timers

    def test_timer_fires_and_advances_workflow(self):
        """Auto-scheduled timer should transition the workflow after delay."""
        engine = WorkflowEngine()

        wf = create_workflow("tmr_fire", "Timer Fire")
        wf.with_initial_step(
            timer_step("wait", "Wait", delay_seconds=0, next_state="done")
        )
        wf.with_step(auto_step("done", "Done", ""))
        engine.register_workflow(wf)

        loop = asyncio.get_event_loop()
        iid = loop.run_until_complete(engine.start_workflow("tmr_fire", {}))

        # Give the timer task a moment to fire (delay_seconds=0 but
        # asyncio.sleep(0) still yields to event loop once)
        loop.run_until_complete(asyncio.sleep(0.1))

        inst = engine.get_workflow_instance(iid)
        assert inst.status == WorkflowStatus.COMPLETED
        assert inst.current_step_id == "done"


# ---------------------------------------------------------------------------
# ED-003: Type mapping aliases
# ---------------------------------------------------------------------------


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
        assert map_reasoning_type("quantum_teleportation") == ReasoningType.ANALYTICAL.value

    def test_all_canonical_names_map_to_themselves(self):
        """Every ReasoningType enum value should map to itself."""
        for rt in ReasoningType:
            assert map_reasoning_type(rt.value) == rt.value
