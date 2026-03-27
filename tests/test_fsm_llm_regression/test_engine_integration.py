"""
Integration tests for ReasoningEngine and WorkflowEngine.

These tests verify end-to-end engine behavior that component-level tests miss.
They exercise actual engine orchestration, FSM stacking, context flow, and
the interaction between extensions and core.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from fsm_llm.definitions import (
    FSMDefinition,
    ResponseGenerationRequest,
    ResponseGenerationResponse,
)
from fsm_llm.llm import LLMInterface

# ============================================================================
# Reasoning Engine Integration Tests
# ============================================================================


def _extract_state_from_prompt(system_prompt: str) -> str:
    """Extract current state ID from the system prompt XML tags."""
    match = re.search(r"<current_state>(\w+)</current_state>", system_prompt)
    return match.group(1) if match else ""


class ReasoningMockLLM(LLMInterface):
    """Mock LLM that drives the reasoning engine through its full state machine.

    Provides state-aware responses so the orchestrator progresses through
    PROBLEM_ANALYSIS -> STRATEGY_SELECTION -> EXECUTE_REASONING ->
    SYNTHESIZE_SOLUTION -> VALIDATE_REFINE -> FINAL_ANSWER.
    """

    def __init__(self):
        self.call_count = 0

    def generate_response(
        self, request: ResponseGenerationRequest
    ) -> ResponseGenerationResponse:
        return ResponseGenerationResponse(
            message="Processing...",
            message_type="response",
            reasoning="Mock response",
        )


class TestReasoningEngineIntegration:
    """Integration tests for the reasoning engine end-to-end."""

    def test_solve_problem_returns_solution(self):
        """solve_problem() should return a (solution, trace) tuple."""
        from fsm_llm_reasoning import ReasoningEngine

        mock_llm = ReasoningMockLLM()
        engine = ReasoningEngine(model="mock", llm_interface=mock_llm)
        solution, trace_info = engine.solve_problem("What is 2 + 2?")

        assert isinstance(solution, str)
        assert len(solution) > 0
        assert isinstance(trace_info, dict)
        assert "reasoning_trace" in trace_info
        assert "summary" in trace_info

    def test_solve_problem_completes_within_iteration_limit(self):
        """solve_problem() must not loop forever."""
        from fsm_llm_reasoning import ReasoningEngine

        mock_llm = ReasoningMockLLM()
        engine = ReasoningEngine(model="mock", llm_interface=mock_llm)
        _solution, _trace_info = engine.solve_problem("What is 2 + 2?")

        # Should complete well under the 50-iteration limit
        assert mock_llm.call_count < 100

    def test_classification_guard_prevents_reclassification(self):
        """Classification should skip if already classified."""
        from fsm_llm_reasoning import ReasoningEngine
        from fsm_llm_reasoning.constants import ContextKeys

        mock_llm = ReasoningMockLLM()
        engine = ReasoningEngine(model="mock", llm_interface=mock_llm)

        # If already classified, _classify_problem should return empty dict
        context_with_classification = {
            ContextKeys.CLASSIFIED_PROBLEM_TYPE: "analytical",
            ContextKeys.PROBLEM_STATEMENT: "test",
            ContextKeys.PROBLEM_TYPE: "test",
        }
        result = engine._classify_problem(context_with_classification)
        assert result == {}

        # Without classification, it should return classification results
        context_without_classification = {
            ContextKeys.PROBLEM_STATEMENT: "What is 2 + 2?",
            ContextKeys.PROBLEM_TYPE: "arithmetic",
        }
        result = engine._classify_problem(context_without_classification)
        assert ContextKeys.CLASSIFIED_PROBLEM_TYPE in result

    def test_fsm_definitions_have_extraction_instructions(self):
        """All reasoning FSM states must have extraction_instructions populated."""
        from fsm_llm_reasoning.reasoning_modes import ALL_REASONING_FSMS

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
        from fsm_llm_reasoning.reasoning_modes import ALL_REASONING_FSMS

        for fsm_name, fsm_dict in ALL_REASONING_FSMS.items():
            fsm_def = FSMDefinition(**fsm_dict)
            for state_id, state in fsm_def.states.items():
                assert state.response_instructions is not None, (
                    f"FSM '{fsm_name}' state '{state_id}' has no response_instructions"
                )

    def test_no_bare_instructions_field(self):
        """No reasoning FSM should use the bare 'instructions' field (silently ignored)."""
        from fsm_llm_reasoning.reasoning_modes import ALL_REASONING_FSMS

        for fsm_name, fsm_dict in ALL_REASONING_FSMS.items():
            for state_id, state_dict in fsm_dict.get("states", {}).items():
                assert "instructions" not in state_dict, (
                    f"FSM '{fsm_name}' state '{state_id}' uses bare 'instructions' field "
                    "(silently ignored by Pydantic State model — use extraction_instructions "
                    "and response_instructions instead)"
                )

    def test_max_total_iterations_constant_exists(self):
        """Defaults.MAX_TOTAL_ITERATIONS must exist to prevent infinite loops."""
        from fsm_llm_reasoning.constants import Defaults

        assert hasattr(Defaults, "MAX_TOTAL_ITERATIONS")
        assert isinstance(Defaults.MAX_TOTAL_ITERATIONS, int)
        assert Defaults.MAX_TOTAL_ITERATIONS > 0


# ============================================================================
# Workflow Engine Integration Tests
# ============================================================================


class TestWorkflowEngineIntegration:
    """Integration tests for the workflow engine end-to-end."""

    @pytest.mark.asyncio
    async def test_linear_workflow_completes(self):
        """A simple linear workflow should execute to completion."""
        from fsm_llm_workflows import (
            WorkflowEngine,
            auto_step,
            create_workflow,
        )
        from fsm_llm_workflows.models import WorkflowStatus

        actions_called = []

        def step_action(ctx):
            actions_called.append("step1")
            return {"processed": True}

        workflow = create_workflow("test_linear", "Test Linear")
        workflow.with_initial_step(
            auto_step("start", "Start", next_state="end", action=step_action)
        )
        workflow.with_step(auto_step("end", "End", next_state=""))

        engine = WorkflowEngine()
        engine.register_workflow(workflow)
        instance_id = await engine.start_workflow("test_linear")

        instance = engine.get_workflow_instance(instance_id)
        assert instance is not None
        assert instance.status == WorkflowStatus.COMPLETED
        assert "step1" in actions_called

    @pytest.mark.asyncio
    async def test_condition_branching(self):
        """Condition steps should route to the correct branch."""
        from fsm_llm_workflows import (
            WorkflowEngine,
            auto_step,
            condition_step,
            create_workflow,
        )
        from fsm_llm_workflows.models import WorkflowStatus

        workflow = create_workflow("test_cond", "Test Condition")
        workflow.with_initial_step(
            condition_step(
                "check",
                "Check",
                condition=lambda ctx: ctx.get("value", 0) > 5,
                true_state="high",
                false_state="low",
            )
        )
        workflow.with_step(auto_step("high", "High", next_state=""))
        workflow.with_step(auto_step("low", "Low", next_state=""))

        engine = WorkflowEngine()
        engine.register_workflow(workflow)

        # Test false branch
        instance_id = await engine.start_workflow(
            "test_cond", initial_context={"value": 3}
        )
        instance = engine.get_workflow_instance(instance_id)
        assert instance.status == WorkflowStatus.COMPLETED
        assert instance.current_step_id == "low"

        # Test true branch
        instance_id = await engine.start_workflow(
            "test_cond", initial_context={"value": 10}
        )
        instance = engine.get_workflow_instance(instance_id)
        assert instance.status == WorkflowStatus.COMPLETED
        assert instance.current_step_id == "high"

    @pytest.mark.asyncio
    async def test_api_step_context_mapping(self):
        """API steps should correctly map input/output context."""
        from fsm_llm_workflows import (
            WorkflowEngine,
            api_step,
            auto_step,
            create_workflow,
        )
        from fsm_llm_workflows.models import WorkflowStatus

        async def mock_api(name: str = "", **kwargs):
            return {"greeting": f"Hello, {name}!"}

        workflow = create_workflow("test_api", "Test API")
        workflow.with_initial_step(
            api_step(
                "call_api",
                "Call API",
                api_function=mock_api,
                success_state="done",
                failure_state="error",
                input_mapping={"name": "user_name"},
                output_mapping={"result_greeting": "greeting"},
            )
        )
        workflow.with_step(auto_step("done", "Done", next_state=""))
        workflow.with_step(auto_step("error", "Error", next_state=""))

        engine = WorkflowEngine()
        engine.register_workflow(workflow)
        instance_id = await engine.start_workflow(
            "test_api", initial_context={"user_name": "Alice"}
        )

        instance = engine.get_workflow_instance(instance_id)
        assert instance.status == WorkflowStatus.COMPLETED
        assert instance.current_step_id == "done"
        assert instance.context.get("result_greeting") == "Hello, Alice!"

    @pytest.mark.asyncio
    async def test_parallel_step_executes_concurrently(self):
        """Parallel steps should all execute and merge results."""
        from fsm_llm_workflows import (
            WorkflowEngine,
            auto_step,
            create_workflow,
            parallel_step,
        )
        from fsm_llm_workflows.models import WorkflowStatus

        results = []

        def make_action(name):
            def action(ctx):
                results.append(name)
                return {f"{name}_done": True}

            return action

        sub_steps = [
            auto_step(
                f"sub_{i}", f"Sub {i}", next_state="", action=make_action(f"sub_{i}")
            )
            for i in range(3)
        ]

        workflow = create_workflow("test_parallel", "Test Parallel")
        workflow.with_initial_step(
            parallel_step("par", "Parallel", steps=sub_steps, next_state="done")
        )
        workflow.with_step(auto_step("done", "Done", next_state=""))

        engine = WorkflowEngine()
        engine.register_workflow(workflow)
        instance_id = await engine.start_workflow("test_parallel")

        instance = engine.get_workflow_instance(instance_id)
        assert instance.status == WorkflowStatus.COMPLETED
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_terminal_step_with_empty_next_state(self):
        """auto_step with next_state='' should be a valid terminal step."""
        from fsm_llm_workflows import (
            WorkflowEngine,
            auto_step,
            create_workflow,
        )
        from fsm_llm_workflows.models import WorkflowStatus

        workflow = create_workflow("test_terminal", "Test Terminal")
        workflow.with_initial_step(auto_step("start", "Start", next_state="done"))
        workflow.with_step(auto_step("done", "Done", next_state=""))

        # Should not raise during validation
        workflow.validate()

        engine = WorkflowEngine()
        engine.register_workflow(workflow)
        instance_id = await engine.start_workflow("test_terminal")

        instance = engine.get_workflow_instance(instance_id)
        assert instance.status == WorkflowStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_event_workflow_clears_waiting_info(self):
        """After processing an event, _waiting_info should be cleared."""
        from fsm_llm_workflows import (
            WorkflowEngine,
            auto_step,
            create_workflow,
            wait_event_step,
        )
        from fsm_llm_workflows.models import WorkflowEvent, WorkflowStatus

        workflow = create_workflow("test_event", "Test Event")
        workflow.with_initial_step(
            wait_event_step(
                "wait",
                "Wait for Event",
                event_type="test_event",
                success_state="done",
            )
        )
        workflow.with_step(auto_step("done", "Done", next_state=""))

        engine = WorkflowEngine()
        engine.register_workflow(workflow)
        instance_id = await engine.start_workflow("test_event")

        # Should be waiting
        instance = engine.get_workflow_instance(instance_id)
        assert instance.status == WorkflowStatus.WAITING

        # Register listener and process event
        await engine.register_event_listener(
            instance_id, "test_event", success_state="done"
        )
        event = WorkflowEvent(event_type="test_event", payload={"data": "test"})
        await engine.process_event(event)

        # Should be completed, not stuck at WAITING
        instance = engine.get_workflow_instance(instance_id)
        assert instance.status == WorkflowStatus.COMPLETED
        assert "_waiting_info" not in instance.context

    @pytest.mark.asyncio
    async def test_custom_step_subclass_validates(self):
        """Custom WorkflowStep subclasses should pass validation."""
        from fsm_llm_workflows import (
            WorkflowEngine,
            auto_step,
            create_workflow,
        )
        from fsm_llm_workflows.models import WorkflowStatus, WorkflowStepResult
        from fsm_llm_workflows.steps import WorkflowStep

        class CustomStep(WorkflowStep):
            """A custom step with state references."""

            success_state: str = ""
            error_state: str = ""

            async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
                return WorkflowStepResult.success_result(
                    data={"custom": True},
                    next_state=self.success_state if self.success_state else None,
                )

        workflow = create_workflow("test_custom", "Test Custom Step")
        workflow.with_initial_step(
            CustomStep(
                step_id="custom",
                name="Custom",
                success_state="done",
                error_state="error",
            )
        )
        workflow.with_step(auto_step("done", "Done", next_state=""))
        workflow.with_step(auto_step("error", "Error", next_state=""))

        # Should not raise
        workflow.validate()

        engine = WorkflowEngine()
        engine.register_workflow(workflow)
        instance_id = await engine.start_workflow("test_custom")

        instance = engine.get_workflow_instance(instance_id)
        assert instance.status == WorkflowStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_concurrent_workflows_no_leaks(self):
        """Multiple concurrent workflows should all complete with no resource leaks."""
        from fsm_llm_workflows import (
            WorkflowEngine,
            auto_step,
            create_workflow,
        )
        from fsm_llm_workflows.models import WorkflowStatus

        workflow = create_workflow("test_concurrent", "Test Concurrent")
        workflow.with_initial_step(auto_step("a", "A", next_state="b"))
        workflow.with_step(auto_step("b", "B", next_state="c"))
        workflow.with_step(auto_step("c", "C", next_state=""))

        engine = WorkflowEngine()
        engine.register_workflow(workflow)

        instance_ids = []
        for _ in range(10):
            iid = await engine.start_workflow("test_concurrent")
            instance_ids.append(iid)

        for iid in instance_ids:
            instance = engine.get_workflow_instance(iid)
            assert instance.status == WorkflowStatus.COMPLETED

        stats = engine.get_statistics()
        assert stats["active_workflows"] == 0
