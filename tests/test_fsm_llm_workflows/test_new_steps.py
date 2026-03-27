from __future__ import annotations

"""
Tests for new workflow step types: AgentStep, RetryStep, SwitchStep.
Also tests for workflow engine improvements (instance cleanup, event warnings).
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from fsm_llm_workflows.models import WorkflowStepResult
from fsm_llm_workflows.steps import (
    AgentStep,
    AutoTransitionStep,
    RetryStep,
    SwitchStep,
)

# ---------------------------------------------------------------
# SwitchStep
# ---------------------------------------------------------------


class TestSwitchStep:
    """Test n-way branching on context keys."""

    @pytest.fixture
    def switch(self):
        return SwitchStep(
            step_id="route",
            name="Route",
            key="intent",
            cases={"buy": "checkout", "browse": "catalog", "help": "support"},
            default_state="fallback",
        )

    def test_routes_to_matching_case(self, switch):
        result = asyncio.get_event_loop().run_until_complete(
            switch.execute({"intent": "buy"})
        )
        assert result.success is True
        assert result.next_state == "checkout"

    def test_routes_to_default(self, switch):
        result = asyncio.get_event_loop().run_until_complete(
            switch.execute({"intent": "unknown"})
        )
        assert result.success is True
        assert result.next_state == "fallback"

    def test_missing_key_uses_default(self, switch):
        result = asyncio.get_event_loop().run_until_complete(switch.execute({}))
        assert result.success is True
        assert result.next_state == "fallback"

    def test_no_default_fails(self):
        switch = SwitchStep(
            step_id="route",
            name="Route",
            key="intent",
            cases={"buy": "checkout"},
        )
        result = asyncio.get_event_loop().run_until_complete(
            switch.execute({"intent": "unknown"})
        )
        assert result.success is False

    def test_numeric_value_converted_to_string(self):
        switch = SwitchStep(
            step_id="route",
            name="Route",
            key="level",
            cases={"1": "low", "2": "high"},
            default_state="unknown",
        )
        result = asyncio.get_event_loop().run_until_complete(
            switch.execute({"level": 1})
        )
        assert result.next_state == "low"


# ---------------------------------------------------------------
# RetryStep
# ---------------------------------------------------------------


class TestRetryStep:
    """Test retry logic with backoff."""

    def test_succeeds_on_first_try(self):
        inner = AutoTransitionStep(
            step_id="inner",
            name="Inner",
            next_state="done",
        )
        step = RetryStep(
            step_id="retry",
            name="Retry",
            step=inner,
            max_retries=3,
        )
        result = asyncio.get_event_loop().run_until_complete(step.execute({}))
        assert result.success is True
        assert result.next_state == "done"

    def test_retries_on_failure(self):
        call_count = {"n": 0}

        class _FailThenSucceed:
            step_id = "inner"
            name = "Inner"

            async def execute(self, context):
                call_count["n"] += 1
                if call_count["n"] < 3:
                    return WorkflowStepResult.failure_result(
                        error="failed", next_state="err"
                    )
                return WorkflowStepResult.success_result(
                    data={"ok": True}, next_state="done"
                )

        step = RetryStep(
            step_id="retry",
            name="Retry",
            step=_FailThenSucceed(),
            max_retries=3,
            backoff_factor=0.01,  # Fast for testing
        )
        result = asyncio.get_event_loop().run_until_complete(step.execute({}))
        assert result.success is True
        assert call_count["n"] == 3

    def test_exhausts_retries(self):
        class _AlwaysFails:
            step_id = "inner"

            async def execute(self, context):
                return WorkflowStepResult.failure_result(
                    error="always fails", next_state="err"
                )

        step = RetryStep(
            step_id="retry",
            name="Retry",
            step=_AlwaysFails(),
            max_retries=2,
            backoff_factor=0.01,
        )
        result = asyncio.get_event_loop().run_until_complete(step.execute({}))
        assert result.success is False


# ---------------------------------------------------------------
# AgentStep
# ---------------------------------------------------------------


class TestAgentStep:
    """Test running agents as workflow steps."""

    def test_basic_execution(self):
        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.answer = "The answer is 42"
        mock_result.success = True
        mock_result.final_context = {"key": "value"}
        mock_agent.run.return_value = mock_result

        step = AgentStep(
            step_id="research",
            name="Research",
            agent=mock_agent,
            task_template="Research {topic}",
            success_state="analyze",
            context_mapping={"findings": "key"},
        )

        result = asyncio.get_event_loop().run_until_complete(
            step.execute({"topic": "AI"})
        )

        assert result.success is True
        assert result.next_state == "analyze"
        mock_agent.run.assert_called_once_with("Research AI")
        assert result.data["findings"] == "value"
        assert result.data["agent_answer"] == "The answer is 42"

    def test_context_mapping_answer_key(self):
        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.answer = "Summary text"
        mock_result.success = True
        mock_result.final_context = {"task": "Summarize X"}  # non-empty so mapping runs
        mock_agent.run.return_value = mock_result

        step = AgentStep(
            step_id="summarize",
            name="Summarize",
            agent=mock_agent,
            success_state="next",
            context_mapping={"summary": "answer"},
        )

        result = asyncio.get_event_loop().run_until_complete(
            step.execute({"task": "Summarize X"})
        )

        assert result.data["summary"] == "Summary text"

    def test_agent_failure(self):
        mock_agent = MagicMock()
        mock_agent.run.side_effect = RuntimeError("Agent crashed")

        step = AgentStep(
            step_id="fail",
            name="Fail",
            agent=mock_agent,
            success_state="next",
            error_state="error",
        )

        result = asyncio.get_event_loop().run_until_complete(
            step.execute({"task": "Do something"})
        )

        assert result.success is False
        assert result.next_state == "error"

    def test_missing_template_key(self):
        mock_agent = MagicMock()

        step = AgentStep(
            step_id="fail",
            name="Fail",
            agent=mock_agent,
            task_template="Research {missing_key}",
            success_state="next",
            error_state="error",
        )

        result = asyncio.get_event_loop().run_until_complete(step.execute({}))

        assert result.success is False


# ---------------------------------------------------------------
# DSL functions
# ---------------------------------------------------------------


class TestNewDSLFunctions:
    """Test DSL factory functions for new step types."""

    def test_agent_step_factory(self):
        from fsm_llm_workflows.dsl import agent_step

        mock_agent = MagicMock()
        step = agent_step(
            "research",
            "Research",
            mock_agent,
            task_template="Research {topic}",
            success_state="analyze",
        )
        assert isinstance(step, AgentStep)
        assert step.step_id == "research"

    def test_retry_step_factory(self):
        from fsm_llm_workflows.dsl import retry_step

        inner = AutoTransitionStep(step_id="inner", name="Inner", next_state="done")
        step = retry_step("retry", "Retry", inner, max_retries=5)
        assert isinstance(step, RetryStep)
        assert step.max_retries == 5

    def test_switch_step_factory(self):
        from fsm_llm_workflows.dsl import switch_step

        step = switch_step(
            "route",
            "Route",
            "intent",
            cases={"buy": "checkout"},
            default_state="fallback",
        )
        assert isinstance(step, SwitchStep)
        assert step.key == "intent"


# ---------------------------------------------------------------
# Workflow Engine: instance cleanup
# ---------------------------------------------------------------


class TestWorkflowInstanceCleanup:
    """Test instance removal and auto-purge."""

    def test_remove_terminal_instance(self):
        from fsm_llm_workflows.engine import WorkflowEngine
        from fsm_llm_workflows.models import WorkflowInstance, WorkflowStatus

        engine = WorkflowEngine()
        instance = WorkflowInstance(
            instance_id="test-1",
            workflow_id="wf-1",
            current_step_id="done",
            status=WorkflowStatus.COMPLETED,
        )
        engine.workflow_instances["test-1"] = instance

        assert engine.remove_instance("test-1") is True
        assert "test-1" not in engine.workflow_instances

    def test_cannot_remove_active_instance(self):
        from fsm_llm_workflows.engine import WorkflowEngine
        from fsm_llm_workflows.models import WorkflowInstance, WorkflowStatus

        engine = WorkflowEngine()
        instance = WorkflowInstance(
            instance_id="test-1",
            workflow_id="wf-1",
            current_step_id="running",
            status=WorkflowStatus.RUNNING,
        )
        engine.workflow_instances["test-1"] = instance

        assert engine.remove_instance("test-1") is False
        assert "test-1" in engine.workflow_instances

    def test_remove_nonexistent_instance(self):
        from fsm_llm_workflows.engine import WorkflowEngine

        engine = WorkflowEngine()
        assert engine.remove_instance("nonexistent") is False

    def test_max_completed_instances_purge(self):
        from datetime import datetime, timezone

        from fsm_llm_workflows.engine import WorkflowEngine
        from fsm_llm_workflows.models import WorkflowInstance, WorkflowStatus

        engine = WorkflowEngine(max_completed_instances=2)

        # Add 4 completed instances with different timestamps
        for i in range(4):
            instance = WorkflowInstance(
                instance_id=f"test-{i}",
                workflow_id="wf-1",
                current_step_id="done",
                status=WorkflowStatus.COMPLETED,
                completed_at=datetime(2026, 1, 1 + i, tzinfo=timezone.utc),
                updated_at=datetime(2026, 1, 1 + i, tzinfo=timezone.utc),
            )
            engine.workflow_instances[f"test-{i}"] = instance

        engine._purge_oldest_terminal_instances()

        # Should keep only 2 newest
        assert len(engine.workflow_instances) == 2
        assert "test-2" in engine.workflow_instances
        assert "test-3" in engine.workflow_instances
        assert "test-0" not in engine.workflow_instances
        assert "test-1" not in engine.workflow_instances

    def test_no_purge_when_limit_none(self):
        from fsm_llm_workflows.engine import WorkflowEngine
        from fsm_llm_workflows.models import WorkflowInstance, WorkflowStatus

        engine = WorkflowEngine(max_completed_instances=None)

        for i in range(10):
            engine.workflow_instances[f"test-{i}"] = WorkflowInstance(
                instance_id=f"test-{i}",
                workflow_id="wf-1",
                current_step_id="done",
                status=WorkflowStatus.COMPLETED,
            )

        engine._purge_oldest_terminal_instances()
        assert len(engine.workflow_instances) == 10
