from __future__ import annotations

"""Regression tests for the plan_execute honesty hole (DECISION
plan_2026-05-31_f08da86d/D-001 [STALE]).

The plan_execute step tracker appends entries to ``step_results`` that carry a
``success`` flag, so they hit the ``_has_execution_evidence`` "list of dicts all
carrying success" branch (``base.py:364``). Before the fix, ``success`` was
``not step_failed`` and ``step_failed`` defaults False, so a weak model that
NARRATES a step (zero tool calls) produced an all-``success=True`` entry and the
guard reported ``success=True`` on filler. The fix ties the per-entry flag to a
REAL tool execution via ``ContextKeys.TOOL_STATUS == "success"`` (written by
``AgentHandlers.execute_tool`` only when a tool genuinely ran)."""

import pytest

from fsm_llm.definitions import (
    DataExtractionResponse,
    FieldExtractionResponse,
    ResponseGenerationResponse,
)
from fsm_llm.llm import LLMInterface
from fsm_llm_agents.base import BaseAgent
from fsm_llm_agents.constants import ContextKeys
from fsm_llm_agents.definitions import AgentConfig, AgentTrace
from fsm_llm_agents.plan_execute import PlanExecuteAgent
from fsm_llm_agents.tools import ToolRegistry


def _trace() -> AgentTrace:
    return AgentTrace(total_iterations=1)


def _check_result(context: dict) -> dict:
    """Invoke the real plan_execute step-result checker against a context."""
    agent = PlanExecuteAgent()
    checker = agent._make_result_checker()
    return checker(context)


class TestPlanExecuteStepResultSuccessFlag:
    def test_narrated_step_no_tool_yields_success_false(self):
        # Model narrated a step with zero tool calls: tool_status absent (or
        # "skipped"). The appended entry must carry success=False.
        ctx = {
            ContextKeys.PLAN_STEPS: ["step 1"],
            ContextKeys.CURRENT_STEP_INDEX: 0,
            "step_result": "I researched the topic and summarized it.",
            # no TOOL_STATUS -> narrated only
        }
        updates = _check_result(ctx)
        entries = updates[ContextKeys.STEP_RESULTS]
        assert entries == [
            {
                "step_index": 0,
                "result": "I researched the topic and summarized it.",
                "success": False,
            }
        ]

    def test_narrated_step_skipped_status_yields_success_false(self):
        ctx = {
            ContextKeys.PLAN_STEPS: ["step 1"],
            ContextKeys.CURRENT_STEP_INDEX: 0,
            "step_result": "Narrated filler.",
            ContextKeys.TOOL_STATUS: "skipped",
        }
        updates = _check_result(ctx)
        assert updates[ContextKeys.STEP_RESULTS][0]["success"] is False

    def test_real_tool_step_yields_success_true(self):
        ctx = {
            ContextKeys.PLAN_STEPS: ["step 1"],
            ContextKeys.CURRENT_STEP_INDEX: 0,
            "step_result": "Tool returned the value 42.",
            ContextKeys.TOOL_STATUS: "success",
        }
        updates = _check_result(ctx)
        assert updates[ContextKeys.STEP_RESULTS][0]["success"] is True

    def test_failed_tool_step_yields_success_false(self):
        # A tool ran but failed (step_failed True) -> not a successful step.
        ctx = {
            ContextKeys.PLAN_STEPS: ["step 1"],
            ContextKeys.CURRENT_STEP_INDEX: 0,
            "step_result": "Tool error.",
            ContextKeys.TOOL_STATUS: "failed",
            ContextKeys.STEP_FAILED: True,
        }
        updates = _check_result(ctx)
        assert updates[ContextKeys.STEP_RESULTS][0]["success"] is False


class TestPlanExecuteEvidenceGuard:
    """End-to-end through _completion_is_real: a narrated-only run must report
    success=False; a real-tool run must report evidence True."""

    def test_narrated_only_run_is_not_real(self):
        ctx = {
            ContextKeys.PLAN_STEPS: ["step 1"],
            ContextKeys.CURRENT_STEP_INDEX: 0,
            "step_result": "I did step 1.",
        }
        final_ctx = {
            ContextKeys.STEP_RESULTS: _check_result(ctx)[ContextKeys.STEP_RESULTS]
        }
        # all entries success=False -> _has_execution_evidence False
        assert (
            BaseAgent._completion_is_real(
                final_ctx, _trace(), None, [ContextKeys.STEP_RESULTS]
            )
            is False
        )

    def test_real_tool_run_is_real(self):
        ctx = {
            ContextKeys.PLAN_STEPS: ["step 1"],
            ContextKeys.CURRENT_STEP_INDEX: 0,
            "step_result": "Tool returned 42.",
            ContextKeys.TOOL_STATUS: "success",
        }
        final_ctx = {
            ContextKeys.STEP_RESULTS: _check_result(ctx)[ContextKeys.STEP_RESULTS]
        }
        assert BaseAgent._completion_is_real(
            final_ctx, _trace(), None, [ContextKeys.STEP_RESULTS]
        )

    def test_none_mode_unchanged(self):
        # Regression: execution_evidence_keys=None keeps the original
        # has_answer_key-OR-tools_executed behavior, byte-identical.
        assert BaseAgent._completion_is_real({}, _trace(), None) is False
        assert BaseAgent._completion_is_real(
            {ContextKeys.FINAL_ANSWER: "answer"}, _trace(), None
        )


class _StepToolLLM(LLMInterface):
    """Plan of ``n`` steps; on each execute_step turn the model selects the
    working ``add`` tool with this step's own input (``a`` = step number)."""

    def __init__(self, n_steps: int) -> None:
        self.model = "mock-model"
        self.n_steps = n_steps
        self.step_turns = 0

    def extract_bulk_data(self, request):
        if "Execute the current plan step" not in request.system_prompt:
            return DataExtractionResponse(extracted_data={})
        self.step_turns += 1
        k = self.step_turns
        return DataExtractionResponse(
            extracted_data={
                "step_result": f"sum {k}",
                ContextKeys.TOOL_NAME: "add",
                ContextKeys.TOOL_INPUT: {"a": k, "b": 10},
            }
        )

    def extract_field(self, request):
        value = (
            [f"Add step {i + 1}" for i in range(self.n_steps)]
            if request.field_name == ContextKeys.PLAN_STEPS
            else None
        )
        return FieldExtractionResponse(
            field_name=request.field_name,
            value=value,
            confidence=1.0 if value is not None else 0.0,
            reasoning="mock",
            is_valid=value is not None,
        )

    def generate_response(self, request):
        return ResponseGenerationResponse(
            message="done", message_type="response", reasoning="mock"
        )


class TestPlanExecuteRecordsTheRealToolOutcome:
    """DECISION plan-2026-09-24T091842-c1d5bfbc/D-024: each step's tool runs
    with the tool selection extracted for THAT step, and ``check_result``
    records that call's outcome. At 37b38f0 the executor ran on execute_step
    entry (before the step's selection was extracted, so step 1 never called a
    tool and step k ran step k-1's input) and every entry recorded failure."""

    @pytest.mark.parametrize("n_steps", [1, 2])
    def test_working_tool_steps_record_success(self, n_steps):
        calls: list[tuple[int, int]] = []

        def add(a: int, b: int) -> int:
            calls.append((a, b))
            return a + b

        registry = ToolRegistry()
        registry.register_function(
            add,
            parameter_schema={
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
        )
        agent = PlanExecuteAgent(
            tools=registry,
            config=AgentConfig(max_iterations=12, timeout_seconds=30.0),
            llm_interface=_StepToolLLM(n_steps),
        )
        result = agent.run("Add the numbers")

        assert calls == [(k, 10) for k in range(1, n_steps + 1)]
        entries = result.final_context[ContextKeys.STEP_RESULTS]
        assert [e["step_index"] for e in entries] == list(range(n_steps))
        assert all(e["success"] is True for e in entries), entries
        assert result.success is True
