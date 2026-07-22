from __future__ import annotations

"""Tests for the degenerate-completion guard (DECISION
plan_2026-05-30_26c9510a/D-001 [STALE]) — a run with neither an answer key nor a tool
call must report success=False instead of leaking planner prose."""

from fsm_llm_agents.base import BaseAgent
from fsm_llm_agents.constants import ContextKeys
from fsm_llm_agents.definitions import AgentTrace, ToolCall


def _trace(*tool_names: str) -> AgentTrace:
    return AgentTrace(
        tool_calls=[ToolCall(tool_name=n, parameters={}) for n in tool_names],
        total_iterations=1,
    )


class TestCompletionGuard:
    def test_no_answer_key_no_tools_is_not_real(self):
        # The planner/orchestrator leak: prose only, nothing executed.
        assert BaseAgent._completion_is_real({}, _trace(), None) is False

    def test_final_answer_key_is_real(self):
        assert BaseAgent._completion_is_real(
            {ContextKeys.FINAL_ANSWER: "the result"}, _trace(), None
        )

    def test_blank_final_answer_is_not_real(self):
        assert (
            BaseAgent._completion_is_real(
                {ContextKeys.FINAL_ANSWER: "   "}, _trace(), None
            )
            is False
        )

    def test_extra_answer_key_is_real(self):
        # Pattern-specific answer key (e.g. debate JUDGE_VERDICT) counts.
        assert BaseAgent._completion_is_real(
            {"judge_verdict": "B wins"}, _trace(), ["judge_verdict"]
        )

    def test_tool_call_alone_is_real(self):
        # Tool ran but no explicit final answer — still genuine work.
        assert BaseAgent._completion_is_real({}, _trace("search"), None)


class TestPlannerExecutionEvidence:
    """Planner mode (DECISION plan_2026-05-31_cb91a9d5/D-001 [STALE]): when
    execution_evidence_keys is supplied, success requires REAL execution
    evidence — an answer key and/or a `delegate` control-action ToolCall are
    NOT sufficient. Covers orchestrator/rewoo/plan_execute B6 filler-success."""

    def test_orchestrator_placeholder_workers_is_not_real(self):
        # No worker_factory: only placeholder results (success=False). The
        # synthesize state set final_answer AND `delegate` is a fake ToolCall —
        # both defeat the legacy guard, but there is no real execution.
        ctx = {
            ContextKeys.FINAL_ANSWER: "I need the results, please provide them.",
            ContextKeys.WORKER_RESULTS: [
                {
                    "subtask": "x",
                    "answer": "[Pending LLM processing: x]",
                    "success": False,
                }
            ],
        }
        assert (
            BaseAgent._completion_is_real(
                ctx, _trace("delegate"), None, [ContextKeys.WORKER_RESULTS]
            )
            is False
        )

    def test_orchestrator_successful_worker_is_real(self):
        ctx = {
            ContextKeys.FINAL_ANSWER: "Paris vs Tokyo: Paris is longer.",
            ContextKeys.WORKER_RESULTS: [
                {"subtask": "x", "answer": "Paris", "success": True}
            ],
        }
        assert BaseAgent._completion_is_real(
            ctx, _trace("delegate"), None, [ContextKeys.WORKER_RESULTS]
        )

    def test_rewoo_empty_evidence_is_not_real(self):
        ctx = {ContextKeys.FINAL_ANSWER: "synthesized prose", ContextKeys.EVIDENCE: {}}
        assert (
            BaseAgent._completion_is_real(ctx, _trace(), None, [ContextKeys.EVIDENCE])
            is False
        )

    def test_rewoo_nonempty_evidence_is_real(self):
        ctx = {ContextKeys.EVIDENCE: {"E1": "17*23 = 391"}}
        assert BaseAgent._completion_is_real(
            ctx, _trace(), None, [ContextKeys.EVIDENCE]
        )

    def test_plan_execute_no_steps_is_not_real(self):
        ctx = {
            ContextKeys.FINAL_ANSWER: "here is the answer",
            ContextKeys.STEP_RESULTS: [],
        }
        assert (
            BaseAgent._completion_is_real(
                ctx, _trace(), None, [ContextKeys.STEP_RESULTS]
            )
            is False
        )

    def test_plan_execute_executed_steps_is_real(self):
        # step_results entries have no `success` flag → non-empty list is evidence.
        ctx = {ContextKeys.STEP_RESULTS: ["step 1 done", "step 2 done"]}
        assert BaseAgent._completion_is_real(
            ctx, _trace(), None, [ContextKeys.STEP_RESULTS]
        )

    def test_has_execution_evidence_dict_vs_list(self):
        assert BaseAgent._has_execution_evidence({"evidence": {"E1": 1}}, ["evidence"])
        assert not BaseAgent._has_execution_evidence({"evidence": {}}, ["evidence"])
        assert not BaseAgent._has_execution_evidence({"k": []}, ["k"])
        # all-failed dict entries → not evidence
        assert not BaseAgent._has_execution_evidence(
            {"wr": [{"success": False}, {"success": False}]}, ["wr"]
        )
