from __future__ import annotations

"""Tests for the degenerate-completion guard (DECISION
plan_2026-05-30_26c9510a/D-001) — a run with neither an answer key nor a tool
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
        assert (
            BaseAgent._completion_is_real({}, _trace(), None) is False
        )

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
