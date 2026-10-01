from __future__ import annotations

"""Tests for fsm_llm.agents.prompts module."""

import pytest

from fsm_llm.agents.prompts import (
    build_conclude_response_instructions,
    build_debate_conclude_response_instructions,
    build_plan_steps_instructions,
    build_think_extraction_instructions,
)
from fsm_llm.agents.tools import ToolRegistry


def _dummy(params):
    return "result"


def _make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_function(
        _dummy,
        name="search",
        description="Search the web",
        parameter_schema={
            "properties": {"query": {"type": "string", "description": "Search query"}}
        },
    )
    registry.register_function(
        _dummy,
        name="calc",
        description="Calculate expression",
    )
    return registry


class TestPromptBuilders:
    """Tests for prompt building functions."""

    def test_think_extraction_includes_tools(self):
        registry = _make_registry()
        instructions = build_think_extraction_instructions(registry)

        assert "search" in instructions
        assert "calc" in instructions
        assert "tool_name" in instructions
        assert "tool_input" in instructions
        assert "should_terminate" in instructions

    def test_think_extraction_includes_observation_guidance(self):
        registry = _make_registry()
        instructions = build_think_extraction_instructions(
            registry, include_observations=True
        )
        assert "previous observations" in instructions.lower()

    def test_think_extraction_without_observations(self):
        registry = _make_registry()
        instructions = build_think_extraction_instructions(
            registry, include_observations=False
        )
        assert "previous observations" not in instructions.lower()

    def test_conclude_response_instructions(self):
        instructions = build_conclude_response_instructions()
        assert "final" in instructions.lower()
        # The task re-anchor stays; an honest answer on thin evidence is asked for.
        assert "ORIGINAL task" in instructions
        assert "could not be determined" in instructions

    @pytest.mark.parametrize(
        "build",
        [
            build_conclude_response_instructions,
            build_debate_conclude_response_instructions,
        ],
    )
    def test_conclude_instructions_name_no_turn_mechanics(self, build):
        # plan 07ad3f8c step 10 (D-031): wording that names a prompt, a signal
        # or "proceeding" made a small model answer the turn, not the task.
        instructions = build().lower()
        for word in ("continue", "signal", "proceed", "prompt", "user"):
            assert word not in instructions, word


class TestPlanStepsInstructions:
    """plan 07ad3f8c step 24.1 (D-055): the PlanExecute planner adds no
    tool-less step and no confirm/wait step, in ``plan`` and ``replan``.

    Live (qwen3.5:4b) the planner added "Combine"/"Compare"/"Synthesize"
    steps and steps that wait for confirmation, each costing execute turns.
    "One tool call per step" is NOT the fix: it split every item into its own
    step. No wording may name a loop, a signal or a message to go on (D-031).
    """

    @pytest.mark.parametrize("replan", [False, True])
    def test_with_tools_forbids_tool_less_and_confirm_steps(self, replan):
        text = build_plan_steps_instructions(_make_registry(), replan=replan)
        assert "Add no step that needs no tool" in text
        assert "unless the task asks for one" in text
        assert "final answer is written from the step results" in text
        assert "Add no step that confirms, waits for or asks for anything." in text
        # The rules come before the tool list.
        assert text.index("Add no step") < text.index("search")

    @pytest.mark.parametrize("replan", [False, True])
    def test_without_tools_keeps_only_the_confirm_rule(self, replan):
        # Tool-less PlanExecute: every step needs no tool, so that rule
        # would forbid every plan.
        text = build_plan_steps_instructions(None, replan=replan)
        assert "needs no tool" not in text
        assert "Add no step that confirms, waits for or asks for anything." in text

    @pytest.mark.parametrize("registry", [None, "tools"])
    @pytest.mark.parametrize("replan", [False, True])
    def test_names_no_turn_mechanics_or_per_call_steps(self, registry, replan):
        reg = _make_registry() if registry else None
        text = build_plan_steps_instructions(reg, replan=replan)
        rules = text.split("\n\n")[0].lower()
        for phrase in (
            "continue",
            "signal",
            "loop",
            "user",
            "message",
            "one tool call per step",
            "exactly",
        ):
            assert phrase not in rules, phrase

    def test_plan_and_replan_states_carry_the_rules(self):
        from fsm_llm.agents.constants import ContextKeys, PlanExecuteStates
        from fsm_llm.agents.fsm_definitions import build_plan_execute_fsm

        fsm = build_plan_execute_fsm(_make_registry(), task_description="t")
        for state in (PlanExecuteStates.PLAN, PlanExecuteStates.REPLAN):
            field = next(
                f
                for f in fsm["states"][state]["field_extractions"]
                if f["field_name"] == ContextKeys.PLAN_STEPS
            )
            assert (
                "Add no step that needs no tool" in (field["extraction_instructions"])
            ), state
            assert "Add no step that confirms" in field["extraction_instructions"]
