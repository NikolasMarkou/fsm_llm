from __future__ import annotations

"""Tests for the B4 under-call fix (re-merge-immune FSM conclude guard).

Plan: plan_2026-05-30_5598b755 / D-004.

Root cause: in the ReAct FSM, ``think -> conclude`` (priority 10) and
``act -> conclude`` (priority 1) fired on ``should_terminate == True`` alone, so
a weak model that set ``should_terminate=true`` on its first think step without a
tool concluded immediately with an empty trace. A CONTEXT_UPDATE handler cannot
fix this (the transition evaluator re-merges the raw extracted should_terminate
over any handler reset -- transition_evaluator.py:162-169). The fix gates both
conclude edges on framework-only context vars the model does NOT extract:
``should_terminate AND (observation_count > 0 OR max_iterations_reached)``. The
execute_tool stall-detector sets ``max_iterations_reached`` so genuinely
tool-free turns still terminate.
"""

from typing import Any

from fsm_llm.definitions import (
    FieldExtractionRequest,
    FieldExtractionResponse,
    ResponseGenerationRequest,
    ResponseGenerationResponse,
)
from fsm_llm.llm import LLMInterface
from fsm_llm_agents.constants import ContextKeys
from fsm_llm_agents.definitions import AgentConfig
from fsm_llm_agents.fsm_definitions import build_react_fsm
from fsm_llm_agents.react import ReactAgent
from fsm_llm_agents.tools import ToolRegistry


def _search(params):
    """Mock search tool."""
    return f"Result for: {params.get('query', '')}"


def _make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_function(
        _search,
        name="search",
        description="Search for information",
        parameter_schema={"properties": {"query": {"type": "string"}}},
    )
    return registry


# --------------------------------------------------------------------------- #
# (c) Structural test on the generated FSM definition (no LLM)                 #
# --------------------------------------------------------------------------- #


def _conclude_logic(state: dict[str, Any]) -> dict[str, Any]:
    for t in state["transitions"]:
        if t["target_state"] == "conclude":
            return t["conditions"][0]["logic"]
    raise AssertionError("no conclude transition found")


class TestConcludeGuardStructure:
    def test_think_and_act_conclude_require_evidence_or_forced(self):
        fsm = build_react_fsm(_make_registry())
        for state_name in ("think", "act"):
            logic = _conclude_logic(fsm["states"][state_name])
            # top-level AND of should_terminate and an OR clause
            assert "and" in logic, f"{state_name} conclude must be an AND"
            clauses = logic["and"]
            text = str(clauses)
            assert "should_terminate" in text
            assert "observation_count" in text, (
                f"{state_name} conclude must gate on observation_count "
                "(re-merge-immune guard missing)"
            )
            assert "max_iterations_reached" in text, (
                f"{state_name} conclude must allow forced termination"
            )


# --------------------------------------------------------------------------- #
# (a) Integration: a perpetual under-caller must route through act + terminate #
# --------------------------------------------------------------------------- #


class _AlwaysUnderCallLLM(LLMInterface):
    """Always tries to terminate on think with no tool (the B4 under-call)."""

    def __init__(self) -> None:
        self.model = "mock-model"

    def extract_field(self, request: FieldExtractionRequest) -> FieldExtractionResponse:
        values: dict[str, Any] = {
            ContextKeys.TOOL_NAME: ContextKeys.NO_TOOL,
            ContextKeys.TOOL_INPUT: {},
            "reasoning": "I can answer from memory.",
            ContextKeys.SHOULD_TERMINATE: True,
            ContextKeys.FINAL_ANSWER: "Best-effort answer.",
        }
        value = values.get(request.field_name)
        return FieldExtractionResponse(
            field_name=request.field_name,
            value=value,
            confidence=1.0 if value is not None else 0.0,
            reasoning="mock",
            is_valid=value is not None,
        )

    def generate_response(
        self, request: ResponseGenerationRequest
    ) -> ResponseGenerationResponse:
        return ResponseGenerationResponse(
            message="Best-effort answer.", message_type="response", reasoning="mock"
        )


class _ToolThenTerminateLLM(LLMInterface):
    """Selects the search tool on the first think, then terminates."""

    def __init__(self) -> None:
        self.model = "mock-model"
        self._think_count = 0  # incremented once per think cycle (on tool_name)

    def extract_field(self, request: FieldExtractionRequest) -> FieldExtractionResponse:
        fname = request.field_name
        # tool_name is the first required key extracted on each think cycle.
        if fname == ContextKeys.TOOL_NAME:
            self._think_count += 1
        if self._think_count <= 1:
            mapping = {
                ContextKeys.TOOL_NAME: "search",
                ContextKeys.TOOL_INPUT: {"query": "capital of France"},
                "reasoning": "I should search.",
                ContextKeys.SHOULD_TERMINATE: False,
            }
        else:
            mapping = {
                ContextKeys.TOOL_NAME: ContextKeys.NO_TOOL,
                ContextKeys.TOOL_INPUT: {},
                "reasoning": "I have the answer now.",
                ContextKeys.SHOULD_TERMINATE: True,
                ContextKeys.FINAL_ANSWER: "Paris.",
            }
        value = mapping.get(fname)
        return FieldExtractionResponse(
            field_name=fname,
            value=value,
            confidence=1.0 if value is not None else 0.0,
            reasoning="mock",
            is_valid=value is not None,
        )

    def generate_response(
        self, request: ResponseGenerationRequest
    ) -> ResponseGenerationResponse:
        return ResponseGenerationResponse(
            message="ok", message_type="response", reasoning="mock"
        )


class TestUnderCallRoutedToActAndTerminates:
    def test_perpetual_under_caller_terminates_via_stall(self):
        """Must NOT conclude on turn 1; must enter act and still terminate
        (stall-detector) — no BudgetExhaustedError, no hang."""
        agent = ReactAgent(
            tools=_make_registry(),
            config=AgentConfig(max_iterations=10, timeout_seconds=30.0),
            llm_interface=_AlwaysUnderCallLLM(),
        )
        result = agent.run("What is the capital of France?")

        # Did not conclude on the first think (the B4 regression would give 1).
        assert result.iterations_used >= 2, (
            "Agent concluded on the first think — B4 under-call regression is back."
        )
        # Terminated within budget (hard ceiling = max_iterations * 3 = 30).
        assert result.iterations_used <= 30
        assert result.answer  # a non-empty answer was produced


class TestNormalToolRunStillConcludes:
    def test_tool_run_then_terminate_concludes_with_observations(self):
        """A run that uses a tool then terminates must conclude normally."""
        agent = ReactAgent(
            tools=_make_registry(),
            config=AgentConfig(max_iterations=10, timeout_seconds=30.0),
            llm_interface=_ToolThenTerminateLLM(),
        )
        result = agent.run("What is the capital of France?")

        assert "search" in result.tools_used, "the tool should have executed"
        assert result.success, "a tool-backed run should report success"
        assert result.iterations_used <= 30
