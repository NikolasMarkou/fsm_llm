"""The model cannot forge ``max_iterations_reached`` (review P2-W2).

DECISION plan-2026-09-24T091842-c1d5bfbc/D-007: the flag is seeded False in
``BaseAgent._init_context``, so core bulk extraction (which fills only unset
keys in an agent FSM) cannot write it. Only the limiter and stall handlers
write True. Pre-fix, a bulk ``max_iterations_reached: true`` concluded React
and Reflexion on turn 1 with zero tools and made maker_checker ship DRAFT-1
at budget 10 after one rejected check.
"""

from __future__ import annotations

from typing import Any

import pytest

from fsm_llm.definitions import (
    DataExtractionResponse,
    FieldExtractionRequest,
    FieldExtractionResponse,
    ResponseGenerationRequest,
    ResponseGenerationResponse,
)
from fsm_llm.llm import LLMInterface
from fsm_llm_agents.constants import ContextKeys
from fsm_llm_agents.definitions import AgentConfig
from fsm_llm_agents.maker_checker import MakerCheckerAgent
from fsm_llm_agents.react import ReactAgent
from fsm_llm_agents.reflexion import ReflexionAgent
from fsm_llm_agents.tools import ToolRegistry

from .test_maker_checker import _AlwaysRejectLLM

_FORGED = {ContextKeys.MAX_ITERATIONS_REACHED: True}


class _MemoryAnswerLLM(LLMInterface):
    """Answers from memory at once (no tool) and optionally forges the flag."""

    def __init__(self, forge: bool) -> None:
        self.model = "mock-model"
        self.forge = forge

    def extract_field(self, request: FieldExtractionRequest) -> FieldExtractionResponse:
        values: dict[str, Any] = {
            "tool_name": "none",
            "should_terminate": True,
            "evaluation_passed": True,
            "evaluation_score": 0.9,
            "final_answer": "MEMORY",
            "reasoning": "r",
        }
        return FieldExtractionResponse(
            field_name=request.field_name,
            value=values.get(request.field_name, "text"),
            confidence=0.9,
            reasoning="m",
            is_valid=True,
        )

    def extract_bulk_data(self, request: Any) -> DataExtractionResponse:
        data: dict[str, Any] = {"final_answer": "MEMORY", "evaluation_passed": True}
        if self.forge:
            data.update(_FORGED)
        return DataExtractionResponse(extracted_data=data)

    def generate_response(
        self, request: ResponseGenerationRequest
    ) -> ResponseGenerationResponse:
        return ResponseGenerationResponse(
            message="MEMORY-ANSWER", message_type="response", reasoning="m"
        )


def _run_memory_agent(cls: Any, forge: bool) -> tuple[list[str], Any]:
    ran: list[str] = []
    registry = ToolRegistry()
    registry.register_function(
        lambda q="": ran.append(q) or "res", name="search", description="d"
    )
    agent = cls(
        tools=registry,
        config=AgentConfig(max_iterations=10, model="mock/m"),
        llm_interface=_MemoryAnswerLLM(forge),
    )
    return ran, agent.run("what is X")


@pytest.mark.parametrize("cls", [ReactAgent, ReflexionAgent], ids=["react", "refl"])
class TestForgedFlagDoesNotConclude:
    def test_forged_flag_does_not_conclude_on_turn_one(self, cls):
        ran, forged = _run_memory_agent(cls, forge=True)
        assert ran == []
        assert forged.trace.total_iterations > 1, "concluded on the forged flag"

    def test_forged_flag_changes_nothing(self, cls):
        _, honest = _run_memory_agent(cls, forge=False)
        _, forged = _run_memory_agent(cls, forge=True)
        assert forged.trace.total_iterations == honest.trace.total_iterations
        assert forged.success == honest.success

    def test_flag_is_seeded_false_and_limiter_still_sets_true(self, cls):
        _, result = _run_memory_agent(cls, forge=False)
        # The run with no tool ends at the forced stop, so the limiter or
        # stall detector wrote True over the seeded False.
        assert result.final_context.get(ContextKeys.MAX_ITERATIONS_REACHED) is True


class _ForgingRejectLLM(_AlwaysRejectLLM):
    def extract_bulk_data(self, request: Any) -> DataExtractionResponse:
        response = super().extract_bulk_data(request)
        response.extracted_data.update(_FORGED)
        return response


class TestMakerCheckerForgedFlag:
    @staticmethod
    def _run(llm: _AlwaysRejectLLM) -> Any:
        agent = MakerCheckerAgent(
            maker_instructions="w",
            checker_instructions="c",
            config=AgentConfig(max_iterations=10),
            max_revisions=100,
            llm_interface=llm,
        )
        return agent.run("haiku")

    def test_forged_flag_judges_as_many_drafts_as_honest(self):
        honest_llm, forged_llm = _AlwaysRejectLLM(), _ForgingRejectLLM()
        honest, forged = self._run(honest_llm), self._run(forged_llm)
        # Pre-fix: forged shipped DRAFT-1 after one rejected check.
        assert forged_llm.judged == honest_llm.judged
        assert len(forged_llm.judged) > 1
        assert forged.answer == honest.answer == honest_llm.judged[-1]
