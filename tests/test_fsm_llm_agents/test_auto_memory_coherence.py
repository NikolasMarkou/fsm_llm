from __future__ import annotations

"""Tests for AutoMemory conversational coherence (plan_2026-05-30_5598b755 / D-005).

Two fixes for the recall/chat-turn filler:
- Part A: the conclude extraction prompt answers from task/context + recalled
  memory, not only tool observations.
- Part B: AutoMemoryReactAgent auto-registers a first-class ``respond`` action so
  conversational/recall turns have a valid act and conclude cleanly (no flailing).
"""


from fsm_llm.definitions import (
    FieldExtractionRequest,
    FieldExtractionResponse,
    ResponseGenerationRequest,
    ResponseGenerationResponse,
)
from fsm_llm.llm import LLMInterface
from fsm_llm_agents.auto_memory import AutoMemoryReactAgent
from fsm_llm_agents.constants import ContextKeys
from fsm_llm_agents.definitions import AgentConfig, ToolCall
from fsm_llm_agents.prompts import build_conclude_extraction_instructions
from fsm_llm_agents.semantic_memory import SemanticMemoryStore
from fsm_llm_agents.tools import ToolRegistry, tool


@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))


def _registry() -> ToolRegistry:
    r = ToolRegistry()
    r.register(calculate._tool_definition)
    return r


# --------------------------------------------------------------------------- #
# Part A — conclude extraction prompt                                          #
# --------------------------------------------------------------------------- #


def test_conclude_extraction_uses_task_and_memory_not_only_observations():
    text = build_conclude_extraction_instructions()
    low = text.lower()
    assert "observations" in low
    # The fix: also draw on the task/context and recalled memory.
    assert "task" in low and "memory" in low
    assert "answer it directly" in low


# --------------------------------------------------------------------------- #
# Part B — respond action registration                                        #
# --------------------------------------------------------------------------- #


class TestRespondRegistration:
    def test_auto_registered_and_idempotent(self):
        reg = _registry()
        AutoMemoryReactAgent(
            tools=reg, config=AgentConfig(model="x"), memory=SemanticMemoryStore()
        )
        assert "respond" in reg
        # second agent on the same registry must not double-register or error
        AutoMemoryReactAgent(
            tools=reg, config=AgentConfig(model="x"), memory=SemanticMemoryStore()
        )
        assert sum(1 for t in reg.list_tools() if t.name == "respond") == 1

    def test_opt_out(self):
        reg = _registry()
        AutoMemoryReactAgent(
            tools=reg,
            config=AgentConfig(model="x"),
            memory=SemanticMemoryStore(),
            enable_respond=False,
        )
        assert "respond" not in reg

    def test_respond_echoes_answer(self):
        reg = _registry()
        AutoMemoryReactAgent(
            tools=reg, config=AgentConfig(model="x"), memory=SemanticMemoryStore()
        )
        result = reg.execute(ToolCall(tool_name="respond", parameters={"answer": "Hi!"}))
        assert result.success
        assert result.result == "Hi!"


# --------------------------------------------------------------------------- #
# Integration — a recall turn answers via respond, no filler / flailing        #
# --------------------------------------------------------------------------- #


class _RespondLLM(LLMInterface):
    """Picks the `respond` action on the first think, then terminates."""

    def __init__(self, answer: str) -> None:
        self.model = "mock-model"
        self._answer = answer
        self._think = 0

    def extract_field(self, request: FieldExtractionRequest) -> FieldExtractionResponse:
        fname = request.field_name
        if fname == ContextKeys.TOOL_NAME:
            self._think += 1
        if self._think <= 1:
            mapping = {
                ContextKeys.TOOL_NAME: "respond",
                ContextKeys.TOOL_INPUT: {"answer": self._answer},
                "reasoning": "No external tool needed; answering from memory.",
                ContextKeys.SHOULD_TERMINATE: False,
            }
        else:
            mapping = {
                ContextKeys.TOOL_NAME: ContextKeys.NO_TOOL,
                ContextKeys.TOOL_INPUT: {},
                "reasoning": "Answered.",
                ContextKeys.SHOULD_TERMINATE: True,
                ContextKeys.FINAL_ANSWER: self._answer,
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
            message=self._answer, message_type="response", reasoning="mock"
        )


def test_recall_turn_answers_via_respond_without_flailing():
    reg = _registry()
    agent = AutoMemoryReactAgent(
        tools=reg,
        config=AgentConfig(model="x", max_iterations=10, timeout_seconds=30.0),
        memory=SemanticMemoryStore(),
        llm_interface=_RespondLLM("Your name is Nikolas."),
    )
    result = agent.run("What is my name?")

    assert "respond" in result.tools_used, "the respond action should have executed"
    assert result.success, "a respond-backed turn should report success"
    assert "Nikolas" in result.answer
    # Concluded promptly via respond -> observation -> conclude (no 3-cycle stall).
    assert result.iterations_used <= 4
