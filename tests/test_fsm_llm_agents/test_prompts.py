from __future__ import annotations

"""Tests for fsm_llm_agents.prompts module."""

from fsm_llm.stdlib.agents.prompts import (
    build_act_response_instructions,
    build_approval_extraction_instructions,
    build_conclude_extraction_instructions,
    build_conclude_response_instructions,
    build_think_extraction_instructions,
    build_think_response_instructions,
)
from fsm_llm.stdlib.agents.tools import ToolRegistry


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

    def test_think_response_instructions(self):
        instructions = build_think_response_instructions()
        assert isinstance(instructions, str)
        assert len(instructions) > 10

    def test_act_response_instructions(self):
        instructions = build_act_response_instructions()
        assert "tool" in instructions.lower()
        assert "observ" in instructions.lower()

    def test_conclude_extraction_instructions(self):
        instructions = build_conclude_extraction_instructions()
        assert "final_answer" in instructions
        assert "confidence" in instructions

    def test_conclude_response_instructions(self):
        instructions = build_conclude_response_instructions()
        assert "final" in instructions.lower()

    def test_approval_extraction_instructions(self):
        instructions = build_approval_extraction_instructions()
        assert "approval" in instructions.lower()
        assert "approval_granted" in instructions
