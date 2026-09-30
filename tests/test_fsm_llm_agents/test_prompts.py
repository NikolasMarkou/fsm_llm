from __future__ import annotations

"""Tests for fsm_llm.agents.prompts module."""

import pytest

from fsm_llm.agents.prompts import (
    build_conclude_extraction_instructions,
    build_conclude_response_instructions,
    build_debate_conclude_response_instructions,
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

    def test_conclude_extraction_instructions(self):
        instructions = build_conclude_extraction_instructions()
        assert "final_answer" in instructions
        assert "confidence" in instructions

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
