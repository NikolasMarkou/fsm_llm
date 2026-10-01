"""Absence pins for the agents names removed by the "no legacy" cleanup.

Each test fails on the commit before the removal: the name existed there,
the positional system prompt built a ReactAgent with a warning, and the
terminal agent states carried extraction instructions no run ever used.
"""

from __future__ import annotations

import importlib
import importlib.util
import warnings
from typing import Any

import pytest

import fsm_llm.agents as agents
from fsm_llm.agents import ToolRegistry, constants, create_agent, exceptions, prompts
from fsm_llm.agents.definitions import ChainStep
from fsm_llm.agents.fsm_definitions import (
    build_adapt_fsm,
    build_debate_fsm,
    build_evalopt_fsm,
    build_maker_checker_fsm,
    build_orchestrator_fsm,
    build_plan_execute_fsm,
    build_prompt_chain_fsm,
    build_react_fsm,
    build_reflexion_fsm,
    build_rewoo_fsm,
    build_self_consistency_fsm,
)
from fsm_llm.agents.parallel_react import build_parallel_react_fsm


def _search(query: str) -> str:
    """Search the web for a query."""
    return f"results for {query}"


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_function(_search, name="search", description="Search the web")
    return registry


# The state members removed in step 20 came back in step 22.2 (D-047: the
# builders read every state name from its class); these two name no state.
_REMOVED_CONSTANTS = [
    ("PromptChainStates", "GATE_PREFIX"),
    ("SelfConsistencyStates", "AGGREGATE"),
    ("Defaults", "EVALUATION_THRESHOLD"),
    ("ErrorMessages", "BUDGET_EXHAUSTED"),
]

_REMOVED_PROMPT_BUILDERS = [
    "build_conclude_extraction_instructions",
    "build_synthesize_extraction_instructions",
    "build_rewoo_solve_extraction_instructions",
    "build_evalopt_output_extraction_instructions",
    "build_maker_checker_output_extraction_instructions",
    "build_orchestrator_synthesize_extraction_instructions",
    "build_combine_extraction_instructions",
    "build_chain_output_extraction_instructions",
]


class TestRemovedAgentNames:
    @pytest.mark.parametrize(
        "first",
        ["You are a helpful assistant", "You are X.", "x" * 33],
    )
    def test_positional_system_prompt_raises_listing_the_patterns(self, first):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError, match=r"Available:.*'react'") as info:
                create_agent(first, [_search])

        assert "Unknown pattern" in str(info.value)
        assert "'debate'" in str(info.value)

    def test_system_prompt_keyword_is_the_replacement(self):
        agent = create_agent(
            "react", [_search], system_prompt="You are a helpful assistant"
        )

        assert agent.config.instructions == "You are a helpful assistant"

    @pytest.mark.parametrize(
        "name", ["_is_legacy_system_prompt", "_LEGACY_PROMPT_MIN_LENGTH"]
    )
    def test_shim_helpers_are_gone(self, name):
        assert not hasattr(agents, name)

    @pytest.mark.parametrize("name", ["DecompositionError", "ToolValidationError"])
    def test_unraised_exceptions_are_gone(self, name):
        assert name not in agents.__all__
        assert not hasattr(agents, name)
        assert not hasattr(exceptions, name)

    def test_decomposition_result_stays(self):
        # A live model (ADaPT), not the removed exception of a similar name.
        assert "DecompositionResult" in agents.__all__

    def test_meta_fsm_module_is_gone(self):
        assert importlib.util.find_spec("fsm_llm.agents.meta_fsm") is None
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("fsm_llm.agents.meta_fsm")

    @pytest.mark.parametrize(("owner", "name"), _REMOVED_CONSTANTS)
    def test_unread_constant_is_gone(self, owner, name):
        assert not hasattr(getattr(constants, owner), name)

    @pytest.mark.parametrize("name", _REMOVED_PROMPT_BUILDERS)
    def test_terminal_extraction_prompt_builder_is_gone(self, name):
        assert not hasattr(prompts, name)


def _definitions() -> list[tuple[str, dict[str, Any]]]:
    chain = [
        ChainStep(
            step_id="outline",
            name="Outline",
            extraction_instructions="Extract a structured outline.",
            response_instructions="Write the outline.",
        ),
        ChainStep(
            step_id="draft",
            name="Draft",
            extraction_instructions="Extract the full draft text.",
            response_instructions="Write the draft.",
        ),
    ]
    return [
        ("react", build_react_fsm(_registry())),
        ("react_hitl", build_react_fsm(_registry(), include_approval_state=True)),
        ("react_classified", build_react_fsm(_registry(), use_classification=True)),
        ("reflexion", build_reflexion_fsm(_registry())),
        ("plan_execute", build_plan_execute_fsm(_registry())),
        ("rewoo", build_rewoo_fsm(_registry())),
        ("parallel_react", build_parallel_react_fsm(_registry())),
        ("evalopt", build_evalopt_fsm()),
        (
            "maker_checker",
            build_maker_checker_fsm(
                maker_instructions="Write a haiku", checker_instructions="Count"
            ),
        ),
        ("orchestrator", build_orchestrator_fsm()),
        ("adapt", build_adapt_fsm(_registry())),
        ("prompt_chain", build_prompt_chain_fsm(chain)),
        ("debate", build_debate_fsm()),
        ("self_consistency", build_self_consistency_fsm()),
    ]


class TestTerminalAgentStatesExtractNothing:
    """Core runs no extraction in a terminal state of an agent run, so a
    terminal state carries no extraction prompt: only the reply is written."""

    @pytest.mark.parametrize(
        ("pattern", "definition"), _definitions(), ids=[n for n, _ in _definitions()]
    )
    def test_terminal_states_have_no_extraction_prompt(self, pattern, definition):
        terminal = {
            sid: state
            for sid, state in definition["states"].items()
            if not state.get("transitions")
        }

        assert terminal, pattern
        for sid, state in terminal.items():
            assert not state.get("extraction_instructions"), (pattern, sid)
            assert not state.get("field_extractions"), (pattern, sid)
            assert not state.get("classification_extractions"), (pattern, sid)
            assert state["response_instructions"], (pattern, sid)
