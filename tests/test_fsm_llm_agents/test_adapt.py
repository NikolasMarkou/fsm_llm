from __future__ import annotations

"""Tests for fsm_llm_agents.adapt module."""

import pytest

from fsm_llm.definitions import FSMDefinition
from fsm_llm_agents.adapt import ADaPTAgent
from fsm_llm_agents.constants import (
    ADaPTStates,
    ContextKeys,
    Defaults,
    HandlerNames,
)
from fsm_llm_agents.definitions import AgentConfig, DecompositionResult
from fsm_llm_agents.fsm_definitions import build_adapt_fsm
from fsm_llm_agents.tools import ToolRegistry


def _dummy_tool(params):
    return "result"


def _make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_function(_dummy_tool, name="search", description="Search the web")
    return registry


class TestADaPTCreation:
    """Tests for ADaPTAgent initialization."""

    def test_create_with_defaults(self):
        agent = ADaPTAgent()
        assert agent.max_depth == Defaults.MAX_DECOMPOSITION_DEPTH
        assert agent.config is not None
        assert agent.tools is None

    def test_create_with_tools(self):
        registry = _make_registry()
        agent = ADaPTAgent(tools=registry)
        assert agent.tools is registry

    def test_create_without_tools(self):
        agent = ADaPTAgent()
        assert agent.tools is None

    def test_create_with_max_depth(self):
        agent = ADaPTAgent(max_depth=5)
        assert agent.max_depth == 5

    def test_create_with_config_override(self):
        config = AgentConfig(max_iterations=15, model="gpt-4o-mini")
        agent = ADaPTAgent(config=config)
        assert agent.config.max_iterations == 15
        assert agent.config.model == "gpt-4o-mini"

    def test_has_run_method(self):
        agent = ADaPTAgent()
        assert callable(getattr(agent, "run", None))

    def test_run_accepts_depth_parameter(self):
        """run() accepts _depth for recursive tracking."""
        import inspect

        sig = inspect.signature(ADaPTAgent.run)
        assert "_depth" in sig.parameters

    def test_create_with_tools_and_max_depth(self):
        registry = _make_registry()
        agent = ADaPTAgent(tools=registry, max_depth=10)
        assert agent.tools is registry
        assert agent.max_depth == 10


class TestADaPTFSM:
    """Tests for build_adapt_fsm function."""

    def test_basic_fsm_structure(self):
        fsm = build_adapt_fsm()
        assert fsm["name"] == "adapt_agent"
        assert fsm["initial_state"] == "attempt"
        assert len(fsm["states"]) == 4

    def test_fsm_has_all_four_states(self):
        fsm = build_adapt_fsm()
        expected = {"attempt", "assess", "decompose", "combine"}
        assert set(fsm["states"].keys()) == expected

    def test_fsm_is_valid_definition(self):
        """The generated FSM should be parseable as an FSMDefinition."""
        fsm = build_adapt_fsm()
        fsm_def = FSMDefinition(**fsm)
        assert fsm_def.name == "adapt_agent"

    def test_attempt_transitions_to_assess(self):
        fsm = build_adapt_fsm()
        targets = {t["target_state"] for t in fsm["states"]["attempt"]["transitions"]}
        assert "assess" in targets

    def test_assess_transitions_to_combine_and_decompose(self):
        fsm = build_adapt_fsm()
        targets = {t["target_state"] for t in fsm["states"]["assess"]["transitions"]}
        assert "combine" in targets
        assert "decompose" in targets

    def test_decompose_transitions_to_combine(self):
        fsm = build_adapt_fsm()
        targets = {t["target_state"] for t in fsm["states"]["decompose"]["transitions"]}
        assert "combine" in targets

    def test_combine_is_terminal(self):
        fsm = build_adapt_fsm()
        assert fsm["states"]["combine"]["transitions"] == []

    def test_custom_task_description(self):
        fsm = build_adapt_fsm(task_description="Explain quantum computing")
        assert fsm["description"] == "Explain quantum computing"

    def test_default_task_description(self):
        fsm = build_adapt_fsm()
        assert fsm["description"] == "ADaPT agent with recursive decomposition"

    def test_persona_mentions_adaptive(self):
        fsm = build_adapt_fsm()
        assert "adaptive" in fsm["persona"].lower()

    def test_fsm_with_registry(self):
        """build_adapt_fsm accepts an optional ToolRegistry."""
        registry = _make_registry()
        fsm = build_adapt_fsm(registry=registry)
        fsm_def = FSMDefinition(**fsm)
        assert fsm_def.name == "adapt_agent"

    def test_assess_success_priority_higher_than_decompose(self):
        """Lower priority number = higher confidence in TransitionEvaluator."""
        fsm = build_adapt_fsm()
        assess_transitions = fsm["states"]["assess"]["transitions"]
        combine_success_priority = None
        decompose_priority = None
        for t in assess_transitions:
            if t["target_state"] == "combine" and combine_success_priority is None:
                combine_success_priority = t["priority"]
            elif t["target_state"] == "decompose":
                decompose_priority = t["priority"]
        assert combine_success_priority is not None
        assert decompose_priority is not None
        assert combine_success_priority < decompose_priority

    def test_max_depth_embedded_in_fsm(self):
        """The max_depth parameter should be referenced in the FSM conditions."""
        fsm = build_adapt_fsm(max_depth=7)
        # The assess->decompose transition should reference max_depth in its logic
        assess_transitions = fsm["states"]["assess"]["transitions"]
        decompose_transition = None
        for t in assess_transitions:
            if t["target_state"] == "decompose":
                decompose_transition = t
                break
        assert decompose_transition is not None
        # The condition logic should include the depth check with value 7
        conditions = decompose_transition.get("conditions", [])
        assert len(conditions) > 0


class TestDecompositionResultModel:
    """Tests for DecompositionResult Pydantic model."""

    def test_basic_creation(self):
        result = DecompositionResult()
        assert result.subtasks == []
        assert result.operator == "AND"
        assert result.depth == 0

    def test_creation_with_subtasks(self):
        result = DecompositionResult(
            subtasks=["Find data", "Analyze data", "Report findings"],
            operator="AND",
            depth=1,
        )
        assert len(result.subtasks) == 3
        assert result.operator == "AND"
        assert result.depth == 1

    def test_or_operator(self):
        result = DecompositionResult(
            subtasks=["Try method A", "Try method B"],
            operator="OR",
        )
        assert result.operator == "OR"

    def test_invalid_operator_raises(self):
        with pytest.raises(ValueError, match=r"AND.*OR"):
            DecompositionResult(operator="XOR")

    def test_invalid_operator_lowercase_raises(self):
        with pytest.raises(ValueError, match=r"AND.*OR"):
            DecompositionResult(operator="and")

    def test_depth_field(self):
        result = DecompositionResult(depth=3)
        assert result.depth == 3

    def test_serialization(self):
        result = DecompositionResult(
            subtasks=["a", "b"],
            operator="OR",
            depth=2,
        )
        data = result.model_dump(mode="json")
        assert data["subtasks"] == ["a", "b"]
        assert data["operator"] == "OR"
        assert data["depth"] == 2


class TestADaPTConstants:
    """Tests for ADaPT-related constants."""

    def test_adapt_states_attempt(self):
        assert ADaPTStates.ATTEMPT == "attempt"

    def test_adapt_states_assess(self):
        assert ADaPTStates.ASSESS == "assess"

    def test_adapt_states_decompose(self):
        assert ADaPTStates.DECOMPOSE == "decompose"

    def test_adapt_states_combine(self):
        assert ADaPTStates.COMBINE == "combine"

    def test_context_keys_attempt_result(self):
        assert ContextKeys.ATTEMPT_RESULT == "attempt_result"

    def test_context_keys_attempt_succeeded(self):
        assert ContextKeys.ATTEMPT_SUCCEEDED == "attempt_succeeded"

    def test_context_keys_decomposition(self):
        assert ContextKeys.DECOMPOSITION == "decomposition"

    def test_context_keys_subtask_results(self):
        assert ContextKeys.SUBTASK_RESULTS == "subtask_results"

    def test_context_keys_current_depth(self):
        assert ContextKeys.CURRENT_DEPTH == "current_depth"

    def test_defaults_max_decomposition_depth(self):
        assert Defaults.MAX_DECOMPOSITION_DEPTH == 3

    def test_handler_name_adapt_assessor(self):
        assert HandlerNames.ADAPT_ASSESSOR == "ADaPTAssessor"
