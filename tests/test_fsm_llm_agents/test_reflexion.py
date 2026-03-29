from __future__ import annotations

"""Tests for fsm_llm_agents.reflexion module and Reflexion FSM definition."""

import pytest

from fsm_llm.definitions import FSMDefinition
from fsm_llm_agents.constants import (
    ContextKeys,
    Defaults,
    HandlerNames,
    ReflexionStates,
)
from fsm_llm_agents.definitions import (
    AgentConfig,
    EvaluationResult,
    ReflexionMemory,
)
from fsm_llm_agents.exceptions import AgentError
from fsm_llm_agents.fsm_definitions import build_reflexion_fsm
from fsm_llm_agents.reflexion import ReflexionAgent
from fsm_llm_agents.tools import ToolRegistry


def _search(params):
    return f"Results for: {params.get('query', '')}"


def _calculate(params):
    return eval(params.get("expression", "0"))


def _make_registry() -> ToolRegistry:
    """Create a registry with dummy tools."""
    registry = ToolRegistry()
    registry.register_function(_search, name="search", description="Search the web")
    registry.register_function(
        _calculate, name="calculate", description="Calculate expression"
    )
    return registry


# ---------------------------------------------------------------------------
# ReflexionAgent creation
# ---------------------------------------------------------------------------


class TestReflexionAgentCreation:
    """Tests for ReflexionAgent initialization."""

    def test_create_with_tools(self):
        registry = _make_registry()
        agent = ReflexionAgent(tools=registry)
        assert agent.tools is registry
        assert agent.config is not None
        assert agent.hitl is None

    def test_create_with_config(self):
        registry = _make_registry()
        config = AgentConfig(max_iterations=5, model="gpt-4o-mini")
        agent = ReflexionAgent(tools=registry, config=config)
        assert agent.config.max_iterations == 5
        assert agent.config.model == "gpt-4o-mini"

    def test_create_empty_registry_raises(self):
        registry = ToolRegistry()
        with pytest.raises(AgentError, match="empty tool registry"):
            ReflexionAgent(tools=registry)

    def test_create_default_config(self):
        registry = _make_registry()
        agent = ReflexionAgent(tools=registry)
        assert isinstance(agent.config, AgentConfig)
        assert agent.config.max_iterations == Defaults.MAX_ITERATIONS

    def test_create_with_custom_evaluation_fn(self):
        registry = _make_registry()
        eval_fn = lambda ctx: EvaluationResult(passed=True, score=1.0)  # noqa: E731
        agent = ReflexionAgent(tools=registry, evaluation_fn=eval_fn)
        assert agent.evaluation_fn is eval_fn

    def test_create_without_evaluation_fn(self):
        registry = _make_registry()
        agent = ReflexionAgent(tools=registry)
        assert agent.evaluation_fn is None

    def test_create_with_custom_max_reflections(self):
        registry = _make_registry()
        agent = ReflexionAgent(tools=registry, max_reflections=5)
        assert agent.max_reflections == 5

    def test_create_default_max_reflections(self):
        registry = _make_registry()
        agent = ReflexionAgent(tools=registry)
        assert agent.max_reflections == Defaults.MAX_REFLECTIONS

    def test_create_with_hitl(self):
        from fsm_llm_agents.hitl import HumanInTheLoop

        registry = _make_registry()
        hitl = HumanInTheLoop(
            approval_policy=lambda call, ctx: True,
            approval_callback=lambda req: True,
        )
        agent = ReflexionAgent(tools=registry, hitl=hitl)
        assert agent.hitl is hitl

    def test_has_register_handlers_method(self):
        registry = _make_registry()
        agent = ReflexionAgent(tools=registry)
        assert hasattr(agent, "_register_handlers")
        assert callable(agent._register_handlers)

    def test_has_run_method(self):
        registry = _make_registry()
        agent = ReflexionAgent(tools=registry)
        assert hasattr(agent, "run")
        assert callable(agent.run)

    def test_stores_tools_reference(self):
        registry = _make_registry()
        agent = ReflexionAgent(tools=registry)
        assert agent.tools is registry
        assert len(agent.tools) == 2


# ---------------------------------------------------------------------------
# Reflexion FSM definition
# ---------------------------------------------------------------------------


class TestReflexionFSM:
    """Tests for build_reflexion_fsm function."""

    def test_basic_fsm_structure(self):
        registry = _make_registry()
        fsm = build_reflexion_fsm(registry)

        assert fsm["name"] == "reflexion_agent"
        assert fsm["initial_state"] == "think"
        assert "persona" in fsm
        assert "states" in fsm

    def test_fsm_is_valid_definition(self):
        """The generated FSM should be parseable as an FSMDefinition."""
        registry = _make_registry()
        fsm = build_reflexion_fsm(registry)
        fsm_def = FSMDefinition(**fsm)
        assert fsm_def.name == "reflexion_agent"

    def test_fsm_has_all_required_states(self):
        registry = _make_registry()
        fsm = build_reflexion_fsm(registry)
        expected_states = {"think", "act", "evaluate", "reflect", "conclude"}
        assert set(fsm["states"].keys()) == expected_states

    def test_fsm_has_exactly_five_states(self):
        registry = _make_registry()
        fsm = build_reflexion_fsm(registry)
        assert len(fsm["states"]) == 5

    def test_think_transitions_to_act_and_conclude(self):
        registry = _make_registry()
        fsm = build_reflexion_fsm(registry)
        targets = {t["target_state"] for t in fsm["states"]["think"]["transitions"]}
        assert targets == {"act", "conclude"}

    def test_act_transitions(self):
        registry = _make_registry()
        fsm = build_reflexion_fsm(registry)
        targets = {t["target_state"] for t in fsm["states"]["act"]["transitions"]}
        assert targets == {"evaluate", "conclude"}

    def test_evaluate_transitions_to_conclude_and_reflect(self):
        registry = _make_registry()
        fsm = build_reflexion_fsm(registry)
        targets = {t["target_state"] for t in fsm["states"]["evaluate"]["transitions"]}
        assert targets == {"conclude", "reflect"}

    def test_reflect_transitions_to_think(self):
        registry = _make_registry()
        fsm = build_reflexion_fsm(registry)
        targets = {t["target_state"] for t in fsm["states"]["reflect"]["transitions"]}
        assert targets == {"think"}

    def test_conclude_is_terminal(self):
        registry = _make_registry()
        fsm = build_reflexion_fsm(registry)
        assert fsm["states"]["conclude"]["transitions"] == []

    def test_states_have_extraction_instructions(self):
        """States that extract data should have extraction_instructions."""
        registry = _make_registry()
        fsm = build_reflexion_fsm(registry)
        for state_name in ("think", "evaluate", "reflect", "conclude"):
            state = fsm["states"][state_name]
            assert "extraction_instructions" in state, (
                f"State '{state_name}' is missing extraction_instructions"
            )
            assert len(state["extraction_instructions"]) > 0

    def test_states_have_response_instructions(self):
        """All states should have response_instructions."""
        registry = _make_registry()
        fsm = build_reflexion_fsm(registry)
        for state_name in fsm["states"]:
            state = fsm["states"][state_name]
            assert "response_instructions" in state, (
                f"State '{state_name}' is missing response_instructions"
            )
            assert len(state["response_instructions"]) > 0

    def test_think_state_has_tool_info(self):
        registry = _make_registry()
        fsm = build_reflexion_fsm(registry)
        think = fsm["states"]["think"]
        assert "search" in think["extraction_instructions"]
        assert "calculate" in think["extraction_instructions"]

    def test_custom_task_description(self):
        registry = _make_registry()
        fsm = build_reflexion_fsm(registry, task_description="Solve math problems")
        assert fsm["description"] == "Solve math problems"

    def test_default_task_description(self):
        registry = _make_registry()
        fsm = build_reflexion_fsm(registry)
        assert (
            "reflexion" in fsm["description"].lower()
            or "evaluation" in fsm["description"].lower()
        )

    def test_persona_mentions_reflect(self):
        registry = _make_registry()
        fsm = build_reflexion_fsm(registry)
        assert (
            "reflect" in fsm["persona"].lower() or "critique" in fsm["persona"].lower()
        )


# ---------------------------------------------------------------------------
# Reflexion models
# ---------------------------------------------------------------------------


class TestReflexionModels:
    """Tests for Reflexion-specific Pydantic models."""

    def test_evaluation_result_passed(self):
        result = EvaluationResult(passed=True, score=0.95, feedback="Good answer")
        assert result.passed is True
        assert result.score == 0.95
        assert result.feedback == "Good answer"

    def test_evaluation_result_failed(self):
        result = EvaluationResult(passed=False, score=0.3, feedback="Incomplete")
        assert result.passed is False
        assert result.score == 0.3

    def test_evaluation_result_defaults(self):
        result = EvaluationResult(passed=True)
        assert result.score == 0.0
        assert result.feedback == ""
        assert result.criteria_met == []

    def test_evaluation_result_with_criteria(self):
        result = EvaluationResult(
            passed=True,
            score=1.0,
            criteria_met=["accuracy", "completeness"],
        )
        assert len(result.criteria_met) == 2
        assert "accuracy" in result.criteria_met

    def test_reflexion_memory_creation(self):
        mem = ReflexionMemory(episode=1, task_summary="Find X", outcome="Success")
        assert mem.episode == 1
        assert mem.task_summary == "Find X"
        assert mem.outcome == "Success"

    def test_reflexion_memory_timestamp_default(self):
        mem = ReflexionMemory(episode=1)
        assert mem.timestamp is not None

    def test_reflexion_memory_defaults(self):
        mem = ReflexionMemory(episode=1)
        assert mem.task_summary == ""
        assert mem.outcome == ""
        assert mem.reflection == ""
        assert mem.lessons == []

    def test_reflexion_memory_with_lessons(self):
        mem = ReflexionMemory(
            episode=2,
            reflection="Should have searched first",
            lessons=["Search before calculating", "Verify sources"],
        )
        assert len(mem.lessons) == 2
        assert "Search before calculating" in mem.lessons

    def test_reflexion_memory_serialization(self):
        """ReflexionMemory should be serializable to dict."""
        mem = ReflexionMemory(
            episode=1,
            task_summary="test",
            outcome="done",
            reflection="looks good",
            lessons=["lesson1"],
        )
        data = mem.model_dump(mode="json")
        assert data["episode"] == 1
        assert data["task_summary"] == "test"
        assert "timestamp" in data


# ---------------------------------------------------------------------------
# Reflexion constants
# ---------------------------------------------------------------------------


class TestReflexionConstants:
    """Tests for Reflexion-specific constants."""

    def test_reflexion_states_exist(self):
        assert ReflexionStates.THINK == "think"
        assert ReflexionStates.ACT == "act"
        assert ReflexionStates.EVALUATE == "evaluate"
        assert ReflexionStates.REFLECT == "reflect"
        assert ReflexionStates.CONCLUDE == "conclude"

    def test_reflexion_context_keys_exist(self):
        assert hasattr(ContextKeys, "EVALUATION_PASSED")
        assert hasattr(ContextKeys, "EVALUATION_SCORE")
        assert hasattr(ContextKeys, "EVALUATION_FEEDBACK")
        assert hasattr(ContextKeys, "EPISODIC_MEMORY")
        assert hasattr(ContextKeys, "REFLECTION_COUNT")

    def test_reflexion_context_key_values(self):
        assert ContextKeys.EVALUATION_PASSED == "evaluation_passed"
        assert ContextKeys.EVALUATION_SCORE == "evaluation_score"
        assert ContextKeys.EVALUATION_FEEDBACK == "evaluation_feedback"
        assert ContextKeys.EPISODIC_MEMORY == "episodic_memory"
        assert ContextKeys.REFLECTION_COUNT == "reflection_count"

    def test_reflexion_defaults(self):
        assert Defaults.MAX_REFLECTIONS == 3
        assert Defaults.EVALUATION_THRESHOLD == 0.7

    def test_reflexion_handler_names(self):
        assert HandlerNames.REFLEXION_EVALUATOR == "ReflexionEvaluator"
        assert HandlerNames.REFLEXION_REFLECTOR == "ReflexionReflector"

    def test_reflexion_states_match_fsm_state_names(self):
        """Constants should match the actual FSM state names."""
        registry = _make_registry()
        fsm = build_reflexion_fsm(registry)
        state_names = set(fsm["states"].keys())

        constant_values = {
            ReflexionStates.THINK,
            ReflexionStates.ACT,
            ReflexionStates.EVALUATE,
            ReflexionStates.REFLECT,
            ReflexionStates.CONCLUDE,
        }
        assert state_names == constant_values


# ---------------------------------------------------------------------------
# Reflexion integration (skip without LLM)
# ---------------------------------------------------------------------------


class TestReflexionAgentIntegration:
    """Integration tests for ReflexionAgent.run() -- require mocking LLM."""

    @pytest.mark.slow
    def test_run_requires_llm(self):
        """ReflexionAgent.run() needs a real or mock LLM -- skip in unit tests."""
        pytest.skip("Requires LLM interface -- run with real_llm marker")
