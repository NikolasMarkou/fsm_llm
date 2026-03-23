from __future__ import annotations

"""Tests for fsm_llm_agents.plan_execute module and Plan-Execute FSM definition."""

import pytest

from fsm_llm.definitions import FSMDefinition
from fsm_llm_agents.constants import (
    ContextKeys,
    Defaults,
    HandlerNames,
    PlanExecuteStates,
)
from fsm_llm_agents.definitions import (
    AgentConfig,
    PlanStep,
)
from fsm_llm_agents.fsm_definitions import build_plan_execute_fsm
from fsm_llm_agents.plan_execute import PlanExecuteAgent
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
# PlanExecuteAgent creation
# ---------------------------------------------------------------------------


class TestPlanExecuteAgentCreation:
    """Tests for PlanExecuteAgent initialization."""

    def test_create_with_tools(self):
        registry = _make_registry()
        agent = PlanExecuteAgent(tools=registry)
        assert agent.tools is registry
        assert agent.config is not None

    def test_create_without_tools(self):
        """PlanExecuteAgent supports tool-less mode (LLM-only execution)."""
        agent = PlanExecuteAgent()
        assert agent.tools is None
        assert agent.config is not None

    def test_create_with_none_tools(self):
        agent = PlanExecuteAgent(tools=None)
        assert agent.tools is None

    def test_create_with_config(self):
        registry = _make_registry()
        config = AgentConfig(max_iterations=5, model="gpt-4o-mini")
        agent = PlanExecuteAgent(tools=registry, config=config)
        assert agent.config.max_iterations == 5
        assert agent.config.model == "gpt-4o-mini"

    def test_create_default_config(self):
        registry = _make_registry()
        agent = PlanExecuteAgent(tools=registry)
        assert isinstance(agent.config, AgentConfig)
        assert agent.config.max_iterations == Defaults.MAX_ITERATIONS

    def test_create_with_custom_max_replans(self):
        registry = _make_registry()
        agent = PlanExecuteAgent(tools=registry, max_replans=5)
        assert agent.max_replans == 5

    def test_create_default_max_replans(self):
        registry = _make_registry()
        agent = PlanExecuteAgent(tools=registry)
        assert agent.max_replans == Defaults.MAX_REPLANS

    def test_has_register_handlers_method(self):
        registry = _make_registry()
        agent = PlanExecuteAgent(tools=registry)
        assert hasattr(agent, "_register_handlers")
        assert callable(agent._register_handlers)

    def test_has_run_method(self):
        registry = _make_registry()
        agent = PlanExecuteAgent(tools=registry)
        assert hasattr(agent, "run")
        assert callable(agent.run)

    def test_handlers_created_with_tools(self):
        registry = _make_registry()
        agent = PlanExecuteAgent(tools=registry)
        assert agent._handlers is not None

    def test_handlers_none_without_tools(self):
        agent = PlanExecuteAgent()
        assert agent._handlers is None

    def test_stores_tools_reference(self):
        registry = _make_registry()
        agent = PlanExecuteAgent(tools=registry)
        assert agent.tools is registry
        assert len(agent.tools) == 2


# ---------------------------------------------------------------------------
# Plan-Execute FSM definition
# ---------------------------------------------------------------------------


class TestPlanExecuteFSM:
    """Tests for build_plan_execute_fsm function."""

    def test_basic_fsm_structure(self):
        registry = _make_registry()
        fsm = build_plan_execute_fsm(registry)

        assert fsm["name"] == "plan_execute_agent"
        assert fsm["initial_state"] == "plan"
        assert "persona" in fsm
        assert "states" in fsm

    def test_fsm_is_valid_definition(self):
        """The generated FSM should be parseable as an FSMDefinition."""
        registry = _make_registry()
        fsm = build_plan_execute_fsm(registry)
        fsm_def = FSMDefinition(**fsm)
        assert fsm_def.name == "plan_execute_agent"

    def test_fsm_without_tools_is_valid(self):
        """FSM should be valid even without a tool registry."""
        fsm = build_plan_execute_fsm()
        fsm_def = FSMDefinition(**fsm)
        assert fsm_def.name == "plan_execute_agent"

    def test_fsm_has_all_required_states(self):
        registry = _make_registry()
        fsm = build_plan_execute_fsm(registry)
        expected_states = {
            "plan",
            "execute_step",
            "check_result",
            "replan",
            "synthesize",
        }
        assert set(fsm["states"].keys()) == expected_states

    def test_fsm_has_exactly_five_states(self):
        registry = _make_registry()
        fsm = build_plan_execute_fsm(registry)
        assert len(fsm["states"]) == 5

    def test_plan_transitions_to_execute_step(self):
        registry = _make_registry()
        fsm = build_plan_execute_fsm(registry)
        targets = {t["target_state"] for t in fsm["states"]["plan"]["transitions"]}
        assert targets == {"execute_step"}

    def test_execute_step_transitions_to_check_result(self):
        registry = _make_registry()
        fsm = build_plan_execute_fsm(registry)
        targets = {
            t["target_state"] for t in fsm["states"]["execute_step"]["transitions"]
        }
        assert targets == {"check_result"}

    def test_check_result_transitions(self):
        """check_result should transition to synthesize, replan, or execute_step."""
        registry = _make_registry()
        fsm = build_plan_execute_fsm(registry)
        targets = {
            t["target_state"] for t in fsm["states"]["check_result"]["transitions"]
        }
        assert targets == {"synthesize", "replan", "execute_step"}

    def test_replan_transitions_to_execute_step(self):
        registry = _make_registry()
        fsm = build_plan_execute_fsm(registry)
        targets = {t["target_state"] for t in fsm["states"]["replan"]["transitions"]}
        assert targets == {"execute_step"}

    def test_synthesize_is_terminal(self):
        registry = _make_registry()
        fsm = build_plan_execute_fsm(registry)
        assert fsm["states"]["synthesize"]["transitions"] == []

    def test_states_have_extraction_instructions(self):
        """States that extract data should have extraction_instructions."""
        registry = _make_registry()
        fsm = build_plan_execute_fsm(registry)
        for state_name in (
            "plan",
            "execute_step",
            "check_result",
            "replan",
            "synthesize",
        ):
            state = fsm["states"][state_name]
            assert "extraction_instructions" in state, (
                f"State '{state_name}' is missing extraction_instructions"
            )
            assert len(state["extraction_instructions"]) > 0

    def test_states_have_response_instructions(self):
        """All states should have response_instructions."""
        registry = _make_registry()
        fsm = build_plan_execute_fsm(registry)
        for state_name in fsm["states"]:
            state = fsm["states"][state_name]
            assert "response_instructions" in state, (
                f"State '{state_name}' is missing response_instructions"
            )
            assert len(state["response_instructions"]) > 0

    def test_custom_task_description(self):
        registry = _make_registry()
        fsm = build_plan_execute_fsm(registry, task_description="Compare populations")
        assert fsm["description"] == "Compare populations"

    def test_default_task_description(self):
        registry = _make_registry()
        fsm = build_plan_execute_fsm(registry)
        assert (
            "plan" in fsm["description"].lower()
            or "execute" in fsm["description"].lower()
        )

    def test_persona_mentions_plan(self):
        registry = _make_registry()
        fsm = build_plan_execute_fsm(registry)
        assert "plan" in fsm["persona"].lower()

    def test_check_result_has_all_steps_complete_condition(self):
        """Transition to synthesize should check all_steps_complete."""
        registry = _make_registry()
        fsm = build_plan_execute_fsm(registry)
        transitions = fsm["states"]["check_result"]["transitions"]
        synthesize_trans = [t for t in transitions if t["target_state"] == "synthesize"]
        assert len(synthesize_trans) == 1
        conditions = synthesize_trans[0].get("conditions", [])
        assert any("all_steps_complete" in str(c.get("logic", {})) for c in conditions)


# ---------------------------------------------------------------------------
# PlanStep model
# ---------------------------------------------------------------------------


class TestPlanStepModel:
    """Tests for PlanStep Pydantic model."""

    def test_basic_creation(self):
        step = PlanStep(step_id=1, description="Search for population data")
        assert step.step_id == 1
        assert step.description == "Search for population data"

    def test_status_defaults_to_pending(self):
        step = PlanStep(step_id=1, description="Step 1")
        assert step.status == "pending"

    def test_result_defaults_to_empty(self):
        step = PlanStep(step_id=1, description="Step 1")
        assert step.result == ""

    def test_dependencies_default_to_empty(self):
        step = PlanStep(step_id=1, description="Step 1")
        assert step.dependencies == []

    def test_with_dependencies(self):
        step = PlanStep(step_id=3, description="Step 3", dependencies=[1, 2])
        assert step.dependencies == [1, 2]

    def test_with_custom_status(self):
        step = PlanStep(step_id=1, description="Step 1", status="completed")
        assert step.status == "completed"

    def test_with_result(self):
        step = PlanStep(
            step_id=1,
            description="Search",
            result="Found: Paris population is 2M",
        )
        assert "Paris" in step.result

    def test_serialization(self):
        step = PlanStep(step_id=1, description="Step 1", dependencies=[2])
        data = step.model_dump()
        assert data["step_id"] == 1
        assert data["description"] == "Step 1"
        assert data["status"] == "pending"
        assert data["dependencies"] == [2]


# ---------------------------------------------------------------------------
# Plan-Execute constants
# ---------------------------------------------------------------------------


class TestPlanExecuteConstants:
    """Tests for Plan-Execute specific constants."""

    def test_plan_execute_states_exist(self):
        assert PlanExecuteStates.PLAN == "plan"
        assert PlanExecuteStates.EXECUTE_STEP == "execute_step"
        assert PlanExecuteStates.CHECK_RESULT == "check_result"
        assert PlanExecuteStates.REPLAN == "replan"
        assert PlanExecuteStates.SYNTHESIZE == "synthesize"

    def test_plan_execute_context_keys_exist(self):
        assert hasattr(ContextKeys, "PLAN_STEPS")
        assert hasattr(ContextKeys, "CURRENT_STEP_INDEX")
        assert hasattr(ContextKeys, "STEP_RESULTS")
        assert hasattr(ContextKeys, "ALL_STEPS_COMPLETE")
        assert hasattr(ContextKeys, "STEP_FAILED")

    def test_plan_execute_context_key_values(self):
        assert ContextKeys.PLAN_STEPS == "plan_steps"
        assert ContextKeys.CURRENT_STEP_INDEX == "current_step_index"
        assert ContextKeys.STEP_RESULTS == "step_results"
        assert ContextKeys.ALL_STEPS_COMPLETE == "all_steps_complete"
        assert ContextKeys.STEP_FAILED == "step_failed"

    def test_plan_execute_defaults(self):
        assert Defaults.MAX_PLAN_STEPS == 10
        assert Defaults.MAX_REPLANS == 2

    def test_plan_execute_handler_names(self):
        assert HandlerNames.PLAN_STEP_EXECUTOR == "PlanStepExecutor"
        assert HandlerNames.PLAN_STEP_CHECKER == "PlanStepChecker"

    def test_states_match_fsm_state_names(self):
        """Constants should match the actual FSM state names."""
        registry = _make_registry()
        fsm = build_plan_execute_fsm(registry)
        state_names = set(fsm["states"].keys())

        constant_values = {
            PlanExecuteStates.PLAN,
            PlanExecuteStates.EXECUTE_STEP,
            PlanExecuteStates.CHECK_RESULT,
            PlanExecuteStates.REPLAN,
            PlanExecuteStates.SYNTHESIZE,
        }
        assert state_names == constant_values


# ---------------------------------------------------------------------------
# Plan-Execute integration (skip without LLM)
# ---------------------------------------------------------------------------


class TestPlanExecuteAgentIntegration:
    """Integration tests for PlanExecuteAgent.run() -- require mocking LLM."""

    @pytest.mark.slow
    def test_run_requires_llm(self):
        """PlanExecuteAgent.run() needs a real or mock LLM -- skip in unit tests."""
        pytest.skip("Requires LLM interface -- run with real_llm marker")
