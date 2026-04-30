from __future__ import annotations

"""Tests for fsm_llm_agents.rewoo module and REWOO FSM definition."""

import pytest

from fsm_llm.dialog.definitions import FSMDefinition
from fsm_llm.stdlib.agents.constants import (
    ContextKeys,
    Defaults,
    HandlerNames,
    REWOOStates,
)
from fsm_llm.stdlib.agents.definitions import (
    AgentConfig,
)
from fsm_llm.stdlib.agents.exceptions import AgentError
from fsm_llm.stdlib.agents.fsm_definitions import build_rewoo_fsm
from fsm_llm.stdlib.agents.rewoo import REWOOAgent
from fsm_llm.stdlib.agents.tools import ToolRegistry


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
# REWOOAgent creation
# ---------------------------------------------------------------------------


class TestREWOOAgentCreation:
    """Tests for REWOOAgent initialization."""

    def test_create_with_tools(self):
        registry = _make_registry()
        agent = REWOOAgent(tools=registry)
        assert agent.tools is registry
        assert agent.config is not None

    def test_create_with_config(self):
        registry = _make_registry()
        config = AgentConfig(max_iterations=5, model="gpt-4o-mini")
        agent = REWOOAgent(tools=registry, config=config)
        assert agent.config.max_iterations == 5
        assert agent.config.model == "gpt-4o-mini"

    def test_create_empty_registry_raises(self):
        registry = ToolRegistry()
        with pytest.raises(AgentError, match="empty tool registry"):
            REWOOAgent(tools=registry)

    def test_create_default_config(self):
        registry = _make_registry()
        agent = REWOOAgent(tools=registry)
        assert isinstance(agent.config, AgentConfig)
        assert agent.config.max_iterations == Defaults.MAX_ITERATIONS

    def test_has_register_handlers_method(self):
        registry = _make_registry()
        agent = REWOOAgent(tools=registry)
        assert hasattr(agent, "_register_handlers")
        assert callable(agent._register_handlers)

    def test_has_run_method(self):
        registry = _make_registry()
        agent = REWOOAgent(tools=registry)
        assert hasattr(agent, "run")
        assert callable(agent.run)

    def test_stores_tools_reference(self):
        registry = _make_registry()
        agent = REWOOAgent(tools=registry)
        assert agent.tools is registry
        assert len(agent.tools) == 2


# ---------------------------------------------------------------------------
# REWOO FSM definition
# ---------------------------------------------------------------------------


class TestREWOOFSM:
    """Tests for build_rewoo_fsm function."""

    def test_basic_fsm_structure(self):
        registry = _make_registry()
        fsm = build_rewoo_fsm(registry)

        assert fsm["name"] == "rewoo_agent"
        assert fsm["initial_state"] == "plan_all"
        assert "persona" in fsm
        assert "states" in fsm

    def test_fsm_is_valid_definition(self):
        """The generated FSM should be parseable as an FSMDefinition."""
        registry = _make_registry()
        fsm = build_rewoo_fsm(registry)
        fsm_def = FSMDefinition(**fsm)
        assert fsm_def.name == "rewoo_agent"

    def test_fsm_has_all_required_states(self):
        registry = _make_registry()
        fsm = build_rewoo_fsm(registry)
        expected_states = {"plan_all", "execute_plans", "solve"}
        assert set(fsm["states"].keys()) == expected_states

    def test_fsm_has_exactly_three_states(self):
        """REWOO is the most efficient pattern -- only 3 states."""
        registry = _make_registry()
        fsm = build_rewoo_fsm(registry)
        assert len(fsm["states"]) == 3

    def test_plan_all_transitions_to_execute_plans(self):
        registry = _make_registry()
        fsm = build_rewoo_fsm(registry)
        targets = {t["target_state"] for t in fsm["states"]["plan_all"]["transitions"]}
        assert targets == {"execute_plans"}

    def test_execute_plans_transitions_to_solve(self):
        registry = _make_registry()
        fsm = build_rewoo_fsm(registry)
        targets = {
            t["target_state"] for t in fsm["states"]["execute_plans"]["transitions"]
        }
        assert targets == {"solve"}

    def test_solve_is_terminal(self):
        registry = _make_registry()
        fsm = build_rewoo_fsm(registry)
        assert fsm["states"]["solve"]["transitions"] == []

    def test_solve_has_empty_transitions_list(self):
        """Terminal state should have an empty list, not missing transitions key."""
        registry = _make_registry()
        fsm = build_rewoo_fsm(registry)
        assert isinstance(fsm["states"]["solve"]["transitions"], list)
        assert len(fsm["states"]["solve"]["transitions"]) == 0

    def test_linear_flow_plan_execute_solve(self):
        """REWOO should have a strictly linear flow: plan_all -> execute_plans -> solve."""
        registry = _make_registry()
        fsm = build_rewoo_fsm(registry)

        # Verify each state has exactly one outgoing transition (except terminal)
        assert len(fsm["states"]["plan_all"]["transitions"]) == 1
        assert len(fsm["states"]["execute_plans"]["transitions"]) == 1
        assert len(fsm["states"]["solve"]["transitions"]) == 0

        # Verify the chain
        assert (
            fsm["states"]["plan_all"]["transitions"][0]["target_state"]
            == "execute_plans"
        )
        assert (
            fsm["states"]["execute_plans"]["transitions"][0]["target_state"] == "solve"
        )

    def test_plan_all_has_extraction_instructions(self):
        registry = _make_registry()
        fsm = build_rewoo_fsm(registry)
        state = fsm["states"]["plan_all"]
        assert "extraction_instructions" in state
        assert len(state["extraction_instructions"]) > 0

    def test_plan_all_has_tool_info(self):
        registry = _make_registry()
        fsm = build_rewoo_fsm(registry)
        plan_state = fsm["states"]["plan_all"]
        assert "search" in plan_state["extraction_instructions"]
        assert "calculate" in plan_state["extraction_instructions"]

    def test_solve_has_extraction_instructions(self):
        registry = _make_registry()
        fsm = build_rewoo_fsm(registry)
        state = fsm["states"]["solve"]
        assert "extraction_instructions" in state
        assert len(state["extraction_instructions"]) > 0

    def test_states_have_response_instructions(self):
        """All states should have response_instructions."""
        registry = _make_registry()
        fsm = build_rewoo_fsm(registry)
        for state_name in fsm["states"]:
            state = fsm["states"][state_name]
            assert "response_instructions" in state, (
                f"State '{state_name}' is missing response_instructions"
            )
            assert len(state["response_instructions"]) > 0

    def test_custom_task_description(self):
        registry = _make_registry()
        fsm = build_rewoo_fsm(registry, task_description="Calculate compound interest")
        assert fsm["description"] == "Calculate compound interest"

    def test_default_task_description(self):
        registry = _make_registry()
        fsm = build_rewoo_fsm(registry)
        assert (
            "rewoo" in fsm["description"].lower()
            or "planning" in fsm["description"].lower()
        )

    def test_persona_mentions_plan(self):
        registry = _make_registry()
        fsm = build_rewoo_fsm(registry)
        assert "plan" in fsm["persona"].lower()


# ---------------------------------------------------------------------------
# REWOO evidence substitution
# ---------------------------------------------------------------------------


class TestREWOOEvidenceSubstitution:
    """Tests for the REWOO evidence reference substitution logic."""

    def test_substitute_string_reference(self):
        registry = _make_registry()
        agent = REWOOAgent(tools=registry)
        evidence = {"E1": "Paris population: 2M"}
        result = agent._substitute_evidence_refs("The answer is #E1", evidence)
        assert result == "The answer is Paris population: 2M"

    def test_substitute_multiple_references(self):
        registry = _make_registry()
        agent = REWOOAgent(tools=registry)
        evidence = {"E1": "value1", "E2": "value2"}
        result = agent._substitute_evidence_refs("#E1 and #E2", evidence)
        assert result == "value1 and value2"

    def test_substitute_missing_reference_placeholder(self):
        registry = _make_registry()
        agent = REWOOAgent(tools=registry)
        evidence = {"E1": "value1"}
        result = agent._substitute_evidence_refs("#E3 not found", evidence)
        assert result == "[unavailable] not found"

    def test_substitute_in_dict(self):
        registry = _make_registry()
        agent = REWOOAgent(tools=registry)
        evidence = {"E1": "Paris"}
        result = agent._substitute_evidence_refs(
            {"query": "Population of #E1"}, evidence
        )
        assert result["query"] == "Population of Paris"

    def test_substitute_in_list(self):
        registry = _make_registry()
        agent = REWOOAgent(tools=registry)
        evidence = {"E1": "value"}
        result = agent._substitute_evidence_refs(["#E1", "no ref"], evidence)
        assert result == ["value", "no ref"]

    def test_substitute_non_string_passthrough(self):
        registry = _make_registry()
        agent = REWOOAgent(tools=registry)
        evidence = {"E1": "value"}
        assert agent._substitute_evidence_refs(42, evidence) == 42
        assert agent._substitute_evidence_refs(None, evidence) is None
        assert agent._substitute_evidence_refs(True, evidence) is True

    def test_substitute_nested_dict(self):
        registry = _make_registry()
        agent = REWOOAgent(tools=registry)
        evidence = {"E1": "deep_value"}
        result = agent._substitute_evidence_refs({"outer": {"inner": "#E1"}}, evidence)
        assert result["outer"]["inner"] == "deep_value"


# ---------------------------------------------------------------------------
# REWOO constants
# ---------------------------------------------------------------------------


class TestREWOOConstants:
    """Tests for REWOO-specific constants."""

    def test_rewoo_states_exist(self):
        assert REWOOStates.PLAN_ALL == "plan_all"
        assert REWOOStates.EXECUTE_PLANS == "execute_plans"
        assert REWOOStates.SOLVE == "solve"

    def test_rewoo_context_keys_exist(self):
        assert hasattr(ContextKeys, "EVIDENCE")
        assert hasattr(ContextKeys, "PLAN_BLUEPRINT")

    def test_rewoo_context_key_values(self):
        assert ContextKeys.EVIDENCE == "evidence"
        assert ContextKeys.PLAN_BLUEPRINT == "plan_blueprint"

    def test_rewoo_handler_names(self):
        assert HandlerNames.REWOO_EXECUTOR == "REWOOExecutor"

    def test_states_match_fsm_state_names(self):
        """Constants should match the actual FSM state names."""
        registry = _make_registry()
        fsm = build_rewoo_fsm(registry)
        state_names = set(fsm["states"].keys())

        constant_values = {
            REWOOStates.PLAN_ALL,
            REWOOStates.EXECUTE_PLANS,
            REWOOStates.SOLVE,
        }
        assert state_names == constant_values


# ---------------------------------------------------------------------------
# REWOO plan execution handler
# ---------------------------------------------------------------------------


class TestREWOOPlanExecution:
    """Tests for the _execute_all_plans handler method."""

    def test_execute_empty_plan(self):
        registry = _make_registry()
        agent = REWOOAgent(tools=registry)
        context = {ContextKeys.PLAN_BLUEPRINT: [], ContextKeys.AGENT_TRACE: []}
        result = agent._execute_all_plans(context)
        assert result[ContextKeys.EVIDENCE] == {}
        assert ContextKeys.ITERATION_COUNT not in result

    def test_execute_non_list_plan_returns_empty(self):
        registry = _make_registry()
        agent = REWOOAgent(tools=registry)
        context = {
            ContextKeys.PLAN_BLUEPRINT: "not a list",
            ContextKeys.AGENT_TRACE: [],
        }
        result = agent._execute_all_plans(context)
        assert result[ContextKeys.EVIDENCE] == {}

    def test_execute_single_plan_step(self):
        registry = _make_registry()
        agent = REWOOAgent(tools=registry)
        context = {
            ContextKeys.PLAN_BLUEPRINT: [
                {
                    "plan_id": 1,
                    "tool_name": "search",
                    "tool_input": {"query": "test"},
                    "description": "Search for test",
                }
            ],
            ContextKeys.AGENT_TRACE: [],
        }
        result = agent._execute_all_plans(context)
        assert "E1" in result[ContextKeys.EVIDENCE]
        assert ContextKeys.ITERATION_COUNT not in result
        assert len(result[ContextKeys.AGENT_TRACE]) == 1

    def test_execute_plan_with_evidence_substitution(self):
        registry = _make_registry()
        agent = REWOOAgent(tools=registry)
        context = {
            ContextKeys.PLAN_BLUEPRINT: [
                {
                    "plan_id": 1,
                    "tool_name": "search",
                    "tool_input": {"query": "France population"},
                    "description": "Get population",
                },
                {
                    "plan_id": 2,
                    "tool_name": "search",
                    "tool_input": {"query": "verify #E1"},
                    "description": "Verify population",
                },
            ],
            ContextKeys.AGENT_TRACE: [],
        }
        result = agent._execute_all_plans(context)
        assert "E1" in result[ContextKeys.EVIDENCE]
        assert "E2" in result[ContextKeys.EVIDENCE]
        assert ContextKeys.ITERATION_COUNT not in result

    def test_execute_plan_skips_non_dict_steps(self):
        registry = _make_registry()
        agent = REWOOAgent(tools=registry)
        context = {
            ContextKeys.PLAN_BLUEPRINT: [
                "not a dict",
                42,
                {
                    "plan_id": 1,
                    "tool_name": "search",
                    "tool_input": {"query": "test"},
                    "description": "Valid step",
                },
            ],
            ContextKeys.AGENT_TRACE: [],
        }
        result = agent._execute_all_plans(context)
        # Only the valid dict step should produce evidence
        assert "E1" in result[ContextKeys.EVIDENCE]
        assert len(result[ContextKeys.AGENT_TRACE]) == 1


# ---------------------------------------------------------------------------
# REWOO integration (skip without LLM)
# ---------------------------------------------------------------------------


class TestREWOOAgentIntegration:
    """Integration tests for REWOOAgent.run() -- require mocking LLM."""

    @pytest.mark.slow
    def test_run_requires_llm(self):
        """REWOOAgent.run() needs a real or mock LLM -- skip in unit tests."""
        pytest.skip("Requires LLM interface -- run with real_llm marker")
