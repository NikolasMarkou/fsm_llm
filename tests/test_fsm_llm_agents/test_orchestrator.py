from __future__ import annotations

"""Tests for fsm_llm_agents.orchestrator module."""


from fsm_llm.definitions import FSMDefinition
from fsm_llm_agents.constants import (
    ContextKeys,
    Defaults,
    HandlerNames,
    OrchestratorStates,
)
from fsm_llm_agents.definitions import AgentConfig, AgentResult
from fsm_llm_agents.fsm_definitions import build_orchestrator_fsm
from fsm_llm_agents.orchestrator import OrchestratorAgent
from fsm_llm_agents.tools import ToolRegistry


def _dummy_tool(params):
    return "result"


def _make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_function(_dummy_tool, name="search", description="Search the web")
    return registry


def _dummy_worker(subtask: str) -> AgentResult:
    return AgentResult(answer=f"Solved: {subtask}", success=True)


class TestOrchestratorCreation:
    """Tests for OrchestratorAgent initialization."""

    def test_create_with_worker_factory(self):
        agent = OrchestratorAgent(worker_factory=_dummy_worker)
        assert agent.worker_factory is _dummy_worker
        assert agent.config is not None

    def test_create_without_worker_factory(self):
        agent = OrchestratorAgent()
        assert agent.worker_factory is None

    def test_create_with_tools(self):
        registry = _make_registry()
        agent = OrchestratorAgent(tools=registry)
        assert agent.tools is registry

    def test_create_without_tools(self):
        agent = OrchestratorAgent()
        assert agent.tools is None

    def test_create_with_max_workers(self):
        agent = OrchestratorAgent(max_workers=10)
        assert agent.max_workers == 10

    def test_create_with_default_max_workers(self):
        agent = OrchestratorAgent()
        assert agent.max_workers == Defaults.MAX_WORKERS

    def test_create_with_config_override(self):
        config = AgentConfig(max_iterations=20, model="gpt-4")
        agent = OrchestratorAgent(config=config)
        assert agent.config.max_iterations == 20
        assert agent.config.model == "gpt-4"

    def test_has_run_method(self):
        agent = OrchestratorAgent()
        assert callable(getattr(agent, "run", None))

    def test_create_with_worker_factory_and_tools(self):
        registry = _make_registry()
        agent = OrchestratorAgent(worker_factory=_dummy_worker, tools=registry)
        assert agent.worker_factory is _dummy_worker
        assert agent.tools is registry


class TestOrchestratorFSM:
    """Tests for build_orchestrator_fsm function."""

    def test_basic_fsm_structure(self):
        fsm = build_orchestrator_fsm()
        assert fsm["name"] == "orchestrator_agent"
        assert fsm["initial_state"] == "orchestrate"
        assert len(fsm["states"]) == 4

    def test_fsm_has_all_four_states(self):
        fsm = build_orchestrator_fsm()
        expected = {"orchestrate", "delegate", "collect", "synthesize"}
        assert set(fsm["states"].keys()) == expected

    def test_fsm_is_valid_definition(self):
        """The generated FSM should be parseable as an FSMDefinition."""
        fsm = build_orchestrator_fsm()
        fsm_def = FSMDefinition(**fsm)
        assert fsm_def.name == "orchestrator_agent"

    def test_orchestrate_transitions_to_delegate(self):
        fsm = build_orchestrator_fsm()
        targets = {
            t["target_state"] for t in fsm["states"]["orchestrate"]["transitions"]
        }
        assert "delegate" in targets

    def test_delegate_transitions_to_collect(self):
        fsm = build_orchestrator_fsm()
        targets = {t["target_state"] for t in fsm["states"]["delegate"]["transitions"]}
        assert "collect" in targets

    def test_collect_transitions_to_synthesize_and_orchestrate(self):
        fsm = build_orchestrator_fsm()
        targets = {t["target_state"] for t in fsm["states"]["collect"]["transitions"]}
        assert "synthesize" in targets
        assert "orchestrate" in targets

    def test_synthesize_is_terminal(self):
        fsm = build_orchestrator_fsm()
        assert fsm["states"]["synthesize"]["transitions"] == []

    def test_custom_task_description(self):
        fsm = build_orchestrator_fsm(task_description="Plan a project")
        assert fsm["description"] == "Plan a project"

    def test_default_task_description(self):
        fsm = build_orchestrator_fsm()
        assert fsm["description"] == "Orchestrator-Workers agent"

    def test_persona_mentions_orchestrator(self):
        fsm = build_orchestrator_fsm()
        assert "orchestrator" in fsm["persona"].lower()

    def test_synthesize_priority_higher_than_orchestrate_loop(self):
        """Lower priority number = higher confidence in TransitionEvaluator.

        The collect state has a conditional synthesize (priority 10) and
        a fallback synthesize (priority 900).  Only the conditional one
        must beat the orchestrate loop (priority 300).
        """
        fsm = build_orchestrator_fsm()
        collect_transitions = fsm["states"]["collect"]["transitions"]
        synth_priority = None
        orch_priority = None
        for t in collect_transitions:
            if t["target_state"] == "synthesize" and t.get("conditions"):
                synth_priority = t["priority"]
            elif t["target_state"] == "orchestrate":
                orch_priority = t["priority"]
        assert synth_priority is not None
        assert orch_priority is not None
        assert synth_priority < orch_priority


class TestOrchestratorConstants:
    """Tests for orchestrator-related constants."""

    def test_orchestrator_states_orchestrate(self):
        assert OrchestratorStates.ORCHESTRATE == "orchestrate"

    def test_orchestrator_states_delegate(self):
        assert OrchestratorStates.DELEGATE == "delegate"

    def test_orchestrator_states_collect(self):
        assert OrchestratorStates.COLLECT == "collect"

    def test_orchestrator_states_synthesize(self):
        assert OrchestratorStates.SYNTHESIZE == "synthesize"

    def test_context_keys_subtasks(self):
        assert ContextKeys.SUBTASKS == "subtasks"

    def test_context_keys_worker_results(self):
        assert ContextKeys.WORKER_RESULTS == "worker_results"

    def test_context_keys_delegation_plan(self):
        assert ContextKeys.DELEGATION_PLAN == "delegation_plan"

    def test_context_keys_all_collected(self):
        assert ContextKeys.ALL_COLLECTED == "all_collected"

    def test_defaults_max_workers(self):
        assert Defaults.MAX_WORKERS == 5

    def test_handler_name_orchestrator_delegator(self):
        assert HandlerNames.ORCHESTRATOR_DELEGATOR == "OrchestratorDelegator"


class TestOrchestratorDelegation:
    """Tests for the internal delegation handler logic."""

    def test_delegate_to_workers_calls_factory(self):
        """Verify the worker delegation handler passes subtasks to worker_factory."""
        agent = OrchestratorAgent(worker_factory=_dummy_worker)

        context = {
            ContextKeys.SUBTASKS: ["task1", "task2"],
            ContextKeys.WORKER_RESULTS: [],
            ContextKeys.AGENT_TRACE: [],
        }

        result = agent._delegate_to_workers(context)

        assert ContextKeys.WORKER_RESULTS in result
        worker_results = result[ContextKeys.WORKER_RESULTS]
        assert len(worker_results) == 2
        assert worker_results[0]["success"] is True
        assert "Solved: task1" in worker_results[0]["answer"]

    def test_delegate_without_factory_uses_placeholders(self):
        """Without a worker_factory, delegation stores placeholders."""
        agent = OrchestratorAgent()

        context = {
            ContextKeys.SUBTASKS: ["subtask_a"],
            ContextKeys.WORKER_RESULTS: [],
            ContextKeys.AGENT_TRACE: [],
        }

        result = agent._delegate_to_workers(context)
        worker_results = result[ContextKeys.WORKER_RESULTS]
        assert len(worker_results) == 1
        assert "Pending LLM processing" in worker_results[0]["answer"]

    def test_delegate_respects_max_workers(self):
        """Delegation should not exceed max_workers."""
        agent = OrchestratorAgent(worker_factory=_dummy_worker, max_workers=2)

        context = {
            ContextKeys.SUBTASKS: ["t1", "t2", "t3", "t4", "t5"],
            ContextKeys.WORKER_RESULTS: [],
            ContextKeys.AGENT_TRACE: [],
        }

        result = agent._delegate_to_workers(context)
        worker_results = result[ContextKeys.WORKER_RESULTS]
        assert len(worker_results) == 2
