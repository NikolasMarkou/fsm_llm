from __future__ import annotations

"""Tests for fsm_llm_agents.orchestrator module."""


import re
from typing import Any

from fsm_llm.definitions import (
    DataExtractionResponse,
    FieldExtractionRequest,
    FieldExtractionResponse,
    FSMDefinition,
    ResponseGenerationRequest,
    ResponseGenerationResponse,
)
from fsm_llm.llm import LLMInterface
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


class _DecisionLLM(LLMInterface):
    """Mock LLM that answers ``key`` from ``decisions`` in order.

    Every other field, and every ``- "name"`` a bulk prompt lists, gets a
    filler value. One turn's field pass and bulk pass see the same decision;
    the last decision repeats once the list is exhausted.
    """

    def __init__(self, key: str, decisions: list[bool]) -> None:
        self.model = "mock-model"
        self.key = key
        self.decisions = decisions
        self.asked = 0
        self._pending: bool | None = None

    def _next(self) -> bool:
        value = self.decisions[min(self.asked, len(self.decisions) - 1)]
        self.asked += 1
        return value

    def extract_field(self, request: FieldExtractionRequest) -> FieldExtractionResponse:
        name = request.field_name
        value: Any = "text"
        if name == self.key:
            value = self._pending = self._next()
        elif name == ContextKeys.SUBTASKS:
            value = ["subtask"]
        return FieldExtractionResponse(
            field_name=name, value=value, confidence=0.9, reasoning="m", is_valid=True
        )

    def extract_bulk_data(self, request: Any) -> DataExtractionResponse:
        data: dict[str, Any] = {}
        for name in re.findall(r'- "(\w+)"', request.system_prompt):
            if name != self.key:
                data[name] = "text"
            elif self._pending is not None:
                data[name], self._pending = self._pending, None
            else:
                data[name] = self._next()
        return DataExtractionResponse(extracted_data=data)

    def generate_response(
        self, request: ResponseGenerationRequest
    ) -> ResponseGenerationResponse:
        return ResponseGenerationResponse(
            message="ok", message_type="response", reasoning="m"
        )


def _run_recording_states(agent: Any, task: str) -> tuple[Any, list[str]]:
    """Run the agent; return (result, state before each loop turn)."""
    states: list[str] = []
    real_hook = agent._on_loop_iteration

    def _hook(api: Any, conv_id: str, iteration: int) -> None:
        states.append(api.get_current_state(conv_id))
        real_hook(api, conv_id, iteration)

    agent._on_loop_iteration = _hook
    return agent.run(task), states


class TestAllCollectedTypedExtraction:
    """D-009: ``collect`` declares ``all_collected`` as a typed bool extraction."""

    def test_collect_declares_bool_extraction(self):
        state = build_orchestrator_fsm()["states"][OrchestratorStates.COLLECT]
        fields = {f["field_name"]: f for f in state.get("field_extractions", [])}
        assert fields[ContextKeys.ALL_COLLECTED]["field_type"] == "bool"
        FSMDefinition.model_validate(build_orchestrator_fsm("task"))

    def test_second_round_decision_is_extracted(self):
        # Round 1 says more work is needed, round 2 says done: the run must
        # synthesize after the second delegation, not replay round 1's False.
        workers: list[str] = []

        def worker(subtask: str) -> AgentResult:
            workers.append(subtask)
            return AgentResult(answer="w", success=True)

        llm = _DecisionLLM(ContextKeys.ALL_COLLECTED, [False, True])
        agent = OrchestratorAgent(
            worker_factory=worker,
            config=AgentConfig(max_iterations=12),
            llm_interface=llm,
        )
        result, states = _run_recording_states(agent, "task")

        assert len(workers) == 2
        assert states.count(OrchestratorStates.COLLECT) == 2
        assert result.success
