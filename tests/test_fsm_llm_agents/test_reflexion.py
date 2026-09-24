from __future__ import annotations

"""Tests for fsm_llm_agents.reflexion module and Reflexion FSM definition."""

from typing import Any

import pytest

from fsm_llm.definitions import (
    FieldExtractionRequest,
    FieldExtractionResponse,
    FSMDefinition,
    ResponseGenerationRequest,
    ResponseGenerationResponse,
)
from fsm_llm.expressions import evaluate_logic
from fsm_llm.llm import LLMInterface
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
        """Terminal states should have non-empty response_instructions.
        Intermediate states (think, act) may use empty string to skip Pass 2."""
        registry = _make_registry()
        fsm = build_reflexion_fsm(registry)
        for state_name in fsm["states"]:
            state = fsm["states"][state_name]
            if state_name in ("conclude",):
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


# ---------------------------------------------------------------------------
# D-008: conclude needs tool evidence (plan-2026-09-24T091842-c1d5bfbc)
# ---------------------------------------------------------------------------


class _FieldMapLLM(LLMInterface):
    """Mock LLM returning ``field_map[name]`` for every extracted field."""

    def __init__(self, field_map: dict[str, Any]) -> None:
        self.model = "mock-model"
        self.field_map = field_map

    def extract_field(self, request: FieldExtractionRequest) -> FieldExtractionResponse:
        value = self.field_map.get(request.field_name)
        return FieldExtractionResponse(
            field_name=request.field_name,
            value=value,
            confidence=0.9 if value is not None else 0.0,
            reasoning="mock",
            is_valid=value is not None,
        )

    def generate_response(
        self, request: ResponseGenerationRequest
    ) -> ResponseGenerationResponse:
        return ResponseGenerationResponse(
            message="final answer", message_type="response", reasoning="mock"
        )


def _run_recording_states(agent: Any) -> tuple[Any, list[str]]:
    """Run the agent; return (result, state before each loop turn)."""
    states: list[str] = []
    real_hook = agent._on_loop_iteration

    def _hook(api: Any, conv_id: str, iteration: int) -> None:
        states.append(api.get_current_state(conv_id))
        real_hook(api, conv_id, iteration)

    agent._on_loop_iteration = _hook
    return agent.run("What is 2 + 2?"), states


def _conclude_conditions(fsm: dict[str, Any], state: str) -> list[dict[str, Any]]:
    for transition in fsm["states"][state]["transitions"]:
        if transition["target_state"] == "conclude":
            return transition["conditions"]
    raise AssertionError(f"{state} has no conclude edge")


def _conclude_passes(fsm: dict[str, Any], state: str, ctx: dict[str, Any]) -> bool:
    return all(
        bool(evaluate_logic(c["logic"], ctx)) for c in _conclude_conditions(fsm, state)
    )


_ANSWER_FROM_MEMORY: dict[str, Any] = {
    ContextKeys.TOOL_NAME: ContextKeys.NO_TOOL,
    ContextKeys.TOOL_INPUT: {},
    ContextKeys.SHOULD_TERMINATE: True,
    ContextKeys.EVALUATION_PASSED: False,
    ContextKeys.EVALUATION_SCORE: 0.1,
    ContextKeys.REFLECTION: "use a tool",
    ContextKeys.FINAL_ANSWER: "4",
}


class TestReflexionConcludeNeedsEvidence:
    """Turn-1 ``should_terminate=True`` with no tool must not conclude (D-008)."""

    def test_turn_one_terminate_without_tool_goes_to_act(self):
        agent = ReflexionAgent(
            tools=_make_registry(),
            config=AgentConfig(max_iterations=6),
            llm_interface=_FieldMapLLM(_ANSWER_FROM_MEMORY),
        )
        result, states = _run_recording_states(agent)

        assert len(states) >= 2, "Reflexion concluded on turn 1 with no tool"
        assert states[1] == "act"
        assert result.answer
        assert len(states) <= 3 * 6

    @pytest.mark.parametrize("state", ["think", "act"])
    def test_conclude_edge_needs_observation_or_forced_stop(self, state):
        fsm = build_reflexion_fsm(_make_registry())
        terminate = {ContextKeys.SHOULD_TERMINATE: True}

        assert not _conclude_passes(fsm, state, terminate)
        assert _conclude_passes(
            fsm, state, {**terminate, ContextKeys.OBSERVATION_COUNT: 1}
        )
        assert _conclude_passes(
            fsm, state, {**terminate, ContextKeys.MAX_ITERATIONS_REACHED: True}
        )
        assert not _conclude_passes(
            fsm,
            state,
            {ContextKeys.SHOULD_TERMINATE: False, ContextKeys.OBSERVATION_COUNT: 3},
        )

    @pytest.mark.parametrize("approval", [False, True])
    def test_think_keeps_unconditional_act_fallback(self, approval):
        fsm = build_reflexion_fsm(_make_registry(), include_approval_state=approval)
        fallback = [
            t
            for t in fsm["states"]["think"]["transitions"]
            if t["target_state"] == "act"
        ]
        assert len(fallback) == 1
        assert not fallback[0].get("conditions")

    def test_evaluate_conclude_needs_observation_or_forced_stop(self):
        """Review W3: a self-evaluated pass with no tool run must not conclude."""
        fsm = build_reflexion_fsm(_make_registry())
        passed = {ContextKeys.EVALUATION_PASSED: True}

        assert not _conclude_passes(fsm, "evaluate", passed)
        assert _conclude_passes(
            fsm, "evaluate", {**passed, ContextKeys.OBSERVATION_COUNT: 1}
        )
        assert _conclude_passes(
            fsm, "evaluate", {**passed, ContextKeys.MAX_ITERATIONS_REACHED: True}
        )
        assert not _conclude_passes(
            fsm,
            "evaluate",
            {ContextKeys.EVALUATION_PASSED: False, ContextKeys.OBSERVATION_COUNT: 3},
        )

    def test_self_passed_memory_answer_reflects_until_the_forced_stop(self):
        """Pre-fix: think, act, evaluate, conclude with zero tools (3 turns)."""
        field_map = {**_ANSWER_FROM_MEMORY, ContextKeys.EVALUATION_PASSED: True}
        agent = ReflexionAgent(
            tools=_make_registry(),
            config=AgentConfig(max_iterations=8),
            llm_interface=_FieldMapLLM(field_map),
        )
        result, states = _run_recording_states(agent)

        assert "reflect" in states, states
        assert result.final_context.get(ContextKeys.MAX_ITERATIONS_REACHED) is True
        assert len(states) <= 3 * 8
