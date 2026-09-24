from __future__ import annotations

"""Tests for fsm_llm_agents.debate module."""


import re
from typing import Any

import pytest

from fsm_llm.definitions import (
    DataExtractionResponse,
    FieldExtractionRequest,
    FieldExtractionResponse,
    FSMDefinition,
    ResponseGenerationRequest,
    ResponseGenerationResponse,
)
from fsm_llm.llm import LLMInterface
from fsm_llm_agents.constants import ContextKeys, DebateStates, Defaults, HandlerNames
from fsm_llm_agents.debate import DebateAgent
from fsm_llm_agents.definitions import AgentConfig, DebateRound
from fsm_llm_agents.fsm_definitions import build_debate_fsm


class TestDebateAgentCreation:
    """Tests for DebateAgent initialization."""

    def test_create_with_defaults(self):
        agent = DebateAgent()
        assert agent.num_rounds == Defaults.MAX_DEBATE_ROUNDS
        assert agent.config is not None

    def test_create_with_custom_num_rounds(self):
        agent = DebateAgent(num_rounds=5)
        assert agent.num_rounds == 5

    def test_create_with_custom_personas(self):
        agent = DebateAgent(
            proposer_persona="I am proposer",
            critic_persona="I am critic",
            judge_persona="I am judge",
        )
        assert agent.proposer_persona == "I am proposer"
        assert agent.critic_persona == "I am critic"
        assert agent.judge_persona == "I am judge"

    def test_create_with_default_personas(self):
        agent = DebateAgent()
        assert "advocate" in agent.proposer_persona.lower()
        assert "critic" in agent.critic_persona.lower()
        assert "judge" in agent.judge_persona.lower()

    def test_create_with_config_override(self):
        config = AgentConfig(max_iterations=5, model="gpt-4o")
        agent = DebateAgent(config=config)
        assert agent.config.max_iterations == 5
        assert agent.config.model == "gpt-4o"

    def test_num_rounds_clamped_to_minimum_one(self):
        agent = DebateAgent(num_rounds=0)
        assert agent.num_rounds == 1

    def test_num_rounds_negative_clamped_to_one(self):
        agent = DebateAgent(num_rounds=-3)
        assert agent.num_rounds == 1

    def test_no_tool_registry_needed(self):
        """DebateAgent does not require a ToolRegistry."""
        agent = DebateAgent()
        assert not hasattr(agent, "tools") or agent.__dict__.get("tools") is None

    def test_has_run_method(self):
        agent = DebateAgent()
        assert callable(getattr(agent, "run", None))

    def test_run_writes_no_per_run_budget_onto_self(self, monkeypatch):
        """PT-02: run() computes the FSM budget without storing it on self."""
        agent = DebateAgent(num_rounds=2)
        seen: dict = {}

        def _fake_standard_run(*args, **kwargs):
            seen.update(kwargs)
            return "done"

        monkeypatch.setattr(agent, "_standard_run", _fake_standard_run)
        before = set(vars(agent))
        assert agent.run("topic") == "done"
        assert "_max_fsm_iterations" not in vars(agent)
        assert set(vars(agent)) - before == set()
        assert seen["max_iterations"] == (
            2 * Defaults.FSM_BUDGET_MULTIPLIER * Defaults.DEBATE_STATES_PER_ROUND
        )


class TestDebateFSM:
    """Tests for build_debate_fsm function."""

    def test_basic_fsm_structure(self):
        fsm = build_debate_fsm()
        assert fsm["name"] == "debate_agent"
        assert fsm["initial_state"] == "propose"
        assert len(fsm["states"]) == 5

    def test_fsm_has_all_five_states(self):
        fsm = build_debate_fsm()
        expected = {"propose", "critique", "counter", "judge", "conclude"}
        assert set(fsm["states"].keys()) == expected

    def test_fsm_is_valid_definition(self):
        """The generated FSM should be parseable as an FSMDefinition."""
        fsm = build_debate_fsm()
        fsm_def = FSMDefinition(**fsm)
        assert fsm_def.name == "debate_agent"

    def test_propose_transitions_to_critique(self):
        fsm = build_debate_fsm()
        targets = {t["target_state"] for t in fsm["states"]["propose"]["transitions"]}
        assert "critique" in targets

    def test_critique_transitions_to_counter(self):
        fsm = build_debate_fsm()
        targets = {t["target_state"] for t in fsm["states"]["critique"]["transitions"]}
        assert "counter" in targets

    def test_counter_transitions_to_judge(self):
        fsm = build_debate_fsm()
        targets = {t["target_state"] for t in fsm["states"]["counter"]["transitions"]}
        assert "judge" in targets

    def test_judge_transitions_to_conclude_and_propose(self):
        fsm = build_debate_fsm()
        targets = {t["target_state"] for t in fsm["states"]["judge"]["transitions"]}
        assert "conclude" in targets
        assert "propose" in targets

    def test_conclude_is_terminal(self):
        fsm = build_debate_fsm()
        assert fsm["states"]["conclude"]["transitions"] == []

    def test_custom_task_description(self):
        fsm = build_debate_fsm(task_description="Debate about AI safety")
        assert fsm["description"] == "Debate about AI safety"

    def test_default_task_description(self):
        fsm = build_debate_fsm()
        assert fsm["description"] == "Debate agent"

    def test_conclude_priority_higher_than_propose(self):
        """Lower priority number = higher confidence in TransitionEvaluator."""
        fsm = build_debate_fsm()
        judge_transitions = fsm["states"]["judge"]["transitions"]
        conclude_priority = None
        propose_priority = None
        for t in judge_transitions:
            if t["target_state"] == "conclude":
                conclude_priority = t["priority"]
            elif t["target_state"] == "propose":
                propose_priority = t["priority"]
        assert conclude_priority is not None
        assert propose_priority is not None
        assert conclude_priority < propose_priority


class TestDebateRoundModel:
    """Tests for DebateRound Pydantic model."""

    def test_basic_creation(self):
        round_entry = DebateRound(round_num=1)
        assert round_entry.round_num == 1
        assert round_entry.proposition == ""
        assert round_entry.critique == ""
        assert round_entry.counter_argument == ""
        assert round_entry.judge_verdict == ""

    def test_creation_with_all_fields(self):
        round_entry = DebateRound(
            round_num=2,
            proposition="Cities should ban cars",
            critique="This ignores accessibility needs",
            counter_argument="Alternative transport solutions exist",
            judge_verdict="Counter-argument is stronger",
        )
        assert round_entry.round_num == 2
        assert "ban cars" in round_entry.proposition
        assert "accessibility" in round_entry.critique

    def test_serialization(self):
        round_entry = DebateRound(
            round_num=1,
            proposition="Test proposition",
        )
        data = round_entry.model_dump(mode="json")
        assert data["round_num"] == 1
        assert data["proposition"] == "Test proposition"
        assert "counter_argument" in data


class TestDebateConstants:
    """Tests for debate-related constants."""

    def test_debate_states_propose(self):
        assert DebateStates.PROPOSE == "propose"

    def test_debate_states_critique(self):
        assert DebateStates.CRITIQUE == "critique"

    def test_debate_states_counter(self):
        assert DebateStates.COUNTER == "counter"

    def test_debate_states_judge(self):
        assert DebateStates.JUDGE == "judge"

    def test_debate_states_conclude(self):
        assert DebateStates.CONCLUDE == "conclude"

    def test_context_keys_proposition(self):
        assert ContextKeys.PROPOSITION == "proposition"

    def test_context_keys_critique(self):
        assert ContextKeys.CRITIQUE == "critique"

    def test_context_keys_counter_argument(self):
        assert ContextKeys.COUNTER_ARGUMENT == "counter_argument"

    def test_context_keys_judge_verdict(self):
        assert ContextKeys.JUDGE_VERDICT == "judge_verdict"

    def test_context_keys_debate_rounds(self):
        assert ContextKeys.DEBATE_ROUNDS == "debate_rounds"

    def test_context_keys_current_round(self):
        assert ContextKeys.CURRENT_ROUND == "current_round"

    def test_context_keys_consensus_reached(self):
        assert ContextKeys.CONSENSUS_REACHED == "consensus_reached"

    def test_defaults_max_debate_rounds(self):
        assert Defaults.MAX_DEBATE_ROUNDS == 3

    def test_handler_name_debate_judge(self):
        assert HandlerNames.DEBATE_JUDGE == "DebateJudge"


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


class TestConsensusTypedExtraction:
    """D-009: ``judge`` declares ``consensus_reached`` as a typed bool extraction."""

    def test_judge_declares_bool_extraction(self):
        state = build_debate_fsm()["states"][DebateStates.JUDGE]
        fields = {f["field_name"]: f for f in state.get("field_extractions", [])}
        assert fields[ContextKeys.CONSENSUS_REACHED]["field_type"] == "bool"
        FSMDefinition.model_validate(build_debate_fsm("topic", max_rounds=2))

    def test_round_one_consensus_concludes(self):
        agent = DebateAgent(
            num_rounds=3,
            llm_interface=_DecisionLLM(ContextKeys.CONSENSUS_REACHED, [True]),
        )
        _, states = _run_recording_states(agent, "topic")

        assert states.count(DebateStates.JUDGE) == 1

    def test_second_round_consensus_concludes(self):
        agent = DebateAgent(
            num_rounds=3,
            llm_interface=_DecisionLLM(ContextKeys.CONSENSUS_REACHED, [False, True]),
        )
        _, states = _run_recording_states(agent, "topic")

        assert states.count(DebateStates.JUDGE) == 2

    @pytest.mark.parametrize("rounds", [1, 2, 3])
    def test_no_consensus_stops_at_round_cap(self, rounds):
        # The judge extracts False every round; the round cap must still win
        # (at HEAD, 1 round stopped only because consensus was seeded False
        # and never extracted).
        agent = DebateAgent(
            num_rounds=rounds,
            llm_interface=_DecisionLLM(ContextKeys.CONSENSUS_REACHED, [False]),
        )
        result, states = _run_recording_states(agent, "topic")

        assert states.count(DebateStates.JUDGE) == rounds
        assert len(result.final_context[ContextKeys.DEBATE_ROUNDS]) == rounds


class TestIterationLimiterBoundary:
    """D-013 (item 15): the early limiter fires at count ``max - 1``, not before."""

    def test_limiter_triggers_at_max_minus_one_not_max_minus_two(self):
        agent = DebateAgent(num_rounds=2)
        limiter = agent._make_iteration_limiter()
        limit = agent._fsm_budget()

        before = limiter({ContextKeys.ITERATION_COUNT: limit - 3})
        assert before == {ContextKeys.ITERATION_COUNT: limit - 2}

        at = limiter({ContextKeys.ITERATION_COUNT: limit - 2})
        assert at[ContextKeys.ITERATION_COUNT] == limit - 1
        assert at[ContextKeys.CONSENSUS_REACHED] is True
        assert at[ContextKeys.SHOULD_TERMINATE] is True
