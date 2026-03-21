from __future__ import annotations

"""Tests for fsm_llm_agents.debate module."""


from fsm_llm.definitions import FSMDefinition
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
