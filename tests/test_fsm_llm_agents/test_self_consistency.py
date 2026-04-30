from __future__ import annotations

"""Tests for fsm_llm_agents.self_consistency module."""

import pytest

from fsm_llm.dialog.definitions import FSMDefinition
from fsm_llm_agents.constants import ContextKeys, Defaults, SelfConsistencyStates
from fsm_llm_agents.definitions import AgentConfig
from fsm_llm_agents.exceptions import AgentError
from fsm_llm_agents.fsm_definitions import build_self_consistency_fsm
from fsm_llm_agents.self_consistency import SelfConsistencyAgent, _majority_vote


class TestSelfConsistencyCreation:
    """Tests for SelfConsistencyAgent initialization."""

    def test_create_with_defaults(self):
        agent = SelfConsistencyAgent()
        assert agent.num_samples == Defaults.NUM_SAMPLES
        assert agent.config is not None
        assert agent.aggregation_fn is _majority_vote

    def test_create_with_custom_num_samples(self):
        agent = SelfConsistencyAgent(num_samples=7)
        assert agent.num_samples == 7

    def test_create_with_custom_aggregation_fn(self):
        custom_fn = lambda samples: max(samples, key=len)  # noqa: E731
        agent = SelfConsistencyAgent(aggregation_fn=custom_fn)
        assert agent.aggregation_fn is custom_fn

    def test_create_with_num_samples_zero_raises(self):
        with pytest.raises(AgentError, match="at least 1"):
            SelfConsistencyAgent(num_samples=0)

    def test_create_with_num_samples_negative_raises(self):
        with pytest.raises(AgentError, match="at least 1"):
            SelfConsistencyAgent(num_samples=-5)

    def test_create_with_config_override(self):
        config = AgentConfig(max_iterations=3, model="gpt-4o-mini")
        agent = SelfConsistencyAgent(config=config, num_samples=3)
        assert agent.config.max_iterations == 3
        assert agent.config.model == "gpt-4o-mini"
        assert agent.num_samples == 3

    def test_create_with_num_samples_one(self):
        agent = SelfConsistencyAgent(num_samples=1)
        assert agent.num_samples == 1

    def test_no_tool_registry_needed(self):
        """SelfConsistencyAgent does not require a ToolRegistry."""
        agent = SelfConsistencyAgent()
        assert not hasattr(agent, "tools") or agent.__dict__.get("tools") is None

    def test_has_run_method(self):
        agent = SelfConsistencyAgent()
        assert callable(getattr(agent, "run", None))


class TestSelfConsistencyFSM:
    """Tests for build_self_consistency_fsm function."""

    def test_basic_fsm_structure(self):
        fsm = build_self_consistency_fsm()
        assert fsm["name"] == "self_consistency_sample"
        assert fsm["initial_state"] == "generate"
        assert "generate" in fsm["states"]

    def test_fsm_is_valid_definition(self):
        """The generated FSM should be parseable as an FSMDefinition."""
        fsm = build_self_consistency_fsm()
        fsm_def = FSMDefinition(**fsm)
        assert fsm_def.name == "self_consistency_sample"

    def test_generate_state_is_terminal(self):
        fsm = build_self_consistency_fsm()
        assert fsm["states"]["generate"]["transitions"] == []

    def test_only_one_state(self):
        fsm = build_self_consistency_fsm()
        assert len(fsm["states"]) == 1
        assert "generate" in fsm["states"]

    def test_custom_task_description(self):
        fsm = build_self_consistency_fsm(task_description="What is 2+2?")
        assert fsm["description"] == "What is 2+2?"

    def test_default_task_description(self):
        fsm = build_self_consistency_fsm()
        assert fsm["description"] == "Self-consistency single sample"

    def test_persona_mentions_precise(self):
        fsm = build_self_consistency_fsm()
        assert "precise" in fsm["persona"].lower()

    def test_generate_state_has_extraction_instructions(self):
        fsm = build_self_consistency_fsm()
        generate = fsm["states"]["generate"]
        assert "extraction_instructions" in generate
        assert len(generate["extraction_instructions"]) > 0


class TestSelfConsistencyConstants:
    """Tests for self-consistency related constants."""

    def test_self_consistency_states_generate(self):
        assert SelfConsistencyStates.GENERATE == "generate"

    def test_self_consistency_states_aggregate(self):
        assert SelfConsistencyStates.AGGREGATE == "aggregate"

    def test_context_keys_samples(self):
        assert ContextKeys.SAMPLES == "samples"

    def test_context_keys_aggregated_answer(self):
        assert ContextKeys.AGGREGATED_ANSWER == "aggregated_answer"

    def test_defaults_num_samples(self):
        assert Defaults.NUM_SAMPLES == 5

    def test_defaults_sample_temperature_range(self):
        assert isinstance(Defaults.SAMPLE_TEMPERATURE_RANGE, tuple)
        assert len(Defaults.SAMPLE_TEMPERATURE_RANGE) == 2
        low, high = Defaults.SAMPLE_TEMPERATURE_RANGE
        assert low < high


class TestMajorityVote:
    """Tests for the _majority_vote default aggregation function."""

    def test_majority_vote_simple(self):
        result = _majority_vote(["Paris", "Paris", "London"])
        assert result == "Paris"

    def test_majority_vote_all_same(self):
        result = _majority_vote(["yes", "yes", "yes"])
        assert result == "yes"

    def test_majority_vote_empty_list(self):
        result = _majority_vote([])
        assert result == ""

    def test_majority_vote_whitespace_normalization(self):
        result = _majority_vote(["  Paris ", "Paris", " Paris"])
        assert result == "Paris"

    def test_majority_vote_single_element(self):
        result = _majority_vote(["only"])
        assert result == "only"

    def test_majority_vote_skips_empty_strings(self):
        result = _majority_vote(["", "", "answer", "answer"])
        assert result == "answer"

    def test_majority_vote_all_empty(self):
        result = _majority_vote(["", "  ", ""])
        assert result == ""
