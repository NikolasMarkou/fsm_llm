from __future__ import annotations

"""Tests for meta-agent FSM definition generation."""

from fsm_llm.definitions import FSMDefinition
from fsm_llm_meta.constants import MetaStates
from fsm_llm_meta.fsm_definitions import build_meta_fsm


class TestBuildMetaFSM:
    def test_returns_dict(self):
        fsm = build_meta_fsm()
        assert isinstance(fsm, dict)

    def test_has_required_keys(self):
        fsm = build_meta_fsm()
        assert "name" in fsm
        assert "description" in fsm
        assert "initial_state" in fsm
        assert "states" in fsm
        assert "persona" in fsm

    def test_initial_state_is_welcome(self):
        fsm = build_meta_fsm()
        assert fsm["initial_state"] == MetaStates.WELCOME

    def test_all_states_present(self):
        fsm = build_meta_fsm()
        expected_states = {
            MetaStates.WELCOME,
            MetaStates.CLASSIFY,
            MetaStates.GATHER_OVERVIEW,
            MetaStates.DESIGN_STRUCTURE,
            MetaStates.DEFINE_CONNECTIONS,
            MetaStates.REVIEW,
            MetaStates.OUTPUT,
        }
        assert set(fsm["states"].keys()) == expected_states

    def test_output_is_terminal(self):
        fsm = build_meta_fsm()
        output_state = fsm["states"][MetaStates.OUTPUT]
        assert output_state["transitions"] == []

    def test_review_has_two_transitions(self):
        fsm = build_meta_fsm()
        review = fsm["states"][MetaStates.REVIEW]
        assert len(review["transitions"]) == 2
        targets = {t["target_state"] for t in review["transitions"]}
        assert MetaStates.OUTPUT in targets
        assert MetaStates.DESIGN_STRUCTURE in targets

    def test_validates_as_fsm_definition(self):
        """The generated FSM dict should produce a valid FSMDefinition."""
        fsm = build_meta_fsm()
        definition = FSMDefinition(**fsm)
        assert definition.name == "meta_agent"
        assert len(definition.states) == 7

    def test_all_transitions_reference_valid_states(self):
        fsm = build_meta_fsm()
        state_ids = set(fsm["states"].keys())
        for state_id, state in fsm["states"].items():
            for transition in state["transitions"]:
                assert transition["target_state"] in state_ids, (
                    f"State '{state_id}' has transition to unknown "
                    f"state '{transition['target_state']}'"
                )

    def test_non_terminal_states_have_transitions(self):
        fsm = build_meta_fsm()
        for state_id, state in fsm["states"].items():
            if state_id != MetaStates.OUTPUT:
                assert len(state["transitions"]) > 0, (
                    f"Non-terminal state '{state_id}' has no transitions"
                )
