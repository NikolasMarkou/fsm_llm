from __future__ import annotations

"""Tests for meta-agent FSM definitions module.

Tests that the MetaBuilderAgent FSM definition can be built from
fsm_llm_agents.meta_fsm.
"""


class TestFSMDefinitionsModule:
    def test_importable(self):
        from fsm_llm_agents.meta_fsm import build_meta_builder_fsm  # noqa: F401

    def test_builds_fsm_dict(self):
        from fsm_llm_agents.meta_fsm import build_meta_builder_fsm

        fsm = build_meta_builder_fsm()
        assert isinstance(fsm, dict)
        assert "name" in fsm
        assert "states" in fsm
