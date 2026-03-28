from __future__ import annotations

"""Tests for MetaBuilderAgent.

Tests the FSM-driven architecture: API + handlers + classification_extractions.
"""

from unittest.mock import MagicMock

import pytest

from fsm_llm_agents.constants import MetaBuilderStates
from fsm_llm_agents.definitions import (
    ArtifactType,
    MetaBuilderConfig,
    MetaBuilderResult,
)
from fsm_llm_agents.exceptions import MetaBuilderError
from fsm_llm_agents.meta_builder import MetaBuilderAgent


class TestMetaAgentInit:
    def test_default_config(self):
        agent = MetaBuilderAgent()
        assert agent.meta_config.temperature == 0.7
        assert agent.meta_config.max_turns == 50

    def test_custom_config(self):
        config = MetaBuilderConfig(model="gpt-4o", max_turns=10)
        agent = MetaBuilderAgent(config=config)
        assert agent.meta_config.model == "gpt-4o"

    def test_initial_state(self):
        agent = MetaBuilderAgent()
        assert not agent.is_complete()
        assert agent._builder is None
        assert agent._started is False


class TestMetaAgentLifecycle:
    def test_send_before_start_raises(self):
        agent = MetaBuilderAgent()
        with pytest.raises(MetaBuilderError, match="not been started"):
            agent.send("hello")

    def test_get_result_before_complete_raises(self):
        agent = MetaBuilderAgent()
        with pytest.raises(MetaBuilderError, match="not complete"):
            agent.get_result()

    def test_double_start_raises(self):
        agent = MetaBuilderAgent()
        agent._started = True
        with pytest.raises(MetaBuilderError, match="already been started"):
            agent.start()

    def test_max_turns_exceeded(self):
        config = MetaBuilderConfig(max_turns=1)
        agent = MetaBuilderAgent(config=config)
        agent._started = True
        agent._turn_count = 1
        with pytest.raises(MetaBuilderError, match="Maximum turns"):
            agent.send("hello")


class TestMetaAgentBuild:
    """Test the _do_build handler logic directly."""

    def test_do_build_creates_builder(self):
        agent = MetaBuilderAgent()

        # Mock _call_llm_json to return a complete FSM spec
        agent._call_llm_json = MagicMock(
            return_value={
                "name": "Bot",
                "description": "A bot",
                "initial_state": "start",
                "states": [
                    {"id": "start", "description": "Greeting", "purpose": "Greet"},
                    {"id": "end", "description": "Goodbye", "purpose": "End"},
                ],
                "transitions": [
                    {"source": "start", "target": "end", "description": "Done"}
                ],
            }
        )

        # Call _do_build directly with context
        context = {
            "artifact_type": "fsm",
            "artifact_name": "Bot",
            "artifact_description": "A bot",
        }
        result = agent._do_build(context)

        assert agent._builder is not None
        assert agent._artifact_type == ArtifactType.FSM
        assert len(agent._builder.states) == 2
        assert "builder_summary" in result

    def test_do_build_with_dict_states(self):
        """Build handles dict-format states from LLM."""
        agent = MetaBuilderAgent()

        agent._call_llm_json = MagicMock(
            return_value={
                "name": "Bot",
                "description": "A bot",
                "initial_state": "greeting",
                "states": {
                    "greeting": {
                        "description": "Hello",
                        "purpose": "Greet user",
                        "transitions": [
                            {"target_state": "end", "description": "Done"}
                        ],
                    },
                    "end": {
                        "description": "Goodbye",
                        "purpose": "End conversation",
                    },
                },
            }
        )

        context = {
            "artifact_type": "fsm",
            "artifact_name": "Bot",
            "artifact_description": "A bot",
        }
        agent._do_build(context)

        assert len(agent._builder.states) == 2
        assert "greeting" in agent._builder.states
        assert "end" in agent._builder.states
        # Verify embedded transition was extracted
        greeting_transitions = agent._builder.states["greeting"]["transitions"]
        assert len(greeting_transitions) == 1
        assert greeting_transitions[0]["target_state"] == "end"

    def test_do_build_failure_records_error(self):
        agent = MetaBuilderAgent()

        # Mock _call_llm_json to return empty (simulating LLM failure)
        agent._call_llm_json = MagicMock(return_value={})

        context = {"artifact_type": "fsm"}
        agent._do_build(context)

        assert len(agent._build_errors) > 0
        assert agent._builder is not None  # Builder created even on failure

    def test_do_build_with_user_request_as_description(self):
        """User request is used as description when no explicit description."""
        agent = MetaBuilderAgent()
        agent._call_llm_json = MagicMock(return_value={})

        context = {
            "artifact_type": "fsm",
            "user_request": "Build me a customer support chatbot",
        }
        agent._do_build(context)

        assert agent._requirements.get("artifact_description") == (
            "Build me a customer support chatbot"
        )


class TestMetaAgentRevision:
    """Test the _do_revision handler logic directly."""

    def _make_agent_with_builder(self) -> MetaBuilderAgent:
        from fsm_llm_agents.meta_builders import FSMBuilder

        agent = MetaBuilderAgent()
        agent._artifact_type = ArtifactType.FSM
        builder = FSMBuilder()
        builder.set_overview("Bot", "A bot")
        builder.add_state("s1", "State 1", "Purpose")
        agent._builder = builder
        return agent

    def test_revision_applies_new_spec(self):
        agent = self._make_agent_with_builder()
        agent._call_llm_json = MagicMock(
            return_value={
                "name": "Bot",
                "description": "A revised bot",
                "initial_state": "s1",
                "states": [
                    {"id": "s1", "description": "State 1", "purpose": "P1"},
                    {"id": "s2", "description": "State 2", "purpose": "P2"},
                ],
                "transitions": [
                    {"source": "s1", "target": "s2", "description": "Go"}
                ],
            }
        )

        context = {"revision_request": "add another state"}
        result = agent._do_revision(context)

        assert len(agent._builder.states) == 2
        assert "builder_summary" in result

    def test_revision_empty_spec_keeps_original(self):
        agent = self._make_agent_with_builder()
        agent._call_llm_json = MagicMock(return_value={})

        original_states = dict(agent._builder.states)
        context = {"revision_request": "add state"}
        agent._do_revision(context)

        assert agent._builder.states == original_states

    def test_revision_without_builder_returns_empty(self):
        agent = MetaBuilderAgent()
        result = agent._do_revision({"revision_request": "change something"})
        assert result == {}


class TestMetaAgentInternalState:
    def test_internal_state_initial(self):
        agent = MetaBuilderAgent()
        state = agent.get_internal_state()
        assert state["phase"] == MetaBuilderStates.INTAKE
        assert state["turn_count"] == 0
        assert state["builder_summary"] is None

    def test_internal_state_with_builder(self):
        from fsm_llm_agents.meta_builders import FSMBuilder

        agent = MetaBuilderAgent()
        agent._artifact_type = ArtifactType.FSM
        builder = FSMBuilder()
        builder.set_overview("Bot", "Desc")
        agent._builder = builder

        state = agent.get_internal_state()
        assert state["artifact_type"] == "fsm"
        assert state["builder_summary"] is not None


class TestMetaAgentOutput:
    def test_output_module_imports(self):
        from fsm_llm_agents.meta_output import (
            format_artifact_json,
            format_summary,
            save_artifact,
        )

        assert callable(format_artifact_json)
        assert callable(format_summary)
        assert callable(save_artifact)

    def test_format_artifact_json(self):
        from fsm_llm_agents.meta_output import format_artifact_json

        result = format_artifact_json({"name": "test", "states": {}})
        assert '"name": "test"' in result

    def test_format_summary(self):
        from fsm_llm_agents.meta_output import format_summary

        result = MetaBuilderResult(
            artifact_type=ArtifactType.FSM,
            artifact={"name": "Bot"},
            is_valid=True,
            conversation_turns=5,
        )
        summary = format_summary(result)
        assert "fsm" in summary
        assert "Bot" in summary
        assert "5" in summary

    def test_save_artifact(self, tmp_path):
        from fsm_llm_agents.meta_output import save_artifact

        artifact = {"name": "test", "states": {}}
        path = save_artifact(artifact, tmp_path / "test.json")
        assert path.exists()
        content = path.read_text()
        assert '"name": "test"' in content


class TestMetaAgentImports:
    """Verify the public API is importable."""

    def test_main_imports(self):
        import fsm_llm_agents

        assert hasattr(fsm_llm_agents, "MetaBuilderAgent")
        assert hasattr(fsm_llm_agents, "FSMBuilder")
        assert hasattr(fsm_llm_agents, "WorkflowBuilder")
        assert hasattr(fsm_llm_agents, "AgentBuilder")
        assert hasattr(fsm_llm_agents, "ArtifactType")
        assert hasattr(fsm_llm_agents, "MetaBuilderConfig")
        assert hasattr(fsm_llm_agents, "MetaBuilderResult")
        assert hasattr(fsm_llm_agents, "MetaBuilderError")
        assert hasattr(fsm_llm_agents, "create_builder_tools")
        assert hasattr(fsm_llm_agents, "create_fsm_tools")

    def test_version(self):
        from fsm_llm_agents import __version__

        assert isinstance(__version__, str)


class TestTypeAliasResolution:
    """Test that artifact type aliases are correctly resolved."""

    def test_fsm_aliases(self):
        agent = MetaBuilderAgent()
        for alias in ["fsm", "chatbot", "conversation", "state machine", "bot"]:
            result = agent._resolve_artifact_type(alias)
            assert result == ArtifactType.FSM, f"'{alias}' should resolve to FSM"

    def test_workflow_aliases(self):
        agent = MetaBuilderAgent()
        for alias in ["workflow", "pipeline", "process", "automation"]:
            result = agent._resolve_artifact_type(alias)
            assert result == ArtifactType.WORKFLOW, (
                f"'{alias}' should resolve to WORKFLOW"
            )

    def test_agent_aliases(self):
        agent = MetaBuilderAgent()
        for alias in ["agent", "tools", "react", "agentic"]:
            result = agent._resolve_artifact_type(alias)
            assert result == ArtifactType.AGENT, f"'{alias}' should resolve to AGENT"

    def test_unknown_returns_none(self):
        agent = MetaBuilderAgent()
        assert agent._resolve_artifact_type("nonsense") is None

    def test_none_returns_none(self):
        agent = MetaBuilderAgent()
        assert agent._resolve_artifact_type(None) is None


class TestBuildResult:
    """Test _build_result() method."""

    def test_build_result_with_valid_builder(self):
        from fsm_llm_agents.meta_builders import FSMBuilder

        agent = MetaBuilderAgent()
        agent._artifact_type = ArtifactType.FSM
        builder = FSMBuilder()
        builder.set_overview("Bot", "A bot")
        builder.add_state("start", "Start", "Begin")
        builder.set_initial_state("start")
        agent._builder = builder

        agent._build_result()
        assert agent._result is not None
        assert agent._result.artifact_type == ArtifactType.FSM
        assert "Bot" in agent._result.artifact_json

    def test_build_result_with_no_builder(self):
        agent = MetaBuilderAgent()
        agent._build_result()
        assert agent._result is not None
        assert agent._result.success is False
        assert agent._result.is_valid is False


class TestFSMDefinition:
    """Test that the FSM definition is properly structured."""

    def test_builds_fsm_dict(self):
        from fsm_llm_agents.meta_fsm import build_meta_builder_fsm

        fsm = build_meta_builder_fsm()
        assert isinstance(fsm, dict)
        assert "name" in fsm
        assert "states" in fsm
        assert fsm["initial_state"] == MetaBuilderStates.INTAKE

    def test_has_three_states(self):
        from fsm_llm_agents.meta_fsm import build_meta_builder_fsm

        fsm = build_meta_builder_fsm()
        states = fsm["states"]
        assert MetaBuilderStates.INTAKE in states
        assert MetaBuilderStates.REVIEW in states
        assert MetaBuilderStates.OUTPUT in states
        assert len(states) == 3

    def test_intake_has_classification_extractions(self):
        from fsm_llm_agents.meta_fsm import build_meta_builder_fsm

        fsm = build_meta_builder_fsm()
        intake = fsm["states"][MetaBuilderStates.INTAKE]
        assert "classification_extractions" in intake
        fields = [
            ce["field_name"] for ce in intake["classification_extractions"]
        ]
        assert "artifact_type" in fields

    def test_review_has_classification_extractions(self):
        from fsm_llm_agents.meta_fsm import build_meta_builder_fsm

        fsm = build_meta_builder_fsm()
        review = fsm["states"][MetaBuilderStates.REVIEW]
        assert "classification_extractions" in review
        fields = [
            ce["field_name"] for ce in review["classification_extractions"]
        ]
        assert "review_decision" in fields

    def test_output_is_terminal(self):
        from fsm_llm_agents.meta_fsm import build_meta_builder_fsm

        fsm = build_meta_builder_fsm()
        output = fsm["states"][MetaBuilderStates.OUTPUT]
        assert output["transitions"] == []
