from __future__ import annotations

"""Tests for MetaBuilderAgent — agentic architecture."""

import pytest

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


class TestTypeDetection:
    """Test artifact type detection from user text."""

    def test_fsm_aliases(self):
        agent = MetaBuilderAgent()
        for text in [
            "build a chatbot",
            "create an FSM",
            "conversation bot",
            "state machine for support",
        ]:
            result = agent._detect_type(text)
            assert result == ArtifactType.FSM, f"'{text}' should resolve to FSM"

    def test_workflow_aliases(self):
        agent = MetaBuilderAgent()
        for text in [
            "build a workflow",
            "data pipeline",
            "automation process",
            "ETL steps",
        ]:
            result = agent._detect_type(text)
            assert result == ArtifactType.WORKFLOW, (
                f"'{text}' should resolve to WORKFLOW"
            )

    def test_agent_aliases(self):
        agent = MetaBuilderAgent()
        for text in [
            "build an agent with tools",
            "react pattern",
            "research agent",
        ]:
            result = agent._detect_type(text)
            assert result == ArtifactType.AGENT, f"'{text}' should resolve to AGENT"

    def test_unknown_defaults_to_fsm(self):
        agent = MetaBuilderAgent()
        assert agent._detect_type("build something amazing") == ArtifactType.FSM

    def test_just_build_defaults_to_fsm(self):
        agent = MetaBuilderAgent()
        assert agent._detect_type("just build it") == ArtifactType.FSM


class TestBuildTrigger:
    """Test that build trigger phrases are detected."""

    def test_build_triggers(self):
        for phrase in ["build it", "go", "done", "approve", "yes", "lgtm"]:
            assert MetaBuilderAgent._is_build_trigger(phrase), (
                f"'{phrase}' should trigger build"
            )

    def test_non_triggers(self):
        for phrase in ["add a state", "I want 3 states", "change the name"]:
            assert not MetaBuilderAgent._is_build_trigger(phrase), (
                f"'{phrase}' should not trigger build"
            )


class TestInternalState:
    def test_initial_state(self):
        agent = MetaBuilderAgent()
        state = agent.get_internal_state()
        assert state["phase"] == "collecting"
        assert state["turn_count"] == 0
        assert state["builder_summary"] is None

    def test_state_with_builder(self):
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


class TestBuildResult:
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
        assert agent._result.final_context["artifact_type"] == "fsm"

    def test_build_result_with_no_builder(self):
        agent = MetaBuilderAgent()
        agent._build_result()
        assert agent._result is not None
        assert agent._result.success is False
        assert agent._result.is_valid is False


class TestStartSendFlow:
    """Test the turn-by-turn conversation flow."""

    def test_start_with_message_detects_type(self):
        agent = MetaBuilderAgent()
        response = agent.start("build me a chatbot")
        assert "FSM" in response
        assert agent._artifact_type == ArtifactType.FSM
        assert agent._started is True

    def test_start_without_message_shows_welcome(self):
        agent = MetaBuilderAgent()
        response = agent.start()
        assert "FSM" in response
        assert "Workflow" in response
        assert "Agent" in response

    def test_send_accumulates_messages(self):
        agent = MetaBuilderAgent()
        agent.start("build a chatbot")
        agent.send("with greeting, help, and goodbye states")
        assert len(agent._messages) == 2

    def test_send_after_complete_raises(self):
        agent = MetaBuilderAgent()
        agent._started = True
        agent._complete = True
        with pytest.raises(MetaBuilderError, match="already completed"):
            agent.send("hello")


class TestCreateBuilder:
    def test_creates_fsm_builder(self):
        from fsm_llm_agents.meta_builders import FSMBuilder

        agent = MetaBuilderAgent()
        builder = agent._create_builder(ArtifactType.FSM)
        assert isinstance(builder, FSMBuilder)

    def test_creates_workflow_builder(self):
        from fsm_llm_agents.meta_builders import WorkflowBuilder

        agent = MetaBuilderAgent()
        builder = agent._create_builder(ArtifactType.WORKFLOW)
        assert isinstance(builder, WorkflowBuilder)

    def test_creates_agent_builder(self):
        from fsm_llm_agents.meta_builders import AgentBuilder

        agent = MetaBuilderAgent()
        builder = agent._create_builder(ArtifactType.AGENT)
        assert isinstance(builder, AgentBuilder)


class TestLegacyFSMDefinition:
    """Test that the legacy FSM definition still loads."""

    def test_builds_fsm_dict(self):
        from fsm_llm_agents.meta_fsm import build_meta_builder_fsm

        fsm = build_meta_builder_fsm()
        assert isinstance(fsm, dict)
        assert "name" in fsm
        assert "states" in fsm
