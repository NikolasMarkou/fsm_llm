from __future__ import annotations

"""Tests for MetaAgent."""

from unittest.mock import MagicMock, patch

import pytest

from fsm_llm_meta.agent import MetaAgent
from fsm_llm_meta.constants import MetaPhases
from fsm_llm_meta.definitions import ArtifactType, MetaAgentConfig, MetaAgentResult
from fsm_llm_meta.exceptions import MetaAgentError


class TestMetaAgentInit:
    def test_default_config(self):
        agent = MetaAgent()
        assert agent.config.temperature == 0.7
        assert agent.config.max_turns == 50

    def test_custom_config(self):
        config = MetaAgentConfig(model="gpt-4o", max_turns=10)
        agent = MetaAgent(config=config)
        assert agent.config.model == "gpt-4o"

    def test_initial_state(self):
        agent = MetaAgent()
        assert not agent.is_complete()
        assert agent._phase == MetaPhases.INTAKE
        assert agent._builder is None


class TestMetaAgentLifecycle:
    def test_send_before_start_raises(self):
        agent = MetaAgent()
        with pytest.raises(MetaAgentError, match="not been started"):
            agent.send("hello")

    def test_get_result_before_complete_raises(self):
        agent = MetaAgent()
        with pytest.raises(MetaAgentError, match="not complete"):
            agent.get_result()

    def test_double_start_raises(self):
        agent = MetaAgent()
        agent._started = True
        with pytest.raises(MetaAgentError, match="already been started"):
            agent.start()

    def test_send_after_done_raises(self):
        agent = MetaAgent()
        agent._started = True
        agent._phase = MetaPhases.DONE
        with pytest.raises(MetaAgentError, match="already completed"):
            agent.send("hello")

    def test_max_turns_exceeded(self):
        config = MetaAgentConfig(max_turns=1)
        agent = MetaAgent(config=config)
        agent._started = True
        agent._turn_count = 1
        with pytest.raises(MetaAgentError, match="Maximum turns"):
            agent.send("hello")


class TestMetaAgentIntake:
    """Test intake phase logic."""

    def _make_agent(self) -> MetaAgent:
        agent = MetaAgent()
        agent._started = True
        agent._llm = MagicMock()
        return agent

    def test_welcome_message_when_no_initial(self):
        """start() with no message returns welcome."""
        agent = MetaAgent()
        with patch.object(MetaAgent, "_handle_intake") as mock:
            # Bypass LLM init
            agent._started = False
            with patch(
                "fsm_llm_meta.agent.LiteLLMInterface"
            ):
                response = agent.start()
        assert "FSM" in response
        assert "Workflow" in response
        assert "Agent" in response

    def test_intake_with_complete_info(self):
        """Complete info in one message triggers build."""
        agent = self._make_agent()

        # Mock extraction to return complete requirements
        agent._llm.extract_data.return_value = MagicMock(
            extracted_data={
                "artifact_type": "fsm",
                "artifact_name": "TestBot",
                "artifact_description": "A test bot",
                "artifact_persona": "Friendly",
                "components": ["greeting state", "farewell state"],
            }
        )

        # Mock the build phase
        with patch.object(agent, "_do_build", return_value="Review presentation"):
            response = agent._handle_intake(
                "Build me an FSM called TestBot for testing"
            )
        assert response == "Review presentation"
        assert agent._artifact_type == ArtifactType.FSM

    def test_intake_missing_name_asks_followup(self):
        """Missing name triggers follow-up question."""
        agent = self._make_agent()

        agent._llm.extract_data.return_value = MagicMock(
            extracted_data={
                "artifact_type": "fsm",
                "artifact_name": None,
                "artifact_description": None,
            }
        )

        response = agent._handle_intake("I want to build a chatbot")
        assert agent._phase == MetaPhases.INTAKE  # Still in intake
        assert "name" in response.lower()

    def test_intake_extraction_failure_graceful(self):
        """Extraction failure doesn't crash."""
        agent = self._make_agent()
        agent._llm.extract_data.side_effect = Exception("LLM down")

        response = agent._handle_intake("Build something")
        assert agent._phase == MetaPhases.INTAKE


class TestMetaAgentReview:
    """Test review phase logic."""

    def _make_review_agent(self) -> MetaAgent:
        agent = MetaAgent()
        agent._started = True
        agent._phase = MetaPhases.REVIEW
        agent._artifact_type = ArtifactType.FSM

        # Set up a builder with some content
        from fsm_llm_meta.builders import FSMBuilder

        builder = FSMBuilder()
        builder.set_overview("Bot", "A bot")
        builder.add_state("s1", "State 1", "Purpose")
        agent._builder = builder
        return agent

    def test_approve_completes(self):
        agent = self._make_review_agent()
        response = agent.send("yes")
        assert agent.is_complete()
        assert agent._result is not None

    def test_approve_variants(self):
        for word in ["ok", "looks good", "lgtm", "approve", "ship it"]:
            agent = self._make_review_agent()
            agent.send(word)
            assert agent.is_complete(), f"'{word}' should approve"

    def test_revise_stays_in_review(self):
        agent = self._make_review_agent()
        with patch.object(agent, "_do_revision", return_value="Updated"):
            response = agent.send("add another state")
        assert agent._phase == MetaPhases.REVIEW  # _do_revision sets it

    def test_get_result_after_approve(self):
        agent = self._make_review_agent()
        agent.send("approve")
        result = agent.get_result()
        assert isinstance(result, MetaAgentResult)
        assert result.artifact_type == ArtifactType.FSM
        assert "Bot" in result.artifact_json


class TestMetaAgentBuild:
    """Test build phase via ReactAgent."""

    def test_do_build_creates_builder(self):
        agent = MetaAgent()
        agent._started = True
        agent._artifact_type = ArtifactType.FSM
        agent._requirements = {
            "artifact_name": "Bot",
            "artifact_description": "A bot",
        }
        agent._conversation_history = [
            {"role": "user", "content": "Build a bot"}
        ]

        with patch("fsm_llm_meta.agent.ReactAgent") as MockReact:
            mock_instance = MockReact.return_value
            mock_instance.run.return_value = MagicMock(answer="done")
            response = agent._do_build()

        assert agent._builder is not None
        assert agent._phase == MetaPhases.REVIEW
        MockReact.return_value.run.assert_called_once()

    def test_do_build_failure_still_goes_to_review(self):
        agent = MetaAgent()
        agent._started = True
        agent._artifact_type = ArtifactType.FSM
        agent._requirements = {}
        agent._conversation_history = []

        with patch("fsm_llm_meta.agent.ReactAgent") as MockReact:
            MockReact.return_value.run.side_effect = Exception("Build failed")
            response = agent._do_build()

        assert agent._phase == MetaPhases.REVIEW  # Still goes to review


class TestMetaAgentInternalState:
    def test_internal_state_initial(self):
        agent = MetaAgent()
        state = agent.get_internal_state()
        assert state["phase"] == MetaPhases.INTAKE
        assert state["turn_count"] == 0
        assert state["builder_summary"] is None

    def test_internal_state_with_builder(self):
        from fsm_llm_meta.builders import FSMBuilder

        agent = MetaAgent()
        agent._artifact_type = ArtifactType.FSM
        builder = FSMBuilder()
        builder.set_overview("Bot", "Desc")
        agent._builder = builder

        state = agent.get_internal_state()
        assert state["artifact_type"] == "fsm"
        assert state["builder_summary"] is not None


class TestMetaAgentOutput:
    def test_output_module_imports(self):
        from fsm_llm_meta.output import (
            format_artifact_json,
            format_summary,
            save_artifact,
        )

        assert callable(format_artifact_json)
        assert callable(format_summary)
        assert callable(save_artifact)

    def test_format_artifact_json(self):
        from fsm_llm_meta.output import format_artifact_json

        result = format_artifact_json({"name": "test", "states": {}})
        assert '"name": "test"' in result

    def test_format_summary(self):
        from fsm_llm_meta.output import format_summary

        result = MetaAgentResult(
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
        from fsm_llm_meta.output import save_artifact

        artifact = {"name": "test", "states": {}}
        path = save_artifact(artifact, tmp_path / "test.json")
        assert path.exists()
        content = path.read_text()
        assert '"name": "test"' in content


class TestMetaAgentImports:
    """Verify the public API is importable."""

    def test_main_imports(self):
        import fsm_llm_meta

        assert hasattr(fsm_llm_meta, "MetaAgent")
        assert hasattr(fsm_llm_meta, "FSMBuilder")
        assert hasattr(fsm_llm_meta, "WorkflowBuilder")
        assert hasattr(fsm_llm_meta, "AgentBuilder")
        assert hasattr(fsm_llm_meta, "ArtifactType")
        assert hasattr(fsm_llm_meta, "MetaAgentConfig")
        assert hasattr(fsm_llm_meta, "MetaAgentResult")
        assert hasattr(fsm_llm_meta, "MetaAgentError")
        assert hasattr(fsm_llm_meta, "create_builder_tools")
        assert hasattr(fsm_llm_meta, "create_fsm_tools")

    def test_version(self):
        from fsm_llm_meta import __version__

        assert isinstance(__version__, str)


class TestTypeAliasResolution:
    """Test that artifact type aliases are correctly resolved."""

    def test_fsm_aliases(self):
        agent = MetaAgent()
        for alias in ["fsm", "chatbot", "conversation", "state machine", "bot"]:
            result = agent._resolve_artifact_type(alias)
            assert result == ArtifactType.FSM, f"'{alias}' should resolve to FSM"

    def test_workflow_aliases(self):
        agent = MetaAgent()
        for alias in ["workflow", "pipeline", "process", "automation"]:
            result = agent._resolve_artifact_type(alias)
            assert result == ArtifactType.WORKFLOW, f"'{alias}' should resolve to WORKFLOW"

    def test_agent_aliases(self):
        agent = MetaAgent()
        for alias in ["agent", "tools", "react", "agentic"]:
            result = agent._resolve_artifact_type(alias)
            assert result == ArtifactType.AGENT, f"'{alias}' should resolve to AGENT"

    def test_unknown_returns_none(self):
        agent = MetaAgent()
        assert agent._resolve_artifact_type("nonsense") is None

    def test_none_returns_none(self):
        agent = MetaAgent()
        assert agent._resolve_artifact_type(None) is None
