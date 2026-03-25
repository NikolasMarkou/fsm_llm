from __future__ import annotations

"""Tests for MetaAgent."""

import pytest

from fsm_llm_meta.agent import MetaAgent
from fsm_llm_meta.definitions import MetaAgentConfig
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
        assert agent._api is None


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
        """Starting twice should raise."""
        agent = MetaAgent()
        # We can't easily mock the LLM here, so we just test the error path
        # by manually setting _api
        agent._api = object()  # type: ignore[assignment]
        with pytest.raises(MetaAgentError, match="already been started"):
            agent.start()


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
        from fsm_llm_meta.definitions import ArtifactType, MetaAgentResult
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

        # Verify key symbols are exported
        assert hasattr(fsm_llm_meta, "MetaAgent")
        assert hasattr(fsm_llm_meta, "FSMBuilder")
        assert hasattr(fsm_llm_meta, "WorkflowBuilder")
        assert hasattr(fsm_llm_meta, "AgentBuilder")
        assert hasattr(fsm_llm_meta, "ArtifactType")
        assert hasattr(fsm_llm_meta, "MetaAgentConfig")
        assert hasattr(fsm_llm_meta, "MetaAgentResult")
        assert hasattr(fsm_llm_meta, "MetaAgentError")
        assert hasattr(fsm_llm_meta, "BuilderError")

    def test_version(self):
        from fsm_llm_meta import __version__

        assert isinstance(__version__, str)
