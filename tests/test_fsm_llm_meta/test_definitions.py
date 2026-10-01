from __future__ import annotations

"""Tests for meta-agent definitions and models."""

import pytest

from fsm_llm.agents.definitions import (
    ArtifactType,
    BuildProgress,
    MetaBuilderConfig,
    MetaBuilderResult,
)


class TestArtifactType:
    def test_values(self):
        assert ArtifactType.FSM == "fsm"
        assert ArtifactType.WORKFLOW == "workflow"
        assert ArtifactType.AGENT == "agent"

    def test_from_string(self):
        assert ArtifactType("fsm") == ArtifactType.FSM
        assert ArtifactType("workflow") == ArtifactType.WORKFLOW
        assert ArtifactType("agent") == ArtifactType.AGENT

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            ArtifactType("invalid")


class TestBuildProgress:
    def test_percentage_empty(self):
        p = BuildProgress()
        assert p.percentage == 0.0

    def test_percentage_partial(self):
        p = BuildProgress(total_required=4, completed=2, missing=["a", "b"])
        assert p.percentage == 50.0

    def test_percentage_complete(self):
        p = BuildProgress(total_required=4, completed=4, missing=[])
        assert p.percentage == 100.0

    def test_is_complete(self):
        assert not BuildProgress(total_required=0).is_complete
        assert not BuildProgress(
            total_required=4, completed=2, missing=["x"]
        ).is_complete
        assert BuildProgress(total_required=4, completed=4, missing=[]).is_complete


class TestMetaBuilderConfig:
    def test_defaults(self):
        config = MetaBuilderConfig()
        assert config.temperature == 0.7
        assert config.max_tokens == 4096
        assert config.max_turns == 50

    def test_max_iterations_is_the_inherited_unused_field(self):
        """D-012 of plan 944e2692: the override (25) is gone; the inherited
        AgentConfig field stays so callers passing it keep validating."""
        from fsm_llm.agents.definitions import AgentConfig

        assert MetaBuilderConfig().max_iterations == AgentConfig().max_iterations
        assert MetaBuilderConfig(max_iterations=20).max_iterations == 20

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("build_temperature", 0.3),
            ("build_max_iterations", 25),
            ("build_timeout_seconds", 120.0),
        ],
    )
    def test_removed_build_fields_are_rejected(self, field, value):
        """D-012 of plan 944e2692: the three unread build_* fields are removed;
        the config is extra="forbid", so an old caller gets an error naming
        the key (the parent commit accepted them)."""
        from pydantic import ValidationError

        assert field not in MetaBuilderConfig.model_fields
        with pytest.raises(ValidationError, match=field):
            MetaBuilderConfig(**{field: value})

    def test_custom_values(self):
        config = MetaBuilderConfig(model="gpt-4o", temperature=0.3, max_turns=100)
        assert config.model == "gpt-4o"
        assert config.temperature == 0.3
        assert config.max_turns == 100

    def test_invalid_max_turns(self):
        with pytest.raises(ValueError, match="max_turns must be at least 1"):
            MetaBuilderConfig(max_turns=0)


class TestMetaBuilderResult:
    def test_minimal(self):
        result = MetaBuilderResult(artifact_type=ArtifactType.FSM)
        assert result.is_valid
        assert result.artifact == {}
        assert result.artifact_json == ""
        assert result.validation_errors == []
        assert result.conversation_turns == 0

    def test_with_errors(self):
        result = MetaBuilderResult(
            artifact_type=ArtifactType.WORKFLOW,
            is_valid=False,
            validation_errors=["Missing name"],
        )
        assert not result.is_valid
        assert len(result.validation_errors) == 1
