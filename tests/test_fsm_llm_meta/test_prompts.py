from __future__ import annotations

"""Tests for meta-agent prompt builders (hybrid architecture)."""

from fsm_llm.agents.definitions import ArtifactType
from fsm_llm.agents.meta_builders import FSMArtifactBuilder
from fsm_llm.agents.meta_prompts import (
    build_review_presentation,
    build_welcome_message,
)


class TestWelcomeMessage:
    def test_mentions_three_types(self):
        msg = build_welcome_message()
        assert "FSM" in msg
        assert "Workflow" in msg
        assert "Agent" in msg

    def test_not_empty(self):
        assert len(build_welcome_message()) > 20


class TestReviewPresentation:
    def test_valid_artifact(self):
        builder = FSMArtifactBuilder()
        builder.set_overview("Bot", "A bot")
        builder.add_state("s1", "State 1", "Purpose")

        msg = build_review_presentation(builder, ArtifactType.FSM)
        assert "FSM" in msg
        assert "Bot" in msg
        assert "approve" in msg.lower() or "changes" in msg.lower()

    def test_invalid_artifact_shows_errors(self):
        builder = FSMArtifactBuilder()
        msg = build_review_presentation(builder, ArtifactType.FSM)
        assert "error" in msg.lower()
