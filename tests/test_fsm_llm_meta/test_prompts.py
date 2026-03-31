from __future__ import annotations

"""Tests for meta-agent prompt builders (hybrid architecture)."""

from fsm_llm_agents.definitions import ArtifactType
from fsm_llm_agents.meta_builders import FSMBuilder
from fsm_llm_agents.meta_prompts import (
    build_followup_message,
    build_output_message,
    build_review_presentation,
    build_welcome_message,
)


class TestWelcomeMessage:
    def test_mentions_four_types(self):
        msg = build_welcome_message()
        assert "FSM" in msg
        assert "Workflow" in msg
        assert "Agent" in msg
        assert "Monitor" in msg

    def test_not_empty(self):
        assert len(build_welcome_message()) > 20


class TestFollowupMessage:
    def test_no_type(self):
        msg = build_followup_message(
            artifact_type=None, has_name=False, has_description=False
        )
        assert "FSM" in msg or "build" in msg.lower()

    def test_type_but_no_name(self):
        msg = build_followup_message(
            artifact_type=ArtifactType.FSM, has_name=False, has_description=True
        )
        assert "name" in msg.lower()

    def test_type_and_name_but_no_desc(self):
        msg = build_followup_message(
            artifact_type=ArtifactType.WORKFLOW, has_name=True, has_description=False
        )
        assert "description" in msg.lower()


class TestReviewPresentation:
    def test_valid_artifact(self):
        builder = FSMBuilder()
        builder.set_overview("Bot", "A bot")
        builder.add_state("s1", "State 1", "Purpose")

        msg = build_review_presentation(builder, ArtifactType.FSM)
        assert "FSM" in msg
        assert "Bot" in msg
        assert "approve" in msg.lower() or "changes" in msg.lower()

    def test_invalid_artifact_shows_errors(self):
        builder = FSMBuilder()
        msg = build_review_presentation(builder, ArtifactType.FSM)
        assert "error" in msg.lower()


class TestOutputMessage:
    def test_includes_json(self):
        msg = build_output_message('{"name": "Bot"}')
        assert '"name": "Bot"' in msg
        assert "fsm-llm" in msg
