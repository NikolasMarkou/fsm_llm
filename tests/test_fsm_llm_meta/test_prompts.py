from __future__ import annotations

"""Tests for meta-agent prompt builders."""

from fsm_llm_meta.builders import FSMBuilder
from fsm_llm_meta.definitions import ArtifactType
from fsm_llm_meta.prompts import (
    INTAKE_SYSTEM_PROMPT,
    build_followup_message,
    build_intake_user_message,
    build_output_message,
    build_review_presentation,
    build_revision_prompt,
    build_task_prompt,
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


class TestIntakePrompts:
    def test_system_prompt_exists(self):
        assert isinstance(INTAKE_SYSTEM_PROMPT, str)
        assert "JSON" in INTAKE_SYSTEM_PROMPT

    def test_intake_user_message(self):
        history = [
            {"role": "user", "content": "Build me a chatbot"},
            {"role": "assistant", "content": "What kind?"},
            {"role": "user", "content": "Customer support"},
        ]
        msg = build_intake_user_message(history)
        assert "Build me a chatbot" in msg
        assert "Customer support" in msg
        # Should not include assistant messages
        assert "What kind?" not in msg

    def test_intake_user_message_empty(self):
        msg = build_intake_user_message([])
        assert "artifact_type" in msg  # Contains extraction instructions


class TestFollowupMessage:
    def test_no_type(self):
        msg = build_followup_message(artifact_type=None, has_name=False, has_description=False)
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


class TestTaskPrompt:
    def test_basic_fsm_task(self):
        prompt = build_task_prompt(
            artifact_type=ArtifactType.FSM,
            name="Bot",
            description="A test bot",
            persona="Friendly",
            components=["greeting", "farewell"],
        )
        assert "FSM" in prompt
        assert "Bot" in prompt
        assert "greeting" in prompt
        assert "set_overview" in prompt

    def test_workflow_task(self):
        prompt = build_task_prompt(
            artifact_type=ArtifactType.WORKFLOW,
            name="Pipeline",
            description="Process data",
            persona=None,
            components=None,
        )
        assert "WORKFLOW" in prompt
        assert "Pipeline" in prompt
        assert "add_step" in prompt

    def test_agent_task(self):
        prompt = build_task_prompt(
            artifact_type=ArtifactType.AGENT,
            name="SearchAgent",
            description="Search things",
            persona=None,
            components=["search tool"],
        )
        assert "AGENT" in prompt
        assert "search tool" in prompt
        assert "set_agent_type" in prompt

    def test_includes_user_messages(self):
        prompt = build_task_prompt(
            artifact_type=ArtifactType.FSM,
            name="Bot",
            description="Bot",
            persona=None,
            components=None,
            user_messages="Build me something cool",
        )
        assert "Build me something cool" in prompt

    def test_no_ask_instruction(self):
        prompt = build_task_prompt(
            artifact_type=ArtifactType.FSM,
            name="Bot",
            description="Bot",
            persona=None,
            components=None,
        )
        assert "Do NOT ask" in prompt


class TestRevisionPrompt:
    def test_includes_feedback(self):
        prompt = build_revision_prompt(
            revision_request="Add an error state",
            builder_summary="States: greeting, farewell",
        )
        assert "Add an error state" in prompt
        assert "greeting" in prompt
        assert "Do NOT rebuild" in prompt


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
        # No overview, no states = errors
        msg = build_review_presentation(builder, ArtifactType.FSM)
        assert "error" in msg.lower()


class TestOutputMessage:
    def test_includes_json(self):
        msg = build_output_message('{"name": "Bot"}')
        assert '"name": "Bot"' in msg
        assert "fsm-llm" in msg
