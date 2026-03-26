from __future__ import annotations

"""Tests for meta-agent prompt builders."""

from fsm_llm_meta.prompts import (
    build_classify_extraction_instructions,
    build_classify_response_instructions,
    build_output_response_instructions,
    build_welcome_response_instructions,
)


class TestWelcomePrompt:
    def test_returns_string(self):
        result = build_welcome_response_instructions()
        assert isinstance(result, str)

    def test_mentions_three_types(self):
        result = build_welcome_response_instructions()
        assert "FSM" in result
        assert "Workflow" in result
        assert "Agent" in result

    def test_not_empty(self):
        result = build_welcome_response_instructions()
        assert len(result) > 20


class TestClassifyExtractionPrompt:
    def test_returns_string(self):
        result = build_classify_extraction_instructions()
        assert isinstance(result, str)

    def test_mentions_json(self):
        result = build_classify_extraction_instructions()
        assert "JSON" in result

    def test_mentions_valid_types(self):
        result = build_classify_extraction_instructions()
        assert "fsm" in result
        assert "workflow" in result
        assert "agent" in result

    def test_provides_examples(self):
        result = build_classify_extraction_instructions()
        assert "artifact_type" in result


class TestClassifyResponsePrompt:
    def test_returns_string(self):
        result = build_classify_response_instructions()
        assert isinstance(result, str)

    def test_not_empty(self):
        result = build_classify_response_instructions()
        assert len(result) > 10


class TestOutputPrompt:
    def test_returns_string(self):
        result = build_output_response_instructions()
        assert isinstance(result, str)

    def test_mentions_json(self):
        result = build_output_response_instructions()
        assert "JSON" in result

    def test_mentions_fsm_llm(self):
        result = build_output_response_instructions()
        assert "fsm-llm" in result
