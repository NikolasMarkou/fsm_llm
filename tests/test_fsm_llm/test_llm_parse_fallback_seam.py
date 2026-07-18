"""Robustness tests for the LLM response-parsing degradation ladder (T4 / D-016).

The documented ladder is: **structured JSON -> embedded-JSON fallback -> raw text**.
The contract is that a ``_parse_*`` method NEVER raises for a well-formed but
oversized / oddly-typed model response — it degrades down the ladder instead.

These tests drive the REAL parse methods
(``LiteLLMInterface._parse_response_generation_response`` and
``._parse_field_extraction_response``); nothing here re-implements the ladder.
The only thing faked is the litellm response envelope
(``response.choices[0].message.content``), which is the actual seam the parsers
read from.
"""

from __future__ import annotations

import json

import pytest

from fsm_llm.definitions import (
    FieldExtractionRequest,
    FieldExtractionResponse,
    ResponseGenerationResponse,
)
from fsm_llm.llm import _RESPONSE_MESSAGE_MAX_LEN, LiteLLMInterface


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    """Minimal stand-in for a litellm completion envelope."""

    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


@pytest.fixture
def llm() -> LiteLLMInterface:
    return LiteLLMInterface(model="test", api_key="test")


@pytest.fixture
def field_request() -> FieldExtractionRequest:
    return FieldExtractionRequest(
        system_prompt="extract the field",
        user_message="some user text",
        field_name="destination",
        field_type="str",
    )


class TestCapDriftGuard:
    """The duplicated 5000 literal cannot silently diverge from the model."""

    def test_constant_matches_the_declared_model_max_length(self):
        metadata = ResponseGenerationResponse.model_fields["message"].metadata
        declared = [
            m.max_length for m in metadata if getattr(m, "max_length", None) is not None
        ]
        assert declared, "ResponseGenerationResponse.message lost its max_length"
        assert _RESPONSE_MESSAGE_MAX_LEN == declared[0], (
            f"llm._RESPONSE_MESSAGE_MAX_LEN ({_RESPONSE_MESSAGE_MAX_LEN}) has drifted "
            f"from ResponseGenerationResponse.message max_length ({declared[0]}); "
            "update both (definitions.py and llm.py)"
        )


class TestResponseGenerationLadderNeverRaises:
    """SC-7: no ValidationError escapes ``_parse_response_generation_response``."""

    def test_oversized_message_via_embedded_json_fallback(self, llm):
        """The reported defect: >5000-char message in the fallback branch."""
        oversized = "A" * 6000
        # Prose prefix => not `_looks_like_json`, so the PRIMARY rung is skipped
        # and the embedded-JSON fallback is the branch under test.
        content = "Here is my answer: " + json.dumps({"message": oversized})

        result = llm._parse_response_generation_response(_FakeResponse(content))

        assert isinstance(result, ResponseGenerationResponse)
        assert len(result.message) <= _RESPONSE_MESSAGE_MAX_LEN
        assert result.message  # usable, not empty

    def test_oversized_raw_text_at_the_terminal_rung(self, llm):
        """The real trap: the bottom rung builds the SAME capped model."""
        result = llm._parse_response_generation_response(_FakeResponse("B" * 12000))

        assert isinstance(result, ResponseGenerationResponse)
        assert len(result.message) == _RESPONSE_MESSAGE_MAX_LEN
        assert result.message == "B" * _RESPONSE_MESSAGE_MAX_LEN

    def test_oversized_message_via_primary_structured_rung(self, llm):
        """Pure JSON >5000: primary rung fails, ladder still lands on its feet."""
        content = json.dumps({"message": "C" * 9000})

        result = llm._parse_response_generation_response(_FakeResponse(content))

        assert isinstance(result, ResponseGenerationResponse)
        assert len(result.message) <= _RESPONSE_MESSAGE_MAX_LEN

    def test_oversized_reasoning_does_not_raise(self, llm):
        """`reasoning` carries the same 5000 cap as `message`."""
        content = "Answer: " + json.dumps({"message": "hi", "reasoning": "R" * 7000})

        result = llm._parse_response_generation_response(_FakeResponse(content))

        assert isinstance(result, ResponseGenerationResponse)

    @pytest.mark.parametrize(
        "content",
        [
            "",
            "   ",
            "```",
            "```json\n```",
            "```json\n{\n```",
            '{"message": "he said \\"hi {nested} there\\" and left"}',
            'prose {"message": "brace } inside \\"a {string}\\" value"} tail',
            "{not json at all",
            "[1, 2, 3]",
            '{"message": null}',
            '{"message": 42}',
            '{"message_type": "response"}',
            "null",
        ],
        ids=[
            "empty",
            "whitespace",
            "bare-fence",
            "empty-json-fence",
            "truncated-fence",
            "escaped-quotes-and-braces",
            "nested-braces-in-string-value",
            "malformed-object",
            "top-level-array",
            "null-message",
            "int-message",
            "message-key-absent",
            "literal-null",
        ],
    )
    def test_degenerate_inputs_return_rather_than_raise(self, llm, content):
        result = llm._parse_response_generation_response(_FakeResponse(content))
        assert isinstance(result, ResponseGenerationResponse)

    def test_non_string_content_types_return_rather_than_raise(self, llm):
        for content in (None, 123, ["a", "b"], {"other": "key"}):
            result = llm._parse_response_generation_response(_FakeResponse(content))
            assert isinstance(result, ResponseGenerationResponse)


class TestFieldExtractionLadderNeverRaises:
    """SC-7: no ValidationError/TypeError escapes ``_parse_field_extraction_response``."""

    def test_oversized_reasoning_via_embedded_json_fallback(self, llm, field_request):
        content = "Result: " + json.dumps(
            {"value": "Paris", "reasoning": "R" * 7000, "confidence": 0.9}
        )

        result = llm._parse_field_extraction_response(
            _FakeResponse(content), field_request
        )

        assert isinstance(result, FieldExtractionResponse)

    def test_non_numeric_confidence_in_fallback_does_not_raise(
        self, llm, field_request
    ):
        """`float({...})` raises TypeError, which is NOT a ValueError."""
        content = "Result: " + json.dumps({"value": "Paris", "confidence": {"a": 1}})

        result = llm._parse_field_extraction_response(
            _FakeResponse(content), field_request
        )

        assert isinstance(result, FieldExtractionResponse)

    def test_non_numeric_confidence_in_primary_rung_does_not_raise(
        self, llm, field_request
    ):
        """Sibling found by the step-4 sweep: the PRIMARY rung escaped too."""
        content = json.dumps({"value": "Paris", "confidence": {"a": 1}})

        result = llm._parse_field_extraction_response(
            _FakeResponse(content), field_request
        )

        assert isinstance(result, FieldExtractionResponse)

    def test_null_confidence_in_primary_rung_does_not_raise(self, llm, field_request):
        content = json.dumps({"value": "Paris", "confidence": None})

        result = llm._parse_field_extraction_response(
            _FakeResponse(content), field_request
        )

        assert isinstance(result, FieldExtractionResponse)

    @pytest.mark.parametrize(
        "content",
        [
            "",
            "   ",
            "```",
            "```json\n```",
            '{"value": "he said \\"hi {nested} there\\""}',
            'prose {"value": "brace } in \\"a {string}\\""} tail',
            "{not json at all",
            "[1, 2, 3]",
            '{"value": null}',
            '{"confidence": "not-a-number"}',
            "null",
        ],
        ids=[
            "empty",
            "whitespace",
            "bare-fence",
            "empty-json-fence",
            "escaped-quotes-and-braces",
            "nested-braces-in-string-value",
            "malformed-object",
            "top-level-array",
            "null-value",
            "non-numeric-confidence-string",
            "literal-null",
        ],
    )
    def test_degenerate_inputs_return_rather_than_raise(
        self, llm, field_request, content
    ):
        result = llm._parse_field_extraction_response(
            _FakeResponse(content), field_request
        )
        assert isinstance(result, FieldExtractionResponse)

    def test_terminal_rung_reports_failure_rather_than_raising(
        self, llm, field_request
    ):
        """The bottom rung is the last line of defence: it returns, flagged."""
        result = llm._parse_field_extraction_response(
            _FakeResponse(None), field_request
        )

        assert isinstance(result, FieldExtractionResponse)
        assert result.is_valid is False
        assert result.value is None
        assert result.validation_error


class TestNormalResponsesUnchanged:
    """Non-regression: short/well-formed responses behave exactly as before."""

    def test_structured_response_still_parsed_from_the_primary_rung(self, llm):
        content = json.dumps(
            {"message": "Hello there", "message_type": "response", "reasoning": "why"}
        )

        result = llm._parse_response_generation_response(_FakeResponse(content))

        assert result.message == "Hello there"
        assert result.message_type == "response"
        assert result.reasoning == "why"

    def test_plain_text_response_still_used_verbatim(self, llm):
        result = llm._parse_response_generation_response(_FakeResponse("Just text."))

        assert result.message == "Just text."
        assert result.message_type == "response"

    def test_embedded_json_fallback_still_wins_over_raw_text(self, llm):
        content = '<think>musing</think> {"message": "Extracted!"}'

        result = llm._parse_response_generation_response(_FakeResponse(content))

        assert result.message == "Extracted!"

    def test_field_extraction_structured_response_unchanged(self, llm, field_request):
        content = json.dumps(
            {"value": "Paris", "confidence": 0.8, "reasoning": "stated explicitly"}
        )

        result = llm._parse_field_extraction_response(
            _FakeResponse(content), field_request
        )

        assert result.value == "Paris"
        assert result.confidence == pytest.approx(0.8)
        assert result.reasoning == "stated explicitly"

    def test_field_extraction_fallback_still_extracts(self, llm, field_request):
        content = "Sure! " + json.dumps({"value": "Paris", "confidence": 0.7})

        result = llm._parse_field_extraction_response(
            _FakeResponse(content), field_request
        )

        assert result.value == "Paris"
