"""Seam regression tests for audit-fix loop 3 (plan-2026-09-19T175721-21cd7f8e).

Every test drives a public entry point (``API.converse`` or ``LiteLLMInterface``
with a patched ``fsm_llm.llm.completion``). Sections are appended one per plan
step; harness helpers are reused from the iteration-1 seam file.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from fsm_llm.definitions import (
    FieldExtractionConfig,
    FieldExtractionRequest,
    ResponseGenerationRequest,
)
from fsm_llm.llm import LiteLLMInterface
from fsm_llm.pipeline import MessagePipeline
from tests.test_fsm_llm.test_audit_iter1_seam import _fake_response

# ══════════════════════════════════════════════════════════════
# Step 2 / RB-01: uncoercible confidence at the field-extraction rungs
# ══════════════════════════════════════════════════════════════


def _extract_field(reply: str, field_type: str = "str"):
    req = FieldExtractionRequest(
        system_prompt="extract the field",
        user_message="My name is Bob.",
        field_name="name",
        field_type=field_type,  # type: ignore[arg-type]
    )
    with (
        patch("fsm_llm.llm.completion") as mock_comp,
        patch(
            "fsm_llm.llm.get_supported_openai_params",
            return_value=["response_format"],
        ),
    ):
        mock_comp.return_value = _fake_response(reply)
        return LiteLLMInterface(model="gpt-4o", api_key="k").extract_field(req)


_BAD_CONFIDENCES = ['"high"', "null", '{"x": 1}', '"95%"', '"0.9,"']


class TestFieldExtractionUncoercibleConfidence:
    @pytest.mark.parametrize("bad", _BAD_CONFIDENCES)
    def test_primary_rung_keeps_the_value_at_half_confidence(self, bad):
        out = _extract_field(
            f'{{"field_name": "name", "value": "Bob", "confidence": {bad}}}'
        )
        assert out.value == "Bob"
        assert out.confidence == 0.5
        assert out.is_valid

    @pytest.mark.parametrize("bad", _BAD_CONFIDENCES)
    def test_embedded_json_rung_keeps_the_value_at_half_confidence(self, bad):
        # A prose prefix skips the primary rung; the embedded-JSON rung runs.
        out = _extract_field(
            f'Sure: {{"field_name": "name", "value": "Bob", "confidence": {bad}}} ok'
        )
        assert out.value == "Bob"
        assert out.confidence == 0.5
        assert out.is_valid

    @pytest.mark.parametrize("bad", _BAD_CONFIDENCES)
    def test_a_thresholded_field_rejects_the_unscored_value(self, bad):
        out = _extract_field(
            f'{{"field_name": "name", "value": "Bob", "confidence": {bad}}}'
        )
        checked = MessagePipeline._validate_field_extraction(
            out,
            FieldExtractionConfig(
                field_name="name",
                field_type="str",
                extraction_instructions="the name",
                confidence_threshold=0.9,
            ),
        )
        assert not checked.is_valid

    def test_a_normal_confidence_is_unchanged(self):
        out = _extract_field(
            '{"field_name": "name", "value": "Bob", "confidence": 0.8}'
        )
        assert (out.value, out.confidence) == ("Bob", 0.8)
        out = _extract_field(
            'Sure: {"field_name": "name", "value": "Bob", "confidence": 0.7} ok'
        )
        assert (out.value, out.confidence) == ("Bob", 0.7)


# ══════════════════════════════════════════════════════════════
# Step 3 / RB-02: a structured reply with a `reasoning` key
# ══════════════════════════════════════════════════════════════

_STRUCTURED_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "reply", "schema": {"type": "object"}},
}


def _generate(content: str, response_format=None):
    req = ResponseGenerationRequest(
        system_prompt="s",
        user_message="u",
        extracted_data={},
        context={},
        transition_occurred=False,
        previous_state=None,
        response_format=response_format,
    )
    with (
        patch("fsm_llm.llm.completion") as mock_comp,
        patch(
            "fsm_llm.llm.get_supported_openai_params",
            return_value=["response_format"],
        ),
    ):
        mock_comp.return_value = _fake_response(content)
        out = LiteLLMInterface(model="gpt-4o", api_key="k").generate_response(req)
    return out.message


class TestStructuredReplyWithReasoningKey:
    def test_answer_plus_reasoning_returns_the_json_text(self):
        body = json.dumps({"answer": "42", "reasoning": "because 6x7"})
        message = _generate(body, _STRUCTURED_FORMAT)
        assert json.loads(message) == {"answer": "42", "reasoning": "because 6x7"}

    def test_answer_only_is_unchanged(self):
        body = json.dumps({"answer": "42"})
        assert json.loads(_generate(body, _STRUCTURED_FORMAT)) == {"answer": "42"}

    def test_schema_with_message_key_still_returns_the_message(self):
        body = json.dumps({"message": "hi there", "reasoning": "polite"})
        assert _generate(body, _STRUCTURED_FORMAT) == "hi there"

    def test_unstructured_reasoning_only_reply_still_returns_the_reasoning(self):
        body = json.dumps({"reasoning": "because x"})
        assert _generate(body, None) == "because x"


# ══════════════════════════════════════════════════════════════
# Step 4 / RB-10: fence-span skip and Strategy 3/4 parity
# ══════════════════════════════════════════════════════════════


class TestFencedExampleDoesNotHideOrLeakAnObject:
    def test_reply_with_an_object_before_a_fenced_example(self):
        # The provider wraps its envelope in prose and then shows a fenced
        # schema example. The object BEFORE the fence must be parsed whole:
        # the regex fallback would cut its escaped-quote message at the `\\`.
        content = (
            'Here you go: {"message": "say \\"hi\\" now"}\n'
            'Schema example:\n```json\n[{"a": 1}]\n```\nHope this helps.'
        )
        assert _generate(content) == 'say "hi" now'

    def test_reply_with_only_a_fenced_array_does_not_leak_its_message(self):
        message = _generate('```json\n[{"message": "leak"}]\n```')
        assert message != "leak"
