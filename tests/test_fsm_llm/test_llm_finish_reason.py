"""A reply cut by ``max_tokens`` is detected (plan 0068b055, step 12.23, D-024).

The provider's ``finish_reason`` is read once: it lands on
``CompletionResponse``, a WARNING names the cut, the structured parse paths
name it in their errors, and a half JSON envelope is never shipped as the
reply. A cut PROSE reply and every ``stop`` reply behave as before. Provider
replies are real litellm ``ModelResponse`` objects; the provider is
``fsm_llm.llm.completion``, patched.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from litellm.types.utils import Choices, Message, ModelResponse

from fsm_llm.classification import Classifier
from fsm_llm.definitions import (
    BulkExtractionRequest,
    ClassificationResponseError,
    CompletionRequest,
    CompletionResponse,
    FieldExtractionRequest,
    LLMResponseError,
    ResponseGenerationRequest,
)
from fsm_llm.llm import LiteLLMInterface, _finish_reason, cut_off_note
from fsm_llm.logging import logger

_CUT_ENVELOPE = '{ "message": "CONSOLIDATED LEGAL RISK ASSESSMENT. Recommend'
_CUT_PROSE = "The assessment covers three areas. First, the contract terms, which"
_APOLOGY = "I'm sorry, I couldn't generate a proper response. Please try again."


def _reply(content: str | None, finish: str | None) -> ModelResponse:
    return ModelResponse(
        choices=[Choices(message=Message(content=content), finish_reason=finish)]
    )


class _Provider:
    """Patch the one provider binding; return the replies in order."""

    def __init__(self, *replies: Any) -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    def _completion(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.replies[min(len(self.calls), len(self.replies)) - 1]

    def __enter__(self) -> _Provider:
        self._patches = [
            patch("fsm_llm.llm.completion", side_effect=self._completion),
            patch("fsm_llm.llm.get_supported_openai_params", return_value=[]),
        ]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc: object) -> None:
        for p in reversed(self._patches):
            p.stop()


def _logged(run: Any) -> tuple[Any, list[str]]:
    """Run ``run()`` with the library logger on; return (result, WARNING texts)."""
    records: list[str] = []
    sink = logger.add(lambda m: records.append(m.record["message"]), level="WARNING")
    logger.enable("fsm_llm")
    try:
        result = run()
    finally:
        logger.disable("fsm_llm")
        logger.remove(sink)
    return result, records


def _generate(content: str, finish: str | None, **kw: Any) -> tuple[str, int]:
    llm = LiteLLMInterface("gpt-4o", max_tokens=1000, **kw)
    with _Provider(_reply(content, finish)) as provider:
        out = llm.generate_response(
            ResponseGenerationRequest(system_prompt="s", user_message="u")
        )
    return out.message, len(provider.calls)


def _bulk(content: str, finish: str | None) -> Any:
    llm = LiteLLMInterface("gpt-4o", max_tokens=1000)
    with _Provider(_reply(content, finish)):
        return llm.extract_bulk_data(
            BulkExtractionRequest(system_prompt="s", user_message="u")
        )


def _field(content: str, finish: str | None, field_type: str = "list") -> Any:
    llm = LiteLLMInterface("gpt-4o", max_tokens=1000)
    with _Provider(_reply(content, finish)):
        return llm.extract_field(
            FieldExtractionRequest(
                system_prompt="s",
                user_message="u",
                field_name="plan",
                field_type=field_type,
            )
        )


class TestFinishReasonLandsOnCompletionResponse:
    @pytest.mark.parametrize("finish", ["length", "stop", "tool_calls"])
    def test_final_reply_carries_the_provider_reason(self, finish):
        llm = LiteLLMInterface("gpt-4o")
        with _Provider(_reply("hello", finish)):
            out = llm.complete(
                CompletionRequest(messages=[{"role": "user", "content": "hi"}])
            )
        assert out.kind == "final"
        assert out.finish_reason == finish
        assert out.truncated is (finish == "length")

    def test_tool_call_turn_carries_it(self):
        message = Message(
            content=None,
            tool_calls=[
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "add", "arguments": '{"a": 1}'},
                }
            ],
        )
        reply = ModelResponse(
            choices=[Choices(message=message, finish_reason="tool_calls")]
        )
        llm = LiteLLMInterface("gpt-4o")
        with _Provider(reply):
            out = llm.complete(
                CompletionRequest(messages=[{"role": "user", "content": "hi"}])
            )
        assert out.kind == "calls"
        assert out.finish_reason == "tool_calls"
        assert not out.truncated

    def test_a_cut_malformed_turn_keeps_the_reason(self):
        message = Message(
            content=None,
            tool_calls=[
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "add", "arguments": '{"a": '},
                }
            ],
        )
        reply = ModelResponse(
            choices=[Choices(message=message, finish_reason="length")]
        )
        llm = LiteLLMInterface("gpt-4o")
        with _Provider(reply):
            out = llm.complete(
                CompletionRequest(messages=[{"role": "user", "content": "hi"}])
            )
        assert out.kind == "malformed"
        assert out.truncated

    def test_no_reason_is_none(self):
        assert CompletionResponse(kind="final", text="x").finish_reason is None
        assert not CompletionResponse(kind="final", text="x").truncated

    def test_reader_accepts_only_a_string(self):
        assert _finish_reason(_reply("x", "length")) == "length"
        assert _finish_reason(MagicMock()) is None
        assert _finish_reason(None) is None
        assert _finish_reason(object()) is None
        assert _finish_reason(SimpleNamespace(choices=[SimpleNamespace()])) is None
        assert _finish_reason(SimpleNamespace(choices=[])) is None


class TestWarningNamesTheCut:
    def test_length_warns_with_model_and_max_tokens(self):
        llm = LiteLLMInterface("gpt-4o", max_tokens=321)
        _, records = _logged(
            lambda: _run_complete(llm, _reply("part of an answ", "length"))
        )
        joined = " | ".join(records)
        assert "cut off" in joined
        assert "gpt-4o" in joined
        assert "max_tokens=321" in joined
        assert "LLM_MAX_TOKENS" in joined

    def test_one_warning_per_call(self):
        llm = LiteLLMInterface("gpt-4o", max_tokens=321)
        _, records = _logged(lambda: _run_complete(llm, _reply("part", "length")))
        assert sum("cut off" in r for r in records) == 1

    @pytest.mark.parametrize("finish", ["stop", None])
    def test_a_whole_reply_does_not_warn(self, finish):
        llm = LiteLLMInterface("gpt-4o", max_tokens=321)
        _, records = _logged(lambda: _run_complete(llm, _reply("whole", finish)))
        assert not any("cut off" in r for r in records)

    def test_every_non_stream_path_warns(self):
        _, records = _logged(lambda: _generate(_CUT_PROSE, "length"))
        assert any("cut off" in r for r in records)
        _, records = _logged(lambda: _field("[1, 2", "length"))
        assert any("cut off" in r for r in records)


def _run_complete(llm: LiteLLMInterface, reply: Any) -> CompletionResponse:
    with _Provider(reply):
        return llm.complete(
            CompletionRequest(messages=[{"role": "user", "content": "hi"}])
        )


class TestStructuredParseNamesTheCut:
    def test_bulk_extraction_cut_says_max_tokens(self):
        with pytest.raises(LLMResponseError) as exc:
            _bulk('{"extracted_data": {"name": "Ad', "length")
        text = str(exc.value)
        assert "cut off at max_tokens=1000" in text
        assert "finish_reason=length" in text
        assert text.startswith("Bulk data extraction failed")

    def test_bulk_extraction_invalid_json_with_stop_keeps_the_old_message(self):
        with pytest.raises(LLMResponseError) as exc:
            _bulk('{"extracted_data": {"name": "Ad', "stop")
        text = str(exc.value)
        assert "cut off" not in text
        assert text.startswith("Bulk data extraction failed")

    def test_bulk_extraction_whole_json_with_length_still_parses(self):
        out = _bulk('{"extracted_data": {"a": 1}}', "length")
        assert out.extracted_data == {"a": 1}

    def test_field_extraction_cut_list_says_max_tokens(self):
        out = _field('[{"tool_name": "search", "tool_input": "x"}, {"tool', "length")
        assert out.is_valid is False
        assert "cut off at max_tokens=1000" in (out.validation_error or "")

    def test_field_extraction_unparseable_with_stop_keeps_the_old_text(self):
        out = _field('[{"tool_name": "search"}, {"tool', "stop")
        assert out.is_valid is False
        assert out.validation_error == "Extraction produced no usable value"

    def test_field_extraction_whole_value_with_length_is_untouched(self):
        out = _field('{"value": [1, 2], "confidence": 0.9}', "length")
        assert out.is_valid is True
        assert out.value == [1, 2]

    def test_classification_cut_says_max_tokens(self):
        cut = CompletionResponse(
            kind="final", text='{"intent": "bu', finish_reason="length"
        )
        with pytest.raises(ClassificationResponseError) as exc:
            Classifier._extract_response(cut, max_tokens=256)
        assert "cut off at max_tokens=256" in str(exc.value)

    def test_classification_with_stop_keeps_the_old_message(self):
        whole = CompletionResponse(kind="final", text="no json", finish_reason="stop")
        with pytest.raises(ClassificationResponseError) as exc:
            Classifier._extract_response(whole, max_tokens=256)
        assert str(exc.value) == "Failed to parse LLM JSON.\nResponse: no json"

    def test_note_names_the_limit_or_just_the_setting(self):
        assert "max_tokens=77" in cut_off_note(77)
        assert "max_tokens=" not in cut_off_note(None)


class TestResponseGenerationNeverShipsAHalfEnvelope:
    def test_cut_envelope_is_replaced_and_not_retried(self):
        message, calls = _generate(_CUT_ENVELOPE, "length")
        assert "cut off" in message
        assert "CONSOLIDATED" not in message
        assert message != _APOLOGY
        assert calls == 1  # D-020 retries only on the generic apology

    def test_cut_array_envelope_is_replaced(self):
        message, _ = _generate('[{"message": "partial', "length")
        assert "cut off" in message

    def test_a_complete_message_inside_a_cut_envelope_is_still_recovered(self):
        """The embedded-JSON rung runs first: a whole ``message`` wins."""
        message, _ = _generate('{"message": "a", "note": {"x": 1', "length")
        assert message == "a"

    def test_cut_prose_is_shipped_as_is(self):
        message, _ = _generate(_CUT_PROSE, "length")
        assert message == _CUT_PROSE

    def test_whole_envelope_with_length_is_unwrapped_as_before(self):
        message, _ = _generate('{"message": "All done."}', "length")
        assert message == "All done."

    @pytest.mark.parametrize("finish", ["stop", None])
    def test_the_same_fragment_without_a_cut_is_shipped_as_text(self, finish):
        message, _ = _generate(_CUT_ENVELOPE, finish)
        assert message == _CUT_ENVELOPE

    def test_brace_prose_that_was_not_cut_is_shipped(self):
        message, _ = _generate("{name}, welcome! Glad to help.", "stop")
        assert message == "{name}, welcome! Glad to help."

    def test_structured_request_cut_is_replaced(self):
        llm = LiteLLMInterface("gpt-4o", max_tokens=1000)
        with _Provider(_reply('{"message": "half', "length")):
            out = llm.generate_response(
                ResponseGenerationRequest(
                    system_prompt="s",
                    user_message="u",
                    response_format={"type": "json_object"},
                )
            )
        assert "cut off" in out.message
