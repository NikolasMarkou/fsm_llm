"""Regression tests for the 2026-11-05 intent-vs-behaviour audit fixes."""

from __future__ import annotations

import json
from typing import ClassVar

import pytest


@pytest.fixture
def _clean_logging():
    """Snapshot and restore every piece of global logging state touched here."""
    from loguru import logger

    import fsm_llm.logging as log_module

    before_ids = list(log_module._library_handler_ids)
    before_streams = dict(log_module._stream_handler_ids)
    before_flag = log_module._file_handler_initialized
    log_module._file_handler_initialized = False
    added: list[int] = []
    yield added
    for hid in added:
        try:
            logger.remove(hid)
        except ValueError:
            pass
    for hid in list(log_module._library_handler_ids):
        if hid not in before_ids:
            try:
                logger.remove(hid)
            except ValueError:
                pass
    log_module._library_handler_ids[:] = before_ids
    log_module._stream_handler_ids.clear()
    log_module._stream_handler_ids.update(before_streams)
    log_module._file_handler_initialized = before_flag


class TestJsonFileSink:
    """F-HM-02: the JSON file sink writes one valid JSON object per line."""

    def test_markup_like_text_and_exception_stay_valid_jsonl(
        self, tmp_path, capfd, _clean_logging
    ):
        from loguru import logger

        from fsm_llm.logging import setup_logging

        _clean_logging.append(
            setup_logging(
                sink="file", format="json", level="INFO", log_dir=str(tmp_path)
            )
        )

        logger.info("tag <current_context> inside")
        logger.info("backslash \\<x> and <red>not a colour</red>")
        (lambda: logger.info("from a lambda"))()
        try:
            raise RuntimeError("boom <b>")
        except RuntimeError:
            logger.exception("boom")
        logger.complete()

        files = list(tmp_path.iterdir())
        assert files
        lines = [
            line for f in files for line in f.read_text().splitlines() if line.strip()
        ]
        entries = [json.loads(line) for line in lines]
        messages = [e["message"] for e in entries]
        assert "tag <current_context> inside" in messages
        assert "backslash \\<x> and <red>not a colour</red>" in messages
        assert "from a lambda" in messages
        assert "boom" in messages
        lambda_entry = next(e for e in entries if e["message"] == "from a lambda")
        assert lambda_entry["function"] == "<lambda>"
        boom = next(e for e in entries if e["message"] == "boom")
        assert boom["exception_type"] == "RuntimeError"
        assert boom["exception_message"] == "boom <b>"
        assert not any("Traceback" in line for line in lines)
        assert "Logging error" not in capfd.readouterr().err


class TestStrictBoolCoercion:
    """F-LC-01: a bool field accepts a closed yes/no vocabulary, nothing else.

    Drives the real validation seam and the real llm.py unstructured rung, so the
    two readers are pinned to the same answers.
    """

    _TRUE: ClassVar[list[str]] = [
        "Yes.",
        " true",
        "true\n",
        "Y",
        "'true'",
        '"Yes"',
        "1",
        "TRUE!",
    ]
    _FALSE: ClassVar[list[str]] = ["No!", "'false'", "n", "False", " no ", "0", "NO."]
    _UNKNOWN: ClassVar[list[str]] = [
        "maybe",
        "yeah",
        "sure",
        "affirmative",
        "",
        "  ",
        "nope",
        "2",
        ".",
    ]

    @staticmethod
    def _validate(value):
        from fsm_llm.definitions import FieldExtractionConfig, FieldExtractionResponse
        from fsm_llm.pipeline import MessagePipeline

        return MessagePipeline._validate_field_extraction(
            FieldExtractionResponse(
                field_name="agree", value=value, confidence=1.0, is_valid=True
            ),
            FieldExtractionConfig(
                field_name="agree",
                field_type="bool",
                extraction_instructions="did they agree",
                confidence_threshold=0.0,
            ),
        )

    @staticmethod
    def _rung(text):
        from unittest.mock import MagicMock

        from fsm_llm.definitions import FieldExtractionRequest
        from fsm_llm.llm import LiteLLMInterface

        response = MagicMock()
        choice = MagicMock()
        choice.message.content = text
        response.choices = [choice]
        request = FieldExtractionRequest(
            system_prompt="extract",
            user_message="x",
            field_name="agree",
            field_type="bool",
        )
        return LiteLLMInterface(model="test-model")._parse_field_extraction_response(
            response, request
        )

    @pytest.mark.parametrize("text", _TRUE)
    def test_true_words_coerce_to_true(self, text):
        out = self._validate(text)
        assert out.is_valid is True
        assert out.value is True

    @pytest.mark.parametrize("text", _FALSE)
    def test_false_words_coerce_to_false(self, text):
        out = self._validate(text)
        assert out.is_valid is True
        assert out.value is False

    @pytest.mark.parametrize("text", _UNKNOWN)
    def test_unknown_words_are_invalid_not_false(self, text):
        out = self._validate(text)
        assert out.is_valid is False
        assert "bool" in (out.validation_error or "")

    def test_non_string_branches_unchanged(self):
        assert self._validate(True).value is True
        assert self._validate(False).value is False
        assert self._validate(1).value is True
        assert self._validate(0).value is False
        assert self._validate({"a": 1}).is_valid is False
        assert self._validate([True]).is_valid is False

    @pytest.mark.parametrize("text", _TRUE)
    def test_llm_rung_agrees_on_true(self, text):
        if not text.strip():
            pytest.skip("empty")
        out = self._rung(text)
        assert out.is_valid is True and out.value is True

    @pytest.mark.parametrize("text", _FALSE)
    def test_llm_rung_agrees_on_false(self, text):
        out = self._rung(text)
        assert out.is_valid is True and out.value is False

    @pytest.mark.parametrize("text", ["maybe", "yeah", "sure", "affirmative", "nope"])
    def test_llm_rung_unknown_word_is_invalid(self, text):
        out = self._rung(text)
        assert out.is_valid is False
        assert out.value is None
