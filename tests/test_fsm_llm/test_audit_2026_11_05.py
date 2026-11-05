"""Regression tests for the 2026-11-05 intent-vs-behaviour audit fixes."""

from __future__ import annotations

import json

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
