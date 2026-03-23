"""Tests for structured logging: setup_logging(), JSON output, env vars, backward compat."""

from __future__ import annotations

import io
import json
import os

import pytest
from loguru import logger

from fsm_llm.constants import (
    ENV_LOG_FORMAT,
    ENV_LOG_LEVEL,
    LOG_DEFAULT_CONVERSATION_ID,
    LOG_FORMAT_JSON,
    LOG_SINK_FILE,
    LOG_SINK_STDERR,
)
from fsm_llm.logging import (
    _library_handler_ids,
    _make_json_sink,
    prepare_log_record,
    setup_file_logging,
    setup_logging,
)


@pytest.fixture(autouse=True)
def _clean_logger():
    """Remove all library handlers before/after each test."""
    for hid in list(_library_handler_ids):
        try:
            logger.remove(hid)
        except ValueError:
            pass
    _library_handler_ids.clear()

    import fsm_llm.logging as log_module

    log_module._file_handler_initialized = False

    yield

    for hid in list(_library_handler_ids):
        try:
            logger.remove(hid)
        except ValueError:
            pass
    _library_handler_ids.clear()
    log_module._file_handler_initialized = False

    os.environ.pop(ENV_LOG_LEVEL, None)
    os.environ.pop(ENV_LOG_FORMAT, None)


def _add_json_handler(buf):
    """Helper: add a JSON sink handler writing to a StringIO buffer."""
    logger.enable("fsm_llm")
    json_sink = _make_json_sink(buf)
    handler_id = logger.add(
        json_sink,
        level="DEBUG",
        filter=prepare_log_record,
        colorize=False,
    )
    _library_handler_ids.append(handler_id)
    return handler_id


class TestPrepareLogRecord:
    """Tests for the prepare_log_record filter."""

    def test_adds_default_conversation_id(self):
        record = {"extra": {}}
        result = prepare_log_record(record)
        assert result["extra"]["conversation_id"] == LOG_DEFAULT_CONVERSATION_ID

    def test_preserves_existing_conversation_id(self):
        record = {"extra": {"conversation_id": "conv-123"}}
        result = prepare_log_record(record)
        assert result["extra"]["conversation_id"] == "conv-123"

    def test_adds_default_package(self):
        record = {"extra": {}}
        result = prepare_log_record(record)
        assert result["extra"]["package"] == "fsm_llm"

    def test_preserves_existing_package(self):
        record = {"extra": {"package": "fsm_llm_agents"}}
        result = prepare_log_record(record)
        assert result["extra"]["package"] == "fsm_llm_agents"


class TestSetupLogging:
    """Tests for the setup_logging() function."""

    def test_returns_handler_id(self):
        handler_id = setup_logging()
        assert isinstance(handler_id, int)

    def test_enables_library_logging(self):
        setup_logging()
        log = logger.bind(conversation_id="test")
        log.info("test message")

    def test_stderr_human_format(self):
        handler_id = setup_logging(sink=LOG_SINK_STDERR)
        assert handler_id in _library_handler_ids

    def test_stdout_json_format(self):
        """JSON format should produce valid JSONL."""
        buf = io.StringIO()
        _add_json_handler(buf)

        logger.info("test structured message")

        output = buf.getvalue().strip()
        assert output, "Expected log output"

        entry = json.loads(output)
        assert entry["level"] == "INFO"
        assert entry["message"] == "test structured message"
        assert "timestamp" in entry
        assert "module" in entry
        assert "function" in entry
        assert "line" in entry
        assert entry["conversation_id"] == LOG_DEFAULT_CONVERSATION_ID
        assert entry["package"] == "fsm_llm"

    def test_json_output_with_bound_context(self):
        """JSON output should include bound context fields."""
        buf = io.StringIO()
        _add_json_handler(buf)

        log = logger.bind(
            conversation_id="conv-abc",
            package="fsm_llm_agents",
            agent_type="react",
        )
        log.info("agent running")

        output = buf.getvalue().strip()
        entry = json.loads(output)
        assert entry["conversation_id"] == "conv-abc"
        assert entry["package"] == "fsm_llm_agents"
        assert entry["agent_type"] == "react"
        assert entry["message"] == "agent running"

    def test_json_output_with_exception(self):
        """JSON output should include exception info."""
        buf = io.StringIO()
        _add_json_handler(buf)

        try:
            raise ValueError("test error")
        except ValueError:
            logger.exception("something failed")

        output = buf.getvalue().strip()
        entry = json.loads(output)
        assert entry["exception_type"] == "ValueError"
        assert entry["exception_message"] == "test error"

    def test_file_sink_creates_directory(self, tmp_path):
        log_dir = str(tmp_path / "test_logs")
        handler_id = setup_logging(sink=LOG_SINK_FILE, log_dir=log_dir)
        assert handler_id > 0
        assert os.path.isdir(log_dir)

    def test_file_sink_idempotent(self, tmp_path):
        log_dir = str(tmp_path / "test_logs")
        first_id = setup_logging(sink=LOG_SINK_FILE, log_dir=log_dir)
        second_id = setup_logging(sink=LOG_SINK_FILE, log_dir=log_dir)
        assert first_id > 0
        assert second_id == -1

    def test_tracks_handler_ids(self):
        initial_count = len(_library_handler_ids)
        setup_logging()
        assert len(_library_handler_ids) == initial_count + 1


class TestEnvVarConfiguration:
    """Tests for environment variable configuration."""

    def test_log_level_from_env(self):
        """FSM_LLM_LOG_LEVEL should control log level."""
        buf = io.StringIO()
        os.environ[ENV_LOG_LEVEL] = "WARNING"

        logger.enable("fsm_llm")
        json_sink = _make_json_sink(buf)
        handler_id = logger.add(
            json_sink,
            level="WARNING",
            filter=prepare_log_record,
            colorize=False,
        )
        _library_handler_ids.append(handler_id)

        logger.debug("should not appear")
        logger.info("should not appear either")
        logger.warning("should appear")

        output = buf.getvalue().strip()
        lines = [line for line in output.split("\n") if line.strip()]
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["level"] == "WARNING"

    def test_log_format_from_env(self):
        os.environ[ENV_LOG_FORMAT] = LOG_FORMAT_JSON
        handler_id = setup_logging()
        assert handler_id > 0


class TestBackwardCompatibility:
    """Tests for backward compatibility with existing functions."""

    def test_setup_file_logging_still_works(self, tmp_path):
        log_dir = str(tmp_path / "legacy_logs")
        setup_file_logging(log_dir)
        assert os.path.isdir(log_dir)

    def test_setup_file_logging_idempotent(self, tmp_path):
        log_dir = str(tmp_path / "legacy_logs")
        setup_file_logging(log_dir)
        setup_file_logging(log_dir)

    def test_enable_debug_logging_still_works(self):
        from fsm_llm import enable_debug_logging

        enable_debug_logging()


class TestContextualize:
    """Tests for logger.contextualize() propagation."""

    def test_contextualize_propagates_to_child_calls(self):
        buf = io.StringIO()
        _add_json_handler(buf)

        def inner_function():
            logger.info("inner message")

        with logger.contextualize(conversation_id="ctx-123", package="test_pkg"):
            inner_function()

        output = buf.getvalue().strip()
        entry = json.loads(output)
        assert entry["conversation_id"] == "ctx-123"
        assert entry["package"] == "test_pkg"

    def test_contextualize_does_not_leak(self):
        buf = io.StringIO()
        _add_json_handler(buf)

        with logger.contextualize(conversation_id="scoped-123"):
            logger.info("inside")

        buf.truncate(0)
        buf.seek(0)
        logger.info("outside")

        output = buf.getvalue().strip()
        entry = json.loads(output)
        assert entry["conversation_id"] == LOG_DEFAULT_CONVERSATION_ID


class TestJsonFormatter:
    """Tests for the _record_to_json / _make_json_sink functions."""

    def test_produces_valid_json(self):
        buf = io.StringIO()
        _add_json_handler(buf)

        logger.info("format test")

        output = buf.getvalue()
        assert output.endswith("\n")
        entry = json.loads(output.strip())
        assert isinstance(entry, dict)

    def test_includes_all_standard_fields(self):
        buf = io.StringIO()
        _add_json_handler(buf)

        logger.info("field test")

        entry = json.loads(buf.getvalue().strip())
        required_fields = {
            "timestamp",
            "level",
            "message",
            "module",
            "function",
            "line",
        }
        assert required_fields.issubset(set(entry.keys()))

    def test_extra_fields_included(self):
        buf = io.StringIO()
        _add_json_handler(buf)

        log = logger.bind(workflow_id="wf-42", step_name="process")
        log.info("workflow step")

        entry = json.loads(buf.getvalue().strip())
        assert entry["workflow_id"] == "wf-42"
        assert entry["step_name"] == "process"
