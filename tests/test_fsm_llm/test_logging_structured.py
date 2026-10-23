"""Tests for structured logging: setup_logging(), JSON output, env vars, backward compat."""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from loguru import logger

from fsm_llm.constants import (
    ENV_LOG_FORMAT,
    ENV_LOG_LEVEL,
    LOG_DEFAULT_CONVERSATION_ID,
    LOG_FORMAT_HUMAN,
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
        record = {"extra": {"package": "fsm_llm.agents"}}
        result = prepare_log_record(record)
        assert result["extra"]["package"] == "fsm_llm.agents"


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
            package="fsm_llm.agents",
            agent_type="react",
        )
        log.info("agent running")

        output = buf.getvalue().strip()
        entry = json.loads(output)
        assert entry["conversation_id"] == "conv-abc"
        assert entry["package"] == "fsm_llm.agents"
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


class TestEnableDebugLoggingHandlerDedup:
    """``enable_debug_logging()`` (``__init__.py``) must register its stderr
    handler in ``setup_logging()``'s own dedup dict (``_stream_handler_ids``,
    ``logging.py``) -- otherwise a LATER ``setup_logging(sink="stderr",
    format="human")`` call cannot see the handler already exists and adds a
    SECOND stderr handler, duplicating every subsequent log line
    (item 5 / D-013)."""

    def test_setup_logging_after_enable_debug_logging_does_not_duplicate(self):
        """RED before the fix: `second_id` was a real new handler id (> 0),
        not -1, and _library_handler_ids grew by one."""
        from fsm_llm import enable_debug_logging

        enable_debug_logging()
        before_count = len(_library_handler_ids)

        second_id = setup_logging(sink=LOG_SINK_STDERR, format=LOG_FORMAT_HUMAN)

        assert second_id == -1, (
            "setup_logging() added a SECOND stderr handler after "
            "enable_debug_logging() instead of recognizing the existing one "
            f"(got handler_id={second_id})"
        )
        assert len(_library_handler_ids) == before_count

    def test_single_log_call_produces_one_line_not_two(self, capsys):
        """The end-user-visible symptom: one logger.warning() call must
        produce exactly one printed line, not two."""
        from fsm_llm import enable_debug_logging

        enable_debug_logging()
        setup_logging(sink=LOG_SINK_STDERR, format=LOG_FORMAT_HUMAN)

        logger.warning("only once, please")

        captured = capsys.readouterr()
        occurrences = captured.err.count("only once, please")
        assert occurrences == 1, (
            f"expected exactly 1 log line, got {occurrences} in:\n{captured.err}"
        )


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


# ``setup_logging()`` and ``enable_debug_logging()`` own the process's output
# for the library's records: loguru's default stderr handler (id 0) would print
# every ``fsm_llm`` record a second time. The helper replaces it with an
# equivalent sink that skips library records and resolves ``sys.stderr`` at
# write time, so the host's own records keep printing exactly once. Capturing
# needs a child process (process-global handlers), and the child must see the
# tree under test, so its ``sys.path`` comes from ``fsm_llm.__file__``
# (``python -I`` would import the editable install, not a copy of the tree).
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_CHILD = """
import sys
sys.path.insert(0, {root!r})
import fsm_llm
from loguru import logger
{body}
def named(name):
    def patch(record):
        record["name"] = name
    return logger.patch(patch)
named("fsm_llm.x").warning("LIBREC")
named("fsm_llm_contrib.y").warning("LOOKREC")
named("__main__").warning("HOSTREC")
named("__main__").info("QUIETREC")
"""


def _child(body: str, **env: str) -> list[str]:
    """Run ``body`` in a fresh interpreter; return its stderr lines, no ANSI."""
    root = str(Path(sys.modules["fsm_llm"].__file__).parent.parent)
    clean = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("LOGURU_", "FSM_LLM_LOG"))
    }
    result = subprocess.run(
        [sys.executable, "-c", _CHILD.format(root=root, body=body)],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
        env={**clean, **env},
    )
    return _ANSI.sub("", result.stderr).splitlines()


def _count(lines: list[str], marker: str, separator: str) -> int:
    return sum(line.endswith(f" {separator} {marker}") for line in lines)


class TestDefaultHandlerIsReplacedNotDoubled:
    """I-127: a library record printed twice after ``setup_logging()``: once by
    loguru's default handler (format ``... - msg``), once by the library's own
    (``... | msg``). I-149: the library's own stderr handler printed host
    records too, so a host record also printed twice; it now passes library
    records only (JSON and file sinks keep host records)."""

    @pytest.mark.parametrize(
        "body", ["fsm_llm.setup_logging()", "fsm_llm.enable_debug_logging()"]
    )
    def test_each_record_prints_once_and_the_host_default_output_is_kept(self, body):
        lines = _child(body)

        assert _count(lines, "LIBREC", "|") == 1
        assert _count(lines, "LIBREC", "-") == 0
        # The host record: by its own default sink only, never by the library's.
        assert _count(lines, "HOSTREC", "-") == 1
        assert _count(lines, "HOSTREC", "|") == 0
        # A lookalike package name is the host's, not ours.
        assert _count(lines, "LOOKREC", "-") == 1
        assert _count(lines, "LOOKREC", "|") == 0

    def test_a_json_sink_keeps_host_records(self):
        lines = _child('fsm_llm.setup_logging(sink="stderr", format="json")')

        assert _count(lines, "HOSTREC", "-") == 1
        assert sum('"message": "HOSTREC"' in line for line in lines) == 1
        assert sum('"message": "LIBREC"' in line for line in lines) == 1

    def test_the_file_sink_keeps_host_records_and_library_records_leave_stderr(
        self, tmp_path
    ):
        """After ``sink="file"`` a library record goes to the file only."""
        lines = _child(f"fsm_llm.setup_logging(sink='file', log_dir={str(tmp_path)!r})")

        (log_file,) = tmp_path.glob("fsm-llm_*.log")
        text = log_file.read_text()
        assert "LIBREC" in text
        assert "HOSTREC" in text
        assert _count(lines, "LIBREC", "-") == 0
        assert _count(lines, "LIBREC", "|") == 0
        assert _count(lines, "HOSTREC", "-") == 1

    def test_host_records_are_not_lost_when_the_host_sink_is_gone(self):
        """A host that dropped every sink gets them from the library's handler."""
        lines = _child("logger.remove()\nfsm_llm.setup_logging()")

        assert _count(lines, "HOSTREC", "|") == 1
        assert _count(lines, "HOSTREC", "-") == 0

    @pytest.mark.parametrize(
        "call",
        [
            'setup_logging(level="NOPE")',
            "setup_logging(sink='file', log_dir=sys.executable)",
        ],
    )
    def test_a_call_that_raises_leaves_the_default_handler_alone(self, call):
        lines = _child(
            "from fsm_llm.logging import setup_logging\n"
            f"try:\n    {call}\nexcept Exception:\n    pass\n"
            "try:\n    logger.remove(0)\n    print('ID0-THERE', file=sys.stderr)\n"
            "except ValueError:\n    print('ID0-GONE', file=sys.stderr)"
        )

        assert "ID0-THERE" in lines
        assert "ID0-GONE" not in lines

    def test_the_other_sinks_replace_it_too(self):
        lines = _child('fsm_llm.setup_logging(sink="stdout", format="json")')

        assert _count(lines, "LIBREC", "-") == 0
        assert _count(lines, "HOSTREC", "-") == 1

    def test_a_second_call_is_a_no_op(self):
        lines = _child(
            "fsm_llm.setup_logging()\n"
            "from fsm_llm.logging import replace_default_handler\n"
            "replace_default_handler()\nreplace_default_handler()"
        )

        assert _count(lines, "HOSTREC", "-") == 1

    def test_a_missing_default_handler_changes_nothing(self):
        """A host that already removed id 0 gets no sink added and no error."""
        lines = _child(
            "logger.remove()\n"
            "from fsm_llm.logging import replace_default_handler\n"
            "replace_default_handler()"
        )

        assert lines == []

    def test_the_hosts_own_first_sink_is_not_taken_for_the_default(self):
        """With ``LOGURU_AUTOINIT=false`` the host's first sink is id 0."""
        lines = _child(
            'logger.add(sys.stderr, format="HOST>{message}")\nfsm_llm.setup_logging()',
            LOGURU_AUTOINIT="false",
        )

        assert "HOST>HOSTREC" in lines

    def test_loguru_level_and_format_are_honoured(self):
        lines = _child(
            "fsm_llm.setup_logging()",
            LOGURU_LEVEL="WARNING",
            LOGURU_FORMAT="HOST>{message}",
        )

        assert lines.count("HOST>HOSTREC") == 1
        assert "HOST>QUIETREC" not in lines
        assert "HOST>LIBREC" not in lines


class TestHostOutputFollowsSysStderr:
    """W4: the replacement default sink looks ``sys.stderr`` up at write time.
    Bound at setup time it would keep a capture stream for good (and fail once
    that stream is closed), where loguru's own id 0 wrote to the real one."""

    def test_a_closed_stream_at_setup_does_not_break_host_output(self):
        lines = _child(
            "import io\n"
            "sys.stderr = io.StringIO()\n"
            'fsm_llm.setup_logging(sink="stdout", format="json")\n'
            "sys.stderr.close()\n"
            "sys.stderr = sys.__stderr__"
        )

        assert not any("Logging error" in line for line in lines)
        assert _count(lines, "HOSTREC", "-") == 1
        assert _count(lines, "LIBREC", "-") == 0

    def test_redirect_stderr_is_followed_for_its_duration_only(self):
        lines = _child(
            "import contextlib, io\n"
            "buf = io.StringIO()\n"
            "with contextlib.redirect_stderr(buf):\n"
            '    fsm_llm.setup_logging(sink="stdout", format="json")\n'
            '    logger.warning("INBUF")\n'
            'logger.warning("AFTER")\n'
            "sys.stderr.write(''.join('BUF:' + x + '\\n' "
            "for x in buf.getvalue().splitlines()))"
        )

        assert _count([x[4:] for x in lines if x.startswith("BUF:")], "INBUF", "-") == 1
        assert _count(lines, "AFTER", "-") == 1
        assert not any(x.startswith("BUF:") and x.endswith("AFTER") for x in lines)
        assert not any("Logging error" in line for line in lines)
