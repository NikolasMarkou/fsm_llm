"""RED-first tests for the step-7 logging fixes (plan 5980fbb7).

F-SU-01  ``setup_logging`` accepts a lower-case or blank level (argument and env)
F-SU-07  JSON log lines label host records correctly; a bound extra cannot
         overwrite the identity fields
F-SU-08  ``setup_file_logging`` returns the handler id; unknown sink or format
         is refused
F-XS-09  the validate / visualize entry points do not print DEBUG records
F-AF-16  ``enable_debug_logging`` lives in ``fsm_llm.logging`` and registers
         through ``setup_logging`` (one format, one dedup key)
"""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from loguru import logger

import fsm_llm.logging as log_module
from fsm_llm.constants import (
    ENV_LOG_FORMAT,
    ENV_LOG_LEVEL,
    LOG_FORMAT_HUMAN,
    LOG_FORMAT_JSON,
    LOG_SINK_FILE,
    LOG_SINK_STDERR,
    LOG_SINK_STDOUT,
)
from fsm_llm.logging import (
    _make_json_sink,
    _record_to_json,
    prepare_log_record,
    reset_handlers,
    setup_file_logging,
    setup_logging,
)


def _plain(text):
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


_REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _clean_handlers(monkeypatch):
    monkeypatch.delenv(ENV_LOG_LEVEL, raising=False)
    monkeypatch.delenv(ENV_LOG_FORMAT, raising=False)
    reset_handlers()
    yield
    reset_handlers()


# --------------------------------------------------------------
# F-SU-01
# --------------------------------------------------------------


class TestLevelIsNormalised:
    def test_lowercase_level_argument(self, capsys):
        setup_logging(sink=LOG_SINK_STDOUT, level="info")
        logger.debug("below-level")
        logger.info("at-level")
        out = capsys.readouterr().out
        assert "at-level" in out and "below-level" not in out

    @pytest.mark.parametrize("raw", ["info", " Info ", "INFO"])
    def test_env_level_any_case(self, monkeypatch, capsys, raw):
        monkeypatch.setenv(ENV_LOG_LEVEL, raw)
        setup_logging(sink=LOG_SINK_STDOUT)
        logger.debug("below-level")
        logger.info("at-level")
        out = capsys.readouterr().out
        assert "at-level" in out and "below-level" not in out

    @pytest.mark.parametrize("raw", ["", "   "])
    def test_blank_env_level_means_unset(self, monkeypatch, capsys, raw):
        monkeypatch.setenv(ENV_LOG_LEVEL, raw)
        setup_logging(sink=LOG_SINK_STDOUT)
        logger.debug("debug-shows")
        assert "debug-shows" in capsys.readouterr().out

    def test_unknown_explicit_level_still_raises(self):
        with pytest.raises(ValueError):
            setup_logging(sink=LOG_SINK_STDOUT, level="loud")

    def test_runner_survives_lowercase_env_level(self, tmp_path, monkeypatch, capsys):
        """The shipped CLI died with a traceback on FSM_LLM_LOG_LEVEL=info."""
        from fsm_llm import runner

        api = MagicMock()
        api.start_conversation.return_value = ("conv-1", "HELLO")
        api.has_conversation_ended.side_effect = [True]
        api.get_data.return_value = {}
        monkeypatch.chdir(tmp_path)
        env = {"LLM_MODEL": "test-model", ENV_LOG_LEVEL: "info"}
        with (
            patch.dict(os.environ, env, clear=True),
            patch("fsm_llm.runner.dotenv.load_dotenv"),
            patch("fsm_llm.runner.API.from_file", return_value=api),
        ):
            assert runner.main("/tmp/t.json", 5, 1000) == 0
        assert "System: HELLO" in capsys.readouterr().out


# --------------------------------------------------------------
# F-SU-07
# --------------------------------------------------------------


def _json_lines(emit):
    buf = io.StringIO()
    logger.enable("fsm_llm")
    hid = logger.add(
        _make_json_sink(buf), level="DEBUG", filter=prepare_log_record, colorize=False
    )
    try:
        emit()
    finally:
        logger.remove(hid)
    return [json.loads(line) for line in buf.getvalue().splitlines()]


class TestJsonLineIdentity:
    def test_host_record_is_not_labelled_as_the_library(self):
        (entry,) = _json_lines(lambda: logger.info("host app line"))
        assert entry["message"] == "host app line"
        assert "package" not in entry

    def test_library_record_is_labelled(self):
        lib = logger.patch(lambda r: r.update(name="fsm_llm.pipeline"))
        (entry,) = _json_lines(lambda: lib.info("lib line"))
        assert entry["package"] == "fsm_llm"
        assert entry["module"] == "fsm_llm.pipeline"

    def test_explicit_package_extra_is_kept(self):
        (entry,) = _json_lines(lambda: logger.bind(package="mine").info("x"))
        assert entry["package"] == "mine"

    def test_bound_extra_cannot_overwrite_core_fields(self):
        (entry,) = _json_lines(
            lambda: logger.bind(level="HACK", message="spoof", line=-1, ok=1).info(
                "real"
            )
        )
        assert entry["level"] == "INFO"
        assert entry["message"] == "real"
        assert entry["line"] != -1
        assert entry["ok"] == 1

    def test_core_fields_keep_their_order_before_extras(self):
        (entry,) = _json_lines(lambda: logger.bind(zz=1).info("x"))
        keys = list(entry)
        assert keys[:6] == [
            "timestamp",
            "level",
            "message",
            "module",
            "function",
            "line",
        ]

    def test_record_to_json_is_pure_on_the_record(self):
        record = {
            "time": __import__("datetime").datetime(2026, 1, 1),
            "level": MagicMock(),
            "message": "m",
            "name": "n",
            "function": "f",
            "line": 3,
            "extra": {"message": "evil"},
            "exception": None,
        }
        record["level"].name = "INFO"
        assert json.loads(_record_to_json(record))["message"] == "m"


# --------------------------------------------------------------
# F-SU-08
# --------------------------------------------------------------


class TestFileLoggingHandlerId:
    def test_setup_file_logging_returns_the_handler_id(self, tmp_path):
        hid = setup_file_logging(str(tmp_path / "logs"))
        assert isinstance(hid, int) and hid > 0
        logger.remove(hid)  # removable by the returned id

    def test_second_call_is_deduplicated(self, tmp_path):
        setup_file_logging(str(tmp_path / "logs"))
        assert setup_file_logging(str(tmp_path / "logs")) == -1

    def test_unknown_sink_is_refused(self):
        with pytest.raises(ValueError, match="sink"):
            setup_logging(sink="bogus")

    def test_unknown_format_is_refused(self):
        with pytest.raises(ValueError, match="format"):
            setup_logging(sink=LOG_SINK_STDOUT, format="bogus")

    def test_unknown_env_format_is_refused(self, monkeypatch):
        monkeypatch.setenv(ENV_LOG_FORMAT, "bogus")
        with pytest.raises(ValueError, match="format"):
            setup_logging(sink=LOG_SINK_STDOUT)

    def test_format_case_is_normalised(self, capsys):
        setup_logging(sink=LOG_SINK_STDOUT, format="JSON")
        logger.info("as-json")
        line = capsys.readouterr().out.strip().splitlines()[-1]
        assert json.loads(line)["message"] == "as-json"

    def test_refused_call_registers_nothing(self):
        before = list(log_module._library_handler_ids)
        with pytest.raises(ValueError):
            setup_logging(sink="bogus")
        assert log_module._library_handler_ids == before

    def test_known_values_still_accepted(self, tmp_path):
        for sink in (LOG_SINK_STDERR, LOG_SINK_STDOUT):
            for fmt in (LOG_FORMAT_HUMAN, LOG_FORMAT_JSON):
                assert setup_logging(sink=sink, format=fmt) != 0
        assert LOG_SINK_FILE


# --------------------------------------------------------------
# F-XS-09: CLI entry points
# --------------------------------------------------------------

_FSM = _REPO / "examples" / "basic" / "form_filling" / "fsm.json"


def _stderr_of(args, env_extra=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("FSM_LLM_LOG")}
    env.pop("LOGURU_LEVEL", None)
    env.update(env_extra or {})
    proc = subprocess.run(
        [sys.executable, *args],
        capture_output=True,
        text=True,
        cwd=str(_REPO),
        env=env,
        timeout=120,
    )
    proc.stderr = _plain(proc.stderr)
    return proc


class TestCliEntryPointsLogAtInfo:
    @pytest.mark.parametrize(
        "args",
        [
            ["-m", "fsm_llm", "--mode", "validate", "--fsm", str(_FSM)],
            [
                "-c",
                "import sys; from fsm_llm.validator import main_cli; "
                f"sys.argv=['x','--fsm',{str(_FSM)!r}]; main_cli()",
            ],
        ],
        ids=["fsm-llm --mode validate", "fsm-llm-validate"],
    )
    def test_validate_prints_no_debug_record(self, args):
        proc = _stderr_of(args)
        assert proc.returncode == 0, proc.stderr
        assert "| DEBUG" not in proc.stderr
        assert "| INFO" in proc.stderr  # diagnostics still reach the user
        assert json.loads(proc.stdout)["is_valid"] is True

    def test_validate_debug_is_opt_in_through_the_env(self):
        proc = _stderr_of(
            ["-m", "fsm_llm", "--mode", "validate", "--fsm", str(_FSM)],
            {ENV_LOG_LEVEL: "debug"},
        )
        assert "| DEBUG" in proc.stderr

    def test_visualize_cli_keeps_stdout_clean_and_prints_no_debug(self):
        proc = _stderr_of(
            [
                "-c",
                "import sys; from fsm_llm.visualizer import main_cli; "
                f"sys.argv=['x','--fsm',{str(_FSM)!r}]; main_cli()",
            ]
        )
        assert proc.returncode == 0, proc.stderr
        assert "| DEBUG" not in proc.stderr
        assert "Form Filling" in proc.stdout


# --------------------------------------------------------------
# F-AF-16: one path for enable_debug_logging
# --------------------------------------------------------------


_LIB = logger.patch(lambda r: r.update(name="fsm_llm.sample"))


class TestEnableDebugLoggingIsOnePath:
    def test_function_lives_in_logging_and_is_re_exported(self):
        import fsm_llm

        assert fsm_llm.enable_debug_logging is log_module.enable_debug_logging

    def test_registers_the_shared_human_format(self, capsys):
        import fsm_llm

        fsm_llm.enable_debug_logging()
        _LIB.debug("dbg-line")
        err = capsys.readouterr().err
        assert "dbg-line" in err
        # setup_logging's LOG_HUMAN_FORMAT carries the date; the old inline
        # format string did not.
        assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", _plain(err))

    def test_ignores_a_json_env_format(self, monkeypatch, capsys):
        import fsm_llm

        monkeypatch.setenv(ENV_LOG_FORMAT, "json")
        fsm_llm.enable_debug_logging()
        _LIB.debug("dbg-line")
        err = capsys.readouterr().err
        assert "dbg-line" in err and '"message"' not in err
