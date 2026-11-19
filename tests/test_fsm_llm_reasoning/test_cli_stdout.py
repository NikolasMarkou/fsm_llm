"""The reasoning CLI's own output reaches the terminal without the logger.

``fsm_llm`` log records are disabled library-wide, so product output goes
through ``print()`` and failures through ``report_error``; these cases call
``main()`` in-process, where a logger-based message was dropped.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from fsm_llm.reasoning.__main__ import main

_PROBE = (
    "import sys\n"
    "import fsm_llm.reasoning.__main__ as m\n"
    "def _solve(*_args):\n"
    "    raise RuntimeError('boom')\n"
    "m.solve_problem_with_engine = _solve\n"
    "sys.argv = ['x', 'what is 2+2?', *sys.argv[1:]]\n"
    "sys.exit(m.main())\n"
)


def _run(monkeypatch: pytest.MonkeyPatch, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["x", *argv])
    return main()


class TestOutputWithoutTheLogger:
    def test_list_types_prints_to_stdout(self, monkeypatch, capsys) -> None:
        assert _run(monkeypatch, "--list-types") == 0
        captured = capsys.readouterr()
        assert "analytical" in captured.out
        assert captured.err == ""

    @pytest.mark.parametrize(
        ("context", "message"),
        [("{bad", "invalid JSON context"), ("[1]", "must be a JSON object")],
    )
    def test_a_bad_context_is_one_stderr_line(
        self, monkeypatch, capsys, context: str, message: str
    ) -> None:
        with pytest.raises(SystemExit) as excinfo:
            _run(monkeypatch, "hello", "--context", context)
        assert excinfo.value.code == 1
        err = capsys.readouterr().err
        assert err.startswith("python -m fsm_llm.reasoning: error: ")
        assert message in err

    def test_an_argument_error_is_reported(self, monkeypatch, capsys) -> None:
        assert _run(monkeypatch, "hello", "-q", "-v") == 1
        assert "mutually exclusive" in capsys.readouterr().err


class TestTracebackFollowsTheParsedFlag:
    @pytest.mark.parametrize(
        ("flag", "traceback"), [("--verb", True), ("-v", True), (None, False)]
    )
    def test_the_abbreviation_counts_as_verbose(
        self, flag: str | None, traceback: bool
    ) -> None:
        completed = subprocess.run(
            [sys.executable, "-c", _PROBE, *([flag] if flag else [])],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert completed.returncode == 1, completed.stderr
        assert "error: boom" in completed.stderr
        assert ("Traceback" in completed.stderr) is traceback


_AS_MAIN = (
    "import runpy, sys\n"
    "import fsm_llm.reasoning.engine as engine\n"
    "MODE = sys.argv[1]\n"
    "class Engine:\n"
    "    def __init__(self, *a, **k):\n"
    "        pass\n"
    "    def solve_problem(self, *a, **k):\n"
    "        if MODE == 'interrupt':\n"
    "            raise KeyboardInterrupt\n"
    "        if MODE == 'fail':\n"
    "            raise RuntimeError('engine exploded')\n"
    "        return 'four', {'reasoning_trace': {'steps': []}}\n"
    "engine.ReasoningEngine = Engine\n"
    "rest = sys.argv[2:]\n"
    "sys.argv = ['python -m fsm_llm.reasoning', 'what is 2+2?', '--model', 'x/y', *rest]\n"
    "runpy.run_module('fsm_llm.reasoning', run_name='__main__', alter_sys=True)\n"
)


class TestEveryErrorIsOneStderrLine:
    """D-053 of plan 51fde95b: run as ``__main__`` (the module name the library
    disable does not cover), a failed solve, a failed save and Ctrl-C each print
    one line on stderr and nothing from the logger."""

    @staticmethod
    def _run(mode: str, *argv: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-c", _AS_MAIN, mode, *argv],
            capture_output=True,
            text=True,
            timeout=60,
        )

    @pytest.mark.parametrize(
        ("mode", "argv", "text"),
        [
            ("fail", (), "error: engine exploded"),
            ("interrupt", (), "interrupted by user"),
        ],
        ids=["failed-solve", "ctrl-c"],
    )
    def test_one_line_and_no_logger_record(self, mode, argv, text) -> None:
        completed = self._run(mode, *argv)
        lines = [ln for ln in completed.stderr.splitlines() if ln.strip()]
        assert completed.returncode == 1, completed.stderr
        assert len(lines) == 1, completed.stderr
        assert lines[0].startswith("python -m fsm_llm.reasoning: ")
        assert text in lines[0] and "ERROR" not in lines[0] and "|" not in lines[0]

    def test_a_failed_save_is_one_line_too(self, tmp_path) -> None:
        # A parent that is a regular file cannot be a directory on any platform
        # (``/dev/null/x`` is a writable path on Windows).
        blocker = tmp_path / "blocker"
        blocker.write_text("x")
        completed = self._run("ok", "--save", str(blocker / "x" / "out.txt"))
        lines = [ln for ln in completed.stderr.splitlines() if ln.strip()]
        assert completed.returncode == 1, completed.stderr
        assert len(lines) == 1, completed.stderr
        assert lines[0].startswith("python -m fsm_llm.reasoning: ")
        assert "error:" in lines[0] and "ERROR" not in lines[0] and "|" not in lines[0]
