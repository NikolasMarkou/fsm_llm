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
