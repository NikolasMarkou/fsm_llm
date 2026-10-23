"""Tests for ``fsm_llm.cli_support``: the one skeleton under ``fsm-llm-okf`` and
``fsm-llm-skills`` (exit mapping, BrokenPipe parking, output escape,
``--max-seconds`` type). Control and non-ASCII characters are built with
``chr``: this file stays pure ASCII."""

from __future__ import annotations

import argparse
import inspect
import io
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import fsm_llm
import fsm_llm.agent_skills_cli as skills_cli
import fsm_llm.okf_cli as okf_cli
from fsm_llm import cli_support

_CLIS = pytest.mark.parametrize("mod", [okf_cli, skills_cli], ids=["okf", "skills"])
_ESC, _BIDI, _CJK = chr(0x1B), chr(0x202E), chr(0x7F51)
_SRC = Path(fsm_llm.__file__).resolve().parent.parent


class _Boom(Exception):
    """A stand-in for a CLI's domain error."""


def _raises(error: BaseException | type[BaseException]) -> Any:
    def command(args: Any) -> int:
        raise error

    return command


def _parser(func: Any) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("t")
    parser.add_argument("--n", type=cli_support.positive_seconds, default=1.0)
    parser.set_defaults(func=func)
    return parser


def _run(func: Any, *argv: str) -> int:
    return cli_support.run_cli(_parser(func), list(argv), prog="t", errors=(_Boom,))


def _max_seconds_types(mod: Any) -> list[Any]:
    """The ``type`` of every ``--max-seconds`` option of *mod*'s parser."""
    subs = [
        a
        for a in mod._build_parser()._actions
        if isinstance(a, argparse._SubParsersAction)
    ]
    return [
        action.type
        for sub in subs[0].choices.values()
        for action in sub._actions
        if "--max-seconds" in action.option_strings
    ]


class TestBothClisUseTheSharedCode:
    @_CLIS
    def test_the_entry_points_and_the_type_are_the_shared_ones(self, mod: Any) -> None:
        types = _max_seconds_types(mod)

        assert mod.run_cli is cli_support.run_cli
        assert mod.exit_main is cli_support.exit_main
        assert types and all(t is cli_support.positive_seconds for t in types)

    @_CLIS
    def test_the_skeleton_is_not_re_inlined(self, mod: Any) -> None:
        source = inspect.getsource(mod)
        needles = ["BrokenPipeError", "os.dup2", ".reconfigure(", "math.nan"]

        assert [n for n in needles if n in source] == []

    def test_the_skills_cli_imports_nothing_from_the_okf_cli(self) -> None:
        assert "okf_cli" not in inspect.getsource(skills_cli)

    def test_importing_it_loads_no_subpackage_and_fsm_llm_does_not_load_it(
        self, tmp_path: Path
    ) -> None:
        probe = (
            f"import sys; sys.path.insert(0, {str(_SRC)!r})\n"
            "import fsm_llm\n"
            "assert 'fsm_llm.cli_support' not in sys.modules\n"
            "import fsm_llm.cli_support\n"
            "subs = ('agents', 'reasoning', 'workflows', 'monitor', 'eval')\n"
            "bad = [m for m in sys.modules if m.split('.')[:2] in "
            "[['fsm_llm', s] for s in subs]]\n"
            "assert not bad, bad\n"
        )

        done = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=tmp_path,
        )

        assert done.returncode == 0, done.stderr


class TestRunCli:
    def test_the_return_code_of_the_command_is_returned(self) -> None:
        assert (_run(lambda args: 0), _run(lambda args: 7)) == (0, 7)

    def test_ctrl_c_is_one_line_and_130(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert _run(_raises(KeyboardInterrupt)) == 130
        assert capsys.readouterr().err == "t: interrupted\n"

    @pytest.mark.parametrize("error", [_Boom("bad input"), FileNotFoundError("gone")])
    def test_a_domain_or_os_error_is_one_stripped_line_and_1(
        self, capsys: pytest.CaptureFixture[str], error: Exception
    ) -> None:
        assert _run(_raises(error)) == 1
        assert capsys.readouterr().err == f"t: error: {error}\n"

    def test_hostile_error_text_stays_one_short_line(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        text = f"a{_ESC}[31m\nb{_BIDI}c\r\n{'z' * 5000}"

        assert _run(_raises(_Boom(text))) == 1
        err = capsys.readouterr().err
        assert len(err.splitlines()) == 1 and err.endswith("\n")
        assert _ESC not in err and _BIDI not in err and len(err) < 1100

    @pytest.mark.parametrize("error", [RuntimeError, BrokenPipeError])
    def test_a_bug_and_a_broken_pipe_are_left_to_the_caller(
        self, error: type[BaseException]
    ) -> None:
        with pytest.raises(error):
            _run(_raises(error))

    def test_a_usage_error_is_exit_2_and_echoed_text_is_no_traceback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        raw = io.BytesIO()
        monkeypatch.setattr(sys, "stderr", io.TextIOWrapper(raw, encoding="ascii"))
        monkeypatch.setattr(sys, "stdout", io.StringIO())  # no reconfigure: left alone

        with pytest.raises(SystemExit) as info:
            _run(lambda args: 0, _CJK)  # "unrecognized arguments" echoes it
        sys.stderr.flush()

        assert info.value.code == 2 and b"\\u7f51" in raw.getvalue()


class TestExitMain:
    @pytest.mark.parametrize("code", [0, 1, 130])
    def test_it_exits_with_the_code_of_main(self, code: int) -> None:
        with pytest.raises(SystemExit) as info:
            cli_support.exit_main(lambda: code)

        assert info.value.code == code

    @_CLIS
    def test_a_broken_stdout_pipe_parks_stdout_and_exits_1(
        self, mod: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def talk(args: Any) -> int:
            print("a line for a reader that is gone")
            return 0

        read_fd, write_fd = os.pipe()
        os.close(read_fd)  # the reader is gone: the first flush gets EPIPE
        stream = open(write_fd, "w", closefd=True)
        monkeypatch.setattr(sys, "stdout", stream)
        monkeypatch.setattr(mod, "_cmd_list", talk)
        monkeypatch.setattr(sys, "argv", [mod._PROG, "list", "somewhere"])

        try:
            with pytest.raises(SystemExit) as info:
                mod.main_cli()
            assert info.value.code == 1
            stream.flush()  # parked on /dev/null now: nothing left to fail
        finally:
            monkeypatch.undo()
            stream.close()


class TestSmallHelpers:
    @pytest.mark.parametrize("text", ["0", "-1", "nan", "inf", "-inf", "x", ""])
    def test_positive_seconds_refuses_what_is_not_a_finite_positive_number(
        self, text: str
    ) -> None:
        with pytest.raises(argparse.ArgumentTypeError):
            cli_support.positive_seconds(text)

    def test_positive_seconds_accepts_a_positive_number(self) -> None:
        assert cli_support.positive_seconds("0.5") == 0.5
