"""Tests for the fsm-llm-eval command line (fsm_llm.eval.__main__)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fsm_llm.eval import __main__ as cli
from fsm_llm.eval.__main__ import build_parser, main_cli
from fsm_llm.eval.examples import ExampleReport, ExampleResult

_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def examples_dir(tmp_path: Path) -> Path:
    for name, body in (
        ("basic/ok/run.py", "print('State: a')\nprint('State: b')\n"),
        ("basic/bad/run.py", "import sys\nsys.exit(3)\n"),
    ):
        path = tmp_path / "examples" / name
        path.parent.mkdir(parents=True)
        path.write_text(body, encoding="utf-8")
    return tmp_path / "examples"


class TestUsage:
    def test_no_subcommand_exits_1(self, capsys):
        assert main_cli([]) == 1
        assert "COMMAND" in capsys.readouterr().err

    def test_unknown_flag_exits_1_not_2(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main_cli(["examples", "--no-such-flag"])
        assert exc.value.code == 1

    def test_help_exits_0(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main_cli(["examples", "--help"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        for flag in (
            "--model",
            "--workers",
            "--timeout",
            "--category",
            "--filter",
            "--output-dir",
            "--list",
            "--examples-dir",
            "--python",
            "--config",
            "--fail-under",
        ):
            assert flag in out

    def test_version(self, capsys):
        assert main_cli(["--version"]) == 0
        assert capsys.readouterr().out.startswith("fsm-llm-eval ")

    def test_flag_defaults_are_none_so_config_layers_apply(self):
        args = build_parser().parse_args(["examples"])
        for name in ("model", "workers", "timeout", "output_dir", "fail_under"):
            assert getattr(args, name) is None


class TestList:
    def test_list_prints_names_and_modes(self, examples_dir, capsys):
        assert (
            main_cli(["examples", "--list", "--examples-dir", str(examples_dir)]) == 0
        )
        out = capsys.readouterr().out
        assert out.startswith("Discovered 2 examples:\n\n")
        assert "  basic/bad" in out and "[automated, no stdin]" in out

    def test_no_match_exits_1(self, examples_dir, capsys):
        argv = [
            "examples",
            "--list",
            "--examples-dir",
            str(examples_dir),
            "--filter",
            "zzz",
        ]
        assert main_cli(argv) == 1
        assert "No examples found" in capsys.readouterr().err

    def test_repository_examples_are_listed(self, capsys, monkeypatch):
        monkeypatch.chdir(_REPO_ROOT)
        assert main_cli(["examples", "--list"]) == 0
        out = capsys.readouterr().out
        assert "  basic/simple_greeting " in out
        assert "  classification/classified_transitions_manual " in out


class TestConfigErrors:
    def test_bad_config_key_exits_1(self, examples_dir, tmp_path, capsys):
        cfg = tmp_path / "cfg.json"
        cfg.write_text(json.dumps({"wrokers": 3}), encoding="utf-8")
        argv = [
            "examples",
            "--list",
            "--examples-dir",
            str(examples_dir),
            "--config",
            str(cfg),
        ]
        assert main_cli(argv) == 1
        assert "cfg.json" in capsys.readouterr().err

    def test_missing_config_exits_1(self, examples_dir, tmp_path):
        argv = ["examples", "--list", "--config", str(tmp_path / "nope.json")]
        assert main_cli(argv) == 1

    def test_bad_flag_value_exits_1(self, examples_dir):
        assert main_cli(["examples", "--list", "--workers", "0"]) == 1

    def test_config_file_supplies_settings(self, examples_dir, tmp_path, capsys):
        cfg = tmp_path / "cfg.json"
        cfg.write_text(
            json.dumps({"examples_dir": str(examples_dir), "name_filter": "ok"}),
            encoding="utf-8",
        )
        assert main_cli(["examples", "--list", "--config", str(cfg)]) == 0
        assert "Discovered 1 examples" in capsys.readouterr().out


class TestRun:
    def _argv(self, examples_dir: Path, out: Path, *extra: str) -> list[str]:
        return [
            "examples",
            "--examples-dir",
            str(examples_dir),
            "--output-dir",
            str(out),
            "--model",
            "ollama_chat/test:1b",
            *extra,
        ]

    def test_run_writes_outputs_and_exits_0_on_low_score(
        self, examples_dir, tmp_path, capsys
    ):
        out = tmp_path / "run"
        assert main_cli(self._argv(examples_dir, out)) == 0
        assert (out / "scorecard.md").is_file()
        data = json.loads((out / "results.json").read_text())
        assert data["model"] == "ollama_chat/test:1b"
        assert data["total_examples"] == 2
        stdout = capsys.readouterr().out
        assert "Health Score:" in stdout and "basic/bad" in stdout

    def test_fail_under_exits_2_when_below(self, examples_dir, tmp_path, capsys):
        out = tmp_path / "run"
        assert main_cli(self._argv(examples_dir, out, "--fail-under", "90")) == 2
        assert "below --fail-under" in capsys.readouterr().err

    def test_fail_under_exits_0_when_met(self, examples_dir, tmp_path):
        out = tmp_path / "run"
        argv = self._argv(examples_dir, out, "--filter", "ok", "--fail-under", "100")
        assert main_cli(argv) == 0

    def test_nonempty_output_dir_exits_1(self, examples_dir, tmp_path, capsys):
        out = tmp_path / "run"
        out.mkdir()
        (out / "results.json").write_text("{}")
        assert main_cli(self._argv(examples_dir, out)) == 1
        assert "not empty" in capsys.readouterr().err

    def test_main_cli_does_not_enable_library_logging(self, examples_dir, monkeypatch):
        import fsm_llm.logging as fsm_logging

        called: list[str] = []
        monkeypatch.setattr(fsm_logging, "setup_cli_logging", called.append)
        main_cli(["examples", "--list", "--examples-dir", str(examples_dir)])
        assert called == []


class TestInputErrors:
    def test_missing_examples_dir_is_named(self, tmp_path, capsys):
        missing = tmp_path / "nowhere"
        assert main_cli(["examples", "--list", "--examples-dir", str(missing)]) == 1
        err = capsys.readouterr().err
        assert "Examples directory not found" in err and str(missing) in err

    def test_no_match_names_the_directory(self, examples_dir, capsys):
        argv = ["examples", "--list", "--examples-dir", str(examples_dir)]
        assert main_cli([*argv, "--filter", "zzz"]) == 1
        assert str(examples_dir.resolve()) in capsys.readouterr().err

    def test_non_utf8_config_exits_1_with_path(self, examples_dir, tmp_path, capsys):
        cfg = tmp_path / "cfg.json"
        cfg.write_bytes(b"\xff\xfe{")
        argv = ["examples", "--list", "--examples-dir", str(examples_dir)]
        assert main_cli([*argv, "--config", str(cfg)]) == 1
        assert str(cfg) in capsys.readouterr().err


def _report_scoring(score_sum: int, count: int, run_dir: Path) -> ExampleReport:
    """A fake report of ``count`` examples whose scores add up to ``score_sum``."""
    scores = [4] * (score_sum // 4) + ([score_sum % 4] if score_sum % 4 else [])
    scores += [0] * (count - len(scores))
    results = [
        ExampleResult(f"c/e{i}", "c", 0, 0.0, "", "", False, score=s)
        for i, s in enumerate(scores)
    ]
    return ExampleReport(run_dir, "m", results, score_sum / (count * 4) * 100, 0.0)


class TestFailUnderBoundary:
    @pytest.mark.parametrize(("threshold", "code"), [("58", 0), ("58.1", 2)])
    def test_health_116_of_200_meets_58(
        self, examples_dir, tmp_path, monkeypatch, threshold, code
    ):
        """Defect guarded: 116/200 health is 57.99999... in floats, so
        ``--fail-under 58`` exited 2 while printing 58.0% (review W1)."""
        monkeypatch.setattr(
            cli,
            "run_examples",
            lambda targets, config, model, run_dir, progress: _report_scoring(
                116, 50, run_dir
            ),
        )
        argv = ["examples", "--examples-dir", str(examples_dir)]
        out = ["--output-dir", str(tmp_path / "o"), "--fail-under", threshold]
        assert main_cli([*argv, *out]) == code


class TestInterruptExit:
    def test_ctrl_c_exits_130_with_partial_report(
        self, examples_dir, tmp_path, monkeypatch, capsys
    ):
        def ctrl_c(*_args):
            raise KeyboardInterrupt

        monkeypatch.setattr(cli, "_print_progress", ctrl_c)
        out = tmp_path / "run"
        argv = ["examples", "--examples-dir", str(examples_dir), "--workers", "1"]
        assert main_cli([*argv, "--output-dir", str(out)]) == 130
        assert json.loads((out / "results.json").read_text())["interrupted"] is True
        assert "Interrupted" in capsys.readouterr().err
