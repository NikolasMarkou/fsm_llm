"""Tests for the fsm-llm-eval command line (fsm_llm.eval.__main__)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fsm_llm.eval.__main__ import build_parser, main_cli

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
