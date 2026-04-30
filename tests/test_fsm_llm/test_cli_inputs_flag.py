"""Tests for the R12-residual ``--inputs FILE`` CLI flag.

Plan: plans/plan_2026-04-27_32652286/plan.md — Step 2 (Bundle A).

Coverage:

- SC5: `fsm-llm run pkg.mod:factory --inputs path.json` runs the factory
  with the JSON dict unpacked as **env to .invoke().
- E9: bad path / malformed JSON → exit code 5 with informative stderr.
- --env wins over --inputs on collision (CLI explicit beats file-supplied).
- --inputs file with non-dict top level → exit code 5.
- --inputs absent → existing behavior preserved (env-only).
"""

from __future__ import annotations

import json

import pytest

from fsm_llm.cli.main import main_cli

# ---------------------------------------------------------------------------
# A tiny pure factory we can target without LLMs / network.
# Lives here so tests can use `tests.test_fsm_llm.test_cli_inputs_flag:trivial_factory`.
# ---------------------------------------------------------------------------


def trivial_factory():
    """Factory returning a closed term: app(abs_('x', var('x')), var('y'))."""
    from fsm_llm.runtime import abs_, app, var

    return app(abs_("x", var("x")), var("y"))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture
def inputs_json(tmp_path):
    """Write a JSON inputs file and return its path."""
    p = tmp_path / "inputs.json"
    p.write_text(json.dumps({"y": "from-file"}))
    return str(p)


@pytest.fixture
def malformed_inputs_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not valid json")
    return str(p)


@pytest.fixture
def list_inputs_json(tmp_path):
    """A JSON file whose top level is a list, not a dict."""
    p = tmp_path / "listy.json"
    p.write_text(json.dumps(["item1", "item2"]))
    return str(p)


class TestInputsHappyPath:
    def test_inputs_file_loaded_and_unpacked(self, inputs_json, capsys):
        rc = main_cli(
            [
                "run",
                "tests.test_fsm_llm.test_cli_inputs_flag:trivial_factory",
                "--inputs",
                inputs_json,
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        # The factory's reduction yields var("y") env-resolved → "from-file".
        assert "from-file" in out

    def test_inputs_file_unpacks_dict(self, tmp_path, capsys):
        """Multiple keys flow through correctly."""
        path = tmp_path / "multi.json"
        path.write_text(json.dumps({"y": "alpha", "extra": "ignored"}))
        rc = main_cli(
            [
                "run",
                "tests.test_fsm_llm.test_cli_inputs_flag:trivial_factory",
                "--inputs",
                str(path),
            ]
        )
        assert rc == 0
        assert "alpha" in capsys.readouterr().out


class TestInputsErrorPaths:
    def test_missing_file_exits_5(self, capsys):
        # E9: bad path → exit 5.
        with pytest.raises(SystemExit) as exc_info:
            main_cli(
                [
                    "run",
                    "tests.test_fsm_llm.test_cli_inputs_flag:trivial_factory",
                    "--inputs",
                    "/nonexistent/inputs/path/12345.json",
                ]
            )
        assert exc_info.value.code == 5
        err = capsys.readouterr().err
        assert "not found" in err.lower() or "no such file" in err.lower()

    def test_malformed_json_exits_5(self, malformed_inputs_json, capsys):
        # E9: malformed JSON → exit 5 with parse error in message.
        with pytest.raises(SystemExit) as exc_info:
            main_cli(
                [
                    "run",
                    "tests.test_fsm_llm.test_cli_inputs_flag:trivial_factory",
                    "--inputs",
                    malformed_inputs_json,
                ]
            )
        assert exc_info.value.code == 5
        err = capsys.readouterr().err
        assert "json" in err.lower()

    def test_non_dict_top_level_exits_5(self, list_inputs_json, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main_cli(
                [
                    "run",
                    "tests.test_fsm_llm.test_cli_inputs_flag:trivial_factory",
                    "--inputs",
                    list_inputs_json,
                ]
            )
        assert exc_info.value.code == 5
        err = capsys.readouterr().err
        assert "object" in err.lower() or "dict" in err.lower()


class TestEnvVsInputsPrecedence:
    def test_env_wins_over_inputs_on_key_collision(self, inputs_json, capsys):
        # --env y=cli-wins overrides {"y": "from-file"} from inputs.json.
        rc = main_cli(
            [
                "run",
                "tests.test_fsm_llm.test_cli_inputs_flag:trivial_factory",
                "--inputs",
                inputs_json,
                "--env",
                "y=cli-wins",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "cli-wins" in out
        assert "from-file" not in out


class TestNoInputsFlag:
    def test_no_inputs_flag_works_unchanged(self, capsys):
        # Sanity: pre-existing env-only path still works.
        rc = main_cli(
            [
                "run",
                "tests.test_fsm_llm.test_cli_inputs_flag:trivial_factory",
                "--env",
                "y=env-only",
            ]
        )
        assert rc == 0
        assert "env-only" in capsys.readouterr().out
