"""``--env-file`` on every CLI that needs a model.

Plan d454fd62 step 18 (D-005). ``fsm-llm`` got the flag in step 17; here
``fsm-llm-eval``, ``fsm-llm-meta``, ``python -m fsm_llm.reasoning`` and
``fsm-llm-monitor`` get the same one through ``cli_support.add_env_file`` and
``apply_env_file``. The CLIs that call no model (``fsm-llm-okf``,
``fsm-llm-skills``, ``fsm-llm-validate``, ``fsm-llm-visualize``) must not have
it. Precedence: ``--model`` flag, process environment, file, then one stderr
line and exit 1.

Every child process runs in a temporary directory with a scrubbed environment
whose model-server variables point at a closed port. The "file supplies the
model" tests replace the first step after model resolution (an engine class, a
run directory, an agent class, the uvicorn server) with a probe inside the
child, so the real argument parsing, ``--env-file`` handling and model
resolution run, and no model is ever called. No test prints the content of a
``.env`` file.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import fsm_llm
from fsm_llm import cli_support
from fsm_llm.definitions import ModelNotConfiguredError

_REPO = Path(fsm_llm.__file__).resolve().parent.parent
_DATASET = _REPO / "evaluation" / "datasets" / "simple_greeting_cases.json"
_MESSAGE = ModelNotConfiguredError.DEFAULT_MESSAGE
_NO_MODEL_CLIS = (
    "fsm_llm.okf_cli",
    "fsm_llm.agent_skills_cli",
    "fsm_llm.validator",
    "fsm_llm.visualizer",
)


def _child_env(**extra: str) -> dict[str, str]:
    keep = ("PATH", "HOME", "SYSTEMROOT", "TMPDIR", "TEMP", "TMP", "LANG", "LC_ALL")
    env = {key: os.environ[key] for key in keep if key in os.environ}
    env["PYTHONIOENCODING"] = "utf-8"
    env["OLLAMA_API_BASE"] = "http://127.0.0.1:9"
    env["OLLAMA_HOST"] = "http://127.0.0.1:9"
    env["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    env.update(extra)
    return env


def _run(args: list[str], cwd: Path, **extra_env: str):
    return subprocess.run(
        [sys.executable, *args],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=cwd,
        env=_child_env(**extra_env),
        check=False,
    )


def _lines(proc: subprocess.CompletedProcess[str]) -> list[str]:
    return [line for line in proc.stderr.splitlines() if line.strip()]


# A probe replaces the first step after model resolution, prints the model the
# CLI resolved as ``MODEL=<name>`` and stops the process with exit 0.
_PROBES = {
    "reasoning": """
        import sys
        import fsm_llm.reasoning.__main__ as cli
        from fsm_llm.llm import resolve_model

        class Engine:
            def __init__(self, model=None, **kwargs):
                print("MODEL=" + resolve_model(model))
                sys.exit(0)

        cli.ReasoningEngine = Engine
        sys.argv = ["python -m fsm_llm.reasoning", "a problem", *ARGS]
        sys.exit(cli.main())
    """,
    "eval": """
        import sys
        import fsm_llm.eval.__main__ as cli

        def probe(config, model, dataset):
            print("MODEL=" + model)
            sys.exit(0)

        cli.create_run_dir = probe
        sys.exit(cli.main_cli(["run", DATASET, *ARGS]))
    """,
    "meta": """
        import sys
        import fsm_llm.agents.meta_cli as cli

        class Agent:
            def __init__(self, config=None, **kwargs):
                print("MODEL=" + config.model)
                sys.exit(0)

        cli.MetaBuilderAgent = Agent
        sys.exit(cli.main(ARGS))
    """,
    "monitor": """
        import os, sys
        import uvicorn
        import fsm_llm.monitor.__main__ as cli

        class Server:
            started = True
            def __init__(self, config): pass
            def run(self):
                print("MODEL=" + os.environ.get("LLM_MODEL", "<unset>"))
                sys.exit(0)

        uvicorn.Server = Server
        sys.argv = ["fsm-llm-monitor", "--no-browser", *ARGS]
        cli.main_cli()
    """,
}


def _probe(name: str, args: list[str], cwd: Path, **extra_env: str):
    script = f"ARGS = {args!r}\nDATASET = {str(_DATASET)!r}\n" + textwrap.dedent(
        _PROBES[name]
    )
    return _run(["-c", script], cwd, **extra_env)


def _model_line(proc: subprocess.CompletedProcess[str]) -> str:
    found = [x for x in proc.stdout.splitlines() if x.startswith("MODEL=")]
    assert len(found) == 1, (proc.stdout, proc.stderr)
    return found[0]


# The four CLIs, as a real command line without a probe (the run stops before a
# model is called: a missing file or no model).
_REAL = {
    "reasoning": ["-m", "fsm_llm.reasoning", "a problem"],
    "eval": ["-m", "fsm_llm.eval", "run", str(_DATASET)],
    "meta": ["-m", "fsm_llm.agents.meta_cli"],
    "monitor": ["-m", "fsm_llm.monitor", "--no-browser"],
}
_NEEDS_A_MODEL = ("reasoning", "eval", "meta")
_ALL = (*_NEEDS_A_MODEL, "monitor")


@pytest.mark.parametrize("cli", _ALL)
class TestEnvFileFlagOnEachCli:
    def test_help_lists_the_flag(self, cli, tmp_path):
        proc = _run([*_REAL[cli], "--help"], tmp_path)
        assert proc.returncode == 0, proc.stderr
        assert "--env-file" in proc.stdout

    def test_a_missing_file_exits_1_with_one_line_and_no_traceback(self, cli, tmp_path):
        proc = _run([*_REAL[cli], "--env-file", str(tmp_path / "no.env")], tmp_path)
        assert proc.returncode == 1, proc.stderr
        assert "Traceback" not in proc.stderr
        lines = _lines(proc)
        assert len(lines) == 1, proc.stderr
        assert "no.env" in lines[0]

    def test_a_directory_is_not_a_file(self, cli, tmp_path):
        proc = _run([*_REAL[cli], "--env-file", str(tmp_path)], tmp_path)
        assert proc.returncode == 1, proc.stderr
        assert "Traceback" not in proc.stderr
        assert len(_lines(proc)) == 1, proc.stderr


@pytest.mark.parametrize("cli", _NEEDS_A_MODEL)
class TestModelFromTheFile:
    def test_unset_model_exits_1_with_one_line_and_no_traceback(self, cli, tmp_path):
        proc = _run(_REAL[cli], tmp_path)
        assert proc.returncode == 1, proc.stderr
        assert "Traceback" not in proc.stderr
        lines = _lines(proc)
        assert len(lines) == 1, proc.stderr
        assert _MESSAGE in lines[0]

    def test_a_file_without_a_model_still_exits_1_with_one_line(self, cli, tmp_path):
        env_file = tmp_path / "my.env"
        env_file.write_text("SOME_OTHER_VARIABLE=1\n", encoding="utf-8")
        proc = _run([*_REAL[cli], "--env-file", str(env_file)], tmp_path)
        assert proc.returncode == 1, proc.stderr
        assert "Traceback" not in proc.stderr
        lines = _lines(proc)
        assert len(lines) == 1, proc.stderr
        assert _MESSAGE in lines[0]

    def test_a_cwd_dotenv_is_not_read(self, cli, tmp_path):
        (tmp_path / ".env").write_text("LLM_MODEL=stale\n", encoding="utf-8")
        proc = _run(_REAL[cli], tmp_path)
        assert proc.returncode == 1, proc.stderr
        assert _MESSAGE in proc.stderr

    def test_the_file_supplies_the_model(self, cli, tmp_path):
        env_file = tmp_path / "my.env"
        env_file.write_text("LLM_MODEL=from-file\n", encoding="utf-8")
        proc = _probe(cli, ["--env-file", str(env_file)], tmp_path)
        assert proc.returncode == 0, proc.stderr
        assert _model_line(proc) == "MODEL=from-file"

    def test_the_process_environment_beats_the_file(self, cli, tmp_path):
        env_file = tmp_path / "my.env"
        env_file.write_text("LLM_MODEL=from-file\n", encoding="utf-8")
        proc = _probe(
            cli, ["--env-file", str(env_file)], tmp_path, LLM_MODEL="from-process"
        )
        assert proc.returncode == 0, proc.stderr
        assert _model_line(proc) == "MODEL=from-process"

    def test_the_model_flag_beats_the_file(self, cli, tmp_path):
        env_file = tmp_path / "my.env"
        env_file.write_text("LLM_MODEL=from-file\n", encoding="utf-8")
        proc = _probe(
            cli, ["--env-file", str(env_file), "--model", "from-flag"], tmp_path
        )
        assert proc.returncode == 0, proc.stderr
        assert _model_line(proc) == "MODEL=from-flag"


class TestMonitorFile:
    """The dashboard needs no model to start: it exports the file's variables
    so a later launch request finds ``LLM_MODEL`` (400 only when none is set)."""

    def test_starts_without_a_model_and_without_the_flag(self, tmp_path):
        proc = _probe("monitor", [], tmp_path)
        assert proc.returncode == 0, proc.stderr
        assert _model_line(proc) == "MODEL=<unset>"

    def test_the_file_reaches_the_environment(self, tmp_path):
        env_file = tmp_path / "my.env"
        env_file.write_text("LLM_MODEL=from-file\n", encoding="utf-8")
        proc = _probe("monitor", ["--env-file", str(env_file)], tmp_path)
        assert proc.returncode == 0, proc.stderr
        assert _model_line(proc) == "MODEL=from-file"

    def test_the_process_environment_beats_the_file(self, tmp_path):
        env_file = tmp_path / "my.env"
        env_file.write_text("LLM_MODEL=from-file\n", encoding="utf-8")
        proc = _probe(
            "monitor", ["--env-file", str(env_file)], tmp_path, LLM_MODEL="from-process"
        )
        assert _model_line(proc) == "MODEL=from-process"

    def test_a_missing_file_is_refused_before_the_server_starts(self, tmp_path):
        proc = _probe("monitor", ["--env-file", str(tmp_path / "no.env")], tmp_path)
        assert proc.returncode == 1, proc.stderr
        assert "MODEL=" not in proc.stdout
        assert len(_lines(proc)) == 1, proc.stderr


class TestEvalUsageStillExits1:
    """AGENTS.md: ``fsm-llm-eval`` exits 1 for a usage error, 2 only below
    ``--fail-under``; the new flag must not change that."""

    def test_a_flag_without_its_value_exits_1(self, tmp_path):
        proc = _run(
            ["-m", "fsm_llm.eval", "run", str(_DATASET), "--env-file"], tmp_path
        )
        assert proc.returncode == 1, proc.stderr

    def test_the_examples_command_takes_the_flag_too(self, tmp_path):
        proc = _run(
            ["-m", "fsm_llm.eval", "examples", "--env-file", str(tmp_path / "no.env")],
            tmp_path,
        )
        assert proc.returncode == 1, proc.stderr
        assert "Traceback" not in proc.stderr
        assert len(_lines(proc)) == 1, proc.stderr


class TestNoModelClisHaveNoFlag:
    @pytest.mark.parametrize("module", _NO_MODEL_CLIS)
    def test_help_has_no_env_file(self, module, tmp_path):
        args = ["-m", module, "--help"]
        proc = _run(args, tmp_path)
        assert proc.returncode == 0, proc.stderr
        assert "--env-file" not in proc.stdout

    def test_the_okf_and_skills_subcommands_have_none_either(self, tmp_path):
        for module, sub in (
            ("fsm_llm.okf_cli", "validate"),
            ("fsm_llm.agent_skills_cli", "validate"),
        ):
            proc = _run(["-m", module, sub, "--help"], tmp_path)
            assert "--env-file" not in proc.stdout, module

    def test_only_the_model_clis_import_the_helper(self):
        src = _REPO / "src"
        users = sorted(
            str(path.relative_to(src))
            for path in src.rglob("*.py")
            if "add_env_file" in path.read_text(encoding="utf-8")
        )
        assert users == [
            "__main__.py",
            "agents/meta_cli.py",
            "cli_support.py",
            "eval/__main__.py",
            "monitor/__main__.py",
            "reasoning/__main__.py",
        ]


class TestSharedHelpers:
    def test_exported(self):
        assert "add_env_file" in cli_support.__all__
        assert "apply_env_file" in cli_support.__all__

    def test_add_env_file_defines_the_flag_with_a_none_default(self):
        parser = argparse.ArgumentParser()
        cli_support.add_env_file(parser)
        assert parser.parse_args([]).env_file is None
        assert parser.parse_args(["--env-file", "x.env"]).env_file == "x.env"

    def test_apply_is_a_no_op_without_the_flag(self, monkeypatch):
        monkeypatch.setattr(
            cli_support, "load_env", lambda *a, **k: pytest.fail("called")
        )
        cli_support.apply_env_file(argparse.Namespace(env_file=None))
        cli_support.apply_env_file(argparse.Namespace())  # no such attribute

    def test_apply_loads_the_given_path_and_never_overrides(
        self, monkeypatch, tmp_path
    ):
        env_file = tmp_path / "my.env"
        env_file.write_text("LLM_MODEL=from-file\n", encoding="utf-8")
        monkeypatch.setenv("LLM_MODEL", "from-process")
        cli_support.apply_env_file(argparse.Namespace(env_file=str(env_file)))
        assert os.environ["LLM_MODEL"] == "from-process"

    def test_apply_raises_file_not_found_for_a_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            cli_support.apply_env_file(
                argparse.Namespace(env_file=str(tmp_path / "no.env"))
            )
