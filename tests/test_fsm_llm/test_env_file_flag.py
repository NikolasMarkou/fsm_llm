"""``--env-file`` on every CLI that needs a model.

Plan d454fd62 step 18 (D-005), reworked in step 18.1. ``fsm-llm`` got the flag
in step 17; here ``fsm-llm-eval``, ``fsm-llm-meta``, ``python -m
fsm_llm.reasoning`` and ``fsm-llm-monitor`` get the same one through
``cli_support.add_env_file`` and ``apply_env_file``. The CLIs that call no model
(``fsm-llm-okf``, ``fsm-llm-skills``, ``fsm-llm-validate``,
``fsm-llm-visualize``) must not have it. Precedence: ``--model`` flag, process
environment, file, then one stderr line and exit 1.

Almost every check runs in process: the CLI's own ``main`` is called with
``capsys`` and ``SystemExit`` is caught (argparse ``--help`` and usage errors,
the monitor's ``sys.exit``). The "file supplies the model" checks replace the
first step after model resolution (an engine class, a run directory, an agent
class, the uvicorn server) with a probe, so the real argument parsing,
``--env-file`` handling and model resolution run and no model is ever called.
An autouse fixture gives each test an empty ``LLM_MODEL``, a temporary working
directory and a restored ``os.environ``.

Six child interpreters remain (``TestRealProcesses``), one real end to end per
CLI family through the module entry point (the wiring an in-process call
cannot see): reasoning, eval, meta and monitor each refuse a bad ``--env-file``
or a missing model as one line and exit 1, and two probes prove the file
supplies the model and the process environment beats it. Every child runs in a
temporary directory with a scrubbed environment whose model-server variables
point at a closed port, with a generous timeout (a loaded full-suite run slows
imports). No test prints the content of a ``.env`` file.
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
from fsm_llm.core.definitions import ModelNotConfiguredError

_REPO = Path(fsm_llm.__file__).resolve().parents[2]
_DATASET = _REPO / "evaluation" / "datasets" / "simple_greeting_cases.json"
_MESSAGE = ModelNotConfiguredError.DEFAULT_MESSAGE
_NEEDS_A_MODEL = ("reasoning", "eval", "meta")
_ALL = (*_NEEDS_A_MODEL, "monitor")


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    """No model, a closed model server, an empty working directory, and the
    environment put back afterwards (``apply_env_file`` writes ``os.environ``)."""
    saved = dict(os.environ)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setenv("OLLAMA_API_BASE", "http://127.0.0.1:9")
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")
    monkeypatch.chdir(tmp_path)
    yield
    os.environ.clear()
    os.environ.update(saved)


def _write_env(tmp_path: Path, text: str = "LLM_MODEL=from-file\n") -> Path:
    env_file = tmp_path / "my.env"
    env_file.write_text(text, encoding="utf-8")
    return env_file


class _Probe:
    """A CLI main replaced after model resolution: prints ``MODEL=<name>`` and
    stops the process with exit 0 (a ``SystemExit``)."""

    def __init__(self, resolved: str) -> None:
        print("MODEL=" + resolved)
        raise SystemExit(0)


def _install_probe(cli: str, monkeypatch) -> None:
    if cli == "reasoning":
        import fsm_llm.reasoning.__main__ as mod
        from fsm_llm.llm import resolve_model

        class Engine:
            def __init__(self, model=None, **kwargs):
                _Probe(resolve_model(model))

        monkeypatch.setattr(mod, "ReasoningEngine", Engine)
    elif cli == "eval":
        import fsm_llm.eval.__main__ as mod

        def create_run_dir(config, model, dataset):
            _Probe(model)

        monkeypatch.setattr(mod, "create_run_dir", create_run_dir)
    elif cli == "meta":
        import fsm_llm.agents.meta_cli as mod

        class Agent:
            def __init__(self, config=None, **kwargs):
                _Probe(config.model)

        monkeypatch.setattr(mod, "MetaBuilderAgent", Agent)
    else:
        import uvicorn

        import fsm_llm.logging as logging_mod
        import fsm_llm.monitor.server as server

        class Server:
            started = True

            def __init__(self, config):
                pass

            def run(self):
                _Probe(os.environ.get("LLM_MODEL", "<unset>"))

        monkeypatch.setattr(uvicorn, "Server", Server)
        # The real ones switch library logging on and set the server's globals.
        monkeypatch.setattr(server, "configure", lambda **kwargs: None)
        monkeypatch.setattr(logging_mod, "enable_library_logging", lambda: None)


def _call(cli: str, args: list[str], monkeypatch, capsys, *, probe: bool = False):
    """Run one CLI's own ``main`` in process; return ``(exit code, stdout,
    stderr)``. ``SystemExit`` (argparse, ``sys.exit``) is its exit code."""
    # The monitor is always probed: were --env-file ignored, the real uvicorn
    # server would start and block this process instead of failing the test.
    if probe or cli == "monitor":
        _install_probe(cli, monkeypatch)
    try:
        if cli == "reasoning":
            import fsm_llm.reasoning.__main__ as mod

            monkeypatch.setattr(
                sys, "argv", ["python -m fsm_llm.reasoning", "a problem", *args]
            )
            code = mod.main()
        elif cli == "eval":
            import fsm_llm.eval.__main__ as mod

            code = mod.main_cli(["run", str(_DATASET), *args])
        elif cli == "meta":
            import fsm_llm.agents.meta_cli as mod

            code = mod.main(args)
        else:
            import fsm_llm.monitor.__main__ as mod

            monkeypatch.setattr(sys, "argv", ["fsm-llm-monitor", "--no-browser", *args])
            code = mod.main_cli()
    except SystemExit as exc:
        code = exc.code
    captured = capsys.readouterr()
    return (0 if code is None else code), captured.out, captured.err


def _lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip()]


def _model_line(out: str) -> str:
    found = [x for x in out.splitlines() if x.startswith("MODEL=")]
    assert len(found) == 1, out
    return found[0]


@pytest.mark.parametrize("cli", _ALL)
class TestEnvFileFlagOnEachCli:
    def test_help_lists_the_flag(self, cli, monkeypatch, capsys):
        code, out, err = _call(cli, ["--help"], monkeypatch, capsys)
        assert code == 0, err
        assert "--env-file" in out

    def test_a_missing_file_exits_1_with_one_line_and_no_traceback(
        self, cli, tmp_path, monkeypatch, capsys
    ):
        code, _, err = _call(
            cli, ["--env-file", str(tmp_path / "no.env")], monkeypatch, capsys
        )
        assert code == 1, err
        assert "Traceback" not in err
        assert len(_lines(err)) == 1, err
        assert "no.env" in err

    def test_a_directory_is_not_a_file(self, cli, tmp_path, monkeypatch, capsys):
        code, _, err = _call(cli, ["--env-file", str(tmp_path)], monkeypatch, capsys)
        assert code == 1, err
        assert "Traceback" not in err
        assert len(_lines(err)) == 1, err


@pytest.mark.parametrize("cli", _NEEDS_A_MODEL)
class TestModelFromTheFile:
    def test_unset_model_exits_1_with_one_line_and_no_traceback(
        self, cli, monkeypatch, capsys
    ):
        code, _, err = _call(cli, [], monkeypatch, capsys)
        assert code == 1, err
        assert "Traceback" not in err
        assert len(_lines(err)) == 1, err
        assert _MESSAGE in err

    def test_a_file_without_a_model_still_exits_1_with_one_line(
        self, cli, tmp_path, monkeypatch, capsys
    ):
        env_file = _write_env(tmp_path, "SOME_OTHER_VARIABLE=1\n")
        code, _, err = _call(cli, ["--env-file", str(env_file)], monkeypatch, capsys)
        assert code == 1, err
        assert "Traceback" not in err
        assert len(_lines(err)) == 1, err
        assert _MESSAGE in err

    def test_a_cwd_dotenv_is_not_read(self, cli, tmp_path, monkeypatch, capsys):
        (tmp_path / ".env").write_text("LLM_MODEL=stale\n", encoding="utf-8")
        code, _, err = _call(cli, [], monkeypatch, capsys)
        assert code == 1, err
        assert _MESSAGE in err

    def test_the_file_supplies_the_model(self, cli, tmp_path, monkeypatch, capsys):
        env_file = _write_env(tmp_path)
        code, out, err = _call(
            cli, ["--env-file", str(env_file)], monkeypatch, capsys, probe=True
        )
        assert code == 0, err
        assert _model_line(out) == "MODEL=from-file"

    def test_the_process_environment_beats_the_file(
        self, cli, tmp_path, monkeypatch, capsys
    ):
        env_file = _write_env(tmp_path)
        monkeypatch.setenv("LLM_MODEL", "from-process")
        code, out, err = _call(
            cli, ["--env-file", str(env_file)], monkeypatch, capsys, probe=True
        )
        assert code == 0, err
        assert _model_line(out) == "MODEL=from-process"

    @pytest.mark.parametrize("blank", ["", "  "])
    def test_a_blank_process_model_does_not_block_the_file(
        self, cli, tmp_path, monkeypatch, capsys, blank
    ):
        env_file = _write_env(tmp_path)
        monkeypatch.setenv("LLM_MODEL", blank)
        code, out, err = _call(
            cli, ["--env-file", str(env_file)], monkeypatch, capsys, probe=True
        )
        assert code == 0, err
        assert _model_line(out) == "MODEL=from-file"

    def test_the_model_flag_beats_the_file(self, cli, tmp_path, monkeypatch, capsys):
        env_file = _write_env(tmp_path)
        code, out, err = _call(
            cli,
            ["--env-file", str(env_file), "--model", "from-flag"],
            monkeypatch,
            capsys,
            probe=True,
        )
        assert code == 0, err
        assert _model_line(out) == "MODEL=from-flag"


class TestMonitorFile:
    """The dashboard needs no model to start: it exports the file's variables
    so a later launch request finds ``LLM_MODEL`` (400 only when none is set)."""

    def test_starts_without_a_model_and_without_the_flag(self, monkeypatch, capsys):
        code, out, err = _call("monitor", [], monkeypatch, capsys, probe=True)
        assert code == 0, err
        assert _model_line(out) == "MODEL=<unset>"

    def test_the_file_reaches_the_environment(self, tmp_path, monkeypatch, capsys):
        env_file = _write_env(tmp_path)
        code, out, err = _call(
            "monitor", ["--env-file", str(env_file)], monkeypatch, capsys, probe=True
        )
        assert code == 0, err
        assert _model_line(out) == "MODEL=from-file"

    def test_the_process_environment_beats_the_file(
        self, tmp_path, monkeypatch, capsys
    ):
        env_file = _write_env(tmp_path)
        monkeypatch.setenv("LLM_MODEL", "from-process")
        _, out, _ = _call(
            "monitor", ["--env-file", str(env_file)], monkeypatch, capsys, probe=True
        )
        assert _model_line(out) == "MODEL=from-process"

    def test_a_missing_file_is_refused_before_the_server_starts(
        self, tmp_path, monkeypatch, capsys
    ):
        code, out, err = _call(
            "monitor",
            ["--env-file", str(tmp_path / "no.env")],
            monkeypatch,
            capsys,
            probe=True,
        )
        assert code == 1, err
        assert "MODEL=" not in out
        assert len(_lines(err)) == 1, err


class TestEvalUsageStillExits1:
    """AGENTS.md: ``fsm-llm-eval`` exits 1 for a usage error, 2 only below
    ``--fail-under``; the new flag must not change that."""

    def test_a_flag_without_its_value_exits_1(self, monkeypatch, capsys):
        code, _, _ = _call("eval", ["--env-file"], monkeypatch, capsys)
        assert code == 1

    def test_the_examples_command_takes_the_flag_too(
        self, tmp_path, monkeypatch, capsys
    ):
        import fsm_llm.eval.__main__ as mod

        code = mod.main_cli(["examples", "--env-file", str(tmp_path / "no.env")])
        err = capsys.readouterr().err
        assert code == 1, err
        assert "Traceback" not in err
        assert len(_lines(err)) == 1, err


class TestNoModelClisHaveNoFlag:
    """``--help`` of the four CLIs that call no model never lists the flag."""

    @staticmethod
    def _help(module: str, argv: list[str], monkeypatch, capsys) -> str:
        import fsm_llm.agent_skills_cli as skills
        import fsm_llm.core.verify.cli as validator
        import fsm_llm.core.visualize.cli as visualizer
        import fsm_llm.okf_cli as okf

        with pytest.raises(SystemExit) as stop:
            if module == "okf_cli":
                okf.main(argv)
            elif module == "agent_skills_cli":
                skills.main(argv)
            else:
                monkeypatch.setattr(sys, "argv", [module, *argv])
                (validator if module == "validator" else visualizer).main_cli()
        assert stop.value.code == 0
        return capsys.readouterr().out

    @pytest.mark.parametrize(
        "module", ("okf_cli", "agent_skills_cli", "validator", "visualizer")
    )
    def test_help_has_no_env_file(self, module, monkeypatch, capsys):
        out = self._help(module, ["--help"], monkeypatch, capsys)
        assert "usage" in out.lower()
        assert "--env-file" not in out

    @pytest.mark.parametrize("module", ("okf_cli", "agent_skills_cli"))
    def test_the_okf_and_skills_subcommands_have_none_either(
        self, module, monkeypatch, capsys
    ):
        out = self._help(module, ["validate", "--help"], monkeypatch, capsys)
        assert "usage" in out.lower()
        assert "--env-file" not in out

    def test_only_the_model_clis_import_the_helper(self):
        src = _REPO / "src" / "fsm_llm"
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


# Child-process proofs. Each child is one interpreter start with the rule-13
# environment (model server on a closed port, local model-cost map), a temporary
# working directory and a scrubbed environment; the timeout is long because a
# full-suite run on a loaded machine slows imports a great deal.
_CHILD_TIMEOUT_SECONDS = 180


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
        timeout=_CHILD_TIMEOUT_SECONDS,
        cwd=cwd,
        env=_child_env(**extra_env),
        check=False,
    )


# A probe replaces the first step after model resolution, prints the model the
# CLI resolved as ``MODEL=<name>`` and stops the process with exit 0.
_EVAL_PROBE = """
import sys
import fsm_llm.eval.__main__ as cli

def probe(config, model, dataset):
    print("MODEL=" + model)
    sys.exit(0)

cli.create_run_dir = probe
sys.exit(cli.main_cli(["run", DATASET, "--env-file", ENV_FILE]))
"""

_MONITOR_PROBE = """
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
sys.argv = ["fsm-llm-monitor", "--no-browser", "--env-file", ENV_FILE]
cli.main_cli()
"""

_REAL = {
    "reasoning": ["-m", "fsm_llm.reasoning", "a problem"],
    "eval": ["-m", "fsm_llm.eval", "run", str(_DATASET)],
    "meta": ["-m", "fsm_llm.agents.meta_cli"],
    "monitor": ["-m", "fsm_llm.monitor", "--no-browser"],
}


def _probe_script(body: str, env_file: Path) -> str:
    return (
        f"DATASET = {str(_DATASET)!r}\nENV_FILE = {str(env_file)!r}\n"
        + textwrap.dedent(body)
    )


class TestRealProcesses:
    """Six interpreter starts, through the module entry points."""

    def test_reasoning_with_no_model_is_one_line(self, tmp_path):
        (tmp_path / ".env").write_text("LLM_MODEL=stale\n", encoding="utf-8")
        proc = _run(_REAL["reasoning"], tmp_path)
        assert proc.returncode == 1, proc.stderr
        assert "Traceback" not in proc.stderr
        assert len(_lines(proc.stderr)) == 1, proc.stderr
        assert _MESSAGE in proc.stderr

    def test_eval_with_a_missing_file_is_one_line(self, tmp_path):
        proc = _run([*_REAL["eval"], "--env-file", str(tmp_path / "no.env")], tmp_path)
        assert proc.returncode == 1, proc.stderr
        assert "Traceback" not in proc.stderr
        assert len(_lines(proc.stderr)) == 1, proc.stderr
        assert "no.env" in proc.stderr

    def test_meta_with_a_file_without_a_model_is_one_line(self, tmp_path):
        env_file = _write_env(tmp_path, "SOME_OTHER_VARIABLE=1\n")
        proc = _run([*_REAL["meta"], "--env-file", str(env_file)], tmp_path)
        assert proc.returncode == 1, proc.stderr
        assert "Traceback" not in proc.stderr
        assert len(_lines(proc.stderr)) == 1, proc.stderr
        assert _MESSAGE in proc.stderr

    def test_monitor_with_a_directory_is_one_line(self, tmp_path):
        proc = _run([*_REAL["monitor"], "--env-file", str(tmp_path)], tmp_path)
        assert proc.returncode == 1, proc.stderr
        assert "Traceback" not in proc.stderr
        assert len(_lines(proc.stderr)) == 1, proc.stderr

    def test_eval_file_supplies_the_model(self, tmp_path):
        env_file = _write_env(tmp_path)
        proc = _run(["-c", _probe_script(_EVAL_PROBE, env_file)], tmp_path)
        assert proc.returncode == 0, proc.stderr
        assert _model_line(proc.stdout) == "MODEL=from-file"

    def test_monitor_process_environment_beats_the_file(self, tmp_path):
        env_file = _write_env(tmp_path)
        proc = _run(
            ["-c", _probe_script(_MONITOR_PROBE, env_file)],
            tmp_path,
            LLM_MODEL="from-process",
        )
        assert proc.returncode == 0, proc.stderr
        assert _model_line(proc.stdout) == "MODEL=from-process"


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
