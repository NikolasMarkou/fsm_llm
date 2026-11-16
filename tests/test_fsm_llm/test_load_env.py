"""``fsm_llm.load_env`` and the ``fsm-llm`` flags ``--env-file`` and ``--model``.

Plan d454fd62 step 17 (D-005). Nothing reads a ``.env`` file unasked any more
(step 16); ``load_env`` is the one explicit way in and ``--env-file PATH`` is
its CLI form. Precedence everywhere: argument, then the process environment,
then the explicit file, then ``ModelNotConfiguredError``.

The child-process tests run in a temporary directory with a scrubbed
environment whose model-server variables point at a closed port, so none of
them can reach a real model server; the in-process tests replace ``API``.
No test prints the content of a ``.env`` file.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import fsm_llm
from fsm_llm.core.definitions import ModelNotConfiguredError
from tests.test_fsm_llm.support import _child_env

_SRC = Path(fsm_llm.__file__).resolve().parent
_MESSAGE = ModelNotConfiguredError.DEFAULT_MESSAGE


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """An empty working directory and no ``LLM_*`` variable in the process."""
    for name in ("LLM_MODEL", "LLM_TEMPERATURE", "LLM_MAX_TOKENS", "LOAD_ENV_PROBE"):
        monkeypatch.delenv(name, raising=False)
    # Variables a test makes ``load_env`` set: record "unset" as the value to
    # restore, so a later ``monkeypatch.setenv`` cannot restore a loaded value.
    for name in ("LOAD_ENV_A", "LOAD_ENV_B", "LOAD_ENV_KEEP", "LOAD_ENV_NOWHERE"):
        monkeypatch.setenv(name, "x")
        monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    return tmp_path


# --------------------------------------------------------------------------
# load_env
# --------------------------------------------------------------------------


class TestLoadEnv:
    def test_exported_from_the_package(self):
        assert "load_env" in fsm_llm.__all__
        assert fsm_llm.load_env.__module__ == "fsm_llm.env"

    def test_explicit_path_sets_missing_keys_and_returns_the_path(self, clean_env):
        env_file = clean_env / "my.env"
        env_file.write_text("LLM_MODEL=from-file\nLOAD_ENV_PROBE=1\n", encoding="utf-8")
        assert fsm_llm.load_env(env_file) == env_file
        assert os.environ["LLM_MODEL"] == "from-file"
        assert os.environ["LOAD_ENV_PROBE"] == "1"

    def test_a_string_path_is_accepted(self, clean_env):
        env_file = clean_env / "my.env"
        env_file.write_text("LLM_MODEL=from-file\n", encoding="utf-8")
        assert fsm_llm.load_env(str(env_file)) == env_file
        assert os.environ["LLM_MODEL"] == "from-file"

    def test_a_set_variable_is_never_overridden(self, clean_env, monkeypatch):
        env_file = clean_env / "my.env"
        env_file.write_text("LLM_MODEL=from-file\n", encoding="utf-8")
        monkeypatch.setenv("LLM_MODEL", "from-process")
        fsm_llm.load_env(env_file)
        assert os.environ["LLM_MODEL"] == "from-process"

    def test_override_true_replaces_a_set_variable(self, clean_env, monkeypatch):
        env_file = clean_env / "my.env"
        env_file.write_text("LLM_MODEL=from-file\n", encoding="utf-8")
        monkeypatch.setenv("LLM_MODEL", "from-process")
        fsm_llm.load_env(env_file, override=True)
        assert os.environ["LLM_MODEL"] == "from-file"

    @pytest.mark.parametrize("blank", ["", "   ", "\t"])
    def test_a_blank_variable_counts_as_unset_and_the_file_fills_it(
        self, clean_env, monkeypatch, blank
    ):
        """Review gate 1, concern 8: ``resolve_model`` treats a blank ``LLM_MODEL``
        as unset, so ``load_env`` must not let one block the file."""
        env_file = clean_env / "my.env"
        env_file.write_text("LLM_MODEL=from-file\nLOAD_ENV_PROBE=1\n", encoding="utf-8")
        monkeypatch.setenv("LLM_MODEL", blank)
        monkeypatch.setenv("LOAD_ENV_PROBE", blank)
        fsm_llm.load_env(env_file)
        assert os.environ["LLM_MODEL"] == "from-file"
        assert os.environ["LOAD_ENV_PROBE"] == "1"

    def test_a_blank_variable_stays_when_the_file_gives_it_no_value(
        self, clean_env, monkeypatch
    ):
        env_file = clean_env / "my.env"
        env_file.write_text(
            "LLM_MODEL=\nLOAD_ENV_PROBE\nOTHER_PROBE=1\n", encoding="utf-8"
        )
        monkeypatch.setenv("LLM_MODEL", "")
        monkeypatch.setenv("LOAD_ENV_PROBE", " ")
        monkeypatch.delenv("OTHER_PROBE", raising=False)
        fsm_llm.load_env(env_file)
        assert os.environ["LLM_MODEL"] == ""
        assert os.environ["LOAD_ENV_PROBE"] == " "
        assert os.environ["OTHER_PROBE"] == "1"

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_a_blank_variable_resolves_an_interpolated_entry_like_an_unset_one(
        self, clean_env, monkeypatch, blank
    ):
        """Re-review of the gate 1 fixes, concern 1: ``B=${A}`` resolves with the
        PROCESS value of ``A`` first. A blank ``B`` must get what an unset ``B``
        gets, never a value built from the file's own ``A`` that the process
        overrode."""
        env_file = clean_env / "my.env"
        env_file.write_text(
            "LOAD_ENV_A=fromfile\nLOAD_ENV_B=${LOAD_ENV_A}\n", encoding="utf-8"
        )
        monkeypatch.setenv("LOAD_ENV_A", "proc")
        monkeypatch.delenv("LOAD_ENV_B", raising=False)
        fsm_llm.load_env(env_file)
        unset_result = os.environ["LOAD_ENV_B"]
        assert unset_result == "proc"

        monkeypatch.setenv("LOAD_ENV_B", blank)
        fsm_llm.load_env(env_file)
        assert os.environ["LOAD_ENV_B"] == unset_result
        assert os.environ["LOAD_ENV_A"] == "proc"

    def test_a_blank_variable_the_file_resolves_to_blank_stays_as_it_is(
        self, clean_env, monkeypatch
    ):
        env_file = clean_env / "my.env"
        env_file.write_text("LOAD_ENV_B=${LOAD_ENV_NOWHERE}\n", encoding="utf-8")
        monkeypatch.delenv("LOAD_ENV_NOWHERE", raising=False)
        monkeypatch.setenv("LOAD_ENV_B", "   ")
        fsm_llm.load_env(env_file)
        assert os.environ["LOAD_ENV_B"] == "   "

    def test_a_blank_variable_the_file_does_not_name_stays_as_it_is(
        self, clean_env, monkeypatch
    ):
        env_file = clean_env / "my.env"
        env_file.write_text("LOAD_ENV_PROBE=1\n", encoding="utf-8")
        monkeypatch.setenv("LOAD_ENV_KEEP", "")
        fsm_llm.load_env(env_file)
        assert os.environ["LOAD_ENV_KEEP"] == ""
        assert os.environ["LOAD_ENV_PROBE"] == "1"

    def test_override_true_is_unchanged_for_a_blank_variable(
        self, clean_env, monkeypatch
    ):
        env_file = clean_env / "my.env"
        env_file.write_text("LOAD_ENV_PROBE=fromfile\n", encoding="utf-8")
        monkeypatch.setenv("LOAD_ENV_PROBE", "")
        fsm_llm.load_env(env_file, override=True)
        assert os.environ["LOAD_ENV_PROBE"] == "fromfile"

    def test_a_non_blank_variable_still_wins_over_the_file(
        self, clean_env, monkeypatch
    ):
        env_file = clean_env / "my.env"
        env_file.write_text("LLM_MODEL=from-file\n", encoding="utf-8")
        monkeypatch.setenv("LLM_MODEL", " keep ")
        fsm_llm.load_env(env_file)
        assert os.environ["LLM_MODEL"] == " keep "

    def test_override_is_keyword_only(self, clean_env):
        env_file = clean_env / "my.env"
        env_file.write_text("LLM_MODEL=from-file\n", encoding="utf-8")
        with pytest.raises(TypeError):
            fsm_llm.load_env(env_file, True)  # type: ignore[misc]

    def test_a_missing_explicit_path_raises_file_not_found(self, clean_env):
        with pytest.raises(FileNotFoundError, match=r"nope\.env"):
            fsm_llm.load_env(clean_env / "nope.env")
        assert "LLM_MODEL" not in os.environ

    def test_a_directory_is_not_a_file(self, clean_env):
        with pytest.raises(FileNotFoundError):
            fsm_llm.load_env(clean_env)

    def test_no_path_searches_the_working_directory(self, clean_env):
        (clean_env / ".env").write_text("LLM_MODEL=from-cwd\n", encoding="utf-8")
        found = fsm_llm.load_env()
        assert found is not None and found.resolve() == (clean_env / ".env").resolve()
        assert os.environ["LLM_MODEL"] == "from-cwd"

    def test_no_path_searches_upward_from_the_working_directory(self, clean_env):
        (clean_env / ".env").write_text("LLM_MODEL=from-parent\n", encoding="utf-8")
        child = clean_env / "a" / "b"
        child.mkdir(parents=True)
        os.chdir(child)
        found = fsm_llm.load_env()
        assert found is not None and found.resolve() == (clean_env / ".env").resolve()
        assert os.environ["LLM_MODEL"] == "from-parent"

    def test_no_file_found_returns_none_and_sets_nothing(self, clean_env, monkeypatch):
        # Search from a directory the repository `.env` cannot be above.
        monkeypatch.setattr("dotenv.find_dotenv", lambda *a, **k: "")
        assert fsm_llm.load_env() is None
        assert "LLM_MODEL" not in os.environ

    def test_the_search_never_starts_at_the_package_location(self, clean_env):
        # The repository root (a parent of this package in a checkout) holds a
        # git-ignored `.env` for development; a working directory without one
        # must not pick it up. Run from a directory outside the repository.
        probe = (
            "import os, fsm_llm; "
            "r = fsm_llm.load_env(); "
            "print(repr(os.environ.get('LLM_MODEL')), r is None)"
        )
        proc = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=clean_env,
            env=_child_env(),
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "None True"

    def test_the_file_content_is_never_logged_or_echoed(
        self, clean_env, monkeypatch, capsys
    ):
        env_file = clean_env / "my.env"
        env_file.write_text("OPENAI_API_KEY=sk-test-secret-value\n", encoding="utf-8")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        try:
            fsm_llm.load_env(env_file)
            captured = capsys.readouterr()
            assert "sk-test-secret-value" not in captured.out + captured.err
        finally:
            os.environ.pop("OPENAI_API_KEY", None)


class TestOneDotenvSeam:
    def test_dotenv_is_imported_only_by_the_env_module(self):
        offenders = []
        for path in sorted(_SRC.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                if any(n == "dotenv" or n.startswith("dotenv.") for n in names):
                    offenders.append(path.relative_to(_SRC).as_posix())
        assert offenders == ["env.py"]

    def test_the_runner_no_longer_exposes_dotenv(self):
        from fsm_llm import runner

        assert not hasattr(runner, "dotenv")

    def test_the_env_module_imports_only_stdlib_dotenv_and_constants(self):
        tree = ast.parse((_SRC / "env.py").read_text(encoding="utf-8"))
        local: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level:
                local.add(node.module or "")
        assert local <= {"constants"}


# --------------------------------------------------------------------------
# runner.main: --model and --env-file, in process
# --------------------------------------------------------------------------


def _run_main(**kwargs):
    """Call ``runner.main`` with a fake ``API``; return ``(result, from_file)``."""
    api = MagicMock()
    api.start_conversation.return_value = ("conv-1", "Hello!")
    api.has_conversation_ended.return_value = True
    api.get_data.return_value = {}
    with (
        patch("fsm_llm.runner.API.from_file", return_value=api) as from_file,
        patch("fsm_llm.runner.setup_file_logging"),
    ):
        from fsm_llm.runner import main

        result = main("/tmp/t.json", 5, 1000, **kwargs)
    return result, from_file


class TestRunnerModelAndEnvFile:
    def test_without_a_model_it_raises_the_shared_error(self, clean_env):
        with pytest.raises(ModelNotConfiguredError):
            _run_main()

    def test_the_model_argument_is_used(self, clean_env):
        result, from_file = _run_main(model="arg-model")
        assert result == 0
        assert from_file.call_args.kwargs["model"] == "arg-model"

    def test_the_model_argument_beats_the_environment(self, clean_env, monkeypatch):
        monkeypatch.setenv("LLM_MODEL", "env-model")
        _, from_file = _run_main(model="arg-model")
        assert from_file.call_args.kwargs["model"] == "arg-model"

    def test_a_blank_model_argument_falls_through_to_the_environment(
        self, clean_env, monkeypatch
    ):
        monkeypatch.setenv("LLM_MODEL", "env-model")
        _, from_file = _run_main(model="  ")
        assert from_file.call_args.kwargs["model"] == "env-model"

    def test_the_env_file_supplies_the_model(self, clean_env):
        env_file = clean_env / "my.env"
        env_file.write_text("LLM_MODEL=file-model\n", encoding="utf-8")
        _, from_file = _run_main(env_file=str(env_file))
        assert from_file.call_args.kwargs["model"] == "file-model"

    def test_the_process_environment_beats_the_env_file(self, clean_env, monkeypatch):
        env_file = clean_env / "my.env"
        env_file.write_text("LLM_MODEL=file-model\n", encoding="utf-8")
        monkeypatch.setenv("LLM_MODEL", "env-model")
        _, from_file = _run_main(env_file=str(env_file))
        assert from_file.call_args.kwargs["model"] == "env-model"

    def test_a_blank_process_model_does_not_block_the_env_file(
        self, clean_env, monkeypatch
    ):
        env_file = clean_env / "my.env"
        env_file.write_text("LLM_MODEL=file-model\n", encoding="utf-8")
        monkeypatch.setenv("LLM_MODEL", "")
        _, from_file = _run_main(env_file=str(env_file))
        assert from_file.call_args.kwargs["model"] == "file-model"

    def test_the_model_argument_beats_the_env_file(self, clean_env):
        env_file = clean_env / "my.env"
        env_file.write_text("LLM_MODEL=file-model\n", encoding="utf-8")
        _, from_file = _run_main(model="arg-model", env_file=str(env_file))
        assert from_file.call_args.kwargs["model"] == "arg-model"

    def test_the_env_file_also_supplies_temperature_and_max_tokens(self, clean_env):
        env_file = clean_env / "my.env"
        env_file.write_text(
            "LLM_MODEL=file-model\nLLM_TEMPERATURE=0.25\nLLM_MAX_TOKENS=77\n",
            encoding="utf-8",
        )
        _, from_file = _run_main(env_file=str(env_file))
        assert from_file.call_args.kwargs["temperature"] == 0.25
        assert from_file.call_args.kwargs["max_tokens"] == 77

    def test_a_missing_env_file_raises_file_not_found(self, clean_env):
        with pytest.raises(FileNotFoundError):
            _run_main(env_file=str(clean_env / "nope.env"))

    def test_nothing_is_read_unasked(self, clean_env):
        """A `.env` in the working directory is NOT loaded without --env-file."""
        (clean_env / ".env").write_text("LLM_MODEL=stale\n", encoding="utf-8")
        with pytest.raises(ModelNotConfiguredError):
            _run_main()
        assert "LLM_MODEL" not in os.environ

    def test_load_env_is_not_called_without_an_env_file(self, clean_env):
        with patch("fsm_llm.runner.load_env") as loader:
            _run_main(model="arg-model")
        loader.assert_not_called()

    def test_load_env_is_called_with_the_given_path(self, clean_env):
        with patch("fsm_llm.runner.load_env") as loader:
            _run_main(model="arg-model", env_file="some.env")
        loader.assert_called_once_with("some.env")


# --------------------------------------------------------------------------
# The fsm-llm command, in a child process
# --------------------------------------------------------------------------


def _cli(args: list[str], cwd: Path, **extra_env: str):
    return subprocess.run(
        [sys.executable, "-m", "fsm_llm", *args],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=cwd,
        env=_child_env(**extra_env),
        check=False,
    )


def _stderr_lines(proc: subprocess.CompletedProcess[str]) -> list[str]:
    return [line for line in proc.stderr.splitlines() if line.strip()]


class TestFsmLlmCommand:
    def test_unset_model_exits_1_with_one_line_and_no_traceback(self, tmp_path):
        proc = _cli(["--fsm", str(tmp_path / "f.json")], tmp_path)
        assert proc.returncode == 1, proc.stderr
        assert "Traceback" not in proc.stderr
        lines = _stderr_lines(proc)
        assert len(lines) == 1, proc.stderr
        assert _MESSAGE in lines[0]

    def test_a_cwd_dotenv_is_not_read(self, tmp_path):
        (tmp_path / ".env").write_text("LLM_MODEL=stale\n", encoding="utf-8")
        proc = _cli(["--fsm", str(tmp_path / "f.json")], tmp_path)
        assert proc.returncode == 1
        assert _MESSAGE in proc.stderr

    def test_the_env_file_supplies_the_model(self, tmp_path):
        """With the model supplied by the file the run goes on to the FSM file,
        which does not exist: the one error line names it, not the model."""
        env_file = tmp_path / "my.env"
        env_file.write_text("LLM_MODEL=ollama_chat/never-called\n", encoding="utf-8")
        proc = _cli(
            ["--fsm", str(tmp_path / "missing.json"), "--env-file", str(env_file)],
            tmp_path,
        )
        assert proc.returncode == 1, proc.stderr
        assert "Traceback" not in proc.stderr
        assert _MESSAGE not in proc.stderr
        assert "missing.json" in proc.stderr

    def test_the_model_flag_supplies_the_model(self, tmp_path):
        proc = _cli(
            [
                "--fsm",
                str(tmp_path / "missing.json"),
                "--model",
                "ollama_chat/never-called",
            ],
            tmp_path,
        )
        assert proc.returncode == 1, proc.stderr
        assert _MESSAGE not in proc.stderr
        assert "missing.json" in proc.stderr

    def test_the_short_model_flag_exists(self, tmp_path):
        proc = _cli(
            ["--fsm", str(tmp_path / "missing.json"), "-m", "ollama_chat/never-called"],
            tmp_path,
        )
        assert proc.returncode == 1, proc.stderr
        assert _MESSAGE not in proc.stderr

    def test_a_blank_shell_model_does_not_block_the_env_file(self, tmp_path):
        """``LLM_MODEL=""`` in the shell plus ``--env-file`` naming a model: the
        run reaches the missing FSM file instead of 'No LLM model is set'."""
        env_file = tmp_path / "my.env"
        env_file.write_text("LLM_MODEL=ollama_chat/never-called\n", encoding="utf-8")
        proc = _cli(
            ["--fsm", str(tmp_path / "missing.json"), "--env-file", str(env_file)],
            tmp_path,
            LLM_MODEL="",
        )
        assert proc.returncode == 1, proc.stderr
        assert _MESSAGE not in proc.stderr
        assert "missing.json" in proc.stderr

    def test_a_missing_env_file_exits_1_with_one_line(self, tmp_path):
        proc = _cli(
            ["--fsm", str(tmp_path / "f.json"), "--env-file", str(tmp_path / "no.env")],
            tmp_path,
        )
        assert proc.returncode == 1, proc.stderr
        assert "Traceback" not in proc.stderr
        lines = _stderr_lines(proc)
        assert len(lines) == 1, proc.stderr
        assert "no.env" in lines[0]

    def test_the_process_environment_beats_the_env_file(self, tmp_path):
        """The file names a model; the process names another. The run reaches
        the missing FSM file either way, so ask the child directly."""
        probe = (
            "import os, sys; from fsm_llm import load_env; "
            "load_env(sys.argv[1]); print(os.environ['LLM_MODEL'])"
        )
        env_file = tmp_path / "my.env"
        env_file.write_text("LLM_MODEL=file-model\n", encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, "-c", probe, str(env_file)],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=tmp_path,
            env=_child_env(LLM_MODEL="process-model"),
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "process-model"

    def test_help_lists_the_new_flags(self, tmp_path):
        proc = _cli(["--help"], tmp_path)
        assert proc.returncode == 0, proc.stderr
        assert "--env-file" in proc.stdout
        assert "--model" in proc.stdout
        assert "-m" in proc.stdout

    def test_help_says_nothing_about_an_implicit_dotenv(self, tmp_path):
        proc = _cli(["--help"], tmp_path)
        text = proc.stdout.lower()
        assert "loads .env" not in text
        assert "automatically" not in text
