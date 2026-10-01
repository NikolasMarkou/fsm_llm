"""Smoke tests for the fsm_llm.agents CLI entry points.

Covers ``fsm_llm.agents.meta_cli.main_cli`` (the ``fsm-llm-meta`` script)
with ``MetaBuilderAgent`` replaced by a fake, so no LLM is ever called, and
``python -m fsm_llm.agents`` (``fsm_llm.agents.__main__.main``).
"""

from __future__ import annotations

import json

import pytest

from fsm_llm.agents import __main__ as agents_main
from fsm_llm.agents import meta_cli
from fsm_llm.agents.__version__ import __version__
from fsm_llm.agents.definitions import MetaBuilderResult
from fsm_llm.agents.exceptions import MetaBuilderError

_ARTIFACT = {"name": "Bot", "initial_state": "start", "states": {}}


def _valid_result() -> MetaBuilderResult:
    return MetaBuilderResult(
        artifact=_ARTIFACT,
        artifact_json=json.dumps(_ARTIFACT),
        is_valid=True,
    )


def _install_fake_agent(monkeypatch, outcome) -> None:
    """Replace ``MetaBuilderAgent`` in ``meta_cli``.

    ``outcome`` is either a ``MetaBuilderResult`` returned by
    ``run_interactive`` or a ``BaseException`` instance it raises.
    """

    class _FakeAgent:
        def __init__(self, config=None):
            self.config = config

        def run_interactive(self):
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

    monkeypatch.setattr(meta_cli, "MetaBuilderAgent", _FakeAgent)


def _run_meta_cli(monkeypatch, *argv: str) -> None:
    monkeypatch.setattr("sys.argv", ["fsm-llm-meta", *argv])
    meta_cli.main_cli()


class TestMetaCli:
    def test_output_writes_artifact_file(self, monkeypatch, tmp_path, capsys):
        _install_fake_agent(monkeypatch, _valid_result())
        out = tmp_path / "sub" / "artifact.json"

        _run_meta_cli(monkeypatch, "--output", str(out))

        assert out.exists()
        assert json.loads(out.read_text(encoding="utf-8")) == _ARTIFACT
        assert "Artifact saved to" in capsys.readouterr().out

    def test_no_output_prints_artifact_json(self, monkeypatch, capsys):
        _install_fake_agent(monkeypatch, _valid_result())

        _run_meta_cli(monkeypatch)

        assert "Generated artifact JSON:" in capsys.readouterr().out

    def test_meta_builder_error_exits_1(self, monkeypatch, capsys):
        _install_fake_agent(monkeypatch, MetaBuilderError("boom"))

        with pytest.raises(SystemExit) as exc:
            _run_meta_cli(monkeypatch)

        assert exc.value.code == 1
        assert "Error: boom" in capsys.readouterr().out

    def test_keyboard_interrupt_at_the_prompt_exits_130(self, monkeypatch, capsys):
        """Ctrl-C at the ``> `` prompt exits 130 like every project CLI. The
        real ``MetaBuilderAgent.run_interactive`` runs (only its session
        start is stubbed, no LLM call); at f85e07a it swallowed the interrupt
        and the CLI exited 1 with "Artifact has validation errors" (review
        round 3, pass 13 concern 5, D-045)."""
        agent_cls = meta_cli.MetaBuilderAgent
        monkeypatch.setattr(agent_cls, "start", lambda self: "What to build?")

        def interrupted(prompt: str = "") -> str:
            assert prompt == "> "
            raise KeyboardInterrupt

        monkeypatch.setattr("builtins.input", interrupted)

        with pytest.raises(SystemExit) as exc:
            _run_meta_cli(monkeypatch)

        out = capsys.readouterr().out
        assert exc.value.code == 130
        assert "Aborted." in out
        assert "validation errors" not in out

    def test_eof_at_the_prompt_still_ends_the_session(self, monkeypatch, capsys):
        """EOF (Ctrl-D) stays a normal end of the session."""
        agent_cls = meta_cli.MetaBuilderAgent
        monkeypatch.setattr(agent_cls, "start", lambda self: "What to build?")

        def eof(prompt: str = "") -> str:
            raise EOFError

        monkeypatch.setattr("builtins.input", eof)
        result = agent_cls().run_interactive()

        assert "Session ended by user." in capsys.readouterr().out
        assert result.is_valid is False

    def test_invalid_result_exits_1_and_writes_nothing(self, monkeypatch, tmp_path):
        result = _valid_result()
        result.is_valid = False
        result.validation_errors = ["no terminal state"]
        _install_fake_agent(monkeypatch, result)
        out = tmp_path / "artifact.json"

        with pytest.raises(SystemExit) as exc:
            _run_meta_cli(monkeypatch, "--output", str(out))

        assert exc.value.code == 1
        assert not out.exists()


class TestAgentsMain:
    def test_info_exits_0_and_prints_version(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["fsm_llm.agents", "--info"])

        with pytest.raises(SystemExit) as exc:
            agents_main.main()

        assert exc.value.code == 0
        assert f"fsm_llm.agents v{__version__}" in capsys.readouterr().out

    def test_list_tools_is_rejected(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["fsm_llm.agents", "--list-tools"])

        with pytest.raises(SystemExit) as exc:
            agents_main.main()

        assert exc.value.code == 2
        assert "unrecognized arguments: --list-tools" in capsys.readouterr().err
