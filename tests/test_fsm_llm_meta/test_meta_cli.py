"""The ``fsm-llm-meta`` console script (``fsm_llm.agents.meta_cli``).

Output to a terminal that cannot encode the artifact, and a domain error as
one stderr line.
"""

from __future__ import annotations

import io
import json
import sys

import pytest

from fsm_llm.agents import meta_cli
from fsm_llm.agents.definitions import MetaBuilderResult
from fsm_llm.agents.exceptions import MetaBuilderError


def _ascii_stream() -> io.TextIOWrapper:
    """A stream that cannot encode non-ASCII text (an ASCII terminal)."""
    return io.TextIOWrapper(io.BytesIO(), encoding="ascii", write_through=True)


class TestCliSurvivesAnUnencodableStdout:
    """F-AM-02: an artifact named "Café bot" is printed, not lost."""

    def _install(self, monkeypatch, outcome):
        class _FakeAgent:
            def __init__(self, config=None):
                pass

            def run_interactive(self):
                if isinstance(outcome, BaseException):
                    raise outcome
                return outcome

        monkeypatch.setattr(meta_cli, "MetaBuilderAgent", _FakeAgent)
        out, err = _ascii_stream(), _ascii_stream()
        monkeypatch.setattr(sys, "stdout", out)
        monkeypatch.setattr(sys, "stderr", err)
        monkeypatch.setattr(sys, "argv", ["fsm-llm-meta"])
        return out, err

    def test_the_artifact_is_printed_with_escapes(self, monkeypatch):
        artifact = {"name": "Café bot", "initial_state": "s", "states": {}}
        out, _ = self._install(
            monkeypatch,
            MetaBuilderResult(
                artifact=artifact,
                artifact_json=json.dumps(artifact, ensure_ascii=False),
                is_valid=True,
            ),
        )
        with pytest.raises(SystemExit) as exit_info:
            meta_cli.main_cli()
        assert exit_info.value.code == 0
        assert "Caf\\xe9 bot" in out.buffer.getvalue().decode("ascii")

    def test_a_domain_error_is_one_stderr_line_and_exit_1(self, monkeypatch):
        out, err = self._install(monkeypatch, MetaBuilderError("Café failed"))
        with pytest.raises(SystemExit) as exit_info:
            meta_cli.main_cli()
        assert exit_info.value.code == 1
        assert out.buffer.getvalue() == b""
        assert "fsm-llm-meta: error: Caf" in err.buffer.getvalue().decode("ascii")

    def test_a_bad_option_value_is_no_traceback(self, monkeypatch):
        _, err = self._install(monkeypatch, MetaBuilderError("unused"))
        monkeypatch.setattr(sys, "argv", ["fsm-llm-meta", "--temperature", "9"])
        assert meta_cli.main() == 1
        assert "fsm-llm-meta: error:" in err.buffer.getvalue().decode("ascii")
