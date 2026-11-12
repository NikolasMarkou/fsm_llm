"""Importing ``fsm_llm`` never loads a ``.env`` file (plan d454fd62, step 16, D-005).

litellm calls ``load_dotenv()`` at import unless ``LITELLM_MODE=PRODUCTION``.
``fsm_llm.llm`` (the one module that imports litellm) sets that mode around its
own ``import litellm`` and restores the environment afterwards, so a stale
``.env`` can no longer fill ``LLM_MODEL`` or a key behind the caller's back.

Every test runs a child interpreter (the import is the behaviour under test)
with a scrubbed environment: the model-server variables point at a closed
port, so nothing here can reach a real model server. No test prints the
content of a ``.env`` file; the repository one is only checked for key
presence.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"

_STALE = "stale-model-from-dotenv"

_PROBE = """
import os
import sys

if os.environ.get("PROBE_LITELLM_FIRST") == "1":
    import litellm  # noqa: F401
import fsm_llm  # noqa: F401
import litellm

print("LLM_MODEL=" + ("SET" if os.environ.get("LLM_MODEL") else "UNSET"))
print("VALUE=" + os.environ.get("LLM_MODEL", ""))
print("MODE_ENV=" + repr(os.environ.get("LITELLM_MODE")))
print("MODE_LITELLM=" + repr(litellm.litellm_mode))
"""


def _child_env(**extra: str) -> dict[str, str]:
    """A scrubbed child environment: no ``LLM_MODEL``, no ``LITELLM_MODE``."""
    keep = ("PATH", "HOME", "SYSTEMROOT", "TMPDIR", "TEMP", "TMP", "LANG", "LC_ALL")
    env = {key: os.environ[key] for key in keep if key in os.environ}
    env["PYTHONIOENCODING"] = "utf-8"
    env["OLLAMA_API_BASE"] = "http://127.0.0.1:9"
    env["OLLAMA_HOST"] = "http://127.0.0.1:9"
    env["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    env.update(extra)
    return env


def _probe(cwd: Path, **extra: str) -> dict[str, str]:
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        timeout=120,
        env=_child_env(**extra),
        cwd=cwd,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    out: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            out[key] = value
    return out


@pytest.fixture
def stale_env_dir(tmp_path: Path) -> Path:
    (tmp_path / ".env").write_text(f"LLM_MODEL={_STALE}\n", encoding="utf-8")
    return tmp_path


class TestImportLoadsNoDotenv:
    def test_cwd_dotenv_leaves_llm_model_unset(self, stale_env_dir: Path):
        got = _probe(stale_env_dir)
        assert got["LLM_MODEL"] == "UNSET"
        assert got["VALUE"] == ""

    def test_repository_dotenv_leaves_llm_model_unset(self):
        """The repo ``.env`` (ignored, may set ``LLM_MODEL``) is not loaded
        either: python-dotenv would find it from the repository root."""
        got = _probe(_REPO)
        assert got["LLM_MODEL"] == "UNSET"

    def test_litellm_reports_production_mode(self, stale_env_dir: Path):
        assert _probe(stale_env_dir)["MODE_LITELLM"] == "'PRODUCTION'"

    def test_litellm_mode_is_not_left_in_the_environment(self, stale_env_dir: Path):
        """A leaked variable would reach child processes (eval examples)."""
        assert _probe(stale_env_dir)["MODE_ENV"] == "None"


class TestDocumentedEscapes:
    def test_explicit_dev_mode_still_loads_dotenv(self, stale_env_dir: Path):
        got = _probe(stale_env_dir, LITELLM_MODE="DEV")
        assert got["VALUE"] == _STALE
        assert got["MODE_ENV"] == "'DEV'"
        assert got["MODE_LITELLM"] == "'DEV'"

    def test_explicit_production_mode_is_kept(self, stale_env_dir: Path):
        got = _probe(stale_env_dir, LITELLM_MODE="PRODUCTION")
        assert got["LLM_MODEL"] == "UNSET"
        assert got["MODE_ENV"] == "'PRODUCTION'"

    def test_importing_litellm_first_still_loads_dotenv(self, stale_env_dir: Path):
        """The documented limit: litellm already read the file before
        ``fsm_llm`` was imported and that cannot be undone."""
        got = _probe(stale_env_dir, PROBE_LITELLM_FIRST="1")
        assert got["VALUE"] == _STALE


class TestSingleSite:
    def test_litellm_mode_is_named_only_in_the_llm_layer(self):
        users = sorted(
            str(p.relative_to(_SRC))
            for p in _SRC.rglob("*.py")
            if "LITELLM_MODE" in p.read_text(encoding="utf-8")
        )
        assert users == ["llm.py"]
