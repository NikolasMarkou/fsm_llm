"""The hard per-test timeout is on and cannot vanish silently (plan ace7a8c8, D-060).

`faulthandler_timeout` only dumps stacks. Two CI legs hung to the job limit (45 minutes on
Windows 3.15, 15 minutes then exit 139 on Linux 3.12), so `pytest-timeout` stops a stuck test
with the thread method (the signal method does not exist on Windows). These rows read the
configuration and prove the plugin really stops a hung test in a throwaway project.
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _ini() -> dict:
    data = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return data["tool"]["pytest"]["ini_options"]


class TestTheTimeoutIsConfigured:
    def test_the_plugin_is_loaded_and_reads_the_configured_values(self, request):
        config = request.config
        assert config.pluginmanager.hasplugin("timeout"), (
            "pytest-timeout is not installed"
        )
        ini = _ini()
        assert float(config.getini("timeout")) == float(ini["timeout"]) == 300
        assert config.getini("timeout_method") == ini["timeout_method"] == "thread"

    def test_the_stack_dump_fires_before_the_hard_limit_kills(self):
        # The dump is a diagnostic that kills nothing; it only means something when it fires
        # BEFORE the thread method ends the process (at 900 s it could never fire: the 300 s
        # limit killed first, review note 6 of plan ace7a8c8). It names every thread when a
        # test is slow but not yet killed.
        ini = _ini()
        assert 0 < ini["faulthandler_timeout"] < ini["timeout"]

    def test_the_dev_extra_declares_the_plugin(self):
        data = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        dev = data["project"]["optional-dependencies"]["dev"]
        assert any(line.startswith("pytest-timeout") for line in dev)


class TestTheTimeoutStopsAHungTest:
    def test_a_sleeping_test_is_stopped_with_a_timeout_report(self, tmp_path):
        (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
        (tmp_path / "test_hang.py").write_text(
            "import time\n\ndef test_hang():\n    time.sleep(30)\n", encoding="utf-8"
        )
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "--timeout=1",
                "--timeout-method=thread",
                str(tmp_path),
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=60,
        )
        out = proc.stdout + proc.stderr
        assert proc.returncode != 0
        assert "Timeout" in out, out

    def test_the_stack_dump_comes_before_the_kill_when_it_is_set_lower(self, tmp_path):
        # The configured order (dump at 240 s, kill at 300 s) at a small scale: the built-in
        # dump prints "Timeout (0:00:01)!" with every thread, then the thread method kills.
        (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
        (tmp_path / "test_hang.py").write_text(
            "import time\n\ndef test_hang():\n    time.sleep(30)\n", encoding="utf-8"
        )
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "--timeout=4",
                "--timeout-method=thread",
                "-o",
                "faulthandler_timeout=1",
                str(tmp_path),
            ],
            cwd=tmp_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # one pipe, so the order of the two reports is real
            text=True,
            timeout=60,
        )
        out = proc.stdout
        assert proc.returncode != 0
        assert "Timeout (0:00:01)!" in out, out
        assert out.index("Timeout (0:00:01)!") < out.index("+ Timeout +"), out
