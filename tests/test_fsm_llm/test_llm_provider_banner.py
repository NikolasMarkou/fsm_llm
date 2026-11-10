"""litellm's "Give Feedback / Get Help" banner stays off stdout (plan 0068b055
step 12.20, D-023).

litellm prints a red banner to stdout on every mapped provider exception while
``litellm.suppress_debug_info`` is False. ``fsm_llm.llm`` (the one module that
imports litellm) sets it once at import.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import litellm

import fsm_llm.llm  # noqa: F401  (the import is the behaviour under test)

_SRC = Path(__file__).resolve().parents[2] / "src"

# A provider call that fails fast and offline: nothing listens on 127.0.0.1:1.
_FAILING_CALL = """
from fsm_llm.definitions import CompletionRequest, LLMResponseError
from fsm_llm.llm import LiteLLMInterface

llm = LiteLLMInterface(
    model="openai/does-not-exist",
    api_key="sk-test",
    api_base="http://127.0.0.1:1/v1",
    timeout=5,
)
try:
    llm.complete(CompletionRequest(messages=[{"role": "user", "content": "hi"}]))
except LLMResponseError:
    print("RAISED")
"""


def _run_child(code: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "LITELLM_MODE": "PRODUCTION"}
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=90,
        env=env,
        check=False,
    )


class TestSuppressDebugInfo:
    def test_importing_the_llm_layer_sets_it(self):
        assert litellm.suppress_debug_info is True

    def test_fresh_interpreter_sees_it_after_import(self):
        proc = _run_child(
            "import litellm\n"
            "assert litellm.suppress_debug_info is False\n"
            "import fsm_llm.llm\n"
            "print(litellm.suppress_debug_info)\n"
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "True"

    def test_only_the_llm_layer_configures_it(self):
        users = sorted(
            str(p.relative_to(_SRC))
            for p in _SRC.rglob("*.py")
            if "suppress_debug_info" in p.read_text(encoding="utf-8")
        )
        assert users == ["llm.py"]


class TestProviderFailurePrintsNoBanner:
    def test_failed_call_keeps_the_banner_off_stdout(self):
        proc = _run_child(_FAILING_CALL)
        assert proc.returncode == 0, proc.stderr
        assert "RAISED" in proc.stdout
        assert "Give Feedback" not in proc.stdout
        assert "LiteLLM.Info" not in proc.stdout

    def test_the_banner_does_print_when_the_setting_is_false(self):
        """Guards the test above: without the setting litellm does print it."""
        proc = _run_child(
            "import litellm\nimport fsm_llm.llm\nlitellm.suppress_debug_info = False\n"
            + _FAILING_CALL
        )
        assert proc.returncode == 0, proc.stderr
        assert "Give Feedback" in proc.stdout
