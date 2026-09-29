"""
Tests for the reasoning CLI's ``--verbose`` logging switch (D-016).

``fsm_llm`` log output is disabled library-wide, so ``--verbose`` must turn it
back on or the engine's records never reach the terminal. Each case runs in a
fresh interpreter: enabling logging flips process-global loguru state, which
must not leak into this test process.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

# The engine is replaced by a stub that logs from an engine-named module, so
# no LLM is called and the probe is subject to the "fsm_llm" disable.
_PROBE = (
    "import sys\n"
    "import fsm_llm.reasoning.__main__ as m\n"
    "from fsm_llm.logging import logger\n"
    "probe = {'__name__': 'fsm_llm.reasoning.engine', 'logger': logger}\n"
    "def _solve(*_args):\n"
    "    exec(\"logger.info('I-PROBE'); logger.debug('D-PROBE')\", probe)\n"
    "    return 'ok', {}\n"
    "m.solve_problem_with_engine = _solve\n"
    "sys.argv = ['fsm-llm-reasoning', 'what is 2+2?', '-o', 'json', *sys.argv[1:]]\n"
    "sys.exit(m.main())\n"
)


class TestVerboseLogging:
    @pytest.mark.parametrize(
        ("flags", "info_lines"),
        [(["--verbose"], 1), ([], 0)],
    )
    def test_verbose_enables_library_logging(
        self, monkeypatch: pytest.MonkeyPatch, flags: list[str], info_lines: int
    ) -> None:
        monkeypatch.delenv("FSM_LLM_LOG_LEVEL", raising=False)
        completed = subprocess.run(
            [sys.executable, "-c", _PROBE, *flags],
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert completed.stderr.count("I-PROBE") == info_lines
        assert "D-PROBE" not in completed.stderr
