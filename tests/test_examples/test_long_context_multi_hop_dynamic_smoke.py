from __future__ import annotations

"""Live smoke test for examples/long_context/multi_hop_demo (--dynamic)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_PATH = REPO_ROOT / "examples" / "long_context" / "multi_hop_demo" / "run.py"


@pytest.mark.real_llm
@pytest.mark.slow
def test_multi_hop_dynamic_demo_smoke() -> None:
    """Live smoke: --dynamic produces VERIFICATION block with theorem2 PASS.

    Exercises the M5 slice 6 multi_hop_dynamic factory on qwen3.5:4b.
    Asserts the VERIFICATION block reports oracle_calls_match_dynamic_planner
    as True and that exit code is 0.
    """
    env = os.environ.copy()
    env.setdefault("LLM_MODEL", "ollama_chat/qwen3.5:4b")
    result = subprocess.run(
        [sys.executable, str(DEMO_PATH), "--dynamic", "--max-hops", "3"],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    out = result.stdout + result.stderr
    assert result.returncode == 0, f"demo failed:\n{out}"
    assert "oracle_calls_match_dynamic_planner: True" in out, (
        f"strict T2 check missing or failed:\n{out}"
    )
    assert "oracle_calls_within_upper_bound : True" in out, (
        f"upper-bound T2 check missing or failed:\n{out}"
    )
