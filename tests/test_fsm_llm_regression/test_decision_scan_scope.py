"""`decision_index.py --check` fails on an anchor in a tracked file the scanner never reads.

The scanner reads `src`, `tests`, `scripts`, `.github` and four root files; a comment-form anchor
in `docs/` or `examples/` would be judged by the planner and never by CI. `--check` lists the
tracked files it does not read (`outside_scope`) and fails on them. The head is built in pieces so
this file holds no anchor itself.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_fsm_llm_regression.support import REPO

SCRIPT = REPO / "scripts" / "decision_index.py"
HEAD = "<!--" + " DECISION"
TOKEN = "plan-2026-11-10T194139-2cefeb61/D-001"


def _check(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", str(SCRIPT), "--root", str(root), "--check"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_an_anchor_in_a_tracked_unscanned_file_fails_the_check(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "docs").mkdir()
    page = tmp_path / "docs" / "note.md"
    page.write_text("plain prose, no anchor\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "docs/note.md"], check=True)
    assert _check(tmp_path).returncode == 0
    page.write_text(f"{HEAD} {TOKEN} never edit this -->\n", encoding="utf-8")
    done = _check(tmp_path)
    assert done.returncode == 1, done.stdout + done.stderr
    assert "docs/note.md" in done.stderr and TOKEN in done.stderr
