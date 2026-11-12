"""The public API snapshot `docs/api.txt` equals what the code declares now.

`scripts/api_snapshot.py` prints every public name (the static `__all__` of
the seven packages and four core modules, plus `schema_registry`, `codes` and
`flow_analysis`) with its signature. A removed, renamed or retyped public
parameter changes that output, so this test fails until the snapshot is
regenerated (`make api-snapshot`) and reviewed in the same commit.

The generator runs in a child process: a test session may hold monkeypatched
modules, and the snapshot must be the same for any caller.
"""

import difflib
import pathlib
import subprocess
import sys

import pytest

pytest.importorskip("fastapi")  # fsm_llm.monitor needs it, like test_packaging

_REPO_ROOT = pathlib.Path(__file__).parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "api_snapshot.py"
_SNAPSHOT = _REPO_ROOT / "docs" / "api.txt"
_HINT = "run `make api-snapshot` and review `git diff docs/api.txt`"


def _generate() -> str:
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT)],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
    )
    assert proc.returncode == 0, (
        f"api_snapshot.py exited {proc.returncode}: " + (proc.stderr[-2000:])
    )
    return proc.stdout


class TestApiSnapshot:
    def test_snapshot_file_exists(self):
        assert _SNAPSHOT.is_file(), f"{_SNAPSHOT} is missing: {_HINT}"

    def test_snapshot_matches_the_code(self):
        assert _SNAPSHOT.is_file(), f"{_SNAPSHOT} is missing: {_HINT}"
        expected = _SNAPSHOT.read_text(encoding="utf-8")
        actual = _generate()
        if actual == expected:
            return
        diff = list(
            difflib.unified_diff(
                expected.splitlines(),
                actual.splitlines(),
                "docs/api.txt",
                "generated",
                lineterm="",
                n=0,
            )
        )
        shown = "\n".join(line[:300] for line in diff[:40])
        pytest.fail(
            f"the public API differs from docs/api.txt ({len(diff)} diff lines, "
            f"first 40):\n{shown}\n{_HINT}"
        )

    def test_snapshot_shape(self):
        assert _SNAPSHOT.is_file(), f"{_SNAPSHOT} is missing: {_HINT}"
        text = _SNAPSHOT.read_text(encoding="utf-8")
        assert text.endswith("\n") and not text.endswith("\n\n")
        assert "__version__" not in text
        for header in (
            "# fsm_llm",
            "# fsm_llm.schema_registry",
            "# fsm_llm.codes",
            "# fsm_llm.flow_analysis",
            "# fsm_llm.synthesis",
        ):
            assert header in text.splitlines(), header
