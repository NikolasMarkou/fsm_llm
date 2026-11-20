"""`scripts/changelog_check.py`: a smoke test of the CHANGELOG form check.

The script judges the `###` headings, the `[Unreleased]` compare reference and the exact
`**BREAKING:**` marker (the migration record: each breaking entry names the old form and the
new). The wide table of fixture cases and the mutation scenarios of the script's own rules were
dropped (plan ace7a8c8 step 4: they pinned the script's internals, and the script is also run
by `make changelog-check`, the CI lint job and `scripts/gate.sh push`). What stays: the real
`CHANGELOG.md` is clean, the exact marker spelling is enforced, and two refusals.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "changelog_check.py"

_REFS = (
    "[Unreleased]: https://github.com/o/r/compare/v0.12.0...HEAD\n"
    "[0.12.0]: https://github.com/o/r/releases/tag/v0.12.0\n"
)
_RELEASE = "## [0.12.0] - 2026-10-02\n\n### Added\n\n- Old thing.\n\n"


def _doc(unreleased: str, *, refs: str = _REFS) -> str:
    return f"# Changelog\n\n## [Unreleased]\n\n{unreleased}{_RELEASE}{refs}"


def _check(path: Path, *argv: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), "check", "--file", str(path), *argv],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _findings(tmp_path: Path, text: str) -> subprocess.CompletedProcess[str]:
    path = tmp_path / "CHANGELOG.md"
    path.write_text(text, encoding="utf-8")
    return _check(path)


def test_the_real_changelog_is_clean() -> None:
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT), "check"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout == ""


def test_a_well_formed_changelog_with_the_exact_marker_is_clean(tmp_path: Path) -> None:
    text = _doc("### Changed\n\n- **BREAKING:** `old` is now `new`.\n\n")
    proc = _findings(tmp_path, text)
    assert proc.returncode == 0 and proc.stdout == "", proc.stdout


@pytest.mark.parametrize(
    "bullet",
    ["- **Breaking:** x.", "- **breaking:** x.", "- **BREAKING** x."],
)
def test_a_misspelt_breaking_marker_is_refused(tmp_path: Path, bullet: str) -> None:
    proc = _findings(tmp_path, _doc(f"### Changed\n\n{bullet}\n\n"))
    assert proc.returncode == 1 and ": CL005: " in proc.stdout, proc.stdout


def test_a_missing_compare_reference_is_refused(tmp_path: Path) -> None:
    refs = _REFS.split("\n", 1)[1]
    proc = _findings(tmp_path, _doc("### Added\n\n- A new thing.\n\n", refs=refs))
    assert proc.returncode == 1 and ": CL004: " in proc.stdout, proc.stdout


def test_an_unknown_section_heading_is_refused(tmp_path: Path) -> None:
    proc = _findings(tmp_path, _doc("### Notes\n\n- n\n\n"))
    assert proc.returncode == 1 and ": CL010: " in proc.stdout, proc.stdout
