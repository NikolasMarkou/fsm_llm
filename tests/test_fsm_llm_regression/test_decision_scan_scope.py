"""No decision anchor hides outside the files the anchor gate reads (final REFLECT concern 5).

`scripts/decision_index.py` scans `src`, `tests`, `scripts`, `.github` and four root
files; the planner's `validate-plan.mjs` (`walkSourceFiles`) walks the whole project. A
comment-form anchor (`#`, `//`, `--`, `/* */` or `<!-- -->` followed by the decision head
and a plan-id token) written in `docs/*.md`, `mkdocs.yml`, `.pre-commit-config.yaml`,
`examples/` or a root `.md` file would be checked by the planner and never by CI. The
scope difference is deliberate (see the module docstring of the script) and this test
closes it: it reads EVERY tracked file the scanner does not read, with the scanner's own
grammar (`outside_scope`), and fails with the list.

The fixture heads are built in pieces (`HTML = "<!--" + " DECISION"`): the planner and the
scanner read a head followed by a plan-id-like token in ANY string of a `.py` file.

Not run when `git` is missing or the checkout is not a git repository (an sdist or a
tarball has no `git ls-files`).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

from tests.test_fsm_llm_regression.support import REPO
from tests.test_fsm_llm_regression.test_decision_index import P1, A, _load, _write

HTML = "<!--" + " DECISION"
SLASH = "//" + " DECISION"
DASHES = "--" + " DECISION"
TOKEN = f"{P1}/D-001"

#: Tracked paths the guard does not read: generated or manifest text that QUOTES anchors
#: on purpose (a decision index lists each anchor with its rule, the manifest holds
#: `<plan-id>/D-NNN | date | rationale` lines). A path that does not exist is harmless.
EXCLUDED = ("docs/decisions/", "plans/ANCHORS.md")


@pytest.fixture(scope="module")
def di() -> ModuleType:
    return _load()


def stray_anchors(di: ModuleType, root: Path, tracked: list[str]) -> list[str]:
    """`path: token` for every anchor in a tracked file outside the scanner's reach."""
    kept = [t for t in tracked if not t.startswith(EXCLUDED)]
    return [f"{a.path}: {a.token}" for a in di.outside_scope(root, kept)]


def _git_tracked() -> list[str]:
    if shutil.which("git") is None or not (REPO / ".git").exists():
        pytest.skip("not a git checkout: there is no `git ls-files` to read")
    done = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO,
        capture_output=True,
        timeout=60,
        check=False,
    )
    if done.returncode != 0:
        pytest.skip("`git ls-files` failed: not a usable git checkout")
    return [p for p in done.stdout.decode("utf-8", "replace").split("\0") if p]


class TestTheRealTree:
    def test_no_tracked_file_outside_the_scan_roots_holds_an_anchor(
        self, di: ModuleType
    ) -> None:
        tracked = _git_tracked()
        assert tracked, "git ls-files listed nothing"
        found = stray_anchors(di, REPO, tracked)
        assert not found, (
            "a decision anchor sits where `scripts/decision_index.py` does not read it "
            "(the planner would check it, CI never would). Move it under src, tests, "
            "scripts or .github, or widen SCAN_DIRS/SCAN_FILES on purpose:\n  "
            + "\n  ".join(found)
        )

    def test_the_guard_really_reads_files_outside_the_roots(
        self, di: ModuleType
    ) -> None:
        """Vacuity check: the set the guard reads is non-empty and is NOT the scanned set."""
        tracked = _git_tracked()
        scanned = {rel for rel, _ in di._files(REPO)}
        outside = sorted(set(tracked) - scanned)
        assert len(outside) > 100
        assert "mkdocs.yml" in outside
        assert any(p.startswith("examples/") for p in outside)
        assert not (set(outside) & scanned)


class TestPlantedAnchors:
    """Mutants: an anchor planted in a file the scanner does not read is reported."""

    @staticmethod
    def _tree(tmp_path: Path, rel: str, text: str) -> tuple[Path, list[str]]:
        _write(tmp_path, "src/ok.py", "x = 1\n")
        _write(tmp_path, rel, text)
        return tmp_path, ["src/ok.py", rel]

    @pytest.mark.parametrize(
        ("rel", "text"),
        [
            ("docs/guide.md", f"# Guide\n\ntext\n{HTML} {TOKEN} rule -->\n"),
            ("docs/guide.md", f"{HTML} {TOKEN}\n   a long rule\n-->\n"),
            ("README.md", f"{HTML} {TOKEN} rule -->\n"),
            ("AGENTS.md", f"{HTML} {TOKEN} [STALE] rule -->\n"),
            ("mkdocs.yml", f"site_name: x\n{A} {TOKEN}: keep the strict build\n"),
            (".pre-commit-config.yaml", f"repos: []\n{A} {TOKEN}: pinned on purpose\n"),
            ("examples/run.py", f"x = 1\n{A} {TOKEN}: frozen baseline\n"),
            ("docs/site.css", f"/* DECISION {TOKEN}: no web fonts */\n"),
            ("web/app.js", f"{SLASH} {TOKEN}: no framework\n"),
            ("db/schema.sql", f"{DASHES} {TOKEN}: no triggers\n"),
            ("setup.cfg", f"[x]\n{A} {TOKEN}: why\n"),
            ("NOTES", f"{A} {TOKEN}: no suffix at all\n"),
        ],
        ids=[
            "docs-md",
            "docs-md-multiline",
            "root-readme",
            "root-agents-stale",
            "mkdocs-yml",
            "dot-yaml",
            "examples-py",
            "css-block",
            "js-slash",
            "sql-dashes",
            "cfg-unknown-suffix",
            "no-suffix",
        ],
    )
    def test_an_anchor_outside_the_roots_fails_the_guard(
        self, di: ModuleType, tmp_path: Path, rel: str, text: str
    ) -> None:
        root, tracked = self._tree(tmp_path, rel, text)
        assert stray_anchors(di, root, tracked) == [f"{rel}: {TOKEN}"]

    def test_the_same_tree_without_the_plant_passes(
        self, di: ModuleType, tmp_path: Path
    ) -> None:
        root, tracked = self._tree(tmp_path, "docs/guide.md", "# Guide\n\ntext\n")
        assert stray_anchors(di, root, tracked) == []

    def test_a_planted_anchor_in_a_copy_of_a_real_file_is_found(
        self, di: ModuleType, tmp_path: Path
    ) -> None:
        """The mutant of the task text: a real `mkdocs.yml` and a real docs page with an
        anchor appended."""
        for rel, tail in (
            ("mkdocs.yml", f"\n{A} {TOKEN}: planted\n"),
            ("AGENTS.md", f"\n{HTML} {TOKEN} planted -->\n"),
        ):
            source = REPO / rel
            if not source.is_file():
                pytest.skip(f"{rel} is not part of this checkout")
            copy = tmp_path / rel.replace("/", "_")
            _write(copy, rel, source.read_text(encoding="utf-8") + tail)
            assert stray_anchors(di, copy, [rel]) == [f"{rel}: {TOKEN}"]
            assert stray_anchors(di, copy, []) == []


class TestNothingElseTrips:
    def test_files_the_scanner_reads_are_not_the_guards_business(
        self, di: ModuleType, tmp_path: Path
    ) -> None:
        for rel in ("src/a.py", "tests/t.py", "scripts/s.py", ".github/ci.yml"):
            _write(tmp_path, rel, f"{A} {TOKEN}: scanned elsewhere\n")
        tracked = ["src/a.py", "tests/t.py", "scripts/s.py", ".github/ci.yml"]
        assert stray_anchors(di, tmp_path, tracked) == []

    def test_code_spans_and_fences_in_markdown_are_not_anchors(
        self, di: ModuleType, tmp_path: Path
    ) -> None:
        text = f"Write `{HTML} {TOKEN} -->`.\n\n```\n{HTML} {TOKEN} -->\n```\n"
        _write(tmp_path, "docs/g.md", text)
        assert stray_anchors(di, tmp_path, ["docs/g.md"]) == []

    def test_prose_citations_and_the_grammar_placeholder_are_not_anchors(
        self, di: ModuleType, tmp_path: Path
    ) -> None:
        text = f"See {P1}/D-001 in the plan.\nAnchors read `{A} plan-<id>/D-NNN`.\n"
        _write(tmp_path, "docs/g.md", text)
        _write(tmp_path, "NOTES", text)
        assert stray_anchors(di, tmp_path, ["docs/g.md", "NOTES"]) == []

    def test_an_untracked_missing_symlinked_or_binary_file_is_skipped(
        self, di: ModuleType, tmp_path: Path
    ) -> None:
        _write(tmp_path, "docs/untracked.md", f"{HTML} {TOKEN} x -->\n")
        _write(tmp_path, "docs/blob.bin", b"\x00\x01" + f"{A} {TOKEN}\n".encode())
        real = _write(tmp_path, "docs/real.md", f"{HTML} {TOKEN} x -->\n")
        (tmp_path / "docs" / "link.md").symlink_to(real)
        tracked = ["docs/blob.bin", "docs/link.md", "docs/missing.md"]
        assert stray_anchors(di, tmp_path, tracked) == []

    @pytest.mark.parametrize(
        "rel",
        ["docs/decisions/index.md", "docs/decisions/deep/x.md", "plans/ANCHORS.md"],
    )
    def test_the_generated_index_and_the_manifest_are_excluded(
        self, di: ModuleType, tmp_path: Path, rel: str
    ) -> None:
        _write(tmp_path, rel, f"{HTML} {TOKEN} quoted -->\n")
        assert di.outside_scope(tmp_path, [rel]) != []  # the scanner would see it
        assert stray_anchors(di, tmp_path, [rel]) == []  # the guard leaves it out

    def test_a_missing_excluded_path_is_harmless(
        self, di: ModuleType, tmp_path: Path
    ) -> None:
        assert stray_anchors(di, tmp_path, ["docs/decisions/index.md"]) == []
