"""Text added to the repository carries no em dash (U+2014): a per-file ratchet.

The project rule is plain direct language with no em dashes in new text. Inherited text
has thousands of them (the sweep of step 121.2 counted 334 in `src/` alone at the base
commit), and rewriting history to satisfy a rule is not the point, so this is a RATCHET and
not a ban: `tests/fixtures/em_dash_allowlist.json` maps every tracked text file that has an
em dash to its exact count, and the test fails when

* a file has MORE em dashes than its entry (or has some and no entry): new text must be
  clean, and
* a file has FEWER than its entry, or an entry names a file that is gone or has none: the
  allowlist only tightens, so removed dashes cannot be put back unseen.

Read: every file `git ls-files` lists (the index only: a foreign untracked file, a scratch
file or a build product never turns the ratchet red, so the live working tree can be gated;
a new file is judged from `git add` on), decodable as UTF-8 with no NUL byte. Never read: `plans/` (the plan record),
`docs/2026_roadmap.md` (never edited) and `EVALUATE.md` (an
append-only run log). Not run when `git` is missing or the checkout is not a git repository.

Regenerate after removing dashes, after deleting or moving a file that held some (step 123
deletes most of `docs/`), or on purpose when a new file needs one as data:

    python -m tests.test_fsm_llm_regression.test_em_dash_ratchet --write

Fixtures here build the character with `chr(0x2014)`, so this file has none.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import symlink_or_skip
from tests.test_fsm_llm_regression.support import REPO

DASH = chr(0x2014)  # built from its code point: this file holds no literal one
ALLOWLIST = REPO / "tests" / "fixtures" / "em_dash_allowlist.json"
REGENERATE = "python -m tests.test_fsm_llm_regression.test_em_dash_ratchet --write"
#: Path prefixes (or exact paths) that are never read.
EXEMPT = (
    "plans/",
    "docs/2026_roadmap.md",
    "EVALUATE.md",
)


def tracked_files(root: Path = REPO) -> list[str] | None:
    """The tracked files of the checkout (`git ls-files`), or None without git.

    Interface contract: `root` is a work tree. Returns sorted repo-relative posix paths of
    the index (a file added with `git add` counts, an untracked one does not), or `None`
    when git is missing, `root` has no `.git` or git fails.
    """
    if shutil.which("git") is None or not (root / ".git").exists():
        return None
    done = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        capture_output=True,
        timeout=60,
        check=False,
    )
    if done.returncode != 0:
        return None
    names = done.stdout.decode("utf-8", "replace").split("\0")
    return sorted({n for n in names if n})


def count_dashes(root: Path, files: list[str]) -> dict[str, int]:
    """`{path: count}` for every readable text file outside `EXEMPT` that has a dash."""
    counts: dict[str, int] = {}
    for rel in files:
        if rel.startswith(EXEMPT):
            continue
        path = root / rel
        if not path.is_file() or path.is_symlink():
            continue
        try:
            text = path.read_bytes().decode("utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if "\x00" in text:
            continue
        found = text.count(DASH)
        if found:
            counts[rel] = found
    return counts


def problems(counts: dict[str, int], allowed: dict[str, int]) -> list[str]:
    """Why the ratchet is red; `[]` when `counts` equals `allowed` file by file."""
    out: list[str] = []
    for rel in sorted(set(counts) | set(allowed)):
        have, may = counts.get(rel, 0), allowed.get(rel, 0)
        if have > may:
            out.append(f"{rel}: {have} em dash(es), the allowlist has {may}")
        elif have < may:
            out.append(f"{rel}: {have} em dash(es), the allowlist still says {may}")
    return out


def render(counts: dict[str, int]) -> str:
    return json.dumps(counts, indent=1, sort_keys=True, ensure_ascii=True) + "\n"


def _allowed() -> dict[str, int]:
    return json.loads(ALLOWLIST.read_text(encoding="utf-8"))


class TestTheRealTree:
    def test_no_file_gained_an_em_dash_and_the_allowlist_is_tight(self) -> None:
        files = tracked_files()
        if files is None:
            pytest.skip("not a git checkout: there is no file list to read")
        bad = problems(count_dashes(REPO, files), _allowed())
        assert not bad, (
            "em dash ratchet (new text carries none; the allowlist only tightens). "
            "Replace the dash with a comma, a colon or parentheses; or, when files were "
            f"removed or cleaned or a dash is deliberate data, run `{REGENERATE}` and "
            "commit the allowlist:\n  " + "\n  ".join(bad)
        )

    def test_the_allowlist_is_the_rendering_of_its_own_counts(self) -> None:
        text = ALLOWLIST.read_text(encoding="utf-8")
        assert text == render(json.loads(text)), f"regenerate with `{REGENERATE}`"
        assert DASH not in text

    def test_no_exempt_path_and_no_zero_entry_is_listed(self) -> None:
        allowed = _allowed()
        assert not [p for p in allowed if p.startswith(EXEMPT)]
        assert all(isinstance(n, int) and n > 0 for n in allowed.values())

    def test_the_ratchet_is_not_vacuous(self) -> None:
        """It reads a lot of files and some have dashes (the inherited text)."""
        files = tracked_files()
        if files is None:
            pytest.skip("not a git checkout")
        counts = count_dashes(REPO, files)
        assert len(files) > 500 and len(counts) > 50
        assert "src/fsm_llm/agents/base/__init__.py" in counts


class TestTrackedFilesOnly:
    def test_an_untracked_file_is_not_read_and_an_added_one_is(
        self, tmp_path: Path
    ) -> None:
        if shutil.which("git") is None:
            pytest.skip("git is missing")
        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
        (tmp_path / "tracked.md").write_text("x\n", encoding="utf-8")
        (tmp_path / "foreign.md").write_text(f"x {DASH}\n", encoding="utf-8")
        subprocess.run(["git", "add", "tracked.md"], cwd=tmp_path, check=True)
        assert tracked_files(tmp_path) == ["tracked.md"]
        subprocess.run(["git", "add", "foreign.md"], cwd=tmp_path, check=True)
        assert tracked_files(tmp_path) == ["foreign.md", "tracked.md"]

    def test_a_directory_without_git_gives_none(self, tmp_path: Path) -> None:
        assert tracked_files(tmp_path) is None


class TestMutants:
    """A planted dash is caught; clean and unchanged trees are not."""

    @staticmethod
    def _tree(root: Path) -> dict[str, int]:
        (root / "src").mkdir()
        (root / "src" / "a.py").write_text("# clean\n", encoding="utf-8")
        (root / "src" / "b.py").write_text(f"# a {DASH} b\n", encoding="utf-8")
        files = ["src/a.py", "src/b.py"]
        return count_dashes(root, files)

    def test_an_unchanged_tree_is_green(self, tmp_path: Path) -> None:
        counts = self._tree(tmp_path)
        assert counts == {"src/b.py": 1}
        assert problems(counts, dict(counts)) == []

    def test_a_dash_added_to_a_clean_tracked_file_fails(self, tmp_path: Path) -> None:
        allowed = self._tree(tmp_path)
        (tmp_path / "src" / "a.py").write_text(f"# x {DASH} y\n", encoding="utf-8")
        bad = problems(count_dashes(tmp_path, ["src/a.py", "src/b.py"]), allowed)
        assert bad == ["src/a.py: 1 em dash(es), the allowlist has 0"]

    def test_a_second_dash_in_a_listed_file_fails(self, tmp_path: Path) -> None:
        allowed = self._tree(tmp_path)
        (tmp_path / "src" / "b.py").write_text(f"{DASH}{DASH}\n", encoding="utf-8")
        bad = problems(count_dashes(tmp_path, ["src/a.py", "src/b.py"]), allowed)
        assert bad == ["src/b.py: 2 em dash(es), the allowlist has 1"]

    def test_a_new_file_with_a_dash_fails(self, tmp_path: Path) -> None:
        allowed = self._tree(tmp_path)
        (tmp_path / "NEW.md").write_text(f"x {DASH}\n", encoding="utf-8")
        files = ["src/a.py", "src/b.py", "NEW.md"]
        assert problems(count_dashes(tmp_path, files), allowed) == [
            "NEW.md: 1 em dash(es), the allowlist has 0"
        ]

    def test_a_removed_dash_or_a_deleted_file_asks_for_a_regeneration(
        self, tmp_path: Path
    ) -> None:
        allowed = self._tree(tmp_path)
        (tmp_path / "src" / "b.py").write_text("# clean now\n", encoding="utf-8")
        assert problems(count_dashes(tmp_path, ["src/a.py", "src/b.py"]), allowed) == [
            "src/b.py: 0 em dash(es), the allowlist still says 1"
        ]
        (tmp_path / "src" / "b.py").unlink()
        assert problems(count_dashes(tmp_path, ["src/a.py", "src/b.py"]), allowed) == [
            "src/b.py: 0 em dash(es), the allowlist still says 1"
        ]

    @pytest.mark.parametrize("rel", [*EXEMPT, "plans/x/progress.md"])
    def test_exempt_paths_are_never_read(self, tmp_path: Path, rel: str) -> None:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"x {DASH}\n", encoding="utf-8")
        assert count_dashes(tmp_path, [rel]) == {}

    def test_binary_non_utf8_missing_and_symlinked_files_are_skipped(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "blob.bin").write_bytes(b"\x00" + DASH.encode() + b"\x00")
        (tmp_path / "latin.txt").write_bytes(b"\xff" + DASH.encode())
        real = tmp_path / "real.md"
        real.write_text(f"{DASH}\n", encoding="utf-8")
        symlink_or_skip(tmp_path / "link.md", real)
        counts = count_dashes(
            tmp_path, ["blob.bin", "latin.txt", "gone.md", "link.md", "real.md"]
        )
        assert counts == {"real.md": 1}

    def test_the_regeneration_renders_the_counts_sorted_and_ascii(self) -> None:
        text = render({"b.py": 2, "a.py": 1})
        assert text == '{\n "a.py": 1,\n "b.py": 2\n}\n'


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        sys.exit(f"usage: {REGENERATE}")
    listed = tracked_files()
    if listed is None:
        sys.exit("not a git checkout: there is no file list to read")
    ALLOWLIST.write_text(render(count_dashes(REPO, listed)), encoding="utf-8")
    print(f"wrote {ALLOWLIST.relative_to(REPO)}")
