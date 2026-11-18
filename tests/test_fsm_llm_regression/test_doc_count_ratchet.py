"""No tracked document states a count that changes when the repository changes: a ratchet.

A number such as "19,100 tests", "103 test files" or "32 contracts" in a document is a
hand-kept copy of a fact the repository can measure, so it drifts. The rule of the project
is that a document states no such count (a reader measures it: `make test`, or
`pytest --collect-only -q | tail -1`), and this test keeps one from coming back. It is a
per-file ratchet in the manner of `test_em_dash_ratchet.py`, with a stronger fixture:
`tests/fixtures/doc_count_allowlist.json` maps every tracked `.md` file that has a
count-shaped string to the multiset of those STRINGS (not line numbers), so the test fails
when

* a file gains a match (a new count, a new file with one, a limit that changed: `800
  physical lines` becomes `801 physical lines`), or
* an entry no longer matches (the allowlist only tightens: a removed match asks for a
  regeneration).

A legitimate number that looks like a count (a limit such as the 800-line module cap) is
listed on purpose; the failure message says how.

Shapes (`N` is a number of digits, with thousands commas and an optional `+`, that is not
part of a version, a date, an id, a path or a `$` amount; matched on one line, text with
its whitespace collapsed):

* S1 `N (word){0,2} test files | test modules | test cases | tests | collected`
* S2 `collects N`, `collected N`
* S3 a marker name in backticks followed by `(N` (`slow`, `integration`, `real_llm`,
  `examples`, `live`), and `N (of them) slow | live | gated | real_llm`
* S4 `N (word) files | modules | lines | contracts`
* S5 a table row led by a backticked `test_x.py` or `test_suite/` that holds a bare
  numeric cell (the match is the first cell, ` | `, the numeric cell)

A line is scanned in full, a fenced block included (the counts of the root AGENTS.md lived
in bash fences). `TestShapes` holds the contract: lines that MUST hit and lines that MUST
NOT; change a regex and that table together.

Read: every `.md` file `git ls-files --cached --others --exclude-standard` lists, decodable
as UTF-8 with no NUL byte, that is not a symlink. Never read: `plans/`,
`scripts/bench_data/`, `evaluation/`, `examples/`, `CHANGELOG.md` (release history),
`EVALUATE.md` (an append-only run log) and any `docs/*roadmap.md` (a dated planning page).
Not run when `git` is missing or the checkout is not a git repository.

NOT detected, on purpose (a human reads these): a number written as a word ("seven
subpackages"), a product count without a shape above ("18 agent patterns", "9 strategies",
"100 runnable examples"), a range ("10-12 tests"), and anything in `.py` comments or
docstrings. The guard catches the common creep, a `(N tests)` pasted into a doc; it is not
a proof that no count is left.

Regenerate after removing a count, after deleting a file that held one, or on purpose when
a new document needs a limit:

    python -m tests.test_fsm_llm_regression.test_doc_count_ratchet --write

The fixtures below name numbers and never an em dash.
"""

from __future__ import annotations

import fnmatch
import json
import re
import sys
from pathlib import Path

import pytest

from tests.conftest import symlink_or_skip
from tests.test_fsm_llm_regression.support import REPO
from tests.test_fsm_llm_regression.test_em_dash_ratchet import tracked_files

ALLOWLIST = REPO / "tests" / "fixtures" / "doc_count_allowlist.json"
REGENERATE = "python -m tests.test_fsm_llm_regression.test_doc_count_ratchet --write"
#: Path prefixes that are never read.
EXEMPT_PREFIXES = ("plans/", "scripts/bench_data/", "evaluation/", "examples/")
#: Exact paths that are never read (release history, an append-only run log).
EXEMPT_PATHS = ("CHANGELOG.md", "EVALUATE.md")
#: Globs that are never read; `fnmatchcase`, so Windows case and separator rules do not apply.
EXEMPT_GLOBS = ("docs/*roadmap.md",)

# A number: not part of a version (0.13.0), an id (D-004, 21e6f00d), a path, a `$` amount
# or the tail of a thousands group (the `,` of "$1,200").
_N = r"(?<![\w.\-/`$])(?<!\d,)(?:\d{1,3}(?:,\d{3})+|\d+)\+?"
_WORD = r"(?:[A-Za-z][\w\-]*\s+)"
SHAPES: dict[str, re.Pattern[str]] = {
    "S1": re.compile(
        rf"{_N}\s+{_WORD}{{0,2}}(?:test files?|test modules|test cases|tests?|collected)\b",
        re.IGNORECASE,
    ),
    "S2": re.compile(rf"\bcollect(?:s|ed)\s+{_N}", re.IGNORECASE),
    "S3a": re.compile(
        rf"`(?:slow|integration|real_llm|examples|live)`\s*\(\s*{_N}", re.IGNORECASE
    ),
    "S3b": re.compile(
        rf"{_N}\s+(?:of them\s+)?(?:slow|live|gated|real_llm)\b", re.IGNORECASE
    ),
    "S4": re.compile(
        rf"{_N}\s+{_WORD}?(?:files?|modules?|lines|contracts)\b", re.IGNORECASE
    ),
}
_S5_ROW = re.compile(r"^\s*\|\s*`(?:test_\w+\.py|test_\w+/)`")
_S5_CELL = re.compile(r"^(?:\d{1,3}(?:,\d{3})+|\d+)\+?(?:\s*\(.*\))?$")


def is_exempt(rel: str) -> bool:
    """True for a path the ratchet never reads."""
    return (
        rel.startswith(EXEMPT_PREFIXES)
        or rel in EXEMPT_PATHS
        or any(fnmatch.fnmatchcase(rel, g) for g in EXEMPT_GLOBS)
    )


def _cells(line: str) -> list[str]:
    """The cells of a markdown table row, split on `|` outside backticks."""
    cells: list[str] = []
    cur: list[str] = []
    tick = False
    for ch in line.strip().strip("|"):
        if ch == "`":
            tick = not tick
        if ch == "|" and not tick:
            cells.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    cells.append("".join(cur).strip())
    return cells


def shape_hits(line: str) -> dict[str, list[str]]:
    """`{shape: [matched text]}` for one line (whitespace of the text collapsed)."""
    hits: dict[str, list[str]] = {}
    for name, rx in SHAPES.items():
        found = [" ".join(m.group(0).split()) for m in rx.finditer(line)]
        if found:
            hits[name] = found
    if _S5_ROW.match(line):
        cells = _cells(line)
        for cell in cells[1:]:
            if _S5_CELL.match(cell):
                hits["S5"] = [" | ".join((" ".join(cells[0].split()), cell))]
                break
    return hits


def matches_in_text(text: str) -> dict[str, int]:
    """`{matched text: n}` of a document. One span found by two shapes counts once."""
    found: dict[str, int] = {}
    for line in text.splitlines():
        seen: set[str] = set()
        for texts in shape_hits(line).values():
            for matched in texts:
                if matched not in seen:
                    seen.add(matched)
                    found[matched] = found.get(matched, 0) + 1
    return found


def readable_docs(root: Path, files: list[str]) -> dict[str, str]:
    """`{path: text}` of the `.md` files outside the exemptions that can be read."""
    docs: dict[str, str] = {}
    for rel in files:
        if not rel.lower().endswith(".md") or is_exempt(rel):
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
        docs[rel] = text
    return docs


def count_matches(root: Path, files: list[str]) -> dict[str, dict[str, int]]:
    """`{path: {matched text: n}}` for every readable `.md` that has a count-shaped string."""
    counts: dict[str, dict[str, int]] = {}
    for rel, text in readable_docs(root, files).items():
        found = matches_in_text(text)
        if found:
            counts[rel] = found
    return counts


def problems(
    have: dict[str, dict[str, int]], allowed: dict[str, dict[str, int]]
) -> list[str]:
    """Why the ratchet is red; `[]` when `have` equals `allowed` file by file and string by string."""
    out: list[str] = []
    for rel in sorted(set(have) | set(allowed)):
        mine, may = have.get(rel, {}), allowed.get(rel, {})
        for text in sorted(set(mine) | set(may)):
            n, m = mine.get(text, 0), may.get(text, 0)
            if n > m:
                out.append(f"{rel}: {text!r} x{n}, the allowlist has {m}")
            elif n < m:
                out.append(f"{rel}: {text!r} x{n}, the allowlist still says {m}")
    return out


def render(counts: dict[str, dict[str, int]]) -> str:
    return json.dumps(counts, indent=1, sort_keys=True, ensure_ascii=True) + "\n"


def _allowed() -> dict[str, dict[str, int]]:
    return json.loads(ALLOWLIST.read_text(encoding="utf-8"))


class TestTheRealTree:
    def test_no_document_gained_a_count_and_the_allowlist_is_tight(self) -> None:
        files = tracked_files()
        if files is None:
            pytest.skip("not a git checkout: there is no file list to read")
        bad = problems(count_matches(REPO, files), _allowed())
        assert not bad, (
            "doc count ratchet: a document states a count (a test, file, module, line "
            "or contract count drifts; a reader measures it with `make test` or "
            "`pytest --collect-only -q`). State no count; or, for a real limit or an "
            f"illustration, or when a count was removed, run `{REGENERATE}` and commit "
            "the allowlist:\n  " + "\n  ".join(bad)
        )

    def test_the_allowlist_is_the_rendering_of_its_own_data(self) -> None:
        text = ALLOWLIST.read_text(encoding="utf-8")
        assert text == render(json.loads(text)), f"regenerate with `{REGENERATE}`"
        assert text.isascii()

    def test_no_exempt_path_and_no_zero_entry_is_listed(self) -> None:
        allowed = _allowed()
        assert not [p for p in allowed if is_exempt(p)]
        assert all(p.lower().endswith(".md") for p in allowed)
        for rel, texts in allowed.items():
            assert texts, f"{rel} has no string"
            assert all(isinstance(n, int) and n > 0 for n in texts.values()), rel

    def test_the_ratchet_is_not_vacuous(self) -> None:
        """It reads many documents, not a handful (a broken file listing reads none)."""
        files = tracked_files()
        if files is None:
            pytest.skip("not a git checkout")
        docs = readable_docs(REPO, files)
        assert len(docs) > 50
        assert "AGENTS.md" in docs and "README.md" in docs
        assert not [p for p in docs if is_exempt(p)]


#: (line, exactly the shapes that name it): a count of the repository, in every form it took.
MUST_HIT = [
    ("Run the full suite (19,100 tests).", {"S1"}),
    ("pytest tests/  # collected (19,100 collected)", {"S1"}),
    ("The core suite has 8,259 tests.", {"S1"}),
    ("It collects 19,049 tests in the CI slice.", {"S2", "S1"}),
    ("pytest --collect-only found collected 19,100", {"S2"}),
    ("The `tests/test_fsm_llm/` folder holds 103 test files.", {"S1", "S4"}),
    ("216 fixture-only tests need no model", {"S1"}),
    ("a `slow` (8 of them) marker", {"S3a"}),
    ("a `real_llm` (12) marker", {"S3a"}),
    ("21 tests, 8 slow", {"S1", "S3b"}),
    ("4 of them live against a model", {"S3b"}),
    ("5 gated", {"S3b"}),
    ("32 contracts", {"S4"}),
    ("every module is under 800 physical lines", {"S4"}),
    ("the package has 120 modules", {"S4"}),
    ("1,399 lines", {"S4"}),
    ("a 1 MB file", {"S4"}),
    ("| `test_packaging.py` | 385 |", {"S5"}),
    ("| `test_fsm_llm_agents/` | 3,428 | the agents suite |", {"S5"}),
    ("| `test_x.py` | 21 (8 slow) | an entry |", {"S5", "S3b"}),
    ("| `test_y.py` | 7+ | an entry |", {"S5"}),
]
#: Lines that state no count: versions, ids, dates, ports, money, limits of another kind.
MUST_NOT_HIT = [
    "Python framework (v0.13.0, Apache-2.0, Python 3.11-3.15)",
    "pip install llm-fsm==0.13.0 on Python 3.12",
    "schema version 4.2 and 4.0",
    "see D-004 and plan 21e6f00d, step 15.3",
    "Run 006 is 95.3% (N=3 median)",
    "max_steps=10 and depth 16",
    "the dashboard listens on 127.0.0.1:8420",
    "port 8420 is the default",
    "2026-11-18 and 2026-09-29",
    "costs $150 files and $1,200 lines",
    "the file src/fsm_llm/runtime/log.py has a lock",
    "tests/test_fsm_llm_regression/test_import_contract.py checks the layers",
    "| `test_packaging.py` | wiring of the package |",
    "| `test_packaging.py` | 3.11 |",
    "| Folder | What it checks |",
    "the 1.5 files rule and a 3.5 lines rule",
    "run `make test` to measure the suite",
    "a test file with many tests",
]


class TestShapes:
    @pytest.mark.parametrize(("line", "shapes"), MUST_HIT)
    def test_a_count_line_hits_its_shapes(self, line: str, shapes: set[str]) -> None:
        hits = shape_hits(line)
        assert set(hits) == shapes, f"{line!r}: wanted {shapes}, got {hits}"
        assert matches_in_text(line), line

    @pytest.mark.parametrize("line", MUST_NOT_HIT)
    def test_a_line_that_states_no_count_does_not_hit(self, line: str) -> None:
        assert shape_hits(line) == {}, line

    def test_the_matched_text_has_its_whitespace_collapsed(self) -> None:
        assert matches_in_text("about 19,100   tests  ") == {"19,100 tests": 1}

    def test_one_span_found_by_two_shapes_counts_once(self) -> None:
        assert set(shape_hits("103 test files")) == {"S1", "S4"}
        assert matches_in_text("103 test files") == {"103 test files": 1}

    def test_every_shape_has_a_line_that_only_it_hits(self) -> None:
        """Disabling one shape must turn a test red: each has its own unique line."""
        sole = {next(iter(s)) for _, s in MUST_HIT if len(s) == 1}
        assert sole == {*SHAPES, "S5"}


class TestMutants:
    """A planted count is caught; clean and unchanged trees are not."""

    CLEAN = "Nothing is counted here.\n"
    LIMIT = "A module is at most 800 physical lines.\n"

    def _tree(self, root: Path) -> dict[str, dict[str, int]]:
        (root / "docs").mkdir()
        (root / "docs" / "clean.md").write_text(self.CLEAN, encoding="utf-8")
        (root / "docs" / "limit.md").write_text(self.LIMIT, encoding="utf-8")
        return count_matches(root, ["docs/clean.md", "docs/limit.md"])

    @staticmethod
    def _scan(root: Path, *extra: str) -> dict[str, dict[str, int]]:
        return count_matches(root, ["docs/clean.md", "docs/limit.md", *extra])

    def test_an_unchanged_tree_is_green(self, tmp_path: Path) -> None:
        allowed = self._tree(tmp_path)
        assert allowed == {"docs/limit.md": {"800 physical lines": 1}}
        assert problems(self._scan(tmp_path), allowed) == []

    def test_a_count_added_to_a_clean_file_fails(self, tmp_path: Path) -> None:
        allowed = self._tree(tmp_path)
        (tmp_path / "docs" / "clean.md").write_text(
            "The suite has 19,100 tests.\n", encoding="utf-8"
        )
        assert problems(self._scan(tmp_path), allowed) == [
            "docs/clean.md: '19,100 tests' x1, the allowlist has 0"
        ]

    def test_a_count_added_to_a_listed_file_fails(self, tmp_path: Path) -> None:
        allowed = self._tree(tmp_path)
        (tmp_path / "docs" / "limit.md").write_text(
            self.LIMIT + "and 103 test files\n", encoding="utf-8"
        )
        assert problems(self._scan(tmp_path), allowed) == [
            "docs/limit.md: '103 test files' x1, the allowlist has 0"
        ]

    def test_a_second_copy_of_a_listed_string_fails(self, tmp_path: Path) -> None:
        allowed = self._tree(tmp_path)
        (tmp_path / "docs" / "limit.md").write_text(self.LIMIT * 2, encoding="utf-8")
        assert problems(self._scan(tmp_path), allowed) == [
            "docs/limit.md: '800 physical lines' x2, the allowlist has 1"
        ]

    def test_a_changed_limit_fails_both_ways(self, tmp_path: Path) -> None:
        allowed = self._tree(tmp_path)
        (tmp_path / "docs" / "limit.md").write_text(
            self.LIMIT.replace("800", "801"), encoding="utf-8"
        )
        assert problems(self._scan(tmp_path), allowed) == [
            "docs/limit.md: '800 physical lines' x0, the allowlist still says 1",
            "docs/limit.md: '801 physical lines' x1, the allowlist has 0",
        ]

    def test_a_new_file_with_a_count_fails(self, tmp_path: Path) -> None:
        allowed = self._tree(tmp_path)
        (tmp_path / "NEW.md").write_text("32 contracts\n", encoding="utf-8")
        assert problems(self._scan(tmp_path, "NEW.md"), allowed) == [
            "NEW.md: '32 contracts' x1, the allowlist has 0"
        ]

    def test_a_count_inside_a_fenced_block_fails(self, tmp_path: Path) -> None:
        allowed = self._tree(tmp_path)
        (tmp_path / "docs" / "clean.md").write_text(
            "```bash\nmake test  # Run (19,100 tests)\n```\n", encoding="utf-8"
        )
        assert problems(self._scan(tmp_path), allowed) == [
            "docs/clean.md: '19,100 tests' x1, the allowlist has 0"
        ]

    def test_a_count_in_a_table_cell_fails(self, tmp_path: Path) -> None:
        allowed = self._tree(tmp_path)
        (tmp_path / "docs" / "clean.md").write_text(
            "| Folder | Tests |\n| --- | --- |\n| `test_a.py` | 12 |\n",
            encoding="utf-8",
        )
        assert problems(self._scan(tmp_path), allowed) == [
            "docs/clean.md: '`test_a.py` | 12' x1, the allowlist has 0"
        ]

    def test_a_removed_count_or_a_deleted_file_asks_for_a_regeneration(
        self, tmp_path: Path
    ) -> None:
        allowed = self._tree(tmp_path)
        (tmp_path / "docs" / "limit.md").write_text(self.CLEAN, encoding="utf-8")
        gone = ["docs/limit.md: '800 physical lines' x0, the allowlist still says 1"]
        assert problems(self._scan(tmp_path), allowed) == gone
        (tmp_path / "docs" / "limit.md").unlink()
        assert problems(self._scan(tmp_path), allowed) == gone

    @pytest.mark.parametrize(
        "rel",
        [
            "plans/p/progress.md",
            "scripts/bench_data/README.md",
            "evaluation/run/scorecard.md",
            "examples/basic/README.md",
            "CHANGELOG.md",
            "EVALUATE.md",
            "docs/2026_roadmap.md",
            "docs/x_roadmap.md",
            "docs/20261116_mcp_roadmap.md",
        ],
    )
    def test_exempt_paths_are_never_read(self, tmp_path: Path, rel: str) -> None:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("The suite has 19,100 tests.\n", encoding="utf-8")
        assert is_exempt(rel)
        assert count_matches(tmp_path, [rel]) == {}

    @pytest.mark.parametrize(
        "rel",
        ["docs/roadmap_notes.md", "docs/Roadmap.md", "docs/a/AGENTS.md", "README.md"],
    )
    def test_a_path_near_an_exemption_is_still_read(
        self, tmp_path: Path, rel: str
    ) -> None:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("The suite has 19,100 tests.\n", encoding="utf-8")
        assert not is_exempt(rel)
        assert count_matches(tmp_path, [rel]) == {rel: {"19,100 tests": 1}}

    def test_only_markdown_is_read(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("# 19,100 tests\n", encoding="utf-8")
        (tmp_path / "a.txt").write_text("19,100 tests\n", encoding="utf-8")
        assert count_matches(tmp_path, ["a.py", "a.txt"]) == {}

    def test_binary_non_utf8_missing_and_symlinked_files_are_skipped(
        self, tmp_path: Path
    ) -> None:
        planted = b"19,100 tests"
        (tmp_path / "blob.md").write_bytes(b"\x00" + planted)
        (tmp_path / "latin.md").write_bytes(b"\xff" + planted)
        real = tmp_path / "real.md"
        real.write_text("19,100 tests\n", encoding="utf-8")
        symlink_or_skip(tmp_path / "link.md", real)
        counts = count_matches(
            tmp_path, ["blob.md", "latin.md", "gone.md", "link.md", "real.md"]
        )
        assert counts == {"real.md": {"19,100 tests": 1}}

    def test_the_regeneration_renders_sorted_ascii_json(self) -> None:
        text = render({"b.md": {"2 files": 1}, "a.md": {"9 lines": 2, "1 file": 1}})
        assert text == (
            '{\n "a.md": {\n  "1 file": 1,\n  "9 lines": 2\n },\n'
            ' "b.md": {\n  "2 files": 1\n }\n}\n'
        )


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        sys.exit(f"usage: {REGENERATE}")
    listed = tracked_files()
    if listed is None:
        sys.exit("not a git checkout: there is no file list to read")
    ALLOWLIST.write_text(render(count_matches(REPO, listed)), encoding="utf-8")
    print(f"wrote {ALLOWLIST.relative_to(REPO)}")
