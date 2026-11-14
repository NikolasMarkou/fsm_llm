"""`scripts/decision_index.py`: the anchor scanner and the manifest parser (roadmap row 0.5).

The script finds every `# DECISION [plan-id/]D-NNN [STALE]` anchor of the repository,
parses the append-only manifest `plans/ANCHORS.md` and sorts each anchor into one state:

* `resolved`   a manifest line exists for the anchor's `<plan-id>/D-NNN`;
* `retired`    the anchor carries `[STALE]` and there is no manifest line (the plan record
               is gone for good; the comment in the code is the record; D-008);
* `unresolved` a live anchor with no manifest line;
* `badprefix`  the prefix is not a plan id (the commit-tag form, a typo) or is missing.

The grammar is the planner's (`validate-plan.mjs` `buildAnchorRegexes`,
`LOOSE_ANCHOR_PREFIX_PATTERN` and the manifest line regex), so the numbers of the script
and of `validate-plan.mjs` agree. Most tests run on synthetic trees under `tmp_path`; the
last classes read the real repository with loose numbers.

Script-side contract the tests pin: stdlib only, never imports `fsm_llm`, never touches
`sys.path`, no line number in any record or report (D-008: a line number is stale at the
next commit), deterministic order.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tests.conftest import posix_only, skip_if_root, symlink_or_skip
from tests.test_fsm_llm_regression.support import REPO

SCRIPT = REPO / "scripts" / "decision_index.py"
MULTISET = REPO / "tests" / "fixtures" / "anchor_multiset.json"
MANIFEST = REPO / "plans" / "ANCHORS.md"

P1 = "plan-2026-11-10T194139-2cefeb61"
P2 = "plan-2026-09-19T175721-21cd7f8e"
P3 = "plan-2026-07-19T075908-70b6bdec"
LEGACY = "plan_2026-05-29_0c00a594"
#: the hash-comment head, built so this file holds no anchor-looking line itself: the
#: planner and `scripts/decision_index.py` read a `#`, a space and the word in any string
A = "#" + " DECISION"


def _load() -> ModuleType:
    """Import the script by path under a private name (it is not a package module)."""
    spec = importlib.util.spec_from_file_location("_decision_index_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # `@dataclass` looks its module up by name
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[spec.name]
        raise
    return module


@pytest.fixture(scope="module")
def di() -> ModuleType:
    return _load()


def _write(root: Path, rel: str, text: str | bytes) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(text, bytes):
        path.write_bytes(text)
    else:
        path.write_text(text, encoding="utf-8", newline="")
    return path


def _one(di: ModuleType, root: Path, rel: str, text: str | bytes) -> Any:
    """Scan one file and return its only anchor."""
    _write(root, rel, text)
    found = di.scan(root)
    assert len(found) == 1, found
    return found[0]


def _paths(found: list[Any]) -> list[str]:
    return [a.path for a in found]


# --------------------------------------------------------------------------- grammar


class TestAnchorGrammar:
    """`# DECISION [plan-id/]D-NNN [STALE]` exactly as the planner reads it."""

    def test_a_modern_anchor_has_every_field(self, di, tmp_path):
        a = _one(di, tmp_path, "src/a.py", f"{A} {P1}/D-007 keep the lock order\n")
        assert a.token == f"{P1}/D-007"
        assert a.plan_id == P1
        assert a.number == 7
        assert a.stale is False
        assert a.path == "src/a.py"
        assert a.key == (P1, 7)

    def test_a_legacy_plan_id_is_a_plan_id(self, di, tmp_path):
        a = _one(di, tmp_path, "src/a.py", f"{A} {LEGACY}/D-002 old rule\n")
        assert a.plan_id == LEGACY and a.number == 2

    def test_stale_is_read_after_the_token(self, di, tmp_path):
        a = _one(di, tmp_path, "src/a.py", f"{A} {P1}/D-007 [STALE]: gone\n")
        assert a.stale is True
        assert a.text == "gone"

    def test_stale_needs_the_bracketed_marker_next_to_the_token(self, di, tmp_path):
        a = _one(di, tmp_path, "src/a.py", f"{A} {P1}/D-007: the STALE rule\n")
        assert a.stale is False

    def test_the_number_has_three_or_more_digits_and_keeps_its_value(
        self, di, tmp_path
    ):
        _write(tmp_path, "src/a.py", f"{A} {P1}/D-1000 four digits\n")
        _write(tmp_path, "src/b.py", f"{A} {P1}/D-0006 padded\n")
        _write(tmp_path, "src/c.py", f"{A} {P1}/D-12 too short\n")
        _write(tmp_path, "src/d.py", f"{A} {P1}/D-1234x glued letter\n")
        found = di.scan(tmp_path)
        assert [(a.path, a.number) for a in found] == [
            ("src/a.py", 1000),
            ("src/b.py", 6),
        ]

    def test_a_number_longer_than_int_reads_is_skipped_not_a_crash(self, di, tmp_path):
        _write(tmp_path, "src/a.py", f"{A} {P1}/D-{'9' * 5000}: x\n")
        _write(tmp_path, "src/b.py", f"{A} {P1}/D-{'9' * 30}: ok\n")
        assert [(a.path, a.number) for a in di.scan(tmp_path)] == [
            ("src/b.py", 10**30 - 1)
        ]

    @pytest.mark.parametrize(
        "line",
        [
            f"#DECISION {P1}/D-001 no space after the hash\n",
            f"#{A} {P1}/D-001 two hashes\n",
            f"// DECISION {P1}/D-001 slash comment in a python file\n",
            f"x = '# DECISIONS {P1}/D-001'\n",
            f"# decision {P1}/D-001 lower case\n",
            f"# DECISION: {P1}/D-001 a colon before the token\n",
            f"{A} {P1}/D-001x glued letter\n",
            f"{A} {P1} / D-001 spaces around the slash\n",
            f"# see DECISION for {P1}/D-001\n",
            f"[{P1}/iter-1/step-2] D-001 a commit tag, not an anchor\n",
        ],
    )
    def test_text_that_is_not_the_grammar_is_not_an_anchor(self, di, tmp_path, line):
        _write(tmp_path, "src/a.py", line)
        assert di.scan(tmp_path) == []

    def test_a_token_in_a_string_or_docstring_is_not_a_comment_anchor_unless_it_has_the_hash(
        self, di, tmp_path
    ):
        _write(
            tmp_path,
            "src/a.py",
            f'"""T1 (DECISION {P1}/D-013): prose in a docstring."""\n'
            f"x = 'DECISION {P1}/D-014'\n",
        )
        assert di.scan(tmp_path) == []

    @pytest.mark.parametrize("end", ["\n", ":\n", ": rule\n", " rule\n", "\t\n", ""])
    def test_the_token_may_end_at_a_colon_space_or_the_line_end(
        self, di, tmp_path, end
    ):
        a = _one(di, tmp_path, "src/a.py", f"{A} {P1}/D-007" + end)
        assert a.number == 7

    def test_an_anchor_may_trail_code_but_a_line_holds_one(self, di, tmp_path):
        _write(
            tmp_path,
            "src/a.py",
            f"x = 1  {A} {P1}/D-001 first {A} {P2}/D-002 second\n",
        )
        found = di.scan(tmp_path)
        assert [a.number for a in found] == [1]

    def test_indented_anchors_count(self, di, tmp_path):
        _write(tmp_path, "src/a.py", f"def f():\n    {A} {P1}/D-003 why\n    pass\n")
        assert [a.number for a in di.scan(tmp_path)] == [3]

    def test_crlf_files_are_read(self, di, tmp_path):
        a = _one(di, tmp_path, "src/a.py", f"{A} {P1}/D-003 first\r\n# more\r\n")
        assert a.number == 3 and a.text == "first\nmore"

    @pytest.mark.parametrize(
        ("rel", "line"),
        [
            ("src/x.js", f"// DECISION {P1}/D-004 slash family\n"),
            ("src/x.ts", f"// DECISION {P1}/D-004 slash family\n"),
            ("src/x.go", f"// DECISION {P1}/D-004 slash family\n"),
            ("src/x.sql", f"-- DECISION {P1}/D-004 dash family\n"),
            ("src/x.sh", f"{A} {P1}/D-004 hash family\n"),
            ("src/x.yml", f"{A} {P1}/D-004 hash family\n"),
            ("src/x.toml", f"{A} {P1}/D-004 hash family\n"),
            ("src/x.js", f"/* DECISION {P1}/D-004 block family */\n"),
            ("src/x.md", f"<!-- DECISION {P1}/D-004 html family -->\n"),
        ],
    )
    def test_each_comment_family_the_planner_reads(self, di, tmp_path, rel, line):
        a = _one(di, tmp_path, rel, line)
        assert a.number == 4 and a.plan_id == P1

    @pytest.mark.parametrize(
        ("rel", "line"),
        [
            ("src/x.py", f"// DECISION {P1}/D-004 slash in a hash family file\n"),
            ("src/x.js", f"{A} {P1}/D-004 hash in a slash family file\n"),
            ("src/x.md", f"{A} {P1}/D-004 a markdown heading, not a comment\n"),
            ("src/x.md", f"`<!-- DECISION {P1}/D-004 -->` in a code span\n"),
            ("src/x.md", f"```\n<!-- DECISION {P1}/D-004 -->\n```\n"),
            ("src/x.md", f"<!-- DECISION {P1}/D-004 never closed\n"),
            ("src/x.json", f'{{"a": "{A} {P1}/D-004"}}\n'),
            ("src/x.png", f"{A} {P1}/D-004\n"),
            ("src/x.pyc", f"{A} {P1}/D-004\n"),
            ("src/x", f"{A} {P1}/D-004\n"),
        ],
    )
    def test_the_wrong_family_or_a_non_text_suffix_is_not_scanned(
        self, di, tmp_path, rel, line
    ):
        _write(tmp_path, rel, line)
        assert di.scan(tmp_path) == []

    def test_a_html_comment_holds_several_anchors(self, di, tmp_path):
        _write(
            tmp_path,
            "src/x.md",
            f"<!--\nDECISION {P1}/D-001 one\nDECISION {P2}/D-002 two\n-->\n",
        )
        assert [a.number for a in di.scan(tmp_path)] == [1, 2]

    def test_the_root_files_are_hash_comment_files_whatever_their_suffix(
        self, di, tmp_path
    ):
        _write(tmp_path, "constraints.txt", f"{A} {P1}/D-010: one pin\n")
        _write(tmp_path, "Makefile", f"{A} {P1}/D-011: a target\n")
        _write(tmp_path, "tox.ini", f"{A} {P1}/D-012: an env\n")
        _write(tmp_path, "pyproject.toml", f"{A} {P1}/D-013: a table\n")
        found = di.scan(tmp_path)
        assert [(a.path, a.number) for a in found] == [
            ("Makefile", 11),
            ("constraints.txt", 10),
            ("pyproject.toml", 13),
            ("tox.ini", 12),
        ]

    def test_a_txt_file_inside_a_scanned_directory_is_not_a_hash_comment_file(
        self, di, tmp_path
    ):
        _write(tmp_path, "scripts/notes.txt", f"{A} {P1}/D-010: prose\n")
        assert di.scan(tmp_path) == []


class TestBadPrefix:
    """The planner's loose pass: `[^\\s/]+` where a plan id belongs, not a plan id."""

    @pytest.mark.parametrize(
        "prefix",
        [
            "plan-2026-07-18-80b0bd4d",  # the commit-tag form of the id
            "plan-2026-11-12-d454fd62",
            "foo",
            "plan-2026-09-19T175721-21cd7f8",  # a 7 hex tail
            "plan-2026-09-19T175721-21cd7f8e1",  # a 9 hex tail
            "plan-2026-09-19T175721-21cd7f8e-x",  # trailing text
            "xplan-2026-09-19T175721-21cd7f8e",  # leading text
            "plan-2026-09-19T175721-21cd7f8e.",
            "plan-2026-09-19T175721-21CD7F8E",  # upper case hex
            "plan_2026-05-29_0c00a59",  # a short legacy tail
            "plan-2026-09-19T17572-21cd7f8e",  # a 5 digit time
            "Plan-2026-09-19T175721-21cd7f8e",
        ],
    )
    def test_a_prefix_that_is_not_a_plan_id_is_kept_with_no_plan_id(
        self, di, tmp_path, prefix
    ):
        a = _one(di, tmp_path, "tests/t.py", f"{A} {prefix}/D-013: text\n")
        assert a.prefix == prefix
        assert a.plan_id is None
        assert a.key == (None, 13)
        assert di.state_of(a, {}) == "badprefix"

    def test_a_bare_decision_number_has_no_plan_id(self, di, tmp_path):
        a = _one(di, tmp_path, "src/a.py", f"{A} D-013: bare\n")
        assert a.prefix is None and a.plan_id is None and a.token == "D-013"
        assert di.state_of(a, {}) == "badprefix"

    def test_a_bad_prefix_stays_badprefix_even_with_stale_or_a_matching_number(
        self, di, tmp_path
    ):
        a = _one(di, tmp_path, "src/a.py", f"{A} foo/D-013 [STALE]: x\n")
        entry = di.ManifestEntry(P1, 13, "2026-01-01", "r")
        assert di.state_of(a, {(P1, 13): entry}) == "badprefix"

    def test_a_prefix_with_a_slash_inside_is_no_anchor_at_all(self, di, tmp_path):
        _write(tmp_path, "src/a.py", f"{A} a/b/D-001: two slashes\n")
        assert di.scan(tmp_path) == []

    def test_a_valid_prefix_is_never_badprefix(self, di, tmp_path):
        a = _one(di, tmp_path, "src/a.py", f"{A} {P1}/D-013: ok\n")
        assert di.state_of(a, {}) != "badprefix"


# --------------------------------------------------------------------------- manifest


def _manifest(tmp_path: Path, text: str | bytes) -> Path:
    return _write(tmp_path, "plans/ANCHORS.md", text)


class TestManifestParser:
    def test_the_header_and_blank_lines_are_ignored(self, di, tmp_path):
        path = _manifest(
            tmp_path,
            "# Decision Anchor Manifest\n"
            "*Format: `<plan-id>/D-NNN | YYYY-MM-DD | one-line rationale`.*\n\n"
            f"{P1}/D-001 | 2026-11-10 | why one\n",
        )
        manifest = di.load_manifest(path)
        assert list(manifest) == [(P1, 1)]
        entry = manifest[(P1, 1)]
        assert (entry.plan_id, entry.number) == (P1, 1)
        assert (entry.date, entry.rationale) == ("2026-11-10", "why one")

    def test_a_pipe_inside_the_rationale_stays_in_the_rationale(self, di, tmp_path):
        path = _manifest(
            tmp_path,
            f"{P1}/D-001 | 2026-11-10 | a | b |  c | d\n",
        )
        entry = di.load_manifest(path)[(P1, 1)]
        assert entry.date == "2026-11-10"
        assert entry.rationale == "a | b |  c | d"

    def test_angle_brackets_and_escapes_survive_byte_for_byte(self, di, tmp_path):
        rationale = (
            r"shown to Pass 2 in a <response> block, \<escaped\> &lt;x&gt; `<a|b>`"
        )
        path = _manifest(tmp_path, f"{P1}/D-001 | 2026-11-10 | {rationale}\n")
        assert di.load_manifest(path)[(P1, 1)].rationale == rationale

    def test_legacy_ids_and_padding(self, di, tmp_path):
        path = _manifest(
            tmp_path,
            f"{LEGACY}/D-002 | 2026-05-29 | legacy\n{P2}/D-0006 | 2026-09-19 | padded\n",
        )
        assert set(di.load_manifest(path)) == {(LEGACY, 2), (P2, 6)}

    def test_whitespace_before_the_first_pipe_is_allowed(self, di, tmp_path):
        path = _manifest(
            tmp_path,
            f"{P1}/D-001\t |2026-11-10| tight\n{P1}/D-002|2026-11-10| none\n",
        )
        manifest = di.load_manifest(path)
        assert manifest[(P1, 1)].rationale == "tight"
        assert manifest[(P1, 2)].rationale == "none"

    @pytest.mark.parametrize(
        "line",
        [
            f" {P1}/D-001 | 2026-11-10 | indented",
            f"- {P1}/D-001 | 2026-11-10 | a list item",
            f"{P1}/D-001: 2026-11-10 | colon, no pipe",
            f"{P1}/D-001 2026-11-10 | no pipe after the token",
            f"{P1}/D-01 | 2026-11-10 | two digits",
            f"{P1}/D-001x | 2026-11-10 | glued letter",
            "plan-2026-11-10-2cefeb61/D-001 | 2026-11-10 | commit tag form",
            f"see {P1}/D-001 | 2026-11-10 | prose first",
            "D-001 | 2026-11-10 | bare",
        ],
    )
    def test_a_line_that_is_not_a_manifest_line_is_not_an_entry(
        self, di, tmp_path, line
    ):
        assert di.load_manifest(_manifest(tmp_path, line + "\n")) == {}

    def test_a_line_with_no_date_or_rationale_still_registers_the_key(
        self, di, tmp_path
    ):
        # the planner resolves on `<plan-id>/D-NNN[ \t]*|` alone
        path = _manifest(tmp_path, f"{P1}/D-001 |\n{P1}/D-002 | 2026-11-10\n")
        manifest = di.load_manifest(path)
        assert manifest[(P1, 1)].rationale == ""
        assert manifest[(P1, 2)].date == "2026-11-10"
        assert manifest[(P1, 2)].rationale == ""

    def test_the_first_line_of_a_duplicate_key_wins(self, di, tmp_path):
        path = _manifest(
            tmp_path,
            f"{P1}/D-001 | 2026-11-10 | first\n{P1}/D-001 | 2026-11-11 | second\n",
        )
        assert di.load_manifest(path)[(P1, 1)].rationale == "first"

    def test_crlf_and_unicode_line_separators_do_not_split_a_line(self, di, tmp_path):
        text = (
            f"{P1}/D-001 | 2026-11-10 | a\u2028b\x0bc\x0cd\u0085e\r\n"
            f"{P1}/D-002 | 2026-11-10 | next\r\n"
        )
        manifest = di.load_manifest(_manifest(tmp_path, text))
        assert manifest[(P1, 1)].rationale == "a\u2028b\x0bc\x0cd\u0085e"
        assert manifest[(P1, 2)].rationale == "next"

    def test_a_missing_manifest_is_empty_not_an_error(self, di, tmp_path):
        assert di.load_manifest(tmp_path / "plans" / "ANCHORS.md") == {}

    def test_a_non_utf8_manifest_does_not_crash(self, di, tmp_path):
        data = f"{P1}/D-001 | 2026-11-10 | caf".encode() + b"\xe9 \xff\xfe\n"
        manifest = di.load_manifest(_manifest(tmp_path, data))
        assert (P1, 1) in manifest

    def test_the_result_order_is_the_file_order(self, di, tmp_path):
        path = _manifest(
            tmp_path,
            f"{P2}/D-009 | 2026-09-19 | b\n{P1}/D-001 | 2026-11-10 | a\n",
        )
        assert list(di.load_manifest(path)) == [(P2, 9), (P1, 1)]


# --------------------------------------------------------------------------- classify


def _entry(di: ModuleType, plan: str, number: int) -> Any:
    return di.ManifestEntry(plan, number, "2026-01-01", "r")


class TestClassify:
    def _tree(self, tmp_path: Path) -> None:
        _write(
            tmp_path,
            "src/a.py",
            f"{A} {P1}/D-001 in the manifest\n"
            f"{A} {P1}/D-002 [STALE]: gone, no line\n"
            f"{A} {P1}/D-003 live, no line\n"
            f"{A} plan-2026-07-18-80b0bd4d/D-004 commit tag form\n"
            f"{A} D-005 bare\n"
            f"{A} {P2}/D-006 [STALE]: stale but in the manifest\n"
            f"{A} {LEGACY}/D-007 [STALE]: legacy, no line\n"
            f"{A} {LEGACY}/D-008: legacy in the manifest\n",
        )

    def _manifest(self, di: ModuleType) -> dict[Any, Any]:
        return {
            (P1, 1): _entry(di, P1, 1),
            (P2, 6): _entry(di, P2, 6),
            (LEGACY, 8): _entry(di, LEGACY, 8),
        }

    def test_each_state(self, di, tmp_path):
        self._tree(tmp_path)
        got = di.classify(di.scan(tmp_path), self._manifest(di))
        assert {
            state: [a.number for a in anchors] for state, anchors in got.items()
        } == {
            "resolved": [1, 6, 8],
            "retired": [2, 7],
            "unresolved": [3],
            "badprefix": [4, 5],
        }

    def test_a_stale_anchor_with_a_manifest_line_is_resolved_not_retired(
        self, di, tmp_path
    ):
        a = _one(di, tmp_path, "src/a.py", f"{A} {P2}/D-006 [STALE]: x\n")
        assert di.state_of(a, {(P2, 6): _entry(di, P2, 6)}) == "resolved"
        assert di.state_of(a, {}) == "retired"

    def test_the_manifest_key_is_the_plan_id_and_the_number_together(
        self, di, tmp_path
    ):
        a = _one(di, tmp_path, "src/a.py", f"{A} {P1}/D-001: x\n")
        other_plan = {(P2, 1): _entry(di, P2, 1)}
        other_number = {(P1, 2): _entry(di, P1, 2)}
        assert di.state_of(a, other_plan) == "unresolved"
        assert di.state_of(a, other_number) == "unresolved"

    def test_a_padded_number_resolves_against_the_same_value(self, di, tmp_path):
        a = _one(di, tmp_path, "src/a.py", f"{A} {P1}/D-0006: padded\n")
        assert di.state_of(a, {(P1, 6): _entry(di, P1, 6)}) == "resolved"

    def test_the_result_names_every_state_in_a_fixed_order(self, di):
        assert tuple(di.classify([], {})) == (
            "resolved",
            "retired",
            "unresolved",
            "badprefix",
        )
        assert di.STATES == ("resolved", "retired", "unresolved", "badprefix")

    def test_a_state_list_keeps_the_scan_order(self, di, tmp_path):
        _write(tmp_path, "src/b.py", f"{A} {P1}/D-020: b\n")
        _write(
            tmp_path,
            "src/a.py",
            f"{A} {P1}/D-010: a\n{A} {P1}/D-009: a2\n",
        )
        got = di.classify(di.scan(tmp_path), {})
        assert [(a.path, a.number) for a in got["unresolved"]] == [
            ("src/a.py", 10),
            ("src/a.py", 9),
            ("src/b.py", 20),
        ]

    def test_every_anchor_lands_in_exactly_one_state(self, di, tmp_path):
        self._tree(tmp_path)
        found = di.scan(tmp_path)
        got = di.classify(found, self._manifest(di))
        assert sum(len(v) for v in got.values()) == len(found)


# --------------------------------------------------------------------------- walk


class TestWalk:
    def test_the_roots_are_the_ones_the_plan_names(self, di):
        assert tuple(di.SCAN_DIRS) == ("src", "tests", "scripts", ".github")
        assert tuple(di.SCAN_FILES) == (
            "pyproject.toml",
            "constraints.txt",
            "Makefile",
            "tox.ini",
        )

    def test_only_the_named_roots_are_read(self, di, tmp_path):
        line = f"{A} {P1}/D-001: x\n"
        for rel in (
            "src/a.py",
            "tests/a.py",
            "scripts/a.py",
            ".github/workflows/ci.yml",
            "pyproject.toml",
            "docs/a.py",
            "examples/a.py",
            "evaluation/a.py",
            "other.py",
            "setup.py",
            "images/a.py",
        ):
            _write(tmp_path, rel, line)
        assert _paths(di.scan(tmp_path)) == [
            ".github/workflows/ci.yml",
            "pyproject.toml",
            "scripts/a.py",
            "src/a.py",
            "tests/a.py",
        ]

    @pytest.mark.parametrize(
        "skipped",
        [
            "src/plans/a.py",
            "src/build/a.py",
            "src/dist/a.py",
            "src/node_modules/a.py",
            "src/__pycache__/a.py",
            "src/fsm_llm.egg-info/a.py",
            "src/pkg/x.egg-info/a.py",
            "src/.hidden/a.py",
            "src/pkg/.git/a.py",
            "src/.hidden.py",
            "tests/.cache/a.py",
        ],
    )
    def test_skipped_directories_and_dot_entries(self, di, tmp_path, skipped):
        _write(tmp_path, "src/keep.py", f"{A} {P1}/D-001: x\n")
        _write(tmp_path, skipped, f"{A} {P1}/D-002: x\n")
        assert _paths(di.scan(tmp_path)) == ["src/keep.py"]

    def test_names_that_only_look_like_skipped_ones_are_read(self, di, tmp_path):
        for rel in ("src/plans_old/a.py", "src/mybuild/a.py", "src/build.py"):
            _write(tmp_path, rel, f"{A} {P1}/D-001: x\n")
        assert len(di.scan(tmp_path)) == 3

    def test_the_depth_limit_is_twelve_levels_below_the_project_root(
        self, di, tmp_path
    ):
        line = f"{A} {P1}/D-001: x\n"
        deep12 = "/".join(["src"] + [f"d{i}" for i in range(11)])  # depth 12
        deep13 = "/".join(["src"] + [f"d{i}" for i in range(12)])  # depth 13
        _write(tmp_path, f"{deep12}/ok.py", line)
        _write(tmp_path, f"{deep13}/too_deep.py", line)
        assert _paths(di.scan(tmp_path)) == [f"{deep12}/ok.py"]

    def test_the_depth_limit_counts_for_every_scan_directory(self, di, tmp_path):
        line = f"{A} {P1}/D-001: x\n"
        deep13 = "/".join(["tests"] + [f"d{i}" for i in range(12)])
        _write(tmp_path, f"{deep13}/a.py", line)
        _write(
            tmp_path, ".github/" + "/".join(f"d{i}" for i in range(12)) + "/a.yml", line
        )
        assert di.scan(tmp_path) == []

    def test_a_symlink_is_never_followed(self, di, tmp_path):
        _write(tmp_path, "src/real.py", f"{A} {P1}/D-001: x\n")
        outside = tmp_path.parent / (tmp_path.name + "_outside")
        _write(outside, "a.py", f"{A} {P1}/D-002: outside\n")
        symlink_or_skip(tmp_path / "src" / "linkdir", outside)
        symlink_or_skip(tmp_path / "src" / "link.py", outside / "a.py")
        assert _paths(di.scan(tmp_path)) == ["src/real.py"]

    def test_a_missing_root_or_directory_is_an_empty_scan(self, di, tmp_path):
        assert di.scan(tmp_path / "nope") == []
        assert di.scan(tmp_path) == []

    def test_a_directory_named_like_a_root_file_does_not_crash(self, di, tmp_path):
        (tmp_path / "Makefile").mkdir()
        (tmp_path / "tox.ini").write_text(f"{A} {P1}/D-001: x\n", encoding="utf-8")
        assert _paths(di.scan(tmp_path)) == ["tox.ini"]

    def test_paths_are_posix_and_relative_to_the_root(self, di, tmp_path):
        _write(tmp_path, "src/pkg/sub/a.py", f"{A} {P1}/D-001: x\n")
        (path,) = _paths(di.scan(tmp_path))
        assert path == "src/pkg/sub/a.py"
        assert "\\" not in path and not path.startswith(("/", "./"))

    def test_a_non_utf8_file_is_read_with_replacement(self, di, tmp_path):
        a = _one(
            di,
            tmp_path,
            "src/a.py",
            b"\xff\xfe\x00 garbage \x80\n"
            + A.encode()
            + b" "
            + P1.encode()
            + b"/D-001: caf\xe9\n",
        )
        assert a.number == 1 and a.scope == "-"

    def test_a_binary_python_file_does_not_crash(self, di, tmp_path):
        _write(tmp_path, "src/blob.py", bytes(range(256)) * 8)
        _write(tmp_path, "src/nul.py", b"x = 1\x00\n")
        _write(tmp_path, "src/ok.py", f"{A} {P1}/D-001: x\n")
        assert _paths(di.scan(tmp_path)) == ["src/ok.py"]

    def test_a_deeply_nested_expression_does_not_crash_the_ast_pass(self, di, tmp_path):
        text = "x = " + "(" * 400 + "1" + ")" * 400 + f"\n{A} {P1}/D-001: x\n"
        a = _one(di, tmp_path, "src/a.py", text)
        assert a.scope == "-"

    @posix_only("a mode of 0 is not enforced on Windows")
    def test_an_unreadable_file_is_skipped(self, di, tmp_path):
        skip_if_root()
        path = _write(tmp_path, "src/locked.py", f"{A} {P1}/D-001: x\n")
        _write(tmp_path, "src/ok.py", f"{A} {P1}/D-002: x\n")
        path.chmod(0)
        try:
            assert _paths(di.scan(tmp_path)) == ["src/ok.py"]
        finally:
            path.chmod(0o644)


class TestDeterminism:
    FILES = (
        "src/z.py",
        "src/a.py",
        "src/pkg/m.py",
        "src/pkg/b.py",
        "tests/t.py",
        "scripts/s.py",
        "pyproject.toml",
        ".github/workflows/w.yml",
    )

    def _build(self, root: Path, order: tuple[str, ...]) -> None:
        for rel in order:
            _write(
                root,
                rel,
                f"{A} {P1}/D-001: first\n{A} {P2}/D-002: second\n",
            )

    def test_two_scans_of_one_tree_are_identical(self, di, tmp_path):
        self._build(tmp_path, self.FILES)
        assert di.scan(tmp_path) == di.scan(tmp_path)

    def test_creation_order_does_not_matter(self, di, tmp_path):
        one, two = tmp_path / "one", tmp_path / "two"
        self._build(one, self.FILES)
        self._build(two, tuple(reversed(self.FILES)))
        assert di.scan(one) == di.scan(two)

    def test_the_order_is_by_path_then_file_order_whatever_the_directory_order(
        self, di, tmp_path, monkeypatch
    ):
        self._build(tmp_path, self.FILES)
        expected = [(a.path, a.number) for a in di.scan(tmp_path)]
        real = os.scandir

        class _Reversed:
            def __init__(self, path: Any) -> None:
                with real(path) as it:
                    self._entries = list(it)[::-1]

            def __enter__(self) -> list[Any]:
                return self._entries

            def __exit__(self, *exc: object) -> None:
                return None

            def __iter__(self) -> Any:
                return iter(self._entries)

        monkeypatch.setattr(os, "scandir", _Reversed)
        assert [(a.path, a.number) for a in di.scan(tmp_path)] == expected
        paths = [p for p, _ in expected]
        assert paths == sorted(paths)
        # the scan order is the path order; inside one file the order of the text
        assert [n for p, n in expected if p == "src/a.py"] == [1, 2]

    def test_the_report_is_identical_between_runs(self, tmp_path):
        self._build(tmp_path, self.FILES)
        runs = [
            subprocess.run(
                [sys.executable, str(SCRIPT), "--root", str(tmp_path), "--report"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                cwd=tmp_path,
                timeout=120,
                check=False,
            )
            for _ in range(2)
        ]
        assert runs[0].returncode == 0, runs[0].stderr
        assert runs[0].stdout == runs[1].stdout


# --------------------------------------------------------------------------- scope and text


class TestEnclosingDefinition:
    SRC = (
        f"{A} {P1}/D-001 module level\n"
        "class Outer:\n"
        f"    {A} {P1}/D-002 in the class body\n"
        "    class Inner:\n"
        "        def method(self):\n"
        f"            {A} {P1}/D-003 nested class method\n"
        "            def helper():\n"
        f"                {A} {P1}/D-004 nested function\n"
        "                return 1\n"
        "            return helper\n"
        "    def sibling(self):\n"
        f"        x = 1  {A} {P1}/D-005 trailing\n"
        "        return x\n"
        "\n"
        "def top():\n"
        f"    {A} {P1}/D-006 plain function\n"
        "    return 2\n"
        "\n"
        "async def coroutine():\n"
        f"    {A} {P1}/D-007 async\n"
        "    return 3\n"
        "\n"
        f"{A} {P1}/D-008 above a def is outside it\n"
        "def after():\n"
        "    return 4\n"
        "\n"
        "@decorator(\n"
        "    arg=1,\n"
        ")\n"
        "def decorated():\n"
        f"    {A} {P1}/D-009 decorated\n"
        "    return 5\n"
        "\n"
        "class WithDoc:\n"
        '    """Doc.\n'
        f"    {A} {P1}/D-010 inside a docstring line of the class\n"
        '    """\n'
    )

    def test_the_innermost_definition_names_the_scope(self, di, tmp_path):
        _write(tmp_path, "src/a.py", self.SRC)
        scopes = {a.number: a.scope for a in di.scan(tmp_path)}
        assert scopes == {
            1: "-",
            2: "Outer",
            3: "Outer.Inner.method",
            4: "Outer.Inner.method.helper",
            5: "Outer.sibling",
            6: "top",
            7: "coroutine",
            8: "-",
            9: "decorated",
            10: "WithDoc",
        }

    def test_a_python_file_that_does_not_parse_has_scope_dash(self, di, tmp_path):
        a = _one(
            di,
            tmp_path,
            "src/a.py",
            f"def broken(:\n    {A} {P1}/D-001: x\n",
        )
        assert a.scope == "-"

    def test_a_markdown_anchor_is_named_by_the_nearest_preceding_heading(
        self, di, tmp_path
    ):
        _write(
            tmp_path,
            "src/AGENTS.md",
            "# Top\n<!-- DECISION " + P1 + "/D-001 first -->\n"
            "## Second `section`\ntext\n<!-- DECISION " + P1 + "/D-002 second -->\n",
        )
        assert [a.scope for a in di.scan(tmp_path)] == ["Top", "Second `section`"]

    def test_a_markdown_anchor_before_any_heading_has_scope_dash(self, di, tmp_path):
        a = _one(di, tmp_path, "src/a.md", f"<!-- DECISION {P1}/D-001 x -->\n# Late\n")
        assert a.scope == "-"

    def test_a_toml_anchor_is_named_by_its_table(self, di, tmp_path):
        text = (
            f"{A} {P1}/D-001 before any table\n"
            "[project]\nname = 'x'\n"
            f"    {A} {P1}/D-002 in the project table\n"
            "[tool.ruff.lint]\n"
            f"{A} {P1}/D-003 in the lint table\n"
        )
        _write(tmp_path, "pyproject.toml", text)
        assert [a.scope for a in di.scan(tmp_path)] == [
            "-",
            "project",
            "tool.ruff.lint",
        ]

    def test_a_makefile_anchor_is_named_by_its_target(self, di, tmp_path):
        _write(
            tmp_path,
            "Makefile",
            f"{A} {P1}/D-001 before a target\n"
            "test: ## run\n\tpytest\n"
            f"{A} {P1}/D-002 after test\n"
            "lint-imports:\n\tlint-imports\n",
        )
        assert [a.scope for a in di.scan(tmp_path)] == ["-", "test"]

    def test_other_files_have_scope_dash(self, di, tmp_path):
        _write(tmp_path, "constraints.txt", f"{A} {P1}/D-001: x\n")
        _write(tmp_path, ".github/w.yml", f"{A} {P1}/D-002: x\n")
        assert [a.scope for a in di.scan(tmp_path)] == ["-", "-"]


class TestCommentBlock:
    def test_the_text_is_the_contiguous_comment_run_without_the_head(
        self, di, tmp_path
    ):
        a = _one(
            di,
            tmp_path,
            "src/a.py",
            f"x = 0\n{A} {P1}/D-001: keep the order.\n"
            "# Do NOT sort here.\n#   indented line\n#\n# after a bare hash\n"
            "y = 1\n# not part of it\n",
        )
        assert a.text == (
            "keep the order.\nDo NOT sort here.\n  indented line\n\nafter a bare hash"
        )

    @pytest.mark.parametrize("sep", [" ", ": ", " - ", " \u2014 ", " \u2013 "])
    def test_the_separator_after_the_token_is_dropped(self, di, tmp_path, sep):
        a = _one(di, tmp_path, "src/a.py", f"{A} {P1}/D-001{sep}the rule\n")
        assert a.text == "the rule"

    def test_a_head_only_first_line_gives_the_next_lines(self, di, tmp_path):
        a = _one(di, tmp_path, "src/a.py", f"{A} {P1}/D-001\n# the rule\n")
        assert a.text == "the rule"

    def test_an_anchor_with_no_text_has_empty_text(self, di, tmp_path):
        a = _one(di, tmp_path, "src/a.py", f"{A} {P1}/D-001 [STALE]\nx = 1\n")
        assert a.text == ""

    def test_the_run_stops_at_blank_line_code_or_the_next_anchor(self, di, tmp_path):
        _write(
            tmp_path,
            "src/a.py",
            f"{A} {P1}/D-001: one\n# more one\n"
            f"{A} {P2}/D-002: two\n# more two\n\n# stray comment\n",
        )
        one, two = di.scan(tmp_path)
        assert one.text == "one\nmore one"
        assert two.text == "two\nmore two"

    def test_an_indented_run_keeps_relative_indentation(self, di, tmp_path):
        a = _one(
            di,
            tmp_path,
            "src/a.py",
            f"def f():\n    {A} {P1}/D-001: rule\n    #   nested\n    return 1\n",
        )
        assert a.text == "rule\n  nested"

    def test_a_trailing_comment_continues_on_full_comment_lines_only(
        self, di, tmp_path
    ):
        a = _one(
            di,
            tmp_path,
            "src/a.py",
            f"x = 1  {A} {P1}/D-001: why\n# and more\ny = 2  # unrelated\n",
        )
        assert a.text == "why\nand more"

    def test_a_block_and_a_html_comment_give_their_inner_lines(self, di, tmp_path):
        _write(
            tmp_path,
            "src/a.js",
            f"/*\n * DECISION {P1}/D-001: the rule\n * second line\n */\n",
        )
        _write(
            tmp_path,
            "src/a.md",
            f"<!--\nDECISION {P2}/D-002: html rule\nsecond\n-->\n",
        )
        got = {a.number: a.text for a in di.scan(tmp_path)}
        assert got == {1: "the rule\nsecond line", 2: "html rule\nsecond"}

    def test_a_slash_comment_run(self, di, tmp_path):
        a = _one(
            di,
            tmp_path,
            "src/a.js",
            f"// DECISION {P1}/D-001: rule\n// second\nlet x = 1;\n",
        )
        assert a.text == "rule\nsecond"


class TestNoLineNumbers:
    def test_no_record_carries_a_line_number(self, di):
        fields = {f for f in di.Anchor.__dataclass_fields__}
        assert fields == {
            "token",
            "prefix",
            "plan_id",
            "number",
            "stale",
            "path",
            "scope",
            "text",
        }
        assert not any("line" in f or "offset" in f or "lineno" in f for f in fields)
        assert not any("line" in f for f in di.ManifestEntry.__dataclass_fields__), (
            "the manifest entry is plan, number, date and rationale"
        )

    def test_the_report_names_no_file_line(self, tmp_path):
        _write(tmp_path, "src/a.py", f"{A} {P1}/D-001: x\n")
        out = _run(tmp_path, "--report").stdout
        assert ".py:" not in out and "line" not in out.lower()


# --------------------------------------------------------------------------- CLI


def _run(root: Path, *args: str, manifest: Path | None = None) -> Any:
    cmd = [sys.executable, str(SCRIPT), "--root", str(root), *args]
    if manifest is not None:
        cmd += ["--manifest", str(manifest)]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=root,
        timeout=120,
        check=False,
    )


class TestMain:
    def _tree(self, root: Path) -> None:
        _write(
            root,
            "src/a.py",
            f"{A} {P1}/D-001: in the manifest\n"
            f"{A} {P1}/D-002 [STALE]: gone\n"
            f"{A} {P1}/D-003: live\n"
            f"{A} {P2}/D-004: live too\n"
            f"{A} foo/D-005: typo\n",
        )
        _write(
            root,
            "plans/ANCHORS.md",
            f"# header\n{P1}/D-001 | 2026-11-10 | a\n{P2}/D-099 | 2026-09-19 | unused\n",
        )

    def test_report_prints_the_counts_and_exits_zero_even_with_problems(self, tmp_path):
        self._tree(tmp_path)
        proc = _run(tmp_path, "--report")
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.splitlines() == [
            "anchors: 5",
            "resolved: 1",
            "retired: 1",
            "unresolved: 2",
            "badprefix: 1",
            "distinct decisions: 4",
            "manifest entries: 2",
        ]
        assert proc.stderr == ""

    def test_the_manifest_flag_names_another_file(self, tmp_path):
        self._tree(tmp_path)
        other = _write(
            tmp_path,
            "elsewhere.md",
            f"{P1}/D-001 | 2026-11-10 | a\n{P1}/D-003 | 2026-11-10 | b\n{P2}/D-004 | 2026-09-19 | c\n",
        )
        out = _run(tmp_path, "--report", manifest=other).stdout.splitlines()
        assert out[1:5] == [
            "resolved: 3",
            "retired: 1",
            "unresolved: 0",
            "badprefix: 1",
        ]
        assert out[-1] == "manifest entries: 3"

    def test_the_default_manifest_is_plans_anchors_md_under_the_root(self, tmp_path):
        _write(tmp_path, "src/a.py", f"{A} {P1}/D-001: x\n")
        out = _run(tmp_path, "--report").stdout.splitlines()
        assert "unresolved: 1" in out and out[-1] == "manifest entries: 0"

    def test_without_a_mode_the_command_is_a_usage_error(self, tmp_path):
        proc = _run(tmp_path)
        assert proc.returncode == 2
        assert "--report" in proc.stderr

    def test_a_missing_root_is_one_stderr_line_and_exit_one(self, tmp_path):
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--root", str(tmp_path / "nope"), "--report"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=tmp_path,
            timeout=120,
            check=False,
        )
        assert proc.returncode == 1
        assert proc.stdout == "" and len(proc.stderr.strip().splitlines()) == 1

    def test_main_is_callable_in_process_and_returns_the_exit_code(
        self, di, tmp_path, capsys
    ):
        self._tree(tmp_path)
        code = di.main(["--root", str(tmp_path), "--report"])
        assert code == 0
        assert "anchors: 5" in capsys.readouterr().out

    def test_the_script_runs_with_no_src_on_the_path_and_imports_no_fsm_llm(
        self, tmp_path
    ):
        probe = (
            "import importlib.util, sys\n"
            "before = list(sys.path)\n"
            f"spec = importlib.util.spec_from_file_location('di', {str(SCRIPT)!r})\n"
            "m = importlib.util.module_from_spec(spec)\n"
            "sys.modules['di'] = m\n"
            "spec.loader.exec_module(m)\n"
            f"m.main(['--root', {str(tmp_path)!r}, '--report'])\n"
            "print('PATH_SAME', sys.path == before)\n"
            "print('FSM_LLM_LOADED', 'fsm_llm' in sys.modules)\n"
        )
        _write(tmp_path, "src/a.py", f"{A} {P1}/D-001: x\n")
        proc = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=tmp_path,
            timeout=120,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        assert "PATH_SAME True" in proc.stdout
        assert "FSM_LLM_LOADED False" in proc.stdout


# --------------------------------------------------------------------------- the script's own source


def _script_tree() -> ast.Module:
    return ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))


class TestScriptSource:
    def test_it_imports_the_standard_library_only(self):
        names: set[str] = set()
        for node in ast.walk(_script_tree()):
            if isinstance(node, ast.Import):
                names |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                assert node.level == 0, "no relative import in a script"
                names.add((node.module or "").split(".")[0])
        assert names, "the script imports nothing: the scan is vacuous"
        assert names <= set(sys.stdlib_module_names), sorted(
            names - set(sys.stdlib_module_names)
        )
        assert "fsm_llm" not in names

    def test_it_never_names_fsm_llm_in_code(self):
        # a string `import_module("fsm_llm...")` would pass the import scan above
        for node in ast.walk(_script_tree()):
            if isinstance(node, ast.Call):
                func = node.func
                name = (
                    func.attr
                    if isinstance(func, ast.Attribute)
                    else getattr(func, "id", "")
                )
                assert name not in {"__import__", "import_module", "find_spec"}

    def test_it_never_touches_sys_path(self):
        for node in ast.walk(_script_tree()):
            if isinstance(node, ast.Attribute) and node.attr == "path":
                assert not (
                    isinstance(node.value, ast.Name) and node.value.id == "sys"
                ), "sys.path is read or written"
            if isinstance(node, ast.ImportFrom) and node.module == "sys":
                assert "path" not in {a.name for a in node.names}

    def test_the_scanner_for_this_file_itself_is_not_vacuous(self):
        # `test_it_never_touches_sys_path` must see attributes at all
        assert any(isinstance(n, ast.Attribute) for n in ast.walk(_script_tree()))

    def test_every_text_open_names_an_encoding(self):
        for node in ast.walk(_script_tree()):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in {"read_text", "write_text"}:
                    assert any(k.arg == "encoding" for k in node.keywords)


# --------------------------------------------------------------------------- the real tree


def _expected_src_tokens() -> Counter[str]:
    data = json.loads(MULTISET.read_text(encoding="utf-8"))
    expected: Counter[str] = Counter(data["baseline"])
    for entry in data["reasons"]:
        expected[entry["token"]] += entry["delta"]
    return +expected


@pytest.fixture(scope="module")
def real(di) -> list[Any]:
    return di.scan(REPO)


@pytest.fixture(scope="module")
def real_manifest(di) -> dict[Any, Any]:
    if not MANIFEST.is_file():
        pytest.skip("plans/ANCHORS.md is not in this checkout")
    return di.load_manifest(MANIFEST)


class TestRealTree:
    """Loose numbers over the real repository (775 anchors at the time of the plan)."""

    #: the plans whose live anchors had no manifest line before step 99, and the open plan
    LIVE_PLANS = frozenset(
        {
            "plan-2026-07-21T045419-9925aa3a",
            "plan-2026-07-21T072826-e3131cc2",
            "plan-2026-07-21T082818-4c63deac",
            "plan-2026-07-21T191807-bf7ffe24",
            "plan-2026-07-22T114536-879d04a0",
            "plan-2026-09-29T184639-65baa765",
            "plan-2026-11-12T023759-9013b0e3",
            "plan-2026-11-12T073150-d454fd62",
        }
    )

    def test_the_anchor_count_is_of_the_order_the_survey_found(self, real):
        assert 700 <= len(real) <= 900, len(real)

    def test_anchors_are_found_in_the_package_the_tests_and_the_scripts(self, real):
        top = Counter(a.path.split("/")[0] for a in real)
        assert top["src"] > 600
        assert top["tests"] > 0 and top["scripts"] > 0
        assert any(a.path.startswith("src/fsm_llm/") for a in real)
        assert not any(
            a.path.startswith(("plans/", "docs/", "examples/")) for a in real
        )

    def test_the_src_anchors_are_a_sub_multiset_of_the_multiset_guard(self, real):
        """The two guards agree: no anchor is absent from `tests/fixtures/anchor_multiset.json`.

        The multiset test counts every textual mention of a token in `src/` (comments,
        docstrings, strings, prose), the scanner only the `# DECISION` comment anchors,
        so the scanner count is below the total and equality of totals cannot hold (at
        the time of this step: 766 anchors against 827 mentions, 61 mentions are
        docstring or prose references). What must hold is containment per token: an
        anchor the scanner finds that the guard does not know would be an unguarded
        rule.
        """
        scanned = Counter(
            f"{a.plan_id}/D-{a.number:03d}"
            for a in real
            if a.path.startswith("src/") and a.plan_id is not None
        )
        expected = _expected_src_tokens()
        extra = scanned - expected
        assert not extra, dict(list(extra.items())[:10])
        assert sum(scanned.values()) <= sum(expected.values())
        # not vacuous: most mentions are anchors
        assert sum(scanned.values()) > 0.8 * sum(expected.values())

    def test_every_real_anchor_has_a_plan_id_and_a_scope_or_a_dash(self, real):
        assert all(a.scope for a in real)
        assert all(a.path == a.path.replace("\\", "/") for a in real)
        assert {a.prefix is None for a in real} == {False}, "a bare anchor exists"

    def test_the_real_tree_has_every_state_but_few_live_unresolved(
        self, di, real, real_manifest
    ):
        got = di.classify(real, real_manifest)
        counts = {s: len(v) for s, v in got.items()}
        assert counts["resolved"] > 400, counts
        assert counts["retired"] > 50, counts
        assert counts["badprefix"] == 0, [a.token for a in got["badprefix"]]
        assert sum(counts.values()) == len(real)
        # Before step 99 the live anchors of six closed plans have no manifest line (26
        # anchors, 19 keys in the survey), and so have the open plan's own anchors (its
        # CLOSE appends them) and, in a checkout that lacks the five uncommitted lines,
        # plan 9013b0e3: `0 < unresolved <= 60`, all of them in the eight plans below.
        # Step 99 appends the manifest lines and this is rewritten there to the live
        # anchors of the open plan only.
        assert 0 < counts["unresolved"] <= 60, counts
        plans = {a.plan_id for a in got["unresolved"]}
        assert plans <= self.LIVE_PLANS, sorted(plans - self.LIVE_PLANS)

    def test_a_retired_anchor_is_stale_and_a_live_one_never_is(
        self, di, real, real_manifest
    ):
        got = di.classify(real, real_manifest)
        assert all(a.stale for a in got["retired"])
        assert not any(a.stale for a in got["unresolved"])

    def test_the_manifest_has_the_known_shape(self, real_manifest):
        assert len(real_manifest) > 350
        assert all(k[0].startswith(("plan-", "plan_")) for k in real_manifest)
        assert all(e.date[:2] == "20" for e in real_manifest.values())
