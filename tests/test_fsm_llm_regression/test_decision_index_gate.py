"""The generated decision index `docs/decisions/index.md` and its gate (roadmap row 0.5).

`scripts/decision_index.py` renders every `# DECISION` anchor of the repository into
`docs/decisions/index.md` and `--check` fails when that file is stale or when an anchor is
`unresolved` or has a `badprefix`. The index is GENERATED FROM THE ANCHORS: any commit that
adds, moves or re-words an anchor, or appends to `plans/ANCHORS.md`, must run
`make decision-index` or `TestCommittedIndex` goes red.

Most tests build synthetic trees under `tmp_path`. Fixture text is built in pieces
(`A = "#" + " DECISION"`): the planner and the scanner read that head followed by a
plan-id-like token in ANY string of a `.py` file as an anchor (step 98).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tests.test_fsm_llm_regression.support import REPO
from tests.test_fsm_llm_regression.test_decision_index import (
    MANIFEST,
    P1,
    P2,
    P3,
    SCRIPT,
    A,
    _load,
    _write,
)

INDEX = REPO / "docs" / "decisions" / "index.md"
REGENERATE = "make decision-index"
LEGACY_OLD = "plan_2026-05-29_0c00a594"


@pytest.fixture(scope="module")
def di() -> ModuleType:
    return _load()


def _manifest_text(*lines: str) -> str:
    return "# header\n" + "\n".join(lines) + "\n"


def _env(**changes: str | None) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "CI"}
    for key, value in changes.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    return env


def _run(
    root: Path, *args: str, env: dict[str, str] | None = None, cwd: Path | None = None
) -> Any:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=cwd or root,
        env=env if env is not None else _env(),
        timeout=120,
        check=False,
    )


def _render(di: ModuleType, root: Path) -> str:
    manifest = di.load_manifest(root / "plans" / "ANCHORS.md")
    return di.render(di.scan(root), manifest)


def _healthy(root: Path) -> None:
    """A tree whose three anchors resolve, retire and resolve."""
    _write(
        root,
        "src/pkg/a.py",
        "class Foo:\n"
        "    def bar(self):\n"
        f"        {A} {P1}/D-002: keep the lock order\n"
        "        # never take the second lock first\n"
        "        return 1\n"
        "\n"
        f"{A} {P2}/D-007 [STALE]: gone with its plan\n"
        "def top():\n"
        "    return 2\n",
    )
    _write(
        root,
        "plans/ANCHORS.md",
        _manifest_text(
            f"{P1}/D-002 | 2026-11-10 | Locks are taken in one order.",
            f"{P3}/D-009 | 2026-07-19 | A decision whose anchor was removed.",
        ),
    )


def _write_index(root: Path, name: str = "index.md") -> Path:
    out = root / "docs" / "decisions" / name
    proc = _run(root, "--write", str(out))
    assert proc.returncode == 0, proc.stderr
    return out


# --------------------------------------------------------------------------- real tree


class TestCommittedIndex:
    """The committed index equals what the scanner renders from the working tree."""

    def test_the_committed_index_is_current_and_the_anchors_resolve(self, di):
        assert INDEX.is_file(), f"docs/decisions/index.md is missing: {REGENERATE}"
        manifest = di.load_manifest(MANIFEST)
        found = di.check_index(INDEX, di.scan(REPO), manifest)
        assert found == [], (
            f"run `{REGENERATE}` and commit docs/decisions/index.md (any step that "
            "adds, moves or re-words a decision anchor, or appends to "
            "plans/ANCHORS.md, regenerates it); HAND-OFF: plans/ANCHORS.md is the "
            "owner's manifest and may still be uncommitted",
            found,
        )

    def test_the_real_cli_exits_zero_on_the_committed_index(self):
        proc = _run(REPO, "--check", str(INDEX), cwd=REPO)
        assert proc.returncode == 0, (REGENERATE, proc.stderr)
        assert proc.stdout.startswith("ok:") and proc.stderr == ""

    def test_the_committed_index_has_the_promised_shape(self):
        data = INDEX.read_bytes()
        text = data.decode("ascii")
        assert text.endswith("\n") and not text.endswith("\n\n")
        assert "\r" not in text
        assert not [n for n, ln in enumerate(text.split("\n"), 1) if ln != ln.rstrip()]
        assert len(data) < 300_000, "keep it well under the 500 KB pre-commit limit"
        for section in (
            "## Summary",
            "## Decisions",
            "## Retired",
            "## Manifest entries with no code anchor",
        ):
            assert section in text, section
        assert "## Anchors that do not resolve" not in text

    def test_the_committed_index_names_no_line_number_and_no_absolute_path(self):
        text = INDEX.read_text(encoding="ascii")
        assert str(REPO) not in text and "/home/" not in text
        # a manifest rationale may quote `file.py:123` (it is the ledger's own text); a
        # place of the index never carries one
        places = [ln for ln in text.split("\n") if ln.startswith(("- Where", "  - `"))]
        assert places and not [p for p in places if re.search(r"\.py[:#]", p)]

    def test_a_stale_edit_of_the_committed_index_makes_check_exit_one(
        self, di, tmp_path
    ):
        """The RED run on a COPY: one changed word is enough."""
        copy = tmp_path / "index.md"
        copy.write_bytes(
            INDEX.read_bytes().replace(b"# Decision index", b"# Decisions")
        )
        manifest = di.load_manifest(MANIFEST)
        found = di.check_index(copy, di.scan(REPO), manifest)
        assert len(found) == 1 and "out of date" in found[0]
        assert REGENERATE in found[0]
        proc = _run(REPO, "--check", str(copy), cwd=REPO)
        assert proc.returncode == 1
        assert REGENERATE in proc.stderr and "committed" in proc.stderr

    def test_a_removed_decision_block_makes_check_exit_one(self, di, tmp_path):
        text = INDEX.read_text(encoding="ascii")
        cut = text.index("\n#### ")
        end = text.index("\n#### ", cut + 1)
        copy = tmp_path / "index.md"
        copy.write_bytes((text[:cut] + text[end:]).encode("ascii"))
        found = di.check_index(copy, di.scan(REPO), di.load_manifest(MANIFEST))
        assert found and "out of date" in found[0]


# --------------------------------------------------------------------------- the gate


class TestGate:
    def test_a_healthy_tree_with_a_fresh_index_passes(self, tmp_path):
        _healthy(tmp_path)
        index = _write_index(tmp_path)
        proc = _run(tmp_path, "--check", str(index))
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.startswith("ok:") and proc.stderr == ""

    def test_a_stale_index_exits_one_and_names_the_fix(self, tmp_path):
        _healthy(tmp_path)
        index = _write_index(tmp_path)
        index.write_bytes(index.read_bytes().replace(b"lock order", b"LOCK ORDER"))
        proc = _run(tmp_path, "--check", str(index))
        assert proc.returncode == 1
        assert "out of date" in proc.stderr
        assert "make decision-index" in proc.stderr
        assert "--write" in proc.stderr
        assert "LOCK ORDER" in proc.stderr, "the diff hint shows the changed row"

    def test_a_missing_index_exits_one_and_names_the_fix(self, tmp_path):
        _healthy(tmp_path)
        proc = _run(tmp_path, "--check", str(tmp_path / "nope.md"))
        assert proc.returncode == 1
        assert "missing" in proc.stderr and "make decision-index" in proc.stderr

    def test_an_unresolved_anchor_exits_one_even_when_the_index_is_fresh(
        self, tmp_path
    ):
        _healthy(tmp_path)
        _write(
            tmp_path,
            "src/pkg/b.py",
            f"{A} {P3}/D-044: a live anchor nobody recorded\ndef f():\n    pass\n",
        )
        index = _write_index(tmp_path)  # regenerated AFTER planting: not just stale
        proc = _run(tmp_path, "--check", str(index))
        assert proc.returncode == 1
        assert "no line in plans/ANCHORS.md" in proc.stderr
        assert f"{P3}/D-044" in proc.stderr
        assert "out of date" not in proc.stderr, "the index itself is current"

    def test_a_badprefix_anchor_exits_one_even_when_the_index_is_fresh(self, tmp_path):
        _healthy(tmp_path)
        _write(tmp_path, "src/pkg/c.py", f"{A} plan-2026-07-18-80b0bd4d/D-013: x\n")
        _write(tmp_path, "src/pkg/d.py", f"{A} D-021: a bare number\n")
        index = _write_index(tmp_path)
        proc = _run(tmp_path, "--check", str(index))
        assert proc.returncode == 1
        assert "no plan id" in proc.stderr
        assert (
            "D-021" in proc.stderr and "plan-2026-07-18-80b0bd4d/D-013" in proc.stderr
        )
        assert "out of date" not in proc.stderr

    def test_a_retired_anchor_alone_is_green(self, tmp_path):
        _write(tmp_path, "src/a.py", f"{A} {P2}/D-007 [STALE]: gone\n")
        _write(tmp_path, "plans/ANCHORS.md", _manifest_text())
        index = _write_index(tmp_path)
        proc = _run(tmp_path, "--check", str(index))
        assert proc.returncode == 0, proc.stderr

    def test_both_causes_are_reported_in_one_run(self, tmp_path):
        _healthy(tmp_path)
        index = _write_index(tmp_path)
        _write(tmp_path, "src/pkg/b.py", f"{A} {P3}/D-044: new\n{A} nope/D-045: typo\n")
        proc = _run(tmp_path, "--check", str(index))
        assert proc.returncode == 1
        assert proc.stderr.count("error:") == 3, proc.stderr
        assert "out of date" in proc.stderr and "no plan id" in proc.stderr

    def test_check_reads_the_manifest_only_never_a_local_decisions_file(self, tmp_path):
        """`CI` unset and a local decisions.md that names the key: still exit 1."""
        _healthy(tmp_path)
        _write(tmp_path, "src/pkg/b.py", f"{A} {P3}/D-044: new\n")
        _write(
            tmp_path,
            f"plans/{P3}/decisions.md",
            "# decisions\n\n## D-044 | EXECUTE | 2026-11-14\nbody\n",
        )
        index = _write_index(tmp_path)
        for ci in (None, "1"):
            proc = _run(tmp_path, "--check", str(index), env=_env(CI=ci))
            assert proc.returncode == 1, (ci, proc.stderr)
            assert f"{P3}/D-044" in proc.stderr

    def test_crlf_in_the_committed_index_is_read_as_lf(self, tmp_path):
        """`.gitattributes` pins LF, but a checkout that translated must not fail."""
        _healthy(tmp_path)
        index = _write_index(tmp_path)
        index.write_bytes(index.read_bytes().replace(b"\n", b"\r\n"))
        assert _run(tmp_path, "--check", str(index)).returncode == 0

    def test_a_mode_is_required_and_the_modes_exclude_each_other(self, tmp_path):
        assert _run(tmp_path).returncode == 2
        assert _run(tmp_path, "--report", "--check", "x.md").returncode == 2
        assert _run(tmp_path, "--write", "x.md", "--check", "x.md").returncode == 2

    def test_write_creates_missing_directories_and_prints_nothing(self, tmp_path):
        _healthy(tmp_path)
        out = tmp_path / "deep" / "er" / "index.md"
        proc = _run(tmp_path, "--write", str(out))
        assert proc.returncode == 0 and proc.stdout == "" and proc.stderr == ""
        assert out.read_bytes().endswith(b"\n")

    def test_write_exits_zero_even_with_an_unresolved_anchor(self, tmp_path):
        _healthy(tmp_path)
        _write(tmp_path, "src/pkg/b.py", f"{A} {P3}/D-044: new\n")
        index = _write_index(tmp_path)
        assert "## Anchors that do not resolve" in index.read_text(encoding="ascii")

    def test_main_returns_the_exit_code_in_process(self, di, tmp_path, capsys):
        _healthy(tmp_path)
        out = tmp_path / "index.md"
        assert di.main(["--root", str(tmp_path), "--write", str(out)]) == 0
        assert di.main(["--root", str(tmp_path), "--check", str(out)]) == 0
        out.write_text("stale\n", encoding="ascii")
        assert di.main(["--root", str(tmp_path), "--check", str(out)]) == 1
        assert "out of date" in capsys.readouterr().err


# --------------------------------------------------------------------------- render


class TestRender:
    def test_two_renders_of_one_tree_are_equal(self, di, tmp_path):
        _healthy(tmp_path)
        assert _render(di, tmp_path) == _render(di, tmp_path)

    def test_creation_order_does_not_matter(self, di, tmp_path):
        names = ("src/z.py", "src/a.py", "src/m/b.py", "tests/t.py", "scripts/s.py")

        def build(root: Path, order: tuple[str, ...]) -> None:
            for rel in order:
                _write(
                    root,
                    rel,
                    f"{A} {P1}/D-001: first in {rel}\n{A} {P2}/D-002: second\n",
                )
            _write(
                root,
                "plans/ANCHORS.md",
                _manifest_text(
                    f"{P1}/D-001 | 2026-11-10 | a", f"{P2}/D-002 | 2026-09-19 | b"
                ),
            )

        one, two = tmp_path / "one", tmp_path / "two"
        build(one, names)
        build(two, tuple(reversed(names)))
        assert _render(di, one) == _render(di, two)

    def test_directory_listing_order_does_not_matter(self, di, tmp_path, monkeypatch):
        for rel in ("src/z.py", "src/a.py", "src/m/b.py", "tests/t.py"):
            _write(tmp_path, rel, f"{A} {P1}/D-001: x\n")
        _write(
            tmp_path, "plans/ANCHORS.md", _manifest_text(f"{P1}/D-001 | 2026-11-10 | a")
        )
        expected = _render(di, tmp_path)
        real = os.scandir

        class _Reversed:
            def __init__(self, path: Any) -> None:
                with real(path) as it:
                    self._entries = list(it)[::-1]

            def __enter__(self) -> list[Any]:
                return self._entries

            def __exit__(self, *exc: object) -> None:
                return None

        monkeypatch.setattr(os, "scandir", _Reversed)
        assert _render(di, tmp_path) == expected

    def test_a_line_number_never_leaks(self, di, tmp_path):
        """The same anchor at line 2 and at line 60 renders the same text."""
        one, two = tmp_path / "one", tmp_path / "two"
        body = f"def f():\n    {A} {P1}/D-001: the rule\n    return 1\n"
        _write(one, "src/a.py", body)
        _write(two, "src/a.py", "x = 1\n" * 58 + body)
        for root in (one, two):
            _write(
                root, "plans/ANCHORS.md", _manifest_text(f"{P1}/D-001 | 2026-11-10 | a")
            )
        assert _render(di, one) == _render(di, two)
        assert ".py:" not in _render(di, one)

    def test_no_absolute_path_and_no_root_name_leaks(self, di, tmp_path):
        _healthy(tmp_path)
        text = _render(di, tmp_path)
        assert str(tmp_path) not in text and tmp_path.name not in text

    def test_ascii_only_with_visible_escapes(self, di, tmp_path):
        _write(
            tmp_path,
            "src/a.py",
            f"{A} {P1}/D-001: café — never `xé` \U0001f600\n"
            "# second “quoted” → arrow\n",
        )
        _write(
            tmp_path,
            "plans/ANCHORS.md",
            _manifest_text(f"{P1}/D-001 | 2026-11-10 | Résumé — über 中"),
        )
        text = _render(di, tmp_path)
        text.encode("ascii")
        assert "\\u00e9" in text and "\\U0001f600" in text and "\\u4e2d" in text
        assert "caf\\u00e9 - never" in text, "an em dash becomes a hyphen"
        assert '"quoted" -> arrow' in text

    def test_a_control_character_is_escaped_not_written(self, di, tmp_path):
        _write(
            tmp_path,
            "src/a.py",
            f"{A} {P1}/D-001: bell\x07 and escape\x1b here\n",
        )
        _write(
            tmp_path, "plans/ANCHORS.md", _manifest_text(f"{P1}/D-001 | 2026-11-10 | a")
        )
        text = _render(di, tmp_path)
        assert "\x07" not in text and "\x1b" not in text
        assert "\\x07" in text and "\\x1b" in text

    def test_no_trailing_space_and_one_final_newline(self, di, tmp_path):
        _write(
            tmp_path,
            "src/a.py",
            f"{A} {P1}/D-001: tail   \t\n#   spaces   \n#\n",
        )
        _write(
            tmp_path,
            "plans/ANCHORS.md",
            _manifest_text(f"{P1}/D-001 | 2026-11-10 | rationale with trailing   \t "),
        )
        text = _render(di, tmp_path)
        assert text.endswith("\n") and not text.endswith("\n\n")
        assert [ln for ln in text.split("\n") if ln != ln.rstrip()] == []
        assert "\t" not in text and "\r" not in text

    def test_inner_whitespace_runs_and_tabs_collapse_to_one_space(self, di, tmp_path):
        _write(
            tmp_path,
            "src/a.py",
            f"{A} {P1}/D-001: two   spaces\tand a tab\n#  indented   more\n",
        )
        _write(
            tmp_path,
            "plans/ANCHORS.md",
            _manifest_text(f"{P1}/D-001 | 2026-11-10 | rationale   with\tgaps  inside"),
        )
        text = _render(di, tmp_path)
        assert "- Rationale (2026-11-10): rationale with gaps inside" in text
        assert "- Rule: `two spaces and a tab indented more`" in text
        assert "  " not in text.replace("\n  - ", "")  # only the nested list indent

    def test_an_empty_tree_renders_a_valid_index(self, di, tmp_path):
        text = _render(di, tmp_path)
        assert text.startswith("# Decision index\n")
        assert text.endswith("None.\n") and "## Retired" in text
        assert [ln for ln in text.split("\n") if ln != ln.rstrip()] == []

    def test_retired_anchors_are_in_their_own_section_not_in_decisions(
        self, di, tmp_path
    ):
        _healthy(tmp_path)
        text = _render(di, tmp_path)
        decisions, retired = text.split("## Retired")
        retired = retired.split("## Manifest entries")[0]
        assert f"#### {P1}/D-002" in decisions
        assert f"#### {P2}/D-007" not in decisions
        assert f"#### {P2}/D-007" in retired
        assert f"#### {P1}/D-002" not in retired
        assert "[STALE]" in retired
        assert "plan directory that held the record is gone" in retired
        assert "may still bind" in retired

    def test_a_stale_anchor_with_a_manifest_line_stays_in_decisions(self, di, tmp_path):
        _write(tmp_path, "src/a.py", f"{A} {P1}/D-001 [STALE]: old\n")
        _write(
            tmp_path, "plans/ANCHORS.md", _manifest_text(f"{P1}/D-001 | 2026-11-10 | a")
        )
        text = _render(di, tmp_path)
        decisions, retired = text.split("## Retired")
        assert f"#### {P1}/D-001" in decisions and f"#### {P1}/D-001" not in retired
        assert "[STALE]" in decisions.split("## Decisions")[1]

    def test_the_unanchored_manifest_entries_have_their_own_section(self, di, tmp_path):
        _healthy(tmp_path)
        text = _render(di, tmp_path)
        tail = text.split("## Manifest entries with no code anchor")[1]
        assert f"`{P3}/D-009`" in tail and "anchor was removed" in tail
        assert f"{P1}/D-002" not in tail, "an anchored entry is not listed there"
        assert f"#### {P3}/D-009" not in text

    def test_plans_are_in_date_order_then_id_order_and_decisions_by_number(
        self, di, tmp_path
    ):
        later = "plan-2026-11-10T194139-2cefeb61"
        same_day_a = "plan-2026-11-10T080000-aaaaaaaa"
        legacy = LEGACY_OLD
        keys = [(later, 1), (P2, 12), (P2, 3), (legacy, 5), (same_day_a, 2), (P3, 1)]
        _write(
            tmp_path,
            "src/a.py",
            "".join(f"{A} {p}/D-{n:03d}: x\n" for p, n in keys),
        )
        _write(
            tmp_path,
            "plans/ANCHORS.md",
            _manifest_text(*(f"{p}/D-{n:03d} | 2026-01-01 | r" for p, n in keys)),
        )
        text = _render(di, tmp_path)
        headings = [ln[5:] for ln in text.split("\n") if ln.startswith("#### ")]
        assert headings == [
            f"{legacy}/D-005",  # 2026-05-29
            f"{P3}/D-001",  # 2026-07-19
            f"{P2}/D-003",  # 2026-09-19, number order
            f"{P2}/D-012",
            f"{same_day_a}/D-002",  # 2026-11-10 08:00
            f"{later}/D-001",  # 2026-11-10 19:41
        ]
        plans = [ln[4:] for ln in text.split("\n") if ln.startswith("### ")]
        assert plans == [legacy, P3, P2, same_day_a, later]

    def test_render_does_not_depend_on_the_order_of_its_input(self, di, tmp_path):
        """`render` sorts for itself: shuffled anchors and a reversed manifest give the
        same text, so no ordering rests on the order `scan` happens to return."""
        keys = [(P1, 1), (P2, 12), (P2, 3), (LEGACY_OLD, 5), (P3, 1), (P3, 9)]
        for rel in ("src/z.py", "src/a.py", "src/m/b.py"):
            _write(
                tmp_path,
                rel,
                "".join(f"{A} {p}/D-{n:03d}: in {rel}\n" for p, n in keys),
            )
        _write(
            tmp_path,
            "src/gone.py",
            f"{A} {P2}/D-099 [STALE]: x\n{A} {P1}/D-098 [STALE]: y\n",
        )
        _write(
            tmp_path,
            "plans/ANCHORS.md",
            _manifest_text(
                *(f"{p}/D-{n:03d} | 2026-01-01 | r" for p, n in keys),
                f"{P1}/D-050 | 2026-01-02 | unanchored one",
                f"{P3}/D-050 | 2026-01-02 | unanchored two",
            ),
        )
        anchors = di.scan(tmp_path)
        manifest = di.load_manifest(tmp_path / "plans" / "ANCHORS.md")
        expected = di.render(anchors, manifest)
        shuffled = di.render(anchors[::-1], dict(reversed(list(manifest.items()))))
        assert shuffled == expected
        assert di.render(anchors[3:] + anchors[:3], manifest) == expected

    def test_a_place_is_a_path_and_the_enclosing_class_method(self, di, tmp_path):
        _healthy(tmp_path)
        text = _render(di, tmp_path)
        assert "- Where: `src/pkg/a.py`, `Foo.bar`" in text
        assert "`src/pkg/a.py`, module level [STALE]" in text, (
            "above a def is module level"
        )

    def test_several_places_are_listed_sorted_and_a_repeat_is_counted(
        self, di, tmp_path
    ):
        _write(tmp_path, "src/z.py", f"{A} {P1}/D-001: z\n")
        _write(
            tmp_path,
            "src/a.py",
            f"def f():\n    {A} {P1}/D-001: a\n    pass\n\n"
            f"def g():\n    {A} {P1}/D-001: g\n    {A} {P1}/D-001: g2\n    pass\n",
        )
        _write(
            tmp_path, "plans/ANCHORS.md", _manifest_text(f"{P1}/D-001 | 2026-11-10 | a")
        )
        text = _render(di, tmp_path)
        block = text.split(f"#### {P1}/D-001")[1].split("\n\n")[0]
        places = [
            ln.strip() for ln in block.split("\n") if ln.strip().startswith("- `")
        ]
        assert places == [
            "- `src/a.py`, `f`",
            "- `src/a.py`, `g` (2 anchors)",
            "- `src/z.py`, module level",
        ]

    def test_the_rule_summary_is_the_first_text_and_is_cut_at_the_width(
        self, di, tmp_path
    ):
        long = "word " * 80
        _write(tmp_path, "src/a.py", f"{A} {P1}/D-001: {long}\n# continued\n")
        _write(tmp_path, "src/b.py", f"{A} {P1}/D-002: short rule\n# second row\n")
        _write(
            tmp_path,
            "plans/ANCHORS.md",
            _manifest_text(
                f"{P1}/D-001 | 2026-11-10 | a", f"{P1}/D-002 | 2026-11-10 | b"
            ),
        )
        text = _render(di, tmp_path)
        rules = [ln for ln in text.split("\n") if ln.startswith("- Rule: ")]
        assert rules[1] == "- Rule: `short rule second row`", "rows are joined"
        body = rules[0][len("- Rule: `") : -1]
        assert body.endswith("...") and len(body) <= di.SUMMARY_WIDTH
        assert body.startswith("word word")

    def test_the_summary_width_is_the_documented_constant(self, di):
        assert di.SUMMARY_WIDTH == 120

    def test_markdown_characters_of_a_summary_stay_inert(self, di, tmp_path):
        _write(
            tmp_path,
            "src/a.py",
            f"{A} {P1}/D-001: `x` and __init__ and <b>bold</b> | pipe\n",
        )
        _write(
            tmp_path, "plans/ANCHORS.md", _manifest_text(f"{P1}/D-001 | 2026-11-10 | a")
        )
        rule = next(
            ln for ln in _render(di, tmp_path).split("\n") if ln.startswith("- Rule: ")
        )
        assert rule == "- Rule: `'x' and __init__ and <b>bold</b> | pipe`"

    def test_a_rationale_keeps_a_pipe_and_escapes_a_tag_outside_code(
        self, di, tmp_path
    ):
        _write(tmp_path, "src/a.py", f"{A} {P1}/D-001: x\n")
        _write(
            tmp_path,
            "plans/ANCHORS.md",
            _manifest_text(
                f"{P1}/D-001 | 2026-11-10 | uses a | pipe, a <tag> and `<code>` too"
            ),
        )
        rationale = next(
            ln
            for ln in _render(di, tmp_path).split("\n")
            if ln.startswith("- Rationale")
        )
        assert rationale == (
            "- Rationale (2026-11-10): uses a | pipe, a &lt;tag> and `<code>` too"
        )

    def test_an_unresolved_section_exists_only_when_something_does_not_resolve(
        self, di, tmp_path
    ):
        _healthy(tmp_path)
        assert "## Anchors that do not resolve" not in _render(di, tmp_path)
        _write(tmp_path, "src/b.py", f"{A} {P3}/D-044: live\n{A} bad/D-045: typo\n")
        text = _render(di, tmp_path)
        section = text.split("## Anchors that do not resolve")[1].split("\n## ")[0]
        assert f"`{P3}/D-044` (unresolved): `src/b.py`, module level" in section
        assert "`bad/D-045` (badprefix)" in section
        assert text.index("## Retired") < text.index("## Anchors that do not resolve")

    def test_the_summary_counts_are_the_scan_counts(self, di, tmp_path):
        _healthy(tmp_path)
        text = _render(di, tmp_path)
        assert "- Anchors: 2 in 1 files, 2 distinct decisions." in text
        assert "- Resolved (the manifest has the decision): 1." in text
        assert "- Retired (marked `[STALE]`, no manifest line): 1." in text
        assert "- Unresolved: 0. Bad prefix: 0." in text
        assert "- Manifest entries: 2, of which 1 have no anchor in the tree." in text

    def test_render_is_pure_it_reads_no_file_and_no_environment(
        self, di, tmp_path, monkeypatch
    ):
        _healthy(tmp_path)
        anchors = di.scan(tmp_path)
        manifest = di.load_manifest(tmp_path / "plans" / "ANCHORS.md")
        expected = di.render(anchors, manifest)

        def boom(*_a: Any, **_k: Any) -> Any:
            raise AssertionError("render touched the file system or the environment")

        monkeypatch.setattr(Path, "read_bytes", boom)
        monkeypatch.setattr(Path, "read_text", boom)
        monkeypatch.setattr(Path, "open", boom)
        monkeypatch.setattr(os, "scandir", boom)
        monkeypatch.setattr(os, "getenv", boom)
        monkeypatch.setattr(os.environ, "get", boom)
        assert di.render(anchors, manifest) == expected


# --------------------------------------------------------------------------- CI independence


class TestCiIndependence:
    def _local_tree(self, root: Path) -> None:
        _healthy(root)
        _write(root, "src/pkg/b.py", f"{A} {P3}/D-044: live, only a local record\n")
        _write(
            root,
            f"plans/{P3}/decisions.md",
            "# decisions\n\n## D-044 | EXECUTE | 2026-11-14\nmore text\n",
        )

    def test_ci_set_and_unset_render_byte_identically_in_process(
        self, di, tmp_path, monkeypatch
    ):
        self._local_tree(tmp_path)
        monkeypatch.delenv("CI", raising=False)
        unset = _render(di, tmp_path)
        monkeypatch.setenv("CI", "1")
        assert _render(di, tmp_path) == unset
        monkeypatch.setenv("CI", "")
        assert _render(di, tmp_path) == unset

    def test_ci_set_and_unset_write_byte_identical_files_through_the_cli(
        self, tmp_path
    ):
        self._local_tree(tmp_path)
        out = {}
        for label, ci in (("unset", None), ("one", "1"), ("true", "true")):
            target = tmp_path / f"{label}.md"
            proc = _run(tmp_path, "--write", str(target), env=_env(CI=ci))
            assert proc.returncode == 0, proc.stderr
            out[label] = target.read_bytes()
        assert out["unset"] == out["one"] == out["true"]

    def test_a_local_decisions_file_never_changes_the_rendered_index(
        self, di, tmp_path
    ):
        with_local, without = tmp_path / "with", tmp_path / "without"
        self._local_tree(with_local)
        self._local_tree(without)
        for p in (without / "plans" / P3 / "decisions.md",):
            p.unlink()
        assert _render(di, with_local) == _render(di, without)

    def test_check_gives_the_same_exit_code_and_messages_in_ci_and_locally(
        self, tmp_path
    ):
        self._local_tree(tmp_path)
        index = _write_index(tmp_path)
        runs = [
            _run(tmp_path, "--check", str(index), env=_env(CI=ci))
            for ci in (None, "1", "true")
        ]
        assert {r.returncode for r in runs} == {1}
        assert len({r.stderr for r in runs}) == 1

    def test_ci_is_set_means_a_non_empty_value(self, di, monkeypatch):
        monkeypatch.delenv("CI", raising=False)
        assert di.ci_is_set() is False
        monkeypatch.setenv("CI", "")
        assert di.ci_is_set() is False
        monkeypatch.setenv("CI", "true")
        assert di.ci_is_set() is True

    def test_the_script_names_the_environment_in_one_place_only(self):
        """Only `ci_is_set` reads the environment; `render` and `check_index` do not."""
        import ast

        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        readers = set()
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.FunctionDef):
                continue
            for node in ast.walk(fn):
                if isinstance(node, ast.Attribute) and node.attr in {
                    "environ",
                    "getenv",
                }:
                    readers.add(fn.name)
        assert readers == {"ci_is_set"}


# --------------------------------------------------------------------------- the report


class TestReport:
    def _report(self, root: Path, *, ci: str | None = None) -> list[str]:
        proc = _run(root, "--report", env=_env(CI=ci))
        assert proc.returncode == 0, proc.stderr
        return proc.stdout.splitlines()

    def _tree(self, root: Path) -> None:
        _write(
            root,
            "src/a.py",
            f"{A} {P1}/D-001: ok\n"
            "x = 1  # trailing\n"
            "s = '# not a comment'\n"
            f"{A} {P2}/D-002 [STALE]: old\n"
            f"{A} {P3}/D-003: live\n",
        )
        _write(root, "plans/ANCHORS.md", _manifest_text(f"{P1}/D-001 | 2026-11-10 | a"))

    def test_the_share_of_resolved_and_retired(self, tmp_path):
        self._tree(tmp_path)
        out = self._report(tmp_path)
        assert "resolved + retired: 2 of 3 (66.7%)" in out
        assert "manifest entries without an anchor: 0" in out

    def test_comment_share_counts_comment_tokens_over_source_rows(self, di, tmp_path):
        _write(
            tmp_path,
            "src/a.py",
            "# one\nx = 1  # two\ns = '# not a comment'\n\n'''\n# inside a string\n'''\n",
        )
        share = di.comment_share(tmp_path)
        assert (share.comments, share.rows, share.files) == (2, 7, 1)
        assert share.percent == pytest.approx(100 * 2 / 7)

    def test_comment_share_reads_only_python_files_under_src(self, di, tmp_path):
        _write(tmp_path, "src/a.py", "# c\nx = 1\n")
        _write(tmp_path, "src/notes.md", "# heading\n")
        _write(tmp_path, "tests/t.py", "# c\n# c\n# c\n")
        _write(tmp_path, "scripts/s.py", "# c\n")
        share = di.comment_share(tmp_path)
        assert (share.comments, share.rows, share.files) == (1, 2, 1)

    def test_a_file_that_does_not_tokenize_is_left_out_of_both_sides(
        self, di, tmp_path
    ):
        _write(tmp_path, "src/a.py", "# c\nx = 1\n")
        _write(tmp_path, "src/bad.py", "x = (\n# open paren never closed\n")
        share = di.comment_share(tmp_path)
        assert (share.comments, share.rows, share.files) == (1, 2, 1)

    def test_an_empty_tree_has_no_share_and_does_not_divide_by_zero(self, di, tmp_path):
        assert di.comment_share(tmp_path).percent is None
        out = self._report(tmp_path)
        assert (
            "comment share: n/a (0 comment tokens of 0 source rows, 0 files" in out[-1]
        )
        assert "resolved + retired: 0 of 0 (n/a)" in out

    def test_the_report_line_for_the_share_says_it_is_not_gated(self, tmp_path):
        self._tree(tmp_path)
        last = self._report(tmp_path)[-1]
        assert last.startswith("comment share: ") and "not gated" in last
        assert (
            "4 comment tokens of 5 source rows" in last
        )  # three anchors, one trailing
        assert "80.0%" in last

    def test_resolved_local_counts_only_while_ci_is_unset(self, tmp_path):
        self._tree(tmp_path)
        _write(
            tmp_path,
            f"plans/{P3}/decisions.md",
            "# d\n\n## D-003 | EXECUTE | 2026-11-14\nbody\n## D-004 | x | y\n",
        )
        assert "resolved-local: 1 of 1 unresolved" in self._report(tmp_path)
        assert "resolved-local: n/a (CI is set)" in self._report(tmp_path, ci="1")

    def test_resolved_local_changes_neither_the_unresolved_count_nor_the_exit(
        self, tmp_path
    ):
        self._tree(tmp_path)
        _write(tmp_path, f"plans/{P3}/decisions.md", "## D-003 | EXECUTE | d\n")
        out = self._report(tmp_path)
        assert "unresolved: 1" in out, "still unresolved: the manifest decides"

    def test_resolved_local_needs_the_heading_of_the_same_number(self, di, tmp_path):
        _write(tmp_path, "src/a.py", f"{A} {P3}/D-003: x\n{A} {P3}/D-030: y\n")
        _write(
            tmp_path,
            f"plans/{P3}/decisions.md",
            "## D-030 | a | b\nD-003 in prose, not a heading\n### D-003\n",
        )
        found = di.local_resolved(tmp_path, di.scan(tmp_path))
        assert [a.number for a in found] == [30]

    def test_resolved_local_is_empty_when_ci_is_set(self, di, tmp_path, monkeypatch):
        _write(tmp_path, "src/a.py", f"{A} {P3}/D-003: x\n")
        _write(tmp_path, f"plans/{P3}/decisions.md", "## D-003 | a | b\n")
        monkeypatch.setenv("CI", "1")
        assert di.local_resolved(tmp_path, di.scan(tmp_path)) == []
        monkeypatch.delenv("CI")
        assert len(di.local_resolved(tmp_path, di.scan(tmp_path))) == 1

    def test_the_report_exits_zero_with_every_kind_of_problem(self, tmp_path):
        self._tree(tmp_path)
        _write(tmp_path, "src/b.py", f"{A} bad/D-001: typo\n")
        assert _run(tmp_path, "--report").returncode == 0

    def test_the_report_on_the_real_tree(self):
        proc = _run(REPO, "--report", cwd=REPO)
        assert proc.returncode == 0, proc.stderr
        rows = dict(ln.split(": ", 1) for ln in proc.stdout.splitlines())
        share = float(rows["comment share"].split("%")[0])
        assert 5.0 < share < 30.0, rows["comment share"]
        assert "not gated" in rows["comment share"]
        assert rows["unresolved"] == "0" and rows["badprefix"] == "0"
        assert rows["resolved + retired"].endswith("(100.0%)")

    def test_comment_share_reproduces_the_roadmap_baseline_definition(self, di):
        """12.87% at commit 4b24387d was 10,169 comment tokens over 78,983 source rows
        of 145 files of `src/**/*.py`: the same tokenize definition, row for row."""
        assert round(100 * 10169 / 78983, 2) == 12.87
        share = di.comment_share(REPO)
        assert share.files > 145 and share.rows > 78983 // 2
