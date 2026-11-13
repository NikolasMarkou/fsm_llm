"""The `plan-<id>/D-NNN` tokens of `src/` only move, they are never lost or added silently.

A `# DECISION plan-<id>/D-NNN` anchor states what NOT to do next to the code it
guards. The layout steps (65 to 94) cut modules apart; an anchor that is dropped,
duplicated or left behind with the wrong half turns a recorded rule into folklore.
The multiset of tokens, counted over the whole tree and ignoring which file holds
them, must therefore stay equal while code moves.

`tests/fixtures/anchor_multiset.json`:

    {"baseline": {"<token>": <count>, ...},
     "reasons":  [{"token": "<token>", "delta": <int>, "reason": "<why>"}, ...]}

`baseline` is the multiset at HEAD of step 60 (written once, never regenerated).
A step that adds or removes a token on purpose appends one `reasons` entry per
token in the same commit, with the signed change and the reason; the expected
multiset is `baseline` plus every delta, and the tree must equal it. Do not edit
`baseline` to make the test pass.

Scope, the same as the planner's anchor scan (`validate-plan.mjs`
`walkSourceFiles`): every file under `src/` whose extension is in
`ANCHOR_EXTENSIONS` (the planner's `ANCHOR_SOURCE_EXTS`; today only `.py` and
`.md` exist under `src/`, and no `.md` there carries a token), skipping the
directories of `SKIP_DIRS` and dot entries, at most 12 levels deep. This test
counts every textual occurrence of a token, comments, docstrings and strings
alike (a superset of the comment-only anchor scan: a token kept in a docstring
is still a reference that must not be lost). Both the current form
`plan-2026-11-12T073150-d454fd62/D-001` and the legacy form
`plan_2026-05-29_0c00a594/D-001` count.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from tests.test_fsm_llm_regression.support import REPO

FIXTURE = REPO / "tests" / "fixtures" / "anchor_multiset.json"

TOKEN = re.compile(r"plan[-_][0-9T-]+[-_][0-9a-f]{8}/D-[0-9]+")
ANCHOR_EXTENSIONS = frozenset(
    {
        *(".py", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".rb", ".go", ".rs", ".c"),
        *(".h", ".cpp", ".hpp", ".java", ".kt", ".sql", ".md", ".sh", ".bash"),
        *(".zsh", ".yml", ".yaml", ".toml", ".r", ".pl", ".pm", ".tf", ".jsx"),
        *(".cc", ".swift", ".scala", ".cs", ".php"),
    }
)
SKIP_DIRS = frozenset(
    {
        "node_modules",
        ".git",
        "dist",
        "build",
        "plans",
        "target",
        "__pycache__",
        ".cache",
        "vendor",
        "out",
    }
)
MAX_DEPTH = 12


def scan_tokens(root: Path = REPO / "src") -> Counter[str]:
    """Count every token under `root` (no file names kept: only the multiset matters)."""
    counts: Counter[str] = Counter()

    def walk(directory: Path, depth: int) -> None:
        if depth > MAX_DEPTH:
            return
        for entry in sorted(directory.iterdir()):
            if entry.name.startswith(".") or entry.is_symlink():
                continue
            if entry.is_dir():
                if entry.name not in SKIP_DIRS:
                    walk(entry, depth + 1)
            elif entry.suffix in ANCHOR_EXTENSIONS:
                counts.update(
                    TOKEN.findall(entry.read_text(encoding="utf-8", errors="replace"))
                )

    walk(root, 0)
    return counts


def render(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def load(path: Path = FIXTURE) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def reason_problems(reasons: Any) -> list[str]:
    """The `reasons` list must hold only complete entries."""
    if not isinstance(reasons, list):
        return ["`reasons` must be a list"]
    bad: list[str] = []
    for i, entry in enumerate(reasons):
        ok = isinstance(entry, dict) and {"token", "delta", "reason"} <= set(entry)
        if not ok:
            bad.append(f"reasons[{i}] needs `token`, `delta` and `reason`")
            continue
        if not isinstance(entry["token"], str) or not TOKEN.fullmatch(entry["token"]):
            bad.append(f"reasons[{i}]: `{entry['token']}` is not a plan-id/D-NNN token")
        if (
            not isinstance(entry["delta"], int)
            or isinstance(entry["delta"], bool)
            or entry["delta"] == 0
        ):
            bad.append(f"reasons[{i}]: `delta` must be a non-zero integer")
        if not isinstance(entry["reason"], str) or not entry["reason"].strip():
            bad.append(f"reasons[{i}]: `reason` must say why")
    return bad


def expected_counts(data: Mapping[str, Any]) -> Counter[str]:
    expected: Counter[str] = Counter(data["baseline"])
    for entry in data["reasons"]:
        expected[entry["token"]] += entry["delta"]
    return +expected  # drops zero counts


def multiset_problems(data: Mapping[str, Any], actual: Mapping[str, int]) -> list[str]:
    """A readable diff of the tokens lost and gained against the expected multiset."""
    bad = reason_problems(data["reasons"])
    if bad:
        return bad
    expected = expected_counts(data)
    for token, n in sorted(Counter(data["baseline"]).items()):
        if n < 1:
            bad.append(f"baseline count of {token} is {n}")
    for token in sorted({e["token"] for e in data["reasons"]}):
        net = data["baseline"].get(token, 0) + sum(
            e["delta"] for e in data["reasons"] if e["token"] == token
        )
        if net < 0:
            bad.append(f"reasons remove more of {token} than the baseline holds")
    lost = Counter(expected) - Counter(actual)
    gained = Counter(actual) - Counter(expected)
    bad += [
        f"LOST   {token} x{n}: the anchor is gone from src/"
        for token, n in sorted(lost.items())
    ]
    bad += [
        f"GAINED {token} x{n}: not in the baseline, add a reason or remove it"
        for token, n in sorted(gained.items())
    ]
    return bad


class TestTheRealTree:
    def test_the_file_is_deterministic_json(self) -> None:
        text = FIXTURE.read_bytes().decode("utf-8")
        assert text == render(json.loads(text))

    def test_the_tokens_only_moved(self) -> None:
        problems = multiset_problems(load(), scan_tokens())
        assert problems == [], "\n" + "\n".join(problems[:60])

    def test_the_baseline_is_not_vacuous(self) -> None:
        total = sum(load()["baseline"].values())
        assert total > 500 and len(load()["baseline"]) > 300


def _data(
    baseline: dict[str, int], reasons: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    return {"baseline": baseline, "reasons": reasons or []}


T1 = "plan-2026-09-19T175721-21cd7f8e/D-006"
T2 = "plan-2026-11-12T073150-d454fd62/D-005"
LEGACY = "plan_2026-05-29_0c00a594/D-002"


class TestScanner:
    """The scan itself, on scratch trees."""

    def _tree(self, root: Path) -> None:
        (root / "pkg").mkdir(parents=True)
        (root / "a.py").write_text(
            f"# DECISION {T1}: one\n# DECISION {T1}: two\nx = '{T2}'\n",
            encoding="utf-8",
        )
        (root / "pkg" / "b.py").write_text(
            f"# DECISION {LEGACY} [STALE]: legacy\n", encoding="utf-8"
        )

    def test_every_occurrence_is_counted_both_forms(self, tmp_path: Path) -> None:
        self._tree(tmp_path)
        assert scan_tokens(tmp_path) == Counter({T1: 2, T2: 1, LEGACY: 1})

    def test_the_count_ignores_which_file_holds_a_token(self, tmp_path: Path) -> None:
        self._tree(tmp_path)
        before = scan_tokens(tmp_path)
        text = (tmp_path / "a.py").read_text(encoding="utf-8")
        (tmp_path / "a.py").write_text("", encoding="utf-8")
        (tmp_path / "pkg" / "moved.py").write_text(text, encoding="utf-8")
        assert scan_tokens(tmp_path) == before

    def test_skipped_directories_and_extensions_are_not_read(
        self, tmp_path: Path
    ) -> None:
        for skipped in ("__pycache__", "node_modules", ".hidden"):
            (tmp_path / skipped).mkdir()
            (tmp_path / skipped / "x.py").write_text(
                f"# DECISION {T1}\n", encoding="utf-8"
            )
        (tmp_path / "data.json").write_text(f'"{T1}"', encoding="utf-8")
        assert scan_tokens(tmp_path) == Counter()

    def test_markdown_counts_like_the_planner_scan(self, tmp_path: Path) -> None:
        (tmp_path / "AGENTS.md").write_text(f"see {T1}\n", encoding="utf-8")
        assert scan_tokens(tmp_path) == Counter({T1: 1})


BASE = {T1: 2, T2: 1}


class TestMultisetProblems:
    def test_an_equal_multiset_is_clean(self) -> None:
        assert multiset_problems(_data(BASE), Counter(BASE)) == []

    def test_a_deleted_anchor_line_is_reported_as_lost(self, tmp_path: Path) -> None:
        # the mutation proof: one anchor line removed from a scratch source file
        src = tmp_path / "a.py"
        src.write_text(
            f"# DECISION {T1}\n# DECISION {T1}\nx = '{T2}'\n", encoding="utf-8"
        )
        assert multiset_problems(_data(BASE), scan_tokens(tmp_path)) == []
        src.write_text(f"# DECISION {T1}\nx = '{T2}'\n", encoding="utf-8")
        assert multiset_problems(_data(BASE), scan_tokens(tmp_path)) == [
            f"LOST   {T1} x1: the anchor is gone from src/"
        ]

    def test_a_duplicated_anchor_is_reported_as_gained(self) -> None:
        bad = multiset_problems(_data(BASE), Counter({T1: 2, T2: 2}))
        assert bad == [
            f"GAINED {T2} x1: not in the baseline, add a reason or remove it"
        ]

    def test_a_new_token_is_reported_as_gained(self) -> None:
        bad = multiset_problems(_data(BASE), Counter({**BASE, LEGACY: 1}))
        assert len(bad) == 1 and bad[0].startswith(f"GAINED {LEGACY}")

    def test_a_reason_explains_a_deliberate_change(self) -> None:
        reasons = [
            {"token": T1, "delta": -1, "reason": "the mirror it guarded was deleted"}
        ]
        assert multiset_problems(_data(BASE, reasons), Counter({T1: 1, T2: 1})) == []
        reasons = [{"token": LEGACY, "delta": 1, "reason": "a new anchor"}]
        assert (
            multiset_problems(_data(BASE, reasons), Counter({**BASE, LEGACY: 1})) == []
        )

    def test_a_reason_does_not_cover_a_different_change(self) -> None:
        reasons = [{"token": T1, "delta": -1, "reason": "gone on purpose"}]
        bad = multiset_problems(_data(BASE, reasons), Counter({T2: 1}))
        assert bad == [f"LOST   {T1} x1: the anchor is gone from src/"]

    @pytest.mark.parametrize(
        "entry, needle",
        [
            ({"token": T1, "delta": -1}, "needs `token`, `delta` and `reason`"),
            ({"token": T1, "delta": 0, "reason": "x"}, "non-zero integer"),
            ({"token": T1, "delta": 1.5, "reason": "x"}, "non-zero integer"),
            (
                {"token": "D-006", "delta": -1, "reason": "x"},
                "not a plan-id/D-NNN token",
            ),
            ({"token": T1, "delta": -1, "reason": "  "}, "must say why"),
            ({"token": T2, "delta": -3, "reason": "too many"}, "remove more of"),
        ],
    )
    def test_an_incomplete_or_impossible_reason_is_refused(
        self, entry: dict[str, Any], needle: str
    ) -> None:
        bad = multiset_problems(_data(BASE, [entry]), Counter(BASE))
        assert any(needle in p for p in bad), bad


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        sys.exit(
            "usage: python -m tests.test_fsm_llm_regression.test_decision_anchor_multiset --write"
        )
    if FIXTURE.exists():
        sys.exit(
            f"{FIXTURE.relative_to(REPO)} exists: the baseline is written once, add `reasons` instead"
        )
    FIXTURE.write_text(
        render({"baseline": dict(scan_tokens()), "reasons": []}),
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote {FIXTURE.relative_to(REPO)}")
