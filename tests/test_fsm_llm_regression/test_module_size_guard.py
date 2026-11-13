"""No module of the package is over 800 lines, and the 25 that are may not grow.

The 0.4 split of the layout plan (steps 65 to 94) cuts every module over 800
physical lines. Until a module is cut it sits in `OVERSIZED` with its size
today as a CEILING: it may shrink (lower the ceiling when it does) but never
grow, and a new module may not start over the limit. Each split step deletes
the module from `OVERSIZED` in the same commit; at the end the table is empty.

Counting (reproduced at the commit the split map was measured on, `dfd14d80`:
145 modules, 78,970 lines, 25 over 800 holding 42,490; the explorer recorded
78,983 and 42,503 from a working tree 13 lines larger): the physical lines of a
file, `len(text.splitlines())` on UTF-8 text. For every file of the package that
equals `wc -l` (no stray form feed or separator character). Comment and
docstring lines count: the rule is about the file, not about code.

A table entry is also a failure when it has gone stale: the module no longer
exists (renamed or split without deleting the entry) or is at 800 lines or
fewer (delete the entry). Keys are dotted module names so the table holds
before and after the move of the package to `src/fsm_llm/`.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from tests.test_fsm_llm_regression.support import source_modules

LIMIT = 800

# dotted module -> ceiling (its physical lines when step 60 recorded it)
OVERSIZED: dict[str, int] = {
    "fsm_llm.agents.base": 1846,
    "fsm_llm.agents.handlers": 968,
    "fsm_llm.agents.meta_builder": 1101,
    "fsm_llm.agents.meta_builders": 1158,
    "fsm_llm.agents.tools": 1399,
    "fsm_llm.api": 2335,
    "fsm_llm.monitor.instance_manager": 1974,
    "fsm_llm.monitor.server": 1677,
    "fsm_llm.okf": 1350,
    "fsm_llm.workflows.engine": 1570,
    "fsm_llm.workflows.steps": 1208,
}


def line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def module_sizes(repo: Path | None = None) -> dict[str, int]:
    modules = source_modules() if repo is None else source_modules(repo)
    return {name: line_count(path) for name, path in modules.items()}


def size_problems(
    sizes: Mapping[str, int],
    oversized: Mapping[str, int] = OVERSIZED,
    limit: int = LIMIT,
) -> list[str]:
    """Readable problems; `[]` when every module is within the limit or its ceiling."""
    bad: list[str] = []
    for name, size in sorted(sizes.items()):
        ceiling = oversized.get(name)
        if ceiling is None and size > limit:
            bad.append(
                f"{name} has {size} lines, over the limit of {limit} and not in OVERSIZED: split it"
            )
        elif ceiling is not None and size > ceiling:
            bad.append(
                f"{name} grew to {size} lines, over its OVERSIZED ceiling of {ceiling}"
            )
    for name, ceiling in sorted(oversized.items()):
        if name not in sizes:
            bad.append(
                f"{name} is in OVERSIZED (ceiling {ceiling}) but no such module exists: delete the entry"
            )
        elif sizes[name] <= limit:
            bad.append(
                f"{name} has {sizes[name]} lines, within the limit of {limit} but still in OVERSIZED "
                f"(ceiling {ceiling}): delete the entry"
            )
    return bad


def test_no_module_over_800_lines() -> None:
    problems = size_problems(module_sizes())
    assert problems == [], "\n" + "\n".join(problems)


def test_the_table_is_what_step_60_measured() -> None:
    """At most 25 modules and 42,361 lines (measured at HEAD of step 60): a split shrinks both."""
    assert len(OVERSIZED) <= 25
    assert sum(OVERSIZED.values()) <= 42361
    assert all(ceiling > LIMIT for ceiling in OVERSIZED.values())


TABLE = {"fsm_llm.big": 900}


class TestSizeProblems:
    """The mutation proofs, on synthetic sizes and scratch files."""

    def test_a_clean_tree_is_clean(self) -> None:
        assert size_problems({"fsm_llm.big": 900, "fsm_llm.small": 800}, TABLE) == []

    def test_a_shrunk_module_is_clean(self) -> None:
        assert size_problems({"fsm_llm.big": 850}, TABLE) == []

    def test_a_listed_module_that_grows_fails(self) -> None:
        assert size_problems({"fsm_llm.big": 905}, TABLE) == [
            "fsm_llm.big grew to 905 lines, over its OVERSIZED ceiling of 900"
        ]

    def test_a_new_module_over_the_limit_fails(self) -> None:
        bad = size_problems({"fsm_llm.big": 900, "fsm_llm.new": 801}, TABLE)
        assert bad == [
            "fsm_llm.new has 801 lines, over the limit of 800 and not in OVERSIZED: split it"
        ]

    def test_a_missing_module_is_a_stale_entry(self) -> None:
        bad = size_problems({}, TABLE)
        assert bad == [
            "fsm_llm.big is in OVERSIZED (ceiling 900) but no such module exists: delete the entry"
        ]

    def test_a_module_at_the_limit_is_a_stale_entry(self) -> None:
        bad = size_problems({"fsm_llm.big": 800}, TABLE)
        assert len(bad) == 1 and "still in OVERSIZED" in bad[0]

    def test_five_lines_appended_to_a_listed_module_fail(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        (src / "__init__.py").write_text("", encoding="utf-8")
        (src / "big.py").write_text("x = 1\n" * 900, encoding="utf-8")
        table = {"fsm_llm.big": 900}
        assert size_problems(module_sizes(tmp_path), table) == []
        with (src / "big.py").open("a", encoding="utf-8") as fh:
            fh.write("y = 2\n" * 5)
        bad = size_problems(module_sizes(tmp_path), table)
        assert bad == [
            "fsm_llm.big grew to 905 lines, over its OVERSIZED ceiling of 900"
        ]

    @pytest.mark.parametrize(
        "text, lines", [("a\nb\n", 2), ("a\nb", 2), ("", 0), ("\n", 1), ("a\n\n", 2)]
    )
    def test_the_count_is_the_physical_line_count(
        self, tmp_path: Path, text: str, lines: int
    ) -> None:
        path = tmp_path / "m.py"
        path.write_bytes(text.encode("utf-8"))
        assert line_count(path) == lines
