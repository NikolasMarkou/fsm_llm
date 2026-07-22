"""Repo-wide packaging invariants: every `src/` package is wired into every slot.

Five hand-maintained files enumerate the project's packages one name at a time:
`pyproject.toml`, `Makefile`, `tox.ini`, `.github/workflows/python-package.yml`
and `MANIFEST.in`. Adding a package means editing all of them, and a miss is
SILENT -- the package just quietly stops being type-checked, stops being counted
in coverage, or stops shipping in the sdist. Nothing errors.

This file is that missing check. The expected package list is derived from the
filesystem (`src/*/__init__.py`), never hardcoded, so it cannot go stale on the
one day it matters: the day a seventh package lands.
"""

from __future__ import annotations

import collections
import os
import pathlib
import re
import subprocess
import sys

import pytest

_REPO_ROOT = pathlib.Path(__file__).parents[1]

#: Every importable package under `src/`, derived from disk. `src/*.egg-info/`
#: and any stray directory without an `__init__.py` are excluded by construction.
SRC_PACKAGES: frozenset[str] = frozenset(
    p.name
    for p in sorted((_REPO_ROOT / "src").iterdir())
    if p.is_dir() and (p / "__init__.py").is_file()
)

#: Matches a package name wherever one is spelled out in a build file.
_PKG = r"(fsm_llm[a-z_]*)"

# DECISION plan-2026-07-21T191807-bf7ffe24/D-045
# `MANIFEST.in` has never listed `fsm_llm_monitor`. That is a PRE-EXISTING defect
# that predates the harness package, and fixing it is out of scope here (it
# changes what ships in the sdist, which deserves its own decision).
#
# Do NOT "fix" the failure by dropping the MANIFEST slot, by asserting a subset
# instead of set equality, or by adding further names to this set -- any of those
# turns a named, ratcheted exception back into an invisible hole. The exception is
# itself pinned by `test_manifest_known_gaps_are_closed` below, which is
# `xfail(strict=True)`: the moment someone adds the missing line, that test XPASSes
# and FAILS the suite, forcing this set to shrink. See decisions.md D-045.
_MANIFEST_KNOWN_GAPS = frozenset({"fsm_llm_monitor"})


def _read(rel: str) -> str:
    return (_REPO_ROOT / rel).read_text(encoding="utf-8")


def _section(rel: str, header: str) -> str:
    """Return the body of an INI/TOML section, header line excluded."""
    body: list[str] = []
    inside = False
    for line in _read(rel).splitlines():
        if line.strip() == header:
            inside = True
            continue
        if inside and line.startswith("["):
            break
        if inside:
            body.append(line)
    assert inside, f"section {header!r} not found in {rel}"
    return "\n".join(body)


def _one_line(rel: str, needle: str) -> str:
    """Return the single line of `rel` containing `needle` (asserts uniqueness)."""
    hits = [ln for ln in _read(rel).splitlines() if needle in ln]
    assert len(hits) == 1, (
        f"expected exactly 1 line containing {needle!r} in {rel}, found {len(hits)}"
    )
    return hits[0]


# ══════════════════════════════════════════════════════════════
# The slots: each extracts the package names that slot actually names
# ══════════════════════════════════════════════════════════════


def _slot_pyproject_package_data() -> set[str]:
    body = _section("pyproject.toml", "[tool.setuptools.package-data]")
    return set(re.findall(rf"^{_PKG} = \[", body, re.MULTILINE))


def _slot_pyproject_isort() -> set[str]:
    line = _one_line("pyproject.toml", "known-first-party")
    return set(re.findall(rf'"{_PKG}"', line))


def _slot_makefile_type_check() -> set[str]:
    return set(re.findall(rf"src/{_PKG}/", _one_line("Makefile", "-m mypy")))


def _slot_makefile_coverage() -> set[str]:
    return set(re.findall(rf"--cov={_PKG}\b", _one_line("Makefile", "--cov-report")))


def _slot_tox_coverage() -> set[str]:
    return set(re.findall(rf"--cov={_PKG}\b", _one_line("tox.ini", "pytest {posargs")))


def _slot_tox_type() -> set[str]:
    return set(re.findall(rf"src/{_PKG}/", _one_line("tox.ini", "mypy src/")))


def _slot_ci_mypy() -> set[str]:
    line = _one_line(".github/workflows/python-package.yml", "mypy src/")
    return set(re.findall(rf"src/{_PKG}/", line))


def _slot_manifest() -> set[str]:
    return set(
        re.findall(rf"recursive-include src/{_PKG} \*\.py", _read("MANIFEST.in"))
    )


#: slot id -> (extractor, packages this slot is documented NOT to cover)
_SLOTS: dict[str, tuple[object, frozenset[str]]] = {
    "pyproject:package-data": (_slot_pyproject_package_data, frozenset()),
    "pyproject:ruff-isort-known-first-party": (_slot_pyproject_isort, frozenset()),
    "makefile:type-check": (_slot_makefile_type_check, frozenset()),
    "makefile:coverage": (_slot_makefile_coverage, frozenset()),
    "tox:testenv-coverage": (_slot_tox_coverage, frozenset()),
    "tox:testenv-type": (_slot_tox_type, frozenset()),
    "ci:mypy": (_slot_ci_mypy, frozenset()),
    "manifest.in:recursive-include": (_slot_manifest, _MANIFEST_KNOWN_GAPS),
}


class TestEveryPackageIsWired:
    """Every package under `src/` appears in every slot that enumerates packages."""

    def test_package_list_is_not_empty(self):
        # Guards the derivation itself: an empty set would make every other
        # assertion below vacuously true.
        assert len(SRC_PACKAGES) >= 5, SRC_PACKAGES
        assert "fsm_llm" in SRC_PACKAGES

    @pytest.mark.parametrize("slot_id", sorted(_SLOTS))
    def test_slot_enumerates_every_package(self, slot_id: str):
        extract, gaps = _SLOTS[slot_id]
        expected = SRC_PACKAGES - gaps
        actual = extract()  # type: ignore[operator]
        assert actual == expected, (
            f"slot {slot_id} is out of sync with src/: "
            f"missing {sorted(expected - actual)}, "
            f"stale {sorted(actual - expected)}"
        )

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "PRE-EXISTING: MANIFEST.in has never listed fsm_llm_monitor. When this "
            "XPASSes the gap has been closed -- remove fsm_llm_monitor from "
            "_MANIFEST_KNOWN_GAPS and delete this xfail marker."
        ),
    )
    def test_manifest_known_gaps_are_closed(self):
        assert _slot_manifest() == SRC_PACKAGES

    def test_every_package_ships_a_py_typed_marker(self):
        # Makes the package-data slot meaningful: listing `py.typed` in
        # package-data does nothing if the marker file is absent from the tree.
        missing = [
            pkg
            for pkg in sorted(SRC_PACKAGES)
            if not (_REPO_ROOT / "src" / pkg / "py.typed").is_file()
        ]
        assert not missing, f"packages without a py.typed marker: {missing}"

    def test_packages_find_stays_on_auto_discovery(self):
        # Slot 5 of the wiring checklist is deliberately a no-op: setuptools
        # auto-discovers everything under `where = ["src"]`. If anyone ever adds
        # an explicit include/exclude here it becomes a NINTH enumerating slot,
        # and this file must grow an entry for it -- so pin the assumption.
        body = _section("pyproject.toml", "[tool.setuptools.packages.find]")
        assert "include" not in body and "exclude" not in body, (
            "packages.find is no longer bare auto-discovery; add it to _SLOTS"
        )


class TestPackageBackedExtrasAreInstalled:
    """Each non-core package has an extra, and every install list requests it."""

    #: `fsm_llm` is the core package and has no extra; every other package's
    #: extra is its name minus the `fsm_llm_` prefix.
    EXPECTED = frozenset(
        pkg.removeprefix("fsm_llm_") for pkg in SRC_PACKAGES if pkg != "fsm_llm"
    )

    def test_every_package_backed_extra_is_declared(self):
        body = _section("pyproject.toml", "[project.optional-dependencies]")
        declared = set(re.findall(r"^([a-z0-9_]+) = ", body, re.MULTILINE))
        assert self.EXPECTED <= declared, sorted(self.EXPECTED - declared)

    @pytest.mark.parametrize(
        "label,rel,needle",
        [
            ("pyproject:all", "pyproject.toml", None),
            ("makefile:install-dev", "Makefile", "pip install -c constraints.txt -e"),
            ("tox:testenv-extras", "tox.ini", "extras = dev,"),
            (
                "ci:install",
                ".github/workflows/python-package.yml",
                "pip install -c constraints.txt -e",
            ),
        ],
    )
    def test_install_list_requests_every_package_extra(
        self, label: str, rel: str, needle: str | None
    ):
        if needle is None:
            text = _section("pyproject.toml", "[project.optional-dependencies]")
            requested = set(re.findall(r"fsm-llm\[([a-z0-9_]+)\]", text))
        else:
            requested = set(re.findall(r"[\[,]([a-z0-9_]+)", _one_line(rel, needle)))
        missing = self.EXPECTED - requested
        assert not missing, f"{label} does not request extras: {sorted(missing)}"


# ══════════════════════════════════════════════════════════════
# Documented test counts: every hand-maintained literal == measured
# ══════════════════════════════════════════════════════════════

#: A count literal as the docs spell it: "5,107" or "1751". Comma-tolerant.
_COUNT = r"(\d[\d,]*)"


def _num(token: str) -> int:
    return int(token.replace(",", ""))


def _pinned_counts(rel: str, pattern: str) -> list[int]:
    """Every numeric token `pattern` captures in `rel`, as ints.

    Matching NOTHING is a loud failure, never a silent pass: a regex that
    quietly stopped matching (because someone rephrased the doc line) would
    recreate the exact drift hole this class exists to close.
    """
    hits = re.findall(pattern, _read(rel), re.MULTILINE)
    assert hits, (
        f"count-pinning regex {pattern!r} matched nothing in {rel} -- the doc "
        f"anchor text moved; update the pattern here, do not delete the check"
    )
    return [_num(h) for h in hits]


# DECISION plan-2026-07-22T114536-879d04a0/D-015
# The measured side comes from ONE guarded `pytest --collect-only -q` SUBPROCESS
# (sys.executable, module-scoped fixture), not from the running pytest session.
# Do NOT rewrite this to count `session.items` from inside the very process
# being counted -- that is circular (this run's own -k/-m/path selection changes
# the answer) -- and do NOT add per-suite subprocesses (one collection already
# costs seconds; nine would cost minutes). Do NOT "fix" a failure by loosening
# a regex to match nothing, by asserting `>=`, or by skipping when parsing
# fails: unparseable output FAILS loudly below. CHANGELOG.md is deliberately
# NOT pinned (frozen release history), and CLAUDE.md's "N passed / N skipped /
# N xfailed" line is a RUN result at a named commit, not a collection count --
# both exclusions are scope, not oversights. See decisions.md D-015.
@pytest.fixture(scope="module")
def measured() -> tuple[int, dict[str, int], dict[str, int]]:
    """(total, per-suite-dir counts, per-root-file counts), measured once.

    The child env drops every variable that changes what gets collected or
    probed at collection time: PYTEST_ADDOPTS (an outer `-m "not slow"`
    would deselect in the child and skew the count), FSM_LLM_HARNESS_LIVE
    (armed, the live suite's skipif probes a socket at collection), and the
    SKIP_SLOW_TESTS/TEST_REAL_LLM knobs. Docs document the FULL collection.
    `-o addopts=` neutralizes pyproject's `addopts = "-v --tb=short"` for
    the same reason: its `-v` cancels `-q` and flips the output to the
    unparseable tree format (measured, not hypothetical).
    """
    env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in (
            "PYTEST_ADDOPTS",
            "FSM_LLM_HARNESS_LIVE",
            "SKIP_SLOW_TESTS",
            "TEST_REAL_LLM",
        )
    }
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
            "tests/",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    assert proc.returncode == 0, (
        f"collection subprocess failed (rc={proc.returncode});\n"
        f"stdout tail: {proc.stdout[-2000:]}\nstderr tail: {proc.stderr[-2000:]}"
    )
    node_ids = [
        line
        for line in proc.stdout.splitlines()
        if line.startswith("tests/") and "::" in line
    ]
    summary = re.search(r"^(\d+) tests collected", proc.stdout, re.MULTILINE)
    assert summary is not None, (
        "could not find 'N tests collected' in --collect-only output -- "
        f"pytest output format changed?\ntail: {proc.stdout[-2000:]}"
    )
    total = int(summary.group(1))
    assert total == len(node_ids), (
        f"summary says {total} collected but {len(node_ids)} node ids "
        "parsed -- the node-id filter is wrong; fix it, do not skip"
    )
    per_suite: collections.Counter[str] = collections.Counter()
    per_root_file: collections.Counter[str] = collections.Counter()
    for node_id in node_ids:
        parts = node_id.split("::", 1)[0].split("/")
        if len(parts) >= 3:  # tests/<suite-dir>/<file>.py
            per_suite[parts[1]] += 1
        else:  # tests/<file>.py
            per_root_file[parts[1]] += 1
    return total, dict(per_suite), dict(per_root_file)


@pytest.mark.slow
class TestDocumentedTestCountsMatchCollection:
    """Every hand-maintained test-count literal equals `pytest --collect-only`.

    Same defect class as the package-wiring tests above: hand-maintained
    parallel copies of a filesystem-derivable fact. CLAUDE.md, README.md and
    src/fsm_llm_harness/CLAUDE.md each spell test counts as prose literals
    ("5,107 tests", a 9-suite breakdown table, "1,751 tests"); every new test
    silently strands them, and nothing errors. In the predecessor plans the
    literals drifted for days at a time (3,305/2,382 stated vs 5,107 measured)
    before a manual `--collect-only` audit caught it. This class measures once
    per run (one subprocess, marked slow) and pins every literal to it.

    Excluded on purpose: CHANGELOG.md (frozen history -- its counts describe
    the state AT each release, and must never be retro-edited) and CLAUDE.md's
    "N passed / N skipped / N xfailed" sentence (a run result at a pinned
    commit, not a collection count; re-verified by release-gate runs instead).
    """

    @pytest.mark.parametrize(
        "rel,pattern",
        [
            ("CLAUDE.md", rf"pytest -v \({_COUNT} tests\)"),
            ("CLAUDE.md", rf"Run all tests \({_COUNT} collected\)"),
            ("README.md", rf"Run full test suite \({_COUNT} tests\)"),
        ],
    )
    def test_total_literals(self, measured, rel: str, pattern: str):
        total, _, _ = measured
        for documented in _pinned_counts(rel, pattern):
            assert documented == total, (
                f"{rel} documents {documented} total tests (pattern "
                f"{pattern!r}) but --collect-only measures {total}"
            )

    def test_per_suite_table(self, measured):
        # CLAUDE.md's Testing block: `pytest tests/<suite>/  # ... (N tests)`.
        # Dict equality both ways: a NEW suite directory must be added to the
        # table, and a deleted one must leave it.
        _, per_suite, _ = measured
        documented = {
            suite: _num(count)
            for suite, count in re.findall(
                rf"^pytest tests/(test_[a-z_]+)/\s+#[^(\n]*\({_COUNT} tests\)",
                _read("CLAUDE.md"),
                re.MULTILINE,
            )
        }
        assert documented, "per-suite table regex matched nothing in CLAUDE.md"
        assert documented == per_suite, (
            f"CLAUDE.md per-suite table out of sync: doc-only "
            f"{ {k: v for k, v in documented.items() if k not in per_suite} }, "
            f"missing {sorted(set(per_suite) - set(documented))}, "
            f"wrong {[k for k in documented if per_suite.get(k) != documented[k]]} "
            f"(measured: {per_suite})"
        )

    def test_suite_sum_and_remainder(self, measured):
        # `# The N suites above sum to S. The remaining R are ...`
        total, per_suite, _ = measured
        line = re.search(
            rf"The (\d+) suites above sum to {_COUNT}\. The remaining {_COUNT} ",
            _read("CLAUDE.md"),
        )
        assert line is not None, "sum/remainder sentence not found in CLAUDE.md"
        assert int(line.group(1)) == len(per_suite)
        assert _num(line.group(2)) == sum(per_suite.values())
        assert _num(line.group(3)) == total - sum(per_suite.values())

    def test_root_file_breakdown(self, measured):
        # Every root-level test file must be listed as `tests/<name> (N)` with
        # the measured N -- and nothing extra may be listed. Derived from the
        # collection, so a brand-new root-level test file fails this until
        # CLAUDE.md names it.
        _, _, per_root_file = measured
        documented = {
            name: _num(count)
            for name, count in re.findall(
                rf"tests/(test_[a-z_]+\.py) \({_COUNT}\)", _read("CLAUDE.md")
            )
        }
        assert documented == per_root_file, (
            f"CLAUDE.md root-file breakdown {documented} != measured {per_root_file}"
        )

    def test_harness_package_doc_literals(self, measured):
        # src/fsm_llm_harness/CLAUDE.md spells its own suite's count twice
        # ("1,751 tests, 10 test files" and the status paragraph). EVERY
        # "N tests" token in that file must equal the measured harness count,
        # and "N test files" must equal the on-disk test_*.py file count.
        _, per_suite, _ = measured
        rel = "src/fsm_llm_harness/CLAUDE.md"
        harness = per_suite["test_fsm_llm_harness"]
        for documented in _pinned_counts(rel, rf"{_COUNT} tests"):
            assert documented == harness, (
                f"{rel} says {documented} tests; measured {harness}"
            )
        n_files = len(
            list((_REPO_ROOT / "tests" / "test_fsm_llm_harness").glob("test_*.py"))
        )
        for documented in _pinned_counts(rel, rf"{_COUNT} test files"):
            assert documented == n_files, (
                f"{rel} says {documented} test files; on disk: {n_files}"
            )
