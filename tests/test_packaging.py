"""Repo-wide packaging invariants: one package in `src/fsm_llm/`, wired into every slot.

Five hand-maintained files name the package: `pyproject.toml`, `Makefile`,
`tox.ini`, `.github/workflows/python-package.yml` and `MANIFEST.in`. A miss is
SILENT -- the package just quietly stops being type-checked, stops being counted
in coverage, or stops shipping in the sdist. Nothing errors.

This file is that missing check. The package list is derived from the
filesystem and `pyproject.toml`, never hardcoded, so it cannot go stale. The
layout is ONE top-level package, `fsm_llm`, the directory `src/fsm_llm/` (found by
`[tool.setuptools.packages.find]`), with seven extras-backed subpackages and the
layer packages that ship with the core (`LAYER_PACKAGES`, `security` first;
`src/fsm_llm/*/__init__.py`); the sets are derived and pinned exactly, so a
stray second top-level package or an unlisted directory package fails here.
"""

from __future__ import annotations

import ast
import collections
import configparser
import fnmatch
import importlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tomllib
import zipfile

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

_REPO_ROOT = pathlib.Path(__file__).parents[1]
_SCRIPTS = str(_REPO_ROOT / "scripts")
if _SCRIPTS not in sys.path:  # scripts/ is not a package
    sys.path.insert(0, _SCRIPTS)

import audit_pth
import doc_counts

from tests.test_fsm_llm_regression.support import package_dir


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


def _packages_under(root: pathlib.Path) -> frozenset[str]:
    """Names of the directories under `root` that have an `__init__.py`."""
    return frozenset(
        p.name
        for p in sorted(root.iterdir())
        if p.is_dir() and (p / "__init__.py").is_file()
    )


def _package_dirs() -> dict[str, pathlib.Path]:
    """Import name -> source directory, from `packages.find` in pyproject.toml.

    The top-level packages `setuptools` finds: a directory under a `where` entry
    that has an `__init__.py` and matches an `include` pattern.
    """
    find = tomllib.loads(_read("pyproject.toml"))["tool"]["setuptools"]["packages"][
        "find"
    ]
    found: dict[str, pathlib.Path] = {}
    for where in find["where"]:
        for child in sorted((_REPO_ROOT / where).iterdir()):
            if (
                child.is_dir()
                and (child / "__init__.py").is_file()
                and any(fnmatch.fnmatchcase(child.name, p) for p in find["include"])
            ):
                found[child.name] = child
    assert found, "packages.find finds no package"
    return found


#: Import name -> directory of every top-level package, from pyproject.toml.
#: A directory without an `__init__.py` is not a package; the `*.egg-info/` that
#: an editable install leaves in `src/` and any stray directory are excluded by
#: construction.
PACKAGE_DIRS: dict[str, pathlib.Path] = _package_dirs()

#: Every importable top-level package name.
SRC_PACKAGES: frozenset[str] = frozenset(PACKAGE_DIRS)

#: Every package directly under `src/fsm_llm/`, derived from disk.
_ON_DISK_PACKAGES: frozenset[str] = _packages_under(PACKAGE_DIRS["fsm_llm"])

#: The layer packages of the 0.4 layout (plan d454fd62, D-014): they ship with the core,
#: need no extra and have no per-suite test folder. The set is the layer packages that
#: exist NOW; each split step adds the name of the package it creates (security at step
#: 67, core at step 69, llm and context at step 72, handlers at step 77, engine at
#: step 78, okf at step 83), up to the target `security, core, llm, context, engine, tools, policy,
#: okf, skills, handlers`. `test_layer_packages_are_exactly_the_non_extra_packages_on_disk`
#: pins it to the disk, so a layer package with no entry here, or an entry with no
#: package, fails.
LAYER_PACKAGES: frozenset[str] = frozenset(
    {
        "context",
        "core",
        "engine",
        "handlers",
        "llm",
        "okf",
        "policy",
        "security",
        "skills",
        "tools",
    }
)

#: Every extras-backed subpackage of `fsm_llm` (they sit directly in `src/fsm_llm/`):
#: the on-disk packages that are not layer packages.
SUBPACKAGES: frozenset[str] = _ON_DISK_PACKAGES - LAYER_PACKAGES

#: The four former sibling packages that now live under `fsm_llm`, plus `eval`
#: (added in plan-2026-09-29T061903-581c2634) and `synthesis`
#: (added in plan-2026-11-10T194139-2cefeb61) and `testing`
#: (added in plan-2026-11-12T073150-d454fd62).
_EXPECTED_SUBPACKAGES = frozenset(
    {"agents", "reasoning", "workflows", "monitor", "eval", "synthesis", "testing"}
)

#: Matches a package name wherever one is spelled out in a build file.
_PKG = r"(fsm_llm[a-z_]*)"


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


def _slot_type_check(rel: str, needle: str) -> set[str]:
    """Packages type-checked by a build file: the `-p` names of its mypy line."""
    return set(re.findall(rf"\bmypy\b.*?-p {_PKG}\b", _one_line(rel, needle)))


def _slot_makefile_type_check() -> set[str]:
    return _slot_type_check("Makefile", "-m mypy")


def _slot_makefile_coverage() -> set[str]:
    return set(re.findall(rf"--cov={_PKG}\b", _one_line("Makefile", "--cov-report")))


def _slot_tox_coverage() -> set[str]:
    return set(re.findall(rf"--cov={_PKG}\b", _one_line("tox.ini", "pytest {posargs")))


def _slot_tox_type() -> set[str]:
    return _slot_type_check("tox.ini", "-m mypy")


def _slot_ci_mypy() -> set[str]:
    return _slot_type_check(".github/workflows/python-package.yml", "-m mypy")


def _slot_manifest() -> set[str]:
    # `src` is the directory holding the one package; the line names it, not the package.
    if re.search(r"^recursive-include src \*\.py$", _read("MANIFEST.in"), re.M):
        return set(SRC_PACKAGES)
    return set()


#: slot id -> extractor
_SLOTS = {
    "pyproject:package-data": _slot_pyproject_package_data,
    "pyproject:ruff-isort-known-first-party": _slot_pyproject_isort,
    "makefile:type-check": _slot_makefile_type_check,
    "makefile:coverage": _slot_makefile_coverage,
    "tox:testenv-coverage": _slot_tox_coverage,
    "tox:testenv-type": _slot_tox_type,
    "ci:mypy": _slot_ci_mypy,
    "manifest.in:recursive-include": _slot_manifest,
}


class TestEveryPackageIsWired:
    """The one package under `src/` appears in every slot that names packages."""

    def test_package_list_is_exactly_fsm_llm(self):
        # Guards the derivation itself: an empty set would make every other
        # assertion below vacuously true, and a second top-level package would
        # undo the single-package layout.
        assert SRC_PACKAGES == {"fsm_llm"}, SRC_PACKAGES

    def test_subpackage_list_is_exactly_the_seven(self):
        assert SUBPACKAGES == _EXPECTED_SUBPACKAGES, SUBPACKAGES

    def test_layer_packages_are_exactly_the_non_extra_packages_on_disk(self):
        # A directory package that is neither one of the seven extras-backed
        # subpackages nor a layer package is a miss in this file; a layer package
        # that is listed but absent is a stale entry.
        on_disk_layers = _ON_DISK_PACKAGES - _EXPECTED_SUBPACKAGES
        assert on_disk_layers == LAYER_PACKAGES, (
            f"on disk, not listed: {sorted(on_disk_layers - LAYER_PACKAGES)}; "
            f"listed, not on disk: {sorted(LAYER_PACKAGES - on_disk_layers)}"
        )
        assert not LAYER_PACKAGES & _EXPECTED_SUBPACKAGES

    @pytest.mark.parametrize("layer", sorted(LAYER_PACKAGES))
    def test_a_layer_package_needs_no_extra(self, layer: str):
        # It ships with the core: no extra of its name, so `pip install llm-fsm` has it.
        body = _section("pyproject.toml", "[project.optional-dependencies]")
        declared = set(re.findall(r"^([a-z0-9_]+) = ", body, re.MULTILINE))
        assert layer not in declared
        assert (PACKAGE_DIRS["fsm_llm"] / layer / "__init__.py").is_file()

    @pytest.mark.parametrize("slot_id", sorted(_SLOTS))
    def test_slot_enumerates_every_package(self, slot_id: str):
        actual = _SLOTS[slot_id]()
        assert actual == SRC_PACKAGES, (
            f"slot {slot_id} is out of sync with src/: "
            f"missing {sorted(SRC_PACKAGES - actual)}, "
            f"stale {sorted(actual - SRC_PACKAGES)}"
        )

    def test_every_package_ships_a_py_typed_marker(self):
        # Makes the package-data slot meaningful: listing `py.typed` in
        # package-data does nothing if the marker file is absent from the tree.
        # PEP 561: the top-level marker covers every subpackage (D-010).
        missing = [
            pkg
            for pkg in sorted(SRC_PACKAGES)
            if not (PACKAGE_DIRS[pkg] / "py.typed").is_file()
        ]
        assert not missing, f"packages without a py.typed marker: {missing}"

    def test_packages_find_matches_disk(self):
        # `[tool.setuptools.packages.find]` replaces an explicit package list: a
        # new subpackage ships without a pyproject edit. Pin that wiring (no
        # `package-dir`, no `packages` list, `include` narrowed to `fsm_llm*`,
        # `src` as the one search root) and let setuptools itself name what it
        # finds, so a subpackage missing an `__init__.py` fails here.
        from setuptools import find_packages

        setuptools_table = _pyproject()["tool"]["setuptools"]
        assert "package-dir" not in setuptools_table
        assert "packages" not in setuptools_table or isinstance(
            setuptools_table["packages"], dict
        )
        find = setuptools_table["packages"]["find"]
        assert find["where"] == ["src"]
        assert find["include"] == ["fsm_llm*"]
        assert find["namespaces"] is False
        found = set(
            find_packages(where=str(_REPO_ROOT / "src"), include=find["include"])
        )
        root = PACKAGE_DIRS["fsm_llm"]
        expected = {"fsm_llm"} | {
            "fsm_llm." + init.parent.relative_to(root).as_posix().replace("/", ".")
            for init in root.rglob("__init__.py")
            if init.parent != root and "__pycache__" not in init.parts
        }
        assert found == expected, (
            f"missing {sorted(expected - found)}, stale {sorted(found - expected)}"
        )


class TestSubpackagesImport:
    """`from fsm_llm import <sub>` works."""

    @pytest.mark.parametrize("sub", sorted(_EXPECTED_SUBPACKAGES))
    def test_subpackages_importable_from_fsm_llm(self, sub: str):
        if sub == "monitor":
            pytest.importorskip("fastapi")
        module = importlib.import_module("fsm_llm")
        importlib.import_module(f"fsm_llm.{sub}")
        assert getattr(module, sub).__name__ == f"fsm_llm.{sub}"


def _monitor_package_data_globs() -> list[str]:
    body = _section("pyproject.toml", "[tool.setuptools.package-data]")
    line = re.search(r'^"fsm_llm\.monitor" = \[(.*)\]$', body, re.MULTILINE)
    assert line is not None, '"fsm_llm.monitor" package-data entry not found'
    return re.findall(r'"([^"]+)"', line.group(1))


class TestMonitorPackageData:
    """Every monitor frontend file is shipped (D-006: `static/*` missed the
    `pages/`, `services/` and `utils/` subdirectories)."""

    def test_monitor_package_data_covers_every_static_file(self):
        monitor = PACKAGE_DIRS["fsm_llm"] / "monitor"
        covered = {
            path
            for pattern in _monitor_package_data_globs()
            for path in monitor.glob(pattern)
            if path.is_file()
        }
        wanted = {
            path
            for sub in ("static", "templates")
            for path in (monitor / sub).rglob("*")
            if path.is_file()
        }
        # Non-vacuity: nested frontend files exist, which a flat `static/*`
        # glob would not cover.
        static = monitor / "static"
        assert any(p.parent != static for p in static.rglob("*") if p.is_file())
        missing = sorted(str(p.relative_to(monitor)) for p in wanted - covered)
        assert not missing, f"monitor files not covered by package-data: {missing}"


class TestPackageBackedExtrasAreInstalled:
    """Each subpackage has an extra, and every install list requests it."""

    #: `fsm_llm` is the core package and has no extra; every subpackage's
    #: extra is its name.
    EXPECTED = SUBPACKAGES

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
        ],
    )
    def test_install_list_requests_every_package_extra(
        self, label: str, rel: str, needle: str | None
    ):
        if needle is None:
            requested = _self_referenced_extras()
        else:
            requested = set(re.findall(r"[\[,]([a-z0-9_]+)", _one_line(rel, needle)))
        missing = self.EXPECTED - requested
        assert not missing, f"{label} does not request extras: {sorted(missing)}"

    def test_every_ci_install_line_requests_every_package_extra(self):
        """CI installs the package in three jobs (`lint`, the `test` matrix and `supply-chain`),
        one line each: all name every extra, so mypy, the matrix and the vulnerability audit see
        the same environment (the count and the `mcp` extra are pinned by
        `test_ci_workflow.py`, the identity of the audit's line by `test_supply_chain.py`)."""
        rel = ".github/workflows/python-package.yml"
        lines = [
            ln
            for ln in _read(rel).splitlines()
            if "pip install -c constraints.txt -e" in ln
        ]
        assert len(lines) == 3, lines
        for line in lines:
            requested = set(re.findall(r"[\[,]([a-z0-9_]+)", line))
            missing = self.EXPECTED - requested
            assert not missing, f"ci:install does not request extras: {sorted(missing)}"


# ══════════════════════════════════════════════════════════════
# Module docstrings are real docstrings (not inert strings after a future import)
# ══════════════════════════════════════════════════════════════

#: Every module under `src/`, derived from disk.
SRC_MODULES: tuple[pathlib.Path, ...] = tuple(
    sorted((_REPO_ROOT / "src").rglob("*.py"))
)


def _future_import_precedes_docstring(path: pathlib.Path) -> bool:
    """True when the module's first statement is a `from __future__` import and
    its SECOND statement is a bare string literal -- the string the author meant
    as the module docstring, which Python then leaves out of `__doc__`."""
    body = ast.parse(path.read_text(encoding="utf-8")).body
    if len(body) < 2:
        return False
    first, second = body[0], body[1]
    return (
        isinstance(first, ast.ImportFrom)
        and first.module == "__future__"
        and isinstance(second, ast.Expr)
        and isinstance(second.value, ast.Constant)
        and isinstance(second.value.value, str)
    )


class TestCiRunsEveryTestOfItsMarkerSet:
    """CI selects tests by marker only. A `--deselect` hides a test behind a
    node id that stays in the file after the test is renamed, so the exclusion
    silently outlives its reason (the one CI carried named a test that had not
    existed since a rename)."""

    def test_the_ci_pytest_commands_deselect_nothing(self):
        # The marker selection is the job env `PYTEST_MARKERS`; the plain and the coverage
        # form of the one selection are the two commands that read it.
        lines = [
            ln
            for ln in _read(".github/workflows/python-package.yml").splitlines()
            if 'python -m pytest tests/ -q -m "$PYTEST_MARKERS"' in ln
        ]
        assert len(lines) == 2, lines
        assert not [ln for ln in lines if "--deselect" in ln], lines

    def test_no_workflow_or_task_file_deselects_a_test(self):
        for rel in ("Makefile", "tox.ini", ".github/workflows/python-package.yml"):
            code = [
                ln for ln in _read(rel).splitlines() if not ln.lstrip().startswith("#")
            ]
            assert not [ln for ln in code if "--deselect" in ln], rel


class TestModuleDocstringsAreReal:
    """plan-2026-09-20T165703-0d9c218e D-003: a module docstring placed AFTER
    `from __future__ import annotations` is an inert expression, so
    `module.__doc__` is None and `help()` shows nothing. 81 modules had this
    defect; the sweep moved every docstring above the future import."""

    def test_module_list_is_not_empty(self):
        assert len(SRC_MODULES) >= 50, len(SRC_MODULES)

    def test_no_module_hides_its_docstring_behind_a_future_import(self):
        offenders = [
            str(p.relative_to(_REPO_ROOT))
            for p in SRC_MODULES
            if _future_import_precedes_docstring(p)
        ]
        assert not offenders, (
            "module docstring sits after `from __future__ import annotations` "
            f"(so __doc__ is None) in: {offenders}"
        )


#: A test file or folder named after the audit, review or plan step that wrote it.
#: Tests are named for the behaviour they pin (`test_<module>.py`), so a dated or
#: process name is refused and the class goes into a feature file. Scope: every
#: `.py` file and every folder under `tests/` (`__pycache__` skipped); data files
#: (JSON, text) are out of scope, so a year inside a fixture name is not matched.
#:
#: The rule works on WHOLE name segments (the stem split on `_`, `-` and `.`), so a
#: word that only contains a refused word is clean (`preview`, `rfc2045`,
#: `reviewer`, `prefixes`, `round_robin`). A name is dated when one segment is
#: `fixes`, `round<N>`, `iter<N>`, `phase<N>`, `step<N>` or `gate<N>` (the digit is
#: required, so `iter`, `round` and `phase` in a feature name stay allowed), a
#: counted process word followed by a number segment (`round_2`, `iter-2`), a year
#: from 2020 to 2030 or a segment that starts with one (`2026`, `2026nov`), a
#: date-like 8-digit run (`20261113`), or a compound (`reviewfixes`, `auditfixes`,
#: `reviewround2`). `review` and `audit` are real feature words here (`review_queue`,
#: `build_review_presentation`, `audit_log`, `policy_audit`), so they are refused
#: only where they name the process: a trailing or bare `review` (`test_review.py`,
#: `test_regression_review.py`), a bare `audit` (`test_audit.py`), and `audit`
#: followed by `sweep(s)` (the other process words after `review` or `audit`, such
#: as `fixes`, `round<N>`, `iter<N>` and a year, are refused on their own). A
#: trailing `audit` (`test_policy_audit.py`) and `review` before a feature word
#: (`test_review_queue.py`) are clean. Review gate 1 concern 16 (plan d454fd62 step
#: 58.1) replaced the substring regex that refused `test_preview.py`; the re-review
#: (step 58.2, concern 5) let `review` and a trailing `audit` through as feature
#: words and caught the year, date and compound shapes the first segment rule
#: missed. The accept and refuse tables below and the 38 historical names pin it;
#: `test_regression_review.py` is the one historical name that needs the
#: trailing-`review` rule.
_PROCESS_SEGMENT = re.compile(r"fixes|round\d+|iter\d+|phase\d+|step\d+|gate\d+")
_YEAR_SEGMENT = re.compile(r"20(?:2\d|30)[a-z0-9]*")
_DATE_SEGMENT = re.compile(r"(?:19|20)\d\d(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])")
_COMPOUND_SEGMENT = re.compile(
    r"(?:review|audit)(?:fixes|rounds?\d*|iters?\d+|phases?\d+|sweeps?)"
)
_COUNTED_WORDS = frozenset({"round", "iter", "phase", "step", "gate"})


def _is_dated_name(name: str) -> bool:
    """True when a file or folder name carries a dated or process segment."""
    stem = name.lower().removesuffix(".py")
    segments = [s for s in re.split(r"[_.\-]+", stem) if s]
    words = [s for s in segments if s != "test"]
    for position, segment in enumerate(segments):
        following = segments[position + 1] if position + 1 < len(segments) else None
        if (
            _PROCESS_SEGMENT.fullmatch(segment)
            or _YEAR_SEGMENT.fullmatch(segment)
            or _DATE_SEGMENT.fullmatch(segment)
            or _COMPOUND_SEGMENT.fullmatch(segment)
        ):
            return True
        if segment in _COUNTED_WORDS and following and following.isdigit():
            return True
        if segment == "review" and following is None:
            return True
        if segment == "audit" and (
            words == ["audit"] or following in ("sweep", "sweeps")
        ):
            return True
    return False


def _dated_test_names(names: list[str]) -> list[str]:
    """The repo-relative `tests/` paths (posix) whose file or folder name is dated."""
    return [n for n in names if _is_dated_name(n.rsplit("/", 1)[-1])]


class TestNoDatedTestFiles:
    """Test files are named for behaviour, never for the audit that found it.

    Steps 34 to 42 of plan-2026-11-12T073150-d454fd62 folded 41 files named
    `test_audit_*`, `test_*_review_*`, `test_*_fixes`, `test_*_iter2` and the like
    into feature files; this guard keeps that state.
    """

    @pytest.mark.parametrize(
        "name",
        [
            "tests/test_fsm_llm/test_audit_iter1_seam.py",
            "tests/test_fsm_llm_agents/test_review_fixes_round2_agents.py",
            "tests/test_fsm_llm_monitor/test_audit_2026_09_28.py",
            "tests/test_fsm_llm_regression/test_regression_iter2.py",
            "tests/test_fsm_llm_agents/test_strands_phase2.py",
            "tests/test_fsm_llm_agents/test_bug_fixes.py",
            "tests/test_fsm_llm_review_round1",
        ],
    )
    def test_the_pattern_refuses_a_dated_name(self, name):
        assert _dated_test_names([name]) == [name]

    @pytest.mark.parametrize(
        "name",
        [
            "tests/test_fsm_llm/test_okf.py",
            "tests/test_fsm_llm/test_okf_cli.py",
            "tests/test_fsm_llm_agents/test_removed_legacy.py",
            "tests/test_fsm_llm_regression/test_regression_cli_and_exports.py",
            "tests/test_fsm_llm_workflows/test_wait_events.py",
            "tests/test_fsm_llm/test_iteration_budget.py",
            "tests/test_fsm_llm/test_rounding.py",
            "tests/test_fsm_llm/fixtures/context_key_corpus.py",
        ],
    )
    def test_the_pattern_keeps_a_behaviour_name(self, name):
        assert _dated_test_names([name]) == []

    @pytest.mark.parametrize(
        "name",
        [
            # Review gate 1 concern 16: the refuse table of the segment rule.
            "tests/test_fsm_llm/test_review_round2_core.py",
            "tests/test_fsm_llm/test_audit_2026_09_21.py",
            "tests/test_fsm_llm/test_okf_cli_frontmatter_fixes.py",
            "tests/test_fsm_llm/test_audit_sweeps.py",
            "tests/test_fsm_llm/test_audit.py",
            "tests/test_fsm_llm/test_regression_iter12.py",
            "tests/test_fsm_llm/test_strands_phase10.py",
            "tests/test_fsm_llm/test_round3.py",
            "tests/test_fsm_llm/test_fixes.py",
            "tests/test_fsm_llm/test_notes-2026-review.py",
            "tests/test_fsm_llm/test_cleanup.2026.py",
            "tests/test_fsm_llm_audit_2026_09",
            # Re-review concern 5: year, date, compound and counted-word shapes.
            "tests/test_fsm_llm/test_20261113.py",
            "tests/test_fsm_llm/test_2026nov.py",
            "tests/test_fsm_llm/test_reviewfixes.py",
            "tests/test_fsm_llm/test_auditfixes.py",
            "tests/test_fsm_llm/test_reviewround2.py",
            "tests/test_fsm_llm/test_round_2.py",
            "tests/test_fsm_llm/test_iter-2.py",
            "tests/test_fsm_llm/test_step58.py",
            "tests/test_fsm_llm/test_gate1.py",
            "tests/test_fsm_llm/test_review.py",
            "tests/test_fsm_llm/test_regression_review.py",
            "tests/test_fsm_llm_review_2025",
        ],
    )
    def test_the_segment_rule_refuses(self, name):
        assert _dated_test_names([name]) == [name]

    @pytest.mark.parametrize(
        "name",
        [
            # The accept table: a word that merely CONTAINS a refused word, or a
            # feature that is called an audit log, is a behaviour name.
            "tests/test_fsm_llm/test_preview.py",
            "tests/test_fsm_llm/test_previews_of_states.py",
            "tests/test_fsm_llm/test_rfc2045.py",
            "tests/test_fsm_llm/test_http2020s.py",
            "tests/test_fsm_llm/test_audit_log.py",
            "tests/test_fsm_llm/test_audit_trail_handler.py",
            "tests/test_fsm_llm/test_audit_events.py",
            "tests/test_fsm_llm/test_prefixes.py",
            "tests/test_fsm_llm/test_round_robin.py",
            "tests/test_fsm_llm/test_iterable_tools.py",
            "tests/test_fsm_llm/test_phase_shift.py",
            "tests/test_fsm_llm/test_reviewer_agent.py",
            "tests/test_fsm_llm/test_sha2562.py",
            "tests/test_fsm_llm_audit_log",
            # Re-review concern 5: `review` and `audit` are feature words.
            "tests/test_fsm_llm/test_review_presentation.py",
            "tests/test_fsm_llm/test_review_queue.py",
            "tests/test_fsm_llm/test_code_review_agent.py",
            "tests/test_fsm_llm/test_policy_audit.py",
            "tests/test_fsm_llm/test_security_audit.py",
            "tests/test_fsm_llm/test_year_2038.py",
            "tests/test_fsm_llm/test_iso8601_dates.py",
        ],
    )
    def test_the_segment_rule_accepts(self, name):
        assert _dated_test_names([name]) == []

    #: The distinct basenames of the 41 dated files that step 33 to 42 folded away
    #: (41 files, 38 distinct names: `test_audit_2026_11_05.py` and
    #: `test_audit_fixes.py` existed in two folders), read from `git ls-tree
    #: 4b24387d -r tests`. The segment rule must keep refusing every one.
    _HISTORICAL_DATED_NAMES = (
        "test_audit_2026_09_21.py",
        "test_audit_2026_09_22.py",
        "test_audit_2026_09_27.py",
        "test_audit_2026_09_28.py",
        "test_audit_2026_11_05.py",
        "test_audit_fixes.py",
        "test_audit_iter1_seam.py",
        "test_audit_iter2_seam.py",
        "test_audit_iter3_seam.py",
        "test_audit_iter4_seam.py",
        "test_audit_sweeps.py",
        "test_bug_fixes.py",
        "test_core_behaviour_fixes.py",
        "test_definitions_behaviour_fixes.py",
        "test_epistemic_fixes.py",
        "test_handlers_visualizer_fixes.py",
        "test_instance_manager_fixes.py",
        "test_integrations_fixes.py",
        "test_llm_layer_behaviour_fixes.py",
        "test_logging_behaviour_fixes.py",
        "test_meta_cleanup_fixes.py",
        "test_okf_cli_frontmatter_fixes.py",
        "test_patterns_fixes.py",
        "test_regression_iter2.py",
        "test_regression_review.py",
        "test_review_fixes.py",
        "test_review_fixes_core.py",
        "test_review_fixes_core_round2.py",
        "test_review_fixes_meta.py",
        "test_review_fixes_round2_agents.py",
        "test_review_fixes_tools_native.py",
        "test_review_round1_agents.py",
        "test_review_round1_core.py",
        "test_review_round2_agents.py",
        "test_review_round2_callers.py",
        "test_review_round2_core.py",
        "test_security_review_fixes.py",
        "test_strands_phase2.py",
    )

    def test_the_segment_rule_still_refuses_every_historical_dated_name(self):
        assert len(self._HISTORICAL_DATED_NAMES) == 38
        names = [f"tests/x/{n}" for n in self._HISTORICAL_DATED_NAMES]
        assert _dated_test_names(names) == names

    def test_no_test_file_or_folder_under_tests_has_a_dated_name(self):
        tests = _REPO_ROOT / "tests"
        names = [
            p.relative_to(_REPO_ROOT).as_posix()
            for p in sorted(tests.rglob("*"))
            if "__pycache__" not in p.parts and (p.is_dir() or p.suffix == ".py")
        ]
        assert len(names) > 100, "the walk found too few paths to mean anything"
        assert _dated_test_names(names) == [], (
            "name a test file for the behaviour it pins (a feature file), not "
            "for the audit, review round or plan step that found it"
        )


class TestNoTopLevelNameIsBoundTwice:
    """A doubled top-level name (a copy-paste slip) silently keeps the last one.
    Ruff F811 sees a redefined def or import, not a doubled constant. `if` and
    `try` branches are not scanned (a fallback binding is deliberate)."""

    @staticmethod
    def _bound_names(path: pathlib.Path) -> list[str]:
        names: list[str] = []
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    names += [
                        n.id
                        for n in ast.walk(target)
                        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
                    ]
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                if isinstance(node.target, ast.Name):
                    names.append(node.target.id)
            elif isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                overload = any(
                    "overload" in ast.unparse(d)
                    for d in getattr(node, "decorator_list", [])
                )
                if not overload:
                    names.append(node.name)
        return names

    def test_no_src_module_binds_a_top_level_name_twice(self):
        offenders = {}
        for path in SRC_MODULES:
            names = self._bound_names(path)
            twice = sorted({n for n in names if names.count(n) > 1})
            if twice:
                offenders[str(path.relative_to(_REPO_ROOT))] = twice
        assert not offenders, offenders


#: The install hint of another project: `pip install fsm-llm`, with options, quotes or extras
#: (`pip install -U "fsm-llm[a]"`). `llm-fsm` (this project) never matches, and neither does
#: `fsm-llm-<x>` (a different name).
_OTHER_PROJECT_HINT = re.compile(r"pip\s+install\s+(?:-\S+\s+)*[\"']?fsm-llm(?![-\w.])")

#: Tracked text where an install hint may stand. The CHANGELOG and the roadmaps are history and
#: plan text (they quote the old name on purpose); `tests/` documents the ban itself; `examples/`
#: are frozen baselines (AGENTS.md: never edited unless asked); `plans/` is not tracked.
_HINT_SCAN_ROOTS = ("src", "docs", ".github")
_HINT_SCAN_FILES = (
    "README.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "AGENTS.md",
    "pyproject.toml",
    "Makefile",
    "tox.ini",
    ".env.example",
)
_HINT_SCAN_SKIP = ("docs/2026_roadmap.md",)
_HINT_SUFFIXES = (".py", ".md", ".yml", ".yaml", ".toml", ".txt", ".ini")


def _hint_scan_files() -> list[pathlib.Path]:
    found = [_REPO_ROOT / f for f in _HINT_SCAN_FILES]
    for root in _HINT_SCAN_ROOTS:
        found += [
            p
            for p in sorted((_REPO_ROOT / root).rglob("*"))
            if p.is_file()
            and p.suffix in _HINT_SUFFIXES
            and "__pycache__" not in p.parts
        ]
    skip = {_REPO_ROOT / s for s in _HINT_SCAN_SKIP}
    return [p for p in found if p.is_file() and p not in skip]


class TestNoPypiInstallHint:
    """`fsm-llm` on PyPI is another project: no tracked text tells a reader to install it.

    This project is published as `llm-fsm` (step 113 of plan d454fd62, D-020), so
    `pip install llm-fsm` and `pip install "llm-fsm[extra]"` are the legal shapes.
    """

    def test_no_pypi_install_hint_in_src(self):
        needle = "pip install fsm-llm"
        hits = [
            f"{p.relative_to(_REPO_ROOT)}:{n}"
            for p in sorted((_REPO_ROOT / "src").rglob("*.py"))
            for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
            if needle in line
        ]
        assert not hits, f"hint that installs another project: {hits}"

    def test_no_hint_of_the_other_project_in_tracked_text(self):
        files = _hint_scan_files()
        assert len(files) > 200, len(
            files
        )  # non-vacuous: the scan really walks src/docs
        hits = [
            f"{p.relative_to(_REPO_ROOT)}:{n}"
            for p in files
            for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
            if _OTHER_PROJECT_HINT.search(line)
        ]
        assert not hits, f"hint that installs another project: {hits}"

    @pytest.mark.parametrize(
        "line",
        [
            "pip install fsm-llm",
            'pip install "fsm-llm[agents]"',
            "pip install 'fsm-llm[a,b]'",
            "pip install -U fsm-llm",
            "python -m pip install --upgrade fsm-llm",
            "pip  install  fsm-llm[monitor]",
            "Install with: pip install fsm-llm[agents,reasoning]",
        ],
    )
    def test_the_other_projects_hint_is_recognised(self, line: str):
        assert _OTHER_PROJECT_HINT.search(line)

    @pytest.mark.parametrize(
        "line",
        [
            "pip install llm-fsm",
            'pip install "llm-fsm[agents]"',
            "pip install --force-reinstall llm-fsm",
            "pip install -e .",
            "pip install fsm-llm-extra",
            "fsm-llm --fsm x.json",
        ],
    )
    def test_this_projects_hint_and_other_text_is_not(self, line: str):
        assert not _OTHER_PROJECT_HINT.search(line)

    def test_the_clone_install_hint_is_gone_from_python_source(self):
        # The nine hints that said `From a clone: pip install -e ".[x]"` name the PyPI
        # distribution now (`test_distribution_name.py` pins each one with its extra).
        hits = [
            f"{p.relative_to(_REPO_ROOT)}:{n}"
            for p in sorted((_REPO_ROOT / "src").rglob("*.py"))
            for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
            if "pip install -e" in line or "From a clone" in line
        ]
        assert not hits, f"stale clone-install hint: {hits}"


#: Newest Python minor that has a classifier. Raised (with evidence from the
#: uv runs) when 3.13 to 3.15 are classified; it is a constant so that adding a
#: classifier without the evidence step fails here.
_NEWEST_CLASSIFIED_MINOR = 12


def _pyproject() -> dict:
    return tomllib.loads(_read("pyproject.toml"))


#: How many `llm-fsm[extra]` self-references the `all` extra holds today (one per extra that
#: is not `dev`). The extraction below must find at least this many, so a rename that leaves the
#: references on the old name can never make it match nothing and pass on an empty set.
_SELF_REFERENCES_TODAY = 10


def _self_referenced_extras() -> set[str]:
    """Extras the `all` extra requests from the distribution itself (`llm-fsm[x]` entries).

    Reads `[project].name` and parses every entry of `all` with `packaging`; an entry
    that names another distribution, or the project under a stale name, fails here,
    so the set cannot be silently empty or point at a different project.
    """
    project = _pyproject()["project"]
    entries = project["optional-dependencies"]["all"]
    assert len(entries) >= _SELF_REFERENCES_TODAY, entries
    requested: set[str] = set()
    for entry in entries:
        requirement = Requirement(entry)
        assert _normalized(requirement.name) == _normalized(project["name"]), (
            f"`all` entry {entry!r} does not name the project {project['name']!r}"
        )
        assert len(requirement.extras) == 1, entry
        requested |= requirement.extras
    return requested


def _python_floor_minor() -> int:
    spec = _pyproject()["project"]["requires-python"]
    m = re.fullmatch(r">=\s*3\.(\d+)", spec)
    assert m, f"requires-python must be a plain '>=3.N' floor, got {spec!r}"
    return int(m.group(1))


def _classified_minors() -> list[int]:
    prefix = "Programming Language :: Python :: 3."
    return sorted(
        int(c[len(prefix) :])
        for c in _pyproject()["project"]["classifiers"]
        if c.startswith(prefix)
    )


class TestClassifiersMatchFloor:
    """`requires-python` and the version classifiers are one fact, stated twice."""

    def test_the_floor_is_python_3_11(self):
        assert _python_floor_minor() == 11

    def test_classifiers_run_contiguously_from_the_floor(self):
        floor = _python_floor_minor()
        minors = _classified_minors()
        assert minors, "no Python version classifier"
        assert minors[0] == floor, f"oldest classifier 3.{minors[0]}, floor 3.{floor}"
        assert minors == list(range(floor, minors[-1] + 1)), minors

    def test_the_newest_classifier_is_the_evidenced_version(self):
        assert _classified_minors()[-1] == _NEWEST_CLASSIFIED_MINOR

    def test_setuptools_floor_reads_the_license_expression(self):
        """`license = "Apache-2.0"` (PEP 639) needs setuptools 77 or later."""
        requires = _pyproject()["build-system"]["requires"]
        floors = [
            int(m.group(1))
            for r in requires
            if (m := re.fullmatch(r"setuptools\s*>=\s*(\d+)(?:\.\d+)*", r.strip()))
        ]
        assert floors and floors[0] >= 77, requires

    def test_the_build_requires_holds_only_the_backend(self):
        """Review gate 1 concern 17: `wheel` was listed without a bound. setuptools
        77 builds wheels, editable wheels and sdists itself (`bdist_wheel` has
        been part of it since 70.1), so a second build dependency is one more
        unbounded name to pin and keep."""
        requires = _pyproject()["build-system"]["requires"]
        assert [_normalized(Requirement(r).name) for r in requires] == ["setuptools"]


def _normalized(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


#: The distribution name of this project (a different project owns `fsm-llm` on PyPI).
_DISTRIBUTION = "llm-fsm"


def _dist_stem(name: str | None = None) -> str:
    """The file-name form of a distribution name: `llm-fsm` is `llm_fsm` (PEP 427, 625).

    Wheel, sdist, `.dist-info` and the editable `.pth` all start with it; reads
    `[project] name` when no name is given.
    """
    if name is None:
        name = _pyproject()["project"]["name"]
    return re.sub(r"[-_.]+", "_", name).lower()


def _declared_requirements() -> list[tuple[str, Requirement]]:
    """(where, requirement) for every non-self requirement pyproject.toml declares.

    Reads `[project].dependencies`, every extra and `[dependency-groups]` (string
    entries; an `{include-group = ...}` table repeats another group). A
    `<project name>[extra]` entry is the package naming itself and is skipped.
    """
    data = _pyproject()
    found: list[tuple[str, str]] = [
        ("project.dependencies", r) for r in data["project"].get("dependencies", [])
    ]
    for extra, reqs in data["project"].get("optional-dependencies", {}).items():
        found += [(f"extra:{extra}", r) for r in reqs]
    for group, reqs in data.get("dependency-groups", {}).items():
        found += [(f"group:{group}", r) for r in reqs if isinstance(r, str)]
    out = [(where, Requirement(text)) for where, text in found]
    own = _normalized(data["project"]["name"])
    return [(w, r) for w, r in out if _normalized(r.name) != own]


def _constraint_pins() -> dict[str, list[str]]:
    """`name -> [version, ...]` for every `name==version` line of constraints.txt.

    A line that is not an exact pin fails here, so the file stays a pure pin list.
    """
    pins: dict[str, list[str]] = {}
    for raw in _read("constraints.txt").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = re.fullmatch(
            r"([A-Za-z0-9][A-Za-z0-9._-]*)==([0-9][A-Za-z0-9.!+_-]*)", line
        )
        assert m, f"constraints.txt line is not an exact pin: {raw!r}"
        pins.setdefault(_normalized(m.group(1)), []).append(m.group(2))
    return pins


def _has_upper_bound(spec: SpecifierSet) -> bool:
    return any(s.operator in {"<", "<=", "==", "~="} for s in spec)


class TestDeclaredRangesMatchPins:
    """No declared range admits an untested major; every pin lies in its range.

    A range with no upper bound lets a fresh install pick a release nobody ran the
    suite on. The exact pins in `constraints.txt` are the tested set, so a declared
    range is acceptable only when it is bounded and contains that pin.
    """

    def test_every_declared_requirement_has_an_upper_bound(self):
        unbounded = [
            f"{where}: {req}"
            for where, req in _declared_requirements()
            if not _has_upper_bound(req.specifier)
        ]
        assert not unbounded, unbounded

    def test_every_declared_requirement_has_one_exact_pin_in_its_range(self):
        pins = _constraint_pins()
        problems = []
        for where, req in _declared_requirements():
            versions = pins.get(_normalized(req.name), [])
            if len(versions) != 1:
                problems.append(f"{where}: {req} has {len(versions)} pins")
            elif not req.specifier.contains(Version(versions[0]), prereleases=True):
                problems.append(
                    f"{where}: pin {req.name}=={versions[0]} is outside {req}"
                )
        assert not problems, problems

    def test_the_dev_extra_installs_the_build_backend_the_tests_import(self):
        """Step 115.1 (Python 3.13 evidence run): `test_packages_find_matches_disk` imports
        `setuptools` and the wheel tests build with `--no-isolation`, but nothing in the
        `dev` extra installed it: `ensurepip` stopped seeding setuptools in Python 3.12 and
        `pip install -e` only puts it into an isolated build environment, so on every
        CI leg from 3.12 up the test failed with `ModuleNotFoundError`. On 3.11 the
        virtualenv seed hid it."""
        importer = re.compile(r"^\s*(?:import|from)\s+setuptools\b", re.MULTILINE)
        importers = [
            p.relative_to(_REPO_ROOT).as_posix()
            for p in sorted((_REPO_ROOT / "tests").rglob("*.py"))
            if importer.search(p.read_text(encoding="utf-8"))
        ]
        assert importers, (
            "no test imports setuptools any more: drop this rule and the pin"
        )
        dev = {
            _normalized(Requirement(r).name)
            for r in _pyproject()["project"]["optional-dependencies"]["dev"]
        }
        assert "setuptools" in dev, (
            f"{importers} import setuptools, the dev extra lacks it"
        )

    def test_the_build_backend_is_pinned_inside_its_range(self):
        pins = _constraint_pins()
        for text in _pyproject()["build-system"]["requires"]:
            req = Requirement(text)
            versions = pins.get(_normalized(req.name), [])
            assert len(versions) == 1, f"{text!r} needs one pin in constraints.txt"
            assert req.specifier.contains(Version(versions[0]), prereleases=True), (
                f"{req.name}=={versions[0]} is outside {text!r}"
            )

    def test_constraints_hold_exactly_one_litellm_line(self):
        """D-002 of plan c1d5bfbc: one litellm pin for every Python, no marker split."""
        assert len(_constraint_pins()["litellm"]) == 1
        lines = [
            ln
            for ln in _read("constraints.txt").splitlines()
            if ln.strip().startswith("litellm")
        ]
        assert len(lines) == 1, lines


# ══════════════════════════════════════════════════════════════
# Wheel content (step 61 of plan d454fd62): the wheel ships the package and
# its package data, under `fsm_llm/`, and nothing from the rest of the repo.
# Every path below is relative to the PACKAGE ROOT (`package_dir`, which is
# `src/fsm_llm/`; it was `src/` while the layout was flat), so the check held on
# both sides of that move.
# ══════════════════════════════════════════════════════════════

#: Non-`.py` files the wheel carries, relative to the package root. Written out
#: by hand on purpose: derived from `[tool.setuptools.package-data]`, a dropped
#: glob would shrink both sides of the comparison and still pass.
_WHEEL_DATA_GLOBS = ("py.typed", "monitor/static/**/*", "monitor/templates/*")

#: Repo-root files setuptools reads while building (readme, license, manifest).
_BUILD_ROOT_FILES = ("pyproject.toml", "README.md", "LICENSE", "MANIFEST.in")

#: Top-level repo directories that must never reach a wheel.
_NEVER_SHIPPED = ("tests", "examples", "scripts", "docs", "plans", "src")


def _tracked(*paths: str) -> list[str]:
    """Repo-relative tracked files under `paths`; skips the test without git."""
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not installed")
    done = subprocess.run(
        [git, "-C", str(_REPO_ROOT), "ls-files", "-z", "--", *paths],
        capture_output=True,
        check=False,
    )
    if done.returncode != 0:
        pytest.skip("not a git checkout")
    return [p for p in done.stdout.decode("utf-8").split("\0") if p]


def _stage_tracked_tree(dest: pathlib.Path) -> None:
    """Copy the tracked `src/` tree and the build root files into `dest`.

    The build runs on this copy, so setuptools leaves no `build/` or
    `*.egg-info` in the repository and an untracked stray file in the working
    tree cannot change what is shipped.
    """
    for rel in [*_tracked("src"), *_BUILD_ROOT_FILES]:
        source = _REPO_ROOT / rel
        if source.is_file():
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def _expected_wheel_files(staged: pathlib.Path) -> set[str]:
    """Wheel entries a staged tree must produce: every module, py.typed and the data."""
    root = package_dir(staged)
    rels = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*.py")
        if "__pycache__" not in p.parts
    }
    for pattern in _WHEEL_DATA_GLOBS:
        rels |= {
            p.relative_to(root).as_posix() for p in root.glob(pattern) if p.is_file()
        }
    return {f"fsm_llm/{rel}" for rel in rels}


#: Output (stdout + stderr) of the wheel builds `_build_wheel` ran, for the warning check.
_WHEEL_BUILD_OUTPUT: list[str] = []


def _build_wheel(staged: pathlib.Path, outdir: pathlib.Path) -> pathlib.Path:
    """Build a wheel from `staged` with the interpreter's own `build` and setuptools."""
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(outdir),
            str(staged),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=600,
        check=False,
    )
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    _WHEEL_BUILD_OUTPUT.append(done.stdout + done.stderr)
    wheels = sorted(outdir.glob("*.whl"))
    assert len(wheels) == 1, wheels
    return wheels[0]


def _wheel_file_problems(names: set[str], expected: set[str]) -> list[str]:
    """Problems between a wheel's entries and the expected package files."""
    problems: list[str] = []
    payload = {n for n in names if not n.split("/")[0].endswith(".dist-info")}
    for name in sorted(payload):
        if name.split("/")[0] in _NEVER_SHIPPED:
            problems.append(f"{name}: repo content that must never ship")
        elif not name.startswith("fsm_llm/"):
            problems.append(f"{name}: outside the top-level fsm_llm/ directory")
    inside = {n for n in payload if n.startswith("fsm_llm/")}
    problems += [f"{n}: shipped but not expected" for n in sorted(inside - expected)]
    problems += [f"{n}: expected but not shipped" for n in sorted(expected - payload)]
    return problems


def _dist_info_problems(wheel: pathlib.Path) -> list[str]:
    """Problems in the wheel's metadata: project name, version and console scripts."""
    project = _pyproject()["project"]
    stem = _dist_stem(project["name"])
    problems: list[str] = []
    with zipfile.ZipFile(wheel) as archive:
        dirs = {n.split("/")[0] for n in archive.namelist() if ".dist-info/" in n}
        wanted = f"{stem}-{project['version']}.dist-info"
        if dirs != {wanted}:
            return [f"dist-info directories are {sorted(dirs)}, expected {wanted}"]
        metadata = archive.read(f"{wanted}/METADATA").decode("utf-8")
        names = [ln[6:] for ln in metadata.splitlines() if ln.startswith("Name: ")]
        if names != [project["name"]]:
            problems.append(f"METADATA Name is {names}, expected {project['name']!r}")
        entry_points = configparser.ConfigParser(delimiters=("=",), interpolation=None)
        entry_points.optionxform = str  # type: ignore[assignment, method-assign]
        entry_points.read_string(archive.read(f"{wanted}/entry_points.txt").decode())
        shipped = dict(entry_points["console_scripts"])
        declared = dict(project["scripts"])
        if shipped != declared:
            problems.append(
                f"entry_points.txt {shipped} != [project.scripts] {declared}"
            )
    return problems


class TestTheDistributionIsLlmFsm:
    """Non-slow facts of the rename (step 113, D-020) that need no build.

    `[project] name` decides the wheel, sdist, `.dist-info` and `.pth` file names;
    the slow classes below check the built files themselves.
    """

    def test_the_project_name_is_llm_fsm(self):
        assert _pyproject()["project"]["name"] == _DISTRIBUTION

    def test_the_file_name_form_is_llm_fsm(self):
        assert _dist_stem() == "llm_fsm"
        assert _dist_stem("LLM.fsm") == "llm_fsm"
        assert _dist_stem("fsm-llm") == "fsm_llm"  # the other project's form differs

    def test_the_self_references_of_the_all_extra_are_found(self):
        assert len(_self_referenced_extras()) >= _SELF_REFERENCES_TODAY

    def test_no_stale_self_reference_is_left_in_pyproject(self):
        stale = re.findall(r"fsm-llm\[", _read("pyproject.toml"))
        assert stale == []

    def test_the_extraction_refuses_a_self_reference_under_the_old_name(
        self, monkeypatch
    ):
        data = _pyproject()
        data["project"]["optional-dependencies"]["all"][0] = "fsm-llm[workflows]"
        monkeypatch.setattr(sys.modules[__name__], "_pyproject", lambda: data)
        with pytest.raises(AssertionError, match="does not name the project"):
            _self_referenced_extras()

    def test_the_extraction_cannot_match_nothing(self, monkeypatch):
        data = _pyproject()
        data["project"]["optional-dependencies"]["all"] = []
        monkeypatch.setattr(sys.modules[__name__], "_pyproject", lambda: data)
        with pytest.raises(AssertionError):
            _self_referenced_extras()


class TestWheelComparison:
    """The comparison itself, on synthetic wheel listings (no build)."""

    _EXPECTED = frozenset(
        {"fsm_llm/__init__.py", "fsm_llm/py.typed", "fsm_llm/monitor/static/a.js"}
    )

    def _problems(self, names: set[str]) -> list[str]:
        return _wheel_file_problems(
            names | {"fsm_llm-1.dist-info/METADATA"}, self._EXPECTED
        )

    def test_the_exact_file_set_has_no_problem(self):
        assert self._problems(set(self._EXPECTED)) == []

    def test_a_stray_module_is_reported(self):
        stray = self._EXPECTED | {"fsm_llm/zz_stray.py"}
        assert self._problems(stray) == [
            "fsm_llm/zz_stray.py: shipped but not expected"
        ]

    def test_a_dropped_data_file_is_reported(self):
        short = self._EXPECTED - {"fsm_llm/monitor/static/a.js"}
        assert self._problems(short) == [
            "fsm_llm/monitor/static/a.js: expected but not shipped"
        ]

    @pytest.mark.parametrize("top", _NEVER_SHIPPED)
    def test_repo_content_is_reported(self, top):
        found = self._problems(self._EXPECTED | {f"{top}/x.py"})
        assert found == [f"{top}/x.py: repo content that must never ship"]

    def test_a_file_outside_the_package_is_reported(self):
        found = self._problems(self._EXPECTED | {"other_pkg/x.py"})
        assert found == ["other_pkg/x.py: outside the top-level fsm_llm/ directory"]


class TestPackageInitImportsNoSubpackage:
    """`import fsm_llm` loads only the core (AGENTS.md): the package `__init__`
    has no import statement that names a subpackage. import-linter cannot say
    this (a contract's source module includes every descendant), so it is here."""

    def test_the_package_init_has_no_subpackage_import(self):
        root = package_dir(_REPO_ROOT)
        # The extras-backed subpackages only: a layer package ships with the core, so the
        # package `__init__` may import it.
        subpackages = SUBPACKAGES
        assert subpackages >= _EXPECTED_SUBPACKAGES  # non-vacuous
        tree = ast.parse((root / "__init__.py").read_text(encoding="utf-8"))
        named: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level == 1 and node.module:
                    named.append(node.module)
                elif node.level == 0 and (node.module or "").startswith("fsm_llm."):
                    named.append(node.module.removeprefix("fsm_llm."))
                elif node.level == 1 and not node.module:
                    named += [a.name for a in node.names]
            elif isinstance(node, ast.Import):
                named += [
                    a.name.removeprefix("fsm_llm.")
                    for a in node.names
                    if a.name.startswith("fsm_llm.")
                ]
        hits = sorted(n for n in named if n.split(".")[0] in subpackages)
        assert not hits, f"fsm_llm/__init__.py imports subpackages: {hits}"


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory):
    """`(wheel path, entry names, expected entries)` of one wheel built from the tracked tree."""
    base = tmp_path_factory.mktemp("wheel")
    staged = base / "tree"
    staged.mkdir()
    _stage_tracked_tree(staged)
    expected = _expected_wheel_files(staged)
    wheel = _build_wheel(staged, base / "out")
    with zipfile.ZipFile(wheel) as archive:
        return wheel, set(archive.namelist()), expected


@pytest.mark.slow
class TestWheelContent:
    """A real wheel, built once from the tracked tree (seconds, but a build)."""

    def test_the_wheel_holds_the_package_files_and_nothing_else(self, built_wheel):
        _wheel, names, expected = built_wheel
        # Non-vacuous: modules, the typing marker and nested frontend files.
        assert "fsm_llm/__init__.py" in expected and "fsm_llm/py.typed" in expected
        assert any(n.startswith("fsm_llm/monitor/static/pages/") for n in expected)
        assert any(n.startswith("fsm_llm/monitor/templates/") for n in expected)
        assert _wheel_file_problems(names, expected) == []

    def test_the_wheel_metadata_names_the_project_and_its_scripts(self, built_wheel):
        wheel, _names, _expected = built_wheel
        assert _dist_info_problems(wheel) == []

    def test_the_build_warns_of_no_package_missing_from_the_configuration(
        self, built_wheel
    ):
        """setuptools (84) warned "Package 'fsm_llm.monitor.static' is absent from the
        `packages` configuration": `include_package_data` (the default of a pyproject
        project) read the static directories as importable packages. The data files are
        listed by `[tool.setuptools.package-data]`, so the default is off. A newer
        setuptools may turn that warning into an error and break the release build."""
        assert _WHEEL_BUILD_OUTPUT, "the wheel fixture ran no build"
        log = _WHEEL_BUILD_OUTPUT[-1]
        assert "is absent from the `packages` configuration" not in log, log[-3000:]

    def test_the_wheel_file_name_is_the_one_of_the_llm_fsm_distribution(
        self, built_wheel
    ):
        wheel, _names, _expected = built_wheel
        version = _pyproject()["project"]["version"]
        assert _pyproject()["project"]["name"] == _DISTRIBUTION
        assert wheel.name == f"llm_fsm-{version}-py3-none-any.whl"

    def test_the_wheel_metadata_requires_the_extras_of_its_own_name(self, built_wheel):
        wheel, _names, _expected = built_wheel
        with zipfile.ZipFile(wheel) as archive:
            (meta,) = [
                n for n in archive.namelist() if n.endswith(".dist-info/METADATA")
            ]
            text = archive.read(meta).decode("utf-8")
        assert "Name: llm-fsm\n" in text
        requires = [
            Requirement(ln[len("Requires-Dist: ") :])
            for ln in text.splitlines()
            if ln.startswith("Requires-Dist: ")
        ]
        selfref = [r for r in requires if _normalized(r.name) == "llm-fsm"]
        assert len(selfref) >= _SELF_REFERENCES_TODAY, selfref
        assert all(r.marker is not None and "all" in str(r.marker) for r in selfref)
        assert [r for r in requires if _normalized(r.name) == "fsm-llm"] == []


@pytest.fixture(scope="module")
def built_sdist(tmp_path_factory):
    """The sdist built from the staged tracked tree (a second, seconds-long build)."""
    base = tmp_path_factory.mktemp("sdist")
    staged = base / "tree"
    staged.mkdir()
    _stage_tracked_tree(staged)
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--sdist",
            "--no-isolation",
            "--outdir",
            str(base / "out"),
            str(staged),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=600,
        check=False,
    )
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    sdists = sorted((base / "out").glob("*.tar.gz"))
    assert len(sdists) == 1, sdists
    return sdists[0]


@pytest.mark.slow
class TestSdistName:
    def test_the_sdist_file_name_is_the_one_of_the_llm_fsm_distribution(
        self, built_sdist
    ):
        version = _pyproject()["project"]["version"]
        assert built_sdist.name == f"llm_fsm-{version}.tar.gz"


# ══════════════════════════════════════════════════════════════
# Explicit text encodings (D-011, D-018): Windows and the C locale do not
# default to UTF-8, so every text file call names its encoding.
# ══════════════════════════════════════════════════════════════

#: Receivers of an `.open(...)` that is not a file open (os.open takes flags).
_NOT_A_FILE_OPEN_RECEIVERS = {"os", "webbrowser"}
#: Calls whose mode defaults to binary: only an explicit text mode is checked.
_BINARY_DEFAULT_CALLS = {"NamedTemporaryFile", "TemporaryFile", "SpooledTemporaryFile"}


def _text_mode(call: ast.Call, positional_index: int, default: str) -> bool:
    """True when the call opens in text mode (or the mode is not a literal)."""
    mode_node: ast.expr | None = None
    if len(call.args) > positional_index:
        mode_node = call.args[positional_index]
    for kw in call.keywords:
        if kw.arg == "mode":
            mode_node = kw.value
    if mode_node is None:
        return "b" not in default
    if isinstance(mode_node, ast.Constant) and isinstance(mode_node.value, str):
        return "b" not in mode_node.value
    return True  # a computed mode cannot be proved binary


def _unencoded_text_calls(tree: ast.AST) -> list[int]:
    """Line numbers of text-mode file calls with no `encoding` argument."""
    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (
            func.attr
            if isinstance(func, ast.Attribute)
            else func.id
            if isinstance(func, ast.Name)
            else ""
        )
        has_encoding = any(kw.arg in ("encoding", None) for kw in node.keywords)
        if name in ("read_text", "write_text"):
            positional_encoding = 1 if name == "read_text" else 2
            if not has_encoding and len(node.args) < positional_encoding:
                lines.append(node.lineno)
        elif name == "open" and isinstance(func, ast.Name):
            if _text_mode(node, 1, "r") and not has_encoding:
                lines.append(node.lineno)
        elif name == "open" and isinstance(func, ast.Attribute):
            receiver = func.value
            if isinstance(receiver, ast.Name) and (
                receiver.id in _NOT_A_FILE_OPEN_RECEIVERS
            ):
                continue
            if _text_mode(node, 0, "r") and not has_encoding:
                lines.append(node.lineno)
        elif name == "fdopen":
            if _text_mode(node, 1, "r") and not has_encoding:
                lines.append(node.lineno)
        elif name in _BINARY_DEFAULT_CALLS:
            if _text_mode(node, 0, "w+b") and not has_encoding:
                lines.append(node.lineno)
    return sorted(lines)


class TestExplicitEncodings:
    """No text-mode file call in `src/`, `scripts/` or `tests/` relies on the locale."""

    def test_no_text_file_call_without_an_encoding(self):
        offenders: list[str] = []
        roots = [_REPO_ROOT / "src", _REPO_ROOT / "scripts", _REPO_ROOT / "tests"]
        for root in roots:
            for path in sorted(root.rglob("*.py")):
                rel = path.relative_to(_REPO_ROOT)
                # scripts/bench_data is frozen, tracked history: never edited.
                if rel.parts[:2] == ("scripts", "bench_data"):
                    continue
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(rel))
                offenders += [f"{rel}:{n}" for n in _unencoded_text_calls(tree)]
        assert not offenders, "text file call without encoding=: " + ", ".join(
            offenders
        )

    def test_the_scanner_flags_and_accepts_the_known_shapes(self):
        bad = ast.parse(
            "p.read_text()\n"
            "p.write_text(s)\n"
            "open(p)\n"
            "open(p, 'w')\n"
            "p.open()\n"
            "os.fdopen(fd, 'w')\n"
            "tempfile.NamedTemporaryFile(mode='w')\n"
            "open(p, mode)\n"
        )
        assert _unencoded_text_calls(bad) == [1, 2, 3, 4, 5, 6, 7, 8]
        good = ast.parse(
            "p.read_text(encoding='utf-8')\n"
            "p.write_text(s, encoding='utf-8')\n"
            "p.read_text('utf-8')\n"
            "open(p, 'rb')\n"
            "p.open('wb')\n"
            "os.fdopen(fd, 'wb')\n"
            "os.open(p, flags)\n"
            "webbrowser.open(url)\n"
            "tempfile.NamedTemporaryFile()\n"
            "open(p, 'w', encoding='utf-8')\n"
        )
        assert _unencoded_text_calls(good) == []

    def test_a_utf8_session_file_loads_under_the_c_locale(self, tmp_path):
        """src/fsm_llm/engine/session.py `load` read the file with the locale encoding."""
        session = {
            "conversation_id": "c1",
            "fsm_id": "f1",
            "current_state": "start",
            "context_data": {"city": "Zürich", "note": "日本"},
        }
        (tmp_path / "s1.json").write_bytes(
            json.dumps(session, ensure_ascii=False).encode("utf-8")
        )
        script = (
            "import locale, sys\n"
            "from fsm_llm.engine.session import FileSessionStore\n"
            "assert sys.flags.utf8_mode == 0, 'UTF-8 mode is on'\n"
            "print(locale.getpreferredencoding(False))\n"
            "state = FileSessionStore(sys.argv[1]).load('s1')\n"
            "print(ascii(None if state is None else state.context_data))\n"
        )
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("PYTHONUTF8", "PYTHONIOENCODING", "LANG", "LC_CTYPE")
        }
        env.update(LC_ALL="C", PYTHONCOERCECLOCALE="0")
        proc = subprocess.run(
            [sys.executable, "-X", "utf8=0", "-c", script, str(tmp_path)],
            capture_output=True,
            text=True,
            encoding="ascii",
            env=env,
            cwd=tmp_path,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        encoding, loaded = proc.stdout.splitlines()
        assert encoding.lower() in ("ansi_x3.4-1968", "ascii", "us-ascii"), encoding
        assert loaded == ascii(session["context_data"])


# ══════════════════════════════════════════════════════════════
# Portability: no import-time POSIX call, reasoned POSIX-only marks, LF checkout
# ══════════════════════════════════════════════════════════════

#: Calls that do not exist on Windows (or fork the interpreter): never at import
#: or decorator time, because that breaks collection of the whole module there.
_POSIX_CALLS = frozenset({"geteuid", "getuid", "getgid", "fork", "forkpty"})


def _call_name(call: ast.Call) -> str:
    func = call.func
    return func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")


def _import_time_calls(tree: ast.AST, names: frozenset[str]) -> list[tuple[int, str]]:
    """``(line, name)`` of every call of ``names`` that runs at import time.

    Module level, class bodies, decorators and default values run at import; a
    function or lambda BODY does not.
    """
    found: list[tuple[int, str]] = []

    class _Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            if _call_name(node) in names:
                found.append((node.lineno, _call_name(node)))
            self.generic_visit(node)

        def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            for expr in node.decorator_list:
                self.visit(expr)
            self.visit(node.args)
            if node.returns is not None:
                self.visit(node.returns)  # the body is skipped on purpose

        visit_FunctionDef = visit_AsyncFunctionDef = _function

        def visit_Lambda(self, node: ast.Lambda) -> None:
            self.visit(node.args)  # a lambda body runs later

    _Visitor().visit(tree)
    return found


def _posix_only_without_reason(tree: ast.AST) -> list[int]:
    """Lines of ``posix_only(...)`` calls with no non-blank string-literal reason."""
    bad: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node) == "posix_only":
            first = node.args[0] if node.args else None
            if not (
                isinstance(first, ast.Constant)
                and isinstance(first.value, str)
                and first.value.strip()
            ):
                bad.append(node.lineno)
    return bad


def _bare_symlink_calls(tree: ast.AST) -> list[int]:
    """Lines of ``x.symlink_to(...)`` and ``os.symlink(...)`` calls."""
    return [
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and _call_name(n) in {"symlink_to", "symlink"}
    ]


def _test_trees() -> list[tuple[pathlib.Path, ast.AST]]:
    out = []
    for path in sorted((_REPO_ROOT / "tests").rglob("*.py")):
        rel = path.relative_to(_REPO_ROOT)
        out.append(
            (rel, ast.parse(path.read_text(encoding="utf-8"), filename=str(rel)))
        )
    return out


def _gitattributes() -> dict[str, set[str]]:
    rules: dict[str, set[str]] = {}
    text = (_REPO_ROOT / ".gitattributes").read_text(encoding="utf-8")
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            pattern, *attrs = line.split()
            rules.setdefault(pattern, set()).update(attrs)
    return rules


class TestPortabilityRules:
    """Windows and macOS legs: the test tree never trips on a POSIX-only call."""

    def test_no_test_calls_a_posix_only_function_at_import_time(self):
        offenders = [
            f"{rel}:{line} {name}()"
            for rel, tree in _test_trees()
            for line, name in _import_time_calls(tree, _POSIX_CALLS)
        ]
        assert not offenders, (
            "POSIX-only call at import or decorator time (move it into the test "
            "body, `tests.conftest.skip_if_root` does the root check): "
            + ", ".join(offenders)
        )

    def test_every_posix_only_mark_carries_a_reason(self):
        offenders = [
            f"{rel}:{line}"
            for rel, tree in _test_trees()
            for line in _posix_only_without_reason(tree)
        ]
        assert not offenders, "posix_only() without a reason: " + ", ".join(offenders)

    def test_symlinks_go_through_the_shared_helper(self):
        offenders = [
            f"{rel}:{line}"
            for rel, tree in _test_trees()
            if rel.as_posix() != "tests/conftest.py"
            for line in _bare_symlink_calls(tree)
        ]
        assert not offenders, (
            "bare symlink_to or os.symlink (use tests.conftest.symlink_or_skip, "
            "which skips where symlinks are refused): " + ", ".join(offenders)
        )

    def test_the_scanners_flag_and_accept_the_known_shapes(self):
        bad = ast.parse(
            "import os, pytest\n"
            "X = os.geteuid()\n"
            "@pytest.mark.skipif(os.getuid() == 0, reason='r')\n"
            "def f(a=os.fork()):\n"
            "    os.geteuid()\n"
            "class C:\n"
            "    y = os.getgid()\n"
        )
        hits = _import_time_calls(bad, _POSIX_CALLS)
        assert [name for _, name in hits] == ["geteuid", "getuid", "fork", "getgid"]
        ok = ast.parse(
            "def f():\n"
            "    os.geteuid()\n"
            "    return lambda: os.fork()\n"
            "g = lambda: os.getuid()\n"
        )
        assert _import_time_calls(ok, _POSIX_CALLS) == []
        assert _posix_only_without_reason(
            ast.parse("posix_only()\nposix_only('')\nposix_only(r)\nposix_only('why')")
        ) == [1, 2, 3]
        assert _bare_symlink_calls(
            ast.parse("a.symlink_to(b)\nos.symlink(a, b)\nsymlink_or_skip(a, b)")
        ) == [1, 2]

    def test_the_posix_only_mark_skips_on_windows_only(self, monkeypatch):
        from tests.conftest import posix_only

        mark = posix_only("needs a mode bit").mark
        assert mark.name == "skipif"
        assert mark.kwargs["reason"] == "needs a mode bit"
        assert mark.args == (sys.platform == "win32",)

    def test_a_dangling_link_is_skipped_not_failed_when_links_are_refused(
        self, tmp_path, monkeypatch
    ):
        from tests.conftest import symlink_or_skip

        def refuse(self, target, target_is_directory=False):
            raise OSError("no privilege")

        monkeypatch.setattr(pathlib.Path, "symlink_to", refuse)
        with pytest.raises(pytest.skip.Exception):
            symlink_or_skip(tmp_path / "l", tmp_path / "t")


class TestGitattributesPinLineEndings:
    """`actions/checkout` on Windows must not turn LF into CRLF (D-018)."""

    _LF_PATTERNS = (
        "*.py",
        "*.json",
        "*.md",
        "*.txt",
        "*.toml",
        "*.yml",
        "*.yaml",
        "*.sh",
    )

    def test_text_types_are_checked_out_with_lf(self):
        rules = _gitattributes()
        assert {"text=auto", "eol=lf"} <= rules.get("*", set())
        for pattern in self._LF_PATTERNS:
            assert {"text", "eol=lf"} <= rules.get(pattern, set()), pattern

    def test_images_and_archives_are_binary(self):
        rules = _gitattributes()
        for pattern in ("*.png", "*.jpg", "*.gif", "*.zip", "*.whl"):
            assert "binary" in rules.get(pattern, set()), pattern

    @pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
    def test_git_resolves_the_attributes_for_real_paths(self):
        def attrs(path: str) -> dict[str, str]:
            out = subprocess.run(
                ["git", "check-attr", "text", "eol", "--", path],
                cwd=_REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            ).stdout
            return {
                line.split(": ")[1]: line.split(": ")[2]
                for line in out.splitlines()
                if line.count(": ") == 2
            }

        for path in ("docs/api.txt", "pyproject.toml", "tests/conftest.py"):
            assert attrs(path).get("eol") == "lf", path
        assert attrs("images/fsm-llm-logo-1.png").get("text") == "unset"


# ══════════════════════════════════════════════════════════════
# Documented test counts: every hand-maintained literal == measured
# ══════════════════════════════════════════════════════════════

#: A count literal as the docs spell it: "5,107" or "1751". Comma-tolerant.
_COUNT = r"(\d[\d,]*)"


#: The `make test` line of CONTRIBUTING.md (the README keeps no count: it is the PyPI page).
_MAKE_TEST_PIN = rf"Run full test suite \({_COUNT} tests\)"


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
# NOT pinned (frozen release history), and AGENTS.md's "N passed / N skipped /
# N xfailed" line is a RUN result at a named commit, not a collection count --
# both exclusions are scope, not oversights. See decisions.md D-015.
@pytest.fixture(scope="module")
def measured() -> tuple[int, dict[str, int], dict[str, int]]:
    """(total, per-suite-dir counts, per-root-file counts), measured once.

    The child env drops every variable that changes what gets collected or
    probed at collection time: PYTEST_ADDOPTS (an outer `-m "not slow"`
    would deselect in the child and skew the count) and the
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


class TestTheMakeTestPinLivesInContributing:
    """Step 105 of plan d454fd62: the README is the PyPI page and holds no count line.

    The `make test` line ("Run full test suite (N tests)") that the slow class below pins
    moved to CONTRIBUTING.md. These checks are cheap (no collection), so the retarget is
    guarded in the default run; the count VALUE is the slow class's business.
    """

    def test_contributing_holds_the_line_once(self):
        assert len(_pinned_counts("CONTRIBUTING.md", _MAKE_TEST_PIN)) == 1

    def test_the_readme_holds_no_such_line(self):
        assert not re.search(_MAKE_TEST_PIN, _read("README.md"))

    @pytest.mark.parametrize("rel", ["AGENTS.md", "tests/AGENTS.md", "tests/README.md"])
    def test_the_prose_names_contributing_not_the_readme(self, rel):
        text = _read(rel)
        mentions = [m.start() for m in re.finditer(r"Run full test suite", text)]
        assert mentions, f"{rel} no longer names the `Run full test suite` line"
        for at in mentions:
            window = text[max(0, at - 80) : at + 40]
            assert "CONTRIBUTING.md" in window, f"{rel}: {window!r}"
            assert "README" not in window, f"{rel}: {window!r}"


@pytest.mark.slow
class TestDocumentedTestCountsMatchCollection:
    """Every hand-maintained test-count literal equals `pytest --collect-only`.

    Same defect class as the package-wiring tests above: hand-maintained
    parallel copies of a filesystem-derivable fact. AGENTS.md and README.md
    each spell test counts as prose literals
    ("5,107 tests", a 9-suite breakdown table); every new test
    silently strands them, and nothing errors. In the predecessor plans the
    literals drifted for days at a time (3,305/2,382 stated vs 5,107 measured)
    before a manual `--collect-only` audit caught it. This class measures once
    per run (one subprocess, marked slow) and pins every literal to it.

    Excluded on purpose: CHANGELOG.md (frozen history -- its counts describe
    the state AT each release, and must never be retro-edited) and AGENTS.md's
    "N passed / N skipped / N xfailed" sentence (a run result at a pinned
    commit, not a collection count; re-verified by release-gate runs instead).
    """

    @pytest.mark.parametrize(
        "rel,pattern",
        [
            ("AGENTS.md", rf"pytest -v \({_COUNT} tests\)"),
            ("AGENTS.md", rf"Run all tests \({_COUNT} collected\)"),
            ("CONTRIBUTING.md", _MAKE_TEST_PIN),
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
        # AGENTS.md's Testing block: `pytest tests/<suite>/  # ... (N tests)`.
        # Dict equality both ways: a NEW suite directory must be added to the
        # table, and a deleted one must leave it.
        _, per_suite, _ = measured
        documented = {
            suite: _num(count)
            for suite, count in re.findall(
                rf"^pytest tests/(test_[a-z_]+)/\s+#[^(\n]*\({_COUNT} tests\)",
                _read("AGENTS.md"),
                re.MULTILINE,
            )
        }
        assert documented, "per-suite table regex matched nothing in AGENTS.md"
        assert documented == per_suite, (
            f"AGENTS.md per-suite table out of sync: doc-only "
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
            _read("AGENTS.md"),
        )
        assert line is not None, "sum/remainder sentence not found in AGENTS.md"
        assert int(line.group(1)) == len(per_suite)
        assert _num(line.group(2)) == sum(per_suite.values())
        assert _num(line.group(3)) == total - sum(per_suite.values())

    def test_root_file_breakdown(self, measured):
        # Every root-level test file must be listed as `tests/<name> (N)` with
        # the measured N -- and nothing extra may be listed. Derived from the
        # collection, so a brand-new root-level test file fails this until
        # AGENTS.md names it.
        _, _, per_root_file = measured
        documented = {
            name: _num(count)
            for name, count in re.findall(
                rf"tests/(test_[a-z_]+\.py) \({_COUNT}\)", _read("AGENTS.md")
            )
        }
        assert documented == per_root_file, (
            f"AGENTS.md root-file breakdown {documented} != measured {per_root_file}"
        )


@pytest.fixture(scope="module")
def doc_measured() -> dict:
    """One more child collection (about 10 s) for `scripts/doc_counts.py`."""
    if doc_counts.MCP_INSTALLED:
        pytest.skip(doc_counts.MCP_SKIP)
    return doc_counts.measure()


@pytest.mark.slow
class TestEveryDocumentedCountMatchesCollection:
    """Every count token of the shapes `scripts/doc_counts.py` names equals the collection.

    The class above pins the count literals of two files (AGENTS.md and README.md). This one runs the sweep of
    `scripts/doc_counts.py` (`make counts`) over the tracked .md files: `N tests`
    style tokens, marker, `slow`, live and file counts, the per-file rows. A
    token of one of those shapes that no rule names fails too. A number in prose
    without such a shape (N-007) is not read. Dated records (CHANGELOG.md,
    plans/) are skipped by that script.
    """

    def test_no_documented_count_drifts(self, doc_measured):
        problems = doc_counts.scan(doc_measured)
        assert problems == [], "\n".join(problems)


@pytest.mark.parametrize(
    ("line", "bad"),
    [
        ("| `test_a.py` | 21 (8 `slow`) |", 0),
        ("| `test_a.py` | 21 (9 `slow`) |", 1),
        ("| `test_a.py` | 22 (8 `slow`) |", 1),
        ("- `test_a.py` - 21 offline tests", 0),
        ("- `test_a.py` - 22 offline tests", 1),
        ("Of them, 3 live Ollama tests and 3 gated by `requires_live`.", 0),
        ("Of them, 4 live Ollama tests.", 1),
        ('`-m "not slow"` skips the 8 `slow` and the 8 marked `slow` tests.', 0),
        ('`-m "not slow"` skips the 9 marked `slow` tests.', 1),
        ("The suite holds 269 tests in 7 files.", 0),
        ("The suite holds 269 tests in 8 files.", 1),
    ],
)
def test_doc_counts_reads_the_slow_live_and_file_shapes(line, bad):
    """Guards: a slow, live or file count in a doc that no rule of doc_counts read."""
    m = {k: collections.Counter() for k in ("files", "slow_files", "live_files")}
    m |= {k: collections.Counter() for k in ("classes", "markers", "test_files")}
    m |= {k: collections.Counter() for k in ("suites", "slow_suites", "live_suites")}
    path = "tests/test_s/test_a.py"
    m["by_base"] = {"test_a.py": [path]}
    m["files"][path], m["slow_files"][path], m["live_files"][path] = 21, 8, 3
    m["suites"]["test_s"], m["slow_suites"]["test_s"], m["live_suites"]["test_s"] = (
        269,
        8,
        3,
    )
    m["test_files"]["test_s"] = 7
    doc = doc_counts._Doc(m, "tests/test_s/AGENTS.md")
    doc.line(1, line)
    assert len(doc.bad) == bad, doc.bad


@pytest.mark.parametrize(
    ("line", "bad"),
    [
        ("- **Offline test depth.** 13,000 tests run without a model.", 0),
        ("- **Offline test depth.** 12,999 tests run without a model.", 1),
    ],
)
def test_doc_counts_reads_the_offline_test_depth_total(line, bad):
    """Guards: the roadmap's whole-suite claim is a total-collection token, not unnamed."""
    m = {k: collections.Counter() for k in ("files", "classes", "markers", "suites")}
    m["total"] = 13000
    doc = doc_counts._Doc(m, "docs/2026_roadmap.md")
    doc.line(1, line)
    assert len(doc.bad) == bad, doc.bad


def _rows_measured() -> dict:
    """A hand-built `measure()` result: suite `test_s` holds two test files."""
    m = {k: collections.Counter() for k in ("files", "classes", "markers")}
    m["files"] |= {
        "tests/test_s/test_a.py": 3,
        "tests/test_s/test_b.py": 4,
        "tests/test_s/conftest.py": 0,
        "tests/test_root.py": 2,
    }
    m["by_base"] = {
        "test_a.py": ["tests/test_s/test_a.py"],
        "test_b.py": ["tests/test_s/test_b.py"],
        "conftest.py": ["tests/test_s/conftest.py"],
        "test_root.py": ["tests/test_root.py"],
    }
    return m


@pytest.mark.parametrize(
    ("md", "lines", "want"),
    [
        # a complete README table is clean (conftest needs no row)
        ("tests/test_s/README.md", ["- `test_a.py` - x", "| `test_b.py` | y |"], []),
        # a README without the row of one test file: MISSING-ROW
        ("tests/test_s/README.md", ["- `test_a.py` - x"], ["MISSING-ROW"]),
        # one row may group files: "`a.py`, `b.py` - ..." and "`a.py` + `b.py`"
        ("tests/test_s/README.md", ["- `test_a.py`, `test_b.py` - x"], []),
        ("tests/test_s/README.md", ["| `test_a.py` + `test_b.py` | x |"], []),
        # a row for a file that no longer exists: DEAD-ROW (also in AGENTS.md)
        (
            "tests/test_s/README.md",
            ["- `test_a.py` - x", "- `test_b.py` - y", "- `test_gone.py` - z"],
            ["DEAD-ROW"],
        ),
        ("tests/test_s/AGENTS.md", ["| `test_gone.py` | z |"], ["DEAD-ROW"]),
        # AGENTS.md may group files by feature: no completeness rule for it
        ("tests/test_s/AGENTS.md", ["| `test_a.py`, `test_b.py` | both |"], []),
        # the root README table counts the root files only, not the suites'
        ("tests/README.md", ["- `test_root.py` - 2 tests"], []),
        ("tests/README.md", ["- `conftest.py` - fixtures"], ["MISSING-ROW"]),
        # a doc outside tests/ is not read; a nested fixtures README is read for rows
        ("docs/guide.md", ["- `test_gone.py` - z"], []),
        ("tests/test_s/fixtures/README.md", ["- `test_gone.py` - z"], ["DEAD-ROW"]),
        # prose that names a file mid-line is not a row
        ("tests/test_s/AGENTS.md", ["See `test_gone.py` for more."], []),
    ],
)
def test_doc_counts_checks_the_per_file_rows(md, lines, want):
    """Guards: a dead row or a missing row in a tests/ doc goes unseen by the counts."""
    got = doc_counts.file_rows(_rows_measured(), md, lines)
    assert [g.split()[0] for g in got] == want, got


# ══════════════════════════════════════════════════════════════
# scripts/audit_pth.py has no allowlist: the editable install is a plain path
# ══════════════════════════════════════════════════════════════


class TestAuditPthNeedsNoAllowlist:
    """`pip install -e .` of the src layout writes a `.pth` with one directory
    path, which the audit passes on its own merits; the code-bearing finder hook
    of the old flat layout is code like any other and is flagged."""

    _FINDER_HOOK = (
        "import __editable___fsm_llm_0_12_0_finder; "
        "__editable___fsm_llm_0_12_0_finder.install()\n"
    )

    def _audit(self, tmp_path, monkeypatch, name: str, text: str) -> int:
        (tmp_path / name).write_text(text, encoding="utf-8")
        monkeypatch.setattr(audit_pth.site, "getsitepackages", lambda: [str(tmp_path)])
        return audit_pth.audit()

    @staticmethod
    def _pth_name() -> str:
        """The `.pth` file an editable install of this project writes (`pip install -e .`).

        Derived from `[project]`, never from the installed metadata, so the test holds
        on a machine that still has the old `fsm_llm-<version>` editable install and on one
        that has the renamed `llm_fsm-<version>` one.
        """
        version = _pyproject()["project"]["version"]
        return f"__editable__.{_dist_stem()}-{version}.pth"

    def test_the_pth_name_is_the_one_of_the_renamed_distribution(self):
        assert self._pth_name() == (
            f"__editable__.llm_fsm-{_pyproject()['project']['version']}.pth"
        )

    def test_the_audit_carries_no_editable_finder_allowlist(self):
        assert not hasattr(audit_pth, "EDITABLE_FINDER")
        assert not hasattr(audit_pth, "is_editable_finder")

    def test_the_plain_path_pth_of_an_editable_install_passes(
        self, tmp_path, monkeypatch
    ):
        text = str(_REPO_ROOT / "src") + "\n"
        assert self._audit(tmp_path, monkeypatch, self._pth_name(), text) == 0

    def test_the_old_finder_hook_is_flagged(self, tmp_path, monkeypatch, capsys):
        name = self._pth_name()
        assert self._audit(tmp_path, monkeypatch, name, self._FINDER_HOOK) == 1
        assert name in capsys.readouterr().err


class TestAuditPthReadsWhatSiteReads:
    """A `.pth` is decoded by `site` (UTF-8 with an optional BOM, then the
    locale encoding on 3.13+), and Windows tools write both a BOM and ANSI
    bytes. The audit must neither crash on a file it cannot decode nor be
    blinded by a BOM in front of an `import` line."""

    _CODE = b"import os; os.system('echo owned')\n"

    def _audit(self, tmp_path, monkeypatch, name: str, data: bytes):
        (tmp_path / name).write_bytes(data)
        monkeypatch.setattr(audit_pth.site, "getsitepackages", lambda: [str(tmp_path)])
        return audit_pth.audit()

    def test_a_pth_that_is_not_utf8_does_not_crash_the_audit(
        self, tmp_path, monkeypatch
    ):
        path_only = "C:\\Users\\J\u00fcrgen\\src\n".encode("cp1252")
        assert self._audit(tmp_path, monkeypatch, "legacy.pth", path_only) == 0

    def test_code_in_a_pth_that_is_not_utf8_is_still_flagged(
        self, tmp_path, monkeypatch, capsys
    ):
        data = b"# J\xfcrgen\n" + self._CODE
        assert self._audit(tmp_path, monkeypatch, "evil.pth", data) == 1
        assert "evil.pth" in capsys.readouterr().err

    def test_a_bom_does_not_hide_code_from_the_audit(
        self, tmp_path, monkeypatch, capsys
    ):
        data = b"\xef\xbb\xbf" + self._CODE
        assert self._audit(tmp_path, monkeypatch, "bom.pth", data) == 1
        assert "bom.pth" in capsys.readouterr().err
