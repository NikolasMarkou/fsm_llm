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

import pathlib
import re

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
