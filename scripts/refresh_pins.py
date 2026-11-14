#!/usr/bin/env python
"""Propose newer pins for litellm and mcp: the newest release that has aged and fits the matrix.

Stdlib only, Python 3.11+. This is the script behind `.github/workflows/refresh-pins.yml`
(roadmap row 0.10, plan d454fd62 step 110); the workflow runs it weekly, then the whole suite,
and only then opens a pull request. Run by hand:

    refresh_pins.py [--packages litellm mcp] [--constraints constraints.txt]
                    [--pyproject pyproject.toml] [--workflow .github/workflows/python-package.yml]
                    [--python 3.11 3.12 ...] [--now 2026-11-14T00:00:00Z]
                    [--json-dir DIR] [--dry-run]

For each package it reads the PyPI JSON (`https://pypi.org/pypi/<package>/json`, or
`DIR/<package>.json` with `--json-dir`, which touches no network) and takes the HIGHEST release
that passes every rule below, then rewrites the one `<package>==<version>` line of
`constraints.txt`. All other bytes of the file stay as they were.

A release is eligible only when ALL of these hold (each one fails closed):

* stable: plain `N.N` to `N.N.N.N` digits. Pre-releases, dev releases, post-releases and epochs
  are skipped.
* not yanked: a release is skipped when ANY of its files is yanked (a release with no files is
  skipped too).
* cooldown: its EARLIEST file upload is at least 7 days before `--now` (`COOLDOWN`). The cooldown
  exists because litellm 1.82.7 and 1.82.8 were malware for hours in March 2026: a bad release
  is usually yanked within days, so a pin never lands on a release that young. Exactly 7 days
  old passes.
* Requires-Python: its `requires_python` admits EVERY supported Python (see `Python versions`
  below). A release that does not declare `requires_python`, or declares a form this script does
  not understand, is skipped. The evaluator knows `>= <= > < == != ~=` (with `.*` wildcards for
  `==` and `!=`). A version `3.N` is tested at both `3.N.0` and `3.N.9999`, so a bound that cuts
  inside a minor series (`>=3.11.5`) excludes the release instead of guessing the runner patch.
* same major as the CURRENT pin: this script never crosses a major version; that is a human
  decision (D-025: bump deliberately). `litellm>=1.109.0,<2.0` and `mcp>=2.2,<3` already say so;
  the rule holds even if a range is widened by hand.
* inside the declared range (`[project]` dependencies, extras and dependency groups of
  `pyproject.toml`), OR excluded only by an upper bound (`<` or `<=`) of the same major: then the
  range is rewritten with that bound moved to the next major (`<1.115` becomes `<2.0`, the same
  number of components), so `TestDeclaredRangesMatchPins` keeps holding. With the present
  `<2.0` and `<3` bounds nothing is ever widened.

It never downgrades: when the best eligible release is older than the current pin the pin stays.

Python versions: by default the `python-version` values of the `test` matrix in
`.github/workflows/python-package.yml` (the versions CI actually proves), read with a tiny
reader that takes the lines under `matrix:`; `--python` replaces them. Every one must be admitted
by `requires-python` of `pyproject.toml`, else the script stops (the matrix and the package
metadata disagree). The workflow is the one source that stays true after the classifier list of
`pyproject.toml` grows to 3.15 (plan step 115).

Output (stdout, deterministic for one input and one `--now`): a markdown table (package, old
pin, new pin, release date, age in days, reason), the skipped newer releases, with `--dry-run`
the unified diff, and as the LAST line `changed=true` or `changed=false` for the workflow.

Exit codes: 0 done (changed or not), 1 any network, data or consistency failure (nothing is
written: every file is built in memory, then written through a temp file and `os.replace`),
2 usage error. `--dry-run` never writes.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
import tomllib
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONSTRAINTS = ROOT / "constraints.txt"
DEFAULT_PYPROJECT = ROOT / "pyproject.toml"
DEFAULT_WORKFLOW = ROOT / ".github" / "workflows" / "python-package.yml"
TRACKED = ("litellm", "mcp")
COOLDOWN = timedelta(days=7)
PYPI_URL = "https://pypi.org/pypi/{package}/json"
FETCH_TIMEOUT_SECONDS = 30
_PATCH_HIGH = 9999

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_STABLE = re.compile(r"^\d+(?:\.\d+){1,3}$")
_CLAUSE = re.compile(r"^(~=|==|!=|<=|>=|<|>)\s*(\d+(?:\.\d+)*)(\.\*)?$")
_REQUIREMENT = re.compile(
    r"^\s*(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*"
    r"(?P<spec>[<>=!~][^;]*)?(?:;.*)?$"
)

Fetcher = Callable[[str], Any]


class RefreshError(Exception):
    """A refusal with a message for stderr; `main` turns it into exit 1."""


# --------------------------------------------------------------------------------------
# Versions and specifiers (a minimal PEP 440 subset; anything else fails closed)
# --------------------------------------------------------------------------------------

Version = tuple[int, ...]


def normalize(name: str) -> str:
    """PEP 503 name: lower case, runs of `-_.` become one `-`."""
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_release(text: str) -> Version | None:
    """The tuple of a stable release string (`2.3.0`), or None for anything else."""
    return tuple(int(p) for p in text.split(".")) if _STABLE.match(text) else None


def _padded(left: Version, right: Version) -> tuple[Version, Version]:
    width = max(len(left), len(right))
    return (
        left + (0,) * (width - len(left)),
        right + (0,) * (width - len(right)),
    )


def _compare(left: Version, right: Version) -> int:
    a, b = _padded(left, right)
    return (a > b) - (a < b)


Clause = tuple[str, Version, bool]


def parse_specifier(text: str) -> list[Clause]:
    """Parse `>=3.10,<3.16` into clauses `(operator, version, wildcard)`; raise ValueError.

    Contract: comma separated clauses, spaces allowed; an empty text is an empty list (admits
    everything). An operator outside `~= == != <= >= < >`, a pre-release or a local suffix, a
    wildcard on anything but `==` and `!=`, and `~=` with a single component all raise.
    """
    clauses: list[Clause] = []
    for raw in text.split(","):
        part = raw.strip()
        if not part:
            if text.strip():
                raise ValueError(f"empty clause in {text!r}")
            continue
        match = _CLAUSE.match(part)
        if match is None:
            raise ValueError(f"unsupported clause {part!r}")
        op, number, star = match.group(1), match.group(2), match.group(3)
        version = tuple(int(p) for p in number.split("."))
        if star and op not in ("==", "!="):
            raise ValueError(f"wildcard not allowed with {op!r}: {part!r}")
        if op == "~=" and len(version) < 2:
            raise ValueError(f"~= needs two components: {part!r}")
        clauses.append((op, version, bool(star)))
    return clauses


def _clause_admits(clause: Clause, candidate: Version) -> bool:
    op, version, wildcard = clause
    if wildcard:
        prefix = candidate[: len(version)] + (0,) * max(
            0, len(version) - len(candidate)
        )
        same = prefix == version
        return same if op == "==" else not same
    cmp = _compare(candidate, version)
    if op == "~=":
        upper = (*version[:-2], version[-2] + 1)
        return cmp >= 0 and _compare(candidate, upper) < 0
    return {
        ">=": cmp >= 0,
        "<=": cmp <= 0,
        ">": cmp > 0,
        "<": cmp < 0,
        "==": cmp == 0,
        "!=": cmp != 0,
    }[op]


def admits(clauses: list[Clause], candidate: Version) -> bool:
    """True when every clause admits `candidate` (a parsed release or Python version)."""
    return all(_clause_admits(c, candidate) for c in clauses)


def admits_python(requires_python: str | None, python: str) -> bool:
    """True when `requires_python` admits Python `3.N` at both `3.N.0` and `3.N.9999`.

    A missing, empty or not understood `requires_python` is False (fail closed).
    """
    if not requires_python or not requires_python.strip():
        return False
    try:
        clauses = parse_specifier(requires_python)
    except ValueError:
        return False
    base = parse_release(python)
    if base is None or len(base) != 2:
        return False
    return admits(clauses, (*base, 0)) and admits(clauses, (*base, _PATCH_HIGH))


# --------------------------------------------------------------------------------------
# Reading PyPI JSON
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Release:
    """One release of a package: its tuple, text, earliest upload, and the facts that gate it."""

    text: str
    version: Version
    uploaded: datetime | None
    yanked: bool
    requires_python: str | None


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def read_releases(package: str, document: Any) -> list[Release]:
    """The stable releases of a PyPI JSON document; raise RefreshError on a malformed one."""
    if not isinstance(document, dict) or not isinstance(document.get("releases"), dict):
        raise RefreshError(f"{package}: the JSON has no `releases` object")
    info = document.get("info")
    if isinstance(info, dict) and isinstance(info.get("name"), str):
        if normalize(info["name"]) != normalize(package):
            raise RefreshError(
                f"{package}: the JSON describes {info['name']!r}, not {package!r}"
            )
    found: list[Release] = []
    for text, files in document["releases"].items():
        version = parse_release(str(text))
        if version is None:
            continue  # pre-release, dev, post or epoch: never eligible
        file_list = (
            [f for f in files if isinstance(f, dict)] if isinstance(files, list) else []
        )
        times = [_parse_time(f.get("upload_time_iso_8601")) for f in file_list]
        known = [t for t in times if t is not None]
        # A file with no readable time makes the whole release unreadable (fail closed).
        uploaded = min(known) if known and len(known) == len(times) else None
        pythons = {f.get("requires_python") for f in file_list}
        found.append(
            Release(
                text=str(text),
                version=version,
                uploaded=uploaded,
                yanked=(not file_list) or any(f.get("yanked") for f in file_list),
                requires_python=pythons.pop() if len(pythons) == 1 else None,
            )
        )
    return sorted(found, key=lambda r: r.version)


def fetch_pypi(package: str) -> Any:
    """GET the JSON of one package from PyPI (the only network call of this script)."""
    request = urllib.request.Request(
        PYPI_URL.format(package=package),
        headers={"Accept": "application/json", "User-Agent": "pin-refresh-script"},
    )
    try:
        with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT_SECONDS) as response:
            return json.load(response)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise RefreshError(f"{package}: cannot read PyPI JSON: {exc}") from exc


def dir_fetcher(directory: Path) -> Fetcher:
    """A fetcher that reads `DIR/<package>.json` (canned files; no network)."""

    def fetch(package: str) -> Any:
        path = directory / f"{package}.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RefreshError(f"{package}: cannot read {path}: {exc}") from exc

    return fetch


# --------------------------------------------------------------------------------------
# The files: the matrix, the declared ranges, the pin lines
# --------------------------------------------------------------------------------------


def matrix_pythons(workflow_text: str) -> list[str]:
    """The `python-version` values of the `matrix:` block of a workflow, sorted, unique.

    A tiny reader, not a YAML parser: it takes the lines indented below the first `matrix:` line
    and every `3.N` token on the lines that say `python-version`.
    """
    lines = workflow_text.splitlines()
    for index, line in enumerate(lines):
        if line.strip().startswith("matrix:"):
            indent = len(line) - len(line.lstrip())
            block: list[str] = []
            for follow in lines[index + 1 :]:
                if follow.strip() and len(follow) - len(follow.lstrip()) <= indent:
                    break
                block.append(follow)
            tokens = {
                token
                for row in block
                if "python-version" in row and not row.lstrip().startswith("#")
                for token in re.findall(r"\b3\.\d{1,2}\b", row)
            }
            return sorted(tokens, key=lambda t: tuple(int(p) for p in t.split(".")))
    return []


@dataclass
class Declaration:
    """One requirement string of `pyproject.toml` for a tracked package."""

    text: str
    clauses: list[Clause]
    spec: str


def declarations(pyproject: dict[str, Any], package: str) -> list[Declaration]:
    """Every requirement string of `package` in dependencies, extras and dependency groups."""
    project = pyproject.get("project") or {}
    pools: list[Any] = [project.get("dependencies") or []]
    pools += list((project.get("optional-dependencies") or {}).values())
    pools += list((pyproject.get("dependency-groups") or {}).values())
    found: list[Declaration] = []
    for pool in pools:
        for item in pool:
            match = _REQUIREMENT.match(item) if isinstance(item, str) else None
            if match is None or normalize(match["name"]) != normalize(package):
                continue
            spec = (match["spec"] or "").strip()
            try:
                clauses = parse_specifier(spec)
            except ValueError as exc:
                raise RefreshError(
                    f"{package}: cannot read range {item!r}: {exc}"
                ) from exc
            found.append(Declaration(text=item, clauses=clauses, spec=spec))
    return found


def read_pin(constraints: str, package: str) -> tuple[int, Version, str]:
    """`(line index, version tuple, version text)` of the one `package==x` line."""
    hits: list[tuple[int, Version, str]] = []
    for index, line in enumerate(constraints.splitlines()):
        match = re.match(
            r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*==\s*(\S+?)\s*(?:#.*)?$", line
        )
        if match and normalize(match[1]) == normalize(package):
            version = parse_release(match[2])
            if version is None:
                raise RefreshError(
                    f"{package}: the pin {match[2]!r} is not a stable release"
                )
            hits.append((index, version, match[2]))
    if len(hits) != 1:
        raise RefreshError(
            f"{package}: constraints.txt needs exactly one `{package}==X.Y.Z` line, found {len(hits)}"
        )
    return hits[0]


def rewrite_pin(constraints: str, package: str, new_version: str) -> str:
    """`constraints` with the version of the one pin line of `package` replaced; nothing else."""
    index, _, old = read_pin(constraints, package)
    lines = constraints.splitlines(keepends=True)
    line = lines[index]
    start = line.index("==")
    cut = line.index(old, start)
    lines[index] = line[:cut] + new_version + line[cut + len(old) :]
    return "".join(lines)


def widen_upper_bound(spec: str, major: int) -> str | None:
    """Move the `<`/`<=` bound of `spec` to the next major, or None when it cannot be done.

    Only a spec with exactly one upper-bound clause whose version is in the same major as the
    current pin is widened (`<1.115` to `<2.0`, `<=1.115.2` to `<2.0.0`); the new bound keeps
    the number of components of the old one.
    """
    parts = [p.strip() for p in spec.split(",")]
    uppers = [p for p in parts if p.startswith("<")]
    if len(uppers) != 1:
        return None
    match = re.match(r"^<=?\s*(\d+(?:\.\d+)*)$", uppers[0])
    if match is None:
        return None
    old = tuple(int(p) for p in match[1].split("."))
    if old[0] != major:
        return None
    new = f"<{major + 1}" + ".0" * (len(old) - 1)
    return ",".join(new if p == uppers[0] else p for p in parts)


def range_decision(
    declared: list[Declaration], release: Release, major: int
) -> tuple[bool, list[str | None]]:
    """`(ok, new_specs)`: does every declaration admit the release, or widen to admit it?

    `new_specs[i]` is the rewritten spec of declaration `i` when it had to be widened, else None.
    """
    new_specs: list[str | None] = []
    for item in declared:
        if admits(item.clauses, release.version):
            new_specs.append(None)
            continue
        rest = [c for c in item.clauses if c[0] not in ("<", "<=")]
        widened = widen_upper_bound(item.spec, major)
        if (
            release.version[0] != major
            or widened is None
            or not admits(rest, release.version)
        ):
            return False, []
        new_specs.append(widened)
    return True, new_specs


# --------------------------------------------------------------------------------------
# Choosing
# --------------------------------------------------------------------------------------


@dataclass
class Choice:
    """The outcome for one package."""

    package: str
    old: str
    new: str
    release_date: str
    age_days: str
    reason: str
    changed: bool
    skipped: list[str] = field(default_factory=list)
    new_specs: list[str | None] = field(default_factory=list)


def _age_days(release: Release, now: datetime) -> int:
    assert release.uploaded is not None
    return max(0, (now - release.uploaded).days)


def why_not(
    release: Release,
    now: datetime,
    pythons: list[str],
    declared: list[Declaration],
    major: int,
) -> str | None:
    """The first rule `release` breaks (a short phrase), or None when it is eligible."""
    if release.yanked:
        return "yanked or without files"
    if release.uploaded is None:
        return "no readable upload time"
    age = now - release.uploaded
    if age < COOLDOWN:
        return f"cooldown ({age.days} days old, needs {COOLDOWN.days})"
    for python in pythons:
        if not admits_python(release.requires_python, python):
            return f"Requires-Python {release.requires_python or 'missing'} excludes {python}"
    if release.version[0] != major:
        return f"another major ({release.version[0]}, the pin is {major})"
    if not range_decision(declared, release, major)[0]:
        return "outside the declared range"
    return None


def choose(
    package: str,
    document: Any,
    pin_text: str,
    declared: list[Declaration],
    pythons: list[str],
    now: datetime,
) -> Choice:
    """Pick the highest eligible release of `package`; never lower than the pin."""
    pin = parse_release(pin_text)
    assert pin is not None
    releases = read_releases(package, document)
    verdicts = [(r, why_not(r, now, pythons, declared, pin[0])) for r in releases]
    eligible = [r for r, problem in verdicts if problem is None]
    best = eligible[-1] if eligible else None
    current = next((r for r in releases if _compare(r.version, pin) == 0), None)
    skipped = [
        f"{package} {r.text}: {problem}"
        for r, problem in reversed(verdicts)
        if problem is not None and _compare(r.version, pin) > 0
    ][:8]
    if best is not None and _compare(best.version, pin) > 0:
        ok, new_specs = range_decision(declared, best, pin[0])
        assert ok
        return Choice(
            package,
            pin_text,
            best.text,
            best.uploaded.date().isoformat() if best.uploaded else "-",
            str(_age_days(best, now)),
            "newer eligible release"
            + (" (declared range widened)" if any(s for s in new_specs) else ""),
            True,
            skipped,
            new_specs,
        )
    if best is not None and _compare(best.version, pin) < 0:
        reason = (
            f"newest eligible release {best.text} is older than the pin (no downgrade)"
        )
    elif best is None:
        reason = "no release is eligible yet (cooldown, range or Python); the pin stays"
    else:
        reason = "the pin is the newest eligible release"
    date = current.uploaded.date().isoformat() if current and current.uploaded else "-"
    age = str(_age_days(current, now)) if current and current.uploaded else "-"
    return Choice(package, pin_text, pin_text, date, age, reason, False, skipped, [])


# --------------------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------------------


def _atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".refresh-tmp")
    try:
        with open(temporary, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.replace(temporary, path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise RefreshError(f"cannot write {path}: {exc.strerror or exc}") from exc


def _read_text(path: Path, what: str) -> str:
    try:
        with open(path, encoding="utf-8", newline="") as handle:
            return handle.read()
    except OSError as exc:
        raise RefreshError(f"cannot read {what} {path}: {exc.strerror or exc}") from exc


def _diff(name: str, before: str, after: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            f"a/{name}",
            f"b/{name}",
        )
    )


def render(choices: list[Choice], now: datetime, pythons: list[str]) -> list[str]:
    """The markdown report lines (table, then the skipped newer releases)."""
    out = [
        f"## Pin refresh at {now.strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "",
        f"Cooldown {COOLDOWN.days} days; Python versions checked: {', '.join(pythons)}.",
        "",
        "| package | old pin | new pin | release date | age (days) | reason |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for c in choices:
        out.append(
            f"| {c.package} | {c.old} | {c.new} | {c.release_date} | {c.age_days} | {c.reason} |"
        )
    skipped = [line for c in choices for line in c.skipped]
    if skipped:
        out += ["", "Newer releases skipped:", ""] + [f"- {line}" for line in skipped]
    return out


def run(
    *,
    constraints_path: Path,
    pyproject_path: Path,
    packages: list[str],
    pythons: list[str],
    now: datetime,
    fetch: Fetcher,
    dry_run: bool,
    out: Callable[[str], None] = print,
) -> bool:
    """Do the refresh; return True when a file changed (or would, with `dry_run`).

    Raises RefreshError before writing anything when any package fails.
    """
    constraints = _read_text(constraints_path, "constraints")
    pyproject_text = _read_text(pyproject_path, "pyproject")
    try:
        pyproject = tomllib.loads(pyproject_text)
    except tomllib.TOMLDecodeError as exc:
        raise RefreshError(f"{pyproject_path} is not valid TOML: {exc}") from exc
    floor = (pyproject.get("project") or {}).get("requires-python")
    for python in pythons:
        if not admits_python(floor, python):
            raise RefreshError(
                f"Python {python} is outside requires-python {floor!r} of {pyproject_path}"
            )
    choices: list[Choice] = []
    new_constraints = constraints
    new_pyproject = pyproject_text
    for package in sorted(set(packages)):
        _, _, pin_text = read_pin(constraints, package)
        declared = declarations(pyproject, package)
        if not declared:
            raise RefreshError(f"{package}: no declared range in {pyproject_path}")
        pin = parse_release(pin_text)
        assert pin is not None
        for item in declared:
            if not admits(item.clauses, pin):
                raise RefreshError(
                    f"{package}: the pin {pin_text} is outside the declared range {item.text!r}"
                )
        choice = choose(package, fetch(package), pin_text, declared, pythons, now)
        choices.append(choice)
        if choice.changed:
            new_constraints = rewrite_pin(new_constraints, package, choice.new)
            for item, new_spec in zip(declared, choice.new_specs, strict=True):
                if new_spec is None:
                    continue
                updated = item.text.replace(item.spec, new_spec, 1)
                quoted = f'"{item.text}"'
                if new_pyproject.count(quoted) < 1:
                    raise RefreshError(
                        f"{package}: {item.text!r} not found in {pyproject_path}"
                    )
                new_pyproject = new_pyproject.replace(quoted, f'"{updated}"')
    changed = new_constraints != constraints or new_pyproject != pyproject_text
    for line in render(choices, now, pythons):
        out(line)
    if changed:
        out("")
        diff = _diff(constraints_path.name, constraints, new_constraints) + _diff(
            pyproject_path.name, pyproject_text, new_pyproject
        )
        if dry_run:
            out("Dry run: nothing written. The change would be:")
            out("")
        out("```diff")
        out(diff.rstrip("\n"))
        out("```")
        if not dry_run:
            if new_constraints != constraints:
                _atomic_write(constraints_path, new_constraints)
            if new_pyproject != pyproject_text:
                _atomic_write(pyproject_path, new_pyproject)
    return changed


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> Any:
        self.print_usage(sys.stderr)
        print(f"{self.prog}: error: {message}", file=sys.stderr)
        raise SystemExit(2)


def _package(value: str) -> str:
    if not _NAME.match(value):
        raise argparse.ArgumentTypeError(f"not a package name: {value!r}")
    return value


def _python(value: str) -> str:
    version = parse_release(value)
    if version is None or len(version) != 2:
        raise argparse.ArgumentTypeError(f"expected a version like 3.12, got {value!r}")
    return value


def _now(value: str) -> datetime:
    parsed = _parse_time(value)
    if parsed is None:
        raise argparse.ArgumentTypeError(f"not an ISO 8601 time: {value!r}")
    return parsed.astimezone(UTC)


def _build_parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="refresh_pins.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("--packages", nargs="+", type=_package, default=list(TRACKED))
    parser.add_argument("--constraints", type=Path, default=DEFAULT_CONSTRAINTS)
    parser.add_argument("--pyproject", type=Path, default=DEFAULT_PYPROJECT)
    parser.add_argument("--workflow", type=Path, default=DEFAULT_WORKFLOW)
    parser.add_argument(
        "--python", nargs="+", type=_python, dest="pythons", default=None
    )
    parser.add_argument(
        "--now", type=_now, default=None, help="ISO 8601 (default: now, UTC)"
    )
    parser.add_argument(
        "--json-dir", type=Path, default=None, help="read DIR/<package>.json"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="write nothing, print the diff"
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    fetch: Fetcher | None = None,
    stdout: Callable[[str], None] = print,
) -> int:
    """Run the CLI; return the exit code. `fetch` replaces the network (tests)."""
    try:
        args = _build_parser().parse_args(argv)
    except SystemExit as exc:
        return 2 if exc.code else 0
    try:
        if args.pythons:
            pythons = sorted(
                set(args.pythons), key=lambda p: tuple(map(int, p.split(".")))
            )
        else:
            pythons = matrix_pythons(_read_text(args.workflow, "workflow"))
            if not pythons:
                raise RefreshError(
                    f"no python-version found in the matrix of {args.workflow}"
                )
        now = args.now or datetime.now(UTC)
        if fetch is None:
            fetch = dir_fetcher(args.json_dir) if args.json_dir else fetch_pypi
        lines: list[str] = []
        changed = run(
            constraints_path=args.constraints,
            pyproject_path=args.pyproject,
            packages=args.packages,
            pythons=pythons,
            now=now,
            fetch=fetch,
            dry_run=args.dry_run,
            out=lines.append,
        )
    except RefreshError as exc:
        print(f"refresh_pins: {exc}", file=sys.stderr)
        return 1
    for line in lines:
        stdout(line)
    stdout(f"changed={'true' if changed else 'false'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
