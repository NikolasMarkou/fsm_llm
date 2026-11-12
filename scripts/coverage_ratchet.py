#!/usr/bin/env python
"""A coverage ratchet: the committed baseline only goes up.

Stdlib only, no network. Three commands, exit 0 when fine and 1 on any failure
(a usage error exits 1 too):

    coverage_ratchet.py check COVERAGE_JSON [--baseline FILE] [--tolerance 0.10]
    coverage_ratchet.py update COVERAGE_JSON [--baseline FILE]
    coverage_ratchet.py guard [--baseline FILE] [--ref origin/main] [--repo DIR]

COVERAGE_JSON is the report of `pytest --cov --cov-branch --cov-report=json:PATH`.

* `check` fails when the combined or the branch percent is below the baseline
  minus the tolerance (default 0.10 percentage points). The line percent is only
  recorded.
* `update` raises each number of the baseline to floor(measured, 0.01) and never
  lowers one; an unchanged baseline is not rewritten. It creates the file when it
  does not exist. Run it only on the canonical leg (Linux, Python 3.12, `mcp`
  installed): a local run without `mcp` measures too low to be a baseline.
* `guard` compares the working baseline with `git show REF:coverage-baseline.json`
  and fails on any decrease of combined, branch or line; a REF that holds no
  baseline yet passes. That makes "never decreasing" hold across a branch, not
  just inside one run.

Percentages are `decimal.Decimal` so a floor to two places is exact. Output is
deterministic (fixed key order, no timestamp from the clock: the `measured` date
is the one in the report).
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal, InvalidOperation
from pathlib import Path
from typing import Any, NoReturn

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE = ROOT / "coverage-baseline.json"
DEFAULT_TOLERANCE = Decimal("0.10")
DEFAULT_REF = "origin/main"
SCHEMA = 1
CANONICAL = "linux / 3.12 / mcp installed"
CENT = Decimal("0.01")
#: Baseline keys the ratchet compares, with the name used in messages.
METRICS = (
    ("combined_percent", "combined"),
    ("branch_percent", "branch"),
    ("line_percent", "line"),
)
_REQUIRED_KEYS = ("schema", "canonical", *(key for key, _ in METRICS))


class RatchetError(Exception):
    """A refusal with a message for stderr; `main` turns it into exit 1."""


@dataclass(frozen=True)
class Totals:
    """Percentages read from a coverage report: `combined`, `branch`, `line`, a `measured` date."""

    combined: Decimal
    branch: Decimal
    line: Decimal
    measured: str


def _load_json(path: Path, what: str) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RatchetError(f"cannot read {what} {path}: {exc.strerror or exc}") from exc
    try:
        data = json.loads(text, parse_float=Decimal)
    except ValueError as exc:
        raise RatchetError(f"{what} {path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise RatchetError(f"{what} {path} is not a JSON object")
    return data


def _decimal(value: Any, label: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise RatchetError(f"{label} is not a number: {value!r}")
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise RatchetError(f"{label} is not a number: {value!r}") from exc
    if not number.is_finite() or not Decimal(0) <= number <= Decimal(100):
        raise RatchetError(f"{label} is outside 0 to 100: {value!r}")
    return number


def _ratio(totals: dict[str, Any], covered: str, total: str) -> Decimal | None:
    if covered in totals and total in totals:
        count = Decimal(str(totals[total]))
        if count > 0:
            return Decimal(str(totals[covered])) * 100 / count
    return None


def read_report(path: Path) -> Totals:
    """Read the totals of a `coverage json` report.

    Parameters: `path` of the report. Returns `Totals`. Raises `RatchetError`
    when the file is unreadable, is not a report, or holds no branch data (the
    run lacked `--cov-branch`). The branch and line percentages fall back to the
    covered and total counts when a coverage release does not write the percent.
    """
    data = _load_json(path, "coverage report")
    totals = data.get("totals")
    if not isinstance(totals, dict) or "percent_covered" not in totals:
        raise RatchetError(f"coverage report {path} has no totals.percent_covered")
    branch = totals.get("percent_branches_covered")
    if branch is None:
        branch = _ratio(totals, "covered_branches", "num_branches")
    if branch is None:
        raise RatchetError(
            f"coverage report {path} has no branch data: run pytest with --cov-branch"
        )
    line = totals.get("percent_statements_covered")
    if line is None:
        line = _ratio(totals, "covered_lines", "num_statements")
    if line is None:
        raise RatchetError(f"coverage report {path} has no statement percentage")
    meta = data.get("meta")
    stamp = meta.get("timestamp", "") if isinstance(meta, dict) else ""
    return Totals(
        combined=_decimal(totals["percent_covered"], "percent_covered"),
        branch=_decimal(branch, "percent_branches_covered"),
        line=_decimal(line, "percent_statements_covered"),
        measured=str(stamp)[:10],
    )


def read_baseline(path: Path) -> dict[str, Any]:
    """Read and validate a baseline file; its percentages come back as `Decimal`.

    Raises `RatchetError` for a missing, unreadable or malformed file or a schema
    this script does not know.
    """
    data = _load_json(path, "baseline")
    missing = [key for key in _REQUIRED_KEYS if key not in data]
    if missing:
        raise RatchetError(f"baseline {path} lacks {', '.join(missing)}")
    if data["schema"] != SCHEMA:
        raise RatchetError(
            f"baseline {path} has schema {data['schema']!r}, not {SCHEMA}"
        )
    for key, _ in METRICS:
        data[key] = _decimal(data[key], key)
    return data


def _floor(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_FLOOR)


def _number(value: Decimal) -> float:
    return float(value)


def _tolerance(text: str) -> Decimal:
    try:
        value = Decimal(text)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from exc
    if not value.is_finite() or value < 0:
        raise argparse.ArgumentTypeError(f"must be zero or more: {text!r}")
    return value


def command_check(report: Path, baseline: Path, tolerance: Decimal) -> int:
    """Compare a report with the baseline; return 0 or 1 and print one line per metric."""
    totals = read_report(report)
    base = read_baseline(baseline)
    failed = False
    gated = (
        ("combined_percent", "combined", totals.combined),
        ("branch_percent", "branch", totals.branch),
    )
    for key, name, measured in gated:
        floor = base[key] - tolerance
        if measured < floor:
            failed = True
            print(
                f"FAIL {name}: {measured:.2f} is below {floor:.2f} "
                f"(baseline {base[key]:.2f} minus {tolerance} pp)",
                file=sys.stderr,
            )
        else:
            print(
                f"ok {name}: {measured:.2f} (baseline {base[key]:.2f}, floor {floor:.2f})"
            )
    return 1 if failed else 0


def command_update(
    report: Path, baseline: Path, python: str | None, mcp: str | None
) -> int:
    """Raise the baseline to the report's floors; never lower; create the file if absent."""
    totals = read_report(report)
    floors = {
        "combined_percent": _floor(totals.combined),
        "branch_percent": _floor(totals.branch),
        "line_percent": _floor(totals.line),
    }
    old = read_baseline(baseline) if baseline.exists() else None
    raised = old is None
    merged: dict[str, Decimal] = {}
    for key, _ in METRICS:
        previous = old[key] if old else None
        merged[key] = floors[key] if previous is None else max(previous, floors[key])
        raised = raised or (previous is not None and floors[key] > previous)
    if not raised:
        print(f"baseline unchanged: {baseline.name} is already at or above this report")
        return 0
    data = {
        "schema": SCHEMA,
        "canonical": CANONICAL,
        **{key: _number(merged[key]) for key, _ in METRICS},
        "python": python or platform.python_version(),
        "mcp": mcp if mcp is not None else _installed_version("mcp"),
        "measured": totals.measured,
    }
    text = json.dumps(data, indent=2) + "\n"
    try:
        with baseline.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    except OSError as exc:
        raise RatchetError(f"cannot write {baseline}: {exc.strerror or exc}") from exc
    print(
        "baseline raised: "
        + ", ".join(f"{name} {merged[key]:.2f}" for key, name in METRICS)
    )
    return 0


def _installed_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
    except OSError as exc:
        raise RatchetError(f"cannot run git: {exc.strerror or exc}") from exc


def command_guard(baseline: Path, ref: str, repo: Path) -> int:
    """Fail when the working baseline is below the one committed at `ref`.

    `repo` is the git repository, `baseline` the working file (its name relative to
    `repo` is looked up at `ref`). A ref with no baseline yet passes.
    """
    working = read_baseline(baseline)
    inside = _git(repo, "rev-parse", "--is-inside-work-tree")
    if inside.returncode != 0:
        raise RatchetError(f"{repo} is not a git repository")
    if _git(repo, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}").returncode:
        raise RatchetError(f"git ref {ref} does not exist (fetch it before the guard)")
    try:
        relative = baseline.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError as exc:
        raise RatchetError(f"{baseline} is not inside {repo}") from exc
    shown = _git(repo, "show", f"{ref}:{relative}")
    if shown.returncode != 0:
        print(f"ok: {ref} holds no {relative} yet, nothing to compare")
        return 0
    try:
        committed = json.loads(shown.stdout, parse_float=Decimal)
    except ValueError as exc:
        raise RatchetError(f"{relative} at {ref} is not valid JSON: {exc}") from exc
    if not isinstance(committed, dict):
        raise RatchetError(f"{relative} at {ref} is not a JSON object")
    failed = False
    for key, name in METRICS:
        if key not in committed:
            continue
        before = _decimal(committed[key], f"{key} at {ref}")
        if working[key] < before:
            failed = True
            print(
                f"FAIL {name}: {working[key]:.2f} is below {before:.2f} at {ref}",
                file=sys.stderr,
            )
        else:
            print(f"ok {name}: {working[key]:.2f} (at {ref} {before:.2f})")
    return 1 if failed else 0


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        print(f"{self.prog}: error: {message}", file=sys.stderr)
        raise SystemExit(1)


def _build_parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="coverage_ratchet.py", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(
        dest="command", required=True, parser_class=_Parser
    )
    for name in ("check", "update"):
        sub = commands.add_parser(name)
        sub.add_argument("report", type=Path, metavar="COVERAGE_JSON")
        sub.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
        if name == "check":
            sub.add_argument("--tolerance", type=_tolerance, default=DEFAULT_TOLERANCE)
        else:
            sub.add_argument("--python", help="version to record (default: this one)")
            sub.add_argument("--mcp", help="mcp version to record (default: installed)")
    guard = commands.add_parser("guard")
    guard.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    guard.add_argument("--ref", default=DEFAULT_REF)
    guard.add_argument("--repo", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI; return the exit code (0 ok, 1 failure or usage error)."""
    try:
        args = _build_parser().parse_args(argv)
    except SystemExit as exc:
        return 1 if exc.code else 0
    try:
        if args.command == "check":
            return command_check(args.report, args.baseline, args.tolerance)
        if args.command == "update":
            return command_update(args.report, args.baseline, args.python, args.mcp)
        repo = args.repo or args.baseline.resolve().parent
        return command_guard(args.baseline, args.ref, repo)
    except RatchetError as exc:
        print(f"coverage_ratchet: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
