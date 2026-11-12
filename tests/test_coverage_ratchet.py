"""Offline tests for ``scripts/coverage_ratchet.py`` and the coverage configuration.

The ratchet makes "coverage never decreases" structural: ``check`` compares a
``coverage json`` report with the committed ``coverage-baseline.json``, ``update``
only raises that file, ``guard`` compares it with ``origin/main``. Every test uses
tmp files and a throwaway git repository; nothing here runs coverage or the network.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tomllib
from decimal import Decimal
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import coverage_ratchet as cr

BASELINE = {
    "schema": 1,
    "canonical": "linux / 3.12 / mcp installed",
    "combined_percent": 95.02,
    "branch_percent": 92.06,
    "line_percent": 95.92,
    "python": "3.12.3",
    "mcp": "2.2.0",
    "measured": "2026-11-12",
}


def _report(combined: float, branch: float | None, line: float = 95.0) -> dict:
    """The part of a `coverage json` report the ratchet reads."""
    totals: dict = {"percent_covered": combined, "percent_statements_covered": line}
    if branch is not None:
        totals["percent_branches_covered"] = branch
    return {
        "meta": {"timestamp": "2026-11-12T10:20:30.123456", "branch_coverage": True},
        "totals": totals,
    }


def _write_json(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _baseline(tmp_path: Path, **overrides) -> Path:
    return _write_json(tmp_path / "coverage-baseline.json", {**BASELINE, **overrides})


def _cov(tmp_path: Path, combined: float, branch: float | None, line: float = 95.0):
    return _write_json(tmp_path / "cov.json", _report(combined, branch, line))


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class TestCheck:
    """`check` fails when combined or branch percent drops below baseline minus 0.10 pp."""

    def test_equal_to_the_baseline_passes(self, tmp_path, capsys):
        cov = _cov(tmp_path, 95.02, 92.06)
        code = cr.main(["check", str(cov), "--baseline", str(_baseline(tmp_path))])
        assert code == 0
        assert "ok" in capsys.readouterr().out

    def test_exactly_the_tolerance_below_passes(self, tmp_path):
        cov = _cov(tmp_path, 94.92, 91.96)
        assert cr.main(["check", str(cov), "--baseline", str(_baseline(tmp_path))]) == 0

    def test_combined_below_the_tolerance_fails(self, tmp_path, capsys):
        cov = _cov(tmp_path, 94.91, 92.06)
        code = cr.main(["check", str(cov), "--baseline", str(_baseline(tmp_path))])
        assert code == 1
        captured = capsys.readouterr()
        assert "combined" in captured.err + captured.out

    def test_branch_below_the_tolerance_fails(self, tmp_path, capsys):
        cov = _cov(tmp_path, 96.0, 91.95)
        code = cr.main(["check", str(cov), "--baseline", str(_baseline(tmp_path))])
        assert code == 1
        captured = capsys.readouterr()
        assert "branch" in captured.err + captured.out

    def test_a_higher_measure_passes(self, tmp_path):
        cov = _cov(tmp_path, 97.5, 94.0)
        assert cr.main(["check", str(cov), "--baseline", str(_baseline(tmp_path))]) == 0

    def test_tolerance_option_widens_and_narrows(self, tmp_path):
        base = str(_baseline(tmp_path))
        cov = str(_cov(tmp_path, 94.5, 91.6))
        assert cr.main(["check", cov, "--baseline", base, "--tolerance", "0.6"]) == 0
        assert cr.main(["check", cov, "--baseline", base, "--tolerance", "0.3"]) == 1

    def test_default_tolerance_is_ten_hundredths(self):
        assert cr.DEFAULT_TOLERANCE == Decimal("0.10")

    def test_a_report_without_branch_data_fails(self, tmp_path, capsys):
        cov = _cov(tmp_path, 99.0, None)
        code = cr.main(["check", str(cov), "--baseline", str(_baseline(tmp_path))])
        assert code == 1
        assert "--cov-branch" in capsys.readouterr().err

    def test_a_missing_baseline_fails(self, tmp_path, capsys):
        cov = _cov(tmp_path, 99.0, 99.0)
        code = cr.main(["check", str(cov), "--baseline", str(tmp_path / "nope.json")])
        assert code == 1
        assert "baseline" in capsys.readouterr().err

    def test_a_missing_or_unreadable_report_fails(self, tmp_path):
        base = str(_baseline(tmp_path))
        assert cr.main(["check", str(tmp_path / "no.json"), "--baseline", base]) == 1
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        assert cr.main(["check", str(bad), "--baseline", base]) == 1

    def test_a_negative_tolerance_is_refused(self, tmp_path):
        cov = _cov(tmp_path, 99.0, 99.0)
        base = str(_baseline(tmp_path))
        args = ["check", str(cov), "--baseline", base, "--tolerance", "-1"]
        assert cr.main(args) == 1

    def test_the_default_baseline_is_the_repo_root_file(self):
        assert cr.DEFAULT_BASELINE == ROOT / "coverage-baseline.json"

    def test_percent_is_derived_from_counts_when_a_field_is_absent(self, tmp_path):
        """An older coverage json has counts but no percent_branches_covered."""
        report = {
            "meta": {"timestamp": "2026-11-12T10:20:30"},
            "totals": {
                "percent_covered": 95.5,
                "covered_lines": 90,
                "num_statements": 100,
                "covered_branches": 46,
                "num_branches": 50,
            },
        }
        cov = _write_json(tmp_path / "c.json", report)
        totals = cr.read_report(cov)
        assert totals.line == Decimal("90")
        assert totals.branch == Decimal("92")


class TestUpdate:
    """`update` raises the committed baseline and never lowers it."""

    def test_creates_the_file_with_floors_to_two_places(self, tmp_path):
        out = tmp_path / "coverage-baseline.json"
        cov = _cov(tmp_path, 95.0299, 92.0599, 95.9999)
        code = cr.main(["update", str(cov), "--baseline", str(out), "--mcp", "2.2.0"])
        assert code == 0
        data = _read(out)
        assert data["combined_percent"] == 95.02
        assert data["branch_percent"] == 92.05
        assert data["line_percent"] == 95.99
        assert data["schema"] == 1
        assert data["canonical"] == "linux / 3.12 / mcp installed"
        assert data["measured"] == "2026-11-12"
        assert data["mcp"] == "2.2.0"
        assert re.fullmatch(r"\d+\.\d+\.\d+", data["python"])

    def test_raises_each_number_that_improved(self, tmp_path):
        base = _baseline(tmp_path)
        cov = _cov(tmp_path, 96.123, 93.456, 96.5)
        assert cr.main(["update", str(cov), "--baseline", str(base)]) == 0
        data = _read(base)
        assert (data["combined_percent"], data["branch_percent"]) == (96.12, 93.45)
        assert data["line_percent"] == 96.5

    def test_never_lowers_a_number(self, tmp_path):
        base = _baseline(tmp_path)
        cov = _cov(tmp_path, 90.0, 80.0, 70.0)
        assert cr.main(["update", str(cov), "--baseline", str(base)]) == 0
        data = _read(base)
        assert data["combined_percent"] == 95.02
        assert data["branch_percent"] == 92.06
        assert data["line_percent"] == 95.92

    def test_a_mixed_report_raises_one_and_keeps_the_other(self, tmp_path):
        base = _baseline(tmp_path)
        cov = _cov(tmp_path, 96.0, 90.0, 95.92)
        assert cr.main(["update", str(cov), "--baseline", str(base)]) == 0
        data = _read(base)
        assert data["combined_percent"] == 96.0
        assert data["branch_percent"] == 92.06

    def test_an_unchanged_baseline_is_not_rewritten(self, tmp_path, capsys):
        base = _baseline(tmp_path)
        before = base.read_bytes()
        os.utime(base, (1, 1))
        cov = _cov(tmp_path, 95.02, 92.06, 95.92)
        assert cr.main(["update", str(cov), "--baseline", str(base)]) == 0
        assert base.read_bytes() == before
        assert base.stat().st_mtime == 1
        assert "unchanged" in capsys.readouterr().out

    def test_floor_never_rounds_up(self, tmp_path):
        out = tmp_path / "b.json"
        cov = _cov(tmp_path, 92.0599999, 80.009999, 99.999999)
        assert cr.main(["update", str(cov), "--baseline", str(out)]) == 0
        data = _read(out)
        assert data["combined_percent"] == 92.05
        assert data["branch_percent"] == 80.0
        assert data["line_percent"] == 99.99

    def test_output_is_deterministic(self, tmp_path):
        cov = str(_cov(tmp_path, 95.5, 93.5, 96.5))
        a, b = tmp_path / "a.json", tmp_path / "b.json"
        for out in (a, b):
            args = ["update", cov, "--baseline", str(out), "--mcp", "2.2.0"]
            assert cr.main([*args, "--python", "3.12.3"]) == 0
        assert a.read_bytes() == b.read_bytes()
        assert a.read_bytes().endswith(b"\n")
        assert b"\r" not in a.read_bytes()

    def test_a_report_without_branch_data_is_refused_and_writes_nothing(self, tmp_path):
        out = tmp_path / "b.json"
        cov = _cov(tmp_path, 99.0, None)
        assert cr.main(["update", str(cov), "--baseline", str(out)]) == 1
        assert not out.exists()

    def test_a_corrupt_baseline_is_refused_not_overwritten(self, tmp_path):
        out = tmp_path / "b.json"
        out.write_text("{broken", encoding="utf-8")
        cov = _cov(tmp_path, 99.0, 99.0)
        assert cr.main(["update", str(cov), "--baseline", str(out)]) == 1
        assert out.read_text(encoding="utf-8") == "{broken"

    def test_check_passes_against_a_baseline_just_written(self, tmp_path):
        out = tmp_path / "b.json"
        cov = str(_cov(tmp_path, 95.0299, 92.0599, 95.5))
        assert cr.main(["update", cov, "--baseline", str(out)]) == 0
        assert cr.main(["check", cov, "--baseline", str(out)]) == 0


def _git(repo: Path, *args: str) -> None:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_CONFIG_SYSTEM"] = os.devnull
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        check=True,
        capture_output=True,
        env=env,
    )


def _repo_with_origin_main(tmp_path: Path, committed: dict | None) -> Path:
    """A throwaway repo whose `origin/main` ref holds `committed` (no remote, no network)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "README").write_text("x\n", encoding="utf-8")
    if committed is not None:
        _write_json(repo / "coverage-baseline.json", committed)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    return repo


class TestGuard:
    """`guard` compares the working baseline with `origin/main` and fails on any decrease."""

    def _run(self, repo: Path, *extra: str) -> int:
        return cr.main(
            ["guard", "--baseline", str(repo / "coverage-baseline.json"), *extra]
        )

    def test_equal_passes(self, tmp_path):
        repo = _repo_with_origin_main(tmp_path, BASELINE)
        assert self._run(repo) == 0

    def test_raised_passes(self, tmp_path):
        repo = _repo_with_origin_main(tmp_path, BASELINE)
        _write_json(
            repo / "coverage-baseline.json",
            {**BASELINE, "combined_percent": 96.0, "branch_percent": 93.0},
        )
        assert self._run(repo) == 0

    @pytest.mark.parametrize(
        "key", ["combined_percent", "branch_percent", "line_percent"]
    )
    def test_any_decrease_fails(self, tmp_path, capsys, key):
        repo = _repo_with_origin_main(tmp_path, BASELINE)
        _write_json(
            repo / "coverage-baseline.json", {**BASELINE, key: BASELINE[key] - 0.01}
        )
        assert self._run(repo) == 1
        captured = capsys.readouterr()
        assert key.removesuffix("_percent") in captured.err + captured.out

    def test_decrease_is_judged_against_origin_main_not_the_working_head(
        self, tmp_path
    ):
        """A lowering commit on the branch is caught even though HEAD equals the file."""
        repo = _repo_with_origin_main(tmp_path, BASELINE)
        _write_json(
            repo / "coverage-baseline.json", {**BASELINE, "combined_percent": 90.0}
        )
        _git(repo, "commit", "-q", "-am", "lower it")
        assert self._run(repo) == 1

    def test_no_baseline_on_origin_main_yet_passes(self, tmp_path, capsys):
        repo = _repo_with_origin_main(tmp_path, None)
        _write_json(repo / "coverage-baseline.json", BASELINE)
        assert self._run(repo) == 0
        assert "origin/main" in capsys.readouterr().out

    def test_a_missing_ref_fails(self, tmp_path, capsys):
        repo = _repo_with_origin_main(tmp_path, BASELINE)
        assert self._run(repo, "--ref", "origin/nope") == 1
        assert "origin/nope" in capsys.readouterr().err

    def test_a_missing_working_baseline_fails(self, tmp_path):
        repo = _repo_with_origin_main(tmp_path, BASELINE)
        (repo / "coverage-baseline.json").unlink()
        assert self._run(repo) == 1

    def test_a_directory_that_is_not_a_repo_fails(self, tmp_path):
        _write_json(tmp_path / "coverage-baseline.json", BASELINE)
        assert self._run(tmp_path) == 1


class TestCommandLine:
    def test_script_runs_as_a_program_with_exit_codes(self, tmp_path):
        base = _baseline(tmp_path)
        good, bad = _cov(tmp_path, 95.02, 92.06), tmp_path / "bad.json"
        _write_json(bad, _report(50.0, 50.0))
        run = [sys.executable, str(SCRIPTS / "coverage_ratchet.py"), "check"]
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        ok = subprocess.run(
            [*run, str(good), "--baseline", str(base)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
        )
        fail = subprocess.run(
            [*run, str(bad), "--baseline", str(base)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
        )
        assert (ok.returncode, fail.returncode) == (0, 1)

    def test_a_usage_error_exits_one_not_two(self, tmp_path):
        assert cr.main([]) == 1
        assert cr.main(["frobnicate"]) == 1
        assert cr.main(["check"]) == 1

    def test_output_is_deterministic(self, tmp_path, capsys):
        base = str(_baseline(tmp_path))
        cov = str(_cov(tmp_path, 95.0, 92.0))
        outs = []
        for _ in range(2):
            cr.main(["check", cov, "--baseline", base])
            outs.append(capsys.readouterr())
        assert outs[0] == outs[1]
        assert outs[0].out or outs[0].err

    def test_top_level_imports_are_stdlib_only(self):
        import ast

        tree = ast.parse((SCRIPTS / "coverage_ratchet.py").read_text(encoding="utf-8"))
        roots = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                roots.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots.add(node.module.split(".")[0])
        assert roots <= set(sys.stdlib_module_names), roots - set(
            sys.stdlib_module_names
        )

    def test_script_opens_no_network(self):
        text = (SCRIPTS / "coverage_ratchet.py").read_text(encoding="utf-8")
        assert "socket" not in text and "urllib" not in text and "http" not in text


class TestCoverageConfiguration:
    """The config the ratchet depends on: branch mode, package source, one pin."""

    def _pyproject(self) -> dict:
        return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    def test_run_section_turns_branch_on_for_the_package(self):
        run = self._pyproject()["tool"]["coverage"]["run"]
        assert run["branch"] is True
        assert run["source"] == ["fsm_llm"]

    def test_report_section_shows_two_places_and_missing_lines(self):
        report = self._pyproject()["tool"]["coverage"]["report"]
        assert report["precision"] == 2
        assert report["show_missing"] is True

    def test_coverage_is_pinned_exactly_once_in_constraints(self):
        lines = [
            ln
            for ln in (ROOT / "constraints.txt")
            .read_text(encoding="utf-8")
            .splitlines()
            if re.match(r"coverage==\d", ln)
        ]
        assert len(lines) == 1, lines

    def test_coverage_is_in_the_dev_extra_with_an_upper_bound(self):
        dev = self._pyproject()["project"]["optional-dependencies"]["dev"]
        mine = [r for r in dev if re.match(r"coverage\b", r)]
        assert len(mine) == 1 and "<8" in mine[0], mine

    @pytest.mark.parametrize("rel", ["Makefile", "tox.ini"])
    def test_coverage_commands_run_branch_coverage_on_the_package(self, rel):
        lines = [
            ln
            for ln in (ROOT / rel).read_text(encoding="utf-8").splitlines()
            if "--cov=fsm_llm" in ln
        ]
        assert len(lines) == 1, lines
        assert "--cov-branch" in lines[0]
