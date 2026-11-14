"""The CI workflow: a nine-leg matrix, pinned actions, a read-only token and a coverage ratchet.

Plan d454fd62, step 108 (roadmap rows 0.1 and 0.2). `.github/workflows/python-package.yml` is
what every push and pull request runs, and none of it runs on a developer machine, so it is read
here as data.

* `workflow_problems(text)` is a pure function of the workflow text and returns a set of problem
  codes. The real workflow must return the empty set (`TestTheRealWorkflow`); each of the
  mutants of `MUTANTS` is the real text with one edit and must return its own code
  (`TestMutantsAreKilled`): a check that cannot fail is not a check.
* The legs are COMPUTED from the YAML (`matrix_legs`, GitHub's own include and exclude rules for
  the shapes used here), never grepped, so the matrix may be written as an explicit `include`
  list or as an `os` by `python-version` product with `exclude`; the exact set of nine
  (os, python) pairs is what is pinned.
* Only the standard library and `yaml.safe_load` are used. PyYAML reads the key `on` as the
  boolean `True` (YAML 1.1); `_load` puts it back under `"on"`.
* The action pins are 40 hex characters with the release tag in a trailing `# vX.Y.Z` comment,
  the same table as `docs.yml` (progress.md "Step 106"); `test_supply_chain.py` (step 109) reads
  every workflow file for the same rule and for the write-permission allowlist.
* The job `lint` runs once (ruff, format, mypy, the `.pth` audit, the import contracts, the
  decision gate, `actionlint` and `zizmor`); the job `test` is the matrix and runs in parallel
  with it, beside `supply-chain` (audits the installed environment, `test_supply_chain.py`) and
  `dependency-review` (pull requests only, the one job with a write scope). The canonical leg
  (Linux, Python 3.12, `mcp` installed: the one the coverage baseline is measured on) runs the
  coverage form of the same marker selection, then the ratchet.
"""

from __future__ import annotations

import itertools
import re
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW = _ROOT / ".github" / "workflows" / "python-package.yml"

#: The nine legs: Linux on every supported Python, macOS and Windows on the two ends.
_LINUX = "ubuntu-24.04"
_EXPECTED_LEGS = {(_LINUX, v) for v in ("3.11", "3.12", "3.13", "3.14", "3.15")} | {
    (os_name, v) for os_name in ("macos-15", "windows-2025") for v in ("3.11", "3.15")
}
_CANONICAL_LEG = (_LINUX, "3.12")

#: The check name of a leg, which branch protection refers to (the default name of a matrix leg
#: appends every value of its `include` entry, so the canonical leg would end in `, true`).
_TEST_JOB_NAME = "test (${{ matrix.os }}, Python ${{ matrix.python-version }})"

#: The marker selection AGENTS.md documents for CI ("with no deselect").
_MARKERS = "not slow and not real_llm and not integration"
_PYTEST_PLAIN = 'python -m pytest tests/ -q -m "$PYTEST_MARKERS"'
_PYTEST_COVERAGE = _PYTEST_PLAIN + " --cov --cov-branch --cov-report=json"
_IMPORT_CONTRACT = (
    "python -m pytest tests/test_fsm_llm_regression/test_import_contract.py -q"
)
_RATCHET_CHECK = "python scripts/coverage_ratchet.py check coverage.json"
_AUDIT = "python scripts/audit_pth.py"
_LINTERS_INSTALL = "pip install -c constraints.txt zizmor actionlint-py"
_ZIZMOR = "zizmor --offline .github/workflows"
_ACTIONLINT = "actionlint"

#: The four jobs: `lint` and the `test` matrix, the audit of the installed environment, and the
#: dependency review (pull requests only), whose write scope is the one grant the file has.
_JOBS = {"lint", "test", "supply-chain", "dependency-review"}
_REVIEW_PERMISSIONS = {"contents": "read", "pull-requests": "write"}
_CONCURRENCY_GROUP = "ci-${{ github.workflow }}-${{ github.ref }}"
_CONCURRENCY_CANCEL = "${{ github.event_name == 'pull_request' }}"

_IF_PLAIN = "${{ !matrix.coverage }}"
_IF_COVERAGE = "${{ matrix.coverage }}"
_IF_GUARD = "${{ matrix.coverage && github.event_name == 'pull_request' }}"

_SHA_USES = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+@[0-9a-f]{40}$")
_USES_LINE = re.compile(r"^\s*(?:-\s*)?uses:\s*(?P<ref>\S+)(?P<rest>.*)$", re.MULTILINE)
_TAG_COMMENT = re.compile(r"^\s*#\s*v\d+\.\d+\.\d+\s*$")
_INSTALL = re.compile(r'pip install -c constraints\.txt -e "\.\[(?P<extras>[^\]]+)\]"')


# --------------------------------------------------------------------------------------
# Reading the workflow
# --------------------------------------------------------------------------------------


def _load(text: str) -> dict[str, Any]:
    """`yaml.safe_load` with the key `on` (read by PyYAML as `True`) put back as `"on"`."""
    data = yaml.safe_load(text)
    assert isinstance(data, dict), "the workflow is not a mapping"
    if True in data and "on" not in data:
        data["on"] = data.pop(True)
    return data


def _steps(job: dict[str, Any]) -> list[dict[str, Any]]:
    return [s for s in job.get("steps", []) if isinstance(s, dict)]


def _runs(job: dict[str, Any]) -> list[str]:
    return [str(s["run"]).strip() for s in _steps(job) if "run" in s]


def _code_lines(text: str) -> list[str]:
    """The lines of the workflow that are not full-line comments."""
    return [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]


def _step_running(job: dict[str, Any], needle: str) -> dict[str, Any] | None:
    for step in _steps(job):
        if needle in str(step.get("run", "")):
            return step
    return None


def _index_of(job: dict[str, Any], needle: str) -> int:
    for i, step in enumerate(_steps(job)):
        if needle in str(step.get("run", "")):
            return i
    return -1


def _uses_steps(job: dict[str, Any], action: str) -> list[dict[str, Any]]:
    return [s for s in _steps(job) if str(s.get("uses", "")).startswith(action + "@")]


def matrix_legs(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    """The legs of a `strategy.matrix`, by GitHub's rules for the shapes used here.

    The axes (every key but `include` and `exclude`) make a product; an `exclude` entry removes
    every leg it matches; an `include` entry adds its extra keys to every original leg it
    matches without overwriting an axis value, and becomes a leg of its own when it matches none.
    With no axis at all, every `include` entry is a leg.
    """
    axes = {k: v for k, v in matrix.items() if k not in ("include", "exclude")}
    legs: list[dict[str, Any]] = (
        [
            dict(zip(axes, combo, strict=True))
            for combo in itertools.product(*axes.values())
        ]
        if axes
        else []
    )
    excluded = matrix.get("exclude") or []
    legs = [
        leg
        for leg in legs
        if not any(all(leg.get(k) == v for k, v in ex.items()) for ex in excluded)
    ]
    original = list(legs)
    for inc in matrix.get("include") or []:
        matched = False
        for leg in original:
            if all(leg[k] == v for k, v in inc.items() if k in axes):
                leg.update({k: v for k, v in inc.items() if k not in axes})
                matched = True
        if not matched or not axes:
            legs.append(dict(inc))
    return legs


def _pairs(legs: list[dict[str, Any]]) -> list[tuple[Any, Any]]:
    return [(leg.get("os"), leg.get("python-version")) for leg in legs]


def _install_extras(job: dict[str, Any]) -> list[set[str]]:
    """The extras named by every `pip install -c constraints.txt -e` line of a job."""
    found: list[set[str]] = []
    for run in _runs(job):
        for line in run.splitlines():
            if "pip install -c constraints.txt -e" in line:
                match = _INSTALL.search(line)
                found.append(set(match["extras"].split(",")) if match else set())
    return found


def workflow_problems(text: str) -> set[str]:
    """Return the codes of every rule the workflow text breaks (empty set: it is sound)."""
    found: set[str] = set()
    try:
        wf = _load(text)
    except Exception:
        return {"unreadable"}
    jobs = wf.get("jobs") or {}
    lint = jobs.get("lint") or {}
    test = jobs.get("test") or {}
    code = "\n".join(_code_lines(text))

    # Triggers: push and pull request to main, manual dispatch; nothing privileged.
    on = wf.get("on") or {}
    if not isinstance(on, dict):
        found.add("triggers")
    else:
        if (on.get("push") or {}).get("branches") != ["main"]:
            found.add("push-trigger")
        if (on.get("pull_request") or {}).get("branches") != ["main"]:
            found.add("pull-request-trigger")
        if "workflow_dispatch" not in on:
            found.add("dispatch-trigger")
        if set(on) != {"push", "pull_request", "workflow_dispatch"}:
            found.add("extra-trigger")
    if "pull_request_target" in code:
        found.add("pull-request-target")

    # Permissions: read at the top; only the dependency review grants itself a write scope (its
    # failure comment), no token beyond it.
    if wf.get("permissions") != {"contents": "read"}:
        found.add("top-permissions")
    for job_id, job in jobs.items():
        if job_id == "dependency-review":
            if job.get("permissions") != _REVIEW_PERMISSIONS:
                found.add("job-permissions")
        elif "permissions" in job:
            found.add("job-permissions")
    if re.search(r"\bid-token\b", code) or len(re.findall(r":\s*write\b", code)) != 1:
        found.add("extra-permission")
    if re.search(r"\bsecrets\.|GITHUB_TOKEN", code):
        found.add("secrets")

    # A superseded pull-request run is cancelled; a push to main never is.
    concurrency = wf.get("concurrency") or {}
    if (
        concurrency.get("group") != _CONCURRENCY_GROUP
        or concurrency.get("cancel-in-progress") != _CONCURRENCY_CANCEL
    ):
        found.add("concurrency")

    # Jobs: lint once, the matrix beside it (no `needs`), the two supply-chain jobs, nothing else
    # (docs.yml owns the docs).
    if set(jobs) != _JOBS:
        found.add("jobs")
    if any("needs" in job for job in jobs.values()):
        found.add("job-needs")
    if "mkdocs" in code:
        found.add("docs-build-in-ci")

    # The matrix: the exact nine legs, computed; strings, not floats; no early cancellation.
    strategy = test.get("strategy") or {}
    legs = matrix_legs(strategy.get("matrix") or {})
    pairs = _pairs(legs)
    if set(pairs) != _EXPECTED_LEGS or len(pairs) != len(_EXPECTED_LEGS):
        found.add("matrix-legs")
    if not all(isinstance(leg.get("python-version"), str) for leg in legs):
        found.add("python-version-type")
    if strategy.get("fail-fast") is not False:
        found.add("fail-fast")
    coverage_legs = [leg for leg in legs if leg.get("coverage")]
    if [_pairs([leg])[0] for leg in coverage_legs] != [_CANONICAL_LEG]:
        found.add("coverage-legs")

    # Runners, timeouts, shell, environment.
    if test.get("name") != _TEST_JOB_NAME:
        found.add("job-name")
    if test.get("runs-on") != "${{ matrix.os }}":
        found.add("runner")
    if lint.get("runs-on") != _LINUX:
        found.add("runner")
    if "-latest" in code:
        found.add("runner")
    for name, floor in (("lint", 10), ("test", 25)):
        value = (jobs.get(name) or {}).get("timeout-minutes")
        if not isinstance(value, int) or isinstance(value, bool) or value < floor:
            found.add("timeout")
    if (wf.get("defaults") or {}).get("run", {}).get("shell") != "bash":
        found.add("default-shell")
    env = test.get("env") or {}
    if env.get("PYTHONUTF8") != "1":
        found.add("pythonutf8")
    if env.get("PYTEST_MARKERS") != _MARKERS:
        found.add("markers")
    for job in jobs.values():
        for step in _steps(job):
            run = step.get("run")
            if isinstance(run, str) and "\n" in run.strip():
                if "set -euo pipefail" not in run:
                    found.add("shell-strict")
            if isinstance(run, str) and "${{" in run:
                found.add("run-interpolation")

    # Actions: a 40-hex SHA and a `# vX.Y.Z` comment on every `uses:`; three actions only.
    parsed_uses = [
        str(s["uses"]) for job in jobs.values() for s in _steps(job) if "uses" in s
    ]
    lines = list(_USES_LINE.finditer(text))
    if len(lines) != len(parsed_uses) or not parsed_uses:
        found.add("uses-count")
    for match in lines:
        if not _SHA_USES.match(match["ref"]):
            found.add("action-pin")
        if not _TAG_COMMENT.match(match["rest"]):
            found.add("action-tag-comment")
    if {u.split("@")[0] for u in parsed_uses} != {
        "actions/checkout",
        "actions/setup-python",
        "actions/dependency-review-action",
    }:
        found.add("action-set")
    for name, job in jobs.items():
        checkouts = _uses_steps(job, "actions/checkout")
        if (
            len(checkouts) != 1
            or (checkouts[0].get("with") or {}).get("persist-credentials") is not False
        ):
            found.add("persist-credentials")
        if name == "dependency-review":
            continue
        pythons = _uses_steps(job, "actions/setup-python")
        if len(pythons) != 1:
            found.add("setup-python")
            continue
        with_ = pythons[0].get("with") or {}
        want = "${{ matrix.python-version }}" if name == "test" else "3.12"
        if with_.get("python-version") != want:
            found.add("setup-python-version")
        if with_.get("cache") != "pip" or "constraints.txt" not in str(
            with_.get("cache-dependency-path")
        ):
            found.add("pip-cache")
        if "allow-prereleases" in with_:
            found.add("prereleases")

    # One install line per job that needs the package, naming `dev` and (on the matrix) `mcp`;
    # the `.pth` audit after it in the two jobs that run it.
    for name in ("lint", "test", "supply-chain"):
        job = jobs.get(name) or {}
        extras = _install_extras(job)
        if len(extras) != 1 or "dev" not in extras[0]:
            found.add("install")
        if name == "test" and not (extras and "mcp" in extras[0]):
            found.add("mcp")
        if name == "supply-chain":
            continue
        audit = _runs(job).count(_AUDIT)
        if audit != 1 or _index_of(job, _AUDIT) < _index_of(
            job, "pip install -c constraints.txt -e"
        ):
            found.add("audit-" + name)

    # The matrix job runs the marker selection, no deselect, the coverage form on one leg only.
    pytest_steps = [s for s in _steps(test) if "pytest" in str(s.get("run", ""))]
    shapes = {str(s["run"]).strip(): s.get("if") for s in pytest_steps}
    if shapes != {_PYTEST_PLAIN: _IF_PLAIN, _PYTEST_COVERAGE: _IF_COVERAGE}:
        found.add("pytest-steps")
    if "--deselect" in code:
        found.add("deselect")
    if "mypy" in "\n".join(_runs(test)):
        found.add("mypy-in-test")

    # The ratchet: check on the canonical leg after the run, guard on pull requests only.
    check = _step_running(test, _RATCHET_CHECK)
    if (
        check is None
        or check.get("if") != _IF_COVERAGE
        or _index_of(test, _RATCHET_CHECK) < _index_of(test, "--cov-report=json")
    ):
        found.add("ratchet-check")
    guard = _step_running(test, "coverage_ratchet.py guard")
    if guard is None or guard.get("if") != _IF_GUARD:
        found.add("ratchet-guard")
    elif "git fetch" not in str(guard["run"]) or "main" not in str(guard["run"]):
        found.add("ratchet-guard-fetch")

    # The lint job: every command, once, and the decision gate with its step summary.
    lint_runs = _runs(lint)
    for label, command in (
        ("ruff-check", "ruff check src/ tests/"),
        ("ruff-format", "ruff format --check src/ tests/"),
        ("mypy", "python -m mypy -p fsm_llm"),
        ("import-contract", _IMPORT_CONTRACT),
        (
            "decision-check",
            "python scripts/decision_index.py --check docs/decisions/index.md",
        ),
        ("linters-install", _LINTERS_INSTALL),
        ("zizmor", _ZIZMOR),
        ("actionlint", _ACTIONLINT),
    ):
        if lint_runs.count(command) != 1:
            found.add("lint-" + label)
    summary = _step_running(lint, "decision_index.py --report")
    if summary is None or "GITHUB_STEP_SUMMARY" not in str(summary["run"]):
        found.add("lint-decision-summary")
    return found


def _real_text() -> str:
    return _WORKFLOW.read_text(encoding="utf-8")


# --------------------------------------------------------------------------------------
# The matrix reader, on synthetic matrices
# --------------------------------------------------------------------------------------


class TestMatrixLegs:
    def test_an_include_only_matrix_is_its_entries(self):
        legs = matrix_legs({"include": [{"os": "a", "v": "1"}, {"os": "b", "v": "2"}]})
        assert [(leg["os"], leg["v"]) for leg in legs] == [("a", "1"), ("b", "2")]

    def test_a_product_with_excludes(self):
        legs = matrix_legs(
            {
                "os": ["a", "b"],
                "v": ["1", "2", "3"],
                "exclude": [{"os": "b", "v": "2"}, {"os": "b", "v": "3"}],
            }
        )
        assert [(leg["os"], leg["v"]) for leg in legs] == [
            ("a", "1"),
            ("a", "2"),
            ("a", "3"),
            ("b", "1"),
        ]

    def test_an_include_adds_a_key_to_matching_legs_and_makes_a_new_leg_otherwise(self):
        legs = matrix_legs(
            {
                "os": ["a", "b"],
                "include": [{"os": "a", "cov": True}, {"os": "c", "cov": True}],
            }
        )
        assert legs == [{"os": "a", "cov": True}, {"os": "b"}, {"os": "c", "cov": True}]

    def test_an_include_never_overwrites_an_axis_value(self):
        legs = matrix_legs({"v": ["1", "2"], "include": [{"v": "1", "extra": "x"}]})
        assert legs == [{"v": "1", "extra": "x"}, {"v": "2"}]


# --------------------------------------------------------------------------------------
# The real workflow
# --------------------------------------------------------------------------------------


class TestTheRealWorkflow:
    def test_the_workflow_keeps_its_name_and_reads_as_yaml(self):
        wf = _load(_real_text())
        assert wf["name"] == "Python package"

    def test_it_breaks_no_rule(self):
        assert workflow_problems(_real_text()) == set()

    def test_the_matrix_is_exactly_the_nine_legs(self):
        wf = _load(_real_text())
        legs = matrix_legs(wf["jobs"]["test"]["strategy"]["matrix"])
        assert len(legs) == 9
        assert set(_pairs(legs)) == _EXPECTED_LEGS
        assert len(set(_pairs(legs))) == 9

    def test_python_versions_are_strings_so_3_15_is_not_a_float(self):
        wf = _load(_real_text())
        legs = matrix_legs(wf["jobs"]["test"]["strategy"]["matrix"])
        assert {leg["python-version"] for leg in legs} == {
            "3.11",
            "3.12",
            "3.13",
            "3.14",
            "3.15",
        }
        assert all(isinstance(leg["python-version"], str) for leg in legs)

    def test_the_matrix_does_not_cancel_the_other_legs(self):
        wf = _load(_real_text())
        assert wf["jobs"]["test"]["strategy"]["fail-fast"] is False

    def test_the_floor_of_the_matrix_is_the_floor_of_requires_python(self):
        wf = _load(_real_text())
        legs = matrix_legs(wf["jobs"]["test"]["strategy"]["matrix"])
        floor = min(
            (int(v.split(".")[0]), int(v.split(".")[1]))
            for v in (leg["python-version"] for leg in legs)
        )
        project = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        assert project["project"]["requires-python"] == f">={floor[0]}.{floor[1]}"

    def test_triggers_are_push_and_pull_request_to_main_and_dispatch(self):
        on = _load(_real_text())["on"]
        assert on["push"] == {"branches": ["main"]}
        assert on["pull_request"] == {"branches": ["main"]}
        assert "workflow_dispatch" in on
        assert set(on) == {"push", "pull_request", "workflow_dispatch"}

    def test_the_token_is_read_only_except_the_dependency_review_comment(self):
        wf = _load(_real_text())
        assert wf["permissions"] == {"contents": "read"}
        for name, job in wf["jobs"].items():
            if name == "dependency-review":
                assert job["permissions"] == _REVIEW_PERMISSIONS
            else:
                assert "permissions" not in job, name
        code = "\n".join(_code_lines(_real_text()))
        assert "id-token" not in code and "secrets." not in code
        assert "pull_request_target" not in code
        assert len(re.findall(r":\s*write\b", code)) == 1

    def test_runners_are_pinned_never_latest(self):
        wf = _load(_real_text())
        assert wf["jobs"]["lint"]["runs-on"] == _LINUX
        assert wf["jobs"]["test"]["runs-on"] == "${{ matrix.os }}"
        assert "-latest" not in "\n".join(_code_lines(_real_text()))
        legs = matrix_legs(wf["jobs"]["test"]["strategy"]["matrix"])
        assert {leg["os"] for leg in legs} == {_LINUX, "macos-15", "windows-2025"}

    def test_the_timeout_is_one_plain_number_long_enough_for_macos_and_windows(self):
        wf = _load(_real_text())
        timeout = wf["jobs"]["test"]["timeout-minutes"]
        assert isinstance(timeout, int) and timeout >= 25
        assert isinstance(wf["jobs"]["lint"]["timeout-minutes"], int)

    def test_utf8_mode_is_on_for_every_leg(self):
        # Windows 3.11 to 3.14 default to cp1252 (PEP 686 only from 3.15): D-018.
        assert _load(_real_text())["jobs"]["test"]["env"]["PYTHONUTF8"] == "1"

    def test_the_default_shell_is_bash_so_windows_legs_run_the_same_scripts(self):
        assert _load(_real_text())["defaults"]["run"]["shell"] == "bash"

    def test_every_action_is_a_sha_pin_with_its_release_tag(self):
        matches = list(_USES_LINE.finditer(_real_text()))
        # checkout and setup-python in three jobs, checkout and the review in the fourth
        assert len(matches) == 8
        for match in matches:
            assert _SHA_USES.match(match["ref"]), match["ref"]
            assert _TAG_COMMENT.match(match["rest"]), match["rest"]

    def test_one_action_has_one_pin_in_this_file_and_the_same_pin_as_docs_yml(self):
        def pins(text: str) -> dict[str, set[str]]:
            found: dict[str, set[str]] = {}
            for match in _USES_LINE.finditer(text):
                action, _, sha = match["ref"].partition("@")
                found.setdefault(action, set()).add(sha + match["rest"].strip())
            return found

        here = pins(_real_text())
        docs = pins((_ROOT / ".github" / "workflows" / "docs.yml").read_text("utf-8"))
        assert {a: len(v) for a, v in here.items()} == {
            "actions/checkout": 1,
            "actions/setup-python": 1,
            "actions/dependency-review-action": 1,
        }
        assert {a: v for a, v in here.items() if a in docs} == docs

    def test_checkouts_drop_their_credentials(self):
        # `guard` fetches `main` anonymously: the repository is public, so no token is needed.
        for job in _load(_real_text())["jobs"].values():
            checkout = _uses_steps(job, "actions/checkout")[0]
            assert checkout["with"]["persist-credentials"] is False

    def test_python_3_15_is_the_stable_release_not_a_prerelease(self):
        # actions/python-versions lists 3.15.2 stable for linux 24.04, darwin and win32 x64
        # (checked 2026-11-14), so no leg needs `allow-prereleases`.
        for job in _load(_real_text())["jobs"].values():
            for step in _uses_steps(job, "actions/setup-python"):
                assert "allow-prereleases" not in step["with"]

    def test_the_pip_cache_keys_on_the_pins(self):
        for job in _load(_real_text())["jobs"].values():
            for step in _uses_steps(job, "actions/setup-python"):
                assert step["with"]["cache"] == "pip"
                assert "constraints.txt" in step["with"]["cache-dependency-path"]

    def test_each_job_that_needs_the_package_has_one_editable_install_naming_dev(self):
        jobs = _load(_real_text())["jobs"]
        for name, job in jobs.items():
            if name == "dependency-review":
                continue  # reads the dependency graph through the API, installs nothing
            extras = _install_extras(job)
            assert len(extras) == 1, name
            assert "dev" in extras[0], name
        # `mcp` (beyond the package extras) makes the real stdio MCP tests run, not skip.
        assert "mcp" in _install_extras(jobs["test"])[0]

    def test_the_pth_audit_runs_after_the_install_in_the_lint_and_test_jobs(self):
        for name, job in _load(_real_text())["jobs"].items():
            if name not in ("lint", "test"):
                continue
            assert _runs(job).count(_AUDIT) == 1, name
            assert _index_of(job, _AUDIT) > _index_of(job, "pip install -c"), name

    def test_the_selection_is_the_documented_one_and_nothing_is_deselected(self):
        wf = _load(_real_text())
        assert wf["jobs"]["test"]["env"]["PYTEST_MARKERS"] == _MARKERS
        text = (_ROOT / "AGENTS.md").read_text(encoding="utf-8")
        bullet = next(ln for ln in text.splitlines() if ln.startswith("- CI ("))
        documented = re.search(r'pytest `-m "([^"]+)"` with no deselect', bullet)
        assert documented is not None, bullet
        assert documented[1] == _MARKERS
        assert "--deselect" not in "\n".join(_code_lines(_real_text()))

    def test_the_two_pytest_steps_split_the_legs_so_each_runs_once(self):
        test = _load(_real_text())["jobs"]["test"]
        shapes = {
            str(s["run"]).strip(): s["if"]
            for s in _steps(test)
            if "pytest" in str(s.get("run", ""))
        }
        assert shapes == {_PYTEST_PLAIN: _IF_PLAIN, _PYTEST_COVERAGE: _IF_COVERAGE}
        # the coverage form is the plain form plus the flags: one selection, written once
        assert _PYTEST_COVERAGE.startswith(_PYTEST_PLAIN)

    def test_only_the_canonical_leg_measures_coverage(self):
        wf = _load(_real_text())
        legs = matrix_legs(wf["jobs"]["test"]["strategy"]["matrix"])
        flagged = [leg for leg in legs if leg.get("coverage")]
        assert [_pairs([leg])[0] for leg in flagged] == [_CANONICAL_LEG]
        # the leg the baseline is measured on (scripts/coverage_ratchet.py CANONICAL)
        baseline = _ROOT / "coverage-baseline.json"
        assert "linux / 3.12" in baseline.read_text(encoding="utf-8")

    def test_the_ratchet_checks_after_the_run_and_guards_pull_requests_only(self):
        test = _load(_real_text())["jobs"]["test"]
        assert _step_running(test, _RATCHET_CHECK)["if"] == _IF_COVERAGE
        guard = _step_running(test, "coverage_ratchet.py guard")
        assert guard["if"] == _IF_GUARD
        assert "git fetch" in guard["run"]
        assert _index_of(test, _RATCHET_CHECK) > _index_of(test, "--cov-report=json")

    def test_mypy_and_the_import_contracts_run_once_in_the_lint_job_only(self):
        wf = _load(_real_text())
        lint, test = _runs(wf["jobs"]["lint"]), _runs(wf["jobs"]["test"])
        assert lint.count("python -m mypy -p fsm_llm") == 1
        assert _IMPORT_CONTRACT in lint
        assert not [r for r in test if "mypy" in r or "ruff" in r]

    def test_the_decision_gate_and_its_summary_run_in_the_lint_job(self):
        lint = _load(_real_text())["jobs"]["lint"]
        assert (
            "python scripts/decision_index.py --check docs/decisions/index.md"
            in _runs(lint)
        )
        summary = _step_running(lint, "decision_index.py --report")
        assert summary is not None
        assert "GITHUB_STEP_SUMMARY" in summary["run"]

    def test_the_four_jobs_run_in_parallel(self):
        for job in _load(_real_text())["jobs"].values():
            assert "needs" not in job

    def test_the_docs_build_stays_in_docs_yml(self):
        assert "mkdocs" not in _real_text()

    def test_no_event_field_is_interpolated_into_a_script_and_scripts_fail_fast(self):
        for job in _load(_real_text())["jobs"].values():
            for step in _steps(job):
                run = str(step.get("run", ""))
                assert "${{" not in run, step.get("name")
                if "\n" in run.strip():
                    assert "set -euo pipefail" in run, step.get("name")

    def test_a_push_never_cancels_a_run_and_a_superseded_pull_request_run_is_cancelled(
        self,
    ):
        wf = _load(_real_text())
        assert wf["concurrency"] == {
            "group": _CONCURRENCY_GROUP,
            "cancel-in-progress": _CONCURRENCY_CANCEL,
        }

    def test_the_lint_job_runs_the_two_workflow_linters_after_installing_their_pins(
        self,
    ):
        lint = _load(_real_text())["jobs"]["lint"]
        runs = _runs(lint)
        assert runs.count(_LINTERS_INSTALL) == 1
        assert runs.count(_ZIZMOR) == 1 and runs.count(_ACTIONLINT) == 1
        assert runs.index(_LINTERS_INSTALL) < runs.index(_ZIZMOR)
        assert runs.index(_LINTERS_INSTALL) < runs.index(_ACTIONLINT)

    def test_the_old_workflow_text_breaks_the_rules(self):
        # The parent state: one three-leg job on ubuntu-latest with floating tags.
        old = (
            "name: Python package\non:\n  push:\n    branches: [ 'main' ]\n"
            "jobs:\n  test:\n    runs-on: ubuntu-latest\n    steps:\n"
            "    - uses: actions/checkout@v5\n"
        )
        problems = workflow_problems(old)
        assert {"matrix-legs", "runner", "action-pin", "top-permissions"} <= problems


# --------------------------------------------------------------------------------------
# Mutants: the real text with one edit each; every rule must catch its own
# --------------------------------------------------------------------------------------


def _edit(old: str, new: str, count: int = 1) -> Any:
    def mutate(text: str) -> str:
        assert text.count(old) >= count, f"{old!r} not found {count} time(s)"
        return text.replace(old, new, count)

    return mutate


def _edit_nth(old: str, new: str, nth: int) -> Any:
    """Replace only the `nth` (0-based) occurrence of `old`."""

    def mutate(text: str) -> str:
        parts = text.split(old)
        assert len(parts) > nth + 1, f"{old!r} has no occurrence number {nth}"
        return old.join(parts[: nth + 1]) + new + old.join(parts[nth + 1 :])

    return mutate


def _sub(pattern: str, repl: str, flags: int = re.MULTILINE) -> Any:
    def mutate(text: str) -> str:
        mutated, n = re.subn(pattern, repl, text, count=1, flags=flags)
        assert n == 1, f"{pattern!r} not found"
        return mutated

    return mutate


def _drop_step(name: str, nth: int = 0) -> Any:
    """Remove the `nth` step whose `- name:` line is exactly `name` (to the next step)."""

    def mutate(text: str) -> str:
        pattern = re.compile(
            rf"^      - name: {re.escape(name)}\n(?:        .*\n|\n)*?(?=      - name:|\Z|\n  \w)",
            re.MULTILINE,
        )
        matches = list(pattern.finditer(text))
        assert len(matches) > nth, f"step {name!r} not found"
        m = matches[nth]
        return text[: m.start()] + text[m.end() :]

    return mutate


def _drop_job(job_id: str) -> Any:
    """Remove the job `job_id` (to the next job or the end of the file)."""

    def mutate(text: str) -> str:
        pattern = re.compile(
            rf"^  {re.escape(job_id)}:\n(?:    .*\n|\n)*?(?=  [\w-]+:\n|\Z)",
            re.MULTILINE,
        )
        m = pattern.search(text)
        assert m, f"job {job_id!r} not found"
        return text[: m.start()] + text[m.end() :]

    return mutate


MUTANTS: list[tuple[str, Any, str]] = [
    # the matrix
    (
        "a leg removed (Linux 3.13)",
        _sub(r'^ *- \{os: ubuntu-24\.04, python-version: "3\.13"\}\n', ""),
        "matrix-legs",
    ),
    (
        "a leg added (Linux 3.10)",
        _sub(
            r'^( *)(- \{os: ubuntu-24\.04, python-version: )"3\.11"\}\n',
            r'\1\2"3.10"}\n\1\2"3.11"}\n',
        ),
        "matrix-legs",
    ),
    (
        "a leg added (Windows 3.12)",
        _sub(
            r'^( *)(- \{os: windows-2025, python-version: )"3\.11"\}\n',
            r'\1\2"3.12"}\n\1\2"3.11"}\n',
        ),
        "matrix-legs",
    ),
    (
        "a duplicated leg",
        _sub(
            r'^( *)(- \{os: macos-15, python-version: "3\.11"\}\n)',
            r"\1\2\1\2",
        ),
        "matrix-legs",
    ),
    (
        "macOS 3.15 swapped for 3.14",
        _sub(r'(os: macos-15, python-version: )"3\.15"', r'\1"3.14"'),
        "matrix-legs",
    ),
    (
        "Windows moved to ubuntu",
        _edit("os: windows-2025", "os: ubuntu-24.04", 2),
        "matrix-legs",
    ),
    (
        "fail-fast removed (default true)",
        _edit("      fail-fast: false\n", ""),
        "fail-fast",
    ),
    ("fail-fast true", _edit("fail-fast: false", "fail-fast: true"), "fail-fast"),
    (
        "3.15 unquoted (a YAML float)",
        _edit('python-version: "3.15"', "python-version: 3.15", 3),
        "python-version-type",
    ),
    (
        "coverage on a second leg",
        _sub(
            r'(os: ubuntu-24\.04, python-version: "3\.13")\}',
            r"\1, coverage: true}",
        ),
        "coverage-legs",
    ),
    (
        "coverage on the 3.11 leg instead of 3.12",
        _sub(
            r'(os: ubuntu-24\.04, python-version: "3\.11")\}',
            r"\1, coverage: true}",
        ),
        "coverage-legs",
    ),
    # permissions, triggers
    (
        "top permissions widened",
        _edit("permissions:\n  contents: read\n", "permissions:\n  contents: write\n"),
        "top-permissions",
    ),
    (
        "top permissions removed",
        _edit("permissions:\n  contents: read\n\n", ""),
        "top-permissions",
    ),
    (
        "a job-level permission block",
        _edit(
            "    runs-on: ${{ matrix.os }}\n",
            "    permissions:\n      contents: read\n    runs-on: ${{ matrix.os }}\n",
        ),
        "job-permissions",
    ),
    (
        "id-token granted",
        _edit("  contents: read\n", "  contents: read\n  id-token: write\n"),
        "extra-permission",
    ),
    (
        "pull_request_target added",
        _edit(
            "  workflow_dispatch:\n", "  workflow_dispatch:\n  pull_request_target:\n"
        ),
        "pull-request-target",
    ),
    (
        "dispatch trigger removed",
        _edit("  workflow_dispatch:\n", ""),
        "dispatch-trigger",
    ),
    (
        "push trigger without its branch filter",
        _sub(r"^  push:\n    branches: \[.*\]\n", "  push:\n"),
        "push-trigger",
    ),
    (
        "a secret read",
        _edit("PYTHONUTF8", "TOKEN: ${{ secrets.TOKEN }}\n      PYTHONUTF8"),
        "secrets",
    ),
    # actions
    (
        "an action pinned to a tag",
        _sub(r"actions/checkout@[0-9a-f]{40}", "actions/checkout@v7"),
        "action-pin",
    ),
    (
        "an action pin truncated",
        _sub(r"(actions/setup-python@[0-9a-f]{39})[0-9a-f]", r"\1"),
        "action-pin",
    ),
    (
        "an action pin without its tag comment",
        _sub(r"(actions/setup-python@[0-9a-f]{40})\s+# v[\d.]+", r"\1"),
        "action-tag-comment",
    ),
    (
        "a third action",
        _edit(
            "    steps:\n",
            "    steps:\n      - uses: actions/cache@0000000000000000000000000000000000000000 # v4.0.0\n",
        ),
        "action-set",
    ),
    (
        "checkout persists credentials",
        _edit_nth("persist-credentials: false", "persist-credentials: true", 1),
        "persist-credentials",
    ),
    (
        "checkout without persist-credentials",
        _edit_nth("          persist-credentials: false\n", "", 0),
        "persist-credentials",
    ),
    (
        "setup-python ignores the matrix",
        _edit("python-version: ${{ matrix.python-version }}", 'python-version: "3.12"'),
        "setup-python-version",
    ),
    (
        "no pip cache on the matrix job",
        _edit_nth("          cache: pip\n", "", 1),
        "pip-cache",
    ),
    (
        "prereleases allowed",
        _edit_nth(
            "          cache: pip\n",
            "          cache: pip\n          allow-prereleases: true\n",
            1,
        ),
        "prereleases",
    ),
    # runners, timeouts, environment
    (
        "ubuntu-latest on the lint job",
        _edit("runs-on: ubuntu-24.04", "runs-on: ubuntu-latest"),
        "runner",
    ),
    (
        "macos-latest in the matrix",
        _edit("os: macos-15", "os: macos-latest", 2),
        "runner",
    ),
    (
        "the default job name (the canonical leg would end in true)",
        _edit(
            "    name: test (${{ matrix.os }}, Python ${{ matrix.python-version }})\n",
            "",
        ),
        "job-name",
    ),
    (
        "the matrix job pinned to one runner",
        _edit("runs-on: ${{ matrix.os }}", "runs-on: ubuntu-24.04"),
        "runner",
    ),
    (
        "timeout missing on the matrix job",
        _edit("    timeout-minutes: 45\n", ""),
        "timeout",
    ),
    (
        "timeout too short for macOS and Windows",
        _edit("    timeout-minutes: 45\n", "    timeout-minutes: 20\n"),
        "timeout",
    ),
    (
        "timeout an expression",
        _edit(
            "    timeout-minutes: 45\n",
            "    timeout-minutes: ${{ matrix.os == 'ubuntu-24.04' && 20 || 40 }}\n",
        ),
        "timeout",
    ),
    ("PYTHONUTF8 removed", _sub(r'^      PYTHONUTF8: "1"\n', ""), "pythonutf8"),
    ("PYTHONUTF8 off", _edit('PYTHONUTF8: "1"', 'PYTHONUTF8: "0"'), "pythonutf8"),
    (
        "default shell removed",
        _edit("defaults:\n  run:\n    shell: bash\n\n", ""),
        "default-shell",
    ),
    (
        "install without fail-fast shell options",
        _edit_nth("          set -euo pipefail\n", "", 1),
        "shell-strict",
    ),
    (
        "an event field pasted into a script",
        _edit(
            _PYTEST_PLAIN,
            _PYTEST_PLAIN + " ${{ github.head_ref }}",
        ),
        "run-interpolation",
    ),
    # what the matrix job runs
    (
        "a deselect added",
        _edit(_PYTEST_PLAIN, _PYTEST_PLAIN + " --deselect tests/x.py::t"),
        "deselect",
    ),
    (
        "a marker dropped from the selection",
        _edit(
            "not slow and not real_llm and not integration",
            "not slow and not real_llm",
        ),
        "markers",
    ),
    (
        "coverage form without the branch flag",
        _edit(" --cov --cov-branch --cov-report=json", " --cov --cov-report=json"),
        "pytest-steps",
    ),
    (
        "the plain pytest step runs on every leg",
        _edit("if: ${{ !matrix.coverage }}", "if: ${{ always() }}"),
        "pytest-steps",
    ),
    (
        "the pth audit removed from the matrix job",
        _drop_step("Audit for suspicious .pth files", 1),
        "audit-test",
    ),
    (
        "the pth audit removed from the lint job",
        _drop_step("Audit for suspicious .pth files", 0),
        "audit-lint",
    ),
    (
        "mcp missing from the matrix install",
        _edit_nth(",mcp]", "]", 1),
        "mcp",
    ),
    (
        "a second install line in the matrix job",
        _edit_nth(
            _AUDIT,
            'pip install -c constraints.txt -e ".[dev,mcp]"\n          ' + _AUDIT,
            1,
        ),
        "install",
    ),
    (
        "mypy added to the matrix job",
        _edit_nth(_AUDIT, _AUDIT + "\n          python -m mypy -p fsm_llm", 1),
        "mypy-in-test",
    ),
    (
        "the docs build added",
        _edit(_AUDIT, _AUDIT + "\n          mkdocs build --strict", 2),
        "docs-build-in-ci",
    ),
    # the ratchet
    (
        "the ratchet check removed",
        _drop_step("Check the coverage ratchet"),
        "ratchet-check",
    ),
    (
        "the ratchet check runs on every leg",
        _edit(
            "      - name: Check the coverage ratchet\n        if: ${{ matrix.coverage }}\n",
            "      - name: Check the coverage ratchet\n",
        ),
        "ratchet-check",
    ),
    (
        "the guard runs on pushes too",
        _edit(
            "if: ${{ matrix.coverage && github.event_name == 'pull_request' }}",
            "if: ${{ matrix.coverage }}",
        ),
        "ratchet-guard",
    ),
    (
        "the guard removed",
        _drop_step("Guard the coverage baseline against main"),
        "ratchet-guard",
    ),
    (
        "the guard without a fetch of main",
        _edit("          git fetch --no-tags --depth=1 origin", "          true", 1),
        "ratchet-guard-fetch",
    ),
    # the lint job
    ("ruff check removed", _drop_step("Lint with ruff"), "lint-ruff-check"),
    (
        "ruff format check removed",
        _drop_step("Check formatting with ruff"),
        "lint-ruff-format",
    ),
    ("mypy removed from the lint job", _drop_step("Type check with mypy"), "lint-mypy"),
    (
        "the import contracts removed",
        _drop_step("Check the import contracts"),
        "lint-import-contract",
    ),
    (
        "the decision gate removed",
        _drop_step("Check the decision index"),
        "lint-decision-check",
    ),
    (
        "the decision summary removed",
        _drop_step("Summarise the decision anchors"),
        "lint-decision-summary",
    ),
    # the supply-chain jobs, the concurrency group and the linters
    (
        "the concurrency group removed",
        _sub(r"^concurrency:\n(?:  .*\n)+\n", ""),
        "concurrency",
    ),
    (
        "a push cancels the run in progress too",
        _edit(
            "cancel-in-progress: ${{ github.event_name == 'pull_request' }}",
            "cancel-in-progress: true",
        ),
        "concurrency",
    ),
    (
        "the concurrency group shared by every workflow",
        _edit("group: ci-${{ github.workflow }}-${{ github.ref }}", "group: ci"),
        "concurrency",
    ),
    ("the supply-chain job removed", _drop_job("supply-chain"), "jobs"),
    ("the dependency review job removed", _drop_job("dependency-review"), "jobs"),
    (
        "a write scope on the supply-chain job",
        _edit(
            "  supply-chain:\n    runs-on: ubuntu-24.04\n",
            "  supply-chain:\n    permissions:\n      pull-requests: write\n"
            "    runs-on: ubuntu-24.04\n",
        ),
        "job-permissions",
    ),
    (
        "the dependency review without its write scope",
        _edit("      pull-requests: write\n", "      pull-requests: read\n"),
        "job-permissions",
    ),
    (
        "contents write on the dependency review",
        _edit_nth("      contents: read\n", "      contents: write\n", 0),
        "job-permissions",
    ),
    (
        "a second write scope on the dependency review",
        _edit(
            "      pull-requests: write\n",
            "      pull-requests: write\n      issues: write\n",
        ),
        "extra-permission",
    ),
    (
        "the dependency review waits for the matrix",
        _edit(
            "    if: ${{ github.event_name == 'pull_request' }}\n",
            "    needs: test\n    if: ${{ github.event_name == 'pull_request' }}\n",
        ),
        "job-needs",
    ),
    (
        "the dependency review action pinned to a tag",
        _sub(
            r"actions/dependency-review-action@[0-9a-f]{40}",
            "actions/dependency-review-action@v5",
        ),
        "action-pin",
    ),
    (
        "the dependency review action without its tag comment",
        _sub(r"(actions/dependency-review-action@[0-9a-f]{40})\s+# v[\d.]+", r"\1"),
        "action-tag-comment",
    ),
    (
        "the supply-chain checkout persists credentials",
        _edit_nth("persist-credentials: false", "persist-credentials: true", 2),
        "persist-credentials",
    ),
    (
        "the dependency review checkout persists credentials",
        _edit_nth("persist-credentials: false", "persist-credentials: true", 3),
        "persist-credentials",
    ),
    (
        "no pip cache on the supply-chain job",
        _edit_nth("          cache: pip\n", "", 2),
        "pip-cache",
    ),
    (
        "the supply-chain install removed",
        _drop_step("Install the package and every extra", 2),
        "install",
    ),
    (
        "the linters install removed",
        _drop_step("Install the workflow linters at their pinned versions"),
        "lint-linters-install",
    ),
    (
        "zizmor removed from the lint job",
        _drop_step("Audit the workflows with zizmor"),
        "lint-zizmor",
    ),
    (
        "actionlint removed from the lint job",
        _drop_step("Lint the workflows with actionlint"),
        "lint-actionlint",
    ),
    # jobs
    (
        "the matrix job waits for lint",
        _edit(
            "    runs-on: ${{ matrix.os }}\n",
            "    needs: lint\n    runs-on: ${{ matrix.os }}\n",
        ),
        "job-needs",
    ),
    (
        "a third job",
        _edit(
            "\njobs:\n", "\njobs:\n  extra:\n    runs-on: ubuntu-24.04\n    steps: []\n"
        ),
        "jobs",
    ),
    (
        "a schedule trigger added",
        _edit(
            "  workflow_dispatch:\n",
            "  workflow_dispatch:\n  schedule:\n    - cron: '0 3 * * 1'\n",
        ),
        "extra-trigger",
    ),
    (
        "the pull request trigger without its branch filter",
        _sub(r"^  pull_request:\n    branches: \[.*\]\n", "  pull_request:\n"),
        "pull-request-trigger",
    ),
    (
        "a uses line hidden in a script",
        _edit(
            "          python -m pip install --upgrade pip\n",
            "          python -m pip install --upgrade pip\n"
            "          uses: actions/checkout@v7\n",
        ),
        "uses-count",
    ),
    (
        "a second setup-python step in the matrix job",
        _edit_nth(
            "      - name: Install the package and every extra\n",
            "      - uses: actions/setup-python@6d4fd22b20465ccc0f7354476588e8df7c9ca0e0 # v7.1.0\n"
            "      - name: Install the package and every extra\n",
            1,
        ),
        "setup-python",
    ),
    ("an unreadable file", lambda t: "jobs: [unterminated", "unreadable"),
]


_PRODUCT_FORM = """      matrix:
        os: [ubuntu-24.04, macos-15, windows-2025]
        python-version: ["3.11", "3.12", "3.13", "3.14", "3.15"]
        exclude:
          - {os: macos-15, python-version: "3.12"}
          - {os: macos-15, python-version: "3.13"}
          - {os: macos-15, python-version: "3.14"}
          - {os: windows-2025, python-version: "3.12"}
          - {os: windows-2025, python-version: "3.13"}
          - {os: windows-2025, python-version: "3.14"}
        include:
          - {os: ubuntu-24.04, python-version: "3.12", coverage: true}
"""


class TestTheMatrixMayBeWrittenEitherWay:
    def test_a_product_with_excludes_is_the_same_nine_legs_and_breaks_no_rule(self):
        text = _real_text()
        start = text.index("      matrix:\n")
        end = text.index("    steps:\n", start)
        rewritten = text[:start] + _PRODUCT_FORM + text[end:]
        assert rewritten != text
        legs = matrix_legs(_load(rewritten)["jobs"]["test"]["strategy"]["matrix"])
        assert set(_pairs(legs)) == _EXPECTED_LEGS and len(legs) == 9
        assert workflow_problems(rewritten) == set()

    def test_the_product_form_with_one_exclude_too_few_has_ten_legs_and_is_refused(
        self,
    ):
        text = _real_text()
        start = text.index("      matrix:\n")
        end = text.index("    steps:\n", start)
        short = _PRODUCT_FORM.replace(
            '          - {os: windows-2025, python-version: "3.14"}\n', ""
        )
        rewritten = text[:start] + short + text[end:]
        assert "matrix-legs" in workflow_problems(rewritten)


class TestMutantsAreKilled:
    @pytest.mark.parametrize("label,mutate,code", MUTANTS, ids=[m[0] for m in MUTANTS])
    def test_each_mutant_is_caught_by_its_rule(
        self, label: str, mutate: Any, code: str
    ):
        mutated = mutate(_real_text())
        assert mutated != _real_text()
        assert code in workflow_problems(mutated), (label, workflow_problems(mutated))

    def test_there_are_at_least_twelve_distinct_rules_with_a_mutant(self):
        assert len({m[2] for m in MUTANTS}) >= 12

    def test_every_rule_code_the_checker_can_return_has_a_mutant(self):
        # A rule with no mutant is a rule nobody has seen fail. The scan reads the checker's
        # own source for its `found.add("...")` codes.
        source = Path(__file__).read_text(encoding="utf-8")
        body = source.split("def workflow_problems", 1)[1].split("def _real_text", 1)[0]
        codes = set(re.findall(r'found\.add\("([a-z-]+)"\)', body))
        codes |= {"audit-lint", "audit-test"} | {
            "lint-" + x
            for x in (
                "ruff-check",
                "ruff-format",
                "mypy",
                "import-contract",
                "decision-check",
                "linters-install",
                "zizmor",
                "actionlint",
            )
        }
        killed = {m[2] for m in MUTANTS}
        # `triggers` (a non-mapping `on`) guards the reader itself and has no text-level mutant
        # that keeps the YAML readable.
        assert codes - killed <= {"triggers"}, sorted(codes - killed)
