"""The scheduled pin-refresh workflow gates a pull request on the suite and holds one write scope.

Plan d454fd62, step 110 (roadmap row 0.10). `.github/workflows/refresh-pins.yml` runs weekly,
asks `scripts/refresh_pins.py` for newer litellm and mcp pins (7-day cooldown), and when a pin
moves it installs the new pins, runs the non-slow suite and the audits INSIDE the job, and only
then pushes a branch and opens a pull request. A pull request opened with the default token does
not trigger workflows, so the CI workflow never gates it: the order of the steps in this file is
the gate. None of it runs on a developer machine, so it is read here as data.

* `workflow_problems(text)` is a pure function of the workflow text and returns a set of problem
  codes. The real workflow must return the empty set; each of the mutants of `MUTANTS` is the
  real text with one edit and must return its own code: a check that cannot fail is not a check.
* The job has the one write scope this file allows (`contents` to push the branch,
  `pull-requests` to open the pull request); `test_supply_chain.py` registers it in
  `_WRITE_ALLOWED` and reads this file for the SHA pins.
* The install line, the pytest selection and the mypy command are COMPARED with
  `python-package.yml` (read as data), never copied, so the two workflows cannot drift apart.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOWS = _ROOT / ".github" / "workflows"
_WORKFLOW = _WORKFLOWS / "refresh-pins.yml"
_CI = _WORKFLOWS / "python-package.yml"

_SHA_USES = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+@[0-9a-f]{40}$")
_USES_LINE = re.compile(r"^\s*(?:-\s*)?uses:\s*(?P<ref>\S+)(?P<rest>.*)$", re.MULTILINE)
_TAG_COMMENT = re.compile(r"^\s*#\s*v\d+(?:\.\d+)+\s*$")
_SECRET = re.compile(r"secrets\.([A-Za-z0-9_]+)")
_GUARD = "steps.refresh.outputs.changed == 'true'"
_PYTEST_RUN = 'python -m pytest tests/ -q -m "$PYTEST_MARKERS"'
_MARKERS = "not slow and not real_llm and not integration"
_COMMIT_MESSAGE = "chore(deps): refresh litellm and mcp pins"
_WRITE_SCOPES = {"contents": "write", "pull-requests": "write"}


# --------------------------------------------------------------------------------------
# Reading the workflows
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


def _code_lines(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]


def _find(job: dict[str, Any], needle: str) -> int:
    """Index of the first step whose `run` or `uses` contains `needle`, or 10_000."""
    for index, step in enumerate(_steps(job)):
        if needle in str(step.get("run", "")) or needle in str(step.get("uses", "")):
            return index
    return 10_000


def _ci_reference() -> dict[str, str]:
    """The install line, the marker selection and the mypy command of the CI workflow."""
    ci = _load(_CI.read_text(encoding="utf-8"))
    lint, test = ci["jobs"]["lint"], ci["jobs"]["test"]
    install = next(
        line.strip()
        for step in _steps(lint)
        for line in str(step.get("run", "")).splitlines()
        if line.strip().startswith("pip install -c constraints.txt -e ")
    )
    plain = next(
        str(s["run"]).strip()
        for s in _steps(test)
        if "pytest" in str(s.get("run", "")) and "--cov" not in str(s["run"])
    )
    return {
        "install": install,
        "markers": str(test["env"]["PYTEST_MARKERS"]),
        "pytest": plain,
        "mypy": "python -m mypy -p fsm_llm",
    }


def cron_problem(cron: str) -> str | None:
    """Why `cron` is not a valid weekly 5-field expression at a non-round minute, or None."""
    fields = cron.split()
    if len(fields) != 5:
        return "not five fields"
    minute, hour, day, month, weekday = fields
    if not (minute.isdigit() and 0 <= int(minute) <= 59):
        return "minute is not a number"
    if not (hour.isdigit() and 0 <= int(hour) <= 23):
        return "hour is not a number"
    if day != "*" or month != "*":
        return "not every day of every month"
    if not (weekday.isdigit() and 0 <= int(weekday) <= 6):
        return "weekday is not one day"
    if int(minute) in (0, 15, 30, 45):
        return "round minute"
    return None


def workflow_problems(text: str) -> set[str]:
    """Return the codes of every rule the workflow text breaks (empty set: it is sound)."""
    found: set[str] = set()
    try:
        wf = _load(text)
        reference = _ci_reference()
    except Exception:
        return {"unreadable"}
    jobs = wf.get("jobs") or {}
    code = "\n".join(_code_lines(text))

    # Triggers: weekly schedule and manual dispatch, nothing a contributor can fire.
    on = wf.get("on") or {}
    if not isinstance(on, dict) or set(on) != {"schedule", "workflow_dispatch"}:
        found.add("triggers")
    else:
        schedule = on.get("schedule")
        if (
            not isinstance(schedule, list)
            or len(schedule) != 1
            or cron_problem(str((schedule[0] or {}).get("cron", ""))) is not None
        ):
            found.add("cron")
    if "pull_request_target" in code:
        found.add("pull-request-target")

    # Permissions: read at the top, the two write scopes on the one job.
    if wf.get("permissions") != {"contents": "read"}:
        found.add("top-permissions")
    if set(jobs) != {"refresh"}:
        found.add("jobs")
    job = jobs.get("refresh") or {}
    if job.get("permissions") != _WRITE_SCOPES:
        found.add("job-permissions")
    if re.search(r"\bid-token\b|\bpackages:\s*write|\bactions:\s*write", code):
        found.add("extra-permission")
    if "github.repository" not in str(job.get("if", "")):
        found.add("fork-guard")

    # Runner, timeout, shell, concurrency.
    if job.get("runs-on") != "ubuntu-24.04" or "-latest" in code:
        found.add("runner")
    timeout = job.get("timeout-minutes")
    if not isinstance(timeout, int) or not 60 <= timeout <= 180:
        found.add("timeout")
    if (wf.get("defaults") or {}).get("run", {}).get("shell") != "bash":
        found.add("default-shell")
    if wf.get("concurrency") != {"group": "refresh-pins", "cancel-in-progress": False}:
        found.add("concurrency")

    # Scripts: strict shell, no event data in a script, no swallowed failure.
    steps = _steps(job)
    for step in steps:
        run = step.get("run")
        if (
            isinstance(run, str)
            and "\n" in run.strip()
            and "set -euo pipefail" not in run
        ):
            found.add("shell-strict")
        if isinstance(run, str) and "${{" in run:
            found.add("run-interpolation")
        if step.get("continue-on-error"):
            found.add("continue-on-error")
    if re.search(r"\|\|\s*true\b|\bset \+e\b", code):
        found.add("swallowed-failure")
    if re.search(r"github\.(head_ref|event\.|ref_name)", code):
        found.add("event-data")

    # Actions: SHA pins with a release tag, only the two we know.
    lines = list(_USES_LINE.finditer(text))
    parsed = [str(s["uses"]) for s in steps if "uses" in s]
    if len(lines) != len(parsed) or not parsed:
        found.add("uses-count")
    for match in lines:
        if not _SHA_USES.match(match["ref"]):
            found.add("action-pin")
        if not _TAG_COMMENT.match(match["rest"]):
            found.add("action-tag-comment")
    if {u.split("@")[0] for u in parsed} != {
        "actions/checkout",
        "actions/setup-python",
    }:
        found.add("action-set")

    # Secrets: the default token only.
    if set(_SECRET.findall(code)) != {"GITHUB_TOKEN"}:
        found.add("secrets")

    # Checkout keeps its credentials (it pushes), and says why; Python 3.12.
    checkout = next(
        (s for s in steps if str(s.get("uses", "")).startswith("actions/checkout@")), {}
    )
    if (checkout.get("with") or {}).get("persist-credentials") is not True:
        found.add("checkout-credentials")
    elif (
        "git push"
        not in text.split("persist-credentials: true")[0].rsplit("- name:", 1)[-1]
    ):
        found.add("checkout-credentials-reason")
    setup = next(
        (
            s
            for s in steps
            if str(s.get("uses", "")).startswith("actions/setup-python@")
        ),
        {},
    )
    if str((setup.get("with") or {}).get("python-version")) != "3.12":
        found.add("python-version")

    # The steps and their order: refresh, install, audits, mypy, pytest, then the pull request.
    refresh = _find(job, "scripts/refresh_pins.py")
    install = _find(job, "pip install -c constraints.txt -e ")
    pth = _find(job, "scripts/audit_pth.py")
    audit = _find(job, "pip-audit --skip-editable")
    mypy = _find(job, reference["mypy"])
    pytest_at = _find(job, "pytest")
    pull_request = _find(job, "gh pr create")
    push = _find(job, "git push")
    if 10_000 in (refresh, install, pth, audit, mypy, pytest_at, pull_request, push):
        found.add("steps-missing")
    gates = (install, pth, audit, mypy, pytest_at)
    if not refresh < install < pytest_at < pull_request:
        found.add("order")
    if any(not refresh < gate < pull_request for gate in gates):
        found.add("gate-order")
    if push != pull_request:
        # the push and the pull request are one step, so the branch is never pushed alone
        found.add("push-order")

    # The refresh step: its output feeds the guard, and a missing `changed=` line must fail.
    step = steps[refresh] if refresh < len(steps) else {}
    run = str(step.get("run", ""))
    if (
        step.get("id") != "refresh"
        or "$GITHUB_OUTPUT" not in run
        or "changed=" not in run
    ):
        found.add("refresh-output")
    if "if" in step:
        found.add("refresh-guard")
    if "grep -E '^changed=(true|false)$'" not in run or "pipefail" not in run:
        found.add("refresh-fail-closed")

    # Every step after the refresh runs only when a pin changed.
    for index, later in enumerate(steps):
        if index > refresh and later.get("if") != _GUARD:
            found.add("guard")

    # The gates are the documented ones, no test deselected.
    if pytest_at < len(steps):
        pytest_run = str(steps[pytest_at].get("run", "")).strip()
        if pytest_run != _PYTEST_RUN or pytest_run != reference["pytest"]:
            found.add("pytest-line")
    env = job.get("env") or {}
    if (
        env.get("PYTEST_MARKERS") != _MARKERS
        or env.get("PYTEST_MARKERS") != reference["markers"]
    ):
        found.add("pytest-selection")
    if re.search(r"--deselect|--ignore|\s-k\s|\s-x\b|--lf\b|--maxfail", code):
        found.add("pytest-deselect")
    if install < len(steps):
        install_lines = [
            x.strip() for x in str(steps[install].get("run", "")).splitlines()
        ]
        if reference["install"] not in install_lines:
            found.add("install-line")
    if (
        mypy < len(steps)
        and str(steps[mypy].get("run", "")).strip() != reference["mypy"]
    ):
        found.add("mypy-line")

    # The pull request step.
    if pull_request < len(steps):
        pr = steps[pull_request]
        script = str(pr.get("run", ""))
        if (pr.get("env") or {}).get("GH_TOKEN") != "${{ secrets.GITHUB_TOKEN }}":
            found.add("pr-token")
        needed = (
            'branch="chore/refresh-pins-$(date -u +%Y-%m-%d-%H%M%S)"',
            'git config user.name "github-actions[bot]"',
            "41898282+github-actions[bot]@users.noreply.github.com",
            "git add constraints.txt pyproject.toml",
            f'git commit -m "{_COMMIT_MESSAGE}"',
            'git push origin "$branch"',
            "gh pr create --base main",
            '--head "$branch"',
            f'--title "{_COMMIT_MESSAGE}"',
            "--body-file",
        )
        if not all(token in script for token in needed):
            found.add("pr-shape")
        order = [script.find(t) for t in ("git commit", "git push", "gh pr create")]
        if min(order) < 0 or order != sorted(order):
            found.add("pr-order")
        if re.search(r"git add (-A|--all|\.)(\s|$)", script):
            found.add("pr-add-all")
    return found


def _real_text() -> str:
    return _WORKFLOW.read_text(encoding="utf-8")


# --------------------------------------------------------------------------------------
# The real workflow
# --------------------------------------------------------------------------------------


class TestTheRealWorkflow:
    def test_the_workflow_exists_and_reads_as_yaml(self):
        assert _WORKFLOW.is_file(), ".github/workflows/refresh-pins.yml is missing"
        assert isinstance(_load(_real_text()), dict)

    def test_it_breaks_no_rule(self):
        assert workflow_problems(_real_text()) == set()

    def test_triggers_are_a_weekly_schedule_and_dispatch_only(self):
        on = _load(_real_text())["on"]
        assert set(on) == {"schedule", "workflow_dispatch"}
        assert len(on["schedule"]) == 1
        assert cron_problem(on["schedule"][0]["cron"]) is None

    def test_the_token_is_read_only_except_on_the_one_job(self):
        wf = _load(_real_text())
        assert wf["permissions"] == {"contents": "read"}
        assert wf["jobs"]["refresh"]["permissions"] == _WRITE_SCOPES

    def test_the_gates_run_in_the_job_before_the_pull_request_is_opened(self):
        job = _load(_real_text())["jobs"]["refresh"]
        names = [s.get("name", "") for s in _steps(job)]
        order = [
            "Pick the newest eligible pins",
            "Install the package and every extra at the new pins",
            "Audit for suspicious .pth files",
            "Audit the installed environment for known vulnerabilities",
            "Type check with mypy",
            "Test with pytest",
            "Open the pull request",
        ]
        assert [n for n in names if n in order] == order
        assert names[:2] == ["Check out the repository", "Set up Python"]

    def test_every_step_after_the_refresh_is_guarded_by_its_output(self):
        job = _load(_real_text())["jobs"]["refresh"]
        steps = _steps(job)
        refresh = next(i for i, s in enumerate(steps) if s.get("id") == "refresh")
        assert "if" not in steps[refresh]
        assert [s["if"] for s in steps[refresh + 1 :]] == [_GUARD] * len(
            steps[refresh + 1 :]
        )

    def test_the_pytest_line_is_the_documented_selection_with_no_deselect(self):
        text = _real_text()
        job = _load(text)["jobs"]["refresh"]
        reference = _ci_reference()
        assert job["env"]["PYTEST_MARKERS"] == reference["markers"] == _MARKERS
        step = next(s for s in _steps(job) if "pytest" in str(s.get("run", "")))
        assert step["run"].strip() == _PYTEST_RUN == reference["pytest"]
        assert not re.search(
            r"--deselect|--ignore|\s-k\s", "\n".join(_code_lines(text))
        )

    def test_the_install_line_is_the_one_of_every_ci_job(self):
        job = _load(_real_text())["jobs"]["refresh"]
        run = next(
            s["run"] for s in _steps(job) if "pip install -c" in str(s.get("run", ""))
        )
        assert _ci_reference()["install"] in [ln.strip() for ln in run.splitlines()]

    def test_the_documented_selection_in_agents_md_is_the_same(self):
        agents = (_ROOT / "AGENTS.md").read_text(encoding="utf-8")
        assert '"not slow and not real_llm and not integration"' in agents

    def test_the_cron_is_weekly_at_a_non_round_minute(self):
        cron = _load(_real_text())["on"]["schedule"][0]["cron"]
        minute, hour, day, month, weekday = cron.split()
        assert (day, month) == ("*", "*") and weekday.isdigit()
        assert int(minute) % 15 != 0 and hour.isdigit()

    def test_the_branch_name_is_computed_from_the_date_not_from_an_event(self):
        text = _real_text()
        assert 'branch="chore/refresh-pins-$(date -u +%Y-%m-%d-%H%M%S)"' in text
        assert not re.search(r"github\.(head_ref|event\.|ref_name)", text)

    def test_no_script_interpolates_an_expression(self):
        for step in _steps(_load(_real_text())["jobs"]["refresh"]):
            assert "${{" not in str(step.get("run", "")), step.get("name")

    def test_the_only_secret_named_is_the_default_token(self):
        assert set(_SECRET.findall(_real_text())) == {"GITHUB_TOKEN"}

    def test_every_action_is_a_sha_pin_with_its_release_tag(self):
        matches = list(_USES_LINE.finditer(_real_text()))
        assert len(matches) == 2
        for match in matches:
            assert _SHA_USES.match(match["ref"]), match["ref"]
            assert _TAG_COMMENT.match(match["rest"]), match["rest"]

    def test_the_actions_carry_the_pins_of_the_other_workflows(self):
        ci = {m["ref"] for m in _USES_LINE.finditer(_CI.read_text(encoding="utf-8"))}
        mine = {m["ref"] for m in _USES_LINE.finditer(_real_text())}
        assert mine <= ci

    def test_the_checkout_keeps_its_credentials_and_says_why(self):
        text = _real_text()
        checkout = _steps(_load(text)["jobs"]["refresh"])[0]
        assert checkout["with"]["persist-credentials"] is True
        assert (
            "git push"
            in text.split("persist-credentials: true")[0].rsplit("- name:", 1)[-1]
        )

    def test_it_never_runs_on_a_pull_request_or_a_push(self):
        code = "\n".join(_code_lines(_real_text()))
        assert "pull_request" not in code
        assert not re.search(r"^\s*push:", code, re.MULTILINE)

    def test_a_run_never_cancels_one_in_progress(self):
        conc = _load(_real_text())["concurrency"]
        assert conc == {"group": "refresh-pins", "cancel-in-progress": False}

    def test_the_job_is_skipped_on_a_fork(self):
        job = _load(_real_text())["jobs"]["refresh"]
        assert job["if"] == "${{ github.repository == 'NikolasMarkou/fsm_llm' }}"

    def test_the_cron_checker_knows_valid_and_invalid_expressions(self):
        assert cron_problem("17 5 * * 1") is None
        assert cron_problem("23 4 * * 0") is None
        for bad in (
            "17 5 * * 1 *",
            "17 5 * *",
            "17 25 * * 1",
            "61 5 * * 1",
            "17 5 * * *",
            "17 5 1 * 1",
            "17 5 * 6 1",
            "17 5 * * 7",
            "17 5 * * 1-5",
            "*/5 * * * *",
            "0 5 * * 1",
            "30 5 * * 1",
            "",
        ):
            assert cron_problem(bad) is not None, bad


# --------------------------------------------------------------------------------------
# Mutants: the real text with one edit each; every rule must catch its own
# --------------------------------------------------------------------------------------


def _edit(old: str, new: str, nth: int = 0) -> Callable[[str], str]:
    """Replace the `nth` (0-based) occurrence of `old`; the anchor must exist."""

    def apply(text: str) -> str:
        parts = text.split(old)
        assert len(parts) > nth + 1, f"mutant anchor not in the workflow: {old!r}"
        return old.join(parts[: nth + 1]) + new + old.join(parts[nth + 1 :])

    return apply


def _step_block(text: str, name: str) -> tuple[int, int]:
    """Character range of the step called `name` (from its `- name:` line to the next step)."""
    start = text.index(f"      - name: {name}\n")
    nxt = text.find("\n      - name: ", start + 1)
    return start, (len(text) if nxt < 0 else nxt + 1)


def _drop_step(name: str) -> Callable[[str], str]:
    def apply(text: str) -> str:
        start, end = _step_block(text, name)
        return text[:start] + text[end:]

    return apply


def _move_step(name: str, before: str) -> Callable[[str], str]:
    def apply(text: str) -> str:
        start, end = _step_block(text, name)
        block = text[start:end]
        if not block.endswith("\n"):
            block += "\n"
        text = text[:start] + text[end:]
        at, _ = _step_block(text, before)
        return text[:at] + block + text[at:]

    return apply


_PR = "Open the pull request"
_TESTS = "Test with pytest"
_INSTALL = "Install the package and every extra at the new pins"
_PICK = "Pick the newest eligible pins"
_GUARD_LINE = "        if: steps.refresh.outputs.changed == 'true'\n"
_SHA_CHECKOUT = "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1"
_SHA_PYTHON = "actions/setup-python@6d4fd22b20465ccc0f7354476588e8df7c9ca0e0 # v7.1.0"

MUTANTS: list[tuple[str, Callable[[str], str], str]] = [
    ("the pull request is opened before the tests", _move_step(_PR, _TESTS), "order"),
    (
        "the pull request is opened before the install",
        _move_step(_PR, _INSTALL),
        "order",
    ),
    ("the install comes before the refresh", _move_step(_INSTALL, _PICK), "order"),
    (
        "the guard is removed from the pull request step",
        _edit(_GUARD_LINE, "", 5),
        "guard",
    ),
    ("the guard is removed from the install step", _edit(_GUARD_LINE, "", 0), "guard"),
    ("the guard is removed from the tests", _edit(_GUARD_LINE, "", 4), "guard"),
    (
        "the guard tests the wrong value",
        _edit("changed == 'true'", "changed == 'false'", 5),
        "guard",
    ),
    ("the tests are removed", _drop_step(_TESTS), "steps-missing"),
    ("mypy is removed", _drop_step("Type check with mypy"), "steps-missing"),
    (
        "pip-audit is removed",
        _drop_step("Audit the installed environment for known vulnerabilities"),
        "steps-missing",
    ),
    (
        "the pth audit is removed",
        _drop_step("Audit for suspicious .pth files"),
        "steps-missing",
    ),
    ("the install is removed", _drop_step(_INSTALL), "steps-missing"),
    (
        "a failing test is swallowed",
        _edit(_PYTEST_RUN, _PYTEST_RUN + " || true"),
        "swallowed-failure",
    ),
    (
        "the tests continue on error",
        _edit(
            "        run: " + _PYTEST_RUN + "\n",
            "        continue-on-error: true\n        run: " + _PYTEST_RUN + "\n",
        ),
        "continue-on-error",
    ),
    (
        "a test is deselected",
        _edit(_PYTEST_RUN, _PYTEST_RUN + " --deselect tests/test_packaging.py"),
        "pytest-deselect",
    ),
    (
        "a keyword filter is added",
        _edit(_PYTEST_RUN, _PYTEST_RUN + " -k mcp"),
        "pytest-deselect",
    ),
    (
        "the marker selection is narrowed",
        _edit(f"PYTEST_MARKERS: {_MARKERS}", "PYTEST_MARKERS: not slow"),
        "pytest-selection",
    ),
    (
        "the pytest command drops the markers",
        _edit(_PYTEST_RUN, "python -m pytest tests/ -q"),
        "pytest-line",
    ),
    (
        "the pytest command runs a subset",
        _edit(
            _PYTEST_RUN, 'python -m pytest tests/test_fsm_llm/ -q -m "$PYTEST_MARKERS"'
        ),
        "pytest-line",
    ),
    (
        "the install line drops mcp",
        _edit("testing,mcp]", "testing]"),
        "install-line",
    ),
    (
        "mypy runs on a subset",
        _edit("python -m mypy -p fsm_llm", "python -m mypy -p fsm_llm.core"),
        "mypy-line",
    ),
    (
        "a push trigger is added",
        _edit("\non:\n", '\non:\n  push:\n    branches: ["main"]\n'),
        "triggers",
    ),
    (
        "a pull request trigger is added",
        _edit("\non:\n", "\non:\n  pull_request:\n"),
        "triggers",
    ),
    (
        "the dispatch trigger is removed",
        _edit("  workflow_dispatch:\n", ""),
        "triggers",
    ),
    (
        "pull_request_target is named",
        _edit("\non:\n", "\non:\n  pull_request_target:\n"),
        "pull-request-target",
    ),
    ("the cron has six fields", _edit('"17 5 * * 1"', '"17 5 * * 1 *"'), "cron"),
    ("the cron hour is out of range", _edit('"17 5 * * 1"', '"17 25 * * 1"'), "cron"),
    ("the cron is daily", _edit('"17 5 * * 1"', '"17 5 * * *"'), "cron"),
    ("the cron is every five minutes", _edit('"17 5 * * 1"', '"*/5 * * * *"'), "cron"),
    ("the cron minute is round", _edit('"17 5 * * 1"', '"0 5 * * 1"'), "cron"),
    (
        "the schedule is removed",
        _edit(
            '  schedule:\n    # Mondays 05:17 UTC (a non-round minute: the top of the hour is when scheduled runs queue).\n    - cron: "17 5 * * 1"\n',
            "",
        ),
        "triggers",
    ),
    (
        "the top-level token can write",
        _edit(
            "permissions:\n  contents: read\n\nconcurrency",
            "permissions:\n  contents: write\n\nconcurrency",
        ),
        "top-permissions",
    ),
    (
        "the pull request scope is missing",
        _edit("      pull-requests: write\n", ""),
        "job-permissions",
    ),
    (
        "the write scope is widened",
        _edit(
            "      pull-requests: write\n",
            "      pull-requests: write\n      actions: write\n",
        ),
        "job-permissions",
    ),
    (
        "an id token is requested",
        _edit(
            "      pull-requests: write\n",
            "      pull-requests: write\n      id-token: write\n",
        ),
        "extra-permission",
    ),
    (
        "an action is on a floating tag",
        _edit(_SHA_CHECKOUT, "actions/checkout@v7"),
        "action-pin",
    ),
    (
        "an action pin has no release tag",
        _edit(_SHA_PYTHON, _SHA_PYTHON.replace(" # v7.1.0", "")),
        "action-tag-comment",
    ),
    (
        "an action pin is truncated",
        _edit("@6d4fd22b20465ccc0f7354476588e8df7c9ca0e0", "@6d4fd22b"),
        "action-pin",
    ),
    (
        "a third action is used",
        _edit(
            "      - name: Pick the newest eligible pins\n",
            "      - name: Extra\n        uses: actions/cache@0057852bfaa89a56745cba8c7296529d2fc39830 # v4.3.0\n      - name: Pick the newest eligible pins\n",
        ),
        "action-set",
    ),
    (
        "a second secret is named",
        _edit(
            "GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}",
            "GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}\n          OTHER: ${{ secrets.PAT }}",
        ),
        "secrets",
    ),
    (
        "a personal token replaces the default one",
        _edit("secrets.GITHUB_TOKEN", "secrets.PAT", 1),
        "secrets",
    ),
    (
        "an expression is pasted into a script",
        _edit(
            'branch="chore/refresh-pins-$(date -u +%Y-%m-%d-%H%M%S)"',
            'branch="chore/refresh-pins-${{ github.run_id }}"',
        ),
        "run-interpolation",
    ),
    (
        "the branch name comes from the event",
        _edit(
            '$(date -u +%Y-%m-%d-%H%M%S)"',
            '$GITHUB_HEAD_REF"\n          echo "${{ github.head_ref }}"',
        ),
        "event-data",
    ),
    (
        "the checkout drops its credentials",
        _edit("persist-credentials: true", "persist-credentials: false"),
        "checkout-credentials",
    ),
    (
        "the credentials are kept without saying why",
        _edit("(`git push`)", "(the remote)"),
        "checkout-credentials-reason",
    ),
    (
        "the runner floats",
        _edit("runs-on: ubuntu-24.04", "runs-on: ubuntu-latest"),
        "runner",
    ),
    (
        "the timeout is removed",
        _edit("    timeout-minutes: 90\n", ""),
        "timeout",
    ),
    (
        "the timeout is the platform default",
        _edit("timeout-minutes: 90", "timeout-minutes: 360"),
        "timeout",
    ),
    (
        "a second run cancels the first",
        _edit("cancel-in-progress: false", "cancel-in-progress: true"),
        "concurrency",
    ),
    (
        "the concurrency group is another",
        _edit("group: refresh-pins", "group: ${{ github.ref }}"),
        "concurrency",
    ),
    (
        "the default shell is removed",
        _edit("defaults:\n  run:\n    shell: bash\n\n", ""),
        "default-shell",
    ),
    (
        "the fork guard is removed",
        _edit("    if: ${{ github.repository == 'NikolasMarkou/fsm_llm' }}\n", ""),
        "fork-guard",
    ),
    (
        "a second job is added",
        lambda t: t + "  other:\n    runs-on: ubuntu-24.04\n    steps: []\n",
        "jobs",
    ),
    (
        "the python version is another",
        _edit('python-version: "3.12"', 'python-version: "3.11"'),
        "python-version",
    ),
    (
        "the output line is never written",
        _edit(' | tail -n 1 >> "$GITHUB_OUTPUT"', ""),
        "refresh-output",
    ),
    (
        "the changed line is not required",
        _edit("grep -E '^changed=(true|false)$' ", "grep -E 'changed=' "),
        "refresh-fail-closed",
    ),
    (
        "the refresh step has no id",
        _edit("        id: refresh\n", ""),
        "refresh-output",
    ),
    (
        "the pull request adds every file",
        _edit("git add constraints.txt pyproject.toml", "git add -A"),
        "pr-add-all",
    ),
    (
        "the push is removed",
        _edit('          git push origin "$branch"\n', "", 0),
        "steps-missing",
    ),
    (
        "the commit message is another",
        _edit(f'git commit -m "{_COMMIT_MESSAGE}"', 'git commit -m "update"'),
        "pr-shape",
    ),
    (
        "the pull request is opened against another base",
        _edit("gh pr create --base main", "gh pr create --base dev"),
        "pr-shape",
    ),
    (
        "the pull request has no body file",
        _edit("--body-file", "--body"),
        "pr-shape",
    ),
    (
        "the bot identity is a person",
        _edit(
            'git config user.name "github-actions[bot]"',
            'git config user.name "someone"',
        ),
        "pr-shape",
    ),
    (
        "the strict shell is removed from a script",
        _edit("          set -euo pipefail\n          branch=", "          branch=", 0),
        "shell-strict",
    ),
]


class TestMutantsAreKilled:
    @pytest.mark.parametrize("label,mutate,code", MUTANTS, ids=[m[0] for m in MUTANTS])
    def test_each_mutant_is_caught_by_its_rule(
        self, label: str, mutate: Any, code: str
    ):
        mutated = mutate(_real_text())
        assert mutated != _real_text(), label
        assert code in workflow_problems(mutated), (label, workflow_problems(mutated))

    def test_there_are_at_least_eight_mutants_and_many_rules(self):
        assert len(MUTANTS) >= 8
        assert len({m[2] for m in MUTANTS}) >= 25

    def test_the_real_text_is_the_only_clean_one(self):
        assert workflow_problems(_real_text()) == set()
        assert workflow_problems("") != set()
        assert workflow_problems("not: [valid") == {"unreadable"}
