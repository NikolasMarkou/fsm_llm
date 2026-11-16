"""The scheduled pin-refresh workflow tests on a read-only job and proposes from a write job.

Plan d454fd62, step 110 (roadmap row 0.10), split in two jobs at step 121.1 (final REFLECT
concern 2). `.github/workflows/refresh-pins.yml` runs weekly, asks `scripts/refresh_pins.py` for
newer litellm and mcp pins (7-day cooldown), and when a pin moves

* the `test` job (read-only token, checkout without credentials) stages and uploads the three
  files of the proposal BEFORE it installs anything, then installs the new pins and runs the
  non-slow suite and the audits (it executes brand-new third-party code, so it must hold no
  write scope and no credential);
* the `propose` job (`needs: test`, the only write scope) installs and runs nothing: it downloads
  the artifact, a guard step checks that it holds exactly the three expected regular files and
  that the tree then differs from `main` in pin lines only, and only then pushes a branch and
  opens the pull request.

A pull request opened with the default token does not trigger workflows, so the CI workflow
never gates it: the `test` job is the gate. None of it runs on a developer machine, so it is read
here as data, and the guard and pull request scripts are extracted from the YAML and run in bash
against scratch repositories (no network, a local bare repository stands in for the remote).

* `workflow_problems(text)` is a pure function of the workflow text and returns a set of problem
  codes. The real workflow must return the empty set; each of the mutants of `MUTANTS` is the
  real text with one edit and must return its own code: a check that cannot fail is not a check.
* `test_supply_chain.py` registers the one write job in `_WRITE_ALLOWED` and holds the rule that
  no write job runs a package installer or project code; this file repeats it for `propose`.
* The install line, the pytest selection and the mypy command are COMPARED with
  `python-package.yml` (read as data), never copied, so the two workflows cannot drift apart.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import symlink_or_skip
from tests.test_fsm_llm_regression.workflow_support import (
    _code_lines,
    _load,
    _steps,
)

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
_PROPOSE_IF = "${{ needs.test.outputs.changed == 'true' }}"
_ARTIFACT = "refresh-pins"
_ARTIFACT_PATH = "${{ runner.temp }}/pins"
_ACTIONS = {
    "actions/checkout",
    "actions/setup-python",
    "actions/upload-artifact",
    "actions/download-artifact",
}
_GUARD_STEP = "Check the proposed files and apply them"
_PR_STEP = "Open the pull request"
#: What a job that holds a write token must never run: installers, interpreters, test runners.
_CODE_COMMANDS = re.compile(
    r"^(?:pip[\d.]*|pipx|uv|uvx|poetry|conda|npm|npx|yarn|pnpm|cargo|gem|bundle|go|"
    r"python[\d.]*|pytest|mypy|ruff|tox|nox|make|bash|sh|zsh|node|source)$"
)
_WRAPPERS = frozenset(
    {"sudo", "env", "time", "nice", "ionice", "exec", "xargs", "command", "timeout"}
)
_KEYWORDS = frozenset({"if", "then", "else", "elif", "fi", "do", "done", "while", "!"})


# --------------------------------------------------------------------------------------
# Reading the workflows
# --------------------------------------------------------------------------------------


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


def commands_of(script: str) -> list[str]:
    """The command word of every simple command of a shell script (best effort).

    Heredoc bodies and single-quoted strings are dropped, double quotes are removed, and the
    script is cut at `; & | && || ( ) { } $(` and newlines. Leading keywords, wrappers
    (`sudo`, `env`, `nice`...) and `NAME=value` assignments are skipped. A command that is a
    path (`./x`, `scripts/x`) is reported with its path, since it runs project code.
    """
    text = re.sub(
        r"<<-?\s*['\"]?(?P<tag>\w+)['\"]?[^\n]*\n.*?\n\s*(?P=tag)[ \t]*(?:\n|$)",
        "\n",
        script,
        flags=re.DOTALL,
    )
    text = re.sub(r"'[^']*'", "''", text)
    text = re.sub(r'"(?:[^"\\$`]|\\.|\$(?!\())*"', '""', text).replace('"', "")
    text = re.sub(r"\\\n", " ", text)
    text = re.sub(r"\$\(|[`;&|(){}]|\n", "\n", text)
    found: list[str] = []
    for segment in text.split("\n"):
        words = segment.split()
        while words:
            if words[0] in _WRAPPERS:
                words = words[1:]
                while words and (words[0].startswith("-") or words[0].isdigit()):
                    words = words[1:]
            elif words[0] in _KEYWORDS or re.fullmatch(r"[A-Za-z_]\w*=\S*", words[0]):
                words = words[1:]
            else:
                break
        if words and not words[0].startswith("#"):
            found.append(words[0])
    return found


def runs_code(script: str) -> bool:
    """True when `script` names an installer, an interpreter or a project script."""
    for command in commands_of(script):
        if _CODE_COMMANDS.match(command) or command.startswith(("./", "scripts/")):
            return True
    return False


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

    # Permissions: read at the top and on `test`; the two write scopes on `propose` only.
    if wf.get("permissions") != {"contents": "read"}:
        found.add("top-permissions")
    if set(jobs) != {"test", "propose"}:
        found.add("jobs")
    test = jobs.get("test") or {}
    propose = jobs.get("propose") or {}
    if "permissions" in test:
        found.add("test-permissions")
    if propose.get("permissions") != _WRITE_SCOPES:
        found.add("propose-permissions")
    if re.search(r"\bid-token\b|\bpackages:\s*write|\bactions:\s*write", code):
        found.add("extra-permission")
    if "github.repository" not in str(test.get("if", "")):
        found.add("fork-guard")
    if propose.get("needs") not in ("test", ["test"]):
        found.add("propose-needs")
    if propose.get("if") != _PROPOSE_IF:
        found.add("propose-if")
    if test.get("outputs") != {"changed": "${{ steps.refresh.outputs.changed }}"}:
        found.add("test-output")

    # Runner, timeout, shell, concurrency.
    for job in (test, propose):
        if job.get("runs-on") != "ubuntu-24.04":
            found.add("runner")
    if "-latest" in code:
        found.add("runner")
    for job, low, high in ((test, 60, 180), (propose, 5, 60)):
        timeout = job.get("timeout-minutes")
        if not isinstance(timeout, int) or not low <= timeout <= high:
            found.add("timeout")
    if (wf.get("defaults") or {}).get("run", {}).get("shell") != "bash":
        found.add("default-shell")
    if wf.get("concurrency") != {"group": "refresh-pins", "cancel-in-progress": False}:
        found.add("concurrency")

    # Scripts: strict shell, no event data in a script, no swallowed failure.
    test_steps = _steps(test)
    propose_steps = _steps(propose)
    for step in (*test_steps, *propose_steps):
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

    # Actions: SHA pins with a release tag, only the four we know.
    lines = list(_USES_LINE.finditer(text))
    parsed = [str(s["uses"]) for s in (*test_steps, *propose_steps) if "uses" in s]
    if len(lines) != len(parsed) or not parsed:
        found.add("uses-count")
    for match in lines:
        if not _SHA_USES.match(match["ref"]):
            found.add("action-pin")
        if not _TAG_COMMENT.match(match["rest"]):
            found.add("action-tag-comment")
    if {u.split("@")[0] for u in parsed} != _ACTIONS:
        found.add("action-set")

    # Secrets: the default token only, and only in `propose`.
    if set(_SECRET.findall(code)) != {"GITHUB_TOKEN"}:
        found.add("secrets")
    if "secrets." in json.dumps(test):
        found.add("test-secret")

    # Checkouts: `test` keeps no credentials, `propose` keeps them and says why.
    def checkout_of(steps: list[dict[str, Any]]) -> dict[str, Any]:
        return next(
            (
                s
                for s in steps
                if str(s.get("uses", "")).startswith("actions/checkout@")
            ),
            {},
        )

    if (checkout_of(test_steps).get("with") or {}).get(
        "persist-credentials"
    ) is not False:
        found.add("test-credentials")
    if (checkout_of(propose_steps).get("with") or {}).get(
        "persist-credentials"
    ) is not True:
        found.add("propose-credentials")
    elif (
        "git push"
        not in text.split("persist-credentials: true")[0].rsplit("- name:", 1)[-1]
    ):
        found.add("propose-credentials-reason")
    setup = next(
        (
            s
            for s in test_steps
            if str(s.get("uses", "")).startswith("actions/setup-python@")
        ),
        {},
    )
    if str((setup.get("with") or {}).get("python-version")) != "3.12":
        found.add("python-version")

    # `propose` installs nothing and runs no project code.
    for step in propose_steps:
        if str(step.get("uses", "")).startswith("actions/setup-python@"):
            found.add("propose-runs-code")
        if isinstance(step.get("run"), str) and runs_code(step["run"]):
            found.add("propose-runs-code")

    # The steps of `test` and their order: refresh, stage, upload, install, audits, mypy, pytest.
    refresh = _find(test, "scripts/refresh_pins.py")
    stage = _find(test, "cp constraints.txt pyproject.toml")
    upload = _find(test, "actions/upload-artifact@")
    install = _find(test, "pip install -c constraints.txt -e ")
    pth = _find(test, "scripts/audit_pth.py")
    audit = _find(test, "pip-audit --skip-editable")
    mypy = _find(test, reference["mypy"])
    pytest_at = _find(test, "pytest")
    if 10_000 in (refresh, stage, upload, install, pth, audit, mypy, pytest_at):
        found.add("steps-missing")
    if not refresh < stage < upload < install < pytest_at:
        found.add("order")
    if any(not install <= gate <= pytest_at for gate in (pth, audit, mypy)):
        found.add("gate-order")

    # The refresh step: its output feeds the guard, and a missing `changed=` line must fail.
    step = test_steps[refresh] if refresh < len(test_steps) else {}
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

    # Every step of `test` after the refresh runs only when a pin changed.
    for index, later in enumerate(test_steps):
        if index > refresh and later.get("if") != _GUARD:
            found.add("guard")

    # The artifact: staged and uploaded before the install, downloaded by `propose`, same name.
    if upload < len(test_steps):
        up = test_steps[upload].get("with") or {}
        if (
            up.get("name") != _ARTIFACT
            or up.get("path") != _ARTIFACT_PATH
            or up.get("if-no-files-found") != "error"
        ):
            found.add("artifact")
    stage_run = str(test_steps[stage].get("run", "")) if stage < len(test_steps) else ""
    if (
        'cp constraints.txt pyproject.toml "$stage/"' not in stage_run
        or 'cp "$RUNNER_TEMP/refresh-report.md" "$stage/report.md"' not in stage_run
    ):
        found.add("artifact")
    download = _find(propose, "actions/download-artifact@")
    if download >= len(propose_steps):
        found.add("steps-missing")
    else:
        down = propose_steps[download].get("with") or {}
        if down.get("name") != _ARTIFACT or down.get("path") != _ARTIFACT_PATH:
            found.add("artifact")

    # The gates are the documented ones, no test deselected.
    if pytest_at < len(test_steps):
        pytest_run = str(test_steps[pytest_at].get("run", "")).strip()
        if pytest_run != _PYTEST_RUN or pytest_run != reference["pytest"]:
            found.add("pytest-line")
    env = test.get("env") or {}
    if (
        env.get("PYTEST_MARKERS") != _MARKERS
        or env.get("PYTEST_MARKERS") != reference["markers"]
    ):
        found.add("pytest-selection")
    if re.search(r"--deselect|--ignore|\s-k\s|\s-x\b|--lf\b|--maxfail", code):
        found.add("pytest-deselect")
    if install < len(test_steps):
        install_lines = [
            x.strip() for x in str(test_steps[install].get("run", "")).splitlines()
        ]
        if reference["install"] not in install_lines:
            found.add("install-line")
    if (
        mypy < len(test_steps)
        and str(test_steps[mypy].get("run", "")).strip() != reference["mypy"]
    ):
        found.add("mypy-line")

    # The steps of `propose`: download, guard, then the pull request (commit, push, open).
    guard = _find(propose, "git diff --name-only")
    pull_request = _find(propose, "gh pr create")
    push = _find(propose, "git push")
    if 10_000 in (guard, pull_request, push):
        found.add("guard-missing")
    elif not download < guard < pull_request:
        found.add("propose-order")
    if push != pull_request:
        # the push and the pull request are one step, so the branch is never pushed alone
        found.add("push-order")
    if pull_request < len(propose_steps):
        pr = propose_steps[pull_request]
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
            '"$RUNNER_TEMP/pins/report.md"',
        )
        if not all(token in script for token in needed):
            found.add("pr-shape")
        order = [script.find(t) for t in ("git commit", "git push", "gh pr create")]
        if min(order) < 0 or order != sorted(order):
            found.add("pr-order")
    if re.search(
        r"git add (-A|--all|\.)(\s|$)",
        "\n".join(str(s.get("run", "")) for s in propose_steps),
    ):
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

    def test_there_are_two_jobs_and_only_the_second_can_write(self):
        wf = _load(_real_text())
        assert set(wf["jobs"]) == {"test", "propose"}
        assert wf["permissions"] == {"contents": "read"}
        assert "permissions" not in wf["jobs"]["test"]
        assert wf["jobs"]["propose"]["permissions"] == _WRITE_SCOPES

    def test_propose_waits_for_test_and_for_a_change_and_never_runs_on_failure(self):
        propose = _load(_real_text())["jobs"]["propose"]
        assert propose["needs"] == "test"
        assert propose["if"] == _PROPOSE_IF
        # No status function: the implicit `success()` stays, so a red test job proposes nothing.
        assert not re.search(r"always\(\)|failure\(\)|cancelled\(\)", propose["if"])

    def test_the_test_job_holds_no_credential_and_no_secret(self):
        text = _real_text()
        job = _load(text)["jobs"]["test"]
        checkout = _steps(job)[0]
        assert checkout["with"]["persist-credentials"] is False
        assert "secrets." not in json.dumps(job)

    def test_the_propose_job_keeps_credentials_and_says_why(self):
        text = _real_text()
        checkout = _steps(_load(text)["jobs"]["propose"])[0]
        assert checkout["with"]["persist-credentials"] is True
        assert (
            "git push"
            in text.split("persist-credentials: true")[0].rsplit("- name:", 1)[-1]
        )

    def test_the_propose_job_installs_nothing_and_runs_no_project_code(self):
        job = _load(_real_text())["jobs"]["propose"]
        for step in _steps(job):
            assert not str(step.get("uses", "")).startswith("actions/setup-python@")
            assert not runs_code(str(step.get("run", ""))), step.get("name")

    def test_the_gates_run_in_the_test_job_in_order_after_the_files_are_staged(self):
        job = _load(_real_text())["jobs"]["test"]
        names = [s.get("name", "") for s in _steps(job)]
        order = [
            "Pick the newest eligible pins",
            "Stage the proposed files",
            "Upload the proposed files",
            "Install the package and every extra at the new pins",
            "Audit for suspicious .pth files",
            "Audit the installed environment for known vulnerabilities",
            "Type check with mypy",
            "Test with pytest",
        ]
        assert [n for n in names if n in order] == order
        assert names[:2] == ["Check out the repository", "Set up Python"]
        propose = [
            s.get("name", "") for s in _steps(_load(_real_text())["jobs"]["propose"])
        ]
        assert propose == [
            "Check out the repository",
            "Download the proposed files",
            _GUARD_STEP,
            _PR_STEP,
        ]

    def test_every_step_after_the_refresh_is_guarded_by_its_output(self):
        job = _load(_real_text())["jobs"]["test"]
        steps = _steps(job)
        refresh = next(i for i, s in enumerate(steps) if s.get("id") == "refresh")
        assert "if" not in steps[refresh]
        assert [s["if"] for s in steps[refresh + 1 :]] == [_GUARD] * len(
            steps[refresh + 1 :]
        )

    def test_the_job_output_is_the_refresh_step_output(self):
        job = _load(_real_text())["jobs"]["test"]
        assert job["outputs"] == {"changed": "${{ steps.refresh.outputs.changed }}"}

    def test_the_pytest_line_is_the_documented_selection_with_no_deselect(self):
        text = _real_text()
        job = _load(text)["jobs"]["test"]
        reference = _ci_reference()
        assert job["env"]["PYTEST_MARKERS"] == reference["markers"] == _MARKERS
        step = next(s for s in _steps(job) if "pytest" in str(s.get("run", "")))
        assert step["run"].strip() == _PYTEST_RUN == reference["pytest"]
        assert not re.search(
            r"--deselect|--ignore|\s-k\s", "\n".join(_code_lines(text))
        )

    def test_the_install_line_is_the_one_of_every_ci_job(self):
        job = _load(_real_text())["jobs"]["test"]
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
        for job in _load(_real_text())["jobs"].values():
            for step in _steps(job):
                assert "${{" not in str(step.get("run", "")), step.get("name")

    def test_the_only_secret_named_is_the_default_token(self):
        assert set(_SECRET.findall(_real_text())) == {"GITHUB_TOKEN"}

    def test_every_action_is_a_sha_pin_with_its_release_tag(self):
        matches = list(_USES_LINE.finditer(_real_text()))
        assert len(matches) == 5
        for match in matches:
            assert _SHA_USES.match(match["ref"]), match["ref"]
            assert _TAG_COMMENT.match(match["rest"]), match["rest"]
        assert {m["ref"].split("@")[0] for m in matches} == _ACTIONS

    def test_the_actions_carry_the_pins_of_the_other_workflows(self):
        others = {
            m["ref"]
            for path in _WORKFLOWS.glob("*.yml")
            if path != _WORKFLOW
            for m in _USES_LINE.finditer(path.read_text(encoding="utf-8"))
        }
        mine = {m["ref"] for m in _USES_LINE.finditer(_real_text())}
        assert mine <= others

    def test_the_artifact_is_one_directory_named_the_same_on_both_sides(self):
        jobs = _load(_real_text())["jobs"]
        up = next(
            s for s in _steps(jobs["test"]) if "upload-artifact" in str(s.get("uses"))
        )
        down = next(
            s
            for s in _steps(jobs["propose"])
            if "download-artifact" in str(s.get("uses"))
        )
        assert up["with"]["name"] == down["with"]["name"] == _ARTIFACT
        assert up["with"]["path"] == down["with"]["path"] == _ARTIFACT_PATH
        assert up["with"]["if-no-files-found"] == "error"

    def test_it_never_runs_on_a_pull_request_or_a_push(self):
        code = "\n".join(_code_lines(_real_text()))
        assert "pull_request" not in code
        assert not re.search(r"^\s*push:", code, re.MULTILINE)

    def test_a_run_never_cancels_one_in_progress(self):
        conc = _load(_real_text())["concurrency"]
        assert conc == {"group": "refresh-pins", "cancel-in-progress": False}

    def test_the_test_job_is_skipped_on_a_fork(self):
        job = _load(_real_text())["jobs"]["test"]
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


class TestTheCommandScanner:
    @pytest.mark.parametrize(
        "script",
        [
            "pip install x",
            "python -m pytest",
            "python3.12 -c 1",
            "set -e\nmypy -p x",
            "echo hi && pip install x",
            "if true; then uv pip install x; fi",
            "x=$(pip download y)",
            "FOO=1 pytest",
            "sudo pip install x",
            "nice -n 19 pytest",
            "make test",
            "./scripts/x.sh",
            "scripts/x.py",
            "bash run.sh",
            "echo hi | python -",
            "cat <<'X'\nhello\nX\npip install y",
            "npm install",
            "\\\n  pip install x",
        ],
    )
    def test_a_code_command_is_found(self, script):
        assert runs_code(script), script

    @pytest.mark.parametrize(
        "script",
        [
            "git push origin main",
            "gh pr create --title t",
            "echo 'pip install x and pytest'",
            'echo "run pytest; pip install"',
            "cp a b && cd d",
            "cat <<'X'\npip install y\nX\necho done",
            "grep -E 'x' f | sort",
            "# pip install x",
            "if [ -f x ]; then echo y; fi",
            'echo "$(wc -c < f)"',
        ],
    )
    def test_a_plain_command_is_not_code(self, script):
        assert not runs_code(script), script


# --------------------------------------------------------------------------------------
# The guard and the pull request step, run in bash against scratch repositories
# --------------------------------------------------------------------------------------


def _step_run(job: str, name: str) -> str:
    step = next(
        s for s in _steps(_load(_real_text())["jobs"][job]) if s.get("name") == name
    )
    return str(step["run"])


def _guard_script() -> str:
    return _step_run("propose", _GUARD_STEP)


def _pr_script() -> str:
    return _step_run("propose", _PR_STEP)


def _git(cwd: Path, *argv: str) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.invalid",
    }
    return subprocess.run(
        ["git", *argv],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )


_REPORT = "# Pin refresh\n\n| package | old | new |\n| mcp | 2.2.0 | 2.3.0 |\n\nchanged=true\n"


class _Scratch:
    """A scratch checkout of the real `constraints.txt` and `pyproject.toml` plus an artifact."""

    def __init__(self, base: Path) -> None:
        self.repo = base / "repo"
        self.temp = base / "runner-temp"
        self.pins = self.temp / "pins"
        self.repo.mkdir(parents=True)
        self.pins.mkdir(parents=True)
        for name in ("constraints.txt", "pyproject.toml"):
            shutil.copy(_ROOT / name, self.repo / name)
        (self.repo / "notes.txt").write_text("notes\n", encoding="utf-8")
        _git(self.repo, "init", "-q", "-b", "main")
        _git(self.repo, "add", "constraints.txt", "pyproject.toml", "notes.txt")
        _git(self.repo, "commit", "-q", "-m", "base")
        self.constraints = (self.repo / "constraints.txt").read_text(encoding="utf-8")
        self.pyproject = (self.repo / "pyproject.toml").read_text(encoding="utf-8")

    def artifact(
        self,
        constraints: str | None = None,
        pyproject: str | None = None,
        report: str = _REPORT,
    ) -> None:
        (self.pins / "constraints.txt").write_text(
            self.constraints.replace("mcp==2.2.0", "mcp==2.3.0")
            if constraints is None
            else constraints,
            encoding="utf-8",
            newline="",
        )
        (self.pins / "pyproject.toml").write_text(
            self.pyproject if pyproject is None else pyproject,
            encoding="utf-8",
            newline="",
        )
        (self.pins / "report.md").write_text(report, encoding="utf-8")

    def run(self, script: str, **extra: str) -> subprocess.CompletedProcess[str]:
        env = {
            "PATH": os.environ["PATH"],
            "HOME": str(self.temp),
            "RUNNER_TEMP": str(self.temp),
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
            **extra,
        }
        return subprocess.run(
            ["bash", "-c", script],
            cwd=self.repo,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )


def _widen_mcp(pyproject: str) -> str:
    assert '"mcp>=2.2,<3"' in pyproject
    return pyproject.replace('"mcp>=2.2,<3"', '"mcp>=2.2,<4"')


def _narrow_other_package(constraints: str) -> str:
    return constraints.replace("pydantic==2.14.3", "pydantic==2.14.4")


def _guard_scenarios() -> list[tuple[str, Callable[[_Scratch], None], bool]]:
    """(label, how to arrange the scratch artifact, whether the guard must pass)."""

    def good(s: _Scratch) -> None:
        s.artifact()

    def good_widened(s: _Scratch) -> None:
        s.artifact(pyproject=_widen_mcp(s.pyproject))

    def extra_file(s: _Scratch) -> None:
        s.artifact()
        (s.pins / "extra.txt").write_text("x", encoding="utf-8")

    def missing_report(s: _Scratch) -> None:
        s.artifact()
        (s.pins / "report.md").unlink()

    def missing_pyproject(s: _Scratch) -> None:
        s.artifact()
        (s.pins / "pyproject.toml").unlink()

    def subdirectory(s: _Scratch) -> None:
        s.artifact()
        (s.pins / "sub").mkdir()
        (s.pins / "sub" / "constraints.txt").write_text("x", encoding="utf-8")

    def empty_subdirectory(s: _Scratch) -> None:
        s.artifact()
        (s.pins / "sub").mkdir()

    def hidden_file(s: _Scratch) -> None:
        s.artifact()
        (s.pins / ".hidden").write_text("x", encoding="utf-8")

    def symlinked_constraints(s: _Scratch) -> None:
        s.artifact()
        target = s.temp / "elsewhere.txt"
        target.write_text(
            s.constraints.replace("mcp==2.2.0", "mcp==2.3.0"), encoding="utf-8"
        )
        (s.pins / "constraints.txt").unlink()
        symlink_or_skip(s.pins / "constraints.txt", target)

    def symlinked_report(s: _Scratch) -> None:
        s.artifact()
        target = s.temp / "elsewhere.md"
        target.write_text(_REPORT, encoding="utf-8")
        (s.pins / "report.md").unlink()
        symlink_or_skip(s.pins / "report.md", target)

    def oversized_report(s: _Scratch) -> None:
        s.artifact(report="x" * 1_000_001)

    def another_package_pinned(s: _Scratch) -> None:
        s.artifact(
            constraints=_narrow_other_package(
                s.constraints.replace("mcp==2.2.0", "mcp==2.3.0")
            )
        )

    def a_package_added(s: _Scratch) -> None:
        s.artifact(
            constraints=s.constraints.replace("mcp==2.2.0", "mcp==2.3.0")
            + "evil==1.0\n"
        )

    def a_comment_added(s: _Scratch) -> None:
        s.artifact(
            constraints=s.constraints.replace("mcp==2.2.0", "mcp==2.3.0") + "# hi\n"
        )

    def a_pin_with_a_url(s: _Scratch) -> None:
        s.artifact(
            constraints=s.constraints.replace("mcp==2.2.0", "mcp==2.3.0.dev1+local")
        )

    def constraints_unchanged(s: _Scratch) -> None:
        s.artifact(constraints=s.constraints)

    def pyproject_only(s: _Scratch) -> None:
        s.artifact(constraints=s.constraints, pyproject=_widen_mcp(s.pyproject))

    def pyproject_adds_a_dependency(s: _Scratch) -> None:
        s.artifact(
            pyproject=s.pyproject.replace(
                '"pyyaml>=6.0,<7",', '"pyyaml>=6.0,<7",\n    "evil>=1,<2",'
            )
        )

    def pyproject_changes_a_script(s: _Scratch) -> None:
        s.artifact(
            pyproject=s.pyproject.replace(
                'fsm-llm = "fsm_llm.__main__:main_cli"', 'fsm-llm = "evil:main"'
            )
        )

    def pyproject_widens_another_package(s: _Scratch) -> None:
        s.artifact(
            pyproject=s.pyproject.replace('"pydantic>=2.0,<3"', '"pydantic>=2.0,<4"')
        )

    def pyproject_range_with_a_comment(s: _Scratch) -> None:
        s.artifact(
            pyproject=s.pyproject.replace(
                'mcp = ["mcp>=2.2,<3"]', 'mcp = ["mcp>=2.2,<4"]  # evil'
            )
        )

    def tree_has_an_untracked_file(s: _Scratch) -> None:
        s.artifact()
        (s.repo / "stray.txt").write_text("x", encoding="utf-8")

    def tree_has_a_modified_file(s: _Scratch) -> None:
        s.artifact()
        (s.repo / "notes.txt").write_text("changed\n", encoding="utf-8")

    return [
        ("the expected files pass", good, True),
        ("a widened mcp range passes", good_widened, True),
        ("an extra file in the artifact", extra_file, False),
        ("the report missing", missing_report, False),
        ("the pyproject copy missing", missing_pyproject, False),
        ("a subdirectory with a file", subdirectory, False),
        ("an empty subdirectory", empty_subdirectory, False),
        ("a hidden file", hidden_file, False),
        ("constraints.txt is a symbolic link", symlinked_constraints, False),
        ("report.md is a symbolic link", symlinked_report, False),
        ("a report over one megabyte", oversized_report, False),
        ("a pin of another package changes", another_package_pinned, False),
        ("a package line is added", a_package_added, False),
        ("a comment line is added", a_comment_added, False),
        ("a pin that is not plain digits", a_pin_with_a_url, False),
        ("constraints.txt is unchanged", constraints_unchanged, False),
        ("only pyproject.toml changes", pyproject_only, False),
        ("pyproject adds a dependency", pyproject_adds_a_dependency, False),
        ("pyproject changes a console script", pyproject_changes_a_script, False),
        ("pyproject widens another package", pyproject_widens_another_package, False),
        ("a range line with trailing text", pyproject_range_with_a_comment, False),
        ("the tree holds an untracked file", tree_has_an_untracked_file, False),
        ("the tree holds another modified file", tree_has_a_modified_file, False),
    ]


def guard_problems(script: str, base: Path) -> list[str]:
    """Labels of the scenarios the guard script judges wrongly (empty: it is sound)."""
    wrong = []
    for number, (label, arrange, must_pass) in enumerate(_guard_scenarios()):
        scratch = _Scratch(base / f"scenario-{number}")
        arrange(scratch)
        proc = scratch.run(script)
        if (proc.returncode == 0) != must_pass:
            wrong.append(
                f"{label}: exit {proc.returncode}: {proc.stderr.strip()[:120]}"
            )
    return wrong


class TestTheGuardStep:
    def test_the_shipped_guard_judges_every_scenario_correctly(self, tmp_path: Path):
        assert guard_problems(_guard_script(), tmp_path) == []

    def test_a_passing_guard_leaves_exactly_the_proposal_in_the_tree(
        self, tmp_path: Path
    ):
        scratch = _Scratch(tmp_path)
        scratch.artifact(pyproject=_widen_mcp(scratch.pyproject))
        proc = scratch.run(_guard_script())
        assert proc.returncode == 0, proc.stderr
        names = _git(scratch.repo, "diff", "--name-only").stdout.split()
        assert sorted(names) == ["constraints.txt", "pyproject.toml"]
        assert (
            _git(scratch.repo, "ls-files", "--others", "--exclude-standard").stdout
            == ""
        )
        assert "mcp==2.3.0" in (scratch.repo / "constraints.txt").read_text(
            encoding="utf-8"
        )

    def test_a_failing_guard_names_its_reason_on_standard_error(self, tmp_path: Path):
        scratch = _Scratch(tmp_path)
        scratch.artifact()
        (scratch.pins / "extra.txt").write_text("x", encoding="utf-8")
        proc = scratch.run(_guard_script())
        assert proc.returncode != 0
        assert "unexpected entries" in proc.stderr

    def test_every_scenario_that_must_fail_fails_for_the_right_reason_only_once(self):
        labels = [label for label, _, _ in _guard_scenarios()]
        assert len(labels) == len(set(labels))
        assert sum(1 for _, _, ok in _guard_scenarios() if ok) == 2


def _drop(old: str) -> Callable[[str], str]:
    def apply(text: str) -> str:
        assert old in text, f"script mutant anchor not found: {old!r}"
        return text.replace(old, "true", 1)

    return apply


def _swap(old: str, new: str) -> Callable[[str], str]:
    def apply(text: str) -> str:
        assert old in text, f"script mutant anchor not found: {old!r}"
        return text.replace(old, new, 1)

    return apply


#: Each mutant of the guard script must be judged wrong by at least one scenario.
GUARD_MUTANTS: list[tuple[str, Callable[[str], str]]] = [
    (
        "the entry listing is not compared",
        _swap(
            'if [ "$listing" != "./constraints.txt ./pyproject.toml ./report.md " ]',
            "if false",
        ),
    ),
    (
        "the listing ignores a hidden file",
        _swap("find . -mindepth 1 |", "find . -mindepth 1 -not -name '.*' |"),
    ),
    (
        "a link is accepted as a file",
        _swap(
            'if [ -L "$pins/$name" ] || [ ! -f "$pins/$name" ]',
            'if [ ! -e "$pins/$name" ]',
        ),
    ),
    (
        "the size limit is removed",
        _swap('if [ "$(wc -c < "$pins/$name")" -gt 1000000 ]', "if false"),
    ),
    (
        "untracked files are not looked for",
        _swap('if [ -n "$(git ls-files --others --exclude-standard)" ]', "if false"),
    ),
    (
        "the changed paths are not compared",
        _swap(
            'if [ "$touched" != "constraints.txt " ] && [ "$touched" != "constraints.txt pyproject.toml " ]',
            "if false",
        ),
    ),
    (
        "a pyproject-only change is accepted",
        _swap(
            '[ "$touched" != "constraints.txt " ] &&',
            '[ "$touched" != "pyproject.toml " ] && [ "$touched" != "constraints.txt " ] &&',
        ),
    ),
    (
        "the constraints lines are not compared",
        _swap('grep -vE "$pin_line"; then', "false; then"),
    ),
    (
        "the constraints pin pattern accepts any package",
        _swap("(litellm|mcp)==[0-9]+", "[a-z]+==[0-9]+"),
    ),
    (
        "the constraints pin pattern accepts any version text",
        _swap("==[0-9]+(\\.[0-9]+)*$'", "==.*$'"),
    ),
    (
        "the pyproject lines are not compared",
        _swap('grep -vE "$range_line"; then', "false; then"),
    ),
    (
        "the pyproject pattern accepts any package",
        _swap("(litellm|mcp)>=", "[a-z]+>="),
    ),
    (
        "the pyproject pattern is not anchored at the end",
        _swap("(\\]|,)?$'", "(\\]|,)?'"),
    ),
    (
        "the header lines of the diff are not skipped",
        _swap(
            "| grep -vE '^(\\+\\+\\+|---) ' | grep -vE \"$pin_line\"",
            '| grep -vE "$pin_line"',
        ),
    ),
]


class TestTheGuardScriptIsMutationChecked:
    @pytest.mark.parametrize(
        "label,mutate", GUARD_MUTANTS, ids=[m[0] for m in GUARD_MUTANTS]
    )
    def test_each_mutant_is_judged_wrong_by_a_scenario(
        self, label: str, mutate: Callable[[str], str], tmp_path: Path
    ):
        script = _guard_script()
        mutated = mutate(script)
        assert mutated != script
        assert guard_problems(mutated, tmp_path) != [], label


class TestThePullRequestStep:
    """Replay of the step against a local bare repository and a stub `gh` (nothing is sent)."""

    def _run(
        self, tmp_path: Path
    ) -> tuple[_Scratch, Path, subprocess.CompletedProcess[str]]:
        scratch = _Scratch(tmp_path)
        scratch.artifact(pyproject=_widen_mcp(scratch.pyproject))
        assert scratch.run(_guard_script()).returncode == 0
        remote = tmp_path / "remote.git"
        _git(tmp_path, "init", "-q", "--bare", str(remote))
        _git(scratch.repo, "remote", "add", "origin", str(remote))
        stub = tmp_path / "bin"
        stub.mkdir()
        log = tmp_path / "gh.log"
        (stub / "gh").write_text(
            "#!/usr/bin/env bash\n"
            'printf "%s\\n" "$@" > "$GH_LOG"\n'
            'while [ "$#" -gt 0 ]; do\n'
            '  if [ "$1" = "--body-file" ]; then cp "$2" "$GH_LOG.body"; fi\n'
            "  shift\n"
            "done\n",
            encoding="utf-8",
        )
        (stub / "gh").chmod(0o755)
        proc = scratch.run(
            _pr_script(),
            PATH=os.pathsep.join([str(stub), os.environ["PATH"]]),
            GH_TOKEN="stub-token",
            GH_LOG=str(log),
            GITHUB_SERVER_URL="https://github.example",
            GITHUB_REPOSITORY="owner/repo",
            GITHUB_RUN_ID="42",
        )
        return scratch, remote, proc

    def test_it_commits_the_two_files_pushes_one_branch_and_opens_the_request(
        self, tmp_path: Path
    ):
        _, remote, proc = self._run(tmp_path)
        assert proc.returncode == 0, proc.stderr
        branches = _git(remote, "branch", "--list").stdout.split()
        assert len(branches) == 1 and re.fullmatch(
            r"chore/refresh-pins-\d{4}-\d\d-\d\d-\d{6}", branches[0]
        ), branches
        changed = _git(
            remote, "show", "--name-only", "--format=%an <%ae>%n%s", branches[0]
        ).stdout.split("\n")
        assert (
            changed[0]
            == "github-actions[bot] <41898282+github-actions[bot]@users.noreply.github.com>"
        )
        assert changed[1] == _COMMIT_MESSAGE
        assert sorted(x for x in changed[2:] if x) == [
            "constraints.txt",
            "pyproject.toml",
        ]
        args = (tmp_path / "gh.log").read_text(encoding="utf-8").split("\n")
        assert args[:6] == ["pr", "create", "--base", "main", "--head", branches[0]]
        assert "--title" in args and _COMMIT_MESSAGE in args

    def test_the_body_is_the_report_without_the_output_line_and_the_run_link(
        self, tmp_path: Path
    ):
        _, _, proc = self._run(tmp_path)
        assert proc.returncode == 0, proc.stderr
        body = (tmp_path / "gh.log.body").read_text(encoding="utf-8")
        assert "| mcp | 2.2.0 | 2.3.0 |" in body
        assert "changed=" not in body
        assert body.rstrip().endswith(
            "Run: https://github.example/owner/repo/actions/runs/42"
        )


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
    end = text.find("\n  propose:\n", start + 1)
    if nxt < 0 or (0 <= end < nxt):
        nxt = end
    return start, (len(text) if nxt < 0 else nxt + 1)


def _drop_step(name: str) -> Callable[[str], str]:
    def apply(text: str) -> str:
        start, end = _step_block(text, name)
        return text[:start] + text[end:]

    return apply


def _move_step_to_end(name: str) -> Callable[[str], str]:
    def apply(text: str) -> str:
        start, end = _step_block(text, name)
        block = text[start:end]
        if not block.endswith("\n"):
            block += "\n"
        text = text[:start] + text[end:]
        return text.rstrip("\n") + "\n" + block

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


_PICK = "Pick the newest eligible pins"
_STAGE = "Stage the proposed files"
_UPLOAD = "Upload the proposed files"
_INSTALL = "Install the package and every extra at the new pins"
_TESTS = "Test with pytest"
_DOWNLOAD = "Download the proposed files"
_GUARD_LINE = "        if: steps.refresh.outputs.changed == 'true'\n"
_SHA_CHECKOUT = "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1"
_SHA_PYTHON = "actions/setup-python@6d4fd22b20465ccc0f7354476588e8df7c9ca0e0 # v7.1.0"
_PROPOSE_HEAD = "  propose:\n"
_PROPOSE_STEPS = "    steps:\n      - name: Check out the repository\n"


def _in_propose(old: str, new: str) -> Callable[[str], str]:
    """Replace the first `old` inside the propose job only."""

    def apply(text: str) -> str:
        at = text.index(_PROPOSE_HEAD)
        head, tail = text[:at], text[at:]
        assert old in tail, f"mutant anchor not in the propose job: {old!r}"
        return head + tail.replace(old, new, 1)

    return apply


def _in_test(old: str, new: str) -> Callable[[str], str]:
    def apply(text: str) -> str:
        at = text.index(_PROPOSE_HEAD)
        head, tail = text[:at], text[at:]
        assert old in head, f"mutant anchor not in the test job: {old!r}"
        return head.replace(old, new, 1) + tail

    return apply


def _add_propose_step(step: str) -> Callable[[str], str]:
    """Insert a step right after the checkout of the propose job."""

    def apply(text: str) -> str:
        at = text.index(_PROPOSE_HEAD)
        marker = "      - name: Download the proposed files\n"
        i = text.index(marker, at)
        return text[:i] + step + text[i:]

    return apply


def _drop_propose_permissions(text: str) -> str:
    at = text.index(_PROPOSE_HEAD)
    tail = re.sub(r"    permissions:\n(?:      .*\n)+", "", text[at:], count=1)
    return text[:at] + tail


MUTANTS: list[tuple[str, Callable[[str], str], str]] = [
    # Order of the test job
    ("the install comes before the refresh", _move_step(_INSTALL, _PICK), "order"),
    ("the install comes before the staging", _move_step(_INSTALL, _STAGE), "order"),
    ("the upload comes before the staging", _move_step(_UPLOAD, _STAGE), "order"),
    ("the staging is removed", _drop_step(_STAGE), "steps-missing"),
    ("the upload is removed", _drop_step(_UPLOAD), "steps-missing"),
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
    ("the download is removed", _drop_step(_DOWNLOAD), "steps-missing"),
    ("the guard step is removed", _drop_step(_GUARD_STEP), "guard-missing"),
    (
        "the guard step runs after the pull request",
        _move_step_to_end(_GUARD_STEP),
        "propose-order",
    ),
    (
        "the guard step runs before the download",
        _move_step(_GUARD_STEP, _DOWNLOAD),
        "propose-order",
    ),
    (
        "mypy runs before the install",
        _move_step("Type check with mypy", _INSTALL),
        "gate-order",
    ),
    (
        "the commit comes after the push",
        _edit(
            '          git commit -m "chore(deps): refresh litellm and mcp pins"\n'
            '          git push origin "$branch"\n',
            '          git push origin "$branch"\n'
            '          git commit -m "chore(deps): refresh litellm and mcp pins"\n',
        ),
        "pr-order",
    ),
    (
        "the branch is pushed by an earlier step",
        _edit(
            "      - name: Check the proposed files and apply them\n",
            "      - name: Push early\n        run: git push origin HEAD\n"
            "      - name: Check the proposed files and apply them\n",
        ),
        "push-order",
    ),
    (
        "the refresh step has a condition",
        _edit("        id: refresh\n", "        id: refresh\n        if: always()\n"),
        "refresh-guard",
    ),
    (
        "a uses line is hidden in a script",
        _edit(
            "          set -euo pipefail\n          branch=",
            "          set -euo pipefail\n          uses: actions/checkout@v7\n          branch=",
        ),
        "uses-count",
    ),
    # Guards of the test job's steps
    ("the guard is removed from the stage step", _edit(_GUARD_LINE, "", 0), "guard"),
    ("the guard is removed from the install step", _edit(_GUARD_LINE, "", 2), "guard"),
    ("the guard is removed from the tests", _edit(_GUARD_LINE, "", 6), "guard"),
    (
        "the guard tests the wrong value",
        _edit("changed == 'true'", "changed == 'false'", 2),
        "guard",
    ),
    # Permissions and credentials
    (
        "the test job gets a write scope",
        _in_test(
            "    outputs:\n", "    permissions:\n      contents: write\n    outputs:\n"
        ),
        "test-permissions",
    ),
    (
        "the test job gets a read scope block",
        _in_test(
            "    outputs:\n", "    permissions:\n      contents: read\n    outputs:\n"
        ),
        "test-permissions",
    ),
    (
        "the test checkout keeps its credentials",
        _in_test("persist-credentials: false", "persist-credentials: true"),
        "test-credentials",
    ),
    (
        "the test checkout has no credentials key",
        _in_test("        with:\n          persist-credentials: false\n", ""),
        "test-credentials",
    ),
    (
        "the propose checkout drops its credentials",
        _in_propose("persist-credentials: true", "persist-credentials: false"),
        "propose-credentials",
    ),
    (
        "the credentials are kept without saying why",
        _in_propose("(`git push`)", "(the remote)"),
        "propose-credentials-reason",
    ),
    (
        "the propose job loses the pull request scope",
        _in_propose("      pull-requests: write\n", ""),
        "propose-permissions",
    ),
    (
        "the propose job has no permissions block",
        _drop_propose_permissions,
        "propose-permissions",
    ),
    (
        "the propose write scope is widened",
        _in_propose(
            "      pull-requests: write\n",
            "      pull-requests: write\n      actions: write\n",
        ),
        "propose-permissions",
    ),
    (
        "an id token is requested",
        _in_propose(
            "      pull-requests: write\n",
            "      pull-requests: write\n      id-token: write\n",
        ),
        "extra-permission",
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
        "the test job reads a secret",
        _in_test(
            '      PYTHONUTF8: "1"\n',
            '      PYTHONUTF8: "1"\n      GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}\n',
        ),
        "test-secret",
    ),
    # needs and if
    ("the needs is removed", _in_propose("    needs: test\n", ""), "propose-needs"),
    (
        "the needs names another job",
        _in_propose("    needs: test\n", "    needs: other\n"),
        "propose-needs",
    ),
    (
        "the if is removed",
        _in_propose("    if: ${{ needs.test.outputs.changed == 'true' }}\n", ""),
        "propose-if",
    ),
    (
        "the if runs on failure too",
        _in_propose(
            "needs.test.outputs.changed == 'true'",
            "always() && needs.test.outputs.changed == 'true'",
        ),
        "propose-if",
    ),
    (
        "the if tests the wrong value",
        _in_propose("changed == 'true'", "changed == 'false'"),
        "propose-if",
    ),
    (
        "the job output is removed",
        _in_test("      changed: ${{ steps.refresh.outputs.changed }}\n", ""),
        "test-output",
    ),
    (
        "the job output reads another step",
        _in_test(
            "changed: ${{ steps.refresh.outputs.changed }}",
            "changed: ${{ steps.pick.outputs.changed }}",
        ),
        "test-output",
    ),
    # The propose job runs no project code
    (
        "the propose job installs a package",
        _add_propose_step("      - name: Install\n        run: pip install requests\n"),
        "propose-runs-code",
    ),
    (
        "the propose job installs with python -m pip",
        _add_propose_step(
            "      - name: Install\n        run: python -m pip install requests\n"
        ),
        "propose-runs-code",
    ),
    (
        "the propose job installs with uv",
        _add_propose_step(
            "      - name: Install\n        run: uv pip install requests\n"
        ),
        "propose-runs-code",
    ),
    (
        "the propose job runs a project script",
        _add_propose_step(
            "      - name: Run\n        run: python scripts/refresh_pins.py\n"
        ),
        "propose-runs-code",
    ),
    (
        "the propose job runs the tests",
        _add_propose_step("      - name: Run\n        run: pytest tests/\n"),
        "propose-runs-code",
    ),
    (
        "the propose job runs make",
        _add_propose_step("      - name: Run\n        run: make test\n"),
        "propose-runs-code",
    ),
    (
        "the propose job sets up python",
        _add_propose_step(
            "      - name: Set up Python\n        uses: " + _SHA_PYTHON + "\n"
        ),
        "propose-runs-code",
    ),
    (
        "the propose job runs a script behind an assignment",
        _add_propose_step("      - name: Run\n        run: FOO=1 ./scripts/x.sh\n"),
        "propose-runs-code",
    ),
    # Pull request step
    (
        "the pull request adds every file",
        _in_propose("git add constraints.txt pyproject.toml", "git add -A"),
        "pr-add-all",
    ),
    (
        "the pull request adds the dot path",
        _in_propose("git add constraints.txt pyproject.toml", "git add ."),
        "pr-add-all",
    ),
    (
        "the push is removed",
        _in_propose('          git push origin "$branch"\n', ""),
        "guard-missing",
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
    ("the pull request has no body file", _edit("--body-file", "--body"), "pr-shape"),
    (
        "the body reads the report from the workspace",
        _edit('"$RUNNER_TEMP/pins/report.md"', "report.md"),
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
        "the pull request token is another",
        _edit("GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}", "GH_TOKEN: ${{ secrets.PAT }}"),
        "pr-token",
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
        "the strict shell is removed from a script",
        _edit("          set -euo pipefail\n          branch=", "          branch="),
        "shell-strict",
    ),
    # The artifact
    (
        "the artifact is uploaded under another name",
        _in_test("          name: refresh-pins\n", "          name: pins\n"),
        "artifact",
    ),
    (
        "the artifact is downloaded under another name",
        _in_propose("          name: refresh-pins\n", "          name: pins\n"),
        "artifact",
    ),
    (
        "the upload tolerates a missing file",
        _edit("if-no-files-found: error", "if-no-files-found: ignore"),
        "artifact",
    ),
    (
        "the report is not staged",
        _edit('          cp "$RUNNER_TEMP/refresh-report.md" "$stage/report.md"\n', ""),
        "artifact",
    ),
    (
        "the proposal files are not staged",
        _edit(
            'cp constraints.txt pyproject.toml "$stage/"',
            'cp constraints.txt "$stage/"',
        ),
        "steps-missing",
    ),
    (
        "the download lands elsewhere",
        _in_propose("path: ${{ runner.temp }}/pins", "path: ${{ runner.temp }}/other"),
        "artifact",
    ),
    # Triggers, cron, runner, timeout, pins
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
        "a fifth action is used",
        _edit(
            "      - name: Pick the newest eligible pins\n",
            "      - name: Extra\n        uses: actions/cache@0057852bfaa89a56745cba8c7296529d2fc39830 # v4.3.0\n      - name: Pick the newest eligible pins\n",
        ),
        "action-set",
    ),
    (
        "the runner floats",
        _edit("runs-on: ubuntu-24.04", "runs-on: ubuntu-latest"),
        "runner",
    ),
    (
        "the propose runner floats",
        _in_propose("runs-on: ubuntu-24.04", "runs-on: ubuntu-latest"),
        "runner",
    ),
    ("the test timeout is removed", _edit("    timeout-minutes: 90\n", ""), "timeout"),
    (
        "the test timeout is the platform default",
        _edit("timeout-minutes: 90", "timeout-minutes: 360"),
        "timeout",
    ),
    (
        "the propose timeout is removed",
        _in_propose("    timeout-minutes: 15\n", ""),
        "timeout",
    ),
    (
        "a second run cancels the first",
        _edit("cancel-in-progress: false", "cancel-in-progress: true"),
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
        "a third job is added",
        lambda t: t + "  other:\n    runs-on: ubuntu-24.04\n    steps: []\n",
        "jobs",
    ),
    (
        "the python version is another",
        _edit('python-version: "3.12"', 'python-version: "3.11"'),
        "python-version",
    ),
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
        _edit("testing,mcp,otel,a2a]", "testing]"),
        "install-line",
    ),
    (
        "mypy runs on a subset",
        _edit("python -m mypy -p fsm_llm", "python -m mypy -p fsm_llm.core"),
        "mypy-line",
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
]


class TestMutantsAreKilled:
    @pytest.mark.parametrize("label,mutate,code", MUTANTS, ids=[m[0] for m in MUTANTS])
    def test_each_mutant_is_caught_by_its_rule(
        self, label: str, mutate: Any, code: str
    ):
        mutated = mutate(_real_text())
        assert mutated != _real_text(), label
        assert code in workflow_problems(mutated), (label, workflow_problems(mutated))

    def test_there_are_many_mutants_and_many_rules(self):
        assert len(MUTANTS) >= 80
        assert len({m[2] for m in MUTANTS}) >= 40

    def test_every_code_the_checker_can_return_has_a_mutant(self):
        source = Path(__file__).read_text(encoding="utf-8")
        body = source.split("def workflow_problems", 1)[1].split("def _real_text", 1)[0]
        codes = set(re.findall(r'found\.add\("([a-z-]+)"\)', body)) | {"unreadable"}
        killed = {m[2] for m in MUTANTS}
        assert codes - killed - {"unreadable"} == set(), sorted(codes - killed)

    def test_the_real_text_is_the_only_clean_one(self):
        assert workflow_problems(_real_text()) == set()
        assert workflow_problems("") != set()
        assert workflow_problems("not: [valid") == {"unreadable"}
