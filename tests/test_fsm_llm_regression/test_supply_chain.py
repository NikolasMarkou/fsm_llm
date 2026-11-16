"""The supply-chain checks: a vulnerable dependency fails the build, and every action is pinned.

Plan d454fd62, step 109 (roadmap row 0.10). Nothing here reaches the network: the real
`pip-audit` runs in CI, and these tests read the files that make it run, as data.

* `supply_chain_problems(text)` reads `.github/workflows/python-package.yml`: the job
  `supply-chain` installs the package and audits the installed environment with
  `pip-audit --skip-editable`, then runs a CANARY (`pip-audit` over a fixture that pins
  vulnerable releases, which must exit 1 with a non-empty report), and the job
  `dependency-review` runs on pull requests only with `pull-requests: write` on that job alone.
* `canary_behaviour_problems(script, tmp_path)` runs the canary script of the workflow in bash
  against a stub `pip-audit`, so the polarity ("the step passes only when the audit FAILS") is
  tested on the shipped text, not grepped. A canary that passed on an unreachable advisory
  service would prove nothing, so exit 2, exit 0 and an empty report all fail the step.
* `cross_file_problems(files)` reads EVERY file under `.github/workflows` (a glob, so the
  workflows of steps 110 and 112 are covered the day they land): every `uses:` is a 40-hex SHA
  with a `# v..` comment, no `pull_request_target`, a top-level `permissions` that grants no
  write, and a job-level write only where `_WRITE_ALLOWED` names it. A later workflow that
  needs a write scope adds its line to that table in the same commit.
* `write_job_code_problems(files, waivers)` reads the same files for the rule of step 121.1
  (final REFLECT concern 2): a job that holds a write scope installs no package and runs no
  project code (no installer, interpreter, test runner or project script in a `run:` and no
  `setup-*` action), so a freshly released third-party package can never execute next to a
  repository write token. The one named waiver is the docs `deploy` job (`_INSTALL_WAIVERS`).
* `pin_problems(constraints, pyproject)`, `fixture_problems(...)` and `policy_problems(...)`
  read `constraints.txt` plus the `dev` extra, the canary fixture and `SECURITY.md`.
* Each checker returns a set of problem codes; the real files must return the empty set and each
  mutant (the real text with one edit) must return its own code: a check that cannot fail is
  not a check. A test fails when a code has no mutant.
* Only the standard library, `packaging` and `yaml.safe_load` are used (the helper functions are
  copied from `test_ci_workflow.py`: a test module never imports another).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOWS = _ROOT / ".github" / "workflows"
_CI = _WORKFLOWS / "python-package.yml"
_FIXTURE_REL = "tests/fixtures/canary-vulnerable-pins.txt"
_FIXTURE = _ROOT / _FIXTURE_REL
_SECURITY = _ROOT / "SECURITY.md"
_REFRESH_WORKFLOW = _WORKFLOWS / "refresh-pins.yml"

_AUDIT = "pip-audit --skip-editable"
_LINTER_PINS = ("zizmor", "actionlint-py")

#: Job-level `write` grants the repository allows, keyed by (file name, job id). Everything else
#: is read-only. A workflow that needs a write scope (the release of step 112, the pin
#: refresh of step 110) registers its line here, in the commit that adds the workflow, and the
#: policy sentence in SECURITY.md says why.
_WRITE_ALLOWED: dict[tuple[str, str], set[str]] = {
    # mike pushes the `gh-pages` branch (tag builds only).
    ("docs.yml", "deploy"): {"contents"},
    # the failure comment of the dependency review (pull requests only).
    ("python-package.yml", "dependency-review"): {"pull-requests"},
    # the weekly pin refresh pushes a branch and opens a pull request, in the `propose` job that
    # runs only after the `test` job (read-only, no credentials) passed every gate; `propose`
    # installs nothing (step 110, split at step 121.1; test_refresh_workflow.py pins both jobs).
    ("refresh-pins.yml", "propose"): {"contents", "pull-requests"},
    # the release workflow (step 112; test_release_workflow.py pins the shape): the OIDC token that
    # PyPI trusted publishing exchanges, on the `publish` job only, and the GitHub release that
    # a pushed tag creates, on the `github-release` job only.
    ("release.yml", "publish"): {"id-token"},
    ("release.yml", "github-release"): {"contents"},
}

#: Jobs that hold a write scope AND run an installer or project code, with the reason. The rule
#: is "a job with a write scope runs no installer and no project code" (step 121.1). The docs
#: `deploy` job is the one waiver: it runs `mike` from the `docs` dependency group, which
#: `constraints.txt` pins exactly, which the weekly pin refresh never moves (it changes litellm
#: and mcp only) and which only a reviewed commit can change, so no new third-party release
#: reaches it unreviewed. Splitting it would need a second job that pushes a built site.
_INSTALL_WAIVERS: dict[tuple[str, str], str] = {
    ("docs.yml", "deploy"): (
        "installs the docs tools, which are pinned exactly in constraints.txt and never moved "
        "by the pin refresh, so no new third-party release reaches it unreviewed"
    ),
}

_SHA_USES = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+@[0-9a-f]{40}$")
_USES_LINE = re.compile(r"^\s*(?:-\s*)?uses:\s*(?P<ref>\S+)(?P<rest>.*)$", re.MULTILINE)
_TAG_COMMENT = re.compile(r"^\s*#\s*v\d+(?:\.\d+)+(?:-[\w.]+)?\s*$")
_INSTALL = re.compile(r'pip install -c constraints\.txt -e "\.\[(?P<extras>[^\]]+)\]"')
_PIN_LINE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([0-9][A-Za-z0-9.!+_-]*)$")


# --------------------------------------------------------------------------------------
# Reading the files
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


def _normalized(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _workflow_texts() -> dict[str, str]:
    """Every workflow file, by name (a glob: a new workflow is covered the day it lands)."""
    paths = sorted([*_WORKFLOWS.glob("*.yml"), *_WORKFLOWS.glob("*.yaml")])
    return {p.name: p.read_text(encoding="utf-8") for p in paths}


def _real_ci() -> str:
    return _CI.read_text(encoding="utf-8")


def _install_line(job: dict[str, Any]) -> list[str]:
    lines = []
    for step in _steps(job):
        for line in str(step.get("run", "")).splitlines():
            if "pip install -c constraints.txt -e" in line:
                lines.append(line.strip())
    return lines


def _index_of(job: dict[str, Any], needle: str) -> int:
    for i, step in enumerate(_steps(job)):
        if needle in str(step.get("run", "")):
            return i
    return -1


def _canary_step(job: dict[str, Any]) -> dict[str, Any] | None:
    for step in _steps(job):
        if _FIXTURE_REL in str(step.get("run", "")):
            return step
    return None


def _constraint_pins(text: str) -> dict[str, list[str]]:
    pins: dict[str, list[str]] = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        match = _PIN_LINE.fullmatch(line)
        assert match, f"constraints.txt line is not an exact pin: {raw!r}"
        pins.setdefault(_normalized(match.group(1)), []).append(match.group(2))
    return pins


# --------------------------------------------------------------------------------------
# The checkers
# --------------------------------------------------------------------------------------


def supply_chain_problems(text: str) -> set[str]:
    """Rule codes broken by the `supply-chain` and `dependency-review` jobs of the CI workflow."""
    found: set[str] = set()
    try:
        wf = _load(text)
    except Exception:
        return {"unreadable"}
    jobs = wf.get("jobs") or {}
    sc = jobs.get("supply-chain")
    dr = jobs.get("dependency-review")

    # The job that audits the installed environment.
    if not isinstance(sc, dict):
        found.add("sc-job")
        sc = {}
    else:
        if sc.get("runs-on") != "ubuntu-24.04":
            found.add("sc-runner")
        value = sc.get("timeout-minutes")
        if not isinstance(value, int) or isinstance(value, bool) or value < 10:
            found.add("sc-timeout")
        installs = _install_line(sc)
        test_installs = _install_line(jobs.get("test") or {})
        if len(installs) != 1 or installs != test_installs:
            found.add("sc-install")
        audits = [
            s
            for s in _steps(sc)
            if str(s.get("run", "")).strip().startswith("pip-audit")
            and "-r " not in str(s.get("run", ""))
        ]
        if len(audits) != 1:
            found.add("sc-audit")
        else:
            run = str(audits[0]["run"]).strip()
            if "--skip-editable" not in run:
                found.add("sc-skip-editable")
            if _index_of(sc, "pip-audit") < _index_of(
                sc, "pip install -c constraints.txt -e"
            ):
                found.add("sc-audit")
            if "||" in run or audits[0].get("continue-on-error") or "if" in audits[0]:
                found.add("sc-soft")
        canary = _canary_step(sc)
        if canary is None:
            found.add("sc-canary")
        else:
            run = str(canary["run"])
            if "--no-deps" not in run or "--disable-pip" not in run:
                found.add("sc-canary-args")
            if canary.get("continue-on-error") or "if" in canary:
                found.add("sc-soft")
            if "pip-audit" not in run:
                found.add("sc-canary")

    # Dependency review: pull requests only, high severity, a write scope on this job alone.
    if not isinstance(dr, dict):
        found.add("dr-job")
        dr = {}
    else:
        condition = str(dr.get("if", "")).replace("${{", "").replace("}}", "").strip()
        if condition != "github.event_name == 'pull_request'":
            found.add("dr-condition")
        if dr.get("permissions") != {"contents": "read", "pull-requests": "write"}:
            found.add("dr-permissions")
        actions = [
            s
            for s in _steps(dr)
            if str(s.get("uses", "")).startswith("actions/dependency-review-action@")
        ]
        if len(actions) != 1:
            found.add("dr-action")
        else:
            with_ = actions[0].get("with") or {}
            if with_.get("fail-on-severity") != "high":
                found.add("dr-severity")
            if str(with_.get("warn-only", "false")).lower() == "true":
                found.add("dr-soft")
            if str(with_.get("vulnerability-check", "true")).lower() == "false":
                found.add("dr-soft")
            if actions[0].get("continue-on-error") or dr.get("continue-on-error"):
                found.add("dr-soft")
    for name, job in jobs.items():
        if name != "dependency-review" and "permissions" in job:
            found.add("write-scope")
    return found


def canary_behaviour_problems(script: str, tmp_path: Path) -> set[str]:
    """Run the canary script in bash against a stub `pip-audit` and judge its exit code.

    The step must pass only when the audit exits 1 AND its report lists a vulnerability:
    a clean audit, a service failure (any other exit code) and an empty report all fail it.
    """
    found: set[str] = set()
    scenarios = [
        # (stub exit code, report written, the script must succeed)
        (
            1,
            '{"dependencies": [{"name": "x", "version": "1", "vulns": [{"id": "A-1"}]}]}',
            True,
        ),
        (
            0,
            '{"dependencies": [{"name": "x", "version": "1", "vulns": [{"id": "A-1"}]}]}',
            False,
        ),
        (
            2,
            '{"dependencies": [{"name": "x", "version": "1", "vulns": [{"id": "A-1"}]}]}',
            False,
        ),
        (0, '{"dependencies": [{"name": "x", "version": "1", "vulns": []}]}', False),
        (2, None, False),
        (1, None, False),
        (1, '{"dependencies": [{"name": "x", "version": "1", "vulns": []}]}', False),
    ]
    for number, (code, report, must_pass) in enumerate(scenarios):
        work = tmp_path / f"scenario-{number}"
        (work / "bin").mkdir(parents=True)
        (work / "tmp").mkdir()
        stub = work / "bin" / "pip-audit"
        write_report = ""
        if report is not None:
            write_report = f"printf '%s' '{report}' > \"$out\"\n"
        stub.write_text(
            "#!/usr/bin/env bash\n"
            'out=""\n'
            'while [ "$#" -gt 0 ]; do\n'
            '  if [ "$1" = "--output" ]; then out="$2"; fi\n'
            "  shift\n"
            "done\n"
            f"{write_report}exit {code}\n",
            encoding="utf-8",
        )
        stub.chmod(0o755)
        path = work / "run.sh"
        path.write_text(script, encoding="utf-8")
        env = {
            "PATH": os.pathsep.join(
                [
                    str(work / "bin"),
                    str(Path(sys.executable).parent),
                    os.environ["PATH"],
                ]
            ),
            "RUNNER_TEMP": str(work / "tmp"),
            "HOME": str(work),
        }
        proc = subprocess.run(
            ["bash", str(path)],
            cwd=_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if (proc.returncode == 0) != must_pass:
            found.add("canary-polarity")
    return found


def cross_file_problems(files: dict[str, str]) -> set[str]:
    """Rule codes broken by any workflow file of `files` (name to text)."""
    found: set[str] = set()
    if not files:
        return {"no-workflows"}
    for name, text in files.items():
        try:
            wf = _load(text)
        except Exception:
            found.add("unreadable")
            continue
        jobs = wf.get("jobs") or {}
        if "pull_request_target" in "\n".join(_code_lines(text)):
            found.add("pull-request-target")

        # Every `uses:` is a SHA with its release tag; none is hidden from the reader.
        parsed = [
            str(s["uses"]) for job in jobs.values() for s in _steps(job) if "uses" in s
        ] + [str(job["uses"]) for job in jobs.values() if "uses" in job]
        lines = list(_USES_LINE.finditer(text))
        if len(lines) != len(parsed):
            found.add("uses-count")
        for match in lines:
            ref = match["ref"]
            if ref.startswith("./"):
                continue  # an action that lives in this repository
            if not _SHA_USES.match(ref):
                found.add("action-pin")
            if not _TAG_COMMENT.match(match["rest"]):
                found.add("action-tag-comment")

        # A read-only token at the top; a job-level write only where the table names it.
        top = wf.get("permissions")
        if not isinstance(top, dict):
            found.add("top-permissions-missing")
        elif any(level == "write" for level in top.values()):
            found.add("top-permissions-write")
        for job_id, job in jobs.items():
            granted = job.get("permissions")
            if granted is None:
                continue
            if isinstance(granted, str):
                if granted != "read-all":
                    found.add("job-write")
                continue
            allowed = _WRITE_ALLOWED.get((name, job_id), set())
            for scope, level in granted.items():
                if level == "write" and scope not in allowed:
                    found.add("job-write")
    return found


#: Commands that install a package or run an interpreter, a test runner or a project script.
_CODE_COMMANDS = re.compile(
    r"^(?:pip[\d.]*|pipx|uv|uvx|poetry|conda|npm|npx|yarn|pnpm|cargo|gem|bundle|go|"
    r"python[\d.]*|pytest|mypy|ruff|tox|nox|make|bash|sh|zsh|node|source)$"
)
_WRAPPERS = frozenset(
    {"sudo", "env", "time", "nice", "ionice", "exec", "xargs", "command", "timeout"}
)
_KEYWORDS = frozenset({"if", "then", "else", "elif", "fi", "do", "done", "while", "!"})


def runs_code(script: str) -> bool:
    """True when a shell `script` names an installer, an interpreter or a project script.

    Best effort, copied from `test_refresh_workflow.py` (a test module never imports another):
    heredoc bodies, single-quoted and plain double-quoted strings are dropped, the rest is cut at
    `; & | ( ) { } $(` and newlines, and the first word of each piece (after keywords, wrappers
    such as `nice` or `sudo` with their options, and `NAME=value` assignments) is judged.
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
    for piece in text.split("\n"):
        words = piece.split()
        while words:
            if words[0] in _WRAPPERS:
                words = words[1:]
                while words and (words[0].startswith("-") or words[0].isdigit()):
                    words = words[1:]
            elif words[0] in _KEYWORDS or re.fullmatch(r"[A-Za-z_]\w*=\S*", words[0]):
                words = words[1:]
            else:
                break
        if not words or words[0].startswith("#"):
            continue
        if _CODE_COMMANDS.match(words[0]) or words[0].startswith(("./", "scripts/")):
            return True
    return False


def _job_writes(job: dict[str, Any]) -> bool:
    granted = job.get("permissions")
    if isinstance(granted, str):
        return granted == "write-all"
    return isinstance(granted, dict) and "write" in granted.values()


def _job_runs_code(job: dict[str, Any]) -> bool:
    for step in _steps(job):
        if str(step.get("uses", "")).startswith("actions/setup-"):
            return True
        if isinstance(step.get("run"), str) and runs_code(step["run"]):
            return True
    return False


def write_job_code_problems(
    files: dict[str, str], waivers: dict[tuple[str, str], str]
) -> set[str]:
    """Rule codes for write-scope jobs that run code, and for waivers that no longer apply."""
    found: set[str] = set()
    named: set[tuple[str, str]] = set()
    for name, text in files.items():
        try:
            jobs = _load(text).get("jobs") or {}
        except Exception:
            found.add("unreadable")
            continue
        for job_id, job in jobs.items():
            if not _job_writes(job):
                continue
            runs = _job_runs_code(job)
            if (name, job_id) in waivers:
                named.add((name, job_id))
                if not runs:
                    found.add("waiver-unused")
            elif runs:
                found.add("write-job-runs-code")
    if set(waivers) - named:
        found.add("waiver-unused")
    return found


def pin_problems(constraints: str, pyproject: str) -> set[str]:
    """Rule codes for the pins of `pip-audit` and of the two workflow linters."""
    found: set[str] = set()
    pins = _constraint_pins(constraints)
    data = tomllib.loads(pyproject)
    dev = [Requirement(r) for r in data["project"]["optional-dependencies"]["dev"]]
    mine = [r for r in dev if _normalized(r.name) == "pip-audit"]
    if len(pins.get("pip-audit", [])) != 1:
        found.add("audit-pin-missing")
    if len(mine) != 1:
        found.add("audit-dev-missing")
    else:
        spec: SpecifierSet = mine[0].specifier
        if not any(s.operator in {"<", "<=", "==", "~="} for s in spec):
            found.add("audit-dev-unbounded")
        elif len(pins.get("pip-audit", [])) == 1 and not spec.contains(
            Version(pins["pip-audit"][0]), prereleases=True
        ):
            found.add("audit-pin-outside")
    for tool in _LINTER_PINS:
        if len(pins.get(tool, [])) != 1:
            found.add("linter-pin-missing")
    return found


def fixture_problems(text: str, constraints: str) -> set[str]:
    """Rule codes for the canary fixture (pins of vulnerable releases, never installed)."""
    found: set[str] = set()
    pinned: dict[str, str] = {}
    comments = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped.startswith("#"):
            comments.append(stripped.lower())
            continue
        if not stripped:
            continue
        match = _PIN_LINE.fullmatch(stripped)
        if match is None:
            found.add("fixture-not-a-pin")
            continue
        pinned[_normalized(match.group(1))] = match.group(2)
    if not pinned:
        found.add("fixture-no-pin")
    joined = " ".join(comments)
    if "never" not in joined or "canary" not in joined:
        found.add("fixture-no-warning")
    real = _constraint_pins(constraints)
    if any(real.get(name) == [version] for name, version in pinned.items()):
        found.add("fixture-collides")
    return found


_POLICY_PHRASES: dict[str, tuple[str, ...]] = {
    "policy-audit": ("pip-audit", "canary", "supply-chain"),
    "policy-actions-pinned": ("commit sha", "pull_request_target"),
    "policy-dependency-review": ("dependency review", "pull request"),
    "policy-linters": ("zizmor", "actionlint"),
    "policy-pth": (".pth",),
    "policy-litellm": ("1.82.7", "1.82.8"),
    "policy-write-scopes": ("pull-requests", "contents", "id-token"),
    "policy-range-rule": (
        "no declared range admits a major version other than the tested one",
        "setuptools>=77,<85",
        "first version component",
    ),
    "policy-write-jobs-run-no-code": (
        "no job that holds a write scope installs a package",
        "docs `deploy`",
        "pinned exactly in `constraints.txt`",
    ),
    "policy-trusted-publishing": (
        "trusted publishing",
        "release.yml",
        "attestations",
        "no long-lived",
        "twine upload",
    ),
    "policy-release-build-once": ("built once", "twine check --strict"),
    "policy-sbom": ("sbom", "cyclonedx 1.6", "github release"),
}


def policy_problems(text: str, refresh_exists: bool) -> set[str]:
    """Rule codes for the supply-chain section of SECURITY.md.

    The refresh workflow of step 110 is described only once it exists, and must be described
    once it does (its file name, the cooldown and the 7-day age): a policy sentence for a job
    that is not there would be a false claim.
    """
    found: set[str] = set()
    low = text.lower()
    for code, phrases in _POLICY_PHRASES.items():
        if not all(p.lower() in low for p in phrases):
            found.add(code)
    mentions_refresh = "cooldown" in low or "refresh-pins" in low
    if refresh_exists and not all(
        phrase in low
        for phrase in ("`propose`", "`test` job", "artifact", "no credentials")
    ):
        found.add("policy-refresh-split")
    if mentions_refresh != refresh_exists:
        found.add("policy-refresh-claim")
    if refresh_exists and not all(
        phrase in low for phrase in ("refresh-pins", "cooldown", "7-day")
    ):
        found.add("policy-refresh-claim")
    return found


# --------------------------------------------------------------------------------------
# The real files
# --------------------------------------------------------------------------------------


class TestTheSupplyChainJob:
    def test_the_workflow_breaks_no_supply_chain_rule(self):
        assert supply_chain_problems(_real_ci()) == set()

    def test_the_audit_and_the_canary_are_steps_of_one_job_after_the_install(self):
        job = _load(_real_ci())["jobs"]["supply-chain"]
        assert _index_of(job, _AUDIT) > _index_of(
            job, "pip install -c constraints.txt -e"
        )
        canary = _canary_step(job)
        assert canary is not None
        assert "pip-audit -r " + _FIXTURE_REL in canary["run"]

    def test_the_audit_reads_the_installed_environment_and_skips_only_the_editable_package(
        self,
    ):
        job = _load(_real_ci())["jobs"]["supply-chain"]
        runs = [str(s["run"]).strip() for s in _steps(job) if "run" in s]
        assert _AUDIT in runs

    def test_the_job_installs_the_same_environment_as_the_matrix(self):
        jobs = _load(_real_ci())["jobs"]
        assert _install_line(jobs["supply-chain"]) == _install_line(jobs["test"])
        assert len(_install_line(jobs["supply-chain"])) == 1

    def test_dependency_review_runs_on_pull_requests_with_the_only_write_scope(self):
        jobs = _load(_real_ci())["jobs"]
        job = jobs["dependency-review"]
        assert job["if"] == "${{ github.event_name == 'pull_request' }}"
        assert job["permissions"] == {"contents": "read", "pull-requests": "write"}
        for name, other in jobs.items():
            if name != "dependency-review":
                assert "permissions" not in other, name
        assert _load(_real_ci())["permissions"] == {"contents": "read"}

    def test_the_review_fails_a_pull_request_that_adds_a_high_advisory(self):
        job = _load(_real_ci())["jobs"]["dependency-review"]
        step = next(
            s for s in _steps(job) if str(s.get("uses", "")).startswith("actions/dep")
        )
        assert step["with"]["fail-on-severity"] == "high"
        assert "warn-only" not in step["with"]

    def test_ci_cancels_a_superseded_pull_request_run_only(self):
        wf = _load(_real_ci())
        assert (
            wf["concurrency"]["group"] == "ci-${{ github.workflow }}-${{ github.ref }}"
        )
        assert (
            wf["concurrency"]["cancel-in-progress"]
            == "${{ github.event_name == 'pull_request' }}"
        )


class TestTheCanaryScript:
    def _script(self) -> str:
        job = _load(_real_ci())["jobs"]["supply-chain"]
        canary = _canary_step(job)
        assert canary is not None
        return str(canary["run"])

    def test_the_shipped_script_passes_only_when_the_audit_fails_with_a_report(
        self, tmp_path: Path
    ):
        assert canary_behaviour_problems(self._script(), tmp_path) == set()

    def test_a_clean_audit_fails_the_step(self, tmp_path: Path):
        # the canary must be red on a clean exit: "the audit passed" is the failure
        script = self._script()
        scenario = tmp_path / "clean"
        (scenario / "bin").mkdir(parents=True)
        (scenario / "tmp").mkdir()
        stub = scenario / "bin" / "pip-audit"
        stub.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        stub.chmod(0o755)
        (scenario / "run.sh").write_text(script, encoding="utf-8")
        proc = subprocess.run(
            ["bash", str(scenario / "run.sh")],
            cwd=_ROOT,
            env={
                "PATH": os.pathsep.join(
                    [
                        str(scenario / "bin"),
                        str(Path(sys.executable).parent),
                        os.environ["PATH"],
                    ]
                ),
                "RUNNER_TEMP": str(scenario / "tmp"),
                "HOME": str(scenario),
            },
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert proc.returncode != 0
        assert "expected 1" in proc.stdout + proc.stderr


class TestTheFixtureAndThePins:
    def test_the_fixture_the_canary_audits_exists(self):
        assert _FIXTURE.is_file()

    def test_the_fixture_pins_vulnerable_releases_and_warns(self):
        text = _FIXTURE.read_text(encoding="utf-8")
        constraints = (_ROOT / "constraints.txt").read_text(encoding="utf-8")
        assert fixture_problems(text, constraints) == set()
        pins = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
        assert len(pins) >= 2, (
            "several advisories: the canary stays red if one is withdrawn"
        )

    def test_the_fixture_is_only_ever_read_by_pip_audit(self):
        # An install of the fixture would put known-vulnerable code into an environment.
        files = [
            _ROOT / "Makefile",
            _ROOT / "tox.ini",
            _ROOT / "pyproject.toml",
            _ROOT / "constraints.txt",
            *sorted(_WORKFLOWS.glob("*.y*ml")),
        ]
        for path in files:
            if not path.is_file():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if "canary-vulnerable-pins" in line and not line.lstrip().startswith(
                    "#"
                ):
                    assert "pip-audit" in line, (path.name, line)

    def test_pip_audit_and_the_linters_are_pinned_and_pip_audit_is_a_bounded_dev_tool(
        self,
    ):
        constraints = (_ROOT / "constraints.txt").read_text(encoding="utf-8")
        pyproject = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert pin_problems(constraints, pyproject) == set()

    def test_the_lint_job_installs_the_linters_from_the_pins(self):
        lint = _load(_real_ci())["jobs"]["lint"]
        runs = [str(s["run"]).strip() for s in _steps(lint) if "run" in s]
        assert "pip install -c constraints.txt zizmor actionlint-py" in runs


class TestEveryWorkflowFile:
    def test_the_glob_finds_the_four_workflows_of_this_plan_so_far(self):
        assert {
            "python-package.yml",
            "docs.yml",
            "refresh-pins.yml",
            "release.yml",
        } <= set(_workflow_texts())

    def test_no_workflow_breaks_a_cross_file_rule(self):
        assert cross_file_problems(_workflow_texts()) == set()

    def test_every_uses_line_in_every_workflow_is_a_sha_with_a_release_tag(self):
        for name, text in _workflow_texts().items():
            matches = list(_USES_LINE.finditer(text))
            assert matches, name
            for match in matches:
                if match["ref"].startswith("./"):
                    continue
                assert _SHA_USES.match(match["ref"]), (name, match["ref"])
                assert _TAG_COMMENT.match(match["rest"]), (name, match["rest"])

    def test_one_action_has_one_pin_across_the_workflows(self):
        pins: dict[str, set[str]] = {}
        for text in _workflow_texts().values():
            for match in _USES_LINE.finditer(text):
                action, _, sha = match["ref"].partition("@")
                pins.setdefault(action, set()).add(sha)
        assert {a: s for a, s in pins.items() if len(s) != 1} == {}

    def test_no_workflow_uses_pull_request_target(self):
        for name, text in _workflow_texts().items():
            assert "pull_request_target" not in "\n".join(_code_lines(text)), name

    def test_every_workflow_has_a_read_only_top_level_token(self):
        for name, text in _workflow_texts().items():
            top = _load(text).get("permissions")
            assert isinstance(top, dict), name
            assert "write" not in top.values(), name

    def test_the_write_allowlist_names_grants_that_exist(self):
        texts = _workflow_texts()
        for (name, job_id), scopes in _WRITE_ALLOWED.items():
            job = _load(texts[name])["jobs"][job_id]
            granted = {s for s, level in job["permissions"].items() if level == "write"}
            assert granted == scopes, (name, job_id)

    def test_no_job_outside_the_allowlist_grants_a_write(self):
        for name, text in _workflow_texts().items():
            for job_id, job in (_load(text).get("jobs") or {}).items():
                perms = job.get("permissions") or {}
                writes = {s for s, level in perms.items() if level == "write"}
                assert writes <= _WRITE_ALLOWED.get((name, job_id), set()), (
                    name,
                    job_id,
                )


class TestWriteJobsRunNoCode:
    def test_no_write_job_installs_a_package_or_runs_project_code_except_the_waiver(
        self,
    ):
        assert write_job_code_problems(_workflow_texts(), _INSTALL_WAIVERS) == set()

    def test_the_one_waiver_is_the_docs_deploy_job_and_says_why(self):
        assert set(_INSTALL_WAIVERS) == {("docs.yml", "deploy")}
        reason = _INSTALL_WAIVERS[("docs.yml", "deploy")]
        assert "pinned exactly in constraints.txt" in reason
        assert "never moved by the pin refresh" in reason

    def test_the_waived_job_really_installs_and_really_writes(self):
        job = _load(_workflow_texts()["docs.yml"])["jobs"]["deploy"]
        assert _job_writes(job) and _job_runs_code(job)

    def test_the_waiver_is_stated_in_the_workflow_and_in_security_md(self):
        assert "_INSTALL_WAIVERS" in _workflow_texts()["docs.yml"]
        assert "docs `deploy`" in _SECURITY.read_text(encoding="utf-8")

    def test_the_docs_tools_are_all_pinned_exactly_and_the_refresh_moves_only_two_packages(
        self,
    ):
        pins = _constraint_pins((_ROOT / "constraints.txt").read_text(encoding="utf-8"))
        groups = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
            "dependency-groups"
        ]["docs"]
        for text in groups:
            assert len(pins[_normalized(Requirement(text).name)]) == 1, text
        script = (_ROOT / "scripts" / "refresh_pins.py").read_text(encoding="utf-8")
        assert re.search(r'^TRACKED = \("litellm", "mcp"\)$', script, re.MULTILINE), (
            "the refresh no longer moves litellm and mcp only: revisit the waiver"
        )

    def test_the_workflows_that_write_and_must_run_no_code_are_the_expected_ones(self):
        writers = {
            (name, job_id)
            for name, text in _workflow_texts().items()
            for job_id, job in (_load(text).get("jobs") or {}).items()
            if _job_writes(job)
        }
        assert writers == {
            ("docs.yml", "deploy"),
            ("python-package.yml", "dependency-review"),
            ("refresh-pins.yml", "propose"),
            ("release.yml", "publish"),
            ("release.yml", "github-release"),
        }


class TestSecurityPolicy:
    def test_security_md_states_the_policy_that_exists(self):
        text = _SECURITY.read_text(encoding="utf-8")
        assert (
            policy_problems(text, refresh_exists=_REFRESH_WORKFLOW.is_file()) == set()
        )

    def test_security_md_keeps_the_litellm_incident_and_the_pth_audit(self):
        text = _SECURITY.read_text(encoding="utf-8")
        assert "1.82.7" in text and ("audit_pth" in text or "make audit" in text)


# --------------------------------------------------------------------------------------
# Mutants: the real text with one edit each; every rule must catch its own
# --------------------------------------------------------------------------------------


def _edit(old: str, new: str, count: int = 1) -> Callable[[str], str]:
    def mutate(text: str) -> str:
        assert text.count(old) >= count, f"{old!r} not found {count} time(s)"
        return text.replace(old, new, count)

    return mutate


def _edit_all(old: str, new: str) -> Callable[[str], str]:
    def mutate(text: str) -> str:
        assert old in text, f"{old!r} not found"
        return text.replace(old, new)

    return mutate


def _edit_nth(old: str, new: str, nth: int) -> Callable[[str], str]:
    def mutate(text: str) -> str:
        parts = text.split(old)
        assert len(parts) > nth + 1, f"{old!r} has no occurrence number {nth}"
        return old.join(parts[: nth + 1]) + new + old.join(parts[nth + 1 :])

    return mutate


def _drop_step(name: str, nth: int = 0) -> Callable[[str], str]:
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


def _drop_job(job_id: str) -> Callable[[str], str]:
    def mutate(text: str) -> str:
        pattern = re.compile(
            rf"^  {re.escape(job_id)}:\n(?:    .*\n|\n)*?(?=  [\w-]+:\n|\Z)",
            re.MULTILINE,
        )
        m = pattern.search(text)
        assert m, f"job {job_id!r} not found"
        return text[: m.start()] + text[m.end() :]

    return mutate


_CANARY_NAME = "Prove that the audit fails on a known-vulnerable pin (canary)"

WORKFLOW_MUTANTS: list[tuple[str, Callable[[str], str], str]] = [
    ("the supply-chain job removed", _drop_job("supply-chain"), "sc-job"),
    (
        "the supply-chain job on ubuntu-latest",
        _edit_nth("runs-on: ubuntu-24.04", "runs-on: ubuntu-latest", 1),
        "sc-runner",
    ),
    (
        "the supply-chain job without a timeout",
        _edit_nth("    timeout-minutes: 20\n", "", 1),
        "sc-timeout",
    ),
    (
        "the supply-chain install loses the mcp extra",
        _edit_nth(",mcp]", "]", 2),
        "sc-install",
    ),
    (
        "the audit step removed",
        _drop_step("Audit the installed environment for known vulnerabilities"),
        "sc-audit",
    ),
    (
        "the audit runs without --skip-editable",
        _edit("run: pip-audit --skip-editable", "run: pip-audit"),
        "sc-skip-editable",
    ),
    (
        "the audit failure swallowed",
        _edit(
            "run: pip-audit --skip-editable", "run: pip-audit --skip-editable || true"
        ),
        "sc-soft",
    ),
    (
        "the audit marked continue-on-error",
        _edit(
            "        run: pip-audit --skip-editable\n",
            "        continue-on-error: true\n        run: pip-audit --skip-editable\n",
        ),
        "sc-soft",
    ),
    ("the canary removed", _drop_step(_CANARY_NAME), "sc-canary"),
    (
        "the canary audits without --no-deps",
        _edit(" --no-deps --disable-pip", " --disable-pip"),
        "sc-canary-args",
    ),
    (
        "the canary audits without --disable-pip",
        _edit(" --no-deps --disable-pip", " --no-deps"),
        "sc-canary-args",
    ),
    (
        "the canary marked continue-on-error",
        _edit(
            f"      - name: {_CANARY_NAME}\n",
            f"      - name: {_CANARY_NAME}\n        continue-on-error: true\n",
        ),
        "sc-soft",
    ),
    (
        "the canary runs only on pushes",
        _edit(
            f"      - name: {_CANARY_NAME}\n",
            f"      - name: {_CANARY_NAME}\n        if: github.event_name == 'push'\n",
        ),
        "sc-soft",
    ),
    (
        "the canary aimed at another file",
        _edit(_FIXTURE_REL, "tests/fixtures/other.txt"),
        "sc-canary",
    ),
    ("dependency review removed", _drop_job("dependency-review"), "dr-job"),
    (
        "dependency review runs on pushes too",
        _edit("    if: ${{ github.event_name == 'pull_request' }}\n", ""),
        "dr-condition",
    ),
    (
        "dependency review without its write scope",
        _edit("      pull-requests: write\n", "      pull-requests: read\n"),
        "dr-permissions",
    ),
    (
        "dependency review fails only on critical",
        _edit("fail-on-severity: high", "fail-on-severity: critical"),
        "dr-severity",
    ),
    (
        "dependency review only warns",
        _edit(
            "fail-on-severity: high",
            "fail-on-severity: high\n          warn-only: true",
        ),
        "dr-soft",
    ),
    (
        "the dependency review action removed",
        _edit("actions/dependency-review-action@", "actions/some-other-action@"),
        "dr-action",
    ),
    (
        "pull-requests write on the wrong job",
        _edit(
            "  supply-chain:\n    runs-on: ubuntu-24.04\n",
            "  supply-chain:\n    permissions:\n      pull-requests: write\n"
            "    runs-on: ubuntu-24.04\n",
        ),
        "write-scope",
    ),
    ("an unreadable file", lambda t: "jobs: [unterminated", "unreadable"),
]


class TestTheWorkflowMutantsAreKilled:
    @pytest.mark.parametrize(
        "label,mutate,code", WORKFLOW_MUTANTS, ids=[m[0] for m in WORKFLOW_MUTANTS]
    )
    def test_each_mutant_is_caught_by_its_rule(
        self, label: str, mutate: Callable[[str], str], code: str
    ):
        mutated = mutate(_real_ci())
        assert mutated != _real_ci()
        assert code in supply_chain_problems(mutated), (
            label,
            supply_chain_problems(mutated),
        )


def _canary_script() -> str:
    canary = _canary_step(_load(_real_ci())["jobs"]["supply-chain"])
    assert canary is not None
    return str(canary["run"])


CANARY_MUTANTS: list[tuple[str, Callable[[str], str]]] = [
    ("the polarity inverted", _edit("-ne 1", "-eq 1")),
    ("the exit code never judged", _edit('if [ "$status" -ne 1 ]', "if false")),
    (
        "the audit failure swallowed",
        _edit("|| status=$?", "|| true"),
    ),
    ("the report never read", _edit("if found < 1:", "if False:")),
    ("the report read from nowhere", _edit("sys.argv[1]", '"/nonexistent"')),
    ("the failure message but no exit", _edit("  exit 1\n", "  true\n")),
]


class TestTheCanaryScriptMutantsAreKilled:
    @pytest.mark.parametrize(
        "label,mutate", CANARY_MUTANTS, ids=[m[0] for m in CANARY_MUTANTS]
    )
    def test_each_mutant_has_the_wrong_polarity(
        self, label: str, mutate: Callable[[str], str], tmp_path: Path
    ):
        script = _canary_script()
        mutated = mutate(script)
        assert mutated != script
        assert "canary-polarity" in canary_behaviour_problems(mutated, tmp_path)


def _files() -> dict[str, str]:
    return _workflow_texts()


def _with(
    name: str, mutate: Callable[[str], str]
) -> Callable[[dict[str, str]], dict[str, str]]:
    def apply(files: dict[str, str]) -> dict[str, str]:
        out = dict(files)
        out[name] = mutate(out[name])
        return out

    return apply


def _add_file(name: str, text: str) -> Callable[[dict[str, str]], dict[str, str]]:
    def apply(files: dict[str, str]) -> dict[str, str]:
        return {**files, name: text}

    return apply


_NEW_WORKFLOW = (
    "name: New\non:\n  push:\n    branches: [main]\npermissions:\n  contents: read\n"
    "jobs:\n  one:\n    runs-on: ubuntu-24.04\n    steps:\n"
    "      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1\n"
)

CROSS_MUTANTS: list[tuple[str, Callable[[dict[str, str]], dict[str, str]], str]] = [
    (
        "a floating tag in python-package.yml",
        _with(
            "python-package.yml",
            lambda t: re.sub(
                r"actions/checkout@[0-9a-f]{40}", "actions/checkout@v7", t, count=1
            ),
        ),
        "action-pin",
    ),
    (
        "a floating tag in docs.yml",
        _with(
            "docs.yml",
            lambda t: re.sub(
                r"actions/setup-python@[0-9a-f]{40}",
                "actions/setup-python@v7",
                t,
                count=1,
            ),
        ),
        "action-pin",
    ),
    (
        "a truncated SHA in docs.yml",
        _with(
            "docs.yml",
            lambda t: re.sub(
                r"(actions/checkout@[0-9a-f]{39})[0-9a-f]", r"\1", t, count=1
            ),
        ),
        "action-pin",
    ),
    (
        "a SHA without its release tag in docs.yml",
        _with(
            "docs.yml",
            lambda t: re.sub(
                r"(actions/checkout@[0-9a-f]{40})\s+# v[\d.]+", r"\1", t, count=1
            ),
        ),
        "action-tag-comment",
    ),
    (
        "a new workflow file with an unpinned action",
        _add_file(
            "extra.yml",
            _NEW_WORKFLOW.replace(
                "3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1", "v7"
            ),
        ),
        "action-pin",
    ),
    (
        "a new workflow file without a top-level permissions block",
        _add_file(
            "extra.yml", _NEW_WORKFLOW.replace("permissions:\n  contents: read\n", "")
        ),
        "top-permissions-missing",
    ),
    (
        "a new workflow file with a write token at the top",
        _add_file(
            "extra.yml", _NEW_WORKFLOW.replace("contents: read", "contents: write")
        ),
        "top-permissions-write",
    ),
    (
        "a new workflow file on pull_request_target",
        _add_file(
            "extra.yml",
            _NEW_WORKFLOW.replace(
                "push:\n    branches: [main]", "pull_request_target:"
            ),
        ),
        "pull-request-target",
    ),
    (
        "pull_request_target added to docs.yml",
        _with(
            "docs.yml",
            _edit(
                "  workflow_dispatch:\n",
                "  workflow_dispatch:\n  pull_request_target:\n",
            ),
        ),
        "pull-request-target",
    ),
    (
        "a write scope on the wrong job",
        _with(
            "python-package.yml",
            _edit(
                "  supply-chain:\n    runs-on: ubuntu-24.04\n",
                "  supply-chain:\n    permissions:\n      pull-requests: write\n"
                "    runs-on: ubuntu-24.04\n",
            ),
        ),
        "job-write",
    ),
    (
        "id-token write on the docs deploy job",
        _with(
            "docs.yml",
            _edit(
                "      contents: write\n",
                "      contents: write\n      id-token: write\n",
            ),
        ),
        "job-write",
    ),
    (
        "id-token write on the release build job",
        _with(
            "release.yml",
            _edit(
                "  build:\n    runs-on: ubuntu-24.04\n",
                "  build:\n    permissions:\n      id-token: write\n"
                "    runs-on: ubuntu-24.04\n",
            ),
        ),
        "job-write",
    ),
    (
        "contents write on the release publish job",
        _with(
            "release.yml",
            _edit(
                "      id-token: write\n",
                "      id-token: write\n      contents: write\n",
            ),
        ),
        "job-write",
    ),
    (
        "id-token write on the github-release job",
        _with(
            "release.yml",
            _edit(
                "      contents: write\n",
                "      contents: write\n      id-token: write\n",
            ),
        ),
        "job-write",
    ),
    (
        "a floating tag on the publish action",
        _with(
            "release.yml",
            lambda t: re.sub(
                r"pypa/gh-action-pypi-publish@[0-9a-f]{40}",
                "pypa/gh-action-pypi-publish@release/v1",
                t,
            ),
        ),
        "action-pin",
    ),
    (
        "write-all on the docs build job",
        _with(
            "docs.yml",
            _edit(
                "  build:\n    runs-on: ubuntu-24.04\n",
                "  build:\n    permissions: write-all\n    runs-on: ubuntu-24.04\n",
            ),
        ),
        "job-write",
    ),
    (
        "a uses line hidden in a script of docs.yml",
        _with(
            "docs.yml",
            _edit(
                "          set -euo pipefail\n",
                "          set -euo pipefail\n          uses: actions/checkout@v7\n",
            ),
        ),
        "uses-count",
    ),
    ("no workflow at all", lambda files: {}, "no-workflows"),
    (
        "an unreadable workflow",
        _add_file("broken.yml", "jobs: [unterminated"),
        "unreadable",
    ),
]


class TestTheCrossFileMutantsAreKilled:
    @pytest.mark.parametrize(
        "label,mutate,code", CROSS_MUTANTS, ids=[m[0] for m in CROSS_MUTANTS]
    )
    def test_each_mutant_is_caught_by_its_rule(
        self,
        label: str,
        mutate: Callable[[dict[str, str]], dict[str, str]],
        code: str,
    ):
        mutated = mutate(_files())
        assert mutated != _files()
        assert code in cross_file_problems(mutated), (
            label,
            cross_file_problems(mutated),
        )

    def test_a_local_action_needs_no_sha(self):
        extra = _NEW_WORKFLOW + "      - uses: ./.github/actions/local\n"
        assert cross_file_problems({"extra.yml": extra}) == set()


def _constraints() -> str:
    return (_ROOT / "constraints.txt").read_text(encoding="utf-8")


def _pyproject() -> str:
    return (_ROOT / "pyproject.toml").read_text(encoding="utf-8")


PIN_MUTANTS: list[tuple[str, Callable[[str, str], tuple[str, str]], str]] = [
    (
        "the pip-audit pin removed",
        lambda c, p: (re.sub(r"^pip-audit==.*\n", "", c, flags=re.MULTILINE), p),
        "audit-pin-missing",
    ),
    (
        "pip-audit pinned twice",
        lambda c, p: (c + "pip-audit==2.9.0\n", p),
        "audit-pin-missing",
    ),
    (
        "pip-audit missing from the dev extra",
        lambda c, p: (c, re.sub(r'^\s*"pip-audit[^"]*",\n', "", p, flags=re.MULTILINE)),
        "audit-dev-missing",
    ),
    (
        "pip-audit in the dev extra without an upper bound",
        lambda c, p: (c, re.sub(r'"pip-audit[^"]*"', '"pip-audit>=2.9"', p)),
        "audit-dev-unbounded",
    ),
    (
        "the pip-audit pin outside its range",
        lambda c, p: (c, re.sub(r'"pip-audit[^"]*"', '"pip-audit>=1.0,<2"', p)),
        "audit-pin-outside",
    ),
    (
        "the zizmor pin removed",
        lambda c, p: (re.sub(r"^zizmor==.*\n", "", c, flags=re.MULTILINE), p),
        "linter-pin-missing",
    ),
    (
        "the actionlint-py pin removed",
        lambda c, p: (re.sub(r"^actionlint-py==.*\n", "", c, flags=re.MULTILINE), p),
        "linter-pin-missing",
    ),
]


class TestThePinMutantsAreKilled:
    @pytest.mark.parametrize(
        "label,mutate,code", PIN_MUTANTS, ids=[m[0] for m in PIN_MUTANTS]
    )
    def test_each_mutant_is_caught_by_its_rule(
        self, label: str, mutate: Any, code: str
    ):
        constraints, pyproject = mutate(_constraints(), _pyproject())
        assert (constraints, pyproject) != (_constraints(), _pyproject())
        assert code in pin_problems(constraints, pyproject), label


def _fixture_text() -> str:
    return _FIXTURE.read_text(encoding="utf-8")


FIXTURE_MUTANTS: list[tuple[str, Callable[[str], str], str]] = [
    (
        "no pinned requirement",
        lambda t: (
            "\n".join(ln for ln in t.splitlines() if ln.startswith("#") or not ln)
            + "\n"
        ),
        "fixture-no-pin",
    ),
    (
        "the warning removed",
        lambda t: (
            "\n".join(ln for ln in t.splitlines() if not ln.startswith("#")) + "\n"
        ),
        "fixture-no-warning",
    ),
    ("a floating requirement", lambda t: t + "jinja2>=2\n", "fixture-not-a-pin"),
    (
        "a pin equal to a real pin of constraints.txt",
        lambda t: t + "loguru==0.8.0\n",
        "fixture-collides",
    ),
]


class TestTheFixtureMutantsAreKilled:
    @pytest.mark.parametrize(
        "label,mutate,code", FIXTURE_MUTANTS, ids=[m[0] for m in FIXTURE_MUTANTS]
    )
    def test_each_mutant_is_caught_by_its_rule(
        self, label: str, mutate: Callable[[str], str], code: str
    ):
        mutated = mutate(_fixture_text())
        assert mutated != _fixture_text()
        assert code in fixture_problems(mutated, _constraints()), label


def _security() -> str:
    return _SECURITY.read_text(encoding="utf-8")


POLICY_MUTANTS: list[tuple[str, Callable[[str], str], str]] = [
    (
        "the pip-audit sentence removed",
        _edit_all("pip-audit", "an audit"),
        "policy-audit",
    ),
    ("the canary word removed", _edit_all("canary", "probe"), "policy-audit"),
    (
        "the SHA pin sentence removed",
        _edit_all("commit SHA", "commit"),
        "policy-actions-pinned",
    ),
    (
        "the pull_request_target sentence removed",
        _edit_all("pull_request_target", "a trigger"),
        "policy-actions-pinned",
    ),
    (
        "the dependency review sentence removed",
        _edit_all("ependency review", "ependency check"),
        "policy-dependency-review",
    ),
    ("zizmor removed", _edit_all("zizmor", "a linter"), "policy-linters"),
    ("the .pth audit removed", _edit_all(".pth", ".pyc"), "policy-pth"),
    (
        "the litellm incident versions removed",
        _edit_all("1.82.7", "1.82.x"),
        "policy-litellm",
    ),
    (
        "the write scopes left unexplained",
        _edit_all("pull-requests", "comments"),
        "policy-write-scopes",
    ),
    (
        "the id-token scope left unexplained",
        _edit_all("id-token", "oidc"),
        "policy-write-scopes",
    ),
    (
        "trusted publishing not described",
        _edit_all("rusted publishing", "rusted upload"),
        "policy-trusted-publishing",
    ),
    (
        "the release workflow not named",
        _edit_all("release.yml", "the release workflow"),
        "policy-trusted-publishing",
    ),
    (
        "attestations not described",
        _edit_all("attestations", "signatures"),
        "policy-trusted-publishing",
    ),
    (
        "the no-token claim removed",
        _edit_all("no long-lived", "a short"),
        "policy-trusted-publishing",
    ),
    (
        "the twine upload ban removed",
        _edit_all("twine upload", "an upload"),
        "policy-trusted-publishing",
    ),
    (
        "the build-once claim removed",
        _edit_all("built once", "built"),
        "policy-release-build-once",
    ),
    (
        "the strict distribution check removed",
        _edit_all("twine check --strict", "a check"),
        "policy-release-build-once",
    ),
    (
        "the SBOM not described",
        _edit_all("SBOM", "list"),
        "policy-sbom",
    ),
    (
        "the SBOM format not named",
        _edit_all("CycloneDX", "a"),
        "policy-sbom",
    ),
    (
        "the GitHub release not named",
        _edit_all("GitHub release", "tag page"),
        "policy-sbom",
    ),
    (
        "the range rule removed",
        _edit_all(
            "no declared range admits a major version other than the tested one",
            "ranges are bounded",
        ),
        "policy-range-rule",
    ),
    (
        "the build backend exception not named",
        _edit_all("setuptools>=77,<85", "setuptools"),
        "policy-range-rule",
    ),
    (
        "the first-component rule removed",
        _edit_all("first version component", "version"),
        "policy-range-rule",
    ),
    (
        "the write-job rule removed",
        _edit_all("no job that holds a write scope installs a package", "a job"),
        "policy-write-jobs-run-no-code",
    ),
    (
        "the docs waiver removed",
        _edit_all("docs `deploy`", "a deploy"),
        "policy-write-jobs-run-no-code",
    ),
    (
        "the waiver reason removed",
        _edit_all("pinned exactly in `constraints.txt`", "pinned"),
        "policy-write-jobs-run-no-code",
    ),
    (
        "the propose job not described",
        _edit_all("`propose`", "the second job"),
        "policy-refresh-split",
    ),
    (
        "the test job not described",
        _edit_all("`test` job", "first job"),
        "policy-refresh-split",
    ),
    (
        "the artifact not described",
        _edit_all("artifact", "file"),
        "policy-refresh-split",
    ),
    (
        "the credential-free test job not described",
        _edit_all("no credentials", "credentials"),
        "policy-refresh-split",
    ),
    (
        "the refresh described without its cooldown",
        _edit_all("cooldown", "delay"),
        "policy-refresh-claim",
    ),
    (
        "the refresh described without its file name",
        _edit_all("refresh-pins", "the refresh workflow"),
        "policy-refresh-claim",
    ),
    (
        "the refresh described without its age",
        _edit_all("7-day", "short"),
        "policy-refresh-claim",
    ),
    (
        "the refresh sentence removed",
        lambda t: "\n".join(
            ln
            for ln in t.splitlines()
            if "refresh-pins" not in ln and "cooldown" not in ln
        ),
        "policy-refresh-claim",
    ),
]


class TestThePolicyMutantsAreKilled:
    @pytest.mark.parametrize(
        "label,mutate,code", POLICY_MUTANTS, ids=[m[0] for m in POLICY_MUTANTS]
    )
    def test_each_mutant_is_caught_by_its_rule(
        self, label: str, mutate: Callable[[str], str], code: str
    ):
        mutated = mutate(_security())
        assert mutated != _security()
        assert code in policy_problems(mutated, refresh_exists=True), label

    def test_a_refresh_workflow_that_exists_must_be_described(self):
        text = _security()
        assert policy_problems(text, refresh_exists=True) == set()
        assert "policy-refresh-claim" in policy_problems(
            "\n".join(
                ln
                for ln in text.splitlines()
                if "refresh-pins" not in ln and "cooldown" not in ln
            ),
            refresh_exists=True,
        )

    def test_a_refresh_described_while_no_workflow_exists_is_a_false_claim(self):
        assert "policy-refresh-claim" in policy_problems(
            _security(), refresh_exists=False
        )
        without = "\n".join(
            ln
            for ln in _security().splitlines()
            if "refresh-pins" not in ln and "cooldown" not in ln
        )
        assert "policy-refresh-claim" not in policy_problems(
            without, refresh_exists=False
        )


_RUN_STEP = "      - name: Injected\n        run: {command}\n"


def _add_run_step(
    name: str, before: str, command: str
) -> Callable[[dict[str, str]], tuple[dict[str, str], dict[tuple[str, str], str]]]:
    """Insert a `run:` step before the step `before` of workflow `name` (waivers unchanged)."""

    def apply(files: dict[str, str]):
        out = dict(files)
        marker = f"      - name: {before}\n"
        assert marker in out[name], before
        out[name] = out[name].replace(
            marker, _RUN_STEP.format(command=command) + marker, 1
        )
        return out, dict(_INSTALL_WAIVERS)

    return apply


def _add_uses_step(
    name: str, before: str, action: str
) -> Callable[[dict[str, str]], tuple[dict[str, str], dict[tuple[str, str], str]]]:
    def apply(files: dict[str, str]):
        out = dict(files)
        marker = f"      - name: {before}\n"
        assert marker in out[name], before
        out[name] = out[name].replace(
            marker, f"      - name: Injected\n        uses: {action}\n" + marker, 1
        )
        return out, dict(_INSTALL_WAIVERS)

    return apply


_SETUP = "actions/setup-python@6d4fd22b20465ccc0f7354476588e8df7c9ca0e0 # v7.1.0"

WRITE_CODE_MUTANTS: list[tuple[str, Any, str]] = [
    (
        "the propose job installs a package",
        _add_run_step(
            "refresh-pins.yml", "Download the proposed files", "pip install x"
        ),
        "write-job-runs-code",
    ),
    (
        "the propose job runs the tests",
        _add_run_step(
            "refresh-pins.yml", "Download the proposed files", "python -m pytest tests/"
        ),
        "write-job-runs-code",
    ),
    (
        "the propose job runs mypy",
        _add_run_step(
            "refresh-pins.yml", "Download the proposed files", "mypy -p fsm_llm"
        ),
        "write-job-runs-code",
    ),
    (
        "the propose job runs a project script",
        _add_run_step(
            "refresh-pins.yml", "Download the proposed files", "scripts/refresh_pins.py"
        ),
        "write-job-runs-code",
    ),
    (
        "the propose job sets up python",
        _add_uses_step("refresh-pins.yml", "Download the proposed files", _SETUP),
        "write-job-runs-code",
    ),
    (
        "the release job installs with uv",
        _add_run_step("release.yml", "Download the SBOM", "uv pip install x"),
        "write-job-runs-code",
    ),
    (
        "the publish job runs make",
        _add_run_step("release.yml", "Publish to PyPI", "make build"),
        "write-job-runs-code",
    ),
    (
        "the publish job sets up python",
        _add_uses_step("release.yml", "Publish to PyPI", _SETUP),
        "write-job-runs-code",
    ),
    (
        "the dependency review job runs the tests",
        _add_run_step(
            "python-package.yml",
            "Review the dependency changes of the pull request",
            "nice -n 19 pytest",
        ),
        "write-job-runs-code",
    ),
    (
        "the waiver of the docs deploy job is removed",
        lambda files: (files, {}),
        "write-job-runs-code",
    ),
    (
        "a waiver names a job that does not write",
        lambda files: (
            files,
            {**_INSTALL_WAIVERS, ("python-package.yml", "lint"): "x"},
        ),
        "waiver-unused",
    ),
    (
        "a waiver names a write job that runs no code",
        lambda files: (
            files,
            {**_INSTALL_WAIVERS, ("python-package.yml", "dependency-review"): "x"},
        ),
        "waiver-unused",
    ),
    (
        "a waiver names a job that is gone",
        lambda files: (files, {**_INSTALL_WAIVERS, ("docs.yml", "gone"): "x"}),
        "waiver-unused",
    ),
    (
        "an unreadable workflow",
        lambda files: (
            {**files, "broken.yml": "jobs: [unterminated"},
            dict(_INSTALL_WAIVERS),
        ),
        "unreadable",
    ),
]


class TestTheWriteJobMutantsAreKilled:
    @pytest.mark.parametrize(
        "label,mutate,code", WRITE_CODE_MUTANTS, ids=[m[0] for m in WRITE_CODE_MUTANTS]
    )
    def test_each_mutant_is_caught_by_its_rule(
        self, label: str, mutate: Any, code: str
    ):
        files, waivers = mutate(_files())
        assert (files, waivers) != (_files(), _INSTALL_WAIVERS), label
        assert code in write_job_code_problems(files, waivers), label

    @pytest.mark.parametrize(
        "script",
        [
            "pip install x",
            "python -m pytest",
            "nice -n 19 pytest",
            "echo hi && uv pip install x",
            "FOO=1 ./scripts/x.sh",
            "make test",
        ],
    )
    def test_the_scanner_finds_code(self, script: str):
        assert runs_code(script)

    @pytest.mark.parametrize(
        "script",
        [
            "git push origin main",
            "gh release create v1 dist/* --verify-tag",
            "echo 'pip install x'",
            'echo "run pytest; pip install"',
            "cp a b && cd d",
            "cat <<'X'\npip install y\nX\necho done",
        ],
    )
    def test_the_scanner_ignores_plain_commands(self, script: str):
        assert not runs_code(script)


class TestEveryRuleHasAMutant:
    def test_at_least_twelve_distinct_mutants_in_total(self):
        total = (
            len(WORKFLOW_MUTANTS)
            + len(CANARY_MUTANTS)
            + len(CROSS_MUTANTS)
            + len(PIN_MUTANTS)
            + len(FIXTURE_MUTANTS)
            + len(POLICY_MUTANTS)
        )
        assert total >= 12

    def test_every_code_a_checker_can_return_has_a_mutant(self):
        source = Path(__file__).read_text(encoding="utf-8")
        body = source.split("def supply_chain_problems", 1)[1].split(
            "# The real files", 1
        )[0]
        codes = set(re.findall(r'(?:found\.add|return \{)\(?"([a-z-]+)"', body))
        codes |= set(re.findall(r'"(policy-[a-z-]+)":', body))
        killed = (
            {m[2] for m in WORKFLOW_MUTANTS}
            | {m[2] for m in CROSS_MUTANTS}
            | {m[2] for m in PIN_MUTANTS}
            | {m[2] for m in FIXTURE_MUTANTS}
            | {m[2] for m in POLICY_MUTANTS}
            | {m[2] for m in WRITE_CODE_MUTANTS}
            | {"canary-polarity"}
        )
        assert codes - killed == set(), sorted(codes - killed)
