"""The release workflow publishes a checked tag once, through trusted publishing, with an SBOM.

Plan d454fd62, step 112 (roadmap row 0.1). `.github/workflows/release.yml` is the only thing that
uploads the package to PyPI, and an upload cannot be undone, so a slip in it is either a bad
release that nobody can take back (a tag that does not match the version, a pre-release published
as final, a build that is not the one that was checked) or a supply-chain hole (a long-lived
token, a write scope on the wrong job, a floating action tag). None of it runs on a developer
machine, so it is read here as data and its scripts are run on synthetic inputs.

* `workflow_problems(text)` is a pure function of the workflow text and returns a set of problem
  codes. The real workflow must return the empty set (`TestTheRealWorkflow`); each mutant of
  `MUTANTS` is the real text with one edit and must return its own code (`TestMutantsAreKilled`):
  a check that cannot fail is not a check.
* The shell and Python scripts inside the workflow (the tag and version guard, the release-notes
  extraction, the file-name check, the wheel finder, the SBOM check) are extracted from the YAML
  and run in bash on synthetic trees (`TestTheGuards` and the other `Test...` classes), so the
  shipped text is what is tested and nothing is copied. A mutant of each script must fail its
  battery (`TestTheShippedScriptsAreMutationChecked`).
* Only the standard library, `packaging` and `yaml.safe_load` are used. PyYAML reads the key `on`
  as the boolean `True` (YAML 1.1); `_load` puts it back under `"on"`.
* The environment URL names `llm-fsm`, the distribution of plan step 113 (the rename). Until
  `[project] name` says so the `publish` job cannot succeed; `TestTheDistributionName` pins both
  facts so that step 113 changes one place and cannot forget the other.
* The cross-file rules (every `uses:` a SHA with a release tag, a write scope only where the
  allowlist of `test_supply_chain.py` names it, a read-only top-level token) are checked there for
  every workflow file, this one included.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.test_fsm_llm_regression.workflow_support import (
    _assert_multi_line_scripts_fail_fast,
    _assert_mutant_caught,
    _assert_runners_pinned_with_timeouts,
    _code_lines,
    _drop_comment,
    _edit,
    _load,
    _step_running,
    _steps,
    _unpin_checkout,
    _uses_steps,
)

_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW = _ROOT / ".github" / "workflows" / "release.yml"
_DOCS_WORKFLOW = _ROOT / ".github" / "workflows" / "docs.yml"

_SHA_USES = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+@[0-9a-f]{40}$")
_USES_LINE = re.compile(r"^\s*(?:-\s*)?uses:\s*(?P<ref>\S+)(?P<rest>.*)$", re.MULTILINE)
_TAG_COMMENT = re.compile(r"^\s*#\s*v(?P<major>\d+)\.\d+\.\d+\s*$")

_JOBS = {"build", "sbom", "publish", "github-release"}
_ACTIONS = {
    "actions/checkout",
    "actions/setup-python",
    "actions/upload-artifact",
    "actions/download-artifact",
    "pypa/gh-action-pypi-publish",
}
_ENVIRONMENT = {"name": "pypi", "url": "https://pypi.org/p/llm-fsm"}
_RELEASE_COMMAND = (
    'gh release create "$TAG" dist/* sbom/sbom.cdx.json --title "$TAG" '
    "--notes-file notes/release-notes.md --verify-tag"
)
#: The extract step: the notes carry absolute links under the tag, written before the cut.
_EXTRACT_COMMAND = (
    'python scripts/changelog_check.py extract "$VERSION" '
    '--link-base "https://github.com/${REPO}/blob/${TAG}/" > release-notes.md'
)
#: The fit step drops a link text that a cut would leave open.
_OPEN_LINK_LOOP = 'while (open_link := re.search(r"\\[[^\\[\\]]*\\Z", cut)):'
_SBOM_COMMAND = (
    'cyclonedx-py environment "$RUNNER_TEMP/sbom-env/bin/python" '
    "--pyproject pyproject.toml --mc-type library --sv 1.6 --of JSON -o sbom.cdx.json"
)
#: A long-lived credential or an upload outside the trusted publisher.
_CREDENTIAL = re.compile(
    r"\bsecrets\.|GITHUB_TOKEN|twine\s+upload|pypi-token|api[-_]token|\bTWINE_|"
    r"\bpassword\b|PYPI_[A-Z_]*(?:TOKEN|PASSWORD|API)",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------------------
# Reading the workflow
# --------------------------------------------------------------------------------------


def _step_id(job: dict[str, Any], step_id: str) -> dict[str, Any] | None:
    return next((s for s in _steps(job) if s.get("id") == step_id), None)


def _index(job: dict[str, Any], step: dict[str, Any] | None) -> int:
    for i, candidate in enumerate(_steps(job)):
        if candidate is step:
            return i
    return 10_000


def _python_body(run: str) -> str:
    """The Python source of a `python - <<'PY' ... PY` heredoc inside a step script."""
    match = re.search(r"python - <<'PY'\n(.*?)\nPY\s*$", run, re.DOTALL)
    assert match, "no python heredoc in the script"
    return match.group(1) + "\n"


def _write_perms(perms: Any) -> set[str]:
    if not isinstance(perms, dict):
        return set()
    return {scope for scope, level in perms.items() if level == "write"}


def workflow_problems(text: str) -> set[str]:
    """Return the codes of every rule the workflow text breaks (empty set: it is sound)."""
    found: set[str] = set()
    try:
        wf = _load(text)
    except Exception:
        return {"unreadable"}
    jobs = wf.get("jobs") or {}
    build = jobs.get("build") or {}
    sbom = jobs.get("sbom") or {}
    publish = jobs.get("publish") or {}
    release = jobs.get("github-release") or {}
    code = "\n".join(_code_lines(text))

    # Identity and triggers: a pushed `v*` tag and nothing else.
    if wf.get("name") != "release":
        found.add("name")
    on = wf.get("on") or {}
    if not isinstance(on, dict):
        found.add("triggers")
    else:
        if on.get("push") != {"tags": ["v*"]}:
            found.add("push-trigger")
        if set(on) != {"push"}:
            found.add("extra-trigger")
    if "pull_request_target" in code:
        found.add("pull-request-target")
    conc = wf.get("concurrency") or {}
    if (
        not isinstance(conc, dict)
        or conc.get("group") != "release"
        or conc.get("cancel-in-progress") is not False
    ):
        found.add("concurrency")

    # Permissions: read at the top; `id-token: write` on publish only; `contents: write` on
    # github-release only; build and sbom carry no block at all.
    if wf.get("permissions") != {"contents": "read"}:
        found.add("top-permissions")
    if set(jobs) != _JOBS:
        found.add("jobs")
    if "id-token" in (wf.get("permissions") or {}) or any(
        "id-token" in (job.get("permissions") or {})
        for name, job in jobs.items()
        if name != "publish"
    ):
        found.add("id-token-scope")
    if publish.get("permissions") != {"id-token": "write"}:
        found.add("publish-permissions")
    if release.get("permissions") != {"contents": "write"}:
        found.add("release-permissions")
    for name in ("build", "sbom"):
        if "permissions" in (jobs.get(name) or {}):
            found.add("job-permissions")
    for name, job in jobs.items():
        if name not in ("publish", "github-release") and _write_perms(
            job.get("permissions")
        ):
            found.add("extra-write")
    if _write_perms(publish.get("permissions")) - {"id-token"} or _write_perms(
        release.get("permissions")
    ) - {"contents"}:
        found.add("extra-write")
    if re.search(
        r"\b(?:packages|pages|attestations|actions|id-token)\s*:\s*write", code
    ) and (
        "id-token: write" not in code
        or len(re.findall(r"\bid-token\s*:\s*write", code)) != 1
    ):
        found.add("id-token-scope")

    # The publish job: environment, no container, no reusable workflow, hosted runner.
    if publish.get("environment") != _ENVIRONMENT:
        found.add("environment")
    for name, job in jobs.items():
        if name != "publish" and "environment" in job:
            found.add("extra-environment")
    if (
        "container" in publish
        or "services" in publish
        or "uses" in publish
        or "strategy" in publish
        or publish.get("runs-on") != "ubuntu-24.04"
    ):
        found.add("publish-isolation")

    # Job order: the irreversible upload runs last, after the build and the SBOM.
    if sbom.get("needs") != "build":
        found.add("sbom-needs")
    if _as_set(publish.get("needs")) != {"build", "sbom"}:
        found.add("publish-needs")
    if _as_set(release.get("needs")) != {"publish", "sbom"}:
        found.add("release-needs")

    # Runners, timeouts, shell, interpolation.
    for job in jobs.values():
        if job.get("runs-on") != "ubuntu-24.04":
            found.add("runner")
        if not isinstance(job.get("timeout-minutes"), int):
            found.add("timeout")
        for step in _steps(job):
            run = step.get("run")
            if isinstance(run, str) and "\n" in run.strip():
                if "set -euo pipefail" not in run:
                    found.add("shell-strict")
            if isinstance(run, str) and "${{" in run:
                found.add("run-interpolation")
    if "-latest" in code:
        found.add("runner")
    if (wf.get("defaults") or {}).get("run", {}).get("shell") != "bash":
        found.add("default-shell")

    # Actions: a 40-hex SHA and a `# vX.Y.Z` comment on every `uses:`, a known set, one pin each.
    parsed_uses = [
        str(s["uses"]) for job in jobs.values() for s in _steps(job) if "uses" in s
    ]
    lines = list(_USES_LINE.finditer(text))
    if len(lines) != len(parsed_uses) or not parsed_uses:
        found.add("uses-count")
    pins: dict[str, set[str]] = {}
    for match in lines:
        if not _SHA_USES.match(match["ref"]):
            found.add("action-pin")
        tag = _TAG_COMMENT.match(match["rest"])
        if not tag:
            found.add("action-tag-comment")
        action, _, sha = match["ref"].partition("@")
        pins.setdefault(action, set()).add(sha + match["rest"].strip())
        if action.startswith("actions/") and action.endswith("-artifact"):
            if not tag or int(tag["major"]) < 4:
                found.add("artifact-version")
    if {a for a in pins} != _ACTIONS or any(len(v) != 1 for v in pins.values()):
        found.add("action-set")

    # Credentials: trusted publishing only. The default token appears once, for `gh`.
    if _CREDENTIAL.search(code):
        found.add("no-token")
    if len(re.findall(r"github\.token", code)) != 1:
        found.add("token-scope")

    # Build job.
    checkouts = _uses_steps(build, "actions/checkout")
    if (
        len(checkouts) != 1
        or (checkouts[0].get("with") or {}).get("persist-credentials") is not False
    ):
        found.add("build-persist-credentials")
    sbom_checkouts = _uses_steps(sbom, "actions/checkout")
    if (
        len(sbom_checkouts) != 1
        or (sbom_checkouts[0].get("with") or {}).get("persist-credentials") is not False
    ):
        found.add("sbom-persist-credentials")
    for job in (build, sbom):
        pythons = _uses_steps(job, "actions/setup-python")
        if (
            len(pythons) != 1
            or str((pythons[0].get("with") or {}).get("python-version")) != "3.12"
            or "cache" in (pythons[0].get("with") or {})
        ):
            found.add("build-python")
    install = _step_running(build, "pip install")
    if install is None or install["run"].strip() != (
        "python -m pip install -c constraints.txt build twine"
    ):
        found.add("build-install")

    guard = _step_id(build, "version")
    grun = str((guard or {}).get("run"))
    if guard is None or guard.get("env") != {"TAG": "${{ github.ref_name }}"}:
        found.add("guard-env")
    if 'tag != "v" + declared' not in grun or "sys.exit" not in grun:
        found.add("guard-tag")
    if (
        "is_prerelease" not in grun
        or "is_devrelease" not in grun
        or "InvalidVersion" not in grun
        or "parsed.local is not None" not in grun
    ):
        found.add("guard-final")
    if (
        'pathlib.Path("pyproject.toml")' not in grun
        or '["project"]["name"]' not in grun
        or 're.sub(r"[-_.]+", "_", project).lower()' not in grun
        or "-py3-none-any.whl" not in grun
        or "GITHUB_OUTPUT" not in grun
    ):
        found.add("guard-names")
    if 'pathlib.Path("src/fsm_llm/__version__.py")' not in grun:
        found.add("guard-version-file")

    extract = _step_running(build, "changelog_check.py")
    erun = str((extract or {}).get("run"))
    if (
        extract is None
        or _EXTRACT_COMMAND not in erun
        or "test -s release-notes.md" not in erun
        or (extract.get("env") or {})
        != {
            "VERSION": "${{ steps.version.outputs.version }}",
            "TAG": "${{ github.ref_name }}",
            "REPO": "${{ github.repository }}",
        }
    ):
        found.add("extract")

    fit = _step_running(build, "limit = 120_000")
    frun_text = str((fit or {}).get("run"))
    if (
        fit is None
        or (fit.get("env") or {})
        != {"TAG": "${{ github.ref_name }}", "REPO": "${{ github.repository }}"}
        or "if len(text) > limit:" not in frun_text
        or 'cut.count("```") % 2' not in frun_text
        or _OPEN_LINK_LOOP not in frun_text
        or "os.environ['REPO']" not in frun_text
        or "os.environ['TAG']" not in frun_text
    ):
        found.add("notes-limit")

    if (
        extract is not None
        and fit is not None
        and _index(build, extract) > _index(build, fit)
    ):
        found.add("notes-order")  # the links are made absolute BEFORE the notes are cut

    built = _step_running(build, "python -m build")
    if (
        built is None
        or built["run"].strip() != "python -m build"
        or (built.get("env") or {}).get("PIP_CONSTRAINT")
        != "${{ github.workspace }}/constraints.txt"
    ):
        found.add("build-command")
    checked = _step_running(build, "twine check")
    if checked is None or checked["run"].strip() != "twine check --strict dist/*":
        found.add("twine-strict")
    names = _step_running(build, "LC_ALL=C sort")
    nrun = str((names or {}).get("run"))
    if (
        names is None
        or (names.get("env") or {})
        != {
            "WHEEL": "${{ steps.version.outputs.wheel }}",
            "SDIST": "${{ steps.version.outputs.sdist }}",
        }
        or "exit 1" not in nrun
        or '[ "$actual" != "$wanted" ]' not in nrun
    ):
        found.add("names-check")
    order = [guard, extract, fit, built, checked, names]
    positions = [_index(build, s) for s in order]
    uploads = _uses_steps(build, "actions/upload-artifact")
    if positions != sorted(positions) or len(set(positions)) != 6:
        found.add("build-order")
    if uploads and min(_index(build, u) for u in uploads) < max(positions):
        found.add("build-order")

    # Artifacts: dist and notes from build, the SBOM from sbom; each must exist and expire.
    expected_uploads = {
        "build": {"dist": "dist/", "release-notes": "release-notes.md"},
        "sbom": {"sbom": "sbom.cdx.json"},
    }
    for name, wanted in expected_uploads.items():
        got = {}
        for step in _uses_steps(jobs.get(name) or {}, "actions/upload-artifact"):
            w = step.get("with") or {}
            got[w.get("name")] = w.get("path")
            retention = w.get("retention-days")
            if w.get("if-no-files-found") != "error" or not (
                isinstance(retention, int) and 1 <= retention <= 30
            ):
                found.add("upload-settings")
        if got != wanted:
            found.add("upload-settings")

    # SBOM job.
    sdown = _uses_steps(sbom, "actions/download-artifact")
    if [(s.get("with") or {}).get("name") for s in sdown] != ["dist"]:
        found.add("sbom-download")
    finder = _step_id(sbom, "wheel")
    frun = str((finder or {}).get("run"))
    if (
        finder is None
        or 'if [ "${#wheels[@]}" -ne 1 ]' not in frun
        or "exit 1" not in frun
        or 'echo "path=${wheels[0]}" >> "$GITHUB_OUTPUT"' not in frun
    ):
        found.add("sbom-wheel")
    wheel_install = _step_running(sbom, 'sbom-env/bin/python" -m pip install')
    wrun = str((wheel_install or {}).get("run"))
    if (
        wheel_install is None
        or '-m pip install -c constraints.txt "${WHEEL}[all]"' not in wrun
        or (wheel_install.get("env") or {}).get("WHEEL")
        != "${{ steps.wheel.outputs.path }}"
    ):
        found.add("sbom-extras")
    tool = _step_running(sbom, "pip install -c constraints.txt cyclonedx-bom")
    if tool is None or tool["run"].strip() != (
        "python -m pip install -c constraints.txt cyclonedx-bom"
    ):
        found.add("sbom-tool")
    writer = _step_running(sbom, "cyclonedx-py")
    swrun = str((writer or {}).get("run"))
    if writer is None or _SBOM_COMMAND not in swrun:
        found.add("sbom-command")
    if (
        'bom["bomFormat"] == "CycloneDX"' not in swrun
        or 'bom["components"]' not in swrun
    ):
        found.add("sbom-validate")
    # Find the wheel, install it, and only then write the SBOM, with the tool installed first.
    sbom_order = [
        _index(sbom, finder),
        _index(sbom, wheel_install),
        _index(sbom, writer),
    ]
    if sbom_order != sorted(sbom_order) or _index(sbom, tool) > sbom_order[2]:
        found.add("sbom-order")

    # Publish job: exactly the download and the publish action; nothing else can run there.
    psteps = _steps(publish)
    if [s.get("uses", "").split("@")[0] for s in psteps] != [
        "actions/download-artifact",
        "pypa/gh-action-pypi-publish",
    ]:
        found.add("publish-steps")
    elif (psteps[0].get("with") or {}) != {"name": "dist", "path": "dist"}:
        found.add("publish-steps")
    elif (psteps[1].get("with") or {}).get("attestations") is not True:
        found.add("publish-attestations")
    elif (psteps[1].get("with") or {}) != {
        "packages-dir": "dist/",
        "attestations": True,
    }:
        found.add("publish-with")
    elif "env" in psteps[1] or "run" in psteps[1]:
        found.add("publish-with")

    # GitHub release job: no checkout, the exact `gh` call, the default token and nothing else.
    if _uses_steps(release, "actions/checkout"):
        found.add("release-checkout")
    rdown = [
        ((s.get("with") or {}).get("name"), (s.get("with") or {}).get("path"))
        for s in _uses_steps(release, "actions/download-artifact")
    ]
    if rdown != [("dist", "dist"), ("sbom", "sbom"), ("release-notes", "notes")]:
        found.add("release-download")
    gh = _step_running(release, "gh release create")
    if gh is None or gh["run"].count(_RELEASE_COMMAND) != 1:
        found.add("release-command")
    if gh is None or (gh.get("env") or {}) != {
        "GH_TOKEN": "${{ github.token }}",
        "GH_REPO": "${{ github.repository }}",
        "TAG": "${{ github.ref_name }}",
    }:
        found.add("release-env")
    return found


def _as_set(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, list):
        return {str(v) for v in value}
    return set()


def _real_text() -> str:
    return _WORKFLOW.read_text(encoding="utf-8")


# --------------------------------------------------------------------------------------
# The real workflow
# --------------------------------------------------------------------------------------


class TestTheRealWorkflow:
    def test_the_workflow_exists_under_the_name_the_trusted_publisher_registers(self):
        # PyPI matches the publisher on the workflow FILE NAME; renaming the file voids it.
        assert _WORKFLOW.is_file(), ".github/workflows/release.yml is missing"
        assert _WORKFLOW.name == "release.yml"
        assert isinstance(_load(_real_text()), dict)

    def test_it_breaks_no_rule(self):
        assert workflow_problems(_real_text()) == set()

    def test_a_pushed_version_tag_is_the_only_trigger(self):
        wf = _load(_real_text())
        assert wf["on"] == {"push": {"tags": ["v*"]}}
        assert wf["name"] == "release"

    def test_the_workflow_never_cancels_a_release_in_flight(self):
        assert _load(_real_text())["concurrency"] == {
            "group": "release",
            "cancel-in-progress": False,
        }

    def test_the_four_jobs_and_their_order(self):
        jobs = _load(_real_text())["jobs"]
        assert set(jobs) == _JOBS
        assert "needs" not in jobs["build"]
        assert jobs["sbom"]["needs"] == "build"
        assert jobs["publish"]["needs"] == ["build", "sbom"]
        assert jobs["github-release"]["needs"] == ["publish", "sbom"]

    def test_read_only_at_the_top_and_two_scopes_each_on_its_own_job(self):
        wf = _load(_real_text())
        assert wf["permissions"] == {"contents": "read"}
        assert "permissions" not in wf["jobs"]["build"]
        assert "permissions" not in wf["jobs"]["sbom"]
        assert wf["jobs"]["publish"]["permissions"] == {"id-token": "write"}
        assert wf["jobs"]["github-release"]["permissions"] == {"contents": "write"}
        code = "\n".join(_code_lines(_real_text()))
        assert len(re.findall(r"id-token\s*:\s*write", code)) == 1
        assert len(re.findall(r"contents\s*:\s*write", code)) == 1

    def test_the_publish_job_names_the_environment_and_the_project_url(self):
        publish = _load(_real_text())["jobs"]["publish"]
        assert publish["environment"] == {
            "name": "pypi",
            "url": "https://pypi.org/p/llm-fsm",
        }

    def test_the_publish_job_is_a_plain_hosted_job(self):
        publish = _load(_real_text())["jobs"]["publish"]
        assert publish["runs-on"] == "ubuntu-24.04"
        for key in ("container", "services", "uses", "strategy"):
            assert key not in publish, key
        assert not [
            k
            for k in publish
            if k
            not in {
                "needs",
                "runs-on",
                "timeout-minutes",
                "environment",
                "permissions",
                "steps",
            }
        ]

    def test_the_publish_job_only_downloads_and_publishes_with_attestations(self):
        steps = _steps(_load(_real_text())["jobs"]["publish"])
        assert [s["uses"].split("@")[0] for s in steps] == [
            "actions/download-artifact",
            "pypa/gh-action-pypi-publish",
        ]
        assert steps[1]["with"] == {"packages-dir": "dist/", "attestations": True}

    def test_no_long_lived_credential_is_named_anywhere(self):
        code = "\n".join(_code_lines(_real_text()))
        assert _CREDENTIAL.search(code) is None
        assert "secrets." not in code and "twine upload" not in code

    def test_the_default_token_is_named_once_and_only_for_gh(self):
        wf = _load(_real_text())
        step = _step_running(wf["jobs"]["github-release"], "gh release create")
        assert step["env"]["GH_TOKEN"] == "${{ github.token }}"
        assert len(re.findall(r"github\.token", _real_text())) == 1

    def test_the_release_job_creates_the_release_for_the_verified_tag(self):
        wf = _load(_real_text())
        step = _step_running(wf["jobs"]["github-release"], "gh release create")
        assert step["run"].strip().splitlines()[-1] == _RELEASE_COMMAND
        assert step["env"] == {
            "GH_TOKEN": "${{ github.token }}",
            "GH_REPO": "${{ github.repository }}",
            "TAG": "${{ github.ref_name }}",
        }
        assert not _uses_steps(wf["jobs"]["github-release"], "actions/checkout")

    def test_the_build_job_guards_builds_and_checks_in_that_order(self):
        build = _load(_real_text())["jobs"]["build"]
        ids_and_runs = [
            (s.get("id"), str(s.get("run", "")), str(s.get("uses", "")))
            for s in _steps(build)
        ]
        order = [
            next(
                i
                for i, (sid, run, uses) in enumerate(ids_and_runs)
                if needle in (sid or "") or needle in run or needle in uses
            )
            for needle in (
                "version",
                "changelog_check.py extract",
                "limit = 120_000",
                "python -m build",
                "twine check --strict dist/*",
                "LC_ALL=C sort",
                "upload-artifact",
            )
        ]
        assert order == sorted(order) and len(set(order)) == 7

    def test_the_build_runs_the_backend_under_the_repository_pins(self):
        build = _load(_real_text())["jobs"]["build"]
        step = _step_running(build, "python -m build")
        assert step["env"] == {
            "PIP_CONSTRAINT": "${{ github.workspace }}/constraints.txt"
        }

    def test_the_release_notes_come_from_the_changelog_extractor_into_a_file(self):
        build = _load(_real_text())["jobs"]["build"]
        step = _step_running(build, "changelog_check.py")
        assert step["env"] == {
            "VERSION": "${{ steps.version.outputs.version }}",
            "TAG": "${{ github.ref_name }}",
            "REPO": "${{ github.repository }}",
        }
        assert _EXTRACT_COMMAND in step["run"]
        assert "${{" not in step["run"]
        assert "test -s release-notes.md" in step["run"]

    def test_the_artifacts_are_named_uniquely_and_fail_when_empty(self):
        wf = _load(_real_text())
        names = []
        for job in wf["jobs"].values():
            for step in _uses_steps(job, "actions/upload-artifact"):
                names.append(step["with"]["name"])
                assert step["with"]["if-no-files-found"] == "error"
        assert sorted(names) == ["dist", "release-notes", "sbom"]

    def test_the_sbom_is_written_from_the_wheel_installed_with_every_extra(self):
        sbom = _load(_real_text())["jobs"]["sbom"]
        install = _step_running(
            sbom, '-m pip install -c constraints.txt "${WHEEL}[all]"'
        )
        assert install is not None
        writer = _step_running(sbom, "cyclonedx-py")
        assert _SBOM_COMMAND in writer["run"]
        assert "sbom.cdx.json" in _step_running(sbom, "cyclonedx-py")["run"]

    def test_every_action_is_a_sha_pin_with_its_release_tag(self):
        text = _real_text()
        matches = list(_USES_LINE.finditer(text))
        assert len(matches) == 13  # build 4, sbom 4, publish 2, github-release 3
        for match in matches:
            assert _SHA_USES.match(match["ref"]), match["ref"]
            assert _TAG_COMMENT.match(match["rest"]), match["rest"]

    def test_one_action_has_one_pin_in_the_whole_file(self):
        pins: dict[str, set[str]] = {}
        for match in _USES_LINE.finditer(_real_text()):
            action, _, sha = match["ref"].partition("@")
            pins.setdefault(action, set()).add(sha + match["rest"].strip())
        assert {a for a in pins} == _ACTIONS
        assert all(len(v) == 1 for v in pins.values()), pins

    def test_the_artifact_actions_are_the_v4_and_later_semantics(self):
        for match in _USES_LINE.finditer(_real_text()):
            if match["ref"].startswith(
                ("actions/upload-artifact", "actions/download-artifact")
            ):
                assert int(_TAG_COMMENT.match(match["rest"])["major"]) >= 4

    def test_no_event_field_is_interpolated_into_a_script(self):
        for job in _load(_real_text())["jobs"].values():
            for step in _steps(job):
                assert "${{" not in str(step.get("run", "")), step.get("name")

    def test_no_privileged_trigger_and_no_floating_runner(self):
        code = "\n".join(_code_lines(_real_text()))
        assert "pull_request_target" not in code
        assert "workflow_dispatch" not in code
        assert "-latest" not in code

    def test_every_job_has_a_timeout_and_a_pinned_runner(self):
        _assert_runners_pinned_with_timeouts(_load(_real_text()))

    def test_every_multi_line_script_fails_fast(self):
        _assert_multi_line_scripts_fail_fast(_load(_real_text()))

    def test_the_checkouts_keep_no_credentials(self):
        wf = _load(_real_text())
        for name in ("build", "sbom"):
            (checkout,) = _uses_steps(wf["jobs"][name], "actions/checkout")
            assert checkout["with"]["persist-credentials"] is False


# --------------------------------------------------------------------------------------
# The scripts of the workflow, run on synthetic trees
# --------------------------------------------------------------------------------------


def _script(job: str, *, step_id: str | None = None, needle: str | None = None) -> str:
    j = _load(_real_text())["jobs"][job]
    step = _step_id(j, step_id) if step_id else _step_running(j, needle or "")
    assert step is not None, (job, step_id, needle)
    return str(step["run"])


def _run(
    script: str, cwd: Path, env: dict[str, str] | None = None
) -> tuple[int, str, str]:
    """Run a workflow script in bash inside `cwd`; return (exit code, $GITHUB_OUTPUT, stderr)."""
    cwd.mkdir(parents=True, exist_ok=True)
    out = cwd / "github_output"
    out.write_text("", encoding="utf-8")
    sh = cwd / "step.sh"
    sh.write_text(script, encoding="utf-8")
    full_env = {
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
        "GITHUB_OUTPUT": str(out),
        "HOME": str(cwd),
        **(env or {}),
    }
    proc = subprocess.run(
        ["bash", str(sh)],
        cwd=cwd,
        env=full_env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return proc.returncode, out.read_text(encoding="utf-8"), proc.stderr


def _tree(cwd: Path, declared: str | None, project: str) -> None:
    pkg = cwd / "src" / "fsm_llm"
    pkg.mkdir(parents=True, exist_ok=True)
    body = (
        "# no version here\n" if declared is None else f'__version__ = "{declared}"\n'
    )
    (pkg / "__version__.py").write_text(body, encoding="utf-8")
    (cwd / "pyproject.toml").write_text(
        f'[project]\nname = "{project}"\nversion = "0"\n', encoding="utf-8"
    )


#: (case id, tag, declared version, project name, expected: output lines or a stderr fragment).
GUARD_CASES: list[tuple[str, str, str | None, str, Any]] = [
    (
        "final",
        "v0.13.0",
        "0.13.0",
        "fsm-llm",
        [
            "version=0.13.0",
            "wheel=fsm_llm-0.13.0-py3-none-any.whl",
            "sdist=fsm_llm-0.13.0.tar.gz",
        ],
    ),
    (
        "final-renamed",
        "v0.13.0",
        "0.13.0",
        "llm-fsm",
        [
            "version=0.13.0",
            "wheel=llm_fsm-0.13.0-py3-none-any.whl",
            "sdist=llm_fsm-0.13.0.tar.gz",
        ],
    ),
    (
        "name-normalised",
        "v1.0",
        "1.0",
        "Fsm.LLM--x__y",
        [
            "version=1.0",
            "wheel=fsm_llm_x_y-1.0-py3-none-any.whl",
            "sdist=fsm_llm_x_y-1.0.tar.gz",
        ],
    ),
    (
        "post-release",
        "v0.13.0.post1",
        "0.13.0.post1",
        "llm-fsm",
        [
            "version=0.13.0.post1",
            "wheel=llm_fsm-0.13.0.post1-py3-none-any.whl",
            "sdist=llm_fsm-0.13.0.post1.tar.gz",
        ],
    ),
    ("tag-newer", "v0.13.1", "0.13.0", "llm-fsm", "does not match"),
    ("tag-older", "v0.13.0", "0.13.1", "llm-fsm", "does not match"),
    ("tag-pre-version-final", "v0.13.0rc1", "0.13.0", "llm-fsm", "does not match"),
    ("tag-without-v", "0.13.0", "0.13.0", "llm-fsm", "does not match"),
    ("tag-capital-v", "V0.13.0", "0.13.0", "llm-fsm", "does not match"),
    ("rc", "v0.13.1rc1", "0.13.1rc1", "llm-fsm", "pre-release"),
    ("alpha", "v0.13.1a1", "0.13.1a1", "llm-fsm", "pre-release"),
    ("beta", "v0.13.1b2", "0.13.1b2", "llm-fsm", "pre-release"),
    ("dev", "v0.13.1.dev3", "0.13.1.dev3", "llm-fsm", "pre-release"),
    ("post-dev", "v0.13.0.post1.dev2", "0.13.0.post1.dev2", "llm-fsm", "pre-release"),
    ("non-canonical-rc", "v0.13.1-rc.1", "0.13.1-rc.1", "llm-fsm", "plain PEP 440"),
    ("non-canonical-post", "v0.13.0-1", "0.13.0-1", "llm-fsm", "plain PEP 440"),
    ("local", "v0.13.0+local", "0.13.0+local", "llm-fsm", "plain PEP 440"),
    ("epoch", "v1!0.13.0", "1!0.13.0", "llm-fsm", "plain PEP 440"),
    ("garbage", "vgarbage", "garbage", "llm-fsm", "PEP 440"),
    ("no-version", "v0.13.0", None, "llm-fsm", "__version__"),
]


def _guard_failures(script: str, base: Path) -> list[str]:
    """The ids of the GUARD_CASES the script gets wrong (empty: it behaves as specified)."""
    wrong = []
    for case_id, tag, declared, project, expected in GUARD_CASES:
        cwd = base / case_id
        _tree(cwd, declared, project)
        code, output, err = _run(script, cwd, {"TAG": tag})
        if isinstance(expected, list):
            if code != 0 or output.splitlines() != expected:
                wrong.append(case_id)
        elif code == 0 or output != "" or expected not in err:
            wrong.append(case_id)
    return wrong


class TestTheGuards:
    @pytest.mark.parametrize("case", GUARD_CASES, ids=[c[0] for c in GUARD_CASES])
    def test_the_tag_and_version_guard(self, tmp_path: Path, case: Any):
        _, tag, declared, project, expected = case
        _tree(tmp_path, declared, project)
        code, output, err = _run(
            _script("build", step_id="version"), tmp_path, {"TAG": tag}
        )
        if isinstance(expected, list):
            assert code == 0, err
            assert output.splitlines() == expected
        else:
            assert code != 0
            assert output == "", "a failed guard must write no output"
            assert expected in err, err

    def test_a_failed_guard_names_the_expected_tag(self, tmp_path: Path):
        _tree(tmp_path, "0.13.0", "llm-fsm")
        _, _, err = _run(
            _script("build", step_id="version"), tmp_path, {"TAG": "v0.13.1"}
        )
        assert "expected v0.13.0" in err and "src/fsm_llm/__version__.py" in err

    def test_a_pre_release_message_says_final_releases_only(self, tmp_path: Path):
        _tree(tmp_path, "0.13.1rc1", "llm-fsm")
        _, _, err = _run(
            _script("build", step_id="version"), tmp_path, {"TAG": "v0.13.1rc1"}
        )
        assert "final releases only" in err

    def test_the_real_tree_and_its_own_tag_pass_with_the_names_of_its_project(
        self, tmp_path: Path
    ):
        declared = re.search(
            r'__version__\s*=\s*"([^"]+)"',
            (_ROOT / "src" / "fsm_llm" / "__version__.py").read_text(encoding="utf-8"),
        )
        assert declared is not None
        (tmp_path / "src" / "fsm_llm").mkdir(parents=True)
        shutil.copy(
            _ROOT / "src" / "fsm_llm" / "__version__.py",
            tmp_path / "src" / "fsm_llm" / "__version__.py",
        )
        shutil.copy(_ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
        code, output, err = _run(
            _script("build", step_id="version"), tmp_path, {"TAG": "v" + declared[1]}
        )
        assert code == 0, err
        project = tomllib.loads((_ROOT / "pyproject.toml").read_text("utf-8"))[
            "project"
        ]["name"]
        stem = re.sub(r"[-_.]+", "_", project).lower()
        assert output.splitlines() == [
            f"version={declared[1]}",
            f"wheel={stem}-{declared[1]}-py3-none-any.whl",
            f"sdist={stem}-{declared[1]}.tar.gz",
        ]

    def test_the_real_tree_with_a_tag_of_another_version_fails(self, tmp_path: Path):
        (tmp_path / "src" / "fsm_llm").mkdir(parents=True)
        shutil.copy(
            _ROOT / "src" / "fsm_llm" / "__version__.py",
            tmp_path / "src" / "fsm_llm" / "__version__.py",
        )
        shutil.copy(_ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
        code, output, err = _run(
            _script("build", step_id="version"), tmp_path, {"TAG": "v99.0.0"}
        )
        assert code != 0 and output == "" and "does not match" in err


def _docs_guard() -> str:
    deploy = _load(_DOCS_WORKFLOW.read_text(encoding="utf-8"))["jobs"]["deploy"]
    return str(_step_id(deploy, "version")["run"])


class TestTheGuardAgreesWithTheDocsWorkflow:
    """The docs workflow classifies a version as pre-release or final; so must this one."""

    @pytest.mark.parametrize(
        "version",
        [
            "0.13.0",
            "1.0",
            "0.13.0.post1",
            "0.13.1rc1",
            "0.13.1a1",
            "0.13.1b2",
            "0.13.1.dev3",
            "0.13.0.post1.dev2",
        ],
    )
    def test_a_version_is_final_here_exactly_when_it_is_final_there(
        self, tmp_path: Path, version: str
    ):
        _tree(tmp_path / "rel", version, "llm-fsm")
        rel_code, _, _ = _run(
            _script("build", step_id="version"),
            tmp_path / "rel",
            {"TAG": "v" + version},
        )
        _tree(tmp_path / "docs", version, "llm-fsm")
        doc_code, doc_out, doc_err = _run(
            _docs_guard(), tmp_path / "docs", {"GITHUB_REF_NAME": "v" + version}
        )
        assert doc_code == 0, doc_err
        assert (rel_code == 0) == ("prerelease=false" in doc_out.splitlines())


# --------------------------------------------------------------------------------------
# The release notes, the file names, the wheel finder and the SBOM check
# --------------------------------------------------------------------------------------

_CHANGELOG = """# Changelog

## [Unreleased]

## [0.13.0] - 2026-11-20

### Added

- The thing that shipped.

### Fixed

- The bug that was fixed.

## [0.12.9] - 2026-11-01

## [0.12.0] - 2026-10-01

### Added

- Old.
"""


def _unreleased_has_notes(text: str) -> bool:
    """True while `## [Unreleased]` holds a body (it is empty right after a release is cut)."""
    match = re.search(r"^## \[Unreleased\]\n(.*?)(?=^## \[)", text, re.M | re.S)
    return bool(match and match.group(1).strip())


def _newest_release(text: str) -> str:
    return re.findall(r"^## \[(\d+\.\d+\.\d+)\]", text, re.MULTILINE)[0]


#: The environment the extract step gets from the workflow (`github.ref_name`, `github.repository`).
_NOTES_ENV = {"TAG": "v0.13.0", "REPO": "NikolasMarkou/fsm_llm"}
_LINK_BASE = "https://github.com/NikolasMarkou/fsm_llm/blob/v0.13.0/"

#: A section with relative guide links in every position the rewrite must tell apart: a plain
#: link, a `./docs/` link, a link in `<...>`, an inline code span and a fenced block that quote
#: the relative form, a link outside `docs/`, and an absolute link.
_LINKED_CHANGELOG = """# Changelog

## [Unreleased]

## [0.13.0] - 2026-11-20

### Changed

- **BREAKING:** a thing moved. [Upgrade guide](docs/upgrade-guide/v0.13.0/a.md#moved)
- Also [the guide](./docs/guide.md) and [angle](<docs/up guide.md>) are links.
- The rule: no `](docs/x.md)` link stays relative; see [the repo](https://example.org/docs/y.md).
- A [file outside docs](CONTRIBUTING.md) is not rewritten.

```text
[kept](docs/in-a-fence.md)
```
"""
_LINKED_NOTES = f"""### Changed

- **BREAKING:** a thing moved. [Upgrade guide]({_LINK_BASE}docs/upgrade-guide/v0.13.0/a.md#moved)
- Also [the guide]({_LINK_BASE}docs/guide.md) and [angle](<{_LINK_BASE}docs/up guide.md>) are links.
- The rule: no `](docs/x.md)` link stays relative; see [the repo](https://example.org/docs/y.md).
- A [file outside docs](CONTRIBUTING.md) is not rewritten.

```text
[kept](docs/in-a-fence.md)
```
"""


def _relative_doc_links(text: str) -> list[str]:
    """`](docs/` targets left outside code fences and inline code spans."""
    no_fences = re.sub(r"^(```+|~~~+).*?^\1[ \t]*$", "", text, flags=re.S | re.M)
    no_code = re.sub(r"(`+)(.+?)\1", "", no_fences, flags=re.S)
    return re.findall(r"\]\(<?(?:\./)?docs/[^)>]*", no_code)


def _notes_tree(cwd: Path, changelog: str = _CHANGELOG) -> None:
    (cwd / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy(_ROOT / "scripts" / "changelog_check.py", cwd / "scripts")
    (cwd / "CHANGELOG.md").write_text(changelog, encoding="utf-8")


def _notes_failures(script: str, base: Path) -> list[str]:
    wrong = []
    cwd = base / "ok"
    _notes_tree(cwd)
    code, _, _ = _run(script, cwd, {"VERSION": "0.13.0", **_NOTES_ENV})
    notes = (cwd / "release-notes.md").read_text(encoding="utf-8") if code == 0 else ""
    if (
        code != 0
        or notes
        != "### Added\n\n- The thing that shipped.\n\n### Fixed\n\n- The bug that was fixed.\n"
    ):
        wrong.append("ok")
    # The links of the notes point at the tag, not at the working tree of the reader.
    cwd = base / "links"
    _notes_tree(cwd, _LINKED_CHANGELOG)
    code, _, _ = _run(script, cwd, {"VERSION": "0.13.0", **_NOTES_ENV})
    notes = (cwd / "release-notes.md").read_text(encoding="utf-8") if code == 0 else ""
    if code != 0 or notes != _LINKED_NOTES or _relative_doc_links(notes):
        wrong.append("links")
    cwd = base / "links-other-tag"
    _notes_tree(cwd, _LINKED_CHANGELOG)
    env = {"TAG": "v9.9.9", "REPO": "someone/else"}
    code, _, _ = _run(script, cwd, {"VERSION": "0.13.0", **env})
    notes = (cwd / "release-notes.md").read_text(encoding="utf-8") if code == 0 else ""
    if (
        code != 0
        or "](https://github.com/someone/else/blob/v9.9.9/docs/guide.md)" not in notes
        or "NikolasMarkou" in notes
    ):
        wrong.append("links-follow-the-tag-and-repository")
    for case_id, version in (
        ("missing", "0.99.0"),
        ("empty", "0.12.9"),
        ("unreleased", "0.13.1"),
    ):
        cwd = base / case_id
        _notes_tree(cwd)
        code, _, _ = _run(script, cwd, {"VERSION": version, **_NOTES_ENV})
        if code == 0:
            wrong.append(case_id)
    return wrong


def _filler(size: int) -> str:
    """Lines `- xxx...` of at most 55 characters that add up to exactly `size`."""
    assert size >= 110
    lines, left = [], size
    while left >= 110:
        lines.append(55)
        left -= 55
    lines.extend([left // 2, left - left // 2])
    return "".join("- " + "x" * (n - 3) + "\n" for n in lines)


_FILLER_END = "xxxx"  # the last whole filler line, after the cut stripped its line end


def _link_cut_cases() -> dict[str, tuple[str, tuple[str, ...]]]:
    """Notes whose 120000th character sits where a link would be cut. Value: (notes, what must go).

    `straddle`: the character is inside the target of an absolute link (a line is kept whole or
    dropped whole, so no half target may survive). `open-text`: the cut falls on the line break
    inside a link text, `[the upgrade` / `guide](...)`; the open text must not survive either.
    """
    tail = "- a changelog line of moderate length, enough to add up.\n" * 400
    target = _LINK_BASE + "docs/upgrade-guide/v0.13.0/aaaa.md#bbbb"
    straddle = _filler(119_950) + f"- see [the guide]({target}) and more text.\n" + tail
    opener = "- see [the upgrade\n"
    closer = "guide](" + target + ") and " + "more text " * 60 + "\n"
    open_text = _filler(119_900) + opener + closer + tail
    return {
        "link-straddle": (straddle, ("aaaa.md", "[the guide]")),
        "link-open-text": (open_text, ("[the upgrade", "guide](")),
    }


def _fit_failures(script: str, base: Path) -> list[str]:
    """Run the notes-fitting step on short, boundary, long and fence-open notes."""
    wrong = []
    env = {"TAG": "v0.13.0", "REPO": "NikolasMarkou/fsm_llm"}
    line = "- a changelog line of moderate length, enough to add up.\n"
    boundary = (line * 3000)[:120_000]
    cases = {
        "short": ("### Added\n\n- one\n", "same"),
        "boundary": (boundary, "same"),
        "long": (line * 5000, "cut"),
        "long-open-fence": ("```\n" + line * 5000, "cut"),
        "long-unicode": ("\u00e9" * 200_000, "cut"),
    }
    for case_id, (notes, expected) in cases.items():
        cwd = base / case_id
        cwd.mkdir(parents=True)
        (cwd / "release-notes.md").write_text(notes, encoding="utf-8")
        code, _, _ = _run(script, cwd, env)
        out = (cwd / "release-notes.md").read_text(encoding="utf-8")
        link = "https://github.com/NikolasMarkou/fsm_llm/blob/v0.13.0/CHANGELOG.md"
        if code != 0:
            wrong.append(case_id)
        elif expected == "same" and out != notes:
            wrong.append(case_id)
        elif case_id == "long" and not out.split("\n\n---\n\n")[0].endswith("add up."):
            wrong.append(case_id)  # the cut must end on a whole line
        elif expected == "cut" and not (
            len(out) <= 125_000
            and out.endswith(f"in [CHANGELOG.md]({link}).\n")
            and out.count("```") % 2 == 0
            and out.startswith(notes[:50])
        ):
            wrong.append(case_id)
    for case_id, (notes, gone) in _link_cut_cases().items():
        cwd = base / case_id
        cwd.mkdir(parents=True)
        (cwd / "release-notes.md").write_text(notes, encoding="utf-8")
        code, _, _ = _run(script, cwd, env)
        out = (cwd / "release-notes.md").read_text(encoding="utf-8")
        kept = out.split("\n\n---\n\n")[0]
        if (
            code != 0
            or any(part in out for part in gone)
            or not kept.endswith(_FILLER_END)
        ):
            wrong.append(case_id)
    return wrong


def _names_failures(script: str, base: Path) -> list[str]:
    wrong = []
    good = ("llm_fsm-0.13.0-py3-none-any.whl", "llm_fsm-0.13.0.tar.gz")
    env = {"WHEEL": good[0], "SDIST": good[1]}
    cases = {
        "exact": (good, True),
        "reversed-creation": (good[::-1], True),
        "extra-file": ((*good, "notes.txt"), False),
        "missing-sdist": (good[:1], False),
        "missing-wheel": (good[1:], False),
        "old-name": (
            ("fsm_llm-0.13.0-py3-none-any.whl", "fsm_llm-0.13.0.tar.gz"),
            False,
        ),
        "other-version": (
            ("llm_fsm-0.13.1-py3-none-any.whl", "llm_fsm-0.13.1.tar.gz"),
            False,
        ),
        "empty": ((), False),
    }
    for case_id, (files, ok) in cases.items():
        cwd = base / case_id
        (cwd / "dist").mkdir(parents=True)
        for name in files:
            (cwd / "dist" / name).write_text("x", encoding="utf-8")
        code, _, _ = _run(script, cwd, env)
        if (code == 0) != ok:
            wrong.append(case_id)
    return wrong


def _wheel_failures(script: str, base: Path) -> list[str]:
    wrong = []
    cases = {
        "one": (
            ("llm_fsm-0.13.0-py3-none-any.whl", "llm_fsm-0.13.0.tar.gz"),
            "dist/llm_fsm-0.13.0-py3-none-any.whl",
        ),
        "none": (("llm_fsm-0.13.0.tar.gz",), None),
        "two": (("a-1-py3-none-any.whl", "b-1-py3-none-any.whl"), None),
        "no-dist-dir": ((), None),
    }
    for case_id, (files, expected) in cases.items():
        cwd = base / case_id
        cwd.mkdir(parents=True)
        if case_id != "no-dist-dir":
            (cwd / "dist").mkdir()
        for name in files:
            (cwd / "dist" / name).write_text("x", encoding="utf-8")
        code, output, _ = _run(script, cwd)
        if expected is None:
            if code == 0 or output != "":
                wrong.append(case_id)
        elif code != 0 or output.strip() != f"path={expected}":
            wrong.append(case_id)
    return wrong


def _sbom_failures(script: str, base: Path) -> list[str]:
    """Run the Python check of the SBOM step on three synthetic documents."""
    wrong = []
    body = _python_body(script)
    cases = {
        "valid": (
            {
                "bomFormat": "CycloneDX",
                "specVersion": "1.6",
                "components": [{"name": "x"}],
            },
            True,
        ),
        "wrong-format": (
            {"bomFormat": "SPDX", "specVersion": "1.6", "components": [{"name": "x"}]},
            False,
        ),
        "no-components": (
            {"bomFormat": "CycloneDX", "specVersion": "1.6", "components": []},
            False,
        ),
        "no-key": ({"specVersion": "1.6", "components": [{"name": "x"}]}, False),
    }
    for case_id, (document, ok) in cases.items():
        cwd = base / case_id
        cwd.mkdir(parents=True)
        (cwd / "sbom.cdx.json").write_text(json.dumps(document), encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, "-c", body],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if (proc.returncode == 0) != ok:
            wrong.append(case_id)
    return wrong


class TestTheOtherScripts:
    def test_the_release_notes_are_the_section_of_the_version_in_a_file(
        self, tmp_path: Path
    ):
        assert (
            _notes_failures(_script("build", needle="changelog_check.py"), tmp_path)
            == []
        )

    def test_a_section_with_no_body_refuses_the_release(self, tmp_path: Path):
        _notes_tree(tmp_path)
        code, _, err = _run(
            _script("build", needle="changelog_check.py"),
            tmp_path,
            {"VERSION": "0.12.9", **_NOTES_ENV},
        )
        assert code != 0 and "0.12.9" in err

    def test_the_real_changelog_yields_notes_for_unreleased_and_for_the_newest_release(
        self, tmp_path: Path
    ):
        text = (_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        versions = re.findall(r"^## \[(\d+\.\d+\.\d+)\]", text, re.MULTILINE)
        assert versions, "the real CHANGELOG has no dated release"
        # `[Unreleased]` is empty right after a release is cut (it then has no notes to extract).
        pending = ("Unreleased",) if _unreleased_has_notes(text) else ()
        for i, version in enumerate((*pending, versions[0])):
            cwd = tmp_path / str(i)
            (cwd / "scripts").mkdir(parents=True)
            shutil.copy(_ROOT / "scripts" / "changelog_check.py", cwd / "scripts")
            (cwd / "CHANGELOG.md").write_text(text, encoding="utf-8")
            code, _, err = _run(
                _script("build", needle="changelog_check.py"),
                cwd,
                {"VERSION": version, **_NOTES_ENV},
            )
            assert code == 0, (version, err)
            assert (cwd / "release-notes.md").stat().st_size > 0

    def test_short_notes_pass_and_long_notes_are_cut_below_the_github_limit(
        self, tmp_path: Path
    ):
        script = _script("build", needle="limit = 120_000")
        assert _fit_failures(script, tmp_path) == []

    def test_the_real_changelog_notes_fit_after_the_cut(self, tmp_path: Path):
        # The real [Unreleased] section is far above the 125000 characters GitHub accepts.
        extract = _script("build", needle="changelog_check.py")
        (tmp_path / "scripts").mkdir()
        shutil.copy(_ROOT / "scripts" / "changelog_check.py", tmp_path / "scripts")
        shutil.copy(_ROOT / "CHANGELOG.md", tmp_path / "CHANGELOG.md")
        text = (_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        version = "Unreleased" if _unreleased_has_notes(text) else _newest_release(text)
        code, _, err = _run(extract, tmp_path, {"VERSION": version, **_NOTES_ENV})
        assert code == 0, err
        code, _, err = _run(
            _script("build", needle="limit = 120_000"),
            tmp_path,
            {"TAG": "v0.13.0", "REPO": "NikolasMarkou/fsm_llm"},
        )
        assert code == 0, err
        notes = (tmp_path / "release-notes.md").read_text(encoding="utf-8")
        assert 0 < len(notes) <= 125_000
        assert notes.count("```") % 2 == 0

    def test_the_file_name_check(self, tmp_path: Path):
        assert _names_failures(_script("build", needle="LC_ALL=C sort"), tmp_path) == []

    def test_the_name_check_reports_both_listings(self, tmp_path: Path):
        (tmp_path / "dist").mkdir()
        (tmp_path / "dist" / "stray.txt").write_text("x", encoding="utf-8")
        code, _, err = _run(
            _script("build", needle="LC_ALL=C sort"),
            tmp_path,
            {"WHEEL": "a.whl", "SDIST": "a.tar.gz"},
        )
        assert code != 0 and "stray.txt" in err and "a.tar.gz" in err

    def test_the_wheel_finder_needs_exactly_one_wheel(self, tmp_path: Path):
        assert _wheel_failures(_script("sbom", step_id="wheel"), tmp_path) == []

    def test_the_sbom_check_needs_a_cyclonedx_document_with_components(
        self, tmp_path: Path
    ):
        assert _sbom_failures(_script("sbom", needle="cyclonedx-py"), tmp_path) == []


# --------------------------------------------------------------------------------------
# Every shipped script has mutants that its battery kills
# --------------------------------------------------------------------------------------


def _long_linked_changelog() -> str:
    """A 0.13.0 section just under the GitHub limit, with a guide link on every bullet.

    The raw section is below 120000 characters, so a cut BEFORE the links are made absolute would
    keep all of it; the absolute links add about 54 characters each and push it far above the
    limit, so only a cut AFTER the rewrite keeps the body under the GitHub limit.
    """
    head = (
        "# Changelog\n\n## [Unreleased]\n\n## [0.13.0] - 2026-11-20\n\n### Changed\n\n"
    )
    lines: list[str] = []
    size = 0
    while True:
        i = len(lines)
        line = (
            f"- **BREAKING:** item {i:05d} "
            + "word " * 12
            + f"[Upgrade guide](docs/upgrade-guide/v0.13.0/page-{i % 7}.md#section-{i})\n"
        )
        if size + len(line) > 119_000:
            return head + "".join(lines)
        lines.append(line)
        size += len(line)


_BULLET = re.compile(
    r"- \*\*BREAKING:\*\* item \d{5} (?:word ){12}"
    r"\[Upgrade guide\]\(https://github\.com/NikolasMarkou/fsm_llm/blob/v0\.13\.0/"
    r"docs/upgrade-guide/v0\.13\.0/page-\d\.md#section-\d+\)"
)


def _final_notes_problems(notes: str) -> list[str]:
    """What is wrong with the notes `gh release create` would receive, as short phrases."""
    problems: list[str] = []
    if len(notes) > 125_000:
        problems.append("over the GitHub limit")
    if _relative_doc_links(notes):
        problems.append("a relative docs link is left")
    if "These notes are cut at the GitHub release limit." not in notes:
        problems.append("not cut")
    kept = notes.split("\n\n---\n\n")[0]
    for line in kept.splitlines():
        if line.startswith("- ") and not _BULLET.fullmatch(line):
            problems.append("a damaged bullet: ..." + line[-50:])
            break
    return problems


def _notes_pipeline(
    cwd: Path, extract: str, fit: str, *, rewrite_late: bool = False
) -> str:
    """Run the extract step and the cut step the way the workflow orders them; the notes."""
    _notes_tree(cwd, _long_linked_changelog())
    env = {"VERSION": "0.13.0", **_NOTES_ENV}
    code, _, err = _run(extract, cwd, env)
    assert code == 0, err
    code, _, err = _run(fit, cwd, _NOTES_ENV)
    assert code == 0, err
    notes = (cwd / "release-notes.md").read_text(encoding="utf-8")
    if rewrite_late:  # the wrong order: the links are made absolute after the cut
        module = _load_checker()
        notes = module.rewrite_doc_links(notes, _LINK_BASE)
    return notes


def _load_checker() -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_changelog_check_notes", _ROOT / "scripts" / "changelog_check.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


class TestTheReleaseNotesLinks:
    """The body of the GitHub release links the tag, never the reader's working tree (step 118)."""

    def test_the_notes_after_the_extract_and_the_cut_are_sound(self, tmp_path: Path):
        notes = _notes_pipeline(
            tmp_path,
            _script("build", needle="changelog_check.py"),
            _script("build", needle="limit = 120_000"),
        )
        assert _final_notes_problems(notes) == []
        assert (
            notes.count("](https://github.com/NikolasMarkou/fsm_llm/blob/v0.13.0/docs/")
            > 500
        )

    def test_the_fixture_is_below_the_limit_before_the_rewrite_and_above_after(
        self, tmp_path: Path
    ):
        body = _load_checker().extract_text(_long_linked_changelog(), "0.13.0")
        assert len(body) < 120_000
        assert len(_load_checker().rewrite_doc_links(body, _LINK_BASE)) > 125_000

    def test_links_made_absolute_after_the_cut_break_the_limit(self, tmp_path: Path):
        # The wrong order: extract WITHOUT the base (so the raw 119k characters are not cut), then
        # the rewrite. The body is over the limit and not cut.
        plain = _script("build", needle="changelog_check.py").replace(
            ' --link-base "https://github.com/${REPO}/blob/${TAG}/"', ""
        )
        notes = _notes_pipeline(
            tmp_path,
            plain,
            _script("build", needle="limit = 120_000"),
            rewrite_late=True,
        )
        problems = _final_notes_problems(notes)
        assert "over the GitHub limit" in problems and "not cut" in problems

    def test_notes_with_no_base_keep_relative_links(self, tmp_path: Path):
        plain = _script("build", needle="changelog_check.py").replace(
            ' --link-base "https://github.com/${REPO}/blob/${TAG}/"', ""
        )
        notes = _notes_pipeline(
            tmp_path, plain, _script("build", needle="limit = 120_000")
        )
        assert "a relative docs link is left" in _final_notes_problems(notes)

    def test_the_real_changelog_notes_have_no_relative_link_after_the_steps(
        self, tmp_path: Path
    ):
        (tmp_path / "scripts").mkdir()
        shutil.copy(_ROOT / "scripts" / "changelog_check.py", tmp_path / "scripts")
        shutil.copy(_ROOT / "CHANGELOG.md", tmp_path / "CHANGELOG.md")
        text = (tmp_path / "CHANGELOG.md").read_text(encoding="utf-8")
        version = "Unreleased" if _unreleased_has_notes(text) else _newest_release(text)
        code, _, err = _run(
            _script("build", needle="changelog_check.py"),
            tmp_path,
            {"VERSION": version, **_NOTES_ENV},
        )
        assert code == 0, err
        code, _, err = _run(
            _script("build", needle="limit = 120_000"), tmp_path, _NOTES_ENV
        )
        assert code == 0, err
        notes = (tmp_path / "release-notes.md").read_text(encoding="utf-8")
        assert _relative_doc_links(notes) == []
        assert len(notes) <= 125_000


def _real_notes(cwd: Path, version: str) -> tuple[str, str]:
    """The notes of the REAL changelog after the extract and the cut steps: (before, after the cut)."""
    (cwd / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy(_ROOT / "scripts" / "changelog_check.py", cwd / "scripts")
    shutil.copy(_ROOT / "CHANGELOG.md", cwd / "CHANGELOG.md")
    env = {"VERSION": version, "TAG": f"v{version}", "REPO": "NikolasMarkou/fsm_llm"}
    code, _, err = _run(_script("build", needle="changelog_check.py"), cwd, env)
    assert code == 0, err
    before = (cwd / "release-notes.md").read_text(encoding="utf-8")
    code, _, err = _run(_script("build", needle="limit = 120_000"), cwd, env)
    assert code == 0, err
    return before, (cwd / "release-notes.md").read_text(encoding="utf-8")


class TestTheRealReleaseNotes:
    """The body of the release the tag will publish, read from the real files (step 119).

    The section of 0.13.0 is far longer than a release body can hold, so the workflow cuts it at a
    line end and points at the full text. What must survive that cut is the Highlights block at
    the top: the short version, the upgrade guides and the pointer to the full list.
    """

    def test_the_notes_open_with_the_highlights_and_the_cut_keeps_them(
        self, tmp_path: Path
    ):
        version = tomllib.loads((_ROOT / "pyproject.toml").read_text("utf-8"))[
            "project"
        ]["version"]
        before, after = _real_notes(tmp_path, version)
        assert before.startswith("**Highlights**\n"), before[:80]
        first_heading = after.index("\n### ")
        highlights = after[:first_heading]
        assert before.startswith(highlights)
        assert len(highlights) < 8_000
        # The cut replaced the tail, kept the head whole and ends with the pointer.
        assert len(before) > 120_000, "the premise of the cut: the section is long"
        assert len(after) < len(before)
        assert after.rstrip().endswith(
            f"[CHANGELOG.md](https://github.com/NikolasMarkou/fsm_llm/blob/v{version}/CHANGELOG.md)."
        )
        assert "These notes are cut at the GitHub release limit." in after
        assert _relative_doc_links(after) == []
        assert len(after) <= 125_000
        assert (
            f"](https://github.com/NikolasMarkou/fsm_llm/blob/v{version}/docs/upgrade-guide/index.md)"
            in highlights
        )

    def test_the_notes_keep_the_first_sections_after_the_highlights(
        self, tmp_path: Path
    ):
        _, after = _real_notes(tmp_path, "0.13.0")
        head, _, rest = after.partition("\n### Added\n")
        assert head.startswith("**Highlights**") and rest, (
            "### Added follows the Highlights"
        )

    def test_the_unreleased_section_is_empty_so_its_extraction_fails_by_design(
        self, tmp_path: Path
    ):
        (tmp_path / "scripts").mkdir()
        shutil.copy(_ROOT / "scripts" / "changelog_check.py", tmp_path / "scripts")
        shutil.copy(_ROOT / "CHANGELOG.md", tmp_path / "CHANGELOG.md")
        env = {"VERSION": "Unreleased", **_NOTES_ENV}
        code, _, _ = _run(_script("build", needle="changelog_check.py"), tmp_path, env)
        assert code != 0, "an empty [Unreleased] must not produce release notes"

    def test_the_guard_accepts_the_tag_of_the_release_and_names_the_files(
        self, tmp_path: Path
    ):
        (tmp_path / "src" / "fsm_llm").mkdir(parents=True)
        shutil.copy(
            _ROOT / "src" / "fsm_llm" / "__version__.py",
            tmp_path / "src" / "fsm_llm" / "__version__.py",
        )
        shutil.copy(_ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
        code, output, err = _run(
            _script("build", step_id="version"), tmp_path, {"TAG": "v0.13.0"}
        )
        assert code == 0, err
        assert output.splitlines() == [
            "version=0.13.0",
            "wheel=llm_fsm-0.13.0-py3-none-any.whl",
            "sdist=llm_fsm-0.13.0.tar.gz",
        ]

    def test_the_guard_refuses_the_tag_of_the_previous_release(self, tmp_path: Path):
        (tmp_path / "src" / "fsm_llm").mkdir(parents=True)
        shutil.copy(
            _ROOT / "src" / "fsm_llm" / "__version__.py",
            tmp_path / "src" / "fsm_llm" / "__version__.py",
        )
        shutil.copy(_ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
        code, output, err = _run(
            _script("build", step_id="version"), tmp_path, {"TAG": "v0.12.0"}
        )
        assert code != 0 and output == ""
        assert "expected v0.13.0" in err


def _swap(old: str, new: str, count: int = 1) -> Callable[[str], str]:
    def apply(script: str) -> str:
        assert old in script, f"script mutant anchor not found: {old!r}"
        return script.replace(old, new, count)

    return apply


GUARD_SCRIPT_MUTANTS: list[tuple[str, Callable[[str], str]]] = [
    ("tag comparison removed", _swap('if tag != "v" + declared:', "if False:")),
    (
        "pre-release check removed",
        _swap("if parsed.is_prerelease or parsed.is_devrelease:", "if False:"),
    ),
    (
        "pre-releases not refused",
        _swap("parsed.is_prerelease or parsed.is_devrelease", "parsed.is_devrelease"),
    ),
    ("local versions accepted", _swap(" or parsed.local is not None", "")),
    ("epochs accepted", _swap(" or parsed.epoch", "")),
    ("non-canonical forms accepted", _swap("str(parsed) != declared or ", "")),
    (
        "wrong version file",
        _swap("src/fsm_llm/__version__.py", "src/fsm_llm/__init__.py"),
    ),
    (
        "project name not normalised",
        _swap('re.sub(r"[-_.]+", "_", project).lower()', "project"),
    ),
    (
        "normalised without lower",
        _swap(
            're.sub(r"[-_.]+", "_", project).lower()', 're.sub(r"[-_.]+", "_", project)'
        ),
    ),
    ("hyphen only normalised", _swap('r"[-_.]+"', 'r"-+"')),
    ("wheel tag wrong", _swap("-py3-none-any.whl", "-py3-any.whl")),
    (
        "invalid version accepted",
        _swap("except InvalidVersion:", "except ZeroDivisionError:"),
    ),
    (
        "a missing assignment ignored",
        _swap('sys.exit(f"error: no __version__ string assignment in {path}")', "pass"),
    ),
    (
        "output written before the checks",
        _swap(
            'with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as out:',
            'with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as out:\n    out.write("version=early\\n")\nwith open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as out:',
        ),
    ),
]

NOTES_SCRIPT_MUTANTS: list[tuple[str, Callable[[str], str]]] = [
    ("the output goes to stdout", _swap(" > release-notes.md", "")),
    ("the section is not required", _swap("extract", "check")),
    (
        "a failed extraction ignored",
        lambda s: s.replace(
            " > release-notes.md", " > release-notes.md || true", 1
        ).replace("test -s release-notes.md", "true", 1),
    ),
    ("the wrong version extracted", _swap('extract "$VERSION"', "extract Unreleased")),
    (
        "the link base option missing",
        _swap(' --link-base "https://github.com/${REPO}/blob/${TAG}/"', ""),
    ),
    ("the links pinned to the main branch", _swap("blob/${TAG}/", "blob/main/")),
    (
        "the links name another repository",
        _swap("github.com/${REPO}/", "github.com/example/"),
    ),
    ("the links lose the tag directory", _swap("/blob/${TAG}/", "/")),
]

FIT_SCRIPT_MUTANTS: list[tuple[str, Callable[[str], str]]] = [
    ("a limit above what GitHub accepts", _swap("limit = 120_000", "limit = 130_000")),
    ("long notes not cut", _swap("if len(text) > limit:", "if False:")),
    ("short notes rewritten", _swap("if len(text) > limit:", "if len(text) > 0:")),
    ("an open code fence not closed", _swap('if cut.count("```") % 2:', "if False:")),
    ("the pointer names no repository", _swap("os.environ['REPO']", "'example'")),
    ("the pointer names no tag", _swap("os.environ['TAG']", "'main'")),
    ("the cut keeps a partial line", _swap('.rsplit("\\n", 1)[0]', "")),
    (
        "a link text left open at the cut",
        _swap("while (open_link :=", "while False and (open_link :="),
    ),
    (
        "the cut file is not written",
        _swap('path.write_text(text, encoding="utf-8")', "pass"),
    ),
]

NAMES_SCRIPT_MUTANTS: list[tuple[str, Callable[[str], str]]] = [
    ("a mismatch does not fail", _swap("exit 1", "exit 0")),
    (
        "the comparison is negated",
        _swap('[ "$actual" != "$wanted" ]', '[ "$actual" == "$wanted" ]'),
    ),
    (
        "only the wheel is expected",
        _swap('printf \'%s\\n\' "$SDIST" "$WHEEL"', "printf '%s\\n' \"$WHEEL\""),
    ),
    (
        "only the wheel is checked",
        _swap("-maxdepth 1 ", "-maxdepth 1 -name '*.whl' "),
    ),
]

WHEEL_SCRIPT_MUTANTS: list[tuple[str, Callable[[str], str]]] = [
    ("two wheels pass", _swap("-ne 1", "-gt 2")),
    (
        "no wheel passes",
        lambda s: s.replace("-ne 1", "-gt 1", 1).replace(
            "set -euo pipefail", "set -eo pipefail", 1
        ),
    ),
    ("the mismatch does not fail", _swap("exit 1", "exit 0")),
    ("the wrong file reported", _swap("${wheels[0]}", "${wheels[1]}")),
]

SBOM_SCRIPT_MUTANTS: list[tuple[str, Callable[[str], str]]] = [
    (
        "the format is not checked",
        _swap('assert bom["bomFormat"] == "CycloneDX", bom["bomFormat"]', "pass"),
    ),
    (
        "no components pass",
        _swap('assert bom["components"], "the SBOM lists no component"', "pass"),
    ),
]


class TestTheShippedScriptsAreMutationChecked:
    @pytest.mark.parametrize(
        "label,mutate", GUARD_SCRIPT_MUTANTS, ids=[m[0] for m in GUARD_SCRIPT_MUTANTS]
    )
    def test_the_guard_battery_kills_each_mutant(
        self, tmp_path: Path, label: str, mutate: Callable[[str], str]
    ):
        script = _script("build", step_id="version")
        mutated = mutate(script)
        assert mutated != script
        assert _guard_failures(mutated, tmp_path), label

    @pytest.mark.parametrize(
        "label,mutate", NOTES_SCRIPT_MUTANTS, ids=[m[0] for m in NOTES_SCRIPT_MUTANTS]
    )
    def test_the_notes_battery_kills_each_mutant(
        self, tmp_path: Path, label: str, mutate: Callable[[str], str]
    ):
        script = _script("build", needle="changelog_check.py")
        mutated = mutate(script)
        assert mutated != script
        assert _notes_failures(mutated, tmp_path), label

    @pytest.mark.parametrize(
        "label,mutate", FIT_SCRIPT_MUTANTS, ids=[m[0] for m in FIT_SCRIPT_MUTANTS]
    )
    def test_the_fit_battery_kills_each_mutant(
        self, tmp_path: Path, label: str, mutate: Callable[[str], str]
    ):
        script = _script("build", needle="limit = 120_000")
        mutated = mutate(script)
        assert mutated != script
        assert _fit_failures(mutated, tmp_path), label

    @pytest.mark.parametrize(
        "label,mutate", NAMES_SCRIPT_MUTANTS, ids=[m[0] for m in NAMES_SCRIPT_MUTANTS]
    )
    def test_the_names_battery_kills_each_mutant(
        self, tmp_path: Path, label: str, mutate: Callable[[str], str]
    ):
        script = _script("build", needle="LC_ALL=C sort")
        mutated = mutate(script)
        assert mutated != script
        assert _names_failures(mutated, tmp_path), label

    @pytest.mark.parametrize(
        "label,mutate", WHEEL_SCRIPT_MUTANTS, ids=[m[0] for m in WHEEL_SCRIPT_MUTANTS]
    )
    def test_the_wheel_battery_kills_each_mutant(
        self, tmp_path: Path, label: str, mutate: Callable[[str], str]
    ):
        script = _script("sbom", step_id="wheel")
        mutated = mutate(script)
        assert mutated != script
        assert _wheel_failures(mutated, tmp_path), label

    @pytest.mark.parametrize(
        "label,mutate", SBOM_SCRIPT_MUTANTS, ids=[m[0] for m in SBOM_SCRIPT_MUTANTS]
    )
    def test_the_sbom_battery_kills_each_mutant(
        self, tmp_path: Path, label: str, mutate: Callable[[str], str]
    ):
        script = _script("sbom", needle="cyclonedx-py")
        mutated = mutate(script)
        assert mutated != script
        assert _sbom_failures(mutated, tmp_path), label


# --------------------------------------------------------------------------------------
# The distribution name and the tool pins
# --------------------------------------------------------------------------------------


def _pyproject() -> dict[str, Any]:
    return tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _pins() -> dict[str, list[str]]:
    pins: dict[str, list[str]] = {}
    for line in (_ROOT / "constraints.txt").read_text(encoding="utf-8").splitlines():
        match = re.match(
            r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([0-9][A-Za-z0-9.!+_-]*)$", line.strip()
        )
        if match:
            pins.setdefault(re.sub(r"[-_.]+", "-", match[1]).lower(), []).append(
                match[2]
            )
    return pins


class TestTheDistributionName:
    def test_the_environment_url_names_the_llm_fsm_project(self):
        # A trusted publisher is registered per project name, so the publish job can only
        # succeed when the environment url and `[project] name` are both `llm-fsm`.
        publish = _load(_real_text())["jobs"]["publish"]
        assert publish["environment"]["url"] == "https://pypi.org/p/llm-fsm"

    def test_the_project_name_is_llm_fsm(self):
        # Tightened at step 113 (the rename): it accepted both names before.
        assert _pyproject()["project"]["name"] == "llm-fsm"

    def test_the_publish_url_matches_the_project(self):
        name = _pyproject()["project"]["name"]
        url = _load(_real_text())["jobs"]["publish"]["environment"]["url"]
        assert url == f"https://pypi.org/p/{name}"


class TestTheToolPins:
    @pytest.mark.parametrize("tool", ["build", "twine", "cyclonedx-bom"])
    def test_each_release_tool_has_one_exact_pin(self, tool: str):
        assert len(_pins().get(tool, [])) == 1, tool

    def test_twine_and_cyclonedx_are_ci_tools_in_no_extra(self):
        extras = _pyproject()["project"]["optional-dependencies"]
        listed = " ".join(" ".join(v) for v in extras.values()).lower()
        assert "twine" not in listed and "cyclonedx" not in listed

    def test_the_workflow_installs_the_tools_only_through_the_pins(self):
        code = "\n".join(_code_lines(_real_text()))
        for line in code.splitlines():
            if "pip install" in line:
                assert "-c constraints.txt" in line, line


# --------------------------------------------------------------------------------------
# The owner steps are written down
# --------------------------------------------------------------------------------------


class TestTheOwnerSteps:
    def test_contributing_names_the_publisher_the_environment_and_the_tag_flow(self):
        text = (_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
        for needle in (
            "release.yml",
            "NikolasMarkou",
            "fsm_llm",
            "environment",
            "pypi",
            "v*",
            "required reviewer",
            "attestation",
            "SBOM",
            "git tag -a",
            "git push origin",
        ):
            assert needle in text, needle

    def test_contributing_says_the_workflow_never_runs_from_a_branch_or_a_dispatch(
        self,
    ):
        text = (_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8").lower()
        assert "only a pushed tag" in text or "only from a tag" in text


# --------------------------------------------------------------------------------------
# Mutants: each edit of the real text must be caught by its own rule
# --------------------------------------------------------------------------------------


def _unpin_publish(text: str) -> str:
    mutated = re.sub(
        r"pypa/gh-action-pypi-publish@[0-9a-f]{40}",
        "pypa/gh-action-pypi-publish@release/v1",
        text,
    )
    assert mutated != text
    return mutated


def _short_sha(text: str) -> str:
    mutated = re.sub(
        r"(actions/upload-artifact@[0-9a-f]{39})[0-9a-f]", r"\1", text, count=1
    )
    assert mutated != text
    return mutated


def _artifact_v3(text: str) -> str:
    mutated = re.sub(
        r"(actions/upload-artifact@[0-9a-f]{40}) # v7\.0\.2", r"\1 # v3.1.0", text
    )
    assert mutated != text
    return mutated


def _two_pins(text: str) -> str:
    # The second download-artifact line points at another commit: one action, two pins.
    old = "actions/download-artifact@9000827ccba6bdab643e8b6fd33ac0654aef8333"
    assert text.count(old) >= 2
    return text.replace(old, "actions/download-artifact@" + "a" * 40, 1)


_BUILD_HEAD = "  build:\n    runs-on: ubuntu-24.04\n"
_SBOM_HEAD = "  sbom:\n    needs: build\n    runs-on: ubuntu-24.04\n"
_RELEASE_HEAD = (
    "  github-release:\n    needs: [publish, sbom]\n    runs-on: ubuntu-24.04\n"
)

MUTANTS: list[tuple[str, Any, str]] = [
    ("workflow renamed", _edit("name: release\n", "name: Release\n"), "name"),
    (
        "top permissions widened",
        _edit(
            "permissions:\n  contents: read\n\nconcurrency",
            "permissions:\n  contents: write\n\nconcurrency",
        ),
        "top-permissions",
    ),
    (
        "top permissions removed",
        _edit("permissions:\n  contents: read\n\nconcurrency", "concurrency"),
        "top-permissions",
    ),
    (
        "id-token at the top level",
        _edit(
            "permissions:\n  contents: read\n\nconcurrency",
            "permissions:\n  contents: read\n  id-token: write\n\nconcurrency",
        ),
        "id-token-scope",
    ),
    (
        "id-token on the build job",
        _edit(_BUILD_HEAD, _BUILD_HEAD + "    permissions:\n      id-token: write\n"),
        "id-token-scope",
    ),
    (
        "id-token on the sbom job",
        _edit(_SBOM_HEAD, _SBOM_HEAD + "    permissions:\n      id-token: write\n"),
        "id-token-scope",
    ),
    (
        "id-token on the release job",
        _edit(
            "      contents: write\n", "      contents: write\n      id-token: write\n"
        ),
        "id-token-scope",
    ),
    (
        "contents write on the publish job",
        _edit(
            "      id-token: write\n", "      id-token: write\n      contents: write\n"
        ),
        "publish-permissions",
    ),
    (
        "contents write on the build job",
        _edit(_BUILD_HEAD, _BUILD_HEAD + "    permissions:\n      contents: write\n"),
        "extra-write",
    ),
    (
        "any block on the sbom job",
        _edit(_SBOM_HEAD, _SBOM_HEAD + "    permissions:\n      contents: read\n"),
        "job-permissions",
    ),
    (
        "id-token dropped from the publish job",
        _edit("      id-token: write\n", "      id-token: none\n"),
        "publish-permissions",
    ),
    (
        "release permission lowered",
        _edit("      contents: write\n", "      contents: read\n"),
        "release-permissions",
    ),
    (
        "packages write added to the release job",
        _edit(
            "      contents: write\n", "      contents: write\n      packages: write\n"
        ),
        "extra-write",
    ),
    (
        "environment renamed",
        _edit("      name: pypi\n", "      name: release\n"),
        "environment",
    ),
    (
        "environment url changed to another project",
        _edit("url: https://pypi.org/p/llm-fsm", "url: https://pypi.org/p/fsm-llm"),
        "environment",
    ),
    (
        "environment url changed to a mirror",
        _edit(
            "url: https://pypi.org/p/llm-fsm", "url: https://test.pypi.org/p/llm-fsm"
        ),
        "environment",
    ),
    (
        "environment removed from the publish job",
        _edit(
            "    environment:\n      name: pypi\n      url: https://pypi.org/p/llm-fsm\n",
            "",
        ),
        "environment",
    ),
    (
        "environment added to the build job",
        _edit(_BUILD_HEAD, _BUILD_HEAD + "    environment: pypi\n"),
        "extra-environment",
    ),
    (
        "a password on the publish action",
        _edit(
            "          attestations: true\n",
            "          attestations: true\n          password: ${{ secrets.PYPI_API_TOKEN }}\n",
        ),
        "no-token",
    ),
    (
        "a twine upload step",
        _edit(
            "run: twine check --strict dist/*",
            "run: twine check --strict dist/* && twine upload dist/*",
        ),
        "no-token",
    ),
    (
        "a secret named for the default token",
        _edit("GH_TOKEN: ${{ github.token }}", "GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}"),
        "no-token",
    ),
    (
        "the default token named twice",
        _edit(
            "          GH_REPO: ${{ github.repository }}\n",
            "          GH_REPO: ${{ github.repository }}\n          OTHER: ${{ github.token }}\n",
        ),
        "token-scope",
    ),
    (
        "the tag guard removed",
        _edit('if tag != "v" + declared:', 'if tag == "":'),
        "guard-tag",
    ),
    (
        "the final-release guard removed",
        _edit("parsed.is_prerelease or parsed.is_devrelease", "False"),
        "guard-final",
    ),
    (
        "the local-version guard removed",
        _edit("parsed.local is not None", "False"),
        "guard-final",
    ),
    (
        "the guard reads the wrong file",
        _edit(
            'pathlib.Path("src/fsm_llm/__version__.py")',
            'pathlib.Path("src/fsm_llm/__init__.py")',
        ),
        "guard-version-file",
    ),
    (
        "the guard stops deriving the file names",
        _edit('re.sub(r"[-_.]+", "_", project).lower()', "project"),
        "guard-names",
    ),
    (
        "the tag read from the head ref",
        _edit(
            "TAG: ${{ github.ref_name }}\n        run: |\n          set -euo",
            "TAG: ${{ github.head_ref }}\n        run: |\n          set -euo",
        ),
        "guard-env",
    ),
    (
        "the release-notes extraction removed",
        _edit(
            _EXTRACT_COMMAND,
            _EXTRACT_COMMAND.replace("extract", "check", 1),
        ),
        "extract",
    ),
    (
        "the extraction not written to a file",
        _edit(_EXTRACT_COMMAND, _EXTRACT_COMMAND.removesuffix(" > release-notes.md")),
        "extract",
    ),
    (
        "the link base option removed from the extraction",
        _edit(' --link-base "https://github.com/${REPO}/blob/${TAG}/"', ""),
        "extract",
    ),
    (
        "the link base pinned to the main branch",
        _edit(
            '--link-base "https://github.com/${REPO}/blob/${TAG}/"',
            '--link-base "https://github.com/${REPO}/blob/main/"',
        ),
        "extract",
    ),
    (
        "the link base names another repository",
        _edit(
            '--link-base "https://github.com/${REPO}/blob/${TAG}/"',
            '--link-base "https://github.com/example/example/blob/${TAG}/"',
        ),
        "extract",
    ),
    (
        "the link base takes the tag from an expression in the script",
        _edit(
            '--link-base "https://github.com/${REPO}/blob/${TAG}/"',
            '--link-base "https://github.com/${REPO}/blob/${{ github.ref_name }}/"',
        ),
        "extract",
    ),
    (
        "the extraction no longer gets the tag",
        _edit(
            "          TAG: ${{ github.ref_name }}\n          REPO: ${{ github.repository }}\n        run: |\n          set -euo pipefail\n          python scripts/changelog_check.py extract",
            "          REPO: ${{ github.repository }}\n        run: |\n          set -euo pipefail\n          python scripts/changelog_check.py extract",
        ),
        "extract",
    ),
    (
        "the links made absolute after the cut (the cut step runs first)",
        lambda text: _fit_before_extract(text),
        "notes-order",
    ),
    (
        "the cut leaves a link text open",
        _edit(_OPEN_LINK_LOOP, "while False:"),
        "notes-limit",
    ),
    (
        "the empty-notes check removed",
        _edit("          test -s release-notes.md\n", ""),
        "extract",
    ),
    (
        "strict removed from twine",
        _edit("twine check --strict dist/*", "twine check dist/*"),
        "twine-strict",
    ),
    (
        "twine check removed",
        _edit("run: twine check --strict dist/*", "run: echo skipped"),
        "twine-strict",
    ),
    (
        "the build backend unpinned",
        _edit(
            "          PIP_CONSTRAINT: ${{ github.workspace }}/constraints.txt\n", ""
        ),
        "build-command",
    ),
    (
        "the build tools installed without the pins",
        _edit("pip install -c constraints.txt build twine", "pip install build twine"),
        "build-install",
    ),
    (
        "the file-name check cannot fail",
        _edit(
            "              exit 1\n          fi\n      - name: Upload the distributions",
            "              exit 0\n          fi\n      - name: Upload the distributions",
        ),
        "names-check",
    ),
    (
        "attestations switched off",
        _edit("attestations: true", "attestations: false"),
        "publish-attestations",
    ),
    (
        "attestations left to the default",
        _edit("          attestations: true\n", ""),
        "publish-attestations",
    ),
    (
        "skip-existing on the publish action",
        _edit(
            "          attestations: true\n",
            "          attestations: true\n          skip-existing: true\n",
        ),
        "publish-with",
    ),
    (
        "another packages directory",
        _edit("packages-dir: dist/", "packages-dir: ."),
        "publish-with",
    ),
    (
        "a script step in the publish job",
        _edit(
            "      - name: Publish to PyPI\n",
            "      - name: Say hello\n        run: echo hello\n      - name: Publish to PyPI\n",
        ),
        "publish-steps",
    ),
    ("an action pinned to a tag", _unpin_checkout, "action-pin"),
    ("the publish action pinned to a branch", _unpin_publish, "action-pin"),
    ("an action pin truncated", _short_sha, "action-pin"),
    ("an action pin without its tag comment", _drop_comment, "action-tag-comment"),
    ("the artifact action at v3", _artifact_v3, "artifact-version"),
    ("one action with two pins", _two_pins, "action-set"),
    (
        "pull_request_target",
        _edit(
            'on:\n  push:\n    tags: ["v*"]\n',
            'on:\n  pull_request_target:\n  push:\n    tags: ["v*"]\n',
        ),
        "pull-request-target",
    ),
    (
        "a container on the publish job",
        _edit(
            "    environment:\n      name: pypi\n",
            "    container: python:3.12\n    environment:\n      name: pypi\n",
        ),
        "publish-isolation",
    ),
    (
        "a service on the publish job",
        _edit(
            "    environment:\n      name: pypi\n",
            "    services:\n      redis:\n        image: redis\n    environment:\n      name: pypi\n",
        ),
        "publish-isolation",
    ),
    (
        "a reusable workflow as the publish job",
        _edit(
            "    environment:\n      name: pypi\n",
            "    uses: ./.github/workflows/publish.yml\n    environment:\n      name: pypi\n",
        ),
        "publish-isolation",
    ),
    (
        "the publish job on a floating runner",
        _edit(
            "    needs: [build, sbom]\n    runs-on: ubuntu-24.04\n",
            "    needs: [build, sbom]\n    runs-on: ubuntu-latest\n",
        ),
        "publish-isolation",
    ),
    (
        "a manual dispatch trigger",
        _edit("on:\n  push:\n", "on:\n  workflow_dispatch:\n  push:\n"),
        "extra-trigger",
    ),
    (
        "a branch push trigger",
        _edit(
            'on:\n  push:\n    tags: ["v*"]\n',
            'on:\n  push:\n    tags: ["v*"]\n    branches: ["main"]\n',
        ),
        "push-trigger",
    ),
    (
        "every tag triggers the release",
        _edit('tags: ["v*"]', 'tags: ["*"]'),
        "push-trigger",
    ),
    (
        "a pull request trigger",
        _edit("on:\n  push:\n", "on:\n  pull_request:\n  push:\n"),
        "extra-trigger",
    ),
    (
        "the release not waiting for the publish",
        _edit("needs: [publish, sbom]", "needs: [sbom]"),
        "release-needs",
    ),
    (
        "the release not waiting for the sbom",
        _edit("needs: [publish, sbom]", "needs: [publish]"),
        "release-needs",
    ),
    (
        "the publish not waiting for the build",
        _edit("needs: [build, sbom]", "needs: [sbom]"),
        "publish-needs",
    ),
    (
        "the publish not waiting for the sbom",
        _edit("needs: [build, sbom]", "needs: [build]"),
        "publish-needs",
    ),
    (
        "the sbom not waiting for the build",
        _edit(_SBOM_HEAD, "  sbom:\n    runs-on: ubuntu-24.04\n"),
        "sbom-needs",
    ),
    (
        "the sbom installed without the extras",
        _edit('"${WHEEL}[all]"', '"$WHEEL"'),
        "sbom-extras",
    ),
    (
        "the sbom installed without the pins",
        _edit(
            '-m pip install -c constraints.txt "${WHEEL}[all]"',
            '-m pip install "${WHEEL}[all]"',
        ),
        "sbom-extras",
    ),
    (
        "the sbom tool unpinned",
        _edit(
            "pip install -c constraints.txt cyclonedx-bom", "pip install cyclonedx-bom"
        ),
        "sbom-tool",
    ),
    (
        "the sbom of another input",
        _edit("cyclonedx-py environment", "cyclonedx-py requirements"),
        "sbom-command",
    ),
    (
        "the sbom as XML",
        _edit("--of JSON", "--of XML"),
        "sbom-command",
    ),
    (
        "the sbom not validated",
        _edit('assert bom["bomFormat"] == "CycloneDX", bom["bomFormat"]', "pass"),
        "sbom-validate",
    ),
    (
        "the wheel finder removed",
        _edit(
            '          if [ "${#wheels[@]}" -ne 1 ]; then', "          if false; then"
        ),
        "sbom-wheel",
    ),
    (
        "an artifact that may be empty",
        _edit(
            "          if-no-files-found: error\n",
            "          if-no-files-found: warn\n",
        ),
        "upload-settings",
    ),
    (
        "an artifact that never expires",
        _edit("          retention-days: 14\n", "          retention-days: 0\n"),
        "upload-settings",
    ),
    (
        "the dist artifact uploaded from another path",
        _edit("          path: dist/\n", "          path: .\n"),
        "upload-settings",
    ),
    (
        "the sbom left out of the release",
        _edit(" sbom/sbom.cdx.json", ""),
        "release-command",
    ),
    (
        "the tag not verified",
        _edit(" --verify-tag", ""),
        "release-command",
    ),
    (
        "the notes not attached",
        _edit(" --notes-file notes/release-notes.md", ""),
        "release-command",
    ),
    (
        "the repository not named for gh",
        _edit("          GH_REPO: ${{ github.repository }}\n", ""),
        "release-env",
    ),
    (
        "the release job checks the repository out",
        _edit(
            "    steps:\n      - name: Download the distributions\n        uses: actions/download-artifact@9000827ccba6bdab643e8b6fd33ac0654aef8333 # v8.0.2\n        with:\n          name: dist\n          path: dist\n      - name: Download the SBOM",
            "    steps:\n      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1\n      - name: Download the distributions\n        uses: actions/download-artifact@9000827ccba6bdab643e8b6fd33ac0654aef8333 # v8.0.2\n        with:\n          name: dist\n          path: dist\n      - name: Download the SBOM",
        ),
        "release-checkout",
    ),
    (
        "the release downloads dropped",
        _edit(
            "          name: release-notes\n          path: notes\n",
            "          name: release-notes\n          path: .\n",
        ),
        "release-download",
    ),
    (
        "ubuntu-latest on the build job",
        _edit(_BUILD_HEAD, "  build:\n    runs-on: ubuntu-latest\n"),
        "runner",
    ),
    (
        "timeout removed from the sbom job",
        _edit(
            "  sbom:\n    needs: build\n    runs-on: ubuntu-24.04\n    timeout-minutes: 20\n",
            "  sbom:\n    needs: build\n    runs-on: ubuntu-24.04\n",
        ),
        "timeout",
    ),
    (
        "ref name interpolated into a script",
        _edit(
            "run: twine check --strict dist/*",
            'run: echo "${{ github.ref_name }}" && twine check --strict dist/*',
        ),
        "run-interpolation",
    ),
    (
        "pipefail dropped from the extraction",
        _edit(
            "          set -euo pipefail\n          python scripts/changelog_check.py",
            "          python scripts/changelog_check.py",
        ),
        "shell-strict",
    ),
    (
        "credentials persisted on the build checkout",
        _edit(
            "          persist-credentials: false\n",
            "          persist-credentials: true\n",
        ),
        "build-persist-credentials",
    ),
    (
        "credentials persisted on the sbom checkout",
        lambda t: _nth_edit(
            t,
            "          persist-credentials: false\n",
            "          persist-credentials: true\n",
            1,
        ),
        "sbom-persist-credentials",
    ),
    (
        "python 3.11 for the build",
        _edit('python-version: "3.12"', 'python-version: "3.11"'),
        "build-python",
    ),
    (
        "a pip cache on the release build",
        _edit(
            'python-version: "3.12"\n', 'python-version: "3.12"\n          cache: pip\n'
        ),
        "build-python",
    ),
    (
        "concurrency that cancels a release",
        _edit("cancel-in-progress: false", "cancel-in-progress: true"),
        "concurrency",
    ),
    (
        "default shell removed",
        _edit("defaults:\n  run:\n    shell: bash\n\n", ""),
        "default-shell",
    ),
    (
        "the upload moved before the checks",
        lambda t: _move_upload_first(t),
        "build-order",
    ),
    (
        "the notes fitting step removed",
        _edit("limit = 120_000", "limit = 10**9"),
        "notes-limit",
    ),
    (
        "the notes fitting step without its repository",
        _edit(
            "          REPO: ${{ github.repository }}\n        run: |\n          set -euo pipefail\n          python - <<'PY'\n          import os\n",
            "        run: |\n          set -euo pipefail\n          python - <<'PY'\n          import os\n",
        ),
        "notes-limit",
    ),
    (
        "a fifth job",
        lambda t: (
            t
            + "\n  extra:\n    runs-on: ubuntu-24.04\n    timeout-minutes: 5\n    steps:\n      - run: echo hi\n"
        ),
        "jobs",
    ),
    (
        "the trigger written as a bare event name",
        _edit('on:\n  push:\n    tags: ["v*"]\n', "on: push\n"),
        "triggers",
    ),
    (
        "the sbom job downloads the wrong artifact",
        _edit(
            "          name: dist\n          path: dist\n      - name: Find the wheel",
            "          name: sbom\n          path: dist\n      - name: Find the wheel",
        ),
        "sbom-download",
    ),
    (
        "the sbom written before the wheel is installed",
        lambda t: _swap_sbom_steps(t),
        "sbom-order",
    ),
    (
        "a uses line hidden inside a script",
        _edit(
            "          " + _EXTRACT_COMMAND + "\n",
            "          " + _EXTRACT_COMMAND + "\n          uses: actions/checkout@v4\n",
        ),
        "uses-count",
    ),
    ("an unreadable file", lambda t: "jobs: [unterminated", "unreadable"),
]


def _fit_before_extract(text: str) -> str:
    """Move the step that cuts the notes in front of the step that extracts them."""
    extract = re.search(
        r"^      - name: Extract the release notes from the changelog\n(?:        .*\n|\n)*?(?=      - name:)",
        text,
        re.MULTILINE,
    )
    fit = re.search(
        r"^      - name: Fit the release notes to the GitHub limit\n(?:        .*\n|\n)*?(?=      - name:)",
        text,
        re.MULTILINE,
    )
    assert extract and fit and extract.end() == fit.start()
    return text[: extract.start()] + fit.group(0) + extract.group(0) + text[fit.end() :]


def _nth_edit(text: str, old: str, new: str, nth: int) -> str:
    """Replace the `nth` (0-based) occurrence of `old`."""
    parts = text.split(old)
    assert len(parts) > nth + 1, f"mutant anchor occurrence {nth} not found: {old!r}"
    return old.join(parts[: nth + 1]) + new + old.join(parts[nth + 1 :])


def _swap_sbom_steps(text: str) -> str:
    """Move the SBOM writing step in front of the wheel install."""
    start = text.index("      - name: Write the SBOM\n")
    end = text.index("      - name: Upload the SBOM\n")
    block = text[start:end]
    without = text[:start] + text[end:]
    anchor = "      - name: Install the wheel with every extra in a fresh environment\n"
    assert anchor in without
    return without.replace(anchor, block + anchor, 1)


def _move_upload_first(text: str) -> str:
    start = text.index("      - name: Upload the distributions\n")
    end = text.index("      - name: Upload the release notes\n")
    block = text[start:end]
    without = text[:start] + text[end:]
    anchor = "      - name: Install the build tools\n"
    assert anchor in without
    return without.replace(anchor, block + anchor, 1)


class TestMutantsAreKilled:
    @pytest.mark.parametrize("label,mutate,code", MUTANTS, ids=[m[0] for m in MUTANTS])
    def test_each_mutant_is_caught_by_its_rule(
        self, label: str, mutate: Any, code: str
    ):
        _assert_mutant_caught(_real_text(), mutate, code, label, workflow_problems)

    def test_there_are_at_least_fourteen_distinct_rules_with_a_mutant(self):
        assert len({m[2] for m in MUTANTS}) >= 14

    def test_every_rule_the_checker_can_return_has_a_mutant(self):
        source = Path(__file__).read_text(encoding="utf-8")
        body = source.split("def workflow_problems", 1)[1].split("def _as_set", 1)[0]
        codes = set(re.findall(r'(?:found\.add|return \{)\(?"([a-z-]+)"', body))
        killed = {m[2] for m in MUTANTS}
        assert codes - killed == set(), sorted(codes - killed)
