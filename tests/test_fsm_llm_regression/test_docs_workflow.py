"""The docs workflow builds strictly on every PR and tag, and deploys only a checked tag.

Plan d454fd62, step 106 (roadmap row 0.7). `.github/workflows/docs.yml` is the only thing that
publishes the documentation site, so a slip in it is either a broken site that nobody sees (the
strict build removed, the deploy aimed at the wrong ref) or a supply-chain hole (a floating
action tag, a write token on a pull request, an event field pasted into a shell). None of it
runs on a developer machine, so it is read here as data.

* `workflow_problems(text)` is a pure function of the workflow text and returns a set of problem
  codes. The real workflow must return the empty set (`TestTheRealWorkflow`); each of the
  mutants of `MUTANTS` is the real text with one edit and must return its own code
  (`TestMutantsAreKilled`): a check that cannot fail is not a check.
* The guard step of the deploy job (tag equals `src/fsm_llm/__version__.py`, final or
  pre-release) is a script inside the workflow; `TestTheTagGuard` extracts it from the YAML and
  runs it on synthetic refs, so the shipped text is what is tested and nothing is copied.
* Only the standard library and `yaml.safe_load` are used. PyYAML reads the key `on` as the
  boolean `True` (YAML 1.1); `_load` puts it back under `"on"`.
* The action pins are 40 hex characters with the release tag in a trailing `# vX.Y.Z` comment.
  Step 108 (the CI matrix), 109 and 110 pin their workflows to the same table
  (progress.md "Step 106"); the cross-file check arrives with step 109.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
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
_WORKFLOW = _ROOT / ".github" / "workflows" / "docs.yml"

_SHA_USES = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+@[0-9a-f]{40}$")
_USES_LINE = re.compile(r"^\s*(?:-\s*)?uses:\s*(?P<ref>\S+)(?P<rest>.*)$", re.MULTILINE)
_TAG_COMMENT = re.compile(r"^\s*#\s*v\d+\.\d+\.\d+\s*$")
_MIKE_DEPLOY = 'mike deploy --push --update-aliases "$VERSION"'


# --------------------------------------------------------------------------------------
# Reading the workflow
# --------------------------------------------------------------------------------------


def workflow_problems(text: str) -> set[str]:
    """Return the codes of every rule the workflow text breaks (empty set: it is sound)."""
    found: set[str] = set()
    try:
        wf = _load(text)
    except Exception:
        return {"unreadable"}
    jobs = wf.get("jobs") or {}
    build = jobs.get("build") or {}
    deploy = jobs.get("deploy") or {}
    code = "\n".join(_code_lines(text))

    # Triggers: tags v*, pull requests to main, manual dispatch; nothing privileged.
    on = wf.get("on") or {}
    if not isinstance(on, dict):
        found.add("triggers")
    else:
        if (on.get("push") or {}).get("tags") != ["v*"] or "branches" in (
            on.get("push") or {}
        ):
            found.add("push-trigger")
        if (on.get("pull_request") or {}).get("branches") != ["main"]:
            found.add("pull-request-trigger")
        if "workflow_dispatch" not in on:
            found.add("dispatch-trigger")
        if set(on) != {"push", "pull_request", "workflow_dispatch"}:
            found.add("extra-trigger")
    if "pull_request_target" in code:
        found.add("pull-request-target")

    # Permissions: read at the top, write on the deploy job only.
    if wf.get("permissions") != {"contents": "read"}:
        found.add("top-permissions")
    if set(jobs) != {"build", "deploy"}:
        found.add("jobs")
    if "permissions" in build:
        found.add("build-permissions")
    if deploy.get("permissions") != {"contents": "write"}:
        found.add("deploy-permissions")
    if re.search(r"\bid-token\b|\bpages:\s*write|\bpackages:\s*write", code):
        found.add("extra-permission")

    # Runners, timeouts, shell.
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

    # Actions: a 40-hex SHA and a `# vX.Y.Z` comment on every `uses:`.
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
    }:
        found.add("action-set")

    # Secrets: the default token only, never named in the file.
    if re.search(r"\bsecrets\.", code) or "GITHUB_TOKEN" in code:
        found.add("secrets")

    # Build job.
    checkouts = _uses_steps(build, "actions/checkout")
    if (
        len(checkouts) != 1
        or (checkouts[0].get("with") or {}).get("persist-credentials") is not False
    ):
        found.add("build-persist-credentials")
    if not _installs_docs_group(build):
        found.add("build-install")
    strict = _step_running(build, "mkdocs build --strict")
    if (
        strict is None
        or str((strict.get("env") or {}).get("NO_MKDOCS_2_WARNING")) != "1"
        or "mkdocs build --strict" not in str(strict.get("run"))
        or "--strict" not in str(strict.get("run"))
    ):
        found.add("build-strict")
    if not _python_312(build):
        found.add("build-python")

    # Deploy job: tags only, history, no cancel, identity, guard, mike.
    condition = str(deploy.get("if", ""))
    if "refs/tags/v" not in condition or "pull_request" in condition:
        found.add("deploy-condition")
    if deploy.get("needs") != "build":
        found.add("deploy-needs")
    conc = deploy.get("concurrency") or {}
    if (
        conc.get("group") != "docs-deploy"
        or conc.get("cancel-in-progress") is not False
    ):
        found.add("deploy-concurrency")
    dcheck = _uses_steps(deploy, "actions/checkout")
    dwith = (dcheck[0].get("with") or {}) if dcheck else {}
    if len(dcheck) != 1 or dwith.get("fetch-depth") != 0:
        found.add("deploy-fetch-depth")
    if dwith.get("persist-credentials") is False or "mike pushes" not in text:
        found.add("deploy-credentials-reason")
    if not _installs_docs_group(deploy):
        found.add("deploy-install")
    if not _python_312(deploy):
        found.add("deploy-python")
    ident = _step_running(deploy, "git config user.name")
    if (
        ident is None
        or 'git config user.name "github-actions[bot]"' not in ident["run"]
        or "41898282+github-actions[bot]@users.noreply.github.com" not in ident["run"]
    ):
        found.add("deploy-identity")
    guard = next((s for s in _steps(deploy) if s.get("id") == "version"), None)
    if guard is None or not all(
        token in str(guard.get("run"))
        for token in (
            "src/fsm_llm/__version__.py",
            "GITHUB_REF_NAME",
            "GITHUB_OUTPUT",
            "sys.exit",
        )
    ):
        found.add("deploy-guard")
    if _final_deploy_is_wrong(deploy) or _pre_release_deploy_is_wrong(deploy):
        found.add("mike-shape")
    if "set-default" in code:
        found.add("mike-set-default")
    guard_index = next(
        (i for i, s in enumerate(_steps(deploy)) if s.get("id") == "version"), 99
    )
    deploy_indexes = [
        i for i, s in enumerate(_steps(deploy)) if "mike deploy" in str(s.get("run"))
    ]
    if not deploy_indexes or min(deploy_indexes) < guard_index:
        found.add("guard-before-deploy")
    return found


def _installs_docs_group(job: dict[str, Any]) -> bool:
    step = _step_running(job, "--group docs")
    if step is None:
        return False
    run = str(step["run"])
    return "-c constraints.txt" in run and "pip install" in run and " -e " not in run


def _python_312(job: dict[str, Any]) -> bool:
    steps = _uses_steps(job, "actions/setup-python")
    return (
        len(steps) == 1
        and str((steps[0].get("with") or {}).get("python-version")) == "3.12"
    )


def _mike_steps(deploy: dict[str, Any]) -> list[dict[str, Any]]:
    return [s for s in _steps(deploy) if "mike deploy" in str(s.get("run", ""))]


def _final_deploy_is_wrong(deploy: dict[str, Any]) -> bool:
    """True unless exactly one step deploys a final release with the `latest` alias."""
    final = [s for s in _mike_steps(deploy) if s["run"].strip().endswith(" latest")]
    if len(final) != 1:
        return True
    step = final[0]
    return (
        step["run"].strip() != _MIKE_DEPLOY + " latest"
        or step.get("if") != "steps.version.outputs.prerelease == 'false'"
        or (step.get("env") or {}).get("VERSION")
        != "${{ steps.version.outputs.version }}"
        or str((step.get("env") or {}).get("NO_MKDOCS_2_WARNING")) != "1"
    )


def _pre_release_deploy_is_wrong(deploy: dict[str, Any]) -> bool:
    """True unless exactly one step deploys a pre-release and it never names `latest`."""
    pre = [s for s in _mike_steps(deploy) if not s["run"].strip().endswith(" latest")]
    if len(pre) != 1:
        return True
    step = pre[0]
    return (
        step["run"].strip() != _MIKE_DEPLOY
        or "latest" in step["run"]
        or step.get("if") != "steps.version.outputs.prerelease == 'true'"
        or (step.get("env") or {}).get("VERSION")
        != "${{ steps.version.outputs.version }}"
        or str((step.get("env") or {}).get("NO_MKDOCS_2_WARNING")) != "1"
    )


def _real_text() -> str:
    return _WORKFLOW.read_text(encoding="utf-8")


# --------------------------------------------------------------------------------------
# The real workflow
# --------------------------------------------------------------------------------------


class TestTheRealWorkflow:
    def test_the_workflow_exists_and_reads_as_yaml(self):
        assert _WORKFLOW.is_file(), ".github/workflows/docs.yml is missing"
        assert isinstance(_load(_real_text()), dict)

    def test_it_breaks_no_rule(self):
        assert workflow_problems(_real_text()) == set()

    def test_triggers_are_tags_pull_requests_to_main_and_dispatch(self):
        on = _load(_real_text())["on"]
        assert on["push"] == {"tags": ["v*"]}
        assert on["pull_request"] == {"branches": ["main"]}
        assert "workflow_dispatch" in on
        assert set(on) == {"push", "pull_request", "workflow_dispatch"}

    def test_the_pull_request_trigger_is_not_filtered_by_path(self):
        # A docs build on every PR: a page can break from a code edit (an include, a docstring).
        assert "paths" not in _load(_real_text())["on"]["pull_request"]

    def test_read_only_at_the_top_and_write_on_deploy_only(self):
        wf = _load(_real_text())
        assert wf["permissions"] == {"contents": "read"}
        assert "permissions" not in wf["jobs"]["build"]
        assert wf["jobs"]["deploy"]["permissions"] == {"contents": "write"}

    def test_build_runs_the_strict_build_with_the_mkdocs_2_warning_silenced(self):
        build = _load(_real_text())["jobs"]["build"]
        step = _step_running(build, "mkdocs build --strict")
        assert step is not None
        assert step["env"] == {"NO_MKDOCS_2_WARNING": "1"}
        assert step["run"].strip() == "mkdocs build --strict"

    def test_build_installs_the_docs_group_against_the_pins_only(self):
        build = _load(_real_text())["jobs"]["build"]
        step = _step_running(build, "--group docs")
        assert step is not None
        assert "python -m pip install -c constraints.txt --group docs" in step["run"]
        # `--group` needs pip 25.1; the runner's bundled pip may be older.
        assert re.search(r'pip install "pip>=25\.1"', step["run"])

    def test_the_package_itself_is_not_installed(self):
        # No page reads the source (no mkdocstrings block since step 123), so the build needs no install.
        code = "\n".join(_code_lines(_real_text()))
        assert not re.search(r"pip install[^\n]*\s-e\s", code)
        assert "pip install ." not in code

    def test_the_build_checkout_drops_credentials_and_the_deploy_checkout_keeps_them(
        self,
    ):
        wf = _load(_real_text())
        build_checkout = _uses_steps(wf["jobs"]["build"], "actions/checkout")[0]
        deploy_checkout = _uses_steps(wf["jobs"]["deploy"], "actions/checkout")[0]
        assert build_checkout["with"]["persist-credentials"] is False
        assert deploy_checkout["with"]["persist-credentials"] is True
        assert deploy_checkout["with"]["fetch-depth"] == 0
        assert "mike pushes" in _real_text()

    def test_deploy_runs_on_a_version_tag_push_after_the_build_and_never_cancels(self):
        deploy = _load(_real_text())["jobs"]["deploy"]
        assert deploy["needs"] == "build"
        assert deploy["if"] == (
            "github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')"
        )
        assert deploy["concurrency"] == {
            "group": "docs-deploy",
            "cancel-in-progress": False,
        }

    def test_the_bot_identity_is_the_standard_one(self):
        deploy = _load(_real_text())["jobs"]["deploy"]
        run = _step_running(deploy, "git config user.name")["run"]
        assert 'git config user.name "github-actions[bot]"' in run
        assert (
            'git config user.email "41898282+github-actions[bot]@users.noreply.github.com"'
            in run
        )

    def test_the_mike_commands_have_the_documented_shape(self):
        deploy = _load(_real_text())["jobs"]["deploy"]
        shapes = {
            s["if"]: s["run"].strip()
            for s in _steps(deploy)
            if "mike deploy" in str(s.get("run", ""))
        }
        assert shapes == {
            "steps.version.outputs.prerelease == 'true'": _MIKE_DEPLOY,
            "steps.version.outputs.prerelease == 'false'": _MIKE_DEPLOY + " latest",
        }

    def test_ci_never_sets_the_default_version(self):
        # `mike set-default latest` is an owner hand-off after the first deploy (docs/README.md).
        assert "set-default" not in "\n".join(_code_lines(_real_text()))

    def test_every_action_is_a_sha_pin_with_its_release_tag(self):
        text = _real_text()
        matches = list(_USES_LINE.finditer(text))
        assert len(matches) == 4  # checkout and setup-python, in two jobs
        for match in matches:
            assert _SHA_USES.match(match["ref"]), match["ref"]
            assert _TAG_COMMENT.match(match["rest"]), match["rest"]

    def test_the_same_action_has_the_same_pin_in_both_jobs(self):
        pins: dict[str, set[str]] = {}
        for match in _USES_LINE.finditer(_real_text()):
            action, _, sha = match["ref"].partition("@")
            pins.setdefault(action, set()).add(sha + match["rest"].strip())
        assert {a: len(v) for a, v in pins.items()} == {
            "actions/checkout": 1,
            "actions/setup-python": 1,
        }

    def test_no_event_field_is_interpolated_into_a_script(self):
        wf = _load(_real_text())
        for job in wf["jobs"].values():
            for step in _steps(job):
                assert "${{" not in str(step.get("run", "")), step.get("name")

    def test_no_secret_and_no_privileged_trigger(self):
        code = "\n".join(_code_lines(_real_text()))
        assert "secrets." not in code
        assert "pull_request_target" not in code
        assert "-latest" not in code

    def test_every_job_has_a_timeout_and_a_pinned_runner(self):
        _assert_runners_pinned_with_timeouts(_load(_real_text()))

    def test_every_multi_line_script_fails_fast(self):
        _assert_multi_line_scripts_fail_fast(_load(_real_text()))


# --------------------------------------------------------------------------------------
# The guard script, run on synthetic refs
# --------------------------------------------------------------------------------------


def _guard_script() -> str:
    deploy = _load(_real_text())["jobs"]["deploy"]
    step = next(s for s in _steps(deploy) if s.get("id") == "version")
    return str(step["run"])


def _run_guard(tmp_path: Path, tag: str, declared: str | None) -> tuple[int, str, str]:
    """Run the workflow's guard script in a scratch tree; return (exit code, output, stderr)."""
    pkg = tmp_path / "src" / "fsm_llm"
    pkg.mkdir(parents=True, exist_ok=True)
    version_file = pkg / "__version__.py"
    if declared is None:
        version_file.write_text("# no version here\n", encoding="utf-8")
    else:
        version_file.write_text(f'__version__ = "{declared}"\n', encoding="utf-8")
    out = tmp_path / "github_output"
    out.write_text("", encoding="utf-8")
    script = tmp_path / "guard.sh"
    script.write_text(_guard_script(), encoding="utf-8")
    env = {
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
        "GITHUB_REF_NAME": tag,
        "GITHUB_OUTPUT": str(out),
        "HOME": str(tmp_path),
    }
    proc = subprocess.run(
        ["bash", str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return proc.returncode, out.read_text(encoding="utf-8"), proc.stderr


class TestTheTagGuard:
    @pytest.mark.parametrize(
        "tag,declared,prerelease",
        [
            ("v0.13.0", "0.13.0", "false"),
            ("v1.0", "1.0", "false"),
            ("v0.13.0.post1", "0.13.0.post1", "false"),
            ("v0.13.1rc1", "0.13.1rc1", "true"),
            ("v0.13.1a1", "0.13.1a1", "true"),
            ("v0.13.1b2", "0.13.1b2", "true"),
            ("v0.13.1.dev3", "0.13.1.dev3", "true"),
            ("v0.13.1-rc.1", "0.13.1-rc.1", "true"),
            ("v0.13.0.post1.dev2", "0.13.0.post1.dev2", "true"),
        ],
    )
    def test_a_tag_that_equals_the_version_passes_and_is_classified(
        self, tmp_path: Path, tag: str, declared: str, prerelease: str
    ):
        code, output, err = _run_guard(tmp_path, tag, declared)
        assert code == 0, err
        assert output.splitlines() == [
            f"version={declared}",
            f"prerelease={prerelease}",
        ]

    @pytest.mark.parametrize(
        "tag,declared",
        [
            ("v0.13.1", "0.13.0"),
            ("v0.13.0", "0.13.1"),
            ("v0.13.0rc1", "0.13.0"),
            ("v0.13.0", "0.13.0rc1"),
            ("0.13.0", "0.13.0"),
            ("V0.13.0", "0.13.0"),
        ],
    )
    def test_a_tag_that_differs_from_the_version_fails_with_a_message(
        self, tmp_path: Path, tag: str, declared: str
    ):
        code, output, err = _run_guard(tmp_path, tag, declared)
        assert code != 0
        assert output == "", "a failed guard must write no output"
        assert "does not match" in err and "src/fsm_llm/__version__.py" in err

    @pytest.mark.parametrize("declared", ["garbage", "0.13.0+local", "1!0.13.0"])
    def test_a_declared_version_that_is_not_a_plain_pep_440_release_fails(
        self, tmp_path: Path, declared: str
    ):
        code, output, err = _run_guard(tmp_path, "v" + declared, declared)
        assert code != 0
        assert output == ""
        assert "PEP 440" in err

    def test_a_missing_version_assignment_fails(self, tmp_path: Path):
        code, output, err = _run_guard(tmp_path, "v0.13.0", None)
        assert code != 0
        assert output == ""
        assert "__version__" in err

    def test_the_real_tree_tag_of_its_own_version_passes(self, tmp_path: Path):
        # The guard reads the real file shape: `__version__ = "<x>"` (checked on the working tree).
        text = (_ROOT / "src" / "fsm_llm" / "__version__.py").read_text(
            encoding="utf-8"
        )
        declared = re.search(r'__version__\s*=\s*"([^"]+)"', text)
        assert declared is not None
        code, output, err = _run_guard(tmp_path, "v" + declared[1], declared[1])
        assert code == 0, err
        assert output.startswith("version=" + declared[1])


# --------------------------------------------------------------------------------------
# Mutants: each edit of the real text must be caught by its own rule
# --------------------------------------------------------------------------------------


def _short_sha(text: str) -> str:
    mutated = re.sub(
        r"(actions/setup-python@[0-9a-f]{39})[0-9a-f]", r"\1", text, count=1
    )
    assert mutated != text
    return mutated


MUTANTS: list[tuple[str, Any, str]] = [
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
        "write on the build job",
        _edit(
            "  build:\n    runs-on: ubuntu-24.04\n",
            "  build:\n    permissions:\n      contents: write\n    runs-on: ubuntu-24.04\n",
        ),
        "build-permissions",
    ),
    (
        "deploy write permission dropped",
        _edit(
            "    permissions:\n      contents: write\n",
            "    permissions:\n      contents: read\n",
        ),
        "deploy-permissions",
    ),
    (
        "id-token granted on deploy",
        _edit(
            "      contents: write\n", "      contents: write\n      id-token: write\n"
        ),
        "deploy-permissions",
    ),
    ("an action pinned to a tag", _unpin_checkout, "action-pin"),
    ("an action pin truncated", _short_sha, "action-pin"),
    ("an action pin without its tag comment", _drop_comment, "action-tag-comment"),
    (
        "ubuntu-latest on the build job",
        _edit("runs-on: ubuntu-24.04", "runs-on: ubuntu-latest"),
        "runner",
    ),
    (
        "guard reads the wrong file",
        _edit("src/fsm_llm/__version__.py", "src/fsm_llm/__init__.py"),
        "deploy-guard",
    ),
    ("guard does not fail", _edit("sys.exit", "print", 3), "deploy-guard"),
    (
        "latest alias on the pre-release",
        _edit(_MIKE_DEPLOY + "\n", _MIKE_DEPLOY + " latest\n"),
        "mike-shape",
    ),
    (
        "final release without latest",
        _edit(_MIKE_DEPLOY + " latest", _MIKE_DEPLOY),
        "mike-shape",
    ),
    ("alias update removed", _edit(" --update-aliases", "", 2), "mike-shape"),
    (
        "deploy also runs on pull requests",
        _edit(
            "if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')",
            "if: github.event_name == 'pull_request' || startsWith(github.ref, 'refs/tags/v')",
        ),
        "deploy-condition",
    ),
    (
        "deploy on any ref",
        _edit(
            "if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')",
            "if: always()",
        ),
        "deploy-condition",
    ),
    (
        "fetch-depth removed",
        _edit("          fetch-depth: 0\n", ""),
        "deploy-fetch-depth",
    ),
    (
        "strict removed from the build",
        _edit("run: mkdocs build --strict", "run: mkdocs build"),
        "build-strict",
    ),
    (
        "mkdocs 2 warning not silenced",
        _edit(
            '          NO_MKDOCS_2_WARNING: "1"\n        run: mkdocs build --strict',
            "        run: mkdocs build --strict",
        ),
        "build-strict",
    ),
    (
        "concurrency group renamed",
        _edit("group: docs-deploy", "group: docs"),
        "deploy-concurrency",
    ),
    (
        "concurrency cancels",
        _edit("cancel-in-progress: false", "cancel-in-progress: true"),
        "deploy-concurrency",
    ),
    (
        "pull_request_target",
        _edit("  pull_request:\n", "  pull_request_target:\n"),
        "pull-request-target",
    ),
    (
        "ref name interpolated into a script",
        _edit(
            "run: mkdocs build --strict",
            'run: mkdocs build --strict --site-dir "${{ github.ref_name }}"',
        ),
        "run-interpolation",
    ),
    (
        "head ref interpolated into a multi-line script",
        _edit(
            '          python -m pip install "pip>=25.1"\n',
            '          echo "${{ github.head_ref }}"\n          python -m pip install "pip>=25.1"\n',
        ),
        "run-interpolation",
    ),
    (
        "timeout removed from the build",
        _edit("    timeout-minutes: 15\n", "", 1),
        "timeout",
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
        "credentials dropped on the deploy checkout",
        _edit(
            "          persist-credentials: true\n",
            "          persist-credentials: false\n",
        ),
        "deploy-credentials-reason",
    ),
    (
        "set-default run by CI",
        _edit(
            _MIKE_DEPLOY + " latest\n",
            _MIKE_DEPLOY + " latest\n          mike set-default latest\n",
        ),
        "mike-set-default",
    ),
    (
        "constraints dropped from the install",
        _edit("-c constraints.txt ", "", 1),
        "build-install",
    ),
    (
        "the package installed editable",
        _edit("--group docs", "--group docs -e .", 1),
        "build-install",
    ),
    (
        "a secret named",
        _edit(
            '          NO_MKDOCS_2_WARNING: "1"\n        run: mkdocs build --strict',
            '          NO_MKDOCS_2_WARNING: "1"\n          TOKEN: ${{ secrets.PAGES_TOKEN }}\n        run: mkdocs build --strict',
        ),
        "secrets",
    ),
    (
        "pipefail dropped from the install step",
        _edit(
            "          set -euo pipefail\n          # `pip install --group`",
            "          # `pip install --group`",
        ),
        "shell-strict",
    ),
    (
        "python version moved",
        _edit('python-version: "3.12"', 'python-version: "3.11"', 1),
        "build-python",
    ),
    (
        "bot identity changed",
        _edit('github-actions[bot]"\n', 'someone"\n'),
        "deploy-identity",
    ),
    ("deploy runs without the build", _edit("    needs: build\n", ""), "deploy-needs"),
    (
        "dispatch trigger removed",
        _edit("  workflow_dispatch:\n", ""),
        "dispatch-trigger",
    ),
    (
        "tag trigger widened to a branch",
        _edit('    tags: ["v*"]\n', '    tags: ["v*"]\n    branches: ["main"]\n'),
        "push-trigger",
    ),
    ("an unreadable file", lambda t: "jobs: [unterminated", "unreadable"),
]


class TestMutantsAreKilled:
    @pytest.mark.parametrize("label,mutate,code", MUTANTS, ids=[m[0] for m in MUTANTS])
    def test_each_mutant_is_caught_by_its_rule(
        self, label: str, mutate: Any, code: str
    ):
        _assert_mutant_caught(_real_text(), mutate, code, label, workflow_problems)

    def test_there_are_at_least_ten_distinct_rules_with_a_mutant(self):
        assert len({m[2] for m in MUTANTS}) >= 10
