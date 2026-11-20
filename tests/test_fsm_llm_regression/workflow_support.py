"""Helpers shared by the workflow tests: read a workflow, find its steps, mutate its text.

Imported by `test_ci_workflow.py` and `test_supply_chain.py` (and `passthrough_env` by the
testing suite and `test_packaging.py`), which used to carry a copy each (final REFLECT hygiene
items 7 and 8, step 121.2; the other workflow tests were dropped, plan ace7a8c8 step 4). Not collected (no `test_` prefix). Every name keeps
the underscore of the copies it replaced, so the call sites did not change.

Interface contract. Everything under "Reading", "Mutating" and "Asserting" is pure: it reads no
file, no environment variable and no clock, and mutates nothing it is given. The functions under
"Running a script" are the only ones that touch the machine (they write one file, read `PATH`
and `SYSTEMROOT`, and run `bash` once per distinct `PATH`).

Reading (`job` is the parsed mapping of one job, `text` the workflow text):

- `_load(text) -> dict`: `yaml.safe_load` with the key `on` (read by PyYAML as `True`) put
  back as `"on"`; fails the test with an assertion when the text is not a mapping, and lets a
  `yaml.YAMLError` through for unreadable YAML.
- `_steps(job) -> list[dict]`: the steps that are mappings, `[]` when there are none.
- `_code_lines(text) -> list[str]`: the lines that are not full-line comments.
- `_step_running(job, needle) -> dict | None`: the first step whose `run` contains `needle`.
- `_uses_steps(job, action) -> list[dict]`: the steps whose `uses` starts with `action@`.
- `_index_of(job, needle) -> int`: the index of the first step whose `run` contains `needle`,
  `-1` when none does.

Mutating (each returns a function `str -> str` that edits the real text; it fails the test with
an assertion when its anchor is not in the text, so a mutant that no longer applies cannot
pass unseen): `_edit(old, new, count=1)` (at least `count` occurrences, replaces the first
`count`), `_edit_nth(old, new, nth)` (only the 0-based `nth` occurrence), `_drop_step(name,
nth=0)` (the `nth` step whose `- name:` line is exactly `name`, in a job at the usual
indentation), `_drop_job(job_id)` (a job, to the next job or the end).

Running a script (the shell-script tests run the `run:` text of a workflow step in bash; the
workflows themselves only ever run on `ubuntu-24.04`, so a machine without a working bash
skips these tests with the reason, never fails them, and Windows skips them
unconditionally: plan d7719b03 D-006 and D-009; the bare name `bash` that the tests run
resolves, through `CreateProcess`, to WSL's launcher in System32 whatever `which` finds first):

- `write_script(path, text) -> None`: writes UTF-8 with `newline="\\n"`, so a Windows checkout
  never hands bash a `\\r` before each newline.
- `passthrough_env() -> dict[str, str]`: `{"SYSTEMROOT": value}` when the variable is set (a
  hand-built child environment on Windows needs it), else `{}`.
- `bash_problem(path=None) -> str | None`: why the scripts cannot run here, or `None`. On
  `sys.platform == "win32"` (read at call time) always a problem, naming D-009, whatever bash
  is installed. Elsewhere: why no usable bash is on `path` (default: the `PATH` of this
  process). It runs `bash -c 'echo ok'` and wants exactly `ok`; no `bash`, a non-zero exit,
  other output or a timeout is a problem. Cached per `PATH`. Never raises.
- `skip_unless_bash(path=None)`: calls the probe and `pytest.skip(reason)` when it reports a
  problem. Call it in a test body or in the helper
  that runs a script, never at import time.

Asserting (it raises `AssertionError`, so a test calls it as it would `assert`):
`_assert_mutant_caught(real, mutate, code,
label, problems)` (the mutant changes the text and `problems(mutated)`, the module's own rule
checker returning a set of codes, contains `code`).
"""

from __future__ import annotations

import functools
import os
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from shutil import which
from typing import Any

import pytest
import yaml

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


def _code_lines(text: str) -> list[str]:
    """The lines of the workflow that are not full-line comments."""
    return [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]


def _step_running(job: dict[str, Any], needle: str) -> dict[str, Any] | None:
    for step in _steps(job):
        if needle in str(step.get("run", "")):
            return step
    return None


def _uses_steps(job: dict[str, Any], action: str) -> list[dict[str, Any]]:
    return [s for s in _steps(job) if str(s.get("uses", "")).startswith(action + "@")]


def _index_of(job: dict[str, Any], needle: str) -> int:
    for i, step in enumerate(_steps(job)):
        if needle in str(step.get("run", "")):
            return i
    return -1


# --------------------------------------------------------------------------------------
# Mutating the workflow text
# --------------------------------------------------------------------------------------


def _edit(old: str, new: str, count: int = 1) -> Callable[[str], str]:
    def mutate(text: str) -> str:
        assert text.count(old) >= count, f"{old!r} not found {count} time(s)"
        return text.replace(old, new, count)

    return mutate


def _edit_nth(old: str, new: str, nth: int) -> Callable[[str], str]:
    """Replace only the `nth` (0-based) occurrence of `old`."""

    def mutate(text: str) -> str:
        parts = text.split(old)
        assert len(parts) > nth + 1, f"{old!r} has no occurrence number {nth}"
        return old.join(parts[: nth + 1]) + new + old.join(parts[nth + 1 :])

    return mutate


def _drop_step(name: str, nth: int = 0) -> Callable[[str], str]:
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


def _drop_job(job_id: str) -> Callable[[str], str]:
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


# --------------------------------------------------------------------------------------
# Running a script: LF-only writer and capability probes
# --------------------------------------------------------------------------------------

_PROBE_TIMEOUT_SECONDS = 30


def write_script(path: Path, text: str) -> None:
    """Write a shell script as UTF-8 with LF line ends on every platform."""
    path.write_text(text, encoding="utf-8", newline="\n")


def passthrough_env() -> dict[str, str]:
    """The variables a hand-built child environment keeps (`SYSTEMROOT`, when set)."""
    value = os.environ.get("SYSTEMROOT")
    return {"SYSTEMROOT": value} if value else {}


def _search_path(path: str | None) -> str:
    return os.environ.get("PATH", "") if path is None else path


def _windows_problem() -> str | None:
    """Why no script runs on Windows (D-009), or ``None`` elsewhere. Reads the platform now."""
    if sys.platform == "win32":
        return (
            "the workflow scripts run on ubuntu-24.04 only (plan d7719b03 D-009): "
            "on Windows the bare name `bash` reaches the WSL launcher in System32, "
            "not the bash `which` finds"
        )
    return None


def bash_problem(path: str | None = None) -> str | None:
    """Why no usable bash is on *path*, or ``None`` (see the module docstring)."""
    if problem := _windows_problem():
        return problem
    return _bash_problem(_search_path(path), os.environ.get("SYSTEMROOT", ""))


@functools.cache
def _bash_problem(path: str, systemroot: str) -> str | None:
    found = which("bash", path=path)
    if found is None:
        return "no bash on PATH"
    env = {"PATH": path, **({"SYSTEMROOT": systemroot} if systemroot else {})}
    try:
        proc = subprocess.run(
            [found, "-c", "echo ok"],
            env=env,
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"bash cannot run: {exc}"
    if proc.returncode != 0 or proc.stdout.strip() != "ok":
        return f"bash -c 'echo ok' gave exit {proc.returncode} and {proc.stdout!r}"
    return None


def skip_unless_bash(path: str | None = None) -> None:
    """Skip the calling test, with the reason, when `bash_problem` reports one."""
    problem = bash_problem(path)
    if problem:
        pytest.skip(f"these tests run a workflow script in bash: {problem}")


# --------------------------------------------------------------------------------------
# Assertions the real-workflow and mutant tests share
# --------------------------------------------------------------------------------------


def _assert_mutant_caught(
    real: str,
    mutate: Callable[[str], str],
    code: str,
    label: str,
    problems: Callable[[str], set[str]],
) -> None:
    mutated = mutate(real)
    assert mutated != real
    assert code in problems(mutated), (label, problems(mutated))
