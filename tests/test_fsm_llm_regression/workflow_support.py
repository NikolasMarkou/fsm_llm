"""Helpers shared by the workflow tests: read a workflow, find its steps, mutate its text.

Imported by `test_docs_workflow.py`, `test_ci_workflow.py`, `test_refresh_workflow.py`,
`test_release_workflow.py` and `test_supply_chain.py`, which used to carry a copy each (final
REFLECT hygiene items 7 and 8, step 121.2). Not collected (no `test_` prefix). Every name keeps
the underscore of the copies it replaced, so the call sites did not change.

Interface contract. Everything here is pure: it reads no file, no environment variable and no
clock, and mutates nothing it is given.

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
indentation), `_drop_job(job_id)` (a job, to the next job or the end). `_unpin_checkout` and
`_drop_comment` are plain `str -> str` mutants: the first checkout pin becomes a tag, the
version comment after the first `actions/setup-python` pin goes.

Asserting (they raise `AssertionError`, so a test calls them as it would `assert`):
`_assert_runners_pinned_with_timeouts(workflow)` (every job on `ubuntu-24.04` with an integer
`timeout-minutes`), `_assert_multi_line_scripts_fail_fast(workflow)` (bash by default and
`set -euo pipefail` in every multi-line `run`) and `_assert_mutant_caught(real, mutate, code,
label, problems)` (the mutant changes the text and `problems(mutated)`, the module's own rule
checker returning a set of codes, contains `code`).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import yaml

_SHA_CHECKOUT = re.compile(r"actions/checkout@[0-9a-f]{40}")


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


def _unpin_checkout(text: str) -> str:
    assert _SHA_CHECKOUT.search(text)
    return _SHA_CHECKOUT.sub("actions/checkout@v7", text, count=1)


def _drop_comment(text: str) -> str:
    mutated = re.sub(
        r"(actions/setup-python@[0-9a-f]{40})\s+# v[\d.]+", r"\1", text, count=1
    )
    assert mutated != text
    return mutated


# --------------------------------------------------------------------------------------
# Assertions the real-workflow and mutant tests share
# --------------------------------------------------------------------------------------


def _assert_runners_pinned_with_timeouts(workflow: dict[str, Any]) -> None:
    for name, job in workflow["jobs"].items():
        assert job["runs-on"] == "ubuntu-24.04", name
        assert isinstance(job["timeout-minutes"], int), name


def _assert_multi_line_scripts_fail_fast(workflow: dict[str, Any]) -> None:
    assert workflow["defaults"]["run"]["shell"] == "bash"
    for job in workflow["jobs"].values():
        for step in _steps(job):
            run = str(step.get("run", ""))
            if "\n" in run.strip():
                assert "set -euo pipefail" in run, step.get("name")


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
