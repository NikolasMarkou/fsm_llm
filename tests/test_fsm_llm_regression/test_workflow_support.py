"""`workflow_support.py`: the helpers of the workflow tests fail when they should.

The five workflow tests (`test_docs_workflow.py`, `test_ci_workflow.py`,
`test_refresh_workflow.py`, `test_release_workflow.py`, `test_supply_chain.py`) judge their
mutants and real workflows through these helpers. A helper that asserted nothing would turn
every one of those tests green without checking a thing, so each one is shown to reject a bad
input here, and to give the old answer on a good one (final REFLECT hygiene items 7 and 8).
"""

from __future__ import annotations

import pytest

from tests.test_fsm_llm_regression.workflow_support import (
    _assert_multi_line_scripts_fail_fast,
    _assert_mutant_caught,
    _assert_runners_pinned_with_timeouts,
    _code_lines,
    _drop_comment,
    _drop_job,
    _drop_step,
    _edit,
    _edit_nth,
    _index_of,
    _load,
    _step_running,
    _steps,
    _unpin_checkout,
    _uses_steps,
)

SHA = "a" * 40
TEXT = (
    "# a comment\n"
    "on:\n  push: {}\n"
    "defaults:\n  run:\n    shell: bash\n"
    "jobs:\n"
    "  one:\n"
    "    runs-on: ubuntu-24.04\n"
    "    timeout-minutes: 5\n"
    "    steps:\n"
    f"      - uses: actions/checkout@{SHA}\n"
    f"      - uses: actions/setup-python@{SHA}  # v6.1.0\n"
    "      - name: Build\n"
    "        run: make build\n"
    "      - name: Test\n"
    "        run: |\n"
    "          set -euo pipefail\n"
    "          make test\n"
    "  two:\n"
    "    runs-on: ubuntu-24.04\n"
    "    timeout-minutes: 5\n"
    "    steps:\n"
    "      - name: Build\n"
    "        run: make build\n"
)


class TestReading:
    def test_the_on_key_is_put_back(self) -> None:
        data = _load(TEXT)
        assert "on" in data and True not in data

    def test_a_text_that_is_not_a_mapping_fails(self) -> None:
        with pytest.raises(AssertionError, match="not a mapping"):
            _load("- a\n- b\n")

    def test_steps_keep_the_mappings_only(self) -> None:
        job = {"steps": [{"run": "x"}, "text", None, {"uses": "y"}]}
        assert _steps(job) == [{"run": "x"}, {"uses": "y"}]
        assert _steps({}) == []

    def test_code_lines_drop_the_full_line_comments_only(self) -> None:
        text = "# a\n  # b\nx: 1  # c\n\ny: 2\n"
        assert _code_lines(text) == ["x: 1  # c", "", "y: 2"]

    def test_step_finders(self) -> None:
        job = _load(TEXT)["jobs"]["one"]
        assert _step_running(job, "make test") is not None
        assert _step_running(job, "absent") is None
        assert _index_of(job, "make build") == 2
        assert _index_of(job, "absent") == -1
        assert len(_uses_steps(job, "actions/checkout")) == 1
        assert _uses_steps(job, "actions/check") == []  # the `@` is part of the match


class TestMutators:
    def test_edit_replaces_and_refuses_a_missing_anchor(self) -> None:
        assert _edit("a", "b")("aaa") == "baa"
        assert _edit("a", "b", 2)("aaa") == "bba"
        with pytest.raises(AssertionError, match="not found"):
            _edit("z", "b")("aaa")
        with pytest.raises(AssertionError, match="not found 2"):
            _edit("a", "b", 2)("xay")

    def test_edit_nth_replaces_one_occurrence(self) -> None:
        assert _edit_nth("a", "b", 1)("aaa") == "aba"
        with pytest.raises(AssertionError, match="occurrence number 3"):
            _edit_nth("a", "b", 3)("aaa")

    def test_drop_step_and_drop_job_remove_exactly_their_text(self) -> None:
        dropped = _drop_step("Build")(TEXT)
        assert dropped.count("- name: Build") == 1 and "make test" in dropped
        assert _drop_step("Build", 1)(TEXT).count("- name: Build") == 1
        assert "two:" not in _drop_job("two")(TEXT)
        assert "one:" in _drop_job("two")(TEXT)
        with pytest.raises(AssertionError, match="not found"):
            _drop_step("Nope")(TEXT)
        with pytest.raises(AssertionError, match="not found"):
            _drop_job("three")(TEXT)

    def test_unpin_and_drop_comment(self) -> None:
        assert "actions/checkout@v7" in _unpin_checkout(TEXT)
        assert "# v6.1.0" not in _drop_comment(TEXT)
        with pytest.raises(AssertionError):
            _unpin_checkout("nothing")
        with pytest.raises(AssertionError):
            _drop_comment("nothing")


class TestAssertions:
    def test_runners_and_timeouts(self) -> None:
        _assert_runners_pinned_with_timeouts(_load(TEXT))
        floating = _load(TEXT.replace("ubuntu-24.04", "ubuntu-latest", 1))
        with pytest.raises(AssertionError):
            _assert_runners_pinned_with_timeouts(floating)
        with pytest.raises(AssertionError):
            _assert_runners_pinned_with_timeouts(
                _load(TEXT.replace("timeout-minutes: 5", "timeout-minutes: x", 1))
            )

    def test_multi_line_scripts_fail_fast(self) -> None:
        _assert_multi_line_scripts_fail_fast(_load(TEXT))
        with pytest.raises(AssertionError):
            _assert_multi_line_scripts_fail_fast(
                _load(TEXT.replace("set -euo pipefail", "set -e"))
            )
        with pytest.raises(AssertionError):
            _assert_multi_line_scripts_fail_fast(
                _load(TEXT.replace("shell: bash", "shell: sh"))
            )

    def test_a_mutant_must_change_the_text_and_be_caught(self) -> None:
        def problems(text: str) -> set[str]:
            return {"rule"} if "BAD" in text else set()

        _assert_mutant_caught("ok", lambda t: t + "BAD", "rule", "caught", problems)
        with pytest.raises(AssertionError):  # the mutant changed nothing
            _assert_mutant_caught("BAD", lambda t: t, "rule", "same", problems)
        with pytest.raises(AssertionError):  # the rule did not see it
            _assert_mutant_caught("ok", lambda t: t + "x", "rule", "missed", problems)
        with pytest.raises(AssertionError):  # another rule saw it, not this one
            _assert_mutant_caught("ok", lambda t: t + "BAD", "other", "wrong", problems)
