"""`workflow_support.py`: the helpers of the workflow tests fail when they should.

The five workflow tests (`test_docs_workflow.py`, `test_ci_workflow.py`,
`test_refresh_workflow.py`, `test_release_workflow.py`, `test_supply_chain.py`) judge their
mutants and real workflows through these helpers. A helper that asserted nothing would turn
every one of those tests green without checking a thing, so each one is shown to reject a bad
input here, and to give the old answer on a good one (final REFLECT hygiene items 7 and 8).
"""

from __future__ import annotations

import ast
import contextlib
import os
import sys
import types
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.conftest import posix_only
from tests.test_fsm_llm_regression import workflow_support as ws
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
    bash_problem,
    gnu_find_problem,
    passthrough_env,
    skip_unless_bash,
    skip_unless_gnu_find,
    write_script,
)

_TESTS = Path(__file__).resolve().parents[1]
_RELEASE = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "release.yml"

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


# --------------------------------------------------------------------------------------
# Running a script: the LF writer and the capability probes (plan d7719b03, D-006)
# --------------------------------------------------------------------------------------


def _stub(directory: Path, name: str, body: str) -> Path:
    """An executable `name` in `directory` (a shell stub)."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    write_script(path, "#!/bin/sh\n" + body)
    path.chmod(0o755)
    return path


@contextlib.contextmanager
def _pretend_platform(name: str) -> Iterator[None]:
    """`sys.platform` is *name* inside the block (the helpers read it at call time)."""
    before = sys.platform
    sys.platform = name
    try:
        yield
    finally:
        sys.platform = before


def _probe_contract_failures(mod: types.SimpleNamespace, base: Path) -> list[str]:
    """The labels of the probe expectations `mod` (the real module or a mutant) breaks.

    Each case is a PATH and whether the probe must report a problem there. The stubs stand
    for what a Windows runner offers: no bash, a WSL-like bash that fails, a bash
    that prints something else, a BSD find that knows no `-printf`.
    """
    real = os.environ["PATH"]
    nobash = base / "nobash"
    nobash.mkdir(parents=True)
    _stub(base / "failing-bash", "bash", "echo 'no distribution' >&2\nexit 1\n")
    _stub(base / "odd-bash", "bash", "echo something else\n")
    _stub(
        base / "bsd-find", "find", "echo 'find: -printf: unknown primary' >&2\nexit 1\n"
    )
    _stub(base / "gnu-find", "find", "printf x\n")
    _stub(base / "good-bash", "bash", "echo ok\n")
    both = os.pathsep.join
    cases = [
        ("bash missing", mod.bash_problem(str(nobash)), True),
        ("bash exits 1", mod.bash_problem(str(base / "failing-bash")), True),
        ("bash prints other text", mod.bash_problem(str(base / "odd-bash")), True),
        ("bash says ok", mod.bash_problem(str(base / "good-bash")), False),
        ("find without bash", mod.gnu_find_problem(str(nobash)), True),
        (
            "BSD find first",
            mod.gnu_find_problem(both([str(base / "bsd-find"), real])),
            True,
        ),
        (
            "GNU-like find first",
            mod.gnu_find_problem(both([str(base / "gnu-find"), real])),
            False,
        ),
    ]
    failures = [label for label, got, bad in cases if (got is not None) != bad]
    gnu_path = both([str(base / "gnu-find"), real])
    with _pretend_platform("win32"):  # a usable bash and GNU find, and still no run
        failures += [
            label
            for label, got in [
                ("win32: bash_problem", mod.bash_problem(gnu_path)),
                ("win32: gnu_find_problem", mod.gnu_find_problem(gnu_path)),
            ]
            if got is None
        ]
        for label, call in [
            ("win32: skip_unless_bash", mod.skip_unless_bash),
            ("win32: skip_unless_gnu_find", mod.skip_unless_gnu_find),
        ]:
            try:
                call(gnu_path)
                failures.append(label)
            except pytest.skip.Exception:
                pass
    for label, call, path, skips in [
        ("skip_unless_bash on no bash", mod.skip_unless_bash, str(nobash), True),
        (
            "skip_unless_bash on a bash",
            mod.skip_unless_bash,
            str(base / "good-bash"),
            False,
        ),
        (
            "skip_unless_gnu_find on BSD find",
            mod.skip_unless_gnu_find,
            both([str(base / "bsd-find"), real]),
            True,
        ),
        (
            "skip_unless_gnu_find on GNU find",
            mod.skip_unless_gnu_find,
            both([str(base / "gnu-find"), real]),
            False,
        ),
    ]:
        try:
            call(path)
            skipped = False
        except pytest.skip.Exception:
            skipped = True
        if skipped != skips:
            failures.append(label)
    return failures


def _variant(old: str, new: str) -> types.SimpleNamespace:
    """The `workflow_support` module with `old` replaced by `new` (a mutant), fresh caches."""
    source = Path(ws.__file__).read_text(encoding="utf-8")
    assert source.count(old) == 1, old
    namespace: dict[str, object] = {"__name__": "workflow_support_mutant"}
    exec(compile(source.replace(old, new), "<mutant>", "exec"), namespace)
    return types.SimpleNamespace(**namespace)


_BASH_CALL = (
    'return _bash_problem(_search_path(path), os.environ.get("SYSTEMROOT", ""))'
)
_FIND_CALL = (
    'return _gnu_find_problem(_search_path(path), os.environ.get("SYSTEMROOT", ""))'
)
_BASH_WIN = "    if problem := _windows_problem():\n        return problem\n    return _bash_problem("
_FIND_WIN = "    if problem := _windows_problem():\n        return problem\n    return _gnu_find_problem("
_BASH_SKIP = '    if problem:\n        pytest.skip(f"these tests run'


class TestScriptWriter:
    def test_the_script_is_written_with_lf_only(self, tmp_path: Path) -> None:
        write_script(tmp_path / "s.sh", "set -e\necho a\n")
        assert (tmp_path / "s.sh").read_bytes() == b"set -e\necho a\n"

    def test_the_writer_asks_for_no_newline_translation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Linux cannot translate "\n" to "\r\n", so the argument itself is what is pinned.
        seen: dict[str, object] = {}

        def recorder(self: Path, data: str, **kwargs: object) -> int:
            seen.update(kwargs)
            return 0

        monkeypatch.setattr(Path, "write_text", recorder)
        write_script(tmp_path / "s.sh", "x\n")
        assert seen == {"encoding": "utf-8", "newline": "\n"}

    def test_passthrough_env_carries_systemroot_only_when_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SYSTEMROOT", "C:\\Windows")
        assert passthrough_env() == {"SYSTEMROOT": "C:\\Windows"}
        monkeypatch.delenv("SYSTEMROOT")
        assert passthrough_env() == {}
        monkeypatch.setenv("SYSTEMROOT", "")
        assert passthrough_env() == {}

    def test_the_release_runner_hands_systemroot_to_the_script(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tests.test_fsm_llm_regression.test_release_workflow import _run

        monkeypatch.setenv("SYSTEMROOT", "C:\\Windows")
        _, _, err = _run('echo "[${SYSTEMROOT-unset}]" >&2', tmp_path / "a")
        monkeypatch.delenv("SYSTEMROOT")
        _, _, err_unset = _run('echo "[${SYSTEMROOT-unset}]" >&2', tmp_path / "b")

        assert err.strip() == "[C:\\Windows]" and err_unset.strip() == "[unset]"


class TestProbes:
    @posix_only("the stubs are executable shell scripts")
    def test_the_probes_report_what_a_windows_runner_would_offer(
        self, tmp_path: Path
    ) -> None:
        assert _probe_contract_failures(ws_namespace(), tmp_path) == []

    @posix_only("the stubs are executable shell scripts")
    def test_on_win32_a_usable_bash_is_still_a_problem(self, tmp_path: Path) -> None:
        # The scripts run on ubuntu-24.04 only (plan d7719b03 D-009): on Windows the
        # bare name `bash` reaches WSL's launcher, whatever `which` found first.
        stub = _stub(tmp_path / "bin", "bash", "echo ok\n")
        assert bash_problem(str(stub.parent)) is None  # usable by the probe's own test

        with _pretend_platform("win32"):
            for problem in (
                bash_problem(str(stub.parent)),
                gnu_find_problem(os.environ["PATH"]),
            ):
                assert problem and "D-009" in problem and "ubuntu-24.04" in problem

    def test_on_win32_both_skip_helpers_skip_with_the_reason(self) -> None:
        with _pretend_platform("win32"):
            for helper in (skip_unless_bash, skip_unless_gnu_find):
                with pytest.raises(pytest.skip.Exception, match="D-009"):
                    helper(os.environ["PATH"])

    @posix_only("the stubs are executable shell scripts")
    @pytest.mark.parametrize("platform", ["linux", "darwin", "cygwin"])
    def test_on_any_other_platform_a_usable_bash_does_not_skip(
        self, tmp_path: Path, platform: str
    ) -> None:
        stub = _stub(tmp_path / "bin", "bash", "echo ok\n")
        with _pretend_platform(platform):
            assert bash_problem(str(stub.parent)) is None
            skip_unless_bash(str(stub.parent))  # raises when it skips

    @pytest.mark.skipif(
        sys.platform != "linux",
        reason="asserts that the probes do NOT skip on Linux (real bash, GNU find)",
    )
    def test_on_linux_nothing_is_skipped_because_the_probes_find_bash_and_gnu_find(
        self,
    ) -> None:
        assert bash_problem() is None
        assert gnu_find_problem() is None
        try:
            skip_unless_bash()
            skip_unless_gnu_find()
        except pytest.skip.Exception as skipped:  # pragma: no cover - the failure
            pytest.fail(f"every shell-script test would be skipped on Linux: {skipped}")

    @posix_only("the stubs are executable shell scripts")
    @pytest.mark.parametrize(
        "label,old,new",
        [
            (
                "the win32 rule is gone",
                '    if sys.platform == "win32":',
                "    if False:",
            ),
            ("bash_problem ignores win32", _BASH_WIN, "    return _bash_problem("),
            (
                "gnu_find_problem ignores win32",
                _FIND_WIN,
                "    return _gnu_find_problem(",
            ),
            ("bash_problem always None", _BASH_CALL, "return None"),
            ("bash_problem always a problem", _BASH_CALL, 'return "always"'),
            ("gnu_find_problem always None", _FIND_CALL, "return None"),
            ("gnu_find_problem always a problem", _FIND_CALL, 'return "always"'),
            (
                "skip_unless_bash never skips",
                _BASH_SKIP,
                '    if False:\n        pytest.skip(f"these tests run',
            ),
            (
                "skip_unless_bash always skips",
                "    problem = bash_problem(path)\n    if problem:",
                "    problem = bash_problem(path) or 'x'\n    if problem:",
            ),
        ],
    )
    def test_a_probe_mutant_breaks_the_contract(
        self, tmp_path: Path, label: str, old: str, new: str
    ) -> None:
        assert _probe_contract_failures(_variant(old, new), tmp_path), label


def ws_namespace() -> types.SimpleNamespace:
    """The real module as a namespace (the same shape as a mutant)."""
    return types.SimpleNamespace(
        bash_problem=bash_problem,
        gnu_find_problem=gnu_find_problem,
        skip_unless_bash=skip_unless_bash,
        skip_unless_gnu_find=skip_unless_gnu_find,
    )


def _calls(path: Path) -> set[str]:
    """The names of the functions `path` calls, plus the `["bash", ...]` lists it holds."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call):
            func = node.func
            found.add(
                func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
            )
        if (
            isinstance(node, ast.List | ast.Tuple)
            and node.elts
            and isinstance(node.elts[0], ast.Constant)
            and node.elts[0].value == "bash"
        ):
            found.add('["bash"]')
    return found


class TestEveryBashRunnerAsksFirst:
    def _modules(self) -> dict[str, set[str]]:
        return {
            p.name: _calls(p)
            for p in sorted(_TESTS.rglob("test_*.py"))
            if p.name != Path(__file__).name
        }

    def test_the_four_workflow_modules_run_bash(self) -> None:
        runners = {n for n, c in self._modules().items() if '["bash"]' in c}
        assert runners == {
            "test_docs_workflow.py",
            "test_refresh_workflow.py",
            "test_release_workflow.py",
            "test_supply_chain.py",
        }

    def test_every_module_that_runs_bash_calls_the_probe_and_the_lf_writer(
        self,
    ) -> None:
        for name, calls in self._modules().items():
            if '["bash"]' in calls:
                assert "skip_unless_bash" in calls, name
                assert "write_script" in calls, name

    def test_the_release_name_check_asks_for_gnu_find(self) -> None:
        assert "-printf" in _RELEASE.read_text(encoding="utf-8")  # else drop the rule
        assert "skip_unless_gnu_find" in self._modules()["test_release_workflow.py"]

    @pytest.mark.parametrize("form", ["['bash', 'x']", "('bash', 'x')"])
    def test_the_scan_sees_a_runner_that_does_not_ask(
        self, tmp_path: Path, form: str
    ) -> None:
        bad = tmp_path / "test_x.py"
        bad.write_text(f"import subprocess\nsubprocess.run({form})\n", encoding="utf-8")
        assert '["bash"]' in _calls(bad) and "skip_unless_bash" not in _calls(bad)
