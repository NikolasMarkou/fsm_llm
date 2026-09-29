"""Tests for fsm_llm.eval.examples: discovery, subprocess runs, reports.

A tiny examples tree is written to a tmp dir; every script runs under the
current interpreter, so no LLM and no repository example is touched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from fsm_llm.eval import (
    EvalConfig,
    EvalError,
    discover_examples,
    get_timeout,
    merge_config,
    run_examples,
)
from fsm_llm.eval import examples as examples_mod
from fsm_llm.eval.constants import CATEGORY_TIMEOUTS, EVALUATOR_NAME, EXAMPLE_TIMEOUTS
from fsm_llm.eval.examples import ExampleTarget, create_output_dir, run_example

_REPO_ROOT = Path(__file__).resolve().parents[2]
_HISTORICAL_RESULTS = (
    _REPO_ROOT / "evaluation" / "2026-04-01_23-35_4025a78_qwen3.5-4b" / "results.json"
)

_OK_SCRIPT = """\
import os, sys
print("model=" + os.environ["LLM_MODEL"])
print("stdin=" + sys.stdin.read().replace("\\n", "|"))
print("State: start")
print("State: done")
print("Extraction rate: 3/3 (100%)")
"""
_BOOM_SCRIPT = "raise RuntimeError('boom')\n"
_SLOW_SCRIPT = "import time\nprint('started', flush=True)\ntime.sleep(30)\n"
_MANUAL_SCRIPT = "x = input('name? ')\nprint('manual got', x)\n"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def examples_dir(tmp_path: Path) -> Path:
    root = tmp_path / "examples"
    _write(root / "basic" / "ok" / "run.py", _OK_SCRIPT)
    _write(root / "basic" / "ok" / "run_manual.py", _MANUAL_SCRIPT)
    _write(root / "basic" / "boom" / "run.py", _BOOM_SCRIPT)
    _write(root / "agents" / "slow" / "run.py", _SLOW_SCRIPT)
    _write(root / "toplevel" / "run.py", "print('too shallow')\n")  # 2 parts: skipped
    return root


def _config(examples_dir: Path, **overrides) -> EvalConfig:
    layer = {
        "examples_dir": str(examples_dir),
        "output_root": str(examples_dir.parent / "evaluation"),
        "model": "ollama_chat/test:1b",
        "workers": 4,
        "example_inputs": {"basic/ok": "hello\nquit\n"},
        # agents default to 180 s in the built-in table, which beats --timeout
        "category_timeouts": {"agents": 1},
    }
    return merge_config(layer, overrides)


class TestDiscovery:
    def test_names_categories_and_order(self, examples_dir: Path):
        names = [t.name for t in discover_examples(_config(examples_dir))]
        # all run.py (sorted by path) first, then run_manual.py; 2-part skipped
        assert names == ["agents/slow", "basic/boom", "basic/ok", "basic/ok_manual"]

    def test_target_fields(self, examples_dir: Path):
        targets = {t.name: t for t in discover_examples(_config(examples_dir))}
        assert targets["basic/ok"].category == "basic"
        assert targets["basic/ok"].stdin_data == "hello\nquit\n"
        assert targets["basic/ok_manual"].interactive is True
        assert targets["basic/ok"].interactive is False
        assert targets["basic/boom"].stdin_data is None
        assert targets["agents/slow"].timeout == 1
        assert targets["basic/ok"].timeout == 120
        assert targets["basic/ok"].path.is_absolute()

    def test_category_and_name_filters(self, examples_dir: Path):
        by_cat = discover_examples(_config(examples_dir, category="basic"))
        assert {t.category for t in by_cat} == {"basic"}
        by_name = discover_examples(_config(examples_dir, name_filter="ok"))
        assert [t.name for t in by_name] == ["basic/ok", "basic/ok_manual"]

    def test_missing_dir_finds_nothing(self, tmp_path: Path):
        assert discover_examples(_config(tmp_path / "nope")) == []


class TestTimeoutPrecedence:
    def test_example_table_beats_category_and_default(self):
        assert get_timeout("advanced/e_commerce", "advanced", 10) == 300

    def test_category_table_beats_default_even_when_default_is_lower(self):
        assert get_timeout("agents/new", "agents", 5) == CATEGORY_TIMEOUTS["agents"]

    def test_default_when_untabled(self):
        assert get_timeout("basic/new", "basic", 7) == 7

    def test_config_tables_override_builtins(self, examples_dir: Path):
        config = _config(examples_dir, example_timeouts={"agents/slow": 3})
        slow = next(t for t in discover_examples(config) if t.name == "agents/slow")
        assert slow.timeout == 3
        # the built-in table itself is untouched
        assert "agents/slow" not in EXAMPLE_TIMEOUTS


@pytest.fixture
def finished_run(examples_dir: Path):
    config = _config(examples_dir)
    targets = discover_examples(config)
    run_dir = create_output_dir(config, "ollama_chat/test:1b")
    seen: list[tuple[int, int, str]] = []
    report = run_examples(
        targets,
        config,
        "ollama_chat/test:1b",
        run_dir,
        progress=lambda done, total, r: seen.append((done, total, r.name)),
    )
    return report, seen


class TestRunExamples:
    def test_results_sorted_and_scored(self, finished_run):
        report, _ = finished_run
        by_name = {r.name: r for r in report.results}
        assert [r.name for r in report.results] == sorted(by_name)
        assert by_name["basic/ok"].score == 4
        assert by_name["basic/ok"].exit_code == 0
        assert by_name["basic/boom"].score == 0  # fast crash, empty stdout
        assert by_name["basic/boom"].failures == ["F-CODE"]

    def test_stdin_and_model_reach_the_subprocess(self, finished_run):
        report, _ = finished_run
        ok = next(r for r in report.results if r.name == "basic/ok")
        assert "model=ollama_chat/test:1b" in ok.stdout
        assert "stdin=hello|quit|" in ok.stdout

    def test_timeout_keeps_partial_output(self, finished_run):
        report, _ = finished_run
        slow = next(r for r in report.results if r.name == "agents/slow")
        assert slow.timed_out is True
        assert slow.exit_code is None
        assert slow.error == "Timeout after 1s"
        assert slow.score == 1
        assert slow.failures == ["F-LOOP"]

    def test_progress_called_once_per_example(self, finished_run):
        report, seen = finished_run
        assert [done for done, _, _ in seen] == [1, 2, 3, 4]
        assert {total for _, total, _ in seen} == {4}
        assert sorted(name for _, _, name in seen) == [r.name for r in report.results]

    def test_log_files_at_historical_paths(self, finished_run):
        report, _ = finished_run
        log = report.run_dir / "logs" / "basic" / "basic_ok_manual.log"
        text = log.read_text(encoding="utf-8")
        assert text.startswith("# Example: basic/ok_manual\n# Exit code: ")
        assert "=== STDOUT ===" in text and "=== STDERR ===" in text
        slow_log = report.run_dir / "logs" / "agents" / "agents_slow.log"
        assert "# Error: Timeout after 1s" in slow_log.read_text(encoding="utf-8")

    def test_results_json_keeps_every_historical_key(self, finished_run):
        report, _ = finished_run
        data = json.loads((report.run_dir / "results.json").read_text())
        old = json.loads(_HISTORICAL_RESULTS.read_text())
        assert set(old) <= set(data)
        assert set(data) - set(old) == {
            "wall_time_s",
            "workers",
            "default_timeout",
            "evaluator",
        }
        assert set(old["results"][0]) == set(data["results"][0])
        assert set(data["distribution"]) == {"0", "1", "2", "3", "4"}
        assert data["total_examples"] == 4
        assert data["evaluator"] == EVALUATOR_NAME
        assert data["workers"] == 4
        assert data["default_timeout"] == 120

    def test_scorecard_sections(self, finished_run):
        report, _ = finished_run
        text = (report.run_dir / "scorecard.md").read_text(encoding="utf-8")
        for heading in ("## Scores", "## Summary", "## Timing"):
            assert heading in text
        assert "| basic/boom | 0 (CRASH) |" in text
        assert "- **Total wall time**:" in text
        assert "- **Total example time**:" in text
        assert f"**{report.health:.1f}%**" in text

    def test_wall_time_is_real_not_summed(self, finished_run):
        report, _ = finished_run
        # four examples in parallel: the slow one alone takes about 1 s
        assert report.wall_time < sum(r.duration for r in report.results) + 1.0
        assert report.wall_time >= max(r.duration for r in report.results)


class TestFailureRecording:
    def test_crashed_worker_is_recorded_not_dropped(self, examples_dir, monkeypatch):
        real = examples_mod.run_example

        def flaky(target, *args):
            if target.name == "basic/ok":
                raise OSError("fork failed")
            return real(target, *args)

        monkeypatch.setattr(examples_mod, "run_example", flaky)
        config = _config(examples_dir, category="basic")
        run_dir = create_output_dir(config, "m")
        report = run_examples(discover_examples(config), config, "m", run_dir)
        ok = next(r for r in report.results if r.name == "basic/ok")
        assert ok.exit_code == -1
        assert ok.score == 0
        assert "fork failed" in (ok.error or "")
        assert len(report.results) == 3

    def test_missing_interpreter_scores_zero(self, tmp_path: Path):
        target = ExampleTarget("c/x", "c", tmp_path / "run.py", False, None, 5)
        result = run_example(target, "m", str(tmp_path / "no-python"), tmp_path)
        assert result.exit_code == -1
        assert result.score == 0

    def test_empty_run_writes_report_without_timing(self, tmp_path: Path):
        config = merge_config({"examples_dir": str(tmp_path)})
        run_dir = tmp_path / "out"
        run_dir.mkdir()
        report = run_examples([], config, "m", run_dir)
        assert report.health == 0.0
        text = (run_dir / "scorecard.md").read_text(encoding="utf-8")
        assert "## Timing" not in text


class TestOutputDir:
    def test_second_run_never_reuses_the_first_dir(self, examples_dir: Path):
        config = _config(examples_dir)
        first = create_output_dir(config, "ollama_chat/test:1b")
        (first / "marker").write_text("keep")
        second = create_output_dir(config, "ollama_chat/test:1b")
        assert first != second
        assert (first / "marker").read_text() == "keep"
        assert first.name.endswith("_test-1b")

    def test_explicit_dir_created_when_missing(self, tmp_path: Path):
        out = tmp_path / "a" / "b"
        assert create_output_dir(EvalConfig(output_dir=str(out)), "m") == out
        assert out.is_dir()

    def test_explicit_nonempty_dir_is_refused(self, tmp_path: Path):
        (tmp_path / "old.txt").write_text("x")
        with pytest.raises(EvalError, match="not empty"):
            create_output_dir(EvalConfig(output_dir=str(tmp_path)), "m")

    def test_default_interpreter_is_the_running_one(self, examples_dir, monkeypatch):
        seen: list[str] = []

        def capture(target, model, python, cwd):
            seen.append(python)
            return examples_mod._crashed(target, "not run")

        monkeypatch.setattr(examples_mod, "run_example", capture)
        config = _config(examples_dir, name_filter="boom")
        run_examples(
            discover_examples(config), config, "m", create_output_dir(config, "m")
        )
        assert seen == [sys.executable]
