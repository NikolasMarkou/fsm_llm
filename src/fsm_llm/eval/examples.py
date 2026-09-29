"""
Examples evaluation: run every ``run.py`` under an examples tree and score it.

Discovers ``<examples_dir>/<category>/<example>/run.py`` (and ``run_manual.py``),
runs each in its own subprocess with a canned stdin script, scores the output
with the 0-4 heuristic in :mod:`fsm_llm.eval.scoring`, and writes the historical
``scripts/eval.py`` layout into the run directory: ``scorecard.md``,
``results.json`` and ``logs/<category>/<category>_<example>.log``.

Differences from ``scripts/eval.py`` at 0facf56 (scores are unchanged): the
interpreter defaults to the running one, a crashed worker is recorded as a
score-0 result instead of being dropped, the scorecard reports real wall time
(the old sum of durations is now "Total example time"), a run directory is
never reused, Ctrl-C stops the run and still writes a partial report, and
``results.json`` gains ``wall_time_s``, ``workers``, ``default_timeout``,
``evaluator`` and ``interrupted``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from concurrent.futures import Future
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ._pool import run_interruptible
from .config import EvalConfig
from .constants import (
    CATEGORY_TIMEOUTS,
    EVALUATOR_NAME,
    EXAMPLE_INPUTS,
    EXAMPLE_SCRIPT_NAMES,
    EXAMPLE_TIMEOUTS,
    MAX_SCORE,
    SCORE_LABELS,
)
from .records import git_short_hash, open_run_dir, write_json
from .scoring import classify_result

#: Called once per finished example: ``(completed, total, result)``.
ProgressCallback = Callable[[int, int, "ExampleResult"], None]


@dataclass(frozen=True)
class ExampleTarget:
    """One discovered example script and how to run it."""

    name: str  # "<category>/<dir>" or "<category>/<dir>_manual"
    category: str
    path: Path
    interactive: bool  # source contains "input(" (informational, --list)
    stdin_data: str | None
    timeout: int


@dataclass
class ExampleResult:
    """Outcome of one example run; ``score``/``failures`` are set once, on completion."""

    name: str
    category: str
    exit_code: int | None  # None = timeout, -1 = could not run
    duration: float
    stdout: str
    stderr: str
    timed_out: bool
    error: str | None = None
    score: int = 0
    failures: list[str] = field(default_factory=list)


@dataclass
class ExampleReport:
    """A finished examples run: its directory, sorted results and health score."""

    run_dir: Path
    model: str
    results: list[ExampleResult]
    health: float
    wall_time: float
    interrupted: bool = False  # Ctrl-C: results hold only finished examples


def get_timeout(
    name: str,
    category: str,
    default: int,
    example_timeouts: Mapping[str, int] = EXAMPLE_TIMEOUTS,
    category_timeouts: Mapping[str, int] = CATEGORY_TIMEOUTS,
) -> int:
    """Timeout for one example: per-example table, then category table, then default."""
    # DECISION plan-2026-09-29T061903-581c2634/D-003
    # Do NOT let `default` (the --timeout flag) win over the tables, and do not
    # take min()/max() with it. This is the historical scripts/eval.py rule;
    # changing it changes which examples time out, and new scores would stop
    # being comparable with Run 001-006 in EVALUATE.md. To shorten or lengthen
    # a tabled example, override the table entry from a --config file.
    if name in example_timeouts:
        return example_timeouts[name]
    if category in category_timeouts:
        return category_timeouts[category]
    return default


def discover_examples(config: EvalConfig) -> list[ExampleTarget]:
    """Find the example scripts selected by ``config``, sorted like the old runner.

    Interface contract (callers: the ``examples`` CLI, ``--list``, tests):
        - Scans ``config.examples_dir`` for every name in
          ``EXAMPLE_SCRIPT_NAMES`` (all ``run.py`` first, then all
          ``run_manual.py``, each sorted by path). Scripts fewer than two
          directories deep are skipped.
        - Names are ``<category>/<dir>`` plus ``_<x>`` for ``run_<x>.py``;
          ``config.category`` must equal the category and ``config.name_filter``
          must be a substring of the name.
        - stdin and timeout come from the built-in tables merged with the
          config's table overrides.
        - Returns ``[]`` when the directory is missing or nothing matches.
    """
    root = Path(config.examples_dir)
    inputs = {**EXAMPLE_INPUTS, **config.example_inputs}
    example_timeouts = {**EXAMPLE_TIMEOUTS, **config.example_timeouts}
    category_timeouts = {**CATEGORY_TIMEOUTS, **config.category_timeouts}

    scripts = [
        path for script in EXAMPLE_SCRIPT_NAMES for path in sorted(root.rglob(script))
    ]
    targets = []
    for script in scripts:
        parts = script.relative_to(root).parts  # ("agents", "debate", "run.py")
        if len(parts) < 3:
            continue
        category = parts[0]
        suffix = "" if script.stem == "run" else f"_{script.stem[len('run_') :]}"
        name = f"{category}/{parts[1]}{suffix}"
        if config.category and category != config.category:
            continue
        if config.name_filter and config.name_filter not in name:
            continue
        try:
            interactive = "input(" in script.read_text()
        except OSError:
            interactive = False
        targets.append(
            ExampleTarget(
                name=name,
                category=category,
                path=script.resolve(),
                interactive=interactive,
                stdin_data=inputs.get(name),
                timeout=get_timeout(
                    name, category, config.timeout, example_timeouts, category_timeouts
                ),
            )
        )
    return targets


def _decode(stream: str | bytes | None) -> str:
    """Partial output captured before a timeout, as text."""
    if isinstance(stream, bytes):
        return stream.decode("utf-8", errors="replace")
    return stream or ""


def run_example(
    target: ExampleTarget, model: str, python: str, cwd: Path
) -> ExampleResult:
    """Run one example to completion or timeout and return its scored result.

    The subprocess gets a copy of the environment plus ``LLM_MODEL=model`` and
    ``PYTHONDONTWRITEBYTECODE=1``, the target's stdin script (or none), and
    ``cwd``. Never raises: a timeout gives ``exit_code=None, timed_out=True``
    with the partial output; any other failure to run gives ``exit_code=-1``.
    """
    env = os.environ.copy()
    env["LLM_MODEL"] = model
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    start = time.monotonic()
    try:
        proc = subprocess.run(
            [python, str(target.path)],
            cwd=str(cwd),
            input=target.stdin_data,
            capture_output=True,
            text=True,
            timeout=target.timeout,
            env=env,
        )
        result = ExampleResult(
            target.name,
            target.category,
            exit_code=proc.returncode,
            duration=time.monotonic() - start,
            stdout=proc.stdout,
            stderr=proc.stderr,
            timed_out=False,
        )
    except subprocess.TimeoutExpired as exc:
        result = ExampleResult(
            target.name,
            target.category,
            exit_code=None,
            duration=time.monotonic() - start,
            stdout=_decode(exc.stdout),
            stderr=_decode(exc.stderr),
            timed_out=True,
            error=f"Timeout after {target.timeout}s",
        )
    except Exception as exc:  # interpreter missing, not executable, OS limits
        result = _crashed(target, str(exc), time.monotonic() - start)
    result.score, result.failures = classify_result(result)
    return result


def _crashed(target: ExampleTarget, error: str, duration: float = 0.0) -> ExampleResult:
    """A score-0 result for an example that could not be run at all."""
    result = ExampleResult(
        target.name,
        target.category,
        exit_code=-1,
        duration=duration,
        stdout="",
        stderr=error,
        timed_out=False,
        error=error,
    )
    result.score, result.failures = classify_result(result)
    return result


def write_example_log(result: ExampleResult, output_dir: Path) -> Path:
    """Write ``logs/<category>/<category>_<example>.log`` and return its path."""
    cat_dir = output_dir / "logs" / result.category
    cat_dir.mkdir(parents=True, exist_ok=True)
    log_path = cat_dir / f"{result.name.replace('/', '_')}.log"
    lines = [
        f"# Example: {result.name}",
        f"# Exit code: {result.exit_code}",
        f"# Duration: {result.duration:.1f}s",
        f"# Timed out: {result.timed_out}",
        *([f"# Error: {result.error}"] if result.error else []),
        f"# {'=' * 60}",
        "",
        "=== STDOUT ===",
    ]
    text = (
        "\n".join(lines) + "\n" + result.stdout + "\n\n=== STDERR ===\n" + result.stderr
    )
    log_path.write_text(text, encoding="utf-8")
    return log_path


def health_score(results: list[ExampleResult]) -> float:
    """Percentage of the maximum score reached; 0.0 for no results."""
    max_possible = len(results) * MAX_SCORE
    return sum(r.score for r in results) / max_possible * 100 if max_possible else 0.0


def create_output_dir(config: EvalConfig, model: str) -> Path:
    """Create the run directory (:func:`fsm_llm.eval.records.open_run_dir`).

    A fresh directory is named after the git commit of the examples tree's
    parent. Raises ``EvalError`` when no directory can be created.
    """
    repo = Path(config.examples_dir).resolve().parent
    return open_run_dir(config.output_dir, config.output_root, model, cwd=repo)


def run_examples(
    targets: list[ExampleTarget],
    config: EvalConfig,
    model: str,
    run_dir: Path,
    progress: ProgressCallback | None = None,
) -> ExampleReport:
    """Run ``targets`` on a thread pool and write logs, scorecard and results.json.

    Interface contract (callers: the ``examples`` CLI and the Python API):
        - ``config.workers`` threads each drive one subprocess
          (:func:`run_example`); the interpreter is ``config.python`` or
          ``sys.executable``, the working directory the parent of
          ``config.examples_dir``.
        - Each log is written as its example finishes, then ``progress`` is
          called. A worker that raises becomes a score-0 result with the
          error text.
        - Ctrl-C (``KeyboardInterrupt``) cancels the examples not yet started
          and returns a partial report (``interrupted=True``, finished examples
          only), still written to disk.
        - Returns the report with results sorted by name.
    """
    python = config.python or sys.executable
    cwd = Path(config.examples_dir).resolve().parent
    results: list[ExampleResult] = []

    def record(target: ExampleTarget, future: Future[ExampleResult]) -> None:
        try:
            result = future.result()
        except Exception as exc:  # runner bug or OS error: record, never drop
            result = _crashed(target, f"runner error: {exc}")
        results.append(result)
        write_example_log(result, run_dir)
        if progress is not None:
            progress(len(results), len(targets), result)

    start = time.monotonic()
    interrupted = run_interruptible(
        lambda t: run_example(t, model, python, cwd), targets, config.workers, record
    )
    wall_time = time.monotonic() - start
    results.sort(key=lambda r: r.name)
    report = ExampleReport(
        run_dir, model, results, health_score(results), wall_time, interrupted
    )
    write_scorecard(report, config, git_short_hash(cwd))
    return report


def write_scorecard(
    report: ExampleReport,
    config: EvalConfig,
    git_hash: str,
    now: datetime | None = None,
) -> Path:
    """Write ``scorecard.md`` and ``results.json`` for ``report``; return the scorecard path.

    ``results.json`` keeps every key of the ``scripts/eval.py`` format (``date``,
    ``git_commit``, ``model``, ``health_score``, ``total_examples``,
    ``distribution``, ``results[name, category, score, failures, duration,
    exit_code, timed_out]``) and adds ``wall_time_s``, ``workers``,
    ``default_timeout``, ``evaluator`` and ``interrupted``.
    """
    now = now or datetime.now()
    results = report.results
    total = len(results)
    score_sum = sum(r.score for r in results)
    max_possible = total * MAX_SCORE
    dist = {i: sum(1 for r in results if r.score == i) for i in range(MAX_SCORE + 1)}
    cat_scores: dict[str, list[int]] = {}
    failure_counts: dict[str, int] = {}
    for r in results:
        cat_scores.setdefault(r.category, []).append(r.score)
        for code in r.failures:
            failure_counts[code] = failure_counts.get(code, 0) + 1

    stamp = now.strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# Evaluation: {stamp}",
        "",
        f"- **Date**: {stamp}",
        f"- **Git commit**: {git_hash}",
        f"- **Model**: {report.model}",
        f"- **Example count**: {total}",
        f"- **Workers**: {config.workers}",
        f"- **Evaluator**: {EVALUATOR_NAME}",
        "",
        "## Scores",
        "",
        "| # | Example | Score | Duration | Failures | Notes |",
        "|---|---------|-------|----------|----------|-------|",
    ]
    for i, r in enumerate(results, 1):
        if r.timed_out:
            note = f"Timeout ({r.duration:.0f}s)"
        elif r.exit_code != 0 and r.exit_code is not None:
            note = f"Exit {r.exit_code}"
        else:
            note = f"{r.duration:.1f}s"
        lines.append(
            f"| {i} | {r.name} | {r.score} ({SCORE_LABELS[r.score]}) | "
            f"{r.duration:.1f}s | {', '.join(r.failures)} | {note} |"
        )
    lines += [
        "",
        "## Summary",
        "",
        f"- **Total examples**: {total}",
        *(
            ["- **Interrupted**: yes, partial run (finished examples only)"]
            if report.interrupted
            else []
        ),
        f"- **Score distribution**: {dist[4]}x4, {dist[3]}x3, {dist[2]}x2, "
        f"{dist[1]}x1, {dist[0]}x0",
        f"- **Health Score**: {score_sum}/{max_possible} = **{report.health:.1f}%**",
        "- **Category breakdown**:",
    ]
    for cat in sorted(cat_scores):
        cat_sum, cat_max = sum(cat_scores[cat]), len(cat_scores[cat]) * MAX_SCORE
        lines.append(f"  - {cat}: {cat_sum}/{cat_max} ({cat_sum / cat_max * 100:.0f}%)")
    if failure_counts:
        top = sorted(failure_counts.items(), key=lambda item: -item[1])
        codes = ", ".join(f"{code} ({count})" for code, count in top)
        lines.append(f"- **Top failure codes**: {codes}")
    if results:
        durations = [r.duration for r in results]
        lines += [
            "",
            "## Timing",
            "",
            f"- **Total wall time**: {report.wall_time:.1f}s ({config.workers} workers)",
            f"- **Total example time**: {sum(durations):.1f}s (sequential equivalent)",
            f"- **Fastest**: {min(durations):.1f}s",
            f"- **Slowest**: {max(durations):.1f}s",
            f"- **Mean**: {sum(durations) / len(durations):.1f}s",
        ]
    lines.append("")

    scorecard_path = report.run_dir / "scorecard.md"
    scorecard_path.write_text("\n".join(lines), encoding="utf-8")
    data: dict[str, Any] = {
        "date": now.isoformat(),
        "git_commit": git_hash,
        "model": report.model,
        "health_score": round(report.health, 1),
        "total_examples": total,
        "distribution": {str(score): count for score, count in dist.items()},
        "results": [
            {
                "name": r.name,
                "category": r.category,
                "score": r.score,
                "failures": r.failures,
                "duration": round(r.duration, 1),
                "exit_code": r.exit_code,
                "timed_out": r.timed_out,
            }
            for r in results
        ],
        "wall_time_s": round(report.wall_time, 1),
        "workers": config.workers,
        "default_timeout": config.timeout,
        "evaluator": EVALUATOR_NAME,
        "interrupted": report.interrupted,
    }
    write_json(report.run_dir / "results.json", data)
    return scorecard_path
