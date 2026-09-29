"""
Command-line entry point for :mod:`fsm_llm.eval` (``fsm-llm-eval``).

Usage::

    fsm-llm-eval examples                         # every example, 4 workers
    fsm-llm-eval examples --category agents --workers 6
    fsm-llm-eval examples --filter react --model gpt-4o-mini
    fsm-llm-eval examples --list                  # discovered examples only
    fsm-llm-eval examples --config eval.json --fail-under 80
    fsm-llm-eval run cases.json                   # 3 trials per case
    fsm-llm-eval run cases.jsonl --trials 5 --fail-under 90
    fsm-llm-eval run cases.json --list            # case ids only

Settings precedence: built-in defaults < a dataset's embedded ``config`` (``run``
only) < ``--config FILE`` < explicit flags.

**Exit codes.** ``0`` success, also for low scores unless ``--fail-under`` is
given; ``1`` usage error, bad config, unwritable output, or nothing matched;
``2`` only when ``--fail-under PCT`` is given and the score is below it.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .cases import TrialResult, load_cases, run_cases
from .config import EvalConfig, load_config, merge_config, resolve_model
from .constants import (
    EXIT_BELOW_THRESHOLD,
    EXIT_ERROR,
    EXIT_OK,
    MAX_SCORE,
    SCORE_LABELS,
)
from .examples import (
    ExampleResult,
    create_output_dir,
    discover_examples,
    run_examples,
)
from .exceptions import EvalError
from .records import open_run_dir

__all__ = ["main_cli", "run"]

_PROG = "fsm-llm-eval"
_RULE = "=" * 60


class _Parser(argparse.ArgumentParser):
    """An ``ArgumentParser`` whose usage errors exit 1, not argparse's 2.

    Exit 2 is reserved for "below ``--fail-under``", so a CI job can tell a
    regression from a typo'd flag.
    """

    def error(self, message: str) -> Any:
        self.print_usage(sys.stderr)
        print(f"{self.prog}: error: {message}", file=sys.stderr)
        raise SystemExit(EXIT_ERROR)


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    """Options every evaluation command shares. Every default is ``None``,
    so only flags the user actually passed override the config layers."""
    parser.add_argument(
        "--model", help="LLM model (default: $LLM_MODEL, else the framework default)"
    )
    parser.add_argument("--workers", type=int, help="Parallel workers (default: 4)")
    parser.add_argument(
        "--output-dir", help="Run directory, used verbatim; must be new or empty"
    )
    parser.add_argument("--config", help="JSON file of settings (flags override it)")
    parser.add_argument(
        "--fail-under",
        type=float,
        metavar="PCT",
        help="Exit 2 when the score is below PCT (0-100)",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser; each subparser sets ``func`` to its handler."""
    parser = _Parser(
        prog=_PROG, description="Evaluate FSM-LLM examples and conversations."
    )
    parser.add_argument("--version", action="store_true", help="Show version and exit")
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    examples = subparsers.add_parser(
        "examples",
        help="Run example scripts and score them 0-4",
        description="Run every examples/<category>/<name>/run.py, score it, and "
        "write scorecard.md, results.json and logs/ to a new run directory.",
    )
    _add_common_options(examples)
    examples.add_argument(
        "--timeout", type=int, help="Default seconds per example (default: 120)"
    )
    examples.add_argument("--category", help="Only this category (e.g. agents)")
    examples.add_argument(
        "--filter", dest="name_filter", help="Only names containing this substring"
    )
    examples.add_argument(
        "--examples-dir", help="Examples tree to scan (default: examples)"
    )
    examples.add_argument(
        "--python", help="Interpreter for the example scripts (default: this one)"
    )
    examples.add_argument(
        "--list", action="store_true", help="List discovered examples and exit"
    )
    examples.set_defaults(func=_cmd_examples)

    cases = subparsers.add_parser(
        "run",
        help="Run scripted conversations and check expectations",
        description="Run every case of a conversation dataset (.json or .jsonl) "
        "N times and write rows.jsonl, results.json and summary.md to a new run "
        "directory.",
    )
    cases.add_argument("dataset", help="Dataset file (.json list/object or .jsonl)")
    _add_common_options(cases)
    cases.add_argument("--trials", type=int, help="Trials per case (default: 3)")
    cases.add_argument("--list", action="store_true", help="List cases and exit")
    cases.set_defaults(func=_cmd_run)
    return parser


def _config_from(
    args: argparse.Namespace,
    fields: Sequence[str],
    base_layer: dict[str, Any] | None = None,
) -> EvalConfig:
    """Merge defaults < ``base_layer`` < ``--config`` file < the given flags in ``fields``."""
    file_layer = load_config(args.config) if args.config else None
    flag_layer = {
        name: getattr(args, name) for name in fields if getattr(args, name) is not None
    }
    return merge_config(base_layer, file_layer, flag_layer)


def _print_progress(completed: int, total: int, result: ExampleResult) -> None:
    """One line per finished example (the ``scripts/eval.py`` format)."""
    label = SCORE_LABELS[result.score]
    fails = f" [{', '.join(result.failures)}]" if result.failures else ""
    timeout = " TIMEOUT" if result.timed_out else ""
    print(
        f"  [{completed:2d}/{total}] {result.name:45s} "
        f"{result.score} ({label:7s}) {result.duration:6.1f}s{timeout}{fails}",
        flush=True,
    )


def _cmd_examples(args: argparse.Namespace) -> int:
    """Handle ``examples``: list, or run, report and apply ``--fail-under``."""
    config = _config_from(
        args,
        (
            "model",
            "workers",
            "timeout",
            "category",
            "name_filter",
            "output_dir",
            "examples_dir",
            "python",
            "fail_under",
        ),
    )
    targets = discover_examples(config)
    if not targets:
        print("No examples found matching filters.", file=sys.stderr)
        return EXIT_ERROR
    if args.list:
        print(f"Discovered {len(targets)} examples:\n")
        for t in targets:
            mode = "interactive" if t.interactive else "automated"
            stdin = "has stdin" if t.stdin_data else "no stdin"
            print(f"  {t.name:45s} [{mode}, {stdin}]")
        return EXIT_OK

    model = resolve_model(config)
    run_dir = create_output_dir(config, model)
    print("FSM-LLM Evaluation Runner", _RULE, sep="\n")
    print(f"  Model:    {model}")
    print(f"  Workers:  {config.workers}")
    print(f"  Timeout:  {config.timeout}s (default)")
    print(f"  Examples: {len(targets)}")
    print(f"  Output:   {run_dir}")
    print(_RULE, flush=True)
    print()

    report = run_examples(targets, config, model, run_dir, progress=_print_progress)
    score_sum = sum(r.score for r in report.results)
    print(f"\n{_RULE}")
    print(
        f"  Health Score: {score_sum}/{len(report.results) * MAX_SCORE} = {report.health:.1f}%"
    )
    print(
        f"  Wall time:   {report.wall_time:.1f}s ({len(report.results)} examples, {config.workers} workers)"
    )
    print(f"  Scorecard:   {run_dir / 'scorecard.md'}")
    print(f"  Logs:        {run_dir / 'logs'}/")
    print(f"  JSON:        {run_dir / 'results.json'}")
    print(_RULE)
    if config.fail_under is not None and report.health < config.fail_under:
        print(
            f"Health score {report.health:.1f}% is below --fail-under {config.fail_under:g}%",
            file=sys.stderr,
        )
        return EXIT_BELOW_THRESHOLD
    return EXIT_OK


def _print_trial(completed: int, total: int, trial: TrialResult) -> None:
    """One line per finished trial."""
    status = "PASS" if trial.passed else "FAIL"
    why = f" [{trial.first_failure()}]" if not trial.passed else ""
    label = f"{trial.case_id}#{trial.trial}"
    print(
        f"  [{completed:2d}/{total}] {label:45s} {status} {trial.duration:6.1f}s{why}",
        flush=True,
    )


def _cmd_run(args: argparse.Namespace) -> int:
    """Handle ``run``: list, or run the cases, report and apply ``--fail-under``."""
    cases, embedded = load_cases(args.dataset)
    config = _config_from(
        args,
        ("model", "workers", "trials", "output_dir", "fail_under"),
        base_layer=embedded,
    )
    if args.list:
        print(f"Loaded {len(cases)} cases from {args.dataset}:\n")
        for case in cases:
            print(f"  {case.id:45s} [{len(case.turns)} turns] {case.description}")
        return EXIT_OK

    model = resolve_model(config)
    git_cwd = Path(args.dataset).resolve().parent
    run_dir = open_run_dir(config.output_dir, config.output_root, model, cwd=git_cwd)
    print("FSM-LLM Conversation Evaluation", _RULE, sep="\n")
    print(f"  Model:    {model}")
    print(f"  Dataset:  {args.dataset}")
    print(f"  Cases:    {len(cases)} x {config.trials} trials")
    print(f"  Workers:  {config.workers}")
    print(f"  Output:   {run_dir}")
    print(_RULE, flush=True)
    print()

    report = run_cases(
        cases, config, run_dir, progress=_print_trial, dataset=args.dataset
    )
    overall = report.overall
    rate = overall["rate"] * 100
    lo, hi = overall["wilson_ci"]
    print(f"\n{_RULE}")
    print(
        f"  Pass rate:  {overall['k']}/{overall['n']} = {rate:.1f}% "
        f"(Wilson 95% CI [{lo * 100:.1f}%, {hi * 100:.1f}%])"
    )
    print(f"  Wall time:  {report.wall_time:.1f}s")
    print(f"  Summary:    {run_dir / 'summary.md'}")
    print(f"  Rows:       {run_dir / 'rows.jsonl'}")
    print(f"  JSON:       {run_dir / 'results.json'}")
    print(_RULE)
    if config.fail_under is not None and rate < config.fail_under:
        print(
            f"Pass rate {rate:.1f}% is below --fail-under {config.fail_under:g}%",
            file=sys.stderr,
        )
        return EXIT_BELOW_THRESHOLD
    return EXIT_OK


def main_cli(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv`` and run one subcommand (``run()`` is the process entry).

    Interface contract:
        - ``argv``: arguments WITHOUT the program name; ``None`` reads
          ``sys.argv[1:]``.
        - Returns the exit code (0, 1, or 2 under ``--fail-under``).
          ``EvalError`` (bad config, unwritable output) is reported on stderr
          as exit 1, never raised.
        - Raises ``SystemExit`` only from argparse (``--help`` exits 0, a
          usage error exits 1).
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        from .__version__ import __version__

        print(f"{_PROG} {__version__}")
        return EXIT_OK
    handler: Callable[[argparse.Namespace], int] | None = getattr(args, "func", None)
    if handler is None:
        parser.print_help(sys.stderr)
        return EXIT_ERROR
    try:
        return handler(args)
    except EvalError as exc:
        print(f"{_PROG}: error: {exc}", file=sys.stderr)
        return EXIT_ERROR


def run(argv: Sequence[str] | None = None) -> int:
    """Process entry for ``fsm-llm-eval``, ``python -m fsm_llm.eval`` and ``scripts/eval.py``.

    Turns on ``fsm_llm`` log output at WARNING (``$FSM_LLM_LOG_LEVEL``
    overrides it), then returns ``main_cli(argv)``.
    """
    # Enable logging HERE, at the process entry, never in main_cli: tests call
    # main_cli in-process and must not flip global logging (see D-016 of plan
    # 3a032517 on fsm_llm.logging.setup_cli_logging).
    from fsm_llm.logging import setup_cli_logging

    setup_cli_logging("WARNING")
    return main_cli(argv)


if __name__ == "__main__":  # pragma: no cover - process entry point
    sys.exit(run())
