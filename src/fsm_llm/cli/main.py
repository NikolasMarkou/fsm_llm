"""Top-level ``fsm-llm`` CLI dispatcher.

Implements 6 subcommands:

* ``run`` — execute an FSM JSON or a stdlib factory term interactively
  / one-shot. See :mod:`fsm_llm.cli.run`.
* ``explain`` — print :meth:`fsm_llm.program.Program.explain` output
  (Plan list, leaf schemas, AST shape) for an FSM JSON or factory
  target. See :mod:`fsm_llm.cli.explain`.
* ``validate`` — delegate to :func:`fsm_llm.validator.main`.
* ``visualize`` — delegate to :func:`fsm_llm.visualizer.main`.
* ``meta`` — delegate to
  :func:`fsm_llm.stdlib.agents.meta_cli.main_cli`.
* ``monitor`` — delegate to :func:`fsm_llm_monitor.__main__.main_cli`.

The legacy ``fsm-llm`` script (mapped to
:func:`fsm_llm.__main__.main_cli`) is preserved verbatim for back-compat
(invariant I4); the old ``--mode`` flag style is unchanged. The new
unified subparser surface lives here.

Subcommand selection:
    The first positional argument is treated as the subcommand name.
    Remaining arguments are passed through to the subcommand's own
    parser. The aliases (``fsm-llm-validate`` etc.) construct an
    ``argv`` list of ``["validate", *original_argv]`` and call
    :func:`main_cli` — this is the
    ``# DECISION D-PLAN-04`` aliasing pattern.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from ..__version__ import __version__

# --------------------------------------------------------------
# Subcommand names (closed registry — adding one requires edits in:
#   1. SUBCOMMANDS below
#   2. _add_<name>_subparser() function below
#   3. _dispatch() branch below
# This is intentional — the closed-set + strategy-dispatch idiom from
# plans/LESSONS.md "Closed Enum + Strategy Dispatch".)
# --------------------------------------------------------------
SUBCOMMANDS: tuple[str, ...] = (
    "run",
    "explain",
    "validate",
    "visualize",
    "meta",
    "monitor",
)


def _add_run_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``run`` subcommand parser.

    ``run <target>`` accepts:
      * a path to an FSM JSON file (``*.json``) → FSM interactive mode
        (delegates to :func:`fsm_llm.runner.main`).
      * a factory string ``pkg.mod:factory`` → term mode (Program.from_factory),
        prints the factory's ``Program.run(**env)`` result. Env values
        come from ``--env key=value`` flags (string-typed).
    """
    p = subparsers.add_parser(
        "run",
        help="Run an FSM JSON conversation or evaluate a stdlib factory term.",
        description=(
            "Run target. Target can be a path to an FSM JSON file (interactive "
            "conversation) or a factory string of the form 'pkg.mod:factory_name' "
            "(one-shot Term evaluation via Program.from_factory)."
        ),
    )
    p.add_argument(
        "target",
        help="FSM JSON path (e.g. examples/basic/echo_bot/echo_bot.json) "
        "or factory string (e.g. fsm_llm.stdlib.long_context:niah).",
    )
    p.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Env binding for term/factory mode (repeatable). String-typed.",
    )
    p.add_argument(
        "--factory-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Keyword argument forwarded to the factory call (repeatable). "
        "Best-effort: integer/float-coerced when possible, else string.",
    )
    p.add_argument(
        "--history-size",
        type=int,
        default=None,
        help="(FSM mode only) max conversation history size.",
    )
    p.add_argument(
        "--message-length",
        type=int,
        default=None,
        help="(FSM mode only) max message character length.",
    )
    p.add_argument(
        "--turns",
        type=int,
        default=None,
        help="(FSM mode only) cap conversation at N turns "
        "(for non-interactive smoke tests). Reads turns from stdin.",
    )


def _add_explain_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``explain`` subcommand parser."""
    p = subparsers.add_parser(
        "explain",
        help="Print Program.explain() output (Plan list, leaf schemas, AST shape).",
        description=(
            "Static description of the wrapped Term. Walks the AST and prints "
            "the indented node-kind skeleton plus per-Leaf schema map. Plans "
            "list is currently empty for FSM-mode programs (Theorem-2 cost "
            "model not yet wired for FSM; will land in a future R6 plan)."
        ),
    )
    p.add_argument(
        "target",
        help="FSM JSON path or factory string ('pkg.mod:factory').",
    )
    p.add_argument(
        "--factory-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Keyword argument forwarded to the factory call (repeatable).",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of human-readable indented output.",
    )
    p.add_argument(
        "--n",
        type=int,
        default=None,
        help=(
            "Input rank for the planner. When supplied with --K, "
            "Program.explain() returns one Plan per discovered Fix subtree. "
            "Omit to keep the R1 no-arg contract (plans=[])."
        ),
    )
    p.add_argument(
        "--K",
        type=int,
        default=None,
        help="Oracle context-window budget (tokens). See --n.",
    )
    p.add_argument(
        "--tau",
        type=int,
        default=None,
        help="(Optional) override planner tau.",
    )


def _add_validate_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``validate`` subcommand (delegates to validator.main)."""
    p = subparsers.add_parser(
        "validate",
        help="Validate an FSM definition file.",
        description="Lint an FSM JSON definition and emit a pass/fail report.",
    )
    p.add_argument(
        "--fsm", "-f", required=True, help="Path to FSM definition JSON file"
    )


def _add_visualize_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``visualize`` subcommand (delegates to visualizer.main)."""
    p = subparsers.add_parser(
        "visualize",
        help="Render an FSM as ASCII diagram.",
        description="Produce an ASCII diagram of an FSM definition.",
    )
    p.add_argument(
        "--fsm", "-f", required=True, help="Path to FSM definition JSON file"
    )
    p.add_argument(
        "--style",
        "-s",
        default="full",
        choices=["full", "compact", "minimal"],
        help="Visualization style (default: full)",
    )


def _add_meta_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``meta`` subcommand (delegates to stdlib.agents.meta_cli)."""
    # The meta CLI builds its own parser internally; we accept --help here
    # but otherwise pass through. We register a minimal parser that simply
    # captures ``rest`` so usage prints sensibly under ``fsm-llm meta -h``.
    p = subparsers.add_parser(
        "meta",
        help="Interactive meta-builder (FSM/agent/workflow artifact builder).",
        description="Delegate to fsm_llm.stdlib.agents.meta_cli. Pass --help "
        "after the subcommand to see meta-builder-specific options.",
        add_help=False,  # meta_cli has its own --help
    )
    p.add_argument("rest", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)


def _add_monitor_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``monitor`` subcommand (delegates to fsm_llm_monitor)."""
    p = subparsers.add_parser(
        "monitor",
        help="Launch the FSM-LLM monitor web dashboard.",
        description="Delegate to fsm_llm_monitor.__main__. Pass --help after "
        "the subcommand to see monitor-specific options.",
        add_help=False,  # monitor has its own --help
    )
    p.add_argument("rest", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fsm-llm",
        description=(
            f"FSM-LLM unified CLI v{__version__}. Run / explain / validate / "
            f"visualize / meta / monitor for both FSM JSON and λ-DSL targets."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"fsm_llm v{__version__}",
    )
    sub = parser.add_subparsers(
        dest="subcommand",
        metavar="{run,explain,validate,visualize,meta,monitor}",
        required=True,
    )
    _add_run_subparser(sub)
    _add_explain_subparser(sub)
    _add_validate_subparser(sub)
    _add_visualize_subparser(sub)
    _add_meta_subparser(sub)
    _add_monitor_subparser(sub)
    return parser


def _dispatch(args: argparse.Namespace, raw_argv: Sequence[str]) -> int:
    """Strategy dispatch over SUBCOMMANDS.

    Each branch is intentionally minimal — heavy lifting lives in the
    delegated module. Adding a subcommand requires touching SUBCOMMANDS,
    one ``_add_<name>_subparser`` function, and one branch here.
    """
    name = args.subcommand
    if name == "run":
        from .run import run as _run

        return _run(args)
    if name == "explain":
        from .explain import explain as _explain

        return _explain(args)
    if name == "validate":
        from ..validator import main as _validate

        return int(_validate(fsm_path=args.fsm) or 0)
    if name == "visualize":
        from ..visualizer import main as _visualize

        return int(_visualize(fsm_path=args.fsm, style=args.style) or 0)
    if name == "meta":
        # meta_cli.main_cli reads sys.argv directly. We rebuild argv to
        # be ``[<prog>, *rest_after_subcommand]`` so the meta parser
        # behaves as if invoked directly via ``fsm-llm-meta``.
        from ..stdlib.agents.meta_cli import main_cli as _meta_main

        old_argv = sys.argv
        rest: list[str] = list(getattr(args, "rest", []) or [])
        try:
            sys.argv = ["fsm-llm-meta", *rest]
            _meta_main()
        finally:
            sys.argv = old_argv
        return 0
    if name == "monitor":
        from fsm_llm_monitor.__main__ import main_cli as _monitor_main

        old_argv = sys.argv
        rest = list(getattr(args, "rest", []) or [])
        try:
            sys.argv = ["fsm-llm-monitor", *rest]
            _monitor_main()
        finally:
            sys.argv = old_argv
        return 0
    # Defensive: argparse.required=True should make this unreachable.
    raise AssertionError(f"unknown subcommand: {name!r}")


def main_cli(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``fsm-llm`` console script.

    ``argv`` defaults to ``sys.argv[1:]``. Returns the exit code.
    Aliases (``fsm-llm-validate`` etc.) construct an argv list of
    ``["<subcmd>", *original_argv[1:]]`` and call :func:`main_cli` —
    see ``# DECISION D-PLAN-04``.
    """
    if argv is None:
        argv = sys.argv[1:]
    parser = _build_parser()
    args = parser.parse_args(list(argv))
    return _dispatch(args, list(argv))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main_cli())
