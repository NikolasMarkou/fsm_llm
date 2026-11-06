"""
CLI entry point for the meta-builder agent.

Usage:
    fsm-llm-meta --model gpt-4o-mini --output my_fsm.json
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from pydantic import ValidationError

from fsm_llm.cli_support import exit_main, report_error, run_cli
from fsm_llm.constants import (
    CLI_EXIT_FAILURE,
    CLI_EXIT_OK,
    DEFAULT_LLM_MODEL,
    ENV_LLM_MODEL,
)

from .constants import MetaDefaults
from .definitions import MetaBuilderConfig
from .exceptions import MetaBuilderError
from .meta_builder import MetaBuilderAgent
from .meta_output import format_summary, save_artifact

_PROG = "fsm-llm-meta"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=_PROG,
        description="Interactively build FSMs, Workflows, and Agents",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"LLM model to use (default: env {ENV_LLM_MODEL}, else {DEFAULT_LLM_MODEL})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path for the artifact JSON",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=f"LLM temperature (default: {MetaDefaults.TEMPERATURE})",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help=f"Maximum conversation turns (default: {MetaDefaults.MAX_TURNS})",
    )
    parser.set_defaults(func=_run)
    return parser


def _run(args: argparse.Namespace) -> int:
    """The session and its report: summary and artifact to stdout, the rest to stderr."""
    config_kwargs: dict = {}
    if args.model:
        config_kwargs["model"] = args.model
    if args.temperature is not None:
        config_kwargs["temperature"] = args.temperature
    if args.max_turns is not None:
        config_kwargs["max_turns"] = args.max_turns

    agent = MetaBuilderAgent(config=MetaBuilderConfig(**config_kwargs))
    result = agent.run_interactive()

    print("\n" + "=" * 60)
    print(format_summary(result))
    print("=" * 60)

    if not result.is_valid:
        report_error(_PROG, "artifact has validation errors; fix them and try again")
        return CLI_EXIT_FAILURE
    if result.artifact_json:
        if args.output:
            path = save_artifact(result.artifact, args.output)
            print(f"\nArtifact saved to: {path}")
        else:
            print("\nGenerated artifact JSON:")
            print(result.artifact_json)
    return CLI_EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``fsm-llm-meta`` and return its exit code (0, 1 or 130).

    Interface contract: a usage error or ``--help`` raises ``SystemExit``
    (argparse); a ``MetaBuilderError``, an invalid option value or an
    ``OSError`` is one stderr line and exit 1, never a traceback; unencodable
    text is written as ``\\uXXXX`` escapes (the standard streams are
    reconfigured for the rest of the process, as in ``fsm-llm-okf``).
    """
    return run_cli(
        _build_parser(), argv, prog=_PROG, errors=(MetaBuilderError, ValidationError)
    )


def main_cli() -> None:
    """Console-script entry point for ``fsm-llm-meta``."""
    exit_main(main)


if __name__ == "__main__":
    main_cli()
