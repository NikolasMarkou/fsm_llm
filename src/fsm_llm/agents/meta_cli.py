"""
CLI entry point for the meta-builder agent.

Usage:
    fsm-llm-meta --model gpt-4o-mini --output my_fsm.json
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, cast

from pydantic import ValidationError

from fsm_llm.agents.constants import MetaDefaults
from fsm_llm.cli_support import (
    add_env_file,
    apply_env_file,
    exit_main,
    report_error,
    run_cli,
)
from fsm_llm.constants import (
    CLI_EXIT_FAILURE,
    CLI_EXIT_OK,
    ENV_LLM_MODEL,
)
from fsm_llm.core.definitions import ModelNotConfiguredError

from .definitions import MetaBuilderConfig
from .exceptions import MetaBuilderError
from .meta_builder import MetaBuilderAgent
from .meta_output import format_summary, save_artifact

if TYPE_CHECKING:
    from fsm_llm.engine.executables import Executable

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
        help=f"LLM model to use (default: env {ENV_LLM_MODEL}; there is no built-in model)",
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
    parser.add_argument(
        "--executables",
        type=Path,
        metavar="DIR",
        help="directory of pattern documents (*.json): the id of each is a valid "
        "agent type beside the built-in ones (a file named like a built-in is refused; "
        "the files are read as operator input, not as model output)",
    )
    add_env_file(parser)
    parser.set_defaults(func=_run)
    return parser


def _user_agent_types(directory: Path | None) -> list[str]:
    """The ids of the pattern files of ``--executables DIR`` (none for no directory).

    Contract: ``FileNotFoundError`` for a directory that does not exist (one stderr line, exit 1);
    each refused file is one stderr warning (``load_directory_registry``); a built-in id is never
    listed (the files cannot replace one).
    """
    if directory is None:
        return []
    from fsm_llm.agents.executables import builtin_executables
    from fsm_llm.engine.executables.directory import load_directory_registry

    builtins = builtin_executables()
    # the read-only attributes of PatternExecutable satisfy the protocol (as FsmBodyExecutable does)
    registry = load_directory_registry(
        directory, cast("list[Executable]", builtins), prog=_PROG
    )
    known = {executable.id for executable in builtins}
    return [e.id for e in registry.list(kind="pattern") if e.id not in known]


def _run(args: argparse.Namespace) -> int:
    """The session and its report: summary and artifact to stdout, the rest to stderr."""
    apply_env_file(
        args
    )  # before the model is resolved (a missing file: OSError, exit 1)
    config_kwargs: dict = {}
    if args.model:
        config_kwargs["model"] = args.model
    if args.temperature is not None:
        config_kwargs["temperature"] = args.temperature
    if args.max_turns is not None:
        config_kwargs["max_turns"] = args.max_turns

    agent_types = _user_agent_types(args.executables)
    agent = MetaBuilderAgent(
        config=MetaBuilderConfig(**config_kwargs),
        **({"agent_types": agent_types} if agent_types else {}),
    )
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
    (argparse); a ``MetaBuilderError``, ``ModelNotConfiguredError`` (no model set),
    an invalid option value or an ``OSError`` is one stderr line and exit 1, never a traceback; unencodable
    text is written as ``\\uXXXX`` escapes (the standard streams are
    reconfigured for the rest of the process, as in ``fsm-llm-okf``).
    """
    return run_cli(
        _build_parser(),
        argv,
        prog=_PROG,
        errors=(MetaBuilderError, ModelNotConfiguredError, ValidationError),
    )


def main_cli() -> None:
    """Console-script entry point for ``fsm-llm-meta``."""
    exit_main(main)


if __name__ == "__main__":
    main_cli()
