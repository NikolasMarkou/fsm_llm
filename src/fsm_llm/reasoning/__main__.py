"""
Command-line interface for the FSM-LLM reasoning engine.

Supports multiple output formats, reasoning type selection, and detailed tracing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fsm_llm.cli_support import (
    CLI_EXIT_INTERRUPTED,
    add_env_file,
    apply_env_file,
    escape_unencodable_output,
    report_error,
)
from fsm_llm.context.filtering import redacting_json_default
from fsm_llm.core.definitions import ModelNotConfiguredError
from fsm_llm.logging import logger, setup_cli_logging
from fsm_llm.text import capped_line

from .__version__ import __version__
from .constants import ContextKeys
from .engine import ReasoningEngine
from .executables import load_strategy_registry
from .utilities import get_available_reasoning_types

if TYPE_CHECKING:
    from fsm_llm.engine.executables import ExecutableRegistry

# ============================================================================
# CONSTANTS
# ============================================================================

_PROG = "python -m fsm_llm.reasoning"


class OutputFormat:
    """Available output formats."""

    TEXT = "text"
    JSON = "json"
    DETAILED = "detailed"

    @classmethod
    def choices(cls) -> list[str]:
        """Get all available output format choices."""
        return [cls.TEXT, cls.JSON, cls.DETAILED]


class CLIDefaults:
    """CLI-specific default values."""

    OUTPUT_FORMAT = OutputFormat.TEXT
    MAX_TRACE_STEPS_DISPLAY = 10
    SEPARATOR_LENGTH = 60
    SECTION_SEPARATOR = "=" * SEPARATOR_LENGTH
    SUBSECTION_SEPARATOR = "-" * SEPARATOR_LENGTH


# ============================================================================
# ARGUMENT PARSING
# ============================================================================


def setup_argument_parser() -> argparse.ArgumentParser:
    """
    Set up and configure the command-line argument parser.

    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description=f"FSM-LLM Reasoning Engine v{__version__}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "What is 2 + 2?"
  %(prog)s "Design a recommendation system" --type analytical
  %(prog)s "Explain why the sky is blue" --context '{"audience": "child"}'
  %(prog)s --list-types
  %(prog)s "Solve this logic puzzle" --output detailed --save results.json
  %(prog)s "Plan a launch" --executables my_strategies --type my_strategy

Run --list-types for the available reasoning types (with --executables DIR it also lists
the strategies found in DIR).
        """,
    )

    # Positional argument
    parser.add_argument(
        "problem",
        nargs="?",
        help="Problem statement to solve (use quotes if it contains spaces)",
    )

    # Reasoning configuration
    reasoning_group = parser.add_argument_group("reasoning options")
    reasoning_group.add_argument(
        "--type",
        "-t",
        metavar="TYPE",
        help="Force a specific reasoning type, by name (see --list-types for options)",
    )
    reasoning_group.add_argument(
        "--executables",
        type=Path,
        metavar="DIR",
        help="Directory of strategy documents (*.json) to choose from beside the "
        "built-in strategies",
    )
    reasoning_group.add_argument(
        "--context",
        "-c",
        type=str,
        metavar="JSON",
        help='Initial context as JSON string (e.g., \'{"key": "value"}\')',
    )

    # Model configuration
    model_group = parser.add_argument_group("model options")
    model_group.add_argument(
        "--model",
        "-m",
        default=None,
        metavar="MODEL",
        help="LLM model to use (default: env LLM_MODEL; there is no built-in model)",
    )
    add_env_file(parser)

    # Output configuration
    output_group = parser.add_argument_group("output options")
    output_group.add_argument(
        "--output",
        "-o",
        choices=OutputFormat.choices(),
        default=CLIDefaults.OUTPUT_FORMAT,
        metavar="FORMAT",
        help=f"Output format: {', '.join(OutputFormat.choices())} (default: {CLIDefaults.OUTPUT_FORMAT})",
    )
    output_group.add_argument(
        "--save",
        "-s",
        type=Path,
        metavar="FILE",
        help="Save results to file (format auto-detected from extension)",
    )
    output_group.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging and detailed output",
    )
    output_group.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Minimal output (solution only, no formatting)",
    )

    # Information commands
    info_group = parser.add_argument_group("information")
    info_group.add_argument(
        "--list-types",
        action="store_true",
        help="List all available reasoning types with descriptions",
    )
    info_group.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    """
    Validate command-line arguments for consistency and correctness.

    Args:
        args: Parsed command-line arguments

    Raises:
        argparse.ArgumentError: If arguments are invalid or inconsistent
    """
    # Check for conflicting output options
    if args.quiet and args.verbose:
        raise argparse.ArgumentError(
            None, "--quiet and --verbose are mutually exclusive"
        )

    # Validate save file extension if provided
    if args.save:
        valid_extensions = {".json", ".txt", ".md"}
        if args.save.suffix.lower() not in valid_extensions:
            logger.warning(
                f"Unusual file extension '{args.save.suffix}'. "
                f"Recommended: {', '.join(valid_extensions)}"
            )


# ============================================================================
# INPUT/OUTPUT HANDLING
# ============================================================================


def parse_context_json(context_str: str | None) -> dict[str, Any]:
    """
    Parse initial context from JSON string with comprehensive error handling.

    Args:
        context_str: JSON string containing initial context

    Returns:
        dict[str, Any]: Parsed context dictionary

    Raises:
        SystemExit: If JSON parsing fails
    """
    if not context_str:
        return {}

    try:
        context = json.loads(context_str)
        if not isinstance(context, dict):
            report_error(
                _PROG, "error: context must be a JSON object, not an array or a value"
            )
            sys.exit(1)
        return context
    except json.JSONDecodeError as e:
        report_error(
            _PROG,
            f"error: invalid JSON context: {e} "
            '(example: \'{"key": "value", "number": 42}\')',
        )
        sys.exit(1)


def format_solution_output(
    solution: str, trace_info: dict[str, Any], output_format: str, quiet: bool
) -> str:
    """
    Format the solution output according to the specified format.

    Args:
        solution: The final solution string
        trace_info: Dictionary containing reasoning trace information
        output_format: Output format (text, json, detailed)
        quiet: Whether to use minimal output

    Returns:
        str: Formatted output string
    """
    if quiet:
        return solution.strip()

    if output_format == OutputFormat.JSON:
        return _format_json_output(solution, trace_info)
    elif output_format == OutputFormat.DETAILED:
        return _format_detailed_output(solution, trace_info)
    else:  # text format
        return _format_text_output(solution, trace_info)


def _format_json_output(solution: str, trace_info: dict[str, Any]) -> str:
    """Format output as JSON."""
    trace = trace_info.get("reasoning_trace", {})

    output_data = {
        "solution": solution,
        "metadata": {
            "confidence": trace.get("final_confidence", "unknown"),
            "total_steps": trace.get("total_steps", 0),
            "reasoning_types_used": trace.get("reasoning_types_used", []),
            "execution_time_seconds": trace.get("execution_time_seconds", "unknown"),
        },
        "summary": trace_info.get("summary", "No summary available"),
    }

    return json.dumps(
        output_data, indent=2, ensure_ascii=False, default=redacting_json_default
    )


def _format_detailed_output(solution: str, trace_info: dict[str, Any]) -> str:
    """Format output with detailed reasoning trace."""
    trace = trace_info.get("reasoning_trace", {})
    steps = trace.get("steps", [])

    lines = [
        CLIDefaults.SECTION_SEPARATOR,
        "SOLUTION",
        CLIDefaults.SECTION_SEPARATOR,
        solution,
        "",
        CLIDefaults.SECTION_SEPARATOR,
        "REASONING DETAILS",
        CLIDefaults.SECTION_SEPARATOR,
        trace_info.get("summary", "No summary available"),
        "",
        "Reasoning Trace:",
    ]

    # Show trace steps (limited to avoid overwhelming output)
    for i, step in enumerate(steps[: CLIDefaults.MAX_TRACE_STEPS_DISPLAY], 1):
        step_desc = f"{step.get('from', '?')} → {step.get('to', '?')}"
        if "action" in step:
            step_desc += f" ({step['action']})"
        lines.append(f"  {i:2d}. {step_desc}")

    # Show truncation notice if needed
    remaining_steps = len(steps) - CLIDefaults.MAX_TRACE_STEPS_DISPLAY
    if remaining_steps > 0:
        lines.append(f"  ... ({remaining_steps} more steps)")

    return "\n".join(lines)


def _format_text_output(solution: str, trace_info: dict[str, Any]) -> str:
    """Format output as simple text."""
    return "\n".join(
        [
            f"Solution: {solution}",
            "",
            trace_info.get("summary", "No summary available"),
        ]
    )


def save_results_to_file(
    save_path: Path,
    problem: str,
    solution: str,
    trace_info: dict[str, Any],
    output_format: str,
) -> None:
    """
    Save results to the specified file with appropriate format.

    Args:
        save_path: Path where to save the results
        problem: Original problem statement
        solution: Generated solution
        trace_info: Reasoning trace information
        output_format: Format used for display output
    """
    # Create parent directories if they don't exist
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine save format based on file extension or output format
    if save_path.suffix.lower() == ".json" or output_format == OutputFormat.JSON:
        _save_as_json(save_path, problem, solution, trace_info)
    else:
        _save_as_text(save_path, problem, solution, trace_info, output_format)

    logger.info(f"Results saved to: {save_path}")


def _save_as_json(
    save_path: Path, problem: str, solution: str, trace_info: dict[str, Any]
) -> None:
    """Save results as JSON file."""
    data = {
        "problem": problem,
        "solution": solution,
        "trace_info": trace_info,
        "timestamp": trace_info.get("reasoning_trace", {}).get("timestamp", "unknown"),
    }

    with save_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=redacting_json_default)


def _save_as_text(
    save_path: Path,
    problem: str,
    solution: str,
    trace_info: dict[str, Any],
    output_format: str,
) -> None:
    """Save results as text file."""
    with save_path.open("w", encoding="utf-8") as f:
        f.write(f"Problem: {problem}\n")
        f.write(CLIDefaults.SECTION_SEPARATOR + "\n")
        f.write(
            format_solution_output(solution, trace_info, output_format, quiet=False)
        )


# ============================================================================
# INFORMATION COMMANDS
# ============================================================================


def display_available_reasoning_types(
    reasoning_types: dict[str, str] | None = None,
) -> None:
    """Print all available reasoning types with descriptions to stdout.

    ``reasoning_types`` is the name -> description table to print (``main`` passes the one of
    its registry); ``None`` is the nine built-ins.
    """
    # Product output: print(), never the logger (the `fsm_llm` disable would
    # drop it when main() is called in-process). Same rule as runner.py D-018.
    print("Available Reasoning Types")
    print(CLIDefaults.SECTION_SEPARATOR)

    if reasoning_types is None:
        reasoning_types = get_available_reasoning_types()

    # Calculate column width for alignment
    max_name_length = max(len(name) for name in reasoning_types.keys())

    for type_name, description in reasoning_types.items():
        print(f"  {type_name:<{max_name_length}} - {description}")

    print(CLIDefaults.SECTION_SEPARATOR)
    print(f"Total: {len(reasoning_types)} reasoning types available")


# ============================================================================
# MAIN EXECUTION LOGIC
# ============================================================================


def solve_problem_with_engine(
    problem: str,
    initial_context: dict[str, Any],
    model: str | None,
    verbose: bool,
    registry: ExecutableRegistry | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    Initialize the reasoning engine and solve the problem.

    The engine is closed on every way out (a solve, an error, Ctrl-C).

    Args:
        problem: Problem statement to solve
        initial_context: Initial context for reasoning
        model: LLM model to use; ``None`` reads env ``LLM_MODEL``
        verbose: Whether to enable verbose logging
        registry: The registry the solve chooses its strategy from (``None``: the built-ins)

    Returns:
        tuple[str, dict[str, Any]]: Solution and trace information

    Raises:
        Exception: If reasoning engine initialization or problem solving fails
    """
    # Any failure propagates: main() reports it as the one error line, never the
    # logger (its record carries a timestamp, D-053 of plan 51fde95b).
    # Initialize engine
    if verbose:
        logger.info(f"Initializing reasoning engine with model: {model}")

    with ReasoningEngine(model=model, registry=registry) as engine:
        # Solve problem
        if verbose:
            logger.info("Starting problem solving process...")
            logger.info(CLIDefaults.SUBSECTION_SEPARATOR)

        solution, trace_info = engine.solve_problem(problem, initial_context)

    if verbose:
        logger.info(CLIDefaults.SUBSECTION_SEPARATOR)
        logger.info("Problem solving completed successfully")

    return solution, trace_info


def load_registry(directory: Path | None) -> ExecutableRegistry | None:
    """The registry of ``--executables DIR``, or ``None`` when the option is not given.

    Contract: the nine built-ins plus the ``*.json`` strategy documents of *directory*
    (``executables.load_strategy_registry``); a built-in id in a file is refused, never
    replaced. Each file that could not be read and each collision is ONE stderr line
    ``python -m fsm_llm.reasoning: warning: <label>: <why>`` and the run goes on. Raises
    ``FileNotFoundError`` (one ``error:`` line, exit 1, in ``main``) when *directory* is not an
    existing directory.
    """
    if directory is None:
        return None
    if not directory.is_dir():
        raise FileNotFoundError(f"--executables {directory}: not an existing directory")
    registry, report = load_strategy_registry([directory])
    for label, reasons in report.failed.items():
        for reason in reasons:
            report_error(_PROG, f"warning: {label}: {capped_line(reason, 300)}")
    for collision in report.collisions:
        report_error(_PROG, f"warning: {capped_line(collision, 300)}")
    return registry


def main() -> int:
    """
    Main CLI entry point with comprehensive error handling.

    Returns:
        int: Exit code (0 for success, non-zero for error)
    """
    # An unencodable character is written as \uXXXX, never ends a finished solve.
    escape_unencodable_output()
    verbose = False  # argparse accepts a prefix (--verb): read the parsed flag
    try:
        # Parse and validate arguments
        parser = setup_argument_parser()
        args = parser.parse_args()
        verbose = args.verbose

        validate_arguments(args)
        apply_env_file(args)  # before the model is resolved

        if args.verbose:
            # DECISION plan-2026-09-29T044048-3a032517/D-016
            # --verbose must show the engine's own records, which the
            # library-wide disable of "fsm_llm" silences. Enable them only
            # here, on the flag. Do NOT enable logging unconditionally or in
            # solve_problem_with_engine: without --verbose the CLI stays quiet.
            setup_cli_logging("INFO")

        # The strategies of --executables are read first: --type and --list-types follow the
        # registry (the built-ins when there is none)
        registry = load_registry(args.executables)
        types = get_available_reasoning_types(registry=registry)
        if args.type is not None and args.type not in types:
            parser.error(
                f"argument --type/-t: invalid choice: {args.type!r} "
                f"(choose from {', '.join(repr(name) for name in types)})"
            )

        # Handle information commands
        if args.list_types:
            display_available_reasoning_types(types)
            return 0

        # Validate required arguments
        if not args.problem:
            parser.error("Problem statement is required (use --help for examples)")

        # Parse initial context
        initial_context = parse_context_json(args.context)

        # Override reasoning type if specified
        if args.type:
            initial_context[ContextKeys.PREFERRED_REASONING_TYPE] = args.type

        # Solve the problem
        solution, trace_info = solve_problem_with_engine(
            args.problem, initial_context, args.model, args.verbose, registry
        )

        # Format and display output
        output = format_solution_output(solution, trace_info, args.output, args.quiet)

        print(output)

        # Save results if requested
        if args.save:
            save_results_to_file(
                args.save, args.problem, solution, trace_info, args.output
            )

        return 0

    except KeyboardInterrupt:
        report_error(_PROG, "interrupted by user")
        return CLI_EXIT_INTERRUPTED

    except argparse.ArgumentError as e:
        report_error(_PROG, f"error: {e}")
        return 1

    except (ModelNotConfiguredError, FileNotFoundError) as e:
        # A configuration failure (no model, a missing --env-file), not a
        # crash: one stderr line, no traceback.
        report_error(_PROG, f"error: {e}")
        return 1

    except Exception as e:
        if verbose:
            report_error(_PROG, f"error: {e}")
            import traceback

            traceback.print_exc()
        else:  # one line, the hint inside it (D-053)
            report_error(
                _PROG, f"error: {e} (use --verbose for detailed error information)"
            )
        return 1


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    sys.exit(main())
