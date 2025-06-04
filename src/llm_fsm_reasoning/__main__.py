"""
Command-line interface for the LLM-FSM reasoning engine.

This module provides a comprehensive CLI for interacting with the reasoning engine,
supporting multiple output formats, reasoning type selection, and detailed tracing.

Author: LLM-FSM Reasoning Engine
Python Version: 3.11+
Dependencies: llm-fsm, argparse, json
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from llm_fsm.logging import logger
from .engine import ReasoningEngine
from .constants import Defaults, ReasoningType
from .utilities import get_available_reasoning_types
from .__version__ import __version__


# ============================================================================
# CONSTANTS
# ============================================================================

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
        description=f"LLM-FSM Reasoning Engine v{__version__}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  %(prog)s "What is 2 + 2?"
  %(prog)s "Design a recommendation system" --type analytical
  %(prog)s "Explain why the sky is blue" --context '{{"audience": "child"}}'
  %(prog)s --list-types
  %(prog)s "Solve this logic puzzle" --output detailed --save results.json

Available reasoning types: {', '.join([rt.value for rt in ReasoningType])}
        """
    )

    # Positional argument
    parser.add_argument(
        "problem",
        nargs="?",
        help="Problem statement to solve (use quotes if it contains spaces)"
    )

    # Reasoning configuration
    reasoning_group = parser.add_argument_group("reasoning options")
    reasoning_group.add_argument(
        "--type", "-t",
        choices=[rt.value for rt in ReasoningType],
        metavar="TYPE",
        help="Force a specific reasoning type (see --list-types for options)"
    )
    reasoning_group.add_argument(
        "--context", "-c",
        type=str,
        metavar="JSON",
        help="Initial context as JSON string (e.g., '{\"key\": \"value\"}')"
    )

    # Model configuration
    model_group = parser.add_argument_group("model options")
    model_group.add_argument(
        "--model", "-m",
        default=Defaults.MODEL,
        metavar="MODEL",
        help=f"LLM model to use (default: {Defaults.MODEL})"
    )

    # Output configuration
    output_group = parser.add_argument_group("output options")
    output_group.add_argument(
        "--output", "-o",
        choices=OutputFormat.choices(),
        default=CLIDefaults.OUTPUT_FORMAT,
        metavar="FORMAT",
        help=f"Output format: {', '.join(OutputFormat.choices())} (default: {CLIDefaults.OUTPUT_FORMAT})"
    )
    output_group.add_argument(
        "--save", "-s",
        type=Path,
        metavar="FILE",
        help="Save results to file (format auto-detected from extension)"
    )
    output_group.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging and detailed output"
    )
    output_group.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output (solution only, no formatting)"
    )

    # Information commands
    info_group = parser.add_argument_group("information")
    info_group.add_argument(
        "--list-types",
        action="store_true",
        help="List all available reasoning types with descriptions"
    )
    info_group.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}"
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
        raise argparse.ArgumentError(None, "--quiet and --verbose are mutually exclusive")

    # Validate save file extension if provided
    if args.save:
        valid_extensions = {'.json', '.txt', '.md'}
        if args.save.suffix.lower() not in valid_extensions:
            logger.warning(f"Unusual file extension '{args.save.suffix}'. "
                         f"Recommended: {', '.join(valid_extensions)}")


# ============================================================================
# INPUT/OUTPUT HANDLING
# ============================================================================

def parse_context_json(context_str: Optional[str]) -> Dict[str, Any]:
    """
    Parse initial context from JSON string with comprehensive error handling.

    Args:
        context_str: JSON string containing initial context

    Returns:
        Dict[str, Any]: Parsed context dictionary

    Raises:
        SystemExit: If JSON parsing fails
    """
    if not context_str:
        return {}

    try:
        context = json.loads(context_str)
        if not isinstance(context, dict):
            logger.error("Context must be a JSON object (dictionary), not array or primitive")
            sys.exit(1)
        return context
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON context: {e}")
        logger.error(f"Provided: {context_str}")
        logger.error("Example: '{\"key\": \"value\", \"number\": 42}'")
        sys.exit(1)


def format_solution_output(
    solution: str,
    trace_info: Dict[str, Any],
    output_format: str,
    quiet: bool
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


def _format_json_output(solution: str, trace_info: Dict[str, Any]) -> str:
    """Format output as JSON."""
    trace = trace_info.get("reasoning_trace", {})

    output_data = {
        "solution": solution,
        "metadata": {
            "confidence": trace.get("final_confidence", "unknown"),
            "total_steps": trace.get("total_steps", 0),
            "reasoning_types_used": trace.get("reasoning_types_used", []),
            "execution_time": trace.get("execution_time", "unknown")
        },
        "summary": trace_info.get("summary", "No summary available")
    }

    return json.dumps(output_data, indent=2, ensure_ascii=False)


def _format_detailed_output(solution: str, trace_info: Dict[str, Any]) -> str:
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
        "Reasoning Trace:"
    ]

    # Show trace steps (limited to avoid overwhelming output)
    for i, step in enumerate(steps[:CLIDefaults.MAX_TRACE_STEPS_DISPLAY], 1):
        step_desc = f"{step.get('from', '?')} → {step.get('to', '?')}"
        if 'action' in step:
            step_desc += f" ({step['action']})"
        lines.append(f"  {i:2d}. {step_desc}")

    # Show truncation notice if needed
    remaining_steps = len(steps) - CLIDefaults.MAX_TRACE_STEPS_DISPLAY
    if remaining_steps > 0:
        lines.append(f"  ... ({remaining_steps} more steps)")

    # Add metadata
    lines.extend([
        "",
        "Metadata:",
        f"  • Confidence: {trace.get('final_confidence', 'unknown')}",
        f"  • Total Steps: {trace.get('total_steps', 0)}",
        f"  • Reasoning Types: {', '.join(trace.get('reasoning_types_used', []))}"
    ])

    return "\n".join(lines)


def _format_text_output(solution: str, trace_info: Dict[str, Any]) -> str:
    """Format output as simple text."""
    lines = [
        f"Solution: {solution}",
        "",
        trace_info.get("summary", "No summary available")
    ]

    trace = trace_info.get("reasoning_trace", {})
    if "final_confidence" in trace:
        lines.append(f"Confidence: {trace['final_confidence']}")

    return "\n".join(lines)


def save_results_to_file(
    save_path: Path,
    problem: str,
    solution: str,
    trace_info: Dict[str, Any],
    output_format: str
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
    try:
        # Create parent directories if they don't exist
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine save format based on file extension or output format
        if save_path.suffix.lower() == '.json' or output_format == OutputFormat.JSON:
            _save_as_json(save_path, problem, solution, trace_info)
        else:
            _save_as_text(save_path, problem, solution, trace_info, output_format)

        logger.info(f"Results saved to: {save_path}")

    except Exception as e:
        logger.error(f"Failed to save results to {save_path}: {e}")
        raise


def _save_as_json(save_path: Path, problem: str, solution: str, trace_info: Dict[str, Any]) -> None:
    """Save results as JSON file."""
    data = {
        "problem": problem,
        "solution": solution,
        "trace_info": trace_info,
        "timestamp": trace_info.get("reasoning_trace", {}).get("timestamp", "unknown")
    }

    with save_path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _save_as_text(
    save_path: Path,
    problem: str,
    solution: str,
    trace_info: Dict[str, Any],
    output_format: str
) -> None:
    """Save results as text file."""
    with save_path.open('w', encoding='utf-8') as f:
        f.write(f"Problem: {problem}\n")
        f.write(CLIDefaults.SECTION_SEPARATOR + "\n")
        f.write(format_solution_output(solution, trace_info, output_format, quiet=False))


# ============================================================================
# INFORMATION COMMANDS
# ============================================================================

def display_available_reasoning_types() -> None:
    """Display all available reasoning types with descriptions."""
    logger.info("Available Reasoning Types")
    logger.info(CLIDefaults.SECTION_SEPARATOR)

    reasoning_types = get_available_reasoning_types()

    # Calculate column width for alignment
    max_name_length = max(len(name) for name in reasoning_types.keys())

    for type_name, description in reasoning_types.items():
        logger.info(f"  {type_name:<{max_name_length}} - {description}")

    logger.info(CLIDefaults.SECTION_SEPARATOR)
    logger.info(f"Total: {len(reasoning_types)} reasoning types available")


# ============================================================================
# MAIN EXECUTION LOGIC
# ============================================================================

def solve_problem_with_engine(
    problem: str,
    initial_context: Dict[str, Any],
    model: str,
    verbose: bool
) -> Tuple[str, Dict[str, Any]]:
    """
    Initialize the reasoning engine and solve the problem.

    Args:
        problem: Problem statement to solve
        initial_context: Initial context for reasoning
        model: LLM model to use
        verbose: Whether to enable verbose logging

    Returns:
        Tuple[str, Dict[str, Any]]: Solution and trace information

    Raises:
        Exception: If reasoning engine initialization or problem solving fails
    """
    try:
        # Initialize engine
        if verbose:
            logger.info(f"Initializing reasoning engine with model: {model}")

        engine = ReasoningEngine(model=model)

        # Solve problem
        if verbose:
            logger.info("Starting problem solving process...")
            logger.info(CLIDefaults.SUBSECTION_SEPARATOR)

        solution, trace_info = engine.solve_problem(problem, initial_context)

        if verbose:
            logger.info(CLIDefaults.SUBSECTION_SEPARATOR)
            logger.info("Problem solving completed successfully")

        return solution, trace_info

    except Exception as e:
        logger.error(f"Failed to solve problem: {e}")
        raise


def main() -> int:
    """
    Main CLI entry point with comprehensive error handling.

    Returns:
        int: Exit code (0 for success, non-zero for error)
    """
    try:
        # Parse and validate arguments
        parser = setup_argument_parser()
        args = parser.parse_args()

        validate_arguments(args)

        # Handle information commands
        if args.list_types:
            display_available_reasoning_types()
            return 0

        # Validate required arguments
        if not args.problem:
            parser.error("Problem statement is required (use --help for examples)")

        # Parse initial context
        initial_context = parse_context_json(args.context)

        # Override reasoning type if specified
        if args.type:
            initial_context["preferred_reasoning_type"] = args.type

        # Solve the problem
        solution, trace_info = solve_problem_with_engine(
            args.problem,
            initial_context,
            args.model,
            args.verbose
        )

        # Format and display output
        output = format_solution_output(solution, trace_info, args.output, args.quiet)

        if not args.quiet:
            print(output)
        else:
            print(solution)

        # Save results if requested
        if args.save:
            save_results_to_file(args.save, args.problem, solution, trace_info, args.output)

        return 0

    except KeyboardInterrupt:
        logger.info("\nOperation interrupted by user")
        return 1

    except argparse.ArgumentError as e:
        logger.error(f"Argument error: {e}")
        return 1

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if '--verbose' in sys.argv or '-v' in sys.argv:
            import traceback
            logger.error("Full traceback:")
            traceback.print_exc()
        else:
            logger.error("Use --verbose for detailed error information")
        return 1


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    sys.exit(main())