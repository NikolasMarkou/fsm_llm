"""
Command-line interface for the reasoning engine.
Enhanced with better argument handling and output formatting.
"""
import sys
import json
import argparse
from typing import Optional

from llm_fsm.logging import logger
from .engine import ReasoningEngine
from .constants import Defaults, ReasoningType
from .utilities import get_available_reasoning_types
from .__version__ import __version__


def setup_parser() -> argparse.ArgumentParser:
    """Set up command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=f"LLM-FSM Reasoning Engine v{__version__}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
              %(prog)s "What is 2 + 2?"
              %(prog)s "Design a recommendation system" --type analytical
              %(prog)s "Explain why the sky is blue" --context '{"audience": "child"}'
              %(prog)s --list-types
        """
    )

    # Problem input
    parser.add_argument(
        "problem",
        nargs="?",
        help="Problem to solve (quote if contains spaces)"
    )

    # Optional arguments
    parser.add_argument(
        "--type", "-t",
        choices=[rt.value for rt in ReasoningType],
        help="Force a specific reasoning type"
    )

    parser.add_argument(
        "--context", "-c",
        type=str,
        help="Initial context as JSON string"
    )

    parser.add_argument(
        "--model", "-m",
        default=Defaults.MODEL,
        help=f"LLM model to use (default: {Defaults.MODEL})"
    )

    parser.add_argument(
        "--output", "-o",
        choices=["text", "json", "detailed"],
        default="text",
        help="Output format (default: text)"
    )

    parser.add_argument(
        "--save", "-s",
        type=str,
        help="Save results to file"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output (solution only)"
    )

    # Information commands
    parser.add_argument(
        "--list-types",
        action="store_true",
        help="List available reasoning types"
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}"
    )

    return parser


def parse_context(context_str: Optional[str]) -> dict:
    """Parse context from JSON string."""
    if not context_str:
        return {}

    try:
        return json.loads(context_str)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON context: {e}")
        sys.exit(1)


def format_output(solution: str, trace_info: dict, output_format: str, quiet: bool) -> str:
    """Format output based on selected format."""
    if quiet:
        return solution

    if output_format == "json":
        return json.dumps({
            "solution": solution,
            "confidence": trace_info["reasoning_trace"]["final_confidence"],
            "steps": trace_info["reasoning_trace"]["total_steps"],
            "reasoning_types": trace_info["reasoning_trace"]["reasoning_types_used"]
        }, indent=2)

    elif output_format == "detailed":
        lines = [
            "=" * 60,
            "SOLUTION:",
            "=" * 60,
            solution,
            "",
            "=" * 60,
            "REASONING DETAILS:",
            "=" * 60,
            trace_info["summary"],
            "",
            "Trace Steps:"
        ]

        for i, step in enumerate(trace_info["reasoning_trace"]["steps"][:10]):
            lines.append(f"  {i+1}. {step['from']} → {step['to']}")

        if len(trace_info["reasoning_trace"]["steps"]) > 10:
            lines.append(f"  ... ({len(trace_info['reasoning_trace']['steps']) - 10} more steps)")

        return "\n".join(lines)

    else:  # text format
        lines = [
            f"Solution: {solution}",
            "",
            trace_info["summary"]
        ]
        return "\n".join(lines)


def list_reasoning_types():
    """Display available reasoning types."""
    logger.info("Available Reasoning Types:")
    logger.info("=" * 60)

    for type_name, description in get_available_reasoning_types().items():
        logger.info(f"  {type_name:<20} - {description}")

    logger.info("=" * 60)


def main():
    """Main CLI entry point."""
    parser = setup_parser()
    args = parser.parse_args()

    # Handle information commands
    if args.list_types:
        list_reasoning_types()
        return 0

    # Check for problem
    if not args.problem:
        parser.error("Problem statement is required")

    # Parse context
    initial_context = parse_context(args.context)

    # Override reasoning type if specified
    if args.type:
        initial_context["preferred_reasoning_type"] = args.type

    try:
        # Initialize engine
        if not args.quiet:
            logger.info(f"Initializing reasoning engine with model: {args.model}")

        engine = ReasoningEngine(model=args.model)

        # Solve problem
        if not args.quiet:
            logger.info("Solving problem...")
            logger.info("-" * 60)

        solution, trace_info = engine.solve_problem(args.problem, initial_context)

        # Format and display output
        output = format_output(solution, trace_info, args.output, args.quiet)
        logger.info(output)

        logger.info(f"The answer to the question is: {solution}")

        # Save if requested
        if args.save:
            with open(args.save, 'w') as f:
                if args.output == "json" or args.save.endswith('.json'):
                    json.dump({
                        "problem": args.problem,
                        "solution": solution,
                        "trace": trace_info
                    }, f, indent=2)
                else:
                    f.write(output)

            if not args.quiet:
                logger.info(f"\nResults saved to: {args.save}")

        return 0

    except KeyboardInterrupt:
        logger.info("\n\nInterrupted by user")
        return 1

    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())