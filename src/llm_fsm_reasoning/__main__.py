"""
Example usage of the refactored reasoning engine.
"""
import sys
import json
import argparse

# --------------------------------------------------------------
# local imports
# --------------------------------------------------------------

from llm_fsm.logging import logger
from .engine import ReasoningEngine
from .__version__ import __version__

# --------------------------------------------------------------


def main_cli():
    # Initialize the reasoning engine
    engine = ReasoningEngine(model="gpt-4o-mini", temperature=0.7)

    parser = (
        argparse.ArgumentParser(
            description=f"Reasoning engine using FSM-LLM v{__version__}"
        )
    )
    parser.add_argument(
        "--problem", "-p",
        type=str,
        required=True,
        help="Clearly define the problem you want to solve"
    )

    parser.add_argument(
        "--version", "-v",
        action="store_true",
        help="Output version information"
    )
    args = parser.parse_args()

    if args.version:
        print(f"llm_fsm v{__version__}")
        return 0


    try:
        # Solve the problem
        solution, trace_info = engine.solve_problem(
            args.problem,
            initial_context={}
        )

        # Display results
        logger.info(f"Solution:\n{solution[:500]}...")
        logger.info(f"Reasoning Details:")
        logger.info(f"- Total steps: {trace_info['reasoning_trace']['total_steps']}")
        logger.info(f"- Reasoning types used: {', '.join(trace_info['reasoning_trace']['reasoning_types_used'])}")
        logger.info(f"- Final confidence: {trace_info['reasoning_trace']['final_confidence']:.2f}")
    except Exception as e:
        logger.error(f"Error solving problem {args.problem}: \n{e}")
        return -1

    return 0

# --------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(main_cli())

# --------------------------------------------------------------
