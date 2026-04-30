from __future__ import annotations

"""
CLI entry point for fsm_llm.stdlib.agents.

Usage:
    python -m fsm_llm.stdlib.agents --list-tools
    python -m fsm_llm.stdlib.agents --version
"""

import argparse
import sys

from .__version__ import __version__


def setup_argument_parser() -> argparse.ArgumentParser:
    """Configure the argument parser."""
    parser = argparse.ArgumentParser(
        prog="fsm_llm.stdlib.agents",
        description="FSM-LLM Agentic Patterns: ReAct and HITL agents",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"fsm_llm.stdlib.agents {__version__}",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show package information",
    )
    return parser


def main() -> None:
    """CLI entry point."""
    parser = setup_argument_parser()
    args = parser.parse_args()

    if args.info:
        print(f"fsm_llm.stdlib.agents v{__version__}")
        print("Agentic patterns for FSM-LLM:")
        print("  - ReactAgent: ReAct (Reasoning + Acting) loop with tool use")
        print("  - HumanInTheLoop: Approval gates, escalation, and overrides")
        print("  - ToolRegistry: Tool management with schema and execution")
        print()
        print("Usage:")
        print("  from fsm_llm.stdlib.agents import ReactAgent, ToolRegistry")
        print()
        print("  registry = ToolRegistry()")
        print(
            '  registry.register_function(my_tool, name="search", description="Search the web")'
        )
        print("  agent = ReactAgent(tools=registry)")
        print('  result = agent.run("What is the weather in Paris?")')
        print("  print(result.answer)")
        sys.exit(0)

    parser.print_help()


if __name__ == "__main__":
    main()
