from __future__ import annotations

"""
CLI entry point for fsm_llm_monitor.

Usage:
    python -m fsm_llm_monitor [--help]
"""

import argparse
import sys


def main_cli() -> None:
    """Main CLI entry point for fsm_llm_monitor."""
    parser = argparse.ArgumentParser(
        prog="fsm-llm-monitor",
        description="FSM-LLM Monitor — Terminal-based monitoring dashboard",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version and exit",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show monitor info and exit",
    )
    args = parser.parse_args()

    if args.version:
        from .__version__ import __version__

        print(f"fsm_llm_monitor {__version__}")
        sys.exit(0)

    if args.info:
        from .__version__ import __version__

        print(f"FSM-LLM Monitor v{__version__}")
        print("Terminal-based monitoring dashboard for FSM-LLM")
        print()
        print("Features:")
        print("  - Real-time FSM conversation monitoring")
        print("  - Agent and workflow execution tracking")
        print("  - Live log streaming with filters")
        print("  - FSM definition viewer with state graph")
        print("  - Settings CRUD")
        print()
        print("Requires: textual")
        sys.exit(0)

    # Launch the TUI app
    try:
        from .app import MonitorApp
    except ImportError as e:
        print(f"Error: Could not import monitor app: {e}", file=sys.stderr)
        print("Make sure textual is installed: pip install fsm-llm[monitor]", file=sys.stderr)
        sys.exit(1)

    app = MonitorApp()
    app.run()


if __name__ == "__main__":
    main_cli()
