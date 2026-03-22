from __future__ import annotations

"""
CLI entry point for fsm_llm_monitor.

Usage:
    python -m fsm_llm_monitor [--host HOST] [--port PORT]
    fsm-llm-monitor [--host HOST] [--port PORT]
"""

import argparse
import sys
import webbrowser


def main_cli() -> None:
    """Main CLI entry point for fsm_llm_monitor."""
    parser = argparse.ArgumentParser(
        prog="fsm-llm-monitor",
        description="FSM-LLM Monitor — Web-based monitoring dashboard",
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
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8420,
        help="Port to bind to (default: 8420)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't auto-open the browser",
    )
    args = parser.parse_args()

    if args.version:
        from .__version__ import __version__

        print(f"fsm_llm_monitor {__version__}")
        sys.exit(0)

    if args.info:
        from .__version__ import __version__

        print(f"FSM-LLM Monitor v{__version__}")
        print("Web-based monitoring dashboard for FSM-LLM")
        print()
        print("Features:")
        print("  - Real-time FSM conversation monitoring")
        print("  - Agent and workflow execution tracking")
        print("  - Live log streaming with filters")
        print("  - FSM definition viewer with state graph")
        print("  - Settings CRUD")
        print()
        print("Requires: fastapi, uvicorn, jinja2")
        sys.exit(0)

    # Launch the web server
    try:
        import uvicorn

        from .server import app, configure
    except ImportError as e:
        print(f"Error: Could not import dependencies: {e}", file=sys.stderr)
        print(
            "Make sure deps are installed: pip install fsm-llm[monitor]",
            file=sys.stderr,
        )
        sys.exit(1)

    configure()

    url = f"http://{args.host}:{args.port}"
    print(f"FSM-LLM Monitor starting at {url}")
    print("Press Ctrl+C to stop")

    if not args.no_browser:
        import threading

        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main_cli()
