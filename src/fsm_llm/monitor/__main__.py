"""
CLI entry point for fsm_llm.monitor.

Usage:
    python -m fsm_llm.monitor [--host HOST] [--port PORT] [--api-key KEY] [--otel]
    fsm-llm-monitor [--host HOST] [--port PORT] [--api-key KEY] [--otel]
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time
import webbrowser

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
_WILDCARD_HOSTS = {"0.0.0.0", "::", ""}


def _browser_url(host: str, port: int) -> str:
    """URL a local browser can open for a bind address."""
    if host in _WILDCARD_HOSTS:
        host = "127.0.0.1"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{port}"


def _open_when_listening(url: str, host: str, port: int, timeout: float = 15.0) -> None:
    """Open the browser once the port accepts connections (never if the
    server failed to bind)."""
    probe = "127.0.0.1" if host in _WILDCARD_HOSTS else host
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((probe, port), timeout=0.5):
                webbrowser.open(url)
                return
        except OSError:
            time.sleep(0.2)


def main_cli() -> None:
    """Main CLI entry point for fsm_llm.monitor."""
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
    parser.add_argument(
        "--api-key",
        default=None,
        help=(
            "Require this key for mutating and sensitive routes "
            "(default: FSM_LLM_MONITOR_API_KEY env var)"
        ),
    )
    parser.add_argument(
        "--otel",
        action="store_true",
        help="Export monitor events as OpenTelemetry spans (console exporter)",
    )
    args = parser.parse_args()

    if args.version:
        from .__version__ import __version__

        print(f"fsm_llm.monitor {__version__}")
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

    # DECISION plan-2026-09-29T044048-3a032517/D-008
    # The single library-wide loguru disable of "fsm_llm" also silences this
    # subpackage and every other one, so the dashboard Logs page would stay
    # empty. The CLI entry point owns logging config (like runner/validator/
    # visualizer), so enable it here. Do NOT move this into InstanceManager or
    # server import: a library object must not flip global loguru state.
    from fsm_llm.logging import enable_library_logging

    enable_library_logging()

    api_key = args.api_key or os.environ.get("FSM_LLM_MONITOR_API_KEY") or None
    non_loopback = args.host not in _LOOPBACK_HOSTS
    # On a LAN/wildcard bind the Host header is the machine's address, which
    # the localhost-only allow-list would refuse.
    trusted_hosts = None
    if args.host in _WILDCARD_HOSTS:
        trusted_hosts = ["*"]
    elif non_loopback:
        trusted_hosts = ["localhost", "127.0.0.1", "::1", args.host]
    configure(api_key=api_key, trusted_hosts=trusted_hosts)

    if non_loopback and api_key is None:
        print(
            f"WARNING: binding to {args.host} without an API key: anyone who can "
            "reach this port can launch agents and read conversations. Pass "
            "--api-key or set FSM_LLM_MONITOR_API_KEY.",
            file=sys.stderr,
        )

    if args.otel:
        from .otel import OTELExporter
        from .server import get_manager

        exporter = OTELExporter(service_name="fsm-llm-monitor")
        exporter.enable(get_manager().global_collector)

    url = _browser_url(args.host, args.port)
    print(f"FSM-LLM Monitor starting at {url}")
    print("Press Ctrl+C to stop")

    if not args.no_browser:
        threading.Thread(
            target=_open_when_listening,
            args=(url, args.host, args.port),
            daemon=True,
        ).start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main_cli()
