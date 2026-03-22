from __future__ import annotations

"""
FastAPI web server for fsm_llm_monitor.

Serves the retro 90s dashboard UI and provides REST + WebSocket APIs
for real-time monitoring of FSM conversations, agents, and workflows.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from .bridge import MonitorBridge
from .definitions import MonitorConfig

STATIC_DIR = Path(__file__).parent / "static"
TEMPLATE_DIR = Path(__file__).parent / "templates"

app = FastAPI(title="FSM-LLM Monitor", docs_url="/api/docs")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Global bridge instance — set via configure()
_bridge: MonitorBridge | None = None


def configure(bridge: MonitorBridge | None = None) -> None:
    """Configure the global bridge for the web server."""
    global _bridge
    _bridge = bridge or MonitorBridge()


def get_bridge() -> MonitorBridge:
    global _bridge
    if _bridge is None:
        _bridge = MonitorBridge()
    return _bridge


# --- HTML Pages ---


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


# --- REST API ---


@app.get("/api/metrics")
async def api_metrics() -> dict[str, Any]:
    bridge = get_bridge()
    metrics = bridge.get_metrics()
    return metrics.model_dump()


@app.get("/api/conversations")
async def api_conversations() -> list[dict[str, Any]]:
    bridge = get_bridge()
    snapshots = bridge.get_all_conversation_snapshots()
    return [s.model_dump() for s in snapshots]


@app.get("/api/conversations/{conversation_id}")
async def api_conversation(conversation_id: str) -> dict[str, Any] | None:
    bridge = get_bridge()
    snap = bridge.get_conversation_snapshot(conversation_id)
    if snap is None:
        return {"error": "not found"}
    return snap.model_dump()


@app.get("/api/events")
async def api_events(limit: int = 50) -> list[dict[str, Any]]:
    bridge = get_bridge()
    events = bridge.get_recent_events(limit=limit)
    return [e.model_dump() for e in events]


@app.get("/api/logs")
async def api_logs(limit: int = 100, level: str = "INFO") -> list[dict[str, Any]]:
    bridge = get_bridge()
    logs = bridge.collector.get_logs(limit=limit, level=level)
    return [r.model_dump() for r in logs]


@app.get("/api/fsm/load")
async def api_fsm_load(path: str) -> dict[str, Any]:
    bridge = get_bridge()
    snap = bridge.load_fsm_from_file(path)
    if snap is None:
        return {"error": "failed to load"}
    return snap.model_dump()


@app.get("/api/config")
async def api_config_get() -> dict[str, Any]:
    bridge = get_bridge()
    return bridge.config.model_dump()


@app.post("/api/config")
async def api_config_set(config: MonitorConfig) -> dict[str, str]:
    bridge = get_bridge()
    bridge.config = config
    return {"status": "ok"}


@app.get("/api/info")
async def api_info() -> dict[str, str]:
    from .__version__ import __version__

    info = {"monitor_version": __version__}
    try:
        from fsm_llm import __version__ as cv

        info["fsm_llm_version"] = cv
    except ImportError:
        pass
    return info


# --- WebSocket for real-time updates ---


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    bridge = get_bridge()
    last_event_count = 0

    try:
        while True:
            await asyncio.sleep(1.0)
            metrics = bridge.get_metrics()
            current_count = metrics.total_events

            # Send metrics update
            data: dict[str, Any] = {
                "type": "metrics",
                "data": metrics.model_dump(),
            }

            # Include new events if any
            if current_count > last_event_count:
                new_count = min(current_count - last_event_count, 50)
                events = bridge.get_recent_events(limit=new_count)
                data["events"] = [e.model_dump() for e in events]
                last_event_count = current_count

            # Serialize datetime objects
            await websocket.send_text(
                json.dumps(data, default=str)
            )
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
