from __future__ import annotations

"""
FastAPI web server for fsm_llm_monitor.

Serves the Grafana-inspired dark dashboard UI and provides REST + WebSocket APIs
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

# Flow data loaded from static/flows.json
_flows: dict[str, Any] = {}


def _load_flows() -> dict[str, Any]:
    """Load agent/workflow flow definitions from the JSON data file."""
    flows_path = STATIC_DIR / "flows.json"
    if flows_path.exists():
        return json.loads(flows_path.read_text())
    return {"agents": {}, "workflows": {}}


def configure(bridge: MonitorBridge | None = None) -> None:
    """Configure the global bridge for the web server."""
    global _bridge, _flows
    _bridge = bridge or MonitorBridge()
    _flows = _load_flows()


def get_bridge() -> MonitorBridge:
    global _bridge, _flows
    if _bridge is None:
        _bridge = MonitorBridge()
    if not _flows:
        _flows = _load_flows()
    return _bridge


# --- HTML Pages ---


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html")


# --- REST API: Monitoring ---


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

    info: dict[str, str] = {"monitor_version": __version__}
    try:
        from fsm_llm import __version__ as cv

        info["fsm_llm_version"] = cv
    except ImportError:
        pass
    return info


# --- REST API: FSM Visualization ---


@app.post("/api/fsm/load")
async def api_fsm_load(request: Request) -> dict[str, Any]:
    """Load FSM definition from JSON body."""
    try:
        data = await request.json()
    except Exception:
        return {"error": "invalid JSON"}
    bridge = get_bridge()
    snap = bridge.load_fsm_from_dict(data)
    if snap is None:
        return {"error": "failed to parse FSM definition"}
    return snap.model_dump()


def _fsm_snapshot_to_viz(snap: Any) -> dict[str, Any]:
    """Convert an FSMSnapshot to visualization data (nodes + edges)."""
    nodes = []
    edges = []
    for i, state in enumerate(snap.states):
        nodes.append(
            {
                "id": state.state_id,
                "label": state.state_id,
                "description": state.description,
                "purpose": state.purpose,
                "is_initial": state.is_initial,
                "is_terminal": state.is_terminal,
                "x": 150 + (i % 4) * 200,
                "y": 80 + (i // 4) * 140,
            }
        )
        for t in state.transitions:
            edges.append(
                {
                    "from": state.state_id,
                    "to": t.target_state,
                    "label": t.description[:30] if t.description else "",
                    "priority": t.priority,
                }
            )
    return {"fsm": snap.model_dump(), "nodes": nodes, "edges": edges}


@app.post("/api/fsm/visualize")
async def api_fsm_visualize(request: Request) -> dict[str, Any]:
    """Accept FSM JSON definition and return visualization data."""
    try:
        data = await request.json()
    except Exception:
        return {"error": "invalid JSON"}
    bridge = get_bridge()
    snap = bridge.load_fsm_from_dict(data)
    if snap is None:
        return {"error": "failed to parse FSM definition"}
    return _fsm_snapshot_to_viz(snap)


@app.get("/api/fsm/visualize/preset/{preset_id:path}")
async def api_fsm_visualize_preset(preset_id: str) -> dict[str, Any]:
    """Load an FSM preset by ID and return visualization data."""
    base = _find_examples_dir()
    if base is None:
        return {"error": "examples directory not found"}
    if ".." in preset_id or preset_id.startswith("/"):
        return {"error": "invalid preset ID"}
    file_path = base / preset_id
    if not file_path.exists() or not file_path.is_file():
        return {"error": "preset not found"}
    try:
        file_path.resolve().relative_to(base.resolve())
    except ValueError:
        return {"error": "invalid preset ID"}
    try:
        data = json.loads(file_path.read_text())
    except Exception:
        return {"error": "failed to read preset"}
    bridge = get_bridge()
    snap = bridge.load_fsm_from_dict(data)
    if snap is None:
        return {"error": "failed to parse FSM definition"}
    return _fsm_snapshot_to_viz(snap)


# --- REST API: Presets ---


def _find_examples_dir() -> Path | None:
    """Locate the examples/ directory without exposing paths to clients."""
    base = Path(__file__).parent.parent.parent / "examples"
    if not base.exists():
        base = base.parent / "examples"
    return base if base.exists() else None


@app.get("/api/presets")
async def api_presets() -> dict[str, list[dict[str, str]]]:
    """Scan examples/ directory for FSM presets.

    Returns preset metadata only — no filesystem paths exposed.
    """
    base = _find_examples_dir()
    if base is None:
        return {"fsm": []}

    fsm_presets: list[dict[str, str]] = []
    for category in [
        "basic",
        "intermediate",
        "advanced",
        "classification",
        "reasoning",
    ]:
        cat_dir = base / category
        if not cat_dir.exists():
            continue
        for example_dir in sorted(cat_dir.iterdir()):
            if not example_dir.is_dir():
                continue
            fsm_files = list(example_dir.glob("*.json"))
            for f in fsm_files:
                name = example_dir.name.replace("_", " ").title()
                preset_id = f"{category}/{example_dir.name}/{f.name}"
                try:
                    data = json.loads(f.read_text())
                    desc = data.get("description", "")
                except Exception:
                    desc = ""
                fsm_presets.append(
                    {
                        "name": f"{name} ({f.name})",
                        "id": preset_id,
                        "category": category,
                        "description": desc,
                    }
                )

    return {"fsm": fsm_presets}


@app.get("/api/preset/fsm/{preset_id:path}")
async def api_preset_fsm(preset_id: str) -> dict[str, Any]:
    """Load an FSM preset by ID and return its JSON content."""
    base = _find_examples_dir()
    if base is None:
        return {"error": "examples directory not found"}
    if ".." in preset_id or preset_id.startswith("/"):
        return {"error": "invalid preset ID"}
    file_path = base / preset_id
    if not file_path.exists() or not file_path.is_file():
        return {"error": "preset not found"}
    try:
        file_path.resolve().relative_to(base.resolve())
    except ValueError:
        return {"error": "invalid preset ID"}
    try:
        return json.loads(file_path.read_text())
    except Exception:
        return {"error": "failed to read preset"}


# --- REST API: Pattern Visualization (from flows.json) ---


@app.get("/api/agent/visualize")
async def api_agent_visualize(agent_type: str = "ReactAgent") -> dict[str, Any]:
    """Return visualization data for an agent pattern flow."""
    agents = _flows.get("agents", {})
    flow = agents.get(agent_type)
    if flow is None:
        return {"error": f"unknown agent type: {agent_type}"}

    nodes = []
    for i, node in enumerate(flow["nodes"]):
        nodes.append(
            {
                **node,
                "x": 150 + (i % 4) * 200,
                "y": 80 + (i // 4) * 140,
            }
        )

    return {
        "info": {
            "name": agent_type,
            "description": flow["description"],
            "state_count": len(flow["nodes"]),
        },
        "nodes": nodes,
        "edges": flow["edges"],
        "agent_types": list(agents.keys()),
    }


@app.get("/api/workflow/visualize")
async def api_workflow_visualize(
    workflow_id: str = "order_processing",
) -> dict[str, Any]:
    """Return visualization data for a workflow pattern."""
    workflows = _flows.get("workflows", {})
    flow = workflows.get(workflow_id)
    if flow is None:
        return {"error": f"unknown workflow: {workflow_id}"}

    nodes = []
    for i, node in enumerate(flow["nodes"]):
        nodes.append(
            {
                **node,
                "x": 150 + (i % 4) * 200,
                "y": 80 + (i // 4) * 140,
            }
        )

    return {
        "info": {
            "name": flow["name"],
            "description": flow["description"],
            "step_count": len(flow["nodes"]),
        },
        "nodes": nodes,
        "edges": flow["edges"],
        "workflow_ids": list(workflows.keys()),
    }


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

            data: dict[str, Any] = {
                "type": "metrics",
                "data": metrics.model_dump(),
            }

            if current_count > last_event_count:
                new_count = min(current_count - last_event_count, 50)
                events = bridge.get_recent_events(limit=new_count)
                data["events"] = [e.model_dump() for e in events]
                last_event_count = current_count

            await websocket.send_text(json.dumps(data, default=str))
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
