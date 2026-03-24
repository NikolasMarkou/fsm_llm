from __future__ import annotations

"""
FastAPI web server for fsm_llm_monitor.

Serves the Grafana-inspired dark dashboard UI and provides REST + WebSocket APIs
for real-time monitoring, launching, and controlling FSM conversations, agents,
and workflows.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from fsm_llm.logging import logger

from .bridge import MonitorBridge
from .definitions import (
    EndConversationRequest,
    LaunchAgentRequest,
    LaunchFSMRequest,
    LaunchWorkflowRequest,
    MonitorConfig,
    SendMessageRequest,
    StartConversationRequest,
    WorkflowAdvanceRequest,
    WorkflowCancelRequest,
)
from .instance_manager import InstanceManager

STATIC_DIR = Path(__file__).parent / "static"
TEMPLATE_DIR = Path(__file__).parent / "templates"

app = FastAPI(title="FSM-LLM Monitor", docs_url="/api/docs")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Global instance manager — set via configure()
_manager: InstanceManager | None = None

# Flow data loaded from static/flows.json
_flows: dict[str, Any] = {}


def _load_flows() -> dict[str, Any]:
    """Load agent/workflow flow definitions from the JSON data file."""
    flows_path = STATIC_DIR / "flows.json"
    if flows_path.exists():
        result: dict[str, Any] = json.loads(flows_path.read_text())
        return result
    return {"agents": {}, "workflows": {}}


def configure(
    bridge: MonitorBridge | None = None,
    manager: InstanceManager | None = None,
) -> None:
    """Configure the global instance manager for the web server.

    Accepts either a MonitorBridge (backward compat) or an InstanceManager.
    If a bridge is provided, an InstanceManager is created wrapping it.
    """
    global _manager, _flows
    _flows = _load_flows()

    if manager is not None:
        _manager = manager
    elif bridge is not None:
        _manager = InstanceManager(config=bridge.config)
        if bridge._api is not None:
            _manager.connect_bridge(bridge._api)
    else:
        _manager = InstanceManager()


def get_manager() -> InstanceManager:
    global _manager, _flows
    if _manager is None:
        _manager = InstanceManager()
    if not _flows:
        _flows = _load_flows()
    return _manager


# Backward compatibility alias
def get_bridge() -> MonitorBridge:
    """Backward compat: returns a MonitorBridge-compatible wrapper."""
    mgr = get_manager()
    bridge = MonitorBridge(config=mgr.config)
    bridge._collector = mgr.global_collector
    return bridge


# --- HTML Pages ---


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html")


# --- REST API: Monitoring ---


@app.get("/api/metrics")
async def api_metrics() -> dict[str, Any]:
    mgr = get_manager()
    metrics = mgr.get_metrics()
    return metrics.model_dump()


@app.get("/api/conversations")
async def api_conversations(
    include_ended: bool = True,
) -> list[dict[str, Any]]:
    mgr = get_manager()
    snapshots = mgr.get_all_conversation_snapshots(include_ended=include_ended)
    return [s.model_dump() for s in snapshots]


@app.get("/api/conversations/{conversation_id}")
async def api_conversation(conversation_id: str) -> dict[str, Any] | None:
    mgr = get_manager()
    snap = mgr.get_conversation_snapshot(conversation_id)
    if snap is None:
        raise HTTPException(status_code=404, detail="not found")
    return snap.model_dump()


@app.get("/api/events")
async def api_events(limit: int = 50) -> list[dict[str, Any]]:
    mgr = get_manager()
    events = mgr.get_events(limit=limit)
    return [e.model_dump() for e in events]


@app.get("/api/logs")
async def api_logs(limit: int = 100, level: str = "INFO") -> list[dict[str, Any]]:
    mgr = get_manager()
    logs = mgr.global_collector.get_logs(limit=limit, level=level)
    return [r.model_dump() for r in logs]


@app.get("/api/config")
async def api_config_get() -> dict[str, Any]:
    mgr = get_manager()
    return mgr.config.model_dump()


@app.post("/api/config")
async def api_config_set(config: MonitorConfig) -> dict[str, str]:
    mgr = get_manager()
    mgr.config = config
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


# --- REST API: Capabilities ---


@app.get("/api/capabilities")
async def api_capabilities() -> dict[str, bool]:
    """Return which extensions are installed."""
    mgr = get_manager()
    return mgr.get_capabilities()


# --- REST API: Instance Management ---


@app.get("/api/instances")
async def api_instances(type: str | None = None) -> list[dict[str, Any]]:
    """List all managed instances."""
    mgr = get_manager()
    instances = mgr.list_instances(type_filter=type)
    return [i.model_dump() for i in instances]


@app.get("/api/instances/{instance_id}")
async def api_instance_detail(instance_id: str) -> dict[str, Any]:
    """Get detailed info for a managed instance."""
    mgr = get_manager()
    inst = mgr.get_instance(instance_id)
    if inst is None:
        raise HTTPException(status_code=404, detail="instance not found")
    return inst.to_info().model_dump()


@app.get("/api/instances/{instance_id}/events")
async def api_instance_events(
    instance_id: str, limit: int = 50
) -> list[dict[str, Any]]:
    """Get events for a specific managed instance."""
    mgr = get_manager()
    collector = mgr.get_instance_collector(instance_id)
    if collector is None:
        return []
    events = collector.get_events(limit=limit)
    return [e.model_dump() for e in events]


@app.delete("/api/instances/{instance_id}")
async def api_instance_destroy(instance_id: str) -> dict[str, str]:
    """Destroy a managed instance."""
    mgr = get_manager()
    try:
        mgr.destroy_instance(instance_id)
        return {"status": "ok"}
    except KeyError as e:
        raise HTTPException(status_code=404, detail="instance not found") from e


# --- REST API: FSM Launch/Control ---


@app.post("/api/fsm/launch")
async def api_fsm_launch(req: LaunchFSMRequest) -> dict[str, Any]:
    """Launch a new FSM instance."""
    mgr = get_manager()
    try:
        managed = mgr.launch_fsm(
            preset_id=req.preset_id,
            fsm_json=req.fsm_json,
            model=req.model,
            temperature=req.temperature,
            label=req.label,
        )
        return managed.to_info().model_dump()
    except Exception as e:
        logger.error(f"Failed to launch FSM: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.post("/api/fsm/{instance_id}/start")
async def api_fsm_start_conversation(
    instance_id: str, req: StartConversationRequest
) -> dict[str, Any]:
    """Start a new conversation on a managed FSM."""
    mgr = get_manager()
    try:
        conv_id, response = await asyncio.to_thread(
            mgr.start_conversation, instance_id, req.initial_context
        )
        return {"conversation_id": conv_id, "response": response}
    except Exception as e:
        logger.error(f"Failed to start conversation on instance {instance_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.post("/api/fsm/{instance_id}/converse")
async def api_fsm_converse(instance_id: str, req: SendMessageRequest) -> dict[str, Any]:
    """Send a message to an FSM conversation."""
    mgr = get_manager()
    try:
        result = await asyncio.to_thread(
            mgr.send_message, instance_id, req.conversation_id, req.message
        )
        return result
    except Exception as e:
        logger.error(f"Failed to send message to instance {instance_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.post("/api/fsm/{instance_id}/end")
async def api_fsm_end_conversation(
    instance_id: str, req: EndConversationRequest
) -> dict[str, str]:
    """End a conversation on a managed FSM."""
    mgr = get_manager()
    try:
        await asyncio.to_thread(mgr.end_conversation, instance_id, req.conversation_id)
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Failed to end conversation on instance {instance_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.get("/api/fsm/{instance_id}/conversations")
async def api_fsm_conversations(instance_id: str) -> list[dict[str, Any]]:
    """List conversations on a managed FSM instance."""
    mgr = get_manager()
    try:
        snapshots = mgr.get_fsm_conversations(instance_id)
        return [s.model_dump() for s in snapshots]
    except Exception as e:
        logger.error(f"Failed to get conversations for instance {instance_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


# --- REST API: Workflow Launch/Control ---


@app.post("/api/workflow/launch")
async def api_workflow_launch(req: LaunchWorkflowRequest) -> dict[str, Any]:
    """Launch a new workflow instance."""
    mgr = get_manager()
    try:
        managed = mgr.launch_workflow(
            preset_id=req.preset_id,
            definition_json=req.definition_json,
            initial_context=req.initial_context,
            label=req.label,
        )
        return managed.to_info().model_dump()
    except Exception as e:
        logger.error(f"Failed to launch workflow: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.post("/api/workflow/{instance_id}/advance")
async def api_workflow_advance(
    instance_id: str, req: WorkflowAdvanceRequest
) -> dict[str, Any]:
    """Advance a workflow instance."""
    mgr = get_manager()
    try:
        result = await mgr.advance_workflow(
            instance_id, req.workflow_instance_id, req.user_input
        )
        return {"advanced": result}
    except Exception as e:
        logger.error(f"Failed to advance workflow on instance {instance_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.post("/api/workflow/{instance_id}/cancel")
async def api_workflow_cancel(
    instance_id: str, req: WorkflowCancelRequest
) -> dict[str, Any]:
    """Cancel a workflow instance."""
    mgr = get_manager()
    try:
        result = await mgr.cancel_workflow(
            instance_id, req.workflow_instance_id, req.reason
        )
        return {"cancelled": result}
    except Exception as e:
        logger.error(f"Failed to cancel workflow on instance {instance_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.get("/api/workflow/{instance_id}/status")
async def api_workflow_status(
    instance_id: str, workflow_instance_id: str = ""
) -> dict[str, Any]:
    """Get workflow instance status."""
    mgr = get_manager()
    try:
        return mgr.get_workflow_status(instance_id, workflow_instance_id)
    except Exception as e:
        logger.error(f"Failed to get workflow status for instance {instance_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.get("/api/workflow/{instance_id}/instances")
async def api_workflow_instances(instance_id: str) -> list[dict[str, Any]]:
    """List all workflow instances on a managed workflow engine."""
    mgr = get_manager()
    try:
        return mgr.get_workflow_instances(instance_id)
    except Exception as e:
        logger.error(f"Failed to list workflow instances for {instance_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


# --- REST API: Agent Launch/Control ---


@app.post("/api/agent/launch")
async def api_agent_launch(req: LaunchAgentRequest) -> dict[str, Any]:
    """Launch an agent in a background thread."""
    mgr = get_manager()
    try:
        managed = mgr.launch_agent(
            agent_type=req.agent_type,
            task=req.task,
            tools_config=req.tools,
            model=req.model,
            max_iterations=req.max_iterations,
            timeout_seconds=req.timeout_seconds,
            label=req.label,
        )
        return managed.to_info().model_dump()
    except Exception as e:
        logger.error(f"Failed to launch agent: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.get("/api/agent/{instance_id}/status")
async def api_agent_status(instance_id: str) -> dict[str, Any]:
    """Get agent execution status."""
    mgr = get_manager()
    try:
        return mgr.get_agent_status(instance_id)
    except Exception as e:
        logger.error(f"Failed to get agent status for instance {instance_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.get("/api/agent/{instance_id}/result")
async def api_agent_result(instance_id: str) -> dict[str, Any]:
    """Get final agent result."""
    mgr = get_manager()
    try:
        return mgr.get_agent_result(instance_id)
    except Exception as e:
        logger.error(f"Failed to get agent result for instance {instance_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.post("/api/agent/{instance_id}/cancel")
async def api_agent_cancel(instance_id: str) -> dict[str, str]:
    """Cancel a running agent."""
    mgr = get_manager()
    try:
        mgr.cancel_agent(instance_id)
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Failed to cancel agent {instance_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


# --- REST API: FSM Visualization ---


@app.post("/api/fsm/load")
async def api_fsm_load(request: Request) -> dict[str, Any]:
    """Load FSM definition from JSON body."""
    try:
        data = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail="invalid JSON") from e
    bridge = get_bridge()
    snap = bridge.load_fsm_from_dict(data)
    if snap is None:
        raise HTTPException(status_code=400, detail="failed to parse FSM definition")
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
    except Exception as e:
        raise HTTPException(status_code=400, detail="invalid JSON") from e
    bridge = get_bridge()
    snap = bridge.load_fsm_from_dict(data)
    if snap is None:
        raise HTTPException(status_code=400, detail="failed to parse FSM definition")
    return _fsm_snapshot_to_viz(snap)


@app.get("/api/fsm/visualize/preset/{preset_id:path}")
async def api_fsm_visualize_preset(preset_id: str) -> dict[str, Any]:
    """Load an FSM preset by ID and return visualization data."""
    base = _find_examples_dir()
    if base is None:
        raise HTTPException(status_code=404, detail="examples directory not found")
    if ".." in preset_id or preset_id.startswith("/"):
        raise HTTPException(status_code=400, detail="invalid preset ID")
    file_path = base / preset_id
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="preset not found")
    try:
        file_path.resolve().relative_to(base.resolve())
    except ValueError as e:
        raise HTTPException(status_code=400, detail="invalid preset ID") from e
    try:
        data = json.loads(file_path.read_text())
    except Exception as e:
        raise HTTPException(status_code=500, detail="failed to read preset") from e
    bridge = get_bridge()
    snap = bridge.load_fsm_from_dict(data)
    if snap is None:
        raise HTTPException(status_code=400, detail="failed to parse FSM definition")
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
        raise HTTPException(status_code=404, detail="examples directory not found")
    if ".." in preset_id or preset_id.startswith("/"):
        raise HTTPException(status_code=400, detail="invalid preset ID")
    file_path = base / preset_id
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="preset not found")
    try:
        file_path.resolve().relative_to(base.resolve())
    except ValueError as e:
        raise HTTPException(status_code=400, detail="invalid preset ID") from e
    try:
        result: dict[str, Any] = json.loads(file_path.read_text())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail="failed to read preset") from e


# --- REST API: Pattern Visualization (from flows.json) ---


@app.get("/api/agent/visualize")
async def api_agent_visualize(agent_type: str = "ReactAgent") -> dict[str, Any]:
    """Return visualization data for an agent pattern flow."""
    agents = _flows.get("agents", {})
    flow = agents.get(agent_type)
    if flow is None:
        raise HTTPException(status_code=404, detail=f"unknown agent type: {agent_type}")

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
        raise HTTPException(status_code=404, detail=f"unknown workflow: {workflow_id}")

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
    mgr = get_manager()
    last_event_count = 0
    last_log_count = 0

    try:
        while True:
            await asyncio.sleep(1.0)
            metrics = mgr.get_metrics()
            current_count = metrics.total_events

            data: dict[str, Any] = {
                "type": "metrics",
                "data": metrics.model_dump(),
            }

            if current_count > last_event_count:
                new_count = min(current_count - last_event_count, 50)
                events = mgr.get_events(limit=new_count)
                data["events"] = [e.model_dump() for e in events]
                last_event_count = current_count

            # Push new log records
            current_log_count = mgr.global_collector.total_logs
            if current_log_count > last_log_count:
                new_log_count = min(current_log_count - last_log_count, 50)
                logs = mgr.global_collector.get_logs(limit=new_log_count)
                data["logs"] = [r.model_dump() for r in logs]
                data["log_count"] = current_log_count
                last_log_count = current_log_count

            # Always include instance list so status changes propagate
            instances = mgr.list_instances()
            data["instances"] = [i.model_dump() for i in instances]

            # Include real-time status for running agents
            running_agents = [
                i
                for i in instances
                if i.instance_type == "agent" and i.status == "running"
            ]
            if running_agents:
                data["agent_updates"] = {
                    i.instance_id: mgr.get_agent_status(i.instance_id)
                    for i in running_agents
                }

            await websocket.send_text(json.dumps(data, default=str))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug(f"WebSocket error: {e}")
        try:
            await websocket.close()
        except Exception:
            pass
