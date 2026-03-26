from __future__ import annotations

"""
FastAPI web server for fsm_llm_monitor.

Serves the Grafana-inspired dark dashboard UI and provides REST + WebSocket APIs
for real-time monitoring, launching, and controlling FSM conversations, agents,
and workflows.
"""

import asyncio
import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from fsm_llm.logging import logger

from .bridge import MonitorBridge
from .definitions import (
    BuilderSendRequest,
    BuilderStartRequest,
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
from .instance_manager import InstanceManager, _find_examples_dir, validate_preset_id

STATIC_DIR = Path(__file__).parent / "static"
TEMPLATE_DIR = Path(__file__).parent / "templates"

# Default timeout for synchronous FSM/LLM operations wrapped in asyncio.to_thread
_LLM_OPERATION_TIMEOUT = 120.0


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage server startup/shutdown lifecycle."""
    yield
    # Shutdown: remove loguru sink to prevent resource leaks
    if _manager is not None:
        _manager.global_collector.cleanup()


# CORS origins — defaults to localhost only for security. Override via
# configure(cors_origins=["*"]) or the MONITOR_CORS_ORIGINS env var.
_CORS_ORIGINS: list[str] = [
    "http://localhost:8420",
    "http://127.0.0.1:8420",
]
_CORS_ORIGIN_REGEX: str | None = r"https?://(localhost|127\.0\.0\.1)(:\d+)?"

app = FastAPI(title="FSM-LLM Monitor", docs_url="/api/docs", lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_origin_regex=_CORS_ORIGIN_REGEX,
    allow_methods=["*"],
    allow_headers=["*"],
)
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
    cors_origins: list[str] | None = None,
) -> None:
    """Configure the global instance manager for the web server.

    Accepts either a MonitorBridge (backward compat) or an InstanceManager.
    If a bridge is provided, an InstanceManager is created wrapping it.

    :param cors_origins: List of allowed CORS origins. Defaults to localhost only.
        Pass ``["*"]`` to allow all origins (not recommended for production).
    """
    global _manager, _flows, _bridge_cache, _CORS_ORIGINS, _CORS_ORIGIN_REGEX
    if cors_origins is not None:
        _CORS_ORIGINS[:] = cors_origins
        _CORS_ORIGIN_REGEX = None  # Disable regex when explicit origins are provided
        # Update the middleware stack. user_middleware is only effective before
        # the first request builds the middleware stack, so this must be called
        # before the server starts handling requests.
        for middleware in app.user_middleware:
            if hasattr(middleware, "kwargs"):
                if "allow_origins" in middleware.kwargs:
                    middleware.kwargs["allow_origins"] = cors_origins
                if "allow_origin_regex" in middleware.kwargs:
                    middleware.kwargs["allow_origin_regex"] = None
    _flows = _load_flows()
    _bridge_cache = None  # Reset cached bridge

    # Clean up old manager's loguru sink to prevent accumulation
    if _manager is not None:
        _manager.global_collector.cleanup()

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


# Backward compatibility alias — cached to avoid creating detached instances
_bridge_cache: MonitorBridge | None = None


def get_bridge() -> MonitorBridge:
    """Backward compat: returns a MonitorBridge-compatible wrapper."""
    global _bridge_cache
    mgr = get_manager()
    if _bridge_cache is None or _bridge_cache._collector is not mgr.global_collector:
        _bridge_cache = MonitorBridge(config=mgr.config)
        _bridge_cache._collector = mgr.global_collector
    return _bridge_cache


# --- HTML Pages ---


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


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
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/fsm/{instance_id}/start")
async def api_fsm_start_conversation(
    instance_id: str, req: StartConversationRequest
) -> dict[str, Any]:
    """Start a new conversation on a managed FSM."""
    mgr = get_manager()
    try:
        conv_id, response = await asyncio.wait_for(
            asyncio.to_thread(mgr.start_conversation, instance_id, req.initial_context),
            timeout=_LLM_OPERATION_TIMEOUT,
        )
        return {"conversation_id": conv_id, "response": response}
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="LLM operation timed out") from None
    except Exception as e:
        logger.error(f"Failed to start conversation on instance {instance_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/fsm/{instance_id}/converse")
async def api_fsm_converse(instance_id: str, req: SendMessageRequest) -> dict[str, Any]:
    """Send a message to an FSM conversation."""
    mgr = get_manager()
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                mgr.send_message, instance_id, req.conversation_id, req.message
            ),
            timeout=_LLM_OPERATION_TIMEOUT,
        )
        return result
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="LLM operation timed out") from None
    except Exception as e:
        logger.error(f"Failed to send message to instance {instance_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/fsm/{instance_id}/end")
async def api_fsm_end_conversation(
    instance_id: str, req: EndConversationRequest
) -> dict[str, str]:
    """End a conversation on a managed FSM."""
    mgr = get_manager()
    try:
        await asyncio.wait_for(
            asyncio.to_thread(mgr.end_conversation, instance_id, req.conversation_id),
            timeout=_LLM_OPERATION_TIMEOUT,
        )
        return {"status": "ok"}
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="LLM operation timed out") from None
    except Exception as e:
        logger.error(f"Failed to end conversation on instance {instance_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/fsm/{instance_id}/conversations")
async def api_fsm_conversations(instance_id: str) -> list[dict[str, Any]]:
    """List conversations on a managed FSM instance."""
    mgr = get_manager()
    try:
        snapshots = mgr.get_fsm_conversations(instance_id)
        return [s.model_dump() for s in snapshots]
    except Exception as e:
        logger.error(f"Failed to get conversations for instance {instance_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


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
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/workflow/{instance_id}/advance")
async def api_workflow_advance(
    instance_id: str, req: WorkflowAdvanceRequest
) -> dict[str, Any]:
    """Advance a workflow instance."""
    mgr = get_manager()
    try:
        result = await asyncio.wait_for(
            mgr.advance_workflow(instance_id, req.workflow_instance_id, req.user_input),
            timeout=_LLM_OPERATION_TIMEOUT,
        )
        return {"advanced": result}
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="Workflow advance timed out",
        ) from None
    except Exception as e:
        logger.error(f"Failed to advance workflow on instance {instance_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/workflow/{instance_id}/cancel")
async def api_workflow_cancel(
    instance_id: str, req: WorkflowCancelRequest
) -> dict[str, Any]:
    """Cancel a workflow instance."""
    mgr = get_manager()
    try:
        result = await asyncio.wait_for(
            mgr.cancel_workflow(instance_id, req.workflow_instance_id, req.reason),
            timeout=_LLM_OPERATION_TIMEOUT,
        )
        return {"cancelled": result}
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="Workflow cancel timed out",
        ) from None
    except Exception as e:
        logger.error(f"Failed to cancel workflow on instance {instance_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/workflow/{instance_id}/status")
async def api_workflow_status(
    instance_id: str, workflow_instance_id: str = ""
) -> dict[str, Any]:
    """Get workflow instance status."""
    if not workflow_instance_id:
        raise HTTPException(
            status_code=400,
            detail="workflow_instance_id query parameter is required",
        )
    mgr = get_manager()
    try:
        return mgr.get_workflow_status(instance_id, workflow_instance_id)
    except Exception as e:
        logger.error(f"Failed to get workflow status for instance {instance_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/workflow/{instance_id}/instances")
async def api_workflow_instances(instance_id: str) -> list[dict[str, Any]]:
    """List all workflow instances on a managed workflow engine."""
    mgr = get_manager()
    try:
        return mgr.get_workflow_instances(instance_id)
    except Exception as e:
        logger.error(f"Failed to list workflow instances for {instance_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


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
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/agent/{instance_id}/status")
async def api_agent_status(instance_id: str) -> dict[str, Any]:
    """Get agent execution status."""
    mgr = get_manager()
    try:
        return mgr.get_agent_status(instance_id)
    except Exception as e:
        logger.error(f"Failed to get agent status for instance {instance_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/agent/{instance_id}/result")
async def api_agent_result(instance_id: str) -> dict[str, Any]:
    """Get final agent result."""
    mgr = get_manager()
    try:
        return mgr.get_agent_result(instance_id)
    except Exception as e:
        logger.error(f"Failed to get agent result for instance {instance_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/agent/{instance_id}/cancel")
async def api_agent_cancel(instance_id: str) -> dict[str, str]:
    """Cancel a running agent."""
    mgr = get_manager()
    try:
        mgr.cancel_agent(instance_id)
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Failed to cancel agent {instance_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


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


def _resolve_preset_path(preset_id: str) -> Path:
    """Validate and resolve a preset ID to a filesystem path.

    Raises HTTPException on invalid/missing preset.
    """
    base = _find_examples_dir()
    if base is None:
        raise HTTPException(status_code=404, detail="examples directory not found")
    try:
        return validate_preset_id(preset_id, base)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="invalid preset ID") from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail="preset not found") from e


def _read_preset_json(preset_id: str) -> dict[str, Any]:
    """Load a preset JSON file by ID with path traversal protection."""
    file_path = _resolve_preset_path(preset_id)
    try:
        result: dict[str, Any] = json.loads(file_path.read_text())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail="failed to read preset") from e


@app.get("/api/fsm/visualize/preset/{preset_id:path}")
async def api_fsm_visualize_preset(preset_id: str) -> dict[str, Any]:
    """Load an FSM preset by ID and return visualization data."""
    data = _read_preset_json(preset_id)
    bridge = get_bridge()
    snap = bridge.load_fsm_from_dict(data)
    if snap is None:
        raise HTTPException(status_code=400, detail="failed to parse FSM definition")
    return _fsm_snapshot_to_viz(snap)


# --- REST API: Presets ---


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
                parse_error = ""
                try:
                    data = json.loads(f.read_text())
                    desc = data.get("description", "")
                except Exception as parse_err:
                    logger.debug(f"Failed to parse preset {f}: {parse_err}")
                    desc = ""
                    parse_error = str(parse_err)
                preset_entry: dict[str, str] = {
                    "name": f"{name} ({f.name})",
                    "id": preset_id,
                    "category": category,
                    "description": desc,
                }
                if parse_error:
                    preset_entry["parse_error"] = parse_error
                fsm_presets.append(preset_entry)

    return {"fsm": fsm_presets}


@app.get("/api/preset/fsm/{preset_id:path}")
async def api_preset_fsm(preset_id: str) -> dict[str, Any]:
    """Load an FSM preset by ID and return its JSON content."""
    return _read_preset_json(preset_id)


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


# --- Builder (Meta-Agent) ---

# Session store: session_id -> (agent, created_at_timestamp)
_builder_sessions: dict[str, tuple[Any, float]] = {}
_BUILDER_SESSION_TTL = 3600.0  # 1 hour


def _cleanup_stale_builder_sessions() -> None:
    """Remove builder sessions older than TTL."""
    import time

    now = time.time()
    stale = [
        sid
        for sid, (_, ts) in _builder_sessions.items()
        if now - ts > _BUILDER_SESSION_TTL
    ]
    for sid in stale:
        _builder_sessions.pop(sid, None)
        logger.debug(f"Cleaned up stale builder session: {sid}")


@app.post("/api/builder/start")
async def api_builder_start(req: BuilderStartRequest) -> dict[str, Any]:
    """Start a new builder session using the meta-agent."""
    _cleanup_stale_builder_sessions()
    try:
        from fsm_llm_meta import MetaAgent, MetaAgentConfig
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="fsm_llm_meta package not installed",
        ) from None

    config_kwargs: dict[str, Any] = {}
    if req.model:
        config_kwargs["model"] = req.model
    if req.temperature is not None:
        config_kwargs["temperature"] = req.temperature
    if req.max_tokens:
        config_kwargs["max_tokens"] = req.max_tokens

    config = MetaAgentConfig(**config_kwargs)
    agent = MetaAgent(config=config)

    try:
        initial_message = req.artifact_type or ""
        response = await asyncio.wait_for(
            asyncio.to_thread(agent.start, initial_message),
            timeout=_LLM_OPERATION_TIMEOUT,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="LLM operation timed out") from None
    except Exception as e:
        logger.error(f"Builder start failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

    import time
    import uuid

    session_id = f"builder-{uuid.uuid4().hex[:8]}"
    _builder_sessions[session_id] = (agent, time.time())

    return {
        "session_id": session_id,
        "response": response,
        "is_complete": agent.is_complete(),
    }


@app.post("/api/builder/send")
async def api_builder_send(req: BuilderSendRequest) -> dict[str, Any]:
    """Send a message to an existing builder session."""
    entry = _builder_sessions.get(req.session_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Builder session not found")
    agent = entry[0]

    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(agent.send, req.message),
            timeout=_LLM_OPERATION_TIMEOUT,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="LLM operation timed out") from None
    except Exception as e:
        logger.error(f"Builder send failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

    result: dict[str, Any] = {
        "response": response,
        "is_complete": agent.is_complete(),
    }

    if agent.is_complete():
        try:
            build_result = agent.get_result()
            result["artifact"] = build_result.artifact
            result["artifact_json"] = build_result.artifact_json
            result["artifact_type"] = build_result.artifact_type.value
            result["is_valid"] = build_result.is_valid
            result["validation_errors"] = build_result.validation_errors
        except Exception as e:
            logger.error(f"Builder result extraction failed: {e}")
            result["artifact"] = {}
            result["artifact_json"] = "{}"
            result["artifact_type"] = "unknown"
            result["is_valid"] = False
            result["validation_errors"] = [f"Result extraction failed: {e}"]
            result["error"] = str(e)

        # Clean up session
        _builder_sessions.pop(req.session_id, None)

    return result


@app.get("/api/builder/result/{session_id}")
async def api_builder_result(session_id: str) -> dict[str, Any]:
    """Get the current state of a builder session."""
    entry = _builder_sessions.get(session_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Builder session not found")
    agent = entry[0]

    result: dict[str, Any] = {
        "session_id": session_id,
        "is_complete": agent.is_complete(),
    }

    if agent.is_complete():
        try:
            build_result = agent.get_result()
            result["artifact"] = build_result.artifact
            result["artifact_json"] = build_result.artifact_json
            result["artifact_type"] = build_result.artifact_type.value
            result["is_valid"] = build_result.is_valid
        except Exception as e:
            result["error"] = str(e)

    return result


@app.delete("/api/builder/{session_id}")
async def api_builder_delete(session_id: str) -> dict[str, Any]:
    """Delete a builder session."""
    removed = _builder_sessions.pop(session_id, None)
    return {"deleted": removed is not None}


# --- WebSocket for real-time updates ---


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    mgr = get_manager()
    last_event_count = 0
    last_log_count = 0

    try:
        _cleanup_counter = 0
        while True:
            await asyncio.sleep(mgr.config.refresh_interval)
            # Periodic cleanup of stale builder sessions (every ~60 cycles)
            _cleanup_counter += 1
            if _cleanup_counter >= 60:
                _cleanup_counter = 0
                _cleanup_stale_builder_sessions()
            metrics = mgr.get_metrics()
            current_count = metrics.total_events

            data: dict[str, Any] = {
                "type": "metrics",
                "data": metrics.model_dump(),
            }

            if current_count > last_event_count:
                events = mgr.global_collector.get_events_since(
                    last_event_count, limit=50
                )
                if events:
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
                agent_updates: dict[str, Any] = {}
                for i in running_agents:
                    try:
                        agent_updates[i.instance_id] = mgr.get_agent_status(
                            i.instance_id
                        )
                    except (KeyError, Exception):
                        pass  # Instance destroyed mid-poll; skip it
                if agent_updates:
                    data["agent_updates"] = agent_updates

            await websocket.send_text(json.dumps(data, default=str))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug(f"WebSocket error: {e}")
        try:
            await websocket.close()
        except Exception as close_err:
            logger.debug(f"WebSocket close failed: {close_err}")
