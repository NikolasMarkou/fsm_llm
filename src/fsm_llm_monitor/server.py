from __future__ import annotations

"""
FastAPI web server for fsm_llm_monitor.

Serves the retro 90s dashboard UI and provides REST + WebSocket APIs
for real-time monitoring and launching of FSMs, agents, and workflows.
"""

import asyncio
import json
import threading
import traceback
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
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

# Active sessions tracked by the monitor
_fsm_sessions: dict[str, Any] = {}  # conv_id -> API instance
_agent_results: dict[str, dict[str, Any]] = {}  # job_id -> result


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

    info: dict[str, str] = {"monitor_version": __version__}
    try:
        from fsm_llm import __version__ as cv

        info["fsm_llm_version"] = cv
    except ImportError:
        pass
    return info


# --- REST API: Presets (scan examples/) ---


@app.get("/api/presets")
async def api_presets() -> dict[str, list[dict[str, str]]]:
    """Scan examples/ directory for FSM, agent, and workflow presets."""

    # __file__ is src/fsm_llm_monitor/server.py → parent.parent.parent = project root
    base = Path(__file__).parent.parent.parent / "examples"
    if not base.exists():
        # Fallback: editable install, __file__ under src/
        base = base.parent / "examples"
    if not base.exists():
        return {"fsm": [], "agent": [], "workflow": []}

    fsm_presets: list[dict[str, str]] = []
    agent_presets: list[dict[str, str]] = []
    workflow_presets: list[dict[str, str]] = []

    # FSM presets: any directory with fsm.json
    for category in ["basic", "intermediate", "advanced", "classification", "reasoning"]:
        cat_dir = base / category
        if not cat_dir.exists():
            continue
        for example_dir in sorted(cat_dir.iterdir()):
            if not example_dir.is_dir():
                continue
            fsm_files = list(example_dir.glob("*.json"))
            for f in fsm_files:
                name = example_dir.name.replace("_", " ").title()
                try:
                    data = json.loads(f.read_text())
                    desc = data.get("description", "")
                except Exception:
                    desc = ""
                fsm_presets.append({
                    "name": f"{name} ({f.name})",
                    "path": str(f),
                    "category": category,
                    "description": desc,
                })

    # Agent presets: directories in examples/agents/
    agents_dir = base / "agents"
    if agents_dir.exists():
        for example_dir in sorted(agents_dir.iterdir()):
            if not example_dir.is_dir():
                continue
            run_py = example_dir / "run.py"
            desc = ""
            if run_py.exists():
                # Extract docstring first line
                text = run_py.read_text()
                for line in text.split("\n"):
                    line = line.strip().strip('"').strip("'")
                    if line and not line.startswith("#") and not line.startswith("import"):
                        desc = line
                        break
            agent_presets.append({
                "name": example_dir.name.replace("_", " ").title(),
                "id": example_dir.name,
                "path": str(example_dir),
                "description": desc,
            })

    # Workflow presets: directories in examples/workflows/
    wf_dir = base / "workflows"
    if wf_dir.exists():
        for example_dir in sorted(wf_dir.iterdir()):
            if not example_dir.is_dir():
                continue
            workflow_presets.append({
                "name": example_dir.name.replace("_", " ").title(),
                "id": example_dir.name,
                "path": str(example_dir),
            })

    return {
        "fsm": fsm_presets,
        "agent": agent_presets,
        "workflow": workflow_presets,
    }


@app.get("/api/fsm/visualize")
async def api_fsm_visualize(path: str) -> dict[str, Any]:
    """Load FSM and return visualization data (nodes + edges for SVG rendering)."""
    bridge = get_bridge()
    snap = bridge.load_fsm_from_file(path)
    if snap is None:
        return {"error": "failed to load"}

    nodes = []
    edges = []
    for i, state in enumerate(snap.states):
        nodes.append({
            "id": state.state_id,
            "label": state.state_id,
            "description": state.description,
            "purpose": state.purpose,
            "is_initial": state.is_initial,
            "is_terminal": state.is_terminal,
            "x": 150 + (i % 4) * 200,
            "y": 80 + (i // 4) * 140,
        })
        for t in state.transitions:
            edges.append({
                "from": state.state_id,
                "to": t.target_state,
                "label": t.description[:30] if t.description else "",
                "priority": t.priority,
            })

    return {
        "fsm": snap.model_dump(),
        "nodes": nodes,
        "edges": edges,
    }


# --- REST API: Visualize Agent ---

# Hardcoded agent pattern flows — the state graph is fixed per agent type,
# only the tool list varies at runtime.
_AGENT_FLOWS: dict[str, dict[str, Any]] = {
    "ReactAgent": {
        "description": "ReAct loop: reason about the task, execute a tool, observe the result, repeat or conclude.",
        "nodes": [
            {"id": "think", "label": "think", "description": "Reason about the task and select a tool", "is_initial": True, "is_terminal": False},
            {"id": "act", "label": "act", "description": "Execute the selected tool", "is_initial": False, "is_terminal": False},
            {"id": "conclude", "label": "conclude", "description": "Produce the final answer", "is_initial": False, "is_terminal": True},
        ],
        "edges": [
            {"from": "think", "to": "act", "label": "tool selected"},
            {"from": "act", "to": "think", "label": "observation recorded"},
            {"from": "think", "to": "conclude", "label": "should_terminate"},
        ],
    },
    "ReflexionAgent": {
        "description": "ReAct with self-reflection: evaluate results and reflect on failures before retrying.",
        "nodes": [
            {"id": "think", "label": "think", "description": "Reason about the task", "is_initial": True, "is_terminal": False},
            {"id": "act", "label": "act", "description": "Execute the selected tool", "is_initial": False, "is_terminal": False},
            {"id": "evaluate", "label": "evaluate", "description": "Evaluate the result", "is_initial": False, "is_terminal": False},
            {"id": "reflect", "label": "reflect", "description": "Self-reflect on failure", "is_initial": False, "is_terminal": False},
            {"id": "conclude", "label": "conclude", "description": "Produce the final answer", "is_initial": False, "is_terminal": True},
        ],
        "edges": [
            {"from": "think", "to": "act", "label": "tool selected"},
            {"from": "act", "to": "evaluate", "label": "observation"},
            {"from": "evaluate", "to": "conclude", "label": "passed"},
            {"from": "evaluate", "to": "reflect", "label": "failed"},
            {"from": "reflect", "to": "think", "label": "retry"},
        ],
    },
    "PlanExecuteAgent": {
        "description": "Plan decomposition and sequential execution with replanning on failure.",
        "nodes": [
            {"id": "plan", "label": "plan", "description": "Create execution plan", "is_initial": True, "is_terminal": False},
            {"id": "execute_step", "label": "execute_step", "description": "Execute the current step", "is_initial": False, "is_terminal": False},
            {"id": "check_result", "label": "check_result", "description": "Check step result", "is_initial": False, "is_terminal": False},
            {"id": "replan", "label": "replan", "description": "Revise the plan", "is_initial": False, "is_terminal": False},
            {"id": "synthesize", "label": "synthesize", "description": "Synthesize final answer", "is_initial": False, "is_terminal": True},
        ],
        "edges": [
            {"from": "plan", "to": "execute_step", "label": "plan ready"},
            {"from": "execute_step", "to": "check_result", "label": "step done"},
            {"from": "check_result", "to": "execute_step", "label": "next step"},
            {"from": "check_result", "to": "replan", "label": "step failed"},
            {"from": "check_result", "to": "synthesize", "label": "all done"},
            {"from": "replan", "to": "execute_step", "label": "revised"},
        ],
    },
    "DebateAgent": {
        "description": "Multi-perspective debate with judge for consensus-building.",
        "nodes": [
            {"id": "propose", "label": "propose", "description": "Proposer presents argument", "is_initial": True, "is_terminal": False},
            {"id": "critique", "label": "critique", "description": "Critic challenges argument", "is_initial": False, "is_terminal": False},
            {"id": "counter", "label": "counter", "description": "Counter-argument presented", "is_initial": False, "is_terminal": False},
            {"id": "judge", "label": "judge", "description": "Judge evaluates arguments", "is_initial": False, "is_terminal": False},
            {"id": "conclude", "label": "conclude", "description": "Final verdict reached", "is_initial": False, "is_terminal": True},
        ],
        "edges": [
            {"from": "propose", "to": "critique", "label": "proposition made"},
            {"from": "critique", "to": "counter", "label": "critique given"},
            {"from": "counter", "to": "judge", "label": "counter made"},
            {"from": "judge", "to": "propose", "label": "another round"},
            {"from": "judge", "to": "conclude", "label": "consensus"},
        ],
    },
    "SelfConsistencyAgent": {
        "description": "Multiple samples with voting: run N times, aggregate by majority vote.",
        "nodes": [
            {"id": "generate", "label": "generate", "description": "Generate one answer sample", "is_initial": True, "is_terminal": False},
            {"id": "output", "label": "output", "description": "Output the answer", "is_initial": False, "is_terminal": True},
        ],
        "edges": [
            {"from": "generate", "to": "output", "label": "answer generated"},
        ],
    },
    "REWOOAgent": {
        "description": "Planning-first: one LLM call to plan all tool calls, execute without LLM, then synthesize.",
        "nodes": [
            {"id": "plan_all", "label": "plan_all", "description": "Generate complete tool plan", "is_initial": True, "is_terminal": False},
            {"id": "execute_plans", "label": "execute_plans", "description": "Execute all tools sequentially", "is_initial": False, "is_terminal": False},
            {"id": "solve", "label": "solve", "description": "Synthesize final answer", "is_initial": False, "is_terminal": True},
        ],
        "edges": [
            {"from": "plan_all", "to": "execute_plans", "label": "plan ready"},
            {"from": "execute_plans", "to": "solve", "label": "all executed"},
        ],
    },
    "PromptChainAgent": {
        "description": "Sequential prompt pipeline: each step transforms input for the next.",
        "nodes": [
            {"id": "step_0", "label": "step_0", "description": "First chain step", "is_initial": True, "is_terminal": False},
            {"id": "step_1", "label": "step_1", "description": "Second chain step", "is_initial": False, "is_terminal": False},
            {"id": "step_2", "label": "step_2", "description": "Third chain step", "is_initial": False, "is_terminal": False},
            {"id": "output", "label": "output", "description": "Final chain output", "is_initial": False, "is_terminal": True},
        ],
        "edges": [
            {"from": "step_0", "to": "step_1", "label": "gate passed"},
            {"from": "step_1", "to": "step_2", "label": "gate passed"},
            {"from": "step_2", "to": "output", "label": "chain complete"},
        ],
    },
    "EvaluatorOptimizerAgent": {
        "description": "Iterative evaluation and optimization: generate, evaluate, refine until quality passes.",
        "nodes": [
            {"id": "generate", "label": "generate", "description": "Generate initial output", "is_initial": True, "is_terminal": False},
            {"id": "evaluate", "label": "evaluate", "description": "Evaluate output quality", "is_initial": False, "is_terminal": False},
            {"id": "refine", "label": "refine", "description": "Refine based on feedback", "is_initial": False, "is_terminal": False},
            {"id": "output", "label": "output", "description": "Final refined output", "is_initial": False, "is_terminal": True},
        ],
        "edges": [
            {"from": "generate", "to": "evaluate", "label": "draft ready"},
            {"from": "evaluate", "to": "output", "label": "quality passed"},
            {"from": "evaluate", "to": "refine", "label": "needs improvement"},
            {"from": "refine", "to": "evaluate", "label": "refined"},
        ],
    },
    "MakerCheckerAgent": {
        "description": "Draft-review verification loop: maker drafts, checker reviews, revise until approved.",
        "nodes": [
            {"id": "make", "label": "make", "description": "Maker generates draft", "is_initial": True, "is_terminal": False},
            {"id": "check", "label": "check", "description": "Checker reviews draft", "is_initial": False, "is_terminal": False},
            {"id": "revise", "label": "revise", "description": "Maker revises draft", "is_initial": False, "is_terminal": False},
            {"id": "output", "label": "output", "description": "Approved output", "is_initial": False, "is_terminal": True},
        ],
        "edges": [
            {"from": "make", "to": "check", "label": "draft ready"},
            {"from": "check", "to": "output", "label": "approved"},
            {"from": "check", "to": "revise", "label": "needs revision"},
            {"from": "revise", "to": "check", "label": "revised"},
        ],
    },
    "OrchestratorAgent": {
        "description": "Task decomposition and delegation: orchestrate subtasks across workers.",
        "nodes": [
            {"id": "orchestrate", "label": "orchestrate", "description": "Decompose task into subtasks", "is_initial": True, "is_terminal": False},
            {"id": "delegate", "label": "delegate", "description": "Delegate subtask to worker", "is_initial": False, "is_terminal": False},
            {"id": "collect", "label": "collect", "description": "Collect worker results", "is_initial": False, "is_terminal": False},
            {"id": "synthesize", "label": "synthesize", "description": "Synthesize final answer", "is_initial": False, "is_terminal": True},
        ],
        "edges": [
            {"from": "orchestrate", "to": "delegate", "label": "subtasks ready"},
            {"from": "delegate", "to": "collect", "label": "delegated"},
            {"from": "collect", "to": "orchestrate", "label": "more work needed"},
            {"from": "collect", "to": "synthesize", "label": "all collected"},
        ],
    },
    "ADaPTAgent": {
        "description": "Adaptive complexity: try first, decompose on failure, combine results.",
        "nodes": [
            {"id": "attempt", "label": "attempt", "description": "Attempt to solve directly", "is_initial": True, "is_terminal": False},
            {"id": "assess", "label": "assess", "description": "Assess attempt result", "is_initial": False, "is_terminal": False},
            {"id": "decompose", "label": "decompose", "description": "Decompose into sub-problems", "is_initial": False, "is_terminal": False},
            {"id": "combine", "label": "combine", "description": "Combine results", "is_initial": False, "is_terminal": True},
        ],
        "edges": [
            {"from": "attempt", "to": "assess", "label": "attempt made"},
            {"from": "assess", "to": "combine", "label": "success"},
            {"from": "assess", "to": "decompose", "label": "too complex"},
            {"from": "decompose", "to": "combine", "label": "depth limit"},
        ],
    },
    "ReasoningReactAgent": {
        "description": "ReAct with structured reasoning: enhanced think step uses reasoning engine.",
        "nodes": [
            {"id": "think", "label": "think", "description": "Reason with structured reasoning", "is_initial": True, "is_terminal": False},
            {"id": "act", "label": "act", "description": "Execute the selected tool", "is_initial": False, "is_terminal": False},
            {"id": "conclude", "label": "conclude", "description": "Produce the final answer", "is_initial": False, "is_terminal": True},
        ],
        "edges": [
            {"from": "think", "to": "act", "label": "tool selected"},
            {"from": "act", "to": "think", "label": "observation recorded"},
            {"from": "think", "to": "conclude", "label": "should_terminate"},
        ],
    },
}


@app.get("/api/agent/visualize")
async def api_agent_visualize(agent_type: str = "ReactAgent") -> dict[str, Any]:
    """Return visualization data for an agent pattern flow."""
    flow = _AGENT_FLOWS.get(agent_type)
    if flow is None:
        return {"error": f"unknown agent type: {agent_type}"}

    # Assign grid positions to nodes
    nodes = []
    for i, node in enumerate(flow["nodes"]):
        nodes.append({
            **node,
            "x": 150 + (i % 4) * 200,
            "y": 80 + (i // 4) * 140,
        })

    return {
        "info": {
            "name": agent_type,
            "description": flow["description"],
            "state_count": len(flow["nodes"]),
        },
        "nodes": nodes,
        "edges": flow["edges"],
        "agent_types": list(_AGENT_FLOWS.keys()),
    }


# --- REST API: Visualize Workflow ---

# Hardcoded workflow patterns — workflow definitions are Python DSL,
# so we provide the known patterns as static graph data.
_WORKFLOW_FLOWS: dict[str, dict[str, Any]] = {
    "order_processing": {
        "name": "Order Processing",
        "description": "End-to-end order workflow: collect details, validate, process payment, confirm.",
        "nodes": [
            {"id": "start", "label": "start", "description": "Initialize order", "step_type": "auto", "is_initial": True, "is_terminal": False},
            {"id": "collect_order", "label": "collect_order", "description": "Collect order details via FSM", "step_type": "conversation", "is_initial": False, "is_terminal": False},
            {"id": "validate", "label": "validate", "description": "Validate order data", "step_type": "condition", "is_initial": False, "is_terminal": False},
            {"id": "process_payment", "label": "process_payment", "description": "Process payment", "step_type": "api", "is_initial": False, "is_terminal": False},
            {"id": "send_confirmation", "label": "send_confirmation", "description": "Send confirmation", "step_type": "auto", "is_initial": False, "is_terminal": False},
            {"id": "completed", "label": "completed", "description": "Order complete", "step_type": "auto", "is_initial": False, "is_terminal": True},
            {"id": "failed", "label": "failed", "description": "Order failed", "step_type": "auto", "is_initial": False, "is_terminal": True},
        ],
        "edges": [
            {"from": "start", "to": "collect_order", "label": "begin"},
            {"from": "collect_order", "to": "validate", "label": "details collected"},
            {"from": "validate", "to": "process_payment", "label": "valid"},
            {"from": "validate", "to": "failed", "label": "invalid"},
            {"from": "process_payment", "to": "send_confirmation", "label": "payment ok"},
            {"from": "process_payment", "to": "failed", "label": "payment failed"},
            {"from": "send_confirmation", "to": "completed", "label": "confirmed"},
        ],
    },
    "data_pipeline": {
        "name": "Data Pipeline",
        "description": "ETL workflow: extract data, transform with LLM, validate, load to destination.",
        "nodes": [
            {"id": "extract", "label": "extract", "description": "Extract data from source", "step_type": "api", "is_initial": True, "is_terminal": False},
            {"id": "transform", "label": "transform", "description": "Transform data with LLM", "step_type": "llm", "is_initial": False, "is_terminal": False},
            {"id": "validate", "label": "validate", "description": "Validate transformed data", "step_type": "condition", "is_initial": False, "is_terminal": False},
            {"id": "load", "label": "load", "description": "Load to destination", "step_type": "api", "is_initial": False, "is_terminal": False},
            {"id": "done", "label": "done", "description": "Pipeline complete", "step_type": "auto", "is_initial": False, "is_terminal": True},
            {"id": "error", "label": "error", "description": "Pipeline error", "step_type": "auto", "is_initial": False, "is_terminal": True},
        ],
        "edges": [
            {"from": "extract", "to": "transform", "label": "data extracted"},
            {"from": "transform", "to": "validate", "label": "transformed"},
            {"from": "validate", "to": "load", "label": "valid"},
            {"from": "validate", "to": "error", "label": "invalid"},
            {"from": "load", "to": "done", "label": "loaded"},
        ],
    },
    "approval_flow": {
        "name": "Approval Flow",
        "description": "Multi-stage approval: submit, review, approve/reject, notify.",
        "nodes": [
            {"id": "submit", "label": "submit", "description": "Submit request", "step_type": "conversation", "is_initial": True, "is_terminal": False},
            {"id": "review", "label": "review", "description": "Review request", "step_type": "llm", "is_initial": False, "is_terminal": False},
            {"id": "decide", "label": "decide", "description": "Approve or reject", "step_type": "condition", "is_initial": False, "is_terminal": False},
            {"id": "notify_approved", "label": "notify_approved", "description": "Notify approval", "step_type": "auto", "is_initial": False, "is_terminal": True},
            {"id": "notify_rejected", "label": "notify_rejected", "description": "Notify rejection", "step_type": "auto", "is_initial": False, "is_terminal": True},
        ],
        "edges": [
            {"from": "submit", "to": "review", "label": "submitted"},
            {"from": "review", "to": "decide", "label": "reviewed"},
            {"from": "decide", "to": "notify_approved", "label": "approved"},
            {"from": "decide", "to": "notify_rejected", "label": "rejected"},
        ],
    },
    "parallel_processing": {
        "name": "Parallel Processing",
        "description": "Fan-out/fan-in: split work into parallel branches, aggregate results.",
        "nodes": [
            {"id": "start", "label": "start", "description": "Initialize work", "step_type": "auto", "is_initial": True, "is_terminal": False},
            {"id": "parallel", "label": "parallel", "description": "Execute branches in parallel", "step_type": "parallel", "is_initial": False, "is_terminal": False},
            {"id": "aggregate", "label": "aggregate", "description": "Aggregate parallel results", "step_type": "auto", "is_initial": False, "is_terminal": False},
            {"id": "done", "label": "done", "description": "Processing complete", "step_type": "auto", "is_initial": False, "is_terminal": True},
        ],
        "edges": [
            {"from": "start", "to": "parallel", "label": "fan out"},
            {"from": "parallel", "to": "aggregate", "label": "all done"},
            {"from": "aggregate", "to": "done", "label": "aggregated"},
        ],
    },
    "timer_wait": {
        "name": "Timer & Event Wait",
        "description": "Workflow with timer delays and event-based triggers.",
        "nodes": [
            {"id": "start", "label": "start", "description": "Begin workflow", "step_type": "auto", "is_initial": True, "is_terminal": False},
            {"id": "process", "label": "process", "description": "Process initial data", "step_type": "llm", "is_initial": False, "is_terminal": False},
            {"id": "wait_delay", "label": "wait_delay", "description": "Wait for timer", "step_type": "timer", "is_initial": False, "is_terminal": False},
            {"id": "wait_event", "label": "wait_event", "description": "Wait for external event", "step_type": "event", "is_initial": False, "is_terminal": False},
            {"id": "finalize", "label": "finalize", "description": "Finalize workflow", "step_type": "auto", "is_initial": False, "is_terminal": True},
        ],
        "edges": [
            {"from": "start", "to": "process", "label": "begin"},
            {"from": "process", "to": "wait_delay", "label": "processed"},
            {"from": "wait_delay", "to": "wait_event", "label": "timer elapsed"},
            {"from": "wait_event", "to": "finalize", "label": "event received"},
        ],
    },
}


@app.get("/api/workflow/visualize")
async def api_workflow_visualize(workflow_id: str = "order_processing") -> dict[str, Any]:
    """Return visualization data for a workflow pattern."""
    flow = _WORKFLOW_FLOWS.get(workflow_id)
    if flow is None:
        return {"error": f"unknown workflow: {workflow_id}"}

    # Assign grid positions to nodes
    nodes = []
    for i, node in enumerate(flow["nodes"]):
        nodes.append({
            **node,
            "x": 150 + (i % 4) * 200,
            "y": 80 + (i // 4) * 140,
        })

    return {
        "info": {
            "name": flow["name"],
            "description": flow["description"],
            "step_count": len(flow["nodes"]),
        },
        "nodes": nodes,
        "edges": flow["edges"],
        "workflow_ids": list(_WORKFLOW_FLOWS.keys()),
    }


# --- REST API: Launch FSM ---


class FSMLaunchRequest(BaseModel):
    fsm_path: str
    model: str = "gpt-4o-mini"
    temperature: float = 0.5
    max_tokens: int = 1000
    initial_context: dict[str, Any] | None = None


class FSMConverseRequest(BaseModel):
    message: str


@app.post("/api/launch/fsm")
async def launch_fsm(req: FSMLaunchRequest) -> dict[str, Any]:
    """Launch an FSM conversation from a JSON file."""
    try:
        from fsm_llm import API

        api_instance = API.from_file(
            path=req.fsm_path,
            model=req.model,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )

        # Connect monitor bridge to this API
        bridge = get_bridge()
        if not bridge.connected:
            bridge.connect(api_instance)

        conv_id, initial_response = api_instance.start_conversation(
            initial_context=req.initial_context
        )

        _fsm_sessions[conv_id] = api_instance

        return {
            "conversation_id": conv_id,
            "initial_response": initial_response,
            "state": api_instance.get_current_state(conv_id),
        }
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


@app.post("/api/fsm/{conversation_id}/converse")
async def fsm_converse(
    conversation_id: str, req: FSMConverseRequest
) -> dict[str, Any]:
    """Send a message to an active FSM conversation."""
    api_instance = _fsm_sessions.get(conversation_id)
    if api_instance is None:
        return {"error": "conversation not found"}

    try:
        response = api_instance.converse(req.message, conversation_id)
        return {
            "response": response,
            "state": api_instance.get_current_state(conversation_id),
            "ended": api_instance.has_conversation_ended(conversation_id),
            "data": api_instance.get_data(conversation_id),
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/fsm/{conversation_id}/end")
async def fsm_end(conversation_id: str) -> dict[str, Any]:
    """End an FSM conversation."""
    api_instance = _fsm_sessions.pop(conversation_id, None)
    if api_instance is None:
        return {"error": "conversation not found"}
    try:
        api_instance.end_conversation(conversation_id)
    except Exception:
        pass
    return {"status": "ended"}


@app.get("/api/fsm/sessions")
async def fsm_sessions() -> list[dict[str, Any]]:
    """List active FSM sessions launched from monitor."""
    sessions = []
    for conv_id, api_inst in list(_fsm_sessions.items()):
        try:
            sessions.append({
                "conversation_id": conv_id,
                "state": api_inst.get_current_state(conv_id),
                "ended": api_inst.has_conversation_ended(conv_id),
            })
        except Exception:
            sessions.append({
                "conversation_id": conv_id,
                "state": "unknown",
                "ended": True,
            })
    return sessions


# --- REST API: Launch Agent ---


class AgentLaunchRequest(BaseModel):
    task: str
    model: str = "gpt-4o-mini"
    max_iterations: int = 10
    temperature: float = 0.5
    timeout_seconds: float = 300.0


@app.post("/api/launch/agent")
async def launch_agent(req: AgentLaunchRequest) -> dict[str, Any]:
    """Launch a ReactAgent with basic tools (runs in background thread)."""
    try:
        from fsm_llm_agents import AgentConfig, ReactAgent, ToolRegistry
    except ImportError:
        return {"error": "fsm_llm_agents not installed"}

    job_id = str(uuid.uuid4())[:8]
    _agent_results[job_id] = {"status": "running", "task": req.task}

    def _run_agent() -> None:
        try:
            registry = ToolRegistry()

            # Register a basic echo tool so the agent has something to work with
            def echo(text: str) -> str:
                """Return the input text as-is."""
                return text

            registry.register_function(
                echo, name="echo", description="Echo text back"
            )

            config = AgentConfig(
                model=req.model,
                max_iterations=req.max_iterations,
                temperature=req.temperature,
                timeout_seconds=req.timeout_seconds,
            )

            agent = ReactAgent(tools=registry, config=config)
            result = agent.run(req.task)

            _agent_results[job_id] = {
                "status": "completed",
                "task": req.task,
                "answer": result.answer,
                "success": result.success,
                "iterations_used": result.iterations_used,
                "tools_used": result.tools_used,
            }
        except Exception as e:
            _agent_results[job_id] = {
                "status": "failed",
                "task": req.task,
                "error": str(e),
            }

    thread = threading.Thread(target=_run_agent, daemon=True)
    thread.start()

    return {"job_id": job_id, "status": "running", "task": req.task}


@app.get("/api/agent/{job_id}")
async def agent_status(job_id: str) -> dict[str, Any]:
    """Get status/result of an agent job."""
    result = _agent_results.get(job_id)
    if result is None:
        return {"error": "job not found"}
    return result


@app.get("/api/agent/jobs")
async def agent_jobs() -> list[dict[str, Any]]:
    """List all agent jobs."""
    return [{"job_id": k, **v} for k, v in _agent_results.items()]


# --- REST API: Launch Workflow ---


class WorkflowLaunchRequest(BaseModel):
    workflow_id: str = "demo"
    initial_context: dict[str, Any] | None = None


_workflow_engine: Any = None
_workflow_instances: dict[str, str] = {}  # instance_id -> workflow_id


@app.post("/api/launch/workflow")
async def launch_workflow(req: WorkflowLaunchRequest) -> dict[str, Any]:
    """Launch a workflow (requires pre-registered workflow definitions)."""
    global _workflow_engine
    try:
        from fsm_llm_workflows import WorkflowEngine
    except ImportError:
        return {"error": "fsm_llm_workflows not installed"}

    if _workflow_engine is None:
        _workflow_engine = WorkflowEngine()

    try:
        instance_id = await _workflow_engine.start_workflow(
            workflow_id=req.workflow_id,
            initial_context=req.initial_context,
        )
        _workflow_instances[instance_id] = req.workflow_id
        return {"instance_id": instance_id, "status": "started"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/workflow/{instance_id}")
async def workflow_status(instance_id: str) -> dict[str, Any]:
    """Get workflow instance status."""
    global _workflow_engine
    if _workflow_engine is None:
        return {"error": "no workflow engine"}
    try:
        status = _workflow_engine.get_workflow_status(instance_id)
        context = _workflow_engine.get_workflow_context(instance_id)
        return {
            "instance_id": instance_id,
            "status": status.value if hasattr(status, "value") else str(status),
            "context": context,
        }
    except Exception as e:
        return {"error": str(e)}


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
            await websocket.send_text(json.dumps(data, default=str))
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
