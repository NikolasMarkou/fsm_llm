"""Routes: FSM visualization, FSM presets, the agent preview graphs and the workflow
pattern flows (``flows.json``)."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from fastapi import HTTPException
from pydantic import ValidationError
from starlette.requests import Request

from fsm_llm.core.visualize import build_fsm_graph
from fsm_llm.logging import logger

from ..constants import VIZ_EDGE_LABEL_CHARS
from ..definitions import FSMSnapshot, StateInfo, TransitionInfo
from ..instance_manager import find_examples_dir, load_preset_json, read_json_file
from .application import app, get_flows, get_manager
from .support import _json_body

# Preset cache: (result, timestamp). Presets rarely change; avoid re-scanning disk.
_PRESET_CACHE_TTL = 60.0
_preset_cache: tuple[dict[str, list[dict[str, str]]], float] | None = None

# --- REST API: FSM Visualization ---


@app.post("/api/fsm/load")
async def api_fsm_load(request: Request) -> dict[str, Any]:
    """Load FSM definition from JSON body."""
    return _snapshot_or_400(await _json_body(request)).model_dump()


def _snapshot_or_400(data: Any) -> FSMSnapshot:
    """Parse an FSM definition dict into a snapshot, or raise 400."""
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="failed to parse FSM definition")
    try:
        return _fsm_dict_to_snapshot(data)
    except Exception as e:
        logger.debug(f"Failed to convert FSM dict to snapshot: {e}")
        raise HTTPException(
            status_code=400, detail="failed to parse FSM definition"
        ) from e


def _fsm_dict_to_snapshot(data: dict[str, Any]) -> FSMSnapshot:
    """Convert a raw FSM definition dict to an FSMSnapshot (the info panel)."""
    states_data = data.get("states", {})
    initial_state = data.get("initial_state", "")

    states = []
    for state_id, state_def in states_data.items():
        transitions_raw = state_def.get("transitions", [])
        transitions = []
        for t in transitions_raw:
            conditions = t.get("conditions") or []
            transitions.append(
                TransitionInfo(
                    target_state=t.get("target_state", ""),
                    description=t.get("description", ""),
                    priority=t.get("priority", 100),
                    condition_count=len(conditions),
                    has_logic=any(c.get("logic") for c in conditions),
                )
            )

        states.append(
            StateInfo(
                state_id=state_id,
                description=state_def.get("description", ""),
                purpose=state_def.get("purpose", ""),
                is_initial=(state_id == initial_state),
                is_terminal=len(transitions_raw) == 0,
                transition_count=len(transitions_raw),
                transitions=transitions,
            )
        )

    return FSMSnapshot(
        name=data.get("name", ""),
        description=data.get("description", ""),
        version=data.get("version", ""),
        initial_state=initial_state,
        persona=data.get("persona"),
        state_count=len(states),
        states=states,
    )


def _graph_error_reason(error: Exception) -> str:
    """Why core could not graph a definition, for the 400 detail.

    ``build_fsm_graph`` documents ``ValueError`` for a malformed shape; its
    message names the state, transition or field. A pydantic error is reduced
    to ``field: message`` pairs (no documentation URL, no input echo). Any
    other exception type is not part of that contract and gets a fixed text,
    so no internal message reaches the client.
    """
    if isinstance(error, ValidationError):
        return "; ".join(
            f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
            for item in error.errors(include_url=False, include_input=False)
        )
    if isinstance(error, ValueError):
        return str(error)
    return "the definition has an unexpected shape"


def _fsm_viz_or_400(data: Any) -> dict[str, Any]:
    """The visualize payload of an FSM definition dict, or raise 400.

    ``nodes`` and ``edges`` are core's ``build_fsm_graph`` output in the key
    names the page reads (``from``/``to``), plus the edge label cut to
    ``VIZ_EDGE_LABEL_CHARS``. The page lays the nodes out itself.
    """
    # DECISION plan-2026-09-30T062855-07ad3f8c/D-018
    # Nodes and edges come from core's build_fsm_graph. Do NOT derive them from
    # the FSMSnapshot below (it only feeds the info panel) or from the raw
    # dict: a second graph builder drifts from the Mermaid/DOT exporters. A
    # definition core cannot graph (unknown initial state, dangling target) is
    # a 400, not a diagram with edges to nowhere (D-040).
    snap = _snapshot_or_400(data)
    try:
        graph = build_fsm_graph(data)
    except Exception as e:
        logger.debug(f"Failed to build the FSM graph: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"failed to parse FSM definition: {_graph_error_reason(e)}",
        ) from e
    nodes = [
        {
            "id": node.id,
            "label": node.id,
            "description": node.description,
            "purpose": node.purpose,
            "is_initial": node.is_initial,
            "is_terminal": node.is_terminal,
        }
        for node in graph.nodes
    ]
    edges = [
        {
            "from": edge.source,
            "to": edge.target,
            "label": edge.description[:VIZ_EDGE_LABEL_CHARS],
            "priority": edge.priority,
        }
        for edge in graph.edges
    ]
    return {"fsm": snap.model_dump(), "nodes": nodes, "edges": edges}


@app.post("/api/fsm/visualize")
async def api_fsm_visualize(request: Request) -> dict[str, Any]:
    """Accept FSM JSON definition and return visualization data."""
    return _fsm_viz_or_400(await _json_body(request))


def _read_preset_json(preset_id: str) -> dict[str, Any]:
    """Load a preset JSON file by ID with path traversal protection."""
    try:
        return load_preset_json(preset_id)
    except FileNotFoundError as e:
        detail = (
            "examples directory not found"
            if find_examples_dir() is None
            else "preset not found"
        )
        raise HTTPException(status_code=404, detail=detail) from e
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
        raise HTTPException(status_code=500, detail="failed to read preset") from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail="invalid preset ID") from e


@app.get("/api/fsm/visualize/preset/{preset_id:path}")
async def api_fsm_visualize_preset(preset_id: str) -> dict[str, Any]:
    """Load an FSM preset by ID and return visualization data."""
    data = _read_preset_json(preset_id)
    return _fsm_viz_or_400(data)


# --- REST API: Presets ---


def _scan_fsm_presets() -> dict[str, list[dict[str, str]]]:
    """Scan the examples/ directory for FSM presets (blocking disk I/O).

    Returns preset metadata only — no filesystem paths exposed. Run off the
    event loop via ``asyncio.to_thread``.
    """
    base = find_examples_dir()
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
                    data = read_json_file(f)
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


@app.get("/api/presets")
async def api_presets() -> dict[str, list[dict[str, str]]]:
    """Return FSM presets, cached for _PRESET_CACHE_TTL seconds.

    The filesystem scan runs off the event loop via ``asyncio.to_thread``.
    """
    global _preset_cache
    if _preset_cache is not None:
        cached_result, cached_at = _preset_cache
        if time.monotonic() - cached_at < _PRESET_CACHE_TTL:
            return cached_result

    result = await asyncio.to_thread(_scan_fsm_presets)
    _preset_cache = (result, time.monotonic())
    return result


@app.get("/api/preset/fsm/{preset_id:path}")
async def api_preset_fsm(preset_id: str) -> dict[str, Any]:
    """Load an FSM preset by ID and return its JSON content."""
    return _read_preset_json(preset_id)


# --- REST API: Pattern Visualization (agents by executable id, workflows from flows.json) ---


@app.get("/api/agent/visualize")
async def api_agent_visualize(agent_type: str = "react") -> dict[str, Any]:
    """Draw a listed agent id from its preview FSM through ``_fsm_viz_or_400`` (404 otherwise, id never echoed)."""
    try:
        preview = await asyncio.to_thread(get_manager().agent_preview, agent_type)
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown agent type") from None
    if preview is None:
        raise HTTPException(status_code=404, detail="no preview for this agent type")
    return _fsm_viz_or_400(preview)


@app.get("/api/workflow/visualize")
async def api_workflow_visualize(
    workflow_id: str = "order_processing",
) -> dict[str, Any]:
    """Return visualization data for a workflow pattern."""
    flow = get_flows().get("workflows", {}).get(workflow_id)
    if flow is None:
        raise HTTPException(status_code=404, detail=f"unknown workflow: {workflow_id}")
    return {
        "info": {
            "name": flow["name"],
            "description": flow["description"],
            "step_count": len(flow["nodes"]),
        },
        "nodes": flow["nodes"],
        "edges": flow["edges"],
    }
