"""Routes: launch and control of a managed agent."""

from __future__ import annotations

import asyncio
from typing import Any

from ..definitions import LaunchAgentRequest
from .application import app, get_manager
from .support import _GATED, _raise_http

# --- REST API: Agent Launch/Control ---


@app.post("/api/agent/launch", dependencies=_GATED)
async def api_agent_launch(req: LaunchAgentRequest) -> dict[str, Any]:
    """Launch an agent in a background thread."""
    mgr = get_manager()
    try:
        managed = await asyncio.to_thread(
            mgr.launch_agent,
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
        # Unknown/disabled agent type or missing tools -> 400; capacity -> 429;
        # extension unavailable -> 501.
        _raise_http(e, "launch agent")


@app.get("/api/agent/types")
async def api_agent_types() -> dict[str, Any]:
    """The registry rows with their launch and draw flags (open like capabilities; launch is gated)."""
    mgr = get_manager()
    return {"agent_types": await asyncio.to_thread(mgr.agent_types)}


@app.get("/api/agent/{instance_id}/status", dependencies=_GATED)
async def api_agent_status(instance_id: str) -> dict[str, Any]:
    """Get agent execution status."""
    mgr = get_manager()
    try:
        return mgr.get_agent_status(instance_id)
    except Exception as e:
        _raise_http(e, f"get agent status for instance {instance_id}")


@app.get("/api/agent/{instance_id}/result", dependencies=_GATED)
async def api_agent_result(instance_id: str) -> dict[str, Any]:
    """Get final agent result."""
    mgr = get_manager()
    try:
        return mgr.get_agent_result(instance_id)
    except Exception as e:
        _raise_http(e, f"get agent result for instance {instance_id}")


@app.post("/api/agent/{instance_id}/cancel", dependencies=_GATED)
async def api_agent_cancel(instance_id: str) -> dict[str, Any]:
    """Cancel a running agent (``cancelled`` is False when it had finished)."""
    mgr = get_manager()
    try:
        cancelled = mgr.cancel_agent(instance_id)
        return {"status": "ok", "cancelled": cancelled}
    except Exception as e:
        _raise_http(e, f"cancel agent {instance_id}")
