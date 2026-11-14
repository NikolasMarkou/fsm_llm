"""Routes: workflow presets, launch and control of a managed workflow engine."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import HTTPException

from fsm_llm.logging import logger

from ..definitions import (
    LaunchWorkflowRequest,
    WorkflowAdvanceRequest,
    WorkflowCancelRequest,
    WorkflowEventRequest,
)
from .application import app, get_manager
from .support import _GATED, _LLM_OPERATION_TIMEOUT, _bounded, _raise_http

# --- REST API: Workflow Launch/Control ---


@app.get("/api/workflow/presets")
async def api_workflow_presets() -> dict[str, list[dict[str, str]]]:
    """List built-in workflow presets available for launch."""
    mgr = get_manager()
    return {"workflows": mgr.get_workflow_presets()}


@app.post("/api/workflow/launch", dependencies=_GATED)
async def api_workflow_launch(req: LaunchWorkflowRequest) -> dict[str, Any]:
    """Launch a workflow preset and start an instance."""
    mgr = get_manager()
    try:
        # Making room destroys finished instances (a thread join, API.close):
        # keep it off the event loop like the other launch routes.
        managed = await asyncio.to_thread(
            mgr.launch_workflow,
            preset_id=req.preset_id,
            definition_json=req.definition_json,
            label=req.label,
        )
    except Exception as e:
        _raise_http(e, "launch workflow")
    try:
        wf_instance_id = await asyncio.wait_for(
            mgr.start_workflow_instance(
                managed.instance_id,
                managed.workflow_id,
                req.initial_context,
            ),
            timeout=_LLM_OPERATION_TIMEOUT,
        )
    except BaseException as e:
        # Do not leave a half-launched engine behind.
        try:
            await asyncio.to_thread(mgr.end_instance, managed.instance_id)
        except Exception as cleanup_err:
            logger.debug(f"Failed to clean up workflow launch: {cleanup_err}")
        if isinstance(e, asyncio.TimeoutError):
            raise HTTPException(
                status_code=504, detail="Workflow start timed out"
            ) from None
        if not isinstance(e, Exception):
            raise
        _raise_http(e, "start workflow")
    info = managed.to_info().model_dump()
    info["workflow_instance_id"] = wf_instance_id
    return info


@app.post("/api/workflow/{instance_id}/advance", dependencies=_GATED)
async def api_workflow_advance(
    instance_id: str, req: WorkflowAdvanceRequest
) -> dict[str, Any]:
    """Advance a workflow instance."""
    mgr = get_manager()
    result = await _bounded(
        mgr.advance_workflow(instance_id, req.workflow_instance_id, req.user_input),
        f"advance workflow on instance {instance_id}",
        "Workflow advance timed out",
    )
    return {"advanced": result}


@app.post("/api/workflow/{instance_id}/cancel", dependencies=_GATED)
async def api_workflow_cancel(
    instance_id: str, req: WorkflowCancelRequest
) -> dict[str, Any]:
    """Cancel a workflow instance."""
    mgr = get_manager()
    result = await _bounded(
        mgr.cancel_workflow(instance_id, req.workflow_instance_id, req.reason),
        f"cancel workflow on instance {instance_id}",
        "Workflow cancel timed out",
    )
    return {"cancelled": result}


@app.post("/api/workflow/{instance_id}/event", dependencies=_GATED)
async def api_workflow_event(
    instance_id: str, req: WorkflowEventRequest
) -> dict[str, Any]:
    """Deliver an event to a workflow engine (wakes waiting instances)."""
    mgr = get_manager()
    affected = await _bounded(
        mgr.send_workflow_event(
            instance_id, req.event_type, req.payload, req.workflow_instance_id or None
        ),
        f"deliver workflow event on instance {instance_id}",
        "Workflow event delivery timed out",
    )
    return {"affected": affected}


@app.get("/api/workflow/{instance_id}/status", dependencies=_GATED)
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
        _raise_http(e, f"get workflow status for instance {instance_id}")


@app.get("/api/workflow/{instance_id}/instances", dependencies=_GATED)
async def api_workflow_instances(instance_id: str) -> list[dict[str, Any]]:
    """List all workflow instances on a managed workflow engine."""
    mgr = get_manager()
    try:
        return mgr.get_workflow_instances(instance_id)
    except Exception as e:
        _raise_http(e, f"list workflow instances for {instance_id}")
