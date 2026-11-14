"""Routes: launch and control of a managed FSM (start, converse, end, list)."""

from __future__ import annotations

import asyncio
from typing import Any

from ..definitions import (
    EndConversationRequest,
    LaunchFSMRequest,
    SendMessageRequest,
    StartConversationRequest,
)
from .application import app, get_manager
from .support import _GATED, _bounded, _raise_http

# --- REST API: FSM Launch/Control ---


@app.post("/api/fsm/launch", dependencies=_GATED)
async def api_fsm_launch(req: LaunchFSMRequest) -> dict[str, Any]:
    """Launch a new FSM instance."""
    mgr = get_manager()
    try:
        # launch_fsm reads preset files from disk; keep it off the event loop.
        managed = await asyncio.to_thread(
            mgr.launch_fsm,
            preset_id=req.preset_id,
            fsm_json=req.fsm_json,
            model=req.model,
            temperature=req.temperature,
            label=req.label,
        )
        return managed.to_info().model_dump()
    except Exception as e:
        _raise_http(e, "launch FSM")


@app.post("/api/fsm/{instance_id}/start", dependencies=_GATED)
async def api_fsm_start_conversation(
    instance_id: str, req: StartConversationRequest
) -> dict[str, Any]:
    """Start a new conversation on a managed FSM."""
    mgr = get_manager()
    conv_id, response = await _bounded(
        asyncio.to_thread(mgr.start_conversation, instance_id, req.initial_context),
        f"start conversation on instance {instance_id}",
        "LLM operation timed out",
    )
    return {"conversation_id": conv_id, "response": response}


@app.post("/api/fsm/{instance_id}/converse", dependencies=_GATED)
async def api_fsm_converse(instance_id: str, req: SendMessageRequest) -> dict[str, Any]:
    """Send a message to an FSM conversation."""
    mgr = get_manager()
    return await _bounded(
        asyncio.to_thread(
            mgr.send_message, instance_id, req.conversation_id, req.message
        ),
        f"send message to instance {instance_id}",
        "LLM operation timed out",
    )


@app.post("/api/fsm/{instance_id}/end", dependencies=_GATED)
async def api_fsm_end_conversation(
    instance_id: str, req: EndConversationRequest
) -> dict[str, str]:
    """End a conversation on a managed FSM."""
    mgr = get_manager()
    await _bounded(
        asyncio.to_thread(mgr.end_conversation, instance_id, req.conversation_id),
        f"end conversation on instance {instance_id}",
        "LLM operation timed out",
    )
    return {"status": "ok"}


@app.get("/api/fsm/{instance_id}/conversations", dependencies=_GATED)
async def api_fsm_conversations(instance_id: str) -> list[dict[str, Any]]:
    """List conversations on a managed FSM instance."""
    mgr = get_manager()
    try:
        snapshots = await asyncio.to_thread(mgr.get_fsm_conversations, instance_id)
        return [s.model_dump() for s in snapshots]
    except Exception as e:
        _raise_http(e, f"get conversations for instance {instance_id}")
