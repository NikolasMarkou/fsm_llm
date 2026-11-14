"""The ``/ws`` dashboard push."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from fsm_llm.context.filtering import redacting_json_default
from fsm_llm.logging import logger

from ..instance_manager import InstanceManager
from .application import _ws_authorized, app, get_manager
from .routes_builder import _cleanup_stale_builder_sessions

# --- WebSocket for real-time updates ---


_AGENT_UPDATE_KEYS = (
    "status",
    "current_state",
    "iteration_count",
    "last_tool_call",
    "transition_count",
)


def _status_or_none(mgr: InstanceManager, instance_id: str) -> dict[str, Any] | None:
    """An agent's status, or None when it is gone or unreadable (a destroyed
    instance mid-poll is expected; anything else is logged)."""
    try:
        return mgr.get_agent_status(instance_id)
    except KeyError:
        return None
    except Exception as e:
        logger.debug(f"Failed to get agent status for {instance_id}: {e}")
        return None


def _agent_update(status: dict[str, Any]) -> dict[str, Any]:
    """What the tick ships of an agent's status: its progress fields and the
    size of its log, so the page can tell when to refetch."""
    return {
        **{k: status[k] for k in _AGENT_UPDATE_KEYS if k in status},
        "log_entries": len(status.get("conversation_log", ())),
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    if not await _ws_authorized(websocket):
        return
    mgr = get_manager()
    event_cursor = mgr.global_collector.get_metrics().total_events
    log_cursor = mgr.global_collector.total_logs
    # Replay a little history to a new client.
    event_cursor = max(0, event_cursor - 50)
    log_cursor = max(0, log_cursor - 50)
    last_dashboard_config_version = mgr.dashboard_config_version

    try:
        _cleanup_counter = 0
        while True:
            await asyncio.sleep(get_manager().config.refresh_interval)
            # Re-read the manager each cycle so a re-configure() takes effect.
            current = get_manager()
            if current is not mgr:
                mgr = current
                event_cursor = mgr.global_collector.get_metrics().total_events
                log_cursor = mgr.global_collector.total_logs
            # Periodic cleanup of stale builder sessions (every ~60 cycles)
            _cleanup_counter += 1
            if _cleanup_counter >= 60:
                _cleanup_counter = 0
                await _cleanup_stale_builder_sessions()
            metrics = mgr.get_metrics()

            data: dict[str, Any] = {
                "type": "metrics",
                "data": metrics.model_dump(),
            }

            # DECISION plan-2026-09-28T090000-3c9e41d2/D-004
            # Oldest-first slices from a cursor taken with the slice; sent
            # newest-first as before. Bursts beyond 50 continue next cycle.
            events, event_cursor = mgr.global_collector.events_after(
                event_cursor, limit=50
            )
            if events:
                data["events"] = [e.model_dump() for e in reversed(events)]

            logs, log_cursor = mgr.global_collector.logs_after(log_cursor, limit=50)
            if logs:
                data["logs"] = [r.model_dump() for r in reversed(logs)]

            # Always include instance list so status changes propagate
            instances = mgr.list_instances()
            data["instances"] = [i.model_dump() for i in instances]

            # A change signal per running agent (the page refetches the status
            # of the one it shows) and the ids of running workflows (the page
            # only needs to know one is running); never the logs or the status.
            agent_updates = {
                i.instance_id: _agent_update(status)
                for i in instances
                if i.instance_type == "agent" and i.status == "running"
                if (status := _status_or_none(mgr, i.instance_id)) is not None
            }
            if agent_updates:
                data["agent_updates"] = agent_updates
            running_workflows = [
                i.instance_id
                for i in instances
                if i.instance_type == "workflow" and i.status == "running"
            ]
            if running_workflows:
                data["workflow_updates"] = running_workflows

            # The page refetches /api/dashboard/config when the version moves.
            if mgr.dashboard_config_version != last_dashboard_config_version:
                last_dashboard_config_version = mgr.dashboard_config_version
                data["dashboard_config_version"] = last_dashboard_config_version

            # DECISION plan-2026-09-22T080837-8b258a25/D-008
            # `default=str` pushed an arbitrary object's `__str__` (a secret in
            # a context value) to every dashboard browser. Share the core hook;
            # do NOT re-implement the scalar keep-set here.
            await websocket.send_text(json.dumps(data, default=redacting_json_default))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug(f"WebSocket error: {e}")
        try:
            await websocket.close()
        except Exception as close_err:
            logger.debug(f"WebSocket close failed: {close_err}")
