"""Routes: metrics, conversations, activity, events, logs, the monitor and dashboard
configuration, the instance list and the instance detail, events and destroy."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import HTTPException
from pydantic import ValidationError

from fsm_llm import __version__ as core_version
from fsm_llm.logging import logger

from .. import eventlog
from ..__version__ import __version__
from ..definitions import DashboardAlert, DashboardConfig, DashboardPanel, MonitorConfig
from ..exceptions import MonitorError
from .application import app, get_manager
from .support import _GATED, _bounded, _raise_http

# --- REST API: Monitoring ---


@app.get("/api/metrics")
async def api_metrics() -> dict[str, Any]:
    mgr = get_manager()
    try:
        metrics = mgr.get_metrics()
    except MonitorError as e:
        _raise_http(e, "read metrics")
    return metrics.model_dump()


# DECISION plan-2026-09-28T090000-3c9e41d2/D-005
# Conversation snapshots wait on the conversation's turn lock (a whole LLM
# call). Do NOT call them on the event loop: every route and /ws froze for the
# length of a turn. They run in a worker thread.


@app.get("/api/conversations", dependencies=_GATED)
async def api_conversations(
    include_ended: bool = True,
) -> list[dict[str, Any]]:
    mgr = get_manager()
    try:
        snapshots = await asyncio.to_thread(
            mgr.get_all_conversation_snapshots, include_ended=include_ended
        )
    except MonitorError as e:
        _raise_http(e, "list conversations")
    return [s.model_dump() for s in snapshots]


@app.get("/api/conversations/{conversation_id}", dependencies=_GATED)
async def api_conversation(conversation_id: str) -> dict[str, Any]:
    mgr = get_manager()
    snap = await asyncio.to_thread(mgr.get_conversation_snapshot, conversation_id)
    if snap is None:
        raise HTTPException(status_code=404, detail="not found")
    return snap.model_dump()


@app.get("/api/activity", dependencies=_GATED)
async def api_activity(
    include_ended: bool = True,
) -> list[dict[str, Any]]:
    """Get unified activity list: FSM conversations, agent tasks, workflow instances."""
    mgr = get_manager()
    try:
        items = await asyncio.to_thread(
            mgr.get_all_activity_snapshots, include_ended=include_ended
        )
    except MonitorError as e:
        _raise_http(e, "list activity")
    return [item.model_dump() for item in items]


@app.get("/api/events", dependencies=_GATED)
async def api_events(limit: int = 50) -> list[dict[str, Any]]:
    mgr = get_manager()
    try:
        events = mgr.get_events(limit=limit)
    except MonitorError as e:
        _raise_http(e, "list events")
    return [e.model_dump() for e in events]


@app.get("/api/logs", dependencies=_GATED)
async def api_logs(limit: int = 100, level: str = "INFO") -> list[dict[str, Any]]:
    mgr = get_manager()
    try:
        logs = mgr.global_collector.get_logs(limit=limit, level=level)
    except MonitorError as e:
        _raise_http(e, "list logs")
    return [r.model_dump() for r in logs]


@app.get("/api/config")
async def api_config_get() -> dict[str, Any]:
    mgr = get_manager()
    return mgr.config.model_dump()


@app.post("/api/config", dependencies=_GATED)
async def api_config_set(config: MonitorConfig) -> dict[str, str]:
    mgr = get_manager()
    mgr.config = config
    return {"status": "ok"}


# --- REST API: Custom Dashboard Config ---


@app.get("/api/dashboard/config")
async def api_dashboard_config_get() -> dict[str, Any]:
    """Get the active custom dashboard configuration."""
    mgr = get_manager()
    cfg = mgr.dashboard_config
    if cfg is None:
        return {"active": False, "config": None}
    return {"active": True, "config": cfg.model_dump()}


def _parse_dashboard_config(req: dict[str, Any]) -> Any:
    """Build a DashboardConfig from MonitorBuilder ``to_dict()`` output.

    Accepts the output itself (``{"name", "panels", "alerts", "config":
    {refresh_interval_seconds, retention_hours}}``) or that output wrapped as
    ``{"config": <output>}``. Raises ``ValueError`` on a malformed body.
    """
    wrapped = req.get("config")
    is_wrapped = (
        isinstance(wrapped, dict)
        and set(req) <= {"config"}
        and any(k in wrapped for k in ("name", "panels", "alerts", "description"))
    )
    raw = wrapped if is_wrapped else req
    if not isinstance(raw, dict):
        raise ValueError("dashboard config must be an object")

    panels_raw = raw.get("panels") or {}
    alerts_raw = raw.get("alerts") or {}
    settings = raw.get("config") or {}
    if not isinstance(panels_raw, dict) or not isinstance(alerts_raw, dict):
        raise ValueError("'panels' and 'alerts' must be objects keyed by id")
    if not isinstance(settings, dict) or {"panels", "alerts"} & settings.keys():
        # The whole GET answer (`{"active", "config": <dashboard>}`) has a
        # dashboard where the settings go: refuse it, never store it empty.
        raise ValueError("'config' must be an object of settings, not a dashboard")

    try:
        # Defaults are the models' own; the id comes from the key.
        panels = [
            DashboardPanel.model_validate(
                {"title": str(pid), **pdata, "panel_id": str(pid)}
            )
            for pid, pdata in panels_raw.items()
        ]
        alerts = [
            DashboardAlert.model_validate({**adata, "alert_id": str(aid)})
            for aid, adata in alerts_raw.items()
        ]
        return DashboardConfig(
            name=raw.get("name", ""),
            description=raw.get("description", ""),
            panels=panels,
            alerts=alerts,
            refresh_interval_seconds=settings.get("refresh_interval_seconds", 30),
            retention_hours=settings.get("retention_hours", 24),
        )
    except (ValidationError, TypeError) as e:
        raise ValueError(f"invalid dashboard config: {e}") from e


@app.post("/api/dashboard/config", dependencies=_GATED)
async def api_dashboard_config_set(req: dict[str, Any]) -> dict[str, str]:
    """Apply a custom dashboard config from MonitorBuilder output (either the
    ``to_dict()`` output itself or wrapped as ``{"config": ...}``)."""
    mgr = get_manager()
    try:
        cfg = _parse_dashboard_config(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    mgr.dashboard_config = cfg
    logger.info(
        f"Dashboard config applied: {cfg.name} "
        f"({len(cfg.panels)} panels, {len(cfg.alerts)} alerts)"
    )
    return {"status": "ok"}


@app.delete("/api/dashboard/config", dependencies=_GATED)
async def api_dashboard_config_delete() -> dict[str, str]:
    """Remove the custom dashboard configuration."""
    mgr = get_manager()
    mgr.dashboard_config = None
    return {"status": "ok"}


@app.get("/api/info")
async def api_info() -> dict[str, str]:
    return {"monitor_version": __version__, "fsm_llm_version": core_version}


# --- REST API: Capabilities ---


@app.get("/api/capabilities")
async def api_capabilities() -> dict[str, Any]:
    """Return which extensions are installed and the launchable agent types."""
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


@app.get("/api/instances/{instance_id}/events", dependencies=_GATED)
async def api_instance_events(
    instance_id: str, limit: int = 50
) -> list[dict[str, Any]]:
    """Get events for a specific managed instance."""
    mgr = get_manager()
    collector = mgr.get_instance_collector(instance_id)
    if collector is None:
        raise HTTPException(status_code=404, detail="instance not found")
    events = collector.get_events(limit=limit)
    return [e.model_dump() for e in events]


@app.delete("/api/instances/{instance_id}", dependencies=_GATED)
async def api_instance_destroy(instance_id: str) -> dict[str, str]:
    """Destroy a managed instance."""
    mgr = get_manager()
    try:
        # DECISION plan-2026-09-12T065608-089d0ec7/D-007: end_instance can
        # now block for up to its bounded agent-teardown join timeout (see
        # instance_manager/lifecycle.py); keep it off the event loop the same way the
        # launch routes already do, rather than shrinking the join timeout to
        # sub-second (which would make the teardown warning fire on every
        # merely-slow agent, not just a genuinely hung one).
        await asyncio.to_thread(mgr.end_instance, instance_id)
        return {"status": "ok"}
    except Exception as e:
        _raise_http(e, f"destroy instance {instance_id}")


# --- REST API: event log browse (read only) ---


@app.get("/api/eventlog", dependencies=_GATED)
async def api_eventlog_list() -> dict[str, Any]:
    """The logs of the event log the dashboard was given (none: ``available`` false)."""
    log = get_manager().event_log
    if log is None:
        return {"available": False, "conversations": []}
    return {
        "available": True,
        "conversations": await asyncio.to_thread(eventlog.list_logs, log),
    }


@app.get("/api/eventlog/{conversation_id}", dependencies=_GATED)
async def api_eventlog_timeline(
    conversation_id: str, since_seq: int = 0, limit: int = 200
) -> dict[str, Any]:
    """One log as whitelisted events (no blob, no ``*_ref``) plus its usage fold."""
    log = get_manager().event_log
    if log is None:
        raise HTTPException(status_code=404, detail="no event log")
    return await _bounded(
        asyncio.to_thread(
            eventlog.timeline, log, conversation_id, since_seq=since_seq, limit=limit
        ),
        f"read event log {conversation_id}",
        "event log read timed out",
    )
