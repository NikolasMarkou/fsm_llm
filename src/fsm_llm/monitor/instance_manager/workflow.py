"""Workflow instances of ``InstanceManager``: launch, runs, hook (plan d454fd62 step 93; the old ``monitor/instance_manager.py``, same dotted path).

``WorkflowsMixin`` holds ``launch_workflow``, the engine lifecycle hook (``_workflow_hook``, the one place a
run's bookkeeping and terminal events come from, D-006), the async start, advance, event and cancel calls and
the two status reads, with ``_status_str``. The method bodies are the old ones, moved.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from fsm_llm.logging import logger
from fsm_llm.workflows import (
    WorkflowEngine,
    WorkflowEvent,
)
from fsm_llm.workflows.constants import CONTEXT_KEY_CANCELLATION_REASON

from ..collector import redact_context
from ..constants import (
    EVENT_WORKFLOW_ADVANCED,
    EVENT_WORKFLOW_CANCELLED,
    EVENT_WORKFLOW_COMPLETED,
    EVENT_WORKFLOW_EVENT_DELIVERED,
    EVENT_WORKFLOW_FAILED,
    EVENT_WORKFLOW_STARTED,
    TERMINAL_STATUSES,
)
from ..definitions import (
    MonitorEvent,
)
from .managed import ManagedWorkflow, _new_instance_id
from .surface import _InstanceSurface
from .workflow_presets import _WORKFLOW_PRESETS


def _status_str(status: Any) -> str:
    """Extract a string from a status value (may be an enum or plain string)."""
    return status.value if hasattr(status, "value") else str(status)


class WorkflowsMixin(_InstanceSurface):
    """Launching workflow engines and driving their runs."""

    def get_workflow_presets(self) -> list[dict[str, str]]:
        """List built-in workflow presets available for launch."""
        presets: list[dict[str, str]] = []
        for preset_id in sorted(_WORKFLOW_PRESETS):
            try:
                definition = _WORKFLOW_PRESETS[preset_id]()
                presets.append(
                    {
                        "id": preset_id,
                        "name": definition.name,
                        "description": definition.description,
                    }
                )
            except Exception as e:  # pragma: no cover - defensive
                logger.debug(f"Failed to build workflow preset {preset_id}: {e}")
        return presets

    def launch_workflow(
        self,
        preset_id: str | None = None,
        definition_json: dict[str, Any] | None = None,
        label: str = "",
    ) -> ManagedWorkflow:
        """Launch a workflow engine seeded with a built-in preset definition.

        The returned :class:`ManagedWorkflow` has its ``engine`` populated and the
        preset's definition registered, with ``workflow_id`` set. Callers start an
        instance via :meth:`start_workflow_instance`.

        Arbitrary ``definition_json`` is intentionally unsupported (see
        DECISION plan_2026-05-29_0c00a594/D-002 [STALE]).
        """
        if definition_json is not None and preset_id is None:
            raise ValueError(
                "Custom workflow JSON is not supported via the dashboard; "
                "choose a built-in workflow preset instead. "
                f"Available: {', '.join(sorted(_WORKFLOW_PRESETS))}"
            )
        if preset_id is None:
            raise ValueError(
                "A workflow preset_id is required. "
                f"Available: {', '.join(sorted(_WORKFLOW_PRESETS))}"
            )
        if preset_id not in _WORKFLOW_PRESETS:
            raise ValueError(
                f"Unknown workflow preset: {preset_id}. "
                f"Available: {', '.join(sorted(_WORKFLOW_PRESETS))}"
            )

        instance_id = _new_instance_id()
        engine = WorkflowEngine()
        definition = _WORKFLOW_PRESETS[preset_id]()
        engine.register_workflow(definition)

        managed = ManagedWorkflow(
            instance_id=instance_id,
            label=label or f"Workflow-{instance_id[:6]}",
            source=preset_id,
        )
        managed.engine = engine
        managed.workflow_id = definition.workflow_id
        try:
            managed.loop = asyncio.get_running_loop()
        except RuntimeError:
            managed.loop = None
        engine.add_hook(self._workflow_hook(managed))
        self._register_instance(managed, self._new_collector(), "Workflow")
        return managed

    def _workflow_hook(self, managed: ManagedWorkflow) -> Any:
        """Engine lifecycle hook: one ``workflow_advanced`` event per completed
        step (to the global and the instance collector) and the terminal
        bookkeeping for every run, however it ended (API call, timer, event).

        # DECISION plan-2026-09-28T090000-3c9e41d2/D-006
        # Run completion is tracked HERE, from the engine's status changes.
        # Do NOT go back to checking the status after advance/cancel/event
        # calls: runs that ended any other way (a timer, a failure during
        # start, an event) stayed "active" forever and step metrics counted
        # API calls instead of steps.
        """
        instance_id = managed.instance_id

        _record = self._record_event

        def _hook(event_name: str, wf_instance: Any, data: dict[str, Any]) -> None:
            wf_id = str(getattr(wf_instance, "instance_id", ""))
            if event_name == "step_completed":
                _record(
                    MonitorEvent(
                        event_type=EVENT_WORKFLOW_ADVANCED,
                        message=f"Workflow step {data.get('step_id', '')}: {wf_id}",
                        data={
                            "instance_id": instance_id,
                            "workflow_instance_id": wf_id,
                            "step_id": data.get("step_id", ""),
                            "success": data.get("success", True),
                        },
                        level="INFO" if data.get("success", True) else "WARNING",
                    )
                )
                return
            if event_name != "status_changed":
                return
            status = str(data.get("status", ""))
            with self._lock:
                if wf_id in managed.run_statuses:
                    managed.run_statuses[wf_id] = status
                if status in TERMINAL_STATUSES:
                    if wf_id in managed.active_instance_ids:
                        managed.active_instance_ids.remove(wf_id)
                managed.refresh_status()
            terminal_events = {
                "completed": (EVENT_WORKFLOW_COMPLETED, "INFO"),
                "failed": (EVENT_WORKFLOW_FAILED, "ERROR"),
                "cancelled": (EVENT_WORKFLOW_CANCELLED, "INFO"),
            }
            if status in terminal_events:
                event_type, level = terminal_events[status]
                event_data: dict[str, Any] = {
                    "instance_id": instance_id,
                    "workflow_instance_id": wf_id,
                }
                if status == "cancelled":
                    context = getattr(wf_instance, "context", {}) or {}
                    event_data["reason"] = context.get(
                        CONTEXT_KEY_CANCELLATION_REASON, ""
                    )
                if status == "failed":
                    event_data["error"] = str(data.get("error") or "")[:500]
                _record(
                    MonitorEvent(
                        event_type=event_type,
                        message=f"Workflow {status}: {wf_id}",
                        data=event_data,
                        level=level,
                    )
                )

        return _hook

    async def start_workflow_instance(
        self,
        instance_id: str,
        workflow_id: str,
        initial_context: dict[str, Any] | None = None,
    ) -> str:
        """Start a workflow run on a managed workflow engine.

        The run is tracked as active before it starts; the engine hook
        removes it (and emits completed/failed/cancelled) when it ends, which
        for pure auto/condition presets happens inside the engine's ``start``.
        """
        inst = self._get_workflow(instance_id)
        # The engine's timers and background runs live on THIS loop. The
        # launch route builds the instance in a worker thread (no loop there),
        # so record the owning loop here for ``_teardown``.
        inst.loop = asyncio.get_running_loop()
        wf_instance_id = str(uuid.uuid4())
        with self._lock:
            inst.run_ids.append(wf_instance_id)
            inst.run_statuses[wf_instance_id] = "running"
            inst.active_instance_ids.append(wf_instance_id)
            inst.refresh_status()
        self._emit_event(
            EVENT_WORKFLOW_STARTED,
            message=f"Workflow instance started: {wf_instance_id}",
            data={
                "instance_id": instance_id,
                "workflow_instance_id": wf_instance_id,
            },
        )
        try:
            await inst.engine.start(
                workflow_id=workflow_id,
                initial_context=initial_context or {},
                instance_id=wf_instance_id,
            )
        finally:
            # A run that failed before the engine registered it never reports
            # a status change: settle it here.
            if inst.engine.get_instance(wf_instance_id) is None:
                with self._lock:
                    inst.run_statuses[wf_instance_id] = "failed"
                    if wf_instance_id in inst.active_instance_ids:
                        inst.active_instance_ids.remove(wf_instance_id)
                    inst.refresh_status()
        return wf_instance_id

    def _validate_workflow_instance_id(
        self, inst: ManagedWorkflow, instance_id: str, wf_instance_id: str
    ) -> None:
        """Validate that a workflow instance ID belongs to the managed instance."""
        if wf_instance_id not in inst.active_instance_ids:
            raise KeyError(
                f"Workflow instance '{wf_instance_id}' not found in "
                f"managed instance '{instance_id}'"
            )

    async def advance_workflow(
        self,
        instance_id: str,
        wf_instance_id: str,
        user_input: str = "",
    ) -> bool:
        """Advance a workflow run (step and completion events come from the
        engine hook)."""
        inst = self._get_workflow(instance_id)
        self._validate_workflow_instance_id(inst, instance_id, wf_instance_id)
        result = await inst.engine.advance(wf_instance_id, user_input)
        return bool(result)

    async def send_workflow_event(
        self,
        instance_id: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
        wf_instance_id: str | None = None,
    ) -> list[str]:
        """Deliver an event to a managed workflow engine.

        With ``wf_instance_id`` the event targets that workflow instance only
        (and is buffered for it if it is not waiting yet); otherwise it is
        broadcast to every instance waiting for ``event_type``. Returns the
        workflow instance ids the event woke.
        """
        inst = self._get_workflow(instance_id)
        if wf_instance_id:
            self._validate_workflow_instance_id(inst, instance_id, wf_instance_id)
        event = WorkflowEvent(
            event_type=event_type,
            payload=payload or {},
            instance_id=wf_instance_id or None,
        )
        affected = await inst.engine.process_event(event)

        self._emit_event(
            EVENT_WORKFLOW_EVENT_DELIVERED,
            message=f"Workflow event delivered: {event_type}",
            data={
                "instance_id": instance_id,
                "event_type": event_type,
                "affected": list(affected),
            },
        )
        return list(affected)

    async def cancel_workflow(
        self,
        instance_id: str,
        wf_instance_id: str,
        reason: str = "",
    ) -> bool:
        """Cancel a workflow instance."""
        inst = self._get_workflow(instance_id)
        self._validate_workflow_instance_id(inst, instance_id, wf_instance_id)
        result = await inst.engine.cancel(wf_instance_id, reason=reason)
        return bool(result)

    def get_workflow_status(
        self, instance_id: str, wf_instance_id: str
    ) -> dict[str, Any]:
        """Get workflow instance status and context."""
        inst = self._get_workflow(instance_id)
        # Accept ids the engine still knows about (e.g. just-completed/cancelled
        # instances removed from active_instance_ids) so status stays queryable;
        # only fall back to the active-list validator if the engine has no record.
        wf_instance = inst.engine.get_instance(wf_instance_id)
        if wf_instance is None:
            self._validate_workflow_instance_id(inst, instance_id, wf_instance_id)
            raise KeyError(f"Workflow run not found: {wf_instance_id}")
        drop_internal = not self.config.show_internal_keys
        status = inst.engine.get_status(wf_instance_id)
        context = inst.engine.get_context(wf_instance_id)
        status_str = _status_str(status).lower()

        # Extract step history if available
        history: list[dict[str, Any]] = []
        raw_history = getattr(wf_instance, "history", [])
        for entry in raw_history:
            history.append(
                {
                    "step_id": getattr(entry, "step_id", ""),
                    "message": getattr(entry, "message", ""),
                    "timestamp": str(getattr(entry, "timestamp", "")),
                    "data": redact_context(
                        getattr(entry, "data", None) or {}, drop_internal
                    ),
                    "error": getattr(entry, "error", None),
                }
            )

        return {
            "workflow_instance_id": wf_instance_id,
            "status": status_str,
            "current_step": getattr(wf_instance, "current_step_id", ""),
            "context": redact_context(context or {}, drop_internal),
            "history": history,
            "created_at": str(getattr(wf_instance, "created_at", "")),
            "updated_at": str(getattr(wf_instance, "updated_at", "")),
        }

    def get_workflow_instances(self, instance_id: str) -> list[dict[str, Any]]:
        """List all workflow instances on a managed workflow engine."""
        inst = self._get_workflow(instance_id)
        results: list[dict[str, Any]] = []
        for wf_id in inst.active_instance_ids:
            try:
                status_data = self.get_workflow_status(instance_id, wf_id)
                results.append(status_data)
            except Exception as e:
                logger.debug(f"Failed to get workflow status for {wf_id}: {e}")
                results.append({"workflow_instance_id": wf_id, "status": "unknown"})
        return results
