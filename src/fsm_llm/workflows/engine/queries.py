"""The read-only queries of ``WorkflowEngine`` (plan d454fd62 step 91; the old ``workflows/engine.py``, same dotted path).

``QueriesMixin``: ``get_instance``, ``get_definition``, ``get_status``, ``get_context``,
``list_active_instances`` and ``get_statistics``. The method bodies are the old ones, moved.
"""

from __future__ import annotations

from typing import Any

from ..definitions import WorkflowDefinition
from ..models import WorkflowInstance, WorkflowStatus
from .surface import _EngineSurface


class QueriesMixin(_EngineSurface):
    """The read-only queries of ``WorkflowEngine``."""

    def get_instance(self, instance_id: str) -> WorkflowInstance | None:
        """Get a workflow instance by ID (the live object)."""
        return self.workflow_instances.get(instance_id)

    def get_definition(self, workflow_id: str) -> WorkflowDefinition | None:
        """Get a registered workflow definition by ID."""
        return self.workflow_definitions.get(workflow_id)

    def get_status(self, instance_id: str) -> WorkflowStatus | None:
        """Get the status of a workflow instance."""
        instance = self.get_instance(instance_id)
        return instance.status if instance else None

    def get_context(self, instance_id: str) -> dict[str, Any] | None:
        """Get a shallow copy of a workflow instance's context."""
        instance = self.get_instance(instance_id)
        return dict(instance.context) if instance else None

    def list_active_instances(self) -> list[str]:
        """Get a list of active workflow instance IDs."""
        return [
            instance_id
            for instance_id, instance in self.workflow_instances.items()
            if instance.is_active()
        ]

    def get_statistics(self) -> dict[str, Any]:
        """Get workflow engine statistics."""
        statuses: dict[str, int] = {}
        for instance in self.workflow_instances.values():
            status = instance.status.value
            statuses[status] = statuses.get(status, 0) + 1

        return {
            "total_workflows": len(self.workflow_instances),
            "active_workflows": len(self.list_active_instances()),
            "registered_definitions": len(self.workflow_definitions),
            "event_listeners": sum(
                len(listeners) for listeners in self.event_listeners.values()
            ),
            "active_timers": sum(
                1 for t in self.timers.values() if t.task is None or not t.task.done()
            ),
            "buffered_events": sum(len(b) for b in self._event_buffers.values()),
            "status_breakdown": statuses,
        }
