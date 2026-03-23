from __future__ import annotations

"""
Core data models for the FSM-LLM Workflow System.
"""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class WorkflowStatus(str, Enum):
    """Status of a workflow instance."""

    PENDING = "pending"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Valid status transitions — prevents illegal state changes like COMPLETED→RUNNING
_VALID_STATUS_TRANSITIONS: dict[WorkflowStatus, set[WorkflowStatus]] = {
    WorkflowStatus.PENDING: {
        WorkflowStatus.RUNNING,
        WorkflowStatus.FAILED,
        WorkflowStatus.CANCELLED,
    },
    WorkflowStatus.RUNNING: {
        WorkflowStatus.WAITING,
        WorkflowStatus.COMPLETED,
        WorkflowStatus.FAILED,
        WorkflowStatus.CANCELLED,
    },
    WorkflowStatus.WAITING: {
        WorkflowStatus.RUNNING,
        WorkflowStatus.FAILED,
        WorkflowStatus.CANCELLED,
    },
    WorkflowStatus.COMPLETED: set(),
    WorkflowStatus.FAILED: set(),
    WorkflowStatus.CANCELLED: set(),
}


class WorkflowEvent(BaseModel):
    """Represents an event that can trigger workflow transitions."""

    event_type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    model_config = ConfigDict()

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Custom serialization with datetime handling."""
        data = super().model_dump(**kwargs)
        if isinstance(data.get("timestamp"), datetime):
            data["timestamp"] = data["timestamp"].isoformat()
        return data


class WorkflowStepResult(BaseModel):
    """Result of a workflow step execution."""

    success: bool
    data: dict[str, Any] = Field(default_factory=dict)
    next_state: str | None = None
    message: str | None = None
    error: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("error", mode="before")
    @classmethod
    def convert_exception_to_string(cls, v):
        """Convert Exception objects to error strings."""
        if isinstance(v, Exception):
            return str(v)
        return v

    @classmethod
    def success_result(
        cls,
        data: dict[str, Any] | None = None,
        next_state: str | None = None,
        message: str | None = None,
    ) -> WorkflowStepResult:
        """Create a successful result."""
        return cls(
            success=True, data=data or {}, next_state=next_state, message=message
        )

    @classmethod
    def failure_result(
        cls, error: str, next_state: str | None = None, message: str | None = None
    ) -> WorkflowStepResult:
        """Create a failure result."""
        return cls(
            success=False,
            error=error,
            next_state=next_state,
            message=message or f"Step failed: {error}",
        )


class WorkflowHistoryEntry(BaseModel):
    """An entry in the workflow execution history."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    step_id: str
    status: str
    message: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class WorkflowInstance(BaseModel):
    """Runtime instance of a workflow."""

    instance_id: str
    workflow_id: str
    current_step_id: str
    context: dict[str, Any] = Field(default_factory=dict)
    status: WorkflowStatus = WorkflowStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    deadline: datetime | None = None
    error: str | None = None
    history: list[WorkflowHistoryEntry] = Field(default_factory=list)

    def update_status(
        self, status: WorkflowStatus, error: Exception | None = None
    ) -> None:
        """Update the workflow status."""
        from .exceptions import WorkflowStateError

        allowed = _VALID_STATUS_TRANSITIONS.get(self.status, set())
        if status != self.status and status not in allowed:
            raise WorkflowStateError(
                current_state=self.current_step_id,
                operation="update_status",
                message=f"Invalid status transition: {self.status.value} → {status.value}",
            )
        self.status = status
        self.updated_at = datetime.now(timezone.utc)

        if status in [
            WorkflowStatus.COMPLETED,
            WorkflowStatus.FAILED,
            WorkflowStatus.CANCELLED,
        ]:
            self.completed_at = datetime.now(timezone.utc)

        if error:
            self.error = str(error)

        # Add to history
        self.add_history_entry(
            step_id=self.current_step_id,
            message=f"Status changed to {status.value}",
            data={"status": status.value, "error": str(error) if error else None},
        )

    def add_history_entry(
        self, step_id: str, message: str, data: dict[str, Any] | None = None
    ) -> None:
        """Add an entry to the workflow history."""
        self.history.append(
            WorkflowHistoryEntry(
                step_id=step_id,
                status=self.status.value,
                message=message,
                data=data or {},
            )
        )

    def is_active(self) -> bool:
        """Check if the workflow is in an active state."""
        return self.status in [WorkflowStatus.RUNNING, WorkflowStatus.WAITING]

    def is_terminal(self) -> bool:
        """Check if the workflow is in a terminal state."""
        return self.status in [
            WorkflowStatus.COMPLETED,
            WorkflowStatus.FAILED,
            WorkflowStatus.CANCELLED,
        ]


class EventListener(BaseModel):
    """Information about a workflow instance listening for an event."""

    instance_id: str
    success_state: str
    event_mapping: dict[str, str] = Field(default_factory=dict)
    registered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    timeout_at: datetime | None = None

    def is_expired(self) -> bool:
        """Check if the event listener has expired."""
        return (
            self.timeout_at is not None and datetime.now(timezone.utc) > self.timeout_at
        )


class WaitEventConfig(BaseModel):
    """Configuration for waiting for an event."""

    event_type: str
    success_state: str
    timeout_seconds: int | None = None
    timeout_state: str | None = None
    event_mapping: dict[str, str] = Field(default_factory=dict)

    @field_validator("timeout_seconds")
    @classmethod
    def validate_timeout(cls, v):
        if v is not None and v <= 0:
            raise ValueError("Timeout must be positive")
        return v
