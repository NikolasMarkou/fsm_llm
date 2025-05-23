"""
Core data models for the LLM-FSM Workflow System.
"""

import uuid
from enum import Enum
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict

from .exceptions import WorkflowValidationError


class WorkflowStatus(str, Enum):
    """Status of a workflow instance."""
    PENDING = "pending"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowEvent(BaseModel):
    """Represents an event that can trigger workflow transitions."""
    event_type: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    model_config = ConfigDict()

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Custom serialization with datetime handling."""
        data = super().model_dump(**kwargs)
        if isinstance(data.get('timestamp'), datetime):
            data['timestamp'] = data['timestamp'].isoformat()
        return data


class WorkflowStepResult(BaseModel):
    """Result of a workflow step execution."""
    success: bool
    data: Dict[str, Any] = Field(default_factory=dict)
    next_state: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator('error', mode='before')
    @classmethod
    def convert_exception_to_string(cls, v):
        """Convert Exception objects to error strings."""
        if isinstance(v, Exception):
            return str(v)
        return v

    @classmethod
    def success_result(cls, data: Dict[str, Any] = None, next_state: str = None,
                      message: str = None) -> 'WorkflowStepResult':
        """Create a successful result."""
        return cls(
            success=True,
            data=data or {},
            next_state=next_state,
            message=message
        )

    @classmethod
    def failure_result(cls, error: str, next_state: str = None,
                      message: str = None) -> 'WorkflowStepResult':
        """Create a failure result."""
        return cls(
            success=False,
            error=error,
            next_state=next_state,
            message=message or f"Step failed: {error}"
        )


class WorkflowHistoryEntry(BaseModel):
    """An entry in the workflow execution history."""
    timestamp: datetime = Field(default_factory=datetime.now)
    step_id: str
    status: str
    message: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class WorkflowInstance(BaseModel):
    """Runtime instance of a workflow."""
    instance_id: str
    workflow_id: str
    current_step_id: str
    context: Dict[str, Any] = Field(default_factory=dict)
    status: WorkflowStatus = WorkflowStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    history: List[WorkflowHistoryEntry] = Field(default_factory=list)

    def update_status(self, status: WorkflowStatus, error: Optional[Exception] = None) -> None:
        """Update the workflow status."""
        self.status = status
        self.updated_at = datetime.now()

        if status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]:
            self.completed_at = datetime.now()

        if error:
            self.error = str(error)

        # Add to history
        self.add_history_entry(
            step_id=self.current_step_id,
            message=f"Status changed to {status.value}",
            data={"status": status.value, "error": str(error) if error else None}
        )

    def add_history_entry(self, step_id: str, message: str, data: Dict[str, Any] = None) -> None:
        """Add an entry to the workflow history."""
        self.history.append(WorkflowHistoryEntry(
            step_id=step_id,
            status=self.status.value,
            message=message,
            data=data or {}
        ))

    def is_active(self) -> bool:
        """Check if the workflow is in an active state."""
        return self.status in [WorkflowStatus.RUNNING, WorkflowStatus.WAITING]

    def is_terminal(self) -> bool:
        """Check if the workflow is in a terminal state."""
        return self.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]


class EventListener(BaseModel):
    """Information about a workflow instance listening for an event."""
    instance_id: str
    success_state: str
    event_mapping: Dict[str, str] = Field(default_factory=dict)
    registered_at: datetime = Field(default_factory=datetime.now)
    timeout_at: Optional[datetime] = None

    def is_expired(self) -> bool:
        """Check if the event listener has expired."""
        return self.timeout_at is not None and datetime.now() > self.timeout_at


class WaitEventConfig(BaseModel):
    """Configuration for waiting for an event."""
    event_type: str
    success_state: str
    timeout_seconds: Optional[int] = None
    timeout_state: Optional[str] = None
    event_mapping: Dict[str, str] = Field(default_factory=dict)

    @field_validator('timeout_seconds')
    @classmethod
    def validate_timeout(cls, v):
        if v is not None and v <= 0:
            raise ValueError("Timeout must be positive")
        return v