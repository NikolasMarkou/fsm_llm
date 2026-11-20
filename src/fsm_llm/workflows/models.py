"""
Core data models for the FSM-LLM Workflow System.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from fsm_llm.constants import MAX_CONTEXT_FILTER_DEPTH, has_internal_prefix
from fsm_llm.context.filtering import ContextFilterError, filter_context_tree
from fsm_llm.core.codes import Reason, Status
from fsm_llm.core.results import RunResult
from fsm_llm.security import is_forbidden_context_entry

from .constants import DEFAULT_MAX_HISTORY_ENTRIES
from .exceptions import WorkflowStateError


class WorkflowStatus(str, Enum):
    """Status of a workflow instance."""

    PENDING = "pending"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


#: Statuses an instance never leaves.
_TERMINAL_STATUSES = frozenset(
    {WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED}
)

#: How each workflow status reads on the shared ``RunResult`` envelope (D-047).
#: ``Status`` has no waiting or pending member: both read ``running`` and the
#: workflow status itself stays in ``RunResult.detail["workflow_status"]``.
_RUN_STATUS: dict[WorkflowStatus, tuple[Status, Reason | None]] = {
    WorkflowStatus.PENDING: (Status.RUNNING, None),
    WorkflowStatus.RUNNING: (Status.RUNNING, None),
    WorkflowStatus.WAITING: (Status.RUNNING, None),
    WorkflowStatus.COMPLETED: (Status.OK, None),
    WorkflowStatus.FAILED: (Status.FAILED, None),
    WorkflowStatus.CANCELLED: (Status.ABORTED, Reason.CANCELLED),
}

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
        WorkflowStatus.COMPLETED,
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
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    instance_id: str | None = None
    """Deliver only to this workflow instance. ``None`` (the default) delivers
    to every instance whose wait matches the event (broadcast). A targeted
    event that arrives before the instance waits for it is buffered for that
    instance and consumed by its next matching wait."""

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
    next_step_id: str | None = None
    message: str | None = None
    error: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

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
        next_step_id: str | None = None,
        message: str | None = None,
    ) -> WorkflowStepResult:
        """Create a successful result."""
        return cls(
            success=True, data=data or {}, next_step_id=next_step_id, message=message
        )

    @classmethod
    def failure_result(
        cls, error: str, next_step_id: str | None = None, message: str | None = None
    ) -> WorkflowStepResult:
        """Create a failure result."""
        return cls(
            success=False,
            error=error,
            next_step_id=next_step_id,
            message=message or f"Step failed: {error}",
        )


class WorkflowHistoryEntry(BaseModel):
    """An entry in the workflow execution history."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    step_id: str
    status: str
    message: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class WorkflowInstance(BaseModel):
    """Runtime instance of a workflow."""

    model_config = ConfigDict(extra="forbid")

    instance_id: str
    workflow_id: str
    current_step_id: str
    context: dict[str, Any] = Field(default_factory=dict)
    status: WorkflowStatus = WorkflowStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    deadline: datetime | None = None
    max_seconds: float | None = None
    """Configured total time limit (seconds) for the whole workflow, if any.
    Reported as ``WorkflowTimeoutError.timeout_seconds`` on deadline breach so
    the value reflects the configured timeout, not the construction-to-start gap."""
    error: str | None = None
    history: list[WorkflowHistoryEntry] = Field(default_factory=list)
    max_history_entries: int | None = DEFAULT_MAX_HISTORY_ENTRIES
    """Keep at most this many history entries (oldest dropped first).
    ``None`` keeps every entry."""

    def update_status(
        self, status: WorkflowStatus, error: Exception | None = None
    ) -> None:
        """Update the workflow status.

        Re-setting the current status is allowed; it records a history entry
        only when it carries an error (so a RUNNING -> RUNNING step transition
        does not add a redundant "Status changed" entry).
        """
        allowed = _VALID_STATUS_TRANSITIONS.get(self.status, set())
        if status != self.status and status not in allowed:
            raise WorkflowStateError(
                f"Invalid status transition: {self.status.value} → {status.value}",
                current_step_id=self.current_step_id,
                operation="update_status",
            )
        changed = status != self.status
        self.status = status
        self.updated_at = datetime.now(UTC)

        if changed and status in _TERMINAL_STATUSES:
            self.completed_at = datetime.now(UTC)

        if error:
            self.error = str(error)

        if changed or error:
            self.add_history_entry(
                step_id=self.current_step_id,
                message=f"Status changed to {status.value}",
                data={"status": status.value, "error": str(error) if error else None},
                error=str(error) if error else None,
            )

    def add_history_entry(
        self,
        step_id: str,
        message: str,
        data: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """Add an entry to the workflow history.

        ``data`` is shallow-copied so later context mutation does not rewrite
        the recorded entry. The oldest entries are dropped once
        ``max_history_entries`` is exceeded.
        """
        self.history.append(
            WorkflowHistoryEntry(
                step_id=step_id,
                status=self.status.value,
                message=message,
                data=dict(data or {}),
                error=error,
            )
        )
        if (
            self.max_history_entries is not None
            and len(self.history) > self.max_history_entries
        ):
            del self.history[: len(self.history) - self.max_history_entries]

    def is_active(self) -> bool:
        """Check if the workflow is in an active state."""
        return self.status in [WorkflowStatus.RUNNING, WorkflowStatus.WAITING]

    def is_terminal(self) -> bool:
        """Check if the workflow is in a terminal state."""
        return self.status in _TERMINAL_STATUSES

    def run_result(self) -> RunResult:
        """This instance as the core ``RunResult`` envelope (D-047).

        ``status`` and ``reason`` come from ``_RUN_STATUS`` (a failed instance
        has no reason: ``error`` is message text and is never branched on, it
        only rides in ``detail["error"]``); ``handle`` is ``instance_id``;
        ``outputs`` is the PUBLIC NON-SECRET context (a workflow declares no
        output keys, so there is no declared list to restrict to): a copy of
        ``context`` without the top-level internal-prefixed keys and without
        every entry ``is_forbidden_context_entry`` flags, at any depth (D-055). A
        context the shared filter refuses (``ContextFilterError``) gives empty
        ``outputs`` and ``detail["outputs_withheld"] == "unfilterable"``;
        ``identity`` is empty.
        """
        status, reason = _RUN_STATUS[self.status]
        detail: dict[str, Any] = {
            "workflow_id": self.workflow_id,
            "current_step_id": self.current_step_id,
            "workflow_status": self.status.value,
        }
        if self.error is not None:
            detail["error"] = self.error
        public = {
            key: value
            for key, value in self.context.items()
            if not has_internal_prefix(key)
        }
        try:
            outputs = filter_context_tree(
                public,
                MAX_CONTEXT_FILTER_DEPTH,
                lambda key, value, path: (
                    "secret" if is_forbidden_context_entry(key, value) else None
                ),
            )
        except ContextFilterError:
            outputs = {}
            detail["outputs_withheld"] = "unfilterable"
        return RunResult(
            status=status,
            reason=reason,
            outputs=outputs,
            detail=detail,
            handle=self.instance_id,
        )


class EventListener(BaseModel):
    """Information about a workflow instance listening for an event."""

    model_config = ConfigDict(extra="forbid")

    instance_id: str
    success_step_id: str
    event_mapping: dict[str, str] = Field(default_factory=dict)
    registered_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    timeout_at: datetime | None = None

    correlation_key: str | None = None
    """When set, only events whose ``payload[correlation_key]`` equals
    ``correlation_value`` are delivered."""
    correlation_value: Any = None
    """The instance's ``context[correlation_key]`` when the listener was
    registered."""
    timeout_step_id: str | None = None
    """Where a timed-out wait goes; ``None`` fails the instance."""
    step_id: str | None = None
    """The step whose wait registered this listener; an event is delivered
    only while the instance is still at that step."""

    def is_expired(self) -> bool:
        """Check if the event listener has expired."""
        return self.timeout_at is not None and datetime.now(UTC) > self.timeout_at


class WaitEventConfig(BaseModel):
    """Configuration for waiting for an event.

    ``success_step_id=""`` completes the workflow when the event arrives.
    With ``timeout_seconds`` set, the wait times out after that many seconds:
    to ``timeout_step_id`` when given, otherwise the instance FAILS (it never
    stays WAITING forever). ``timeout_step_id`` without ``timeout_seconds`` is
    rejected. ``correlation_key`` restricts delivery to events whose
    ``payload[correlation_key]`` equals the instance's
    ``context[correlation_key]``.
    """

    model_config = ConfigDict(extra="forbid")

    event_type: str
    success_step_id: str
    timeout_seconds: float | None = None
    timeout_step_id: str | None = None
    event_mapping: dict[str, str] = Field(default_factory=dict)
    correlation_key: str | None = None

    @field_validator("event_type")
    @classmethod
    def validate_event_type(cls, v):
        if not v:
            raise ValueError("event_type must be a non-empty string")
        return v

    @field_validator("timeout_seconds")
    @classmethod
    def validate_timeout(cls, v):
        if v is not None and v <= 0:
            raise ValueError("Timeout must be positive")
        return v

    @model_validator(mode="after")
    def validate_timeout_step_id_needs_timeout(self):
        if self.timeout_step_id and self.timeout_seconds is None:
            raise ValueError(
                "timeout_step_id requires timeout_seconds (the wait would never time out)"
            )
        return self
