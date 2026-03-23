from __future__ import annotations

"""
Pydantic models for fsm_llm_monitor.

Defines event, metric, configuration, and snapshot models used by
the collector, bridge, and TUI screens.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from .constants import (
    DEFAULT_LOG_LEVEL,
    DEFAULT_MAX_EVENTS,
    DEFAULT_MAX_LOG_LINES,
    DEFAULT_REFRESH_INTERVAL,
)


class MonitorEvent(BaseModel):
    """A single observable event captured from the FSM system."""

    event_type: str
    timestamp: datetime = Field(default_factory=datetime.now)
    conversation_id: str | None = None
    source_state: str | None = None
    target_state: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    level: str = "INFO"
    message: str = ""


class LogRecord(BaseModel):
    """A captured log record from loguru."""

    timestamp: datetime = Field(default_factory=datetime.now)
    level: str = "INFO"
    message: str = ""
    module: str = ""
    function: str = ""
    line: int = 0
    conversation_id: str | None = None


class MetricSnapshot(BaseModel):
    """Point-in-time metric snapshot of the FSM system."""

    timestamp: datetime = Field(default_factory=datetime.now)
    active_conversations: int = 0
    total_events: int = 0
    total_errors: int = 0
    total_transitions: int = 0
    events_per_type: dict[str, int] = Field(default_factory=dict)
    states_visited: dict[str, int] = Field(default_factory=dict)


class ConversationSnapshot(BaseModel):
    """Snapshot of a single conversation's state."""

    conversation_id: str
    current_state: str = ""
    state_description: str = ""
    is_terminal: bool = False
    context_data: dict[str, Any] = Field(default_factory=dict)
    message_history: list[dict[str, str]] = Field(default_factory=list)
    stack_depth: int = 1
    last_extraction: dict[str, Any] | None = None
    last_transition: dict[str, Any] | None = None
    last_response: dict[str, Any] | None = None


class StateInfo(BaseModel):
    """Information about a single FSM state."""

    state_id: str
    description: str = ""
    purpose: str = ""
    is_initial: bool = False
    is_terminal: bool = False
    transition_count: int = 0
    transitions: list[TransitionInfo] = Field(default_factory=list)


class TransitionInfo(BaseModel):
    """Information about a single FSM transition."""

    target_state: str
    description: str = ""
    priority: int = 0
    condition_count: int = 0
    has_logic: bool = False


class FSMSnapshot(BaseModel):
    """Snapshot of an FSM definition for display."""

    name: str = ""
    description: str = ""
    version: str = ""
    initial_state: str = ""
    persona: str | None = None
    state_count: int = 0
    states: list[StateInfo] = Field(default_factory=list)


class MonitorConfig(BaseModel):
    """Configuration for the monitor."""

    refresh_interval: float = DEFAULT_REFRESH_INTERVAL
    max_events: int = DEFAULT_MAX_EVENTS
    max_log_lines: int = DEFAULT_MAX_LOG_LINES
    log_level: str = DEFAULT_LOG_LEVEL
    show_internal_keys: bool = False
    auto_scroll_logs: bool = True


# --- Instance Management Models ---


class InstanceInfo(BaseModel):
    """Summary of a managed instance for listing."""

    instance_id: str
    instance_type: str  # "fsm" | "workflow" | "agent"
    label: str = ""
    status: str = "running"  # "running" | "completed" | "failed" | "cancelled"
    created_at: datetime = Field(default_factory=datetime.now)
    source: str = "custom"  # preset ID or "custom"
    conversation_count: int = 0
    active_workflows: int = 0
    agent_type: str = ""


class LaunchFSMRequest(BaseModel):
    """Request to launch an FSM from preset or raw JSON."""

    preset_id: str | None = None
    fsm_json: dict[str, Any] | None = None
    model: str = "gpt-4o-mini"
    temperature: float = 0.5
    label: str = ""


class StartConversationRequest(BaseModel):
    """Request to start a conversation on a launched FSM."""

    initial_context: dict[str, Any] = Field(default_factory=dict)


class SendMessageRequest(BaseModel):
    """Request to send a message to an FSM conversation."""

    message: str
    conversation_id: str


class EndConversationRequest(BaseModel):
    """Request to end a conversation."""

    conversation_id: str


class StubToolConfig(BaseModel):
    """Configuration for a stub tool (code-free agent tool)."""

    name: str
    description: str
    stub_response: str = "Tool executed successfully"


class LaunchAgentRequest(BaseModel):
    """Request to launch an agent."""

    agent_type: str = "ReactAgent"
    task: str
    model: str = "gpt-4o-mini"
    max_iterations: int = 10
    timeout_seconds: float = 120.0
    tools: list[StubToolConfig] = Field(default_factory=list)
    label: str = ""


class LaunchWorkflowRequest(BaseModel):
    """Request to launch a workflow."""

    preset_id: str | None = None
    definition_json: dict[str, Any] | None = None
    initial_context: dict[str, Any] = Field(default_factory=dict)
    label: str = ""


class WorkflowAdvanceRequest(BaseModel):
    """Request to advance a workflow instance."""

    workflow_instance_id: str
    user_input: str = ""


class WorkflowCancelRequest(BaseModel):
    """Request to cancel a workflow instance."""

    workflow_instance_id: str
    reason: str = ""


# Update forward references
StateInfo.model_rebuild()
