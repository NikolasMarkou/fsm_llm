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


# Update forward references
StateInfo.model_rebuild()
