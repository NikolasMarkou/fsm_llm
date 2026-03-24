from __future__ import annotations

"""
Event collector for fsm_llm_monitor.

Captures FSM lifecycle events via handler hooks and log records via a loguru sink.
Stores events in bounded deques and computes basic metrics.
"""

import threading
from collections import deque
from datetime import datetime
from typing import Any

from fsm_llm.logging import logger

from .constants import (
    DEFAULT_MAX_EVENTS,
    DEFAULT_MAX_LOG_LINES,
    EVENT_CONTEXT_UPDATE,
    EVENT_CONVERSATION_END,
    EVENT_CONVERSATION_START,
    EVENT_ERROR,
    EVENT_LOG,
    EVENT_POST_PROCESSING,
    EVENT_PRE_PROCESSING,
    EVENT_STATE_TRANSITION,
    MONITOR_HANDLER_NAME,
    MONITOR_HANDLER_PRIORITY,
)
from .definitions import LogRecord, MetricSnapshot, MonitorEvent


class EventCollector:
    """Collects FSM lifecycle events and log records.

    Thread-safe. Uses bounded deques to prevent memory leaks.
    """

    def __init__(
        self,
        max_events: int = DEFAULT_MAX_EVENTS,
        max_log_lines: int = DEFAULT_MAX_LOG_LINES,
    ) -> None:
        self._max_events = max_events
        self._max_log_lines = max_log_lines
        self._events: deque[MonitorEvent] = deque(maxlen=max_events)
        self._logs: deque[LogRecord] = deque(maxlen=max_log_lines)
        self._lock = threading.Lock()

        # Metric counters
        self._total_events = 0
        self._total_errors = 0
        self._total_transitions = 0
        self._events_per_type: dict[str, int] = {}
        self._states_visited: dict[str, int] = {}
        self._active_conversations: set[str] = set()
        self._total_logs = 0

        # Loguru sink ID for cleanup
        self._log_sink_id: int | None = None

    @property
    def handler_name(self) -> str:
        return MONITOR_HANDLER_NAME

    @property
    def handler_priority(self) -> int:
        return MONITOR_HANDLER_PRIORITY

    def record_event(self, event: MonitorEvent) -> None:
        """Record a monitor event. Thread-safe."""
        with self._lock:
            self._events.append(event)
            self._total_events += 1
            self._events_per_type[event.event_type] = (
                self._events_per_type.get(event.event_type, 0) + 1
            )

            if event.event_type == EVENT_ERROR:
                self._total_errors += 1

            if event.event_type == EVENT_STATE_TRANSITION:
                self._total_transitions += 1
                if event.target_state:
                    self._states_visited[event.target_state] = (
                        self._states_visited.get(event.target_state, 0) + 1
                    )

            if event.event_type == EVENT_CONVERSATION_START and event.conversation_id:
                self._active_conversations.add(event.conversation_id)
            elif event.event_type == EVENT_CONVERSATION_END and event.conversation_id:
                self._active_conversations.discard(event.conversation_id)

    def record_log(self, record: LogRecord) -> None:
        """Record a log entry. Thread-safe."""
        with self._lock:
            self._logs.append(record)
            self._total_logs += 1

    @property
    def total_logs(self) -> int:
        """Total number of log records received (monotonically increasing)."""
        return self._total_logs

    def get_events(self, limit: int = 0) -> list[MonitorEvent]:
        """Get recent events, newest first."""
        with self._lock:
            events = list(self._events)
        events.reverse()
        if limit > 0:
            return events[:limit]
        return events

    def get_logs(self, limit: int = 0, level: str | None = None) -> list[LogRecord]:
        """Get recent logs, newest first. Optionally filter by level."""
        level_order = {
            "DEBUG": 0,
            "INFO": 1,
            "WARNING": 2,
            "ERROR": 3,
            "CRITICAL": 4,
        }
        with self._lock:
            logs = list(self._logs)
        logs.reverse()
        if level and level in level_order:
            min_level = level_order[level]
            logs = [r for r in logs if level_order.get(r.level, 0) >= min_level]
        if limit > 0:
            return logs[:limit]
        return logs

    def get_metrics(self) -> MetricSnapshot:
        """Get current metric snapshot."""
        with self._lock:
            return MetricSnapshot(
                timestamp=datetime.now(),
                active_conversations=len(self._active_conversations),
                total_events=self._total_events,
                total_errors=self._total_errors,
                total_transitions=self._total_transitions,
                events_per_type=dict(self._events_per_type),
                states_visited=dict(self._states_visited),
            )

    def get_events_by_conversation(
        self, conversation_id: str, limit: int = 0
    ) -> list[MonitorEvent]:
        """Get events for a specific conversation."""
        with self._lock:
            events = [e for e in self._events if e.conversation_id == conversation_id]
        events.reverse()
        if limit > 0:
            return events[:limit]
        return events

    def clear(self) -> None:
        """Clear all collected data."""
        with self._lock:
            self._events.clear()
            self._logs.clear()
            self._total_events = 0
            self._total_errors = 0
            self._total_transitions = 0
            self._events_per_type.clear()
            self._states_visited.clear()
            self._active_conversations.clear()
            self._total_logs = 0

    def cleanup(self) -> None:
        """Remove the loguru sink if one was registered."""
        if self._log_sink_id is not None:
            try:
                from loguru import logger as _loguru_logger

                _loguru_logger.remove(self._log_sink_id)
            except (ValueError, Exception) as e:
                logger.debug(f"Failed to remove loguru sink {self._log_sink_id}: {e}")
            finally:
                self._log_sink_id = None

    # Note: __del__ was intentionally removed. Calling loguru.logger.remove()
    # during interpreter shutdown is unreliable. Use cleanup() explicitly.

    def create_loguru_sink(self) -> Any:
        """Create a loguru sink function that feeds into this collector."""

        def _sink(message: Any) -> None:
            record = message.record
            log_record = LogRecord(
                timestamp=record["time"].replace(tzinfo=None),
                level=record["level"].name,
                message=str(record["message"]),
                module=record["module"],
                function=str(record["function"]) if record["function"] else "",
                line=record["line"],
                conversation_id=record["extra"].get("conversation_id"),
            )
            self.record_log(log_record)

            # Also record as an event for the event stream
            self.record_event(
                MonitorEvent(
                    event_type=EVENT_LOG,
                    timestamp=log_record.timestamp,
                    conversation_id=log_record.conversation_id,
                    level=log_record.level,
                    message=log_record.message,
                )
            )

        return _sink

    def create_handler_callbacks(self) -> dict[str, Any]:
        """Create callback functions for each handler timing point.

        Returns a dict mapping timing names to callback functions
        that can be used with the handler system.
        """
        return {
            "START_CONVERSATION": self._on_start_conversation,
            "PRE_PROCESSING": self._on_pre_processing,
            "POST_PROCESSING": self._on_post_processing,
            "PRE_TRANSITION": self._on_pre_transition,
            "POST_TRANSITION": self._on_post_transition,
            "CONTEXT_UPDATE": self._on_context_update,
            "END_CONVERSATION": self._on_end_conversation,
            "ERROR": self._on_error,
        }

    def _on_start_conversation(self, context: dict[str, Any]) -> dict[str, Any]:
        conv_id = context.get("_conversation_id", "")
        self.record_event(
            MonitorEvent(
                event_type=EVENT_CONVERSATION_START,
                conversation_id=conv_id,
                message=f"Conversation started: {conv_id}",
            )
        )
        return {}

    def _on_pre_processing(self, context: dict[str, Any]) -> dict[str, Any]:
        conv_id = context.get("_conversation_id", "")
        self.record_event(
            MonitorEvent(
                event_type=EVENT_PRE_PROCESSING,
                conversation_id=conv_id,
                message="Processing started",
            )
        )
        return {}

    def _on_post_processing(self, context: dict[str, Any]) -> dict[str, Any]:
        conv_id = context.get("_conversation_id", "")
        self.record_event(
            MonitorEvent(
                event_type=EVENT_POST_PROCESSING,
                conversation_id=conv_id,
                message="Processing completed",
            )
        )
        return {}

    def _on_pre_transition(self, context: dict[str, Any]) -> dict[str, Any]:
        conv_id = context.get("_conversation_id", "")
        source = context.get("_current_state", "")
        target = context.get("_target_state", "")
        self.record_event(
            MonitorEvent(
                event_type=EVENT_STATE_TRANSITION,
                conversation_id=conv_id,
                source_state=source,
                target_state=target,
                message=f"Transition: {source} -> {target}",
                level="INFO",
            )
        )
        return {}

    def _on_post_transition(self, context: dict[str, Any]) -> dict[str, Any]:
        # Post-transition is informational; pre-transition already captured
        return {}

    def _on_context_update(self, context: dict[str, Any]) -> dict[str, Any]:
        conv_id = context.get("_conversation_id", "")
        self.record_event(
            MonitorEvent(
                event_type=EVENT_CONTEXT_UPDATE,
                conversation_id=conv_id,
                message="Context updated",
            )
        )
        return {}

    def _on_end_conversation(self, context: dict[str, Any]) -> dict[str, Any]:
        conv_id = context.get("_conversation_id", "")
        self.record_event(
            MonitorEvent(
                event_type=EVENT_CONVERSATION_END,
                conversation_id=conv_id,
                message=f"Conversation ended: {conv_id}",
            )
        )
        return {}

    def _on_error(self, context: dict[str, Any]) -> dict[str, Any]:
        conv_id = context.get("_conversation_id", "")
        error = context.get("_error", "Unknown error")
        self.record_event(
            MonitorEvent(
                event_type=EVENT_ERROR,
                conversation_id=conv_id,
                level="ERROR",
                message=f"Error: {error}",
                data={"error": str(error)},
            )
        )
        return {}
