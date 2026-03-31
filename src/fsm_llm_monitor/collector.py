from __future__ import annotations

"""
Event collector for fsm_llm_monitor.

Captures FSM lifecycle events via handler hooks and log records via a loguru sink.
Stores events in bounded deques and computes basic metrics.
"""

import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any

from fsm_llm.logging import logger

from .constants import (
    DEFAULT_MAX_EVENTS,
    DEFAULT_MAX_LOG_LINES,
    EVENT_AGENT_ITERATION,
    EVENT_AGENT_TOOL_CALL,
    EVENT_CONTEXT_UPDATE,
    EVENT_CONVERSATION_END,
    EVENT_CONVERSATION_START,
    EVENT_ERROR,
    EVENT_POST_PROCESSING,
    EVENT_PRE_PROCESSING,
    EVENT_STATE_TRANSITION,
    EVENT_WORKFLOW_ADVANCED,
    MONITOR_HANDLER_NAME,
    MONITOR_HANDLER_PRIORITY,
)
from .definitions import LogRecord, MetricSnapshot, MonitorEvent
from .exceptions import MetricCollectionError


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
        # Agent/workflow counters
        self._total_agent_iterations = 0
        self._total_tool_calls = 0
        self._total_workflow_steps = 0

        # Loguru sink ID for cleanup
        self._log_sink_id: int | None = None

    @property
    def handler_name(self) -> str:
        return MONITOR_HANDLER_NAME

    @property
    def handler_priority(self) -> int:
        return MONITOR_HANDLER_PRIORITY

    def get_events_since(self, after_total: int, limit: int = 50) -> list[MonitorEvent]:
        """Get events added after a given total count, newest first.

        Use with ``total_events`` to stream only new events to clients.
        """
        with self._lock:
            new_count = self._total_events - after_total
            if new_count <= 0:
                return []
            # Take the min(new_count, limit, deque_size) newest entries
            take = min(new_count, limit, len(self._events))
            # Iterate from the right end of the deque — avoids copying the
            # entire deque to a list when only a few recent entries are needed.
            events = self._events
            result = [events[-1 - i] for i in range(take)]
        return result

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

            if event.event_type == EVENT_AGENT_ITERATION:
                self._total_agent_iterations += 1
            elif event.event_type == EVENT_AGENT_TOOL_CALL:
                self._total_tool_calls += 1
            elif event.event_type == EVENT_WORKFLOW_ADVANCED:
                self._total_workflow_steps += 1

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
        """Get current metric snapshot.

        Raises ``MetricCollectionError`` if metric aggregation fails.
        """
        try:
            with self._lock:
                return MetricSnapshot(
                    timestamp=datetime.now(timezone.utc),
                    active_conversations=len(self._active_conversations),
                    total_events=self._total_events,
                    total_errors=self._total_errors,
                    total_transitions=self._total_transitions,
                    events_per_type=dict(self._events_per_type),
                    states_visited=dict(self._states_visited),
                    total_agent_iterations=self._total_agent_iterations,
                    total_tool_calls=self._total_tool_calls,
                    total_workflow_steps=self._total_workflow_steps,
                )
        except Exception as e:
            raise MetricCollectionError(f"Failed to collect metrics: {e}") from e

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
            self._total_agent_iterations = 0
            self._total_tool_calls = 0
            self._total_workflow_steps = 0

    def cleanup(self) -> None:
        """Remove the loguru sink if one was registered."""
        if self._log_sink_id is not None:
            try:
                from loguru import logger as _loguru_logger

                _loguru_logger.remove(self._log_sink_id)
            except Exception as e:
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
                timestamp=record["time"].astimezone(timezone.utc),
                level=record["level"].name,
                message=str(record["message"]),
                module=record["module"],
                function=str(record["function"]) if record["function"] else "",
                line=record["line"],
                conversation_id=record["extra"].get("conversation_id"),
            )
            self.record_log(log_record)

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
            "POST_TRANSITION": self._on_post_transition,  # no-op; pre-transition captures data
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

    # ------------------------------------------------------------------
    # Context-snapshot capture (for agent/workflow conversation views)
    # ------------------------------------------------------------------

    # Keys always present but not informative for display
    _CONTEXT_NOISE_KEYS = frozenset({"task", "should_terminate"})
    _CONTEXT_EMPTY_VALUES = frozenset({"", "None", "False", "0", "[]", "{}", "null"})

    @staticmethod
    def snapshot_context(
        ctx: dict[str, Any],
        max_value_len: int = 1500,
    ) -> dict[str, str]:
        """Extract display-worthy key-value pairs from an FSM context dict.

        Filters out internal (``_``-prefixed) keys, noise keys, and empty values.
        """
        out: dict[str, str] = {}
        for k, v in ctx.items():
            if k.startswith("_") or k in EventCollector._CONTEXT_NOISE_KEYS:
                continue
            s = str(v) if v is not None else ""
            if s in EventCollector._CONTEXT_EMPTY_VALUES:
                continue
            out[k] = s[:max_value_len]
        return out

    def create_context_capture_callbacks(
        self,
        sink: Any,
    ) -> dict[str, Any]:
        """Create handler callbacks that capture context snapshots into *sink*.

        *sink* must support ``append(entry: dict)`` and be protected by a
        ``_conv_lock`` threading.Lock attribute (e.g. ``ManagedAgent``).

        Returns a dict mapping timing names to callback functions, same
        format as ``create_handler_callbacks`` — ready for handler registration.
        """

        def _emit(entry: dict[str, Any]) -> None:
            with sink._conv_lock:
                sink.conversation_log.append(entry)

        def _ts() -> str:
            return datetime.now(timezone.utc).isoformat()

        def _on_start(ctx: dict[str, Any]) -> dict[str, Any]:
            _emit({
                "type": "start",
                "state": ctx.get("_current_state", ""),
                "conversation_id": ctx.get("_conversation_id", ""),
                "timestamp": _ts(),
            })
            return {}

        def _on_post_processing(ctx: dict[str, Any]) -> dict[str, Any]:
            data = self.snapshot_context(ctx)
            if not data:
                return {}
            _emit({
                "type": "context",
                "state": ctx.get("_current_state", ""),
                "data": data,
                "timestamp": _ts(),
            })
            return {}

        def _on_pre_transition(ctx: dict[str, Any]) -> dict[str, Any]:
            source = ctx.get("_current_state", "")
            target = ctx.get("_target_state", "")
            if not target or target == source:
                return {}
            _emit({
                "type": "transition",
                "source": source,
                "target": target,
                "timestamp": _ts(),
            })
            return {}

        def _on_context_update(ctx: dict[str, Any]) -> dict[str, Any]:
            data = self.snapshot_context(ctx)
            if not data:
                return {}
            _emit({
                "type": "context",
                "state": ctx.get("_current_state", ""),
                "data": data,
                "timestamp": _ts(),
            })
            return {}

        def _on_end(ctx: dict[str, Any]) -> dict[str, Any]:
            _emit({
                "type": "end",
                "state": ctx.get("_current_state", ""),
                "data": self.snapshot_context(ctx),
                "timestamp": _ts(),
            })
            return {}

        def _on_error(ctx: dict[str, Any]) -> dict[str, Any]:
            _emit({
                "type": "error",
                "state": ctx.get("_current_state", ""),
                "error": str(ctx.get("_error", "Unknown error")),
                "timestamp": _ts(),
            })
            return {}

        return {
            "START_CONVERSATION": _on_start,
            "POST_PROCESSING": _on_post_processing,
            "PRE_TRANSITION": _on_pre_transition,
            "CONTEXT_UPDATE": _on_context_update,
            "END_CONVERSATION": _on_end,
            "ERROR": _on_error,
        }
