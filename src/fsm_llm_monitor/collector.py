"""
Event collector for fsm_llm_monitor.

Captures FSM lifecycle events via handler hooks and log records via a loguru sink.
Stores events in bounded deques and computes basic metrics.
"""

from __future__ import annotations

import json
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any

from fsm_llm.constants import (
    MAX_CONTEXT_FILTER_DEPTH,
    has_internal_prefix,
    is_forbidden_context_entry,
)
from fsm_llm.logging import logger
from fsm_llm.utilities import (
    filter_context_tree,
    redact_non_json_leaf,
    redacting_json_default,
)

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
        Streaming clients should prefer :meth:`events_after`, which returns a
        consistent cursor.
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

    @staticmethod
    def _slice_after(
        buffer: deque[Any], total: int, cursor: int, limit: int
    ) -> tuple[list[Any], int]:
        """Oldest-first records recorded after ``cursor`` (a total count),
        at most ``limit``, plus the cursor to use next time.

        Records that fell out of the bounded buffer are skipped (the cursor
        jumps over them). A cursor ahead of ``total`` (the collector was
        cleared) restarts from the beginning of the buffer.
        """
        if cursor > total:
            cursor = total - len(buffer)
        available = min(total - cursor, len(buffer))
        if available <= 0:
            return [], total
        first = len(buffer) - available  # index of the oldest unseen record
        take = min(available, max(limit, 0))
        items = [buffer[first + i] for i in range(take)]
        return items, total - available + take

    def events_after(
        self, cursor: int, limit: int = 50
    ) -> tuple[list[MonitorEvent], int]:
        """Events recorded after ``cursor``, OLDEST first, and the next cursor.

        # DECISION plan-2026-09-28T090000-3c9e41d2/D-004
        # Stream from a cursor taken under the SAME lock as the slice. Do NOT
        # go back to reading the total first and slicing "newest N" later:
        # that re-sent the newest records every cycle and never delivered the
        # middle of a burst.
        """
        with self._lock:
            return self._slice_after(self._events, self._total_events, cursor, limit)

    def logs_after(self, cursor: int, limit: int = 50) -> tuple[list[LogRecord], int]:
        """Log records received after ``cursor``, OLDEST first, and the next
        cursor (see :meth:`events_after`)."""
        with self._lock:
            return self._slice_after(self._logs, self._total_logs, cursor, limit)

    def resize(self, max_events: int, max_log_lines: int) -> None:
        """Change the buffer bounds, keeping the newest records."""
        with self._lock:
            if max_events != self._max_events:
                self._events = deque(self._events, maxlen=max_events)
                self._max_events = max_events
            if max_log_lines != self._max_log_lines:
                self._logs = deque(self._logs, maxlen=max_log_lines)
                self._max_log_lines = max_log_lines

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
        """Get recent logs, newest first. Optionally filter by minimum level
        (case-insensitive; loguru's TRACE and SUCCESS levels are ranked)."""
        with self._lock:
            logs = list(self._logs)
        logs.reverse()
        min_level = _LEVEL_ORDER.get(level.upper()) if level else None
        if min_level is not None:
            logs = [
                r for r in logs if _LEVEL_ORDER.get(r.level.upper(), 0) >= min_level
            ]
        if limit > 0:
            return logs[:limit]
        return logs

    def get_logs_since(self, after_total: int, limit: int = 50) -> list[LogRecord]:
        """Get logs added after a given total count, newest first.

        Mirrors :meth:`get_events_since` so the WebSocket loop can stream only
        new logs without dropping the middle of a burst.
        """
        with self._lock:
            new_count = self._total_logs - after_total
            if new_count <= 0:
                return []
            take = min(new_count, limit, len(self._logs))
            logs = self._logs
            result = [logs[-1 - i] for i in range(take)]
        return result

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
        """Clear all collected data.

        Totals restart at zero; streaming cursors ahead of the new totals are
        handled by :meth:`events_after` / :meth:`logs_after`.
        """
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

        Every callback takes ``(context, current_state=None,
        target_state=None)``: the monitor's handler wrapper passes the states
        the core hands to ``should_execute`` (the core never puts
        ``_target_state`` into the handler context). Without them the
        callbacks fall back to the context keys.
        """
        return {
            "START_CONVERSATION": self._on_start_conversation,
            "PRE_PROCESSING": self._on_pre_processing,
            "POST_PROCESSING": self._on_post_processing,
            "PRE_TRANSITION": self._on_pre_transition,
            "POST_TRANSITION": self._on_post_transition,  # no-op; not registered
            "CONTEXT_UPDATE": self._on_context_update,
            "END_CONVERSATION": self._on_end_conversation,
            "ERROR": self._on_error,
        }

    def _on_start_conversation(
        self,
        context: dict[str, Any],
        current_state: str | None = None,
        target_state: str | None = None,
    ) -> dict[str, Any]:
        conv_id = context.get("_conversation_id", "")
        state = _state_of(context, current_state)
        self.record_event(
            MonitorEvent(
                event_type=EVENT_CONVERSATION_START,
                conversation_id=conv_id,
                target_state=state or None,
                message=f"Conversation started: {conv_id}",
            )
        )
        if state:
            with self._lock:
                self._states_visited[state] = self._states_visited.get(state, 0) + 1
        return {}

    def _on_pre_processing(
        self,
        context: dict[str, Any],
        current_state: str | None = None,
        target_state: str | None = None,
    ) -> dict[str, Any]:
        conv_id = context.get("_conversation_id", "")
        self.record_event(
            MonitorEvent(
                event_type=EVENT_PRE_PROCESSING,
                conversation_id=conv_id,
                message="Processing started",
            )
        )
        return {}

    def _on_post_processing(
        self,
        context: dict[str, Any],
        current_state: str | None = None,
        target_state: str | None = None,
    ) -> dict[str, Any]:
        conv_id = context.get("_conversation_id", "")
        self.record_event(
            MonitorEvent(
                event_type=EVENT_POST_PROCESSING,
                conversation_id=conv_id,
                message="Processing completed",
            )
        )
        return {}

    def _on_pre_transition(
        self,
        context: dict[str, Any],
        current_state: str | None = None,
        target_state: str | None = None,
    ) -> dict[str, Any]:
        conv_id = context.get("_conversation_id", "")
        source = _state_of(context, current_state)
        target = target_state or context.get("_target_state", "") or ""
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

    def _on_post_transition(
        self,
        context: dict[str, Any],
        current_state: str | None = None,
        target_state: str | None = None,
    ) -> dict[str, Any]:
        # Post-transition is informational; pre-transition already captured
        return {}

    def _on_context_update(
        self,
        context: dict[str, Any],
        current_state: str | None = None,
        target_state: str | None = None,
    ) -> dict[str, Any]:
        conv_id = context.get("_conversation_id", "")
        self.record_event(
            MonitorEvent(
                event_type=EVENT_CONTEXT_UPDATE,
                conversation_id=conv_id,
                message="Context updated",
            )
        )
        return {}

    def _on_end_conversation(
        self,
        context: dict[str, Any],
        current_state: str | None = None,
        target_state: str | None = None,
    ) -> dict[str, Any]:
        conv_id = context.get("_conversation_id", "")
        state = _state_of(context, current_state)
        self.record_event(
            MonitorEvent(
                event_type=EVENT_CONVERSATION_END,
                conversation_id=conv_id,
                message=f"Conversation ended: {conv_id}",
                data={"state": state},
            )
        )
        return {}

    def _on_error(
        self,
        context: dict[str, Any],
        current_state: str | None = None,
        target_state: str | None = None,
    ) -> dict[str, Any]:
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

        Filters out internal-prefixed keys, secret-looking entries (at every
        depth), noise keys, and empty values. Values are rendered as text
        WITHOUT calling an object's ``__str__`` (non-JSON leaves become
        ``<redacted:Type>``).
        """
        ctx = redact_context(ctx)
        out: dict[str, str] = {}
        for k, v in ctx.items():
            # DECISION plan-2026-07-20T040150-876e7164/D-003 [STALE]
            # TWO layers: `has_internal_prefix` is the canonical security
            # predicate (do NOT re-inline `k.startswith("_")` -- case-sensitive
            # and blind to `system_`/`internal_`/`__`), and _CONTEXT_NOISE_KEYS
            # is a separate display-only suppression list that must be kept
            # alongside it, not folded into it. See decisions.md D-003.
            if has_internal_prefix(k) or k in EventCollector._CONTEXT_NOISE_KEYS:
                continue
            s = _display_text(v)
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

        def _on_start(
            ctx: dict[str, Any],
            current_state: str | None = None,
            target_state: str | None = None,
        ) -> dict[str, Any]:
            _emit(
                {
                    "type": "start",
                    "state": _state_of(ctx, current_state),
                    "conversation_id": ctx.get("_conversation_id", ""),
                    "timestamp": _ts(),
                }
            )
            return {}

        def _on_post_processing(
            ctx: dict[str, Any],
            current_state: str | None = None,
            target_state: str | None = None,
        ) -> dict[str, Any]:
            data = self.snapshot_context(ctx)
            if not data:
                return {}
            _emit(
                {
                    "type": "context",
                    "state": _state_of(ctx, current_state),
                    "data": data,
                    "timestamp": _ts(),
                }
            )
            return {}

        def _on_pre_transition(
            ctx: dict[str, Any],
            current_state: str | None = None,
            target_state: str | None = None,
        ) -> dict[str, Any]:
            source = _state_of(ctx, current_state)
            target = target_state or ctx.get("_target_state", "") or ""
            if not target or target == source:
                return {}
            _emit(
                {
                    "type": "transition",
                    "source": source,
                    "target": target,
                    "timestamp": _ts(),
                }
            )
            return {}

        def _on_context_update(
            ctx: dict[str, Any],
            current_state: str | None = None,
            target_state: str | None = None,
        ) -> dict[str, Any]:
            data = self.snapshot_context(ctx)
            if not data:
                return {}
            _emit(
                {
                    "type": "context",
                    "state": _state_of(ctx, current_state),
                    "data": data,
                    "timestamp": _ts(),
                }
            )
            return {}

        def _on_end(
            ctx: dict[str, Any],
            current_state: str | None = None,
            target_state: str | None = None,
        ) -> dict[str, Any]:
            _emit(
                {
                    "type": "end",
                    "state": _state_of(ctx, current_state),
                    "data": self.snapshot_context(ctx),
                    "timestamp": _ts(),
                }
            )
            return {}

        def _on_error(
            ctx: dict[str, Any],
            current_state: str | None = None,
            target_state: str | None = None,
        ) -> dict[str, Any]:
            _emit(
                {
                    "type": "error",
                    "state": _state_of(ctx, current_state),
                    "error": str(ctx.get("_error", "Unknown error")),
                    "timestamp": _ts(),
                }
            )
            return {}

        return {
            "START_CONVERSATION": _on_start,
            "POST_PROCESSING": _on_post_processing,
            "PRE_TRANSITION": _on_pre_transition,
            "CONTEXT_UPDATE": _on_context_update,
            "END_CONVERSATION": _on_end,
            "ERROR": _on_error,
        }


# ------------------------------------------------------------------
# Module helpers
# ------------------------------------------------------------------

# Minimum-level ranking for log filters (loguru's numeric levels).
_LEVEL_ORDER: dict[str, int] = {
    "TRACE": 5,
    "DEBUG": 10,
    "INFO": 20,
    "SUCCESS": 25,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}


def _state_of(context: dict[str, Any], state: str | None) -> str:
    """The state the core passed to the handler, else the context's copy."""
    if state:
        return state
    value = context.get("_current_state", "")
    return value if isinstance(value, str) else ""


def _drop_for_display(
    drop_internal: bool,
) -> Any:
    def _should_drop(key: Any, value: Any, _full_key: str) -> str | None:
        if isinstance(key, str):
            if drop_internal and has_internal_prefix(key):
                return "internal"
            if is_forbidden_context_entry(key, value):
                return "forbidden"
        return None

    return _should_drop


def redact_context(data: Any, drop_internal: bool = True) -> Any:
    """A copy of ``data`` safe to show on the dashboard.

    Secret-looking entries (``is_forbidden_context_entry``) are dropped at
    every depth, internal-prefix keys too unless ``drop_internal`` is False,
    and non-JSON leaves become ``<redacted:Type>`` (their ``__str__`` is never
    called). Non-dict input is returned leaf-redacted.

    # DECISION plan-2026-09-28T090000-3c9e41d2/D-002
    # ONE redaction path for everything the monitor shows (conversation
    # snapshots, agent logs and results, workflow context). Do NOT re-inline
    # a top-level key filter or call ``str(value)`` before this runs: a
    # nested secret and an object's ``__str__`` both reached the dashboard.
    """
    if not isinstance(data, dict):
        if isinstance(data, list | tuple):
            return [redact_context(v, drop_internal) for v in data]
        return redact_non_json_leaf(data)
    try:
        return filter_context_tree(
            data,
            MAX_CONTEXT_FILTER_DEPTH,
            _drop_for_display(drop_internal),
            leaf=redact_non_json_leaf,
        )
    except Exception:
        # A pathological cyclic value: fall back to a top-level filter.
        should_drop = _drop_for_display(drop_internal)
        return {
            k: redact_non_json_leaf(v)
            for k, v in data.items()
            if should_drop(k, v, str(k)) is None
        }


def _display_text(value: Any) -> str:
    """Text for an already-redacted value (containers as JSON)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict | list | tuple):
        try:
            return json.dumps(value, default=redacting_json_default)
        except (TypeError, ValueError):
            return f"<redacted:{type(value).__name__}>"
    if isinstance(value, bool | int | float):
        return str(value)
    return redacting_json_default(value)
