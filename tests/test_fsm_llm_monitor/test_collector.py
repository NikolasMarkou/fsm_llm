from __future__ import annotations

"""Tests for fsm_llm_monitor.collector."""

import threading
from datetime import timezone

from fsm_llm_monitor.collector import EventCollector
from fsm_llm_monitor.constants import (
    EVENT_CONVERSATION_END,
    EVENT_CONVERSATION_START,
    EVENT_ERROR,
    EVENT_STATE_TRANSITION,
)
from fsm_llm_monitor.definitions import LogRecord, MonitorEvent


class TestEventCollector:
    def test_create(self):
        collector = EventCollector()
        metrics = collector.get_metrics()
        assert metrics.total_events == 0
        assert metrics.active_conversations == 0

    def test_record_event(self):
        collector = EventCollector()
        event = MonitorEvent(
            event_type="test",
            conversation_id="c1",
            message="hello",
        )
        collector.record_event(event)
        assert collector.get_metrics().total_events == 1

    def test_events_newest_first(self):
        collector = EventCollector()
        for i in range(3):
            collector.record_event(MonitorEvent(event_type="test", message=f"msg-{i}"))
        events = collector.get_events()
        assert events[0].message == "msg-2"
        assert events[2].message == "msg-0"

    def test_events_limit(self):
        collector = EventCollector()
        for i in range(10):
            collector.record_event(MonitorEvent(event_type="test", message=f"msg-{i}"))
        events = collector.get_events(limit=3)
        assert len(events) == 3

    def test_bounded_deque(self):
        collector = EventCollector(max_events=5)
        for i in range(10):
            collector.record_event(MonitorEvent(event_type="test", message=f"msg-{i}"))
        # Deque holds max 5, but total count is still 10
        events = collector.get_events()
        assert len(events) == 5
        assert collector.get_metrics().total_events == 10

    def test_error_counting(self):
        collector = EventCollector()
        collector.record_event(MonitorEvent(event_type=EVENT_ERROR, message="fail"))
        collector.record_event(MonitorEvent(event_type="test", message="ok"))
        metrics = collector.get_metrics()
        assert metrics.total_errors == 1
        assert metrics.total_events == 2

    def test_transition_counting(self):
        collector = EventCollector()
        collector.record_event(
            MonitorEvent(
                event_type=EVENT_STATE_TRANSITION,
                source_state="start",
                target_state="end",
            )
        )
        metrics = collector.get_metrics()
        assert metrics.total_transitions == 1
        assert metrics.states_visited["end"] == 1

    def test_active_conversation_tracking(self):
        collector = EventCollector()
        collector.record_event(
            MonitorEvent(
                event_type=EVENT_CONVERSATION_START,
                conversation_id="c1",
            )
        )
        assert collector.get_metrics().active_conversations == 1

        collector.record_event(
            MonitorEvent(
                event_type=EVENT_CONVERSATION_START,
                conversation_id="c2",
            )
        )
        assert collector.get_metrics().active_conversations == 2

        collector.record_event(
            MonitorEvent(
                event_type=EVENT_CONVERSATION_END,
                conversation_id="c1",
            )
        )
        assert collector.get_metrics().active_conversations == 1

    def test_get_events_by_conversation(self):
        collector = EventCollector()
        collector.record_event(
            MonitorEvent(event_type="test", conversation_id="c1", message="a")
        )
        collector.record_event(
            MonitorEvent(event_type="test", conversation_id="c2", message="b")
        )
        collector.record_event(
            MonitorEvent(event_type="test", conversation_id="c1", message="c")
        )
        events = collector.get_events_by_conversation("c1")
        assert len(events) == 2
        assert all(e.conversation_id == "c1" for e in events)


class TestLogCapture:
    def test_record_log(self):
        collector = EventCollector()
        record = LogRecord(
            level="INFO",
            message="test log",
            module="test",
        )
        collector.record_log(record)
        logs = collector.get_logs()
        assert len(logs) == 1
        assert logs[0].message == "test log"

    def test_logs_newest_first(self):
        collector = EventCollector()
        for i in range(3):
            collector.record_log(LogRecord(message=f"log-{i}"))
        logs = collector.get_logs()
        assert logs[0].message == "log-2"

    def test_log_level_filter(self):
        collector = EventCollector()
        collector.record_log(LogRecord(level="DEBUG", message="debug"))
        collector.record_log(LogRecord(level="INFO", message="info"))
        collector.record_log(LogRecord(level="ERROR", message="error"))

        logs = collector.get_logs(level="ERROR")
        assert len(logs) == 1
        assert logs[0].level == "ERROR"

        logs = collector.get_logs(level="INFO")
        assert len(logs) == 2  # INFO and ERROR

    def test_log_bounded_deque(self):
        collector = EventCollector(max_log_lines=3)
        for i in range(5):
            collector.record_log(LogRecord(message=f"log-{i}"))
        logs = collector.get_logs()
        assert len(logs) == 3


class TestHandlerCallbacks:
    def test_callbacks_dict(self):
        collector = EventCollector()
        callbacks = collector.create_handler_callbacks()
        assert "START_CONVERSATION" in callbacks
        assert "ERROR" in callbacks
        assert len(callbacks) == 8

    def test_start_conversation_callback(self):
        collector = EventCollector()
        callbacks = collector.create_handler_callbacks()
        result = callbacks["START_CONVERSATION"]({"_conversation_id": "c1"})
        assert result == {}
        events = collector.get_events()
        assert len(events) == 1
        assert events[0].event_type == EVENT_CONVERSATION_START

    def test_pre_transition_callback(self):
        collector = EventCollector()
        callbacks = collector.create_handler_callbacks()
        callbacks["PRE_TRANSITION"](
            {
                "_conversation_id": "c1",
                "_current_state": "start",
                "_target_state": "end",
            }
        )
        events = collector.get_events()
        assert len(events) == 1
        assert events[0].source_state == "start"
        assert events[0].target_state == "end"

    def test_error_callback(self):
        collector = EventCollector()
        callbacks = collector.create_handler_callbacks()
        callbacks["ERROR"]({"_conversation_id": "c1", "_error": "Something broke"})
        events = collector.get_events()
        assert len(events) == 1
        assert events[0].level == "ERROR"
        assert "Something broke" in events[0].message


class TestLogSink:
    def test_create_loguru_sink(self):
        collector = EventCollector()
        sink = collector.create_loguru_sink()
        assert callable(sink)


class TestClear:
    def test_clear_all(self):
        collector = EventCollector()
        collector.record_event(MonitorEvent(event_type="test"))
        collector.record_log(LogRecord(message="test"))
        collector.record_event(
            MonitorEvent(
                event_type=EVENT_CONVERSATION_START,
                conversation_id="c1",
            )
        )
        assert collector.get_metrics().total_events > 0

        collector.clear()
        metrics = collector.get_metrics()
        assert metrics.total_events == 0
        assert metrics.active_conversations == 0
        assert len(collector.get_events()) == 0
        assert len(collector.get_logs()) == 0


class TestGetEventsSince:
    def test_returns_empty_when_no_new_events(self):
        collector = EventCollector()
        collector.record_event(MonitorEvent(event_type="test", message="old"))
        # Claim we've seen 1 event already
        events = collector.get_events_since(after_total=1)
        assert events == []

    def test_returns_new_events_only(self):
        collector = EventCollector()
        for i in range(5):
            collector.record_event(MonitorEvent(event_type="test", message=f"msg-{i}"))
        # Claim we've seen first 3 events
        events = collector.get_events_since(after_total=3)
        assert len(events) == 2
        # Newest first
        assert events[0].message == "msg-4"
        assert events[1].message == "msg-3"

    def test_respects_limit(self):
        collector = EventCollector()
        for i in range(10):
            collector.record_event(MonitorEvent(event_type="test", message=f"msg-{i}"))
        events = collector.get_events_since(after_total=0, limit=3)
        assert len(events) == 3

    def test_handles_overflow_from_bounded_deque(self):
        collector = EventCollector(max_events=5)
        for i in range(10):
            collector.record_event(MonitorEvent(event_type="test", message=f"msg-{i}"))
        # Total is 10 but deque only has 5 items. Ask for 7 new events.
        events = collector.get_events_since(after_total=3, limit=50)
        # Can only return what's in the deque (5 items), not the full 7
        assert len(events) == 5

    def test_negative_new_count(self):
        collector = EventCollector()
        collector.record_event(MonitorEvent(event_type="test", message="msg"))
        events = collector.get_events_since(after_total=999)
        assert events == []


class TestCleanup:
    def test_cleanup_without_sink(self):
        collector = EventCollector()
        # Should not raise
        collector.cleanup()
        assert collector._log_sink_id is None

    def test_cleanup_with_sink(self):
        collector = EventCollector()
        from loguru import logger as _loguru_logger

        sink = collector.create_loguru_sink()
        collector._log_sink_id = _loguru_logger.add(sink, level="DEBUG")
        assert collector._log_sink_id is not None
        collector.cleanup()
        assert collector._log_sink_id is None

    def test_cleanup_idempotent(self):
        collector = EventCollector()
        collector.cleanup()
        collector.cleanup()
        assert collector._log_sink_id is None


class TestLogSinkNoEventDouble:
    """Verify that the loguru sink only records a LogRecord, not a MonitorEvent."""

    def test_sink_records_log_only(self):
        collector = EventCollector()
        from loguru import logger as _loguru_logger

        sink = collector.create_loguru_sink()
        collector._log_sink_id = _loguru_logger.add(sink, level="DEBUG")
        try:
            _loguru_logger.info("test message from sink")
            # Should have log records but no extra events from the sink itself
            logs = collector.get_logs()
            log_msgs = [
                r.message for r in logs if "test message from sink" in r.message
            ]
            assert len(log_msgs) >= 1

            # Events should NOT contain EVENT_LOG entries from the sink
            events = collector.get_events()
            log_events = [e for e in events if e.event_type == "log"]
            assert len(log_events) == 0
        finally:
            collector.cleanup()


class TestMetricsTimestamp:
    """Verify that get_metrics returns UTC-aware timestamps."""

    def test_metrics_timestamp_is_utc(self):
        collector = EventCollector()
        metrics = collector.get_metrics()
        assert metrics.timestamp.tzinfo is not None
        assert metrics.timestamp.tzinfo == timezone.utc

    def test_metrics_after_events(self):
        collector = EventCollector()
        collector.record_event(MonitorEvent(event_type="test", message="hello"))
        metrics = collector.get_metrics()
        assert metrics.total_events == 1
        assert metrics.timestamp.tzinfo == timezone.utc


class TestLogSinkTimestamp:
    """Verify that the loguru sink preserves timezone info."""

    def test_sink_preserves_timezone(self):
        collector = EventCollector()
        from loguru import logger as _loguru_logger

        sink = collector.create_loguru_sink()
        collector._log_sink_id = _loguru_logger.add(sink, level="DEBUG")
        try:
            _loguru_logger.info("tz test")
            logs = collector.get_logs()
            tz_logs = [r for r in logs if "tz test" in r.message]
            assert len(tz_logs) >= 1
            # Timestamp should have timezone info (UTC)
            assert tz_logs[0].timestamp.tzinfo is not None
        finally:
            collector.cleanup()


class TestHandlerCallbackBehavior:
    """Additional handler callback tests for coverage."""

    def test_post_processing_callback(self):
        collector = EventCollector()
        callbacks = collector.create_handler_callbacks()
        result = callbacks["POST_PROCESSING"]({"_conversation_id": "conv-1"})
        assert result == {}
        events = collector.get_events()
        assert len(events) == 1
        assert events[0].event_type == "post_processing"

    def test_context_update_callback(self):
        collector = EventCollector()
        callbacks = collector.create_handler_callbacks()
        result = callbacks["CONTEXT_UPDATE"]({"_conversation_id": "conv-1"})
        assert result == {}
        events = collector.get_events()
        assert events[0].event_type == "context_update"

    def test_end_conversation_callback(self):
        collector = EventCollector()
        callbacks = collector.create_handler_callbacks()
        callbacks["END_CONVERSATION"]({"_conversation_id": "conv-1"})
        events = collector.get_events()
        assert events[0].event_type == "conversation_end"
        assert "conv-1" in events[0].message

    def test_post_transition_callback_is_noop(self):
        """POST_TRANSITION should return empty dict and not record events."""
        collector = EventCollector()
        callbacks = collector.create_handler_callbacks()
        result = callbacks["POST_TRANSITION"]({"_conversation_id": "conv-1"})
        assert result == {}
        assert len(collector.get_events()) == 0

    def test_pre_processing_callback(self):
        collector = EventCollector()
        callbacks = collector.create_handler_callbacks()
        result = callbacks["PRE_PROCESSING"]({"_conversation_id": "conv-1"})
        assert result == {}
        events = collector.get_events()
        assert events[0].event_type == "pre_processing"

    def test_handler_name_and_priority(self):
        collector = EventCollector()
        assert collector.handler_name == "fsm_llm_monitor"
        assert collector.handler_priority == 9999


class TestLogLevelFiltering:
    """Additional log level filter edge cases."""

    def test_warning_level_includes_error_and_critical(self):
        collector = EventCollector()
        collector.record_log(LogRecord(level="DEBUG", message="d"))
        collector.record_log(LogRecord(level="INFO", message="i"))
        collector.record_log(LogRecord(level="WARNING", message="w"))
        collector.record_log(LogRecord(level="ERROR", message="e"))
        collector.record_log(LogRecord(level="CRITICAL", message="c"))
        logs = collector.get_logs(level="WARNING")
        levels = {r.level for r in logs}
        assert levels == {"WARNING", "ERROR", "CRITICAL"}

    def test_unknown_level_filter_returns_all(self):
        collector = EventCollector()
        collector.record_log(LogRecord(level="DEBUG", message="d"))
        collector.record_log(LogRecord(level="INFO", message="i"))
        logs = collector.get_logs(level="UNKNOWN")
        # Unknown level not in level_order dict, so filter is skipped
        assert len(logs) == 2

    def test_log_limit(self):
        collector = EventCollector()
        for i in range(10):
            collector.record_log(LogRecord(level="INFO", message=f"msg-{i}"))
        logs = collector.get_logs(limit=3)
        assert len(logs) == 3


class TestThreadSafety:
    def test_concurrent_writes(self):
        collector = EventCollector(max_events=10000)
        errors = []

        def _writer(prefix: str, count: int) -> None:
            try:
                for i in range(count):
                    collector.record_event(
                        MonitorEvent(
                            event_type="test",
                            message=f"{prefix}-{i}",
                        )
                    )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=_writer, args=(f"t{i}", 100)) for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert collector.get_metrics().total_events == 500
