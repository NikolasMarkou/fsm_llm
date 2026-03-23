from __future__ import annotations

"""Tests for fsm_llm_monitor.collector."""

import threading

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
