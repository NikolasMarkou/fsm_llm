from __future__ import annotations

"""Tests for fsm_llm.monitor.collector."""

import threading
from datetime import UTC
from typing import Any

from fsm_llm.handlers import HandlerTiming
from fsm_llm.monitor.collector import EventCollector
from fsm_llm.monitor.constants import (
    EVENT_AGENT_ITERATION,
    EVENT_AGENT_TOOL_CALL,
    EVENT_CONVERSATION_END,
    EVENT_CONVERSATION_START,
    EVENT_ERROR,
    EVENT_STATE_TRANSITION,
    EVENT_WORKFLOW_ADVANCED,
)
from fsm_llm.monitor.definitions import LogRecord, MonitorEvent
from fsm_llm.monitor.instance_manager import (
    ManagedAgent,
    _MonitorHandler,
    register_monitor_handlers,
    unregister_monitor_handlers,
)
from tests.test_fsm_llm_monitor.support import _manager


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
        assert len(callbacks) == 7

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
        # The core passes the target state to the handler, not the context.
        callbacks["PRE_TRANSITION"](
            {"_conversation_id": "c1", "_current_state": "start"}, target_state="end"
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

            # Log records go to the log channel, never into the event stream
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
        assert metrics.timestamp.tzinfo == UTC

    def test_metrics_after_events(self):
        collector = EventCollector()
        collector.record_event(MonitorEvent(event_type="test", message="hello"))
        metrics = collector.get_metrics()
        assert metrics.total_events == 1
        assert metrics.timestamp.tzinfo == UTC


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

    def test_there_is_no_post_transition_observer(self):
        """PRE_TRANSITION records the transition; nothing observes POST."""
        collector = EventCollector()
        assert "POST_TRANSITION" not in collector.create_handler_callbacks()
        assert not hasattr(collector, "_on_post_transition")

    def test_pre_processing_callback(self):
        collector = EventCollector()
        callbacks = collector.create_handler_callbacks()
        result = callbacks["PRE_PROCESSING"]({"_conversation_id": "conv-1"})
        assert result == {}
        events = collector.get_events()
        assert events[0].event_type == "pre_processing"


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


class TestCollectorExtendedMetrics:
    """Tests for agent/workflow metric counters."""

    def test_agent_iteration_counter(self):
        collector = EventCollector()
        for i in range(5):
            collector.record_event(
                MonitorEvent(event_type=EVENT_AGENT_ITERATION, message=f"iter {i}")
            )
        metrics = collector.get_metrics()
        assert metrics.total_agent_iterations == 5

    def test_tool_call_counter(self):
        collector = EventCollector()
        for _ in range(3):
            collector.record_event(
                MonitorEvent(event_type=EVENT_AGENT_TOOL_CALL, message="tool")
            )
        metrics = collector.get_metrics()
        assert metrics.total_tool_calls == 3

    def test_workflow_step_counter(self):
        collector = EventCollector()
        for _ in range(4):
            collector.record_event(
                MonitorEvent(event_type=EVENT_WORKFLOW_ADVANCED, message="step")
            )
        metrics = collector.get_metrics()
        assert metrics.total_workflow_steps == 4

    def test_clear_resets_extended_counters(self):
        collector = EventCollector()
        collector.record_event(
            MonitorEvent(event_type=EVENT_AGENT_ITERATION, message="iter")
        )
        collector.record_event(
            MonitorEvent(event_type=EVENT_AGENT_TOOL_CALL, message="tool")
        )
        collector.record_event(
            MonitorEvent(event_type=EVENT_WORKFLOW_ADVANCED, message="step")
        )
        collector.clear()
        metrics = collector.get_metrics()
        assert metrics.total_agent_iterations == 0
        assert metrics.total_tool_calls == 0
        assert metrics.total_workflow_steps == 0


class _FakeHandlerSystem:
    def __init__(self) -> None:
        self.handlers: list[Any] = []


class _FakeAPI:
    """Just enough of fsm_llm.API for handler registration."""

    def __init__(self) -> None:
        self.handler_system = _FakeHandlerSystem()

    def register_handler(self, handler: Any) -> None:
        self.handler_system.handlers.append(handler)


def _fire(api: _FakeAPI, timing: HandlerTiming, current: str, target: str | None):
    """Dispatch like the core: should_execute then execute, per handler."""
    ctx = {"_conversation_id": "c1"}
    for h in api.handler_system.handlers:
        if h.should_execute(timing, current, target, ctx, None):
            h.execute(dict(ctx))


class TestTransitionStates:
    def test_handler_passes_core_states_to_collector(self):
        collector = EventCollector()
        api = _FakeAPI()
        register_monitor_handlers(api, collector)

        _fire(api, HandlerTiming.START_CONVERSATION, "greet", None)
        _fire(api, HandlerTiming.PRE_TRANSITION, "greet", "checkout")

        transitions = [
            e for e in collector.get_events() if e.event_type == EVENT_STATE_TRANSITION
        ]
        assert transitions[0].source_state == "greet"
        assert transitions[0].target_state == "checkout"
        visited = collector.get_metrics().states_visited
        assert visited == {"greet": 1, "checkout": 1}

    def test_states_survive_a_handler_timeout(self):
        """F-HV-01: with ``handler_timeout`` the handler runs in a worker thread;
        the states set by ``should_execute`` must still reach ``execute``."""
        from fsm_llm.handlers import HandlerSystem

        collector = EventCollector()
        api = _FakeAPI()
        api.handler_system = HandlerSystem(handler_timeout=5)
        api.register_handler = api.handler_system.register_handler
        register_monitor_handlers(api, collector)

        api.handler_system.execute_handlers(
            HandlerTiming.PRE_TRANSITION,
            "greet",
            "checkout",
            {"_conversation_id": "c1"},
        )

        transitions = [
            e for e in collector.get_events() if e.event_type == EVENT_STATE_TRANSITION
        ]
        assert [(e.source_state, e.target_state) for e in transitions] == [
            ("greet", "checkout")
        ]

    def test_capture_log_records_transition_and_end_state(self):
        agent = ManagedAgent("a1")
        collector = EventCollector()
        from fsm_llm.monitor.instance_manager import _build_context_capture_handlers

        api = _FakeAPI()
        for h in _build_context_capture_handlers(collector, agent):
            api.register_handler(h)
        _fire(api, HandlerTiming.START_CONVERSATION, "think", None)
        _fire(api, HandlerTiming.PRE_TRANSITION, "think", "act")
        _fire(api, HandlerTiming.END_CONVERSATION, "conclude", None)

        kinds = [
            (e["type"], e.get("state") or e.get("target"))
            for e in agent.conversation_log
        ]
        assert ("start", "think") in kinds
        assert ("transition", "act") in kinds
        assert ("end", "conclude") in kinds

    def test_end_event_reports_final_state(self):
        collector = EventCollector()
        api = _FakeAPI()
        register_monitor_handlers(api, collector)
        _fire(api, HandlerTiming.END_CONVERSATION, "done", None)
        end = collector.get_events()[0]
        assert end.data["state"] == "done"


class TestStreamCursors:
    def test_burst_is_delivered_in_order_without_repeats(self):
        c = EventCollector(max_log_lines=1000)
        for i in range(120):
            c.record_log(LogRecord(level="INFO", message=str(i)))
        seen: list[str] = []
        cursor = 0
        for _ in range(5):
            logs, cursor = c.logs_after(cursor, limit=50)
            seen.extend(r.message for r in logs)
        assert seen == [str(i) for i in range(120)]

    def test_dropped_records_are_skipped_and_cleared_collector_restarts(self):
        c = EventCollector(max_events=10)
        for i in range(25):
            c.record_event(MonitorEvent(event_type="t", message=str(i)))
        events, cursor = c.events_after(0, limit=50)
        assert [e.message for e in events] == [str(i) for i in range(15, 25)]
        assert cursor == 25
        c.clear()
        c.record_event(MonitorEvent(event_type="t", message="new"))
        events, cursor = c.events_after(cursor, limit=50)
        assert [e.message for e in events] == ["new"]

    def test_log_level_filter_is_case_insensitive_and_ranks_success(self):
        c = EventCollector()
        for level in ("DEBUG", "INFO", "SUCCESS", "ERROR"):
            c.record_log(LogRecord(level=level, message=level))
        assert {r.level for r in c.get_logs(level="info")} == {
            "INFO",
            "SUCCESS",
            "ERROR",
        }
        assert {r.level for r in c.get_logs(level="warning")} == {"ERROR"}


class TestHandlerLifecycle:
    def test_register_twice_does_not_stack(self):
        collector = EventCollector()
        api = _FakeAPI()
        register_monitor_handlers(api, collector)
        register_monitor_handlers(api, collector)
        _fire(api, HandlerTiming.PRE_TRANSITION, "a", "b")
        assert collector.get_metrics().total_transitions == 1

    def test_unregister_stops_recording(self):
        collector = EventCollector()
        api = _FakeAPI()
        register_monitor_handlers(api, collector)
        assert unregister_monitor_handlers(api, collector) == 7
        _fire(api, HandlerTiming.PRE_TRANSITION, "a", "b")
        assert collector.get_metrics().total_events == 0

    def test_attach_api_twice_and_close(self):
        api = _FakeAPI()
        mgr = _manager()
        mgr.attach_api(api)
        mgr.attach_api(api)
        _fire(api, HandlerTiming.PRE_TRANSITION, "a", "b")
        assert mgr.global_collector.get_metrics().total_transitions == 1
        mgr.close()
        _fire(api, HandlerTiming.PRE_TRANSITION, "b", "c")
        assert mgr.global_collector.get_metrics().total_transitions == 1

    def test_attaching_another_api_switches_the_previous_one_off(self):
        first, second = _FakeAPI(), _FakeAPI()
        mgr = _manager()
        mgr.attach_api(first)
        mgr.attach_api(second)
        _fire(first, HandlerTiming.PRE_TRANSITION, "a", "b")
        assert mgr.global_collector.get_metrics().total_transitions == 0
        _fire(second, HandlerTiming.PRE_TRANSITION, "a", "b")
        assert mgr.global_collector.get_metrics().total_transitions == 1
        mgr.global_collector.cleanup()

    def test_handler_never_writes_context(self):
        handler = _MonitorHandler(
            "h", HandlerTiming.PRE_TRANSITION, lambda *a, **k: None, EventCollector()
        )
        assert handler.should_execute(HandlerTiming.PRE_TRANSITION, "a", "b", {})
        assert handler.execute({"x": 1}) == {}
