from __future__ import annotations

"""Tests for fsm_llm_monitor.definitions models."""

from datetime import datetime

import pytest

from fsm_llm_monitor.definitions import (
    ConversationSnapshot,
    FSMSnapshot,
    LogRecord,
    MetricSnapshot,
    MonitorConfig,
    MonitorEvent,
    StateInfo,
    TransitionInfo,
)


class TestMonitorEvent:
    def test_create_minimal(self):
        event = MonitorEvent(event_type="test")
        assert event.event_type == "test"
        assert event.conversation_id is None
        assert event.data == {}
        assert event.level == "INFO"
        assert isinstance(event.timestamp, datetime)

    def test_create_full(self):
        event = MonitorEvent(
            event_type="state_transition",
            conversation_id="conv-123",
            source_state="start",
            target_state="end",
            data={"key": "value"},
            level="WARNING",
            message="Transition occurred",
        )
        assert event.event_type == "state_transition"
        assert event.conversation_id == "conv-123"
        assert event.source_state == "start"
        assert event.target_state == "end"
        assert event.data == {"key": "value"}

    def test_serialization_roundtrip(self):
        event = MonitorEvent(
            event_type="test",
            conversation_id="c1",
            message="hello",
        )
        data = event.model_dump()
        restored = MonitorEvent(**data)
        assert restored.event_type == event.event_type
        assert restored.conversation_id == event.conversation_id


class TestLogRecord:
    def test_create_minimal(self):
        record = LogRecord()
        assert record.level == "INFO"
        assert record.message == ""
        assert record.module == ""

    def test_create_full(self):
        record = LogRecord(
            level="ERROR",
            message="Something failed",
            module="fsm",
            function="process",
            line=42,
            conversation_id="conv-1",
        )
        assert record.level == "ERROR"
        assert record.line == 42
        assert record.conversation_id == "conv-1"


class TestMetricSnapshot:
    def test_defaults(self):
        snap = MetricSnapshot()
        assert snap.active_conversations == 0
        assert snap.total_events == 0
        assert snap.total_errors == 0
        assert snap.total_transitions == 0
        assert snap.events_per_type == {}
        assert snap.states_visited == {}

    def test_with_data(self):
        snap = MetricSnapshot(
            active_conversations=3,
            total_events=100,
            total_errors=2,
            total_transitions=50,
            events_per_type={"state_transition": 50, "error": 2},
            states_visited={"start": 10, "end": 5},
        )
        assert snap.active_conversations == 3
        assert snap.events_per_type["state_transition"] == 50


class TestConversationSnapshot:
    def test_create_minimal(self):
        snap = ConversationSnapshot(conversation_id="conv-1")
        assert snap.conversation_id == "conv-1"
        assert snap.current_state == ""
        assert snap.is_terminal is False
        assert snap.stack_depth == 1

    def test_create_full(self):
        snap = ConversationSnapshot(
            conversation_id="conv-1",
            current_state="greeting",
            state_description="Greet the user",
            is_terminal=False,
            context_data={"name": "Alice"},
            message_history=[{"user": "Hi"}, {"system": "Hello!"}],
            stack_depth=2,
        )
        assert snap.current_state == "greeting"
        assert len(snap.message_history) == 2
        assert snap.stack_depth == 2


class TestStateInfo:
    def test_create(self):
        info = StateInfo(
            state_id="start",
            description="Initial state",
            purpose="Welcome user",
            is_initial=True,
            transition_count=2,
        )
        assert info.state_id == "start"
        assert info.is_initial is True
        assert info.transition_count == 2

    def test_with_transitions(self):
        trans = TransitionInfo(
            target_state="next",
            description="Move forward",
            priority=100,
        )
        info = StateInfo(
            state_id="start",
            transitions=[trans],
            transition_count=1,
        )
        assert len(info.transitions) == 1
        assert info.transitions[0].target_state == "next"


class TestTransitionInfo:
    def test_create(self):
        trans = TransitionInfo(
            target_state="end",
            description="Finish conversation",
            priority=50,
            condition_count=2,
            has_logic=True,
        )
        assert trans.target_state == "end"
        assert trans.has_logic is True


class TestFSMSnapshot:
    def test_create_minimal(self):
        snap = FSMSnapshot()
        assert snap.name == ""
        assert snap.state_count == 0
        assert snap.states == []

    def test_create_full(self):
        snap = FSMSnapshot(
            name="TestFSM",
            description="A test FSM",
            version="4.1",
            initial_state="start",
            persona="A helpful bot",
            state_count=2,
            states=[
                StateInfo(state_id="start", is_initial=True),
                StateInfo(state_id="end", is_terminal=True),
            ],
        )
        assert snap.name == "TestFSM"
        assert snap.state_count == 2
        assert snap.states[0].is_initial is True


class TestMonitorConfig:
    def test_defaults(self):
        config = MonitorConfig()
        assert config.refresh_interval == 1.0
        assert config.max_events == 1000
        assert config.max_log_lines == 5000
        assert config.log_level == "INFO"
        assert config.show_internal_keys is False
        assert config.auto_scroll_logs is True

    def test_custom_config(self):
        config = MonitorConfig(
            refresh_interval=0.5,
            max_events=500,
            log_level="DEBUG",
            show_internal_keys=True,
        )
        assert config.refresh_interval == 0.5
        assert config.max_events == 500
        assert config.log_level == "DEBUG"
        assert config.show_internal_keys is True
