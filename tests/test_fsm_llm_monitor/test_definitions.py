from __future__ import annotations

"""Tests for fsm_llm_monitor.definitions models."""

from datetime import datetime

from fsm_llm_monitor.definitions import (
    ActivityItem,
    ConversationSnapshot,
    FSMSnapshot,
    InstanceInfo,
    LogRecord,
    MetricSnapshot,
    MonitorConfig,
    MonitorEvent,
    StateInfo,
    TransitionInfo,
    model_to_dict,
    normalize_message_history,
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


class TestNormalizeMessageHistory:
    def test_standard_format(self):
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        result = normalize_message_history(msgs)
        assert result == msgs

    def test_shorthand_format(self):
        msgs = [{"user": "Hi"}, {"system": "Hello"}]
        result = normalize_message_history(msgs)
        assert result == [
            {"role": "user", "content": "Hi"},
            {"role": "system", "content": "Hello"},
        ]

    def test_mixed_format(self):
        msgs = [
            {"role": "user", "content": "Hi"},
            {"system": "Hello"},
        ]
        result = normalize_message_history(msgs)
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "system"

    def test_fallback_unknown_key(self):
        msgs = [{"assistant": "Hi there"}]
        result = normalize_message_history(msgs)
        assert result == [{"role": "assistant", "content": "Hi there"}]

    def test_empty_list(self):
        assert normalize_message_history([]) == []

    def test_empty_dict(self):
        result = normalize_message_history([{}])
        assert result == []


class TestModelToDict:
    def test_none(self):
        assert model_to_dict(None) is None

    def test_dict(self):
        d = {"a": 1}
        assert model_to_dict(d) == {"a": 1}

    def test_pydantic_model(self):
        config = MonitorConfig()
        result = model_to_dict(config)
        assert isinstance(result, dict)
        assert "refresh_interval" in result

    def test_unknown_type(self):
        assert model_to_dict(42) is None
        assert model_to_dict("string") is None


class TestInstanceInfo:
    def test_created_at_is_utc(self):
        info = InstanceInfo(instance_id="test", instance_type="fsm")
        assert info.created_at.tzinfo is not None

    def test_default_status(self):
        info = InstanceInfo(instance_id="test", instance_type="fsm")
        assert info.status == "running"

    def test_custom_fields(self):
        info = InstanceInfo(
            instance_id="a1",
            instance_type="agent",
            label="MyAgent",
            status="completed",
            agent_type="ReactAgent",
        )
        assert info.label == "MyAgent"
        assert info.agent_type == "ReactAgent"
        assert info.status == "completed"


class TestStubToolConfig:
    def test_defaults(self):
        from fsm_llm_monitor.definitions import StubToolConfig

        cfg = StubToolConfig(name="search", description="Search the web")
        assert cfg.stub_response == "Tool executed successfully"

    def test_custom_response(self):
        from fsm_llm_monitor.definitions import StubToolConfig

        cfg = StubToolConfig(name="calc", description="Calculate", stub_response="42")
        assert cfg.stub_response == "42"


class TestRequestModels:
    """Verify request model defaults and serialization."""

    def test_launch_fsm_request_defaults(self):
        from fsm_llm_monitor.definitions import LaunchFSMRequest

        req = LaunchFSMRequest()
        assert req.preset_id is None
        assert req.fsm_json is None
        assert req.temperature == 0.5
        assert req.label == ""

    def test_send_message_request(self):
        from fsm_llm_monitor.definitions import SendMessageRequest

        req = SendMessageRequest(message="hello", conversation_id="c1")
        assert req.message == "hello"

    def test_launch_agent_request_defaults(self):
        from fsm_llm_monitor.definitions import LaunchAgentRequest

        req = LaunchAgentRequest(task="Do something")
        assert req.agent_type == "ReactAgent"
        assert req.max_iterations == 10
        assert req.tools == []

    def test_workflow_advance_request(self):
        from fsm_llm_monitor.definitions import WorkflowAdvanceRequest

        req = WorkflowAdvanceRequest(workflow_instance_id="w1", user_input="yes")
        assert req.user_input == "yes"


class TestDashboardConfigModels:
    """Tests for custom dashboard configuration models."""

    def test_dashboard_panel(self):
        from fsm_llm_monitor.definitions import DashboardPanel

        p = DashboardPanel(panel_id="p1", title="CPU", panel_type="gauge", metric="cpu")
        assert p.panel_id == "p1"
        assert p.panel_type == "gauge"

    def test_dashboard_alert(self):
        from fsm_llm_monitor.definitions import DashboardAlert

        a = DashboardAlert(
            alert_id="a1", metric="errors", condition=">", threshold=10.0
        )
        assert a.threshold == 10.0
        assert a.condition == ">"

    def test_dashboard_config(self):
        from fsm_llm_monitor.definitions import (
            DashboardAlert,
            DashboardConfig,
            DashboardPanel,
        )

        cfg = DashboardConfig(
            name="Test",
            panels=[DashboardPanel(panel_id="p1", title="P1", metric="m1")],
            alerts=[DashboardAlert(alert_id="a1", metric="m1", threshold=5.0)],
        )
        assert cfg.name == "Test"
        assert len(cfg.panels) == 1
        assert len(cfg.alerts) == 1
        assert cfg.refresh_interval_seconds == 30

    def test_dashboard_config_defaults(self):
        from fsm_llm_monitor.definitions import DashboardConfig

        cfg = DashboardConfig()
        assert cfg.panels == []
        assert cfg.alerts == []
        assert cfg.retention_hours == 24


class TestActivityItem:
    """Tests for the ActivityItem model."""

    def test_create_fsm_conversation(self):
        item = ActivityItem(
            item_id="conv-123",
            item_type="fsm_conversation",
            instance_id="inst-1",
            label="conv-123",
            status="active",
            current_step="greeting",
            detail="Greeting state",
            message_count=5,
        )
        assert item.item_id == "conv-123"
        assert item.item_type == "fsm_conversation"
        assert item.message_count == 5
        assert item.is_terminal is False

    def test_create_agent_task(self):
        item = ActivityItem(
            item_id="agent-456",
            item_type="agent_task",
            instance_id="agent-456",
            label="ReactAgent-abc",
            status="running",
            current_step="iter 3/10",
            detail="ReactAgent",
            message_count=3,
        )
        assert item.item_type == "agent_task"
        assert item.detail == "ReactAgent"

    def test_create_workflow_instance(self):
        item = ActivityItem(
            item_id="wf-789",
            item_type="workflow_instance",
            instance_id="wf-eng-1",
            label="Order Processing",
            status="completed",
            current_step="done",
            is_terminal=True,
        )
        assert item.item_type == "workflow_instance"
        assert item.is_terminal is True

    def test_defaults(self):
        item = ActivityItem(item_id="test", item_type="fsm_conversation")
        assert item.instance_id == ""
        assert item.label == ""
        assert item.status == "active"
        assert item.current_step == ""
        assert item.detail == ""
        assert item.message_count == 0
        assert item.is_terminal is False

    def test_model_dump(self):
        item = ActivityItem(
            item_id="test",
            item_type="agent_task",
            status="failed",
            is_terminal=True,
        )
        d = item.model_dump()
        assert d["item_id"] == "test"
        assert d["item_type"] == "agent_task"
        assert d["status"] == "failed"
        assert d["is_terminal"] is True


class TestMetricSnapshotExtended:
    """Tests for the extended MetricSnapshot fields."""

    def test_extended_fields_default(self):
        snap = MetricSnapshot()
        assert snap.active_agents == 0
        assert snap.active_workflows == 0
        assert snap.total_agent_iterations == 0
        assert snap.total_tool_calls == 0
        assert snap.total_workflow_steps == 0

    def test_extended_fields_set(self):
        snap = MetricSnapshot(
            active_agents=2,
            active_workflows=1,
            total_agent_iterations=15,
            total_tool_calls=30,
            total_workflow_steps=8,
        )
        assert snap.active_agents == 2
        assert snap.active_workflows == 1
        assert snap.total_agent_iterations == 15
        assert snap.total_tool_calls == 30
        assert snap.total_workflow_steps == 8

    def test_extended_fields_in_dump(self):
        snap = MetricSnapshot(active_agents=1, total_tool_calls=5)
        d = snap.model_dump()
        assert "active_agents" in d
        assert "active_workflows" in d
        assert "total_agent_iterations" in d
        assert "total_tool_calls" in d
        assert "total_workflow_steps" in d
