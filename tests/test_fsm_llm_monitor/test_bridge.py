from __future__ import annotations

"""Tests for fsm_llm_monitor.bridge."""

import json
import tempfile
from unittest.mock import MagicMock

from fsm_llm_monitor.bridge import MonitorBridge, _fsm_dict_to_snapshot
from fsm_llm_monitor.definitions import (
    MonitorConfig,
)


def _minimal_fsm_dict() -> dict:
    return {
        "name": "TestFSM",
        "description": "A test FSM",
        "version": "4.1",
        "initial_state": "start",
        "persona": "Test bot",
        "states": {
            "start": {
                "id": "start",
                "description": "Starting state",
                "purpose": "Welcome user",
                "transitions": [
                    {
                        "target_state": "end",
                        "description": "Move to end",
                        "priority": 100,
                        "conditions": [
                            {
                                "description": "Always",
                                "logic": {"==": [1, 1]},
                            }
                        ],
                    }
                ],
            },
            "end": {
                "id": "end",
                "description": "End state",
                "purpose": "Say goodbye",
                "transitions": [],
            },
        },
    }


class TestFSMDictToSnapshot:
    def test_basic_conversion(self):
        snap = _fsm_dict_to_snapshot(_minimal_fsm_dict())
        assert snap.name == "TestFSM"
        assert snap.version == "4.1"
        assert snap.initial_state == "start"
        assert snap.state_count == 2

    def test_state_flags(self):
        snap = _fsm_dict_to_snapshot(_minimal_fsm_dict())
        start = next(s for s in snap.states if s.state_id == "start")
        end = next(s for s in snap.states if s.state_id == "end")
        assert start.is_initial is True
        assert start.is_terminal is False
        assert end.is_initial is False
        assert end.is_terminal is True

    def test_transition_info(self):
        snap = _fsm_dict_to_snapshot(_minimal_fsm_dict())
        start = next(s for s in snap.states if s.state_id == "start")
        assert start.transition_count == 1
        assert len(start.transitions) == 1
        assert start.transitions[0].target_state == "end"
        assert start.transitions[0].has_logic is True
        assert start.transitions[0].condition_count == 1

    def test_empty_fsm(self):
        snap = _fsm_dict_to_snapshot({"states": {}})
        assert snap.state_count == 0
        assert snap.states == []


class TestMonitorBridge:
    def test_create_without_api(self):
        bridge = MonitorBridge()
        assert bridge.connected is False
        assert bridge.get_active_conversations() == []

    def test_create_with_config(self):
        config = MonitorConfig(refresh_interval=0.5, max_events=100)
        bridge = MonitorBridge(config=config)
        assert bridge.config.refresh_interval == 0.5
        assert bridge.config.max_events == 100

    def test_get_metrics_empty(self):
        bridge = MonitorBridge()
        metrics = bridge.get_metrics()
        assert metrics.total_events == 0
        assert metrics.active_conversations == 0

    def test_load_fsm_from_file(self):
        bridge = MonitorBridge()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(_minimal_fsm_dict(), f)
            f.flush()
            snap = bridge.load_fsm_from_file(f.name)

        assert snap is not None
        assert snap.name == "TestFSM"
        assert snap.state_count == 2

    def test_load_fsm_from_file_not_found(self):
        bridge = MonitorBridge()
        snap = bridge.load_fsm_from_file("/nonexistent/path.json")
        assert snap is None

    def test_load_fsm_from_dict(self):
        bridge = MonitorBridge()
        snap = bridge.load_fsm_from_dict(_minimal_fsm_dict())
        assert snap is not None
        assert snap.name == "TestFSM"

    def test_load_fsm_from_dict_invalid(self):
        bridge = MonitorBridge()
        # Should not crash on bad data
        snap = bridge.load_fsm_from_dict({"invalid": True})
        assert snap is not None
        assert snap.state_count == 0

    def test_get_recent_events_empty(self):
        bridge = MonitorBridge()
        events = bridge.get_recent_events()
        assert events == []

    def test_disconnect(self):
        bridge = MonitorBridge()
        bridge.disconnect()
        assert bridge.connected is False

    def test_get_conversation_snapshot_no_api(self):
        bridge = MonitorBridge()
        snap = bridge.get_conversation_snapshot("conv-1")
        assert snap is None

    def test_get_all_conversation_snapshots_no_api(self):
        bridge = MonitorBridge()
        snaps = bridge.get_all_conversation_snapshots()
        assert snaps == []


class TestMonitorBridgeWithMockAPI:
    def _make_mock_api(self):
        api = MagicMock()
        api.list_active_conversations.return_value = ["c1", "c2"]
        api.get_stack_depth.return_value = 1
        api.fsm_manager.get_complete_conversation.return_value = {
            "current_state": {
                "id": "greeting",
                "description": "Greet user",
                "is_terminal": False,
            },
            "collected_data": {"name": "Alice"},
            "conversation_history": [{"user": "Hi"}, {"system": "Hello!"}],
            "last_extraction_response": None,
            "last_transition_decision": None,
            "last_response_generation": None,
        }
        # Make register_handler a no-op
        api.register_handler = MagicMock()
        return api

    def test_connect(self):
        api = self._make_mock_api()
        bridge = MonitorBridge(api=api)
        assert bridge.connected is True
        # Should have registered 8 handlers (one per timing)
        assert api.register_handler.call_count == 8

    def test_get_active_conversations(self):
        api = self._make_mock_api()
        bridge = MonitorBridge(api=api)
        convs = bridge.get_active_conversations()
        assert convs == ["c1", "c2"]

    def test_get_conversation_snapshot(self):
        api = self._make_mock_api()
        bridge = MonitorBridge(api=api)
        snap = bridge.get_conversation_snapshot("c1")
        assert snap is not None
        assert snap.conversation_id == "c1"
        assert snap.current_state == "greeting"
        assert snap.context_data == {"name": "Alice"}
        assert len(snap.message_history) == 2
        assert snap.message_history[0] == {"role": "user", "content": "Hi"}
        assert snap.message_history[1] == {"role": "system", "content": "Hello!"}

    def test_get_all_conversation_snapshots(self):
        api = self._make_mock_api()
        bridge = MonitorBridge(api=api)
        snaps = bridge.get_all_conversation_snapshots()
        assert len(snaps) == 2

    def test_api_exception_handling(self):
        api = self._make_mock_api()
        api.list_active_conversations.side_effect = RuntimeError("boom")
        bridge = MonitorBridge(api=api)
        # Should not raise, returns empty
        convs = bridge.get_active_conversations()
        assert convs == []

    def test_conversation_snapshot_exception(self):
        api = self._make_mock_api()
        api.fsm_manager.get_complete_conversation.side_effect = RuntimeError("fail")
        bridge = MonitorBridge(api=api)
        snap = bridge.get_conversation_snapshot("c1")
        assert snap is None


class TestMonitorBridgeConnectNone:
    """Tests for the connect(None) edge case."""

    def test_connect_none_does_not_set_connected(self):
        bridge = MonitorBridge()
        bridge.connect(None)
        assert bridge.connected is False

    def test_create_with_none_api(self):
        bridge = MonitorBridge(api=None)
        assert bridge.connected is False
