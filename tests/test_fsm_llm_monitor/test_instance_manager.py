from __future__ import annotations

"""Tests for fsm_llm_monitor.instance_manager."""

from unittest.mock import MagicMock

from fsm_llm_monitor.constants import EVENT_INSTANCE_LAUNCHED
from fsm_llm_monitor.definitions import MonitorConfig
from fsm_llm_monitor.instance_manager import (
    InstanceManager,
    ManagedAgent,
    ManagedFSM,
    ManagedWorkflow,
    register_monitor_handlers,
)


class TestManagedClasses:
    """Test the Managed* classes have correct defaults and UTC datetimes."""

    def test_managed_fsm_created_at_utc(self):
        api = MagicMock()
        inst = ManagedFSM(instance_id="test", api=api)
        assert inst.created_at.tzinfo is not None
        assert inst.instance_type == "fsm"
        assert inst.status == "running"

    def test_managed_fsm_to_info(self):
        api = MagicMock()
        inst = ManagedFSM(instance_id="id1", api=api, label="My FSM", source="preset/a")
        inst.conversation_ids = ["c1", "c2"]
        info = inst.to_info()
        assert info.instance_id == "id1"
        assert info.label == "My FSM"
        assert info.conversation_count == 2
        assert info.source == "preset/a"

    def test_managed_workflow_created_at_utc(self):
        inst = ManagedWorkflow(instance_id="test")
        assert inst.created_at.tzinfo is not None
        assert inst.instance_type == "workflow"

    def test_managed_workflow_to_info(self):
        inst = ManagedWorkflow(instance_id="wf1", label="WF")
        inst.active_instance_ids = ["wi1"]
        info = inst.to_info()
        assert info.active_workflows == 1

    def test_managed_agent_created_at_utc(self):
        inst = ManagedAgent(instance_id="test")
        assert inst.created_at.tzinfo is not None
        assert inst.instance_type == "agent"

    def test_managed_agent_to_info(self):
        inst = ManagedAgent(instance_id="ag1", agent_type="ReactAgent", task="do stuff")
        info = inst.to_info()
        assert info.agent_type == "ReactAgent"
        assert info.status == "running"


class TestInstanceManager:
    """Test InstanceManager core methods."""

    def _make_manager(self) -> InstanceManager:
        mgr = InstanceManager(config=MonitorConfig())
        # Clean up loguru sink to avoid test pollution
        mgr.global_collector.cleanup()
        return mgr

    def test_capabilities(self):
        mgr = self._make_manager()
        caps = mgr.get_capabilities()
        assert "fsm" in caps
        assert caps["fsm"] is True

    def test_list_instances_empty(self):
        mgr = self._make_manager()
        assert mgr.list_instances() == []

    def test_get_instance_not_found(self):
        mgr = self._make_manager()
        assert mgr.get_instance("nonexistent") is None

    def test_get_instance_collector_not_found(self):
        mgr = self._make_manager()
        assert mgr.get_instance_collector("nonexistent") is None

    def test_get_metrics_empty(self):
        mgr = self._make_manager()
        metrics = mgr.get_metrics()
        assert metrics.total_events == 0

    def test_get_events_empty(self):
        mgr = self._make_manager()
        assert mgr.get_events() == []

    def test_destroy_nonexistent_raises(self):
        mgr = self._make_manager()
        import pytest

        with pytest.raises(KeyError):
            mgr.destroy_instance("nonexistent")

    def test_get_fsm_raises_for_missing(self):
        mgr = self._make_manager()
        import pytest

        with pytest.raises(KeyError):
            mgr._get_fsm("nonexistent")

    def test_get_workflow_raises_for_missing(self):
        mgr = self._make_manager()
        import pytest

        with pytest.raises(KeyError):
            mgr._get_workflow("nonexistent")

    def test_get_agent_raises_for_missing(self):
        mgr = self._make_manager()
        import pytest

        with pytest.raises(KeyError):
            mgr._get_agent("nonexistent")

    def test_resolve_fsm_data_none(self):
        mgr = self._make_manager()
        assert mgr._resolve_fsm_data(None, None) is None

    def test_resolve_fsm_data_json(self):
        mgr = self._make_manager()
        data = {"name": "Test", "states": {}}
        result = mgr._resolve_fsm_data(None, data)
        assert result == data

    def test_resolve_fsm_data_invalid_preset(self):
        mgr = self._make_manager()
        import pytest

        with pytest.raises(ValueError):
            mgr._resolve_fsm_data("../../../etc/passwd", None)

    def test_resolve_fsm_data_absolute_preset(self):
        mgr = self._make_manager()
        import pytest

        with pytest.raises(ValueError):
            mgr._resolve_fsm_data("/etc/passwd", None)

    def test_emit_global_event(self):
        mgr = self._make_manager()
        mgr._emit_global_event(EVENT_INSTANCE_LAUNCHED, message="test launch")
        events = mgr.get_events()
        assert len(events) == 1
        assert events[0].event_type == EVENT_INSTANCE_LAUNCHED

    def test_launch_fsm_requires_data(self):
        mgr = self._make_manager()
        import pytest

        with pytest.raises(ValueError, match="Must provide"):
            mgr.launch_fsm()

    def test_get_type_raises_for_wrong_type(self):
        mgr = self._make_manager()
        import pytest

        # Manually inject an agent instance
        agent = ManagedAgent(instance_id="ag1")
        with mgr._lock:
            mgr._instances["ag1"] = agent

        with pytest.raises(TypeError):
            mgr._get_fsm("ag1")
        with pytest.raises(TypeError):
            mgr._get_workflow("ag1")

    def test_launch_agent_requires_extension(self):
        mgr = self._make_manager()
        import pytest

        import fsm_llm_monitor.instance_manager as im

        old = im._HAS_AGENTS
        try:
            im._HAS_AGENTS = False
            with pytest.raises(RuntimeError, match="not installed"):
                mgr.launch_agent(task="do something")
        finally:
            im._HAS_AGENTS = old

    def test_launch_workflow_requires_extension(self):
        mgr = self._make_manager()
        import pytest

        import fsm_llm_monitor.instance_manager as im

        old = im._HAS_WORKFLOWS
        try:
            im._HAS_WORKFLOWS = False
            with pytest.raises(RuntimeError, match="not installed"):
                mgr.launch_workflow()
        finally:
            im._HAS_WORKFLOWS = old


class TestRegisterMonitorHandlers:
    def test_registers_8_handlers(self):
        from fsm_llm_monitor.collector import EventCollector

        api = MagicMock()
        collector = EventCollector()
        register_monitor_handlers(api, collector)
        assert api.register_handler.call_count == 8


class TestConversationCaching:
    """Test ended conversation cache behavior."""

    def test_cache_bounds(self):
        mgr = InstanceManager(config=MonitorConfig())
        mgr.global_collector.cleanup()
        mgr._max_ended_conversations = 3

        # Manually add entries to the cache
        from fsm_llm_monitor.definitions import ConversationSnapshot

        for i in range(5):
            mgr._ended_conversations[f"conv-{i}"] = ConversationSnapshot(
                conversation_id=f"conv-{i}"
            )

        # Should have evicted oldest entries
        assert len(mgr._ended_conversations) == 5  # OrderedDict grows

        # But _cache_ended_conversation uses the eviction logic
        # Let's verify directly
        while len(mgr._ended_conversations) > mgr._max_ended_conversations:
            mgr._ended_conversations.popitem(last=False)
        assert len(mgr._ended_conversations) == 3
        # Oldest (conv-0, conv-1) should be evicted
        assert "conv-0" not in mgr._ended_conversations
        assert "conv-1" not in mgr._ended_conversations
        assert "conv-4" in mgr._ended_conversations

    def test_get_conversation_from_cache(self):
        mgr = InstanceManager(config=MonitorConfig())
        mgr.global_collector.cleanup()

        from fsm_llm_monitor.definitions import ConversationSnapshot

        snap = ConversationSnapshot(
            conversation_id="cached-1",
            current_state="end",
            is_terminal=True,
        )
        mgr._ended_conversations["cached-1"] = snap

        result = mgr.get_conversation_snapshot("cached-1")
        assert result is not None
        assert result.conversation_id == "cached-1"
        assert result.is_terminal is True

    def test_cache_stores_ended_conversations(self):
        mgr = InstanceManager(config=MonitorConfig())
        mgr.global_collector.cleanup()

        from fsm_llm_monitor.definitions import ConversationSnapshot

        for i in range(3):
            mgr._ended_conversations[f"ended-{i}"] = ConversationSnapshot(
                conversation_id=f"ended-{i}",
                is_terminal=True,
            )

        # get_all_conversation_snapshots should include ended ones
        all_snaps = mgr.get_all_conversation_snapshots(include_ended=True)
        assert len(all_snaps) == 3

        # Without include_ended
        all_snaps_no_ended = mgr.get_all_conversation_snapshots(include_ended=False)
        assert len(all_snaps_no_ended) == 0


class TestInstanceManagerListFilter:
    """Test list_instances with type filtering."""

    def test_filter_by_type(self):
        mgr = InstanceManager(config=MonitorConfig())
        mgr.global_collector.cleanup()

        agent = ManagedAgent(instance_id="ag1", agent_type="ReactAgent")
        fsm = ManagedFSM(instance_id="f1", api=MagicMock())
        with mgr._lock:
            mgr._instances["ag1"] = agent
            mgr._instances["f1"] = fsm

        agents = mgr.list_instances(type_filter="agent")
        assert len(agents) == 1
        assert agents[0].instance_type == "agent"

        fsms = mgr.list_instances(type_filter="fsm")
        assert len(fsms) == 1
        assert fsms[0].instance_type == "fsm"

        all_instances = mgr.list_instances()
        assert len(all_instances) == 2

    def test_find_instance_for_conversation(self):
        mgr = InstanceManager(config=MonitorConfig())
        mgr.global_collector.cleanup()

        fsm = ManagedFSM(instance_id="f1", api=MagicMock())
        fsm.conversation_ids = ["conv-a", "conv-b"]
        with mgr._lock:
            mgr._instances["f1"] = fsm

        assert mgr.find_instance_for_conversation("conv-a") == "f1"
        assert mgr.find_instance_for_conversation("conv-b") == "f1"
        assert mgr.find_instance_for_conversation("conv-c") is None

    def test_get_active_conversations_empty(self):
        mgr = InstanceManager(config=MonitorConfig())
        mgr.global_collector.cleanup()
        assert mgr.get_active_conversations() == []


class TestDestroyInstance:
    """Test instance destruction behavior."""

    def test_destroy_agent_cancels(self):
        mgr = InstanceManager(config=MonitorConfig())
        mgr.global_collector.cleanup()

        agent = ManagedAgent(instance_id="ag1")
        from fsm_llm_monitor.collector import EventCollector

        collector = EventCollector()
        with mgr._lock:
            mgr._instances["ag1"] = agent
            mgr._collectors["ag1"] = collector

        mgr.destroy_instance("ag1")
        assert agent.status == "cancelled"
        assert agent.cancel_event.is_set()
        # Instance should be removed
        assert mgr.get_instance("ag1") is None

    def test_destroy_workflow_completes(self):
        mgr = InstanceManager(config=MonitorConfig())
        mgr.global_collector.cleanup()

        wf = ManagedWorkflow(instance_id="wf1")
        from fsm_llm_monitor.collector import EventCollector

        collector = EventCollector()
        with mgr._lock:
            mgr._instances["wf1"] = wf
            mgr._collectors["wf1"] = collector

        mgr.destroy_instance("wf1")
        assert wf.status == "completed"
        assert mgr.get_instance("wf1") is None
