from __future__ import annotations

"""Tests for fsm_llm.monitor.instance_manager."""

import threading
from typing import ClassVar
from unittest import mock
from unittest.mock import MagicMock

from fsm_llm.monitor.constants import EVENT_INSTANCE_LAUNCHED
from fsm_llm.monitor.definitions import MonitorConfig
from fsm_llm.monitor.instance_manager import (
    InstanceManager,
    ManagedAgent,
    ManagedFSM,
    ManagedWorkflow,
    register_monitor_handlers,
    snapshot_from_api,
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

        import fsm_llm.monitor.instance_manager as im

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

        import fsm_llm.monitor.instance_manager as im

        old = im._HAS_WORKFLOWS
        try:
            im._HAS_WORKFLOWS = False
            with pytest.raises(RuntimeError, match="not installed"):
                mgr.launch_workflow()
        finally:
            im._HAS_WORKFLOWS = old


class TestRegisterMonitorHandlers:
    def test_registers_7_handlers(self):
        from fsm_llm.monitor.collector import EventCollector

        api = MagicMock()
        collector = EventCollector()
        register_monitor_handlers(api, collector)
        # 7 of the 8 timings: nothing observes POST_TRANSITION.
        assert api.register_handler.call_count == 7


def _api_with_conversations(collected_data=None):
    """A mock API with two active conversations sitting in state ``greeting``."""
    api = MagicMock()
    api.list_active_conversations.return_value = ["c1", "c2"]
    api.get_stack_depth.return_value = 1
    api.fsm_manager.get_complete_conversation.return_value = {
        "current_state": {
            "id": "greeting",
            "description": "Greet user",
            "is_terminal": False,
        },
        "collected_data": collected_data or {"name": "Alice"},
        "conversation_history": [{"user": "Hi"}, {"system": "Hello!"}],
        "last_extraction_response": None,
        "last_transition_decision": None,
        "last_response_generation": None,
    }
    return api


class TestAttachApi:
    """`InstanceManager.attach_api`: show an API the caller created."""

    def _manager(self, **config):
        mgr = InstanceManager(config=MonitorConfig(**config))
        mgr.global_collector.cleanup()
        return mgr

    def test_nothing_attached_shows_nothing(self):
        mgr = self._manager()
        assert mgr.get_active_conversations() == []
        assert mgr.get_conversation_snapshot("c1") is None
        assert mgr.get_all_conversation_snapshots() == []

    def test_attach_registers_the_monitor_handlers(self):
        api = _api_with_conversations()
        self._manager().attach_api(api)
        assert api.register_handler.call_count == 7

    def test_attached_api_conversations_are_listed(self):
        mgr = self._manager()
        mgr.attach_api(_api_with_conversations())
        assert mgr.get_active_conversations() == ["c1", "c2"]

    def test_attached_api_conversation_snapshot(self):
        mgr = self._manager()
        mgr.attach_api(_api_with_conversations())
        snap = mgr.get_conversation_snapshot("c1")
        assert snap is not None
        assert snap.conversation_id == "c1"
        assert snap.current_state == "greeting"
        assert snap.context_data == {"name": "Alice"}
        assert snap.message_history == [
            {"role": "user", "content": "Hi"},
            {"role": "system", "content": "Hello!"},
        ]

    def test_all_snapshots_include_the_attached_api(self):
        mgr = self._manager()
        mgr.attach_api(_api_with_conversations())
        snaps = mgr.get_all_conversation_snapshots()
        assert [s.conversation_id for s in snaps] == ["c1", "c2"]

    def test_listing_failure_of_the_attached_api_is_not_raised(self):
        api = _api_with_conversations()
        api.list_active_conversations.side_effect = RuntimeError("boom")
        mgr = self._manager()
        mgr.attach_api(api)
        assert mgr.get_active_conversations() == []

    def test_snapshot_failure_of_the_attached_api_is_none(self):
        api = _api_with_conversations()
        api.fsm_manager.get_complete_conversation.side_effect = RuntimeError("fail")
        mgr = self._manager()
        mgr.attach_api(api)
        assert mgr.get_conversation_snapshot("c1") is None

    def test_internal_keys_hidden_unless_configured(self):
        data = {"name": "Alice", "_internal_secret": "shh", "_internal_note": "n"}
        hidden = self._manager(show_internal_keys=False)
        hidden.attach_api(_api_with_conversations(data))
        assert hidden.get_conversation_snapshot("c1").context_data == {"name": "Alice"}

        shown = self._manager(show_internal_keys=True)
        shown.attach_api(_api_with_conversations(data))
        context = shown.get_conversation_snapshot("c1").context_data
        assert "_internal_note" in context
        # Secret-looking entries stay hidden even when internal keys are shown.
        assert "_internal_secret" not in context

    def test_failed_registration_raises_and_keeps_the_previous_api(self):
        import pytest

        from fsm_llm.monitor.exceptions import MonitorConnectionError

        mgr = self._manager()
        mgr.attach_api(_api_with_conversations())
        broken = _api_with_conversations()
        broken.list_active_conversations.return_value = ["other"]
        broken.register_handler.side_effect = RuntimeError("no handlers here")

        with pytest.raises(MonitorConnectionError, match="no handlers here"):
            mgr.attach_api(broken)
        assert mgr.get_active_conversations() == ["c1", "c2"]


class TestSnapshotInternalKeyHiding:
    """`show_internal_keys=False` must actually hide the whole internal class.

    Pins step 3 of plan-2026-07-20T040150-876e7164 (F-13). This was the
    CONFIRMED leak: the dashboard's "hide internal keys" knob used a
    case-sensitive `k.startswith("_")` check, so `system_`/`internal_`/`__`
    keys were rendered in `ConversationSnapshot.context_data` regardless.
    """

    @staticmethod
    def _api_with_context(context_data):
        api = MagicMock()
        api.fsm_manager.get_complete_conversation.return_value = {
            "current_state": {"id": "collect", "description": "Collecting"},
            "collected_data": context_data,
            "conversation_history": [],
            "last_extraction_response": None,
            "last_transition_decision": None,
            "last_response_generation": None,
        }
        api.get_stack_depth.return_value = 0
        return api

    _MIXED_CONTEXT: ClassVar[dict[str, str]] = {
        "order_id": "A-1",
        "_private": "no",
        "system_password": "hunter2",
        "internal_token": "tok",
        "__dunder": "no",
        "System_Password": "hunter2",
        "Internal_token": "tok",
        "SYSTEM_secret": "shh",
    }

    def test_hidden_when_show_internal_keys_false(self):
        snap = snapshot_from_api(
            self._api_with_context(dict(self._MIXED_CONTEXT)),
            "conv-1",
            show_internal_keys=False,
        )

        assert snap is not None
        assert snap.context_data == {"order_id": "A-1"}
        for leaked in (
            "_private",
            "system_password",
            "internal_token",
            "__dunder",
            "System_Password",
            "Internal_token",
            "SYSTEM_secret",
        ):
            assert leaked not in snap.context_data, (
                f"internal key rendered despite show_internal_keys=False: {leaked}"
            )

    def test_shown_when_show_internal_keys_true(self):
        """The other direction: the knob is a knob, not an unconditional strip."""
        snap = snapshot_from_api(
            self._api_with_context(dict(self._MIXED_CONTEXT)),
            "conv-1",
            show_internal_keys=True,
        )

        assert snap is not None
        # Internal keys are shown, but secret-looking entries never are.
        assert snap.context_data == {
            "order_id": "A-1",
            "_private": "no",
            "internal_token": "tok",
            "__dunder": "no",
            "Internal_token": "tok",
        }
        for secret in ("system_password", "System_Password", "SYSTEM_secret"):
            assert secret not in snap.context_data


class TestConversationCaching:
    """Test ended conversation cache behavior."""

    def test_cache_bounds(self):
        mgr = InstanceManager(config=MonitorConfig())
        mgr.global_collector.cleanup()
        mgr._max_ended_conversations = 3

        # Manually add entries to the cache
        from fsm_llm.monitor.definitions import ConversationSnapshot

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

        from fsm_llm.monitor.definitions import ConversationSnapshot

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

        from fsm_llm.monitor.definitions import ConversationSnapshot

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
        from fsm_llm.monitor.collector import EventCollector

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
        from fsm_llm.monitor.collector import EventCollector

        collector = EventCollector()
        with mgr._lock:
            mgr._instances["wf1"] = wf
            mgr._collectors["wf1"] = collector

        mgr.destroy_instance("wf1")
        assert wf.status == "completed"
        assert mgr.get_instance("wf1") is None

    def test_destroy_agent_joins_bounded_and_warns_if_still_alive(self):
        """A never-returning agent thread must not hang destroy_instance."""
        import time

        mgr = InstanceManager(config=MonitorConfig())
        mgr.global_collector.cleanup()

        agent = ManagedAgent(instance_id="ag-hang")
        never_returns = threading.Event()

        def _hang() -> None:
            # Never checks cancel_event; simulates agent.run() with no
            # cancellation hook, blocked on e.g. an in-flight LLM call.
            never_returns.wait()

        thread = threading.Thread(target=_hang, daemon=True)
        agent.thread = thread
        thread.start()

        from fsm_llm.monitor.collector import EventCollector

        collector = EventCollector()
        with mgr._lock:
            mgr._instances["ag-hang"] = agent
            mgr._collectors["ag-hang"] = collector

        with mock.patch("fsm_llm.monitor.instance_manager.logger") as mock_logger:
            start = time.monotonic()
            mgr.destroy_instance("ag-hang")
            elapsed = time.monotonic() - start

        # Bounded: destroy_instance returns well within a few seconds, not
        # indefinitely (the thread never returns).
        assert elapsed < 5.0
        assert agent.status == "cancelled"
        assert agent.cancel_event.is_set()
        assert thread.is_alive()
        mock_logger.warning.assert_called_once()
        assert "ag-hang" in mock_logger.warning.call_args[0][0]

        # Cleanup: release the background thread so it doesn't leak past the
        # test.
        never_returns.set()
        thread.join(timeout=2.0)

    def test_destroy_agent_joins_quickly_when_thread_finishes(self):
        """A thread that finishes promptly after cancellation is joined
        without triggering the still-alive warning."""
        mgr = InstanceManager(config=MonitorConfig())
        mgr.global_collector.cleanup()

        agent = ManagedAgent(instance_id="ag-quick")

        def _quick() -> None:
            agent.cancel_event.wait(timeout=2.0)

        thread = threading.Thread(target=_quick, daemon=True)
        agent.thread = thread
        thread.start()

        from fsm_llm.monitor.collector import EventCollector

        collector = EventCollector()
        with mgr._lock:
            mgr._instances["ag-quick"] = agent
            mgr._collectors["ag-quick"] = collector

        with mock.patch("fsm_llm.monitor.instance_manager.logger") as mock_logger:
            mgr.destroy_instance("ag-quick")

        thread.join(timeout=2.0)
        assert not thread.is_alive()
        mock_logger.warning.assert_not_called()


class TestActivitySnapshots:
    """Tests for the unified activity snapshot aggregation."""

    def test_empty_activity(self):
        mgr = InstanceManager()
        mgr.global_collector.cleanup()
        items = mgr.get_all_activity_snapshots()
        assert items == []

    def test_agent_appears_in_activity(self):
        mgr = InstanceManager()
        mgr.global_collector.cleanup()

        agent = ManagedAgent(
            instance_id="agent-1",
            agent_type="ReactAgent",
            task="test task",
            label="Test Agent",
        )
        agent.max_iterations = 10

        from fsm_llm.monitor.collector import EventCollector

        collector = EventCollector()
        with mgr._lock:
            mgr._instances["agent-1"] = agent
            mgr._collectors["agent-1"] = collector

        items = mgr.get_all_activity_snapshots()
        agent_items = [i for i in items if i.item_type == "agent_task"]
        assert len(agent_items) == 1
        assert agent_items[0].item_id == "agent-1"
        assert agent_items[0].detail == "ReactAgent"
        assert agent_items[0].label == "Test Agent"

    def test_metrics_include_active_agents(self):
        mgr = InstanceManager()
        mgr.global_collector.cleanup()

        agent = ManagedAgent(instance_id="a1", agent_type="ReactAgent", task="t")
        agent.status = "running"

        from fsm_llm.monitor.collector import EventCollector

        collector = EventCollector()
        with mgr._lock:
            mgr._instances["a1"] = agent
            mgr._collectors["a1"] = collector

        metrics = mgr.get_metrics()
        assert metrics.active_agents == 1
        assert metrics.active_workflows == 0

    def test_metrics_include_active_workflows(self):
        mgr = InstanceManager()
        mgr.global_collector.cleanup()

        wf = ManagedWorkflow(instance_id="wf1")
        wf.status = "running"

        from fsm_llm.monitor.collector import EventCollector

        collector = EventCollector()
        with mgr._lock:
            mgr._instances["wf1"] = wf
            mgr._collectors["wf1"] = collector

        metrics = mgr.get_metrics()
        assert metrics.active_agents == 0
        assert metrics.active_workflows == 1


class TestFindExamplesDir:
    def test_examples_dir_resolves_to_repo_examples(self):
        # instance_manager.py sits one level deeper since the move under
        # src/fsm_llm/monitor/; a wrong depth silently returns None.
        from pathlib import Path

        from fsm_llm.monitor.instance_manager import _find_examples_dir

        repo = Path(__file__).resolve().parents[2]
        assert _find_examples_dir() == repo / "examples"


class TestWorkflowPresets:
    """Workflow presets wiring (plan_2026-05-29_0c00a594 Step 1)."""

    def _make_manager(self) -> InstanceManager:
        mgr = InstanceManager(config=MonitorConfig())
        mgr.global_collector.cleanup()
        return mgr

    def test_presets_listed(self):
        mgr = self._make_manager()
        presets = mgr.get_workflow_presets()
        ids = {p["id"] for p in presets}
        assert {"demo_linear", "demo_branching"} <= ids
        for p in presets:
            assert p["name"] and "description" in p

    async def test_launch_registers_and_completes_on_start(self):
        mgr = self._make_manager()
        managed = mgr.launch_workflow(preset_id="demo_linear")
        assert managed.workflow_id == "demo_linear"

        wf_id = await mgr.start_workflow_instance(
            managed.instance_id, managed.workflow_id, {}
        )
        status = mgr.get_workflow_status(managed.instance_id, wf_id)
        assert status["status"] == "completed"
        assert status["history"]  # steps were recorded
        # auto-completed instance is not left dangling in the active list
        assert wf_id not in managed.active_instance_ids

    async def test_branching_routes_on_context(self):
        mgr = self._make_manager()
        managed = mgr.launch_workflow(preset_id="demo_branching")
        wf_id = await mgr.start_workflow_instance(
            managed.instance_id, managed.workflow_id, {"amount": 5000}
        )
        status = mgr.get_workflow_status(managed.instance_id, wf_id)
        assert status["status"] == "completed"
        assert status["context"].get("tier") == "high"

    async def test_advance_completed_instance_rejected(self):
        import pytest

        mgr = self._make_manager()
        managed = mgr.launch_workflow(preset_id="demo_linear")
        wf_id = await mgr.start_workflow_instance(
            managed.instance_id, managed.workflow_id, {}
        )
        # auto-completed instance was removed from the active list...
        assert managed.active_instance_ids == []
        # ...so advancing it is rejected as no-longer-active (status stays queryable).
        with pytest.raises(KeyError):
            await mgr.advance_workflow(managed.instance_id, wf_id)
        assert (
            mgr.get_workflow_status(managed.instance_id, wf_id)["status"] == "completed"
        )

    def test_definition_json_rejected(self):
        import pytest

        mgr = self._make_manager()
        with pytest.raises(ValueError, match="not supported"):
            mgr.launch_workflow(definition_json={"foo": "bar"})

    def test_unknown_preset_rejected(self):
        import pytest

        mgr = self._make_manager()
        with pytest.raises(ValueError, match="Unknown workflow preset"):
            mgr.launch_workflow(preset_id="does_not_exist")

    def test_missing_preset_id_rejected(self):
        import pytest

        mgr = self._make_manager()
        with pytest.raises(ValueError, match="preset_id is required"):
            mgr.launch_workflow()

    async def test_status_redacts_secret_shaped_context(self):
        mgr = self._make_manager()
        managed = mgr.launch_workflow(preset_id="demo_branching")
        wf_id = await mgr.start_workflow_instance(
            managed.instance_id,
            managed.workflow_id,
            {
                "amount": 5000,
                "password": "hunter2",
                "nested": {"password": "x", "ok": 1},
            },
        )
        context = mgr.get_workflow_status(managed.instance_id, wf_id)["context"]
        assert context.get("tier") == "high"
        assert "password" not in context
        assert context["nested"] == {"ok": 1}

    async def test_send_workflow_event(self):
        import pytest

        mgr = self._make_manager()
        managed = mgr.launch_workflow(preset_id="demo_linear")
        # Nothing waits in the demo presets: a broadcast wakes nobody.
        assert (
            await mgr.send_workflow_event(managed.instance_id, "ping", {"a": 1}) == []
        )
        # A target that is not a known workflow instance is rejected.
        with pytest.raises(KeyError):
            await mgr.send_workflow_event(
                managed.instance_id, "ping", {}, wf_instance_id="nope"
            )


class TestDisabledAgentTypes:
    """EvaluatorOptimizer/MakerChecker are not launchable (plan Step 3, D-001)."""

    def _make_manager(self) -> InstanceManager:
        mgr = InstanceManager(config=MonitorConfig())
        mgr.global_collector.cleanup()
        return mgr

    def test_evaluator_optimizer_not_launchable(self):
        import pytest

        mgr = self._make_manager()
        with pytest.raises(ValueError, match="Unknown agent type"):
            mgr.launch_agent(agent_type="EvaluatorOptimizerAgent", task="x")

    def test_maker_checker_not_launchable(self):
        import pytest

        mgr = self._make_manager()
        with pytest.raises(ValueError, match="Unknown agent type"):
            mgr.launch_agent(agent_type="MakerCheckerAgent", task="x")

    def test_supported_agent_types(self):
        from fsm_llm.monitor.instance_manager import _AGENT_CLASSES

        assert "EvaluatorOptimizerAgent" not in _AGENT_CLASSES
        assert "MakerCheckerAgent" not in _AGENT_CLASSES
        # core types still present
        assert "ReactAgent" in _AGENT_CLASSES
        assert "DebateAgent" in _AGENT_CLASSES


class TestAgentConcurrencyHardening:
    """Snapshot-under-lock + dead-thread resolution (plan Step 4, D-003)."""

    def _make_manager(self) -> InstanceManager:
        mgr = InstanceManager(config=MonitorConfig())
        mgr.global_collector.cleanup()
        return mgr

    def _inject_agent(self, mgr, inst):
        from fsm_llm.monitor.collector import EventCollector

        with mgr._lock:
            mgr._instances[inst.instance_id] = inst
            mgr._collectors[inst.instance_id] = EventCollector()

    def test_dead_cancelling_thread_resolves_to_cancelled(self):
        mgr = self._make_manager()
        inst = ManagedAgent(instance_id="a1")
        inst.status = "cancelling"
        inst.cancel_event.set()
        inst.thread = MagicMock()
        inst.thread.is_alive.return_value = False
        self._inject_agent(mgr, inst)

        status = mgr.get_agent_status("a1")
        assert status["status"] == "cancelled"

    def test_dead_running_thread_without_result_is_failed(self):
        mgr = self._make_manager()
        inst = ManagedAgent(instance_id="a2")
        inst.status = "running"
        inst.thread = MagicMock()
        inst.thread.is_alive.return_value = False
        self._inject_agent(mgr, inst)

        status = mgr.get_agent_status("a2")
        assert status["status"] == "failed"

    def test_get_agent_result_distinguishes_failed_from_running(self):
        mgr = self._make_manager()
        failed = ManagedAgent(instance_id="f1")
        failed.status = "failed"
        failed.error = "boom"
        self._inject_agent(mgr, failed)
        running = ManagedAgent(instance_id="r1")
        running.status = "running"
        self._inject_agent(mgr, running)

        fr = mgr.get_agent_result("f1")
        rr = mgr.get_agent_result("r1")
        assert fr["status"] == "failed"
        assert fr["error"] == "boom"
        assert rr["status"] == "running"
        assert rr["error"] == "Agent has not completed yet"

    def test_dashboard_config_version_increments_under_lock(self):
        from fsm_llm.monitor.definitions import DashboardConfig

        mgr = self._make_manager()
        v0 = mgr.dashboard_config_version
        mgr.dashboard_config = DashboardConfig()
        assert mgr.dashboard_config_version == v0 + 1


class TestStubToolExecution:
    """DECISION plan-2026-09-24T091842-c1d5bfbc/D-024 (review W5): the stub
    tools a launched agent gets (``stub_fn(**kwargs)``, no schema) must run."""

    def test_launched_agent_stub_tool_returns_its_response(self, monkeypatch):
        import pytest

        pytest.importorskip("fsm_llm.agents")
        from fsm_llm.agents.definitions import ToolCall
        from fsm_llm.monitor import instance_manager as im
        from fsm_llm.monitor.definitions import StubToolConfig

        captured: dict = {}
        done = threading.Event()

        class _CapturingAgent:
            def __init__(self, **kwargs):
                captured["tools"] = kwargs["tools"]

            def run(self, task):
                done.set()
                raise RuntimeError("stop after capture")

        monkeypatch.setitem(im._AGENT_CLASSES, "ReactAgent", _CapturingAgent)
        mgr = InstanceManager(config=MonitorConfig())
        mgr.global_collector.cleanup()
        mgr.launch_agent(
            agent_type="ReactAgent",
            task="x",
            tools_config=[
                StubToolConfig(name="lookup", description="d", stub_response="STUB")
            ],
        )
        assert done.wait(5), "agent thread never ran"

        registry = captured["tools"]
        for params in ({}, {"query": "x"}, {"input": "x"}):
            result = registry.execute(ToolCall(tool_name="lookup", parameters=params))
            assert result.success, (params, result.error)
            assert result.result == "STUB"


class TestLaunchConstructsEveryAgentClass:
    """plan-2026-09-29T103145-06a5ec0a D-003: agent constructors reject a
    ``tools=`` they cannot use. The monitor passes ``tools`` only to tool-based
    classes, so every launchable class must still construct with the monitor's
    ``config``/``handlers``(/``tools``) kwargs, even when the request carries a
    tools config for a tool-less class."""

    def test_every_class_constructs_through_launch(self, monkeypatch):
        import pytest

        from fsm_llm.monitor import instance_manager as im
        from fsm_llm.monitor.definitions import StubToolConfig

        if not im._AGENT_CLASSES:
            pytest.skip("fsm_llm.agents not importable")
        assert len(im._AGENT_CLASSES) == 7

        constructed: dict[str, threading.Event] = {}
        for name, real_cls in list(im._AGENT_CLASSES.items()):
            event = threading.Event()
            constructed[name] = event

            def _run(self, task, _event=event):
                _event.set()
                raise RuntimeError("stop after construction")

            monkeypatch.setitem(
                im._AGENT_CLASSES, name, type(name, (real_cls,), {"run": _run})
            )

        mgr = InstanceManager(config=MonitorConfig())
        mgr.global_collector.cleanup()
        launched = {}
        for name in constructed:
            launched[name] = mgr.launch_agent(
                agent_type=name,
                task="x",
                tools_config=[
                    StubToolConfig(name="lookup", description="d", stub_response="S")
                ],
            )
        for name, event in constructed.items():
            assert event.wait(5), f"{name} did not construct: {launched[name].error}"


# should_terminate is grounded only by the tool observation, so the scripted
# run is: think (pick the tool), act (run it), think (conclude), conclude.
_REACT_FACTS: dict = {
    "tool_name": ("lookup", "capital"),
    "tool_input": ({"query": "capital of France"}, "capital"),
    "should_terminate": (True, "is Paris"),
}
_REACT_TASK = "What is the capital of France?"
_REACT_ANSWER = "The capital of France is Paris."

_CHAT_FSM: dict = {
    "name": "NameCapture",
    "description": "Ask for a name, then greet",
    "initial_state": "ask",
    "persona": "A friendly assistant",
    "states": {
        "ask": {
            "id": "ask",
            "description": "Ask for the name",
            "purpose": "Learn the user's name",
            "response_instructions": "Ask for the user's name",
            "required_context_keys": ["name"],
            "transitions": [
                {
                    "target_state": "greet",
                    "description": "The name is known",
                    "conditions": [
                        {
                            "description": "name was given",
                            "requires_context_keys": ["name"],
                            "logic": {"!!": [{"var": "name"}]},
                        }
                    ],
                }
            ],
        },
        "greet": {
            "id": "greet",
            "description": "Greet by name",
            "purpose": "Greet the user",
            "response_instructions": "Greet the user by name",
        },
    },
}


def _event_path(collector) -> list[str]:
    """Event kinds oldest first; a transition as ``kind:source>target``."""
    return [
        f"{e.event_type}:{e.source_state}>{e.target_state}"
        if e.event_type == "state_transition"
        else e.event_type
        for e in reversed(collector.get_events(limit=0))
    ]


class TestAgentRunEventsOnTheCoreLoop:
    """plan-2026-09-30T062855-07ad3f8c step 14: a launched agent runs on
    core's bounded run (no user turns); the priority-9999 observers injected
    through ``handlers=`` still see one event per timing per step. The pinned
    sequence is the one the same scripted run produced at d4b1626, where the
    agent sent a "Continue." turn per step. The FSM chat path stays on
    ``converse``."""

    def _launch(self, monkeypatch):
        import pytest

        pytest.importorskip("fsm_llm.agents")
        from fsm_llm.monitor import instance_manager as im
        from fsm_llm.monitor.definitions import StubToolConfig
        from tests.conftest import PromptGroundedLLM

        llm = PromptGroundedLLM(facts=_REACT_FACTS, default_response=_REACT_ANSWER)
        real_cls = im._AGENT_CLASSES["ReactAgent"]

        def _init(self, **kwargs):
            real_cls.__init__(self, llm_interface=llm, **kwargs)

        monkeypatch.setitem(
            im._AGENT_CLASSES,
            "ReactAgent",
            type("ReactAgent", (real_cls,), {"__init__": _init}),
        )
        mgr = InstanceManager(config=MonitorConfig())
        mgr.global_collector.cleanup()
        managed = mgr.launch_agent(
            agent_type="ReactAgent",
            task=_REACT_TASK,
            tools_config=[
                StubToolConfig(
                    name="lookup",
                    description="Look up a fact",
                    stub_response=_REACT_ANSWER,
                )
            ],
        )
        managed.thread.join(10)
        assert not managed.thread.is_alive(), "agent thread did not finish"
        return mgr, managed, llm

    def test_observers_see_each_step_of_a_scripted_react_run(self, monkeypatch):
        mgr, managed, llm = self._launch(monkeypatch)

        assert (managed.status, managed.error) == ("completed", None)
        assert managed.result.answer == _REACT_ANSWER
        path = [
            "conversation_start",
            "pre_processing",
            "context_update",
            "state_transition:think>act",
            "post_processing",
            "pre_processing",
            "state_transition:act>think",
            "post_processing",
            "pre_processing",
            "context_update",
            "state_transition:think>conclude",
            "post_processing",
            "conversation_end",
        ]
        assert _event_path(mgr._collectors[managed.instance_id]) == path
        # The global collector has the same handler events between its own
        # launch and result events.
        assert _event_path(mgr.global_collector) == [
            "instance_launched",
            "agent_started",
            *path,
            "agent_iteration",
            "agent_iteration",
            "agent_tool_call",
            "agent_completed",
        ]
        # No step carried a user message.
        assert {request.user_message for _, request in llm.requests} == {None}

    def test_status_and_conversation_log_need_no_turn_text(self, monkeypatch):
        import json

        mgr, managed, _ = self._launch(monkeypatch)

        status = mgr.get_agent_status(managed.instance_id)
        assert status["answer"] == _REACT_ANSWER
        assert status["success"] is True
        assert [tool["tool_name"] for tool in status["tools_used"]] == ["lookup"]
        log = status["conversation_log"]
        assert [entry["type"] for entry in log] == [
            "start",
            "context",
            "transition",
            "context",
            "transition",
            "context",
            "context",
            "transition",
            "context",
            "end",
        ]
        rendered = json.dumps(log, default=str)
        assert "Continue." not in rendered
        assert not any(f"[{state}]" in rendered for state in ("think", "act"))
        assert _REACT_ANSWER in rendered

    def test_fsm_chat_path_sends_the_user_message_through_converse(self, monkeypatch):
        from fsm_llm import API
        from fsm_llm.monitor import instance_manager as im
        from tests.conftest import MockLLM2Interface

        llm = MockLLM2Interface(
            extraction_data={"name": "Alice"}, response_text="Hello Alice!"
        )
        real_from_definition = API.from_definition

        def _from_definition(definition, **kwargs):
            return real_from_definition(definition, llm_interface=llm)

        monkeypatch.setattr(im.API, "from_definition", _from_definition)
        mgr = InstanceManager(config=MonitorConfig())
        mgr.global_collector.cleanup()
        inst = mgr.launch_fsm(fsm_json=_CHAT_FSM)
        conv_id, greeting = mgr.start_conversation(inst.instance_id)
        assert greeting == "Hello Alice!"

        reply = mgr.send_message(inst.instance_id, conv_id, "My name is Alice")

        assert reply["response"] == "Hello Alice!"
        assert reply["current_state"] == "greet"
        assert "My name is Alice" in {
            request.user_message for _, request in llm.call_history
        }
        assert _event_path(mgr._collectors[inst.instance_id]) == [
            "conversation_start",
            "pre_processing",
            "context_update",
            "state_transition:ask>greet",
            "post_processing",
            "conversation_end",
        ]
        assert inst.status == "completed"
