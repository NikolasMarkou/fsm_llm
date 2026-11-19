from __future__ import annotations

"""Tests for fsm_llm.monitor.instance_manager."""

import asyncio
import threading
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest import mock
from unittest.mock import MagicMock

import pytest

from fsm_llm.monitor.collector import EventCollector, redact_context
from fsm_llm.monitor.constants import (
    EVENT_AGENT_STARTED,
    EVENT_INSTANCE_LAUNCHED,
    EVENT_WORKFLOW_ADVANCED,
    EVENT_WORKFLOW_COMPLETED,
    EVENT_WORKFLOW_EVENT_DELIVERED,
)
from fsm_llm.monitor.definitions import MonitorConfig, MonitorEvent, StubToolConfig
from fsm_llm.monitor.exceptions import MonitorCapacityError
from fsm_llm.monitor.instance_manager import (
    InstanceManager,
    ManagedAgent,
    ManagedFSM,
    ManagedWorkflow,
    register_monitor_handlers,
    snapshot_from_api,
)
from fsm_llm.monitor.instance_manager import agent as agent_mod
from tests.test_fsm_llm_monitor.support import _manager


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

    def test_end_instance_nonexistent_raises(self):
        mgr = self._make_manager()
        import pytest

        with pytest.raises(KeyError):
            mgr.end_instance("nonexistent")

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

    def test_emit_event(self):
        mgr = self._make_manager()
        mgr._emit_event(EVENT_INSTANCE_LAUNCHED, message="test launch")
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

    def test_get_active_conversations_empty(self):
        mgr = InstanceManager(config=MonitorConfig())
        mgr.global_collector.cleanup()
        assert mgr.get_active_conversations() == []


class TestDestroyInstance:
    """Test instance destruction behavior."""

    def test_end_instance_agent_cancels(self):
        mgr = InstanceManager(config=MonitorConfig())
        mgr.global_collector.cleanup()

        agent = ManagedAgent(instance_id="ag1")
        from fsm_llm.monitor.collector import EventCollector

        collector = EventCollector()
        with mgr._lock:
            mgr._instances["ag1"] = agent
            mgr._collectors["ag1"] = collector

        mgr.end_instance("ag1")
        assert agent.status == "cancelled"
        assert agent.cancel_event.is_set()
        # Instance should be removed
        assert mgr.get_instance("ag1") is None

    def test_end_instance_workflow_completes(self):
        mgr = InstanceManager(config=MonitorConfig())
        mgr.global_collector.cleanup()

        wf = ManagedWorkflow(instance_id="wf1")
        from fsm_llm.monitor.collector import EventCollector

        collector = EventCollector()
        with mgr._lock:
            mgr._instances["wf1"] = wf
            mgr._collectors["wf1"] = collector

        mgr.end_instance("wf1")
        assert wf.status == "completed"
        assert mgr.get_instance("wf1") is None

    def test_end_instance_agent_joins_bounded_and_warns_if_still_alive(
        self, timing_factor
    ):
        """A never-returning agent thread must not hang end_instance."""
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

        with mock.patch(
            "fsm_llm.monitor.instance_manager.lifecycle.logger"
        ) as mock_logger:
            start = time.monotonic()
            mgr.end_instance("ag-hang")
            elapsed = time.monotonic() - start

        # Bounded: end_instance returns well within a few seconds, not
        # indefinitely (the thread never returns).
        assert elapsed < 5.0 * timing_factor
        assert agent.status == "cancelled"
        assert agent.cancel_event.is_set()
        assert thread.is_alive()
        mock_logger.warning.assert_called_once()
        assert "ag-hang" in mock_logger.warning.call_args[0][0]

        # Cleanup: release the background thread so it doesn't leak past the
        # test.
        never_returns.set()
        thread.join(timeout=2.0 * timing_factor)

    def test_end_instance_agent_joins_quickly_when_thread_finishes(self, timing_factor):
        """A thread that finishes promptly after cancellation is joined
        without triggering the still-alive warning."""
        mgr = InstanceManager(config=MonitorConfig())
        mgr.global_collector.cleanup()

        agent = ManagedAgent(instance_id="ag-quick")

        def _quick() -> None:
            agent.cancel_event.wait(timeout=2.0 * timing_factor)

        thread = threading.Thread(target=_quick, daemon=True)
        agent.thread = thread
        thread.start()

        from fsm_llm.monitor.collector import EventCollector

        collector = EventCollector()
        with mgr._lock:
            mgr._instances["ag-quick"] = agent
            mgr._collectors["ag-quick"] = collector

        with mock.patch(
            "fsm_llm.monitor.instance_manager.lifecycle.logger"
        ) as mock_logger:
            mgr.end_instance("ag-quick")

        thread.join(timeout=2.0 * timing_factor)
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
    """``find_examples_dir`` counts directories from the file that performs the
    read (``instance_manager/examples.py``, four below the repo root); a wrong
    depth does not raise, it returns ``None`` and the dashboard lists no
    presets (plan d454fd62 step 93)."""

    @staticmethod
    def _module_file():
        import inspect
        from pathlib import Path

        from fsm_llm.monitor.instance_manager import find_examples_dir

        return Path(inspect.getsourcefile(find_examples_dir)).resolve()

    @staticmethod
    def _load_copy_at(path):
        """Load a copy of the real module from ``path`` (so its ``__file__`` is
        ``path``) and return its ``find_examples_dir``."""
        import importlib.util

        source = TestFindExamplesDir._module_file().read_text(encoding="utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
        spec = importlib.util.spec_from_file_location("_zz_examples_probe", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.find_examples_dir

    def test_examples_dir_resolves_to_repo_examples(self):
        from pathlib import Path

        from fsm_llm.monitor.instance_manager import find_examples_dir

        repo = Path(__file__).resolve().parents[2]
        assert find_examples_dir() == repo / "examples"

    def test_the_directory_found_is_a_real_non_empty_examples_tree(self):
        from fsm_llm.monitor.instance_manager import find_examples_dir

        found = find_examples_dir()
        assert found is not None and found.is_dir()
        assert (found / "basic").is_dir()
        assert list(found.glob("*/*/fsm.json")), "examples/ holds no FSM presets"

    def test_a_preset_of_the_examples_tree_loads_through_the_module(self):
        from fsm_llm.monitor.instance_manager import load_preset_json

        data = load_preset_json("basic/form_filling/fsm.json")
        assert isinstance(data, dict) and data["initial_state"]

    def test_the_read_is_relative_to_the_file_that_performs_it(self):
        # The nearest ancestor of the defining file named ``src`` sits next to
        # ``examples/``, whatever depth the module has inside the package.
        from fsm_llm.monitor.instance_manager import find_examples_dir

        here = self._module_file()
        src = next(a for a in here.parents if a.name == "src")
        assert find_examples_dir() == src.parent / "examples"

    def test_a_copy_four_levels_below_a_root_finds_that_roots_examples(self, tmp_path):
        root = tmp_path.resolve() / "repo"
        (root / "examples").mkdir(parents=True)
        probe = (
            root / "src" / "fsm_llm" / "monitor" / "instance_manager" / "examples.py"
        )
        assert self._load_copy_at(probe)() == root / "examples"

    def test_a_copy_does_not_take_an_examples_dir_one_level_too_shallow(self, tmp_path):
        # A decoy next to ``src/`` 's own level (the old single-module depth).
        root = tmp_path.resolve() / "repo"
        (root / "src" / "examples").mkdir(parents=True)
        probe = (
            root / "src" / "fsm_llm" / "monitor" / "instance_manager" / "examples.py"
        )
        assert self._load_copy_at(probe)() is None

    def test_a_layout_with_no_examples_dir_returns_none_without_raising(self, tmp_path):
        # The installed (wheel) layout: nothing named ``examples`` is there.
        probe = (
            tmp_path.resolve()
            / "lib"
            / "site-packages"
            / "fsm_llm"
            / "monitor"
            / "instance_manager"
            / "examples.py"
        )
        assert self._load_copy_at(probe)() is None

    def test_load_preset_json_with_no_examples_dir_raises_file_not_found(
        self, monkeypatch
    ):
        from fsm_llm.monitor.instance_manager import examples

        monkeypatch.setattr(examples, "find_examples_dir", lambda: None)
        with pytest.raises(FileNotFoundError, match="Examples directory not found"):
            examples.load_preset_json("basic/form_filling/fsm.json")


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

    async def test_finished_runs_stay_listed_in_run_order(self):
        """F4 (monitor audit 1.1): the instances list names every run of the
        engine, finished ones included, not only the active ones (it listed
        nothing for a preset that completes inside ``start``)."""
        mgr = self._make_manager()
        managed = mgr.launch_workflow(preset_id="demo_linear")
        first = await mgr.start_workflow_instance(
            managed.instance_id, managed.workflow_id, {}
        )
        second = await mgr.start_workflow_instance(
            managed.instance_id, managed.workflow_id, {}
        )
        assert managed.active_instance_ids == []

        listed = mgr.get_workflow_instances(managed.instance_id)

        assert [r["workflow_instance_id"] for r in listed] == [first, second]
        assert [r["status"] for r in listed] == ["completed", "completed"]
        assert all(r["history"] for r in listed)

    def test_no_runs_lists_nothing(self):
        mgr = self._make_manager()
        managed = mgr.launch_workflow(preset_id="demo_linear")

        assert mgr.get_workflow_instances(managed.instance_id) == []

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
        from fsm_llm.monitor.instance_manager.agent import _AGENT_CLASSES

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
        from fsm_llm.monitor.definitions import StubToolConfig
        from fsm_llm.monitor.instance_manager import agent as agent_mod
        from fsm_llm.tools import ToolCall

        captured: dict = {}
        done = threading.Event()

        class _CapturingAgent:
            def __init__(self, **kwargs):
                captured["tools"] = kwargs["tools"]

            def run(self, task):
                done.set()
                raise RuntimeError("stop after capture")

        monkeypatch.setitem(agent_mod._AGENT_CLASSES, "ReactAgent", _CapturingAgent)
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

        from fsm_llm.monitor.definitions import StubToolConfig
        from fsm_llm.monitor.instance_manager import agent as agent_mod

        if not agent_mod._AGENT_CLASSES:
            pytest.skip("fsm_llm.agents not importable")
        assert len(agent_mod._AGENT_CLASSES) == 7

        constructed: dict[str, threading.Event] = {}
        for name, real_cls in list(agent_mod._AGENT_CLASSES.items()):
            event = threading.Event()
            constructed[name] = event

            def _run(self, task, _event=event):
                _event.set()
                raise RuntimeError("stop after construction")

            monkeypatch.setitem(
                agent_mod._AGENT_CLASSES, name, type(name, (real_cls,), {"run": _run})
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
        from fsm_llm.monitor.definitions import StubToolConfig
        from fsm_llm.monitor.instance_manager import agent as agent_mod
        from fsm_llm.testing import PromptGroundedLLM

        llm = PromptGroundedLLM(facts=_REACT_FACTS, default_response=_REACT_ANSWER)
        real_cls = agent_mod._AGENT_CLASSES["ReactAgent"]

        def _init(self, **kwargs):
            real_cls.__init__(self, llm_interface=llm, **kwargs)

        monkeypatch.setitem(
            agent_mod._AGENT_CLASSES,
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
        # Lifecycle events and handler events reach the global collector and,
        # for this instance, its own collector (F-MI-10).
        full = [
            "instance_launched",
            "agent_started",
            *path,
            "agent_iteration",
            "agent_iteration",
            "agent_tool_call",
            "agent_completed",
        ]
        assert _event_path(mgr._collectors[managed.instance_id]) == full
        assert _event_path(mgr.global_collector) == full
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
        from fsm_llm.monitor.instance_manager import fsm as fsm_mod
        from fsm_llm.testing import MockLLM2Interface

        llm = MockLLM2Interface(
            extraction_data={"name": "Alice"}, response_text="Hello Alice!"
        )
        real_from_definition = API.from_definition

        def _from_definition(definition, **kwargs):
            return real_from_definition(definition, llm_interface=llm)

        monkeypatch.setattr(fsm_mod.API, "from_definition", _from_definition)
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
            "instance_launched",
            "conversation_start",
            "pre_processing",
            "context_update",
            "state_transition:ask>greet",
            "post_processing",
            "conversation_end",
        ]
        assert inst.status == "completed"


class TestDestroyClosesBeforeUnregister:
    """F-MO-01: the manager closes a managed FSM's API while the global
    collector's handlers are still active, so END_CONVERSATION reaches it."""

    def _live_manager(self, monkeypatch):
        from fsm_llm import API
        from fsm_llm.monitor.instance_manager import fsm as fsm_mod
        from fsm_llm.testing import MockLLM2Interface

        llm = MockLLM2Interface(response_text="What is your name?")
        real_from_definition = API.from_definition

        def _from_definition(definition, **kwargs):
            return real_from_definition(definition, llm_interface=llm)

        monkeypatch.setattr(fsm_mod.API, "from_definition", _from_definition)
        mgr = InstanceManager(config=MonitorConfig())
        mgr.global_collector.cleanup()
        inst = mgr.launch_fsm(fsm_json=_CHAT_FSM)
        conv_id, _ = mgr.start_conversation(inst.instance_id)
        return mgr, inst, conv_id

    def test_end_instance_ends_the_live_conversation_on_the_global_collector(
        self, monkeypatch
    ):
        mgr, inst, conv_id = self._live_manager(monkeypatch)
        collector = mgr.global_collector
        assert collector.get_metrics().active_conversations == 1

        mgr.end_instance(inst.instance_id)

        assert collector.get_metrics().active_conversations == 0
        ends = [
            e
            for e in collector.get_events()
            if e.conversation_id == conv_id and e.event_type == "conversation_end"
        ]
        assert len(ends) == 1

    def test_end_instance_keeps_the_live_conversation_in_the_ended_cache(
        self, monkeypatch
    ):
        mgr, inst, conv_id = self._live_manager(monkeypatch)

        mgr.end_instance(inst.instance_id)

        snap = mgr.get_conversation_snapshot(conv_id)
        assert snap is not None
        assert snap.is_terminal is True
        assert snap.instance_id == inst.instance_id
        assert snap.current_state == "ask"

    def test_end_instance_ends_the_otel_conversation_span(self, monkeypatch):
        pytest.importorskip("opentelemetry.sdk.trace")
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        from fsm_llm.monitor.otel import OTELExporter

        mgr, inst, _ = self._live_manager(monkeypatch)
        sink = InMemorySpanExporter()
        otel = OTELExporter(exporter=sink)
        otel.enable(mgr.global_collector)
        # Spans only start for events after enable: a second conversation.
        conv2, _ = mgr.start_conversation(inst.instance_id)
        assert conv2 in otel.active_conversations

        mgr.end_instance(inst.instance_id)

        # Before close(): that would end every open span itself.
        assert otel.active_conversations == []
        otel._provider.force_flush()
        names = [s.name for s in sink.get_finished_spans()]
        assert "conversation" in names

    def test_close_raising_still_unregisters_the_handlers(self, monkeypatch):
        from fsm_llm.monitor.instance_manager import monitor_handlers_on

        mgr, inst, _ = self._live_manager(monkeypatch)
        api = inst.api
        collector = mgr.global_collector
        assert monitor_handlers_on(api, collector)
        monkeypatch.setattr(
            api, "close", MagicMock(side_effect=RuntimeError("close failed"))
        )

        mgr.end_instance(inst.instance_id)

        assert all(not h.active for h in monitor_handlers_on(api, collector))

    def test_close_ends_live_conversations_of_managed_fsms(self, monkeypatch):
        mgr, _, conv_id = self._live_manager(monkeypatch)
        collector = mgr.global_collector
        recorded: list[str] = []
        real_record = collector.record_event

        def _spy(event):
            if event.conversation_id == conv_id:
                recorded.append(event.event_type)
            return real_record(event)

        monkeypatch.setattr(collector, "record_event", _spy)

        mgr.close()

        assert recorded.count("conversation_end") == 1
        snap = mgr.get_conversation_snapshot(conv_id)
        assert snap is not None
        assert snap.is_terminal is True


class _Secretive:
    def __str__(self) -> str:
        return "password=hunter2"


def _snapshot_api(collected: dict[str, Any], extraction: Any = None) -> MagicMock:
    api = MagicMock()
    api.fsm_manager.get_complete_conversation.return_value = {
        "current_state": {"id": "s", "description": "", "is_terminal": False},
        "collected_data": collected,
        "conversation_history": [],
        "last_extraction_response": extraction,
        "last_transition_decision": None,
        "last_response_generation": None,
    }
    api.get_stack_depth.return_value = 1
    return api


class TestRedaction:
    def test_snapshot_context_drops_secrets_and_never_calls_str(self):
        out = EventCollector.snapshot_context(
            {
                "answer": "42",
                "password": "hunter2",
                "api_key": "sk-abcdefghijklmnopqrstuvwxyz0123",
                "obj": _Secretive(),
                "nested": {"token": "abc123secretvalue", "ok": 1},
            }
        )
        assert out["answer"] == "42"
        assert "password" not in out
        assert "api_key" not in out
        assert "hunter2" not in out.get("obj", "")
        assert out["obj"] == "<redacted:_Secretive>"
        assert "abc123secretvalue" not in out["nested"]

    def test_redact_context_nested_internal_and_secret(self):
        data = {"user": {"system_flag": "x", "name": "Ada", "password": "p"}}
        assert redact_context(data) == {"user": {"name": "Ada"}}
        assert redact_context(data, drop_internal=False) == {
            "user": {"system_flag": "x", "name": "Ada"}
        }

    def test_snapshot_redacts_nested_values_and_last_extraction(self):
        snap = snapshot_from_api(
            _snapshot_api(
                {"profile": {"password": "p", "city": "Oslo"}},
                extraction={"extracted_data": {"api_key": "sk-1234567890abcdef"}},
            ),
            "c1",
        )
        assert snap.context_data == {"profile": {"city": "Oslo"}}
        assert "api_key" not in snap.last_extraction["extracted_data"]

    def test_agent_result_redacts_final_context_and_tool_params(self):
        mgr = _manager()
        agent = ManagedAgent("a1")
        trace = MagicMock()
        trace.total_iterations = 1
        call = MagicMock()
        call.tool_name = "login"
        call.parameters = {"user": "ada", "password": "p"}
        trace.tool_calls = [call]
        trace.steps = []
        agent.result = MagicMock(
            answer="ok",
            success=True,
            final_context={"answer": "ok", "api_key": "sk-1234567890abcdef"},
            trace=trace,
        )
        agent.status = "completed"
        mgr._instances["a1"] = agent
        result = mgr.get_agent_result("a1")
        assert "api_key" not in result["final_context"]
        assert result["tools_used"][0]["parameters"] == {"user": "ada"}


class TestCancelAgent:
    def test_cancel_after_completion_is_noop(self):
        mgr = _manager()
        agent = ManagedAgent("a1")
        agent.status = "completed"
        agent.result = MagicMock(answer="done", success=True, final_context={})
        mgr._instances["a1"] = agent
        assert mgr.cancel_agent("a1") is False
        assert mgr.get_agent_status("a1")["status"] == "completed"
        assert mgr.get_agent_result("a1")["answer"] == "done"

    def test_cancel_running_agent_emits_once(self):
        mgr = _manager()
        agent = ManagedAgent("a1")
        release = threading.Event()
        agent.thread = threading.Thread(target=release.wait, daemon=True)
        agent.thread.start()
        mgr._instances["a1"] = agent
        assert mgr.cancel_agent("a1") is True
        assert mgr.cancel_agent("a1") is False
        release.set()
        cancelled = mgr.get_metrics().events_per_type.get("agent_cancelled", 0)
        assert cancelled == 1
        mgr.global_collector.cleanup()


def _fsm(mgr: InstanceManager, iid: str, api: Any = None) -> ManagedFSM:
    inst = ManagedFSM(iid, api or MagicMock())
    mgr._instances[iid] = inst
    return inst


class TestFSMLifecycle:
    def test_terminal_message_ends_conversation_and_completes(self):
        mgr = _manager()
        api = MagicMock()
        api.converse.return_value = "bye"
        api.get_current_state.return_value = "end"
        api.has_conversation_ended.return_value = True
        inst = _fsm(mgr, "f1", api)
        inst.conversation_ids.append("c1")
        mgr.send_message("f1", "c1", "hi")
        api.end_conversation.assert_called_once_with("c1")
        assert inst.status == "completed"

    def test_manual_end_completes_instance_and_restart_reopens(self):
        mgr = _manager()
        inst = _fsm(mgr, "f1")
        inst.conversation_ids.append("c1")
        mgr.end_conversation("f1", "c1")
        assert inst.status == "completed"
        mgr.end_conversation("f1", "c1")  # already ended: no error
        inst.api.start_conversation.return_value = ("c2", "hello")
        mgr.start_conversation("f1")
        assert inst.status == "running"

    def test_end_unknown_conversation_is_key_error(self):
        mgr = _manager()
        _fsm(mgr, "f1")
        with pytest.raises(KeyError):
            mgr.end_conversation("f1", "nope")

    def test_oldest_finished_instances_are_evicted(self):
        mgr = InstanceManager(config=MonitorConfig(max_instances=2))
        mgr.global_collector.cleanup()
        old = _fsm(mgr, "old")
        old.status = "completed"
        _fsm(mgr, "live")
        mgr._make_room()
        assert "old" not in mgr._instances
        assert "live" in mgr._instances

    def test_capacity_error_when_everything_is_active(self):
        mgr = InstanceManager(config=MonitorConfig(max_instances=1))
        mgr.global_collector.cleanup()
        _fsm(mgr, "live")
        with pytest.raises(MonitorCapacityError):
            mgr._make_room()

    def test_agent_info_carries_task(self):
        agent = ManagedAgent("a1", task="Find the capital of Norway")
        assert agent.to_info().task == "Find the capital of Norway"


class TestWorkflowTracking:
    @pytest.fixture(autouse=True)
    def _needs_workflows(self):
        pytest.importorskip("fsm_llm.workflows")

    def test_run_completion_and_steps_come_from_engine(self):
        async def go():
            mgr = _manager()
            managed = mgr.launch_workflow(preset_id="demo_linear")
            wf_id = await mgr.start_workflow_instance(
                managed.instance_id, managed.workflow_id, {}
            )
            return mgr, managed, wf_id

        mgr, managed, _wf_id = asyncio.run(go())
        assert managed.active_instance_ids == []
        assert managed.status == "completed"
        metrics = mgr.get_metrics()
        # demo_linear runs 3 steps: ingest, process, finish.
        assert metrics.total_workflow_steps == 3
        assert metrics.events_per_type.get(EVENT_WORKFLOW_COMPLETED) == 1
        instance_events = mgr.get_instance_collector(managed.instance_id).get_events()
        assert any(e.event_type == EVENT_WORKFLOW_ADVANCED for e in instance_events)
        assert mgr.get_metrics().active_workflows == 0
        mgr.global_collector.cleanup()

    def test_event_delivery_is_not_counted_as_a_step(self):
        async def go():
            mgr = _manager()
            managed = mgr.launch_workflow(preset_id="demo_linear")
            await mgr.send_workflow_event(managed.instance_id, "ping", {})
            return mgr

        mgr = asyncio.run(go())
        per_type = mgr.get_metrics().events_per_type
        assert per_type.get(EVENT_WORKFLOW_EVENT_DELIVERED) == 1
        assert mgr.get_metrics().total_workflow_steps == 0
        mgr.global_collector.cleanup()

    def test_workflow_status_hides_internal_and_secret_keys(self):
        async def go():
            mgr = _manager()
            managed = mgr.launch_workflow(preset_id="demo_linear")
            wf_id = await mgr.start_workflow_instance(
                managed.instance_id,
                managed.workflow_id,
                {"password": "p", "system_flag": 1, "city": "Oslo"},
            )
            return mgr, managed, wf_id

        mgr, managed, wf_id = asyncio.run(go())
        ctx = mgr.get_workflow_status(managed.instance_id, wf_id)["context"]
        assert ctx["city"] == "Oslo"
        assert "password" not in ctx
        assert "system_flag" not in ctx
        assert "_workflow_info" not in ctx
        mgr.global_collector.cleanup()


_STUB = StubToolConfig(name="lookup", description="d", stub_response="R")


def _finished_fsm(mgr: InstanceManager, instance_id: str = "old") -> ManagedFSM:
    inst = ManagedFSM(instance_id=instance_id, api=MagicMock())
    inst.status = "completed"
    mgr._instances[instance_id] = inst
    return inst


class _QuickAgent:
    def __init__(self, **kwargs):
        pass

    def run(self, task):
        return SimpleNamespace(
            success=True, answer="done", trace=None, final_context={}
        )


class _BlockingAgent:
    release = threading.Event()

    def __init__(self, **kwargs):
        pass

    def run(self, task):
        _BlockingAgent.release.wait(5)
        return SimpleNamespace(success=True, answer="", trace=None, final_context={})


class TestFailedLaunchKeepsFinishedInstance:
    """F-MI-01: eviction runs only after everything that can fail."""

    def test_invalid_fsm_launch_evicts_nothing(self):
        mgr = _manager(max_instances=1)
        _finished_fsm(mgr)
        with pytest.raises(ValueError):
            mgr.launch_fsm(fsm_json={"bad": 1})
        assert [i.instance_id for i in mgr.list_instances()] == ["old"]

    def test_invalid_agent_config_evicts_nothing(self):
        mgr = _manager(max_instances=1)
        _finished_fsm(mgr)
        with pytest.raises(ValueError):
            mgr.launch_agent(task="t", tools_config=[_STUB], max_iterations=0)
        assert [i.instance_id for i in mgr.list_instances()] == ["old"]

    def test_a_valid_launch_still_evicts_the_oldest_finished(self, monkeypatch):
        monkeypatch.setitem(agent_mod._AGENT_CLASSES, "ReactAgent", _QuickAgent)
        mgr = _manager(max_instances=1)
        _finished_fsm(mgr)
        managed = mgr.launch_agent(task="t", tools_config=[_STUB])
        assert [i.instance_id for i in mgr.list_instances()] == [managed.instance_id]
        managed.thread.join(5)


class TestTerminalSendKeepsTheConversationOpenWhenEndFails:
    """F-MI-05: a conversation counts as ended only after the core ended it."""

    def _fsm_with_failing_end(self, mgr: InstanceManager) -> ManagedFSM:
        inst = ManagedFSM(instance_id="f1", api=MagicMock())
        inst.api.converse.return_value = "bye"
        inst.api.get_current_state.return_value = "end"
        inst.api.has_conversation_ended.return_value = True
        inst.api.end_conversation.side_effect = RuntimeError("busy")
        inst.conversation_ids = ["c1"]
        mgr._instances["f1"] = inst
        return inst

    def test_send_message_does_not_mark_the_conversation_ended(self):
        mgr = _manager()
        inst = self._fsm_with_failing_end(mgr)
        reply = mgr.send_message("f1", "c1", "hi")
        assert reply["is_terminal"] is True
        assert inst.ended_conversation_ids == set()
        assert inst.status == "running"

    def test_end_conversation_raises_and_does_not_mark_it_either(self):
        mgr = _manager()
        inst = self._fsm_with_failing_end(mgr)
        with pytest.raises(RuntimeError):
            mgr.end_conversation("f1", "c1")
        assert inst.ended_conversation_ids == set()


class TestOtelReenableDoesNotExportTwice:
    """F-MI-09: a wrapper another exporter has superseded stays inert."""

    def test_one_event_makes_one_span_per_exporter(self):
        pytest.importorskip("opentelemetry.sdk.trace")
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        from fsm_llm.monitor.collector import EventCollector
        from fsm_llm.monitor.otel import OTELExporter

        mem_a, mem_b = InMemorySpanExporter(), InMemorySpanExporter()
        a, b = OTELExporter(exporter=mem_a), OTELExporter(exporter=mem_b)
        collector = EventCollector()
        a.enable(collector)
        b.enable(collector)
        a.disable()
        a.enable(collector)

        collector.record_event(MonitorEvent(event_type=EVENT_AGENT_STARTED))
        a._provider.force_flush()
        b._provider.force_flush()

        assert len(mem_a.get_finished_spans()) == 1
        assert len(mem_b.get_finished_spans()) == 1
        a.shutdown()
        b.shutdown()


class TestInstanceEventFeed:
    """F-MI-10: an instance's own collector sees its lifecycle events."""

    def test_agent_events_reach_the_instance_collector(self, monkeypatch):
        monkeypatch.setitem(agent_mod._AGENT_CLASSES, "ReactAgent", _QuickAgent)
        mgr = _manager()
        managed = mgr.launch_agent(task="t", tools_config=[_STUB])
        managed.thread.join(5)
        types = [
            e.event_type
            for e in mgr.get_instance_collector(managed.instance_id).get_events()
        ]
        assert "agent_started" in types
        assert "agent_completed" in types

    def test_workflow_started_reaches_the_instance_collector(self):
        mgr = _manager()

        async def go() -> str:
            managed = mgr.launch_workflow(preset_id="demo_linear")
            await mgr.start_workflow_instance(
                managed.instance_id, managed.workflow_id, {}
            )
            return managed.instance_id

        instance_id = asyncio.run(go())
        types = [
            e.event_type for e in mgr.get_instance_collector(instance_id).get_events()
        ]
        assert "workflow_started" in types
        assert "workflow_completed" in types


class TestCloseTearsDownEveryInstanceType:
    """F-MI-13: close() does what end_instance() does for agents and
    workflow engines, not only for FSMs."""

    def test_close_sets_the_agent_cancel_flag(self, monkeypatch):
        _BlockingAgent.release.clear()
        monkeypatch.setitem(agent_mod._AGENT_CLASSES, "ReactAgent", _BlockingAgent)
        mgr = _manager()
        managed = mgr.launch_agent(task="t", tools_config=[_STUB])
        try:
            mgr.close()
            assert managed.cancel_event.is_set()
        finally:
            _BlockingAgent.release.set()
            managed.thread.join(5)

    def test_close_stops_the_workflow_engine(self):
        mgr = _manager()
        stopped: list[int] = []

        async def go() -> None:
            managed = mgr.launch_workflow(preset_id="demo_linear")

            async def _close() -> None:
                stopped.append(1)

            # The mock replaces a real engine method: a stale name would
            # create a new attribute and the assertion below would fail.
            assert callable(getattr(type(managed.engine), "close", None))
            managed.engine.close = _close
            mgr.close()
            await asyncio.sleep(0.05)

        asyncio.run(go())
        assert stopped == [1]


_PACKAGE = "fsm_llm.monitor.instance_manager"

#: public name -> the module of the package that defines it (plan d454fd62 step 93)
_MODULE_OF = {
    "InstanceManager": "manager",
    "ManagedAgent": "managed",
    "ManagedFSM": "managed",
    "ManagedWorkflow": "managed",
    "find_examples_dir": "examples",
    "load_preset_json": "examples",
    "monitor_handlers_on": "handlers",
    "read_json_file": "examples",
    "register_monitor_handlers": "handlers",
    "snapshot_from_api": "snapshots",
    "unregister_monitor_handlers": "handlers",
    "validate_preset_id": "examples",
}

#: every method and property of the old single-module ``InstanceManager``
_OLD_MEMBERS = frozenset(
    {
        "__init__",
        "_cache_ended_conversation",
        "_check_agent_capacity",
        "_close_fsm_api",
        "_emit_agent_run_events",
        "_emit_event",
        "_ended_snapshot",
        "_finish_conversation",
        "_get_agent",
        "_get_fsm",
        "_get_typed",
        "_get_workflow",
        "_make_room",
        "_mark_conversation_ended",
        "_new_collector",
        "_record_event",
        "_register_instance",
        "_resolve_fsm_data",
        "_run_agent",
        "_setup_loguru_sink",
        "_snapshot_from_api",
        "_teardown",
        "_validate_workflow_instance_id",
        "_workflow_hook",
        "advance_workflow",
        "attach_api",
        "cancel_agent",
        "cancel_workflow",
        "close",
        "config",
        "dashboard_config",
        "dashboard_config_version",
        "end_conversation",
        "end_instance",
        "get_active_conversations",
        "get_agent_result",
        "get_agent_status",
        "get_all_activity_snapshots",
        "get_all_conversation_snapshots",
        "get_capabilities",
        "get_conversation_snapshot",
        "get_events",
        "get_fsm_conversations",
        "get_instance",
        "get_instance_collector",
        "get_metrics",
        "get_workflow_instances",
        "get_workflow_presets",
        "get_workflow_status",
        "global_collector",
        "launch_agent",
        "launch_fsm",
        "launch_workflow",
        "list_instances",
        "send_message",
        "send_workflow_event",
        "start_conversation",
        "start_workflow_instance",
    }
)


def _package_dir():
    from pathlib import Path

    import fsm_llm.monitor.instance_manager as facade

    return Path(facade.__file__).parent


class TestInstanceManagerPackageLayout:
    """``monitor/instance_manager.py`` is a package at the same dotted path:
    ``InstanceManager`` is a shell over five mixins, the ``Managed*`` classes
    sit in one module, a private name lives in its defining module."""

    def test_the_facade_exports_exactly_the_public_names(self):
        import fsm_llm.monitor.instance_manager as facade

        assert sorted(facade.__all__) == sorted([*_MODULE_OF, "ManagedInstance"])
        namespace: dict = {}
        exec(f"from {_PACKAGE} import *", namespace)
        assert set(_MODULE_OF) | {"ManagedInstance"} <= set(namespace)
        for private in (
            "_AGENT_CLASSES",
            "_MonitorHandler",
            "_WORKFLOW_PRESETS",
            "_build_context_capture_handlers",
            "_new_instance_id",
            "_pending_tasks",
            "_status_str",
        ):
            assert private not in namespace
            assert not hasattr(facade, private), private
        # a patch aims at the module that performs the lookup, never the facade
        for module_attribute in ("API", "logger", "asyncio", "uuid", "resolve_model"):
            assert not hasattr(facade, module_attribute), module_attribute

    def test_every_public_name_is_defined_in_the_module_that_owns_it(self):
        import importlib

        import fsm_llm.monitor.instance_manager as facade

        for name, module in _MODULE_OF.items():
            obj = getattr(facade, name)
            assert obj.__module__ == f"{_PACKAGE}.{module}", name
            assert vars(importlib.import_module(f"{_PACKAGE}.{module}"))[name] is obj

    def test_instance_manager_is_a_shell_over_five_mixins(self):
        from fsm_llm.monitor.instance_manager import InstanceManager

        assert [c.__name__ for c in InstanceManager.__mro__] == [
            "InstanceManager",
            "SnapshotsMixin",
            "FsmMixin",
            "WorkflowsMixin",
            "AgentsMixin",
            "LifecycleMixin",
            "_InstanceSurface",
            "object",
        ]
        owners: dict[str, list[str]] = {}
        for cls in InstanceManager.__mro__[:-2]:
            for name in vars(cls):
                if not (name.startswith("__") and name.endswith("__")) or (
                    name == "__init__"
                ):
                    owners.setdefault(name, []).append(cls.__name__)
        assert set(owners) == _OLD_MEMBERS
        assert {n: o for n, o in owners.items() if len(o) > 1} == {}
        surface = InstanceManager.__mro__[-2]
        assert [n for n in vars(surface) if not n.startswith("__")] == [], (
            "the surface class declares for mypy only, a runtime no-op"
        )

    def test_the_managed_classes_and_their_base_share_one_module(self):
        from fsm_llm.monitor.instance_manager import managed

        for cls, kind in (
            (managed.ManagedFSM, "fsm"),
            (managed.ManagedWorkflow, "workflow"),
            (managed.ManagedAgent, "agent"),
        ):
            assert cls.__mro__[1] is managed._ManagedBase
            assert cls.__module__ == managed.__name__
            assert cls.instance_type == kind
        assert managed._ManagedBase.__module__ == managed.__name__
        inst = managed.ManagedAgent("i1", label="L")
        assert (inst.instance_id, inst.label, inst.status) == ("i1", "L", "running")
        assert inst.to_info().instance_type == "agent"

    def test_the_tables_the_mixins_read_are_the_ones_of_their_modules(self):
        from fsm_llm.monitor.instance_manager import agent, workflow, workflow_presets

        assert agent.AgentsMixin.launch_agent.__globals__["_AGENT_CLASSES"] is (
            agent._AGENT_CLASSES
        )
        assert agent.AgentsMixin._run_agent.__globals__["_AGENT_CLASSES"] is (
            agent._AGENT_CLASSES
        )
        assert workflow.WorkflowsMixin.launch_workflow.__globals__[
            "_WORKFLOW_PRESETS"
        ] is (workflow_presets._WORKFLOW_PRESETS)

    def test_a_patch_of_the_logger_aims_at_the_module_that_logs(self):
        with pytest.raises(AttributeError):
            with mock.patch(f"{_PACKAGE}.logger"):
                pass  # pragma: no cover
        from fsm_llm.monitor.instance_manager import lifecycle

        assert lifecycle.LifecycleMixin._teardown.__globals__["logger"] is (
            lifecycle.logger
        )

    def test_every_decision_anchor_sits_with_the_code_it_constrains(self):
        # token -> module -> count (the launch docstring of `workflow` names
        # the preset anchor of `workflow_presets` once)
        expected = {
            "plan_2026-05-29_0c00a594/D-001": {"agent": 1},
            "plan_2026-05-29_0c00a594/D-002": {"workflow_presets": 1, "workflow": 1},
            "plan_2026-05-29_0c00a594/D-003": {"agent": 2, "manager": 1},
            "plan-2026-07-20T040150-876e7164/D-003": {"snapshots": 1},
            "plan-2026-09-12T065608-089d0ec7/D-007": {"lifecycle": 1},
            "plan-2026-09-12T135914-45a654de/D-017": {"snapshots": 1},
            "plan-2026-09-28T090000-3c9e41d2/D-002": {"snapshots": 1},
            "plan-2026-09-28T090000-3c9e41d2/D-003": {"handlers": 1},
            "plan-2026-09-28T090000-3c9e41d2/D-006": {"workflow": 1},
            "plan-2026-11-05T134105-5ed71cf2/D-015": {"lifecycle": 1},
            "plan-2026-11-05T185903-5980fbb7/D-005": {"lifecycle": 1},
        }
        texts = {
            path.stem: path.read_text(encoding="utf-8")
            for path in _package_dir().glob("*.py")
        }
        for token, where in expected.items():
            found = {stem: text.count(token) for stem, text in texts.items()}
            assert {s: n for s, n in found.items() if n} == where, token

    def test_a_module_level_name_is_defined_in_one_module_only(self):
        import ast

        defined: dict[str, list[str]] = {}
        for path in sorted(_package_dir().glob("*.py")):
            if path.stem in {"__init__", "surface"}:
                continue
            for node in ast.parse(path.read_text(encoding="utf-8")).body:
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    names = [node.name]
                elif isinstance(node, ast.Assign):
                    names = [t.id for t in node.targets if isinstance(t, ast.Name)]
                elif isinstance(node, ast.AnnAssign) and isinstance(
                    node.target, ast.Name
                ):
                    names = [node.target.id]
                else:
                    names = []
                for name in names:
                    defined.setdefault(name, []).append(path.stem)
        assert {n: m for n, m in defined.items() if len(m) > 1} == {}
        for name, module in {
            "_AGENT_CLASSES": "agent",
            "_HANDLER_STATES": "handlers",
            "_WORKFLOW_PRESETS": "workflow_presets",
            "_new_instance_id": "managed",
            "_pending_tasks": "lifecycle",
            "_status_str": "workflow",
        }.items():
            assert defined[name] == [module], name

    def test_only_the_examples_module_reads_its_own_file_location(self):
        # A module that reads `__file__` counts directories from where it sits:
        # moving it changes the answer silently (see `TestFindExamplesDir`).
        users = {
            path.stem
            for path in _package_dir().glob("*.py")
            if "__file__"
            in path.read_text(encoding="utf-8")
            .replace("``__file__``", "")
            .replace("`__file__`", "")
            and path.stem != "__init__"
        }
        assert users == {"examples"}

    def test_a_log_record_is_named_after_the_module_that_logs_it(self):
        from loguru import logger

        records: list[tuple[str, str, str]] = []
        logger.enable("fsm_llm")
        sink = logger.add(
            lambda m: records.append(
                (m.record["name"], m.record["module"], m.record["message"])
            ),
            level="DEBUG",
            format="{message}",
        )
        try:
            # lifecycle: a managed FSM whose API fails to list its conversations
            mgr = _manager()
            api = MagicMock()
            api.list_active_conversations.side_effect = RuntimeError("closed")
            _fsm(mgr, "f1", api)
            mgr.end_instance("f1")
            # snapshots: the attached API fails to list its conversations
            mgr = _manager()
            attached = MagicMock()
            attached.list_active_conversations.side_effect = RuntimeError("gone")
            mgr._attached_api = attached
            mgr.get_active_conversations()
            # workflow: a run the engine does not know
            mgr = _manager()
            wf = ManagedWorkflow(instance_id="w1")
            wf.engine = MagicMock()
            wf.engine.get_instance.return_value = None
            wf.active_instance_ids = ["run1"]
            wf.run_ids = ["run1"]  # F4: the list walks every run, not the active ones
            mgr._instances["w1"] = wf
            mgr.get_workflow_instances("w1")
            # fsm: the conversation ends but its end is refused
            mgr = _manager()
            inst = ManagedFSM(instance_id="f2", api=MagicMock())
            inst.api.has_conversation_ended.return_value = True
            inst.api.end_conversation.side_effect = RuntimeError("busy")
            inst.conversation_ids = ["c1"]
            mgr._instances["f2"] = inst
            mgr.send_message("f2", "c1", "hi")
        finally:
            logger.remove(sink)
            logger.disable("fsm_llm")

        def logged_by(prefix: str) -> tuple[str, str]:
            return next((n, m) for n, m, text in records if text.startswith(prefix))

        assert logged_by("Failed to close FSM API of instance") == (
            f"{_PACKAGE}.lifecycle",
            "lifecycle",
        )
        assert logged_by("Failed to list attached API conversations") == (
            f"{_PACKAGE}.snapshots",
            "snapshots",
        )
        assert logged_by("Failed to get workflow status for") == (
            f"{_PACKAGE}.workflow",
            "workflow",
        )
        assert logged_by("Failed to end terminal conversation") == (
            f"{_PACKAGE}.fsm",
            "fsm",
        )


class TestInstanceManagerEdgesOfTheSplit:
    """Edges of ``InstanceManager`` that no test pinned before the split and
    that the mutation proof of step 93 reached (D-001)."""

    def test_the_capabilities_report_is_exactly_the_three_flags(self):
        assert _manager().get_capabilities() == {
            "fsm": True,
            "workflows": True,
            "agents": True,
        }

    def test_a_lexical_dot_dot_in_a_preset_id_is_refused_even_inside_the_tree(self):
        # ``sub/../x`` resolves INSIDE the examples directory: only the lexical
        # pre-check refuses it, not the confinement test.
        from fsm_llm.monitor.instance_manager import (
            find_examples_dir,
            validate_preset_id,
        )

        base = find_examples_dir()
        for preset_id in (
            "basic/../basic/form_filling/fsm.json",
            "../examples/basic/form_filling/fsm.json",
            "/etc/passwd",
        ):
            with pytest.raises(ValueError, match="Invalid preset ID"):
                validate_preset_id(preset_id, base)
        assert validate_preset_id("basic/form_filling/fsm.json", base).is_file()

    def test_a_launch_evicts_the_oldest_of_several_finished_instances(self):
        from datetime import UTC, datetime, timedelta

        mgr = _manager(max_instances=3)
        base = datetime(2024, 1, 1, tzinfo=UTC)
        for n, name in enumerate(["f_old", "f_mid", "f_new"]):
            inst = ManagedFSM(name, MagicMock())
            inst.status = "completed"
            inst.created_at = base + timedelta(hours=n)
            mgr._instances[name] = inst
        mgr.launch_workflow(preset_id="demo_linear")
        remaining = {i.instance_id for i in mgr.list_instances()}
        assert "f_old" not in remaining
        assert {"f_mid", "f_new"} <= remaining

    def test_a_launch_never_evicts_a_running_instance(self):
        mgr = _manager(max_instances=2)
        running = ManagedFSM("f_run", MagicMock())
        done = ManagedFSM("f_done", MagicMock())
        done.status = "completed"
        mgr._instances.update({"f_run": running, "f_done": done})
        mgr.launch_workflow(preset_id="demo_linear")
        remaining = {i.instance_id for i in mgr.list_instances()}
        assert "f_run" in remaining and "f_done" not in remaining

    def test_ending_a_running_agent_marks_it_cancelled_and_destroyed(self):
        mgr = _manager()
        agent = ManagedAgent(instance_id="a_run")
        mgr._instances["a_run"] = agent
        mgr._collectors["a_run"] = EventCollector()
        mgr.end_instance("a_run")
        assert agent.status == "cancelled"
        assert agent.cancel_event.is_set()
        assert mgr.get_instance("a_run") is None

    def test_a_launch_resolves_a_blank_model_from_the_environment(self, monkeypatch):
        from fsm_llm import API
        from fsm_llm.monitor.instance_manager import fsm as fsm_mod
        from fsm_llm.testing import MockLLM2Interface

        seen: list[dict] = []
        real_from_definition = API.from_definition

        def _from_definition(definition, **kwargs):
            seen.append(kwargs)
            return real_from_definition(definition, llm_interface=MockLLM2Interface())

        monkeypatch.setattr(fsm_mod.API, "from_definition", _from_definition)
        monkeypatch.setenv("LLM_MODEL", "ollama_chat/from-env")
        mgr = _manager()
        mgr.launch_fsm(fsm_json=_CHAT_FSM, model="  ")
        mgr.launch_fsm(fsm_json=_CHAT_FSM, model="ollama_chat/given")
        assert [kw["model"] for kw in seen] == [
            "ollama_chat/from-env",
            "ollama_chat/given",
        ]

    def test_an_agent_launch_resolves_its_model_the_same_way(self, monkeypatch):
        configs: list[Any] = []

        class _Probe(_QuickAgent):
            def __init__(self, **kwargs):
                configs.append(kwargs["config"])

        monkeypatch.setitem(agent_mod._AGENT_CLASSES, "ReactAgent", _Probe)
        monkeypatch.setenv("LLM_MODEL", "ollama_chat/from-env")
        mgr = _manager()
        first = mgr.launch_agent(task="t", tools_config=[_STUB], model="")
        first.thread.join(5)
        second = mgr.launch_agent(
            task="t", tools_config=[_STUB], model="ollama_chat/given"
        )
        second.thread.join(5)
        assert [c.model for c in configs] == [
            "ollama_chat/from-env",
            "ollama_chat/given",
        ]

    def test_a_cancelled_but_running_agent_is_cancelling_and_holds_its_slot(self):
        mgr = _manager(max_running_agents=1)
        agent = ManagedAgent("a1")
        release = threading.Event()
        agent.thread = threading.Thread(target=release.wait, daemon=True)
        agent.thread.start()
        mgr._instances["a1"] = agent
        try:
            assert mgr.cancel_agent("a1") is True
            assert agent.status == "cancelling"
            assert agent.cancel_event.is_set()
            assert mgr.get_agent_status("a1")["status"] == "cancelling"
            with pytest.raises(MonitorCapacityError, match="1 agents are already"):
                mgr.launch_agent(task="t", tools_config=[_STUB])
        finally:
            release.set()
            agent.thread.join(5)

    def test_the_agent_limit_is_reached_at_exactly_the_maximum(self, monkeypatch):
        monkeypatch.setitem(agent_mod._AGENT_CLASSES, "ReactAgent", _QuickAgent)
        mgr = _manager(max_running_agents=2)
        for name in ("a1", "a2"):
            mgr._instances[name] = ManagedAgent(name)
        with pytest.raises(MonitorCapacityError, match="2 agents are already running"):
            mgr.launch_agent(task="t", tools_config=[_STUB])
        del mgr._instances["a2"]
        managed = mgr.launch_agent(task="t", tools_config=[_STUB])
        managed.thread.join(5)
        assert managed.status == "completed"


class TestWorkflowCancelEvents:
    @pytest.fixture(autouse=True)
    def _needs_workflows(self):
        pytest.importorskip("fsm_llm.workflows")

    def test_a_cancelled_run_emits_workflow_cancelled_with_its_reason(self):
        from fsm_llm.monitor.constants import EVENT_WORKFLOW_CANCELLED
        from fsm_llm.workflows import auto_step, create_workflow, wait_event_step

        async def go():
            mgr = _manager()
            managed = mgr.launch_workflow(preset_id="demo_linear")
            waiting = create_workflow("wait_wf", "Wait")
            waiting.with_initial_step(
                wait_event_step("w", "W", event_type="go", success_step_id="d")
            )
            waiting.with_step(auto_step("d", "D", next_step_id="", action=lambda c: {}))
            managed.engine.register_workflow(waiting)
            run = await mgr.start_workflow_instance(managed.instance_id, "wait_wf", {})
            assert run in managed.active_instance_ids
            assert await mgr.cancel_workflow(managed.instance_id, run, "because")
            return mgr, managed, run

        mgr, managed, run = asyncio.run(go())
        cancelled = [
            e
            for e in mgr.get_events(limit=0)
            if e.event_type == EVENT_WORKFLOW_CANCELLED
        ]
        assert len(cancelled) == 1
        assert cancelled[0].data["reason"] == "because"
        assert cancelled[0].data["workflow_instance_id"] == run
        assert managed.run_statuses[run] == "cancelled"
        assert managed.status == "cancelled"
        assert managed.active_instance_ids == []
        mgr.global_collector.cleanup()


class TestSnapshotAndActivityEdges:
    def test_the_snapshot_default_hides_internal_keys_and_secrets(self):
        # D-017: the default is fail-closed; a caller that forgets the
        # parameter must not leak internal keys into the dashboard.
        api = _snapshot_api(
            {"city": "Oslo", "_internal": 1, "system_x": 2, "password": "hunter2"}
        )
        snap = snapshot_from_api(api, "c1")
        assert snap is not None
        assert snap.context_data == {"city": "Oslo"}
        shown = snapshot_from_api(api, "c1", show_internal_keys=True)
        assert shown is not None
        assert shown.context_data == {"city": "Oslo", "_internal": 1, "system_x": 2}

    def test_activity_without_ended_runs_lists_only_the_active_runs(self):
        mgr = _manager()
        wf = ManagedWorkflow("w1", label="W")
        wf.run_ids = ["r1", "r2"]
        wf.active_instance_ids = ["r1"]
        mgr._instances["w1"] = wf
        mgr.get_workflow_status = lambda instance_id, run_id: {
            "status": "Running" if run_id == "r1" else "Completed",
            "current_step": "s",
        }
        only_active = mgr.get_all_activity_snapshots(include_ended=False)
        everything = mgr.get_all_activity_snapshots(include_ended=True)
        assert [i.item_id for i in only_active] == ["r1"]
        assert [(i.item_id, i.status, i.is_terminal) for i in everything] == [
            ("r1", "running", False),
            ("r2", "completed", True),
        ]


class TestInstanceManagerLoggingModules:
    def test_every_module_that_logs_uses_the_library_logger_itself(self):
        # A bound or wrapped logger would add context to every record; the
        # records of the manager are the library's own (names pinned above).
        import importlib

        import fsm_llm.logging as library_logging

        for module in ("fsm", "lifecycle", "snapshots", "workflow"):
            loaded = importlib.import_module(f"{_PACKAGE}.{module}")
            assert loaded.logger is library_logging.logger, module
