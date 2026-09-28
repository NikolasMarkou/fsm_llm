"""
Regression tests for the 2026-09-28 fsm_llm_monitor audit (non-HTTP parts).

Each class pins one finding; every test here failed on the pre-fix code.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any
from unittest.mock import MagicMock

import pytest

from fsm_llm.handlers import HandlerTiming
from fsm_llm_monitor.bridge import MonitorBridge, _fsm_dict_to_snapshot
from fsm_llm_monitor.collector import EventCollector, redact_context
from fsm_llm_monitor.constants import (
    EVENT_STATE_TRANSITION,
    EVENT_WORKFLOW_ADVANCED,
    EVENT_WORKFLOW_COMPLETED,
    EVENT_WORKFLOW_EVENT_DELIVERED,
)
from fsm_llm_monitor.definitions import (
    LogRecord,
    MonitorConfig,
    MonitorEvent,
    TransitionInfo,
)
from fsm_llm_monitor.exceptions import MonitorCapacityError
from fsm_llm_monitor.instance_manager import (
    InstanceManager,
    ManagedAgent,
    ManagedFSM,
    _MonitorHandler,
    register_monitor_handlers,
    snapshot_from_api,
    unregister_monitor_handlers,
)

# ---------------------------------------------------------------
# helpers
# ---------------------------------------------------------------


class _Secretive:
    def __str__(self) -> str:
        return "password=hunter2"


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


def _manager() -> InstanceManager:
    mgr = InstanceManager(config=MonitorConfig())
    mgr.global_collector.cleanup()
    return mgr


# ---------------------------------------------------------------
# H1: transition events carry the core's states
# ---------------------------------------------------------------


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

    def test_capture_log_records_transition_and_end_state(self):
        agent = ManagedAgent("a1")
        collector = EventCollector()
        from fsm_llm_monitor.instance_manager import _build_context_capture_handlers

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


# ---------------------------------------------------------------
# H4: one redaction path
# ---------------------------------------------------------------


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


# ---------------------------------------------------------------
# M1: stream cursors
# ---------------------------------------------------------------


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


# ---------------------------------------------------------------
# M2: config bounds and effects
# ---------------------------------------------------------------


class TestConfig:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"refresh_interval": 0},
            {"refresh_interval": -1},
            {"refresh_interval": float("nan")},
            {"refresh_interval": float("inf")},
            {"max_events": -5},
            {"max_log_lines": 0},
            {"log_level": "LOUD"},
        ],
    )
    def test_invalid_values_rejected(self, kwargs):
        with pytest.raises(ValueError):
            MonitorConfig(**kwargs)

    def test_log_level_normalised(self):
        assert MonitorConfig(log_level="debug").log_level == "DEBUG"

    def test_setting_config_resizes_global_buffers(self):
        mgr = _manager()
        for _ in range(50):
            mgr.global_collector.record_event(MonitorEvent(event_type="t"))
        mgr.config = MonitorConfig(max_events=20)
        assert len(mgr.global_collector.get_events()) == 20
        mgr.global_collector.cleanup()


# ---------------------------------------------------------------
# M3: cancel_agent keeps finished agents
# ---------------------------------------------------------------


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


# ---------------------------------------------------------------
# M4: handler registration is idempotent and removable
# ---------------------------------------------------------------


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

    def test_bridge_reconnect_and_disconnect(self):
        api = _FakeAPI()
        bridge = MonitorBridge(api=api)
        bridge.connect(api)
        _fire(api, HandlerTiming.PRE_TRANSITION, "a", "b")
        assert bridge.collector.get_metrics().total_transitions == 1
        bridge.disconnect()
        _fire(api, HandlerTiming.PRE_TRANSITION, "b", "c")
        assert bridge.collector.get_metrics().total_transitions == 1

    def test_handler_never_writes_context(self):
        handler = _MonitorHandler(
            "h", HandlerTiming.PRE_TRANSITION, lambda *a, **k: None, EventCollector()
        )
        assert handler.should_execute(HandlerTiming.PRE_TRANSITION, "a", "b", {})
        assert handler.execute({"x": 1}) == {}


# ---------------------------------------------------------------
# M8 / L: instance lifecycle, eviction and status tracking
# ---------------------------------------------------------------


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


class TestDefaults:
    def test_transition_priority_defaults_to_core_default(self):
        assert TransitionInfo(target_state="x").priority == 100
        snap = _fsm_dict_to_snapshot(
            {
                "initial_state": "a",
                "states": {"a": {"transitions": [{"target_state": "b"}]}, "b": {}},
            }
        )
        assert snap.states[0].transitions[0].priority == 100


# ---------------------------------------------------------------
# Workflow tracking via engine hooks
# ---------------------------------------------------------------


workflows = pytest.importorskip("fsm_llm_workflows")


class TestWorkflowTracking:
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


class TestCliHelpers:
    def test_browser_url_for_wildcard_and_ipv6_binds(self):
        from fsm_llm_monitor.__main__ import _browser_url

        assert _browser_url("0.0.0.0", 8420) == "http://127.0.0.1:8420"
        assert _browser_url("::", 8420) == "http://127.0.0.1:8420"
        assert _browser_url("::1", 8420) == "http://[::1]:8420"
        assert _browser_url("localhost", 9000) == "http://localhost:9000"
