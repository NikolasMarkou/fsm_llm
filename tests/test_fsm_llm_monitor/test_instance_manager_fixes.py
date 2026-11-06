"""Behaviour fixes of the instance manager, collector and OTEL exporter
(plan-2026-11-05T185903-5980fbb7 step 16: F-MI-01, 05, 09, 10, 12, 13)."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from fsm_llm.monitor import instance_manager as im
from fsm_llm.monitor.constants import EVENT_AGENT_STARTED
from fsm_llm.monitor.definitions import MonitorConfig, MonitorEvent, StubToolConfig
from fsm_llm.monitor.instance_manager import InstanceManager, ManagedFSM

_STUB = StubToolConfig(name="lookup", description="d", stub_response="R")


def _manager(**config) -> InstanceManager:
    mgr = InstanceManager(config=MonitorConfig(**config))
    mgr.global_collector.cleanup()
    return mgr


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
        monkeypatch.setitem(im._AGENT_CLASSES, "ReactAgent", _QuickAgent)
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
        monkeypatch.setitem(im._AGENT_CLASSES, "ReactAgent", _QuickAgent)
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


_UTF8_SCRIPT = """
import json
from pathlib import Path
from fsm_llm.monitor import server
from fsm_llm.monitor.instance_manager import InstanceManager, find_examples_dir

preset = "basic/form_filling/fsm.json"
expected = json.loads(
    (find_examples_dir() / preset).read_bytes().decode("utf-8")
)
mgr = InstanceManager()
mgr.global_collector.cleanup()
assert mgr._resolve_fsm_data(preset, None) == expected
assert server._read_preset_json(preset) == expected
scanned = server._scan_fsm_presets()["fsm"]
assert not [p for p in scanned if p.get("parse_error")], scanned
assert server._load_flows()["agents"]
print("OK")
"""


class TestPresetJsonIsReadAsUtf8:
    """F-MI-12 / F-MS-01: presets and flows.json decode as UTF-8 whatever the
    locale says."""

    def test_presets_load_under_an_ascii_locale(self):
        env = dict(
            os.environ,
            PYTHONUTF8="0",
            PYTHONCOERCECLOCALE="0",
            LC_ALL="C",
            LANG="C",
        )
        proc = subprocess.run(
            [sys.executable, "-c", _UTF8_SCRIPT],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert "OK" in proc.stdout


class TestShutdownTearsDownEveryInstanceType:
    """F-MI-13: shutdown() does what destroy_instance() does for agents and
    workflow engines, not only for FSMs."""

    def test_shutdown_sets_the_agent_cancel_flag(self, monkeypatch):
        _BlockingAgent.release.clear()
        monkeypatch.setitem(im._AGENT_CLASSES, "ReactAgent", _BlockingAgent)
        mgr = _manager()
        managed = mgr.launch_agent(task="t", tools_config=[_STUB])
        try:
            mgr.shutdown()
            assert managed.cancel_event.is_set()
        finally:
            _BlockingAgent.release.set()
            managed.thread.join(5)

    def test_shutdown_stops_the_workflow_engine(self):
        mgr = _manager()
        stopped: list[int] = []

        async def go() -> None:
            managed = mgr.launch_workflow(preset_id="demo_linear")

            async def _shutdown() -> None:
                stopped.append(1)

            managed.engine.shutdown = _shutdown
            mgr.shutdown()
            await asyncio.sleep(0.05)

        asyncio.run(go())
        assert stopped == [1]


class TestRouteLaunchedWorkflowIsStopped:
    """REVIEW-C1: ``POST /api/workflow/launch`` runs ``launch_workflow`` in a
    worker thread, where no loop is running; the engine still lives on the
    server loop, so destroy and shutdown must stop it there."""

    @staticmethod
    def _launch(client, mgr):
        r = client.post(
            "/api/workflow/launch",
            json={"preset_id": mgr.get_workflow_presets()[0]["id"]},
        )
        assert r.status_code == 200, r.text
        inst = mgr._instances[r.json()["instance_id"]]
        stopped: list[int] = []
        original = inst.engine.shutdown

        async def _spy() -> None:
            stopped.append(1)
            await original()

        inst.engine.shutdown = _spy
        return inst, stopped

    @staticmethod
    def _wait(stopped: list[int]) -> None:
        import time

        deadline = time.monotonic() + 2
        while not stopped and time.monotonic() < deadline:
            time.sleep(0.02)

    def test_destroy_stops_the_engine(self):
        from fastapi.testclient import TestClient

        from fsm_llm.monitor import server

        mgr = _manager()
        server.configure(manager=mgr)
        with TestClient(server.app) as client:
            inst, stopped = self._launch(client, mgr)
            assert inst.loop is not None
            r = client.delete(f"/api/instances/{inst.instance_id}")
            assert r.status_code == 200, r.text
            self._wait(stopped)
        assert stopped == [1]

    def test_shutdown_stops_the_engine(self):
        from fastapi.testclient import TestClient

        from fsm_llm.monitor import server

        mgr = _manager()
        server.configure(manager=mgr)
        with TestClient(server.app) as client:
            _inst, stopped = self._launch(client, mgr)
            mgr.shutdown()
            self._wait(stopped)
        assert stopped == [1]
