"""
Server consistency fixes of plan 5980fbb7 step 17: visualize routes without a
prior ``get_manager()``, security headers on guard refusals, workflow launch
off the event loop, failures logged on the monitoring routes, a lean ``/ws``
tick, the ``fsm-llm-monitor`` Ctrl-C code, and the two SPA pins that have no
JS harness (4403 close, ``highlightText`` inside entities).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from fsm_llm.constants import CLI_EXIT_INTERRUPTED
from fsm_llm.monitor import server as server_module
from fsm_llm.monitor.definitions import DashboardConfig, InstanceInfo, MonitorConfig
from fsm_llm.monitor.exceptions import MetricCollectionError
from fsm_llm.monitor.instance_manager import InstanceManager
from fsm_llm.monitor.server import app, configure

STATIC = Path(server_module.__file__).parent / "static"


def _client() -> tuple[TestClient, InstanceManager]:
    mgr = InstanceManager(config=MonitorConfig(refresh_interval=0.5))
    configure(manager=mgr)
    return TestClient(app), mgr


@pytest.fixture(autouse=True)
def _reset_key():
    yield
    with mock.patch.dict(os.environ, {"FSM_LLM_MONITOR_API_KEY": ""}):
        configure(manager=InstanceManager())


class TestVisualizeWithoutManager:
    """F-MS-03: the pattern flows load on first use, not on the first
    ``get_manager()`` call (``uvicorn fsm_llm.monitor.server:app`` never
    calls ``configure()``)."""

    @pytest.mark.parametrize(
        "url",
        [
            "/api/agent/visualize?agent_type=ReactAgent",
            "/api/workflow/visualize?workflow_id=order_processing",
        ],
    )
    def test_first_request_finds_the_flow(self, url, monkeypatch):
        client, _ = _client()
        monkeypatch.setattr(server_module, "_flows", {})
        resp = client.get(url)
        assert resp.status_code == 200, resp.text
        assert resp.json()["nodes"]

    def test_visualize_payload_carries_no_server_layout(self):
        client, _ = _client()
        data = client.get("/api/agent/visualize?agent_type=ReactAgent").json()
        assert set(data) == {"info", "nodes", "edges"}
        assert all("x" not in n and "y" not in n for n in data["nodes"])
        wf = client.get("/api/workflow/visualize?workflow_id=order_processing").json()
        assert set(wf) == {"info", "nodes", "edges"}


class TestGuardRefusalsCarrySecurityHeaders:
    """F-MS-09: a 400/403/413 from the security guard is still a response of
    this server; it carries CSP, nosniff and frame headers like any other."""

    _NAMES = (
        "Content-Security-Policy",
        "X-Frame-Options",
        "X-Content-Type-Options",
        "Referrer-Policy",
    )

    def _assert_headers(self, resp):
        for name in self._NAMES:
            assert name in resp.headers, name

    def test_bad_host(self):
        client, _ = _client()
        resp = client.get("/health", headers={"Host": "attacker.example"})
        assert resp.status_code == 400
        self._assert_headers(resp)

    def test_bad_origin(self):
        client, _ = _client()
        resp = client.post("/api/agent/x/cancel", headers={"Origin": "http://evil.x"})
        assert resp.status_code == 403
        self._assert_headers(resp)

    def test_oversized_body(self):
        client, _ = _client()
        resp = client.post(
            "/api/config", content=b"x", headers={"Content-Length": str(10**9)}
        )
        assert resp.status_code == 413
        self._assert_headers(resp)


class TestWorkflowLaunchStaysOffTheLoop:
    """F-MS-10: ``launch_workflow`` evicts finished instances (a thread join,
    an ``API.close``), so it runs in a worker like the other launch routes."""

    def test_launch_workflow_runs_in_a_worker_thread(self):
        client, mgr = _client()
        seen: dict[str, bool] = {}

        def fake_launch(**_kw):
            try:
                asyncio.get_running_loop()
                seen["on_loop"] = True
            except RuntimeError:
                seen["on_loop"] = False
            info = InstanceInfo(instance_id="wf1", instance_type="workflow")
            return SimpleNamespace(
                instance_id="wf1", workflow_id="w", to_info=lambda: info
            )

        async def fake_start(*_a, **_kw):
            return "run-1"

        mgr.launch_workflow = fake_launch  # type: ignore[method-assign]
        mgr.start_workflow_instance = fake_start  # type: ignore[method-assign]
        resp = client.post("/api/workflow/launch", json={"preset_id": "demo_linear"})
        assert resp.status_code == 200, resp.text
        assert resp.json()["workflow_instance_id"] == "run-1"
        assert seen == {"on_loop": False}


class TestMonitoringFailuresAreLogged:
    """F-MS-11: a ``MonitorError`` from a monitoring route is still a 500 with
    the generic detail, and the cause is logged."""

    def test_metrics_failure_is_logged(self):
        client, mgr = _client()
        mgr.get_metrics = mock.Mock(  # type: ignore[method-assign]
            side_effect=MetricCollectionError("collector exploded")
        )
        with mock.patch.object(server_module, "logger") as log:
            resp = client.get("/api/metrics")
        assert resp.status_code == 500
        assert resp.json() == {"detail": "Internal server error"}
        assert any("collector exploded" in str(c) for c in log.error.call_args_list)

    def test_builder_internal_state_failure_is_logged(self):
        agent = mock.Mock()
        agent.get_internal_state.side_effect = RuntimeError("no state")
        with mock.patch.object(server_module, "logger") as log:
            state = server_module._get_builder_internal_state(agent)
        assert state == {"phase": "unknown", "turn_count": 0}
        assert log.debug.called or log.warning.called


class TestWebSocketTick:
    """F-MS-02 / F-XC-17: the tick ships what ``ws.js`` reads and nothing else
    (no per-workflow status build, no agent logs, no ``log_count``, no
    dashboard config body)."""

    def _first_tick(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"type": "auth", "api_key": ""}))
            return ws.receive_json()

    def test_no_dead_fields(self):
        client, _ = _client()
        data = self._first_tick(client)
        assert data["type"] == "metrics"
        assert "log_count" not in data
        assert "dashboard_config" not in data

    def test_running_workflows_are_ids_and_cost_no_status_build(self):
        client, mgr = _client()
        info = InstanceInfo(
            instance_id="wf1", instance_type="workflow", status="running"
        )
        mgr.list_instances = lambda *a, **k: [info]  # type: ignore[method-assign]
        mgr.get_workflow_instances = mock.Mock(  # type: ignore[method-assign]
            side_effect=AssertionError("per-workflow status built on the tick")
        )
        data = self._first_tick(client)
        assert data["workflow_updates"] == ["wf1"]

    def test_running_agents_ship_a_summary_not_the_log(self):
        client, mgr = _client()
        info = InstanceInfo(instance_id="ag1", instance_type="agent", status="running")
        mgr.list_instances = lambda *a, **k: [info]  # type: ignore[method-assign]
        mgr.get_agent_status = lambda _id: {  # type: ignore[method-assign]
            "instance_id": "ag1",
            "status": "running",
            "current_state": "think",
            "iteration_count": 2,
            "conversation_log": [{"type": "start"}, {"type": "context", "data": {}}],
        }
        update = self._first_tick(client)["agent_updates"]["ag1"]
        assert "conversation_log" not in update
        assert update["log_entries"] == 2
        assert update["iteration_count"] == 2

    def test_dashboard_config_change_ships_a_version_only(self):
        client, mgr = _client()
        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"type": "auth", "api_key": ""}))
            ws.receive_json()
            mgr.dashboard_config = DashboardConfig(name="custom")
            for _ in range(5):
                data = ws.receive_json()
                if "dashboard_config_version" in data:
                    break
            else:
                pytest.fail("dashboard_config_version never pushed")
        assert data["dashboard_config_version"] == mgr.dashboard_config_version
        assert "dashboard_config" not in data


class TestMonitorCliExit:
    """F-MS-13: Ctrl-C ends ``fsm-llm-monitor`` with 130 like every other CLI
    (``uvicorn.run`` swallows ``KeyboardInterrupt`` and exits 0)."""

    def test_ctrl_c_exits_130(self, monkeypatch):
        import uvicorn

        from fsm_llm.monitor import __main__ as monitor_main

        monkeypatch.setattr(sys, "argv", ["fsm-llm-monitor", "--no-browser"])
        monkeypatch.setattr(
            uvicorn.Server, "run", mock.Mock(side_effect=KeyboardInterrupt)
        )
        with pytest.raises(SystemExit) as exc:
            monitor_main.main_cli()
        assert exc.value.code == CLI_EXIT_INTERRUPTED


class TestStaticPins:
    """No JS harness: pin SPA behaviour on the static files."""

    def test_ws_client_stops_on_4403_like_4401(self):
        ws_js = (STATIC / "services" / "ws.js").read_text(encoding="utf-8")
        assert f"= {server_module._WS_CLOSE_FORBIDDEN};" in ws_js
        assert f"= {server_module._WS_CLOSE_UNAUTHORIZED};" in ws_js
        assert "WS_FORBIDDEN" in ws_js.split("ws.onclose", 1)[1]

    def test_ws_client_reads_only_what_the_server_sends(self):
        ws_js = (STATIC / "services" / "ws.js").read_text(encoding="utf-8")
        assert "dashboard_config_version" in ws_js
        for dead in ("log_count", "state.agentUpdates", "state.workflowUpdates"):
            assert dead not in ws_js

    @pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
    @pytest.mark.parametrize(
        ("text", "query", "expected"),
        [
            ("a & b", "amp", "a &amp; b"),
            ("x < y", "lt", "x &lt; y"),
            ("it's", "39", "it&#39;s"),
            ("Error: boom", "err", '<span class="search-highlight">Err</span>or: boom'),
            ("a & b", "&", 'a <span class="search-highlight">&amp;</span> b'),
            ("a<b", "<", 'a<span class="search-highlight">&lt;</span>b'),
        ],
    )
    def test_highlight_text_does_not_match_inside_entities(self, text, query, expected):
        dom = (STATIC / "utils" / "dom.js").as_uri()
        script = (
            f"import({json.dumps(dom)}).then(m => "
            f"process.stdout.write(m.highlightText({json.dumps(text)}, "
            f"{json.dumps(query)})))"
        )
        proc = subprocess.run(
            ["node", "-e", script], capture_output=True, text=True, timeout=60
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout == expected

    @pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
    @pytest.mark.parametrize(
        "path", sorted(p.relative_to(STATIC).as_posix() for p in STATIC.rglob("*.js"))
    )
    def test_every_static_module_parses(self, path):
        proc = subprocess.run(
            ["node", "--check", str(STATIC / path)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 0, proc.stderr

    @pytest.mark.parametrize(
        "path", sorted(p.relative_to(STATIC).as_posix() for p in STATIC.rglob("*.js"))
    )
    def test_every_named_import_is_exported_by_its_module(self, path):
        """The un-exports and moves of F-MS-16/17 must not strand an importer."""
        source = (STATIC / path).read_text(encoding="utf-8")
        for names, module in re.findall(
            r"import\s*\{([^}]*)\}\s*from\s*'([^']+)'", source
        ):
            target = (STATIC / path).parent / module
            text = target.read_text(encoding="utf-8")
            exported = set(
                re.findall(
                    r"export\s+(?:async\s+)?(?:function|const|let|class)\s+([\w$]+)",
                    text,
                )
            )
            for group in re.findall(r"export\s*\{([^}]*)\}", text):
                exported |= {n.split(" as ")[-1].strip() for n in group.split(",")}
            wanted = {n.split(" as ")[0].strip() for n in names.split(",") if n.strip()}
            assert wanted <= exported, (path, module, wanted - exported)

    @pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
    def test_pager_is_one_helper_with_the_old_markup(self):
        dom = (STATIC / "utils" / "dom.js").as_uri()
        script = (
            f"import({json.dumps(dom)}).then(m => {{"
            "const one = {style: {}}, two = {style: {}};"
            "m.renderPagination(one, 0, 1, 'x-page');"
            "m.renderPagination(two, 1, 3, 'inst-page');"
            "process.stdout.write(JSON.stringify([one, two]));})"
        )
        proc = subprocess.run(
            ["node", "-e", script], capture_output=True, text=True, timeout=60
        )
        assert proc.returncode == 0, proc.stderr
        one, two = json.loads(proc.stdout)
        assert one == {"style": {"display": "none"}}
        assert two["style"] == {"display": "flex"}
        assert two["innerHTML"] == (
            '<button class="btn btn-sm" data-action="inst-page-prev">&laquo; Prev</button>'
            '<span class="pagination-info">Page 2 of 3</span>'
            '<button class="btn btn-sm" data-action="inst-page-next">Next &raquo;</button>'
        )
