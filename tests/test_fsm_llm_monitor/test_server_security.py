"""
Server-level regression tests for the 2026-09-28 fsm_llm.monitor audit:
origin/host checks, API-key gating of sensitive reads and the WebSocket,
error mapping, request bounds, dashboard config parsing and builder guards.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from unittest import mock

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from fsm_llm.definitions import ConversationBusyError
from fsm_llm.monitor import server as server_module
from fsm_llm.monitor.definitions import MonitorConfig
from fsm_llm.monitor.exceptions import MonitorCapacityError
from fsm_llm.monitor.instance_manager import InstanceManager, validate_preset_id
from fsm_llm.monitor.server import app, configure


def _client(api_key: str | None = None) -> TestClient:
    mgr = InstanceManager(config=MonitorConfig(refresh_interval=0.5))
    configure(manager=mgr, api_key=api_key)
    return TestClient(app)


@pytest.fixture(autouse=True)
def _reset_key():
    yield
    with mock.patch.dict(os.environ, {"FSM_LLM_MONITOR_API_KEY": ""}):
        configure(manager=InstanceManager())


class TestOriginAndHost:
    def test_cross_origin_post_refused(self):
        client = _client()
        resp = client.post(
            "/api/agent/x/cancel", headers={"Origin": "http://evil.example"}
        )
        assert resp.status_code == 403

    def test_same_origin_post_allowed(self):
        client = _client()
        resp = client.post(
            "/api/agent/x/cancel", headers={"Origin": "http://testserver"}
        )
        assert resp.status_code == 404  # reached the route: unknown agent

    def test_foreign_host_header_refused(self):
        client = _client()
        resp = client.get("/health", headers={"Host": "attacker.example"})
        assert resp.status_code == 400

    def test_oversized_body_refused(self):
        client = _client()
        resp = client.post("/api/fsm/load", content=b"{" + b" " * 1_100_000 + b"}")
        assert resp.status_code == 413

    def test_security_headers_present(self):
        client = _client()
        resp = client.get("/health")
        assert "frame-ancestors 'none'" in resp.headers["content-security-policy"]
        assert resp.headers["x-content-type-options"] == "nosniff"


class TestApiKeyGating:
    def test_auth_endpoint_reports_requirement(self):
        assert _client().get("/api/auth").json() == {"auth_required": False}
        assert _client("s3cr3t").get("/api/auth").json() == {"auth_required": True}

    @pytest.mark.parametrize(
        "path",
        [
            "/api/conversations",
            "/api/activity",
            "/api/events",
            "/api/logs",
            "/api/agent/x/status",
            "/api/agent/x/result",
            "/api/builder/result/x",
            "/api/fsm/x/conversations",
        ],
    )
    def test_sensitive_reads_need_the_key(self, path):
        client = _client("s3cr3t")
        assert client.get(path).status_code == 401
        assert client.get(path, headers={"X-API-Key": "s3cr3t"}).status_code != 401

    def test_env_key_applies_without_configure(self):
        script = (
            "import os; os.environ['FSM_LLM_MONITOR_API_KEY']='k1';"
            "from fsm_llm.monitor import server; print(server._api_key)"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert proc.stdout.strip().splitlines()[-1] == "k1"

    def test_reconfiguring_the_same_manager_keeps_its_log_sink(self):
        mgr = InstanceManager()
        configure(manager=mgr)
        sink = mgr.global_collector._log_sink_id
        configure(manager=mgr, api_key="k2")
        assert sink is not None
        assert mgr.global_collector._log_sink_id == sink


class TestWebSocket:
    def test_streams_without_a_key(self):
        client = _client()
        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"type": "auth", "api_key": ""}))
            assert ws.receive_json()["type"] == "metrics"

    def test_wrong_key_closes_4401(self):
        client = _client("s3cr3t")
        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"type": "auth", "api_key": "nope"}))
            with pytest.raises(WebSocketDisconnect) as exc:
                ws.receive_text()
        assert exc.value.code == 4401

    def test_right_key_streams(self):
        client = _client("s3cr3t")
        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"type": "auth", "api_key": "s3cr3t"}))
            assert ws.receive_json()["type"] == "metrics"

    def test_cross_origin_socket_closed(self):
        client = _client()
        with client.websocket_connect(
            "/ws", headers={"origin": "http://evil.example"}
        ) as ws:
            with pytest.raises(WebSocketDisconnect) as exc:
                ws.receive_text()
        assert exc.value.code == 4403


class TestErrorMapping:
    def test_busy_conversation_is_409(self):
        client = _client()
        with mock.patch.object(
            InstanceManager, "send_message", side_effect=ConversationBusyError("busy")
        ):
            resp = client.post(
                "/api/fsm/x/converse", json={"message": "hi", "conversation_id": "c"}
            )
        assert resp.status_code == 409

    def test_capacity_is_429(self):
        client = _client()
        with mock.patch.object(
            InstanceManager, "launch_agent", side_effect=MonitorCapacityError("full")
        ):
            resp = client.post("/api/agent/launch", json={"task": "x"})
        assert resp.status_code == 429

    def test_missing_extension_is_501(self):
        client = _client()
        with mock.patch.object(
            InstanceManager,
            "launch_workflow",
            side_effect=NotImplementedError("not installed"),
        ):
            resp = client.post("/api/workflow/launch", json={"preset_id": "x"})
        assert resp.status_code == 501

    def test_unknown_fsm_preset_is_404(self):
        client = _client()
        resp = client.post(
            "/api/fsm/launch", json={"preset_id": "basic/nope/missing.json"}
        )
        assert resp.status_code == 404


class TestRequestBounds:
    @pytest.mark.parametrize(
        "body",
        [
            {"refresh_interval": 0},
            {"max_events": -1},
            {"log_level": "LOUD"},
        ],
    )
    def test_config_bounds(self, body):
        client = _client()
        assert client.post("/api/config", json=body).status_code == 422

    @pytest.mark.parametrize(
        "body",
        [
            {"task": "x", "max_iterations": 0},
            {"task": "x", "max_iterations": 10_000},
            {"task": "x", "timeout_seconds": 0},
            {"task": ""},
        ],
    )
    def test_agent_launch_bounds(self, body):
        client = _client()
        assert client.post("/api/agent/launch", json=body).status_code == 422

    def test_message_length_bound(self):
        client = _client()
        resp = client.post(
            "/api/fsm/x/converse",
            json={"message": "x" * 10_001, "conversation_id": "c"},
        )
        assert resp.status_code == 422


class TestDashboardConfig:
    def test_unwrapped_builder_output_is_applied(self):
        client = _client()
        resp = client.post(
            "/api/dashboard/config",
            json={
                "name": "Ops",
                "panels": {"p1": {"title": "Errors", "metric": "total_errors"}},
                "alerts": {},
                "config": {"refresh_interval_seconds": 15, "retention_hours": 2},
            },
        )
        assert resp.status_code == 200
        cfg = client.get("/api/dashboard/config").json()["config"]
        assert cfg["name"] == "Ops"
        assert cfg["refresh_interval_seconds"] == 15
        assert cfg["panels"][0]["title"] == "Errors"

    def test_malformed_body_is_400(self):
        client = _client()
        resp = client.post(
            "/api/dashboard/config", json={"panels": ["not", "a", "map"]}
        )
        assert resp.status_code == 400


class TestBuilderGuards:
    def test_busy_session_reports_busy_and_refuses_delete(self):
        client = _client()
        server_module._builder_sessions["b1"] = (mock.MagicMock(), 0.0)
        server_module._builder_busy.add("b1")
        try:
            assert client.get("/api/builder/result/b1").json()["busy"] is True
            assert client.delete("/api/builder/b1").status_code == 409
        finally:
            server_module._builder_busy.discard("b1")
            server_module._builder_sessions.pop("b1", None)


class TestPresetValidation:
    def test_traversal_and_absolute_ids_rejected(self, tmp_path):
        (tmp_path / "ok.json").write_text("{}")
        assert validate_preset_id("ok.json", tmp_path) == tmp_path / "ok.json"
        for bad in ("../etc/passwd", "/etc/passwd", "a/../../x.json"):
            with pytest.raises(ValueError):
                validate_preset_id(bad, tmp_path)
        with pytest.raises(FileNotFoundError):
            validate_preset_id("missing.json", tmp_path)
