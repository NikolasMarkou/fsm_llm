"""
Server-level regression tests for the 2026-09-28 fsm_llm.monitor audit:
origin/host checks, API-key gating of sensitive reads and the WebSocket,
error mapping, request bounds, dashboard config parsing and builder guards.
"""

from __future__ import annotations

import errno
import json
import os
import subprocess
import sys
from unittest import mock

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from fsm_llm.core.definitions import ConversationBusyError
from fsm_llm.monitor.definitions import MonitorConfig
from fsm_llm.monitor.exceptions import MonitorCapacityError
from fsm_llm.monitor.instance_manager import InstanceManager, validate_preset_id
from fsm_llm.monitor.server import app, configure, routes_builder


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

    def test_null_origin_post_refused(self):
        # sandboxed iframes and some redirects send `Origin: null`
        client = _client()
        resp = client.post("/api/agent/x/cancel", headers={"Origin": "null"})
        assert resp.status_code == 403
        assert resp.json() == {"detail": "cross-origin request refused"}

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
            "from fsm_llm.monitor.server import application; print(application._api_key)"
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

    def test_configuring_another_manager_closes_the_previous_one(self):
        first, second = InstanceManager(), InstanceManager()
        configure(manager=first)
        with mock.patch.object(first, "close", wraps=first.close) as closed:
            configure(manager=second)
        closed.assert_called_once_with()
        assert first.global_collector._log_sink_id is None
        assert second.global_collector._log_sink_id is not None


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

    @pytest.mark.parametrize(
        "method, path, body, patched, detail",
        [
            (
                "post",
                "/api/fsm/x/converse",
                {"message": "hi", "conversation_id": "c"},
                "send_message",
                "LLM operation timed out",
            ),
            (
                "post",
                "/api/fsm/x/start",
                {},
                "start_conversation",
                "LLM operation timed out",
            ),
        ],
    )
    def test_a_timed_out_llm_call_is_504(self, method, path, body, patched, detail):
        # `_bounded`: a timeout is a 504 with the route's own detail, not a 500
        client = _client()
        with mock.patch.object(InstanceManager, patched, side_effect=TimeoutError()):
            resp = getattr(client, method)(path, json=body)
        assert resp.status_code == 504
        assert resp.json() == {"detail": detail}

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

    def test_a_long_agent_type_is_refused_with_a_400_that_does_not_echo_it(self):
        """The id bound is the exact-match 400: a pydantic 422 would carry the raw id in ``input``."""
        client = _client()
        resp = client.post(
            "/api/agent/launch", json={"task": "x", "agent_type": "<i>" + "x" * 65}
        )
        assert resp.status_code == 400
        assert "x" * 65 not in resp.text

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
        routes_builder._builder_sessions["b1"] = (mock.MagicMock(), 0.0)
        routes_builder._builder_busy.add("b1")
        try:
            assert client.get("/api/builder/result/b1").json()["busy"] is True
            assert client.delete("/api/builder/b1").status_code == 409
        finally:
            routes_builder._builder_busy.discard("b1")
            routes_builder._builder_sessions.pop("b1", None)

    def test_a_second_send_to_a_busy_session_is_refused(self):
        client = _client()
        agent = mock.MagicMock()
        routes_builder._builder_sessions["b2"] = (agent, 0.0)
        routes_builder._builder_busy.add("b2")
        try:
            resp = client.post(
                "/api/builder/send", json={"session_id": "b2", "message": "again"}
            )
            assert resp.status_code == 409
            assert resp.json() == {
                "detail": "Builder session is processing another message"
            }
            agent.send.assert_not_called()
        finally:
            routes_builder._builder_busy.discard("b2")
            routes_builder._builder_sessions.pop("b2", None)

    def test_the_stale_sweep_keeps_a_session_with_a_send_in_flight(self):
        import asyncio
        import time

        old = time.time() - 2 * routes_builder._BUILDER_SESSION_TTL
        routes_builder._builder_sessions["idle"] = (mock.MagicMock(), old)
        routes_builder._builder_sessions["busy"] = (mock.MagicMock(), old)
        routes_builder._builder_sessions["fresh"] = (mock.MagicMock(), time.time())
        routes_builder._builder_busy.add("busy")
        try:
            asyncio.run(routes_builder._cleanup_stale_builder_sessions())
            assert sorted(routes_builder._builder_sessions) == ["busy", "fresh"]
        finally:
            routes_builder._builder_busy.discard("busy")
            for sid in ("idle", "busy", "fresh"):
                routes_builder._builder_sessions.pop(sid, None)


_GOOD_DASHBOARD = {
    "name": "Ops",
    "panels": {"p1": {"title": "Errors", "metric": "total_errors"}},
    "alerts": {"a1": {"metric": "total_errors", "threshold": 3}},
    "config": {"refresh_interval_seconds": 15, "retention_hours": 2},
}


class TestDashboardConfigRefusesWhatItWouldDrop:
    """F8 (monitor audit 2.2): a panel or alert that is not an object, and a
    body that is the whole ``GET`` answer, are refused with 400 and leave the
    applied config alone, instead of dropping the item (or the whole config)
    silently behind a 200."""

    def _applied(self, client):
        assert (
            client.post("/api/dashboard/config", json=_GOOD_DASHBOARD).status_code
            == 200
        )
        return client.get("/api/dashboard/config").json()

    @pytest.mark.parametrize(
        "body",
        [
            {"panels": {"p1": 1}},
            {"panels": {"p1": "x"}},
            {"panels": {"p1": None}},
            {"panels": {"p1": {"title": "ok"}, "p2": [1]}},
            {"alerts": {"a1": 3}},
            {"alerts": {"a1": {"metric": "m"}, "a2": "x"}},
            {"config": {"panels": {"p1": 7}}},
        ],
        ids=[
            "panel-int",
            "panel-str",
            "panel-null",
            "one-bad-panel-among-good",
            "alert-int",
            "one-bad-alert-among-good",
            "wrapped-bad-panel",
        ],
    )
    def test_a_non_object_item_is_400_and_the_config_stays(self, body):
        client = _client()
        before = self._applied(client)

        resp = client.post("/api/dashboard/config", json=body)

        assert resp.status_code == 400
        assert client.get("/api/dashboard/config").json() == before

    def test_posting_the_whole_get_answer_is_400_and_does_not_wipe_the_config(self):
        client = _client()
        before = self._applied(client)

        resp = client.post("/api/dashboard/config", json=before)

        assert resp.status_code == 400
        assert client.get("/api/dashboard/config").json() == before

    def test_the_config_member_of_the_get_answer_is_still_400(self):
        client = _client()
        before = self._applied(client)

        for body in (before["config"], {"config": before["config"]}):
            assert client.post("/api/dashboard/config", json=body).status_code == 400
        assert client.get("/api/dashboard/config").json() == before

    def test_a_good_body_still_applies_and_replaces(self):
        client = _client()
        self._applied(client)

        resp = client.post(
            "/api/dashboard/config",
            json={"name": "Two", "panels": {"p9": {"title": "T", "metric": "m"}}},
        )

        assert resp.status_code == 200
        cfg = client.get("/api/dashboard/config").json()["config"]
        assert cfg["name"] == "Two" and [p["panel_id"] for p in cfg["panels"]] == ["p9"]


class _GatedAgent:
    """A builder agent whose ``send`` blocks on a gate (a slow worker thread)."""

    def __init__(self) -> None:
        import threading

        self.started = threading.Event()
        self.gate = threading.Event()

    def send(self, message: str) -> str:
        self.started.set()
        self.gate.wait(30)
        return "done"

    def is_complete(self) -> bool:
        return False


class TestACancelledBuilderSendReleasesTheSession:
    """F7 (monitor audit 1.2): a handler task that is cancelled (a server or
    embedder that cancels it) leaves its worker thread running; the session is
    released when that thread ends, never left busy for good (a stuck session
    answers every later send and delete with 409)."""

    def _cancel_then_finish(self) -> bool:
        import asyncio

        from fsm_llm.monitor.definitions import BuilderSendRequest

        agent = _GatedAgent()
        routes_builder._builder_sessions["b1"] = (agent, 0.0)

        async def scenario() -> bool:
            request = BuilderSendRequest(session_id="b1", message="hello")
            task = asyncio.ensure_future(routes_builder.api_builder_send(request))
            for _ in range(500):
                if agent.started.is_set():
                    break
                await asyncio.sleep(0.01)
            assert agent.started.is_set() and "b1" in routes_builder._builder_busy
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            # still busy while the worker thread is running: no second send
            assert "b1" in routes_builder._builder_busy
            agent.gate.set()
            for _ in range(500):
                if "b1" not in routes_builder._builder_busy:
                    break
                await asyncio.sleep(0.01)
            return "b1" not in routes_builder._builder_busy

        try:
            return asyncio.run(scenario())
        finally:
            agent.gate.set()
            routes_builder._builder_busy.discard("b1")
            routes_builder._builder_sessions.pop("b1", None)

    def test_the_session_is_released_when_the_worker_ends(self):
        assert self._cancel_then_finish() is True


class TestPresetValidation:
    def test_traversal_and_absolute_ids_rejected(self, tmp_path):
        (tmp_path / "ok.json").write_text("{}", encoding="utf-8")
        assert validate_preset_id("ok.json", tmp_path) == tmp_path / "ok.json"
        for bad in ("../etc/passwd", "/etc/passwd", "a/../../x.json"):
            with pytest.raises(ValueError):
                validate_preset_id(bad, tmp_path)
        with pytest.raises(FileNotFoundError):
            validate_preset_id("missing.json", tmp_path)

    @pytest.mark.parametrize(
        "failure",
        [
            OSError(errno.ELOOP, "Too many levels of symbolic links"),
            OSError(errno.EACCES, "Permission denied"),
            RuntimeError("Symlink loop from 'a'"),  # Python 3.11 and 3.12
        ],
        ids=["eloop", "eacces", "runtime-loop"],
    )
    def test_a_resolve_failure_is_an_invalid_preset(self, tmp_path, failure):
        """``resolve_within`` raises ``OSError`` or ``RuntimeError`` for a symlink loop
        (and on Windows for a reparse point it cannot follow); that must read as an
        invalid id (a 400), never reach the route as a 500."""
        (tmp_path / "ok.json").write_text("{}", encoding="utf-8")

        with mock.patch(
            "fsm_llm.monitor.instance_manager.examples.resolve_within",
            side_effect=failure,
        ):
            with pytest.raises(ValueError, match="Invalid preset ID"):
                validate_preset_id("ok.json", tmp_path)

    @pytest.mark.parametrize(
        "route", ["/api/preset/fsm/ok.json", "/api/fsm/visualize/preset/ok.json"]
    )
    def test_a_resolve_failure_answers_400_on_the_preset_routes(self, tmp_path, route):
        (tmp_path / "ok.json").write_text("{}", encoding="utf-8")
        client = _client()

        with (
            mock.patch(
                "fsm_llm.monitor.instance_manager.examples.find_examples_dir",
                return_value=tmp_path,
            ),
            mock.patch(
                "fsm_llm.monitor.instance_manager.examples.resolve_within",
                side_effect=OSError(errno.ELOOP, "Too many levels of symbolic links"),
            ),
        ):
            response = client.get(route)

        assert response.status_code == 400
        assert response.json()["detail"] == "invalid preset ID"
