"""
Server-level regression tests for the 2026-09-28 fsm_llm.monitor audit:
origin/host checks, API-key gating of sensitive reads and the WebSocket,
error mapping, request bounds, dashboard config parsing and builder guards.
"""

from __future__ import annotations

import errno
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from fsm_llm.core.definitions import ConversationBusyError
from fsm_llm.monitor.collector import EventCollector
from fsm_llm.monitor.constants import MAX_QUERY_LIMIT
from fsm_llm.monitor.definitions import MonitorConfig, MonitorEvent
from fsm_llm.monitor.exceptions import MonitorCapacityError
from fsm_llm.monitor.instance_manager import InstanceManager, validate_preset_id
from fsm_llm.monitor.instance_manager.managed import ManagedAgent, ManagedFSM
from fsm_llm.monitor.server import (
    app,
    configure,
    routes_builder,
    routes_monitoring,
)


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
            "/api/instances",
            "/api/instances/x",
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

    def test_an_end_that_loses_the_claim_is_a_json_409(self):
        # D-005 of plan 9283c1fd: the loser of two concurrent ends is a 409, not a 500
        mgr = InstanceManager(config=MonitorConfig(refresh_interval=0.5))
        inst = ManagedFSM("f1", mock.MagicMock())
        inst.conversation_ids.append("c1")
        inst.ending_conversation_ids.add("c1")  # another request holds the claim
        mgr._instances["f1"] = inst
        configure(manager=mgr)

        resp = TestClient(app).post("/api/fsm/f1/end", json={"conversation_id": "c1"})

        assert resp.status_code == 409
        assert resp.json() == {"detail": "conversation is busy with another turn"}
        inst.api.end_conversation.assert_not_called()

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


class TestDashboardConfigFalsyNonObjects:
    """Step 1, G1: ``panels``, ``alerts`` and ``config`` default only when the
    key is absent or null; ``[]``, ``""``, ``false`` and ``0`` are 400 and the
    applied dashboard (and its version) stay."""

    def _applied(self, client):
        assert (
            client.post("/api/dashboard/config", json=_GOOD_DASHBOARD).status_code
            == 200
        )
        return client.get("/api/dashboard/config").json()

    @pytest.mark.parametrize("key", ["panels", "alerts", "config"])
    @pytest.mark.parametrize(
        "value", [[], "", False, 0], ids=["list", "str", "false", "zero"]
    )
    def test_a_falsy_non_object_is_400_and_the_config_stays(self, key, value):
        client = _client()
        before = self._applied(client)
        version = routes_monitoring.get_manager().dashboard_config_version

        resp = client.post("/api/dashboard/config", json={key: value})

        assert resp.status_code == 400
        assert client.get("/api/dashboard/config").json() == before
        assert routes_monitoring.get_manager().dashboard_config_version == version

    @pytest.mark.parametrize(
        "body",
        [{}, {"panels": None}, {"alerts": None, "config": None}, {"name": "N"}],
    )
    def test_null_and_absent_keys_default(self, body):
        client = _client()
        self._applied(client)

        assert client.post("/api/dashboard/config", json=body).status_code == 200

        cfg = client.get("/api/dashboard/config").json()["config"]
        assert cfg["panels"] == [] and cfg["alerts"] == []
        assert cfg["refresh_interval_seconds"] == 30 and cfg["retention_hours"] == 24

    def test_the_get_answer_of_a_dashboard_with_no_panels_posts_back_as_400(self):
        client = _client()
        assert (
            client.post("/api/dashboard/config", json={"name": "Empty"}).status_code
            == 200
        )
        before = client.get("/api/dashboard/config").json()
        assert before["config"]["panels"] == []

        for body in (before, before["config"], {"config": before["config"]}):
            assert client.post("/api/dashboard/config", json=body).status_code == 400
        assert client.get("/api/dashboard/config").json() == before


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


_TASK = "MY PRIVATE TASK: wire 5000 to account 12345"


def _agent_manager() -> InstanceManager:
    """A manager that holds one agent (with its collector) whose task and label are private."""
    mgr = InstanceManager(config=MonitorConfig(refresh_interval=0.5))
    agent = ManagedAgent(
        instance_id="ag1", agent_type="react", task=_TASK, label="private label"
    )
    with mgr._lock:
        mgr._instances["ag1"] = agent
        mgr._collectors["ag1"] = EventCollector()
    return mgr


def _client_with_agent(api_key: str | None = None) -> TestClient:
    configure(manager=_agent_manager(), api_key=api_key)
    return TestClient(app)


class TestInstanceReadsNeedTheKey:
    """The instance list and detail show an agent's
    ``task`` and ``label``, so they take the key like every other read that shows
    user text; a server with no key configured is unchanged."""

    _PATHS = ("/api/instances", "/api/instances/ag1")

    @pytest.mark.parametrize("path", _PATHS)
    @pytest.mark.parametrize(
        "headers", [{}, {"X-API-Key": "wrong"}, {"Authorization": "Bearer wrong"}]
    )
    def test_no_key_or_a_wrong_key_is_401_and_leaks_nothing(self, path, headers):
        resp = _client_with_agent("s3cr3t").get(path, headers=headers)
        assert resp.status_code == 401
        assert _TASK not in resp.text and "private label" not in resp.text

    @pytest.mark.parametrize("path", _PATHS)
    @pytest.mark.parametrize(
        "headers", [{"X-API-Key": "s3cr3t"}, {"Authorization": "Bearer s3cr3t"}]
    )
    def test_the_key_in_either_header_reads(self, path, headers):
        resp = _client_with_agent("s3cr3t").get(path, headers=headers)
        assert resp.status_code == 200
        assert _TASK in resp.text

    @pytest.mark.parametrize("path", _PATHS)
    def test_a_server_with_no_key_is_open_as_before(self, path):
        resp = _client_with_agent().get(path)
        assert resp.status_code == 200

    def test_health_and_config_stay_open_with_a_key_set(self):
        client = _client_with_agent("s3cr3t")
        assert client.get("/health").status_code == 200
        assert client.get("/api/config").status_code == 200


class TestListLimitsAreBounded:
    """``limit`` of the three list routes is 1 to ``MAX_QUERY_LIMIT``; the event log
    route keeps its own clamp (``limit=0`` is 200 there, ``test_event_log_browse.py``)."""

    _ROUTES = ("/api/events", "/api/logs", "/api/instances/ag1/events")

    def test_the_cap_is_one_named_constant_of_a_thousand(self):
        assert MAX_QUERY_LIMIT == 1000

    @pytest.mark.parametrize("route", _ROUTES)
    @pytest.mark.parametrize("limit", ["0", "-1", "1001", "1000000000", "abc"])
    def test_out_of_range_is_422(self, route, limit):
        resp = _client_with_agent().get(route, params={"limit": limit})
        assert resp.status_code == 422

    @pytest.mark.parametrize("route", _ROUTES)
    @pytest.mark.parametrize("limit", ["1", "50", str(MAX_QUERY_LIMIT)])
    def test_in_range_is_200(self, route, limit):
        resp = _client_with_agent().get(route, params={"limit": limit})
        assert resp.status_code == 200

    @pytest.mark.parametrize("route", _ROUTES)
    def test_the_default_is_unchanged(self, route):
        assert _client_with_agent().get(route).status_code == 200

    def test_a_limit_returns_at_most_that_many_rows(self):
        mgr = _agent_manager()
        configure(manager=mgr)
        client = TestClient(app)
        for i in range(60):
            mgr.global_collector.record_event(
                MonitorEvent(event_type="x", message=str(i))
            )
        assert len(client.get("/api/events", params={"limit": 7}).json()) == 7
        assert len(client.get("/api/events", params={"limit": 1000}).json()) == 60

    def test_the_three_routes_read_the_one_constant(self):
        """The bound is imported, never repeated: a literal beside the constant drifts."""
        src = Path(routes_monitoring.__file__).read_text(encoding="utf-8")
        assert src.count("le=MAX_QUERY_LIMIT") == 3
        assert not re.search(r"\ble=\d", src)

    def test_the_422_body_echoes_nothing_but_the_number(self):
        body = _client_with_agent().get("/api/events", params={"limit": "1001"}).text
        assert "1001" in body and "Traceback" not in body


class TestAPresetIsAJsonFile:
    """A directory, an empty id and a file that is not
    JSON are no preset: ``FileNotFoundError`` and so 404 on every route, never the 500
    of reading a directory."""

    # the last one is longer than a file name may be (the filesystem answers ENAMETOOLONG)
    _NOT_PRESETS = (
        "basic",
        "basic/",
        "",
        "basic/event_registration/run.py",
        "a" * 300 + ".json",
    )

    def test_unit_rows_on_a_tmp_directory(self, tmp_path):
        (tmp_path / "ok.json").write_text("{}", encoding="utf-8")
        (tmp_path / "note.txt").write_text("x", encoding="utf-8")
        (tmp_path / "dir.json").mkdir()
        (tmp_path / "sub").mkdir()
        assert validate_preset_id("ok.json", tmp_path) == tmp_path / "ok.json"
        for bad in ("sub", "sub/", "", "note.txt", "dir.json", "a" * 300 + ".json"):
            with pytest.raises(FileNotFoundError):
                validate_preset_id(bad, tmp_path)
        for escaping in ("../x.json", "/etc/passwd", "a/../b.json"):
            with pytest.raises(ValueError):
                validate_preset_id(escaping, tmp_path)

    @pytest.mark.parametrize("preset_id", _NOT_PRESETS)
    def test_the_read_route_is_404(self, preset_id):
        resp = _client().get(f"/api/preset/fsm/{preset_id}")
        assert resp.status_code == 404
        assert resp.json() == {"detail": "preset not found"}

    @pytest.mark.parametrize("preset_id", [i for i in _NOT_PRESETS if i])
    def test_the_visualize_route_is_404_and_names_no_path(self, preset_id):
        resp = _client().get(f"/api/fsm/visualize/preset/{preset_id}")
        assert resp.status_code == 404
        assert resp.json() == {"detail": "preset not found"}

    @pytest.mark.parametrize("preset_id", _NOT_PRESETS)
    def test_the_launch_route_is_404(self, preset_id):
        resp = _client().post("/api/fsm/launch", json={"preset_id": preset_id})
        assert resp.status_code == 404
        assert resp.json() == {"detail": "not found"}

    def test_a_real_preset_still_launches_reads_and_draws(self):
        client = _client()
        assert (
            client.get("/api/preset/fsm/basic/form_filling/fsm.json").status_code == 200
        )
        assert (
            client.get(
                "/api/fsm/visualize/preset/basic/form_filling/fsm.json"
            ).status_code
            == 200
        )
        listed = client.get("/api/presets").json()["fsm"]
        assert listed and all(
            client.get(f"/api/preset/fsm/{p['id']}").status_code == 200
            for p in listed[:5]
        )


class TestUnknownIdsEchoNothing:
    def test_the_unknown_workflow_text_has_no_id(self):
        resp = _client().get(
            "/api/workflow/visualize", params={"workflow_id": "<b>probe</b>"}
        )
        assert resp.status_code == 404
        assert resp.json() == {"detail": "unknown workflow"}
        assert "probe" not in resp.text


class TestActivitySurvivesADestroyedAgent:
    """An agent ``end_instance`` removed while the list was built is skipped."""

    def _manager_with_two_agents(self) -> InstanceManager:
        mgr = InstanceManager(config=MonitorConfig(refresh_interval=0.5))
        with mgr._lock:
            for name in ("gone", "kept"):
                mgr._instances[name] = ManagedAgent(
                    instance_id=name, agent_type="react", task="t", label=name
                )
        return mgr

    def test_the_manager_skips_the_destroyed_agent(self):
        mgr = self._manager_with_two_agents()
        real = mgr.get_agent_status

        def status(instance_id):
            if instance_id == "gone":
                raise KeyError(instance_id)
            return real(instance_id)

        with mock.patch.object(mgr, "get_agent_status", side_effect=status):
            items = mgr.get_all_activity_snapshots()
        assert [i.item_id for i in items if i.item_type == "agent_task"] == ["kept"]

    def test_the_route_answers_200_without_that_item(self):
        mgr = self._manager_with_two_agents()
        configure(manager=mgr)
        real = mgr.get_agent_status

        def status(instance_id):
            if instance_id == "gone":
                raise KeyError(instance_id)
            return real(instance_id)

        with mock.patch.object(mgr, "get_agent_status", side_effect=status):
            resp = TestClient(app).get("/api/activity")
        assert resp.status_code == 200
        assert [i["item_id"] for i in resp.json()] == ["kept"]
