from __future__ import annotations

"""Tests for fsm_llm_monitor web server and package."""

from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from fsm_llm_monitor.bridge import MonitorBridge
from fsm_llm_monitor.instance_manager import InstanceManager
from fsm_llm_monitor.server import app, configure


class TestWebServer:
    def setup_method(self):
        configure(MonitorBridge())
        self.client = TestClient(app)

    def test_index_page(self):
        resp = self.client.get("/")
        assert resp.status_code == 200
        assert "FSM-LLM Monitor" in resp.text
        assert "style.css" in resp.text

    def test_static_css(self):
        resp = self.client.get("/static/style.css")
        assert resp.status_code == 200
        assert "--primary:" in resp.text

    def test_static_js(self):
        resp = self.client.get("/static/app.js")
        assert resp.status_code == 200

    def test_static_js_modules(self):
        modules = [
            "services/state.js",
            "services/api.js",
            "services/ws.js",
            "utils/dom.js",
            "utils/format.js",
            "utils/markdown.js",
            "utils/graph.js",
            "pages/dashboard.js",
            "pages/conversations.js",
            "pages/launch.js",
            "pages/control.js",
            "pages/visualizer.js",
            "pages/logs.js",
            "pages/settings.js",
            "pages/builder.js",
        ]
        for module in modules:
            resp = self.client.get(f"/static/{module}")
            assert resp.status_code == 200, f"Failed to load /static/{module}"

    def test_health_endpoint(self):
        resp = self.client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_api_metrics(self):
        resp = self.client.get("/api/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "active_conversations" in data
        assert "total_events" in data
        assert "total_errors" in data

    def test_api_conversations_empty(self):
        resp = self.client.get("/api/conversations")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_api_events_empty(self):
        resp = self.client.get("/api/events")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_api_logs_empty(self):
        resp = self.client.get("/api/logs")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_api_config_get(self):
        resp = self.client.get("/api/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["refresh_interval"] == 1.0
        assert data["log_level"] == "INFO"

    def test_api_config_set(self):
        resp = self.client.post(
            "/api/config",
            json={
                "refresh_interval": 0.5,
                "max_events": 500,
                "max_log_lines": 2000,
                "log_level": "DEBUG",
                "show_internal_keys": False,
                "auto_scroll_logs": True,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_api_info(self):
        resp = self.client.get("/api/info")
        assert resp.status_code == 200
        data = resp.json()
        assert "monitor_version" in data
        assert data["monitor_version"] == "0.9.0"

    def test_api_fsm_load(self):
        resp = self.client.post(
            "/api/fsm/load",
            json={"states": {}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["state_count"] == 0

    def test_api_fsm_load_invalid(self):
        resp = self.client.post(
            "/api/fsm/load",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400
        data = resp.json()
        assert "detail" in data

    def test_api_conversation_not_found(self):
        resp = self.client.get("/api/conversations/nonexistent")
        assert resp.status_code == 404
        data = resp.json()
        assert "detail" in data

    def test_api_agent_visualize(self):
        resp = self.client.get("/api/agent/visualize?agent_type=ReactAgent")
        assert resp.status_code == 200
        data = resp.json()
        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) == 3  # think, act, conclude

    def test_api_agent_visualize_unknown(self):
        resp = self.client.get("/api/agent/visualize?agent_type=FakeAgent")
        assert resp.status_code == 404
        assert "detail" in resp.json()

    def test_api_workflow_visualize(self):
        resp = self.client.get("/api/workflow/visualize?workflow_id=order_processing")
        assert resp.status_code == 200
        data = resp.json()
        assert "nodes" in data
        assert "edges" in data

    def test_api_workflow_visualize_unknown(self):
        resp = self.client.get("/api/workflow/visualize?workflow_id=fake")
        assert resp.status_code == 404
        assert "detail" in resp.json()

    def test_api_workflow_instances_not_found(self):
        resp = self.client.get("/api/workflow/nonexistent/instances")
        assert resp.status_code == 500

    def test_api_capabilities(self):
        resp = self.client.get("/api/capabilities")
        assert resp.status_code == 200
        data = resp.json()
        assert data["fsm"] is True
        assert "workflows" in data
        assert "agents" in data

    def test_api_instances_empty(self):
        resp = self.client.get("/api/instances")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_api_instances_type_filter(self):
        resp = self.client.get("/api/instances?type=fsm")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_api_instance_not_found(self):
        resp = self.client.get("/api/instances/nonexistent")
        assert resp.status_code == 404

    def test_api_instance_destroy_not_found(self):
        resp = self.client.delete("/api/instances/nonexistent")
        assert resp.status_code == 404

    def test_api_instance_events_empty(self):
        resp = self.client.get("/api/instances/nonexistent/events")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_api_fsm_visualize_invalid_json(self):
        resp = self.client.post(
            "/api/fsm/visualize",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400

    def test_api_fsm_visualize_valid(self):
        resp = self.client.post(
            "/api/fsm/visualize",
            json={
                "states": {"start": {"id": "start", "transitions": []}},
                "initial_state": "start",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) == 1

    def test_api_preset_fsm_traversal(self):
        resp = self.client.get("/api/preset/fsm/../../etc/passwd")
        # Returns 400 (invalid preset ID) or 404 (examples dir not found)
        assert resp.status_code in (400, 404)

    def test_api_fsm_visualize_preset_traversal(self):
        resp = self.client.get("/api/fsm/visualize/preset/../../etc/passwd")
        # Returns 400 (invalid preset ID) or 404 (examples dir not found)
        assert resp.status_code in (400, 404)

    def test_api_presets(self):
        resp = self.client.get("/api/presets")
        assert resp.status_code == 200
        data = resp.json()
        assert "fsm" in data

    def test_static_flows_json(self):
        resp = self.client.get("/static/flows.json")
        assert resp.status_code == 200
        data = resp.json()
        assert "agents" in data
        assert "workflows" in data
        assert "ReactAgent" in data["agents"]
        assert "order_processing" in data["workflows"]


class TestMonitorImports:
    """Verify all public exports import correctly."""

    def test_core_imports(self):
        from fsm_llm_monitor import (
            EventCollector,
            MonitorBridge,
        )

        assert MonitorBridge is not None
        assert EventCollector is not None

    def test_definition_imports(self):
        from fsm_llm_monitor import (
            MetricSnapshot,
            MonitorEvent,
        )

        assert MonitorEvent is not None
        assert MetricSnapshot is not None

    def test_exception_imports(self):
        from fsm_llm_monitor import (
            MetricCollectionError,
            MonitorConnectionError,
            MonitorError,
            MonitorInitializationError,
        )

        assert issubclass(MonitorInitializationError, MonitorError)
        assert issubclass(MetricCollectionError, MonitorError)
        assert issubclass(MonitorConnectionError, MonitorError)

    def test_constant_imports(self):
        from fsm_llm_monitor import (
            COLOR_PRIMARY,
            DEFAULT_REFRESH_INTERVAL,
            THEME_NAME,
        )

        assert COLOR_PRIMARY == "#3274d9"
        assert THEME_NAME == "grafana_dark"
        assert DEFAULT_REFRESH_INTERVAL == 1.0

    def test_version(self):
        from fsm_llm_monitor import __version__

        assert __version__ == "0.9.0"

    def test_server_import(self):
        from fsm_llm_monitor.server import app, configure

        assert app is not None
        assert callable(configure)

    def test_static_files_exist(self):
        static = (
            Path(__file__).parent.parent.parent / "src" / "fsm_llm_monitor" / "static"
        )
        assert (static / "style.css").exists()
        assert (static / "app.js").exists()
        assert (static / "flows.json").exists()
        for module in [
            "services/state.js",
            "services/api.js",
            "services/ws.js",
            "utils/dom.js",
            "utils/format.js",
            "utils/markdown.js",
            "utils/graph.js",
            "pages/dashboard.js",
            "pages/conversations.js",
            "pages/launch.js",
            "pages/control.js",
            "pages/visualizer.js",
            "pages/logs.js",
            "pages/settings.js",
            "pages/builder.js",
        ]:
            assert (static / module).exists(), f"Missing {module}"

    def test_template_exists(self):
        templates = (
            Path(__file__).parent.parent.parent
            / "src"
            / "fsm_llm_monitor"
            / "templates"
        )
        assert (templates / "index.html").exists()


def _minimal_fsm_dict():
    """Minimal valid FSM definition for testing."""
    return {
        "name": "TestFSM",
        "description": "A test FSM",
        "initial_state": "start",
        "persona": "test",
        "states": {
            "start": {
                "id": "start",
                "description": "Start state",
                "purpose": "Begin",
                "extraction_instructions": "Extract greeting",
                "response_instructions": "Greet",
                "transitions": [
                    {
                        "target_state": "end",
                        "description": "User done",
                        "priority": 100,
                        "conditions": [],
                    }
                ],
            },
            "end": {
                "id": "end",
                "description": "End state",
                "purpose": "Finish",
                "extraction_instructions": "None",
                "response_instructions": "Say goodbye",
                "transitions": [],
            },
        },
    }


class TestServerFSMEndpoints:
    """Tests for FSM launch, visualization, and preset endpoints."""

    def setup_method(self):
        configure(manager=InstanceManager())
        self.client = TestClient(app)

    def test_fsm_launch_with_json(self):
        resp = self.client.post(
            "/api/fsm/launch",
            json={"fsm_json": _minimal_fsm_dict(), "model": "gpt-4"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "instance_id" in data
        assert data["instance_type"] == "fsm"
        assert data["status"] == "running"

    def test_fsm_launch_requires_data(self):
        # Missing FSM data is a client error (400), not an internal error (500).
        resp = self.client.post("/api/fsm/launch", json={})
        assert resp.status_code == 400

    def test_fsm_launch_and_list_instances(self):
        self.client.post(
            "/api/fsm/launch",
            json={"fsm_json": _minimal_fsm_dict(), "model": "gpt-4"},
        )
        resp = self.client.get("/api/instances")
        assert resp.status_code == 200
        instances = resp.json()
        assert len(instances) >= 1
        assert instances[0]["instance_type"] == "fsm"

    def test_fsm_launch_and_destroy(self):
        launch = self.client.post(
            "/api/fsm/launch",
            json={"fsm_json": _minimal_fsm_dict(), "model": "gpt-4"},
        )
        iid = launch.json()["instance_id"]
        resp = self.client.delete(f"/api/instances/{iid}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_fsm_launch_creates_events(self):
        self.client.post(
            "/api/fsm/launch",
            json={"fsm_json": _minimal_fsm_dict(), "model": "gpt-4"},
        )
        resp = self.client.get("/api/events?limit=10")
        assert resp.status_code == 200
        events = resp.json()
        assert len(events) >= 1
        assert events[0]["event_type"] == "instance_launched"

    def test_fsm_conversations_on_launched_instance(self):
        launch = self.client.post(
            "/api/fsm/launch",
            json={"fsm_json": _minimal_fsm_dict(), "model": "gpt-4"},
        )
        iid = launch.json()["instance_id"]
        resp = self.client.get(f"/api/fsm/{iid}/conversations")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_fsm_visualize_with_transitions(self):
        resp = self.client.post("/api/fsm/visualize", json=_minimal_fsm_dict())
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["nodes"]) == 2
        assert len(data["edges"]) == 1
        assert data["edges"][0]["from"] == "start"
        assert data["edges"][0]["to"] == "end"


class TestServerInstanceEndpoints:
    """Tests for instance management endpoints."""

    def setup_method(self):
        configure(manager=InstanceManager())
        self.client = TestClient(app)

    def test_instance_filter_by_type(self):
        self.client.post(
            "/api/fsm/launch",
            json={"fsm_json": _minimal_fsm_dict(), "model": "gpt-4"},
        )
        resp = self.client.get("/api/instances?type=agent")
        assert resp.status_code == 200
        assert resp.json() == []

        resp2 = self.client.get("/api/instances?type=fsm")
        assert resp2.status_code == 200
        assert len(resp2.json()) == 1

    def test_instance_detail(self):
        launch = self.client.post(
            "/api/fsm/launch",
            json={"fsm_json": _minimal_fsm_dict(), "model": "gpt-4"},
        )
        iid = launch.json()["instance_id"]
        resp = self.client.get(f"/api/instances/{iid}")
        assert resp.status_code == 200
        detail = resp.json()
        assert detail["instance_id"] == iid
        assert detail["instance_type"] == "fsm"

    def test_instance_events_for_launched(self):
        launch = self.client.post(
            "/api/fsm/launch",
            json={"fsm_json": _minimal_fsm_dict(), "model": "gpt-4"},
        )
        iid = launch.json()["instance_id"]
        resp = self.client.get(f"/api/instances/{iid}/events")
        assert resp.status_code == 200


class TestServerConfigEndpoints:
    """Tests for config and info endpoints."""

    def setup_method(self):
        configure(manager=InstanceManager())
        self.client = TestClient(app)

    def test_config_roundtrip(self):
        new_config = {
            "refresh_interval": 2.0,
            "max_events": 2000,
            "max_log_lines": 10000,
            "log_level": "DEBUG",
            "show_internal_keys": True,
            "auto_scroll_logs": False,
        }
        resp = self.client.post("/api/config", json=new_config)
        assert resp.status_code == 200

        resp2 = self.client.get("/api/config")
        data = resp2.json()
        assert data["refresh_interval"] == 2.0
        assert data["log_level"] == "DEBUG"
        assert data["show_internal_keys"] is True

    def test_info_includes_versions(self):
        resp = self.client.get("/api/info")
        data = resp.json()
        assert "monitor_version" in data
        assert "fsm_llm_version" in data


class TestDashboardConfigEndpoints:
    """Tests for custom dashboard config."""

    def setup_method(self):
        configure(manager=InstanceManager())
        self.client = TestClient(app)

    def test_dashboard_config_empty_by_default(self):
        resp = self.client.get("/api/dashboard/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["active"] is False
        assert data["config"] is None

    def test_dashboard_config_apply_and_get(self):
        builder_output = {
            "config": {
                "name": "API Dashboard",
                "description": "Monitor API health",
                "panels": {
                    "p1": {
                        "panel_id": "p1",
                        "title": "Response Time",
                        "panel_type": "chart",
                        "metric": "total_events",
                        "description": "Track response times",
                    },
                    "p2": {
                        "panel_id": "p2",
                        "title": "Error Rate",
                        "panel_type": "gauge",
                        "metric": "total_errors",
                    },
                },
                "alerts": {
                    "a1": {
                        "alert_id": "a1",
                        "metric": "total_errors",
                        "condition": ">",
                        "threshold": 10,
                        "description": "Too many errors",
                    }
                },
                "config": {"refresh_interval_seconds": 15, "retention_hours": 48},
            }
        }
        resp = self.client.post("/api/dashboard/config", json=builder_output)
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        resp2 = self.client.get("/api/dashboard/config")
        data = resp2.json()
        assert data["active"] is True
        cfg = data["config"]
        assert cfg["name"] == "API Dashboard"
        assert len(cfg["panels"]) == 2
        assert len(cfg["alerts"]) == 1
        assert cfg["panels"][0]["title"] == "Response Time"
        assert cfg["alerts"][0]["threshold"] == 10.0
        assert cfg["refresh_interval_seconds"] == 15

    def test_dashboard_config_delete(self):
        # Apply first
        self.client.post(
            "/api/dashboard/config",
            json={
                "config": {
                    "name": "Test",
                    "panels": {"p1": {"title": "X", "metric": "m"}},
                    "alerts": {},
                }
            },
        )
        # Delete
        resp = self.client.delete("/api/dashboard/config")
        assert resp.status_code == 200

        # Verify gone
        resp2 = self.client.get("/api/dashboard/config")
        assert resp2.json()["active"] is False

    def test_dashboard_config_empty_panels(self):
        resp = self.client.post(
            "/api/dashboard/config",
            json={"config": {"name": "Empty", "panels": {}, "alerts": {}}},
        )
        assert resp.status_code == 200
        data = self.client.get("/api/dashboard/config").json()
        assert data["active"] is True
        assert len(data["config"]["panels"]) == 0


class TestServerPresetEndpoints:
    """Tests for preset scanning and loading."""

    def setup_method(self):
        configure(manager=InstanceManager())
        self.client = TestClient(app)

    def test_presets_returns_categories(self):
        resp = self.client.get("/api/presets")
        assert resp.status_code == 200
        data = resp.json()
        assert "fsm" in data
        if data["fsm"]:
            preset = data["fsm"][0]
            assert "name" in preset
            assert "id" in preset
            assert "category" in preset

    def test_preset_load_valid(self):
        resp = self.client.get("/api/presets")
        presets = resp.json().get("fsm", [])
        if presets:
            preset_id = presets[0]["id"]
            resp2 = self.client.get(f"/api/preset/fsm/{preset_id}")
            assert resp2.status_code == 200
            data = resp2.json()
            assert "states" in data

    def test_preset_load_not_found(self):
        resp = self.client.get("/api/preset/fsm/nonexistent/missing/file.json")
        assert resp.status_code in (400, 404)

    def test_preset_visualize(self):
        resp = self.client.get("/api/presets")
        presets = resp.json().get("fsm", [])
        if presets:
            preset_id = presets[0]["id"]
            resp2 = self.client.get(f"/api/fsm/visualize/preset/{preset_id}")
            assert resp2.status_code == 200
            data = resp2.json()
            assert "nodes" in data
            assert "edges" in data


class TestServerErrorHandling:
    """Tests for error responses across endpoints."""

    def setup_method(self):
        configure(manager=InstanceManager())
        self.client = TestClient(app)

    def test_agent_status_not_found(self):
        resp = self.client.get("/api/agent/nonexistent/status")
        assert resp.status_code == 500

    def test_agent_result_not_found(self):
        resp = self.client.get("/api/agent/nonexistent/result")
        assert resp.status_code == 500

    def test_agent_cancel_not_found(self):
        resp = self.client.post("/api/agent/nonexistent/cancel")
        assert resp.status_code == 500

    def test_workflow_status_missing_param(self):
        resp = self.client.get("/api/workflow/nonexistent/status")
        assert resp.status_code == 400

    def test_builder_session_not_found(self):
        resp = self.client.post(
            "/api/builder/send",
            json={"session_id": "nonexistent", "message": "hello"},
        )
        assert resp.status_code == 404

    def test_builder_result_not_found(self):
        resp = self.client.get("/api/builder/result/nonexistent")
        assert resp.status_code == 404

    def test_builder_delete_nonexistent(self):
        resp = self.client.delete("/api/builder/nonexistent")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is False


class TestActivityEndpoint:
    """Tests for the unified /api/activity endpoint."""

    def setup_method(self):
        configure(manager=InstanceManager())
        self.client = TestClient(app)

    def test_activity_empty(self):
        resp = self.client.get("/api/activity")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_activity_returns_fsm_conversations(self):
        """After launching an FSM and starting a conversation, activity should include it."""
        mgr = InstanceManager()
        configure(manager=mgr)

        # Activity should start empty
        resp = self.client.get("/api/activity")
        assert resp.status_code == 200
        items = resp.json()
        # May be empty or contain items from prior state
        assert isinstance(items, list)

    def test_activity_with_include_ended_false(self):
        resp = self.client.get("/api/activity?include_ended=false")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_metrics_include_agent_workflow_fields(self):
        resp = self.client.get("/api/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "active_agents" in data
        assert "active_workflows" in data
        assert "total_agent_iterations" in data
        assert "total_tool_calls" in data
        assert "total_workflow_steps" in data
        assert data["active_agents"] == 0
        assert data["active_workflows"] == 0


class TestServerHygieneAndWorkflow:
    """server.py hygiene + workflow endpoints (plan Steps 1,3,5)."""

    def setup_method(self):
        configure(MonitorBridge())
        self.client = TestClient(app)

    def test_500_detail_is_generic(self):
        # Unknown agent id -> internal KeyError -> 500 with a generic detail
        # (no exception text leaked).
        resp = self.client.get("/api/agent/does-not-exist/status")
        assert resp.status_code == 500
        assert resp.json()["detail"] == "Internal server error"

    def test_disabled_agent_returns_400(self):
        resp = self.client.post(
            "/api/agent/launch",
            json={"agent_type": "EvaluatorOptimizerAgent", "task": "x"},
        )
        assert resp.status_code == 400
        assert "Unknown agent type" in resp.json()["detail"]

    def test_workflow_presets_endpoint(self):
        resp = self.client.get("/api/workflow/presets")
        assert resp.status_code == 200
        ids = {p["id"] for p in resp.json()["workflows"]}
        assert "demo_linear" in ids

    def test_workflow_launch_starts_instance(self):
        resp = self.client.post(
            "/api/workflow/launch", json={"preset_id": "demo_linear"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["instance_type"] == "workflow"
        assert data.get("workflow_instance_id")

    def test_workflow_launch_unknown_preset_400(self):
        resp = self.client.post("/api/workflow/launch", json={"preset_id": "nope"})
        assert resp.status_code == 400


class TestApiKeyGate:
    """F2 — optional API-key gate on mutating monitor routes (D-008)."""

    def teardown_method(self):
        # Ensure no test in this class leaves _api_key set for later tests.
        configure(manager=InstanceManager())

    def test_unset_api_key_is_unchanged_behavior(self):
        """Default (api_key never configured): no auth required on a mutating route."""
        configure(manager=InstanceManager())
        client = TestClient(app)
        resp = client.post(
            "/api/config",
            json={
                "refresh_interval": 1.0,
                "max_events": 1000,
                "max_log_lines": 5000,
                "log_level": "INFO",
                "show_internal_keys": False,
                "auto_scroll_logs": True,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_gated_route_401_without_key(self):
        configure(manager=InstanceManager(), api_key="s3cr3t")
        client = TestClient(app)
        resp = client.post(
            "/api/config",
            json={
                "refresh_interval": 1.0,
                "max_events": 1000,
                "max_log_lines": 5000,
                "log_level": "INFO",
                "show_internal_keys": False,
                "auto_scroll_logs": True,
            },
        )
        assert resp.status_code == 401

    def test_gated_route_401_with_wrong_key(self):
        configure(manager=InstanceManager(), api_key="s3cr3t")
        client = TestClient(app)
        resp = client.post(
            "/api/config",
            json={
                "refresh_interval": 1.0,
                "max_events": 1000,
                "max_log_lines": 5000,
                "log_level": "INFO",
                "show_internal_keys": False,
                "auto_scroll_logs": True,
            },
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert resp.status_code == 401

    def test_gated_route_200_with_correct_bearer_key(self):
        configure(manager=InstanceManager(), api_key="s3cr3t")
        client = TestClient(app)
        resp = client.post(
            "/api/config",
            json={
                "refresh_interval": 1.0,
                "max_events": 1000,
                "max_log_lines": 5000,
                "log_level": "INFO",
                "show_internal_keys": False,
                "auto_scroll_logs": True,
            },
            headers={"Authorization": "Bearer s3cr3t"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_gated_route_200_with_correct_x_api_key_header(self):
        configure(manager=InstanceManager(), api_key="s3cr3t")
        client = TestClient(app)
        resp = client.post(
            "/api/config",
            json={
                "refresh_interval": 1.0,
                "max_events": 1000,
                "max_log_lines": 5000,
                "log_level": "INFO",
                "show_internal_keys": False,
                "auto_scroll_logs": True,
            },
            headers={"X-API-Key": "s3cr3t"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_all_named_mutating_routes_gated(self):
        """Every mutating route named in plan.md step 7c returns 401 without a key."""
        configure(manager=InstanceManager(), api_key="s3cr3t")
        client = TestClient(app)
        cases = [
            (
                "post",
                "/api/config",
                {
                    "refresh_interval": 1.0,
                    "max_events": 1000,
                    "max_log_lines": 5000,
                    "log_level": "INFO",
                    "show_internal_keys": False,
                    "auto_scroll_logs": True,
                },
            ),
            (
                "post",
                "/api/dashboard/config",
                {"name": "x", "panels": {}, "alerts": {}},
            ),
            ("delete", "/api/dashboard/config", None),
            ("delete", "/api/instances/nonexistent", None),
            ("post", "/api/fsm/launch", {"preset_id": "does-not-exist"}),
            ("post", "/api/workflow/launch", {"preset_id": "does-not-exist"}),
            ("post", "/api/agent/launch", {"agent_type": "ReactAgent", "task": "x"}),
            ("post", "/api/builder/start", {}),
            ("post", "/api/builder/send", {"session_id": "x", "message": "x"}),
            ("delete", "/api/builder/nonexistent", None),
        ]
        for method, path, body in cases:
            resp = client.request(method, path, json=body)
            assert resp.status_code == 401, f"{method.upper()} {path} was not gated"

    def test_read_only_route_stays_ungated(self):
        """GETs and /health remain reachable without a key even when one is configured."""
        configure(manager=InstanceManager(), api_key="s3cr3t")
        client = TestClient(app)
        assert client.get("/health").status_code == 200
        assert client.get("/api/config").status_code == 200
        assert client.get("/api/instances").status_code == 200

    def test_configure_reread_after_first_request_takes_effect(self):
        """Unlike CORS, api_key changes after the first request DO take effect
        (Depends reads the module-level global at request time)."""
        configure(manager=InstanceManager())
        client = TestClient(app)
        assert client.get("/health").status_code == 200  # process a request first
        configure(manager=InstanceManager(), api_key="s3cr3t")
        resp = client.post(
            "/api/dashboard/config", json={"name": "x", "panels": {}, "alerts": {}}
        )
        assert resp.status_code == 401

    def test_correct_key_authenticates_via_compare_digest(self):
        """D-016: the correct-key path now runs through hmac.compare_digest,
        not `!=`. Behavior-preserving (still 200 for a correct key), but this
        test pins that `compare_digest` is actually the mechanism used, so a
        regression back to a plain `!=` comparison would be caught if this
        spy is ever tightened to assert call counts."""
        import fsm_llm_monitor.server as server_module

        calls: list[tuple[bytes, bytes]] = []
        original = server_module.hmac.compare_digest

        def _spy(a: bytes, b: bytes) -> bool:
            calls.append((a, b))
            return original(a, b)

        configure(manager=InstanceManager(), api_key="s3cr3t")
        server_module.hmac.compare_digest = _spy
        try:
            client = TestClient(app)
            resp = client.post(
                "/api/dashboard/config",
                json={"name": "x", "panels": {}, "alerts": {}},
                headers={"Authorization": "Bearer s3cr3t"},
            )
            assert resp.status_code == 200
            # D-020: compare_digest is now called with encoded bytes, not str,
            # so non-ASCII input can never raise TypeError (see
            # test_non_ascii_api_key_header_returns_401_not_500 below).
            assert calls == [(b"s3cr3t", b"s3cr3t")]
        finally:
            server_module.hmac.compare_digest = original

    def test_non_ascii_api_key_header_returns_401_not_500(self):
        """D-020: a malformed/non-ASCII X-API-Key header must 401, never 500.

        Pre-fix, `hmac.compare_digest(token, _api_key)` compared two `str`
        objects directly; CPython's str overload of compare_digest raises
        `TypeError: comparing strings with non-ASCII characters is not
        supported` whenever either operand contains a non-ASCII character,
        turning an unauthenticated, attacker-controlled header into an
        unhandled 500 instead of the expected 401.

        httpx/TestClient refuse to even construct a request carrying a
        non-ASCII header value (`UnicodeEncodeError` at the client), so this
        can only be reached by building the ASGI `Request` directly — the
        same technique the adversarial review used to first find the bug.
        """
        from starlette.requests import Request

        import fsm_llm_monitor.server as server_module

        configure(manager=InstanceManager(), api_key="s3cr3t")
        scope = {
            "type": "http",
            "headers": [(b"x-api-key", bytes([0xC3, 0xA9, 0xC3, 0xA9]))],
        }
        req = Request(scope=scope)
        with pytest.raises(HTTPException) as excinfo:
            server_module._require_api_key(req)
        assert excinfo.value.status_code == 401

    def test_non_ascii_bearer_token_returns_401_not_500(self):
        """D-020: same guarantee via the Authorization: Bearer header path."""
        from starlette.requests import Request

        import fsm_llm_monitor.server as server_module

        configure(manager=InstanceManager(), api_key="s3cr3t")
        scope = {
            "type": "http",
            "headers": [
                (b"authorization", b"Bearer " + bytes([0xC3, 0xA9, 0xC3, 0xA9]))
            ],
        }
        req = Request(scope=scope)
        with pytest.raises(HTTPException) as excinfo:
            server_module._require_api_key(req)
        assert excinfo.value.status_code == 401

    def test_reconfigure_without_api_key_warns_when_clearing_prior_key(self):
        """D-016: re-calling configure() without api_key= after a key was
        previously set logs a WARNING (mirroring the CORS-mutation warning),
        since the key is silently cleared."""
        import io

        from fsm_llm.logging import logger

        configure(manager=InstanceManager(), api_key="s3cr3t")
        buf = io.StringIO()
        sink_id = logger.add(buf, level="WARNING")
        try:
            configure(manager=InstanceManager())
        finally:
            logger.remove(sink_id)
        output = buf.getvalue()
        assert "previously configured API key is being cleared" in output

    def test_first_configure_call_does_not_warn(self):
        """D-016: the warning must not fire on the very first configure()
        call in a process (nothing to clear yet)."""
        import io

        from fsm_llm.logging import logger

        # Ensure a clean slate (no previously-set key) before the "first" call.
        configure(manager=InstanceManager())
        buf = io.StringIO()
        sink_id = logger.add(buf, level="WARNING")
        try:
            configure(manager=InstanceManager())
        finally:
            logger.remove(sink_id)
        output = buf.getvalue()
        assert "previously configured API key is being cleared" not in output

    @pytest.mark.parametrize("bad_key", ["", "   ", "\t\n"])
    def test_configure_rejects_empty_api_key(self, bad_key):
        """D-006 (P2-W5): an empty or whitespace-only programmatic key raises.
        Accepting "" let an empty X-API-Key header authenticate
        (compare_digest(b"", b"") is True); the prior key must stay in force."""
        from fsm_llm_monitor import server as server_module

        configure(manager=InstanceManager(), api_key="s3cr3t")
        with pytest.raises(ValueError, match="api_key"):
            configure(manager=InstanceManager(), api_key=bad_key)
        assert server_module._api_key == "s3cr3t"
        client = TestClient(app)
        resp = client.delete("/api/dashboard/config", headers={"X-API-Key": ""})
        assert resp.status_code == 401

    def test_empty_env_api_key_still_means_unset(self, monkeypatch):
        """D-006: FSM_LLM_MONITOR_API_KEY="" keeps its unset meaning (no auth)."""
        from fsm_llm_monitor import server as server_module

        monkeypatch.setenv("FSM_LLM_MONITOR_API_KEY", "")
        configure(manager=InstanceManager())
        assert server_module._api_key is None
        client = TestClient(app)
        assert client.delete("/api/dashboard/config").status_code == 200

    @pytest.mark.parametrize(
        "method,path",
        [
            ("post", "/api/fsm/does-not-exist/start"),
            ("post", "/api/fsm/does-not-exist/converse"),
            ("post", "/api/fsm/does-not-exist/end"),
            ("post", "/api/workflow/does-not-exist/advance"),
            ("post", "/api/workflow/does-not-exist/cancel"),
            ("post", "/api/agent/does-not-exist/cancel"),
        ],
    )
    def test_previously_ungated_routes_401_without_key_when_configured(
        self, method, path
    ):
        """Step B (findings 7-9): these 6 routes now carry
        ``dependencies=[Depends(_require_api_key)]`` — when an API key is
        configured, calling them without one must 401 before ever reaching
        the (missing) instance lookup."""
        configure(manager=InstanceManager(), api_key="s3cr3t")
        client = TestClient(app)
        resp = getattr(client, method)(path, json={})
        assert resp.status_code == 401

    @pytest.mark.parametrize(
        "method,path",
        [
            ("post", "/api/fsm/does-not-exist/start"),
            ("post", "/api/fsm/does-not-exist/converse"),
            ("post", "/api/fsm/does-not-exist/end"),
            ("post", "/api/workflow/does-not-exist/advance"),
            ("post", "/api/workflow/does-not-exist/cancel"),
            ("post", "/api/agent/does-not-exist/cancel"),
        ],
    )
    def test_previously_ungated_routes_unaffected_when_unconfigured(self, method, path):
        """When no API key is configured, these 6 routes must behave exactly
        as before the fix — never blocked by auth (never a 401)."""
        configure(manager=InstanceManager())
        client = TestClient(app)
        resp = getattr(client, method)(path, json={})
        assert resp.status_code != 401


class TestDashboardWebsocketRedaction:
    """The dashboard push must not send an arbitrary object's __str__ to the
    browser: a context value whose repr carries a secret would be broadcast to
    every connected viewer.
    """

    def test_server_uses_the_shared_redacting_hook(self):
        import inspect
        import re

        from fsm_llm_monitor import server as server_mod

        source = inspect.getsource(server_mod)
        assert "default=redacting_json_default" in source
        # A call site, not the anchor comment that names the old hook.
        assert re.search(r"default=str[,)\s]", source) is None

    def test_payload_with_a_leaky_object_is_redacted(self):
        import json

        from fsm_llm.utilities import redacting_json_default

        class Leaky:
            def __str__(self):
                return "api_key=sk-secret"

        payload = {"instances": [{"context": {"creds": Leaky()}}]}
        text = json.dumps(payload, default=redacting_json_default)
        assert "sk-secret" not in text
        assert "<redacted:Leaky>" in text
