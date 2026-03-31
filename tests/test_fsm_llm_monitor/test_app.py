from __future__ import annotations

"""Tests for fsm_llm_monitor web server and package."""

from pathlib import Path

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
        assert data["monitor_version"] == "0.3.0"

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

        assert __version__ == "0.3.0"

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
        resp = self.client.post("/api/fsm/launch", json={})
        assert resp.status_code == 500

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
