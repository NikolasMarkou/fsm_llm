from __future__ import annotations

"""Tests for fsm_llm_monitor web server and package."""

from pathlib import Path

from fastapi.testclient import TestClient

from fsm_llm_monitor.bridge import MonitorBridge
from fsm_llm_monitor.server import app, configure


class TestWebServer:
    def setup_method(self):
        configure(MonitorBridge())
        self.client = TestClient(app)

    def test_index_page(self):
        resp = self.client.get("/")
        assert resp.status_code == 200
        assert "FSM-LLM MONITOR" in resp.text
        assert "#3274d9" in resp.text or "style.css" in resp.text

    def test_static_css(self):
        resp = self.client.get("/static/style.css")
        assert resp.status_code == 200
        assert "#3274d9" in resp.text

    def test_static_js(self):
        resp = self.client.get("/static/app.js")
        assert resp.status_code == 200
        assert "connectWS" in resp.text

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

    def test_template_exists(self):
        templates = (
            Path(__file__).parent.parent.parent
            / "src"
            / "fsm_llm_monitor"
            / "templates"
        )
        assert (templates / "index.html").exists()
