from __future__ import annotations

"""Tests for fsm_llm_monitor.app."""


from fsm_llm_monitor.app import MonitorApp
from fsm_llm_monitor.bridge import MonitorBridge
from fsm_llm_monitor.definitions import MonitorConfig


class TestMonitorApp:
    def test_create_default(self):
        app = MonitorApp()
        assert app.bridge is not None
        assert app.bridge.connected is False

    def test_create_with_bridge(self):
        bridge = MonitorBridge()
        app = MonitorApp(bridge=bridge)
        assert app.bridge is bridge

    def test_create_with_config(self):
        config = MonitorConfig(refresh_interval=0.5)
        app = MonitorApp(config=config)
        assert app.bridge.config.refresh_interval == 0.5

    def test_app_title(self):
        app = MonitorApp()
        assert app.TITLE == "FSM-LLM MONITOR"

    def test_app_has_css(self):
        app = MonitorApp()
        assert app.CSS is not None
        assert "#000000" in app.CSS  # retro black background
        assert "#00ff00" in app.CSS  # retro green

    def test_app_bindings(self):
        app = MonitorApp()
        binding_keys = [b.key for b in app.BINDINGS]
        assert "d" in binding_keys  # dashboard
        assert "f" in binding_keys  # fsm viewer
        assert "c" in binding_keys  # conversations
        assert "a" in binding_keys  # agents
        assert "w" in binding_keys  # workflows
        assert "l" in binding_keys  # logs
        assert "s" in binding_keys  # settings
        assert "q" in binding_keys  # quit


class TestMonitorImports:
    """Verify all public exports import correctly."""

    def test_core_imports(self):
        from fsm_llm_monitor import (
            EventCollector,
            MonitorApp,
            MonitorBridge,
        )

        assert MonitorApp is not None
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

        assert COLOR_PRIMARY == "#00ff00"
        assert THEME_NAME == "retro_green"
        assert DEFAULT_REFRESH_INTERVAL == 1.0

    def test_version(self):
        from fsm_llm_monitor import __version__

        assert __version__ == "0.3.0"
