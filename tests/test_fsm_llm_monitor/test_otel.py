from __future__ import annotations

"""Tests for fsm_llm_monitor.otel module."""

import sys
import threading
from types import ModuleType
from unittest.mock import MagicMock

import pytest

# --- OTEL mock setup ---
# opentelemetry is an optional dependency. We mock the entire package tree
# so tests run without it installed.


def _build_otel_mocks():
    """Build mock modules for the opentelemetry package tree."""
    trace_mod = ModuleType("opentelemetry.trace")
    trace_mod.StatusCode = MagicMock()
    trace_mod.StatusCode.ERROR = "ERROR"
    trace_mod.set_tracer_provider = MagicMock()
    trace_mod.get_tracer = MagicMock()
    trace_mod.set_span_in_context = MagicMock(return_value=MagicMock())

    sdk_trace_mod = ModuleType("opentelemetry.sdk.trace")
    sdk_trace_mod.TracerProvider = MagicMock

    export_mod = ModuleType("opentelemetry.sdk.trace.export")
    export_mod.BatchSpanProcessor = MagicMock()
    export_mod.ConsoleSpanExporter = MagicMock()

    return {
        "opentelemetry": ModuleType("opentelemetry"),
        "opentelemetry.trace": trace_mod,
        "opentelemetry.sdk": ModuleType("opentelemetry.sdk"),
        "opentelemetry.sdk.trace": sdk_trace_mod,
        "opentelemetry.sdk.trace.export": export_mod,
    }


@pytest.fixture(autouse=True)
def _mock_otel():
    """Inject mock opentelemetry modules before every test."""
    mocks = _build_otel_mocks()
    saved = {}
    for name, mod in mocks.items():
        saved[name] = sys.modules.get(name)
        sys.modules[name] = mod

    # Force reimport of otel module so it picks up the mocked packages
    mod_key = "fsm_llm_monitor.otel"
    saved_otel = sys.modules.pop(mod_key, None)

    yield mocks

    # Restore original state
    sys.modules.pop(mod_key, None)
    if saved_otel is not None:
        sys.modules[mod_key] = saved_otel
    for name, original in saved.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


def _import_otel():
    """Import the otel module (must be called after fixture injects mocks)."""
    from fsm_llm_monitor.otel import OTELExporter

    return OTELExporter


def _make_event(event_type="conversation_start", conv_id="conv-1", **kwargs):
    from fsm_llm_monitor.definitions import MonitorEvent

    return MonitorEvent(
        event_type=event_type,
        conversation_id=conv_id,
        **kwargs,
    )


# =============================================================================
# Tests
# =============================================================================


class TestOTELExporterInit:
    def test_init_creates_exporter(self):
        cls = _import_otel()
        exporter = cls(service_name="test-service")
        assert exporter.is_enabled is False
        assert exporter.active_conversations == []

    def test_init_with_custom_exporter(self):
        cls = _import_otel()
        custom_exp = MagicMock()
        exporter = cls(service_name="test", exporter=custom_exp)
        assert exporter is not None

    def test_generate_trace_id(self):
        cls = _import_otel()
        tid = cls.generate_trace_id()
        assert isinstance(tid, str)
        assert len(tid) == 32  # uuid4 hex

    def test_generate_span_id(self):
        cls = _import_otel()
        sid = cls.generate_span_id()
        assert isinstance(sid, str)
        assert len(sid) == 16  # uuid4 hex[:16]


class TestOTELExporterEnable:
    def test_enable_wraps_record_event(self):
        cls = _import_otel()
        exporter = cls()
        collector = MagicMock()
        original_fn = collector.record_event

        exporter.enable(collector)

        assert exporter.is_enabled is True
        # record_event should have been replaced
        assert collector.record_event is not original_fn

    def test_enable_idempotent_on_same_collector(self):
        cls = _import_otel()
        exporter = cls()
        collector = MagicMock()

        exporter.enable(collector)
        wrapped_1 = collector.record_event
        exporter.enable(collector)
        wrapped_2 = collector.record_event

        # Should not double-wrap
        assert wrapped_1 is wrapped_2

    def test_enable_on_different_collector_disables_first(self):
        cls = _import_otel()
        exporter = cls()
        collector1 = MagicMock()
        collector2 = MagicMock()
        original1 = collector1.record_event

        exporter.enable(collector1)
        assert exporter.is_enabled is True

        exporter.enable(collector2)
        # First collector should have original restored
        assert collector1.record_event is original1
        assert exporter.is_enabled is True


class TestOTELExporterDisable:
    def test_disable_restores_original(self):
        cls = _import_otel()
        exporter = cls()
        collector = MagicMock()
        original_fn = collector.record_event

        exporter.enable(collector)
        assert collector.record_event is not original_fn

        exporter.disable()
        assert collector.record_event is original_fn
        assert exporter.is_enabled is False

    def test_disable_ends_active_spans(self):
        cls = _import_otel()
        exporter = cls()
        collector = MagicMock()
        exporter.enable(collector)

        # Simulate a conversation start to create a span
        mock_span = MagicMock()
        exporter._conversation_spans["conv-1"] = mock_span

        exporter.disable()
        mock_span.end.assert_called_once()
        assert exporter.active_conversations == []

    def test_disable_without_enable_is_safe(self):
        cls = _import_otel()
        exporter = cls()
        exporter.disable()  # Should not raise
        assert exporter.is_enabled is False


class TestOTELExporterShutdown:
    def test_shutdown_disables_and_flushes(self):
        cls = _import_otel()
        exporter = cls()
        collector = MagicMock()
        exporter.enable(collector)
        exporter.shutdown()
        assert exporter.is_enabled is False


class TestOTELEventRouting:
    def _make_exporter_with_collector(self):
        cls = _import_otel()
        exporter = cls()
        collector = MagicMock()
        exporter.enable(collector)
        return exporter, collector

    def test_conversation_start_creates_span(self):
        exporter, collector = self._make_exporter_with_collector()
        event = _make_event("conversation_start", "conv-1")
        exporter._route_event(event)
        assert "conv-1" in exporter.active_conversations

    def test_conversation_end_closes_span(self):
        exporter, collector = self._make_exporter_with_collector()
        mock_span = MagicMock()
        exporter._conversation_spans["conv-1"] = mock_span

        event = _make_event("conversation_end", "conv-1")
        exporter._route_event(event)

        mock_span.end.assert_called_once()
        assert "conv-1" not in exporter.active_conversations

    def test_state_transition_event(self):
        exporter, collector = self._make_exporter_with_collector()
        event = _make_event(
            "state_transition",
            "conv-1",
            source_state="start",
            target_state="end",
        )
        # Should not raise
        exporter._route_event(event)

    def test_processing_event(self):
        exporter, collector = self._make_exporter_with_collector()
        event = _make_event("pre_processing", "conv-1")
        exporter._route_event(event)
        event2 = _make_event("post_processing", "conv-1")
        exporter._route_event(event2)

    def test_error_event_sets_span_status(self):
        exporter, collector = self._make_exporter_with_collector()
        mock_span = MagicMock()
        exporter._conversation_spans["conv-1"] = mock_span

        event = _make_event("error", "conv-1", message="test error")
        exporter._route_event(event)

        mock_span.set_status.assert_called_once()

    def test_lifecycle_events(self):
        exporter, collector = self._make_exporter_with_collector()
        for evt_type in [
            "agent_started",
            "agent_completed",
            "agent_failed",
            "workflow_started",
            "workflow_completed",
            "workflow_cancelled",
        ]:
            event = _make_event(evt_type, "conv-1")
            exporter._route_event(event)  # Should not raise

    def test_unknown_event_type_ignored(self):
        exporter, collector = self._make_exporter_with_collector()
        event = _make_event("unknown_event", "conv-1")
        exporter._route_event(event)  # Should not raise

    def test_export_event_catches_exceptions(self):
        exporter, collector = self._make_exporter_with_collector()
        # Force _route_event to raise
        exporter._route_event = MagicMock(side_effect=RuntimeError("test"))
        event = _make_event("conversation_start", "conv-1")
        # Should not raise — _export_event catches exceptions
        exporter._export_event(event)

    def test_export_event_skips_when_disabled(self):
        exporter, collector = self._make_exporter_with_collector()
        exporter.disable()
        event = _make_event("conversation_start", "conv-1")
        exporter._export_event(event)
        assert "conv-1" not in exporter._conversation_spans


class TestOTELThreadSafety:
    def test_concurrent_enable_disable(self):
        cls = _import_otel()
        exporter = cls()
        collector = MagicMock()
        errors = []

        def enable_disable():
            try:
                for _ in range(10):
                    exporter.enable(collector)
                    exporter.disable()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=enable_disable) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert errors == [], f"Thread safety errors: {errors}"
