"""OTELExporter (``fsm_llm.monitor.otel``): thread-safe span handling and the exporter lifecycle.

Tests the monitor exporter from the agents suite; they stay here because a move would change their
pinned (suite, class::test) ids. Skipped per test when the opentelemetry api AND sdk are missing.
"""

from __future__ import annotations

import threading
from unittest.mock import Mock

import pytest


def _has_otel() -> bool:
    """True only when the exporter's own import (api AND sdk) succeeded.

    ``find_spec("opentelemetry")`` is not enough: the mcp extra pulls in
    opentelemetry-api without the sdk, and the exporter then raises.
    """
    try:
        from fsm_llm.monitor.otel import _HAS_OTEL
    except ImportError:
        return False
    return _HAS_OTEL


class TestOTELThreadSafety:
    """Verify OTEL exporter has thread-safe span management."""

    @pytest.mark.skipif(not _has_otel(), reason="opentelemetry sdk not installed")
    def test_has_spans_lock(self):
        """OTELExporter has _spans_lock attribute."""
        from fsm_llm.monitor.otel import OTELExporter

        exporter = OTELExporter(service_name="test")
        assert hasattr(exporter, "_spans_lock")
        assert isinstance(exporter._spans_lock, type(threading.Lock()))

    @pytest.mark.skipif(not _has_otel(), reason="opentelemetry sdk not installed")
    def test_concurrent_events_no_crash(self):
        """Concurrent event recording doesn't crash."""
        from fsm_llm.monitor.definitions import MonitorEvent
        from fsm_llm.monitor.otel import OTELExporter

        exporter = OTELExporter(service_name="test")
        collector = Mock()
        collector.record_event = Mock()
        exporter.enable(collector)

        errors = []

        def record_events(conv_prefix, count):
            try:
                for i in range(count):
                    start = MonitorEvent(
                        event_type="conversation_start",
                        conversation_id=f"{conv_prefix}-{i}",
                        message=f"test {i}",
                    )
                    exporter._export_event(start)
                    end = MonitorEvent(
                        event_type="conversation_end",
                        conversation_id=f"{conv_prefix}-{i}",
                        message="done",
                    )
                    exporter._export_event(end)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=record_events, args=(f"t{i}", 20)) for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"


class TestOTELExporter:
    """Tests for OTELExporter."""

    def test_import_without_otel(self):
        """OTELExporter module imports even without opentelemetry."""
        from fsm_llm.monitor.otel import OTELExporter

        assert OTELExporter is not None

    @pytest.mark.skipif(
        not _has_otel(),
        reason="opentelemetry not installed",
    )
    def test_exporter_initialization(self):
        """OTELExporter initializes with service name."""
        from fsm_llm.monitor.otel import OTELExporter

        exporter = OTELExporter(service_name="test-service")
        assert not exporter.is_enabled
        assert exporter.active_conversations == []

    @pytest.mark.skipif(
        not _has_otel(),
        reason="opentelemetry not installed",
    )
    def test_enable_disable(self):
        """Enable and disable OTEL export."""
        from fsm_llm.monitor.otel import OTELExporter

        collector = Mock()
        collector.record_event = Mock()

        exporter = OTELExporter(service_name="test")
        exporter.enable(collector)
        assert exporter.is_enabled

        exporter.disable()
        assert not exporter.is_enabled
