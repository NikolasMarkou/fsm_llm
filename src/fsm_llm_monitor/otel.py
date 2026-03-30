from __future__ import annotations

"""
OpenTelemetry (OTEL) Observability Adapter.

Wraps EventCollector events into OTEL spans and metrics using the adapter
pattern. Does not modify the existing EventCollector — observes its events
and exports them to OTEL-compatible backends (Jaeger, Datadog, Langfuse, etc.).
"""

import uuid
from typing import Any

from fsm_llm.logging import logger

from .definitions import MonitorEvent

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
    )
    from opentelemetry.trace import StatusCode

    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False


def _require_otel() -> None:
    if not _HAS_OTEL:
        raise ImportError(
            "OTEL support requires opentelemetry packages. "
            "Install with: pip install fsm-llm[otel] or "
            "pip install opentelemetry-api opentelemetry-sdk"
        )


class OTELExporter:
    """Exports EventCollector events to OpenTelemetry spans and metrics.

    Wraps the existing monitoring infrastructure without modifying it.
    Events are converted to spans with appropriate attributes.

    Example::

        from fsm_llm_monitor import EventCollector
        from fsm_llm_monitor.otel import OTELExporter

        collector = EventCollector()
        otel = OTELExporter(service_name="my-chatbot")
        otel.enable(collector)

        # Events recorded by the collector are now also exported as OTEL spans
    """

    def __init__(
        self,
        service_name: str = "fsm-llm",
        exporter: Any | None = None,
    ) -> None:
        _require_otel()
        self._service_name = service_name
        self._enabled = False
        self._collector = None

        # Set up tracer
        provider = TracerProvider()
        if exporter is not None:
            provider.add_span_processor(BatchSpanProcessor(exporter))
        else:
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(provider)
        self._tracer = trace.get_tracer(service_name)
        self._provider = provider

        # Active spans by conversation_id
        self._conversation_spans: dict[str, Any] = {}

        logger.info(f"OTELExporter initialized for service '{service_name}'")

    def enable(self, collector: Any) -> None:
        """Enable OTEL export for an EventCollector.

        Wraps the collector's record_event method to also emit OTEL spans.
        """
        _require_otel()
        self._collector = collector
        self._enabled = True

        # Wrap the collector's record_event
        original_record = collector.record_event

        def wrapped_record(event: MonitorEvent) -> None:
            original_record(event)
            self._export_event(event)

        collector.record_event = wrapped_record
        logger.info("OTEL export enabled")

    def disable(self) -> None:
        """Disable OTEL export."""
        self._enabled = False
        # End any active conversation spans
        for _conv_id, span in list(self._conversation_spans.items()):
            span.end()
        self._conversation_spans.clear()
        logger.info("OTEL export disabled")

    def shutdown(self) -> None:
        """Shutdown the OTEL provider, flushing pending spans."""
        self.disable()
        if hasattr(self._provider, "shutdown"):
            self._provider.shutdown()

    def _export_event(self, event: MonitorEvent) -> None:
        """Convert a MonitorEvent to OTEL spans."""
        if not self._enabled:
            return

        try:
            self._route_event(event)
        except Exception as e:
            logger.debug(f"OTEL export error (non-fatal): {e}")

    def _route_event(self, event: MonitorEvent) -> None:
        """Route event to the appropriate span handler."""
        event_type = event.event_type

        if event_type == "conversation_start":
            self._on_conversation_start(event)
        elif event_type == "conversation_end":
            self._on_conversation_end(event)
        elif event_type == "state_transition":
            self._on_state_transition(event)
        elif event_type == "pre_processing":
            self._on_processing(event, "pre_processing")
        elif event_type == "post_processing":
            self._on_processing(event, "post_processing")
        elif event_type == "error":
            self._on_error(event)
        elif event_type in (
            "agent_started", "agent_completed", "agent_failed",
            "workflow_started", "workflow_completed", "workflow_cancelled",
        ):
            self._on_lifecycle_event(event)

    def _on_conversation_start(self, event: MonitorEvent) -> None:
        """Start a conversation-level span."""
        conv_id = event.conversation_id or "unknown"
        span = self._tracer.start_span(
            f"conversation.{conv_id}",
            attributes={
                "fsm_llm.conversation_id": conv_id,
                "fsm_llm.event_type": event.event_type,
                "fsm_llm.service": self._service_name,
            },
        )
        self._conversation_spans[conv_id] = span

    def _on_conversation_end(self, event: MonitorEvent) -> None:
        """End the conversation-level span."""
        conv_id = event.conversation_id or "unknown"
        span = self._conversation_spans.pop(conv_id, None)
        if span:
            span.set_attribute("fsm_llm.final_state", event.data.get("state", ""))
            span.end()

    def _on_state_transition(self, event: MonitorEvent) -> None:
        """Record a state transition as a child span."""
        conv_id = event.conversation_id or "unknown"
        parent = self._conversation_spans.get(conv_id)

        ctx = trace.set_span_in_context(parent) if parent else None

        with self._tracer.start_as_current_span(
            f"transition.{event.source_state}->{event.target_state}",
            context=ctx,
            attributes={
                "fsm_llm.conversation_id": conv_id,
                "fsm_llm.source_state": event.source_state or "",
                "fsm_llm.target_state": event.target_state or "",
                "fsm_llm.event_type": "state_transition",
            },
        ):
            pass  # Span auto-ends on context exit

    def _on_processing(self, event: MonitorEvent, phase: str) -> None:
        """Record a processing event as a span."""
        conv_id = event.conversation_id or "unknown"
        parent = self._conversation_spans.get(conv_id)
        ctx = trace.set_span_in_context(parent) if parent else None

        attrs: dict[str, Any] = {
            "fsm_llm.conversation_id": conv_id,
            "fsm_llm.event_type": phase,
        }
        # Add LLM-specific attributes if present
        data = event.data
        if "model" in data:
            attrs["fsm_llm.model"] = str(data["model"])
        if "tokens" in data:
            attrs["fsm_llm.tokens"] = int(data["tokens"])
        if "latency_ms" in data:
            attrs["fsm_llm.latency_ms"] = float(data["latency_ms"])

        with self._tracer.start_as_current_span(
            f"processing.{phase}",
            context=ctx,
            attributes=attrs,
        ):
            pass

    def _on_error(self, event: MonitorEvent) -> None:
        """Record an error event."""
        conv_id = event.conversation_id or "unknown"
        parent = self._conversation_spans.get(conv_id)

        if parent:
            parent.set_status(StatusCode.ERROR, event.message)
            parent.set_attribute("fsm_llm.error", event.message)

    def _on_lifecycle_event(self, event: MonitorEvent) -> None:
        """Record agent/workflow lifecycle events."""
        with self._tracer.start_as_current_span(
            f"lifecycle.{event.event_type}",
            attributes={
                "fsm_llm.event_type": event.event_type,
                "fsm_llm.conversation_id": event.conversation_id or "",
                "fsm_llm.message": event.message,
            },
        ):
            pass

    @staticmethod
    def generate_trace_id() -> str:
        """Generate a trace ID for manual correlation."""
        return uuid.uuid4().hex

    @staticmethod
    def generate_span_id() -> str:
        """Generate a span ID for manual correlation."""
        return uuid.uuid4().hex[:16]

    @property
    def is_enabled(self) -> bool:
        """Check if OTEL export is enabled."""
        return self._enabled

    @property
    def active_conversations(self) -> list[str]:
        """Return conversation IDs with active spans."""
        return list(self._conversation_spans.keys())
