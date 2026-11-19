"""Plan 51fde95b step 12.2: LLM usage on ``MetricSnapshot`` (read-only meter reads, never
``reset_usage``) and the two workflow lifecycle spans."""

from __future__ import annotations

from unittest import mock

import pytest
from fastapi.testclient import TestClient

from fsm_llm import LLMCallCounts, LLMUsage
from fsm_llm.monitor.definitions import MonitorEvent
from fsm_llm.monitor.instance_manager.managed import (
    ManagedFSM,
    add_usage,
    api_usage,
)
from fsm_llm.monitor.server import app, configure
from fsm_llm.testing import MockLLM2Interface
from tests.test_fsm_llm_monitor.support import _manager


def _usage(calls: int = 1, tokens: int = 10, kind: str = "complete") -> LLMUsage:
    counts = LLMCallCounts(
        calls=calls,
        errors=1,
        usage_missing=2,
        prompt_tokens=tokens - 1,
        completion_tokens=1,
        total_tokens=tokens,
    )
    return LLMUsage(**counts.model_dump(), by_kind={kind: counts})


class _Meter:
    """An interface double with a meter; counts what the monitor does to it."""

    def __init__(self, usage: LLMUsage) -> None:
        self._usage, self.reads, self.resets = usage, 0, 0

    def usage(self) -> LLMUsage:
        self.reads += 1
        return self._usage

    def reset_usage(self) -> LLMUsage:
        self.resets += 1
        return self._usage


def _api(llm) -> mock.MagicMock:
    api = mock.MagicMock()
    api.get_llm_interface.return_value = llm
    api.list_active_conversations.return_value = []
    return api


def _with_fsm(mgr, instance_id: str, llm) -> mock.MagicMock:
    api = _api(llm)
    mgr._instances[instance_id] = ManagedFSM(instance_id=instance_id, api=api)
    return api


class TestUsageHelpers:
    def test_add_usage_sums_every_field_and_kind(self):
        total = add_usage(
            add_usage(None, _usage(1, 10, "complete")), _usage(2, 5, "extract")
        )
        again = add_usage(total, _usage(3, 7, "complete"))
        assert (again.calls, again.errors, again.usage_missing) == (6, 3, 6)
        assert (again.prompt_tokens, again.completion_tokens) == (19, 3)
        assert again.total_tokens == 22
        assert again.by_kind["complete"].calls == 4
        assert again.by_kind["complete"].total_tokens == 17
        assert again.by_kind["extract"].total_tokens == 5

    def test_add_usage_of_nothing_is_the_value(self):
        value = _usage()
        assert add_usage(None, value) is value

    def test_an_interface_without_usage_reports_none(self):
        assert api_usage(_api(MockLLM2Interface())) is None
        assert api_usage(_api(object())) is None

    def test_a_usage_that_is_not_a_usage_or_raises_reports_none(self):
        class Raises:
            def usage(self):
                raise RuntimeError("no")

        assert api_usage(_api(mock.MagicMock())) is None
        assert api_usage(_api(Raises())) is None

    def test_a_meter_is_read_never_reset(self):
        meter = _Meter(_usage())
        found = api_usage(_api(meter))
        assert found is not None and found[1].calls == 1
        assert (meter.reads, meter.resets) == (1, 0)


class TestTheTotalOnTheMetrics:
    def test_no_meter_no_usage(self):
        mgr = _manager()
        assert mgr.get_metrics().usage is None
        _with_fsm(mgr, "f1", MockLLM2Interface())
        assert mgr.get_metrics().usage is None

    def test_fsms_and_the_attached_api_are_summed(self):
        mgr = _manager()
        _with_fsm(mgr, "f1", _Meter(_usage(1, 10)))
        _with_fsm(mgr, "f2", _Meter(_usage(2, 20, "extract")))
        mgr._attached_api = _api(_Meter(_usage(4, 40)))
        usage = mgr.get_metrics().usage
        assert usage is not None
        assert (usage.calls, usage.total_tokens) == (7, 70)
        assert usage.by_kind["complete"].calls == 5

    def test_one_interface_shared_by_two_apis_counts_once(self):
        mgr = _manager()
        shared = _Meter(_usage(3, 30))
        _with_fsm(mgr, "f1", shared)
        _with_fsm(mgr, "f2", shared)
        mgr._attached_api = _api(shared)
        usage = mgr.get_metrics().usage
        assert usage is not None and usage.calls == 3

    def test_an_interface_that_fails_leaves_the_others_counted(self):
        mgr = _manager()
        _with_fsm(mgr, "f1", mock.MagicMock())
        _with_fsm(mgr, "f2", _Meter(_usage(2, 20)))
        usage = mgr.get_metrics().usage
        assert usage is not None and usage.calls == 2

    def test_an_ended_instance_keeps_what_it_used(self):
        mgr = _manager()
        gone = _Meter(_usage(2, 20))
        _with_fsm(mgr, "f1", gone)
        _with_fsm(mgr, "f2", _Meter(_usage(1, 10)))
        before = mgr.get_metrics().usage

        mgr.end_instance("f1")
        after = mgr.get_metrics().usage

        assert before is not None and after is not None
        assert after.calls == before.calls == 3
        assert after.total_tokens == before.total_tokens == 30
        assert (gone.resets, gone.reads) == (0, 2)  # the fold read it once

    def test_close_counts_every_interface_once_and_resets_nothing(self):
        mgr = _manager()
        meters = [_Meter(_usage(1, 10)), _Meter(_usage(2, 20))]
        for n, meter in enumerate(meters):
            _with_fsm(mgr, f"f{n}", meter)
        mgr._attached_api = _api(meters[0])  # the same interface

        mgr.close()

        usage = mgr.get_metrics().usage
        assert usage is not None and usage.calls == 3
        assert [m.resets for m in meters] == [0, 0]

    def test_the_metrics_route_carries_it(self):
        mgr = _manager()
        _with_fsm(mgr, "f1", _Meter(_usage(2, 20)))
        configure(manager=mgr)
        body = TestClient(app).get("/api/metrics").json()
        assert body["usage"]["calls"] == 2
        assert body["usage"]["by_kind"]["complete"]["total_tokens"] == 20


class TestTheWorkflowSpans:
    @pytest.fixture
    def spans(self):
        pytest.importorskip("opentelemetry.sdk.trace")
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        from fsm_llm.monitor.otel import OTELExporter

        memory = InMemorySpanExporter()
        exporter = OTELExporter(service_name="t", exporter=memory)

        def run(event: MonitorEvent) -> list:
            exporter._route_event(event)
            exporter._provider.force_flush()
            return list(memory.get_finished_spans())

        yield run
        exporter.shutdown()

    def test_a_step_becomes_a_span_with_ids_only(self, spans):
        (span,) = spans(
            MonitorEvent(
                event_type="workflow_advanced",
                message="Workflow step s1: w1 with secret text",
                data={
                    "instance_id": "i1",
                    "workflow_instance_id": "w1",
                    "step_id": "s1",
                    "success": True,
                },
            )
        )
        assert span.name == "lifecycle.workflow_advanced"
        assert dict(span.attributes) == {
            "fsm_llm.event_type": "workflow_advanced",
            "fsm_llm.conversation_id": "",
            "fsm_llm.instance_id": "i1",
            "fsm_llm.workflow_instance_id": "w1",
            "fsm_llm.step_id": "s1",
        }

    def test_a_delivered_event_exports_no_caller_text(self, spans):
        (span,) = spans(
            MonitorEvent(
                event_type="workflow_event_delivered",
                message="Workflow event delivered: SECRET-EVENT",
                data={
                    "instance_id": "i1",
                    "event_type": "SECRET-EVENT",
                    "affected": ["w1"],
                },
            )
        )
        assert span.name == "lifecycle.workflow_event_delivered"
        assert "SECRET-EVENT" not in repr(dict(span.attributes))
        assert span.attributes["fsm_llm.instance_id"] == "i1"
        assert not any("latency" in key for key in span.attributes)

    def test_a_non_text_step_id_is_not_exported(self, spans):
        (span,) = spans(
            MonitorEvent(
                event_type="workflow_advanced", data={"instance_id": "i1", "step_id": 7}
            )
        )
        assert "fsm_llm.step_id" not in span.attributes
