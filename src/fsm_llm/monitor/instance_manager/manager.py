"""``InstanceManager``, the shell over its mixins (plan d454fd62 step 93; the old ``monitor/instance_manager.py``, same dotted path).

The class holds the constructor, the properties, the loguru sink, ``attach_api`` and ``close``, the instance
queries, the metrics, the typed getters and the two event recorders; every other method lives in one mixin of
this package (``snapshots``, ``fsm``, ``workflow``, ``agent``, ``lifecycle``).
"""

from __future__ import annotations

import collections
import threading
from typing import Any, TypeVar

from fsm_llm import API
from fsm_llm.logging import logger

from ..collector import EventCollector
from ..definitions import (
    ConversationSnapshot,
    DashboardConfig,
    InstanceInfo,
    MetricSnapshot,
    MonitorConfig,
    MonitorEvent,
)
from ..exceptions import (
    MonitorConnectionError,
    MonitorInitializationError,
)
from .agent import AgentsMixin
from .fsm import FsmMixin
from .handlers import register_monitor_handlers, unregister_monitor_handlers
from .lifecycle import LifecycleMixin
from .managed import ManagedAgent, ManagedFSM, ManagedInstance, ManagedWorkflow
from .snapshots import SnapshotsMixin
from .workflow import WorkflowsMixin

_T = TypeVar("_T")


class InstanceManager(
    SnapshotsMixin,
    FsmMixin,
    WorkflowsMixin,
    AgentsMixin,
    LifecycleMixin,
):
    """Manages multiple FSM, workflow, and agent instances.

    Provides launch, control, and query interfaces for all three instance types,
    with per-instance event collection and a global aggregated view.
    """

    def __init__(self, config: MonitorConfig | None = None) -> None:
        self._config = config or MonitorConfig()
        self._instances: dict[str, ManagedInstance] = {}
        self._collectors: dict[str, EventCollector] = {}
        self._global_collector = self._new_collector()
        self._lock = threading.RLock()
        self._closed = False

        # Register loguru sink so log records flow into the collector
        self._setup_loguru_sink()

        # Cache for ended conversations (no longer queryable from API)
        # Bounded to prevent unbounded memory growth in long-running monitors
        self._ended_conversations: collections.OrderedDict[
            str, ConversationSnapshot
        ] = collections.OrderedDict()
        self._max_ended_conversations = 1000

        # An API created by the caller and attached with attach_api()
        self._attached_api: API | None = None

        # Custom dashboard configuration from MonitorBuilder
        self._dashboard_config: DashboardConfig | None = None
        self._dashboard_config_version: int = 0

    @property
    def dashboard_config(self) -> DashboardConfig | None:
        """Get the active custom dashboard config, if any."""
        return self._dashboard_config

    @dashboard_config.setter
    def dashboard_config(self, config: DashboardConfig | None) -> None:
        # version is a read-modify-write compound op; lock so concurrent setters
        # do not lose an increment (DECISION plan_2026-05-29_0c00a594/D-003 [STALE]).
        with self._lock:
            self._dashboard_config = config
            self._dashboard_config_version += 1

    @property
    def dashboard_config_version(self) -> int:
        """Monotonic counter incremented on every dashboard config change."""
        with self._lock:
            return self._dashboard_config_version

    def _setup_loguru_sink(self) -> None:
        """Register a loguru sink that feeds log records into the global collector."""
        try:
            from loguru import logger as _loguru_logger

            sink = self._global_collector.create_loguru_sink()
            self._global_collector._log_sink_id = _loguru_logger.add(
                sink, level=self._config.log_level
            )
        except Exception as e:
            raise MonitorInitializationError(
                f"Failed to register loguru sink: {e}"
            ) from e

    @property
    def config(self) -> MonitorConfig:
        return self._config

    @config.setter
    def config(self, value: MonitorConfig) -> None:
        """Apply a new config: resizes the global collector's buffers and
        re-registers the log sink at ``log_level``. Per-instance collectors
        keep their size (they only hold that instance's events)."""
        old = self._config
        self._config = value
        self._global_collector.resize(value.max_events, value.max_log_lines)
        if value.log_level != old.log_level:
            self._global_collector.cleanup()
            self._setup_loguru_sink()

    @property
    def global_collector(self) -> EventCollector:
        return self._global_collector

    def attach_api(self, api: API) -> None:
        """Show an ``API`` the caller created: its events and conversations.

        Registers the monitor handlers on ``api`` (they feed the global
        collector) and lists its conversations next to the launched ones. One
        API is attached at a time: attaching the same one again changes
        nothing, attaching another switches the handlers on the previous one
        off.

        Raises:
            MonitorConnectionError: the handlers could not be registered; the
                previously attached API (if any) stays attached.
        """
        try:
            register_monitor_handlers(api, self._global_collector)
        except Exception as e:
            raise MonitorConnectionError(
                f"Failed to register monitor handlers: {e}"
            ) from e
        with self._lock:
            previous = self._attached_api
            self._attached_api = api
        if previous is not None and previous is not api:
            unregister_monitor_handlers(previous, self._global_collector)
            # Its handlers are off now, so its END events will never arrive:
            # stop counting its live conversations (the API stays the
            # caller's, never closed here, D-015 of plan 5ed71cf2).
            try:
                live = previous.list_active_conversations()
            except Exception as e:
                logger.debug(f"Failed to list previous API conversations: {e}")
            else:
                self._global_collector.forget_conversations(live)

    def close(self) -> None:
        """Detach from every API, tear every instance down like
        ``end_instance`` (agents are signalled, not joined) and remove the
        log sink (the manager can no longer be used afterwards). Runs once:
        a second call (the server's shutdown after the caller's own) does
        nothing."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            attached_api = self._attached_api
            instances = list(self._instances.values())
        if attached_api is not None:
            unregister_monitor_handlers(attached_api, self._global_collector)
        for inst in instances:
            self._teardown(inst, join_agent=False)
        self._global_collector.cleanup()

    def _new_collector(self) -> EventCollector:
        return EventCollector(
            max_events=self._config.max_events,
            max_log_lines=self._config.max_log_lines,
        )

    def get_capabilities(self) -> dict[str, bool]:
        return {
            "fsm": True,
            "workflows": True,
            "agents": True,
        }

    def list_instances(self, type_filter: str | None = None) -> list[InstanceInfo]:
        with self._lock:
            instances = list(self._instances.values())
        result = []
        for inst in instances:
            if type_filter and inst.instance_type != type_filter:
                continue
            result.append(inst.to_info())
        return result

    def get_instance(self, instance_id: str) -> ManagedInstance | None:
        with self._lock:
            return self._instances.get(instance_id)

    def get_instance_collector(self, instance_id: str) -> EventCollector | None:
        with self._lock:
            return self._collectors.get(instance_id)

    def get_metrics(self) -> MetricSnapshot:
        metrics = self._global_collector.get_metrics()
        # Enrich with live instance counts
        with self._lock:
            metrics.active_agents = sum(
                1
                for inst in self._instances.values()
                if isinstance(inst, ManagedAgent) and inst.status == "running"
            )
            metrics.active_workflows = sum(
                1
                for inst in self._instances.values()
                if isinstance(inst, ManagedWorkflow) and inst.status == "running"
            )
        return metrics

    def get_events(self, limit: int = 50) -> list[MonitorEvent]:
        return self._global_collector.get_events(limit=limit)

    def _get_typed(self, instance_id: str, expected_type: type[_T], label: str) -> _T:
        """Get an instance by ID and verify its type."""
        inst = self.get_instance(instance_id)
        if inst is None:
            raise KeyError(f"Instance not found: {instance_id}")
        if not isinstance(inst, expected_type):
            raise TypeError(f"Instance {instance_id} is not {label}")
        return inst

    def _get_fsm(self, instance_id: str) -> ManagedFSM:
        return self._get_typed(instance_id, ManagedFSM, "an FSM")

    def _get_workflow(self, instance_id: str) -> ManagedWorkflow:
        return self._get_typed(instance_id, ManagedWorkflow, "a workflow")

    def _get_agent(self, instance_id: str) -> ManagedAgent:
        return self._get_typed(instance_id, ManagedAgent, "an agent")

    def _emit_event(
        self,
        event_type: str,
        message: str = "",
        data: dict[str, Any] | None = None,
        level: str = "INFO",
    ) -> None:
        """Record a lifecycle event on the global collector and on the
        instance's own collector."""
        self._record_event(
            MonitorEvent(
                event_type=event_type,
                message=message,
                data=data or {},
                level=level,
            )
        )

    def _record_event(self, event: MonitorEvent) -> None:
        """Record on the global collector and, when ``data["instance_id"]``
        names an instance, on that instance's collector (what
        ``/api/instances/{id}/events`` shows)."""
        self._global_collector.record_event(event)
        collector = self._collectors.get(event.data.get("instance_id", ""))
        if collector is not None:
            collector.record_event(event.model_copy())
