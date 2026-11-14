"""The declarations the mixins of ``InstanceManager`` share (mypy only; plan d454fd62 step 93; the old ``monitor/instance_manager.py``, same dotted path).

What a mixin of the old ``fsm_llm.monitor.instance_manager.InstanceManager`` may rely on: the attributes the
constructor of the shell sets and the methods the shell and the other mixins define, declared for mypy only
(a runtime no-op, like ``workflows/engine/surface.py``). A mixin that calls a method it does not define is
checked against the declaration here; a method called only from its own mixin or from the shell (which
inherits every mixin) needs none.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import collections
    import threading

    from fsm_llm import API

    from ..collector import EventCollector
    from ..definitions import ConversationSnapshot, MonitorConfig, MonitorEvent
    from .managed import ManagedAgent, ManagedFSM, ManagedInstance, ManagedWorkflow


class _InstanceSurface:
    """What every mixin of ``InstanceManager`` may rely on (mypy only)."""

    if TYPE_CHECKING:
        _config: MonitorConfig
        _instances: dict[str, ManagedInstance]
        _collectors: dict[str, EventCollector]
        _global_collector: EventCollector
        _lock: threading.RLock
        _ended_conversations: collections.OrderedDict[str, ConversationSnapshot]
        _max_ended_conversations: int
        _attached_api: API | None

        @property
        def config(self) -> MonitorConfig: ...

        # InstanceManager (manager.py)
        def _new_collector(self) -> EventCollector: ...

        def _get_fsm(self, instance_id: str) -> ManagedFSM: ...

        def _get_workflow(self, instance_id: str) -> ManagedWorkflow: ...

        def _get_agent(self, instance_id: str) -> ManagedAgent: ...

        def _emit_event(
            self,
            event_type: str,
            message: str = "",
            data: dict[str, Any] | None = None,
            level: str = "INFO",
        ) -> None: ...

        def _record_event(self, event: MonitorEvent) -> None: ...

        # SnapshotsMixin (snapshots.py)
        def _ended_snapshot(
            self, conversation_id: str
        ) -> ConversationSnapshot | None: ...

        def _cache_ended_conversation(
            self, api: API, conversation_id: str, instance_id: str = ""
        ) -> None: ...

        def _snapshot_from_api(
            self, api: API, conversation_id: str
        ) -> ConversationSnapshot | None: ...

        # WorkflowsMixin (workflow.py)
        def get_workflow_status(
            self, instance_id: str, wf_instance_id: str
        ) -> dict[str, Any]: ...

        # AgentsMixin (agent.py)
        def get_agent_status(self, instance_id: str) -> dict[str, Any]: ...

        # LifecycleMixin (lifecycle.py)
        def _register_instance(
            self,
            managed: ManagedInstance,
            collector: EventCollector,
            noun: str,
            **launched_data: Any,
        ) -> None: ...
