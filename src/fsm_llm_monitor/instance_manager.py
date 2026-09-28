"""
Instance manager for fsm_llm_monitor.

Manages multiple concurrent FSM, workflow, and agent instances with
per-instance event collection and lifecycle management.
"""

from __future__ import annotations

import asyncio
import collections
import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

_T = TypeVar("_T")

from fsm_llm import API, HandlerTiming
from fsm_llm.constants import DEFAULT_LLM_MODEL
from fsm_llm.handlers import BaseHandler
from fsm_llm.logging import logger

from .collector import EventCollector, redact_context
from .constants import (
    EVENT_AGENT_CANCELLED,
    EVENT_AGENT_COMPLETED,
    EVENT_AGENT_FAILED,
    EVENT_AGENT_ITERATION,
    EVENT_AGENT_STARTED,
    EVENT_AGENT_TOOL_CALL,
    EVENT_INSTANCE_DESTROYED,
    EVENT_INSTANCE_LAUNCHED,
    EVENT_WORKFLOW_ADVANCED,
    EVENT_WORKFLOW_CANCELLED,
    EVENT_WORKFLOW_COMPLETED,
    EVENT_WORKFLOW_EVENT_DELIVERED,
    EVENT_WORKFLOW_FAILED,
    EVENT_WORKFLOW_STARTED,
    MONITOR_HANDLER_NAME,
    MONITOR_HANDLER_PRIORITY,
)
from .definitions import (
    ActivityItem,
    ConversationSnapshot,
    DashboardConfig,
    InstanceInfo,
    MetricSnapshot,
    MonitorConfig,
    MonitorEvent,
    StubToolConfig,
    model_to_dict,
    normalize_message_history,
)
from .exceptions import MonitorCapacityError, MonitorInitializationError

# Optional imports for workflows and agents
_HAS_WORKFLOWS = False
_HAS_AGENTS = False

try:
    from fsm_llm_workflows import (
        WorkflowDefinition,
        WorkflowEngine,
        WorkflowEvent,
        WorkflowStep,
        WorkflowStepResult,
        auto_step,
        condition_step,
        create_workflow,
    )

    # DECISION plan_2026-05-29_0c00a594/D-002 [STALE]: the dashboard exposes a small set of
    # built-in, DSL-built workflow presets rather than accepting arbitrary
    # `definition_json`. WorkflowDefinition.steps are WorkflowStep ABC subclasses
    # carrying Python callables; there is no JSON->workflow loader, so arbitrary
    # user JSON cannot be turned into a runnable workflow. Presets use only pure
    # steps (auto/condition) so they run deterministically with no external LLM/API.
    class _TerminalStep(WorkflowStep):
        """No-op terminal step (no outgoing transition) for monitor demo presets.

        A step whose result carries no ``next_state`` and which references no
        other state is treated as terminal by ``WorkflowDefinition`` and drives
        the instance to COMPLETED.
        """

        async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
            return WorkflowStepResult.success_result(
                message=f"Reached terminal step '{self.step_id}'"
            )

    def _preset_linear() -> WorkflowDefinition:
        """Linear pipeline: ingest -> process -> finish (auto-completes on launch)."""
        wf = create_workflow(
            "demo_linear",
            "Demo: Linear Pipeline",
            "Two automatic processing steps followed by a terminal step.",
        )
        wf.with_initial_step(
            auto_step(
                "ingest",
                "Ingest",
                next_state="process",
                action=lambda ctx: {"ingested": True},
            )
        )
        wf.with_step(
            auto_step(
                "process",
                "Process",
                next_state="finish",
                action=lambda ctx: {"processed": True},
            )
        )
        wf.with_step(_TerminalStep(step_id="finish", name="Finish"))
        return wf

    def _preset_branching() -> WorkflowDefinition:
        """Conditional branch on the initial-context ``amount`` value."""
        wf = create_workflow(
            "demo_branching",
            "Demo: Conditional Branch",
            "Routes to a high/low path based on the 'amount' context value.",
        )
        wf.with_initial_step(
            condition_step(
                "route",
                "Route by amount",
                condition=lambda ctx: float(ctx.get("amount", 0) or 0) >= 1000,
                true_state="high_path",
                false_state="low_path",
            )
        )
        wf.with_step(
            auto_step(
                "high_path",
                "High Path",
                next_state="finish",
                action=lambda ctx: {"tier": "high"},
            )
        )
        wf.with_step(
            auto_step(
                "low_path",
                "Low Path",
                next_state="finish",
                action=lambda ctx: {"tier": "low"},
            )
        )
        wf.with_step(_TerminalStep(step_id="finish", name="Finish"))
        return wf

    # Registry of preset builders. Each call returns a fresh WorkflowDefinition so
    # concurrent launches never share mutable instance state.
    _WORKFLOW_PRESETS: dict[str, Any] = {
        "demo_linear": _preset_linear,
        "demo_branching": _preset_branching,
    }

    _HAS_WORKFLOWS = True
except ImportError:
    _WORKFLOW_PRESETS = {}

try:
    from fsm_llm_agents import (
        ADaPTAgent,
        AgentConfig,
        DebateAgent,
        PlanExecuteAgent,
        ReactAgent,
        ReflexionAgent,
        REWOOAgent,
        SelfConsistencyAgent,
        ToolRegistry,
    )

    # DECISION plan_2026-05-29_0c00a594/D-001 [STALE]: EvaluatorOptimizerAgent and
    # MakerCheckerAgent are intentionally NOT launchable from the dashboard.
    # Their constructors require arguments that cannot be supplied from a web
    # form (a Python `evaluation_fn` callable / maker+checker instruction strings),
    # so launching them always raised TypeError. They remain importable as library
    # classes and in flows.json for visualization. Do not re-add here without also
    # wiring a way to supply their required constructor arguments.
    _AGENT_CLASSES: dict[str, type] = {
        "ReactAgent": ReactAgent,
        "ReflexionAgent": ReflexionAgent,
        "PlanExecuteAgent": PlanExecuteAgent,
        "REWOOAgent": REWOOAgent,
        "ADaPTAgent": ADaPTAgent,
        "DebateAgent": DebateAgent,
        "SelfConsistencyAgent": SelfConsistencyAgent,
    }

    # Agent types that require a ToolRegistry
    _TOOL_BASED_AGENTS = {
        "ReactAgent",
        "ReflexionAgent",
        "PlanExecuteAgent",
        "REWOOAgent",
        "ADaPTAgent",
    }

    _HAS_AGENTS = True
except ImportError:
    _AGENT_CLASSES = {}
    _TOOL_BASED_AGENTS = set()


class ManagedFSM:
    """A managed FSM API instance."""

    def __init__(
        self,
        instance_id: str,
        api: API,
        label: str = "",
        source: str = "custom",
    ) -> None:
        self.instance_id = instance_id
        self.instance_type = "fsm"
        self.api = api
        self.label = label
        self.source = source
        self.status = "running"
        self.created_at = datetime.now(timezone.utc)
        self.conversation_ids: list[str] = []
        self.ended_conversation_ids: set[str] = set()

    def to_info(self) -> InstanceInfo:
        return InstanceInfo(
            instance_id=self.instance_id,
            instance_type=self.instance_type,
            label=self.label,
            status=self.status,
            created_at=self.created_at,
            source=self.source,
            conversation_count=len(self.conversation_ids),
        )


class ManagedWorkflow:
    """A managed workflow engine instance."""

    def __init__(
        self,
        instance_id: str,
        label: str = "",
        source: str = "custom",
    ) -> None:
        self.instance_id = instance_id
        self.instance_type = "workflow"
        self.label = label
        self.source = source
        self.status = "running"
        self.created_at = datetime.now(timezone.utc)
        self.engine: Any = None  # WorkflowEngine
        self.workflow_id: str = ""
        self.active_instance_ids: list[str] = []
        # Every run started on this engine, in order, and its last status.
        self.run_ids: list[str] = []
        self.run_statuses: dict[str, str] = {}
        # Event loop the engine's timers run on (set at launch).
        self.loop: asyncio.AbstractEventLoop | None = None

    def to_info(self) -> InstanceInfo:
        return InstanceInfo(
            instance_id=self.instance_id,
            instance_type=self.instance_type,
            label=self.label,
            status=self.status,
            created_at=self.created_at,
            source=self.source,
            active_workflows=len(self.active_instance_ids),
        )

    def refresh_status(self) -> None:
        """``running`` while any run is active, else the last run's status."""
        if self.active_instance_ids or not self.run_statuses:
            self.status = "running"
            return
        self.status = self.run_statuses[self.run_ids[-1]]


class ManagedAgent:
    """A managed agent instance running in a background thread."""

    def __init__(
        self,
        instance_id: str,
        agent_type: str = "ReactAgent",
        task: str = "",
        label: str = "",
    ) -> None:
        self.instance_id = instance_id
        self.instance_type = "agent"
        self.agent_type = agent_type
        self.task = task
        self.label = label
        self.status = "running"
        self.created_at = datetime.now(timezone.utc)
        self.thread: threading.Thread | None = None
        self.cancel_event = threading.Event()
        self.max_iterations: int = 10
        self.result: Any = None  # AgentResult
        self.error: str | None = None
        # Live conversation log — populated by context capture handlers
        self.conversation_log: list[dict[str, Any]] = []
        self._conv_lock = threading.Lock()

    def to_info(self) -> InstanceInfo:
        return InstanceInfo(
            instance_id=self.instance_id,
            instance_type=self.instance_type,
            label=self.label,
            status=self.status,
            created_at=self.created_at,
            agent_type=self.agent_type,
            task=self.task[:200],
        )


ManagedInstance = ManagedFSM | ManagedWorkflow | ManagedAgent


class _MonitorHandler(BaseHandler):
    """Observer handler that hands the core's state arguments to a monitor
    callback.

    # DECISION plan-2026-09-28T090000-3c9e41d2/D-003
    # The core passes ``current_state``/``target_state`` to ``should_execute``
    # only; the handler context has no ``_target_state``. Do NOT go back to a
    # ``create_handler(...).do(callback)`` lambda that reads the states from
    # the context: every transition event then had an empty target. The
    # states are kept per thread between ``should_execute`` and ``execute``
    # (the core calls them back to back on the same thread).
    """

    def __init__(
        self,
        name: str,
        timing: HandlerTiming,
        callback: Any,
        collector: EventCollector,
        priority: int = MONITOR_HANDLER_PRIORITY,
    ) -> None:
        super().__init__(name=name, priority=priority)
        self.timing = timing
        self.callback = callback
        self.collector = collector
        self.active = True
        self._states = threading.local()

    def should_execute(
        self,
        timing: HandlerTiming,
        current_state: str,
        target_state: str | None,
        context: dict[str, Any],
        updated_keys: set[str] | None = None,
    ) -> bool:
        if not self.active or timing != self.timing:
            return False
        self._states.current = current_state
        self._states.target = target_state
        return True

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        current = getattr(self._states, "current", None)
        target = getattr(self._states, "target", None)
        self.callback(context, current_state=current, target_state=target)
        # Observe only: never write into the conversation context.
        return {}


def _handlers_for(
    callbacks: dict[str, Any],
    collector: EventCollector,
    name_prefix: str,
    priority: int,
) -> list[_MonitorHandler]:
    return [
        _MonitorHandler(
            name=f"{name_prefix}_{timing_name.lower()}",
            timing=HandlerTiming[timing_name],
            callback=callback,
            collector=collector,
            priority=priority,
        )
        for timing_name, callback in callbacks.items()
        # POST_TRANSITION is a no-op observer; registering it would only make
        # the core deep-copy the context once more per transition.
        if timing_name != "POST_TRANSITION"
    ]


def monitor_handlers_on(api: API, collector: EventCollector) -> list[_MonitorHandler]:
    """The active monitor handlers of ``collector`` registered on ``api``."""
    handlers = getattr(getattr(api, "handler_system", None), "handlers", []) or []
    return [
        h
        for h in handlers
        if isinstance(h, _MonitorHandler) and h.collector is collector and h.active
    ]


def register_monitor_handlers(
    api: API, collector: EventCollector
) -> list[_MonitorHandler]:
    """Register observer handlers on an API instance for event collection.

    Idempotent per (api, collector): calling it again does not stack a second
    set (the core has no way to unregister a handler). Returns the handlers
    now active for this collector; ``unregister_monitor_handlers`` switches
    them off.
    """
    existing = monitor_handlers_on(api, collector)
    if existing:
        return existing
    handlers = _handlers_for(
        collector.create_handler_callbacks(),
        collector,
        MONITOR_HANDLER_NAME,
        MONITOR_HANDLER_PRIORITY,
    )
    for handler in handlers:
        api.register_handler(handler)
    return handlers


def unregister_monitor_handlers(api: API, collector: EventCollector) -> int:
    """Switch off ``collector``'s monitor handlers on ``api``; returns how many."""
    handlers = monitor_handlers_on(api, collector)
    for handler in handlers:
        handler.active = False
    return len(handlers)


def _build_monitor_handlers(collector: EventCollector) -> list[Any]:
    """Build a list of handler objects for injection into agents via api_kwargs."""
    return list(
        _handlers_for(
            collector.create_handler_callbacks(),
            collector,
            MONITOR_HANDLER_NAME,
            MONITOR_HANDLER_PRIORITY,
        )
    )


def _build_context_capture_handlers(
    collector: EventCollector,
    sink: ManagedAgent,
) -> list[Any]:
    """Build handler objects that capture context snapshots into a ManagedAgent.

    Delegates to ``EventCollector.create_context_capture_callbacks`` for the
    actual capture logic, then wraps each callback as a handler object suitable
    for injection into agents via ``api_kwargs["handlers"]``.
    """
    return list(
        _handlers_for(
            collector.create_context_capture_callbacks(sink),
            collector,
            f"{MONITOR_HANDLER_NAME}_capture",
            MONITOR_HANDLER_PRIORITY - 1,
        )
    )


# Strong references to fire-and-forget tasks (the loop keeps only weak ones).
_pending_tasks: set[Any] = set()


def _tools_used(tool_calls: Any) -> list[dict[str, Any]]:
    """Tool name and redacted parameters for each traced tool call."""
    return [
        {
            "tool_name": getattr(tc, "tool_name", ""),
            "parameters": redact_context(getattr(tc, "parameters", None) or {}),
        }
        for tc in tool_calls or []
    ]


def _find_examples_dir() -> Path | None:
    """Locate the examples/ directory."""
    base = Path(__file__).parent.parent.parent / "examples"
    if not base.exists():
        base = base.parent / "examples"
    return base if base.exists() else None


def _status_str(status: Any) -> str:
    """Extract a string from a status value (may be an enum or plain string)."""
    return status.value if hasattr(status, "value") else str(status)


def validate_preset_id(preset_id: str, base: Path) -> Path:
    """Validate a preset ID against path traversal and resolve to a file path.

    Raises ``ValueError`` if the ID is invalid or escapes the base directory.
    Raises ``FileNotFoundError`` if the resolved file does not exist.
    """
    if ".." in preset_id or preset_id.startswith("/"):
        raise ValueError("Invalid preset ID")
    file_path = base / preset_id
    try:
        file_path.resolve().relative_to(base.resolve())
    except ValueError as e:
        raise ValueError("Invalid preset ID") from e
    if not file_path.exists():
        raise FileNotFoundError(f"Preset not found: {preset_id}")
    return file_path


def snapshot_from_api(
    api: API,
    conversation_id: str,
    show_internal_keys: bool = False,
) -> ConversationSnapshot | None:
    """Build a ConversationSnapshot from a live API instance.

    Shared helper used by both MonitorBridge and InstanceManager to avoid
    duplicating the snapshot extraction logic.
    """
    try:
        complete = api.fsm_manager.get_complete_conversation(conversation_id)
        if complete is None:
            return None
        current_state = complete.get("current_state", {})
        context_data = complete.get("collected_data", {})
        # DECISION plan-2026-07-20T040150-876e7164/D-003 [STALE]
        # This is the CONFIRMED leak: `show_internal_keys=False` is the
        # dashboard's "hide it" knob, and the old `k.startswith("_")` here
        # hid only literal `_`-prefixed keys -- `system_password`,
        # `internal_token` and every case variant were rendered anyway in
        # `ConversationSnapshot.context_data` (F-13, measured end-to-end).
        # Do NOT re-inline the startswith check. See decisions.md D-003.
        # Secret-looking entries are dropped at every depth regardless of
        # `show_internal_keys` (plan-2026-09-28T090000-3c9e41d2/D-002).
        context_data = redact_context(
            context_data, drop_internal=not show_internal_keys
        )

        def _redacted(obj: Any) -> dict[str, Any] | None:
            as_dict = model_to_dict(obj)
            return None if as_dict is None else redact_context(as_dict)

        return ConversationSnapshot(
            conversation_id=conversation_id,
            current_state=current_state.get("id", ""),
            state_description=current_state.get("description", ""),
            is_terminal=current_state.get("is_terminal", False),
            context_data=context_data,
            message_history=normalize_message_history(
                complete.get("conversation_history", [])
            ),
            stack_depth=api.get_stack_depth(conversation_id),
            last_extraction=_redacted(complete.get("last_extraction_response")),
            last_transition=_redacted(complete.get("last_transition_decision")),
            last_response=_redacted(complete.get("last_response_generation")),
        )
    except Exception as e:
        logger.debug(f"Failed to get conversation snapshot for {conversation_id}: {e}")
        return None


class InstanceManager:
    """Manages multiple FSM, workflow, and agent instances.

    Provides launch, control, and query interfaces for all three instance types,
    with per-instance event collection and a global aggregated view.
    """

    def __init__(self, config: MonitorConfig | None = None) -> None:
        self._config = config or MonitorConfig()
        self._instances: dict[str, ManagedInstance] = {}
        self._collectors: dict[str, EventCollector] = {}
        self._global_collector = EventCollector(
            max_events=self._config.max_events,
            max_log_lines=self._config.max_log_lines,
        )
        self._lock = threading.RLock()

        # Register loguru sink so log records flow into the collector
        self._setup_loguru_sink()

        # Cache for ended conversations (no longer queryable from API)
        # Bounded to prevent unbounded memory growth in long-running monitors
        self._ended_conversations: collections.OrderedDict[
            str, ConversationSnapshot
        ] = collections.OrderedDict()
        self._max_ended_conversations = 1000

        # For backward compat: an externally-connected bridge API
        self._bridge_api: API | None = None
        self._bridge_collector: EventCollector | None = None

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

    def connect_bridge(self, api: API) -> None:
        """Connect an external API instance (backward compat with MonitorBridge).

        Idempotent for the same API; connecting a different API switches the
        monitor handlers on the previous one off.
        """
        with self._lock:
            previous = self._bridge_api
            self._bridge_api = api
            self._bridge_collector = self._global_collector
        if previous is not None and previous is not api:
            unregister_monitor_handlers(previous, self._global_collector)
        register_monitor_handlers(api, self._global_collector)

    def shutdown(self) -> None:
        """Detach from every API and remove the log sink (the manager can no
        longer be used afterwards)."""
        with self._lock:
            bridge_api = self._bridge_api
            fsms = [i for i in self._instances.values() if isinstance(i, ManagedFSM)]
        if bridge_api is not None:
            unregister_monitor_handlers(bridge_api, self._global_collector)
        for inst in fsms:
            unregister_monitor_handlers(inst.api, self._global_collector)
        self._global_collector.cleanup()

    # --- Capabilities ---

    def get_capabilities(self) -> dict[str, bool]:
        return {
            "fsm": True,
            "workflows": _HAS_WORKFLOWS,
            "agents": _HAS_AGENTS,
        }

    # --- Instance Query ---

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

    # --- Global Metrics ---

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

    # --- Conversation queries (aggregated across all FSM instances + bridge) ---

    def get_active_conversations(self) -> list[str]:
        result: list[str] = []
        # From bridge API (snapshot the reference under lock; D-003)
        with self._lock:
            bridge_api = self._bridge_api
        if bridge_api is not None:
            try:
                result.extend(bridge_api.list_active_conversations())
            except Exception as e:
                logger.debug(f"Failed to list bridge API conversations: {e}")
        # From managed FSMs
        with self._lock:
            for inst in self._instances.values():
                if isinstance(inst, ManagedFSM) and inst.status == "running":
                    try:
                        result.extend(inst.api.list_active_conversations())
                    except Exception as e:
                        logger.debug(
                            f"Failed to list conversations for instance {inst.instance_id}: {e}"
                        )
        return result

    def find_instance_for_conversation(self, conversation_id: str) -> str | None:
        """Find the instance_id that owns a given conversation."""
        with self._lock:
            for inst_id, inst in self._instances.items():
                if (
                    isinstance(inst, ManagedFSM)
                    and conversation_id in inst.conversation_ids
                ):
                    return inst_id
        return None

    def get_conversation_snapshot(
        self, conversation_id: str
    ) -> ConversationSnapshot | None:
        """Find and return a conversation snapshot from any FSM instance."""
        # Check bridge API first (snapshot the reference under lock; D-003)
        with self._lock:
            bridge_api = self._bridge_api
        if bridge_api is not None:
            snap = self._snapshot_from_api(bridge_api, conversation_id)
            if snap is not None:
                return snap
        # Check managed FSMs
        with self._lock:
            fsm_instances = [
                (inst_id, inst)
                for inst_id, inst in self._instances.items()
                if isinstance(inst, ManagedFSM)
            ]
        for inst_id, inst in fsm_instances:
            snap = self._snapshot_from_api(inst.api, conversation_id)
            if snap is not None:
                snap.instance_id = inst_id
                return snap
        # Fallback to ended conversation cache
        with self._lock:
            if conversation_id in self._ended_conversations:
                return self._ended_conversations[conversation_id]
        return None

    def get_all_conversation_snapshots(
        self, include_ended: bool = True
    ) -> list[ConversationSnapshot]:
        seen: set[str] = set()
        snapshots: list[ConversationSnapshot] = []

        # Active conversations from all FSM APIs
        for conv_id in self.get_active_conversations():
            snap = self.get_conversation_snapshot(conv_id)
            if snap is not None and conv_id not in seen:
                snapshots.append(snap)
                seen.add(conv_id)

        # All known conversation IDs from managed instances
        with self._lock:
            all_inst_conv_ids = [
                (inst.instance_id if isinstance(inst, ManagedFSM) else "", conv_id)
                for inst in self._instances.values()
                if isinstance(inst, ManagedFSM)
                for conv_id in inst.conversation_ids
            ]
        for inst_id, conv_id in all_inst_conv_ids:
            if conv_id in seen:
                continue
            snap = self.get_conversation_snapshot(conv_id)
            if snap is not None:
                snap.instance_id = inst_id
                snapshots.append(snap)
                seen.add(conv_id)

        # Ended conversations from cache
        if include_ended:
            with self._lock:
                for conv_id, snap in self._ended_conversations.items():
                    if conv_id not in seen:
                        snapshots.append(snap)
                        seen.add(conv_id)

        return snapshots

    def get_all_activity_snapshots(
        self, include_ended: bool = True
    ) -> list[ActivityItem]:
        """Get a unified list of all activity: FSM conversations, agent tasks, workflow instances."""
        items: list[ActivityItem] = []

        # FSM conversations
        for snap in self.get_all_conversation_snapshots(include_ended=include_ended):
            status = "ended" if snap.is_terminal else "active"
            items.append(
                ActivityItem(
                    item_id=snap.conversation_id,
                    item_type="fsm_conversation",
                    instance_id=snap.instance_id,
                    label=snap.conversation_id[:12],
                    status=status,
                    current_step=snap.current_state,
                    detail=snap.state_description,
                    message_count=len(snap.message_history),
                    is_terminal=snap.is_terminal,
                )
            )

        # Agent tasks
        with self._lock:
            agents = [
                (inst_id, inst)
                for inst_id, inst in self._instances.items()
                if isinstance(inst, ManagedAgent)
            ]
        for inst_id, inst in agents:
            status_data = self.get_agent_status(inst_id)
            iteration_count = status_data.get("iteration_count", 0)
            total_iterations = status_data.get("total_iterations", 0)
            iters = total_iterations if total_iterations else iteration_count
            current_step = f"iter {iters}/{inst.max_iterations}"
            if inst.status != "running":
                current_step = inst.status
            items.append(
                ActivityItem(
                    item_id=inst_id,
                    item_type="agent_task",
                    instance_id=inst_id,
                    label=inst.label,
                    status=inst.status,
                    current_step=current_step,
                    detail=inst.agent_type,
                    message_count=iters,
                    created_at=inst.created_at,
                    is_terminal=inst.status in ("completed", "failed", "cancelled"),
                )
            )

        # Workflow instances
        with self._lock:
            workflows = [
                (wf_inst_id, wf_inst)
                for wf_inst_id, wf_inst in self._instances.items()
                if isinstance(wf_inst, ManagedWorkflow)
            ]
        for wf_inst_id, wf_inst in workflows:
            with self._lock:
                run_ids = list(wf_inst.run_ids)
                active = set(wf_inst.active_instance_ids)
            for wf_id in run_ids:
                if not include_ended and wf_id not in active:
                    continue
                try:
                    wf_status = self.get_workflow_status(wf_inst_id, wf_id)
                    raw_status = wf_status.get("status", "unknown")
                    normalized_status = raw_status.lower()
                    items.append(
                        ActivityItem(
                            item_id=wf_id,
                            item_type="workflow_instance",
                            instance_id=wf_inst_id,
                            label=wf_inst.label,
                            status=normalized_status,
                            current_step=wf_status.get("current_step", ""),
                            detail=f"workflow {wf_id[:8]}",
                            message_count=0,
                            created_at=wf_inst.created_at,
                            is_terminal=normalized_status
                            in ("completed", "failed", "cancelled"),
                        )
                    )
                except Exception as e:
                    logger.debug(f"Failed to get workflow activity for {wf_id}: {e}")

        return items

    def _cache_ended_conversation(
        self, api: API, conversation_id: str, instance_id: str = ""
    ) -> None:
        """Cache a conversation snapshot before it's removed from the API."""
        try:
            snap = self._snapshot_from_api(api, conversation_id)
            if snap is not None:
                snap.is_terminal = True
                if instance_id:
                    snap.instance_id = instance_id
                with self._lock:
                    self._ended_conversations[conversation_id] = snap
                    # Evict oldest entries if cache exceeds max size
                    while (
                        len(self._ended_conversations) > self._max_ended_conversations
                    ):
                        self._ended_conversations.popitem(last=False)
        except Exception as e:
            logger.debug(f"Failed to cache ended conversation {conversation_id}: {e}")

    def _snapshot_from_api(
        self, api: API, conversation_id: str
    ) -> ConversationSnapshot | None:
        return snapshot_from_api(
            api,
            conversation_id,
            show_internal_keys=self.config.show_internal_keys,
        )

    # --- FSM Operations ---

    def launch_fsm(
        self,
        preset_id: str | None = None,
        fsm_json: dict[str, Any] | None = None,
        model: str = DEFAULT_LLM_MODEL,
        temperature: float = 0.5,
        label: str = "",
    ) -> ManagedFSM:
        """Launch a new FSM instance from preset or raw JSON."""
        fsm_data = self._resolve_fsm_data(preset_id, fsm_json)
        if fsm_data is None:
            raise ValueError("Must provide either preset_id or fsm_json")
        self._make_room()

        instance_id = str(uuid.uuid4())[:12]
        source = preset_id or "custom"
        if not label:
            label = fsm_data.get("name", f"FSM-{instance_id[:6]}")

        api = API.from_definition(fsm_data, model=model, temperature=temperature)

        # Create per-instance collector and wire to both per-instance and global
        collector = EventCollector(
            max_events=self._config.max_events,
            max_log_lines=self._config.max_log_lines,
        )
        register_monitor_handlers(api, collector)
        register_monitor_handlers(api, self._global_collector)

        managed = ManagedFSM(
            instance_id=instance_id,
            api=api,
            label=label,
            source=source,
        )

        with self._lock:
            self._instances[instance_id] = managed
            self._collectors[instance_id] = collector

        self._emit_global_event(
            EVENT_INSTANCE_LAUNCHED,
            message=f"FSM launched: {label}",
            data={"instance_type": "fsm", "instance_id": instance_id},
        )
        logger.info(f"Launched FSM instance {instance_id}: {label}")
        return managed

    def start_conversation(
        self,
        instance_id: str,
        initial_context: dict[str, Any] | None = None,
    ) -> tuple[str, str]:
        """Start a conversation on a managed FSM instance."""
        with self._lock:
            inst = self._get_fsm(instance_id)
        # Run LLM call outside the lock to avoid blocking other operations
        conv_id, response = inst.api.start_conversation(initial_context or {})
        with self._lock:
            inst.conversation_ids.append(conv_id)
            if inst.status == "completed":
                inst.status = "running"
        return conv_id, response

    def send_message(
        self,
        instance_id: str,
        conversation_id: str,
        message: str,
    ) -> dict[str, Any]:
        """Send a message to a conversation and return response + state info."""
        with self._lock:
            inst = self._get_fsm(instance_id)
        response = inst.api.converse(message, conversation_id)
        current_state = inst.api.get_current_state(conversation_id)
        is_terminal = inst.api.has_conversation_ended(conversation_id)

        if is_terminal:
            # Cache the snapshot, then end the conversation so the core fires
            # END_CONVERSATION (closing its monitor/OTEL span) and frees it.
            self._cache_ended_conversation(inst.api, conversation_id, instance_id)
            try:
                inst.api.end_conversation(conversation_id)
            except Exception as e:
                logger.debug(
                    f"Failed to end terminal conversation {conversation_id}: {e}"
                )
            self._mark_conversation_ended(inst, conversation_id)

        return {
            "response": response,
            "current_state": current_state,
            "is_terminal": is_terminal,
        }

    def end_conversation(self, instance_id: str, conversation_id: str) -> None:
        """End a conversation on a managed FSM instance."""
        with self._lock:
            inst = self._get_fsm(instance_id)
            if conversation_id not in inst.conversation_ids:
                raise KeyError(f"Conversation not found: {conversation_id}")
            already_ended = conversation_id in inst.ended_conversation_ids
        if already_ended:
            return
        # Cache snapshot before the API removes the conversation data
        self._cache_ended_conversation(inst.api, conversation_id, instance_id)
        inst.api.end_conversation(conversation_id)
        # Keep conversation_id in the list so it remains visible in the UI
        self._mark_conversation_ended(inst, conversation_id)

    def _mark_conversation_ended(self, inst: ManagedFSM, conversation_id: str) -> None:
        """Record an ended conversation; complete the instance when every
        conversation it started has ended."""
        with self._lock:
            inst.ended_conversation_ids.add(conversation_id)
            if (
                inst.status == "running"
                and inst.conversation_ids
                and inst.ended_conversation_ids.issuperset(inst.conversation_ids)
            ):
                inst.status = "completed"

    def get_fsm_conversations(self, instance_id: str) -> list[ConversationSnapshot]:
        """Get all conversation snapshots for a managed FSM instance (active + ended)."""
        with self._lock:
            inst = self._get_fsm(instance_id)
            all_ids = list(inst.conversation_ids)
        snapshots = []
        seen = set()
        for conv_id in all_ids:
            snap = self._snapshot_from_api(inst.api, conv_id)
            if snap is not None:
                snap.instance_id = instance_id
                snapshots.append(snap)
                seen.add(conv_id)
            else:
                # Conversation was ended — try cache
                with self._lock:
                    cached = self._ended_conversations.get(conv_id)
                if cached is not None:
                    snapshots.append(cached)
                    seen.add(conv_id)
        return snapshots

    # --- Workflow Operations ---

    def get_workflow_presets(self) -> list[dict[str, str]]:
        """List built-in workflow presets available for launch."""
        presets: list[dict[str, str]] = []
        for preset_id in sorted(_WORKFLOW_PRESETS):
            try:
                definition = _WORKFLOW_PRESETS[preset_id]()
                presets.append(
                    {
                        "id": preset_id,
                        "name": definition.name,
                        "description": definition.description,
                    }
                )
            except Exception as e:  # pragma: no cover - defensive
                logger.debug(f"Failed to build workflow preset {preset_id}: {e}")
        return presets

    def launch_workflow(
        self,
        preset_id: str | None = None,
        definition_json: dict[str, Any] | None = None,
        initial_context: dict[str, Any] | None = None,
        label: str = "",
    ) -> ManagedWorkflow:
        """Launch a workflow engine seeded with a built-in preset definition.

        The returned :class:`ManagedWorkflow` has its ``engine`` populated and the
        preset's definition registered, with ``workflow_id`` set. Callers start an
        instance via :meth:`start_workflow_instance`.

        Arbitrary ``definition_json`` is intentionally unsupported (see
        DECISION plan_2026-05-29_0c00a594/D-002 [STALE]).
        """
        if not _HAS_WORKFLOWS:
            raise NotImplementedError("fsm_llm_workflows extension is not installed")

        if definition_json is not None and preset_id is None:
            raise ValueError(
                "Custom workflow JSON is not supported via the dashboard; "
                "choose a built-in workflow preset instead. "
                f"Available: {', '.join(sorted(_WORKFLOW_PRESETS))}"
            )
        if preset_id is None:
            raise ValueError(
                "A workflow preset_id is required. "
                f"Available: {', '.join(sorted(_WORKFLOW_PRESETS))}"
            )
        if preset_id not in _WORKFLOW_PRESETS:
            raise ValueError(
                f"Unknown workflow preset: {preset_id}. "
                f"Available: {', '.join(sorted(_WORKFLOW_PRESETS))}"
            )

        instance_id = str(uuid.uuid4())[:12]
        if not label:
            label = f"Workflow-{instance_id[:6]}"

        self._make_room()
        engine = WorkflowEngine()
        definition = _WORKFLOW_PRESETS[preset_id]()
        engine.register_workflow(definition)

        collector = EventCollector(
            max_events=self._config.max_events,
            max_log_lines=self._config.max_log_lines,
        )

        managed = ManagedWorkflow(
            instance_id=instance_id,
            label=label,
            source=preset_id or "custom",
        )
        managed.engine = engine
        managed.workflow_id = definition.workflow_id
        try:
            managed.loop = asyncio.get_running_loop()
        except RuntimeError:
            managed.loop = None
        engine.add_hook(self._workflow_hook(managed, collector))

        with self._lock:
            self._instances[instance_id] = managed
            self._collectors[instance_id] = collector

        self._emit_global_event(
            EVENT_INSTANCE_LAUNCHED,
            message=f"Workflow launched: {label}",
            data={"instance_type": "workflow", "instance_id": instance_id},
        )
        logger.info(f"Launched workflow instance {instance_id}: {label}")
        return managed

    def _workflow_hook(
        self, managed: ManagedWorkflow, collector: EventCollector
    ) -> Any:
        """Engine lifecycle hook: one ``workflow_advanced`` event per completed
        step (to the global and the instance collector) and the terminal
        bookkeeping for every run, however it ended (API call, timer, event).

        # DECISION plan-2026-09-28T090000-3c9e41d2/D-006
        # Run completion is tracked HERE, from the engine's status changes.
        # Do NOT go back to checking the status after advance/cancel/event
        # calls: runs that ended any other way (a timer, a failure during
        # start, an event) stayed "active" forever and step metrics counted
        # API calls instead of steps.
        """
        instance_id = managed.instance_id

        def _record(event: MonitorEvent) -> None:
            self._global_collector.record_event(event)
            collector.record_event(event.model_copy())

        def _hook(event_name: str, wf_instance: Any, data: dict[str, Any]) -> None:
            wf_id = str(getattr(wf_instance, "instance_id", ""))
            if event_name == "step_completed":
                _record(
                    MonitorEvent(
                        event_type=EVENT_WORKFLOW_ADVANCED,
                        message=f"Workflow step {data.get('step_id', '')}: {wf_id}",
                        data={
                            "instance_id": instance_id,
                            "workflow_instance_id": wf_id,
                            "step_id": data.get("step_id", ""),
                            "success": data.get("success", True),
                        },
                        level="INFO" if data.get("success", True) else "WARNING",
                    )
                )
                return
            if event_name != "status_changed":
                return
            status = str(data.get("status", ""))
            with self._lock:
                if wf_id in managed.run_statuses:
                    managed.run_statuses[wf_id] = status
                if status in ("completed", "failed", "cancelled"):
                    if wf_id in managed.active_instance_ids:
                        managed.active_instance_ids.remove(wf_id)
                managed.refresh_status()
            terminal_events = {
                "completed": (EVENT_WORKFLOW_COMPLETED, "INFO"),
                "failed": (EVENT_WORKFLOW_FAILED, "ERROR"),
                "cancelled": (EVENT_WORKFLOW_CANCELLED, "INFO"),
            }
            if status in terminal_events:
                event_type, level = terminal_events[status]
                event_data: dict[str, Any] = {
                    "instance_id": instance_id,
                    "workflow_instance_id": wf_id,
                }
                if status == "cancelled":
                    context = getattr(wf_instance, "context", {}) or {}
                    event_data["reason"] = context.get("_cancellation_reason", "")
                if status == "failed":
                    event_data["error"] = str(data.get("error") or "")[:500]
                _record(
                    MonitorEvent(
                        event_type=event_type,
                        message=f"Workflow {status}: {wf_id}",
                        data=event_data,
                        level=level,
                    )
                )

        return _hook

    async def start_workflow_instance(
        self,
        instance_id: str,
        workflow_id: str,
        initial_context: dict[str, Any] | None = None,
    ) -> str:
        """Start a workflow run on a managed workflow engine.

        The run is tracked as active before it starts; the engine hook
        removes it (and emits completed/failed/cancelled) when it ends, which
        for pure auto/condition presets happens inside ``start_workflow``.
        """
        inst = self._get_workflow(instance_id)
        wf_instance_id = str(uuid.uuid4())
        with self._lock:
            inst.run_ids.append(wf_instance_id)
            inst.run_statuses[wf_instance_id] = "running"
            inst.active_instance_ids.append(wf_instance_id)
            inst.refresh_status()
        self._emit_global_event(
            EVENT_WORKFLOW_STARTED,
            message=f"Workflow instance started: {wf_instance_id}",
            data={
                "instance_id": instance_id,
                "workflow_instance_id": wf_instance_id,
            },
        )
        try:
            await inst.engine.start_workflow(
                workflow_id=workflow_id,
                initial_context=initial_context or {},
                instance_id=wf_instance_id,
            )
        finally:
            # A run that failed before the engine registered it never reports
            # a status change: settle it here.
            if inst.engine.get_workflow_instance(wf_instance_id) is None:
                with self._lock:
                    inst.run_statuses[wf_instance_id] = "failed"
                    if wf_instance_id in inst.active_instance_ids:
                        inst.active_instance_ids.remove(wf_instance_id)
                    inst.refresh_status()
        return wf_instance_id

    def _validate_workflow_instance_id(
        self, inst: Any, instance_id: str, wf_instance_id: str
    ) -> None:
        """Validate that a workflow instance ID belongs to the managed instance."""
        if (
            hasattr(inst, "active_instance_ids")
            and wf_instance_id not in inst.active_instance_ids
        ):
            raise KeyError(
                f"Workflow instance '{wf_instance_id}' not found in "
                f"managed instance '{instance_id}'"
            )

    async def advance_workflow(
        self,
        instance_id: str,
        wf_instance_id: str,
        user_input: str = "",
    ) -> bool:
        """Advance a workflow run (step and completion events come from the
        engine hook)."""
        inst = self._get_workflow(instance_id)
        self._validate_workflow_instance_id(inst, instance_id, wf_instance_id)
        result = await inst.engine.advance_workflow(wf_instance_id, user_input)
        return bool(result)

    async def send_workflow_event(
        self,
        instance_id: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
        wf_instance_id: str | None = None,
    ) -> list[str]:
        """Deliver an event to a managed workflow engine.

        With ``wf_instance_id`` the event targets that workflow instance only
        (and is buffered for it if it is not waiting yet); otherwise it is
        broadcast to every instance waiting for ``event_type``. Returns the
        workflow instance ids the event woke.
        """
        inst = self._get_workflow(instance_id)
        if wf_instance_id:
            self._validate_workflow_instance_id(inst, instance_id, wf_instance_id)
        event = WorkflowEvent(
            event_type=event_type,
            payload=payload or {},
            instance_id=wf_instance_id or None,
        )
        affected = await inst.engine.process_event(event)

        self._emit_global_event(
            EVENT_WORKFLOW_EVENT_DELIVERED,
            message=f"Workflow event delivered: {event_type}",
            data={
                "instance_id": instance_id,
                "event_type": event_type,
                "affected": list(affected),
            },
        )
        return list(affected)

    async def cancel_workflow(
        self,
        instance_id: str,
        wf_instance_id: str,
        reason: str = "",
    ) -> bool:
        """Cancel a workflow instance."""
        inst = self._get_workflow(instance_id)
        self._validate_workflow_instance_id(inst, instance_id, wf_instance_id)
        result = await inst.engine.cancel_workflow(wf_instance_id, reason=reason)
        return bool(result)

    def get_workflow_status(
        self, instance_id: str, wf_instance_id: str
    ) -> dict[str, Any]:
        """Get workflow instance status and context."""
        inst = self._get_workflow(instance_id)
        # Accept ids the engine still knows about (e.g. just-completed/cancelled
        # instances removed from active_instance_ids) so status stays queryable;
        # only fall back to the active-list validator if the engine has no record.
        wf_instance = inst.engine.get_workflow_instance(wf_instance_id)
        if wf_instance is None:
            self._validate_workflow_instance_id(inst, instance_id, wf_instance_id)
            raise KeyError(f"Workflow run not found: {wf_instance_id}")
        drop_internal = not self.config.show_internal_keys
        status = inst.engine.get_workflow_status(wf_instance_id)
        context = inst.engine.get_workflow_context(wf_instance_id)
        status_str = _status_str(status).lower()

        # Extract step history if available
        history: list[dict[str, Any]] = []
        raw_history = getattr(wf_instance, "history", [])
        for entry in raw_history:
            history.append(
                {
                    "step_id": getattr(entry, "step_id", ""),
                    "message": getattr(entry, "message", ""),
                    "timestamp": str(getattr(entry, "timestamp", "")),
                    "data": redact_context(
                        getattr(entry, "data", None) or {}, drop_internal
                    ),
                    "error": getattr(entry, "error", None),
                }
            )

        return {
            "workflow_instance_id": wf_instance_id,
            "status": status_str,
            "current_step": getattr(wf_instance, "current_step_id", ""),
            "context": redact_context(context or {}, drop_internal),
            "history": history,
            "created_at": str(getattr(wf_instance, "created_at", "")),
            "updated_at": str(getattr(wf_instance, "updated_at", "")),
        }

    def get_workflow_instances(self, instance_id: str) -> list[dict[str, Any]]:
        """List all workflow instances on a managed workflow engine."""
        inst = self._get_workflow(instance_id)
        results: list[dict[str, Any]] = []
        for wf_id in inst.active_instance_ids:
            try:
                status_data = self.get_workflow_status(instance_id, wf_id)
                results.append(status_data)
            except Exception as e:
                logger.debug(f"Failed to get workflow status for {wf_id}: {e}")
                results.append({"workflow_instance_id": wf_id, "status": "unknown"})
        return results

    # --- Agent Operations ---

    def launch_agent(
        self,
        agent_type: str = "ReactAgent",
        task: str = "",
        tools_config: list[StubToolConfig] | None = None,
        model: str = DEFAULT_LLM_MODEL,
        max_iterations: int = 10,
        timeout_seconds: float = 120.0,
        label: str = "",
    ) -> ManagedAgent:
        """Launch an agent in a background thread."""
        if not _HAS_AGENTS:
            raise NotImplementedError("fsm_llm_agents extension is not installed")

        if agent_type not in _AGENT_CLASSES:
            raise ValueError(
                f"Unknown agent type: {agent_type}. "
                f"Available: {', '.join(sorted(_AGENT_CLASSES))}"
            )

        needs_tools = agent_type in _TOOL_BASED_AGENTS
        if needs_tools and not tools_config:
            raise ValueError(
                f"{agent_type} requires at least one tool to be configured"
            )

        with self._lock:
            running = sum(
                1
                for inst in self._instances.values()
                if isinstance(inst, ManagedAgent)
                and inst.status in ("running", "cancelling")
            )
        if running >= self._config.max_running_agents:
            raise MonitorCapacityError(
                f"{running} agents are already running "
                f"(max_running_agents={self._config.max_running_agents})"
            )
        self._make_room()

        instance_id = str(uuid.uuid4())[:12]
        if not label:
            label = f"{agent_type}-{instance_id[:6]}"

        # Build tool registry from stub configs (only for tool-based agents)
        registry: ToolRegistry | None = None
        if needs_tools and tools_config:
            registry = ToolRegistry()
            for tool_cfg in tools_config:
                stub_response = tool_cfg.stub_response

                def _make_stub(resp: str) -> Any:
                    def stub_fn(**kwargs: Any) -> str:
                        return resp

                    return stub_fn

                registry.register_function(
                    _make_stub(stub_response),
                    name=tool_cfg.name,
                    description=tool_cfg.description,
                )

        # Create per-instance collector
        collector = EventCollector(
            max_events=self._config.max_events,
            max_log_lines=self._config.max_log_lines,
        )

        # Build monitor handlers to inject into the agent's internal API
        monitor_handlers = _build_monitor_handlers(collector)
        global_handlers = _build_monitor_handlers(self._global_collector)

        config = AgentConfig(
            model=model,
            max_iterations=max_iterations,
            timeout_seconds=timeout_seconds,
        )

        managed = ManagedAgent(
            instance_id=instance_id,
            agent_type=agent_type,
            task=task,
            label=label,
        )
        managed.max_iterations = max_iterations

        # Build conversation-capturing handlers (need managed reference)
        conv_capture_handlers = _build_context_capture_handlers(collector, managed)
        all_handlers = monitor_handlers + global_handlers + conv_capture_handlers

        with self._lock:
            self._instances[instance_id] = managed
            self._collectors[instance_id] = collector

        self._emit_global_event(
            EVENT_INSTANCE_LAUNCHED,
            message=f"Agent launched: {label}",
            data={
                "instance_type": "agent",
                "instance_id": instance_id,
                "agent_type": agent_type,
            },
        )
        self._emit_global_event(
            EVENT_AGENT_STARTED,
            message=f"Agent started: {task[:100]}",
            data={"instance_id": instance_id, "task": task},
        )

        # Launch in background thread
        def _run_agent() -> None:
            try:
                if managed.cancel_event.is_set():
                    with self._lock:
                        managed.status = "cancelled"
                    return

                agent_cls = _AGENT_CLASSES[agent_type]
                kwargs: dict[str, Any] = {
                    "config": config,
                    "handlers": all_handlers,
                }
                if needs_tools and registry is not None:
                    kwargs["tools"] = registry
                agent = agent_cls(**kwargs)
                result = agent.run(task)

                with self._lock:
                    # agent.run() cannot be interrupted: keep its result even
                    # when a cancel arrived while it ran.
                    managed.result = result
                    if managed.cancel_event.is_set():
                        managed.status = "cancelled"
                        return
                    managed.status = "completed" if result.success else "failed"

                # Emit fine-grained iteration/tool events for metrics
                if hasattr(result, "trace") and result.trace:
                    trace = result.trace
                    total_iters = getattr(trace, "total_iterations", 0)
                    for i in range(total_iters):
                        self._emit_global_event(
                            EVENT_AGENT_ITERATION,
                            message=f"Agent iteration {i + 1}: {label}",
                            data={"instance_id": instance_id, "iteration": i + 1},
                        )
                    tool_calls = getattr(trace, "tool_calls", [])
                    for tc in tool_calls:
                        tool_name = getattr(tc, "tool_name", "")
                        self._emit_global_event(
                            EVENT_AGENT_TOOL_CALL,
                            message=f"Tool call: {tool_name}",
                            data={
                                "instance_id": instance_id,
                                "tool_name": tool_name,
                            },
                        )

                event_type = (
                    EVENT_AGENT_COMPLETED if result.success else EVENT_AGENT_FAILED
                )
                self._emit_global_event(
                    event_type,
                    message=f"Agent {'completed' if result.success else 'failed'}: {label}",
                    data={
                        "instance_id": instance_id,
                        "success": result.success,
                        "answer": result.answer[:200] if result.answer else "",
                    },
                )
            except Exception as e:
                with self._lock:
                    cancelled = managed.cancel_event.is_set()
                    if cancelled:
                        managed.status = "cancelled"
                    else:
                        managed.status = "failed"
                        managed.error = str(e)[:1000]
                # cancel_agent already emitted agent_cancelled; do not count
                # the cancellation twice.
                if not cancelled:
                    self._emit_global_event(
                        EVENT_AGENT_FAILED,
                        message=f"Agent failed: {label} ({type(e).__name__})",
                        data={"instance_id": instance_id, "error": str(e)[:500]},
                        level="ERROR",
                    )

        thread = threading.Thread(
            target=_run_agent, name=f"agent-{instance_id}", daemon=True
        )
        managed.thread = thread
        thread.start()

        logger.info(f"Launched agent {instance_id}: {label}")
        return managed

    def cancel_agent(self, instance_id: str) -> bool:
        """Signal an agent to cancel.

        Returns False (and changes nothing) when the agent has already
        finished: a completed or failed agent keeps its status and result.
        """
        inst = self._get_agent(instance_id)
        with self._lock:
            finished = inst.status in ("completed", "failed", "cancelled") or (
                inst.thread is not None and not inst.thread.is_alive()
            )
            if finished or inst.status == "cancelling":
                return False
            inst.cancel_event.set()
            inst.status = "cancelling"
        self._emit_global_event(
            EVENT_AGENT_CANCELLED,
            message=f"Agent cancelled: {inst.label}",
            data={"instance_id": instance_id},
        )
        return True

    def get_agent_status(self, instance_id: str) -> dict[str, Any]:
        """Get agent status including real-time progress and partial results."""
        inst = self._get_agent(instance_id)

        # Reconcile a dead thread and snapshot all mutable fields under one lock
        # so status/result/error are mutually consistent for the rest of this call
        # (DECISION plan_2026-05-29_0c00a594/D-003 [STALE]). A thread that died while
        # "running" or "cancelling" is resolved here so it never stays stuck.
        with self._lock:
            if (
                inst.thread
                and not inst.thread.is_alive()
                and inst.status
                in (
                    "running",
                    "cancelling",
                )
            ):
                if inst.cancel_event.is_set():
                    inst.status = "cancelled"
                else:
                    inst.status = "completed" if inst.result is not None else "failed"
            status = inst.status
            inst_result = inst.result
            inst_error = inst.error

        result: dict[str, Any] = {
            "instance_id": instance_id,
            "agent_type": inst.agent_type,
            "task": inst.task,
            "status": status,
            "created_at": str(inst.created_at),
            "max_iterations": inst.max_iterations,
        }

        with inst._conv_lock:
            conversation_log = list(inst.conversation_log)

        # Derive real-time progress from per-instance collector events
        collector = self._collectors.get(instance_id)
        if collector and status == "running":
            events = collector.get_events(limit=0)
            transitions = [e for e in events if e.event_type == "state_transition"]
            transition_count = len(transitions)
            # get_events is newest first.
            current_state = next(
                (e.target_state for e in transitions if e.target_state), ""
            )
            # Count iterations by counting entries into the "think" state,
            # which is universal across agent patterns. Fall back to the
            # heuristic (transition_count+1)//2 for non-standard patterns.
            think_count = sum(1 for e in transitions if e.target_state == "think")
            iteration_count = (
                think_count if think_count > 0 else (transition_count + 1) // 2
            )
            last_tool = ""
            for entry in reversed(conversation_log):
                tool = (entry.get("data") or {}).get("tool_name")
                if tool:
                    last_tool = str(tool)
                    break
            result["current_state"] = current_state
            result["iteration_count"] = iteration_count
            result["last_tool_call"] = last_tool
            result["transition_count"] = transition_count

        if inst_result is not None:
            result["answer"] = getattr(inst_result, "answer", "")
            result["success"] = getattr(inst_result, "success", False)
            if hasattr(inst_result, "trace") and inst_result.trace:
                trace = inst_result.trace
                result["total_iterations"] = getattr(trace, "total_iterations", 0)
                tool_calls = getattr(trace, "tool_calls", [])
                result["tools_used"] = _tools_used(tool_calls)

        if inst_error:
            result["error"] = inst_error

        # Live conversation log (captured and redacted by the capture handlers)
        result["conversation_log"] = conversation_log

        return result

    def get_agent_result(self, instance_id: str) -> dict[str, Any]:
        """Get final agent result (if complete) with full trace steps."""
        inst = self._get_agent(instance_id)
        # Snapshot under one lock so status/result/error agree, and distinguish
        # "not finished yet" from "finished with no result / failed"
        # (DECISION plan_2026-05-29_0c00a594/D-003 [STALE]).
        with self._lock:
            inst_result = inst.result
            inst_status = inst.status
            inst_error = inst.error
        if inst_result is None:
            return {
                "status": inst_status,
                "error": inst_error or "Agent has not completed yet",
            }

        result: dict[str, Any] = {
            "status": inst_status,
            "answer": getattr(inst_result, "answer", ""),
            "success": getattr(inst_result, "success", False),
            "final_context": redact_context(
                getattr(inst_result, "final_context", None) or {}
            ),
        }
        if hasattr(inst_result, "trace") and inst_result.trace:
            trace = inst_result.trace
            result["total_iterations"] = getattr(trace, "total_iterations", 0)
            tool_calls = getattr(trace, "tool_calls", [])
            result["tools_used"] = _tools_used(tool_calls)
            # Full trace steps for visualization
            steps = getattr(trace, "steps", [])
            result["trace_steps"] = [
                {
                    "state": getattr(s, "state", ""),
                    "reasoning": getattr(s, "reasoning", ""),
                    "tool_name": getattr(s, "tool_name", ""),
                    "tool_input": redact_context(getattr(s, "tool_input", "")),
                    "tool_result": getattr(s, "observation", ""),
                    "timestamp": str(getattr(s, "timestamp", "")),
                }
                for s in steps
            ]
        return result

    # --- Destroy ---

    def destroy_instance(self, instance_id: str) -> None:
        """Destroy a managed instance and clean up resources."""
        with self._lock:
            inst = self._instances.pop(instance_id, None)
            self._collectors.pop(instance_id, None)
            if inst is not None:
                if isinstance(inst, ManagedAgent):
                    inst.cancel_event.set()
                    if inst.status in ("running", "cancelling"):
                        inst.status = "cancelled"
                elif inst.status == "running":
                    inst.status = "completed"

        if inst is None:
            raise KeyError(f"Instance not found: {instance_id}")

        if isinstance(inst, ManagedFSM):
            # Detach the global collector's handlers (the API object may be
            # shared by the caller) and end every conversation.
            unregister_monitor_handlers(inst.api, self._global_collector)
            try:
                inst.api.close()
            except Exception as e:
                logger.debug(f"Failed to close FSM API during instance destroy: {e}")
        elif isinstance(inst, ManagedWorkflow):
            # Stop the engine's timers and background runs on the loop that
            # owns them (destroy may run in a worker thread).
            engine, loop = inst.engine, inst.loop
            if engine is not None and loop is not None and not loop.is_closed():
                try:
                    running = asyncio.get_running_loop()
                except RuntimeError:
                    running = None
                if running is loop:
                    task = loop.create_task(engine.shutdown())
                    _pending_tasks.add(task)
                    task.add_done_callback(_pending_tasks.discard)
                else:
                    asyncio.run_coroutine_threadsafe(engine.shutdown(), loop)
        elif isinstance(inst, ManagedAgent):
            # DECISION plan-2026-09-12T065608-089d0ec7/D-007
            # `agent.run()` has no cancellation hook (see base.py's `run()`
            # abstract signature) so setting cancel_event cannot interrupt
            # in-flight work — do NOT assume the background thread stops
            # here. Bound our OWN wait instead: join with a short timeout so
            # destroy_instance (called synchronously from an async FastAPI
            # route) never blocks indefinitely, and log a warning if the
            # thread is still alive after the timeout so the leak stays
            # observable. True mid-run cancellation remains OUT OF SCOPE —
            # see decisions.md D-007.
            if inst.thread is not None and inst.thread.is_alive():
                inst.thread.join(timeout=1.5)
                if inst.thread.is_alive():
                    logger.warning(
                        f"Agent {instance_id} did not stop within timeout "
                        "after cancellation; background thread still running"
                    )

        self._emit_global_event(
            EVENT_INSTANCE_DESTROYED,
            message=f"Instance destroyed: {inst.label}",
            data={
                "instance_type": inst.instance_type,
                "instance_id": instance_id,
            },
        )

    def _make_room(self) -> None:
        """Keep at most ``max_instances`` instances: evict the oldest
        finished ones, and refuse a launch when everything is still active."""
        with self._lock:
            excess = len(self._instances) + 1 - self._config.max_instances
            if excess <= 0:
                return
            finished = sorted(
                (
                    inst
                    for inst in self._instances.values()
                    if inst.status in ("completed", "failed", "cancelled")
                ),
                key=lambda i: i.created_at,
            )
            victims = [i.instance_id for i in finished[:excess]]
        for victim in victims:
            try:
                self.destroy_instance(victim)
            except KeyError:
                pass
        with self._lock:
            if len(self._instances) >= self._config.max_instances:
                raise MonitorCapacityError(
                    f"{len(self._instances)} instances are active "
                    f"(max_instances={self._config.max_instances}); destroy one first"
                )

    # --- Private Helpers ---

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

    def _resolve_fsm_data(
        self,
        preset_id: str | None,
        fsm_json: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Resolve FSM definition from preset ID or raw JSON."""
        if fsm_json is not None:
            return fsm_json

        if preset_id is not None:
            base = _find_examples_dir()
            if base is None:
                raise FileNotFoundError("Examples directory not found")
            file_path = validate_preset_id(preset_id, base)
            result: dict[str, Any] = json.loads(file_path.read_text())
            return result

        return None

    def _emit_global_event(
        self,
        event_type: str,
        message: str = "",
        data: dict[str, Any] | None = None,
        level: str = "INFO",
    ) -> None:
        """Emit an event to the global collector."""
        self._global_collector.record_event(
            MonitorEvent(
                event_type=event_type,
                message=message,
                data=data or {},
                level=level,
            )
        )
