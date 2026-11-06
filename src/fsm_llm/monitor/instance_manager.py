"""
Instance manager for fsm_llm.monitor.

Manages multiple concurrent FSM, workflow, and agent instances with
per-instance event collection and lifecycle management.
"""

from __future__ import annotations

import asyncio
import collections
import contextvars
import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

_T = TypeVar("_T")

from fsm_llm import API, HandlerTiming
from fsm_llm.agents import (
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
from fsm_llm.constants import DEFAULT_LLM_MODEL
from fsm_llm.handlers import BaseHandler
from fsm_llm.logging import logger
from fsm_llm.security import resolve_within
from fsm_llm.workflows import (
    WorkflowDefinition,
    WorkflowEngine,
    WorkflowEvent,
    WorkflowStep,
    WorkflowStepResult,
    auto_step,
    condition_step,
    create_workflow,
)

from .collector import EventCollector, redact_context
from .constants import (
    ACTIVE_AGENT_STATUSES,
    DEFAULT_AGENT_MAX_ITERATIONS,
    DEFAULT_AGENT_TIMEOUT_SECONDS,
    DEFAULT_FSM_TEMPERATURE,
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
    TERMINAL_STATUSES,
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
from .exceptions import (
    MonitorCapacityError,
    MonitorConnectionError,
    MonitorInitializationError,
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


class _ManagedBase:
    """The fields every managed instance shares."""

    instance_type = ""

    def __init__(self, instance_id: str, label: str = "") -> None:
        self.instance_id = instance_id
        self.label = label
        self.status = "running"
        self.created_at = datetime.now(timezone.utc)

    def _info(self, **extra: Any) -> InstanceInfo:
        return InstanceInfo(
            instance_id=self.instance_id,
            instance_type=self.instance_type,
            label=self.label,
            status=self.status,
            created_at=self.created_at,
            **extra,
        )


class ManagedFSM(_ManagedBase):
    """A managed FSM API instance."""

    instance_type = "fsm"

    def __init__(
        self,
        instance_id: str,
        api: API,
        label: str = "",
        source: str = "custom",
    ) -> None:
        super().__init__(instance_id, label)
        self.api = api
        self.source = source
        self.conversation_ids: list[str] = []
        self.ended_conversation_ids: set[str] = set()

    def to_info(self) -> InstanceInfo:
        return self._info(
            source=self.source, conversation_count=len(self.conversation_ids)
        )


class ManagedWorkflow(_ManagedBase):
    """A managed workflow engine instance."""

    instance_type = "workflow"

    def __init__(
        self,
        instance_id: str,
        label: str = "",
        source: str = "custom",
    ) -> None:
        super().__init__(instance_id, label)
        self.source = source
        self.engine: Any = None  # WorkflowEngine
        self.workflow_id: str = ""
        self.active_instance_ids: list[str] = []
        # Every run started on this engine, in order, and its last status.
        self.run_ids: list[str] = []
        self.run_statuses: dict[str, str] = {}
        # Event loop the engine's timers run on (set at launch when one is
        # running, and by start_workflow_instance, which is where it matters).
        self.loop: asyncio.AbstractEventLoop | None = None

    def to_info(self) -> InstanceInfo:
        return self._info(
            source=self.source, active_workflows=len(self.active_instance_ids)
        )

    def refresh_status(self) -> None:
        """``running`` while any run is active, else the last run's status."""
        if self.active_instance_ids or not self.run_statuses:
            self.status = "running"
            return
        self.status = self.run_statuses[self.run_ids[-1]]


class ManagedAgent(_ManagedBase):
    """A managed agent instance running in a background thread."""

    instance_type = "agent"

    def __init__(
        self,
        instance_id: str,
        agent_type: str = "ReactAgent",
        task: str = "",
        label: str = "",
        max_iterations: int = DEFAULT_AGENT_MAX_ITERATIONS,
    ) -> None:
        super().__init__(instance_id, label)
        self.agent_type = agent_type
        self.task = task
        self.thread: threading.Thread | None = None
        self.cancel_event = threading.Event()
        self.max_iterations = max_iterations
        self.result: Any = None  # AgentResult
        self.error: str | None = None
        # Live conversation log — populated by context capture handlers
        self.conversation_log: list[dict[str, Any]] = []
        self._conv_lock = threading.Lock()

    def to_info(self) -> InstanceInfo:
        return self._info(agent_type=self.agent_type, task=self.task[:200])


ManagedInstance = ManagedFSM | ManagedWorkflow | ManagedAgent


# The (current_state, target_state) the core passed to the last ``should_execute``
# of this context; ``execute`` reads it back (D-003 below). A ContextVar, not a
# ``threading.local``: with ``handler_timeout`` the core runs ``execute`` on a
# worker thread that starts from a copy of the caller's context.
_HANDLER_STATES: contextvars.ContextVar[tuple[str | None, str | None]] = (
    contextvars.ContextVar("monitor_handler_states", default=(None, None))
)


class _MonitorHandler(BaseHandler):
    """Observer handler that hands the core's state arguments to a monitor
    callback.

    # DECISION plan-2026-09-28T090000-3c9e41d2/D-003
    # The core passes ``current_state``/``target_state`` to ``should_execute``
    # only; the handler context has no ``_target_state``. Do NOT go back to a
    # ``create_handler(...).do(callback)`` lambda that reads the states from
    # the context: every transition event then had an empty target. The
    # states are kept in ``_HANDLER_STATES`` between ``should_execute`` and
    # ``execute`` (the core calls them back to back). Do NOT go back to a
    # ``threading.local``: with ``handler_timeout`` set ``execute`` runs on a
    # worker thread and read ``None, None`` (plan-2026-11-05T185903-5980fbb7,
    # F-HV-01); the worker starts from ``contextvars.copy_context()``.
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
        _HANDLER_STATES.set((current_state, target_state))
        return True

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        current, target = _HANDLER_STATES.get()
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


def _agent_result_fields(result: Any) -> dict[str, Any]:
    """Answer and success of an ``AgentResult`` and, when it carries a trace,
    its iteration count and redacted tool calls."""
    fields: dict[str, Any] = {
        "answer": getattr(result, "answer", ""),
        "success": getattr(result, "success", False),
    }
    trace = getattr(result, "trace", None)
    if trace:
        fields["total_iterations"] = getattr(trace, "total_iterations", 0)
        fields["tools_used"] = _tools_used(getattr(trace, "tool_calls", []))
    return fields


def _stub_fn(response: str) -> Any:
    def stub_fn(**kwargs: Any) -> str:
        return response

    return stub_fn


def _stub_registry(tools_config: list[StubToolConfig]) -> ToolRegistry:
    """A registry of tools that return their configured ``stub_response``."""
    registry = ToolRegistry()
    for tool_cfg in tools_config:
        registry.register_function(
            _stub_fn(tool_cfg.stub_response),
            name=tool_cfg.name,
            description=tool_cfg.description,
        )
    return registry


def _new_instance_id() -> str:
    return str(uuid.uuid4())[:12]


def find_examples_dir() -> Path | None:
    """Locate the examples/ directory."""
    # parents[3] is the repo root: src/fsm_llm/monitor/instance_manager.py.
    base = Path(__file__).resolve().parents[3] / "examples"
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
    # The confinement test is the shared `security.resolve_within`; the lexical
    # pre-check above stays, as it also refuses `sub/../x.json`, which would
    # land inside.
    try:
        confined = resolve_within(base.resolve(), preset_id)
    except ValueError as e:  # an embedded NUL byte
        raise ValueError("Invalid preset ID") from e
    if confined is None:
        raise ValueError("Invalid preset ID")
    file_path = base / preset_id
    if not file_path.exists():
        raise FileNotFoundError(f"Preset not found: {preset_id}")
    return file_path


def read_json_file(path: Path) -> Any:
    """Parse a JSON file as UTF-8, whatever the locale (the one reader of the
    monitor's presets and ``flows.json``)."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_preset_json(preset_id: str) -> dict[str, Any]:
    """Read an example FSM preset by ID.

    Raises ``FileNotFoundError`` (no examples directory, or no such preset),
    ``ValueError`` (an ID that escapes the examples directory) or the JSON and
    OS errors of reading the file.
    """
    base = find_examples_dir()
    if base is None:
        raise FileNotFoundError("Examples directory not found")
    result: dict[str, Any] = read_json_file(validate_preset_id(preset_id, base))
    return result


def snapshot_from_api(
    api: API,
    conversation_id: str,
    show_internal_keys: bool = False,
) -> ConversationSnapshot | None:
    """Build a ConversationSnapshot from a live API instance.

    Returns ``None`` when the conversation is unknown or cannot be read.
    """
    # DECISION plan-2026-09-12T135914-45a654de/D-017
    # The default is fail-CLOSED (show_internal_keys=False) so a caller that
    # forgets the parameter cannot leak internal context keys into the
    # dashboard. Do NOT flip it to True "for convenience"; a caller that wants
    # internal keys passes the config flag explicitly.
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
        self._global_collector = self._new_collector()
        self._lock = threading.RLock()

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

    def shutdown(self) -> None:
        """Detach from every API, tear every instance down like
        ``destroy_instance`` (agents are signalled, not joined) and remove the
        log sink (the manager can no longer be used afterwards)."""
        with self._lock:
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

    def _close_fsm_api(self, inst: ManagedFSM, instance_id: str) -> None:
        """End every live conversation of a manager-owned FSM API, then detach
        the global collector's handlers.

        # DECISION plan-2026-11-05T134105-5ed71cf2/D-015
        # Close BEFORE unregistering: ``api.close()`` fires END_CONVERSATION,
        # and an inactive handler records nothing, so the conversation stayed
        # "active" on the global collector and its OTEL span was never ended.
        # Do NOT swap the two calls back. The unregister stays in a
        # ``finally`` so a failing close cannot leave the handlers on. Live
        # conversations are cached first so the Conversations view keeps
        # them, as for a conversation ended through ``end_conversation``.
        # ``attach_api`` (a caller-owned API) is left alone on purpose:
        # closing a caller's API is a policy choice.
        """
        api = inst.api
        try:
            for conversation_id in api.list_active_conversations():
                self._cache_ended_conversation(api, conversation_id, instance_id)
            api.close()
        except Exception as e:
            logger.debug(f"Failed to close FSM API of instance {instance_id}: {e}")
        finally:
            unregister_monitor_handlers(api, self._global_collector)

    # --- Capabilities ---

    def get_capabilities(self) -> dict[str, bool]:
        return {
            "fsm": True,
            "workflows": True,
            "agents": True,
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

    # --- Conversation queries (all FSM instances + the attached API) ---

    def get_active_conversations(self) -> list[str]:
        result: list[str] = []
        # From the attached API (snapshot the reference under lock; D-003)
        with self._lock:
            attached_api = self._attached_api
        if attached_api is not None:
            try:
                result.extend(attached_api.list_active_conversations())
            except Exception as e:
                logger.debug(f"Failed to list attached API conversations: {e}")
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

    def get_conversation_snapshot(
        self, conversation_id: str
    ) -> ConversationSnapshot | None:
        """Find and return a conversation snapshot from any FSM instance."""
        # Check the attached API first (snapshot the reference under lock; D-003)
        with self._lock:
            attached_api = self._attached_api
        if attached_api is not None:
            snap = self._snapshot_from_api(attached_api, conversation_id)
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
        return self._ended_snapshot(conversation_id)

    def _ended_snapshot(self, conversation_id: str) -> ConversationSnapshot | None:
        with self._lock:
            return self._ended_conversations.get(conversation_id)

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
                (inst.instance_id, conv_id)
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
                    is_terminal=inst.status in TERMINAL_STATUSES,
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
                            is_terminal=normalized_status in TERMINAL_STATUSES,
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
        temperature: float = DEFAULT_FSM_TEMPERATURE,
        label: str = "",
    ) -> ManagedFSM:
        """Launch a new FSM instance from preset or raw JSON."""
        fsm_data = self._resolve_fsm_data(preset_id, fsm_json)
        if fsm_data is None:
            raise ValueError("Must provide either preset_id or fsm_json")
        instance_id = _new_instance_id()
        api = API.from_definition(fsm_data, model=model, temperature=temperature)

        # Per-instance collector, wired next to the global one
        collector = self._new_collector()
        register_monitor_handlers(api, collector)
        register_monitor_handlers(api, self._global_collector)

        managed = ManagedFSM(
            instance_id=instance_id,
            api=api,
            label=label or fsm_data.get("name", f"FSM-{instance_id[:6]}"),
            source=preset_id or "custom",
        )
        self._register_instance(managed, collector, "FSM")
        return managed

    def _register_instance(
        self,
        managed: ManagedInstance,
        collector: EventCollector,
        noun: str,
        **launched_data: Any,
    ) -> None:
        """Make room for ``managed``, add it with its collector and announce
        the launch.

        # DECISION plan-2026-11-05T185903-5980fbb7/D-005
        # Callers build and validate everything that can fail BEFORE this.
        # Do NOT move ``_make_room`` back to the top of a ``launch_*``: a
        # request that then failed validation had already destroyed the oldest
        # finished instance and its history.
        """
        self._make_room()
        instance_id = managed.instance_id
        with self._lock:
            self._instances[instance_id] = managed
            self._collectors[instance_id] = collector
        self._emit_event(
            EVENT_INSTANCE_LAUNCHED,
            message=f"{noun} launched: {managed.label}",
            data={
                "instance_type": managed.instance_type,
                "instance_id": instance_id,
                **launched_data,
            },
        )
        logger.info(
            f"Launched {managed.instance_type} instance {instance_id}: {managed.label}"
        )

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
            # The reply is already produced: a refusal to end is logged, the
            # conversation stays open and a later end_conversation retries.
            try:
                self._finish_conversation(inst, conversation_id)
            except Exception as e:
                logger.debug(
                    f"Failed to end terminal conversation {conversation_id}: {e}"
                )

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
        if not already_ended:
            self._finish_conversation(inst, conversation_id)

    def _finish_conversation(self, inst: ManagedFSM, conversation_id: str) -> None:
        """Cache the snapshot, end the conversation in the core (it fires
        END_CONVERSATION, closing the monitor/OTEL span, and frees it), and only
        then record it as ended: a refusal (``ConversationBusyError``)
        propagates and leaves it unmarked. The id stays in ``conversation_ids``
        so the conversation remains visible in the UI."""
        self._cache_ended_conversation(inst.api, conversation_id, inst.instance_id)
        inst.api.end_conversation(conversation_id)
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
        for conv_id in all_ids:
            snap = self._snapshot_from_api(inst.api, conv_id)
            if snap is not None:
                snap.instance_id = instance_id
            else:
                snap = self._ended_snapshot(conv_id)  # ended: try the cache
            if snap is not None:
                snapshots.append(snap)
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
        label: str = "",
    ) -> ManagedWorkflow:
        """Launch a workflow engine seeded with a built-in preset definition.

        The returned :class:`ManagedWorkflow` has its ``engine`` populated and the
        preset's definition registered, with ``workflow_id`` set. Callers start an
        instance via :meth:`start_workflow_instance`.

        Arbitrary ``definition_json`` is intentionally unsupported (see
        DECISION plan_2026-05-29_0c00a594/D-002 [STALE]).
        """
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

        instance_id = _new_instance_id()
        engine = WorkflowEngine()
        definition = _WORKFLOW_PRESETS[preset_id]()
        engine.register_workflow(definition)

        managed = ManagedWorkflow(
            instance_id=instance_id,
            label=label or f"Workflow-{instance_id[:6]}",
            source=preset_id,
        )
        managed.engine = engine
        managed.workflow_id = definition.workflow_id
        try:
            managed.loop = asyncio.get_running_loop()
        except RuntimeError:
            managed.loop = None
        engine.add_hook(self._workflow_hook(managed))
        self._register_instance(managed, self._new_collector(), "Workflow")
        return managed

    def _workflow_hook(self, managed: ManagedWorkflow) -> Any:
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

        _record = self._record_event

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
                if status in TERMINAL_STATUSES:
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
        # The engine's timers and background runs live on THIS loop. The
        # launch route builds the instance in a worker thread (no loop there),
        # so record the owning loop here for ``_teardown``.
        inst.loop = asyncio.get_running_loop()
        wf_instance_id = str(uuid.uuid4())
        with self._lock:
            inst.run_ids.append(wf_instance_id)
            inst.run_statuses[wf_instance_id] = "running"
            inst.active_instance_ids.append(wf_instance_id)
            inst.refresh_status()
        self._emit_event(
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
        self, inst: ManagedWorkflow, instance_id: str, wf_instance_id: str
    ) -> None:
        """Validate that a workflow instance ID belongs to the managed instance."""
        if wf_instance_id not in inst.active_instance_ids:
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

        self._emit_event(
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
        max_iterations: int = DEFAULT_AGENT_MAX_ITERATIONS,
        timeout_seconds: float = DEFAULT_AGENT_TIMEOUT_SECONDS,
        label: str = "",
    ) -> ManagedAgent:
        """Launch an agent in a background thread."""
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
        self._check_agent_capacity()

        instance_id = _new_instance_id()
        registry = _stub_registry(tools_config or []) if needs_tools else None
        collector = self._new_collector()
        config = AgentConfig(
            model=model,
            max_iterations=max_iterations,
            timeout_seconds=timeout_seconds,
        )
        managed = ManagedAgent(
            instance_id=instance_id,
            agent_type=agent_type,
            task=task,
            label=label or f"{agent_type}-{instance_id[:6]}",
            max_iterations=max_iterations,
        )
        # Handlers injected into the agent's internal API: the instance's and
        # the global collector's events, and the conversation capture.
        handlers = (
            _build_monitor_handlers(collector)
            + _build_monitor_handlers(self._global_collector)
            + _build_context_capture_handlers(collector, managed)
        )

        self._register_instance(managed, collector, "Agent", agent_type=agent_type)
        self._emit_event(
            EVENT_AGENT_STARTED,
            message=f"Agent started: {task[:100]}",
            data={"instance_id": instance_id, "task": task},
        )

        thread = threading.Thread(
            target=self._run_agent,
            args=(managed, config, registry, handlers),
            name=f"agent-{instance_id}",
            daemon=True,
        )
        managed.thread = thread
        thread.start()
        return managed

    def _check_agent_capacity(self) -> None:
        with self._lock:
            running = sum(
                1
                for inst in self._instances.values()
                if isinstance(inst, ManagedAgent)
                and inst.status in ACTIVE_AGENT_STATUSES
            )
        if running >= self._config.max_running_agents:
            raise MonitorCapacityError(
                f"{running} agents are already running "
                f"(max_running_agents={self._config.max_running_agents})"
            )

    def _run_agent(
        self,
        managed: ManagedAgent,
        config: AgentConfig,
        registry: ToolRegistry | None,
        handlers: list[Any],
    ) -> None:
        """Body of the agent's background thread."""
        try:
            if managed.cancel_event.is_set():
                with self._lock:
                    managed.status = "cancelled"
                return

            kwargs: dict[str, Any] = {"config": config, "handlers": handlers}
            if registry is not None:
                kwargs["tools"] = registry
            agent = _AGENT_CLASSES[managed.agent_type](**kwargs)
            result = agent.run(managed.task)

            with self._lock:
                # agent.run() cannot be interrupted: keep its result even
                # when a cancel arrived while it ran.
                managed.result = result
                if managed.cancel_event.is_set():
                    managed.status = "cancelled"
                    return
                managed.status = "completed" if result.success else "failed"
            self._emit_agent_run_events(managed, result)
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
                self._emit_event(
                    EVENT_AGENT_FAILED,
                    message=f"Agent failed: {managed.label} ({type(e).__name__})",
                    data={"instance_id": managed.instance_id, "error": str(e)[:500]},
                    level="ERROR",
                )

    def _emit_agent_run_events(self, managed: ManagedAgent, result: Any) -> None:
        """Iteration, tool-call and completed/failed events of a finished run,
        read from ``result.trace`` (for metrics)."""
        fields = _agent_result_fields(result)
        instance_id, label = managed.instance_id, managed.label
        for i in range(fields.get("total_iterations", 0)):
            self._emit_event(
                EVENT_AGENT_ITERATION,
                message=f"Agent iteration {i + 1}: {label}",
                data={"instance_id": instance_id, "iteration": i + 1},
            )
        for tool in fields.get("tools_used", []):
            self._emit_event(
                EVENT_AGENT_TOOL_CALL,
                message=f"Tool call: {tool['tool_name']}",
                data={"instance_id": instance_id, "tool_name": tool["tool_name"]},
            )
        success = result.success
        self._emit_event(
            EVENT_AGENT_COMPLETED if success else EVENT_AGENT_FAILED,
            message=f"Agent {'completed' if success else 'failed'}: {label}",
            data={
                "instance_id": instance_id,
                "success": success,
                "answer": fields["answer"][:200] if fields["answer"] else "",
            },
        )

    def cancel_agent(self, instance_id: str) -> bool:
        """Signal an agent to cancel.

        Returns False (and changes nothing) when the agent has already
        finished: a completed or failed agent keeps its status and result.
        """
        inst = self._get_agent(instance_id)
        with self._lock:
            finished = inst.status in TERMINAL_STATUSES or (
                inst.thread is not None and not inst.thread.is_alive()
            )
            if finished or inst.status == "cancelling":
                return False
            inst.cancel_event.set()
            inst.status = "cancelling"
        self._emit_event(
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
                and inst.status in ACTIVE_AGENT_STATUSES
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
            result.update(_agent_result_fields(inst_result))

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
            **_agent_result_fields(inst_result),
            "final_context": redact_context(
                getattr(inst_result, "final_context", None) or {}
            ),
        }
        trace = getattr(inst_result, "trace", None)
        if trace:
            # Full trace steps for visualization
            result["trace_steps"] = [
                {
                    "state": getattr(s, "state", ""),
                    "reasoning": getattr(s, "reasoning", ""),
                    "tool_name": getattr(s, "tool_name", ""),
                    "tool_input": redact_context(getattr(s, "tool_input", "")),
                    "tool_result": getattr(s, "observation", ""),
                    "timestamp": str(getattr(s, "timestamp", "")),
                }
                for s in getattr(trace, "steps", [])
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
                    if inst.status in ACTIVE_AGENT_STATUSES:
                        inst.status = "cancelled"
                elif inst.status == "running":
                    inst.status = "completed"

        if inst is None:
            raise KeyError(f"Instance not found: {instance_id}")

        self._teardown(inst, join_agent=True)
        self._emit_event(
            EVENT_INSTANCE_DESTROYED,
            message=f"Instance destroyed: {inst.label}",
            data={
                "instance_type": inst.instance_type,
                "instance_id": instance_id,
            },
        )

    def _teardown(self, inst: ManagedInstance, *, join_agent: bool) -> None:
        """Release what an instance holds: end an FSM's conversations, stop a
        workflow engine, signal an agent. ``destroy_instance`` and
        ``shutdown`` share it; only ``destroy_instance`` waits for the agent."""
        if isinstance(inst, ManagedFSM):
            # End every conversation while the global handlers are still
            # active, then detach them (see ``_close_fsm_api``).
            self._close_fsm_api(inst, inst.instance_id)
        elif isinstance(inst, ManagedWorkflow):
            # Stop the engine's timers and background runs on the loop that
            # owns them (this may run in a worker thread).
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
        else:
            inst.cancel_event.set()
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
            if join_agent and inst.thread is not None and inst.thread.is_alive():
                inst.thread.join(timeout=1.5)
                if inst.thread.is_alive():
                    logger.warning(
                        f"Agent {inst.instance_id} did not stop within timeout "
                        "after cancellation; background thread still running"
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
                    if inst.status in TERMINAL_STATUSES
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
        return None if preset_id is None else load_preset_json(preset_id)

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
