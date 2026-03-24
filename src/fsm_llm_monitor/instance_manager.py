from __future__ import annotations

"""
Instance manager for fsm_llm_monitor.

Manages multiple concurrent FSM, workflow, and agent instances with
per-instance event collection and lifecycle management.
"""

import collections
import json
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fsm_llm import API, HandlerTiming, create_handler
from fsm_llm.constants import DEFAULT_LLM_MODEL
from fsm_llm.logging import logger

from .collector import EventCollector
from .constants import (
    EVENT_AGENT_COMPLETED,
    EVENT_AGENT_FAILED,
    EVENT_AGENT_STARTED,
    EVENT_INSTANCE_DESTROYED,
    EVENT_INSTANCE_LAUNCHED,
    EVENT_WORKFLOW_ADVANCED,
    EVENT_WORKFLOW_CANCELLED,
    EVENT_WORKFLOW_COMPLETED,
    EVENT_WORKFLOW_STARTED,
    MONITOR_HANDLER_NAME,
    MONITOR_HANDLER_PRIORITY,
)
from .definitions import (
    ConversationSnapshot,
    InstanceInfo,
    MetricSnapshot,
    MonitorConfig,
    MonitorEvent,
    StubToolConfig,
    model_to_dict,
    normalize_message_history,
)

# Optional imports for workflows and agents
_HAS_WORKFLOWS = False
_HAS_AGENTS = False

try:
    from fsm_llm_workflows import WorkflowEngine

    _HAS_WORKFLOWS = True
except ImportError:
    pass

try:
    from fsm_llm_agents import (
        ADaPTAgent,
        AgentConfig,
        DebateAgent,
        EvaluatorOptimizerAgent,
        MakerCheckerAgent,
        PlanExecuteAgent,
        ReactAgent,
        ReflexionAgent,
        REWOOAgent,
        SelfConsistencyAgent,
        ToolRegistry,
    )

    _AGENT_CLASSES: dict[str, type] = {
        "ReactAgent": ReactAgent,
        "ReflexionAgent": ReflexionAgent,
        "PlanExecuteAgent": PlanExecuteAgent,
        "REWOOAgent": REWOOAgent,
        "ADaPTAgent": ADaPTAgent,
        "DebateAgent": DebateAgent,
        "SelfConsistencyAgent": SelfConsistencyAgent,
        "EvaluatorOptimizerAgent": EvaluatorOptimizerAgent,
        "MakerCheckerAgent": MakerCheckerAgent,
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
    pass


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
        self.created_at = datetime.now()
        self.conversation_ids: list[str] = []

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
        self.created_at = datetime.now()
        self.engine: Any = None  # WorkflowEngine
        self.workflow_id: str = ""
        self.active_instance_ids: list[str] = []

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
        self.created_at = datetime.now()
        self.thread: threading.Thread | None = None
        self.cancel_event = threading.Event()
        self.result: Any = None  # AgentResult
        self.error: str | None = None

    def to_info(self) -> InstanceInfo:
        return InstanceInfo(
            instance_id=self.instance_id,
            instance_type=self.instance_type,
            label=self.label,
            status=self.status,
            created_at=self.created_at,
            agent_type=self.agent_type,
        )


ManagedInstance = ManagedFSM | ManagedWorkflow | ManagedAgent


def register_monitor_handlers(api: API, collector: EventCollector) -> None:
    """Register observer handlers on an API instance for event collection.

    Extracted as a standalone function so both MonitorBridge and InstanceManager
    can reuse this pattern.
    """
    callbacks = collector.create_handler_callbacks()
    for timing_name, callback in callbacks.items():
        timing = HandlerTiming[timing_name]
        handler = (
            create_handler(f"{MONITOR_HANDLER_NAME}_{timing_name.lower()}")
            .at(timing)
            .with_priority(MONITOR_HANDLER_PRIORITY)
            .do(callback)
        )
        api.register_handler(handler)


def _build_monitor_handlers(collector: EventCollector) -> list[Any]:
    """Build a list of handler objects for injection into agents via api_kwargs."""
    callbacks = collector.create_handler_callbacks()
    handlers = []
    for timing_name, callback in callbacks.items():
        timing = HandlerTiming[timing_name]
        handler = (
            create_handler(f"{MONITOR_HANDLER_NAME}_{timing_name.lower()}")
            .at(timing)
            .with_priority(MONITOR_HANDLER_PRIORITY)
            .do(callback)
        )
        handlers.append(handler)
    return handlers


def _find_examples_dir() -> Path | None:
    """Locate the examples/ directory."""
    base = Path(__file__).parent.parent.parent / "examples"
    if not base.exists():
        base = base.parent / "examples"
    return base if base.exists() else None


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

    def _setup_loguru_sink(self) -> None:
        """Register a loguru sink that feeds log records into the global collector."""
        try:
            from loguru import logger as _loguru_logger

            sink = self._global_collector.create_loguru_sink()
            self._global_collector._log_sink_id = _loguru_logger.add(
                sink, level="DEBUG"
            )
        except Exception as e:
            logger.debug(f"Failed to register loguru sink: {e}")

    @property
    def config(self) -> MonitorConfig:
        return self._config

    @config.setter
    def config(self, value: MonitorConfig) -> None:
        self._config = value

    @property
    def global_collector(self) -> EventCollector:
        return self._global_collector

    def connect_bridge(self, api: API) -> None:
        """Connect an external API instance (backward compat with MonitorBridge)."""
        self._bridge_api = api
        self._bridge_collector = self._global_collector
        register_monitor_handlers(api, self._global_collector)

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
        return self._global_collector.get_metrics()

    def get_events(self, limit: int = 50) -> list[MonitorEvent]:
        return self._global_collector.get_events(limit=limit)

    # --- Conversation queries (aggregated across all FSM instances + bridge) ---

    def get_active_conversations(self) -> list[str]:
        result: list[str] = []
        # From bridge API
        if self._bridge_api is not None:
            try:
                result.extend(self._bridge_api.list_active_conversations())
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
        # Check bridge API first
        if self._bridge_api is not None:
            snap = self._snapshot_from_api(self._bridge_api, conversation_id)
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

    @staticmethod
    def _snapshot_from_api(
        api: API, conversation_id: str
    ) -> ConversationSnapshot | None:
        try:
            complete = api.fsm_manager.get_complete_conversation(conversation_id)
            if complete is None:
                return None
            current_state = complete.get("current_state", {})
            return ConversationSnapshot(
                conversation_id=conversation_id,
                current_state=current_state.get("id", ""),
                state_description=current_state.get("description", ""),
                is_terminal=current_state.get("is_terminal", False),
                context_data=complete.get("collected_data", {}),
                message_history=normalize_message_history(
                    complete.get("conversation_history", [])
                ),
                stack_depth=api.get_stack_depth(conversation_id),
                last_extraction=model_to_dict(complete.get("last_extraction_response")),
                last_transition=model_to_dict(complete.get("last_transition_decision")),
                last_response=model_to_dict(complete.get("last_response_generation")),
            )
        except Exception:
            return None

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

        # Cache snapshot when conversation reaches terminal state
        if is_terminal:
            self._cache_ended_conversation(inst.api, conversation_id, instance_id)

        # Auto-complete instance when all conversations reach terminal state
        if is_terminal and inst.status == "running" and inst.conversation_ids:
            all_terminal = all(
                inst.api.has_conversation_ended(cid) for cid in inst.conversation_ids
            )
            if all_terminal:
                inst.status = "completed"

        return {
            "response": response,
            "current_state": current_state,
            "is_terminal": is_terminal,
        }

    def end_conversation(self, instance_id: str, conversation_id: str) -> None:
        """End a conversation on a managed FSM instance."""
        with self._lock:
            inst = self._get_fsm(instance_id)
        # Cache snapshot before the API removes the conversation data
        self._cache_ended_conversation(inst.api, conversation_id, instance_id)
        inst.api.end_conversation(conversation_id)
        # Keep conversation_id in the list so it remains visible in the UI

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

    def launch_workflow(
        self,
        preset_id: str | None = None,
        definition_json: dict[str, Any] | None = None,
        initial_context: dict[str, Any] | None = None,
        label: str = "",
    ) -> ManagedWorkflow:
        """Launch a new workflow instance."""
        if not _HAS_WORKFLOWS:
            raise RuntimeError("fsm_llm_workflows extension is not installed")

        instance_id = str(uuid.uuid4())[:12]
        if not label:
            label = f"Workflow-{instance_id[:6]}"

        engine = WorkflowEngine()

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

    async def start_workflow_instance(
        self,
        instance_id: str,
        workflow_id: str,
        initial_context: dict[str, Any] | None = None,
    ) -> str:
        """Start a workflow execution on a managed workflow engine."""
        inst = self._get_workflow(instance_id)
        wf_instance_id = await inst.engine.start_workflow(
            workflow_id=workflow_id,
            initial_context=initial_context or {},
        )
        inst.active_instance_ids.append(wf_instance_id)

        self._emit_global_event(
            EVENT_WORKFLOW_STARTED,
            message=f"Workflow instance started: {wf_instance_id}",
            data={
                "instance_id": instance_id,
                "workflow_instance_id": wf_instance_id,
            },
        )
        return str(wf_instance_id)

    async def advance_workflow(
        self,
        instance_id: str,
        wf_instance_id: str,
        user_input: str = "",
    ) -> bool:
        """Advance a workflow instance."""
        inst = self._get_workflow(instance_id)
        result = await inst.engine.advance_workflow(wf_instance_id, user_input)

        self._emit_global_event(
            EVENT_WORKFLOW_ADVANCED,
            message=f"Workflow advanced: {wf_instance_id}",
            data={
                "instance_id": instance_id,
                "workflow_instance_id": wf_instance_id,
                "advanced": result,
            },
        )

        # Check if completed
        try:
            status = inst.engine.get_workflow_status(wf_instance_id)
            if hasattr(status, "value"):
                status_str = status.value
            else:
                status_str = str(status)
            if status_str in ("completed", "COMPLETED"):
                self._emit_global_event(
                    EVENT_WORKFLOW_COMPLETED,
                    message=f"Workflow completed: {wf_instance_id}",
                    data={
                        "instance_id": instance_id,
                        "workflow_instance_id": wf_instance_id,
                    },
                )
                if wf_instance_id in inst.active_instance_ids:
                    inst.active_instance_ids.remove(wf_instance_id)
        except Exception as e:
            logger.debug(f"Failed to check workflow completion status: {e}")

        return bool(result)

    async def cancel_workflow(
        self,
        instance_id: str,
        wf_instance_id: str,
        reason: str = "",
    ) -> bool:
        """Cancel a workflow instance."""
        inst = self._get_workflow(instance_id)
        result = await inst.engine.cancel_workflow(wf_instance_id, reason=reason)

        if result:
            self._emit_global_event(
                EVENT_WORKFLOW_CANCELLED,
                message=f"Workflow cancelled: {wf_instance_id}",
                data={
                    "instance_id": instance_id,
                    "workflow_instance_id": wf_instance_id,
                    "reason": reason,
                },
            )
            if wf_instance_id in inst.active_instance_ids:
                inst.active_instance_ids.remove(wf_instance_id)

        return bool(result)

    def get_workflow_status(
        self, instance_id: str, wf_instance_id: str
    ) -> dict[str, Any]:
        """Get workflow instance status and context."""
        inst = self._get_workflow(instance_id)
        try:
            wf_instance = inst.engine.get_workflow_instance(wf_instance_id)
            status = inst.engine.get_workflow_status(wf_instance_id)
            context = inst.engine.get_workflow_context(wf_instance_id)
            status_str = status.value if hasattr(status, "value") else str(status)
            return {
                "workflow_instance_id": wf_instance_id,
                "status": status_str,
                "current_step": getattr(wf_instance, "current_step_id", ""),
                "context": context,
                "created_at": str(getattr(wf_instance, "created_at", "")),
                "updated_at": str(getattr(wf_instance, "updated_at", "")),
            }
        except Exception as e:
            return {"error": str(e)}

    def get_workflow_instances(self, instance_id: str) -> list[dict[str, Any]]:
        """List all workflow instances on a managed workflow engine."""
        inst = self._get_workflow(instance_id)
        results: list[dict[str, Any]] = []
        for wf_id in inst.active_instance_ids:
            try:
                status_data = self.get_workflow_status(instance_id, wf_id)
                results.append(status_data)
            except Exception:
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
            raise RuntimeError("fsm_llm_agents extension is not installed")

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
        all_handlers = monitor_handlers + global_handlers

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
                agent_cls = _AGENT_CLASSES[agent_type]
                kwargs: dict[str, Any] = {
                    "config": config,
                    "handlers": all_handlers,
                }
                if needs_tools and registry is not None:
                    kwargs["tools"] = registry
                agent = agent_cls(**kwargs)
                result = agent.run(task)
                managed.result = result
                managed.status = "completed" if result.success else "failed"
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
                managed.status = "failed"
                managed.error = str(e)
                self._emit_global_event(
                    EVENT_AGENT_FAILED,
                    message=f"Agent failed: {label} - {e}",
                    data={"instance_id": instance_id, "error": str(e)},
                    level="ERROR",
                )

        thread = threading.Thread(
            target=_run_agent, name=f"agent-{instance_id}", daemon=True
        )
        managed.thread = thread
        thread.start()

        logger.info(f"Launched agent {instance_id}: {label}")
        return managed

    def cancel_agent(self, instance_id: str) -> None:
        """Signal an agent to cancel."""
        inst = self._get_agent(instance_id)
        inst.cancel_event.set()
        inst.status = "cancelled"
        self._emit_global_event(
            EVENT_AGENT_FAILED,
            message=f"Agent cancelled: {inst.label}",
            data={"instance_id": instance_id},
        )

    def get_agent_status(self, instance_id: str) -> dict[str, Any]:
        """Get agent status including real-time progress and partial results."""
        inst = self._get_agent(instance_id)

        # Check if thread is still alive
        with self._lock:
            if inst.thread and not inst.thread.is_alive() and inst.status == "running":
                inst.status = "completed" if inst.result else "failed"

        result: dict[str, Any] = {
            "instance_id": instance_id,
            "agent_type": inst.agent_type,
            "task": inst.task,
            "status": inst.status,
            "created_at": str(inst.created_at),
        }

        # Derive real-time progress from per-instance collector events
        collector = self._collectors.get(instance_id)
        if collector and inst.status == "running":
            events = collector.get_events(limit=0)
            transition_count = 0
            current_state = ""
            last_tool = ""
            for evt in events:
                if evt.event_type == "state_transition":
                    transition_count += 1
                    if evt.target_state:
                        current_state = evt.target_state
                    if (
                        evt.target_state == "act"
                        and isinstance(evt.data, dict)
                        and evt.data.get("tool_name")
                    ):
                        last_tool = evt.data["tool_name"]
                    elif isinstance(evt.data, dict) and evt.data.get("tool_name"):
                        last_tool = evt.data["tool_name"]
            # Each think→act→think cycle is roughly one iteration
            iteration_count = (transition_count + 1) // 2
            result["current_state"] = current_state
            result["iteration_count"] = iteration_count
            result["last_tool_call"] = last_tool
            result["transition_count"] = transition_count

        if inst.result is not None:
            result["answer"] = inst.result.answer
            result["success"] = inst.result.success
            if hasattr(inst.result, "trace") and inst.result.trace:
                trace = inst.result.trace
                result["total_iterations"] = getattr(trace, "total_iterations", 0)
                tool_calls = getattr(trace, "tool_calls", [])
                result["tools_used"] = [
                    {
                        "tool_name": getattr(tc, "tool_name", ""),
                        "parameters": getattr(tc, "parameters", {}),
                    }
                    for tc in tool_calls
                ]

        if inst.error:
            result["error"] = inst.error

        return result

    def get_agent_result(self, instance_id: str) -> dict[str, Any]:
        """Get final agent result (if complete) with full trace steps."""
        inst = self._get_agent(instance_id)
        if inst.result is None:
            return {"error": "Agent has not completed yet", "status": inst.status}

        result: dict[str, Any] = {
            "answer": inst.result.answer,
            "success": inst.result.success,
            "final_context": getattr(inst.result, "final_context", {}),
        }
        if hasattr(inst.result, "trace") and inst.result.trace:
            trace = inst.result.trace
            result["total_iterations"] = getattr(trace, "total_iterations", 0)
            tool_calls = getattr(trace, "tool_calls", [])
            result["tools_used"] = [
                {
                    "tool_name": getattr(tc, "tool_name", ""),
                    "parameters": getattr(tc, "parameters", {}),
                }
                for tc in tool_calls
            ]
            # Full trace steps for visualization
            steps = getattr(trace, "steps", [])
            result["trace_steps"] = [
                {
                    "state": getattr(s, "state", ""),
                    "reasoning": getattr(s, "reasoning", ""),
                    "tool_name": getattr(s, "tool_name", ""),
                    "tool_input": getattr(s, "tool_input", ""),
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

        if inst is None:
            raise KeyError(f"Instance not found: {instance_id}")

        if isinstance(inst, ManagedFSM):
            for conv_id in list(inst.conversation_ids):
                try:
                    inst.api.end_conversation(conv_id)
                except Exception as e:
                    logger.debug(
                        f"Failed to end conversation {conv_id} during instance destroy: {e}"
                    )
            inst.status = "completed"
        elif isinstance(inst, ManagedWorkflow):
            inst.status = "completed"
        elif isinstance(inst, ManagedAgent):
            inst.cancel_event.set()
            inst.status = "cancelled"

        self._emit_global_event(
            EVENT_INSTANCE_DESTROYED,
            message=f"Instance destroyed: {inst.label}",
            data={
                "instance_type": inst.instance_type,
                "instance_id": instance_id,
            },
        )

    # --- Private Helpers ---

    def _get_fsm(self, instance_id: str) -> ManagedFSM:
        inst = self.get_instance(instance_id)
        if inst is None:
            raise KeyError(f"Instance not found: {instance_id}")
        if not isinstance(inst, ManagedFSM):
            raise TypeError(f"Instance {instance_id} is not an FSM")
        return inst

    def _get_workflow(self, instance_id: str) -> ManagedWorkflow:
        inst = self.get_instance(instance_id)
        if inst is None:
            raise KeyError(f"Instance not found: {instance_id}")
        if not isinstance(inst, ManagedWorkflow):
            raise TypeError(f"Instance {instance_id} is not a workflow")
        return inst

    def _get_agent(self, instance_id: str) -> ManagedAgent:
        inst = self.get_instance(instance_id)
        if inst is None:
            raise KeyError(f"Instance not found: {instance_id}")
        if not isinstance(inst, ManagedAgent):
            raise TypeError(f"Instance {instance_id} is not an agent")
        return inst

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
            if ".." in preset_id or preset_id.startswith("/"):
                raise ValueError("Invalid preset ID")
            file_path = base / preset_id
            try:
                file_path.resolve().relative_to(base.resolve())
            except ValueError as e:
                raise ValueError("Invalid preset ID") from e
            if not file_path.exists():
                raise FileNotFoundError(f"Preset not found: {preset_id}")
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
