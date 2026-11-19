"""Agent instances of ``InstanceManager``: launch, background run, status (plan d454fd62 step 93; the old ``monitor/instance_manager.py``, same dotted path).

``AgentsMixin``: ``launch_agent`` (a daemon thread per agent), the capacity check, the body of the thread, the
events read from the finished run, ``cancel_agent``, ``get_agent_status`` and ``get_agent_result``; with the
launchable classes (``_AGENT_CLASSES``, ``_TOOL_BASED_AGENTS``, D-001), the stub tool registry and the two
readers of an ``AgentResult``. The method bodies are the old ones, moved.
"""

from __future__ import annotations

import threading
from typing import Any

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
from fsm_llm.llm.interface import resolve_tool_model

from ..collector import redact_context
from ..constants import (
    ACTIVE_AGENT_STATUSES,
    DEFAULT_AGENT_MAX_ITERATIONS,
    DEFAULT_AGENT_TIMEOUT_SECONDS,
    EVENT_AGENT_CANCELLED,
    EVENT_AGENT_COMPLETED,
    EVENT_AGENT_FAILED,
    EVENT_AGENT_ITERATION,
    EVENT_AGENT_STARTED,
    EVENT_AGENT_TOOL_CALL,
    TERMINAL_STATUSES,
)
from ..definitions import (
    StubToolConfig,
)
from ..exceptions import (
    MonitorCapacityError,
)
from .handlers import _build_context_capture_handlers, _build_monitor_handlers
from .managed import ManagedAgent, _new_instance_id
from .surface import _InstanceSurface

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
# OrchestratorAgent is not launchable either (D-058 of plan 51fde95b): without a
# `worker_factory` (a Python callable) every subtask is a placeholder with
# success False, so a dashboard run always ends `failed`.

# Agent types that require a ToolRegistry
_TOOL_BASED_AGENTS = {
    "ReactAgent",
    "ReflexionAgent",
    "PlanExecuteAgent",
    "REWOOAgent",
    "ADaPTAgent",
}


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


class AgentsMixin(_InstanceSurface):
    """Launching agents in background threads and reading their runs."""

    def launch_agent(
        self,
        agent_type: str = "ReactAgent",
        task: str = "",
        tools_config: list[StubToolConfig] | None = None,
        model: str | None = None,
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
            model=resolve_tool_model(model),
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
