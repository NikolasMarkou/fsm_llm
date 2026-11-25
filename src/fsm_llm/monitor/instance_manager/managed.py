"""What the manager keeps for each instance it launched (plan d454fd62 step 93; the old ``monitor/instance_manager.py``, same dotted path).

``ManagedFSM``, ``ManagedWorkflow`` and ``ManagedAgent`` over one private base (``_ManagedBase``: the id, label,
status and creation time every instance shares, and ``_info``), the ``ManagedInstance`` union and
``_new_instance_id``, the id a launch gives, and ``api_usage`` / ``add_usage``, the read-only usage reads of the
manager. The three classes and their base stay in ONE module: each ``__init__`` calls ``super().__init__`` and
``_info`` is the one place an ``InstanceInfo`` is built. Imports no other module of the package.
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from datetime import UTC, datetime
from typing import Any

from fsm_llm import API, LLMCallCounts, LLMUsage

from ..constants import (
    DEFAULT_AGENT_MAX_ITERATIONS,
)
from ..definitions import (
    InstanceInfo,
)


class _ManagedBase:
    """The fields every managed instance shares."""

    instance_type = ""

    def __init__(self, instance_id: str, label: str = "") -> None:
        self.instance_id = instance_id
        self.label = label
        self.status = "running"
        self.created_at = datetime.now(UTC)

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
        agent_type: str = "react",
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


def _new_instance_id() -> str:
    return str(uuid.uuid4())[:12]


def api_usage(api: Any) -> tuple[int, LLMUsage] | None:
    """``(id of the interface, its usage())`` for an API, or None.

    Read-only: only ``usage()`` is called, never ``reset_usage()`` (that moves
    the host's marker). An interface with no ``usage`` (a double) or a
    ``usage()`` that is not an ``LLMUsage`` reports none; never raises.
    """
    try:
        llm = api.get_llm_interface()
        usage = llm.usage()
    except Exception:
        return None
    return (id(llm), usage) if isinstance(usage, LLMUsage) else None


def add_usage(total: LLMUsage | None, more: LLMUsage) -> LLMUsage:
    """``total + more``, field by field and kind by kind (``None`` is zero)."""
    if total is None:
        return more
    names, zero = tuple(LLMCallCounts.model_fields), LLMCallCounts()

    def _sum(a: LLMCallCounts, b: LLMCallCounts) -> dict[str, int]:
        return {n: getattr(a, n) + getattr(b, n) for n in names}

    kinds = {
        k: LLMCallCounts(**_sum(total.by_kind.get(k, zero), more.by_kind.get(k, zero)))
        for k in total.by_kind.keys() | more.by_kind.keys()
    }
    return LLMUsage(**_sum(total, more), by_kind=kinds)
