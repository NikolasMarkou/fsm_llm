from __future__ import annotations

"""
Monitor bridge connecting the EventCollector to a live FSM API instance.

Provides a unified query interface for the TUI, handling graceful degradation
when optional extensions (agents, workflows, reasoning) are not installed.
"""

import json
from pathlib import Path
from typing import Any

from fsm_llm import API, HandlerTiming, create_handler

from .collector import EventCollector
from .constants import MONITOR_HANDLER_NAME, MONITOR_HANDLER_PRIORITY
from .definitions import (
    ConversationSnapshot,
    FSMSnapshot,
    MetricSnapshot,
    MonitorConfig,
    MonitorEvent,
    StateInfo,
    TransitionInfo,
)


class MonitorBridge:
    """Bridge between the FSM API and the monitor TUI.

    Wires up an EventCollector to capture lifecycle events from the API,
    and provides a unified query interface for the TUI screens.
    """

    def __init__(
        self,
        api: API | None = None,
        config: MonitorConfig | None = None,
    ) -> None:
        self._api = api
        self._config = config or MonitorConfig()
        self._collector = EventCollector(
            max_events=self._config.max_events,
            max_log_lines=self._config.max_log_lines,
        )
        self._connected = False

        if api is not None:
            self.connect(api)

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def collector(self) -> EventCollector:
        return self._collector

    @property
    def config(self) -> MonitorConfig:
        return self._config

    @config.setter
    def config(self, value: MonitorConfig) -> None:
        self._config = value

    def connect(self, api: API) -> None:
        """Connect to an API instance and register monitor handlers."""
        self._api = api
        self._register_handlers()
        self._connected = True

    def disconnect(self) -> None:
        """Disconnect from the API."""
        self._api = None
        self._connected = False

    def _register_handlers(self) -> None:
        """Register observer handlers at all timing points."""
        if self._api is None:
            return

        callbacks = self._collector.create_handler_callbacks()

        for timing_name, callback in callbacks.items():
            timing = HandlerTiming[timing_name]
            handler = (
                create_handler(f"{MONITOR_HANDLER_NAME}_{timing_name.lower()}")
                .at(timing)
                .with_priority(MONITOR_HANDLER_PRIORITY)
                .do(callback)
            )
            self._api.register_handler(handler)

    # --- Query Interface ---

    def get_metrics(self) -> MetricSnapshot:
        """Get current system metrics."""
        return self._collector.get_metrics()

    def get_active_conversations(self) -> list[str]:
        """Get list of active conversation IDs."""
        if self._api is None:
            return []
        try:
            return self._api.list_active_conversations()
        except Exception:
            return []

    def get_conversation_snapshot(
        self, conversation_id: str
    ) -> ConversationSnapshot | None:
        """Get a snapshot of a specific conversation."""
        if self._api is None:
            return None
        try:
            complete = self._api.fsm_manager.get_complete_conversation(conversation_id)
            if complete is None:
                return None

            current_state = complete.get("current_state", {})
            return ConversationSnapshot(
                conversation_id=conversation_id,
                current_state=current_state.get("id", ""),
                state_description=current_state.get("description", ""),
                is_terminal=current_state.get("is_terminal", False),
                context_data=complete.get("collected_data", {}),
                message_history=complete.get("conversation_history", []),
                stack_depth=self._api.get_stack_depth(conversation_id),
                last_extraction=_model_to_dict(
                    complete.get("last_extraction_response")
                ),
                last_transition=_model_to_dict(
                    complete.get("last_transition_decision")
                ),
                last_response=_model_to_dict(
                    complete.get("last_response_generation")
                ),
            )
        except Exception:
            return None

    def get_all_conversation_snapshots(self) -> list[ConversationSnapshot]:
        """Get snapshots for all active conversations."""
        snapshots = []
        for conv_id in self.get_active_conversations():
            snap = self.get_conversation_snapshot(conv_id)
            if snap is not None:
                snapshots.append(snap)
        return snapshots

    def get_recent_events(self, limit: int = 50) -> list[MonitorEvent]:
        """Get recent events."""
        return self._collector.get_events(limit=limit)

    def load_fsm_from_file(self, path: str) -> FSMSnapshot | None:
        """Load an FSM definition from a JSON file and return a snapshot."""
        try:
            file_path = Path(path)
            if not file_path.exists():
                return None
            data = json.loads(file_path.read_text())
            return _fsm_dict_to_snapshot(data)
        except Exception:
            return None

    def load_fsm_from_dict(self, data: dict[str, Any]) -> FSMSnapshot | None:
        """Convert an FSM definition dict to a snapshot."""
        try:
            return _fsm_dict_to_snapshot(data)
        except Exception:
            return None


def _model_to_dict(obj: Any) -> dict[str, Any] | None:
    """Convert a Pydantic model or dict to a plain dict."""
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return obj
    return None


def _fsm_dict_to_snapshot(data: dict[str, Any]) -> FSMSnapshot:
    """Convert a raw FSM definition dict to an FSMSnapshot."""
    states_data = data.get("states", {})
    initial_state = data.get("initial_state", "")

    states = []
    for state_id, state_def in states_data.items():
        transitions_raw = state_def.get("transitions", [])
        transitions = []
        for t in transitions_raw:
            conditions = t.get("conditions", [])
            transitions.append(
                TransitionInfo(
                    target_state=t.get("target_state", ""),
                    description=t.get("description", ""),
                    priority=t.get("priority", 0),
                    condition_count=len(conditions),
                    has_logic=any(c.get("logic") for c in conditions),
                )
            )

        is_terminal = len(transitions_raw) == 0
        states.append(
            StateInfo(
                state_id=state_id,
                description=state_def.get("description", ""),
                purpose=state_def.get("purpose", ""),
                is_initial=(state_id == initial_state),
                is_terminal=is_terminal,
                transition_count=len(transitions_raw),
                transitions=transitions,
            )
        )

    return FSMSnapshot(
        name=data.get("name", ""),
        description=data.get("description", ""),
        version=data.get("version", ""),
        initial_state=initial_state,
        persona=data.get("persona"),
        state_count=len(states),
        states=states,
    )
