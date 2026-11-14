"""Conversation snapshots and the activity list of ``InstanceManager`` (plan d454fd62 step 93; the old ``monitor/instance_manager.py``, same dotted path).

``snapshot_from_api`` (one conversation of one ``API`` as a ``ConversationSnapshot``, fail-closed on internal
keys) and ``SnapshotsMixin``: the queries over the attached API and every managed FSM, the cache of ended
conversations, and the unified activity list of FSM conversations, agents and workflow runs. The method
bodies are the old ones, moved.
"""

from __future__ import annotations

from typing import Any

from fsm_llm import API
from fsm_llm.logging import logger

from ..collector import redact_context
from ..constants import (
    TERMINAL_STATUSES,
)
from ..definitions import (
    ActivityItem,
    ConversationSnapshot,
    model_to_dict,
    normalize_message_history,
)
from .managed import ManagedAgent, ManagedFSM, ManagedWorkflow
from .surface import _InstanceSurface


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


class SnapshotsMixin(_InstanceSurface):
    """Conversation snapshots, the ended-conversation cache and the activity list."""

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
