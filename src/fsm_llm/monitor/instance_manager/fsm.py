"""FSM instances of ``InstanceManager``: launch, conversations, messages (plan d454fd62 step 93; the old ``monitor/instance_manager.py``, same dotted path).

``FsmMixin``: ``launch_fsm`` (a preset or raw JSON to a managed ``API``), ``start_conversation``,
``send_message``, ``end_conversation`` and the two helpers that end a conversation and complete the instance,
``get_fsm_conversations`` and ``_resolve_fsm_data``. The method bodies are the old ones, moved.
"""

from __future__ import annotations

from typing import Any

from fsm_llm import API
from fsm_llm.core.codes import MessageSource, message_source
from fsm_llm.llm import resolve_model
from fsm_llm.logging import logger

from ..constants import (
    DEFAULT_FSM_TEMPERATURE,
)
from ..definitions import (
    ConversationSnapshot,
)
from ..exceptions import ConversationEndingError, MonitorCapacityError
from .examples import load_preset_json
from .handlers import register_monitor_handlers
from .managed import ManagedFSM, _new_instance_id
from .surface import _InstanceSurface


class FsmMixin(_InstanceSurface):
    """Launching FSM instances and driving their conversations."""

    def launch_fsm(
        self,
        preset_id: str | None = None,
        fsm_json: dict[str, Any] | None = None,
        model: str | None = None,
        temperature: float = DEFAULT_FSM_TEMPERATURE,
        label: str = "",
    ) -> ManagedFSM:
        """Launch a new FSM instance from preset or raw JSON."""
        fsm_data = self._resolve_fsm_data(preset_id, fsm_json)
        if fsm_data is None:
            raise ValueError("Must provide either preset_id or fsm_json")
        instance_id = _new_instance_id()
        api = API.from_definition(
            fsm_data, model=resolve_model(model), temperature=temperature
        )

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
        try:
            self._register_instance(managed, collector, "FSM")
        except MonitorCapacityError:
            api.close()  # refused: nothing was registered, the API holds nothing else
            raise
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
        with message_source(MessageSource.REMOTE_CLIENT):
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
        self._finish_conversation(inst, conversation_id)

    def _finish_conversation(self, inst: ManagedFSM, conversation_id: str) -> None:
        """Cache the snapshot, end the conversation in the core (it fires
        END_CONVERSATION, closing the monitor/OTEL span, and frees it), and only
        then record it as ended: a refusal (``ConversationBusyError``)
        propagates and leaves it unmarked. The id stays in ``conversation_ids``
        so the conversation remains visible in the UI. An id already ended is a
        no-op; one being ended by another caller is refused (409).

        # DECISION plan-2026-11-27T201735-9283c1fd/D-005
        # The check (ended? being ended?) and the claim are ONE ``_lock`` scope: in two
        # scopes both callers pass the check and the loser fails inside the core with
        # a plain ``FSMError`` (a 500). The end itself runs OUTSIDE ``_lock`` (it fires
        # END handlers) and the claim is dropped in a ``finally`` (a refused end must
        # stay retryable). Do NOT hold ``_lock`` across ``api.end_conversation``, go
        # back to check-then-act, or answer the loser with a silent success.
        """
        with self._lock:
            if conversation_id in inst.ended_conversation_ids:
                return
            if conversation_id in inst.ending_conversation_ids:
                raise ConversationEndingError(
                    "conversation is already being ended",
                    conversation_id=conversation_id,
                )
            inst.ending_conversation_ids.add(conversation_id)
        try:
            self._cache_ended_conversation(inst.api, conversation_id, inst.instance_id)
            inst.api.end_conversation(conversation_id)
            self._mark_conversation_ended(inst, conversation_id)
        finally:
            with self._lock:
                inst.ending_conversation_ids.discard(conversation_id)

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

    def _resolve_fsm_data(
        self,
        preset_id: str | None,
        fsm_json: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Resolve FSM definition from preset ID or raw JSON."""
        if fsm_json is not None:
            return fsm_json
        return None if preset_id is None else load_preset_json(preset_id)
