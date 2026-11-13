"""The declarations the mixins of ``API`` share (mypy only).

Layer ``engine`` (plan d454fd62, step 81): what a mixin of the old ``fsm_llm.api.API`` may rely on:
the attributes the constructor of the shell sets and the methods the other mixins define, declared
for mypy only (a runtime no-op, like ``engine/turn/surface.py``). A mixin that defines one of these
is checked against the declaration here; the module-level helpers that took the class take this
class (a mixin passes ``self``, which is not an ``API``).
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import os
    from collections.abc import Iterator

    from fsm_llm.core.definitions import AdvanceResult, FSMDefinition
    from fsm_llm.engine.fsm.manager import FSMManager
    from fsm_llm.engine.session import SessionStore
    from fsm_llm.handlers import HandlerSystem
    from fsm_llm.llm import LLMInterface

    from .stack import FSMStackFrame


class _ApiSurface:
    """What every mixin of ``API`` may rely on: the attributes of the shell and the methods of the
    other mixins, declared for mypy only (a runtime no-op).
    """

    if TYPE_CHECKING:
        llm_interface: LLMInterface
        fsm_definition: FSMDefinition
        fsm_id: str
        handler_system: HandlerSystem
        fsm_manager: FSMManager
        _temp_fsm_definitions: dict[str, FSMDefinition]
        _pending_push_ids: set[str]
        _stack_lock: threading.Lock
        conversation_stacks: dict[str, list[FSMStackFrame]]
        _last_accessed: dict[str, float]
        _ended_conversations: dict[str, dict[str, Any]]
        _MAX_ENDED_CACHE: int
        _session_store: SessionStore | None

        def _get_current_fsm_conversation_id(self, conversation_id: str) -> str: ...

        def _release_unreferenced_temp_definitions(self) -> None: ...

        def _ended_cache_entry(self, conversation_id: str) -> dict[str, Any] | None: ...

        def _conversation_gone(self, conversation_id: str) -> bool: ...

        def has_conversation_ended(self, conversation_id: str) -> bool: ...

        def get_current_state(self, conversation_id: str) -> str: ...

        def end_conversation(self, conversation_id: str) -> None: ...

        def list_active_conversations(self) -> list[str]: ...

        def get_stack_depth(self, conversation_id: str) -> int: ...

        def save_session(self, conversation_id: str) -> None: ...

        def start_conversation(
            self,
            initial_context: dict[str, Any] | None = None,
            *,
            _suppress_start: bool = False,
        ) -> tuple[str, str]: ...

        def advance(self, conversation_id: str) -> AdvanceResult: ...

        def advance_stream(self, conversation_id: str) -> Iterator[str]: ...

        @classmethod
        def process_fsm_definition(
            cls,
            fsm_definition: FSMDefinition | dict[str, Any] | str | os.PathLike[str],
        ) -> tuple[FSMDefinition, str]: ...
