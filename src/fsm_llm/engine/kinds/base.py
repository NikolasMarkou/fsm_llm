"""``StateExecutor``: the base class of what a state's kind does in Pass 1 of a turn.

Layer ``engine`` (plan d454fd62, step 80): split out of the old ``fsm_llm.executors``; the class
body is unchanged except that ``pipeline`` is typed ``_PipelineServices`` (the members of the
turn pipeline an executor may use, ``services.py``) instead of ``Any``. The contract is the
package docstring of ``fsm_llm.engine.kinds``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from fsm_llm.aio import SYNC
from fsm_llm.core.definitions import DataExtractionResponse, FSMInstance, State

from .services import _PipelineServices


async def _on_worker(call: Callable[..., Any], *args: Any) -> Any:
    """The rule of ``llm.interface._on_worker`` for several arguments: inline in sync mode
    (no loop to hop to), on a worker thread in async mode."""
    if SYNC.get():
        return call(*args)
    return await asyncio.to_thread(call, *args)


class StateExecutor:
    """What the turn pipeline calls for a state of one kind (see the module
    docstring for the contract). Subclass it and register the instance."""

    # Whether the extracted data is passed through ``clean_context_keys`` before
    # it is committed (it drops None values and internal-prefixed keys).
    cleans_extracted: bool = True

    def extract(
        self,
        pipeline: _PipelineServices,
        instance: FSMInstance,
        user_message: str | None,
        conversation_id: str,
        state: State,
    ) -> DataExtractionResponse:
        raise NotImplementedError

    async def aextract(
        self,
        pipeline: _PipelineServices,
        instance: FSMInstance,
        user_message: str | None,
        conversation_id: str,
        state: State,
    ) -> DataExtractionResponse:
        """The async twin of ``extract``. The default runs ``extract`` on a worker thread in
        async mode, so an executor with only the sync method keeps working (a cancel returns
        at once and the thread runs on)."""
        response: DataExtractionResponse = await _on_worker(
            self.extract, pipeline, instance, user_message, conversation_id, state
        )
        return response

    def after_transition(
        self,
        pipeline: _PipelineServices,
        instance: FSMInstance,
        user_message: str | None,
        conversation_id: str,
        extraction_response: DataExtractionResponse,
        previous_state: str | None,
        new_state: State,
    ) -> None:
        """Nothing runs after a transition into this kind of state."""
        return None

    async def aafter_transition(
        self,
        pipeline: _PipelineServices,
        instance: FSMInstance,
        user_message: str | None,
        conversation_id: str,
        extraction_response: DataExtractionResponse,
        previous_state: str | None,
        new_state: State,
    ) -> None:
        """The async twin of ``after_transition``, with the same default as ``aextract``."""
        await _on_worker(
            self.after_transition,
            pipeline,
            instance,
            user_message,
            conversation_id,
            extraction_response,
            previous_state,
            new_state,
        )
