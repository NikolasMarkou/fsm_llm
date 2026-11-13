"""``StateExecutor``: the base class of what a state's kind does in Pass 1 of a turn.

Layer ``engine`` (plan d454fd62, step 80): split out of the old ``fsm_llm.executors``; the class
body is unchanged except that ``pipeline`` is typed ``_PipelineServices`` (the members of the
turn pipeline an executor may use, ``services.py``) instead of ``Any``. The contract is the
package docstring of ``fsm_llm.engine.kinds``.
"""

from __future__ import annotations

from fsm_llm.core.definitions import DataExtractionResponse, FSMInstance, State

from .services import _PipelineServices


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
