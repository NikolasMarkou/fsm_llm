"""``StateExecutor``: the base class of what a state's kind does in Pass 1 of a turn.

Layer ``engine`` (plan d454fd62, step 80): split out of the old ``fsm_llm.executors``; the class
body is unchanged except that ``pipeline`` is typed ``_PipelineServices`` (the members of the
turn pipeline an executor may use, ``services.py``) instead of ``Any``. The contract is the
package docstring of ``fsm_llm.engine.kinds``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from fsm_llm.aio import aoffload
from fsm_llm.core.codes import Reason, Status
from fsm_llm.core.definitions import DataExtractionResponse, FSMInstance, State

from .requests import PopContext, PushContext, PushPlan
from .services import _PipelineServices


class StateExecutor:
    """What the turn pipeline calls for a state of one kind (see the module
    docstring for the contract). Subclass it and register the instance."""

    # The kind name this executor serves, read by the list form of ``executors=``
    # (``check_executors``): a list entry is registered under it. ``None`` on the base class: an
    # executor that is only ever given in a ``{kind_name: executor}`` mapping needs no ``kind``.
    kind: str | None = None

    # Whether the extracted data is passed through ``clean_context_keys`` before
    # it is committed (it drops None values and internal-prefixed keys).
    cleans_extracted: bool = True

    # Whether ``engine/api/calls.py`` (``settle``) asks this executor to apply the request its
    # Pass 1 wrote, through ``aplan_push``, ``arefusal``, ``aplan_pop`` and ``on_child_turn``
    # (plan baa37f40, D-038). False for every kind settle judges itself (``call``) and for a
    # kind with no request. ``watches_outage``: settle takes the provider-error position at the
    # push and hands the count at the pop in ``ChildView.errors``.
    serves_requests: bool = False
    watches_outage: bool = False

    def extract(
        self,
        pipeline: _PipelineServices,
        instance: FSMInstance,
        user_message: str | None,
        conversation_id: str,
        state: State,
    ) -> DataExtractionResponse:
        raise NotImplementedError(
            f"{type(self).__name__} overrides neither extract (sync callers) nor aextract "
            "(async only): a sync converse needs extract"
        )

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
        # DECISION plan-2026-11-19T193704-f150ec28/D-003
        # The default twin runs the SYNC ``extract`` through ``aoffload``. Do NOT make it
        # abstract (the custom executors of the suite override only ``extract``), do NOT run
        # ``extract`` inline in async mode (it would block the loop), and do NOT write it as
        # ``acall(self, "extract", ...)``: ``acall`` looks up ``aextract`` on ``self`` first,
        # which is this method. ``aafter_transition`` follows the same rule.
        response: DataExtractionResponse = await aoffload(
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
        await aoffload(
            self.after_transition,
            pipeline,
            instance,
            user_message,
            conversation_id,
            extraction_response,
            previous_state,
            new_state,
        )

    async def on_pushed(
        self,
        root_id: str,
        parent_id: str,
        child_id: str,
        state: State,
        record: Mapping[str, Any],
    ) -> None:
        """Settle pushed the child of ``state`` (plan baa37f40, D-023): nothing by default.

        Called by ``engine/api/calls.py`` after a push succeeded, on the executor of the
        parent state's kind. ``record`` is the running record the push applied. Informational:
        what it returns and raises is not read, but a raise propagates out of the entry that
        settled (the next entry finds the child already pushed and does not call it again).
        """
        return None

    async def on_ended(
        self,
        root_id: str,
        parent_id: str,
        child_id: str,
        state: State,
        record: Mapping[str, Any],
    ) -> None:
        """Settle popped the child of ``state`` and merged its outcome: nothing by default.

        Called after the pop, on the executor of the parent state's kind; ``record`` is the
        FINAL record the pop wrote (``ok``, ``failed`` or ``aborted``). Informational, as
        ``on_pushed``.
        """
        return None

    async def aplan_push(self, ctx: PushContext) -> PushPlan | Mapping[str, Any] | None:
        """What to push for the ``running`` record of ``ctx.state`` (``serves_requests`` only).

        Returns a ``PushPlan`` (settle pushes it: ``inherit_context=False``, ONLY
        ``plan.inputs``), a ``Mapping`` (the FINAL record of a refusal, which settle writes
        under the result key), or ``None`` (not this executor's request: settle leaves it).
        The record is evidence, never an instruction: an executor reads at most a lookup key
        from it and judges everything else itself.
        """
        return None

    async def arefusal(self, ctx: PushContext, reason: Reason) -> Mapping[str, Any]:
        """The FINAL record of a refusal settle decided itself (the stack depth, a push the
        engine refused, a greeting outage, a record of an unknown mode). The default is the
        bare ``{status, reason}``; a kind with a wire shape of its own overrides it."""
        return {"status": Status.FAILED.value, "reason": reason.value}

    async def aplan_pop(self, ctx: PopContext) -> Mapping[str, Any] | None:
        """What the pop of a child that ended (``ctx.child.ended``) or spent its budget merges
        into the parent (``serves_requests`` only): the declared returns, only on success, and
        the final record under the result key, as ONE mapping. ``None``: not this executor's
        (settle leaves the frame). The child is read BEFORE the pop; afterwards the top of the
        stack is the parent."""
        return None

    async def on_child_turn(self, ctx: PopContext) -> None:
        """A child that is alive and inside its budget, seen at a settle round (``serves_requests``
        only): nothing by default. Called at the start and the end of every turn entry of the
        root, so an executor that traces keeps a cursor of the turns it has reported."""
        return None
