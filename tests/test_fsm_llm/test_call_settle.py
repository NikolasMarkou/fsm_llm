"""``settle``: what applies a call record after the turn (plan baa37f40, step 12, D-023, roadmap E3.2).

The call executor (``test_call_executor.py``) writes a ``running`` request in Pass 1. ``settle``
(``engine/api/calls.py``) runs at the start and the end of ``aconverse`` and ``aadvance`` and at
the normal end of a stream, after the manager released the conversation: it pushes the child (the
declared inputs only, never the parent's context), and when the child ended or its own budget is
spent it pops it with ONE merge of the declared ``returns`` and the final record. It takes the
target, the inputs and the budgets from the parent's own definition and context, never from the
record. Each row names the mutant it kills (census D1, D3, D7). Offline: a scripted model, a
one-class resolver, events and bounded waits instead of sleeps.
"""

from __future__ import annotations

import asyncio
import copy
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import pytest

from fsm_llm import API, create_handler
from fsm_llm.constants import DEFAULT_MAX_STACK_DEPTH
from fsm_llm.core.definitions import (
    ConversationBusyError,
    FSMDefinition,
    FSMError,
    LLMResponseError,
)
from fsm_llm.engine.api.calls import settle
from fsm_llm.engine.kinds import CallExecutor
from fsm_llm.engine.kinds.targets import ResolvedTarget
from fsm_llm.handlers import HandlerTiming
from fsm_llm.runtime import EventLog, EventType
from fsm_llm.testing import ScriptedLLM
from tests.test_fsm_llm import kind_corpus as kc
from tests.test_fsm_llm.test_async_cancel import _gate_is_free
from tests.test_fsm_llm.test_call_state import (
    _call,
    _call_state,
    _choose,
    _edge,
    _failure_edge,
    _parent,
)

_BOUND = 10  # seconds: a wait that would hang fails instead
_TARGET = "reasoning.analytical"
_OTHER = "reasoning.deductive"


def _child_definition(
    steps: int = 1, greeting: str | None = None, speaks: str | None = None
) -> dict[str, Any]:
    """``c1 -> c2 -> ... -> c<steps+1>``, every state silent but ``c1`` when it greets and
    ``c2`` when it speaks (a step that reaches the model)."""
    ids = [f"c{n}" for n in range(1, steps + 2)]
    states = []
    for n, state_id in enumerate(ids):
        said = greeting if n == 0 else speaks if n == 1 else None
        fields: dict[str, Any] = {"response_instructions": said or ""}
        if n < len(ids) - 1:
            fields["transitions"] = [kc._go(ids[n + 1])]
        states.append(kc._state(state_id, **fields))
    return kc._fsm("Child", "c1", *states)


class _Resolver:
    """Answers a definition per id and records every question."""

    def __init__(self, known: dict[str, dict[str, Any]]) -> None:
        self.definitions = {k: FSMDefinition(**v) for k, v in known.items()}
        self.asked: list[str] = []

    def resolve_target(self, target_id: str, *, conversation_id: str | None = None):
        self.asked.append(target_id)
        definition = self.definitions.get(target_id)
        if definition is None:
            return None
        return ResolvedTarget(
            definition, f"template:{target_id}", f"instance:{target_id}"
        )


class _Gate(ScriptedLLM):
    """After ``arm()`` the next ``agenerate_response`` sets ``entered`` and waits until
    ``release`` is set (or the task is cancelled); every other call answers as a ``ScriptedLLM``."""

    armed = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    def arm(self) -> None:
        self.armed = True

    async def agenerate_response(self, request):
        if self.armed:
            self.armed = False
            self.entered.set()
            await self.release.wait()
        return await super().agenerate_response(request)


@dataclass
class Flow:
    api: API
    cid: str
    frame: str  # the id of the PARENT frame
    resolver: _Resolver
    llm: ScriptedLLM

    def record(self) -> Any:
        return self.api.fsm_manager.copy_raw_context_data(self.frame).get("call_result")

    def parent_state(self) -> str:
        return self.api.fsm_manager.get_conversation_state(self.frame)

    def depth(self) -> int:
        return self.api.get_stack_depth(self.cid)

    def child_frame(self) -> str:
        return self.api.conversation_stacks[self.cid][1].conversation_id


async def _flow(
    *,
    config: dict[str, Any] | None = None,
    steps: int = 1,
    greeting: str | None = None,
    speaks: str | None = None,
    context: dict[str, Any] | None = None,
    llm: ScriptedLLM | None = None,
    known: dict[str, dict[str, Any]] | None = None,
    handlers: bool = True,
    api_kwargs: dict[str, Any] | None = None,
    **state_overrides: Any,
) -> Flow:
    """A parent API that has moved from ``start`` into the call state ``work``: the NEXT step
    runs the executor. The child writes ``solution`` as it enters its last state."""
    resolver = _Resolver(
        known
        if known is not None
        else {_TARGET: _child_definition(steps, greeting, speaks)}
    )
    llm = ScriptedLLM(default_response="r") if llm is None else llm
    state_overrides.setdefault(
        "response_instructions", ""
    )  # silent: None is not silent
    parts: list[Any] = []
    if handlers:
        parts.append(
            create_handler("answer")
            .on_state_entry(f"c{steps + 1}")
            .do(lambda ctx: {"solution": f"solved {ctx.get('topic')}"})
        )
    api = API.from_definition(
        _parent(config, **state_overrides),
        llm_interface=llm,
        targets=resolver,
        handlers=parts,
        **(api_kwargs or {}),
    )
    cid, _ = await api.astart_conversation({"topic": "why", **(context or {})})
    frame = api.conversation_stacks[cid][0].conversation_id
    await api.aadvance(cid)  # start -> work: the arrival runs nothing
    flow = Flow(api, cid, frame, resolver, llm)
    assert flow.parent_state() == "work" and flow.record() is None
    return flow


def _final(status: str, reason: str | None, target: str = _TARGET) -> dict[str, Any]:
    return {
        "status": status,
        "state": "work",
        "reason": reason,
        "target": target,
        "template_identity": f"template:{target}",
        "instance_identity": f"instance:{target}",
    }


class TestTheHappyPath:
    async def test_a_run_pushes_pops_once_and_fires_the_success_edge(self):
        """The whole flow through ``run_until_terminal``: the executor writes the request, settle
        pushes the child at the end of that step, the child runs, its last step pops it with the
        declared return and the final record in one merge, and the success edge fires on the next
        step. Kills: a settle that never pushes, never pops or pops twice."""
        flow = await _flow()
        api, cid = flow.api, flow.cid
        results = await api.arun_until_terminal(cid, max_steps=10)
        assert flow.parent_state() == "done" and flow.depth() == 1
        assert flow.record() == _final("ok", None)
        assert api.get_data(cid)["answer"] == "solved why"
        assert "solution" not in api.get_data(cid)  # only the declared return is merged
        assert [r.state_after for r in results] == ["work", "c2", "done"]
        assert list(api.fsm_manager.instances) == [flow.frame]  # no child left

    async def test_the_child_receives_only_the_declared_inputs(self):
        """Kills (D1): ``inherit_context=True`` or an input set built from the whole context (the
        child would hold ``_approval_granted``, the secret-looking entry and the extras)."""
        flow = await _flow(
            steps=2,
            context={
                "_approval_granted": {"call": "x"},
                "api_key": "sk-secret-value",
                "unrelated": "do not hand over",
            },
        )
        await flow.api.aadvance(
            flow.cid
        )  # the executor writes the request, settle pushes
        assert flow.depth() == 2
        raw = flow.api.fsm_manager.copy_raw_context_data(flow.child_frame())
        assert raw["topic"] == "why"
        public = {k for k in raw if not k.startswith("_")}
        assert public == {"topic"}
        assert "_approval_granted" not in raw
        assert "sk-secret-value" not in repr(raw) and "do not hand over" not in repr(
            raw
        )

    async def test_the_values_the_child_gets_are_copies(self):
        """A mutation of a handed-over list inside the child never reaches the parent."""
        flow = await _flow(steps=2, context={"topic": ["a"]})
        await flow.api.aadvance(flow.cid)
        child = flow.api.fsm_manager.instances[flow.child_frame()]
        child.context.data["topic"].append("changed")
        assert flow.api.fsm_manager.copy_raw_context_data(flow.frame)["topic"] == ["a"]

    async def test_the_result_key_is_one_handler_only_write_of_the_pop(self):
        """The pop writes ``answer`` and the final record in the SAME delta: a handler that
        watches either key runs once. Kills: a pop that merges the returns and then the record."""
        passes: list[set[str]] = []

        def seen(ctx: dict[str, Any]) -> dict[str, Any]:
            passes.append(set(ctx))
            return {}

        flow = await _flow()
        flow.api.register_handler(
            create_handler("watch")
            .at(HandlerTiming.CONTEXT_UPDATE)
            .when_keys_updated("answer", "call_result")
            .do(seen)
        )
        before = len(passes)
        await flow.api.arun_until_terminal(flow.cid, max_steps=10)
        # one pass for the executor's own record write and one for the single merge of the pop
        assert len(passes) - before == 2


class TestTheSyncNames:
    def test_the_sync_run_drives_the_same_flow_with_no_event_loop(self):
        """``run_until_terminal`` on the sync names pushes, pops and merges as the async run
        does (``settle`` is reached through ``run_sync``). Kills: a settle that needs a loop."""
        resolver = _Resolver({_TARGET: _child_definition()})
        api = API.from_definition(
            _parent(response_instructions=""),
            llm_interface=ScriptedLLM(default_response="r"),
            targets=resolver,
            handlers=[
                create_handler("answer")
                .on_state_entry("c2")
                .do(lambda ctx: {"solution": f"solved {ctx.get('topic')}"})
            ],
        )
        cid, _ = api.start_conversation({"topic": "why"})
        results = api.run_until_terminal(cid, max_steps=10)
        assert [r.state_after for r in results] == ["work", "work", "c2", "done"]
        assert api.get_data(cid)["answer"] == "solved why"
        assert api.get_stack_depth(cid) == 1


class TestEveryFailureHasItsReason:
    async def test_a_missing_return_key_fails_the_call_and_merges_nothing(self):
        flow = await _flow(config=_call(returns={"answer": "never_written"}))
        await flow.api.arun_until_terminal(flow.cid, max_steps=10)
        assert flow.record() == _final("failed", "return_keys_missing")
        assert "answer" not in flow.api.get_data(flow.cid)
        assert flow.parent_state() == "apology" and flow.depth() == 1

    async def test_a_spent_child_budget_aborts_and_pops_the_frame(self):
        """Two steps of work and ``max_steps=1``: the engine pops its OWN frame at the budget
        stop (the one case of D-023) with nothing merged. Kills: a settle that leaves the spent
        frame (the run loop would then raise on the child's next turn)."""
        flow = await _flow(config=_call(max_steps=1), steps=2)
        await flow.api.arun_until_terminal(flow.cid, max_steps=10)
        assert flow.record() == _final("aborted", "budget_spent")
        assert "answer" not in flow.api.get_data(flow.cid)
        assert flow.parent_state() == "apology" and flow.depth() == 1
        assert list(flow.api.fsm_manager.instances) == [flow.frame]

    async def test_the_budget_of_the_call_is_the_budget_of_the_frame(self):
        flow = await _flow(config=_call(max_steps=5, max_seconds=60), steps=2)
        await flow.api.aadvance(flow.cid)
        manager = flow.api.fsm_manager
        budget = manager._turn_budgets[flow.child_frame()]
        assert (budget.max_steps, budget.max_seconds) == (5, 60.0)

    async def test_a_child_that_ends_exactly_on_its_last_step_is_ok_not_aborted(self):
        flow = await _flow(config=_call(max_steps=1), steps=1)
        await flow.api.arun_until_terminal(flow.cid, max_steps=10)
        assert flow.record() == _final("ok", None)

    async def test_a_stack_at_the_depth_limit_refuses_the_push(self):
        """The call frame is the 10th: settle maps the depth to ``depth_refused`` (the executor
        cannot see the stack) and the failure edge fires on the next turn."""
        resolver = _Resolver({_TARGET: _child_definition()})
        api = API.from_definition(
            kc._fsm("Root", "r", kc._state("r", response_instructions="")),
            llm_interface=ScriptedLLM(default_response="r"),
            targets=resolver,
        )
        cid, _ = await api.astart_conversation()
        plain = kc._fsm("Plain", "p", kc._state("p", response_instructions=""))
        for _ in range(DEFAULT_MAX_STACK_DEPTH - 2):
            await api.apush_fsm(cid, copy.deepcopy(plain), inherit_context=False)
        await api.apush_fsm(
            cid,
            _parent(response_instructions=""),
            {"topic": "why"},
            inherit_context=False,
        )
        assert api.get_stack_depth(cid) == DEFAULT_MAX_STACK_DEPTH
        await api.aadvance(cid)  # start -> work
        await api.aadvance(
            cid
        )  # the executor writes the request; settle refuses the push
        frame = api.conversation_stacks[cid][-1].conversation_id
        record = api.fsm_manager.copy_raw_context_data(frame)["call_result"]
        assert (record["status"], record["reason"]) == ("failed", "depth_refused")
        assert api.get_stack_depth(cid) == DEFAULT_MAX_STACK_DEPTH
        assert (
            resolver.asked == [_TARGET] * 2
        )  # the executor, then settle; refused by the depth
        step = await api.aadvance(cid)
        assert step.state_after == "apology"

    async def test_a_push_the_definition_refuses_is_push_failed(self):
        """A target whose definition the stack refuses (a classifier interface it cannot send):
        ``push_failed``, and nothing is left behind. Kills: a refusal that escapes the entry."""
        flow = await _flow()
        flow.resolver.definitions[_TARGET] = FSMDefinition(**_child_definition())
        original = flow.api.apush_fsm

        async def refused(*args: Any, **kwargs: Any) -> str:
            raise ValueError("the stack refuses this definition")

        flow.api.apush_fsm = refused  # type: ignore[method-assign]
        try:
            await flow.api.aadvance(flow.cid)
        finally:
            flow.api.apush_fsm = original  # type: ignore[method-assign]
        assert flow.record() == _final("failed", "push_failed")
        assert flow.depth() == 1
        assert (await flow.api.aadvance(flow.cid)).state_after == "apology"

    async def test_a_provider_error_of_the_greeting_keeps_the_request(self):
        """An outage in the child's greeting propagates (v1 maps no ``provider_error``): the
        record stays ``running``, nothing is orphaned and the next entry asks again."""
        llm = ScriptedLLM(default_response="r")
        flow = await _flow(greeting="Hello.", llm=llm, steps=2)
        with patch.object(
            llm, "generate_response", side_effect=LLMResponseError("down")
        ):
            with pytest.raises(LLMResponseError):
                await flow.api.aadvance(flow.cid)
        assert flow.record()["status"] == "running" and flow.depth() == 1
        assert _gate_is_free(flow.api, flow.cid)
        await flow.api.aadvance(flow.cid)  # the start of this entry pushes it again
        assert flow.depth() == 2


class TestTheRecordIsNotTrusted:
    @staticmethod
    def _forge(flow: Flow, **fields: Any) -> None:
        record = {
            "status": "running",
            "mode": "frame",
            "target": "evil.id",
            "template_identity": "t",
            "instance_identity": "i",
            "request": {"target": "evil.id", "inputs": {"_approval_granted": True}},
            **fields,
        }
        flow.api.fsm_manager.update_conversation_context(
            flow.frame, {"call_result": record}
        )

    async def test_a_forged_record_cannot_name_a_target_the_state_does_not(self):
        """The state calls ``_TARGET``; the record says ``evil.id`` and carries a grant in its
        request. Settle pushes ``_TARGET`` with the declared input and nothing else. Kills (D3):
        a settle that reads the target or the inputs from ``request``."""
        known = {_TARGET: _child_definition(2), "evil.id": _child_definition(2)}
        flow = await _flow(known=known)
        self._forge(flow)
        await flow.api.aadvance(flow.cid)
        assert flow.depth() == 2
        assert flow.resolver.asked == [
            _TARGET
        ]  # the executor skipped, settle asked once
        raw = flow.api.fsm_manager.copy_raw_context_data(flow.child_frame())
        assert raw["topic"] == "why" and "_approval_granted" not in raw

    async def test_a_forged_record_cannot_reach_a_target_outside_the_closed_list(self):
        """A ``choose`` state whose key holds a value outside the list: whatever the record says,
        settle fails the call (``unknown_target``) and pushes nothing."""
        config = {k: v for k, v in _call().items() if k != "target"}
        config["choose"] = _choose()
        known = {
            _TARGET: _child_definition(),
            _OTHER: _child_definition(),
            "evil.id": _child_definition(),
        }
        flow = await _flow(config=config, known=known, context={"picked": "evil.id"})
        self._forge(flow, target=_TARGET)
        await flow.api.aadvance(flow.cid)
        assert flow.depth() == 1 and flow.resolver.asked == []
        record = flow.record()
        assert (record["status"], record["reason"]) == ("failed", "unknown_target")

    async def test_a_forged_record_takes_the_fallback_not_its_own_target(self):
        config = {k: v for k, v in _call().items() if k != "target"}
        config["choose"] = _choose(fallback=_OTHER)
        known = {_TARGET: _child_definition(2), _OTHER: _child_definition(2)}
        flow = await _flow(config=config, known=known, context={"picked": "evil.id"})
        self._forge(flow, target=_TARGET)
        await flow.api.aadvance(flow.cid)
        assert flow.depth() == 2 and flow.resolver.asked == [_OTHER]

    async def test_a_record_with_another_mode_is_left_alone(self):
        """Rule (b) re-requests a ``frame`` record only (the delegated mode of a later phase
        owns its own)."""
        flow = await _flow()
        self._forge(flow, mode="delegated", target=_TARGET)
        before = flow.record()
        await flow.api.aadvance(flow.cid)
        assert flow.depth() == 1 and flow.resolver.asked == []
        assert flow.record() == before

    @pytest.mark.parametrize("status", ["ok", "failed", "aborted"])
    async def test_a_final_record_pushes_nothing(self, status):
        flow = await _flow()
        self._forge(flow, status=status, target=_TARGET)
        await flow.api.aadvance(flow.cid)
        assert flow.depth() == 1 and flow.resolver.asked == []


class TestCancel:
    async def _cancelled_in_the_greeting(self, flow: Flow, llm: _Gate) -> None:
        llm.arm()
        task = asyncio.create_task(flow.api.aadvance(flow.cid))
        await asyncio.wait_for(llm.entered.wait(), _BOUND)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, _BOUND)

    async def test_a_cancel_in_the_greeting_leaves_no_orphan_and_the_next_entry_heals(
        self,
    ):
        """The cancel lands after the parent committed its request, inside the push. Nothing is
        orphaned, the gate and the root mark are free, the request is still ``running`` and the
        next entry pushes. Kills: a cancel swallowed by settle or by the push (the task would
        return), a mark released outside ``finally`` (the next entry would be refused), a
        request cleared by the cancel."""
        llm = _Gate(default_response="r")
        flow = await _flow(greeting="Hello.", llm=llm, steps=2)
        await self._cancelled_in_the_greeting(flow, llm)
        api = flow.api
        assert flow.depth() == 1 and list(api.fsm_manager.instances) == [flow.frame]
        assert not api._pending_push_ids
        assert _gate_is_free(api, flow.cid)
        assert flow.record()["status"] == "running"
        assert api._call_marks.settling == set()
        step = await api.aadvance(flow.cid)  # the start of this entry heals
        assert flow.depth() == 2
        assert step.state_before == "c1"  # the entry then ran the child's first step

    async def test_a_cancel_between_the_commit_and_the_push_heals_on_the_next_entry(
        self,
    ):
        flow = await _flow(steps=2)
        original = flow.api.apush_fsm

        async def cancelled(*args: Any, **kwargs: Any) -> str:
            raise asyncio.CancelledError

        flow.api.apush_fsm = cancelled  # type: ignore[method-assign]
        try:
            with pytest.raises(asyncio.CancelledError):
                await flow.api.aadvance(flow.cid)
        finally:
            flow.api.apush_fsm = original  # type: ignore[method-assign]
        assert flow.depth() == 1 and flow.record()["status"] == "running"
        assert flow.api._call_marks.settling == set()
        await flow.api.aadvance(flow.cid)
        assert flow.depth() == 2

    async def test_a_cancel_in_a_step_of_the_child_keeps_the_frame_and_the_request(
        self,
    ):
        """The child's turn is cancelled mid-model-call: the engine rolls that turn back, the
        frame stays, the parent record stays ``running`` and nothing is popped."""
        llm = _Gate(default_response="r")
        flow = await _flow(steps=2, speaks="Speak.", llm=llm)
        await flow.api.aadvance(
            flow.cid
        )  # the executor writes the request; settle pushes
        assert flow.depth() == 2
        llm.arm()
        task = asyncio.create_task(flow.api.aadvance(flow.cid))
        await asyncio.wait_for(llm.entered.wait(), _BOUND)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, _BOUND)
        assert flow.depth() == 2 and flow.record()["status"] == "running"
        assert flow.api.get_current_state(flow.cid) == "c1"
        assert _gate_is_free(flow.api, flow.cid)
        assert flow.api._call_marks.settling == set()


class TestTwoDrivers:
    async def test_a_second_entry_during_the_push_is_refused_and_does_not_wait(self):
        """One driver is inside the push (the child's greeting is parked): a second entry on the
        same root is refused with ``FSMError`` at once. Kills: no root mark, a mark that waits."""
        llm = _Gate(default_response="r")
        flow = await _flow(greeting="Hello.", llm=llm, steps=2)
        llm.arm()
        first = asyncio.create_task(flow.api.aadvance(flow.cid))
        await asyncio.wait_for(llm.entered.wait(), _BOUND)
        with pytest.raises(FSMError, match="settled"):
            await asyncio.wait_for(flow.api.aconverse("hi", flow.cid), _BOUND)
        with pytest.raises(FSMError, match="settled"):
            await asyncio.wait_for(flow.api.aadvance(flow.cid), _BOUND)
        llm.release.set()
        await asyncio.wait_for(first, _BOUND)
        assert flow.depth() == 2
        assert flow.api._call_marks.settling == set()

    async def test_settle_never_acts_while_a_turn_of_the_top_frame_is_in_flight(self):
        """The call state speaks and its Pass 2 is parked: the request is in the context but the
        turn may still roll back. A settle from another task does nothing and does not block.
        Kills: a settle that pushes on a request a rolled-back turn would remove."""
        llm = _Gate(default_response="r")
        flow = await _flow(llm=llm, response_instructions="Working.")
        llm.arm()
        turn = asyncio.create_task(flow.api.aadvance(flow.cid))
        await asyncio.wait_for(llm.entered.wait(), _BOUND)
        assert await asyncio.wait_for(settle(flow.api, flow.cid), _BOUND) == []
        assert flow.depth() == 1
        llm.release.set()
        await asyncio.wait_for(turn, _BOUND)
        assert flow.depth() == 2  # the turn's own end settled it

    async def test_a_plain_conversation_pays_no_manager_read(self):
        """The cheap exit: a stack with no call state asks the manager nothing."""
        never = kc._when("never", {"==": [{"var": "never"}, True]})
        api = API.from_definition(
            kc._fsm(
                "Plain",
                "p",
                kc._state(
                    "p",
                    response_instructions="Hi.",
                    transitions=[kc._go("q", conditions=never)],
                ),
                kc._state("q", response_instructions="Bye."),
            ),
            llm_interface=ScriptedLLM(default_response="r"),
        )
        cid, _ = await api.astart_conversation()
        seen: list[str] = []
        manager = api.fsm_manager
        for name in ("peek_turn", "get_conversation_state", "has_conversation_ended"):
            real = getattr(manager, name)

            def spy(*a: Any, _name: str = name, _real: Any = real, **k: Any) -> Any:
                seen.append(_name)
                return _real(*a, **k)

            setattr(manager, name, spy)
        await api.aconverse("hi", cid)
        assert seen == []


class TestStreamsAndReplies:
    async def test_the_child_greeting_is_part_of_the_reply_of_the_step(self):
        """``AdvanceResult.response`` is the reply of the step then the greeting, joined by a
        newline (D-023): the parent speaks in its call state and the child greets."""
        llm = ScriptedLLM(
            responses={"work": "Working.", "c1": "Hello."}, default_response="r"
        )
        flow = await _flow(greeting="Hello.", llm=llm, response_instructions="Working.")
        step = await flow.api.aadvance(flow.cid)
        assert step.response == "Working.\nHello." and flow.depth() == 2
        assert (step.state_before, step.state_after) == ("work", "work")

    async def test_a_silent_parent_reply_is_the_greeting_alone(self):
        flow = await _flow(greeting="Hello.")
        step = await flow.api.aadvance(flow.cid)
        assert step.response == "r" and flow.depth() == 2

    async def test_aconverse_returns_the_same_join(self):
        llm = ScriptedLLM(
            responses={"work": "Working.", "c1": "Hello."}, default_response="r"
        )
        flow = await _flow(greeting="Hello.", llm=llm, response_instructions="Working.")
        reply = await flow.api.aconverse("go", flow.cid)
        assert reply == "Working.\nHello." and flow.depth() == 2

    async def test_the_next_message_goes_to_the_child(self):
        llm = ScriptedLLM(
            responses={"work": "Working.", "c1": "Hello."}, default_response="r"
        )
        flow = await _flow(
            steps=2, greeting="Hello.", llm=llm, response_instructions="Working."
        )
        await flow.api.aconverse("go", flow.cid)
        assert flow.api.get_current_state(flow.cid) == "c1"
        await flow.api.aconverse("again", flow.cid)
        assert flow.api.get_current_state(flow.cid) == "c2"

    async def test_a_stream_yields_the_greeting_after_its_reply_at_its_normal_end(self):
        llm = ScriptedLLM(
            responses={"work": "Working.", "c1": "Hello."}, default_response="r"
        )
        flow = await _flow(greeting="Hello.", llm=llm, response_instructions="Working.")
        chunks = [c async for c in flow.api.aadvance_stream(flow.cid)]
        assert "".join(chunks) == "Working.\nHello." and flow.depth() == 2

    async def test_an_abandoned_stream_defers_the_push_to_the_next_entry(self):
        """Settle never runs in a stream's ``finally``: the request stays ``running`` and the
        next entry pushes. Kills: a settle in the ``finally`` of the stream."""
        flow = await _flow(response_instructions="Working.", steps=2)
        async with aclosing(flow.api.aadvance_stream(flow.cid)) as stream:
            async for _ in stream:
                break  # the reader walks away after the first chunk
        assert flow.depth() == 1 and flow.record()["status"] == "running"
        assert _gate_is_free(flow.api, flow.cid)
        await flow.api.aadvance(flow.cid)
        assert flow.depth() == 2


class TestOnePopOneMerge:
    async def test_a_refused_end_after_the_merge_is_retried_without_a_second_merge(
        self,
    ):
        """The merge lands, ending the child is refused: the entry raises, the frame stays, the
        record is final. The next entry pops it with NO second merge. Kills: a retry that merges
        again (the handler would run twice, ``stack/pop`` twice)."""
        passes: list[Any] = []
        flow = await _flow(steps=1)
        flow.api.register_handler(
            create_handler("count")
            .at(HandlerTiming.CONTEXT_UPDATE)
            .when_keys_updated("answer")
            .do(lambda ctx: passes.append(ctx.get("answer")) or {})
        )
        await flow.api.aadvance(flow.cid)  # the request, the push
        child = flow.child_frame()
        manager = flow.api.fsm_manager
        real_end = manager.end_conversation
        refusals = []

        def refuse_once(conversation_id: str, *args: Any, **kwargs: Any) -> None:
            if conversation_id == child and not refusals:
                refusals.append(conversation_id)
                raise ConversationBusyError("refused", conversation_id=conversation_id)
            return real_end(conversation_id, *args, **kwargs)

        manager.end_conversation = refuse_once  # type: ignore[method-assign]
        with pytest.raises(ConversationBusyError):
            await flow.api.aadvance(
                flow.cid
            )  # the child's last step; its pop is refused
        assert flow.depth() == 2 and flow.record() == _final("ok", None)
        assert passes == ["solved why"]
        # a host edit of the merged key between the refusal and the retry survives the retry:
        # the retry pops and merges NOTHING (a second merge would write the child's value back)
        manager.update_conversation_context(flow.frame, {"answer": "host edit"})
        passes.clear()
        await flow.api.aadvance(flow.cid)  # the start of this entry pops, no merge
        assert flow.depth() == 1 and passes == []
        assert flow.api.get_data(flow.cid)["answer"] == "host edit"
        assert flow.record() == _final("ok", None)
        assert flow.api._call_marks.ending == set()

    async def test_the_log_holds_one_push_and_one_pop(self, tmp_path):
        log = EventLog(tmp_path)
        flow = await _flow(api_kwargs={"event_log": log})
        await flow.api.arun_until_terminal(flow.cid, max_steps=10)
        kinds = [e["type"] for e in log.read(flow.cid)]
        assert kinds.count(EventType.STACK_PUSH.value) == 1
        assert kinds.count(EventType.STACK_POP.value) == 1


class TestTheHooks:
    async def test_the_executor_hears_of_the_push_and_the_pop(self):
        heard: list[tuple[str, str, str]] = []

        class Hearing(CallExecutor):
            async def on_pushed(self, root_id, parent_id, child_id, state, record):
                heard.append(("pushed", parent_id, child_id))

            async def on_ended(self, root_id, parent_id, child_id, state, record):
                heard.append(("ended", parent_id, child_id))
                assert record["status"] == "ok"

        flow = await _flow(api_kwargs={"executors": [Hearing()]})
        await flow.api.arun_until_terminal(flow.cid, max_steps=10)
        assert [h[0] for h in heard] == ["pushed", "ended"]
        assert heard[0][1] == heard[1][1] == flow.frame and heard[0][2] == heard[1][2]


def _sequence(*, loop: bool = False) -> dict[str, Any]:
    """``start -> work -> second -> done``: two call states with the DEFAULT result key
    (``call_result``), each on its own target. With ``loop`` the ``second`` state goes back to
    ``work`` until it has been entered twice (a handler counts ``visits``)."""
    first = _call(target=_TARGET, returns={"answer": "solution"})
    second = _call(target=_OTHER, returns={"answer2": "solution"})
    definition = _parent(
        first,
        extra_states={
            "second": _call_state(second, id="second", response_instructions="")
        },
        response_instructions="",
    )
    states = definition["states"]
    states["start"]["transitions"] = [states["start"]["transitions"][0]]
    ok = {"==": [{"var": "call_result.status"}, "ok"]}
    states["work"]["transitions"] = [
        _edge("second", 1, ok),
        _failure_edge(),
    ]
    if loop:
        visits = {"var": "visits"}
        states["second"]["transitions"] = [
            _edge("work", 1, {"and": [ok, {"<": [visits, 2]}]}),
            _edge("done", 1, {"and": [ok, {">=": [visits, 2]}]}),
            _failure_edge(),
        ]
    return definition


def _two_child_targets() -> dict[str, dict[str, Any]]:
    one, two = _child_definition(), _child_definition()
    one["name"], two["name"] = "ChildOne", "ChildTwo"
    return {_TARGET: one, _OTHER: two}


async def _run_sequence(
    definition: dict[str, Any],
) -> tuple[API, str, list[str]]:
    """Run ``definition`` to its end; returns ``(api, id, names of the children pushed)``."""
    handlers = [
        create_handler("answer")
        .on_state_entry("c2")
        .do(lambda ctx: {"solution": f"solved {ctx.get('topic')}"}),
        create_handler("count")
        .on_state_entry("second")
        .do(lambda ctx: {"visits": (ctx.get("visits") or 0) + 1}),
    ]
    api = API.from_definition(
        definition,
        llm_interface=ScriptedLLM(default_response="r"),
        targets=_Resolver(_two_child_targets()),
        handlers=handlers,
    )
    pushed: list[str] = []
    original = api.apush_fsm

    async def spy(root_id: str, child: Any, *args: Any, **kwargs: Any) -> Any:
        pushed.append(child.name)
        return await original(root_id, child, *args, **kwargs)

    api.apush_fsm = spy  # type: ignore[method-assign]
    cid, _ = await api.astart_conversation({"topic": "why"})
    await api.arun_until_terminal(cid, max_steps=30)
    return api, cid, pushed


class TestTwoCallStates:
    async def test_two_call_states_in_sequence_both_run(self):
        """Both states use the default ``call_result``: the second must not take the first's
        final ``ok`` for its own answer. Kills (D-025): a skip on any non-None value, and a final
        record with no ``state`` stamp (the second state would then skip)."""
        api, cid, pushed = await _run_sequence(_sequence())
        assert pushed == ["ChildOne", "ChildTwo"]
        assert api.get_stack_depth(cid) == 1
        data = api.get_data(cid)
        assert data["answer"] == "solved why" and data["answer2"] == "solved why"
        record = data["call_result"]
        assert (record["status"], record["state"], record["target"]) == (
            "ok",
            "second",
            _OTHER,
        )
        assert (
            api.fsm_manager.get_conversation_state(
                api.conversation_stacks[cid][0].conversation_id
            )
            == "done"
        )

    async def test_a_state_revisited_after_another_call_state_runs_again(self):
        """``work -> second -> work -> second -> done``: the record in the key belongs to the
        OTHER call state at each re-entry, so each visit runs its call (four pushes in order).
        The stale-record skip of the SAME state is pinned in ``test_call_log``."""
        api, cid, pushed = await _run_sequence(_sequence(loop=True))
        assert pushed == ["ChildOne", "ChildTwo", "ChildOne", "ChildTwo"]
        assert api.get_stack_depth(cid) == 1
        assert api.get_data(cid)["call_result"]["state"] == "second"

    async def test_settle_never_applies_a_running_record_of_another_state(self):
        """A ``running`` record stamped by another state (a handler, a snapshot, a log) sits in
        the key while the parent is in ``work``: settle pushes nothing. Kills (D-025): a settle
        that reads the status and mode and not the stamp."""
        flow = await _flow(steps=2)
        foreign = {
            "status": "running",
            "state": "elsewhere",
            "target": _OTHER,
            "mode": "frame",
            "request": {"target": _OTHER, "inputs": {}},
        }
        flow.api.update_context(flow.cid, {"call_result": foreign})
        assert await settle(flow.api, flow.cid) == []
        assert flow.depth() == 1 and flow.record() == foreign
        assert flow.resolver.asked == []
