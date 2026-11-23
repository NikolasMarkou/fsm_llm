"""``API.apush_fsm``, the async twin of ``push_fsm`` (plan baa37f40, step 9, D-020).

``apush_fsm`` is the one body (a mirror of ``astart_conversation``) and ``push_fsm`` is the thin
``run_sync`` wrapper over it: the sync name keeps every patch interception and its no-loop mode.
The greeting of the pushed child awaits its model call, so a cancel can land there; a cancelled
push leaves no orphan child, no ``_pending_push_ids`` entry, no registered temp definition, the
turn gate free and the parent as it was, and it is never swallowed. Rows use events and bounded
waits, never sleeps or the clock. Each cancel row names the mutant it kills.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
from typing import Any

import pytest

from fsm_llm import API
from fsm_llm.constants import DEFAULT_MAX_STACK_DEPTH
from fsm_llm.core.definitions import FSMError
from fsm_llm.core.definitions import RunBudgetExceededError as Spent
from fsm_llm.runtime import EventLog, EventType
from fsm_llm.testing import ScriptedLLM
from tests.conftest import current_timing_factor
from tests.test_fsm_llm.support import _state
from tests.test_fsm_llm.test_async_cancel import _gate_is_free, _seen
from tests.test_fsm_llm.test_async_engine import _CHAT, _CHILD

# seconds: a wait that would hang fails instead (scaled under a tracer)
_BOUND = 10 * current_timing_factor()
_SILENT: dict[str, Any] = {
    "name": "Silent",
    "description": "A child with no response instructions: its greeting makes no model call",
    "initial_state": "quiet",
    "states": {
        "quiet": _state(
            "quiet", transitions=[{"target_state": "end", "description": "go"}]
        )
    }
    | {"end": _state("end")},
}


class _Parked(ScriptedLLM):
    """After ``arm()`` the first ``agenerate_response`` sets ``entered`` and waits for ever
    (until the task is cancelled); every other call answers like a ``ScriptedLLM``."""

    armed = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.entered = asyncio.Event()

    def arm(self) -> None:
        self.armed = True

    async def agenerate_response(self, request):
        if self.armed:
            self.armed = False
            self.entered.set()
            await asyncio.Event().wait()
        return await super().agenerate_response(request)


async def _api(cls=ScriptedLLM, **kwargs):
    llm = cls(default_response="r")
    api = API.from_definition(_CHAT(), llm_interface=llm, **kwargs)
    cid, _ = await api.astart_conversation()
    return api, llm, cid


def _kid() -> dict[str, Any]:
    return copy.deepcopy(_CHILD())


class TestTheTwins:
    async def test_the_async_push_returns_the_greeting_and_adds_the_frame(self):
        """Kills: an ``apush_fsm`` that is missing, that returns no greeting or adds no frame."""
        api, llm, cid = await _api()
        before = len(llm.call_history)
        reply = await api.apush_fsm(cid, _kid(), {"seed": 1}, inherit_context=False)
        assert reply == "r"
        assert api.get_stack_depth(cid) == 2
        assert len(llm.call_history) == before + 1  # the child's greeting, nothing else
        assert api.get_current_state(cid) == "x"
        assert api.get_data(cid)["seed"] == 1

    async def test_the_two_names_agree(self):
        """The same push through both names gives the same reply, depth, top state and frame."""
        sync_api, _, sync_cid = await _api()
        async_api, _, async_cid = await _api()
        kwargs: dict[str, Any] = {
            "context_to_pass": {"k": "v"},
            "return_context": {"r": 1},
            "shared_context_keys": ["k"],
            "inherit_context": False,
        }
        sync_reply = sync_api.push_fsm(sync_cid, _kid(), **kwargs)
        async_reply = await async_api.apush_fsm(async_cid, _kid(), **kwargs)
        assert sync_reply == async_reply
        frames = [
            [
                (f.fsm_id, f.return_context, f.shared_context_keys, f.preserve_history)
                for f in api.conversation_stacks[cid]
            ]
            for api, cid in ((sync_api, sync_cid), (async_api, async_cid))
        ]
        assert frames[0] == frames[1] and len(frames[0]) == 2
        for api, cid in ((sync_api, sync_cid), (async_api, async_cid)):
            assert api.get_current_state(cid) == "x"
            assert api.get_data(cid)["k"] == "v"

    def test_the_sync_name_is_the_wrapper_over_the_async_body(self, monkeypatch):
        """``push_fsm`` hands every argument to ``apush_fsm`` (the one body) and returns what it
        returns. Kills: a second sync body."""
        seen: list[tuple[Any, ...]] = []

        async def recorder(self, *args, **kwargs):
            seen.append((args, kwargs))
            return "from the body"

        monkeypatch.setattr(API, "apush_fsm", recorder)
        api = API.from_definition(_CHAT(), llm_interface=ScriptedLLM())
        out = api.push_fsm(
            "cid", {"d": 1}, {"a": 1}, {"b": 2}, ["c"], True, False, max_steps=3
        )
        assert out == "from the body"
        assert seen == [
            (
                ("cid", {"d": 1}, {"a": 1}, {"b": 2}, ["c"], True, False),
                {"max_steps": 3, "max_seconds": None},
            )
        ]

    def test_the_sync_name_needs_no_loop_and_keeps_its_patch(self):
        """No event loop runs here, and a patch of the manager's SYNC ``start_conversation``
        still intercepts the push. Kills: a sync name that awaits the async twin of the manager."""
        api = API.from_definition(
            _CHAT(), llm_interface=ScriptedLLM(default_response="r")
        )
        cid, _ = api.start_conversation()
        manager = api.fsm_manager
        calls: list[str] = []
        real = manager.start_conversation

        def spy(fsm_id, *args, **kwargs):
            calls.append(fsm_id)
            return real(fsm_id, *args, **kwargs)

        manager.start_conversation = spy  # type: ignore[method-assign]
        try:
            assert api.push_fsm(cid, _kid()) == "r"
        finally:
            del manager.start_conversation
        assert len(calls) == 1 and api.get_stack_depth(cid) == 2

    async def test_the_async_name_awaits_the_manager_twin(self):
        """In async mode the hop is the manager's ``astart_conversation``."""
        api, _, cid = await _api()
        manager = api.fsm_manager
        calls: list[str] = []
        real = manager.astart_conversation

        async def spy(fsm_id, *args, **kwargs):
            calls.append(fsm_id)
            return await real(fsm_id, *args, **kwargs)

        manager.astart_conversation = spy  # type: ignore[method-assign]
        try:
            await api.apush_fsm(cid, _kid())
        finally:
            del manager.astart_conversation
        assert len(calls) == 1

    def test_the_two_names_share_one_signature(self):
        sync = inspect.signature(API.push_fsm)
        twin = inspect.signature(API.apush_fsm)
        assert sync.parameters.keys() == twin.parameters.keys()
        assert [p.default for p in sync.parameters.values()] == [
            p.default for p in twin.parameters.values()
        ]
        assert inspect.iscoroutinefunction(API.apush_fsm)


class TestRefusals:
    async def test_an_unknown_conversation_is_refused_before_anything_registers(self):
        api, _, cid = await _api()
        with pytest.raises(FSMError, match="Conversation not found"):
            await api.apush_fsm("nope", _kid())
        assert api.get_stack_depth(cid) == 1
        assert not api._pending_push_ids and not api._temp_fsm_definitions

    @pytest.mark.parametrize("limit", [{"max_steps": 0}, {"max_seconds": -1.0}])
    async def test_a_bad_budget_is_refused_before_anything_registers(self, limit):
        api, _, cid = await _api()
        with pytest.raises(ValueError):
            await api.apush_fsm(cid, _kid(), **limit)
        assert set(api.fsm_manager.instances) == {cid}
        assert not api._pending_push_ids and not api._temp_fsm_definitions

    async def test_the_depth_limit_refuses_and_leaves_nothing_behind(self):
        api, _, cid = await _api()
        for _ in range(DEFAULT_MAX_STACK_DEPTH - 1):
            await api.apush_fsm(cid, copy.deepcopy(_SILENT))
        instances = set(api.fsm_manager.instances)
        with pytest.raises(FSMError, match="depth limit"):
            await api.apush_fsm(cid, copy.deepcopy(_SILENT))
        assert set(api.fsm_manager.instances) == instances
        assert not api._pending_push_ids
        assert api.get_stack_depth(cid) == DEFAULT_MAX_STACK_DEPTH

    async def test_the_frame_budget_is_the_argument(self):
        api, _, cid = await _api()
        await api.apush_fsm(cid, _kid(), max_steps=1)
        await api.aconverse("k", cid)
        with pytest.raises(Spent) as refused:
            await api.aconverse("k2", cid)
        assert refused.value.limit == 1


async def _cancel_greeting(api, llm, cid) -> asyncio.Task:
    """Start a push whose greeting parks, cancel it there, return the cancelled task."""
    llm.arm()
    task = asyncio.create_task(api.apush_fsm(cid, _kid()))
    await asyncio.wait_for(llm.entered.wait(), _BOUND)
    task.cancel()
    await asyncio.wait({task}, timeout=_BOUND)
    assert task.cancelled()  # never swallowed, never converted
    return task


class TestACancelledPush:
    @pytest.mark.parametrize("logged", [False, True])
    async def test_a_cancel_in_the_greeting_leaves_nothing_behind(
        self, logged, tmp_path
    ):
        """Kills: an orphan child instance, a leaked ``_pending_push_ids`` entry or temp
        definition, a held turn gate, a parent that changed, a cancel that was swallowed."""
        kwargs = {"event_log": EventLog(tmp_path / "log")} if logged else {}
        api, llm, cid = await _api(_Parked, **kwargs)
        before = _seen(api, cid)
        await _cancel_greeting(api, llm, cid)
        manager = api.fsm_manager
        assert api.get_stack_depth(cid) == 1
        assert set(manager.instances) == {cid}  # no orphan child
        assert set(manager._conversation_locks) == {cid}
        assert api.list_active_conversations() == [cid]
        assert not api._pending_push_ids
        assert not api._temp_fsm_definitions
        assert _gate_is_free(api, cid)
        assert not api._stack_lock.locked()
        assert _seen(api, cid) == before
        if logged:
            kinds = [e["type"] for e in api._event_log.read(cid)]
            assert EventType.STACK_PUSH.value not in kinds
            # the greeting request the cancel cut has its end: no request is left open
            assert kinds.count(EventType.LLM_REQUEST.value) == 1
            assert kinds.count(EventType.PROVIDER_ERROR.value) == 1

    async def test_the_parent_and_a_retry_work_after_the_cancel(self):
        api, llm, cid = await _api(_Parked)
        await _cancel_greeting(api, llm, cid)
        assert await api.aconverse("again", cid) == "r"
        assert await api.apush_fsm(cid, _kid()) == "r"
        assert api.get_stack_depth(cid) == 2
        assert not api._pending_push_ids

    async def test_the_stack_lock_is_never_held_while_the_greeting_waits(self):
        """Kills: ``_stack_lock`` held across the manager call (a suspended task would hold a
        plain thread lock that every sibling needing the stack then blocks the loop on)."""
        api, llm, cid = await _api(_Parked)
        llm.arm()
        task = asyncio.create_task(api.apush_fsm(cid, _kid()))
        await asyncio.wait_for(llm.entered.wait(), _BOUND)
        try:
            assert api._stack_lock.acquire(blocking=False)
            api._stack_lock.release()
            assert api.get_stack_depth(cid) == 1  # a sibling reader is not blocked
        finally:
            task.cancel()
            await asyncio.wait({task}, timeout=_BOUND)

    async def test_the_cancel_is_not_converted_into_an_error(self):
        """Kills: an ``except BaseException`` (or a bare ``except``) around the body that turns
        the cancel into an ``FSMError`` or a normal return."""
        api, llm, cid = await _api(_Parked)
        task = await _cancel_greeting(api, llm, cid)
        with pytest.raises(asyncio.CancelledError):
            await task
