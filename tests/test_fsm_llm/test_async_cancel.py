"""A cancelled async turn changes nothing (plan f150ec28, step 4, D-002).

The sweep runs a scripted turn once to count its model awaits N, then once per k in 1..N on a
double that parks the k-th await (``_Parking``), cancels the task there and checks what the
caller can see: state, data, metadata, working memory and history equal their pre-turn values,
the gate is free, no ERROR handler ran, a retry of the same turn answers like an uncancelled
run, and with the event log on the turn ends ``rolled_back``, every request has its end and a
fold of the log is the pre-turn conversation. A cancel is never swallowed: the task ends
cancelled. Rows use events and bounded waits, never sleeps or the clock.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

from fsm_llm import API, FileSessionStore
from fsm_llm.core.definitions import (
    ConversationBusyError,
    DataExtractionResponse,
    FSMError,
    ResponseGenerationResponse,
    RunBudgetExceededError,
)
from fsm_llm.engine.kinds import ConversationExecutor, StateExecutor
from fsm_llm.handlers import HandlerTiming, create_handler
from fsm_llm.llm import LLMInterface
from fsm_llm.memory import WorkingMemory
from fsm_llm.runtime import EventLog
from fsm_llm.runtime.stores import fold_session
from fsm_llm.testing import ScriptedLLM
from tests.test_fsm_llm.test_async_engine import (
    _CHAT,
    _CID,
    _RUN,
    _SCENARIOS,
    _VOLATILE,
    _collect,
    _Gated,
    _run_api,
    _stream_api,
    _Streaming,
)

_BOUND = 10  # seconds: a wait that would hang fails instead


class _Parking(_Streaming):
    """The ``at``-th model await after ``park`` (counted from 1; a stream chunk counts) sets
    ``entered``, runs ``dirty`` (the commits of a handler) and waits until it is cancelled."""

    calls = 0
    park_at = 0
    dirty: Any = None

    def park(self, at: int, dirty: Any = None) -> None:
        self.calls, self.park_at, self.dirty = 0, at, dirty
        self.entered.clear()

    async def _hop(self) -> None:
        self.calls += 1
        if self.calls == self.park_at:
            if self.dirty:
                self.dirty()
            self.entered.set()
            await asyncio.Event().wait()
        await super()._hop()


def _seen(api, cid) -> dict[str, Any]:
    """Everything a turn can touch that a caller can see (clock keys left out)."""
    instance = api.fsm_manager.instances[cid]
    context = instance.context
    memory = context.working_memory
    return {
        "state": instance.current_state,
        "data": {k: v for k, v in context.data.items() if k not in _VOLATILE},
        "metadata": dict(context.metadata),
        "memory": None if memory is None else memory.to_dict(),
        "history": list(context.conversation.exchanges),
        "visible": api.get_data(cid),
    }


def _dirty(api, cid):
    """Writes into every restorable place, as the commits of a handler would."""
    context = api.fsm_manager.instances[cid].context

    def write() -> None:
        context.data["junk"] = 1
        context.metadata["junk"] = 1
        context.working_memory.set("scratch", "junk", 1)

    return write


def _attach_memory(api, cid) -> None:
    api.fsm_manager.instances[cid].context.working_memory = WorkingMemory()


def _step(api, cid, message):
    return api.aadvance(cid) if message is None else api.aconverse(message, cid)


async def _cancel(llm: _Parking, work) -> asyncio.Task:
    """Run ``work`` until the parked await, cancel it there and return the cancelled task."""
    task = asyncio.create_task(work)
    await asyncio.wait_for(llm.entered.wait(), _BOUND)
    task.cancel()
    await asyncio.wait({task}, timeout=_BOUND)
    assert task.cancelled()  # never swallowed, never converted
    return task


def _gate_is_free(api, cid) -> bool:
    manager = api.fsm_manager
    return not manager._active_turns and not manager._conversation_locks[cid].locked()


async def _prepare(name, turn, log_dir=None, **api_kwargs):
    """A started conversation of a scenario with its first ``turn`` turns done, a working
    memory attached, and its double (script included) as it stands before the turn."""
    scenario = _SCENARIOS[name]
    llm = scenario.llm(_Parking)
    if log_dir is not None:
        api_kwargs["event_log"] = EventLog(log_dir)
    api = API.from_definition(scenario.fsm(), llm_interface=llm, **api_kwargs)
    cid, _ = await api.astart_conversation(scenario.context, _conversation_id=_CID)
    _attach_memory(api, cid)
    for _ in range(scenario.advance):
        await api.aadvance(cid)
    for message in scenario.turns[:turn]:
        await _step(api, cid, message)
    return api, llm, cid


def _log_rows(api, kind):
    return [e for e in api.event_log.read(_CID) if e["type"] == kind]


#: (scenario, index of the swept turn): a field extraction, a bulk extraction, a
#: classification, the extraction after a transition and Pass 2 (plain); an AMBIGUOUS
#: classifier tie; a completion state; message-free steps.
_SWEPT = [("plain", 0), ("plain", 1), ("tie", 0), ("completion", 0), ("steps", 1)]


class TestTheSweep:
    @pytest.mark.parametrize("logged", [False, True])
    @pytest.mark.parametrize(("name", "turn"), _SWEPT)
    async def test_a_cancel_at_every_model_await_restores_the_pre_turn_conversation(
        self, name, turn, logged, tmp_path
    ):
        """Kills: no arm around Pass 1 (a committed transition and extraction survive), no pop
        of the user message (an orphan exchange), a release outside the ``finally`` (the gate
        stays held), an arm that catches ``Exception`` only (every k fails), an ERROR handler
        that runs on a cancel, a swallowed cancel."""
        message = _SCENARIOS[name].turns[turn]
        api, llm, cid = await _prepare(
            name, turn, tmp_path / "control" if logged else None
        )
        llm.park(-1)  # counts only
        expected = await _step(api, cid, message)
        awaits, after = llm.calls, _seen(api, cid)
        assert awaits > 0
        fired: list[str] = []  # the ERROR handlers of every run append here
        for k in range(1, awaits + 1):
            where = tmp_path / f"k{k}" if logged else None
            api, llm, cid = await _prepare(name, turn, where)
            api.register_handler(
                create_handler("errors")
                .at(HandlerTiming.ERROR)
                .do(lambda context: fired.append("error"))
            )
            before, script = _seen(api, cid), list(llm.replies)
            llm.park(k, _dirty(api, cid))
            await _cancel(llm, _step(api, cid, message))
            assert _seen(api, cid) == before, f"k={k}"
            assert _gate_is_free(api, cid) and fired == [], f"k={k}"
            if logged:
                self._the_log_shows_a_rolled_back_turn(api, cid, before)
            llm.park(-1)
            llm.replies[:] = script
            assert await _step(api, cid, message) == expected, f"k={k}"
            assert _seen(api, cid) == after, f"k={k}"

    @staticmethod
    def _the_log_shows_a_rolled_back_turn(api, cid, before) -> None:
        events = list(api.event_log.read(_CID))
        kinds = [e["type"] for e in events]
        assert [e["outcome"] for e in events if e["type"] == "turn/end"][-1] == (
            "rolled_back"
        )
        assert kinds.count("llm/request") == (
            kinds.count("llm/response") + kinds.count("provider/error")
        )
        assert kinds.count("provider/error") >= 1
        folded = fold_session(events, api.event_log.get_json)
        assert folded is not None
        assert folded.current_state == before["state"]
        assert folded.conversation_history == before["history"]


class TestWhatElseACancelLeaves:
    async def test_a_cancelled_turn_still_counts_against_a_lifetime_budget(self):
        """Kills: a charge that moves into the rolled-back part (the cancelled turn is free)."""
        llm = _Parking(default_response="ok")
        api = API.from_definition(_CHAT(), llm_interface=llm)
        cid, _ = await api.astart_conversation(max_steps=2)
        llm.park(1)
        await _cancel(llm, api.aconverse("one", cid))
        llm.park(-1)
        assert await api.aconverse("two", cid) == "ok"
        with pytest.raises(RunBudgetExceededError):
            await api.aconverse("three", cid)

    async def test_a_session_store_saves_nothing_for_a_cancelled_turn(self, tmp_path):
        """Kills: an auto-save that runs on the way out of a cancelled turn."""
        store = FileSessionStore(tmp_path)
        llm = _Parking(default_response="ok")
        api = API.from_definition(_CHAT(), llm_interface=llm, session_store=store)
        cid, _ = await api.astart_conversation()
        llm.park(1)
        await _cancel(llm, api.aconverse("one", cid))
        assert store.load(cid) is None

    async def test_a_cancelled_greeting_leaves_nothing_registered(self):
        """Kills: a greeting without a cancel arm (an instance, a lock and a budget stay under
        an id nobody holds, and the same id is refused as live)."""
        llm = _Parking(default_response="ok")
        api = API.from_definition(_CHAT(), llm_interface=llm)
        llm.park(1)
        await _cancel(llm, api.astart_conversation(max_steps=3, _conversation_id=_CID))
        manager = api.fsm_manager
        assert manager.instances == {} and manager._conversation_locks == {}
        assert manager._turn_budgets == {}
        llm.park(-1)
        assert (await api.astart_conversation(_conversation_id=_CID))[0] == _CID


class _SyncOnly(LLMInterface):
    """A sync-only interface: its async twins are the default ones, on a worker thread. After
    ``hold`` its reply blocks until ``proceed`` is set."""

    def __init__(self) -> None:
        self.hold = False
        self.entered, self.proceed, self.done = (threading.Event() for _ in range(3))

    def generate_response(self, request):
        if self.hold:
            self.entered.set()
            self.proceed.wait(_BOUND)
        self.done.set()
        return ResponseGenerationResponse(message="late")


class TestASyncOnlyInterface:
    async def test_the_late_reply_lands_after_the_rollback_and_changes_nothing(self):
        """The documented hazard: the cancel returns at once and the worker thread runs on. The
        state is the pre-turn state before the late reply and after it. Kills: a rollback that
        waits for the worker, a late reply that is committed."""
        llm = _SyncOnly()
        api = API.from_definition(_CHAT(), llm_interface=llm)
        cid, _ = await api.astart_conversation()
        before = _seen(api, cid)
        llm.hold = True
        llm.done.clear()
        loop = asyncio.get_running_loop()
        task = asyncio.create_task(api.aconverse("hi", cid))
        try:
            assert await loop.run_in_executor(None, llm.entered.wait, _BOUND)
            task.cancel()
            await asyncio.wait({task}, timeout=_BOUND)
            assert task.cancelled() and not llm.done.is_set()  # returned at once
            assert _seen(api, cid) == before and _gate_is_free(api, cid)
        finally:
            llm.proceed.set()
        assert await loop.run_in_executor(None, llm.done.wait, _BOUND)
        assert _seen(api, cid) == before
        llm.hold = False
        assert await api.aconverse("again", cid) == "late"


# ----------------------------------------------------------------------------------------------
# Step 4.2: streams and the run loop
# ----------------------------------------------------------------------------------------------


class TestStreams:
    @pytest.mark.parametrize("how", ["cancel", "close"])
    async def test_a_cancel_mid_pass_2_rolls_back_while_aclose_keeps_the_partial_reply(
        self, how
    ):
        """At the same point (chunk 'a' delivered, the second chunk awaited): a cancel of the
        consumer rolls the whole turn back and persists no partial reply, ``aclose`` keeps
        today's abandonment rule (the user message and the partial reply stay). Kills: an arm
        that leaves the user exchange or the partial reply on a cancel, a close that rolls back."""
        api, llm = _stream_api(_Parking)
        cid, _ = await api.astart_conversation()
        _attach_memory(api, cid)
        before = _seen(api, cid)
        stream = api.aconverse_stream("hi", cid)
        assert await stream.__anext__() == "a"
        if how == "cancel":
            llm.park(1, _dirty(api, cid))
            await _cancel(llm, _next(stream))
            assert _seen(api, cid) == before
        else:
            await stream.aclose()
            assert api.get_conversation_history(cid)[-2:] == [
                {"user": "hi"},
                {"system": "a"},
            ]
        assert _gate_is_free(api, cid)

    @pytest.mark.parametrize("logged", [False, True])
    async def test_a_task_cancelled_at_any_chunk_wait_leaves_the_pre_turn_conversation(
        self, logged, tmp_path
    ):
        """The consumer task is cancelled while a chunk is awaited, the first one included.
        Kills: no cancel arm in the stream closure or the stream turn body, a gate that stays
        held, (logged) a turn that does not end ``rolled_back`` with every request paired."""
        for k in (1, 2, 3):
            options = {"event_log": EventLog(tmp_path / f"k{k}")} if logged else {}
            api, llm = _stream_api(_Parking, **options)
            cid, _ = await api.astart_conversation(_conversation_id=_CID)
            _attach_memory(api, cid)
            before = _seen(api, cid)
            llm.park(k, _dirty(api, cid))
            await _cancel(llm, _collect(api.aconverse_stream("hi", cid)))
            assert _seen(api, cid) == before, f"k={k}"
            assert _gate_is_free(api, cid), f"k={k}"
            if logged:
                TestTheSweep._the_log_shows_a_rolled_back_turn(api, cid, before)
            llm.park(-1)
            assert await _collect(api.aconverse_stream("hi", cid)) == ["a", "b", "c"]


async def _next(stream) -> str:
    return await stream.__anext__()


class TestRunLoops:
    @pytest.mark.parametrize(("parked_in", "charged"), [("hook", 1), ("model", 2)])
    async def test_a_cancel_between_rounds_leaves_the_conversation_usable(
        self, parked_in, charged
    ):
        """Round 1 ran; the cancel lands in the ``before_step`` hook of round 2 (one turn
        charged) or in the reply of round 2 (two charged): either way the conversation is in the
        state of the last finished round, the gate is free and the budget counted what started.
        Kills: a run loop that holds the gate between rounds, a lost charge."""
        llm = _Parking(default_response="r")
        api = API.from_definition(_RUN(), llm_interface=llm)
        cid, _ = await api.astart_conversation(max_steps=2)

        async def hook(n):
            if parked_in == "hook" and n == 2:
                llm.entered.set()
                await asyncio.Event().wait()

        llm.park(2 if parked_in == "model" else -1)
        run = api.arun_until_terminal(cid, max_steps=9, before_step=hook)
        await _cancel(llm, run)
        assert api.get_current_state(cid) == "b" and _gate_is_free(api, cid)
        assert api.get_conversation_history(cid) == [{"system": "r"}] * 2
        llm.park(-1)
        if charged == 1:
            assert (await api.aadvance(cid)).state_after == "c"
        with pytest.raises(RunBudgetExceededError):
            await api.aadvance(cid)


# ----------------------------------------------------------------------------------------------
# Step 7.6: limits of the cancel and parity rules that the docs name (D-020). A row that fails
# here means the limit was fixed: remove its doc sentence and the row together.
# ----------------------------------------------------------------------------------------------


class TestKnownLimits:
    async def test_a_consumer_cancelled_in_its_own_await_abandons_the_stream(self):
        """A cancel in the CONSUMER's code (the loop body, here its own await) never reaches
        the stream: that is abandonment, not rollback. The partial reply stays and the
        conversation stays claimed while the stream object lives, until ``aclose`` frees it.
        Kills (documented rule): none; fails when the rule changes, as a fix would."""
        api, _ = _stream_api()
        cid, _ = await api.astart_conversation()
        stream = api.aconverse_stream("hi", cid)
        got, parked = [], asyncio.Event()

        async def consumer() -> None:
            async for chunk in stream:
                got.append(chunk)
                parked.set()
                await asyncio.Event().wait()  # the consumer's own await (a socket send)

        task = asyncio.create_task(consumer())
        await asyncio.wait_for(parked.wait(), _BOUND)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert got == ["a"]
        assert not _gate_is_free(api, cid)
        with pytest.raises(FSMError, match="already being processed"):
            await api.aconverse("again", cid)
        await stream.aclose()  # the abandonment ends here: the partial reply is kept
        assert api.get_conversation_history(cid)[-2:] == [
            {"user": "hi"},
            {"system": "a"},
        ]
        assert _gate_is_free(api, cid)
        assert await api.aconverse("again", cid) == "abc"

    async def test_a_sync_only_executor_may_write_into_the_conversation_after_a_cancel(
        self,
    ):
        """The default ``aextract`` runs a sync-only executor's ``extract`` on a worker thread
        with the live instance. A cancel rolls the turn back and returns at once, and the thread
        goes on: its late write lands after the rollback. Kills (documented rule): none."""
        entered, go_on, finished = (threading.Event() for _ in range(3))

        class SyncOnly(StateExecutor):
            def extract(self, pipeline, instance, user_message, conversation_id, state):
                entered.set()
                go_on.wait(_BOUND)
                instance.context.data["late_write"] = "from a cancelled turn"
                finished.set()
                return DataExtractionResponse(extracted_data={})

        api = API.from_definition(
            _CHAT(),
            llm_interface=ScriptedLLM(default_response="ok"),
            executors={"conversation": SyncOnly()},
        )
        cid, _ = await api.astart_conversation()
        before = _seen(api, cid)
        task = asyncio.create_task(api.aconverse("hi", cid))
        await asyncio.to_thread(entered.wait, _BOUND)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert _seen(api, cid) == before and _gate_is_free(api, cid)
        go_on.set()
        await asyncio.to_thread(finished.wait, _BOUND)
        assert api.get_data(cid)["late_write"] == "from a cancelled turn"

    async def test_an_executor_subclass_that_overrides_only_the_sync_extract_is_bypassed(
        self,
    ):
        """``converse`` honours an override of the sync ``extract``; ``aconverse`` finds the
        inherited native ``aextract`` first and never calls it (no warning). The same override
        with its ``aextract`` twin is called in both modes. Kills (documented rule): none."""

        class SyncExtract(ConversationExecutor):
            calls = 0

            def extract(self, pipeline, instance, user_message, conversation_id, state):
                self.calls += 1
                return DataExtractionResponse(extracted_data={"done": True})

        class BothExtracts(SyncExtract):
            async def aextract(self, *args):
                return self.extract(*args)

        def build(executor):
            llm = ScriptedLLM(default_response="ok")
            return API.from_definition(
                _CHAT(), llm_interface=llm, executors={"conversation": executor}
            )

        results = {}
        for name, cls in (("sync only", SyncExtract), ("with twin", BothExtracts)):
            sync_executor, async_executor = cls(), cls()
            sync_api, async_api = build(sync_executor), build(async_executor)
            sync_cid, _ = sync_api.start_conversation()
            sync_api.converse("hi", sync_cid)
            async_cid, _ = await async_api.astart_conversation()
            await async_api.aconverse("hi", async_cid)
            results[name] = (
                sync_executor.calls,
                async_executor.calls,
                sync_api.get_current_state(sync_cid),
                async_api.get_current_state(async_cid),
            )
        assert results == {
            "sync only": (1, 0, "bye", "chat"),
            "with twin": (1, 1, "bye", "bye"),
        }

    async def test_the_task_that_ran_a_streams_first_step_owns_the_conversation_between_chunks(
        self,
    ):
        """The claim is bound in the Context of the task that ran the stream's FIRST step. After
        a hand-off to another task to drain, a sibling is refused, and the first task still
        reads the conversation mid-turn (it could end it as well). Kills (documented rule):
        none."""
        api, _ = _stream_api()
        cid, _ = await api.astart_conversation()
        read_now = asyncio.Event()

        async def sibling() -> dict:  # created BEFORE the claim: it holds no ticket
            await read_now.wait()
            return api.get_data(cid)

        sibling_task = asyncio.create_task(sibling())
        stream = api.aconverse_stream("hi", cid)
        assert await stream.__anext__() == "a"  # the claim is bound in THIS task
        parked, go_on, got = asyncio.Event(), asyncio.Event(), []

        async def drain() -> None:
            async for chunk in stream:
                got.append(chunk)
                parked.set()
                await go_on.wait()

        drainer = asyncio.create_task(drain())
        await asyncio.wait_for(parked.wait(), _BOUND)
        read_now.set()
        with pytest.raises(ConversationBusyError):
            await sibling_task
        assert api.get_data(cid) is not None  # the first task passes the gate mid-turn
        go_on.set()
        await asyncio.wait_for(drainer, _BOUND)
        assert got == ["b", "c"] and _gate_is_free(api, cid)

    async def test_a_second_run_loop_on_a_busy_conversation_fails_with_the_gate_refusal(
        self,
    ):
        """The second ``arun_until_terminal`` fails at its first read, with
        ``ConversationBusyError``, not the "already being processed" text of a second turn
        (both are ``FSMError``). Kills (documented rule): none."""
        api, llm, cid = _run_api(_Gated)
        llm.arm()
        first = asyncio.create_task(api.arun_until_terminal(cid, max_steps=9))
        await asyncio.wait_for(llm.entered.wait(), _BOUND)
        with pytest.raises(ConversationBusyError) as refused:
            await api.arun_until_terminal(cid, max_steps=9)
        assert "already being processed" not in str(refused.value)
        llm.release.set()
        assert [r.state_after for r in await first] == ["b", "c"]
