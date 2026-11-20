"""The async bodies of the pipeline answer like the sync ones (plan f150ec28, step 2).

Each body is written once as ``async def`` and the sync name is a ``run_sync`` wrapper, so a
row runs the same scripted work twice: sync, and async on a double whose ``a*`` twins really
suspend (``_Suspending``, one loop turn per call). A row compares what a caller can see: the
reply, the data, the history, the state, the bytes of every request in order and the usage
meter. ``suspensions`` equal to the number of calls proves every model call went through its
``a`` twin in async mode (a body that called the sync name there would not suspend). The
mutants each row kills are named in its docstring.
"""

from __future__ import annotations

import asyncio
import gc
import json
import threading
import time
from collections.abc import Callable
from contextlib import ExitStack
from functools import partial
from types import SimpleNamespace
from typing import Any, NamedTuple
from unittest.mock import patch

import pytest

from fsm_llm import API, FileSessionStore
from fsm_llm.aio import run_sync
from fsm_llm.core.definitions import (
    DataExtractionResponse,
    FSMError,
    RunBudgetExceededError,
)
from fsm_llm.engine.kinds import CompletionExecutor, ConversationExecutor, StateExecutor
from fsm_llm.handlers import HandlerTiming, create_handler
from fsm_llm.runtime import EventLog
from fsm_llm.testing import ScriptedLLM, final_reply
from tests.conftest import bind_api_clock
from tests.test_fsm_llm import kind_corpus as kc

#: Context keys that differ between two runs of the same script (an id or a clock).
_VOLATILE = {
    "_conversation_id",
    "_conversation_start",
    "_timestamp",
    "_transition_timestamp",
}


class _Suspending(ScriptedLLM):
    """A ``ScriptedLLM`` whose ``a*`` twins suspend once, then answer like the sync method."""

    suspensions = 0

    async def _hop(self) -> None:
        self.suspensions += 1
        await asyncio.sleep(0)

    async def acomplete(self, request):
        await self._hop()
        return await super().acomplete(request)

    async def agenerate_response(self, request):
        await self._hop()
        return await super().agenerate_response(request)

    async def aextract_field(self, request):
        await self._hop()
        return await super().aextract_field(request)

    async def aextract_bulk_data(self, request):
        await self._hop()
        return await super().aextract_bulk_data(request)


def _classify(intent: str):
    body = {"reasoning": "r", "intent": intent, "confidence": 0.9, "entities": {}}
    return final_reply(json.dumps(body))


def _fsm(*states: dict, initial: str = "a") -> Callable[[], dict]:
    return lambda: kc._fsm("Probe", initial, *states)


def _boom(context):
    raise RuntimeError("boom")


class _Scenario(NamedTuple):
    """A definition, a script for the double, and the messages of its turns (``None``: a
    message-free step); ``advance`` steps run before the work starts."""

    fsm: Callable[[], dict]
    llm: Callable[[type], ScriptedLLM]
    turns: tuple[str | None, ...] = ()
    context: dict | None = None
    advance: int = 0
    handlers: tuple = ()


_PLAIN_SCRIPT = dict(
    extraction_data={"name": "Ada", "item": "lamp"},
    responses={"greet": "Hello", "choose": "What now?", "confirm": "Done."},
)
_TIE = _fsm(
    kc._state("a", response_instructions="Hi.", transitions=[kc._go("b"), kc._go("c")]),
    kc._state("b", response_instructions="B."),
    kc._state("c", response_instructions="C."),
)
_QUIET = _fsm(
    kc._state("a", response_instructions="", transitions=[kc._go("end")]),
    kc._state("end", response_instructions=""),
)
_SCENARIOS = {
    # field, bulk (the additive pass of the second turn), classification, the extraction
    # after a transition, Pass 2
    "plain": _Scenario(
        kc._plain_fsm,
        lambda cls: cls([_classify("buy")] * 2, **_PLAIN_SCRIPT),
        ("I am Ada", "I want a lamp"),
    ),
    # an AMBIGUOUS tie: the classifier picks the target
    "tie": _Scenario(
        _TIE,
        lambda cls: cls([_classify("b")], responses={"a": "Hi.", "b": "B."}),
        ("pick",),
    ),
    # silent states: no greeting call, a reply of "" and nothing added to the history
    "quiet": _Scenario(_QUIET, lambda cls: cls(), ("go",)),
    # a completion state: Pass 1 is one ``complete`` call; two message-free steps first
    "completion": _Scenario(
        kc._tool_fsm,
        lambda cls: cls([final_reply("5")], default_response="Fine."),
        ("please hurry",),
        {"_completion_messages": [{"role": "user", "content": "What is 2 + 3?"}]},
        advance=1,
    ),
    "steps": _Scenario(
        kc._tool_fsm,
        lambda cls: cls([final_reply("5")], default_response="Fine."),
        (None, None),
        {"_completion_messages": [{"role": "user", "content": "What is 2 + 3?"}]},
    ),
    # Pass 2 fails (no reply for "choose"): the whole turn rolls back
    "pass2_failure": _Scenario(
        kc._plain_fsm,
        lambda cls: cls(
            [_classify("buy")],
            extraction_data={"name": "Ada"},
            responses={"greet": "Hi"},
        ),
        ("I am Ada",),
    ),
    # a critical POST_PROCESSING handler raises after Pass 1 committed: the turn rolls back
    "handler_failure": _Scenario(
        kc._plain_fsm,
        lambda cls: cls([_classify("buy")], **_PLAIN_SCRIPT),
        ("I am Ada",),
        handlers=(
            create_handler("boom")
            .at(HandlerTiming.POST_PROCESSING)
            .critical()
            .do(_boom),
        ),
    ),
}


def _requests(llm: ScriptedLLM) -> list[tuple[str, str]]:
    """Every call in order with its request bytes (what ``test_kind_equality`` pins)."""
    return [
        (name, json.dumps(req.model_dump(mode="json"), sort_keys=True))
        for name, req in llm.call_history
    ]


def _bind(name: str, cls: type[ScriptedLLM]):
    """A started conversation of a scenario: its pipeline, instance, id and state."""
    scenario = _SCENARIOS[name]
    llm = scenario.llm(cls)
    api = API.from_definition(scenario.fsm(), llm_interface=llm)
    for handler in scenario.handlers:
        api.register_handler(handler)
    cid, _ = api.start_conversation(initial_context=scenario.context)
    for _ in range(scenario.advance):
        api.advance(cid)
    manager = api.fsm_manager
    instance = manager.instances[cid]
    return SimpleNamespace(
        llm=llm,
        pipeline=manager._pipeline,
        instance=instance,
        cid=cid,
        state=manager._pipeline.get_state(instance, cid),
        calls0=len(llm.call_history),
        turns=scenario.turns,
    )


def _seen(bound, outcome) -> dict[str, Any]:
    """What a caller can see of a conversation after some work."""
    instance = bound.instance
    return {
        "outcome": outcome,
        "state": instance.current_state,
        "data": {k: v for k, v in instance.context.data.items() if k not in _VOLATILE},
        "history": list(instance.context.conversation.exchanges),
        "requests": _requests(bound.llm),
        "usage": bound.llm.usage(),
    }


async def _parity(name: str, sync_work, async_work) -> int:
    """Run the work sync on a plain double and async on a suspending one and compare; prove
    every model call of the async run went through its ``a`` twin. Returns the suspensions."""
    want, got = _bind(name, ScriptedLLM), _bind(name, _Suspending)
    expected = _seen(want, sync_work(want))
    assert _seen(got, await async_work(got)) == expected
    assert got.llm.suspensions == len(got.llm.call_history) - got.calls0
    return got.llm.suspensions


def _outcome(call, show=lambda value: value):
    """``show(call())``, or the type of the error it raised (a message names the double)."""
    try:
        return show(call())
    except Exception as exc:
        return type(exc).__name__


async def _aoutcome(awaitable, show=lambda value: value):
    try:
        return show(await awaitable)
    except Exception as exc:
        return type(exc).__name__


def dump(response):
    return response.model_dump()


class TestExecutorTwins:
    @pytest.mark.parametrize(
        ("name", "executor"),
        [("plain", ConversationExecutor), ("completion", CompletionExecutor)],
    )
    async def test_aextract_answers_like_extract(self, name, executor):
        """Kills: a native ``aextract`` that differs from ``extract``, a model call that skips
        its ``a`` twin (suspensions), a call outside the request order."""
        message = _SCENARIOS[name].turns[0]

        def args(b):
            return b.pipeline, b.instance, message, b.cid, b.state

        suspended = await _parity(
            name,
            lambda b: _outcome(lambda: executor().extract(*args(b)), dump),
            lambda b: _aoutcome(executor().aextract(*args(b)), dump),
        )
        assert suspended > 0


class _SyncOnly(StateExecutor):
    """The shape of the custom executors of the suite: only the sync methods."""

    def __init__(self) -> None:
        self.threads: list[int] = []

    def extract(self, pipeline, instance, user_message, conversation_id, state):
        self.threads.append(threading.get_ident())
        return DataExtractionResponse(extracted_data={"k": 1})

    def after_transition(self, *args):
        self.threads.append(threading.get_ident())


class TestDefaultTwinsOfAnExecutor:
    """Kills: no default (AttributeError), a default that never leaves the loop thread
    (the worker row), a default that always hops (the sync-mode row)."""

    async def test_async_mode_runs_the_sync_methods_on_a_worker_thread(self):
        executor = _SyncOnly()
        got = await executor.aextract(None, None, "m", "c", None)
        await executor.aafter_transition(
            None, None, "m", "c", DataExtractionResponse(), "a", None
        )
        assert got.extracted_data == {"k": 1}
        assert len(executor.threads) == 2
        assert threading.get_ident() not in executor.threads

    def test_sync_mode_runs_them_inline_with_no_loop(self):
        executor = _SyncOnly()
        got = run_sync(executor.aextract(None, None, "m", "c", None))
        run_sync(
            executor.aafter_transition(
                None, None, "m", "c", DataExtractionResponse(), "a", None
            )
        )
        assert got.extracted_data == {"k": 1}
        assert executor.threads == [threading.get_ident()] * 2


class TestGreetingAndPass1:
    @pytest.mark.parametrize(("name", "calls"), [("plain", 1), ("quiet", 0)])
    async def test_agenerate_initial_response_answers_like_its_sync_name(
        self, name, calls
    ):
        """Kills: a greeting call that skips its ``a`` twin (suspensions), a silent state
        that makes a call (the request lists differ)."""
        suspended = await _parity(
            name,
            lambda b: b.pipeline.generate_initial_response(b.instance, b.cid),
            lambda b: b.pipeline.agenerate_initial_response(b.instance, b.cid),
        )
        assert suspended == calls

    @pytest.mark.parametrize("name", ["plain", "tie"])
    async def test_pass1_answers_like_its_sync_wrapper(self, name):
        """Pass 1 end to end: the executor extraction, the transition (a classifier tie
        included) and the extraction after the transition. Kills: any hop of those that
        takes the sync name in async mode (suspensions), the sync wrapper and the async
        body disagreeing."""
        message = _SCENARIOS[name].turns[0]

        def flat(result):
            return result[0].model_dump(), result[1], result[2]

        def args(b):
            b.instance.context.conversation.add_user_message(message)
            return b.instance, message, b.cid

        async def run_async(b):
            return flat(
                await b.pipeline._aexecute_extraction_and_transition_pass(*args(b))
            )

        suspended = await _parity(
            name,
            lambda b: flat(
                b.pipeline._execute_extraction_and_transition_pass(*args(b))
            ),
            run_async,
        )
        assert suspended > 0

    def test_a_patch_of_a_sync_name_still_intercepts_pass1_in_sync_mode(self):
        """The two hops that tests patch (``test_regression_transition_eval``). Kills: a hop
        that awaits the async name directly, so a patch of the sync name is bypassed."""
        bound = _bind("plain", ScriptedLLM)
        found = DataExtractionResponse(extracted_data={}, confidence=1.0)
        with (
            patch.object(
                ConversationExecutor, "extract", return_value=found
            ) as extract,
            patch.object(
                bound.pipeline,
                "_execute_transition_evaluation_and_execution",
                return_value=(False, None),
            ) as transition,
        ):
            result = bound.pipeline._execute_extraction_and_transition_pass(
                bound.instance, "I am Ada", bound.cid
            )
        assert result == (found, False, None)
        extract.assert_called_once()
        transition.assert_called_once()
        assert bound.instance.current_state == "greet"


def _play(bound):
    """The scenario's turns, then a greeting, through the SYNC names (a failure is its type)."""
    pipeline, instance, cid = bound.pipeline, bound.instance, bound.cid
    out = []
    for message in bound.turns:
        if message is None:
            out.append(_outcome(partial(pipeline.advance, instance, cid), repr))
            continue
        instance.context.conversation.add_user_message(message)
        out.append(_outcome(partial(pipeline.process, instance, message, cid)))
    out.append(_outcome(partial(pipeline.generate_initial_response, instance, cid)))
    return out


async def _aplay(bound):
    """The same work through ``aprocess``, ``aadvance`` and ``agenerate_initial_response``."""
    pipeline, instance, cid = bound.pipeline, bound.instance, bound.cid
    out = []
    for message in bound.turns:
        if message is None:
            out.append(await _aoutcome(pipeline.aadvance(instance, cid), repr))
            continue
        instance.context.conversation.add_user_message(message)
        out.append(await _aoutcome(pipeline.aprocess(instance, message, cid)))
    out.append(await _aoutcome(pipeline.agenerate_initial_response(instance, cid)))
    return out


class TestSyncModeKeepsTheSyncNames:
    @pytest.mark.parametrize("name", ["plain", "tie", "completion"])
    def test_sync_mode_never_reaches_an_async_twin(self, name):
        """Every model call of a sync turn takes the sync name, so a patch of it intercepts.
        Kills: a body that awaits an ``a`` twin directly instead of ``acall`` (poisoned here)."""
        want = _bind(name, ScriptedLLM)
        expected = _seen(want, _play(want))
        got = _bind(name, ScriptedLLM)
        with ExitStack() as stack:
            for twin in ("acomplete", "agenerate_response", "aextract_field"):
                stack.enter_context(
                    patch.object(got.llm, twin, side_effect=AssertionError)
                )
            stack.enter_context(
                patch.object(got.llm, "aextract_bulk_data", side_effect=AssertionError)
            )
            assert _seen(got, _play(got)) == expected


class TestTurns:
    @pytest.mark.parametrize("name", sorted(_SCENARIOS))
    async def test_aprocess_and_aadvance_answer_like_process_and_advance(self, name):
        """Per kind of turn (plain: field, bulk, classification and the extraction after a
        transition; a classifier tie; a completion state; silent states; message-free steps;
        a Pass 2 failure; a handler failure): the reply or the error type, the data, the
        history, the state, the request bytes in order and the usage meter. The sync and the
        async name run ONE body, so this proves the transport, not the logic. Kills: a model
        call that skips its ``a`` twin or takes it in sync mode, an async-only difference in
        what a failed turn leaves behind (the failure rows)."""
        await _parity(name, _play, _aplay)


# ----------------------------------------------------------------------------------------------
# Step 3.1: the API entries (aconverse, aadvance, astart_conversation) over the manager gate
# ----------------------------------------------------------------------------------------------

_CID = (
    "conv-1"  # a fixed id (the internal restore seam) so two logs compare line by line
)
_CHAT = _fsm(
    kc._state(
        "chat",
        response_instructions="Reply.",
        transitions=[
            kc._go("bye", conditions=kc._when("done", {"==": [{"var": "done"}, True]}))
        ],
    ),
    kc._state("bye", response_instructions="Bye."),
    initial="chat",
)


class _Gated(_Suspending):
    """After ``arm()`` the first ``agenerate_response`` parks on ``release`` (``entered`` is set)."""

    armed = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entered, self.release = asyncio.Event(), asyncio.Event()

    def arm(self) -> None:
        self.armed = True

    async def agenerate_response(self, request):
        if self.armed:
            self.armed = False
            self.entered.set()
            await self.release.wait()
        return await super().agenerate_response(request)


def _events(log: EventLog) -> list[tuple]:
    """The log of ``_CID`` as comparable rows: every field but the clock and the blob refs."""
    return [
        tuple(
            sorted((k, repr(v)) for k, v in e.items() if k != "ts" and "_ref" not in k)
        )
        for e in log.read(_CID)
    ]


def _build_api(name: str, cls: type[ScriptedLLM], **api_kwargs):
    scenario = _SCENARIOS[name]
    llm = scenario.llm(cls)
    api = API.from_definition(scenario.fsm(), llm_interface=llm, **api_kwargs)
    for handler in scenario.handlers:
        api.register_handler(handler)
    return api, llm, scenario


def _sync_work(api, scenario):
    cid, greeting = api.start_conversation(scenario.context, _conversation_id=_CID)
    out = [greeting]
    for message in scenario.turns:
        call = (
            partial(api.advance, cid)
            if message is None
            else partial(api.converse, message, cid)
        )
        out.append(_outcome(call, repr))
    return out


async def _async_work(api, scenario):
    cid, greeting = await api.astart_conversation(
        scenario.context, _conversation_id=_CID
    )
    out = [greeting]
    for message in scenario.turns:
        call = api.aadvance(cid) if message is None else api.aconverse(message, cid)
        out.append(await _aoutcome(call, repr))
    return out


def _api_seen(api, llm, out):
    return {
        "out": out,
        "data": api.get_data(_CID),
        "history": api.get_conversation_history(_CID),
        "state": api.get_current_state(_CID),
        "requests": _requests(llm),
        "usage": llm.usage(),
    }


class TestApiEntries:
    @pytest.mark.parametrize("name", sorted(_SCENARIOS))
    @pytest.mark.parametrize("logged", [False, True])
    async def test_the_async_entries_answer_like_the_sync_ones(
        self, name, logged, tmp_path
    ):
        """``astart_conversation``, ``aconverse`` and ``aadvance`` per kind of turn, with the
        event log off and on: replies or error types, data, history, state, request bytes, usage
        and, logged, every event but its clock and blob refs. Kills: a hop that takes a sync name
        in async mode (suspensions), an entry that skips ``event_scope`` (the log is empty)."""
        logs = [EventLog(tmp_path / "sync"), EventLog(tmp_path / "async")]
        opts = [{"event_log": log} if logged else {} for log in logs]
        api, llm, scenario = _build_api(name, ScriptedLLM, **opts[0])
        want = _api_seen(api, llm, _sync_work(api, scenario))
        api2, llm2, _ = _build_api(name, _Suspending, **opts[1])
        got = _api_seen(api2, llm2, await _async_work(api2, scenario))
        assert got == want
        assert llm2.suspensions == len(llm2.call_history)
        if logged:
            assert _events(logs[1]) == _events(logs[0])
            kinds = {dict(row)["type"] for row in _events(logs[1])}
            # absolute: the turn boundary was written (a lost event scope writes none)
            assert {"'turn/start'", "'turn/end'"} <= kinds

    async def test_a_failed_async_turn_leaves_the_pre_turn_conversation_and_the_gate_free(
        self,
    ):
        """Absolute (both names run one body, so parity cannot see a lost rollback). Kills: the
        user message left in the history, a claim that is not released."""
        llm = ScriptedLLM(default_response="ok")  # no extraction script: Pass 2 only
        api = API.from_definition(_CHAT(), llm_interface=llm)
        cid, _ = await api.astart_conversation()
        before = api.get_conversation_history(cid)
        with patch.object(llm, "generate_response", side_effect=RuntimeError("down")):
            with pytest.raises(FSMError):
                await api.aconverse("hi", cid)
        assert api.get_conversation_history(cid) == before
        assert await api.aconverse("again", cid) == "ok"  # the gate was released

    async def test_a_session_store_saves_after_a_successful_async_turn_only(
        self, tmp_path
    ):
        """Kills: ``aconverse`` or ``aadvance`` without ``_auto_save_session``, a save after a
        failed turn."""
        store = FileSessionStore(tmp_path)
        llm = ScriptedLLM(default_response="ok")
        api = API.from_definition(_CHAT(), llm_interface=llm, session_store=store)
        cid, _ = await api.astart_conversation()
        assert store.load(cid) is None
        await api.aconverse("hi", cid)
        first = store.load(cid)
        assert first is not None and len(first.conversation_history) == 3
        await api.aadvance(cid)
        second = store.load(cid)
        assert second is not None and len(second.conversation_history) == 4
        with patch.object(llm, "generate_response", side_effect=RuntimeError("down")):
            with pytest.raises(FSMError):
                await api.aconverse("hi", cid)
        assert store.load(cid) == second

    @pytest.mark.parametrize("error", [RuntimeError("down"), FSMError("down")])
    async def test_a_failing_greeting_leaves_nothing_registered(self, error):
        """The greeting fails with a plain error and with an ``FSMError`` (the two failure
        arms): the same error type as the sync name and no instance left behind. Kills: a
        failure arm of the async body that does not clean up."""
        llm = ScriptedLLM(default_response="ok")
        api = API.from_definition(_CHAT(), llm_interface=llm)
        with patch.object(llm, "generate_response", side_effect=error):
            want = _outcome(api.start_conversation)
            assert await _aoutcome(api.astart_conversation()) == want == "FSMError"
        assert api.fsm_manager.instances == {}


class TestInterleaveAndRefusal:
    async def test_a_second_conversation_runs_while_the_first_model_call_is_suspended(
        self,
    ):
        """One API, one loop: B's whole turn completes and a heartbeat task runs while A's model
        call is parked. Kills: a model call that blocks the loop thread instead of awaiting, a
        conversation lock held across the whole API."""
        llm = _Gated(default_response="ok")
        api = API.from_definition(_CHAT(), llm_interface=llm)
        (a, _), (b, _) = (
            await api.astart_conversation(),
            await api.astart_conversation(),
        )
        llm.arm()
        ticks = 0

        async def beat():
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0)

        first = asyncio.create_task(api.aconverse("one", a))
        await asyncio.wait_for(llm.entered.wait(), 10)  # a bound: a hang fails the row
        heart = asyncio.create_task(beat())
        assert await api.aconverse("two", b) == "ok"
        heart.cancel()
        assert not first.done() and ticks > 0
        llm.release.set()
        assert await first == "ok"

    async def test_a_second_turn_on_a_busy_conversation_is_refused_and_the_first_finishes(
        self,
    ):
        """Another task (async or sync name, a step too) is refused with the existing text and
        waits for nothing; the first turn ends intact and the gate is free afterwards. Kills: a
        claim that is skipped in the async body (no refusal), a turn that does not release."""
        llm = _Gated(default_response="ok")
        api = API.from_definition(_CHAT(), llm_interface=llm)
        cid, _ = await api.astart_conversation()
        llm.arm()
        first = asyncio.create_task(api.aconverse("one", cid))
        await asyncio.wait_for(llm.entered.wait(), 10)  # a bound: a hang fails the row
        for refused in (
            api.aconverse("two", cid),
            api.aadvance(cid),
        ):
            with pytest.raises(FSMError, match="already being processed"):
                await refused
        with pytest.raises(FSMError, match="already being processed"):
            api.converse("two", cid)  # the sync name on the loop thread
        llm.release.set()
        assert await first == "ok"
        assert [list(e) for e in api.get_conversation_history(cid)] == [
            ["system"],
            ["user"],
            ["system"],
        ]
        assert await api.aconverse("three", cid) == "ok"


# ----------------------------------------------------------------------------------------------
# Step 3.2: streams (aconverse_stream, aadvance_stream and the sync names over them)
# ----------------------------------------------------------------------------------------------


class _Streaming(_Gated):
    """``_Gated`` whose async stream suspends once per chunk, so a consumer can be parked."""

    async def agenerate_response_stream(self, request):
        async for chunk in super().agenerate_response_stream(request):
            await self._hop()
            yield chunk


def _stream_api(cls=_Streaming, **kwargs):
    llm = cls(stream_chunks=["a", "b", "c"], default_response="abc")
    return API.from_definition(_CHAT(), llm_interface=llm, **kwargs), llm


async def _collect(stream) -> list[str]:
    return [chunk async for chunk in stream]


def _loop_errors() -> list[dict]:
    """Install a recorder on the running loop's exception handler (what an asyncgen
    finalizer that could not reset a ContextVar token reports)."""
    seen: list[dict] = []
    asyncio.get_running_loop().set_exception_handler(
        lambda _loop, ctx: seen.append(ctx)
    )
    return seen


async def _settle(done) -> None:
    """Let the loop run until ``done()`` (a finalizer task), bounded by turns, no sleeps."""
    for _ in range(200):
        if done():
            return
        await asyncio.sleep(0)


class TestStreams:
    @pytest.mark.parametrize("logged", [False, True])
    async def test_the_async_streams_yield_what_the_sync_streams_yield(
        self, logged, tmp_path
    ):
        """Chunks, history, data, state, request bytes, usage and (logged) every event but its
        clock and blob refs, for a message stream and a message-free step stream, sync names
        against async names on a double whose chunks really suspend. Kills: a stream that skips
        its ``a`` twin (suspensions), a lost event scope on the stream (the log is empty)."""
        logs = [EventLog(tmp_path / "sync"), EventLog(tmp_path / "async")]
        want_api, want_llm = _stream_api(
            ScriptedLLM, **({"event_log": logs[0]} if logged else {})
        )
        got_api, got_llm = _stream_api(**({"event_log": logs[1]} if logged else {}))
        for api in (want_api, got_api):
            api.start_conversation(_conversation_id=_CID)
        want = [
            list(want_api.converse_stream("hi", _CID)),
            list(want_api.advance_stream(_CID)),
        ]
        got = [
            await _collect(got_api.aconverse_stream("hi", _CID)),
            await _collect(got_api.aadvance_stream(_CID)),
        ]
        assert got == want == [["a", "b", "c"], ["a", "b", "c"]]
        assert _api_seen(got_api, got_llm, got) == _api_seen(want_api, want_llm, want)
        assert (
            got_llm.suspensions == 10
        )  # per stream: two field extractions, three chunks
        if logged:
            assert _events(logs[1]) == _events(logs[0])
            kinds = {dict(row)["type"] for row in _events(logs[1])}
            # absolute: the turn boundary was written (a lost event scope writes none)
            assert {"'turn/start'", "'turn/end'"} <= kinds

    async def test_a_stream_that_is_never_iterated_holds_nothing(self):
        """The id is checked at call time and the claim waits for the first step. Kills: a claim
        taken at call time (the next turn is refused)."""
        api, _ = _stream_api()
        cid, _ = await api.astart_conversation()
        unused = api.aconverse_stream("hi", cid)
        assert await api.aconverse("again", cid) == "abc"
        with pytest.raises(ValueError):
            api.aconverse_stream("hi", "no-such-conversation")
        await unused.aclose()

    async def test_a_stream_first_iterated_in_another_task_finishes_and_releases(self):
        """Steps may hop tasks (each its own Context): the claim of the first step is released
        by ticket in the last one. Kills: a release that needs the Context of the claim."""
        api, _ = _stream_api()
        cid, _ = await api.astart_conversation()
        stream = api.aconverse_stream("hi", cid)
        first = await asyncio.create_task(stream.__anext__())
        assert [first, *await _collect(stream)] == ["a", "b", "c"]
        assert await api.aconverse("again", cid) == "abc"

    async def test_a_consumer_that_breaks_without_aclose_leaves_no_loop_error_and_no_claim(
        self,
    ):
        """The probe p7 shape: the finalizer runs ``aclose`` in a new task (another Context) and
        no ContextVar token may be reset there; the abandoned turn keeps the partial reply and
        releases the conversation. Kills: a ContextVar bound across a ``yield`` (the loop's
        exception handler sees a ``ValueError``), a release that stays unreached."""
        errors = _loop_errors()
        api, _ = _stream_api()
        cid, _ = await api.astart_conversation()
        stream = api.aconverse_stream("hi", cid)
        async for _chunk in stream:
            break
        del stream
        gc.collect()
        await _settle(lambda: not api.fsm_manager._active_turns)
        assert not api.fsm_manager._active_turns and errors == []
        assert api.get_conversation_history(cid)[-2:] == [
            {"user": "hi"},
            {"system": "a"},
        ]
        assert await api.aconverse("again", cid) == "abc"

    async def test_aclose_keeps_the_partial_reply_and_a_busy_stream_refuses_a_second_turn(
        self,
    ):
        """While the stream is parked mid-reply a second turn is refused; ``aclose`` then keeps
        the user message and the partial reply (today's abandonment rule) and frees the
        conversation. Kills: a GeneratorExit that drops the partial reply, a release outside the
        ``finally``."""
        api, _ = _stream_api()
        cid, _ = await api.astart_conversation()
        stream = api.aconverse_stream("hi", cid)
        assert await stream.__anext__() == "a"
        with pytest.raises(FSMError, match="already being processed"):
            await api.aconverse("again", cid)
        await stream.aclose()
        assert api.get_conversation_history(cid)[-2:] == [
            {"user": "hi"},
            {"system": "a"},
        ]
        assert await api.aconverse("again", cid) == "abc"


# ----------------------------------------------------------------------------------------------
# Step 3.3: the run loops (arun_until_terminal, arun_until_terminal_stream)
# ----------------------------------------------------------------------------------------------

_RUN = _fsm(
    kc._state("a", response_instructions="A.", transitions=[kc._go("b")]),
    kc._state("b", response_instructions="B.", transitions=[kc._go("c")]),
    kc._state("c", response_instructions="C."),
    initial="a",
)
_CHILD = _fsm(
    kc._state("x", response_instructions="X.", transitions=[kc._go("y")]),
    kc._state("y", response_instructions="Y."),
    initial="x",
)


def _run_api(cls=_Suspending, **kwargs):
    llm = cls(default_response="r", stream_chunks=["p", "q"])
    api = API.from_definition(_RUN(), llm_interface=llm, **kwargs)
    cid, _ = api.start_conversation(_conversation_id=_CID)
    return api, llm, cid


def _steps(results) -> list[tuple]:
    return [(r.state_before, r.state_after, r.response, r.ended) for r in results]


class _Ticking:
    """The ``time`` binding of the API modules: ``monotonic`` moves when the double says so."""

    now = 1000.0

    def monotonic(self) -> float:
        return self.now

    def __getattr__(self, name):
        return getattr(time, name)


class _SlowReplies(_Suspending):
    """Every reply costs ``seconds`` on a fake clock."""

    def __init__(self, clock, seconds, **kwargs):
        super().__init__(**kwargs)
        self.clock, self.seconds = clock, seconds

    def generate_response(self, request):
        self.clock.now += self.seconds
        return super().generate_response(request)


class TestRunLoops:
    async def test_the_async_run_answers_like_the_sync_run(self):
        """Steps, replies, the final state and the request bytes of a multi-step run, with a
        double whose calls suspend. Kills: a step that takes ``advance`` in async mode."""
        want_api, want_llm, _ = _run_api(ScriptedLLM)
        got_api, got_llm, _ = _run_api()
        want = want_api.run_until_terminal(_CID, max_steps=9)
        got = await got_api.arun_until_terminal(_CID, max_steps=9)
        assert (
            _steps(got)
            == _steps(want)
            == [
                ("a", "b", "r", False),
                ("b", "c", "r", True),
            ]
        )
        assert _api_seen(got_api, got_llm, got) == _api_seen(want_api, want_llm, want)
        assert (
            got_llm.suspensions == len(got_llm.call_history) - 1
        )  # all but the greeting

    async def test_the_async_stream_run_yields_the_sync_chunks(self):
        """Kills: a stream step that takes ``advance_stream`` in async mode, a lost round."""
        want_api, _, _ = _run_api(ScriptedLLM)
        got_api, _, _ = _run_api()
        want = list(want_api.run_until_terminal_stream(_CID, max_steps=9))
        got = await _collect(got_api.arun_until_terminal_stream(_CID, max_steps=9))
        assert got == want == ["p", "q", "p", "q"]
        assert got_api.get_current_state(_CID) == "c"

    async def test_a_spent_step_budget_raises_after_the_steps_it_allowed(self):
        """Absolute: one step ran and is kept, the next never started. Kills: a budget check
        that moves after the step, a check that is skipped."""
        api, _, _ = _run_api()
        with pytest.raises(RunBudgetExceededError) as spent:
            await api.arun_until_terminal(_CID, max_steps=1)
        assert (spent.value.budget, spent.value.limit, spent.value.steps_done) == (
            "steps",
            1,
            1,
        )
        assert api.get_current_state(_CID) == "b"
        with pytest.raises(RunBudgetExceededError) as sync_spent:
            _run_api()[0].run_until_terminal(_CID, max_steps=1)
        assert str(spent.value) == str(sync_spent.value)

    async def test_a_spent_seconds_budget_stops_before_the_next_step(self, monkeypatch):
        """The injected clock moves 10 s per reply against a 5 s budget: step 1 runs, step 2
        never starts. Kills: a seconds check that is skipped or run after the step."""
        clock = _Ticking()
        bind_api_clock(monkeypatch, clock)
        api, _, _ = _run_api(lambda **kw: _SlowReplies(clock, 10.0, **kw))
        clock.now = 1000.0  # the greeting's cost is before the run's own start
        with pytest.raises(RunBudgetExceededError) as spent:
            await api.arun_until_terminal(_CID, max_steps=9, max_seconds=5.0)
        assert (spent.value.budget, spent.value.steps_done) == ("seconds", 1)
        assert api.get_current_state(_CID) == "b"

    async def test_an_async_before_step_is_awaited_and_may_push_and_pop(self):
        """The hook returns a coroutine: the run awaits it (it pushes a child FSM before step 2
        and pops the ended child when it is offered), and the path equals the run with a SYNC
        hook. Kills: a hook result that is not awaited (no push: the path differs), a hook
        that runs after the step."""

        def hook_body(api, n):
            if n == 2 and api.get_stack_depth(_CID) == 1:
                api.push_fsm(_CID, _CHILD())
            elif api.get_stack_depth(_CID) > 1 and api.has_conversation_ended(_CID):
                api.pop_fsm(_CID)

        want_api, _, _ = _run_api(ScriptedLLM)
        want = want_api.run_until_terminal(
            _CID, max_steps=9, before_step=lambda n: hook_body(want_api, n)
        )
        got_api, _, _ = _run_api()
        offered: list[int] = []

        async def hook(n):
            await asyncio.sleep(0)  # a real suspension: the run must await it
            offered.append(n)
            hook_body(got_api, n)

        got = await got_api.arun_until_terminal(_CID, max_steps=9, before_step=hook)
        assert _steps(got) == _steps(want)
        assert [r.state_before for r in got] == [
            "a",
            "x",
            "b",
        ]  # the child ran in between
        assert offered and got_api.get_stack_depth(_CID) == 1
