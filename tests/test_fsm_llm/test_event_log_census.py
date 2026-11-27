"""Row 1.18 at the engine: a logged async turn does no blocking log I/O on the event loop (plan
3f4530d3 step 8b, decisions D-013 to D-015).

"The 6.5 budget" is defined as a CALL CENSUS, never a stopwatch: recorders keyed on the loop
thread count ``fsync`` (the log, the blobs, their directories), ``os.replace`` of a blob,
``mkdir``, temp-file creation and ``os.open`` for one scripted logged conversation, and every one
is ZERO on the loop. The total across ALL threads equals the sync run (non-vacuity: the durability
is moved, not dropped). A parked fsync plus a loop task that ticks shows the loop stays free with
no clock: inline, the stub can never be released and the row fails by assertion.

A tool, approval or handler body that runs inline on the loop still fsyncs there: the tool row
pins that known limit as exactly the ``tool/call`` fsync.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import copy
import json
import os
import stat
import sys
import tempfile
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from fsm_llm import API, HandlerTiming, create_handler
from fsm_llm.runtime import EventLog, blobs
from fsm_llm.runtime.lanes import Lanes
from fsm_llm.runtime.stores import jsonl
from fsm_llm.testing import ScriptedLLM
from fsm_llm.tools import ToolCall, ToolRegistry
from tests.test_fsm_llm import event_log_child as child
from tests.test_fsm_llm import kind_corpus as kc

PARK_CAP_SECONDS = 10.0  # caps a hang only; no assertion reads a duration
TICKS = 5


async def _ticker(entered: threading.Event, released: threading.Event) -> None:
    """A LOOP task: once a write is parked (*entered*), tick ``TICKS`` times, then release it."""
    while not entered.is_set():
        await asyncio.sleep(0)
    for _ in range(TICKS):
        await asyncio.sleep(0)
    released.set()


def _chat(*, silent: bool) -> dict[str, Any]:
    """chat (extracts a name, speaks unless *silent*) -> bye."""
    first: dict[str, Any] = {
        "extraction_instructions": "Extract the name",
        "required_context_keys": ["name"],
        "transitions": [
            kc._go("bye", conditions=kc._when("name", {"!=": [{"var": "name"}, None]}))
        ],
    }
    if not silent:
        first["response_instructions"] = "Greet and ask the name."
    return kc._fsm(
        "Census",
        "chat",
        kc._state("chat", **first),
        kc._state("bye", response_instructions="Say goodbye."),
    )


def _llm() -> ScriptedLLM:
    return ScriptedLLM(
        extraction_data={"name": "Ada"}, responses={"chat": "Hi", "bye": "Bye"}
    )


class Census:
    """Counts of blocking calls, total and on the loop thread (``loop_ident``)."""

    #: what must be EQUAL between the sync and the async run: a directory fsync, a mkdir and
    #: the ``open`` of a directory depend on whether a blob's two-letter prefix directory is
    #: new, which the random ids inside the blobs decide, so they are counted apart.
    DURABLE = ("fsync", "replace", "tempfile")

    def __init__(self) -> None:
        self.loop_ident: int | None = None
        self.on_loop: collections.Counter[str] = collections.Counter()
        self.total: collections.Counter[str] = collections.Counter()

    def _wrap(self, name: str, real: Callable[..., Any]) -> Callable[..., Any]:
        def counted(*args: Any, **kwargs: Any) -> Any:
            label = name
            if name == "fsync" and stat.S_ISDIR(os.fstat(args[0]).st_mode):
                label = "dirsync"
            self.total[label] += 1
            if threading.get_ident() == self.loop_ident:
                self.on_loop[label] += 1
            return real(*args, **kwargs)

        return counted

    def durable(self) -> dict[str, int]:
        return {name: self.total[name] for name in self.DURABLE}

    @contextlib.contextmanager
    def watching(self) -> Iterator[Census]:
        with contextlib.ExitStack() as stack:
            for target, attr, name in (
                (os, "fsync", "fsync"),
                (jsonl, "_fsync", "fsync"),
                (blobs, "_replace", "replace"),
                (os, "mkdir", "mkdir"),
                (tempfile, "mkstemp", "tempfile"),
                (os, "open", "open"),
            ):
                stack.enter_context(
                    mock.patch.object(
                        target, attr, self._wrap(name, getattr(target, attr))
                    )
                )
            yield self


def _api(root: Path, *, silent: bool = False) -> API:
    return API.from_definition(
        _chat(silent=silent), llm_interface=_llm(), event_log=EventLog(root)
    )


async def _async_conversation(root: Path, *, silent: bool, census: Census) -> API:
    api = _api(root, silent=silent)
    census.loop_ident = threading.get_ident()
    with census.watching():
        cid, _ = await api.astart_conversation()
        await api.aconverse("my name is Ada", cid)
    return api


def _sync_conversation(root: Path, *, silent: bool, census: Census) -> API:
    api = _api(root, silent=silent)
    with census.watching():
        cid, _ = api.start_conversation()
        api.converse("my name is Ada", cid)
    return api


def _types(api: API) -> list[str]:
    log = api.event_log
    assert log is not None
    return [e["type"] for cid in log.conversation_ids() for e in log.read(cid)]


class TestTheLoopCensus:
    @pytest.mark.parametrize("silent", [False, True], ids=["speaking", "silent"])
    async def test_a_logged_async_conversation_blocks_the_loop_on_no_log_io(
        self, tmp_path: Path, silent: bool
    ) -> None:
        """Kills: a boundary write (turn start or end, a model request or reply, the first
        writer open) left on the loop; a blob written by an emit on the loop."""
        census = Census()
        api = await _async_conversation(tmp_path, silent=silent, census=census)
        assert _types(api).count("turn/end") >= 1
        assert dict(census.on_loop) == {}, dict(census.on_loop)
        api.close()

    @pytest.mark.parametrize("silent", [False, True], ids=["speaking", "silent"])
    async def test_the_async_totals_equal_the_sync_run(
        self, tmp_path: Path, silent: bool
    ) -> None:
        """Non-vacuity: the durability moved to a lane, none of it was dropped."""
        asynchronous, synchronous = Census(), Census()
        api = await _async_conversation(
            tmp_path / "a", silent=silent, census=asynchronous
        )
        api.close()
        other = await asyncio.to_thread(
            _sync_conversation, tmp_path / "s", silent=silent, census=synchronous
        )
        other.close()
        assert synchronous.total["fsync"] >= 10  # a vacuous census would count none
        assert asynchronous.durable() == synchronous.durable()
        if sys.platform != "win32":  # a directory cannot be fsynced on Windows
            assert asynchronous.total["dirsync"] >= asynchronous.total["replace"]
        assert _types(api) == _types(other)

    async def test_a_lane_that_runs_inline_turns_the_census_red(
        self, tmp_path: Path
    ) -> None:
        """The mutant, kept as a row: the census must be able to fail."""

        async def inline(self: Lanes, root_id: str, fn: Any, /, *args: Any) -> Any:
            return fn(*args)

        census = Census()
        with mock.patch.object(Lanes, "lane_call", inline):
            api = await _async_conversation(tmp_path, silent=False, census=census)
        assert sum(census.on_loop.values()) >= 20
        api.close()


class TestTheLoopStaysFree:
    async def test_a_parked_fsync_does_not_stop_the_loop(self, tmp_path: Path) -> None:
        """No clock. The first fsync stub parks on an Event that a LOOP task sets after
        ``TICKS`` ticks of ``sleep(0)``. On a lane the loop ticks and the stub is released;
        inline the loop cannot tick, the stub waits out its cap and the row fails by assertion."""
        entered, released = threading.Event(), threading.Event()
        starved: list[bool] = []
        real = os.fsync
        loop_ident = threading.get_ident()

        def parked(fd: int) -> None:
            if not entered.is_set():
                entered.set()
                if not released.wait(PARK_CAP_SECONDS):
                    starved.append(threading.get_ident() == loop_ident)
            real(fd)

        api = _api(tmp_path)
        with (
            mock.patch.object(os, "fsync", parked),
            mock.patch.object(jsonl, "_fsync", parked),
        ):
            task = asyncio.create_task(_ticker(entered, released))
            cid, _ = await api.astart_conversation()
            await api.aconverse("my name is Ada", cid)
            await task
        assert starved == [], "an fsync parked the loop thread itself"
        assert released.is_set()
        api.close()

    async def test_the_same_row_fails_when_the_lane_is_inline(
        self, tmp_path: Path
    ) -> None:
        """The mutant of the row above: run inline, the first fsync starves the loop."""

        async def inline(self: Lanes, root_id: str, fn: Any, /, *args: Any) -> Any:
            return fn(*args)

        entered, released = threading.Event(), threading.Event()
        starved: list[bool] = []
        real = os.fsync
        loop_ident = threading.get_ident()

        def parked(fd: int) -> None:
            if not entered.is_set():
                entered.set()
                if not released.wait(
                    0.2
                ):  # a short cap: this row EXPECTS the starvation
                    starved.append(threading.get_ident() == loop_ident)
            real(fd)

        api = _api(tmp_path)
        with (
            mock.patch.object(Lanes, "lane_call", inline),
            mock.patch.object(os, "fsync", parked),
            mock.patch.object(jsonl, "_fsync", parked),
        ):
            task = asyncio.create_task(_ticker(entered, released))
            await api.astart_conversation()
            await task
        assert starved == [True]
        api.close()


class TestTheKnownLimit:
    async def test_a_tool_turn_adds_exactly_the_tool_call_fsync_on_the_loop(
        self, tmp_path: Path
    ) -> None:
        """A tool body runs inline on the loop and its ``tool/call`` is a fsynced checkpoint:
        one fsync, no blob, no directory, nothing else (the stated known limit)."""
        registry = ToolRegistry()
        registry.register_function(lambda n: n, name="charge")
        run_it = (
            create_handler("run-registry-tool")
            .at(HandlerTiming.PRE_PROCESSING)
            .on_state("run_tools")
            .do(
                lambda ctx: (
                    registry.execute(ToolCall(tool_name="charge", parameters={"n": 5}))
                    and {}
                )
            )
        )
        api = API.from_definition(
            child.tool_fsm(),
            llm_interface=child.ChildLLM(child.scripted_replies()),
            event_log=EventLog(tmp_path),
            handlers=[run_it],
        )
        census = Census()
        census.loop_ident = threading.get_ident()
        with census.watching():
            cid, _ = await api.astart_conversation(
                initial_context={
                    child.MESSAGES: [{"role": "user", "content": child.QUESTION}]
                }
            )
            await api.aadvance(cid)  # intake -> call_model
            await api.aadvance(cid)  # the model asks for a tool: -> run_tools
            before = dict(census.on_loop)
            await api.aadvance(cid)  # the handler runs the tool
        assert before == {}, before
        assert dict(census.on_loop) == {"fsync": 1}, dict(census.on_loop)
        kinds = [e["type"] for e in api.event_log.read(cid)]
        assert kinds.count("tool/call") == kinds.count("tool/result") == 1
        api.close()


def _log_bytes(api: API, ts_free: bool = True) -> list[dict[str, Any]]:
    """Every event of every tree of *api*'s log, ``ts`` removed, the conversation id named
    ``ID`` (it is minted per run), in log order."""
    log = api.event_log
    assert log is not None
    out: list[dict[str, Any]] = []
    for root in log.conversation_ids():
        for event in log.read(root):
            line = json.dumps(
                {k: v for k, v in event.items() if k != "ts"}, sort_keys=True
            ).replace(root, "ID")
            out.append(json.loads(line))
    return out


class TestAsyncParity:
    @pytest.mark.parametrize("silent", [False, True], ids=["speaking", "silent"])
    async def test_the_log_of_an_async_run_equals_the_sync_run_up_to_ts(
        self, tmp_path: Path, silent: bool
    ) -> None:
        """Kills: an async-only write path that adds, drops or reorders an event, or writes a
        different blob (the refs are content hashes, so a blob that differs shows here)."""
        asynchronous = await _async_conversation(
            tmp_path / "a", silent=silent, census=Census()
        )
        synchronous = await asyncio.to_thread(
            _sync_conversation, tmp_path / "s", silent=silent, census=Census()
        )
        assert _log_bytes(asynchronous) == _log_bytes(synchronous)
        assert len(_log_bytes(asynchronous)) >= 8
        asynchronous.close()
        synchronous.close()


def _provider_reply(**kwargs: Any) -> Any:
    from tests.test_fsm_llm.event_log_support import _reply

    if kwargs.get("response_format"):
        return _reply(
            json.dumps({"extracted_data": {"name": "Ada"}, "confidence": 0.9})
        )
    return _reply("Hi")


async def _async_provider_reply(**kwargs: Any) -> Any:
    return _provider_reply(**kwargs)


@contextlib.contextmanager
def _patched_provider() -> Iterator[None]:
    provider = "fsm_llm.llm.providers.litellm"
    with (
        mock.patch(f"{provider}.completion", side_effect=_provider_reply),
        mock.patch(f"{provider}.acompletion", side_effect=_async_provider_reply),
        mock.patch(f"{provider}.get_supported_openai_params", return_value=[]),
    ):
        yield


class TestTheObservedPath:
    """A ``LiteLLMInterface`` logs through the request observer, whose async twins hop."""

    @staticmethod
    def _api(root: Path) -> API:
        from fsm_llm.llm import LiteLLMInterface

        return API.from_definition(
            _chat(silent=False),
            llm_interface=LiteLLMInterface(model="gpt-4o", api_key="sk-unused"),
            event_log=EventLog(root),
        )

    async def test_the_request_and_the_reply_of_an_observed_call_do_no_io_on_the_loop(
        self, tmp_path: Path
    ) -> None:
        """Kills: ``_asend`` calling the sync observer hooks (the request blob, its fsync and
        the reply blob on the loop); the totals must still equal the sync transport."""
        asynchronous, synchronous = Census(), Census()
        with _patched_provider():
            api = self._api(tmp_path / "a")
            asynchronous.loop_ident = threading.get_ident()
            with asynchronous.watching():
                cid, _ = await api.astart_conversation()
                await api.aconverse("my name is Ada", cid)
            other = self._api(tmp_path / "s")

            def drive() -> None:
                with synchronous.watching():
                    sid, _ = other.start_conversation()
                    other.converse("my name is Ada", sid)

            await asyncio.to_thread(drive)
        assert dict(asynchronous.on_loop) == {}, dict(asynchronous.on_loop)
        assert synchronous.total["fsync"] >= 10
        assert asynchronous.durable() == synchronous.durable()
        assert _types(api) == _types(other)
        api.close()
        other.close()


class TestWhatRunsWhere:
    async def test_the_state_the_request_and_the_delta_are_built_on_the_loop(
        self, tmp_path: Path
    ) -> None:
        """The state blob reads the live instance and the request body a live request: built on
        the loop BEFORE a job takes them, never on a lane thread (an orphan job could read an
        instance a rollback is restoring). Kills: a body built inside the job."""
        loop = threading.get_ident()
        where: collections.defaultdict[str, set[int]] = collections.defaultdict(set)
        names = {
            "state": "fsm_llm.engine.events.state_blob",
            "request": "fsm_llm.runtime.calls.request_body",
            "delta": "fsm_llm.engine.turn.events.filter_context",
        }

        def spy(label: str, real: Callable[..., Any]) -> Callable[..., Any]:
            def seen(*args: Any, **kwargs: Any) -> Any:
                where[label].add(threading.get_ident())
                return real(*args, **kwargs)

            return seen

        with contextlib.ExitStack() as stack:
            for label, dotted in names.items():
                module, attr = dotted.rsplit(".", 1)
                real = getattr(__import__(module, fromlist=[attr]), attr)
                stack.enter_context(mock.patch(dotted, spy(label, real)))
            api = _api(tmp_path)
            cid, _ = await api.astart_conversation()
            await api.aconverse("my name is Ada", cid)
        assert set(where) == set(names), dict(where)
        assert all(idents == {loop} for idents in where.values()), dict(where)
        api.close()

    async def test_the_redaction_hook_runs_on_a_lane_thread_in_async_mode(
        self, tmp_path: Path
    ) -> None:
        """The hook runs inside ``put_json``, which is inside the job: a hook that blocks
        holds a lane, never the loop, and it must be pure and thread safe (the docs say so).
        Kills: a blob put on the loop (the hook seen on the loop thread)."""
        loop = threading.get_ident()
        seen: set[int] = set()

        def hook(kind: object, value: Any) -> Any:
            seen.add(threading.get_ident())
            return value

        api = API.from_definition(
            _chat(silent=False),
            llm_interface=_llm(),
            event_log=EventLog(tmp_path / "a", redact=hook),
        )
        cid, _ = await api.astart_conversation()
        await api.aconverse("my name is Ada", cid)
        api.close()
        assert seen and loop not in seen, (loop, seen)

        seen.clear()
        other = API.from_definition(
            _chat(silent=False),
            llm_interface=_llm(),
            event_log=EventLog(tmp_path / "s", redact=hook),
        )

        def drive() -> int:
            sid, _ = other.start_conversation()
            other.converse("my name is Ada", sid)
            return threading.get_ident()

        caller = await asyncio.to_thread(drive)
        other.close()
        assert seen == {caller}, (caller, seen)

    async def test_with_the_log_off_no_lane_is_touched(self) -> None:
        async def refuse(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("a lane hop with the log off")

        with mock.patch.object(Lanes, "lane_call", refuse):
            api = API.from_definition(_chat(silent=False), llm_interface=_llm())
            cid, _ = await api.astart_conversation()
            assert await api.aconverse("my name is Ada", cid) == "Bye"
        api.close()

    async def test_the_request_is_on_disk_and_fsynced_before_the_model_is_called(
        self, tmp_path: Path
    ) -> None:
        """Row 1.18 keeps the checkpoint rule: ``llm/request`` is fsynced BEFORE the call (a
        job that fsynced after the call, or one that was only queued, fails here). The double
        reads the log when the model is called; the fsync spy says how many lines the last
        fsync of the log covered."""
        if not Path("/proc/self/fd").exists():
            pytest.skip("needs /proc to name the file an fsync was called on")
        log = EventLog(tmp_path)
        synced: list[int] = []
        real = jsonl._fsync

        def fsync(fd: int) -> None:
            real(fd)
            path = Path(os.readlink(f"/proc/self/fd/{fd}"))
            if path.suffix == ".jsonl":
                synced.append(len(path.read_text(encoding="utf-8").splitlines()))

        class Probe(ScriptedLLM):
            checked = False

            def generate_response(self, request: Any) -> Any:
                (root,) = log.conversation_ids()
                lines = (tmp_path / "logs" / f"{root}.jsonl").read_text(
                    encoding="utf-8"
                )
                lines = lines.splitlines()
                assert json.loads(lines[-1])["type"] == "llm/request"
                assert synced and synced[-1] >= len(lines), (synced, len(lines))
                Probe.checked = True
                return super().generate_response(request)

        api = API.from_definition(
            _chat(silent=False),
            llm_interface=Probe(responses={"chat": "Hi", "bye": "Bye"}),
            event_log=log,
        )
        with mock.patch.object(jsonl, "_fsync", fsync):
            await api.astart_conversation()
        assert Probe.checked
        api.close()


class TestAReopenUnderChurn:
    """Plan 77e5fef9 (bounded writers, D-004): which thread re-opens an evicted conversation.

    The first write of a turn is a lane job, so a re-open normally runs on a lane thread. The
    named limit: when more turns are in flight than ``max_open_writers``, an INLINE line append
    of an evicted id (the plain line writes that stay on the loop) re-opens on the loop and
    adds an ``os.open``, a lock and a tail read to the loop census. The bound must be at least
    the number of conversations in flight; below it the loop pays a re-open."""

    @staticmethod
    @contextlib.contextmanager
    def _watching_opens(loop_ident: int) -> Iterator[list[tuple[bool, bool]]]:
        """Every writer open as ``(is a re-open, on the loop thread)``."""
        seen: list[tuple[bool, bool]] = []
        real = jsonl.JsonlWriter._open

        def recording(self: jsonl.JsonlWriter) -> None:
            seen.append((not self._create, threading.get_ident() == loop_ident))
            real(self)

        with mock.patch.object(jsonl.JsonlWriter, "_open", recording):
            yield seen

    async def _two_conversations(
        self, root: Path, bound: int, *, together: bool
    ) -> tuple[list[tuple[bool, bool]], API]:
        log = EventLog(root, max_open_writers=bound)
        api = API.from_definition(
            _chat(silent=False), llm_interface=_llm(), event_log=log
        )
        with self._watching_opens(threading.get_ident()) as opens:
            first, _ = await api.astart_conversation()
            second, _ = await api.astart_conversation()
            if together:
                await asyncio.gather(
                    api.aconverse("my name is Ada", first),
                    api.aconverse("my name is Ada", second),
                )
            else:
                await api.aconverse("my name is Ada", first)
                await api.aconverse("my name is Ada", second)
        return opens, api

    async def test_a_bound_at_least_the_conversations_in_flight_never_reopens(
        self, tmp_path: Path
    ) -> None:
        census = Census()
        census.loop_ident = threading.get_ident()
        with census.watching():
            opens, api = await self._two_conversations(tmp_path, 2, together=True)
        assert opens == [
            (False, False),
            (False, False),
        ]  # the two first opens, on lanes
        assert dict(census.on_loop) == {}, dict(census.on_loop)  # the loop census holds
        api.close()

    async def test_a_reopen_of_a_turn_that_runs_alone_is_a_lane_job_not_the_loop(
        self, tmp_path: Path
    ) -> None:
        """Bound 1, two conversations taking their turns one after the other: each turn
        re-opens its evicted conversation, on a lane thread (never on the loop)."""
        opens, api = await self._two_conversations(tmp_path, 1, together=False)
        reopens = [on_loop for again, on_loop in opens if again]
        assert len(reopens) >= 2 and not any(reopens), opens
        assert _types(api).count("turn/end") == 2
        api.close()

    async def test_a_bound_below_the_turns_in_flight_records_where_the_loop_pays(
        self, tmp_path: Path
    ) -> None:
        """The known limit as a record: how many re-opens meet the loop thread depends on the
        interleaving (counted, never timed), so the row pins that re-opens happen, that every
        one is recorded with its thread, and that no turn fails or loses an event."""
        opens, api = await self._two_conversations(tmp_path, 1, together=True)
        assert sum(again for again, _ in opens) >= 1, opens
        assert _types(api).count("turn/end") == 2
        log = api.event_log
        assert log is not None
        for cid in log.conversation_ids():
            seqs = [e["seq"] for e in log.read(cid)]
            assert seqs == list(range(1, len(seqs) + 1))
        api.close()


def _streaming_llm() -> ScriptedLLM:
    return ScriptedLLM(
        extraction_data={"name": "Ada"},
        responses={"chat": "Hi", "bye": "Bye"},
        stream_chunks=["B", "ye"],
    )


def _stream_api(root: Path, *, interface: Any = None) -> API:
    return API.from_definition(
        _chat(silent=False),
        llm_interface=interface or _streaming_llm(),
        event_log=EventLog(root),
    )


async def _async_stream(api: API, census: Census) -> list[str]:
    """Start, then ONE streamed turn (Pass 2 streams the reply), the census over both."""
    census.loop_ident = threading.get_ident()
    with census.watching():
        cid, _ = await api.astart_conversation()
        return [c async for c in api.aconverse_stream("my name is Ada", cid)]


def _sync_stream(api: API, census: Census) -> list[str]:
    with census.watching():
        cid, _ = api.start_conversation()
        return list(api.converse_stream("my name is Ada", cid))


class TestAStreamedTurn:
    """Plan 77e5fef9 step 5 (S2a, D-007): the joined reply of a streamed turn is a lane job."""

    async def test_a_streamed_async_turn_blocks_the_loop_on_no_log_io(
        self, tmp_path: Path
    ) -> None:
        """Kills: the reply blob and the ``llm/response`` line written on the loop in the
        ``finally`` of the stream."""
        census = Census()
        api = _stream_api(tmp_path)
        assert await _async_stream(api, census) == ["B", "ye"]
        assert _types(api).count("llm/response") >= 2
        assert dict(census.on_loop) == {}, dict(census.on_loop)
        api.close()

    async def test_the_async_totals_and_the_event_order_equal_the_sync_stream(
        self, tmp_path: Path
    ) -> None:
        """G3. Non-vacuity: the durability moved to a lane, none of it was dropped, and after
        the drain (``close``) the types come in the sync order (response before ``turn/end``)."""
        asynchronous, synchronous = Census(), Census()
        api = _stream_api(tmp_path / "a")
        assert await _async_stream(api, asynchronous) == ["B", "ye"]
        other = _stream_api(tmp_path / "s")
        assert await asyncio.to_thread(_sync_stream, other, synchronous) == ["B", "ye"]
        api.close()
        other.close()
        assert synchronous.total["fsync"] >= 10
        assert asynchronous.durable() == synchronous.durable()
        types = _types(api)
        assert types == _types(other)
        assert types[-2:] == ["llm/response", "turn/end"]
        assert _log_bytes(api) == _log_bytes(other)

    async def test_a_lane_after_that_runs_inline_turns_the_census_red(
        self, tmp_path: Path
    ) -> None:
        """The mutant, kept as a row (M23): the reply written inline puts its blob on the loop."""

        def inline(self: Lanes, root_id: str, fn: Any) -> None:
            fn()

        census = Census()
        api = _stream_api(tmp_path)
        with mock.patch.object(Lanes, "after", inline):
            await _async_stream(api, census)
        assert sum(census.on_loop.values()) >= 5, dict(census.on_loop)
        api.close()

    async def test_a_parked_fsync_of_the_reply_does_not_stop_the_loop(
        self, tmp_path: Path
    ) -> None:
        """No clock. The first fsync AFTER the first chunk (the reply blob) parks on an Event
        a LOOP task sets after ``TICKS`` ticks. On a lane the loop ticks; inline it starves."""
        entered, released, chunked = (threading.Event() for _ in range(3))
        starved: list[bool] = []
        real = os.fsync
        loop_ident = threading.get_ident()

        def parked(fd: int) -> None:
            if chunked.is_set() and not entered.is_set():
                entered.set()
                if not released.wait(PARK_CAP_SECONDS):
                    starved.append(threading.get_ident() == loop_ident)
            real(fd)

        api = _stream_api(tmp_path)
        with (
            mock.patch.object(os, "fsync", parked),
            mock.patch.object(jsonl, "_fsync", parked),
        ):
            task = asyncio.create_task(_ticker(entered, released))
            cid, _ = await api.astart_conversation()
            async for _chunk_text in api.aconverse_stream("my name is Ada", cid):
                chunked.set()
            await task
        assert starved == [], "the reply blob parked the loop thread itself"
        assert entered.is_set() and released.is_set()
        api.close()

    async def test_an_observed_interface_stream_does_no_io_on_the_loop(
        self, tmp_path: Path
    ) -> None:
        """The provider path (``source`` provider): the observer logs the request, the reply of
        the stream is the same deferred job. Totals equal the sync transport."""
        from fsm_llm.llm import LiteLLMInterface
        from tests.test_fsm_llm.event_log_support import _chunk

        def reply(**kwargs: Any) -> Any:
            if kwargs.get("stream"):
                return iter([_chunk("B"), _chunk("ye")])
            return _provider_reply(**kwargs)

        async def areply(**kwargs: Any) -> Any:
            return reply(**kwargs)

        provider = "fsm_llm.llm.providers.litellm"
        asynchronous, synchronous = Census(), Census()
        with (
            mock.patch(f"{provider}.completion", side_effect=reply),
            mock.patch(f"{provider}.acompletion", side_effect=areply),
            mock.patch(f"{provider}.get_supported_openai_params", return_value=[]),
        ):
            make = lambda root: _stream_api(  # noqa: E731
                root, interface=LiteLLMInterface(model="gpt-4o", api_key="sk-unused")
            )
            api, other = make(tmp_path / "a"), make(tmp_path / "s")
            assert await _async_stream(api, asynchronous) == ["B", "ye"]
            assert await asyncio.to_thread(_sync_stream, other, synchronous) == [
                "B",
                "ye",
            ]
        api.close()
        other.close()
        assert dict(asynchronous.on_loop) == {}, dict(asynchronous.on_loop)
        assert asynchronous.durable() == synchronous.durable()
        assert _types(api) == _types(other)  # the provider replies carry random ids


_SILENT_KID: dict[str, Any] = kc._fsm(
    "SilentKid",
    "quiet",
    kc._state("quiet", response_instructions="", transitions=[kc._go("end")]),
    kc._state("end", response_instructions=""),
)  # its greeting makes no model call: the only blobs of its push are the push's own
_TALKING_KID: dict[str, Any] = kc._fsm(
    "TalkingKid",
    "quiet",
    kc._state("quiet", response_instructions="Say hello.", transitions=[kc._go("end")]),
    kc._state("end", response_instructions="Say bye."),
)  # its greeting is a model call: two lane hops before the frame is appended


def _push_api(root: Path) -> API:
    """The census root with a model that also answers the greeting of the pushed child."""
    llm = ScriptedLLM(
        extraction_data={"name": "Ada"},
        responses={"chat": "Hi", "bye": "Bye", "quiet": "Q", "end": "E"},
    )
    return API.from_definition(
        _chat(silent=False), llm_interface=llm, event_log=EventLog(root)
    )


async def _async_push(api: API, census: Census) -> str:
    """Start (a lane thread serves the tree), then ONE awaited push of a silent child, the
    census over the push alone, then the drain (``release``: the deferred job has run)."""
    cid, _ = await api.astart_conversation()
    census.loop_ident = threading.get_ident()
    with census.watching():
        await api.apush_fsm(cid, copy.deepcopy(_SILENT_KID))
        assert api.event_log is not None
        api.event_log.release(cid)
    return cid


def _sync_push(api: API, census: Census) -> str:
    cid, _ = api.start_conversation()
    with census.watching():
        api.push_fsm(cid, copy.deepcopy(_SILENT_KID))
    return cid


class TestAnAsyncPush:
    """Plan 77e5fef9 step 6 (S2b, D-008): the three blobs and the line of ``stack/push`` are a
    lane job, the loop does no log I/O for a push."""

    async def test_an_awaited_push_blocks_the_loop_on_no_log_io(
        self, tmp_path: Path
    ) -> None:
        """Kills (M29): the push written on the loop."""
        census = Census()
        api = _push_api(tmp_path)
        await _async_push(api, census)
        assert "stack/push" in _types(api)
        assert dict(census.on_loop) == {}, dict(census.on_loop)
        api.close()

    async def test_the_async_totals_and_the_event_order_equal_the_sync_push(
        self, tmp_path: Path
    ) -> None:
        """G3. The durability moved to a lane and none was dropped (three blobs, one checkpoint
        line), and after the drain the log holds the sync run's types in the sync order."""
        asynchronous, synchronous = Census(), Census()
        api = _push_api(tmp_path / "a")
        await _async_push(api, asynchronous)
        other = _push_api(tmp_path / "s")
        await asyncio.to_thread(_sync_push, other, synchronous)
        api.close()
        other.close()
        assert synchronous.total["replace"] == 3 and synchronous.total["fsync"] >= 3
        assert asynchronous.durable() == synchronous.durable()
        assert _types(api) == _types(other)
        assert _types(api)[-1] == "stack/push"

    async def test_a_lane_after_that_runs_inline_turns_the_census_red(
        self, tmp_path: Path
    ) -> None:
        """The mutant, kept as a row (M29): the push written inline puts its blobs on the loop."""

        def inline(self: Lanes, root_id: str, fn: Any) -> None:
            fn()

        census = Census()
        api = _push_api(tmp_path)
        with mock.patch.object(Lanes, "after", inline):
            await _async_push(api, census)
        assert census.on_loop["replace"] == 3 and census.on_loop["fsync"] >= 3
        api.close()

    async def test_a_parked_fsync_of_the_push_does_not_stop_the_loop(
        self, tmp_path: Path
    ) -> None:
        """No clock. The first fsync of the push parks on an Event a LOOP task sets after
        ``TICKS`` ticks: on a lane the loop ticks, inline it starves (the row fails)."""
        entered, released, armed = (threading.Event() for _ in range(3))
        starved: list[bool] = []
        real = os.fsync
        loop_ident = threading.get_ident()

        def parked(fd: int) -> None:
            if armed.is_set() and not entered.is_set():
                entered.set()
                if not released.wait(PARK_CAP_SECONDS):
                    starved.append(threading.get_ident() == loop_ident)
            real(fd)

        api = _push_api(tmp_path)
        cid, _ = await api.astart_conversation()
        with (
            mock.patch.object(os, "fsync", parked),
            mock.patch.object(jsonl, "_fsync", parked),
        ):
            task = asyncio.create_task(_ticker(entered, released))
            armed.set()
            await api.apush_fsm(cid, copy.deepcopy(_SILENT_KID))
            await task
        assert starved == [], "the push blob parked the loop thread itself"
        assert entered.is_set() and released.is_set()
        api.close()
