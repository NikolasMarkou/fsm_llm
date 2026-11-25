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

        async def ticker() -> None:
            while not entered.is_set():
                await asyncio.sleep(0)
            for _ in range(TICKS):
                await asyncio.sleep(0)
            released.set()

        api = _api(tmp_path)
        with (
            mock.patch.object(os, "fsync", parked),
            mock.patch.object(jsonl, "_fsync", parked),
        ):
            task = asyncio.create_task(ticker())
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

        async def ticker() -> None:
            while not entered.is_set():
                await asyncio.sleep(0)
            for _ in range(TICKS):
                await asyncio.sleep(0)
            released.set()

        api = _api(tmp_path)
        with (
            mock.patch.object(Lanes, "lane_call", inline),
            mock.patch.object(os, "fsync", parked),
            mock.patch.object(jsonl, "_fsync", parked),
        ):
            task = asyncio.create_task(ticker())
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
