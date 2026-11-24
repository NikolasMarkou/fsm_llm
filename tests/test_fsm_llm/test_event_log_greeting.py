"""The root greeting is a logged model call (plan 3f4530d3 step 2; roadmap row 1.1).

Before this step ``API.start_conversation`` made the greeting before any scope could name its
conversation, so the first model call of every logged conversation was missing from the log
(D-021 of plan 21e6f00d). The API now mints the id (``start_event_id``), passes it to the manager
(the ``conversation_id=`` seam of ``API.resume``) and binds frame 0 around the manager call
(``start_event_scope``). Rows: the pair and its place before the start record; the durability order
(the request is written and fsynced BEFORE the model call, census D8); the bytes (the logged blob is
the request the provider got, and a log that is off changes nothing, census D5); a silent state, a
restore and a resume keep their paths; fork, resume and ``EventLogSessionStore`` read a greeted log;
a failed, cancelled or refused greeting releases the writer and leaves no conversation; a log killed
INSIDE the greeting is refused by ``API.resume`` with a message, never a crash. Counts and
sequences, never timing.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import json
from pathlib import Path
from typing import Any
from unittest import mock
from unittest.mock import patch

import pytest

from fsm_llm import API, FileSessionStore
from fsm_llm.core.definitions import FSMError
from fsm_llm.engine.fsm.manager import FSMManager
from fsm_llm.engine.session import EventLogSessionStore
from fsm_llm.llm import LiteLLMInterface
from fsm_llm.llm.observer import OBSERVED_CALL_KEYS
from fsm_llm.runtime import EventLog, canonical_json
from fsm_llm.runtime.stores import fold_session, fold_stack, jsonl
from fsm_llm.testing import ScriptedLLM
from tests.test_fsm_llm.event_log_support import _Dead, _reply, tree_bytes
from tests.test_fsm_llm.support import _state
from tests.test_fsm_llm.test_event_log_producers import _CHAT, _events, _Failing
from tests.test_fsm_llm.test_fsm_lock_order import _RecLock

COMPLETION = "fsm_llm.llm.providers.litellm.completion"
_GREETING_PAIR = ["llm/request", "llm/response", "context/record"]
_SILENT: dict[str, Any] = {
    "name": "Silent",
    "description": "The initial state says nothing, so the start makes no model call",
    "initial_state": "quiet",
    "states": {"quiet": _state("quiet")},
}


def _chat(log: EventLog | None, llm: Any = None, **kwargs: Any) -> API:
    llm = llm or ScriptedLLM(responses={"chat": "Welcome"})
    return API(copy.deepcopy(_CHAT), llm_interface=llm, event_log=log, **kwargs)


def _open_fds() -> int | None:
    path = Path("/proc/self/fd")
    return len(list(path.iterdir())) if path.is_dir() else None


class _Spied(ScriptedLLM):
    """Records the model call in the same list the patched write and fsync append to."""

    order: list[str]

    def generate_response(self, request: Any) -> Any:
        self.order.append("model")
        return super().generate_response(request)


def _ordered_start(tmp_path: Path) -> list[str]:
    """``write <type>``, ``fsync`` and ``model`` in the order they happened during a start."""
    order: list[str] = []
    real_write, real_fsync = jsonl._write, jsonl._fsync

    def write(fd: int, data: Any) -> int:
        for line in bytes(data).split(b"\n")[:-1]:
            order.append(f"write {json.loads(line)['type']}")
        return real_write(fd, data)

    def fsync(fd: int) -> None:
        order.append("fsync")
        real_fsync(fd)

    llm = _Spied(responses={"chat": "Welcome"})
    llm.order = order
    with (
        mock.patch.object(jsonl, "_write", write),
        mock.patch.object(jsonl, "_fsync", fsync),
        EventLog(tmp_path / "ev") as log,
    ):
        _chat(log, llm).start_conversation()
    return order


class TestTheGreetingIsLogged:
    def test_the_first_event_is_the_greeting_request_and_its_reply_comes_before_the_start_record(
        self, log: EventLog
    ):
        cid, greeting = _chat(log).start_conversation()
        assert greeting == "Welcome"
        events = list(log.read(cid))
        assert [e["type"] for e in events] == _GREETING_PAIR
        request, response, record = events
        assert (request["seq"], request["frame"], request["source"]) == (1, 0, "engine")
        assert request["call_type"] == "greeting"
        assert response["request_seq"] == request["seq"] and response["frame"] == 0
        state = log.get_json(record["state_ref"])
        assert state["conversation_history"] == [{"system": "Welcome"}]

    def test_the_request_is_written_and_fsynced_before_the_model_call(
        self, tmp_path: Path
    ):
        """Census D8: ``llm/request`` is on disk before the thing it announces."""
        order = _ordered_start(tmp_path)
        assert order[:5] == [
            "write llm/request",
            "fsync",
            "model",
            "write llm/response",
            "write context/record",
        ]

    def test_the_bytes_of_the_logged_request_are_the_bytes_the_provider_got(
        self, log: EventLog
    ):
        """Census D5, the rebuild row for the greeting: the blob IS the request sent."""
        api = API(
            copy.deepcopy(_CHAT),
            llm_interface=LiteLLMInterface(model="gpt-4o"),
            event_log=log,
        )
        with patch(COMPLETION, return_value=_reply("Welcome")) as completion:
            cid, _ = api.start_conversation()
        (sent,) = completion.call_args_list
        (request, *_rest) = _events(log, cid, "llm/request")
        expected = canonical_json(
            {k: sent.kwargs[k] for k in OBSERVED_CALL_KEYS if k in sent.kwargs}
        )
        assert log.get_blob(request["prompt_ref"]) == expected

    def test_a_silent_initial_state_writes_no_model_event(self, log: EventLog):
        llm = ScriptedLLM([])
        api = API(copy.deepcopy(_SILENT), llm_interface=llm, event_log=log)
        cid, greeting = api.start_conversation()
        assert greeting == "" and llm.call_history == []
        assert [e["type"] for e in log.read(cid)] == ["context/record"]

    def test_a_restore_session_writes_no_greeting(self, log: EventLog, tmp_path: Path):
        store = FileSessionStore(tmp_path / "sessions")
        api = _chat(log, session_store=store)
        cid, _ = api.start_conversation()
        api.save_session(cid)
        llm = ScriptedLLM(responses={"chat": "Welcome"})
        restored = _chat(log, llm, session_store=store)
        restored_id, _ = restored.restore_session(cid) or (None, None)
        assert restored_id is not None and llm.call_history == []
        kinds = [e["type"] for e in log.read(restored_id)]
        assert kinds == ["context/record"]
        assert next(iter(log.read(restored_id)))["source"] == "restore"

    def test_the_manager_lock_sequence_of_a_start_is_the_same_with_the_log_on(
        self, log: EventLog
    ):
        """Census D7: the scope and the minted id add no lock. The log-on start differs from the
        log-off one only by the state read of its own start record, after the manager returned."""

        def sequence(event_log: EventLog | None) -> list[tuple[str, str]]:
            events: list[tuple[str, str, int]] = []
            api = _chat(event_log)
            manager = api.fsm_manager
            manager._lock = _RecLock("lock", manager._lock, events)  # type: ignore[assignment]
            manager._fsm_cache_lock = _RecLock(  # type: ignore[assignment]
                "cache", manager._fsm_cache_lock, events
            )
            api.start_conversation()
            return [(name, what) for name, what, _ in events]

        off, on = sequence(None), sequence(log)
        assert off and on[: len(off)] == off
        assert on[len(off) :] == [
            ("cache", "acquire"),
            ("cache", "release"),
            ("lock", "acquire"),
            ("lock", "release"),
        ]


class TestAGreetedLogIsRead:
    def test_fold_session_and_fold_stack_read_the_start_after_the_greeting(
        self, log: EventLog
    ):
        cid, _ = _chat(log).start_conversation()
        projection = fold_session(log.read(cid), log.get_json)
        assert projection is not None and projection.current_state == "chat"
        assert projection.conversation_history == [{"system": "Welcome"}]
        folded = fold_stack(log.read(cid), log.get_json)
        assert folded is not None and folded.depth == 1

    def test_fork_at_turn_zero_and_the_log_store_rebuild_a_greeted_conversation(
        self, log: EventLog
    ):
        api = _chat(log)
        cid, _ = api.start_conversation()
        api.converse("Ann", cid)
        forked_id, _ = api.fork(cid, at_turn=0)
        assert forked_id != cid
        assert api.get_conversation_history(forked_id) == [{"system": "Welcome"}]
        restored = _chat(None, session_store=EventLogSessionStore(log)).restore_session(
            cid
        )
        assert restored is not None

    def test_resume_of_a_greeted_log_keeps_the_greeting_and_calls_no_model(
        self, log: EventLog
    ):
        api = _chat(log)
        cid, _ = api.start_conversation()
        log.release(cid)  # the writer of the first process is gone
        llm = ScriptedLLM([])
        resumed = _chat(log, llm)
        resumed_id, _ = resumed.resume(cid)
        assert resumed_id == cid and llm.call_history == []
        assert resumed.get_conversation_history(cid) == [{"system": "Welcome"}]


class _Down(ScriptedLLM):
    def generate_response(self, request: Any) -> Any:
        raise RuntimeError("PRIVATE provider message")


class _DiesInTheCall(ScriptedLLM):
    """The process vanishes inside the model call: a ``BaseException`` no library arm eats."""

    def generate_response(self, request: Any) -> Any:
        raise _Dead


class _Parked(ScriptedLLM):
    """The async greeting parks until it is cancelled."""

    def __init__(self) -> None:
        super().__init__(responses={"chat": "Welcome"})
        self.started = asyncio.Event()

    async def agenerate_response(self, request: Any) -> Any:
        self.started.set()
        await asyncio.Event().wait()


def _no_writers(log: EventLog) -> bool:
    return not any(log.has_writer(cid) for cid in log.conversation_ids())


class TestAFailedGreetingReleasesTheWriter:
    def test_a_provider_error_leaves_the_pair_no_conversation_and_no_held_writer(
        self, log: EventLog
    ):
        api = _chat(log, _Down(responses={"chat": "Welcome"}))
        before = _open_fds()
        with pytest.raises(FSMError):
            api.start_conversation()
        (cid,) = log.conversation_ids()
        assert [e["type"] for e in log.read(cid)] == ["llm/request", "provider/error"]
        assert next(iter(log.read(cid)))["call_type"] == "greeting"
        assert api.list_active_conversations() == [] and _no_writers(log)
        assert _open_fds() == before  # no descriptor stays with an id nobody holds
        assert "PRIVATE" not in tree_bytes(log.root).decode("utf-8", "replace")

    def test_the_same_api_starts_again_after_a_failed_greeting(self, log: EventLog):
        llm = _Failing(responses={"chat": "Welcome"})
        api = _chat(log, llm)
        llm.down = True
        with pytest.raises(FSMError):
            api.start_conversation()
        llm.down = False
        cid, greeting = api.start_conversation()
        assert (
            greeting == "Welcome" and _no_writers(log) is False
        )  # the live one is held
        assert [e["type"] for e in log.read(cid)] == _GREETING_PAIR
        assert len(log.conversation_ids()) == 2

    def test_a_cancelled_async_greeting_leaves_the_pair_and_no_held_writer(
        self, log: EventLog
    ):
        llm = _Parked()
        api = _chat(log, llm)

        async def scenario() -> None:
            task = asyncio.ensure_future(api.astart_conversation())
            await llm.started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(scenario())
        (cid,) = log.conversation_ids()
        kinds = [e["type"] for e in log.read(cid)]
        assert kinds == ["llm/request", "provider/error"]
        assert api.list_active_conversations() == [] and _no_writers(log)

    def test_a_death_inside_the_call_holds_no_lock_and_resume_refuses_by_name(
        self, log: EventLog
    ):
        """A request with no reply, no error and no ``turn/start``: the log a kill leaves."""
        api = _chat(log, _DiesInTheCall(responses={"chat": "Welcome"}))
        with pytest.raises(_Dead):
            api.start_conversation()
        (cid,) = log.conversation_ids()
        assert [e["type"] for e in log.read(cid)] == ["llm/request"]
        assert _no_writers(log)
        llm = ScriptedLLM([])
        with pytest.raises(FSMError, match="no recorded state"):
            _chat(log, llm).resume(cid)
        assert llm.call_history == []
        assert [e["type"] for e in log.read(cid)] == ["llm/request"]  # nothing added
        assert _no_writers(log)  # the refusal gave the writer lock back
        with pytest.raises(FSMError, match="no recorded state"):
            _chat(log).fork(cid, at_turn=0)
        assert EventLogSessionStore(log).load(cid) is None  # no state to restore


class TestTheLogOffPath:
    def test_the_id_is_minted_only_for_a_logged_plain_start(self, log: EventLog):
        from fsm_llm.engine.api.events import start_event_id

        logged, plain = _chat(log), _chat(None)
        assert start_event_id(plain, False, None) is None
        assert start_event_id(logged, True, None) is None  # restore_session
        assert start_event_id(logged, False, "given-1") is None  # resume and fork
        minted = start_event_id(logged, False, None)
        assert isinstance(minted, str) and len(minted) == 36

    def test_the_manager_gets_an_id_only_when_the_log_is_on(self, log: EventLog):
        def kwargs_of(event_log: EventLog | None) -> dict[str, Any]:
            api = _chat(event_log)
            with patch.object(
                FSMManager, "start_conversation", autospec=True, wraps=None
            ) as manager_start:
                manager_start.return_value = ("c-1", "")
                with contextlib.suppress(Exception):
                    api.start_conversation()
            return manager_start.call_args.kwargs

        assert "conversation_id" not in kwargs_of(None)
        assert len(kwargs_of(log)["conversation_id"]) == 36

    def test_a_suppressed_start_logs_nothing_and_mints_nothing(self, log: EventLog):
        cid, _ = _chat(log).start_conversation(_suppress_start=True)
        assert list(log.read(cid)) == [] or [e["type"] for e in log.read(cid)] == []
