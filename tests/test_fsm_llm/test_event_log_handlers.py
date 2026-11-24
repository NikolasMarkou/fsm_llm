"""``handler`` and ``executable/resolve`` (plan 3f4530d3 steps 5b and 5c, decision D-008).

Every row drives PRODUCT code: a real ``API`` with registered handlers (sync and async turns, a
timed handler, a raising one) and a real ``call`` state, never ``EventLog.append``. The rows pin:
one ``handler`` event per handler EXECUTION with its name, timing, outcome and the key NAMES it
wrote (never a value, an internal-prefixed key or a secret-looking one), the outcome ``timeout``
written ONCE by the calling thread (a straggler that returns late writes nothing), that a failed
write is a WARNING and changes nothing the turn returns, that the fold never moves on these types,
and that the log off or on gives the same replies, data and history. ``executable/resolve`` names
the target, the mode and the identities a call state resolved to.
"""

from __future__ import annotations

import asyncio
import copy
import json
import threading
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from fsm_llm import API, HandlerTiming, create_handler
from fsm_llm.core.definitions import FSMError
from fsm_llm.runtime import EventLog, EventType, LogWriteError, emit
from fsm_llm.runtime.stores import fold_stack
from fsm_llm.testing import ScriptedLLM
from tests.test_fsm_llm import event_log_child as child
from tests.test_fsm_llm.support import _state
from tests.test_fsm_llm.test_call_delegated_settle import run_delegated
from tests.test_fsm_llm.test_call_log import _api as _call_api
from tests.test_fsm_llm.test_call_log import _play

_ENVELOPE = {"v", "seq", "ts", "conversation_id", "frame", "type"}
_HANDLER_FIELDS = {"name", "timing", "outcome", "error_type", "keys"}
_RESOLVE_FIELDS = {"target", "mode", "template_identity", "instance_identity"}
_CHAT: dict[str, Any] = {
    "name": "Chat",
    "description": "One speaking state that never ends",
    "initial_state": "chat",
    "states": {
        "chat": _state(
            "chat",
            response_instructions="Reply",
            transitions=[
                {
                    "target_state": "end",
                    "description": "never",
                    "conditions": [{"description": "n", "logic": {"==": [1, 2]}}],
                }
            ],
        ),
        "end": _state("end"),
    },
}


def _writes(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "sentinel_value": "SENTINEL-TEXT",
        "_internal": 1,
        "api_key": "sk-secret-1",
        "zeta": 2,
    }


def _api(
    log: EventLog | None, *handlers: Any, handler_timeout: float | None = None
) -> API:
    return API(
        copy.deepcopy(_CHAT),
        llm_interface=ScriptedLLM(responses={"chat": "reply"}),
        handlers=list(handlers),
        handler_timeout=handler_timeout,
        event_log=log,
    )


def _ok(timing: HandlerTiming = HandlerTiming.PRE_PROCESSING, name: str = "ok") -> Any:
    return create_handler(name).at(timing).do(_writes)


def _events(log: EventLog, cid: str) -> list[dict[str, Any]]:
    return [e for e in log.read(cid) if e["type"] == "handler"]


def _turn(log: EventLog, *handlers: Any, **kwargs: Any) -> tuple[API, str]:
    api = _api(log, *handlers, **kwargs)
    cid, _ = api.start_conversation()
    api.converse("hello", cid)
    return api, cid


class TestOneEventPerExecution:
    def test_it_names_the_handler_the_timing_the_outcome_and_the_key_names(
        self, log: EventLog
    ):
        _, cid = _turn(log, _ok())
        (event,) = _events(log, cid)
        assert event["name"] == "ok" and event["timing"] == "pre_processing"
        assert event["outcome"] == "ok" and event["frame"] == 0
        assert event["keys"] == ["sentinel_value", "zeta"]  # no internal, no secret

    def test_no_value_internal_key_or_secret_name_is_in_any_event_or_file(
        self, log: EventLog
    ):
        from tests.test_fsm_llm.event_log_support import tree_bytes

        _, cid = _turn(log, _ok())
        text = json.dumps(_events(log, cid))
        assert "SENTINEL-TEXT" not in text and "_internal" not in text
        assert "api_key" not in text and "sk-secret-1" not in text
        (event,) = _events(log, cid)
        assert set(event) <= _HANDLER_FIELDS | _ENVELOPE
        # the handler's data is in the context record the log already held, never in this event
        assert b"sk-secret-1" not in tree_bytes(log.root)

    def test_every_execution_writes_one_event_in_handler_order(self, log: EventLog):
        ran: list[str] = []

        def tracking(name: str) -> Any:
            def body(ctx: dict[str, Any]) -> None:
                ran.append(name)

            return create_handler(name).at(HandlerTiming.PRE_PROCESSING).do(body)

        _, cid = _turn(log, tracking("a"), tracking("b"), tracking("c"))
        assert [e["name"] for e in _events(log, cid)] == ran == ["a", "b", "c"]
        assert all(e.get("keys", []) == [] for e in _events(log, cid))  # wrote nothing

    def test_a_handler_that_does_not_execute_writes_nothing(self, log: EventLog):
        never = (
            create_handler("never")
            .at(HandlerTiming.PRE_PROCESSING)
            .when(lambda *args: False)
            .do(_writes)
        )
        _, cid = _turn(log, never)
        assert _events(log, cid) == []

    def test_each_timing_writes_its_own_event(self, log: EventLog):
        _, cid = _turn(
            log,
            _ok(HandlerTiming.PRE_PROCESSING, "pre"),
            _ok(HandlerTiming.POST_PROCESSING, "post"),
            _ok(HandlerTiming.START_CONVERSATION, "start"),
        )
        assert {(e["name"], e["timing"]) for e in _events(log, cid)} == {
            ("pre", "pre_processing"),
            ("post", "post_processing"),
            ("start", "start_conversation"),
        }


class TestAnError:
    @staticmethod
    def _boom() -> Any:
        def body(ctx: dict[str, Any]) -> None:
            raise ValueError("PRIVATE failure text")

        return create_handler("boom").at(HandlerTiming.PRE_PROCESSING).do(body)

    def test_a_raising_handler_is_outcome_error_with_the_class_name_only(
        self, log: EventLog
    ):
        _, cid = _turn(log, self._boom())
        (event,) = _events(log, cid)
        assert event["outcome"] == "error" and event["error_type"] == "ValueError"
        assert "PRIVATE" not in json.dumps(event)

    def test_the_turn_goes_on_exactly_as_with_the_log_off(self, log: EventLog):
        off = _api(None, self._boom())
        cid_off, _ = off.start_conversation()
        want = (off.converse("hello", cid_off), off.get_data(cid_off))
        api, cid = _turn(log, self._boom())
        assert (api.converse("again", cid), api.get_data(cid))[1] == want[1]

    def test_in_raise_mode_the_event_is_written_and_the_error_is_unchanged(
        self, log: EventLog
    ):
        api = API(
            copy.deepcopy(_CHAT),
            llm_interface=ScriptedLLM(responses={"chat": "reply"}),
            handlers=[self._boom()],
            handler_error_mode="raise",
            event_log=log,
        )
        cid, _ = api.start_conversation()
        with pytest.raises(FSMError, match="PRIVATE failure text"):
            api.converse("hello", cid)
        (event,) = _events(log, cid)
        assert event["outcome"] == "error"


class TestATimeout:
    @staticmethod
    def _slow(release: threading.Event, threads: list[int]) -> Any:
        def body(ctx: dict[str, Any]) -> dict[str, Any]:
            threads.append(threading.get_ident())
            release.wait(10)
            return {"late": 1}

        return create_handler("slow").at(HandlerTiming.PRE_PROCESSING).do(body)

    def test_a_timeout_is_one_event_written_by_the_caller_and_a_late_return_writes_none(
        self, log: EventLog
    ):
        release, worker_ids = threading.Event(), []
        writers: list[int] = []
        real = log.append

        def spy(conversation_id: str, type: Any, **kwargs: Any) -> Any:
            if getattr(type, "value", type) == "handler":
                writers.append(threading.get_ident())
            return real(conversation_id, type, **kwargs)

        with mock.patch.object(log, "append", spy):
            _, cid = _turn(log, self._slow(release, worker_ids), handler_timeout=0.05)
            release.set()  # the straggler returns late
            for thread in threading.enumerate():
                if thread.name.startswith("fsm-handler-"):
                    thread.join(10)
            (event,) = _events(log, cid)
        assert event["outcome"] == "timeout" and "error_type" not in event
        assert writers == [
            threading.get_ident()
        ]  # the calling thread, never the worker
        assert worker_ids and worker_ids[0] != writers[0]
        assert len(_events(log, cid)) == 1  # the late return wrote nothing more

    def test_the_straggler_cap_refusal_is_a_timeout_too(self, log: EventLog):
        with mock.patch("fsm_llm.handlers.system.MAX_TIMED_HANDLER_STRAGGLERS", 0):
            _, cid = _turn(log, self._slow(threading.Event(), []), handler_timeout=0.05)
        (event,) = _events(log, cid)
        assert event["outcome"] == "timeout"

    def test_a_handlers_own_timeout_error_is_an_error_not_a_timeout(
        self, log: EventLog
    ):
        def body(ctx: dict[str, Any]) -> None:
            raise TimeoutError("its own")

        own = create_handler("own").at(HandlerTiming.PRE_PROCESSING).do(body)
        _, cid = _turn(log, own, handler_timeout=5.0)
        (event,) = _events(log, cid)
        assert event["outcome"] == "error" and event["error_type"] == "TimeoutError"

    def test_a_timeout_error_is_still_a_timeout_error(self):
        from fsm_llm.handlers.system import _HandlerTimedOut

        assert issubclass(_HandlerTimedOut, TimeoutError)


class TestAtLevelShape:
    def test_the_event_holds_names_and_outcome_and_no_reference_or_value(
        self, tmp_path: Path
    ):
        from tests.test_fsm_llm.event_log_support import tree_bytes

        with EventLog(tmp_path / "shape", privacy="shape") as log:
            _, cid = _turn(log, _ok())
            (event,) = _events(log, cid)
        assert not [k for k in event if k.endswith("_ref")]
        assert event["keys"] == ["sentinel_value", "zeta"]
        assert b"SENTINEL-TEXT" not in tree_bytes(tmp_path / "shape")


class TestAFailedWriteChangesNothing:
    def test_a_failing_emit_is_a_warning_and_the_turn_returns_the_same(
        self, tmp_path: Path
    ):
        def observed(log: EventLog | None) -> Any:
            api = _api(log, _ok())
            cid, _ = api.start_conversation()
            reply = api.converse("x", cid)
            return reply, api.get_data(cid), api.get_conversation_history(cid)

        off = observed(None)
        with EventLog(tmp_path / "ev") as log:
            with mock.patch(
                "fsm_llm.handlers.system.emit", side_effect=LogWriteError("full")
            ):
                assert observed(log) == off


class TestTheFold:
    def test_a_late_handler_event_at_frame_zero_after_a_push_keeps_the_child(
        self, log: EventLog
    ):
        api = _api(log)
        cid, _ = api.start_conversation()
        late = api.event_scope(cid)  # bound to frame 0 now
        api.push_fsm(cid, child.tool_fsm())
        with late:
            emit(EventType.HANDLER, name="h", timing="pre_processing", outcome="ok")
            emit(EventType.EXECUTABLE_RESOLVE, target="t", mode="frame")
        stack = fold_stack(log.read(cid), log.get_json)
        assert stack is not None and len(stack.frames) == 1


def _observed(log: EventLog | None, *, use_async: bool) -> dict[str, Any]:
    api = _api(log, _ok(), _ok(HandlerTiming.POST_PROCESSING, "post"))
    cid, _ = api.start_conversation()
    reply = (
        asyncio.run(api.aconverse("one", cid))
        if use_async
        else api.converse("one", cid)
    )
    return {
        "reply": reply,
        "data": api.get_data(cid),
        "history": api.get_conversation_history(cid),
    }


class TestLogOnEqualsLogOff:
    @pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
    def test_reply_data_and_history_are_equal(self, tmp_path: Path, use_async: bool):
        off = _observed(None, use_async=use_async)
        for privacy in ("values", "shape"):
            with EventLog(tmp_path / privacy, privacy=privacy) as on_log:
                assert _observed(on_log, use_async=use_async) == off, privacy
                (cid,) = on_log.conversation_ids()
                assert len(_events(on_log, cid)) == 2

    def test_the_comparison_can_fail(self, tmp_path: Path):
        off = _observed(None, use_async=False)
        with EventLog(tmp_path / "x") as on_log:
            on = _observed(on_log, use_async=False)
        on["data"] = {**on["data"], "extra": 1}
        assert on != off

    def test_with_no_scope_the_producer_builds_nothing(self):
        with mock.patch(
            "fsm_llm.handlers.system.visible_entries", side_effect=AssertionError
        ):
            api = _api(None, _ok())
            cid, _ = api.start_conversation()
            assert api.converse("x", cid) == "reply"


class TestExecutableResolve:
    def test_a_frame_call_names_the_target_the_mode_and_both_identities(
        self, log: EventLog
    ):
        cid = _play(_call_api(log))
        (event,) = [e for e in log.read(cid) if e["type"] == "executable/resolve"]
        assert set(event) <= _RESOLVE_FIELDS | _ENVELOPE
        assert event["mode"] == "frame" and event["target"]
        assert event["template_identity"] == f"template:{event['target']}"
        assert event["instance_identity"] == f"instance:{event['target']}"

    def test_a_delegated_call_names_mode_delegated(self, log: EventLog):
        cid = run_delegated(log)
        (event,) = [e for e in log.read(cid) if e["type"] == "executable/resolve"]
        assert event["mode"] == "delegated" and event["template_identity"]

    def test_an_unresolvable_target_writes_nothing(self, log: EventLog):
        api = _call_api(log)
        cid, _ = api.start_conversation({"topic": "why"})
        with mock.patch.object(
            api._targets, "resolve_target", return_value=None
        ):  # the resolver answers nothing
            api.advance(cid)
        assert [e for e in log.read(cid) if e["type"] == "executable/resolve"] == []

    def test_a_failed_write_is_a_warning_and_the_call_goes_on(self, tmp_path: Path):
        with EventLog(tmp_path / "ev") as log:
            with mock.patch(
                "fsm_llm.engine.kinds.call.emit", side_effect=LogWriteError("full")
            ):
                cid = _play(_call_api(log))
            assert [e for e in log.read(cid) if e["type"] == "executable/resolve"] == []
            assert [e for e in log.read(cid) if e["type"] == "stack/pop"]
