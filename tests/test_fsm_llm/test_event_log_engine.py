"""The engine side of the event log, part 1 (plan 21e6f00d step 7): the opt-in
``API(event_log=...)``, the turn boundary (``turn/start`` and ``turn/end`` written by the manager,
four outcomes), the log-off guarantee and the lock order with the log on.

Every model is a ``ScriptedLLM`` (an engine-path double: it never reaches ``LiteLLMInterface._send``).
Bytes, counts and sequences, never timing. Unverified until the Windows run: nothing here opens a
file itself (the files are ``stores.jsonl``'s and ``blobs``').
"""

from __future__ import annotations

import copy
import inspect
import math
import threading
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from fsm_llm import API, APIBuilder
from fsm_llm.core.definitions import FSMDefinition, FSMError
from fsm_llm.core.redaction import REDACTED
from fsm_llm.handlers import HandlerTiming, create_handler
from fsm_llm.runtime import (
    EventLog,
    EventType,
    LogLockedError,
    LogWriteError,
    TurnOutcome,
    bind,
    current_scope,
)
from fsm_llm.runtime.stores import fold_session, jsonl
from fsm_llm.runtime.usage import usage_from_events
from fsm_llm.testing import ScriptedLLM
from tests.test_fsm_llm.event_log_support import _chunk, _reply, _Wrapper, tree_bytes
from tests.test_fsm_llm.test_fsm_lock_order import (
    _PINNED,
    _short,
    order_problems,
)

_STATE = {
    "chat": {
        "id": "chat",
        "description": "Chat",
        "purpose": "Answer",
        "response_instructions": "Reply briefly",
        "transitions": [
            {
                "target_state": "done",
                "description": "Never taken here",
                "conditions": [{"description": "never", "logic": {"==": [1, 2]}}],
            }
        ],
    },
    "done": {
        "id": "done",
        "description": "Terminal",
        "purpose": "End",
        "response_instructions": "Say goodbye",
    },
}
_FSM = {
    "name": "EngineLog",
    "description": "Two states for the engine log tests",
    "initial_state": "chat",
    "states": _STATE,
}
# The same FSM, but the first turn always moves chat -> done (so the next turn is refused).
_FSM_GO = copy.deepcopy(_FSM)
_FSM_GO["states"]["chat"]["transitions"][0]["conditions"][0]["logic"] = {"==": [1, 1]}
# A message turn extracts `name` (committed), then a critical PRE_TRANSITION handler fails.
_FSM_PARTIAL = copy.deepcopy(_FSM_GO)
_FSM_PARTIAL["states"]["chat"]["extraction_instructions"] = "Extract the name"
_FSM_PARTIAL["states"]["chat"]["required_context_keys"] = ["name"]

START, END = EventType.TURN_START.value, EventType.TURN_END.value
RECORD = EventType.CONTEXT_RECORD.value


def _llm() -> ScriptedLLM:
    return ScriptedLLM(
        responses={"chat": "ok", "done": "bye"}, stream_chunks=["o", "k"]
    )


def _api(log: EventLog, llm: Any = None, definition: Any = _FSM, **kwargs: Any) -> API:
    return API(definition, llm_interface=llm or _llm(), event_log=log, **kwargs)


def _types(log: EventLog, cid: str) -> list[str]:
    return [e["type"] for e in log.read(cid)]


def _boundary(log: EventLog, cid: str) -> list[str]:
    """The turn boundary of a conversation: the state-carrying types only (the producers of
    plan step 8 write executor, transition and model-call events between them)."""
    return [t for t in _types(log, cid) if t in (RECORD, START, END)]


def _of(log: EventLog, cid: str, *kinds: str) -> list[dict[str, Any]]:
    return [e for e in log.read(cid) if e["type"] in kinds]


def _state(log: EventLog, event: dict[str, Any]) -> dict[str, Any]:
    state: dict[str, Any] = log.get_json(event["state_ref"])
    return state


class _Interrupting(ScriptedLLM):
    def generate_response(self, request: Any) -> Any:
        raise KeyboardInterrupt


class TestTurnBoundary:
    def test_a_committed_turn_logs_start_then_end(self, log: EventLog):
        api = _api(log)
        cid, greeting = api.start_conversation()
        assert api.converse("hello", cid) == "ok"
        assert _boundary(log, cid) == [RECORD, START, END]
        start, end = _of(log, cid, START, END)
        assert start["has_user_message"] is True and start["frame"] == 0
        assert log.get_json(start["user_message_ref"]) == "hello"
        assert end["outcome"] == TurnOutcome.COMMITTED.value
        before, after = _state(log, start), _state(log, end)
        assert before["conversation_history"] == [{"system": greeting}]
        assert after["conversation_history"][-2:] == [
            {"user": "hello"},
            {"system": "ok"},
        ]
        assert after["current_state"] == "chat" and after["fsm_id"] == api.fsm_id

    def test_the_end_state_of_a_turn_is_the_start_state_of_the_next(
        self, log: EventLog
    ):
        api = _api(log)
        cid, _ = api.start_conversation()
        api.converse("one", cid)
        api.converse("two", cid)
        first_start, first_end, second_start, second_end = _of(log, cid, START, END)
        assert first_end["state_ref"] == second_start["state_ref"]
        assert second_end["state_ref"] != second_start["state_ref"]
        # the record written at start is the state the first turn started from
        record = _of(log, cid, RECORD)[0]
        assert record["state_ref"] == first_start["state_ref"]

    def test_the_start_record_holds_the_greeting(self, log: EventLog):
        api = _api(log)
        cid, greeting = api.start_conversation()
        (record,) = _of(log, cid, RECORD)
        assert greeting == "ok"
        assert record["source"] == "start" and record["state"] == "chat"
        assert _state(log, record)["conversation_history"] == [{"system": "ok"}]

    def test_advance_is_one_logged_turn_without_a_user_message(self, log: EventLog):
        api = _api(log)
        cid, _ = api.start_conversation()
        api.advance(cid)
        start, end = _of(log, cid, START, END)
        assert start["has_user_message"] is False and "user_message_ref" not in start
        assert end["outcome"] == "committed"

    def test_a_consumed_stream_is_one_committed_turn(self, log: EventLog):
        api = _api(log)
        cid, _ = api.start_conversation()
        assert "".join(api.converse_stream("hi", cid)) == "ok"
        assert _boundary(log, cid) == [RECORD, START, END]
        assert _of(log, cid, END)[0]["outcome"] == "committed"
        assert "".join(api.advance_stream(cid)) == "ok"
        assert _boundary(log, cid)[-2:] == [START, END]

    def test_a_turn_run_inside_an_except_block_is_still_committed(self, log: EventLog):
        """``sys.exc_info()`` in a ``finally`` would answer with the CALLER's exception here."""
        api = _api(log)
        cid, _ = api.start_conversation()
        try:
            raise ValueError("handled by the caller")
        except ValueError:
            api.converse("hi", cid)
            "".join(api.converse_stream("again", cid))
        assert [e["outcome"] for e in _of(log, cid, END)] == ["committed", "committed"]

    def test_a_pass2_failure_is_rolled_back_to_the_pre_turn_state(self, log: EventLog):
        llm = _llm()
        api = _api(log, llm)
        cid, _ = api.start_conversation()
        llm.responses.clear()  # the next Pass 2 is an unscripted call
        with pytest.raises(FSMError):
            api.converse("hi", cid)
        start, end = _of(log, cid, START, END)
        assert end["outcome"] == "rolled_back"
        assert end["state_ref"] == start["state_ref"]
        assert (
            api.get_conversation_history(cid)
            == _state(log, start)["conversation_history"]
        )

    def test_an_abandoned_stream_is_abandoned(self, log: EventLog):
        api = _api(log)
        cid, _ = api.start_conversation()
        stream = api.converse_stream("hi", cid)
        assert next(stream) == "o"
        stream.close()
        assert _boundary(log, cid) == [RECORD, START, END]
        start, end = _of(log, cid, START, END)
        assert end["outcome"] == "abandoned"
        assert end["state_ref"] != start["state_ref"]  # the partial reply is kept

    def test_an_interrupt_is_interrupted(self, log: EventLog):
        llm = _Interrupting(responses={"chat": "ok"})
        api = _api(log, llm)
        with patch.object(
            _Interrupting, "generate_response", ScriptedLLM.generate_response
        ):
            cid, _ = api.start_conversation()  # the greeting is answered normally
        with pytest.raises(KeyboardInterrupt):
            api.converse("hi", cid)
        assert _of(log, cid, END)[0]["outcome"] == "interrupted"

    def test_a_pass1_failure_that_keeps_a_partial_commit_is_abandoned(
        self, log: EventLog
    ):
        llm = ScriptedLLM(
            extraction_data={"name": "Bo"}, responses={"chat": "ok", "done": "bye"}
        )
        boom = (
            create_handler("boom")
            .at(HandlerTiming.PRE_TRANSITION)
            .critical()
            .do(lambda ctx: (_ for _ in ()).throw(RuntimeError("handler failed")))
        )
        api = _api(log, llm, _FSM_PARTIAL, handlers=[boom])
        cid, _ = api.start_conversation()
        with pytest.raises(FSMError):
            api.converse("I am Bo", cid)
        start, end = _of(log, cid, START, END)
        assert end["outcome"] == "abandoned"
        assert _state(log, end)["context_data"].get("name") == "Bo"
        assert "name" not in _state(log, start)["context_data"]

    def test_a_refused_turn_on_a_terminal_state_logs_the_pair(self, log: EventLog):
        api = _api(log, definition=_FSM_GO)
        cid, _ = api.start_conversation()
        assert api.converse("go", cid) == "bye"
        with pytest.raises(FSMError, match="ended"):
            api.converse("again", cid)
        assert [e["outcome"] for e in _of(log, cid, END)] == [
            "committed",
            "rolled_back",
        ]
        assert _boundary(log, cid)[-2:] == [START, END]

    def test_a_failing_turn_start_fails_the_turn_and_leaks_no_lock(self, log: EventLog):
        api = _api(log)
        cid, _ = api.start_conversation()
        with patch.object(jsonl, "_write", side_effect=OSError("disk full")):
            with pytest.raises(FSMError):
                api.converse("hi", cid)
        manager = api.fsm_manager
        assert manager._active_turns == set()
        assert manager._conversation_locks[api.get_sub_conversation_id(cid)].acquire(
            blocking=False
        )
        manager._conversation_locks[api.get_sub_conversation_id(cid)].release()
        # the message was never added: the history is the greeting only
        assert api.get_conversation_history(cid) == [{"system": "ok"}]

    def test_a_start_that_cannot_be_logged_ends_the_conversation(self, log: EventLog):
        api = _api(log)
        with patch.object(jsonl, "_write", side_effect=OSError("disk full")):
            with pytest.raises(FSMError):
                api.start_conversation()
        assert api.list_active_conversations() == []
        assert api.fsm_manager.instances == {}

    def test_a_failing_turn_end_hides_neither_the_reply_nor_the_lock(
        self, log: EventLog
    ):
        api = _api(log)
        cid, _ = api.start_conversation()
        # The turn has two checkpoints: the Pass-2 `llm/request`, then `turn/end`. The first
        # flush succeeds, the second fails: only `turn/end` is lost, the reply is not.
        with patch.object(jsonl, "_fsync", side_effect=[None, OSError("no flush")]):
            assert api.converse("hi", cid) == "ok"  # turn/end is a checkpoint
        assert api.fsm_manager._active_turns == set()
        with pytest.raises(FSMError):  # the poisoned writer refuses the next turn
            api.converse("again", cid)


class TestOutcomeRule:
    @pytest.mark.parametrize(
        ("error", "unchanged", "expected"),
        [
            (None, True, "committed"),
            (None, False, "committed"),
            (KeyboardInterrupt, True, "interrupted"),
            (SystemExit, False, "interrupted"),
            (GeneratorExit, True, "abandoned"),
            (GeneratorExit, False, "abandoned"),
            (FSMError, True, "rolled_back"),
            (FSMError, False, "abandoned"),
            (RuntimeError, True, "rolled_back"),
            (RuntimeError, False, "abandoned"),
        ],
    )
    def test_the_rule(self, error: Any, unchanged: bool, expected: str):
        from fsm_llm.engine.events import _outcome

        assert _outcome(error, unchanged) == TurnOutcome(expected)


class TestStateBlob:
    """``state_blob`` carries internal keys only through the definition's own allow-list (D-009)."""

    def test_only_the_declared_transcript_keys_travel(self, log: EventLog):
        from fsm_llm.engine.events import declared_internal_keys, state_blob
        from tests.test_fsm_llm.test_completion_state import _tool_fsm

        transcript = [{"role": "user", "content": "What is 2 + 3?"}]
        api = API.from_definition(
            _tool_fsm(), llm_interface=ScriptedLLM(), event_log=log
        )
        cid, _ = api.start_conversation(
            initial_context={
                "_completion_messages": transcript,
                "_approval_granted": {"tool": "pay", "id": "forged-grant"},
                "_run_output_keys": ["answer"],
                "system_note": "internal-note",
                "api_key": "sk-SECRET-0001",
                "user_name": "Bo",
            }
        )
        assert declared_internal_keys(api.fsm_definition) == ("_completion_messages",)
        (record,) = _of(log, cid, RECORD)
        blob = _state(log, record)
        assert blob["internal"] == {"_completion_messages": transcript}
        assert blob["context_data"] == {"api_key": REDACTED, "user_name": "Bo"}
        raw = tree_bytes(log.root)
        for forbidden in (b"forged-grant", b"_run_output_keys", b"internal-note"):
            assert forbidden not in raw
        assert b"sk-SECRET-0001" not in raw
        instance = api.fsm_manager.instances[api.get_sub_conversation_id(cid)]
        assert (
            "_approval_granted" in instance.context.data
        )  # present in memory, not logged
        assert state_blob(instance, api.fsm_definition)["internal"] == blob["internal"]

    def test_a_definition_with_no_completion_state_declares_no_internal_key(self):
        from fsm_llm.engine.events import declared_internal_keys

        assert declared_internal_keys(FSMDefinition(**_FSM)) == ()


class TestStrangler:
    def test_with_the_log_off_nothing_is_bound_or_written(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.chdir(tmp_path)
        seen: list[object] = []

        class Spy(ScriptedLLM):
            def generate_response(self, request: Any) -> Any:
                seen.append(current_scope())
                return super().generate_response(request)

        def refuse(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("event log work with the log off")

        llm = Spy(responses={"chat": "ok", "done": "bye"}, stream_chunks=["o", "k"])
        with (
            patch("fsm_llm.engine.api.events.activate", refuse),
            patch("fsm_llm.engine.api.events.scoped_iter", refuse),
            patch("fsm_llm.engine.events.state_blob", refuse),
        ):
            api = API(_FSM, llm_interface=llm)
            cid, _ = api.start_conversation()
            api.converse("a", cid)
            api.advance(cid)
            "".join(api.converse_stream("b", cid))
            api.close()
        assert seen and all(scope is None for scope in seen)
        assert list(tmp_path.iterdir()) == []

    def test_replies_and_data_are_identical_with_the_log_on_and_off(
        self, log: EventLog
    ):
        def run(api: API) -> tuple[Any, ...]:
            cid, greeting = api.start_conversation()
            replies = [
                api.converse("a", cid),
                api.advance(cid).response,
                "".join(api.converse_stream("b", cid)),
                "".join(api.advance_stream(cid)),
            ]
            return (
                greeting,
                replies,
                api.get_data(cid),
                api.get_current_state(cid),
                api.get_conversation_history(cid),
            )

        off = run(API(_FSM, llm_interface=_llm()))
        on = run(_api(log))
        assert on == off
        assert len(log.conversation_ids()) == 1

    def test_the_lock_order_holds_with_the_log_on(self, log: EventLog):
        from tests.test_fsm_llm.test_fsm_lock_order import _OPERATIONS, _build

        for operation in (
            "process_message",
            "advance",
            "process_message_stream",
            "advance_stream",
        ):
            manager, cid, events = _build()
            with bind(log, f"lock-order-{operation.replace('_', '-')}"):
                _OPERATIONS[operation](manager, cid)
            assert order_problems(events) == []
            assert _short(events) == _PINNED[operation], operation
        written = {
            n: _types(log, n)
            for n in log.conversation_ids()
            if n.startswith("lock-order-")
        }
        assert len(written) == 4
        assert all(
            kinds[:1] == [START] and kinds[-1:] == [END] for kinds in written.values()
        )

    def test_the_writer_takes_no_manager_lock(self, log: EventLog):
        """Appends happen on the turn thread while ``conv_lock`` is held, and take only the
        log's own lock: the recorded sequence has no lock the pinned one lacks."""
        from tests.test_fsm_llm.test_fsm_lock_order import _build

        manager, cid, events = _build()
        with bind(log, "own-lock-only"):
            manager.process_message(cid, "hi")
        assert {name for name, _, _ in events} <= {"lock", "cache", "conv"}
        assert _short(events) == _PINNED["process_message"]
        assert _boundary(log, "own-lock-only") == [START, END]


class TestOptIn:
    def test_event_log_is_keyword_only(self):
        parameter = inspect.signature(API.__init__).parameters["event_log"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is None

    def test_a_path_makes_an_owned_log_that_close_releases(self, tmp_path: Path):
        api = API(_FSM, llm_interface=_llm(), event_log=tmp_path / "owned")
        cid, _ = api.start_conversation()
        api.converse("hi", cid)
        owned = api._event_log
        assert isinstance(owned, EventLog) and owned.root == tmp_path / "owned"
        api.close()
        with pytest.raises(LogWriteError, match="closed"):
            owned.append(cid, EventType.TURN_START)
        assert (tmp_path / "owned" / "logs" / f"{cid}.jsonl").is_file()

    def test_a_string_path_is_accepted(self, tmp_path: Path):
        api = API(_FSM, llm_interface=_llm(), event_log=str(tmp_path / "text"))
        cid, _ = api.start_conversation()
        api.close()
        assert (tmp_path / "text" / "logs" / f"{cid}.jsonl").is_file()

    def test_a_passed_log_stays_open_after_close(self, log: EventLog):
        api = _api(log)
        cid, _ = api.start_conversation()
        api.close()
        assert log.append(cid, EventType.TURN_START) > 0

    def test_ending_a_conversation_releases_its_writer(self, log: EventLog):
        api = _api(log)
        cid, _ = api.start_conversation()
        api.converse("hi", cid)
        with pytest.raises(LogLockedError):
            EventLog(log.root).append(cid, EventType.TURN_START)
        api.end_conversation(cid)
        with EventLog(log.root) as other:  # the lock is free: the fd was released
            assert other.append(cid, EventType.TURN_START) > 0

    def test_an_unsupported_value_is_refused(self):
        with pytest.raises(TypeError, match="event_log"):
            API(_FSM, llm_interface=_llm(), event_log=42)  # type: ignore[arg-type]

    def test_an_injected_interface_and_an_event_log_go_together(self, log: EventLog):
        api = API(_FSM, llm_interface=_llm(), event_log=log)
        cid, _ = api.start_conversation()
        assert api.converse("hi", cid) == "ok"

    def test_the_builder_sets_the_log(self, log: EventLog):
        api = (
            APIBuilder()
            .set_definition(_FSM)
            .set_llm_interface(_llm())
            .set_event_log(log)
            .build()
        )
        cid, _ = api.start_conversation()
        assert _types(log, cid) == [RECORD]

    def test_a_pushed_child_turn_carries_the_root_id_and_a_frame(self, log: EventLog):
        api = _api(log)
        cid, _ = api.start_conversation()
        api.push_fsm(cid, FSMDefinition(**_FSM))
        api.converse("inside the child", cid)
        starts = _of(log, cid, START)
        assert [e["frame"] for e in starts] == [1]
        assert _of(log, cid, END)[0]["frame"] == 1


class TestThreads:
    def test_two_conversations_log_to_two_files_from_two_threads(self, log: EventLog):
        api = _api(log)
        ids = [api.start_conversation()[0] for _ in range(2)]
        errors: list[BaseException] = []

        def work(cid: str) -> None:
            try:
                for _ in range(3):
                    api.converse("hi", cid)
            except BaseException as exc:  # pragma: no cover - the assertion reports it
                errors.append(exc)

        threads = [threading.Thread(target=work, args=(cid,)) for cid in ids]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert errors == []
        for cid in ids:
            kinds = _types(log, cid)
            assert _boundary(log, cid) == [RECORD, *[START, END] * 3]
            assert [e["seq"] for e in log.read(cid)] == list(range(1, len(kinds) + 1))


class TestAHugeInt:
    def test_an_int_too_big_for_json_text_works_with_the_log_on_as_off(
        self, log: EventLog
    ):
        big = math.factorial(2000)  # over the 4,300 digit limit of int to text

        def run(event_log: EventLog | None) -> tuple[Any, ...]:
            api = API(_FSM, llm_interface=_llm(), event_log=event_log)
            cid, _ = api.start_conversation(initial_context={"n": big})
            api.update_context(cid, {"m": big + 1})
            reply = api.converse("hi", cid)
            data = api.get_data(cid)
            return reply, data["n"] == big, data["m"] == big + 1

        assert run(None) == run(log) == ("ok", True, True)
        stored = tree_bytes(log.root)
        assert f"<int:{big.bit_length()} bits>".encode() in stored


class _AbortsMidStream(ScriptedLLM):
    """Streams two chunks, then dies the way a ^C or ``sys.exit`` does."""

    abort: type[BaseException] = KeyboardInterrupt

    def generate_response_stream(self, request: Any) -> Any:
        yield "par"
        yield "tial"
        raise self.abort


class _RaisesBeforeItsIterator(ScriptedLLM):
    """A plain ``def``: it opens the connection eagerly and fails before any iterator exists."""

    def generate_response_stream(self, request: Any) -> Any:
        raise ConnectionError("connect failed before the first chunk")


class _EagerWrapper(_Wrapper):
    """A custom interface around a real one that reads the whole provider stream up front."""

    def generate_response_stream(self, request: Any) -> Any:
        return iter(list(self.inner.generate_response_stream(request)))


def _streamed_provider(**kwargs: Any) -> Any:
    return iter([_chunk("o"), _chunk("k")]) if kwargs.get("stream") else _reply("ok")


class TestAStreamThroughTheEngine:
    """D3 and D5 (plan 52ce5464, runtime audit 1.2, M1 and 2.4): what a streamed Pass 2 writes."""

    def test_an_eager_wrapper_around_a_real_interface_logs_one_request(
        self, log: EventLog
    ) -> None:
        api = _api(log, _EagerWrapper())
        with patch("fsm_llm.llm.providers.litellm.completion", _streamed_provider):
            cid, _ = api.start_conversation()
            assert "".join(api.converse_stream("hi", cid)) == "ok"
        streamed = [e for e in _of(log, cid, "llm/request") if e["stream"]]
        assert [e["source"] for e in streamed] == ["engine"]  # never also "provider"
        (response,) = [e for e in _of(log, cid, "llm/response") if e.get("stream")]
        assert response["abandoned"] is False
        assert response["request_seq"] == streamed[0]["seq"]

    def test_a_plain_def_stream_that_raises_is_a_provider_error_not_an_abandon(
        self, log: EventLog
    ) -> None:
        api = _api(log, _RaisesBeforeItsIterator(responses={"chat": "ok"}))
        cid, _ = api.start_conversation()
        with pytest.raises(
            FSMError
        ):  # the manager wraps it, chained from the ConnectionError
            "".join(api.converse_stream("hi", cid))
        (request,) = [e for e in _of(log, cid, "llm/request") if e["stream"]]
        (error,) = _of(log, cid, "provider/error")
        assert error["error_type"] == "ConnectionError" and error["stream"] is True
        assert error["request_seq"] == request["seq"]
        assert not [e for e in _of(log, cid, "llm/response") if e.get("stream")]
        assert usage_from_events(log.read(cid)).errors == 1  # the outage is counted
        assert _of(log, cid, END)[-1]["outcome"] == "rolled_back"

    @pytest.mark.parametrize("abort", [KeyboardInterrupt, SystemExit])
    def test_a_non_exception_abort_mid_stream_is_an_abandoned_reply_and_no_provider_error(
        self, log: EventLog, abort: type[BaseException]
    ) -> None:
        """The pin of what a ^C does (D5): the reply is ``abandoned``, the turn ``interrupted``,
        no ``provider/error`` (only an ``Exception`` is an outage), and a restore holds what the
        live conversation holds."""
        llm = _AbortsMidStream(responses={"chat": "ok"})
        llm.abort = abort
        api = _api(log, llm)
        cid, _ = api.start_conversation()
        with pytest.raises(abort):
            "".join(api.converse_stream("hi", cid))
        events = [e for e in log.read(cid) if e["type"].startswith(("llm/", "turn/"))]
        assert [e["type"] for e in events][-3:] == [
            "llm/request",
            "llm/response",
            "turn/end",
        ]
        assert events[-2]["abandoned"] is True and events[-2]["stream"] is True
        assert events[-1]["outcome"] == "interrupted"
        assert not _of(log, cid, "provider/error")
        restored = fold_session(log.read(cid), log.get_json)
        assert restored is not None
        assert restored.conversation_history == api.get_conversation_history(cid)
