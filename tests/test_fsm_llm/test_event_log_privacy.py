"""Privacy of the event log (plan 5a6a1f80 step 10, roadmap row 1.11a): the ``shape`` level, the
``LogPrivacyError`` of every fold, and the user's redaction hook.

Three layers: a whole scenario under ``shape`` read as bytes (the done-when: nothing the user
typed, nothing a model said, no ``*_ref``, no ``blobs/``), the refusals of every fold, fork,
resume and restore on such a log (and on a ``values`` log a ``shape`` writer continued), and the
hook (output on disk, original absent, fail closed, after the built-in filters). Counts, bytes
and sequences, never timing.
"""

from __future__ import annotations

import ast
import copy
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

import fsm_llm
from fsm_llm import API, FileSessionStore, HandlerTiming, create_handler
from fsm_llm.core.definitions import FSMError
from fsm_llm.engine.session import EventLogSessionStore
from fsm_llm.llm import LiteLLMInterface
from fsm_llm.policy import HumanInTheLoop
from fsm_llm.runtime import (
    PRODUCED_TYPES,
    EventLog,
    EventPayloadError,
    EventType,
    LogPrivacyError,
    PrivacyLevel,
    RedactionKind,
    blob_ref,
    canonical_json,
)
from fsm_llm.runtime.privacy import SHAPE_FIELDS, run_redaction
from fsm_llm.runtime.stores import fold_session, fold_stack
from fsm_llm.testing import ScriptedLLM
from fsm_llm.tools import ToolCall
from tests.test_fsm_llm.event_log_support import _reply, tree_bytes, tree_files
from tests.test_fsm_llm.support import _c_log_capture
from tests.test_fsm_llm.test_call_delegated_settle import run_delegated
from tests.test_fsm_llm.test_event_log_engine import _FSM_PARTIAL
from tests.test_fsm_llm.test_event_log_extraction import _llm as extraction_llm
from tests.test_fsm_llm.test_event_log_extraction import _turn as extraction_turn
from tests.test_fsm_llm.test_event_log_handlers import _ok as handler_ok
from tests.test_fsm_llm.test_event_log_handlers import _turn as handler_turn
from tests.test_fsm_llm.test_event_log_producers import (
    _CHAT,
    _CHILD,
    _FSM,
    _TABLE,
    COMPLETION,
    NOT_YET_PRODUCED,
    _events,
    _Failing,
    _run,
    _script,
)

_ENVELOPE = {"v", "seq", "ts", "conversation_id", "frame", "type", "lvl"}
#: What the scenario of ``_run`` types, or a model says, that no shape byte may hold.
_TEXT = [
    b"Ann",
    b"hello child",
    b"left reply",
    b"working",
    b"all done",
    b"child hello",
    b"child bye",
    b"Greet the user",
    b"Extract the mood",
    b"Left reply",
    b"2+2?",
    b"FieldExtractionRequest",
    b"sha256:",
    b"_ref",
]


def _shape(tmp_path: Path, name: str = "ev") -> EventLog:
    return EventLog(tmp_path / name, privacy="shape")


def _walk(node: Any, edit: Any) -> Any:
    """*edit* applied to every string of a JSON value."""
    if isinstance(node, str):
        return edit(node)
    if isinstance(node, list):
        return [_walk(item, edit) for item in node]
    if isinstance(node, dict):
        return {key: _walk(value, edit) for key, value in node.items()}
    return node


def _scrub(kind: RedactionKind, value: Any) -> Any:
    return _walk(value, lambda text: text.replace("Ann", "[NAME]"))


def _chat_api(log: EventLog, llm: Any = None, **kwargs: Any) -> API:
    llm = llm or ScriptedLLM(responses={"chat": "ok"})
    return API(copy.deepcopy(_CHAT), llm_interface=llm, event_log=log, **kwargs)


def _all_events(log: EventLog) -> list[dict[str, Any]]:
    return [e for cid in log.conversation_ids() for e in log.read(cid)]


class TestTheLevelAndItsDefault:
    def test_the_default_level_is_values_and_it_stores_blobs(self, tmp_path: Path):
        with EventLog(tmp_path / "ev") as log:
            assert log.privacy is PrivacyLevel.VALUES and log.stores_values
            _, cid, _ = _run(log)
            assert all("lvl" not in e for e in log.read(cid))
            assert any(e.get("state_ref") for e in log.read(cid))
        assert (tmp_path / "ev" / "blobs").is_dir()
        assert b"Ann" in tree_bytes(tmp_path / "ev")  # the documented honest limit

    def test_the_level_is_a_closed_vocabulary(self, tmp_path: Path):
        assert PrivacyLevel.from_string("shape") is PrivacyLevel.SHAPE
        assert EventLog(tmp_path / "a", privacy=PrivacyLevel.SHAPE).privacy.value == (
            "shape"
        )
        for bad in ("Shape", "values ", "", None, 3):
            with pytest.raises(ValueError, match="privacy level"):
                PrivacyLevel.from_string(bad)  # type: ignore[arg-type]
            with pytest.raises(ValueError, match="privacy level"):
                EventLog(tmp_path / "b", privacy=bad)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="callable"):
            EventLog(tmp_path / "c", redact="nope")  # type: ignore[arg-type]

    def test_the_marker_is_reserved_at_every_level(self, tmp_path: Path):
        for log in (EventLog(tmp_path / "v"), _shape(tmp_path)):
            with log:
                with pytest.raises(EventPayloadError, match="lvl"):
                    log.append("c-1", EventType.TURN_START, lvl="values")
                with pytest.raises(EventPayloadError, match="lvl"):
                    log.append_batch("c-1", [{"type": "turn/start", "lvl": "x"}])
                assert not list(log.read("c-1"))

    def test_the_allow_list_is_per_type_names_only_and_never_a_reference(self):
        assert PRODUCED_TYPES <= set(SHAPE_FIELDS)
        assert (
            set(SHAPE_FIELDS) - PRODUCED_TYPES <= NOT_YET_PRODUCED
        )  # a list ahead of its producer
        for kind, names in SHAPE_FIELDS.items():
            assert not {n for n in names if n.endswith("_ref")}, kind
            assert not names & _ENVELOPE, kind
        assert SHAPE_FIELDS[EventType.TOOL_CALL] >= {"name", "call_id", "argument_keys"}
        assert "arguments" not in SHAPE_FIELDS[EventType.TOOL_CALL]

    def test_an_unknown_field_and_an_unlisted_type_fail_closed(self, tmp_path: Path):
        ref = "sha256:" + "a" * 64
        with _shape(tmp_path) as log:
            log.append(
                "c-1",
                EventType.TURN_START,
                state="s",
                has_user_message=True,
                state_ref=ref,
                user_message_ref=ref,
                note="typed text",
            )
            log.append("c-1", EventType.EXTRACTION, secret="typed text", n=1)
            first, second = log.read("c-1")
        assert {k for k in first} - _ENVELOPE == {"state", "has_user_message"}
        assert first["lvl"] == "shape"
        assert {k for k in second} <= _ENVELOPE  # declared, unproduced: no field at all
        assert second["lvl"] == "shape"
        assert b"typed text" not in tree_bytes(tmp_path / "ev")

    def test_a_shape_log_refuses_a_blob_and_creates_no_directory(self, tmp_path: Path):
        with _shape(tmp_path) as log:
            with pytest.raises(EventPayloadError, match="shape"):
                log.put_json({"a": 1})
            with pytest.raises(EventPayloadError, match="shape"):
                log.put_blob(b"x")
            assert log.digest_json({"a": 1}) == blob_ref(canonical_json({"a": 1}))
        assert not (tmp_path / "ev" / "blobs").exists()


class TestAShapeLog:
    def test_the_whole_scenario_keeps_its_events_and_holds_no_value(
        self, tmp_path: Path
    ):
        with _shape(tmp_path) as log:
            _, cid, spans = _run(log)
            events = list(log.read(cid))
            for name, (first, last) in spans.items():
                got = [
                    (e["frame"], e["type"]) for e in events if first <= e["seq"] <= last
                ]
                assert got == _TABLE[name], name
            assert (
                len(events) == 46
            )  # the root greeting's pair and two extractions included
            assert {e["lvl"] for e in events} == {"shape"}
            assert not [e for e in events if any(k.endswith("_ref") for k in e)]
        root = tmp_path / "ev"
        assert [p.name for p in tree_files(root)] == [f"{cid}.jsonl"]
        assert not (root / "blobs").exists()
        data = tree_bytes(root)
        assert [word for word in _TEXT if word in data] == []

    def test_every_type_the_log_produces_obeys_its_allow_list(self, tmp_path: Path):
        """The union of the scenario, a failing provider, a tool call, a budget stop and a
        provider-path call, every field of every event inside its type's list."""
        with _shape(tmp_path) as log:
            _run(log)
            failing = _Failing(responses={"chat": "hi"})
            api = _chat_api(log, failing)
            failing_id, _ = api.start_conversation()
            failing.down = True
            with pytest.raises(FSMError):
                api.converse("Ann", failing_id)
            capped = _chat_api(log, max_steps=1)
            capped_id, _ = capped.start_conversation()
            capped.converse("one", capped_id)
            with pytest.raises(FSMError):
                capped.converse("two", capped_id)
            secret = {"card": "4111 1111 1111 1111", "b": 1, "a": 2}
            with log.tool_call("tools-1", name="pay", call_id="c1", arguments=secret):
                pass
            run_delegated(log)  # run/start and run/end of a delegated call
            extraction_turn(log, extraction_llm(data={}))  # miss and out_of_scope (5a)
            handler_turn(log, handler_ok())  # a handler event (5b)
            with capped.event_scope(
                capped_id
            ):  # an approval ask (plan 3f4530d3 step 4)
                HumanInTheLoop(approval_callback=lambda request: True).request_approval(
                    ToolCall(tool_name="pay", parameters=secret), {}
                )
            events = _all_events(log)
        # `tool/call` has no product producer yet (step 3): the helper above writes it, so its
        # allow-list is still walked here
        wanted = PRODUCED_TYPES | {EventType.TOOL_CALL}
        assert {e["type"] for e in events} == {t.value for t in wanted}
        for event in events:
            allowed = SHAPE_FIELDS[EventType(event["type"])] | _ENVELOPE
            assert set(event) <= allowed, event["type"]
        call = next(e for e in events if e["type"] == "tool/call")
        assert call["argument_keys"] == ["a", "b", "card"]
        assert "arguments" not in call
        ask = next(e for e in events if e["type"] == "approval/request")
        assert ask["argument_keys"] == ["a", "b", "card"] and "arguments" not in ask
        assert b"4111" not in tree_bytes(tmp_path / "ev")
        budget = next(e for e in events if e["type"] == "decision")
        assert (budget["kind"], budget["budget"]) == ("budget_spent", "steps")

    def test_a_provider_path_call_keeps_its_codes_and_tokens_and_no_prompt(
        self, tmp_path: Path
    ):
        usage = {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15}
        reply = _reply("secret reply text")
        reply.usage = usage
        with _shape(tmp_path) as log:
            llm = LiteLLMInterface(model="gpt-4o")
            api = _chat_api(log, llm)
            with patch(COMPLETION, return_value=reply):
                cid, _ = api.start_conversation()
                api.converse("typed secret words", cid)
            with patch(COMPLETION, side_effect=RuntimeError("PRIVATE boom")):
                with pytest.raises(FSMError):
                    api.converse("second typed words", cid)
            events = list(log.read(cid))
        responses = [e for e in events if e["type"] == "llm/response"]
        assert responses and responses[0]["source"] == "provider"
        assert responses[0]["model"] == "gpt-4o"
        assert (responses[0]["prompt_tokens"], responses[0]["total_tokens"]) == (11, 15)
        (error,) = [e for e in events if e["type"] == "provider/error"]
        assert error["error_type"] == "RuntimeError"
        assert [e for e in events if e["type"] == "llm/request"]
        data = tree_bytes(tmp_path / "ev")
        for word in (b"secret reply", b"typed secret", b"second typed", b"PRIVATE"):
            assert word not in data
        assert b"_ref" not in data

    def test_a_rolled_back_turn_and_an_abandoned_turn_stay_distinct(
        self, tmp_path: Path
    ):
        with _shape(tmp_path) as log:
            llm = ScriptedLLM(responses={"chat": "ok"})
            api = _chat_api(log, llm)
            cid, _ = api.start_conversation()
            llm.responses.clear()  # the next Pass 2 is an unscripted call
            with pytest.raises(FSMError):
                api.converse("hi", cid)
            llm.responses["chat"] = "ok"
            stream = api.converse_stream("again", cid)
            next(stream)
            stream.close()
            api.converse("last", cid)
            outcomes = [e["outcome"] for e in _events(log, cid, "turn/end")]
        assert outcomes == ["rolled_back", "abandoned", "committed"]

    @pytest.mark.parametrize("variant", ["shape", "hook"])
    def test_a_partial_commit_is_abandoned_not_rolled_back(
        self, tmp_path: Path, variant: str
    ):
        """The state digest still differs when a failed turn kept a Pass-1 commit."""
        boom = (
            create_handler("boom")
            .at(HandlerTiming.PRE_TRANSITION)
            .critical()
            .do(lambda ctx: (_ for _ in ()).throw(RuntimeError("handler failed")))
        )
        log = (
            _shape(tmp_path)
            if variant == "shape"
            else EventLog(tmp_path / "ev", redact=_scrub)
        )
        with log:
            llm = ScriptedLLM(
                extraction_data={"name": "Bo"}, responses={"chat": "ok", "done": "bye"}
            )
            api = API(
                copy.deepcopy(_FSM_PARTIAL),
                llm_interface=llm,
                event_log=log,
                handlers=[boom],
            )
            cid, _ = api.start_conversation()
            with pytest.raises(FSMError):
                api.converse("I am Bo", cid)
            (end,) = _events(log, cid, "turn/end")
        assert end["outcome"] == "abandoned"

    def test_a_pushed_frame_records_names_and_counts_only(self, tmp_path: Path):
        with _shape(tmp_path) as log, _c_log_capture() as messages:
            api = API(copy.deepcopy(_FSM), llm_interface=_script(), event_log=log)
            cid, _ = api.start_conversation()
            api.push_fsm(cid, copy.deepcopy(_CHILD), return_context={"k": "TOPSECRET"})
            child = api.get_sub_conversation_id(cid)
            api.pop_fsm(cid, context_to_return={"result": "TOPSECRET"})
            push, pop = _events(log, cid, "stack/push", "stack/pop")
        # no producer even tried a blob: a refused put would be a swallowed WARNING
        assert [m for m in messages if "event log" in m] == []
        assert push["child_conversation_id"] == child and push["depth"] == 2
        assert pop["merged_key_count"] >= 1
        assert b"TOPSECRET" not in tree_bytes(tmp_path / "ev")

    def test_update_context_keeps_the_key_names_and_no_delta(self, tmp_path: Path):
        with _shape(tmp_path) as log:
            api = _chat_api(log)
            cid, _ = api.start_conversation()
            api.update_context(cid, {"note": "TOPSECRET", "other": 1})
            record = _events(log, cid, "context/record")[-1]
        assert (record["source"], record["keys"]) == (
            "update_context",
            ["note", "other"],
        )
        assert "delta_ref" not in record
        assert b"TOPSECRET" not in tree_bytes(tmp_path / "ev")

    def test_the_same_scenario_at_values_still_writes_every_ref(self, tmp_path: Path):
        """A parity row: a hook that returns its input and no hook write the same blobs."""
        names: list[set[str]] = []
        for i, hook in enumerate((None, lambda kind, value: value)):
            with EventLog(tmp_path / f"ev{i}", redact=hook) as log:
                _run(log)
            names.append(
                {
                    p.name
                    for p in tree_files(tmp_path / f"ev{i}")
                    if p.parent.parent.name == "blobs"
                }
            )
        assert names[0] == names[1] and len(names[0]) > 20


class TestEveryFoldRefusesAShapeLog:
    @staticmethod
    def _built(tmp_path: Path) -> tuple[str, Path]:
        with _shape(tmp_path) as log:
            _, cid, _ = _run(log)
        return cid, tmp_path / "ev"

    def test_the_folds_raise_and_an_empty_log_is_still_none(self, tmp_path: Path):
        cid, root = self._built(tmp_path)
        with EventLog(root) as log:  # whatever level the reader has
            for fold in (fold_session, fold_stack):
                with pytest.raises(LogPrivacyError, match="shape"):
                    fold(log.read(cid), log.get_json)
                assert fold(iter(()), log.get_json) is None
            with pytest.raises(LogPrivacyError):
                fold_session(log.query(cid, frame=1), log.get_json)  # any frame

    def test_the_marker_is_found_whatever_its_frame_or_type(self, tmp_path: Path):
        values = {"v": 1, "seq": 1, "ts": "t", "conversation_id": "c", "frame": 0}
        ok = {**values, "type": "turn/start", "state_ref": "sha256:" + "0" * 64}
        late = {**values, "seq": 2, "frame": 3, "type": "executor/start", "lvl": "x"}
        for fold in (fold_session, fold_stack):
            with pytest.raises(LogPrivacyError):
                fold([ok, late], lambda ref: {})

    def test_the_session_store_and_restore_raise_and_create_nothing(
        self, tmp_path: Path
    ):
        cid, root = self._built(tmp_path)
        with EventLog(root) as log:
            store = EventLogSessionStore(log)
            with pytest.raises(LogPrivacyError, match="state") as err:
                store.load(cid)
            assert isinstance(err.value, FSMError)
            assert store.load("never-logged") is None
            api = API(copy.deepcopy(_FSM), llm_interface=_script(), session_store=store)
            with pytest.raises(LogPrivacyError):
                api.restore_session(cid)
            assert api.list_active_conversations() == []

    def test_fork_is_refused_and_leaves_the_source_alone(self, tmp_path: Path):
        with _shape(tmp_path) as log:
            api, cid, _ = _run(log)
            for at_turn in (0, 1, 2):
                with pytest.raises(LogPrivacyError):
                    api.fork(cid, at_turn=at_turn)
            assert api.list_active_conversations() == [cid]
            assert api.get_current_state(cid)  # still usable

    def test_resume_is_refused_and_the_lock_is_released(self, tmp_path: Path):
        cid, root = self._built(tmp_path)
        before = tree_bytes(root)
        for level in ("shape", "values"):
            with EventLog(root, privacy=level) as log:
                api = API(copy.deepcopy(_FSM), llm_interface=_script(), event_log=log)
                with pytest.raises(LogPrivacyError):
                    api.resume(cid)
                assert not log.has_writer(cid)
                assert api.list_active_conversations() == []
        assert tree_bytes(root) == before  # nothing was appended

    def test_a_values_log_continued_by_a_shape_writer_is_refused_from_there(
        self, tmp_path: Path
    ):
        root = tmp_path / "ev"
        with EventLog(root) as log:
            api = _chat_api(log)
            cid, _ = api.start_conversation()
            api.converse("one", cid)
        with EventLog(root) as log:  # the log was values so far: a fold works
            assert fold_session(log.read(cid), log.get_json) is not None
        with EventLog(root, privacy="shape") as log:
            api = _chat_api(log)
            api.resume(cid)  # the values prefix reads fine
            api.converse("two", cid)
            with pytest.raises(LogPrivacyError):
                fold_session(log.read(cid), log.get_json)
            with pytest.raises(LogPrivacyError):
                api.fork(cid, at_turn=2)
            forked, state = api.fork(
                cid, at_turn=1
            )  # the cut ends before the shape part
            assert state.current_state == "chat" and api.get_current_state(forked)
        lines = list(EventLog(root).read(cid))
        assert "lvl" not in lines[0] and "lvl" in lines[-1]
        with EventLog(root) as log:
            with pytest.raises(LogPrivacyError):
                _chat_api(log).resume(cid)


class TestTheRedactionHook:
    def test_the_hook_output_is_on_disk_and_the_original_is_not(self, tmp_path: Path):
        seen: set[RedactionKind] = set()

        def hook(kind: RedactionKind, value: Any) -> Any:
            seen.add(kind)
            return _scrub(kind, value)

        with EventLog(tmp_path / "ev", redact=hook) as log:
            api, cid, _ = _run(log)
            assert api.get_data(cid)["name"] == "Ann"  # the engine still sees it
            refs = [e["state_ref"] for e in log.read(cid) if "state_ref" in e]
            assert "[NAME]" in str(log.get_json(refs[-1]))
        data = tree_bytes(tmp_path / "ev")
        assert b"Ann" not in data and b"[NAME]" in data
        assert seen >= {
            RedactionKind.PROMPT,
            RedactionKind.RESPONSE,
            RedactionKind.USER_MESSAGE,
            RedactionKind.STATE,
            RedactionKind.DELTA,
            RedactionKind.DEFINITION,
        }

    def test_a_turn_still_tells_rolled_back_from_abandoned_under_a_hook(
        self, tmp_path: Path
    ):
        with EventLog(tmp_path / "ev", redact=_scrub) as log:
            llm = ScriptedLLM(responses={"chat": "ok"})
            api = _chat_api(log, llm)
            cid, _ = api.start_conversation()
            llm.responses.clear()
            with pytest.raises(FSMError):
                api.converse("Ann", cid)
            assert _events(log, cid, "turn/end")[0]["outcome"] == "rolled_back"

    def test_the_inline_arguments_of_a_tool_call_go_through_it_after_the_filter(
        self, tmp_path: Path
    ):
        calls: list[tuple[RedactionKind, Any]] = []

        def hook(kind: RedactionKind, value: Any) -> Any:
            calls.append((kind, copy.deepcopy(value)))
            return _scrub(kind, value)

        arguments = {"who": "Ann", "api_key": "sk-live-0123456789abcdef0123456789"}
        with EventLog(tmp_path / "ev", redact=hook) as log:
            with log.tool_call("t-1", name="greet", call_id="c1", arguments=arguments):
                pass
            (call,) = [e for e in log.read("t-1") if e["type"] == "tool/call"]
            with log.tool_call(
                "t-1", name="big", call_id="c2", arguments={"x": "Ann" * 5000}
            ):
                pass
            big = [e for e in log.read("t-1") if e["type"] == "tool/call"][-1]
            assert log.get_json(big["arguments_ref"]) == {"x": "[NAME]" * 5000}
        (kind, seen), _ = calls[0], calls[1]
        assert kind is RedactionKind.ARGUMENTS
        assert seen["who"] == "Ann" and seen["api_key"] != arguments["api_key"]
        assert call["arguments"]["who"] == "[NAME]"
        assert b"sk-live" not in tree_bytes(tmp_path / "ev")
        assert len(calls) == 2  # one hook call per value: the blob path never re-hooks

    def test_a_secret_looking_context_entry_never_reaches_it_raw(self, tmp_path: Path):
        seen: list[str] = []

        def hook(kind: RedactionKind, value: Any) -> Any:
            seen.append(repr(value))
            return value

        with EventLog(tmp_path / "ev", redact=hook) as log:
            api = _chat_api(log)
            cid, _ = api.start_conversation(
                initial_context={"api_key": "sk-live-0123456789abcdef0123456789"}
            )
            api.update_context(cid, {"password": "hunter2hunter2"})
            api.converse("hi", cid)
        joined = "".join(seen)
        assert "sk-live" not in joined and "hunter2" not in joined
        assert "<redacted>" in joined

    def test_the_hook_gets_a_copy_so_it_cannot_change_what_is_sent(
        self, tmp_path: Path
    ):
        def vandal(kind: RedactionKind, value: Any) -> Any:
            if isinstance(value, dict):
                value.clear()
            return {"gone": True}

        sent: list[Any] = []

        def completion(**kwargs: Any) -> Any:
            sent.append(copy.deepcopy(kwargs["messages"]))
            return _reply("fine")

        with EventLog(tmp_path / "ev", redact=vandal) as log:
            api = _chat_api(log, LiteLLMInterface(model="gpt-4o"))
            with patch(COMPLETION, side_effect=completion):
                cid, _ = api.start_conversation()
                api.converse("hi", cid)
        assert sent and all(messages for messages in sent)
        assert b"gone" in tree_bytes(tmp_path / "ev")

    def test_the_calls_are_in_the_closed_vocabulary(self):
        assert [k.value for k in RedactionKind] == [
            "prompt",
            "response",
            "user_message",
            "state",
            "delta",
            "arguments",
            "definition",
        ]
        with pytest.raises(ValueError, match="redaction kind"):
            RedactionKind.from_string("other")


class TestTheHookFailsClosed:
    @staticmethod
    def _deep() -> list[Any]:
        value: list[Any] = []
        for _ in range(100_000):
            value = [value]
        return value

    @pytest.mark.parametrize(
        "result",
        [{1, 2}, object(), float("nan"), {"k": object()}, b"bytes", "deep"],
    )
    def test_a_result_that_is_not_json_or_too_deep_is_refused_and_nothing_is_written(
        self, tmp_path: Path, result: Any
    ):
        out = self._deep() if result == "deep" else result
        with EventLog(tmp_path / "ev", redact=lambda kind, value: out) as log:
            with pytest.raises(EventPayloadError):
                log.put_json({"a": 1}, kind="state")
        assert not (tmp_path / "ev" / "blobs").exists()

    def test_an_exception_is_a_payload_error_that_names_only_the_type(
        self, tmp_path: Path
    ):
        def hook(kind: RedactionKind, value: Any) -> Any:
            raise RuntimeError("PRIVATE hook message")

        with EventLog(tmp_path / "ev", redact=hook) as log:
            with pytest.raises(EventPayloadError, match="RuntimeError") as err:
                log.put_json({"a": 1}, kind="state")
        assert "PRIVATE" not in str(err.value) and err.value.__cause__ is None
        assert not (tmp_path / "ev" / "blobs").exists()

    def test_a_value_too_deep_to_hand_over_never_reaches_the_hook(self, tmp_path: Path):
        called: list[int] = []
        hook = lambda kind, value: called.append(1) or value  # noqa: E731
        with EventLog(tmp_path / "ev", redact=hook) as log:
            with pytest.raises(EventPayloadError, match="too deep"):
                log.put_json(self._deep(), kind="state")
        assert called == []

    def test_a_missing_or_unknown_kind_is_refused_while_a_hook_is_set(
        self, tmp_path: Path
    ):
        with EventLog(tmp_path / "ev", redact=lambda kind, value: value) as log:
            for kind in (None, "other", "State", 3):
                with pytest.raises(EventPayloadError, match="hook"):
                    log.put_json({"a": 1}, kind=kind)  # type: ignore[arg-type]
            assert log.get_json(log.put_json({"a": 1}, kind="state")) == {"a": 1}
        with pytest.raises(EventPayloadError, match="hook"):
            run_redaction(lambda kind, value: value, "other", {})

    def test_raw_bytes_are_refused_while_a_hook_is_set_and_allowed_without(
        self, tmp_path: Path
    ):
        with EventLog(tmp_path / "ev", redact=lambda kind, value: value) as log:
            with pytest.raises(EventPayloadError, match="hook"):
                log.put_blob(b"raw")
        with EventLog(tmp_path / "plain") as log:
            assert log.get_blob(log.put_blob(b"raw")) == b"raw"
            assert log.get_json(log.put_json({"a": 1}, kind="state")) == {"a": 1}

    def test_a_failing_prompt_hook_means_the_model_is_not_called(self, tmp_path: Path):
        def hook(kind: RedactionKind, value: Any) -> Any:
            if kind is RedactionKind.PROMPT:
                raise RuntimeError("no")
            return value

        llm = ScriptedLLM(responses={"chat": "ok"})
        with EventLog(tmp_path / "ev", redact=lambda k, v: v) as log:
            api = _chat_api(log, llm)
            cid, _ = api.start_conversation()
            log._redact = hook  # the hook that starts failing mid conversation
            made, history = len(llm.call_history), api.get_conversation_history(cid)
            with pytest.raises(FSMError):
                api.converse("typed words", cid)
            assert len(llm.call_history) == made  # fail closed: no model call
            assert api.get_conversation_history(cid) == history  # the turn rolled back
        assert b"FieldExtractionRequest" not in tree_bytes(tmp_path / "ev")

    def test_a_failing_prompt_hook_refuses_the_greeting_and_leaves_no_conversation(
        self, tmp_path: Path
    ):
        """The greeting is a logged request now (plan 3f4530d3 step 2): the hook that cannot
        redact its prompt means no model call, no registered conversation, no held writer."""

        def hook(kind: RedactionKind, value: Any) -> Any:
            if kind is RedactionKind.PROMPT:
                raise RuntimeError("no")
            return value

        llm = ScriptedLLM(responses={"chat": "ok"})
        with EventLog(tmp_path / "ev", redact=hook) as log:
            api = _chat_api(log, llm)
            with pytest.raises(Exception, match="could not be recorded"):
                api.start_conversation()
            assert llm.call_history == []
            assert api.list_active_conversations() == []
            assert [log.has_writer(c) for c in log.conversation_ids()] == [False] * len(
                log.conversation_ids()
            )

    def test_a_failing_state_hook_stops_the_turn_before_any_call(self, tmp_path: Path):
        def hook(kind: RedactionKind, value: Any) -> Any:
            if kind is RedactionKind.STATE:
                raise RuntimeError("no")
            return value

        llm = ScriptedLLM(responses={"chat": "ok"})
        with EventLog(tmp_path / "ev", redact=lambda k, v: v) as log:
            api = _chat_api(log, llm)
            cid, _ = api.start_conversation()
            log._redact = hook  # the hook that starts failing mid conversation
            made = len(llm.call_history)
            with pytest.raises(FSMError):
                api.converse("typed words", cid)
            assert len(llm.call_history) == made
            assert _events(log, cid, "turn/start") == []  # no turn began

    def test_a_failing_response_hook_never_hides_the_reply(self, tmp_path: Path):
        def hook(kind: RedactionKind, value: Any) -> Any:
            if kind is RedactionKind.RESPONSE:
                raise RuntimeError("no")
            return value

        with EventLog(tmp_path / "ev", redact=hook) as log:
            api = _chat_api(log)
            cid, _ = api.start_conversation()
            assert api.converse("hi", cid) == "ok"
            replies = _events(log, cid, "llm/response")
        assert replies and all(e.get("unrecorded") for e in replies)
        assert not any("response_ref" in e for e in replies)

    def test_a_failing_arguments_hook_stops_the_tool_before_it_runs(
        self, tmp_path: Path
    ):
        def hook(kind: RedactionKind, value: Any) -> Any:
            raise RuntimeError("no")

        ran: list[int] = []
        with EventLog(tmp_path / "ev", redact=hook) as log:
            with pytest.raises(EventPayloadError):
                with log.tool_call("t-1", name="pay", call_id="c1", arguments={"a": 1}):
                    ran.append(1)
            assert ran == [] and not list(log.read("t-1"))


class TestEveryProducerGoesThroughPutJson:
    """The sweep of the plan: a producer that wrote a blob another way would bypass the hook
    and the shape refusal."""

    @staticmethod
    def _modules() -> Iterator[tuple[Path, ast.Module]]:
        for path in sorted(Path(fsm_llm.__file__).parent.rglob("*.py")):
            yield path, ast.parse(path.read_text(encoding="utf-8"))

    def test_no_module_outside_the_log_reaches_the_blob_store_but_put_json(self):
        inner = {"log.py", "blobs.py"}
        for path, tree in self._modules():
            if path.name in inner and path.parent.name == "runtime":
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    assert node.func.attr != "put_blob", path
                    if node.func.attr == "put_json":
                        assert any(k.arg == "kind" for k in node.keywords), (
                            f"{path}:{node.lineno} put_json without a kind"
                        )
                if isinstance(node, ast.Attribute):
                    assert node.attr != "_blobs", path
                if isinstance(node, ast.Name):
                    assert node.id != "BlobStore" or path.name == "__init__.py", path

    def test_inside_the_log_only_three_methods_touch_the_store(self):
        path = Path(fsm_llm.__file__).parent / "runtime" / "log.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        methods = {
            m.name: m
            for cls in tree.body
            if isinstance(cls, ast.ClassDef) and cls.name == "EventLog"
            for m in cls.body
            if isinstance(m, ast.FunctionDef)
        }
        users = {
            name
            for name, method in methods.items()
            for node in ast.walk(method)
            if isinstance(node, ast.Attribute) and node.attr == "_blobs"
        }
        assert users == {
            "__init__",
            "put_blob",
            "put_json",
            "_append_call",
            "get_blob",
            "get_json",
        }
        for name in ("put_blob", "put_json"):
            assert any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "_refuse_values"
                for n in ast.walk(methods[name])
            ), name


def test_a_resume_on_a_values_log_still_restores_under_a_plain_session_store(
    tmp_path: Path,
):
    """The default path is untouched: values log, no hook, snapshot restore and log restore."""
    with EventLog(tmp_path / "ev") as log:
        api = _chat_api(log, session_store=FileSessionStore(tmp_path / "s"))
        cid, _ = api.start_conversation()
        api.converse("hi", cid)
        state = EventLogSessionStore(log).load(cid)
    assert state is not None and state.conversation_history
