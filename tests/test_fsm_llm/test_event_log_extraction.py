"""The extraction family: ``extraction``, ``extraction/miss`` and ``intent/out_of_scope`` (plan
3f4530d3 step 5a, decision D-008).

Every row drives PRODUCT code (a real ``API`` turn over a scripted model), never
``EventLog.append``. The rows pin: what each event names (state, key names, a delta blob at level
``values``, never a value at ``shape``), when it is written (after the model call that produced
the data, after the extractor's clean-up), what it never names (an internal-prefixed or
secret-looking key, user text), that a failed write is a WARNING and changes nothing the turn
returns, that the fold never moves on these types, and that the log off or on gives the same
replies, data and history.
"""

from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from fsm_llm import API
from fsm_llm.core.definitions import LLMResponseError
from fsm_llm.runtime import EventLog, LogWriteError, visible_entries
from fsm_llm.runtime.stores import fold_stack
from fsm_llm.testing import ScriptedLLM, final_reply
from tests.test_fsm_llm.event_log_support import tree_bytes
from tests.test_fsm_llm.support import _state

FAMILY = ("extraction", "extraction/miss", "intent/out_of_scope")
_INTENTS = [
    {"name": "buy", "description": "wants to purchase"},
    {"name": "browse", "description": "is looking around"},
]


_ENVELOPE = {"v", "seq", "ts", "conversation_id", "frame", "type"}
_ALLOWED = {
    "extraction": {"state", "keys", "delta_ref"},
    "extraction/miss": {"state", "keys"},
    "intent/out_of_scope": {"state", "field", "intent", "confidence", "low_confidence"},
}


def _classify(intent: str, confidence: float) -> Any:
    return final_reply(
        json.dumps({"reasoning": "r", "intent": intent, "confidence": confidence})
    )


def _fsm(**extra: Any) -> dict[str, Any]:
    """One state that asks for ``city`` and ``api_key`` and classifies ``user_intent``."""
    return {
        "name": "Extract",
        "description": "Extraction family",
        "initial_state": "ask",
        "states": {
            "ask": _state(
                "ask",
                required_context_keys=["city", "api_key"],
                response_instructions="Ask",
                classification_extractions=[
                    {
                        "field_name": "user_intent",
                        "intents": _INTENTS,
                        "fallback_intent": "browse",
                        "confidence_threshold": 0.7,
                    }
                ],
                transitions=[
                    {
                        "target_state": "end",
                        "description": "never",
                        "conditions": [{"description": "n", "logic": {"==": [1, 2]}}],
                    }
                ],
                **extra,
            ),
            "end": _state("end"),
        },
    }


class _Down(ScriptedLLM):
    """An extractor that fails: every field extraction raises an ``LLMResponseError``."""

    def extract_field(self, request: Any) -> Any:
        raise LLMResponseError("provider down")


def _llm(
    intent: str = "browse",
    confidence: float = 0.9,
    *,
    data: dict[str, Any] | None = None,
    cls: type[ScriptedLLM] = ScriptedLLM,
    replies: int = 1,
) -> ScriptedLLM:
    return cls(
        [_classify(intent, confidence)] * replies,
        extraction_data={"city": "Rome"} if data is None else data,
        responses={"ask": "reply"},
    )


def _api(log: EventLog | None, llm: ScriptedLLM, **fsm_extra: Any) -> API:
    return API(copy.deepcopy(_fsm(**fsm_extra)), llm_interface=llm, event_log=log)


def _turn(log: EventLog, llm: ScriptedLLM | None = None, **fsm_extra: Any) -> str:
    api = _api(log, llm or _llm(), **fsm_extra)
    cid, _ = api.start_conversation()
    api.converse("my private words", cid)
    return cid


def _of(log: EventLog, cid: str, *types: str) -> list[dict[str, Any]]:
    return [e for e in log.read(cid) if e["type"] in types]


class TestExtraction:
    def test_it_names_the_state_and_the_extracted_keys_and_holds_the_delta_by_reference(
        self, log: EventLog
    ):
        cid = _turn(log)
        (event,) = _of(log, cid, "extraction")
        assert event["state"] == "ask" and event["frame"] == 0
        assert event["keys"] == ["city", "user_intent"]
        assert log.get_json(event["delta_ref"]) == {
            "city": "Rome",
            "user_intent": "browse",
        }

    def test_it_comes_after_the_executor_result_and_before_the_transition(
        self, log: EventLog
    ):
        cid = _turn(log, _llm(data={}))
        order = [e["type"] for e in log.read(cid)]
        assert order.index("executor/result") < order.index("extraction")
        assert order.index("extraction") < order.index("extraction/miss")
        assert order.index("extraction/miss") < order.index("transition")

    def test_it_is_written_after_the_model_calls_that_produced_the_data(
        self, log: EventLog
    ):
        cid = _turn(log)
        events = list(log.read(cid))
        seq = next(e["seq"] for e in events if e["type"] == "extraction")
        responses = [
            e["seq"]
            for e in events
            if e["type"] == "llm/response"
            and e.get("call_type") in {"field_extraction", "classification"}
        ]
        assert responses and max(responses) < seq

    def test_nothing_is_written_when_nothing_was_extracted(self, log: EventLog):
        cid = _turn(log, _llm(data={}))
        # the fallback classification still stores its intent: a delta of one key
        assert [e["keys"] for e in _of(log, cid, "extraction")] == [["user_intent"]]
        llm = _llm(
            "buy", 0.2, data={}
        )  # a discarded low-confidence intent: no delta at all
        cid2 = _turn(log, llm)
        assert _of(log, cid2, "extraction") == []

    def test_a_secret_looking_or_internal_key_is_never_named_or_held(
        self, log: EventLog
    ):
        llm = _llm(data={"city": "Rome", "api_key": "sk-live-123"})
        cid = _turn(log, llm)
        (event,) = _of(log, cid, "extraction")
        assert "api_key" not in event["keys"]
        assert "sk-live-123" not in json.dumps(log.get_json(event["delta_ref"]))
        assert "sk-live-123" not in json.dumps(event)

    def test_the_visible_rule_drops_internal_secret_and_non_string_keys(self):
        entries = {
            "city": "Rome",
            "_internal": 1,
            "system_x": 2,
            "api_key": "sk-1",
            "password": "p",
            3: "n",
        }
        assert visible_entries(entries) == {"city": "Rome"}
        assert visible_entries(dict.fromkeys(["name", "_x", "user_token"])) == {
            "name": None
        }

    def test_the_text_of_the_user_message_is_in_no_event_field(self, log: EventLog):
        cid = _turn(log, _llm(data={}))
        events = _of(log, cid, *FAMILY)
        assert {e["type"] for e in events} == set(FAMILY)
        for event in events:
            assert "my private words" not in json.dumps(event)
            # a field the producer adds later (a model's reasoning text) fails here
            assert set(event) <= _ALLOWED[event["type"]] | _ENVELOPE, event


class TestExtractionMiss:
    def test_a_declared_key_still_none_is_a_miss_with_reason_null(self, log: EventLog):
        cid = _turn(log, _llm(data={}))
        (miss,) = _of(log, cid, "extraction/miss")
        assert miss["state"] == "ask" and "reason" not in miss
        assert miss["keys"] == ["city"]  # api_key is secret-looking: never named

    def test_an_extractor_failure_is_a_provider_error_and_the_key_is_still_a_miss(
        self, log: EventLog
    ):
        cid = _turn(log, _llm(data={}, cls=_Down))
        assert _of(log, cid, "provider/error")
        (miss,) = _of(log, cid, "extraction/miss")
        assert miss["keys"] == ["city"]

    def test_a_key_set_earlier_is_not_a_miss(self, log: EventLog):
        api = _api(log, _llm(replies=2))
        cid, _ = api.start_conversation()
        api.converse("first", cid)
        assert _of(log, cid, "extraction/miss") == []
        api.llm_interface.extraction_data = {}  # the second turn extracts nothing new
        before = len(list(log.read(cid)))
        api.converse("second", cid)
        later = [e for e in log.read(cid) if e["seq"] > before]
        assert [e for e in later if e["type"] == "extraction/miss"] == []

    def test_a_handler_only_key_is_never_a_miss(self, log: EventLog):
        fsm = _fsm()
        fsm["handler_only_keys"] = ["city"]
        api = API(fsm, llm_interface=_llm(data={}), event_log=log)
        cid, _ = api.start_conversation()
        api.converse("x", cid)
        assert _of(log, cid, "extraction/miss") == []

    def test_a_kind_with_no_declared_extraction_writes_neither_event(
        self, log: EventLog
    ):
        from tests.test_fsm_llm.test_event_log_producers import _run

        _, cid, spans = _run(log)
        first, last = spans["advance"]  # the completion state's turn
        got = [e["type"] for e in log.read(cid) if first <= e["seq"] <= last]
        assert "extraction" not in got and "extraction/miss" not in got


class TestIntentOutOfScope:
    def test_a_fallback_intent_is_out_of_scope_with_its_confidence(self, log: EventLog):
        cid = _turn(log, _llm("browse", 0.9))
        (event,) = _of(log, cid, "intent/out_of_scope")
        assert event["state"] == "ask" and event["field"] == "user_intent"
        assert event["intent"] == "browse" and event["confidence"] == 0.9
        assert event["low_confidence"] is False

    def test_a_discarded_low_confidence_intent_is_flagged(self, log: EventLog):
        cid = _turn(log, _llm("buy", 0.2))
        (event,) = _of(log, cid, "intent/out_of_scope")
        assert event["intent"] == "buy" and event["low_confidence"] is True

    def test_a_confident_in_scope_intent_writes_none(self, log: EventLog):
        cid = _turn(log, _llm("buy", 0.95))
        assert _of(log, cid, "intent/out_of_scope") == []

    def test_a_secret_looking_field_name_is_not_written(self, log: EventLog):
        fsm = _fsm()
        state = fsm["states"]["ask"]
        state["classification_extractions"][0]["field_name"] = "api_token"
        api = API(fsm, llm_interface=_llm(), event_log=log)
        cid, _ = api.start_conversation()
        api.converse("x", cid)
        assert _of(log, cid, "intent/out_of_scope") == []


class TestAtLevelShape:
    def test_the_family_holds_names_and_flags_and_no_reference_or_text(
        self, tmp_path: Path
    ):
        with EventLog(tmp_path / "shape", privacy="shape") as log:
            cid = _turn(log, _llm(data={}))
            events = _of(log, cid, *FAMILY)
        assert {e["type"] for e in events} == set(FAMILY)
        for event in events:
            assert not [k for k in event if k.endswith("_ref")]
        assert b"my private words" not in tree_bytes(tmp_path / "shape")


class TestAFailedWriteChangesNothing:
    @pytest.mark.parametrize("kind", FAMILY)
    def test_a_failing_emit_is_a_warning_and_the_turn_returns_the_same(
        self, tmp_path: Path, kind: str
    ):
        def observed(log: EventLog | None) -> Any:
            api = _api(log, _llm(data={}))
            cid, _ = api.start_conversation()
            reply = api.converse("x", cid)
            return reply, api.get_data(cid), api.get_conversation_history(cid)

        off = observed(None)
        real_emit = __import__("fsm_llm.runtime", fromlist=["emit"]).emit

        def failing(type: Any, **fields: Any) -> Any:
            if str(getattr(type, "value", type)) == kind:
                raise LogWriteError("disk full")
            return real_emit(type, **fields)

        with EventLog(tmp_path / "ev") as log:
            with (
                mock.patch("fsm_llm.engine.turn.events.emit", failing),
                mock.patch(
                    "fsm_llm.engine.kinds.conversation.classification.emit", failing
                ),
            ):
                on = observed(log)
        assert on == off

    def test_a_failing_blob_write_is_a_warning_too(self, tmp_path: Path):
        with EventLog(tmp_path / "ev") as log:
            api = _api(log, _llm())
            cid, _ = api.start_conversation()
            real = log.put_json

            def put_json(value: Any, *, kind: Any = None) -> str:
                if getattr(kind, "value", kind) == "delta":
                    raise LogWriteError("x")
                return real(value, kind=kind)

            with mock.patch.object(log, "put_json", put_json):
                assert api.converse("x", cid) == "reply"
            assert _of(log, cid, "extraction") == []


class TestTheFold:
    def test_a_late_family_event_at_frame_zero_after_a_push_keeps_the_child(
        self, log: EventLog
    ):
        from tests.test_fsm_llm import event_log_child as child

        api = _api(log, _llm())
        cid, _ = api.start_conversation()
        late = api.event_scope(cid)  # bound to frame 0 now
        api.push_fsm(cid, child.tool_fsm())
        from fsm_llm.runtime import EventType, emit

        with late:
            emit(EventType.EXTRACTION, state="ask", keys=["city"])
            emit(EventType.EXTRACTION_MISS, state="ask", keys=["city"], reason="null")
            emit(EventType.INTENT_OUT_OF_SCOPE, state="ask", field="f", intent="i")
        stack = fold_stack(log.read(cid), log.get_json)
        assert stack is not None and len(stack.frames) == 1


def _observed(log: EventLog | None, *, use_async: bool) -> dict[str, Any]:
    api = _api(log, _llm(replies=2, data={"city": "Rome"}))
    cid, _ = api.start_conversation()
    if use_async:
        reply = asyncio.run(api.aconverse("one", cid))
    else:
        reply = api.converse("one", cid)
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
                assert _of(on_log, cid, *FAMILY)

    def test_the_comparison_can_fail(self, tmp_path: Path):
        off = _observed(None, use_async=False)
        with EventLog(tmp_path / "x") as on_log:
            on = _observed(on_log, use_async=False)
        on["data"] = {**on["data"], "extra": 1}
        assert on != off

    def test_with_no_scope_the_producers_build_nothing(self):
        with mock.patch(
            "fsm_llm.engine.turn.events.visible_entries", side_effect=AssertionError
        ):
            api = _api(None, _llm(data={}))
            cid, _ = api.start_conversation()
            assert api.converse("x", cid) == "reply"
