"""The signals reader of one frame (``runtime/signals.py``, decision D-012 of plan c720601d).

Real rows drive PRODUCT code (``make_rig`` synthesis runs and plain ``API`` turns over scripted or
patched models) at BOTH privacy levels; the pure rows hand-build the events a real log cannot be
made to write (a killed turn, a committed turn with one failed call). The rule under test:
failures count EVERY ``provider/error`` of the frame, evidence only comes from clean turn windows,
no blob, no ``source``, no ``ts``.
"""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
from typing import Any, ClassVar
from unittest import mock

import pytest

from fsm_llm import API
from fsm_llm.core.codes import LlmFailureCode
from fsm_llm.core.definitions import LLMResponseError
from fsm_llm.llm import LiteLLMInterface
from fsm_llm.runtime import EventLog
from fsm_llm.runtime.blobs import BlobStore
from fsm_llm.runtime.signals import FrameSignals, frame_signals
from fsm_llm.runtime.usage import provider_errors_since, watch_frame_signals
from fsm_llm.synthesis import SynthesisPolicy
from fsm_llm.testing import ScriptedLLM, final_reply
from tests.test_fsm_llm.support import _state
from tests.test_fsm_llm_synthesis.conftest import ScriptedRunnerLLM, good_fsm
from tests.test_fsm_llm_synthesis.rig import chat_reply, make_rig, named_error

LEVELS = ("values", "shape")
COMPLETION = "fsm_llm.llm.providers.litellm.completion"


class _Down(ScriptedRunnerLLM):
    """A cheap model whose every field extraction fails with a provider error of a named class."""

    def __init__(self, name: str) -> None:
        super().__init__({}, {})
        self.name = name

    def extract_field(self, request: Any) -> Any:
        raise LLMResponseError("provider down") from named_error(self.name)


class _DownAfterGreeting(ScriptedRunnerLLM):
    """A cheap model that fails the reply of the child after its greeting: the turn rolls back."""

    greeted = False

    def generate_response(self, request: Any) -> Any:
        if self._state_of(request) == "ask":
            if self.greeted:
                raise LLMResponseError("provider down") from named_error(
                    "RateLimitError"
                )
            self.greeted = True
        return super().generate_response(request)


def _rig_signals(
    tmp_path: Path, level: str, cheap: ScriptedRunnerLLM
) -> tuple[FrameSignals, Any, EventLog]:
    """Run a synthesized child to its budget and read its frame (1) through the watcher."""
    log = EventLog(tmp_path / "ev", privacy=level)
    rig = make_rig(
        good_fsm(), cheap=cheap, policy=SynthesisPolicy(max_turns=3), event_log=log
    )
    read = watch_frame_signals(log, rig.conv, 1)
    try:
        rig.run(30)
    except LLMResponseError:
        pass  # a reply that cannot be written fails the run: the log is what is read
    got = read()
    assert got == frame_signals(log.read(rig.conv), 1)
    return got, rig, log


@pytest.mark.parametrize("level", LEVELS)
class TestOutageFromTheRealLog:
    @pytest.mark.parametrize(
        ("name", "code"),
        [
            ("RateLimitError", LlmFailureCode.RATE_LIMITED),
            ("Timeout", LlmFailureCode.TIMEOUT),
            ("APIConnectionError", LlmFailureCode.TRANSPORT),
            ("AuthenticationError", LlmFailureCode.AUTH),
        ],
    )
    def test_an_outage_is_seen_as_failures_and_gives_no_evidence(
        self, tmp_path: Path, level: str, name: str, code: LlmFailureCode
    ):
        got, rig, log = _rig_signals(tmp_path, level, _Down(name))
        events = list(log.read(rig.conv))
        # the side effects of an outage really are in the log: a miss rides every failed turn
        assert [e for e in events if e["type"] == "extraction/miss"]
        assert got.failures == ((code.value, got.errors),) and got.errors >= 3
        assert got.errors == provider_errors_since(log, rig.conv, 1, 0)
        assert (got.misses, got.out_of_scope, got.extraction_failed, got.cut) == (
            (),
            (),
            (),
            (),
        )

    def test_a_rolled_back_outage_turn_keeps_its_failures_and_gives_no_evidence(
        self, tmp_path: Path, level: str
    ):
        got, rig, log = _rig_signals(tmp_path, level, _DownAfterGreeting({}, {}))
        ends = [e["outcome"] for e in log.read(rig.conv) if e["type"] == "turn/end"]
        events = list(log.read(rig.conv))
        assert "rolled_back" in ends
        assert [e for e in events if e["type"] == "extraction/miss"]  # the side effect
        assert got.failures == (("rate_limited", got.errors),) and got.errors >= 1
        assert (got.misses, got.extraction_failed, got.cut) == ((), (), ())

    def test_an_unclassified_error_is_the_unknown_code(
        self, tmp_path: Path, level: str
    ):
        got, _, _ = _rig_signals(tmp_path, level, _Down("SomethingElse"))
        assert got.failures == (("unknown", got.errors),)

    def test_a_committed_turn_that_misses_a_key_is_a_clean_miss(
        self, tmp_path: Path, level: str
    ):
        got, _, _ = _rig_signals(tmp_path, level, ScriptedRunnerLLM({}, {}))
        assert got.failures == () and got.errors == 0
        assert got.misses == (("ask", "answer", 3),)
        assert got.extraction_failed == () and got.cut == ()

    @pytest.mark.parametrize("data", [{"town": "Rome"}, {"answer": None}])
    def test_a_wrong_key_or_a_null_value_is_a_miss_and_nothing_else(
        self, tmp_path: Path, level: str, data: dict[str, Any]
    ):
        got, _, _ = _rig_signals(tmp_path, level, ScriptedRunnerLLM(data, {}))
        assert got.misses == (("ask", "answer", 3),)
        assert (got.failures, got.extraction_failed, got.cut) == ((), (), ())

    def test_a_frame_other_than_the_childs_gives_nothing(
        self, tmp_path: Path, level: str
    ):
        _, rig, log = _rig_signals(tmp_path, level, _Down("RateLimitError"))
        assert frame_signals(log.read(rig.conv), 0) == FrameSignals()
        assert frame_signals(log.read(rig.conv), 2) == FrameSignals()

    def test_no_blob_is_read(self, tmp_path: Path, level: str):
        log = EventLog(tmp_path / "ev", privacy=level)
        rig = make_rig(
            good_fsm(),
            cheap=_Down("RateLimitError"),
            policy=SynthesisPolicy(max_turns=2),
            event_log=log,
        )
        read = watch_frame_signals(log, rig.conv, 1)
        rig.run(30)
        reads: list[str] = []
        real = BlobStore.get

        def spy(self: BlobStore, ref: str) -> bytes:
            reads.append(ref)
            return real(self, ref)

        with mock.patch.object(BlobStore, "get", spy):
            got = read()
            direct = frame_signals(log.read(rig.conv), 1)
        assert reads == [] and got == direct and got.errors > 0


def _provider(extract: Any) -> Any:
    def call(**kwargs: Any) -> Any:
        system = kwargs["messages"][0]["content"] if kwargs["messages"] else ""
        if "the value of the field" in system or kwargs.get("response_format"):
            return extract
        return chat_reply(json.dumps({"message": "hello", "reasoning": "r"}))

    return call


@pytest.mark.parametrize("level", LEVELS)
class TestACutReplyFromTheRealLog:
    def test_a_cut_extraction_reply_is_cut_evidence_and_no_failure_code(
        self, tmp_path: Path, level: str
    ):
        log = EventLog(tmp_path / "ev", privacy=level)
        llm = LiteLLMInterface(model="gpt-4o", api_key="sk-unused")
        with mock.patch(COMPLETION, _provider(chat_reply('{"val', "length"))):
            rig = make_rig(
                good_fsm(),
                cheap=llm,
                policy=SynthesisPolicy(max_turns=2),
                event_log=log,
            )
            rig.run(30)
        got = frame_signals(log.read(rig.conv), 1)
        assert got.failures == () and got.errors == 0
        assert got.cut and all(state == "ask" for state, _ in got.cut)
        assert LlmFailureCode.MALFORMED_RESPONSE.value not in dict(got.failures)


def _bulk_fsm() -> dict[str, Any]:
    """One state that extracts by instruction only (the bulk call), staying until ``city``."""
    return {
        "name": "Bulk",
        "description": "Bulk extraction",
        "initial_state": "ask",
        "states": {
            "ask": _state(
                "ask",
                extraction_instructions="Extract the city",
                response_instructions="Ask",
                transitions=[
                    {
                        "target_state": "end",
                        "description": "city known",
                        "conditions": [
                            {
                                "description": "c",
                                "requires_context_keys": ["city"],
                                "logic": {"!=": [{"var": "city"}, None]},
                            }
                        ],
                    }
                ],
            ),
            "end": _state("end"),
        },
    }


@pytest.mark.parametrize("level", LEVELS)
class TestAFailedBulkExtractionFromTheRealLog:
    @pytest.mark.parametrize(
        "reply", [chat_reply(""), chat_reply('{"extracted_data', "length")]
    )
    def test_an_empty_or_cut_reply_is_extraction_failed_and_no_provider_error(
        self, tmp_path: Path, level: str, reply: Any
    ):
        log = EventLog(tmp_path / "ev", privacy=level)
        llm = LiteLLMInterface(model="gpt-4o", api_key="sk-unused")
        with mock.patch(COMPLETION, _provider(reply)):
            api = API(_bulk_fsm(), llm_interface=llm, event_log=log)
            cid, _ = api.start_conversation()
            api.converse("hello", cid)
        got = frame_signals(log.read(cid), 0)
        assert got.extraction_failed == (("ask", 1),)
        assert got.failures == () and got.errors == 0
        assert LlmFailureCode.MALFORMED_RESPONSE.value not in dict(got.failures)


_INTENTS = [
    {"name": "buy", "description": "wants to purchase"},
    {"name": "browse", "description": "is looking around"},
]


def _classify_fsm() -> dict[str, Any]:
    return {
        "name": "Classify",
        "description": "Classification",
        "initial_state": "ask",
        "states": {
            "ask": _state(
                "ask",
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
            ),
            "end": _state("end"),
        },
    }


@pytest.mark.parametrize("level", LEVELS)
class TestOutOfScopeFromTheRealLog:
    @pytest.mark.parametrize(("intent", "confidence"), [("browse", 0.9), ("buy", 0.2)])
    def test_a_fallback_or_a_discarded_intent_is_out_of_scope(
        self, tmp_path: Path, level: str, intent: str, confidence: float
    ):
        log = EventLog(tmp_path / "ev", privacy=level)
        reply = final_reply(
            json.dumps({"reasoning": "r", "intent": intent, "confidence": confidence})
        )
        llm = ScriptedLLM([reply], extraction_data={}, responses={"ask": "reply"})
        api = API(copy.deepcopy(_classify_fsm()), llm_interface=llm, event_log=log)
        cid, _ = api.start_conversation()
        api.converse("my private words", cid)
        got = frame_signals(log.read(cid), 0)
        assert got.out_of_scope == (("ask", "user_intent", 1),)

    def test_a_confident_in_scope_intent_is_not(self, tmp_path: Path, level: str):
        log = EventLog(tmp_path / "ev", privacy=level)
        reply = final_reply(
            json.dumps({"reasoning": "r", "intent": "buy", "confidence": 0.95})
        )
        llm = ScriptedLLM([reply], extraction_data={}, responses={"ask": "reply"})
        api = API(copy.deepcopy(_classify_fsm()), llm_interface=llm, event_log=log)
        cid, _ = api.start_conversation()
        api.converse("x", cid)
        assert frame_signals(log.read(cid), 0).out_of_scope == ()


def _ev(seq: int, kind: str, frame: int = 1, **fields: Any) -> dict[str, Any]:
    return {"seq": seq, "ts": f"t{seq}", "frame": frame, "type": kind, **fields}


def _turn(
    start: int, *middle: tuple[str, dict[str, Any]], outcome: str | None = "committed"
) -> list[dict[str, Any]]:
    """One turn window in state ``ask``: ``turn/start``, the middle events, ``turn/end``."""
    events = [_ev(start, "turn/start", state="ask", source="user")]
    for offset, (kind, fields) in enumerate(middle, 1):
        events.append(_ev(start + offset, kind, **fields))
    if outcome is not None:
        events.append(_ev(start + len(middle) + 1, "turn/end", outcome=outcome))
    return events


_MISS = ("extraction/miss", {"state": "ask", "keys": ["city"]})
_BAD = ("executor/result", {"state": "ask", "extraction_failed": True})
_CUT = ("llm/response", {"finish_reason": "length", "purpose": "extract"})
_FAIL = ("provider/error", {"failure_code": "timeout", "purpose": "extract"})


class TestTheWindowRule:
    def test_a_clean_window_gives_its_evidence(self):
        got = frame_signals(_turn(1, _MISS, _BAD, _CUT), 1)
        assert got.misses == (("ask", "city", 1),)
        assert got.extraction_failed == (("ask", 1),)
        assert got.cut == ()  # the cut reply is the failed extraction: one, not two
        assert got.failures == () and got.errors == 0

    def test_a_cut_reply_that_failed_the_extraction_is_one_bad_window(self):
        """Review concern 4 (probe p1): the real log of one cut turn holds two cut replies and
        one failed result, three events of one turn."""
        got = frame_signals(_turn(1, _CUT, _CUT, _BAD), 1)
        assert (got.extraction_failed, got.cut) == ((("ask", 1),), ())

    def test_a_cut_reply_that_did_not_fail_the_extraction_is_one_cut_window(self):
        assert frame_signals(_turn(1, _CUT, _CUT), 1).cut == (("ask", 1),)
        both = _turn(1, _CUT) + _turn(10, _BAD)  # two turns, two bad windows
        got = frame_signals(both, 1)
        assert (got.extraction_failed, got.cut) == ((("ask", 1),), (("ask", 1),))

    def test_several_events_of_a_kind_and_name_in_one_window_count_once(self):
        scope = ("intent/out_of_scope", {"state": "ask", "field": "tone"})
        twice = ("extraction/miss", {"state": "ask", "keys": ["city", "city"]})
        got = frame_signals(_turn(1, _MISS, _MISS, twice, scope, scope, _BAD, _BAD), 1)
        assert got.misses == (("ask", "city", 1),)
        assert got.out_of_scope == (("ask", "tone", 1),)
        assert got.extraction_failed == (("ask", 1),)

    def test_a_window_counts_each_key_once_and_a_count_is_the_number_of_turns(self):
        pair = ("extraction/miss", {"state": "ask", "keys": ["a", "b"]})
        events = []
        for start in (1, 10, 20):
            events += _turn(start, pair, pair, _CUT, _CUT, _BAD)
        got = frame_signals(events, 1)
        assert got.misses == (("ask", "a", 3), ("ask", "b", 3))
        assert (got.extraction_failed, got.cut) == ((("ask", 3),), ())

    def test_a_rolled_back_window_gives_no_evidence_and_keeps_its_failures(self):
        got = frame_signals(
            _turn(1, _MISS, _BAD, _CUT, _FAIL, outcome="rolled_back"), 1
        )
        assert got.failures == (("timeout", 1),) and got.errors == 1
        assert (got.misses, got.extraction_failed, got.cut) == ((), (), ())
        clean = frame_signals(_turn(1, _MISS, _BAD, _CUT, outcome="rolled_back"), 1)
        assert clean == FrameSignals()

    def test_a_tainted_committed_window_drops_the_evidence_and_keeps_the_failure(self):
        got = frame_signals(_turn(1, _MISS, _FAIL, _BAD), 1)
        assert got.failures == (("timeout", 1),)
        assert (got.misses, got.extraction_failed) == ((), ())

    def test_a_window_with_no_turn_end_gives_no_evidence_but_its_failures_count(self):
        got = frame_signals(_turn(1, _MISS, _FAIL, outcome=None), 1)
        assert got.failures == (("timeout", 1),) and got.misses == ()
        assert frame_signals(_turn(1, _MISS, outcome=None), 1) == FrameSignals()

    def test_a_killed_turn_does_not_swallow_the_next_clean_one(self):
        events = _turn(1, _MISS, outcome=None) + _turn(10, _MISS)
        assert frame_signals(events, 1).misses == (("ask", "city", 1),)

    def test_a_failure_outside_any_window_counts(self):
        got = frame_signals([_ev(1, "provider/error", failure_code="auth")], 1)
        assert got.failures == (("auth", 1),)

    def test_evidence_outside_a_window_is_ignored(self):
        events = [_ev(1, "extraction/miss", state="ask", keys=["city"])]
        assert frame_signals(events, 1) == FrameSignals()

    def test_counts_add_up_over_turns_and_are_sorted(self):
        events = _turn(1, _MISS) + _turn(10, _MISS)
        events += _turn(20, ("extraction/miss", {"state": "ask", "keys": ["a", "b"]}))
        assert frame_signals(events, 1).misses == (
            ("ask", "a", 1),
            ("ask", "b", 1),
            ("ask", "city", 2),
        )

    def test_a_cut_reply_of_another_purpose_is_no_cut(self):
        reply = ("llm/response", {"finish_reason": "length", "purpose": "reply"})
        assert frame_signals(_turn(1, reply), 1).cut == ()

    def test_a_foreign_failure_code_is_unknown(self):
        events = [_ev(1, "provider/error"), _ev(2, "provider/error", failure_code="x")]
        assert frame_signals(events, 1).failures == (("unknown", 2),)

    def test_a_torn_or_foreign_event_raises_nothing(self):
        events = _turn(
            1,
            ("extraction/miss", {"state": "ask", "keys": "city"}),
            ("extraction/miss", {"keys": ["city"]}),
            ("intent/out_of_scope", {"state": "ask"}),
            ("never/seen", {"x": 1}),
        )
        events.append({"seq": 99})
        assert frame_signals(events, 1) == FrameSignals()


class TestANameIsAnIdentifierOrTheEventIsNotRecorded:
    """Plan c720601d step 19.1: a model-written key or state name reaches the wire of a trace, so
    only an ASCII identifier of at most 80 characters is recorded (review probe p9, p10)."""

    HOSTILE: ClassVar[list[str]] = [
        "x,ask:ghost",  # the wire separators
        "a:b",
        "a,b",
        "a" * 81,
        "caf\u00e9",  # not ASCII
        "\u043a\u043b\u044e\u0447",
        "x\u200b",  # a zero width character
        "city\n",  # a trailing newline
        "1city",
        "a b",
        "",
    ]

    @pytest.mark.parametrize("name", HOSTILE)
    def test_a_hostile_key_is_dropped_and_the_good_one_beside_it_stays(self, name: str):
        miss = ("extraction/miss", {"state": "ask", "keys": [name, "city"]})

        assert frame_signals(_turn(1, miss), 1).misses == (("ask", "city", 1),)

    @pytest.mark.parametrize("name", HOSTILE)
    def test_a_hostile_state_gives_no_evidence_of_any_kind(self, name: str):
        events = (
            ("extraction/miss", {"state": name, "keys": ["city"]}),
            ("intent/out_of_scope", {"state": name, "field": "city"}),
            ("executor/result", {"state": name, "extraction_failed": True}),
        )

        assert frame_signals(_turn(1, *events), 1) == FrameSignals()

    @pytest.mark.parametrize("name", HOSTILE)
    def test_a_hostile_field_of_a_scope_event_is_dropped(self, name: str):
        scope = ("intent/out_of_scope", {"state": "ask", "field": name})

        assert frame_signals(_turn(1, scope), 1).out_of_scope == ()

    def test_the_longest_name_and_a_plain_one_are_kept(self):
        longest = "a" * 80
        miss = ("extraction/miss", {"state": longest, "keys": [longest, "_x9"]})

        got = frame_signals(_turn(1, miss), 1).misses

        assert got == ((longest, "_x9", 1), (longest, longest, 1))

    def test_the_one_rule_is_a_function_the_trace_shares(self):
        from fsm_llm.runtime.signals import is_signal_name

        assert is_signal_name("ask") and is_signal_name("a" * 80)
        assert not is_signal_name("a" * 81)
        assert not any(is_signal_name(n) for n in self.HOSTILE)
        assert not any(is_signal_name(n) for n in (None, 3, b"ask", ["ask"]))


class TestTheOrderAndTheFieldsItReads:
    def test_seq_is_the_order_never_ts(self):
        events = _turn(1, _MISS, _FAIL, _BAD)
        for event in events:
            event["ts"] = f"{99 - event['seq']:04d}"  # time runs backwards
        assert frame_signals(events, 1) == frame_signals(
            _turn(1, _MISS, _FAIL, _BAD), 1
        )
        clean = _turn(1, _MISS)
        for event in clean:
            event["ts"] = f"{99 - event['seq']:04d}"
        assert frame_signals(clean, 1).misses == (("ask", "city", 1),)

    @pytest.mark.parametrize("source", ["user", "simulator", "run", "remote_client"])
    def test_the_source_of_a_turn_changes_nothing(self, source: str):
        events = _turn(1, _MISS, _CUT)
        events[0]["source"] = source
        assert frame_signals(events, 1) == frame_signals(_turn(1, _MISS, _CUT), 1)

    def test_the_module_reads_no_source_ts_or_blob_name(self):
        path = Path(__file__).parents[2] / "src/fsm_llm/runtime/signals.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {
            n.value
            for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
        } - {ast.get_docstring(tree)}
        attributes = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        assert not {"source", "ts"} & names
        assert not {"get_json", "get_blob", "fold_session", "fold_stack"} & attributes
        imports = {
            n.module
            for n in ast.walk(tree)
            if isinstance(n, ast.ImportFrom) and n.module
        }
        assert not [m for m in imports if m.startswith("fsm_llm.engine")]


class TestTheWatcher:
    def test_no_log_gives_empty_signals(self):
        assert watch_frame_signals(None, "c", 1)() == FrameSignals()

    def test_a_failing_log_gives_empty_signals(self, tmp_path: Path):
        log = EventLog(tmp_path / "ev")
        read = watch_frame_signals(log, "c", 1)
        with mock.patch.object(EventLog, "query", side_effect=OSError("gone")):
            assert read() == FrameSignals()

    def test_a_missing_conversation_is_empty_not_an_error(self, tmp_path: Path):
        log = EventLog(tmp_path / "ev")
        assert watch_frame_signals(log, "never-written", 1)() == FrameSignals()
