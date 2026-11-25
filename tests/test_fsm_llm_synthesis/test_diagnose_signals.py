"""The diagnosis from real runs: seeded defects end to end (step 10 of plan c720601d, D-012 and
D-016).

Every row drives PRODUCT code: ``make_rig(event_log=)`` builds a real ``API`` with the real
``SynthesizeExecutor``, a seeded cheap model makes the child run end in a real non-ok state, the
``TraceRecorder`` folds the trace the executor wrote, and ``diagnose`` reads it. Each row runs at
BOTH privacy levels of the event log. What is pinned: an outage is never a fault (even with
misses in the log), a bad extraction reply and a cut one are ``malformed-extraction`` and never
``malformed_response``, a wrong key and a null value are an observed ``extraction-miss``, a
fallback intent and a discarded one are ``intent-out-of-scope``, a collection that resolves is no
fault, evidence of another state is no evidence, the ``shape`` log gives the diagnosis of the
``values`` log with no blob read, a lost reader leaves the new rules inert, and no value reaches
a trace row or a diagnosis.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest import mock

import pytest

from fsm_llm.core.codes import LlmFailureCode
from fsm_llm.core.definitions import (
    DataExtractionResponse,
    FSMDefinition,
    LLMResponseError,
)
from fsm_llm.llm import LiteLLMInterface
from fsm_llm.runtime import EventLog
from fsm_llm.runtime.blobs import BlobStore
from fsm_llm.runtime.signals import frame_signals
from fsm_llm.synthesis import (
    Diagnosis,
    RunTrace,
    SynthesisPolicy,
    TraceRecorder,
    diagnose,
)
from fsm_llm.synthesis.traces import signal_fields
from fsm_llm.testing import ScriptedLLM, final_reply
from tests.test_fsm_llm_synthesis.conftest import (
    ScriptedRunnerLLM,
    good_fsm,
    stuck_fsm,
)
from tests.test_fsm_llm_synthesis.rig import (
    REPLIES,
    Rig,
    chat_reply,
    make_rig,
    named_error,
)

LEVELS = ("values", "shape")
CANARY = "CANARY-7f3a91"
NPF = "not-procedure-fault"
COMPLETION = "fsm_llm.llm.providers.litellm.completion"
ACOMPLETION = "fsm_llm.llm.providers.litellm.acompletion"
WRITER = "fsm_llm.runtime.log.time"


def _down_error(name: str) -> LLMResponseError:
    error = LLMResponseError("provider down")
    error.__cause__ = named_error(name)
    return error


class _Seeded(ScriptedRunnerLLM):
    """The cheap model with a seeded defect. It counts the extraction calls of ``answer`` (two
    per turn while a gate is blocked). It extracts ``before``; with ``later`` or ``down`` given
    it switches after the call ``after`` (``after=0``: from the first call) to raising the
    outage named ``down`` or extracting ``later``. With neither it never switches. ``asked``
    is the spy: every extraction call as ``(field, the data it was answered from, the value or
    data it returned)``, so a row proves what the seed really did."""

    def __init__(
        self,
        before: dict[str, Any] | None = None,
        *,
        after: int = 0,
        later: dict[str, Any] | None = None,
        down: str = "",
    ) -> None:
        super().__init__(before or {}, dict(REPLIES))
        self.before, self.after, self.later, self.down = (
            before or {},
            after,
            later,
            down,
        )
        self.calls = 0
        self.asked: list[tuple[str, dict[str, Any], Any]] = []

    def _past(self) -> bool:
        return (self.later is not None or bool(self.down)) and self.calls > self.after

    def extract_field(self, request: Any) -> Any:
        if request.field_name == "answer":
            self.calls += 1
        if self._past() and self.down:
            raise _down_error(self.down)
        self.extraction_data = (self.later or {}) if self._past() else self.before
        reply = super().extract_field(request)
        self.asked.append((request.field_name, dict(self.extraction_data), reply.value))
        return reply

    def extract_bulk_data(self, request: Any) -> DataExtractionResponse:
        """The re-extraction of a blocked gate: the same data the field calls would give."""
        self.call_history.append(("extract_bulk_data", request))
        data = (self.later or {}) if self._past() else self.before
        self.asked.append(("bulk", dict(data), dict(data)))
        return DataExtractionResponse(extracted_data=dict(data))


def _provider(extract: Any) -> Callable[..., Any]:
    """A provider whose extraction calls answer ``extract`` and whose replies are well formed."""

    def call(**kwargs: Any) -> Any:
        system = kwargs["messages"][0]["content"] if kwargs["messages"] else ""
        if "the value of the field" in system or kwargs.get("response_format"):
            return extract
        return chat_reply(json.dumps({"message": "hello", "reasoning": "r"}))

    return call


def _classifying(intent: str, confidence: float) -> ScriptedLLM:
    reply = final_reply(
        json.dumps({"reasoning": "r", "intent": intent, "confidence": confidence})
    )
    cheap = ScriptedLLM([reply] * 12, extraction_data={}, responses=dict(REPLIES))
    cheap.model = "runner"
    return cheap


def _classify_fsm() -> dict[str, Any]:
    """ask classifies ``answer`` into buy or browse (fallback browse) and leaves on buy."""
    fsm = good_fsm()
    ask = fsm["states"]["ask"]
    ask.pop("field_extractions")
    ask["classification_extractions"] = [
        {
            "field_name": "answer",
            "intents": [
                {"name": "buy", "description": "wants to purchase"},
                {"name": "browse", "description": "is looking around"},
            ],
            "fallback_intent": "browse",
            "confidence_threshold": 0.7,
        }
    ]
    ask["transitions"][0]["conditions"] = [
        {
            "description": "buy",
            "requires_context_keys": ["answer"],
            "logic": {"==": [{"var": "answer"}, "buy"]},
        }
    ]
    return fsm


def _nick_stuck_fsm() -> dict[str, Any]:
    """ask collects ``answer`` (and an optional ``nick`` nobody gives), then ``mid`` is stuck on a
    value gate: the only misses of the run are in ``ask``, the stuck state is ``mid``."""
    fsm = stuck_fsm()
    fsm["states"]["ask"]["field_extractions"].append(
        {"field_name": "nick", "field_type": "str", "extraction_instructions": "a nick"}
    )
    return fsm


@dataclass
class Outcome:
    """What one seeded run left: the trace the executor wrote, the diagnosis, the rig and what
    the run read from the blob store."""

    trace: RunTrace
    diagnoses: tuple[Diagnosis, ...]
    rig: Rig
    reads: list[str]
    log_dir: Path
    stream: list[tuple[str, dict[str, Any]]]
    definition: FSMDefinition
    policy: SynthesisPolicy
    cheap: Any = None
    async_calls: list[str] = dataclasses.field(default_factory=list)

    @property
    def codes(self) -> list[tuple[str, str, tuple[str, ...]]]:
        return [(d.code, d.state, d.keys) for d in self.diagnoses]


def drive(
    tmp_path: Path,
    fsm: dict[str, Any],
    cheap: Any,
    level: str,
    *,
    turns: int = 3,
    classify: bool = False,
    context: dict[str, Any] | None = None,
    provider: Any = None,
    hook: Callable[[str, Rig], None] | None = None,
    clock: Callable[[], float] | None = None,
    use_async: bool = False,
) -> Outcome:
    """Run the seeded child to its end at ``level``, fold its trace and diagnose it. With
    ``use_async`` the run is ``arun_until_terminal`` and the provider is patched at its async
    binding too (``async_calls`` names what the async binding served)."""
    async_calls: list[str] = []
    policy = SynthesisPolicy(max_turns=turns, allow_classification=classify)
    traces: list[RunTrace] = []
    recorder = TraceRecorder(traces.append)
    holder: list[Rig] = []
    stream: list[tuple[str, dict[str, Any]]] = []

    def sink(event: str, fields: dict[str, Any]) -> None:
        if hook is not None and holder:
            hook(event, holder[0])
        stream.append((event, dict(fields)))
        recorder(event, fields)

    reads: list[str] = []
    real_get = BlobStore.get

    def spy(self: BlobStore, ref: str) -> bytes:
        reads.append(ref)
        return real_get(self, ref)

    log = EventLog(tmp_path / "ev", privacy=level)
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(BlobStore, "get", spy))
        if clock is not None:
            stack.enter_context(mock.patch(WRITER, SimpleNamespace(time=clock)))
        if provider is not None:
            stack.enter_context(mock.patch(COMPLETION, provider))

            async def aprovider(**kwargs: Any) -> Any:
                async_calls.append(kwargs.get("model", ""))
                return provider(**kwargs)

            stack.enter_context(mock.patch(ACOMPLETION, aprovider))
        rig = make_rig(
            fsm,
            cheap=cheap,
            policy=policy,
            sink=sink,
            event_log=log,
            context=context,
        )
        holder.append(rig)
        try:
            if use_async:
                asyncio.run(rig.api.arun_until_terminal(rig.conv, max_steps=40))
            else:
                rig.run(40)
        finally:
            log.close()
    (trace,) = traces
    definition = FSMDefinition.model_validate(fsm)
    got = diagnose([trace], definition, policy, return_keys=("answer",))
    return Outcome(
        trace,
        got,
        rig,
        reads,
        tmp_path / "ev",
        stream,
        definition,
        policy,
        cheap,
        async_calls,
    )


def _bulk_fsm(gate: str | None = None) -> dict[str, Any]:
    """ask extracts ``answer`` by instruction (the bulk call); with ``gate`` the edge needs that
    exact value, so a junk value from a cut reply still leaves the child stuck."""
    fsm = good_fsm()
    ask = fsm["states"]["ask"]
    ask.pop("field_extractions")
    ask["extraction_instructions"] = "The user's answer"
    if gate is not None:
        ask["transitions"][0]["conditions"][0]["logic"] = {
            "==": [{"var": "answer"}, gate]
        }
    return fsm


def _provider_llm() -> LiteLLMInterface:
    return LiteLLMInterface(model="gpt-4o", api_key="sk-unused")


class _OneFieldDown(ScriptedRunnerLLM):
    """``answer`` is extracted and ``nick`` fails with a rate limit: a turn that COMMITS with
    provider errors inside it."""

    def __init__(self) -> None:
        super().__init__({"answer": "x"}, dict(REPLIES))

    def extract_field(self, request: Any) -> Any:
        if request.field_name == "nick":
            raise _down_error("RateLimitError")
        return super().extract_field(request)


def _seed_outage(name: str) -> Callable[[Path, str], Outcome]:
    return lambda tmp, level: drive(tmp, good_fsm(), _Seeded(down=name), level)


def _seed_mixed(tmp: Path, level: str) -> Outcome:
    """Three clean turns of a wrong key (two extraction calls each), then a timeout for good."""
    return drive(
        tmp,
        good_fsm(),
        _Seeded({"town": "x"}, after=6, down="Timeout"),
        level,
        turns=8,
    )


def _seed_committed_outage(tmp: Path, level: str) -> Outcome:
    return drive(tmp, _nick_stuck_fsm(), _OneFieldDown(), level, turns=5)


def _seed_bad_reply(
    reply: Any, *, gate: str | None = None, turns: int = 3
) -> Callable[[Path, str], Outcome]:
    return lambda tmp, level: drive(
        tmp,
        _bulk_fsm(gate),
        _provider_llm(),
        level,
        turns=turns,
        provider=_provider(reply),
    )


def _seed_wrong_key(tmp: Path, level: str) -> Outcome:
    return drive(tmp, good_fsm(), _Seeded({"town": "x"}), level)


def _seed_null(tmp: Path, level: str) -> Outcome:
    return drive(tmp, good_fsm(), _Seeded({"answer": None}), level)


def _seed_intent(intent: str, confidence: float) -> Callable[[Path, str], Outcome]:
    return lambda tmp, level: drive(
        tmp, _classify_fsm(), _classifying(intent, confidence), level, classify=True
    )


def _seed_resolving(tmp: Path, level: str) -> Outcome:
    return drive(tmp, good_fsm(), _Seeded({}, after=2, later={"answer": "42"}), level)


def _seed_late_wrong_value(tmp: Path, level: str) -> Outcome:
    """Three misses of ``answer``, then it arrives but the gate wants another value: stuck on
    the value, no longer on a miss."""
    fsm = good_fsm()
    fsm["states"]["ask"]["transitions"][0]["conditions"][0]["logic"] = {
        "==": [{"var": "answer"}, "never"]
    }
    cheap = _Seeded({}, after=6, later={"answer": "42"})
    return drive(tmp, fsm, cheap, level, turns=6)


def _seed_other_state(tmp: Path, level: str) -> Outcome:
    """Three misses of ``nick`` in ``ask``, then the run is stuck on a gate of ``mid``."""
    cheap = _Seeded({}, after=4, later={"answer": "42"})
    return drive(tmp, _nick_stuck_fsm(), cheap, level, turns=7)


EMPTY = chat_reply("")
CUT = chat_reply('{"extracted_data', "length")
ASK = "ask"
MISS = ("extraction-miss", ASK, ("answer",))
SEEDS: dict[str, tuple[Callable[[Path, str], Outcome], list[tuple[Any, ...]]]] = {
    "rate_limited": (_seed_outage("RateLimitError"), [(NPF, "", ())]),
    "timeout": (_seed_outage("Timeout"), [(NPF, "", ())]),
    "transport": (_seed_outage("APIConnectionError"), [(NPF, "", ())]),
    "auth": (_seed_outage("AuthenticationError"), [(NPF, "", ())]),
    "unknown": (_seed_outage("SomethingOdd"), [(NPF, "", ())]),
    "mixed": (_seed_mixed, [(NPF, "", ())]),
    "committed": (_seed_committed_outage, [(NPF, "", ())]),
    "empty": (_seed_bad_reply(EMPTY), [("malformed-extraction", ASK, ())]),
    "cut": (_seed_bad_reply(CUT, gate="42"), [("malformed-extraction", ASK, ())]),
    "wrong_key": (_seed_wrong_key, [MISS]),
    "null": (_seed_null, [MISS]),
    "fallback": (
        _seed_intent("browse", 0.3),
        [("intent-out-of-scope", ASK, ("answer",))],
    ),
    "honest_fallback": (
        _seed_intent("browse", 0.9),
        [("value-gate-unmet", ASK, ("answer",))],
    ),
    "discarded": (
        _seed_intent("buy", 0.2),
        [("intent-out-of-scope", ASK, ("answer",))],
    ),
    "resolving": (_seed_resolving, [(NPF, "", ())]),
    "late_wrong_value": (
        _seed_late_wrong_value,
        [("value-gate-unmet", ASK, ("answer",))],
    ),
    "other_state": (_seed_other_state, [("value-gate-unmet", "mid", ("answer",))]),
}


def _answered(cheap: _Seeded, data: dict[str, Any], value: Any) -> bool:
    """Whether some extraction of ``answer`` was answered from ``data`` and gave ``value``."""
    return ("answer", data, value) in cheap.asked


# what the extractor of each seeded model was really asked and gave (the spy of ``_Seeded``)
SPIED: dict[str, Callable[[_Seeded], bool]] = {
    "rate_limited": lambda c: c.asked == [] and c.calls >= 3,  # it never answered
    "mixed": lambda c: _answered(c, {"town": "x"}, None) and c.calls > 6,
    "resolving": lambda c: (
        _answered(c, {}, None) and _answered(c, {"answer": "42"}, "42")
    ),
    "late_wrong_value": lambda c: (
        _answered(c, {}, None) and _answered(c, {"answer": "42"}, "42")
    ),
    "other_state": lambda c: _answered(c, {"answer": "42"}, "42"),
}


def _all_events(outcome: Outcome) -> list[dict[str, Any]]:
    log = EventLog(outcome.log_dir)
    try:
        return list(log.read(outcome.rig.conv))
    finally:
        log.close()


def _events(outcome: Outcome) -> list[dict[str, Any]]:
    """The events of the child frame (1) of the run."""
    return [e for e in _all_events(outcome) if e.get("frame") == 1]


def _trace_codes(trace: RunTrace) -> list[str]:
    return [code for code, _ in trace.failures]


@pytest.mark.parametrize("level", LEVELS)
class TestEverySeededDefectIsDiagnosed:
    @pytest.mark.parametrize("name", list(SEEDS))
    def test_the_seed_ends_in_a_real_run_and_gets_its_diagnosis(
        self, tmp_path: Path, level: str, name: str
    ):
        seed, expected = SEEDS[name]

        outcome = seed(tmp_path, level)

        assert outcome.codes == expected
        assert outcome.reads == []  # no run, at either level, reads a blob
        assert (
            outcome.rig.api.get_stack_depth(outcome.rig.conv) == 1
        )  # the child popped
        if name in SPIED:
            assert SPIED[name](outcome.cheap), outcome.cheap.asked
        if name == "resolving":
            assert outcome.trace.status == "ok"
        else:
            assert outcome.trace.status == "aborted"


@pytest.mark.parametrize("level", LEVELS)
class TestAnOutageIsNeverAFault:
    @pytest.mark.parametrize(
        ("name", "code"),
        [
            ("rate_limited", LlmFailureCode.RATE_LIMITED),
            ("timeout", LlmFailureCode.TIMEOUT),
            ("transport", LlmFailureCode.TRANSPORT),
            ("auth", LlmFailureCode.AUTH),
            ("unknown", LlmFailureCode.UNKNOWN),
        ],
    )
    def test_the_log_holds_the_misses_and_the_trace_the_code_and_no_fault(
        self, tmp_path: Path, level: str, name: str, code: LlmFailureCode
    ):
        outcome = SEEDS[name][0](tmp_path, level)

        events = _events(outcome)
        assert [
            e for e in events if e["type"] == "extraction/miss"
        ]  # the side effect is real
        assert set(_trace_codes(outcome.trace)) == {code.value}
        assert outcome.trace.failures[0][1] >= 3
        assert (outcome.trace.misses, outcome.trace.bad_extractions) == ((), ())
        assert [d.code for d in outcome.diagnoses] == [NPF]

    @pytest.mark.parametrize(
        "name", ["rate_limited", "timeout", "transport", "auth", "unknown"]
    )
    def test_the_code_alone_decides_when_the_reason_is_wrong_or_stale(
        self, tmp_path: Path, level: str, name: str
    ):
        trace = SEEDS[name][0](tmp_path, level).trace
        stale = dataclasses.replace(trace, reason="budget_spent")
        definition = FSMDefinition.model_validate(good_fsm())

        got = diagnose([stale], definition, SynthesisPolicy(), return_keys=("answer",))

        assert [d.code for d in got] == [NPF]

    def test_misses_before_the_outage_do_not_make_it_a_fault(
        self, tmp_path: Path, level: str
    ):
        outcome = SEEDS["mixed"][0](tmp_path, level)
        trace = outcome.trace

        assert trace.misses == (
            ("ask", "answer", 3),
        )  # three clean turns of a wrong key
        assert _trace_codes(trace) == ["timeout"]
        assert [d.code for d in outcome.diagnoses] == [NPF]
        # the same evidence without the outage is a fault: the code is what protects it
        bare = dataclasses.replace(trace, failures=(), reason="budget_spent")
        definition = FSMDefinition.model_validate(good_fsm())
        got = diagnose([bare], definition, SynthesisPolicy(), return_keys=("answer",))
        assert [(d.code, d.state, d.keys) for d in got] == [MISS]

    def test_a_turn_that_committed_with_a_failed_call_gives_no_evidence(
        self, tmp_path: Path, level: str
    ):
        outcome = SEEDS["committed"][0](tmp_path, level)

        events = _events(outcome)
        window: list[dict[str, Any]] = []
        tainted_commits = 0
        for event in events:
            if event["type"] == "turn/start":
                window = []
            window.append(event)
            if event["type"] == "turn/end" and event["outcome"] == "committed":
                tainted_commits += any(e["type"] == "provider/error" for e in window)
        assert tainted_commits >= 1  # a real committed turn with a provider error in it
        assert [e for e in events if e["type"] == "extraction/miss"]
        assert _trace_codes(outcome.trace) == ["rate_limited"]
        assert outcome.trace.misses == ()  # its miss of nick is dropped as tainted
        assert outcome.trace.bad_extractions == ()


@pytest.mark.parametrize("level", LEVELS)
class TestABadExtractionReplyIsNotAMalformedResponse:
    @pytest.mark.parametrize("name", ["empty", "cut"])
    def test_an_empty_or_a_cut_reply_is_malformed_extraction_with_no_failure_code(
        self, tmp_path: Path, level: str, name: str
    ):
        outcome = SEEDS[name][0](tmp_path, level)

        assert outcome.codes == [("malformed-extraction", ASK, ())]
        assert outcome.trace.failures == ()
        assert LlmFailureCode.MALFORMED_RESPONSE.value not in _trace_codes(
            outcome.trace
        )
        assert not [e for e in _events(outcome) if e["type"] == "provider/error"]
        assert outcome.trace.bad_extractions[0][0] == ASK

    def test_one_cut_turn_is_one_unit_not_three(self, tmp_path: Path, level: str):
        """Review concern 4 (probe p1): one cut reply wrote two cut replies and one failed
        result, which counted as three and met the threshold of three after ONE turn."""
        one = _seed_bad_reply(CUT, gate="42", turns=1)(tmp_path / "a", level)
        three = _seed_bad_reply(CUT, gate="42", turns=3)(tmp_path / "b", level)

        assert one.trace.bad_extractions == ((ASK, 1),)
        assert "malformed-extraction" not in [c for c, *_ in one.codes]
        assert three.trace.bad_extractions == ((ASK, 3),)
        assert [c for c, *_ in three.codes] == ["malformed-extraction"]

    def test_the_count_must_reach_the_dwell_threshold(self, tmp_path: Path, level: str):
        threshold = SynthesisPolicy().dwell_threshold
        short = _seed_bad_reply(EMPTY, turns=threshold - 1)(tmp_path / "a", level)
        full = _seed_bad_reply(EMPTY, turns=threshold)(tmp_path / "b", level)

        assert short.trace.bad_extractions == ((ASK, threshold - 1),)
        assert [c for c, *_ in short.codes] == ["extraction-miss"]  # not yet malformed
        assert full.trace.bad_extractions == ((ASK, threshold),)
        assert [c for c, *_ in full.codes] == ["malformed-extraction"]


@pytest.mark.parametrize("level", LEVELS)
class TestAMissIsObservedNotInferred:
    @pytest.mark.parametrize("name", ["wrong_key", "null"])
    def test_a_wrong_key_or_a_null_value_is_an_observed_extraction_miss(
        self, tmp_path: Path, level: str, name: str
    ):
        outcome = SEEDS[name][0](tmp_path, level)

        # the spy proves the seed: the extractor was asked for ``answer`` from this very data
        # (not from an empty dict) and gave nothing back (the review's probe p11)
        seeded = {"wrong_key": {"town": "x"}, "null": {"answer": None}}[name]
        asked = outcome.cheap.asked
        assert len(asked) >= 6 and {field for field, _, _ in asked} == {"answer"}
        assert all(data == seeded and value is None for _, data, value in asked)
        assert outcome.trace.misses == ((ASK, "answer", 3),)
        assert "answer" not in outcome.trace.keys
        assert outcome.codes == [MISS]

    def test_evidence_of_another_state_is_no_evidence_for_the_stuck_one(
        self, tmp_path: Path, level: str
    ):
        outcome = SEEDS["other_state"][0](tmp_path, level)

        assert outcome.trace.stuck == "mid"
        assert (
            ASK,
            "nick",
            3,
        ) in outcome.trace.misses  # three misses, in the other state
        assert outcome.codes == [("value-gate-unmet", "mid", ("answer",))]

    def test_a_miss_of_an_optional_field_the_gate_never_reads_is_not_the_fault(
        self, tmp_path: Path, level: str
    ):
        """Review concern 3 (probe p4), as a real run: ``nick`` is never given, ``answer`` is,
        and the gate on ``answer`` wants another value. The run is stuck on the value."""
        fsm = good_fsm()
        ask = fsm["states"]["ask"]
        ask["field_extractions"].append(
            {"field_name": "nick", "field_type": "str", "extraction_instructions": "n"}
        )
        ask["transitions"][0]["conditions"][0]["logic"] = {
            "==": [{"var": "answer"}, "never"]
        }

        outcome = drive(tmp_path, fsm, _Seeded({}, later={"answer": "42"}), level)

        assert outcome.trace.misses == ((ASK, "nick", 3),)  # the miss is real and clean
        assert "answer" in outcome.trace.keys
        assert outcome.codes == [("value-gate-unmet", ASK, ("answer",))]

    def test_a_collection_that_resolves_later_is_no_fault(
        self, tmp_path: Path, level: str
    ):
        outcome = SEEDS["resolving"][0](tmp_path, level)

        assert outcome.trace.status == "ok"
        assert outcome.trace.misses == (
            (ASK, "answer", 1),
        )  # the miss was real and clean
        assert [d.code for d in outcome.diagnoses] == [NPF]


@pytest.mark.parametrize("level", LEVELS)
class TestAFieldNamedLikeTheWireIsNoEvidenceForAnotherKey:
    """Review probe p10 (plan c720601d step 19.1): a model-written field ``x,ask:ghost`` is read
    back from the log as the key ``ghost`` of state ``ask`` by a decoder that splits on the
    wire separators. No state declares ``ghost``."""

    @pytest.mark.parametrize("name", ["x,ask:ghost", "x,mid:answer", "a:b"])
    def test_the_miss_of_such_a_field_is_not_a_miss_of_another_key(
        self, tmp_path: Path, level: str, name: str
    ):
        fsm = good_fsm()
        fsm["states"]["ask"]["field_extractions"].append(
            {"field_name": name, "field_type": "str", "extraction_instructions": "x"}
        )

        outcome = drive(tmp_path, fsm, _Seeded({"town": "x"}), level)

        logged = [e for e in _events(outcome) if e["type"] == "extraction/miss"]
        assert name in logged[0]["keys"]  # the real run missed the hostile field too
        assert outcome.trace.misses == ((ASK, "answer", 3),)
        assert outcome.codes == [MISS]


@pytest.mark.parametrize("level", LEVELS)
class TestAnIntentOutOfScope:
    @pytest.mark.parametrize("name", ["fallback", "discarded"])
    def test_a_fallback_or_a_discarded_intent_is_out_of_scope(
        self, tmp_path: Path, level: str, name: str
    ):
        outcome = SEEDS[name][0](tmp_path, level)

        assert outcome.trace.scopes == ((ASK, "answer", 3),)
        assert outcome.codes == [("intent-out-of-scope", ASK, ("answer",))]

    @pytest.mark.parametrize("turns", [1, 3, 5])
    def test_an_honest_fallback_never_flips_the_code_however_often_it_repeats(
        self, tmp_path: Path, level: str, turns: int
    ):
        """Review concern 6 (probe p2): a user who says "just browsing" three times used to turn
        ``value-gate-unmet`` into ``intent-out-of-scope``. The fallback at 0.9 is in scope."""
        outcome = drive(
            tmp_path,
            _classify_fsm(),
            _classifying("browse", 0.9),
            level,
            classify=True,
            turns=turns,
        )

        events = [e for e in _events(outcome) if e["type"] == "intent/out_of_scope"]
        assert len(events) == turns  # the event is still written, once per turn
        assert all(e["low_confidence"] is False for e in events)
        assert outcome.trace.scopes == ()
        assert outcome.codes == [("value-gate-unmet", ASK, ("answer",))]

    def test_a_confident_in_scope_intent_leaves_no_scope_and_no_fault(
        self, tmp_path: Path, level: str
    ):
        outcome = _seed_intent("buy", 0.95)(tmp_path, level)

        assert outcome.trace.status == "ok" and outcome.trace.scopes == ()
        assert [d.code for d in outcome.diagnoses] == [NPF]


class TestTheShapeLogGivesTheDiagnosisOfTheValuesLog:
    @pytest.mark.parametrize("name", list(SEEDS))
    def test_the_same_trace_and_diagnosis_and_no_blob_read_on_the_shape_run(
        self, tmp_path: Path, name: str
    ):
        seed = SEEDS[name][0]

        values = seed(tmp_path / "v", "values")
        shape = seed(tmp_path / "s", "shape")

        assert shape.trace == values.trace
        assert shape.diagnoses == values.diagnoses
        assert shape.reads == [] and values.reads == []
        shape_root = tmp_path / "s" / "ev"
        assert not (shape_root / "blobs").exists() or not list(
            (shape_root / "blobs").rglob("*")
        )  # a shape log holds no blob at all
        assert list((tmp_path / "v" / "ev" / "blobs").rglob("*"))  # a values log does

    def test_the_spy_can_see_a_read(self, tmp_path: Path):
        outcome = SEEDS["wrong_key"][0](tmp_path, "values")
        log = EventLog(outcome.log_dir)
        real = BlobStore.get
        seen: list[str] = []
        with mock.patch.object(
            BlobStore, "get", lambda self, ref: (seen.append(ref), real(self, ref))[1]
        ):
            refs = [
                e[k]
                for e in log.read(outcome.rig.conv)
                for k in e
                if k.endswith("_ref") and isinstance(e[k], str)
            ]
            log.get_json(refs[0])
        log.close()
        assert seen  # the spy is wired to the store the reader would use


@pytest.mark.parametrize("level", LEVELS)
class TestTheOrderIsSeqNeverTheClock:
    @pytest.mark.parametrize("name", ["mixed", "empty", "wrong_key"])
    def test_a_clock_that_runs_backwards_changes_nothing(
        self, tmp_path: Path, level: str, name: str
    ):
        ticks = iter(range(1_900_000_000, 1_000_000_000, -7))
        seeds = {
            "mixed": lambda tmp, lvl, **kw: drive(
                tmp,
                good_fsm(),
                _Seeded({"town": "x"}, after=6, down="Timeout"),
                lvl,
                turns=8,
                **kw,
            ),
            "empty": lambda tmp, lvl, **kw: drive(
                tmp,
                _bulk_fsm(),
                _provider_llm(),
                lvl,
                provider=_provider(EMPTY),
                **kw,
            ),
            "wrong_key": lambda tmp, lvl, **kw: drive(
                tmp, good_fsm(), _Seeded({"town": "x"}), lvl, **kw
            ),
        }
        forward = seeds[name](tmp_path / "f", level)
        backward = seeds[name](tmp_path / "b", level, clock=lambda: float(next(ticks)))

        stamps = [e["ts"] for e in _events(backward)]
        assert stamps == sorted(stamps, reverse=True) and len(set(stamps)) > 1
        assert backward.trace == forward.trace
        assert backward.diagnoses == forward.diagnoses


@pytest.mark.parametrize("level", LEVELS)
class TestALostReaderLeavesTheNewRulesInert:
    """A restart or a fork loses the reader taken at the push: the child's signals are empty and
    the diagnosis is the one the rules gave before the log existed (D-012, G2 of the map)."""

    @staticmethod
    def _lose(event: str, rig: Rig) -> None:
        if event == "pushed":
            rig.api._call_marks.signals.clear()

    def test_a_bad_extraction_reads_as_the_old_inference(
        self, tmp_path: Path, level: str
    ):
        outcome = drive(
            tmp_path,
            _bulk_fsm(),
            _provider_llm(),
            level,
            provider=_provider(EMPTY),
            hook=self._lose,
        )

        assert outcome.trace.status == "aborted" and outcome.trace.stuck == ASK
        assert (outcome.trace.misses, outcome.trace.bad_extractions) == ((), ())
        assert outcome.codes == [MISS]  # never malformed-extraction

    def test_a_fallback_intent_reads_as_the_old_gate_rule(
        self, tmp_path: Path, level: str
    ):
        outcome = drive(
            tmp_path,
            _classify_fsm(),
            _classifying("browse", 0.3),
            level,
            classify=True,
            hook=self._lose,
        )

        assert outcome.trace.scopes == ()
        assert [c for c, *_ in outcome.codes] == [
            "value-gate-unmet"
        ]  # never intent-out-of-scope

    def test_an_outage_with_a_lost_reader_is_still_no_fault_by_its_reason(
        self, tmp_path: Path, level: str
    ):
        outcome = drive(
            tmp_path, good_fsm(), _Seeded(down="Timeout"), level, hook=self._lose
        )

        assert outcome.trace.failures == () and outcome.trace.reason == "provider_error"
        assert [d.code for d in outcome.diagnoses] == [NPF]


def _gate_never() -> dict[str, Any]:
    """good_fsm whose edge wants the value ``never``: a value the user gave leaves it stuck."""
    fsm = good_fsm()
    fsm["states"]["ask"]["transitions"][0]["conditions"][0]["logic"] = {
        "==": [{"var": "answer"}, "never"]
    }
    return fsm


@pytest.mark.parametrize("level", LEVELS)
class TestNoValueReachesATraceOrADiagnosis:
    """Three ways a user value is in a run: a caller input, a value under a key nobody reads (the
    extractor is handed it and the pipeline never keeps it) and a value extracted under the
    declared key (it IS in the run). The spy proves each entered; none reaches a trace row, a
    diagnosis or the event stream, and only a values log holds the ones that did enter."""

    @staticmethod
    def _nowhere(outcome: Outcome) -> list[Path]:
        rows = json.dumps(outcome.trace.to_row())
        assert CANARY not in rows
        assert CANARY not in repr(outcome.diagnoses)
        assert outcome.stream and CANARY not in repr(outcome.stream)
        root = outcome.log_dir
        for path in root.rglob("*.jsonl"):
            assert CANARY not in path.read_text(encoding="utf-8")
        files = [p for p in root.rglob("*") if p.is_file()]
        return [p for p in files if CANARY.encode() in p.read_bytes()]

    def test_a_caller_input_is_nowhere_in_a_trace_and_in_a_values_log_only(
        self, tmp_path: Path, level: str
    ):
        outcome = drive(
            tmp_path,
            good_fsm(),
            _Seeded({"town": CANARY}),
            level,
            context={"topic": CANARY},
        )

        assert outcome.codes == [MISS] and outcome.trace.misses
        assert _answered(outcome.cheap, {"town": CANARY}, None)
        assert bool(self._nowhere(outcome)) == (level == "values")

    def test_a_value_under_a_wrong_key_is_handed_to_the_extractor_and_kept_nowhere(
        self, tmp_path: Path, level: str
    ):
        outcome = drive(tmp_path, good_fsm(), _Seeded({"town": CANARY}), level)

        assert outcome.codes == [MISS]
        assert _answered(
            outcome.cheap, {"town": CANARY}, None
        )  # the spy: it was handed
        assert self._nowhere(outcome) == []  # no pipeline step kept it, at either level

    def test_a_value_extracted_under_the_declared_key_is_in_a_values_log_only(
        self, tmp_path: Path, level: str
    ):
        outcome = drive(tmp_path, _gate_never(), _Seeded({"answer": CANARY}), level)

        assert outcome.codes == [("value-gate-unmet", ASK, ("answer",))]
        assert _answered(outcome.cheap, {"answer": CANARY}, CANARY)  # it was extracted
        assert "answer" in outcome.trace.keys  # and the run holds it
        assert bool(self._nowhere(outcome)) == (level == "values")


@pytest.mark.parametrize("level", LEVELS)
class TestTheAsyncRunLoopGivesTheSameCodesAndDiagnosis:
    """Review blind spot: nothing pinned the signals path through ``arun_until_terminal``. The
    outage seeds and the bad-reply seeds run again through the async loop; the trace and the
    diagnosis are the sync ones, and the async provider binding really served the call."""

    ASYNC: ClassVar[dict[str, Callable[[Path, str], Outcome]]] = {
        "rate_limited": lambda tmp, level: drive(
            tmp, good_fsm(), _Seeded(down="RateLimitError"), level, use_async=True
        ),
        "timeout": lambda tmp, level: drive(
            tmp, good_fsm(), _Seeded(down="Timeout"), level, use_async=True
        ),
        "empty": lambda tmp, level: drive(
            tmp,
            _bulk_fsm(),
            _provider_llm(),
            level,
            provider=_provider(EMPTY),
            use_async=True,
        ),
        "cut": lambda tmp, level: drive(
            tmp,
            _bulk_fsm("42"),
            _provider_llm(),
            level,
            provider=_provider(CUT),
            use_async=True,
        ),
        "wrong_key": lambda tmp, level: drive(
            tmp, good_fsm(), _Seeded({"town": "x"}), level, use_async=True
        ),
    }

    @pytest.mark.parametrize("name", list(ASYNC))
    def test_the_async_loop_leaves_the_trace_and_the_diagnosis_of_the_sync_one(
        self, tmp_path: Path, level: str, name: str
    ):
        asynchronous = self.ASYNC[name](tmp_path / "a", level)
        synchronous = SEEDS[name][0](tmp_path / "s", level)

        assert asynchronous.codes == SEEDS[name][1]
        assert asynchronous.trace == synchronous.trace
        assert asynchronous.diagnoses == synchronous.diagnoses
        assert asynchronous.reads == []
        if name in ("empty", "cut"):
            assert asynchronous.async_calls and not synchronous.async_calls
        else:
            assert (asynchronous.cheap.asked, asynchronous.cheap.calls) == (
                synchronous.cheap.asked,
                synchronous.cheap.calls,
            )


def _retrace(outcome: Outcome, events: list[dict[str, Any]]) -> RunTrace:
    """The trace the real run would have left had its log been ``events``: the recorded trace
    events of the run, with the signals of the child (frame 1) read from ``events``."""
    fields = signal_fields(frame_signals(events, 1))
    rows: list[RunTrace] = []
    recorder = TraceRecorder(rows.append)
    for event, got in outcome.stream:
        kept = {k: v for k, v in got.items() if k not in SIGNAL_KEYS}
        recorder(event, {**kept, **fields} if event == "aborted" else kept)
    (trace,) = rows
    return trace


SIGNAL_KEYS = ("failures", "misses", "scopes", "bad_extractions")


def _last(events: list[dict[str, Any]], kind: str) -> int:
    return max(i for i, e in enumerate(events) if e["type"] == kind and e["frame"] == 1)


def _diagnose(
    outcome: Outcome, trace: RunTrace
) -> list[tuple[str, str, tuple[str, ...]]]:
    got = diagnose([trace], outcome.definition, outcome.policy, return_keys=("answer",))
    return [(d.code, d.state, d.keys) for d in got]


@pytest.mark.parametrize("level", LEVELS)
class TestTheRealLogAsAFaultOrACrashLeavesIt:
    """Three things a real run does not write by itself but a real log can hold: a turn that
    rolled back with no provider error in it, a turn whose end was never written (a killed turn
    or a lost write), and events of another frame between the push and the read. The log of a
    real empty-reply run (one bad extraction per turn, three turns) is edited that way; the
    threshold is three, so each edit that drops one bad extraction moves the diagnosis from
    ``malformed-extraction`` to ``extraction-miss``."""

    @staticmethod
    def _empty(tmp_path: Path, level: str) -> tuple[Outcome, list[dict[str, Any]]]:
        outcome = SEEDS["empty"][0](tmp_path, level)
        events = _all_events(outcome)
        assert _retrace(outcome, events) == outcome.trace  # the replay is faithful
        return outcome, events

    def test_a_turn_that_rolled_back_without_a_provider_error_gives_no_evidence(
        self, tmp_path: Path, level: str
    ):
        outcome, events = self._empty(tmp_path, level)
        events[_last(events, "turn/end")]["outcome"] = "rolled_back"

        trace = _retrace(outcome, events)

        assert trace.bad_extractions == ((ASK, 2),) and trace.failures == ()
        assert [c for c, *_ in _diagnose(outcome, trace)] == ["extraction-miss"]

    def test_a_turn_with_no_end_gives_no_evidence(self, tmp_path: Path, level: str):
        outcome, events = self._empty(tmp_path, level)
        del events[_last(events, "turn/end")]

        trace = _retrace(outcome, events)

        assert trace.bad_extractions == ((ASK, 2),)
        assert [c for c, *_ in _diagnose(outcome, trace)] == ["extraction-miss"]

    def test_a_turn_killed_in_the_middle_is_dropped_and_the_next_one_still_counts(
        self, tmp_path: Path, level: str
    ):
        outcome, events = self._empty(tmp_path, level)
        first_end = next(
            i
            for i, e in enumerate(events)
            if e["type"] == "turn/end" and e["frame"] == 1
        )
        del events[first_end]

        trace = _retrace(outcome, events)

        assert trace.bad_extractions == ((ASK, 2),)  # not 3, and not 1

    def test_events_of_another_frame_are_nobody_elses_evidence(
        self, tmp_path: Path, level: str
    ):
        outcome, events = self._empty(tmp_path, level)
        del events[_last(events, "turn/end")]  # two bad extractions are left
        top = max(e["seq"] for e in events)
        foreign = [
            {"type": "turn/start", "state": ASK, "source": "user"},
            {"type": "executor/result", "state": ASK, "extraction_failed": True},
            {"type": "provider/error", "failure_code": "auth", "purpose": "extract"},
            {"type": "turn/end", "outcome": "committed"},
        ]
        for offset, event in enumerate(foreign, 1):
            events.append({"seq": top + offset, "frame": 0, **event})

        trace = _retrace(outcome, events)

        assert trace.bad_extractions == ((ASK, 2),) and trace.failures == ()
        assert [c for c, *_ in _diagnose(outcome, trace)] == ["extraction-miss"]

    def test_the_comparison_can_fail_the_unedited_log_is_malformed_extraction(
        self, tmp_path: Path, level: str
    ):
        outcome, events = self._empty(tmp_path, level)

        assert [c for c, *_ in _diagnose(outcome, _retrace(outcome, events))] == [
            "malformed-extraction"
        ]


@pytest.mark.parametrize("level", LEVELS)
class TestAMissThatLaterResolvedIsNoMissOfTheKey:
    def test_a_key_that_arrived_is_not_missing_when_the_run_is_stuck_on_its_value(
        self, tmp_path: Path, level: str
    ):
        outcome = SEEDS["late_wrong_value"][0](tmp_path, level)

        assert outcome.trace.misses == ((ASK, "answer", 3),)  # three real misses
        assert "answer" in outcome.trace.keys  # then the key arrived
        assert outcome.codes == [("value-gate-unmet", ASK, ("answer",))]


@pytest.mark.parametrize("level", LEVELS)
class TestARunWithNothingToReportWritesNoSignalKey:
    def test_an_ok_run_with_the_log_on_has_the_row_it_always_had(
        self, tmp_path: Path, level: str
    ):
        outcome = drive(
            tmp_path, good_fsm(), _Seeded({}, later={"answer": "42"}), level
        )

        assert outcome.trace.status == "ok"
        assert not set(SIGNAL_KEYS) & set(outcome.trace.to_row())
        assert [d.code for d in outcome.diagnoses] == [NPF]
