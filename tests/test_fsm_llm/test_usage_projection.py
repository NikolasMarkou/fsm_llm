"""Usage as a projection (roadmap row 1.13, plan 5a6a1f80 step 9, D-014, D-015, D-027).

The meter of an interface is a ledger with a marker: ``reset_usage()`` moves the marker and keeps
the history, a mark survives it, and the ledger and the events of the log fold through ONE record
shape and ONE function (``fsm_llm.llm.usage.fold``). A reader picks one source, never the sum.
The provider is the patched ``fsm_llm.llm.providers.litellm.completion`` binding. Counts, never
timing.
"""

from __future__ import annotations

import copy
import importlib
import importlib.util
import sys
import threading
from types import ModuleType
from typing import Any
from unittest.mock import patch

import pytest
from litellm.types.utils import Choices, Message, ModelResponse

import fsm_llm.llm.usage as usage_module
from fsm_llm import API
from fsm_llm.core.definitions import (
    FSMError,
    LLMResponseError,
    LLMUsage,
    ResponseGenerationRequest,
)
from fsm_llm.llm import LiteLLMEmbedder, LiteLLMInterface, LLMInterface
from fsm_llm.llm.usage import (
    UsageRecord,
    _UsageMeter,
    fold,
    ledger_mark,
    ledger_since,
    usage_counts,
    usage_kind,
)
from fsm_llm.runtime import EventLog, bind, llm_call
from fsm_llm.testing import ScriptedLLM
from tests.test_fsm_llm.event_log_support import _chunk, _Wrapper
from tests.test_fsm_llm.test_event_log_producers import _CHAT, _FSM, _script
from tests.test_fsm_llm_regression.support import REPO

ROOT = "root-1"
COMPLETION = "fsm_llm.llm.providers.litellm.completion"
_FIELDS = (
    "calls",
    "errors",
    "usage_missing",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
)


def projection() -> ModuleType:
    """``fsm_llm.runtime.usage``, imported on use (a missing module fails the test using it)."""
    return importlib.import_module("fsm_llm.runtime.usage")


def _reply(text: str = "ok", prompt: int | None = 11, completion: int = 4) -> Any:
    kwargs: dict[str, Any] = {}
    if prompt is not None:
        kwargs["usage"] = {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }
    return ModelResponse(
        choices=[Choices(message=Message(content=text), finish_reason="stop")], **kwargs
    )


def _request(text: str = "hi") -> ResponseGenerationRequest:
    return ResponseGenerationRequest(system_prompt="Greet.", user_message=text)


def _llm() -> LiteLLMInterface:
    return LiteLLMInterface(model="gpt-4o", api_key="sk-unused")


def _events(log: EventLog, *types: str) -> list[dict[str, Any]]:
    return list(log.query(ROOT, types=types or None))


def _dump(usage: LLMUsage) -> dict[str, Any]:
    return usage.model_dump()


class TestTheLedgerAndItsMarker:
    def test_reset_moves_a_marker_and_keeps_the_history(self):
        meter = _UsageMeter()
        origin = meter.mark()
        meter.record("generate", _reply())
        meter.record_error("extract")

        harvested = meter.snapshot(reset=True)

        assert (harvested.calls, harvested.errors) == (2, 1)
        assert meter.snapshot() == LLMUsage()  # the view starts again from zero
        assert meter.since(origin).calls == 2  # the history is still there
        assert meter.since(origin).by_kind["extract"].errors == 1

    def test_a_mark_taken_before_a_reset_survives_it(self):
        meter = _UsageMeter()
        meter.record_error("generate")
        mark = meter.mark()
        meter.record_error("generate")
        meter.snapshot(reset=True)
        meter.record_error("extract")
        meter.snapshot(reset=True)
        meter.record_error("extract")

        since = meter.since(mark)

        assert (since.calls, since.errors) == (3, 3)
        assert {k: v.errors for k, v in since.by_kind.items()} == {
            "generate": 1,
            "extract": 2,
        }
        assert (
            meter.snapshot().calls == 1
        )  # the view is only the calls since the last reset

    def test_a_marker_is_per_interface_instance(self):
        first, second = _llm(), _llm()
        with patch(COMPLETION, lambda **kw: _reply()):
            first.generate_response(_request())
            second.generate_response(_request())
        mark_second = ledger_mark(second)

        first.reset_usage()
        with patch(COMPLETION, lambda **kw: _reply()):
            second.generate_response(_request())

        assert first.usage().calls == 0
        assert second.usage().calls == 2  # untouched by the reset of the other
        assert ledger_since(second, mark_second).calls == 1  # type: ignore[arg-type]
        assert ledger_since(first, ledger_mark(first)).calls == 0  # type: ignore[arg-type]

    def test_snapshot_with_reset_is_atomic_and_loses_nothing_under_8_threads(self):
        meter = _UsageMeter()
        origin = meter.mark()
        done = threading.Event()
        harvested: list[int] = []

        def reader() -> None:
            while not done.is_set():
                harvested.append(meter.snapshot(reset=True).calls)

        def worker() -> None:
            for _ in range(500):
                meter.record("generate", _reply())

        watcher = threading.Thread(target=reader)
        watcher.start()
        workers = [threading.Thread(target=worker) for _ in range(8)]
        for thread in workers:
            thread.start()
        for thread in workers:
            thread.join()
        done.set()
        watcher.join()

        assert sum(harvested) + meter.snapshot().calls == 4000
        assert meter.since(origin).calls == 4000  # and the history lost nothing either
        assert meter.since(origin).prompt_tokens == 4000 * 11

    def test_the_ledger_is_exact_for_a_hundred_thousand_calls_in_constant_state(self):
        meter = _UsageMeter()
        for index in range(100_000):
            if index % 4 == 0:
                meter.record_error("generate")
            else:
                meter.record("generate", _reply())

        usage = meter.snapshot()

        assert (usage.calls, usage.errors) == (100_000, 25_000)
        assert usage.prompt_tokens == 75_000 * 11
        assert set(meter._counts) == {
            "generate"
        }  # one dict of six counters, not a list
        assert len(meter._counts["generate"]) == 6


class TestOneRecordShapeAndOneFold:
    def test_the_fold_of_records_equals_a_meter_fed_the_same_calls(self):
        meter = _UsageMeter()
        meter.record("generate", _reply(prompt=7, completion=2))
        meter.record("stream", None)
        meter.record_error("extract")
        meter.record("generate", _reply(prompt=None))  # a reply with no usage

        folded = fold(
            [
                UsageRecord("generate", True, (7, 2, 9)),
                UsageRecord("stream", True, None),
                UsageRecord("extract", False),
                UsageRecord("generate", True, None),
            ]
        )

        assert folded == meter.snapshot()
        assert (folded.calls, folded.errors, folded.usage_missing) == (4, 1, 2)
        assert [k for k in folded.by_kind] == ["extract", "generate", "stream"]

    def test_an_empty_fold_is_no_usage(self):
        assert fold([]) == LLMUsage()

    @pytest.mark.parametrize(
        ("call_type", "stream", "kind"),
        [
            ("response_generation", False, "generate"),
            ("field_extraction", False, "extract"),
            ("data_extraction", False, "extract"),
            ("classification", False, "classify"),
            ("anything else", False, "complete"),
            ("response_generation", True, "stream"),
            ("", True, "stream"),
        ],
    )
    def test_the_kind_of_a_call_is_one_rule(self, call_type, stream, kind):
        assert usage_kind(call_type, stream) == kind

    def test_the_token_reader_is_the_one_the_meter_uses(self):
        assert usage_counts(
            {"usage": {"prompt_tokens": 3, "completion_tokens": 2}}
        ) == (
            3,
            2,
            5,
        )
        assert usage_counts({"usage": {"prompt_tokens": True}}) == (0, 0, 0)
        assert usage_counts({}) is None
        assert not hasattr(usage_module, "_usage_of")


class TestAnObjectWithoutALedger:
    def test_only_an_interface_with_the_metered_usage_has_one(self):
        class Own(LiteLLMInterface):
            def usage(self) -> LLMUsage:
                return LLMUsage(calls=9)

        assert ledger_mark(_llm()) == {}
        assert ledger_mark(ScriptedLLM([])) == {}
        assert ledger_mark(LiteLLMEmbedder(model="text-embedding-3-small")) == {}
        for other in (
            Own(model="gpt-4o"),
            object(),
            None,
            _Wrapper(),
        ):
            assert ledger_mark(other) is None
            with pytest.raises(ValueError, match="no usage ledger"):
                ledger_since(other, {})

    def test_a_default_usage_was_not_added_to_the_interface(self):
        # D-014: a default `LLMInterface.usage()` would make every double "metered" with zeros
        assert not hasattr(LLMInterface, "usage")


def _read_a_stream_that_is_cut(log: EventLog) -> LiteLLMInterface:
    """Read one streamed reply that raises half way, as the engine does (``stream_done`` in a
    ``finally``), through the event scope of ROOT."""

    def completion(**kwargs: Any) -> Any:
        def chunks() -> Any:
            yield _chunk("he")
            raise RuntimeError("cut")

        return chunks()

    llm, parts = _llm(), []
    with patch(COMPLETION, completion), bind(log, ROOT):
        with pytest.raises(LLMResponseError):
            with llm_call(llm, "response_stream", _request(), stream=True) as call:
                try:
                    for chunk in call.stream(llm.generate_response_stream(_request())):
                        parts.append(chunk)
                finally:
                    call.stream_done("".join(parts))
    return llm


class TestTheEventsWriteTokensInline:
    def test_a_provider_reply_with_usage_writes_its_three_integers_and_its_model(
        self, log
    ):
        with patch(COMPLETION, lambda **kw: _reply(prompt=11, completion=4)):
            with bind(log, ROOT):
                _llm().generate_response(_request())

        (response,) = _events(log, "llm/response")

        assert (
            response["prompt_tokens"],
            response["completion_tokens"],
            response["total_tokens"],
        ) == (11, 4, 15)
        assert response["model"] == "gpt-4o"
        assert all(type(response[k]) is int for k in response if k.endswith("_tokens"))

    def test_a_reply_with_no_usage_writes_no_token_field(self, log):
        with patch(COMPLETION, lambda **kw: _reply(prompt=None)):
            with bind(log, ROOT):
                _llm().generate_response(_request())

        (response,) = _events(log, "llm/response")

        assert not [k for k in response if k.endswith("_tokens")]

    def test_an_engine_logged_reply_and_a_stream_carry_no_token_field(self, log):
        llm = ScriptedLLM([], default_response="hello", stream_chunks=["he", "llo"])
        with bind(log, ROOT):
            with llm_call(llm, "response_generation", _request()) as call:
                call.response(llm.generate_response(_request()))
            with llm_call(llm, "response_stream", _request(), stream=True) as call:
                call.stream_done("".join(call.stream(["he", "llo"])))

        responses = _events(log, "llm/response")

        assert len(responses) == 2
        assert not [k for r in responses for k in r if k.endswith("_tokens")]

    def test_a_stream_that_errors_marks_its_error_as_a_stream_and_writes_no_response(
        self, log
    ):
        _read_a_stream_that_is_cut(log)

        (error,) = _events(log, "provider/error")
        assert error["stream"] is True and not _events(log, "llm/response")

    def test_a_failed_call_that_is_not_a_stream_has_no_stream_field(self, log):
        def completion(**kwargs: Any) -> Any:
            raise RuntimeError("down")

        with patch(COMPLETION, completion), bind(log, ROOT):
            with pytest.raises(LLMResponseError):
                _llm().generate_response(_request())

        (error,) = _events(log, "provider/error")
        assert "stream" not in error


class TestTheProjectionOfEvents:
    def _event(self, kind: str, **fields: Any) -> dict[str, Any]:
        return {"type": kind, "call_type": "response_generation", **fields}

    def test_it_folds_responses_and_errors_and_skips_every_other_event(self):
        events = [
            {"type": "llm/request", "call_type": "response_generation"},
            self._event(
                "llm/response", prompt_tokens=5, completion_tokens=2, total_tokens=7
            ),
            self._event("provider/error", error_type="Down"),
            {"type": "turn/start"},
            self._event("llm/response", unrecorded=True),
            self._event("llm/response", call_type="field_extraction", prompt_tokens=1),
        ]

        got = projection().usage_from_events(events)

        assert (got.calls, got.errors, got.usage_missing) == (4, 1, 2)
        assert (got.prompt_tokens, got.completion_tokens, got.total_tokens) == (5, 2, 7)
        assert {k: v.calls for k, v in got.by_kind.items()} == {
            "generate": 3,
            "extract": 1,
        }

    @pytest.mark.parametrize(
        "tokens",
        [
            {"prompt_tokens": True, "completion_tokens": 1, "total_tokens": 2},
            {"prompt_tokens": -1, "completion_tokens": 1, "total_tokens": 0},
            {"prompt_tokens": "5", "completion_tokens": 1, "total_tokens": 6},
            {"prompt_tokens": 5, "completion_tokens": 1},
            {},
        ],
    )
    def test_a_response_without_three_clean_integers_is_usage_missing(self, tokens):
        got = projection().usage_from_events([self._event("llm/response", **tokens)])

        assert (got.calls, got.usage_missing, got.total_tokens) == (1, 1, 0)

    def test_a_stream_flag_makes_the_kind_stream_for_a_response_and_an_error(self):
        events = [
            self._event("llm/response", stream=True),
            self._event("provider/error", stream=True),
            self._event("provider/error", stream=False),
        ]

        got = projection().usage_from_events(events)

        assert {k: (v.calls, v.errors) for k, v in got.by_kind.items()} == {
            "stream": (2, 1),
            "generate": (1, 1),
        }

    def test_it_never_reads_a_blob_so_a_missing_one_changes_nothing(self, log):
        with patch(COMPLETION, lambda **kw: _reply(prompt=3, completion=1)):
            with bind(log, ROOT):
                _llm().generate_response(_request())
        for path in (log.root / "blobs").rglob("*"):
            if path.is_file():
                path.unlink()

        got = projection().usage_from_events(log.read(ROOT))

        assert (got.calls, got.total_tokens) == (1, 4)


def _provider(state: dict[str, Any]) -> Any:
    def completion(**kwargs: Any) -> Any:
        if state.get("down"):
            raise RuntimeError("provider down")
        if kwargs.get("stream"):
            return iter([_chunk("wo"), _chunk("rk")])
        return _reply("working", prompt=9, completion=3)

    return completion


class TestTheProjectionEqualsTheMeter:
    def test_for_an_observed_interface_over_a_mixed_scenario(self, log):
        llm, state = _llm(), {}
        api = API(copy.deepcopy(_CHAT), llm_interface=llm, event_log=log)
        with patch(COMPLETION, _provider(state)):
            # the root greeting runs before any scope can name its conversation (D-021 of
            # plan 21e6f00d): the one call the log lacks, so the meter is read from after it
            cid, _ = api.start_conversation()
            mark = ledger_mark(llm)
            api.converse("one", cid)
            state["down"] = True
            with pytest.raises(FSMError):
                api.converse("two", cid)  # a provider error, and the turn rolls back
            state["down"] = False
            api.converse("three", cid)  # its retry
            assert "".join(api.converse_stream("four", cid)) == "work"

        meter = ledger_since(llm, mark)  # type: ignore[arg-type]
        folded = projection().usage_from_events(log.read(cid))

        assert _dump(folded) == _dump(meter)
        assert folded.errors >= 1 and folded.usage_missing >= 1  # not vacuous
        assert folded.total_tokens > 0 and "stream" in folded.by_kind

    def test_an_engine_logged_double_carries_usage_missing_like_scripted_llm(self, log):
        llm = _script()
        api = API(copy.deepcopy(_FSM), llm_interface=llm, event_log=log)
        cid, _ = api.start_conversation()
        mark = ledger_mark(llm)
        api.converse("Ann", cid)
        api.advance(cid)

        meter = ledger_since(llm, mark)  # type: ignore[arg-type]
        folded = projection().usage_from_events(log.read(cid))

        assert _dump(folded) == _dump(meter)
        assert folded.calls > 3 and folded.usage_missing == folded.calls
        assert folded.total_tokens == 0

    def test_a_wrapper_around_an_observed_interface_is_counted_once(self, log):
        inner = _llm()
        api = API(copy.deepcopy(_CHAT), llm_interface=_Wrapper(inner), event_log=log)
        with patch(COMPLETION, _provider({})):
            cid, _ = api.start_conversation()
            mark = ledger_mark(inner)
            api.converse("one", cid)

        folded = projection().usage_from_events(log.read(cid))

        assert folded.calls == ledger_since(inner, mark).calls == 1  # type: ignore[arg-type]

    def test_a_stream_that_errors_mid_read_is_one_error_where_the_meter_saw_a_call(
        self, log
    ):
        llm = _read_a_stream_that_is_cut(log)

        folded = projection().usage_from_events(log.read(ROOT))

        assert (llm.usage().calls, llm.usage().errors) == (1, 0)  # counted at the send
        assert (folded.calls, folded.errors) == (1, 1)  # one error, never two calls
        assert set(folded.by_kind) == {"stream"}

    def test_the_log_off_path_changes_nothing_in_usage(self, tmp_path):
        def run(log: EventLog | None) -> dict[str, Any]:
            llm, state = _llm(), {}
            api = API(copy.deepcopy(_CHAT), llm_interface=llm, event_log=log)
            with patch(COMPLETION, _provider(state)):
                cid, _ = api.start_conversation()
                api.converse("one", cid)
                state["down"] = True
                with pytest.raises(FSMError):
                    api.converse("two", cid)
                state["down"] = False
                "".join(api.converse_stream("three", cid))
            return _dump(llm.usage())

        with EventLog(tmp_path / "events") as opened:
            assert run(None) == run(opened)


class TestPositionsInTheLog:
    def _log_with_errors(self, log: EventLog) -> None:
        with bind(log, ROOT, 0):
            with patch(COMPLETION, lambda **kw: _reply()):
                _llm().generate_response(_request())
        for frame in (1, 1, 0, 2):
            with (
                bind(log, ROOT, frame),
                patch(COMPLETION, side_effect=RuntimeError("down")),
                pytest.raises(LLMResponseError),
            ):
                _llm().generate_response(_request())

    def test_log_position_is_the_last_seq_and_zero_for_nothing(self, log):
        assert projection().log_position(log, ROOT) == 0
        self._log_with_errors(log)

        assert projection().log_position(log, ROOT) == max(
            e["seq"] for e in log.read(ROOT)
        )

    def test_errors_since_counts_one_frame_after_a_seq_and_nothing_else(self, log):
        self._log_with_errors(log)
        errors = [e for e in log.read(ROOT) if e["type"] == "provider/error"]
        first_one = next(e["seq"] for e in errors if e["frame"] == 1)

        since = projection().provider_errors_since
        assert since(log, ROOT, 1, 0) == 2
        assert since(log, ROOT, 1, first_one) == 1  # strictly after the position
        assert since(log, ROOT, 0, 0) == 1 and since(log, ROOT, 2, 0) == 1
        assert since(log, ROOT, 3, 0) == 0


class _Duck:
    """An interface with its own ``usage()`` and no ledger (the doubles of the repository)."""

    def __init__(self) -> None:
        self.errors = 0

    def usage(self) -> LLMUsage:
        return LLMUsage(calls=self.errors, errors=self.errors)


class TestTheOutageReader:
    def test_the_ledger_answers_with_no_log_and_survives_a_reset(self):
        llm = _llm()
        with patch(COMPLETION, side_effect=RuntimeError("down")):
            with pytest.raises(LLMResponseError):
                llm.generate_response(_request())
            read = projection().watch_provider_errors(llm, None, ROOT, 0)
            with pytest.raises(LLMResponseError):
                llm.generate_response(_request())
            llm.reset_usage()
            with pytest.raises(LLMResponseError):
                llm.generate_response(_request())

        assert (
            read() == 2
        )  # the one before the position is not counted, the reset is no loss

    def test_an_interface_with_its_own_usage_is_read_before_and_after(self):
        duck = _Duck()
        duck.errors = 5
        read = projection().watch_provider_errors(duck, None, ROOT, 0)
        assert read() == 0
        duck.errors = 7
        assert read() == 2
        duck.errors = 1  # a counter that went down is never a negative outage
        assert read() == 0

    def test_an_interface_with_no_meter_and_no_log_reads_zero(self):
        read = projection().watch_provider_errors(object(), None, ROOT, 0)

        assert read() == 0

    def test_the_log_answers_for_an_interface_with_no_meter(self, log):
        read = projection().watch_provider_errors(object(), log, ROOT, 1)
        for frame in (1, 0, 1):
            with (
                bind(log, ROOT, frame),
                patch(COMPLETION, side_effect=RuntimeError("down")),
                pytest.raises(LLMResponseError),
            ):
                _llm().generate_response(_request())

        assert read() == 2

    def test_the_log_wins_and_is_never_added_to_the_ledger(self, log):
        llm = _llm()
        read = projection().watch_provider_errors(llm, log, ROOT, 0)
        with patch(COMPLETION, side_effect=RuntimeError("down")), bind(log, ROOT, 0):
            with pytest.raises(LLMResponseError):
                llm.generate_response(_request())

        assert llm.usage().errors == 1  # both sources saw it
        assert read() == 1  # and one source answers: 1, never 2

    def test_a_source_that_fails_falls_back_one_level(self, log):
        llm = _llm()
        read = projection().watch_provider_errors(llm, log, ROOT, 0)
        with patch(COMPLETION, side_effect=RuntimeError("down")):
            with pytest.raises(LLMResponseError):
                llm.generate_response(_request())  # no scope: only the ledger saw it
        with patch.object(log, "query", side_effect=OSError("gone")):
            assert read() == 1

    def test_a_log_that_cannot_be_read_at_the_push_is_skipped(self, log):
        llm = _llm()
        with patch.object(log, "read", side_effect=OSError("gone")):
            read = projection().watch_provider_errors(llm, log, ROOT, 0)
        with patch(COMPLETION, side_effect=RuntimeError("down")):
            with pytest.raises(LLMResponseError):
                llm.generate_response(_request())

        assert read() == 1

    def test_the_position_is_taken_when_the_reader_is_made(self, log):
        llm = _llm()
        with patch(COMPLETION, side_effect=RuntimeError("down")), bind(log, ROOT, 0):
            with pytest.raises(LLMResponseError):
                llm.generate_response(_request())
            read = projection().watch_provider_errors(llm, log, ROOT, 0)

        assert read() == 0


class TestTheSupersededAnchor:
    def test_the_old_anchor_still_parses_and_the_new_one_sits_at_the_ledger(self):
        spec = importlib.util.spec_from_file_location(
            "_decision_index_for_usage", REPO / "scripts" / "decision_index.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
            found = {
                (a.path, a.plan_id, a.number)
                for a in module.scan(REPO)
                if a.path.startswith("src/fsm_llm/llm/")
            }
        finally:
            sys.modules.pop(spec.name, None)

        # plan 3f4530d3, step 1a: the anchor moved with `_begin_send` to `providers/send.py`
        old = (
            "src/fsm_llm/llm/providers/send.py",
            "plan-2026-10-01T093600-944e2692",
            4,
        )
        new = ("src/fsm_llm/llm/usage.py", "plan-2026-11-18T071726-5a6a1f80", 14)
        assert old in found and new in found
