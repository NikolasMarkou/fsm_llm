"""Contract of `ScriptedLLM`, the queue-driven `LLMInterface` double.

A test scripts the `complete` replies in order (text, `CompletionResponse` or an
exception to raise), the field answers (`extraction_data`) and the Pass-2 reply
per state (`responses`). A call nothing scripted is recorded in `unexpected`,
raises `UnscriptedCallError` and is counted as an error; because the pipeline
swallows a raised extraction error, `assert_exhausted()` is what fails the test.
Expected status and kind values come from `fsm_llm.codes`.
"""

from __future__ import annotations

import ast
import inspect
import json
import subprocess
import sys
import textwrap

import pytest

from fsm_llm import API
from fsm_llm.codes import CompletionKind
from fsm_llm.core.definitions import (
    BulkExtractionRequest,
    CompletionRequest,
    CompletionResponse,
    FieldExtractionRequest,
    LLMResponseError,
    LLMUsage,
    ResponseGenerationRequest,
)
from fsm_llm.llm import LLMInterface, implements_complete
from fsm_llm.testing import (
    ScriptedLLM,
    UnscriptedCallError,
    final_reply,
    malformed_reply,
    tool_call_reply,
)

_TAG = "<current_state>{}</current_state>"


def _complete(call_type: str = "completion") -> CompletionRequest:
    return CompletionRequest(
        messages=[{"role": "user", "content": "hi"}], call_type=call_type
    )


def _bulk() -> BulkExtractionRequest:
    return BulkExtractionRequest(system_prompt="s", user_message="hi")


def _field(name: str = "city") -> FieldExtractionRequest:
    return FieldExtractionRequest.model_validate(
        {"system_prompt": "extract", "user_message": "Rome", "field_name": name}
    )


def _reply(state: str | None = None) -> ResponseGenerationRequest:
    prompt = "reply" if state is None else f"x {_TAG.format(state)} y"
    return ResponseGenerationRequest(system_prompt=prompt, user_message="hi")


def _fsm(*, extract: bool = True) -> dict:
    start: dict = {
        "id": "start",
        "description": "ask",
        "purpose": "collect the city",
        "response_instructions": "Ask for the city",
        "transitions": [
            {
                "target_state": "done",
                "description": "city known",
                "priority": 100,
                "conditions": [
                    {"description": "city", "requires_context_keys": ["city"]}
                ],
            }
        ],
    }
    if extract:
        start["required_context_keys"] = ["city"]
    return {
        "name": "scripted_probe",
        "description": "start collects a city, done ends",
        "initial_state": "start",
        "states": {
            "start": start,
            "done": {
                "id": "done",
                "description": "end",
                "purpose": "wrap up",
                "response_instructions": "Say goodbye",
            },
        },
    }


class TestShape:
    def test_is_an_llm_interface_that_completes(self) -> None:
        llm = ScriptedLLM()
        assert isinstance(llm, LLMInterface)
        assert implements_complete(llm)

    def test_defaults(self) -> None:
        llm = ScriptedLLM()
        assert llm.replies == []
        assert llm.requests == []
        assert llm.call_history == []
        assert llm.unexpected == []
        assert llm.model is None
        assert llm.extraction_data is None
        assert llm.responses == {}

    def test_model_is_settable_and_constructor_keywords_only_after_replies(
        self,
    ) -> None:
        assert ScriptedLLM(model="fake/one").model == "fake/one"
        with pytest.raises(TypeError):
            ScriptedLLM((), {"a": 1})  # type: ignore[misc]

    def test_constructor_copies_its_arguments(self) -> None:
        replies = ["a"]
        data = {"city": "Rome"}
        responses = {"start": "hi"}
        llm = ScriptedLLM(replies, extraction_data=data, responses=responses)
        replies.append("b")
        data["city"] = "Oslo"
        responses["start"] = "changed"
        assert llm.replies == ["a"]
        assert llm.extraction_data == {"city": "Rome"}
        assert llm.responses == {"start": "hi"}

    def test_a_reply_of_another_type_is_refused_at_construction(self) -> None:
        with pytest.raises(TypeError, match="reply"):
            ScriptedLLM([42])  # type: ignore[list-item]

    def test_an_exception_class_is_refused_with_the_fix(self) -> None:
        with pytest.raises(TypeError, match="instance"):
            ScriptedLLM([RuntimeError])  # type: ignore[list-item]


class TestComplete:
    def test_replies_are_returned_in_order(self) -> None:
        reply_b = tool_call_reply(("lookup", {"q": 1}))
        llm = ScriptedLLM(["first", reply_b, malformed_reply()])
        one, two, three = (llm.complete(_complete()) for _ in range(3))
        assert (one.kind, one.text) == (CompletionKind.FINAL, "first")
        assert two is reply_b
        assert three.kind == CompletionKind.MALFORMED
        assert llm.replies == []

    def test_a_text_item_is_a_final_reply(self) -> None:
        assert ScriptedLLM(["x"]).complete(_complete()) == final_reply("x")

    def test_an_exception_item_is_raised(self) -> None:
        boom = LLMResponseError("provider down")
        llm = ScriptedLLM([boom, "after"])
        with pytest.raises(LLMResponseError) as seen:
            llm.complete(_complete())
        assert seen.value is boom
        assert llm.complete(_complete()).text == "after"

    def test_requests_hold_the_completion_requests_only(self) -> None:
        llm = ScriptedLLM(["a"], extraction_data={}, responses={"s": "r"})
        request = _complete()
        llm.complete(request)
        llm.extract_field(_field())
        llm.generate_response(_reply("s"))
        assert llm.requests == [request]
        assert [name for name, _ in llm.call_history] == [
            "complete",
            "extract_field",
            "generate_response",
        ]
        assert llm.calls("complete") == [request]
        assert llm.calls("generate_response") == [llm.call_history[2][1]]

    def test_a_reply_appended_later_is_used(self) -> None:
        llm = ScriptedLLM()
        llm.replies.append("late")
        assert llm.complete(_complete()).text == "late"

    def test_a_late_reply_of_another_type_is_refused_when_taken(self) -> None:
        llm = ScriptedLLM()
        llm.replies.append(7)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="reply"):
            llm.complete(_complete())


class TestUnscriptedCalls:
    def test_an_empty_queue_is_recorded_raised_and_clear(self) -> None:
        llm = ScriptedLLM()
        request = _complete()
        with pytest.raises(UnscriptedCallError, match="complete") as seen:
            llm.complete(request)
        assert "nothing scripted" in str(seen.value)
        assert llm.unexpected == [("complete", request)]

    def test_the_error_is_an_assertion_error_so_an_unswallowed_one_fails_the_test(
        self,
    ) -> None:
        assert issubclass(UnscriptedCallError, AssertionError)

    def test_field_extraction_without_extraction_data_is_unscripted(self) -> None:
        llm = ScriptedLLM()
        with pytest.raises(UnscriptedCallError):
            llm.extract_field(_field())
        with pytest.raises(UnscriptedCallError):
            llm.extract_bulk_data(_bulk())
        assert [name for name, _ in llm.unexpected] == [
            "extract_field",
            "extract_bulk_data",
        ]

    def test_a_reply_for_a_state_nothing_names_is_unscripted(self) -> None:
        llm = ScriptedLLM(responses={"a": "x"})
        with pytest.raises(UnscriptedCallError):
            llm.generate_response(_reply("b"))
        with pytest.raises(UnscriptedCallError):
            llm.generate_response(_reply())
        assert len(llm.unexpected) == 2

    def test_assert_exhausted_fails_on_an_unexpected_call_and_says_which(self) -> None:
        llm = ScriptedLLM()
        with pytest.raises(UnscriptedCallError):
            llm.complete(_complete())
        with pytest.raises(AssertionError, match="unscripted") as seen:
            llm.assert_exhausted()
        assert "complete" in str(seen.value)

    def test_assert_exhausted_fails_on_a_reply_never_used(self) -> None:
        llm = ScriptedLLM(["used", "left over"])
        llm.complete(_complete())
        with pytest.raises(AssertionError, match="1 scripted") as seen:
            llm.assert_exhausted()
        assert "left over" in str(seen.value)

    def test_assert_exhausted_passes_when_everything_was_used(self) -> None:
        llm = ScriptedLLM(["a", "b"])
        llm.complete(_complete())
        llm.complete(_complete())
        llm.assert_exhausted()

    def test_assert_exhausted_passes_on_a_fresh_double(self) -> None:
        ScriptedLLM().assert_exhausted()

    def test_assert_exhausted_is_not_an_assert_statement(self) -> None:
        tree = ast.parse(
            textwrap.dedent(inspect.getsource(ScriptedLLM.assert_exhausted))
        )
        assert not [n for n in ast.walk(tree) if isinstance(n, ast.Assert)]
        code = (
            "from fsm_llm.testing import ScriptedLLM\n"
            "llm = ScriptedLLM(['unused'])\n"
            "try:\n"
            "    llm.assert_exhausted()\n"
            "except AssertionError:\n"
            "    print('raised')\n"
        )
        out = subprocess.run(
            [sys.executable, "-O", "-c", code],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        assert out.stdout.strip() == "raised"


class TestPipelineSeesUnexpectedCalls:
    """The extraction `except Exception` swallows a raised error: the double must not."""

    def test_a_swallowed_unscripted_extraction_is_still_detectable(self) -> None:
        llm = ScriptedLLM(responses={"start": "hi", "done": "bye"})
        api = API(fsm_definition=_fsm(), llm_interface=llm)
        conversation_id, _ = api.start_conversation()
        api.converse("Rome", conversation_id)  # does not raise: the pipeline swallows
        assert any(name == "extract_field" for name, _ in llm.unexpected)
        assert api.get_current_state(conversation_id) == "start"  # nothing extracted
        with pytest.raises(AssertionError, match="unscripted"):
            llm.assert_exhausted()

    def test_a_fully_scripted_turn_leaves_nothing_unexpected(self) -> None:
        llm = ScriptedLLM(
            extraction_data={"city": "Rome"}, responses={"start": "hi", "done": "bye"}
        )
        api = API(fsm_definition=_fsm(), llm_interface=llm)
        conversation_id, greeting = api.start_conversation()
        assert greeting == "hi"
        assert api.converse("Rome", conversation_id) == "bye"
        assert api.get_current_state(conversation_id) == "done"
        assert llm.unexpected == []
        llm.assert_exhausted()


class TestExtraction:
    def test_extract_field_answers_from_extraction_data(self) -> None:
        llm = ScriptedLLM(extraction_data={"city": "Rome"})
        got = llm.extract_field(_field("city"))
        assert (got.value, got.confidence, got.is_valid) == ("Rome", 1.0, True)

    def test_a_name_not_in_extraction_data_is_not_extracted_and_not_unexpected(
        self,
    ) -> None:
        llm = ScriptedLLM(extraction_data={"city": "Rome"})
        got = llm.extract_field(_field("age"))
        assert (got.value, got.confidence, got.is_valid) == (None, 0.0, False)
        assert llm.unexpected == []

    def test_empty_extraction_data_is_scripted_as_nothing_to_extract(self) -> None:
        llm = ScriptedLLM(extraction_data={})
        assert llm.extract_field(_field()).value is None
        assert llm.unexpected == []

    def test_extraction_data_is_mutable_between_turns(self) -> None:
        llm = ScriptedLLM(extraction_data={})
        llm.extraction_data = {"city": "Oslo"}
        assert llm.extract_field(_field()).value == "Oslo"

    def test_bulk_extraction_returns_a_copy_of_extraction_data(self) -> None:
        llm = ScriptedLLM(extraction_data={"city": "Rome", "age": 3})
        got = llm.extract_bulk_data(_bulk())
        assert got.extracted_data == {"city": "Rome", "age": 3}
        got.extracted_data["city"] = "changed"
        assert llm.extraction_data == {"city": "Rome", "age": 3}


class TestGenerateResponse:
    def test_reply_by_state_in_the_prompt(self) -> None:
        llm = ScriptedLLM(responses={"a": "text a", "b": "text b"})
        assert llm.generate_response(_reply("b")).message == "text b"
        assert llm.generate_response(_reply("a")).message == "text a"

    def test_default_response_covers_other_states(self) -> None:
        llm = ScriptedLLM(responses={"a": "text a"}, default_response="fallback")
        assert llm.generate_response(_reply("zzz")).message == "fallback"
        assert llm.generate_response(_reply()).message == "fallback"
        assert llm.unexpected == []

    def test_reply_is_a_plain_response_type_not_truncated(self) -> None:
        got = ScriptedLLM(default_response="x").generate_response(_reply())
        assert (got.message, got.message_type, got.truncated) == (
            "x",
            "response",
            False,
        )


class TestStream:
    def test_default_streams_the_scripted_reply_as_one_chunk(self) -> None:
        llm = ScriptedLLM(responses={"a": "whole"})
        assert list(llm.generate_response_stream(_reply("a"))) == ["whole"]

    def test_scripted_chunks_are_yielded_in_order(self) -> None:
        llm = ScriptedLLM(stream_chunks=("Hel", "lo ", "there"))
        assert list(llm.generate_response_stream(_reply())) == ["Hel", "lo ", "there"]
        assert list(llm.generate_response_stream(_reply())) == ["Hel", "lo ", "there"]

    def test_stream_is_recorded_and_metered_as_stream_not_generate(self) -> None:
        llm = ScriptedLLM(stream_chunks=("a", "b"))
        list(llm.generate_response_stream(_reply()))
        usage = llm.usage()
        assert set(usage.by_kind) == {"stream"}
        assert usage.by_kind["stream"].calls == 1
        assert usage.by_kind["stream"].usage_missing == 1
        assert [name for name, _ in llm.call_history] == ["generate_response_stream"]

    def test_an_unscripted_stream_is_recorded_and_raises_when_iterated(self) -> None:
        llm = ScriptedLLM()
        with pytest.raises(UnscriptedCallError):
            list(llm.generate_response_stream(_reply()))
        assert [name for name, _ in llm.unexpected] == ["generate_response_stream"]

    def test_a_streamed_converse_reaches_the_user_in_chunks(self) -> None:
        llm = ScriptedLLM(
            extraction_data={"city": "Rome"},
            responses={"start": "hi"},
            stream_chunks=("by", "e"),
        )
        api = API(fsm_definition=_fsm(), llm_interface=llm)
        conversation_id, _ = api.start_conversation()
        assert "".join(api.converse_stream("Rome", conversation_id)) == "bye"


class TestUsage:
    def test_a_fresh_double_has_no_usage(self) -> None:
        assert ScriptedLLM().usage() == LLMUsage()

    def test_every_answered_call_counts_under_its_kind(self) -> None:
        llm = ScriptedLLM(
            ["a", "b", "c"], extraction_data={"city": "x"}, default_response="r"
        )
        llm.complete(_complete())
        llm.complete(_complete("classification"))
        llm.complete(_complete("custom_call"))
        llm.extract_field(_field())
        llm.extract_bulk_data(_bulk())
        llm.generate_response(_reply())
        usage = llm.usage()
        assert usage.calls == 6
        assert usage.errors == 0
        assert {k: v.calls for k, v in usage.by_kind.items()} == {
            "complete": 2,
            "classify": 1,
            "extract": 2,
            "generate": 1,
        }
        assert sum(v.calls for v in usage.by_kind.values()) == usage.calls

    def test_a_raising_item_counts_an_error(self) -> None:
        llm = ScriptedLLM([LLMResponseError("down"), "ok"])
        with pytest.raises(LLMResponseError):
            llm.complete(_complete())
        llm.complete(_complete())
        usage = llm.usage()
        assert (usage.calls, usage.errors) == (2, 1)
        assert usage.by_kind["complete"].errors == 1

    def test_an_unscripted_call_counts_an_error_too(self) -> None:
        llm = ScriptedLLM()
        with pytest.raises(UnscriptedCallError):
            llm.complete(_complete("classification"))
        usage = llm.usage()
        assert (usage.calls, usage.errors) == (1, 1)
        assert usage.by_kind["classify"].errors == 1

    def test_usage_is_a_snapshot(self) -> None:
        llm = ScriptedLLM(["a", "b"])
        llm.complete(_complete())
        before = llm.usage()
        llm.complete(_complete())
        assert before.calls == 1
        assert llm.usage().calls == 2

    def test_reset_usage_returns_the_old_counts_and_clears(self) -> None:
        llm = ScriptedLLM(["a"])
        llm.complete(_complete())
        assert llm.reset_usage().calls == 1
        assert llm.usage() == LLMUsage()

    def test_the_synthesis_simulator_accepts_the_double(self) -> None:
        from fsm_llm.synthesis.simulation import Simulator

        Simulator(ScriptedLLM(["x"]))


class TestSubclassing:
    def test_a_subclass_can_override_and_call_super(self) -> None:
        class Counting(ScriptedLLM):
            def complete(self, request: CompletionRequest) -> CompletionResponse:
                self.seen = getattr(self, "seen", 0) + 1
                return super().complete(request)

        llm = Counting(["a"])
        assert llm.complete(_complete()).text == "a"
        assert llm.seen == 1
        assert json.dumps(llm.usage().calls) == "1"
