"""The turn's cut-off record (plan 0068b055, step 12.28, D-025).

A reply the provider cut off at ``max_tokens`` reaches the pipeline as
``truncated=True`` on the response (set by ``llm.py`` from the provider's
``finish_reason``). The pipeline writes ONE internal key,
``definitions.CONTEXT_KEY_CUT_OFF``, into the conversation context for the turn
that produced it: ``{"reply": True}`` and/or ``{"fields": [names]}``. The key is
removed at the start of every turn, is in no prompt, is not in ``API.get_data``
and is not in a saved session. Streamed replies are never flagged (a stream
carries no finish reason).
"""

from __future__ import annotations

from typing import Any

import pytest
from litellm.types.utils import Choices, Message, ModelResponse

from fsm_llm import API
from fsm_llm.definitions import (
    CONTEXT_KEY_CUT_OFF,
    FieldExtractionRequest,
    ResponseGenerationRequest,
)
from fsm_llm.llm import LiteLLMInterface
from tests.conftest import MockLLM2Interface


def _fsm() -> dict[str, Any]:
    """a --(mood, topic extracted)--> b --> c (terminal); a and b speak."""
    return {
        "name": "cut",
        "description": "cut-off record probe",
        "initial_state": "a",
        "persona": "p",
        "states": {
            "a": {
                "id": "a",
                "description": "a",
                "purpose": "ask",
                "response_instructions": "Reply about the mood.",
                "field_extractions": [
                    {
                        "field_name": "mood",
                        "field_type": "str",
                        "extraction_instructions": "The user's mood.",
                    },
                    {
                        "field_name": "topic",
                        "field_type": "str",
                        "extraction_instructions": "The topic.",
                    },
                ],
                "transitions": [
                    {"target_state": "b", "description": "go", "priority": 100}
                ],
            },
            "b": {
                "id": "b",
                "description": "b",
                "purpose": "middle",
                "response_instructions": "Reply in the middle.",
                "transitions": [
                    {"target_state": "c", "description": "go", "priority": 100}
                ],
            },
            "c": {
                "id": "c",
                "description": "c",
                "purpose": "end",
                "response_instructions": "Say goodbye.",
            },
        },
    }


def _api(llm: Any, **kw: Any) -> tuple[API, str]:
    api = API.from_definition(_fsm(), llm_interface=llm, **kw)
    conv_id, _ = api.start_conversation()
    return api, conv_id


def _raw(api: API, conv_id: str) -> dict[str, Any]:
    raw = api.fsm_manager.copy_raw_context_data(api.get_sub_conversation_id(conv_id))
    assert raw is not None
    return raw


_DATA = {"mood": "calm", "topic": "rain"}


class TestTurnRecord:
    def test_a_whole_turn_leaves_no_key(self):
        api, cid = _api(MockLLM2Interface(extraction_data=_DATA))
        api.converse("hi", cid)
        assert CONTEXT_KEY_CUT_OFF not in _raw(api, cid)

    def test_a_cut_reply_is_recorded(self):
        api, cid = _api(MockLLM2Interface(extraction_data=_DATA, cut_reply=True))
        api.converse("hi", cid)
        assert _raw(api, cid)[CONTEXT_KEY_CUT_OFF] == {"reply": True}

    def test_cut_fields_are_recorded_in_call_order(self):
        llm = MockLLM2Interface(extraction_data=_DATA, cut_fields=["topic", "mood"])
        api, cid = _api(llm)
        api.converse("hi", cid)
        assert _raw(api, cid)[CONTEXT_KEY_CUT_OFF] == {"fields": ["mood", "topic"]}

    def test_reply_and_field_share_the_one_key(self):
        llm = MockLLM2Interface(
            extraction_data=_DATA, cut_fields=["mood"], cut_reply=True
        )
        api, cid = _api(llm)
        api.converse("hi", cid)
        assert _raw(api, cid)[CONTEXT_KEY_CUT_OFF] == {
            "reply": True,
            "fields": ["mood"],
        }

    def test_the_next_whole_turn_clears_it(self):
        llm = MockLLM2Interface(extraction_data=_DATA, cut_reply=True)
        api, cid = _api(llm)
        api.converse("hi", cid)
        assert CONTEXT_KEY_CUT_OFF in _raw(api, cid)

        llm.cut_reply = False
        api.converse("again", cid)

        assert CONTEXT_KEY_CUT_OFF not in _raw(api, cid)

    def test_a_cut_turn_after_a_whole_one_sets_it_again(self):
        llm = MockLLM2Interface(extraction_data=_DATA)
        api, cid = _api(llm)
        api.converse("hi", cid)
        assert CONTEXT_KEY_CUT_OFF not in _raw(api, cid)

        llm.cut_reply = True
        api.converse("again", cid)

        assert _raw(api, cid)[CONTEXT_KEY_CUT_OFF] == {"reply": True}

    def test_a_silent_turn_after_a_cut_one_clears_it(self):
        fsm = _fsm()
        fsm["states"]["c"]["response_instructions"] = ""
        llm = MockLLM2Interface(extraction_data=_DATA, cut_reply=True)
        api = API.from_definition(fsm, llm_interface=llm)
        cid, _ = api.start_conversation()
        api.converse("hi", cid)  # a to b; b spoke and its reply was cut
        assert CONTEXT_KEY_CUT_OFF in _raw(api, cid)

        api.advance(cid)  # b to c; c is silent: no reply, so nothing can be cut

        assert CONTEXT_KEY_CUT_OFF not in _raw(api, cid)

    def test_a_message_free_step_records_it_too(self):
        llm = MockLLM2Interface(extraction_data=_DATA)
        api, cid = _api(llm)
        api.converse("hi", cid)
        llm.cut_reply = True

        api.advance(cid)

        assert _raw(api, cid)[CONTEXT_KEY_CUT_OFF] == {"reply": True}

    def test_a_cut_greeting_is_recorded_and_the_first_turn_clears_it(self):
        llm = MockLLM2Interface(extraction_data=_DATA, cut_reply=True)
        api = API.from_definition(_fsm(), llm_interface=llm)
        cid, _ = api.start_conversation()
        assert _raw(api, cid)[CONTEXT_KEY_CUT_OFF] == {"reply": True}

        llm.cut_reply = False
        api.converse("hi", cid)

        assert CONTEXT_KEY_CUT_OFF not in _raw(api, cid)

    def test_a_caller_seeded_key_is_gone_after_the_first_turn(self):
        llm = MockLLM2Interface(extraction_data=_DATA)
        api = API.from_definition(_fsm(), llm_interface=llm)
        cid, _ = api.start_conversation({CONTEXT_KEY_CUT_OFF: {"reply": True}})

        api.converse("hi", cid)

        assert CONTEXT_KEY_CUT_OFF not in _raw(api, cid)

    def test_a_failed_turn_restores_the_previous_record(self):
        llm = MockLLM2Interface(extraction_data=_DATA, cut_reply=True)
        api, cid = _api(llm)
        api.converse("hi", cid)
        before = _raw(api, cid)[CONTEXT_KEY_CUT_OFF]

        def boom(request):
            raise RuntimeError("provider down")

        llm.generate_response = boom  # type: ignore[method-assign]
        with pytest.raises(Exception, match="provider down"):
            api.converse("again", cid)

        assert _raw(api, cid)[CONTEXT_KEY_CUT_OFF] == before

    def test_a_streamed_reply_is_not_flagged(self):
        llm = MockLLM2Interface(extraction_data=_DATA, cut_reply=True)
        api, cid = _api(llm)

        "".join(api.converse_stream("hi", cid))

        assert CONTEXT_KEY_CUT_OFF not in _raw(api, cid)


class TestKeyStaysInternal:
    def test_get_data_does_not_show_it(self):
        api, cid = _api(MockLLM2Interface(extraction_data=_DATA, cut_reply=True))
        api.converse("hi", cid)
        assert CONTEXT_KEY_CUT_OFF in _raw(api, cid)
        assert CONTEXT_KEY_CUT_OFF not in api.get_data(cid)

    def test_no_prompt_shows_it(self):
        """Pass 1 records the cut of ``mood``; the ``topic`` extraction and the
        Pass-2 prompt are built after that, with the key in the context."""
        llm = MockLLM2Interface(extraction_data=_DATA, cut_fields=["mood"])
        api, cid = _api(llm)
        api.converse("hi", cid)
        assert CONTEXT_KEY_CUT_OFF in _raw(api, cid)

        prompts = [
            f"{r.system_prompt}\n{r.user_message}\n{getattr(r, 'context', '')}"
            for kind, r in llm.call_history
        ]
        # The greeting call came first; the turn made two extractions and Pass 2.
        assert [k for k, _ in llm.call_history][-3:] == [
            "extract_field",
            "extract_field",
            "generate_response",
        ]
        assert "rain" in prompts[-1]  # the context IS rendered, without the key
        assert not [p for p in prompts[-3:] if "cut_off" in p]

    def test_a_saved_session_does_not_carry_it(self, tmp_path):
        from fsm_llm import FileSessionStore

        store = FileSessionStore(str(tmp_path))
        llm = MockLLM2Interface(extraction_data=_DATA, cut_reply=True)
        api, cid = _api(llm, session_store=store)
        api.converse("hi", cid)
        assert CONTEXT_KEY_CUT_OFF in _raw(api, cid)

        api.save_session(cid)

        saved = "".join(p.read_text() for p in tmp_path.iterdir())
        assert "_cut_off" not in saved
        assert "mood" in saved  # the file does carry the turn's data


class TestCompletionStateRecordsTheCut:
    def test_a_cut_completion_turn_names_its_result_key(self):
        """The documented result shape ``{kind, text, calls}`` is unchanged."""
        from fsm_llm.definitions import CompletionResponse
        from tests.test_fsm_llm.test_completion_state import (
            _RESULT,
            _start,
            _to_call_model,
            _tool_fsm,
        )

        class _Cut(MockLLM2Interface):
            def complete(self, request):
                return CompletionResponse(
                    kind="final", text="half an ans", finish_reason="length"
                )

        api = API.from_definition(_tool_fsm(), llm_interface=_Cut())
        cid = _start(api)
        _to_call_model(api, cid)

        api.advance(cid)  # the call_model turn: one cut `complete` call

        assert _raw(api, cid)[CONTEXT_KEY_CUT_OFF] == {"fields": [_RESULT]}
        assert set(_raw(api, cid)[_RESULT]) == {"kind", "text", "calls"}

    def test_a_whole_completion_turn_records_nothing(self):
        from fsm_llm.definitions import CompletionResponse
        from tests.test_fsm_llm.test_completion_state import (
            _start,
            _to_call_model,
            _tool_fsm,
        )

        class _Whole(MockLLM2Interface):
            def complete(self, request):
                return CompletionResponse(
                    kind="final", text="whole", finish_reason="stop"
                )

        api = API.from_definition(_tool_fsm(), llm_interface=_Whole())
        cid = _start(api)
        _to_call_model(api, cid)
        api.advance(cid)

        assert CONTEXT_KEY_CUT_OFF not in _raw(api, cid)


class TestLlmLayerFlagsTheResponse:
    """``llm.py`` sets ``truncated`` from the provider's ``finish_reason``."""

    @staticmethod
    def _provider(content: str, finish: str | None):
        from unittest.mock import patch

        reply = ModelResponse(
            choices=[Choices(message=Message(content=content), finish_reason=finish)]
        )
        return (
            patch("fsm_llm.llm.completion", return_value=reply),
            patch("fsm_llm.llm.get_supported_openai_params", return_value=[]),
        )

    def _generate(self, content: str, finish: str | None):
        p1, p2 = self._provider(content, finish)
        with p1, p2:
            return LiteLLMInterface("gpt-4o", max_tokens=1000).generate_response(
                ResponseGenerationRequest(system_prompt="s", user_message="u")
            )

    def _field(self, content: str, finish: str | None):
        p1, p2 = self._provider(content, finish)
        with p1, p2:
            return LiteLLMInterface("gpt-4o", max_tokens=1000).extract_field(
                FieldExtractionRequest(
                    system_prompt="s",
                    user_message="u",
                    field_name="plan",
                    field_type="list",
                )
            )

    def test_a_cut_envelope_ships_the_apology_flagged(self):
        out = self._generate(
            '{ "message": "CONSOLIDATED ASSESSMENT. Recommend', "length"
        )
        assert "cut off" in out.message
        assert out.truncated is True

    def test_a_cut_prose_reply_ships_as_is_flagged(self):
        out = self._generate("The assessment covers three areas. First, the", "length")
        assert out.message.startswith("The assessment")
        assert out.truncated is True

    @pytest.mark.parametrize("finish", ["stop", None])
    def test_a_whole_reply_is_not_flagged(self, finish):
        assert self._generate("All done.", finish).truncated is False

    def test_a_cut_list_field_is_flagged(self):
        out = self._field('[{"tool_name": "search"}, {"tool', "length")
        assert out.is_valid is False
        assert out.truncated is True

    @pytest.mark.parametrize("finish", ["stop", None])
    def test_a_whole_field_is_not_flagged(self, finish):
        out = self._field('{"value": [1, 2]}', finish)
        assert out.value == [1, 2]
        assert out.truncated is False

    def test_the_apology_retry_keeps_the_flag_of_its_own_reply(self):
        """A generic apology (nothing usable, not cut) is retried once; the flag
        comes from the retried reply."""
        from unittest.mock import patch

        first = ModelResponse(
            choices=[
                Choices(
                    message=Message(content='{"message": ""}'), finish_reason="stop"
                )
            ]
        )
        second = ModelResponse(
            choices=[
                Choices(
                    message=Message(content="Hello there, friend"),
                    finish_reason="length",
                )
            ]
        )
        with (
            patch("fsm_llm.llm.completion", side_effect=[first, second]),
            patch("fsm_llm.llm.get_supported_openai_params", return_value=[]),
        ):
            out = LiteLLMInterface("gpt-4o", max_tokens=1000).generate_response(
                ResponseGenerationRequest(system_prompt="s", user_message="u")
            )
        assert out.message == "Hello there, friend"
        assert out.truncated is True
