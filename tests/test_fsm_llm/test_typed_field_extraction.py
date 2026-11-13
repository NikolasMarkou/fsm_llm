"""Core ``typed_field_extraction``: the typed per-field builder moved from agents.

The builder returns a raw ``field_extractions`` entry with a narrowed prompt
context; these tests pin its contract and that the narrowing holds through a
real ``API`` turn.
"""

from __future__ import annotations

import json
from typing import ClassVar, get_args
from unittest.mock import patch

import pytest

from fsm_llm import API, FieldExtractionConfig, typed_field_extraction
from fsm_llm.constants import EXTRACTION_ENVELOPE_KEYS, FIELD_PROMPT_CONTEXT_LABEL
from fsm_llm.core.definitions import TypedFieldType
from fsm_llm.testing import PromptGroundedLLM
from tests.test_fsm_llm.support import _color_fsm, _fake_response

_NOTE = "Trip note: the client flies to Lyon on Friday."
_HIDDEN = "UNLISTED-MARKER-4411"


class TestShape:
    def test_entry_is_a_valid_core_config(self):
        field = typed_field_extraction(
            "city",
            "str",
            "Name the destination city.",
            context_keys=["trip_note", "region"],
            required=False,
        )
        config = FieldExtractionConfig.model_validate(field)

        assert (config.field_name, config.field_type) == ("city", "str")
        assert config.required is False
        assert config.context_keys == ["trip_note", "region"]

    def test_instruction_text(self):
        field = typed_field_extraction(
            "city", "str", "Name the city.", context_keys=["trip_note", "region"]
        )

        assert field["extraction_instructions"] == (
            "Extract the 'city' field (str) from the task and the 'trip_note', "
            "'region' values in the 'Already extracted:' context. Name the city."
        )
        assert FIELD_PROMPT_CONTEXT_LABEL in field["extraction_instructions"]

    def test_required_defaults_to_true(self):
        field = typed_field_extraction("city", "str", "x", context_keys=["note"])
        assert field["required"] is True

    def test_context_keys_order_kept_duplicates_dropped(self):
        field = typed_field_extraction(
            "city", "str", "x", context_keys=("b", "a", "b", "c", "a")
        )
        assert field["context_keys"] == ["b", "a", "c"]

    @pytest.mark.parametrize("field_type", list(get_args(TypedFieldType)))
    def test_supported_types(self, field_type):
        field = typed_field_extraction("item", field_type, "x", context_keys=["n"])
        assert FieldExtractionConfig.model_validate(field).field_type == field_type

    def test_typed_field_type_is_the_agent_set(self):
        assert set(get_args(TypedFieldType)) == {"str", "float", "list", "bool", "any"}


class TestRefusals:
    @pytest.mark.parametrize("field_type", ["dict", "int", "string"])
    def test_other_types_rejected(self, field_type):
        with pytest.raises(ValueError, match="unsupported"):
            typed_field_extraction(
                "item",
                field_type,  # type: ignore[arg-type]
                "x",
                context_keys=["n"],
            )

    @pytest.mark.parametrize("name", sorted(EXTRACTION_ENVELOPE_KEYS))
    def test_envelope_named_field_rejected(self, name):
        with pytest.raises(ValueError, match="envelope"):
            typed_field_extraction(name, "str", "x", context_keys=["n"])

    def test_envelope_keys_are_the_reply_envelope_names(self):
        assert EXTRACTION_ENVELOPE_KEYS == {
            "reasoning",
            "confidence",
            "value",
            "field_name",
            "extracted_data",
        }

    @pytest.mark.parametrize("keys", [(), []])
    def test_empty_context_keys_rejected(self, keys):
        with pytest.raises(ValueError, match="at least one context key"):
            typed_field_extraction("city", "str", "x", context_keys=keys)

    @pytest.mark.parametrize("keys", ["task", b"task", None, ["task", 1]])
    def test_non_collection_context_keys_rejected(self, keys):
        # A bare str was taken as its characters ('t', 'a', 's', 'k'), and
        # None raised TypeError instead of the documented ValueError.
        with pytest.raises(ValueError, match="context_keys"):
            typed_field_extraction("city", "str", "x", context_keys=keys)

    @pytest.mark.parametrize("key", ["_secret", "system_x", "internal_y", "__z"])
    def test_internal_context_key_rejected(self, key):
        with pytest.raises(ValueError, match="not allowed"):
            typed_field_extraction("city", "str", "x", context_keys=["note", key])


def _trip_fsm(field: dict) -> dict:
    """intro (silent, unconditional) -> collect (typed field) -> done."""
    return {
        "name": "trip",
        "description": "Typed field narrowing",
        "initial_state": "intro",
        "states": {
            "intro": {
                "id": "intro",
                "description": "Intro",
                "purpose": "Move on",
                "response_instructions": "",
                "transitions": [
                    {"target_state": "collect", "description": "Go", "priority": 100}
                ],
            },
            "collect": {
                "id": "collect",
                "description": "Collect the city",
                "purpose": "Get the city",
                "extraction_instructions": "",
                "field_extractions": [field],
                "response_instructions": "",
                "transitions": [
                    {
                        "target_state": "done",
                        "description": "City known",
                        "priority": 100,
                        "conditions": [
                            {
                                "description": "city set",
                                "requires_context_keys": ["city"],
                                "logic": {"!!": [{"var": "city"}]},
                            }
                        ],
                    }
                ],
            },
            "done": {
                "id": "done",
                "description": "Done",
                "purpose": "Report",
                "response_instructions": "Report the city",
            },
        },
    }


class TestNarrowingThroughApi:
    def _run(self, context_keys: list[str]) -> tuple[PromptGroundedLLM, dict]:
        llm = PromptGroundedLLM(facts={"city": ("Lyon", "flies to Lyon")})
        field = typed_field_extraction(
            "city", "str", "Name the city.", context_keys=context_keys
        )
        api = API.from_definition(_trip_fsm(field), llm_interface=llm)
        conv_id, _ = api.start_conversation({"trip_note": _NOTE, "unlisted": _HIDDEN})
        api.advance(conv_id)  # intro -> collect
        api.advance(conv_id)  # collect: per-field call
        return llm, api.get_data(conv_id)

    def test_listed_key_reaches_the_prompt_unlisted_does_not(self):
        llm, data = self._run(["trip_note"])

        (request,) = llm.calls("extract_field")
        assert set(request.context) == {"trip_note"}
        assert _NOTE in request.system_prompt
        assert _HIDDEN not in request.system_prompt
        assert data["city"] == "Lyon"

    def test_no_bulk_call_for_a_typed_only_state(self):
        llm, _ = self._run(["trip_note"])
        assert llm.calls("extract_bulk_data") == []

    def test_unlisted_evidence_is_not_grounded(self):
        # Only `unlisted` is shown; the evidence sits in `trip_note`.
        llm, data = self._run(["unlisted"])

        assert all(_NOTE not in r.system_prompt for r in llm.calls("extract_field"))
        assert data.get("city") is None


class TestDictValueForStrFieldIsRejected:
    """A dict-wrapped answer for a str field is not stored and does not gate."""

    @staticmethod
    def _completion(field_payload: str):
        def fake(**kwargs):
            fmt = kwargs.get("response_format")
            if fmt is not None:
                return _fake_response(field_payload)
            return _fake_response(json.dumps({"message": "ok", "reasoning": ""}))

        return fake

    def _converse(self, field_payload: str):
        with (
            patch(
                "fsm_llm.llm.providers.litellm.completion",
                side_effect=self._completion(field_payload),
            ),
            patch(
                "fsm_llm.llm.providers.litellm.get_supported_openai_params",
                return_value=["response_format"],
            ),
        ):
            api = API.from_definition(_color_fsm(), model="gpt-4o", api_key="test")
            cid, _ = api.start_conversation()
            api.converse("My favorite color is blue.", cid)
            return api.get_data(cid), api.get_current_state(cid)

    def test_dict_value_is_not_stored_and_transition_does_not_fire(self):
        data, state = self._converse(
            '{"field_name": "favorite_color", "value": {"blue": "blue"}, '
            '"confidence": 0.9}'
        )
        assert "favorite_color" not in data
        assert state == "profile"

    def test_list_value_is_not_stored_for_str_field(self):
        data, state = self._converse(
            '{"field_name": "favorite_color", "value": ["blue"], "confidence": 0.9}'
        )
        assert "favorite_color" not in data
        assert state == "profile"

    def test_plain_string_value_is_stored_and_transition_fires(self):
        """Vacuity guard: the rejection is a filter, not a dead extractor."""
        data, state = self._converse(
            '{"field_name": "favorite_color", "value": "blue", "confidence": 0.9}'
        )
        assert data.get("favorite_color") == "blue"
        assert state == "done"


class TestStrictBoolCoercion:
    """F-LC-01: a bool field accepts a closed yes/no vocabulary, nothing else.

    Drives the real validation seam and the real llm.py unstructured rung, so the
    two readers are pinned to the same answers.
    """

    _TRUE: ClassVar[list[str]] = [
        "Yes.",
        " true",
        "true\n",
        "Y",
        "'true'",
        '"Yes"',
        "1",
        "TRUE!",
    ]
    _FALSE: ClassVar[list[str]] = ["No!", "'false'", "n", "False", " no ", "0", "NO."]
    _UNKNOWN: ClassVar[list[str]] = [
        "maybe",
        "yeah",
        "sure",
        "affirmative",
        "",
        "  ",
        "nope",
        "2",
        ".",
    ]

    @staticmethod
    def _validate(value):
        from fsm_llm.core.definitions import (
            FieldExtractionConfig,
            FieldExtractionResponse,
        )
        from fsm_llm.engine.turn.message_pipeline import MessagePipeline

        return MessagePipeline._validate_field_extraction(
            FieldExtractionResponse(
                field_name="agree", value=value, confidence=1.0, is_valid=True
            ),
            FieldExtractionConfig(
                field_name="agree",
                field_type="bool",
                extraction_instructions="did they agree",
                confidence_threshold=0.0,
            ),
        )

    @staticmethod
    def _rung(text):
        from unittest.mock import MagicMock

        from fsm_llm.core.definitions import FieldExtractionRequest
        from fsm_llm.llm import LiteLLMInterface

        response = MagicMock()
        choice = MagicMock()
        choice.message.content = text
        response.choices = [choice]
        request = FieldExtractionRequest(
            system_prompt="extract",
            user_message="x",
            field_name="agree",
            field_type="bool",
        )
        return LiteLLMInterface(model="test-model")._parse_field_extraction_response(
            response, request
        )

    @pytest.mark.parametrize("text", _TRUE)
    def test_true_words_coerce_to_true(self, text):
        out = self._validate(text)
        assert out.is_valid is True
        assert out.value is True

    @pytest.mark.parametrize("text", _FALSE)
    def test_false_words_coerce_to_false(self, text):
        out = self._validate(text)
        assert out.is_valid is True
        assert out.value is False

    @pytest.mark.parametrize("text", _UNKNOWN)
    def test_unknown_words_are_invalid_not_false(self, text):
        out = self._validate(text)
        assert out.is_valid is False
        assert "bool" in (out.validation_error or "")

    def test_non_string_branches_unchanged(self):
        assert self._validate(True).value is True
        assert self._validate(False).value is False
        assert self._validate(1).value is True
        assert self._validate(0).value is False
        assert self._validate({"a": 1}).is_valid is False
        assert self._validate([True]).is_valid is False

    @pytest.mark.parametrize("text", _TRUE)
    def test_llm_rung_agrees_on_true(self, text):
        if not text.strip():
            pytest.skip("empty")
        out = self._rung(text)
        assert out.is_valid is True and out.value is True

    @pytest.mark.parametrize("text", _FALSE)
    def test_llm_rung_agrees_on_false(self, text):
        out = self._rung(text)
        assert out.is_valid is True and out.value is False

    @pytest.mark.parametrize("text", ["maybe", "yeah", "sure", "affirmative", "nope"])
    def test_llm_rung_unknown_word_is_invalid(self, text):
        out = self._rung(text)
        assert out.is_valid is False
        assert out.value is None
