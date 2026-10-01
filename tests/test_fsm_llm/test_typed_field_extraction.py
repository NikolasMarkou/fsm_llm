"""Core ``typed_field_extraction``: the typed per-field builder moved from agents.

The builder returns a raw ``field_extractions`` entry with a narrowed prompt
context; these tests pin its contract and that the narrowing holds through a
real ``API`` turn.
"""

from __future__ import annotations

from typing import get_args

import pytest

from fsm_llm import API, FieldExtractionConfig, typed_field_extraction
from fsm_llm.constants import EXTRACTION_ENVELOPE_KEYS, FIELD_PROMPT_CONTEXT_LABEL
from fsm_llm.definitions import TypedFieldType
from tests.conftest import PromptGroundedLLM

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
