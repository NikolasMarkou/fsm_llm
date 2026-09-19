"""Seam regression tests for audit-fix loop 1 (plan-2026-09-19T175721-21cd7f8e).

Every test here drives a public entry point (``API.converse`` or
``LiteLLMInterface`` with a patched ``fsm_llm.llm.completion``), because the
defects these pin were invisible to helper-level tests. Sections are appended
one per plan step.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from fsm_llm.api import API
from fsm_llm.definitions import FieldExtractionRequest
from fsm_llm.llm import LiteLLMInterface

OLLAMA_MODEL = "ollama_chat/qwen3.5:9b-q8_0"


def _fake_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


# ══════════════════════════════════════════════════════════════
# Step 2 / LV-01: typed field-extraction `value` + dict rejection
# ══════════════════════════════════════════════════════════════


def _field_request(field_type: str = "str") -> FieldExtractionRequest:
    return FieldExtractionRequest(
        system_prompt="extract the field",
        user_message="My favorite color is blue.",
        field_name="favorite_color",
        field_type=field_type,  # type: ignore[arg-type]
    )


class TestFieldExtractionTypedSchemaOnTheWire:
    """`LiteLLMInterface.extract_field` sends the field-typed schema (Ollama)."""

    @staticmethod
    def _run(model: str, field_type: str):
        interface = LiteLLMInterface(model=model, api_key="test")
        with (
            patch("fsm_llm.llm.completion") as mock_comp,
            patch(
                "fsm_llm.llm.get_supported_openai_params",
                return_value=["response_format"],
            ),
        ):
            mock_comp.return_value = _fake_response(
                '{"field_name": "favorite_color", "value": "blue", "confidence": 0.9}'
            )
            interface.extract_field(_field_request(field_type))
            return mock_comp.call_args[1]

    @pytest.mark.parametrize("field_type", ["str", "int", "float", "bool", "list"])
    def test_ollama_value_type_forbids_object(self, field_type):
        params = self._run(OLLAMA_MODEL, field_type)
        schema = params["response_format"]["json_schema"]["schema"]
        value_type = schema["properties"]["value"]["type"]
        assert "object" not in value_type

    def test_ollama_dict_field_still_allows_object(self):
        params = self._run(OLLAMA_MODEL, "dict")
        schema = params["response_format"]["json_schema"]["schema"]
        assert "object" in schema["properties"]["value"]["type"]

    def test_ollama_prompt_no_longer_shows_empty_value_schema(self):
        params = self._run(OLLAMA_MODEL, "any")
        prompt = params["messages"][-1]["content"]
        assert '"value": {}' not in prompt
        assert '"value": {"type": [' in prompt

    def test_non_ollama_keeps_json_object_format(self):
        params = self._run("gpt-4o", "str")
        assert params["response_format"] == {"type": "json_object"}


def _color_fsm() -> dict:
    return {
        "name": "ColorBot",
        "description": "typed field seam",
        "version": "4.1",
        "initial_state": "profile",
        "persona": "Concise.",
        "states": {
            "profile": {
                "id": "profile",
                "description": "collect color",
                "purpose": "Learn favorite color",
                "field_extractions": [
                    {
                        "field_name": "favorite_color",
                        "field_type": "str",
                        "extraction_instructions": "Favorite color, one word.",
                        "required": True,
                    }
                ],
                "response_instructions": "Reply in one short sentence.",
                "transitions": [
                    {
                        "target_state": "done",
                        "description": "got color",
                        "conditions": [
                            {
                                "description": "has color",
                                "requires_context_keys": ["favorite_color"],
                            }
                        ],
                    }
                ],
            },
            "done": {
                "id": "done",
                "description": "end",
                "purpose": "end",
                "response_instructions": "Say goodbye.",
                "transitions": [],
            },
        },
    }


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
                "fsm_llm.llm.completion", side_effect=self._completion(field_payload)
            ),
            patch(
                "fsm_llm.llm.get_supported_openai_params",
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


class TestZeroConfidenceValueIsNotAccepted:
    """A self-reported confidence of 0.0 is "could not ground it", not a value."""

    @staticmethod
    def _converse(confidence: str):
        payload = (
            '{"field_name": "favorite_color", "value": "blue", '
            f'"confidence": {confidence}}}'
        )

        def fake(**kwargs):
            if kwargs.get("response_format") is not None:
                return _fake_response(payload)
            return _fake_response(json.dumps({"message": "ok", "reasoning": ""}))

        with (
            patch("fsm_llm.llm.completion", side_effect=fake),
            patch(
                "fsm_llm.llm.get_supported_openai_params",
                return_value=["response_format"],
            ),
        ):
            api = API.from_definition(_color_fsm(), model="gpt-4o", api_key="test")
            cid, _ = api.start_conversation()
            api.converse("My favorite color is blue.", cid)
            return api.get_data(cid), api.get_current_state(cid)

    def test_zero_confidence_value_is_left_unset(self):
        data, state = self._converse("0.0")
        assert "favorite_color" not in data
        assert state == "profile"

    def test_low_but_nonzero_confidence_is_still_stored(self):
        data, state = self._converse("0.1")
        assert data.get("favorite_color") == "blue"
        assert state == "done"
