from __future__ import annotations

"""Unit tests for fsm_llm.ollama — Ollama-specific helpers."""

import copy

import pytest

from fsm_llm.ollama import (
    EXTRACTION_JSON_SCHEMA,
    FIELD_EXTRACTION_JSON_SCHEMA,
    apply_ollama_params,
    build_ollama_response_format,
    is_ollama_model,
)

# ------------------------------------------------------------------
# is_ollama_model
# ------------------------------------------------------------------


class TestIsOllamaModel:
    @pytest.mark.parametrize(
        "model",
        [
            "ollama_chat/qwen3.5:4b",
            "ollama/llama3:8b",
            "OLLAMA_CHAT/mixtral:latest",
            "ollama_chat/qwen3.5-4b-32k",
        ],
    )
    def test_ollama_models_detected(self, model):
        assert is_ollama_model(model) is True

    @pytest.mark.parametrize(
        "model",
        [
            "gpt-4o",
            "gpt-4o-mini",
            "claude-3-opus",
            "anthropic/claude-3-sonnet",
            "together_ai/llama-3",
        ],
    )
    def test_non_ollama_models_rejected(self, model):
        assert is_ollama_model(model) is False


# ------------------------------------------------------------------
# apply_ollama_params
# ------------------------------------------------------------------


class TestApplyOllamaParams:
    def test_structured_sets_all_params(self):
        params = {"temperature": 0.5}
        apply_ollama_params(params, "ollama_chat/qwen3.5:4b", structured=True)

        assert params["reasoning_effort"] == "none"
        assert params["temperature"] == 0

    def test_non_structured_preserves_temperature(self):
        params = {"temperature": 0.7}
        apply_ollama_params(params, "ollama_chat/qwen3.5:4b", structured=False)

        assert params["reasoning_effort"] == "none"
        assert params["temperature"] == 0.7  # preserved

    def test_noop_for_non_ollama(self):
        params = {"temperature": 0.5}
        original = copy.deepcopy(params)
        apply_ollama_params(params, "gpt-4o", structured=True)

        assert params == original

    def test_preserves_existing_extra_body_keys(self):
        params = {"extra_body": {"options": {"num_predict": 200}}}
        apply_ollama_params(params, "ollama_chat/qwen3.5:4b", structured=True)

        assert params["extra_body"]["options"]["num_predict"] == 200
        assert params["reasoning_effort"] == "none"


# ------------------------------------------------------------------
# build_ollama_response_format
# ------------------------------------------------------------------


class TestBuildOllamaResponseFormat:
    def test_extraction_format(self):
        fmt = build_ollama_response_format("data_extraction")
        assert fmt is not None
        assert fmt["type"] == "json_schema"
        assert fmt["json_schema"]["name"] == "data_extraction"
        assert fmt["json_schema"]["schema"] is EXTRACTION_JSON_SCHEMA

    def test_transition_decision_has_no_schema(self):
        """The transition schema was removed (step 9.2): a transition_decision
        call type gets the documented unknown-call-type fallback, ``None``."""
        assert build_ollama_response_format("transition_decision") is None

    def test_response_generation_returns_none(self):
        assert build_ollama_response_format("response_generation") is None

    def test_unknown_call_type_returns_none(self):
        assert build_ollama_response_format("unknown") is None


class TestFieldExtractionValueTypes:
    """LV-01: the `value` TYPE (not a description) steers the model."""

    @staticmethod
    def _value_type(field_type):
        fmt = build_ollama_response_format("field_extraction", field_type)
        assert fmt is not None
        return fmt["json_schema"]["schema"]["properties"]["value"]["type"]

    @pytest.mark.parametrize("field_type", [None, "any", "str", "int", "float"])
    def test_scalar_types_exclude_object(self, field_type):
        assert "object" not in self._value_type(field_type)

    @pytest.mark.parametrize("field_type", ["bool", "list"])
    def test_bool_and_list_exclude_object(self, field_type):
        assert "object" not in self._value_type(field_type)

    def test_dict_includes_object(self):
        assert "object" in self._value_type("dict")

    def test_every_type_allows_null(self):
        for ft in (None, "any", "str", "int", "float", "bool", "list", "dict"):
            assert "null" in self._value_type(ft)

    def test_unknown_field_type_falls_back_to_default_union(self):
        assert self._value_type("weird") == self._value_type(None)

    def test_default_has_no_empty_value_schema(self):
        assert FIELD_EXTRACTION_JSON_SCHEMA["properties"]["value"] != {}

    def test_field_type_ignored_for_other_call_types(self):
        fmt = build_ollama_response_format("data_extraction", "str")
        assert fmt is not None
        assert fmt["json_schema"]["schema"] is EXTRACTION_JSON_SCHEMA

    def test_typed_formats_do_not_alias_the_shared_constant(self):
        fmt = build_ollama_response_format("field_extraction", "dict")
        assert fmt is not None
        fmt["json_schema"]["schema"]["properties"]["value"]["type"].append("x")
        assert "x" not in self._value_type("dict")


# ------------------------------------------------------------------
# JSON Schema constants
# ------------------------------------------------------------------


class TestJsonSchemaConstants:
    def test_extraction_schema_has_required_fields(self):
        assert "extracted_data" in EXTRACTION_JSON_SCHEMA["properties"]
        assert "confidence" in EXTRACTION_JSON_SCHEMA["properties"]
        assert "extracted_data" in EXTRACTION_JSON_SCHEMA["required"]
        assert "confidence" in EXTRACTION_JSON_SCHEMA["required"]
