from __future__ import annotations

"""Unit tests for fsm_llm.ollama — Ollama-specific helpers."""

import copy
import json
from unittest.mock import patch

import pytest

from fsm_llm.core.definitions import FieldExtractionRequest
from fsm_llm.llm import LiteLLMInterface
from fsm_llm.ollama import (
    EXTRACTION_JSON_SCHEMA,
    FIELD_EXTRACTION_JSON_SCHEMA,
    apply_ollama_params,
    build_ollama_response_format,
    is_ollama_model,
)
from tests.test_fsm_llm.support import _fake_response

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


class TestIsOllamaModelPrefix:
    @pytest.mark.parametrize(
        "model",
        [
            "ollama_chat/qwen3.5:9b-q8_0",
            "ollama/llama3:8b",
            "OLLAMA_CHAT/mixtral:latest",
            "Ollama/Llama3",
        ],
    )
    def test_ollama_prefixes_detected(self, model):
        assert is_ollama_model(model) is True

    @pytest.mark.parametrize(
        "model",
        [
            "openai/my-ollama-proxy",
            "azure/gpt-4o-not-ollama",
            "together_ai/ollama-tuned",
            "my-ollama-finetune",
            "gpt-4o",
            "",
        ],
    )
    def test_name_containing_ollama_is_not_ollama(self, model):
        assert is_ollama_model(model) is False

    def test_extract_field_on_lookalike_model_sends_generic_json_object(self):
        req = FieldExtractionRequest(
            system_prompt="extract the field",
            user_message="My favorite color is blue.",
            field_name="favorite_color",
            field_type="str",  # type: ignore[arg-type]
        )
        formats: dict[str, dict] = {}
        for model in ("azure/gpt-4o-not-ollama", "ollama_chat/qwen3.5:9b-q8_0"):
            with (
                patch("fsm_llm.llm.completion") as mock_comp,
                patch(
                    "fsm_llm.llm.get_supported_openai_params",
                    return_value=["response_format"],
                ),
            ):
                mock_comp.return_value = _fake_response(
                    '{"field_name": "favorite_color", "value": "blue",'
                    ' "confidence": 0.9}'
                )
                LiteLLMInterface(model=model, api_key="k").extract_field(req)
                formats[model] = mock_comp.call_args.kwargs["response_format"]
        assert formats["azure/gpt-4o-not-ollama"]["type"] == "json_object"
        # vacuity guard: a real Ollama model still gets the typed grammar
        assert formats["ollama_chat/qwen3.5:9b-q8_0"]["type"] == "json_schema"


def _memo_fsm(retries: int) -> dict:
    """One state, three required keys (all per-field), transition reads ``k1``."""
    return {
        "name": "F",
        "description": "d",
        "version": "4.1",
        "initial_state": "a",
        "persona": "p",
        "states": {
            "a": {
                "id": "a",
                "description": "d",
                "purpose": "p",
                "required_context_keys": ["k1", "k2", "k3"],
                "extraction_retries": retries,
                "response_instructions": "Reply.",
                "transitions": [
                    {
                        "target_state": "b",
                        "description": "x",
                        "priority": 1,
                        "conditions": [
                            {
                                "description": "n",
                                "requires_context_keys": ["k1"],
                                "logic": {"has_context": "k1"},
                            }
                        ],
                    }
                ],
            },
            "b": {
                "id": "b",
                "description": "d",
                "purpose": "p",
                "response_instructions": "r",
                "transitions": [],
            },
        },
    }


def _field_calls(model: str, retries: int, succeed: str | None = None):
    """(calls, unique prompts) of per-field provider calls for one turn."""
    prompts: list[str] = []

    def comp(**kw):
        system = kw["messages"][0]["content"]
        if kw.get("response_format") is not None and "Extract the field" in system:
            prompts.append(system)
            if succeed and f"'{succeed}'" in system:
                return _fake_response(
                    json.dumps({"field_name": succeed, "value": "v", "confidence": 0.9})
                )
            return _fake_response(
                json.dumps({"field_name": "x", "value": None, "confidence": 0.0})
            )
        return _fake_response(json.dumps({"message": "ok", "reasoning": ""}))

    with (
        patch("fsm_llm.llm.completion", side_effect=comp),
        patch(
            "fsm_llm.llm.get_supported_openai_params",
            return_value=["response_format"],
        ),
    ):
        from fsm_llm import API

        api = API.from_definition(_memo_fsm(retries), model=model, api_key="x")
        cid, _ = api.start_conversation()
        api.converse("hello", cid)
    return len(prompts), len(set(prompts))


class TestOllamaNullExtractionMemo:
    def test_three_null_keys_cost_one_call_each_on_ollama(self):
        # RED on HEAD: 12 calls (1 + 3 retries) for 3 unique prompts
        assert _field_calls("ollama_chat/qwen3.5:9b", 3) == (3, 3)

    def test_the_default_single_retry_no_longer_doubles_a_null_field(self):
        assert _field_calls("ollama_chat/qwen3.5:9b", 1) == (3, 3)

    def test_one_success_still_retries_the_others_once_with_the_new_context(self):
        # first pass 3 calls; retry 1 re-asks k2 and k3 with a different prompt
        # (k1 is no longer in the dynamic context); retry 2 and 3 hit the memo
        assert _field_calls("ollama_chat/qwen3.5:9b", 3, succeed="k1") == (5, 5)

    def test_a_non_ollama_provider_keeps_every_retry_as_a_resample(self):
        assert _field_calls("gpt-4o", 3) == (12, 3)
        assert _field_calls("gpt-4o", 3, succeed="k1") == (9, 5)

    def test_a_lookalike_model_name_is_not_treated_as_ollama(self):
        assert _field_calls("openai/my-ollama-proxy", 3) == (12, 3)

    def test_an_interface_whose_model_is_not_a_str_disables_the_memo(self):
        from unittest.mock import Mock

        from fsm_llm import API
        from fsm_llm.core.definitions import (
            FieldExtractionResponse,
            ResponseGenerationResponse,
        )
        from fsm_llm.llm import LLMInterface

        class _Scripted(LLMInterface):
            def __init__(self):
                self.model = Mock()  # truthy, not a str
                self.field_calls = 0

            def generate_response(self, request):
                return ResponseGenerationResponse(message="ok")

            def extract_field(self, request):
                self.field_calls += 1
                return FieldExtractionResponse(
                    field_name=request.field_name, value=None, confidence=0.0
                )

            def extract_bulk_data(self, request):
                from fsm_llm.core.definitions import DataExtractionResponse

                return DataExtractionResponse(extracted_data={}, confidence=0.0)

        llm = _Scripted()
        api = API.from_definition(_memo_fsm(3), llm_interface=llm)
        cid, _ = api.start_conversation()
        api.converse("hello", cid)
        assert llm.field_calls == 12

    def test_a_successful_result_is_never_served_from_the_memo(self):
        # k1 succeeds with the same prompt text on every call; each extraction
        # must reach the provider (values are not memoised, only nulls).
        seen: list[str] = []

        def comp(**kw):
            system = kw["messages"][0]["content"]
            if kw.get("response_format") is not None and "Extract the field" in system:
                seen.append(system)
                return _fake_response(
                    json.dumps(
                        {
                            "field_name": "k1",
                            "value": f"v{len(seen)}",
                            "confidence": 0.9,
                        }
                    )
                )
            return _fake_response(json.dumps({"message": "ok", "reasoning": ""}))

        fsm = _memo_fsm(0)
        fsm["states"]["a"]["required_context_keys"] = ["k1"]
        with (
            patch("fsm_llm.llm.completion", side_effect=comp),
            patch(
                "fsm_llm.llm.get_supported_openai_params",
                return_value=["response_format"],
            ),
        ):
            from fsm_llm import API

            api = API.from_definition(fsm, model="ollama_chat/q", api_key="x")
            cid, _ = api.start_conversation()
            api.converse("hello", cid)
        assert api.get_data(cid)["k1"] == "v1"
        assert len(seen) == 1
