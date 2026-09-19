"""Seam regression tests for audit-fix loop 2 (plan-2026-09-19T175721-21cd7f8e).

Every test drives a public entry point (``API.converse`` or ``LiteLLMInterface``
with a patched ``fsm_llm.llm.completion``). Sections are appended one per plan
step; harness helpers are reused from the iteration-1 seam file.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from fsm_llm.definitions import FieldExtractionRequest
from fsm_llm.llm import LiteLLMInterface
from fsm_llm.ollama import is_ollama_model
from tests.test_fsm_llm.test_audit_iter1_seam import (
    _ambiguous_fsm,
    _classified_fsm,
    _ConnectionHarness,
    _fake_response,
)

# ══════════════════════════════════════════════════════════════
# Step 2 / LS-18: Ollama detection is an exact provider prefix
# ══════════════════════════════════════════════════════════════


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


# ══════════════════════════════════════════════════════════════
# Step 2 / RA-05: reserved names in LiteLLMInterface kwargs
# ══════════════════════════════════════════════════════════════


class TestClassifierIgnoresReservedInterfaceKwargs:
    @pytest.mark.parametrize("reserved", ["config", "schema"])
    def test_classification_extraction_survives_reserved_kwarg(self, reserved):
        h = _ConnectionHarness(
            _classified_fsm(),
            "buy",
            model="gpt-4o",
            api_key="sk-x",
            timeout=9,
            **{reserved: {"user_thing": 1}},
        )
        _, state = h.run()
        # RED on HEAD: TypeError swallowed, classification silently skipped
        assert state == "shop"
        assert len(h.classifier_calls) == 1
        call = h.classifier_calls[0]
        assert call["api_key"] == "sk-x"
        assert call["timeout"] == 9
        assert reserved not in call

    @pytest.mark.parametrize("reserved", ["config", "schema"])
    def test_ambiguous_transition_survives_reserved_kwarg(self, reserved):
        h = _ConnectionHarness(
            _ambiguous_fsm(),
            "billing",
            model="gpt-4o",
            api_key="sk-x",
            timeout=9,
            **{reserved: {"user_thing": 1}},
        )
        _, state = h.run()
        assert state == "billing"
        assert len(h.classifier_calls) == 1
        call = h.classifier_calls[0]
        assert call["api_key"] == "sk-x"
        assert call["timeout"] == 9
        assert reserved not in call

    def test_helper_drops_reserved_names_only(self):
        from unittest.mock import MagicMock

        from fsm_llm.pipeline import MessagePipeline

        llm = MagicMock()
        llm.model = "gpt-4o"
        llm.timeout = 5
        llm.kwargs = {
            "schema": 1,
            "model": 2,
            "config": 3,
            "api_key": "k",
            "api_base": "b",
        }
        pipe = MessagePipeline.__new__(MessagePipeline)
        pipe.llm_interface = llm
        assert pipe._classifier_connection_kwargs() == {
            "api_key": "k",
            "api_base": "b",
            "timeout": 5,
        }
