from __future__ import annotations

"""Unit tests for llm.py — LiteLLMInterface with mocked litellm."""

from unittest.mock import MagicMock, patch

import pytest

from fsm_llm.runtime._litellm import LiteLLMInterface, LLMInterface
from fsm_llm.types import (
    LLMResponseError,
    ResponseGenerationRequest,
)


class TestLiteLLMInterfaceInit:
    """Test LiteLLMInterface initialization and validation."""

    def test_requires_non_empty_model(self):
        with pytest.raises(ValueError, match="non-empty"):
            LiteLLMInterface(model="")

    def test_requires_non_empty_model_whitespace(self):
        with pytest.raises(ValueError, match="non-empty"):
            LiteLLMInterface(model="   ")

    def test_rejects_invalid_temperature(self):
        with pytest.raises(ValueError, match="temperature"):
            LiteLLMInterface(model="test", temperature=3.0)

    def test_rejects_negative_temperature(self):
        with pytest.raises(ValueError, match="temperature"):
            LiteLLMInterface(model="test", temperature=-0.1)

    def test_rejects_zero_max_tokens(self):
        with pytest.raises(ValueError, match="max_tokens"):
            LiteLLMInterface(model="test", max_tokens=0)

    def test_valid_init(self):
        llm = LiteLLMInterface(model="test-model", temperature=0.7, max_tokens=500)
        assert llm.model == "test-model"
        assert llm.temperature == 0.7
        assert llm.max_tokens == 500

    def test_api_key_stored_in_kwargs(self):
        llm = LiteLLMInterface(model="test-model", api_key="test-key-123")
        assert llm.kwargs.get("api_key") == "test-key-123"

    def test_implements_llm_interface(self):
        assert issubclass(LiteLLMInterface, LLMInterface)


def _mock_llm_response(content: str) -> MagicMock:
    """Create a mock litellm response with given content."""
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_choice.message.thinking = None
    mock_response.choices = [mock_choice]
    return mock_response


class TestGenerateResponse:
    """Test response generation via LLM."""

    @patch("fsm_llm.runtime._litellm.completion")
    @patch("fsm_llm.runtime._litellm.get_supported_openai_params", return_value=[])
    def test_generate_response_returns_message(self, mock_params, mock_completion):
        mock_completion.return_value = _mock_llm_response("Hello! How can I help you?")

        llm = LiteLLMInterface(model="test-model")
        request = ResponseGenerationRequest(
            system_prompt="You are a helpful assistant",
            user_message="Hi",
            extracted_data={},
            context={},
            transition_occurred=False,
            previous_state=None,
        )
        response = llm.generate_response(request)
        assert response.message == "Hello! How can I help you?"

    @patch("fsm_llm.runtime._litellm.completion", side_effect=Exception("timeout"))
    @patch("fsm_llm.runtime._litellm.get_supported_openai_params", return_value=[])
    def test_generate_response_raises_on_error(self, mock_params, mock_completion):
        llm = LiteLLMInterface(model="test-model")
        request = ResponseGenerationRequest(
            system_prompt="test",
            user_message="test",
            extracted_data={},
            context={},
            transition_occurred=False,
            previous_state=None,
        )
        with pytest.raises(LLMResponseError, match="Response generation failed"):
            llm.generate_response(request)


class TestMakeLLMCall:
    """Test the _make_llm_call method."""

    @patch("fsm_llm.runtime._litellm.completion")
    @patch("fsm_llm.runtime._litellm.get_supported_openai_params", return_value=[])
    def test_invalid_response_structure_raises(self, mock_params, mock_completion):
        mock_completion.return_value = MagicMock(choices=[])

        llm = LiteLLMInterface(model="test-model")
        with pytest.raises(LLMResponseError, match="Invalid response"):
            llm._make_llm_call([{"role": "user", "content": "hi"}], "test")

    @patch("fsm_llm.runtime._litellm.completion")
    @patch("fsm_llm.runtime._litellm.get_supported_openai_params", return_value=[])
    def test_none_response_raises(self, mock_params, mock_completion):
        mock_completion.return_value = None

        llm = LiteLLMInterface(model="test-model")
        with pytest.raises(LLMResponseError):
            llm._make_llm_call([{"role": "user", "content": "hi"}], "test")

    @patch("fsm_llm.runtime._litellm.completion")
    @patch(
        "fsm_llm.runtime._litellm.get_supported_openai_params",
        return_value=["response_format"],
    )
    def test_json_object_for_non_ollama_extraction(self, mock_params, mock_completion):
        """Non-Ollama models use json_object format for extraction."""
        mock_completion.return_value = _mock_llm_response("{}")

        llm = LiteLLMInterface(model="test-model")
        llm._make_llm_call([{"role": "user", "content": "hi"}], "data_extraction")

        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs.get("response_format") == {"type": "json_object"}

    @patch("fsm_llm.runtime._litellm.completion")
    @patch(
        "fsm_llm.runtime._litellm.get_supported_openai_params",
        return_value=["response_format"],
    )
    def test_no_json_mode_for_response_generation(self, mock_params, mock_completion):
        mock_completion.return_value = _mock_llm_response("Hello!")

        llm = LiteLLMInterface(model="test-model")
        llm._make_llm_call([{"role": "user", "content": "hi"}], "response_generation")

        call_kwargs = mock_completion.call_args.kwargs
        assert "response_format" not in call_kwargs


class TestOllamaLLMCallParams:
    """Test Ollama-specific parameter handling in _make_llm_call."""

    @patch("fsm_llm.runtime._litellm.completion")
    @patch(
        "fsm_llm.runtime._litellm.get_supported_openai_params",
        return_value=["response_format"],
    )
    def test_ollama_uses_json_schema_for_extraction(self, mock_params, mock_completion):
        """Ollama models use json_schema format (not json_object) for extraction."""
        mock_completion.return_value = _mock_llm_response(
            '{"extracted_data": {}, "confidence": 0.9}'
        )

        llm = LiteLLMInterface(model="ollama_chat/qwen3.5:4b")
        llm._make_llm_call([{"role": "user", "content": "hi"}], "data_extraction")

        call_kwargs = mock_completion.call_args.kwargs
        rf = call_kwargs.get("response_format")
        assert rf is not None
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["name"] == "data_extraction"

    @patch("fsm_llm.runtime._litellm.completion")
    @patch(
        "fsm_llm.runtime._litellm.get_supported_openai_params",
        return_value=["response_format"],
    )
    def test_ollama_sets_reasoning_effort_none(self, mock_params, mock_completion):
        """Ollama models set reasoning_effort=none to disable thinking."""
        mock_completion.return_value = _mock_llm_response(
            '{"extracted_data": {}, "confidence": 0.9}'
        )

        llm = LiteLLMInterface(model="ollama_chat/qwen3.5:4b")
        llm._make_llm_call([{"role": "user", "content": "hi"}], "data_extraction")

        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs.get("reasoning_effort") == "none"

    @patch("fsm_llm.runtime._litellm.completion")
    @patch(
        "fsm_llm.runtime._litellm.get_supported_openai_params",
        return_value=["response_format"],
    )
    def test_ollama_forces_temperature_zero_for_structured(
        self, mock_params, mock_completion
    ):
        """Ollama structured calls force temperature=0."""
        mock_completion.return_value = _mock_llm_response(
            '{"extracted_data": {}, "confidence": 0.9}'
        )

        llm = LiteLLMInterface(model="ollama_chat/qwen3.5:4b", temperature=0.7)
        llm._make_llm_call([{"role": "user", "content": "hi"}], "data_extraction")

        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs["temperature"] == 0

    @patch("fsm_llm.runtime._litellm.completion")
    @patch("fsm_llm.runtime._litellm.get_supported_openai_params", return_value=[])
    def test_ollama_preserves_temperature_for_response_generation(
        self, mock_params, mock_completion
    ):
        """Ollama response_generation preserves user temperature."""
        mock_completion.return_value = _mock_llm_response("Hello!")

        llm = LiteLLMInterface(model="ollama_chat/qwen3.5:4b", temperature=0.7)
        llm._make_llm_call([{"role": "user", "content": "hi"}], "response_generation")

        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs["temperature"] == 0.7
